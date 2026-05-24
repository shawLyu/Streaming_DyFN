import os
import sys
import warnings
from pathlib import Path
from typing import *

if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import click
import numpy as np
import torch
import utils3d
from tqdm import tqdm, trange

from video_depth_anything.video_depth import VideoDepthAnything
from third_party_models import pdcnet

# Reuse the current MoGe reconstruction renderer/IO behavior so the VDA demo
# produces the same output layout and accepts the same camera controls.
from video_reconstruction import (  # noqa: E402
    apply_transform,
    get_correspondence_pdcnet,
    read_frames,
    render_sequence,
    solve_pose_ransac,
    write_ply,
    write_video,
)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


def normalized_intrinsics_from_fov(fov_x: float, height: int, width: int) -> np.ndarray:
    """Return intrinsics in normalized image coordinates, matching MoGe outputs."""
    aspect = width / height
    fx = 1.0 / (2.0 * np.tan(np.deg2rad(fov_x) / 2.0))
    fy = fx * aspect
    return np.array(
        [
            [fx, 0.0, 0.5],
            [0.0, fy, 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def depth_to_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project a depth map using normalized image-coordinate intrinsics."""
    h, w = depth.shape
    u, v = np.meshgrid(
        (np.arange(w, dtype=np.float32) + 0.5) / w,
        (np.arange(h, dtype=np.float32) + 0.5) / h,
    )
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    points = np.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], axis=-1)
    return points.astype(np.float32)


def depth_edge(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    edge_fn = getattr(utils3d.numpy, 'depth_edge', None)
    if edge_fn is not None:
        return edge_fn(depth, rtol=0.05, mask=mask)
    return utils3d.numpy.depth_map_edge(depth, rtol=0.05, mask=mask)


def finite_depth_mask(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth > 1e-6)


def estimate_depth_affine_from_reference(
    depths: np.ndarray,
    reference_result_dir: Path,
    target_shape: Tuple[int, int],
) -> Tuple[float, float]:
    """Estimate global VDA depth scale/shift from a reference reconstruction cache."""
    if not reference_result_dir.exists():
        raise FileNotFoundError(f"Reference result directory not found: {reference_result_dir}")

    src_values, ref_values = [], []
    for i in range(depths.shape[0]):
        prefix = f'{i:05d}'
        ref_depth_path = reference_result_dir / f'{prefix}_depth_registered.exr'
        ref_mask_path = reference_result_dir / f'{prefix}_mask.png'
        if not ref_depth_path.exists() or not ref_mask_path.exists():
            continue

        src_depth = depths[i].astype(np.float32)
        ref_depth = cv2.imread(str(ref_depth_path), cv2.IMREAD_UNCHANGED)
        ref_mask = cv2.imread(str(ref_mask_path), cv2.IMREAD_GRAYSCALE)
        if ref_depth is None or ref_mask is None:
            continue

        height, width = target_shape
        if ref_depth.shape != (height, width):
            ref_depth = cv2.resize(ref_depth.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            ref_mask = cv2.resize(ref_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

        valid = (
            (ref_mask > 0)
            & np.isfinite(src_depth)
            & np.isfinite(ref_depth)
            & (src_depth > 1e-6)
            & (ref_depth > 1e-6)
        )
        if valid.sum() > 1000:
            src_values.append(src_depth[valid].reshape(-1, 1))
            ref_values.append(ref_depth[valid].reshape(-1, 1))

    if not src_values:
        raise RuntimeError(f"No overlapping valid depths found in {reference_result_dir}")

    src = np.concatenate(src_values, axis=0)
    ref = np.concatenate(ref_values, axis=0)
    design = np.concatenate([src, np.ones_like(src)], axis=1)
    scale, shift = np.linalg.lstsq(design, ref, rcond=None)[0].ravel()
    return float(scale), float(shift)


class VdaSceneReconstructor:
    def __init__(self, video_path: Path, output_dir: Path):
        self.video_name = video_path.stem
        self.output_dir = output_dir / self.video_name
        self.result_dir = self.output_dir / 'result'
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.images: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.points_local: List[np.ndarray] = []
        self.points_world: List[np.ndarray] = []
        self.poses: List[np.ndarray] = []
        self.intrinsics: List[np.ndarray] = []

        self.vda_model = None
        self.pdcnet_model = None

    def has_cached_results(self) -> bool:
        return len(list(self.result_dir.glob('*_cam.npz'))) > 0

    def load_models(self, checkpoint_path: str, encoder: str, metric: bool):
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Video Depth Anything checkpoint not found: {checkpoint}. "
                "Download video_depth_anything_vitl.pth or pass --pretrained to the correct file."
            )

        print(f"Loading Video Depth Anything model from {checkpoint}...")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        self.vda_model = VideoDepthAnything(**MODEL_CONFIGS[encoder], metric=metric)
        self.vda_model.load_state_dict(torch.load(str(checkpoint), map_location='cpu'), strict=True)
        self.vda_model = self.vda_model.to(DEVICE).eval()

        print("Loading PDCNet model...")
        self.pdcnet_model = pdcnet.load_model('pretrained/PDCNet_megadepth.pth.tar')

    def load_cached_results(self, frames: List[np.ndarray]):
        self.images = []
        files = sorted(self.result_dir.glob('*_points.exr'))
        if not files:
            raise FileNotFoundError("No cached VDA reconstruction files found.")

        print(f"Loading {len(files)} cached frames from {self.result_dir}...")
        for p in tqdm(files, desc='Loading Cache'):
            try:
                idx = int(p.name.split('_')[0])
            except ValueError:
                continue
            if idx >= len(frames):
                continue

            prefix = f'{idx:05d}'
            mask_path = self.result_dir / f'{prefix}_mask.png'
            cam_path = self.result_dir / f'{prefix}_cam.npz'
            if not mask_path.exists() or not cam_path.exists():
                continue

            points = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
            cam = np.load(str(cam_path))
            pose, intrinsics = cam['pose'], cam['intrinsics']

            self.images.append(frames[idx])
            self.points_local.append(points)
            self.masks.append(mask)
            self.points_world.append(apply_transform(points, pose).astype(np.float32))
            self.poses.append(pose)
            self.intrinsics.append(intrinsics)
        print("Cache load complete.")

    def run_inference(
        self,
        images: List[np.ndarray],
        fov_x: float = None,
        ref_offsets: List[int] = (1, 5, 21),
        input_size_model: int = 518,
        target_fps: int = -1,
        invert_depth: bool = True,
        depth_scale: float = 1.0,
        depth_shift: float = 0.0,
        align_depth_reference: Optional[Path] = None,
    ):
        if self.vda_model is None or self.pdcnet_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        self.images = images
        num_frames = len(images)
        if num_frames == 0:
            raise ValueError("No frames were loaded from the input video.")

        frames = np.stack([img.astype(np.uint8) for img in images], axis=0)
        print(f"Running Video Depth Anything on {num_frames} frames...")
        with torch.no_grad():
            depths, inferred_fps = self.vda_model.infer_video_depth(
                frames,
                target_fps,
                input_size=input_size_model,
                device=DEVICE.type,
                fp32=True,
            )
        depths = np.asarray(depths, dtype=np.float32)
        if invert_depth:
            depths = 1.0 / np.maximum(depths, 1e-6)
        if align_depth_reference is not None:
            depth_scale, depth_shift = estimate_depth_affine_from_reference(
                depths,
                align_depth_reference,
                target_shape=images[0].shape[:2],
            )
            print(
                f"Aligning VDA depths to {align_depth_reference}: "
                f"scale={depth_scale:.6f}, shift={depth_shift:.6f}"
            )
        if depth_scale != 1.0 or depth_shift != 0.0:
            depths = depth_scale * depths + depth_shift
            depths = np.maximum(depths, 1e-6)
        print(f"VDA inference complete. Depth shape: {depths.shape}, FPS: {inferred_fps}")

        print("Running Registration...")
        for i_curr in trange(num_frames, desc='Processing Frames'):
            img = images[i_curr]
            h, width = img.shape[:2]
            depth = depths[i_curr]
            if depth.shape != (h, width):
                depth = cv2.resize(depth, (width, h), interpolation=cv2.INTER_LINEAR)

            intrinsics = normalized_intrinsics_from_fov(fov_x or 60.0, h, width)
            curr_points = depth_to_points(depth, intrinsics)
            curr_mask = finite_depth_mask(depth)
            curr_mask &= ~depth_edge(depth, curr_mask)

            if i_curr == 0:
                pose = np.eye(4, dtype=np.float32)
                inlier_mask = np.zeros_like(curr_mask, dtype=bool)
            else:
                ref_indices = [i_curr - k for k in ref_offsets if i_curr - k >= 0]
                p_list, q_list, w_list, pixel_indices = [], [], [], []

                for i_ref in ref_indices:
                    ref_img = self.images[i_ref]
                    ref_pts_world = self.points_world[i_ref]
                    ref_mask = self.masks[i_ref]
                    pix_ref, pix_curr = get_correspondence_pdcnet(
                        self.pdcnet_model,
                        ref_img,
                        ref_mask,
                        img,
                        curr_mask,
                    )
                    if len(pix_curr) == 0:
                        continue
                    curr_corr = curr_points[pix_curr[:, 1], pix_curr[:, 0]]
                    ref_corr = ref_pts_world[pix_ref[:, 1], pix_ref[:, 0]]
                    valid = np.isfinite(curr_corr).all(axis=1) & np.isfinite(ref_corr).all(axis=1)
                    if not np.any(valid):
                        continue

                    curr_corr = curr_corr[valid]
                    ref_corr = ref_corr[valid]
                    pix_curr = pix_curr[valid]
                    p_list.append(curr_corr)
                    q_list.append(ref_corr)
                    w_list.append(1.0 / np.maximum(curr_corr[:, 2], 1e-6))
                    pixel_indices.append(pix_curr[:, 1] * width + pix_curr[:, 0])

                if p_list:
                    p = np.concatenate(p_list, axis=0)
                    q = np.concatenate(q_list, axis=0)
                    weights = np.concatenate(w_list, axis=0)
                    if p.shape[0] < 10:
                        pose = self.poses[-1].copy()
                        inliers = np.zeros(p.shape[0], dtype=bool)
                    else:
                        pose, inliers = solve_pose_ransac(p, q, weights)
                    inlier_mask = np.zeros_like(curr_mask, dtype=bool)
                    all_pixels = np.concatenate(pixel_indices, axis=0)
                    all_pixels = all_pixels[inliers]
                    inlier_mask[all_pixels // width, all_pixels % width] = True
                else:
                    pose = self.poses[-1].copy()
                    inlier_mask = np.zeros_like(curr_mask, dtype=bool)

            curr_points_world = apply_transform(curr_points, pose)
            self.masks.append(curr_mask)
            self.points_local.append(curr_points)
            self.points_world.append(curr_points_world.astype(np.float32))
            self.poses.append(pose)
            self.intrinsics.append(intrinsics)
            self._save_frame_result(i_curr, curr_points, curr_mask, pose, intrinsics, depth, inlier_mask)

    def _save_frame_result(self, idx, points, mask, pose, intrinsics, depth, inlier_mask):
        prefix = f'{idx:05d}'
        cv2.imwrite(str(self.result_dir / f'{prefix}_points.exr'), points.astype(np.float32))
        cv2.imwrite(str(self.result_dir / f'{prefix}_mask.png'), (mask * 255).astype(np.uint8))
        cv2.imwrite(str(self.result_dir / f'{prefix}_depth_registered.exr'), depth.astype(np.float32))
        cv2.imwrite(str(self.result_dir / f'{prefix}_inlier_mask.png'), (inlier_mask * 255).astype(np.uint8))
        np.savez(str(self.result_dir / f'{prefix}_cam.npz'), pose=pose, intrinsics=intrinsics)

    def rescale_scene(self, target_scale: float = 1.0) -> float:
        """Normalize world points and camera translations to a target bbox diagonal."""
        valid_points = [
            points[mask]
            for points, mask in zip(self.points_world, self.masks)
            if points.shape[:2] == mask.shape and np.any(mask)
        ]
        if not valid_points:
            raise RuntimeError("Cannot rescale scene because no valid points were generated.")

        all_points = np.concatenate(valid_points, axis=0)
        center = all_points.mean(axis=0)
        bbox_size = float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)))
        if bbox_size <= 1e-8:
            raise RuntimeError("Cannot rescale scene because the point cloud bbox is degenerate.")

        scale_factor = float(target_scale) / bbox_size
        print(
            f"Rescaling VDA scene: bbox={bbox_size:.6f}, "
            f"target={target_scale:.6f}, scale_factor={scale_factor:.6f}"
        )
        for i in range(len(self.points_world)):
            self.points_world[i] = ((self.points_world[i] - center) * scale_factor + center).astype(np.float32)
            self.poses[i] = self.poses[i].copy()
            self.poses[i][:3, 3] = (self.poses[i][:3, 3] - center) * scale_factor + center

        return scale_factor

    def process_point_cloud(self, voxel_size: float = 0.002, combined_stride: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        print("Merging Point Clouds...")
        all_points, all_colors = [], []
        pc_dir = self.output_dir / 'pointclouds'
        pc_dir.mkdir(exist_ok=True)
        combined_stride = max(int(combined_stride), 1)

        count = min(len(self.images), len(self.masks), len(self.points_world))
        for i in tqdm(range(count), desc='Exporting PLY'):
            mask = self.masks[i]
            pts = self.points_world[i][mask]
            colors = self.images[i][mask]
            write_ply(pc_dir / f'frame_{i:05d}.ply', pts, colors)

            if i % combined_stride == 0 and len(pts) > 0:
                all_points.append(pts)
                all_colors.append(colors)

        if not all_points:
            raise RuntimeError("No valid point clouds were generated.")

        combined_pts = np.concatenate(all_points, axis=0)
        combined_col = np.concatenate(all_colors, axis=0)

        if voxel_size > 0:
            print(f"Downsampling (Voxel: {voxel_size})...")
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(combined_pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(combined_col.astype(np.float64) / 255.0)
            pcd = pcd.voxel_down_sample(voxel_size)
            combined_pts = np.asarray(pcd.points).astype(np.float32)
            combined_col = (np.asarray(pcd.colors) * 255).astype(np.uint8)

        write_ply(self.output_dir / 'combined_pointcloud.ply', combined_pts, combined_col)
        return combined_pts, combined_col


@click.command()
@click.option('--video', 'video_path', required=True, type=str, help='Path to video file or image folder.')
@click.option('--output', 'output_path', default='video_output', help='Root output directory.')
@click.option('--pretrained', default='pretrained/video_depth_anything_vitl.pth', help='Video Depth Anything checkpoint path.')
@click.option('--encoder', type=click.Choice(['vits', 'vitb', 'vitl']), default='vitl', help='Video Depth Anything encoder size.')
@click.option('--metric', is_flag=True, help='Use a metric Video Depth Anything checkpoint.')
@click.option('--fov', 'fov_x', type=float, default=None, help='Horizontal FOV override.')
@click.option('--input-size', type=int, default=640, help='Resize input video to this longer side dimension.')
@click.option('--input-size-model', type=int, default=518, help='VDA model inference input size.')
@click.option('--target-fps', type=int, default=-1, help='VDA target FPS. -1 keeps original/loaded frame rate.')
@click.option('--start', type=int, default=None, help='Start frame.')
@click.option('--end', type=int, default=None, help='End frame.')
@click.option('--skip', type=int, default=1, help='Skip frames.')
@click.option('--fps', type=int, default=24, help='Output video FPS.')
@click.option('--voxel-size', type=float, default=0.002, help='Voxel size for global point cloud (0 to disable).')
@click.option('--combined-stride', type=int, default=15, help='Add every Nth frame to combined_pointcloud.ply. Use 1 to include every frame.')
@click.option('--accumulate', 'accumulate_interval', type=int, default=0, help='Accumulate points every k frames for rendering. 0 keeps all previous frame point clouds.')
@click.option('--camera-distance', type=float, default=1.35, help='Multiplier that moves the render camera farther from the look-at point.')
@click.option('--camera-height', type=float, default=0.15, help='Camera height offset as a fraction of the scene bounding-box diagonal.')
@click.option('--render-point-size', type=int, default=1, help='Rendered point splat size in pixels. Use 1 for the finest point cloud.')
@click.option('--pullback-frames', type=int, default=0, help='Append this many pullback frames after trajectory following.')
@click.option('--pullback-distance', type=float, default=3.0, help='Pullback distance as a fraction of the scene bounding-box diagonal.')
@click.option('--ref-offset', 'ref_offsets', multiple=True, type=int, default=(1, 5, 21), help='Reference frame offsets for registration. Can be repeated.')
@click.option('--invert-depth/--no-invert-depth', default=True, help='Invert VDA output before back-projection. Relative VDA checkpoints usually need this.')
@click.option('--depth-scale', type=float, default=1.0, help='Manual affine scale applied to VDA depth before back-projection.')
@click.option('--depth-shift', type=float, default=0.0, help='Manual affine shift applied to VDA depth before back-projection.')
@click.option('--align-depth-reference', type=click.Path(path_type=Path), default=None, help='Reference result/ cache dir used to estimate VDA depth scale and shift.')
@click.option('--use-cache/--no-use-cache', default=None, help='Use existing cached reconstruction results without prompting.')
@click.option('--rescale', is_flag=True, help='Normalize point cloud scale for visualization. Leave unset to render reconstructed scale.')
@click.option('--target-scale', type=float, default=1.0, help='Target bbox diagonal used with --rescale before PLY export and rendering.')
def main(
    video_path,
    output_path,
    pretrained,
    encoder,
    metric,
    fov_x,
    input_size,
    input_size_model,
    target_fps,
    start,
    end,
    skip,
    fps,
    voxel_size,
    combined_stride,
    accumulate_interval,
    camera_distance,
    camera_height,
    render_point_size,
    pullback_frames,
    pullback_distance,
    ref_offsets,
    invert_depth,
    depth_scale,
    depth_shift,
    align_depth_reference,
    use_cache,
    rescale,
    target_scale,
):
    video_path = Path(video_path)
    output_path = Path(output_path)
    scene = VdaSceneReconstructor(video_path, output_path)

    print("Reading input frames...")
    frames = read_frames(video_path, start, end, skip, target_size=input_size)
    (scene.output_dir / 'video').mkdir(parents=True, exist_ok=True)
    write_video(scene.output_dir / 'video' / 'input.mp4', frames, fps=fps)

    if scene.has_cached_results():
        if use_cache is None:
            use_cache = click.confirm(f"Found existing results in {scene.result_dir}. Do you want to use them?")
    elif use_cache:
        raise FileNotFoundError(f"No cached reconstruction results found in {scene.result_dir}")
    else:
        use_cache = False

    if use_cache:
        scene.load_cached_results(frames)
    else:
        scene.load_models(pretrained, encoder=encoder, metric=metric)
        scene.run_inference(
            frames,
            fov_x=fov_x,
            ref_offsets=list(ref_offsets),
            input_size_model=input_size_model,
            target_fps=target_fps,
            invert_depth=invert_depth,
            depth_scale=depth_scale,
            depth_shift=depth_shift,
            align_depth_reference=align_depth_reference,
        )

    if rescale:
        scene.rescale_scene(target_scale=target_scale)

    scene.process_point_cloud(voxel_size=voxel_size, combined_stride=combined_stride)
    render_sequence(
        scene,
        fps=fps,
        accumulate_interval=accumulate_interval,
        rescale=False,
        camera_distance=camera_distance,
        camera_height=camera_height,
        render_point_size=render_point_size,
        pullback_frames=pullback_frames,
        pullback_distance=pullback_distance,
    )

    print(f"Processing complete. Results saved to {scene.output_dir}")


if __name__ == '__main__':
    main()
