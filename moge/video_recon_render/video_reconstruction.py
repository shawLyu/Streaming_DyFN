import os
import sys
import json
import itertools
import subprocess
from pathlib import Path
from typing import *

# Environment Setup
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Windows compatibility

import numpy as np
import cv2
import torch
import click
import open3d as o3d
import utils3d
from tqdm import tqdm, trange
from scipy.interpolate import CubicSpline

# Model Imports
from moge.model.v1 import MoGeModel
from third_party_models import pdcnet

# Global Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an (..., 3) point array."""
    original_shape = points.shape
    points_flat = points.reshape(-1, 3)
    points_h = np.concatenate([points_flat, np.ones((points_flat.shape[0], 1), dtype=points.dtype)], axis=1)
    transformed = (transform @ points_h.T).T[:, :3]
    return transformed.reshape(original_shape)


def solve_pose_rigid(p: np.ndarray, q: np.ndarray, w: np.ndarray = None) -> np.ndarray:
    """Weighted rigid alignment from source points p to target points q."""
    if w is None:
        w = np.ones(p.shape[0], dtype=np.float32)
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, 1e-8)
    w = w / w.sum()
    p64, q64 = p.astype(np.float64), q.astype(np.float64)
    p_center = (p64 * w[:, None]).sum(axis=0)
    q_center = (q64 * w[:, None]).sum(axis=0)
    p0, q0 = p64 - p_center, q64 - q_center
    covariance = (p0 * w[:, None]).T @ q0
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = q_center - rotation @ p_center
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation.astype(np.float32)
    pose[:3, 3] = translation.astype(np.float32)
    return pose


def camera_frustum_mesh(intrinsics: np.ndarray, frustum_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Create a lightweight camera frustum wireframe in camera coordinates."""
    if intrinsics.shape == (4, 4):
        intrinsics = intrinsics[:3, :3]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    rays = np.stack([(corners[:, 0] - cx) / fx, (corners[:, 1] - cy) / fy, np.ones(4)], axis=1)
    rays = rays / (np.linalg.norm(rays, axis=1, keepdims=True) + 1e-8) * frustum_scale
    verts = np.concatenate([np.zeros((1, 3), dtype=np.float32), rays.astype(np.float32)], axis=0)
    edges = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]], dtype=np.int32)
    return verts, edges


def rasterize_point_cloud_simple(
    image_size: Tuple[int, int],
    points: np.ndarray,
    attributes: np.ndarray,
    view: np.ndarray,
    projection: np.ndarray,
    point_size: int = 1,
) -> Dict[str, np.ndarray]:
    """Small CPU point renderer used only for reconstruction preview videos."""
    height, width = image_size
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    mask = np.zeros((height, width), dtype=bool)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    clip = (projection @ view @ points_h.T).T
    valid_w = np.abs(clip[:, 3]) > 1e-8
    ndc = clip[:, :3] / (clip[:, 3:4] + 1e-8)
    valid = valid_w & np.all(np.isfinite(ndc), axis=1) & (ndc[:, 2] > -1) & (ndc[:, 2] < 1)
    xy = np.empty((points.shape[0], 2), dtype=np.int32)
    xy[:, 0] = ((ndc[:, 0] + 1) * 0.5 * width).astype(np.int32)
    xy[:, 1] = ((1 - ndc[:, 1]) * 0.5 * height).astype(np.int32)
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    radius = max((int(point_size) - 1) // 2, 0)
    for (x, y), z, color in zip(xy[valid], ndc[valid, 2], attributes[valid]):
        x0, x1 = max(x - radius, 0), min(x + radius + 1, width)
        y0, y1 = max(y - radius, 0), min(y + radius + 1, height)
        patch = depth[y0:y1, x0:x1]
        closer = z < patch
        if np.any(closer):
            patch[closer] = z
            image_patch = image[y0:y1, x0:x1]
            image_patch[closer] = np.clip(color, 0, 255).astype(np.uint8)
            mask[y0:y1, x0:x1][closer] = True
    return {"image": image, "mask": mask, "depth": depth}


# ==============================================================================
#                               IO UTILITIES
# ==============================================================================

def read_frames(path: Union[str, Path], start: int = None, end: int = None, 
               skip: int = 1, target_size: int = None) -> List[np.ndarray]:
    """Reads frames from a video file or a folder of images."""
    path = Path(path)
    frames = []
    
    if path.is_file():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if target_size:
            scale = target_size / max(original_width, original_height)
            target_w = int(round(original_width * scale))
            target_h = int(round(original_height * scale))
        else:
            target_w, target_h = original_width, original_height
        skip = max(skip or 1, 1)
        if start is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            
        current_idx = start or 0
        pbar = tqdm(total=end - start if end else total_frames - current_idx, desc="Reading Video")
        
        while True:
            ret, frame = cap.read()
            if not ret or (end is not None and current_idx >= end):
                break
            
            if (current_idx - (start or 0)) % skip == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                frames.append(frame)
            
            current_idx += 1
            pbar.update(1)
        cap.release()
        pbar.close()
        
    elif path.is_dir():
        # Image folder handling
        image_paths = sorted(itertools.chain(path.glob('*.jpg'), path.glob('*.png')))[start:end:skip]
        for p in tqdm(image_paths, desc='Reading Frames'):
            img = cv2.imread(str(p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if target_size:
                h, w = img.shape[:2]
                scale = target_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
            frames.append(img)
            
    else:
        raise ValueError(f"Invalid path: {path}")
        
    return frames


def extract_corresponding_pixels(flow, mask_shape):
    h, w = mask_shape[:2]
    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    corresponding_x = (grid_x + flow[..., 0]).astype(int)
    corresponding_y = (grid_y + flow[..., 1]).astype(int)

    valid_mask = (corresponding_x >= 0) & (corresponding_x < w) & (corresponding_y >= 0) & (corresponding_y < h)
    return valid_mask, corresponding_x, corresponding_y

def write_video(path: Union[str, Path], frames: List[np.ndarray], fps: int = 24):
    """Writes a list of RGB numpy arrays to an MP4 video."""
    if not frames:
        return
    path = Path(path)
    h, w = frames[0].shape[:2]
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(tmp_path), fourcc, fps, (w, h))
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()

    try:
        subprocess.run(
            [
                'ffmpeg',
                '-y',
                '-loglevel',
                'error',
                '-i',
                str(tmp_path),
                '-c:v',
                'libx264',
                '-pix_fmt',
                'yuv420p',
                '-movflags',
                '+faststart',
                str(path),
            ],
            check=True,
        )
        tmp_path.unlink(missing_ok=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        os.replace(tmp_path, path)

def write_ply(path: Union[str, Path], points: np.ndarray, colors: np.ndarray = None):
    """Writes point cloud to PLY using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        c = colors.astype(np.float64) / 255.0 if colors.max() > 1.0 else colors.astype(np.float64)
        pcd.colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_point_cloud(str(path), pcd)

# ==============================================================================
#                            GEOMETRY & MATH
# ==============================================================================

def solve_pose_ransac(p: np.ndarray, q: np.ndarray, w: np.ndarray = None, 
                      max_iters: int = 20, hypothetical_size: int = 10, 
                      inlier_thresh: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves for rigid transformation between two sets of 3D points using RANSAC.
    p: Source points (Nx3)
    q: Target points (Nx3)
    """
    n = p.shape[0]
    if w is None: w = np.ones(n)
    
    best_score = 0.0
    best_inliers = np.zeros(n, dtype=bool)
    best_pose = np.eye(4, dtype=np.float32)

    for _ in range(max_iters):
        # Sample random points
        maybe_indices = np.random.choice(n, size=hypothetical_size, replace=False)
        try:
            pose = solve_pose_rigid(p[maybe_indices], q[maybe_indices], w[maybe_indices])
        except np.linalg.LinAlgError:
            continue
            
        # Verify model
        transformed_p = apply_transform(p, pose)
        errors = w * np.linalg.norm(transformed_p - q, axis=1)
        inliers = errors < inlier_thresh
        
        # Soft score
        score = inlier_thresh * n - np.clip(errors, None, inlier_thresh).sum()
        
        if score > best_score:
            best_score = score
            best_inliers = inliers
            # Refine pose with all inliers
            if inliers.sum() > 3:
                best_pose = solve_pose_rigid(p[inliers], q[inliers], w[inliers])
    
    return best_pose, best_inliers

def get_correspondence_pdcnet(model, img_ref, mask_ref, img_query, mask_query):
    """Computes dense correspondence between two images using PDCNet."""
    h, w = img_query.shape[:2]
    
    # Prepare tensors
    t_ref = torch.tensor(img_ref, dtype=torch.uint8, device=DEVICE).permute(2, 0, 1)
    t_query = torch.tensor(img_query, dtype=torch.uint8, device=DEVICE).permute(2, 0, 1)
    
    # Inference
    flow, confidence = pdcnet.predict_flow(model, t_query, t_ref)
    flow = flow.cpu().numpy()
    confidence = confidence.cpu().numpy()
    
    # Map coordinates
    uv_ref = utils3d.numpy.image_uv(height=h, width=w)
    uv_tgt = uv_ref + flow
    
    pix_ref = utils3d.numpy.uv_to_pixel(uv_ref, width=w, height=h).round().astype(int)
    pix_tgt = utils3d.numpy.uv_to_pixel(uv_tgt, width=w, height=h).round().astype(int)
    
    # Filter valid matches
    # 1. Confidence threshold
    # 2. Within image bounds
    # 3. Inside valid mask regions for both images
    valid = (
        (confidence > 0.5) &
        (pix_tgt >= 0).all(axis=-1) &
        (pix_tgt <= [w - 1, h - 1]).all(axis=-1) &
        mask_ref &
        mask_query[pix_tgt.clip(0, [w - 1, h - 1])[:, :, 1], pix_tgt.clip(0, [w - 1, h - 1])[:, :, 0]]
    )
    
    return pix_ref[valid], pix_tgt[valid]

# ==============================================================================
#                            CORE PIPELINE CLASS
# ==============================================================================

class SceneReconstructor:
    def __init__(self, video_path: Path, output_dir: Path):
        self.video_name = video_path.stem
        self.output_dir = output_dir / self.video_name
        self.result_dir = self.output_dir / 'result'
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # Data Storage
        self.images: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.points_local: List[np.ndarray] = []
        self.points_world: List[np.ndarray] = []
        self.poses: List[np.ndarray] = []
        self.intrinsics: List[np.ndarray] = []
        
        # Models (Lazy loaded)
        self.moge_model = None
        self.pdcnet_model = None

    def has_cached_results(self) -> bool:
        """Checks if there are any valid result files in the output directory."""
        # We check for camera pose files as an indicator of completion
        return len(list(self.result_dir.glob('*_cam.npz'))) > 0

    def load_models(self, moge_path: str):
        """Loads heavy ML models. Only call this if inference is needed."""
        print(f"Loading MoGe model from {moge_path}...")
        self.moge_model = MoGeModel.from_pretrained(moge_path).to(DEVICE).eval()
        print("Loading PDCNet model...")
        self.pdcnet_model = pdcnet.load_model('pretrained/PDCNet_megadepth.pth.tar')

    def load_cached_results(self, frames: List[np.ndarray]):
        """Loads pre-calculated results from disk, skipping inference."""
        self.images = []
        files = sorted(self.result_dir.glob('*_points.exr'))
        
        if len(files) == 0:
            raise FileNotFoundError("No cached files found despite check.")

        print(f"Loading {len(files)} cached frames from {self.result_dir}...")
        
        for p in tqdm(files, desc='Loading Cache'):
            # Parse index from filename (e.g., 00005_points.exr -> 5)
            try:
                idx_str = p.name.split('_')[0]
                idx = int(idx_str)
            except ValueError:
                continue

            # Ensure we don't exceed loaded video frames
            if idx >= len(frames):
                continue
            
            # Load Data
            pts = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            mask_path = self.result_dir / f'{idx_str}_mask.png'
            cam_path = self.result_dir / f'{idx_str}_cam.npz'
            
            if not mask_path.exists() or not cam_path.exists():
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
            cam_data = np.load(str(cam_path))
            pose = cam_data['pose']
            intrinsics = cam_data['intrinsics']
            
            # Reconstruct world points
            pts_world = apply_transform(pts, pose)
            
            # Store
            self.images.append(frames[idx])
            self.points_local.append(pts)
            self.masks.append(mask)
            self.points_world.append(pts_world)
            self.poses.append(pose)
            self.intrinsics.append(intrinsics)
            
        print("Cache load complete.")

    def run_inference(self, images: List[np.ndarray], fov_x: float = None, 
                     ref_offsets: List[int] = [1, 5, 21]):
        """Runs MoGe per frame, then aligns frames using PDCNet + RANSAC."""
        if self.moge_model is None or self.pdcnet_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        self.images = images
        num_frames = len(images)
        print("Running Inference & Registration...")
        
        prev_state = None
        
        for i_curr in trange(num_frames, desc='Processing Frames'):
            img = images[i_curr]
            
            # 1. MoGe Depth Estimation
            img_tensor = torch.tensor(img.astype(np.float32) / 255.0, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)
            output = self.moge_model.infer(img_tensor, fov_x=fov_x, prev_state=prev_state, image_based=False)
            
            curr_points = output['points'].cpu().numpy()
            curr_mask = output['mask'].cpu().numpy()
            curr_intrinsics = output['intrinsics'].cpu().numpy()
            prev_state = output['prev_state']
            
            # Clean mask based on depth edges
            curr_mask &= ~utils3d.numpy.depth_edge(curr_points[:, :, 2], rtol=0.05, mask=curr_mask)

            # 2. Pose Estimation (Registration)
            if i_curr == 0:
                # For the first frame, just normalize the scale and set identity pose
                pose = np.mean(1 / curr_points[curr_mask, 2]) * np.eye(4, dtype=np.float32)
                # pose = np.eye(4, dtype=np.float32)
                inlier_mask = np.zeros_like(curr_mask, dtype=bool)
            else:
                ref_indices = [i_curr - k for k in ref_offsets if i_curr - k >= 0]
                p_list, q_list, w_list = [], [], []
                
                for i_ref in ref_indices:
                    ref_img = self.images[i_ref]
                    ref_pts_world = self.points_world[i_ref]
                    ref_mask = self.masks[i_ref]
                    
                    pix_ref, pix_curr = get_correspondence_pdcnet(
                        self.pdcnet_model, ref_img, ref_mask, img, curr_mask
                    )
                    
                    p_list.append(curr_points[pix_curr[:, 1], pix_curr[:, 0]])
                    q_list.append(ref_pts_world[pix_ref[:, 1], pix_ref[:, 0]])
                    w_list.append(1.0 / curr_points[pix_curr[:, 1], pix_curr[:, 0], 2])

                if p_list:
                    pose, _ = solve_pose_ransac(np.concatenate(p_list), np.concatenate(q_list), np.concatenate(w_list))
                else:
                    pose = self.poses[-1]

            # 3. Store & Save
            curr_points_world = apply_transform(curr_points, pose)
            
            self.masks.append(curr_mask)
            self.points_local.append(curr_points)
            self.points_world.append(curr_points_world.astype(np.float32))
            self.poses.append(pose)
            self.intrinsics.append(curr_intrinsics)

            self._save_frame_result(i_curr, curr_points, curr_mask, pose, curr_intrinsics, output['depth'].cpu().numpy())

    def _save_frame_result(self, idx, points, mask, pose, intrinsics, depth):
        prefix = f'{idx:05d}'
        cv2.imwrite(str(self.result_dir / f'{prefix}_points.exr'), points.astype(np.float32))
        cv2.imwrite(str(self.result_dir / f'{prefix}_mask.png'), (mask * 255).astype(np.uint8))
        scale = np.linalg.det(pose[:3, :3])
        cv2.imwrite(str(self.result_dir / f'{prefix}_depth_registered.exr'), (scale * depth).astype(np.float32))
        np.savez(str(self.result_dir / f'{prefix}_cam.npz'), pose=pose, intrinsics=intrinsics)

    def process_point_cloud(self, voxel_size: float = 0.002) -> Tuple[np.ndarray, np.ndarray]:
        # ... [Same as previous logic for combining point clouds] ...
        print("Merging Point Clouds...")
        all_points, all_colors = [], []
        pc_dir = self.output_dir / 'pointclouds'
        pc_dir.mkdir(exist_ok=True)

        for i in tqdm(range(len(self.images)), desc='Exporting PLY'):
            mask = self.masks[i]
            # Safety check if cached masks don't match list length
            if i >= len(self.points_world): break
            
            pts = self.points_world[i][mask]
            colors = self.images[i][mask]
            
            write_ply(pc_dir / f'frame_{i:05d}.ply', pts, colors)
            
            if i % 15 == 0: 
                all_points.append(pts)
                all_colors.append(colors)
        
        combined_pts = np.concatenate(all_points, axis=0)
        combined_col = np.concatenate(all_colors, axis=0)
        
        if voxel_size > 0:
            print(f"Downsampling (Voxel: {voxel_size})...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(combined_pts.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(combined_col.astype(np.float64) / 255.0)
            pcd = pcd.voxel_down_sample(voxel_size)
            combined_pts = np.asarray(pcd.points).astype(np.float32)
            combined_col = (np.asarray(pcd.colors) * 255).astype(np.uint8)

        write_ply(self.output_dir / 'combined_pointcloud.ply', combined_pts, combined_col)
        return combined_pts, combined_col

# ==============================================================================
#                            RENDERING & TRAJECTORY
# ==============================================================================

def compute_smooth_trajectory(poses: List[np.ndarray], 
                            points_world_frames: List[np.ndarray], 
                            masks: List[np.ndarray],
                            accumulate_interval: int = 0,
                            camera_distance: float = 1.35,
                            camera_height: float = 0.0) -> Tuple[Callable, Callable]:
    """
    Computes a smooth camera path (Eye and LookAt) using cubic splines.
    Returns two functions: get_eye(t) and get_look_at(t).
    """
    num_frames = len(poses)
    
    # 1. Extract Raw Positions
    raw_eyes = np.array([p[:3, 3] for p in poses])
    raw_look_ats = []
    
    # Calculate LookAt center per frame based on visible points
    global_center = np.mean(np.concatenate([p[m] for p, m in zip(points_world_frames[::10], masks[::10])]), axis=0)
    
    for i in range(num_frames):
        # Determine valid points for this frame (local or accumulated)
        if accumulate_interval <= 0:
            indices = list(range(i + 1))
        else:
            indices = sorted(list(set([i] + list(range(0, i + 1, accumulate_interval)))))
        
        valid_pts = []
        for idx in indices:
            p = points_world_frames[idx][masks[idx]]
            if len(p) > 0: valid_pts.append(p)
            
        if valid_pts:
            center = np.concatenate(valid_pts, axis=0).mean(axis=0)
        else:
            center = global_center
        raw_look_ats.append(center)
    
    raw_look_ats = np.array(raw_look_ats)

    # 2. Smooth Arrays (Moving Average)
    def smooth_arr(arr, win=5):
        if len(arr) < win: return arr
        res = np.zeros_like(arr)
        for i in range(len(arr)):
            s, e = max(0, i - win//2), min(len(arr), i + win//2 + 1)
            res[i] = arr[s:e].mean(axis=0)
        return res

    smooth_eyes = smooth_arr(raw_eyes, 5)
    smooth_look_ats = smooth_arr(raw_look_ats, 9)
    smooth_eyes = smooth_look_ats + (smooth_eyes - smooth_look_ats) * camera_distance
    smooth_eyes = smooth_eyes + np.array([0.0, -camera_height, 0.0], dtype=np.float32)

    # 3. Create Splines
    t = np.arange(num_frames)
    spline_eye = CubicSpline(t, smooth_eyes)
    spline_look = CubicSpline(t, smooth_look_ats)

    return spline_eye, spline_look

def draw_frustums(image: np.ndarray, 
                  curr_idx: int, 
                  poses: List[np.ndarray], 
                  intrinsics: List[np.ndarray],
                  view_matrix: np.ndarray, 
                  proj_matrix: np.ndarray,
                  frustum_scale: float = 0.1):
    """Draws camera wireframes for historical poses onto the image."""
    h, w = image.shape[:2]
    overlay = image.copy() # Operate on copy if needed, or draw directly
    
    # Loop through historical frames up to current
    for i in range(curr_idx + 1):
        pose = poses[i]
        intr = intrinsics[i]
        extrinsics = np.linalg.inv(pose)
        
        # Color generation (HSV -> RGB)
        hue = int((i / max(len(poses), 1)) * 180)
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0].tolist()
        
        # Get Frustum Geometry
        verts, edges = camera_frustum_mesh(intr, frustum_scale)
        verts_world = apply_transform(verts, pose)
        
        # Project to Screen
        vertices_homogeneous = np.concatenate([verts_world, np.ones((verts_world.shape[0], 1), dtype=np.float32)], axis=1)
        vertices_clip = (proj_matrix @ view_matrix @ vertices_homogeneous.T).T
        # Perspective divide
        verts_ndc = vertices_clip[:, :3] / (vertices_clip[:, 3:4] + 1e-8)
        
        screen_pts = []
        for v in verts_ndc:
            sx = int((v[0] + 1) * 0.5 * w)
            sy = int((1 - v[1]) * 0.5 * h)
            screen_pts.append((sx, sy))
            
        # Draw Lines
        line_width = 3 if i == curr_idx else 1
        for e in edges:
            p1, p2 = screen_pts[e[0]], screen_pts[e[1]]
            # Clip roughly check
            if -1000 < p1[0] < w+1000 and -1000 < p1[1] < h+1000:
                cv2.line(overlay, p1, p2, color_rgb, line_width)
                
    return overlay

def render_sequence(scene: SceneReconstructor, 
                   fps: int = 24, 
                   accumulate_interval: int = 0,
                   rescale: bool = False,
                   target_scale: float = 1.0,
                   camera_distance: float = 1.35,
                   camera_height: float = 0.15,
                   render_point_size: int = 1):
    
    print("Rendering Video...")
    num_frames = len(scene.images)
    render_w, render_h = 640, 480
    
    # 1. Rescale World (Optional)
    # Important: We perform rescaling on copies or modifying lists carefully
    # For brevity, let's assume we modify scene.points_world and scene.poses in place if rescale=True
    bbox_size = 1.0
    if len(scene.points_world) > 0:
        all_p = np.concatenate([p[m] for p, m in zip(scene.points_world, scene.masks)])
        bbox_size = np.linalg.norm(all_p.max(axis=0) - all_p.min(axis=0))
    if rescale and len(scene.points_world) > 0:
        center = all_p.mean(axis=0)
        scale_factor = target_scale / (bbox_size + 1e-6)
        
        print(f"Rescaling scene by factor {scale_factor:.4f}...")
        for i in range(num_frames):
            # Scale points
            scene.points_world[i] = (scene.points_world[i] - center) * scale_factor + center
            # Scale pose translation
            scene.poses[i][:3, 3] = (scene.poses[i][:3, 3] - center) * scale_factor + center
            
    # 2. Compute Trajectory
    get_eye, get_look = compute_smooth_trajectory(
        scene.poses,
        scene.points_world,
        scene.masks,
        accumulate_interval,
        camera_distance,
        camera_height * bbox_size,
    )
    
    # 3. Setup Projection
    up_vec = np.array([0, -1, 0], dtype=np.float32) # Standard screen space up
    proj = utils3d.numpy.perspective_from_fov(
        fov=np.deg2rad(60),
        width=render_w,
        height=render_h,
        near=0.01,
        far=1000,
    )
    
    rendered_frames = []
    
    for i in trange(num_frames, desc='Rendering'):
        # Determine camera for this frame
        view = utils3d.numpy.view_look_at(get_eye(i), get_look(i), up_vec).astype(np.float32)
        
        # Gather points to render (current + accumulated history)
        if accumulate_interval <= 0:
            indices = list(range(i + 1))
        else:
            indices = sorted(list(set([i] + list(range(0, i + 1, accumulate_interval)))))
            
        render_pts = []
        render_cols = []
        for idx in indices:
            mask = scene.masks[idx]
            render_pts.append(scene.points_world[idx][mask])
            render_cols.append(scene.images[idx][mask])
            
        pts_combined = np.concatenate(render_pts).astype(np.float32)
        cols_combined = np.concatenate(render_cols).astype(np.float32)
        
        # Rasterize Points
        out = rasterize_point_cloud_simple(
            (render_h, render_w), points=pts_combined,
            attributes=cols_combined, point_size=render_point_size,
            view=view,
            projection=proj,
        )
        
        # Convert to Image
        img_render = out['image'].copy()
        if img_render.max() <= 1.0: img_render = (img_render * 255).astype(np.uint8)
        
        # Draw Frustums
        frustum_size = max(bbox_size * 0.02, 0.05) if rescale else 0.1
        img_final = draw_frustums(img_render, i, scene.poses, scene.intrinsics, view, proj, frustum_size)
        
        rendered_frames.append(img_final.astype(np.uint8))
        
    # Write Final Video
    vid_path = scene.output_dir / 'video'
    vid_path.mkdir(exist_ok=True)
    write_video(vid_path / 'render.mp4', rendered_frames, fps=fps)


# ==============================================================================
#                               MAIN ENTRY POINT
# ==============================================================================

@click.command()
@click.option('--video', 'video_path', required=True, type=str, help='Path to video file or image folder.')
@click.option('--output', 'output_path', default='video_output', help='Root output directory.')
@click.option('--pretrained', default='Ruicheng/moge-vitl', help='MoGe model path.')
@click.option('--fov', 'fov_x', type=float, default=None, help='Horizontal FOV override.')
@click.option('--input-size', type=int, default=640, help='Resize input video to this longer side dimension.')
@click.option('--start', type=int, default=None, help='Start frame.')
@click.option('--end', type=int, default=None, help='End frame.')
@click.option('--skip', type=int, default=1, help='Skip frames.')
@click.option('--fps', type=int, default=24, help='Output video FPS.')
@click.option('--voxel-size', type=float, default=0.002, help='Voxel size for global point cloud (0 to disable).')
@click.option('--accumulate', 'accumulate_interval', type=int, default=0, help='Accumulate points every k frames for rendering. 0 keeps all previous frame point clouds.')
@click.option('--camera-distance', type=float, default=1.35, help='Multiplier that moves the render camera farther from the look-at point.')
@click.option('--camera-height', type=float, default=0.15, help='Camera height offset as a fraction of the scene bounding-box diagonal.')
@click.option('--render-point-size', type=int, default=1, help='Rendered point splat size in pixels. Use 1 for the finest point cloud.')
@click.option('--use-cache/--no-use-cache', default=None, help='Use existing cached reconstruction results without prompting.')
@click.option('--rescale', is_flag=True, help='Normalize point cloud scale for visualization. Leave unset to render reconstructed scale.')
def main(video_path, output_path, pretrained, fov_x, input_size, start, end, skip, fps, voxel_size, accumulate_interval, camera_distance, camera_height, render_point_size, use_cache, rescale):
    
    # 1. Setup
    video_path = Path(video_path)
    output_path = Path(output_path)
    scene = SceneReconstructor(video_path, output_path)
    
    # 2. Read Input Video
    print("Reading input frames...")
    frames = read_frames(video_path, start, end, skip, target_size=input_size)
    
    # Save raw input video for reference
    (scene.output_dir / 'video').mkdir(parents=True, exist_ok=True)
    write_video(scene.output_dir / 'video' / 'input.mp4', frames, fps=fps)
    
    # 3. Check for Cache & Decide Logic
    if scene.has_cached_results():
        if use_cache is None:
            # Ask user if they want to use existing results
            use_cache = click.confirm(f"Found existing results in {scene.result_dir}. Do you want to use them?")
    elif use_cache:
        raise FileNotFoundError(f"No cached reconstruction results found in {scene.result_dir}")
    else:
        use_cache = False
            
    if use_cache:
        # PATH A: Use Cache (Fast, No Model Load)
        scene.load_cached_results(frames)
    else:
        # PATH B: Run Inference (Slow, Load Models)
        scene.load_models(pretrained)
        scene.run_inference(frames, fov_x=fov_x)
    
    # 4. Generate Global Point Cloud
    scene.process_point_cloud(voxel_size=voxel_size)
    
    # 5. Render Visualization
    # Ensure helper functions like render_sequence are defined in the file
    render_sequence(
        scene,
        fps=fps,
        accumulate_interval=accumulate_interval,
        rescale=rescale,
        camera_distance=camera_distance,
        camera_height=camera_height,
        render_point_size=render_point_size,
    )
    
    print(f"Processing complete. Results saved to {scene.output_dir}")

if __name__ == '__main__':
    main()