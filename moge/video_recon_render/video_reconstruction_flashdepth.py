import os
import sys
from pathlib import Path
from typing import *
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'    # A workaround for potential compatibility issue with Windows
import itertools
import json
import json
import warnings

import numpy as np
from tqdm import tqdm, trange
import utils3d
import click
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import open3d as o3d
from omegaconf import DictConfig

from baselines.flashdepth.model import FlashDepth
from third_party_models import pdcnet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def solve_pose_ransac(
    p: np.ndarray,
    q: np.ndarray,
    w: np.ndarray = None,
    max_iters: int = 20,
    hypothetical_size: int = 10,
    inlier_thresh: float = 0.02
) -> np.ndarray:
    n = p.shape[0]
    if w is None:
        w = np.ones(p.shape[0])
    
    best_score, best_inlines = 0., np.zeros(n, dtype=bool)
    best_solution = np.eye(4, dtype=np.float32)

    for _ in range(max_iters):
        maybe_inliers = np.random.choice(n, size=hypothetical_size, replace=False)
        try:
            pose = utils3d.np.solve_pose(p[maybe_inliers], q[maybe_inliers], w[maybe_inliers], mode='rigid')
        except np.linalg.LinAlgError:
            continue
        transformed_p = utils3d.np.transform_points(p, pose)
        errors = w * np.linalg.norm(transformed_p - q, axis=1)
        inliers = errors < inlier_thresh
        
        score = inlier_thresh * n - np.clip(errors, None, inlier_thresh).sum()
        if score > best_score:
            best_score, best_inlines = score, inliers
            best_solution = utils3d.np.solve_pose(p[inliers], q[inliers], w[inliers], mode='rigid')
    
    return best_solution, best_inlines


def extract_corresponding_pixels(flow, mask_shape):
    h, w = mask_shape[:2]
    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    corresponding_x = (grid_x + flow[..., 0]).astype(int)
    corresponding_y = (grid_y + flow[..., 1]).astype(int)

    valid_mask = (corresponding_x >= 0) & (corresponding_x < w) & (corresponding_y >= 0) & (corresponding_y < h)
    return valid_mask, corresponding_x, corresponding_y


def read_frames_from_video(video_path, start: int = None, end: int = None, skip: int = None,target_size: Union[int, Tuple[int, int]] = None):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if isinstance(target_size, int):
        original_width, original_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        longer_size = max(original_width, original_height)
        target_width, target_height = int(original_width * target_size / longer_size), int(original_height * target_size / longer_size)
    else:
        target_width, target_height = target_size

    if start is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if end is not None and cap.get(cv2.CAP_PROP_POS_FRAMES) >= end:
            break
        if skip is not None and (cap.get(cv2.CAP_PROP_POS_FRAMES) - (start or 0)) % skip != 0:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return frames


def read_frames_from_folder(path: Union[str, os.PathLike], start: int = None, end: int = None, skip: int = None, target_size: Union[int, Tuple[int, int]] = None) -> Iterable[np.ndarray]:
    frame_paths = sorted(Path(path).glob('*.jpg'))
    frames = []
    for p in tqdm(frame_paths[start:end:skip], desc='Reading frames'):
        image = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if isinstance(target_size, int):
            longer_side = max(image.shape[:2])
            image = cv2.resize(image, (int(image.shape[1] * target_size / longer_side), int(image.shape[0] * target_size / longer_side)), cv2.INTER_AREA)
        elif isinstance(target_size, tuple):
            image = cv2.resize(image, target_size, cv2.INTER_AREA)
        frames.append(image)
    return frames


def write_video(path: Union[str, os.PathLike], frames: List[np.ndarray], fps: int = 20):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()


def write_ply(path: Union[str, os.PathLike], points: np.ndarray, colors: np.ndarray = None):
    """
    Write point cloud to PLY file format using open3d.
    
    Args:
        path: Output PLY file path
        points: Nx3 array of 3D points
        colors: Optional Nx3 array of RGB colors (0-255)
    """
    # Create open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    
    if colors is not None:
        # Ensure colors are in 0-1 range for open3d
        if colors.max() > 1.0:
            colors_normalized = colors.astype(np.float64) / 255.0
        else:
            colors_normalized = colors.astype(np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
    
    # Write PLY file
    o3d.io.write_point_cloud(str(path), pcd)


def downsample_pointcloud_voxel(points: np.ndarray, colors: np.ndarray = None, voxel_size: float = 0.01):
    """
    Downsample point cloud using voxel grid filtering with open3d.
    
    Args:
        points: Nx3 array of 3D points
        colors: Optional Nx3 array of RGB colors (0-255)
        voxel_size: Size of each voxel for downsampling
    
    Returns:
        Downsampled points and colors (if provided)
    """
    if len(points) == 0:
        return points, colors
    
    # Create open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    
    if colors is not None:
        # Ensure colors are in 0-1 range for open3d
        if colors.max() > 1.0:
            colors_normalized = colors.astype(np.float64) / 255.0
        else:
            colors_normalized = colors.astype(np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
    
    # Downsample using open3d
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Convert back to numpy arrays
    downsampled_points = np.asarray(downsampled_pcd.points).astype(np.float32)
    
    if colors is not None:
        downsampled_colors = np.asarray(downsampled_pcd.colors)
        # Convert colors back to 0-255 range if original was in that range
        if colors.max() > 1.0:
            downsampled_colors = (downsampled_colors * 255).astype(np.uint8)
        else:
            downsampled_colors = downsampled_colors.astype(np.float32)
    else:
        downsampled_colors = None
    
    return downsampled_points, downsampled_colors



def find_correspondence_by_pdcnet(pdcnet_model, image_ref: np.ndarray, mask_ref: np.ndarray, image_query: np.ndarray, mask_query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image_query.shape[:2]
    
    image_ref = torch.tensor(image_ref, dtype=torch.uint8, device=device).permute(2, 0, 1)
    image_query = torch.tensor(image_query, dtype=torch.uint8, device=device).permute(2, 0, 1)
    flow, confidence = pdcnet.predict_flow(pdcnet_model, image_query, image_ref)
    flow, confidence = flow.cpu().numpy(), confidence.cpu().numpy()
    
    uv_ref = utils3d.np.uv_map(height, width)
    uv_tgt = uv_ref + flow
    pixel_ref, pixel_tgt = utils3d.np.uv_to_pixel(uv_ref, (height, width)), utils3d.np.uv_to_pixel(uv_tgt, (height, width))
    pixel_ref, pixel_tgt = pixel_ref.round().astype(int), pixel_tgt.round().astype(int)
    valid = np.where(
        (confidence > 0.5) 
        & (pixel_tgt >= 0).all(axis=-1) 
        & (pixel_tgt <= [width - 1, height - 1]).all(axis=-1) 
        & mask_ref 
        & mask_query[pixel_tgt.clip(0, [width - 1, height - 1])[:, :, 1], pixel_tgt.clip(0, [width - 1, height - 1])[:, :, 0]]
    )
    pixel_ref, pixel_tgt = pixel_ref[valid], pixel_tgt[valid]
    
    return pixel_ref, pixel_tgt


def depth_to_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Back-project depth image to 3D points in camera space.
    
    Args:
        depth: Depth image (H, W) in meters
        intrinsics: 3x3 or 4x4 intrinsic matrix
    
    Returns:
        points: (H, W, 3) array of 3D points in camera space
    """
    h, w = depth.shape
    
    # Extract intrinsic parameters
    if intrinsics.shape == (4, 4):
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
    else:
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
    
    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert to normalized coordinates
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    
    # Back-project to 3D
    x = x_norm * depth
    y = y_norm * depth
    z = depth
    
    points = np.stack([x, y, z], axis=-1)
    return points


def intrinsics_from_fov(fov_x: float, height: int, width: int) -> np.ndarray:
    """
    Compute intrinsic matrix from horizontal FOV.
    
    Args:
        fov_x: Horizontal field of view in degrees
        height: Image height
        width: Image width
    
    Returns:
        intrinsics: 3x3 intrinsic matrix
    """
    fov_x_rad = np.deg2rad(fov_x)
    fx = width / (2 * np.tan(fov_x_rad / 2))
    fy = fx  # Assume square pixels
    cx = width / 2
    cy = height / 2
    
    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    return intrinsics



@click.command()
@click.option('--video', 'video_path', type=str, help='Input video path')
@click.option('--fov_x', type=float, default=None, help='Horizontal field of view in degrees')
@click.option('--start', type=int, default=None, help='Start frame index')
@click.option('--end', type=int, default=None, help='End frame index')
@click.option('--skip', type=int, default=1, help='Frame skip rate')
@click.option('--input_size', type=int, default=640, help='Resize the input video to a specific size (longer side)')
@click.option('--ref-offset', 'ref_offset', type=click.Tuple([int, int, int]), default=[1, 5, 21], help='Reference frame offset')
@click.option('--camera', 'camera_path', type=str, default=None, help='Trajectory file path')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, required=True, help='Path to FlashDepth pretrained model checkpoint')
@click.option('--output', 'output_path', type=str, default='video_output', help='Output directory')
@click.option('--fps', type=int, default=24, help='Output video FPS')
@click.option('--voxel-size', 'voxel_size', type=float, default=0.002, help='Voxel size for point cloud downsampling (0 to disable)')
@click.option('--max_res', type=int, default=1024, help='Maximum resolution dimension for model inference')
def main(video_path: str, fov_x: float, start: int, end: int, skip: int, input_size: int, ref_offset: List[int], camera_path: str, pretrained_model_name_or_path: str, output_path: str, fps: int, voxel_size: float, max_res: int):
    if Path(video_path).is_file():
        image_frames = read_frames_from_video(video_path, start, end, skip, target_size=input_size)
    elif Path(video_path).is_dir():
        image_frames = read_frames_from_folder(video_path, start, end, skip, target_size=input_size)
    else:
        raise ValueError(f"Invalid video path: {video_path}")
    video_name = Path(video_path).stem

    input_height, input_width = image_frames[0].shape[:2]
    num_frames = len(image_frames)

    prediction_frames, pose_frames, canonical_points_frames, inlier_mask_frames = [], [], [], []

    # Check if there are existing results
    if Path(output_path, video_name, 'result').exists():
        use_existing_results = click.confirm(f"Found existing results in {output_path}/{video_name}/result. Do you want to use them?")
    else:
        use_existing_results = False

    if use_existing_results:
        # Load existing results
        points_paths = sorted(Path(output_path, video_name, 'result').glob('*_points.exr'))
        for p in tqdm(points_paths, desc='Loading existing results'):
            points = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(p.as_posix().replace('_points.exr', '_mask.png')), cv2.IMREAD_GRAYSCALE) > 0
            cam = np.load(str(p.as_posix().replace('_points.exr', '_cam.npz')))
            pose, intrinsics = cam['pose'], cam['intrinsics']
            inlier_mask = cv2.imread(str(p.as_posix().replace('_points.exr', '_inlier_mask.png')), cv2.IMREAD_GRAYSCALE) > 0
            canonical_points = utils3d.np.transform_points(points, pose)
            prediction_frames.append((points, mask))
            canonical_points_frames.append(canonical_points)
            pose_frames.append((pose, intrinsics))
            inlier_mask_frames.append(inlier_mask)
    else:
        # Load FlashDepth model
        print(f"==> Loading FlashDepth model: {pretrained_model_name_or_path}")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        # Hardcoded configs (following recon_scannet_flashdepth.py)
        hybrid_configs = {
            'use_hybrid': False, 
            'teacher_model_path': None, 
            'teacher_resolution': 490, 
            'layers_to_skip': [1, 2, 3], 
            'num_blocks': 4, 
            'mlp_expand': 2, 
            'num_heads': 2
        }
        model_configs = {
            'vit_size': 'vitl',
            'patch_size': 14,
            'attn_class': 'MemEffAttention',
            'use_mamba': True, 
            'mamba_type': 'add', 
            'num_mamba_layers': 4, 
            'downsample_mamba': [0.1], 
            'mamba_pos_embed': None, 
            'mamba_in_dpt_layer': [3], 
            'mamba_d_conv': 4, 
            'mamba_d_state': 256, 
            'use_hydra': False, 
            'use_transformer_rnn': False, 
            'use_xlstm': False,
        }

        eval_args = {
            'save_depth_npy': False,
            'save_vis_map': False,
            'out_video': False,
            'out_mp4': False,
            'use_mamba': True,
            'resolution': 518,
            'print_time': True,
            'loss_type': 'l1',
            'use_all_frames': True,
            'use_metrics': False,
            'dummy_timing': False
        }
        
        flashdepth_model = FlashDepth(**dict( 
            batch_size=1, 
            hybrid_configs=DictConfig(hybrid_configs),
            training=False,
            **DictConfig(model_configs),
        )).to(device)
        flashdepth_model.load_state_dict(torch.load(pretrained_model_name_or_path, weights_only=True)["model"])
        flashdepth_model.eval()
        print(f"==> Model loaded on {device}")
        
        # Prepare images for FlashDepth inference
        print("==> Preparing images for FlashDepth inference...")
        color_images = []
        original_sizes = []
        for curr_image in image_frames:
            h, w = curr_image.shape[:2]
            original_sizes.append((h, w))
            
            # Resize to be divisible by patch_size (14) and within max_res
            patch_size = 14
            height = round(h / patch_size) * patch_size
            width = round(w / patch_size) * patch_size
            
            if max(height, width) > max_res:
                scale = max_res / max(h, w)
                height = round(h * scale / patch_size) * patch_size
                width = round(w * scale / patch_size) * patch_size
            
            if (height, width) != (h, w):
                curr_image_resized = cv2.resize(curr_image, (width, height), interpolation=cv2.INTER_LINEAR)
            else:
                curr_image_resized = curr_image
            
            color_images.append(curr_image_resized.astype(np.float32) / 255.0)
        
        # Run FlashDepth inference on all frames
        print(f"==> Running FlashDepth inference on {num_frames} frames...")
        with torch.no_grad():
            image_tensor = torch.from_numpy(np.stack(color_images)).permute(0, 3, 1, 2).to(device)
            # Ensure dimensions are divisible by patch_size
            if image_tensor.shape[-2] % 14 != 0 or image_tensor.shape[-1] % 14 != 0:
                h_new = image_tensor.shape[-2] // 14 * 14
                w_new = image_tensor.shape[-1] // 14 * 14
                image_tensor = F.interpolate(image_tensor, (h_new, w_new), mode='bilinear', align_corners=False)
            
            output = flashdepth_model(image_tensor.unsqueeze(0), **eval_args)
            depth_est_all = output['depth'].cpu().numpy()  # Shape: [B, N, H, W]
            
            # Handle depth shape: [B, N, H, W] -> [N, H, W]
            if depth_est_all.ndim == 4:
                # Remove batch dimension (B=1)
                depth_est_all = depth_est_all.squeeze(0)  # Now shape: [N, H, W]
            elif depth_est_all.ndim == 3:
                # Already in [N, H, W] format
                pass
            else:
                raise ValueError(f"Unexpected depth shape: {depth_est_all.shape}")
        
        print(f"==> FlashDepth inference complete. Depth shape: {depth_est_all.shape}")
        
        # Compute intrinsics from fov_x if provided, otherwise estimate from image size
        if fov_x is not None:
            intrinsics_3x3 = intrinsics_from_fov(fov_x, input_height, input_width)
        else:
            # Default intrinsics (assume reasonable FOV)
            intrinsics_3x3 = intrinsics_from_fov(60.0, input_height, input_width)
        
        # Convert to 4x4 for consistency
        intrinsics = np.eye(4, dtype=np.float32)
        intrinsics[:3, :3] = intrinsics_3x3
        
        # Load PDCNet for correspondence
        pdcnet_model = pdcnet.load_model('pretrained/PDCNet_megadepth.pth.tar')
        
        Path(output_path, video_name, 'result').mkdir(parents=True, exist_ok=True)
        
        # Process each frame: convert depth to points and solve pose
        for i_curr in trange(num_frames, desc='Processing frames'):
            curr_image = image_frames[i_curr]
            h_orig, w_orig = original_sizes[i_curr]
            
            # Get depth for current frame
            if i_curr >= depth_est_all.shape[0]:
                print(f"Warning: Frame {i_curr} exceeds depth array size {depth_est_all.shape[0]}. Skipping.")
                continue
            d_est = depth_est_all[i_curr].copy()  # Shape: [H, W]
            
            # Resize depth to original image size if needed
            if d_est.shape != (h_orig, w_orig):
                d_est = cv2.resize(d_est, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            
            # Resize intrinsics if image was resized
            if (h_orig, w_orig) != (input_height, input_width):
                scale_h = h_orig / input_height
                scale_w = w_orig / input_width
                intrinsics_scaled = intrinsics.copy()
                intrinsics_scaled[0, 0] *= scale_w
                intrinsics_scaled[1, 1] *= scale_h
                intrinsics_scaled[0, 2] *= scale_w
                intrinsics_scaled[1, 2] *= scale_h
            else:
                intrinsics_scaled = intrinsics
            
            # Back-project depth to 3D points
            curr_points = depth_to_points(d_est, intrinsics_scaled)
            
            # Create mask (valid depth points)
            # curr_mask = (d_est > 0.1) & (d_est < 80.0)
            curr_mask = ~utils3d.np.depth_map_edge(d_est, rtol=0.05, mask=None)
            
            # Solve pose
            if i_curr == 0:
                # For the first frame, just normalize the scale and set identity pose
                mean_depth = np.mean(d_est[curr_mask])
                if mean_depth > 0:
                    scale_factor = 1.0 / mean_depth
                else:
                    scale_factor = 1.0
                pose = scale_factor * np.eye(4, dtype=np.float32)
                inlier_mask = np.zeros_like(curr_mask, dtype=bool)
            else:
                # Similar registration with previous reference frames
                ref_indices = [i_curr - i for i in ref_offset if i_curr - i >= 0]
                assert len(ref_indices) > 0
                p, q, w, pixel_indices = [], [], [], []
                for i_ref in ref_indices:
                    ref_image, ref_points, ref_mask = image_frames[i_ref], canonical_points_frames[i_ref], prediction_frames[i_ref][1]
                    corresp_pixel_ref, corresp_pixel_curr = find_correspondence_by_pdcnet(pdcnet_model, ref_image, ref_mask, curr_image, curr_mask)
                    p.append(curr_points[corresp_pixel_curr[:, 1], corresp_pixel_curr[:, 0]])
                    q.append(ref_points[corresp_pixel_ref[:, 1], corresp_pixel_ref[:, 0]])
                    w.append(1 / curr_points[corresp_pixel_curr[:, 1], corresp_pixel_curr[:, 0], 2])
                    pixel_indices.append(corresp_pixel_curr[:, 1] * w_orig + corresp_pixel_curr[:, 0])
                p, q, w, pixel_indices = np.concatenate(p), np.concatenate(q), np.concatenate(w), np.concatenate(pixel_indices)
                
                # Check if we have enough correspondences
                if len(p) < 10:  # Need at least 10 points for RANSAC
                    print(f"Warning: Only {len(p)} correspondences found for frame {i_curr}. Using identity pose.")
                    pose = np.eye(4, dtype=np.float32)
                    inlier_mask = np.zeros_like(curr_mask, dtype=bool)
                else:
                    pose, inlines = solve_pose_ransac(p, q, w)
                    inlier_pixel_indices, inlier_pixel_cnts = np.unique(pixel_indices[inlines], return_counts=True)
                    inlier_pixel_indices = inlier_pixel_indices[inlier_pixel_cnts == len(ref_indices)]
                    inlier_mask = np.zeros_like(curr_mask)
                    inlier_mask[inlier_pixel_indices // w_orig, inlier_pixel_indices % w_orig] = True
            
            s = np.linalg.det(pose[:3, :3])

            # Save intermediate results
            cv2.imwrite(str(Path(output_path, video_name, 'result', f'{i_curr:05d}_points.exr')), curr_points.astype(np.float32))
            cv2.imwrite(str(Path(output_path, video_name, 'result', f'{i_curr:05d}_depth_registered.exr')), s * d_est.astype(np.float32))
            cv2.imwrite(str(Path(output_path, video_name, 'result', f'{i_curr:05d}_mask.png')), (curr_mask * 255).astype(np.uint8))
            np.savez(str(Path(output_path, video_name, 'result', f'{i_curr:05d}_cam.npz')), pose=pose, intrinsics=intrinsics_scaled[:3, :3])
            cv2.imwrite(str(Path(output_path, video_name, 'result', f'{i_curr:05d}_inlier_mask.png')), (inlier_mask * 255).astype(np.uint8))
            
            curr_points_canonical = utils3d.np.transform_points(curr_points, pose)
            prediction_frames.append((curr_points, curr_mask))
            pose_frames.append((pose, intrinsics_scaled[:3, :3].astype(np.float32)))
            canonical_points_frames.append(curr_points_canonical.astype(np.float32))
            inlier_mask_frames.append(inlier_mask)

    # Combine all point clouds and save to PLY
    print('Combining point clouds from all frames...')
    combined_points = []
    combined_colors = []
    
    # Create directory for separate point clouds
    separate_pc_dir = Path(output_path, video_name, 'pointclouds')
    separate_pc_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(num_frames), desc='Processing point clouds'):
        mask = prediction_frames[idx][1]
        points = canonical_points_frames[idx][mask]
        colors = image_frames[idx][mask]
        # Ensure colors are in 0-255 uint8 format
        if colors.dtype != np.uint8:
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            else:
                colors = colors.astype(np.uint8)
        
        # Save individual point cloud
        frame_ply_path = separate_pc_dir / f'frame_{idx:05d}.ply'
        write_ply(frame_ply_path, points, colors)
        
        # Add to combined point cloud (every 15th frame)
        if idx % 15 == 0:
            combined_points.append(points)
            combined_colors.append(colors)
    
    combined_points = np.concatenate(combined_points, axis=0)
    combined_colors = np.concatenate(combined_colors, axis=0)
    
    # Downsample point cloud if voxel_size > 0
    if voxel_size > 0:
        print(f'Downsampling point cloud with voxel size {voxel_size}...')
        original_count = combined_points.shape[0]
        combined_points, combined_colors = downsample_pointcloud_voxel(combined_points, combined_colors, voxel_size=voxel_size)
        print(f'Downsampled from {original_count} to {combined_points.shape[0]} points ({100 * combined_points.shape[0] / original_count:.1f}%)')
    
    # Save combined point cloud with camera frustums to PLY
    ply_output_path = Path(output_path, video_name, 'combined_pointcloud.ply')
    print(f'Saving combined point cloud to {ply_output_path}...')
    write_ply(ply_output_path, combined_points, combined_colors)
    
    # Render
    if camera_path is not None:
        render_height, render_width = 768, 1024

        # Load camera trajectory
        with open(camera_path, 'r') as f:
            camera_config = json.load(f)
            eye_traj = CubicSpline(
                np.linspace(0, num_frames - 1, len(camera_config['eye'])), 
                np.array(camera_config['eye'], dtype=np.float32), 
                bc_type="periodic" if camera_config['eye'][0] == camera_config['eye'][-1] else "not-a-knot"
            )
            look_at_traj = CubicSpline(
                np.linspace(0, num_frames - 1, len(camera_config['look_at'])), 
                np.array(camera_config['look_at'], dtype=np.float32), 
                bc_type="periodic" if camera_config['look_at'][0] == camera_config['look_at'][-1] else "not-a-knot"
            )
            up = np.array(camera_config['up'], dtype=np.float32)
        render_projection = utils3d.np.perspective_from_fov(fov_y=np.deg2rad(camera_config['fov']), near=0.01, far=np.inf, aspect_ratio=render_width / render_height)

        # Save input video
        Path(output_path, video_name, 'video').mkdir(exist_ok=True)
        write_video(Path(output_path, video_name, 'video', 'input.mp4'), image_frames, fps=fps)

        # Render per-frame point cloud
        render_frames = []
        for idx in trange(num_frames, desc='Render'):
            _, mask = prediction_frames[idx]
            render_view = utils3d.np.view_look_at(eye=eye_traj(idx), look_at=look_at_traj(idx), up=up).astype(np.float32)
            
            pose, intrinsics = pose_frames[idx]
            extrinsics = np.linalg.inv(pose)

            # Render point cloud
            render_output = utils3d.np.rasterize_point_cloud(
                (render_height, render_width),
                points=canonical_points_frames[idx][mask].astype(np.float32),
                attributes=image_frames[idx][mask].astype(np.float32),
                point_sizes=3,
                point_shape='circle',
                view=render_view,
                projection=render_projection,
                return_depth=True
            )
            # Render camera frustum
            camera_vertices, camera_edges, _ = utils3d.np.create_camera_frustum_mesh(extrinsics, intrinsics, 0.1)
            render_output = utils3d.np.rasterize_lines(
                (render_height, render_width),
                vertices=camera_vertices,
                lines=camera_edges,
                attributes=np.array([[0, 255, 0]], dtype=np.float32).repeat(camera_vertices.shape[0], axis=0),
                view=render_view,
                projection=render_projection,
                line_width=3,
                background=render_output,
                return_depth=True
            )
            render_image = np.where(render_output['mask'][:, :, None], render_output['image'], 255).astype(np.uint8)    # White background
            render_frames.append(render_image)

        # Save rendered video
        output_width = 720
        output_video_frames = []
        for i in range(num_frames):
            output_image = np.concatenate([
                cv2.resize(image_frames[i], (output_width, int(output_width / input_width * input_height))),
                cv2.resize(render_frames[i], (output_width, int(output_width / render_width * render_height))),
            ], axis=0)
            output_video_frames.append(output_image)
        write_video(Path(output_path, video_name, 'video', 'render.mp4'), output_video_frames, fps=fps)



if __name__ == '__main__':
    main()


