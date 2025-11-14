import os
import sys
from pathlib import Path
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

import cv2
import click
import json
import numpy as np
from tqdm import tqdm
import open3d as o3d
import torch
import warnings
import utils3d

from moge.model.v1 import MoGeModel
from moge.video_benchmark.eval.metric import *
import moge.video_benchmark.eval.metric as metric

eval_metrics = [
    "abs_relative_difference",
    "delta1_acc",
    "rmse_linear",
    "squared_relative_difference",
    "rmse_log",
    "log10",
    "delta2_acc",
    "delta3_acc",
    "i_rmse",
    "silog_rmse",
]


def load_intrinsic(intrinsic_path):
    """Load 4x4 intrinsic matrix from file."""
    intrinsic = np.loadtxt(intrinsic_path)
    if intrinsic.shape == (3, 3):
        # Convert 3x3 to 4x4 if needed
        intrinsic_4x4 = np.eye(4)
        intrinsic_4x4[:3, :3] = intrinsic
        return intrinsic_4x4
    return intrinsic


def load_pose(pose_path):
    """Load 4x4 pose matrix (camera to world) from file."""
    pose = np.loadtxt(pose_path)
    if pose.shape == (3, 4):
        # Convert 3x4 to 4x4 if needed
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :] = pose
        return pose_4x4
    return pose


def depth_to_points(depth, intrinsic, depth_scale=1000.0):
    """
    Back-project depth image to 3D points in camera space.
    
    Args:
        depth: Depth image (H, W) in meters (after dividing by depth_scale)
        intrinsic: 4x4 intrinsic matrix
        depth_scale: Scale factor for depth (default 1000.0 for ScanNet)
    
    Returns:
        points: (H, W, 3) array of 3D points in camera space
    """
    h, w = depth.shape
    
    # Get focal lengths and principal point from intrinsic
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    
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


def align_depth_lstsq(depth_pred, depth_gt, depth_min=0.1, depth_max=80.0):
    """
    Align predicted depth to GT depth using least squares: gt = scale * pred + shift
    
    Args:
        depth_pred: Predicted depth map (H, W)
        depth_gt: GT depth map (H, W)
        depth_min: Minimum valid depth
        depth_max: Maximum valid depth
    
    Returns:
        depth_aligned: Aligned depth map (H, W)
        scale: Scale factor
        shift: Shift factor
    """
    # Create valid mask
    valid_mask = (depth_gt > depth_min) & (depth_gt < depth_max) & (depth_pred > 1e-3)
    
    if np.sum(valid_mask) < 100:
        # Not enough valid points, return original prediction
        return depth_pred, 1.0, 0.0
    
    # Prepare data for lstsq
    pred_masked = depth_pred[valid_mask].reshape(-1, 1)
    gt_masked = depth_gt[valid_mask].reshape(-1, 1)
    
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    
    # Solve for scale and shift
    try:
        X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
        scale, shift = X.flatten()
    except (np.linalg.LinAlgError, ValueError):
        # If lstsq fails, return original prediction
        return depth_pred, 1.0, 0.0
    
    if not (np.isfinite(scale) and np.isfinite(shift)):
        return depth_pred, 1.0, 0.0
    
    # Apply alignment
    depth_aligned = scale * depth_pred + shift
    depth_aligned = np.clip(depth_aligned, a_min=depth_min, a_max=depth_max)
    
    return depth_aligned, scale, shift


def process_frame_gt(color_path, depth_path, intrinsic, pose_path, depth_scale=1000.0, depth_min=0.1, depth_max=80.0):
    """
    Process a single frame using GT depth and return world points and colors.
    
    Args:
        color_path: Path to color image
        depth_path: Path to depth image
        intrinsic: 4x4 intrinsic matrix (numpy array)
        pose_path: Path to pose matrix
        depth_scale: Scale factor for depth (1000.0 for ScanNet)
        depth_min: Minimum valid depth in meters
        depth_max: Maximum valid depth in meters
    
    Returns:
        world_points: (N, 3) array of 3D points in world space
        colors: (N, 3) array of RGB colors (0-255)
    """
    # Load color image
    color = cv2.imread(str(color_path))
    if color is None:
        raise ValueError(f"Failed to load color image: {color_path}")
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # Crop black borders (ScanNet format)
    color = color[8:-8, 11:-11, :]
    h, w = color.shape[:2]
    
    # Load depth image
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to load depth image: {depth_path}")
    
    # Convert depth from 16-bit (with scale 1000) to meters
    depth_m = depth.astype(np.float32) / depth_scale
    
    # Resize depth if it doesn't match color
    if depth_m.shape != (h, w):
        depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Load pose
    pose = load_pose(pose_path)

    if np.isinf(pose).any():
        print(f"Warning: Infinite pose for {pose_path}")
        return None, None
    # Create validity mask
    valid_mask = (depth_m > depth_min) & (depth_m < depth_max)
    
    # Back-project to 3D points in camera space
    points_cam = depth_to_points(depth_m, intrinsic, depth_scale=1.0)
    
    # Transform to world space
    points_cam_flat = points_cam.reshape(-1, 3)
    points_cam_hom = np.concatenate([points_cam_flat, np.ones((points_cam_flat.shape[0], 1))], axis=1)
    points_world_hom = (pose @ points_cam_hom.T).T
    points_world_flat = points_world_hom[:, :3]
    points_world = points_world_flat.reshape(h, w, 3)
    
    # Get valid points and colors
    valid_mask_flat = valid_mask.flatten()
    world_points = points_world.reshape(-1, 3)[valid_mask_flat]
    colors = color.reshape(-1, 3)[valid_mask_flat]
    
    return world_points, colors


def process_frame_predicted(color_path, depth_aligned, depth_orig, points_pred, mask_pred, intrinsic, pose_path, 
                             depth_min=0.1, depth_max=80.0):
    """
    Process a single frame using aligned predicted depth and return world points and colors.
    
    Args:
        color_path: Path to color image
        depth_aligned: Aligned depth map (H, W) in meters
        depth_orig: Original predicted depth map (H, W) in meters (before alignment)
        points_pred: Predicted 3D points in camera space (H, W, 3), can be None
        mask_pred: Predicted validity mask (H, W), can be None
        intrinsic: 4x4 intrinsic matrix (numpy array)
        pose_path: Path to pose matrix
        depth_min: Minimum valid depth in meters
        depth_max: Maximum valid depth in meters
    
    Returns:
        world_points: (N, 3) array of 3D points in world space
        colors: (N, 3) array of RGB colors (0-255)
    """
    # Load color image
    color = cv2.imread(str(color_path))
    if color is None:
        raise ValueError(f"Failed to load color image: {color_path}")
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # Crop black borders (ScanNet format)
    color = color[8:-8, 11:-11, :]
    h, w = color.shape[:2]
    
    # Resize depths if needed
    if depth_aligned.shape != (h, w):
        depth_aligned = cv2.resize(depth_aligned, (w, h), interpolation=cv2.INTER_LINEAR)
    if depth_orig.shape != (h, w):
        depth_orig = cv2.resize(depth_orig, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Load pose
    pose = load_pose(pose_path)
    if np.isinf(pose).any():
        print(f"Warning: Infinite pose for {pose_path}")
        return None, None
    # Use predicted points if available and scale them, otherwise back-project
    if points_pred is not None and points_pred.shape[:2] == (h, w):
        # Scale predicted points according to aligned depth along their rays
        depth_orig_safe = np.where(depth_orig < 1e-6, 1e-6, depth_orig)
        ray_scaler = depth_aligned / depth_orig_safe
        points_cam = points_pred * ray_scaler[..., np.newaxis]
    else:
        # Back-project from aligned depth
        points_cam = depth_to_points(depth_aligned, intrinsic, depth_scale=1.0)
    
    # Use predicted mask if available
    if mask_pred is not None:
        if mask_pred.shape[:2] != (h, w):
            mask_pred = cv2.resize(mask_pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        valid_mask = mask_pred & (depth_aligned > depth_min) & (depth_aligned < depth_max)
    else:
        valid_mask = (depth_aligned > depth_min) & (depth_aligned < depth_max)
    
    # Transform to world space
    points_cam_flat = points_cam.reshape(-1, 3)
    points_cam_hom = np.concatenate([points_cam_flat, np.ones((points_cam_flat.shape[0], 1))], axis=1)
    points_world_hom = (pose @ points_cam_hom.T).T
    points_world_flat = points_world_hom[:, :3]
    points_world = points_world_flat.reshape(h, w, 3)
    
    # Get valid points and colors
    valid_mask_flat = valid_mask.flatten()
    world_points = points_world.reshape(-1, 3)[valid_mask_flat]
    colors = color.reshape(-1, 3)[valid_mask_flat]
    
    return world_points, colors


def save_point_cloud(points_list, colors_list, output_path, voxel_size=0.02):
    """Combine and save point cloud."""
    if not points_list:
        print(f"Warning: No points to save for {output_path}")
        return
    
    print(f"Combining {len(points_list)} point clouds...")
    combined_points = np.concatenate(points_list, axis=0)
    combined_colors = np.concatenate(colors_list, axis=0)
    
    print(f"Total points: {len(combined_points)}")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors / 255.0)  # Open3D expects colors in [0, 1]
    
    # Downsample if requested
    if voxel_size > 0:
        print(f"Downsampling with voxel size {voxel_size}...")
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Points after downsampling: {len(pcd_down.points)}")
        pcd = pcd_down
    
    # Save point cloud
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"Successfully saved point cloud to {output_path}")
    print(f"  Total points: {len(pcd.points)}")
    print(f"  Bounding box: {pcd.get_axis_aligned_bounding_box()}")


@click.command()
@click.option('--scene_dir', type=click.Path(exists=True), required=True,
              help='Path to scene directory containing color/, depth/, intrinsic/, pose/ subdirectories')
@click.option('--output', type=click.Path(), default=None,
              help='Output directory for point clouds. If None, saves to scene_dir/')
@click.option('--depth_scale', type=float, default=1000.0,
              help='Depth scale factor (default: 1000.0 for ScanNet)')
@click.option('--depth_min', type=float, default=0.1,
              help='Minimum valid depth in meters (default: 0.1)')
@click.option('--depth_max', type=float, default=80.0,
              help='Maximum valid depth in meters (default: 80.0)')
@click.option('--voxel_size', type=float, default=0.02,
              help='Voxel size for downsampling (default: 0.02). Set to 0 to disable downsampling.')
@click.option('--frame_skip', type=int, default=1,
              help='Process every Nth frame (default: 1, process all frames)')
@click.option('--start_frame', type=int, default=0,
              help='Start frame index (default: 0)')
@click.option('--num_frames', type=int, default=-1,
              help='Number of frames to process (default: -1, process all frames)')
@click.option('--use_model_depth', is_flag=True,
              help='Use model predicted depth and generate aligned point clouds')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default='Ruicheng/moge-vitl',
              help='Pretrained model name or path (default: Ruicheng/moge-vitl)')
@click.option('--resolution_level', type=int, default=9,
              help='Resolution level for inference [0-9] (default: 9)')
@click.option('--num_tokens', type=int, default=None,
              help='Number of tokens for inference. Overrides resolution_level.')
@click.option('--use_fp16', is_flag=True,
              help='Use fp16 precision for 2x faster inference')
@click.option('--image_based', is_flag=True,
              help='Use image-based inference (affects only model inference method, marked in filename)')
@click.option('--enable_eval', is_flag=True,
              help='Enable evaluation metrics computation (requires GT depth)')
def main(scene_dir, output, depth_scale, depth_min, depth_max, voxel_size, frame_skip, start_frame, num_frames,
         use_model_depth, pretrained_model_name_or_path, resolution_level, num_tokens, use_fp16, image_based, enable_eval):
    """
    Reconstruct a ScanNet scene by combining point clouds from color, depth, intrinsic, and pose data.
    Always generates GT reconstruction. If --use_model_depth is set, also generates per_frame_align and sequential_align reconstructions.
    """
    scene_dir = Path(scene_dir)
    
    # Check subdirectories exist
    color_dir = scene_dir / 'color'
    depth_dir = scene_dir / 'depth'
    intrinsic_dir = scene_dir / 'intrinsic'
    pose_dir = scene_dir / 'pose'
    
    for subdir, name in [(color_dir, 'color'), (depth_dir, 'depth'), (intrinsic_dir, 'intrinsic'), (pose_dir, 'pose')]:
        if not subdir.exists():
            print(f"Error: {name} directory not found: {subdir}")
            return
    
    # Load global intrinsic files
    intrinsic_color_file = intrinsic_dir / 'intrinsic_color.txt'
    if not intrinsic_color_file.exists():
        print(f"Error: intrinsic_color.txt not found in {intrinsic_dir}")
        return
    
    # Load intrinsic_color.txt (4x4 matrix) - used for all frames
    intrinsic_color = load_intrinsic(intrinsic_color_file)
    print(f"Loaded intrinsic_color.txt: shape {intrinsic_color.shape}")
    
    # Find all frame files
    color_files_all = list(color_dir.glob('*.jpg')) + list(color_dir.glob('*.png'))
    
    if not color_files_all:
        print(f"Error: No color images found in {color_dir}")
        return
    
    # Sort files numerically by extracting the number from filename
    def extract_number(file_path):
        """Extract numeric part from filename for sorting."""
        stem = file_path.stem
        if stem.startswith('frame_'):
            stem = stem[6:]
        try:
            return int(stem)
        except ValueError:
            return float('inf')
    
    color_files = sorted(color_files_all, key=extract_number)
    
    # Extract frame names and match with corresponding files
    frame_names = []
    for color_file in color_files:
        stem = color_file.stem
        if stem.startswith('frame_'):
            frame_name = stem[6:]
        else:
            frame_name = stem
        
        # Check if corresponding files exist
        depth_file = depth_dir / f"{frame_name}.png"
        if not depth_file.exists():
            depth_file = depth_dir / f"frame_{frame_name}.png"
        
        pose_file = pose_dir / f"{frame_name}.txt"
        if not pose_file.exists():
            pose_file = pose_dir / f"frame_{frame_name}.txt"
        
        if depth_file.exists() and pose_file.exists():
            frame_names.append((color_file, depth_file, pose_file, frame_name))
    
    if not frame_names:
        print(f"Error: No matching frame sets found. Check file naming conventions.")
        print(f"  Found {len(color_files)} color images")
        print(f"  Sample color file: {color_files[0]}")
        return
    
    total_frames = len(frame_names)
    
    # Apply start_frame and num_frames
    start_idx = max(0, start_frame)
    if start_idx >= total_frames:
        print(f"Error: start_frame {start_frame} is out of bounds (found {total_frames} frames).")
        return
    
    if num_frames == -1:
        end_idx = total_frames
    else:
        end_idx = min(total_frames, start_idx + frame_skip * num_frames)
    
    frame_names = frame_names[start_idx:end_idx:frame_skip]
    
    print(f"Found {total_frames} total frames")
    print(f"Processing frames [{start_idx}:{end_idx}] with skip={frame_skip}")
    print(f"  -> {len(frame_names)} frames to process")
    if frame_names:
        print(f"  First frame: {frame_names[0][3]}, Last frame: {frame_names[-1][3]}")
    
    # Set up output directory
    if output is None:
        output_dir = scene_dir
    else:
        output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== Step 1: Always reconstruct GT point cloud ==========
    print("\n" + "="*60)
    print("Step 1: Reconstructing GT point cloud")
    print("="*60)
    all_world_points_gt = []
    all_colors_gt = []
    
    for color_file, depth_file, pose_file, frame_name in tqdm(frame_names, desc="Processing GT frames"):
        world_points, colors = process_frame_gt(
            color_file, depth_file, intrinsic_color, pose_file,
            depth_scale=depth_scale, depth_min=depth_min, depth_max=depth_max
        )
        if world_points is None or colors is None:
            continue
        all_world_points_gt.append(world_points)
        all_colors_gt.append(colors)
    
    if all_world_points_gt:
        filename_suffix = "_image_based" if (use_model_depth and image_based) else ""
        gt_output_path = output_dir / f"gt{filename_suffix}.ply"
        save_point_cloud(all_world_points_gt, all_colors_gt, gt_output_path, voxel_size)
    
    # ========== Step 2: If using model, generate predicted depth and aligned reconstructions ==========
    if not use_model_depth:
        print("\nDone! (Only GT reconstruction generated)")
        return
    
    print("\n" + "="*60)
    print("Step 2: Running model inference")
    print("="*60)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> Loading MoGe model: {pretrained_model_name_or_path}")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()
    print(f"==> Model loaded on {device}")
    
    # Load all color images
    print("==> Loading all color images...")
    color_images = []
    for color_file, _, _, _ in frame_names:
        color = cv2.imread(str(color_file))
        if color is not None:
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            # Crop black borders (ScanNet format)
            color = color[8:-8, 11:-11, :]
            color_images.append(color.astype(np.float32) / 255.0)
    
    if not color_images:
        print("Error: No valid color images found")
        return
    
    # Run model inference
    print(f"==> Running MoGe inference ({'image-based' if image_based else 'video-based'})...")
    with torch.no_grad():
        image_tensor = torch.from_numpy(np.stack(color_images)).permute(0, 3, 1, 2).to(device)
        output = model.infer_video(
            image_tensor, fov_x=None, resolution_level=resolution_level,
            num_tokens=num_tokens, use_fp16=use_fp16, image_based=image_based
        )
        depth_est_all = output['depth'].cpu().numpy()
        points_est_all = output['points'].cpu().numpy()
        mask_est_all = output['mask'].cpu().numpy()
    
    print("==> Model inference complete")
    
    # ========== Step 3: Compute sequence-level alignment (global scale and shift) ==========
    print("\n" + "="*60)
    print("Step 3: Computing sequence-level alignment")
    print("="*60)
    
    all_pred_depths_masked = []
    all_gt_depths_masked = []
    
    for i, (_, depth_file, _, _) in enumerate(tqdm(frame_names, desc="Collecting data for sequence alignment")):
        if i >= len(depth_est_all):
            break
        
        d_est = depth_est_all[i]
        mask_est = mask_est_all[i]
        
        # Load GT depth (keep original size)
        depth_gt = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        if depth_gt is None:
            continue
        
        depth_gt = depth_gt.astype(np.float32) / depth_scale
        h_gt, w_gt = depth_gt.shape[:2]
        # Resize predicted depth to match GT depth size
        if d_est.shape != (h_gt, w_gt):
            d_est_resized = cv2.resize(d_est, (w_gt, h_gt), interpolation=cv2.INTER_LINEAR)
            mask_est_resized = cv2.resize(mask_est.astype(np.uint8), (w_gt, h_gt), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            d_est_resized = d_est
            mask_est_resized = mask_est
        
        # Create valid mask
        valid_mask_i = (depth_gt > depth_min) & (depth_gt < depth_max) & (d_est_resized > 1e-3)
        valid_mask_i &= utils3d.numpy.depth_edge(d_est_resized, rtol=0.03, mask=mask_est_resized)
        
        if np.sum(valid_mask_i) > 100:
            all_pred_depths_masked.append(d_est_resized[valid_mask_i].reshape(-1, 1))
            all_gt_depths_masked.append(depth_gt[valid_mask_i].reshape(-1, 1))
    
    if not all_pred_depths_masked:
        print("Warning: No valid data for sequence alignment. Skipping sequence-aligned reconstruction.")
        seq_scale, seq_shift = 1.0, 0.0
    else:
        # Solve for global scale and shift
        print("==> Solving for sequence-level scale and shift...")
        all_preds = np.concatenate(all_pred_depths_masked)
        all_gts = np.concatenate(all_gt_depths_masked)
        
        _ones = np.ones_like(all_preds)
        A = np.concatenate([all_preds, _ones], axis=-1)
        
        X = np.linalg.lstsq(A, all_gts, rcond=None)[0]
        seq_scale, seq_shift = X.flatten()
        print(f"==> Sequence alignment: scale = {seq_scale:.4f}, shift = {seq_shift:.4f}")
    
    # ========== Step 4: Reconstruct per-frame aligned point cloud ==========
    print("\n" + "="*60)
    print("Step 4: Reconstructing per-frame aligned point cloud")
    print("="*60)
    
    all_world_points_perframe = []
    all_colors_perframe = []
    
    # Store aligned depths and GT depths for evaluation
    per_frame_aligned_depths = []
    per_frame_gt_depths = []
    
    for idx, (color_file, depth_file, pose_file, frame_name) in enumerate(tqdm(frame_names, desc="Processing per-frame aligned frames")):
        if idx >= len(depth_est_all):
            break
        
        try:
            d_est = depth_est_all[idx]
            pts_est = points_est_all[idx]
            m_est = mask_est_all[idx]
            
            # Load GT depth (keep original size)
            depth_gt = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            if depth_gt is None:
                continue
            
            depth_gt = depth_gt.astype(np.float32) / depth_scale
            h_gt, w_gt = depth_gt.shape[:2]
            
            # Resize predicted depth and related data to match GT depth size
            if d_est.shape != (h_gt, w_gt):
                d_est = cv2.resize(d_est, (w_gt, h_gt), interpolation=cv2.INTER_LINEAR)
                if pts_est is not None and pts_est.shape[:2] != (h_gt, w_gt):
                    pts_est = cv2.resize(pts_est, (w_gt, h_gt), interpolation=cv2.INTER_LINEAR)
                if m_est is not None and m_est.shape[:2] != (h_gt, w_gt):
                    m_est = cv2.resize(m_est.astype(np.uint8), (w_gt, h_gt), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # Per-frame alignment
            depth_aligned, scale_i, shift_i = align_depth_lstsq(d_est, depth_gt, depth_min, depth_max)
            print(f"==> Per-frame alignment: scale = {scale_i:.4f}, shift = {shift_i:.4f}")
            print(f"==> Sequence alignment: scale = {seq_scale:.4f}, shift = {seq_shift:.4f}")
            
            # Store for evaluation if enabled
            if enable_eval:
                per_frame_aligned_depths.append(depth_aligned)
                per_frame_gt_depths.append(depth_gt)
            
            mask_valid = (depth_gt > depth_min) & (depth_gt < depth_max) & (depth_aligned > 1e-3)
            # Process frame with aligned depth
            world_points, colors = process_frame_predicted(
                color_file, depth_aligned, d_est, pts_est, mask_valid, intrinsic_color, pose_file,
                depth_min=depth_min, depth_max=depth_max
            )
            if world_points is None or colors is None:
                continue
            
            all_world_points_perframe.append(world_points)
            all_colors_perframe.append(colors)
        except Exception as e:
            print(f"Warning: Failed to process per-frame aligned frame {frame_name}: {e}")
            continue
    
    if all_world_points_perframe:
        filename_suffix = "_image_based" if image_based else ""
        perframe_output_path = output_dir / f"per_frame_align{filename_suffix}.ply"
        save_point_cloud(all_world_points_perframe, all_colors_perframe, perframe_output_path, voxel_size)
    
    # ========== Step 5: Reconstruct sequence-aligned point cloud ==========
    print("\n" + "="*60)
    print("Step 5: Reconstructing sequence-aligned point cloud")
    print("="*60)
    
    all_world_points_seq = []
    all_colors_seq = []
    
    # Store aligned depths and GT depths for evaluation
    seq_aligned_depths = []
    seq_gt_depths = []
    
    for idx, (color_file, depth_file, pose_file, frame_name) in enumerate(tqdm(frame_names, desc="Processing sequence-aligned frames")):
        if idx >= len(depth_est_all):
            break
        
        try:
            d_est = depth_est_all[idx]
            pts_est = points_est_all[idx]
            m_est = mask_est_all[idx]
            
            # Load GT depth to get its size (keep original size)
            depth_gt = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            if depth_gt is None:
                continue
            
            depth_gt = depth_gt.astype(np.float32) / depth_scale
            h_gt, w_gt = depth_gt.shape[:2]
            
            # Resize predicted depth and related data to match GT depth size
            if d_est.shape != (h_gt, w_gt):
                d_est = cv2.resize(d_est, (w_gt, h_gt), interpolation=cv2.INTER_NEAREST)
                if pts_est is not None and pts_est.shape[:2] != (h_gt, w_gt):
                    pts_est = cv2.resize(pts_est, (w_gt, h_gt), interpolation=cv2.INTER_LINEAR)
                if m_est is not None and m_est.shape[:2] != (h_gt, w_gt):
                    m_est = cv2.resize(m_est.astype(np.uint8), (w_gt, h_gt), interpolation=cv2.INTER_NEAREST).astype(bool)

            # Apply sequence-level alignment
            depth_aligned = seq_scale * d_est + seq_shift
            depth_aligned = np.clip(depth_aligned, a_min=depth_min, a_max=depth_max)
            mask_valid = (depth_gt > depth_min) & (depth_gt < depth_max) & (depth_aligned > 1e-3)

            # Store for evaluation if enabled
            if enable_eval:
                seq_aligned_depths.append(depth_aligned)
                seq_gt_depths.append(depth_gt)
            
            # Process frame with aligned depth
            world_points, colors = process_frame_predicted(
                color_file, depth_aligned, d_est, None, mask_valid, intrinsic_color, pose_file,
                depth_min=depth_min, depth_max=depth_max
            )
            if world_points is None or colors is None:
                continue
            
            all_world_points_seq.append(world_points)
            all_colors_seq.append(colors)
        except Exception as e:
            print(f"Warning: Failed to process sequence-aligned frame {frame_name}: {e}")
            continue
    
    if all_world_points_seq:
        filename_suffix = "_image_based" if image_based else ""
        seq_output_path = output_dir / f"sequential_align{filename_suffix}.ply"
        save_point_cloud(all_world_points_seq, all_colors_seq, seq_output_path, voxel_size)
    
    # ========== Step 6: Evaluation metrics (if enabled) ==========
    if enable_eval and use_model_depth:
        print("\n" + "="*60)
        print("Step 6: Computing evaluation metrics")
        print("="*60)
        
        eval_results = {}
        
        # Evaluate per-frame aligned depths
        if per_frame_aligned_depths and per_frame_gt_depths:
            print("==> Computing metrics for per-frame aligned depths...")
            per_frame_aligned_depths_stack = np.stack(per_frame_aligned_depths, axis=0)
            per_frame_gt_depths_stack = np.stack(per_frame_gt_depths, axis=0)
            
            # Create valid mask
            valid_mask = np.logical_and(per_frame_gt_depths_stack > depth_min, 
                                       per_frame_gt_depths_stack < depth_max)
            valid_mask = np.logical_and(valid_mask, per_frame_aligned_depths_stack > 1e-3)
            
            n = valid_mask.sum((-1, -2))
            valid_frames = (n > 0)
            
            if valid_frames.sum() > 0:
                per_frame_aligned_torch = torch.from_numpy(per_frame_aligned_depths_stack[valid_frames]).to(device)
                per_frame_gt_torch = torch.from_numpy(per_frame_gt_depths_stack[valid_frames]).to(device)
                valid_mask_torch = torch.from_numpy(valid_mask[valid_frames]).to(device)
                
                per_frame_metrics = {}
                for metric_name in eval_metrics:
                    metric_func = getattr(metric, metric_name)
                    metric_value = metric_func(per_frame_aligned_torch, per_frame_gt_torch, valid_mask_torch).item()
                    per_frame_metrics[metric_name + '_per_frame_aligned_lstsq'] = float(metric_value)
                    print(f"  {metric_name}_per_frame_aligned_lstsq: {metric_value:.6f}")
                
                eval_results['per_frame_aligned_lstsq'] = per_frame_metrics
            else:
                print("Warning: No valid frames for per-frame aligned evaluation")
        
        # Evaluate sequence-aligned depths
        if seq_aligned_depths and seq_gt_depths:
            print("==> Computing metrics for sequence-aligned depths...")
            seq_aligned_depths_stack = np.stack(seq_aligned_depths, axis=0)
            seq_gt_depths_stack = np.stack(seq_gt_depths, axis=0)
            
            # Create valid mask
            valid_mask = np.logical_and(seq_gt_depths_stack > depth_min, 
                                       seq_gt_depths_stack < depth_max)
            valid_mask = np.logical_and(valid_mask, seq_aligned_depths_stack > 1e-3)
            
            n = valid_mask.sum((-1, -2))
            valid_frames = (n > 0)
            
            if valid_frames.sum() > 0:
                seq_aligned_torch = torch.from_numpy(seq_aligned_depths_stack[valid_frames]).to(device)
                seq_gt_torch = torch.from_numpy(seq_gt_depths_stack[valid_frames]).to(device)
                valid_mask_torch = torch.from_numpy(valid_mask[valid_frames]).to(device)
                
                seq_metrics = {}
                for metric_name in eval_metrics:
                    metric_func = getattr(metric, metric_name)
                    metric_value = metric_func(seq_aligned_torch, seq_gt_torch, valid_mask_torch).item()
                    seq_metrics[metric_name + '_lstsq'] = float(metric_value)
                    print(f"  {metric_name}_lstsq: {metric_value:.6f}")
                
                eval_results['lstsq'] = seq_metrics
            else:
                print("Warning: No valid frames for sequence-aligned evaluation")
        
        # Save evaluation results
        if eval_results:
            filename_suffix = "_image_based" if image_based else ""
            eval_output_path = output_dir / f"evaluation_results{filename_suffix}.json"
            with open(eval_output_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"\n==> Evaluation results saved to: {eval_output_path}")
    
    print("\n" + "="*60)
    print("All reconstructions completed!")
    print("="*60)


if __name__ == '__main__':
    main()
