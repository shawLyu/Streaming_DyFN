import os
import sys
from pathlib import Path
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

import cv2
import torch
import click
import mediapy
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
import json
import open3d as o3d  # 
import warnings

from moge.model.v1 import MoGeModel
# from moge.utils.io import save_ply # 我们将使用 open3d 来保存
from moge.utils.vis import colorize_depth, colorize_normal, colorize_depth_video
import utils3d

def compute_svd_raw(feature):
    """Compute top-3 singular vectors of feature map (not normalized)."""
    c, h, w = feature.shape
    feature_2d = feature.reshape(c, -1)

    uh, _, vh = np.linalg.svd(feature_2d, full_matrices=False)
    if uh[0, 0] < 0:
        vh[0] = -vh[0]
    if uh[0, 1] < 0:
        vh[1] = -vh[1]
    comps = [vh[0].reshape(h, w), vh[1].reshape(h, w), vh[2].reshape(h, w)]  # raw values
    return np.stack(comps, axis=-1)  # (H, W, 3)

def normalize_svd(svd_img, vmin, vmax):
    return (svd_img - vmin) / (vmax - vmin + 1e-8)


@click.command(help='Video Depth Inference, Alignment, and Stitching')
# --- MoGe模型相关的参数 (保留) ---
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default='Ruicheng/moge-vitl', help='Pretrained model name or path. Defaults to "Ruicheng/moge-vitl"')
@click.option('--output_dir', type=click.Path(), default='outputs_aligned', help='Directory to save output results')
@click.option('--save_video', is_flag=True, help='Save output as video (showing aligned depth)')
@click.option('--max_res', type=int, default=1024, help='Maximum resolution dimension')
@click.option('--depth_max', type=float, default=80, help='Maximum depth value for visualization')
@click.option('--resolution_level', type=int, default=9, help='Resolution level for inference [0-9].')
@click.option('--num_tokens', type=int, default=None, help='Number of tokens for inference. Overrides resolution_level.')
@click.option('--use_fp16', is_flag=True, help='Use fp16 precision for 2x faster inference.')
@click.option('--image_based', is_flag=True, help='Use image-based inference.')
@click.option('--ema_only', is_flag=True, help='Only use the EMA model for inference.')
@click.option('--threshold', type=float, default=0.03, help='Threshold for removing edges.')

# --- 新增：用于对齐和拼接的参数 ---
@click.option('--preprocessed_dir', type=click.Path(exists=True), required=True, 
              help='Path to the preprocessed data dir (containing points_cam/, pose/, video/)')
@click.option('--sequence', type=str, required=True, 
              help='Sequence name (must match subfolder in preprocessed_dir)')
@click.option('--video_path', type=click.Path(exists=True), default=None,
              help='Path to input video. If None, assumes <preprocessed_dir>/video/<sequence>.mp4')
@click.option('--start_frame', type=int, default=0, help='Start frame index from the preprocessed data')
@click.option('--num_frames', type=int, default=-1, help='Number of frames to process (-1 for all)')
@click.option('--voxel_size', type=float, default=0.02, help='Voxel size for down-sampling final stitched cloud')
@click.option('--save_ply_', is_flag=True, help='Save the final *stitched* point cloud.')
@click.option('--save_frames_', is_flag=True, help='Save the *aligned* per-frame information (points, depth, etc.)')

# --- 保留的可视化选项 ---
@click.option('--vis_normal', is_flag=True, help='Visualize the normal map.')
@click.option('--depth_show', is_flag=True, help='Show the depth map.')
@click.option('--align_method', type=click.Choice(['per_frame', 'global']), default='per_frame', 
              help='Alignment method: per_frame (lstsq each frame) or global (one lstsq for all frames).')

# ... [All your @click.option arguments remain the same] ...
def main(
    # Model args
    pretrained_model_name_or_path: str,
    output_dir: str,
    save_video: bool,
    max_res: int,
    depth_max: float,
    resolution_level: int,
    num_tokens: int,
    use_fp16: bool,
    image_based: bool,
    ema_only: bool,
    threshold: float,
    # New Alignment args
    preprocessed_dir: str,
    sequence: str,
    video_path: str,
    start_frame: int,
    num_frames: int,
    voxel_size: float,
    save_ply_: bool,
    save_frames_: bool,
    # Vis args
    vis_normal: bool,
    depth_show: bool,
    align_method: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Suppress warnings for median-of-empty-slice, etc.
    warnings.filterwarnings("ignore", category=RuntimeWarning) 

    assert not (image_based and ema_only), "Image-based inference and EMA-only inference cannot be used together."

    # --- 1. Set up paths ---
    prep_dir = Path(preprocessed_dir)
    seq_name = sequence
    points_cam_gt_dir = prep_dir / 'points_cam' / seq_name
    pose_gt_dir = prep_dir / 'pose' / seq_name
    
    if video_path is None:
        video_path_obj = prep_dir / 'video' / f"{seq_name}.mp4"
    else:
        video_path_obj = Path(video_path)

    if not all([points_cam_gt_dir.exists(), pose_gt_dir.exists(), video_path_obj.exists()]):
        print(f"Error: Missing required preprocessed data.")
        print(f"  Points GT Dir: {points_cam_gt_dir} (Exists: {points_cam_gt_dir.exists()})")
        print(f"  Pose GT Dir: {pose_gt_dir} (Exists: {pose_gt_dir.exists()})")
        print(f"  Video Path: {video_path_obj} (Exists: {video_path_obj.exists()})")
        return

    # --- 2. Determine frame list from GT data ---
    print(f"==> Scanning GT data in {pose_gt_dir}")
    # Find all frames that have *both* pose and points_cam files
    gt_pose_files = {f.stem for f in pose_gt_dir.glob("*.npy")}
    gt_points_files = {f.stem for f in points_cam_gt_dir.glob("*.npy")}
    
    basenames = sorted(list(gt_pose_files.intersection(gt_points_files)))
    
    if not basenames:
        print(f"Error: No matching 'pose' and 'points_cam' files found in {prep_dir} for sequence {seq_name}")
        return

    total_available = len(basenames)
    # start_frame and num_frames are list indices
    start_idx = max(0, start_frame) 
    
    if start_idx >= total_available:
        print(f"Error: start_frame {start_frame} is out of bounds (found {total_available} matching GT files).")
        return
        
    if num_frames == -1:
        end_idx = total_available
    else:
        end_idx = min(total_available, start_idx + num_frames)
        
    # This is the list of frame *names* (e.g., '0050', '0051')
    frames_to_process = basenames[start_idx:end_idx]
    # This is the list of *video indices* to read (e.g., 0, 1)
    frame_indices_to_read = list(range(start_idx, end_idx))
    
    if not frames_to_process:
        print("No frames to process in the specified range.")
        return
        
    print(f"==> Found {len(frames_to_process)} frames to process (from list index {start_idx} to {end_idx-1})")
    print(f"    (First frame: {frames_to_process[0]}, Last frame: {frames_to_process[-1]})")


    # --- 3. Read video frames (Corrected logic) ---
    print(f"==> Loading video frames from: {video_path_obj}")
    vid = VideoReader(str(video_path_obj), ctx=cpu(0))
    
    # Read the video frames corresponding to the GT list indices
    frames = vid.get_batch(frame_indices_to_read).asnumpy().astype("float32") / 255.0
    frames_np_colors = (frames * 255).astype(np.uint8)
    print(f"==> Video frames loaded, shape: {frames.shape}")

    # --- 4. Run MoGe model ---
    model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()

    with torch.no_grad():
        image_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(device)
        print(f"==> Running MoGe inference...")
        output = model.infer_video(image_tensor, fov_x=None, resolution_level=resolution_level,
                                   num_tokens=num_tokens, use_fp16=use_fp16, 
                                   image_based=image_based, ema_only=ema_only)

        points_est_all = output['points'].cpu().numpy()
        depth_est_all = output['depth'].cpu().numpy()
        mask_est_all = output['mask'].cpu().numpy()
        intrinsics_est_all = output['intrinsics'].cpu().numpy()
    print(f"==> MoGe inference complete.")

    # --- 5. Align, stitch, and save ---
    all_world_points = []
    all_world_colors = []
    aligned_depth_video = [] # For saving video

    frames_output_dir = Path(output_dir, "frames")
    save_path = frames_output_dir / seq_name
    if save_frames_:
        save_path.mkdir(exist_ok=True, parents=True)
        print(f"==> Saving aligned per-frame data to {save_path}")

    # --- MODIFIED: Branch based on alignment method ---
    
    if align_method == 'global':
        print("==> Starting GLOBAL alignment: Pass 1/2 (Collecting data...)")
        
        all_pred_depths_masked = []
        all_gt_depths_masked = []
        gt_poses_to_process = [] # We need to store GT poses for the second pass
        
        # --- Pass 1: Collect all valid (pred, gt) depth pairs ---
        for i, frame_name in enumerate(tqdm(frames_to_process, desc="Global Align: Pass 1/2")):
            d_est = depth_est_all[i]
            
            try:
                pts_cam_gt = np.load(points_cam_gt_dir / f"{frame_name}.npy")
                pose_gt = np.load(pose_gt_dir / f"{frame_name}.npy")
                gt_poses_to_process.append(pose_gt) # Store this for later
            except FileNotFoundError:
                print(f"Warning: Missing GT data for frame {frame_name}. Skipping.")
                gt_poses_to_process.append(None) # Add a placeholder
                continue
                
            d_gt = pts_cam_gt[..., 2] # Z-depth
            # Alignment mask: valid in both GT and prediction
            valid_mask_i = (d_gt > 1e-3) & (d_gt < depth_max) & (d_est > 1e-3)
            
            if np.sum(valid_mask_i) > 100:
                all_pred_depths_masked.append(d_est[valid_mask_i].reshape(-1, 1))
                all_gt_depths_masked.append(d_gt[valid_mask_i].reshape(-1, 1))

        if not all_pred_depths_masked:
            print("Error: No valid overlapping data found for global alignment. Aborting.")
            return

        # --- Solve for GLOBAL scale and shift ---
        print("==> Solving for global scale and shift...")
        all_preds = np.concatenate(all_pred_depths_masked)
        all_gts = np.concatenate(all_gt_depths_masked)
        
        _ones = np.ones_like(all_preds)
        A = np.concatenate([all_preds, _ones], axis=-1)
        
        X = np.linalg.lstsq(A, all_gts, rcond=None)[0]
        global_scale, global_shift = X.flatten()
        print(f"==> Global Alignment Found: scale = {global_scale:.4f}, shift = {global_shift:.4f}")

        # --- Pass 2: Apply global transform and stitch ---
        print("==> Starting GLOBAL alignment: Pass 2/2 (Applying and stitching...)")
        for i, frame_name in enumerate(tqdm(frames_to_process, desc="Global Align: Pass 2/2")):
            pose_gt = gt_poses_to_process[i]
            if pose_gt is None: # Skip frames that failed in pass 1
                aligned_depth_video.append(np.zeros_like(depth_est_all[i]))
                continue
            try:
                pts_cam_gt = np.load(points_cam_gt_dir / f"{frame_name}.npy")
            except FileNotFoundError:
                print(f"Warning: Missing GT data for frame {frame_name}. Skipping.")
                continue
                
            d_est = depth_est_all[i]
            pts_cam_est = points_est_all[i]
            m_est = mask_est_all[i]
            color = frames_np_colors[i]
            d_gt = pts_cam_gt[..., 2]

            # Apply GLOBAL scale and shift
            depth_aligned = global_scale * d_est + global_shift
            depth_aligned = np.clip(depth_aligned, a_min=1e-3, a_max=depth_max)

            # Scale points along their rays
            d_est_safe = np.where(d_est < 1e-6, 1e-6, d_est)
            ray_scaler = depth_aligned / d_est_safe
            pts_cam_aligned = pts_cam_est * ray_scaler[..., np.newaxis]
            
            # --- Transform, Filter, Collect (same as before) ---
            h, w, _ = pts_cam_aligned.shape
            pts_flat = pts_cam_aligned.reshape(-1, 3)
            pts_hom = np.concatenate([pts_flat, np.ones((h*w, 1))], axis=1)
            pts_world_flat = (pose_gt @ pts_hom.T).T[:, :3]
            
            colors_flat = color.reshape(-1, 3)
            # Filter by MoGe mask AND new depth_max limit
            final_mask = m_est & (d_gt <= depth_max) & (d_gt > 1e-3)
            final_mask_flat = final_mask.flatten()
            
            all_world_points.append(pts_world_flat[final_mask_flat])
            all_world_colors.append(colors_flat[final_mask_flat])
            aligned_depth_video.append(depth_aligned)
            
            if save_frames_:
                cv2.imwrite(str(save_path / f'image_{frame_name}.jpg'), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_path / f'depth_aligned_{frame_name}.exr'), depth_aligned, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                cv2.imwrite(str(save_path / f'points_aligned_cam_{frame_name}.exr'), cv2.cvtColor(pts_cam_aligned.astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

    
    elif align_method == 'per_frame':
        print("==> Starting PER-FRAME alignment...")
        # This is the logic from the previous script
        for i, frame_name in enumerate(tqdm(frames_to_process, desc="Aligning and stitching frames")):
            # Get estimated results for this frame
            pts_cam_est = points_est_all[i]
            d_est = depth_est_all[i]
            m_est = mask_est_all[i] # MoGe's validity mask
            color = frames_np_colors[i]
            
            # Load GT data for this frame
            try:
                pts_cam_gt = np.load(points_cam_gt_dir / f"{frame_name}.npy")
                pose_gt = np.load(pose_gt_dir / f"{frame_name}.npy")
            except FileNotFoundError:
                print(f"Warning: Missing GT data for frame {frame_name}. Skipping.")
                continue
                
            # Get GT depth
            d_gt = pts_cam_gt[..., 2] # Z-depth in camera space
            
            # --- PER-FRAME LSTSQ ALIGNMENT ---
            # 1. Create a valid mask for alignment
            valid_mask_i = (d_gt > 1e-3) & (d_gt < depth_max) & (d_est > 1e-3)
            
            if np.sum(valid_mask_i) < 100: # Not enough points to align
                print(f"Warning: Insufficient valid points for lstsq on frame {frame_name}. Skipping.")
                aligned_depth_video.append(np.zeros_like(d_est)) # Add blank frame
                continue

            # 2. Prepare data for lstsq
            pred_depth_i_masked = d_est[valid_mask_i].reshape(-1, 1)
            gt_depth_i_masked = d_gt[valid_mask_i].reshape(-1, 1)
            
            _ones = np.ones_like(pred_depth_i_masked)
            A = np.concatenate([pred_depth_i_masked, _ones], axis=-1)
            
            # 3. Solve for scale and shift
            try:
                X = np.linalg.lstsq(A, gt_depth_i_masked, rcond=None)[0]
                scale, shift = X.flatten() # gt = scale * pred + shift
            except np.linalg.LinAlgError:
                print(f"Warning: SVD did not converge for frame {frame_name}. Skipping.")
                aligned_depth_video.append(np.zeros_like(d_est))
                continue
                
            if not np.isfinite(scale) or not np.isfinite(shift):
                print(f"Warning: Invalid scale/shift ({scale}, {shift}) for frame {frame_name}. Skipping.")
                aligned_depth_video.append(np.zeros_like(d_est))
                continue
                
            # 4. Apply scale and shift to the full depth map
            depth_aligned = scale * d_est + shift
            
            # 5. Clip aligned depth
            depth_aligned = np.clip(depth_aligned, a_min=1e-3, a_max=depth_max)
            
            # 6. Apply this aligned depth back to the 3D points
            d_est_safe = np.where(d_est < 1e-6, 1e-6, d_est)
            ray_scaler = depth_aligned / d_est_safe
            pts_cam_aligned = pts_cam_est * ray_scaler[..., np.newaxis] 
            
            # --- Transform to world coordinates ---
            h, w, _ = pts_cam_aligned.shape
            pts_flat = pts_cam_aligned.reshape(-1, 3)
            pts_hom = np.concatenate([pts_flat, np.ones((h*w, 1))], axis=1)
            pts_world_flat = (pose_gt @ pts_hom.T).T[:, :3]
            
            # --- Filter and collect points for stitching ---
            colors_flat = color.reshape(-1, 3)
            
            # Filter by MoGe mask AND new depth_max limit
            final_mask = m_est & (depth_aligned <= depth_max) & (depth_aligned > 1e-3)
            final_mask_flat = final_mask.flatten()
            
            valid_world_pts = pts_world_flat[final_mask_flat]
            valid_world_colors = colors_flat[final_mask_flat]
            
            all_world_points.append(valid_world_pts)
            all_world_colors.append(valid_world_colors)
            
            # --- Save per-frame data (if requested) ---
            aligned_depth_video.append(depth_aligned) # For video
            
            if save_frames_:
                cv2.imwrite(str(save_path / f'image_{frame_name}.jpg'), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_path / f'depth_aligned_{frame_name}.exr'), depth_aligned, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                cv2.imwrite(str(save_path / f'points_aligned_cam_{frame_name}.exr'), cv2.cvtColor(pts_cam_aligned.astype(np.float32), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
    
    # --- 6. Create and save final stitched point cloud ---
    if save_ply_ and all_world_points:
        print(f"==> Concatenating {len(all_world_points)} point clouds...")
        combined_points = np.concatenate(all_world_points, axis=0)
        combined_colors = np.concatenate(all_world_colors, axis=0)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        pcd.colors = o3d.utility.Vector3dVector(combined_colors / 255.0) # O3D expects 0-1
        
        print(f"==> Downsampling final cloud with voxel size {voxel_size}...")
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Add alignment method to output file name
        if image_based:
            output_ply_path = Path(output_dir) / f"{seq_name}_{align_method}_aligned_stitched_vox{voxel_size}_image_based.ply"
        else:
            output_ply_path = Path(output_dir) / f"{seq_name}_{align_method}_aligned_stitched_vox{voxel_size}.ply"
        o3d.io.write_point_cloud(str(output_ply_path), pcd_down)
        print(f"==> Successfully saved stitched point cloud to {output_ply_path}")
        print(f"    Total points before downsampling: {len(combined_points)}")
        print(f"    Total points after downsampling: {len(pcd_down.points)}")
        
    # --- 7. Save visualization video (using aligned depth) ---
    if save_video:
        print(f"==> Generating aligned depth video...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Prepare aligned depth visualization
        disp_preds = np.stack(aligned_depth_video, axis=0)
        disp_preds = np.where(disp_preds > 0, disp_preds, np.nan) # Mask out zeros
        
        # Calculate visualization range
        # Use a safe percentile to avoid extremes, especially if shift is large
        vis_min, vis_max = np.nanquantile(disp_preds[disp_preds <= depth_max], [0.02, 0.98])

        if depth_show:
            min_disp, max_disp = vis_min, vis_max
        else:
            # Invert for disparity visualization
            disp_preds_inv = 1.0 / (disp_preds + 1e-8)
            min_disp, max_disp = np.nanquantile(disp_preds_inv, [0.02, 0.98])
            disp_preds = disp_preds_inv # Use the inverted one for coloring
            
        disp_preds = np.clip(disp_preds, min_disp, max_disp)
        depth_preds_color = colorize_depth_video(disp_preds, min_disp=min_disp, max_disp=max_disp)
        depth_preds_color = np.stack(depth_preds_color, axis=0)

        # 1x2 grid: [frame | aligned_depth]
        grid_video = np.concatenate([frames_np_colors, depth_preds_color], axis=2)
        
        # Add alignment method to output file name
        if image_based:
            output_path = Path(output_dir) / f'{seq_name}_{align_method}_aligned_depth_image_based.mp4'
        else:
            output_path = Path(output_dir) / f'{seq_name}_{align_method}_aligned_depth.mp4'
        # Use the original video's average FPS
        fps = vid.get_avg_fps()
        mediapy.write_video(str(output_path), grid_video, fps=fps, crf=18)
        print(f"==> Saved aligned video visualization to {output_path} (fps={fps})")

if __name__ == '__main__':
    main()