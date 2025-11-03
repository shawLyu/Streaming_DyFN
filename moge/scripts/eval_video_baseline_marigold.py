import os
import sys
from pathlib import Path
from typing import *
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

# Add Marigold to path
# Try multiple possible paths for Marigold
_script_file = Path(__file__).absolute()
_possible_marigold_paths = []

# Try environment variable first
if os.environ.get("MARIGOLD_PATH"):
    _possible_marigold_paths.append(Path(os.environ.get("MARIGOLD_PATH")))

# Try relative to script location
_possible_marigold_paths.append(_script_file.parents[3] / "Marigold")  # From script: AR_depth/Marigold

# Try common absolute paths
_possible_marigold_paths.append(Path("./Marigold"))

# Try from workspace root (if script is in v_MoGe/moge/scripts)
if len(_script_file.parts) >= 5:
    _possible_marigold_paths.append(_script_file.parents[4] / "Marigold")

# Filter out invalid paths
_possible_marigold_paths = [p for p in _possible_marigold_paths if p and Path(p).exists()]

marigold_root = None
for marigold_path in _possible_marigold_paths:
    marigold_path = Path(marigold_path).absolute()
    # Check if it's the Marigold root (contains marigold package directory)
    if (marigold_path / "marigold" / "__init__.py").exists():
        marigold_root = marigold_path
        break

if marigold_root is None:
    raise ImportError(
        f"Could not find Marigold directory. Tried paths: {_possible_marigold_paths}. "
        f"Current script location: {_script_file}. "
        f"Please ensure Marigold is installed or set MARIGOLD_PATH environment variable."
    )

marigold_root = str(marigold_root)
if marigold_root not in sys.path:
    sys.path.insert(0, marigold_root)

import glob
import json
import click
import torch
import mediapy
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
import torch.nn.functional as F
from PIL import Image

# Import Marigold
from marigold import MarigoldDepthPipeline

# Import MoGe utils for metrics and visualization
from moge.utils.vis import colorize_depth_video
from moge.utils.alignment import align_depth_affine
from moge.utils.geometry_torch import mask_aware_nearest_resize
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

def read_video_frames(video_path, target_fps, max_res, silent=False):
    vid = VideoReader(video_path, ctx=cpu(0))
    if not silent:
        print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
    
    original_height, original_width = vid.get_batch([0]).shape[1:3]
    height = round(original_height / 64) * 64
    width = round(original_width / 64) * 64
    
    if max(height, width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale / 64) * 64
        width = round(original_width * scale / 64) * 64

    vid = VideoReader(video_path, ctx=cpu(0), width=original_width, height=original_height)

    fps = vid.get_avg_fps() if target_fps == -1 else target_fps
    stride = round(vid.get_avg_fps() / fps)
    stride = max(stride, 1)
    frames_idx = list(range(0, len(vid), stride))
    if not silent:
        print(f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}")
    
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0
    return frames, fps


def disparity_to_depth(disparity):
    """Convert disparity to depth. Assumes disparity > 0."""
    depth = np.zeros_like(disparity)
    valid_mask = disparity > 1e-6  # Avoid division by zero
    depth[valid_mask] = 1.0 / disparity[valid_mask]
    return depth


@click.command(help='Evaluate video results using Marigold')
@click.option('--eval_dataset_list', multiple=True, default=None, help='List of datasets to evaluate. Can be specified multiple times, e.g., --eval_dataset_list sintel --eval_dataset_list scannet')
@click.option('--video_dir_path', type=click.Path(exists=True), required=True, help='Path to evaluated video file')
@click.option('--checkpoint', type=str, default='./Marigold/prs-eth/marigold-depth-v1-1', help='Marigold checkpoint path or hub name. Defaults to local path.')
@click.option('--output_dir', type=click.Path(), default='outputs_video_marigold', help='Directory to save output results')
@click.option('--align_method', type=str, default='all', help='Method to align depth. Defaults to "all". Options: "lstsq", "searching", "all".')
@click.option('--save_video', is_flag=True, help='Save output as video')
@click.option('--target_fps', type=int, default=15, help='Target frames per second for video processing')
@click.option('--max_res', type=int, default=1024, help='Maximum resolution dimension')
@click.option('--depth_max', type=float, default=80, help='Maximum depth value for visualization')
@click.option('--denoise_steps', type=int, default=10, help='Diffusion denoising steps for Marigold')
@click.option('--ensemble_size', type=int, default=1, help='Number of predictions to be ensembled for Marigold')
@click.option('--processing_res', type=int, default=768, help='Resolution to which the input is resized before performing estimation. 0 uses original resolution')
@click.option('--use_fp16', is_flag=True, help='Use fp16 precision for faster inference')
@click.option('--silent', is_flag=True, help='Do not print any information.')
def main(
    eval_dataset_list: tuple,
    video_dir_path: str,
    checkpoint: str,
    output_dir: str,
    align_method: str,
    save_video: bool,
    target_fps: int,
    max_res: int,
    depth_max: float,
    denoise_steps: int,
    ensemble_size: int,
    processing_res: int,
    use_fp16: bool,
    silent: bool,
):
    # Handle eval_dataset_list: convert tuple to list and use default if empty
    if eval_dataset_list is None or len(eval_dataset_list) == 0:
        eval_dataset_list = ['sintel', 'scannet', 'KITTI', 'bonn']
    else:
        eval_dataset_list = list(eval_dataset_list)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load Marigold model
    if use_fp16:
        dtype = torch.float16
        variant = "fp16"
    else:
        dtype = torch.float32
        variant = None
    
    if not silent:
        print(f"Loading Marigold model: {checkpoint}")
    pipe = MarigoldDepthPipeline.from_pretrained(
        checkpoint, variant=variant, torch_dtype=dtype
    )
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        if not silent:
            print("Proceeding without xformers")
    
    pipe = pipe.to(device)
    if not silent:
        print(f"Loaded Marigold depth pipeline: scale_invariant={pipe.scale_invariant}, shift_invariant={pipe.shift_invariant}")
    
    depth_max = {"sintel": 70, "scannet": 10, "KITTI": 80, "bonn": 10, "NYUv2": 10}
    
    # Initialize results storage
    all_results = {}
    
    for dataset in eval_dataset_list:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*50}")
        
        dataset_path = Path(video_dir_path) / dataset
        if not dataset_path.exists():
            print(f"Warning: Dataset path {dataset_path} does not exist, skipping...")
            continue

        video_path_list = glob.glob(str(dataset_path / "*.mp4"))
        gt_depth_path_list = glob.glob(str(dataset_path / "*.npz"))

        print(f"==> Found {len(video_path_list)} videos and {len(gt_depth_path_list)} gt depths for {dataset}")

        if len(video_path_list) == 0 or len(gt_depth_path_list) == 0:
            print(f"Warning: No videos or ground truth depths found for {dataset}, skipping...")
            continue

        # Initialize metrics storage for this dataset
        dataset_metrics = {}
        if align_method == 'lstsq' or align_method == 'all':
            dataset_metrics.update({metric_name + '_lstsq': [] for metric_name in eval_metrics})
            dataset_metrics.update({metric_name + '_per_frame_aligned_lstsq': [] for metric_name in eval_metrics})
        if align_method == 'searching' or align_method == 'all':
            dataset_metrics.update({metric_name + '_searching': [] for metric_name in eval_metrics})
            dataset_metrics.update({metric_name + '_per_frame_aligned_searching': [] for metric_name in eval_metrics})
        dataset_metrics['video_names'] = []
        
        for video_path, gt_depth_path in zip(video_path_list, gt_depth_path_list):
            video_name = Path(video_path).stem
            if not silent:
                print(f"\n==> Processing {video_name} and {Path(gt_depth_path).stem}")
            if video_name == "mountain_1_rgb_left":
                depth_max["sintel"] = 1000
            else:
                depth_max["sintel"] = 70
            
            try:
                frames, fps = read_video_frames(video_path, target_fps, max_res, silent=silent)
                height, width = frames.shape[1:3]

                # Load GT depth (may be disparity or depth)
                # Note: The key 'disparity' might actually contain depth values in some datasets
                # We check if values look like disparity (typically < 1) or depth (typically > 1)
                gt_data = np.load(gt_depth_path)
                if 'disparity' in gt_data.files:
                    gt_raw = gt_data['disparity']  # (t, 1, h, w) or (t, h, w)
                    # Check if values are disparity (small, typically < 1) or depth (larger, typically > 1)
                    # If most valid values are < 1, assume it's disparity and convert to depth
                    gt_flat = gt_raw.flatten()
                    valid_values = gt_flat[gt_flat > 1e-6]
                    if len(valid_values) > 0 and np.median(valid_values) < 1.0:
                        # Looks like disparity, convert to depth: depth = 1 / disparity
                        gt_depth = disparity_to_depth(gt_raw)
                    else:
                        # Already depth values
                        gt_depth = gt_raw
                else:
                    gt_depth = gt_data['arr_0']  # Assume it's already depth
                
                # Ensure gt_depth has shape (t, 1, h, w) or (t, h, w)
                if gt_depth.ndim == 3:
                    gt_depth = gt_depth[:, None, :, :]  # (t, h, w) -> (t, 1, h, w)
                elif gt_depth.ndim == 2:
                    gt_depth = gt_depth[None, None, :, :]  # (h, w) -> (1, 1, h, w)
                
                gt_depth = np.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)

                with torch.no_grad():
                    dataset_metrics['video_names'].append(video_name)
                    
                    # Process each frame with Marigold
                    depth_preds = []
                    if not silent:
                        frame_iter = tqdm(enumerate(frames), desc=f"Processing {video_name}", total=len(frames))
                    else:
                        frame_iter = enumerate(frames)
                    
                    for frame_idx, frame in frame_iter:
                        # Convert frame to PIL Image
                        frame_uint8 = (frame * 255).astype(np.uint8)
                        input_image = Image.fromarray(frame_uint8)
                        
                        # Run Marigold inference
                        pipe_out = pipe(
                            input_image,
                            denoising_steps=denoise_steps,
                            ensemble_size=ensemble_size,
                            processing_res=processing_res,
                            match_input_res=True,  # Output at input resolution
                            batch_size=0,
                            color_map=None,
                            show_progress_bar=False,
                            resample_method="bilinear",
                            generator=None,
                        )
                        
                        # Get depth prediction (shape: [H, W])
                        depth_pred_frame = pipe_out.depth_np
                        depth_preds.append(depth_pred_frame)
                    
                    # Stack depth predictions: (T, H, W) -> (T, 1, H, W)
                    depth = np.stack(depth_preds, axis=0)  # (T, H, W)
                    depth = depth[:, None, :, :]  # (T, 1, H, W)
                    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Resize depth to match GT if needed
                    if depth.shape[-2:] != gt_depth.shape[-2:]:
                        depth_tensor = torch.from_numpy(depth).to(device)  # (T, 1, H, W)
                        depth_tensor = F.interpolate(
                            depth_tensor, 
                            size=(gt_depth.shape[-2], gt_depth.shape[-1]), 
                            mode='bilinear', 
                            align_corners=False
                        )  # (T, 1, H_gt, W_gt)
                        depth = depth_tensor.cpu().numpy()
                    
                    valid_mask = np.logical_and(gt_depth > 1e-3, gt_depth < depth_max[dataset])

                    if align_method == 'lstsq' or align_method == 'all':
                        # Evaluate metric for per frame alignment for lstsq
                        scales_per_frame, shifts_per_frame = [], []
                        per_frame_aligned_pred_depth = []
                        for pred_depth_i, gt_depth_i in zip(depth, gt_depth):
                            valid_mask_i = np.logical_and(gt_depth_i > 1e-3, gt_depth_i < depth_max[dataset])
                            pred_depth_i_masked = pred_depth_i[valid_mask_i].reshape(-1, 1)
                            gt_depth_i_masked = gt_depth_i[valid_mask_i].reshape(-1, 1)
                            _ones = np.ones_like(pred_depth_i_masked)
                            A = np.concatenate([pred_depth_i_masked, _ones], axis=-1)
                            X = np.linalg.lstsq(A, gt_depth_i_masked, rcond=None)[0]
                            scale, shift = X # gt = scale * pred + shift
                            scales_per_frame.append(scale)
                            shifts_per_frame.append(shift)
                            aligned_pred_depth_i = scale * pred_depth_i + shift
                            aligned_pred_depth_i = np.clip(aligned_pred_depth_i, a_min=1e-3, a_max=depth_max[dataset])
                            per_frame_aligned_pred_depth.append(aligned_pred_depth_i)

                        per_frame_aligned_pred_depth = np.stack(per_frame_aligned_pred_depth, axis=0)
                        per_frame_aligned_pred_depth = torch.from_numpy(per_frame_aligned_pred_depth).to(device)

                        # Depth alignment
                        gt_depth_masked = gt_depth[valid_mask].reshape((-1, 1)).astype(np.float64)
                        pred_depth_masked = depth[valid_mask].reshape((-1, 1)).astype(np.float64)
                        _ones = np.ones_like(pred_depth_masked)
                        A = np.concatenate([pred_depth_masked, _ones], axis=-1)
                        X = np.linalg.lstsq(A, gt_depth_masked, rcond=None)[0]
                        scale_global, shift_global = X # gt = scale * pred + shift
                        aligned_pred_depth = scale_global * depth + shift_global
                        aligned_pred_depth = np.clip(aligned_pred_depth, a_min=1e-3, a_max=depth_max[dataset]) 
                        aligned_pred_depth = torch.from_numpy(aligned_pred_depth).to(device)

                        n = valid_mask.sum((-1, -2))
                        valid_frame = (n > 0)

                        gt_depth_torch = torch.from_numpy(gt_depth).to(device)
                        valid_mask_torch = torch.from_numpy(valid_mask).to(device)
                        aligned_pred_depth_valid_torch = aligned_pred_depth[valid_frame]
                        gt_depth_valid_torch = gt_depth_torch[valid_frame]
                        valid_mask_valid_torch = valid_mask_torch[valid_frame]
                        per_frame_aligned_pred_depth_valid_torch = per_frame_aligned_pred_depth[valid_frame]

                        # evaluate metric 
                        sample_metric_depth = []
                        metric_funcs_depth = [getattr(metric, _met) for _met in eval_metrics]
                        for met_func in metric_funcs_depth:
                            _metric_name = met_func.__name__
                            _metric = met_func(aligned_pred_depth_valid_torch, gt_depth_valid_torch, valid_mask_valid_torch).item()
                            sample_metric_depth.append(_metric)
                            dataset_metrics[_metric_name + '_lstsq'].append(_metric)
                        
                        
                        if not silent:
                            print(f"==> Depth metrics for {video_name}:")
                            for metric_name, metric_value in zip(eval_metrics, sample_metric_depth):
                                print(f"  {metric_name}_lstsq: {metric_value:.6f}")

                        # evaluate metric for per frame alignment
                        sample_metric_depth_per_frame = []
                        metric_funcs_depth_per_frame = [getattr(metric, _met) for _met in eval_metrics]
                        for met_func in metric_funcs_depth_per_frame:
                            _metric_name = met_func.__name__
                            _metric = met_func(per_frame_aligned_pred_depth_valid_torch, gt_depth_valid_torch, valid_mask_valid_torch).item()
                            sample_metric_depth_per_frame.append(_metric)
                            dataset_metrics[_metric_name + '_per_frame_aligned_lstsq'].append(_metric)
                        
                        if not silent:
                            print(f"==> Depth metrics for {video_name} aligned by per frame:")
                            for metric_name, metric_value in zip(eval_metrics, sample_metric_depth_per_frame):
                                print(f"  {metric_name}_per_frame_aligned_lstsq: {metric_value:.6f}")

                    if align_method == 'searching' or align_method == 'all':
                        # Per frame aligned using mask_aware_nearest_resize
                        per_frame_aligned_pred_depth_searching = []
                        for pred_depth_i, gt_depth_i in zip(depth, gt_depth):
                            valid_mask_i = np.logical_and(gt_depth_i > 1e-3, gt_depth_i < depth_max[dataset])
                            
                            depth_mask = torch.from_numpy(valid_mask_i).squeeze().to(device)
                            
                            _, lr_mask, lr_index = mask_aware_nearest_resize(None, depth_mask, (64, 64), return_index=True)
                            
                            gt_depth_torch = torch.from_numpy(gt_depth_i).squeeze().to(device)
                            pred_depth_torch = torch.from_numpy(pred_depth_i).squeeze().to(device)
                            
                            pred_depth_lr_masked, gt_depth_lr_masked = pred_depth_torch[lr_index][lr_mask], gt_depth_torch[lr_index][lr_mask]
                            try:
                                scale_searching, shift_searching = align_depth_affine(pred_depth_lr_masked, gt_depth_lr_masked, 1 / gt_depth_lr_masked)
                            except Exception as e:
                                if not silent:
                                    print(f"Warning: Error in searching alignment: {e}")
                                # Fallback to identity transformation
                                scale_searching = torch.tensor(1.0, device=device)
                                shift_searching = torch.tensor(0.0, device=device)
                            
                            # Apply alignment to full depth map
                            aligned_pred_depth_i = scale_searching.item() * pred_depth_i + shift_searching.item()
                            aligned_pred_depth_i = np.clip(aligned_pred_depth_i, a_min=1e-3, a_max=depth_max[dataset])
                            per_frame_aligned_pred_depth_searching.append(aligned_pred_depth_i)
                        
                        per_frame_aligned_pred_depth_searching = np.stack(per_frame_aligned_pred_depth_searching, axis=0)
                        per_frame_aligned_pred_depth_searching = torch.from_numpy(per_frame_aligned_pred_depth_searching).to(device)

                        # Global aligned using mask_aware_nearest_resize
                        pred_depth_lr_list = []
                        gt_depth_lr_list = []
                        for pred_depth_i, gt_depth_i in zip(depth, gt_depth):
                            valid_mask_i = np.logical_and(gt_depth_i > 1e-3, gt_depth_i < depth_max[dataset])
                            
                            depth_mask = torch.from_numpy(valid_mask_i).squeeze().to(device)
                            
                            _, lr_mask, lr_index = mask_aware_nearest_resize(None, depth_mask, (64, 64), return_index=True)
                            
                            gt_depth_torch = torch.from_numpy(gt_depth_i).squeeze().to(device)
                            pred_depth_torch = torch.from_numpy(pred_depth_i).squeeze().to(device)
                            
                            pred_depth_lr_masked, gt_depth_lr_masked = pred_depth_torch[lr_index][lr_mask], gt_depth_torch[lr_index][lr_mask]
                            pred_depth_lr_list.append(pred_depth_lr_masked)
                            gt_depth_lr_list.append(gt_depth_lr_masked)
                        pred_depth_lr_all = torch.cat(pred_depth_lr_list, dim=0)
                        gt_depth_lr_all = torch.cat(gt_depth_lr_list, dim=0)

                        try:
                            scale_all, shift_all = align_depth_affine(pred_depth_lr_all[::gt_depth.shape[0]], gt_depth_lr_all[::gt_depth.shape[0]], 1 / gt_depth_lr_all[::gt_depth.shape[0]])
                        except Exception as e:
                            if not silent:
                                print(f"Warning: Error in global searching alignment: {e}")
                            # Fallback to identity transformation
                            scale_all = torch.tensor(1.0, device=device)
                            shift_all = torch.tensor(0.0, device=device)
                        
                        # Apply global alignment to full depth maps
                        aligned_pred_depth_searching = scale_all.item() * depth + shift_all.item()
                        aligned_pred_depth_searching = np.clip(aligned_pred_depth_searching, a_min=1e-3, a_max=depth_max[dataset]) 
                        aligned_pred_depth_searching = torch.from_numpy(aligned_pred_depth_searching).to(device)

                        n = valid_mask.sum((-1, -2))
                        valid_frame = (n > 0)

                        gt_depth_torch = torch.from_numpy(gt_depth).to(device)
                        valid_mask_torch = torch.from_numpy(valid_mask).to(device)
                        aligned_pred_depth_valid_torch = aligned_pred_depth_searching[valid_frame]
                        gt_depth_valid_torch = gt_depth_torch[valid_frame]
                        valid_mask_valid_torch = valid_mask_torch[valid_frame]
                        per_frame_aligned_pred_depth_valid_torch = per_frame_aligned_pred_depth_searching[valid_frame]

                        # evaluate metric for global searching alignment
                        sample_metric_depth_searching = []
                        metric_funcs_depth_searching = [getattr(metric, _met) for _met in eval_metrics]
                        for met_func in metric_funcs_depth_searching:
                            _metric_name = met_func.__name__
                            _metric = met_func(aligned_pred_depth_valid_torch, gt_depth_valid_torch, valid_mask_valid_torch).item()
                            sample_metric_depth_searching.append(_metric)
                            dataset_metrics[_metric_name + '_searching'].append(_metric)
                        
                        if not silent:
                            print(f"==> Depth metrics for {video_name} (searching):")
                            for metric_name, metric_value in zip(eval_metrics, sample_metric_depth_searching):
                                print(f"  {metric_name}_searching: {metric_value:.6f}")

                        # evaluate metric for per frame searching alignment
                        sample_metric_depth_per_frame_searching = []
                        metric_funcs_depth_per_frame_searching = [getattr(metric, _met) for _met in eval_metrics]
                        for met_func in metric_funcs_depth_per_frame_searching:
                            _metric_name = met_func.__name__
                            _metric = met_func(per_frame_aligned_pred_depth_valid_torch, gt_depth_valid_torch, valid_mask_valid_torch).item()
                            sample_metric_depth_per_frame_searching.append(_metric)
                            dataset_metrics[_metric_name + '_per_frame_aligned_searching'].append(_metric)
                        
                        if not silent:
                            print(f"==> Depth metrics for {video_name} aligned by per frame (searching):")
                            for metric_name, metric_value in zip(eval_metrics, sample_metric_depth_per_frame_searching):
                                print(f"  {metric_name}_per_frame_aligned_searching: {metric_value:.6f}")
                    

                    if save_video:
                        Path(output_dir).mkdir(parents=True, exist_ok=True)
                        # Create depth visualization for video saving
                        depth_vis = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                        min_depth, max_depth = np.nanquantile(depth_vis[depth_vis > 0], 0.01), np.nanquantile(depth_vis[depth_vis > 0], 0.99)
                        depth_preds_color = colorize_depth_video(depth_vis, min_depth=min_depth, max_depth=max_depth)
                        
                        # Save the visualized results
                        depth_preds_color = np.stack(depth_preds_color, axis=0)
                        frames_np = (frames * 255).astype(np.uint8)
                        combined_video = np.concatenate([frames_np, depth_preds_color], axis=2)
                        output_path = os.path.join(output_dir, f'{video_name}_depth.mp4')
                        mediapy.write_video(output_path, combined_video, fps=10, crf=18)
                        print(f"==> Saved video: {output_path}")
                        
            except Exception as e:
                print(f"Error processing {video_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Calculate average metrics for this dataset
        if dataset_metrics['video_names']:
            print(f"\n{'='*30}")
            print(f"Dataset: {dataset} - Average Metrics")
            print(f"{'='*30}")
            
            dataset_summary = {}
            for metric_name in eval_metrics:
                # Add lstsq metrics
                if align_method == 'lstsq' or align_method == 'all':
                    if dataset_metrics[metric_name + '_lstsq']:
                        avg_metric_lstsq = np.mean(dataset_metrics[metric_name + '_lstsq'])
                        std_metric_lstsq = np.std(dataset_metrics[metric_name + '_lstsq'])
                        dataset_summary[metric_name + '_lstsq'] = {
                            'mean': float(avg_metric_lstsq),
                            'std': float(std_metric_lstsq)
                        }
                        avg_metric_per_frame_lstsq = np.mean(dataset_metrics[metric_name + '_per_frame_aligned_lstsq'])
                        std_metric_per_frame_lstsq = np.std(dataset_metrics[metric_name + '_per_frame_aligned_lstsq'])
                        dataset_summary[metric_name + '_per_frame_aligned_lstsq'] = {
                            'mean': float(avg_metric_per_frame_lstsq),
                            'std': float(std_metric_per_frame_lstsq)
                        }
                        print(f"{metric_name}_lstsq: {avg_metric_lstsq:.6f} ± {std_metric_lstsq:.6f}")
                        print(f"{metric_name}_per_frame_aligned_lstsq: {avg_metric_per_frame_lstsq:.6f} ± {std_metric_per_frame_lstsq:.6f}")
                
                # Add searching metrics
                if align_method == 'searching' or align_method == 'all':
                    if dataset_metrics[metric_name + '_searching']:
                        avg_metric_searching = np.mean(dataset_metrics[metric_name + '_searching'])
                        std_metric_searching = np.std(dataset_metrics[metric_name + '_searching'])
                        dataset_summary[metric_name + '_searching'] = {
                            'mean': float(avg_metric_searching),
                            'std': float(std_metric_searching)
                        }
                        avg_metric_per_frame_searching = np.mean(dataset_metrics[metric_name + '_per_frame_aligned_searching'])
                        std_metric_per_frame_searching = np.std(dataset_metrics[metric_name + '_per_frame_aligned_searching'])
                        dataset_summary[metric_name + '_per_frame_aligned_searching'] = {
                            'mean': float(avg_metric_per_frame_searching),
                            'std': float(std_metric_per_frame_searching)
                        }
                        print(f"{metric_name}_searching: {avg_metric_searching:.6f} ± {std_metric_searching:.6f}")
                        print(f"{metric_name}_per_frame_aligned_searching: {avg_metric_per_frame_searching:.6f} ± {std_metric_per_frame_searching:.6f}")
            
            dataset_summary['video_names'] = dataset_metrics['video_names']
            dataset_summary['num_videos'] = len(dataset_metrics['video_names'])
            all_results[dataset] = dataset_summary

            # Save per-scene (per-video) results
            per_scene_results = []
            for idx, video_name in enumerate(dataset_metrics['video_names']):
                scene_result = {'video_name': video_name}
                
                # Add lstsq metrics
                if align_method == 'lstsq' or align_method == 'all':
                    for metric_name in eval_metrics:
                        if len(dataset_metrics[metric_name + '_lstsq']) > idx:
                            scene_result[metric_name + '_lstsq'] = float(dataset_metrics[metric_name + '_lstsq'][idx])
                        else:
                            scene_result[metric_name + '_lstsq'] = None
                    for metric_name in eval_metrics:
                        if len(dataset_metrics[metric_name + '_per_frame_aligned_lstsq']) > idx:
                            scene_result[metric_name + '_per_frame_aligned_lstsq'] = float(dataset_metrics[metric_name + '_per_frame_aligned_lstsq'][idx])
                        else:
                            scene_result[metric_name + '_per_frame_aligned_lstsq'] = None
                
                # Add searching metrics
                if align_method == 'searching' or align_method == 'all':
                    for metric_name in eval_metrics:
                        if len(dataset_metrics[metric_name + '_searching']) > idx:
                            scene_result[metric_name + '_searching'] = float(dataset_metrics[metric_name + '_searching'][idx])
                        else:
                            scene_result[metric_name + '_searching'] = None
                    for metric_name in eval_metrics:
                        if len(dataset_metrics[metric_name + '_per_frame_aligned_searching']) > idx:
                            scene_result[metric_name + '_per_frame_aligned_searching'] = float(dataset_metrics[metric_name + '_per_frame_aligned_searching'][idx])
                        else:
                            scene_result[metric_name + '_per_frame_aligned_searching'] = None
                
                per_scene_results.append(scene_result)
            # Save per-scene results to a JSON file
            per_scene_file = os.path.join(output_dir, f"{dataset}_per_scene_results.json")
            with open(per_scene_file, 'w') as f:
                json.dump(per_scene_results, f, indent=2)
            print(f"Per-scene results saved to: {per_scene_file}")

        else:
            print(f"No successful evaluations for dataset: {dataset}")

    # Save results to JSON file
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    
    # Print overall summary
    for dataset, results in all_results.items():
        print(f"\nDataset: {dataset}")
        print(f"Number of videos: {results['num_videos']}")
        for metric_name in eval_metrics:
            if align_method == 'lstsq' or align_method == 'all':
                if metric_name + '_lstsq' in results and results[metric_name + '_lstsq']['mean'] is not None:
                    print(f"  {metric_name}_lstsq: {results[metric_name + '_lstsq']['mean']:.6f} ± {results[metric_name + '_lstsq']['std']:.6f}")
            if align_method == 'searching' or align_method == 'all':
                if metric_name + '_searching' in results and results[metric_name + '_searching']['mean'] is not None:
                    print(f"  {metric_name}_searching: {results[metric_name + '_searching']['mean']:.6f} ± {results[metric_name + '_searching']['std']:.6f}")
    
    print(f"\nResults saved to: {results_file}")

if __name__ == '__main__':
    main()

