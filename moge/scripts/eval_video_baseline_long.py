import os
import sys
from pathlib import Path
from typing import *
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

import cv2
import glob
import json
import click
import torch
import utils3d
import mediapy
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
import torch.nn.functional as F
from sklearn.linear_model import RANSACRegressor, LinearRegression

from moge.model.v1 import MoGeModel
from moge.utils.io import save_ply
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


@click.command(help='Evaluate video results')
# @click.option('--eval_dataset_list', type=list[str], default=['sintel', 'scannet', 'KITTI', 'bonn', 'NYUv2'], help='List of datasets to evaluate')
# @click.option('--eval_dataset_list', type=list[str], default=['sintel', 'scannet', 'KITTI', 'bonn'], help='List of datasets to evaluate')
@click.option('--eval_dataset_list', type=list[str], default=['scannet_long'], help='List of datasets to evaluate')
@click.option('--video_dir_path', type=click.Path(exists=True), required=True, help='Path to evaluated video file')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default='Ruicheng/moge-vitl', help='Pretrained model name or path. Defaults to "Ruicheng/moge-vitl"')
@click.option('--output_dir', type=click.Path(), default='outputs_video', help='Directory to save output results')
@click.option('--align_method', type=str, default='all', help='Method to align depth. Defaults to "all". Options: "lstsq", "searching", "all".')
@click.option('--save_video', is_flag=True, help='Save output as video')
@click.option('--target_fps', type=int, default=15, help='Target frames per second for video processing')
@click.option('--max_res', type=int, default=1024, help='Maximum resolution dimension')
@click.option('--depth_max', type=float, default=80, help='Maximum depth value for visualization')
@click.option('--resolution_level', type=int, default=9, help='An integer [0-9] for the resolution level for inference. \
Higher value means more tokens and the finer details will be captured, but inference can be slower. \
Defaults to 9. Note that it is irrelevant to the output size, which is always the same as the input size. \
`resolution_level` actually controls `num_tokens`. See `num_tokens` for more details.')
@click.option('--num_tokens', type=int, default=None, help='number of tokens used for inference. A integer in the (suggested) range of `[1200, 2500]`. \
`resolution_level` will be ignored if `num_tokens` is provided. Default: None')
@click.option('--use_fp16', is_flag=True, help='Use fp16 precision for 2x faster inference.')
@click.option('--image_based', is_flag=True, help='Use image-based inference.')
@click.option('--silent', is_flag=True, help='Do not print any information.')
def main(
    eval_dataset_list: List[str],
    video_dir_path: str,
    pretrained_model_name_or_path: str,
    output_dir: str,
    align_method: str,
    save_video: bool,
    target_fps: int,
    max_res: int,
    depth_max: float,
    resolution_level: int,
    num_tokens: int,
    use_fp16: bool,
    image_based: bool,
    silent: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()
    depth_max = {"sintel": 70, "scannet": 10, "KITTI": 80, "bonn": 10, "NYUv2": 10, "scannet_long": 10}
    
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

        video_path_list = sorted(glob.glob(str(dataset_path / "*.mp4")))
        gt_depth_path_list = sorted(glob.glob(str(dataset_path / "*.npz")))

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

                gt_depth = np.load(gt_depth_path)['disparity'] \
                    if 'disparity' in np.load(gt_depth_path).files else \
                    np.load(gt_depth_path)['arr_0']  # (t, 1, h, w)
                
                gt_depth = np.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)

                with torch.no_grad():
                    # Use sliding window of size 3 with stride 1
                    dataset_metrics['video_names'].append(video_name)
                    image_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(device)
                    output = model.infer_video(image_tensor, fov_x=None, resolution_level=resolution_level, 
                                            num_tokens=num_tokens, use_fp16=use_fp16, image_based=image_based)

                    if gt_depth.shape[-2] != image_tensor.shape[-2] or gt_depth.shape[-1] != image_tensor.shape[-1]:
                        output['depth'] = F.interpolate(output['depth'][:, None, ...], size=(gt_depth.shape[-2], gt_depth.shape[-1]), mode='bilinear', align_corners=False).squeeze(1)

                    points = output['points'].cpu().numpy()
                    depth = output['depth'].cpu().numpy()
                    mask = output['mask'].cpu().numpy()
                    # Prepare the depth visualization
                    depth = depth[:, None, :, :]
                    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                    valid_mask = np.logical_and(gt_depth > 1e-3, gt_depth < depth_max[dataset])

                    if align_method == 'lstsq' or align_method == 'all':
                        # Test different frame counts: 100, 200, 300, 400, 500
                        frame_counts = [100, 200, 300, 400, 500]
                        total_frames = depth.shape[0]
                        
                        for num_frames in frame_counts:
                            # Use only the first num_frames
                            actual_frames = min(num_frames, total_frames)
                            depth_subset = depth[:actual_frames]
                            gt_depth_subset = gt_depth[:actual_frames]
                            valid_mask_subset = valid_mask[:actual_frames]
                            
                            if not silent:
                                print(f"\n==> Processing {video_name} with {actual_frames} frames")
                            
                            # Evaluate metric for per frame alignment for lstsq
                            scales_per_frame, shifts_per_frame = [], []
                            per_frame_aligned_pred_depth = []
                            for pred_depth_i, gt_depth_i in zip(depth_subset, gt_depth_subset):
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

                            # Depth alignment using subset
                            gt_depth_masked_subset = gt_depth_subset[valid_mask_subset].reshape((-1, 1)).astype(np.float64)
                            pred_depth_masked_subset = depth_subset[valid_mask_subset].reshape((-1, 1)).astype(np.float64)
                            _ones = np.ones_like(pred_depth_masked_subset)
                            A = np.concatenate([pred_depth_masked_subset, _ones], axis=-1)
                            X = np.linalg.lstsq(A, gt_depth_masked_subset, rcond=None)[0]
                            scale_global, shift_global = X # gt = scale * pred + shift
                            aligned_pred_depth = scale_global * depth_subset + shift_global
                            aligned_pred_depth = np.clip(aligned_pred_depth, a_min=1e-3, a_max=depth_max[dataset]) 
                            aligned_pred_depth = torch.from_numpy(aligned_pred_depth).to(device)

                            n = valid_mask_subset.sum((-1, -2))
                            valid_frame = (n > 0)

                            gt_depth_torch = torch.from_numpy(gt_depth_subset).to(device)
                            valid_mask_torch = torch.from_numpy(valid_mask_subset).to(device)
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
                                # Store metrics with frame count suffix
                                metric_key = f"{_metric_name}_lstsq_{actual_frames}frames"
                                if metric_key not in dataset_metrics:
                                    dataset_metrics[metric_key] = []
                                dataset_metrics[metric_key].append(_metric)
                            
                            
                            if not silent:
                                print(f"==> Depth metrics for {video_name} ({actual_frames} frames):")
                                for metric_name, metric_value in zip(eval_metrics, sample_metric_depth):
                                    print(f"  {metric_name}_lstsq: {metric_value:.6f}")

                            # evaluate metric for per frame alignment
                            sample_metric_depth_per_frame = []
                            metric_funcs_depth_per_frame = [getattr(metric, _met) for _met in eval_metrics]
                            for met_func in metric_funcs_depth_per_frame:
                                _metric_name = met_func.__name__
                                _metric = met_func(per_frame_aligned_pred_depth_valid_torch, gt_depth_valid_torch, valid_mask_valid_torch).item()
                                sample_metric_depth_per_frame.append(_metric)
                                # Store metrics with frame count suffix
                                metric_key = f"{_metric_name}_per_frame_aligned_lstsq_{actual_frames}frames"
                                if metric_key not in dataset_metrics:
                                    dataset_metrics[metric_key] = []
                                dataset_metrics[metric_key].append(_metric)
                            
                            if not silent:
                                print(f"==> Depth metrics for {video_name} aligned by per frame ({actual_frames} frames):")
                                for metric_name, metric_value in zip(eval_metrics, sample_metric_depth_per_frame):
                                    print(f"  {metric_name}_per_frame_aligned_lstsq: {metric_value:.6f}")
                            
                            # Save results for this frame count
                            frame_results = {
                                'video_name': video_name,
                                'num_frames': actual_frames,
                                'global_aligned': {},
                                'per_frame_aligned': {}
                            }
                            for metric_name, metric_value in zip(eval_metrics, sample_metric_depth):
                                frame_results['global_aligned'][metric_name + '_lstsq'] = float(metric_value)
                            for metric_name, metric_value in zip(eval_metrics, sample_metric_depth_per_frame):
                                frame_results['per_frame_aligned'][metric_name + '_per_frame_aligned_lstsq'] = float(metric_value)
                            
                            # Save to file
                            frame_results_file = os.path.join(output_dir, f"{video_name}_lstsq_{actual_frames}frames.json")
                            Path(output_dir).mkdir(parents=True, exist_ok=True)
                            with open(frame_results_file, 'w') as f:
                                json.dump(frame_results, f, indent=2)
                            if not silent:
                                print(f"==> Saved results to: {frame_results_file}")

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
                                import ipdb; ipdb.set_trace()
                            
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

                        scale_all, shift_all = align_depth_affine(pred_depth_lr_all[::gt_depth.shape[0]], gt_depth_lr_all[::gt_depth.shape[0]], 1 / gt_depth_lr_all[::gt_depth.shape[0]])
                        
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
                continue

        # Calculate average metrics for this dataset
        if dataset_metrics['video_names']:
            print(f"\n{'='*30}")
            print(f"Dataset: {dataset} - Average Metrics")
            print(f"{'='*30}")
            
            dataset_summary = {}
            frame_counts = [100, 200, 300, 400, 500]
            
            for metric_name in eval_metrics:
                # Add lstsq metrics for different frame counts
                if align_method == 'lstsq' or align_method == 'all':
                    for num_frames in frame_counts:
                        # Global aligned metrics
                        metric_key_global = f"{metric_name}_lstsq_{num_frames}frames"
                        if metric_key_global in dataset_metrics and dataset_metrics[metric_key_global]:
                            avg_metric = np.mean(dataset_metrics[metric_key_global])
                            std_metric = np.std(dataset_metrics[metric_key_global])
                            dataset_summary[metric_key_global] = {
                                'mean': float(avg_metric),
                                'std': float(std_metric)
                            }
                            print(f"{metric_key_global}: {avg_metric:.6f} ± {std_metric:.6f}")
                        
                        # Per frame aligned metrics
                        metric_key_per_frame = f"{metric_name}_per_frame_aligned_lstsq_{num_frames}frames"
                        if metric_key_per_frame in dataset_metrics and dataset_metrics[metric_key_per_frame]:
                            avg_metric = np.mean(dataset_metrics[metric_key_per_frame])
                            std_metric = np.std(dataset_metrics[metric_key_per_frame])
                            dataset_summary[metric_key_per_frame] = {
                                'mean': float(avg_metric),
                                'std': float(std_metric)
                            }
                            print(f"{metric_key_per_frame}: {avg_metric:.6f} ± {std_metric:.6f}")
                
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
                
                # Add lstsq metrics for different frame counts
                if align_method == 'lstsq' or align_method == 'all':
                    for num_frames in frame_counts:
                        for metric_name in eval_metrics:
                            metric_key_global = f"{metric_name}_lstsq_{num_frames}frames"
                            if metric_key_global in dataset_metrics and len(dataset_metrics[metric_key_global]) > idx:
                                scene_result[metric_key_global] = float(dataset_metrics[metric_key_global][idx])
                            else:
                                scene_result[metric_key_global] = None
                            
                            metric_key_per_frame = f"{metric_name}_per_frame_aligned_lstsq_{num_frames}frames"
                            if metric_key_per_frame in dataset_metrics and len(dataset_metrics[metric_key_per_frame]) > idx:
                                scene_result[metric_key_per_frame] = float(dataset_metrics[metric_key_per_frame][idx])
                            else:
                                scene_result[metric_key_per_frame] = None
                
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
    frame_counts = [100, 200, 300, 400, 500]
    for dataset, results in all_results.items():
        print(f"\nDataset: {dataset}")
        print(f"Number of videos: {results['num_videos']}")
        for metric_name in eval_metrics:
            if align_method == 'lstsq' or align_method == 'all':
                for num_frames in frame_counts:
                    metric_key_global = f"{metric_name}_lstsq_{num_frames}frames"
                    metric_key_per_frame = f"{metric_name}_per_frame_aligned_lstsq_{num_frames}frames"
                    if metric_key_global in results and results[metric_key_global]['mean'] is not None:
                        print(f"  {metric_key_global}: {results[metric_key_global]['mean']:.6f} ± {results[metric_key_global]['std']:.6f}")
                    if metric_key_per_frame in results and results[metric_key_per_frame]['mean'] is not None:
                        print(f"  {metric_key_per_frame}: {results[metric_key_per_frame]['mean']:.6f} ± {results[metric_key_per_frame]['std']:.6f}")
            if align_method == 'searching' or align_method == 'all':
                if metric_name + '_searching' in results and results[metric_name + '_searching']['mean'] is not None:
                    print(f"  {metric_name}_searching: {results[metric_name + '_searching']['mean']:.6f} ± {results[metric_name + '_searching']['std']:.6f}")
    
    print(f"\nResults saved to: {results_file}")

if __name__ == '__main__':
    main()
