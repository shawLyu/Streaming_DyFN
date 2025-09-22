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

def read_video_frames(video_path, target_fps, max_res):
    vid = VideoReader(video_path, ctx=cpu(0))
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
    print(f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}")
    
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0
    return frames, fps


@click.command(help='Evaluate video results')
# @click.option('--eval_dataset_list', type=list[str], default=['sintel', 'scannet', 'KITTI', 'bonn', 'NYUv2'], help='List of datasets to evaluate')
@click.option('--eval_dataset_list', type=list[str], default=['scannet'], help='List of datasets to evaluate')
@click.option('--video_dir_path', type=click.Path(exists=True), required=True, help='Path to evaluated video file')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default='Ruicheng/moge-vitl', help='Pretrained model name or path. Defaults to "Ruicheng/moge-vitl"')
@click.option('--output_dir', type=click.Path(), default='outputs_video', help='Directory to save output results')
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
def main(
    eval_dataset_list: List[str],
    video_dir_path: str,
    pretrained_model_name_or_path: str,
    output_dir: str,
    save_video: bool,
    target_fps: int,
    max_res: int,
    depth_max: float,
    resolution_level: int,
    num_tokens: int,
    use_fp16: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()
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
        dataset_metrics = {metric_name: [] for metric_name in eval_metrics}
        dataset_metrics.update({metric_name + '_per_frame_aligned': [] for metric_name in eval_metrics})
        dataset_metrics['video_names'] = []
        
        for video_path, gt_depth_path in zip(video_path_list, gt_depth_path_list):
            video_name = Path(video_path).stem
            print(f"\n==> Processing {video_name} and {Path(gt_depth_path).stem}")
            
            try:
                frames, fps = read_video_frames(video_path, target_fps, max_res)
                height, width = frames.shape[1:3]

                gt_depth = np.load(gt_depth_path)['disparity'] \
                    if 'disparity' in np.load(gt_depth_path).files else \
                    np.load(gt_depth_path)['arr_0']  # (t, 1, h, w)
                
                gt_depth = np.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)

                with torch.no_grad():
                    # Use sliding window of size 3 with stride 1
                    image_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(device)
                    output = model.infer_video(image_tensor, fov_x=None, resolution_level=resolution_level, 
                                            num_tokens=num_tokens, use_fp16=use_fp16)

                    if gt_depth.shape[-2] != image_tensor.shape[-2] or gt_depth.shape[-1] != image_tensor.shape[-1]:
                        output['depth'] = F.interpolate(output['depth'][:, None, ...], size=(gt_depth.shape[-2], gt_depth.shape[-1]), mode='bilinear', align_corners=False).squeeze(1)

                    points = output['points'].cpu().numpy()
                    depth = output['depth'].cpu().numpy()
                    mask = output['mask'].cpu().numpy()
                    # Prepare the depth visualization
                    depth = depth[:, None, :, :]
                    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                    valid_mask = np.logical_and(gt_depth > 1e-3, gt_depth < depth_max[dataset])

                    # Evaluate metric for per frame alignment
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

                    # Disparity alignment
                    # gt_disp_masked = 1 / (gt_depth[valid_mask].reshape((-1, 1)).astype(np.float64) + 1e-8)
                    # pred_disp_masked = 1 / (depth[valid_mask].reshape((-1, 1)).astype(np.float64) + 1e-8)
                    # disp = 1 / (depth.astype(np.float64) + 1e-8)
                    # _ones = np.ones_like(pred_disp_masked)
                    # A = np.concatenate([pred_disp_masked, _ones], axis=-1)
                    # X = np.linalg.lstsq(A, gt_disp_masked, rcond=None)[0]
                    # scale, shift = X # gt = scale * pred + shift
                    # aligned_pred_disp = scale * disp + shift
                    # aligned_pred_disp = np.clip(aligned_pred_disp, a_min=1e-3, a_max=None) 
                    # depth_placeholder = np.zeros_like(aligned_pred_disp)
                    # non_negative_mask = aligned_pred_disp > 0
                    # depth_placeholder[non_negative_mask] = 1 / aligned_pred_disp[non_negative_mask]

                    # aligned_pred_depth_from_disp = torch.from_numpy(depth_placeholder).to(device)
                    gt_depth = torch.from_numpy(gt_depth).to(device)
                    valid_mask = torch.from_numpy(valid_mask).to(device)
                    aligned_pred_depth = aligned_pred_depth[valid_frame]
                    gt_depth = gt_depth[valid_frame]
                    valid_mask = valid_mask[valid_frame]
                    per_frame_aligned_pred_depth = per_frame_aligned_pred_depth[valid_frame]

                    # evaluate metric 
                    sample_metric_depth = []
                    metric_funcs_depth = [getattr(metric, _met) for _met in eval_metrics]
                    for met_func in metric_funcs_depth:
                        _metric_name = met_func.__name__
                        _metric = met_func(aligned_pred_depth, gt_depth, valid_mask).item()
                        sample_metric_depth.append(_metric)
                        dataset_metrics[_metric_name].append(_metric)
                    
                    dataset_metrics['video_names'].append(video_name)
                    
                    print(f"==> Depth metrics for {video_name}:")
                    for metric_name, metric_value in zip(eval_metrics, sample_metric_depth):
                        print(f"  {metric_name}: {metric_value:.6f}")

                    # evaluate metric for per frame alignment
                    sample_metric_depth_per_frame = []
                    metric_funcs_depth_per_frame = [getattr(metric, _met) for _met in eval_metrics]
                    for met_func in metric_funcs_depth_per_frame:
                        _metric_name = met_func.__name__
                        _metric = met_func(per_frame_aligned_pred_depth, gt_depth, valid_mask).item()
                        sample_metric_depth_per_frame.append(_metric)
                        dataset_metrics[_metric_name + '_per_frame_aligned'].append(_metric)
                    
                    print(f"==> Depth metrics for {video_name} aligned by per frame:")
                    for metric_name, metric_value in zip(eval_metrics, sample_metric_depth_per_frame):
                        print(f"  {metric_name}_per_frame_aligned: {metric_value:.6f}")
                    
                    # sample_metric_depth_from_disp = []
                    # metric_funcs_depth = [getattr(metric, _met) for _met in eval_metrics]
                    # for met_func in metric_funcs_depth:
                    #     _metric_name = met_func.__name__
                    #     _metric = met_func(aligned_pred_depth_from_disp, gt_depth, valid_mask).item()
                    #     sample_metric_depth_from_disp.append(_metric)
                    #     dataset_metrics[_metric_name + '_from_disp'].append(_metric)
                    
                    # dataset_metrics['video_names'].append(video_name)
                    
                    # print(f"==> Depth metrics for {video_name} aligned by disparity:")
                    # for metric_name, metric_value in zip(eval_metrics, sample_metric_depth_from_disp):
                    #     print(f"  {metric_name}_from_disp: {metric_value:.6f}")

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
            for metric_name in eval_metrics:
                if dataset_metrics[metric_name]:
                    avg_metric = np.mean(dataset_metrics[metric_name])
                    std_metric = np.std(dataset_metrics[metric_name])
                    dataset_summary[metric_name] = {
                        'mean': float(avg_metric),
                        'std': float(std_metric)
                        # 'values': [float(x) for x in dataset_metrics[metric_name]]
                    }
                    avg_metric_per_frame_aligned = np.mean(dataset_metrics[metric_name + '_per_frame_aligned'])
                    std_metric_per_frame_aligned = np.std(dataset_metrics[metric_name + '_per_frame_aligned'])
                    dataset_summary[metric_name + '_per_frame_aligned'] = {
                        'mean': float(avg_metric_per_frame_aligned),
                        'std': float(std_metric_per_frame_aligned)
                    }
                    print(f"{metric_name}: {avg_metric:.6f} ± {std_metric:.6f}")
                    print(f"{metric_name}_per_frame_aligned: {avg_metric_per_frame_aligned:.6f} ± {std_metric_per_frame_aligned:.6f}")
                else:
                    dataset_summary[metric_name] = {'mean': None, 'std': None, 'values': []}
            
            dataset_summary['video_names'] = dataset_metrics['video_names']
            dataset_summary['num_videos'] = len(dataset_metrics['video_names'])
            all_results[dataset] = dataset_summary

            # Save per-scene (per-video) results
            per_scene_results = []
            for idx, video_name in enumerate(dataset_metrics['video_names']):
                scene_result = {'video_name': video_name}
                for metric_name in eval_metrics:
                    if len(dataset_metrics[metric_name]) > idx:
                        scene_result[metric_name] = float(dataset_metrics[metric_name][idx])
                    else:
                        scene_result[metric_name] = None
                per_scene_results.append(scene_result)
                for metric_name in eval_metrics:
                    if len(dataset_metrics[metric_name + '_per_frame_aligned']) > idx:
                        scene_result[metric_name + '_per_frame_aligned'] = float(dataset_metrics[metric_name + '_per_frame_aligned'][idx])
                    else:
                        scene_result[metric_name + '_per_frame_aligned'] = None
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
            if results[metric_name]['mean'] is not None:
                print(f"  {metric_name}: {results[metric_name]['mean']:.6f} ± {results[metric_name]['std']:.6f}")
    
    print(f"\nResults saved to: {results_file}")

if __name__ == '__main__':
    main()
