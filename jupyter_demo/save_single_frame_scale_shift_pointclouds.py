#!/usr/bin/env python3
"""Save point clouds for the SingleFrameScaleShift notebook sweep."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.append(str(REPO_ROOT))

from moge.model.v1 import MoGeModel  # noqa: E402
from moge.utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift  # noqa: E402


@dataclass(frozen=True)
class SceneConfig:
    key: str
    display_name: str
    image_path: Path
    depth_path: Path


SCENES = {
    "indoor": SceneConfig(
        key="indoor",
        display_name="Indoor (ScanNet)",
        image_path=REPO_ROOT / "assets/video_data/scene0707_00_first_image.png",
        depth_path=REPO_ROOT / "assets/video_data/scene0707_00_first_depth.npy",
    ),
    "outdoor": SceneConfig(
        key="outdoor",
        display_name="Outdoor (Sintel)",
        image_path=REPO_ROOT / "assets/video_data/market_2_first_image.png",
        depth_path=REPO_ROOT / "assets/video_data/market_2_first_depth.npy",
    ),
}


def load_rgb_image(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return image


def resize_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))
    return np.asarray(pil.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]

    if colors is not None:
        colors = np.asarray(colors).reshape(-1, 3)[finite]
        if colors.dtype != np.uint8:
            colors = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)

    vertex = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        vertex.tofile(f)


def depth_to_points_np(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    ys, xs = np.meshgrid(
        (np.arange(height, dtype=np.float32) + 0.5) / height,
        (np.arange(width, dtype=np.float32) + 0.5) / width,
        indexing="ij",
    )
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    points = np.stack([(xs - cx) / fx * depth, (ys - cy) / fy * depth, depth], axis=-1)
    return points


@torch.no_grad()
def extract_backbone_features(model: MoGeModel, image_tensor: torch.Tensor, num_tokens: int):
    _, _, orig_h, orig_w = image_tensor.shape

    resize_factor = ((num_tokens * 14**2) / (orig_h * orig_w)) ** 0.5
    rh, rw = int(orig_h * resize_factor), int(orig_w * resize_factor)
    image_resized = F.interpolate(
        image_tensor, (rh, rw), mode="bicubic", align_corners=False, antialias=True
    )
    image_normed = (image_resized - model.image_mean) / model.image_std
    image_14 = F.interpolate(
        image_normed,
        (rh // 14 * 14, rw // 14 * 14),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    hidden_states = model.backbone.get_intermediate_layers(
        image_14, model.intermediate_layers, return_class_token=True
    )
    return hidden_states, image_14, orig_h, orig_w


@torch.no_grad()
def decode_with_adjusted_features(
    model: MoGeModel,
    hidden_states,
    image_14: torch.Tensor,
    orig_h: int,
    orig_w: int,
    mean_multiplier: float = 1.0,
    std_multiplier: float = 1.0,
):
    img_h, img_w = image_14.shape[-2:]
    patch_h, patch_w = img_h // 14, img_w // 14

    x = torch.stack(
        [
            proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
            for proj, (feat, _) in zip(model.head.projects, hidden_states)
        ],
        dim=1,
    ).sum(dim=1)

    x_mean = x.mean(dim=[2, 3], keepdim=True)
    x_std = x.std(dim=[2, 3], keepdim=True)
    x_norm = (x - x_mean) / (x_std + 1e-6)
    x = x_norm * (std_multiplier * x_std) + (mean_multiplier * x_mean)

    for block in model.head.upsample_blocks:
        uv = normalized_view_plane_uv(
            width=x.shape[-1],
            height=x.shape[-2],
            aspect_ratio=img_w / img_h,
            dtype=x.dtype,
            device=x.device,
        ).permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], dim=1)
        for layer in block:
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)

    x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
    uv = normalized_view_plane_uv(
        width=x.shape[-1],
        height=x.shape[-2],
        aspect_ratio=img_w / img_h,
        dtype=x.dtype,
        device=x.device,
    ).permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
    x = torch.cat([x, uv], dim=1)

    if isinstance(model.head.output_block, nn.ModuleList):
        output = [
            torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            for blk in model.head.output_block
        ]
    else:
        output = torch.utils.checkpoint.checkpoint(model.head.output_block, x, use_reentrant=False)
    points, mask = output

    points = F.interpolate(points, (orig_h, orig_w), mode="bilinear", align_corners=False)
    mask = F.interpolate(mask, (orig_h, orig_w), mode="bilinear", align_corners=False)
    points = points.permute(0, 2, 3, 1)
    points = model._remap_points(points)

    mask_binary = mask.squeeze(1) > model.mask_threshold
    focal, shift = recover_focal_shift(points, mask_binary)
    aspect_ratio = orig_w / orig_h
    fx = focal / 2 * (1 + aspect_ratio**2) ** 0.5 / aspect_ratio
    fy = focal / 2 * (1 + aspect_ratio**2) ** 0.5
    intrinsics = torch.zeros((points.shape[0], 3, 3), dtype=points.dtype, device=points.device)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = 0.5
    intrinsics[:, 1, 2] = 0.5
    intrinsics[:, 2, 2] = 1.0
    depth = points[..., 2] + shift[..., None, None]

    return {
        "depth": depth.squeeze(0),
        "mask": mask_binary.squeeze(0),
        "intrinsics": intrinsics.squeeze(0),
    }


def compute_alignment(pred_depth: np.ndarray, gt_depth: np.ndarray, depth_min: float, depth_max: float):
    pred_depth = np.squeeze(pred_depth)
    gt_depth = np.squeeze(gt_depth)
    pred_resized = F.interpolate(
        torch.from_numpy(pred_depth).float()[None, None],
        size=gt_depth.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze().numpy()

    valid = (gt_depth > depth_min) & (gt_depth < depth_max) & np.isfinite(pred_resized)
    pred_v = pred_resized[valid].reshape(-1, 1)
    gt_v = gt_depth[valid].reshape(-1, 1)
    coefficients = np.linalg.lstsq(
        np.concatenate([pred_v, np.ones_like(pred_v)], axis=-1), gt_v, rcond=None
    )[0]
    scale, shift = float(coefficients[0, 0]), float(coefficients[1, 0])

    aligned = np.clip(scale * pred_v + shift, depth_min, depth_max)
    absrel = float(np.nanmean(np.abs(gt_v - aligned) / (gt_v + 1e-6)))
    return scale, shift, absrel, pred_resized


def safe_float_token(value: float) -> str:
    return f"{value:.3f}".replace("-", "m").replace(".", "p")


def save_scene_pointclouds(
    model: MoGeModel,
    scene: SceneConfig,
    output_dir: Path,
    mean_range: np.ndarray,
    std_range: np.ndarray,
    num_tokens: int,
    device: str,
    depth_min: float,
    depth_max: float,
    save_aligned: bool,
) -> list[dict[str, float | str]]:
    rgb = load_rgb_image(scene.image_path)
    gt_depth = np.load(scene.depth_path)
    image_tensor = torch.from_numpy(rgb[None]).permute(0, 3, 1, 2).to(device)
    hidden_states, image_14, orig_h, orig_w = extract_backbone_features(model, image_tensor, num_tokens)
    pred_colors = rgb.reshape(-1, 3)

    records: list[dict[str, float | str]] = []
    gt_saved = False
    total = len(mean_range) * len(std_range)
    progress = tqdm(total=total, desc=f"{scene.key}: saving point clouds")

    for alpha in mean_range:
        for beta in std_range:
            output = decode_with_adjusted_features(
                model,
                hidden_states,
                image_14,
                orig_h,
                orig_w,
                mean_multiplier=float(alpha),
                std_multiplier=float(beta),
            )
            depth = output["depth"].detach().float().cpu().numpy()
            mask = output["mask"].detach().cpu().numpy()
            intrinsics = output["intrinsics"].detach().float().cpu().numpy()
            scale, shift, absrel, pred_resized = compute_alignment(depth, gt_depth, depth_min, depth_max)

            if not gt_saved:
                gt_colors = resize_rgb(rgb, gt_depth.shape[0], gt_depth.shape[1])
                gt_mask = (gt_depth > depth_min) & (gt_depth < depth_max) & np.isfinite(gt_depth)
                gt_points = depth_to_points_np(gt_depth.astype(np.float32), intrinsics)
                write_binary_ply(
                    output_dir / f"{scene.key}_gt.ply",
                    gt_points[gt_mask],
                    gt_colors[gt_mask],
                )
                gt_saved = True

            pred_points = depth_to_points_np(depth, intrinsics)
            pred_mask = mask & np.isfinite(depth) & (depth > depth_min) & (depth < depth_max)
            stem = (
                f"{scene.key}_alpha_{safe_float_token(float(alpha))}"
                f"_beta_{safe_float_token(float(beta))}"
            )
            pred_file = output_dir / f"{stem}_pred.ply"
            write_binary_ply(pred_file, pred_points[pred_mask], pred_colors[pred_mask.reshape(-1)])

            aligned_file = ""
            if save_aligned:
                aligned_depth = scale * pred_resized + shift
                aligned_mask = (
                    (gt_depth > depth_min)
                    & (gt_depth < depth_max)
                    & np.isfinite(aligned_depth)
                    & (aligned_depth > depth_min)
                    & (aligned_depth < depth_max)
                )
                aligned_points = depth_to_points_np(aligned_depth.astype(np.float32), intrinsics)
                aligned_colors = resize_rgb(rgb, gt_depth.shape[0], gt_depth.shape[1])
                aligned_file = str(output_dir / f"{stem}_aligned.ply")
                write_binary_ply(aligned_file, aligned_points[aligned_mask], aligned_colors[aligned_mask])

            records.append(
                {
                    "scene": scene.key,
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "scale": scale,
                    "shift": shift,
                    "absrel": absrel,
                    "pred_ply": str(pred_file),
                    "aligned_ply": aligned_file,
                    "gt_ply": str(output_dir / f"{scene.key}_gt.ply"),
                }
            )
            progress.update(1)

    progress.close()
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", default="Ruicheng/moge-vitl", help="MoGe checkpoint/name.")
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "single_frame_scale_shift_pointclouds")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-tokens", type=int, default=1500)
    parser.add_argument("--mean-min", type=float, default=0.5)
    parser.add_argument("--mean-max", type=float, default=2.0)
    parser.add_argument("--mean-count", type=int, default=20)
    parser.add_argument("--std-min", type=float, default=0.5)
    parser.add_argument("--std-max", type=float, default=2.0)
    parser.add_argument("--std-count", type=int, default=20)
    parser.add_argument("--depth-min", type=float, default=1e-3)
    parser.add_argument("--depth-max", type=float, default=20.0)
    parser.add_argument("--scene", choices=["all", *SCENES.keys()], default="all")
    parser.add_argument("--save-aligned", action="store_true", help="Also save scale/shift aligned PLYs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mean_range = np.linspace(args.mean_min, args.mean_max, args.mean_count)
    std_range = np.linspace(args.std_min, args.std_max, args.std_count)
    selected_scenes = SCENES.values() if args.scene == "all" else [SCENES[args.scene]]

    print(f"Loading pretrained MoGe model from {args.pretrained}...")
    model = MoGeModel.from_pretrained(args.pretrained).to(args.device).eval()
    print(f"Saving point clouds to {args.output_dir}")

    all_records: list[dict[str, float | str]] = []
    for scene in selected_scenes:
        all_records.extend(
            save_scene_pointclouds(
                model=model,
                scene=scene,
                output_dir=args.output_dir,
                mean_range=mean_range,
                std_range=std_range,
                num_tokens=args.num_tokens,
                device=args.device,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
                save_aligned=args.save_aligned,
            )
        )

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)

    metadata = {
        "pretrained": args.pretrained,
        "num_tokens": args.num_tokens,
        "mean_range": mean_range.tolist(),
        "std_range": std_range.tolist(),
        "depth_min": args.depth_min,
        "depth_max": args.depth_max,
        "scene": args.scene,
        "save_aligned": args.save_aligned,
        "num_records": len(all_records),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done. Wrote {len(all_records)} predicted point clouds, metrics.csv, metadata.json, and GT PLYs.")


if __name__ == "__main__":
    main()
