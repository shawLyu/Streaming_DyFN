import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))
import gzip
import json
import numpy as np
from PIL import Image
import gc
from tqdm import tqdm
import warnings
import utils3d
from moge.utils.io import write_depth, write_image, write_meta


warnings.filterwarnings("ignore")

DATA_ROOT = "data/dynamic_replica/extracted"
OUTPUT_DIR = "data/moge_process_dataset/dynamic_replica"
SPLIT = "train"  # or "val", "test"
NUM_WORKERS = 64

# --- DataClass and Loader (from dynamic_replica_hdf5.py) ---
from typing import List, Optional
from dataclasses import dataclass
from pytorch3d.implicitron.dataset.types import (
    FrameAnnotation as ImplicitronFrameAnnotation,
    load_dataclass
)

@dataclass
class DynamicReplicaFrameAnnotation(ImplicitronFrameAnnotation):
    camera_name: Optional[str] = None
    instance_id_map_path: Optional[str] = None
    flow_forward: Optional[str] = None
    flow_forward_mask: Optional[str] = None
    flow_backward: Optional[str] = None
    flow_backward_mask: Optional[str] = None
    trajectories: Optional[str] = None

def _get_pytorch3d_camera(entry_viewpoint, image_size, scale: float):
    import torch
    principal_point = torch.tensor(entry_viewpoint.principal_point, dtype=torch.float)
    focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)
    half_image_size_wh_orig = (
        torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
    )
    fmt = entry_viewpoint.intrinsics_format
    if fmt.lower() == "ndc_norm_image_bounds":
        rescale = half_image_size_wh_orig
    elif fmt.lower() == "ndc_isotropic":
        rescale = half_image_size_wh_orig.min()
    else:
        raise ValueError(f"Unknown intrinsics format: {fmt}")
    principal_point_px = half_image_size_wh_orig - principal_point * rescale
    focal_length_px = focal_length * rescale
    R = torch.tensor(entry_viewpoint.R, dtype=torch.float)
    T = torch.tensor(entry_viewpoint.T, dtype=torch.float)
    R_pytorch3d = R.clone()
    T_pytorch3d = T.clone()
    T_pytorch3d[..., :2] *= -1
    R_pytorch3d[..., :, :2] *= -1
    tvec = T_pytorch3d
    return R_pytorch3d, tvec, focal_length_px, principal_point_px

def _load_16big_png_depth(depth_png):
    with Image.open(depth_png) as depth_pil:
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth

def collect_frame_data(data_root, split):
    frame_annotations_file = f'frame_annotations_{split}.jgz'
    frame_annots_path = os.path.join(data_root, split, frame_annotations_file)
    with gzip.open(frame_annots_path, "rt", encoding="utf8") as zipfile:
        frame_annots_list = load_dataclass(zipfile, List[DynamicReplicaFrameAnnotation])
    all_frames = [f for f in frame_annots_list if f.camera_name == 'left']
    frames = []
    for idx, frame in enumerate(all_frames):
        frames.append({
            "frame": frame,
            "sequence_name": getattr(frame, "sequence_name", "dynamic_replica"),
            "frame_idx": idx
        })
    return frames

def write_index_file(output_dir, instance_paths):
    output_dir_path = Path(output_dir)
    existing_paths = []
    for path in instance_paths:
        instance_dir = output_dir_path / path
        if (instance_dir / 'image.jpg').exists() and \
           (instance_dir / 'depth.png').exists() and \
           (instance_dir / 'meta.json').exists():
            existing_paths.append(path)
    if existing_paths:
        index_path = output_dir_path / 'index.txt'
        with open(index_path, 'w') as f:
            for path in sorted(existing_paths):
                f.write(f'{path}\n')
        print(f"Wrote index file to {index_path} with {len(existing_paths)} instances")
    else:
        print("No valid instances found, index.txt not created")

if __name__ == "__main__":
    print("Collecting frame data...")
    all_frames = collect_frame_data(DATA_ROOT, SPLIT)
    print(f"Found {len(all_frames)} frames to process")

    print("Processing frames...")
    all_instance_paths = [f"{frame['sequence_name']}/{frame['frame_idx']:05d}" 
                         for frame in all_frames]

    import concurrent.futures

    def process_single_frame(frame_data):
        try:
            frame = frame_data["frame"]
            # Load RGB image
            img_path = os.path.join(DATA_ROOT, SPLIT, frame.image.path)
            rgb = np.array(Image.open(img_path))[:, :, :3]
            h, w = rgb.shape[:2]
            # Load depth
            depth_path = os.path.join(DATA_ROOT, SPLIT, frame.depth.path)
            depth = _load_16big_png_depth(depth_path)
            depth_mask_inf = depth >= 65.535
            depth_mask = ~depth_mask_inf
            depth = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth, np.nan))
            depth[depth == 0.0] = np.nan
            # Get camera parameters
            R, t, focal, pp = _get_pytorch3d_camera(
                frame.viewpoint, frame.image.size, scale=1.0
            )
            # Create intrinsics matrix
            K = np.eye(3, dtype=np.float32)
            K[0, 0] = focal[0].item()
            K[1, 1] = focal[1].item()
            K[0, 2] = pp[0].item()
            K[1, 2] = pp[1].item()
            K = utils3d.numpy.normalize_intrinsics(K, w, h)
            # Create camera-to-world transform
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = R.numpy().T
            c2w[:3, 3] = -R.numpy().T @ t.numpy()
            # Save processed data
            save_path = Path(OUTPUT_DIR) / frame_data["sequence_name"] / f"{frame_data['frame_idx']:05d}"
            save_path.mkdir(parents=True, exist_ok=True)
            write_image(save_path / 'image.jpg', rgb, quality=95)
            write_depth(save_path / 'depth.png', depth, unit=1)
            write_meta(save_path / 'meta.json', {
                'intrinsics': K.tolist(),
                'camera_pose': c2w.tolist()
            })
        except Exception as e:
            print(f"Error processing {frame_data['sequence_name']}/{frame_data['frame_idx']:05d}: {e}")
        gc.collect()

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_single_frame, all_frames), total=len(all_frames)))

    write_index_file(OUTPUT_DIR, all_instance_paths)
