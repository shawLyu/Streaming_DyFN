import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))
import cv2
import numpy as np
from PIL import Image
import glob
import gc
import warnings
import json
from tqdm import tqdm
import imageio
import utils3d

from moge.utils.io import write_depth, write_image, write_meta

warnings.filterwarnings("ignore")

DATA_ROOT = "data/fsd/extracted"
OUTPUT_DIR = "data/moge_process_dataset/fsd"
NUM_WORKERS = 64

def depth_uint8_decoding(depth_uint8, scale=1000):
    """Decode depth from uint8 format."""
    depth_uint8 = depth_uint8.astype(float)
    out = depth_uint8[...,0]*255*255 + depth_uint8[...,1]*255 + depth_uint8[...,2]
    return out/float(scale)

def collect_frame_data(data_root):
    all_frames = []
    scene_dirs = sorted(os.listdir(data_root))
    for scene_id in tqdm(scene_dirs, desc="Collecting scenes"):
        scene_path = os.path.join(data_root, scene_id, "dataset", "data")
        if not os.path.isdir(scene_path):
            continue
        left_rgb_pattern = os.path.join(scene_path, "left", "rgb", "*.jpg")
        frame_paths = sorted(glob.glob(left_rgb_pattern))
        for frame_path in frame_paths:
            frame_id = os.path.basename(frame_path).split('.')[0]
            left_rgb_path = os.path.join(scene_path, "left", "rgb", f"{frame_id}.jpg")
            right_rgb_path = os.path.join(scene_path, "right", "rgb", f"{frame_id}.jpg")
            disparity_path = os.path.join(scene_path, "left", "disparity", f"{frame_id}.png")
            if os.path.exists(left_rgb_path) and os.path.exists(right_rgb_path) and \
               os.path.exists(disparity_path):
                all_frames.append({
                    'left_rgb_path': left_rgb_path,
                    'right_rgb_path': right_rgb_path,
                    'disparity_path': disparity_path,
                    'sequence_name': scene_id,
                    'frame_idx': frame_id
                })

    return all_frames

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
    if os.path.exists("fsd_frame_info.json"):
        print("Loading frame info from preprocessed JSON")
        with open("fsd_frame_info.json", "r") as f:
            all_frames = json.load(f)
    else:
        print("Collecting frame data...")
        all_frames = collect_frame_data(DATA_ROOT)
        print(f"Found {len(all_frames)} frames to process")
        with open("fsd_frame_info.json", "w") as f:
            json.dump(all_frames, f, indent=4)

    print("Processing frames...")
    all_instance_paths = [f"{frame['sequence_name']}/{frame['frame_idx']}" 
                         for frame in all_frames]

    import concurrent.futures

    def process_single_frame(frame_data):
        try:
            # Load RGB image
            rgb = cv2.imread(frame_data['left_rgb_path'])
            if rgb is None:
                print(f"Failed to load image: {frame_data['left_rgb_path']}")
                return
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            # Load and process disparity
            disp = imageio.imread(frame_data['disparity_path'])
            if disp is None:
                print(f"Failed to load disparity: {frame_data['disparity_path']}")
                return
            if disp.ndim == 2:
                disp = np.stack([disp, disp, disp], axis=-1)
            disp = depth_uint8_decoding(disp)
            # Camera parameters
            focal_length = 2.87
            stereo_baseline = 0.15
            horiz_aperture = 5.76
            fx = focal_length * w / horiz_aperture
            fy = fx
            # Camera intrinsics
            K = np.array([
                [fx, 0, w/2],
                [0, fy, h/2],
                [0, 0, 1]
            ], dtype=np.float32)
            K = utils3d.numpy.normalize_intrinsics(K, w, h)
            # Convert disparity to depth
            depth = fx * stereo_baseline / disp
            if "chao" in frame_data['sequence_name']:
                depth_mask_inf = depth < -1
            else:
                # Mask: set depth > 256 to inf, else nan if invalid
                depth_mask_inf = depth > 1024
            depth_mask = ~depth_mask_inf
            depth = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth, np.nan))
            depth[depth == 0.0] = np.nan
            depth[disp == 0.0] = np.nan
            if "chao" in frame_data['sequence_name']:
                depth[depth > 1024] = np.nan
            # Camera pose (identity)
            c2w = np.eye(4, dtype=np.float32)
            # Save processed data
            save_path = Path(OUTPUT_DIR) / frame_data['sequence_name'] / f"{frame_data['frame_idx']}"
            save_path.mkdir(parents=True, exist_ok=True)
            write_image(save_path / 'image.jpg', rgb, quality=95)
            write_depth(save_path / 'depth.png', depth, unit=1)
            write_meta(save_path / 'meta.json', {
                'intrinsics': K.tolist(),
                'camera_pose': c2w.tolist()
            })
        except Exception as e:
            print(f"Error processing {frame_data['sequence_name']}/{frame_data['frame_idx']}: {e}")
        gc.collect()

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_single_frame, all_frames), total=len(all_frames)))

    write_index_file(OUTPUT_DIR, all_instance_paths)
