import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))
import cv2
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import warnings
import gc
import utils3d

from moge.utils.io import write_depth, write_image, write_meta

warnings.filterwarnings("ignore")

DATA_ROOT = 'data/kenburn3d/extracted'
OUTPUT_DIR = 'data/moge_process_dataset/kenburn3d_new'
NUM_WORKERS = 32

def load_exr_depth(exr_path):
    """Load depth from EXR file using OpenCV and scale as in hdf5 script."""
    depth = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"Failed to load EXR file: {exr_path}")
    if len(depth.shape) > 2 and depth.shape[2] > 1:
        depth = depth[:, :, 0]
    # Convert to meters (as in hdf5: depth/100)
    depth = depth.astype(np.float32) / 100.
    # Clip and mask as in hdf5
    depth_mask_inf = depth >= 600.0
    depth_mask = ~depth_mask_inf
    depth = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth, np.nan))
    depth[depth < 0.001] = np.nan
    return depth

def get_intrinsic_from_json(json_path):
    """Compute intrinsics from FOV as in hdf5 script."""
    with open(json_path, 'r') as f:
        meta = json.load(f)
    fltFov = meta['fltFov']
    fltFocal = 0.5 * 512 * np.tan(np.radians(90.0) - (0.5 * np.radians(fltFov)))
    K = np.array([
        [fltFocal, 0, 256],
        [0, fltFocal, 256],
        [0, 0, 1]
    ], dtype=np.float32)
    return K

def collect_frame_data(data_root):
    all_frames = []
    for scene_name in os.listdir(data_root):
        scene_path = os.path.join(data_root, scene_name)
        if not os.path.isdir(scene_path) or scene_name.endswith('-depth') or scene_name.endswith('-normal'):
            continue
        rgb_files = sorted([f for f in os.listdir(scene_path) if f.endswith('bl-image.png')])
        for idx, rgb_file in enumerate(rgb_files):
            rgb_path = os.path.join(scene_path, rgb_file)
            # Get corresponding depth path
            depth_path = rgb_path.replace(scene_name, f"{scene_name}-depth").replace('bl-image.png', 'bl-depth.exr')
            if not os.path.exists(depth_path):
                continue
            # Get corresponding meta json
            rgb_index = os.path.basename(rgb_path).split('-')[0]
            json_path = os.path.join(scene_path, f'{rgb_index}-meta.json')
            if not os.path.exists(json_path):
                continue
            sequence_name = scene_name
            frame_idx = idx
            all_frames.append({
                "image_path": rgb_path,
                "depth_path": depth_path,
                "json_path": json_path,
                "sequence_name": sequence_name,
                "frame_idx": frame_idx
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
    print("Collecting frame data...")
    all_frames = collect_frame_data(DATA_ROOT)
    print(f"Found {len(all_frames)} frames to process")

    print("Processing frames...")
    all_instance_paths = [f"{frame['sequence_name']}/{frame['frame_idx']:04d}" 
                         for frame in all_frames]

    import concurrent.futures

    def process_single_frame(frame_data):
        try:
            # Load RGB image
            rgb = np.array(Image.open(frame_data["image_path"])).astype(np.uint8)
            h, w = rgb.shape[:2]
            # Load depth map
            depth = load_exr_depth(frame_data["depth_path"])
            # Camera intrinsics
            K = get_intrinsic_from_json(frame_data["json_path"])
            K = utils3d.numpy.normalize_intrinsics(K, w, h)
            # Camera pose (identity, as in hdf5)
            c2w = np.eye(4, dtype=np.float32)
            # Save processed data
            save_path = Path(OUTPUT_DIR) / frame_data["sequence_name"] / f"{frame_data['frame_idx']:04d}"
            save_path.mkdir(parents=True, exist_ok=True)
            write_image(save_path / 'image.jpg', rgb, quality=95)
            write_depth(save_path / 'depth.png', depth, unit=1)
            write_meta(save_path / 'meta.json', {
                'intrinsics': K.tolist(),
                'camera_pose': c2w.tolist()
            })
        except Exception as e:
            print(f"Error processing {frame_data['sequence_name']}/{frame_data['frame_idx']:04d}: {e}")
        gc.collect()

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_single_frame, all_frames), total=len(all_frames)))

    write_index_file(OUTPUT_DIR, all_instance_paths)
