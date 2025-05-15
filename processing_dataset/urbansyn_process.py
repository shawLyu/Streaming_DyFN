import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))
import numpy as np
from PIL import Image
import glob
import gc
import warnings
import json
from tqdm import tqdm
import OpenEXR
import cv2
import utils3d

from moge.utils.io import write_depth, write_image, write_meta

warnings.filterwarnings("ignore")

DATA_ROOT = "data/urban_sync"
OUTPUT_DIR = "data/moge_process_dataset/urban_sync"
CAMERA_METADATA_PATH = "data/urban_sync/camera_metadata.json"
NUM_WORKERS = 32

def load_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    channels = header['channels'].keys()
    data = {}
    for channel in channels:
        channel_data = exr_file.channel(channel)
        data[channel] = np.frombuffer(channel_data, dtype=np.float32)
    exr_file.close()
    return data

def compute_camera_intrinsics(metadata_file, image_width, image_height):
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    camera_params = metadata['parameters'][0]['Camera']
    focal_length = camera_params['focalLength_mm']
    sensor_width = camera_params['sensorWidth_mm']
    sensor_height = camera_params['sensorHeight_mm']
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = focal_length / sensor_width * image_width
    K[1, 1] = focal_length / sensor_height * image_height
    K[0, 2] = image_width / 2
    K[1, 2] = image_height / 2
    return K

def collect_frame_data(data_root):
    all_frames = []
    depth_dir = os.path.join(data_root, "depth")
    image_dir = os.path.join(data_root, "rgb")
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.exr")))
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    assert len(depth_files) == len(image_files), "Depth and image files number mismatch"
    for idx, (depth_file, image_file) in enumerate(zip(depth_files, image_files)):
        sequence_name = f"UrbanSync"
        frame_idx = f"{idx:03d}"
        all_frames.append({
            "image_path": image_file,
            "depth_path": depth_file,
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
    all_instance_paths = [f"{frame['sequence_name']}/{frame['frame_idx']}" 
                         for frame in all_frames]

    import concurrent.futures

    def process_single_frame(frame_data):
        try:
            # Load RGB image
            rgb = Image.open(frame_data["image_path"])
            w, h = rgb.size
            rgb_np = np.array(rgb)
            # Load depth
            depth = cv2.imread(frame_data["depth_path"], cv2.IMREAD_UNCHANGED)
            if depth is None:
                # Attempt fallback if there's a '.exr.1' file
                alt_exr_1 = depth_file + ".1"
                if os.path.exists(alt_exr_1):
                    temp_exr = depth_file.replace(".exr", "_tmp.exr")
                    os.rename(alt_exr_1, temp_exr)
                    depth = cv2.imread(temp_exr, cv2.IMREAD_UNCHANGED)
                    if depth is None:
                        return f"Error reading depth file (fallback) {temp_exr}"
                    depth *= 1e5
                else:
                    return f"Error reading depth file {depth_file}"
            else:
                depth *= 1e5  # multiply by 1e5, consistent with your original code
            # Mask: set depth >= 65500 to inf, else nan if invalid
            depth_mask_inf = depth >= 256
            depth_mask = ~depth_mask_inf
            depth = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth, np.nan))
            depth[depth < 0.001] = np.nan
            # Camera intrinsics
            K = compute_camera_intrinsics(CAMERA_METADATA_PATH, w, h)
            K = utils3d.numpy.normalize_intrinsics(K, w, h)
            # Camera pose (identity)
            c2w = np.eye(4, dtype=np.float32)
            # Save processed data
            save_path = Path(OUTPUT_DIR) / frame_data["sequence_name"] / f"{frame_data['frame_idx']}"
            save_path.mkdir(parents=True, exist_ok=True)
            write_image(save_path / 'image.jpg', rgb_np, quality=95)
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
