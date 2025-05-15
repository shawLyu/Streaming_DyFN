import os
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
import Imath
import utils3d

from moge.utils.io import write_depth, write_image, write_meta

warnings.filterwarnings("ignore")

DATA_ROOT = "/home/luban/dataset/mvs_sync/GTAV_1080"  # Set your MVS-Synth root directory
OUTPUT_DIR = "data/moge_process_dataset/mvs_synth"
NUM_WORKERS = 32

def list_all_sequences(data_dir):
    """List all sequences in the dataset."""
    sequences = []
    for seq_dir in sorted(os.listdir(data_dir)):
        seq_path = os.path.join(data_dir, seq_dir)
        if os.path.isdir(seq_path):
            sequences.append(seq_dir)
    return sequences

def load_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(OpenEXR.FLOAT)
    channels = list(header['channels'].keys())
    if len(channels) == 1:
        channel_data = exr_file.channel(channels[0], pt)
        return np.frombuffer(channel_data, dtype=np.float32).reshape(height, width)
    elif len(channels) == 3:
        try:
            red = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
            green = np.frombuffer(exr_file.channel('G', pt), dtype=np.float32)
            blue = np.frombuffer(exr_file.channel('B', pt), dtype=np.float32)
        except KeyError:
            red = np.frombuffer(exr_file.channel(channels[0], pt), dtype=np.float32)
            green = np.frombuffer(exr_file.channel(channels[1], pt), dtype=np.float32)
            blue = np.frombuffer(exr_file.channel(channels[2], pt), dtype=np.float32)
        return np.dstack((
            red.reshape(height, width),
            green.reshape(height, width),
            blue.reshape(height, width)
        ))
    raise ValueError(f"Unsupported EXR format: {len(channels)} channels detected")

def load_pose_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if 'extrinsic' in data:
        extrinsics = np.array(data['extrinsic'], dtype=np.float32)
    else:
        extrinsics = np.eye(4, dtype=np.float32)
    if all(k in data for k in ['f_x', 'f_y', 'c_x', 'c_y']):
        fx = data['f_x']
        fy = data['f_y']
        cx = data['c_x']
        cy = data['c_y']
        intrinsics = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
    else:
        intrinsics = np.array([
            [1158.0, 0.0, 960.0],
            [0.0, 1158.0, 540.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
    rotation_matrix = extrinsics[:3, :3]
    det = np.linalg.det(rotation_matrix)
    if det < 0:
        extrinsics[:3, :3] = -rotation_matrix
    w2c = extrinsics
    c2w = np.linalg.inv(w2c)
    return c2w, intrinsics

def collect_frame_data(data_root):
    all_frames = []
    sequences = list_all_sequences(data_root)
    for sequence in tqdm(sequences, desc="Collecting sequences"):
        sequence_path = os.path.join(data_root, sequence)
        images_dir = os.path.join(sequence_path, "images")
        depths_dir = os.path.join(sequence_path, "depths")
        poses_dir = os.path.join(sequence_path, "poses")
        if not all(os.path.exists(d) for d in [images_dir, depths_dir, poses_dir]):
            continue
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        depth_files = sorted(glob.glob(os.path.join(depths_dir, "*.exr")))
        pose_files = sorted(glob.glob(os.path.join(poses_dir, "*.json")))
        if not (len(image_files) == len(depth_files) == len(pose_files)):
            continue
        sequence_name = f"mvs_synth_{sequence}"
        for idx, (img_path, depth_path, pose_path) in enumerate(zip(image_files, depth_files, pose_files)):
            all_frames.append({
                "image_path": img_path,
                "depth_path": depth_path,
                "pose_path": pose_path,
                "sequence_name": sequence_name,
                "frame_idx": idx
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
    all_instance_paths = [f"{frame['sequence_name']}/{frame['frame_idx']:05d}" 
                         for frame in all_frames]

    import concurrent.futures

    def process_single_frame(frame_data):
        try:
            # Load RGB image
            rgb = np.array(Image.open(frame_data["image_path"]))
            h, w = rgb.shape[:2]
            # Load depth map
            depth = load_exr(frame_data["depth_path"]).copy()
            depth_mask_inf = depth > 1000
            depth_mask = ~depth_mask_inf
            depth = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth, np.nan))
            depth[depth < 0.001] = np.nan
            # Load pose and intrinsics
            c2w, K = load_pose_from_json(frame_data["pose_path"])
            K = utils3d.numpy.normalize_intrinsics(K, w, h)
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
