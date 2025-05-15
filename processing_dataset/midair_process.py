import os
import cv2
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
import warnings
import sys
import h5py
sys.path.append(str(Path(__file__).absolute().parents[1]))

warnings.filterwarnings("ignore")
import utils3d

from moge.utils.io import write_depth, write_image, write_meta
from moge.utils.tools import multithead_execute, catch_exception

DOWNLOAD_DIR = 'data/midair/MidAir_extracted'
OUTPUT_DIR = 'data/moge_process_dataset/midair'
NUM_WORKERS = 64

def quaternion_to_matrix(qx, qy, qz, qw):
    """Convert a quaternion into a rotation matrix."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    R = np.array([
        [1.0 - 2.0*(yy + zz), 2.0*(xy - wz),       2.0*(xz + wy)],
        [2.0*(xy + wz),       1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
        [2.0*(xz - wy),       2.0*(yz + wx),       1.0 - 2.0*(xx+yy)]
    ], dtype=np.float32)
    return R

def pose_to_matrix(position, quaternion):
    """Convert position and quaternion to a 4x4 transformation matrix."""
    qw, qx, qy, qz = quaternion
    R_ned2cv = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    R = quaternion_to_matrix(qx, qy, qz, qw) @ R_ned2cv
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R
    mat[:3, 3] = position
    return mat

def open_float16(image_path):
    pic = Image.open(image_path)
    img = np.asarray(pic, np.uint16)
    img.dtype = np.float16
    return img

def get_intrinsic(w, h, f=512):
    """Return the camera intrinsic matrix for MidAir."""
    K = np.eye(3, dtype=np.float32)
    K[0,0] = f
    K[1,1] = f
    K[0,2] = w / 2
    K[1,2] = h / 2
    return K

def collect_frame_data(base_dir):
    """Collect all frames from all sensor_records.hdf5 files, following midair_hdf5.py logic."""
    all_frames = []
    sensor_files = find_sensor_records(base_dir)
    for database_path in tqdm(sensor_files, desc="Scanning sensor_records.hdf5"):
        # Extract environment info from path
        rel_path = os.path.relpath(database_path, base_dir)
        env_parts = rel_path.split(os.sep)
        env_name = '/'.join(env_parts[:-1])  # Use path-style joining

        try:
            database = h5py.File(database_path, "r")
        except Exception as e:
            print(f"Could not open {database_path}: {e}")
            continue

        for sequence_name in database:
            seq_data = database[sequence_name]
            color_data = seq_data["camera_data"]["color_left"]
            depth_data = seq_data["camera_data"]["depth"]
            position_data = seq_data["groundtruth"]["position"]
            attitude_data = seq_data["groundtruth"]["attitude"]

            for i in range(len(color_data)):
                frame_data = {
                    'rgb_path': os.path.join(DOWNLOAD_DIR, env_name, color_data[i].decode("utf-8")),
                    'depth_path': os.path.join(DOWNLOAD_DIR, env_name, depth_data[i].decode("utf-8")),
                    'position': position_data[i * 4, :],
                    'quaternion': attitude_data[i * 4, :],
                    'sequence_name': f"{env_name.replace('/', '_')}_{sequence_name}",
                    'frame_idx': i
                }
                all_frames.append(frame_data)
        database.close()
    return all_frames

def write_index_file(output_dir, instance_paths):
    """Write index.txt containing all existing instance paths."""
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

def find_sensor_records(root_dir):
    """Find all sensor_records.hdf5 files in the dataset directory."""
    sensor_files = []
    for root, dirs, files in os.walk(root_dir):
        if 'sensor_records.hdf5' in files:
            sensor_files.append(os.path.join(root, 'sensor_records.hdf5'))
    return sensor_files

if __name__ == "__main__":
    print("Collecting frame data...")
    all_frames = collect_frame_data(DOWNLOAD_DIR)
    print(f"Found {len(all_frames)} frames to process")

    print("Processing frames...")
    all_instance_paths = [f"{frame['sequence_name']}/{frame['frame_idx']:04d}" 
                         for frame in all_frames]

    @multithead_execute(all_frames, num_workers=NUM_WORKERS)
    @catch_exception
    def process_frames(frame_data):
        # Load RGB image
        rgb = np.array(Image.open(frame_data['rgb_path']))
        h, w = rgb.shape[:2]
        # Load depth map
        depth = open_float16(frame_data['depth_path']).astype(np.float32)
        # Convert depth to zc as in hdf5 script
        f = 512
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        dx = u - w / 2
        dy = v - h / 2
        norm = np.sqrt(dx**2 + dy**2 + f**2)
        rxy = depth / norm
        xc = rxy * dx
        yc = rxy * dy
        zc = rxy * f
        depth_map = zc

        depth_mask_inf = depth >= 1250
        depth_mask = ~depth_mask_inf
        depth_map = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth_map, np.nan))
        depth_map[depth_map < 0.001] = np.nan
        # Get camera pose
        c2w = pose_to_matrix(frame_data['position'], frame_data['quaternion'])
        # Camera intrinsics
        K = get_intrinsic(w, h, f)
        K = utils3d.numpy.normalize_intrinsics(K, w, h)
        # Save processed data
        save_path = Path(OUTPUT_DIR) / f"{frame_data['sequence_name']}" / f"{frame_data['frame_idx']:04d}"
        save_path.mkdir(parents=True, exist_ok=True)
        try:
            write_image(save_path / 'image.jpg', rgb, quality=95)
            write_depth(save_path / 'depth.png', depth_map, unit=1)
            write_meta(save_path / 'meta.json', {
                'intrinsics': K.tolist(),
                'camera_pose': c2w.tolist()
            })
        except Exception as e:
            print(f"Error processing {frame_data['sequence_name']}/{frame_data['frame_idx']:04d}: {e}")

    write_index_file(OUTPUT_DIR, all_instance_paths)
