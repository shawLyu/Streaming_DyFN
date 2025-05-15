import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import gc
from scipy.spatial.transform import Rotation, Slerp
import warnings
import concurrent.futures
from typing import Union
warnings.filterwarnings("ignore")
import sys
sys.path.append(str(Path(__file__).absolute().parents[1]))
import utils3d

# You may need to adjust these utility imports
from moge.utils.io import write_depth, write_image, write_meta

DATASET_DIR = 'data/arkit/upsampling'
CAM_DIR = 'data/arkit/raw'
METADATA_CSV = os.path.join(DATASET_DIR, "metadata.csv")
OUTPUT_DIR = 'data/moge_process_dataset/arkit'
NUM_WORKERS = 64

def get_intrinsic_from_txt(intrinsics_path):
    intrinsics = np.loadtxt(intrinsics_path)
    W, H, fx, fy, cx, cy = intrinsics
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    return K, int(W), int(H)

def read_poses(traj_file):
    poses_timestamps = []
    poses = []
    with open(traj_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 7:
                    timestamp = float(parts[0])
                    rotation = [float(x) for x in parts[1:4]]
                    translation = [float(x) for x in parts[4:7]]
                    poses_timestamps.append(timestamp)
                    poses.append((rotation, translation))
    return poses_timestamps, poses

def interpolate_pose(poses_timestamps, poses, depth_timestamp):
    closest_idx = np.searchsorted(poses_timestamps, depth_timestamp)
    if closest_idx == 0:
        interpolated_pose = poses[0]
    elif closest_idx == len(poses_timestamps):
        interpolated_pose = poses[-1]
    else:
        prev_idx = closest_idx - 1
        next_idx = closest_idx
        t_prev = poses_timestamps[prev_idx]
        t_next = poses_timestamps[next_idx]
        pose_prev = poses[prev_idx]
        pose_next = poses[next_idx]
        alpha = (depth_timestamp - t_prev) / (t_next - t_prev)
        rot_prev = Rotation.from_rotvec(pose_prev[0])
        rot_next = Rotation.from_rotvec(pose_next[0])
        key_rots = Rotation.concatenate([rot_prev, rot_next])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        interpolated_rot = slerp([alpha])[0].as_rotvec()
        interpolated_trans = (1 - alpha) * np.array(pose_prev[1]) + alpha * np.array(pose_next[1])
        interpolated_pose = (interpolated_rot, interpolated_trans.tolist())
    return interpolated_pose

def pose_to_matrix(rotvec, translation):
    rot_matrix = Rotation.from_rotvec(rotvec).as_matrix()
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot_matrix
    c2w[:3, 3] = translation
    c2w = np.linalg.inv(c2w)
    return c2w

def collect_frame_data():
    df = pd.read_csv(METADATA_CSV)
    all_frames = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Scanning ARKit metadata"):
        video_id = row['video_id']
        fold = row['fold']
        sequence_name = f"{fold}_{video_id}"
        image_dir = os.path.join(DATASET_DIR, fold, str(video_id), "wide")
        depth_dir = os.path.join(DATASET_DIR, fold, str(video_id), "highres_depth")
        intrinsics_dir = os.path.join(CAM_DIR, fold, str(video_id), "wide_intrinsics")
        traj_file = os.path.join(CAM_DIR, fold, str(video_id), "lowres_wide.traj")
        if not os.path.exists(traj_file):
            continue
        try:
            image_list = sorted(os.listdir(image_dir))
            depth_list = sorted(os.listdir(depth_dir))
            intrinsics_list = sorted(os.listdir(intrinsics_dir))
        except Exception:
            continue
        if len(image_list) != len(depth_list):
            continue
        poses_timestamps, poses = read_poses(traj_file)
        for frame_idx, (image_path, depth_path, intrinsics_path) in enumerate(zip(image_list, depth_list, intrinsics_list)):
            all_frames.append({
                'sequence_name': sequence_name,
                'frame_idx': frame_idx,
                'image_path': os.path.join(image_dir, image_path),
                'depth_path': os.path.join(depth_dir, depth_path),
                'intrinsics_path': os.path.join(intrinsics_dir, intrinsics_path),
                'poses_timestamps': poses_timestamps,
                'poses': poses,
                'depth_timestamp': float(image_path.split('_')[-1][:-4])
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

def normalize_intrinsics(
    intrinsics: np.ndarray,
    width: Union[int, np.ndarray],
    height: Union[int, np.ndarray],
    integer_pixel_centers: bool = True
) -> np.ndarray:
    """
    Normalize intrinsics from pixel cooridnates to uv coordinates

    Args:
        intrinsics (np.ndarray): [..., 3, 3] camera intrinsics(s) to normalize
        width (int | np.ndarray): [...] image width(s)
        height (int | np.ndarray): [...] image height(s)
        integer_pixel_centers (bool): whether the integer pixel coordinates are at the center of the pixel. If False, the integer coordinates are at the left-top corner of the pixel.

    Returns:
        (np.ndarray): [..., 3, 3] normalized camera intrinsics(s)
    """
    zeros = np.zeros_like(width)
    ones = np.ones_like(width)
    if integer_pixel_centers:
        transform = np.stack([
            1 / width, zeros, 0.5 / width,
            zeros, 1 / height, 0.5 / height,
            zeros, zeros, ones
        ]).reshape(*zeros.shape, 3, 3)
    else:
        transform = np.stack([
            1 / width, zeros, zeros,
            zeros, 1 / height, zeros,
            zeros, zeros, ones
        ]).reshape(*zeros.shape, 3, 3)
    return transform @ intrinsics

def process_frame(frame_data):
    try:
        image = np.array(Image.open(frame_data['image_path']))
        depth = cv2.imread(frame_data['depth_path'], cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0  # meters
        depth_mask_inf = depth >= 20
        depth_mask = ~depth_mask_inf
        depth = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth, np.nan))
        depth[depth < 0.001] = np.nan

        K, w, h = get_intrinsic_from_txt(frame_data['intrinsics_path'])

        K = normalize_intrinsics(K, w, h, integer_pixel_centers=False)
        interpolated_pose = interpolate_pose(
            frame_data['poses_timestamps'],
            frame_data['poses'],
            frame_data['depth_timestamp']
        )
        c2w = pose_to_matrix(interpolated_pose[0], interpolated_pose[1])
        save_path = Path(OUTPUT_DIR) / f"{frame_data['sequence_name']}" / f"{frame_data['frame_idx']:05d}"
        save_path.mkdir(parents=True, exist_ok=True)
        write_image(save_path / 'image.jpg', image, quality=95)
        write_depth(save_path / 'depth.png', depth, unit=1)
        write_meta(save_path / 'meta.json', {
            'intrinsics': K.tolist(),
            'camera_pose': c2w.tolist()
        })
        gc.collect()
        return f"{frame_data['sequence_name']}/{frame_data['frame_idx']:05d}"
    except Exception as e:
        print(f"Error processing {frame_data['sequence_name']}/{frame_data['frame_idx']:05d}: {e}")
        return None

if __name__ == "__main__":
    print("Collecting frame data...")
    all_frames = collect_frame_data()
    print(f"Found {len(all_frames)} frames to process")

    all_instance_paths = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_frame, frame) for frame in all_frames]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing frames"):
            result = f.result()
            if result is not None:
                all_instance_paths.append(result)

    write_index_file(OUTPUT_DIR, all_instance_paths)