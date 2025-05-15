import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image
from glob import glob
from typing import *
import warnings
import utils3d
warnings.filterwarnings("ignore")

from moge.utils.io import write_depth, write_image, write_meta
from moge.utils.tools import multithead_execute, catch_exception

seg_sky_dict = {
    "amusement": 182,
}

DOWNLOAD_DIR = '/home/luban/dataset/tartanair'
OUTPUT_DIR = 'data/moge_process_dataset/tartanair_correct'
NUM_WORKERS = 64

def get_intrinsic() -> np.ndarray:
    """Return the fixed camera intrinsic matrix for TartanAir."""
    K = np.eye(3, dtype=np.float32)
    K[0,0] = 320.0
    K[1,1] = 320.0
    K[0,2] = 320.0
    K[1,2] = 240.0
    return K

def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert pose from [tx, ty, tz, qx, qy, qz, qw] to a 4x4 transformation matrix."""
    tx, ty, tz, qx, qy, qz, qw = pose
    R = quaternion_to_matrix(qx, qy, qz, qw)

    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R
    mat[:3, 3] = [tx, ty, tz]

    T = np.array([[0,1,0,0],
                    [0,0,1,0],
                    [1,0,0,0],
                    [0,0,0,1]]).astype(np.float32)
    T_inv = np.linalg.inv(T)
    mat = T@mat@T_inv
    return mat

def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
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

def collect_frame_data(data_dir: str) -> List[Dict]:
    """Collect all frame data paths and metadata."""
    all_frames = []
    intrinsics = get_intrinsic()

    # Walk through the directory structure
    for env in os.listdir(data_dir):
        env_path = os.path.join(data_dir, env)
        if env != "amusement":
            continue
        if not os.path.isdir(env_path):
            continue
            
        for diff in os.listdir(env_path):
            diff_path = os.path.join(env_path, diff)
            if not os.path.isdir(diff_path):
                continue
                
            for traj in tqdm(os.listdir(diff_path), desc=f"Collecting {env}/{diff}"):
                traj_path = os.path.join(diff_path, traj)
                if not os.path.isdir(traj_path):
                    continue

                sequence_name = f"{env}/{diff}_{traj}"
                
                # Check required directories and files
                image_dir = os.path.join(traj_path, "image_left")
                depth_dir = os.path.join(traj_path, "depth_left")
                seg_dir = os.path.join(traj_path, "seg_left")
                pose_file = os.path.join(traj_path, "pose_left.txt")
                
                if not all(os.path.exists(p) for p in [image_dir, depth_dir, pose_file, seg_dir]):
                    print(f"Skipping {sequence_name}: Missing required files")
                    continue

                try:
                    poses = np.loadtxt(pose_file)  # (N, 7): [tx ty tz qx qy qz qw]
                except:
                    print(f"Skipping {sequence_name}: Could not load poses")
                    continue

                # Get all image files and sort them
                rgb_files = sorted(glob(os.path.join(image_dir, "*_left.png")))
                
                # Collect frame data
                for rgb_path in rgb_files:
                    frame_idx = int(os.path.basename(rgb_path).split('_')[0])
                    depth_path = os.path.join(depth_dir, f"{frame_idx:06d}_left_depth.npy")
                    seg_path = os.path.join(seg_dir, f"{frame_idx:06d}_left_seg.npy")
                    if not os.path.exists(depth_path) or not os.path.exists(seg_path):
                        continue

                    save_path = Path(OUTPUT_DIR) / sequence_name / f"{frame_idx:04d}"
                    
                    # Skip if already processed
                    if os.path.exists(save_path / 'image.jpg') and \
                       os.path.exists(save_path / 'depth.png') and \
                       os.path.exists(save_path / 'meta.json'):
                        continue

                    all_frames.append({
                        'rgb_path': rgb_path,
                        'depth_path': depth_path,
                        'seg_path': seg_path,
                        'pose': poses[frame_idx],
                        'sequence_name': sequence_name,
                        'frame_idx': frame_idx,
                        'intrinsics': intrinsics,
                        'save_path': save_path,
                        'env': env,
                    })

    return all_frames

def write_index_file(output_dir: str, instance_paths: List[str]):
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

if __name__ == "__main__":
    # Collect all frame data first
    print("Collecting frame data...")
    all_frames = collect_frame_data(DOWNLOAD_DIR)
    print(f"Found {len(all_frames)} frames to process")

    # Process frames in parallel
    print("Processing frames...")
    all_instance_paths = [f"{frame['sequence_name']}/{frame['frame_idx']:04d}" 
                         for frame in all_frames]
    
    # Fix: Use the decorator pattern consistently
    @multithead_execute(all_frames, num_workers=NUM_WORKERS)
    @catch_exception
    def process_frames(frame_data: Dict):
        """Process a single frame with the given data."""
        # Load RGB image
        rgb = np.array(Image.open(frame_data['rgb_path']))

        # Load seg array
        seg = np.load(frame_data['seg_path'])
        if frame_data['env'] == "amusement":
            sky_mask = seg == seg_sky_dict[frame_data['env']]
        else:
            sky_mask = np.ones_like(seg, dtype=np.bool)
        
        # Load depth map
        depth = np.load(frame_data['depth_path']).astype(np.float32)
        depth_mask_inf = depth >= 655.35 | sky_mask
        depth_mask = ~depth_mask_inf
        depth = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth, np.nan))
        depth[depth < 0.001] = np.nan
        # Get camera pose
        c2w = pose_to_matrix(frame_data['pose'])
        
        # Normalize intrinsics
        intrinsics_saved = utils3d.numpy.normalize_intrinsics(
            frame_data['intrinsics'], 
            rgb.shape[1], 
            rgb.shape[0]
        )
        
        # Save processed data
        save_path = frame_data['save_path']
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            write_image(save_path / 'image.jpg', rgb, quality=95)
            write_depth(save_path / 'depth.png', depth, unit=1)
            write_meta(save_path / 'meta.json', {
                'intrinsics': intrinsics_saved.tolist(),
                'camera_pose': c2w.tolist()
            })
        except Exception as e:
            print(f"Error processing {frame_data['sequence_name']}/{frame_data['frame_idx']:04d}: {e}")

    # Write index file after processing all scenes
    # write_index_file(OUTPUT_DIR, all_instance_paths)
