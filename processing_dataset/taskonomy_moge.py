#!/usr/bin/env python3
# --------------------------------------------------------
# Preprocessing code for the Taskonomy dataset
# --------------------------------------------------------
import sys
import os
import os.path as osp
import json
import argparse
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import PIL.Image
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from scipy.spatial.transform import Rotation

sys.path.append(str(Path(__file__).absolute().parents[1]))
warnings.filterwarnings("ignore")
import utils3d
from moge.utils.io import write_depth, write_image, write_meta
from moge.utils.tools import multithead_execute, catch_exception

# Constants
MAX_DEPTH = 65535  # Maximum value for uint16
DEPTH_SCALE = 512  # Scale factor for depth values (Taskonomy uses 1/512m units)
MAX_RANGE = 100    # Maximum range in meters for Taskonomy depth

def list_point_views(taskonomy_dir):
    """List all point-view pairs in the dataset."""
    point_info_dirs = glob(osp.join(taskonomy_dir, "*point_info"))
    if len(point_info_dirs) == 0:
        print(f"Error: point_info directory not found in {taskonomy_dir}")
        return []
    
    point_views = []
    for point_info_dir in point_info_dirs:
        for item in os.listdir(point_info_dir + "/point_info"):
            if item.endswith(".json") and os.path.isfile(os.path.join(point_info_dir, "point_info", item)):
                # Extract point and view IDs from filename
                # Format: point_X_view_Y_domain_point_info.json
                parts = item.split("_")
                if len(parts) >= 4 and parts[0] == "point" and parts[2] == "view":
                    try:
                        point_id = parts[1]
                        view_id = parts[3]
                        point_views.append((point_info_dir, point_id, view_id, item))
                    except:
                        continue
    
    return sorted(point_views)

def load_camera_data(point_file, debug=False):
    """Load camera data from a point JSON file using OpenCV conventions."""
    with open(point_file, 'r') as f:
        data = json.load(f)
    
    if debug:
        print(f"Raw camera data keys: {data.keys()}")
    
    # Extract camera parameters
    resolution = data.get('resolution', 512)
    fov_rads = data.get('field_of_view_rads', 1.0)
    camera_location = np.array(data.get('camera_location', [0, 0, 0]))
    camera_rotation = np.array(data.get('camera_rotation_final', [0, 0, 0]))
    
    if debug:
        print(f"Camera location: {camera_location}")
        print(f"Camera rotation (radians): {camera_rotation}")
        print(f"Field of view (radians): {fov_rads}")
        print(f"Resolution: {resolution}")
    
    # Compute focal length from field of view
    focal_length = (resolution / 2) / np.tan(fov_rads / 2)
    
    # Create intrinsic matrix
    K = np.array([
        [focal_length, 0, resolution / 2],
        [0, focal_length, resolution / 2],
        [0, 0, 1]
    ])
    
    # Convert Euler angles to rotation matrix
    r = Rotation.from_euler('xyz', camera_rotation)
    rotation_matrix = r.as_matrix()
    
    # Taskonomy uses a coordinate system where:
    # - X is right
    # - Y is up
    # - Z is backward (from the camera)
    
    # OpenCV uses:
    # - X is right
    # - Y is down
    # - Z is forward (into the scene)
    
    # Convert from Taskonomy to OpenCV coordinate system
    # Need to flip Y and Z axes
    conversion = np.array([
        [1, 0, 0],
        [0, -1, 0],  # Flip Y axis
        [0, 0, -1]   # Flip Z axis
    ])
    
    # Apply conversion to rotation
    R = rotation_matrix @ conversion
    
    # Create camera-to-world transformation matrix
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = camera_location
    
    if debug:
        print(f"Intrinsic matrix K:\n{K}")
        print(f"Rotation matrix R:\n{R}")
        print(f"Camera-to-world matrix:\n{c2w}")
    
    return K, c2w, resolution


def process_depth(depth_array):
    """Process depth array to handle invalid values and convert to meters."""
    # Convert to float32
    depth = depth_array.astype(np.float32)
    
    # Taskonomy depth is stored as 1/512 meters
    depth_map = depth / DEPTH_SCALE
    
    # Handle invalid values similar to MidAir approach
    depth_mask_inf = depth_map >= MAX_RANGE  # Values too far (>128m)
    depth_mask = ~depth_mask_inf
    
    # Set far values to inf, invalid values to NaN
    depth_map = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth_map, np.nan))
    
    # Set very close values to NaN
    depth_map[depth_map < 0.001] = np.nan
    
    return depth_map

def collect_frame_data(taskonomy_dir):
    """Collect all frames from the dataset."""
    all_frames = []
    point_views = list_point_views(taskonomy_dir)
    
    for point_info_dir, point_id, view_id, json_file in tqdm(point_views, desc="Scanning dataset"):
        scene_id = point_info_dir.split("/")[-1].split("_")[0]
        point_file = os.path.join(point_info_dir, "point_info", json_file)
        rgb_path = os.path.join(taskonomy_dir, scene_id + "_rgb", "rgb", f"point_{point_id}_view_{view_id}_domain_rgb.png")
        depth_path = os.path.join(taskonomy_dir, scene_id + "_depth_zbuffer", "depth_zbuffer", f"point_{point_id}_view_{view_id}_domain_depth_zbuffer.png")
        
        # Skip if any required file is missing
        if not all(os.path.exists(p) for p in [point_file, rgb_path, depth_path]):
            continue
            
        frame_data = {
            'rgb_path': rgb_path,
            'depth_path': depth_path,
            'point_file': point_file,
            'sequence_name': f"{scene_id}",
            'frame_idx': int(point_id + view_id)
        }
        all_frames.append(frame_data)
    
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

def main():
    parser = argparse.ArgumentParser(description="Preprocess Taskonomy dataset.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the Taskonomy dataset.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for processed data.")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Collecting frame data...")
    all_frames = collect_frame_data(args.root_dir)
    print(f"Found {len(all_frames)} frames to process")
    
    print("Processing frames...")
    all_instance_paths = [f"{frame['sequence_name']}/{frame['frame_idx']:04d}" 
                         for frame in all_frames]
    
    @multithead_execute(all_frames, num_workers=32)
    @catch_exception
    def process_frames(frame_data):
        # Load RGB image
        rgb = np.array(PIL.Image.open(frame_data['rgb_path']))
        
        # Load depth map
        depth_img = PIL.Image.open(frame_data['depth_path'])
        depth_map = process_depth(np.array(depth_img))
        
        # Get camera data
        K, c2w, resolution = load_camera_data(frame_data['point_file'])
        
        # Normalize intrinsics
        h, w = rgb.shape[:2]
        K_norm = utils3d.numpy.normalize_intrinsics(K, w, h)
        
        # Save processed data
        save_path = Path(args.out_dir) / f"{frame_data['sequence_name']}" / f"{frame_data['frame_idx']:04d}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            write_image(save_path / 'image.jpg', rgb, quality=95)
            write_depth(save_path / 'depth.png', depth_map, unit=1)
            write_meta(save_path / 'meta.json', {
                'intrinsics': K_norm.tolist(),
                'camera_pose': c2w.tolist()
            })
        except Exception as e:
            print(f"Error processing {frame_data['sequence_name']}/{frame_data['frame_idx']:04d}: {e}")
    
    write_index_file(args.out_dir, all_instance_paths)
    print(f"Processing complete. Results saved to {args.out_dir}")

if __name__ == "__main__":
    mp.freeze_support()
    main()