#!/usr/bin/env python3
# --------------------------------------------------------
# Preprocessing code for the BlendedMVS dataset
# dataset at https://github.com/YoYo000/BlendedMVS
# 1) Download BlendedMVS.zip
# 2) Download BlendedMVS+.zip
# 3) Download BlendedMVS++.zip
# 4) Unzip everything in the same directory
# 5) python blendedmvs_to_hdf5.py --root_dir /path/to/blendedMVS/ --out_dir /path/to/output_dir
# --------------------------------------------------------
import os
import os.path as osp
import re
import numpy as np
import argparse
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import sys
import warnings

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

sys.path.append(str(Path(__file__).absolute().parents[1]))
warnings.filterwarnings("ignore")
import utils3d
from moge.utils.io import write_depth, write_image, write_meta
from moge.utils.tools import multithead_execute, catch_exception


def list_sequences(db_root):
    """List all sequences in the BlendedMVS dataset."""
    print(">> Listing all sequences")
    sequences = [f for f in os.listdir(db_root) if len(f) == 24]
    assert sequences, f"Did not find any sequences at {db_root}"
    print(f"   (found {len(sequences)} sequences)")
    return sequences

def load_pfm_file(file_path):
    """Load a PFM file."""
    with open(file_path, "rb") as file:
        header = file.readline().decode("UTF-8").strip()

        if header == "PF":
            is_color = True
        elif header == "Pf":
            is_color = False
        else:
            raise ValueError("The provided file is not a valid PFM file.")

        dimensions = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("UTF-8"))
        if dimensions:
            img_width, img_height = map(int, dimensions.groups())
        else:
            raise ValueError("Invalid PFM header format.")

        endian_scale = float(file.readline().decode("UTF-8").strip())
        if endian_scale < 0:
            dtype = "<f"  # little-endian
        else:
            dtype = ">f"  # big-endian

        data_buffer = file.read()
        img_data = np.frombuffer(data_buffer, dtype=dtype)

        if is_color:
            img_data = np.reshape(img_data, (img_height, img_width, 3))
        else:
            img_data = np.reshape(img_data, (img_height, img_width))

        img_data = cv2.flip(img_data, 0)

    return img_data

def load_pose(path, ret_44=False):
    """Load camera pose from file."""
    with open(path) as f:
        RT = np.loadtxt(f, skiprows=1, max_rows=4, dtype=np.float32)
        assert RT.shape == (4, 4)
        RT = np.linalg.inv(RT)  # world2cam to cam2world

        f.seek(0)
        K = np.loadtxt(f, skiprows=7, max_rows=3, dtype=np.float32)
        assert K.shape == (3, 3)

    if ret_44:
        return K, RT
    return K, RT[:3, :3], RT[:3, 3]

def opengl_to_opencv(matrix):
    """
    convert from OpenGL coordinate system (X right, Y up, Z back) to OpenCV coordinate system (X right, Y down, Z front)
    """
    # create a transformation matrix to flip Y and Z
    flip = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],  # flip Y
        [0, 0, -1, 0],  # flip Z
        [0, 0, 0, 1]
    ])
    # apply the transformation
    return np.matmul(matrix, flip)

def normalize_intrinsics(K, width, height):
    """Normalize intrinsic matrix to have focal length relative to image size."""
    K_norm = K.copy()

    K_norm[0, 2] /= width
    K_norm[1, 2] /= height
    return K_norm

def collect_frame_data(db_root):
    """Collect all frames from the dataset."""
    all_frames = []
    
    # Get all sequences
    sequences = list_sequences(db_root)
    
    for seq_name in tqdm(sequences, desc="Scanning sequences"):
        seq_dir = osp.join(db_root, seq_name)
        cam_dir = osp.join(seq_dir, "cams")
        
        # Get all view names
        view_names = [f[:-8] for f in os.listdir(cam_dir) if not f.startswith("pair")]
        
        for view_idx, view_name in enumerate(view_names):
            # Store frame data
            frame_data = {
                'seq_name': seq_name,
                'view_name': view_name,
                'seq_dir': seq_dir,
                'sequence_name': seq_name,
                'frame_idx': view_idx
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
    parser = argparse.ArgumentParser(description="Preprocess BlendedMVS dataset.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the BlendedMVS dataset.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for processed data.")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    num_workers = 32
    
    print("Collecting frame data...")
    all_frames = collect_frame_data(args.root_dir)
    print(f"Found {len(all_frames)} frames to process")
    
    print("Processing frames...")
    all_instance_paths = [f"{frame['sequence_name']}/{frame['frame_idx']:04d}" 
                         for frame in all_frames]
    
    @multithead_execute(all_frames, num_workers=num_workers)
    @catch_exception
    def process_frames(frame_data):
        seq_dir = frame_data['seq_dir']
        view_name = frame_data['view_name']
        
            # Load camera parameters
        cam_path = osp.join(seq_dir, "cams", view_name + "_cam.txt")
        intrinsics, R_cam2world, t_cam2world = load_pose(cam_path)
        
        # Load RGB image
        img_path = osp.join(seq_dir, "blended_images", view_name + ".jpg")
        color_image = cv2.cvtColor(
            cv2.imread(img_path, cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB
        )
        
        # Load depth map
        depth_path = osp.join(seq_dir, "rendered_depth_maps", view_name + ".pfm")
        depthmap = load_pfm_file(depth_path)
        
        # Convert depth to float32 and handle invalid values
        depth_float32 = depthmap.astype(np.float32)
        
        # Handle extreme values similar to MidAir approach
        # Set very far values to infinity (e.g., beyond 1250 meters)
        depth_mask_inf = np.zeros_like(depth_float32).astype(bool)
        depth_mask = ~depth_mask_inf
        
        # Set far values to inf, invalid values to NaN
        depth_float32 = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth_float32, np.nan))
        
        # Set very close values (less than 0.001m) to NaN
        depth_float32[depth_float32 <= 0.001] = np.nan
        
        # Create camera-to-world matrix
        cam2world = np.eye(4)
        cam2world[:3, :3] = R_cam2world
        cam2world[:3, 3] = t_cam2world
        
        # Convert to opencv coordinate system
        # cam2world_opencv = opengl_to_opencv(cam2world)
        cam2world_opencv = cam2world
        
        # Normalize intrinsics
        h, w = color_image.shape[:2]
        K_norm = utils3d.numpy.normalize_intrinsics(intrinsics, w, h)
        
        # Save processed data
        save_path = Path(args.out_dir) / frame_data['sequence_name'] / f"{frame_data['frame_idx']:04d}"
        save_path.mkdir(parents=True, exist_ok=True)
        try:
            write_image(save_path / 'image.jpg', color_image, quality=100)
            write_depth(save_path / 'depth.png', depth_float32, unit=1)  # unit=1 means meters
            write_meta(save_path / 'meta.json', {
                'intrinsics': K_norm.tolist(),
                'camera_pose': cam2world_opencv.tolist()
            })
            
        except Exception as e:
            print(f"Error processing {frame_data['sequence_name']}/{frame_data['frame_idx']:04d}: {e}")
    
    write_index_file(args.out_dir, all_instance_paths)
    print(f"Processing complete. Results saved to {args.out_dir}")

if __name__ == "__main__":
    mp.freeze_support()
    main()