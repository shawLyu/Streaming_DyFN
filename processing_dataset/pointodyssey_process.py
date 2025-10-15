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
sys.path.append(str(Path(__file__).absolute().parents[1]))

warnings.filterwarnings("ignore")
import utils3d

from moge.utils.io import write_depth, write_image, write_meta
from moge.utils.tools import multithead_execute, catch_exception

DOWNLOAD_DIR = 'data/pointodyssey'
OUTPUT_DIR = 'data/moge_process_dataset/pointodyssey'
NUM_WORKERS = 64

def get_intrinsic(w, h, fx=960, fy=540):
    """Return the camera intrinsic matrix for PointOdyssey."""
    K = np.eye(3, dtype=np.float32)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = w / 2
    K[1,2] = h / 2
    return K

def depth_read(path):
    """Read depth from PointOdyssey format."""
    depth16 = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    depth = depth16.astype(np.float32) / 65535.0 * 1000.0  # 1000 is the max depth in the dataset
    
    # pixels where depth==0 are invalid (indoor scenes, no sky)
    min_depth = 0.01  # 1cm, reasonable value based on observations
    invalid_mask = depth == 0
    depth[invalid_mask] = np.nan
    
    # Set invalid depths to nan
    depth[depth < min_depth] = np.nan
    depth[depth > 1000] = np.nan
    
    return depth

def collect_frame_data(base_dir):
    """Collect all frames from PointOdyssey dataset."""
    all_frames = []
    
    # Selected scenes from the original code
    cut3r_selected_scenes = [
        'cnb_dlab_0215_3rd', 'cnb_dlab_0215_ego1',
        'cnb_dlab_0225_3rd', 'cnb_dlab_0225_ego1',
        'dancing', 'dancingroom0_3rd', 'footlab_3rd',
        'footlab_ego1', 'footlab_ego2', 'girl', 'girl_egocentric',
        'human_egocentric', 'human_in_scene', 'human_in_scene1',
        'kg', 'kg_ego1', 'kg_ego2',
        'kitchen_gfloor', 'kitchen_gfloor_ego1', 'kitchen_gfloor_ego2',
        'scene_carb_h_tables', 'scene_carb_h_tables_ego1', 'scene_carb_h_tables_ego2',
        'scene_j716_3rd', 'scene_j716_ego1', 'scene_j716_ego2',
        'scene_recording_20210910_S05_S06_0_3rd', 'scene_recording_20210910_S05_S06_0_ego2',
        'scene1_0129', 'scene1_0129_ego',
        'seminar_h52_3rd', 'seminar_h52_ego1', 'seminar_h52_ego2'
    ]
    
    train_dir = os.path.join(base_dir, 'train')
    if not os.path.exists(train_dir):
        print(f"Training directory not found: {train_dir}")
        return all_frames
    
    for scene_name in tqdm(cut3r_selected_scenes, desc="Scanning scenes"):
        scene_path = os.path.join(train_dir, scene_name)
        if not os.path.exists(scene_path):
            continue
            
        rgb_path = os.path.join(scene_path, 'rgbs')
        depth_path = os.path.join(scene_path, 'depths')
        anno_path = os.path.join(scene_path, "anno.npz")

        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            continue
            
        # Get sorted image files
        all_imgs = [f for f in os.listdir(rgb_path) if f.endswith('.jpg')]
        sorted_imgs = sorted(all_imgs, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        anno = np.load(anno_path)
        cam_ints = anno['intrinsics'].astype(np.float32)
        cam_exts = anno['extrinsics'].astype(np.float32)
        
        for i, img_name in enumerate(sorted_imgs):
            depth_name = img_name.replace('.jpg', '.png').replace('rgb', 'depth')
            
            rgb_file_path = os.path.join(rgb_path, img_name)
            depth_file_path = os.path.join(depth_path, depth_name)
            
            if not os.path.exists(depth_file_path):
                continue
                
            frame_idx = int(os.path.basename(img_name).split('_')[1].split('.')[0])
            
            frame_data = {
                'rgb_path': rgb_file_path,
                'depth_path': depth_file_path,
                'sequence_name': scene_name,
                'frame_idx': frame_idx,
                "K": cam_ints[i],
                "c2w": cam_exts[i]
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
        depth_map = depth_read(frame_data['depth_path'])
        
        # Camera intrinsics (using default values for PointOdyssey)
        K = frame_data["K"]
        K = utils3d.numpy.normalize_intrinsics(K, w, h)
        
        # For PointOdyssey, we don't have camera poses, so use identity
        c2w = frame_data["c2w"]
        
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