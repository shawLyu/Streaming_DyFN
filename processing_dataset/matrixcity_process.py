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
import concurrent.futures

from moge.utils.io import write_depth, write_image, write_meta
from moge.utils.tools import multithead_execute, catch_exception

warnings.filterwarnings("ignore")

category = ["big_city", "small_city"]

DATA_ROOT = "data/matrixcity"
OUTPUT_DIR = "data/moge_process_dataset/matrixcity_test"
NUM_WORKERS = 80

def load_exr_depth(exr_path):
    """Load depth from EXR file using OpenCV."""
    depth = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"Failed to load EXR file: {exr_path}")
    if len(depth.shape) > 2 and depth.shape[2] > 1:
        depth = depth[:, :, 0]
    return depth

def compute_intrinsics_from_fov(width, height, fov_x):
    fx = width / (2 * np.tan(fov_x / 2))
    fy = fx
    cx = width / 2
    cy = height / 2
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    return K

def collect_frame_data(data_root):
    all_frames = []
    for cat in category:
        cat_path = os.path.join(data_root, cat)
        depth_path = cat_path + "_depth"
        if not os.path.exists(cat_path):
            continue
        for view in os.listdir(cat_path):
            view_path = os.path.join(cat_path, view)
            if not os.path.isdir(view_path):
                continue
            for split in ["train", "test"]:
                split_path = os.path.join(view_path, split)
                if not os.path.isdir(split_path):
                    continue
                for scene in os.listdir(split_path):
                    scene_path = os.path.join(split_path, scene)
                    if "aerial" not in scene:
                        depth_scene_path = os.path.join(depth_path, view, split, scene) + "_depth"
                    else:
                        depth_scene_path = os.path.join(depth_path + "_float32", view, split, scene) + "_depth"
                    if not os.path.isdir(scene_path) or not os.path.isdir(depth_scene_path):
                        continue
                    transform_path = os.path.join(scene_path, "transforms.json")
                    if not os.path.exists(transform_path):
                        continue
                    with open(transform_path, "r") as f:
                        transform = json.load(f)
                    camera_angle_x = transform["camera_angle_x"]
                    frames = transform["frames"]
                    sequence_name = f"matrix_city_{cat}_{view}_{split}_{scene}"
                    for frame_idx, frame in enumerate(frames):
                        frame_index = frame["frame_index"]
                        transform_matrix = np.array(frame["rot_mat"], dtype=np.float32)
                        transform_matrix[:3, :3] *= 100
                        transform_matrix[:, 1:3] *= -1
                        image_path = os.path.join(scene_path, f"{frame_index:04d}.png")
                        depth_file = os.path.join(depth_scene_path, f"{frame_index:04d}.exr")
                        if not os.path.exists(image_path) or not os.path.exists(depth_file):
                            continue
                        all_frames.append({
                            "image_path": image_path,
                            "depth_path": depth_file,
                            "K": None,  # to be filled in process_frames
                            "c2w": transform_matrix.tolist(),
                            "camera_angle_x": camera_angle_x,
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
    saved_json_path = os.path.join("./frame_list/matrixcity_frame_info.json")
    if os.path.exists(saved_json_path):
        with open(saved_json_path, "r") as f:
            all_frames = json.load(f)
    else:
        all_frames = collect_frame_data(DATA_ROOT)
        with open(saved_json_path, "w") as f:
            json.dump(all_frames, f, indent=4)

    print(f"Found {len(all_frames)} frames to process")

    print("Processing frames...")
    all_instance_paths = [f"{frame['sequence_name']}/{frame['frame_idx']:04d}" 
                         for frame in all_frames]

    def process_single_frame(frame_data):
        try:
            # Load RGB image
            rgb = cv2.imread(frame_data["image_path"])
            if rgb is None:
                print(f"Failed to load image: {frame_data['image_path']}")
                return None  # Return None on failure
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            # Load depth map
            try:
                depth = load_exr_depth(frame_data["depth_path"]) / 100.
            except Exception as e:
                print(f"Failed to load depth: {frame_data['depth_path']} ({e})")
                return None  # Return None on failure
            depth_mask_inf = depth > 655.
            depth_mask = ~depth_mask_inf
            depth = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth, np.nan))
            depth[depth < 0.001] = np.nan
            if "aerial" in frame_data["sequence_name"]:
                depth[depth_mask_inf] = np.nan
            # Camera intrinsics
            K = compute_intrinsics_from_fov(w, h, frame_data["camera_angle_x"])
            K = utils3d.numpy.normalize_intrinsics(K, w, h)
            # Camera pose
            c2w = np.array(frame_data["c2w"], dtype=np.float32)
            # Save processed data
            save_path = Path(OUTPUT_DIR) / frame_data["sequence_name"] / f"{frame_data['frame_idx']:04d}"
            save_path.mkdir(parents=True, exist_ok=True)
            write_image(save_path / 'image.jpg', rgb, quality=95)
            write_depth(save_path / 'depth.png', depth, unit=1)
            write_meta(save_path / 'meta.json', {
                'intrinsics': K.tolist(),
                'camera_pose': c2w.tolist()
            })
            return f"{frame_data['sequence_name']}/{frame_data['frame_idx']:04d}"  # Return instance path on success
        except Exception as e:
            print(f"Error processing {frame_data['sequence_name']}/{frame_data['frame_idx']:04d}: {e}")
            return None  # Return None on failure

        gc.collect()

    # Use multiprocessing for frame processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(process_single_frame, all_frames), total=len(all_frames)))

    # Filter out failed frames (None)
    successful_instance_paths = [r for r in results if r is not None]

    write_index_file(OUTPUT_DIR, successful_instance_paths)
