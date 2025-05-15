import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))
import cv2
import numpy as np
from PIL import Image
import glob
import gc
import warnings
import json
from tqdm import tqdm
import utils3d

from moge.utils.io import write_depth, write_image, write_meta

warnings.filterwarnings("ignore")

DATA_ROOT = "data/Structured3D/extracted"
OUTPUT_DIR = "data/moge_process_dataset/structured3d"
NUM_WORKERS = 32

def build_opengl_c2w(v, t, u):
    v = np.asarray(v, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    forward = t / np.linalg.norm(t)
    right = np.cross(forward, u)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    R = np.stack([right, up, -forward], axis=1)  # OpenGL-style
    c2w_gl = np.eye(4, dtype=np.float32)
    c2w_gl[:3, :3] = R
    c2w_gl[:3, 3] = v / 1000.0  # convert mm to m
    return c2w_gl

def convert_opengl_to_opencv(c2w_gl):
    R_cv_gl = np.diag([1, -1, -1])  # flip y and z
    c2w_cv = np.eye(4, dtype=np.float32)
    c2w_cv[:3, :3] = c2w_gl[:3, :3] @ R_cv_gl
    c2w_cv[:3, 3] = c2w_gl[:3, 3]
    return c2w_cv

def build_opencv_intrinsics(xfov, yfov, width, height):
    fx = width / (2.0 * np.tan(xfov))
    fy = height / (2.0 * np.tan(yfov))
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    return K

def collect_frame_data(data_root):
    all_frames = []
    scenes = sorted(os.listdir(data_root))
    for scene in tqdm(scenes, desc="Collecting scenes"):
        scene_path = os.path.join(data_root, scene)
        if not os.path.isdir(scene_path):
            continue
        indexs = os.listdir(os.path.join(scene_path, "2D_rendering"))
        for index in indexs:
            cams = sorted(os.listdir(os.path.join(scene_path, "2D_rendering", index, "perspective", "full")))
            for cam_idx, cam in enumerate(cams):
                cam_path = os.path.join(scene_path, "2D_rendering", index, "perspective", "full", cam)
                rgb_path = os.path.join(cam_path, "rgb_rawlight.png")
                depth_path = os.path.join(cam_path, "depth.png")
                normal_path = os.path.join(cam_path, "normal.png")
                poses_path = os.path.join(cam_path, "camera_pose.txt")
                if not (os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(poses_path)):
                    continue
                sequence_name = f"{scene}"
                frame_idx = f"{index}_{cam_idx}"
                all_frames.append({
                    'rgb_path': rgb_path,
                    'depth_path': depth_path,
                    'normal_path': normal_path,
                    'poses_path': poses_path,
                    'sequence_name': sequence_name,
                    'frame_idx': frame_idx
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
            rgb = cv2.imread(frame_data['rgb_path'])
            if rgb is None:
                print(f"Failed to load image: {frame_data['rgb_path']}")
                return
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            # Load depth map
            depth = cv2.imread(frame_data['depth_path'], cv2.IMREAD_ANYDEPTH) / 1000.
            if depth is None:
                print(f"Failed to load depth: {frame_data['depth_path']}")
                return
            depth = depth.astype(np.float32)
            depth_mask_inf = depth >= 650.0
            depth_mask = ~depth_mask_inf
            depth = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth, np.nan))
            depth[depth < 0.001] = np.nan
            # Load camera pose
            pose_vals = np.loadtxt(frame_data['poses_path'])
            vx, vy, vz, tx, ty, tz, ux, uy, uz, xfov, yfov, _ = pose_vals
            v = np.array([vx, vy, vz])
            t = np.array([tx, ty, tz])
            u = np.array([ux, uy, uz])
            c2w_gl = build_opengl_c2w(v, t, u)
            c2w_cv = convert_opengl_to_opencv(c2w_gl)
            # Camera intrinsics
            K = build_opencv_intrinsics(xfov, yfov, w, h)
            K = utils3d.numpy.normalize_intrinsics(K, w, h)
            # Save processed data
            save_path = Path(OUTPUT_DIR) / frame_data['sequence_name'] / f"{frame_data['frame_idx']}"
            save_path.mkdir(parents=True, exist_ok=True)
            write_image(save_path / 'image.jpg', rgb, quality=95)
            write_depth(save_path / 'depth.png', depth, unit=1)
            write_meta(save_path / 'meta.json', {
                'intrinsics': K.tolist(),
                'camera_pose': c2w_cv.tolist()
            })
        except Exception as e:
            print(f"Error processing {frame_data['sequence_name']}/{frame_data['frame_idx']}: {e}")
        gc.collect()

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_single_frame, all_frames), total=len(all_frames)))

    write_index_file(OUTPUT_DIR, all_instance_paths)
