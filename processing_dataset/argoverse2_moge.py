import os
import sys
import json
import numpy as np
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from tqdm import tqdm
from PIL import Image
import logging
from pathlib import Path
import cv2
import warnings

import av2.utils.io as io_utils
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.se3 import SE3

sys.path.append(str(Path(__file__).absolute().parents[1]))
from moge.utils.io import write_depth, write_image, write_meta
from moge.utils.tools import multithead_execute, catch_exception
import utils3d

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_DEPTH = 65535  # Maximum value for uint16

def list_all_log_ids(data_root):
    """List all log IDs in the dataset."""
    log_ids = []
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "sensors")):
            log_ids.append(item)
    return log_ids

def normalize_intrinsics(K, width, height):
    """Normalize intrinsic matrix to have focal length relative to image size."""
    K_norm = K.copy()
    K_norm[0, 0] /= width
    K_norm[1, 1] /= height
    K_norm[0, 2] /= width
    K_norm[1, 2] /= height
    return K_norm

def collect_log_frames(args):
    """Collect all frames for a single log (for multiprocessing)."""
    log_id, data_root_str = args
    data_root = Path(data_root_str)
    frames = []
    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)
    log_map_dirpath = data_root / log_id / "map"
    try:
        avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)
    except Exception as e:
        logger.error(f"Error loading map for {log_id}: {e}")
        return []
    for cam_enum in list(RingCameras):
        cam_name = cam_enum.value
        try:
            pinhole_camera = loader.get_log_pinhole_camera(log_id=log_id, cam_name=cam_name)
        except Exception as e:
            logger.error(f"Error getting camera for {log_id}, {cam_name}: {e}")
            continue
        cam_im_fpaths = loader.get_ordered_log_cam_fpaths(log_id, cam_name)
        if not cam_im_fpaths:
            logger.warning(f"No images found for camera {cam_name} in log {log_id}")
            continue
        for i, im_fpath in enumerate(cam_im_fpaths):
            cam_timestamp_ns = int(im_fpath.stem)
            city_SE3_ego = loader.get_city_SE3_ego(log_id, cam_timestamp_ns)
            if city_SE3_ego is None:
                continue
            lidar_fpath = loader.get_closest_lidar_fpath(log_id, cam_timestamp_ns)
            if lidar_fpath is None:
                continue
            frame_data = {
                'log_id': log_id,
                'cam_name': cam_name,
                'image_path': im_fpath,
                'lidar_path': lidar_fpath,
                'timestamp': cam_timestamp_ns,
                'lidar_timestamp': int(lidar_fpath.stem),
                'sequence_name': f"{log_id}_{cam_name}",
                'frame_idx': i
            }
            frames.append(frame_data)
    return frames

def collect_frame_data(data_root, num_logs=None, num_workers=8):
    """Collect all frames from the dataset using multiprocessing."""
    all_frames = []
    data_root = Path(data_root)
    log_ids = list_all_log_ids(str(data_root))
    if num_logs is not None and num_logs > 0:
        log_ids = log_ids[:num_logs]
    logger.info(f"Processing {len(log_ids)} log IDs (multiprocessing)")
    args_list = [(log_id, str(data_root)) for log_id in log_ids]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(collect_log_frames, args_list), total=len(args_list), desc="Scanning logs"))
    for frames in results:
        all_frames.extend(frames)
    # Loader is not returned anymore, since it's not needed outside
    return all_frames, None

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
        logger.info(f"Wrote index file to {index_path} with {len(existing_paths)} instances")
    else:
        logger.warning("No valid instances found, index.txt not created")

def process_frame_worker(frame_data, root_dir, out_dir):
    """Process a single frame (for multiprocessing)."""
    log_id = frame_data['log_id']
    cam_name = frame_data['cam_name']
    im_fpath = frame_data['image_path']
    lidar_fpath = frame_data['lidar_path']
    cam_timestamp_ns = frame_data['timestamp']
    lidar_timestamp_ns = frame_data['lidar_timestamp']

    # Re-initialize loader inside the process
    loader = AV2SensorDataLoader(data_dir=Path(root_dir), labels_dir=Path(root_dir))

    # Get camera intrinsics
    pinhole_camera = loader.get_log_pinhole_camera(log_id=log_id, cam_name=cam_name)
    camera_intrinsics = pinhole_camera.intrinsics.K
    
    # Get ego vehicle pose
    city_SE3_ego = loader.get_city_SE3_ego(log_id, cam_timestamp_ns)
    
    # Load RGB image
    try:
        img_bgr = io_utils.read_img(im_fpath, channel_order="BGR")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error reading image {im_fpath}: {e}")
        raise e
    
    # Load LiDAR points
    try:
        lidar_points_ego = io_utils.read_lidar_sweep(lidar_fpath, attrib_spec="xyz")
    except Exception as e:
        logger.error(f"Error reading LiDAR data {lidar_fpath}: {e}")
        raise e
    
    # Transform points to city coordinates
    lidar_points_city = city_SE3_ego.transform_point_cloud(lidar_points_ego)
    
    # Get map for ground point detection
    log_map_dirpath = Path(root_dir) / log_id / "map"
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)
    
    # Get ground points (but don't filter)
    is_ground_logicals = avm.get_ground_points_boolean(lidar_points_city)
    
    # Transform back to ego coordinates
    lidar_points_ego = city_SE3_ego.inverse().transform_point_cloud(lidar_points_city)
    
    # Project points to camera
    (uv, points_cam, is_valid_points) = loader.project_ego_to_img_motion_compensated(
        points_lidar_time=lidar_points_ego,
        cam_name=cam_name,
        cam_timestamp_ns=cam_timestamp_ns,
        lidar_timestamp_ns=lidar_timestamp_ns,
        log_id=log_id,
    )
    
    if is_valid_points is None or uv is None or points_cam is None:
        raise ValueError(f"Invalid projection for timestamp {cam_timestamp_ns}")
    
    # Filter valid points
    uv_valid = uv[is_valid_points]
    points_cam_valid = points_cam[is_valid_points]
    
    # Round to integer pixel coordinates
    uv_int = np.round(uv_valid).astype(np.int32)
    
    # Get image dimensions
    height_px, width_px = img_rgb.shape[:2]
    
    # Create depth map
    depth_map = np.zeros((height_px, width_px), dtype=np.float32)
    
    # Extract valid depths (z-coordinate in camera frame)
    depths_valid = points_cam_valid[:, 2]
    
    # Fill depth map
    for idx, (u, v) in enumerate(uv_int):
        if 0 <= v < height_px and 0 <= u < width_px:
            # If multiple points project to the same pixel, keep the closest one
            if depth_map[v, u] == 0 or depths_valid[idx] < depth_map[v, u]:
                depth_map[v, u] = depths_valid[idx]
    
    # Handle invalid values
    depth_map[depth_map == 0] = np.nan

    # Handle extreme values similar to MidAir approach
    # Set very far values to infinity (e.g., beyond 1250 meters)
    depth_mask_inf = depth_map >= 1250  
    depth_mask = ~depth_mask_inf

    # Set far values to inf, invalid values to NaN
    depth_map = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth_map, np.nan))

    # Set very close values (less than 0.001m) to NaN
    depth_map[depth_map < 0.001] = np.nan
    
    # Calculate camera pose (cam2w)
    # First get ego to world (city) transform
    ego2world = city_SE3_ego
    # Then get camera to ego transform
    ego2cam = pinhole_camera.ego_SE3_cam
    # Combine to get camera to world transform
    cam2world = ego2world.compose(ego2cam)
    
    # Extract rotation matrix and translation vector
    R = cam2world.rotation
    t = cam2world.translation
    
    # Create 4x4 transformation matrix
    cam2w_matrix = np.eye(4)
    cam2w_matrix[:3, :3] = R
    cam2w_matrix[:3, 3] = t
    
    # Normalize intrinsics
    # K_norm = normalize_intrinsics(camera_intrinsics, width_px, height_px)
    K_norm = utils3d.numpy.normalize_intrinsics(camera_intrinsics, width_px, height_px)
    
    # Save processed data
    save_path = Path(out_dir) / f"{frame_data['sequence_name']}" / f"{frame_data['frame_idx']:04d}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        write_image(save_path / 'image.jpg', img_rgb, quality=95)
        write_depth(save_path / 'depth.png', depth_map, unit=1)
        write_meta(save_path / 'meta.json', {
            'intrinsics': K_norm.tolist(),
            'camera_pose': cam2w_matrix.tolist()
        })
    except Exception as e:
        logger.error(f"Error saving data for {frame_data['sequence_name']}/{frame_data['frame_idx']:04d}: {e}")
        raise e

def main():
    parser = argparse.ArgumentParser(description="Preprocess Argoverse2 dataset.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the Argoverse2 dataset.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for processed data.")
    args = parser.parse_args()
    
    num_workers = 32
    num_logs = None
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Collecting frame data...")
    all_frames, _ = collect_frame_data(args.root_dir, num_logs)
    print(f"Found {len(all_frames)} frames to process")
    
    print("Processing frames...")
    all_instance_paths = [f"{frame['sequence_name']}/{frame['frame_idx']:04d}" 
                         for frame in all_frames]

    # Multiprocessing with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for frame_data in all_frames:
            futures.append(
                executor.submit(process_frame_worker, frame_data, args.root_dir, args.out_dir)
            )
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing frames"):
            try:
                f.result()
            except Exception as e:
                logger.error(f"Error in processing: {e}")

    write_index_file(args.out_dir, all_instance_paths)
    logger.info(f"Processing complete. Results saved to {args.out_dir}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
