#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Preprocessing code for the WayMo Open dataset
# dataset at https://github.com/waymo-research/waymo-open-dataset
# 1) Accept the license
# 2) download all training/*.tfrecord files from Perception Dataset, version 1.4.2
# 3) put all .tfrecord files in '/path/to/waymo_dir'
# 4) install the waymo_open_dataset package with
#    `python3 -m pip install gcsfs waymo-open-dataset-tf-2-12-0==1.6.4`
# 5) execute this script as `python waymo_to_hdf5.py --root_dir /path/to/waymo_dir --out_dir /path/to/output_dir`
# --------------------------------------------------------
import sys
import os
import os.path as osp
from pathlib import Path
import json
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import PIL.Image
import numpy as np
import warnings
import torch
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

sys.path.append(str(Path(__file__).absolute().parents[1]))
from moge.utils.io import write_depth, write_image, write_meta
from moge.utils.tools import multithead_execute, catch_exception
from moge.utils import cropping


from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils


# Constants
RESOLUTION = 512  # Target resolution for images

def parallel_map(*args, **kwargs):
    """Same as parallel_threads, with processes"""
    import multiprocessing as mp

    kwargs["Pool"] = mp.Pool
    return parallel_threads(*args, **kwargs)

def geotrf(Trf, pts, ncol=None, norm=False):
    """Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    if (
        isinstance(Trf, torch.Tensor)
        and isinstance(pts, torch.Tensor)
        and Trf.ndim == 3
        and pts.ndim == 4
    ):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = (
                torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts)
                + Trf[:, None, None, :d, d]
            )
        else:
            raise ValueError(f"bad shape, not ending with 3 or 4, for {pts.shape=}")
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], "batch size does not match"
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:

                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:

                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res

def inv(mat):
    """Invert a torch or numpy matrix"""
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f"bad matrix type = {type(mat)}")

def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _list_sequences(db_root):
    """List all TFRecord files in the dataset."""
    print(">> Looking for sequences in", db_root)
    res = sorted(f for f in os.listdir(db_root) if f.endswith(".tfrecord"))
    print(f"    found {len(res)} sequences")
    return res

def extract_frames_one_seq(filename):
    """Extract frames from a single TFRecord file."""
    print(">> Opening", filename)
    dataset = tf.data.TFRecordDataset(str(filename), compression_type="")

    calib = None
    frames = []

    for data in tqdm(dataset, leave=False):
        frame = open_dataset.Frame()
        frame.ParseFromString(data.numpy())

        content = frame_utils.parse_range_image_and_camera_projection(frame)
        range_images, camera_projections, _, range_image_top_pose = content

        views = {}
        frames.append((frame.context.name, views))

        # once in a sequence, read camera calibration info
        if calib is None:
            calib = []
            for cam in frame.context.camera_calibrations:
                calib.append(
                    (
                        cam.name,
                        dict(
                            width=cam.width,
                            height=cam.height,
                            intrinsics=list(cam.intrinsic),
                            extrinsics=list(cam.extrinsic.transform),
                        ),
                    )
                )

        # convert LIDAR to pointcloud
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose
        )

        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)
        cp_points_all = np.concatenate(cp_points, axis=0)

        # The distance between lidar points and vehicle frame origin.
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        for i, image in enumerate(frame.images):
            # select relevant 3D points for this view
            mask = tf.equal(cp_points_all_tensor[..., 0], image.name)
            cp_points_msk_tensor = tf.cast(
                tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32
            )

            pose = np.asarray(image.pose.transform).reshape(4, 4)
            timestamp = image.pose_timestamp

            rgb = tf.image.decode_jpeg(image.image).numpy()

            pix = cp_points_msk_tensor[..., 1:3].numpy().round().astype(np.int16)
            pts3d = points_all[mask.numpy()]

            views[image.name] = dict(
                img=rgb, pose=pose, pixels=pix, pts3d=pts3d, timestamp=timestamp
            )

    return calib, frames

def normalize_intrinsics(K, width, height):
    """Normalize intrinsic matrix to have focal length relative to image size."""
    K_norm = K.copy()
    K_norm[0, 0] /= width
    K_norm[1, 1] /= height
    K_norm[0, 2] /= width
    K_norm[1, 2] /= height
    return K_norm

def process_frame(frame_data, output_dir):
    """Process a single frame and save in MoGe format."""
    # Extract data
    view = frame_data['view']
    cam_K = frame_data['cam_K']
    cam_res = frame_data['cam_res']
    cam_to_car = frame_data['cam_to_car']
    axes_transformation = frame_data['axes_transformation']
    sequence_name = frame_data['sequence_name']
    frame_idx = frame_data['frame_idx']
    
    # Create output path
    save_path = Path(output_dir) / sequence_name / f"{frame_idx:04d}"
    if os.path.exists(save_path / 'image.jpg') and os.path.exists(save_path / 'depth.png') and os.path.exists(save_path / 'meta.json'):
        return
    
    # Get image and 3D points
    image = view["img"]
    car_to_world = view["pose"]
    pos2d = view["pixels"].round().astype(np.uint16)
    pts3d = view["pts3d"]  # already in the car frame
    
    # Transform points to camera coordinate system
    pts3d = geotrf(axes_transformation @ inv(cam_to_car), pts3d)
    
    # Rescale image
    W, H = cam_res
    output_resolution = (RESOLUTION, 1) if W > H else (1, RESOLUTION)
    pil_image = PIL.Image.fromarray(image)
    image_rescaled, _, intrinsics2 = cropping.rescale_image_depthmap(
        pil_image, None, cam_K, output_resolution
    )
    
    # Create depth map
    W_new, H_new = image_rescaled.size
    depthmap = np.zeros((H_new, W_new), dtype=np.float32)
    
    # Transform pixel coordinates to new resolution
    pos2d_new = geotrf(intrinsics2 @ inv(cam_K), pos2d).round().astype(np.int16)
    x, y = pos2d_new.T
    
    # Fill depth map
    for i in range(len(x)):
        if 0 <= y[i] < H_new and 0 <= x[i] < W_new:
            # If multiple points project to the same pixel, keep the closest one
            if depthmap[y[i], x[i]] == 0 or pts3d[i, 2] < depthmap[y[i], x[i]]:
                depthmap[y[i], x[i]] = pts3d[i, 2]
    
    # Handle invalid values
    depth_mask_inf = depthmap > 1000
    depth_mask = ~depth_mask_inf
    depthmap = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depthmap, np.nan))
    depthmap[depthmap == 0] = np.nan
    
    # Calculate camera to world transform
    cam2world = car_to_world @ cam_to_car @ inv(axes_transformation)
    
    # Normalize intrinsics
    K_norm = normalize_intrinsics(intrinsics2, W_new, H_new)
    
    # Save processed data
    save_path.mkdir(parents=True, exist_ok=True)
    try:
        write_image(save_path / 'image.jpg', np.array(image_rescaled), quality=95)
        write_depth(save_path / 'depth.png', depthmap, unit=1)
        write_meta(save_path / 'meta.json', {
            'intrinsics': K_norm.tolist(),
            'camera_pose': cam2world.tolist()
        })
        return True
    except Exception as e:
        print(f"Error processing {sequence_name}/{frame_idx:04d}: {e}")
        return False

def process_sequence(seq_path, output_dir):
    """Process a complete TFRecord sequence file."""
    seq_name = os.path.basename(seq_path)
    print(f">> Processing sequence {seq_name}")
    
    try:
        with tf.device("/CPU:0"):
            calib, frames = extract_frames_one_seq(seq_path)
    except RuntimeError:
        print(f"/!\\ Error with sequence {seq_name} /!\\", file=sys.stderr)
        return []
    
    # Process calibration data
    cam_K = {}
    cam_distortion = {}
    cam_res = {}
    cam_to_car = {}
    
    for cam_idx, cam_info in calib:
        cam_idx = str(cam_idx)
        cam_res[cam_idx] = (W, H) = (cam_info["width"], cam_info["height"])
        f1, f2, cx, cy, k1, k2, p1, p2, k3 = cam_info["intrinsics"]
        cam_K[cam_idx] = np.asarray([(f1, 0, cx), (0, f2, cy), (0, 0, 1)])
        cam_distortion[cam_idx] = np.asarray([k1, k2, p1, p2, k3])
        cam_to_car[cam_idx] = np.asarray(cam_info["extrinsics"]).reshape(4, 4)  # cam-to-vehicle
    
    # Define coordinate transformation
    axes_transformation = np.array(
        [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    )
    
    processed_instances = []
    
    # Process each frame
    for f, (frame_name, views) in enumerate(frames):
        for cam_idx, view in views.items():
            cam_idx = str(cam_idx)
            
            # Skip if camera calibration is missing
            if cam_idx not in cam_K:
                continue
            
            # Create sequence name
            sequence_name = f"{seq_name[:-9]}_{cam_idx}"  # Remove .tfrecord extension
            
            # Process frame
            frame_data = {
                'seq_name': seq_name,
                'cam_idx': cam_idx,
                'frame_idx': f,
                'view': view,
                'cam_K': cam_K[cam_idx],
                'cam_res': cam_res[cam_idx],
                'cam_to_car': cam_to_car[cam_idx],
                'axes_transformation': axes_transformation,
                'sequence_name': sequence_name
            }
            
            instance_path = f"{sequence_name}/{f:04d}"
            save_path = Path(output_dir) / instance_path
            
            # Skip if already processed
            if os.path.exists(save_path / 'image.jpg') and \
               os.path.exists(save_path / 'depth.png') and \
               os.path.exists(save_path / 'meta.json'):
                processed_instances.append(instance_path)
                continue
                
            if process_frame(frame_data, output_dir):
                processed_instances.append(instance_path)
    
    return processed_instances

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
    parser = argparse.ArgumentParser(description="Preprocess Waymo dataset.")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the Waymo dataset.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for processed data.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker processes.")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Get all sequence files
    sequences = _list_sequences(args.root_dir)
    sequence_paths = [os.path.join(args.root_dir, seq) for seq in sequences]
    
    print(f"Found {len(sequence_paths)} sequences to process")
    
    # Process sequences in parallel
    all_processed_instances = []
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_sequence, seq_path, args.out_dir) 
                  for seq_path in sequence_paths]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing sequences"):
            try:
                processed_instances = future.result()
                all_processed_instances.extend(processed_instances)
            except Exception as e:
                print(f"Error in worker process: {e}")
    
    # Write index file after processing all sequences
    write_index_file(args.out_dir, all_processed_instances)
    print(f"Processing complete. Results saved to {args.out_dir}")

if __name__ == "__main__":
    mp.freeze_support()
    main()