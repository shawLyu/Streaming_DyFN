import os
import sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import json
import argparse
import random
from scipy.spatial.transform import Rotation
import PIL.Image as Image

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).absolute().parents[1]))

# Import MoGe utilities
from moge.utils.io import write_depth, write_image, write_meta
from moge.utils.tools import multithead_execute, catch_exception
import utils3d

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannetpp_dir", required=True, help="Path to ScanNetPP dataset")
    parser.add_argument("--output_dir", required=True, help="Path to prerendered output directory")
    parser.add_argument("--scene_list", required=True, help="Path to text file with scene names")
    parser.add_argument("--moge_output_dir", default="data/scannetpp_moge", help="Output directory for MoGe format")
    parser.add_argument("--num_workers", default=32, type=int, help="Number of workers for parallel processing")
    parser.add_argument("--max_images_per_scene", default=100, type=int, help="Maximum number of images to process per scene")
    return parser

def load_scene_list(scene_list_path):
    """Load the list of scenes from a text file."""
    with open(scene_list_path, "r") as f:
        scenes = [line.strip() for line in f.readlines()]
    return scenes

def pose_from_qwxyz_txyz(elems):
    """Convert quaternion and translation to camera-to-world matrix."""
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose)  # returns cam2world

def load_sfm(sfm_dir, cam_type="dslr"):
    """Load SfM data from COLMAP format."""
    # Regular expressions for frame IDs
    import re
    REGEXPR_DSLR = re.compile(r"^DSC(?P<frameid>\d+).JPG$")
    REGEXPR_IPHONE = re.compile(r"frame_(?P<frameid>\d+).jpg$")
    
    def get_frame_number(name, cam_type="dslr"):
        if cam_type == "dslr":
            regex_expr = REGEXPR_DSLR
        elif cam_type == "iphone":
            regex_expr = REGEXPR_IPHONE
        else:
            raise NotImplementedError(f"wrong {cam_type=} for get_frame_number")
        matches = re.match(regex_expr, name)
        return matches["frameid"]
    
    # Load cameras
    with open(os.path.join(sfm_dir, "cameras.txt"), "r") as f:
        raw = f.read().splitlines()[3:]  # skip header

    intrinsics = {}
    for camera in raw:
        camera = camera.split(" ")
        intrinsics[int(camera[0])] = [camera[1]] + [float(cam) for cam in camera[2:]]

    # Load images
    with open(os.path.join(sfm_dir, "images.txt"), "r") as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith("#")]  # skip header

    img_idx = {}
    img_infos = {}
    for image, points in zip(raw[0::2], raw[1::2]):
        image = image.split(" ")
        points = points.split(" ")

        idx = image[0]
        img_name = image[-1]
        assert img_name not in img_idx, "duplicate db image: " + img_name
        img_idx[img_name] = idx  # register image name

        current_points2D = {
            int(i): (float(x), float(y))
            for i, x, y in zip(points[2::3], points[0::3], points[1::3])
            if i != "-1"
        }
        img_infos[idx] = dict(
            intrinsics=intrinsics[int(image[-2])],
            path=img_name,
            frame_id=get_frame_number(img_name, cam_type),
            cam_to_world=pose_from_qwxyz_txyz(image[1:-2]),
            sparse_pts2d=current_points2D,
        )

    return img_idx, img_infos

def undistort_images(intrinsics, rgb):
    """Undistort images using camera intrinsics."""
    camera_type = intrinsics[0]

    width = int(intrinsics[1])
    height = int(intrinsics[2])
    fx = intrinsics[3]
    fy = intrinsics[4]
    cx = intrinsics[5]
    cy = intrinsics[6]
    distortion = np.array(intrinsics[7:])

    K = np.zeros([3, 3])
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    K[2, 2] = 1

    # Convert to OpenCV format
    K_cv = np.copy(K)
    
    if camera_type == "OPENCV_FISHEYE":
        assert len(distortion) == 4

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_cv,
            distortion,
            (width, height),
            np.eye(3),
            balance=0.0,
        )
        # Make the cx and cy to be the center of the image
        new_K[0, 2] = width / 2.0
        new_K[1, 2] = height / 2.0

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K_cv, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1
        )
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K_cv, distortion, (width, height), 1, (width, height), True
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            K_cv, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1
        )

    undistorted_image = cv2.remap(
        rgb,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    
    return width, height, new_K, undistorted_image

def sample_images(img_infos, max_images):
    """Sample a subset of images if needed."""
    if len(img_infos) <= max_images:
        return img_infos
    
    # Convert to list for random sampling
    img_infos_list = list(img_infos.items())
    sampled_indices = sorted(random.sample(range(len(img_infos_list)), max_images))
    
    # Create new dictionary with sampled images
    sampled_img_infos = {idx: img_info for i, (idx, img_info) in enumerate(img_infos_list) if i in sampled_indices}
    return sampled_img_infos

def write_index_file(output_dir, instance_paths):
    """Write index.txt containing all existing instance paths."""
    # Filter only existing paths
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

def process_scenes(scannetpp_dir, output_dir, scene_list_path, moge_output_dir, num_workers, max_images_per_scene):
    """Process ScanNetPP scenes to MoGe format using prerendered images."""
    import random
    random.seed(42)  # For reproducibility
    
    os.makedirs(moge_output_dir, exist_ok=True)
    
    # Load scene list
    scenes = load_scene_list(scene_list_path)
    
    # Keep track of all instance paths
    all_instance_paths = []
    
    for scene in tqdm(scenes, position=0, leave=True, desc="Processing scenes"):
        data_dir = os.path.join(scannetpp_dir, "data", scene)
        dir_dslr = os.path.join(data_dir, "dslr")
        
        # Check if directories exist
        if not all(os.path.isdir(d) for d in [data_dir, dir_dslr]):
            print(f"Skipping scene {scene}: Missing directories")
            continue
        
        # Load SfM data
        sfm_dir_dslr = os.path.join(dir_dslr, "colmap")
        
        try:
            img_idx_dslr, img_infos_dslr = load_sfm(sfm_dir_dslr, cam_type="dslr")
        except Exception as e:
            print(f"Skipping scene {scene}: Could not load SfM data: {e}")
            continue
        
        # Sample a subset of images if needed
        img_infos_dslr_sampled = sample_images(img_infos_dslr, max_images_per_scene)
        
        # Prerendered output directories
        render_dir = os.path.join(output_dir, scene)
        
        # Process images for DSLR only
        image_tasks = []
        for imgidx, img_info in img_infos_dslr_sampled.items():
            # Create instance path - remove dslr from path
            instance_path = f"{scene}/{img_info['frame_id']}"
            all_instance_paths.append(instance_path)
            
            # Add task
            image_tasks.append({
                'imgidx': imgidx,
                'img_info': img_info,
                'scene': scene,
                'render_dir': render_dir,
                'moge_output_dir': moge_output_dir,
                'scannetpp_dir': scannetpp_dir
            })
        
        # Process images in parallel
        @multithead_execute(image_tasks, num_workers=num_workers)
        @catch_exception
        def process_image(task):
            img_info = task['img_info']
            scene = task['scene']
            render_dir = task['render_dir']
            moge_output_dir = task['moge_output_dir']
            scannetpp_dir = task['scannetpp_dir']
            
            # Create output directory - remove dslr from path
            instance_path = f"{scene}/{img_info['frame_id']}"
            save_path = Path(moge_output_dir) / instance_path
            
            # Skip if already processed
            if (save_path / 'image.jpg').exists() and \
               (save_path / 'depth.png').exists() and \
               (save_path / 'meta.json').exists():
                return
            
            try:
                # Determine file paths for DSLR
                frame_id = img_info['frame_id']
                rgb_filename = f"DSC{frame_id}.JPG"
                depth_filename = f"DSC{frame_id}.png"
                
                # Paths to prerendered files
                rgb_path = os.path.join(render_dir, "dslr", "render_rgb", rgb_filename)
                
                # Use undistorted depth data from the new path
                depth_path = os.path.join("/home/muxin/Completion_Depth/Datasets/scannet++/output", 
                                         scene, "dslr", "undistorted_depth", depth_filename)
                
                # Check if files exist
                if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                    print(f"Skipping {instance_path}: Missing prerendered files")
                    return
                
                # Load RGB image
                rgb = np.array(Image.open(rgb_path))
                
                # Load depth image
                depth = np.array(Image.open(depth_path))

                # Convert depth to mm - MODIFIED
                depth_raw = depth.astype(np.float32)
                
                # Mark invalid depth values (0 or 65535) as NaN
                invalid_mask = (depth_raw == 0) | (depth_raw >= 65535)
                depth = depth_raw / 1000.0  # Convert to meters
                depth[invalid_mask] = np.nan  # Mark invalid pixels
                
                # Load original image for undistortion
                data_dir = os.path.join(scannetpp_dir, "data", scene)
                orig_rgb_path = os.path.join(data_dir, "dslr", "resized_images", img_info['path'])
                if os.path.exists(orig_rgb_path):
                    orig_rgb = np.array(Image.open(orig_rgb_path))
                    _, _, K, rgb = undistort_images(img_info['intrinsics'], orig_rgb)
                else:
                    # If original image not found, use the prerendered one
                    K = np.array([
                        [img_info['intrinsics'][3], 0, img_info['intrinsics'][5]],
                        [0, img_info['intrinsics'][4], img_info['intrinsics'][6]],
                        [0, 0, 1]
                    ])
                
                # Get image dimensions
                H, W = rgb.shape[:2]
                
                # Normalize intrinsics for MoGe format
                intrinsics_normalized = utils3d.numpy.normalize_intrinsics(
                    K, W, H
                )
                
                # Save processed data
                save_path.mkdir(parents=True, exist_ok=True)
                
                # Save image, depth, and metadata
                write_image(save_path / 'image.jpg', rgb, quality=95)
                write_depth(save_path / 'depth.png', depth, unit=1)  # Changed from unit=0 to unit=1 for meters
                write_meta(save_path / 'meta.json', {
                    'intrinsics': intrinsics_normalized.tolist(),
                    'camera_pose': img_info['cam_to_world'].tolist()
                })
            except Exception as e:
                print(f"Error processing {instance_path}: {e}")
    
    # Write index file
    write_index_file(moge_output_dir, all_instance_paths)
    print("Processing complete!")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Process scenes
    process_scenes(
        args.scannetpp_dir,
        args.output_dir,
        args.scene_list,
        args.moge_output_dir,
        args.num_workers,
        args.max_images_per_scene
    )