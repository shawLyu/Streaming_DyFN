import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
import h5py
from typing import *
import warnings
import utils3d
warnings.filterwarnings("ignore")

from moge.utils.io import write_depth, write_image, write_meta
from moge.utils.tools import multithead_execute, catch_exception

DOWNLOAD_DIR = '/home/luban/dataset/hypersim/extracted'
OUTPUT_DIR = 'data/moge_process_dataset/hypersim'
NUM_WORKERS = 32

def get_intrinsic(data_dir: str, scene_name: str) -> np.ndarray:
    """Read the camera intrinsic parameters for a given scene."""
    camera_parameters_path = os.path.join(data_dir, "metadata_camera_parameters.csv")
    df_camera_parameters = pd.read_csv(camera_parameters_path, index_col="scene_name")
    df_ = df_camera_parameters.loc[scene_name]

    width_pixels = int(df_["settings_output_img_width"])
    height_pixels = int(df_["settings_output_img_height"])

    # Extract camera parameters from projection matrix
    M_proj = np.matrix([
        [df_["M_proj_00"], df_["M_proj_01"], df_["M_proj_02"], df_["M_proj_03"]],
        [df_["M_proj_10"], df_["M_proj_11"], df_["M_proj_12"], df_["M_proj_13"]],
        [df_["M_proj_20"], df_["M_proj_21"], df_["M_proj_22"], df_["M_proj_23"]],
        [df_["M_proj_30"], df_["M_proj_31"], df_["M_proj_32"], df_["M_proj_33"]]
    ])

    M_screen_from_ndc = np.matrix([
        [0.5 * (width_pixels - 1), 0, 0, 0.5 * (width_pixels - 1)],
        [0, -0.5 * (height_pixels - 1), 0, 0.5 * (height_pixels - 1)],
        [0, 0, 0.5, 0.5],
        [0, 0, 0, 1.0],
    ])

    screen_from_cam = M_screen_from_ndc @ M_proj
    fx = abs(screen_from_cam[0, 0])
    fy = abs(screen_from_cam[1, 1])
    cx = abs(screen_from_cam[0, 2])
    cy = abs(screen_from_cam[1, 2])

    K = np.eye(3, dtype=np.float32)
    K[0, 0] = float(fx)
    K[1, 1] = float(fy)
    K[0, 2] = float(cx)
    K[1, 2] = float(cy)

    return K

def process_depth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Convert distance to plane depth for Hypersim depth maps."""
    intHeight, intWidth = depth.shape
    fltFocal = intrinsic[0, 0]

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    depth = depth / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return depth

def get_camera_pose(data_dir: str, scene_name: str, index: int, camera: str = "cam_00") -> np.ndarray:
    """Get camera pose matrix for a given frame."""
    camera_dir = os.path.join(data_dir, scene_name, "_detail")
    
    with h5py.File(os.path.join(camera_dir, f"{camera}/camera_keyframe_positions.hdf5"), "r") as f:
        camera_positions = f["dataset"][:]
    with h5py.File(os.path.join(camera_dir, f"{camera}/camera_keyframe_orientations.hdf5"), "r") as f:
        camera_orientations = f["dataset"][:]

    scene_metadata = pd.read_csv(os.path.join(camera_dir, "metadata_scene.csv"))
    meters_per_unit = float(scene_metadata[
        scene_metadata.parameter_name == "meters_per_asset_unit"
    ].parameter_value.iloc[0])

    position = camera_positions[index]
    rotation = camera_orientations[index]
    translation = np.array(position).T * meters_per_unit

    cam2world = np.eye(4)
    cam2world[:3, :3] = rotation
    cam2world[:3, 3] = translation

    gl_to_cv = np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, 1, 1, 1]])
    cam2world *= gl_to_cv

    return cam2world

def tonemap_image(rgb_color: np.ndarray, render_entity_id: np.ndarray) -> np.ndarray:
    """Apply tonemapping to RGB image following CGIntrinsics strategy.
    
    Args:
        rgb_color (np.ndarray): Input RGB image (float32)
        render_entity_id (np.ndarray): Render entity ID mask
    Returns:
        np.ndarray: Tonemapped RGB image
    """
    gamma = 1.0/2.2  # standard gamma correction exponent
    inv_gamma = 1.0/gamma
    percentile = 90  # we want this percentile brightness value in the unmodified image...
    brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling
    
    valid_mask = render_entity_id != -1
    
    if np.count_nonzero(valid_mask) == 0:
        scale = 1.0  # if there are no valid pixels, then set scale to 1.0
    else:
        # "CCIR601 YIQ" method for computing brightness
        brightness = 0.3*rgb_color[:,:,0] + 0.59*rgb_color[:,:,1] + 0.11*rgb_color[:,:,2]
        brightness_valid = brightness[valid_mask]
        
        eps = 0.0001  # threshold to avoid divide-by-zero
        brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)
        
        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:
            scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current
    
    rgb_color_tm = np.power(np.maximum(scale*rgb_color, 0), gamma)
    return np.clip(rgb_color_tm, 0, 1)

def write_index_file(output_dir: str, instance_paths: List[str]):
    """Write index.txt containing all existing instance paths.
    
    Args:
        output_dir (str): Base output directory
        instance_paths (List[str]): List of relative paths to instances
    """
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

if __name__ == "__main__":
    scene_list = [d for d in os.listdir(DOWNLOAD_DIR) 
                 if os.path.isdir(os.path.join(DOWNLOAD_DIR, d)) and 
                 not d.startswith('_')]

    # Keep track of all potential instance paths
    all_instance_paths = []

    for scene_name in (pbar := tqdm(scene_list, desc='Processing scenes')):
        pbar.set_description(scene_name)
        scene_dir = os.path.join(DOWNLOAD_DIR, scene_name)
        
        try:
            intrinsics = get_intrinsic(DOWNLOAD_DIR, scene_name)
        except:
            print(f"Skipping scene {scene_name}: Could not get intrinsics")
            continue
        
        # Process each camera in the scene
        for camera in ["cam_00", "cam_01", "cam_02"]:
            rgb_dir = os.path.join(scene_dir, f"images/scene_{camera}_final_preview")
            depth_dir = os.path.join(scene_dir, f"images/scene_{camera}_geometry_hdf5")
            
            if not all(os.path.isdir(d) for d in [rgb_dir, depth_dir]):
                continue
                
            rgb_files = sorted([p for p in Path(rgb_dir).glob("*.color.jpg")])
            
            # Collect all potential instance paths
            for rgb_path in rgb_files:
                frame_idx = int(rgb_path.name.split(".")[1])
                instance_path = f"{scene_name}/{camera}/{frame_idx:04d}"
                all_instance_paths.append(instance_path)
            
            @multithead_execute(rgb_files, num_workers=NUM_WORKERS)
            @catch_exception
            def process_frame(rgb_path: Path):
                frame_idx = int(rgb_path.name.split(".")[1])
                depth_path = Path(depth_dir) / rgb_path.name.replace("color.jpg", "depth_meters.hdf5")
                entity_id_path = Path(depth_dir) / rgb_path.name.replace("color.jpg", "render_entity_id.hdf5")
                
                save_path = Path(OUTPUT_DIR, scene_name, camera, f"{frame_idx:04d}")
                if os.path.exists(save_path / 'image.jpg') and os.path.exists(save_path / 'depth.png') and os.path.exists(save_path / 'meta.json'):
                    return
                
                # Load RGB image and convert to float32 [0,1]
                rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                # Load depth
                with h5py.File(str(depth_path), "r") as f:
                    depth = f["dataset"][:].astype(np.float32)
                
                # Load render entity ID
                try:
                    with h5py.File(str(entity_id_path), "r") as f:
                        render_entity_id = f["dataset"][:].astype(np.int32)
                except:
                    render_entity_id = np.ones_like(depth, dtype=np.int32)
                
                # Apply tonemapping to RGB
                rgb = tonemap_image(rgb, render_entity_id)
                
                # Process depth
                depth = process_depth(depth, intrinsics)
                depth_mask_inf = depth >= 100
                depth_mask = ~depth_mask_inf
                depth = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth, np.nan))
                
                # Get camera pose
                c2w = get_camera_pose(DOWNLOAD_DIR, scene_name, frame_idx, camera)

                intrinsics_saved = utils3d.numpy.normalize_intrinsics(intrinsics, rgb.shape[1], rgb.shape[0])
                
                # Save processed data
                save_path.mkdir(parents=True, exist_ok=True)
                try:
                    write_image(save_path / 'image.jpg', (rgb * 255).astype(np.uint8), quality=95)
                    write_depth(save_path / 'depth.png', depth, unit=1)
                    write_meta(save_path / 'meta.json', {
                        'intrinsics': intrinsics_saved.tolist(),
                        'camera_pose': c2w.tolist()
                    })
                except:
                    print(f"Skipping frame {frame_idx}: Could not save image or depth")

    # Write index file after processing all scenes, only including existing files
    write_index_file(OUTPUT_DIR, all_instance_paths)