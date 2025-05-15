import os
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parents[1]))
from tqdm import tqdm
from PIL import Image
import glob
import gc
import warnings
import concurrent.futures

from moge.utils.io import write_depth, write_image, write_meta
from moge.utils.tools import multithead_execute, catch_exception

import OpenEXR
import Imath
import utils3d

warnings.filterwarnings("ignore")

ROOT_DIR = "/home/luban/dataset/irs/extracted"  # <-- set your IRS root directory
OUTPUT_DIR = "data/moge_process_dataset/irs"
NUM_WORKERS = 96

def exr2hdr(exrpath):
    file = OpenEXR.InputFile(exrpath)
    pixType = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = file.header()["dataWindow"]
    num_channels = len(file.header()["channels"].keys())
    if num_channels > 1:
        channels = ["R", "G", "B"]
        num_channels = 3
    else:
        channels = ["G"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    pixels = [
        np.frombuffer(file.channel(c, pixType), dtype=np.float32) for c in channels
    ]
    hdr = np.zeros((size[1], size[0], num_channels), dtype=np.float32)
    if num_channels == 1:
        hdr[:, :, 0] = np.reshape(pixels[0], (size[1], size[0]))
    else:
        hdr[:, :, 0] = np.reshape(pixels[0], (size[1], size[0]))
        hdr[:, :, 1] = np.reshape(pixels[1], (size[1], size[0]))
        hdr[:, :, 2] = np.reshape(pixels[2], (size[1], size[0]))
    return hdr

def load_exr(filename):
    hdr = exr2hdr(filename)
    h, w, c = hdr.shape
    if c == 1:
        hdr = np.squeeze(hdr)
    return hdr

def get_intrinsic(w, h, f=480):
    K = np.eye(3, dtype=np.float32)
    K[0,0] = f
    K[1,1] = f
    K[0,2] = w // 2
    K[1,2] = h // 2
    return K

def collect_frame_data(root_dir):
    baseline = 0.1
    f = 480
    all_frames = []
    seq_dirs = []
    for d in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, d)):
            if d == "Store" or d == "Restaurant":
                for sub in os.listdir(os.path.join(root_dir, d)):
                    if os.path.isdir(os.path.join(root_dir, d, sub)):
                        seq_dirs.append(os.path.join(d, sub))
            else:
                seq_dirs.append(d)
    seq_dirs.sort()
    for seq_dir in seq_dirs:
        image_files = sorted(glob.glob(os.path.join(root_dir, seq_dir, "l*.png")))
        disp_files = sorted(glob.glob(os.path.join(root_dir, seq_dir, "d*.exr")))
        sequence_name = seq_dir.replace("/", "_")
        for i, (image_file, disp_file) in enumerate(zip(image_files, disp_files)):
            frame_idx = int(os.path.basename(image_file).split("_")[1].split(".")[0])
            all_frames.append({
                'image_path': image_file,
                'disp_path': disp_file,
                'sequence_name': sequence_name,
                'frame_idx': frame_idx,
                'f': f,
                'baseline': baseline
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

def process_single_frame(frame_data):
    try:
        # Load RGB image
        rgb = np.array(Image.open(frame_data['image_path']))
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        h, w = rgb.shape[:2]
        # Load disparity and convert to depth
        disp = load_exr(frame_data['disp_path']).astype(np.float32)
        if disp.ndim == 3:
            disp = disp[:, :, 0]
        # Depth calculation
        depth = frame_data['baseline'] * frame_data['f'] / disp
        depth_mask_inf = np.zeros_like(depth).astype(np.bool_)
        depth_mask = ~depth_mask_inf
        depth = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth, np.nan))
        depth[disp < 0.5] = np.nan
        # Camera intrinsics
        K = get_intrinsic(w, h, frame_data['f'])
        K = utils3d.numpy.normalize_intrinsics(K, w, h)
        # Camera pose (identity)
        c2w = np.eye(4, dtype=np.float32)
        # Save processed data
        save_path = Path(OUTPUT_DIR) / frame_data['sequence_name'] / f"{frame_data['frame_idx']:04d}"
        save_path.mkdir(parents=True, exist_ok=True)
        write_image(save_path / 'image.jpg', rgb, quality=95)
        write_depth(save_path / 'depth.png', depth, unit=1)
        write_meta(save_path / 'meta.json', {
            'intrinsics': K.tolist(),
            'camera_pose': c2w.tolist()
        })
    except Exception as e:
        print(f"Error processing {frame_data['sequence_name']}/{frame_data['frame_idx']:04d}: {e}")
    gc.collect()

if __name__ == "__main__":
    print("Collecting frame data...")
    all_frames = collect_frame_data(ROOT_DIR)
    print(f"Found {len(all_frames)} frames to process")

    print("Processing frames...")
    all_instance_paths = [f"{frame['sequence_name']}/{frame['frame_idx']:04d}" 
                         for frame in all_frames]

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_single_frame, all_frames), total=len(all_frames)))

    write_index_file(OUTPUT_DIR, all_instance_paths)
