import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))
import numpy as np
from PIL import Image
import glob
import gc
import warnings
import json
from tqdm import tqdm
import cv2
import utils3d

from moge.utils.io import write_depth, write_image, write_meta

warnings.filterwarnings("ignore")

DATA_ROOT = "data/unrealstereo/extracted"
OUTPUT_DIR = "data/moge_process_dataset/unrealstereo"
NUM_WORKERS = 64

def parse_extrinsics(file_path):
    """Parse the extrinsics file to extract the intrinsics and pose matrices."""
    with open(file_path, "r") as file:
        lines = file.readlines()
        intrinsics_data = list(map(float, lines[0].strip().split()))
        intrinsics_matrix = np.array(intrinsics_data).reshape(3, 3)

        cam2world = np.eye(4)
        pose_data = list(map(float, lines[1].strip().split()))
        pose_matrix = np.array(pose_data).reshape(3, 4)
        cam2world[:3] = pose_matrix
        cam2world = np.linalg.inv(cam2world)

        return intrinsics_matrix, cam2world

def rescale_image_depthmap(
    image, depthmap, camera_intrinsics, output_resolution=(512, 384), force=True
):
    """Jointly rescale a (image, depthmap) so that (out_width, out_height) >= output_res"""
    from PIL import Image as PILImage
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:
        return (image, depthmap, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)
    image = image.resize(tuple(output_resolution), resample=PILImage.LANCZOS if scale_final < 1 else PILImage.BICUBIC)
    if depthmap is not None:
        depthmap = cv2.resize(
            depthmap,
            tuple(output_resolution),
            fx=scale_final,
            fy=scale_final,
            interpolation=cv2.INTER_NEAREST,
        )
    # Adjust intrinsics
    margins = input_resolution * scale_final - output_resolution
    offset = 0.5 * margins
    K = camera_intrinsics.copy()
    K[0, 2] = K[0, 2] * scale_final - offset[0]
    K[1, 2] = K[1, 2] * scale_final - offset[1]
    K[0, 0] *= scale_final
    K[1, 1] *= scale_final
    return image, depthmap, K

def collect_frame_data(data_root):
    all_frames = []
    envs = [f for f in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root, f))]
    for env in tqdm(envs, desc="Collecting environments"):
        frame_dir = os.path.join(data_root, env)
        for subscene in ["0", "1"]:
            rgb_dir = os.path.join(frame_dir, f"Image{subscene}")
            disp_dir = os.path.join(frame_dir, f"Disp{subscene}")
            ext_dir = os.path.join(frame_dir, f"Extrinsics{subscene}")
            if not all(os.path.isdir(d) for d in [rgb_dir, disp_dir, ext_dir]):
                continue
            frame_num = len(os.listdir(rgb_dir))
            for i in range(frame_num):
                rgb_path = os.path.join(rgb_dir, f"{i:05d}.png")
                disp_path = os.path.join(disp_dir, f"{i:05d}.npy")
                ext_path0 = os.path.join(frame_dir, "Extrinsics0", f"{i:05d}.txt")
                ext_path1 = os.path.join(frame_dir, "Extrinsics1", f"{i:05d}.txt")
                if all(os.path.exists(p) for p in [rgb_path, disp_path, ext_path0, ext_path1]):
                    all_frames.append({
                        'rgb_path': rgb_path,
                        'disp_path': disp_path,
                        'ext_path0': ext_path0,
                        'ext_path1': ext_path1,
                        'sequence_name': env,
                        'frame_idx': f"{subscene}_{i:05d}",
                        'subscene': subscene
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
            img = Image.open(frame_data['rgb_path']).convert("RGB")
            # Load disparity and compute depth
            disp = np.load(frame_data['disp_path']).astype(np.float32)
            K0, c2w0 = parse_extrinsics(frame_data['ext_path0'])
            K1, c2w1 = parse_extrinsics(frame_data['ext_path1'])
            if frame_data['subscene'] == "0":
                K = K0
                c2w = c2w0
            else:
                K = K1
                c2w = c2w1
            # Calculate baseline and depth
            baseline = (np.linalg.inv(c2w0) @ c2w1)[0, 3]
            depth = baseline * K[0, 0] / disp
            img_rescaled, depth_rescaled, K_rescaled = rescale_image_depthmap(
                img, depth, K, output_resolution=(512, 384)
            )

            depth_mask_inf = depth > 255
            depth_mask = ~depth_mask_inf
            depth = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth, np.nan))
            depth[depth == 0.0] = np.nan
            depth[disp == 0.0] = np.nan
            # Rescale image and depth
            K_rescaled = utils3d.numpy.normalize_intrinsics(K_rescaled, 512, 384)
            rgb_np = np.array(img_rescaled)
            # Save processed data
            save_path = Path(OUTPUT_DIR) / frame_data['sequence_name'] / f"{frame_data['frame_idx']}"
            save_path.mkdir(parents=True, exist_ok=True)
            write_image(save_path / 'image.jpg', rgb_np, quality=95)
            write_depth(save_path / 'depth.png', depth_rescaled, unit=1)
            write_meta(save_path / 'meta.json', {
                'intrinsics': K_rescaled.tolist(),
                'camera_pose': c2w.tolist()
            })
        except Exception as e:
            print(f"Error processing {frame_data['sequence_name']}/{frame_data['frame_idx']}: {e}")
        gc.collect()

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_single_frame, all_frames), total=len(all_frames)))

    write_index_file(OUTPUT_DIR, all_instance_paths)
