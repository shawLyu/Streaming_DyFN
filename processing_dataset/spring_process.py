import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))
import numpy as np
from PIL import Image
import h5py
import glob
import gc
import warnings
from tqdm import tqdm
import utils3d

from moge.utils.io import write_depth, write_image, write_meta

warnings.filterwarnings("ignore")

SPRING_BASELINE = 0.065

DATA_ROOT = "data/spring/extracted"
OUTPUT_DIR = "data/moge_process_dataset/spring_new"
SPLIT = "train"  # or "val", "test"
NUM_WORKERS = 64

def get_depth(disp1, intrinsics, baseline=SPRING_BASELINE):
    """Get depth from reference frame disparity and camera intrinsics."""
    return intrinsics[0] * baseline / (disp1 + 1e-4)

def readDsp5Disp(filename):
    """Read disparity from dsp5 file."""
    with h5py.File(filename, "r") as f:
        if "disparity" not in f.keys():
            raise IOError(f"File {filename} does not have a 'disparity' key")
        return f["disparity"][()]

def collect_frame_data(data_root, split):
    all_frames = []
    spring_path = os.path.join(data_root, "spring", split)
    sequences = sorted(os.listdir(spring_path))
    for seq in tqdm(sequences, desc="Collecting frame information"):
        disp_pattern = os.path.join(spring_path, seq, "disp1_left", "*.dsp5")
        disp_files = sorted(glob.glob(disp_pattern))
        intrinsics_path = os.path.join(spring_path, seq, "cam_data", "intrinsics.txt")
        extrinsics_path = os.path.join(spring_path, seq, "cam_data", "extrinsics.txt")
        if not os.path.exists(intrinsics_path):
            continue
        for disp_file in disp_files:
            frame_id = int(os.path.basename(disp_file).split('_')[-1].split('.')[0])
            img_path = os.path.join(spring_path, seq, "frame_left", f"frame_left_{frame_id:04d}.png")
            if os.path.exists(img_path):
                all_frames.append({
                    'img_path': img_path,
                    'disp_path': disp_file,
                    'intrinsics_path': intrinsics_path,
                    'extrinsics_path': extrinsics_path,
                    'sequence_name': seq,
                    'frame_id': frame_id,
                    'frame_idx': f"{frame_id:04d}"
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
    all_frames = collect_frame_data(DATA_ROOT, SPLIT)
    print(f"Found {len(all_frames)} frames to process")

    print("Processing frames...")
    all_instance_paths = [f"{frame['sequence_name']}/{frame['frame_idx']}" 
                         for frame in all_frames]

    import concurrent.futures

    def process_single_frame(frame_data):
        try:
            # Load disparity
            disp = readDsp5Disp(frame_data['disp_path']).astype(np.float32)
            disp = disp[::2, ::2]
            # Load intrinsics
            intrinsics = np.loadtxt(frame_data['intrinsics_path'])[frame_data['frame_id'] - 1]
            # Load extrinsics
            extrinsics = np.loadtxt(frame_data['extrinsics_path'])[frame_data['frame_id'] - 1]
            # Load image
            img = np.array(Image.open(frame_data['img_path']))
            h, w = img.shape[:2]
            # Calculate depth
            depth = get_depth(disp, intrinsics)
            # Mask: set depth > 6550 to inf, else nan if invalid
            depth_mask_inf = (depth == np.nan) | (depth > 1250)
            depth_mask = ~depth_mask_inf
            depth = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth, np.nan))
            depth[depth < 0.001] = np.nan
            # Camera intrinsics matrix
            K = np.array([
                [intrinsics[0], 0, intrinsics[2]],
                [0, intrinsics[1], intrinsics[3]],
                [0, 0, 1]
            ], dtype=np.float32)
            K = utils3d.numpy.normalize_intrinsics(K, w, h, integer_pixel_centers=False)
            # Camera pose (identity)
            c2w = np.linalg.inv(np.reshape(extrinsics, (4, 4)))
            # Save processed data
            save_path = Path(OUTPUT_DIR) / frame_data['sequence_name'] / f"{frame_data['frame_idx']}"
            save_path.mkdir(parents=True, exist_ok=True)
            write_image(save_path / 'image.jpg', img, quality=95)
            write_depth(save_path / 'depth.png', depth, unit=1)
            write_meta(save_path / 'meta.json', {
                'intrinsics': K.tolist(),
                'camera_pose': c2w.tolist()
            })
        except Exception as e:
            print(f"Error processing {frame_data['sequence_name']}/{frame_data['frame_idx']}: {e}")
        gc.collect()

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_single_frame, all_frames), total=len(all_frames)))

    write_index_file(OUTPUT_DIR, all_instance_paths)
