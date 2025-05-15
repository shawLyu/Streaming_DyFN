import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[3]))
import zipfile
import math

from typing import *
import io
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm, trange
import numpy as np
import cv2
import utils3d

from moge.utils.io import write_depth, write_image, write_meta
from moge.utils.tools import multithead_execute, catch_exception


DOWNLOAD_DIR = 'download/ApolloSynthetic'
OUTPUT_DIR = 'data/ApolloSynthetic'

NUM_WORKERS = 32


if __name__ == "__main__":
    # Intrinsics are hard-coded for Apollo Synthetic Dataset
    intrinsics = utils3d.numpy.intrinsics_from_focal_center(2015 / 1920, 2015 / 1080, 0.5, 0.5)

    image_zip_filepaths = sorted(p for p in Path(DOWNLOAD_DIR).glob('RGB_*.zip') if 'HEAVY' in p.as_posix())
    for image_zip_filepath in (pbar := tqdm(image_zip_filepaths, desc='Top folders')):
        pbar.set_description(image_zip_filepath.name)
        depth_zip_filepath = image_zip_filepath.with_name(image_zip_filepath.name.replace('RGB', 'Depth'))

        top_folder = image_zip_filepath.stem.replace('RGB_', '')
        with (
            zipfile.ZipFile(image_zip_filepath, 'r') as image_zipfile,
            zipfile.ZipFile(depth_zip_filepath, 'r') as depth_zipfile
        ):  
            # List all frames
            image_filenames = sorted([s for s in image_zipfile.namelist() if s.endswith('.jpg')])

            @multithead_execute(image_filenames, num_workers=NUM_WORKERS)
            @catch_exception
            def process_image(image_filename: str):
                subfolder = '/'.join(image_filename.split('/')[3:-1])
                name = Path(image_filename).stem
                depth_filename = image_filename.replace('RGB', 'Depth').replace('.jpg', '.png')
                save_path = Path(OUTPUT_DIR, top_folder, subfolder, f'{name}.zip')
                # if save_path.exists():
                #     return
                assert depth_filename in depth_zipfile.namelist(), f'{depth_filename} not found in {depth_zipfile.filename}'

                image = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_zipfile.read(image_filename), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                depth = cv2.imdecode(np.frombuffer(depth_zipfile.read(depth_filename), np.uint8), cv2.IMREAD_UNCHANGED)
                depth = (depth[:, :, 2].astype(np.float32) / 255 + depth[:, :, 1].astype(np.float32) / 255 ** 2) * 65536 / 100
                depth_mask_inf = depth >= 655.3
                depth_mask = ~depth_mask_inf
                depth = np.where(depth_mask_inf, np.inf, np.where(depth_mask, depth, np.nan))

                save_path.parent.mkdir(parents=True, exist_ok=True)
                write_image(save_path / 'image.jpg', image, quality=95)
                write_depth(save_path / 'depth.png', depth, unit=1)
                write_meta(save_path /'meta.json', {'intrinsics': intrinsics.tolist()})


                
    