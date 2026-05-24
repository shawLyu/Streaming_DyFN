# Evaluation

This document describes two evaluation settings:

1. video geometry estimation on video benchmarks
2. image geometry estimation on monocular benchmarks

## Video Geometry Estimation

Video geometry evaluation uses [`moge/scripts/eval_video_baseline.py`](../moge/scripts/eval_video_baseline.py). This script loads a DyFN/MoGe checkpoint, runs `model.infer_video(...)` on each video, aligns the predicted depth to ground truth, and reports depth metrics for each dataset.

### Dataset Layout

Pass the root directory of the video benchmark with `--video_dir_path`. The script expects one subdirectory per dataset. By default it evaluates:

```text
sintel
scannet
KITTI
bonn
```

Each dataset directory should contain matching `.mp4` videos and `.npz` ground-truth depth files. For example:

```text
datasets/
|-- sintel/
|   |-- scene_001.mp4
|   `-- scene_001.npz
|-- scannet/
|   |-- scene_002.mp4
|   `-- scene_002.npz
|-- KITTI/
`-- bonn/
```

### Generate Evaluation Data

We follow the [DepthCrafter benchmark](https://github.com/Tencent/DepthCrafter/tree/main/benchmark) data format for video evaluation. DepthCrafter provides dataset extraction scripts under `benchmark/dataset_extract/`. These scripts convert raw benchmark frames into paired RGB videos and ground-truth depth files:

- RGB video: `*.mp4`
- Ground truth: `*.npz` with key `disparity` and shape `(T, 1, H, W)`
- Metadata: a CSV file with `filepath_left` and `filepath_disparity`

First clone DepthCrafter and install the packages needed by its extraction scripts:

```bash
git clone https://github.com/Tencent/DepthCrafter.git
cd DepthCrafter
pip install imageio imageio-ffmpeg pillow tqdm numpy
```

Edit the paths in the `if __name__ == "__main__"` block of each extraction script so they point to your local raw datasets and to a shared output root, for example `./benchmark/datasets/`.

For Sintel:

```python
extract_sintel(
    root="/path/to/Sintel-Depth/training_image/clean",
    depth_root="/path/to/Sintel-Depth/MPI-Sintel-depth-training-20150305/training/depth",
    saved_rgb_dir="./benchmark/datasets/",
    saved_disp_dir="./benchmark/datasets/",
    csv_save_path="./benchmark/datasets/sintel.csv",
    sample_len=-1,
    datatset_name="sintel",
)
```

For ScanNet:

```python
extract_scannet(
    root="/path/to/ScanNet_v2/raw/scans_test",
    saved_rgb_dir="./benchmark/datasets/",
    saved_disp_dir="./benchmark/datasets/",
    csv_save_path="./benchmark/datasets/scannet.csv",
    sample_len=-1,
    datatset_name="scannet",
    scene_number=100,
    scene_frames_len=90 * 3,
    stride=3,
)
```

For KITTI:

```python
extract_kitti(
    root="/path/to/KITTI/raw_data",
    depth_root="/path/to/KITTI/data_depth_annotated/val",
    saved_rgb_dir="./benchmark/datasets/",
    saved_disp_dir="./benchmark/datasets/",
    csv_save_path="./benchmark/datasets/KITTI.csv",
    sample_len=-1,
    datatset_name="KITTI",
    start_frame=0,
    end_frame=110,
)
```

For Bonn:

```python
extract_bonn(
    root="/path/to/Bonn-RGBD",
    depth_root="/path/to/Bonn-RGBD",
    saved_rgb_dir="./benchmark/datasets/",
    saved_disp_dir="./benchmark/datasets/",
    csv_save_path="./benchmark/datasets/bonn.csv",
    sample_len=-1,
    datatset_name="bonn",
    start_frame=30,
    end_frame=140,
)
```

Then run the extraction scripts:

```bash
python benchmark/dataset_extract/dataset_extract_sintel.py
python benchmark/dataset_extract/dataset_extract_scannet.py
python benchmark/dataset_extract/dataset_extract_kitti.py
python benchmark/dataset_extract/dataset_extract_bonn.py
```

After extraction, the output root should be usable directly as `--video_dir_path`:

```text
benchmark/datasets/
|-- sintel/
|   |-- alley_1_rgb_left.mp4
|   `-- alley_1_disparity.npz
|-- scannet/
|-- KITTI/
`-- bonn/
```

You can either evaluate from the DepthCrafter output path directly or copy/symlink it into your local data directory:

```bash
python moge/scripts/eval_video_baseline.py --video_dir_path /path/to/DepthCrafter/benchmark/datasets --pretrained ./pretrained/model.pt
```

### Run Evaluation

Use the pretrained checkpoint with `--pretrained` and the video benchmark root with `--video_dir_path`:

```bash
python moge/scripts/eval_video_baseline.py --video_dir_path ~/data_disk/dataset/local/depthcrafter/datasets/ --pretrained ./pretrained/model.pt
```

By default, results are saved to `outputs_video/`:

```text
outputs_video/
|-- evaluation_results.json
|-- sintel_per_scene_results.json
|-- scannet_per_scene_results.json
|-- KITTI_per_scene_results.json
`-- bonn_per_scene_results.json
```

### Useful Options

```text
--video_dir_path PATH   Root directory containing the video benchmark datasets. Required.
--pretrained PATH       Pretrained model name or local checkpoint path. Defaults to Ruicheng/moge-vitl.
--output_dir PATH       Directory for JSON results and optional videos. Defaults to outputs_video.
--align_method TEXT     Depth alignment method: lstsq, searching, or all. Defaults to all.
--save_video            Save side-by-side RGB/depth visualization videos.
--target_fps INTEGER    Target FPS for frame sampling. Defaults to 15.
--max_res INTEGER       Maximum video resolution dimension. Defaults to 1024.
--resolution_level INT  Inference resolution level from 0 to 9. Defaults to 9.
--num_tokens INTEGER    Override resolution_level by specifying the number of tokens.
--use_fp16              Use fp16 inference for faster evaluation.
--image_based           Run image-based inference instead of video-based inference.
--silent                Reduce printed logs.
```

To save visualization videos as well as metrics:

```bash
python moge/scripts/eval_video_baseline.py --video_dir_path ~/data_disk/dataset/local/depthcrafter/datasets/ --pretrained ./pretrained/model.pt --save_video
```

## Image Geometry Estimation

We provide a unified evaluation script that runs baselines on multiple monocular geometry benchmarks. It takes a baseline model and an evaluation configuration, evaluates samples on the fly, and writes the results to a JSON file.

### Benchmarks

Download the processed datasets from [Hugging Face Datasets](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation) and put them in the `data/eval` directory:

```bash
mkdir -p data/eval
huggingface-cli download Ruicheng/monocular-geometry-evaluation --repo-type dataset --local-dir data/eval --local-dir-use-symlinks False
```

Then unzip the downloaded files:

```bash
cd data/eval
unzip '*.zip'
# rm *.zip # optional: remove the zip files after extraction
```

### Configuration

See [`configs/eval/all_benchmarks.json`](../configs/eval/all_benchmarks.json) for an example configuration covering all monocular benchmarks. You can modify this file to evaluate different benchmarks or different baselines.

### Baseline

Some examples of baselines are provided in [`baselines/`](../baselines/). Pass the path to the baseline model Python file with the `--baseline` argument.

### Run Evaluation

Run [`moge/scripts/eval_baseline.py`](../moge/scripts/eval_baseline.py):

```bash
# Evaluate MoGe on the 10 monocular benchmarks
python moge/scripts/eval_baseline.py --baseline baselines/moge.py --config configs/eval/all_benchmarks.json --output eval_output/moge.json --pretrained Ruicheng/moge-vitl --resolution_level 9

# Evaluate Depth Anything V2 on the 10 monocular benchmarks. NOTE: affine disparity.
python moge/scripts/eval_baseline.py --baseline baselines/da_v2.py --config configs/eval/all_benchmarks.json --output eval_output/da_v2.json
```

The `--baseline`, `--config`, and `--output` arguments are consumed by the evaluation script. Extra arguments, such as `--pretrained` and `--resolution_level`, are forwarded to the selected baseline loader.

Main arguments:

```text
Usage: eval_baseline.py [OPTIONS]

  Evaluation script.

Options:
  --baseline PATH  Path to the baseline model python code.
  --config PATH    Path to the evaluation configurations. Defaults to
                   "configs/eval/all_benchmarks.json".
  --output PATH    Path to the output json file.
  --oracle         Use oracle mode for evaluation, i.e., use the GT intrinsics
                   input.
  --dump_pred      Dump prediction results.
  --dump_gt        Dump ground truth.
  --help           Show this message and exit.
```

### Wrap a Customized Baseline

Wrap any baseline method with [`moge.test.baseline.MGEBaselineInterface`](../moge/test/baseline.py). See [`baselines/`](../baselines/) for more examples.

It is useful to check the baseline implementation by running inference on a small set of images with [`moge/scripts/infer_baselines.py`](../moge/scripts/infer_baselines.py):

```bash
python moge/scripts/infer_baselines.py --baseline baselines/moge.py --input example_images/ --output infer_output/moge --pretrained Ruicheng/moge-vitl --maps --ply
```
