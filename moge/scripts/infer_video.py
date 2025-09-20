import os
import sys
from pathlib import Path
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

import cv2
import torch
import click
import mediapy
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
import json

from moge.model.v1 import MoGeModel
from moge.utils.io import save_ply
from moge.utils.vis import colorize_depth, colorize_normal, colorize_depth_video
import utils3d

def read_video_frames(video_path, target_fps, max_res):
    print("==> processing video: ", video_path)
    vid = VideoReader(video_path, ctx=cpu(0))
    print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
    
    original_height, original_width = vid.get_batch([0]).shape[1:3]
    height = round(original_height / 64) * 64
    width = round(original_width / 64) * 64
    
    if max(height, width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale / 64) * 64
        width = round(original_width * scale / 64) * 64

    vid = VideoReader(video_path, ctx=cpu(0), width=original_width, height=original_height)

    fps = vid.get_avg_fps() if target_fps == -1 else target_fps
    stride = round(vid.get_avg_fps() / fps)
    stride = max(stride, 1)
    frames_idx = list(range(0, len(vid), stride))
    print(f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}")
    
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0
    return frames, fps



@click.command(help='Video Depth Inference Demo')
@click.option('--video_path', type=click.Path(exists=True), required=True, help='Path to input video file')
@click.option('--fov_x', 'fov_x_', type=float, default=None, help='If camera parameters are known, set the horizontal field of view in degrees. Otherwise, MoGe will estimate it.')
@click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default='Ruicheng/moge-vitl', help='Pretrained model name or path. Defaults to "Ruicheng/moge-vitl"')
@click.option('--output_dir', type=click.Path(), default='outputs', help='Directory to save output results')
@click.option('--save_video', is_flag=True, help='Save output as video')
@click.option('--target_fps', type=int, default=15, help='Target frames per second for video processing')
@click.option('--max_res', type=int, default=1024, help='Maximum resolution dimension')
@click.option('--depth_max', type=float, default=80, help='Maximum depth value for visualization')
@click.option('--same_intrinsic', is_flag=True, help='Use the same intrinsic matrix for all frames')
@click.option('--image_infer', is_flag=True, help='Use the same intrinsic matrix for all frames')
@click.option('--resolution_level', type=int, default=9, help='An integer [0-9] for the resolution level for inference. \
Higher value means more tokens and the finer details will be captured, but inference can be slower. \
Defaults to 9. Note that it is irrelevant to the output size, which is always the same as the input size. \
`resolution_level` actually controls `num_tokens`. See `num_tokens` for more details.')
@click.option('--num_tokens', type=int, default=None, help='number of tokens used for inference. A integer in the (suggested) range of `[1200, 2500]`. \
`resolution_level` will be ignored if `num_tokens` is provided. Default: None')
@click.option('--use_fp16', is_flag=True, help='Use fp16 precision for 2x faster inference.')
@click.option('--save_ply_', is_flag=True, help='Save the output as a ply file.')
@click.option('--save_frames_', is_flag=True, help='Save the perframe information')
@click.option('--threshold', type=float, default=0.03, help='Threshold for removing edges. Defaults to 0.03. Smaller value removes more edges. "inf" means no thresholding.')
def main(
    video_path: str,
    fov_x_: float,
    pretrained_model_name_or_path: str,
    output_dir: str,
    save_video: bool,
    target_fps: int,
    max_res: int,
    depth_max: float,
    same_intrinsic: bool,
    image_infer: bool,
    resolution_level: int,
    num_tokens: int,
    use_fp16: bool,
    save_ply_: bool,
    save_frames_: bool,
    threshold: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # frames: [n, h, w, 3], np.array
    frames, fps = read_video_frames(video_path, target_fps, max_res)

    model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()

    # frames = frames[:, 10:-10, 10:-10, :]
    frames = frames[:60, :, :, :]

    height, width = frames.shape[1:3]

    with torch.no_grad():
        # Use sliding window of size 3 with stride 1
        image_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(device)
        output = model.infer_video(image_tensor, fov_x=fov_x_, resolution_level=resolution_level, 
                                   num_tokens=num_tokens, use_fp16=use_fp16)

        points = output['points'].cpu().numpy()
        depth = output['depth'].cpu().numpy()
        mask = output['mask'].cpu().numpy()
        intrinsics = output['intrinsics'].cpu().numpy()
        # Prepare the depth visualization
        depth = np.where((depth > 0) & mask, depth, np.nan)
        disp_preds = 1 / depth

        frames_output_dir = Path(output_dir, "frames")
        save_path = Path(frames_output_dir, Path(video_path).stem)
        save_path.mkdir(exist_ok=True, parents=True)

        normals_list = []
        for i, frame in tqdm(enumerate(frames), total=len(frames), desc="Processing frames"):
            normals, normals_mask = utils3d.numpy.points_to_normals(points[i], mask=mask[i])
            normals_list.append(colorize_normal(normals))
            if save_ply_:
                faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                    points[i],
                    frame.astype(np.float32),
                    utils3d.numpy.image_uv(width=width, height=height),
                    mask=mask[i] & ~(utils3d.numpy.depth_edge(depth[i], rtol=threshold, mask=mask[i]) & utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
                    tri=True
                )
                # When exporting the model, follow the OpenGL coordinate conventions:
                # - world coordinate system: x right, y up, z backward.
                # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
                vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
                save_ply(save_path/f"{i:05d}.ply", vertices, faces, vertex_colors)
            if save_frames_:
                cv2.imwrite(str(save_path / f'image_{i:05d}.jpg'), cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_path / f'depth_vis_{i:05d}.png'), cv2.cvtColor(colorize_depth(depth[i]), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_path / f'depth_{i:05d}.exr'), depth[i], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                cv2.imwrite(str(save_path / f'normal_{i:05d}.png'), cv2.cvtColor(colorize_normal(normals), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(save_path / f'mask_{i:05d}.png'), (mask[i] * 255).astype(np.uint8))
                cv2.imwrite(str(save_path / f'points_{i:05d}.exr'), cv2.cvtColor(points[i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics[i])
                with open(save_path / 'fov.json', 'w') as f:
                    json.dump({
                        'fov_x': round(float(np.rad2deg(fov_x)), 2),
                        'fov_y': round(float(np.rad2deg(fov_y)), 2),
                    }, f)

    min_disp, max_disp = np.nanquantile(disp_preds, 0.01), np.nanquantile(disp_preds, 0.99)
    depth_preds_color = colorize_depth_video(disp_preds, min_disp=min_disp, max_disp=max_disp)
    normals = np.stack(normals_list, axis=0)

    if save_video:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Save the visualized results
        depth_preds_color = np.stack(depth_preds_color, axis=0)
        frames_np = (frames * 255).astype(np.uint8)
        combined_video = np.concatenate([frames_np, depth_preds_color, normals], axis=1)
        video_name = Path(video_path).stem
        output_path = os.path.join(output_dir, f'{video_name}_depth.mp4')
        mediapy.write_video(output_path, combined_video, fps=10, crf=18)

if __name__ == '__main__':
    main()
