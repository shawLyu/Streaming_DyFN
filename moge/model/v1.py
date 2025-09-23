from typing import *
from numbers import Number
from functools import partial
from pathlib import Path
import importlib
import warnings
import json

from tqdm import tqdm
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.version
import utils3d
from huggingface_hub import hf_hub_download
# from timm.models.layers import trunc_normal_

from .layers.block import Block
from .layers.rope import RotaryPositionEmbedding2D, PositionGetter
from ..utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift, gaussian_blur_2d
from .utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing, unwrap_module_with_gradient_checkpointing
from ..utils.tools import timeit
from .modules import *

class Head(nn.Module):
    def __init__(
        self, 
        num_features: int,
        dim_in: int, 
        dim_out: List[int], 
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
        stabilizer_hidden_dim: int = 256,
        stabilizer_type: Literal['GRU', 'ConvGRU'] = 'ConvGRU',
    ):
        super().__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=dim_proj, kernel_size=1, stride=1, padding=0,) for _ in range(num_features)
        ])

        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_upsampler(in_ch + 2, out_ch),
                *(ResidualConvBlock(out_ch, out_ch, dim_times_res_block_hidden * out_ch, activation="relu", norm=res_block_norm) for _ in range(num_res_blocks))
            ) for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
        ])

        self.output_block = nn.ModuleList([
            self._make_output_block(
                dim_upsample[-1] + 2, dim_out_, dim_times_res_block_hidden, last_res_blocks, last_conv_channels, last_conv_size, res_block_norm,
            ) for dim_out_ in dim_out
        ])

        if stabilizer_type == 'GRU':
            self.stabilizer = RecurrentFeatureStabilizerGRU(dim_proj, stabilizer_hidden_dim, epsilon=1e-5)
        elif stabilizer_type == 'ConvGRU':
            self.stabilizer = RecurrentFeatureStabilizer(dim_proj, stabilizer_hidden_dim, epsilon=1e-5)
        else:
            raise ValueError(f"Invalid stabilizer type: {stabilizer_type}")
    
    def _make_upsampler(self, in_channels: int, out_channels: int):
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        )
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(self, dim_in: int, dim_out: int, dim_times_res_block_hidden: int, last_res_blocks: int, last_conv_channels: int, last_conv_size: int, res_block_norm: Literal['group_norm', 'layer_norm']):
        return nn.Sequential(
            nn.Conv2d(dim_in, last_conv_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            *(ResidualConvBlock(last_conv_channels, last_conv_channels, dim_times_res_block_hidden * last_conv_channels, activation='relu', norm=res_block_norm) for _ in range(last_res_blocks)),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_conv_channels, dim_out, kernel_size=last_conv_size, stride=1, padding=last_conv_size // 2, padding_mode='replicate'),
        )

    def forward_recurrent(
        self, 
        hidden_states: torch.Tensor, 
        image: torch.Tensor, 
        prev_state: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        inference_mode: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single frame in a sequence, using the
        recurrent stabilizer.
        
        Args:
            hidden_states (torch.Tensor): Features from the DINOv2 backbone.
            image (torch.Tensor): The input image.
            prev_state (Optional[torch.Tensor]): The previous hidden state from
                                                 the stabilizer (h_{t-1}).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - output: The final depth prediction(s).
                - next_state: The new hidden state (h_t) to be passed to the next frame.
                - feature_before_stabilizer: Features just after projection (for debugging).
                - feature_after_stabilizer: Stabilized features (for debugging).
        """
        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 14, img_w // 14

        # Process the hidden states
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
                for proj, (feat, clstoken) in zip(self.projects, hidden_states)
        ], dim=1).sum(dim=1)

        features_before_stabilizer = x.clone()
        if not inference_mode:
            feature_after_stabilizer = []
            x = x.reshape(batch_size, -1, *x.shape[1:]).permute(1, 0, 2, 3, 4)
            # --- Apply Recurrent Stabilizer ---
            for i in range(x.shape[0]):
                stabilized_feature, prev_state = self.stabilizer(x[i], prev_state)
                feature_after_stabilizer.append(stabilized_feature[:, None, ...])
            self.stabilizer.ema_mean = None
            self.stabilizer.ema_std = None
            x = torch.cat(feature_after_stabilizer, dim=1)
            x = x.reshape(-1, *x.shape[2:])
        else:
            x, prev_state = self.stabilizer(x, prev_state)
        features_after_stabilizer = x.clone()


        # --- Upsample stage (Identical to your simple 'forward' method) ---
        for i, block in enumerate(self.upsample_blocks):
            uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        
        # --- Final Interpolation and Output (Identical to 'forward') ---
        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], dim=1)

        if isinstance(self.output_block, nn.ModuleList):
            output = [torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False) for block in self.output_block]
        else:
            output = torch.utils.checkpoint.checkpoint(self.output_block, x, use_reentrant=False)
        
        if inference_mode:
            return output, prev_state, features_before_stabilizer, features_after_stabilizer
        else:
            return output, prev_state
            
            
    def forward(self, hidden_states: torch.Tensor, image: torch.Tensor, memory_banks: List[torch.Tensor] = None):
        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 14, img_w // 14

        # Process the hidden states
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
                for proj, (feat, clstoken) in zip(self.projects, hidden_states)
        ], dim=1).sum(dim=1)

        # Upsample stage
        # (patch_h, patch_w) -> (patch_h * 2, patch_w * 2) -> (patch_h * 4, patch_w * 4) -> (patch_h * 8, patch_w * 8)
        for i, block in enumerate(self.upsample_blocks):
            # UV coordinates is for awareness of image aspect ratio
            uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        
        # (patch_h * 8, patch_w * 8) -> (img_h, img_w)
        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], dim=1)

        if isinstance(self.output_block, nn.ModuleList):
            output = [torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False) for block in self.output_block]
        else:
            output = torch.utils.checkpoint.checkpoint(self.output_block, x, use_reentrant=False)
        
        return output


class MoGeModel(nn.Module):
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(self, 
        encoder: str = 'dinov2_vitb14', 
        intermediate_layers: Union[int, List[int]] = 4,
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        remap_output: Literal[False, True, 'linear', 'sinh', 'exp', 'sinh_exp'] = 'linear',
        res_block_norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        num_tokens_range: Tuple[Number, Number] = [1200, 2500],
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
        mask_threshold: float = 0.5,
        stabilizer_type: Literal['GRU', 'ConvGRU'] = 'GRU',
        **deprecated_kwargs
    ):
        super(MoGeModel, self).__init__()

        if deprecated_kwargs:
            # Process legacy arguments
            if 'trained_area_range' in deprecated_kwargs:
                num_tokens_range = [deprecated_kwargs['trained_area_range'][0] // 14 ** 2, deprecated_kwargs['trained_area_range'][1] // 14 ** 2]
                del deprecated_kwargs['trained_area_range']
            warnings.warn(f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}")

        self.encoder = encoder
        self.remap_output = remap_output
        self.intermediate_layers = intermediate_layers
        self.num_tokens_range = num_tokens_range
        self.mask_threshold = mask_threshold
        
        # NOTE: We have copied the DINOv2 code in torchhub to this repository.
        # Minimal modifications have been made: removing irrelevant code, unnecessary warnings and fixing importing issues.
        hub_loader = getattr(importlib.import_module(".dinov2.hub.backbones", __package__), encoder)
        self.backbone = hub_loader(pretrained=False)
        dim_feature = self.backbone.blocks[0].attn.qkv.in_features
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = Head(
            num_features=intermediate_layers if isinstance(intermediate_layers, int) else len(intermediate_layers), 
            dim_in=dim_feature, 
            dim_out=[3, 1], 
            dim_proj=dim_proj,
            dim_upsample=dim_upsample,
            dim_times_res_block_hidden=dim_times_res_block_hidden,
            num_res_blocks=num_res_blocks,
            res_block_norm=res_block_norm,
            last_res_blocks=last_res_blocks,
            last_conv_channels=last_conv_channels,
            last_conv_size=last_conv_size,
            stabilizer_type=stabilizer_type
        )

        # self.scale_shift_head = ScaleShiftHead(dim_in=dim_feature)

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)
        
        if torch.__version__ >= '2.0':
            self.enable_pytorch_native_sdpa()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path, IO[bytes]], model_kwargs: Optional[Dict[str, Any]] = None, **hf_kwargs) -> 'MoGeModel':
        """
        Load a model from a checkpoint file.

        ### Parameters:
        - `pretrained_model_name_or_path`: path to the checkpoint file or repo id.
        - `model_kwargs`: additional keyword arguments to override the parameters in the checkpoint.
        - `hf_kwargs`: additional keyword arguments to pass to the `hf_hub_download` function. Ignored if `pretrained_model_name_or_path` is a local path.

        ### Returns:
        - A new instance of `MoGe` with the parameters loaded from the checkpoint.
        """
        if Path(pretrained_model_name_or_path).exists():
            checkpoint = torch.load(pretrained_model_name_or_path, map_location='cpu', weights_only=True)
        else:
            cached_checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename="model.pt",
                **hf_kwargs
            )
            checkpoint = torch.load(cached_checkpoint_path, map_location='cpu', weights_only=True)
        model_config = checkpoint['model_config']
        if model_kwargs is not None:
            model_config.update(model_kwargs)
        model = cls(**model_config)
        model.load_state_dict(checkpoint['model'])
        return model

    def init_weights(self):
        "Load the backbone with pretrained dinov2 weights from torch hub"
        # state_dict = torch.hub.load('facebookresearch/dinov2', self.encoder, pretrained=True).state_dict()
        print("Loading from the local path")
        state_dict = torch.load("/home/luban/.cache/huggingface/hub/checkpoints/dinov2_vitl14_pretrain.pth", 
                                map_location="cpu", weights_only=True)
        
        # Check if state dict loaded successfully
        if state_dict is None:
            raise RuntimeError("Failed to load pretrained weights")
            
        # Count number of parameters in state dict
        num_params = sum(p.numel() for p in state_dict.values())
        print(f"Loaded state dict with {num_params:,} parameters")
        
        # Try loading into backbone and catch any errors
        try:
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            if len(missing_keys) > 0:
                print(f"Missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"Unexpected keys: {unexpected_keys}")
            print("Successfully loaded pretrained weights into backbone")
        except Exception as e:
            raise RuntimeError(f"Failed to load state dict into backbone: {str(e)}")
    
    def enable_gradient_checkpointing(self):
        for i in range(len(self.backbone.blocks)):
            self.backbone.blocks[i] = wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])

    def enable_pytorch_native_sdpa(self):
        for i in range(len(self.backbone.blocks)):
            self.backbone.blocks[i].attn = wrap_dinov2_attention_with_sdpa(self.backbone.blocks[i].attn)
    
    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.remap_output == 'linear':
            pass
        elif self.remap_output =='sinh':
            points = torch.sinh(points)
        elif self.remap_output == 'exp':
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output =='sinh_exp':
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points

    def forward(self, image: torch.Tensor, num_tokens: int) -> Dict[str, torch.Tensor]:

        B, S, C, original_height, original_width = image.shape
        image = image.reshape(B * S, C, original_height, original_width)
        
        # Resize to expected resolution defined by num_tokens
        resize_factor = ((num_tokens * 14 ** 2) / (original_height * original_width)) ** 0.5
        resized_width, resized_height = int(original_width * resize_factor), int(original_height * resize_factor)
        image = F.interpolate(image, (resized_height, resized_width), mode="bicubic", align_corners=False, antialias=True)
    
        # Apply image transformation for DINOv2
        image = (image - self.image_mean) / self.image_std
        image_14 = F.interpolate(image, (resized_height // 14 * 14, resized_width // 14 * 14), mode="bilinear", align_corners=False, antialias=True)

        # Get intermediate layers from the backbone
        features = self.backbone.get_intermediate_layers(image_14, self.intermediate_layers, return_class_token=True)

        # Predict points (and mask)
        output, _ = self.head.forward_recurrent(features, image, batch_size=B)
        points, mask = output

        # # Predict scale and shift from the class token
        # cls_token = features[-1][1].unsqueeze(0)
        # scale = self.scale_shift_head(cls_token)
        # # scale, shift = scale_shift.split(1, dim=-1)
        # scale = scale.exp()
        # print(scale)
        
        # Make sure fp32 precision for output
        with torch.autocast(device_type=image.device.type, dtype=torch.float32):
            # Resize to original resolution
            points = F.interpolate(points, (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
            mask = F.interpolate(mask, (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
            
            # Post-process points and mask
            points, mask = points.permute(0, 2, 3, 1), mask.squeeze(1)
            points = self._remap_points(points)     # slightly improves the performance in case of very large output values

        # points = points * scale.squeeze(0)[..., None, None] + shift.squeeze(0)[..., None, None]
        # points = points * scale.squeeze(0)[..., None, None]
        # points = points
        
        # return_dict = {'points': points, 'mask': mask, "features": features, "metric_scale": scale}
        return_dict = {'points': points, 'mask': mask, "features": features}
        return return_dict


    @torch.inference_mode()
    def infer_video(self, 
        image: torch.Tensor, 
        fov_x: Union[Number, torch.Tensor] = None,
        resolution_level: int = 9,
        num_tokens: int = None,
        apply_mask: bool = True,
        force_projection: bool = True,
        use_fp16: bool = True,
    ) -> Dict[str, torch.Tensor]:
        
        original_height, original_width = image.shape[-2:]
        area = original_height * original_width
        aspect_ratio = original_width / original_height

        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        # Resize to expected resolution defined by num_tokens
        resize_factor = ((num_tokens * 14 ** 2) / (original_height * original_width)) ** 0.5
        resized_width, resized_height = int(original_width * resize_factor), int(original_height * resize_factor)
        image = F.interpolate(image, (resized_height, resized_width), mode="bicubic", align_corners=False, antialias=True)
        # Apply image transformation for DINOv2
        image = (image - self.image_mean) / self.image_std
        image_14 = F.interpolate(image, (resized_height // 14 * 14, resized_width // 14 * 14), mode="bilinear", align_corners=False, antialias=True)

        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 14, img_w // 14

        points_list = []
        masks_list = []
        feature_list = []
        # feature_after_temporal_attention_list = []
        # feature_before_temporal_attention_list = []
        # Get intermediate layers from the backbone
        prev_state = None
        feats = [[] for _ in range(self.intermediate_layers)]
        cls_tokens = [[] for _ in range(self.intermediate_layers)]
        features_before_stabilizer_list = []
        features_after_stabilizer_list = []
        for i in tqdm(range(image_14.shape[0]), desc="Processing frames"):
            features = self.backbone.get_intermediate_layers(image_14[i][None, ...], self.intermediate_layers, return_class_token=True)
            output, prev_state, features_before_stabilizer, features_after_stabilizer = self.head.forward_recurrent(features, image[i][None, ...], prev_state, inference_mode=True)
            features_before_stabilizer_list.append(features_before_stabilizer)
            features_after_stabilizer_list.append(features_after_stabilizer)
            points, mask = output
    
            # Make sure fp32 precision for output
            with torch.autocast(device_type=image.device.type, dtype=torch.float32):
                # Resize to original resolution
                points = F.interpolate(points, (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
                mask = F.interpolate(mask, (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
                
                # Post-process points and mask
                points, mask = points.permute(0, 2, 3, 1), mask.squeeze(1)
                points = self._remap_points(points)     # slightly improves the performance in case of very large output values
                points_list.append(points)
                masks_list.append(mask)
        
        features_before_stabilizer_list = torch.concat(features_before_stabilizer_list, dim=0)
        features_after_stabilizer_list = torch.concat(features_after_stabilizer_list, dim=0)
        points = torch.concat(points_list, dim=0)
        mask = torch.concat(masks_list, dim=0)

        mask_binary = mask > self.mask_threshold

        # Get camera-space point map. (Focal here is the focal length relative to half the image diagonal)
        if fov_x is None:
            focal, shift = recover_focal_shift(points, mask_binary)
            focal = focal.mean(dim=0).expand(points.shape[0])
            shift = shift.mean(dim=0).expand(points.shape[0])
        else:
            focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))
            if focal.ndim == 0:
                focal = focal[None].expand(points.shape[0])
            _, shift = recover_focal_shift(points, mask_binary, focal=focal)
        fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
        fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 
        intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
        depth = points[..., 2] + shift[..., None, None]

        # If projection constraint is forced, recompute the point map using the actual depth map
        if force_projection:
            points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)
        else:
            points = points + torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)[..., None, None, :]

        # Apply mask if needed
        if apply_mask:
            points = torch.where(mask_binary[..., None], points, torch.inf)
            depth = torch.where(mask_binary, depth, torch.inf)

        return_dict = {
            'points': points,
            'intrinsics': intrinsics,
            'depth': depth,
            'mask': mask_binary,
            'mask_prob': torch.sigmoid(mask),
            'feature_after_temporal_attention': features_after_stabilizer_list,
            'feature_before_temporal_attention': features_before_stabilizer_list,
        }

        return return_dict

    @torch.inference_mode()
    def infer(
        self, 
        image: torch.Tensor, 
        fov_x: Union[Number, torch.Tensor] = None,
        resolution_level: int = 9,
        num_tokens: int = None,
        apply_mask: bool = True,
        force_projection: bool = True,
        use_fp16: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        User-friendly inference function

        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W)\
        - `fov_x`: the horizontal camera FoV in degrees. If None, it will be inferred from the predicted point map. Default: None
        - `resolution_level`: An integer [0-9] for the resolution level for inference. 
            The higher, the finer details will be captured, but slower. Defaults to 9. Note that it is irrelevant to the output size, which is always the same as the input size.
            `resolution_level` actually controls `num_tokens`. See `num_tokens` for more details.
        - `num_tokens`: number of tokens used for inference. A integer in the (suggested) range of `[1200, 2500]`.
            `resolution_level` will be ignored if `num_tokens` is provided. Default: None
        - `apply_mask`: if True, the output point map will be masked using the predicted mask. Default: True
        - `force_projection`: if True, the output point map will be recomputed to match the projection constraint. Default: True
        - `use_fp16`: if True, use mixed precision to speed up inference. Default: True
            
        ### Returns

        A dictionary containing the following keys:
        - `points`: output tensor of shape (B, H, W, 3) or (H, W, 3).
        - `depth`: tensor of shape (B, H, W) or (H, W) containing the depth map.
        - `intrinsics`: tensor of shape (B, 3, 3) or (3, 3) containing the camera intrinsics.
        """
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
            num_frames = None
        else:
            num_frames = image.shape[0]
            omit_batch_dim = False

        original_height, original_width = image.shape[-2:]
        area = original_height * original_width
        aspect_ratio = original_width / original_height

        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))
        
        with torch.autocast(device_type=image.device.type, dtype=torch.float16, enabled=use_fp16):
            output = self.forward(image, num_tokens)

        # # points, mask, scale, shift = output['points'], output['mask'], output['metric_scale'], output['shift']
        # points, mask, scale, shift = output['points'], output['mask'], output['metric_scale'], None
        points, mask = output['points'], output['mask']

        mask_binary = mask > self.mask_threshold

        # Get camera-space point map. (Focal here is the focal length relative to half the image diagonal)
        if fov_x is None:
            focal, shift = recover_focal_shift(points, mask_binary)
        else:
            focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))
            if focal.ndim == 0:
                focal = focal[None].expand(points.shape[0])
            _, shift = recover_focal_shift(points, mask_binary, focal=focal)
        fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
        fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 
        intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
        depth = points[..., 2] + shift[..., None, None]

        # If projection constraint is forced, recompute the point map using the actual depth map
        if force_projection:
            points = utils3d.torch.depth_to_points(depth, intrinsics=intrinsics)
        else:
            points = points + torch.stack([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)[..., None, None, :]

        # Apply mask if needed
        if apply_mask:
            points = torch.where(mask_binary[..., None], points, torch.inf)
            depth = torch.where(mask_binary, depth, torch.inf)

        if omit_batch_dim:
            points = points.squeeze(0)
            intrinsics = intrinsics.squeeze(0)
            depth = depth.squeeze(0)
            mask_binary = mask_binary.squeeze(0)
            mask = mask.squeeze(0)

        return_dict = {
            'points': points,
            'intrinsics': intrinsics,
            'depth': depth,
            'mask': mask_binary,
            'mask_prob': torch.sigmoid(mask),
            "features": output["features"]
        }

        return return_dict