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

from ..utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift, gaussian_blur_2d
from .utils import wrap_dinov2_attention_with_sdpa, wrap_module_with_gradient_checkpointing, unwrap_module_with_gradient_checkpointing
from ..utils.tools import timeit


class ResidualConvBlock(nn.Module):  
    def __init__(self, in_channels: int, out_channels: int = None, hidden_channels: int = None, padding_mode: str = 'replicate', activation: Literal['relu', 'leaky_relu', 'silu', 'elu'] = 'relu', norm: Literal['group_norm', 'layer_norm'] = 'group_norm'):  
        super(ResidualConvBlock, self).__init__()  
        if out_channels is None:  
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation =='relu':
            activation_cls = lambda: nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            activation_cls = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation =='silu':
            activation_cls = lambda: nn.SiLU(inplace=True)
        elif activation == 'elu':
            activation_cls = lambda: nn.ELU(inplace=True)
        else:
            raise ValueError(f'Unsupported activation function: {activation}')

        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            activation_cls(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.GroupNorm(hidden_channels // 32 if norm == 'group_norm' else 1, hidden_channels),
            activation_cls(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        )
        
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()  
  
    def forward(self, x):  
        skip = self.skip_connection(x)  
        x = self.layers(x)
        x = x + skip
        return x  

class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float | torch.Tensor = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        gated: bool = False,
        output_dim: int | None = None,
    ):
        super().__init__()
        if gated:
            expansion = int(expansion * 2 / 3)
        hidden_dim = int(input_dim * expansion)
        if output_dim is None:
            output_dim = input_dim
        self.norm = nn.LayerNorm(input_dim)
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU() if not gated else SwiGLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = self.dropout(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.hidden_dim = dim
        context_dim = context_dim or dim
        self.mlp = MLP(dim, expansion=expansion, dropout=dropout, gated=gated)
        self.kv = nn.Linear(context_dim, dim * 2)
        self.q = nn.Linear(dim, dim)
        self.norm_attnx = nn.LayerNorm(dim)
        self.norm_attnctx = nn.LayerNorm(context_dim)
        self.cosine = cosine
        self.out = nn.Linear(dim, dim)
        self.ls1 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()
        self.ls2 = LayerScale(dim, layer_scale) if layer_scale > 0.0 else nn.Identity()

    def attn( self, x: torch.Tensor, attn_bias: torch.Tensor | None = None, 
             context: torch.Tensor | None = None, rope_q: torch.Tensor | None = None, 
             rope_k: torch.Tensor | None = None, token_lens: List[int] | None = None) -> torch.Tensor:
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(self.kv(context), "b n (kv h d) -> b h n d kv", h=self.num_heads, kv=2).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b h n d", h=self.num_heads)

        # if self.cosine:
        #     q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim
        
        if rope_q is not None and rope_k is not None:
            q = self.apply_rope(q, rope_q.unsqueeze(1))
            k = self.apply_rope(k, rope_k.unsqueeze(1))

        x = F.scaled_dot_product_attention( q, k, v, dropout_p=self.dropout)
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        return x

    def apply_rope(self, x, rope):
        x1, x2 = x[..., ::2], x[..., 1::2]
        r1, r2 = rope[..., ::2], rope[..., 1::2]
        return torch.cat([x1 * r2 + x2 * r1, x2 * r2 - x1 * r1], dim=-1)

    def forward( self, x: torch.Tensor, attn_bias: torch.Tensor | None = None, 
                context: torch.Tensor | None = None, rope_q: torch.Tensor | None = None, 
                rope_k: torch.Tensor | None = None) -> torch.Tensor:
        context = x if context is None else context
        x = self.ls1(self.attn( x, attn_bias=attn_bias, context=context, rope_q=rope_q, rope_k=rope_k ) + x)
        x = self.ls2(self.mlp(x)) + x
        return x

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, freqs)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, time_idx):
        return self.emb[time_idx]  # (seq_len, dim)

    def downsample(self, rope_emb, factor):
        pooled = rope_emb.unfold(0, factor, factor).mean(dim=-1)
        return pooled  # (seq_len // factor, dim)

class TemporalHead(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, expansion=4, dropout=0.0, layer_scale=1.0, max_len=1024):
        super().__init__()

        self.rope = RotaryPositionalEmbedding(hidden_dim, max_len=max_len)

        self.temporal_attention = AttentionBlock(
                hidden_dim, num_heads=num_heads,
                expansion=expansion, dropout=dropout,
                layer_scale=layer_scale,)

    def forward(self, hidden_states: torch.Tensor, image: torch.Tensor, sequence_length: int = 12):
        """
        feat_seq: (B*T, H*W, D)
        Returns: same shape (B*T, H*W, D)
        """
        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 14, img_w // 14
        B = image.shape[0] // sequence_length
        D = hidden_states.shape[1] # (B*T, D, H, W)
        hidden_states = hidden_states.reshape(B, sequence_length, D, patch_h, patch_w)

        updated_x = []
        for t in range(sequence_length):
            all_tokens = []
            rope_tokens = []

            for h in range(t + 1):
                feat = hidden_states[:, h, ...] # (B, D, H, W)
                if (t - h) <= 2:
                    comp = feat
                else:
                    comp = feat.mean(dim=(-2, -1)).reshape(B, D, 1, 1)
                Bh, Dh, Hh, Wh = comp.shape
                comp = comp.flatten(2).transpose(1, 2)  # (B, N, D)
                rope = self.rope(t - h).unsqueeze(0).expand(B, Hh * Wh, -1)  # (B, N, D)

                all_tokens.append(comp)
                rope_tokens.append(rope)

            if t == 0:
                updated_x.append(all_tokens[-1].reshape(B, patch_h, patch_w, D).permute(0, 3, 1, 2))
            else:
                context_tokens = torch.cat(all_tokens[:-1], dim=1)         # (B, total, D)
                rope_context = torch.cat(rope_tokens[:-1], dim=1)  # (B, total, D)

                quary_tokens = all_tokens[-1]
                rope_quary = rope_tokens[-1]

                x = self.temporal_attention(quary_tokens, context=context_tokens, rope_q=rope_quary, rope_k=rope_context)
                updated_x.append(x.reshape(B, patch_h, patch_w, D).permute(0, 3, 1, 2))

        return torch.cat(updated_x, dim=0)

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
        last_conv_size: int = 1
    ):
        super().__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=dim_proj, kernel_size=1, stride=1, padding=0,) for _ in range(num_features)
        ])

        self.temporal_attention = TemporalHead(
            hidden_dim=dim_proj, num_heads=1, expansion=4, dropout=0.0, layer_scale=1.0,
        )

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
    
    def forward_with_memory(self, hidden_states: torch.Tensor, image: torch.Tensor, memory_banks: List[torch.Tensor]):
        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 14, img_w // 14

        # Process the hidden states
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
                for proj, (feat, clstoken) in zip(self.projects, hidden_states)
        ], dim=1).sum(dim=1)

        # update memory feature
        if len(memory_banks) > 0:
            quary_feature = x
            _, D, patch_h, patch_w = quary_feature.shape
            # Update the memory banks
            for i , memory_cache in enumerate(memory_banks):
                time_gap = len(memory_banks) - i
                # memory_cache should be the [B, D, H, W]
                if time_gap == 3:
                    memory_cache = memory_cache.mean(dim=(-2, -1)).reshape(1, D, 1, 1)

            quary_rope = self.temporal_attention.rope(0).unsqueeze(0).expand(1, patch_h * patch_w, -1)

            context_tokens = []
            context_rope = []
            for j in range(len(memory_banks)):
                context_token = memory_banks[j].permute(0, 2, 3, 1).reshape(1, -1, D)
                _, num_context_tokens, _ = context_token.shape
                context_tokens.append(context_token)
                context_rope.append(self.temporal_attention.rope(len(memory_banks) - j).unsqueeze(0).expand(1, num_context_tokens, -1))
            context_tokens = torch.cat(context_tokens, dim=1)
            context_rope = torch.cat(context_rope, dim=1)

            quary_feature = quary_feature.reshape(quary_feature.shape[0], D, -1).permute(0, 2, 1)
            x = self.temporal_attention.temporal_attention(quary_feature, context=context_tokens, rope_q=quary_rope, rope_k=context_rope)
            x = x.reshape(x.shape[0], patch_h, patch_w, D).permute(0, 3, 1, 2)
            memory_banks.append(quary_feature.permute(0, 2, 1).reshape(-1, D, patch_h, patch_w))
            # memory_banks.append(x)
        else:
            memory_banks.append(x)

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
            
    def forward(self, hidden_states: torch.Tensor, image: torch.Tensor, memory_banks: List[torch.Tensor] = None):
        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 14, img_w // 14

        # Process the hidden states
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
                for proj, (feat, clstoken) in zip(self.projects, hidden_states)
        ], dim=1).sum(dim=1)

        if self.training:
            x = self.temporal_attention(x, image)
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
            last_conv_size=last_conv_size 
        )

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
        original_height, original_width = image.shape[-2:]
        
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
        output = self.head(features, image)
        points, mask = output
        
        # Make sure fp32 precision for output
        with torch.autocast(device_type=image.device.type, dtype=torch.float32):
            # Resize to original resolution
            points = F.interpolate(points, (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
            mask = F.interpolate(mask, (original_height, original_width), mode='bilinear', align_corners=False, antialias=False)
            
            # Post-process points and mask
            points, mask = points.permute(0, 2, 3, 1), mask.squeeze(1)
            points = self._remap_points(points)     # slightly improves the performance in case of very large output values
        
        return_dict = {'points': points, 'mask': mask}
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

        memory_banks = []
        points_list = []
        masks_list = []
        # Get intermediate layers from the backbone
        for i in tqdm(range(image_14.shape[0]), desc="Processing frames"):
            features = self.backbone.get_intermediate_layers(image_14[i][None, ...], self.intermediate_layers, return_class_token=True)

            output = self.head.forward_with_memory(features, image, memory_banks)
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
            'mask_prob': torch.sigmoid(mask)
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
            'mask_prob': torch.sigmoid(mask)
        }

        return return_dict