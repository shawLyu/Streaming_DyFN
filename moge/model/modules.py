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
            q = self.apply_rope(q, rope_q.unsqueeze(0))
            k = self.apply_rope(k, rope_k.unsqueeze(0))

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

    def get_rope(self, time_idx):
        return self.emb[:time_idx]

    def downsample(self, rope_emb, factor):
        pooled = rope_emb.unfold(0, factor, factor).mean(dim=-1)
        return pooled  # (seq_len // factor, dim)


class BidirectionalHead(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, expansion=4, init_values=1e-2, qk_norm=False):
        super().__init__()
        self.rope = RotaryPositionEmbedding2D(frequency=100.0)
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.block = Block(
                        dim=hidden_dim,
                        num_heads=num_heads,
                        mlp_ratio=expansion,
                        qkv_bias=True,
                        proj_bias=True,
                        ffn_bias=True,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                        attn_type="global",
                        )

    def forward(self, hidden_states: torch.Tensor):
        init_shape = hidden_states.shape
        if hidden_states.dim() == 4:
            S, D, H, W = hidden_states.shape
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(1, S * H * W, D)
            pos = self.position_getter(S, H, W, device=hidden_states.device)
            pos = pos.view(1, S, H*W, 2).view(1, S * H * W, 2)
        else:
            B, S, D = hidden_states.shape
            pos = self.position_getter(S, 1, 1, device=hidden_states.device)
            pos = pos.view(1, S, 2)

        hidden_states = self.block(hidden_states, pos=pos)
        hidden_states = hidden_states.reshape(init_shape)
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.permute(0, 2, 3, 1)
        return hidden_states

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
                if (t - h) <= 4:
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

                query_tokens = all_tokens[-1]
                rope_query = rope_tokens[-1]

                x = self.temporal_attention(query_tokens, context=context_tokens, rope_q=rope_query, rope_k=rope_context)
                updated_x.append(x.reshape(B, patch_h, patch_w, D).permute(0, 3, 1, 2))

        return torch.cat(updated_x, dim=0)

class ScaleShiftHead(nn.Module):
    def __init__(self, dim_in: int):
        super().__init__()
        self.rope = RotaryPositionalEmbedding(dim_in // 4, max_len=1024)
        self.bidirectional_head = nn.ModuleList([AttentionBlock(dim_in, num_heads=4, expansion=4) for _ in range(2)])
        self.scale_head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_in // 2),
            nn.ReLU(),
            nn.Linear(dim_in // 2, 1),
        )
    
    def forward(self, x: torch.Tensor):
        input_shape = x.shape
        x = self.bidirectional_head[0](x, rope_q=self.rope.emb[:input_shape[1]], rope_k=self.rope.emb[:input_shape[1]])
        x = self.bidirectional_head[1](x, rope_q=self.rope.emb[:input_shape[1]], rope_k=self.rope.emb[:input_shape[1]])
        x = self.scale_head(x)
        x = x.reshape(input_shape[0], input_shape[1], 1)
        return x


class RecurrentFeatureStabilizer(nn.Module):
    """
    A learnable module to stabilize feature statistics over time.
    
    It uses a GRUCell to maintain a "learnable EMA" of the feature
    statistics (represented by a hidden state). This hidden state
    is then used to predict affine parameters (gamma, beta) for
    an AdaIN-like normalization.
    """
    def __init__(self, feature_dim: int, hidden_dim: int, fixed_alpha: float = 0.1, epsilon: float = 1e-5):
        """
        Args:
            feature_dim (int): The number of channels 'd' in the input feature map.
            hidden_dim (int): The size of the GRU's hidden state.
            epsilon (float): For numerical stability in normalization.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.fixed_alpha = fixed_alpha

        self.register_buffer("ema_mean", None, persistent=True)
        self.register_buffer("ema_std", None, persistent=True)
        
        # The input to the GRU will be the global average pooled (GAP)
        # feature vector of the current frame.
        self.gru_cell = nn.GRUCell(input_size=feature_dim, hidden_size=hidden_dim)
        
        # These heads predict the *delta* (correction)
        self.gamma_delta_head = nn.Linear(hidden_dim, feature_dim)
        self.beta_delta_head = nn.Linear(hidden_dim, feature_dim)
        
        # Initialize the delta heads to output zero
        self._init_weights()

    def _init_weights(self):
        """Initializes the delta heads to output zero."""
        print("Initializing HybridRecurrentStabilizer: Delta heads set to zero.")
        for head in [self.gamma_delta_head, self.beta_delta_head]:
            nn.init.constant_(head.weight, 0.0)
            nn.init.constant_(head.bias, 0.0)

    def _update_ema_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the fixed EMA baseline statistics.
        This is the non-learnable "safety net".
        """
        # Calculate current stats (detached from graph)
        B = x.shape[0]
        mu_t = torch.mean(x, dim=[2, 3], keepdim=False).detach() # [B, D]
        std_t = torch.std(x, dim=[2, 3], keepdim=False).detach()  # [B, D]

        if self.ema_mean is None:
            # First frame
            self.ema_mean = mu_t.clone()
            self.ema_std = std_t.clone()
        else:
            # Apply fixed EMA update
            # We must update the buffer data directly
            # Handle potential batch size mismatches (e.g., last batch)
            if B == self.ema_mean.shape[0]:
                self.ema_mean.data = (self.fixed_alpha * mu_t + 
                                      (1 - self.fixed_alpha) * self.ema_mean.data)
                self.ema_std.data = (self.fixed_alpha * std_t + 
                                     (1 - self.fixed_alpha) * self.ema_std.data)
            else:
                # If batch size is different, just use the first sample
                # to update the first EMA stat (a reasonable heuristic)
                self.ema_mean.data[0:B] = (self.fixed_alpha * mu_t + 
                                        (1 - self.fixed_alpha) * self.ema_mean.data[0:B])
                self.ema_std.data[0:B] = (self.fixed_alpha * std_t + 
                                       (1 - self.fixed_alpha) * self.ema_std.data[0:B])


        return self.ema_mean, self.ema_std

    def forward(self, x: torch.Tensor, prev_state: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one time step.
        
        Args:
            x (torch.Tensor): Current frame's features, shape [B, D, H, W].
            prev_state (Optional[torch.Tensor]): Previous hidden state from the GRU.
                                                Shape: [B, hidden_dim].
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x_stable (torch.Tensor): Stabilized features, [B, D, H, W].
                - next_state (torch.Tensor): New hidden state, [B, hidden_dim].
        """
        B, D, H, W = x.shape
        
        # --- 1. Baseline Path ---
        # Update and get the fixed EMA statistics
        # mu_base and std_base have shape [B, D]
        mu_base, std_base = self._update_ema_stats(x)
        
        # --- 2. Correction Path (Learnable) ---
        # Get input for GRU
        x_pooled = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1) # [B, D]
        
        if prev_state is None:
            prev_state = torch.zeros(B, self.hidden_dim, dtype=x.dtype, device=x.device)
            
        # Update the hidden state
        next_state = self.gru_cell(x_pooled, prev_state)
        
        # Predict the *correction* (delta)
        # These have shape [B, D]
        gamma_delta = self.gamma_delta_head(next_state)
        beta_delta = self.beta_delta_head(next_state)
        
        # --- 3. Combine Paths ---
        # Add the baseline and the correction
        # We must clamp gamma to be positive
        gamma_final = torch.clamp(std_base + gamma_delta, min=self.epsilon)
        beta_final = mu_base + beta_delta
        
        # --- 4. Apply Normalization ---
        # Calculate current frame's stats (Instance Norm part)
        mu_x = torch.mean(x, dim=[2, 3], keepdim=True)  # [B, D, 1, 1]
        std_x = torch.std(x, dim=[2, 3], keepdim=True)   # [B, D, 1, 1]
        
        # Normalize
        x_norm = (x - mu_x) / (std_x + self.epsilon)
        
        # Reshape gamma/beta for broadcasting: [B, D] -> [B, D, 1, 1]
        gamma_final = gamma_final.unsqueeze(-1).unsqueeze(-1)
        beta_final = beta_final.unsqueeze(-1).unsqueeze(-1)
        
        # Apply combined scale and shift
        x_stable = gamma_final * x_norm + beta_final
        
        return x_stable, next_state