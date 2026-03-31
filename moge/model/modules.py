from typing import *
from numbers import Number
from functools import partial
from pathlib import Path
import importlib
import warnings
import json

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.version
import utils3d
from huggingface_hub import hf_hub_download

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


class ConvGRUCell(nn.Module):
    """
    A basic Convolutional GRU cell.

    Args:
        input_dim (int): Number of channels in the input feature map.
        hidden_dim (int): Number of channels in the hidden state feature map.
        kernel_size (int): Size of the convolutional kernel.
        bias (bool): Whether to add a learnable bias.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool = True):
        super(ConvGRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # Preserve spatial resolution
        self.bias = bias

        # A single convolution layer to compute gates for both input and hidden state, for efficiency.
        # This layer computes the update and reset gates.
        self.conv_gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=2 * hidden_dim,  # for update_gate, reset_gate
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

        # This layer computes the candidate hidden state.
        self.conv_candidate = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=hidden_dim,  # for the new_gate (candidate)
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor: torch.Tensor, h_cur: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_tensor (torch.Tensor): Input tensor, shape [B, C_in, H, W].
            h_cur (Optional[torch.Tensor]): Current hidden state, shape [B, C_hid, H, W].

        Returns:
            torch.Tensor: Next hidden state, shape [B, C_hid, H, W].
        """
        if h_cur is None:
            # Initialize hidden state with zeros if it's the first timestep
            h_cur = torch.zeros(
                input_tensor.shape[0], self.hidden_dim, 
                input_tensor.shape[2], input_tensor.shape[3],
                device=input_tensor.device
            )

        # Concatenate input and hidden state along the channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # Compute the reset and update gates
        combined_conv = self.conv_gates(combined)
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        # Compute the candidate hidden state (new gate)
        # Here we apply the reset gate to the hidden state
        combined_candidate = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        candidate_h = torch.tanh(self.conv_candidate(combined_candidate))

        # Compute the next hidden state
        h_next = (1 - update_gate) * h_cur + update_gate * candidate_h
        return h_next


class RecurrentFeatureStabilizerGRU(nn.Module):
    """
    A learnable module to stabilize feature statistics over time.

    It uses a GRUCell to maintain a "learnable EMA" of the feature
    statistics (represented by a hidden state). This hidden state
    is then used to predict affine parameters (gamma, beta) for
    an AdaIN-like normalization.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        fixed_alpha: float = 0.1,
        epsilon: float = 1e-5,
    ):
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
        mu_t = torch.mean(x, dim=[2, 3], keepdim=False).detach()  # [B, D]
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
                self.ema_mean.data = (
                    self.fixed_alpha * mu_t + (1 - self.fixed_alpha) * self.ema_mean.data
                )
                self.ema_std.data = (
                    self.fixed_alpha * std_t + (1 - self.fixed_alpha) * self.ema_std.data
                )
            else:
                # If batch size is different, just use the first sample
                # to update the first EMA stat (a reasonable heuristic)
                self.ema_mean.data[0:B] = (
                    self.fixed_alpha * mu_t + (1 - self.fixed_alpha) * self.ema_mean.data[0:B]
                )
                self.ema_std.data[0:B] = (
                    self.fixed_alpha * std_t + (1 - self.fixed_alpha) * self.ema_std.data[0:B]
                )
        return self.ema_mean, self.ema_std

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
        ema_only: bool = False,
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
        # [B, D]
        x_pooled = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)

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
        if not ema_only:
            gamma_final = torch.clamp(std_base + gamma_delta, min=self.epsilon)
            beta_final = mu_base + beta_delta
        else:
            gamma_final = std_base
            beta_final = mu_base

        # --- 4. Apply Normalization ---
        # Calculate current frame's stats (Instance Norm part)
        mu_x = torch.mean(x, dim=[2, 3], keepdim=True)  # [B, D, 1, 1]
        std_x = torch.std(x, dim=[2, 3], keepdim=True)  # [B, D, 1, 1]

        # Normalize
        x_norm = (x - mu_x) / (std_x + self.epsilon)

        # Reshape gamma/beta for broadcasting: [B, D] -> [B, D, 1, 1]
        gamma_final = gamma_final.unsqueeze(-1).unsqueeze(-1)
        beta_final = beta_final.unsqueeze(-1).unsqueeze(-1)

        # Apply combined scale and shift
        x_stable = gamma_final * x_norm + beta_final

        return x_stable, next_state

# --- 2. Your Modified RecurrentFeatureStabilizer ---

class RecurrentFeatureStabilizerConvGRU(nn.Module):
    """
    A module that uses ConvGRU to stabilize features spatially.
    It directly learns to predict the affine parameters (gamma and beta) for AdaIN.
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        epsilon: float = 1e-5,
    ):
        """
        Args:
            feature_dim (int): Number of channels 'd' in the input feature map.
            hidden_dim (int): Number of channels in the ConvGRU's hidden state.
            kernel_size (int): Size of the kernel within the ConvGRU cell.
            epsilon (float): A small value for numerical stability.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon

        # Core component: The custom ConvGRUCell we built
        self.conv_gru_cell = ConvGRUCell(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size
        )

        # Prediction heads: Use 1x1 convolutions to map the hidden state to gamma and beta.
        # These are now tensors with the same spatial dimensions as the feature map.
        self.gamma_head = nn.Conv2d(hidden_dim, feature_dim, kernel_size=1)
        self.beta_head = nn.Conv2d(hidden_dim, feature_dim, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights so the module acts as an identity transformation
        at the beginning of training. Gamma is initialized to 1, and beta to 0.
        """
        print("Initializing ConvGRU Stabilizer: gamma->0, beta->0.")
        nn.init.constant_(self.gamma_head.weight, 0.0)
        nn.init.constant_(self.gamma_head.bias, 1.0) # Initialize bias to output 1
        nn.init.constant_(self.beta_head.weight, 0.0)
        nn.init.constant_(self.beta_head.bias, 0.0) # Initialize bias to output 0

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single timestep.

        Args:
            x (torch.Tensor): Features of the current frame, shape [B, D, H, W].
            prev_state (Optional[torch.Tensor]): Hidden state from the previous timestep,
                                                 shape [B, hidden_dim, H, W].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x_stable (torch.Tensor): The stabilized features, [B, D, H, W].
                - next_state (torch.Tensor): The new hidden state, [B, hidden_dim, H, W].
        """
        # 1. Update the hidden state
        # The input is the current feature map x and the previous hidden state
        next_state = self.conv_gru_cell(x, prev_state)

        # 2. Predict gamma and beta from the new hidden state
        # Note: gamma and beta are now tensors of shape [B, D, H, W]
        # gamma = self.gamma_head(next_state)
        # gamma = F.softplus(gamma)
        # gamma = torch.mean(gamma, dim=[2, 3], keepdim=True)  # This was the old method

        # beta = self.beta_head(next_state)
        # beta = torch.mean(beta, dim=[2, 3], keepdim=True)  # This was the old method

        # -- LEARNABLE way --
        # Instead of averaging, use an adaptive pooling to [B, D, 1, 1] (learnable heads already produce global info)
        gamma = F.softplus(self.gamma_head(next_state))
        gamma = F.adaptive_avg_pool2d(gamma, (1, 1))  # Output: [B, D, 1, 1]

        beta = self.beta_head(next_state)
        beta = F.adaptive_avg_pool2d(beta, (1, 1))    # Output: [B, D, 1, 1]

        # 3. Apply AdaIN (Instance Normalization + Affine Transform)
        # Calculate statistics of the current input x
        mu_x = torch.mean(x, dim=[2, 3], keepdim=True)
        std_x = torch.std(x, dim=[2, 3], keepdim=True)

        # Normalize the input
        x_norm = (x - mu_x) / (std_x + self.epsilon)

        # Apply the learned, spatially-varying gamma and beta
        x_stable = gamma * x_norm + beta

        return x_stable, next_state