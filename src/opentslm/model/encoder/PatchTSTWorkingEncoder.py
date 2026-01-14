#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Working PatchTST encoder with all bug fixes applied.
Uses REAL pretrained models with no fallbacks.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import PatchTSTModel

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class PatchTSTWorkingEncoder(TimeSeriesEncoderBase):
    """
    Working PatchTST encoder that bypasses the dimension checking bug.

    This properly loads pretrained PatchTST models and works around
    the patchifier bug by manually creating patches.

    Args:
        model_name: HuggingFace model name for PatchTST
        output_dim: Output dimension (default: ENCODER_OUTPUT_DIM=128)
        dropout: Dropout probability (default: 0.1)
        freeze_backbone: Whether to freeze the PatchTST backbone (default: False)
        device: Device to load the model on
    """

    def __init__(
        self,
        model_name: str = "ibm/patchtst-etth1-pretrain",
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__(output_dim, dropout)

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained model
        print(f"Loading pretrained PatchTST model from {model_name}...")
        self.model = PatchTSTModel.from_pretrained(
            model_name,
            cache_dir="/local/home/wangni/.cache/huggingface"
        )
        print(f"✓ Loaded pretrained PatchTST model")

        # Get model configuration
        self.num_input_channels = self.model.config.num_input_channels
        self.context_length = self.model.config.context_length
        self.patch_length = self.model.config.patch_length
        self.patch_stride = self.model.config.patch_stride if hasattr(self.model.config, 'patch_stride') else self.patch_length

        print(f"  Expected: {self.num_input_channels} channels, {self.context_length} sequence length")
        print(f"  Patch length: {self.patch_length}, stride: {self.patch_stride}")

        # Move to device
        self.model = self.model.to(self.device)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print("  Backbone frozen")

        # Get output dimension from model
        model_output_dim = self.model.config.d_model

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(model_output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.projection = self.projection.to(self.device)

        print(f"PatchTSTWorkingEncoder initialized: {model_output_dim} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PatchTST encoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)

        Returns:
            Encoded representation of shape (batch_size, output_dim)
        """
        batch_size, seq_len, n_features = x.shape

        # Ensure correct sequence length
        if seq_len != self.context_length:
            if seq_len < self.context_length:
                # Pad with zeros
                padding = torch.zeros(
                    batch_size, self.context_length - seq_len, n_features,
                    device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, padding], dim=1)
            else:
                # Truncate
                x = x[:, :self.context_length, :]
            seq_len = self.context_length

        # Ensure correct number of channels
        if n_features != self.num_input_channels:
            if n_features < self.num_input_channels:
                # Pad with zeros
                padding = torch.zeros(
                    batch_size, seq_len, self.num_input_channels - n_features,
                    device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, padding], dim=-1)
            else:
                # Truncate
                x = x[:, :, :self.num_input_channels]

        # Transpose to (batch, channels, seq_len) as expected by PatchTST
        x = x.transpose(1, 2)

        # Bypass the buggy patchifier and work directly with the model components
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            # Scale the input
            if hasattr(self.model, 'scaler'):
                observed_indicator = torch.ones_like(x, dtype=torch.bool)
                x_scaled, loc, scale = self.model.scaler(x, observed_indicator)
            else:
                x_scaled = x

            # Manually create patches to bypass the buggy patchifier
            # The bug checks wrong dimension, so we'll unfold manually
            batch_size, n_channels, sequence_length = x_scaled.shape

            # Calculate number of patches
            num_patches = (sequence_length - self.patch_length) // self.patch_stride + 1

            # Create patches using unfold
            patches = x_scaled.unfold(
                dimension=2,  # Unfold along sequence dimension
                size=self.patch_length,
                step=self.patch_stride
            )  # Shape: (batch, channels, num_patches, patch_length)

            # Reshape patches for encoder input
            # The encoder expects: (batch_size, num_patches, patch_input_size)
            # where patch_input_size = num_channels * patch_length
            patches = patches.permute(0, 2, 1, 3)  # (batch, num_patches, channels, patch_length)
            patches_flat = patches.reshape(batch_size, num_patches, -1)  # (batch, num_patches, channels*patch_length)

            # Pass through encoder directly
            encoder_output = self.model.encoder(patch_input=patches_flat)

            # Extract hidden states
            if hasattr(encoder_output, 'last_hidden_state'):
                hidden_states = encoder_output.last_hidden_state
            else:
                hidden_states = encoder_output

        # Pool over patch dimension
        pooled = hidden_states.mean(dim=1)  # (batch, d_model)

        # Project to output dimension
        output = self.projection(pooled)

        return output

    def get_model_name(self) -> str:
        return f"PatchTSTWorking-{self.model_name.split('/')[-1]}"