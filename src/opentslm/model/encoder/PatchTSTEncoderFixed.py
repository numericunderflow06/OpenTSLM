#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Fixed PatchTST encoder that works around dimension checking bug in transformers.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import PatchTSTModel, PatchTSTConfig

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.EnhancedTimeSeriesEncoderBase import (
    EnhancedTimeSeriesEncoderBase,
    PaddingStrategy,
    PatchingStrategy
)


class PatchTSTEncoderFixed(EnhancedTimeSeriesEncoderBase):
    """
    Fixed PatchTST encoder that properly handles pretrained models.

    This version works around the dimension checking bug in transformers library
    by directly calling the encoder components.

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
        super().__init__(
            output_dim=output_dim,
            dropout=dropout,
            padding_strategy=PaddingStrategy.RIGHT,
            patching_strategy=PatchingStrategy.NON_OVERLAPPING,
            patch_size=12,  # Default for pretrained models
            patch_stride=12,
        )

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone

        # Load pretrained model
        print(f"Loading pretrained PatchTST model from {model_name}...")
        self.model = PatchTSTModel.from_pretrained(model_name)
        print(f"✓ Loaded pretrained PatchTST model")

        # Get model configuration
        self.num_input_channels = self.model.config.num_input_channels
        self.context_length = self.model.config.context_length
        self.patch_length = self.model.config.patch_length

        print(f"  Expected: {self.num_input_channels} channels, {self.context_length} sequence length")

        if device:
            self.model = self.model.to(device)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Get output dimension from model
        model_output_dim = self.model.config.d_model

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(model_output_dim, output_dim),
            nn.ReLU(),
            self.dropout_layer
        )

        if device:
            self.projection = self.projection.to(device)

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
            # Pad or truncate to match expected length
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

        # Ensure correct number of channels
        if n_features != self.num_input_channels:
            # Pad or truncate channels
            if n_features < self.num_input_channels:
                # Pad with zeros
                padding = torch.zeros(
                    batch_size, self.context_length, self.num_input_channels - n_features,
                    device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, padding], dim=-1)
            else:
                # Truncate
                x = x[:, :, :self.num_input_channels]

        # Transpose to (batch, channels, seq_len) as expected by PatchTST
        x = x.transpose(1, 2)

        # Workaround: Directly call the model's forward components
        # to avoid the dimension checking bug

        # Scale the input
        if hasattr(self.model, 'scaler'):
            # Create observed indicator (all ones - no missing values)
            observed_indicator = torch.ones_like(x, dtype=torch.bool)
            x_scaled, loc, scale = self.model.scaler(x, observed_indicator)
        else:
            x_scaled = x
            loc = None
            scale = None

        # Patchify - this is where the bug occurs in the original
        # We'll manually patchify to avoid the issue
        # PatchTST uses unfold to create patches
        batch_size, num_channels, sequence_length = x_scaled.shape

        # Create patches manually
        patches = x_scaled.unfold(
            dimension=-1,
            size=self.patch_length,
            step=self.patch_length
        )  # (batch, channels, num_patches, patch_length)

        # Reshape for processing
        num_patches = patches.shape[2]
        patches = patches.permute(0, 2, 1, 3)  # (batch, num_patches, channels, patch_length)
        patches = patches.reshape(batch_size * num_patches, num_channels, self.patch_length)

        # Pass patches through the model's encoder directly
        # Reshape patches for encoder
        patches_reshaped = patches.reshape(batch_size, num_patches, num_channels * self.patch_length)

        # Use the full encoder forward method
        encoder_output = self.model.encoder(patch_input=patches_reshaped)

        # Extract the last hidden state (already includes positional encoding and layer processing)
        if hasattr(encoder_output, 'last_hidden_state'):
            embedded = encoder_output.last_hidden_state
        else:
            embedded = encoder_output

        # Pool over patches
        pooled = embedded.mean(dim=1)  # (batch, d_model)

        # Project to output dimension
        output = self.projection(pooled)

        return output

    def get_model_name(self) -> str:
        return f"PatchTSTFixed-{self.model_name.split('/')[-1]}"