#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Real TimesFM encoder using the official Google TimesFM library.
"""

import torch
import torch.nn as nn
from typing import Optional
import numpy as np
from timesfm import TimesFM_2p5_200M_torch, ForecastConfig

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.EnhancedTimeSeriesEncoderBase import (
    EnhancedTimeSeriesEncoderBase,
    PaddingStrategy,
    PatchingStrategy
)


class TimesFMEncoderReal(EnhancedTimeSeriesEncoderBase):
    """
    Real TimesFM encoder using Google's official implementation.

    TimesFM is Google's foundation model for time series forecasting,
    using a decoder-only architecture with patching and attention mechanisms.

    Args:
        model_name: Model checkpoint name (default uses 200M model)
        output_dim: Output dimension (default: ENCODER_OUTPUT_DIM=128)
        dropout: Dropout probability (default: 0.0)
        freeze_backbone: Whether to freeze the TimesFM backbone (default: False)
        context_length: Context length for the model (default: 512)
        device: Device to use
    """

    def __init__(
        self,
        model_name: str = "google/timesfm-2p5-200m-torch",
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
        context_length: int = 512,
        device: Optional[str] = None,
    ):
        super().__init__(
            output_dim=output_dim,
            dropout=dropout,
            padding_strategy=PaddingStrategy.RIGHT,
            patching_strategy=PatchingStrategy.NON_OVERLAPPING,
            patch_size=32,  # TimesFM uses patches of 32
            patch_stride=32,
        )

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.context_length = context_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize TimesFM model
        print(f"Loading TimesFM model...")
        self.model = TimesFM_2p5_200M_torch(
            device=self.device,
            checkpoint_path=model_name if "checkpoint" in model_name else None
        )

        # Move the model to the device
        self.model.model = self.model.model.to(self.device)

        # Create forecast config for compilation
        forecast_config = ForecastConfig(
            max_context=context_length,
            max_horizon=128,  # Default horizon for embeddings
            normalize_inputs=True,
            per_core_batch_size=1,
            force_flip_invariance=True,
            infer_is_positive=False
        )

        # Compile the model
        print("Compiling TimesFM model...")
        self.model.compile(forecast_config)

        # TimesFM 2.5 has 200M parameters and hidden dimension of 1280
        encoder_hidden_dim = 1280

        # Freeze backbone if requested
        if freeze_backbone:
            # TimesFM model parameters are not directly exposed in the same way
            # This is a placeholder - actual freezing would need model API support
            print("Note: Freezing not fully supported for TimesFM yet")

        # Projection layer to match output dimension
        self.projection = nn.Sequential(
            nn.Linear(encoder_hidden_dim, output_dim),
            nn.ReLU(),
            self.dropout_layer
        )

        self.projection = self.projection.to(self.device)

        print(f"TimesFMEncoderReal initialized: {encoder_hidden_dim} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TimesFM encoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)

        Returns:
            Encoded representation of shape (batch_size, output_dim)
        """
        batch_size, seq_len, n_features = x.shape

        # TimesFM expects univariate time series
        # If multivariate, process first feature only for now
        if n_features > 1:
            x = x[:, :, 0:1]

        # Pad or truncate sequence if needed
        if seq_len < self.context_length:
            x = self.pad_sequence(x, self.context_length)
        elif seq_len > self.context_length:
            x = x[:, :self.context_length]

        # TimesFM expects shape (batch, length) for univariate
        x = x.squeeze(-1)

        # Access the underlying model to get encoder representations
        # TimesFM model has a transformer-based architecture internally
        # We'll extract embeddings from the forward pass

        # Process all batch items at once
        # TimesFM expects shape (batch, length)
        # Ensure inputs are properly padded to be divisible by patch size
        p = self.model.model.p  # patch size = 32

        # Pad to be divisible by patch size
        current_len = x.shape[1]  # sequence length is dim 1
        pad_len = (p - (current_len % p)) % p
        if pad_len > 0:
            padding = torch.zeros(batch_size, pad_len, device=self.device)
            x = torch.cat([padding, x], dim=1)

        # Create masks (True for padded, False for real data)
        masks = torch.zeros(batch_size, x.shape[1], device=self.device, dtype=torch.bool)
        if pad_len > 0:
            masks[:, :pad_len] = True

        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            # Reshape inputs to patches
            patched_inputs = x.reshape(batch_size, -1, p)
            patched_masks = masks.reshape(batch_size, -1, p)

            # Call the model's forward method to get embeddings
            # The forward returns: (input_embeddings, output_embeddings, output_ts, output_quantile_spread), decode_caches
            outputs, _ = self.model.model(patched_inputs, patched_masks)

            # Extract the output embeddings (second element of tuple)
            input_embeddings, output_embeddings, _, _ = outputs

            # Use the output embeddings which contain the transformer's learned representations
            # Pool over the patch dimension to get fixed-size representation
            if len(output_embeddings.shape) == 3:  # (batch, patches, hidden)
                embeddings = output_embeddings.mean(dim=1)
            else:
                embeddings = output_embeddings

        # Ensure we have the right dimension
        if embeddings.shape[-1] != 1280:
            # Use adaptive layer to match dimensions
            adapter = nn.Linear(embeddings.shape[-1], 1280, device=self.device)
            embeddings = adapter(embeddings)

        # Project to output dimension
        output = self.projection(embeddings)

        return output

    def get_model_name(self) -> str:
        return f"TimesFMReal-200M"