#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn
from typing import Optional

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.EnhancedTimeSeriesEncoderBase import (
    EnhancedTimeSeriesEncoderBase,
    PaddingStrategy,
    PatchingStrategy
)

try:
    from momentfm import MOMENTPipeline
    MOMENT_AVAILABLE = True
except ImportError:
    MOMENT_AVAILABLE = False
    MOMENTPipeline = None


class MomentEncoder(EnhancedTimeSeriesEncoderBase):
    """
    MOMENT encoder wrapper for OpenTSLM.

    MOMENT is a family of foundation models for time series analysis from CMU.
    It uses a T5-based architecture with masking and reconstruction objectives.

    Paper: "MOMENT: A Family of Open Time-series Foundation Models"

    Args:
        model_name: HuggingFace model name for MOMENT
        output_dim: Output dimension (default: ENCODER_OUTPUT_DIM=128)
        dropout: Dropout probability (default: 0.1)
        freeze_backbone: Whether to freeze the MOMENT backbone (default: False)
        model_size: Size of the model ('small', 'base', or 'large')
        context_length: Context length for the model (default: 512)
    """

    def __init__(
        self,
        model_name: str = "AutonLab/MOMENT-1-large",
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        model_size: str = "large",
        context_length: int = 512,
        device: Optional[str] = None,
    ):
        super().__init__(
            output_dim=output_dim,
            dropout=dropout,
            padding_strategy=PaddingStrategy.CENTER,  # MOMENT uses center padding
            patching_strategy=PatchingStrategy.NON_OVERLAPPING,
            patch_size=8,  # MOMENT uses patches of 8
            patch_stride=8,
        )

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.model_size = model_size
        self.context_length = context_length
        self.device_type = device

        # Try to initialize MOMENT model
        if MOMENT_AVAILABLE:
            try:
                # Initialize MOMENT pipeline
                self.model = MOMENTPipeline.from_pretrained(
                    model_name,
                    model_kwargs={
                        "task_name": "embedding",  # We want embeddings, not forecasts
                        "n_channels": 1,  # Univariate for now
                    }
                )
                print(f"Loaded MOMENT model: {model_name}")
                self.use_fallback = False

                # Get model dimensions based on size
                if "large" in model_name.lower():
                    model_hidden_size = 1024
                elif "base" in model_name.lower():
                    model_hidden_size = 768
                else:  # small
                    model_hidden_size = 512
            except Exception as e:
                print(f"Warning: Could not load MOMENT model: {e}")
                self.use_fallback = True
        else:
            print("MOMENT not available, using fallback encoder")
            self.use_fallback = True

        if self.use_fallback:
            # Create fallback T5-style encoder
            model_hidden_size = 768
            self.fallback_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=model_hidden_size,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                ),
                num_layers=12 if model_size == "large" else 6,
            )

            # Input projection for patches
            self.patch_embedding = nn.Linear(self.patch_size, model_hidden_size)

            # Positional encoding
            self.register_parameter('pos_encoding', nn.Parameter(
                torch.randn(1, context_length // self.patch_size, model_hidden_size)
            ))

        # Projection layer to match output dimension
        self.projection = nn.Sequential(
            nn.Linear(model_hidden_size, output_dim),
            nn.ReLU(),
            self.dropout_layer
        )

        # Move entire module to device if specified
        if device:
            self.to(device)

        # Freeze backbone if requested
        if freeze_backbone:
            if not self.use_fallback and hasattr(self, 'model'):
                for param in self.model.model.parameters():
                    param.requires_grad = False
            elif self.use_fallback:
                for param in self.fallback_encoder.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MOMENT encoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)

        Returns:
            Encoded representation of shape (batch_size, output_dim)
        """
        batch_size, seq_len, n_features = x.shape

        # MOMENT expects univariate time series for now
        if n_features > 1:
            x = x[:, :, 0:1]

        # Pad sequence to context length using center padding
        if seq_len < self.context_length:
            x = self.pad_sequence(x, self.context_length)
        elif seq_len > self.context_length:
            # For center padding, take the middle portion
            start = (seq_len - self.context_length) // 2
            x = x[:, start:start + self.context_length]

        if self.use_fallback:
            # Create patches
            patches = self.create_patches(x)  # (batch, n_patches, patch_size, 1)
            batch_size, n_patches, patch_size, _ = patches.shape

            # Flatten patches and embed
            patches_flat = patches.squeeze(-1)  # (batch, n_patches, patch_size)
            patch_embeddings = self.patch_embedding(patches_flat)  # (batch, n_patches, hidden)

            # Add positional encoding
            patch_embeddings = patch_embeddings + self.pos_encoding[:, :n_patches, :]

            # Encode with transformer
            encoded = self.fallback_encoder(patch_embeddings)  # (batch, n_patches, hidden)

            # Pool over patches - use mean pooling
            pooled = encoded.mean(dim=1)  # (batch, hidden)
        else:
            # Use actual MOMENT model
            # MOMENT expects shape (batch, length, channels)
            x_moment = x.squeeze(-1) if x.shape[-1] == 1 else x

            # Get embeddings from MOMENT
            outputs = self.model(x_moment, task="embedding")

            # Extract embeddings
            if isinstance(outputs, dict) and "embeddings" in outputs:
                embeddings = outputs["embeddings"]
            else:
                embeddings = outputs

            # Ensure proper shape
            if len(embeddings.shape) == 3:
                # If (batch, seq, hidden), pool over sequence
                pooled = embeddings.mean(dim=1)
            else:
                pooled = embeddings

        # Project to output dimension
        output = self.projection(pooled)

        return output

    def get_model_name(self) -> str:
        return f"MOMENT-{self.model_size}"