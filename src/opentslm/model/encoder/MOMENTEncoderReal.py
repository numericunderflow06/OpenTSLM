#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Real MOMENT encoder using the official MOMENT package.
"""

import torch
import torch.nn as nn
from typing import Optional
import sys
sys.path.append('/tmp/moment')
from momentfm import MOMENTPipeline

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.EnhancedTimeSeriesEncoderBase import (
    EnhancedTimeSeriesEncoderBase,
    PaddingStrategy,
    PatchingStrategy
)


class MOMENTEncoderReal(EnhancedTimeSeriesEncoderBase):
    """
    Real MOMENT encoder using the official MOMENT package.

    MOMENT is a family of time series foundation models that use T5-based
    architectures for time series representation learning.

    Args:
        model_name: HuggingFace model name for MOMENT
        output_dim: Output dimension (default: ENCODER_OUTPUT_DIM=128)
        dropout: Dropout probability (default: 0.0)
        freeze_backbone: Whether to freeze the MOMENT backbone (default: False)
        device: Device to load the model on
    """

    def __init__(
        self,
        model_name: str = "AutonLab/MOMENT-1-large",
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__(
            output_dim=output_dim,
            dropout=dropout,
            padding_strategy=PaddingStrategy.RIGHT,
            patching_strategy=PatchingStrategy.NON_OVERLAPPING,
            patch_size=8,  # MOMENT default patch size
            patch_stride=8,
        )

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained model using MOMENTPipeline
        print(f"Loading pretrained MOMENT model from {model_name}...")
        self.model = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={
                'task_name': 'embedding',  # Use embedding task for getting embeddings
            }
        )

        # Initialize the model with the new task
        self.model.init()

        print(f"✓ Loaded pretrained MOMENT model")

        # Move model to device
        self.model = self.model.to(self.device)

        # Get model configuration
        # MOMENT models have different hidden dimensions based on size
        if "small" in model_name.lower():
            encoder_hidden_dim = 512
        elif "base" in model_name.lower():
            encoder_hidden_dim = 768
        elif "large" in model_name.lower():
            encoder_hidden_dim = 1024
        else:
            # Default to base size
            encoder_hidden_dim = 768

        print(f"  Hidden dimension: {encoder_hidden_dim}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print("  Backbone frozen")

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(encoder_hidden_dim, output_dim),
            nn.ReLU(),
            self.dropout_layer
        )

        self.projection = self.projection.to(self.device)

        print(f"MOMENTEncoderReal initialized: {encoder_hidden_dim} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MOMENT encoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)

        Returns:
            Encoded representation of shape (batch_size, output_dim)
        """
        batch_size, seq_len, n_features = x.shape

        # MOMENT expects shape (batch, channels, length)
        # Transpose from (batch, seq_len, n_features) to (batch, n_features, seq_len)
        x = x.transpose(1, 2)

        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            # MOMENT model forward pass with embed task
            # The model expects x_enc parameter
            outputs = self.model(x_enc=x)

            # The embed task returns TimeseriesOutputs with embeddings
            if hasattr(outputs, 'embeddings'):
                embeddings = outputs.embeddings
            elif hasattr(outputs, 'output_embeds'):
                embeddings = outputs.output_embeds
            elif hasattr(outputs, 'representations'):
                embeddings = outputs.representations
            else:
                # Try to get any tensor from outputs
                if isinstance(outputs, torch.Tensor):
                    embeddings = outputs
                else:
                    embeddings = None

            # The embeddings should already be pooled by the embed task
            pooled = embeddings

            if pooled is None:
                raise ValueError("Could not extract embeddings from MOMENT model output")

        # Ensure we have 2D tensor
        if len(pooled.shape) > 2:
            pooled = pooled.flatten(start_dim=1)

        # Project to output dimension
        output = self.projection(pooled)

        return output

    def get_model_name(self) -> str:
        return f"MOMENTReal-{self.model_name.split('/')[-1]}"