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
    from transformers import PatchTSTModel, PatchTSTConfig
    PATCHTST_AVAILABLE = True
except ImportError:
    PATCHTST_AVAILABLE = False
    PatchTSTModel = None
    PatchTSTConfig = None


class PatchTSTEncoder(EnhancedTimeSeriesEncoderBase):
    """
    PatchTST encoder wrapper for OpenTSLM.

    PatchTST (Patch Time Series Transformer) is a channel-independent
    transformer model that uses patching to efficiently process long sequences.

    Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"

    Args:
        model_name: HuggingFace model name for PatchTST
        output_dim: Output dimension (default: ENCODER_OUTPUT_DIM=128)
        dropout: Dropout probability (default: 0.1)
        freeze_backbone: Whether to freeze the PatchTST backbone (default: False)
        patch_size: Size of each patch (default: 16)
        patch_stride: Stride for patching (default: 8)
        num_input_channels: Number of input channels/features (default: 1)
        context_length: Context length for the model (default: 512)
    """

    def __init__(
        self,
        model_name: str = "ibm/patchtst-etth1-pretrain",
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        patch_size: int = 16,
        patch_stride: int = 8,
        num_input_channels: int = 1,
        context_length: int = 512,
        device: Optional[str] = None,
    ):
        super().__init__(
            output_dim=output_dim,
            dropout=dropout,
            padding_strategy=PaddingStrategy.RIGHT,
            patching_strategy=PatchingStrategy.NON_OVERLAPPING if patch_stride == patch_size else PatchingStrategy.OVERLAPPING,
            patch_size=patch_size,
            patch_stride=patch_stride,
        )

        if not PATCHTST_AVAILABLE:
            raise ImportError(
                "PatchTST is not available. Please install transformers>=4.35.0"
            )

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.num_input_channels = num_input_channels
        self.context_length = context_length
        self.device_type = device

        # Initialize PatchTST model - always try to load from pretrained first
        try:
            # Load pretrained model from HuggingFace
            print(f"Loading pretrained PatchTST model from {model_name}...")
            self.model = PatchTSTModel.from_pretrained(model_name)
            print(f"✓ Loaded pretrained PatchTST model: {model_name}")

            # Update config parameters if needed
            self.model.config.num_input_channels = num_input_channels
            self.model.config.context_length = context_length
        except Exception as e:
            print(f"Warning: Could not load pretrained model {model_name}: {e}")
            print("Falling back to creating model from config (not pretrained)")

            # Fall back to creating from config (not pretrained)
            config = PatchTSTConfig(
                num_input_channels=num_input_channels,
                context_length=context_length,
                patch_length=patch_size,
                patch_stride=patch_stride,
                d_model=128,
                num_attention_heads=8,
                num_hidden_layers=3,
                ffn_dim=256,
                dropout=dropout,
                attention_dropout=dropout,
                positional_encoding_type="sincos",
                use_cls_token=True,
                scaling="std",
            )
            self.model = PatchTSTModel(config)
            print(f"Created PatchTST model from config (not using pretrained weights)")

        # Move to device if specified
        if device:
            self.model = self.model.to(device)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Get the output dimension from the model
        model_output_dim = self.model.config.d_model

        # Projection layer to match output dimension
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

        # Pad sequence to context length FIRST (before channel adjustment)
        if seq_len < self.context_length:
            x = self.pad_sequence(x, self.context_length)
            seq_len = self.context_length
        elif seq_len > self.context_length:
            x = x[:, :self.context_length]
            seq_len = self.context_length

        # Then ensure we have the right number of channels
        if n_features != self.num_input_channels:
            # Reshape or pad channels as needed
            if n_features < self.num_input_channels:
                # Pad with zeros to match expected channels
                padding = torch.zeros(
                    batch_size, seq_len, self.num_input_channels - n_features,
                    device=x.device, dtype=x.dtype
                )
                x = torch.cat([x, padding], dim=-1)
            else:
                # Truncate extra channels
                x = x[:, :, :self.num_input_channels]

        # PatchTST expects input shape: (batch_size, n_features, seq_len)
        # Transpose from (batch, seq_len, n_features) to (batch, n_features, seq_len)
        x_transposed = x.transpose(1, 2)

        # Forward through PatchTST
        outputs = self.model(
            past_values=x_transposed,  # PatchTST expects (batch, channels, seq_len)
            return_dict=True
        )

        # Extract representation
        if hasattr(outputs, 'prediction_outputs'):
            # Use prediction outputs if available
            hidden_states = outputs.prediction_outputs
        elif hasattr(outputs, 'last_hidden_state'):
            # Use last hidden state
            hidden_states = outputs.last_hidden_state
        else:
            # Fall back to raw outputs
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        # Pool over sequence dimension
        if len(hidden_states.shape) == 3:
            # If shape is (batch, seq, features), pool over sequence
            if hidden_states.shape[1] > 1:
                # Use mean pooling
                pooled = hidden_states.mean(dim=1)
            else:
                pooled = hidden_states.squeeze(1)
        else:
            pooled = hidden_states

        # Project to output dimension
        output = self.projection(pooled)

        return output

    def get_model_name(self) -> str:
        return f"PatchTST-{self.model_name.split('/')[-1]}"