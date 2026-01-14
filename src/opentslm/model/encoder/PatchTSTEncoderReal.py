#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Real PatchTST encoder that fixes the dimension bug in transformers library.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import PatchTSTModel
from transformers.models.patchtst.modeling_patchtst import PatchTSTPatchify

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.EnhancedTimeSeriesEncoderBase import (
    EnhancedTimeSeriesEncoderBase,
    PaddingStrategy,
    PatchingStrategy
)


class PatchTSTPatchifyFixed(PatchTSTPatchify):
    """Fixed version of PatchTSTPatchify that correctly checks sequence dimension."""

    def forward(self, past_values: torch.Tensor):
        """
        Fixed forward that correctly checks sequence length on the last dimension.

        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length)`, *required*):
                Input for patchification

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        # FIX: Check the correct dimension for sequence length
        # Input is (batch, channels, sequence), so sequence is dim -1
        sequence_length = past_values.shape[-1]  # FIXED: was shape[-2]

        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."
            )

        # output: [bs x num_channels x new_sequence_length]
        output = past_values[:, :, self.sequence_start:]  # FIXED: indexing for (batch, channels, seq)

        # output: [bs x num_channels x num_patches x patch_length]
        output = output.unfold(dimension=-1, size=self.patch_length, step=self.patch_stride)  # FIXED: dimension=-1

        return output


class PatchTSTEncoderReal(EnhancedTimeSeriesEncoderBase):
    """
    Real PatchTST encoder with bug fixes for the transformers library issues.

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
            patch_size=12,
            patch_stride=12,
        )

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained model
        print(f"Loading pretrained PatchTST model from {model_name}...")
        self.model = PatchTSTModel.from_pretrained(model_name)

        # Replace the buggy patchifier with our fixed version
        print("Applying dimension bug fix to patchifier...")
        self.model.patchifier = PatchTSTPatchifyFixed(self.model.config)

        print(f"✓ Loaded pretrained PatchTST model")

        # Get model configuration
        self.num_input_channels = self.model.config.num_input_channels
        self.context_length = self.model.config.context_length
        self.patch_length = self.model.config.patch_length

        print(f"  Expected: {self.num_input_channels} channels, {self.context_length} sequence length")

        if self.device:
            self.model = self.model.to(self.device)

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

        if self.device:
            self.projection = self.projection.to(self.device)

        print(f"PatchTSTEncoderReal initialized: {model_output_dim} -> {output_dim}")

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

        # Forward through PatchTST model
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            outputs = self.model(
                past_values=x,
                return_dict=True
            )

        # Extract representation
        if hasattr(outputs, 'prediction_outputs'):
            # Use prediction outputs if available
            hidden_states = outputs.prediction_outputs
        elif hasattr(outputs, 'last_hidden_state'):
            # Use last hidden state
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'encoder_last_hidden_state'):
            hidden_states = outputs.encoder_last_hidden_state
        else:
            # Try to get any hidden states
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        # Pool over sequence/patch dimension if needed
        if len(hidden_states.shape) == 3:
            # Shape is (batch, seq/patches, features)
            pooled = hidden_states.mean(dim=1)
        elif len(hidden_states.shape) == 4:
            # Shape might be (batch, channels, patches, features)
            pooled = hidden_states.mean(dim=(1, 2))
        else:
            pooled = hidden_states

        # Ensure we have 2D tensor
        if len(pooled.shape) > 2:
            pooled = pooled.flatten(start_dim=1)

        # If dimension doesn't match, use adaptive pooling
        if pooled.shape[-1] != self.model.config.d_model:
            # Create a simple linear layer to match dimensions
            adapter = nn.Linear(pooled.shape[-1], self.model.config.d_model).to(pooled.device)
            pooled = adapter(pooled)

        # Project to output dimension
        output = self.projection(pooled)

        return output

    def get_model_name(self) -> str:
        return f"PatchTSTReal-{self.model_name.split('/')[-1]}"