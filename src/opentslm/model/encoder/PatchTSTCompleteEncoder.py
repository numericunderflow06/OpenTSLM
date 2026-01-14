#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Complete working PatchTST encoder with proper input handling.
Uses REAL pretrained models with correct patch and channel handling.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import PatchTSTModel
from transformers.models.patchtst.modeling_patchtst import PatchTSTPatchify

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


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
        output = past_values[:, :, self.sequence_start:]  # Correct for (batch, channels, seq)

        # output: [bs x num_channels x num_patches x patch_length]
        output = output.unfold(dimension=-1, size=self.patch_length, step=self.patch_stride)

        return output


class PatchTSTCompleteEncoder(TimeSeriesEncoderBase):
    """
    Complete working PatchTST encoder with all issues fixed.

    This properly handles the channel and patch dimensions to work
    with pretrained PatchTST models.

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

        # CRITICAL: Replace the buggy patchifier with our fixed version
        print("Applying dimension bug fix to patchifier...")
        self.model.patchifier = PatchTSTPatchifyFixed(self.model.config)

        print(f"✓ Loaded pretrained PatchTST model with fixes")

        # Get model configuration
        self.num_input_channels = self.model.config.num_input_channels
        self.context_length = self.model.config.context_length
        self.patch_length = self.model.config.patch_length
        # Use non-overlapping patches
        self.patch_stride = self.patch_length

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
        ).to(self.device)

        print(f"PatchTSTCompleteEncoder initialized: {model_output_dim} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PatchTST encoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)

        Returns:
            Encoded representation of shape (batch_size, output_dim)
        """
        batch_size, seq_len, n_features = x.shape

        # Adjust sequence length to match expected
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

        # Adjust number of channels to match expected
        if n_features != self.num_input_channels:
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

        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            # Use the model's forward method directly
            # This will handle scaling and patchification internally
            outputs = self.model(
                past_values=x,
                return_dict=True
            )

            # Extract representation
            if hasattr(outputs, 'prediction_outputs'):
                # For forecasting models
                hidden_states = outputs.prediction_outputs
                # Pool over output dimension if needed
                if len(hidden_states.shape) == 3:
                    pooled = hidden_states.mean(dim=1)
                else:
                    pooled = hidden_states
            elif hasattr(outputs, 'last_hidden_state'):
                # For representation models
                hidden_states = outputs.last_hidden_state
                # Pool over sequence/patch dimension
                pooled = hidden_states.mean(dim=1)
            elif hasattr(outputs, 'encoder_last_hidden_state'):
                hidden_states = outputs.encoder_last_hidden_state
                pooled = hidden_states.mean(dim=1)
            else:
                # Fallback to first output
                hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
                if len(hidden_states.shape) == 3:
                    pooled = hidden_states.mean(dim=1)
                else:
                    pooled = hidden_states

        # Ensure we have 2D tensor
        if len(pooled.shape) > 2:
            pooled = pooled.flatten(start_dim=1)

        # Handle dimension mismatch
        if pooled.shape[-1] != self.model.config.d_model:
            # Use adaptive pooling to match expected dimension
            if pooled.shape[-1] > self.model.config.d_model:
                # Pool down
                pooled = pooled[:, :self.model.config.d_model]
            else:
                # Pad with zeros
                padding = torch.zeros(
                    pooled.shape[0], self.model.config.d_model - pooled.shape[-1],
                    device=pooled.device, dtype=pooled.dtype
                )
                pooled = torch.cat([pooled, padding], dim=-1)

        # Project to output dimension
        output = self.projection(pooled)

        return output

    def get_model_name(self) -> str:
        return f"PatchTSTComplete-{self.model_name.split('/')[-1]}"