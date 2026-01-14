#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Simplified Chronos2 encoder that directly uses T5 model from transformers.
This version avoids version compatibility issues with chronos-forecasting package.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import T5ForConditionalGeneration, T5EncoderModel

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class Chronos2EncoderSimple(TimeSeriesEncoderBase):
    """
    Simplified Chronos-2 encoder using T5 directly from transformers.

    This encoder uses the pre-trained Chronos models (which are T5-based)
    directly through transformers library, avoiding chronos package version issues.

    Args:
        model_name: HuggingFace model name for Chronos (default: "amazon/chronos-t5-tiny")
        output_dim: Output dimension (default: ENCODER_OUTPUT_DIM=128)
        dropout: Dropout probability (default: 0.0)
        freeze_backbone: Whether to freeze the T5 backbone (default: False)
        device: Device to load the model on (default: None, uses input device)
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-tiny",
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__(output_dim, dropout)

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone

        # Load T5 encoder directly
        print(f"Loading Chronos model as T5 from {model_name}...")

        # Load the full model first to get the encoder
        full_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.encoder = full_model.encoder

        # Delete the full model to save memory
        del full_model

        if device is not None:
            self.encoder = self.encoder.to(device)

        # Get encoder hidden dimension
        encoder_hidden_dim = self.encoder.config.d_model

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("T5 encoder backbone frozen")

        # Create projection layer
        self.projection = nn.Linear(encoder_hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.output_dropout = nn.Dropout(dropout)

        if device is not None:
            self.projection = self.projection.to(device)
            self.output_norm = self.output_norm.to(device)
            self.output_dropout = self.output_dropout.to(device)

        print(f"Chronos2EncoderSimple initialized: {encoder_hidden_dim} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Chronos-2 encoder.

        Args:
            x: FloatTensor of shape [B, L], a batch of raw time series.
               Where B is batch size and L is sequence length.

        Returns:
            FloatTensor of shape [B, output_dim].
        """
        B, L = x.shape

        # Move to same device as model
        device = next(self.encoder.parameters()).device
        x = x.to(device)

        # Normalize input (simple standardization)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)

        # Quantize to tokens (simple linear binning)
        # Chronos uses 4096 tokens by default
        n_tokens = 4096
        min_val, max_val = -15.0, 15.0

        # Clip values to range
        x = torch.clamp(x, min_val, max_val)

        # Convert to token indices
        tokens = ((x - min_val) / (max_val - min_val) * (n_tokens - 1)).long()

        # Add special tokens if needed
        tokens = tokens + 2  # Reserve 0 for padding, 1 for EOS

        # Create attention mask (all ones since we have no padding)
        attention_mask = torch.ones_like(tokens)

        # Forward through T5 encoder
        encoder_outputs = self.encoder(
            input_ids=tokens,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Get hidden states
        hidden_states = encoder_outputs.last_hidden_state  # [B, L, hidden_dim]

        # Pool over sequence dimension (mean pooling)
        pooled = hidden_states.mean(dim=1)  # [B, hidden_dim]

        # Project to output dimension
        output = self.projection(pooled)
        output = self.output_norm(output)
        output = self.output_dropout(output)

        return output

    def get_model_name(self) -> str:
        return f"Chronos2Simple-{self.model_name.split('/')[-1]}"