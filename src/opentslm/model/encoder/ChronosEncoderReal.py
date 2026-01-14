#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Real Chronos encoder using the chronos package properly.
"""

import torch
import torch.nn as nn
from typing import Optional
from chronos import ChronosPipeline, MeanScaleUniformBins

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class ChronosEncoderReal(TimeSeriesEncoderBase):
    """
    Real Chronos encoder using ChronosPipeline correctly.

    This properly loads and uses Chronos models from HuggingFace.

    Args:
        model_name: HuggingFace model name for Chronos
        output_dim: Output dimension (default: ENCODER_OUTPUT_DIM=128)
        dropout: Dropout probability (default: 0.0)
        freeze_backbone: Whether to freeze the Chronos backbone (default: False)
        device: Device to load the model on
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load Chronos pipeline
        print(f"Loading Chronos model from {model_name}...")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=self.device,
            dtype=torch.float32,  # Use dtype instead of torch_dtype
        )

        # Get the underlying model and tokenizer
        self.tokenizer = self.pipeline.tokenizer

        # Get the actual T5 model
        if hasattr(self.pipeline, 'inner_model'):
            self.model = self.pipeline.inner_model
        elif hasattr(self.pipeline.model, 'model'):
            self.model = self.pipeline.model.model
        else:
            self.model = self.pipeline.model

        # Get encoder dimension from T5 config
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'd_model'):
            encoder_hidden_dim = self.model.config.d_model
        else:
            # Fallback dimensions
            if "tiny" in model_name:
                encoder_hidden_dim = 256
            elif "mini" in model_name:
                encoder_hidden_dim = 512
            elif "small" in model_name:
                encoder_hidden_dim = 768
            elif "base" in model_name:
                encoder_hidden_dim = 768
            else:  # large
                encoder_hidden_dim = 1024

        print(f"Encoder hidden dimension: {encoder_hidden_dim}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print("Chronos backbone frozen")

        # Create projection layer
        self.projection = nn.Linear(encoder_hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.output_dropout = nn.Dropout(dropout)

        # Move to device
        self.projection = self.projection.to(self.device)
        self.output_norm = self.output_norm.to(self.device)
        self.output_dropout = self.output_dropout.to(self.device)

        print(f"ChronosEncoderReal initialized: {encoder_hidden_dim} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Chronos encoder.

        Args:
            x: FloatTensor of shape [B, L, F] where F is number of features.
               For Chronos, we use univariate (F=1).

        Returns:
            FloatTensor of shape [B, output_dim].
        """
        # Handle both 2D and 3D inputs
        if len(x.shape) == 3:
            B, L, F = x.shape
            # Use first feature for univariate model
            x = x[:, :, 0]
        else:
            B, L = x.shape

        # Tokenize using the tokenizer's context_input_transform
        # The tokenizer expects CPU tensors, so keep on CPU for tokenization
        x_cpu = x.cpu() if x.is_cuda else x
        tokens, attention_mask, scale = self.tokenizer.context_input_transform(x_cpu)

        # tokens shape: (batch, length)
        tokens = tokens.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Get encoder outputs
        # We need to access the encoder directly
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            # The model is a T5, so we can access its encoder
            encoder_outputs = self.model.encoder(
                input_ids=tokens,
                attention_mask=attention_mask,
                return_dict=True
            )

            # Get the last hidden states
            hidden_states = encoder_outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # Pool over sequence dimension
        # Use mean pooling over non-padded tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        # Project to output dimension
        output = self.projection(pooled)
        output = self.output_norm(output)
        output = self.output_dropout(output)

        return output

    def get_model_name(self) -> str:
        return f"ChronosReal-{self.model_name.split('/')[-1]}"