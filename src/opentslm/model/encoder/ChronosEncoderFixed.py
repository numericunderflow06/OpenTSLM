#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Fixed Chronos encoder using ChronosPipeline to avoid version issues.
"""

import torch
import torch.nn as nn
from typing import Optional
from chronos import ChronosPipeline

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class ChronosEncoderFixed(TimeSeriesEncoderBase):
    """
    Fixed Chronos encoder using ChronosPipeline from chronos package.

    This version properly uses the ChronosPipeline to load pretrained models
    and extracts encoder embeddings for use in OpenTSLM.

    Args:
        model_name: HuggingFace model name for Chronos
        output_dim: Output dimension (default: ENCODER_OUTPUT_DIM=128)
        dropout: Dropout probability (default: 0.0)
        freeze_backbone: Whether to freeze the Chronos backbone (default: False)
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load Chronos model via pipeline - load on CPU first
        print(f"Loading Chronos model from {model_name}...")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map="cpu",  # Load on CPU first to avoid device issues
            dtype=torch.float32,  # Use dtype instead of torch_dtype
            cache_dir="/local/home/wangni/.cache/huggingface"
        )

        # Move model and tokenizer bins to target device
        self.pipeline.model = self.pipeline.model.to(self.device)
        if hasattr(self.pipeline.tokenizer, 'tokenizer_bins'):
            self.pipeline.tokenizer.tokenizer_bins = self.pipeline.tokenizer.tokenizer_bins.to(self.device)

        # Access the underlying model
        self.model = self.pipeline.model

        # Get encoder hidden dimension based on model size
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

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print("Chronos backbone frozen")

        # Create projection layer
        self.projection = nn.Linear(encoder_hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.output_dropout = nn.Dropout(dropout)

        # Move layers to device
        self.projection = self.projection.to(self.device)
        self.output_norm = self.output_norm.to(self.device)
        self.output_dropout = self.output_dropout.to(self.device)

        print(f"ChronosEncoderFixed initialized: {encoder_hidden_dim} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Chronos encoder.

        Args:
            x: FloatTensor of shape [B, L], a batch of raw time series.

        Returns:
            FloatTensor of shape [B, output_dim].
        """
        B, L = x.shape

        # Move to device
        device = next(self.model.parameters()).device
        x = x.to(device)

        # We need to extract encoder embeddings
        # Use a hook to capture encoder outputs
        encoder_outputs = []

        def hook_fn(module, input, output):
            # Capture the encoder output
            if hasattr(output, 'last_hidden_state'):
                encoder_outputs.append(output.last_hidden_state)
            elif isinstance(output, torch.Tensor):
                encoder_outputs.append(output)
            elif isinstance(output, tuple) and len(output) > 0:
                encoder_outputs.append(output[0])

        # Register hook on the encoder
        if hasattr(self.model, 'encoder'):
            handle = self.model.encoder.register_forward_hook(hook_fn)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'encoder'):
            handle = self.model.model.encoder.register_forward_hook(hook_fn)
        else:
            raise ValueError("Cannot find encoder in model structure")

        # Process through the model (we just need encoder outputs)
        with torch.no_grad():
            # Use the tokenizer to prepare input
            if hasattr(self.pipeline, 'tokenizer'):
                # Tokenize the input
                tokens = self.pipeline.tokenizer.encode(x)
            else:
                # Manual tokenization (simple binning)
                min_val, max_val = -15.0, 15.0
                n_bins = 4094  # Reserve 2 for special tokens

                # Normalize
                x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)

                # Clip and bin
                x_clipped = torch.clamp(x_norm, min_val, max_val)
                tokens = ((x_clipped - min_val) / (max_val - min_val) * n_bins).long() + 1

            # Create attention mask
            attention_mask = torch.ones_like(tokens)

            # Forward through model encoder only
            if hasattr(self.model, 'encoder'):
                _ = self.model.encoder(input_ids=tokens, attention_mask=attention_mask)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'encoder'):
                _ = self.model.model.encoder(input_ids=tokens, attention_mask=attention_mask)

        # Remove hook
        handle.remove()

        # Get encoder outputs
        if encoder_outputs:
            hidden_states = encoder_outputs[-1]  # Get the last captured output

            # Pool over sequence dimension (mean pooling)
            pooled = hidden_states.mean(dim=1)  # [B, hidden_dim]

            # Project to output dimension
            output = self.projection(pooled)
            output = self.output_norm(output)
            output = self.output_dropout(output)

            return output
        else:
            raise ValueError("Failed to capture encoder outputs")

    def get_model_name(self) -> str:
        return f"ChronosFixed-{self.model_name.split('/')[-1]}"