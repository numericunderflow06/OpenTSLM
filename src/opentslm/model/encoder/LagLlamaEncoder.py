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
    from lag_llama.gluon.estimator import LagLlamaEstimator
    from gluonts.torch.model.predictor import PyTorchPredictor
    LAG_LLAMA_AVAILABLE = True
except ImportError:
    LAG_LLAMA_AVAILABLE = False
    LagLlamaEstimator = None
    PyTorchPredictor = None

try:
    from transformers import LlamaModel, LlamaConfig
    from huggingface_hub import hf_hub_download
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    LlamaModel = None
    LlamaConfig = None


class LagLlamaEncoder(EnhancedTimeSeriesEncoderBase):
    """
    Lag-Llama encoder wrapper for OpenTSLM.

    Lag-Llama is a probabilistic time series forecasting model based on the LLaMA
    architecture, specifically designed for univariate time series.

    Paper: "Lag-Llama: Towards Foundation Models for Time Series Forecasting"

    Args:
        model_name: HuggingFace model name or path
        output_dim: Output dimension (default: ENCODER_OUTPUT_DIM=128)
        dropout: Dropout probability (default: 0.1)
        freeze_backbone: Whether to freeze the Lag-Llama backbone (default: False)
        context_length: Context length for the model (default: 32)
        use_rope: Whether to use RoPE (Rotary Position Embedding)
    """

    def __init__(
        self,
        model_name: str = "time-series-foundation-models/Lag-Llama",
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        context_length: int = 32,
        use_rope: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__(
            output_dim=output_dim,
            dropout=dropout,
            padding_strategy=PaddingStrategy.LEFT,  # Lag-Llama uses left padding like Chronos
            patching_strategy=PatchingStrategy.NONE,
            use_nan_padding=True,  # Use NaN for padding like Chronos
        )

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.context_length = context_length
        self.use_rope = use_rope
        self.device_type = device

        # Initialize Lag-Llama model
        if LAG_LLAMA_AVAILABLE:
            try:
                print(f"Loading pretrained Lag-Llama model from {model_name}...")

                # Download checkpoint from HuggingFace
                ckpt_path = hf_hub_download(
                    repo_id=model_name,
                    filename="lag-llama.ckpt",
                    cache_dir="./model_cache"
                )

                # Load the model
                estimator = LagLlamaEstimator.load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    map_location=device or "cpu"
                )

                self.model = estimator.create_predictor(
                    estimator.create_transformation(),
                    estimator.create_lightning_module()
                )

                print(f"✓ Loaded pretrained Lag-Llama model: {model_name}")
                self.use_fallback = False
                model_hidden_size = 256  # Lag-Llama hidden size

            except Exception as e:
                print(f"Warning: Could not load Lag-Llama model: {e}")
                print("Using fallback transformer encoder")
                self.use_fallback = True
        else:
            print("Lag-Llama not available, using fallback encoder")
            self.use_fallback = True

        if self.use_fallback:
            # Create a simplified LLaMA-style encoder as fallback
            model_hidden_size = 512

            # Try to load actual LLaMA architecture if transformers is available
            if TRANSFORMERS_AVAILABLE:
                try:
                    # Create a small LLaMA configuration
                    config = LlamaConfig(
                        hidden_size=model_hidden_size,
                        intermediate_size=2048,
                        num_hidden_layers=8,
                        num_attention_heads=8,
                        max_position_embeddings=context_length,
                        rope_theta=10000.0 if use_rope else None,
                    )
                    self.fallback_model = LlamaModel(config)
                    print("Using LLaMA architecture as fallback")
                except:
                    # Fall back to standard transformer
                    self._create_standard_transformer_fallback(model_hidden_size)
            else:
                self._create_standard_transformer_fallback(model_hidden_size)

            # Input projection for time series
            self.input_projection = nn.Linear(1, model_hidden_size)

        # Output projection layer
        self.projection = nn.Sequential(
            nn.Linear(model_hidden_size, output_dim),
            nn.ReLU(),
            self.dropout_layer
        )

        # Move to device if specified
        if device:
            if self.use_fallback:
                if hasattr(self, 'fallback_model'):
                    self.fallback_model = self.fallback_model.to(device)
                if hasattr(self, 'fallback_encoder'):
                    self.fallback_encoder = self.fallback_encoder.to(device)
                self.input_projection = self.input_projection.to(device)
            self.projection = self.projection.to(device)

        # Freeze backbone if requested
        if freeze_backbone:
            if not self.use_fallback:
                for param in self.model.parameters():
                    param.requires_grad = False
            else:
                if hasattr(self, 'fallback_model'):
                    for param in self.fallback_model.parameters():
                        param.requires_grad = False
                elif hasattr(self, 'fallback_encoder'):
                    for param in self.fallback_encoder.parameters():
                        param.requires_grad = False

    def _create_standard_transformer_fallback(self, hidden_size: int):
        """Create standard transformer as fallback."""
        self.fallback_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=2048,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=8,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Lag-Llama encoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)

        Returns:
            Encoded representation of shape (batch_size, output_dim)
        """
        batch_size, seq_len, n_features = x.shape

        # Lag-Llama is designed for univariate time series
        if n_features > 1:
            x = x[:, :, 0:1]

        # Pad sequence using left padding (like Chronos)
        if seq_len < self.context_length:
            x = self.pad_sequence(x, self.context_length)
        elif seq_len > self.context_length:
            # Take the most recent values (left padding style)
            x = x[:, -self.context_length:]

        if self.use_fallback:
            # Project input to model dimension
            x_proj = self.input_projection(x)  # (batch, seq, hidden)

            if hasattr(self, 'fallback_model'):
                # Use LLaMA model
                outputs = self.fallback_model(inputs_embeds=x_proj)
                hidden_states = outputs.last_hidden_state
            else:
                # Use standard transformer
                hidden_states = self.fallback_encoder(x_proj)

            # Pool over sequence - use last token (like GPT)
            pooled = hidden_states[:, -1, :]
        else:
            # Use actual Lag-Llama model
            # Lag-Llama expects specific input format
            x_univariate = x.squeeze(-1)  # (batch, seq)

            # Get embeddings from the model
            # Note: Lag-Llama is primarily for forecasting, so we extract intermediate representations
            with torch.no_grad():
                # Create a dummy forecast to get encoder states
                forecast = self.model.predict(x_univariate)

                # Extract hidden states (this is model-specific)
                # For now, we use the forecast values as a proxy
                pooled = forecast.mean(dim=1) if len(forecast.shape) > 1 else forecast

        # Project to output dimension
        output = self.projection(pooled)

        return output

    def get_model_name(self) -> str:
        return f"Lag-Llama-{self.model_name.split('/')[-1]}"