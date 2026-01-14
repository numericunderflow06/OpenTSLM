#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

from abc import abstractmethod
from enum import Enum
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn

from opentslm.model_config import ENCODER_OUTPUT_DIM


class PaddingStrategy(Enum):
    """Padding strategies for different encoders."""
    LEFT = "left"       # Pad on the left (e.g., Chronos, Lag-Llama)
    RIGHT = "right"     # Pad on the right (e.g., PatchTST, Informer)
    CENTER = "center"   # Pad on both sides (e.g., Moment)
    NONE = "none"       # No padding needed (e.g., CNN-based models)


class PatchingStrategy(Enum):
    """Patching strategies for different encoders."""
    NONE = "none"                   # No patching
    NON_OVERLAPPING = "non_overlapping"  # Non-overlapping patches
    OVERLAPPING = "overlapping"      # Overlapping patches with stride


class EnhancedTimeSeriesEncoderBase(nn.Module):
    """
    Enhanced base class for time series encoders with padding and patching support.

    This base class provides common functionality for handling different padding
    and patching strategies used by various time series foundation models.
    """

    def __init__(
        self,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
        padding_strategy: PaddingStrategy = PaddingStrategy.RIGHT,
        padding_value: float = 0.0,
        patching_strategy: PatchingStrategy = PatchingStrategy.NONE,
        patch_size: Optional[int] = None,
        patch_stride: Optional[int] = None,
        use_nan_padding: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.dropout = dropout
        self.padding_strategy = padding_strategy
        self.padding_value = padding_value if not use_nan_padding else float('nan')
        self.patching_strategy = patching_strategy
        self.patch_size = patch_size
        self.patch_stride = patch_stride or patch_size
        self.use_nan_padding = use_nan_padding

        # Dropout layer if needed
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def pad_sequence(
        self,
        x: torch.Tensor,
        target_length: int,
        padding_strategy: Optional[PaddingStrategy] = None
    ) -> torch.Tensor:
        """
        Pad sequence to target length using specified strategy.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
            target_length: Target sequence length
            padding_strategy: Override default padding strategy

        Returns:
            Padded tensor
        """
        strategy = padding_strategy or self.padding_strategy
        current_length = x.shape[1]

        if current_length >= target_length:
            # Truncate if necessary
            if strategy == PaddingStrategy.LEFT:
                return x[:, -target_length:]
            elif strategy == PaddingStrategy.CENTER:
                start = (current_length - target_length) // 2
                return x[:, start:start + target_length]
            else:  # RIGHT or NONE
                return x[:, :target_length]

        # Calculate padding needed
        pad_length = target_length - current_length
        padding_shape = list(x.shape)
        padding_shape[1] = pad_length

        # Create padding tensor
        if self.use_nan_padding:
            padding = torch.full(padding_shape, float('nan'), device=x.device, dtype=x.dtype)
        else:
            padding = torch.full(padding_shape, self.padding_value, device=x.device, dtype=x.dtype)

        # Apply padding based on strategy
        if strategy == PaddingStrategy.LEFT:
            return torch.cat([padding, x], dim=1)
        elif strategy == PaddingStrategy.RIGHT:
            return torch.cat([x, padding], dim=1)
        elif strategy == PaddingStrategy.CENTER:
            left_pad = pad_length // 2
            right_pad = pad_length - left_pad
            left_padding = padding[:, :left_pad]
            right_padding = padding[:, :right_pad]
            return torch.cat([left_padding, x, right_padding], dim=1)
        else:  # NONE
            return x

    def create_patches(
        self,
        x: torch.Tensor,
        patch_size: Optional[int] = None,
        patch_stride: Optional[int] = None
    ) -> torch.Tensor:
        """
        Create patches from time series data.

        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
            patch_size: Override default patch size
            patch_stride: Override default patch stride

        Returns:
            Patched tensor of shape (batch_size, n_patches, patch_size, n_features)
        """
        if self.patching_strategy == PatchingStrategy.NONE:
            return x

        patch_size = patch_size or self.patch_size
        patch_stride = patch_stride or self.patch_stride

        if patch_size is None:
            return x

        batch_size, seq_len, n_features = x.shape

        # Use unfold to create patches
        patches = x.unfold(dimension=1, size=patch_size, step=patch_stride)
        # Reshape: (batch, n_patches, n_features, patch_size) -> (batch, n_patches, patch_size, n_features)
        patches = patches.permute(0, 1, 3, 2)

        return patches

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return {
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "padding_strategy": self.padding_strategy.value,
            "padding_value": self.padding_value,
            "patching_strategy": self.patching_strategy.value,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "use_nan_padding": self.use_nan_padding,
        }

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: Input time series of shape (batch_size, seq_len, n_features)

        Returns:
            Encoded representation
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the encoder model."""
        pass