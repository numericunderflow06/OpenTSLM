#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import torch
from momentfm import MOMENTPipeline

from model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class MOMENTEncoder(TimeSeriesEncoderBase):
    """
    Pretrained MOMENT-1-large encoder for time series.

    Wraps the MOMENT foundation model (385M params, 1024-dim, patch_size=8)
    to produce patch-level features. All MOMENT weights are frozen by default.

    Input:  [B, L]  raw time series (any length, will be padded/truncated to 512)
    Output: [B, N, 1024]  patch-level features, N = seq_len // patch_size
    """

    MOMENT_SEQ_LEN = 512  # MOMENT's expected context length
    MOMENT_D_MODEL = 1024
    MOMENT_PATCH_LEN = 8

    def __init__(
        self,
        model_name: str = "AutonLab/MOMENT-1-large",
        freeze: bool = True,
    ):
        super().__init__(output_dim=self.MOMENT_D_MODEL, dropout=0.0)
        self.patch_size = self.MOMENT_PATCH_LEN
        self.freeze = freeze

        # Load pretrained MOMENT in embedding mode
        self.moment = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={"task_name": "embedding"},
        )
        self.moment.init()

        if freeze:
            for p in self.moment.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape [B, L], a batch of raw time series.
        Returns:
            FloatTensor of shape [B, N, 1024], where N = MOMENT_SEQ_LEN // patch_size = 64.
        """
        B, L = x.shape

        # Pad or truncate to MOMENT's expected sequence length (512)
        if L < self.MOMENT_SEQ_LEN:
            pad = x.new_zeros(B, self.MOMENT_SEQ_LEN - L)
            x_padded = torch.cat([x, pad], dim=1)
            # Create input mask: 1 for real values, 0 for padding
            input_mask = torch.ones(B, self.MOMENT_SEQ_LEN, device=x.device)
            input_mask[:, L:] = 0
        elif L > self.MOMENT_SEQ_LEN:
            x_padded = x[:, :self.MOMENT_SEQ_LEN]
            input_mask = torch.ones(B, self.MOMENT_SEQ_LEN, device=x.device)
        else:
            x_padded = x
            input_mask = torch.ones(B, self.MOMENT_SEQ_LEN, device=x.device)

        # MOMENT expects [B, n_channels, seq_len] -> add channel dim
        x_enc = x_padded.unsqueeze(1)  # [B, 1, 512]

        # Get patch-level embeddings (no reduction)
        outputs = self.moment(x_enc=x_enc, input_mask=input_mask, reduction="none")
        # outputs.embeddings: [B, n_channels=1, n_patches=64, d_model=1024]
        embeddings = outputs.embeddings.squeeze(1)  # [B, 64, 1024]

        return embeddings
