#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
PolymarketQADatasetExpanded.py
------------------------------
Expanded PyTorch-style QA dataset for Polymarket with:
- Multiple question types (past analysis + forecasting)
- 80/20 train/val split
- Increased samples through time window augmentation
"""

import sys
import os
from typing import List, Literal, Tuple
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from time_series_datasets.polymarket.polymarket_loader_expanded import (
    create_polymarket_splits,
    QUESTION_TEMPLATES,
)


class PolymarketQADatasetExpanded(QADataset):
    """
    Expanded Polymarket Question-Answer Dataset with:
    - 9 question types (past analysis + forecasting)
    - Multiple samples per market via time windowing
    - 80/20 train/validation split
    - No information leakage in prompts
    """

    def _load_splits(self) -> Tuple[List, List, List]:
        """
        Load and split the Polymarket data into train/validation sets.
        Returns empty list for test since we use 80/20 train/val only.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train, val = create_polymarket_splits(train_frac=0.8, seed=42)
        test = []  # No separate test set, use validation for evaluation
        return train, val, test

    def _get_answer(self, row) -> str:
        """
        Get the pre-computed ground truth answer.

        Args:
            row: Dataset row containing market data and answer

        Returns:
            The ground truth answer string
        """
        return row['answer']

    def _get_pre_prompt(self, row) -> str:
        """
        Get the pre-prompt providing context about the task.

        Args:
            row: Dataset row

        Returns:
            Pre-prompt string
        """
        return "You are an expert in analyzing prediction markets and interpreting time series data."

    def _get_post_prompt(self, row) -> str:
        """
        Get the post-prompt with the specific question for this sample.

        Args:
            row: Dataset row

        Returns:
            Post-prompt string with clear question and expected answer format
        """
        question_type = row['question_type']
        return QUESTION_TEMPLATES[question_type]['post_prompt']

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Create text-time series prompts from the market data.

        IMPORTANT: NO statistics included to prevent information leakage!
        The model must learn to extract features from the time series embeddings.

        Args:
            row: Dataset row containing series data

        Returns:
            List of TextTimeSeriesPrompt objects with the time series
        """
        series = row['series']

        # Convert to tensor if needed
        if isinstance(series, list):
            series_tensor = torch.tensor(series, dtype=torch.float32)
        elif isinstance(series, np.ndarray):
            series_tensor = torch.from_numpy(series).float()
        else:
            series_tensor = series

        # Normalize the series (model never sees raw statistics)
        mean = series_tensor.mean()
        std = series_tensor.std()

        if std > 0:
            normalized_series = (series_tensor - mean) / std
        else:
            normalized_series = series_tensor - mean

        # Create minimal prompt WITHOUT any statistics
        question = row['question']

        text_prompt = (
            f"This is the prediction market price history for the question: '{question}'"
        )

        return [TextTimeSeriesPrompt(text_prompt, normalized_series.tolist())]

    def _format_sample(self, row):
        """Override to preserve metadata."""
        # Get the base formatted sample
        base_sample = super()._format_sample(row)

        # Add metadata
        base_sample['market_id'] = row.get('market_id', '')
        base_sample['question_type'] = row.get('question_type', '')
        base_sample['window'] = row.get('window', '')

        return base_sample


# ---------------------------
# Example usage and testing
# ---------------------------

if __name__ == "__main__":
    print("="*80)
    print("Testing PolymarketQADatasetExpanded")
    print("="*80)

    # Create dataset instances
    print("\n1. Creating datasets...")
    train_dataset = PolymarketQADatasetExpanded("train", "<eos>")
    val_dataset = PolymarketQADatasetExpanded("validation", "<eos>")

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")

    # Example samples
    print("\n" + "="*80)
    print("2. Example samples:")
    print("="*80)

    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]

        print(f"\nSample {i+1}:")
        print(f"  Question Type: {sample.get('question_type', 'N/A')}")
        print(f"  Window: {sample.get('window', 'N/A')}")
        print(f"  Time series text: {sample['time_series_text'][0][:100]}...")
        print(f"  Post-prompt: {sample['post_prompt'][:80]}...")
        print(f"  Answer: {sample['answer']}")
        print(f"  Time series length: {len(sample['time_series'][0])}")

    print("\n" + "="*80)
    print("✅ SUCCESS: Expanded dataset loaded!")
    print("="*80)
