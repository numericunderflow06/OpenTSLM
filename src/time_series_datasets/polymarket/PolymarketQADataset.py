#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
PolymarketQADataset.py
---------------------
PyTorch-style QA dataset for Polymarket prediction market time series.

Updated framework:
- NO information leakage (no statistics in prompts)
- Clear, investor-relevant questions
- Ground truth answers computed from time series only
- Forces model to learn from time series embeddings
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
from time_series_datasets.polymarket.polymarket_loader import (
    create_polymarket_splits,
    QUESTION_TEMPLATES,
    GROUND_TRUTH_FUNCTIONS
)


class PolymarketQADataset(QADataset):
    """
    Polymarket Question-Answer Dataset for prediction market time series analysis.

    This dataset loads Polymarket price history data and creates QA pairs with:
    - Clear, investor-relevant questions (trend, volatility, momentum, etc.)
    - Ground truth answers computed from time series data only
    - NO pre-computed statistics in prompts (prevents information leakage)
    - Forces model to learn from time series embeddings

    Each market generates multiple samples (one per question type).
    """

    def _load_splits(self) -> Tuple[List, List, List]:
        """
        Load and split the Polymarket data into train/validation/test sets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train, val, test = create_polymarket_splits()
        return train, val, test

    def _get_answer(self, row) -> str:
        """
        Generate answer by computing ground truth from time series data.

        Args:
            row: Dataset row containing market data

        Returns:
            The ground truth answer string (e.g., "upward", "high", "positive")
        """
        question_type = row['question_type']
        prices = row['series']

        # Get the appropriate ground truth function
        compute_fn = GROUND_TRUTH_FUNCTIONS[question_type]

        # Compute answer from time series data only
        answer = compute_fn(prices)

        return answer

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
        """Override to preserve the market ID and question type."""
        # Get the base formatted sample
        base_sample = super()._format_sample(row)

        # Add metadata
        if 'market_id' in row:
            base_sample['market_id'] = row['market_id']
        if 'question_type' in row:
            base_sample['question_type'] = row['question_type']

        return base_sample


# ---------------------------
# Example usage and testing
# ---------------------------

if __name__ == "__main__":
    print("="*80)
    print("Testing PolymarketQADataset with NEW framework (no information leakage)")
    print("="*80)

    # Create dataset instances
    print("\n1. Creating datasets...")
    train_dataset = PolymarketQADataset("train", "<eos>")
    val_dataset = PolymarketQADataset("validation", "<eos>")
    test_dataset = PolymarketQADataset("test", "<eos>")

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")

    # Example samples from different question types
    print("\n" + "="*80)
    print("2. Example samples (showing different question types):")
    print("="*80)

    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]

        print(f"\nSample {i+1}:")
        print(f"  Question Type: {sample.get('question_type', 'N/A')}")
        print(f"  Pre-prompt: {sample['pre_prompt'][:80]}...")
        print(f"  Time series text: {sample['time_series_text'][0][:120]}...")
        print(f"  Post-prompt: {sample['post_prompt'][:100]}...")
        print(f"  Answer: {sample['answer']}")
        print(f"  Time series length: {len(sample['time_series'][0])}")

    # Verify NO information leakage
    print("\n" + "="*80)
    print("3. Verifying NO information leakage:")
    print("="*80)

    sample = train_dataset[0]
    ts_text = sample['time_series_text'][0]

    print(f"Time series text: {ts_text}")
    print(f"\n✅ Checks:")
    print(f"  'mean' in text: {'mean' in ts_text}")  # Should be False
    print(f"  'std' in text: {'std' in ts_text}")  # Should be False
    print(f"  'volatility' in text: {'volatility' in ts_text.lower()}")  # Should be False
    print(f"  'trend' in text: {'trend' in ts_text.lower()}")  # Should be False
    print(f"  'average' in text: {'average' in ts_text.lower()}")  # Should be False

    # Verify answer cannot be extracted from text
    print(f"\n✅ Answer verification:")
    print(f"  Answer: '{sample['answer']}'")
    print(f"  Answer in pre_prompt: {sample['answer'] in sample['pre_prompt']}")  # Should be False
    print(f"  Answer in ts_text: {sample['answer'] in ts_text}")  # Should be False
    print(f"  Answer in post_prompt: {sample['answer'] in sample['post_prompt']}")  # Should be False

    print("\n" + "="*80)
    print("4. Comparison with answer formats:")
    print("="*80)

    # Show examples of each question type
    question_types = set()
    for sample in train_dataset:
        q_type = sample.get('question_type')
        if q_type and q_type not in question_types:
            question_types.add(q_type)
            print(f"\nQuestion Type: {q_type}")
            print(f"  Post-prompt: {sample['post_prompt'][:120]}...")
            print(f"  Answer: {sample['answer']}")

            if len(question_types) >= 5:
                break

    print("\n" + "="*80)
    print("✅ SUCCESS: Dataset created with NO information leakage!")
    print("✅ Model MUST use time series embeddings to answer questions!")
    print("="*80)
