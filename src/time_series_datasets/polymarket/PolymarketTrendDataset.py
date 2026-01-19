"""
Enhanced Polymarket dataset for trend prediction training.
Splits time series in half and creates past/future trend questions.
"""

import json
import os
import sys
from typing import Literal, Tuple, List, Dict, Any
import numpy as np
from torch.utils.data import Dataset

from time_series_datasets.QADataset import QADataset


class PolymarketTrendDataset(QADataset):
    """
    Polymarket dataset with time series split in half.
    - First half: used for summary questions (past trend)
    - Second half: used for forecasting questions (future trend)

    Each market generates 2 questions:
    1. Past trend: "What was the trend in the first period?"
    2. Future forecast: "What is the trend in the second period?"

    Answers computed via linear regression slope (increasing/decreasing).
    Markets with slope=0 are excluded.
    """

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
        data_path: str = "/local/home/wangni/polymarket_data/processed/trend_dataset.json",
    ):
        """
        Initialize the trend dataset.

        Args:
            split: Dataset split
            EOS_TOKEN: End of sequence token
            format_sample_str: Whether to format as string
            time_series_format_function: Function to format time series
            data_path: Path to processed data file
        """
        self.data_path = data_path
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and split the dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Processed data not found at {self.data_path}. "
                f"Please run the data processing script first."
            )

        with open(self.data_path, 'r') as f:
            data = json.load(f)

        # Convert to dataset format
        all_samples = []
        for entry in data:
            all_samples.append(entry)

        # Split into train/val/test (80/10/10)
        n = len(all_samples)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        train_data = all_samples[:train_end]
        val_data = all_samples[train_end:val_end]
        test_data = all_samples[val_end:]

        return train_data, val_data, test_data

    def _get_answer(self, row) -> str:
        """Get the ground truth answer."""
        return row['answer']

    def _get_pre_prompt(self, row) -> str:
        """Get the pre-prompt."""
        return ""

    def _get_post_prompt(self, row) -> str:
        """Get the question prompt with explicit output format."""
        question_type = row['question_type']

        if question_type == 'past_trend':
            prompt = (
                "Based on the time series data above, what was the trend in the first period?\n\n"
                "Answer with ONLY one word: increasing or decreasing"
            )
        else:  # future_trend
            prompt = (
                "Based on the time series data above, what is the trend in the second period?\n\n"
                "Answer with ONLY one word: increasing or decreasing"
            )

        return prompt

    def _get_text_time_series_prompt_list(self, row) -> List:
        """Get the time series prompt list."""
        from prompt.text_time_series_prompt import TextTimeSeriesPrompt

        # Get the appropriate half of the time series
        time_series = np.array(row['time_series'])
        question_type = row['question_type']

        # Split in half
        mid_point = len(time_series) // 2

        if question_type == 'past_trend':
            # Use first half for past trend question
            ts_data = time_series[:mid_point]
            description = "This is the prediction market probability time series for the first period:"
        else:  # future_trend
            # Use first half as context (for the model to see)
            # Note: We don't give the second half to the model - that's what we're testing!
            ts_data = time_series[:mid_point]
            description = "This is the prediction market probability time series:"

        return [TextTimeSeriesPrompt(description, ts_data)]


def process_raw_polymarket_data(
    raw_data_path: str = "/local/home/wangni/polymarket_data/raw/market_price_data_enhanced.json",
    output_path: str = "/local/home/wangni/polymarket_data/processed/trend_dataset.json",
    max_markets: int = 1000,
):
    """
    Process raw Polymarket data into trend prediction dataset.

    Args:
        raw_data_path: Path to raw downloaded data
        output_path: Path to save processed data
        max_markets: Maximum number of markets to process
    """
    print("="*80)
    print("Processing Polymarket Data for Trend Prediction")
    print("="*80)

    # Load raw data
    print(f"\n[1/4] Loading raw data from {raw_data_path}...")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data not found at {raw_data_path}")

    with open(raw_data_path, 'r') as f:
        raw_markets = json.load(f)

    print(f"✓ Loaded {len(raw_markets)} markets")

    # Process each market
    print(f"\n[2/4] Processing markets (extracting time series and computing trends)...")
    processed_samples = []
    excluded_count = 0

    for i, market in enumerate(raw_markets[:max_markets]):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{min(len(raw_markets), max_markets)} markets...")

        # Extract price history
        price_history = market.get('price_history', [])
        if len(price_history) < 40:  # Need at least 40 points to split meaningfully
            excluded_count += 1
            continue

        # Extract prices (normalized to 0-1 range)
        prices = []
        for point in price_history:
            price = point.get('p', point.get('price', 0))
            prices.append(float(price))

        prices = np.array(prices)

        # Split in half
        mid_point = len(prices) // 2
        first_half = prices[:mid_point]
        second_half = prices[mid_point:]

        # Compute linear regression slopes
        x_first = np.arange(len(first_half))
        x_second = np.arange(len(second_half))

        slope_first = np.polyfit(x_first, first_half, 1)[0]
        slope_second = np.polyfit(x_second, second_half, 1)[0]

        # Determine trends (exclude if slope is exactly 0)
        if abs(slope_first) < 1e-10:
            excluded_count += 1
            continue

        if abs(slope_second) < 1e-10:
            excluded_count += 1
            continue

        trend_first = "increasing" if slope_first > 0 else "decreasing"
        trend_second = "increasing" if slope_second > 0 else "decreasing"

        # Create two samples: one for past trend, one for future
        base_info = {
            'market_id': market.get('market_id', ''),
            'question_text': market.get('question', ''),
            'time_series': prices.tolist(),
        }

        # Sample 1: Past trend (uses first half)
        past_sample = {
            **base_info,
            'question_type': 'past_trend',
            'answer': trend_first,
            'slope': float(slope_first),
        }
        processed_samples.append(past_sample)

        # Sample 2: Future trend (model sees first half, predicts second half trend)
        future_sample = {
            **base_info,
            'question_type': 'future_trend',
            'answer': trend_second,
            'slope': float(slope_second),
        }
        processed_samples.append(future_sample)

    print(f"✓ Processed {len(raw_markets[:max_markets])} markets")
    print(f"✓ Generated {len(processed_samples)} samples (2 per market)")
    print(f"✓ Excluded {excluded_count} markets (insufficient data or slope=0)")

    # Create output directory
    print(f"\n[3/4] Saving processed data...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(processed_samples, f, indent=2)

    print(f"✓ Saved to {output_path}")

    # Print statistics
    print(f"\n[4/4] Dataset Statistics:")
    print(f"  Total samples: {len(processed_samples)}")
    print(f"  Past trend samples: {sum(1 for s in processed_samples if s['question_type'] == 'past_trend')}")
    print(f"  Future trend samples: {sum(1 for s in processed_samples if s['question_type'] == 'future_trend')}")

    # Label distribution
    increasing_count = sum(1 for s in processed_samples if s['answer'] == 'increasing')
    decreasing_count = sum(1 for s in processed_samples if s['answer'] == 'decreasing')
    print(f"\n  Label distribution:")
    print(f"    Increasing: {increasing_count} ({100*increasing_count/len(processed_samples):.1f}%)")
    print(f"    Decreasing: {decreasing_count} ({100*decreasing_count/len(processed_samples):.1f}%)")

    # Time series length statistics
    ts_lengths = [len(s['time_series']) for s in processed_samples]
    print(f"\n  Time series lengths:")
    print(f"    Min: {min(ts_lengths)}")
    print(f"    Max: {max(ts_lengths)}")
    print(f"    Mean: {np.mean(ts_lengths):.1f}")
    print(f"    Median: {np.median(ts_lengths):.1f}")

    print("\n" + "="*80)
    print("✓ Processing Complete!")
    print("="*80)

    return processed_samples


if __name__ == "__main__":
    # Process the raw data
    process_raw_polymarket_data(
        max_markets=1000
    )
