#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
polymarket_loader.py
-------------------
Data loader for Polymarket prediction market data with ground truth generation.
Designed to prevent information leakage and force model to learn from time series embeddings.
"""

import json
import os
from typing import Tuple, List, Dict, Any
import numpy as np


# ============================================================================
# QUESTION TEMPLATES - Investor-relevant questions with clear answers
# ============================================================================

QUESTION_TYPES = [
    'trend',
    'volatility',
    'momentum',
    'stability',
    'direction'
]

QUESTION_TEMPLATES = {
    'trend': {
        'post_prompt': "What is the current trend in this prediction market? Answer with: upward, downward, or sideways.",
        'possible_answers': ['upward', 'downward', 'sideways']
    },
    'volatility': {
        'post_prompt': "What is the volatility level of this prediction market? Answer with: high, medium, or low.",
        'possible_answers': ['high', 'medium', 'low']
    },
    'momentum': {
        'post_prompt': "Does this prediction market show positive or negative momentum? Answer with: positive or negative.",
        'possible_answers': ['positive', 'negative']
    },
    'stability': {
        'post_prompt': "Is this prediction market price stable or unstable? Answer with: stable or unstable.",
        'possible_answers': ['stable', 'unstable']
    },
    'direction': {
        'post_prompt': "Based on recent price action, is this market's probability likely to increase, decrease, or remain stable? Answer with: increase, decrease, or remain stable.",
        'possible_answers': ['increase', 'decrease', 'remain stable']
    }
}


# ============================================================================
# GROUND TRUTH GENERATION FUNCTIONS
# These compute answers from time series data ONLY (no pre-computed stats)
# ============================================================================

def compute_trend(prices: np.ndarray, window: int = 100) -> str:
    """
    Determine trend direction from time series.

    Args:
        prices: Price history array
        window: Number of recent points to analyze

    Returns:
        'upward', 'downward', or 'sideways'
    """
    if len(prices) < 2:
        return 'sideways'

    # Use recent window or full series if shorter
    window = min(window, len(prices))
    recent_prices = prices[-window:]

    # Linear regression slope
    x = np.arange(len(recent_prices))
    slope = np.polyfit(x, recent_prices, 1)[0]

    # Normalize by price range to get relative slope
    price_range = np.ptp(recent_prices)
    if price_range < 1e-6:  # Essentially flat
        return 'sideways'

    relative_slope = slope / price_range

    # Thresholds for classification
    if relative_slope > 0.003:  # Significant upward slope
        return 'upward'
    elif relative_slope < -0.003:  # Significant downward slope
        return 'downward'
    else:
        return 'sideways'


def compute_volatility(prices: np.ndarray, window: int = 50) -> str:
    """
    Determine volatility level from time series.

    Args:
        prices: Price history array
        window: Number of recent points to analyze

    Returns:
        'high', 'medium', or 'low'
    """
    if len(prices) < 2:
        return 'low'

    # Use recent window
    window = min(window, len(prices))
    recent_prices = prices[-window:]

    # Calculate volatility as std of price changes
    price_changes = np.diff(recent_prices)
    volatility = np.std(price_changes)

    # Normalize by overall price range
    price_range = np.ptp(prices)
    if price_range < 1e-6:
        return 'low'

    normalized_vol = volatility / price_range

    # Classification thresholds
    if normalized_vol > 0.025:
        return 'high'
    elif normalized_vol > 0.01:
        return 'medium'
    else:
        return 'low'


def compute_momentum(prices: np.ndarray, short_window: int = 20, long_window: int = 50) -> str:
    """
    Determine momentum from moving averages.

    Args:
        prices: Price history array
        short_window: Short-term moving average window
        long_window: Long-term moving average window

    Returns:
        'positive' or 'negative'
    """
    if len(prices) < long_window:
        # Fall back to simple comparison
        if len(prices) < 2:
            return 'positive'
        return 'positive' if prices[-1] > prices[0] else 'negative'

    # Calculate moving averages
    short_ma = np.mean(prices[-short_window:])
    long_ma = np.mean(prices[-long_window:])

    return 'positive' if short_ma > long_ma else 'negative'


def compute_stability(prices: np.ndarray, window: int = 50) -> str:
    """
    Determine if market is stable or unstable based on variance.

    Args:
        prices: Price history array
        window: Number of recent points to analyze

    Returns:
        'stable' or 'unstable'
    """
    if len(prices) < 2:
        return 'stable'

    window = min(window, len(prices))
    recent_prices = prices[-window:]

    # Calculate coefficient of variation
    mean_price = np.mean(recent_prices)
    std_price = np.std(recent_prices)

    if mean_price < 1e-6:
        return 'stable'

    cv = std_price / mean_price

    # Threshold for stability
    return 'unstable' if cv > 0.15 else 'stable'


def compute_direction(prices: np.ndarray, lookback: int = 50, lookahead: int = 20) -> str:
    """
    Predict future direction based on recent momentum and trend.

    Args:
        prices: Price history array
        lookback: Historical window to analyze
        lookahead: Forward-looking window (simulated with recent data)

    Returns:
        'increase', 'decrease', or 'remain stable'
    """
    if len(prices) < lookback + lookahead:
        # Fall back to trend
        return compute_trend(prices, window=min(50, len(prices)))

    # Use historical pattern: compare recent segment to prior segment
    recent_mean = np.mean(prices[-lookahead:])
    prior_mean = np.mean(prices[-(lookback+lookahead):-lookahead])

    if prior_mean < 1e-6:
        return 'remain stable'

    change_ratio = (recent_mean - prior_mean) / prior_mean

    if change_ratio > 0.05:
        return 'increase'
    elif change_ratio < -0.05:
        return 'decrease'
    else:
        return 'remain stable'


# Mapping from question type to ground truth function
GROUND_TRUTH_FUNCTIONS = {
    'trend': compute_trend,
    'volatility': compute_volatility,
    'momentum': compute_momentum,
    'stability': compute_stability,
    'direction': compute_direction
}


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_polymarket_data(data_file: str = None) -> List[Dict[str, Any]]:
    """
    Load Polymarket market data from JSON file.

    Args:
        data_file: Path to the JSON data file. If None, uses default location.

    Returns:
        List of market dictionaries with price history
    """
    if data_file is None:
        data_file = "/local/home/wangni/polymarket_data/raw/market_price_data.json"

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Polymarket data file not found: {data_file}")

    with open(data_file, 'r') as f:
        data = json.load(f)

    return data


def process_market_data(market: Dict[str, Any], question_type: str) -> Dict[str, Any]:
    """
    Process a single market's data for a specific question type.

    Args:
        market: Market dictionary with price_history
        question_type: Type of question to generate (from QUESTION_TYPES)

    Returns:
        Processed market dictionary with time series and question-specific data
    """
    price_history = market.get('price_history', [])

    if not price_history or len(price_history) < 10:
        return None

    # Extract prices only
    prices = np.array([entry['p'] for entry in price_history])

    # Validate prices
    if np.isnan(prices).any() or np.isinf(prices).any():
        return None

    # Create processed sample (NO STATISTICS - prevents information leakage!)
    processed = {
        'market_id': market.get('market_id'),
        'question': market.get('question'),
        'description': market.get('description', ''),
        'series': prices,  # Raw time series only
        'num_points': len(prices),
        'question_type': question_type,  # NEW: Question type for this sample
    }

    return processed


def create_polymarket_splits(
    data_file: str = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    samples_per_market: int = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load Polymarket data and create train/validation/test splits.

    Creates multiple samples per market (one for each question type) to increase
    training data diversity.

    Args:
        data_file: Path to the JSON data file
        train_frac: Fraction for training set
        val_frac: Fraction for validation set
        test_frac: Fraction for test set
        seed: Random seed for reproducibility
        samples_per_market: How many question types per market. If None, uses all.

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Load raw data
    raw_data = load_polymarket_data(data_file)

    if not raw_data:
        raise ValueError("No market data loaded!")

    # First, split markets into train/val/test
    np.random.seed(seed)
    n_markets = len(raw_data)
    indices = np.random.permutation(n_markets)

    n_train = int(n_markets * train_frac)
    n_val = int(n_markets * val_frac)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Process each market into multiple samples (one per question type)
    train_samples = []
    val_samples = []
    test_samples = []

    # Determine which question types to use
    if samples_per_market is None:
        question_types_to_use = QUESTION_TYPES
    else:
        question_types_to_use = QUESTION_TYPES[:samples_per_market]

    print(f"Creating {len(question_types_to_use)} samples per market (question types: {question_types_to_use})")

    # Process training markets
    for idx in train_indices:
        market = raw_data[idx]
        for q_type in question_types_to_use:
            processed = process_market_data(market, q_type)
            if processed is not None:
                train_samples.append(processed)

    # Process validation markets
    for idx in val_indices:
        market = raw_data[idx]
        for q_type in question_types_to_use:
            processed = process_market_data(market, q_type)
            if processed is not None:
                val_samples.append(processed)

    # Process test markets
    for idx in test_indices:
        market = raw_data[idx]
        for q_type in question_types_to_use:
            processed = process_market_data(market, q_type)
            if processed is not None:
                test_samples.append(processed)

    if not train_samples:
        raise ValueError("No valid training samples created!")

    print(f"\nPolymarket data splits:")
    print(f"  Markets: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
    print(f"  Samples: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")
    print(f"  Samples per market: {len(question_types_to_use)}")

    return train_samples, val_samples, test_samples


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the loader
    print("Testing Polymarket loader with new framework...")

    train, val, test = create_polymarket_splits()

    print("\n" + "="*80)
    print("Example training samples (different question types):")
    print("="*80)

    for i in range(min(3, len(train))):
        sample = train[i]
        q_type = sample['question_type']

        print(f"\nSample {i+1}:")
        print(f"  Market: {sample['question']}")
        print(f"  Question Type: {q_type}")
        print(f"  Series length: {sample['num_points']}")

        # Compute ground truth
        prices = sample['series']
        answer = GROUND_TRUTH_FUNCTIONS[q_type](prices)

        print(f"  Ground truth answer: {answer}")
        print(f"  Post-prompt: {QUESTION_TEMPLATES[q_type]['post_prompt']}")

    print("\n" + "="*80)
    print("Verifying NO information leakage:")
    print("="*80)
    sample = train[0]
    print(f"✅ Sample keys: {list(sample.keys())}")
    print(f"✅ Contains 'series': {('series' in sample)}")
    print(f"✅ Contains 'question_type': {('question_type' in sample)}")
    print(f"❌ Contains 'price_mean': {('price_mean' in sample)}")  # Should be False
    print(f"❌ Contains 'volatility': {('volatility' in sample)}")  # Should be False
    print(f"❌ Contains 'trend': {('trend' in sample)}")  # Should be False
