#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
polymarket_loader_expanded.py
-----------------------------
Expanded Polymarket data loader with:
1. More diverse question types (past analysis + forecasting)
2. Multiple samples per market using different time windows
3. 80/20 train/val split
4. Ground truth functions for all question types
"""

import json
import os
from typing import Tuple, List, Dict, Any
import numpy as np


# ============================================================================
# EXPANDED QUESTION TYPES
# ============================================================================

QUESTION_TYPES = [
    # Past data analysis
    'trend',
    'volatility',
    'momentum',
    'stability',
    'price_level',
    'change_magnitude',

    # Forecasting (using held-out final portion)
    'next_movement',
    'short_term_direction',
    'reversal_likelihood',
]

QUESTION_TEMPLATES = {
    # === Past Analysis Questions ===
    'trend': {
        'post_prompt': "What is the overall trend in this prediction market over the shown period? Answer with: upward, downward, or sideways.",
        'possible_answers': ['upward', 'downward', 'sideways'],
        'type': 'past'
    },
    'volatility': {
        'post_prompt': "What is the volatility level of this prediction market? Answer with: high, medium, or low.",
        'possible_answers': ['high', 'medium', 'low'],
        'type': 'past'
    },
    'momentum': {
        'post_prompt': "Does this prediction market show positive or negative momentum? Answer with: positive or negative.",
        'possible_answers': ['positive', 'negative'],
        'type': 'past'
    },
    'stability': {
        'post_prompt': "Is this prediction market price stable or unstable? Answer with: stable or unstable.",
        'possible_answers': ['stable', 'unstable'],
        'type': 'past'
    },
    'price_level': {
        'post_prompt': "What is the typical probability level for this prediction market? Answer with: very low (0-20%), low (20-40%), medium (40-60%), high (60-80%), or very high (80-100%).",
        'possible_answers': ['very low', 'low', 'medium', 'high', 'very high'],
        'type': 'past'
    },
    'change_magnitude': {
        'post_prompt': "How significant are the price changes in this market? Answer with: large changes, moderate changes, or small changes.",
        'possible_answers': ['large changes', 'moderate changes', 'small changes'],
        'type': 'past'
    },

    # === Forecasting Questions ===
    'next_movement': {
        'post_prompt': "Based on the market patterns shown, will the probability increase, decrease, or stay flat in the immediate next period? Answer with: increase, decrease, or stay flat.",
        'possible_answers': ['increase', 'decrease', 'stay flat'],
        'type': 'forecast'
    },
    'short_term_direction': {
        'post_prompt': "What is the most likely short-term direction for this market? Answer with: upward trend, downward trend, or sideways movement.",
        'possible_answers': ['upward trend', 'downward trend', 'sideways movement'],
        'type': 'forecast'
    },
    'reversal_likelihood': {
        'post_prompt': "Based on current momentum, is a trend reversal likely to occur soon? Answer with: likely or unlikely.",
        'possible_answers': ['likely', 'unlikely'],
        'type': 'forecast'
    },
}


# ============================================================================
# GROUND TRUTH COMPUTATION FUNCTIONS
# ============================================================================

def compute_trend(prices: np.ndarray, window: int = None) -> str:
    """Determine trend direction from time series."""
    if len(prices) < 2:
        return 'sideways'

    window = min(window or len(prices), len(prices))
    recent_prices = prices[-window:] if window else prices

    # Linear regression slope
    x = np.arange(len(recent_prices))
    slope = np.polyfit(x, recent_prices, 1)[0]

    # Normalize by price range
    price_range = np.ptp(recent_prices)
    if price_range < 1e-6:
        return 'sideways'

    relative_slope = slope / price_range

    if relative_slope > 0.003:
        return 'upward'
    elif relative_slope < -0.003:
        return 'downward'
    else:
        return 'sideways'


def compute_volatility(prices: np.ndarray, window: int = None) -> str:
    """Determine volatility level from time series."""
    if len(prices) < 2:
        return 'low'

    window = min(window or 50, len(prices))
    recent_prices = prices[-window:]

    # Calculate volatility as std of price changes
    price_changes = np.diff(recent_prices)
    volatility = np.std(price_changes)

    # Normalize by overall price range
    price_range = np.ptp(prices)
    if price_range < 1e-6:
        return 'low'

    normalized_vol = volatility / price_range

    if normalized_vol > 0.025:
        return 'high'
    elif normalized_vol > 0.01:
        return 'medium'
    else:
        return 'low'


def compute_momentum(prices: np.ndarray, short_window: int = 20, long_window: int = 50) -> str:
    """Determine momentum from moving averages."""
    if len(prices) < long_window:
        if len(prices) < 2:
            return 'positive'
        return 'positive' if prices[-1] > prices[0] else 'negative'

    short_ma = np.mean(prices[-short_window:])
    long_ma = np.mean(prices[-long_window:])

    return 'positive' if short_ma > long_ma else 'negative'


def compute_stability(prices: np.ndarray, window: int = None) -> str:
    """Determine if market is stable or unstable."""
    if len(prices) < 2:
        return 'stable'

    window = min(window or 50, len(prices))
    recent_prices = prices[-window:]

    # Calculate coefficient of variation
    mean_price = np.mean(recent_prices)
    std_price = np.std(recent_prices)

    if mean_price < 1e-6:
        return 'stable'

    cv = std_price / mean_price

    return 'unstable' if cv > 0.15 else 'stable'


def compute_price_level(prices: np.ndarray) -> str:
    """Determine typical price level."""
    mean_price = np.mean(prices)

    if mean_price < 0.20:
        return 'very low'
    elif mean_price < 0.40:
        return 'low'
    elif mean_price < 0.60:
        return 'medium'
    elif mean_price < 0.80:
        return 'high'
    else:
        return 'very high'


def compute_change_magnitude(prices: np.ndarray) -> str:
    """Determine magnitude of price changes."""
    if len(prices) < 2:
        return 'small changes'

    price_changes = np.abs(np.diff(prices))
    avg_change = np.mean(price_changes)

    # Normalize by overall price range
    price_range = np.ptp(prices)
    if price_range < 1e-6:
        return 'small changes'

    relative_change = avg_change / price_range

    if relative_change > 0.02:
        return 'large changes'
    elif relative_change > 0.01:
        return 'moderate changes'
    else:
        return 'small changes'


def compute_next_movement(history_prices: np.ndarray, future_prices: np.ndarray) -> str:
    """Forecast immediate next movement using held-out data."""
    if len(future_prices) < 1:
        return 'stay flat'

    last_historical = history_prices[-1]
    next_price = future_prices[0]

    change = next_price - last_historical
    threshold = 0.01  # 1% change threshold

    if change > threshold:
        return 'increase'
    elif change < -threshold:
        return 'decrease'
    else:
        return 'stay flat'


def compute_short_term_direction(history_prices: np.ndarray, future_prices: np.ndarray, window: int = 5) -> str:
    """Forecast short-term direction using held-out data."""
    if len(future_prices) < window:
        window = len(future_prices)

    if window < 2:
        return 'sideways movement'

    # Use first 'window' points of future
    future_window = future_prices[:window]

    # Compute trend on future window
    x = np.arange(len(future_window))
    slope = np.polyfit(x, future_window, 1)[0]

    price_range = np.ptp(np.concatenate([history_prices, future_prices]))
    if price_range < 1e-6:
        return 'sideways movement'

    relative_slope = slope / price_range

    if relative_slope > 0.002:
        return 'upward trend'
    elif relative_slope < -0.002:
        return 'downward trend'
    else:
        return 'sideways movement'


def compute_reversal_likelihood(history_prices: np.ndarray, future_prices: np.ndarray, window: int = 10) -> str:
    """Determine if trend reversal occurs in near future."""
    if len(history_prices) < 20 or len(future_prices) < window:
        return 'unlikely'

    # Determine historical trend (last 20 points)
    hist_recent = history_prices[-20:]
    hist_slope = np.polyfit(np.arange(len(hist_recent)), hist_recent, 1)[0]

    # Determine future trend
    future_window = future_prices[:window]
    if len(future_window) < 2:
        return 'unlikely'
    future_slope = np.polyfit(np.arange(len(future_window)), future_window, 1)[0]

    # Check if slopes have opposite signs (reversal)
    if hist_slope * future_slope < 0 and abs(hist_slope) > 0.001:
        return 'likely'
    else:
        return 'unlikely'


# Mapping from question type to ground truth function
GROUND_TRUTH_FUNCTIONS = {
    'trend': lambda h, f=None: compute_trend(h),
    'volatility': lambda h, f=None: compute_volatility(h),
    'momentum': lambda h, f=None: compute_momentum(h),
    'stability': lambda h, f=None: compute_stability(h),
    'price_level': lambda h, f=None: compute_price_level(h),
    'change_magnitude': lambda h, f=None: compute_change_magnitude(h),
    'next_movement': compute_next_movement,
    'short_term_direction': compute_short_term_direction,
    'reversal_likelihood': compute_reversal_likelihood,
}


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_polymarket_data(data_file: str = None) -> List[Dict[str, Any]]:
    """Load Polymarket market data from JSON file."""
    if data_file is None:
        data_file = "/local/home/wangni/polymarket_data/raw/market_price_data.json"

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Polymarket data file not found: {data_file}")

    with open(data_file, 'r') as f:
        data = json.load(f)

    return data


def create_samples_from_market(market: Dict[str, Any], min_length: int = 200) -> List[Dict[str, Any]]:
    """
    Create multiple training samples from a single market using:
    1. Different time windows (to create more diverse samples)
    2. Different question types

    For forecasting questions, we hold out the last 10% of data as "future" for ground truth.
    """
    price_history = market.get('price_history', [])

    if not price_history or len(price_history) < min_length:
        return []

    # Extract prices only
    prices = np.array([entry['p'] for entry in price_history])

    # Validate prices
    if np.isnan(prices).any() or np.isinf(prices).any():
        return []

    samples = []

    # Create samples with different time windows
    # Window 1: Full series (for past analysis)
    # Window 2: First 70% (for forecasting - hold out last 30%)
    # Window 3: Middle 70% (different perspective)

    windows = [
        ('full', 0, len(prices), 0),  # (name, start_idx, end_history_idx, end_total_idx)
        ('early_forecast', 0, int(len(prices) * 0.7), len(prices)),
        ('mid_forecast', int(len(prices) * 0.15), int(len(prices) * 0.85), len(prices)),
    ]

    for window_name, start_idx, history_end_idx, total_end_idx in windows:
        if history_end_idx - start_idx < min_length:
            continue

        history_prices = prices[start_idx:history_end_idx]
        future_prices = prices[history_end_idx:total_end_idx] if total_end_idx > history_end_idx else np.array([])

        # For each question type
        for q_type in QUESTION_TYPES:
            template = QUESTION_TEMPLATES[q_type]

            # Skip forecasting questions if no future data
            if template['type'] == 'forecast' and len(future_prices) < 5:
                continue

            # Compute ground truth
            try:
                if template['type'] == 'forecast':
                    answer = GROUND_TRUTH_FUNCTIONS[q_type](history_prices, future_prices)
                else:
                    answer = GROUND_TRUTH_FUNCTIONS[q_type](history_prices, None)
            except Exception as e:
                print(f"Warning: Failed to compute ground truth for {q_type}: {e}")
                continue

            # Create sample
            sample = {
                'market_id': market.get('market_id'),
                'question': market.get('question'),
                'description': market.get('description', ''),
                'series': history_prices,  # Only history for model input
                'num_points': len(history_prices),
                'question_type': q_type,
                'window': window_name,
                'answer': answer,
            }
            samples.append(sample)

    return samples


def create_polymarket_splits(
    data_file: str = None,
    train_frac: float = 0.8,
    seed: int = 42,
    min_length: int = 200,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load Polymarket data and create train/validation splits with expanded samples.

    Args:
        data_file: Path to the JSON data file
        train_frac: Fraction for training set (rest is validation)
        seed: Random seed for reproducibility
        min_length: Minimum time series length to include

    Returns:
        Tuple of (train_data, val_data)
    """
    # Load raw data
    raw_data = load_polymarket_data(data_file)

    if not raw_data:
        raise ValueError("No market data loaded!")

    print(f"\nLoaded {len(raw_data)} markets from file")

    # Generate samples from each market
    all_samples = []
    for market in raw_data:
        samples = create_samples_from_market(market, min_length=min_length)
        all_samples.extend(samples)

    print(f"Generated {len(all_samples)} total samples from {len(raw_data)} markets")

    # Split samples into train/val
    np.random.seed(seed)
    indices = np.random.permutation(len(all_samples))

    n_train = int(len(all_samples) * train_frac)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]

    print(f"\nPolymarket data splits (80/20):")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples: {len(val_samples)}")

    # Print question type breakdown
    train_q_types = {}
    for s in train_samples:
        q_type = s['question_type']
        train_q_types[q_type] = train_q_types.get(q_type, 0) + 1

    print(f"\nQuestion type breakdown (train):")
    for q_type, count in sorted(train_q_types.items()):
        print(f"  {q_type}: {count}")

    return train_samples, val_samples


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Testing Expanded Polymarket Loader")
    print("="*80)

    train, val = create_polymarket_splits()

    print(f"\n{'='*80}")
    print("Example samples:")
    print("="*80)

    for i in range(min(5, len(train))):
        sample = train[i]
        print(f"\nSample {i+1}:")
        print(f"  Market: {sample['question'][:60]}...")
        print(f"  Window: {sample['window']}")
        print(f"  Question Type: {sample['question_type']}")
        print(f"  Series length: {sample['num_points']}")
        print(f"  Answer: {sample['answer']}")
