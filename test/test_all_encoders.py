#!/usr/bin/env python3
#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Unified test suite for all time series encoders.

This script tests all available encoders with pretrained checkpoints.
"""

import os
import sys
import torch
import time
import traceback
from typing import Dict, Any, List, Tuple

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from opentslm.model_config import ENCODER_OUTPUT_DIM

# Import all encoder classes
from opentslm.model.encoder.Chronos2Encoder import Chronos2Encoder, CHRONOS_AVAILABLE
from opentslm.model.encoder.PatchTSTEncoder import PatchTSTEncoder, PATCHTST_AVAILABLE
from opentslm.model.encoder.TimesFMEncoder import TimesFMEncoder, TIMESFM_AVAILABLE
from opentslm.model.encoder.MomentEncoder import MomentEncoder, MOMENT_AVAILABLE
from opentslm.model.encoder.LagLlamaEncoder import LagLlamaEncoder, LAG_LLAMA_AVAILABLE

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


class EncoderTestConfig:
    """Configuration for encoder tests."""

    def __init__(self, name: str, encoder_class: type, model_name: str,
                 available: bool, test_params: Dict[str, Any] = None):
        self.name = name
        self.encoder_class = encoder_class
        self.model_name = model_name
        self.available = available
        self.test_params = test_params or {}


# Define test configurations for each encoder with pretrained models
ENCODER_CONFIGS = [
    EncoderTestConfig(
        name="Chronos2-Tiny",
        encoder_class=Chronos2Encoder,
        model_name="amazon/chronos-t5-tiny",
        available=CHRONOS_AVAILABLE,
        test_params={"output_dim": ENCODER_OUTPUT_DIM, "device": DEVICE}
    ),
    EncoderTestConfig(
        name="PatchTST-ETTh1",
        encoder_class=PatchTSTEncoder,
        model_name="ibm/patchtst-etth1-pretrain",
        available=PATCHTST_AVAILABLE,
        test_params={
            "output_dim": ENCODER_OUTPUT_DIM,
            "device": DEVICE,
            "context_length": 512,
            "patch_size": 16,
            "num_input_channels": 1
        }
    ),
    EncoderTestConfig(
        name="TimesFM-200M",
        encoder_class=TimesFMEncoder,
        model_name="google/timesfm-1.0-200m",
        available=TIMESFM_AVAILABLE,
        test_params={
            "output_dim": ENCODER_OUTPUT_DIM,
            "device": DEVICE,
            "context_length": 512,
            "backend": "gpu" if DEVICE == "cuda" else "cpu"
        }
    ),
    EncoderTestConfig(
        name="MOMENT-Small",
        encoder_class=MomentEncoder,
        model_name="AutonLab/MOMENT-1-small",
        available=MOMENT_AVAILABLE,
        test_params={
            "output_dim": ENCODER_OUTPUT_DIM,
            "device": DEVICE,
            "model_size": "small",
            "context_length": 512
        }
    ),
    EncoderTestConfig(
        name="Lag-Llama",
        encoder_class=LagLlamaEncoder,
        model_name="time-series-foundation-models/Lag-Llama",
        available=LAG_LLAMA_AVAILABLE,
        test_params={
            "output_dim": ENCODER_OUTPUT_DIM,
            "device": DEVICE,
            "context_length": 32
        }
    ),
]


def create_test_data(batch_size: int = 4, seq_len: int = 100, n_features: int = 1) -> torch.Tensor:
    """Create synthetic time series data for testing."""
    # Create a simple sinusoidal pattern with noise
    t = torch.linspace(0, 4 * torch.pi, seq_len).unsqueeze(0).repeat(batch_size, 1)
    data = torch.sin(t) + 0.1 * torch.randn(batch_size, seq_len)

    # Add features dimension
    data = data.unsqueeze(-1).repeat(1, 1, n_features)

    return data.to(DEVICE)


def test_encoder(config: EncoderTestConfig) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Test a single encoder configuration.

    Returns:
        Tuple of (success, message, metrics)
    """
    print(f"\n{'='*60}")
    print(f"Testing {config.name}")
    print(f"Model: {config.model_name}")
    print(f"{'='*60}")

    if not config.available:
        msg = f"Skipped - Required package not available"
        print(f"❌ {msg}")
        return False, msg, {}

    try:
        # Initialize encoder
        print("Initializing encoder...")
        start_time = time.time()

        encoder = config.encoder_class(
            model_name=config.model_name,
            **config.test_params
        )

        init_time = time.time() - start_time
        print(f"✓ Encoder initialized in {init_time:.2f}s")

        # Create test data
        test_data = create_test_data(batch_size=4, seq_len=100, n_features=1)
        print(f"Test data shape: {test_data.shape}")

        # Test forward pass
        print("Testing forward pass...")
        start_time = time.time()

        with torch.no_grad():
            output = encoder(test_data)

        forward_time = time.time() - start_time
        print(f"✓ Forward pass completed in {forward_time:.2f}s")
        print(f"Output shape: {output.shape}")

        # Validate output
        expected_batch_size = test_data.shape[0]
        expected_output_dim = config.test_params.get("output_dim", ENCODER_OUTPUT_DIM)

        assert output.shape[0] == expected_batch_size, \
            f"Batch size mismatch: {output.shape[0]} != {expected_batch_size}"

        assert output.shape[-1] == expected_output_dim, \
            f"Output dimension mismatch: {output.shape[-1]} != {expected_output_dim}"

        # Test with different sequence lengths
        print("\nTesting with different sequence lengths...")
        for seq_len in [50, 200, 512]:
            test_data_var = create_test_data(batch_size=2, seq_len=seq_len)
            with torch.no_grad():
                output_var = encoder(test_data_var)
            print(f"  Seq length {seq_len}: output shape {output_var.shape} ✓")

        # Calculate memory usage (approximate)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
        else:
            memory_mb = 0

        metrics = {
            "init_time": init_time,
            "forward_time": forward_time,
            "memory_mb": memory_mb,
            "output_shape": list(output.shape),
        }

        msg = f"All tests passed successfully!"
        print(f"\n✅ {msg}")
        return True, msg, metrics

    except Exception as e:
        msg = f"Test failed: {str(e)}"
        print(f"\n❌ {msg}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False, msg, {}


def run_all_tests():
    """Run tests for all encoders."""
    print("\n" + "="*80)
    print("UNIFIED ENCODER TEST SUITE")
    print("="*80)

    results = []
    successful = 0
    failed = 0
    skipped = 0

    for config in ENCODER_CONFIGS:
        success, message, metrics = test_encoder(config)

        results.append({
            "encoder": config.name,
            "success": success,
            "message": message,
            "metrics": metrics
        })

        if success:
            successful += 1
        elif "Skipped" in message:
            skipped += 1
        else:
            failed += 1

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for result in results:
        status = "✅" if result["success"] else ("⏭️" if "Skipped" in result["message"] else "❌")
        print(f"{status} {result['encoder']:20s} - {result['message']}")

        if result["metrics"]:
            print(f"    Init: {result['metrics']['init_time']:.2f}s, "
                  f"Forward: {result['metrics']['forward_time']:.3f}s, "
                  f"Memory: {result['metrics'].get('memory_mb', 0):.1f}MB")

    print(f"\nTotal: {successful} passed, {failed} failed, {skipped} skipped")
    return failed == 0


if __name__ == "__main__":
    # Install missing packages if needed
    print("\nChecking dependencies...")

    missing_packages = []
    if not CHRONOS_AVAILABLE:
        missing_packages.append("chronos-forecasting")
    if not PATCHTST_AVAILABLE:
        missing_packages.append("transformers>=4.35.0")
    if not TIMESFM_AVAILABLE:
        missing_packages.append("timesfm")
    if not MOMENT_AVAILABLE:
        missing_packages.append("momentfm")
    if not LAG_LLAMA_AVAILABLE:
        missing_packages.append("lag-llama")

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print(f"To install all dependencies, run:")
        print(f"pip install chronos-forecasting transformers>=4.35.0")
        print(f"# Note: Some packages (timesfm, momentfm, lag-llama) may require special installation")

    # Run tests
    success = run_all_tests()
    sys.exit(0 if success else 1)