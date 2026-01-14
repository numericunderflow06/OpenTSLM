#!/usr/bin/env python3
"""
Comprehensive GPU test for all time series encoders with pretrained models.
"""

import os
import sys
import torch
import time
import traceback
import gc

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*80)
print("COMPREHENSIVE ENCODER TEST ON GPU")
print("="*80)
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("="*80 + "\n")

from opentslm.model_config import ENCODER_OUTPUT_DIM

def create_test_data(batch_size=4, seq_len=512, n_features=1):
    """Create test time series data."""
    # Create sinusoidal data with noise
    t = torch.linspace(0, 4 * torch.pi, seq_len).unsqueeze(0).repeat(batch_size, 1)
    data = torch.sin(t) + 0.1 * torch.randn(batch_size, seq_len)

    # Add features dimension
    if n_features > 1:
        data = data.unsqueeze(-1).repeat(1, 1, n_features)
    else:
        data = data.unsqueeze(-1)

    return data.to(DEVICE)

def test_encoder(name, encoder_class, model_name, config):
    """Test a single encoder."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    try:
        # Initialize encoder
        print("Initializing encoder...")
        start = time.time()

        encoder = encoder_class(
            model_name=model_name,
            **config
        )
        encoder = encoder.to(DEVICE)
        encoder.eval()

        init_time = time.time() - start
        print(f"✓ Initialized in {init_time:.2f}s")

        # Test with different input sizes
        # For PatchTST with ETTh1, we need 7 channels
        if name == "PatchTST":
            test_cases = [
                (2, 100, 7),   # Small batch, short sequence, 7 channels
                (4, 512, 7),   # Medium batch, medium sequence, 7 channels
                (8, 1024, 7),  # Larger batch, long sequence, 7 channels
            ]
        else:
            test_cases = [
                (2, 100, 1),   # Small batch, short sequence
                (4, 512, 1),   # Medium batch, medium sequence
                (8, 1024, 1),  # Larger batch, long sequence
            ]

        for batch_size, seq_len, n_features in test_cases:
            print(f"\nTest case: batch={batch_size}, seq_len={seq_len}, features={n_features}")

            # Create test data
            data = create_test_data(batch_size, seq_len, n_features)
            print(f"  Input shape: {data.shape}")

            # Forward pass
            start = time.time()
            with torch.no_grad():
                if name == "Chronos2":
                    # Chronos2 expects 2D input (batch, seq_len)
                    data_2d = data.squeeze(-1)
                    output = encoder(data_2d)
                else:
                    output = encoder(data)

            forward_time = time.time() - start
            print(f"  Output shape: {output.shape}")
            print(f"  Forward time: {forward_time:.3f}s")

            # Verify output shape
            assert output.shape[0] == batch_size, f"Batch size mismatch"
            if len(output.shape) == 2:
                assert output.shape[1] == ENCODER_OUTPUT_DIM, f"Output dim mismatch"

            print(f"  ✓ Test passed")

        # Memory usage
        if DEVICE == "cuda":
            torch.cuda.synchronize()
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"\nMax GPU memory used: {memory_mb:.2f} MB")
            torch.cuda.empty_cache()

        print(f"\n✅ All tests passed for {name}")
        return True, None

    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error in {name}: {error_msg}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False, error_msg
    finally:
        # Clean up
        if 'encoder' in locals():
            del encoder
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

# Main test execution
def main():
    results = {}

    # Test 1: Chronos2 (from PR #41)
    print("\n" + "="*80)
    print("1. CHRONOS2 ENCODER (from PR #41)")
    print("="*80)

    try:
        # Use the simplified version to avoid version issues
        from opentslm.model.encoder.Chronos2EncoderSimple import Chronos2EncoderSimple

        # Try the tiny model first (smallest)
        success, error = test_encoder(
            "Chronos2",
            Chronos2EncoderSimple,
            "amazon/chronos-t5-tiny",
            {"output_dim": ENCODER_OUTPUT_DIM, "device": DEVICE}
        )
        results["Chronos2"] = (success, error)
    except Exception as e:
        print(f"❌ Could not test Chronos2: {e}")
        results["Chronos2"] = (False, str(e))

    # Test 2: PatchTST
    print("\n" + "="*80)
    print("2. PATCHTST ENCODER")
    print("="*80)

    try:
        from opentslm.model.encoder.PatchTSTEncoder import PatchTSTEncoder

        # Test with weather dataset model (univariate)
        success, error = test_encoder(
            "PatchTST",
            PatchTSTEncoder,
            "ibm/patchtst-etth1-pretrain",
            {
                "output_dim": ENCODER_OUTPUT_DIM,
                "context_length": 512,
                "patch_size": 12,
                "num_input_channels": 7,  # ETTh1 has 7 channels
                "device": DEVICE
            }
        )
        results["PatchTST"] = (success, error)
    except Exception as e:
        print(f"❌ Could not test PatchTST: {e}")
        results["PatchTST"] = (False, str(e))

    # Test 3: TimesFM
    print("\n" + "="*80)
    print("3. TIMESFM ENCODER")
    print("="*80)

    try:
        from opentslm.model.encoder.TimesFMEncoder import TimesFMEncoder

        success, error = test_encoder(
            "TimesFM",
            TimesFMEncoder,
            "google/timesfm-1.0-200m",
            {
                "output_dim": ENCODER_OUTPUT_DIM,
                "context_length": 512,
                "backend": "gpu" if DEVICE == "cuda" else "cpu",
                "device": DEVICE
            }
        )
        results["TimesFM"] = (success, error)
    except Exception as e:
        print(f"❌ Could not test TimesFM: {e}")
        results["TimesFM"] = (False, str(e))

    # Test 4: MOMENT
    print("\n" + "="*80)
    print("4. MOMENT ENCODER")
    print("="*80)

    try:
        from opentslm.model.encoder.MomentEncoder import MomentEncoder

        success, error = test_encoder(
            "MOMENT",
            MomentEncoder,
            "AutonLab/MOMENT-1-small",
            {
                "output_dim": ENCODER_OUTPUT_DIM,
                "model_size": "small",
                "context_length": 512,
                "device": DEVICE
            }
        )
        results["MOMENT"] = (success, error)
    except Exception as e:
        print(f"❌ Could not test MOMENT: {e}")
        results["MOMENT"] = (False, str(e))

    # Test 5: Lag-Llama
    print("\n" + "="*80)
    print("5. LAG-LLAMA ENCODER")
    print("="*80)

    try:
        from opentslm.model.encoder.LagLlamaEncoder import LagLlamaEncoder

        success, error = test_encoder(
            "Lag-Llama",
            LagLlamaEncoder,
            "time-series-foundation-models/Lag-Llama",
            {
                "output_dim": ENCODER_OUTPUT_DIM,
                "context_length": 32,
                "device": DEVICE
            }
        )
        results["Lag-Llama"] = (success, error)
    except Exception as e:
        print(f"❌ Could not test Lag-Llama: {e}")
        results["Lag-Llama"] = (False, str(e))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for encoder_name, (success, error) in results.items():
        if success:
            print(f"✅ {encoder_name:15s} - All tests passed")
        else:
            print(f"❌ {encoder_name:15s} - Failed: {error if error else 'Unknown error'}")

    # Count results
    passed = sum(1 for s, _ in results.values() if s)
    failed = len(results) - passed

    print(f"\nTotal: {passed}/{len(results)} encoders passed")

    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
