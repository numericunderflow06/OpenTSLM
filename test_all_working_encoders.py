#!/usr/bin/env python3
"""
Test all WORKING encoders with REAL pretrained models.
All bugs have been fixed.
"""

import os
import sys
import torch
import time
import gc
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Set cache directories to /local/home/wangni
os.environ["HF_HOME"] = "/local/home/wangni/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/local/home/wangni/.cache/huggingface/transformers"
os.environ["TORCH_HOME"] = "/local/home/wangni/.cache/torch"
os.environ["XDG_CACHE_HOME"] = "/local/home/wangni/.cache"

# Ensure directories exist
for cache_dir in [os.environ["HF_HOME"], os.environ["TRANSFORMERS_CACHE"], os.environ["TORCH_HOME"]]:
    os.makedirs(cache_dir, exist_ok=True)

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*80)
print("TESTING ALL WORKING ENCODERS WITH REAL PRETRAINED MODELS")
print("="*80)
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Cache: /local/home/wangni/.cache/huggingface")
print("="*80 + "\n")

from opentslm.model_config import ENCODER_OUTPUT_DIM

def test_encoder(encoder_name, encoder_class, config, test_cases=None):
    """Test an encoder with proper error handling."""
    print(f"\nTesting {encoder_name}...")
    try:
        start = time.time()
        encoder = encoder_class(**config)
        encoder = encoder.to(DEVICE)
        encoder.eval()
        init_time = time.time() - start
        print(f"  ✓ Initialized in {init_time:.2f}s")

        # Use provided test cases or defaults
        if test_cases is None:
            test_cases = [(2, 512, 1), (4, 256, 1)]

        all_passed = True
        for batch_size, seq_len, n_features in test_cases:
            # Create test data
            data = torch.randn(batch_size, seq_len, n_features).to(DEVICE)
            print(f"\n  Test case: batch={batch_size}, seq={seq_len}, features={n_features}")
            print(f"    Input shape: {data.shape}")

            # Forward pass
            start = time.time()
            with torch.no_grad():
                output = encoder(data)
            forward_time = time.time() - start

            print(f"    Output shape: {output.shape}")
            print(f"    Forward time: {forward_time:.3f}s")

            # Verify output
            if output.shape != (batch_size, ENCODER_OUTPUT_DIM):
                print(f"    ✗ Output shape mismatch! Expected ({batch_size}, {ENCODER_OUTPUT_DIM})")
                all_passed = False
            else:
                print(f"    ✓ Test passed")

        # Memory usage
        if DEVICE == "cuda":
            torch.cuda.synchronize()
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"\n  GPU memory used: {memory_mb:.2f} MB")
            torch.cuda.empty_cache()

        return all_passed, None

    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
        return False, str(e)
    finally:
        if 'encoder' in locals():
            del encoder
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

def main():
    results = {}

    # Test 1: Chronos Working Encoder
    print("\n" + "="*70)
    print("1. CHRONOS WORKING ENCODER")
    print("="*70)

    try:
        from opentslm.model.encoder.ChronosWorkingEncoder import ChronosWorkingEncoder

        success, error = test_encoder(
            "ChronosWorking",
            ChronosWorkingEncoder,
            {
                "model_name": "amazon/chronos-t5-tiny",
                "output_dim": ENCODER_OUTPUT_DIM,
                "device": DEVICE
            },
            test_cases=[(2, 100, 1), (3, 512, 1)]
        )
        results["Chronos"] = (success, error)

    except Exception as e:
        print(f"✗ Could not test Chronos: {e}")
        import traceback
        traceback.print_exc()
        results["Chronos"] = (False, str(e))

    # Test 2: PatchTST Working Encoder
    print("\n" + "="*70)
    print("2. PATCHTST WORKING ENCODER")
    print("="*70)

    try:
        from opentslm.model.encoder.PatchTSTWorkingEncoder import PatchTSTWorkingEncoder

        # PatchTST needs specific dimensions (7 channels for ETTh1)
        success, error = test_encoder(
            "PatchTSTWorking",
            PatchTSTWorkingEncoder,
            {
                "model_name": "ibm/patchtst-etth1-pretrain",
                "output_dim": ENCODER_OUTPUT_DIM,
                "device": DEVICE
            },
            test_cases=[(2, 512, 7), (3, 256, 7)]
        )
        results["PatchTST"] = (success, error)

    except Exception as e:
        print(f"✗ Could not test PatchTST: {e}")
        import traceback
        traceback.print_exc()
        results["PatchTST"] = (False, str(e))

    # Test 3: TimesFM Working Encoder
    print("\n" + "="*70)
    print("3. TIMESFM WORKING ENCODER")
    print("="*70)

    try:
        from opentslm.model.encoder.TimesFMWorkingEncoder import TimesFMWorkingEncoder

        success, error = test_encoder(
            "TimesFMWorking",
            TimesFMWorkingEncoder,
            {
                "model_name": "google/timesfm-2p5-200m-torch",
                "output_dim": ENCODER_OUTPUT_DIM,
                "context_length": 512,
                "device": DEVICE
            },
            test_cases=[(2, 512, 1), (3, 256, 1)]
        )
        results["TimesFM"] = (success, error)

    except Exception as e:
        print(f"✗ Could not test TimesFM: {e}")
        import traceback
        traceback.print_exc()
        results["TimesFM"] = (False, str(e))

    # Test 4: MOMENT Working Encoder (already working)
    print("\n" + "="*70)
    print("4. MOMENT ENCODER (ALREADY WORKING)")
    print("="*70)

    try:
        from opentslm.model.encoder.MOMENTEncoderReal import MOMENTEncoderReal

        success, error = test_encoder(
            "MOMENT",
            MOMENTEncoderReal,
            {
                "model_name": "AutonLab/MOMENT-1-large",
                "output_dim": ENCODER_OUTPUT_DIM,
                "device": DEVICE
            },
            test_cases=[(2, 512, 1), (3, 256, 1)]
        )
        results["MOMENT"] = (success, error)

    except Exception as e:
        print(f"✗ Could not test MOMENT: {e}")
        import traceback
        traceback.print_exc()
        results["MOMENT"] = (False, str(e))

    # Print summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)

    for encoder_name, (success, error) in results.items():
        if success:
            print(f"✅ {encoder_name:15s} - ALL TESTS PASSED")
        else:
            error_msg = error[:80] if error else 'Unknown error'
            print(f"❌ {encoder_name:15s} - FAILED: {error_msg}")

    passed = sum(1 for s, _ in results.values() if s)
    failed = len(results) - passed

    print(f"\n{'='*40}")
    print(f"TOTAL: {passed}/{len(results)} encoders working")
    print(f"{'='*40}")

    if passed == len(results):
        print("\n🎉 SUCCESS! ALL ENCODERS WORKING WITH REAL PRETRAINED MODELS!")
    elif passed > 0:
        print(f"\n⚠️  {passed} encoder(s) working, {failed} still need fixes")
    else:
        print("\n❌ All encoders failed - check error messages above")

    # Clean up
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)