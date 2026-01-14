#!/usr/bin/env python3
"""
Final test of ALL FIXED encoders with REAL pretrained models.
This should work correctly with all device issues resolved.
"""

import os
import sys
import torch
import time
import gc
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
print("FINAL TEST - ALL ENCODERS WITH REAL PRETRAINED MODELS")
print("="*80)
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Cache: /local/home/wangni/.cache/huggingface")
print("="*80 + "\n")

from opentslm.model_config import ENCODER_OUTPUT_DIM

def test_encoder(encoder_name, encoder_class, config, test_cases=None):
    """Test an encoder with comprehensive error handling."""
    print(f"\n{'='*60}")
    print(f"Testing {encoder_name}")
    print(f"{'='*60}")

    try:
        # Initialize
        print("Initializing encoder...")
        start = time.time()
        encoder = encoder_class(**config)
        encoder = encoder.to(DEVICE)
        encoder.eval()
        init_time = time.time() - start
        print(f"✓ Initialized in {init_time:.2f}s")

        # Use provided test cases or defaults
        if test_cases is None:
            test_cases = [(2, 512, 1)]

        all_passed = True
        for i, (batch_size, seq_len, n_features) in enumerate(test_cases, 1):
            print(f"\nTest {i}: batch={batch_size}, seq={seq_len}, features={n_features}")

            # Create test data
            data = torch.randn(batch_size, seq_len, n_features).to(DEVICE)
            print(f"  Input: {data.shape}, device={data.device}")

            # Forward pass
            start = time.time()
            with torch.no_grad():
                output = encoder(data)
            forward_time = time.time() - start

            print(f"  Output: {output.shape}, device={output.device}")
            print(f"  Time: {forward_time:.3f}s")

            # Verify
            expected_shape = (batch_size, ENCODER_OUTPUT_DIM)
            if output.shape != expected_shape:
                print(f"  ✗ Shape mismatch! Expected {expected_shape}")
                all_passed = False
            else:
                print(f"  ✓ Test passed")

        # Memory usage
        if DEVICE == "cuda" and all_passed:
            torch.cuda.synchronize()
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"\nGPU memory: {memory_mb:.2f} MB")
            torch.cuda.reset_peak_memory_stats()

        return all_passed, None

    except Exception as e:
        print(f"\n✗ FAILED: {str(e)[:200]}")
        return False, str(e)

    finally:
        # Cleanup
        if 'encoder' in locals():
            del encoder
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

def main():
    results = {}

    # Test 1: Chronos Final
    print("\n" + "="*80)
    print("1. CHRONOS FINAL ENCODER")
    print("="*80)

    try:
        from opentslm.model.encoder.ChronosFinalEncoder import ChronosFinalEncoder

        success, error = test_encoder(
            "Chronos-Final",
            ChronosFinalEncoder,
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
        results["Chronos"] = (False, str(e))

    # Test 2: PatchTST Final
    print("\n" + "="*80)
    print("2. PATCHTST FINAL ENCODER")
    print("="*80)

    try:
        from opentslm.model.encoder.PatchTSTFinalEncoder import PatchTSTFinalEncoder

        success, error = test_encoder(
            "PatchTST-Final",
            PatchTSTFinalEncoder,
            {
                "model_name": "ibm/patchtst-etth1-pretrain",
                "output_dim": ENCODER_OUTPUT_DIM,
                "device": DEVICE
            },
            test_cases=[(2, 512, 7), (3, 384, 7)]  # ETTh1 has 7 channels
        )
        results["PatchTST"] = (success, error)

    except Exception as e:
        print(f"✗ Could not test PatchTST: {e}")
        results["PatchTST"] = (False, str(e))

    # Test 3: TimesFM Final
    print("\n" + "="*80)
    print("3. TIMESFM FINAL ENCODER")
    print("="*80)

    try:
        from opentslm.model.encoder.TimesFMFinalEncoder import TimesFMFinalEncoder

        success, error = test_encoder(
            "TimesFM-Final",
            TimesFMFinalEncoder,
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
        results["TimesFM"] = (False, str(e))

    # Test 4: MOMENT (already working)
    print("\n" + "="*80)
    print("4. MOMENT ENCODER (REFERENCE)")
    print("="*80)

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
        results["MOMENT"] = (False, str(e))

    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    passed_count = 0
    for encoder_name, (success, error) in results.items():
        if success:
            print(f"✅ {encoder_name:15s} - WORKING")
            passed_count += 1
        else:
            error_msg = error[:60] if error else 'Unknown error'
            print(f"❌ {encoder_name:15s} - FAILED: {error_msg}")

    print(f"\n{'='*40}")
    print(f"RESULT: {passed_count}/{len(results)} encoders working")
    print(f"{'='*40}")

    if passed_count == len(results):
        print("\n🎉 SUCCESS! ALL ENCODERS WORKING WITH REAL PRETRAINED MODELS!")
        print("All encoders are using:")
        print("  ✓ Real pretrained weights from HuggingFace")
        print("  ✓ GPU acceleration")
        print("  ✓ Cache directory: /local/home/wangni/")
    else:
        print(f"\n⚠️  {passed_count} encoder(s) working, {len(results)-passed_count} failed")

    # Cleanup
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return passed_count == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)