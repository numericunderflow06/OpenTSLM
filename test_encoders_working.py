#!/usr/bin/env python3
"""
Final working test for time series encoders with REAL pretrained models.
All issues fixed, using proper APIs for each library.
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
print("WORKING PRETRAINED ENCODER TEST")
print("="*80)
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Cache dir: /local/home/wangni/.cache/huggingface")
print("="*80 + "\n")

from opentslm.model_config import ENCODER_OUTPUT_DIM

def test_encoder(encoder_name, encoder_class, config):
    """Test an encoder with proper error handling."""
    print(f"\nTesting {encoder_name}...")
    try:
        start = time.time()
        encoder = encoder_class(**config)
        encoder = encoder.to(DEVICE)
        encoder.eval()
        init_time = time.time() - start
        print(f"  ✓ Initialized in {init_time:.2f}s")

        # Create test data based on encoder requirements
        if "PatchTST" in encoder_name:
            # PatchTST needs specific dimensions
            batch_size = 2
            seq_len = encoder.context_length if hasattr(encoder, 'context_length') else 512
            n_features = encoder.num_input_channels if hasattr(encoder, 'num_input_channels') else 7
        else:
            batch_size = 2
            seq_len = 512
            n_features = 1

        # Create test data
        data = torch.randn(batch_size, seq_len, n_features).to(DEVICE)
        print(f"  Input shape: {data.shape}")

        # Forward pass
        start = time.time()
        with torch.no_grad():
            output = encoder(data)
        forward_time = time.time() - start

        print(f"  Output shape: {output.shape}")
        print(f"  Forward time: {forward_time:.3f}s")

        # Verify output
        assert output.shape == (batch_size, ENCODER_OUTPUT_DIM), f"Output shape mismatch"
        print(f"  ✓ Test passed")

        # Memory usage
        if DEVICE == "cuda":
            torch.cuda.synchronize()
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"  GPU memory: {memory_mb:.2f} MB")
            torch.cuda.empty_cache()

        return True, None

    except Exception as e:
        print(f"  ✗ Failed: {str(e)[:100]}")
        return False, str(e)
    finally:
        if 'encoder' in locals():
            del encoder
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

def main():
    results = {}

    # Test 1: Chronos with fixed device handling
    print("\n" + "="*70)
    print("1. CHRONOS ENCODER (FIXED)")
    print("="*70)

    try:
        from opentslm.model.encoder.ChronosEncoderFixed import ChronosEncoderFixed

        success, error = test_encoder(
            "ChronosFixed",
            ChronosEncoderFixed,
            {
                "model_name": "amazon/chronos-t5-tiny",
                "output_dim": ENCODER_OUTPUT_DIM,
                "device": DEVICE
            }
        )
        results["Chronos"] = (success, error)

    except Exception as e:
        print(f"✗ Could not test Chronos: {e}")
        results["Chronos"] = (False, str(e))

    # Test 2: PatchTST with dimension fix
    print("\n" + "="*70)
    print("2. PATCHTST ENCODER (FIXED)")
    print("="*70)

    try:
        from opentslm.model.encoder.PatchTSTEncoderFixed import PatchTSTEncoderFixed

        success, error = test_encoder(
            "PatchTSTFixed",
            PatchTSTEncoderFixed,
            {
                "model_name": "ibm/patchtst-etth1-pretrain",
                "output_dim": ENCODER_OUTPUT_DIM,
                "device": DEVICE
            }
        )
        results["PatchTST"] = (success, error)

    except Exception as e:
        print(f"✗ Could not test PatchTST: {e}")
        results["PatchTST"] = (False, str(e))

    # Test 3: TimesFM with correct API
    print("\n" + "="*70)
    print("3. TIMESFM ENCODER")
    print("="*70)

    try:
        from timesfm import TimesFM_2p5_200M_torch, ForecastConfig

        # Create a simple wrapper
        class TimesFMWrapper(torch.nn.Module):
            def __init__(self, output_dim=ENCODER_OUTPUT_DIM, device="cuda"):
                super().__init__()
                self.device = device
                self.model = TimesFM_2p5_200M_torch(device=device)
                self.config = ForecastConfig(max_context=512, max_horizon=96)
                self.model.compile(self.config)
                self.projection = torch.nn.Linear(1280, output_dim).to(device)

            def forward(self, x):
                batch_size = x.shape[0]
                if len(x.shape) == 3:
                    x = x[:, :, 0]

                with torch.no_grad():
                    p = 32
                    seq_len = x.shape[1]
                    pad_len = (p - (seq_len % p)) % p
                    if pad_len > 0:
                        padding = torch.zeros(batch_size, pad_len, device=self.device)
                        x = torch.cat([padding, x], dim=1)

                    masks = torch.zeros_like(x, dtype=torch.bool)
                    if pad_len > 0:
                        masks[:, :pad_len] = True

                    patched = x.reshape(batch_size, -1, p)
                    patched_masks = masks.reshape(batch_size, -1, p)

                    outputs, _ = self.model.model(patched, patched_masks)
                    _, output_embeddings, _, _ = outputs

                    if len(output_embeddings.shape) == 3:
                        embeddings = output_embeddings.mean(dim=1)
                    else:
                        embeddings = output_embeddings

                return self.projection(embeddings)

        success, error = test_encoder(
            "TimesFM",
            TimesFMWrapper,
            {"output_dim": ENCODER_OUTPUT_DIM, "device": DEVICE}
        )
        results["TimesFM"] = (success, error)

    except Exception as e:
        print(f"✗ Could not test TimesFM: {e}")
        results["TimesFM"] = (False, str(e))

    # Test 4: MOMENT (already working)
    print("\n" + "="*70)
    print("4. MOMENT ENCODER")
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
            }
        )
        results["MOMENT"] = (success, error)

    except Exception as e:
        print(f"✗ Could not test MOMENT: {e}")
        results["MOMENT"] = (False, str(e))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for encoder_name, (success, error) in results.items():
        if success:
            print(f"✅ {encoder_name:15s} - PASSED")
        else:
            error_msg = error[:60] if error else 'Unknown error'
            print(f"❌ {encoder_name:15s} - FAILED: {error_msg}")

    passed = sum(1 for s, _ in results.values() if s)
    failed = len(results) - passed

    print(f"\n{'='*40}")
    print(f"TOTAL: {passed}/{len(results)} encoders passed")
    print(f"{'='*40}")

    if passed == len(results):
        print("\n✅ ALL ENCODERS WORKING WITH PRETRAINED MODELS!")
    elif passed > 0:
        print(f"\n⚠️  {passed} encoder(s) working, {failed} need fixes")
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