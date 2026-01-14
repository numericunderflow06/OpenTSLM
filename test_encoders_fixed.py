#!/usr/bin/env python3
"""
Fixed test for all time series encoders with REAL pretrained models.
Uses correct APIs for each model.
"""

import os
import sys
import torch
import time
import traceback
import gc
import warnings
import numpy as np

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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*80)
print("FIXED REAL PRETRAINED ENCODER TEST ON GPU")
print("="*80)
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"HuggingFace cache: {os.environ['HF_HOME']}")
print(f"Transformers cache: {os.environ['TRANSFORMERS_CACHE']}")
print(f"Torch cache: {os.environ['TORCH_HOME']}")
print("="*80 + "\n")

from opentslm.model_config import ENCODER_OUTPUT_DIM

def create_test_data(batch_size=4, seq_len=512, n_features=1):
    """Create test time series data."""
    # Create sinusoidal data with noise
    t = torch.linspace(0, 4 * torch.pi, seq_len).unsqueeze(0).repeat(batch_size, 1)
    data = torch.sin(t) + 0.1 * torch.randn(batch_size, seq_len)

    # Add features dimension if needed
    if n_features > 1:
        data = data.unsqueeze(-1).repeat(1, 1, n_features)
    else:
        data = data.unsqueeze(-1)

    return data.to(DEVICE)

# Main test execution
def main():
    results = {}

    # Test 1: Chronos with correct API
    print("\n" + "="*80)
    print("1. CHRONOS ENCODER (REAL PRETRAINED - FIXED)")
    print("="*80)

    try:
        from chronos import ChronosPipeline

        print("Testing ChronosPipeline with correct API...")
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map=DEVICE,
            cache_dir="/local/home/wangni/.cache/huggingface",
            torch_dtype=torch.float32
        )

        # Test inference with correct method
        # ChronosPipeline uses context_values not context
        test_data = torch.randn(2, 100).to(DEVICE)

        # The predict method uses different parameters
        forecast, _ = pipeline.predict(
            context=test_data,
            num_samples=20,
            prediction_length=24
        )
        print(f"✓ ChronosPipeline works! Forecast shape: {forecast.shape}")

        # Now test our encoder wrapper
        from opentslm.model.encoder.ChronosEncoderReal import ChronosEncoderReal

        print("\nTesting ChronosEncoderReal wrapper...")
        encoder = ChronosEncoderReal(
            model_name="amazon/chronos-t5-tiny",
            output_dim=ENCODER_OUTPUT_DIM,
            device=DEVICE
        )
        encoder = encoder.to(DEVICE)
        encoder.eval()

        # Test with different input sizes
        test_cases = [(2, 100), (4, 512)]

        for batch_size, seq_len in test_cases:
            print(f"\nTest case: batch={batch_size}, seq_len={seq_len}")
            data = torch.randn(batch_size, seq_len).to(DEVICE)
            print(f"  Input shape: {data.shape}")

            with torch.no_grad():
                output = encoder(data)

            print(f"  Output shape: {output.shape}")
            assert output.shape == (batch_size, ENCODER_OUTPUT_DIM)
            print(f"  ✓ Test passed")

        results["Chronos"] = (True, None)

    except Exception as e:
        print(f"❌ Could not test Chronos: {e}")
        traceback.print_exc()
        results["Chronos"] = (False, str(e))

    # Test 2: PatchTST with correct sequence length
    print("\n" + "="*80)
    print("2. PATCHTST ENCODER (REAL PRETRAINED - FIXED)")
    print("="*80)

    try:
        from transformers import PatchTSTModel

        print("Testing PatchTST with correct config...")

        # First, load the model to check its config
        model = PatchTSTModel.from_pretrained(
            "ibm/patchtst-etth1-pretrain",
            cache_dir="/local/home/wangni/.cache/huggingface"
        )

        print(f"Model config:")
        print(f"  Context length: {model.config.context_length}")
        print(f"  Num input channels: {model.config.num_input_channels}")
        print(f"  Patch length: {model.config.patch_length}")

        model = model.to(DEVICE)

        # Test with exact expected input
        test_data = torch.randn(2, model.config.context_length, model.config.num_input_channels).to(DEVICE)
        print(f"Testing with shape: {test_data.shape}")

        with torch.no_grad():
            outputs = model(past_values=test_data.transpose(1, 2))  # PatchTST expects (batch, channels, seq)
            print(f"✓ PatchTST works! Output shape: {outputs.last_hidden_state.shape}")

        # Test our wrapper with fixed dimensions
        from opentslm.model.encoder.PatchTSTEncoderReal import PatchTSTEncoderReal

        print("\nTesting PatchTSTEncoderReal wrapper...")
        encoder = PatchTSTEncoderReal(
            model_name="ibm/patchtst-etth1-pretrain",
            output_dim=ENCODER_OUTPUT_DIM,
            device=DEVICE
        )
        encoder = encoder.to(DEVICE)
        encoder.eval()

        # Test with correct input dimensions
        batch_size = 2
        data = torch.randn(batch_size, model.config.context_length, model.config.num_input_channels).to(DEVICE)
        print(f"Input shape: {data.shape}")

        with torch.no_grad():
            output = encoder(data)

        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size, ENCODER_OUTPUT_DIM)
        print("✓ Test passed")

        results["PatchTST"] = (True, None)

    except Exception as e:
        print(f"❌ Could not test PatchTST: {e}")
        traceback.print_exc()
        results["PatchTST"] = (False, str(e))

    # Test 3: TimesFM with correct API
    print("\n" + "="*80)
    print("3. TIMESFM ENCODER (REAL PRETRAINED - FIXED)")
    print("="*80)

    try:
        import timesfm

        print("Testing TimesFM with correct API...")

        # Use the correct initialization from timesfm package
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                context_len=512,
                horizon_len=96,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
                backend="gpu" if DEVICE == "cuda" else "cpu"
            )
        )

        # Download and load checkpoint
        checkpoint_path = "/local/home/wangni/.cache/timesfm"
        os.makedirs(checkpoint_path, exist_ok=True)

        tfm.load_from_checkpoint(
            repo_id="google/timesfm-1.0-200m",
            cache_dir=checkpoint_path
        )

        # Test inference
        test_data = np.random.randn(2, 512)
        forecast_point, _ = tfm.forecast(test_data, freq=[0, 0])
        print(f"✓ TimesFM works! Forecast shape: {forecast_point.shape}")

        # Test our wrapper
        from opentslm.model.encoder.TimesFMEncoderReal import TimesFMEncoderReal

        print("\nTesting TimesFMEncoderReal wrapper...")
        encoder = TimesFMEncoderReal(
            model_name="google/timesfm-1.0-200m",
            output_dim=ENCODER_OUTPUT_DIM,
            context_length=512,
            cache_dir=checkpoint_path,
            backend="gpu" if DEVICE == "cuda" else "cpu",
            device=DEVICE
        )
        encoder = encoder.to(DEVICE)
        encoder.eval()

        batch_size = 2
        data = torch.randn(batch_size, 512, 1).to(DEVICE)
        print(f"Input shape: {data.shape}")

        with torch.no_grad():
            output = encoder(data)

        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size, ENCODER_OUTPUT_DIM)
        print("✓ Test passed")

        results["TimesFM"] = (True, None)

    except Exception as e:
        print(f"❌ Could not test TimesFM: {e}")
        traceback.print_exc()
        results["TimesFM"] = (False, str(e))

    # Test 4: MOMENT with correct API
    print("\n" + "="*80)
    print("4. MOMENT ENCODER (REAL PRETRAINED - FIXED)")
    print("="*80)

    try:
        sys.path.append('/tmp/moment')
        from momentfm import MOMENTPipeline

        print("Testing MOMENT with correct API...")
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                'task_name': 'embedding',
                'n_channels': 1,
            },
            cache_dir="/local/home/wangni/.cache/huggingface"
        )
        model.init()
        model = model.to(DEVICE)

        # Test inference with correct API
        test_data = torch.randn(2, 1, 512).to(DEVICE)  # (batch, channels, length)
        with torch.no_grad():
            output = model(x_enc=test_data)
            if hasattr(output, 'embeddings'):
                print(f"✓ MOMENT works! Output embeddings shape: {output.embeddings.shape}")
            else:
                print(f"✓ MOMENT works! Output type: {type(output)}")

        # Test our wrapper
        from opentslm.model.encoder.MOMENTEncoderReal import MOMENTEncoderReal

        print("\nTesting MOMENTEncoderReal wrapper...")
        encoder = MOMENTEncoderReal(
            model_name="AutonLab/MOMENT-1-large",
            output_dim=ENCODER_OUTPUT_DIM,
            context_length=512,
            device=DEVICE
        )
        encoder = encoder.to(DEVICE)
        encoder.eval()

        batch_size = 2
        data = torch.randn(batch_size, 512, 1).to(DEVICE)
        print(f"Input shape: {data.shape}")

        with torch.no_grad():
            output = encoder(data)

        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size, ENCODER_OUTPUT_DIM)
        print("✓ Test passed")

        results["MOMENT"] = (True, None)

    except Exception as e:
        print(f"❌ Could not test MOMENT: {e}")
        traceback.print_exc()
        results["MOMENT"] = (False, str(e))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for encoder_name, (success, error) in results.items():
        if success:
            print(f"✅ {encoder_name:15s} - All tests passed")
        else:
            print(f"❌ {encoder_name:15s} - Failed: {error[:50] if error else 'Unknown error'}")

    # Count results
    passed = sum(1 for s, _ in results.values() if s)
    failed = len(results) - passed

    print(f"\nTotal: {passed}/{len(results)} encoders passed")

    # Memory cleanup
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)