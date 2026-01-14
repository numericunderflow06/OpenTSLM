#!/usr/bin/env python3
"""
Test all time series encoders with REAL pretrained models.
No fallbacks, no simplifications - using actual pretrained weights.
"""

import os
import sys
import torch
import time
import traceback
import gc
import warnings

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
print("REAL PRETRAINED ENCODER TEST ON GPU")
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
        if "PatchTST" in name and "etth1" in model_name.lower():
            # ETTh1 dataset has 7 channels
            test_cases = [(2, 96, 7)]  # Use exact expected input
        elif "TimesFM" in name:
            # TimesFM has specific requirements
            test_cases = [(2, 512, 1)]
        elif "MOMENT" in name:
            # MOMENT expects specific input lengths
            test_cases = [(2, 512, 1)]
        else:
            test_cases = [(2, 100, 1), (4, 512, 1)]

        for batch_size, seq_len, n_features in test_cases:
            print(f"\nTest case: batch={batch_size}, seq_len={seq_len}, features={n_features}")

            # Create test data
            data = create_test_data(batch_size, seq_len, n_features)
            print(f"  Input shape: {data.shape}")

            # Forward pass
            start = time.time()
            with torch.no_grad():
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

    # Test 1: Chronos with real pretrained model
    print("\n" + "="*80)
    print("1. CHRONOS ENCODER (REAL PRETRAINED)")
    print("="*80)

    try:
        from chronos import ChronosPipeline

        # Test if we can load the model directly
        print("Testing direct ChronosPipeline loading...")
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map=DEVICE,
            cache_dir="/local/home/wangni/.cache/huggingface"
        )

        # Test inference
        test_data = torch.randn(2, 100).to(DEVICE)
        forecast = pipeline.predict(
            context=test_data,
            prediction_length=24,
            num_samples=20
        )
        print(f"✓ ChronosPipeline works! Forecast shape: {forecast.shape}")

        # Now test our encoder wrapper
        from opentslm.model.encoder.ChronosEncoderReal import ChronosEncoderReal

        success, error = test_encoder(
            "ChronosReal",
            ChronosEncoderReal,
            "amazon/chronos-t5-tiny",
            {"output_dim": ENCODER_OUTPUT_DIM, "device": DEVICE}
        )
        results["Chronos"] = (success, error)

    except Exception as e:
        print(f"❌ Could not test Chronos: {e}")
        results["Chronos"] = (False, str(e))

    # Test 2: PatchTST with transformers
    print("\n" + "="*80)
    print("2. PATCHTST ENCODER (REAL PRETRAINED)")
    print("="*80)

    try:
        from transformers import PatchTSTModel

        # Test if we can load the model directly
        print("Testing direct PatchTST loading from transformers...")
        model = PatchTSTModel.from_pretrained(
            "ibm/patchtst-etth1-pretrain",
            cache_dir="/local/home/wangni/.cache/huggingface"
        )
        model = model.to(DEVICE)

        # Test inference with correct input shape
        # ETTh1 has 7 channels, sequence length should be divisible by patch_size
        test_data = torch.randn(2, 96, 7).to(DEVICE)  # 96 is divisible by common patch sizes

        with torch.no_grad():
            outputs = model(past_values=test_data)
            print(f"✓ PatchTST works! Output shape: {outputs.last_hidden_state.shape}")

        # Now test our encoder wrapper
        from opentslm.model.encoder.PatchTSTEncoderReal import PatchTSTEncoderReal

        success, error = test_encoder(
            "PatchTSTReal",
            PatchTSTEncoderReal,
            "ibm/patchtst-etth1-pretrain",
            {
                "output_dim": ENCODER_OUTPUT_DIM,
                "num_input_channels": 7,
                "context_length": 96,
                "patch_length": 16,
                "device": DEVICE
            }
        )
        results["PatchTST"] = (success, error)

    except Exception as e:
        print(f"❌ Could not test PatchTST: {e}")
        results["PatchTST"] = (False, str(e))

    # Test 3: TimesFM
    print("\n" + "="*80)
    print("3. TIMESFM ENCODER (REAL PRETRAINED)")
    print("="*80)

    try:
        import timesfm

        # Test if we can load the model
        print("Testing TimesFM loading...")
        tfm = timesfm.TimesFm(
            context_len=512,
            horizon_len=96,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="gpu" if DEVICE == "cuda" else "cpu"
        )

        # Load checkpoint
        checkpoint_path = "/local/home/wangni/.cache/timesfm"
        os.makedirs(checkpoint_path, exist_ok=True)

        tfm.load_from_checkpoint(
            repo_id="google/timesfm-1.0-200m",
            cache_dir=checkpoint_path
        )

        # Test inference
        test_data = torch.randn(2, 512).numpy()
        forecast_point, _ = tfm.forecast(test_data, freq=[0, 0])
        print(f"✓ TimesFM works! Forecast shape: {forecast_point.shape}")

        # Now test our encoder wrapper
        from opentslm.model.encoder.TimesFMEncoderReal import TimesFMEncoderReal

        success, error = test_encoder(
            "TimesFMReal",
            TimesFMEncoderReal,
            "google/timesfm-1.0-200m",
            {
                "output_dim": ENCODER_OUTPUT_DIM,
                "context_length": 512,
                "cache_dir": checkpoint_path,
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
    print("4. MOMENT ENCODER (REAL PRETRAINED)")
    print("="*80)

    try:
        from momentfm import MOMENTPipeline

        # Test if we can load the model
        print("Testing MOMENT loading...")
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                'task_name': 'embedding',
                'n_channels': 1,
                'num_class': None,
                'freeze_encoder': False,
                'freeze_embedder': False
            },
            cache_dir="/local/home/wangni/.cache/huggingface"
        )
        model = model.to(DEVICE)

        # Test inference
        test_data = torch.randn(2, 1, 512).to(DEVICE)
        with torch.no_grad():
            output = model(test_data)
            print(f"✓ MOMENT works! Output shape: {output.embeddings.shape}")

        # Now test our encoder wrapper
        from opentslm.model.encoder.MOMENTEncoderReal import MOMENTEncoderReal

        success, error = test_encoder(
            "MOMENTReal",
            MOMENTEncoderReal,
            "AutonLab/MOMENT-1-large",
            {
                "output_dim": ENCODER_OUTPUT_DIM,
                "context_length": 512,
                "device": DEVICE
            }
        )
        results["MOMENT"] = (success, error)

    except Exception as e:
        print(f"❌ Could not test MOMENT: {e}")
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

    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)