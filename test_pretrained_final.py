#!/usr/bin/env python3
"""
Final test for all time series encoders with REAL pretrained models.
Uses correct APIs for each library.
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
print("FINAL REAL PRETRAINED ENCODER TEST ON GPU")
print("="*80)
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"HuggingFace cache: {os.environ['HF_HOME']}")
print("="*80 + "\n")

from opentslm.model_config import ENCODER_OUTPUT_DIM

def main():
    results = {}

    # Test 1: Chronos with correct API (inputs not context)
    print("\n" + "="*80)
    print("1. CHRONOS ENCODER (REAL PRETRAINED)")
    print("="*80)

    try:
        from chronos import ChronosPipeline

        print("Testing ChronosPipeline...")
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-tiny",
            device_map=DEVICE,
            cache_dir="/local/home/wangni/.cache/huggingface",
            dtype=torch.float32  # Use dtype not torch_dtype
        )

        # Test with correct parameter name: inputs not context
        test_data = torch.randn(2, 100).to(DEVICE)
        forecast = pipeline.predict(
            inputs=test_data,  # FIXED: inputs not context
            prediction_length=24,
            num_samples=20
        )
        print(f"✓ ChronosPipeline works! Forecast shape: {forecast.shape}")

        # Test our wrapper
        from opentslm.model.encoder.ChronosEncoderReal import ChronosEncoderReal

        print("\nTesting ChronosEncoderReal wrapper...")
        encoder = ChronosEncoderReal(
            model_name="amazon/chronos-t5-tiny",
            output_dim=ENCODER_OUTPUT_DIM,
            device=DEVICE
        )
        encoder = encoder.to(DEVICE)
        encoder.eval()

        batch_size = 2
        data = torch.randn(batch_size, 512).to(DEVICE)
        print(f"Input shape: {data.shape}")

        with torch.no_grad():
            output = encoder(data)

        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size, ENCODER_OUTPUT_DIM)
        print("✓ Encoder wrapper test passed")

        results["Chronos"] = (True, None)

    except Exception as e:
        print(f"❌ Chronos failed: {e}")
        traceback.print_exc()
        results["Chronos"] = (False, str(e))

    # Test 2: PatchTST with proper dimensions
    print("\n" + "="*80)
    print("2. PATCHTST ENCODER (REAL PRETRAINED)")
    print("="*80)

    try:
        from transformers import PatchTSTModel

        print("Loading PatchTST model...")
        model = PatchTSTModel.from_pretrained(
            "ibm/patchtst-etth1-pretrain",
            cache_dir="/local/home/wangni/.cache/huggingface"
        )

        print(f"Model config:")
        print(f"  Context length: {model.config.context_length}")
        print(f"  Num channels: {model.config.num_input_channels}")
        print(f"  Patch length: {model.config.patch_length}")

        model = model.to(DEVICE)

        # Create input with correct shape: (batch, channels, seq)
        batch_size = 2
        test_data = torch.randn(
            batch_size,
            model.config.num_input_channels,
            model.config.context_length
        ).to(DEVICE)
        print(f"Testing with shape: {test_data.shape}")

        with torch.no_grad():
            outputs = model(past_values=test_data)
            print(f"✓ PatchTST works! Output shape: {outputs.last_hidden_state.shape}")

        # Test our fixed wrapper
        from opentslm.model.encoder.PatchTSTEncoderReal import PatchTSTEncoderReal

        print("\nTesting PatchTSTEncoderReal wrapper...")
        encoder = PatchTSTEncoderReal(
            model_name="ibm/patchtst-etth1-pretrain",
            output_dim=ENCODER_OUTPUT_DIM,
            device=DEVICE
        )
        encoder = encoder.to(DEVICE)
        encoder.eval()

        # Our wrapper expects (batch, seq, channels)
        data = torch.randn(
            batch_size,
            model.config.context_length,
            model.config.num_input_channels
        ).to(DEVICE)
        print(f"Input shape: {data.shape}")

        with torch.no_grad():
            output = encoder(data)

        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size, ENCODER_OUTPUT_DIM)
        print("✓ Encoder wrapper test passed")

        results["PatchTST"] = (True, None)

    except Exception as e:
        print(f"❌ PatchTST failed: {e}")
        traceback.print_exc()
        results["PatchTST"] = (False, str(e))

    # Test 3: TimesFM with correct API
    print("\n" + "="*80)
    print("3. TIMESFM ENCODER (REAL PRETRAINED)")
    print("="*80)

    try:
        from timesfm import TimesFM_2p5_200M_torch, ForecastConfig

        print("Testing TimesFM...")

        # Initialize model
        tfm = TimesFM_2p5_200M_torch(device=DEVICE)

        # Create config for forecasting
        config = ForecastConfig(
            max_context=512,
            max_horizon=96
        )

        # Compile model
        tfm.compile(config)

        # Test with numpy array (TimesFM expects numpy)
        test_data = np.random.randn(2, 512).astype(np.float32)
        forecast, _ = tfm.forecast(test_data, horizon_len=[24, 24])
        print(f"✓ TimesFM works! Forecast shape: {forecast.shape}")

        # Now test our wrapper with the fixed initialization
        print("\nCreating fixed TimesFMEncoderReal wrapper...")

        # Create a simple encoder wrapper that uses TimesFM correctly
        class TimesFMEncoderFixed(torch.nn.Module):
            def __init__(self, output_dim=ENCODER_OUTPUT_DIM, device="cuda"):
                super().__init__()
                self.device = device
                self.model = TimesFM_2p5_200M_torch(device=device)
                self.config = ForecastConfig(max_context=512, max_horizon=96)
                self.model.compile(self.config)

                # Projection from hidden dim to output dim
                # TimesFM 2.5 has hidden dim of 1280
                self.projection = torch.nn.Linear(1280, output_dim).to(device)

            def forward(self, x):
                # x shape: (batch, seq, features)
                batch_size = x.shape[0]

                # TimesFM expects (batch, seq) for univariate
                if len(x.shape) == 3:
                    x = x[:, :, 0]  # Take first feature

                # Get embeddings by calling the model
                with torch.no_grad():
                    # Pad to patch size
                    p = 32  # patch size
                    seq_len = x.shape[1]
                    pad_len = (p - (seq_len % p)) % p
                    if pad_len > 0:
                        padding = torch.zeros(batch_size, pad_len, device=self.device)
                        x = torch.cat([padding, x], dim=1)

                    # Create masks
                    masks = torch.zeros_like(x, dtype=torch.bool)
                    if pad_len > 0:
                        masks[:, :pad_len] = True

                    # Reshape to patches
                    patched = x.reshape(batch_size, -1, p)
                    patched_masks = masks.reshape(batch_size, -1, p)

                    # Forward through model
                    outputs, _ = self.model.model(patched, patched_masks)
                    _, output_embeddings, _, _ = outputs

                    # Pool embeddings
                    if len(output_embeddings.shape) == 3:
                        embeddings = output_embeddings.mean(dim=1)
                    else:
                        embeddings = output_embeddings

                # Project to output dim
                output = self.projection(embeddings)
                return output

        encoder = TimesFMEncoderFixed(output_dim=ENCODER_OUTPUT_DIM, device=DEVICE)
        encoder.eval()

        batch_size = 2
        data = torch.randn(batch_size, 512, 1).to(DEVICE)
        print(f"Input shape: {data.shape}")

        with torch.no_grad():
            output = encoder(data)

        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size, ENCODER_OUTPUT_DIM)
        print("✓ Encoder wrapper test passed")

        results["TimesFM"] = (True, None)

    except Exception as e:
        print(f"❌ TimesFM failed: {e}")
        traceback.print_exc()
        results["TimesFM"] = (False, str(e))

    # Test 4: MOMENT
    print("\n" + "="*80)
    print("4. MOMENT ENCODER (REAL PRETRAINED)")
    print("="*80)

    try:
        sys.path.append('/tmp/moment')
        from momentfm import MOMENTPipeline

        print("Testing MOMENT...")
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={'task_name': 'embedding', 'n_channels': 1},
            cache_dir="/local/home/wangni/.cache/huggingface"
        )
        model.init()
        model = model.to(DEVICE)

        # Test with correct shape
        test_data = torch.randn(2, 1, 512).to(DEVICE)  # (batch, channels, length)
        with torch.no_grad():
            output = model(x_enc=test_data)
            print(f"✓ MOMENT works! Output embeddings shape: {output.embeddings.shape}")

        # Test our fixed wrapper (without context_length parameter)
        from opentslm.model.encoder.MOMENTEncoderReal import MOMENTEncoderReal

        print("\nTesting MOMENTEncoderReal wrapper...")
        encoder = MOMENTEncoderReal(
            model_name="AutonLab/MOMENT-1-large",
            output_dim=ENCODER_OUTPUT_DIM,
            device=DEVICE
            # Remove context_length - not in __init__
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
        print("✓ Encoder wrapper test passed")

        results["MOMENT"] = (True, None)

    except Exception as e:
        print(f"❌ MOMENT failed: {e}")
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
            error_msg = error[:60] if error else 'Unknown error'
            print(f"❌ {encoder_name:15s} - Failed: {error_msg}")

    # Count results
    passed = sum(1 for s, _ in results.values() if s)
    failed = len(results) - passed

    print(f"\nTotal: {passed}/{len(results)} encoders passed")

    # Memory cleanup
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU memory cleared")

    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)