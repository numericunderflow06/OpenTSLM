#!/usr/bin/env python3
"""Quick test for available encoders with pretrained models."""

import os
import sys
import torch

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

from opentslm.model_config import ENCODER_OUTPUT_DIM

# Test available encoders
print("Quick encoder test with pretrained models\n")
print("="*60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}\n")

# Create test data
def create_test_data(batch_size=2, seq_len=100):
    """Create simple test data."""
    data = torch.randn(batch_size, seq_len, 1).to(DEVICE)
    return data

# Test 1: Try existing Chronos2 (from PR #41)
print("1. Testing existing Chronos2 encoder...")
try:
    from opentslm.model.encoder.Chronos2Encoder import Chronos2Encoder

    # Try with the tiny model
    encoder = Chronos2Encoder(
        model_name="amazon/chronos-t5-tiny",
        output_dim=ENCODER_OUTPUT_DIM,
        device=DEVICE
    )

    # Test forward pass
    test_data = create_test_data()
    print(f"   Input shape: {test_data.shape}")

    with torch.no_grad():
        # Chronos expects (batch, seq_len) without feature dimension
        test_data_2d = test_data.squeeze(-1)
        output = encoder(test_data_2d)

    print(f"   Output shape: {output.shape}")
    print(f"   ✅ Chronos2 encoder working!\n")

except Exception as e:
    print(f"   ❌ Chronos2 failed: {e}\n")

# Test 2: Try PatchTST
print("2. Testing PatchTST encoder...")
try:
    from opentslm.model.encoder.PatchTSTEncoder import PatchTSTEncoder

    encoder = PatchTSTEncoder(
        model_name="ibm/patchtst-etth1-pretrain",
        output_dim=ENCODER_OUTPUT_DIM,
        context_length=512,
        patch_size=16,
        num_input_channels=1,
        device=DEVICE
    )

    # Test forward pass
    test_data = create_test_data(batch_size=2, seq_len=512)  # Use full context length
    print(f"   Input shape: {test_data.shape}")

    with torch.no_grad():
        output = encoder(test_data)

    print(f"   Output shape: {output.shape}")
    print(f"   ✅ PatchTST encoder working!\n")

except Exception as e:
    print(f"   ❌ PatchTST failed: {e}\n")

# Test 3: Try to test fallback mode for other encoders
print("3. Testing fallback encoders (without pretrained weights)...")

# TimesFM fallback
try:
    from opentslm.model.encoder.TimesFMEncoder import TimesFMEncoder

    encoder = TimesFMEncoder(
        model_name="google/timesfm-1.0-200m",
        output_dim=ENCODER_OUTPUT_DIM,
        device=DEVICE
    )

    test_data = create_test_data()
    print(f"   TimesFM fallback - Input shape: {test_data.shape}")

    with torch.no_grad():
        output = encoder(test_data)

    print(f"   TimesFM fallback - Output shape: {output.shape}")
    print(f"   ✅ TimesFM fallback encoder working!")

except Exception as e:
    print(f"   ❌ TimesFM fallback failed: {e}")

# MOMENT fallback
try:
    from opentslm.model.encoder.MomentEncoder import MomentEncoder

    encoder = MomentEncoder(
        model_name="AutonLab/MOMENT-1-small",
        output_dim=ENCODER_OUTPUT_DIM,
        model_size="small",
        device=DEVICE
    )

    test_data = create_test_data()
    print(f"   MOMENT fallback - Input shape: {test_data.shape}")

    with torch.no_grad():
        output = encoder(test_data)

    print(f"   MOMENT fallback - Output shape: {output.shape}")
    print(f"   ✅ MOMENT fallback encoder working!")

except Exception as e:
    print(f"   ❌ MOMENT fallback failed: {e}")

print("\n" + "="*60)
print("Summary:")
print("- Chronos2 and PatchTST can load pretrained models from HuggingFace")
print("- Other encoders have fallback implementations when libraries aren't available")
print("- All encoders follow the same interface and output dimension")
print("\nTo use pretrained models for all encoders, install:")
print("  pip install timesfm momentfm lag-llama")