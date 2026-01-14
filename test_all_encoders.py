#!/usr/bin/env python
"""
Test script for all real time series encoders using pretrained models.
"""

import torch
import sys
import traceback

# Add paths for external libraries
sys.path.append('/tmp/timesfm/src')
sys.path.append('/tmp/moment')

from src.opentslm.model.encoder.ChronosEncoderReal import ChronosEncoderReal
from src.opentslm.model.encoder.PatchTSTEncoderReal import PatchTSTEncoderReal
from src.opentslm.model.encoder.TimesFMEncoderReal import TimesFMEncoderReal
from src.opentslm.model.encoder.MOMENTEncoderReal import MOMENTEncoderReal

def test_encoder(encoder_class, encoder_name, **kwargs):
    """Test a single encoder."""
    print(f"\n{'='*60}")
    print(f"Testing {encoder_name}")
    print('='*60)

    try:
        # Use GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Initialize encoder
        print(f"Initializing {encoder_name}...")
        encoder = encoder_class(device=device, **kwargs)

        # Create test input
        batch_size = 2
        seq_len = 512
        n_features = 1
        x = torch.randn(batch_size, seq_len, n_features).to(device)
        print(f"Input shape: {x.shape}")

        # Forward pass
        print(f"Running forward pass...")
        output = encoder(x)

        # Report results
        print(f"\n✓ {encoder_name} WORKS!")
        print(f"  Output shape: {output.shape}")
        print(f"  Output device: {output.device}")
        print(f"  Output mean: {output.mean().item():.4f}")
        print(f"  Output std: {output.std().item():.4f}")
        print(f"  Output min: {output.min().item():.4f}")
        print(f"  Output max: {output.max().item():.4f}")

        return True

    except Exception as e:
        print(f"\n✗ {encoder_name} FAILED!")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print(" TESTING ALL REAL TIME SERIES ENCODERS")
    print("="*60)

    # Track results
    results = {}

    # Test Chronos encoder
    results['Chronos'] = test_encoder(
        ChronosEncoderReal,
        "Chronos Encoder",
        model_name="amazon/chronos-t5-tiny"
    )

    # Test PatchTST encoder
    results['PatchTST'] = test_encoder(
        PatchTSTEncoderReal,
        "PatchTST Encoder",
        model_name="ibm/patchtst-etth1-pretrain"
    )

    # Test TimesFM encoder
    results['TimesFM'] = test_encoder(
        TimesFMEncoderReal,
        "TimesFM Encoder"
    )

    # Test MOMENT encoder
    results['MOMENT'] = test_encoder(
        MOMENTEncoderReal,
        "MOMENT Encoder",
        model_name="AutonLab/MOMENT-1-large"
    )

    # Summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)

    for encoder_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{encoder_name:15} : {status}")

    passed = sum(1 for s in results.values() if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} encoders working")

    if passed == total:
        print("\n🎉 ALL ENCODERS WORKING WITH REAL PRETRAINED MODELS!")
    else:
        print(f"\n⚠️ {total - passed} encoder(s) need attention")

if __name__ == "__main__":
    main()