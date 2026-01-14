#!/usr/bin/env python3
"""
Stress test encoders with edge cases and challenging inputs.
"""

import os
import sys
import torch
import gc

# Setup paths and environment
os.environ["HF_HOME"] = "/local/home/wangni/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/local/home/wangni/.cache/huggingface/transformers"
os.environ["TORCH_HOME"] = "/local/home/wangni/.cache/torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from opentslm.model_config import ENCODER_OUTPUT_DIM

print("="*60)
print("STRESS TEST - EDGE CASES")
print("="*60)

DEVICE = "cuda"

# Import all encoders
from opentslm.model.encoder.ChronosCompleteEncoder import ChronosCompleteEncoder
from opentslm.model.encoder.PatchTSTCompleteEncoder import PatchTSTCompleteEncoder
from opentslm.model.encoder.TimesFMFinalEncoder import TimesFMFinalEncoder
from opentslm.model.encoder.MOMENTEncoderReal import MOMENTEncoderReal

# Initialize encoders
encoders = {
    'Chronos': ChronosCompleteEncoder(
        model_name="amazon/chronos-t5-tiny",
        output_dim=ENCODER_OUTPUT_DIM,
        device=DEVICE
    ),
    'PatchTST': PatchTSTCompleteEncoder(
        model_name="ibm/patchtst-etth1-pretrain",
        output_dim=ENCODER_OUTPUT_DIM,
        device=DEVICE
    ),
    'TimesFM': TimesFMFinalEncoder(
        model_name="google/timesfm-2p5-200m-torch",
        output_dim=ENCODER_OUTPUT_DIM,
        context_length=512,
        device=DEVICE
    ),
    'MOMENT': MOMENTEncoderReal(
        model_name="AutonLab/MOMENT-1-large",
        output_dim=ENCODER_OUTPUT_DIM,
        device=DEVICE
    )
}

# Move all to GPU and eval mode
for name, encoder in encoders.items():
    encoder.to(DEVICE)
    encoder.eval()

print("\n✅ All encoders loaded\n")

# Test cases
test_cases = {
    'very_short': (1, 10, 1),
    'very_long': (2, 2048, 1),
    'large_batch': (32, 128, 1),
    'zero_input': (2, 100, 1),
    'constant_input': (2, 100, 1),
    'extreme_values': (2, 100, 1),
}

results = {name: [] for name in encoders.keys()}

for test_name, (batch, seq, feat) in test_cases.items():
    print(f"\n{test_name.upper()} Test: batch={batch}, seq={seq}, feat={feat}")
    print("-" * 40)

    # Create test data based on test type
    if test_name == 'zero_input':
        data = torch.zeros(batch, seq, feat).to(DEVICE)
    elif test_name == 'constant_input':
        data = torch.ones(batch, seq, feat).to(DEVICE) * 3.14
    elif test_name == 'extreme_values':
        data = torch.randn(batch, seq, feat).to(DEVICE) * 1000
    else:
        data = torch.randn(batch, seq, feat).to(DEVICE)

    for enc_name, encoder in encoders.items():
        try:
            # Adjust input for encoder requirements
            if enc_name == 'PatchTST':
                # PatchTST needs 7 channels and 512 seq length
                if feat != 7:
                    data_adj = torch.randn(batch, min(seq, 512), 7).to(DEVICE)
                else:
                    data_adj = data[:, :512] if seq > 512 else data
            else:
                data_adj = data

            with torch.no_grad():
                output = encoder(data_adj)

            # Check output
            assert output.shape == (batch, ENCODER_OUTPUT_DIM)
            assert output.device.type == 'cuda'
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

            print(f"  {enc_name:10s}: ✅ Passed (output stats: "
                  f"mean={output.mean().item():.3f}, "
                  f"std={output.std().item():.3f})")
            results[enc_name].append(True)

        except Exception as e:
            print(f"  {enc_name:10s}: ❌ Failed - {str(e)[:50]}")
            results[enc_name].append(False)

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

# Summary
print("\n" + "="*60)
print("STRESS TEST SUMMARY")
print("="*60)

for enc_name, test_results in results.items():
    passed = sum(test_results)
    total = len(test_results)
    status = "✅ ALL PASSED" if passed == total else f"⚠️  {passed}/{total} passed"
    print(f"{enc_name:10s}: {status}")

all_passed = all(all(r) for r in results.values())
if all_passed:
    print("\n🎉 ALL ENCODERS PASSED ALL STRESS TESTS!")
else:
    print("\n⚠️  Some encoders failed stress tests")

# Cleanup
del encoders
gc.collect()
torch.cuda.empty_cache()

sys.exit(0 if all_passed else 1)