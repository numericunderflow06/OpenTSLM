#!/usr/bin/env python3
"""
Quick validation test for Qwen 3 14B with OpenTSLM.
Tests model loading, dataset loading, and basic inference.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torch
from time_series_datasets.polymarket.PolymarketTrendDataset import PolymarketTrendDataset
from time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from model.llm.OpenTSLMSP import OpenTSLMSP
from model_config import PATCH_SIZE

print("=" * 80)
print("Quick Validation Test - Qwen 3 14B + OpenTSLM")
print("=" * 80)

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
print("\n[1/4] Loading model...")
print("  Using 4-bit quantization to reduce memory usage...")
try:
    model = OpenTSLMSP(llm_id="Qwen/Qwen3-14B", device=device, load_in_4bit=True).to(device)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    sys.exit(1)

# Load dataset
print("\n[2/4] Loading dataset...")
try:
    val_dataset = PolymarketTrendDataset("validation", model.get_eos_token())
    print(f"✓ Dataset loaded: {len(val_dataset)} samples")
except Exception as e:
    print(f"✗ Dataset loading failed: {e}")
    sys.exit(1)

# Test inference on 3 samples
print("\n[3/4] Testing inference on 3 samples...")
model.eval()

for i in range(min(3, len(val_dataset))):
    try:
        sample = val_dataset[i]
        batch = extend_time_series_to_match_patch_size_and_aggregate([sample], patch_size=PATCH_SIZE)

        with torch.no_grad():
            predictions = model.generate(batch, max_new_tokens=50)

        print(f"\nSample {i+1}:")
        print(f"  Type: {sample.get('question_type', 'unknown')}")
        print(f"  Market: {sample.get('question_text', 'N/A')[:60]}...")
        print(f"  Generated: {predictions[0]}")
        print(f"  Ground Truth: {sample.get('answer', 'N/A')}")

    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "=" * 80)
print("✓ Validation test passed!")
print("=" * 80)
print("\nThe model is ready for training.")
