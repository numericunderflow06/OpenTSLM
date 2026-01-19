#!/usr/bin/env python3
"""
Download Qwen 3 14B model to local cache.
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set environment variables
os.environ['HF_HOME'] = '/local/home/wangni/.cache/huggingface'
os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', 'YOUR_HF_TOKEN_HERE')
os.environ['TRANSFORMERS_CACHE'] = '/local/home/wangni/.cache/huggingface/transformers'

print("=" * 80)
print("Downloading Qwen 3 14B Model")
print("=" * 80)
print(f"Cache directory: {os.environ['HF_HOME']}")
print(f"Transformers cache: {os.environ['TRANSFORMERS_CACHE']}")
print()

model_id = "Qwen/Qwen3-14B"

print(f"Downloading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    token=os.environ['HF_TOKEN']
)
print("✓ Tokenizer downloaded")

print(f"\nDownloading model for {model_id}...")
print("(This may take several minutes - model is ~28GB)")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    token=os.environ['HF_TOKEN'],
    torch_dtype="auto",
    device_map=None  # Don't load to GPU, just download
)
print("✓ Model downloaded")

print("\n" + "=" * 80)
print("Download complete!")
print("=" * 80)
