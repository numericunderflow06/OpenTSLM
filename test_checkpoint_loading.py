#!/usr/bin/env python3
"""
Quick test to verify checkpoint loading fix works correctly.
"""

import os
import sys
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from types import SimpleNamespace
from model.encoder.CNNTokenizer import CNNTokenizer
from model.llm.TimeSeriesFlamingoWithTrainableEncoder import TimeSeriesFlamingoWithTrainableEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from open_flamingo.open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.open_flamingo.src.utils import extend_instance
from model_config import ENCODER_OUTPUT_DIM
from huggingface_hub import hf_hub_download

print("="*60)
print("Testing Checkpoint Loading Fix")
print("="*60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create model (same as eval script)
llm_id = "meta-llama/Llama-3.2-1B"
max_patches = 1024

print("\nCreating model...")
time_series_encoder = CNNTokenizer(max_patches=max_patches).to(device)
text_tokenizer = AutoTokenizer.from_pretrained(llm_id, local_files_only=False, trust_remote_code=True)
lang_encoder = AutoModelForCausalLM.from_pretrained(
    llm_id, local_files_only=False, trust_remote_code=True,
    device_map={"": device}, attn_implementation="eager"
)

text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
if text_tokenizer.pad_token is None:
    text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

extend_instance(lang_encoder, FlamingoLMMixin)
lang_encoder.set_decoder_layers_attr_name("model.layers")
lang_encoder.resize_token_embeddings(len(text_tokenizer))

flamingo_model = TimeSeriesFlamingoWithTrainableEncoder(
    vision_encoder=SimpleNamespace(visual=time_series_encoder),
    lang_encoder=lang_encoder,
    eoc_token_id=text_tokenizer.encode("<|endofchunk|>")[-1],
    media_token_id=text_tokenizer.encode("<image>")[-1],
    vis_dim=ENCODER_OUTPUT_DIM,
    cross_attn_every_n_layers=1,
)
flamingo_model.to(device)

print("✓ Model created")

# Load checkpoint
print("\nLoading checkpoint...")
checkpoint_path = hf_hub_download(
    repo_id="OpenTSLM/llama-3.2-1b-ecg-flamingo",
    filename="softprompt-llama_3_2_1b-ecg.pt",
    cache_dir="/local/home/wangni/.cache/huggingface"
)

ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

# TEST WITHOUT FIX
print("\n" + "="*60)
print("TEST 1: Loading WITHOUT prefix stripping (original buggy code)")
print("="*60)
missing_keys_buggy, unexpected_keys_buggy = flamingo_model.load_state_dict(ckpt['model_state'], strict=False)
print(f"Missing keys: {len(missing_keys_buggy)}")
print(f"Unexpected keys: {len(unexpected_keys_buggy)}")
if len(missing_keys_buggy) > 0:
    print(f"First 5 missing: {missing_keys_buggy[:5]}")
if len(unexpected_keys_buggy) > 0:
    print(f"First 5 unexpected: {unexpected_keys_buggy[:5]}")

# Reload model to test the fix
print("\nReloading model...")
flamingo_model = TimeSeriesFlamingoWithTrainableEncoder(
    vision_encoder=SimpleNamespace(visual=time_series_encoder),
    lang_encoder=lang_encoder,
    eoc_token_id=text_tokenizer.encode("<|endofchunk|>")[-1],
    media_token_id=text_tokenizer.encode("<image>")[-1],
    vis_dim=ENCODER_OUTPUT_DIM,
    cross_attn_every_n_layers=1,
)
flamingo_model.to(device)

# TEST WITH FIX
print("\n" + "="*60)
print("TEST 2: Loading WITH prefix stripping (FIXED)")
print("="*60)

# Apply the fix
state_dict = ckpt['model_state']
fixed_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('model.'):
        new_key = key[6:]  # Remove 'model.' prefix
        fixed_state_dict[new_key] = value
    else:
        fixed_state_dict[key] = value

print(f"Fixed {len([k for k in state_dict.keys() if k.startswith('model.')])} keys by stripping 'model.' prefix")

missing_keys_fixed, unexpected_keys_fixed = flamingo_model.load_state_dict(fixed_state_dict, strict=False)
print(f"Missing keys: {len(missing_keys_fixed)}")
print(f"Unexpected keys: {len(unexpected_keys_fixed)}")
if len(missing_keys_fixed) > 0:
    print(f"First 5 missing: {missing_keys_fixed[:5]}")
if len(unexpected_keys_fixed) > 0:
    print(f"First 5 unexpected: {unexpected_keys_fixed[:5]}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"WITHOUT fix - Missing keys: {len(missing_keys_buggy)}")
print(f"WITH fix    - Missing keys: {len(missing_keys_fixed)}")
print(f"\nImprovement: {len(missing_keys_buggy) - len(missing_keys_fixed)} keys now loaded correctly!")

if len(missing_keys_fixed) < 10:
    print("\n✅ FIX SUCCESSFUL! Model should now work properly.")
else:
    print("\n⚠️ Still have missing keys. Need further investigation.")
