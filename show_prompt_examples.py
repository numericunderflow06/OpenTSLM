#!/usr/bin/env python3
"""Show example prompts with time series token insertion for both datasets."""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from datasets import Dataset
from transformers import AutoTokenizer

# Load tokenizer to see special tokens
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)
tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

print("=" * 80)
print("SPECIAL TOKENS")
print("=" * 80)
print(f"Media token (<image>): {tokenizer.encode('<image>')}")
print(f"End of chunk token: {tokenizer.encode('<|endofchunk|>')}")
print(f"EOS token: {tokenizer.eos_token}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print()

# Load mini-100 data
qa_file = "data/ecg-qa-mini-100/paraphrased/train/000000.json"
with open(qa_file, 'r') as f:
    qa_samples = json.load(f)

# Take first sample
sample = qa_samples[0]
if isinstance(sample.get('answer'), list):
    sample['answer'] = sample['answer'][0]
sample['clinical_contexts'] = [""]
sample['ecg_paths'] = ["data/mimic_iv_ecg/physionet.org/files/p1085/p10857449/s47851432/47851432.dat"]

# Create dataset
dataset = Dataset.from_list([sample])

# Configure loader
import time_series_datasets.ecg_qa.mimiciv_ecg_loader as loader_module
loader_module.MIMIC_IV_ECG_DATA_DIR = "data/mimic_iv_ecg/physionet.org"

def load_custom():
    return dataset, dataset, dataset

original_loader = loader_module.load_ecg_qa_mimiciv_splits
loader_module.load_ecg_qa_mimiciv_splits = load_custom

try:
    from time_series_datasets.ecg_qa.ECGQAMimicIVDataset import ECGQAMimicIVDataset

    print("=" * 80)
    print("MIMIC-IV ECG-QA DATASET - FORMATTED SAMPLE")
    print("=" * 80)

    ds = ECGQAMimicIVDataset(
        split="train",
        EOS_TOKEN="<|end_of_text|>",
        use_cot_format=False,
        exclude_comparison=False,
        preload_processed_data=False,
    )

    formatted_sample = ds[0]

    print(f"\nQuestion (raw): {sample['question']}")
    print(f"\nAnswer (raw): {sample['answer']}")
    print()

    print("-" * 80)
    print("PRE-PROMPT (before time series):")
    print("-" * 80)
    print(formatted_sample['pre_prompt'])
    print()

    print("-" * 80)
    print("TIME SERIES TEXT (labels for each ECG lead):")
    print("-" * 80)
    for i, ts_text in enumerate(formatted_sample['time_series_text']):
        print(f"  [{i+1}] {ts_text}")
    print()

    print("-" * 80)
    print("POST-PROMPT (after time series):")
    print("-" * 80)
    print(formatted_sample['post_prompt'])
    print()

    print("-" * 80)
    print("ANSWER:")
    print("-" * 80)
    print(formatted_sample['answer'])
    print()

    # Now show how OpenTSLMFlamingo constructs the full prompt
    print("=" * 80)
    print("FULL PROMPT CONSTRUCTION (as done by OpenTSLMFlamingo.pad_and_apply_batch)")
    print("=" * 80)

    media_token = tokenizer.decode([tokenizer.encode("<image>")[-1]])
    endofchunk_token = tokenizer.decode([tokenizer.encode("<|endofchunk|>")[-1]])

    # Reconstruct how the prompt is built
    full_prompt = formatted_sample['pre_prompt']

    for ts_text in formatted_sample['time_series_text']:
        full_prompt += f" {media_token} {ts_text} {endofchunk_token}"

    if formatted_sample['post_prompt']:
        full_prompt += f" {formatted_sample['post_prompt']}"

    print("\nFull prompt with time series tokens:")
    print("-" * 80)
    print(full_prompt)
    print()

    print("=" * 80)
    print("TOKEN BREAKDOWN")
    print("=" * 80)
    print(f"Total time series: {len(formatted_sample['time_series'])} ECG leads")
    print(f"Each time series: {len(formatted_sample['time_series'][0])} samples (1000 samples at 100Hz = 10 seconds)")
    print(f"<image> tokens inserted: {len(formatted_sample['time_series_text'])} (one per ECG lead)")
    print()

    # Show the actual token sequence structure
    print("=" * 80)
    print("PROMPT STRUCTURE")
    print("=" * 80)
    print("""
Structure of the full prompt sent to the model:

1. PRE-PROMPT (task description + question)
2. <image> ECG Lead I label <|endofchunk|>
3. <image> ECG Lead II label <|endofchunk|>
4. <image> ECG Lead III label <|endofchunk|>
5. <image> ECG Lead aVR label <|endofchunk|>
6. <image> ECG Lead aVL label <|endofchunk|>
7. <image> ECG Lead aVF label <|endofchunk|>
8. <image> ECG Lead V1 label <|endofchunk|>
9. <image> ECG Lead V2 label <|endofchunk|>
10. <image> ECG Lead V3 label <|endofchunk|>
11. <image> ECG Lead V4 label <|endofchunk|>
12. <image> ECG Lead V5 label <|endofchunk|>
13. <image> ECG Lead V6 label <|endofchunk|>
14. POST-PROMPT (instructions)
15. [MODEL GENERATES ANSWER HERE]

The <image> tokens are where the model attends to the actual ECG signal data
through the cross-attention mechanism in the Flamingo architecture.
    """)

    # Show comparison sample count
    print("=" * 80)
    print("TOKENIZED LENGTHS")
    print("=" * 80)
    tokens_pre = tokenizer.encode(formatted_sample['pre_prompt'], add_special_tokens=False)
    tokens_post = tokenizer.encode(formatted_sample['post_prompt'], add_special_tokens=False)
    print(f"Pre-prompt tokens: {len(tokens_pre)}")
    print(f"Post-prompt tokens: {len(tokens_post)}")
    print(f"Time series labels tokens: ~{len(formatted_sample['time_series_text'])} × ~20 = ~{len(formatted_sample['time_series_text']) * 20}")
    print(f"Special tokens: {len(formatted_sample['time_series_text'])} × 2 (media + eoc) = {len(formatted_sample['time_series_text']) * 2}")
    print(f"Total input tokens (approx): {len(tokens_pre) + len(tokens_post) + len(formatted_sample['time_series_text']) * 22}")

finally:
    loader_module.load_ecg_qa_mimiciv_splits = original_loader

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)
