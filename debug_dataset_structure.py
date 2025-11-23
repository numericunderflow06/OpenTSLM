#!/usr/bin/env python3
"""Debug script to check what's in the formatted ECGQAMimicIVDataset samples."""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from datasets import Dataset

# Load mini-100 data
qa_file = "data/ecg-qa-mini-100/paraphrased/train/000000.json"
with open(qa_file, 'r') as f:
    qa_samples = json.load(f)

# Take just first sample for debugging
sample = qa_samples[0]

# Convert answer from list to string
if isinstance(sample.get('answer'), list):
    sample['answer'] = sample['answer'][0]

# Add required fields
sample['clinical_contexts'] = [""]
sample['ecg_paths'] = ["data/mimic_iv_ecg/physionet.org/files/p1085/p10857449/s47851432/47851432.dat"]

print("=" * 60)
print("RAW SAMPLE STRUCTURE")
print("=" * 60)
print(f"Keys: {sample.keys()}")
print(f"Has time_series: {'time_series' in sample}")
print(f"Has time_series_text: {'time_series_text' in sample}")
print()

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
    # Import and create dataset
    from time_series_datasets.ecg_qa.ECGQAMimicIVDataset import ECGQAMimicIVDataset

    print("=" * 60)
    print("CREATING ECGQAMimicIVDataset")
    print("=" * 60)

    ds = ECGQAMimicIVDataset(
        split="train",
        EOS_TOKEN="<|end_of_text|>",
        use_cot_format=False,
        exclude_comparison=False,
        preload_processed_data=False,
    )

    print(f"Dataset length: {len(ds)}")
    print()

    # Get formatted sample
    formatted_sample = ds[0]

    print("=" * 60)
    print("FORMATTED SAMPLE STRUCTURE")
    print("=" * 60)
    print(f"Keys: {formatted_sample.keys()}")
    print()
    print(f"Has pre_prompt: {'pre_prompt' in formatted_sample}")
    print(f"Has post_prompt: {'post_prompt' in formatted_sample}")
    print(f"Has time_series: {'time_series' in formatted_sample}")
    print(f"Has time_series_text: {'time_series_text' in formatted_sample}")
    print(f"Has answer: {'answer' in formatted_sample}")
    print()

    if 'time_series' in formatted_sample:
        ts = formatted_sample['time_series']
        print(f"time_series type: {type(ts)}")
        print(f"time_series length: {len(ts) if isinstance(ts, (list, tuple)) else 'N/A'}")
        if isinstance(ts, (list, tuple)) and len(ts) > 0:
            print(f"First time series type: {type(ts[0])}")
            print(f"First time series shape: {len(ts[0]) if hasattr(ts[0], '__len__') else 'N/A'}")

    if 'time_series_text' in formatted_sample:
        tst = formatted_sample['time_series_text']
        print(f"time_series_text type: {type(tst)}")
        print(f"time_series_text length: {len(tst) if isinstance(tst, (list, tuple)) else 'N/A'}")
        if isinstance(tst, (list, tuple)) and len(tst) > 0:
            print(f"First text: {tst[0][:100] if isinstance(tst[0], str) else tst[0]}")

    print()
    print(f"pre_prompt (first 200 chars): {formatted_sample.get('pre_prompt', 'N/A')[:200]}")
    print()
    print(f"post_prompt (first 200 chars): {formatted_sample.get('post_prompt', 'N/A')[:200]}")
    print()
    print(f"answer: {formatted_sample.get('answer', 'N/A')[:100]}")

finally:
    loader_module.load_ecg_qa_mimiciv_splits = original_loader

print()
print("=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
