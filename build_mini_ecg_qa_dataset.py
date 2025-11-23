#!/usr/bin/env python3
"""
Build a mini ECG-QA dataset with 100 randomly sampled ECGs from MIMIC-IV-ECG.
The output format is compatible with the ECG-QA PTB-XL format.
"""

import json
import glob
import os
import shutil
from collections import defaultdict

# Load the sampled study IDs
with open('sampled_100_study_ids.txt', 'r') as f:
    sampled_study_ids = set(int(line.strip()) for line in f)

print(f"Sampled {len(sampled_study_ids)} study IDs")

# Load all MIMIC-IV-ECG QA samples from both template and paraphrased versions
def load_qa_samples(base_dir):
    """Load all QA samples from a directory."""
    all_samples = []
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue

        json_files = sorted(glob.glob(os.path.join(split_dir, '*.json')))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                samples = json.load(f)
                all_samples.extend(samples)

    return all_samples

print("\nLoading paraphrased QA samples...")
paraphrased_dir = 'data/ecg-qa/ecgqa/mimic-iv-ecg/paraphrased'
paraphrased_samples = load_qa_samples(paraphrased_dir)
print(f"Total paraphrased samples: {len(paraphrased_samples)}")

print("\nLoading template QA samples...")
template_dir = 'data/ecg-qa/ecgqa/mimic-iv-ecg/template'
template_samples = load_qa_samples(template_dir)
print(f"Total template samples: {len(template_samples)}")

# Filter samples that belong to our 100 sampled ECGs
def filter_samples(samples, sampled_ids):
    """Filter samples to only include those with ECG IDs in sampled_ids."""
    filtered = []
    for sample in samples:
        # Check if all ECG IDs in the sample are in our sampled set
        ecg_ids = sample['ecg_id']
        if all(ecg_id in sampled_ids for ecg_id in ecg_ids):
            filtered.append(sample)
    return filtered

print("\nFiltering samples...")
filtered_paraphrased = filter_samples(paraphrased_samples, sampled_study_ids)
filtered_template = filter_samples(template_samples, sampled_study_ids)

print(f"Filtered paraphrased samples: {len(filtered_paraphrased)}")
print(f"Filtered template samples: {len(filtered_template)}")

# Get unique ECG IDs from filtered samples
unique_ecg_ids = set()
for sample in filtered_paraphrased:
    unique_ecg_ids.update(sample['ecg_id'])
print(f"\nUnique ECG IDs in filtered dataset: {len(unique_ecg_ids)}")

# Create output directory structure
output_dir = 'data/ecg-qa-mini-100'
os.makedirs(output_dir, exist_ok=True)

# Copy answers.csv and answers_for_each_template.csv
shutil.copy('data/ecg-qa/ecgqa/mimic-iv-ecg/answers.csv',
            os.path.join(output_dir, 'answers.csv'))
shutil.copy('data/ecg-qa/ecgqa/mimic-iv-ecg/answers_for_each_template.csv',
            os.path.join(output_dir, 'answers_for_each_template.csv'))
print(f"\nCopied answers files to {output_dir}")

# Since we have 100 ECGs, we'll use all of them as a single split
# Let's use them all as "train" to maximize the dataset size
# User can split them later if needed

# Create a single train split with all 100 ECGs
train_ecgs_file = os.path.join(output_dir, 'train_ecgs.tsv')
with open(train_ecgs_file, 'w') as f:
    for idx, ecg_id in enumerate(sorted(unique_ecg_ids)):
        f.write(f"{idx}\t{ecg_id}\n")
print(f"Created {train_ecgs_file} with {len(unique_ecg_ids)} ECGs")

# Save filtered samples in PTB-XL compatible format
def save_samples(samples, output_dir, version='paraphrased', samples_per_file=10000):
    """Save samples in PTB-XL compatible format."""
    os.makedirs(os.path.join(output_dir, version, 'train'), exist_ok=True)

    # Reassign sample_ids to be sequential starting from 0
    for idx, sample in enumerate(samples):
        sample['sample_id'] = idx

    # Save samples in chunks
    num_files = (len(samples) + samples_per_file - 1) // samples_per_file
    for file_idx in range(num_files):
        start_idx = file_idx * samples_per_file
        end_idx = min((file_idx + 1) * samples_per_file, len(samples))
        chunk = samples[start_idx:end_idx]

        output_file = os.path.join(output_dir, version, 'train',
                                   f"{start_idx:06d}.json")
        with open(output_file, 'w') as f:
            json.dump(chunk, f, indent=2)

        print(f"Saved {len(chunk)} samples to {output_file}")

print("\nSaving paraphrased samples...")
save_samples(filtered_paraphrased, output_dir, 'paraphrased')

print("\nSaving template samples...")
save_samples(filtered_template, output_dir, 'template')

print(f"\n{'='*60}")
print(f"Mini ECG-QA dataset created successfully!")
print(f"Output directory: {output_dir}")
print(f"  - {len(unique_ecg_ids)} unique ECGs")
print(f"  - {len(filtered_paraphrased)} paraphrased QA pairs")
print(f"  - {len(filtered_template)} template QA pairs")
print(f"{'='*60}")
