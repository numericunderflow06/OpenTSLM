#!/usr/bin/env python3
"""
Build a mini ECG-QA dataset with exactly 100 randomly sampled QA pairs from MIMIC-IV-ECG.
The output format is 100% compatible with the ECG-QA PTB-XL format.
"""

import json
import glob
import os
import shutil
import csv
import random

# Set random seed for reproducibility
random.seed(42)

# Get downloaded ECGs
with open('data/mimic_iv_ecg/physionet.org/record_list.csv', 'r') as f:
    reader = csv.DictReader(f)
    downloaded_study_ids = set()
    base_path = "data/mimic_iv_ecg/physionet.org"

    for record in reader:
        study_id = int(record['study_id'])
        path = f"{base_path}/{record['path']}.dat"
        if os.path.exists(path):
            downloaded_study_ids.add(study_id)

print(f"Downloaded ECGs: {len(downloaded_study_ids)}")

# Load all MIMIC-IV-ECG QA samples
def load_all_qa_samples(base_dir):
    """Load all QA samples from a directory (all splits)."""
    all_samples = []
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue

        json_files = sorted(glob.glob(os.path.join(split_dir, '*.json')))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                samples = json.load(f)
                # Only include samples where all ECGs are downloaded
                for sample in samples:
                    if all(ecg_id in downloaded_study_ids for ecg_id in sample['ecg_id']):
                        all_samples.append(sample)

    return all_samples

# Load paraphrased samples
print("\nLoading paraphrased QA samples...")
paraphrased_dir = 'data/ecg-qa/ecgqa/mimic-iv-ecg/paraphrased'
all_paraphrased = load_all_qa_samples(paraphrased_dir)
print(f"Available paraphrased samples (with downloaded ECGs): {len(all_paraphrased)}")

# Load template samples
print("\nLoading template QA samples...")
template_dir = 'data/ecg-qa/ecgqa/mimic-iv-ecg/template'
all_template = load_all_qa_samples(template_dir)
print(f"Available template samples (with downloaded ECGs): {len(all_template)}")

# Sample exactly 100 QA pairs from paraphrased
print("\nSampling 100 QA pairs...")
sampled_paraphrased = random.sample(all_paraphrased, 100)

# Get the same samples from template version (matching by question_id and ecg_id)
def match_template_samples(paraphrased_samples, template_samples):
    """Find matching template samples for each paraphrased sample."""
    matched = []
    for para_sample in paraphrased_samples:
        for temp_sample in template_samples:
            if (para_sample['question_id'] == temp_sample['question_id'] and
                para_sample['ecg_id'] == temp_sample['ecg_id'] and
                para_sample['template_id'] == temp_sample['template_id']):
                matched.append(temp_sample)
                break
    return matched

sampled_template = match_template_samples(sampled_paraphrased, all_template)

print(f"Sampled paraphrased: {len(sampled_paraphrased)}")
print(f"Matched template: {len(sampled_template)}")

# Get unique ECG IDs
unique_ecg_ids = set()
for sample in sampled_paraphrased:
    unique_ecg_ids.update(sample['ecg_id'])
print(f"Unique ECG IDs: {len(unique_ecg_ids)}")

# Create output directory structure
output_dir = 'data/ecg-qa-mini-100'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Copy answers.csv and answers_for_each_template.csv
shutil.copy('data/ecg-qa/ecgqa/mimic-iv-ecg/answers.csv',
            os.path.join(output_dir, 'answers.csv'))
shutil.copy('data/ecg-qa/ecgqa/mimic-iv-ecg/answers_for_each_template.csv',
            os.path.join(output_dir, 'answers_for_each_template.csv'))
print(f"\nCopied answers files to {output_dir}")

# Create train_ecgs.tsv with all unique ECGs
train_ecgs_file = os.path.join(output_dir, 'train_ecgs.tsv')
with open(train_ecgs_file, 'w') as f:
    for idx, ecg_id in enumerate(sorted(unique_ecg_ids)):
        f.write(f"{idx}\t{ecg_id}\n")
print(f"Created {train_ecgs_file} with {len(unique_ecg_ids)} ECGs")

# Save sampled samples in PTB-XL compatible format
def save_samples(samples, output_dir, version='paraphrased'):
    """Save samples in PTB-XL compatible format."""
    os.makedirs(os.path.join(output_dir, version, 'train'), exist_ok=True)

    # Reassign sample_ids to be sequential starting from 0
    for idx, sample in enumerate(samples):
        sample['sample_id'] = idx

    # Save all samples in one file (since we only have 100)
    output_file = os.path.join(output_dir, version, 'train', '000000.json')
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} samples to {output_file}")

print("\nSaving paraphrased samples...")
save_samples(sampled_paraphrased, output_dir, 'paraphrased')

print("\nSaving template samples...")
save_samples(sampled_template, output_dir, 'template')

# Print statistics
print(f"\n{'='*60}")
print(f"Mini ECG-QA dataset created successfully!")
print(f"Output directory: {output_dir}")
print(f"  - {len(unique_ecg_ids)} unique ECGs")
print(f"  - {len(sampled_paraphrased)} paraphrased QA pairs")
print(f"  - {len(sampled_template)} template QA pairs")

# Count question types
question_types = {}
attribute_types = {}
for sample in sampled_paraphrased:
    qt = sample['question_type']
    at = sample['attribute_type']
    question_types[qt] = question_types.get(qt, 0) + 1
    attribute_types[at] = attribute_types.get(at, 0) + 1

print(f"\nQuestion type distribution:")
for qt, count in sorted(question_types.items()):
    print(f"  {qt}: {count}")

print(f"\nAttribute type distribution:")
for at, count in sorted(attribute_types.items()):
    print(f"  {at}: {count}")

print(f"{'='*60}")
print(f"\nDataset structure (100% compatible with ECG-QA PTB-XL):")
print(f"  ✓ train_ecgs.tsv (ECG ID list)")
print(f"  ✓ answers.csv (all possible answers)")
print(f"  ✓ answers_for_each_template.csv (template-specific answers)")
print(f"  ✓ paraphrased/train/000000.json (paraphrased questions)")
print(f"  ✓ template/train/000000.json (template questions)")
print(f"\nTo use this dataset, simply point your ECG-QA code to:")
print(f"  {os.path.abspath(output_dir)}")
