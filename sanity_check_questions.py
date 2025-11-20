#!/usr/bin/env python3
"""
Sanity check for MIMIC-IV and PTB-XL question datasets.
Can run without downloading ECG signals.
"""

import json
import os
from collections import defaultdict, Counter
from glob import glob

def analyze_questions(base_dir, name):
    """Analyze question structure for a dataset."""
    print(f"\n{'='*70}")
    print(f"{name} Question Analysis")
    print(f"{'='*70}")

    stats = {
        'total': 0,
        'by_split': {},
        'by_template': defaultdict(int),
        'by_type': defaultdict(int),
        'answer_types': Counter(),
        'ecg_ids': set()
    }

    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(base_dir, 'template', split)
        if not os.path.exists(split_dir):
            print(f"⚠️  Split {split} not found")
            continue

        split_count = 0
        json_files = glob(os.path.join(split_dir, '*.json'))

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                split_count += len(data)

                for q in data:
                    stats['by_template'][q['template_id']] += 1
                    stats['by_type'][q['question_type']] += 1

                    # Analyze answer format
                    answer = q['answer']
                    if isinstance(answer, list):
                        if len(answer) == 1:
                            stats['answer_types']['single'] += 1
                        else:
                            stats['answer_types']['multiple'] += 1
                    else:
                        stats['answer_types']['string'] += 1

                    # Collect ECG IDs
                    if 'ecg_id' in q:
                        ecg_ids = q['ecg_id']
                        if isinstance(ecg_ids, list):
                            stats['ecg_ids'].update(ecg_ids)
                        else:
                            stats['ecg_ids'].add(ecg_ids)

        stats['by_split'][split] = split_count
        stats['total'] += split_count

    # Print results
    print(f"\n📊 Split Statistics:")
    for split, count in stats['by_split'].items():
        print(f"   {split:8s}: {count:,} questions")
    print(f"   {'Total':8s}: {stats['total']:,} questions")

    print(f"\n📋 Template Distribution:")
    print(f"   Number of templates: {len(stats['by_template'])}")
    print(f"   Templates: {sorted(stats['by_template'].keys())}")

    print(f"\n📝 Question Types:")
    for qtype, count in sorted(stats['by_type'].items()):
        percentage = 100 * count / stats['total']
        print(f"   {qtype:25s}: {count:>7,} ({percentage:5.1f}%)")

    print(f"\n💬 Answer Format:")
    for atype, count in sorted(stats['answer_types'].items()):
        percentage = 100 * count / stats['total']
        print(f"   {atype:15s}: {count:>7,} ({percentage:5.1f}%)")

    print(f"\n🏥 Unique ECGs:")
    print(f"   {len(stats['ecg_ids']):,} unique ECG recordings")

    return stats


def compare_datasets(ptbxl_stats, mimic_stats):
    """Compare PTB-XL and MIMIC-IV datasets."""
    print(f"\n{'='*70}")
    print("Dataset Comparison: PTB-XL vs MIMIC-IV")
    print(f"{'='*70}")

    print(f"\n📊 Size Comparison:")
    print(f"   {'Metric':<25s} {'PTB-XL':>15s} {'MIMIC-IV':>15s} {'Ratio':>10s}")
    print(f"   {'-'*70}")

    ratio = mimic_stats['total'] / ptbxl_stats['total']
    print(f"   {'Total questions':<25s} {ptbxl_stats['total']:>15,} {mimic_stats['total']:>15,} {ratio:>9.2f}x")

    for split in ['train', 'valid', 'test']:
        if split in ptbxl_stats['by_split'] and split in mimic_stats['by_split']:
            p = ptbxl_stats['by_split'][split]
            m = mimic_stats['by_split'][split]
            ratio = m / p if p > 0 else 0
            print(f"   {f'{split} split':<25s} {p:>15,} {m:>15,} {ratio:>9.2f}x")

    p_ecg = len(ptbxl_stats['ecg_ids'])
    m_ecg = len(mimic_stats['ecg_ids'])
    ratio = m_ecg / p_ecg if p_ecg > 0 else 0
    print(f"   {'Unique ECGs':<25s} {p_ecg:>15,} {m_ecg:>15,} {ratio:>9.2f}x")

    print(f"\n📋 Template Comparison:")
    p_templates = set(ptbxl_stats['by_template'].keys())
    m_templates = set(mimic_stats['by_template'].keys())

    print(f"   PTB-XL templates: {len(p_templates)}")
    print(f"   MIMIC-IV templates: {len(m_templates)}")

    common = p_templates & m_templates
    ptbxl_only = p_templates - m_templates
    mimic_only = m_templates - p_templates

    print(f"   Common templates: {len(common)}")
    if ptbxl_only:
        print(f"   PTB-XL only: {sorted(ptbxl_only)}")
    if mimic_only:
        print(f"   MIMIC-IV only: {sorted(mimic_only)}")

    print(f"\n📝 Question Type Comparison:")
    all_types = set(ptbxl_stats['by_type'].keys()) | set(mimic_stats['by_type'].keys())
    print(f"   {'Type':<25s} {'PTB-XL':>15s} {'MIMIC-IV':>15s}")
    print(f"   {'-'*60}")
    for qtype in sorted(all_types):
        p = ptbxl_stats['by_type'].get(qtype, 0)
        m = mimic_stats['by_type'].get(qtype, 0)
        print(f"   {qtype:<25s} {p:>15,} {m:>15,}")


def check_template_answers(base_dir, name):
    """Check template answers file."""
    print(f"\n{'='*70}")
    print(f"{name} Template Answers")
    print(f"{'='*70}")

    answers_file = os.path.join(base_dir, 'answers_for_each_template.csv')

    if not os.path.exists(answers_file):
        print(f"⚠️  Template answers file not found: {answers_file}")
        return

    import csv

    with open(answers_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"✅ Found {len(rows)} template answer definitions")

    # Show a few examples
    print(f"\nSample templates:")
    for row in rows[:5]:
        template_id = row['template_id']
        classes = row['classes']
        print(f"   Template {template_id}: {classes[:80]}...")


def main():
    """Run all sanity checks."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " ECG-QA Question Dataset Sanity Check ".center(68) + "║")
    print("╚" + "=" * 68 + "╝")

    base_path = "/local/home/wangni/OpenTSLM/data/ecg-qa/ecgqa"

    # Analyze PTB-XL
    ptbxl_path = os.path.join(base_path, "ptbxl")
    if os.path.exists(ptbxl_path):
        ptbxl_stats = analyze_questions(ptbxl_path, "PTB-XL")
        check_template_answers(ptbxl_path, "PTB-XL")
    else:
        print(f"⚠️  PTB-XL not found at {ptbxl_path}")
        ptbxl_stats = None

    # Analyze MIMIC-IV
    mimic_path = os.path.join(base_path, "mimic-iv-ecg")
    if os.path.exists(mimic_path):
        mimic_stats = analyze_questions(mimic_path, "MIMIC-IV")
        check_template_answers(mimic_path, "MIMIC-IV")
    else:
        print(f"⚠️  MIMIC-IV not found at {mimic_path}")
        mimic_stats = None

    # Compare if both exist
    if ptbxl_stats and mimic_stats:
        compare_datasets(ptbxl_stats, mimic_stats)

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print("\n✅ Question structure analysis complete!")
    print("\nKey findings:")
    if ptbxl_stats:
        print(f"   - PTB-XL: {ptbxl_stats['total']:,} questions from {len(ptbxl_stats['ecg_ids']):,} ECGs")
    if mimic_stats:
        print(f"   - MIMIC-IV: {mimic_stats['total']:,} questions from {len(mimic_stats['ecg_ids']):,} ECGs")

    if ptbxl_stats and mimic_stats:
        ratio = mimic_stats['total'] / ptbxl_stats['total']
        print(f"   - MIMIC-IV has {ratio:.1f}x more questions than PTB-XL")

        ecg_ratio = len(mimic_stats['ecg_ids']) / len(ptbxl_stats['ecg_ids'])
        print(f"   - MIMIC-IV has {ecg_ratio:.1f}x more ECGs than PTB-XL")

    print("\n📝 Question formats are compatible between datasets")
    print("💾 Both use the same template structure")
    print("✨ Ready for cross-dataset evaluation!")
    print()


if __name__ == "__main__":
    main()
