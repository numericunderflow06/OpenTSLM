#!/usr/bin/env python3
#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Recompute accuracy for merged dataset using the original baseline evaluation logic.

This script matches the original OpenTSLM evaluation methodology for each task type:
- TSQA (MCQ): First 3 characters comparison (case-insensitive, exact match)
- HAR CoT: Extract label after "Answer:", compare to ground truth label
- SleepEDF CoT: Extract sleep stage after "Answer:", compare to ground truth
- ECG-QA CoT: Extract answer after "Answer:", normalize and compare
- M4 Captioning: No accuracy metric (text generation)
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict


def is_mcq_task(gold: str) -> bool:
    """
    Check if this is an MCQ task based on gold answer format.
    MCQ tasks have gold answers starting with option letters like "(a)", "(b)", etc.
    """
    cleaned = gold.replace("<|end_of_text|>", "").strip()
    return bool(re.match(r'^\([a-h]\)', cleaned, re.IGNORECASE))


def is_cot_task(gold: str) -> bool:
    """
    Check if this is a CoT task based on gold answer containing "Answer:" pattern.
    """
    return "Answer:" in gold and not is_mcq_task(gold)


def evaluate_mcq_baseline(gold: str, prediction: str) -> int:
    """
    Original OpenTSLM baseline evaluation logic from evaluate_tsqa.py.

    This uses:
    - First 3 characters comparison only
    - Lowercase, case-insensitive matching
    - Exact match after extracting answer
    """
    # Clean up strings for comparison
    gt_clean = gold.replace("<|end_of_text|>", "").lower().strip()
    pred_clean = prediction.lower().strip()

    # Only compare the first 3 characters (e.g., "(a)", "(b)", "(c)")
    gt_clean = gt_clean[:3]
    pred_clean = pred_clean[:3]

    # Extract the actual answer from the prediction (everything after "Answer:")
    answer_match = re.search(r'answer:\s*(.+)', pred_clean, re.IGNORECASE)
    if answer_match:
        pred_answer = answer_match.group(1).strip()[:3]
    else:
        pred_answer = pred_clean

    # Calculate accuracy (exact match)
    return int(gt_clean == pred_answer)


def extract_cot_label(text: str) -> str:
    """
    Extract the label after "Answer:" from CoT response.
    Following the original evaluate_har.py and evaluate_sleep_cot.py logic.
    """
    if text is None:
        return ""

    text = text.strip()

    # Remove end of text token
    text = text.replace("<|end_of_text|>", "").strip()

    # Find the last occurrence of 'Answer:' (case-insensitive)
    matches = list(re.finditer(r'answer:\s*', text, re.IGNORECASE))
    if matches:
        # Take everything after the last 'Answer:'
        start = matches[-1].end()
        label = text[start:].strip()
    else:
        # Take the last word as fallback
        words = text.split()
        label = words[-1] if words else ""

    # Remove trailing punctuation
    label = re.sub(r'[\.,;:!?]+$', '', label)

    return label.lower().strip()


def evaluate_cot_baseline(gold: str, prediction: str) -> int:
    """
    Evaluate CoT task by comparing extracted labels.
    """
    gold_label = extract_cot_label(gold)
    pred_label = extract_cot_label(prediction)

    return int(gold_label == pred_label)


def classify_task_type(sample: dict) -> str:
    """
    Classify the task type based on the post_prompt and gold fields.
    """
    post_prompt = sample.get("post_prompt", "").lower()
    gold = sample.get("gold", "")

    # MCQ tasks (TSQA)
    if is_mcq_task(gold):
        if "volatility" in post_prompt:
            return "TSQA_Volatility"
        elif "outlier" in post_prompt:
            return "TSQA_Outliers"
        elif "trend" in post_prompt:
            return "TSQA_Trend"
        elif "seasonality" in post_prompt:
            return "TSQA_Seasonality"
        else:
            return "TSQA_Other"

    # CoT tasks
    if is_cot_task(gold):
        if "accelerometer" in post_prompt or "activity" in post_prompt:
            return "HAR_CoT"
        elif "eeg" in post_prompt or "sleep" in post_prompt:
            return "SleepEDF_CoT"
        elif "ecg" in post_prompt:
            return "ECG_CoT"
        else:
            return "CoT_Other"

    # Captioning tasks
    if "caption" in post_prompt or "describe" in post_prompt:
        return "M4_Captioning"

    return "Other"


def compute_accuracy(predictions_file: str):
    """
    Compute accuracy from predictions file using original baseline methods.

    Args:
        predictions_file: Path to merged_test_predictions.jsonl or similar

    Returns:
        Dictionary with accuracy metrics and breakdown
    """
    results = {
        "total_samples": 0,
        "mcq_samples": 0,
        "mcq_correct": 0,
        "cot_samples": 0,
        "cot_correct": 0,
        "captioning_samples": 0,
        "by_task_type": defaultdict(lambda: {"total": 0, "correct": 0}),
        "mcq_errors": [],
        "cot_errors": [],
    }

    with open(predictions_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            sample = json.loads(line)
            gold = sample.get("gold", "")
            generated = sample.get("generated", "")

            results["total_samples"] += 1
            task_type = classify_task_type(sample)
            results["by_task_type"][task_type]["total"] += 1

            if is_mcq_task(gold):
                # MCQ evaluation
                results["mcq_samples"] += 1
                correct = evaluate_mcq_baseline(gold, generated)
                results["mcq_correct"] += correct
                results["by_task_type"][task_type]["correct"] += correct

                if not correct and len(results["mcq_errors"]) < 10:
                    results["mcq_errors"].append({
                        "line": line_num,
                        "task_type": task_type,
                        "gold_first3": gold.replace("<|end_of_text|>", "").strip()[:3].lower(),
                        "pred_first3": generated.strip()[:3].lower(),
                    })

            elif is_cot_task(gold):
                # CoT evaluation
                results["cot_samples"] += 1
                correct = evaluate_cot_baseline(gold, generated)
                results["cot_correct"] += correct
                results["by_task_type"][task_type]["correct"] += correct

                if not correct and len(results["cot_errors"]) < 10:
                    results["cot_errors"].append({
                        "line": line_num,
                        "task_type": task_type,
                        "gold_label": extract_cot_label(gold),
                        "pred_label": extract_cot_label(generated),
                    })

            else:
                # Captioning - no accuracy metric
                results["captioning_samples"] += 1

    # Compute accuracies
    results["mcq_accuracy"] = (
        results["mcq_correct"] / results["mcq_samples"]
        if results["mcq_samples"] > 0 else 0.0
    )
    results["cot_accuracy"] = (
        results["cot_correct"] / results["cot_samples"]
        if results["cot_samples"] > 0 else 0.0
    )

    # Combined accuracy for tasks with accuracy metrics
    total_eval = results["mcq_samples"] + results["cot_samples"]
    total_correct = results["mcq_correct"] + results["cot_correct"]
    results["overall_accuracy"] = total_correct / total_eval if total_eval > 0 else 0.0

    for task_type in results["by_task_type"]:
        task_data = results["by_task_type"][task_type]
        if task_data["total"] > 0 and task_type not in ["M4_Captioning", "Other"]:
            task_data["accuracy"] = task_data["correct"] / task_data["total"]
        else:
            task_data["accuracy"] = None

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Recompute accuracy for merged dataset using original baseline logic'
    )
    parser.add_argument(
        'predictions_file',
        type=str,
        help='Path to predictions JSONL file (e.g., merged_test_predictions.jsonl)'
    )
    parser.add_argument(
        '--save-metrics',
        type=str,
        default=None,
        help='Path to save metrics JSON file'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("RECOMPUTING ACCURACY WITH ORIGINAL BASELINE LOGIC")
    print("=" * 80)
    print(f"File: {args.predictions_file}")
    print()

    results = compute_accuracy(args.predictions_file)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples: {results['total_samples']}")
    print(f"  MCQ samples: {results['mcq_samples']}")
    print(f"  CoT samples: {results['cot_samples']}")
    print(f"  Captioning samples: {results['captioning_samples']}")
    print()
    print(f"MCQ Accuracy: {results['mcq_correct']}/{results['mcq_samples']} = "
          f"{results['mcq_accuracy']:.4f} ({results['mcq_accuracy']*100:.2f}%)")
    print(f"CoT Accuracy: {results['cot_correct']}/{results['cot_samples']} = "
          f"{results['cot_accuracy']:.4f} ({results['cot_accuracy']*100:.2f}%)")
    print(f"Overall Accuracy (MCQ+CoT): {results['mcq_correct']+results['cot_correct']}/"
          f"{results['mcq_samples']+results['cot_samples']} = "
          f"{results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
    print()

    print("=" * 80)
    print("BREAKDOWN BY TASK TYPE")
    print("=" * 80)
    for task_type, data in sorted(results["by_task_type"].items()):
        if data["accuracy"] is not None:
            print(f"{task_type:20s}: {data['correct']:4d}/{data['total']:4d} = "
                  f"{data['accuracy']:.4f} ({data['accuracy']*100:.2f}%)")
        else:
            print(f"{task_type:20s}: {data['total']:4d} samples (no accuracy metric)")
    print()

    if results["mcq_errors"]:
        print("=" * 80)
        print(f"FIRST {len(results['mcq_errors'])} MCQ ERRORS")
        print("=" * 80)
        for i, err in enumerate(results["mcq_errors"], 1):
            print(f"Error {i} (line {err['line']}, {err['task_type']}): "
                  f"gold='{err['gold_first3']}', pred='{err['pred_first3']}'")

    if results["cot_errors"]:
        print()
        print("=" * 80)
        print(f"FIRST {len(results['cot_errors'])} COT ERRORS")
        print("=" * 80)
        for i, err in enumerate(results["cot_errors"], 1):
            print(f"Error {i} (line {err['line']}, {err['task_type']}): "
                  f"gold='{err['gold_label']}', pred='{err['pred_label']}'")

    if args.save_metrics:
        metrics_to_save = {
            "total_samples": results["total_samples"],
            "mcq_samples": results["mcq_samples"],
            "mcq_correct": results["mcq_correct"],
            "mcq_accuracy": results["mcq_accuracy"],
            "cot_samples": results["cot_samples"],
            "cot_correct": results["cot_correct"],
            "cot_accuracy": results["cot_accuracy"],
            "overall_accuracy": results["overall_accuracy"],
            "captioning_samples": results["captioning_samples"],
            "by_task_type": {
                k: {
                    "total": v["total"],
                    "correct": v["correct"],
                    "accuracy": v["accuracy"]
                }
                for k, v in results["by_task_type"].items()
            }
        }
        with open(args.save_metrics, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"\nMetrics saved to: {args.save_metrics}")


if __name__ == "__main__":
    main()
