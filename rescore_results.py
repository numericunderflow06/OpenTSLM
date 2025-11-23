#!/usr/bin/env python3
"""Re-score existing evaluation results with fixed answer normalization."""

import json
from pathlib import Path

def normalize_answer(answer):
    """Normalize answer for comparison."""
    if isinstance(answer, list):
        answer = answer[0] if len(answer) > 0 else ""
    # Remove special tokens like <|end_of_text|>
    answer = str(answer).replace('<|end_of_text|>', '').replace('<|endofchunk|>', '').replace('<image>', '')
    return answer.lower().strip()

def check_answer_correctness(prediction, ground_truth):
    """Check if prediction matches ground truth."""
    pred_normalized = normalize_answer(prediction)
    gt_normalized = normalize_answer(ground_truth)

    # Check if ground truth appears in prediction or vice versa
    return gt_normalized in pred_normalized or pred_normalized == gt_normalized

# Load results
results_file = "results_evaluation/llama-3.2-1b-ecg-flamingo_mimiciv_mini100_20251123_194505/detailed_results.jsonl"

results = []
with open(results_file, 'r') as f:
    for line in f:
        results.append(json.loads(line))

# Re-score
correct = 0
total = 0

print("Re-scoring with fixed normalization...\n")
print("First 10 samples:")
for i, result in enumerate(results[:10]):
    gt = result['ground_truth']
    pred = result['prediction']
    is_correct = check_answer_correctness(pred, gt)

    if is_correct:
        correct += 1
    total += 1

    print(f"\nSample {i+1}:")
    print(f"  GT: {gt[:80]}")
    print(f"  Pred: {pred[:80]}")
    print(f"  Correct: {is_correct}")

# Score all
correct = 0
for result in results:
    if check_answer_correctness(result['prediction'], result['ground_truth']):
        correct += 1

total = len(results)
accuracy = correct / total if total > 0 else 0

print(f"\n{'='*60}")
print(f"RESCORED RESULTS")
print(f"{'='*60}")
print(f"Total: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.2%}")
