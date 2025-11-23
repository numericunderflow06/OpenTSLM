#!/usr/bin/env python3
"""
Zero-shot evaluation of llama-3.2-1b-ecg-flamingo (softprompt) on MIMIC-IV ECG-QA mini-100 dataset.

This script loads the pretrained soft prompt model from HuggingFace and evaluates it on
our 100-question MIMIC-IV subset, logging all results including exact outputs,
ground truth, and accuracies.
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model.llm.OpenTSLMSP import OpenTSLMSP
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download


def load_mini100_dataset(dataset_path, ecg_data_path, model):
    """Load the MIMIC-IV mini-100 dataset."""
    import json
    from datasets import Dataset

    # Load the train split (which contains all 100 QA pairs)
    qa_samples = []
    qa_file = os.path.join(dataset_path, "paraphrased/train/000000.json")

    with open(qa_file, 'r') as f:
        qa_samples.extend(json.load(f))

    print(f"Loaded {len(qa_samples)} QA pairs")

    # Load ECG IDs
    ecg_ids = []
    with open(os.path.join(dataset_path, "train_ecgs.tsv"), 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                ecg_ids.append(int(parts[1]))

    print(f"Loaded {len(ecg_ids)} ECG IDs")

    # Convert to Dataset
    dataset = Dataset.from_list(qa_samples)

    # Import and temporarily configure the loader
    import time_series_datasets.ecg_qa.mimiciv_ecg_loader as loader_module

    # Store original loader
    original_loader = loader_module.load_ecg_qa_mimiciv_splits
    original_ecg_path = getattr(loader_module, 'MIMIC_IV_ECG_DATA_DIR', None)

    # Set custom paths
    loader_module.MIMIC_IV_ECG_DATA_DIR = ecg_data_path

    def load_custom_mini100():
        return dataset, dataset, dataset

    loader_module.load_ecg_qa_mimiciv_splits = load_custom_mini100

    try:
        # Import dataset class
        from time_series_datasets.ecg_qa.ECGQAMimicIVDataset import ECGQAMimicIVDataset

        # Create dataset
        ds = ECGQAMimicIVDataset(
            split="train",
            EOS_TOKEN=model.get_eos_token(),
            use_cot_format=False,  # Use simple format for zero-shot
            exclude_comparison=False,
            preload_processed_data=False,  # Don't preload to save memory
        )

        return ds

    finally:
        # Restore original loader
        loader_module.load_ecg_qa_mimiciv_splits = original_loader
        if original_ecg_path is not None:
            loader_module.MIMIC_IV_ECG_DATA_DIR = original_ecg_path


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if isinstance(answer, list):
        answer = answer[0] if len(answer) > 0 else ""
    return str(answer).lower().strip()


def check_answer_correctness(prediction, ground_truth):
    """Check if prediction matches ground truth."""
    pred_normalized = normalize_answer(prediction)
    gt_normalized = normalize_answer(ground_truth)

    # Check if ground truth appears in prediction or vice versa
    return gt_normalized in pred_normalized or pred_normalized == gt_normalized


def evaluate_model_on_mimiciv_mini100():
    """Evaluate the model on the MIMIC-IV ECG-QA mini-100 dataset."""

    # Setup
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results_evaluation/llama-3.2-1b-ecg-flamingo_mimiciv_mini100_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")

    # Load model
    print("\n" + "="*60)
    print("Loading model: OpenTSLMSP with llama-3.2-1b-ecg soft prompt")
    print("="*60)

    try:
        # Initialize base model
        print("Initializing base model...")
        model = OpenTSLMSP(
            device=device,
            llm_id="meta-llama/Llama-3.2-1B",
        )
        model.to(device)

        # Download and load soft prompt checkpoint
        print("Downloading soft prompt checkpoint from HuggingFace...")
        checkpoint_path = hf_hub_download(
            repo_id="OpenTSLM/llama-3.2-1b-ecg-flamingo",
            filename="softprompt-llama_3_2_1b-ecg.pt"
        )

        print(f"Loading checkpoint from: {checkpoint_path}")
        model.load_from_file(checkpoint_path)

        model.eval()
        print("✓ Model loaded successfully!")

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load mini-100 dataset
    print("\n" + "="*60)
    print("Loading MIMIC-IV ECG-QA mini-100 dataset")
    print("="*60)

    dataset_path = "data/ecg-qa-mini-100"
    ecg_data_path = "data/mimic_iv_ecg/physionet.org"

    print(f"Dataset path: {dataset_path}")
    print(f"ECG data path: {ecg_data_path}")

    try:
        dataset = load_mini100_dataset(dataset_path, ecg_data_path, model)
        print(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create dataloader
    from time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=128
        ),
    )

    # Run evaluation
    print("\n" + "="*60)
    print("Running zero-shot evaluation")
    print("="*60)

    results = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Generate prediction
            try:
                predictions = model.generate(batch)
                prediction = predictions[0] if isinstance(predictions, list) else predictions
            except Exception as e:
                print(f"\nError generating prediction for sample {batch_idx}: {e}")
                prediction = "[ERROR]"
                import traceback
                traceback.print_exc()

            # Get ground truth
            sample = batch[0]
            ground_truth = sample.get("answer", "")

            # Check correctness
            is_correct = check_answer_correctness(prediction, ground_truth)
            if is_correct:
                correct += 1
            total += 1

            # Store result
            result = {
                "sample_id": batch_idx,
                "question": sample.get("question", ""),
                "question_type": sample.get("question_type", ""),
                "attribute_type": sample.get("attribute_type", ""),
                "ecg_id": sample.get("ecg_id", []),
                "ground_truth": ground_truth,
                "prediction": prediction,
                "is_correct": is_correct,
                "pre_prompt": sample.get("pre_prompt", "")[:200] + "...",  # Truncate for readability
                "post_prompt": sample.get("post_prompt", "")[:200] + "...",
            }
            results.append(result)

            # Print progress every 10 samples
            if (batch_idx + 1) % 10 == 0:
                current_acc = correct / total
                print(f"\nProgress: {batch_idx + 1}/{len(dataset)} | Current accuracy: {current_acc:.2%}")

    # Calculate final accuracy
    accuracy = correct / total if total > 0 else 0.0

    # Calculate per-question-type accuracy
    question_type_stats = {}
    attribute_type_stats = {}

    for result in results:
        # Question type stats
        qtype = result["question_type"]
        if qtype not in question_type_stats:
            question_type_stats[qtype] = {"correct": 0, "total": 0}
        question_type_stats[qtype]["total"] += 1
        if result["is_correct"]:
            question_type_stats[qtype]["correct"] += 1

        # Attribute type stats
        atype = result["attribute_type"]
        if atype not in attribute_type_stats:
            attribute_type_stats[atype] = {"correct": 0, "total": 0}
        attribute_type_stats[atype]["total"] += 1
        if result["is_correct"]:
            attribute_type_stats[atype]["correct"] += 1

    for qtype in question_type_stats:
        stats = question_type_stats[qtype]
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

    for atype in attribute_type_stats:
        stats = attribute_type_stats[atype]
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{total})")

    print(f"\nPer-Question-Type Accuracy:")
    for qtype, stats in sorted(question_type_stats.items()):
        print(f"  {qtype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    print(f"\nPer-Attribute-Type Accuracy:")
    for atype, stats in sorted(attribute_type_stats.items()):
        print(f"  {atype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    # Save results
    print("\n" + "="*60)
    print("Saving results")
    print("="*60)

    # Save detailed results as JSONL
    results_file = results_dir / "detailed_results.jsonl"
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"✓ Saved detailed results to: {results_file}")

    # Save summary metrics
    metrics = {
        "model": "OpenTSLM/llama-3.2-1b-ecg-flamingo (soft prompt)",
        "base_llm": "meta-llama/Llama-3.2-1B",
        "checkpoint": "softprompt-llama_3_2_1b-ecg.pt",
        "dataset": "MIMIC-IV ECG-QA mini-100",
        "dataset_path": dataset_path,
        "timestamp": timestamp,
        "device": device,
        "total_samples": total,
        "correct_predictions": correct,
        "overall_accuracy": accuracy,
        "question_type_accuracy": question_type_stats,
        "attribute_type_accuracy": attribute_type_stats,
    }

    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved metrics to: {metrics_file}")

    # Create a human-readable summary
    summary_file = results_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MIMIC-IV ECG-QA Mini-100 Evaluation Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: OpenTSLM/llama-3.2-1b-ecg-flamingo (soft prompt)\n")
        f.write(f"Base LLM: meta-llama/Llama-3.2-1B\n")
        f.write(f"Checkpoint: softprompt-llama_3_2_1b-ecg.pt\n")
        f.write(f"Dataset: MIMIC-IV ECG-QA mini-100\n")
        f.write(f"Evaluation Date: {timestamp}\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"Overall Results:\n")
        f.write(f"  Total Samples: {total}\n")
        f.write(f"  Correct: {correct}\n")
        f.write(f"  Accuracy: {accuracy:.2%}\n\n")
        f.write(f"Per-Question-Type Accuracy:\n")
        for qtype, stats in sorted(question_type_stats.items()):
            f.write(f"  {qtype}:\n")
            f.write(f"    Accuracy: {stats['accuracy']:.2%}\n")
            f.write(f"    Correct/Total: {stats['correct']}/{stats['total']}\n\n")
        f.write(f"Per-Attribute-Type Accuracy:\n")
        for atype, stats in sorted(attribute_type_stats.items()):
            f.write(f"  {atype}:\n")
            f.write(f"    Accuracy: {stats['accuracy']:.2%}\n")
            f.write(f"    Correct/Total: {stats['correct']}/{stats['total']}\n\n")

        # Add some example predictions
        f.write("\n" + "="*60 + "\n")
        f.write("Example Predictions (first 5 samples)\n")
        f.write("="*60 + "\n\n")
        for i, result in enumerate(results[:5]):
            f.write(f"Sample {i+1}:\n")
            f.write(f"  Question: {result['question']}\n")
            f.write(f"  Ground Truth: {result['ground_truth']}\n")
            f.write(f"  Prediction: {result['prediction']}\n")
            f.write(f"  Correct: {result['is_correct']}\n\n")

    print(f"✓ Saved summary to: {summary_file}")

    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print("="*60)
    print(f"\nResults saved to: {results_dir}")
    print(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{total})")

    return accuracy, results


if __name__ == "__main__":
    evaluate_model_on_mimiciv_mini100()
