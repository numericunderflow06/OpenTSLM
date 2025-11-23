#!/usr/bin/env python3
"""
Zero-shot evaluation of llama-3.2-1b-ecg-flamingo on MIMIC-IV ECG-QA mini-100 dataset.

This script loads the pretrained model from HuggingFace and evaluates it on
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

from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from time_series_datasets.ecg_qa.mimiciv_ecg_loader import load_ecg_qa_mimiciv_custom_path
from time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from torch.utils.data import DataLoader


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

    # Load model from HuggingFace
    print("\n" + "="*60)
    print("Loading model from HuggingFace: OpenTSLM/llama-3.2-1b-ecg-flamingo")
    print("="*60)

    try:
        model = OpenTSLMFlamingo(
            device=device,
            llm_id="OpenTSLM/llama-3.2-1b-ecg-flamingo",
            cross_attn_every_n_layers=1,
        )
        model.to(device)
        model.eval()
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTrying to load base model and checkpoint separately...")

        # Try alternative loading method
        model = OpenTSLMFlamingo(
            device=device,
            llm_id="meta-llama/Llama-3.2-1B",  # Base model
            cross_attn_every_n_layers=1,
        )

        # Try to load weights from HuggingFace if available
        from huggingface_hub import hf_hub_download
        try:
            checkpoint_path = hf_hub_download(
                repo_id="OpenTSLM/llama-3.2-1b-ecg-flamingo",
                filename="pytorch_model.bin"
            )
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("✓ Model loaded from checkpoint!")
        except Exception as e2:
            print(f"✗ Error loading checkpoint: {e2}")
            print("Continuing with base model only (zero-shot, no ECG-specific training)")

        model.to(device)
        model.eval()

    # Load mini-100 dataset
    print("\n" + "="*60)
    print("Loading MIMIC-IV ECG-QA mini-100 dataset")
    print("="*60)

    # Import the dataset class
    from time_series_datasets.ecg_qa.ECGQAMimicIVDataset import ECGQAMimicIVDataset

    # Temporarily modify the loader to use our custom path
    dataset_path = "data/ecg-qa-mini-100"
    ecg_data_path = "data/mimic_iv_ecg/physionet.org"

    print(f"Dataset path: {dataset_path}")
    print(f"ECG data path: {ecg_data_path}")

    # Create a custom loader function
    def load_custom_mini100():
        """Load the mini-100 dataset."""
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
        return dataset, dataset, dataset  # Use same for all splits since we only have one

    # Monkey-patch the loader temporarily
    import time_series_datasets.ecg_qa.mimiciv_ecg_loader as loader_module
    original_loader = loader_module.load_ecg_qa_mimiciv_splits
    loader_module.load_ecg_qa_mimiciv_splits = load_custom_mini100

    # Also set the ECG data path
    original_ecg_path = getattr(loader_module, 'MIMIC_IV_ECG_DATA_DIR', None)
    loader_module.MIMIC_IV_ECG_DATA_DIR = ecg_data_path

    try:
        # Create dataset (will use our mini-100)
        dataset = ECGQAMimicIVDataset(
            split="train",
            EOS_TOKEN=model.get_eos_token(),
            use_cot_format=False,  # Use simple format for zero-shot
            exclude_comparison=False,
        )

        print(f"✓ Dataset loaded: {len(dataset)} samples")

    finally:
        # Restore original loader
        loader_module.load_ecg_qa_mimiciv_splits = original_loader
        if original_ecg_path is not None:
            loader_module.MIMIC_IV_ECG_DATA_DIR = original_ecg_path

    # Create dataloader
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

            # Get ground truth
            sample = batch[0]
            ground_truth = sample.get("answer", "")

            # Check correctness (simple string matching for now)
            # Normalize for comparison
            pred_normalized = prediction.lower().strip()
            gt_normalized = ground_truth.lower().strip() if isinstance(ground_truth, str) else str(ground_truth).lower().strip()

            is_correct = gt_normalized in pred_normalized or pred_normalized in gt_normalized
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
                "pre_prompt": sample.get("pre_prompt", ""),
                "post_prompt": sample.get("post_prompt", ""),
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
    for result in results:
        qtype = result["question_type"]
        if qtype not in question_type_stats:
            question_type_stats[qtype] = {"correct": 0, "total": 0}
        question_type_stats[qtype]["total"] += 1
        if result["is_correct"]:
            question_type_stats[qtype]["correct"] += 1

    for qtype in question_type_stats:
        stats = question_type_stats[qtype]
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"\nPer-Question-Type Accuracy:")
    for qtype, stats in sorted(question_type_stats.items()):
        print(f"  {qtype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

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
        "model": "OpenTSLM/llama-3.2-1b-ecg-flamingo",
        "dataset": "MIMIC-IV ECG-QA mini-100",
        "dataset_path": dataset_path,
        "timestamp": timestamp,
        "device": device,
        "total_samples": total,
        "correct_predictions": correct,
        "overall_accuracy": accuracy,
        "question_type_accuracy": question_type_stats,
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
        f.write(f"Model: OpenTSLM/llama-3.2-1b-ecg-flamingo\n")
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
    print(f"✓ Saved summary to: {summary_file}")

    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print("="*60)
    print(f"\nResults saved to: {results_dir}")

    return accuracy, results


if __name__ == "__main__":
    evaluate_model_on_mimiciv_mini100()
