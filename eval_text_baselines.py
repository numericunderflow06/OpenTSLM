#!/usr/bin/env python3
"""Evaluate text-only LLM baselines on ZuCo eye-tracking reading task classification.

Sends the same eye-tracking data as formatted text to various LLMs and measures
their accuracy at classifying Normal Reading vs Task-Specific Reading.

Usage:
    # All local models
    CUDA_VISIBLE_DEVICES=1 python3 eval_text_baselines.py

    # Specific model
    CUDA_VISIBLE_DEVICES=1 python3 eval_text_baselines.py --models llama-3b

    # With GPT-4o (needs OPENAI_API_KEY env var)
    OPENAI_API_KEY=sk-... python3 eval_text_baselines.py --models gpt-4o
"""

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ZUCO1_PICKLE = Path("/home/wangni/OpenTSLM/data/zuco_et/preprocessed_et.pkl")
ZUCO2_PICKLE = Path("/home/wangni/OpenTSLM/data/zuco2_et/preprocessed_et_zuco2.pkl")

# Subject splits (must match OpenTSLM evaluation)
ZUCO1_TEST_SUBJECTS = ["ZMG", "ZPH"]
ZUCO2_TEST_SUBJECTS = ["YSD", "YSL", "YTL"]

ET_CHANNELS = ["FFD", "GD", "GPT", "TRT", "nFixations"]
ET_LABELS = {
    "FFD": "First Fixation Duration (ms)",
    "GD": "Gaze Duration (ms)",
    "GPT": "Go-Past Time (ms)",
    "TRT": "Total Reading Time (ms)",
    "nFixations": "Number of Fixations",
}

RESULTS_DIR = Path("/home/wangni/OpenTSLM/results/text_baselines")

MODEL_CONFIGS = {
    "llama-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "type": "local",
    },
    "llama-1b": {
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "type": "local",
    },
    "gemma-1b": {
        "hf_id": "google/gemma-3-1b-it",
        "type": "local",
    },
    "gemma-270m": {
        "hf_id": "google/gemma-3-270m-it",
        "type": "local",
    },
    "gpt-4o": {
        "api_model": "gpt-4o",
        "type": "openai",
    },
}

ALL_LOCAL_MODELS = ["llama-3b", "llama-1b", "gemma-1b", "gemma-270m"]
ALL_MODELS = ALL_LOCAL_MODELS + ["gpt-4o"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_test_samples(pickle_path: Path, test_subjects: list[str]) -> list[dict]:
    """Load test samples from preprocessed pickle, matching OpenTSLM's subject split."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    samples = []
    for task, label in [("NR", "Normal Reading"), ("TSR", "Task-Specific Reading")]:
        task_data = data[task]
        for subj in test_subjects:
            if subj not in task_data:
                continue
            for item in task_data[subj]:
                samples.append({
                    "sentence_text": item["sentence_text"],
                    "word_texts": item["word_texts"],
                    "et_metrics": item["et_metrics"],
                    "label": label,
                    "subject": subj,
                })
    return samples


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_prompt(sample: dict) -> str:
    """Format a sample as a text prompt with sentence + per-word ET metrics."""
    words = sample["word_texts"]
    metrics = sample["et_metrics"]
    n_words = len(words)

    lines = []
    lines.append(
        "You are given word-level eye-tracking measurements recorded while a "
        "participant read the following sentence. Based on the eye-tracking "
        "patterns, classify whether the reader was engaged in normal reading "
        "or task-specific reading (where they were actively searching for "
        "specific information).\n"
    )
    lines.append(f'Sentence: "{sample["sentence_text"].strip()}"\n')
    lines.append("Eye-tracking metrics per word:")

    # Header
    header = f"{'Word':<20} {'FFD(ms)':>8} {'GD(ms)':>8} {'GPT(ms)':>8} {'TRT(ms)':>8} {'nFix':>5}"
    lines.append(header)
    lines.append("-" * len(header))

    for i in range(n_words):
        word = words[i] if i < len(words) else "?"
        ffd = metrics["FFD"][i] if i < len(metrics["FFD"]) else 0.0
        gd = metrics["GD"][i] if i < len(metrics["GD"]) else 0.0
        gpt_val = metrics["GPT"][i] if i < len(metrics["GPT"]) else 0.0
        trt = metrics["TRT"][i] if i < len(metrics["TRT"]) else 0.0
        nfix = metrics["nFixations"][i] if i < len(metrics["nFixations"]) else 0.0
        lines.append(
            f"{word:<20} {ffd:>8.1f} {gd:>8.1f} {gpt_val:>8.1f} {trt:>8.1f} {nfix:>5.0f}"
        )

    lines.append("")
    lines.append(
        "Key: FFD=First Fixation Duration, GD=Gaze Duration, "
        "GPT=Go-Past Time, TRT=Total Reading Time, nFix=Number of Fixations. "
        "A value of 0.0 means the word was skipped (not fixated)."
    )
    lines.append("")
    lines.append("Possible labels: Normal Reading, Task-Specific Reading.")
    lines.append("Answer:")

    return "\n".join(lines)


def format_chat_messages(prompt: str) -> list[dict]:
    """Wrap prompt in chat message format."""
    return [
        {
            "role": "system",
            "content": (
                "You are an expert in psycholinguistics and eye-tracking analysis. "
                "Respond with ONLY the classification label, nothing else."
            ),
        },
        {"role": "user", "content": prompt},
    ]


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------


def run_local_model(model_key: str, samples: list[dict], dataset_name: str) -> dict:
    """Run a local HuggingFace model on all samples."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_key]
    hf_id = cfg["hf_id"]
    print(f"\n{'='*60}")
    print(f"Loading {model_key} ({hf_id})...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
    except torch.cuda.OutOfMemoryError:
        return {"model": model_key, "dataset": dataset_name, "error": "OOM during model loading"}
    except Exception as e:
        return {"model": model_key, "dataset": dataset_name, "error": str(e)}

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    predictions = []
    gold_labels = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"  [{model_key}] {dataset_name}: {i}/{len(samples)}...")

        prompt = format_prompt(sample)
        messages = format_chat_messages(prompt)

        try:
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            answer_ids = outputs[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            predictions.append(answer)
            gold_labels.append(sample["label"])

        except torch.cuda.OutOfMemoryError:
            return {
                "model": model_key,
                "dataset": dataset_name,
                "error": f"OOM during inference at sample {i}/{len(samples)}",
                "partial_predictions": len(predictions),
            }
        except Exception as e:
            predictions.append(f"ERROR: {e}")
            gold_labels.append(sample["label"])

    elapsed = time.time() - t0
    print(f"  [{model_key}] {dataset_name}: done in {elapsed:.1f}s")

    # Free GPU memory
    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()

    return compute_metrics(model_key, dataset_name, predictions, gold_labels, elapsed)


def run_openai_model(model_key: str, samples: list[dict], dataset_name: str) -> dict:
    """Run an OpenAI API model on all samples."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"model": model_key, "dataset": dataset_name, "error": "OPENAI_API_KEY not set"}

    try:
        from openai import OpenAI
    except ImportError:
        return {"model": model_key, "dataset": dataset_name, "error": "openai package not installed"}

    client = OpenAI(api_key=api_key)
    cfg = MODEL_CONFIGS[model_key]

    predictions = []
    gold_labels = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        if i % 100 == 0:
            print(f"  [{model_key}] {dataset_name}: {i}/{len(samples)}...")

        prompt = format_prompt(sample)
        messages = format_chat_messages(prompt)

        try:
            response = client.chat.completions.create(
                model=cfg["api_model"],
                messages=messages,
                max_tokens=20,
                temperature=0,
            )
            answer = response.choices[0].message.content.strip()
            predictions.append(answer)
            gold_labels.append(sample["label"])
        except Exception as e:
            predictions.append(f"ERROR: {e}")
            gold_labels.append(sample["label"])

        # Rate limiting
        if i % 50 == 49:
            time.sleep(1)

    elapsed = time.time() - t0
    print(f"  [{model_key}] {dataset_name}: done in {elapsed:.1f}s")

    return compute_metrics(model_key, dataset_name, predictions, gold_labels, elapsed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def parse_prediction(pred: str) -> str:
    """Parse model output to a label. Returns 'Normal Reading', 'Task-Specific Reading', or 'unknown'."""
    p = pred.lower().strip()
    # Remove common prefixes
    for prefix in ["answer:", "the answer is", "classification:", "label:"]:
        if p.startswith(prefix):
            p = p[len(prefix):].strip()

    if "task-specific" in p or "task specific" in p:
        return "Task-Specific Reading"
    elif "normal" in p:
        return "Normal Reading"
    return "unknown"


def compute_metrics(
    model_key: str,
    dataset_name: str,
    predictions: list[str],
    gold_labels: list[str],
    elapsed: float,
) -> dict:
    """Compute accuracy, per-class precision/recall/F1, balanced accuracy, confusion matrix."""
    parsed = [parse_prediction(p) for p in predictions]

    # Confusion matrix: rows=gold, cols=pred
    labels = ["Normal Reading", "Task-Specific Reading"]
    cm = np.zeros((2, 2), dtype=int)
    unknown_count = 0

    for gold, pred in zip(gold_labels, parsed):
        if pred == "unknown":
            unknown_count += 1
            continue
        gi = labels.index(gold)
        pi = labels.index(pred)
        cm[gi][pi] += 1

    total = len(predictions)
    correct = cm[0][0] + cm[1][1]
    accuracy = correct / total if total > 0 else 0.0

    # Per-class metrics
    per_class = {}
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = cm[1 - i][i]
        fn = cm[i][1 - i]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # Macro averages
    macro_precision = np.mean([per_class[l]["precision"] for l in labels])
    macro_recall = np.mean([per_class[l]["recall"] for l in labels])
    macro_f1 = np.mean([per_class[l]["f1"] for l in labels])
    balanced_acc = np.mean([cm[i][i] / max(cm[i].sum(), 1) for i in range(2)])

    result = {
        "model": model_key,
        "dataset": dataset_name,
        "total_samples": total,
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "unknown_predictions": unknown_count,
        "elapsed_seconds": round(elapsed, 1),
    }

    # Save raw predictions
    pred_file = RESULTS_DIR / f"{model_key}_{dataset_name}_predictions.jsonl"
    with open(pred_file, "w") as f:
        for gold, raw, parsed_label in zip(gold_labels, predictions, parsed):
            f.write(json.dumps({"gold": gold, "generated": raw, "parsed": parsed_label}) + "\n")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Text-only LLM baselines for ZuCo ET classification")
    parser.add_argument(
        "--models", nargs="+", default=ALL_LOCAL_MODELS,
        choices=ALL_MODELS,
        help=f"Models to evaluate (default: {ALL_LOCAL_MODELS})",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["zuco1", "zuco2"],
        choices=["zuco1", "zuco2"],
        help="Datasets to evaluate on",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    print("Loading test data...")
    test_data = {}
    if "zuco1" in args.datasets:
        test_data["zuco1"] = load_test_samples(ZUCO1_PICKLE, ZUCO1_TEST_SUBJECTS)
        print(f"  ZuCo 1.0: {len(test_data['zuco1'])} test samples")
    if "zuco2" in args.datasets:
        test_data["zuco2"] = load_test_samples(ZUCO2_PICKLE, ZUCO2_TEST_SUBJECTS)
        print(f"  ZuCo 2.0: {len(test_data['zuco2'])} test samples")

    # Verify counts match OpenTSLM
    for name, samples in test_data.items():
        nr = sum(1 for s in samples if s["label"] == "Normal Reading")
        tsr = sum(1 for s in samples if s["label"] == "Task-Specific Reading")
        print(f"  {name}: {nr} NR + {tsr} TSR = {nr + tsr} total")

    all_results = []

    for model_key in args.models:
        cfg = MODEL_CONFIGS[model_key]
        for dataset_name, samples in test_data.items():
            print(f"\n{'='*60}")
            print(f"Evaluating {model_key} on {dataset_name} ({len(samples)} samples)")
            print(f"{'='*60}")

            try:
                if cfg["type"] == "local":
                    result = run_local_model(model_key, samples, dataset_name)
                elif cfg["type"] == "openai":
                    result = run_openai_model(model_key, samples, dataset_name)
                else:
                    result = {"model": model_key, "dataset": dataset_name, "error": f"Unknown type: {cfg['type']}"}
            except Exception as e:
                result = {
                    "model": model_key,
                    "dataset": dataset_name,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }

            all_results.append(result)

            # Print summary
            if "error" in result:
                print(f"\n  ERROR: {result['error']}")
            else:
                print(f"\n  Accuracy: {result['accuracy']:.4f}")
                print(f"  Balanced Accuracy: {result['balanced_accuracy']:.4f}")
                print(f"  Macro F1: {result['macro_f1']:.4f}")
                print(f"  Unknown predictions: {result['unknown_predictions']}")
                cm = result["confusion_matrix"]["matrix"]
                print(f"  Confusion Matrix:")
                print(f"    {'':>20} Pred NR   Pred TSR")
                print(f"    {'Gold NR':>20} {cm[0][0]:>8} {cm[0][1]:>8}")
                print(f"    {'Gold TSR':>20} {cm[1][0]:>8} {cm[1][1]:>8}")

            # Save incremental results
            results_file = RESULTS_DIR / "all_results.json"
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Model':<15} {'Dataset':<8} {'Accuracy':>10} {'Bal.Acc':>10} {'Macro F1':>10} {'Unknown':>8}")
    print("-" * 65)
    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<15} {r['dataset']:<8} {'ERROR':>10}   {r['error'][:30]}")
        else:
            print(
                f"{r['model']:<15} {r['dataset']:<8} "
                f"{r['accuracy']:>10.4f} {r['balanced_accuracy']:>10.4f} "
                f"{r['macro_f1']:>10.4f} {r['unknown_predictions']:>8}"
            )

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Full results: {RESULTS_DIR}/all_results.json")


if __name__ == "__main__":
    main()
