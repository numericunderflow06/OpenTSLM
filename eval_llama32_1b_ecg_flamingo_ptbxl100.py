#!/usr/bin/env python3
"""
Zero-shot evaluation of llama-3.2-1b-ecg-flamingo on ECG-QA PTB-XL dataset (100 questions).

This script loads the pretrained Flamingo model from HuggingFace and evaluates it on
100 questions from the original ECG-QA PTB-XL dataset for sanity check.
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download


def setup_logging(results_dir):
    """Setup logging to both file and console."""
    log_file = results_dir / "evaluation.log"

    # Create logger
    logger = logging.getLogger('eval')
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


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


def create_modified_opentslm_flamingo(device, llm_id, cross_attn_every_n_layers, max_patches):
    """Create OpenTSLM Flamingo model with specified configuration."""
    from types import SimpleNamespace
    from model.encoder.CNNTokenizer import CNNTokenizer
    from model.llm.TimeSeriesFlamingoWithTrainableEncoder import TimeSeriesFlamingoWithTrainableEncoder
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from open_flamingo.open_flamingo.src.flamingo_lm import FlamingoLMMixin
    from open_flamingo.open_flamingo.src.utils import extend_instance

    # Create encoder with specified max_patches
    time_series_encoder = CNNTokenizer(max_patches=max_patches).to(device)

    # Create tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(
        llm_id,
        local_files_only=False,
        trust_remote_code=True,
    )

    # Load language model
    lang_encoder = AutoModelForCausalLM.from_pretrained(
        llm_id,
        local_files_only=False,
        trust_remote_code=True,
        device_map={"": device},
        attn_implementation="eager",
    )

    # Add Flamingo special tokens
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # Convert to Flamingo
    extend_instance(lang_encoder, FlamingoLMMixin)

    # Infer decoder layers and set up
    def _infer_decoder_layers_attr_name(model):
        model_class_name = model.__class__.__name__.lower()
        if "llama" in model_class_name:
            return "model.layers"
        elif "opt" in model_class_name:
            return "model.decoder.layers"
        elif "gpt" in model_class_name:
            return "transformer.h"
        else:
            return "model.layers"

    decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    # Initialize Flamingo with trainable encoder
    # Wrap encoder in SimpleNamespace as expected by Flamingo
    flamingo_model = TimeSeriesFlamingoWithTrainableEncoder(
        vision_encoder=SimpleNamespace(visual=time_series_encoder),
        lang_encoder=lang_encoder,
        eoc_token_id=text_tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1],
        media_token_id=text_tokenizer("<image>", add_special_tokens=False)["input_ids"][-1],
        vis_dim=128,  # ENCODER_OUTPUT_DIM
        cross_attn_every_n_layers=cross_attn_every_n_layers,
    )

    return flamingo_model, text_tokenizer


def evaluate_model_on_ptbxl100():
    """Evaluate the model on 100 questions from ECG-QA PTB-XL dataset."""

    # Setup
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results_evaluation/llama-3.2-1b-ecg-flamingo_ptbxl100_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(results_dir)
    logger.info("="*60)
    logger.info("ECG-QA PTB-XL 100-Question Evaluation (Sanity Check)")
    logger.info("="*60)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Timestamp: {timestamp}")

    # Load model
    logger.info("\n" + "="*60)
    logger.info("Loading model: OpenTSLM/llama-3.2-1b-ecg-flamingo")
    logger.info("="*60)

    try:
        llm_id = "meta-llama/Llama-3.2-1B"
        max_patches = 1024  # CRITICAL: Must match checkpoint

        logger.info(f"Base LLM: {llm_id}")
        logger.info(f"Max patches: {max_patches} (matches checkpoint)")
        logger.info("Creating Flamingo model...")

        # Create model with correct max_patches
        flamingo_model, text_tokenizer = create_modified_opentslm_flamingo(
            device=device,
            llm_id=llm_id,
            cross_attn_every_n_layers=1,
            max_patches=max_patches,
        )
        flamingo_model.to(device)

        logger.info("✓ Model structure created")

        # Download and load checkpoint
        logger.info("Downloading checkpoint from HuggingFace...")
        checkpoint_path = hf_hub_download(
            repo_id="OpenTSLM/llama-3.2-1b-ecg-flamingo",
            filename="softprompt-llama_3_2_1b-ecg.pt",
            cache_dir="/local/home/wangni/.cache/huggingface"
        )

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load model state
        if 'model_state' in ckpt:
            # Strip 'model.' prefix from checkpoint keys
            state_dict = ckpt['model_state']
            fixed_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.' prefix
                    fixed_state_dict[new_key] = value
                else:
                    fixed_state_dict[key] = value

            logger.info(f"Fixed {len([k for k in state_dict.keys() if k.startswith('model.')])} keys by stripping 'model.' prefix")

            missing_keys, unexpected_keys = flamingo_model.load_state_dict(fixed_state_dict, strict=False)
            logger.info(f"✓ Loaded model from epoch {ckpt.get('epoch', '?')}")
            if missing_keys:
                logger.warning(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        else:
            logger.error("✗ Checkpoint format not recognized")
            return

        flamingo_model.eval()
        logger.info("✓ Model loaded and set to eval mode")

        # Store for generation
        model_info = {
            'model': flamingo_model,
            'tokenizer': text_tokenizer,
            'device': device,
        }

    except Exception as e:
        logger.error(f"✗ Error loading model: {e}", exc_info=True)
        return

    # Load ECG-QA PTB-XL dataset (100 questions)
    logger.info("\n" + "="*60)
    logger.info("Loading ECG-QA PTB-XL dataset (100 questions)")
    logger.info("="*60)

    try:
        # Use ECGQACoTQADataset with 100 samples from test split
        from time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset

        dataset = ECGQACoTQADataset(
            split="test",
            EOS_TOKEN=text_tokenizer.eos_token,
            format_sample_str=False,
            max_samples=100,  # Limit to 100 questions
            exclude_comparison=False,
            preload_processed_data=True,
        )
        logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        logger.error(f"✗ Error loading dataset: {e}", exc_info=True)
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
    logger.info("\n" + "="*60)
    logger.info("Running zero-shot evaluation")
    logger.info("="*60)

    # Wrapper for generation
    class FlamingoWrapper:
        def __init__(self, llm, text_tokenizer, device):
            self.llm = llm
            self.text_tokenizer = text_tokenizer
            self.device = device

        def generate(self, batch, max_new_tokens=50):
            # Import OpenTSLMFlamingo for its helper methods
            from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo

            # Create temp wrapper to use OpenTSLMFlamingo's batch processing
            temp_wrapper = type('obj', (object,), {
                'text_tokenizer': self.text_tokenizer,
                'device': self.device,
                'model': self.llm
            })()

            # Prepare batch
            input_ids, images, attention_mask, _ = OpenTSLMFlamingo.pad_and_apply_batch(
                temp_wrapper, batch, include_labels=True
            )

            # Generate
            with torch.inference_mode():
                gen_ids = self.llm.generate(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.text_tokenizer.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                )

                # Remove input ids from generation
                answer_only_ids = gen_ids[:, input_ids.shape[1]:]

                return self.text_tokenizer.batch_decode(answer_only_ids, skip_special_tokens=True)

    model_wrapper = FlamingoWrapper(flamingo_model, text_tokenizer, device)

    results = []
    correct = 0
    total = 0

    logger.info("Starting evaluation loop...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Generate prediction
            try:
                predictions = model_wrapper.generate(batch, max_new_tokens=50)
                prediction = predictions[0] if isinstance(predictions, list) else predictions
                logger.debug(f"Sample {batch_idx}: Generated prediction: {prediction[:100]}...")
            except Exception as e:
                logger.error(f"Error generating prediction for sample {batch_idx}: {e}")
                prediction = "[ERROR]"
                if batch_idx == 0:  # Print full traceback for first error
                    logger.error("Full traceback for first error:", exc_info=True)

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
            }
            results.append(result)

            # Print progress every 10 samples
            if (batch_idx + 1) % 10 == 0:
                current_acc = correct / total
                logger.info(f"Progress: {batch_idx + 1}/{len(dataset)} | Accuracy: {current_acc:.2%} ({correct}/{total})")
                logger.debug(f"Last prediction: {prediction[:100]}...")
                logger.debug(f"Ground truth: {ground_truth}")

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
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{total})")

    logger.info(f"\nPer-Question-Type Accuracy:")
    for qtype, stats in sorted(question_type_stats.items()):
        logger.info(f"  {qtype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    logger.info(f"\nPer-Attribute-Type Accuracy:")
    for atype, stats in sorted(attribute_type_stats.items()):
        logger.info(f"  {atype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    # Save results
    logger.info("\n" + "="*60)
    logger.info("Saving results")
    logger.info("="*60)

    # Save detailed results as JSONL
    results_file = results_dir / "detailed_results.jsonl"
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    logger.info(f"✓ Saved detailed results to: {results_file}")

    # Save summary metrics
    metrics = {
        "model": "OpenTSLM/llama-3.2-1b-ecg-flamingo",
        "base_llm": "meta-llama/Llama-3.2-1B",
        "checkpoint": "softprompt-llama_3_2_1b-ecg.pt",
        "architecture": "OpenTSLMFlamingo (TimeSeriesFlamingoWithTrainableEncoder)",
        "max_patches": 1024,
        "dataset": "ECG-QA PTB-XL (100 questions)",
        "dataset_class": "ECGQACoTQADataset",
        "split": "test",
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
    logger.info(f"✓ Saved metrics to: {metrics_file}")

    # Save summary text
    summary_file = results_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ECG-QA PTB-XL 100-Question Evaluation Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: OpenTSLM/llama-3.2-1b-ecg-flamingo\n")
        f.write(f"Dataset: ECG-QA PTB-XL (test split, 100 questions)\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Overall Accuracy: {accuracy:.2%} ({correct}/{total})\n\n")
        f.write("Per-Question-Type Accuracy:\n")
        for qtype, stats in sorted(question_type_stats.items()):
            f.write(f"  {qtype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\n")
        f.write("\nPer-Attribute-Type Accuracy:\n")
        for atype, stats in sorted(attribute_type_stats.items()):
            f.write(f"  {atype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\n")
    logger.info(f"✓ Saved summary to: {summary_file}")

    logger.info("\n" + "="*60)
    logger.info("✅ Evaluation completed successfully!")
    logger.info("="*60)
    logger.info(f"\nResults saved to: {results_dir}")

    return accuracy, results


if __name__ == "__main__":
    evaluate_model_on_ptbxl100()
