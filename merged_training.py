#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#
"""
Merged Training Script for OpenTSLM
------------------------------------
This script trains OpenTSLM models on a merged dataset combining all 5 original TSLM datasets:
1. TSQA - Multiple choice questions about time series
2. M4 - Time series captioning
3. HAR CoT - Human activity recognition with chain-of-thought
4. SleepEDF CoT - Sleep stage classification with chain-of-thought
5. ECG-QA CoT - ECG analysis with chain-of-thought

The script:
- Merges all 5 datasets (train/val/test splits)
- Trains on merged train set
- Validates on merged validation sets (selects best model by lowest validation loss)
- Tests on merged test set
- Evaluates on TimeSeriesExam1 benchmark
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import json
import argparse
import re
from typing import List, Optional, Dict, Any, Callable
from time_series_datasets.TSQADataset import TSQADataset
from time_series_datasets.m4.M4QADataset import M4QADataset
from time_series_datasets.timeseriesexam.TimeSeriesExam1QADataset import TimeSeriesExam1QADataset
from time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
from time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
from time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from model.llm.OpenTSLMSP import OpenTSLMSP
from model.projector.MLPProjector import MLPProjector
import datetime
from logger import get_logger, set_global_verbose

from model_config import (
    BATCH_SIZE,
    EARLY_STOP_PAT,
    GRAD_CLIP_NORM,
    LR_ENCODER,
    LR_PROJECTOR,
    NUM_EPOCHS,
    PATCH_SIZE,
    WARMUP_FRAC,
    WEIGHT_DECAY,
)

# Convergence detection parameters
MIN_EPOCHS_BEFORE_STOP = 10  # Don't stop before this many epochs
RELATIVE_IMPROVEMENT_THRESHOLD = 1e-4  # Stop if relative improvement is below this for patience epochs
LOSS_SMOOTHING_WINDOW = 3  # Window size for smoothing validation loss


class ConvergenceDetector:
    """
    Robust convergence detection for training.

    Detects convergence based on:
    1. Patience: No improvement for N epochs
    2. Relative improvement: Improvement is below threshold
    3. Minimum epochs: Don't stop before min_epochs
    4. Loss smoothing: Use smoothed loss to avoid noise
    """

    def __init__(
        self,
        patience: int = EARLY_STOP_PAT,
        min_epochs: int = MIN_EPOCHS_BEFORE_STOP,
        relative_threshold: float = RELATIVE_IMPROVEMENT_THRESHOLD,
        smoothing_window: int = LOSS_SMOOTHING_WINDOW,
    ):
        self.patience = patience
        self.min_epochs = min_epochs
        self.relative_threshold = relative_threshold
        self.smoothing_window = smoothing_window

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.loss_history = []
        self.converged = False
        self.convergence_reason = None

    def _get_smoothed_loss(self) -> float:
        """Get smoothed loss using recent history."""
        if len(self.loss_history) == 0:
            return float("inf")
        window = min(self.smoothing_window, len(self.loss_history))
        return sum(self.loss_history[-window:]) / window

    def _compute_relative_improvement(self, current_loss: float) -> float:
        """Compute relative improvement from best loss."""
        if self.best_loss == float("inf") or self.best_loss == 0:
            return float("inf")
        return (self.best_loss - current_loss) / abs(self.best_loss)

    def update(self, epoch: int, val_loss: float) -> dict:
        """
        Update convergence state with new validation loss.

        Returns dict with:
            - is_best: Whether this is a new best
            - should_stop: Whether training should stop
            - reason: Reason for stopping (if applicable)
            - relative_improvement: Relative improvement from best
            - smoothed_loss: Smoothed validation loss
        """
        self.loss_history.append(val_loss)
        smoothed_loss = self._get_smoothed_loss()
        relative_improvement = self._compute_relative_improvement(val_loss)

        # Check if this is a new best (with small epsilon for numerical stability)
        is_best = val_loss + 1e-6 < self.best_loss

        if is_best:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        # Determine if we should stop
        should_stop = False
        reason = None

        # Only consider stopping after min_epochs
        if epoch >= self.min_epochs:
            # Check patience-based stopping
            if self.epochs_no_improve >= self.patience:
                should_stop = True
                reason = f"No improvement for {self.patience} epochs"

            # Check if improvements are too small (plateaued)
            elif len(self.loss_history) >= self.smoothing_window:
                recent_improvements = []
                for i in range(1, min(self.patience, len(self.loss_history))):
                    if self.loss_history[-i-1] != 0:
                        imp = (self.loss_history[-i-1] - self.loss_history[-i]) / abs(self.loss_history[-i-1])
                        recent_improvements.append(imp)

                if recent_improvements and all(abs(imp) < self.relative_threshold for imp in recent_improvements):
                    should_stop = True
                    reason = f"Loss plateaued (improvements < {self.relative_threshold:.1e})"

        if should_stop:
            self.converged = True
            self.convergence_reason = reason

        return {
            "is_best": is_best,
            "should_stop": should_stop,
            "reason": reason,
            "relative_improvement": relative_improvement,
            "smoothed_loss": smoothed_loss,
            "epochs_no_improve": self.epochs_no_improve,
        }

    def get_state(self) -> dict:
        """Get convergence detector state for checkpointing."""
        return {
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "epochs_no_improve": self.epochs_no_improve,
            "loss_history": self.loss_history,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
        }

    def load_state(self, state: dict):
        """Load convergence detector state from checkpoint."""
        self.best_loss = state.get("best_loss", float("inf"))
        self.best_epoch = state.get("best_epoch", 0)
        self.epochs_no_improve = state.get("epochs_no_improve", 0)
        self.loss_history = state.get("loss_history", [])
        self.converged = state.get("converged", False)
        self.convergence_reason = state.get("convergence_reason", None)


# Dataset classes to merge
MERGED_DATASET_CLASSES = [
    TSQADataset,
    M4QADataset,
    HARCoTQADataset,
    SleepEDFCoTQADataset,
    ECGQACoTQADataset,
]


class _UpsampledDataset(Dataset):
    """
    Wrapper dataset that upsamples a smaller dataset by repeating samples.

    This creates a virtual dataset of a target size by cycling through
    the original dataset's indices.
    """

    def __init__(self, original_dataset: Dataset, target_size: int):
        """
        Args:
            original_dataset: The original dataset to upsample
            target_size: The desired size of the upsampled dataset
        """
        self.original_dataset = original_dataset
        self.target_size = target_size
        self.original_size = len(original_dataset)

    def __len__(self) -> int:
        return self.target_size

    def __getitem__(self, idx: int):
        # Map index to original dataset using modulo
        original_idx = idx % self.original_size
        return self.original_dataset[original_idx]


class MergedTrainer:
    """
    Trainer for merged dataset training.
    Trains on all 5 TSLM datasets combined, then evaluates on TimeSeriesExam1.
    """

    def _sanitize_llm_id(self, llm_id: str) -> str:
        """Sanitize llm_id for use in directory names."""
        if not llm_id:
            return "unknown_llm"
        name = llm_id.split("/")[-1]
        name = name.replace(".", "_").replace("-", "_")
        while "__" in name:
            name = name.replace("__", "_")
        return name

    def __init__(
        self,
        model_type: str,
        device: str = None,
        gradient_checkpointing: bool = False,
        dist_url: str = "env://",
        dist_backend: str = "nccl",
        local_rank: int = int(os.environ.get("LOCAL_RANK", 0)),
        llm_id: str = None,
    ):
        """
        Initialize the merged trainer.

        Args:
            model_type: Either 'OpenTSLMSP' or 'OpenTSLMFlamingo'
            device: Device to use for training ('cuda', 'mps', or 'cpu')
            gradient_checkpointing: Enable gradient checkpointing
            dist_url: URL used to set up distributed training
            dist_backend: Distributed backend
            local_rank: Local GPU rank
            llm_id: LLM model ID
        """
        self.model_type = model_type
        self.device = device or self._get_device()
        if self.device == "mps":
            print("Warning: Using MPS, might not be fully compatible. Use CUDA for best results.")
        self.llm_id = llm_id
        self.llm_id_safe = self._sanitize_llm_id(llm_id)

        # Distributed training parameters
        self.gradient_checkpointing = gradient_checkpointing
        self.dist_url = dist_url
        self.dist_backend = dist_backend
        self.local_rank = local_rank

        # Initialize distributed training if needed
        self.rank = 0
        self.world_size = 1
        if self._should_use_distributed():
            self._init_distributed()

        self.model = self._initialize_model()
        self.results_dir = os.path.join("results_merged", self.llm_id_safe, self.model_type)
        self._create_results_dir()

    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _initialize_model(self):
        """Initialize the specified model type."""
        if self.model_type == "OpenTSLMSP":
            model = OpenTSLMSP(llm_id=self.llm_id, device=self.device).to(self.device)
        elif self.model_type == "OpenTSLMFlamingo":
            model = OpenTSLMFlamingo(
                cross_attn_every_n_layers=1,
                gradient_checkpointing=self.gradient_checkpointing,
                llm_id=self.llm_id,
                device=self.device,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Use DDP for multi-GPU training
        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            )
            if self.rank == 0:
                print(f"Wrapped {self.model_type} with DDP for distributed training")

        return model

    def _create_results_dir(self):
        """Create the results directory structure."""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "results"), exist_ok=True)

    def _get_optimizer(
        self,
        batch_size: int = None,
        lr_encoder: float = None,
        lr_projector: float = None,
        lr_base: float = None,
    ):
        """Get optimizer for the model with configurable learning rates."""
        model = self._get_model()

        if self.model_type == "OpenTSLMSP":
            enc_params = list(model.encoder.parameters())
            proj_params = list(model.projector.projector.parameters())

            encoder_lr = lr_encoder if lr_encoder is not None else LR_ENCODER
            projector_lr = lr_projector if lr_projector is not None else LR_PROJECTOR

            param_groups = [
                {"params": enc_params, "lr": encoder_lr, "weight_decay": WEIGHT_DECAY},
                {"params": proj_params, "lr": projector_lr, "weight_decay": WEIGHT_DECAY},
            ]

            # Add LoRA parameters if enabled
            if hasattr(model, "lora_enabled") and model.lora_enabled:
                lora_params = model.get_lora_parameters()
                if lora_params:
                    param_groups.append({
                        "params": lora_params,
                        "lr": projector_lr,
                        "weight_decay": WEIGHT_DECAY,
                    })
                    if self.rank == 0:
                        print(f"Learning rates for {self.model_type} (with LoRA):")
                        print(f"   Encoder LR: {encoder_lr:.2e}")
                        print(f"   Projector LR: {projector_lr:.2e}")
                        print(f"   LoRA LR: {projector_lr:.2e} ({len(lora_params)} parameters)")
            else:
                if self.rank == 0:
                    print(f"Learning rates for {self.model_type}:")
                    print(f"   Encoder LR: {encoder_lr:.2e}")
                    print(f"   Projector LR: {projector_lr:.2e}")

            return AdamW(param_groups)
        else:
            # For Flamingo
            params_to_optimize = model.named_parameters()
            params_to_optimize = list(
                filter(
                    lambda x: x[1].requires_grad
                    and not getattr(x[1], "exclude_from_optimizer", False),
                    params_to_optimize,
                )
            )

            params_with_wd, params_without_wd = [], []
            for n, p in params_to_optimize:
                if "gated_cross_attn" in n:
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)

            base_lr = lr_base if lr_base is not None else 2e-4

            if self.rank == 0:
                print(f"Learning rate for {self.model_type}: Base LR: {base_lr:.2e}")

            return torch.optim.AdamW(
                [
                    {"params": params_with_wd, "weight_decay": 0.1},
                    {"params": params_without_wd, "weight_decay": 0.0},
                ],
                lr=base_lr,
            )

    def _merge_data_loaders(
        self,
        datasets: List[Dataset],
        shuffle: bool,
        batch_size: int,
        patch_size: int,
        distribute_data: bool = False,
    ) -> DataLoader:
        """Create a merged data loader from multiple datasets."""
        merged_ds = ConcatDataset(datasets)

        if distribute_data and dist.is_initialized():
            sampler = DistributedSampler(
                merged_ds, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle
            )
            return DataLoader(
                merged_ds,
                sampler=sampler,
                batch_size=batch_size,
                collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=patch_size
                ),
            )
        else:
            return DataLoader(
                merged_ds,
                shuffle=shuffle,
                batch_size=batch_size,
                collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=patch_size
                ),
            )

    def _balance_datasets(
        self,
        datasets: List[Dataset],
        dataset_names: List[str] = None,
        target_size: int = None,
    ) -> List[Dataset]:
        """
        Balance datasets by upsampling smaller ones to match target size.

        Args:
            datasets: List of datasets to balance
            dataset_names: Optional names for logging
            target_size: Target size for each dataset. If None, uses max size.

        Returns:
            List of balanced datasets (original datasets are not modified)
        """
        if not datasets:
            return datasets

        # Get original sizes
        original_sizes = [len(ds) for ds in datasets]

        # Determine target size
        if target_size is None:
            target_size = max(original_sizes)

        if self.rank == 0:
            print(f"\nBalancing datasets to target size: {target_size}")
            for i, size in enumerate(original_sizes):
                name = dataset_names[i] if dataset_names else f"Dataset {i}"
                ratio = target_size / size if size > 0 else 0
                print(f"  {name}: {size} -> {target_size} (upsample {ratio:.2f}x)")

        balanced_datasets = []
        for i, ds in enumerate(datasets):
            original_size = len(ds)

            if original_size >= target_size:
                # Dataset is already large enough, use as-is
                balanced_datasets.append(ds)
            else:
                # Upsample by creating a wrapper dataset that repeats samples
                balanced_ds = _UpsampledDataset(ds, target_size)
                balanced_datasets.append(balanced_ds)

        return balanced_datasets

    def _save_checkpoint(self, epoch: int, val_loss: float, optimizer, scheduler):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.results_dir, "checkpoints")

        if dist.is_initialized() and self.rank != 0:
            return

        model = self._get_model()

        if self.model_type == "OpenTSLMSP":
            checkpoint = {
                "encoder_state": model.encoder.state_dict(),
                "projector_state": model.projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            }
            model.save_lora_state_to_checkpoint(checkpoint)
        else:
            model_state = model.state_dict()
            if hasattr(self.model, "module"):
                model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
            checkpoint = {
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            }

        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(checkpoint, checkpoint_path)
        if self.rank == 0:
            print(f"Saved checkpoint to {checkpoint_path}")

    def _save_loss_history(self, epoch: int, train_loss: float, val_loss: float):
        """Save loss history to a file."""
        if dist.is_initialized() and self.rank != 0:
            return

        checkpoint_dir = os.path.join(self.results_dir, "checkpoints")
        loss_history_file = os.path.join(checkpoint_dir, "loss_history.txt")

        os.makedirs(checkpoint_dir, exist_ok=True)

        if not os.path.exists(loss_history_file):
            with open(loss_history_file, "w") as f:
                f.write("Epoch\tTrain_Loss\tVal_Loss\n")
                f.write("-" * 30 + "\n")

        with open(loss_history_file, "a") as f:
            f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")

    def _load_checkpoint(self, optimizer, scheduler, eval_only: bool = False):
        """Load model checkpoint."""
        checkpoint_path = os.path.join(self.results_dir, "checkpoints", "best_model.pt")

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = self._get_model()

            if self.model_type == "OpenTSLMSP":
                model.encoder.load_state_dict(checkpoint["encoder_state"])
                model.projector.load_state_dict(checkpoint["projector_state"])
                try:
                    model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)
                except RuntimeError as e:
                    if self.rank == 0:
                        print(f"Failed to load LoRA state: {e}")
                    raise

                if not eval_only and optimizer is not None and "optimizer_state" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
            else:
                model_state = checkpoint["model_state"]
                if hasattr(self.model, "module"):
                    model_state = {f"module.{k}": v for k, v in model_state.items()}
                self.model.load_state_dict(model_state, strict=False)

                if not eval_only and optimizer is not None and "optimizer_state" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])

            if not eval_only and scheduler is not None and "scheduler_state" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state"])

            return checkpoint.get("epoch", "?"), checkpoint.get("val_loss", float("inf"))
        return None, float("inf")

    def _is_mcq_gold(self, gold: str) -> bool:
        """
        Check if this is an MCQ task based on gold answer format.
        MCQ tasks have gold answers starting with option letters like "(a)", "(b)", etc.
        """
        cleaned = gold.replace("<|end_of_text|>", "").strip()
        return bool(re.match(r'^\([a-h]\)', cleaned, re.IGNORECASE))

    def _calculate_accuracy_baseline(self, gold: str, prediction: str) -> int:
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

    def _calculate_accuracy(self, predictions: List[str], gold_answers: List[str]) -> float:
        """Calculate accuracy for MCQ tasks using original baseline logic."""
        correct = 0
        total = len(predictions)

        for pred, gold in zip(predictions, gold_answers):
            correct += self._calculate_accuracy_baseline(gold, pred)

        return correct / total if total > 0 else 0.0

    def _is_cot_gold(self, gold: str) -> bool:
        """
        Check if this is a CoT task based on gold answer containing "Answer:" pattern.
        """
        return "Answer:" in gold and not self._is_mcq_gold(gold)

    def _extract_cot_label(self, text: str) -> str:
        """
        Extract the label after "Answer:" from CoT response.
        Following the original evaluate_har.py and evaluate_sleep_cot.py logic.
        """
        if text is None:
            return ""

        text = text.strip().replace("<|end_of_text|>", "").strip()

        # Find the last occurrence of 'Answer:' (case-insensitive)
        matches = list(re.finditer(r'answer:\s*', text, re.IGNORECASE))
        if matches:
            start = matches[-1].end()
            label = text[start:].strip()
        else:
            words = text.split()
            label = words[-1] if words else ""

        # Remove trailing punctuation
        label = re.sub(r'[\.,;:!?]+$', '', label)
        return label.lower().strip()

    def _calculate_cot_accuracy(self, gold: str, prediction: str) -> int:
        """Evaluate CoT task by comparing extracted labels."""
        gold_label = self._extract_cot_label(gold)
        pred_label = self._extract_cot_label(prediction)
        return int(gold_label == pred_label)

    def _calculate_merged_accuracy(
        self, predictions: List[str], gold_answers: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate accuracy for merged test set.

        Evaluates:
        - MCQ samples: First 3 chars comparison (original TSQA baseline)
        - CoT samples: Extract label after "Answer:" and compare
        - Captioning samples: Counted but no accuracy metric
        """
        mcq_correct = 0
        mcq_total = 0
        cot_correct = 0
        cot_total = 0
        captioning_total = 0

        for pred, gold in zip(predictions, gold_answers):
            if self._is_mcq_gold(gold):
                mcq_total += 1
                mcq_correct += self._calculate_accuracy_baseline(gold, pred)
            elif self._is_cot_gold(gold):
                cot_total += 1
                cot_correct += self._calculate_cot_accuracy(gold, pred)
            else:
                captioning_total += 1

        mcq_accuracy = mcq_correct / mcq_total if mcq_total > 0 else 0.0
        cot_accuracy = cot_correct / cot_total if cot_total > 0 else 0.0

        total_eval = mcq_total + cot_total
        total_correct = mcq_correct + cot_correct
        overall_accuracy = total_correct / total_eval if total_eval > 0 else 0.0

        return {
            "mcq_accuracy": mcq_accuracy,
            "mcq_correct": mcq_correct,
            "mcq_total": mcq_total,
            "cot_accuracy": cot_accuracy,
            "cot_correct": cot_correct,
            "cot_total": cot_total,
            "overall_accuracy": overall_accuracy,
            "captioning_total": captioning_total,
        }

    def _evaluate(
        self,
        test_loader: DataLoader,
        stage_name: str,
        metric_func: Callable = None,
        epoch: int = None,
    ) -> Dict[str, Any]:
        """Evaluate model on test set."""
        self.model.eval()
        results = []

        max_new_tokens = 2000

        results_file_rank = os.path.join(
            self.results_dir, "results",
            f"test_predictions_rank_{self.rank if dist.is_initialized() else 0}.jsonl",
        )
        final_results_file = os.path.join(self.results_dir, "results", f"{stage_name}_predictions.jsonl")

        os.makedirs(os.path.dirname(results_file_rank), exist_ok=True)

        if self.rank == 0:
            print(f"[Eval] Evaluating {stage_name}...")

        results_fp = open(results_file_rank, "w", encoding="utf-8")
        try:
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Evaluating {stage_name}", disable=self.rank != 0):
                    predictions = self._get_model().generate(batch, max_new_tokens=max_new_tokens)

                    for sample, pred in zip(batch, predictions):
                        result = {
                            "pre_prompt": sample["pre_prompt"],
                            "time_series_text": sample["time_series_text"],
                            "post_prompt": sample["post_prompt"],
                            "generated": pred,
                            "gold": sample["answer"],
                        }
                        results.append(result)
                        results_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                        results_fp.flush()
        finally:
            results_fp.close()

        if dist.is_initialized():
            dist.barrier()

        # Merge per-rank files
        if (not dist.is_initialized()) or (self.rank == 0):
            with open(final_results_file, "w", encoding="utf-8") as merged_fp:
                num_ranks = self.world_size if dist.is_initialized() else 1
                for r in range(num_ranks):
                    part_file = os.path.join(
                        self.results_dir, "results",
                        f"test_predictions_rank_{r}.jsonl",
                    )
                    if os.path.exists(part_file):
                        with open(part_file, "r", encoding="utf-8") as pf:
                            for line in pf:
                                merged_fp.write(line)
            if self.rank == 0:
                print(f"Merged predictions saved to: {final_results_file}")

        metrics = {"test_loss": float("nan")}
        if epoch is not None:
            metrics["epoch"] = epoch

        if metric_func and ((not dist.is_initialized()) or (self.rank == 0)):
            predictions = []
            gold_answers = []
            with open(final_results_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        predictions.append(obj.get("generated", ""))
                        gold_answers.append(obj.get("gold", ""))
                    except Exception:
                        continue
            additional_metrics = metric_func(predictions, gold_answers)
            metrics.update(additional_metrics)

        # Save metrics
        if (not dist.is_initialized()) or (self.rank == 0):
            metrics_file = os.path.join(self.results_dir, "results", f"{stage_name}_metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"Evaluation complete for {stage_name}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.4f}")
                else:
                    print(f"   {metric}: {value}")

        if dist.is_initialized():
            dist.barrier()

        return metrics

    def _get_model(self):
        """Get the underlying model (handles DDP wrapping)."""
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def _should_use_distributed(self) -> bool:
        """Check if distributed training should be used."""
        return ("WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1) or (
            "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) >= 0
        )

    def _init_distributed(self):
        """Initialize distributed training."""
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
        elif "LOCAL_RANK" in os.environ:
            self.rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(
            backend=self.dist_backend,
            init_method=self.dist_url,
            world_size=self.world_size,
            rank=self.rank,
            timeout=datetime.timedelta(hours=999),
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)

        if self.rank == 0:
            print(f"Initialized distributed training with {self.world_size} GPUs")

    def train_merged(
        self,
        num_epochs: int = 30,
        batch_size: int = None,
        lr_encoder: float = 2e-4,
        lr_projector: float = 1e-4,
        lr_base: float = 2e-4,
        eval_only: bool = False,
        max_samples_per_dataset: int = None,
    ) -> Dict[str, Any]:
        """
        Train on merged dataset from all 5 TSLM datasets.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size per GPU
            lr_encoder: Learning rate for encoder
            lr_projector: Learning rate for projector
            lr_base: Base learning rate for Flamingo
            eval_only: Skip training, only run evaluation
            max_samples_per_dataset: Max samples per dataset (for sanity checks)

        Returns:
            Dictionary with training and evaluation metrics
        """
        if batch_size is None:
            batch_size = BATCH_SIZE

        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"Starting Merged Training with {self.model_type}")
            print(f"{'='*60}")
            print(f"Epochs: {num_epochs}")
            print(f"Batch size per GPU: {batch_size}")
            if self.world_size > 1:
                print(f"Effective batch size: {batch_size * self.world_size}")
            if max_samples_per_dataset:
                print(f"Max samples per dataset: {max_samples_per_dataset}")
            print()

        # Create merged datasets
        eos_token = self._get_model().get_eos_token()

        if self.rank == 0:
            print("Loading and merging datasets...")

        train_datasets = []
        val_datasets = []
        test_datasets = []

        for dataset_class in MERGED_DATASET_CLASSES:
            if self.rank == 0:
                print(f"  Loading {dataset_class.__name__}...")

            train_ds = dataset_class("train", EOS_TOKEN=eos_token)
            val_ds = dataset_class("validation", EOS_TOKEN=eos_token)
            test_ds = dataset_class("test", EOS_TOKEN=eos_token)

            # Limit samples if specified (for sanity checks)
            if max_samples_per_dataset:
                train_ds.dataset = train_ds.dataset[:max_samples_per_dataset]
                val_ds.dataset = val_ds.dataset[:max_samples_per_dataset]
                test_ds.dataset = test_ds.dataset[:max_samples_per_dataset]

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)

            if self.rank == 0:
                print(f"    Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

        # Create data loaders
        train_loader = self._merge_data_loaders(
            train_datasets,
            shuffle=True,
            batch_size=batch_size,
            patch_size=PATCH_SIZE,
            distribute_data=self.world_size > 1,
        )

        val_loader = self._merge_data_loaders(
            val_datasets,
            shuffle=False,
            batch_size=1,
            patch_size=PATCH_SIZE,
            distribute_data=False,
        )

        test_loader = self._merge_data_loaders(
            test_datasets,
            shuffle=False,
            batch_size=1,
            patch_size=PATCH_SIZE,
            distribute_data=self.world_size > 1,
        )

        if self.rank == 0:
            total_train = sum(len(ds) for ds in train_datasets)
            total_val = sum(len(ds) for ds in val_datasets)
            total_test = sum(len(ds) for ds in test_datasets)
            print(f"\nMerged dataset sizes:")
            print(f"  Total Train: {total_train}")
            print(f"  Total Val: {total_val}")
            print(f"  Total Test: {total_test}")
            print()

        # Initialize optimizer
        optimizer = self._get_optimizer(batch_size, lr_encoder, lr_projector, lr_base)

        # Scheduler
        total_steps = num_epochs * len(train_loader)
        warmup_steps = int(WARMUP_FRAC * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        if self.rank == 0:
            print(f"Total training steps: {total_steps}")
            print(f"Warmup steps: {warmup_steps}")

        # Load checkpoint if exists
        best_epoch, best_val_loss = self._load_checkpoint(optimizer, scheduler, eval_only=eval_only)
        if best_epoch is not None:
            if self.rank == 0:
                print(f"Resuming from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
        else:
            if self.rank == 0:
                print("Starting fresh training")
            best_val_loss = float("inf")

        # Skip training if eval_only
        if eval_only:
            if self.rank == 0:
                print("Skipping training (eval_only mode)")
            epoch = best_epoch
        else:
            # Initialize convergence detector
            convergence = ConvergenceDetector(
                patience=EARLY_STOP_PAT,
                min_epochs=MIN_EPOCHS_BEFORE_STOP,
                relative_threshold=RELATIVE_IMPROVEMENT_THRESHOLD,
                smoothing_window=LOSS_SMOOTHING_WINDOW,
            )

            # Initialize from checkpoint if resuming
            if best_epoch is not None:
                convergence.best_loss = best_val_loss
                convergence.best_epoch = best_epoch

            start_epoch = best_epoch + 1 if best_epoch is not None else 1

            if self.rank == 0:
                print(f"\nConvergence settings:")
                print(f"  Min epochs before stopping: {MIN_EPOCHS_BEFORE_STOP}")
                print(f"  Patience: {EARLY_STOP_PAT} epochs")
                print(f"  Relative improvement threshold: {RELATIVE_IMPROVEMENT_THRESHOLD:.1e}")
                print()

            for epoch in range(start_epoch, num_epochs + 1):
                if hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(epoch)

                # Training
                self.model.train()
                running_loss = 0.0
                prog = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch}/{num_epochs}",
                    disable=self.rank != 0,
                )

                for i, batch in enumerate(prog):
                    optimizer.zero_grad()
                    loss = self._get_model().compute_loss(batch)
                    loss.backward()
                    clip_grad_norm_(self._get_model().parameters(), GRAD_CLIP_NORM)
                    optimizer.step()
                    scheduler.step()

                    running_loss += loss.item()
                    if self.rank == 0:
                        prog.set_postfix(
                            loss=f"{loss.item():.4f}",
                            lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        )

                avg_train_loss = running_loss / len(train_loader)
                if self.rank == 0:
                    tqdm.write(f"Epoch {epoch} - train loss: {avg_train_loss:.4f}")

                # Validation
                val_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validating", disable=self.rank != 0):
                        val_loss += self._get_model().compute_loss(batch).item()

                avg_val_loss = val_loss / len(val_loader)

                # Synchronize validation loss across all ranks (use average, not sum)
                if dist.is_initialized():
                    val_loss_tensor = torch.tensor(avg_val_loss, device=self.device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    avg_val_loss = val_loss_tensor.item()

                if self.rank == 0:
                    tqdm.write(f"Epoch {epoch} - val loss: {avg_val_loss:.4f}")
                    tqdm.write(f"Epoch {epoch} - best loss: {convergence.best_loss:.4f}")

                self._save_loss_history(epoch, avg_train_loss, avg_val_loss)

                # Update convergence detector
                conv_result = convergence.update(epoch, avg_val_loss)

                # Synchronize convergence state across ranks
                if dist.is_initialized():
                    is_best_tensor = torch.tensor(1 if conv_result["is_best"] else 0, device=self.device)
                    should_stop_tensor = torch.tensor(1 if conv_result["should_stop"] else 0, device=self.device)
                    dist.broadcast(is_best_tensor, src=0)
                    dist.broadcast(should_stop_tensor, src=0)
                    is_best = is_best_tensor.item() > 0
                    should_stop = should_stop_tensor.item() > 0
                else:
                    is_best = conv_result["is_best"]
                    should_stop = conv_result["should_stop"]

                if is_best:
                    self._save_checkpoint(epoch, avg_val_loss, optimizer, scheduler)
                    if self.rank == 0:
                        tqdm.write("New best model saved.\n")
                else:
                    if self.rank == 0:
                        tqdm.write(f"No improvement for {conv_result['epochs_no_improve']}/{EARLY_STOP_PAT} epochs.\n")

                if should_stop:
                    if self.rank == 0:
                        tqdm.write(f"\nConverged after {epoch} epochs: {conv_result['reason']}")
                    break

                # Synchronize best loss for consistent state
                if dist.is_initialized():
                    best_loss_tensor = torch.tensor(convergence.best_loss, device=self.device)
                    dist.broadcast(best_loss_tensor, src=0)
                    convergence.best_loss = best_loss_tensor.item()

            # Log final convergence state
            if self.rank == 0:
                if convergence.converged:
                    print(f"\nTraining converged: {convergence.convergence_reason}")
                else:
                    print(f"\nTraining completed max epochs ({num_epochs})")
                print(f"Best validation loss: {convergence.best_loss:.4f} at epoch {convergence.best_epoch}")

            best_val_loss = convergence.best_loss

        # Load best model and evaluate
        best_epoch, _ = self._load_checkpoint(optimizer, scheduler)
        if best_epoch is not None and self.rank == 0:
            print(f"Loaded best checkpoint from epoch {best_epoch} for evaluation.")

        # Evaluate on merged test set
        if self.rank == 0:
            print(f"\n{'='*60}")
            print("Evaluating on merged test set...")
            print(f"{'='*60}")

        merged_metrics = self._evaluate(
            test_loader,
            "merged_test",
            metric_func=self._calculate_merged_accuracy,
            epoch=best_epoch,
        )

        # Evaluate on TimeSeriesExam1
        if self.rank == 0:
            print(f"\n{'='*60}")
            print("Evaluating on TimeSeriesExam1 benchmark...")
            print(f"{'='*60}")

        tsexam_test = TimeSeriesExam1QADataset("test", EOS_TOKEN=eos_token)
        if max_samples_per_dataset:
            tsexam_test.dataset = tsexam_test.dataset[:max_samples_per_dataset]

        tsexam_loader = self._merge_data_loaders(
            [tsexam_test],
            shuffle=False,
            batch_size=1,
            patch_size=PATCH_SIZE,
            distribute_data=self.world_size > 1,
        )

        tsexam_metrics = self._evaluate(
            tsexam_loader,
            "timeseriesexam1",
            metric_func=lambda preds, golds: {"accuracy": self._calculate_accuracy(preds, golds)},
            epoch=best_epoch,
        )

        # Combine results
        results = {
            "merged_test": merged_metrics,
            "timeseriesexam1": tsexam_metrics,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }

        # Save overall results
        if self.rank == 0:
            results_file = os.path.join(self.results_dir, "results", "final_results.json")
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\n{'='*60}")
            print("Training Complete!")
            print(f"{'='*60}")
            print(f"Results saved to: {self.results_dir}/results/")
            print(f"Best epoch: {best_epoch}")
            print(f"Best val loss: {best_val_loss:.4f}")
            if "mcq_accuracy" in merged_metrics:
                print(f"Merged Test MCQ Accuracy: {merged_metrics['mcq_accuracy']:.4f} "
                      f"({merged_metrics['mcq_correct']}/{merged_metrics['mcq_total']} MCQ samples)")
            if "cot_accuracy" in merged_metrics:
                print(f"Merged Test CoT Accuracy: {merged_metrics['cot_accuracy']:.4f} "
                      f"({merged_metrics['cot_correct']}/{merged_metrics['cot_total']} CoT samples)")
            if "overall_accuracy" in merged_metrics:
                print(f"Merged Test Overall Accuracy: {merged_metrics['overall_accuracy']:.4f}")
            if "accuracy" in tsexam_metrics:
                print(f"TimeSeriesExam1 Accuracy: {tsexam_metrics['accuracy']:.4f}")

        return results

    def train_merged_balanced(
        self,
        num_epochs: int = 30,
        batch_size: int = None,
        lr_encoder: float = 2e-4,
        lr_projector: float = 1e-4,
        lr_base: float = 2e-4,
        eval_only: bool = False,
        max_samples_per_dataset: int = None,
        balance_strategy: str = "max",
    ) -> Dict[str, Any]:
        """
        Train on merged dataset with balanced dataset proportions.

        This version upsamples smaller datasets to match the largest one,
        addressing dataset imbalance issues (e.g., SleepEDF being only 2% of data).

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size per GPU
            lr_encoder: Learning rate for encoder
            lr_projector: Learning rate for projector
            lr_base: Base learning rate for Flamingo
            eval_only: Skip training, only run evaluation
            max_samples_per_dataset: Max samples per dataset (for sanity checks)
            balance_strategy: "max" to match largest dataset, or int for specific target size

        Returns:
            Dictionary with training and evaluation metrics
        """
        if batch_size is None:
            batch_size = BATCH_SIZE

        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"Starting BALANCED Merged Training with {self.model_type}")
            print(f"{'='*60}")
            print(f"Epochs: {num_epochs}")
            print(f"Batch size per GPU: {batch_size}")
            print(f"Balance strategy: {balance_strategy}")
            if self.world_size > 1:
                print(f"Effective batch size: {batch_size * self.world_size}")
            if max_samples_per_dataset:
                print(f"Max samples per dataset: {max_samples_per_dataset}")
            print()

        # Create merged datasets
        eos_token = self._get_model().get_eos_token()

        if self.rank == 0:
            print("Loading datasets...")

        train_datasets = []
        val_datasets = []
        test_datasets = []
        dataset_names = []

        for dataset_class in MERGED_DATASET_CLASSES:
            if self.rank == 0:
                print(f"  Loading {dataset_class.__name__}...")

            train_ds = dataset_class("train", EOS_TOKEN=eos_token)
            val_ds = dataset_class("validation", EOS_TOKEN=eos_token)
            test_ds = dataset_class("test", EOS_TOKEN=eos_token)

            # Limit samples if specified (for sanity checks)
            if max_samples_per_dataset:
                train_ds.dataset = train_ds.dataset[:max_samples_per_dataset]
                val_ds.dataset = val_ds.dataset[:max_samples_per_dataset]
                test_ds.dataset = test_ds.dataset[:max_samples_per_dataset]

            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)
            dataset_names.append(dataset_class.__name__)

            if self.rank == 0:
                print(f"    Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

        # Show original dataset proportions
        if self.rank == 0:
            total_original = sum(len(ds) for ds in train_datasets)
            print(f"\nOriginal dataset proportions (train):")
            for name, ds in zip(dataset_names, train_datasets):
                pct = 100 * len(ds) / total_original
                print(f"  {name}: {len(ds)} ({pct:.1f}%)")

        # Determine target size for balancing
        if balance_strategy == "max":
            target_size = None  # Will use max size
        else:
            target_size = int(balance_strategy)

        # Balance training datasets
        if self.rank == 0:
            print(f"\nBalancing training datasets...")
        balanced_train = self._balance_datasets(train_datasets, dataset_names, target_size)

        # Balance validation datasets (proportionally)
        if self.rank == 0:
            print(f"\nBalancing validation datasets...")
        balanced_val = self._balance_datasets(val_datasets, dataset_names, target_size)

        # Test datasets remain unbalanced for fair evaluation
        if self.rank == 0:
            print(f"\nTest datasets remain unbalanced for fair evaluation.")

        # Create data loaders
        train_loader = self._merge_data_loaders(
            balanced_train,
            shuffle=True,
            batch_size=batch_size,
            patch_size=PATCH_SIZE,
            distribute_data=self.world_size > 1,
        )

        val_loader = self._merge_data_loaders(
            balanced_val,
            shuffle=False,
            batch_size=1,
            patch_size=PATCH_SIZE,
            distribute_data=False,
        )

        test_loader = self._merge_data_loaders(
            test_datasets,
            shuffle=False,
            batch_size=1,
            patch_size=PATCH_SIZE,
            distribute_data=self.world_size > 1,
        )

        if self.rank == 0:
            total_train = sum(len(ds) for ds in balanced_train)
            total_val = sum(len(ds) for ds in balanced_val)
            total_test = sum(len(ds) for ds in test_datasets)
            print(f"\nBalanced dataset sizes:")
            print(f"  Total Train: {total_train} (balanced)")
            print(f"  Total Val: {total_val} (balanced)")
            print(f"  Total Test: {total_test} (original, for fair evaluation)")
            print()

        # Initialize optimizer
        optimizer = self._get_optimizer(batch_size, lr_encoder, lr_projector, lr_base)

        # Scheduler
        total_steps = num_epochs * len(train_loader)
        warmup_steps = int(WARMUP_FRAC * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        if self.rank == 0:
            print(f"Total training steps: {total_steps}")
            print(f"Warmup steps: {warmup_steps}")

        # Load checkpoint if exists
        best_epoch, best_val_loss = self._load_checkpoint(optimizer, scheduler, eval_only=eval_only)
        if best_epoch is not None:
            if self.rank == 0:
                print(f"Resuming from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
        else:
            if self.rank == 0:
                print("Starting fresh training")
            best_val_loss = float("inf")

        # Skip training if eval_only
        if eval_only:
            if self.rank == 0:
                print("Skipping training (eval_only mode)")
            epoch = best_epoch
        else:
            # Initialize convergence detector
            convergence = ConvergenceDetector(
                patience=EARLY_STOP_PAT,
                min_epochs=MIN_EPOCHS_BEFORE_STOP,
                relative_threshold=RELATIVE_IMPROVEMENT_THRESHOLD,
                smoothing_window=LOSS_SMOOTHING_WINDOW,
            )

            # Initialize from checkpoint if resuming
            if best_epoch is not None:
                convergence.best_loss = best_val_loss
                convergence.best_epoch = best_epoch
                start_epoch = best_epoch + 1
            else:
                start_epoch = 1

            # Training loop
            for epoch in range(start_epoch, num_epochs + 1):
                # Training
                avg_train_loss = self._train_epoch(train_loader, optimizer, scheduler, epoch)

                # Validation
                avg_val_loss = self._validate(val_loader, epoch)

                # Record in history
                self._save_loss_history(epoch, avg_train_loss, avg_val_loss)

                # Check convergence (only on rank 0)
                if self.rank == 0:
                    conv_result = convergence.check(avg_val_loss, epoch)
                    # Broadcast to other processes
                    if dist.is_initialized():
                        conv_tensor = torch.tensor(
                            [float(conv_result["is_best"]), float(conv_result["should_stop"])],
                            device=self.device,
                        )
                        dist.broadcast(conv_tensor, src=0)
                    is_best = conv_result["is_best"]
                    should_stop = conv_result["should_stop"]
                else:
                    if dist.is_initialized():
                        conv_tensor = torch.tensor([0.0, 0.0], device=self.device)
                        dist.broadcast(conv_tensor, src=0)
                        is_best = conv_tensor[0].item() > 0.5
                        should_stop = conv_tensor[1].item() > 0.5
                    else:
                        is_best = conv_result["is_best"]
                        should_stop = conv_result["should_stop"]

                if is_best:
                    self._save_checkpoint(epoch, avg_val_loss, optimizer, scheduler)
                    if self.rank == 0:
                        tqdm.write("New best model saved.\n")
                else:
                    if self.rank == 0:
                        tqdm.write(f"No improvement for {conv_result['epochs_no_improve']}/{EARLY_STOP_PAT} epochs.\n")

                if should_stop:
                    if self.rank == 0:
                        tqdm.write(f"\nConverged after {epoch} epochs: {conv_result['reason']}")
                    break

                # Synchronize best loss for consistent state
                if dist.is_initialized():
                    best_loss_tensor = torch.tensor(convergence.best_loss, device=self.device)
                    dist.broadcast(best_loss_tensor, src=0)
                    convergence.best_loss = best_loss_tensor.item()

            # Log final convergence state
            if self.rank == 0:
                if convergence.converged:
                    print(f"\nTraining converged: {convergence.convergence_reason}")
                else:
                    print(f"\nTraining completed max epochs ({num_epochs})")
                print(f"Best validation loss: {convergence.best_loss:.4f} at epoch {convergence.best_epoch}")

            best_val_loss = convergence.best_loss

        # Load best model and evaluate
        best_epoch, _ = self._load_checkpoint(optimizer, scheduler)
        if best_epoch is not None and self.rank == 0:
            print(f"Loaded best checkpoint from epoch {best_epoch} for evaluation.")

        # Evaluate on merged test set (original, unbalanced for fair evaluation)
        if self.rank == 0:
            print(f"\n{'='*60}")
            print("Evaluating on merged test set (original proportions)...")
            print(f"{'='*60}")

        merged_metrics = self._evaluate(
            test_loader,
            "merged_test",
            metric_func=self._calculate_merged_accuracy,
            epoch=best_epoch,
        )

        # Evaluate on TimeSeriesExam1
        if self.rank == 0:
            print(f"\n{'='*60}")
            print("Evaluating on TimeSeriesExam1 benchmark...")
            print(f"{'='*60}")

        tsexam_test = TimeSeriesExam1QADataset("test", EOS_TOKEN=eos_token)
        if max_samples_per_dataset:
            tsexam_test.dataset = tsexam_test.dataset[:max_samples_per_dataset]

        tsexam_loader = self._merge_data_loaders(
            [tsexam_test],
            shuffle=False,
            batch_size=1,
            patch_size=PATCH_SIZE,
            distribute_data=self.world_size > 1,
        )

        tsexam_metrics = self._evaluate(
            tsexam_loader,
            "timeseriesexam1",
            metric_func=lambda preds, golds: {"accuracy": self._calculate_accuracy(preds, golds)},
            epoch=best_epoch,
        )

        # Combine results
        results = {
            "merged_test": merged_metrics,
            "timeseriesexam1": tsexam_metrics,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "balanced_training": True,
            "balance_strategy": balance_strategy,
        }

        # Save overall results
        if self.rank == 0:
            results_file = os.path.join(self.results_dir, "results", "final_results_balanced.json")
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\n{'='*60}")
            print("Balanced Training Complete!")
            print(f"{'='*60}")
            print(f"Results saved to: {self.results_dir}/results/")
            print(f"Best epoch: {best_epoch}")
            print(f"Best val loss: {best_val_loss:.4f}")
            if "mcq_accuracy" in merged_metrics:
                print(f"Merged Test MCQ Accuracy: {merged_metrics['mcq_accuracy']:.4f} "
                      f"({merged_metrics['mcq_correct']}/{merged_metrics['mcq_total']} MCQ samples)")
            if "cot_accuracy" in merged_metrics:
                print(f"Merged Test CoT Accuracy: {merged_metrics['cot_accuracy']:.4f} "
                      f"({merged_metrics['cot_correct']}/{merged_metrics['cot_total']} CoT samples)")
            if "overall_accuracy" in merged_metrics:
                print(f"Merged Test Overall Accuracy: {merged_metrics['overall_accuracy']:.4f}")
            if "accuracy" in tsexam_metrics:
                print(f"TimeSeriesExam1 Accuracy: {tsexam_metrics['accuracy']:.4f}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Merged Training for OpenTSLM Models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["OpenTSLMSP", "OpenTSLMFlamingo"],
        required=True,
        help="Model type to train",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training (default: use value from model_config.py)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--max_samples_per_dataset",
        type=int,
        default=None,
        help="Max samples per dataset (for sanity checks)",
    )
    parser.add_argument(
        "--eval_only",
        default=False,
        action="store_true",
        help="Skip training and only run evaluation",
    )
    parser.add_argument(
        "--llm_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="LLM model ID",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="URL used to set up distributed training",
    )
    parser.add_argument(
        "--dist_backend", default="nccl", type=str, help="Distributed backend"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Local GPU rank",
    )
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--balanced",
        default=False,
        action="store_true",
        help="Use balanced training (upsample smaller datasets to match largest)"
    )
    parser.add_argument(
        "--balance_strategy",
        type=str,
        default="max",
        help="Balance strategy: 'max' to match largest dataset size, or an integer target size"
    )

    args = parser.parse_args()

    set_global_verbose(args.verbose)
    logger = get_logger(verbose=args.verbose)

    trainer = MergedTrainer(
        args.model,
        args.device,
        gradient_checkpointing=args.gradient_checkpointing,
        dist_url=args.dist_url,
        dist_backend=args.dist_backend,
        local_rank=args.local_rank,
        llm_id=args.llm_id,
    )

    if args.balanced:
        results = trainer.train_merged_balanced(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            eval_only=args.eval_only,
            max_samples_per_dataset=args.max_samples_per_dataset,
            balance_strategy=args.balance_strategy,
        )
    else:
        results = trainer.train_merged(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            eval_only=args.eval_only,
            max_samples_per_dataset=args.max_samples_per_dataset,
        )

    logger.info("Final Results:")
    logger.info("=" * 40)
    for key, value in results.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for k, v in value.items():
                if isinstance(v, (int, float)):
                    logger.info(f"  {k}: {v:.4f}")
                else:
                    logger.info(f"  {k}: {v}")
        else:
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")


if __name__ == "__main__":
    main()
