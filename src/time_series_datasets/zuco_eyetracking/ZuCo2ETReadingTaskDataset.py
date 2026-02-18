#
# ZuCo 2.0 Eye-Tracking Reading Task Classification Dataset for OpenTSLM
#
# Task: Given word-level eye-tracking metrics recorded during reading,
# classify whether the reader was engaged in Normal Reading (NR)
# or Task-Specific Reading (TSR).
#
# Identical interface to ZuCoETReadingTaskDataset but uses ZuCo 2.0 data.
#

from datasets import Dataset
from typing import List, Tuple, Literal
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from time_series_datasets.zuco_eyetracking.zuco2_et_loader import (
    load_zuco2_et_reading_task_splits,
    ET_METRICS,
    ET_METRIC_LABELS,
)
import torch
import numpy as np


class ZuCo2ETReadingTaskDataset(QADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        return load_zuco2_et_reading_task_splits()

    def _get_answer(self, row) -> str:
        return row["label"]

    def _get_pre_prompt(self, row) -> str:
        return (
            "You are given word-level eye-tracking measurements recorded during reading. "
            "Each channel represents a different eye-tracking metric measured for each word in the sentence. "
            "Classify whether the reader was engaged in normal reading or task-specific reading."
        )

    def _get_post_prompt(self, row) -> str:
        return (
            "Possible labels: Normal Reading, Task-Specific Reading.\n"
            "Answer:"
        )

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        et_channels = row["et_channels"]
        prompts = []

        for i, metric_name in enumerate(ET_METRICS):
            channel_data = et_channels[i]
            series = torch.tensor(channel_data, dtype=torch.float32)

            mean_val = series.mean().item()
            std_val = series.std().item()

            # z-score normalize
            std_safe = max(std_val, 1e-6)
            series_norm = ((series - mean_val) / std_safe).tolist()

            label = ET_METRIC_LABELS[metric_name]
            text = (
                f"This is the {label}, "
                f"it has mean {mean_val:.4f} and std {std_val:.4f}:"
            )

            prompts.append(TextTimeSeriesPrompt(text, series_norm))

        return prompts

    @staticmethod
    def get_labels() -> List[str]:
        return ["Normal Reading", "Task-Specific Reading"]

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["label"] = row["label"]
        sample["subject"] = row["subject"]
        sample["sentence_text"] = row["sentence_text"]
        return sample


if __name__ == "__main__":
    print("=== ZuCo 2.0 Eye-Tracking Reading Task Dataset ===\n")

    dataset = ZuCo2ETReadingTaskDataset(split="train", EOS_TOKEN="")
    dataset_val = ZuCo2ETReadingTaskDataset(split="validation", EOS_TOKEN="")
    dataset_test = ZuCo2ETReadingTaskDataset(split="test", EOS_TOKEN="")

    print(
        f"Dataset sizes: Train={len(dataset)}, Val={len(dataset_val)}, Test={len(dataset_test)}"
    )

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys: {list(sample.keys())}")
        print(f"Label: {sample['label']}")
        print(f"Subject: {sample['subject']}")
        print(f"Sentence: {sample['sentence_text'][:80]}...")
        print(f"\nPre-prompt:\n{sample['pre_prompt']}")
        print(f"\nPost-prompt:\n{sample['post_prompt']}")
        print(f"\nAnswer: {sample['answer']}")
