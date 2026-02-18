#
# ZuCo EEG Sentiment Classification Dataset for OpenTSLM
#
# Task: Given multi-channel EEG signals recorded during sentiment reading,
# classify the sentiment of the text being read (negative, neutral, positive).
# The model sees both EEG signals and the sentence text.
#

from datasets import Dataset
from typing import List, Tuple, Literal
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from time_series_datasets.zuco_eeg.zuco_eeg_loader import load_zuco_eeg_sentiment_splits
import torch
import numpy as np

CHANNEL_NAMES = [
    "EEG channel 0 (frontal left)",
    "EEG channel 1 (frontal right)",
    "EEG channel 2 (central left)",
    "EEG channel 3 (central)",
    "EEG channel 4 (central right)",
    "EEG channel 5 (parietal left)",
    "EEG channel 6 (parietal right)",
    "EEG channel 7 (occipital)",
]


class ZuCoEEGSentimentDataset(QADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        return load_zuco_eeg_sentiment_splits()

    def _get_answer(self, row) -> str:
        return row["label"]

    def _get_pre_prompt(self, row) -> str:
        sentence = row["sentence_text"]
        return (
            "You are given multi-channel EEG signals recorded while a person reads a sentence. "
            "Classify the sentiment of the text being read based on the brain activity.\n\n"
            f"Sentence: \"{sentence}\""
        )

    def _get_post_prompt(self, row) -> str:
        return (
            "Possible labels: negative, neutral, positive.\n"
            "Answer:"
        )

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        eeg_channels = row["eeg_channels"]
        prompts = []

        for i, channel_data in enumerate(eeg_channels):
            series = torch.tensor(channel_data, dtype=torch.float32)

            mean_val = series.mean().item()
            std_val = series.std().item()

            # z-score normalize
            std_safe = max(std_val, 1e-6)
            series_norm = ((series - mean_val) / std_safe).tolist()

            text = (
                f"This is {CHANNEL_NAMES[i]} EEG signal (downsampled to 100Hz), "
                f"it has mean {mean_val:.4f} and std {std_val:.4f}:"
            )

            prompts.append(TextTimeSeriesPrompt(text, series_norm))

        return prompts

    @staticmethod
    def get_labels() -> List[str]:
        return ["negative", "neutral", "positive"]

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["label"] = row["label"]
        sample["subject"] = row["subject"]
        sample["sentence_text"] = row["sentence_text"]
        sample["sentiment_value"] = row["sentiment_value"]
        return sample


if __name__ == "__main__":
    print("=== ZuCo EEG Sentiment Dataset ===\n")

    dataset = ZuCoEEGSentimentDataset(split="train", EOS_TOKEN="")
    dataset_val = ZuCoEEGSentimentDataset(split="validation", EOS_TOKEN="")
    dataset_test = ZuCoEEGSentimentDataset(split="test", EOS_TOKEN="")

    print(
        f"Dataset sizes: Train={len(dataset)}, Val={len(dataset_val)}, Test={len(dataset_test)}"
    )

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys: {list(sample.keys())}")
        print(f"Label: {sample['label']}")
        print(f"Sentiment value: {sample['sentiment_value']}")
        print(f"Subject: {sample['subject']}")
        print(f"Sentence: {sample['sentence_text'][:80]}...")
        print(f"\nPre-prompt:\n{sample['pre_prompt']}")
        print(f"\nPost-prompt:\n{sample['post_prompt']}")
        print(f"\nAnswer: {sample['answer']}")
