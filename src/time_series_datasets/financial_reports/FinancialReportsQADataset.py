#
# Financial Reports + Stock Price QA Dataset for OpenTSLM
#
# Task: Given a regulatory filing text and the stock's recent price history,
# predict whether the stock price will increase or decrease over the next
# 5 trading days after the filing date.
#

from datasets import Dataset
from typing import List, Tuple, Literal
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from time_series_datasets.financial_reports.financial_reports_loader import (
    load_financial_reports_splits,
)
import torch
import numpy as np


class FinancialReportsQADataset(QADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        return load_financial_reports_splits()

    def _get_answer(self, row) -> str:
        return row["label"]

    def _get_pre_prompt(self, row) -> str:
        filing_type = row["filing_type_specific"]
        company = row["company_name"]
        date = row["filing_date"]
        text = row["filing_text"]

        return (
            f"You are a financial analyst specializing in European real estate equities. "
            f"Below is a {filing_type} filed by {company} on {date}.\n\n"
            f"--- Filing Content ---\n{text}\n--- End of Filing ---\n\n"
            f"You also have the company's daily closing stock price for the 60 trading "
            f"days leading up to this filing."
        )

    def _get_post_prompt(self, row) -> str:
        return (
            "Based on the stock price trend leading up to this filing and the filing content, "
            "will the stock price increase or decrease over the next 5 trading days?\n"
            "(a) Increase\n"
            "(b) Decrease\n"
            "Answer:"
        )

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        prices = row["pre_prices"]
        series = torch.tensor(prices, dtype=torch.float32)

        mean_val = series.mean().item()
        std_val = series.std().item()

        # z-score normalize
        std_safe = max(std_val, 1e-6)
        series_norm = ((series - mean_val) / std_safe).tolist()

        company = row["company_name"]
        ticker = row["ticker"]
        text = (
            f"This is the daily closing stock price of {company} ({ticker}) "
            f"for the 60 trading days before the filing, "
            f"it has mean {mean_val:.4f} and std {std_val:.4f}:"
        )

        return [TextTimeSeriesPrompt(text, series_norm)]

    @staticmethod
    def get_labels() -> List[str]:
        return ["(a)", "(b)"]

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["label"] = row["label"]
        sample["company_name"] = row["company_name"]
        sample["filing_date"] = row["filing_date"]
        sample["post_return"] = row["post_return"]
        return sample


if __name__ == "__main__":
    print("=== Financial Reports QA Dataset ===\n")

    dataset = FinancialReportsQADataset(split="train", EOS_TOKEN="")
    dataset_val = FinancialReportsQADataset(split="validation", EOS_TOKEN="")
    dataset_test = FinancialReportsQADataset(split="test", EOS_TOKEN="")

    print(
        f"Dataset sizes: Train={len(dataset)}, Val={len(dataset_val)}, Test={len(dataset_test)}"
    )

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys: {list(sample.keys())}")
        print(f"Company: {sample['company_name']}")
        print(f"Filing date: {sample['filing_date']}")
        print(f"Label: {sample['label']}")
        print(f"Post return: {sample['post_return']:.4f}")
        print(f"\nPre-prompt preview:\n{sample['pre_prompt'][:300]}...")
        print(f"\nTime series text: {sample['time_series_text']}")
        print(f"Time series shape: {[len(ts) for ts in sample['time_series']]}")
        print(f"\nPost-prompt:\n{sample['post_prompt']}")
        print(f"\nAnswer: {sample['answer']}")
