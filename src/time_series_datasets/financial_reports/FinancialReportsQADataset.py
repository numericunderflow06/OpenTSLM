#
# Financial Reports + Stock Price QA Dataset for OpenTSLM
#
# Task: Given a regulatory filing text and the stock's recent price history,
# answer one of three question types:
#   1. Price return — will the stock price increase or decrease?
#   2. Volatility — will volatility increase, decrease, or remain stable?
#   3. Market direction — will the stock trend bullish, bearish, or sideways?
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

QUESTION_TYPES = ("price_return", "volatility", "direction")
LABEL_KEYS = {
    "price_return": "label",
    "volatility": "volatility_label",
    "direction": "direction_label",
}


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
        train, val, test = load_financial_reports_splits()
        return self._expand(train), self._expand(val), self._expand(test)

    @staticmethod
    def _expand(dataset: Dataset) -> Dataset:
        """Expand each row into 3 rows, one per question type."""
        rows = []
        for row in dataset:
            for qt in QUESTION_TYPES:
                new_row = dict(row)
                new_row["question_type"] = qt
                rows.append(new_row)
        return Dataset.from_list(rows)

    def _get_answer(self, row) -> str:
        return row[LABEL_KEYS[row["question_type"]]]

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
        qt = row["question_type"]
        opening = "Based on the stock price trend leading up to this filing and the filing content, "

        if qt == "price_return":
            return (
                opening
                + "will the stock price increase or decrease over the next 5 trading days?\n"
                "(a) Increase\n"
                "(b) Decrease\n"
                "Answer:"
            )
        elif qt == "volatility":
            return (
                opening
                + "will the stock's daily price volatility increase, decrease, or remain stable "
                "over the next 5 trading days compared to the 60 days before the filing?\n"
                "(a) Increase\n"
                "(b) Decrease\n"
                "(c) Remain stable\n"
                "Answer:"
            )
        else:  # direction
            return (
                opening
                + "what will be the overall market direction of this stock over the next 5 trading days?\n"
                "(a) Bullish (upward trend)\n"
                "(b) Bearish (downward trend)\n"
                "(c) Sideways (no clear trend)\n"
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
        return ["(a)", "(b)", "(c)"]

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["label"] = row[LABEL_KEYS[row["question_type"]]]
        sample["company_name"] = row["company_name"]
        sample["filing_date"] = row["filing_date"]
        sample["post_return"] = row["post_return"]
        sample["question_type"] = row["question_type"]
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
        # Show one sample per question type
        for qt in QUESTION_TYPES:
            for i in range(len(dataset)):
                sample = dataset[i]
                if sample["question_type"] == qt:
                    print(f"\n--- Question type: {qt} ---")
                    print(f"Company: {sample['company_name']}")
                    print(f"Filing date: {sample['filing_date']}")
                    print(f"Label: {sample['label']}")
                    print(f"Post return: {sample['post_return']:.4f}")
                    print(f"\nPost-prompt:\n{sample['post_prompt']}")
                    print(f"\nAnswer: {sample['answer']}")
                    break
