#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Bosch Heat Pump QA Dataset for OpenTSLM.

Each sample consists of:
  - Building metadata (in the pre-prompt)
  - Multiple 1D time series channels (daily energy, temperature, etc.)
  - An expert comment as the answer (either for space heating or DHW)

Two task types are supported:
  - "space_heating": generate the space heating expert comment
  - "dhw": generate the domestic hot water expert comment
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets import Dataset
from typing import List, Tuple, Literal

import numpy as np
from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from time_series_datasets.bosch_heatpump.bosch_heatpump_loader import (
    load_bosch_heatpump_splits,
    TS_FEATURE_LABELS,
    CH_CHANNELS,
    DHW_CHANNELS,
)


class BoschHeatPumpQADataset(QADataset):
    """
    Bosch Heat Pump dataset for generating expert commentary from time series data.

    Supports two task types:
      - "space_heating": predict the space heating comment (Comment_space_heating)
      - "dhw": predict the domestic hot water comment (Comment_domestic_hot_water)
      - "both": create one sample per comment type per system (doubles the dataset)
    """

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
        task_type: Literal["space_heating", "dhw", "both"] = "both",
        max_samples: int = None,
    ):
        """
        Args:
            split: Dataset split
            EOS_TOKEN: End-of-sequence token
            format_sample_str: Whether to format as string (for text-only baselines)
            time_series_format_function: Optional custom formatter
            task_type: Which comment to predict ("space_heating", "dhw", or "both")
            max_samples: Limit samples per split (for quick experiments)
        """
        self.task_type = task_type
        self.max_samples = max_samples
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load the Bosch heat pump dataset and expand by task type."""
        print(f"Loading Bosch heat pump dataset (task_type={self.task_type})...")
        train, val, test = load_bosch_heatpump_splits()

        # Expand: create separate samples for space_heating and/or dhw
        train = self._expand_by_task(train)
        val = self._expand_by_task(val)
        test = self._expand_by_task(test)

        if self.max_samples:
            print(f"Limiting to {self.max_samples} samples per split...")
            if len(train) > self.max_samples:
                train = train.select(range(self.max_samples))
            if len(val) > self.max_samples:
                val = val.select(range(self.max_samples))
            if len(test) > self.max_samples:
                test = test.select(range(self.max_samples))

        print(f"Final sizes: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def _expand_by_task(self, dataset: Dataset) -> Dataset:
        """Add a 'task' column and optionally duplicate rows for both tasks."""
        expanded = []
        for row in dataset:
            row_dict = dict(row)
            if self.task_type in ("space_heating", "both"):
                sample = dict(row_dict)
                sample["task"] = "space_heating"
                expanded.append(sample)
            if self.task_type in ("dhw", "both"):
                sample = dict(row_dict)
                sample["task"] = "dhw"
                expanded.append(sample)
        return Dataset.from_list(expanded)

    def _get_answer(self, row) -> str:
        """Return the expert comment for this task type."""
        if row["task"] == "space_heating":
            return row["Comment_space_heating"]
        else:
            return row["Comment_domestic_hot_water"]

    def _get_pre_prompt(self, row) -> str:
        """Build the pre-prompt with building metadata and task instructions."""
        task = row["task"]

        if task == "space_heating":
            task_description = "space heating (Heizbetrieb)"
            focus = "energy generation and consumption patterns for space heating"
        else:
            task_description = "domestic hot water (Warmwasserbereitung)"
            focus = "energy generation and consumption patterns for domestic hot water"

        # Build building context from metadata (all stored as strings)
        def _add(parts, label, val, suffix=""):
            if val and val not in ("", "Nein", "nan", "0", "0.0"):
                parts.append(f"{label}: {val}{suffix}")

        building_info_parts = []
        building_info_parts.append(f"Building type: {row['Gebaudetyp']}")
        _add(building_info_parts, "Year of construction", row.get("Baujahr"))
        _add(building_info_parts, "Year of renovation", row.get("Jahr_der_Renovierung"))
        _add(building_info_parts, "Heated living area", row.get("Beheizte_Wohnflaeche"), " m²")

        hk1 = row.get("Heizkreis_1", "")
        if hk1:
            hk = f"Heating circuit: {hk1}"
            mvt = row.get("Max_Vorlauftemperatur", "")
            if mvt and mvt not in ("", "0", "0.0"):
                hk += f" (max flow temperature: {mvt}°C)"
            building_info_parts.append(hk)

        hk2 = row.get("Heizkreis_2", "")
        if hk2 and hk2 not in ("", "nan"):
            hk2_str = f"Second heating circuit: {hk2}"
            mvt2 = row.get("Max_Vorlauftemperatur_2", "")
            if mvt2 and mvt2 not in ("", "0", "0.0"):
                hk2_str += f" (max flow temperature: {mvt2}°C)"
            building_info_parts.append(hk2_str)

        buf = row.get("Pufferspeicher_vorhanden", "Nein")
        if buf and buf not in ("Nein", "", "nan"):
            building_info_parts.append(f"Buffer storage: {buf} L")
        speicher = row.get("Speicher", "Nein")
        if speicher and speicher not in ("Nein", "", "nan"):
            building_info_parts.append(f"Storage tank: {speicher} L")
        _add(building_info_parts, "Heat demand", row.get("Waermebedarf"), " kWh")
        _add(building_info_parts, "Heat load for space heating", row.get("heat_load_for_space_heating"), " W")
        _add(building_info_parts, "Number of hot water consumers", row.get("hotWaterSystem_numberOfConsumers"))
        _add(building_info_parts, "Hot water consumption level", row.get("hotWaterSystem_consumptionLevel"))

        building_context = "\n".join(building_info_parts)

        prompt = f"""You are an expert heat pump engineer analyzing operational data from a Bosch heat pump installation.

Observation period: {row['date_start']} to {row['date_end']} ({row['n_days']} days)

Building and system information:
{building_context}

Your task is to analyze the time series sensor data and write an expert assessment of the {task_description} performance.

Instructions:
- Focus on {focus} in the daily time series data.
- Describe observed patterns, trends, and any anomalies.
- Relate the energy behavior to outdoor temperature changes where relevant.
- Consider the building characteristics when interpreting the data.
- Write your assessment as a single, natural paragraph in German.
- Be specific: reference concrete values, dates, and patterns you observe in the data."""

        return prompt

    def _get_post_prompt(self, row) -> str:
        """Post-prompt requesting the expert assessment."""
        if row["task"] == "space_heating":
            return "Expertenbewertung Heizbetrieb:"
        else:
            return "Expertenbewertung Warmwasserbereitung:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Convert the per-system daily features into TextTimeSeriesPrompt objects.

        Each relevant channel becomes a separate 1D time series with a descriptive label
        including mean and std statistics.
        """
        task = row["task"]
        channels = CH_CHANNELS if task == "space_heating" else DHW_CHANNELS

        prompts = []
        for col in channels:
            raw_values = row[col]
            if raw_values is None:
                continue

            arr = np.array(raw_values, dtype=np.float32)
            if arr.size == 0:
                continue

            # Replace any remaining NaN/inf with 0
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            # Compute stats before normalization (for the label)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))

            # Z-score normalize
            if std_val > 1e-6:
                arr_norm = (arr - mean_val) / std_val
            else:
                arr_norm = arr - mean_val

            label = TS_FEATURE_LABELS.get(col, col)
            text = f"{label}, it has mean {mean_val:.2f} and std {std_val:.2f}:"

            prompts.append(TextTimeSeriesPrompt(text, arr_norm.tolist()))

        if not prompts:
            raise RuntimeError(
                f"No valid time series channels for sysid={row['sysid']} task={task}"
            )

        return prompts

    def _format_sample(self, row):
        """Override to preserve task-specific metadata."""
        sample = super()._format_sample(row)
        sample["sysid"] = row["sysid"]
        sample["task"] = row["task"]
        sample["n_days"] = row["n_days"]
        return sample

    def _format_sample_str(self, time_series_format_function, row):
        """Override to preserve metadata for text-only baselines."""
        sample = super()._format_sample_str(time_series_format_function, row)
        sample["sysid"] = row["sysid"]
        sample["task"] = row["task"]
        sample["n_days"] = row["n_days"]
        return sample


if __name__ == "__main__":
    print("Testing BoschHeatPumpQADataset...")
    print()

    # Test with both tasks
    dataset = BoschHeatPumpQADataset(
        split="train", EOS_TOKEN="", task_type="both", max_samples=5
    )
    dataset_val = BoschHeatPumpQADataset(
        split="validation", EOS_TOKEN="", task_type="both", max_samples=5
    )
    dataset_test = BoschHeatPumpQADataset(
        split="test", EOS_TOKEN="", task_type="both", max_samples=5
    )

    print(f"Dataset sizes: Train={len(dataset)}, Val={len(dataset_val)}, Test={len(dataset_test)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample keys: {list(sample.keys())}")
        print(f"sysid: {sample['sysid']}")
        print(f"task: {sample['task']}")
        print(f"n_days: {sample['n_days']}")
        print(f"Number of time series channels: {len(sample['time_series'])}")
        for i, (ts, label) in enumerate(
            zip(sample["time_series"], sample["time_series_text"])
        ):
            ts_arr = np.array(ts)
            print(f"  Channel {i}: {label[:60]}... | len={len(ts_arr)}")
        print(f"\nPre-prompt preview:\n{sample['pre_prompt'][:300]}...")
        print(f"\nPost-prompt: {sample['post_prompt']}")
        print(f"\nAnswer preview: {sample['answer'][:300]}...")
