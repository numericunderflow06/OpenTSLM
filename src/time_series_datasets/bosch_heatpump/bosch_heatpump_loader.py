#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Loader for Bosch heat pump Check-It data.

Reads two CSVs:
  - df_metadata_main_train.csv: one row per system with building metadata + expert comments
  - df_feats_train.csv: daily time series features per system

Joins on sysid, groups daily features into per-system 1D arrays, and creates
two task samples per system (one for space heating, one for domestic hot water).
"""

import os
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from datasets import Dataset


# Default data directory — override with BOSCH_DATA_DIR env var
BOSCH_DATA_DIR = os.environ.get(
    "BOSCH_DATA_DIR",
    os.path.join(os.path.expanduser("~"), "bosch"),
)

# Time series feature columns to extract (order matters — this becomes the channel order)
# These are the core energy & temperature signals relevant to expert commentary.
TS_FEATURE_COLS = [
    "outdoor_temp_mean",
    "outdoor_temp_min",
    "energy_consumption_ch",
    "energy_consumption_dhw",
    "energy_consumption_total",
    "CH_Energy_Out_Total_delta",
    "DHW_Energy_Out_Total_delta",
    "HP_Energy_Out_Total_delta",
    "CH_E21_comp_energy_delta",
    "DHW_E21_comp_energy_delta",
    "CH_EHeat_energy_delta",
    "DHW_EHeat_energy_delta",
    "dhw_share",
]

# Human-readable labels for each channel (used in prompts)
TS_FEATURE_LABELS = {
    "outdoor_temp_mean": "Daily mean outdoor temperature (°C)",
    "outdoor_temp_min": "Daily minimum outdoor temperature (°C)",
    "energy_consumption_ch": "Daily energy consumption for space heating (kWh)",
    "energy_consumption_dhw": "Daily energy consumption for domestic hot water (kWh)",
    "energy_consumption_total": "Daily total energy consumption (kWh)",
    "CH_Energy_Out_Total_delta": "Daily heat output for space heating (kWh)",
    "DHW_Energy_Out_Total_delta": "Daily heat output for domestic hot water (kWh)",
    "HP_Energy_Out_Total_delta": "Daily total heat pump output (kWh)",
    "CH_E21_comp_energy_delta": "Daily compressor energy for space heating (kWh)",
    "DHW_E21_comp_energy_delta": "Daily compressor energy for domestic hot water (kWh)",
    "CH_EHeat_energy_delta": "Daily electric heater energy for space heating (kWh)",
    "DHW_EHeat_energy_delta": "Daily electric heater energy for domestic hot water (kWh)",
    "dhw_share": "Daily share of domestic hot water in total consumption (%)",
}

# Channels relevant to each comment type (space heating vs DHW)
CH_CHANNELS = [
    "outdoor_temp_mean",
    "outdoor_temp_min",
    "energy_consumption_ch",
    "energy_consumption_total",
    "CH_Energy_Out_Total_delta",
    "HP_Energy_Out_Total_delta",
    "CH_E21_comp_energy_delta",
    "CH_EHeat_energy_delta",
]

DHW_CHANNELS = [
    "outdoor_temp_mean",
    "energy_consumption_dhw",
    "energy_consumption_total",
    "DHW_Energy_Out_Total_delta",
    "HP_Energy_Out_Total_delta",
    "DHW_E21_comp_energy_delta",
    "DHW_EHeat_energy_delta",
    "dhw_share",
]


def _load_and_join(
    feats_path: str, meta_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the two CSV files."""
    feats = pd.read_csv(feats_path, parse_dates=["date"])
    meta = pd.read_csv(meta_path)

    # Validate all feature sysids exist in metadata
    missing = set(feats["sysid"].unique()) - set(meta["sysid"].unique())
    if missing:
        raise ValueError(f"{len(missing)} sysids in features but not in metadata: {list(missing)[:5]}")

    return feats, meta


def _clean_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Clean feature data: replace inf with NaN, then forward-fill per system."""
    feats = feats.copy()
    # Replace inf values (found in dhw_share)
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Sort by system and date for proper forward-fill
    feats.sort_values(["sysid", "date"], inplace=True)
    # Forward-fill NaN within each system, then back-fill remaining
    numeric_cols = [c for c in feats.columns if c not in ("sysid", "date")]
    feats[numeric_cols] = feats.groupby("sysid")[numeric_cols].transform(
        lambda x: x.ffill().bfill().fillna(0.0)
    )
    return feats


def _build_system_records(
    feats: pd.DataFrame, meta: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Build one record per system containing:
      - metadata fields (building info)
      - per-channel 1D numpy arrays (daily time series)
      - expert comments
    """
    records = []
    grouped = feats.groupby("sysid")

    for _, meta_row in meta.iterrows():
        sysid = meta_row["sysid"]
        if sysid not in grouped.groups:
            print(f"Warning: sysid {sysid} has no feature data, skipping")
            continue

        sys_feats = grouped.get_group(sysid).sort_values("date")

        # Build time series dict
        ts_dict = {}
        for col in TS_FEATURE_COLS:
            arr = sys_feats[col].values.astype(np.float32)
            ts_dict[col] = arr.tolist()  # store as list for HF Dataset compatibility

        # Store number of days
        n_days = len(sys_feats)

        # Helper: safely convert to string, handling NaN
        def safe_str(val, default=""):
            if pd.isna(val):
                return default
            return str(val)

        # Extract metadata fields — stored as strings for prompt construction
        record = {
            "sysid": sysid,
            "n_days": n_days,
            "date_start": str(sys_feats["date"].iloc[0].date()),
            "date_end": str(sys_feats["date"].iloc[-1].date()),
            # Building metadata (all strings — used only in prompts)
            "Gebaudetyp": safe_str(meta_row.get("Gebaudetyp")),
            "Baujahr": safe_str(meta_row.get("Baujahr")),
            "Jahr_der_Renovierung": safe_str(meta_row.get("Jahr_der_Renovierung")),
            "Beheizte_Wohnflaeche": safe_str(meta_row.get("Beheizte_Wohnflaeche")),
            "Heizkreis_1": safe_str(meta_row.get("Heizkreis_1")),
            "Max_Vorlauftemperatur": safe_str(meta_row.get("Max_Vorlauftemperatur")),
            "Heizkreis_2": safe_str(meta_row.get("Heizkreis_2")),
            "Max_Vorlauftemperatur_2": safe_str(meta_row.get("Max_Vorlauftemperatur_2")),
            "Pufferspeicher_vorhanden": safe_str(meta_row.get("Pufferspeicher_vorhanden"), "Nein"),
            "Speicher": safe_str(meta_row.get("Speicher"), "Nein"),
            "Waermebedarf": safe_str(meta_row.get("Waermebedarf")),
            "heatSystem_DomesticHotWaterincluded": safe_str(meta_row.get("heatSystem.DomesticHotWaterincluded")),
            "hotWaterSystem_numberOfConsumers": safe_str(meta_row.get("hotWaterSystem.numberOfConsumers")),
            "hotWaterSystem_consumptionLevel": safe_str(meta_row.get("hotWaterSystem.consumptionLevel")),
            "heat_load_for_space_heating": safe_str(meta_row.get("heat_load_for_space_heating")),
            # Expert comments (ground truth answers)
            "Comment_space_heating": str(meta_row["Comment_space_heating"]),
            "Comment_domestic_hot_water": str(meta_row["Comment_domestic_hot_water"]),
        }
        # Add time series
        record.update(ts_dict)

        records.append(record)

    return records


def load_bosch_heatpump_splits(
    feats_path: str = None,
    meta_path: str = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the Bosch heat pump data and split into train/val/test.

    Args:
        feats_path: Path to df_feats_train.csv (defaults to BOSCH_DATA_DIR/df_feats_train.csv)
        meta_path: Path to df_metadata_main_train.csv
        val_frac: Fraction of systems for validation
        test_frac: Fraction of systems for test
        seed: Random seed for reproducible splits

    Returns:
        (train_dataset, val_dataset, test_dataset) as HuggingFace Datasets
    """
    if feats_path is None:
        feats_path = os.path.join(BOSCH_DATA_DIR, "df_feats_train.csv")
    if meta_path is None:
        meta_path = os.path.join(BOSCH_DATA_DIR, "df_metadata_main_train.csv")

    print(f"Loading Bosch heat pump data from {BOSCH_DATA_DIR}...")
    feats, meta = _load_and_join(feats_path, meta_path)
    feats = _clean_features(feats)
    records = _build_system_records(feats, meta)
    print(f"Built {len(records)} system records")

    # Split by system (not by row) to avoid data leakage
    rng = np.random.RandomState(seed)
    n = len(records)
    indices = rng.permutation(n)

    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    print(f"Split: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    return (
        Dataset.from_list(train_records),
        Dataset.from_list(val_records),
        Dataset.from_list(test_records),
    )


if __name__ == "__main__":
    train, val, test = load_bosch_heatpump_splits()
    print(f"\nTrain: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Train columns: {train.column_names}")
    sample = train[0]
    print(f"\nSample sysid: {sample['sysid']}")
    print(f"Sample n_days: {sample['n_days']}")
    print(f"Sample outdoor_temp_mean length: {len(sample['outdoor_temp_mean'])}")
    print(f"Sample Comment_space_heating: {sample['Comment_space_heating'][:200]}...")
