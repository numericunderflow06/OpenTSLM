#
# Financial Reports + Stock Price Dataset Loader for OpenTSLM
#
# Downloads stock price data for 15 European real estate companies,
# pairs it with regulatory filing text, and produces a dataset for
# post-filing return direction prediction.
#

import os
import sys
import json
import pickle
import csv
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from time_series_datasets.constants import RAW_DATA

# --- Paths ---
FINANCIAL_DATA_DIR = os.path.join(RAW_DATA, "financial_reports")
STOCK_PRICES_DIR = os.path.join(FINANCIAL_DATA_DIR, "stock_prices")
PREPROCESSED_PKL = os.path.join(FINANCIAL_DATA_DIR, "preprocessed_dataset.pkl")

# Path to the extracted financial reports
REPORTS_BASE = os.path.join(
    os.path.dirname(RAW_DATA),  # OpenTSLM root
    "..",  # parent of OpenTSLM
    "OpenTSLM_project",
    "financial_reports_sample_real_estate",
    "export_sample_tslm_complete",
)

METADATA_CSV = os.path.join(REPORTS_BASE, "metadata.csv")

# --- ISIN to Yahoo Finance ticker mapping ---
# These are European real estate stocks traded on various exchanges
ISIN_TO_TICKER = {
    "DE0007480204": "DEQ.DE",       # Deutsche EuroShop AG (XETRA)
    "DE0008303504": "TEG.DE",       # TAG Immobilien AG (XETRA)
    "AT0000827209": "WXF.VI",       # Warimpex (Vienna)
    # AT0000A21KS2 (Immofinanz/CPI Europe) — delisted, no Yahoo data available
    "BE0003851681": "AED.BR",       # Aedifica (Brussels)
    "BE0003837540": "ATEB.BR",      # Atenor (Brussels)
    "BE0003853703": "MONT.BR",      # Montea (Brussels)
    "BE0003770840": "NEXTA.BR",     # Nextensa (Brussels)
    "BE0003720340": "RET.BR",       # Retail Estates (Brussels)
    "BE0003754687": "VASTB.BR",     # VastNed Belgium (Brussels)
    "FI4000369947": "CTY1S.HE",     # Citycon (Helsinki)
    "NO0010612450": "SBO.OL",       # Selvaag Bolig (Oslo)
    "PLNFI1200018": "BBD.WA",       # BBI Development (Warsaw)
    "PLECHPS00019": "ECH.WA",       # Echo Investment (Warsaw)
    "PLASMOT00030": "TRN.WA",       # Triton Development (Warsaw)
}

# --- Configuration ---
PRE_FILING_DAYS = 60         # Trading days of price history before filing
POST_FILING_DAYS = 5         # Trading days after filing to compute return
MAX_FILING_TEXT_CHARS = 3000 # Max characters of filing text to include
TEST_FRAC = 0.1
VAL_FRAC = 0.1


def download_stock_prices(force: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Download daily stock price data for all companies from Yahoo Finance.
    Caches results to disk as CSV files.

    Returns:
        Dict mapping ISIN to DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    import yfinance as yf

    os.makedirs(STOCK_PRICES_DIR, exist_ok=True)
    prices = {}

    for isin, ticker in tqdm(ISIN_TO_TICKER.items(), desc="Downloading stock prices"):
        csv_path = os.path.join(STOCK_PRICES_DIR, f"{isin}.csv")

        if os.path.exists(csv_path) and not force:
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            if len(df) > 0:
                prices[isin] = df
                continue

        print(f"  Fetching {ticker} (ISIN: {isin})...")
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="max", auto_adjust=True)

            if df.empty:
                print(f"  WARNING: No data returned for {ticker}")
                continue

            df = df.reset_index()
            # Normalize column names and timezone
            if hasattr(df["Date"].dtype, "tz") and df["Date"].dtype.tz is not None:
                df["Date"] = df["Date"].dt.tz_localize(None)
            df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            df = df.sort_values("Date").reset_index(drop=True)
            df.to_csv(csv_path, index=False)
            prices[isin] = df
            print(f"  Got {len(df)} trading days for {ticker} ({df['Date'].min().date()} to {df['Date'].max().date()})")
        except Exception as e:
            print(f"  ERROR fetching {ticker}: {e}")

    return prices


def _get_price_window(
    price_df: pd.DataFrame,
    filing_date: datetime,
    pre_days: int,
    post_days: int,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Extract pre-filing price series and post-filing return.

    Returns:
        Tuple of (pre_prices_array, post_return) or None if insufficient data.
        post_return is the fractional return over post_days trading days.
    """
    filing_ts = pd.Timestamp(filing_date)

    # Find the last trading day on or before the filing date
    mask_before = price_df["Date"] <= filing_ts
    if mask_before.sum() < pre_days:
        return None

    pre_idx = price_df.loc[mask_before].index[-1]
    pre_start_idx = pre_idx - pre_days + 1
    if pre_start_idx < 0:
        return None

    pre_prices = price_df.loc[pre_start_idx:pre_idx, "Close"].values.astype(np.float64)
    if len(pre_prices) != pre_days:
        return None

    # Find post-filing trading days
    mask_after = price_df["Date"] > filing_ts
    post_data = price_df.loc[mask_after]
    if len(post_data) < post_days:
        return None

    close_at_filing = price_df.loc[pre_idx, "Close"]
    close_after = post_data.iloc[post_days - 1]["Close"]
    post_return = (close_after - close_at_filing) / close_at_filing

    return pre_prices, float(post_return)


def _read_filing_text(relative_path: str) -> str:
    """Read the markdown filing text, truncated to MAX_FILING_TEXT_CHARS."""
    full_path = os.path.join(REPORTS_BASE, relative_path)
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(MAX_FILING_TEXT_CHARS + 500)
        # Truncate cleanly at a sentence/paragraph boundary
        if len(text) > MAX_FILING_TEXT_CHARS:
            # Try to cut at last paragraph break
            cut = text[:MAX_FILING_TEXT_CHARS].rfind("\n\n")
            if cut < MAX_FILING_TEXT_CHARS // 2:
                cut = text[:MAX_FILING_TEXT_CHARS].rfind("\n")
            if cut < MAX_FILING_TEXT_CHARS // 2:
                cut = MAX_FILING_TEXT_CHARS
            text = text[:cut].rstrip() + "\n[...]"
        return text.strip()
    except Exception as e:
        return ""


def build_preprocessed_dataset(force: bool = False) -> pd.DataFrame:
    """
    Build the full preprocessed dataset by pairing each filing with
    stock price data. Caches result as a pickle file.

    Returns:
        DataFrame with columns:
            company_name, isin, filing_date, filing_type_class,
            filing_type_specific, filing_language, pre_prices (list),
            post_return (float), label (str), filing_text (str),
            relative_path (str)
    """
    if os.path.exists(PREPROCESSED_PKL) and not force:
        print(f"Loading cached preprocessed dataset from {PREPROCESSED_PKL}")
        return pd.read_pickle(PREPROCESSED_PKL)

    # 1) Download stock prices
    print("Step 1/3: Downloading stock prices...")
    prices = download_stock_prices()
    print(f"  Got price data for {len(prices)} / {len(ISIN_TO_TICKER)} companies")

    # 2) Read metadata
    print("Step 2/3: Reading filing metadata...")
    metadata = []
    with open(METADATA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)
    print(f"  Loaded {len(metadata)} filing records")

    # 3) Pair filings with price data
    print("Step 3/3: Pairing filings with price data...")
    samples = []
    skipped_no_prices = 0
    skipped_no_window = 0
    skipped_no_text = 0

    for row in tqdm(metadata, desc="Processing filings"):
        isin = row["company_isin"]

        if isin not in prices:
            skipped_no_prices += 1
            continue

        filing_date = datetime.strptime(row["filing_date"], "%Y-%m-%d")
        result = _get_price_window(
            prices[isin], filing_date, PRE_FILING_DAYS, POST_FILING_DAYS
        )

        if result is None:
            skipped_no_window += 1
            continue

        pre_prices, post_return = result

        filing_text = _read_filing_text(row["relative_path"])
        if not filing_text:
            skipped_no_text += 1
            continue

        label = "(a)" if post_return >= 0 else "(b)"

        samples.append({
            "company_name": row["company_name"],
            "isin": isin,
            "ticker": ISIN_TO_TICKER[isin],
            "filing_date": row["filing_date"],
            "filing_year": int(row["filing_year"]),
            "filing_type_class": row["filing_type_class"],
            "filing_type_specific": row["filing_type_specific"],
            "filing_language": row["filing_language"],
            "pre_prices": pre_prices.tolist(),
            "post_return": post_return,
            "label": label,
            "filing_text": filing_text,
            "relative_path": row["relative_path"],
        })

    df = pd.DataFrame(samples)

    print(f"\n--- Dataset Statistics ---")
    print(f"Total filings in metadata:  {len(metadata)}")
    print(f"Skipped (no price data):    {skipped_no_prices}")
    print(f"Skipped (insufficient window): {skipped_no_window}")
    print(f"Skipped (no text):          {skipped_no_text}")
    print(f"Final dataset size:         {len(df)}")
    if len(df) > 0:
        print(f"Label distribution:         (a) increase: {(df['label'] == '(a)').sum()}, (b) decrease: {(df['label'] == '(b)').sum()}")
        print(f"Companies represented:      {df['isin'].nunique()}")
        print(f"Date range:                 {df['filing_date'].min()} to {df['filing_date'].max()}")
        print(f"Filing types:               {df['filing_type_specific'].nunique()} unique types")

    os.makedirs(FINANCIAL_DATA_DIR, exist_ok=True)
    df.to_pickle(PREPROCESSED_PKL)
    print(f"Saved preprocessed dataset to {PREPROCESSED_PKL}")

    return df


def load_financial_reports_splits() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load the financial reports dataset and split into train/val/test.
    Uses time-based splitting: earlier filings for training, recent for test.

    Returns:
        Tuple of (train, validation, test) Dataset objects
    """
    df = build_preprocessed_dataset()

    if len(df) == 0:
        raise RuntimeError("No samples in the preprocessed dataset. Check stock price downloads.")

    # Time-based split to avoid data leakage
    df = df.sort_values("filing_date").reset_index(drop=True)
    n = len(df)
    test_start = int(n * (1 - TEST_FRAC))
    val_start = int(n * (1 - TEST_FRAC - VAL_FRAC))

    train_df = df.iloc[:val_start].reset_index(drop=True)
    val_df = df.iloc[val_start:test_start].reset_index(drop=True)
    test_df = df.iloc[test_start:].reset_index(drop=True)

    print(f"\nSplit sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Train period: {train_df['filing_date'].min()} to {train_df['filing_date'].max()}")
    print(f"Val period:   {val_df['filing_date'].min()} to {val_df['filing_date'].max()}")
    print(f"Test period:  {test_df['filing_date'].min()} to {test_df['filing_date'].max()}")

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    print("=== Financial Reports Dataset Preparation ===\n")
    train, val, test = load_financial_reports_splits()

    print(f"\nTrain: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    if len(train) > 0:
        sample = train[0]
        print(f"\nSample keys: {list(sample.keys())}")
        print(f"Company: {sample['company_name']}")
        print(f"Filing date: {sample['filing_date']}")
        print(f"Filing type: {sample['filing_type_specific']}")
        print(f"Pre-prices length: {len(sample['pre_prices'])}")
        print(f"Post return: {sample['post_return']:.4f}")
        print(f"Label: {sample['label']}")
        print(f"Filing text preview: {sample['filing_text'][:200]}...")
