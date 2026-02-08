# Financial Reports + Stock Price Dataset for OpenTSLM

## Overview

This dataset pairs **regulatory filing text** from 15 European real estate companies with **daily stock price time series** to create a post-filing return prediction task. The model receives a company's recent price history and the content of a newly published filing, then predicts whether the stock price will increase or decrease over the following 5 trading days.

## Task Definition

| Component | Description |
|-----------|-------------|
| **Input (time series)** | 60 trading days of daily closing stock prices before the filing date, z-score normalized |
| **Input (text)** | The first ~3,000 characters of the regulatory filing (markdown) |
| **Target** | Binary classification: `(a)` stock increases, `(b)` stock decreases over the next 5 trading days |
| **Metric** | Accuracy (first-3-character match, consistent with OpenTSLM's MCQ evaluation) |

## Sample Format

Each sample is formatted into OpenTSLM's standard dictionary:

```python
{
    "pre_prompt": "You are a financial analyst specializing in European real estate equities. "
                  "Below is a {filing_type} filed by {company} on {date}.\n\n"
                  "--- Filing Content ---\n{filing_text}\n--- End of Filing ---\n\n"
                  "You also have the company's daily closing stock price for the 60 trading "
                  "days leading up to this filing.",

    "time_series": [np.array([ ... 60 normalized values ... ])],

    "time_series_text": ["This is the daily closing stock price of {company} ({ticker}) "
                         "for the 60 trading days before the filing, "
                         "it has mean {m:.4f} and std {s:.4f}:"],

    "post_prompt": "Based on the stock price trend leading up to this filing and the filing "
                   "content, will the stock price increase or decrease over the next 5 trading "
                   "days?\n(a) Increase\n(b) Decrease\nAnswer:",

    "answer": "(a)" or "(b)"
}
```

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total samples | 17,631 |
| Train split | 14,104 (2001-01-09 to 2023-03-21) |
| Validation split | 1,763 (2023-03-21 to 2024-07-08) |
| Test split | 1,764 (2024-07-09 to 2026-01-30) |
| Label balance | (a) increase: 53.4% / (b) decrease: 46.6% |
| Companies | 14 publicly traded European real estate firms |
| Time series length | 60 data points (1 channel) |
| Filing text length | Up to ~3,000 characters |
| Splitting strategy | Time-based (no data leakage) |

## Companies

| Company | Ticker | Country | ISIN | Filings |
|---------|--------|---------|------|--------:|
| Deutsche EuroShop AG | DEQ.DE | DE | DE0007480204 | 1,308 |
| TAG Immobilien AG | TEG.DE | DE | DE0008303504 | 1,746 |
| Warimpex Finanz AG | WXF.VI | AT | AT0000827209 | 612 |
| Aedifica SA | AED.BR | BE | BE0003851681 | 1,497 |
| Atenor | ATEB.BR | BE | BE0003837540 | 1,021 |
| Montea NV | MONT.BR | BE | BE0003853703 | 919 |
| Nextensa SA | NEXTA.BR | BE | BE0003770840 | 809 |
| Retail Estates SA | RET.BR | BE | BE0003720340 | 945 |
| VastNed Belgium NV | VASTB.BR | BE | BE0003754687 | 531 |
| Citycon Oyj | CTY1S.HE | FI | FI4000369947 | 2,254 |
| Selvaag Bolig ASA | SBO.OL | NO | NO0010612450 | 1,280 |
| BBI Development SA | BBD.WA | PL | PLNFI1200018 | 2,041 |
| Echo Investment SA | ECH.WA | PL | PLECHPS00019 | 1,575 |
| Triton Development SA | TRN.WA | PL | PLASMOT00030 | 1,113 |

CPI Europe AG / Immofinanz (AT0000A21KS2) was excluded because the stock was delisted following a takeover and no price data is available on Yahoo Finance.

## Filing Types

The dataset spans 37 distinct filing types across 7 broad classes:

- **Financial Reporting & Info** — earnings releases, annual/quarterly reports, financial statements
- **Annual General Meeting** — AGM information, voting results
- **Equity Information** — share transactions, voting rights, director dealings
- **Management & Remuneration** — board changes, director compensation
- **Investor Communication** — investor presentations, report publication announcements
- **M&A, Partnerships & Legal** — M&A activity, legal proceedings
- **Other** — regulatory filings, ESG reports, listing/delisting

## Languages

Filings are multilingual: English (38%), Polish (23%), Dutch (12%), French (9%), German (7%), Finnish (7%), Norwegian (4%), and traces of other languages.

## Data Pipeline

### Source Data

1. **Filing text**: 19,632 markdown files extracted from `financial_reports_sample_real_estate.zip`, organized as `{company}/{year}/{filing_type}/{date}_{lang}_{id}.md` with a `metadata.csv` index.

2. **Stock prices**: Daily OHLCV data downloaded from Yahoo Finance using the `yfinance` library, keyed by ISIN. Cached as CSV files in `data/financial_reports/stock_prices/`.

### Preprocessing Steps

For each of the 19,632 filings in `metadata.csv`:

1. Look up the company's ISIN in the stock price data.
2. Find the last trading day on or before the filing date.
3. Extract the 60 trading days of closing prices ending on that day.
4. Find the closing price 5 trading days after the filing date.
5. Compute the 5-day forward return: `(close_after - close_at_filing) / close_at_filing`.
6. Read the filing markdown text (truncated to ~3,000 characters at a paragraph boundary).
7. Assign label: `(a)` if return >= 0, `(b)` if return < 0.

Filings are dropped if: the company has no price data (1,981 dropped for Immofinanz), insufficient price history around the filing date (20 dropped), or the filing text is empty (0 dropped). This yields **17,631 usable samples**.

The result is cached as a pickle file at `data/financial_reports/preprocessed_dataset.pkl`.

### Train/Val/Test Split

The dataset uses a **time-based split** to prevent data leakage:

- **Train** (80%): oldest filings, up to ~March 2023
- **Validation** (10%): March 2023 to July 2024
- **Test** (10%): July 2024 to January 2026

## File Structure

```
OpenTSLM/
├── src/time_series_datasets/financial_reports/
│   ├── __init__.py
│   ├── README.md                          # This file
│   ├── financial_reports_loader.py        # Data download, preprocessing, splitting
│   └── FinancialReportsQADataset.py       # QADataset subclass for OpenTSLM
├── data/financial_reports/
│   ├── stock_prices/                      # Cached Yahoo Finance CSVs (one per ISIN)
│   └── preprocessed_dataset.pkl           # Precomputed dataset (17,631 samples)
└── curriculum_learning.py                 # Modified: added stage6_financial_reports
```

## Usage

### As a standalone dataset

```python
from time_series_datasets.financial_reports.FinancialReportsQADataset import FinancialReportsQADataset

train = FinancialReportsQADataset(split="train", EOS_TOKEN="</s>")
val   = FinancialReportsQADataset(split="validation", EOS_TOKEN="</s>")
test  = FinancialReportsQADataset(split="test", EOS_TOKEN="</s>")

sample = train[0]
# sample keys: pre_prompt, time_series, time_series_text, post_prompt, answer,
#              label, company_name, filing_date, post_return
```

### As a curriculum learning stage

```bash
python curriculum_learning.py \
    --model OpenTSLMSP \
    --stages stage6_financial_reports \
    --llm_id meta-llama/Llama-3.2-1B
```

### Rebuild the preprocessed data from scratch

```python
from time_series_datasets.financial_reports.financial_reports_loader import build_preprocessed_dataset

df = build_preprocessed_dataset(force=True)
```

## Configuration

Key parameters in `financial_reports_loader.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PRE_FILING_DAYS` | 60 | Trading days of price history before the filing |
| `POST_FILING_DAYS` | 5 | Trading days after the filing to measure the return |
| `MAX_FILING_TEXT_CHARS` | 3000 | Maximum characters of filing text included |
| `TEST_FRAC` | 0.1 | Fraction of data for the test split |
| `VAL_FRAC` | 0.1 | Fraction of data for the validation split |

## Dependencies

- `yfinance` — for downloading stock price data from Yahoo Finance
- `pandas`, `numpy` — data processing
- `torch`, `datasets` — PyTorch and HuggingFace Datasets (already required by OpenTSLM)
