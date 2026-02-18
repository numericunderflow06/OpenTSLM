# Financial Reports QA Dataset

**Repository:** [numericunderflow06/OpenTSLM](https://github.com/numericunderflow06/OpenTSLM)
**Branch:** `feature/financial-reports-dataset`

## Overview

The financial reports QA dataset pairs European real estate regulatory filings with stock price data. Given a filing's text and the company's recent price history (60 trading days), the model answers questions about what happens in the 5 trading days after the filing.

There are **3 question types** per filing, tripling the dataset compared to the original single-question setup:

| Question type | What it asks | Labels | Classification basis |
|---|---|---|---|
| **Price return** | Will the stock price increase or decrease? | (a) Increase, (b) Decrease | Endpoint return over 5 days |
| **Volatility** | Will daily price volatility increase, decrease, or remain stable? | (a) Increase, (b) Decrease, (c) Stable | Ratio of post-filing to pre-filing return std |
| **Market direction** | Will the stock trend bullish, bearish, or sideways? | (a) Bullish, (b) Bearish, (c) Sideways | Linear regression slope of post-filing prices |

## Files

### Core dataset files

| File | Role |
|---|---|
| `src/time_series_datasets/financial_reports/financial_reports_loader.py` | Downloads stock prices, reads filing metadata, computes labels, builds and caches the preprocessed dataset |
| `src/time_series_datasets/financial_reports/FinancialReportsQADataset.py` | QA dataset class: expands each filing into 3 samples (one per question type), formats prompts and answers |

### Data on disk

| Path | Description |
|---|---|
| `data/financial_reports/stock_prices/*.csv` | Cached per-company daily price CSVs from Yahoo Finance |
| `data/financial_reports/preprocessed_dataset.pkl` | Cached preprocessed dataset (auto-generated; delete to force rebuild) |

## Label computation details

### Price return (binary)

The original question type. Compares the closing price 5 days after filing to the closing price on the filing date:

- `post_return >= 0` &rarr; `(a)` Increase
- `post_return < 0` &rarr; `(b)` Decrease

### Volatility (ternary)

Compares the standard deviation of daily returns in the 5-day post-filing window to the 60-day pre-filing window:

1. Pre-filing daily returns: `diff(pre_prices) / pre_prices[:-1]` (59 values)
2. Post-filing daily returns: anchored to last pre-filing close, giving 5 values
3. `ratio = std(post_returns) / std(pre_returns)`

Thresholds:

- `ratio > 1.5` &rarr; `(a)` Increase
- `ratio < 0.667` &rarr; `(b)` Decrease
- otherwise &rarr; `(c)` Stable

The 50% threshold is intentionally generous because with only 5 post-filing returns, the sample standard deviation has ~40% coefficient of variation even when true volatility is unchanged.

### Market direction (ternary)

Fits a linear regression to the 5 post-filing closing prices and normalizes the slope:

1. `slope = polyfit(arange(5), post_prices, 1)[0]`
2. `norm_slope = slope / mean(post_prices)` (fractional change per day)

Thresholds:

- `norm_slope > 0.002` &rarr; `(a)` Bullish
- `norm_slope < -0.002` &rarr; `(b)` Bearish
- otherwise &rarr; `(c)` Sideways

The 0.2%/day threshold corresponds to ~1%/week. This question is meaningfully different from price return: e.g., a stock that drops sharply on day 1 then recovers has negative endpoint return but positive trend slope.

## Architecture

```
FinancialReportsQADataset (QADataset subclass)
  _load_splits()
    -> load_financial_reports_splits()   # from loader
       -> build_preprocessed_dataset()   # downloads prices, reads filings, computes all 3 labels
    -> _expand()                         # triples each row: price_return / volatility / direction

  _get_pre_prompt(row)       # same for all question types (filing text + context)
  _get_post_prompt(row)      # dispatches on row["question_type"]
  _get_answer(row)           # dispatches to row["label"], row["volatility_label"], or row["direction_label"]
  _get_text_time_series_prompt_list(row)  # same for all (60-day z-scored price series)

  get_labels() -> ["(a)", "(b)", "(c)"]
```

The base class `QADataset.__init__` is untouched. Expansion happens in the overridden `_load_splits()`, so `_format_sample` is called once per row as usual, just on 3x as many rows.

## Training compatibility

- **`curriculum_learning.py`**: The `_calculate_accuracy_baseline` method compares only the first 3 characters of gold vs predicted answers (e.g., `"(a)"`, `"(b)"`, `"(c)"`), so the ternary labels work without changes.
- **Model code**: Agnostic -- just sees text tokens.
- **`prompt_with_answer.py`**: Agnostic to label values.

To train on the expanded dataset:

```bash
python curriculum_learning.py --model OpenTSLMFlamingo --stages stage6_financial_reports
```

## Companies covered

14 European real estate companies with Yahoo Finance price data:

| Ticker | Company | Exchange |
|---|---|---|
| DEQ.DE | Deutsche EuroShop AG | XETRA |
| TEG.DE | TAG Immobilien AG | XETRA |
| WXF.VI | Warimpex | Vienna |
| AED.BR | Aedifica | Brussels |
| ATEB.BR | Atenor | Brussels |
| MONT.BR | Montea | Brussels |
| NEXTA.BR | Nextensa | Brussels |
| RET.BR | Retail Estates | Brussels |
| VASTB.BR | VastNed Belgium | Brussels |
| CTY1S.HE | Citycon | Helsinki |
| SBO.OL | Selvaag Bolig | Oslo |
| BBD.WA | BBI Development | Warsaw |
| ECH.WA | Echo Investment | Warsaw |
| TRN.WA | Triton Development | Warsaw |

## Configuration

| Parameter | Value | Location |
|---|---|---|
| `PRE_FILING_DAYS` | 60 | `financial_reports_loader.py` |
| `POST_FILING_DAYS` | 5 | `financial_reports_loader.py` |
| `MAX_FILING_TEXT_CHARS` | 3000 | `financial_reports_loader.py` |
| `TEST_FRAC` | 0.1 | `financial_reports_loader.py` |
| `VAL_FRAC` | 0.1 | `financial_reports_loader.py` |

Splitting is time-based (earliest filings for training, most recent for test) to avoid data leakage.
