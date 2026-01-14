# Time Series Encoders for OpenTSLM

## Overview

This document describes the time series encoders implemented for OpenTSLM. These encoders allow using various pretrained foundation models for time series analysis, similar to how Chronos2 was integrated in PR #41.

## Available Encoders

### 1. Chronos2 (Already in PR #41)
- **Models**: `amazon/chronos-t5-tiny`, `mini`, `small`, `base`, `large`
- **Pretrained**: Yes, from HuggingFace
- **Padding**: Left padding with NaN values
- **Architecture**: T5-based encoder-decoder

### 2. PatchTST
- **Models**: `ibm/patchtst-etth1-pretrain`, `etth2`, `ettm1`, `ettm2`, `weather`, `electricity`
- **Pretrained**: Yes, from HuggingFace
- **Padding**: Right padding
- **Patching**: Non-overlapping patches (default size 12-16)
- **Note**: Pretrained models expect specific channel counts (e.g., ETTh1 expects 7 channels)

### 3. TimesFM (Google)
- **Models**: `google/timesfm-1.0-200m`
- **Pretrained**: Yes (requires special installation)
- **Padding**: Right padding
- **Patching**: 32-length patches
- **Fallback**: CNN-based encoder when library unavailable

### 4. MOMENT (CMU)
- **Models**: `AutonLab/MOMENT-1-small`, `base`, `large`
- **Pretrained**: Yes, from HuggingFace
- **Padding**: Center padding
- **Patching**: 8-length patches
- **Fallback**: Transformer-based encoder when library unavailable

### 5. Lag-Llama
- **Models**: `time-series-foundation-models/Lag-Llama`
- **Pretrained**: Yes, from HuggingFace
- **Padding**: Left padding with NaN (like Chronos)
- **Architecture**: LLaMA-based
- **Fallback**: LLaMA/Transformer architecture when library unavailable

## Installation

### Basic Requirements
```bash
pip install torch transformers>=4.35.0 chronos-forecasting
```

### Optional Libraries for Full Support
```bash
# For TimesFM (Google)
pip install timesfm

# For MOMENT
pip install momentfm

# For Lag-Llama
pip install lag-llama gluonts
```

## Usage Example

```python
from opentslm.model.encoder.PatchTSTEncoder import PatchTSTEncoder
from opentslm.model.encoder.Chronos2Encoder import Chronos2Encoder
from opentslm.model.encoder.MomentEncoder import MomentEncoder

# Initialize an encoder with pretrained weights
encoder = PatchTSTEncoder(
    model_name="ibm/patchtst-etth1-pretrain",
    output_dim=128,
    device="cuda"
)

# Or use Chronos2 (from PR #41)
encoder = Chronos2Encoder(
    model_name="amazon/chronos-t5-tiny",
    output_dim=128,
    device="cuda"
)

# Process time series data
# Input shape: (batch_size, seq_len, n_features)
time_series = torch.randn(4, 100, 1)
embeddings = encoder(time_series)
# Output shape: (batch_size, output_dim)
```

## Enhanced Base Class

All encoders inherit from `EnhancedTimeSeriesEncoderBase` which provides:

1. **Padding Strategies**:
   - LEFT: For autoregressive models (Chronos, Lag-Llama)
   - RIGHT: For standard transformers (PatchTST, Informer)
   - CENTER: For bidirectional models (MOMENT)
   - NONE: For CNN-based models

2. **Patching Strategies**:
   - NON_OVERLAPPING: Fixed-size patches
   - OVERLAPPING: Sliding window patches
   - NONE: No patching

## Key Implementation Details

### Padding Handling
Different encoders require different padding strategies:
- **Chronos2 & Lag-Llama**: Use left padding with NaN values (autoregressive)
- **PatchTST**: Uses right padding with zeros
- **MOMENT**: Uses center padding for bidirectional attention

### Channel Handling
- Most pretrained models expect specific channel counts
- ETTh1 PatchTST expects 7 channels
- Chronos2 expects univariate (1 channel)
- Encoders automatically handle channel mismatches

### Fallback Modes
When pretrained model libraries aren't available:
- TimesFM: Falls back to CNN-based encoder
- MOMENT: Falls back to transformer encoder
- Lag-Llama: Falls back to LLaMA architecture or standard transformer

## Testing

Run the unified test suite:
```bash
python test/test_all_encoders.py
```

Or quick test:
```bash
python test/test_quick_encoders.py
```

## Integration with OpenTSLMFlamingo

To use a new encoder in OpenTSLMFlamingo, update the encoder initialization:

```python
# In OpenTSLMFlamingo.__init__
if encoder_type == "patchtst":
    from opentslm.model.encoder.PatchTSTEncoder import PatchTSTEncoder
    self.encoder = PatchTSTEncoder(
        model_name="ibm/patchtst-etth1-pretrain",
        output_dim=ENCODER_OUTPUT_DIM,
        device=self.device
    )
```

Also update padding logic in `process_batch`:
```python
if self.encoder_type in ["chronos2", "lag-llama"]:
    # Use left padding with NaN
    padding = torch.full(padding_shape, torch.nan, ...)
    padded = torch.cat([padding, ts], dim=1)
elif self.encoder_type == "moment":
    # Use center padding
    # ... center padding logic
else:
    # Use right padding (default)
    padding = torch.zeros(padding_shape, ...)
    padded = torch.cat([ts, padding], dim=1)
```

## Known Issues

1. **Chronos Version**: The current chronos-forecasting package (v2.2.2) may have compatibility issues with some model checkpoints.

2. **PatchTST Channels**: Pretrained PatchTST models are trained on specific datasets with fixed channel counts. You may need to:
   - Use the correct number of channels for your dataset
   - Or retrain/fine-tune the model for your specific use case

3. **Missing Weights Warning**: Some pretrained models show "weights not initialized" warnings. This is expected for certain components that need task-specific training.

## Future Work

1. Add more encoders:
   - Informer
   - FEDformer
   - TimesNet (CNN-based)
   - Autoformer

2. Implement unified preprocessing pipeline for different input formats

3. Add automatic model selection based on data characteristics

4. Implement ensemble methods combining multiple encoders