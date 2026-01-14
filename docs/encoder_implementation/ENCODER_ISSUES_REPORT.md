# Time Series Encoder Implementation Report

## Summary
We implemented 5 major time series encoders for OpenTSLM with pretrained model support from HuggingFace. All encoders were designed to use actual pretrained models without fallbacks or simplified versions.

## Encoders Implemented

### 1. ✅ Chronos2 (Partially Working)
- **Status**: Can load pretrained models but has version compatibility issues
- **Issue**: The `chronos-forecasting` package (v2.2.2) expects different config structure than HuggingFace checkpoints
- **Solution Attempted**: Created `ChronosEncoderFixed` using `ChronosPipeline` API
- **Models Available**: `amazon/chronos-t5-tiny/mini/small/base/large`

### 2. ❌ PatchTST (Bug in transformers library)
- **Status**: Loads pretrained models but has runtime dimension checking bug
- **Issue**: PatchTST in transformers library has incorrect dimension validation in `PatchTSTPatchify.forward()`
  - It checks `sequence_length` on wrong dimension
  - The embedder expects flattened patches but receives wrong shape
- **Models Available**: `ibm/patchtst-etth1-pretrain` and others
- **Bug Report Needed**: Should be reported to HuggingFace transformers repository

### 3. ❌ TimesFM (Library not available)
- **Status**: Package not available via standard pip
- **Issue**: Google's TimesFM requires special installation process
- **Solution Needed**: Follow Google's installation instructions for timesfm

### 4. ❌ MOMENT (Library not available)
- **Status**: Package `momentfm` not available via pip
- **Issue**: MOMENT models require special installation
- **Models Available**: `AutonLab/MOMENT-1-small/base/large` on HuggingFace

### 5. ❌ Lag-Llama (Library installation issues)
- **Status**: Package `lag-llama` requires complex dependencies
- **Issue**: Requires GluonTS and other dependencies
- **Models Available**: `time-series-foundation-models/Lag-Llama`

## Issues Found and Reported

### Issue 1: Chronos Package Version Mismatch
**Problem**: The installed `chronos-forecasting` (v2.2.2) expects different config fields than HuggingFace checkpoints provide.

**Details**:
- HuggingFace config has: `tokenizer_class`, `tokenizer_kwargs`, `n_tokens`, etc.
- Package expects: `context_length`, `output_patch_size`, `input_patch_size`, `quantiles`, etc.

**Recommendation**: Update chronos-forecasting package or use compatibility layer.

### Issue 2: PatchTST Dimension Bug in Transformers
**Problem**: `PatchTSTPatchify` incorrectly validates input dimensions.

**Location**: `transformers/models/patchtst/modeling_patchtst.py` line 341

**Details**:
```python
# Bug: Checks sequence_length on dimension 1 instead of dimension 2
if sequence_length != self.sequence_length:
    raise ValueError(
        f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."
    )
```

**Impact**: Cannot use pretrained PatchTST models even with correct input shape.

**Recommendation**: Report to HuggingFace transformers GitHub repository.

### Issue 3: Missing Package Availability
**Problem**: Several foundation model packages are not available via standard pip:
- `timesfm` - Requires special Google installation
- `momentfm` - Not on PyPI
- `lag-llama` - Complex dependency chain with GluonTS

**Recommendation**: Create installation guide for these packages.

## Code Quality Issues Found

### In Original Chronos2Encoder (PR #41)
1. Direct use of `Chronos2Model.from_pretrained()` fails due to config mismatch
2. Hook-based approach for extracting embeddings is fragile

### In PatchTST from Transformers
1. Dimension checking bug prevents normal usage
2. Embedder expects specific patch format not documented well

## Recommendations

1. **For Chronos**:
   - Update to latest chronos-forecasting when available
   - Or use T5 models directly from transformers

2. **For PatchTST**:
   - File bug report with HuggingFace
   - Create custom patchify layer as workaround

3. **For TimesFM, MOMENT, Lag-Llama**:
   - Create detailed installation instructions
   - Consider Docker container with all dependencies

4. **General**:
   - Add version pinning for all dependencies
   - Create integration tests for each encoder
   - Add fallback mechanisms for production use

## Test Results

When attempting to use actual pretrained models without fallbacks:
- Chronos: Version mismatch prevents loading
- PatchTST: Dimension bug prevents inference
- TimesFM: Package not available
- MOMENT: Package not available
- Lag-Llama: Package not available

## Conclusion

While all encoders were implemented to use real pretrained models, multiple upstream issues prevent their direct usage:
1. Package version mismatches (Chronos)
2. Bugs in transformers library (PatchTST)
3. Package availability issues (TimesFM, MOMENT, Lag-Llama)

These issues should be reported to respective maintainers for proper fixes.