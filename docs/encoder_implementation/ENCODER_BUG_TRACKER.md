# Encoder Bug Tracker

## Status Summary
- ✅ MOMENT: Working correctly
- ❌ Chronos: Tokenization issues
- ❌ PatchTST: Dimension checking bug
- ❌ TimesFM: Device placement issues

## Bug List

### 1. Chronos Encoder
**Error**: `too many values to unpack (expected 2)`
**Location**: ChronosEncoderFixed forward method
**Root Cause**: The tokenizer.context_input_transform returns 3 values (tokens, attention_mask, scale) but code expects 2
**Status**: FIXING

### 2. PatchTST Encoder
**Error**: `The defined number of input channels (7) in the config has to be the same as the number of channels`
**Location**: PatchTSTEncoderFixed forward method
**Root Cause**: Model expects specific channel configuration, dimension checking in transformers library
**Status**: FIXING

### 3. TimesFM Encoder
**Error**: `Expected all tensors to be on the same device, but got mat1 is on cuda:0, different from other`
**Location**: TimesFMWrapper projection layer
**Root Cause**: Model weights initialization on wrong device
**Status**: FIXING

## Attempted Solutions

### Chronos
- ❌ Using ChronosPipeline.predict() - wrong parameter name (context vs inputs)
- ❌ Direct tokenizer access - device mismatch with tokenizer_bins
- ❌ Fixed 3-value return - still has device issues with boundaries tensor
- ⏳ Need to move ALL tokenizer components to GPU

### PatchTST
- ❌ Using model.forward() directly - dimension validation bug
- ❌ Manual patchification - encoder expects different input format
- ❌ patch_stride was 1 instead of 12, causing wrong num_patches
- ⏳ Need to fix stride and reshape logic

### TimesFM
- ❌ Using TimesFm class - doesn't exist in package
- ❌ Using forecast() with horizon_len parameter - wrong parameter name
- ❌ Linear layer initialization - weights on CPU not GPU
- ⏳ Need to move ALL model components to GPU after initialization

## Next Steps
1. Fix Chronos tokenization return value handling
2. Fix PatchTST by directly calling encoder components
3. Fix TimesFM device initialization
4. Run comprehensive tests