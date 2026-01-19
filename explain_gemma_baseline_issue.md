# Understanding the Gemma "Raw Inference" Problem

## What Happens in Baseline Evaluation

When you do "raw inference" with Gemma (or any LLM) using time series as text tokens, here's the fundamental issue:

### 1. Time Series → Text Token Explosion

Let's look at what happens when you convert time series to text using the Gruver et al. tokenizer (gruver_llmtime_tokenizer.py):

**Example time series value: `0.527`**

Using the Llama formatter (base-10, precision-3):
```
Input: 0.527
Output: " 5 2 7 ,"
```

Each digit becomes a **separate token** that the tokenizer must encode!

### 2. The Token Explosion Problem

Consider a typical time series with 1000 data points:

**OpenTSLM approach:**
- 1000 data points → encoded by CNN/Transformer encoder
- Patchified (patch_size=4): 1000 / 4 = 250 patches
- Projected to LLM space: **250 embedding tokens**
- Total tokens to LLM: **250 tokens**

**Raw text approach (baseline):**
- 1000 data points, each formatted as ` 5 2 7 ,` (4 tokens per value)
- Total text tokens: **4000 tokens**
- Plus separators and formatting: **~4500+ tokens**

### 3. Context Length Problem

Most LLMs have context limits:
- Gemma-2B: 8192 tokens
- Llama-3.2-1B: 8192 tokens (in some configs)

**For the Polymarket dataset:**
- Time series can be very long (hundreds to thousands of points)
- With text encoding, a single sample can easily exceed 8192 tokens
- **Result: The prompt gets truncated!**

### 4. Tokenization Inefficiency

Here's what actually happens in the tokenizer:

```python
# Time series as raw text
prompt = "The time series is: 5 2 7 , 4 3 1 , ..."

# Gemma tokenizer sees each digit as a separate text element
tokens = tokenizer(prompt)
# Result: ['The', 'time', 'series', 'is', ':', '5', '2', '7', ',', ...]
#         Or worse: ['The', 'time', 'series', 'is', ':', ' ', '5', ' ', '2', ...]
```

Each number gets broken into multiple tokens, causing:
1. **Massive token count** - easily 10-20x more tokens than OpenTSLM
2. **Poor semantic representation** - digits are treated as text, not numerical values
3. **Context overflow** - truncation means the model never sees the full time series
4. **Loss of numerical meaning** - "527" as three separate tokens has no numerical semantics

### 5. Why OpenTSLM Works

OpenTSLM avoids this by:

```python
# Instead of text tokens, use learned embeddings
time_series = [0.527, 0.431, ...]  # 1000 values
↓
encoder(time_series)  # CNN + Transformer
↓
patches: [embed_1, embed_2, ..., embed_250]  # 250 patches
↓
projector(patches) → LLM embedding space
↓
250 meaningful embedding tokens (vs 4000+ text tokens)
```

### Concrete Example

**Polymarket data with 500 time series values:**

**Baseline (text tokens):**
```
Question: "Will it rain tomorrow?"
Time series is: 5 2 7 , 4 3 1 , 6 8 2 , ... [continues for 2000+ tokens]
Answer: [TRUNCATED - model never sees this]
```

**OpenTSLM (embeddings):**
```
Question: "Will it rain tomorrow?"
<125 embedding tokens representing the full time series>
Answer: Yes [model sees the full context]
```

## The Bottom Line

**Yes, there IS a problem with raw inference using Gemma (or any LLM) on time series as text:**

1. **Token explosion**: 10-20x more tokens than OpenTSLM
2. **Context overflow**: Exceeds model's max context length
3. **Truncation**: Model never sees complete data or question
4. **Poor representation**: Digits as text lose numerical meaning
5. **Can't actually generate meaningful answers** because the model can't fit the full prompt in context

This is why OpenTSLM's approach (encoding time series as embeddings) is so much better than treating time series as raw text tokens.
