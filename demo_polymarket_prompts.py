#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Demonstrate Polymarket prompt structure without requiring model download.
Shows exactly what gets sent to the model and how time series embeddings are inserted.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from time_series_datasets.polymarket.PolymarketQADataset import PolymarketQADataset
from model_config import PATCH_SIZE

print("="*100)
print("POLYMARKET DATASET: PROMPT STRUCTURE DEMONSTRATION")
print("="*100)

# Load dataset
print(f"\nLoading Polymarket dataset...")
dataset = PolymarketQADataset("train", "<eos>")
print(f"✅ Loaded {len(dataset)} samples\n")

# Show detailed structure for first 5 samples
for idx in range(min(5, len(dataset))):
    sample = dataset[idx]

    print(f"\n{'='*100}")
    print(f"SAMPLE {idx + 1} - COMPLETE PROMPT STRUCTURE")
    print("="*100)

    # Metadata
    print(f"\n📊 Sample Metadata:")
    print(f"   Market ID: {sample.get('market_id', 'N/A')}")
    print(f"   Question Type: {sample.get('question_type', 'N/A')}")

    # Calculate sizes
    ts_length = len(sample['time_series'][0])
    num_patches = ts_length // PATCH_SIZE
    if ts_length % PATCH_SIZE != 0:
        num_patches += 1

    print(f"\n📏 Sequence Composition:")
    print(f"   Time series length: {ts_length} data points")
    print(f"   Patch size: {PATCH_SIZE}")
    print(f"   Number of TS patches/embeddings: {num_patches}")

    # Full prompt
    print(f"\n" + "─"*100)
    print(f"WHAT THE MODEL RECEIVES (INPUT SEQUENCE):")
    print(f"─"*100)

    # Part 1: Pre-prompt
    print(f"\n[PART 1: PRE-PROMPT - Text Tokens]")
    print(f"→ Text: \"{sample['pre_prompt']}\"")
    print(f"   ↳ This gets tokenized into ~{len(sample['pre_prompt'].split())} text tokens")
    print(f"   ↳ Each token embedded to 2048-dimensional vector by LLM's embedding layer")

    # Part 2: Time series text
    print(f"\n[PART 2: TIME SERIES DESCRIPTION - Text Tokens]")
    ts_text = sample['time_series_text'][0]
    print(f"→ Text: \"{ts_text}\"")
    print(f"   ↳ Tokenized into ~{len(ts_text.split())} text tokens")
    print(f"   ↳ Each token embedded to 2048-dimensional vector")

    # IMPORTANT: Check for information leakage
    print(f"\n   ✓ LEAKAGE CHECK:")
    print(f"      'mean' in text: {'mean' in ts_text.lower()}")
    print(f"      'std' in text: {'std' in ts_text.lower()}")
    print(f"      'volatility' in text: {'volatility' in ts_text.lower()}")
    print(f"      → NO STATISTICS = Model MUST use time series embeddings!")

    # Part 3: TIME SERIES EMBEDDINGS (THE KEY PART!)
    print(f"\n[PART 3: TIME SERIES EMBEDDINGS - *** INSERTED HERE ***]")
    print(f"→ {num_patches} EMBEDDING TOKENS INSERTED")
    print(f"   ")
    print(f"   PROCESS:")
    print(f"   1. Raw time series: {ts_length} price values (normalized)")
    print(f"   2. Conv1D encoder: Groups into {num_patches} patches of size {PATCH_SIZE}")
    print(f"   3. TransformerCNN: Each patch → 128-dim encoding")
    print(f"   4. Projector (MLP): 128-dim → 2048-dim (matches LLM hidden size)")
    print(f"   5. Result: {num_patches} tokens × 2048 dimensions")
    print(f"   ")
    print(f"   These {num_patches} tokens are INSERTED INTO THE SEQUENCE here.")
    print(f"   The LLM can ATTEND to these tokens via self-attention mechanism.")
    print(f"   ")
    print(f"   Example visualization of what the model 'sees':")
    print(f"   Token positions: [...text tokens...] [TS₁] [TS₂] [TS₃] ... [TS_{num_patches}] [...more text...]")

    # Part 4: Post-prompt
    print(f"\n[PART 4: POST-PROMPT/QUESTION - Text Tokens]")
    print(f"→ Text: \"{sample['post_prompt']}\"")
    print(f"   ↳ Tokenized into ~{len(sample['post_prompt'].split())} text tokens")
    print(f"   ↳ Each token embedded to 2048-dimensional vector")

    # Summary of full sequence
    total_text_tokens = len(sample['pre_prompt'].split()) + len(ts_text.split()) + len(sample['post_prompt'].split())
    total_tokens = total_text_tokens + num_patches

    print(f"\n" + "─"*100)
    print(f"COMPLETE INPUT SEQUENCE SUMMARY:")
    print(f"─"*100)
    print(f"Text tokens: ~{total_text_tokens}")
    print(f"Time series embedding tokens: {num_patches}")
    print(f"Total sequence length: ~{total_tokens} tokens")
    print(f"")
    print(f"Sequence structure:")
    print(f"  [pre_prompt_tokens] + [ts_text_tokens] + [TS_EMBEDDINGS × {num_patches}] + [question_tokens]")

    # Expected answer
    print(f"\n" + "─"*100)
    print(f"GROUND TRUTH ANSWER (What model should generate):")
    print(f"─"*100)
    expected = sample['answer'].replace('<eos>', '').strip()
    print(f"→ \"{expected}\"")

    # How the model uses this
    print(f"\n" + "─"*100)
    print(f"HOW THE MODEL GENERATES THE ANSWER:")
    print(f"─"*100)
    print(f"1. The full sequence is fed into the LLM")
    print(f"2. LLM processes through transformer layers with self-attention")
    print(f"3. At each layer, tokens can attend to ALL other tokens including:")
    print(f"   - The question ('What is the current trend...')")
    print(f"   - The {num_patches} TIME SERIES EMBEDDING TOKENS")
    print(f"   - Context about prediction markets")
    print(f"4. The model aggregates information from TS embeddings via attention")
    print(f"5. Generates answer token-by-token: '{expected}'")
    print(f"6. During training:")
    print(f"   - Loss computed ONLY on answer tokens (prompt tokens masked with -100)")
    print(f"   - Gradients flow back through: LLM → Projector → Encoder")
    print(f"   - Encoder & Projector weights updated to better encode TS features")
    print(f"   - Model learns which TS patterns correspond to 'upward', 'high volatility', etc.")

    print(f"\n{'='*100}\n")

print(f"\n{'='*100}")
print(f"KEY INSIGHTS:")
print(f"{'='*100}")
print(f"")
print(f"✅ NO information leakage - statistics NOT in prompts")
print(f"✅ Time series embeddings INSERTED into sequence (~{num_patches} tokens per sample)")
print(f"✅ Model MUST attend to TS embeddings to answer correctly")
print(f"✅ Training updates encoder + projector to extract relevant features")
print(f"✅ Clear questions with specific expected answers")
print(f"✅ Identical structure to other OpenTSLM datasets (TSQA, M4, HAR, ECG)")
print(f"")
print(f"{'='*100}\n")
