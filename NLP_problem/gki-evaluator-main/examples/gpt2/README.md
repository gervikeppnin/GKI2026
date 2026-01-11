# GPT-2 Example (Conceptual)

## Overview

This directory contains a **conceptual implementation** of using GPT-2 for byte-level prediction in the GKÍ evaluator. However, it demonstrates a fundamental challenge: **pre-trained models like GPT-2 are too large to fit within the 1MB submission limit**.

## The Size Problem

| Model | Uncompressed | 8-bit | 4-bit | 2-bit | Target |
|-------|-------------|-------|-------|-------|--------|
| distilgpt2 | ~240 MB | ~60 MB | ~30 MB | ~15 MB | **1 MB** |
| gpt2 | ~500 MB | ~125 MB | ~62 MB | ~31 MB | **1 MB** |

Even with aggressive 2-bit quantization, distilgpt2 would still be **15x too large**.

## Potential Solutions

### 1. Extreme Quantization + Pruning
- Combine 2-bit quantization with 95%+ sparsity
- Might achieve ~1MB but with significant performance loss
- Requires custom quantization/pruning pipeline

### 2. Knowledge Distillation
- Train a tiny student model (<250k params) using GPT-2 as teacher
- Student learns to mimic GPT-2's predictions on byte sequences
- More practical for this competition

### 3. Lookup Table Approach
- Pre-compute GPT-2 predictions for common byte patterns
- Package compressed lookup table (e.g., using Bloom filters)
- Trade model size for inference speed

### 4. Hybrid n-gram + Neural
- Use n-gram statistics for common patterns
- Small neural network for rare sequences
- Balance size and performance

## BPE-to-Byte Mapping Challenge

GPT-2 uses **Byte-Pair Encoding (BPE)** tokenization with ~50k tokens, but we need to predict individual bytes (256 values). The mapping is non-trivial:

```
Byte sequence: [72, 101, 108, 108, 111]  # "Hello"
GPT-2 tokens:  [15496]                     # Single token
Next byte:     [32]                        # " " (space)
Next token:    Could be [220], [11], etc.  # Multiple possibilities
```

The implementation needs to:
1. Map byte contexts to token sequences
2. Get token-level predictions from GPT-2
3. Map token probabilities back to byte probabilities
4. Handle multi-byte UTF-8 sequences

## Files

- `submission/model.py`: Conceptual GPT-2 wrapper showing the approach
- `package.py`: Packaging script (creates minimal demo submission)
- `README.md`: This file

## Usage

```bash
# Create demo submission (no actual model weights)
python package.py

# This will create a ~3KB submission (just the code, no weights)
# It would fail evaluation because model weights are missing
```

## Conclusion

**GPT-2 is not practical for this competition's 1MB limit.** However, this example demonstrates:

1. How byte-level prediction would interface with a token-based model
2. The architectural challenges of BPE-to-byte mapping
3. Why custom tiny models (like the MLP baseline) are more suitable

For actual competition submissions, consider:
- Training byte-level models from scratch
- Using simpler architectures (RNNs, small Transformers)
- Leveraging compression and quantization from the start
