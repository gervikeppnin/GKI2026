# Comprehensive GKÍ Evaluator Testing Plan

## Overview
Create an extensive testing framework for the GKÍ Evaluator competition system to validate:
- Scoring metric correctness and soundness
- Docker security and sandboxing
- Baseline model evaluations (dummy + MLP)
- Integration with pre-trained HuggingFace models (GPT-2)
- Edge cases, error handling, and malicious inputs

---

## Phase 1: Environment Setup & Dependencies

### 1.1 Build Docker Image
- Build the evaluation Docker image: `docker build -t gki_evaluator -f docker/Dockerfile docker/`
- Verify image contains all required ML frameworks (PyTorch, TensorFlow, JAX, transformers)

### 1.2 Install Evaluator Package
- Create Python virtual environment
- Install evaluator: `pip install -e .`
- Verify dependencies: torch, datasets, tqdm

### 1.3 Prepare Test Dataset
- Download HuggingFace dataset for evaluation (small subset of fineweb-2 or similar)
- Create test dataset using: `python -c "from datasets import Dataset; Dataset.from_dict({'text': ['Test text here']}).save_to_disk('data/')"`
- Ensure dataset is small enough for quick testing (~1000-5000 bytes)

**Critical Files:**
- `docker/Dockerfile`
- `pyproject.toml`
- `src/gki_evaluator/evaluate.py`

---

## Phase 2: Baseline Model Validation

### 2.1 Test Dummy Baseline
**Goal:** Verify uniform distribution model scores ~8.0 bits/byte

**Steps:**
1. Package dummy submission: `cd examples/dummy && python package.py`
2. Run evaluation: `python -m gki_evaluator.evaluate --submission examples/dummy/output/submission.zip --test-data data/`
3. Verify output/score.json contains score ≈ 8.0 (±0.1 tolerance)

**Expected Result:** Score should be ~8.0 because uniform logits → log₂(256) = 8 bits

**Critical Files:**
- `examples/dummy/submission/model.py`
- `examples/dummy/package.py`

### 2.2 Test MLP Baseline
**Goal:** Verify trained model performs better than dummy baseline

**Steps:**
1. Train MLP: `cd examples/mlp && python train.py`
2. Verify `output/submission.zip` is created and < 1 MB
3. Run evaluation: `python -m gki_evaluator.evaluate --submission examples/mlp/output/submission.zip --test-data data/`
4. Verify score < 8.0 (should be 5-7 range for simple patterns)

**Expected Result:** MLP learns byte patterns, achieving lower loss than uniform distribution

**Critical Files:**
- `examples/mlp/train.py`
- `examples/mlp/submission/model.py`

---

## Phase 3: Pre-trained Model Integration (GPT-2)

### 3.1 Create GPT-2 Submission Wrapper
**Goal:** Adapt pre-trained HuggingFace GPT-2 for byte-level prediction

**Implementation:**
- Create `examples/gpt2/submission/model.py` with:
  - Load GPT-2 tokenizer and model (distilgpt2 for size constraints)
  - Convert byte contexts to tokens using tokenizer
  - Extract logits for next token
  - Map token logits back to byte probabilities (using byte-level encoding)
  - Handle tokenization mismatches (GPT-2 uses BPE, not raw bytes)

**Challenges:**
- GPT-2 is BPE-based, not byte-based → need conversion layer
- Model size: distilgpt2 is ~240MB uncompressed → **won't fit in 1MB limit**
- Alternative approach: Use model as oracle, pre-compute predictions, package lookup table

**Fallback Strategy (if full model doesn't fit):**
- Create a distilled version: train small byte-level model using GPT-2 as teacher
- Or: Use GPT-2 API locally to generate predictions, package as probability tables

### 3.2 Package and Test GPT-2 Submission
**Steps:**
1. Create packaging script: `examples/gpt2/package.py`
2. Compress model weights or create distilled version
3. Verify zip size < 1 MB (critical constraint)
4. Run evaluation and compare score to baselines

**Expected Result:** Should outperform MLP baseline significantly if successful

**Critical Files:**
- `examples/gpt2/submission/model.py` (new)
- `examples/gpt2/package.py` (new)

---

## Phase 4: Scoring Metric Analysis & Validation

### 4.1 Unit Tests for Scoring Components

#### Test: Context Building
**File:** `tests/test_context_building.py` (new)

**Test Cases:**
- Empty text → no pairs
- Single byte → one pair with empty context
- Text with 512+ bytes → verify context window truncation
- UTF-8 multi-byte characters → verify byte-level splitting
- Multiple documents → verify no cross-document contexts

#### Test: Cross-Entropy Calculation
**File:** `tests/test_scoring.py` (new)

**Test Cases:**
- Perfect predictions (logits match targets) → loss ≈ 0
- Uniform logits → loss ≈ 8.0
- Verify LN2 conversion: nats → bits
- Batch size variations (1, 10, 1024) → same average loss

### 4.2 Scoring Metric Soundness Analysis

**Mathematical Validation:**
1. **Cross-entropy formula:**
   ```
   H(p, q) = -Σ p(x) log₂ q(x)
   ```
   Where p = true distribution (one-hot), q = predicted distribution (softmax of logits)

2. **Per-byte interpretation:**
   - Score represents average bits needed to encode next byte
   - Lower = better compression → better language model
   - Baseline (uniform) = 8.0 bits (log₂(256))

3. **Why this metric makes sense for byte-level models:**
   - Direct connection to information theory
   - Interpretable: bits per character/byte
   - Comparable across different model architectures
   - Robust: doesn't depend on specific tokenization scheme

**Potential Issues to Verify:**
- **Numerical stability:** PyTorch's `cross_entropy` uses log-sum-exp trick → stable ✓
- **Batch ordering:** Loss is averaged across all pairs → order-independent ✓
- **Context window edge effects:** Short contexts are fair for all models ✓

**Critical Files:**
- `src/gki_evaluator/evaluate.py:123-140` (evaluate function)

---

## Phase 5: Docker Security & Integration Testing

### 5.1 Security Constraint Verification

#### Test: Network Isolation
- Create submission that attempts HTTP request
- Verify it fails with network error

#### Test: Filesystem Isolation
- Attempt to write to `/app/` (read-only) → should fail
- Verify `/tmp/` writes succeed but are limited

#### Test: Resource Limits
- Allocate > 4GB memory → should be killed
- Fork many processes → should hit pid limit
- Infinite loop → should timeout

### 5.2 Submission Validation Tests

#### Test: Zip Bomb Protection
- Create zip with 1 KB compressed → 100 MB uncompressed → should reject
- Verify extraction stops before decompression bomb explodes

#### Test: Path Traversal Protection
- Create zip with `../../../etc/passwd` → should reject
- Verify only files within `/tmp/submission/` are extracted

#### Test: Submission Size Validation
- 0.5 MB zip → pass
- 1.0 MB zip → pass (at limit)
- 1.1 MB zip → reject

---

## Phase 6: Edge Case & Stress Testing

### 6.1 Edge Cases
- Return empty list → verify error
- Return wrong shape (128 logits instead of 256) → verify error
- Return NaN/Inf values → verify error handling
- Wrong batch size → verify error

### 6.2 Stress Testing
- Evaluate on 100k+ context-target pairs
- Run concurrent evaluations (multiple containers)
- Verify resource limits apply per-container

---

## Phase 7: Automated Test Suite

### 7.1 Test Runner Script
**File:** `tests/run_all_tests.sh`

```bash
#!/bin/bash
set -e

echo "=== Phase 1: Baseline Tests ==="
cd examples/dummy && python package.py
python -m gki_evaluator.evaluate --submission examples/dummy/output/submission.zip --test-data data/

cd examples/mlp && python train.py
python -m gki_evaluator.evaluate --submission examples/mlp/output/submission.zip --test-data data/

echo "=== Phase 2: Validation ==="
python tests/validate_scores.py

echo "=== Phase 3: Unit Tests ==="
pytest tests/

echo "✓ All tests passed!"
```

### 7.2 Score Validation
**File:** `tests/validate_scores.py`

Validates:
- Dummy baseline ≈ 8.0
- MLP beats baseline

---

## Phase 8: Documentation

### 8.1 Test Results Report
**File:** `docs/test_report.md`

Contains:
- Summary of all test results
- Scoring metric analysis
- Security verification results
- Performance benchmarks

### 8.2 Scoring Metric Deep Dive
**File:** `docs/scoring_analysis.md`

Contains:
- Mathematical derivation of cross-entropy loss
- Why bits per byte is the right metric
- Expected scores for different model types

---

## Deliverables

### New Files to Create:
1. **GPT-2 Integration:** `examples/gpt2/submission/model.py`, `examples/gpt2/package.py`
2. **Test Suite:** `tests/test_*.py`, `tests/run_all_tests.sh`, `tests/validate_scores.py`
3. **Documentation:** `docs/test_report.md`, `docs/scoring_analysis.md`

### Verification Checklist
- [ ] Docker image builds successfully
- [ ] Dummy baseline scores ~8.0 bits/byte
- [ ] MLP baseline beats dummy (score < 8.0)
- [ ] GPT-2 submission created (or documented as infeasible)
- [ ] Unit tests pass
- [ ] Security tests verify isolation
- [ ] Edge cases handled
- [ ] Documentation complete

---

## Expected Outcomes

### Scoring Metric Soundness:
- Mathematically sound (cross-entropy is standard)
- Interpretable (bits per byte)
- Fair (same metric for all models)
- Robust (handles edge cases)

### Security Validation:
- Network isolation verified
- Filesystem isolation verified
- Resource limits enforced
- Input validation works

### Baseline Verification:
- Dummy confirms scoring (8.0 bits)
- MLP confirms learning (< 8.0 bits)
- GPT-2 tests complex models
