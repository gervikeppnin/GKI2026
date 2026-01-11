"""Unit tests for scoring calculation."""

import sys
from pathlib import Path
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import after path setup
import torch
import torch.nn.functional as F

# Constants from evaluate.py
LN2 = math.log(2)


def test_uniform_distribution():
    """Uniform logits should give log₂(256) = 8.0 bits."""
    # Create uniform logits (all zeros)
    logits = torch.zeros(100, 256)
    targets = torch.randint(0, 256, (100,))

    # Calculate cross-entropy
    loss = F.cross_entropy(logits, targets, reduction="mean").item() / LN2

    # Should be approximately 8.0 bits
    assert abs(loss - 8.0) < 0.1, \
        f"Uniform distribution should give ~8.0 bits, got {loss:.4f}"


def test_perfect_predictions():
    """Perfect predictions should give loss ≈ 0."""
    # Create logits where correct class has high value
    batch_size = 100
    logits = torch.full((batch_size, 256), -10.0)  # Low values for all classes
    targets = torch.randint(0, 256, (batch_size,))

    # Set correct class to high value
    for i, target in enumerate(targets):
        logits[i, target] = 10.0

    loss = F.cross_entropy(logits, targets, reduction="mean").item() / LN2

    # Should be very close to 0 (perfect predictions)
    assert loss < 0.001, \
        f"Perfect predictions should give ~0 bits, got {loss:.4f}"


def test_worst_case():
    """Always predicting wrong should give high loss."""
    # Create logits where wrong class has high value
    batch_size = 100
    logits = torch.full((batch_size, 256), -10.0)
    targets = torch.randint(0, 256, (batch_size,))

    # Set wrong class to high value
    for i, target in enumerate(targets):
        wrong_class = (target + 1) % 256
        logits[i, wrong_class] = 10.0

    loss = F.cross_entropy(logits, targets, reduction="mean").item() / LN2

    # Should be very high (close to maximum entropy plus some)
    assert loss > 10.0, \
        f"Wrong predictions should give high loss, got {loss:.4f}"


def test_ln2_conversion():
    """Verify nats to bits conversion is correct."""
    # log_e(256) = 5.545177...
    # log_2(256) = 8.0
    # log_e(256) / ln(2) should equal log_2(256)

    log_e_256 = math.log(256)
    log_2_256 = math.log2(256)

    assert abs(log_e_256 / LN2 - log_2_256) < 1e-10, \
        "LN2 conversion is incorrect"

    assert abs(log_2_256 - 8.0) < 1e-10, \
        "log₂(256) should be exactly 8.0"


def test_batch_invariance():
    """Loss should be same regardless of batch size."""
    # Create same logits and targets in different batch sizes
    logits_small = torch.randn(10, 256)
    targets_small = torch.randint(0, 256, (10,))

    # Repeat to make larger batch
    logits_large = logits_small.repeat(10, 1)  # 100 samples
    targets_large = targets_small.repeat(10)

    loss_small = F.cross_entropy(logits_small, targets_small, reduction="mean").item() / LN2
    loss_large = F.cross_entropy(logits_large, targets_large, reduction="mean").item() / LN2

    # Should be identical (mean reduction)
    assert abs(loss_small - loss_large) < 1e-6, \
        f"Batch size should not affect loss: {loss_small:.6f} vs {loss_large:.6f}"


def test_numerical_stability():
    """Large logits should not cause overflow/underflow."""
    batch_size = 100

    # Test with very large logits
    logits_large = torch.randn(batch_size, 256) * 100  # Large scale
    targets = torch.randint(0, 256, (batch_size,))
    loss_large = F.cross_entropy(logits_large, targets, reduction="mean").item() / LN2

    assert not math.isnan(loss_large), "Large logits caused NaN"
    assert not math.isinf(loss_large), "Large logits caused Inf"

    # Test with very small logits
    logits_small = torch.randn(batch_size, 256) * 0.01  # Small scale
    loss_small = F.cross_entropy(logits_small, targets, reduction="mean").item() / LN2

    assert not math.isnan(loss_small), "Small logits caused NaN"
    assert not math.isinf(loss_small), "Small logits caused Inf"


def test_reduction_modes():
    """Test that sum reduction works correctly."""
    logits = torch.zeros(100, 256)
    targets = torch.randint(0, 256, (100,))

    # Mean reduction
    loss_mean = F.cross_entropy(logits, targets, reduction="mean").item() / LN2

    # Sum reduction (should be mean * batch_size)
    loss_sum = F.cross_entropy(logits, targets, reduction="sum").item() / LN2

    expected_sum = loss_mean * 100

    assert abs(loss_sum - expected_sum) < 0.1, \
        f"Sum should be {expected_sum:.2f}, got {loss_sum:.2f}"


def test_softmax_equivalence():
    """Verify cross-entropy matches manual softmax calculation."""
    logits = torch.randn(10, 256)
    targets = torch.randint(0, 256, (10,))

    # PyTorch cross-entropy
    loss_pytorch = F.cross_entropy(logits, targets, reduction="mean").item()

    # Manual calculation
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs)

    # Gather log probabilities for target classes
    target_log_probs = log_probs[range(10), targets]
    loss_manual = -target_log_probs.mean().item()

    assert abs(loss_pytorch - loss_manual) < 1e-6, \
        f"PyTorch and manual cross-entropy differ: {loss_pytorch:.6f} vs {loss_manual:.6f}"


def test_information_theoretic_bounds():
    """Verify loss is within information-theoretic bounds."""
    batch_size = 100
    logits = torch.randn(batch_size, 256)
    targets = torch.randint(0, 256, (batch_size,))

    loss = F.cross_entropy(logits, targets, reduction="mean").item() / LN2

    # Cross-entropy should be >= 0 (non-negative)
    assert loss >= 0, f"Cross-entropy should be non-negative, got {loss:.4f}"

    # For 256 classes, uniform distribution gives 8 bits
    # Random predictions should give loss around 8 bits
    # (might be slightly different due to random logits)
    assert loss < 20.0, \
        f"Loss seems unreasonably high: {loss:.4f} bits (expected < 20)"


if __name__ == "__main__":
    print("Running scoring tests...")

    tests = [
        ("Uniform distribution", test_uniform_distribution),
        ("Perfect predictions", test_perfect_predictions),
        ("Worst case", test_worst_case),
        ("LN2 conversion", test_ln2_conversion),
        ("Batch invariance", test_batch_invariance),
        ("Numerical stability", test_numerical_stability),
        ("Reduction modes", test_reduction_modes),
        ("Softmax equivalence", test_softmax_equivalence),
        ("Information theoretic bounds", test_information_theoretic_bounds),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name}: Unexpected error: {e}")
            failed += 1

    print(f"\n{passed}/{len(tests)} tests passed")
    if failed > 0:
        sys.exit(1)
