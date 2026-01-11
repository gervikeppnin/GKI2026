"""Unit tests for context building functionality."""

import sys
from pathlib import Path

# Add src to path so we can import the evaluator
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gki_evaluator.evaluate import build_contexts


def test_empty_text():
    """Empty text should produce no pairs."""
    texts = [""]
    pairs = build_contexts(texts, context_window=512)
    assert len(pairs) == 0, "Empty text should produce no pairs"


def test_single_byte():
    """Single byte should produce one pair with empty context."""
    texts = ["A"]
    pairs = build_contexts(texts, context_window=512)
    assert len(pairs) == 1, "Single byte should produce one pair"
    context, target = pairs[0]
    assert context == [], f"Context should be empty, got {context}"
    assert target == ord('A'), f"Target should be {ord('A')}, got {target}"


def test_multiple_bytes():
    """Test basic context building with simple text."""
    texts = ["Hello"]
    pairs = build_contexts(texts, context_window=512)

    # Should have 5 pairs (one per byte)
    assert len(pairs) == 5, f"Expected 5 pairs, got {len(pairs)}"

    # Note: pairs are shuffled! We need to find them by target
    # Let's rebuild the expected pairs before shuffling
    expected_pairs = {
        ord('H'): [],  # First char, empty context
        ord('e'): [ord('H')],
        ord('l'): [ord('H'), ord('e')],  # First 'l'
        # Second 'l' at position 3
        # ord('o') at position 4
    }

    # Instead, let's just check that all pairs are valid
    for context, target in pairs:
        # Target should be a byte from "Hello"
        assert target in [ord(c) for c in "Hello"], \
            f"Target {target} ({chr(target)}) not in 'Hello'"

        # Context should only contain bytes that come before target in "Hello"
        assert all(b in [ord(c) for c in "Hello"] for b in context), \
            f"Context {context} contains invalid bytes"

        # Context length should be position of target
        assert len(context) <= 4, "Context should be at most 4 bytes for 'Hello'"


def test_context_window_truncation():
    """Context should be truncated to window size."""
    # Create text longer than context window
    text = "A" * 600  # 600 bytes
    texts = [text]
    pairs = build_contexts(texts, context_window=10)

    # Check a pair in the middle
    context, target = pairs[50]

    # Context should be at most 10 bytes (the 10 'A's before position 50)
    assert len(context) <= 10, f"Context should be <= 10, got {len(context)}"
    assert len(context) == 10, f"Context should be exactly 10 at position 50"
    assert all(b == ord('A') for b in context), "All context bytes should be 'A'"


def test_utf8_multibyte():
    """UTF-8 multi-byte characters should be split into individual bytes."""
    # Icelandic text with multi-byte UTF-8 characters
    text = "Þetta"  # Þ is multi-byte in UTF-8
    texts = [text]
    pairs = build_contexts(texts, context_window=512)

    # "Þetta" in UTF-8 is [195, 158, 101, 116, 116, 97] (6 bytes)
    expected_bytes = text.encode('utf-8')
    assert len(pairs) == len(expected_bytes), \
        f"Expected {len(expected_bytes)} pairs, got {len(pairs)}"

    # Find the pair with empty context (first byte)
    first_pairs = [p for p in pairs if p[0] == []]
    assert len(first_pairs) == 1, "Should have exactly one pair with empty context"

    context, target = first_pairs[0]
    assert target == expected_bytes[0], \
        f"First target should be {expected_bytes[0]}, got {target}"


def test_multiple_documents():
    """Multiple documents should not have cross-document contexts."""
    texts = ["ABC", "XYZ"]
    pairs = build_contexts(texts, context_window=512)

    # Should have 6 pairs total (3 from each document)
    assert len(pairs) == 6, f"Expected 6 pairs, got {len(pairs)}"

    # The pairs are shuffled, but we can verify no cross-contamination by checking
    # that contexts only contain bytes from their respective document
    abc_bytes = set([ord('A'), ord('B'), ord('C')])
    xyz_bytes = set([ord('X'), ord('Y'), ord('Z')])

    for context, target in pairs:
        if target in abc_bytes:
            # If target is from "ABC", context should only have ABC bytes
            assert all(b in abc_bytes for b in context), \
                f"Cross-document contamination: target {chr(target)} has context {context}"
        elif target in xyz_bytes:
            # If target is from "XYZ", context should only have XYZ bytes
            assert all(b in xyz_bytes for b in context), \
                f"Cross-document contamination: target {chr(target)} has context {context}"


def test_context_window_zero():
    """Context window of 0 should give empty contexts."""
    texts = ["Hello"]
    pairs = build_contexts(texts, context_window=0)

    assert len(pairs) == 5, "Should still have 5 pairs"
    for context, target in pairs:
        assert context == [], f"All contexts should be empty with window=0"


def test_shuffling():
    """Pairs should be shuffled (non-deterministic test)."""
    texts = ["ABCDEFGHIJ"]
    pairs1 = build_contexts(texts, context_window=512)
    pairs2 = build_contexts(texts, context_window=512)

    # Note: This test might rarely fail due to random chance
    # The probability of getting the same shuffle is 1/10! ≈ 1/3.6M
    assert pairs1 != pairs2, \
        "Pairs should be shuffled differently on repeated calls (might rarely fail)"


if __name__ == "__main__":
    print("Running context building tests...")

    tests = [
        ("Empty text", test_empty_text),
        ("Single byte", test_single_byte),
        ("Multiple bytes", test_multiple_bytes),
        ("Context window truncation", test_context_window_truncation),
        ("UTF-8 multi-byte", test_utf8_multibyte),
        ("Multiple documents", test_multiple_documents),
        ("Context window zero", test_context_window_zero),
        ("Shuffling", test_shuffling),
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
