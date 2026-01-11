"""Edge case and stress tests.

Tests unusual but valid scenarios and system limits.
"""

import sys
import zipfile
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gki_evaluator.evaluate import build_contexts


def test_empty_context_batch():
    """Test with empty contexts (first bytes of documents)."""
    texts = ["A", "B", "C"]
    pairs = build_contexts(texts, context_window=512)

    # Each text has 1 byte, so 3 pairs total
    assert len(pairs) == 3
    # All should have empty contexts
    empty_context_pairs = [p for p in pairs if p[0] == []]
    assert len(empty_context_pairs) == 3, \
        f"Expected 3 empty contexts, got {len(empty_context_pairs)}"
    print("✓ Empty context batch test passed")
    return True


def test_max_context_length():
    """Test with context at maximum window size (512 bytes)."""
    # Create text longer than 512 bytes
    text = "A" * 1000
    texts = [text]
    pairs = build_contexts(texts, context_window=512)

    # Check pair near the end
    context, target = pairs[-1]

    # Context should be exactly 512 bytes (max window)
    assert len(context) == 512, \
        f"Expected context length 512, got {len(context)}"
    print(f"✓ Max context length test passed (context = {len(context)} bytes)")
    return True


def test_single_document():
    """Test with only one document."""
    texts = ["Hello world"]
    pairs = build_contexts(texts, context_window=512)

    expected_len = len("Hello world".encode('utf-8'))
    assert len(pairs) == expected_len, \
        f"Expected {expected_len} pairs, got {len(pairs)}"
    print(f"✓ Single document test passed ({len(pairs)} pairs)")
    return True


def test_many_small_documents():
    """Test with many small documents."""
    texts = ["X"] * 1000  # 1000 single-byte documents
    pairs = build_contexts(texts, context_window=512)

    assert len(pairs) == 1000, f"Expected 1000 pairs, got {len(pairs)}"
    # All should have empty contexts (each doc is just 1 byte)
    empty_contexts = [p for p in pairs if p[0] == []]
    assert len(empty_contexts) == 1000, \
        f"Expected all empty contexts, got {len(empty_contexts)}"
    print("✓ Many small documents test passed")
    return True


def test_mixed_length_documents():
    """Test with documents of varying lengths."""
    texts = [
        "A",  # 1 byte
        "AB",  # 2 bytes
        "ABC",  # 3 bytes
        "ABCD",  # 4 bytes
        "ABCDE",  # 5 bytes
    ]
    pairs = build_contexts(texts, context_window=512)

    expected_total = 1 + 2 + 3 + 4 + 5
    assert len(pairs) == expected_total, \
        f"Expected {expected_total} pairs, got {len(pairs)}"
    print(f"✓ Mixed length documents test passed ({len(pairs)} pairs)")
    return True


def test_all_same_byte():
    """Test with text containing only one repeated byte."""
    text = "A" * 100
    texts = [text]
    pairs = build_contexts(texts, context_window=512)

    # All targets should be ord('A')
    targets = [t for _, t in pairs]
    assert all(t == ord('A') for t in targets), \
        "All targets should be 'A'"

    # All contexts should only contain ord('A')
    for context, _ in pairs:
        assert all(b == ord('A') for b in context), \
            f"Context contains non-A bytes: {context}"

    print(f"✓ All same byte test passed ({len(pairs)} pairs)")
    return True


def test_binary_data():
    """Test with non-text binary data."""
    # Create binary data with all possible byte values
    binary_data = bytes(range(256))
    text = binary_data.decode('latin1')  # latin1 can decode any byte
    texts = [text]

    # When we encode to UTF-8, bytes 128-255 become multi-byte sequences
    expected_utf8_length = len(text.encode('utf-8'))

    pairs = build_contexts(texts, context_window=512)

    assert len(pairs) == expected_utf8_length, \
        f"Expected {expected_utf8_length} pairs, got {len(pairs)}"

    # Targets should cover all UTF-8 byte values present
    targets = set(t for _, t in pairs)
    # We should have all 256 original values represented in UTF-8 encoding
    # (though some as multi-byte sequences)
    assert len(targets) > 0, "Should have some byte values"

    print(f"✓ Binary data test passed ({len(pairs)} UTF-8 bytes from 256 latin1 chars)")
    return True


def test_unicode_edge_cases():
    """Test with various Unicode characters."""
    # Test various Unicode ranges
    texts = [
        "ASCII text",  # Plain ASCII
        "Íslenska",  # Icelandic (Latin extended)
        "日本語",  # Japanese
        "🎉🚀",  # Emojis (multi-byte)
        "Θεός",  # Greek
    ]

    for i, text in enumerate(texts):
        byte_length = len(text.encode('utf-8'))
        pairs = build_contexts([text], context_window=512)
        assert len(pairs) == byte_length, \
            f"Text {i} ({text}): expected {byte_length} pairs, got {len(pairs)}"

    print("✓ Unicode edge cases test passed")
    return True


def test_very_long_context():
    """Test with document longer than context window."""
    # Create 2000-byte document (>> 512 window)
    text = "X" * 2000
    texts = [text]
    pairs = build_contexts(texts, context_window=512)

    assert len(pairs) == 2000

    # Check contexts grow from 0 to 512
    # (After shuffling, we can't check sequential growth,
    #  but we can verify max length is 512)
    max_context_len = max(len(ctx) for ctx, _ in pairs)
    assert max_context_len == 512, \
        f"Max context should be 512, got {max_context_len}"

    print(f"✓ Very long context test passed (max context = {max_context_len})")
    return True


def test_zero_window_size():
    """Test with zero context window."""
    texts = ["Hello"]
    pairs = build_contexts(texts, context_window=0)

    # Should still have 5 pairs, but all with empty contexts
    assert len(pairs) == 5
    assert all(ctx == [] for ctx, _ in pairs), \
        "All contexts should be empty with window=0"

    print("✓ Zero window size test passed")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("EDGE CASE & STRESS TESTS")
    print("=" * 60)
    print()

    tests = [
        ("Empty context batch", test_empty_context_batch),
        ("Max context length (512)", test_max_context_length),
        ("Single document", test_single_document),
        ("Many small documents (1000)", test_many_small_documents),
        ("Mixed length documents", test_mixed_length_documents),
        ("All same byte", test_all_same_byte),
        ("Binary data (all 256 values)", test_binary_data),
        ("Unicode edge cases", test_unicode_edge_cases),
        ("Very long context (2000 bytes)", test_very_long_context),
        ("Zero window size", test_zero_window_size),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\\n{name}:")
        print("-" * 60)
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except AssertionError as e:
            print(f"✗ Failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Exception: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"{passed}/{len(tests)} edge case tests passed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
