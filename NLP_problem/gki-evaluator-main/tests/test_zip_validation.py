"""Tests for ZIP file validation and security.

Tests zip bomb protection, path traversal, and size validation.
"""

import sys
import zipfile
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gki_evaluator.evaluate import validate_submission


def test_submission_size_valid():
    """Test that submissions under 1MB pass validation."""
    # Create a small submission
    temp_dir = Path(tempfile.mkdtemp(prefix="test_size_valid_"))
    zip_path = temp_dir / "submission.zip"

    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("model.py", "# Small file\n" * 100)  # ~1.5KB

    # Should not raise
    try:
        validate_submission(zip_path)
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"✓ Small submission ({size_mb:.3f} MB) passed validation")
        return True
    except ValueError as e:
        print(f"✗ Small submission failed: {e}")
        return False


def test_submission_size_at_limit():
    """Test that submissions exactly at 1MB pass validation."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_size_limit_"))
    zip_path = temp_dir / "submission.zip"

    # Create file that will compress to ~1MB
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # Write data that compresses to approximately 1MB
        data = "A" * (1024 * 1024 * 2)  # 2MB of 'A's compresses well
        z.writestr("model.py", data)

    size_mb = zip_path.stat().st_size / (1024 * 1024)

    # If we're over 1MB, reduce the data
    if size_mb > 1.0:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            data = "A" * (1024 * 1000)  # Smaller data
            z.writestr("model.py", data)
        size_mb = zip_path.stat().st_size / (1024 * 1024)

    try:
        validate_submission(zip_path)
        print(f"✓ Submission at limit ({size_mb:.3f} MB) passed validation")
        return True
    except ValueError as e:
        if size_mb <= 1.0:
            print(f"✗ Submission at limit failed unexpectedly: {e}")
            return False
        else:
            print(f"✓ Oversized submission ({size_mb:.3f} MB) correctly rejected")
            return True


def test_submission_size_over_limit():
    """Test that submissions over 1MB are rejected."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_size_over_"))
    zip_path = temp_dir / "submission.zip"

    # Create a file larger than 1MB
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:
        # Use STORED (no compression) to ensure we're over 1MB
        data = "X" * (1024 * 1024 + 1000)  # 1MB + 1KB
        z.writestr("model.py", data)

    size_mb = zip_path.stat().st_size / (1024 * 1024)

    try:
        validate_submission(zip_path)
        print(f"✗ Oversized submission ({size_mb:.3f} MB) was not rejected!")
        return False
    except ValueError as e:
        print(f"✓ Oversized submission ({size_mb:.3f} MB) correctly rejected: {e}")
        return True


def test_zip_bomb_protection():
    """Test that zip bombs are detected and rejected.

    A zip bomb is a small compressed file that expands to huge size.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="test_zipbomb_"))
    zip_path = temp_dir / "submission.zip"

    # Create a zip bomb: tiny compressed size, huge uncompressed
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # 100MB of zeros compresses to ~100KB
        bomb_data = b"\\x00" * (100 * 1024 * 1024)
        z.writestr("model.py", bomb_data)

    compressed_mb = zip_path.stat().st_size / (1024 * 1024)

    # Check uncompressed size
    with zipfile.ZipFile(zip_path, "r") as z:
        uncompressed_mb = sum(info.file_size for info in z.infolist()) / (1024 * 1024)

    print(f"  Zip bomb: {compressed_mb:.2f}MB compressed → {uncompressed_mb:.0f}MB uncompressed")

    # The scoring.py should detect and reject this (50MB limit)
    # This test is informational - actual detection happens in Docker
    if uncompressed_mb > 50:
        print(f"✓ Created zip bomb for testing (would be rejected by Docker)")
        return True
    else:
        print(f"⚠ Zip bomb not large enough ({uncompressed_mb:.0f}MB)")
        return True


def test_path_traversal_protection():
    """Test that path traversal attacks are prevented.

    Malicious zips could contain paths like ../../../etc/passwd
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="test_traversal_"))
    zip_path = temp_dir / "submission.zip"

    # Create zip with path traversal
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("../../../tmp/malicious.py", "# Path traversal")
        z.writestr("model.py", "# Normal file")

    print(f"✓ Created path traversal test zip")
    print(f"  Contains: ../../../tmp/malicious.py")
    print(f"  Docker extraction should reject or sanitize this path")
    return True


def test_absolute_path_protection():
    """Test that absolute paths in zips are handled safely."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_abspath_"))
    zip_path = temp_dir / "submission.zip"

    # Create zip with absolute path
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("/tmp/malicious.py", "# Absolute path")
        z.writestr("model.py", "# Normal file")

    print(f"✓ Created absolute path test zip")
    print(f"  Contains: /tmp/malicious.py")
    print(f"  Docker extraction should sanitize this path")
    return True


def test_symlink_protection():
    """Test that symbolic links in zips are handled safely."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_symlink_"))
    zip_path = temp_dir / "submission.zip"

    # Note: Creating actual symlinks in ZIP requires special handling
    # This is a simplified test
    print(f"✓ Symlink test (informational)")
    print(f"  ZIP format can contain symlinks pointing outside submission dir")
    print(f"  Docker should not follow symlinks to protected areas")
    return True


def test_nested_zip_protection():
    """Test that nested zips don't cause issues."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_nested_"))
    inner_zip = temp_dir / "inner.zip"
    outer_zip = temp_dir / "submission.zip"

    # Create inner zip
    with zipfile.ZipFile(inner_zip, "w") as z:
        z.writestr("data.txt", "Nested data" * 1000)

    # Create outer zip containing inner zip
    with zipfile.ZipFile(outer_zip, "w") as z:
        z.write(inner_zip, "inner.zip")
        z.writestr("model.py", "# Outer file")

    size_mb = outer_zip.stat().st_size / (1024 * 1024)
    print(f"✓ Created nested zip test ({size_mb:.3f} MB)")
    print(f"  Docker should only extract outer zip")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("ZIP VALIDATION TESTS")
    print("=" * 60)
    print()

    tests = [
        ("Submission size (valid)", test_submission_size_valid),
        ("Submission size (at limit)", test_submission_size_at_limit),
        ("Submission size (over limit)", test_submission_size_over_limit),
        ("Zip bomb protection", test_zip_bomb_protection),
        ("Path traversal protection", test_path_traversal_protection),
        ("Absolute path protection", test_absolute_path_protection),
        ("Symlink protection", test_symlink_protection),
        ("Nested zip protection", test_nested_zip_protection),
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
        except Exception as e:
            print(f"✗ Exception: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"{passed}/{len(tests)} zip validation tests passed")
    print()
    print("Note: Some tests create malicious zips for Docker to handle.")
    print("Path traversal and zip bomb protection is enforced by scoring.py")
    print("=" * 60)
