"""Integration and protocol error handling tests.

Tests the JSON protocol between evaluator and Docker container,
including malformed requests, wrong shapes, and error cases.
"""

import sys
import zipfile
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_test_submission(code: str, name: str) -> Path:
    """Create a test submission."""
    temp_dir = Path(tempfile.mkdtemp(prefix=f"test_{name}_"))
    submission_dir = temp_dir / "submission"
    submission_dir.mkdir()

    model_file = submission_dir / "model.py"
    model_file.write_text(code)

    zip_path = temp_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(model_file, "model.py")

    return zip_path


def test_wrong_output_shape():
    """Test model that returns wrong number of logits (128 instead of 256)."""
    code = '''
from pathlib import Path

class Model:
    def __init__(self, submission_dir: Path):
        pass

    def predict(self, contexts):
        # Return wrong shape: 128 logits instead of 256
        return [[0.0] * 128 for _ in contexts]
'''

    submission = create_test_submission(code, "wrong_shape")
    print(f"✓ Created wrong shape test: {submission}")
    print("  Expected: Error about shape mismatch (128 vs 256)")
    return True


def test_empty_output():
    """Test model that returns empty list."""
    code = '''
from pathlib import Path

class Model:
    def __init__(self, submission_dir: Path):
        pass

    def predict(self, contexts):
        # Return empty list
        return []
'''

    submission = create_test_submission(code, "empty_output")
    print(f"✓ Created empty output test: {submission}")
    print("  Expected: Error about batch size mismatch")
    return True


def test_nan_logits():
    """Test model that returns NaN values."""
    code = '''
from pathlib import Path

class Model:
    def __init__(self, submission_dir: Path):
        pass

    def predict(self, contexts):
        # Return NaN logits
        return [[float('nan')] * 256 for _ in contexts]
'''

    submission = create_test_submission(code, "nan_logits")
    print(f"✓ Created NaN logits test: {submission}")
    print("  Expected: NaN in loss or error")
    return True


def test_inf_logits():
    """Test model that returns Inf values."""
    code = '''
from pathlib import Path

class Model:
    def __init__(self, submission_dir: Path):
        pass

    def predict(self, contexts):
        # Return Inf logits
        return [[float('inf')] * 256 for _ in contexts]
'''

    submission = create_test_submission(code, "inf_logits")
    print(f"✓ Created Inf logits test: {submission}")
    print("  Expected: Inf in loss or error")
    return True


def test_wrong_batch_size():
    """Test model that returns wrong number of predictions."""
    code = '''
from pathlib import Path

class Model:
    def __init__(self, submission_dir: Path):
        pass

    def predict(self, contexts):
        # Always return 10 predictions, regardless of batch size
        return [[0.0] * 256 for _ in range(10)]
'''

    submission = create_test_submission(code, "wrong_batch")
    print(f"✓ Created wrong batch size test: {submission}")
    print("  Expected: Error about batch size mismatch")
    return True


def test_crash_in_init():
    """Test model that crashes during initialization."""
    code = '''
from pathlib import Path

class Model:
    def __init__(self, submission_dir: Path):
        raise RuntimeError("Intentional crash in __init__")

    def predict(self, contexts):
        return [[0.0] * 256 for _ in contexts]
'''

    submission = create_test_submission(code, "crash_init")
    print(f"✓ Created crash in init test: {submission}")
    print("  Expected: Container fails to start or error message")
    return True


def test_crash_in_predict():
    """Test model that crashes during prediction."""
    code = '''
from pathlib import Path

class Model:
    def __init__(self, submission_dir: Path):
        self.call_count = 0

    def predict(self, contexts):
        self.call_count += 1
        if self.call_count == 2:
            # Crash on second call
            raise RuntimeError("Intentional crash in predict")
        return [[0.0] * 256 for _ in contexts]
'''

    submission = create_test_submission(code, "crash_predict")
    print(f"✓ Created crash in predict test: {submission}")
    print("  Expected: Container dies mid-evaluation")
    return True


def test_missing_predict_method():
    """Test model without predict method."""
    code = '''
from pathlib import Path

class Model:
    def __init__(self, submission_dir: Path):
        pass

    # Missing predict method!
'''

    submission = create_test_submission(code, "no_predict")
    print(f"✓ Created missing predict method test: {submission}")
    print("  Expected: AttributeError when calling predict")
    return True


def test_wrong_predict_signature():
    """Test model with wrong predict signature."""
    code = '''
from pathlib import Path

class Model:
    def __init__(self, submission_dir: Path):
        pass

    def predict(self):  # Missing contexts parameter!
        return [[0.0] * 256]
'''

    submission = create_test_submission(code, "wrong_signature")
    print(f"✓ Created wrong signature test: {submission}")
    print("  Expected: TypeError about missing argument")
    return True


def test_return_dict_instead_of_list():
    """Test model that returns wrong type."""
    code = '''
from pathlib import Path

class Model:
    def __init__(self, submission_dir: Path):
        pass

    def predict(self, contexts):
        # Return dict instead of list
        return {"logits": [[0.0] * 256 for _ in contexts]}
'''

    submission = create_test_submission(code, "wrong_type")
    print(f"✓ Created wrong return type test: {submission}")
    print("  Expected: TypeError or error converting to tensor")
    return True


def test_non_numeric_logits():
    """Test model that returns strings instead of numbers."""
    code = '''
from pathlib import Path

class Model:
    def __init__(self, submission_dir: Path):
        pass

    def predict(self, contexts):
        # Return strings instead of floats
        return [["0.0"] * 256 for _ in contexts]
'''

    submission = create_test_submission(code, "non_numeric")
    print(f"✓ Created non-numeric logits test: {submission}")
    print("  Expected: TypeError or ValueError")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATION & PROTOCOL ERROR TESTS")
    print("=" * 60)
    print()
    print("NOTE: These tests create malformed submissions.")
    print("Run them with the evaluator to verify error handling.")
    print()

    tests = [
        ("Wrong output shape (128 vs 256)", test_wrong_output_shape),
        ("Empty output", test_empty_output),
        ("NaN logits", test_nan_logits),
        ("Inf logits", test_inf_logits),
        ("Wrong batch size", test_wrong_batch_size),
        ("Crash in __init__", test_crash_in_init),
        ("Crash in predict", test_crash_in_predict),
        ("Missing predict method", test_missing_predict_method),
        ("Wrong predict signature", test_wrong_predict_signature),
        ("Return dict instead of list", test_return_dict_instead_of_list),
        ("Non-numeric logits", test_non_numeric_logits),
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
    print(f"{passed}/{len(tests)} test submissions created")
    print()
    print("To test error handling:")
    print("1. Find submissions in /tmp/test_*/submission.zip")
    print("2. Run: .venv/bin/python -m gki_evaluator.evaluate --submission <path> --test-data data/")
    print("3. Verify appropriate error messages or graceful handling")
    print("=" * 60)
