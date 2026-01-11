"""Security and isolation tests for Docker container.

These tests verify that the Docker sandbox properly isolates submissions and
enforces resource limits to prevent malicious behavior.
"""

import sys
import zipfile
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_malicious_submission(code: str, name: str = "malicious") -> Path:
    """Create a malicious submission for testing.

    Args:
        code: Python code for the Model class
        name: Name for the submission (used in temp directory)

    Returns:
        Path to the created submission.zip
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=f"test_{name}_"))
    submission_dir = temp_dir / "submission"
    submission_dir.mkdir()

    model_file = submission_dir / "model.py"
    model_file.write_text(code)

    zip_path = temp_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(model_file, "model.py")

    return zip_path


def test_network_isolation():
    """Test that container cannot access network.

    This is critical for preventing data exfiltration.
    """
    code = '''
from pathlib import Path
import sys

class Model:
    def __init__(self, submission_dir: Path):
        # Try to make network request
        try:
            import urllib.request
            response = urllib.request.urlopen("http://example.com", timeout=5)
            # If this succeeds, network is NOT isolated!
            sys.stderr.write("ERROR: Network access succeeded!\\n")
            raise RuntimeError("Network should be blocked")
        except urllib.error.URLError as e:
            # Expected: network is blocked
            sys.stderr.write(f"Good: Network blocked ({e})\\n")
        except Exception as e:
            sys.stderr.write(f"Network test error: {e}\\n")

    def predict(self, contexts):
        return [[0.0] * 256 for _ in contexts]
'''

    submission = create_malicious_submission(code, "network_test")

    # This would need to actually run the evaluator
    # For now, we verify the code structure is correct
    print(f"✓ Created network test submission at {submission}")
    print("  Manual test: Run evaluator and verify URLError in logs")
    return True


def test_filesystem_write_protection():
    """Test that container cannot write to protected directories.

    The /app directory should be read-only to prevent code injection.
    """
    code = '''
from pathlib import Path
import sys

class Model:
    def __init__(self, submission_dir: Path):
        # Try to write to /app (should fail)
        try:
            with open("/app/injected_code.py", "w") as f:
                f.write("malicious code")
            sys.stderr.write("ERROR: Write to /app succeeded!\\n")
            raise RuntimeError("/app should be read-only")
        except (PermissionError, OSError) as e:
            sys.stderr.write(f"Good: /app is read-only ({e})\\n")

        # Try to write to /tmp (should succeed - it's the working space)
        try:
            with open("/tmp/test_write.txt", "w") as f:
                f.write("test")
            sys.stderr.write("Good: /tmp is writable\\n")
        except Exception as e:
            sys.stderr.write(f"Warning: /tmp write failed ({e})\\n")

    def predict(self, contexts):
        return [[0.0] * 256 for _ in contexts]
'''

    submission = create_malicious_submission(code, "filesystem_test")
    print(f"✓ Created filesystem test submission at {submission}")
    print("  Manual test: Run evaluator and verify PermissionError in logs")
    return True


def test_resource_limits():
    """Test that container enforces memory and CPU limits.

    This prevents denial-of-service attacks.
    """
    code = '''
from pathlib import Path
import sys

class Model:
    def __init__(self, submission_dir: Path):
        sys.stderr.write("Testing resource limits...\\n")

        # Try to allocate large amounts of memory
        # Container has 4GB limit, try to allocate 5GB
        try:
            sys.stderr.write("Attempting to allocate 5GB...\\n")
            data = []
            for i in range(50):  # 50 * 100MB = 5GB
                chunk = [0] * (100 * 1024 * 1024 // 8)  # 100MB
                data.append(chunk)
                sys.stderr.write(f"Allocated {(i+1) * 100}MB\\n")
            sys.stderr.write("ERROR: Allocated 5GB (should have been killed)\\n")
        except MemoryError:
            sys.stderr.write("Good: MemoryError raised\\n")
        except Exception as e:
            sys.stderr.write(f"Memory limit test: {e}\\n")

    def predict(self, contexts):
        return [[0.0] * 256 for _ in contexts]
'''

    submission = create_malicious_submission(code, "resource_test")
    print(f"✓ Created resource test submission at {submission}")
    print("  Manual test: Run evaluator and verify container is killed or MemoryError")
    return True


def test_process_limits():
    """Test that container limits number of processes.

    Prevents fork bombs and similar attacks.
    """
    code = '''
from pathlib import Path
import sys
import os

class Model:
    def __init__(self, submission_dir: Path):
        sys.stderr.write("Testing process limits...\\n")

        # Try to fork many processes (fork bomb)
        try:
            sys.stderr.write("Attempting to fork 200 processes...\\n")
            pids = []
            for i in range(200):
                pid = os.fork()
                if pid > 0:
                    pids.append(pid)
                else:
                    # Child process: exit immediately
                    sys.exit(0)
            sys.stderr.write(f"ERROR: Forked {len(pids)} processes (should have been limited)\\n")
            # Clean up
            for pid in pids:
                try:
                    os.waitpid(pid, 0)
                except:
                    pass
        except (OSError, PermissionError) as e:
            sys.stderr.write(f"Good: Process limit enforced ({e})\\n")
        except Exception as e:
            sys.stderr.write(f"Process limit test: {e}\\n")

    def predict(self, contexts):
        return [[0.0] * 256 for _ in contexts]
'''

    submission = create_malicious_submission(code, "process_test")
    print(f"✓ Created process limit test submission at {submission}")
    print("  Manual test: Run evaluator and verify OSError or process limit")
    return True


def test_capability_restrictions():
    """Test that container has dropped all capabilities.

    Prevents privilege escalation.
    """
    code = '''
from pathlib import Path
import sys
import os

class Model:
    def __init__(self, submission_dir: Path):
        sys.stderr.write("Testing capability restrictions...\\n")

        # Check if running as root (shouldn't be)
        if os.getuid() == 0:
            sys.stderr.write("WARNING: Running as root!\\n")
        else:
            sys.stderr.write(f"Good: Running as uid {os.getuid()}\\n")

        # Try to change user (should fail without CAP_SETUID)
        try:
            os.setuid(0)
            sys.stderr.write("ERROR: setuid succeeded (capabilities not dropped)\\n")
        except PermissionError:
            sys.stderr.write("Good: Cannot setuid (capabilities dropped)\\n")
        except Exception as e:
            sys.stderr.write(f"Capability test: {e}\\n")

    def predict(self, contexts):
        return [[0.0] * 256 for _ in contexts]
'''

    submission = create_malicious_submission(code, "capability_test")
    print(f"✓ Created capability test submission at {submission}")
    print("  Manual test: Run evaluator and verify non-root UID and PermissionError")
    return True


def test_timeout_enforcement():
    """Test that infinite loops are terminated.

    The evaluator should have a timeout for model operations.
    """
    code = '''
from pathlib import Path
import sys
import time

class Model:
    def __init__(self, submission_dir: Path):
        sys.stderr.write("Model initialized\\n")

    def predict(self, contexts):
        sys.stderr.write("Starting infinite loop...\\n")
        # Infinite loop - should timeout
        while True:
            time.sleep(0.1)
        return [[0.0] * 256 for _ in contexts]
'''

    submission = create_malicious_submission(code, "timeout_test")
    print(f"✓ Created timeout test submission at {submission}")
    print("  Manual test: Run evaluator and verify timeout after reasonable duration")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("SECURITY TESTS")
    print("=" * 60)
    print()
    print("NOTE: These tests create malicious submissions for manual testing.")
    print("Each test must be run manually with the evaluator to verify security.")
    print()

    tests = [
        ("Network isolation", test_network_isolation),
        ("Filesystem write protection", test_filesystem_write_protection),
        ("Resource limits (memory)", test_resource_limits),
        ("Process limits (fork bomb)", test_process_limits),
        ("Capability restrictions", test_capability_restrictions),
        ("Timeout enforcement", test_timeout_enforcement),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\\n{name}:")
        print("-" * 60)
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"{passed}/{len(tests)} security test submissions created")
    print()
    print("To run security tests:")
    print("1. Find created submissions in /tmp/test_*")
    print("2. Run: .venv/bin/python -m gki_evaluator.evaluate --submission <path> --test-data data/")
    print("3. Check stderr logs for security validation messages")
    print("=" * 60)
