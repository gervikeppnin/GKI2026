"""Validate baseline scores are within expected ranges."""

import json
import sys
from pathlib import Path


def validate_score(score_file: Path, min_score: float, max_score: float, name: str) -> bool:
    """Validate a score is within expected range.

    Args:
        score_file: Path to score.json
        min_score: Minimum expected score
        max_score: Maximum expected score
        name: Name of the baseline for reporting

    Returns:
        True if valid, False otherwise
    """
    if not score_file.exists():
        print(f"✗ {name}: Score file not found at {score_file}")
        return False

    try:
        with open(score_file) as f:
            data = json.load(f)

        score = data.get("score")
        if score is None:
            print(f"✗ {name}: No 'score' field in {score_file}")
            return False

        if min_score <= score <= max_score:
            print(f"✓ {name}: {score:.4f} bits/byte (expected {min_score:.1f}-{max_score:.1f})")
            return True
        else:
            print(f"✗ {name}: {score:.4f} bits/byte (expected {min_score:.1f}-{max_score:.1f})")
            return False

    except json.JSONDecodeError as e:
        print(f"✗ {name}: Invalid JSON in {score_file}: {e}")
        return False
    except Exception as e:
        print(f"✗ {name}: Error reading {score_file}: {e}")
        return False


def main():
    """Run score validation."""
    print("=" * 60)
    print("BASELINE SCORE VALIDATION")
    print("=" * 60)
    print()

    project_root = Path(__file__).parent.parent
    dummy_score = project_root / "output" / "dummy_score.json"
    mlp_score = project_root / "output" / "mlp_score.json"

    # Validate dummy baseline (should be ~8.0)
    dummy_valid = validate_score(
        dummy_score,
        min_score=7.8,
        max_score=8.2,
        name="Dummy Baseline"
    )

    # Validate MLP baseline (should be < 8.0, typically 2-7 range)
    mlp_valid = validate_score(
        mlp_score,
        min_score=0.0,
        max_score=8.0,
        name="MLP Baseline"
    )

    # Cross-validate: MLP should beat dummy
    if dummy_valid and mlp_valid:
        dummy_data = json.load(open(dummy_score))
        mlp_data = json.load(open(mlp_score))

        dummy_val = dummy_data["score"]
        mlp_val = mlp_data["score"]

        if mlp_val < dummy_val:
            print(f"✓ MLP ({mlp_val:.4f}) beats Dummy ({dummy_val:.4f})")
            cross_valid = True
        else:
            print(f"✗ MLP ({mlp_val:.4f}) should beat Dummy ({dummy_val:.4f})")
            cross_valid = False
    else:
        cross_valid = False

    print()
    print("=" * 60)

    if dummy_valid and mlp_valid and cross_valid:
        print("ALL VALIDATIONS PASSED ✓")
        print("=" * 60)
        return 0
    else:
        print("SOME VALIDATIONS FAILED ✗")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
