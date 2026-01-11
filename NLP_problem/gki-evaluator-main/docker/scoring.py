"""
Predictor service for language model submissions.

Loads a participant's model from /app/submission/submission.zip and serves
next-byte predictions over stdin/stdout JSON protocol. Scoring is performed
by the external evaluator.

Protocol:
    Request:  {"type": "predict", "contexts": [[int, ...], ...]}
    Response: {"status": "ok", "logits": [[float x 256], ...]}

    Request:  {"type": "ping"}
    Response: {"status": "ok"}
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Protocol

SUBMISSION_ARCHIVE = Path("/app/submission/submission.zip")
UNPACKED_DIR = Path("/tmp/submission")
MAX_UNCOMPRESSED = 50 * 1024 * 1024  # 50 MB safety cap
CONTEXT_WINDOW = 512


class ModelProtocol(Protocol):
    """Interface for submission models (framework-agnostic)."""

    def __init__(self, submission_dir: Path) -> None:
        """Initialize model, optionally loading weights from submission_dir."""
        ...

    def predict(self, contexts: list[list[int]]) -> list[list[float]]:
        """Predict next-byte logits for each context.

        Args:
            contexts: Batch of byte sequences, each 0-512 bytes long.

        Returns:
            Batch of logit vectors, each with 256 floats.
        """
        ...


def extract_submission() -> Path:
    """Extract submission.zip with path traversal and zip bomb protection."""
    UNPACKED_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(SUBMISSION_ARCHIVE) as zf:
        total = 0
        for info in zf.infolist():
            target = (UNPACKED_DIR / info.filename).resolve()
            if not str(target).startswith(str(UNPACKED_DIR.resolve())):
                raise ValueError("Path traversal detected")
            total += info.file_size
            if total > MAX_UNCOMPRESSED:
                raise ValueError("Uncompressed size exceeds limit")
        zf.extractall(UNPACKED_DIR)

    return UNPACKED_DIR


def load_model() -> ModelProtocol:
    """Load and instantiate the participant's Model class."""
    root = extract_submission()
    model_file = root / "model.py"

    spec = importlib.util.spec_from_file_location("submission", model_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission"] = module
    spec.loader.exec_module(module)

    return module.Model(submission_dir=root)


def validate_contexts(contexts: list[list[int]]) -> None:
    """Check context lengths are within bounds."""
    for i, ctx in enumerate(contexts):
        if len(ctx) > CONTEXT_WINDOW:
            raise ValueError(f"Context {i} exceeds {CONTEXT_WINDOW} bytes")


def respond(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def serve(model: ModelProtocol) -> None:
    """Main prediction loop over stdin/stdout."""
    print("Ready", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            respond({"status": "error", "error": str(e)})
            continue

        if msg.get("type") == "ping":
            respond({"status": "ok"})
        elif msg.get("type") == "predict":
            try:
                validate_contexts(msg["contexts"])
                logits = model.predict(msg["contexts"])
                respond({"status": "ok", "logits": logits})
            except Exception as e:
                respond({"status": "error", "error": str(e)})
        else:
            respond({"status": "error", "error": "unknown type"})


if __name__ == "__main__":
    try:
        serve(load_model())
    except Exception as e:
        print(f"Failed to start: {e}", file=sys.stderr)
        sys.exit(1)
