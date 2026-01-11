#!/usr/bin/env python3
"""
External evaluator for language model submissions.

Validates submission size, runs the model in a sandboxed Docker container,
streams test contexts to it, and computes per-byte cross-entropy loss.

Usage:
    python evaluate.py --submission submission.zip --test-data dataset/ --output score.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk

MAX_ARCHIVE_MB = 1.0
CONTEXT_WINDOW = 512
BATCH_SIZE = 1024
DOCKER_IMAGE = "gki_evaluator"
LN2 = torch.log(torch.tensor(2.0)).item()


def validate_submission(path: Path) -> None:
    """Check submission exists and is under size limit."""
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_ARCHIVE_MB:
        raise ValueError(f"Submission too large: {size_mb:.2f} MB > {MAX_ARCHIVE_MB} MB")
    print(f"✓ Submission: {size_mb:.2f} MB")


def build_contexts(texts: list[str], context_window: int) -> list[tuple[list[int], int]]:
    """Create (context, target) pairs from text documents."""
    pairs = []
    for text in texts:
        data = text.encode("utf-8")
        for i in range(len(data)):
            start = max(0, i - context_window)
            pairs.append((list(data[start:i]), data[i]))
    random.shuffle(pairs)
    return pairs


class DockerRunner:
    """Communicates with model running in Docker container."""

    def __init__(self, submission: Path, image: str):
        self.submission = submission.resolve()
        self.image = image
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "--network",
                "none",
                "--memory",
                "4g",
                "--cpus",
                "2.0",
                "--pids-limit",
                "100",
                "--security-opt",
                "no-new-privileges",
                "--cap-drop",
                "ALL",
                "--read-only",
                "--tmpfs",
                "/tmp:size=100M,noexec",
                "--platform",
                "linux/amd64",
                "-v",
                f"{self.submission.parent}:/app/submission:ro",
                self.image,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        resp = self._request({"type": "ping"})
        if resp.get("status") != "ok":
            raise RuntimeError(f"Container init failed: {resp}")
        return self

    def __exit__(self, *_):
        if self.proc:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def _request(self, req: dict) -> dict:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"Container died: {self.proc.stderr.read()}")
        return json.loads(line)

    def predict(self, contexts: list[list[int]]) -> torch.Tensor:
        resp = self._request({"type": "predict", "contexts": contexts})
        if resp.get("status") != "ok":
            raise RuntimeError(f"Prediction failed: {resp}")
        return torch.tensor(resp["logits"], dtype=torch.float32)


def evaluate(pairs: list[tuple[list[int], int]], runner: DockerRunner, batch_size: int) -> float:
    """Compute average cross-entropy loss over all context-target pairs."""
    total_loss = 0.0
    n = len(pairs)

    for i in range(0, n, batch_size):
        batch = pairs[i : i + batch_size]
        contexts = [ctx for ctx, _ in batch]
        targets = torch.tensor([tgt for _, tgt in batch], dtype=torch.long)
        logits = runner.predict(contexts)
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item() / LN2

        if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= n:
            print(
                f"  {min(i + batch_size, n):,}/{n:,} | loss: {total_loss / min(i + batch_size, n):.4f}"
            )

    return total_loss / n


def main():
    parser = argparse.ArgumentParser(description="Evaluate a language model submission")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/score.json"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--context-window", type=int, default=CONTEXT_WINDOW)
    parser.add_argument("--docker-image", type=str, default=DOCKER_IMAGE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("EVALUATOR")
    print("=" * 60)

    validate_submission(args.submission)

    dataset = load_from_disk(str(args.test_data))
    texts = dataset["text"]
    print(f"✓ Loaded {len(texts):,} documents")

    pairs = build_contexts(texts, args.context_window)
    print(f"✓ Built {len(pairs):,} context-target pairs")

    with DockerRunner(args.submission, args.docker_image) as runner:
        print("✓ Container ready")
        score = evaluate(pairs, runner, args.batch_size)

    print("=" * 60)
    print(f"SCORE: {score:.6f}")
    print("=" * 60)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "score": score,
                "num_examples": len(pairs),
                "context_window": args.context_window,
            },
            indent=2,
        )
    )
    print(f"✓ Saved to {args.output}")


if __name__ == "__main__":
    sys.exit(main() or 0)
