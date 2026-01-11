# GKÍ Evaluator

Evaluation system for tiny byte-level language model submissions. Models run in a sandboxed Docker container; scoring happens externally.

## Setup

```bash
# Create environment and install
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Build Docker image (one-time)
docker build -t gki_evaluator -f docker/Dockerfile docker/

# Set up test dataset (HuggingFace format)
# e.g. dummy data for testing:
python -c "from datasets import Dataset; Dataset.from_dict({'text': ['Þetta er prófunarsetning!']}).save_to_disk('data/')"
# NOTE: Use the official test dataset for real evaluations.
```

## Evaluate a Submission

```bash
python -m gki_evaluator.evaluate \
    --submission examples/mlp/output/submission.zip \
    --test-data data/
```

Output: `output/score.json` with per-byte cross-entropy loss (lower is better).

## Create a Submission

A `submission.zip` must contain a `model.py` at the root with a `Model` class:

```python
class Model:
    def __init__(self, submission_dir: Path):
        # Optionally load weights from submission_dir
        pass

    def predict(self, contexts: list[list[int]]) -> list[list[float]]:
        # contexts: batch of byte sequences (0-255), up to 512 bytes each
        # returns: batch of logit vectors, 256 floats each
        pass
```

### Constraints

- Archive size: **1 MB** max (compressed)
- Context window: up to **512 bytes**
- Frameworks available: PyTorch, TensorFlow, JAX/Flax, NumPy, SciPy
- CPU-only, no network access

## Examples

```bash
# Dummy baseline (uniform distribution, score ≈ 8.0)
cd examples/dummy && python package.py

# MLP baseline
cd examples/mlp && python train.py
```

Both produce `output/submission.zip`.

## Project Structure

```
├── src/gki_evaluator/    # Evaluator package
│   └── evaluate.py
├── examples/
│   ├── dummy/            # Random baseline
│   └── mlp/              # MLP baseline
├── docker/
│   ├── Dockerfile
│   └── scoring.py        # Runs inside container
└── data/                 # Test dataset (HuggingFace format)
```

## Security

The container runs with: no network, 4GB RAM, 2 CPUs, read-only filesystem, dropped capabilities, non-root user. See [docker/Dockerfile](docker/Dockerfile) for details.
