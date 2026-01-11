"""Random predictor baseline - outputs uniform distribution."""
from pathlib import Path


class Model:
    def __init__(self, submission_dir: Path):
        pass

    def predict(self, contexts: list[list[int]]) -> list[list[float]]:
        return [[0.0] * 256 for _ in contexts]  # Uniform logits
