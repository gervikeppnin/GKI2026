"""Byte-level MLP submission."""

from pathlib import Path

import torch
import torch.nn as nn

HIDDEN, CONTEXT, LAYERS = 128, 5, 2


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(256, HIDDEN)
        layers = [nn.Linear(HIDDEN * CONTEXT, HIDDEN), nn.ReLU()]
        for _ in range(LAYERS - 1):
            layers += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
        self.net = nn.Sequential(*layers, nn.Linear(HIDDEN, 256))

    def forward(self, x):
        return self.net(self.emb(x).flatten(1))


class Model:
    def __init__(self, submission_dir: Path):
        self.model = MLP()
        self.model.load_state_dict(torch.load(submission_dir / "model.bin", map_location="cpu"))
        self.model.eval()

    def predict(self, contexts: list[list[int]]) -> list[list[float]]:
        x = torch.zeros(len(contexts), CONTEXT, dtype=torch.long)
        for i, ctx in enumerate(contexts):
            if ctx:
                n = min(len(ctx), CONTEXT)
                x[i, -n:] = torch.tensor(ctx[-n:])
        with torch.no_grad():
            return self.model(x).tolist()
