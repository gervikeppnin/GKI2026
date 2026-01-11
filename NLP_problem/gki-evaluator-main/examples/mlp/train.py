"""Train a byte-level MLP and package submission."""

import zipfile
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset

# Config
HIDDEN, CONTEXT, LAYERS = 128, 5, 2
BATCH, EPOCHS, LR, MAX_BYTES = 1024, 3, 1e-3, 500_000


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


# Load data
data = bytearray()
for t in load_dataset("HuggingFaceFW/fineweb-2", "isl_Latn", split="train", streaming=True):
    data.extend(t["text"].encode() + b"\n\n")
    if len(data) >= MAX_BYTES:
        break
data = torch.tensor(list(data[:MAX_BYTES]))

# Build windows: [context, target]
windows = data.unfold(0, CONTEXT + 1, 1)
X, Y = windows[:, :CONTEXT], windows[:, -1]

# Train
device = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
model = MLP().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"Training on {device}, {sum(p.numel() for p in model.parameters()):,} params")

for epoch in range(EPOCHS):
    total = 0
    for i in range(0, len(X), BATCH):
        x, y = X[i : i + BATCH].to(device), Y[i : i + BATCH].to(device)
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() / torch.log(torch.tensor(2.0))
    print(f"Epoch {epoch + 1}: {total * BATCH / len(X):.3f}")

# Save
model.cpu()
Path("output").mkdir(exist_ok=True)
torch.save(model.state_dict(), "submission/model.bin")
with zipfile.ZipFile("output/submission.zip", "w") as z:
    z.write("submission/model.py", "model.py")
    z.write("submission/model.bin", "model.bin")
print(f"Saved output/submission.zip ({Path('output/submission.zip').stat().st_size / 1e6:.2f} MB)")
