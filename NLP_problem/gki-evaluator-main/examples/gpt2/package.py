"""Package GPT-2 submission (conceptual demo)."""

import zipfile
from pathlib import Path

# NOTE: This creates a minimal submission showing the structure.
# A real GPT-2 submission would need heavily quantized weights.

Path("output").mkdir(exist_ok=True)

with zipfile.ZipFile("output/submission.zip", "w") as z:
    z.write("submission/model.py", "model.py")
    # In a real submission, you'd also include:
    # z.write("submission/model_quantized.pt", "model_quantized.pt")

size_mb = Path("output/submission.zip").stat().st_size / 1e6
print(f"Saved output/submission.zip ({size_mb:.2f} MB)")

if size_mb > 1.0:
    print(f"WARNING: Submission exceeds 1 MB limit!")
else:
    print(f"✓ Submission is within 1 MB limit")

# Theoretical size calculation
print("\n=== Size Analysis ===")
print("distilgpt2 uncompressed: ~240 MB")
print("With 8-bit quant: ~60 MB")
print("With 4-bit quant: ~30 MB")
print("With 2-bit quant: ~15 MB")
print("Required for 1MB: <250k params (4-bit) or aggressive distillation")
