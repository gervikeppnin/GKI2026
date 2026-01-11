"""Package dummy submission."""
import zipfile
from pathlib import Path

Path("output").mkdir(exist_ok=True)
with zipfile.ZipFile("output/submission.zip", "w") as z:
    z.write("submission/model.py", "model.py")
print("Saved output/submission.zip")
