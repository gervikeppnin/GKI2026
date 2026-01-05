from Network_Traversal.core.model_storage import load_model
from pathlib import Path
import sys

# Ensure we can import modules
sys.path.append(".")

try:
    model_path = Path("models/ig_v2_calibrated.json")
    print(f"Loading {model_path}...")
    model = load_model(model_path)
    print(f"Loaded: {model.name}")
    print(f"Version: {model.version}")
    print(f"Pipe Roughness Keys: {list(model.pipe_roughness.keys())[:5]}")
    
    if model.version >= 2:
        print("Success: Model is Version 2")
    else:
        print("Failure: Model is Version 1 (Legacy)")
        
except Exception as e:
    print(f"Error: {e}")
