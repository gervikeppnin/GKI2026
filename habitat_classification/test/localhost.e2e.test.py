# Test a patch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_training_data, encode_patch
import httpx

patches, _ = load_training_data()
encoded = encode_patch(patches[0])

response = httpx.post('http://localhost:4321/predict', json={'patch': encoded}, headers={'Authorization': 'Bearer abc123'})
print(response.json())