# Test a patch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_training_data, encode_patch
import urllib.request
import json

patches, _ = load_training_data()
encoded = encode_patch(patches[0])

data = json.dumps({'patch': encoded}).encode('utf-8')
req = urllib.request.Request(
    'https://func-doomer-habitat.azurewebsites.net/api/predict',
    data=data,
    headers={'Authorization': 'Bearer abc123', 'Content-Type': 'application/json'}
)

try:
    with urllib.request.urlopen(req) as response:
        print(f"Status: {response.status}")
        body = response.read().decode('utf-8')
        print(f"Response: {body}")
        if response.status == 200:
            print(json.loads(body))
except urllib.error.HTTPError as e:
    print(f"Status: {e.code}")
    print(f"Response: {e.read().decode('utf-8')}")