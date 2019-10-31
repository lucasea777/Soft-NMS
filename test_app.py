import requests
from pathlib import Path
import json
res = requests.post('http://localhost:5000/nms', json=json.loads(Path("test1.json").read_text()))
if res.ok:
    print(res.json())