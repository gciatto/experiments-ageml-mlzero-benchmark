import json
import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/wilt/a7fbce33d504-1f460de77002-b8f667a91497-local/cv_contract.json"

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    if isinstance(data, list):
        print(f"Columns: {list(data[0].keys()) if data else 'Empty'}")
        for row in data[:2]:
            print({k: str(v)[:50] for k, v in row.items()})
    else:
        print(json.dumps(data, indent=2)[:768])

except Exception:
    try:
        with open(file_path, 'r') as f:
            print(f.read(768))
    except Exception as e:
        print(f"File info: {os.path.getsize(file_path)} bytes, Binary/Other.")