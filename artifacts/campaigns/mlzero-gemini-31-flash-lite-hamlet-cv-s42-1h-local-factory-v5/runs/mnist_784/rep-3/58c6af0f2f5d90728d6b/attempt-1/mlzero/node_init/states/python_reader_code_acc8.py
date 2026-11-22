import json
import pandas as pd
import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/mnist_784/30c9e1582974-dda30b451a9f-b8f667a91497-local/cv_contract.json"

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        df = pd.DataFrame(data)
        cols = df.columns.tolist()
        if len(cols) > 20:
            print(f"Columns: {cols[:10]} ... {cols[-10:]}")
        else:
            print(f"Columns: {cols}")
        print(df.head(3).applymap(lambda x: str(x)[:50]).to_string(index=False))
    else:
        print(json.dumps(data, indent=2)[:768])
except Exception:
    with open(file_path, 'r') as f:
        print(f.read(768))