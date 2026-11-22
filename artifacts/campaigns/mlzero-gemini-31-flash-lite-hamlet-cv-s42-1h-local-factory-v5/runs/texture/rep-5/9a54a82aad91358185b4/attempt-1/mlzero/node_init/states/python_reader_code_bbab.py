import json
import pandas as pd
import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/texture/c3e1db050fee-1e668d3f7c50-b8f667a91497-local/cv_contract.json"

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
        
        pd.set_option('display.max_colwidth', 50)
        print(df.head(3).to_string(index=False))
    else:
        print(str(data)[:768])
except Exception:
    with open(file_path, 'r') as f:
        print(f.read(768))