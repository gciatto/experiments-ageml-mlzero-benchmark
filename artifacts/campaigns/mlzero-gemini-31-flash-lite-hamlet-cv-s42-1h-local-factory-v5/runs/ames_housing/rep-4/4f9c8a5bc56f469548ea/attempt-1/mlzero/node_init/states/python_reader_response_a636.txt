import pandas as pd
import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/ames_housing/06b4acbf769d-ebb2a7fe4397-b8f667a91497-local/data.csv"

try:
    df = pd.read_csv(file_path)
    cols = df.columns.tolist()
    if len(cols) > 20:
        print(f"Columns: {cols[:10]} ... {cols[-10:]}")
    else:
        print(f"Columns: {cols}")
    
    pd.set_option('display.max_colwidth', 50)
    print(df.head(3).to_string(index=False))
except Exception:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        print(f.read(768))