import pandas as pd
import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/auto_mpg/fbd590420542-88e12ef7ab58-b8f667a91497-local/data.csv"

try:
    df = pd.read_csv(file_path)
    cols = df.columns
    if len(cols) > 20:
        print(list(cols[:10]) + ["..."] + list(cols[-10:]))
    else:
        print(list(cols))
    
    pd.set_option('display.max_colwidth', 50)
    print(df.head(3).to_string(index=False))
except Exception:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        print(f.read(768))