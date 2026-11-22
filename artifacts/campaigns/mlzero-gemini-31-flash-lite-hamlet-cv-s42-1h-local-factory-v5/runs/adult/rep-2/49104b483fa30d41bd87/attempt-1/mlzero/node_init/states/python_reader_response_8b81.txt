import pandas as pd
import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/adult/547551a449e2-88ada5f9c32f-b8f667a91497-local/data.csv"

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
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            print(f.read(768))
    except Exception as e:
        print(f"File info: {os.path.getsize(file_path)} bytes, Binary/Other.")