import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/california_housing/45e44ec0cb85-d3be51c254a5-b8f667a91497-local/evaluate_cv.py"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"--- File: {os.path.basename(file_path)} ---")
        print(content[:768])
except Exception as e:
    print(f"Could not read file: {e}")