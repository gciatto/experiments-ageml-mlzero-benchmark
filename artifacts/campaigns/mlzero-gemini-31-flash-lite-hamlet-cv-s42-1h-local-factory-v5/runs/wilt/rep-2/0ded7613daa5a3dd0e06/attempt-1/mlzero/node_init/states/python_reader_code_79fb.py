import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/wilt/a7fbce33d504-1f460de77002-b8f667a91497-local/evaluate_cv.py"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"File: {os.path.basename(file_path)}")
        print("-" * 20)
        print(content[:768])
except Exception as e:
    print(f"Could not read file: {e}")