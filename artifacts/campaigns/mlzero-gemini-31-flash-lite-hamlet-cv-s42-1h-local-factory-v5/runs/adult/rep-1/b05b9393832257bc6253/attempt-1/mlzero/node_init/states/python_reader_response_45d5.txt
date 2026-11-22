import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/adult/547551a449e2-88ada5f9c32f-b8f667a91497-local/evaluate_cv.py"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(768)
        print(f"File: {os.path.basename(file_path)}")
        print("-" * 20)
        print(content)
except Exception as e:
    print(f"Error reading file: {e}")