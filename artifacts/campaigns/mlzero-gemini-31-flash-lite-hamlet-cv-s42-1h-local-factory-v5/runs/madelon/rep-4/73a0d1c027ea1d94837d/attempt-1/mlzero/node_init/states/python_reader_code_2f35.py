import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/madelon/52ab263e132d-d495b2ba732a-b8f667a91497-local/evaluate_cv.py"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print("--- File Content (First 768 chars) ---")
        print(content[:768])
except Exception as e:
    print(f"Could not read file as text: {e}")
    file_stats = os.stat(file_path)
    print(f"File size: {file_stats.st_size} bytes")