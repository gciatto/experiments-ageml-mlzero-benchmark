import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/madelon/52ab263e132d-d495b2ba732a-b8f667a91497-local/README.md"

try:
    # For text files, display first few lines (up to 768 characters)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(768)
        print(content)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"Error reading file as text: {e}")