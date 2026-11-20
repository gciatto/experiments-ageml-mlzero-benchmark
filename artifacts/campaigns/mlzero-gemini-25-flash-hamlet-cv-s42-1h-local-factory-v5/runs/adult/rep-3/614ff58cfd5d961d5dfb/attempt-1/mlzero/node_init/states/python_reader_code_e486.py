import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/adult/547551a449e2-88ada5f9c32f-b8f667a91497-local/README.md"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(768)
        print(content)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred while reading the file: {e}")