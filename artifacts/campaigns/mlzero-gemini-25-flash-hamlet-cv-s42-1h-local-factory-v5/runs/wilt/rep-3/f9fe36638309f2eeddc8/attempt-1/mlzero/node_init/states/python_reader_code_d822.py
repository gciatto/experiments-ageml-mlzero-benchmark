import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/wilt/a7fbce33d504-1f460de77002-b8f667a91497-local/README.md"
max_display_chars = 768

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(max_display_chars)
        print(content)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"Error reading file as text: {e}")