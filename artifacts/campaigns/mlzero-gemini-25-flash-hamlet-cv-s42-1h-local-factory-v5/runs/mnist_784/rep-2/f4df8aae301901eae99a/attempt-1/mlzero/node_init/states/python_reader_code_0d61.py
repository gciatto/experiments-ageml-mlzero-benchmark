import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/mnist_784/30c9e1582974-dda30b451a9f-b8f667a91497-local/evaluate_cv.py"

try:
    # Attempt to read as a text file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(768) # Read up to 768 characters
        print(content)
except Exception:
    # If reading as text fails, treat as a binary file
    try:
        with open(file_path, 'rb') as f:
            header = f.read(256) # Read a small portion for basic info
            print(f"File is likely binary or unreadable. First 256 bytes: {header}")
    except Exception as e_binary:
        print(f"Could not read file: {e_binary}")