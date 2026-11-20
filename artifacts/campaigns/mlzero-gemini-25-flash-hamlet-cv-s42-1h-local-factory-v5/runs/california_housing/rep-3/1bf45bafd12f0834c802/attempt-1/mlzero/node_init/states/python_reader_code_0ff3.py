import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/california_housing/45e44ec0cb85-d3be51c254a5-b8f667a91497-local/README.md"

try:
    # Attempt to read the file as a text file (e.g., .md, .txt)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(768) # Read up to 768 characters
        print(content)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except UnicodeDecodeError:
    # If it's not a UTF-8 text file, try reading as binary
    try:
        with open(file_path, 'rb') as f:
            content = f.read(768) # Read up to 768 bytes
            print(f"File content (first {len(content)} bytes):\n{content}")
    except Exception as e_binary:
        print(f"Could not read file as binary: {e_binary}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")