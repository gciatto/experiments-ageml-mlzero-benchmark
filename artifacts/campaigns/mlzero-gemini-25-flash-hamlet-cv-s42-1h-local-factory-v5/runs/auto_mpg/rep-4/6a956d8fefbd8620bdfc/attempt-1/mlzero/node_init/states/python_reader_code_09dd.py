import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/auto_mpg/fbd590420542-88e12ef7ab58-b8f667a91497-local/README.md"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(768)
        print(content)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"Error reading file as text: {e}")
    # Fallback for other errors, try reading as binary to get basic info
    try:
        with open(file_path, 'rb') as f:
            print(f"File is likely binary or corrupted. Size: {os.path.getsize(file_path)} bytes.")
    except Exception as e_bin:
        print(f"Could not even get basic info: {e_bin}")