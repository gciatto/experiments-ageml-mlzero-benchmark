import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/adult/547551a449e2-88ada5f9c32f-b8f667a91497-local/README.md"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(768)
        print(content)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
    # Fallback for other errors, try reading as binary to get basic info
    try:
        with open(file_path, 'rb') as f:
            print(f"File type: binary/unknown. Size: {os.path.getsize(file_path)} bytes.")
    except Exception as e_binary:
        print(f"Could not even read as binary: {e_binary}")