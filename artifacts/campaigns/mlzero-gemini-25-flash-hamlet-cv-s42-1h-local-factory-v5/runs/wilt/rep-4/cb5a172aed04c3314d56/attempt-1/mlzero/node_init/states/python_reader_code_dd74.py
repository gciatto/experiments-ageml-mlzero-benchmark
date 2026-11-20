file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/wilt/a7fbce33d504-1f460de77002-b8f667a91497-local/evaluate_cv.py"

try:
    # Treat as a text file due to .py extension
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(768)
        print(content)
except UnicodeDecodeError:
    # If it's not valid UTF-8, try a common alternative encoding
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read(768)
            print(content)
    except Exception:
        # Fallback to binary if text reading fails completely
        try:
            with open(file_path, 'rb') as f:
                content = f.read(768)
                print(f"Binary content (first 768 bytes):\n{content}")
        except Exception as e_bin:
            print(f"Could not read file even as binary: {e_bin}")
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")