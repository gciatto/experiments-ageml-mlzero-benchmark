import os

def analyze_file(file_path):
    file_size = os.path.getsize(file_path)
    
    if file_size == 0:
        print(f"File is empty: {file_path}")
        return

    try:
        # Treat .py files as text files
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(768)
            print(f"--- Content of {file_path} (first 768 characters) ---")
            print(content)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        # Fallback for other errors, try reading as binary if text fails
        try:
            with open(file_path, 'rb') as f:
                content = f.read(768)
                print(f"--- Binary content of {file_path} (first 768 bytes) ---")
                print(content)
        except Exception as e_bin:
            print(f"Error reading file as binary {file_path}: {e_bin}")

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/ames_housing/06b4acbf769d-ebb2a7fe4397-b8f667a91497-local/evaluate_cv.py"
analyze_file(file_path)