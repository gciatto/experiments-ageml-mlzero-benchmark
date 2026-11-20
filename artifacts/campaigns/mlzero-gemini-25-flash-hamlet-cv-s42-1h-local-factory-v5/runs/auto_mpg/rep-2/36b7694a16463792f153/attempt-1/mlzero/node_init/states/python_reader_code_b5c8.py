import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/auto_mpg/fbd590420542-88e12ef7ab58-b8f667a91497-local/evaluate_cv.py"

def analyze_file(file_path):
    try:
        # Attempt to read as a text file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(768)
            print(content)
            if len(content) == 768:
                print("...")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        # Fallback for other reading errors (e.g., encoding)
        print(f"Could not read as text file. Error: {e}")
        print("Attempting to read as binary for basic info.")
        try:
            with open(file_path, 'rb') as f:
                header = f.read(64)
                print(f"Binary file header (first 64 bytes): {header}")
        except Exception as e_bin:
            print(f"Could not even read as binary. Error: {e_bin}")

analyze_file(file_path)