import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/auto_mpg/fbd590420542-88e12ef7ab58-b8f667a91497-local/evaluate_cv.py"

def analyze_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read(768)
            print(content)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
    except Exception as e:
        print(f"Error reading file {path}: {e}")

if __name__ == "__main__":
    analyze_file(file_path)