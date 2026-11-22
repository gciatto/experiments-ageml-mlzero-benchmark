import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/mnist_784/30c9e1582974-dda30b451a9f-b8f667a91497-local/evaluate_cv.py"

def analyze_file(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            print(f"--- File Content (First 768 chars) ---\n{content}")
    except Exception as e:
        print(f"Could not read file as text: {e}")
        file_stats = os.stat(path)
        print(f"File size: {file_stats.st_size} bytes")

if __name__ == "__main__":
    analyze_file(file_path)