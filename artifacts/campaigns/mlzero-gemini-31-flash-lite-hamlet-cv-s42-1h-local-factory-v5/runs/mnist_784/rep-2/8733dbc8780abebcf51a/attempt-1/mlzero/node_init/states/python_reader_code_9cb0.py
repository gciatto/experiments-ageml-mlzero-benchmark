import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/mnist_784/30c9e1582974-dda30b451a9f-b8f667a91497-local/README.md"

def analyze_file(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read(768)
            print(f"--- File: {os.path.basename(path)} ---")
            print(content)
    except Exception as e:
        print(f"Binary or unreadable file: {os.path.basename(path)}")

if __name__ == "__main__":
    analyze_file(file_path)