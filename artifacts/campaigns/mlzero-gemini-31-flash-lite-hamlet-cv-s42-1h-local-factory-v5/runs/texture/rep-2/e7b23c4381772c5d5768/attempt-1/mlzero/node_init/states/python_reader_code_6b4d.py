import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/texture/c3e1db050fee-1e668d3f7c50-b8f667a91497-local/README.md"

def analyze_file(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read(768)
            print(content)
    except Exception as e:
        print(f"Could not read file as text: {e}")

if __name__ == "__main__":
    analyze_file(file_path)