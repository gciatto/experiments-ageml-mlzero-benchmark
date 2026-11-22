import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/california_housing/45e44ec0cb85-d3be51c254a5-b8f667a91497-local/README.md"

def analyze_file(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read(768)
            print(content)
    except Exception as e:
        print(f"Binary or unreadable file: {path}")
        print(f"Size: {os.path.getsize(path)} bytes")

if __name__ == "__main__":
    analyze_file(file_path)