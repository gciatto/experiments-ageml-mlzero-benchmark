import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/madelon/52ab263e132d-d495b2ba732a-b8f667a91497-local/evaluate_cv.py"

def analyze_file(file_path):
    max_display_chars = 768
    try:
        # Attempt to read as a text file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(max_display_chars)
            print(f"--- Content of {os.path.basename(file_path)} (first {len(content)} chars) ---")
            print(content)
            if len(content) == max_display_chars:
                print("...") # Indicate truncation
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except UnicodeDecodeError:
        # If it's not a UTF-8 text file, treat it as a generic binary file
        print(f"File {os.path.basename(file_path)} is not a readable text file (UnicodeDecodeError). Displaying basic info.")
        try:
            with open(file_path, 'rb') as f:
                binary_content = f.read(256) # Read a small chunk for basic info
                print(f"--- Binary content of {os.path.basename(file_path)} (first {len(binary_content)} bytes) ---")
                print(binary_content)
                if len(binary_content) == 256:
                    print("...")
        except Exception as e_bin:
            print(f"Could not read file even as binary: {e_bin}")
    except Exception as e:
        print(f"An unexpected error occurred while reading {os.path.basename(file_path)}: {e}")

analyze_file(file_path)