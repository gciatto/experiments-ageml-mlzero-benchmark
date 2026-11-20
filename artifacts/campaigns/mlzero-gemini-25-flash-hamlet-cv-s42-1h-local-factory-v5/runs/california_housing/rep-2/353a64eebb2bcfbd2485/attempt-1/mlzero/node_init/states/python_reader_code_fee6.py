import pandas as pd

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/california_housing/45e44ec0cb85-d3be51c254a5-b8f667a91497-local/data.csv"

try:
    # Try to read as CSV. pandas' read_csv has compression='infer' by default,
    # which can handle gzip, bz2, zip, xz, zstd if detected from magic bytes.
    df = pd.read_csv(file_path)

    # Display column names
    columns = df.columns.tolist()
    if len(columns) > 20:
        print("Cols:", columns[:10], "...", columns[-10:])
    else:
        print("Cols:", columns)

    # Show first 2 rows with truncated cell content (50 chars), no index
    print("\nHead (2 rows):\n", df.head(2).to_string(index=False, max_colwidth=50))

except Exception as e:
    # If pandas fails (e.g., not a valid CSV, not a recognized compressed tabular, or file not found),
    # treat it as a text file.
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            print("Text content (first 768 chars):\n", content)
    except Exception as text_e:
        # If even opening as text fails (e.g., binary file, permission error, truly non-existent)
        print(f"Failed to open file: {text_e}")
        print("File might be binary or inaccessible.")