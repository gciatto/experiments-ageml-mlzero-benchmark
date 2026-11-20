import pandas as pd

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/texture/c3e1db050fee-1e668d3f7c50-b8f667a91497-local/data.csv"

try:
    # Attempt to read as a tabular file (CSV)
    df = pd.read_csv(file_path)

    # Display column names
    columns = df.columns.tolist()
    if len(columns) > 20:
        cols_display = columns[:10] + ['...'] + columns[-10:]
        print(f"Columns ({len(columns)} total): {', '.join(cols_display)}")
    else:
        print(f"Columns ({len(columns)} total): {', '.join(columns)}")

    # Show first 3 rows with truncated cell content
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False, max_colwidth=50))

except Exception as e:
    # If reading as CSV fails, treat it as a text file
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768) # Read up to 768 characters
            print(f"Could not parse as tabular data ({type(e).__name__}: {e}). Displaying first 768 characters as text:\n")
            print(content)
    except Exception as text_e:
        print(f"Could not open file as text either: {text_e}")