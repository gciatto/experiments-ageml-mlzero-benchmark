import pandas as pd
import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/mnist_784/30c9e1582974-dda30b451a9f-b8f667a91497-local/cv_contract.json"

def truncate_cell_content(content, length=50):
    s = str(content)
    return s[:length] + '...' if len(s) > length else s

try:
    # Attempt to read as a pandas DataFrame
    df = pd.read_json(file_path)

    if isinstance(df, pd.DataFrame) and not df.empty:
        print("Tabular data (JSON):")
        columns = df.columns.tolist()
        if len(columns) > 20:
            print("Columns:", columns[:10] + ['...'] + columns[-10:])
        else:
            print("Columns:", columns)

        # Display first 3 rows with truncated content
        truncated_head = df.head(3).applymap(truncate_cell_content)
        # Use to_string to control output format, no index, max_rows, max_cols
        print(truncated_head.to_string(index=False, max_rows=3, max_cols=7))

    else:
        # If pandas couldn't parse it into a DataFrame (e.g., it's a JSON object, not an array of objects)
        # or if it's an empty JSON, treat as text.
        raise ValueError("Not a flat tabular JSON structure or empty.")

except Exception as e:
    # If pandas.read_json fails or raises an error, treat as a generic text file.
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            print("Text file content (first 768 chars):")
            print(content)
            if len(content) == 768:
                print("... (truncated)")
    except Exception as text_e:
        print(f"Could not read file as text. Error: {text_e}")
        # Fallback for truly unreadable files (e.g., binary, permission issues)
        if os.path.exists(file_path):
            print(f"File exists. Size: {os.path.getsize(file_path)} bytes.")
        else:
            print("File does not exist.")