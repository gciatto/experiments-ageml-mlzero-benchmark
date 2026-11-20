import json
import pandas as pd
import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/california_housing/45e44ec0cb85-d3be51c254a5-b8f667a91497-local/cv_contract.json"

output_parts = []
max_total_chars = 768
current_output_length = 0

def append_output(text):
    global current_output_length
    global output_parts
    
    if current_output_length >= max_total_chars:
        return

    remaining_space = max_total_chars - current_output_length
    
    if len(text) <= remaining_space:
        output_parts.append(text)
        current_output_length += len(text)
    else:
        # Truncate and add "..." if there's space for it
        if remaining_space >= 3:
            output_parts.append(text[:remaining_space-3] + "...")
            current_output_length = max_total_chars # Mark as full
        else: # Not enough space for "..."
            output_parts.append(text[:remaining_space])
            current_output_length = max_total_chars # Mark as full

try:
    with open(file_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        # Likely tabular data (list of dictionaries)
        try:
            df = pd.DataFrame(data)
            append_output("Tabular JSON data detected.\n")

            # Display column names
            cols = df.columns.tolist()
            if len(cols) > 20:
                append_output(f"Columns ({len(cols)}): {cols[:10] + ['...'] + cols[-10:]}\n")
            else:
                append_output(f"Columns ({len(cols)}): {cols}\n")

            # Display first 3 rows with truncated cell content
            append_output("First 3 rows:\n")
            truncated_df = df.head(3).applymap(lambda x: (str(x)[:50] + '...') if len(str(x)) > 50 else str(x))
            df_str = truncated_df.to_string(index=False, header=True)
            append_output(df_str + "\n")

        except Exception: # If DataFrame conversion fails or is not suitable
            append_output("JSON list of dicts, but not easily tabular or too complex for simple DataFrame. Displaying raw JSON snippet.\n")
            json_str = json.dumps(data, indent=2)
            append_output(json_str)

    elif isinstance(data, dict):
        # Dictionary data
        append_output("JSON dictionary detected.\n")
        append_output("Top-level keys: " + str(list(data.keys())) + "\n")
        append_output("First few key-value pairs:\n")
        count = 0
        for k, v in data.items():
            if count >= 3: # Limit to 3 pairs
                append_output("  ...\n")
                break
            v_str = str(v)
            if len(v_str) > 100: # Truncate value string
                v_str = v_str[:100] + "..."
            append_output(f"  '{k}': {v_str}\n")
            count += 1
    else:
        # Other simple JSON types (e.g., a simple list, string, number)
        append_output("Simple JSON data detected.\n")
        json_str = json.dumps(data, indent=2)
        append_output(json_str)

except json.JSONDecodeError:
    # Not a valid JSON, treat as text file
    append_output("File is not a valid JSON. Treating as text file.\n")
    try:
        with open(file_path, 'r') as f:
            content = f.read(max_total_chars)
            append_output(content)
    except Exception as e:
        append_output(f"Could not read file as text: {e}\n")
except FileNotFoundError:
    append_output(f"Error: File not found at {file_path}\n")
except Exception as e:
    append_output(f"An unexpected error occurred: {e}\n")

print("".join(output_parts))