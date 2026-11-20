import pandas as pd
import json
import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/madelon/52ab263e132d-d495b2ba732a-b8f667a91497-local/cv_contract.json"

output_parts = []
current_output_length = 0
max_output_chars = 768

def append_output(text):
    """Appends text to the output buffer, truncating if max_output_chars is reached."""
    global current_output_length
    if current_output_length >= max_output_chars:
        return

    remaining_chars_for_content = max_output_chars - current_output_length
    
    if len(text) > remaining_chars_for_content:
        # Truncate text and add ellipsis, ensuring ellipsis fits
        if remaining_chars_for_content > 3: # Need at least 4 chars for "a..."
            text_to_add = text[:remaining_chars_for_content - 3] + "..."
        else: # Not enough space for ellipsis, just take what fits
            text_to_add = text[:remaining_chars_for_content]
        current_output_length = max_output_chars # Mark as full
    else:
        text_to_add = text
        current_output_length += len(text_to_add)
    
    output_parts.append(text_to_add)

try:
    # Attempt to read as a pandas DataFrame
    df = pd.read_json(file_path)

    if isinstance(df, pd.DataFrame):
        append_output("File Type: Tabular (JSON converted to DataFrame)\n")

        # Display column names
        columns = df.columns.tolist()
        if len(columns) > 20:
            cols_display = columns[:10] + ["..."] + columns[-10:]
            append_output(f"Columns ({len(columns)} total): {', '.join(cols_display)}\n")
        else:
            append_output(f"Columns: {', '.join(columns)}\n")

        # Show first 2-3 rows with truncated cell content
        append_output("First 3 rows:\n")
        temp_df_str = df.head(3).to_string(index=False, max_colwidth=50)
        append_output(temp_df_str)

    else:
        # If pd.read_json returns something else (e.g., a Series, or a single dict)
        # Try to load it as a generic JSON object and represent it as text.
        append_output("File Type: JSON (non-tabular or complex structure)\n")
        with open(file_path, 'r', encoding='utf-8') as f:
            json_obj = json.load(f)
            obj_str = json.dumps(json_obj, indent=2)
            append_output(f"Content (first few lines):\n{obj_str}")

except Exception as e:
    # If pandas fails, or json.load fails, treat as a plain text file
    append_output(f"Could not parse as JSON/DataFrame. Attempting to read as text. Error: {e}\n")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            append_output("File Type: Text\n")
            append_output("First few lines:\n")
            append_output(content)
    except Exception as text_e:
        append_output(f"Could not read file as text. Error: {text_e}\n")
        append_output("File Type: Binary or unreadable\n")
        append_output("Basic information: File exists but could not be read as text.")

print("".join(output_parts))