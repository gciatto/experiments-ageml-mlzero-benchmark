import pandas as pd
import os
import json

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/adult/547551a449e2-88ada5f9c32f-b8f667a91497-local/cv_contract.json"

output_lines = []

try:
    # Attempt to read as JSON using pandas
    df = pd.read_json(file_path)

    if not df.empty:
        output_lines.append("Column Names:")
        columns = df.columns.tolist()
        if len(columns) > 20:
            output_lines.append(str(columns[:10] + ['...'] + columns[-10:]))
        else:
            output_lines.append(str(columns))

        output_lines.append("\nFirst 2 Rows (first 3 columns, truncated to 50 chars per cell):")
        # Select only the first 3 columns for row display to keep output concise
        display_cols_for_rows = columns[:3]
        df_to_display = df[display_cols_for_rows].head(2)
        output_lines.append(df_to_display.to_string(index=False, max_colwidth=50, show_dimensions=False))

    else:
        # If pandas reads it but it's empty or not a simple DataFrame structure
        output_lines.append(f"File '{os.path.basename(file_path)}' is empty or not a simple tabular JSON.")
        raise ValueError("Not a simple tabular JSON.") # Force fallback to text reading

except Exception as e:
    # If pandas fails (e.g., not a simple JSON array of objects, or malformed)
    # Try reading as a generic text file
    output_lines = [f"Could not parse as tabular. First 768 chars as text:"]
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            output_lines.append(content)
    except Exception:
        # Fallback for binary or unreadable files
        output_lines = [f"File: {os.path.basename(file_path)}"]
        output_lines.append(f"Size: {os.path.getsize(file_path)} bytes")
        output_lines.append("Content type: Binary or unreadable.")

final_output = "\n".join(output_lines)
# Ensure total output is under 768 characters
if len(final_output) > 768:
    final_output = final_output[:765] + "..."

print(final_output)