import pandas as pd
import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/madelon/52ab263e132d-d495b2ba732a-b8f667a91497-local/data.csv"
max_output_chars = 768
output_parts = []

try:
    # Attempt to read as CSV
    df = pd.read_csv(file_path)

    columns = df.columns.tolist()
    num_columns = len(columns)

    # Column names output
    col_str = f"Columns ({num_columns}): "
    if num_columns > 20:
        col_str += str(columns[:10] + ['...'] + columns[-10:])
    else:
        col_str += str(columns)
    output_parts.append(col_str)

    # First 3 rows with truncated cell content (50 chars)
    # Use to_string with max_colwidth for truncation and no index
    with pd.option_context('display.max_colwidth', 50):
        head_str = df.head(3).to_string(index=False)
    output_parts.append(head_str)

except FileNotFoundError:
    output_parts.append(f"File not found: {file_path}")
except pd.errors.EmptyDataError:
    output_parts.append(f"File is empty: {file_path}")
except Exception as e: # Catch all other potential errors (ParserError, UnicodeDecodeError, etc.)
    # Fallback to text file if pandas fails
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_output_chars) # Read up to max_output_chars for text
            output_parts.append(f"Failed to parse as tabular ({type(e).__name__}: {e}). Displaying as text:\n{content}")
    except Exception as e_text:
        output_parts.append(f"Could not read file as text either: {e_text}")

# Join all parts and truncate the final output to ensure it's under max_output_chars
final_output = "\n".join(output_parts)
print(final_output[:max_output_chars])