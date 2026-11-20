import pandas as pd
import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/madelon/52ab263e132d-d495b2ba732a-b8f667a91497-local/data.csv"
MAX_OUTPUT_LEN = 768
output_parts = []

def append_output(text):
    """Appends text to output_parts, ensuring total length does not exceed MAX_OUTPUT_LEN."""
    global output_parts
    # Calculate current length including potential newlines
    current_len = sum(len(s) for s in output_parts) + len(output_parts) 
    
    # Check if adding the new text (plus its newline) would exceed the limit
    if current_len + len(text) + 1 > MAX_OUTPUT_LEN: 
        remaining_len = MAX_OUTPUT_LEN - current_len - 4 # -4 for "...\n"
        if remaining_len > 0:
            output_parts.append(text[:remaining_len] + "...")
        else:
            output_parts.append("...") # Just indicate truncation
        raise StopIteration # Signal to stop further processing
    output_parts.append(text)

try:
    append_output(f"Analyzing: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)

    append_output(f"Shape: {df.shape}")

    columns = df.columns.tolist()
    col_display_list = []
    if len(columns) > 20:
        col_display_list = columns[:10] + ["..."] + columns[-10:]
    else:
        col_display_list = columns
    
    col_str = f"Columns: {', '.join(map(str, col_display_list))}"
    if len(col_str) > 200: # Truncate column list string if too long
        col_str = col_str[:197] + "..."
    append_output(col_str)

    append_output("\nFirst 3 rows:")
    
    df_head = df.head(3)
    
    # Determine which columns to display for the head table
    cols_to_show_for_head_names = [] # For header row, includes "..."
    actual_cols_for_head_names = [] # For data selection, no "..."
    
    if len(columns) > 6: # If more than 6 columns, show first 3 and last 3
        cols_to_show_for_head_names = columns[:3] + ["..."] + columns[-3:]
        actual_cols_for_head_names = columns[:3] + columns[-3:]
    else:
        cols_to_show_for_head_names = columns
        actual_cols_for_head_names = columns

    # Print header row, limiting column name length
    header_row = "|".join([str(col)[:8].ljust(8) for col in cols_to_show_for_head_names])
    append_output(header_row)
    append_output("-" * len(header_row))

    # Print data rows, limiting cell content length
    for index, row in df_head.iterrows():
        row_values = []
        for col_name in actual_cols_for_head_names:
            val = str(row[col_name])
            row_values.append(val[:8].ljust(8))
        append_output("|".join(row_values))

except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError, UnicodeDecodeError, StopIteration) as e:
    if isinstance(e, StopIteration):
        pass # Output already truncated and added to buffer
    else:
        output_parts = [] # Clear any partial output from CSV attempt
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(MAX_OUTPUT_LEN)
                append_output(f"File could not be parsed as CSV. Displaying first {MAX_OUTPUT_LEN} characters as text:\n{content}")
        except UnicodeDecodeError:
            append_output(f"File is likely binary or non-UTF-8 text. Basic info:\nPath: {file_path}\nSize: {os.path.getsize(file_path)} bytes")
        except FileNotFoundError:
            append_output(f"Error: File not found at {file_path}")
        except Exception as text_e:
            append_output(f"An unexpected error occurred while reading as text: {text_e}")

print("\n".join(output_parts))