import pandas as pd
import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/mnist_784/30c9e1582974-dda30b451a9f-b8f667a91497-local/data.csv"

try:
    df = pd.read_csv(file_path)

    columns = df.columns.tolist()
    
    # Display column names: first 10, last 10 if > 20, otherwise all
    cols_for_name_display = []
    if len(columns) > 20:
        cols_for_name_display = columns[:10] + ['...'] + columns[-10:]
    else:
        cols_for_name_display = columns
    
    # Construct the column names string
    column_names_str = f"Columns: {', '.join(map(str, cols_for_name_display))}"
    print(column_names_str)

    # Show first 2-3 rows with truncated cell content (50 chars)
    # To keep total output under 768 chars, limit the number of columns shown in head
    max_cols_for_head_display = 3 # Display only the first 3 columns for head rows
    cols_to_show_in_head = columns[:max_cols_for_head_display]

    def truncate_cell(content):
        s = str(content)
        return s[:50] + '...' if len(s) > 50 else s

    df_head = df[cols_to_show_in_head].head(3)
    
    # Manually format the head output for strict character control
    head_output_lines = []
    
    # Header line for the displayed head columns
    if not df_head.empty:
        header_line = ', '.join([truncate_cell(col) for col in df_head.columns])
        head_output_lines.append(header_line)

        # Data rows
        for _, row in df_head.iterrows():
            row_values = [truncate_cell(val) for val in row]
            head_output_lines.append(', '.join(row_values))
    
    final_head_output = "\n".join(head_output_lines)

    # Check total output length and truncate if necessary
    # Account for the newline character after column_names_str
    current_output_len = len(column_names_str) + 1 + len(final_head_output)
    
    if current_output_len > 768:
        remaining_chars_for_head = 768 - len(column_names_str) - 1 - 5 # -1 for newline, -5 for '...'
        if remaining_chars_for_head > 0:
            final_head_output = final_head_output[:remaining_chars_for_head] + "..."
        else:
            final_head_output = "..." # Fallback if no space left for head output
    
    print(final_head_output)

except Exception as e:
    # Fallback for text file or other errors (e.g., FileNotFoundError, pandas.errors.ParserError)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            print(content)
    except Exception as text_e:
        # If even text reading fails, provide basic info (error message only for brevity)
        print(f"Could not read file: {text_e}")