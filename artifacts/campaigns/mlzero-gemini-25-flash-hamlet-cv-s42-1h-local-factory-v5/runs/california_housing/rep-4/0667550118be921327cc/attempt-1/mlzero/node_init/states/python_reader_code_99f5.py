import pandas as pd
import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/california_housing/45e44ec0cb85-d3be51c254a5-b8f667a91497-local/data.csv"

output_buffer = []

def append_output(s):
    """Appends string to buffer, ensuring it's a string."""
    output_buffer.append(str(s))

try:
    # Attempt to read as a tabular file (CSV)
    df = pd.read_csv(file_path)

    columns = df.columns.tolist()
    
    # Display column names
    append_output("Columns:")
    if len(columns) > 20:
        # Display first 10, last 10, and indicate truncation
        cols_to_print = columns[:10] + ['...'] + columns[-10:]
        append_output(cols_to_print)
    else:
        append_output(columns)

    # Show first 2-3 rows with truncated cell content (50 chars), no index
    append_output("\nFirst 3 rows:")
    
    # Use to_string for controlled output:
    # max_rows=3 ensures only the first 3 rows are displayed.
    # max_colwidth=50 truncates individual cell content to 50 characters.
    # index=False prevents the pandas index from being printed.
    df_string = df.head(3).to_string(max_rows=3, max_colwidth=50, index=False)
    append_output(df_string)

except Exception as e:
    # If failed to open as tabular, treat as text file
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768) # Read up to 768 characters
            append_output("\nFirst few lines (up to 768 characters):")
            append_output(content)
    except Exception as text_e:
        # Fallback for binary or unreadable files
        append_output(f"File '{os.path.basename(file_path)}' is likely binary or unreadable.")
        try:
            file_size = os.path.getsize(file_path)
            append_output(f"File size: {file_size} bytes")
        except OSError:
            append_output("Could not determine file size.")

# Join all parts and truncate if necessary to meet the 768 character limit
final_output = "\n".join(output_buffer)
if len(final_output) > 768:
    # Truncate the final string, ensuring '...' is added if truncated
    print(final_output[:765] + '...')
else:
    print(final_output)