import pandas as pd
import os
import io

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/adult/547551a449e2-88ada5f9c32f-b8f667a91497-local/cv_contract.json"
MAX_OUTPUT_CHARS = 768

output_buffer = io.StringIO()

try:
    # Attempt to read the file as a pandas DataFrame (tabular data)
    df = pd.read_json(file_path)

    # Display column names
    output_buffer.write("Column Names:\n")
    if len(df.columns) > 20:
        output_buffer.write(str(list(df.columns[:10])) + "\n")
        output_buffer.write("...\n")
        output_buffer.write(str(list(df.columns[-10:])) + "\n")
    else:
        output_buffer.write(str(list(df.columns)) + "\n")

    # Display first 2 rows with truncated cell content
    output_buffer.write("\nFirst 2 Rows:\n")
    with pd.option_context('display.max_colwidth', 50):
        output_buffer.write(df.head(2).to_string(index=False))

except Exception as e:
    # If reading as tabular data fails, treat it as a text file
    output_buffer.write(f"Could not read as tabular data ({e}). Reading as text file.\n")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(MAX_OUTPUT_CHARS)
            output_buffer.write(content)
    except Exception as text_e:
        # If reading as text also fails, provide basic file information
        output_buffer.write(f"Could not read as text file either ({text_e}). Providing basic info.\n")
        output_buffer.write(f"File: {file_path}\n")
        output_buffer.write(f"Size: {os.path.getsize(file_path)} bytes\n")

final_output = output_buffer.getvalue()
if len(final_output) > MAX_OUTPUT_CHARS:
    print(final_output[:MAX_OUTPUT_CHARS - 3] + "...")
else:
    print(final_output)