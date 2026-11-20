import pandas as pd

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/ames_housing/06b4acbf769d-ebb2a7fe4397-b8f667a91497-local/data.csv"

try:
    # Attempt to read the file as a CSV
    df = pd.read_csv(file_path)

    output_parts = []

    # Display column names
    columns = df.columns.tolist()
    if len(columns) > 20:
        output_parts.append("Columns (first 10 and last 10):")
        output_parts.append(str(columns[:10] + ['...'] + columns[-10:]))
    else:
        output_parts.append("Columns:")
        output_parts.append(str(columns))

    # Show first row with truncated cell content, limiting to first 7 columns
    # This is done to keep the total output under 768 characters,
    # as showing 2-3 rows with 50-char truncation for many columns would exceed the limit.
    output_parts.append("\nFirst row (truncated, first 7 columns):")
    
    with pd.option_context('display.max_colwidth', 50, 'display.width', 1000):
        head_df = df.iloc[:1, :7] # Select first row and up to first 7 columns
        output_parts.append(head_df.to_string(index=False))

    final_output = "\n".join(output_parts)
    print(final_output[:768]) # Truncate final output to ensure it's under 768 characters

except Exception:
    # If reading as CSV fails, treat it as a text file
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            print("File content (first 768 characters):")
            print(content)
    except Exception as text_e:
        print(f"Could not read file: {text_e}")