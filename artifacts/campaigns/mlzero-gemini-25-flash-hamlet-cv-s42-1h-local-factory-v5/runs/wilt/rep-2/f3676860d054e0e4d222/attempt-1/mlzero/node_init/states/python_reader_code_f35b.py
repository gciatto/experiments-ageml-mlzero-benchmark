import pandas as pd
import json
import io

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/wilt/a7fbce33d504-1f460de77002-b8f667a91497-local/cv_contract.json"
output_buffer = io.StringIO()
MAX_OUTPUT_CHARS = 768

try:
    df = pd.read_json(file_path)

    if isinstance(df, pd.DataFrame):
        output_buffer.write("Tabular data detected.\n")
        
        # Display column names
        columns = df.columns.tolist()
        if len(columns) > 20:
            output_buffer.write(f"Columns ({len(columns)}): {columns[:10] + ['...'] + columns[-10:]}\n")
        else:
            output_buffer.write(f"Columns ({len(columns)}): {columns}\n")

        # Display first 3 rows with truncated content
        output_buffer.write("First 3 rows:\n")
        
        # Create a temporary DataFrame for display to apply truncation
        display_df = df.head(3).copy()
        for col in display_df.columns:
            # Ensure content is string before truncation
            display_df[col] = display_df[col].astype(str).apply(lambda x: (x[:47] + '...') if len(x) > 50 else x)
        
        # Capture the string representation of the DataFrame head without index
        df_head_str = display_df.to_string(index=False)
        output_buffer.write(df_head_str)
        output_buffer.write("\n")

    else:
        # If pd.read_json returns a Series or other non-DataFrame object
        raise ValueError("JSON is not directly tabular for pandas DataFrame.")

except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError, json.JSONDecodeError) as e:
    # If pandas fails or JSON is not tabular, treat as a generic text file
    output_buffer.write(f"Could not parse as tabular data ({type(e).__name__}: {e}). Reading as text.\n")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(MAX_OUTPUT_CHARS) # Read up to MAX_OUTPUT_CHARS
            output_buffer.write(content)
            if len(content) == MAX_OUTPUT_CHARS:
                output_buffer.write("...") # Indicate truncation if full read
    except Exception as text_e:
        output_buffer.write(f"Could not read file as text: {text_e}\n")
except Exception as e:
    output_buffer.write(f"An unexpected error occurred: {e}\n")

final_output = output_buffer.getvalue()
print(final_output[:MAX_OUTPUT_CHARS]) # Final truncation to ensure total output is under MAX_OUTPUT_CHARS