import json
import pandas as pd
import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/ames_housing/06b4acbf769d-ebb2a7fe4397-b8f667a91497-local/cv_contract.json"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        data = json.loads(content)
        
        df_created = False
        df = None

        # Attempt to create a pandas DataFrame if data is a list of dictionaries or a single dictionary
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            try:
                df = pd.DataFrame(data)
                df_created = True
            except Exception:
                pass # Fallback to raw JSON dump if DataFrame creation fails
        elif isinstance(data, dict):
            try:
                df = pd.DataFrame([data]) # Treat a single dict as a single row
                df_created = True
            except Exception:
                pass # Fallback to raw JSON dump if DataFrame creation fails

        if df_created:
            col_names = df.columns.tolist()
            col_display_str = f"Columns: {', '.join(col_names[:10])}"
            if len(col_names) > 10:
                col_display_str += ", ..."
            
            rows_str = df.head(2).to_string(max_colwidth=50, index=False)
            
            # Check if the combined output (columns + rows) exceeds the character limit
            # Add 1 for the newline character between column display and rows
            if len(col_display_str) + len(rows_str) + 1 < 760: # Leave some buffer for safety
                print(col_display_str)
                print(rows_str)
            else:
                # Fallback to truncated raw JSON if tabular output is too long
                json_str = json.dumps(data, indent=2)
                print(json_str[:768])
                if len(json_str) > 768:
                    print("...")
        else:
            # Not a list of dicts, or single dict, or DataFrame creation failed
            json_str = json.dumps(data, indent=2)
            print(json_str[:768])
            if len(json_str) > 768:
                print("...")
                
    except json.JSONDecodeError:
        # Not a valid JSON, treat as a plain text file
        print(content[:768])
        if len(content) > 768:
            print("...")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    # Catch any other unexpected errors during file reading or processing
    print(f"An unexpected error occurred: {e}")