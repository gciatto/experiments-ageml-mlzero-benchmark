import pandas as pd
import json
import os

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/texture/c3e1db050fee-1e668d3f7c50-b8f667a91497-local/cv_contract.json"

def analyze_file(file_path):
    try:
        # 1. Attempt to read as a tabular DataFrame using pandas
        df = pd.read_json(file_path)

        if isinstance(df, pd.DataFrame) and not df.empty:
            # Display column names
            columns = df.columns.tolist()
            if len(columns) > 20:
                print("Columns:", columns[:10] + ['...'] + columns[-10:])
            else:
                print("Columns:", columns)

            # Show first 2-3 rows with truncated cell content (50 chars)
            df_display = df.head(3).copy()
            for col in df_display.columns:
                df_display[col] = df_display[col].astype(str).apply(lambda x: (x[:50] + '...') if len(x) > 50 else x)

            print("\nFirst 3 rows:")
            # Do not show additional index column
            print(df_display.to_string(index=False))
            return

    except (pd.errors.EmptyDataError, ValueError, TypeError, json.JSONDecodeError, pd.io.common.EmptyDataError) as e:
        # pandas failed to read as tabular or JSON was malformed for pandas.
        # Fall through to generic JSON/text handling.
        pass
    except Exception as e:
        # Catch any other unexpected errors during pandas processing (e.g., file not found, permissions)
        # Fall through to generic JSON/text handling, which will also handle file access errors.
        pass

    # 2. If not handled by pandas, try to read as generic JSON
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            json_str = json.dumps(data, indent=2)
            print(f"JSON content (truncated to 768 chars):\n{json_str[:768]}")
            return
    except json.JSONDecodeError:
        # Not valid JSON, fall through to plain text
        pass
    except Exception as e:
        # File access error during JSON load (e.g., file not found, permissions)
        # Fall through to plain text, which will also handle file access errors.
        pass

    # 3. If not handled as tabular or generic JSON, treat as plain text
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(768)
            print(f"Displaying first 768 characters as text:\n{content}")
            return
    except Exception as e:
        # Final fallback: file is likely binary or inaccessible
        print(f"Failed to read file: {e}")
        if os.path.exists(file_path):
            print(f"File is likely binary or inaccessible. Basic info: {os.path.basename(file_path)}, Size: {os.path.getsize(file_path)} bytes")
        else:
            print(f"File not found: {file_path}")

analyze_file(file_path)