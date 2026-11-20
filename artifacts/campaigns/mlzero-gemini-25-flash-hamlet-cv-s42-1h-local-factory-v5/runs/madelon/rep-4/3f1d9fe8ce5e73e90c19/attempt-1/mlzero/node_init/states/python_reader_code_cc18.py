import pandas as pd
import json
from io import StringIO

file_path = "/root/experiments-ageml-mlzero-benchmark/artifacts/datasets-cv/madelon/52ab263e132d-d495b2ba732a-b8f667a91497-local/cv_contract.json"

try:
    df = pd.read_json(file_path)

    if isinstance(df, pd.DataFrame) and not df.empty:
        output_lines = []

        columns = df.columns.tolist()
        if len(columns) > 20:
            cols_display = columns[:10] + ["..."] + columns[-10:]
        else:
            cols_display = columns
        output_lines.append(f"Columns: {', '.join(map(str, cols_display))}")

        buffer = StringIO()
        df.head(3).to_string(buffer, max_colwidth=50, index=False)
        output_lines.append(buffer.getvalue())
        
        final_output = "\n".join(output_lines)
        print(final_output[:768])

    else:
        raise ValueError("Pandas could not interpret JSON as a useful tabular DataFrame or it's empty.")

except Exception as e:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(768)
            print(content)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e_text:
        print(f"Error reading file as text: {e_text}")