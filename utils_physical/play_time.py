import pandas as pd
import sys
from pathlib import Path
def filter_maccabi_haifa(input_excel, output_csv=None):
    df = pd.read_excel(input_excel)
    
    # Filter rows where Team == 'Maccabi Haifa'
    filtered_df = df[df['Team'] == 'Maccabi Haifa']

    # Determine output path
    if output_csv is None:
        output_csv = Path(input_excel).with_suffix('.filtered.csv')

    # Save to CSV
    filtered_df.to_csv(output_csv, index=False)
    print(f"Saved filtered data to: {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_maccabi.py <input_excel_file>")
    else:
        input_file = sys.argv[1]
        filter_maccabi_haifa(input_file)

