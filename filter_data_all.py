import pandas as pd
from datetime import datetime
from pathlib import Path

def parse_time(s):
    try:
        return datetime.strptime(s.strip(), "%H:%M:%S.%f").time()
    except ValueError:
        raise ValueError(f"Invalid time '{s}' – must be HH:MM:SS.fff (e.g., 18:58:00.0)")

def prompt_time_range(label):
    while True:
        try:
            start = input(f"{label} START (HH:MM:SS.fff): ").strip()
            end = input(f"{label} END   (HH:MM:SS.fff): ").strip()
            return parse_time(start), parse_time(end)
        except ValueError as e:
            print(f"❌ {e} Try again.\n")

def filter_by_time(df, start, end):
    return df[(df['Time'] >= start) & (df['Time'] <= end)]

def process_file(file_path, first_half, second_half):
    print(f"📄 Processing: {file_path.name}")
    df = pd.read_csv(file_path)
    df['Time'] = df['Time'].apply(parse_time)

    df_first = filter_by_time(df, *first_half)
    df_second = filter_by_time(df, *second_half)

    # Prepare subfolder
    base_name = file_path.stem
    parent_folder = file_path.parent
    player_folder = parent_folder / base_name
    player_folder.mkdir(parents=True, exist_ok=True)

    # Save filtered CSVs
    df_first.to_csv(player_folder / "first_half.csv", index=False)
    df_second.to_csv(player_folder / "second_half.csv", index=False)

    print(f"✔ Saved in {player_folder}/: first_half.csv, second_half.csv\n")

def main(folder_path):
    folder = Path(folder_path)
    csv_files = sorted(folder.glob("*.csv"))

    if not csv_files:
        print("❌ No CSV files found.")
        return

    print("📋 Enter time ranges (used for ALL files):\n")
    first_half = prompt_time_range("FIRST HALF")
    second_half = prompt_time_range("SECOND HALF")

    for file in csv_files:
        try:
            process_file(file, first_half, second_half)
        except Exception as e:
            print(f"❌ Error with {file.name}: {e}")

# Run from command line
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python split_csv_by_time.py <folder_path>")
    else:
        main(sys.argv[1])


# import pandas as pd
# from datetime import datetime
# from pathlib import Path

# def parse_time(s):
#     try:
#         return datetime.strptime(s.strip(), "%H:%M:%S.%f").time()
#     except ValueError:
#         raise ValueError(f"Invalid time '{s}' – must be HH:MM:SS.fff (e.g., 18:58:00.0)")

# def prompt_time_range(label):
#     while True:
#         try:
#             start = input(f"{label} START (HH:MM:SS.fff): ").strip()
#             end = input(f"{label} END   (HH:MM:SS.fff): ").strip()
#             return parse_time(start), parse_time(end)
#         except ValueError as e:
#             print(f"❌ {e} Try again.\n")

# def filter_by_time(df, start, end):
#     return df[(df['Time'] >= start) & (df['Time'] <= end)]

# def process_file(file_path, full1, full2, first_half, second_half):
#     print(f"📄 Processing: {file_path.name}")
#     df = pd.read_csv(file_path)
#     df['Time'] = df['Time'].apply(lambda s: parse_time(s))

#     # df_full = pd.concat([
#     #     filter_by_time(df, *full1),
#     #     filter_by_time(df, *full2)
#     # ])
#     df_first = filter_by_time(df, *first_half)
#     df_second = filter_by_time(df, *second_half)

#     base_name = file_path.stem
#     folder = file_path.parent
#     # df_full.to_csv(folder / f"{base_name}_full.csv", index=False)
#     df_first.to_csv(folder / f"{base_name}_first_half.csv", index=False)
#     df_second.to_csv(folder / f"{base_name}_second_half.csv", index=False)

#     print(f"✔ Saved: {base_name}_full.csv, _first_half.csv, _second_half.csv\n")

# def main(folder_path):
#     folder = Path(folder_path)
#     csv_files = sorted(folder.glob("*.csv"))

#     if not csv_files:
#         print("❌ No CSV files found.")
#         return

#     print("📋 Enter time ranges (used for ALL files):\n")
#     full1 = prompt_time_range("FULL RANGE 1 (before halftime)")
#     full2 = prompt_time_range("FULL RANGE 2 (after halftime)")
#     first_half = prompt_time_range("FIRST HALF")
#     second_half = prompt_time_range("SECOND HALF")

#     for file in csv_files:
#         try:
#             process_file(file, full1, full2, first_half, second_half)
#         except Exception as e:
#             print(f"❌ Error with {file.name}: {e}")

# # Run
# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python split_csv_by_time.py <folder_path>")
#     else:
#         main(sys.argv[1])