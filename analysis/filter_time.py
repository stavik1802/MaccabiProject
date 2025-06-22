#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime, time

def parse_time_str(s: str) -> time:
    return datetime.strptime(s, "%H:%M:%S.%f").time()

def is_time_in_ranges(t: time, ranges: list[tuple[time, time]]) -> bool:
    return any(start <= t <= end for start, end in ranges)

def filter_csv_by_time_ranges(in_path: str, out_path: str, keep_ranges: list[tuple[time, time]]):
    with open(in_path, 'r', encoding='utf-8', newline='') as fin, \
         open(out_path, 'w', encoding='utf-8', newline='') as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            try:
                t = parse_time_str(row['Time'])
                if is_time_in_ranges(t, keep_ranges):
                    writer.writerow(row)
            except (ValueError, KeyError):
                continue  # skip malformed or missing Time rows

def main():
    parser = argparse.ArgumentParser(description="Filter CSV rows by multiple time ranges (HH:MM:SS.sss).")
    parser.add_argument("input_csv", help="Input CSV file")
    parser.add_argument("output_csv", help="Output CSV file")
    parser.add_argument("--keep", "-k", nargs=2, action='append', metavar=('START', 'END'),
                        required=True, help="Time range to keep, format: HH:MM:SS.sss HH:MM:SS.sss (can be used multiple times)")

    args = parser.parse_args()
    keep_ranges = [(parse_time_str(start), parse_time_str(end)) for start, end in args.keep]

    filter_csv_by_time_ranges(args.input_csv, args.output_csv, keep_ranges)
    print(f"Filtered rows within {len(keep_ranges)} time range(s); output saved to {args.output_csv}")

if __name__ == "__main__":
    main()
