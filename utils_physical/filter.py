"""
Script: filter.py

Description:
    This script provides basic filtering functionality for player tracking data.
    It implements fundamental filtering operations and data cleaning methods
    to prepare raw tracking data for further analysis.

Input:
    - Raw tracking data files
    - Basic filtering parameters
    - Data validation criteria
    - Configuration settings

Output:
    - Filtered tracking data
    - Basic quality metrics
    - Validation reports
    - Filtered data statistics

Usage:
    Run this script to apply basic filtering to tracking data files
"""

#!/usr/bin/env python3
import argparse


def filter_csv(in_path: str, out_path: str, skip_ranges: list[tuple[int, int]]):
    """
    Copy lines from in_path to out_path, omitting any line
    whose 1-based index lies within one of skip_ranges.
    """
    # ensure ranges are sorted (not strictly necessary, but a good habit)
    skip_ranges = sorted(skip_ranges, key=lambda r: r[0])

    with open(in_path, 'r', encoding='utf-8', newline='') as fin, \
            open(out_path, 'w', encoding='utf-8', newline='') as fout:

        for lineno, line in enumerate(fin, start=1):
            # check if lineno is in any skip range
            if any(start <= lineno <= end for start, end in skip_ranges):
                continue
            fout.write(line)


def main():
    parser = argparse.ArgumentParser(
        description="Filter out specified line ranges from a CSV."
    )
    parser.add_argument("input_csv", help="Path to the original CSV file")
    parser.add_argument("output_csv", help="Path to write the filtered CSV")
    parser.add_argument(
        "--skip", "-s", nargs=2, action='append', type=int, metavar=('START', 'END'),
        help="Line range to skip (inclusive). Can be used multiple times."
    )

    args = parser.parse_args()

    if not args.skip:
        parser.error("You must specify at least one --skip START END range")

    # convert list of [START, END] to list of tuples
    skip_ranges = [(start, end) for start, end in args.skip]

    filter_csv(args.input_csv, args.output_csv, skip_ranges)
    print(f"Filtered out {len(skip_ranges)} ranges; output written to {args.output_csv}")


if __name__ == "__main__":
    main()

