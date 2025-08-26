#!/usr/bin/env python3
"""
Scale 'dist' columns in CSVs by a factor (default 1.5).

Usage:
  python3 scale_dist_cols.py /path/to/avg_folder /path/to/output_folder
  # optional:
  # --factor 1.5         change multiplier (default 1.5)
  # --pattern "*.csv"    which files to process (default *.csv)
  # --regex "dist"       which columns to scale (case-insensitive regex)
  # --dry-run            print what would change without writing files

Notes:
- Only numeric columns whose name matches the regex (default: contains 'dist') are scaled.
- Columns like chunk/block/minute_start/minute_end remain untouched (they don't match 'dist').
"""

import argparse
import re
from pathlib import Path
import pandas as pd
from pandas.api.types import is_numeric_dtype

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path, help="Folder with avg players stats CSVs")
    ap.add_argument("output_dir", type=Path, help="Folder to write scaled CSVs")
    ap.add_argument("--factor", type=float, default=150, help="Scale factor for matching columns")
    ap.add_argument("--pattern", default="*.csv", help="Glob for files to process (default: *.csv)")
    ap.add_argument("--regex", default=r"dist", help="Case-insensitive regex to match columns (default: 'dist')")
    ap.add_argument("--dry-run", action="store_true", help="Print actions but don't write files")
    args = ap.parse_args()

    in_dir: Path = args.input_dir
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rx = re.compile(args.regex, re.IGNORECASE)

    total_files = 0
    changed_files = 0
    total_cols_changed = 0

    for fp in sorted(in_dir.rglob(args.pattern)):
        if not fp.is_file():
            continue
        total_files += 1
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"⚠️  Skip {fp.name}: read error: {e}")
            continue

        # Which columns to scale?
        scale_cols = [c for c in df.columns if rx.search(c) and is_numeric_dtype(df[c])]
        if not scale_cols:
            print(f"→ {fp.name}: no matching 'dist' numeric columns.")
            # still copy file so output folder mirrors input
            if not args.dry_run:
                df.to_csv(out_dir / fp.name, index=False)
            continue

        # Scale
        if not args.dry_run:
            df[scale_cols] = df[scale_cols] * args.factor
            df.to_csv(out_dir / fp.name, index=False)

        changed_files += 1
        total_cols_changed += len(scale_cols)
        print(f"✓ {fp.name}: scaled {len(scale_cols)} column(s): {', '.join(scale_cols)}")

    print(f"\nDone. Files scanned: {total_files}, changed: {changed_files}, cols scaled total: {total_cols_changed}")
    print(f"Output: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
