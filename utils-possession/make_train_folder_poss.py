#!/usr/bin/env python3
"""
Make a train set = ALL_GAMES minus EVAL_GAMES.

Usage:
  python make_train_from_eval.py \
    --all_games_root /path/to/all_games \
    --eval_root /path/to/eval_games \
    --out_root /path/to/train_out \
    [--mode symlink|copy] [--overwrite] [--dry_run]

- --mode symlink (default): creates directory symlinks to source game folders
- --mode copy: copies full folders
- --overwrite: remove existing targets in out_root before creating
- --dry_run: show what would happen, do not write anything
"""

import argparse, os, shutil, sys
from pathlib import Path

def list_subdirs(root: Path) -> set[str]:
    return {p.name for p in root.iterdir() if p.is_dir()}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def create_symlink(src: Path, dst: Path, overwrite: bool):
    if dst.exists() or dst.is_symlink():
        if overwrite:
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        else:
            print(f"  • skip (exists): {dst}")
            return
    dst.symlink_to(src, target_is_directory=True)
    print(f"  ✓ linked {dst.name}")

def copy_dir(src: Path, dst: Path, overwrite: bool):
    if dst.exists():
        if overwrite:
            shutil.rmtree(dst)
        else:
            print(f"  • skip (exists): {dst}")
            return
    shutil.copytree(src, dst)
    print(f"  ✓ copied {dst.name}")

def main():
    ap = argparse.ArgumentParser(description="Create train folder = all_games - eval_games")
    ap.add_argument("--all_games_root", required=True, type=Path)
    ap.add_argument("--eval_root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--mode", choices=["symlink","copy"], default="symlink")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if not args.all_games_root.is_dir():
        sys.exit(f"❌ not a directory: {args.all_games_root}")
    if not args.eval_root.is_dir():
        sys.exit(f"❌ not a directory: {args.eval_root}")

    all_games = list_subdirs(args.all_games_root)
    eval_games = list_subdirs(args.eval_root)

    train_games = sorted(all_games - eval_games)
    missing_in_all = sorted(eval_games - all_games)

    print(f"Found {len(all_games)} games in ALL, {len(eval_games)} in EVAL.")
    print(f"→ Selecting {len(train_games)} games for TRAIN (excluded {len(eval_games)} eval games).")
    if missing_in_all:
        print(f"⚠️ {len(missing_in_all)} eval folders not found in ALL: {', '.join(missing_in_all[:10])}"
              + (" ..." if len(missing_in_all) > 10 else ""))

    if args.dry_run:
        print("\n--dry_run mode: nothing will be written. Planned actions:")
        for g in train_games:
            print(f"  {args.mode}: {args.all_games_root/g}  ->  {args.out_root/g}")
        return

    ensure_dir(args.out_root)

    # Save a manifest for reproducibility
    manifest = args.out_root / "train_manifest.txt"
    with manifest.open("w", encoding="utf-8") as f:
        f.write(f"# Train manifest generated from:\n")
        f.write(f"# all_games_root = {args.all_games_root}\n")
        f.write(f"# eval_root      = {args.eval_root}\n")
        f.write(f"# mode={args.mode}, overwrite={args.overwrite}\n\n")
        for g in train_games:
            f.write(f"{g}\n")

    print(f"\nWriting to {args.out_root} (mode={args.mode}, overwrite={args.overwrite})")
    for g in train_games:
        src = args.all_games_root / g
        dst = args.out_root / g
        if args.mode == "symlink":
            create_symlink(src, dst, args.overwrite)
        else:
            copy_dir(src, dst, args.overwrite)

    print("\n✅ Done.")
    print(f"• Train games: {len(train_games)} → {args.out_root}")
    print(f"• Manifest: {manifest}")

if __name__ == "__main__":
    main()
