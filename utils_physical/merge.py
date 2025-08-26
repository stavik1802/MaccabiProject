#!/usr/bin/env python3
"""
Merge two games roots into a combined tree where each game folder contains the union of
player files from both sources.

- For each game name (subfolder) present in either --src_a or --src_b:
    combined/<game>/merged_features_*.csv  (union of both sources, conflict policy applied)
    + optional extras via --extra (e.g., poss.csv)

- After merging, prints a summary:
    * total games created
    * for each game: total player files, and count of "starters" (files having chunk==1 and chunk==241)
    * how many games have >=10 starters, and how many have exactly 10
    * lists of qualifying game names

Usage example:
    python3 merge_game_folders.py \
        --src_a /path/to/games_root_A \
        --src_b /path/to/games_root_B \
        --dest  /path/to/games_combined \
        --prefer newer \
        --link \
        --extra "poss.csv"

After this, point your experiment's --games_root to /path/to/games_combined.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# ---------- Starter rule (1-based 15s chunks; minute 60 => chunk 241) ----------

def read_csv_robust(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return None

def has_chunk_1_and_241(csv_path: Path) -> bool:
    df = read_csv_robust(csv_path)
    if df is None or "chunk" not in df.columns:
        return False
    ch = pd.to_numeric(df["chunk"], errors="coerce").dropna().astype(int)
    return bool((ch == 1).any() and (ch == 241).any())

# ---------- File collection & conflict resolution ----------

PLAYER_GLOB = "merged_features_*.csv"

def collect_files(game_dir: Path, patterns: List[str]) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for pat in patterns:
        for p in game_dir.glob(pat):
            if p.is_file():
                found[p.name] = p
    return found

def pick_file(a: Optional[Path], b: Optional[Path], prefer: str) -> Optional[Path]:
    if a and not b:
        return a
    if b and not a:
        return b
    if not a and not b:
        return None
    # both exist
    if prefer == "a":
        return a
    if prefer == "b":
        return b
    if prefer == "newer":
        ta = a.stat().st_mtime
        tb = b.stat().st_mtime
        return a if ta >= tb else b
    if prefer == "bigger":
        sa = a.stat().st_size
        sb = b.stat().st_size
        return a if sa >= sb else b
    # default fallback
    return a

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def place_file(src: Path, dst: Path, link: bool):
    if dst.exists():
        dst.unlink()
    if link:
        try:
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)

# ---------- Merge per game ----------

def merge_one_game(game: str, src_a: Path, src_b: Path, dest_root: Path,
                   prefer: str, link: bool, extra_patterns: List[str]) -> Dict[str, int]:
    a_dir = src_a / game
    b_dir = src_b / game
    out_dir = dest_root / game
    ensure_dir(out_dir)

    a_files = collect_files(a_dir, [PLAYER_GLOB] + extra_patterns) if a_dir.is_dir() else {}
    b_files = collect_files(b_dir, [PLAYER_GLOB] + extra_patterns) if b_dir.is_dir() else {}

    all_names = sorted(set(a_files.keys()) | set(b_files.keys()))
    counts = {"copied": 0, "linked": 0, "players": 0, "extras": 0}

    for name in all_names:
        chosen = pick_file(a_files.get(name), b_files.get(name), prefer)
        if not chosen:
            continue
        dst = out_dir / name
        before = dst.exists()
        place_file(chosen, dst, link)
        if not before:
            if name.startswith("merged_features_") and name.endswith(".csv"):
                counts["players"] += 1
            else:
                counts["extras"] += 1
            counts["linked" if link else "copied"] += 1

    return counts

# ---------- Audit merged games with starter rule ----------

def audit_merged(dest_root: Path, verbose: bool = False):
    summary = []
    games = [d for d in sorted(dest_root.iterdir()) if d.is_dir()]
    for g in games:
        player_files = sorted(g.glob(PLAYER_GLOB))
        total_players = len(player_files)
        starters = 0
        for f in player_files:
            if has_chunk_1_and_241(f):
                starters += 1
        summary.append((g.name, total_players, starters))
        if verbose:
            print(f"{g.name}: players={total_players}, starters(1&241)={starters}")
    return summary

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_a", required=True, help="First games root (folders per game)")
    ap.add_argument("--src_b", required=True, help="Second games root (folders per game)")
    ap.add_argument("--dest",  required=True, help="Output combined games root")
    ap.add_argument("--prefer", choices=["a", "b", "newer", "bigger"], default="newer",
                    help="When the same file exists in both sources, which to keep")
    ap.add_argument("--link", action="store_true",
                    help="Create symlinks instead of copying (falls back to copy if symlink fails)")
    ap.add_argument("--extra", default="", help="Comma-separated extra patterns to include (e.g. 'poss.csv,*.json')")
    ap.add_argument("--verbose", action="store_true", help="Print per-game audit")
    args = ap.parse_args()

    src_a = Path(args.src_a)
    src_b = Path(args.src_b)
    dest  = Path(args.dest)
    ensure_dir(dest)

    extra_patterns = [x.strip() for x in args.extra.split(",") if x.strip()]

    # Union of game folder names
    games_a = {p.name for p in src_a.iterdir() if p.is_dir()} if src_a.is_dir() else set()
    games_b = {p.name for p in src_b.iterdir() if p.is_dir()} if src_b.is_dir() else set()
    all_games = sorted(games_a | games_b)

    print(f"Found {len(all_games)} distinct game folders.")

    merged_stats = {}
    for game in all_games:
        stats = merge_one_game(
            game=game,
            src_a=src_a,
            src_b=src_b,
            dest_root=dest,
            prefer=args.prefer,
            link=args.link,
            extra_patterns=extra_patterns,
        )
        merged_stats[game] = stats

    print("\n=== Merge summary ===")
    total_players = sum(s["players"] for s in merged_stats.values())
    total_extras  = sum(s["extras"] for s in merged_stats.values())
    print(f"Games created: {len(merged_stats)}")
    print(f"Player files placed: {total_players} | Extra files placed: {total_extras}")

    # Audit: how many starters per merged game by chunk rule
    print("\n=== Starter audit (chunk==1 & chunk==241) ===")
    audit = audit_merged(dest, verbose=args.verbose)
    ge_10 = [g for (g, _, starters) in audit if starters >= 10]
    eq_10 = [g for (g, _, starters) in audit if starters == 10]

    for g, tot, st in audit:
        print(f"  {g}: players={tot}, starters(1&241)={st}")

    print("\n=== Selection summary ===")
    print(f"Games with >=10 starters: {len(ge_10)}")
    if ge_10:
        print("  " + ", ".join(ge_10))
    print(f"Games with exactly 10 starters: {len(eq_10)}")
    if eq_10:
        print("  " + ", ".join(eq_10))

if __name__ == "__main__":
    main()
