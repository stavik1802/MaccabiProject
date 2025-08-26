#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import pandas as pd

MINUTE_COL_CANDIDATES = ("minute_start", "minuteStart", "MinuteStart")

def read_csv_robust(path: Path) -> pd.DataFrame | None:
    try:
        # First try: C engine (supports low_memory)
        return pd.read_csv(path, low_memory=False)
    except Exception as e1:
        try:
            # Fallback: Python engine (no low_memory allowed)
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception as e2:
            print(f"  [!] Failed to read {path.name}: {e2}")
            return None

def find_minute_col(df: pd.DataFrame) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in MINUTE_COL_CANDIDATES:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def has_0_and_60(csv_path: Path) -> bool:
    df = read_csv_robust(csv_path)
    if df is None:
        return False
    col = find_minute_col(df)
    if col is None:
        print(f"  [!] No minute_start column in {csv_path.name}")
        return False
    # Coerce to numeric & check membership
    m = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
    vals = set(m.values.tolist())
    return (0 in vals) and (60 in vals)

def main():
    ap = argparse.ArgumentParser(description="List games with >=10 players having minute_start 0 and 60.")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--require_exact_10", action="store_true")
    args = ap.parse_args()

    game_pat = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    games_ok = []
    rows = []

    for game_dir in sorted(p for p in args.root.iterdir() if p.is_dir() and game_pat.match(p.name)):
        merged_files = sorted(game_dir.glob("merged_features_*.csv"))
        if not merged_files:
            continue

        print(f"\nScanning {game_dir.name} ({len(merged_files)} player files)")
        ok_players, missing_players = [], []

        for f in merged_files:
            if has_0_and_60(f):
                ok_players.append(f.name)
            else:
                missing_players.append(f.name)

        count_ok = len(ok_players)
        meets = (count_ok == 10) if args.require_exact_10 else (count_ok >= 10)
        if meets:
            games_ok.append(game_dir.name)

        rows.append({
            "game": game_dir.name,
            "total_player_files": len(merged_files),
            "players_with_0_and_60": count_ok,
            "meets_criterion": meets,
            "example_missing_players": ";".join(missing_players[:5]),
        })

    print("\n=== Games meeting criterion ===")
    for g in games_ok:
        print(g)
    print(f"\nTotal: {len(games_ok)} games")

    if args.out:
        df = pd.DataFrame(rows).sort_values(["meets_criterion", "game"], ascending=[False, True])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nWrote summary to: {args.out}")

if __name__ == "__main__":
    main()
