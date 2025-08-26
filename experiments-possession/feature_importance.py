#!/usr/bin/env python3
"""
Analyze which physical features most influence possession for your grouped CatBoost model.

Outputs (in --out_dir):
- shap_feature_importance_by_group.csv      (per-feature per-group SHAP)
- shap_feature_importance_aggregated.csv    (base feature aggregated over groups, with direction)
- top_positive_features.txt                 (human-readable summary)
- (optional) permutation_importance.csv     (--do_permutation)

Assumptions:
- Model:     possession_catboost_grouped.pkl
- Features:  catboost_grouped_features.pkl
- Data root: same 5min_windows structure used in training
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

from catboost import Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

# -------------------- Role groups (same as training) --------------------
POS2GROUP = {
    "CB": "G1",
    "CM": "G2", "DM": "G2",
    "RB": "G3", "RW": "G3", "RM": "G3", "LB": "G3", "LW": "G3", "LM": "G3", "UNK": "G3",
    "AM": "G4", "CF": "G4",
}
ALL_GROUPS = ["G1", "G2", "G3", "G4"]

def parse_pos_from_stem(stem: str) -> str:
    try:
        parts = stem.replace("merged_features_", "").split("_")
        return parts[0]
    except Exception:
        return "UNK"

# -------------------- Grouped-feature builder (same logic as train) --------------------
def build_features_grouped(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    players_df: concatenation of all player CSVs for a single game,
    with columns ['minute_start','minute_end','player_id','position','group', <features...>]
    Returns one row per [minute_start, minute_end] with per-group aggregated features + counts.
    """
    key_cols = ["minute_start", "minute_end", "player_id", "position", "group"]
    feature_cols = [c for c in players_df.columns if c not in key_cols]

    rows = []
    for (start, end), window in players_df.groupby(["minute_start", "minute_end"], sort=True):
        row = {"minute_start": start, "minute_end": end}
        # counts per group
        for g in ALL_GROUPS:
            row[f"count_{g}"] = int((window["group"] == g).sum())

        # per-group MEAN for every feature
        for g in ALL_GROUPS:
            sub = window[window["group"] == g]
            if len(sub) == 0:
                for c in feature_cols:
                    row[f"{c}__{g}"] = 0.0
            else:
                means = sub[feature_cols].mean(numeric_only=True)
                for c in feature_cols:
                    row[f"{c}__{g}"] = float(means.get(c, 0.0))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["minute_start", "minute_end"]).reset_index(drop=True)

def load_all_games_grouped(root_folder: str) -> pd.DataFrame:
    """
    root_folder layout:
      /.../5min_windows/
        YYYY-MM-DD/
          merged_features_AM_21.csv
          merged_features_CB_2.csv
          ...
          poss.csv
    """
    root = Path(root_folder)
    all_games = []

    for game in root.iterdir():
        if not game.is_dir():
            continue

        parts = []
        for f in game.glob("merged_features_*.csv"):
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"[!] Failed reading {f}: {e}")
                continue
            stem = f.stem  # e.g., merged_features_AM_21
            pos = parse_pos_from_stem(stem)
            grp = POS2GROUP.get(pos, "UNK")
            df["player_id"] = stem.replace("merged_features_", "")
            df["position"]  = pos
            df["group"]     = grp
            parts.append(df)

        if not parts:
            continue

        players_df = pd.concat(parts, ignore_index=True)

        # minimal cols required
        for col in ("minute_start", "minute_end"):
            if col not in players_df.columns:
                # try to infer from 'minute' or 'block'
                if "block" in players_df.columns:
                    players_df["minute_start"] = (players_df["block"] - 1) * 5
                    players_df["minute_end"]   = players_df["minute_start"] + 5
                else:
                    raise ValueError(f"{game.name}: missing minute_start/minute_end and no 'block' to infer.")

        feats = build_features_grouped(players_df)

        poss_path = game / "poss.csv"
        if not poss_path.exists():
            print(f"[!] Missing poss.csv in {game.name}; skipping game.")
            continue
        poss = pd.read_csv(poss_path)
        # coerce to minutes
        poss["minute_start"] = poss["time_start_sec"] // 60
        poss["minute_end"]   = poss["time_end_sec"]   // 60

        merged = feats.merge(
            poss[["minute_start", "minute_end", "maccabi_haifa_possession_percent"]],
            on=["minute_start", "minute_end"],
            how="inner"
        )
        merged["game"] = game.name
        all_games.append(merged)

    return pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()

# -------------------- SHAP & importance utils --------------------
_FEAT_GROUP_RE = re.compile(r"^(.*)__G([1-4])$")

def split_base_and_group(feat: str) -> Tuple[str, str]:
    """
    Returns (base, group_label). For 'count_G1' returns ('count','G1').
    For 'inst_dist_m_sum__G3' returns ('inst_dist_m_sum', 'G3').
    If no match, returns (feat, 'NA').
    """
    if feat.startswith("count_G"):
        return ("count", feat.split("_", 1)[1])  # ('count', 'G1'..)
    m = _FEAT_GROUP_RE.match(feat)
    if m:
        return (m.group(1), f"G{m.group(2)}")
    return (feat, "NA")

def compute_shap_importance(model, X: pd.DataFrame, sample_size: int = 40000) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      feature, base_feature, group, mean_abs_shap, mean_signed_shap, corr_sign
    corr_sign is the sign of Pearson correlation between feature values and SHAP values
    (rough direction proxy: + means higher feature → higher possession).
    """
    # sample to keep memory reasonable
    if len(X) > sample_size:
        Xs = X.sample(sample_size, random_state=42)
    else:
        Xs = X

    pool = Pool(Xs)
    shap_vals = model.get_feature_importance(data=pool, type="ShapValues")  # shape: [N, M+1]
    shap_vals = np.array(shap_vals)
    # Drop the last column (expected_value)
    shap_phi = shap_vals[:, :-1]  # [N, M]
    features = list(Xs.columns)

    rows = []
    for j, feat in enumerate(features):
        phi = shap_phi[:, j]
        xj = Xs.iloc[:, j].to_numpy()
        mean_abs = float(np.mean(np.abs(phi)))
        mean_signed = float(np.mean(phi))
        # correlation for direction (robust to zero variance)
        corr = 0.0
        std_x = float(np.std(xj))
        std_phi = float(np.std(phi))
        if std_x > 1e-12 and std_phi > 1e-12:
            corr = float(np.corrcoef(xj, phi)[0, 1])

        base, grp = split_base_and_group(feat)
        rows.append({
            "feature": feat,
            "base_feature": base,
            "group": grp,
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
            "corr_sign": np.sign(corr) if abs(corr) >= 0.05 else 0.0  # threshold tiny correlations → 0
        })
    return pd.DataFrame(rows)

def aggregate_by_base_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SHAP metrics across G1..G4 to base_feature level.
    - total_importance = sum of mean_abs_shap over groups
    - net_signed = sum of mean_signed_shap (direction proxy)
    - direction = 'positive' if net_signed>0, 'negative' if <0, 'neutral' else
    """
    agg = df.groupby("base_feature", as_index=False).agg(
        total_importance=("mean_abs_shap", "sum"),
        net_signed=("mean_signed_shap", "sum"),
    )
    def _dir(x):
        if x >  1e-6: return "positive"
        if x < -1e-6: return "negative"
        return "neutral"
    agg["direction"] = agg["net_signed"].apply(_dir)
    # also pivot per-group importance for visibility
    pivot = df.pivot_table(index="base_feature", columns="group", values="mean_abs_shap", aggfunc="sum").reset_index()
    out = agg.merge(pivot, on="base_feature", how="left").sort_values("total_importance", ascending=False)
    return out

# -------------------- Main CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Path to 5min_windows (same root used for training)")
    ap.add_argument("--model_path", default="possession_catboost_grouped.pkl")
    ap.add_argument("--features_path", default="catboost_grouped_features.pkl")
    ap.add_argument("--out_dir", default="feature_importance_out")
    ap.add_argument("--sample_size", type=int, default=40000, help="Rows to sample for SHAP calculation")
    ap.add_argument("--do_permutation", action="store_true", help="Also compute permutation importance on a holdout set")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("📦 Loading model & features…")
    model = joblib.load(args.model_path)
    feature_names: List[str] = joblib.load(args.features_path)

    print("📄 Building grouped dataset…")
    data = load_all_games_grouped(args.data_root)
    if data.empty:
        raise SystemExit("No data found. Check --data_root.")

    y = data["maccabi_haifa_possession_percent"].values.astype(float)
    X = data.drop(columns=["minute_start", "minute_end", "game", "maccabi_haifa_possession_percent"])

    # Reorder / fill to match model features exactly
    missing = [c for c in feature_names if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    X = X[feature_names]

    print(f"🧮 SHAP on up to {args.sample_size} rows …")
    shap_df = compute_shap_importance(model, X, sample_size=args.sample_size)
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(out_dir / "shap_feature_importance_by_group.csv", index=False)
    print(f"✅ wrote {out_dir / 'shap_feature_importance_by_group.csv'}")

    agg_df = aggregate_by_base_feature(shap_df)
    agg_df.to_csv(out_dir / "shap_feature_importance_aggregated.csv", index=False)
    print(f"✅ wrote {out_dir / 'shap_feature_importance_aggregated.csv'}")

    # quick text summary of positives
    top_pos = agg_df[agg_df["direction"] == "positive"].head(25)
    with open(out_dir / "top_positive_features.txt", "w") as f:
        f.write("Top base physical features whose increase tends to RAISE predicted possession (by SHAP):\n")
        for _, r in top_pos.iterrows():
            g_parts = [g for g in ["G1","G2","G3","G4"] if g in agg_df.columns and not pd.isna(r.get(g, np.nan))]
            g_txt = ", ".join([f"{g}:{r.get(g):.4f}" for g in g_parts])
            f.write(f"- {r['base_feature']}: total_importance={r['total_importance']:.4f}  [{g_txt}]\n")
    print(f"✅ wrote {out_dir / 'top_positive_features.txt'}")

    # (Optional) permutation importance on a holdout set for confirmation
    if args.do_permutation:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        base_r2 = r2_score(y_test, model.predict(X_test))
        print(f"🔎 Holdout R^2 baseline: {base_r2:.4f}")

        print("⏱️ Permutation importance (this can take a while)…")
        perm = permutation_importance(
            model, X_test, y_test,
            n_repeats=5, random_state=42, scoring="neg_mean_squared_error"
        )
        perm_df = pd.DataFrame({
            "feature": X_test.columns,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std":  perm.importances_std
        }).sort_values("perm_importance_mean", ascending=False)

        # aggregate by base feature (sum) to match SHAP aggregation
        perm_df["base_feature"] = [split_base_and_group(f)[0] for f in perm_df["feature"]]
        perm_agg = perm_df.groupby("base_feature", as_index=False)["perm_importance_mean"].sum() \
                          .sort_values("perm_importance_mean", ascending=False)
        perm_agg.to_csv(out_dir / "permutation_importance.csv", index=False)
        print(f"✅ wrote {out_dir / 'permutation_importance.csv'}")

    print("\nDone. Inspect CSVs for:")
    print(f"- per-group effects: {out_dir/'shap_feature_importance_by_group.csv'}")
    print(f"- base-feature ranking + direction: {out_dir/'shap_feature_importance_aggregated.csv'}")
    print(f"- human summary: {out_dir/'top_positive_features.txt'}")
    if args.do_permutation:
        print(f"- permutation check: {out_dir/'permutation_importance.csv'}")


if __name__ == "__main__":
    main()
