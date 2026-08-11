"""Fuse graph evidence and link-prediction scores into ranked candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def normalize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    minimum, maximum = float(numeric.min()), float(numeric.max())
    if np.isclose(minimum, maximum):
        return pd.Series(np.zeros(len(numeric)), index=values.index)
    return (numeric - minimum) / (maximum - minimum)


def fuse_task(
    base: pd.DataFrame,
    kg: pd.DataFrame,
    rgcn: pd.DataFrame,
    head: str,
    weights: tuple[float, float, float],
) -> pd.DataFrame:
    keys = [head, "symptom"]
    out = base[keys + ["score", "support_herbs"]].copy()
    out = out.rename(columns={"score": "path_score"})
    out["path_score_norm"] = normalize(out["path_score"])
    if not kg.empty:
        out = out.merge(kg[keys + ["kg_score"]], on=keys, how="left")
    else:
        out["kg_score"] = np.nan
    if not rgcn.empty:
        out = out.merge(rgcn[keys + ["rgcn_score"]], on=keys, how="left")
    else:
        out["rgcn_score"] = np.nan
    out["kg_score"] = pd.to_numeric(out["kg_score"], errors="coerce").fillna(0.0)
    out["rgcn_score"] = pd.to_numeric(out["rgcn_score"], errors="coerce").fillna(0.0)
    out["kg_score_norm"] = normalize(out["kg_score"])
    out["rgcn_score_norm"] = normalize(out["rgcn_score"])
    out["fused_score"] = (
        weights[0] * out["path_score_norm"]
        + weights[1] * out["kg_score_norm"]
        + weights[2] * out["rgcn_score_norm"]
    )
    out = out.sort_values(["fused_score", "support_herbs"], ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--kg-dir", type=Path, required=True)
    parser.add_argument("--rgcn-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-weight", type=float, default=0.5)
    parser.add_argument("--kg-weight", type=float, default=0.25)
    parser.add_argument("--rgcn-weight", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    weights = np.asarray([args.path_weight, args.kg_weight, args.rgcn_weight], dtype=float)
    if np.any(weights < 0) or np.isclose(weights.sum(), 0):
        raise ValueError("Fusion weights must be non-negative and not all zero.")
    weights = tuple((weights / weights.sum()).tolist())

    ingredient_base = pd.read_csv(args.graph_dir / "ingredient_symptom_scores.csv")
    target_base = pd.read_csv(args.graph_dir / "target_symptom_scores.csv")
    ingredient_kg = pd.read_csv(args.kg_dir / "ingredient_symptom_kg_predictions.csv")
    target_kg = pd.read_csv(args.kg_dir / "target_symptom_kg_predictions.csv")
    ingredient_rgcn = pd.read_csv(args.rgcn_dir / "ingredient_symptom_rgcn_predictions.csv")
    target_rgcn = pd.read_csv(args.rgcn_dir / "target_symptom_rgcn_predictions.csv")

    ingredient = fuse_task(ingredient_base, ingredient_kg, ingredient_rgcn, "ingredient", weights)
    target = fuse_task(target_base, target_kg, target_rgcn, "target", weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ingredient.head(args.top_k).to_csv(args.output_dir / "top_ingredient_symptom_candidates.csv", index=False)
    target.head(args.top_k).to_csv(args.output_dir / "top_target_symptom_candidates.csv", index=False)

    metrics = pd.DataFrame([
        {"task": "ingredient_symptom", "candidate_count": len(ingredient), "top_k": min(args.top_k, len(ingredient)), "max_fused_score": float(ingredient["fused_score"].max()) if not ingredient.empty else np.nan},
        {"task": "target_symptom", "candidate_count": len(target), "top_k": min(args.top_k, len(target)), "max_fused_score": float(target["fused_score"].max()) if not target.empty else np.nan},
    ])
    metrics.to_csv(args.output_dir / "ranking_summary.csv", index=False)
    print(f"Wrote ranked candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
