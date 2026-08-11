"""Build a typed ingredient-target-symptom graph from normalized CSV files.

The public pipeline deliberately accepts normalized tabular inputs instead of
shipping the private DOR workbooks. Each input file is small and auditable,
and its schema is documented in ``docs/data_schema.md``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def read_pairs(path: Path, left: str, right: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = {left, right} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    out = frame[[left, right]].copy()
    out[left] = out[left].astype(str).str.strip()
    out[right] = out[right].astype(str).str.strip()
    return out[(out[left] != "") & (out[right] != "")].drop_duplicates()


def typed(kind: str, values: pd.Series) -> pd.Series:
    return kind + ":" + values.astype(str)


def stable_split(head: str, tail: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{head}:{tail}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "test" if value < 0.2 else "validation" if value < 0.3 else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--herb-ingredient", type=Path, required=True)
    parser.add_argument("--herb-symptom", type=Path, required=True)
    parser.add_argument("--ingredient-target", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    herb_ingredient = read_pairs(args.herb_ingredient, "herb", "ingredient")
    herb_symptom = read_pairs(args.herb_symptom, "herb", "symptom")
    ingredient_target = (
        read_pairs(args.ingredient_target, "ingredient", "target")
        if args.ingredient_target
        else pd.DataFrame(columns=["ingredient", "target"])
    )

    edges = []
    for row in herb_ingredient.itertuples(index=False):
        edges.append((f"herb:{row.herb}", "contains", f"ingredient:{row.ingredient}", 1.0))
    for row in herb_symptom.itertuples(index=False):
        edges.append((f"herb:{row.herb}", "associated_with", f"symptom:{row.symptom}", 1.0))
    for row in ingredient_target.itertuples(index=False):
        edges.append((f"ingredient:{row.ingredient}", "targets", f"target:{row.target}", 1.0))

    herb_sets = herb_symptom.groupby("herb")["symptom"].apply(set).to_dict()
    ingredient_herbs = herb_ingredient.groupby("ingredient")["herb"].apply(set).to_dict()
    ingredient_rows = []
    for ingredient, herbs in ingredient_herbs.items():
        symptoms = {s for herb in herbs for s in herb_sets.get(herb, set())}
        for symptom in symptoms:
            support = sum(symptom in herb_sets.get(herb, set()) for herb in herbs)
            score = support / np.sqrt(max(len(herbs), 1) * max(sum(symptom in x for x in herb_sets.values()), 1))
            ingredient_rows.append((ingredient, symptom, float(score), support))

    target_rows = []
    for target, ingredients in ingredient_target.groupby("target")["ingredient"].apply(set).items():
        symptoms = {s for ingredient in ingredients for herb in ingredient_herbs.get(ingredient, set()) for s in herb_sets.get(herb, set())}
        for symptom in symptoms:
            support = sum(
                symptom in herb_sets.get(herb, set())
                for ingredient in ingredients
                for herb in ingredient_herbs.get(ingredient, set())
            )
            target_rows.append((target, symptom, float(support / max(len(ingredients), 1)), support))

    ingredient_scores = pd.DataFrame(ingredient_rows, columns=["ingredient", "symptom", "score", "support_herbs"])
    target_scores = pd.DataFrame(target_rows, columns=["target", "symptom", "score", "support_ingredients"])
    edge_frame = pd.DataFrame(edges, columns=["source", "relation", "target", "weight"]).drop_duplicates()
    nodes = sorted(set(edge_frame["source"]).union(edge_frame["target"]))
    node_frame = pd.DataFrame({"node_id": nodes, "node_type": [node.split(":", 1)[0] for node in nodes], "name": [node.split(":", 1)[1] for node in nodes]})

    for row in ingredient_scores.itertuples(index=False):
        edge_frame.loc[len(edge_frame)] = [f"ingredient:{row.ingredient}", "predicted_related_to", f"symptom:{row.symptom}", row.score]
    for row in target_scores.itertuples(index=False):
        edge_frame.loc[len(edge_frame)] = [f"target:{row.target}", "target_predicted_related_to", f"symptom:{row.symptom}", row.score]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    edge_frame.to_csv(args.output_dir / "graph_edges.csv", index=False)
    node_frame.to_csv(args.output_dir / "graph_nodes.csv", index=False)
    ingredient_scores.to_csv(args.output_dir / "ingredient_symptom_scores.csv", index=False)
    target_scores.to_csv(args.output_dir / "target_symptom_scores.csv", index=False)
    for frame, head in [(ingredient_scores, "ingredient"), (target_scores, "target")]:
        if frame.empty:
            continue
        split = frame[[head, "symptom"]].copy()
        split["split"] = [stable_split(str(h), str(t), args.seed) for h, t in zip(split[head], split["symptom"])]
        split.to_csv(args.output_dir / f"{head}_symptom_splits.csv", index=False)
    print(f"Wrote {len(node_frame)} nodes and {len(edge_frame)} edges to {args.output_dir}")


if __name__ == "__main__":
    main()
