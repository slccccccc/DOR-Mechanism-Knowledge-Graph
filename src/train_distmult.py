                     
                       
"""Train the DistMult link-prediction baseline on a normalized graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GRAPH_DIR = Path("outputs/graph")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def filter_weak_scores(
    df: pd.DataFrame,
    max_rows: int,
    min_support_herbs: int,
    min_score_quantile: float,
    require_mechanism: bool,
    task: str,
) -> pd.DataFrame:
    pass
    if df.empty:
        return df
    out = df.copy()
    if "support_herbs" in out.columns:
        out = out[out["support_herbs"].fillna(0).astype(float) >= min_support_herbs]
    if min_score_quantile > 0 and "score" in out.columns and not out.empty:
        out = out[out["score"] >= out["score"].quantile(min_score_quantile)]
    if require_mechanism:
        if task == "ingredient" and "target_count" in out.columns:
            out = out[out["target_count"].fillna(0).astype(float) > 0]
        if task == "target" and "support_ingredients" in out.columns:
            out = out[out["support_ingredients"].fillna(0).astype(float) >= 2]
    return out.sort_values("score", ascending=False).head(max_rows).reset_index(drop=True)


def add_typed_inverse_relations(triples: pd.DataFrame) -> pd.DataFrame:
    pass
    symmetric_relations = {"interacts_with", "maps_to_dor_symptom_exact"}
    reverse = triples.copy()
    reverse[["source", "target"]] = reverse[["target", "source"]]
    reverse["relation"] = reverse["relation"].map(
        lambda relation: relation if relation in symmetric_relations else f"{relation}__inverse"
    )
    return pd.concat([triples, reverse], ignore_index=True).drop_duplicates().reset_index(drop=True)


def load_triples(
    graph_dir: Path,
    max_predicted_edges: int,
    min_support_herbs: int,
    min_score_quantile: float,
    require_mechanism: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = pd.read_csv(graph_dir / "graph_edges.csv", dtype=str)
    edges = edges.loc[:, ["source", "relation", "target"]].drop_duplicates()

                                                  
    weak_relation_names = {"predicted_related_to", "target_predicted_related_to"}
    observed = edges[~edges["relation"].isin(weak_relation_names)]
    predicted = edges[edges["relation"].eq("predicted_related_to")]
    target_predicted = edges[edges["relation"].eq("target_predicted_related_to")]
    weak_frames = []
    if max_predicted_edges > 0 and not predicted.empty:
        scored = filter_weak_scores(
            pd.read_csv(graph_dir / "ingredient_symptom_scores.csv"),
            max_predicted_edges,
            min_support_herbs,
            min_score_quantile,
            require_mechanism,
            "ingredient",
        )
        split_path = graph_dir / "ingredient_symptom_splits.csv"
        if split_path.exists():
            split = pd.read_csv(split_path, dtype=str)
            scored = scored.merge(split, on=["ingredient", "symptom"], how="inner")
            scored = scored[scored["split"].eq("train")].copy()
        keep = scored.assign(
            source=lambda d: "ingredient:" + d["ingredient"].astype(str),
            relation="associated_with",
            target=lambda d: "symptom:" + d["symptom"].astype(str),
        ).loc[:, ["source", "relation", "target"]]
        weak_frames.append(keep)
    if max_predicted_edges > 0 and not target_predicted.empty and (graph_dir / "target_symptom_scores.csv").exists():
        target_scored = filter_weak_scores(
            pd.read_csv(graph_dir / "target_symptom_scores.csv"),
            max_predicted_edges,
            min_support_herbs,
            min_score_quantile,
            require_mechanism,
            "target",
        )
        split_path = graph_dir / "target_symptom_splits.csv"
        if split_path.exists():
            split = pd.read_csv(split_path, dtype=str)
            target_scored = target_scored.merge(split, on=["target", "symptom"], how="inner")
            target_scored = target_scored[target_scored["split"].eq("train")].copy()
        target_keep = target_scored.assign(
            source=lambda d: "target:" + d["target"].astype(str),
            relation="target_associated_with",
            target=lambda d: "symptom:" + d["symptom"].astype(str),
        ).loc[:, ["source", "relation", "target"]]
        weak_frames.append(target_keep)

    triples = pd.concat([observed, *weak_frames], ignore_index=True).drop_duplicates() if weak_frames else observed.copy()
    triples = add_typed_inverse_relations(triples)

    return triples.reset_index(drop=True), edges


def encode_triples(triples: pd.DataFrame) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    entities = sorted(set(triples["source"]).union(set(triples["target"])))
    relations = sorted(set(triples["relation"]))
    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}
    arr = np.array(
        [[ent2id[h], rel2id[r], ent2id[t]] for h, r, t in triples[["source", "relation", "target"]].itertuples(index=False)],
        dtype=np.int64,
    )
    return arr, ent2id, rel2id


def train_distmult(
    triples: np.ndarray,
    n_entities: int,
    n_relations: int,
    dim: int,
    epochs: int,
    lr: float,
    negative_ratio: int,
    seed: int,
    forbidden_triples: set[tuple[int, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ent = rng.normal(0, 0.1, size=(n_entities, dim)).astype(np.float64)
    rel = rng.normal(0, 0.1, size=(n_relations, dim)).astype(np.float64)
    true_triples = set(map(tuple, triples.tolist()))
    if forbidden_triples:
        true_triples.update(forbidden_triples)
    rel_heads: dict[int, np.ndarray] = {}
    rel_tails: dict[int, np.ndarray] = {}
    for rid in np.unique(triples[:, 1]):
        part = triples[triples[:, 1] == rid]
        rel_heads[int(rid)] = np.unique(part[:, 0])
        rel_tails[int(rid)] = np.unique(part[:, 2])

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(triples))
        total_loss = 0.0
        for idx in order:
            h, r, t = triples[idx]
            samples = [(h, r, t, 1.0)]
            for _ in range(negative_ratio):
                                                                                            
                for _try in range(100):
                    if rng.random() < 0.5 and len(rel_heads[int(r)]) > 0:
                        nh, nt = int(rng.choice(rel_heads[int(r)])), int(t)
                    elif len(rel_tails[int(r)]) > 0:
                        nh, nt = int(h), int(rng.choice(rel_tails[int(r)]))
                    else:
                        nh, nt = int(h), int(rng.integers(n_entities))
                    if (nh, int(r), nt) not in true_triples:
                        samples.append((nh, int(r), nt, 0.0))
                        break

            for sh, sr, st, y in samples:
                eh = ent[sh].copy()
                rr = rel[sr].copy()
                et = ent[st].copy()
                logit = float(np.sum(eh * rr * et))
                p = float(sigmoid(np.array([logit]))[0])
                grad = p - y
                total_loss += -(y * np.log(p + 1e-12) + (1.0 - y) * np.log(1.0 - p + 1e-12))

                ent[sh] -= lr * grad * rr * et
                rel[sr] -= lr * grad * eh * et
                ent[st] -= lr * grad * eh * rr

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"epoch={epoch} loss={total_loss / max(len(order), 1):.6f}")
    return ent, rel


def encode_forbidden_weak_triples(
    graph_dir: Path,
    ent2id: dict[str, int],
    rel2id: dict[str, int],
) -> set[tuple[int, int, int]]:
    pass
    forbidden: set[tuple[int, int, int]] = set()
    specs = [
        ("ingredient_symptom_scores.csv", "ingredient", "ingredient:", "associated_with"),
        ("target_symptom_scores.csv", "target", "target:", "target_associated_with"),
    ]
    for filename, head_col, prefix, relation in specs:
        path = graph_dir / filename
        if not path.exists() or relation not in rel2id:
            continue
        scores = pd.read_csv(path, usecols=[head_col, "symptom"])
        rid = rel2id[relation]
        inverse_relation = f"{relation}__inverse"
        for head, symptom in scores.itertuples(index=False):
            h, t = prefix + str(head), "symptom:" + str(symptom)
            if h in ent2id and t in ent2id:
                forbidden.add((ent2id[h], rid, ent2id[t]))
                if inverse_relation in rel2id:
                    forbidden.add((ent2id[t], rel2id[inverse_relation], ent2id[h]))
    return forbidden


def predict_ingredient_symptom(
    graph_dir: Path,
    ent: np.ndarray,
    rel: np.ndarray,
    ent2id: dict[str, int],
    rel2id: dict[str, int],
    top_k: int,
) -> pd.DataFrame:
    scores = pd.read_csv(graph_dir / "ingredient_symptom_scores.csv")
    ingredients = sorted("ingredient:" + x for x in scores["ingredient"].astype(str).unique())
    symptoms = sorted("symptom:" + x for x in scores["symptom"].astype(str).unique())

    rel_name = next(
        (name for name in ("associated_with", "predicted_related_to", "has_symptom") if name in rel2id),
        None,
    )
    if rel_name is None:
        return pd.DataFrame(columns=["ingredient", "symptom", "kg_score"])
    rvec = rel[rel2id[rel_name]]
    rows = []
    for ing in ingredients:
        if ing not in ent2id:
            continue
        hvec = ent[ent2id[ing]]
        vals = []
        for sym in symptoms:
            if sym not in ent2id:
                continue
            logit = float(np.sum(hvec * rvec * ent[ent2id[sym]]))
            vals.append((sym, float(sigmoid(np.array([logit]))[0])))
        vals.sort(key=lambda x: x[1], reverse=True)
        for sym, score in vals[:top_k]:
            rows.append({"ingredient": ing.split(":", 1)[1], "symptom": sym.split(":", 1)[1], "kg_score": score})
    return pd.DataFrame(rows).sort_values("kg_score", ascending=False).reset_index(drop=True)


def predict_target_symptom(
    graph_dir: Path,
    ent: np.ndarray,
    rel: np.ndarray,
    ent2id: dict[str, int],
    rel2id: dict[str, int],
    top_k: int,
) -> pd.DataFrame:
    scores_path = graph_dir / "target_symptom_scores.csv"
    if not scores_path.exists():
        return pd.DataFrame(columns=["target", "symptom", "kg_score"])
    scores = pd.read_csv(scores_path)
    targets = sorted("target:" + x for x in scores["target"].astype(str).unique())
    symptoms = sorted("symptom:" + x for x in scores["symptom"].astype(str).unique())

    rel_name = next(
        (name for name in ("target_associated_with", "target_predicted_related_to", "has_symptom") if name in rel2id),
        None,
    )
    if rel_name is None:
        return pd.DataFrame(columns=["target", "symptom", "kg_score"])
    rvec = rel[rel2id[rel_name]]
    rows = []
    for tar in targets:
        if tar not in ent2id:
            continue
        hvec = ent[ent2id[tar]]
        vals = []
        for sym in symptoms:
            if sym not in ent2id:
                continue
            logit = float(np.sum(hvec * rvec * ent[ent2id[sym]]))
            vals.append((sym, float(sigmoid(np.array([logit]))[0])))
        vals.sort(key=lambda x: x[1], reverse=True)
        for sym, score in vals[:top_k]:
            rows.append({"target": tar.split(":", 1)[1], "symptom": sym.split(":", 1)[1], "kg_score": score})
    return pd.DataFrame(rows).sort_values("kg_score", ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="")
    p.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--negative-ratio", type=int, default=2)
    p.add_argument("--max-predicted-edges", type=int, default=5000, help="")
    p.add_argument("--min-support-herbs", type=int, default=2, help="")
    p.add_argument("--min-score-quantile", type=float, default=0.0, help="")
    p.add_argument("--require-mechanism", action="store_true", help="")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.graph_dir / "kg_completion_distmult")
    out_dir.mkdir(parents=True, exist_ok=True)

    triples_df, _ = load_triples(
        args.graph_dir,
        args.max_predicted_edges,
        args.min_support_herbs,
        args.min_score_quantile,
        args.require_mechanism,
    )
    triples, ent2id, rel2id = encode_triples(triples_df)
    forbidden_triples = encode_forbidden_weak_triples(args.graph_dir, ent2id, rel2id)
    print(f"triples={len(triples)} entities={len(ent2id)} relations={len(rel2id)}")

    ent, rel = train_distmult(
        triples, len(ent2id), len(rel2id), args.dim, args.epochs, args.lr, args.negative_ratio, args.seed,
        forbidden_triples,
    )
    pred = predict_ingredient_symptom(args.graph_dir, ent, rel, ent2id, rel2id, args.top_k)
    target_pred = predict_target_symptom(args.graph_dir, ent, rel, ent2id, rel2id, args.top_k)
    pred.to_csv(out_dir / "ingredient_symptom_kg_predictions.csv", index=False, encoding="utf-8-sig")
    target_pred.to_csv(out_dir / "target_symptom_kg_predictions.csv", index=False, encoding="utf-8-sig")
    triples_df.to_csv(out_dir / "kg_training_triples.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "entity_embeddings.npy", ent)
    np.save(out_dir / "relation_embeddings.npy", rel)
    pd.DataFrame({"entity": list(ent2id.keys()), "id": list(ent2id.values())}).to_csv(out_dir / "entity_mapping.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"relation": list(rel2id.keys()), "id": list(rel2id.values())}).to_csv(out_dir / "relation_mapping.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
