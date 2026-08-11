                     
                       
"""Train the lightweight relation-aware graph convolutional model."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:                    
    raise SystemExit("PyTorch is required for the R-GCN experiment.") from exc


DEFAULT_GRAPH_DIR = Path("outputs/graph")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def load_data(
    graph_dir: Path,
    max_positive: int,
    min_support_herbs: int,
    min_score_quantile: float,
    require_mechanism: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str], list[str], set[tuple[str, str]], set[tuple[str, str]]]:
    edges = pd.read_csv(graph_dir / "graph_edges.csv", dtype=str)
    weak_relations = {"predicted_related_to", "target_predicted_related_to"}
    base_edges = edges[~edges["relation"].isin(weak_relations)]
    scores = pd.read_csv(graph_dir / "ingredient_symptom_scores.csv")
    target_scores_path = graph_dir / "target_symptom_scores.csv"
    target_scores = pd.read_csv(target_scores_path) if target_scores_path.exists() else pd.DataFrame(columns=["target", "symptom", "score", "support_herbs"])
    if max_positive > 0:
        scores = filter_weak_scores(scores, max_positive, min_support_herbs, min_score_quantile, require_mechanism, "ingredient")
        target_scores = filter_weak_scores(target_scores, max_positive, min_support_herbs, min_score_quantile, require_mechanism, "target")
    all_scores = scores.copy()
    all_target_scores = target_scores.copy()
    ingredient_split_path = graph_dir / "ingredient_symptom_splits.csv"
    target_split_path = graph_dir / "target_symptom_splits.csv"
    if ingredient_split_path.exists():
        scores = scores.merge(pd.read_csv(ingredient_split_path, dtype=str), on=["ingredient", "symptom"], how="inner")
        scores = scores[scores["split"].eq("train")].copy()
    if target_split_path.exists():
        target_scores = target_scores.merge(pd.read_csv(target_split_path, dtype=str), on=["target", "symptom"], how="inner")
        target_scores = target_scores[target_scores["split"].eq("train")].copy()
    positives = scores.assign(
        source=lambda d: "ingredient:" + d["ingredient"].astype(str),
        target=lambda d: "symptom:" + d["symptom"].astype(str),
    ).loc[:, ["source", "target", "score", "support_herbs"]]
    target_positives = target_scores.assign(
        source=lambda d: "target:" + d["target"].astype(str),
        target=lambda d: "symptom:" + d["symptom"].astype(str),
    ).loc[:, ["source", "target", "score", "support_herbs"]]
    all_ingredient_pairs = set(zip("ingredient:" + all_scores["ingredient"].astype(str), "symptom:" + all_scores["symptom"].astype(str)))
    all_target_pairs = set(zip("target:" + all_target_scores["target"].astype(str), "symptom:" + all_target_scores["symptom"].astype(str)))
    ingredients = sorted({head for head, _ in all_ingredient_pairs})
    targets = sorted({head for head, _ in all_target_pairs})
    symptoms = sorted({tail for _, tail in all_ingredient_pairs.union(all_target_pairs)})
    return (
        base_edges.loc[:, ["source", "relation", "target", "weight"]].drop_duplicates(),
        positives, target_positives, ingredients, targets, symptoms, all_ingredient_pairs, all_target_pairs,
    )


def build_index(edges: pd.DataFrame, positives: pd.DataFrame, target_positives: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    nodes = (
        set(edges["source"]).union(edges["target"])
        .union(positives["source"]).union(positives["target"])
        .union(target_positives["source"]).union(target_positives["target"])
    )
    rels = sorted(edges["relation"].unique())
    return {n: i for i, n in enumerate(sorted(nodes))}, {r: i for i, r in enumerate(rels)}


def add_typed_inverse_relations(edges: pd.DataFrame) -> pd.DataFrame:
    pass
    symmetric_relations = {"interacts_with", "maps_to_dor_symptom_exact"}
    reverse = edges.copy()
    reverse[["source", "target"]] = reverse[["target", "source"]]
    reverse["relation"] = reverse["relation"].map(
        lambda relation: relation if relation in symmetric_relations else f"{relation}__inverse"
    )
    return pd.concat([edges, reverse], ignore_index=True).drop_duplicates(
        ["source", "relation", "target"]
    ).reset_index(drop=True)


def build_adj(edges: pd.DataFrame, node2id: dict[str, int], rel2id: dict[str, int], device: torch.device) -> list[torch.Tensor]:
    n = len(node2id)
    adjs: list[torch.Tensor] = []
    for rel, _rid in rel2id.items():
        part = edges[edges["relation"].eq(rel)]
        rows = []
        cols = []
        values = []
        for s, t, weight in part[["source", "target", "weight"]].itertuples(index=False):
            if s in node2id and t in node2id:
                a, b = node2id[s], node2id[t]
                                                                                  
                rows.append(b)
                cols.append(a)
                numeric_weight = float(pd.to_numeric(weight, errors="coerce")) if pd.notna(weight) else 1.0
                if not np.isfinite(numeric_weight) or numeric_weight <= 0:
                    numeric_weight = 1.0
                values.append(numeric_weight)
        if not rows:
            idx = torch.empty((2, 0), dtype=torch.long, device=device)
            val = torch.empty((0,), dtype=torch.float32, device=device)
        else:
            idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
            val = torch.tensor(values, dtype=torch.float32, device=device)
        adj = torch.sparse_coo_tensor(idx, val, (n, n), device=device).coalesce()
        degree = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
        norm_val = adj.values() / degree[adj.indices()[0]]
        adjs.append(torch.sparse_coo_tensor(adj.indices(), norm_val, (n, n), device=device).coalesce())
    return adjs


class SimpleRGCN(nn.Module):
    def __init__(self, n_nodes: int, n_relations: int, dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, dim)
        self.rel_weights = nn.ModuleList()
        in_dim = dim
        for _ in range(layers):
            self.rel_weights.append(nn.ModuleList([nn.Linear(in_dim, hidden, bias=False) for _ in range(n_relations)]))
            in_dim = hidden
        self.self_weights = nn.ModuleList([nn.Linear(dim if i == 0 else hidden, hidden) for i in range(layers)])
        self.dropout = dropout
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, adjs: list[torch.Tensor]) -> torch.Tensor:
        h = self.emb.weight
        for layer, rel_linears in enumerate(self.rel_weights):
            msg = self.self_weights[layer](h)
            for adj, linear in zip(adjs, rel_linears):
                msg = msg + torch.sparse.mm(adj, linear(h))
            h = F.relu(msg)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def score_pairs(self, z: torch.Tensor, heads: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        zh = z[heads]
        zt = z[tails]
        feat = torch.cat([zh, zt, zh * zt, torch.abs(zh - zt)], dim=1)
        return self.decoder(feat).squeeze(1)


def make_pairs(
    positives: pd.DataFrame,
    candidate_heads: list[str],
    candidate_tails: list[str],
    node2id: dict[str, int],
    neg_ratio: int,
    seed: int,
    forbidden_pairs: set[tuple[str, str]] | None = None,
):
    rng = random.Random(seed)
    pos_set = set(zip(positives["source"], positives["target"]))
    forbidden = forbidden_pairs or pos_set
    rows = []
    for s, t in pos_set:
        if s not in node2id or t not in node2id:
            continue
        rows.append((node2id[s], node2id[t], 1.0))
        for _ in range(neg_ratio):
                                                         
                                       
            for _try in range(50):
                ns = rng.choice(candidate_tails)
                if ns in node2id and (s, ns) not in forbidden:
                    rows.append((node2id[s], node2id[ns], 0.0))
                    break
    rng.shuffle(rows)
    arr = np.asarray(rows, dtype=np.float32)
    if len(arr) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    return arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64), arr[:, 2].astype(np.float32)


def concat_pair_sets(*pair_sets):
    hs, ts, ys = [], [], []
    for h, t, y in pair_sets:
        if len(y) == 0:
            continue
        hs.append(h)
        ts.append(t)
        ys.append(y)
    if not ys:
        raise ValueError("")
    return np.concatenate(hs), np.concatenate(ts), np.concatenate(ys)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    edges, positives, target_positives, ingredients, targets, symptoms, all_ingredient_pairs, all_target_pairs = load_data(
        args.graph_dir,
        args.max_positive,
        args.min_support_herbs,
        args.min_score_quantile,
        args.require_mechanism,
    )
    edges = add_typed_inverse_relations(edges)
    node2id, rel2id = build_index(edges, positives, target_positives)
    adjs = build_adj(edges, node2id, rel2id, device)
    ing_pairs = make_pairs(
        positives, ingredients, symptoms, node2id, args.negative_ratio, args.seed, all_ingredient_pairs
    )
    target_pairs = make_pairs(
        target_positives, targets, symptoms, node2id, args.negative_ratio, args.seed + 17, all_target_pairs
    )
    h, t, y = concat_pair_sets(ing_pairs, target_pairs)

    idx = np.arange(len(y))
    np.random.shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]
    h_t = torch.tensor(h, device=device)
    t_t = torch.tensor(t, device=device)
    y_t = torch.tensor(y, device=device)

    model = SimpleRGCN(len(node2id), len(rel2id), args.dim, args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        z = model.encode(adjs)
        logits = model.score_pairs(z, h_t[train_idx], t_t[train_idx])
        loss = F.binary_cross_entropy_with_logits(logits, y_t[train_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                z = model.encode(adjs)
                pred = torch.sigmoid(model.score_pairs(z, h_t[test_idx], t_t[test_idx]))
                label = y_t[test_idx]
                acc = ((pred >= 0.5).float() == label).float().mean().item()
            print(f"epoch={epoch} loss={loss.item():.6f} test_acc={acc:.4f}")

    out_dir = args.out_dir or (args.graph_dir / "rgcn_link_prediction")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    rows = []
    with torch.no_grad():
        z = model.encode(adjs)
        valid_symptoms = [s for s in symptoms if s in node2id]
        for ing in ingredients:
            if ing not in node2id:
                continue
            hid = torch.tensor([node2id[ing]] * len(valid_symptoms), device=device)
            tid = torch.tensor([node2id[s] for s in valid_symptoms], device=device)
            vals = torch.sigmoid(model.score_pairs(z, hid, tid)).cpu().numpy()
            order = np.argsort(-vals)[:args.top_k]
            for j in order:
                rows.append({"ingredient": ing.split(":", 1)[1], "symptom": valid_symptoms[j].split(":", 1)[1], "rgcn_score": float(vals[j])})
    pd.DataFrame(rows).sort_values("rgcn_score", ascending=False).to_csv(out_dir / "ingredient_symptom_rgcn_predictions.csv", index=False, encoding="utf-8-sig")

    target_rows = []
    with torch.no_grad():
        z = model.encode(adjs)
        valid_symptoms = [s for s in symptoms if s in node2id]
        for tar in targets:
            if tar not in node2id:
                continue
            hid = torch.tensor([node2id[tar]] * len(valid_symptoms), device=device)
            tid = torch.tensor([node2id[s] for s in valid_symptoms], device=device)
            vals = torch.sigmoid(model.score_pairs(z, hid, tid)).cpu().numpy()
            order = np.argsort(-vals)[:args.top_k]
            for j in order:
                target_rows.append({"target": tar.split(":", 1)[1], "symptom": valid_symptoms[j].split(":", 1)[1], "rgcn_score": float(vals[j])})
    pd.DataFrame(target_rows).sort_values("rgcn_score", ascending=False).to_csv(out_dir / "target_symptom_rgcn_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"node": list(node2id.keys()), "id": list(node2id.values())}).to_csv(out_dir / "node_mapping.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"relation": list(rel2id.keys()), "id": list(rel2id.values())}).to_csv(
        out_dir / "relation_mapping.csv", index=False, encoding="utf-8-sig"
    )
    torch.save(model.state_dict(), out_dir / "rgcn_model.pt")
    print(f"Wrote outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="")
    p.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--max-positive", type=int, default=8000, help="")
    p.add_argument("--min-support-herbs", type=int, default=2, help="")
    p.add_argument("--min-score-quantile", type=float, default=0.0, help="")
    p.add_argument("--require-mechanism", action="store_true", help="")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--negative-ratio", type=int, default=2)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
