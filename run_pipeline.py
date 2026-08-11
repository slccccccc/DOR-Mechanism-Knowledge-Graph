"""Run the public DOR graph-ranking workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("$", " ".join(map(str, command)))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--herb-ingredient", type=Path, required=True)
    parser.add_argument("--herb-symptom", type=Path, required=True)
    parser.add_argument("--ingredient-target", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    graph_dir = args.output_dir / "graph"
    kg_dir = args.output_dir / "distmult"
    rgcn_dir = args.output_dir / "rgcn"
    evaluation_dir = args.output_dir / "evaluation"
    root = Path(__file__).resolve().parent

    build = [sys.executable, str(root / "src" / "build_graph.py"), "--herb-ingredient", str(args.herb_ingredient), "--herb-symptom", str(args.herb_symptom), "--output-dir", str(graph_dir)]
    if args.ingredient_target:
        build.extend(["--ingredient-target", str(args.ingredient_target)])
    run(build)
    run([sys.executable, str(root / "src" / "train_distmult.py"), "--graph-dir", str(graph_dir), "--out-dir", str(kg_dir), "--epochs", str(args.epochs)])
    run([sys.executable, str(root / "src" / "train_rgcn.py"), "--graph-dir", str(graph_dir), "--out-dir", str(rgcn_dir), "--epochs", str(args.epochs), "--cpu"])
    run([sys.executable, str(root / "src" / "evaluate.py"), "--graph-dir", str(graph_dir), "--kg-dir", str(kg_dir), "--rgcn-dir", str(rgcn_dir), "--output-dir", str(evaluation_dir)])


if __name__ == "__main__":
    main()
