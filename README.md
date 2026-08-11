# DOR Mechanism Knowledge Graph

Research code for constructing and ranking ingredient-target-symptom evidence
for diminished ovarian reserve (DOR) mechanism analysis.

The repository contains code only. The original DOR workbooks, curated name
decisions, PPI tables, model checkpoints, and result files are private and are
not included. Users must provide their own legally distributable, normalized
CSV inputs following [the documented schema](docs/data_schema.md).

## Pipeline

1. Build a typed graph and weakly supervised candidate scores.
2. Train a DistMult link-prediction baseline.
3. Train the relation-aware graph convolutional model.
4. Calibrate and evaluate held-out candidate relations.

The graph builder is independent of Chinese workbook names and local paths.
The training and evaluation scripts consume only the normalized files produced
by the builder.

## Installation

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

## Example commands

```bash
python src/build_graph.py \
  --herb-ingredient examples/herb_ingredient.csv \
  --herb-symptom examples/herb_symptom.csv \
  --ingredient-target examples/ingredient_target.csv \
  --output-dir outputs/graph

python src/train_distmult.py --graph-dir outputs/graph --output-dir outputs/distmult
python src/train_rgcn.py --graph-dir outputs/graph --output-dir outputs/rgcn
python src/evaluate.py --graph-dir outputs/graph --kg-dir outputs/distmult --rgcn-dir outputs/rgcn --output-dir outputs/evaluation
```

The example files are synthetic and are included only to validate the input
contract. They are not DOR observations and must not be interpreted as
biomedical findings.

## Scope and attribution

This project is a computational research workflow. Candidate scores represent
ranked evidence for downstream investigation; they are not clinical diagnoses,
therapeutic recommendations, or experimental validation.

## License

The public release is provided under the MIT License. Data providers and
upstream software retain their respective terms; users are responsible for
checking the license of every dataset and pretrained model they supply.
See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the dependency and
example-data status.
