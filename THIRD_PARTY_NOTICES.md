# Third-Party Notices

This file records the direct software dependencies and the data status of
this public release. It does not replace the license text supplied by an
upstream project.

## Direct dependencies

| Package | License | Official source |
|---|---|---|
| NumPy | BSD 3-Clause | <https://github.com/numpy/numpy> |
| pandas | BSD 3-Clause | <https://github.com/pandas-dev/pandas> |
| scikit-learn | BSD 3-Clause | <https://github.com/scikit-learn/scikit-learn> |
| PyTorch | BSD 3-Clause | <https://github.com/pytorch/pytorch> |
| openpyxl | MIT | <https://foss.heptapod.net/openpyxl/openpyxl> |

The packages above are installed as dependencies and are not copied into this
repository. Their original licenses remain applicable to their own code.

## Data status

The files under `examples/` are synthetic demonstration inputs created to
exercise the public schema. They are not DOR observations and are not derived
from the private workbooks. The original DOR workbooks, curated mappings, PPI
tables, model checkpoints, and private results are not redistributed.

Users supplying external biomedical data are responsible for confirming that
the data license permits the intended use and redistribution.

## Scope of this repository license

The MIT license in `LICENSE` applies to the original code and documentation
created for this release. It does not relicense external datasets, pretrained
models, or third-party packages supplied by users.
