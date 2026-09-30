# Reviewer package contents

This folder is a clean copy of the authors' implementation prepared for code
review. It contains only the proposed PEAC-Log method and supporting material.

Included:

- core parsing pipeline in `main.py` and `src/`;
- ordered dataset preprocessing rules in `config/preprocessing_rules.json`;
- dataset-specific pipeline parameters in `config/pipeline_parameters.json`;
- benchmark, evaluation, sensitivity, ablation, and plotting scripts;
- experiment protocol utilities under `experiments/`;
- automated tests under `tests/`;
- a data-free runnable example under `examples/`;
- installation and usage instructions in `README.md`.
- citation metadata in `CITATION.cff` and Zenodo metadata in `.zenodo.json`;
- the MIT software license in `LICENSE`.

Not included:

- LogHub or other external datasets;
- generated predictions, tables, plots, caches, and temporary files;
- the manuscript and internal working notes.

Run `python -m pytest -q -p no:cacheprovider` from this directory to verify the
package after installing `requirements.txt`.
