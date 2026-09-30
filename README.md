# PEAC-Log

Public repository: https://github.com/hope2409603570/PEAC-Log

This repository contains the authors' PEAC-Log log-parsing pipeline and its
supporting evaluation scripts, ablation studies, and tests.

## Project layout

- `main.py` — stable public API and the four-stage parsing pipeline.
- `src/` — preprocessing, clustering, template extraction, and validated config loaders.
- `config/preprocessing_rules.json` — ordered regular-expression substitutions for all datasets.
- `config/pipeline_parameters.json` — dataset-specific parser hyperparameters.
- `utils/benchmark.py` — full benchmark runner.
- `experiments/` — controlled experiments and claim-scope checks.
- `tests/` — regression tests for metrics, claims, configuration, and preprocessing.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Quick start

```python
from main import PEACLogPipeline, preprocess_logs

raw_logs = [
    "Received block blk_123 from 10.0.0.1:50010",
    "Received block blk_456 from 10.0.0.2:50010",
]

cleaned = preprocess_logs("HDFS", raw_logs)
pipeline = PEACLogPipeline("HDFS")
cluster_to_event, event_to_template, clusters, duration = pipeline.parse(cleaned)
```

The existing compatibility functions, such as `hdfs_standard_preprocess`, are
still exported from `main.py` for older experiment scripts.

Run the bundled data-free demonstration with:

```bash
python -m examples.quick_demo
```

## Editing regular-expression rules

Rules in `config/preprocessing_rules.json` are applied in listed order. Each
rule has the following form:

```json
{
  "name": "rule_01",
  "pattern": "\\b\\d+\\b",
  "replacement": "<*>",
  "flags": ["IGNORECASE"]
}
```

Supported flags are `ASCII`, `DOTALL`, `IGNORECASE`, `MULTILINE`, and `VERBOSE`.
The loader validates the schema, every flag, every pattern, and replacement
group references before preprocessing starts. Unknown dataset names use the
configured `default_dataset`; `Proxifier` is explicitly aliased to `loghub`.

## Dataset download and placement

PEAC-Log uses the 14 annotated full datasets from
[LogHub-2.0](https://github.com/logpai/loghub-2.0). Download the dataset
archives from the official [LogHub-2.0 Zenodo
record](https://zenodo.org/records/8275861), then extract the following
datasets: `Apache`, `BGL`, `Hadoop`, `HDFS`, `HealthApp`, `HPC`,
`Linux`, `Mac`, `OpenSSH`, `OpenStack`, `Proxifier`, `Spark`,
`Thunderbird`, and `Zookeeper`.

Create a `data/` directory at the same level as `main.py`. Each extracted
dataset must be placed in its own case-sensitive directory, with the structured
CSV named exactly as shown below:

```text
data/
├── Apache/Apache_full.log_structured.csv
├── BGL/BGL_full.log_structured.csv
├── Hadoop/Hadoop_full.log_structured.csv
├── HDFS/HDFS_full.log_structured.csv
├── HealthApp/HealthApp_full.log_structured.csv
├── HPC/HPC_full.log_structured.csv
├── Linux/Linux_full.log_structured.csv
├── Mac/Mac_full.log_structured.csv
├── OpenSSH/OpenSSH_full.log_structured.csv
├── OpenStack/OpenStack_full.log_structured.csv
├── Proxifier/Proxifier_full.log_structured.csv
├── Spark/Spark_full.log_structured.csv
├── Thunderbird/Thunderbird_full.log_structured.csv
└── Zookeeper/Zookeeper_full.log_structured.csv
```

The benchmark reads the `LineId`, `Content`, and `EventTemplate` columns
from these files. Do not rename the dataset directories or CSV files. The full
collection is large; the official LogHub-2.0 documentation recommends at least
100 GB of free storage and 16 GB of memory for large-scale benchmarking.

## Processing and modeling workflow

`utils/benchmark.py` reads `LineId`, `Content`, and `EventTemplate` from each
structured CSV and performs the following steps.

1. **Configuration and normalization.** The dataset name selects an ordered
   list of regular-expression substitutions from
   `config/preprocessing_rules.json`. Each expression is compiled and
   validated before use. The selected rules replace variable fields with
   `<*>`. The four parser parameters (`entropy_threshold`,
   `max_check_positions`, `min_similarity`, and `sample_size`) are loaded from
   `config/pipeline_parameters.json`.
2. **Positional-entropy bucketing.** `SmartPreprocessor` splits each normalized
   message on whitespace. For every observed position up to
   `max_check_positions`, it counts token frequencies among messages that reach
   that position and calculates Shannon entropy. Positions with entropy below
   `entropy_threshold` become anchors. A message's bucket key is the ordered
   sequence of its available `position:token` anchors; messages without an
   available anchor use the deterministic `NO_ANCHOR` key.
3. **Within-bucket clustering.** `AdaptiveLSHClusterer` converts every message
   to a token set and computes the pairwise Jaccard distance inside each bucket.
   It sorts the condensed distances and selects the point with the greatest
   perpendicular deviation from the line joining the two endpoints. The final
   distance cutoff is capped at `1 - min_similarity`. Complete-linkage
   hierarchical clustering then divides the bucket at that cutoff.
4. **Template extraction.** `FastTemplateExtractor` orders each cluster by
   message length. If the cluster is larger than `sample_size`, it retains the
   shortest and longest messages and selects the remaining representatives at
   evenly spaced length ranks. The representatives are merged iteratively with
   dynamic-programming LCS alignment; unmatched spans become `<*>`.
5. **Event assignment.** Clusters are processed from largest to smallest. Each
   extracted template is normalized with the template rules in
   `preprocessing_rules.json`. Templates with the same normalized key share an
   event identifier; new keys receive `E1`, `E2`, and so on.
6. **Full-file replay.** Files with at most 50,000 messages are parsed directly.
   For larger files, the benchmark counts every normalized string, trains the
   parser on at most the 8,000 most frequent distinct strings, and builds an
   exact-string cache. A cache miss compares the message with candidate
   templates by token-set Jaccard similarity after token-count filtering, then
   stores the resolved event in the cache for later occurrences.
7. **Output and evaluation.** Predictions are written with `LineId`, `Content`,
   `EventId`, and `EventTemplate`. `utils/evaluate.py` calculates PA, GA, FGA,
   and FTA, and the generated files are stored below `result/`.


## Running the benchmark and tests

After placing the datasets as described above, run:

```bash
python utils/benchmark.py
python -m pytest -q
```

Generated predictions are written below `result/`. Large datasets, generated
results, editor metadata, and caches are excluded by `.gitignore` so a source
release remains compact.

## Reproducibility notes

- Keep rule order unchanged unless intentionally defining a new experiment.
- Keep the JSON configuration files with every reported run so the exact rules
  and parameters can be recovered.

## Citation and license

Citation metadata are provided in `CITATION.cff`. This source release is
distributed under the MIT License; see `LICENSE`.
