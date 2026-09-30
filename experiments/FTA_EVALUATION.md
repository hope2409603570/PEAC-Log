# FTA evaluation

`utils.evaluate.evaluate_metrics` returns `(PA, GA, FGA, FTA)` and writes all
four metrics, PTA/RTA, Matched_Templates and FTA_definition into the summary.
Benchmark and ablation callers inherit FTA automatically; sensitivity output
also includes FTA. CSV rows are matched by LineId, not their physical order.

FTA uses the template counting rule from the public LogHub-2.0 evaluator:
https://github.com/logpai/loghub-2.0/blob/main/benchmark/evaluation/utils/template_level_analysis.py

Let C be the number of predicted normalized templates whose assigned messages
all have that same normalized oracle template. With P predicted and G oracle
normalized templates, PTA=C/P, RTA=C/G, FTA=2C/(P+G).
GA/FGA require exact message-group equality; FTA does not. FTA can exceed FGA.
It is not classification macro F1 and not just intersection of template sets.

Project normalization replaces angle-bracket placeholders with `*`, removes
characters except ASCII letters, digits and `*`, and lowercases the remainder.
The implementation matches the official counting rule after this local
normalization; scores should not be labeled strict raw-token comparisons.

Historical B0/B1 results that declared macro F1 must be reevaluated from their
original predictions. Historical CSV rows without FTA remain blank. Neither
PA nor FGA can reconstruct FTA. Keep each evaluation tied to its original run;
do not fill an old run using another run's predictions.

To evaluate existing predictions without parsing again, supply the actual
ground truth and prediction paths and a separate output path:

```powershell
python utils/evaluate.py --gt <ground_truth.csv> --pred <predictions.csv> --dataset <dataset> --output <summary.csv>
python -B -m unittest discover -s tests -p test_fta_metrics.py
```

Use --runtime only for a measured duration from that same run. Without it,
runtime and EPS are zero (no timing claim). Summary scripts print the number
of datasets with available FTA rather than silently treating missing scores
as zero. Changes do not overwrite historical experiment result files.
