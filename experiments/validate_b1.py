"""Validate B1 quadrant metadata and factor isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.b1_runner import RUN_ROOT, RUN_TO_QUADRANT


def validate(datasets: list[str]) -> dict[str, object]:
    checks: dict[str, bool] = {}
    records: dict[str, dict[str, dict[str, object]]] = {}
    for dataset in datasets:
        records[dataset] = {}
        for run_id in RUN_TO_QUADRANT:
            path = RUN_ROOT / run_id / dataset / "run_metadata.json"
            if not path.exists():
                checks[f"{dataset}_{run_id}_exists"] = False
                continue
            records[dataset][run_id] = json.loads(path.read_text(encoding="utf-8"))
            checks[f"{dataset}_{run_id}_exists"] = True
        if len(records[dataset]) != 4:
            continue
        uu, tt, ut, tu = (records[dataset][run] for run in ("R010", "R011", "R012", "R013"))
        checks[f"{dataset}_UU_UT_same_preprocess"] = uu["preprocessed_sha256"] == ut["preprocessed_sha256"]
        checks[f"{dataset}_TU_TT_same_preprocess"] = tu["preprocessed_sha256"] == tt["preprocessed_sha256"]
        checks[f"{dataset}_UU_TU_same_params"] = uu["config"]["params"] == tu["config"]["params"]
        checks[f"{dataset}_UT_TT_same_params"] = ut["config"]["params"] == tt["config"]["params"]
        checks[f"{dataset}_all_line_counts_equal"] = len({item["rows"] for item in (uu, tt, ut, tu)}) == 1
        checks[f"{dataset}_all_predictions_aligned"] = all(
            item["metrics"]["alignment"]["missing_line_ids"] == []
            and item["metrics"]["alignment"]["extra_line_ids"] == []
            for item in (uu, tt, ut, tu)
        )
    result = {
        "milestone": "M1",
        "datasets": datasets,
        "checks": checks,
        "all_checks_pass": all(checks.values()) if checks else False,
        "records": records,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="Proxifier,BGL")
    parser.add_argument("--output", default=str(RUN_ROOT / "B1_SMOKE_VALIDATION.json"))
    args = parser.parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    result = validate(datasets)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "all_checks_pass": result["all_checks_pass"]}, ensure_ascii=False, indent=2))
    return 0 if result["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

