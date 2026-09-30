"""Run the M0/B0 reproducibility checks and the R000 data manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from experiments.b0_evaluator import (
    EvaluationError,
    align_by_line_id,
    evaluate_prediction,
    normalize_template,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "result_runs"
DATASETS = [
    "Proxifier", "Linux", "HDFS", "Mac", "Apache", "Hadoop", "BGL",
    "HealthApp", "HPC", "OpenSSH", "OpenStack", "Thunderbird", "Zookeeper", "Spark",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_inventory(path: Path, kind: str) -> dict[str, object]:
    row_count = 0
    line_ids: set[str] = set()
    duplicate_line_ids: list[str] = []
    event_ids: set[str] = set()
    templates: set[str] = set()
    columns: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            if "LineId" in row:
                line_id = str(row.get("LineId", ""))
                if line_id in line_ids:
                    duplicate_line_ids.append(line_id)
                line_ids.add(line_id)
            if "EventId" in row:
                event_ids.add(str(row.get("EventId", "")))
            if "EventTemplate" in row:
                templates.add(normalize_template(row.get("EventTemplate")))
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "columns": columns,
        "rows": row_count,
        "unique_line_ids": len(line_ids) if "LineId" in columns else None,
        "duplicate_line_ids": len(duplicate_line_ids),
        "unique_event_ids": len(event_ids) if "EventId" in columns else None,
        "unique_templates": len(templates) if "EventTemplate" in columns else None,
        "line_id_sample": sorted(line_ids)[:3] if line_ids else [],
    }


def _raw_inventory(path: Path) -> dict[str, object]:
    line_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_count, _ in enumerate(handle, start=1):
            pass
    return {
        "kind": "raw_log",
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": line_count,
    }


def _code_inventory() -> dict[str, object]:
    files = []
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or "result_runs" in path.parts:
            continue
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)})
    aggregate = hashlib.sha256(
        "\n".join(f"{item['path']}:{item['sha256']}" for item in files).encode("utf-8")
    ).hexdigest()
    package_versions = {}
    for package in ("numpy", "pandas", "scipy", "tqdm"):
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "aggregate_sha256": aggregate,
        "files": files,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions,
    }


def run_r000(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for dataset in DATASETS:
        dataset_dir = ROOT / "data" / dataset
        structured = dataset_dir / f"{dataset}_full.log_structured.csv"
        templates = dataset_dir / f"{dataset}_full.log_templates.csv"
        raw = dataset_dir / f"{dataset}_full.log"
        for path, kind, inventory in (
            (structured, "structured_csv", lambda item: _csv_inventory(item, kind)),
            (templates, "templates_csv", lambda item: _csv_inventory(item, kind)),
            (raw, "raw_log", lambda item: _raw_inventory(item)),
        ):
            if not path.exists():
                records.append({"dataset": dataset, "kind": kind, "path": str(path), "missing": True})
                continue
            record = inventory(path)
            record["dataset"] = dataset
            records.append(record)
    manifest = {
        "run_id": "R000",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "datasets": DATASETS,
        "files": records,
        "code": _code_inventory(),
        "command": "python -m experiments.run_b0 --run R000",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["dataset", "kind", "path", "bytes", "sha256", "rows", "unique_line_ids", "duplicate_line_ids", "unique_event_ids", "unique_templates", "missing"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fields})
    return manifest


def run_r000_validation(output_dir: Path) -> dict[str, object]:
    """Validate cross-file row and occurrence invariants without rehashing."""

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    checks: dict[str, object] = {
        "dataset_count": len(manifest["datasets"]) == 14,
        "record_count": len(files) == 42,
        "no_missing_files": not any(item.get("missing") for item in files),
        "no_duplicate_structured_line_ids": not any(
            item.get("kind") == "structured_csv" and item.get("duplicate_line_ids", 0)
            for item in files
        ),
    }
    per_dataset: dict[str, object] = {}
    for dataset in manifest["datasets"]:
        structured = next(
            item for item in files
            if item["dataset"] == dataset and item["kind"] == "structured_csv"
        )
        raw = next(
            item for item in files
            if item["dataset"] == dataset and item["kind"] == "raw_log"
        )
        templates_path = ROOT / "data" / dataset / f"{dataset}_full.log_templates.csv"
        occurrence_sum = 0
        normalized_templates: set[str] = set()
        template_rows = 0
        with templates_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                template_rows += 1
                occurrence_sum += int(row["Occurrences"])
                normalized_templates.add(normalize_template(row.get("EventTemplate")))
        per_dataset[dataset] = {
            "raw_rows_equal_structured": raw.get("rows") == structured.get("rows"),
            "template_occurrences_equal_structured": occurrence_sum == structured.get("rows"),
            "template_rows": template_rows,
            "structured_event_ids": structured.get("unique_event_ids"),
            "normalized_template_count": len(normalized_templates),
            "normalized_template_collisions": template_rows - len(normalized_templates),
        }
        checks[f"{dataset}_raw_rows_equal_structured"] = per_dataset[dataset]["raw_rows_equal_structured"]
        checks[f"{dataset}_occurrences_equal_structured"] = per_dataset[dataset]["template_occurrences_equal_structured"]
    checks["all_core_checks_pass"] = all(bool(value) for value in checks.values())
    validation = {
        "run_id": "R000",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "checks": checks,
        "per_dataset": per_dataset,
        "notes": [
            "Normalized template collisions are reported, not treated as row-integrity failures.",
            "FTA uses LogHub-2.0 template purity counting after project normalization; historical macro-F1 scores require reevaluation.",
        ],
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return validation


def run_r000_refresh_code(output_dir: Path) -> dict[str, object]:
    """Refresh the code/dependency fingerprint after B0 tooling changes."""

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["code"] = _code_inventory()
    manifest["code_refreshed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"run_id": "R000_REFRESH_CODE", "code": manifest["code"]}


def _write_fixture(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["LineId", "EventTemplate"])
        writer.writeheader()
        writer.writerows(rows)


def _fixture_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    gt_path = output_dir / "fixture_gt.csv"
    pred_path = output_dir / "fixture_pred.csv"
    shuffled_path = output_dir / "fixture_pred_shuffled.csv"
    gt_rows = [
        {"LineId": str(index), "EventTemplate": label}
        for index, label in enumerate(["A <*>", "A <*>", "B <*>", "B <*>", "C"], start=1)
    ]
    pred_rows = [
        {"LineId": str(index), "EventTemplate": label}
        for index, label in enumerate(["A <*>", "A <*>", "B <*>", "C", "C"], start=1)
    ]
    _write_fixture(gt_path, gt_rows)
    _write_fixture(pred_path, pred_rows)
    shuffled = pred_rows[:]
    random.Random(42).shuffle(shuffled)
    _write_fixture(shuffled_path, shuffled)
    return gt_path, pred_path, shuffled_path


def run_r001(output_dir: Path) -> dict[str, object]:
    gt_path, pred_path, _ = _fixture_paths(output_dir)
    metrics = evaluate_prediction(gt_path, pred_path)
    expected = {"PA": 0.8, "GA": 0.4, "FGA": 1 / 3, "FTA": 2 / 3}
    checks = {key: math.isclose(float(metrics[key]), value, rel_tol=0, abs_tol=1e-12) for key, value in expected.items()}
    if not all(checks.values()):
        raise AssertionError(f"R001 fixture mismatch: {metrics}")
    result = {
        "run_id": "R001",
        "metrics": metrics,
        "expected": expected,
        "checks": checks,
    }
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_r002(output_dir: Path) -> dict[str, object]:
    gt_path, pred_path, shuffled_path = _fixture_paths(output_dir)
    original = evaluate_prediction(gt_path, pred_path)
    shuffled = evaluate_prediction(gt_path, shuffled_path)
    metrics_equal = all(math.isclose(float(original[key]), float(shuffled[key]), rel_tol=0, abs_tol=1e-12) for key in ("PA", "GA", "FGA", "FTA"))
    if not metrics_equal:
        raise AssertionError(f"R002 order sensitivity detected: {original} vs {shuffled}")
    result = {"run_id": "R002", "metrics_equal": metrics_equal, "original": original, "shuffled": shuffled}
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_r003(output_dir: Path) -> dict[str, object]:
    gt_path, pred_path, _ = _fixture_paths(output_dir)
    summary_path = ROOT / "result" / "all_benchmark_results.csv"
    before = sha256_file(summary_path) if summary_path.exists() else None
    metrics = evaluate_prediction(gt_path, pred_path)
    explicit_output = output_dir / "metrics.json"
    explicit_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    after = sha256_file(summary_path) if summary_path.exists() else None
    unchanged = before == after
    if not unchanged:
        raise AssertionError("R003 modified result/all_benchmark_results.csv")
    result = {"run_id": "R003", "main_results_sha256_before": before, "main_results_sha256_after": after, "main_results_unchanged": unchanged, "metrics_path": str(explicit_output.resolve())}
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _load_subset(path: Path, limit: int) -> list[dict[str, str]]:
    # Select evenly spaced rows from a bounded prefix.  This keeps the smoke
    # test deterministic without using ground-truth labels to choose examples.
    source_limit = max(limit, min(10000, limit * 40))
    target_indices = {
        round(index * (source_limit - 1) / max(limit - 1, 1))
        for index in range(limit)
    }
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_index = len(rows)
            if reader.line_num - 2 in target_indices:
                rows.append({"LineId": str(row["LineId"]), "Content": row["Content"], "EventTemplate": row["EventTemplate"]})
            if reader.line_num - 1 >= source_limit:
                break
    return rows


def run_r004(output_dir: Path, subset_rows: int = 300) -> dict[str, object]:
    from main import loghub_standard_preprocess
    from src.stage1_preprocessing import SmartPreprocessor
    from src.stage2_lsh import AdaptiveLSHClusterer
    from src.stage3_4_extraction import FastTemplateExtractor

    dataset = "BGL"
    gt_path = ROOT / "data" / dataset / f"{dataset}_full.log_structured.csv"
    rows = _load_subset(gt_path, subset_rows)
    raw_logs = loghub_standard_preprocess([row["Content"] for row in rows])
    preprocessor = SmartPreprocessor(entropy_threshold=1.5, max_check_positions=12)
    clusterer = AdaptiveLSHClusterer(min_similarity=0.65)
    extractor = FastTemplateExtractor(sample_size=50)
    start = time.perf_counter()
    buckets = preprocessor.fit_and_bucket(raw_logs)
    clusters = []
    bucket_sizes = []
    for bucket in buckets.values():
        bucket_sizes.append(len(bucket))
        clusters.extend(clusterer.cluster_bucket(bucket))
    templates = extractor.extract_templates(clusters)
    event_to_template = {f"E{index}": item["template"] for index, item in enumerate(templates, start=1)}
    pred_rows = []
    for cluster_index, cluster in enumerate(clusters, start=1):
        event_id = f"E{cluster_index}"
        template = event_to_template.get(event_id, "<*>")
        for item in cluster:
            row = rows[item["original_index"]]
            pred_rows.append({"LineId": row["LineId"], "EventTemplate": template})
    pred_rows.sort(key=lambda row: int(row["LineId"]))
    subset_gt_path = output_dir / "bgl_subset_gt.csv"
    _write_fixture(
        subset_gt_path,
        [{"LineId": row["LineId"], "EventTemplate": row["EventTemplate"]} for row in rows],
    )
    pred_path = output_dir / "bgl_subset_pred.csv"
    _write_fixture(pred_path, pred_rows)
    metrics = evaluate_prediction(subset_gt_path, pred_path)
    elapsed = time.perf_counter() - start
    result = {
        "run_id": "R004",
        "dataset": dataset,
        "subset_rows": subset_rows,
        "metrics": metrics,
        "bucket_count": len(buckets),
        "cluster_count": len(clusters),
        "clustered_bucket_count": sum(size > 0 for size in bucket_sizes),
        "max_bucket_size": max(bucket_sizes, default=0),
        "runtime_seconds": elapsed,
        "prediction_path": str(pred_path.resolve()),
    }
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_one(run_id: str, subset_rows: int = 300) -> dict[str, object]:
    output_dir = RUN_ROOT / run_id
    if run_id == "R000":
        return run_r000(output_dir)
    if run_id == "R000_VALIDATE":
        return run_r000_validation(RUN_ROOT / "R000")
    if run_id == "R000_REFRESH_CODE":
        return run_r000_refresh_code(RUN_ROOT / "R000")
    if run_id == "R001":
        return run_r001(output_dir)
    if run_id == "R002":
        return run_r002(output_dir)
    if run_id == "R003":
        return run_r003(output_dir)
    if run_id == "R004":
        return run_r004(output_dir, subset_rows=subset_rows)
    raise ValueError(f"Unknown B0 run: {run_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=["R000", "R000_VALIDATE", "R000_REFRESH_CODE", "R001", "R002", "R003", "R004", "all"], default="R000")
    parser.add_argument("--bgl-rows", type=int, default=300)
    args = parser.parse_args()
    run_ids = ["R000", "R001", "R002", "R003", "R004"] if args.run == "all" else [args.run]
    for run_id in run_ids:
        print(f"[B0] Running {run_id}...")
        result = run_one(run_id, subset_rows=args.bgl_rows)
        print(json.dumps({"run_id": run_id, "status": "PASS", "summary": result.get("metrics", result.get("files", []))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
