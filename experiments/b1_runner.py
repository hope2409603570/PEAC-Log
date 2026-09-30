"""Canonical M1/B1 runner for the four preprocessing/parameter quadrants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from experiments.b0_evaluator import evaluate_prediction, normalize_template
from experiments.b1_config import resolve_quadrant
from experiments.claim_scope_audit import make_protocol_manifest


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "result_runs" / "M1"
DATASETS = [
    "Proxifier", "Linux", "HDFS", "Mac", "Apache", "Hadoop", "BGL",
    "HealthApp", "HPC", "OpenSSH", "OpenStack", "Thunderbird", "Zookeeper", "Spark",
]
RUN_TO_QUADRANT = {"R010": "U-U", "R011": "T-T", "R012": "U-T", "R013": "T-U"}
STREAMING_SIZE_THRESHOLD_BYTES = 64 * 1024 * 1024
STREAMING_BATCH_SIZE = 32768


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def digest_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def dataset_path(dataset: str) -> Path:
    return ROOT / "data" / dataset / f"{dataset}_full.log_structured.csv"


def load_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    """Load full data or deterministic evenly spaced smoke rows."""

    if limit is None:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [
                {
                    "LineId": str(row["LineId"]),
                    "Content": row["Content"],
                    "EventTemplate": row["EventTemplate"],
                }
                for row in csv.DictReader(handle)
            ]

    if limit <= 0:
        raise ValueError("--rows must be positive")
    source_limit = max(limit, min(10000, limit * 40))
    target_indices = {
        round(index * (source_limit - 1) / max(limit - 1, 1))
        for index in range(limit)
    }
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index in target_indices:
                rows.append(
                    {
                        "LineId": str(row["LineId"]),
                        "Content": row["Content"],
                        "EventTemplate": row["EventTemplate"],
                    }
                )
            if index + 1 >= source_limit:
                break
    return rows


def _structured_row(row: dict[str, str], path: Path, row_number: int) -> dict[str, str]:
    """Validate and project one structured CSV row for the runner."""

    try:
        return {
            "LineId": str(row["LineId"]),
            "Content": row["Content"],
            "EventTemplate": row["EventTemplate"],
        }
    except KeyError as exc:
        raise ValueError(f"{path}:{row_number}: missing column {exc.args[0]!r}") from exc


def iter_structured_batches(
    path: Path,
    preprocessor: object,
    batch_size: int = STREAMING_BATCH_SIZE,
) -> Iterable[tuple[list[dict[str, str]], list[str]]]:
    """Yield bounded batches of rows and independently preprocessed contents."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path}: CSV has no header")
        rows: list[dict[str, str]] = []
        for row_number, raw_row in enumerate(reader, start=2):
            rows.append(_structured_row(raw_row, path, row_number))
            if len(rows) >= batch_size:
                processed = preprocessor([row["Content"] for row in rows])
                if len(processed) != len(rows):
                    raise RuntimeError(f"Preprocessor changed row count for {path}")
                yield rows, processed
                rows = []
        if rows:
            processed = preprocessor([row["Content"] for row in rows])
            if len(processed) != len(rows):
                raise RuntimeError(f"Preprocessor changed row count for {path}")
            yield rows, processed


def scan_stream(
    path: Path,
    preprocessor: object,
    batch_size: int = STREAMING_BATCH_SIZE,
) -> tuple[int, str, Counter[str]]:
    """Build global preprocessing frequencies without retaining the dataset."""

    row_count = 0
    frequency: Counter[str] = Counter()
    digest = hashlib.sha256()
    for rows, processed in iter_structured_batches(path, preprocessor, batch_size):
        row_count += len(rows)
        frequency.update(processed)
        for value in processed:
            digest.update(value.encode("utf-8"))
            digest.update(b"\n")
    return row_count, digest.hexdigest(), frequency


def stream_scan_cache_path(output_dir: Path) -> Path:
    return output_dir / "stream_scan_cache.json"


def _template_index(
    template_info: dict[str, tuple[set[str], int]],
) -> tuple[list[tuple[str, tuple[set[str], int]]], dict[str, tuple[int, ...]]]:
    """Index template candidates by token while preserving template order."""

    items = list(template_info.items())
    token_to_indices: defaultdict[str, list[int]] = defaultdict(list)
    for index, (_, (template_set, _)) in enumerate(items):
        for token in template_set:
            token_to_indices[token].append(index)
    return items, {token: tuple(indices) for token, indices in token_to_indices.items()}


def canonical_cache_key(value: str) -> str:
    """Canonicalize embedded numeric and path identifiers for cache lookup."""

    normalized_tokens = []
    for token in value.split():
        if "/" in token:
            token = "<*>"
        else:
            token = re.sub(r"\d+", "<*>", token)
        normalized_tokens.append(token)
    return " ".join(normalized_tokens)


def _cache_lookup(
    processed: str,
    fast_cache: dict[str, str],
    template_info: dict[str, tuple[set[str], int]],
    delta_l: int,
    early_stop: float,
    template_items: list[tuple[str, tuple[set[str], int]]] | None = None,
    token_to_indices: dict[str, tuple[int, ...]] | None = None,
    canonical_cache: dict[str, str] | None = None,
) -> tuple[str, bool]:
    """Resolve one preprocessed line through the trained cache/fallback."""

    if processed in fast_cache:
        return fast_cache[processed], True
    canonical = canonical_cache_key(processed) if canonical_cache is not None else None
    if canonical_cache is not None:
        if canonical in canonical_cache:
            event_id = canonical_cache[canonical]
            fast_cache[processed] = event_id
            return event_id, True
    token_set = set(processed.split())
    token_length = len(token_set)
    best_event, best_score = "E99", 0.0
    if template_items is None:
        template_items = list(template_info.items())
    if token_to_indices is None:
        candidate_indices: Iterable[int] = range(len(template_items))
    else:
        # Candidates with no shared token have Jaccard score zero and cannot
        # beat the E99 default, so they can be omitted without changing the
        # legacy tie-breaking behavior. Sorting restores template insertion
        # order after the inverted-index union.
        candidate_indices = sorted({
            index
            for token in token_set
            for index in token_to_indices.get(token, ())
        })
    for index in candidate_indices:
        candidate, (template_set, template_length) = template_items[index]
        if abs(token_length - template_length) > delta_l:
            continue
        union = token_length + template_length - len(token_set & template_set)
        score = len(token_set & template_set) / union if union else 0.0
        if score > best_score:
            best_event, best_score = candidate, score
            if score >= early_stop:
                break
    fast_cache[processed] = best_event
    if canonical_cache is not None and canonical is not None:
        canonical_cache.setdefault(canonical, best_event)
    return best_event, False


def write_stream_predictions(
    source_path: Path,
    prediction_path: Path,
    preprocessor: object,
    model: dict[str, object],
    delta_l: int,
    early_stop: float,
    batch_size: int = STREAMING_BATCH_SIZE,
) -> tuple[int, str, dict[str, int]]:
    """Predict in source order using bounded memory and an atomic output file."""

    clusters = model["clusters"]
    cluster_to_event = model["cluster_to_event"]
    event_to_template = model["event_to_template"]
    fast_cache: dict[str, str] = {}
    canonical_cache: dict[str, str] = {}
    for cluster_index, cluster in enumerate(clusters, start=1):
        event_id = cluster_to_event.get(cluster_index, "E99")
        for item in cluster:
            raw_log = item["raw_log"]
            fast_cache[raw_log] = event_id
            canonical_cache.setdefault(canonical_cache_key(raw_log), event_id)
    template_info = {
        event_id: (set(template.split()), len(set(template.split())))
        for event_id, template in event_to_template.items()
    }
    template_items, token_to_indices = _template_index(template_info)
    print(
        f"[M1] stream model templates={len(event_to_template)} "
        f"exact_cache={len(fast_cache)} canonical_cache={len(canonical_cache)}",
        flush=True,
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = prediction_path.with_name(prediction_path.name + ".partial")
    rows_written = 0
    digest = hashlib.sha256()
    hits = 0
    fallbacks = 0
    try:
        with partial_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["LineId", "Content", "EventId", "EventTemplate"])
            for rows, processed_batch in iter_structured_batches(source_path, preprocessor, batch_size):
                output_batch: list[tuple[str, str, str, str]] = []
                for row, processed in zip(rows, processed_batch):
                    digest.update(processed.encode("utf-8"))
                    digest.update(b"\n")
                    event_id, cache_hit = _cache_lookup(
                        processed,
                        fast_cache,
                        template_info,
                        delta_l,
                        early_stop,
                        template_items,
                        token_to_indices,
                        canonical_cache,
                    )
                    hits += int(cache_hit)
                    fallbacks += int(not cache_hit)
                    output_batch.append((
                        row["LineId"],
                        row["Content"],
                        event_id,
                        event_to_template.get(event_id, "<*>")
                    ))
                writer.writerows(output_batch)
                rows_written += len(output_batch)
        partial_path.replace(prediction_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return rows_written, digest.hexdigest(), {
        "cache_hits": hits,
        "fallbacks": fallbacks,
        "cache_entries": len(fast_cache),
        "canonical_cache_entries": len(canonical_cache),
    }


def _template_maps(clusters: list[list[dict[str, object]]], sample_size: int) -> tuple[dict[int, str], dict[str, str], list[dict[str, object]]]:
    from src.stage3_4_extraction import FastTemplateExtractor

    extractor = FastTemplateExtractor(sample_size=sample_size)
    template_results = extractor.extract_templates(clusters)
    unique_templates: dict[str, str] = {}
    cluster_to_event: dict[int, str] = {}
    event_to_template: dict[str, str] = {}
    for result in sorted(template_results, key=lambda item: item["log_count"], reverse=True):
        norm_key = normalize_template(result["template"])
        if norm_key not in unique_templates:
            event_id = f"E{len(unique_templates) + 1}"
            unique_templates[norm_key] = event_id
            event_to_template[event_id] = str(result["template"])
        cluster_to_event[int(result["cluster_id"])] = unique_templates[norm_key]
    return cluster_to_event, event_to_template, template_results


def fit_model(raw_logs: list[str], params: object) -> dict[str, object]:
    from src.stage1_preprocessing import SmartPreprocessor
    from src.stage2_lsh import AdaptiveLSHClusterer

    preprocessor = SmartPreprocessor(
        entropy_threshold=float(params.theta),
        max_check_positions=int(params.n_positions),
    )
    clusterer = AdaptiveLSHClusterer(min_similarity=float(params.tau))
    fit_start = time.perf_counter()
    buckets = preprocessor.fit_and_bucket(raw_logs)
    clusters: list[list[dict[str, object]]] = []
    clustered_bucket_count = 0
    for bucket in buckets.values():
        if bucket:
            clustered_bucket_count += 1
            clusters.extend(clusterer.cluster_bucket(bucket))
    cluster_to_event, event_to_template, template_results = _template_maps(
        clusters, int(params.sample_size)
    )
    return {
        "buckets": buckets,
        "clusters": clusters,
        "cluster_to_event": cluster_to_event,
        "event_to_template": event_to_template,
        "template_results": template_results,
        "fit_seconds": time.perf_counter() - fit_start,
        "clustered_bucket_count": clustered_bucket_count,
    }


def rows_from_clusters(
    rows: list[dict[str, str]],
    model: dict[str, object],
) -> list[dict[str, str]]:
    clusters = model["clusters"]
    cluster_to_event = model["cluster_to_event"]
    event_to_template = model["event_to_template"]
    predictions: list[dict[str, str]] = []
    for cluster_index, cluster in enumerate(clusters, start=1):
        event_id = cluster_to_event.get(cluster_index, "E99")
        template = event_to_template.get(event_id, "<*>")
        for item in cluster:
            row = rows[int(item["original_index"])]
            predictions.append(
                {
                    "LineId": row["LineId"],
                    "Content": row["Content"],
                    "EventId": event_id,
                    "EventTemplate": template,
                }
            )
    predictions.sort(key=lambda row: int(row["LineId"]))
    return predictions


def rows_from_cache(
    rows: list[dict[str, str]],
    preprocessed: list[str],
    train_data: list[str],
    model: dict[str, object],
    delta_l: int,
    early_stop: float,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Map full data through the legacy top-K cache protocol."""

    clusters = model["clusters"]
    cluster_to_event = model["cluster_to_event"]
    event_to_template = model["event_to_template"]
    fast_cache: dict[str, str] = {}
    canonical_cache: dict[str, str] = {}
    for cluster_index, cluster in enumerate(clusters, start=1):
        event_id = cluster_to_event.get(cluster_index, "E99")
        for item in cluster:
            raw_log = train_data[int(item["original_index"])]
            fast_cache[raw_log] = event_id
            canonical_cache.setdefault(canonical_cache_key(raw_log), event_id)

    template_info = {
        event_id: (set(template.split()), len(set(template.split())))
        for event_id, template in event_to_template.items()
    }
    template_items, token_to_indices = _template_index(template_info)
    hits = 0
    fallback = 0
    predictions: list[dict[str, str]] = []
    for index, processed in enumerate(preprocessed):
        event_id, cache_hit = _cache_lookup(
            processed,
            fast_cache,
            template_info,
            delta_l,
            early_stop,
            template_items,
            token_to_indices,
            canonical_cache,
        )
        if cache_hit:
            hits += 1
        else:
            fallback += 1
        predictions.append(
            {
                "LineId": rows[index]["LineId"],
                "Content": rows[index]["Content"],
                "EventId": event_id,
                "EventTemplate": event_to_template.get(event_id, "<*>")
            }
        )
    return predictions, {
        "cache_hits": hits,
        "fallbacks": fallback,
        "cache_entries": len(fast_cache),
        "canonical_cache_entries": len(canonical_cache),
    }


def write_prediction(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["LineId", "Content", "EventId", "EventTemplate"])
        writer.writeheader()
        writer.writerows(rows)


def write_groundtruth(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["LineId", "Content", "EventId", "EventTemplate"])
        writer.writeheader()
        writer.writerows(rows)


def run_dataset(
    dataset: str,
    quadrant: str,
    output_dir: Path,
    rows_limit: int | None = None,
    cache_threshold: int = 50000,
) -> dict[str, object]:
    config = resolve_quadrant(dataset, quadrant)
    params = config["params"]
    config_payload = {
        "quadrant": quadrant,
        "dataset": dataset,
        "preprocess_mode": config["preprocess_mode"],
        "parameter_mode": config["parameter_mode"],
        "params": params.as_dict(),
        "rows_limit": rows_limit,
        "cache_threshold": cache_threshold,
    }
    config_hash = stable_hash(config_payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = dataset_path(dataset)
    preprocessor = config["preprocessor"]
    cache_stats: dict[str, int] = {"cache_hits": 0, "fallbacks": 0, "cache_entries": 0}
    sampling_protocol = "full_parse"
    scan_cache_reused = False
    prediction_rows: list[dict[str, str]] | None = None
    preprocessed_sha256: str
    row_count: int
    streaming_path = rows_limit is None and source_path.stat().st_size > STREAMING_SIZE_THRESHOLD_BYTES
    if streaming_path:
        # Large LogHub files are processed in two bounded-memory passes.  The
        # first pass is the declared global-frequency Top-K training protocol;
        # the second writes predictions in source order for streaming eval.
        sampling_protocol = "global_frequency_top_k_non_online"
        scan_cache_path = stream_scan_cache_path(output_dir)
        source_stat = source_path.stat()
        cached_scan: dict[str, object] | None = None
        if scan_cache_path.exists():
            try:
                candidate = json.loads(scan_cache_path.read_text(encoding="utf-8"))
                if (
                    candidate.get("config_hash") == config_hash
                    and candidate.get("source_size") == source_stat.st_size
                    and candidate.get("source_mtime_ns") == source_stat.st_mtime_ns
                    and isinstance(candidate.get("train_data"), list)
                ):
                    cached_scan = candidate
            except (OSError, json.JSONDecodeError):
                cached_scan = None
        if cached_scan is not None:
            row_count = int(cached_scan["row_count"])
            preprocessed_sha256 = str(cached_scan["preprocessed_sha256"])
            train_data = [str(item) for item in cached_scan["train_data"]]
            preprocess_seconds = float(cached_scan.get("scan_seconds", 0.0))
            scan_cache_reused = True
            print(f"[M1] reusing stream scan cache {scan_cache_path}", flush=True)
        else:
            scan_start = time.perf_counter()
            row_count, preprocessed_sha256, frequency = scan_stream(source_path, preprocessor)
            preprocess_seconds = time.perf_counter() - scan_start
            if row_count == 0:
                raise ValueError(f"No rows loaded for {dataset}")
            train_data = [item[0] for item in frequency.most_common(min(params.top_k, len(frequency)))]
            scan_cache_path.write_text(
                json.dumps(
                    {
                        "config_hash": config_hash,
                        "source_size": source_stat.st_size,
                        "source_mtime_ns": source_stat.st_mtime_ns,
                        "row_count": row_count,
                        "preprocessed_sha256": preprocessed_sha256,
                        "train_data": train_data,
                        "scan_seconds": preprocess_seconds,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            del frequency
        model = fit_model(train_data, params)
        prediction_path = output_dir / f"{dataset.lower()}_{quadrant.replace('-', '')}_{config_hash}_predictions.csv"
        infer_start = time.perf_counter()
        written_rows, second_pass_digest, cache_stats = write_stream_predictions(
            source_path,
            prediction_path,
            preprocessor,
            model,
            params.delta_l,
            params.early_stop,
        )
        inference_seconds = time.perf_counter() - infer_start
        if written_rows != row_count:
            raise RuntimeError(
                f"Streaming row count changed for {dataset}: first={row_count}, second={written_rows}"
            )
        if second_pass_digest != preprocessed_sha256:
            raise RuntimeError(f"Streaming preprocessing digest changed between passes for {dataset}")
        load_seconds = 0.0
    else:
        load_start = time.perf_counter()
        rows = load_rows(source_path, rows_limit)
        load_seconds = time.perf_counter() - load_start
        if not rows:
            raise ValueError(f"No rows loaded for {dataset}")

        preprocess_start = time.perf_counter()
        preprocessed = preprocessor([row["Content"] for row in rows])
        preprocess_seconds = time.perf_counter() - preprocess_start
        if len(preprocessed) != len(rows):
            raise RuntimeError(f"Preprocessor changed row count for {dataset}")

        use_cache_protocol = rows_limit is None and len(rows) > cache_threshold
        if use_cache_protocol:
            sampling_protocol = "global_frequency_top_k_non_online"
            frequency = Counter(preprocessed)
            train_data = [item[0] for item in frequency.most_common(min(params.top_k, len(frequency)))]
            model = fit_model(train_data, params)
            infer_start = time.perf_counter()
            prediction_rows, cache_stats = rows_from_cache(
                rows, preprocessed, train_data, model, params.delta_l, params.early_stop
            )
            inference_seconds = time.perf_counter() - infer_start
        else:
            model = fit_model(preprocessed, params)
            infer_start = time.perf_counter()
            prediction_rows = rows_from_clusters(rows, model)
            inference_seconds = time.perf_counter() - infer_start

        prediction_path = output_dir / f"{dataset.lower()}_{quadrant.replace('-', '')}_{config_hash}_predictions.csv"
        write_prediction(prediction_path, prediction_rows)
        row_count = len(rows)
        preprocessed_sha256 = digest_strings(preprocessed)

    if streaming_path:
        use_cache_protocol = True
    else:
        # Keep the variable explicit for metadata even when full_parse is used.
        use_cache_protocol = rows_limit is None and row_count > cache_threshold

    if rows_limit is None:
        groundtruth_path = source_path
    else:
        groundtruth_path = output_dir / f"{dataset.lower()}_{rows_limit}_groundtruth.csv"
        write_groundtruth(groundtruth_path, rows)

    evaluation_start = time.perf_counter()
    metrics = evaluate_prediction(groundtruth_path, prediction_path)
    evaluation_seconds = time.perf_counter() - evaluation_start
    total_seconds = load_seconds + preprocess_seconds + model["fit_seconds"] + inference_seconds + evaluation_seconds
    metadata_record = {
        "run_id": output_dir.parent.name,
        "milestone": "M1",
        "dataset": dataset,
        "quadrant": quadrant,
        "config_hash": config_hash,
        "config": config_payload,
        "source_data": str(source_path.resolve()),
        "source_data_sha256": sha256_file(source_path) if rows_limit is None else None,
        "rows": row_count,
        "preprocessed_sha256": preprocessed_sha256,
        "sampling_protocol": sampling_protocol,
        "non_online_warning": use_cache_protocol,
        "protocol": make_protocol_manifest(
            preprocess_mode=str(config["preprocess_mode"]),
            parameter_mode=str(config["parameter_mode"]),
            sampling_protocol=sampling_protocol,
        ),
        "streaming": {
            "enabled": streaming_path,
            "batch_size": STREAMING_BATCH_SIZE if streaming_path else None,
            "size_threshold_bytes": STREAMING_SIZE_THRESHOLD_BYTES if streaming_path else None,
            "scan_cache_reused": scan_cache_reused,
        },
        "bucket_count": len(model["buckets"]),
        "cluster_count": len(model["clusters"]),
        "template_count": len(model["event_to_template"]),
        "clustered_bucket_count": model["clustered_bucket_count"],
        "cache": cache_stats,
        "timing_seconds": {
            "load": load_seconds,
            "preprocess": preprocess_seconds,
            "fit": model["fit_seconds"],
            "inference": inference_seconds,
            "evaluation": evaluation_seconds,
            "total": total_seconds,
        },
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "groundtruth_path": str(groundtruth_path.resolve()),
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata_record


def run_quadrant(
    run_id: str,
    datasets: list[str],
    rows_limit: int | None = None,
    cache_threshold: int = 50000,
) -> list[dict[str, object]]:
    quadrant = RUN_TO_QUADRANT[run_id]
    records: list[dict[str, object]] = []
    run_dir = RUN_ROOT / run_id
    for dataset in datasets:
        print(f"[M1] {run_id} {quadrant} {dataset} rows={rows_limit or 'full'}")
        dataset_dir = run_dir / dataset
        records.append(run_dataset(dataset, quadrant, dataset_dir, rows_limit, cache_threshold))
    # Rebuild the summary from every completed dataset metadata file so a
    # long 14-dataset run can be resumed in batches without losing prior rows.
    merged_records: list[dict[str, object]] = []
    for dataset_dir in sorted(run_dir.iterdir()) if run_dir.exists() else []:
        metadata_path = dataset_dir / "run_metadata.json"
        if metadata_path.exists():
            merged_records.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    records = merged_records
    summary_path = run_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "dataset", "quadrant", "config_hash", "rows", "PA", "GA", "FGA", "FTA", "bucket_count", "cluster_count", "template_count", "sampling_protocol", "total_seconds"]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metrics = record["metrics"]
            writer.writerow({
                "run_id": record["run_id"],
                "dataset": record["dataset"],
                "quadrant": record["quadrant"],
                "config_hash": record["config_hash"],
                "rows": record["rows"],
                "PA": metrics["PA"],
                "GA": metrics["GA"],
                "FGA": metrics["FGA"],
                "FTA": metrics["FTA"],
                "bucket_count": record["bucket_count"],
                "cluster_count": record["cluster_count"],
                "template_count": record["template_count"],
                "sampling_protocol": record["sampling_protocol"],
                "total_seconds": record["timing_seconds"]["total"],
            })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=sorted(RUN_TO_QUADRANT), required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS), help="Comma-separated dataset names")
    parser.add_argument("--rows", type=int, default=None, help="Deterministic smoke rows per dataset; omit for full")
    parser.add_argument("--cache-threshold", type=int, default=50000)
    args = parser.parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(datasets).difference(DATASETS))
    if unknown:
        raise SystemExit(f"Unknown datasets: {unknown}")
    records = run_quadrant(args.run, datasets, args.rows, args.cache_threshold)
    print(json.dumps({
        "run_id": args.run,
        "quadrant": RUN_TO_QUADRANT[args.run],
        "datasets": [record["dataset"] for record in records],
        "summary_path": str((RUN_ROOT / args.run / "summary.csv").resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
