"""B0 evaluation primitives.

The legacy evaluator in ``utils/evaluate.py`` persists a global summary file
as a side effect.  This module is deliberately side-effect free: callers choose where
to persist a result, and every comparison is keyed by ``LineId``.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable, Mapping


class EvaluationError(ValueError):
    """Raised when ground truth and predictions cannot be aligned safely."""


class _OrderMismatch(Exception):
    """Internal signal to use the generic LineId dictionary fallback."""


def normalize_template(value: object) -> str:
    """Normalize LogHub templates using the existing project convention."""

    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"<(?:NUM|IP|HEX|.*?|\*)>", "*", text)
    text = re.sub(r"[^a-zA-Z0-9*]", "", text)
    return text.lower()


def read_template_rows(path: str | Path) -> tuple[dict[str, str], list[str]]:
    """Read ``LineId`` and ``EventTemplate`` keyed rows from a CSV file."""

    path = Path(path)
    values: dict[str, str] = {}
    order: list[str] = []
    duplicates: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise EvaluationError(f"{path}: CSV has no header")
        required = {"LineId", "EventTemplate"}
        missing_columns = required.difference(reader.fieldnames)
        if missing_columns:
            raise EvaluationError(
                f"{path}: missing required columns {sorted(missing_columns)}"
            )
        for row_number, row in enumerate(reader, start=2):
            line_id = str(row.get("LineId", ""))
            if not line_id.strip():
                raise EvaluationError(f"{path}:{row_number}: empty LineId")
            if line_id in values:
                duplicates.append(line_id)
            if not (row.get("EventTemplate") or "").strip():
                raise EvaluationError(f"{path}:{row_number}: empty EventTemplate")
            values[line_id] = normalize_template(row.get("EventTemplate"))
            order.append(line_id)
    if duplicates:
        sample = ", ".join(duplicates[:5])
        raise EvaluationError(f"{path}: duplicate LineId values (sample: {sample})")
    return values, order


def align_by_line_id(
    groundtruth_path: str | Path, prediction_path: str | Path
) -> tuple[list[str], list[str], list[str], dict[str, object]]:
    """Return aligned normalized labels and a machine-readable alignment report."""

    gt, gt_order = read_template_rows(groundtruth_path)
    pred, pred_order = read_template_rows(prediction_path)
    gt_ids = set(gt)
    pred_ids = set(pred)
    missing = sorted(gt_ids - pred_ids)
    extra = sorted(pred_ids - gt_ids)
    if missing or extra:
        raise EvaluationError(
            "LineId mismatch: "
            f"missing={len(missing)} (sample={missing[:5]}), "
            f"extra={len(extra)} (sample={extra[:5]})"
        )
    def order_digest(order: list[str]) -> str:
        digest = hashlib.sha256()
        for line_id in order:
            digest.update(line_id.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    return (
        gt_order,
        [gt[line_id] for line_id in gt_order],
        [pred[line_id] for line_id in gt_order],
        {
            "groundtruth_rows": len(gt_order),
            "prediction_rows": len(pred_order),
            "groundtruth_order_sha256": order_digest(gt_order),
            "prediction_order_sha256": order_digest(pred_order),
            "groundtruth_order_sample": gt_order[:3] if len(gt_order) <= 6 else gt_order[:3] + gt_order[-3:],
            "prediction_order_sample": pred_order[:3] if len(pred_order) <= 6 else pred_order[:3] + pred_order[-3:],
            "missing_line_ids": missing,
            "extra_line_ids": extra,
            "order_equal": gt_order == pred_order,
        },
    )


def _groups(labels: Iterable[str], line_ids: Iterable[str]) -> dict[str, frozenset[str]]:
    grouped: defaultdict[str, set[str]] = defaultdict(set)
    for label, line_id in zip(labels, line_ids):
        grouped[label].add(line_id)
    return {label: frozenset(ids) for label, ids in grouped.items()}


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_aligned(
    line_ids: Iterable[str], groundtruth: Iterable[str], prediction: Iterable[str]
) -> dict[str, float | int | str]:
    """Compute PA, GA, FGA and the declared template-level FTA metric.

    GA/FGA require exact message groups. FTA follows LogHub-2.0's template
    purity counting rule, using this project's normalization. It is not macro
    classification F1 and does not require complete-group equality.
    """

    ids = list(line_ids)
    gt = list(groundtruth)
    pred = list(prediction)
    if not (len(ids) == len(gt) == len(pred)):
        raise EvaluationError("Aligned metric inputs have different lengths")
    total = len(ids)
    if total == 0:
        raise EvaluationError("Cannot evaluate an empty fixture")
    if len(set(ids)) != total:
        raise EvaluationError("Duplicate LineId in aligned inputs")

    pa = sum(g == p for g, p in zip(gt, pred)) / total
    gt_groups = _groups(gt, ids)
    pred_groups = _groups(pred, ids)
    gt_group_sets = set(gt_groups.values())
    matched_groups = sum(group in gt_group_sets for group in pred_groups.values())
    matched_lines = sum(
        len(group) for group in pred_groups.values() if group in gt_group_sets
    )
    ga = matched_lines / total
    pga = matched_groups / len(pred_groups) if pred_groups else 0.0
    rga = matched_groups / len(gt_groups) if gt_groups else 0.0
    fga = _f1(pga, rga)

    gt_counts = Counter(gt)
    pred_counts = Counter(pred)
    contingency = Counter(zip(gt, pred))
    correct_templates = sum(
        contingency[(label, label)] == count
        for label, count in pred_counts.items()
    )
    pta = correct_templates / len(pred_groups)
    rta = correct_templates / len(gt_groups)
    fta = _f1(pta, rta)

    return {
        "TotalLogs": total,
        "GT_Clusters": len(gt_groups),
        "Pred_Clusters": len(pred_groups),
        "Matched_Clusters": matched_groups,
        "PA": pa,
        "GA": ga,
        "FGA": fga,
        "FTA": fta,
        "PTA": pta,
        "RTA": rta,
        "Matched_Templates": correct_templates,
        "FTA_definition": "loghub2_template_purity_f1_over_normalized_templates",
    }


def _order_digest(order: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line_id in order:
        digest.update(line_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stream_evaluate(
    groundtruth_path: str | Path, prediction_path: str | Path
) -> dict[str, object]:
    """Evaluate sorted files without retaining every LineId in memory."""

    gt_path = Path(groundtruth_path)
    pred_path = Path(prediction_path)
    with (
        gt_path.open("r", encoding="utf-8-sig", newline="") as gt_handle,
        pred_path.open("r", encoding="utf-8-sig", newline="") as pred_handle,
    ):
        gt_reader = csv.DictReader(gt_handle)
        pred_reader = csv.DictReader(pred_handle)
        required = {"LineId", "EventTemplate"}
        if not required.issubset(set(gt_reader.fieldnames or [])):
            raise EvaluationError(f"{gt_path}: missing required columns")
        if not required.issubset(set(pred_reader.fieldnames or [])):
            raise EvaluationError(f"{pred_path}: missing required columns")
        gt_iter = iter(gt_reader)
        pred_iter = iter(pred_reader)
        gt_row = next(gt_iter, None)
        pred_row = next(pred_iter, None)
        previous_id = None
        total = 0
        correct = 0
        gt_counts: Counter[str] = Counter()
        pred_counts: Counter[str] = Counter()
        contingency: Counter[tuple[str, str]] = Counter()
        first_ids: list[str] = []
        last_ids: deque[str] = deque(maxlen=3)
        gt_order_digest = hashlib.sha256()
        pred_order_digest = hashlib.sha256()
        while gt_row is not None or pred_row is not None:
            if gt_row is None or pred_row is None:
                raise _OrderMismatch()
            gt_id = str(gt_row.get("LineId", ""))
            pred_id = str(pred_row.get("LineId", ""))
            if not gt_id or gt_id != pred_id:
                raise _OrderMismatch()
            try:
                numeric_id = int(gt_id)
            except ValueError:
                raise _OrderMismatch()
            if previous_id is not None and numeric_id <= previous_id:
                raise _OrderMismatch()
            previous_id = numeric_id
            gt_order_digest.update(gt_id.encode("utf-8"))
            gt_order_digest.update(b"\n")
            pred_order_digest.update(pred_id.encode("utf-8"))
            pred_order_digest.update(b"\n")
            if not (gt_row.get("EventTemplate") or "").strip() or not (pred_row.get("EventTemplate") or "").strip():
                raise EvaluationError("Empty EventTemplate in evaluation input")
            gt_label = normalize_template(gt_row.get("EventTemplate"))
            pred_label = normalize_template(pred_row.get("EventTemplate"))
            total += 1
            correct += int(gt_label == pred_label)
            gt_counts[gt_label] += 1
            pred_counts[pred_label] += 1
            contingency[(gt_label, pred_label)] += 1
            if len(first_ids) < 3:
                first_ids.append(gt_id)
            last_ids.append(gt_id)
            gt_row = next(gt_iter, None)
            pred_row = next(pred_iter, None)
        if total == 0:
            raise EvaluationError("Cannot evaluate an empty fixture")

    exact_pairs = [
        (gt_label, pred_label, count)
        for (gt_label, pred_label), count in contingency.items()
        if count == gt_counts[gt_label] == pred_counts[pred_label]
    ]
    matched_groups = len(exact_pairs)
    matched_lines = sum(count for _, _, count in exact_pairs)
    ga = matched_lines / total
    pga = matched_groups / len(pred_counts) if pred_counts else 0.0
    rga = matched_groups / len(gt_counts) if gt_counts else 0.0
    fga = _f1(pga, rga)
    correct_templates = sum(
        contingency[(label, label)] == count
        for label, count in pred_counts.items()
    )
    pta = correct_templates / len(pred_counts)
    rta = correct_templates / len(gt_counts)
    fta = _f1(pta, rta)
    order_sample = first_ids if total <= 6 else first_ids + list(last_ids)
    metrics: dict[str, object] = {
        "TotalLogs": total,
        "GT_Clusters": len(gt_counts),
        "Pred_Clusters": len(pred_counts),
        "Matched_Clusters": matched_groups,
        "PA": correct / total,
        "GA": ga,
        "FGA": fga,
        "FTA": fta,
        "PTA": pta,
        "RTA": rta,
        "Matched_Templates": correct_templates,
        "FTA_definition": "loghub2_template_purity_f1_over_normalized_templates",
    }
    metrics["alignment"] = {
        "groundtruth_rows": total,
        "prediction_rows": total,
        "groundtruth_order_sha256": gt_order_digest.hexdigest(),
        "prediction_order_sha256": pred_order_digest.hexdigest(),
        "groundtruth_order_sample": order_sample,
        "prediction_order_sample": order_sample,
        "missing_line_ids": [],
        "extra_line_ids": [],
        "order_equal": True,
        "alignment_mode": "streaming_sorted_line_id",
    }
    return metrics


def evaluate_prediction(
    groundtruth_path: str | Path, prediction_path: str | Path
) -> dict[str, object]:
    """Align two CSVs by ``LineId`` and return metrics without writing files."""

    try:
        result = _stream_evaluate(groundtruth_path, prediction_path)
    except _OrderMismatch:
        ids, gt, pred, alignment = align_by_line_id(groundtruth_path, prediction_path)
        result = evaluate_aligned(ids, gt, pred)
        result["alignment"] = alignment
    result["groundtruth_path"] = str(Path(groundtruth_path).resolve())
    result["prediction_path"] = str(Path(prediction_path).resolve())
    return result
