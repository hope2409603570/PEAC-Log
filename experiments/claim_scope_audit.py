"""Derive defensible paper claims from PEAC-Log run metadata.

The current benchmark fits templates on each evaluated target dataset and may
also initialize a Top-K cache from the complete target file.  Consequently,
these runs are offline batch/replay experiments, even when preprocessing and
parser parameters are shared.  This module makes that scope machine-readable
and prevents tuned or replay runs from being described as out-of-the-box,
zero-shot, cross-system, or online evaluations.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Mapping


EXPECTED_DATASETS = {
    "Proxifier", "Linux", "HDFS", "Mac", "Apache", "Hadoop", "BGL",
    "HealthApp", "HPC", "OpenSSH", "OpenStack", "Thunderbird",
    "Zookeeper", "Spark",
}

AFFIRMATIVE_CLAIM_PATTERNS = {
    "out_of_box": re.compile(
        r"PEAC-Log[^.\n]{0,160}"
        r"(?:is|works?|operates?|provides?|achieves?|supports?|offers?)"
        r"[^.\n]{0,80}out[- ]of[- ]the[- ]box",
        re.IGNORECASE | re.DOTALL,
    ),
    "generalization": re.compile(
        r"PEAC-Log[^.\n]{0,160}(?:"
        r"generalizes?|generalizable|"
        r"(?:demonstrates?|proves?|shows?|supports?|achieves?|offers?)"
        r"[^.\n]{0,80}generaliz(?:ation|ability|able|es?)"
        r")",
        re.IGNORECASE | re.DOTALL,
    ),
    "zero_shot": re.compile(
        r"PEAC-Log[^.\n]{0,160}"
        r"(?:demonstrates?|supports?|achieves?|offers?|enables?)"
        r"[^.\n]{0,80}zero[- ]shot",
        re.IGNORECASE | re.DOTALL,
    ),
    "online": re.compile(
        r"PEAC-Log[^.\n]{0,160}"
        r"(?:achieves?|supports?|provides?|enables?|offers?)"
        r"[^.\n]{0,80}(?:online parsing|online throughput|online acceleration)",
        re.IGNORECASE | re.DOTALL,
    ),
}

NEGATION_MARKERS = re.compile(
    r"\b(?:not|no|never|without|cannot|can't|doesn't|does not|do not|"
    r"did not|rather than|outside)\b",
    re.IGNORECASE,
)


def make_protocol_manifest(
    *,
    preprocess_mode: str,
    parameter_mode: str,
    sampling_protocol: str,
) -> dict[str, object]:
    """Return the scope that follows from the implemented benchmark runner."""

    cache_pretraining = sampling_protocol == "global_frequency_top_k_non_online"
    return {
        "evaluation_mode": "offline_batch_replay",
        "dataset_specific_preprocessing": preprocess_mode == "tuned",
        "dataset_specific_parameters": parameter_mode == "tuned",
        "target_file_used_for_template_generation": True,
        "target_file_used_for_cache_initialization": cache_pretraining,
        "chronological_split": False,
        "held_out_target_system": False,
        "supported_claims": [
            "performance_under_the_declared_dataset_aware_recipe",
            "offline_source_order_replay_throughput",
        ],
        "unsupported_claims": [
            "out_of_the_box_generalization",
            "zero_shot_transfer",
            "cross_system_generalization",
            "chronological_online_parsing",
        ],
    }


def _infer_protocol(record: Mapping[str, object]) -> dict[str, object]:
    config = record.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    return make_protocol_manifest(
        preprocess_mode=str(config.get("preprocess_mode", "unknown")),
        parameter_mode=str(config.get("parameter_mode", "unknown")),
        sampling_protocol=str(record.get("sampling_protocol", "unknown")),
    )


def load_run_metadata(metadata_root: Path) -> list[dict[str, object]]:
    """Load every run_metadata.json beneath a run directory."""

    records = []
    for path in sorted(metadata_root.rglob("run_metadata.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"{path}: metadata root must be a JSON object")
        record["_metadata_path"] = str(path.resolve())
        records.append(record)
    if not records:
        raise FileNotFoundError(f"No run_metadata.json found below {metadata_root}")
    return records


def assess_records(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Summarize coverage and the strongest claims supported by the records."""

    materialized = list(records)
    datasets = {str(record.get("dataset", "")) for record in materialized}
    datasets.discard("")
    protocols = [
        record.get("protocol")
        if isinstance(record.get("protocol"), Mapping)
        else _infer_protocol(record)
        for record in materialized
    ]
    tuned_preprocessing = any(
        bool(protocol.get("dataset_specific_preprocessing")) for protocol in protocols
    )
    tuned_parameters = any(
        bool(protocol.get("dataset_specific_parameters")) for protocol in protocols
    )
    complete_file_cache = any(
        bool(protocol.get("target_file_used_for_cache_initialization"))
        for protocol in protocols
    )
    target_file_fit = any(
        bool(protocol.get("target_file_used_for_template_generation"))
        for protocol in protocols
    )
    return {
        "record_count": len(materialized),
        "datasets": sorted(datasets),
        "complete_14_dataset_coverage": datasets == EXPECTED_DATASETS,
        "uses_dataset_specific_preprocessing": tuned_preprocessing,
        "uses_dataset_specific_parameters": tuned_parameters,
        "uses_complete_file_cache_pretraining": complete_file_cache,
        "uses_target_file_for_template_generation": target_file_fit,
        "claim_verdict": {
            "dataset_aware_offline_batch_replay": True,
            "out_of_the_box_generalization": False,
            "zero_shot_transfer": False,
            "cross_system_generalization": False,
            "chronological_online_parsing": False,
        },
        "reason": (
            "The evaluated target file is used to generate templates"
            + (" and initialize the frequency cache" if complete_file_cache else "")
            + "; no held-out-system or chronological split is present."
        ),
    }


def audit_manuscript(path: Path) -> dict[str, object]:
    """Flag only direct affirmative PEAC-Log scope claims."""

    text = path.read_text(encoding="utf-8")
    findings = {}
    for name, pattern in AFFIRMATIVE_CLAIM_PATTERNS.items():
        matches = []
        for match in pattern.finditer(text):
            # The same terms are appropriate when explicitly negated, e.g.
            # "does not establish zero-shot transfer" or "not generalizable".
            if NEGATION_MARKERS.search(match.group(0)):
                continue
            matches.append(match.group(0).replace("\n", " "))
        if matches:
            findings[name] = matches
    return {
        "path": str(path.resolve()),
        "pass": not findings,
        "affirmative_scope_claims": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("result_runs") / "M1" / "R010",
    )
    parser.add_argument("--paper", type=Path, default=Path("paper") / "main" / "main.tex")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = assess_records(load_run_metadata(args.metadata_root))
    report["manuscript_audit"] = audit_manuscript(args.paper)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["manuscript_audit"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
