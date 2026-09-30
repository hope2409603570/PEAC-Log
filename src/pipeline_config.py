"""Validated JSON configuration for PEAC-Log pipeline hyperparameters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline_parameters.json"
_FIELDS = ("entropy_threshold", "max_check_positions", "min_similarity", "sample_size")


@dataclass(frozen=True)
class PipelineParameters:
    entropy_threshold: float
    max_check_positions: int
    min_similarity: float
    sample_size: int


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    return value


def _validate(values: Mapping[str, object], location: str) -> PipelineParameters:
    missing = [field for field in _FIELDS if field not in values]
    if missing:
        raise ValueError(f"{location} is missing fields: {missing}")

    entropy_threshold = values["entropy_threshold"]
    max_check_positions = values["max_check_positions"]
    min_similarity = values["min_similarity"]
    sample_size = values["sample_size"]
    if not isinstance(entropy_threshold, (int, float)) or entropy_threshold < 0:
        raise ValueError(f"{location}.entropy_threshold must be non-negative")
    if not isinstance(max_check_positions, int) or max_check_positions <= 0:
        raise ValueError(f"{location}.max_check_positions must be a positive integer")
    if not isinstance(min_similarity, (int, float)) or not 0 <= min_similarity <= 1:
        raise ValueError(f"{location}.min_similarity must be between 0 and 1")
    if not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError(f"{location}.sample_size must be a positive integer")

    return PipelineParameters(
        entropy_threshold=float(entropy_threshold),
        max_check_positions=max_check_positions,
        min_similarity=float(min_similarity),
        sample_size=sample_size,
    )


@lru_cache(maxsize=8)
def _load_config(resolved_path: str) -> tuple[Mapping[str, object], Mapping[str, object]]:
    path = Path(resolved_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Pipeline configuration not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in pipeline configuration {path}: {error}") from error

    config = _mapping(raw, "root")
    if config.get("schema_version") != 1:
        raise ValueError("pipeline_parameters.json schema_version must be 1")
    defaults = _mapping(config.get("default"), "default")
    _validate(defaults, "default")
    datasets = _mapping(config.get("datasets", {}), "datasets")
    for name, overrides in datasets.items():
        if not isinstance(name, str) or not name:
            raise ValueError("dataset names must be non-empty strings")
        override_mapping = _mapping(overrides, f"datasets.{name}")
        unknown = sorted(set(override_mapping) - set(_FIELDS))
        if unknown:
            raise ValueError(f"datasets.{name} has unknown fields: {unknown}")
        _validate({**defaults, **override_mapping}, f"datasets.{name}")
    return defaults, datasets


def get_pipeline_parameters(
    dataset_name: str,
    config_path: str | Path | None = None,
) -> PipelineParameters:
    """Return defaults merged with case-insensitive dataset overrides."""

    path = Path(config_path) if config_path is not None else DEFAULT_PIPELINE_CONFIG_PATH
    defaults, datasets = _load_config(str(path.resolve()))
    overrides = datasets.get(dataset_name.strip().lower(), {})
    return _validate({**defaults, **overrides}, f"resolved.{dataset_name}")
