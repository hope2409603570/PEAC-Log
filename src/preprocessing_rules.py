"""JSON-driven, ordered regular-expression preprocessing.

Dataset-specific masking rules live in ``config/preprocessing_rules.json``.
Keeping them outside Python makes the experimental configuration inspectable,
versionable, and editable without changing parser code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Pattern


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES_PATH = PROJECT_ROOT / "config" / "preprocessing_rules.json"

_SUPPORTED_FLAGS = {
    "ASCII": re.ASCII,
    "DOTALL": re.DOTALL,
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "VERBOSE": re.VERBOSE,
}


@dataclass(frozen=True)
class CompiledRule:
    """One validated substitution rule."""

    name: str
    pattern: Pattern[str]
    replacement: str


@dataclass(frozen=True)
class DatasetRules:
    """Ordered rules and final text handling for a dataset."""

    rules: tuple[CompiledRule, ...]
    strip: bool


@dataclass(frozen=True)
class RuleConfig:
    """Fully validated preprocessing configuration."""

    default_dataset: str
    aliases: Mapping[str, str]
    datasets: Mapping[str, DatasetRules]
    template_rules: tuple[CompiledRule, ...]
    template_lowercase: bool
    template_strip: bool


def _require_mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    return value


def _compile_rules(raw_rules: object, location: str) -> tuple[CompiledRule, ...]:
    if not isinstance(raw_rules, list):
        raise ValueError(f"{location} must be a JSON array")

    compiled_rules = []
    for index, raw_rule in enumerate(raw_rules):
        rule_location = f"{location}[{index}]"
        rule = _require_mapping(raw_rule, rule_location)
        name = rule.get("name", f"rule_{index + 1:02d}")
        pattern_text = rule.get("pattern")
        replacement = rule.get("replacement")
        raw_flags = rule.get("flags", [])

        if not isinstance(name, str) or not name:
            raise ValueError(f"{rule_location}.name must be a non-empty string")
        if not isinstance(pattern_text, str):
            raise ValueError(f"{rule_location}.pattern must be a string")
        if not isinstance(replacement, str):
            raise ValueError(f"{rule_location}.replacement must be a string")
        if not isinstance(raw_flags, list) or not all(isinstance(item, str) for item in raw_flags):
            raise ValueError(f"{rule_location}.flags must be an array of strings")

        unknown_flags = sorted(set(raw_flags) - set(_SUPPORTED_FLAGS))
        if unknown_flags:
            raise ValueError(f"{rule_location} uses unsupported flags: {unknown_flags}")
        flags = 0
        for flag_name in raw_flags:
            flags |= _SUPPORTED_FLAGS[flag_name]

        try:
            pattern = re.compile(pattern_text, flags)
            pattern.sub(replacement, "")
        except re.error as error:
            raise ValueError(f"Invalid regex in {rule_location} ({name}): {error}") from error
        compiled_rules.append(CompiledRule(name=name, pattern=pattern, replacement=replacement))

    return tuple(compiled_rules)


@lru_cache(maxsize=8)
def _load_config_cached(resolved_path: str) -> RuleConfig:
    path = Path(resolved_path)
    try:
        raw_config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Preprocessing rule file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in preprocessing rule file {path}: {error}") from error

    config = _require_mapping(raw_config, "root")
    if config.get("schema_version") != 1:
        raise ValueError("preprocessing_rules.json schema_version must be 1")

    raw_datasets = _require_mapping(config.get("datasets"), "datasets")
    datasets: dict[str, DatasetRules] = {}
    for raw_name, raw_dataset in raw_datasets.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("dataset names must be non-empty strings")
        name = raw_name.lower()
        dataset = _require_mapping(raw_dataset, f"datasets.{raw_name}")
        strip = dataset.get("strip", True)
        if not isinstance(strip, bool):
            raise ValueError(f"datasets.{raw_name}.strip must be a boolean")
        datasets[name] = DatasetRules(
            rules=_compile_rules(dataset.get("rules"), f"datasets.{raw_name}.rules"),
            strip=strip,
        )

    default_dataset = config.get("default_dataset")
    if not isinstance(default_dataset, str) or default_dataset.lower() not in datasets:
        raise ValueError("default_dataset must name an entry in datasets")
    default_dataset = default_dataset.lower()

    raw_aliases = _require_mapping(config.get("aliases", {}), "aliases")
    aliases: dict[str, str] = {}
    for raw_alias, raw_target in raw_aliases.items():
        if not isinstance(raw_alias, str) or not isinstance(raw_target, str):
            raise ValueError("alias names and targets must be strings")
        alias, target = raw_alias.lower(), raw_target.lower()
        if target not in datasets:
            raise ValueError(f"alias {raw_alias!r} targets unknown dataset {raw_target!r}")
        aliases[alias] = target

    normalization = _require_mapping(config.get("template_normalization", {}), "template_normalization")
    lowercase = normalization.get("lowercase", True)
    strip = normalization.get("strip", False)
    if not isinstance(lowercase, bool) or not isinstance(strip, bool):
        raise ValueError("template_normalization lowercase/strip values must be booleans")

    return RuleConfig(
        default_dataset=default_dataset,
        aliases=aliases,
        datasets=datasets,
        template_rules=_compile_rules(normalization.get("rules", []), "template_normalization.rules"),
        template_lowercase=lowercase,
        template_strip=strip,
    )


def load_rule_config(rules_path: str | Path | None = None) -> RuleConfig:
    """Load and validate a rule file, using a path-based cache."""

    path = Path(rules_path) if rules_path is not None else DEFAULT_RULES_PATH
    return _load_config_cached(str(path.resolve()))


def resolve_dataset_name(dataset_name: str, rules_path: str | Path | None = None) -> str:
    """Resolve case-insensitive aliases and fall back to the configured default."""

    config = load_rule_config(rules_path)
    name = dataset_name.strip().lower()
    name = config.aliases.get(name, name)
    return name if name in config.datasets else config.default_dataset


def preprocess_logs(
    dataset_name: str,
    raw_logs: Iterable[str],
    rules_path: str | Path | None = None,
) -> list[str]:
    """Apply the selected dataset's ordered substitutions to each log line."""

    config = load_rule_config(rules_path)
    resolved_name = resolve_dataset_name(dataset_name, rules_path)
    dataset = config.datasets[resolved_name]
    cleaned_logs = []
    for raw_log in raw_logs:
        cleaned = raw_log
        for rule in dataset.rules:
            cleaned = rule.pattern.sub(rule.replacement, cleaned)
        cleaned_logs.append(cleaned.strip() if dataset.strip else cleaned)
    return cleaned_logs


def normalize_template_key(template: str, rules_path: str | Path | None = None) -> str:
    """Normalize a template for stable event-ID deduplication."""

    config = load_rule_config(rules_path)
    normalized = template
    for rule in config.template_rules:
        normalized = rule.pattern.sub(rule.replacement, normalized)
    if config.template_strip:
        normalized = normalized.strip()
    if config.template_lowercase:
        normalized = normalized.lower()
    return normalized


def available_datasets(rules_path: str | Path | None = None) -> tuple[str, ...]:
    """Return the configured canonical dataset names."""

    return tuple(sorted(load_rule_config(rules_path).datasets))
