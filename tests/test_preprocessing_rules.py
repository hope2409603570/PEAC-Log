from __future__ import annotations

import json
from pathlib import Path

import pytest

from main import (
    PEACLogPipeline,
    linux_standard_preprocess,
    loghub_standard_preprocess,
)
from src.pipeline_config import get_pipeline_parameters
from src.preprocessing_rules import (
    available_datasets,
    load_rule_config,
    normalize_template_key,
    preprocess_logs,
)


EXPECTED_DATASETS = {
    "apache",
    "bgl",
    "hadoop",
    "hdfs",
    "healthapp",
    "hpc",
    "linux",
    "loghub",
    "mac",
    "openssh",
    "openstack",
    "spark",
    "thunderbird",
    "zookeeper",
}


def test_default_rule_file_is_complete_and_valid() -> None:
    config = load_rule_config()
    assert set(available_datasets()) == EXPECTED_DATASETS
    assert config.default_dataset == "loghub"
    assert all(config.datasets[name].rules for name in EXPECTED_DATASETS)


def test_ordered_rules_preserve_special_case_behavior() -> None:
    raw = "proxy.example.com:8080 12:34 10 MB (IPv6)"
    assert preprocess_logs("Proxifier", [raw]) == ["<*> <*> <*> <*>"]
    assert loghub_standard_preprocess([raw]) == preprocess_logs("loghub", [raw])
    assert preprocess_logs("zookeeper", ["shutting down 123"]) == ["Shutting down <*>"]
    assert preprocess_logs("mac", ["Wake reason: EC.DarkPME 123"]) == ["Wake reason: <*>"]


def test_compatibility_wrapper_uses_json_rules() -> None:
    raw = "user alice (uid=42) value=abc ()"
    assert linux_standard_preprocess([raw]) == preprocess_logs("linux", [raw])
    assert linux_standard_preprocess([raw]) == ["user <*> <*> value=<*> (<*>)"]


def test_unknown_dataset_uses_configured_default() -> None:
    raw = "server.example:9000"
    assert preprocess_logs("unknown-dataset", [raw]) == preprocess_logs("loghub", [raw])


def test_template_normalization_is_configuration_driven() -> None:
    assert normalize_template_key("Task [<*>]: Done!") == "task*done"


def test_pipeline_parameters_replace_dataset_branches() -> None:
    hdfs = get_pipeline_parameters("HDFS")
    assert hdfs.entropy_threshold == 1.0
    assert hdfs.max_check_positions == 15
    assert hdfs.min_similarity == 0.85
    pipeline = PEACLogPipeline("HDFS")
    assert pipeline.preprocessor.entropy_threshold == hdfs.entropy_threshold
    assert pipeline.clusterer.min_distance == pytest.approx(1.0 - hdfs.min_similarity)


def test_invalid_regex_is_rejected_with_location(tmp_path: Path) -> None:
    invalid_config = {
        "schema_version": 1,
        "default_dataset": "demo",
        "aliases": {},
        "datasets": {
            "demo": {
                "strip": True,
                "rules": [{"name": "broken", "pattern": "(", "replacement": ""}],
            }
        },
        "template_normalization": {"rules": [], "lowercase": True, "strip": False},
    }
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(invalid_config), encoding="utf-8")

    with pytest.raises(ValueError, match=r"datasets\.demo\.rules\[0\]"):
        load_rule_config(config_path)
