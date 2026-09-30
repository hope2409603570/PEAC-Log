"""Frozen B1 quadrant configurations.

The U/T prefixes describe preprocessing (Uniform/Tuned) and the suffixes
describe parameters (Uniform/Tuned).  Dataset-specific preprocessing and
parameter values are copied from the current implementation and are retained
only for the T-* comparison cells.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from main import (
    apache_standard_preprocess,
    bgl_standard_preprocess,
    hadoop_standard_preprocess,
    hdfs_standard_preprocess,
    healthapp_standard_preprocess,
    hpc_standard_preprocess,
    linux_standard_preprocess,
    loghub_standard_preprocess,
    mac_standard_preprocess,
    openssh_standard_preprocess,
    openstack_standard_preprocess,
    spark_standard_preprocess,
    thunderbird_standard_preprocess,
    zookeeper_standard_preprocess,
)


@dataclass(frozen=True)
class ParserParams:
    theta: float
    n_positions: int
    tau: float
    sample_size: int = 50
    top_k: int = 8000
    delta_l: int = 12
    early_stop: float = 0.85

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


UNIFORM_PARAMS = ParserParams(
    theta=1.5,
    n_positions=12,
    tau=0.65,
    sample_size=50,
    top_k=8000,
    delta_l=12,
    early_stop=0.85,
)


TUNED_PARAMS: dict[str, ParserParams] = {
    "Proxifier": ParserParams(1.5, 10, 0.60),
    "Linux": ParserParams(1.5, 10, 0.60),
    "HDFS": ParserParams(1.0, 15, 0.85),
    "Mac": ParserParams(2.5, 12, 0.50),
    "Apache": ParserParams(1.5, 10, 0.60),
    "Hadoop": ParserParams(1.0, 15, 0.65),
    "BGL": ParserParams(1.5, 12, 0.88),
    "HealthApp": ParserParams(1.5, 10, 0.85),
    "HPC": ParserParams(1.5, 12, 0.65),
    "OpenSSH": ParserParams(1.5, 12, 0.65),
    "OpenStack": ParserParams(1.5, 12, 0.75),
    "Thunderbird": ParserParams(1.5, 12, 0.65),
    "Zookeeper": ParserParams(1.5, 12, 0.65),
    "Spark": ParserParams(1.5, 12, 0.65),
}


TUNED_PREPROCESSORS: dict[str, Callable[[list[str]], list[str]]] = {
    "Proxifier": loghub_standard_preprocess,
    "Linux": linux_standard_preprocess,
    "HDFS": hdfs_standard_preprocess,
    "Mac": mac_standard_preprocess,
    "Apache": apache_standard_preprocess,
    "Hadoop": hadoop_standard_preprocess,
    "BGL": bgl_standard_preprocess,
    "HealthApp": healthapp_standard_preprocess,
    "HPC": hpc_standard_preprocess,
    "OpenSSH": openssh_standard_preprocess,
    "OpenStack": openstack_standard_preprocess,
    "Thunderbird": thunderbird_standard_preprocess,
    "Zookeeper": zookeeper_standard_preprocess,
    "Spark": spark_standard_preprocess,
}


def resolve_quadrant(dataset: str, quadrant: str) -> dict[str, object]:
    """Resolve one of U-U, U-T, T-U or T-T into an immutable config."""

    if quadrant not in {"U-U", "U-T", "T-U", "T-T"}:
        raise ValueError(f"Unknown B1 quadrant: {quadrant}")
    if dataset not in TUNED_PARAMS:
        raise KeyError(f"Unknown dataset: {dataset}")
    preprocess_mode, parameter_mode = quadrant.split("-")
    params = UNIFORM_PARAMS if parameter_mode == "U" else TUNED_PARAMS[dataset]
    preprocessor = loghub_standard_preprocess if preprocess_mode == "U" else TUNED_PREPROCESSORS[dataset]
    return {
        "quadrant": quadrant,
        "dataset": dataset,
        "preprocess_mode": "uniform" if preprocess_mode == "U" else "tuned",
        "parameter_mode": "uniform" if parameter_mode == "U" else "tuned",
        "params": params,
        "preprocessor": preprocessor,
    }

