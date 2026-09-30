"""Core PEAC-Log pipeline and backwards-compatible preprocessing API.

All dataset-specific regular expressions are defined in
``config/preprocessing_rules.json`` and loaded by
``src.preprocessing_rules``. Pipeline hyperparameters are stored separately in
``config/pipeline_parameters.json``.
"""

from __future__ import annotations

import time

from src.pipeline_config import get_pipeline_parameters
from src.preprocessing_rules import normalize_template_key, preprocess_logs
from src.stage1_preprocessing import SmartPreprocessor
from src.stage2_lsh import AdaptiveLSHClusterer
from src.stage3_4_extraction import FastTemplateExtractor


# These small wrappers preserve the public function names used by existing
# experiment scripts while keeping the rule definitions out of Python code.
def loghub_standard_preprocess(raw_logs):
    return preprocess_logs("loghub", raw_logs)


def linux_standard_preprocess(raw_logs):
    return preprocess_logs("linux", raw_logs)


def hdfs_standard_preprocess(raw_logs):
    return preprocess_logs("hdfs", raw_logs)


def mac_standard_preprocess(raw_logs):
    return preprocess_logs("mac", raw_logs)


def apache_standard_preprocess(raw_logs):
    return preprocess_logs("apache", raw_logs)


def hadoop_standard_preprocess(raw_logs):
    return preprocess_logs("hadoop", raw_logs)


def bgl_standard_preprocess(raw_logs):
    return preprocess_logs("bgl", raw_logs)


def healthapp_standard_preprocess(raw_logs):
    return preprocess_logs("healthapp", raw_logs)


def hpc_standard_preprocess(raw_logs):
    return preprocess_logs("hpc", raw_logs)


def openssh_standard_preprocess(raw_logs):
    return preprocess_logs("openssh", raw_logs)


def openstack_standard_preprocess(raw_logs):
    return preprocess_logs("openstack", raw_logs)


def thunderbird_standard_preprocess(raw_logs):
    return preprocess_logs("thunderbird", raw_logs)


def zookeeper_standard_preprocess(raw_logs):
    return preprocess_logs("zookeeper", raw_logs)


def spark_standard_preprocess(raw_logs):
    return preprocess_logs("spark", raw_logs)


class PEACLogPipeline:
    """Four-stage PEAC-Log pipeline configured by dataset name."""

    def __init__(self, dataset_name="Proxifier", pipeline_config_path=None):
        parameters = get_pipeline_parameters(dataset_name, pipeline_config_path)
        self.preprocessor = SmartPreprocessor(
            entropy_threshold=parameters.entropy_threshold,
            max_check_positions=parameters.max_check_positions,
        )
        self.clusterer = AdaptiveLSHClusterer(min_similarity=parameters.min_similarity)
        self.extractor = FastTemplateExtractor(sample_size=parameters.sample_size)

    def parse(self, raw_logs):
        start_time = time.time()
        print("=" * 50)
        print("Starting PEAC-Log core algorithm...")

        initial_buckets = self.preprocessor.fit_and_bucket(raw_logs)
        print(f"[OK] Stage 1: Successfully created {len(initial_buckets)} feature buckets.")

        all_refined_clusters = []
        for bucket_key, log_items in initial_buckets.items():
            if not log_items:
                continue
            refined_clusters = self.clusterer.cluster_bucket(log_items)
            all_refined_clusters.extend(refined_clusters)

        print(f"[OK] Stage 2: Initially divided into {len(all_refined_clusters)} independent clusters.")
        results = self.extractor.extract_templates(all_refined_clusters)

        unique_templates_map = {}
        cluster_to_event = {}
        event_to_template = {}
        event_counter = 1

        sorted_res = sorted(results, key=lambda item: item["log_count"], reverse=True)
        for result in sorted_res:
            normalized_key = normalize_template_key(result["template"])
            if normalized_key not in unique_templates_map:
                event_id = f"E{event_counter}"
                unique_templates_map[normalized_key] = event_id
                event_to_template[event_id] = result["template"]
                event_counter += 1
            cluster_to_event[result["cluster_id"]] = unique_templates_map[normalized_key]

        duration = time.time() - start_time
        print(f"[OK] Parsing complete! Identified {len(unique_templates_map)} core events.")
        return cluster_to_event, event_to_template, all_refined_clusters, duration


if __name__ == "__main__":
    pass
