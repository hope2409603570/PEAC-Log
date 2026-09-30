from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.claim_scope_audit import (
    assess_records,
    audit_manuscript,
    make_protocol_manifest,
)


class ClaimScopeAuditTests(unittest.TestCase):
    def test_tuned_replay_run_cannot_support_generalization_claims(self) -> None:
        protocol = make_protocol_manifest(
            preprocess_mode="tuned",
            parameter_mode="tuned",
            sampling_protocol="global_frequency_top_k_non_online",
        )
        report = assess_records([
            {
                "dataset": "BGL",
                "sampling_protocol": "global_frequency_top_k_non_online",
                "config": {"preprocess_mode": "tuned", "parameter_mode": "tuned"},
                "protocol": protocol,
            }
        ])
        verdict = report["claim_verdict"]
        self.assertTrue(verdict["dataset_aware_offline_batch_replay"])
        self.assertFalse(verdict["out_of_the_box_generalization"])
        self.assertFalse(verdict["zero_shot_transfer"])
        self.assertFalse(verdict["chronological_online_parsing"])
        self.assertTrue(report["uses_complete_file_cache_pretraining"])

    def test_shared_configuration_is_still_not_zero_shot(self) -> None:
        protocol = make_protocol_manifest(
            preprocess_mode="uniform",
            parameter_mode="uniform",
            sampling_protocol="full_parse",
        )
        report = assess_records([
            {
                "dataset": "Proxifier",
                "sampling_protocol": "full_parse",
                "config": {"preprocess_mode": "uniform", "parameter_mode": "uniform"},
                "protocol": protocol,
            }
        ])
        self.assertTrue(report["uses_target_file_for_template_generation"])
        self.assertFalse(report["claim_verdict"]["zero_shot_transfer"])

    def test_manuscript_audit_accepts_explicit_scope_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.tex"
            path.write_text(
                "These results do not establish out-of-the-box or zero-shot generalization.",
                encoding="utf-8",
            )
            self.assertTrue(audit_manuscript(path)["pass"])

    def test_manuscript_audit_accepts_negated_method_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.tex"
            path.write_text(
                "PEAC-Log is not generalizable or out-of-the-box on new systems.",
                encoding="utf-8",
            )
            self.assertTrue(audit_manuscript(path)["pass"])

    def test_manuscript_audit_rejects_affirmative_generalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.tex"
            path.write_text(
                "PEAC-Log demonstrates strong generalization across systems.",
                encoding="utf-8",
            )
            self.assertFalse(audit_manuscript(path)["pass"])

    def test_manuscript_audit_rejects_generalizable_wording(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.tex"
            path.write_text(
                "PEAC-Log is generalizable and works out-of-the-box on new systems.",
                encoding="utf-8",
            )
            report = audit_manuscript(path)
            self.assertFalse(report["pass"])
            self.assertIn("generalization", report["affirmative_scope_claims"])
            self.assertIn("out_of_box", report["affirmative_scope_claims"])


if __name__ == "__main__":
    unittest.main()
