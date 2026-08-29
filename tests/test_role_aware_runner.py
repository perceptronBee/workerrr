from __future__ import annotations

import unittest
import json
from pathlib import Path

import torch

from worker.role_aware_runner import (
    COOLDOWN_START,
    EPOCHS,
    MATERIAL_PROBE_EPOCHS,
    PRIMARY_SPARSE_EPOCH,
    PROBE_EPOCHS,
    REQUIRED_GPU,
    SOURCE_COMMIT,
    SPARSE_STAGE_IDS,
    WARMUP_EPOCHS,
    load_pinned_halsp,
    open_sparse_modulation_gate,
    role_blind_scales,
    run_cpu_preflight,
    sparse_material_checkpoint,
)
from worker.worker import builtin_registry, validate_job_spec


HALSP_REPOSITORY = Path(__file__).resolve().parents[2] / "efficient_ai_test_repo"


class RoleAwareRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.halsp = load_pinned_halsp(HALSP_REPOSITORY)

    def test_frozen_schedule_and_source(self) -> None:
        self.assertEqual(SOURCE_COMMIT, "087725a74be5407d750c537ac701d82531c68a91")
        self.assertEqual(EPOCHS, 120)
        self.assertEqual(WARMUP_EPOCHS, 6)
        self.assertEqual(COOLDOWN_START - WARMUP_EPOCHS, 96)
        self.assertEqual(EPOCHS - COOLDOWN_START, 18)
        self.assertEqual(PROBE_EPOCHS, (6, 20, 40, 60, 80, 100))
        self.assertEqual(MATERIAL_PROBE_EPOCHS, (20, 40, 60, 80, 100))
        self.assertEqual(PRIMARY_SPARSE_EPOCH, 100)
        self.assertEqual(SPARSE_STAGE_IDS, (2, 3))
        self.assertEqual(REQUIRED_GPU, "NVIDIA L4")

    def test_actual_source_cpu_preflight(self) -> None:
        result = run_cpu_preflight(self.halsp)
        self.assertEqual(
            result,
            {
                "gate_one_equivalence": True,
                "signed_derivative_sum": True,
                "alpha_identity_and_mean": True,
                "modulated_probe_uses_effective_roles": True,
                "reserve_parameters_and_state_frozen": True,
                "exact_k": True,
            },
        )

    def test_role_blind_control_has_two_bounded_identity_dof(self) -> None:
        zero = torch.zeros(9, 2)
        weight, depthwise = role_blind_scales(zero)
        self.assertTrue(torch.equal(weight, torch.ones(9)))
        self.assertTrue(torch.equal(depthwise, torch.ones(9)))
        weight, depthwise = role_blind_scales(torch.tensor([[100.0, -100.0]]))
        self.assertGreaterEqual(float(torch.cat((weight, depthwise)).min()), 0.75)
        self.assertLessEqual(float(torch.cat((weight, depthwise)).max()), 1.25)

    def test_material_gate_uses_only_two_sparse_stages_and_midpoint_median(self) -> None:
        rows = [
            {"stage": 0, "conflict": 1.0, "swap_fraction": 1.0, "role_activity": 1.0},
            {"stage": 1, "conflict": 1.0, "swap_fraction": 1.0, "role_activity": 1.0},
            {"stage": 2, "conflict": 0.20, "swap_fraction": 0.10, "role_activity": 1.0},
            {"stage": 3, "conflict": 0.30, "swap_fraction": 0.122, "role_activity": 1.0},
        ]
        self.assertTrue(sparse_material_checkpoint(rows))
        rows[3]["swap_fraction"] = 0.12
        self.assertFalse(sparse_material_checkpoint(rows))

    def test_zero_activity_never_opens_material_gate(self) -> None:
        rows = [
            {"stage": 2, "conflict": 1.0, "swap_fraction": 1.0, "role_activity": 0.0},
            {"stage": 3, "conflict": 1.0, "swap_fraction": 1.0, "role_activity": 0.0},
        ]
        self.assertFalse(sparse_material_checkpoint(rows))

    def test_stage_b_requires_both_seeds_three_of_five_and_epoch100(self) -> None:
        self.assertTrue(
            open_sparse_modulation_gate(
                {0: [True, True, False, False, True], 1: [True, False, True, False, True]}
            )
        )
        self.assertFalse(
            open_sparse_modulation_gate(
                {0: [True, True, True, False, False], 1: [True, False, True, False, True]}
            )
        )

    def test_frozen_job_is_valid_and_only_fixed_handlers_are_used(self) -> None:
        job_path = Path(__file__).resolve().parents[1] / "jobs" / "role_aware_v1" / "job.json"
        job = validate_job_spec(json.loads(job_path.read_text(encoding="utf-8")))
        self.assertEqual(job["source"]["commit"], SOURCE_COMMIT)
        self.assertEqual([step["handler"] for step in job["steps"]], ["prepare_cifar100", "role_aware_v1"])
        self.assertTrue({"prepare_cifar100", "role_aware_v1"}.issubset(builtin_registry().names))


if __name__ == "__main__":
    unittest.main()
