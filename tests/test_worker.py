from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from worker.prepare_cifar100 import frozen_dataset_spec
from worker.worker import (
    ArtifactVerificationError,
    HandlerRegistry,
    JobValidationError,
    VerifiedDirectorySink,
    advertised_head_for_commit,
    checkout_source,
    finalize_and_release,
    load_job_spec,
    run_steps,
    sha256_file,
    validate_job_spec,
)


COMMIT = "a" * 40


def valid_spec() -> dict:
    return {
        "schema_version": 1,
        "job_id": "night-audit",
        "source": {
            "repo_url": "https://github.com/sp4cing-itu/efficient_ai_test_repo",
            "commit": COMMIT,
        },
        "dataset": frozen_dataset_spec(),
        "max_runtime_seconds": 3600,
        "steps": [
            {
                "id": "prepare",
                "handler": "test_prepare",
                "depends_on": [],
                "when": None,
                "params": {"seed": 0},
            },
            {
                "id": "conditional",
                "handler": "test_conditional",
                "depends_on": ["prepare"],
                "when": {"step": "prepare", "field": "decision", "op": "eq", "value": "run"},
                "params": {},
            },
        ],
    }


class WorkerValidationTests(unittest.TestCase):
    def test_private_source_is_materialized_from_frozen_worker_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = checkout_source(
                {
                    "repo_url": "https://github.com/sp4cing-itu/efficient_ai_test_repo.git",
                    "commit": "087725a74be5407d750c537ac701d82531c68a91",
                },
                Path(directory) / "source",
            )
            self.assertEqual(manifest["transport"], "embedded_frozen_snapshot")
            self.assertEqual(
                manifest["source_file_sha256"],
                "1b1a1e8e3f0c10c1a592de6c5cc49f7784d11aaea0df0b8612b077a0a4801700",
            )

    def test_source_commit_resolves_through_an_advertised_head(self) -> None:
        advertised = (
            f"{'b' * 40}\trefs/heads/main\n"
            f"{COMMIT}\trefs/heads/zeta\n"
            f"{COMMIT}\trefs/heads/alpha\n"
        )
        self.assertEqual(
            advertised_head_for_commit(advertised, COMMIT),
            "refs/heads/alpha",
        )

    def test_source_commit_must_be_an_advertised_head_tip(self) -> None:
        with self.assertRaises(JobValidationError):
            advertised_head_for_commit(f"{'b' * 40}\trefs/heads/main\n", COMMIT)

    def test_requires_full_source_commit(self) -> None:
        spec = valid_spec()
        spec["source"]["commit"] = "main"
        with self.assertRaises(JobValidationError):
            validate_job_spec(spec)

    def test_rejects_non_public_github_source(self) -> None:
        spec = valid_spec()
        spec["source"]["repo_url"] = "file:///content/source"
        with self.assertRaises(JobValidationError):
            validate_job_spec(spec)

    def test_rejects_executable_payload_fields(self) -> None:
        for key in ("command", "shell", "script", "module", "entrypoint", "code"):
            with self.subTest(key=key):
                spec = valid_spec()
                spec["steps"][0]["params"] = {key: "anything"}
                with self.assertRaises(JobValidationError):
                    validate_job_spec(spec)

    def test_rejects_forward_dependency(self) -> None:
        spec = valid_spec()
        spec["steps"][0]["depends_on"] = ["conditional"]
        with self.assertRaises(JobValidationError):
            validate_job_spec(spec)

    def test_rejects_unregistered_handler_before_steps_run(self) -> None:
        spec = valid_spec()
        registry = HandlerRegistry()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(JobValidationError):
                run_steps(spec, registry, root, root / "src", root / "artifacts")

    def test_job_file_requires_matching_hash(self) -> None:
        spec = valid_spec()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "job.json"
            payload = json.dumps(spec).encode("utf-8")
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            loaded = load_job_spec(path, expected)
            self.assertEqual(loaded["source"]["commit"], COMMIT)
            with self.assertRaises(JobValidationError):
                load_job_spec(path, "0" * 64)

    def test_finite_conditions_skip_without_executing_handler(self) -> None:
        events: list[str] = []
        registry = HandlerRegistry()
        registry.register("test_prepare", lambda _ctx, _params: {"decision": "stop"})
        registry.register("test_conditional", lambda _ctx, _params: events.append("ran") or {})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_dir = root / "artifacts"
            result = run_steps(valid_spec(), registry, root, root / "src", artifact_dir)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps"][1]["status"], "skipped")
        self.assertEqual(events, [])


class ArtifactLifecycleTests(unittest.TestCase):
    def test_verified_sink_precedes_runtime_release(self) -> None:
        events: list[str] = []

        class RecordingSink(VerifiedDirectorySink):
            def copy_and_verify(self, archive: Path, expected_sha256: str):
                events.append("copy")
                receipt = super().copy_and_verify(archive, expected_sha256)
                events.append("verify")
                return receipt

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_dir = root / "artifact"
            artifact_dir.mkdir()
            (artifact_dir / "result.json").write_text("{}\n", encoding="utf-8")
            receipt = finalize_and_release(
                artifact_dir,
                root / "result.tar.gz",
                RecordingSink(root / "durable"),
                lambda: events.append("unassign"),
            )
            self.assertTrue(receipt["verified"])
            self.assertEqual(receipt["sha256"], sha256_file(Path(receipt["path"])))
        self.assertEqual(events, ["copy", "verify", "unassign"])

    def test_runtime_is_released_if_sink_verification_fails(self) -> None:
        events: list[str] = []

        class BadSink:
            def copy_and_verify(self, archive: Path, expected_sha256: str):
                events.append("copy_failed")
                return {"verified": False, "sha256": expected_sha256}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_dir = root / "artifact"
            artifact_dir.mkdir()
            (artifact_dir / "failure.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ArtifactVerificationError):
                finalize_and_release(
                    artifact_dir,
                    root / "failed.tar.gz",
                    BadSink(),
                    lambda: events.append("unassign"),
                )
        self.assertEqual(events, ["copy_failed", "unassign"])


if __name__ == "__main__":
    unittest.main()
