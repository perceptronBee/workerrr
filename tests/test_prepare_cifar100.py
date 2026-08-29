from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from worker.prepare_cifar100 import (
    DATASET_CONFIG,
    DATASET_REPO,
    DATASET_REVISION,
    DatasetIdentityError,
    assert_frozen_dataset_spec,
    frozen_dataset_spec,
    frozen_dataset_spec_sha256,
    validate_dataset_identity,
    verify_prepared_manifest,
)


class FakeLabel:
    def __init__(self, count: int) -> None:
        self.names = [f"label-{index}" for index in range(count)]


class FakeSplit:
    def __init__(self, rows: int, fingerprint: str) -> None:
        self._rows = rows
        self._fingerprint = fingerprint
        self.column_names = ["img", "fine_label", "coarse_label"]
        self.features = {
            "img": object(),
            "fine_label": FakeLabel(100),
            "coarse_label": FakeLabel(20),
        }

    def __len__(self) -> int:
        return self._rows


class DatasetIdentityTests(unittest.TestCase):
    def test_frozen_huggingface_identity(self) -> None:
        spec = frozen_dataset_spec()
        self.assertEqual(spec["repo"], DATASET_REPO)
        self.assertEqual(spec["config"], DATASET_CONFIG)
        self.assertEqual(spec["revision"], DATASET_REVISION)
        self.assertEqual(DATASET_REVISION, "aadb3af77e9048adbea6b47c21a81e47dd092ae5")
        self.assertEqual(len(frozen_dataset_spec_sha256()), 64)

    def test_job_cannot_change_dataset_revision(self) -> None:
        candidate = frozen_dataset_spec()
        candidate["revision"] = "main"
        with self.assertRaises(DatasetIdentityError):
            assert_frozen_dataset_spec(candidate)

    def test_split_and_label_identity_is_checked(self) -> None:
        identity = validate_dataset_identity(
            {"train": FakeSplit(50_000, "train-fp"), "test": FakeSplit(10_000, "test-fp")}
        )
        self.assertEqual(identity["split_rows"], {"train": 50_000, "test": 10_000})
        self.assertEqual(identity["fingerprints"]["train"], "train-fp")
        with self.assertRaises(DatasetIdentityError):
            validate_dataset_identity(
                {"train": FakeSplit(49_999, "bad"), "test": FakeSplit(10_000, "test-fp")}
            )

    def test_prepared_manifest_is_content_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "schema_version": 1,
                "dataset_spec": frozen_dataset_spec(),
                "dataset_spec_sha256": frozen_dataset_spec_sha256(),
                "split_rows": {"train": 50_000, "test": 10_000},
            }
            (root / "dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertEqual(verify_prepared_manifest(root)["dataset_spec"], frozen_dataset_spec())
            manifest["dataset_spec_sha256"] = "0" * 64
            (root / "dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaises(DatasetIdentityError):
                verify_prepared_manifest(root)


if __name__ == "__main__":
    unittest.main()

