"""Content-pinned, cache-first CIFAR-100 preparation for the Colab worker.

The data source is intentionally not configurable by a remote job.  Changing
the repository, configuration, or revision requires a reviewed code change and
a new worker commit.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DATASET_REPO = "uoft-cs/cifar100"
DATASET_CONFIG = "cifar100"
DATASET_REVISION = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"
EXPECTED_SPLITS = {"train": 50_000, "test": 10_000}
EXPECTED_COLUMNS = {"img", "fine_label", "coarse_label"}
EXPECTED_FINE_CLASSES = 100
EXPECTED_COARSE_CLASSES = 20
MANIFEST_SCHEMA_VERSION = 1


class DatasetIdentityError(RuntimeError):
    """Raised when cached or downloaded data does not match frozen identity."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def frozen_dataset_spec() -> dict[str, Any]:
    """Return the only dataset identity accepted by this worker."""

    return {
        "repo": DATASET_REPO,
        "config": DATASET_CONFIG,
        "revision": DATASET_REVISION,
        "expected_splits": dict(EXPECTED_SPLITS),
        "expected_columns": sorted(EXPECTED_COLUMNS),
        "fine_classes": EXPECTED_FINE_CLASSES,
        "coarse_classes": EXPECTED_COARSE_CLASSES,
    }


def frozen_dataset_spec_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(frozen_dataset_spec()))


def assert_frozen_dataset_spec(candidate: Mapping[str, Any]) -> None:
    """Reject any job that tries to change the reviewed CIFAR-100 identity."""

    expected = frozen_dataset_spec()
    if dict(candidate) != expected:
        raise DatasetIdentityError(
            "Dataset specification is not the frozen uoft-cs/cifar100 revision"
        )


def _feature_names(feature: Any) -> list[str] | None:
    names = getattr(feature, "names", None)
    if names is None and isinstance(feature, Mapping):
        names = feature.get("names")
    return list(names) if names is not None else None


def validate_dataset_identity(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Validate split sizes, columns and label cardinalities without guessing."""

    if set(dataset.keys()) != set(EXPECTED_SPLITS):
        raise DatasetIdentityError(
            f"Unexpected splits: {sorted(dataset.keys())}; "
            f"expected {sorted(EXPECTED_SPLITS)}"
        )

    fingerprints: dict[str, str | None] = {}
    for split_name, expected_rows in EXPECTED_SPLITS.items():
        split = dataset[split_name]
        if len(split) != expected_rows:
            raise DatasetIdentityError(
                f"{split_name} has {len(split)} rows; expected {expected_rows}"
            )
        columns = set(getattr(split, "column_names", []))
        if columns != EXPECTED_COLUMNS:
            raise DatasetIdentityError(
                f"{split_name} columns are {sorted(columns)}; "
                f"expected {sorted(EXPECTED_COLUMNS)}"
            )
        features = getattr(split, "features", {})
        fine_names = _feature_names(features["fine_label"])
        coarse_names = _feature_names(features["coarse_label"])
        if fine_names is None or len(fine_names) != EXPECTED_FINE_CLASSES:
            raise DatasetIdentityError("fine_label must contain exactly 100 classes")
        if coarse_names is None or len(coarse_names) != EXPECTED_COARSE_CLASSES:
            raise DatasetIdentityError("coarse_label must contain exactly 20 classes")
        fingerprints[split_name] = getattr(split, "_fingerprint", None)

    label_names = _feature_names(dataset["train"].features["fine_label"])
    label_hash = sha256_bytes(canonical_json_bytes(label_names))
    return {
        "split_rows": dict(EXPECTED_SPLITS),
        "columns": sorted(EXPECTED_COLUMNS),
        "fingerprints": fingerprints,
        "fine_label_names_sha256": label_hash,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DatasetIdentityError(f"Manifest must be an object: {path}")
    return value


def verify_prepared_manifest(path: Path) -> dict[str, Any]:
    """Verify that a previously prepared directory belongs to this revision."""

    manifest_path = path / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise DatasetIdentityError(f"Missing prepared manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise DatasetIdentityError("Unsupported prepared manifest schema")
    if manifest.get("dataset_spec") != frozen_dataset_spec():
        raise DatasetIdentityError("Prepared dataset revision/config does not match")
    expected_hash = frozen_dataset_spec_sha256()
    if manifest.get("dataset_spec_sha256") != expected_hash:
        raise DatasetIdentityError("Prepared dataset specification hash mismatch")
    if manifest.get("split_rows") != EXPECTED_SPLITS:
        raise DatasetIdentityError("Prepared dataset split counts do not match")
    return manifest


def _load_hf_dataset(cache_dir: Path, *, local_only: bool) -> Any:
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:  # pragma: no cover - exercised on Colab
        raise RuntimeError(
            "The pinned 'datasets' dependency must be installed before preparation"
        ) from exc

    download_config = DownloadConfig(local_files_only=local_only)
    return load_dataset(
        DATASET_REPO,
        DATASET_CONFIG,
        revision=DATASET_REVISION,
        cache_dir=str(cache_dir),
        download_config=download_config,
    )


def prepare_cifar100(
    prepared_root: str | os.PathLike[str],
    hf_cache_dir: str | os.PathLike[str],
    *,
    allow_network_fallback: bool = True,
) -> dict[str, Any]:
    """Reuse verified prepared data, then HF cache, then the pinned Hub revision.

    The prepared target is content-addressed by the revision.  An existing but
    invalid target is never overwritten: the worker stops so provenance cannot
    silently drift.
    """

    root = Path(prepared_root).resolve()
    cache_dir = Path(hf_cache_dir).resolve()
    target = root / f"cifar100-{DATASET_REVISION}"
    if target.exists():
        manifest = verify_prepared_manifest(target)
        return {"path": str(target), "cache_source": "prepared", "manifest": manifest}

    root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_source = "huggingface_cache"
    try:
        dataset = _load_hf_dataset(cache_dir, local_only=True)
    except Exception as cache_error:
        if not allow_network_fallback:
            raise DatasetIdentityError(
                "Pinned CIFAR-100 was not available in the local Hugging Face cache"
            ) from cache_error
        cache_source = "huggingface_network"
        dataset = _load_hf_dataset(cache_dir, local_only=False)

    identity = validate_dataset_identity(dataset)
    temporary = root / f".{target.name}.preparing"
    if temporary.exists():
        raise DatasetIdentityError(
            f"Stale preparation directory exists; inspect it before retrying: {temporary}"
        )
    dataset.save_to_disk(str(temporary))

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_spec": frozen_dataset_spec(),
        "dataset_spec_sha256": frozen_dataset_spec_sha256(),
        "cache_source": cache_source,
        **identity,
    }
    manifest_path = temporary / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(target)
    verified = verify_prepared_manifest(target)
    return {"path": str(target), "cache_source": cache_source, "manifest": verified}


def load_prepared_cifar100(path: str | os.PathLike[str]) -> Any:
    """Load a prepared dataset only after checking its frozen manifest."""

    prepared = Path(path).resolve()
    verify_prepared_manifest(prepared)
    try:
        from datasets import load_from_disk
    except ImportError as exc:  # pragma: no cover - exercised on Colab
        raise RuntimeError("The pinned 'datasets' dependency is required") from exc
    dataset = load_from_disk(str(prepared))
    validate_dataset_identity(dataset)
    return dataset

