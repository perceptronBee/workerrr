"""Fail-closed, data-only job runner for a manually started Colab runtime.

Remote JSON may choose only handlers compiled into this reviewed worker.  It
cannot supply commands, Python snippets, module names, entry points, or shell
arguments.  Both the worker itself (in the notebook) and experiment source are
checked out by immutable 40-character Git commit IDs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlsplit

try:
    from .prepare_cifar100 import (
        assert_frozen_dataset_spec,
        canonical_json_bytes,
        prepare_cifar100,
    )
except ImportError:  # pragma: no cover - permits direct execution in Colab
    from prepare_cifar100 import (  # type: ignore
        assert_frozen_dataset_spec,
        canonical_json_bytes,
        prepare_cifar100,
    )


SCHEMA_VERSION = 1
MAX_STEPS = 32
MAX_JOB_BYTES = 256 * 1024
MAX_PARAMS_BYTES = 64 * 1024
JOB_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}$")
STEP_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
HEAD_REF_PATTERN = re.compile(r"^refs/heads/[A-Za-z0-9][A-Za-z0-9._/-]*$")
DANGEROUS_KEYS = {
    "args",
    "argv",
    "binary",
    "callable",
    "cmd",
    "code",
    "command",
    "entry_point",
    "entrypoint",
    "eval",
    "exec",
    "executable",
    "function",
    "import",
    "module",
    "python",
    "script",
    "shell",
    "subprocess",
}
ALLOWED_OPERATORS = {"eq", "ne", "lt", "lte", "gt", "gte", "in"}


class JobValidationError(ValueError):
    """Raised before any expensive or executable job work begins."""


class ArtifactVerificationError(RuntimeError):
    """Raised when a supposedly durable artifact copy cannot be verified."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _assert_exact_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    extra = set(value) - allowed
    if extra:
        raise JobValidationError(f"Unsupported {label} fields: {sorted(extra)}")


def _assert_safe_json(value: Any, path: str = "params") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise JobValidationError(f"{path} contains a non-string key")
            if key.lower().replace("-", "_") in DANGEROUS_KEYS:
                raise JobValidationError(f"Executable field is forbidden: {path}.{key}")
            _assert_safe_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_safe_json(child, f"{path}[{index}]")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise JobValidationError(f"{path} contains a non-JSON value")


def validate_public_github_url(repo_url: str) -> str:
    if not isinstance(repo_url, str):
        raise JobValidationError("source.repo_url must be a string")
    parsed = urlsplit(repo_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise JobValidationError("Only public https://github.com owner/repo URLs are allowed")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parts[1] in {".", ".."}:
        raise JobValidationError("GitHub URL must identify exactly one owner/repository")
    repo = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    if not repo or parts[0] in {".", ".."}:
        raise JobValidationError("Invalid GitHub owner/repository")
    return f"https://github.com/{parts[0]}/{repo}.git"


def _validate_when(when: Any, dependencies: list[str], step_id: str) -> None:
    if when is None:
        return
    if not isinstance(when, Mapping):
        raise JobValidationError(f"{step_id}.when must be an object or null")
    _assert_exact_keys(when, {"step", "field", "op", "value"}, f"{step_id}.when")
    if when.get("step") not in dependencies:
        raise JobValidationError(f"{step_id}.when may reference only a direct dependency")
    field = when.get("field")
    if not isinstance(field, str) or not STEP_ID_PATTERN.fullmatch(field):
        raise JobValidationError(f"{step_id}.when.field must be a simple output field")
    if when.get("op") not in ALLOWED_OPERATORS:
        raise JobValidationError(f"{step_id}.when.op is not allowed")
    _assert_safe_json(when.get("value"), f"{step_id}.when.value")


def validate_job_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a finite job before source checkout or GPU work."""

    if not isinstance(spec, Mapping):
        raise JobValidationError("Job specification must be a JSON object")
    _assert_exact_keys(
        spec,
        {"schema_version", "job_id", "source", "dataset", "steps", "max_runtime_seconds"},
        "job",
    )
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise JobValidationError(f"schema_version must equal {SCHEMA_VERSION}")
    job_id = spec.get("job_id")
    if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
        raise JobValidationError("job_id has an invalid format")
    max_runtime = spec.get("max_runtime_seconds")
    if not isinstance(max_runtime, int) or isinstance(max_runtime, bool):
        raise JobValidationError("max_runtime_seconds must be an integer")
    if not 60 <= max_runtime <= 24 * 60 * 60:
        raise JobValidationError("max_runtime_seconds must be between 60 and 86400")

    source = spec.get("source")
    if not isinstance(source, Mapping):
        raise JobValidationError("source must be an object")
    _assert_exact_keys(source, {"repo_url", "commit"}, "source")
    repo_url = validate_public_github_url(source.get("repo_url"))
    commit = source.get("commit")
    if not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit):
        raise JobValidationError("source.commit must be a full 40-character Git commit")

    dataset = spec.get("dataset")
    if not isinstance(dataset, Mapping):
        raise JobValidationError("dataset must be an object")
    assert_frozen_dataset_spec(dataset)

    steps = spec.get("steps")
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
        raise JobValidationError(f"steps must contain between 1 and {MAX_STEPS} items")
    seen: set[str] = set()
    normalized_steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            raise JobValidationError(f"steps[{index}] must be an object")
        _assert_exact_keys(
            raw_step,
            {"id", "handler", "depends_on", "when", "params"},
            f"steps[{index}]",
        )
        step_id = raw_step.get("id")
        handler = raw_step.get("handler")
        dependencies = raw_step.get("depends_on")
        params = raw_step.get("params")
        if not isinstance(step_id, str) or not STEP_ID_PATTERN.fullmatch(step_id):
            raise JobValidationError(f"steps[{index}].id has an invalid format")
        if step_id in seen:
            raise JobValidationError(f"Duplicate step id: {step_id}")
        if not isinstance(handler, str) or not STEP_ID_PATTERN.fullmatch(handler):
            raise JobValidationError(f"{step_id}.handler has an invalid format")
        if not isinstance(dependencies, list) or not all(
            isinstance(dep, str) for dep in dependencies
        ):
            raise JobValidationError(f"{step_id}.depends_on must be a string list")
        if len(set(dependencies)) != len(dependencies):
            raise JobValidationError(f"{step_id}.depends_on contains duplicates")
        missing_or_forward = [dep for dep in dependencies if dep not in seen]
        if missing_or_forward:
            raise JobValidationError(
                f"{step_id} dependencies must reference earlier steps: {missing_or_forward}"
            )
        if not isinstance(params, Mapping):
            raise JobValidationError(f"{step_id}.params must be an object")
        _assert_safe_json(params, f"{step_id}.params")
        if len(canonical_json_bytes(params)) > MAX_PARAMS_BYTES:
            raise JobValidationError(f"{step_id}.params exceeds {MAX_PARAMS_BYTES} bytes")
        when = raw_step.get("when")
        _validate_when(when, dependencies, step_id)
        normalized_steps.append(
            {
                "id": step_id,
                "handler": handler,
                "depends_on": list(dependencies),
                "when": dict(when) if when is not None else None,
                "params": dict(params),
            }
        )
        seen.add(step_id)

    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "source": {"repo_url": repo_url, "commit": commit.lower()},
        "dataset": dict(dataset),
        "steps": normalized_steps,
        "max_runtime_seconds": max_runtime,
    }


def load_job_spec(path: str | os.PathLike[str], expected_sha256: str) -> dict[str, Any]:
    job_path = Path(path).resolve()
    if not SHA256_PATTERN.fullmatch(expected_sha256 or ""):
        raise JobValidationError("An exact 64-character job SHA-256 is required")
    if not job_path.is_file():
        raise JobValidationError(f"Job file does not exist: {job_path}")
    if job_path.stat().st_size > MAX_JOB_BYTES:
        raise JobValidationError(f"Job file exceeds {MAX_JOB_BYTES} bytes")
    actual = sha256_file(job_path)
    if actual.lower() != expected_sha256.lower():
        raise JobValidationError(f"Job SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    with job_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return validate_job_spec(value)


def advertised_head_for_commit(advertised: str, commit: str) -> str:
    """Select a deterministic advertised branch whose tip is the exact commit."""

    matches: list[str] = []
    for line in advertised.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        sha, ref = parts
        if (
            sha.lower() == commit.lower()
            and HEAD_REF_PATTERN.fullmatch(ref)
            and ".." not in ref
            and "//" not in ref
            and not ref.endswith("/")
        ):
            matches.append(ref)
    if not matches:
        raise JobValidationError(
            f"Requested commit {commit} is not the tip of an advertised branch"
        )
    return sorted(matches)[0]


def checkout_source(source: Mapping[str, str], destination: Path) -> dict[str, str]:
    """Checkout an immutable public GitHub commit without invoking a shell."""

    repo_url = validate_public_github_url(source.get("repo_url"))
    commit = source.get("commit", "")
    if not COMMIT_PATTERN.fullmatch(commit):
        raise JobValidationError("A full source commit is required")
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise JobValidationError(f"Source destination is not empty: {destination}")
    advertised = subprocess.run(
        ["git", "ls-remote", "--heads", repo_url],
        check=True,
        shell=False,
        capture_output=True,
        text=True,
    ).stdout
    remote_ref = advertised_head_for_commit(advertised, commit)
    destination.mkdir(parents=True, exist_ok=True)
    commands = [
        ["git", "init", "--quiet", str(destination)],
        ["git", "-C", str(destination), "remote", "add", "origin", repo_url],
        ["git", "-C", str(destination), "fetch", "--quiet", "--depth", "1", "origin", remote_ref],
        ["git", "-C", str(destination), "checkout", "--quiet", "--detach", commit],
    ]
    for command in commands:
        subprocess.run(command, check=True, shell=False)
    resolved = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        check=True,
        shell=False,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    if resolved != commit.lower():
        raise JobValidationError(f"Resolved source commit {resolved} != requested {commit}")
    tree = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD^{tree}"],
        check=True,
        shell=False,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    return {
        "repo_url": repo_url,
        "commit": resolved,
        "advertised_ref": remote_ref,
        "git_tree": tree,
    }


@dataclass(frozen=True)
class StepContext:
    job_spec: Mapping[str, Any]
    work_dir: Path
    source_dir: Path
    artifact_dir: Path
    outputs: Mapping[str, Mapping[str, Any]]


Handler = Callable[[StepContext, Mapping[str, Any]], Mapping[str, Any]]


class HandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, name: str, handler: Handler) -> None:
        if not STEP_ID_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid handler name: {name}")
        if name in self._handlers:
            raise ValueError(f"Handler already registered: {name}")
        self._handlers[name] = handler

    def resolve(self, name: str) -> Handler:
        try:
            return self._handlers[name]
        except KeyError as exc:
            raise JobValidationError(f"Job requested an unregistered handler: {name}") from exc

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._handlers)


def _condition_matches(condition: Mapping[str, Any] | None, outputs: Mapping[str, Any]) -> bool:
    if condition is None:
        return True
    upstream = outputs[condition["step"]]
    if condition["field"] not in upstream:
        raise JobValidationError(
            f"Condition field {condition['field']} missing from {condition['step']} output"
        )
    actual = upstream[condition["field"]]
    expected = condition["value"]
    operator = condition["op"]
    if operator == "eq":
        return actual == expected
    if operator == "ne":
        return actual != expected
    if operator == "lt":
        return actual < expected
    if operator == "lte":
        return actual <= expected
    if operator == "gt":
        return actual > expected
    if operator == "gte":
        return actual >= expected
    if operator == "in":
        if not isinstance(expected, list):
            raise JobValidationError("The 'in' operator requires a list value")
        return actual in expected
    raise JobValidationError(f"Unsupported condition operator: {operator}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def builtin_registry() -> HandlerRegistry:
    registry = HandlerRegistry()

    def prepare_dataset(context: StepContext, params: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {"allow_network_fallback"}
        _assert_exact_keys(params, allowed, "prepare_dataset.params")
        allow_network = params.get("allow_network_fallback", True)
        if not isinstance(allow_network, bool):
            raise JobValidationError("allow_network_fallback must be boolean")
        return prepare_cifar100(
            context.work_dir / "prepared_data",
            context.work_dir / "hf_cache",
            allow_network_fallback=allow_network,
        )

    registry.register("prepare_cifar100", prepare_dataset)
    try:
        from .role_aware_runner import register_handlers
    except ImportError:  # pragma: no cover - direct worker/ path in Colab
        from role_aware_runner import register_handlers  # type: ignore
    register_handlers(registry)
    return registry


def run_steps(
    spec: Mapping[str, Any],
    registry: HandlerRegistry,
    work_dir: Path,
    source_dir: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Run a validated, ordered DAG using only registered handler callables."""

    normalized = validate_job_spec(spec)
    for step in normalized["steps"]:
        registry.resolve(step["handler"])
    started_monotonic = time.monotonic()
    outputs: dict[str, Mapping[str, Any]] = {}
    step_records: list[dict[str, Any]] = []
    status = "completed"
    failure: dict[str, Any] | None = None
    for step in normalized["steps"]:
        if time.monotonic() - started_monotonic > normalized["max_runtime_seconds"]:
            status = "failed"
            failure = {"step": step["id"], "type": "TimeoutError", "message": "Runtime budget exceeded"}
            break
        record: dict[str, Any] = {
            "id": step["id"],
            "handler": step["handler"],
            "started_at_utc": utc_now(),
        }
        try:
            should_run = _condition_matches(step["when"], outputs)
            if not should_run:
                record.update({"status": "skipped", "finished_at_utc": utc_now()})
                outputs[step["id"]] = {"status": "skipped"}
                step_records.append(record)
                continue
            context = StepContext(
                job_spec=normalized,
                work_dir=work_dir,
                source_dir=source_dir,
                artifact_dir=artifact_dir,
                outputs=dict(outputs),
            )
            result = registry.resolve(step["handler"])(context, step["params"])
            if not isinstance(result, Mapping):
                raise TypeError("Handler output must be a JSON object")
            _assert_safe_json(result, f"{step['id']}.output")
            canonical_json_bytes(result)
            output = dict(result)
            outputs[step["id"]] = output
            _write_json(artifact_dir / "steps" / f"{step['id']}.json", output)
            record.update({"status": "completed", "finished_at_utc": utc_now()})
            step_records.append(record)
        except Exception as exc:  # preserve a terminal failure artifact
            status = "failed"
            failure = {
                "step": step["id"],
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            record.update({"status": "failed", "finished_at_utc": utc_now()})
            step_records.append(record)
            break
    return {
        "status": status,
        "started_at_utc": step_records[0]["started_at_utc"] if step_records else utc_now(),
        "finished_at_utc": utc_now(),
        "steps": step_records,
        "outputs": outputs,
        "failure": failure,
    }


class ArtifactSink(Protocol):
    """A durable sink must copy and independently hash-check the archive."""

    def copy_and_verify(self, archive: Path, expected_sha256: str) -> Mapping[str, Any]: ...


class VerifiedDirectorySink:
    """Reference sink for a persistent directory supplied by the caller.

    The notebook does not mount Google Drive.  A caller may point this at an
    already-mounted persistent directory, or replace it with a Drive API sink.
    """

    def __init__(self, target_dir: str | os.PathLike[str]) -> None:
        self.target_dir = Path(target_dir).resolve()

    def copy_and_verify(self, archive: Path, expected_sha256: str) -> Mapping[str, Any]:
        self.target_dir.mkdir(parents=True, exist_ok=True)
        destination = self.target_dir / archive.name
        shutil.copy2(archive, destination)
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise ArtifactVerificationError(
                f"Artifact sink hash mismatch: expected {expected_sha256}, got {actual}"
            )
        digest_path = destination.with_suffix(destination.suffix + ".sha256")
        digest_path.write_text(f"{actual}  {destination.name}\n", encoding="utf-8")
        return {
            "kind": "verified_directory",
            "path": str(destination),
            "sha256": actual,
            "size_bytes": destination.stat().st_size,
            "verified": True,
        }


def create_artifact_archive(artifact_dir: Path, output_path: Path) -> tuple[Path, str]:
    artifact_dir = artifact_dir.resolve()
    output_path = output_path.resolve()
    if not artifact_dir.is_dir():
        raise ArtifactVerificationError(f"Artifact directory is missing: {artifact_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as archive:
        for path in sorted(artifact_dir.rglob("*")):
            if path.is_symlink():
                raise ArtifactVerificationError(f"Symlinks are forbidden in artifacts: {path}")
            if path.is_file():
                archive.add(path, arcname=path.relative_to(artifact_dir))
    digest = sha256_file(output_path)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{digest}  {output_path.name}\n", encoding="utf-8"
    )
    return output_path, digest


def finalize_and_release(
    artifact_dir: Path,
    archive_path: Path,
    sink: ArtifactSink,
    unassign_runtime: Callable[[], None],
) -> Mapping[str, Any]:
    """Copy, verify, then release; release also runs after a failed copy attempt."""

    try:
        archive, digest = create_artifact_archive(artifact_dir, archive_path)
        receipt = dict(sink.copy_and_verify(archive, digest))
        if receipt.get("verified") is not True or receipt.get("sha256") != digest:
            raise ArtifactVerificationError("Artifact sink did not return a verified matching receipt")
        return receipt
    finally:
        # GPU release is deliberately the last lifecycle action.
        unassign_runtime()


def colab_runtime_unassign() -> None:
    """Release the active Colab runtime.  Import is delayed for offline tests."""

    try:
        from google.colab import runtime
    except ImportError as exc:  # pragma: no cover - only present in Colab
        raise RuntimeError("google.colab.runtime is unavailable") from exc
    runtime.unassign()


def execute_job(
    spec: Mapping[str, Any],
    registry: HandlerRegistry,
    work_dir: str | os.PathLike[str],
) -> tuple[Path, dict[str, Any]]:
    """Checkout exact source and run the finite DAG, preserving terminal state."""

    normalized = validate_job_spec(spec)
    root = Path(work_dir).resolve()
    source_dir = root / "source"
    artifact_dir = root / "artifacts" / normalized["job_id"]
    source_manifest = checkout_source(normalized["source"], source_dir)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    _write_json(artifact_dir / "job_spec.json", normalized)
    run_manifest = run_steps(normalized, registry, root, source_dir, artifact_dir)
    manifest = {
        "schema_version": 1,
        "job_id": normalized["job_id"],
        "job_spec_sha256": canonical_sha256(normalized),
        "source": source_manifest,
        **run_manifest,
    }
    _write_json(artifact_dir / "run_manifest.json", manifest)
    return artifact_dir, manifest


def run_lifecycle(
    spec: Mapping[str, Any],
    registry: HandlerRegistry,
    work_dir: str | os.PathLike[str],
    sink: ArtifactSink,
    unassign_runtime: Callable[[], None],
) -> dict[str, Any]:
    """Execute, preserve terminal evidence, verify the durable copy, then release.

    Runtime release is exactly-once even if validation, checkout, execution,
    packaging, copying, or sink verification fails.  A failed experiment is a
    terminal result, never silently promoted to scientific evidence.
    """

    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    released = False

    def release_once() -> None:
        nonlocal released
        if not released:
            released = True
            unassign_runtime()

    try:
        try:
            artifact_dir, manifest = execute_job(spec, registry, root)
        except Exception as exc:
            artifact_dir = root / "artifacts" / "terminal-worker-failure"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "schema_version": 1,
                "status": "failed",
                "finished_at_utc": utc_now(),
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            _write_json(artifact_dir / "run_manifest.json", manifest)
        archive_path = root / "archives" / f"{artifact_dir.name}.tar.gz"
        receipt = finalize_and_release(
            artifact_dir,
            archive_path,
            sink,
            release_once,
        )
        return {"manifest": manifest, "artifact_receipt": dict(receipt)}
    finally:
        release_once()
