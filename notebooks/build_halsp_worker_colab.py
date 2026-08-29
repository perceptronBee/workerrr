"""Build the uploadable GitHub-mediated Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def build() -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "HALSP_ROLE_AWARE_NIGHT_WORKER.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            markdown(
                """
                # HALSP GitHub night worker

                Codex does **not** connect to this Colab session. This notebook
                checks out one reviewed `perceptronBee/workerrr` commit, verifies
                the frozen job hash, writes heartbeat/terminal status back to that
                repository, stores the full verified archive in your Drive, and
                then releases the Colab GPU with `runtime.unassign()`.

                Before **Run all**:

                1. Select an **NVIDIA L4** runtime.
                2. In Colab Secrets add `WORKERRR_TOKEN`, grant notebook access,
                   and use a fine-grained token limited to this repository with
                   Contents read/write permission.
                3. Review the frozen commit/hash values in the configuration cell.

                The token is read only from Colab Secrets, never printed, saved to
                Drive, embedded in Git remotes, or passed to the HALSP source repo.
                """
            ),
            code(
                """
                WORKER_REPO_URL = "https://github.com/perceptronBee/workerrr.git"
                WORKER_REPO_SLUG = "perceptronBee/workerrr"
                STATUS_BRANCH = "night-worker"

                # Filled after the reviewed worker/job release commit is pushed.
                WORKER_COMMIT = "dc63ae9bf61758bb3933c9779f3c8886ac1f6c71"
                WORKER_TREE_SHA256 = "6ac18d4199135de0dbe97e927ac9859c8f023e14106484be2267d1475aed00a5"
                JOB_SPEC_RELPATH = "jobs/role_aware_v1/job.json"
                JOB_SPEC_SHA256 = "1d7e4275d0aeaaab5703e36d860f353f73d417e4342a81571a75d9e030a33f8f"

                DRIVE_ARTIFACT_DIR = "/content/drive/MyDrive/HALSP/role_aware_v1"
                WORK_ROOT = "/content/halsp-role-aware-v1"
                STATUS_PATH = "runtime/role_aware_v1/status.json"
                REQUIRED_GPU = "NVIDIA L4"
                HARD_DEADLINE_SECONDS = 12 * 60 * 60
                HEARTBEAT_SECONDS = 10 * 60
                FAILURE_GRACE_SECONDS = 10 * 60
                """
            ),
            code(
                """
                # One lifecycle cell: any setup/run/finalization exception reaches
                # the same runtime-release path. Do not add execution cells below.
                import base64
                import hashlib
                import json
                import re
                import subprocess
                import sys
                import tarfile
                import threading
                import time
                from datetime import datetime, timezone
                from pathlib import Path
                from urllib.parse import urlsplit

                import requests
                from google.colab import drive, runtime, userdata

                released = False
                terminal = threading.Event()
                reporter_lock = threading.Lock()
                release_lock = threading.Lock()

                def utc_now():
                    return datetime.now(timezone.utc).isoformat()

                def release_once():
                    global released
                    with release_lock:
                        if released:
                            return
                        runtime.unassign()
                        released = True

                def wait_failure_grace():
                    print(
                        "Failure evidence is saved. Keeping the runtime open for "
                        f"{FAILURE_GRACE_SECONDS // 60} minutes for inspection."
                    )
                    time.sleep(FAILURE_GRACE_SECONDS)

                watchdog = threading.Timer(HARD_DEADLINE_SECONDS, release_once)
                watchdog.daemon = True
                watchdog.start()

                token = None
                reporter = None
                try:
                    token = userdata.get("WORKERRR_TOKEN")
                    if not token:
                        raise RuntimeError("Missing Colab Secret: WORKERRR_TOKEN")

                    class GitHubReporter:
                        def __init__(self, repo, branch, path, bearer):
                            self.url = f"https://api.github.com/repos/{repo}/contents/{path}"
                            self.branch = branch
                            self.headers = {
                                "Authorization": f"Bearer {bearer}",
                                "Accept": "application/vnd.github+json",
                                "X-GitHub-Api-Version": "2022-11-28",
                            }

                        def put(self, payload, message):
                            raw = (json.dumps(payload, indent=2, sort_keys=True) + "\\n").encode()
                            with reporter_lock:
                                current = requests.get(
                                    self.url,
                                    headers=self.headers,
                                    params={"ref": self.branch, "_": time.time_ns()},
                                    timeout=30,
                                )
                                body = {
                                    "message": message,
                                    "branch": self.branch,
                                    "content": base64.b64encode(raw).decode(),
                                }
                                if current.status_code == 200:
                                    body["sha"] = current.json()["sha"]
                                elif current.status_code != 404:
                                    current.raise_for_status()
                                response = requests.put(
                                    self.url, headers=self.headers, json=body, timeout=30
                                )
                                response.raise_for_status()

                    reporter = GitHubReporter(
                        WORKER_REPO_SLUG, STATUS_BRANCH, STATUS_PATH, token
                    )
                    reporter.put(
                        {"state": "booting", "updated_at_utc": utc_now()},
                        "worker: booting",
                    )


                    gpu_name = subprocess.run(
                        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                        check=True, capture_output=True, text=True, shell=False,
                    ).stdout.strip()
                    if gpu_name != REQUIRED_GPU:
                        raise RuntimeError(
                            f"GPU mismatch: required {REQUIRED_GPU!r}, received {gpu_name!r}"
                        )

                    drive.mount("/content/drive", force_remount=False)
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--quiet", "datasets==3.6.0"],
                        check=True, shell=False,
                    )

                    COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
                    SHA_RE = re.compile(r"^[0-9a-fA-F]{64}$")

                    def sha256_file(path):
                        digest = hashlib.sha256()
                        with Path(path).open("rb") as handle:
                            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                                digest.update(chunk)
                        return digest.hexdigest()

                    parsed = urlsplit(WORKER_REPO_URL)
                    if parsed.scheme != "https" or parsed.hostname != "github.com":
                        raise RuntimeError("Worker repository must be public HTTPS GitHub")
                    if not COMMIT_RE.fullmatch(WORKER_COMMIT):
                        raise RuntimeError("WORKER_COMMIT is not frozen")
                    if not SHA_RE.fullmatch(WORKER_TREE_SHA256):
                        raise RuntimeError("WORKER_TREE_SHA256 is not frozen")
                    if not SHA_RE.fullmatch(JOB_SPEC_SHA256):
                        raise RuntimeError("JOB_SPEC_SHA256 is not frozen")

                    checkout = Path("/content/pinned-workerrr")
                    if checkout.exists():
                        raise RuntimeError(f"Refusing to overwrite {checkout}")
                    for command in (
                        ["git", "init", "--quiet", str(checkout)],
                        ["git", "-C", str(checkout), "config", "core.autocrlf", "false"],
                        ["git", "-C", str(checkout), "config", "core.eol", "lf"],
                        ["git", "-C", str(checkout), "remote", "add", "origin", WORKER_REPO_URL],
                        ["git", "-C", str(checkout), "fetch", "--quiet", "--depth", "1", "origin", WORKER_COMMIT],
                        ["git", "-C", str(checkout), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
                        ["git", "-C", str(checkout), "remote", "set-url", "--push", "origin", "DISABLED"],
                    ):
                        subprocess.run(command, check=True, shell=False)
                    resolved = subprocess.run(
                        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                        check=True, capture_output=True, text=True, shell=False,
                    ).stdout.strip()
                    if resolved.lower() != WORKER_COMMIT.lower():
                        raise RuntimeError("Worker commit mismatch")

                    tracked = subprocess.run(
                        ["git", "-C", str(checkout), "ls-files", "-z"],
                        check=True, capture_output=True, shell=False,
                    ).stdout.split(b"\\0")
                    tree_hash = hashlib.sha256()
                    for relative in sorted(item.decode() for item in tracked if item):
                        path = checkout / relative
                        blob = subprocess.run(
                            ["git", "-C", str(checkout), "cat-file", "blob", f"HEAD:{relative}"],
                            check=True, capture_output=True, shell=False,
                        ).stdout
                        if path.read_bytes() != blob:
                            raise RuntimeError(f"Checkout bytes differ from Git blob: {relative}")
                        tree_hash.update(relative.encode() + b"\\0")
                        tree_hash.update(hashlib.sha256(blob).digest())
                    if tree_hash.hexdigest() != WORKER_TREE_SHA256:
                        raise RuntimeError("Worker tracked-tree hash mismatch")

                    job_path = (checkout / JOB_SPEC_RELPATH).resolve()
                    if checkout.resolve() not in job_path.parents:
                        raise RuntimeError("Job path escapes checkout")
                    if sha256_file(job_path) != JOB_SPEC_SHA256:
                        raise RuntimeError("Frozen job SHA-256 mismatch")
                    sys.path.insert(0, str(checkout))

                    def heartbeat_loop():
                        while not terminal.wait(HEARTBEAT_SECONDS):
                            try:
                                reporter.put(
                                    {
                                        "state": "running",
                                        "worker_commit": WORKER_COMMIT,
                                        "job_sha256": JOB_SPEC_SHA256,
                                        "gpu": gpu_name,
                                        "updated_at_utc": utc_now(),
                                    },
                                    "worker: heartbeat",
                                )
                            except Exception:
                                pass

                    threading.Thread(target=heartbeat_loop, daemon=True).start()
                    reporter.put(
                        {
                            "state": "running",
                            "worker_commit": WORKER_COMMIT,
                            "job_sha256": JOB_SPEC_SHA256,
                            "gpu": gpu_name,
                            "updated_at_utc": utc_now(),
                        },
                        "worker: started",
                    )

                    from worker.worker import (
                        VerifiedDirectorySink,
                        builtin_registry,
                        load_job_spec,
                        run_lifecycle,
                    )
                    from worker.role_aware_runner import register_handlers

                    class GitHubReportingDriveSink:
                        def __init__(self, target_dir):
                            self.delegate = VerifiedDirectorySink(target_dir)

                        def copy_and_verify(self, archive, expected_sha256):
                            receipt = dict(
                                self.delegate.copy_and_verify(archive, expected_sha256)
                            )
                            with tarfile.open(archive, "r:gz") as bundle:
                                member = bundle.getmember("run_manifest.json")
                                manifest = json.load(bundle.extractfile(member))
                            terminal.set()
                            reporter.put(
                                {
                                    "state": manifest.get("status", "unknown"),
                                    "worker_commit": WORKER_COMMIT,
                                    "job_sha256": JOB_SPEC_SHA256,
                                    "gpu": gpu_name,
                                    "manifest": manifest,
                                    "artifact": receipt,
                                    "updated_at_utc": utc_now(),
                                },
                                "worker: terminal result",
                            )
                            return receipt

                    spec = load_job_spec(job_path, JOB_SPEC_SHA256)
                    registry = builtin_registry()
                    register_handlers(registry)
                    sink = GitHubReportingDriveSink(DRIVE_ARTIFACT_DIR)
                    lifecycle = run_lifecycle(spec, registry, WORK_ROOT, sink, lambda: None)
                    if lifecycle.get("manifest", {}).get("status") != "completed":
                        wait_failure_grace()
                except BaseException as error:
                    terminal.set()
                    if reporter is not None:
                        try:
                            reporter.put(
                                {
                                    "state": "worker_failed",
                                    "error_type": type(error).__name__,
                                    "error": str(error)[:2000],
                                    "updated_at_utc": utc_now(),
                                },
                                "worker: setup or lifecycle failure",
                            )
                        except Exception:
                            pass
                    wait_failure_grace()
                    raise
                finally:
                    terminal.set()
                    watchdog.cancel()
                    release_once()
                """
            ),
        ],
    }


if __name__ == "__main__":
    destination = Path(__file__).with_name("HALSP_ROLE_AWARE_NIGHT_WORKER.ipynb")
    destination.write_text(
        json.dumps(build(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(destination)
