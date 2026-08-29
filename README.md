# HALSP night worker

This repository is a small, auditable control plane for a manually started
Google Colab worker. It is deliberately separate from the shared HALSP research
repository and does not push to, merge, modify, or access that repository at
Colab runtime.

The worker accepts a finite JSON DAG containing only registered handler names,
parameters, dependencies, and simple comparisons. Remote job data cannot carry
shell commands, Python code, modules, entry points, or arbitrary scripts. The
worker repository, the embedded hash-verified experiment source snapshot,
CIFAR-100 dataset, and job file are all pinned by immutable identities and
recorded in the final provenance.

## Frozen data source

- Dataset: `uoft-cs/cifar100`
- Config: `cifar100`
- Revision: `aadb3af77e9048adbea6b47c21a81e47dd092ae5`
- Expected rows: 50,000 train and 10,000 test
- Expected labels: 100 fine classes and 20 coarse classes

Preparation checks, in order, a previously verified prepared directory, the
local Hugging Face cache, and finally the exact Hub revision. A mismatched cache
or prepared manifest stops the job instead of being overwritten.

## Colab flow

1. Review and commit the worker code plus the finite experiment job.
2. Fill the exact worker commit, tracked-tree SHA-256, job path, job SHA-256, and
   durable artifact destination in
   `notebooks/HALSP_ROLE_AWARE_NIGHT_WORKER.ipynb`.
3. Add the fine-grained `WORKERRR_TOKEN` Colab Secret (limited to this repo),
   upload the notebook, select NVIDIA L4, and run all cells once.
4. The notebook checks out the exact worker commit, verifies both hashes, runs
   only registered handlers, writes heartbeat/terminal status to the dedicated
   worker repo, creates a terminal archive, copies it to the configured Drive
   folder, and independently verifies the copied SHA-256.
5. `google.colab.runtime.unassign()` is the final lifecycle action, including
   failure paths, so the GPU session is released.

Codex never connects to the Colab runtime. GitHub is the control/status channel;
the notebook mounts Drive only after the reviewed worker and frozen lifecycle
configuration are established. The token is read from Colab Secrets, is never
written into a Git remote or artifact, and should have access only to this
repository. Ordinary `/content` is rejected as the final sink because it is
ephemeral.

## Local checks

```text
python -m unittest discover -s tests -v
python notebooks/build_halsp_worker_colab.py
```

Large datasets, caches, source checkouts, checkpoints, and result archives stay
outside Git. Only compact job definitions, code, tests, manifests, and summaries
belong here.
