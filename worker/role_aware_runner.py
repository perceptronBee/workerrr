"""Audited HALSP role-aware Focus screen; no Explore and no hard swaps.

The module is a fixed worker handler, not a remotely supplied entry point.  It
adapts the exact pinned HALSP source to the prepared Hugging Face CIFAR-100
DatasetDict, performs the frozen 45k/5k split, and runs at most ten endpoints.
"""

from __future__ import annotations

import copy
import csv
import functools
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .prepare_cifar100 import load_prepared_cifar100
    from .role_aware_core import (
        calculate_role_scores,
        current_form_score,
        exact_k_indices,
        full_support_momentum_update,
        role_alphas,
        selected_swap_fraction,
    )
except ImportError:  # pragma: no cover - supports notebook adding worker/ to sys.path
    from prepare_cifar100 import load_prepared_cifar100  # type: ignore
    from role_aware_core import (  # type: ignore
        calculate_role_scores,
        current_form_score,
        exact_k_indices,
        full_support_momentum_update,
        role_alphas,
        selected_swap_fraction,
    )


SOURCE_REPOSITORY = "https://github.com/sp4cing-itu/efficient_ai_test_repo.git"
SOURCE_COMMIT = "087725a74be5407d750c537ac701d82531c68a91"
HALSP_ALL_SHA256 = "1b1a1e8e3f0c10c1a592de6c5cc49f7784d11aaea0df0b8612b077a0a4801700"
PROTOCOL_VERSION = "halsp-role-aware-v1"
REQUIRED_GPU = "NVIDIA L4"
SEEDS = (0, 1)
SCORERS = ("probe_momentum", "structural_taylor", "role_aware_taylor")
MODULATIONS = ("role_specific", "role_blind")
EPOCHS = 120
WARMUP_EPOCHS = 6
COOLDOWN_EPOCHS = 18
COOLDOWN_START = EPOCHS - COOLDOWN_EPOCHS
INITIAL_SELECTION_EPOCH = 6
MATERIAL_PROBE_EPOCHS = (20, 40, 60, 80, 100)
PROBE_EPOCHS = (INITIAL_SELECTION_EPOCH, *MATERIAL_PROBE_EPOCHS)
PRIMARY_SPARSE_EPOCH = 100
SPARSE_STAGE_IDS = (2, 3)
VALIDATION_SPLIT_SEED = 20_260_826
PROBE_SPLIT_SEED = 20_260_829
VALIDATION_PER_CLASS = 50
PROBE_PER_CLASS = 10
BATCH_SIZE = 256
PROBE_BATCH_SIZE = 250
MAIN_LR = 0.2
ALPHA_LR = 0.02
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9


class RoleAwareProtocolError(RuntimeError):
    pass


@dataclass
class LoaderBundle:
    train: Any
    validation: Any
    probe: Any
    official_test: Any
    train_generator: torch.Generator
    split_manifest: dict[str, Any]


class HFCifarView(torch.utils.data.Dataset):
    def __init__(self, split: Any, indices: Sequence[int] | None, transform: Any) -> None:
        self.split = split
        self.indices = list(range(len(split))) if indices is None else list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.split[self.indices[index]]
        image = row["img"].convert("RGB")
        return self.transform(image), int(row["fine_label"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _indices_sha256(indices: Sequence[int]) -> str:
    payload = ",".join(str(int(index)) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_pinned_halsp(source_dir: str | os.PathLike[str]) -> Any:
    source = Path(source_dir).resolve() / "src" / "halsp_all.py"
    if not source.is_file() or _sha256_file(source) != HALSP_ALL_SHA256:
        raise RoleAwareProtocolError("Pinned halsp_all.py identity mismatch")
    name = f"halsp_role_aware_pinned_{os.getpid()}_{id(source)}"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RoleAwareProtocolError("Cannot import pinned HALSP source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stratified_indices(labels: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
    tensor = torch.as_tensor(labels, dtype=torch.long)
    val_gen = torch.Generator().manual_seed(VALIDATION_SPLIT_SEED)
    probe_gen = torch.Generator().manual_seed(PROBE_SPLIT_SEED)
    train_indices: list[int] = []
    val_indices: list[int] = []
    probe_indices: list[int] = []
    for label in torch.unique(tensor, sorted=True).tolist():
        members = torch.nonzero(tensor == label, as_tuple=False).flatten()
        val_order = members[torch.randperm(members.numel(), generator=val_gen)]
        chosen_val = val_order[:VALIDATION_PER_CLASS]
        remaining = val_order[VALIDATION_PER_CLASS:]
        probe_order = remaining[torch.randperm(remaining.numel(), generator=probe_gen)]
        val_indices.extend(chosen_val.tolist())
        train_indices.extend(remaining.tolist())
        probe_indices.extend(probe_order[:PROBE_PER_CLASS].tolist())
    train_indices.sort()
    val_indices.sort()
    probe_indices.sort()
    if (len(train_indices), len(val_indices), len(probe_indices)) != (45_000, 5_000, 1_000):
        raise RoleAwareProtocolError("CIFAR-100 split cardinality mismatch")
    if set(train_indices) & set(val_indices):
        raise RoleAwareProtocolError("Training and validation splits overlap")
    if not set(probe_indices).issubset(train_indices):
        raise RoleAwareProtocolError("Probe examples must be training-only")
    return train_indices, val_indices, probe_indices


def build_hf_loaders(prepared_path: str | os.PathLike[str], num_workers: int = 2) -> LoaderBundle:
    import torchvision.transforms as T

    dataset = load_prepared_cifar100(prepared_path)
    labels = dataset["train"]["fine_label"]
    train_indices, val_indices, probe_indices = _stratified_indices(labels)
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_gen = torch.Generator().manual_seed(0)
    probe_gen = torch.Generator().manual_seed(PROBE_SPLIT_SEED)
    train = torch.utils.data.DataLoader(
        HFCifarView(dataset["train"], train_indices, train_tf),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        generator=train_gen,
    )
    validation = torch.utils.data.DataLoader(
        HFCifarView(dataset["train"], val_indices, eval_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    probe = torch.utils.data.DataLoader(
        HFCifarView(dataset["train"], probe_indices, eval_tf),
        batch_size=PROBE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        generator=probe_gen,
    )
    official = torch.utils.data.DataLoader(
        HFCifarView(dataset["test"], None, eval_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return LoaderBundle(
        train,
        validation,
        probe,
        official,
        train_gen,
        {
            "validation_split_seed": VALIDATION_SPLIT_SEED,
            "probe_split_seed": PROBE_SPLIT_SEED,
            "train_count": len(train_indices),
            "validation_count": len(val_indices),
            "probe_count": len(probe_indices),
            "official_test_count": len(dataset["test"]),
            "train_indices_sha256": _indices_sha256(train_indices),
            "validation_indices_sha256": _indices_sha256(val_indices),
            "probe_indices_sha256": _indices_sha256(probe_indices),
            "probe_augmentation": "none",
            "probe_updates": 0,
        },
    )


def role_blind_scales(two_dof: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if two_dof.ndim != 2 or two_dof.shape[1] != 2:
        raise ValueError("role-blind control requires [channels, 2]")
    bounded = 1.0 + 0.25 * torch.tanh(two_dof)
    return bounded[:, 0], bounded[:, 1]


def add_modulation_parameters(model: Any, modulation: str) -> None:
    if modulation not in MODULATIONS:
        raise ValueError(f"Unknown modulation: {modulation}")
    for stage_id, stage in enumerate(model.stages):
        stage._role_modulation = modulation if stage_id in SPARSE_STAGE_IDS else "none"
        if stage_id in SPARSE_STAGE_IDS:
            stage.register_parameter(
                "role_dof", nn.Parameter(torch.zeros(stage.mid_channels, 2))
            )


def _role_scales(stage: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ones = torch.ones(
        stage.mid_channels,
        device=stage.master_weight.device,
        dtype=stage.master_weight.dtype,
    )
    modulation = getattr(stage, "_role_modulation", "none")
    if modulation == "none":
        return ones, ones, ones, ones
    if modulation == "role_specific":
        alphas = role_alphas(stage.role_dof).to(dtype=stage.master_weight.dtype)
        return alphas[:, 0], alphas[:, 1], alphas[:, 2], ones
    if modulation == "role_blind":
        weight_scale, depthwise_scale = role_blind_scales(stage.role_dof)
        weight_scale = weight_scale.to(dtype=stage.master_weight.dtype)
        depthwise_scale = depthwise_scale.to(dtype=stage.master_weight.dtype)
        return weight_scale, weight_scale, weight_scale, depthwise_scale
    raise RoleAwareProtocolError(f"Unexpected modulation state: {modulation}")


def _ones_gates(stage: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base = torch.ones(
        stage.mid_channels,
        device=stage.master_weight.device,
        dtype=stage.master_weight.dtype,
    )
    return base, base, base, base


def role_dense_core(
    stage: Any,
    x_expanded: torch.Tensor,
    gates: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    entry_gate, inner_gate, exit_gate, depthwise_gate = gates or _ones_gates(stage)
    entry_alpha, inner_alpha, exit_alpha, depthwise_alpha = _role_scales(stage)
    master = stage.master_weight
    entry_weight = master * (entry_gate * entry_alpha)[:, None, None, None]
    latent = F.conv2d(x_expanded, entry_weight, stride=stage.stride, padding=0)

    mid = stage.mid_channels
    blocks = stage.num_blocks
    dw_scale = (depthwise_gate * depthwise_alpha).repeat(blocks)
    dw_weight = stage.dw_weight * dw_scale[:, None, None, None]
    inner_master = master * (inner_gate * inner_alpha)[:, None, None, None]
    inner_2d = inner_master.view(mid, stage.out_channels)
    expanded = inner_2d.unsqueeze(0).expand(blocks, -1, -1)
    indices = stage._all_inner_cols.unsqueeze(1).expand(-1, mid, -1)
    inner_weights = torch.gather(expanded, 2, indices).unsqueeze(-1).unsqueeze(-1)
    for block in range(blocks):
        out = F.gelu(latent)
        start = stage._dw_block_starts[block]
        out = stage._spatial(out, dw_weight[start : start + mid], mid, block)
        out = F.conv2d(out, inner_weights[block], stride=1, padding=0)
        latent = out + latent
    exit_weight = (
        master * (exit_gate * exit_alpha)[:, None, None, None]
    ).permute(1, 0, 2, 3).contiguous()
    return stage.exit_bn(F.conv2d(latent, exit_weight, stride=1, padding=0))


def set_stage_mask(stage: Any, indices: torch.Tensor) -> None:
    indices = torch.as_tensor(indices, device=stage.master_weight.device, dtype=torch.long)
    expected = stage.mid_channels if getattr(stage, "_role_stage_id", -1) not in SPARSE_STAGE_IDS else stage.mid_channels // 2
    if indices.numel() != expected or torch.unique(indices).numel() != expected:
        raise RoleAwareProtocolError("Mask is not exact-K")
    if int(indices.min()) < 0 or int(indices.max()) >= stage.mid_channels:
        raise RoleAwareProtocolError("Mask index outside stage")
    indices = torch.sort(indices).values
    stage._role_active_idx = indices
    stage._update_sparse_cache(indices)


def role_sparse_core(stage: Any, x_expanded: torch.Tensor) -> torch.Tensor:
    active = getattr(stage, "_role_active_idx", None)
    if active is None:
        raise RoleAwareProtocolError("Sparse phase has no exact-K mask")
    entry_alpha, inner_alpha, exit_alpha, depthwise_alpha = _role_scales(stage)
    master = stage.master_weight.index_select(0, active)
    entry_weight = master * entry_alpha.index_select(0, active)[:, None, None, None]
    latent = F.conv2d(x_expanded, entry_weight, stride=stage.stride, padding=0)
    k = active.numel()
    blocks = stage.num_blocks
    dw_weight = stage.dw_weight.index_select(0, stage._cached_dw_idx)
    dw_scale = depthwise_alpha.index_select(0, active).repeat(blocks)
    dw_weight = dw_weight * dw_scale[:, None, None, None]
    inner_master = master * inner_alpha.index_select(0, active)[:, None, None, None]
    expanded = inner_master.view(k, stage.out_channels).unsqueeze(0).expand(blocks, -1, -1)
    indices = stage._cached_col_compact.unsqueeze(1).expand(-1, k, -1)
    inner_weights = torch.gather(expanded, 2, indices).unsqueeze(-1).unsqueeze(-1)
    for block in range(blocks):
        out = F.gelu(latent)
        block_weight = dw_weight[block * k : (block + 1) * k]
        out = stage._spatial(out, block_weight, k, block)
        out = F.conv2d(out, inner_weights[block], stride=1, padding=0)
        latent = out + latent
    exit_weight = (
        master * exit_alpha.index_select(0, active)[:, None, None, None]
    ).permute(1, 0, 2, 3).contiguous()
    return stage.exit_bn(F.conv2d(latent, exit_weight, stride=1, padding=0))


def install_role_forward(halsp: Any) -> None:
    if getattr(halsp.HalspStage, "_role_aware_v1_installed", False):
        return

    def forward(stage: Any, x: torch.Tensor) -> torch.Tensor:
        up = stage.main_path_upsampler
        x_expanded = torch.cat((up(x), x), dim=1) if up is not None else x
        downsample = stage.downsample_path
        identity = downsample(x_expanded) if downsample is not None else x_expanded
        probe_gates = getattr(stage, "_role_probe_gates", None)
        sparse = (
            probe_gates is None
            and getattr(stage, "_role_stage_id", -1) in SPARSE_STAGE_IDS
            and (
                (stage.training and getattr(stage, "_role_phase", "warmup") == "search")
                or getattr(stage, "_role_force_sparse_eval", False)
            )
        )
        out = role_sparse_core(stage, x_expanded) if sparse else role_dense_core(stage, x_expanded, probe_gates)
        return F.gelu(out + identity)

    halsp.HalspStage.forward = forward
    halsp.HalspStage._role_aware_v1_installed = True


def _set_phase(model: Any, phase: str) -> None:
    if phase not in {"warmup", "search", "cooldown"}:
        raise ValueError(phase)
    for stage_id, stage in enumerate(model.stages):
        stage._role_stage_id = stage_id
        stage._role_phase = phase
        stage._role_force_sparse_eval = False
        if stage_id not in SPARSE_STAGE_IDS and getattr(stage, "_role_active_idx", None) is None:
            set_stage_mask(stage, torch.arange(stage.mid_channels, device=stage.master_weight.device))


def build_model_and_optimizers(
    halsp: Any,
    seed: int,
    modulation: str,
    device: torch.device,
) -> tuple[Any, Any, Any | None, Any, Any]:
    halsp.set_global_seed(seed)
    halsp.configure_backend(deterministic=True)
    model = halsp.HalspResNet50(
        num_classes=100,
        scoring_mode="momentum",
        deformable=False,
        sparsity=0.5,
        verbose=False,
        widths=(64, 128, 256, 512),
        blocks=(3, 4, 6, 3),
        stem_type="cifar",
        deform_stages=(),
    )
    halsp.init_weights(model)
    if modulation != "none":
        add_modulation_parameters(model, modulation)
    model.to(device)
    _set_phase(model, "warmup")
    alpha_ids = {
        id(stage.role_dof)
        for stage in model.stages
        if hasattr(stage, "role_dof") and stage.role_dof is not None
    }
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if id(parameter) in alpha_ids:
            continue
        if name.endswith("master_weight") or name.endswith("dw_weight"):
            no_decay.append(parameter)
        elif parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    optimizer = torch.optim.SGD(
        [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": WEIGHT_DECAY},
        ],
        lr=MAIN_LR,
        momentum=MOMENTUM,
        nesterov=True,
    )
    alpha_optimizer = None
    if alpha_ids:
        alpha_optimizer = torch.optim.SGD(
            [stage.role_dof for stage in model.stages if hasattr(stage, "role_dof")],
            lr=ALPHA_LR,
            momentum=MOMENTUM,
            nesterov=True,
            weight_decay=0.0,
        )
    scheduler = halsp.build_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    return model, optimizer, alpha_optimizer, scheduler, criterion


def _active_master(stage: Any, phase: str) -> torch.Tensor:
    if phase == "search" and stage._role_stage_id in SPARSE_STAGE_IDS:
        return stage._role_active_idx
    return torch.arange(stage.mid_channels, device=stage.master_weight.device)


def _active_depthwise(stage: Any, phase: str) -> torch.Tensor:
    active = _active_master(stage, phase)
    offsets = torch.arange(stage.num_blocks, device=active.device)[:, None] * stage.mid_channels
    return (active[None, :] + offsets).reshape(-1)


def _manual_channel_decay(model: Any, phase: str) -> None:
    for stage in model.stages:
        master_idx = _active_master(stage, phase)
        if stage.master_weight.grad is not None:
            stage.master_weight.grad.index_add_(
                0,
                master_idx,
                stage.master_weight.detach().index_select(0, master_idx) * WEIGHT_DECAY,
            )
        dw_idx = _active_depthwise(stage, phase)
        if stage.dw_weight.grad is not None:
            stage.dw_weight.grad.index_add_(
                0,
                dw_idx,
                stage.dw_weight.detach().index_select(0, dw_idx) * WEIGHT_DECAY,
            )


def _inactive_mask(size: int, active: torch.Tensor, device: torch.device) -> torch.Tensor:
    mask = torch.ones(size, device=device, dtype=torch.bool)
    mask[active] = False
    return mask


def masked_optimizer_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    alpha_optimizer: torch.optim.Optimizer | None,
    phase: str,
) -> None:
    """Step then restore inactive rows and clear their SGD momentum exactly."""

    frozen: list[tuple[nn.Parameter, torch.Tensor, torch.Tensor]] = []
    if phase == "search":
        for stage in model.stages:
            if stage._role_stage_id not in SPARSE_STAGE_IDS:
                continue
            master_inactive = _inactive_mask(
                stage.mid_channels, stage._role_active_idx, stage.master_weight.device
            )
            dw_active = _active_depthwise(stage, phase)
            dw_inactive = _inactive_mask(stage.dw_weight.shape[0], dw_active, stage.dw_weight.device)
            frozen.append((stage.master_weight, master_inactive, stage.master_weight.detach()[master_inactive].clone()))
            frozen.append((stage.dw_weight, dw_inactive, stage.dw_weight.detach()[dw_inactive].clone()))
            if hasattr(stage, "role_dof"):
                frozen.append((stage.role_dof, master_inactive, stage.role_dof.detach()[master_inactive].clone()))
    optimizer.step()
    if alpha_optimizer is not None and phase != "warmup":
        alpha_optimizer.step()
    optimizers = [optimizer] + ([alpha_optimizer] if alpha_optimizer is not None else [])
    with torch.no_grad():
        for parameter, inactive, before in frozen:
            parameter[inactive] = before
            for current in optimizers:
                state = current.state.get(parameter, {}) if current is not None else {}
                buffer = state.get("momentum_buffer")
                if buffer is not None:
                    buffer[inactive] = 0


def _hash_tensor_state(model: Any, optimizers: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not torch.is_tensor(tensor):
            digest.update(name.encode())
            digest.update(repr(tensor).encode())
            continue
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(str(tuple(cpu.shape)).encode())
        raw_view = cpu.reshape(1) if cpu.ndim == 0 else cpu
        digest.update(raw_view.view(torch.uint8).numpy().tobytes())
    for opt_index, optimizer in enumerate(optimizers):
        if optimizer is None:
            continue
        digest.update(f"optimizer-{opt_index}".encode())
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter_index, parameter in enumerate(group["params"]):
                state = optimizer.state.get(parameter, {})
                for key, value in sorted(state.items()):
                    digest.update(f"{group_index}:{parameter_index}:{key}".encode())
                    if torch.is_tensor(value):
                        cpu = value.detach().cpu().contiguous()
                        raw_view = cpu.reshape(1) if cpu.ndim == 0 else cpu
                        digest.update(raw_view.view(torch.uint8).numpy().tobytes())
                    else:
                        digest.update(repr(value).encode())
    return digest.hexdigest()


def _rng_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["python"] != right["python"]:
        return False
    if left["numpy"][0] != right["numpy"][0] or not np.array_equal(left["numpy"][1], right["numpy"][1]):
        return False
    if left["numpy"][2:] != right["numpy"][2:] or not torch.equal(left["torch"], right["torch"]):
        return False
    left_cuda, right_cuda = left.get("cuda"), right.get("cuda")
    if left_cuda is None or right_cuda is None:
        return left_cuda is right_cuda
    return len(left_cuda) == len(right_cuda) and all(
        torch.equal(a, b) for a, b in zip(left_cuda, right_cuda)
    )


def dense_probe(
    halsp: Any,
    model: Any,
    loader: Any,
    criterion: Any,
    device: torch.device,
    optimizer: Any,
    alpha_optimizer: Any | None,
    probe_momentum: list[torch.Tensor | None],
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    """1000-example augment-free full-support probe with no model/SGD step."""

    before_state = _hash_tensor_state(model, (optimizer, alpha_optimizer))
    before_rng = halsp.capture_rng()
    was_training = model.training
    model.eval()
    loader.generator.manual_seed(PROBE_SPLIT_SEED)
    role_sums = [
        [torch.zeros(stage.mid_channels, device=device) for _ in range(4)]
        for stage in model.stages
    ]
    weight_sums = [torch.zeros_like(stage.master_weight) for stage in model.stages]
    total = 0
    try:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            gates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
            targets: list[torch.Tensor] = []
            for stage in model.stages:
                current = tuple(
                    torch.ones(stage.mid_channels, device=device, requires_grad=True)
                    for _ in range(4)
                )
                stage._role_probe_gates = current
                gates.append(current)
                targets.extend(current)
            targets.extend(stage.master_weight for stage in model.stages)
            outputs = model(inputs)
            loss = criterion(outputs.float(), labels)
            gradients = torch.autograd.grad(loss, targets, retain_graph=False, create_graph=False)
            batch = labels.size(0)
            total += batch
            offset = 0
            for stage_id in range(len(model.stages)):
                for role_id in range(4):
                    role_sums[stage_id][role_id].add_(gradients[offset].detach() * batch)
                    offset += 1
            for stage_id in range(len(model.stages)):
                weight_sums[stage_id].add_(gradients[offset + stage_id].detach() * batch)
            for stage in model.stages:
                del stage._role_probe_gates
        if total != 1_000:
            raise RoleAwareProtocolError(f"Probe saw {total} examples instead of 1000")
        diagnostics: list[dict[str, Any]] = []
        updated_momentum: list[torch.Tensor] = []
        for stage_id, stage in enumerate(model.stages):
            e, inner, exit_, depthwise = [value / total for value in role_sums[stage_id]]
            scores = calculate_role_scores(e, inner, exit_, depthwise)
            gradient = weight_sums[stage_id] / total
            momentum = full_support_momentum_update(probe_momentum[stage_id], gradient, MOMENTUM)
            updated_momentum.append(momentum.detach().clone())
            k = stage.mid_channels // 2
            selections = {
                "probe_momentum": exact_k_indices(current_form_score(stage.master_weight, momentum), k),
                "structural_taylor": exact_k_indices(scores.structural_taylor, k),
                "role_aware_taylor": exact_k_indices(scores.role_aware_taylor, k),
            }
            role_activity = float((e.abs() + inner.abs() + exit_.abs()).sum())
            diagnostics.append(
                {
                    "stage": stage_id,
                    "mid_channels": stage.mid_channels,
                    "operational_sparse": stage_id in SPARSE_STAGE_IDS,
                    "conflict": scores.conflict,
                    "role_activity": role_activity,
                    "swap_fraction": selected_swap_fraction(
                        selections["structural_taylor"],
                        selections["role_aware_taylor"],
                        k,
                    ),
                    "selections": {key: value.detach().cpu().tolist() for key, value in selections.items()},
                    "score_sha256": {
                        "probe_momentum": hashlib.sha256(
                            current_form_score(stage.master_weight, momentum).detach().cpu().numpy().tobytes()
                        ).hexdigest(),
                        "structural_taylor": hashlib.sha256(
                            scores.structural_taylor.detach().cpu().numpy().tobytes()
                        ).hexdigest(),
                        "role_aware_taylor": hashlib.sha256(
                            scores.role_aware_taylor.detach().cpu().numpy().tobytes()
                        ).hexdigest(),
                    },
                }
            )
        return diagnostics, updated_momentum
    finally:
        for stage in model.stages:
            if hasattr(stage, "_role_probe_gates"):
                del stage._role_probe_gates
        halsp.restore_rng(before_rng)
        model.train(was_training)
        after_state = _hash_tensor_state(model, (optimizer, alpha_optimizer))
        after_rng = halsp.capture_rng()
        if before_state != after_state or not _rng_equal(before_rng, after_rng):
            raise RoleAwareProtocolError("Dense probe mutated model, optimizer, BN, or RNG state")


def sparse_material_checkpoint(stage_rows: Sequence[Mapping[str, Any]]) -> bool:
    sparse = [row for row in stage_rows if row.get("stage") in SPARSE_STAGE_IDS]
    if len(sparse) != 2:
        raise ValueError("Material gate requires exactly the two sparse stages")
    conflicts = [float(row["conflict"]) for row in sparse]
    swaps = [float(row["swap_fraction"]) for row in sparse]
    activities = [float(row.get("role_activity", 0.0)) for row in sparse]
    median_conflict = sum(sorted(conflicts)) / 2.0
    median_swap = sum(sorted(swaps)) / 2.0
    return bool(
        max(activities) > 1e-10
        and median_conflict >= 0.25
        and median_swap >= 0.111
        and sum(value >= 0.25 for value in conflicts) >= 1
    )


def open_sparse_modulation_gate(seed_flags: Mapping[int, Sequence[bool]]) -> bool:
    if set(seed_flags) != set(SEEDS):
        raise ValueError("Stage-B gate requires seeds 0 and 1")
    return all(len(flags) == 5 and sum(bool(x) for x in flags) >= 3 and bool(flags[-1]) for flags in seed_flags.values())


@torch.no_grad()
def evaluate(model: Any, loader: Any, criterion: Any, device: torch.device, sparse: bool) -> dict[str, float]:
    model.eval()
    for stage in model.stages:
        stage._role_force_sparse_eval = bool(sparse and stage._role_stage_id in SPARSE_STAGE_IDS)
    loss_sum = correct1 = correct5 = count = 0.0
    try:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(inputs)
            outputs = outputs.float()
            loss_sum += float(criterion(outputs, labels)) * labels.size(0)
            _, predicted = outputs.topk(5, 1, True, True)
            matches = predicted.eq(labels[:, None])
            correct1 += float(matches[:, :1].sum())
            correct5 += float(matches.sum())
            count += labels.size(0)
    finally:
        for stage in model.stages:
            stage._role_force_sparse_eval = False
    return {
        "loss": loss_sum / count,
        "top1": 100.0 * correct1 / count,
        "top5": 100.0 * correct5 / count,
    }


def train_epoch(
    halsp: Any,
    model: Any,
    loader: Any,
    generator: torch.Generator,
    optimizer: Any,
    alpha_optimizer: Any | None,
    criterion: Any,
    device: torch.device,
    seed: int,
    epoch: int,
    phase: str,
) -> dict[str, float]:
    model.train()
    halsp.seed_loader_for_epoch(loader, generator, seed, epoch)
    loss_sum = correct = count = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if alpha_optimizer is not None:
            alpha_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(inputs)
            loss = criterion(outputs.float(), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0 if epoch < WARMUP_EPOCHS else 4.0)
        _manual_channel_decay(model, phase)
        masked_optimizer_step(model, optimizer, alpha_optimizer, phase)
        batch = labels.size(0)
        loss_sum += float(loss.detach()) * batch
        correct += float(outputs.detach().float().argmax(1).eq(labels).sum())
        count += batch
    return {"loss": loss_sum / count, "top1": 100.0 * correct / count}


def _checkpoint_payload(model: Any, masks: list[list[int]], modulation: str, scorer: str, seed: int) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "source_commit": SOURCE_COMMIT,
        "epoch": PRIMARY_SPARSE_EPOCH,
        "seed": seed,
        "scorer": scorer,
        "modulation": modulation,
        "model": {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in model.state_dict().items()},
        "masks": masks,
    }


def run_endpoint(
    halsp: Any,
    loaders: LoaderBundle,
    artifact_root: Path,
    device: torch.device,
    seed: int,
    scorer: str,
    modulation: str,
) -> dict[str, Any]:
    name = f"{scorer}-{modulation}-s{seed}"
    run_dir = artifact_root / "endpoints" / name
    run_dir.mkdir(parents=True, exist_ok=False)
    model, optimizer, alpha_optimizer, scheduler, criterion = build_model_and_optimizers(
        halsp, seed, modulation, device
    )
    probe_momentum: list[torch.Tensor | None] = [None] * len(model.stages)
    checkpoint_flags: list[bool] = []
    metrics: list[dict[str, Any]] = []
    config = {
        "protocol": PROTOCOL_VERSION,
        "source_commit": SOURCE_COMMIT,
        "seed": seed,
        "scorer": scorer,
        "modulation": modulation,
        "epochs": EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "search_epochs": COOLDOWN_START - WARMUP_EPOCHS,
        "cooldown_epochs": COOLDOWN_EPOCHS,
        "probe_epochs": list(PROBE_EPOCHS),
        "sparse_stage_ids": list(SPARSE_STAGE_IDS),
        "focus_ratio_sparse_stages": 0.5,
        "explore": False,
        "hard_swap": False,
        "alpha_lr": ALPHA_LR if modulation != "none" else None,
        "alpha_weight_decay": 0.0 if modulation != "none" else None,
    }
    _json_write(run_dir / "config.json", config)
    primary_sparse_path = run_dir / "primary_sparse_epoch100.pt"
    for epoch in range(EPOCHS):
        phase = "warmup" if epoch < WARMUP_EPOCHS else ("search" if epoch < COOLDOWN_START else "cooldown")
        _set_phase(model, phase)
        if epoch in PROBE_EPOCHS:
            diagnostics, probe_momentum = dense_probe(
                halsp,
                model,
                loaders.probe,
                criterion,
                device,
                optimizer,
                alpha_optimizer,
                probe_momentum,
            )
            for row in diagnostics:
                if row["stage"] in SPARSE_STAGE_IDS:
                    selected = torch.tensor(row["selections"][scorer], device=device)
                    set_stage_mask(model.stages[row["stage"]], selected)
            material = (
                sparse_material_checkpoint(diagnostics)
                if epoch in MATERIAL_PROBE_EPOCHS
                else None
            )
            if material is not None:
                checkpoint_flags.append(material)
            _json_write(
                run_dir / "probes" / f"epoch-{epoch:03d}.json",
                {"epoch": epoch, "material": material, "stages": diagnostics},
            )
        started = time.time()
        train_metrics = train_epoch(
            halsp,
            model,
            loaders.train,
            loaders.train_generator,
            optimizer,
            alpha_optimizer,
            criterion,
            device,
            seed,
            epoch,
            phase,
        )
        validation = evaluate(model, loaders.validation, criterion, device, sparse=phase == "search")
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = {
            "epoch": epoch,
            "phase": phase,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "validation_loss": validation["loss"],
            "validation_top1": validation["top1"],
            "validation_top5": validation["top5"],
            "epoch_time_seconds": time.time() - started,
        }
        metrics.append(row)
        if epoch == PRIMARY_SPARSE_EPOCH:
            masks = [stage._role_active_idx.detach().cpu().tolist() for stage in model.stages]
            torch.save(_checkpoint_payload(model, masks, modulation, scorer, seed), primary_sparse_path)
    if len(checkpoint_flags) != 5 or not primary_sparse_path.is_file():
        raise RoleAwareProtocolError("Endpoint did not produce five material probes and epoch-100 state")
    torch.save(model.state_dict(), run_dir / "final.pt")
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    summary = {
        "name": name,
        "seed": seed,
        "scorer": scorer,
        "modulation": modulation,
        "checkpoint_flags": checkpoint_flags,
        "primary_sparse_validation_top1": metrics[PRIMARY_SPARSE_EPOCH]["validation_top1"],
        "final_dense_validation_top1": metrics[-1]["validation_top1"],
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "role_dof": sum(
            stage.role_dof.numel() for stage in model.stages if hasattr(stage, "role_dof")
        ),
        "official_test": None,
    }
    _json_write(run_dir / "summary.json", summary)
    del model, optimizer, alpha_optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def official_test_once(
    halsp: Any,
    loaders: LoaderBundle,
    artifact_root: Path,
    device: torch.device,
    endpoint: dict[str, Any],
) -> dict[str, float]:
    run_dir = artifact_root / "endpoints" / endpoint["name"]
    result_path = run_dir / "official_test_once.json"
    if result_path.exists():
        raise RoleAwareProtocolError("Official test endpoint was requested twice")
    payload = torch.load(run_dir / "primary_sparse_epoch100.pt", map_location="cpu", weights_only=False)
    if payload.get("source_commit") != SOURCE_COMMIT or payload.get("epoch") != PRIMARY_SPARSE_EPOCH:
        raise RoleAwareProtocolError("Primary sparse checkpoint provenance mismatch")
    model, _, _, _, criterion = build_model_and_optimizers(
        halsp, endpoint["seed"], endpoint["modulation"], device
    )
    model.load_state_dict(payload["model"], strict=True)
    _set_phase(model, "search")
    for stage, indices in zip(model.stages, payload["masks"]):
        set_stage_mask(stage, torch.tensor(indices, device=device))
    measured = evaluate(model, loaders.official_test, criterion, device, sparse=True)
    result = {
        "checkpoint_epoch": PRIMARY_SPARSE_EPOCH,
        "test_sparse_loss": measured["loss"],
        "test_sparse_top1": measured["top1"],
        "test_sparse_top5": measured["top5"],
    }
    _json_write(result_path, result)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["official_test"] = result
    _json_write(summary_path, summary)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_cpu_preflight(halsp: Any) -> dict[str, bool]:
    """Tiny actual-source invariants required before any full endpoint."""

    torch.manual_seed(4)
    stage = halsp.HalspStage(8, 8, 4, 1, 1, scoring_mode="momentum", deformable=False)
    stage._role_stage_id = 2
    stage._role_modulation = "none"
    stage.eval()
    inputs = torch.randn(2, 8, 6, 6)
    with torch.no_grad():
        reference = stage._forward_dense(inputs)
        gated = role_dense_core(stage, inputs, _ones_gates(stage))
    if not torch.equal(reference, gated):
        raise RoleAwareProtocolError("All-one role gate is not exactly dense-equivalent")

    separate = tuple(torch.ones(4, requires_grad=True) for _ in range(4))
    separate_loss = role_dense_core(stage, inputs, separate).sum()
    e, inner, exit_, _ = torch.autograd.grad(separate_loss, separate)
    common = torch.ones(4, requires_grad=True)
    depthwise = torch.ones(4, requires_grad=True)
    common_loss = role_dense_core(stage, inputs, (common, common, common, depthwise)).sum()
    common_gradient = torch.autograd.grad(common_loss, common)[0]
    if not torch.allclose(common_gradient, e + inner + exit_, atol=1e-5, rtol=1e-5):
        raise RoleAwareProtocolError("Tied gate derivative does not equal signed role sum")

    initial_dof = torch.zeros(7, 2, requires_grad=True)
    alpha = role_alphas(initial_dof)
    blind_w, blind_dw = role_blind_scales(torch.zeros(7, 2))
    if not torch.equal(alpha, torch.ones_like(alpha)) or not torch.equal(blind_w, blind_dw):
        raise RoleAwareProtocolError("Alpha controls do not start at identity")
    if not torch.allclose(alpha.mean(-1), torch.ones(7)):
        raise RoleAwareProtocolError("Role-specific alphas are not mean-one")
    alpha_gradient = torch.autograd.grad(
        (alpha * alpha.new_tensor([1.0, 2.0, 4.0])).sum(), initial_dof
    )[0]
    if not torch.isfinite(alpha_gradient).all() or not bool(alpha_gradient.abs().max() > 0):
        raise RoleAwareProtocolError("Role-specific modulation cannot learn from identity")

    modulated_stage = copy.deepcopy(stage)
    modulated_stage._role_modulation = "role_specific"
    modulated_stage.register_parameter("role_dof", nn.Parameter(torch.zeros(4, 2)))
    with torch.no_grad():
        modulated_stage.role_dof[:, 0] = torch.tensor([0.4, -0.2, 0.3, -0.1])
    base_gates = tuple(torch.ones(4, requires_grad=True) for _ in range(4))
    modulated_output = role_dense_core(modulated_stage, inputs, base_gates)
    gate_gradients = torch.autograd.grad(modulated_output.square().mean(), base_gates)
    modulated_stage._role_modulation = "none"
    unmodulated_output = role_dense_core(modulated_stage, inputs, _ones_gates(modulated_stage))
    if torch.equal(modulated_output.detach(), unmodulated_output.detach()):
        raise RoleAwareProtocolError("Dense probe bypassed role modulation")
    if not all(torch.isfinite(gradient).all() for gradient in gate_gradients):
        raise RoleAwareProtocolError("Modulated dense probe gate derivative is invalid")

    # Exercise the actual sparse gather plus masked SGD state on a tiny stage.
    reserve_stage = halsp.HalspStage(
        8, 8, 4, 1, 1, scoring_mode="momentum", deformable=False
    )
    reserve_stage._role_stage_id = 2
    reserve_stage._role_phase = "search"
    reserve_stage._role_modulation = "role_specific"
    reserve_stage.register_parameter("role_dof", nn.Parameter(torch.zeros(4, 2)))
    set_stage_mask(reserve_stage, torch.tensor([0, 2]))
    tiny_model = nn.Module()
    tiny_model.stages = nn.ModuleList([reserve_stage])
    main_optimizer = torch.optim.SGD(
        [reserve_stage.master_weight, reserve_stage.dw_weight],
        lr=0.01,
        momentum=0.9,
        nesterov=True,
    )
    alpha_optimizer = torch.optim.SGD(
        [reserve_stage.role_dof], lr=0.02, momentum=0.9, nesterov=True
    )
    # Simulate rows that were active before a mask change: stale Nesterov state
    # must neither move Reserve nor survive deactivation.
    for current_optimizer in (main_optimizer, alpha_optimizer):
        for group in current_optimizer.param_groups:
            for parameter in group["params"]:
                current_optimizer.state[parameter]["momentum_buffer"] = torch.ones_like(parameter)
    inactive = torch.tensor([False, True, False, True])
    inactive_dw = torch.tensor([False, True, False, True])
    before_master = reserve_stage.master_weight.detach()[inactive].clone()
    before_dw = reserve_stage.dw_weight.detach()[inactive_dw].clone()
    before_alpha = reserve_stage.role_dof.detach()[inactive].clone()
    loss = role_sparse_core(reserve_stage, torch.randn(2, 8, 6, 6)).square().mean()
    loss.backward()
    _manual_channel_decay(tiny_model, "search")
    masked_optimizer_step(
        tiny_model, main_optimizer, alpha_optimizer, "search"
    )
    if not torch.equal(reserve_stage.master_weight.detach()[inactive], before_master):
        raise RoleAwareProtocolError("Inactive master rows drifted in masked SGD")
    if not torch.equal(reserve_stage.dw_weight.detach()[inactive_dw], before_dw):
        raise RoleAwareProtocolError("Inactive depthwise rows drifted in masked SGD")
    if not torch.equal(reserve_stage.role_dof.detach()[inactive], before_alpha):
        raise RoleAwareProtocolError("Inactive modulation rows drifted in masked SGD")
    for current_optimizer, parameter, current_inactive in (
        (main_optimizer, reserve_stage.master_weight, inactive),
        (main_optimizer, reserve_stage.dw_weight, inactive_dw),
        (alpha_optimizer, reserve_stage.role_dof, inactive),
    ):
        momentum = current_optimizer.state[parameter].get("momentum_buffer")
        if momentum is None or not torch.equal(momentum[current_inactive], torch.zeros_like(momentum[current_inactive])):
            raise RoleAwareProtocolError("Inactive SGD momentum was not cleared")
    return {
        "gate_one_equivalence": True,
        "signed_derivative_sum": True,
        "alpha_identity_and_mean": True,
        "modulated_probe_uses_effective_roles": True,
        "reserve_parameters_and_state_frozen": True,
        "exact_k": True,
    }


def run_role_aware_v1(context: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
    if dict(params):
        raise RoleAwareProtocolError("role_aware_v1 accepts no remote parameters")
    source = context.job_spec["source"]
    if source["repo_url"] != SOURCE_REPOSITORY or source["commit"] != SOURCE_COMMIT:
        raise RoleAwareProtocolError("Role-aware handler requires the audited HALSP commit")
    prepared = context.outputs.get("prepare")
    if not isinstance(prepared, Mapping) or not isinstance(prepared.get("path"), str):
        raise RoleAwareProtocolError("Verified prepared CIFAR-100 output is missing")
    if not torch.cuda.is_available():
        raise RoleAwareProtocolError("Full role-aware study requires CUDA")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != REQUIRED_GPU:
        raise RoleAwareProtocolError(
            f"Accelerator mismatch: required {REQUIRED_GPU!r}, received {gpu_name!r}"
        )
    halsp = load_pinned_halsp(context.source_dir)
    install_role_forward(halsp)
    preflight = run_cpu_preflight(halsp)
    loaders = build_hf_loaders(prepared["path"])
    _json_write(context.artifact_dir / "split_manifest.json", loaders.split_manifest)
    _json_write(context.artifact_dir / "preflight.json", preflight)
    properties = torch.cuda.get_device_properties(0)
    _json_write(
        context.artifact_dir / "runtime_provenance.json",
        {
            "protocol": PROTOCOL_VERSION,
            "required_gpu": REQUIRED_GPU,
            "actual_gpu": gpu_name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "numpy": np.__version__,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "source_commit": SOURCE_COMMIT,
            "source_file_sha256": HALSP_ALL_SHA256,
        },
    )
    endpoints: list[dict[str, Any]] = []
    rat_flags: dict[int, list[bool]] = {}
    for seed in SEEDS:
        for scorer in SCORERS:
            summary = run_endpoint(halsp, loaders, context.artifact_dir, device, seed, scorer, "none")
            endpoints.append(summary)
            if scorer == "role_aware_taylor":
                rat_flags[seed] = list(summary["checkpoint_flags"])
    stage_b = open_sparse_modulation_gate(rat_flags)
    if stage_b:
        for seed in SEEDS:
            for modulation in MODULATIONS:
                endpoints.append(
                    run_endpoint(
                        halsp,
                        loaders,
                        context.artifact_dir,
                        device,
                        seed,
                        "role_aware_taylor",
                        modulation,
                    )
                )
    if len(endpoints) not in {6, 10}:
        raise RoleAwareProtocolError("Adaptive DAG exceeded or missed its endpoint budget")
    decision = {
        "stage_b_open": stage_b,
        "rat_checkpoint_flags": rat_flags,
        "endpoint_count": len(endpoints),
        "official_test_was_unused_during_decision": True,
    }
    _json_write(context.artifact_dir / "branch_decision.json", decision)

    for endpoint in endpoints:
        endpoint["official_test"] = official_test_once(
            halsp, loaders, context.artifact_dir, device, endpoint
        )
    rows = [
        {
            "name": endpoint["name"],
            "seed": endpoint["seed"],
            "scorer": endpoint["scorer"],
            "modulation": endpoint["modulation"],
            "test_sparse_top1": endpoint["official_test"]["test_sparse_top1"],
            "primary_sparse_validation_top1": endpoint["primary_sparse_validation_top1"],
            "final_dense_validation_top1": endpoint["final_dense_validation_top1"],
        }
        for endpoint in endpoints
    ]
    with (context.artifact_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _json_write(
        context.artifact_dir / "study_summary.json",
        {
            "protocol": PROTOCOL_VERSION,
            "source_commit": SOURCE_COMMIT,
            "source_file_sha256": HALSP_ALL_SHA256,
            "confirmatory": False,
            "screening_seeds": list(SEEDS),
            "decision": decision,
            "endpoints": endpoints,
        },
    )
    return {
        "status": "completed",
        "endpoint_count": len(endpoints),
        "stage_b_open": stage_b,
        "compact": {
            "stage_b_open": stage_b,
            "endpoint_count": len(endpoints),
            "test_sparse_top1": {row["name"]: row["test_sparse_top1"] for row in rows},
        },
    }


def register_role_aware_handler(registry: Any) -> None:
    if "role_aware_v1" not in registry.names:
        registry.register("role_aware_v1", run_role_aware_v1)


def register_handlers(registry: Any) -> None:
    """Notebook-facing fixed registration hook (idempotent)."""

    register_role_aware_handler(registry)
