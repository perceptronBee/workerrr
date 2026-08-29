#!/usr/bin/env python3
# halsp_all.py
# ---------------------------------------------------------------------------
# HALSP ablation -- EVERYTHING IN ONE FILE.
#
# Single file on purpose: with four separate modules it was too easy to end up
# with a stale copy of one of them and get confusing errors. Nothing to import,
# nothing to keep in sync.
#
#   python halsp_all.py --selftest
#   python halsp_all.py --sweep --dataset cifar100  --variants v1_baseline v2_wanda v5_momwanda --seeds 0 1 --epochs 200 --out-dir ./runs_exp2
#   python halsp_all.py --sweep --dataset imagenet100 --variants v1_baseline v2_wanda v5_momwanda --seeds 0 --epochs 60 --data-root /content/imagenet100
#
# Datasets
#   cifar100     32x32,  cifar stem (no downsample), batch 256
#   imagenet100  224x224, resnet stem (7x7 s2 + maxpool), batch 128
# Runs are written to  <out-dir>/<dataset>/<variant>_s<seed>/  so the two
# datasets can never overwrite each other.
# ---------------------------------------------------------------------------

import os
import csv
import json
import time
import random
import shutil
import hashlib
import argparse
import functools
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== MODEL =====================

# --------------------------------------------------------------------------- #
#  fast deformable depthwise convolution
# --------------------------------------------------------------------------- #
def deform_depthwise(x, offset, weight, padding):
    """
    Deformable depthwise conv, numerically identical to
    torchvision.ops.deform_conv2d(x, offset, weight, padding=padding) when
    weight is depthwise (C, 1, k, k) -- verified to 1e-14 in float64, forward
    and all three gradients.

    Why not just call torchvision: its deform_conv2d builds an im2col buffer of
    C*k*k channels and then runs a batched GEMM. With groups=C that GEMM is 1x1
    per group, so the op is pure memory traffic with no arithmetic intensity --
    measured at ~6x slower epochs on ImageNet-100. Here each of the k*k kernel
    taps is a single grid_sample over all channels, scaled by that tap's
    per-channel weight and accumulated. No im2col buffer, and autograd comes
    free from grid_sample, so there is no hand-written backward to get wrong.

    Set HALSP_DEFORM_IMPL=torchvision to fall back (used to re-verify parity).
    """
    if os.environ.get("HALSP_DEFORM_IMPL") == "torchvision":
        from torchvision.ops import deform_conv2d
        return deform_conv2d(x, offset, weight, stride=1, padding=padding)

    N, C, H, W = x.shape
    k = weight.shape[-1]
    dev, dt = x.device, x.dtype

    gy, gx = torch.meshgrid(
        torch.arange(H, device=dev, dtype=dt),
        torch.arange(W, device=dev, dtype=dt),
        indexing="ij",
    )
    denom_y = max(H - 1, 1)
    denom_x = max(W - 1, 1)

    out = None
    for i in range(k):
        for j in range(k):
            t = i * k + j
            sy = gy + (i - padding) + offset[:, 2 * t]
            sx = gx + (j - padding) + offset[:, 2 * t + 1]
            grid = torch.stack(
                (2 * sx / denom_x - 1, 2 * sy / denom_y - 1), dim=-1
            )
            samp = F.grid_sample(x, grid, mode="bilinear",
                                 padding_mode="zeros", align_corners=True)
            w = weight[:, 0, i, j].view(1, C, 1, 1)
            out = samp * w if out is None else out + samp * w
    return out


# halsp_ablation.py
SCORING_MODES = ("momentum", "wanda", "momentum_wanda", "random")
_EPS = 1e-8


# --------------------------------------------------------------------------- #
#  small helpers
# --------------------------------------------------------------------------- #
def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Jaccard overlap of two 1-D index tensors."""
    if a is None or b is None or a.numel() == 0 or b.numel() == 0:
        return float("nan")
    sa, sb = set(a.tolist()), set(b.tolist())
    union = len(sa | sb)
    return len(sa & sb) / union if union else float("nan")


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation between two score vectors."""
    if x is None or y is None or x.numel() < 2:
        return float("nan")
    rx = x.argsort().argsort().double()
    ry = y.argsort().argsort().double()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = rx.norm() * ry.norm()
    return float((rx @ ry / d).item()) if d > 0 else float("nan")


# --------------------------------------------------------------------------- #
#  Stage
# --------------------------------------------------------------------------- #
class HalspStage(nn.Module):
    """
    One HALSP stage: a single shared master_weight used as
      (1) entry projection      C_in  -> C_mid
      (2) inner mixing          C_mid -> C_mid   (cyclic column shift per block)
      (3) exit projection       C_mid -> C_out   (transpose of the same tensor)
    plus per-block depthwise spatial filters stored in one fused tensor.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels,
        num_blocks,
        stride,
        focus_ratio=0.50,
        exploit_ratio=0.90,
        explore_ratio=0.05,
        gn_groups=1,
        kernel_size=3,
        scoring_mode="momentum",
        deformable=False,
        verbose=False,
    ):
        super().__init__()

        if scoring_mode not in SCORING_MODES:
            raise ValueError(f"unknown scoring_mode {scoring_mode!r}")
        pad = kernel_size // 2

        self.scoring_mode = scoring_mode
        self.deformable = deformable

        # base ratios (restored by set_phase("search"))
        self.base_focus_ratio = focus_ratio
        self.base_exploit_ratio = exploit_ratio
        self.base_explore_ratio = explore_ratio

        # live ratios (mutated by set_phase / set_strategy)
        self.dynamic_focus_ratio = focus_ratio
        self.dynamic_exploit_ratio = exploit_ratio
        self.dynamic_explore_ratio = explore_ratio

        self.gn_groups = gn_groups
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.stride = stride
        self._pad = pad

        self._col_shifts = [(i * mid_channels) % out_channels for i in range(num_blocks)]

        if verbose:
            print(
                f"[HalspStage] {num_blocks} blocks | W: {mid_channels}x{out_channels} "
                f"| scoring={scoring_mode}"
            )

        # ---- shared master weight ------------------------------------------
        self.master_weight = nn.Parameter(torch.empty(mid_channels, out_channels, 1, 1))
        nn.init.kaiming_normal_(self.master_weight, mode="fan_out", nonlinearity="relu")

        self.update_freq = 10
        self.ema_momentum = 0.01
        self.current_step = 0

        # ---- caches (persisted via extra_state) -----------------------------
        self.active_pool_cache = None      # Focus Pool indices
        self.dead_pool_cache = None        # Reserve Pool indices
        self.cached_indices = None         # current Active Channel Slice
        self._cached_col_compact = None
        self._cached_dw_idx = None

        # diagnostic references (persisted)
        self._prev_focus_pool = None       # previous refresh's Focus Pool -> churn
        self._prev_explore_idx = None      # last explored channels -> promotion rate

        # ---- fused depthwise filters ---------------------------------------
        self.dw_weight = nn.Parameter(
            torch.empty(num_blocks * mid_channels, 1, kernel_size, kernel_size)
        )
        for i in range(num_blocks):
            s, e = i * mid_channels, (i + 1) * mid_channels
            nn.init.kaiming_normal_(
                self.dw_weight.data[s:e], mode="fan_out", nonlinearity="relu"
            )
            self.dw_weight.data[s:e].mul_(
                1.0 + torch.randn(mid_channels, 1, kernel_size, kernel_size) * 0.02
            )

        self._dw_block_starts = [i * mid_channels for i in range(num_blocks)]

        # ---- deformable offset generators --------------------------------
        # One tiny conv per block. It reads a CHANNEL-AGNOSTIC summary of the
        # latent (mean + max over channels -> 2 maps) rather than the latent
        # itself, so the same module works whether the sparse path is running
        # with num_active channels or the dense path with mid_channels. That is
        # the whole trick: dynamic channel selection changes the channel count
        # every update_freq steps, and a normal offset conv would break on it.
        #
        # offset_groups = 1 -> 2*k*k offset channels shared across channels.
        # Zero-init means the model starts EXACTLY as the non-deformable one,
        # so any measured difference is learned, not an initialisation artefact.
        self.offset_convs = None
        if deformable:
            # Constructing Conv2d draws from the global RNG. Since these are
            # immediately zeroed, that draw is pure waste -- but it would shift
            # the RNG stream and give every later layer different weights than
            # the non-deformable model. Save/restore keeps the two models
            # bit-identical at init, so the ablation compares one change only.
            _rng = torch.get_rng_state()
            self.offset_convs = nn.ModuleList([
                nn.Conv2d(2, 2 * kernel_size * kernel_size, kernel_size,
                          stride=1, padding=pad, bias=True)
                for _ in range(num_blocks)
            ])
            torch.set_rng_state(_rng)
            for m in self.offset_convs:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
                m._is_offset_conv = True   # init_weights() must skip these

        self.exit_bn = nn.BatchNorm2d(out_channels)

        self.register_buffer(
            "_dw_offsets",
            torch.arange(num_blocks, dtype=torch.long) * mid_channels,
            persistent=False,
        )

        # ---- channel expansion ---------------------------------------------
        self.main_path_upsampler = None
        if in_channels != out_channels:
            extra = out_channels - in_channels
            self.main_path_upsampler = nn.Sequential(
                nn.Conv2d(in_channels, extra, kernel_size, 1, pad,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(extra),
            )

        # ---- skip-path downsample -------------------------------------------
        self.downsample_path = None
        if stride != 1:
            self.downsample_path = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size, stride, pad,
                          groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        # ---- dense-path column indices --------------------------------------
        base_cols = torch.arange(mid_channels)
        inner_cols = [
            (base_cols + (i * mid_channels) % out_channels) % out_channels
            for i in range(num_blocks)
        ]
        self.register_buffer("_all_inner_cols", torch.stack(inner_cols), persistent=False)

        # ---- persistent statistics -------------------------------------------
        self.register_buffer(
            "running_input_var", torch.zeros(out_channels, dtype=torch.float32)
        )
        self.register_buffer(
            "channel_usage", torch.zeros(mid_channels, dtype=torch.long)
        )

    # ------------------------------------------------------------------ #
    #  complete state  (this is what makes resume exact)
    # ------------------------------------------------------------------ #
    def get_extra_state(self):
        def cpu(t):
            return None if t is None else t.detach().cpu().clone()

        return {
            "current_step": int(self.current_step),
            "dynamic_focus_ratio": float(self.dynamic_focus_ratio),
            "dynamic_exploit_ratio": float(self.dynamic_exploit_ratio),
            "dynamic_explore_ratio": float(self.dynamic_explore_ratio),
            "active_pool_cache": cpu(self.active_pool_cache),
            "dead_pool_cache": cpu(self.dead_pool_cache),
            "cached_indices": cpu(self.cached_indices),
            "_cached_col_compact": cpu(self._cached_col_compact),
            "_cached_dw_idx": cpu(self._cached_dw_idx),
            "_prev_focus_pool": cpu(self._prev_focus_pool),
            "_prev_explore_idx": cpu(self._prev_explore_idx),
            "scoring_mode": self.scoring_mode,
        }

    def set_extra_state(self, state):
        if not state:
            return
        dev = self.master_weight.device

        saved_mode = state.get("scoring_mode", self.scoring_mode)
        if saved_mode != self.scoring_mode:
            raise RuntimeError(
                f"checkpoint scoring_mode={saved_mode!r} does not match "
                f"model scoring_mode={self.scoring_mode!r} -- wrong checkpoint?"
            )

        def dev_t(k):
            t = state.get(k)
            return None if t is None else t.to(dev)

        self.current_step = int(state["current_step"])
        self.dynamic_focus_ratio = float(state["dynamic_focus_ratio"])
        self.dynamic_exploit_ratio = float(state["dynamic_exploit_ratio"])
        self.dynamic_explore_ratio = float(state["dynamic_explore_ratio"])
        self.active_pool_cache = dev_t("active_pool_cache")
        self.dead_pool_cache = dev_t("dead_pool_cache")
        self.cached_indices = dev_t("cached_indices")
        self._cached_col_compact = dev_t("_cached_col_compact")
        self._cached_dw_idx = dev_t("_cached_dw_idx")
        self._prev_focus_pool = dev_t("_prev_focus_pool")
        self._prev_explore_idx = dev_t("_prev_explore_idx")

    # ------------------------------------------------------------------ #
    #  channel scoring
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _compute_scores(self, mode, momentum):
        """
        Returns (scores[mid], mode_actually_used).

        Fallbacks are explicit and reported so the diagnostics CSV can show how
        often a variant silently degraded to plain magnitude scoring.
        """
        w = self.master_weight.detach()
        mid, out = w.shape[0], w.shape[1]
        w2 = w.view(mid, out).abs()

        v = self.running_input_var
        var_ready = bool(torch.isfinite(v).all()) and float(v.sum()) > 1e-12
        act = v.clamp_min(0).sqrt() if var_ready else None   # Wanda uses ||X||_2

        mom_ready = momentum is not None

        if mode == "random":
            return torch.rand(mid, device=w.device), "random"

        if mode == "momentum":
            if not mom_ready:
                return w2.mean(dim=1), "magnitude_fallback"
            s = (w * momentum).abs().mean(dim=(1, 2, 3))
            if float(s.sum()) < 1e-9:
                return w2.mean(dim=1), "magnitude_fallback"
            return s, "momentum"

        if mode == "wanda":
            if act is None:
                return w2.mean(dim=1), "magnitude_fallback"
            s = (w2 * act.unsqueeze(0)).mean(dim=1)
            if float(s.sum()) < 1e-12:
                return w2.mean(dim=1), "magnitude_fallback"
            return s, "wanda"

        if mode == "momentum_wanda":
            if not mom_ready and act is None:
                return w2.mean(dim=1), "magnitude_fallback"
            if not mom_ready:
                return (w2 * act.unsqueeze(0)).mean(dim=1), "wanda_fallback"
            wm = (w * momentum).abs().view(mid, out)
            if act is None:
                s = wm.mean(dim=1)
                if float(s.sum()) < 1e-9:
                    return w2.mean(dim=1), "magnitude_fallback"
                return s, "momentum_fallback"
            s = (wm * act.unsqueeze(0)).mean(dim=1)
            if float(s.sum()) < 1e-12:
                return w2.mean(dim=1), "magnitude_fallback"
            return s, "momentum_wanda"

        raise ValueError(mode)

    @torch.no_grad()
    def _grow_scores_all(self):
        """
        The GROW criterion (opportunity-map cosine) evaluated for EVERY channel.
        Used only for diagnostics: its rank correlation with the PRUNE criterion
        is the numerical version of IEE's 'criterion consistency' argument.
        """
        if self.active_pool_cache is None or self.active_pool_cache.numel() == 0:
            return None
        w = self.master_weight.detach()
        mid, out = w.shape[0], w.shape[1]
        w2 = w.view(mid, out).abs()
        cov = w2.index_select(0, self.active_pool_cache).sum(dim=0)
        opp = self.running_input_var / (cov + _EPS)
        return F.normalize(w2, dim=1) @ F.normalize(opp, dim=0)

    # ------------------------------------------------------------------ #
    #  topology maintenance
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def run_topology_maintenance(self, external_momentum_map=None, diagnostics=None):
        """
        Re-rank channels into Focus / Reserve pools.
        If `diagnostics` is a list, one dict per stage is appended to it.
        Diagnostics consume NO RNG, so they cannot perturb run-to-run parity.
        """
        momentum = None
        if external_momentum_map is not None:
            momentum = external_momentum_map.get(id(self.master_weight))

        scores, used = self._compute_scores(self.scoring_mode, momentum)

        num_focus = int(self.mid_channels * self.dynamic_focus_ratio)
        sorted_idx = torch.sort(scores, descending=True)[1]

        new_focus = sorted_idx[:num_focus]
        new_reserve = sorted_idx[num_focus:]

        if diagnostics is not None:
            diagnostics.append(
                self._diagnostics(scores, used, new_focus, momentum, num_focus)
            )

        # promotion rate must be measured BEFORE we overwrite the pools
        self._prev_focus_pool = (
            self.active_pool_cache.clone() if self.active_pool_cache is not None else None
        )

        self.active_pool_cache = new_focus
        self.dead_pool_cache = new_reserve
        self._update_sparse_cache(self.get_focus_indices())

    @torch.no_grad()
    def _diagnostics(self, scores, used, new_focus, momentum, num_focus):
        """Cheap, RNG-free mechanism diagnostics computed at each refresh."""
        d = {
            "scoring_used": used,
            "n_focus": int(num_focus),
            "n_mid": int(self.mid_channels),
        }

        # (1) does the criterion even pick different channels than the others?
        for alt in ("momentum", "wanda", "momentum_wanda"):
            if alt == self.scoring_mode:
                continue
            alt_scores, alt_used = self._compute_scores(alt, momentum)
            if alt_used == "magnitude_fallback" and used == "magnitude_fallback":
                d[f"jaccard_vs_{alt}"] = 1.0
                continue
            alt_focus = torch.sort(alt_scores, descending=True)[1][:num_focus]
            d[f"jaccard_vs_{alt}"] = _jaccard(new_focus, alt_focus)

        # (2) churn: how much does the Focus Pool move between refreshes?
        d["focus_churn_jaccard"] = _jaccard(new_focus, self.active_pool_cache)

        # (3) explore effectiveness: did explored channels get promoted?
        if self._prev_explore_idx is not None and self._prev_explore_idx.numel() > 0:
            promoted = torch.isin(self._prev_explore_idx, new_focus).sum().item()
            d["explore_promotion_rate"] = promoted / self._prev_explore_idx.numel()
        else:
            d["explore_promotion_rate"] = float("nan")

        # (4) IEE criterion consistency: prune ranking vs grow ranking
        d["prune_grow_spearman"] = _spearman(scores, self._grow_scores_all())

        # (5) dead channels: never activated so far
        d["never_active_frac"] = float(
            (self.channel_usage == 0).float().mean().item()
        )
        return d

    @torch.no_grad()
    def set_strategy(self, new_exploit_ratio=None, new_explore_ratio=None,
                     new_focus_ratio=None):
        if new_exploit_ratio is not None:
            self.dynamic_exploit_ratio = new_exploit_ratio
        if new_explore_ratio is not None:
            self.dynamic_explore_ratio = new_explore_ratio
        if new_focus_ratio is not None:
            self.dynamic_focus_ratio = new_focus_ratio

        self.cached_indices = None
        self._cached_col_compact = None
        self._cached_dw_idx = None
        self.active_pool_cache = None
        self.dead_pool_cache = None

    @torch.no_grad()
    def _update_sparse_cache(self, active_idx):
        self.cached_indices = active_idx
        out_ch = self.out_channels
        self._cached_col_compact = torch.stack(
            [(active_idx + s) % out_ch for s in self._col_shifts]
        )
        self._cached_dw_idx = (
            active_idx.unsqueeze(0) + self._dw_offsets.unsqueeze(1)
        ).reshape(-1)
        self.channel_usage.index_add_(
            0, active_idx, torch.ones_like(active_idx, dtype=torch.long)
        )

    @torch.no_grad()
    def add_active_channel_decay(self, weight_decay):
        """Add L2 decay only to channel rows used by the current forward pass.

        ``master_weight`` and ``dw_weight`` are whole-tensor parameters.  Native
        SGD decay therefore also shrinks inactive Reserve rows.  When the
        active-only mode is enabled these tensors live in a no-decay optimizer
        group and this method restores the same L2 term only for the rows that
        actually participated in the minibatch.  Explore candidates selected
        for the current minibatch are active and therefore receive decay too.
        """
        if weight_decay <= 0:
            return
        if self.cached_indices is None or self._cached_dw_idx is None:
            raise RuntimeError("active-channel decay requires a populated sparse cache")
        if self.master_weight.grad is not None:
            idx = self.cached_indices
            self.master_weight.grad.index_add_(
                0, idx, self.master_weight.detach().index_select(0, idx) * weight_decay
            )
        if self.dw_weight.grad is not None:
            idx = self._cached_dw_idx
            self.dw_weight.grad.index_add_(
                0, idx, self.dw_weight.detach().index_select(0, idx) * weight_decay
            )

    @torch.no_grad()
    def get_focus_indices(self):
        if self.active_pool_cache is None:
            self.run_topology_maintenance(external_momentum_map=None)
            # Topology maintenance owns cache creation and accounting.  Return
            # that exact object so the caller can avoid selecting/accounting a
            # second time during first use or after a phase reset.
            return self.cached_indices

        w = self.master_weight.detach()
        device = w.device
        active_pool = self.active_pool_cache
        dead_pool = self.dead_pool_cache

        # ---- exploit ----
        n_exploit = int(len(active_pool) * self.dynamic_exploit_ratio)
        if len(active_pool) > 0:
            perm = torch.randperm(len(active_pool), device=device)
            exploit_idx = active_pool[perm[:n_exploit]]
        else:
            exploit_idx = torch.empty(0, device=device, dtype=torch.long)

        # ---- explore ----
        explore_idx = torch.empty(0, device=device, dtype=torch.long)
        n_explore = int(self.mid_channels * self.dynamic_explore_ratio)
        if len(dead_pool) > 0:
            n_explore = min(n_explore, len(dead_pool))
            if n_explore > 0 and self.current_step > 0:
                active_slice = torch.index_select(w, 0, active_pool)
                dead_slice = torch.index_select(w, 0, dead_pool)

                coverage = active_slice.view(len(active_pool), -1).abs().sum(dim=0)
                opportunity = self.running_input_var / (coverage + _EPS)

                dead_w = dead_slice.view(len(dead_pool), -1).abs()
                scores = torch.matmul(
                    F.normalize(dead_w, dim=1), F.normalize(opportunity, dim=0)
                ).add_(_EPS)
                probs = scores / scores.sum()
                chosen = torch.multinomial(probs, n_explore, replacement=False)
                explore_idx = dead_pool[chosen]
            elif n_explore > 0:
                perm = torch.randperm(len(dead_pool), device=device)
                explore_idx = dead_pool[perm[:n_explore]]

        self._prev_explore_idx = explore_idx.clone()

        final = torch.sort(torch.cat((exploit_idx, explore_idx), dim=0))[0]

        if self.gn_groups > 1:
            n_aligned = (len(final) // self.gn_groups) * self.gn_groups
            if n_aligned == 0:
                final = (
                    active_pool[: self.gn_groups]
                    if len(active_pool) >= self.gn_groups
                    else torch.arange(self.gn_groups, device=device, dtype=torch.long)
                )
            else:
                final = final[:n_aligned]
        return final

    # ------------------------------------------------------------------ #
    #  spatial op (plain depthwise, or deformable depthwise)
    # ------------------------------------------------------------------ #
    def _spatial(self, out, weight, groups, block_idx):
        if self.offset_convs is None:
            return F.conv2d(out, weight, stride=1, padding=self._pad,
                            groups=groups)
        stats = torch.cat(
            (out.mean(dim=1, keepdim=True), out.amax(dim=1, keepdim=True)),
            dim=1,
        )
        offset = self.offset_convs[block_idx](stats)
        return deform_depthwise(out, offset.to(out.dtype), weight, self._pad)

    # ------------------------------------------------------------------ #
    #  forward  (math identical to reference halsp.py when deformable=False)
    # ------------------------------------------------------------------ #
    def _forward_sparse(self, x_expanded, active_idx):
        num_active = len(active_idx)
        num_blocks, out_ch = self.num_blocks, self.out_channels

        w_entry = torch.index_select(self.master_weight, 0, active_idx)
        latent = F.conv2d(x_expanded, w_entry, stride=self.stride, padding=0)

        dw_all = torch.index_select(self.dw_weight, 0, self._cached_dw_idx)
        dw_weights = dw_all.split(num_active)

        w_2d = w_entry.view(num_active, out_ch)
        w_exp = w_2d.unsqueeze(0).expand(num_blocks, -1, -1)
        idx_exp = self._cached_col_compact.unsqueeze(1).expand(-1, num_active, -1)
        w_all = torch.gather(w_exp, 2, idx_exp).unsqueeze(-1).unsqueeze(-1)

        for i in range(num_blocks):
            out = F.gelu(latent)
            out = self._spatial(out, dw_weights[i], num_active, i)
            out = F.conv2d(out, w_all[i], stride=1, padding=0)
            latent = out.add_(latent)

        w_exit = w_entry.permute(1, 0, 2, 3).contiguous()
        return self.exit_bn(F.conv2d(latent, w_exit, stride=1, padding=0))

    def _forward_dense(self, x_expanded):
        w_entry = self.master_weight
        num_blocks, mid_ch = self.num_blocks, self.mid_channels
        dw_weight, block_starts = self.dw_weight, self._dw_block_starts

        latent = F.conv2d(x_expanded, w_entry, stride=self.stride, padding=0)

        w_2d = w_entry.view(mid_ch, self.out_channels)
        w_exp = w_2d.unsqueeze(0).expand(num_blocks, -1, -1)
        idx_exp = self._all_inner_cols.unsqueeze(1).expand(-1, mid_ch, -1)
        w_all = torch.gather(w_exp, 2, idx_exp).unsqueeze(-1).unsqueeze(-1)

        for i in range(num_blocks):
            out = F.gelu(latent)
            out = self._spatial(
                out, dw_weight[block_starts[i]: block_starts[i] + mid_ch],
                mid_ch, i,
            )
            out = F.conv2d(out, w_all[i], stride=1, padding=0)
            latent = out.add_(latent)

        w_exit = w_entry.permute(1, 0, 2, 3).contiguous()
        return self.exit_bn(F.conv2d(latent, w_exit, stride=1, padding=0))

    def forward(self, x):
        up = self.main_path_upsampler
        x_expanded = torch.cat((up(x), x), dim=1) if up is not None else x

        ds = self.downsample_path
        identity_global = ds(x_expanded) if ds is not None else x_expanded

        if self.training:
            self.current_step += 1
            should_update = (
                self.cached_indices is None
                or (self.current_step - 1) % self.update_freq == 0
            )
            if should_update:
                with torch.no_grad():
                    # Keep the EMA statistic in FP32 under autocast.  Computing
                    # variance first would leave ``input_var`` in BF16 and both
                    # lose precision and make the in-place lerp dtype-invalid.
                    input_var = x_expanded.detach().float().var(dim=(0, 2, 3))
                    self.running_input_var.lerp_(input_var, self.ema_momentum)
                selected_indices = self.get_focus_indices()
                if selected_indices is not self.cached_indices:
                    self._update_sparse_cache(selected_indices)
            out = self._forward_sparse(x_expanded, self.cached_indices)
        else:
            out = self._forward_dense(x_expanded)

        out.add_(identity_global)
        return F.gelu(out)



# --------------------------------------------------------------------------- #
#  Plain ResNet-50 reference (NO weight sharing, NO channel selection)
# --------------------------------------------------------------------------- #
def build_plain_resnet50(num_classes=100, stem_type="cifar"):
    """
    Stock torchvision ResNet-50, untouched except for the stem.

    This is the *architecture* reference: ~23.7M params against HALSP's ~1.7M.
    For cifar the 7x7/stride-2 stem + maxpool is replaced by a 3x3/stride-1 conv
    so that 32x32 inputs are not destroyed before layer1 -- the standard CIFAR
    adaptation, and the same thing HALSP's cifar stem does.
    """
    import torchvision
    m = torchvision.models.resnet50(weights=None, num_classes=num_classes)
    if stem_type == "cifar":
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
    elif stem_type != "imagenet":
        raise ValueError(f"unknown stem_type {stem_type!r}")
    return m


# --------------------------------------------------------------------------- #
#  Network
# --------------------------------------------------------------------------- #
class HalspResNet50(nn.Module):
    """ResNet-50 skeleton with every bottleneck stage replaced by a HalspStage."""

    def __init__(self, num_classes=100, scoring_mode="momentum",
                 deformable=False, sparsity=0.5, verbose=False,
                 widths=(64, 128, 256, 512), blocks=(3, 4, 6, 3),
                 stem_type="cifar", deform_stages=(3,)):
        super().__init__()
        stem_w = widths[0]
        self.in_channels = stem_w
        self.stages = nn.ModuleList()
        self.scoring_mode = scoring_mode
        self.stem_type = stem_type

        if stem_type == "cifar":
            # 32x32 inputs: no downsampling in the stem, resolution stays 32x32
            # into layer1 (matches the reference halsp.py / parity.py exactly).
            self.stem = nn.Sequential(
                nn.Conv2d(3, stem_w // 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(stem_w // 2), nn.GELU(),
                nn.Conv2d(stem_w // 2, stem_w, 3, 1, 1, bias=False),
                nn.BatchNorm2d(stem_w), nn.GELU(),
            )
        elif stem_type == "imagenet":
            # 224x224 inputs: standard ResNet stem, 224 -> 56 before layer1.
            # This is the block that was left commented out in the original
            # halsp.py ("STANDART RESNET STEM FOR 224X224 IMAGENET SIZE
            # PERFORMANCE COMPARISON") -- wired up here, not reinvented.
            self.stem = nn.Sequential(
                nn.Conv2d(3, stem_w, 7, 2, 3, bias=False),
                nn.BatchNorm2d(stem_w),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            raise ValueError(f"unknown stem_type {stem_type!r}")

        # Deformable placement. Default (3,) = layer4 only.
        #
        # torchvision's deform_conv2d has no fast depthwise path, so every
        # deformable block is far more expensive than the cuDNN depthwise conv
        # it replaces -- measured at ~6x epoch time when applied to layer3+4 on
        # ImageNet-100. layer3 dominates that cost (6 blocks at 14x14 vs
        # layer4's 3 blocks at 7x7), so restricting to layer4 keeps most of the
        # semantic benefit at a fraction of the runtime.
        self.deform_stages = tuple(deform_stages)

        def opts(idx):
            return dict(scoring_mode=scoring_mode,
                        deformable=deformable and (idx in self.deform_stages),
                        kernel_size=3, verbose=verbose)

        self.layer1 = self._make(widths[0], blocks[0], 1, 1.0, 1.0, 0.0, **opts(0))
        self.layer2 = self._make(widths[1], blocks[1], 2, 1.0, 1.0, 0.0, **opts(1))
        self.layer3 = self._make(widths[2], blocks[2], 2, sparsity, 0.9, 0.05, **opts(2))
        self.layer4 = self._make(widths[3], blocks[3], 2, sparsity, 0.9, 0.05, **opts(3))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[3] * 4, num_classes)

    def _make(self, mid, blocks, stride, focus, exploit, explore, **kw):
        out = mid * 4
        stage = HalspStage(self.in_channels, out, mid, blocks, stride,
                           focus, exploit, explore, gn_groups=1, **kw)
        self.in_channels = out
        self.stages.append(stage)
        return stage

    def run_topology_maintenance(self, external_momentum_map=None, diagnostics=None):
        for s in self.stages:
            s.run_topology_maintenance(external_momentum_map, diagnostics)

    def set_phase(self, phase):
        if phase in ("warmup", "cooldown"):
            for s in self.stages:
                s.set_strategy(new_focus_ratio=1.0, new_exploit_ratio=1.0,
                               new_explore_ratio=0.0)
        elif phase == "search":
            for s in self.stages:
                s.set_strategy(new_focus_ratio=s.base_focus_ratio,
                               new_exploit_ratio=s.base_exploit_ratio,
                               new_explore_ratio=s.base_explore_ratio)
        else:
            raise ValueError(phase)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        return self.fc(torch.flatten(x, 1))


def init_weights(model):
    """Reference initialisation (identical order to train.py in the repo)."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if getattr(m, "_is_offset_conv", False):
                continue          # stays zero -> deformable starts as a no-op
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    nn.init.normal_(model.fc.weight, 0, 0.01)
    nn.init.constant_(model.fc.bias, 0)
    return model


# ===================== ENGINE =====================



# --------------------------------------------------------------------------- #
#  determinism
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_backend(deterministic: bool = True):
    """
    NOTE: this model uses torch.gather on the shared weight; its backward is a
    scatter_add, which is non-deterministic on CUDA. Bit-exact reproducibility
    is therefore NOT attainable for this architecture on GPU, no matter what we
    set here -- two identical uninterrupted runs will already differ in the last
    few decimals. That is precisely why the protocol uses 3 seeds and reports
    mean +/- std. What resume guarantees is exact STATE restoration, i.e. it
    adds no divergence beyond that inherent noise.
    """
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except Exception:
        pass


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"])
                        else state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda_states = [s.cpu() if torch.is_tensor(s) else s for s in state["cuda"]]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)


# --------------------------------------------------------------------------- #
#  data  (shuffle + augmentation are a pure function of (seed, epoch))
# --------------------------------------------------------------------------- #
def epoch_seed(base_seed: int, epoch: int) -> int:
    return (base_seed * 1_000_003 + epoch * 9_176 + 12_345) % (2 ** 31 - 1)


def _worker_init(worker_id: int, base: int):
    s = (base + worker_id * 7919) % (2 ** 31 - 1)
    np.random.seed(s % (2 ** 32))
    random.seed(s)


def build_cifar100(data_root="./data", batch_size=256, num_workers=4):
    import torchvision
    import torchvision.transforms as T

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    trainset = torchvision.datasets.CIFAR100(data_root, True, train_tf, download=True)
    testset = torchvision.datasets.CIFAR100(data_root, False, test_tf, download=True)

    gen = torch.Generator()
    gen.manual_seed(0)  # reseeded per epoch

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False,          # required: fresh deterministic workers
        drop_last=False, generator=gen,
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=max(1, num_workers // 2), pin_memory=True,
        persistent_workers=False,
    )
    return train_loader, test_loader, gen


def prepare_imagenet100_layout(data_root):
    """
    The common Kaggle "ImageNet-100" download unpacks into
        data_root/train.X1/<class>/*.JPEG  ... train.X4/<class>/*.JPEG
        data_root/val.X/<class>/*.JPEG
    torchvision.datasets.ImageFolder needs one flat
        data_root/train/<class>/*.JPEG
        data_root/val/<class>/*.JPEG
    This creates that view with symlinks (no file copies, near-zero cost,
    safe to re-run) IF data_root/train doesn't already exist.
    Returns True if it created the layout, False if it was already there.
    """
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        return False

    shard_dirs = sorted(
        d for d in os.listdir(data_root)
        if d.startswith("train.X") and os.path.isdir(os.path.join(data_root, d))
    )
    if not shard_dirs:
        raise FileNotFoundError(
            f"Neither '{train_dir}' nor any 'train.X*' shard found under "
            f"{data_root} -- point data_root at the extracted ImageNet-100 folder."
        )

    os.makedirs(train_dir, exist_ok=True)
    for shard in shard_dirs:
        shard_path = os.path.join(data_root, shard)
        for cls in os.listdir(shard_path):
            src = os.path.join(shard_path, cls)
            dst = os.path.join(train_dir, cls)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

    val_src_candidates = ["val.X", "val", "validation"]
    val_src = next(
        (os.path.join(data_root, v) for v in val_src_candidates
         if os.path.isdir(os.path.join(data_root, v))),
        None,
    )
    if val_src is None:
        raise FileNotFoundError(
            f"No validation folder found (looked for {val_src_candidates}) "
            f"under {data_root}"
        )
    if not os.path.exists(val_dir):
        os.symlink(os.path.abspath(val_src), val_dir)

    return True


def build_imagenet100(data_root="./data/imagenet100", batch_size=128,
                      num_workers=8, image_size=224):
    """
    Standard ImageNet-style pipeline for the ImageNet-100 subset. Expects
    (or auto-builds, via prepare_imagenet100_layout) data_root/train and
    data_root/val, each with one subfolder per class.
    """
    import torchvision
    import torchvision.transforms as T

    prepare_imagenet100_layout(data_root)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = T.Compose([
        T.RandomResizedCrop(image_size),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.Resize(int(image_size * 256 / 224)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.ImageFolder(
        os.path.join(data_root, "train"), train_tf
    )
    testset = torchvision.datasets.ImageFolder(
        os.path.join(data_root, "val"), test_tf
    )

    gen = torch.Generator()
    gen.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False, drop_last=False, generator=gen,
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=max(1, num_workers // 2), pin_memory=True,
        persistent_workers=False,
    )
    return train_loader, test_loader, gen


DATASET_BUILDERS = {
    "cifar100": build_cifar100,
    "imagenet100": build_imagenet100,
}


def seed_loader_for_epoch(loader, gen, base_seed, epoch):
    """Make this epoch's shuffle + augmentation a pure function of (seed, epoch)."""
    es = epoch_seed(base_seed, epoch)
    gen.manual_seed(es)
    loader.worker_init_fn = functools.partial(_worker_init, base=es)


# --------------------------------------------------------------------------- #
#  checkpointing
# --------------------------------------------------------------------------- #
def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointManager:
    def __init__(self, run_dir, drive_dir=None, drive_every=10, fingerprint=""):
        self.run_dir = run_dir
        self.drive_dir = drive_dir
        self.drive_every = drive_every
        self.fingerprint = fingerprint
        os.makedirs(run_dir, exist_ok=True)
        if drive_dir:
            os.makedirs(drive_dir, exist_ok=True)

    @property
    def last(self):
        return os.path.join(self.run_dir, "ckpt_last.pt")

    @property
    def prev(self):
        return os.path.join(self.run_dir, "ckpt_prev.pt")

    def _atomic_save(self, payload, path):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        # read-back verification: a truncated/corrupt file must never be promoted
        probe = torch.load(tmp, map_location="cpu", weights_only=False)
        if probe.get("epoch") != payload["epoch"]:
            raise RuntimeError("checkpoint verification failed")
        del probe
        if os.path.exists(path):
            os.replace(path, self.prev)
        os.replace(tmp, path)

    def save(self, *, epoch, model, optimizer, scheduler, best_acc, history):
        payload = {
            "format": 3,
            "fingerprint": self.fingerprint,
            "epoch": epoch,                       # LAST COMPLETED epoch
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "history": history,
            "rng": capture_rng(),
        }
        self._atomic_save(payload, self.last)

        if self.drive_dir and ((epoch + 1) % self.drive_every == 0):
            self.sync_to_drive()

    def sync_to_drive(self):
        if not self.drive_dir:
            return
        for name in ("ckpt_last.pt", "metrics.csv", "diagnostics.csv", "config.json",
                     "pre_cooldown.pt"):
            src = os.path.join(self.run_dir, name)
            if os.path.exists(src):
                self.sync_file_to_drive(name)

    def sync_file_to_drive(self, name):
        """Atomically publish one run artifact to Drive, then verify its bytes."""
        if not self.drive_dir:
            return
        src = os.path.join(self.run_dir, name)
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        dst = os.path.join(self.drive_dir, name)
        tmp = dst + ".tmp"
        shutil.copyfile(src, tmp)
        if _sha256_file(src) != _sha256_file(tmp):
            raise RuntimeError(f"Drive sync verification failed for {name}")
        os.replace(tmp, dst)

    def load(self):
        """Try local last -> local prev -> drive copy. Returns payload or None."""
        candidates = [self.last, self.prev]
        if self.drive_dir:
            candidates.append(os.path.join(self.drive_dir, "ckpt_last.pt"))
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"[ckpt] corrupt, skipping {path}: {e}")
                continue
            if self.fingerprint and payload.get("fingerprint") != self.fingerprint:
                raise RuntimeError(
                    f"checkpoint at {path} belongs to a different config "
                    f"({payload.get('fingerprint')} != {self.fingerprint})"
                )
            print(f"[ckpt] resuming from {path} @ epoch {payload['epoch']}")
            return payload
        return None


# --------------------------------------------------------------------------- #
#  csv logging (flush + fsync so a hard kill never loses a row)
# --------------------------------------------------------------------------- #
class CsvLog:
    def __init__(self, path, fields):
        self.path = path
        self.fields = fields
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fields).writeheader()
                f.flush()
                os.fsync(f.fileno())

    def append(self, row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self.fields).writerow(
                {k: row.get(k, "") for k in self.fields}
            )
            f.flush()
            os.fsync(f.fileno())


# --------------------------------------------------------------------------- #
#  schedule
# --------------------------------------------------------------------------- #
def phase_boundaries(epochs: int, warmup_epochs=None, cooldown_epochs=None):
    """Return phase boundaries, with optional explicit phase lengths.

    Omitting both overrides preserves the original 5% / 80% / 15% recipe.
    """
    if epochs < 3:
        raise ValueError("epochs must leave room for warmup, search, and cooldown")
    warmup = (max(1, round(0.05 * epochs)) if warmup_epochs is None
              else int(warmup_epochs))
    cooldown_len = (max(1, round(0.15 * epochs)) if cooldown_epochs is None
                    else int(cooldown_epochs))
    if warmup < 1 or cooldown_len < 1:
        raise ValueError("warmup_epochs and cooldown_epochs must be positive")
    if warmup + cooldown_len >= epochs:
        raise ValueError("schedule must contain at least one search epoch")
    cooldown_start = epochs - cooldown_len
    return warmup, cooldown_start


def phase_boundaries_for_cfg(cfg):
    return phase_boundaries(
        cfg["epochs"], cfg.get("warmup_epochs"), cfg.get("cooldown_epochs")
    )


def phase_for_epoch(epoch, warmup_end, cooldown_start):
    if epoch < warmup_end:
        return "warmup"
    if epoch < cooldown_start:
        return "search"
    return "cooldown"


def build_optimizer(model, lr, weight_decay, momentum=0.9, manual_decay=()):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.endswith(suffix) for suffix in manual_decay):
            no_decay.append(p)
            continue
        (no_decay if (p.ndim <= 1 or name.endswith(".bias")) else decay).append(p)
    groups = [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)


def build_scheduler(optimizer, epochs, warmup_end, eta_min=1e-4):
    warm = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_end
    )
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_end, eta_min=eta_min
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warm, cos], milestones=[warmup_end]
    )


def momentum_map(optimizer):
    """id(param) -> SGD momentum buffer. Used as the channel-importance signal."""
    out = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p, {})
            if "momentum_buffer" in st and st["momentum_buffer"] is not None:
                out[id(p)] = st["momentum_buffer"]
    return out


# --------------------------------------------------------------------------- #
#  metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def topk_correct(output, target, topk=(1, 5)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
    return [correct[:k].reshape(-1).float().sum().item() for k in topk]


@torch.no_grad()
def measure_latency(model, device, input_size=(1, 3, 32, 32), warmup=30, iters=100):
    """Dense (eval-mode) inference latency. This is what actually ships."""
    model.eval()
    x = torch.randn(*input_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return {"latency_ms": dt * 1e3, "throughput_img_s": input_size[0] / dt}


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def config_fingerprint(cfg: dict) -> str:
    keys = ("variant", "seed", "epochs", "scoring_mode", "deformable",
            "sparsity", "batch_size", "lr", "weight_decay", "amp",
            "dataset", "stem_type", "widths", "blocks", "input_hw",
            "deform_stages", "warmup_epochs", "cooldown_epochs",
            "active_only_channel_decay")
    blob = json.dumps(
        {k: cfg[k] for k in keys if k in cfg and cfg[k] is not None},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
#  train / eval
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                    warmup_end, amp_dtype=None):
    model.train()
    loss_sum = c1 = c5 = n = 0.0
    clip = 5.0 if epoch < warmup_end else 4.0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        channel_decay = float(getattr(model, "active_channel_weight_decay", 0.0))
        if channel_decay > 0:
            for stage in getattr(model, "stages", ()):
                stage.add_active_channel_decay(channel_decay)
        optimizer.step()

        bs = labels.size(0)
        loss_sum += loss.item() * bs
        a1, a5 = topk_correct(outputs.detach().float(), labels)
        c1 += a1
        c5 += a5
        n += bs

    return {"train_loss": loss_sum / n, "train_top1": 100 * c1 / n,
            "train_top5": 100 * c5 / n}


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype=None):
    model.eval()
    loss_sum = c1 = c5 = n = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                outputs = model(inputs)
        else:
            outputs = model(inputs)
        outputs = outputs.float()
        loss_sum += criterion(outputs, labels).item() * labels.size(0)
        a1, a5 = topk_correct(outputs, labels)
        c1 += a1
        c5 += a5
        n += labels.size(0)
    return {"val_loss": loss_sum / n, "val_top1": 100 * c1 / n,
            "val_top5": 100 * c5 / n}


# ===================== RUNNER =====================




CONFIGS = {
    # architecture reference: stock ResNet-50, no HALSP machinery at all
    "v0_resnet50":     dict(scoring_mode=None,             deformable=False),
    "v1_baseline":     dict(scoring_mode="momentum",        deformable=False),
    "v2_wanda":        dict(scoring_mode="wanda",           deformable=False),
    "v3_deform":       dict(scoring_mode="momentum",        deformable=True),
    "v4_wanda_deform": dict(scoring_mode="wanda",           deformable=True),
    "v5_momwanda":     dict(scoring_mode="momentum_wanda",  deformable=False),
    "v6_momwanda_def": dict(scoring_mode="momentum_wanda",  deformable=True),
    # v7_random removed 2026-08: its only purpose was proving scoring beats
    # random selection, which the CIFAR-100 screen already settled decisively
    # (+1.6..1.9pp, t=6.7-17 vs. seed noise of 0.05-0.4). Keeping it in the
    # sweep would just burn compute on a already-answered question.
}

METRIC_FIELDS = [
    "epoch", "phase", "lr", "train_loss", "train_top1", "train_top5",
    "val_loss", "val_top1", "val_top5", "gap", "best_top1",
    "epoch_time_s", "resumed",
]

DIAG_FIELDS = [
    "epoch", "stage", "scoring_used", "n_focus", "n_mid",
    "jaccard_vs_momentum", "jaccard_vs_wanda", "jaccard_vs_momentum_wanda",
    "focus_churn_jaccard", "explore_promotion_rate",
    "prune_grow_spearman", "never_active_frac",
]


def build_everything(cfg, device):
    if cfg["scoring_mode"] is None:            # plain ResNet-50 reference
        model = build_plain_resnet50(
            num_classes=cfg["num_classes"],
            stem_type=cfg.get("stem_type", "cifar"),
        ).to(device)
        init_weights(model)
        optimizer = build_optimizer(model, cfg["lr"], cfg["weight_decay"])
        warmup_end, cooldown_start = phase_boundaries_for_cfg(cfg)
        scheduler = build_scheduler(optimizer, cfg["epochs"], warmup_end)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        return model, optimizer, scheduler, criterion, warmup_end, cooldown_start

    model = HalspResNet50(
        num_classes=cfg["num_classes"],
        scoring_mode=cfg["scoring_mode"],
        deformable=cfg["deformable"],
        sparsity=cfg["sparsity"],
        verbose=cfg["verbose"],
        widths=tuple(cfg.get("widths", (64, 128, 256, 512))),
        blocks=tuple(cfg.get("blocks", (3, 4, 6, 3))),
        stem_type=cfg.get("stem_type", "cifar"),
        deform_stages=tuple(cfg.get("deform_stages", (3,))),
    ).to(device)
    init_weights(model)

    active_only_decay = bool(cfg.get("active_only_channel_decay", False))
    model.active_channel_weight_decay = (
        float(cfg["weight_decay"]) if active_only_decay else 0.0
    )
    manual_decay = ("master_weight", "dw_weight") if active_only_decay else ()
    optimizer = build_optimizer(
        model, cfg["lr"], cfg["weight_decay"], manual_decay=manual_decay
    )
    warmup_end, cooldown_start = phase_boundaries_for_cfg(cfg)
    scheduler = build_scheduler(optimizer, cfg["epochs"], warmup_end)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    return model, optimizer, scheduler, criterion, warmup_end, cooldown_start


def train_variant(cfg, loader_factory=None):
    device = torch.device(cfg.get("device")
                          or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = cfg.get("dataset", "cifar100")
    # dataset is part of the path, not just the config, so a CIFAR-100 run and
    # an ImageNet-100 run of the same variant/seed can never collide on disk
    # even if --out-dir / --drive-dir are left at their shared defaults.
    run_dir = os.path.join(cfg["out_dir"], dataset, f"{cfg['variant']}_s{cfg['seed']}")
    os.makedirs(run_dir, exist_ok=True)

    drive_dir = None
    if cfg["drive_dir"]:
        drive_dir = os.path.join(
            cfg["drive_dir"], dataset, f"{cfg['variant']}_s{cfg['seed']}"
        )

    fp = config_fingerprint(cfg)
    ckpt = CheckpointManager(run_dir, drive_dir, cfg["drive_every"], fp)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**cfg, "fingerprint": fp}, f, indent=2)

    # ---- seed BEFORE constructing the model: init must be seed-dependent ----
    set_global_seed(cfg["seed"])
    configure_backend(deterministic=cfg["deterministic"])

    model, optimizer, scheduler, criterion, warmup_end, cooldown_start = \
        build_everything(cfg, device)

    factory = loader_factory or DATASET_BUILDERS[dataset]
    train_loader, test_loader, gen = factory(
        cfg["data_root"], cfg["batch_size"], cfg["num_workers"]
    )

    metrics_log = CsvLog(os.path.join(run_dir, "metrics.csv"), METRIC_FIELDS)
    diag_log = CsvLog(os.path.join(run_dir, "diagnostics.csv"), DIAG_FIELDS)

    amp_dtype = torch.bfloat16 if cfg["amp"] == "bf16" else None

    # ---------------- resume ----------------
    start_epoch, best_acc, history, resumed = 0, 0.0, [], False
    payload = ckpt.load()
    if payload is not None:
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        restore_rng(payload["rng"])
        best_acc = payload["best_acc"]
        history = payload["history"]
        start_epoch = payload["epoch"] + 1
        resumed = True
        if start_epoch >= cfg["epochs"]:
            print(f"[{cfg['variant']}] already finished.")
            return finalize(cfg, model, device, run_dir, ckpt, best_acc, history)
    else:
        # phase is applied at boundaries only; on resume the restored ratios
        # already encode the phase, so we must NOT re-apply it (that would clear
        # the Focus/Reserve pools and desynchronise the run).
        if cfg["scoring_mode"] is not None:
            model.set_phase("warmup")

    print(f"[{cfg['variant']} seed={cfg['seed']}] device={device} "
          f"params={count_params(model)/1e6:.2f}M "
          f"warmup<{warmup_end} search<{cooldown_start} epochs={cfg['epochs']} "
          f"{'(RESUMED)' if resumed else ''}")

    # ---------------- epochs ----------------
    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()

        # phase transition exactly at the boundary (resume-safe)
        if cfg["scoring_mode"] is not None:
            if epoch == warmup_end:
                model.set_phase("search")
            elif epoch == cooldown_start:
                model.set_phase("cooldown")
        phase = phase_for_epoch(epoch, warmup_end, cooldown_start)

        seed_loader_for_epoch(train_loader, gen, cfg["seed"], epoch)

        tr = train_one_epoch(model, train_loader, optimizer, criterion,
                               device, epoch, warmup_end, amp_dtype)
        va = evaluate(model, test_loader, criterion, device, amp_dtype)

        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # topology maintenance: search phase only, every 2 epochs
        if (cfg["scoring_mode"] is not None
                and warmup_end <= epoch < cooldown_start
                and (epoch + 1) % 2 == 0):
            diags = []
            model.run_topology_maintenance(momentum_map(optimizer), diags)
            for i, d in enumerate(diags):
                diag_log.append({"epoch": epoch, "stage": i, **d})

        if va["val_top1"] > best_acc:
            best_acc = va["val_top1"]
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))

        row = {
            "epoch": epoch, "phase": phase, "lr": round(lr_now, 6),
            **{k: round(v, 4) for k, v in {**tr, **va}.items()},
            "gap": round(tr["train_top1"] - va["val_top1"], 4),
            "best_top1": round(best_acc, 4),
            "epoch_time_s": round(time.time() - t0, 2),
            "resumed": int(resumed and epoch == start_epoch),
        }
        history.append(row)
        metrics_log.append(row)
        print(f"  e{epoch:3d} [{phase:8s}] loss {tr['train_loss']:.3f} "
              f"train {tr['train_top1']:.2f} val {va['val_top1']:.2f} "
              f"best {best_acc:.2f} ({row['epoch_time_s']}s)")

        ckpt.save(epoch=epoch, model=model, optimizer=optimizer,
                  scheduler=scheduler, best_acc=best_acc, history=history)

        if cfg.get("crash_after_epoch") == epoch:
            raise SystemExit(f"[simulated crash after epoch {epoch}]")

    return finalize(cfg, model, device, run_dir, ckpt, best_acc, history)


def finalize(cfg, model, device, run_dir, ckpt, best_acc, history):
    torch.save(model.state_dict(), os.path.join(run_dir, "final.pt"))
    hw = cfg.get("input_hw", 32)
    lat = measure_latency(model, device, (cfg["latency_batch"], 3, hw, hw))
    summary = {
        "variant": cfg["variant"], "seed": cfg["seed"],
        "best_top1": best_acc,
        "final_top1": history[-1]["val_top1"] if history else None,
        "final_top5": history[-1]["val_top5"] if history else None,
        "params_M": count_params(model) / 1e6,
        "mean_epoch_time_s": (
            sum(h["epoch_time_s"] for h in history) / len(history) if history else None
        ),
        **lat,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    ckpt.sync_to_drive()
    print(f"[{cfg['variant']} s{cfg['seed']}] DONE best={best_acc:.2f} "
          f"latency={lat['latency_ms']:.2f}ms")
    return summary


DATASET_DEFAULTS = {
    "cifar100": dict(
        num_classes=100, stem_type="cifar", input_hw=32,
        batch_size=256, num_workers=4, data_root="./data",
    ),
    "imagenet100": dict(
        num_classes=100, stem_type="imagenet", input_hw=224,
        batch_size=128, num_workers=8, data_root="./data/imagenet100",
    ),
}


def default_cfg(**over):
    dataset = over.get("dataset", "cifar100")
    ds_defaults = DATASET_DEFAULTS[dataset]

    cfg = dict(
        variant="v1_baseline", seed=0, epochs=120,
        warmup_epochs=None, cooldown_epochs=None,
        sparsity=0.5, lr=0.2, weight_decay=5e-4,
        amp="none", deterministic=True,
        out_dir="./runs", drive_dir=None, drive_every=10,
        latency_batch=1, verbose=False, device=None, crash_after_epoch=None,
        widths=(64, 128, 256, 512), blocks=(3, 4, 6, 3),
        deform_stages=(3,),
        active_only_channel_decay=False,
        dataset=dataset,
    )
    cfg.update(ds_defaults)   # dataset-specific defaults (can still be overridden below)
    cfg.update(over)          # explicit caller overrides win
    cfg.update(CONFIGS[cfg["variant"]])
    return cfg


def aggregate(out_dir):
    """Collect every summary.json (out_dir/<dataset>/<variant>_s<seed>/) into one table."""
    rows = []
    for dataset in sorted(os.listdir(out_dir)):
        ds_path = os.path.join(out_dir, dataset)
        if not os.path.isdir(ds_path):
            continue
        for d in sorted(os.listdir(ds_path)):
            p = os.path.join(ds_path, d, "summary.json")
            if os.path.exists(p):
                row = json.load(open(p))
                row["dataset"] = dataset
                rows.append(row)
    if not rows:
        print("no finished runs yet")
        return
    path = os.path.join(out_dir, "all_results.csv")
    fields = ["dataset"] + [k for k in rows[0] if k != "dataset"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)

    for dataset in sorted({r["dataset"] for r in rows}):
        by_variant = {}
        for r in rows:
            if r["dataset"] == dataset:
                by_variant.setdefault(r["variant"], []).append(r["best_top1"])
        print(f"\n[{dataset}]")
        print(f"{'variant':<18} {'n':>2} {'mean':>7} {'std':>6}")
        for v, accs in sorted(by_variant.items()):
            n = len(accs)
            mean = sum(accs) / n
            std = (sum((a - mean) ** 2 for a in accs) / max(1, n - 1)) ** 0.5
            print(f"{v:<18} {n:>2} {mean:>7.2f} {std:>6.2f}")
    print(f"\nwritten: {path}")



# ===================== SELF TEST =====================
def _fake_loaders(data_root, batch_size, num_workers):
    g = torch.Generator().manual_seed(1234)
    xtr = torch.randn(64, 3, 16, 16, generator=g)
    ytr = torch.randint(0, 10, (64,), generator=g)
    xte = torch.randn(32, 3, 16, 16, generator=g)
    yte = torch.randint(0, 10, (32,), generator=g)
    tr = torch.utils.data.TensorDataset(xtr, ytr)
    te = torch.utils.data.TensorDataset(xte, yte)
    gen = torch.Generator()
    gen.manual_seed(0)
    return (torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                        num_workers=0, generator=gen),
            torch.utils.data.DataLoader(te, batch_size=batch_size, shuffle=False,
                                        num_workers=0),
            gen)


_TINY = dict(num_classes=10, widths=(8, 16, 16, 16), blocks=(1, 1, 2, 1),
             batch_size=8, num_workers=0, input_hw=16, device="cpu",
             latency_batch=1, epochs=6, lr=0.05, stem_type="cifar")


def selftest():
    print("[1] scoring modes forward+backward")
    for mode in SCORING_MODES:
        set_global_seed(0)
        m = HalspResNet50(num_classes=10, scoring_mode=mode,
                          widths=(8, 16, 16, 16), blocks=(1, 1, 2, 1))
        m.set_phase("search")
        m.train()
        m(torch.randn(4, 3, 16, 16)).sum().backward()
        assert m.layer3.master_weight.grad is not None
        m.eval()
        with torch.no_grad():
            y = m(torch.randn(4, 3, 16, 16))
        assert torch.isfinite(y).all()
        print(f"   ok  {mode}")

    print("[1b] sparse cache selection accounting")
    set_global_seed(11)
    stage = HalspStage(
        in_channels=8,
        out_channels=8,
        mid_channels=8,
        num_blocks=1,
        stride=1,
        focus_ratio=0.5,
        exploit_ratio=1.0,
        explore_ratio=0.0,
        scoring_mode="momentum",
    )
    stage.update_freq = 2
    stage.train()

    def usage_after_one_selection(before):
        expected = torch.bincount(
            stage.cached_indices, minlength=stage.mid_channels
        )
        assert torch.equal(stage.channel_usage - before, expected)

    usage_zero = stage.channel_usage.clone()
    with torch.no_grad():
        stage(torch.randn(2, 8, 8, 8))
    usage_after_one_selection(usage_zero)
    usage_first = stage.channel_usage.clone()
    with torch.no_grad():
        stage(torch.randn(2, 8, 8, 8))
    assert torch.equal(stage.channel_usage, usage_first)
    with torch.no_grad():
        stage(torch.randn(2, 8, 8, 8))
    usage_after_one_selection(usage_first)
    usage_scheduled = stage.channel_usage.clone()
    stage.set_strategy(
        new_focus_ratio=0.5,
        new_exploit_ratio=1.0,
        new_explore_ratio=0.0,
    )
    with torch.no_grad():
        stage(torch.randn(2, 8, 8, 8))
    usage_after_one_selection(usage_scheduled)
    print("   ok  initial, scheduled, and phase-reset cache accounting")

    print("[2] imagenet stem at 224x224")
    m = init_weights(HalspResNet50(num_classes=100, scoring_mode="wanda",
                                   stem_type="imagenet"))
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 224, 224))
    assert y.shape == (2, 100), y.shape
    m.set_phase("search")
    m.train()
    m(torch.randn(2, 3, 224, 224)).sum().backward()
    print(f"   ok  imagenet stem -> {tuple(y.shape)}, backward ok")

    print("[3] dataset defaults")
    c1 = default_cfg(dataset="cifar100", variant="v1_baseline")
    c2 = default_cfg(dataset="imagenet100", variant="v1_baseline")
    assert c1["stem_type"] == "cifar" and c1["input_hw"] == 32
    assert c2["stem_type"] == "imagenet" and c2["input_hw"] == 224
    print(f"   ok  cifar={c1['stem_type']}/{c1['input_hw']}px/b{c1['batch_size']} "
          f"imagenet={c2['stem_type']}/{c2['input_hw']}px/b{c2['batch_size']}")

    print("[4] resume is bit-exact (2 simulated crashes)")
    root = tempfile.mkdtemp()
    try:
        configure_backend(deterministic=True)

        def cfg(out, **over):
            c = default_cfg(variant="v1_baseline", seed=0, out_dir=out, **_TINY)
            c.update(over)
            c.update(CONFIGS[c["variant"]])
            return c

        ref_dir = os.path.join(root, "ref")
        train_variant(cfg(ref_dir), loader_factory=_fake_loaders)
        ref = torch.load(os.path.join(ref_dir, "cifar100", "v1_baseline_s0",
                                      "final.pt"), map_location="cpu",
                         weights_only=False)

        cr_dir = os.path.join(root, "crash")
        for crash_at in (2, 4):
            try:
                train_variant(cfg(cr_dir, crash_after_epoch=crash_at),
                              loader_factory=_fake_loaders)
            except SystemExit as e:
                print(f"   {e}")
        train_variant(cfg(cr_dir), loader_factory=_fake_loaders)
        got = torch.load(os.path.join(cr_dir, "cifar100", "v1_baseline_s0",
                                      "final.pt"), map_location="cpu",
                         weights_only=False)

        bad = []
        for k in ref:
            a, b = ref[k], got[k]
            if torch.is_tensor(a):
                if not torch.equal(a, b):
                    bad.append(k)
            elif isinstance(a, dict):
                for kk in a:
                    va, vb = a[kk], b[kk]
                    if torch.is_tensor(va):
                        if not torch.equal(va, vb):
                            bad.append(f"{k}.{kk}")
                    elif va != vb:
                        bad.append(f"{k}.{kk}")
        assert not bad, f"{len(bad)} entries differ after resume: {bad[:5]}"
        print(f"   ok  all {len(ref)} state entries bit-identical")

        print("[5] corrupt checkpoint falls back")
        rd = os.path.join(cr_dir, "cifar100", "v1_baseline_s0")
        with open(os.path.join(rd, "ckpt_last.pt"), "wb") as f:
            f.write(b"\x00" * 512)
        ck = CheckpointManager(rd, None, 10,
                               config_fingerprint(cfg(cr_dir)))
        p = ck.load()
        assert p is not None
        print(f"   ok  recovered from ckpt_prev.pt @ epoch {p['epoch']}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\nALL CHECKS PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1_baseline", choices=list(CONFIGS))
    ap.add_argument("--dataset", default="cifar100",
                    choices=list(DATASET_DEFAULTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--warmup-epochs", type=int, default=None)
    ap.add_argument("--cooldown-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="default: 256 for cifar100, 128 for imagenet100")
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="default: 4 for cifar100, 8 for imagenet100")
    ap.add_argument("--amp", default="none", choices=["none", "bf16"])
    ap.add_argument("--no-deterministic", action="store_true")
    ap.add_argument("--out-dir", default="./runs")
    ap.add_argument("--data-root", default=None,
                    help="default: ./data for cifar100, ./data/imagenet100 for imagenet100")
    ap.add_argument("--drive-dir", default=None)
    ap.add_argument("--drive-every", type=int, default=10)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--deform-stages", type=int, nargs="+", default=[3],
                    help="stage indices to make deformable (0=layer1 .. 3=layer4); "
                         "default 3 = layer4 only")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    if a.aggregate:
        aggregate(a.out_dir)
        return

    common = dict(
        dataset=a.dataset,
        epochs=a.epochs, lr=a.lr,
        warmup_epochs=a.warmup_epochs, cooldown_epochs=a.cooldown_epochs,
        amp=a.amp, deterministic=not a.no_deterministic,
        out_dir=a.out_dir,
        drive_dir=a.drive_dir, drive_every=a.drive_every,
        deform_stages=tuple(a.deform_stages),
    )
    # only pass these through if the user actually set them -- otherwise let
    # DATASET_DEFAULTS pick the right value for cifar100 vs imagenet100
    if a.batch_size is not None:
        common["batch_size"] = a.batch_size
    if a.num_workers is not None:
        common["num_workers"] = a.num_workers
    if a.data_root is not None:
        common["data_root"] = a.data_root

    if a.sweep:
        variants = a.variants or [v for v in CONFIGS if not CONFIGS[v]["deformable"]]
        for v in variants:
            for s in a.seeds:
                try:
                    train_variant(default_cfg(variant=v, seed=s, **common))
                except NotImplementedError as e:
                    print(f"[skip] {v}: {e}")
                    break
        aggregate(a.out_dir)
    else:
        train_variant(default_cfg(variant=a.variant, seed=a.seed, **common))


if __name__ == "__main__":
    main()
