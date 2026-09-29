"""The device path (configs/cpu_gpu_equivalence.yaml#device_path): mp_retrieval.m3b_train's fit_model and
evaluate_carve copied under new names with an optional device. This is the route configs/universal_v2.yaml
#m3b_incumbents.must_not prescribes ("v2 code imports M3B modules unchanged or copies a function under a v2 name");
m3b_train.py keeps its pinned bytes.

With device None or "cpu" the copies run the pinned statements. The sampler (draw_indices), the packing (pack_parts,
BatchPrefetcher), the loss (listwise_loss), the metrics (rank_metrics, macro_recall_at_5) and the blocking
(node_budgeted_batches) are the pinned functions themselves, called by identity. With a device, the model moves once
before the optimiser is built, each batch moves on the main thread after BatchPrefetcher.next() (packing stays on CPU
threads), checkpoints hold CPU tensors, and the CUDA generator state is saved and restored beside the CPU one.

Systems code. It names no placement: a placement is nameable only after its verdict in
configs/cpu_gpu_equivalence.yaml, in a later dated authorization block that quotes the settings it passed with
(placement_settings below applies them; placement_block records them)."""

from __future__ import annotations

import copy
import dataclasses
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mp_retrieval.m3b_features import FAMILIES
from mp_retrieval.m3b_models import PackedBatch, listwise_loss
from mp_retrieval.m3b_train import (METRIC_NAMES, BatchPrefetcher, CarveData, EpochTooLong, FitRecord, draw_indices,
                                    macro_recall_at_5, node_budgeted_batches, pack_parts, rank_metrics)

CUBLAS_WORKSPACE = ":4096:8"
MODES = ("det", "default")


# ── devices ──────────────────────────────────────────────────────────────────


def on_device(device) -> bool:
    """True for an accelerator; None and "cpu" run the pinned CPU statements."""
    return device is not None and torch.device(device).type != "cpu"


def batch_to(batch: PackedBatch, device) -> PackedBatch:
    """Every field of a packed batch on ``device``, as a new PackedBatch; the batch it came from is untouched."""
    return dataclasses.replace(batch, **{f.name: getattr(batch, f.name).to(device) for f in dataclasses.fields(batch)})


def model_to(model: torch.nn.Module, device) -> torch.nn.Module:
    """The model on ``device``. Every tensor a frozen v2 model reads at forward time is a parameter or a registered
    buffer (the served relation bank is a non-persistent buffer, so it moves with the model), or is moved where it is
    read (family_mask); a parameter or buffer left behind is refused."""
    model.to(device)
    want = torch.device(device)
    for name, t in [*model.named_parameters(), *model.named_buffers()]:
        if t.device.type != want.type or (want.index is not None and t.device.index != want.index):
            raise RuntimeError(f"{name} stayed on {t.device}")
    return model


def cpu_copy(obj):
    """A detached CPU copy of every tensor in a (nested) state: what a checkpoint holds."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_copy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_copy(v) for v in obj)
    return obj


def _cuda_index(device) -> int:
    d = torch.device(device)
    if d.type != "cuda":
        raise NotImplementedError(f"device {d}: only cuda is covered by the device path")
    return d.index if d.index is not None else torch.cuda.current_device()


# ── settings and the placement record ────────────────────────────────────────


def placement_settings(mode: str | None, threads: int) -> dict:
    """Apply a tested mode: "det" (TF32 off, float32 matmul precision "highest", cudnn.benchmark off, deterministic
    algorithms requested with warn_only, CUBLAS_WORKSPACE_CONFIG set) or "default" (the same without the determinism
    request) or None (CPU work). CUBLAS_WORKSPACE_CONFIG must reach the process before CUDA starts, so a "det" caller
    sets it in the environment before torch initialises CUDA; this function refuses if it is missing. Returns the
    settings read back."""
    torch.set_num_threads(int(threads))
    if mode is not None:
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}: one of {MODES}")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.benchmark = False
        if mode == "det":
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE:
                raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG must be {CUBLAS_WORKSPACE} before CUDA starts")
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None:
                raise RuntimeError("mode default leaves the determinism request off: CUBLAS_WORKSPACE_CONFIG must be unset")
            torch.use_deterministic_algorithms(False)
    return read_settings()


def read_settings() -> dict:
    return {"threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def _driver_version() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def placement_block(device) -> dict:
    """placement_rules.record_the_placement: host, env name@hash, device and driver, torch and CUDA versions,
    determinism and TF32 flags, threads. A record written off the laptop carries this block."""
    dev = torch.device(device if device is not None else "cpu")
    out = {"host": platform.node(), "env": Path(sys.prefix).name, "python": platform.python_version(), "torch": torch.__version__,
           "numpy": np.__version__, "device": str(dev), "cpu": platform.processor(), "logical_cpus": os.cpu_count(), **read_settings()}
    if dev.type == "cuda":
        idx = _cuda_index(dev)
        out.update({"cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "device_name": torch.cuda.get_device_name(idx),
                    "driver": _driver_version()})
    return out


# ── the copies ───────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_carve_placed(model: torch.nn.Module, data: CarveData, batch_size: int = 32, families: tuple[str, ...] = FAMILIES,
                          max_nodes: int = 24_000, pack_workers: int = 2, prefetch_depth: int = 3, device=None) -> dict[str, np.ndarray]:
    """mp_retrieval.m3b_train.evaluate_carve with an optional device: the model must already be on it; each packed
    batch moves there after it is packed on CPU threads, and the scores come back to the CPU for rank_metrics."""
    placed = on_device(device)
    model.eval()
    rows: list[dict] = []
    blocks = node_budgeted_batches(data.pool_ptr, data.n_queries, batch_size, max_nodes)
    with BatchPrefetcher(pack_workers, prefetch_depth) as ahead:
        submitted = 0
        for idx in blocks:
            while ahead.pending < ahead.depth and submitted < len(blocks):
                ahead.submit(data.pack, blocks[submitted], families)
                submitted += 1
            batch = ahead.next()
            scores = model(batch_to(batch, device) if placed else batch).cpu().numpy()
            ptr = batch.qptr.numpy()
            for j, i in enumerate(idx):
                qd = data.query(int(i))
                rows.append(rank_metrics(scores[ptr[j]:ptr[j + 1]], qd["gold"], qd["gold_total"]))
    return {name: np.asarray([r[name] for r in rows], dtype=np.float64) for name in METRIC_NAMES}


def fit_model_placed(model: torch.nn.Module, fits: dict[str, CarveData], selects: dict[str, CarveData], *, seed: int, arm: str,
                     config: dict, max_epochs: int = 6, batches_per_epoch: int = 2000, batch_size: int = 16, patience: int = 2,
                     lr: float = 1e-3, weight_decay: float = 1e-4, clip: float = 1.0, families: tuple[str, ...] = FAMILIES,
                     dataset_draw: str = "per_query", epoch_limit_s: float | None = None, pack_workers: int = 2,
                     prefetch_depth: int = 3, checkpoint: Path | None = None, device=None, log=print) -> tuple[torch.nn.Module, FitRecord]:
    """mp_retrieval.m3b_train.fit_model with an optional device (see the module docstring); the rule, the sampler and
    the checkpoint contract are the pinned ones. A GPU fit is a new draw of the frozen rule (placement_rules.new_seeds):
    its dropout masks come from the device's generator. The returned model is on ``device``; the caller writes
    placement_block(device) into its record."""
    if dataset_draw not in ("per_query", "per_batch"):
        raise ValueError(f"dataset_draw must be per_query or per_batch, got {dataset_draw!r}")
    placed = on_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    names = sorted(fits)
    cursors = {n: [rng.permutation(fits[n].trainable), 0] for n in names}
    if placed:
        model_to(model, device)          # once, before the optimiser is built
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    record = FitRecord(arm=arm, config=dict(config), seed=seed, parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    best_state = cpu_copy(model.state_dict()) if placed else copy.deepcopy(model.state_dict())
    t0 = time.time()
    bad = 0
    first_epoch = 0
    if checkpoint is not None and Path(checkpoint).exists():
        ck = torch.load(checkpoint, weights_only=False)
        if ck["arm"] != arm or ck["seed"] != seed or ck["config"] != dict(config):
            raise ValueError(f"{checkpoint}: belongs to {ck['arm']} {ck['config']} seed {ck['seed']}, not to this fit")
        if placed != ("cuda_rng_state" in ck):
            raise ValueError(f"{checkpoint}: written by a fit on another device type; a fit resumes only on the device it began on")
        model.load_state_dict(ck["model"])
        optimiser.load_state_dict(ck["optimiser"])
        best_state = ck["best_state"]
        record = FitRecord(**ck["record"])
        bad = int(ck["bad"])
        rng.bit_generator.state = ck["numpy_rng_state"]
        torch.set_rng_state(ck["torch_rng_state"])
        if placed:
            torch.cuda.set_rng_state(ck["cuda_rng_state"], _cuda_index(device))
        cursors = {n: [np.asarray(perm), int(pos)] for n, (perm, pos) in ck["cursors"].items()}
        t0 = time.time() - float(ck["elapsed_s"])
        first_epoch = int(ck["epochs_done"])
        log(f"      resumed from {Path(checkpoint).name} after epoch {first_epoch - 1}")
        if ck["finished"]:
            model.load_state_dict(best_state)
            return model, record

    def save_checkpoint(epochs_done: int, finished: bool) -> None:
        if checkpoint is None:
            return
        tmp = Path(checkpoint).with_suffix(".tmp")
        state = {"arm": arm, "seed": seed, "config": dict(config), "model": model.state_dict(), "optimiser": optimiser.state_dict(),
                 "best_state": best_state, "record": asdict(record), "bad": bad, "numpy_rng_state": rng.bit_generator.state,
                 "torch_rng_state": torch.get_rng_state(), "cursors": {n: [perm, pos] for n, (perm, pos) in cursors.items()},
                 "elapsed_s": time.time() - t0, "epochs_done": epochs_done, "finished": finished}
        if placed:
            state["model"], state["optimiser"] = cpu_copy(state["model"]), cpu_copy(state["optimiser"])
            state["cuda_rng_state"] = torch.cuda.get_rng_state(_cuda_index(device))
        torch.save(state, tmp)
        tmp.replace(checkpoint)

    for epoch in range(first_epoch, max_epochs):
        model.train()
        t_epoch = time.time()
        losses = []
        with BatchPrefetcher(pack_workers, prefetch_depth) as ahead:
            drawn = 0
            for _ in range(batches_per_epoch):
                while ahead.pending < ahead.depth and drawn < batches_per_epoch:
                    ahead.submit(pack_parts, fits, draw_indices(fits, names, cursors, rng, batch_size, dataset_draw), families)
                    drawn += 1
                batch = ahead.next()
                if not bool(batch.gold.any()):
                    record.batches_skipped_no_gold += 1
                    continue
                if placed:
                    batch = batch_to(batch, device)
                optimiser.zero_grad(set_to_none=True)
                loss = listwise_loss(model(batch), batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimiser.step()
                losses.append(float(loss.detach()))
                record.steps += 1
        per_dataset = {n: evaluate_carve_placed(model, selects[n], families=families, pack_workers=pack_workers, prefetch_depth=prefetch_depth,
                                                device=device)
                       for n in sorted(selects)}
        macro = macro_recall_at_5(per_dataset)
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None, "select_macro_recall@5": macro,
                 "select_recall@5": {n: float(m["recall@5"].mean()) for n, m in per_dataset.items()}, "seconds": round(time.time() - t_epoch, 1)}
        record.history.append(entry)
        record.epochs_run = epoch + 1
        log(f"      epoch {epoch}: loss {entry['train_loss']:.4f} select macro R@5 {macro:.4f} ({entry['seconds']}s) " +
            " ".join(f"{n}={v:.3f}" for n, v in entry["select_recall@5"].items()))
        if macro > record.best_select_macro_recall5 + 1e-9:
            record.best_select_macro_recall5 = macro
            record.best_epoch = epoch
            best_state = cpu_copy(model.state_dict()) if placed else copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        early_stop = bad >= patience
        record.seconds = round(time.time() - t0, 1)
        save_checkpoint(epoch + 1, finished=early_stop)   # after max_epochs the resumed loop is simply empty
        if epoch_limit_s is not None and entry["seconds"] > epoch_limit_s:
            raise EpochTooLong(f"epoch {epoch} took {entry['seconds']}s, over the declared {epoch_limit_s:.0f}s: the screen halts; "
                               "file the fallback as an amendment before continuing")
        if early_stop:
            break
    model.load_state_dict(best_state)
    record.seconds = round(time.time() - t0, 1)
    return model, record
