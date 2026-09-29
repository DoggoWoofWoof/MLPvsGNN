"""MP-Approx level 5, track MP-ORACLE: fixed one-hop neighbour-distribution moments over level 3's compiled entries and
halo rows, read by an MLP beside node-local features, and the compiled-moment arm, a factorised kernel
phi(q, v)^T psi(u, e) in place of level 3's attention score (configs/mp_approx_l5.yaml).

    python scripts/mp_approx_l5.py --stage run --dataset metaqa       # host: moments, probe, (metaqa) repeat, read -- a fresh process each
    python scripts/mp_approx_l5.py --stage moments --dataset metaqa   # host CPU: moments.npy over level 0's U_q rows (float64 -> float16)
    python scripts/mp_approx_l5.py --stage probe --dataset metaqa     # host_gpu_det: the grid, cross-fitted -> out-of-fold predictions
    python scripts/mp_approx_l5.py --stage repeat --dataset metaqa    # host: the moments again, then metaqa's (r, L5-mom) unit again, into repeat/
    python scripts/mp_approx_l5.py --stage read --dataset metaqa      # host: level 0's quantities over this file's probes -> read.json
    python scripts/mp_approx_l5.py --stage doc                        # laptop: record.json from the read.json files, then the document
    python scripts/mp_approx_l5.py --stage file --date 2026_09_30 --commit <sha> --extra run_extra.json

Measurement only, as at levels 0 to 3: the targets are the GNN's outputs in level 0's sidecars, and nothing here becomes a
retriever, a feature, a teacher or a selection criterion. No GNN is run and no checkpoint is read. Level 0's and level 3's
scripts are imported unchanged; B0-mlp, L3-mean and L3-att are refitted with level 3's fit_units. Every number this file
compares is made on the host (moments on its CPU, probes and read on host_gpu_det); the laptop assembles records only.
"""

from __future__ import annotations

import os
import sys

HOST_STAGES = ("moments", "probe", "repeat", "read", "run")


def _stage_in_argv() -> str | None:
    if "--stage" in sys.argv:
        i = sys.argv.index("--stage")
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else None
    return None


if __name__ == "__main__":   # placement: BLAS and OpenMP pools, and cuBLAS's workspace, are fixed before numpy and torch load
    _HOST = _stage_in_argv() in HOST_STAGES
    _THREADS = "8" if _HOST else "6"
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_var] = _THREADS
    if _HOST:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from numpy.lib.format import open_memmap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import mp_approx_l0 as L0  # noqa: E402  (level 0, imported unchanged)
import mp_approx_l3 as L3  # noqa: E402  (level 3, imported unchanged)
from mp_retrieval.m3b_features import EDGE_ATTR, FAMILIES  # noqa: E402

CONFIG = ROOT / "configs" / "mp_approx_l5.yaml"
OUT = ROOT / "outputs" / "mp_approx_l5"
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "MP_APPROX_L5.md"
LF = chr(10)

DATASETS, SEEDS, FOLDS = L0.DATASETS, L0.SEEDS, L0.FOLDS
SPEC = L3.SPEC                                           # host_gpu_det, level 3's
B0_WIDTH, JL_DIM, HALO_WIDTH, N_ATTR = L3.B0_WIDTH, L3.JL_DIM, L3.HALO_WIDTH, L3.N_ATTR   # 258, 64, 322, 5
EPS_FAMILIES, EPS_WIDTH = L3.EPS_FAMILIES, L3.EPS_WIDTH  # (structural, ner, knn, self), 73
HEADS, HEAD_WIDTH, ATT_WIDTH = L3.HEADS, L3.HEAD_WIDTH, L3.ATT_WIDTH   # 4, 16, 64
ZCOS = 129                                               # halo column: the pool z-score of contract column 0, dense_cos
TAUS = (1.0, 0.25)
REL_TAU = 0.1
RC, RM = EDGE_ATTR.index("rel_compat"), EDGE_ATTR.index("rel_mask")
KERN_RANK = 32
HALO_NAMES = [f"slog{j:03d}" for j in range(129)] + [f"z{j:03d}" for j in range(129)] + [f"xr{j:02d}" for j in range(JL_DIM)]
EPS_NAMES = [f"family_{f}" for f in EPS_FAMILIES] + [f"attr_{a}" for a in EDGE_ATTR] + [f"rel_text{j:02d}" for j in range(JL_DIM)]
MOMENT_BLOCKS = (("self_xr", [f"xr{j:02d}" for j in range(JL_DIM)]), ("full_mean", HALO_NAMES), ("full_std", HALO_NAMES),
                 ("full_max", HALO_NAMES)) + tuple((f"family_mean_{f}", HALO_NAMES) for f in FAMILIES) + (
                ("query_kernel_tau1", HALO_NAMES), ("query_kernel_tau0.25", HALO_NAMES), ("top", HALO_NAMES),
                ("relation_kernel", HALO_NAMES), ("eps_mean", EPS_NAMES), ("attribute_max", list(EDGE_ATTR)),
                ("counts", ["all"] + list(FAMILIES)))
MOMENT_NAMES = [f"{block}|{c}" for block, cols in MOMENT_BLOCKS for c in cols]
N_MOMENTS = len(MOMENT_NAMES)                            # 3366
SL, _c = {}, 0
for _block, _cols in MOMENT_BLOCKS:
    SL[_block] = slice(_c, _c + len(_cols))
    _c += len(_cols)
ENTRY_BLOCK, ROW_CAP = 65536, 8192                       # moments: whole queries per chunk, at most these entries and rows

REFIT = ("B0-mlp", "L3-mean", "L3-att")                  # level 3's probes, refitted with level 3's fit_units
NEW = {"L5-mom": ("MOM", "MSE"), "L5-list": ("MOM", "LIST"), "L5-kern": ("KERN", "MSE")}   # features, objective
GRID = {"r": ("B0-mlp", "L3-mean", "L3-att", "L5-mom", "L5-list", "L5-kern"), "e": ("B0-mlp", "L3-mean", "L3-att", "L5-mom")}
PRIMARY = "L5-mom"
UNIT_ORDER = [("r", PRIMARY)] + [("r", p) for p in GRID["r"] if p != PRIMARY] + [("e", p) for p in GRID["e"]]
REPEAT = ("metaqa", "r", PRIMARY)
EVAL_BATCH_Q = {"MOM": 256, "KERN": 128}
CONTRASTS = {"moments_over_mean": ("r", "L5-mom", "L3-mean"), "moments_vs_attention": ("r", "L5-mom", "L3-att"),
             "moments_over_node_local": ("r", "L5-mom", "B0-mlp"), "kernel_vs_attention": ("r", "L5-kern", "L3-att"),
             "kernel_over_moments": ("r", "L5-kern", "L5-mom"), "objective_moments": ("r", "L5-list", "L5-mom"),
             "edge_moments_vs_attention": ("e", "L5-mom", "L3-att"), "edge_moments_over_mean": ("e", "L5-mom", "L3-mean")}
SHARES = {"moment_share": ("r", "L5-mom", "B0-mlp", "L3-att", "B0-mlp"), "kernel_share": ("r", "L5-kern", "B0-mlp", "L3-att", "B0-mlp")}
STRATA_PROBES = GRID["r"] + ("ref:other_seed",)
BAND_LABELS = {"L0_HIGH": "L5_HIGH", "L0_MID": "L5_MID", "L0_LOW": "L5_LOW", "NOT_READ": "NOT_READ"}
HARD_STOP_DIR = [OUT]   # a dataset's stages and the tests point it at their own directory


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def point_stops() -> None:
    L0.HARD_STOP_DIR[0] = L3.HARD_STOP_DIR[0] = L3.L1.HARD_STOP_DIR[0] = HARD_STOP_DIR[0]


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: level 0's hard_stop, pointed at this file's directory (level 3's and level 1's too)."""
    point_stops()
    L0.hard_stop(message, **evidence)


atomic_json = L3.atomic_json


def verify_inputs(decl: dict, datasets=DATASETS) -> None:
    """inputs: the LF pins of level 0's, level 1's, level 3's and the placement's files; level 0's sidecar meta.json and
    qids.json; level 3's compile meta.json, mirror.json and probe_meta.json. The same on the laptop and the host."""
    inp = decl["inputs"]
    code = [inp[lv][k] for lv in ("level0", "level1", "level3") for k in ("declaration_lf", "script_lf", "tests_lf")]
    code += [inp["placement"][k] for k in ("qualification_lf", "equivalence_lf", "equivalence_script_lf", "device_placement_lf")]
    for pin in code:
        path = ROOT / pin["path"]
        found = L0.lf_sha256(path) if path.exists() else None
        if found != pin["sha256"]:
            hard_stop(f"{pin['path']} is not its pinned sha256", path=pin["path"], pinned=pin["sha256"], found=found)
    for name in datasets:
        p0, p3 = inp["level0"]["sidecars"][name], inp["level3"]["sidecars"][name]
        for pins, key, rel in ((p0, "meta", "meta.json"), (p0, "qids", "qids.json"), (p3, "meta", "meta.json"),
                               (p3, "mirror", "mirror.json"), (p3, "probe_meta", "probes/probe_meta.json")):
            path = ROOT / pins["dir"] / rel
            found = L0.sha256_file(path) if path.exists() else None
            if found != pins[key]:
                hard_stop(f"{name}: {pins['dir']}/{rel} is not its pinned sha256", path=str(path), pinned=pins[key], found=found)


def host_checks(name: str) -> tuple[str, dict, list]:
    """A host process, before it reads a byte: the pins, level 3's mirror verification (imported unchanged; it re-hashes
    every level 0 and level 3 file sent), and host_gpu_det's settings applied and read back (level 3's host_placement)."""
    decl = load_declaration()
    verify_inputs(decl, (name,))
    point_stops()
    mirror_sha = L3.verify_mirror(name)
    place, deviations = L3.host_placement()
    return mirror_sha, place, deviations


# ── moments (host CPU) ───────────────────────────────────────────────────────


def _runs(r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For sorted r: the start of each run of equal values, its value and its length."""
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    return starts, r[starts], np.diff(np.r_[starts, r.size])


def _weighted_mean(X: np.ndarray, w: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(w[:, None] * X, starts, axis=0) / np.add.reduceat(w, starts)[:, None]


def chunk_moments(H: np.ndarray, row: np.ndarray, fam: np.ndarray, attr: np.ndarray, rel_text: np.ndarray,
                  self_xr: np.ndarray) -> np.ndarray:
    """moments for one chunk of whole queries, float64. H (m, 322): each entry's neighbour halo row, in entry order; row:
    the entry's U_q row within the chunk; fam: its family index; attr (m, 5); rel_text (m, 64); self_xr (n, 64). A row's
    entries are reduced in entry order (a stable sort by row). A row without an entry is 0 but for its own X R; a family
    mean over no entry of that family, and the relation kernel over no structural entry with rel_mask 1, are 0."""
    n = self_xr.shape[0]
    out = np.zeros((n, N_MOMENTS))
    out[:, SL["self_xr"]] = self_xr
    if row.size == 0:
        return out
    order = np.argsort(row, kind="stable")
    r, Hs, fs, As, Ts = row[order], H[order], fam[order], attr[order], rel_text[order]
    starts, rows, cnt = _runs(r)
    c = cnt.astype(np.float64)[:, None]
    mean = np.add.reduceat(Hs, starts, axis=0) / c
    out[rows, SL["full_mean"]] = mean
    D = Hs - np.repeat(mean, cnt, axis=0)
    out[rows, SL["full_std"]] = np.sqrt(np.add.reduceat(D * D, starts, axis=0) / c)
    out[rows, SL["full_max"]] = np.maximum.reduceat(Hs, starts, axis=0)
    for i, f in enumerate(FAMILIES):
        sel = fs == i
        if sel.any():
            s2, r2, c2 = _runs(r[sel])
            out[r2, SL[f"family_mean_{f}"]] = np.add.reduceat(Hs[sel], s2, axis=0) / c2.astype(np.float64)[:, None]
    zc = Hs[:, ZCOS]
    zmax = np.repeat(np.maximum.reduceat(zc, starts), cnt)
    for tau, block in zip(TAUS, ("query_kernel_tau1", "query_kernel_tau0.25")):
        out[rows, SL[block]] = _weighted_mean(Hs, np.exp((zc - zmax) / tau), starts)
    first = np.minimum.reduceat(np.where(zc == zmax, np.arange(r.size), r.size), starts)
    out[rows, SL["top"]] = Hs[first]
    rel = (fs == 0) & (As[:, RM] == 1.0)
    if rel.any():
        s3, r3, c3 = _runs(r[rel])
        rc = As[rel, RC]
        w = np.exp((rc - np.repeat(np.maximum.reduceat(rc, s3), c3)) / REL_TAU)
        out[r3, SL["relation_kernel"]] = _weighted_mean(Hs[rel], w, s3)
    onehot = np.eye(len(EPS_FAMILIES))[fs]
    out[rows, SL["eps_mean"]] = np.add.reduceat(np.concatenate([onehot, As, Ts], axis=1), starts, axis=0) / c
    out[rows, SL["attribute_max"]] = np.maximum.reduceat(As, starts, axis=0)
    out[rows, SL["counts"]] = np.log1p(np.concatenate([c, np.add.reduceat(onehot[:, :len(FAMILIES)], starts, axis=0)], axis=1))
    return out


def rel_text_of(slots: np.ndarray, rel_jl: np.ndarray) -> np.ndarray:
    """eps_e's rel_text as level 3 defines it: 8 times the mean of rel_jl over the entry's valid slots, 0 without one (float64)."""
    out = np.zeros((slots.shape[0], JL_DIM))
    if rel_jl.shape[0] == 0:
        return out
    valid = slots >= 0
    cnt = valid.sum(1)
    has = cnt > 0
    vec = np.asarray(rel_jl, dtype=np.float64)[np.where(valid, slots, 0)] * valid[..., None]
    out[has] = vec[has].sum(1) / cnt[has, None]
    return out * L3.REL_TEXT_SCALE


def compute_moments(sc: L0.Sidecar, d3: Path, path: Path, log=print) -> dict:
    """moments.npy (U_q rows x 3366, float16 through level 0's to_f16, level 0's row order) from level 3's sidecar, whole
    queries at a time; written atomically. Returns the facts for moments_meta.json."""
    t0 = time.time()
    if json.loads((d3 / "qids.json").read_text(encoding="utf-8")) != sc.qids:
        hard_stop(f"{d3}: not level 0's queries in level 0's order")
    halo = np.load(d3 / "halo.npy", mmap_mode="r")
    row_halo = np.load(d3 / "row_halo.npy").astype(np.int64)
    ent_row = np.load(d3 / "ent_row.npy").astype(np.int64)
    ent_halo = np.load(d3 / "ent_halo.npy").astype(np.int64)
    ent_family = np.load(d3 / "ent_family.npy").astype(np.int64)
    ent_attr = np.load(d3 / "ent_attr.npy")
    ent_slots = np.load(d3 / "ent_slots.npy")
    rel_jl = np.load(d3 / "rel_jl.npy")
    if halo.shape[1] != HALO_WIDTH or row_halo.size != sc.n_rows:
        hard_stop(f"{d3}: the halo is not the declared layout, or row_halo is not one entry per U_q row")
    ent_query = sc.query[ent_row].astype(np.int64)
    if np.any(np.diff(ent_query) < 0):
        hard_stop(f"{d3}: the entries are not grouped by query")
    ent_ptr = np.concatenate([[0], np.cumsum(np.bincount(ent_query, minlength=sc.n_q))]).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npy")
    mm = open_memmap(tmp, mode="w+", dtype=np.float16, shape=(sc.n_rows, N_MOMENTS))
    q0, chunks, empty = 0, 0, 0
    while q0 < sc.n_q:
        e_lim = np.searchsorted(ent_ptr, ent_ptr[q0] + ENTRY_BLOCK, side="right") - 1
        r_lim = np.searchsorted(sc.ptr, sc.ptr[q0] + ROW_CAP, side="right") - 1
        q1 = int(max(q0 + 1, min(e_lim, r_lim)))
        r0, r1, e0, e1 = int(sc.ptr[q0]), int(sc.ptr[q1]), int(ent_ptr[q0]), int(ent_ptr[q1])
        H = np.asarray(halo[ent_halo[e0:e1]], dtype=np.float64)
        self_xr = np.asarray(halo[row_halo[r0:r1], B0_WIDTH:], dtype=np.float64)
        M = chunk_moments(H, ent_row[e0:e1] - r0, ent_family[e0:e1], np.asarray(ent_attr[e0:e1], dtype=np.float64),
                          rel_text_of(ent_slots[e0:e1], rel_jl), self_xr)
        if not np.isfinite(M).all():
            bad = np.argwhere(~np.isfinite(M))[0]
            hard_stop("moments: a non-finite value", row=int(r0 + bad[0]), column=MOMENT_NAMES[int(bad[1])])
        mm[r0:r1] = L0.to_f16(M, "moments")
        empty += int(r1 - r0 - np.unique(ent_row[e0:e1]).size)
        q0, chunks = q1, chunks + 1
    mm.flush()
    del mm
    os.replace(tmp, path)
    return {"rows": sc.n_rows, "columns": N_MOMENTS, "queries": sc.n_q, "entries": int(ent_ptr[-1]), "rows_without_entry": empty,
            "chunks": chunks, "entry_block": ENTRY_BLOCK, "row_cap": ROW_CAP, "moments_sha256": L0.sha256_file(path),
            "seconds": round(time.time() - t0, 1)}


def stage_moments(name: str, log=print, host: bool = True, d: Path | None = None, d3: Path | None = None,
                  l0_dir: Path | None = None) -> dict:
    """moments: host CPU, inside the dataset's job before the probes; skipped on a restart when moments.npy is the file
    moments_meta.json records."""
    d, d3, l0_dir = d or OUT / name, d3 or L3.OUT / name, l0_dir or L0.OUT / name
    meta_path = d / "moments_meta.json"
    if meta_path.exists() and (d / "moments.npy").exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if L0.sha256_file(d / "moments.npy") == meta["moments_sha256"]:
            log(f"{name}: moments exist ({meta['moments_sha256'][:16]})")
            return meta
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if host:
            mirror_sha, place, deviations = host_checks(name)
        else:
            mirror_sha, place, deviations = None, {"device": "cpu", "test": True}, []
        columns = json.loads((l0_dir / "meta.json").read_text(encoding="utf-8"))["columns"]
        if columns[0] != "dense_cos" or len(columns) != 129:
            hard_stop(f"{name}: contract column 0 is not dense_cos", column0=columns[0], columns=len(columns))
        sc = L0.Sidecar(l0_dir, check=not host)   # on the host the mirror verification checked every array sent
        facts = compute_moments(sc, d3, d / "moments.npy", log)
        meta = {"dataset": name, "utc": L0.utc(), "git_head": L0.git_head(), **facts, "column_names": MOMENT_NAMES,
                "blocks": {b: [SL[b].start, SL[b].stop] for b, _ in MOMENT_BLOCKS}, "zcos_column": ZCOS, "taus": list(TAUS),
                "relation_tau": REL_TAU, "qids_sha256": L0.sha256_file(d3 / "qids.json"),
                "level3_meta_sha256": L0.sha256_file(d3 / "meta.json"), "level0_meta_sha256": L0.sha256_file(l0_dir / "meta.json"),
                "mirror_sha256_host": mirror_sha, "placement": place, "deviations": deviations, "cpu": platform.processor(),
                "numpy": np.__version__, "threads_env": os.environ.get("OPENBLAS_NUM_THREADS")}
        meta.update({"warnings": L3.warning_summary(caught) if host else [], "module_sha256": L3.module_shas()})
    atomic_json(meta_path, meta)
    log(f"{name}: moments {facts['rows']} x {facts['columns']} in {facts['seconds']:.0f}s ({facts['rows_without_entry']} rows "
        f"without an entry), sha256 {facts['moments_sha256'][:16]}")
    return meta


# ── probes (host_gpu_det) ────────────────────────────────────────────────────


class MomentData:
    """One dataset's moments on the device (float16), standardised per batch with level 0's rule: per fold, the training
    queries' U_q rows only; float64, two passes; sd < 1e-6 gives 0; clipped to [-8, 8]; float32."""

    def __init__(self, path: Path, sc: L0.Sidecar, device):
        self.np = np.load(path, mmap_mode="r")
        if self.np.shape != (sc.n_rows, N_MOMENTS) or self.np.dtype != np.float16:
            raise SystemExit(f"{path}: not U_q rows x {N_MOMENTS} float16")
        self.sc, self.device = sc, torch.device(device)
        self.m = torch.empty(self.np.shape, dtype=torch.float16, device=self.device)
        for r0 in range(0, self.np.shape[0], L0.ROW_BLOCK):
            self.m[r0:r0 + L0.ROW_BLOCK] = torch.from_numpy(np.array(self.np[r0:r0 + L0.ROW_BLOCK])).to(self.device)
        self._stats: dict = {}

    def stats(self, fold: int, train_q: np.ndarray):
        if fold in self._stats:
            return self._stats[fold]
        mask = self.sc.rows_of(train_q)
        s, n = np.zeros(N_MOMENTS), 0
        for r0 in range(0, self.sc.n_rows, L0.ROW_BLOCK):
            m = mask[r0:r0 + L0.ROW_BLOCK]
            if m.any():
                blk = np.asarray(self.np[r0:r0 + L0.ROW_BLOCK], dtype=np.float64)[m]
                s += blk.sum(0)
                n += blk.shape[0]
        mu = s / n
        ss = np.zeros(N_MOMENTS)
        for r0 in range(0, self.sc.n_rows, L0.ROW_BLOCK):
            m = mask[r0:r0 + L0.ROW_BLOCK]
            if m.any():
                D = np.asarray(self.np[r0:r0 + L0.ROW_BLOCK], dtype=np.float64)[m] - mu
                ss += (D * D).sum(0)
        sd = np.sqrt(ss / n)
        live = sd >= L0.SD_FLOOR
        scale = np.where(live, sd, 1.0)
        dev = self.device
        out = ((torch.from_numpy(mu).to(dev), torch.from_numpy(scale).to(dev), torch.from_numpy(live).to(dev)),
               {"columns": N_MOMENTS, "dead_columns": int((~live).sum()), "train_rows": int(n)})
        self._stats[fold] = out
        return out

    def standardised(self, rows: torch.Tensor, stats) -> torch.Tensor:
        mu, scale, live = stats
        z = (self.m[rows].to(torch.float64) - mu) / scale
        z = torch.where(live, z, torch.zeros((), dtype=torch.float64, device=z.device))
        return z.clamp(-L0.CLIP, L0.CLIP).to(torch.float32)


class L5Kern(torch.nn.Module):
    """probe.L5_kern, the compiled-moment arm: level 3's L3-att with the score replaced by the factorised kernel
    kappa_e^h = phi_h(q, v)^T psi_h(u, e), phi = softplus(W_phi [n_v | q_tilde]), psi = softplus(W_psi [n_u | eps_e]),
    4 heads of rank 32; alpha = kappa over its sum across the row's entries and its self entry. The values, the 4 heads
    of 16 and the readout on [b_v | m_v] are level 3's."""

    def __init__(self):
        super().__init__()
        self.w_phi = torch.nn.Linear(HALO_WIDTH + JL_DIM, HEADS * KERN_RANK)
        self.w_psi = torch.nn.Linear(HALO_WIDTH + EPS_WIDTH, HEADS * KERN_RANK)
        self.w_val = torch.nn.Linear(HALO_WIDTH + EPS_WIDTH, ATT_WIDTH)
        self.readout = torch.nn.Sequential(torch.nn.Linear(B0_WIDTH + ATT_WIDTH, L0.MLP_HIDDEN), torch.nn.GELU(),
                                           torch.nn.Linear(L0.MLP_HIDDEN, L0.MLP_HIDDEN), torch.nn.GELU(), torch.nn.Linear(L0.MLP_HIDDEN, 1))

    def parts(self, nh, q_rows, ent_u, ent_v, ent_eps, row_h, n_r: int):
        """u, v over the entries and then the self entries (level 3's), phi per row, psi and the values per entry."""
        loops = torch.arange(n_r, dtype=torch.long, device=nh.device)
        u = torch.cat([ent_u, row_h])
        v = torch.cat([ent_v, loops])
        self_eps = torch.zeros(n_r, EPS_WIDTH, dtype=ent_eps.dtype, device=ent_eps.device)
        self_eps[:, L3.SELF_COL] = 1.0
        eps = torch.cat([ent_eps, self_eps])
        phi = Fn.softplus(self.w_phi(torch.cat([nh[row_h], q_rows], dim=1))).view(n_r, HEADS, KERN_RANK)
        W, b = self.w_psi.weight, self.w_psi.bias
        psi = Fn.softplus(Fn.linear(nh, W[:, :HALO_WIDTH])[u] + Fn.linear(eps, W[:, HALO_WIDTH:], b)).view(-1, HEADS, KERN_RANK)
        W, b = self.w_val.weight, self.w_val.bias
        val = (Fn.linear(nh, W[:, :HALO_WIDTH])[u] + Fn.linear(eps, W[:, HALO_WIDTH:], b)).view(-1, HEADS, HEAD_WIDTH)
        return v, phi, psi, val

    def forward(self, bv, nh, q_rows, ent_u, ent_v, ent_eps, row_h, return_alpha: bool = False):
        n_r = bv.shape[0]
        v, phi, psi, val = self.parts(nh, q_rows, ent_u, ent_v, ent_eps, row_h, n_r)
        kappa = (phi[v] * psi).sum(-1)
        den = torch.zeros(n_r, HEADS, dtype=kappa.dtype, device=kappa.device).index_add_(0, v, kappa)
        alpha = kappa / den[v]
        m = torch.zeros(n_r, HEADS, HEAD_WIDTH, dtype=val.dtype, device=val.device).index_add_(0, v, val * alpha.unsqueeze(-1))
        out = self.readout(torch.cat([bv, m.view(n_r, ATT_WIDTH)], dim=1)).squeeze(-1)
        return (out, alpha) if return_alpha else out

    def compiled(self, bv, nh, q_rows, ent_u, ent_v, ent_eps, row_h):
        """The same output from node-level compiled moments, C_v = sum_e psi_e val_e^T and c_v = sum_e psi_e (psi reads
        only the neighbour's row and the edge), with m_v^h = phi_h^T C_v^h / phi_h^T c_v^h: the only query-time part is phi."""
        n_r = bv.shape[0]
        v, phi, psi, val = self.parts(nh, q_rows, ent_u, ent_v, ent_eps, row_h, n_r)
        C = torch.zeros(n_r, HEADS, KERN_RANK, HEAD_WIDTH, dtype=val.dtype, device=val.device).index_add_(0, v, psi.unsqueeze(-1) * val.unsqueeze(-2))
        c = torch.zeros(n_r, HEADS, KERN_RANK, dtype=psi.dtype, device=psi.device).index_add_(0, v, psi)
        m = torch.einsum("nhr,nhrw->nhw", phi, C) / torch.einsum("nhr,nhr->nh", phi, c).unsqueeze(-1)
        return self.readout(torch.cat([bv, m.reshape(n_r, ATT_WIDTH)], dim=1)).squeeze(-1)


def fit_new(pd: L3.ProbeData, md: MomentData | None, probe: str, y: torch.Tensor, teacher, base, bv: torch.Tensor, stats, mstats,
            fit_q: np.ndarray, val_q: np.ndarray, test_q: np.ndarray, seed: int):
    """probe.fitting for one fold, one seed and one of this file's probes: level 3's fit_probe loop with the probe's
    inputs -- the model built on the CPU after torch.manual_seed(seed), then moved to the device; minibatch order from
    default_rng(seed); AdamW; 64 queries per minibatch; at most 30 epochs; stop after 3 epochs without an
    inner-validation improvement of the probe's own objective and restore the best epoch."""
    kind, objective = NEW[probe]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = (L0.ProbeMLP(B0_WIDTH + N_MOMENTS) if kind == "MOM" else L5Kern()).to(pd.device)
    opt = torch.optim.AdamW(net.parameters(), lr=L0.MLP_LR, weight_decay=L0.MLP_WD)
    fit_idx, val_idx, test_idx = np.flatnonzero(fit_q), np.flatnonzero(val_q), np.flatnonzero(test_q)
    if fit_idx.size == 0 or val_idx.size == 0:
        raise SystemExit("an empty inner split")
    kern = kind == "KERN"

    def centred(B):
        if kern:
            nh = pd.halo_standardised(B["hs"], stats)
            p = net(bv[B["rows"]], nh, pd.q_tilde[B["row_q"]], B["ent_u"], B["ent_v"], pd.eps(B["es"]), B["row_h"])
        else:
            p = net(torch.cat([bv[B["rows"]], md.standardised(B["rows"], mstats)], dim=1))
        mean = torch.zeros(B["n_q"], dtype=p.dtype, device=p.device).index_add_(0, B["seg"], p) / B["sizes"]
        return p - mean[B["seg"]]

    def terms(pc, B):
        if objective == "MSE":
            return (pc - y[B["rows"]]) ** 2
        return L3.list_kl(pc, teacher[B["rows"]], base[B["rows"]], B["seg"], B["n_q"])

    def evaluate(qidx, want_pred=False):
        tot, n, preds = 0.0, 0, []
        with torch.no_grad():
            for b in range(0, qidx.size, EVAL_BATCH_Q[kind]):
                B = pd.batch(qidx[b:b + EVAL_BATCH_Q[kind]], kern)
                pc = centred(B)
                t = terms(pc, B)
                tot += float(t.sum())
                n += t.numel()
                if want_pred:
                    preds.append(pc.detach().to("cpu").numpy().astype(np.float64))
        return tot / n, (np.concatenate(preds) if want_pred else None)

    best, best_state, best_epoch, bad, curve = math.inf, None, 0, 0, []
    for epoch in range(1, L0.MLP_EPOCHS + 1):
        order = rng.permutation(fit_idx)
        for b in range(0, order.size, L0.MLP_BATCH_Q):
            B = pd.batch(order[b:b + L0.MLP_BATCH_Q], kern)
            loss = terms(centred(B), B).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        v, _ = evaluate(val_idx)
        curve.append(v)
        if v < best:
            best, best_epoch, bad = v, epoch, 0
            best_state = {key: t.detach().clone() for key, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= L0.MLP_PATIENCE:
                break
    if best_state is None:
        raise SystemExit(f"seed {seed}: no finite inner-validation loss")
    net.load_state_dict(best_state)
    _, pred = evaluate(test_idx, want_pred=True)
    return pred, {"seed": seed, "objective": objective, "epochs_run": len(curve), "best_epoch": best_epoch,
                  "inner_val": [float(v) for v in curve]}


def fit_units(pd: L3.ProbeData, md: MomentData | None, units, pdir: Path, log=print) -> dict:
    """probe.units in the declared order: level 3's probes through level 3's fit_units (imported unchanged), this file's
    through fit_new with level 3's unit layout -- the three seeds times five folds (fold outer, seed 1000 + 10 k + fold),
    written atomically with the fit log and skipped on a restart. Returns seconds per unit fitted here."""
    sc, dev = pd.sc, pd.device
    pdir.mkdir(parents=True, exist_ok=True)
    seconds, cache = {}, {}
    for target, probe in units:
        tag = f"{target}_{probe}"
        if probe in REFIT:
            seconds.update(L3.fit_units(pd, [(target, probe)], pdir, log))
            continue
        unit = pdir / f"unit_{tag}.npz"
        if unit.exists():
            log(f"   {tag}: exists")
            continue
        if NEW[probe][1] == "LIST" and target != "r":
            raise SystemExit(f"{tag}: the LIST objective is declared on r only")
        if not cache:
            r_c, e_c = L0.probe_targets(sc)
            z = np.load(sc.dir / "z.npy")
            cache["targets"] = {"r": {k: torch.from_numpy(r_c[:, k].astype(np.float32)).to(dev) for k in SEEDS},
                                "e": {k: torch.from_numpy(e_c[:, k].astype(np.float32)).to(dev) for k in SEEDS}}
            cache["teacher"] = {k: torch.from_numpy(z[:, 3 + k].astype(np.float32)).to(dev) for k in SEEDS}
            cache["base"] = {k: torch.from_numpy(z[:, k].astype(np.float32)).to(dev) for k in SEEDS}
            cache["b0"] = {}
            del z
        t0 = time.time()
        oof = {f"{target}/{probe}/{k}": np.full(sc.n_rows, np.nan) for k in SEEDS}
        flog = []
        for fold in range(FOLDS):
            train_q = sc.fold != fold
            test_q = ~train_q
            if fold not in cache["b0"]:
                cache["b0"][fold] = L0.standardised(sc, "B0", 0, train_q)
            Xs, st = cache["b0"][fold]
            bv = torch.from_numpy(Xs).to(dev)
            kern = NEW[probe][0] == "KERN"
            stats, hinfo = pd.halo_stats(fold, train_q) if kern else (None, None)
            mstats, minfo = (None, None) if kern else md.stats(fold, train_q)
            entry = {"fold": fold, "standardisation": st, "halo_standardisation": hinfo, "moment_standardisation": minfo, "fits": {}}
            test_rows = sc.rows_of(test_q)
            for k in SEEDS:
                pred, flog_k = fit_new(pd, md, probe, cache["targets"][target][k], cache["teacher"][k], cache["base"][k], bv, stats,
                                       mstats, train_q & ~sc.inner, train_q & sc.inner, test_q, 1000 + 10 * k + fold)
                oof[f"{target}/{probe}/{k}"][test_rows] = pred
                entry["fits"][str(k)] = flog_k
            del bv
            flog.append(entry)
            log(f"   {tag}: fold {fold} done, {time.time() - t0:.0f}s")
        for key, v in oof.items():
            if np.isnan(v).any():
                raise SystemExit(f"{tag}: {key} has rows that no fold predicted")
        tmp = pdir / f"unit_{tag}.tmp.npz"
        np.savez(tmp, **{key.replace("/", "|"): v for key, v in oof.items()})
        os.replace(tmp, unit)
        seconds[tag] = round(time.time() - t0, 1)
        atomic_json(pdir / f"fitlog_{tag}.json", {"seconds": seconds[tag], "folds": flog})
        log(f"   {tag}: written, {seconds[tag]:.0f}s")
    return seconds


def fit_settings() -> dict:
    return {**L3.fit_settings(), "kernel_rank": KERN_RANK, "eval_batch_queries_l5": EVAL_BATCH_Q}


def units_compare(a: Path, b: Path) -> tuple[bool, float]:
    """Whether two unit files hold the same keys with bit-identical arrays, and the largest absolute difference."""
    with np.load(a) as x, np.load(b) as y:
        if sorted(x.files) != sorted(y.files):
            raise SystemExit(f"{a} and {b} do not hold the same keys")
        identical = all(x[k].dtype == y[k].dtype and x[k].tobytes() == y[k].tobytes() for k in x.files)
        return bool(identical), max(float(np.max(np.abs(x[k] - y[k]))) for k in x.files)


def stage_probe(name: str, log=print, host: bool = True, device=None, d: Path | None = None, d3: Path | None = None,
                l0_dir: Path | None = None, repeat: bool = False) -> None:
    """probe (repeat=False) or repeat: metaqa's moments recomputed into repeat/ and compared with moments.npy, then its
    (r, L5-mom) unit fitted again on them and compared with the grid's. A test passes host=False and a CPU device."""
    d, d3, l0_dir = d or OUT / name, d3 or L3.OUT / name, l0_dir or L0.OUT / name
    t_all = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if host:
            mirror_sha, place, deviations = host_checks(name)
            device = SPEC["device"]
        else:
            mirror_sha, place, deviations = None, {"device": str(device), "test": True}, []
        mmeta = json.loads((d / "moments_meta.json").read_text(encoding="utf-8"))
        if L0.sha256_file(d / "moments.npy") != mmeta["moments_sha256"]:
            hard_stop(f"{name}: moments.npy is not the file moments_meta.json records")
        sc = L0.Sidecar(l0_dir, check=not host)
        pd = L3.ProbeData(sc, d3, device, check=not host)
        rep_moments = None
        if repeat:
            if name != REPEAT[0]:
                raise SystemExit(f"the repeat is declared on {REPEAT[0]} only")
            rpath = d / "repeat" / "moments.npy"
            if rpath.exists():
                rpath.unlink()
            facts = compute_moments(sc, d3, rpath, log)
            rep_moments = {"moments_sha256": {"probes": mmeta["moments_sha256"], "repeat": facts["moments_sha256"]},
                           "moments_identical": facts["moments_sha256"] == mmeta["moments_sha256"], "moments_seconds": facts["seconds"]}
            log(f"{name}: moments recomputed, byte-identical {rep_moments['moments_identical']}")
            units, pdir, mpath = [REPEAT[1:]], d / "repeat", rpath
        else:
            units, pdir, mpath = UNIT_ORDER, d / "probes", d / "moments.npy"
        md = MomentData(mpath, sc, device) if any(p in NEW and NEW[p][0] == "MOM" for _t, p in units) else None
        log(f"{name}: {sc.n_q} queries, {sc.n_rows} U_q rows, {pd.halo_np.shape[0]} halo rows, {int(pd.ent_sizes.sum())} entries, "
            f"{N_MOMENTS} moments; device {device}, threads {torch.get_num_threads()}")
        seconds = fit_units(pd, md, units, pdir, log)
        common = {"dataset": name, "utc": L0.utc(), "git_head": L0.git_head(), "spec": SPEC, "placement": place, "deviations": deviations,
                  "threads": torch.get_num_threads(), "seconds_this_process": round(time.time() - t_all, 1), "unit_seconds": seconds,
                  "mirror_sha256_host": mirror_sha, "level3_meta_sha256": L0.sha256_file(d3 / "meta.json"),
                  "level0_meta_sha256": L0.sha256_file(l0_dir / "meta.json"), "moments_sha256": mmeta["moments_sha256"],
                  "fitting": fit_settings()}
        if repeat:
            tag = f"{REPEAT[1]}_{REPEAT[2]}"
            pmeta = json.loads((d / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
            first = d / "probes" / f"unit_{tag}.npz"
            if L0.sha256_file(first) != pmeta["files_sha256"][f"unit_{tag}.npz"]:
                raise SystemExit(f"{first}: not the unit probe_meta.json files")
            identical, max_diff = units_compare(first, pdir / f"unit_{tag}.npz")
            info = {**common, **rep_moments, "unit": tag, "bit_identical": identical, "max_abs_diff": max_diff,
                    "unit_sha256": {"probes": L0.sha256_file(first), "repeat": L0.sha256_file(pdir / f"unit_{tag}.npz")}}
            info.update({"warnings": L3.warning_summary(caught) if host else [], "module_sha256": L3.module_shas()})
            atomic_json(pdir / "repeat.json", info)
            if rep_moments["moments_identical"]:   # its sha256 is filed; an identical 1-2 GB copy is not kept
                del md
                gc.collect()
                try:
                    mpath.unlink()
                except OSError as exc:
                    log(f"{name}: the identical repeat copy was kept ({exc})")
            log(f"{name}: repeat of {tag}: bit-identical {identical}, max |diff| {max_diff:.3e}")
            return
        files = sorted(p.name for p in pdir.glob("unit_*.npz")) + sorted(p.name for p in pdir.glob("fitlog_*.json"))
        want = sorted(f"unit_{t}_{p}.npz" for t, p in UNIT_ORDER) + sorted(f"fitlog_{t}_{p}.json" for t, p in UNIT_ORDER)
        if files != want:
            raise SystemExit(f"{pdir}: not the declared units ({files})")
        meta = {**common, "files_sha256": {f: L0.sha256_file(pdir / f) for f in files}, "grid": {k: list(v) for k, v in GRID.items()},
                "unit_order": [f"{t}/{p}" for t, p in UNIT_ORDER]}
        meta.update({"warnings": L3.warning_summary(caught) if host else [], "module_sha256": L3.module_shas()})
        atomic_json(pdir / "probe_meta.json", meta)
    log(f"{name}: probes done in {time.time() - t_all:.0f}s")


# ── read (host) ──────────────────────────────────────────────────────────────


def band_l5(point: float, interval, readable: bool) -> str:
    """readings.bands: level 0's thresholds under this file's labels."""
    return BAND_LABELS[L0.band(point, interval, readable)]


def relabel(family: dict) -> None:
    for v in family["probes"].values():
        v["band"] = BAND_LABELS[v["band"]]


def refit_comparison(d: Path, d3: Path) -> dict:
    """readings.flags.L3_REFIT_DIFFERS: each refitted unit against level 3's filed unit of the same name, bitwise;
    level 3's units through its probe_meta.json (pinned)."""
    meta3 = json.loads((d3 / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
    out = {}
    for t, p in UNIT_ORDER:
        if p not in REFIT:
            continue
        f = f"unit_{t}_{p}.npz"
        if L0.sha256_file(d3 / "probes" / f) != meta3["files_sha256"].get(f):
            hard_stop(f"level 3's {f} is not the unit its probe_meta.json files", path=str(d3 / "probes" / f))
        same, diff = units_compare(d / "probes" / f, d3 / "probes" / f)
        out[f"{t}/{p}"] = {"bit_identical": same, "max_abs_diff": diff}
    return out


def read_dataset(name: str, d: Path, d3: Path, l0_dir: Path, log=print) -> dict:
    """quantities: level 0's functions over this file's probes and the references, with level 0's resample matrix."""
    sc = L0.Sidecar(l0_dir, check=False)
    probes = L0.load_probes(d)
    pmeta = json.loads((d / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
    mmeta = json.loads((d / "moments_meta.json").read_text(encoding="utf-8"))
    if (pmeta["level3_meta_sha256"] != L0.sha256_file(d3 / "meta.json") or pmeta["level0_meta_sha256"] != L0.sha256_file(l0_dir / "meta.json")
            or pmeta["moments_sha256"] != mmeta["moments_sha256"]):
        hard_stop(f"{name}: the probes were not fitted on these sidecars and moments")
    z = np.load(l0_dir / "z.npy")
    is_gold = np.load(l0_dir / "is_gold.npy")
    qm = np.load(l0_dir / "q_metrics.npy")
    gold_total = np.load(l0_dir / "q_gold_total.npy")
    r_c, e_c = L0.probe_targets(sc)
    W = L0.boot_weights(sc.n_q)
    others = {k: [o for o in SEEDS if o != k] for k in SEEDS}

    def stack(fam, p):
        return np.stack([probes[f"{fam}/{p}/{k}"] for k in SEEDS], 1)
    preds = {fam: {p: stack(fam, p) for p in GRID[fam]} for fam in ("r", "e")}
    preds["r"]["ref:twin"] = np.zeros_like(r_c)
    preds["r"]["ref:other_seed"] = np.stack([r_c[:, others[k]].mean(1) for k in SEEDS], 1)
    preds["e"]["ref:no_edge"] = np.zeros_like(e_c)
    preds["e"]["ref:other_seed"] = np.stack([e_c[:, others[k]].mean(1) for k in SEEDS], 1)
    t0 = time.time()
    meas = {fam: L0.per_query_measures(sc, z, is_gold, gold_total, preds[fam], {"r": r_c, "e": e_c}[fam], fam) for fam in ("r", "e")}
    log(f"   {name}: per-query measures in {time.time() - t0:.0f}s")
    out = {"queries": sc.n_q, "uq_rows": sc.n_rows, "mean_uq": float(sc.sizes.mean()),
           "r": L0.read_family("r", meas["r"], qm, W), "e": L0.read_family("e", meas["e"], qm, W),
           "reproducibility": {"r": L0.reproducibility(r_c, sc, W), "e": L0.reproducibility(e_c, sc, W)}, "strata": {}}
    for s_name, mask in L0.strata_masks(name, sc).items():
        fam_r = L0.read_family("r", meas["r"], qm, W, mask)
        relabel(fam_r)
        out["strata"][s_name] = {"queries": fam_r["queries"], "denominators": fam_r["denominators"],
                                 "probes": {p: {"rho": fam_r["probes"][p]["rho"], "rho_bar": fam_r["probes"][p]["rho_bar"],
                                                "band": fam_r["probes"][p]["band"]} for p in STRATA_PROBES}}
    relabel(out["r"])
    relabel(out["e"])
    out["contrasts"] = contrasts_of(out)
    out["shares"] = shares_of(out)
    for fam in ("r", "e"):
        for v in out[fam]["probes"].values():
            v.pop("_rho_bar_boot", None)
    out["refit"] = refit_comparison(d, d3)
    out["repeat"] = None
    if name == REPEAT[0]:
        rj = d / "repeat" / "repeat.json"
        if not rj.exists():
            hard_stop(f"{name}: the declared repeat has not run before the read", path=str(rj))
        rep = json.loads(rj.read_text(encoding="utf-8"))
        out["repeat"] = {"unit": rep["unit"], "bit_identical": rep["bit_identical"], "max_abs_diff": rep["max_abs_diff"],
                         "moments_identical": rep["moments_identical"], "moments_sha256": rep["moments_sha256"],
                         "repeat_json_sha256": L0.sha256_file(rj)}
    out["moments"] = {k: mmeta[k] for k in ("rows", "columns", "entries", "rows_without_entry", "moments_sha256", "seconds")}
    out.update(readings(out))
    return out


def contrasts_of(out: dict) -> dict:
    """statistics.contrasts: the paired difference of rho_bar, point and 95% interval over the same resamples."""
    res = {}
    for c_name, (fam, a, b) in CONTRASTS.items():
        fp = out[fam]["probes"]
        entry = {"of": f"rho_bar({a}) - rho_bar({b}) on {fam}", "point": None, "ci": None}
        if out[fam]["readable_metrics"]:
            entry.update({"point": fp[a]["rho_bar"]["point"] - fp[b]["rho_bar"]["point"],
                          "ci": L0.ci(fp[a]["_rho_bar_boot"] - fp[b]["_rho_bar_boot"])})
        res[c_name] = entry
    return res


def shares_of(out: dict) -> dict:
    """statistics.shares: (x - b) / (y - c), point and 95% percentile interval over the same resamples; read only where the
    denominator's interval lies above 0."""
    res = {}
    for s_name, (fam, a, b, c, e) in SHARES.items():
        fp = out[fam]["probes"]
        entry = {"of": f"(rho_bar({a}) - rho_bar({b})) / (rho_bar({c}) - rho_bar({e})) on {fam}", "point": None, "ci": None,
                 "denominator": None, "readable": False}
        if out[fam]["readable_metrics"]:
            den_b = fp[c]["_rho_bar_boot"] - fp[e]["_rho_bar_boot"]
            den = {"point": fp[c]["rho_bar"]["point"] - fp[e]["rho_bar"]["point"], "ci": L0.ci(den_b)}
            entry["denominator"] = den
            if den["ci"][0] > 0:
                num_b = fp[a]["_rho_bar_boot"] - fp[b]["_rho_bar_boot"]
                with np.errstate(all="ignore"):
                    entry.update({"point": (fp[a]["rho_bar"]["point"] - fp[b]["rho_bar"]["point"]) / den["point"],
                                  "ci": L0.ci(num_b / den_b), "readable": True})
        res[s_name] = entry
    return res


def readings(out: dict) -> dict:
    """readings: the dataset's band (the primary probe's), the flags and every interpretation_map entry that applies."""
    rp, ep = out["r"]["probes"], out["e"]["probes"]
    reading = rp[PRIMARY]["band"]
    flags = []
    for fam in ("r", "e"):
        for p, v in out[fam]["probes"].items():
            if not p.startswith("ref:") and v["R2"]["point"] >= 0.5 and v["band"] == "L5_LOW":
                flags.append(f"FIT_NOT_RANK ({fam}, {p})")
    if out["reproducibility"]["r"]["mean"]["point"] < 0.5:
        flags.append("SEED_BOUND")
    rep = out.get("repeat")
    if rep is not None and not rep["bit_identical"]:
        flags.append(f"REPEAT_DIFFERS (max |diff| {rep['max_abs_diff']:.3e})")
    if rep is not None and not rep["moments_identical"]:
        flags.append("MOMENTS_DIFFER")
    for unit, v in out.get("refit", {}).items():
        if not v["bit_identical"]:
            flags.append(f"L3_REFIT_DIFFERS ({unit}, max |diff| {v['max_abs_diff']:.3e})")
    c = out["contrasts"]

    def low(key):
        return c[key]["ci"][0] if c[key]["ci"] is not None else None

    def high(key):
        return c[key]["ci"][1] if c[key]["ci"] is not None else None
    interp = []
    if reading == "L5_HIGH":
        interp.append("l5_high")
    if low("moments_over_mean") is not None and low("moments_over_mean") > 0:
        interp.append("moments_add_over_mean")
    if high("moments_vs_attention") is not None and high("moments_vs_attention") >= 0:
        interp.append("moments_not_below_attention")
    if high("moments_vs_attention") is not None and high("moments_vs_attention") < 0:
        interp.append("moments_below_attention")
    if high("kernel_vs_attention") is not None and high("kernel_vs_attention") >= 0:
        interp.append("kernel_not_below_attention")
    if high("kernel_vs_attention") is not None and high("kernel_vs_attention") < 0:
        interp.append("kernel_below_attention")
    if low("objective_moments") is not None and low("objective_moments") > 0:
        interp.append("objective_adds")
    if ep[PRIMARY]["band"] == "L5_HIGH" or (high("edge_moments_vs_attention") is not None and high("edge_moments_vs_attention") >= 0):
        interp.append("edge_effect_moments")
    return {"reading": reading, "flags": flags, "interpretation": interp}


def stage_read(name: str, log=print, host: bool = True, d: Path | None = None, d3: Path | None = None, l0_dir: Path | None = None) -> dict:
    """read: in the dataset's host job, after its probes (and metaqa's repeat); read.json beside the probes."""
    d, d3, l0_dir = d or OUT / name, d3 or L3.OUT / name, l0_dir or L0.OUT / name
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if host:
            mirror_sha, place, deviations = host_checks(name)
        else:
            mirror_sha, place, deviations = None, {"device": "cpu", "test": True}, []
        decl = load_declaration()
        out = read_dataset(name, d, d3, l0_dir, log)
        out.update({"dataset": name, "phase": decl["phase"], "utc": L0.utc(), "primary_probe": f"{PRIMARY} on r_k",
                    "declaration_lf_sha256": L0.lf_sha256(CONFIG), "level3_meta_sha256": L0.sha256_file(d3 / "meta.json"),
                    "probe_meta_sha256": L0.sha256_file(d / "probes" / "probe_meta.json"),
                    "moments_meta_sha256": L0.sha256_file(d / "moments_meta.json"),
                    "level0_meta_sha256": L0.sha256_file(l0_dir / "meta.json"), "mirror_sha256_host": mirror_sha, "spec": SPEC,
                    "placement": place, "deviations": deviations, "seconds": round(time.time() - t0, 1)})
        out.update({"warnings": L3.warning_summary(caught) if host else [], "module_sha256": L3.module_shas()})
    atomic_json(d / "read.json", out)
    log(f"{name}: read in {time.time() - t0:.0f}s -> {out['reading']}; flags {out['flags'] or 'none'}; "
        f"interpretation {out['interpretation'] or 'none'}")
    return out


def stage_run(name: str, log=print) -> None:
    """scheduling: the dataset's host job -- moments, probe, (metaqa) the repeat, the read, each in a fresh process; a
    failed stage stops the rest."""
    stages = ["moments", "probe"] + (["repeat"] if name == REPEAT[0] else []) + ["read"]
    for st in stages:
        log(f"== {name}: --stage {st} (fresh process)")
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", st, "--dataset", name], cwd=ROOT).returncode
        if rc != 0:
            raise SystemExit(f"{name}: --stage {st} exited {rc}; the later stages were not started")
    log(f"== {name}: done")


# ── doc (laptop) ─────────────────────────────────────────────────────────────


f3, fci = L0.f3, L0.fci


def filed_level3() -> dict:
    """Level 3 as its run record files it: the reading and the interpretation entries, per dataset (no values)."""
    decl = L3.load_declaration()
    key = next((k for k in decl if str(k).startswith("run_record_")), None)
    return {name: {"reading": v["reading"], "interpretation": v["interpretation"]} for name, v in (decl[key]["datasets"].items() if key else [])}


def assemble_record(root: Path, datasets) -> dict:
    """record: the host's read.json files, assembled without arithmetic, each with its sha256."""
    decl = load_declaration()
    rec = {"phase": decl["phase"], "utc": L0.utc(), "git_head": L0.git_head(), "declaration_lf_sha256": L0.lf_sha256(CONFIG),
           "registered_question": decl["registered_question"], "primary_probe": f"{PRIMARY} on r_k", "placement": "host_gpu_det",
           "datasets": {}}
    for name in datasets:
        path = root / name / "read.json"
        rec["datasets"][name] = {**json.loads(path.read_text(encoding="utf-8")), "read_sha256": L0.sha256_file(path)}
    return rec


def render_doc(rec: dict, decl: dict, metas: dict, filed3: dict) -> str:
    L = []
    add = L.append
    add("# MP-Approx level 5 (MP-ORACLE): fixed one-hop neighbour-distribution moments, and the compiled-moment arm")
    add("")
    add(f"Declared in `configs/mp_approx_l5.yaml` (status RUN; terminal STOP_FOR_REVIEW). Record: `outputs/mp_approx_l5/record.json` "
        f"(git-ignored), assembled {rec['utc']} at `{rec['git_head'][:7]}` from the host's per-dataset `read.json` files, without arithmetic.")
    add("")
    add(f"Registered question: \"{rec['registered_question']}\"")
    add("")
    add("Level 5 of the MP-Approx ladder, on the MP-ORACLE track, declared with level 4 (typed paths) on the user's message of "
        "2026-09-30. It replaces level 3's learned one-hop attention with fixed summaries of the same one-hop neighbourhood, and it "
        "carries the compiled-moment arm that level 3's l3_high entry licensed on 2wiki. As at every earlier level, the trained GNN's "
        "own outputs are the targets, which the proposal allows \"only to measure approximation capacity\". No probe, moment or "
        "kernel is a retriever, a teacher or a feature.")
    add("")
    add("**Placement: host-native.** The moments were computed on the host CPU inside each dataset's job; every probe this file "
        "compares, B0-mlp, L3-mean and L3-att included, was fitted and read on host_gpu_det as a new draw. No number here is set "
        "beside a number made on the laptop.")
    add("")
    add("## What was measured")
    add("")
    add("- **Moments** (3366 columns per U_q row, float64 from level 3's float16 halo rows, stored float16): v's own X R (64); the "
        "mean, population standard deviation and maximum of the in-neighbours' fixed halo rows (3 x 322); the mean per family "
        "(structural, ner, knn; 3 x 322); two query kernels, weights exp((zc_u - max zc) / tau) with zc_u the neighbour's pool "
        "z-score of dense_cos (halo column 129), tau 1 and 0.25 (2 x 322); the row of the most query-similar neighbour (322); a "
        "relation kernel over structural entries with rel_mask 1, weights exp((rel_compat - max) / 0.1) (322); the mean edge "
        "feature eps_e (73); the maximum of the five edge attributes (5); log1p of the entry counts, all and per family (4). One hop "
        "only, no self entry; a row without an entry keeps only its own X R.")
    add("- **L5-mom**: level 0's MLP on [b_v | M_v], M_v standardised with level 0's rule per fold. No attention and no learned "
        "weighting. **L5-list**: the same with level 3's listwise objective.")
    add("- **L5-kern** (the compiled-moment arm): level 3's L3-att with the score replaced by a positive factorised kernel "
        "kappa = phi(q, v)^T psi(u, e), phi = softplus(W_phi [n_v | q_tilde]), psi = softplus(W_psi [n_u | eps_e]), 4 heads of rank "
        "32. Because psi reads only the neighbour's fixed row and the edge, sum psi val^T and sum psi are node-level moments a "
        "deployed system could compile once; phi is the only query-time part. A laptop test holds the on-the-fly message equal to "
        "the compiled form.")
    add("- **Refitted from level 3** with its code, unchanged: B0-mlp (node-local), L3-mean (uniform weighting of the entries) and "
        "L3-att (learned query-conditioned one-hop attention, the ceiling).")
    add("- **Targets, recovery, U_q and bands** are level 0's: r_k = z(G_k) - z(T_k) (primary), e_k = z(G_k) - z(G0_k); rho_bar is the "
        "mean recovery over the readable metrics among recall@5, full_coverage@5 and hit@1. Metrics within U_q are an **upper bound** "
        "on full-pool metrics. L5_HIGH (>= 0.75, interval low >= 0.50), L5_LOW (<= 0.25, interval high <= 0.50), L5_MID otherwise.")
    add("")
    add("## Readings")
    add("")
    add("| dataset | queries | reading (L5-mom on r) | rho_bar [95% CI] | readable metrics | flags | interpretation |")
    add("|---|---:|---|---|---|---|---|")
    for name, ds in rec["datasets"].items():
        p = ds["r"]["probes"][PRIMARY]
        add(f"| {name} | {ds['queries']:,} | **{ds['reading']}** | {fci(p['rho_bar'])} | {', '.join(ds['r']['readable_metrics']) or 'none'} | "
            f"{'; '.join(ds['flags']) or 'none'} | {', '.join(ds['interpretation']) or 'none'} |")
    add("")
    add("### rho_bar per probe on r (host_gpu_det; one fit per cell, three GNN seeds x five folds)")
    add("")
    add("| dataset | " + " | ".join(GRID["r"]) + " | ref:other_seed |")
    add("|---|" + "---|" * (len(GRID["r"]) + 1))
    for name, ds in rec["datasets"].items():
        rp = ds["r"]["probes"]
        add(f"| {name} | " + " | ".join(f"{fci(rp[p]['rho_bar'])} ({rp[p]['band']})" for p in GRID["r"] + ("ref:other_seed",)) + " |")
    add("")
    add("### Contrasts of rho_bar (paired bootstrap, level 0's resample matrix)")
    add("")
    add("| contrast | of | " + " | ".join(rec["datasets"]) + " |")
    add("|---|---|" + "---|" * len(rec["datasets"]))
    for c, (fam, a, b) in CONTRASTS.items():
        add(f"| {c} | {a} - {b}, {fam} | " + " | ".join(fci(ds["contrasts"][c]) for ds in rec["datasets"].values()) + " |")
    add("")
    add("### Shares of attention's gain over node-local (descriptive; read only where the denominator's interval lies above 0)")
    add("")
    add("| share | of | " + " | ".join(rec["datasets"]) + " |")
    add("|---|---|" + "---|" * len(rec["datasets"]))
    for s, (fam, a, b, c, e) in SHARES.items():
        cells = []
        for ds in rec["datasets"].values():
            v = ds["shares"][s]
            cells.append(fci(v) if v["readable"] else ("not read (denominator " + fci(v["denominator"]) + ")" if v["denominator"] else "not read"))
        add(f"| {s} | ({a} - {b}) / ({c} - {e}), {fam} | " + " | ".join(cells) + " |")
    add("")
    imap = decl["readings"]["interpretation_map"]
    used = sorted({i for ds in rec["datasets"].values() for i in ds["interpretation"]})
    if used:
        add("What the interpretation entries say, as filed before any number (they are not exclusive, and none opens a stage):")
        add("")
        for i in used:
            add(f"- **{i}**: {' '.join(str(imap[i]).split())}")
        add("")
    add("### Level 3, as filed (bands and interpretation entries only)")
    add("")
    add("| dataset | level 3 reading | level 3 interpretation |")
    add("|---|---|---|")
    for name in rec["datasets"]:
        f = filed3.get(name, {})
        add(f"| {name} | {f.get('reading', 'n/a')} | {', '.join(f.get('interpretation', [])) or 'none'} |")
    add("")
    for name, ds in rec["datasets"].items():
        mo = ds["moments"]
        add(f"## {name}")
        add("")
        add(f"{ds['queries']:,} V2_GATE queries ({L0.POPULATION_NOTE.get(name, 'all of them')}), {ds['uq_rows']:,} U_q rows (mean |U_q| "
            f"{ds['mean_uq']:.1f}); {mo['entries']:,} one-hop entries; {mo['rows_without_entry']:,} rows without an entry; moments "
            f"{mo['rows']:,} x {mo['columns']} in {mo['seconds'] / 60:.1f} min on the host CPU (sha256 `{mo['moments_sha256'][:16]}`).")
        add("")
        for fam, label, base in (("r", "r_k = z(G_k) - z(T_k), the GNN over its twin", "T_k"),
                                 ("e", "e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN", "G0_k")):
            F = ds[fam]
            add(f"### Target {label}")
            add("")
            add("| metric | gap M(G_k) - M(" + base + ") | 95% CI | readable |")
            add("|---|---:|---|---|")
            for m, v in F["denominators"].items():
                add(f"| {m} | {f3(v['gap'])} | [{f3(v['ci'][0])}, {f3(v['ci'][1])}] | {'yes' if v['readable'] else 'no'} |")
            add("")
            add("| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |")
            add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
            for p, v in F["probes"].items():
                rho = v["rho"]
                cells = [f3(rho[m]["point"]) + ("" if rho[m]["readable"] else " (nr)") for m in L0.RETRIEVAL]
                add(f"| {p} | {f3(v['R2']['point'])} | {f3(v['spearman']['point'])} | {f3(v['DPR']['point'])} | {f3(v['gold_DPR']['point'])} | "
                    f"{f3(v['top5_overlap']['point'])} | {' | '.join(cells)} | {fci(v['rho_bar'])} | {v['band']} |")
            add("")
        rep = ds["reproducibility"]
        add(f"Seed reproducibility of the targets (mean of the three seed pairs): r {fci(rep['r']['mean'])}; e {fci(rep['e']['mean'])}.")
        add("")
        add("### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)")
        add("")
        add("| stratum | queries | readable | " + " | ".join(f"{p} rho_bar" for p in STRATA_PROBES) + " |")
        add("|---|---:|---|" + "---|" * len(STRATA_PROBES))
        for s, v in ds["strata"].items():
            readable = [m for m, x in v["denominators"].items() if x["readable"]]
            add(f"| {s} | {v['queries']:,} | {', '.join(readable) or 'none'} | " + " | ".join(fci(v["probes"][p]["rho_bar"]) for p in STRATA_PROBES) + " |")
        add("")
    add("## What this does not say")
    add("")
    add("- Every probe is an oracle fit on the GNN's own outputs over V2_GATE queries of the same dataset. Recovery measures capacity "
        "within U_q; it never describes a deployable model, and no probe output, moment or kernel enters any retriever, feature, "
        "teacher or selection.")
    add("- A model that used these moments or the factorised kernel without GNN outputs would be a competitor, and would need its own "
        "declaration under the QLS-U contract. This file does not open one.")
    add("- No reading here says that message passing is unnecessary, that it is not needed, or that the MLP wins; low and high "
        "recovery are both results (readings.wording).")
    add("- No later level or arm is opened by any reading here.")
    add("")
    add("## Reproducibility, placement and compute")
    add("")
    add("| dataset | moments (min) | probes (min) | read (min) | device | driver | determinism warnings | level 3 mirror.json sha256 (host) | repeat | refits bit-identical to level 3 |")
    add("|---|---:|---:|---:|---|---|---:|---|---|---|")
    for name, ds in rec["datasets"].items():
        meta = metas[name]
        place = ds["placement"]
        det = sum(w["count"] for w in ds["warnings"] if w.get("determinism")) + meta.get("probe_determinism_warnings", 0)
        rep = ds.get("repeat")
        rep_s = "n/a" if rep is None else (("unit bit-identical" if rep["bit_identical"] else f"unit differs, max |diff| {rep['max_abs_diff']:.3e}")
                                           + ("; moments byte-identical" if rep["moments_identical"] else "; moments differ"))
        refit = ds.get("refit", {})
        same = sum(v["bit_identical"] for v in refit.values())
        add(f"| {name} | {ds['moments']['seconds'] / 60:.1f} | {meta['probe_seconds'] / 60:.0f} | {ds['seconds'] / 60:.1f} | "
            f"{place.get('device_name', place.get('device'))} | {place.get('driver', 'n/a')} | {det} | `{(ds['mirror_sha256_host'] or 'n/a')[:16]}` | "
            f"{rep_s} | {same} of {len(refit)} |")
    add("")
    devs = sorted({x for ds in rec["datasets"].values() for x in ds["deviations"]} | set(metas.get("_systems_deviations", [])))
    add("Placement: " + f"`{rec['placement']}` = {json.dumps(SPEC)}, applied by level 3's host_placement in every host process, with "
        "CUBLAS_WORKSPACE_CONFIG=:4096:8 set before torch loads. Every host job verified level 3's mirror.json against every level 0 "
        "and level 3 file before reading a byte, and ran from one commit; the LF sha256 of every repository module it imported was "
        "filed and checked against the committed files at the file stage. Deviations: " + ("; ".join(devs) if devs else "none") + ".")
    add("")
    return LF.join(L)


def stage_doc(log=print, out_root: Path | None = None, doc: Path | None = None, datasets=DATASETS, extra_deviations=None) -> dict:
    root = out_root or OUT
    rec = assemble_record(root, datasets)
    rec_path = root / "record.json"
    atomic_json(rec_path, rec)
    metas = {}
    for name in rec["datasets"]:
        m = {"probe_seconds": sum(json.loads(f.read_text(encoding="utf-8"))["seconds"] for f in (root / name / "probes").glob("fitlog_*.json"))}
        pmeta = json.loads((root / name / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
        m["probe_determinism_warnings"] = sum(w["count"] for w in pmeta.get("warnings", []) if w.get("determinism"))
        metas[name] = m
    metas["_systems_deviations"] = list(extra_deviations or [])
    target = doc or DOC
    target.write_text(render_doc(rec, load_declaration(), metas, filed_level3()), encoding="utf-8")
    log(f"wrote {rec_path} and {target}")
    return rec


# ── file (laptop) ────────────────────────────────────────────────────────────


def stage_file(date: str, commit: str, log=print, extra: dict | None = None) -> None:
    """run_record_mp_approx_l5_<date> appended to the declaration after the code and mirror checks; status DECLARED_NOT_RUN -> RUN."""
    text = CONFIG.read_text(encoding="utf-8")
    decl = yaml.safe_load(text)
    key = f"run_record_mp_approx_l5_{date}"
    if key in decl:
        raise SystemExit(f"{key} exists")
    if decl["status"] != "DECLARED_NOT_RUN":
        raise SystemExit(f"status is {decl['status']}, not DECLARED_NOT_RUN")
    if L3.committed_lf_sha(commit, "scripts/mp_approx_l5.py") is None:
        raise SystemExit(f"{commit} does not hold scripts/mp_approx_l5.py")
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    jobs, per = {}, {}
    for name, ds in rec["datasets"].items():
        d = OUT / name
        mmeta = json.loads((d / "moments_meta.json").read_text(encoding="utf-8"))
        pmeta = json.loads((d / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
        read = json.loads((d / "read.json").read_text(encoding="utf-8"))
        if L0.sha256_file(d / "read.json") != ds["read_sha256"]:
            hard_stop(f"{name}: read.json is not the file record.json assembled")
        jobs[f"{name}/moments"], jobs[f"{name}/probe"], jobs[f"{name}/read"] = mmeta["module_sha256"], pmeta["module_sha256"], read["module_sha256"]
        rep = None
        if name == REPEAT[0]:
            rep = json.loads((d / "repeat" / "repeat.json").read_text(encoding="utf-8"))
            jobs[f"{name}/repeat"] = rep["module_sha256"]
        pinned = decl["inputs"]["level3"]["sidecars"][name]["mirror"]
        hosts = {mmeta["mirror_sha256_host"], pmeta["mirror_sha256_host"], read["mirror_sha256_host"]} | ({rep["mirror_sha256_host"]} if rep else set())
        if hosts != {pinned}:
            hard_stop(f"mirror_verification: {name}: level 3's mirror.json on the host is not the pinned file", pinned=pinned, host=sorted(hosts))
        p = ds["r"]["probes"][PRIMARY]
        per[name] = {"reading": ds["reading"], "primary_rho_bar": p["rho_bar"], "readable_metrics": ds["r"]["readable_metrics"],
                     "flags": ds["flags"], "interpretation": ds["interpretation"],
                     "rho_bar_r": {q: v["rho_bar"] for q, v in ds["r"]["probes"].items()},
                     "contrasts": {c: {"point": v["point"], "ci": v["ci"]} for c, v in ds["contrasts"].items()},
                     "shares": {s: {"point": v["point"], "ci": v["ci"], "readable": v["readable"]} for s, v in ds["shares"].items()},
                     "bands_r": {q: v["band"] for q, v in ds["r"]["probes"].items()},
                     "bands_e": {q: v["band"] for q, v in ds["e"]["probes"].items()},
                     "queries": ds["queries"], "uq_rows": ds["uq_rows"], "moments": ds["moments"], "repeat": ds.get("repeat"),
                     "refit": ds["refit"], "level3_mirror_sha256_host": pinned, "moments_meta_sha256": ds["moments_meta_sha256"],
                     "probe_meta_sha256": ds["probe_meta_sha256"], "read_sha256": ds["read_sha256"],
                     "placement": {k: ds["placement"].get(k) for k in ("host", "env", "device_name", "driver", "torch", "cuda")},
                     "deviations": ds["deviations"],
                     "determinism_warnings": {"probe": sum(w["count"] for w in pmeta["warnings"] if w.get("determinism")),
                                              "read": sum(w["count"] for w in read["warnings"] if w.get("determinism"))}}
    bad = L3.code_problems(jobs, lambda path: L3.committed_lf_sha(commit, path))
    if bad:
        hard_stop("identical_code: the host jobs' recorded module sha256 values are not one set equal to the committed files", problems=bad)
    run = {"utc": L0.utc(), "status_moves": "DECLARED_NOT_RUN -> RUN", "terminal": "STOP_FOR_REVIEW", "held_half_read": False,
           "checkpoints_updated": 0, "code_commit": commit,
           "placement": "moments on the host CPU; probe, repeat and read host-native on host_gpu_det; doc and file on the laptop",
           "identical_code": f"{len({p for s in jobs.values() for p in s})} module paths over {len(jobs)} host processes, one sha256 each, "
                             "equal to the committed files",
           "record_sha256": L0.sha256_file(RECORD), "document": str(DOC.relative_to(ROOT)).replace(chr(92), "/"),
           "document_sha256": L0.lf_sha256(DOC), "datasets": per}
    if extra:
        run.update(extra)
    block = yaml.safe_dump(L0.clean({key: run}), sort_keys=False, width=160, allow_unicode=True)
    text = text.replace("status: DECLARED_NOT_RUN", "status: RUN", 1)
    CONFIG.write_text(text.rstrip(LF) + LF + LF + block, encoding="utf-8")
    log(f"filed {key}")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("moments", "probe", "repeat", "read", "run", "doc", "file"))
    ap.add_argument("--dataset", choices=DATASETS, help="every stage but doc and file: one dataset per process")
    ap.add_argument("--date", help="file: the run record's date, e.g. 2026_09_30")
    ap.add_argument("--commit", help="file: the commit every host job ran from (the one that adds this script)")
    ap.add_argument("--extra", type=Path, default=None, help="file: a JSON object of fields added to the run record")
    ap.add_argument("--deviation", action="append", default=[], help="doc: a systems deviation to list (repeatable)")
    args = ap.parse_args()

    def log(s: str) -> None:
        print(f"[{L0.utc()}] {s}", flush=True)

    per_dataset = ("moments", "probe", "repeat", "read", "run")
    if args.stage in per_dataset and args.dataset is None:
        ap.error(f"--stage {args.stage} needs --dataset")
    if args.stage not in per_dataset and args.dataset is not None:
        ap.error(f"--stage {args.stage} reads every dataset")
    if args.stage == "repeat" and args.dataset != REPEAT[0]:
        ap.error(f"the repeat is declared on {REPEAT[0]} only")
    if args.dataset is not None:
        HARD_STOP_DIR[0] = OUT / args.dataset
    point_stops()
    if args.stage == "moments":
        stage_moments(args.dataset, log)
    elif args.stage == "probe":
        stage_probe(args.dataset, log)
    elif args.stage == "repeat":
        stage_probe(args.dataset, log, repeat=True)
    elif args.stage == "read":
        stage_read(args.dataset, log)
    elif args.stage == "run":
        stage_run(args.dataset, log)
    elif args.stage == "doc":
        stage_doc(log, extra_deviations=args.deviation)
    else:
        if not (args.date and args.commit):
            ap.error("--stage file needs --date and --commit")
        extra = json.loads(args.extra.read_text(encoding="utf-8")) if args.extra else None
        stage_file(args.date, args.commit, log, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
