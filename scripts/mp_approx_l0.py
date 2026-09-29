"""MP-Approx level 0, track MP-ORACLE: how much of the trained GNN's effect node-local compiled information predicts
(configs/mp_approx_l0.yaml).

    python scripts/mp_approx_l0.py --stage score --dataset metaqa   # compile + the nine scored functions, integrity-checked -> U_q sidecar
    python scripts/mp_approx_l0.py --stage probe --dataset metaqa   # cross-fitted probes -> out-of-fold predictions
    python scripts/mp_approx_l0.py --stage read                     # quantities, bootstrap, bands -> record.json
    python scripts/mp_approx_l0.py --stage doc
    python scripts/mp_approx_l0.py --stage file --date 2026_09_29 --extra run_extra.json

Measurement only: the GNN's outputs are the probes' targets, and nothing here becomes a retriever, a feature, a teacher
or a selection criterion. Held rows leave before compilation; the six checkpoints are loaded read-only (eval mode,
no_grad) and pinned by sha256. The one piece of copied code is the cell's attention (split_message), which the run
checks against the cell at every step.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":   # placement.threads: BLAS and OpenMP pools are fixed before numpy and torch load
    _THREADS = "4" if "probe" in sys.argv else "6"
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = _THREADS

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

from mp_retrieval.m3b_train import METRIC_NAMES, rank_metrics  # noqa: E402
from mp_retrieval.universal_v2_models import segment_softmax  # noqa: E402

CONFIG = ROOT / "configs" / "mp_approx_l0.yaml"
OUT = ROOT / "outputs" / "mp_approx_l0"
RECORD = OUT / "record.json"
DOC = ROOT / "docs" / "MP_APPROX_L0.md"
LF = chr(10)

DATASETS = ("metaqa", "2wiki", "squad")
SEEDS = (0, 1, 2)
TWIN, GNN = "u_mlp_v2_mix__H128__s{k}", "u_gnn_v2_ef__H128__s{k}"
FUNCS = tuple(f"{fam}{k}" for fam in ("twin", "gnn", "noedge") for k in SEEDS)   # T_k, G_k, G0_k (scoring_pass.functions)
STORED_FUNCS = FUNCS[:6]                                                          # T_k and G_k have stored metrics
KEEP_TOP = 20
CHUNK_NODES = 24000
SCORE_THREADS, PROBE_THREADS = 6, 4
FOLDS = 5
METAQA_PER_HOP = 800
SUBSAMPLE_SALT = "mp_approx_l0|"
STEPS = 3
N_VECTOR = 515
TOP20_METRICS = ("recall@1", "recall@5", "recall@10", "recall@20", "hit@1", "ndcg@5", "ndcg@20", "full_coverage@5", "full_coverage@20")
UQ_EXACT = TOP20_METRICS + ("gold_in_pool",)
HARD_STOP_DIR = [OUT]   # the smoke run points it at its own directory


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lf_sha256(path: Path) -> str:
    """inputs.frozen_code: the sha256 of the bytes with CRLF folded to LF."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_declaration() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def hard_stop(message: str, **evidence) -> None:
    """hard_stops: the evidence goes to hard_stops.json beside the outputs; the status line is left alone."""
    where = HARD_STOP_DIR[0]
    where.mkdir(parents=True, exist_ok=True)
    path = where / "hard_stops.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    rows.append({"utc": utc(), "message": message, **{k: _jsonable(v) for k, v in evidence.items()}})
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    raise SystemExit(f"HARD STOP: {message}")


def verify_pins(decl: dict, datasets=DATASETS) -> None:
    """inputs: every eval array, id list, the contract, the selection, the six weights and records (raw bytes), and the
    frozen code (LF-normalised)."""
    inp = decl["inputs"]
    checks = []
    for name in datasets:
        for part in ("seed0", "seeds_1_2", "query_ids"):
            checks.append((inp["eval_arrays"][name][part]["path"], inp["eval_arrays"][name][part]["sha256"]))
    for key in ("feature_contract", "selection"):
        checks.append((inp[key]["path"], inp[key]["sha256"]))
    for key, pins in inp["checkpoints"].items():
        checks.append((f"{inp['checkpoint_dir']}{key}.pt", pins["weights"]))
        checks.append((f"{inp['checkpoint_dir']}{key}.json", pins["record"]))
    for rel, digest in checks:
        found = sha256_file(ROOT / rel)
        if found != digest:
            hard_stop(f"{rel} is not its pinned sha256", path=rel, pinned=digest, found=found)
    for rel, digest in inp["frozen_code"].items():
        found = lf_sha256(ROOT / rel)
        if found != digest:
            hard_stop(f"frozen code {rel} changed", path=rel, pinned=digest, found=found)


# ── the rules that read ids only ─────────────────────────────────────────────


def hop_from_id(qid: str) -> int:
    """universal_v2_run's rule: metaqa:<k>hop:..."""
    return int(qid.split(":")[1][0])


def metaqa_subsample(ids: list[str], gate: np.ndarray, per_hop: int = METAQA_PER_HOP) -> np.ndarray:
    """population.metaqa: for each hop, the per_hop V2_GATE queries with the smallest sha256(salt + id), returned as
    population rows in population order. Reads the ids and the gate mask only."""
    keep = []
    for h in (1, 2, 3):
        rows = [int(i) for i in np.flatnonzero(gate) if hop_from_id(ids[i]) == h]
        if len(rows) < per_hop:
            raise SystemExit(f"hop {h}: {len(rows)} V2_GATE queries, fewer than {per_hop}")
        rows.sort(key=lambda i: hashlib.sha256((SUBSAMPLE_SALT + ids[i]).encode("utf-8")).hexdigest())
        keep.extend(rows[:per_hop])
    return np.sort(np.asarray(keep, dtype=np.int64))


def fold_of(query_id: str) -> int:
    """probes.folds: D0.2's rule."""
    return int(hashlib.sha256(query_id.encode("utf-8")).hexdigest(), 16) % FOLDS


def is_inner(query_id: str) -> bool:
    """probes.inner_split."""
    return int(hashlib.sha256((query_id + "|inner").encode("utf-8")).hexdigest(), 16) % 10 == 0


# ── the copied attention and the taps ────────────────────────────────────────


def split_message(cell, h, q_state, edge_index, r, self_r):
    """QueryRelationCell.forward's message, copied under a new name (this_file_does_not_touch.frozen). Returns the full
    message m (the tensor the cell passes to its dropout), its neighbour part m_nbr (the sum over the in-edges alone)
    and its self part (the self-loop's alpha_vv (W_mh h_v + W_mr r_self)); m_nbr + m_self = m."""
    N, H, K = h.shape[0], cell.hidden, cell.heads
    E = edge_index.shape[1]
    loops = torch.arange(N, dtype=torch.long, device=h.device)
    u = torch.cat([edge_index[0], loops])
    v = torch.cat([edge_index[1], loops])
    r_all = torch.cat([r, self_r.unsqueeze(0).expand(N, -1)])
    src, dst, qh = cell.src(h), cell.dst(h), cell.query(q_state)
    e = Fn.leaky_relu(src[u] + dst[v] + qh[v] * cell.rel1(r_all) + cell.rel2(r_all), 0.2)
    score = (e.view(-1, K, H // K) * cell.att).sum(-1)
    alpha = segment_softmax(score, v, N)
    msg = (cell.msg_h(h)[u] + cell.msg_r(r_all)).view(-1, K, H // K) * alpha.unsqueeze(-1)
    m = torch.zeros(N, K, H // K, dtype=h.dtype, device=h.device).index_add_(0, v, msg).view(N, H)
    m_nbr = torch.zeros(N, K, H // K, dtype=h.dtype, device=h.device).index_add_(0, v[:E], msg[:E]).view(N, H)
    m_self = msg[E:].reshape(N, H)          # the loops follow the edges, one per node, in node order
    return m, m_nbr, m_self


class Taps:
    """Hooks on one seed pair that record, while on: the twin's vector channels (the columns its input linear reads
    after the gated scalars), its body output and base_z; the GNN's step messages, recomputed by split_message from
    the cell's own inputs, beside the tensor the cell passes to its dropout."""

    def __init__(self, twin, gnn, n_scalar_cols: int):
        self.on = False
        self.n = int(n_scalar_cols)
        self.reset()
        self.handles = [twin.input.linear.register_forward_pre_hook(self._channels),
                        twin.body.register_forward_hook(self._body),
                        twin.input.register_forward_hook(self._base),
                        gnn.cell.register_forward_pre_hook(self._cell),
                        gnn.cell.dropout.register_forward_pre_hook(self._dropout)]

    def reset(self) -> None:
        self.channels = self.body = self.base_z = None
        self.copies, self.nbr, self.selfs, self.actual = [], [], [], []

    def _channels(self, _mod, args):
        if self.on:
            self.channels = args[0][:, self.n:].detach()

    def _body(self, _mod, _args, out):
        if self.on:
            self.body = out.detach()

    def _base(self, _mod, _args, out):
        if self.on:
            self.base_z = out[1].detach()

    def _cell(self, mod, args):
        if self.on:
            m, m_nbr, m_self = split_message(mod, *args)
            self.copies.append(m)
            self.nbr.append(m_nbr)
            self.selfs.append(m_self)

    def _dropout(self, _mod, args):
        if self.on:
            self.actual.append(args[0].detach().clone())

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


def state_digest(model) -> str:
    h = hashlib.sha256()
    for key, t in model.state_dict().items():
        h.update(key.encode("utf-8"))
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def noedge_forward(gnn, batch):
    """G0_k: one forward with message_passing = False, restored at once; the attribute is asserted True and the
    state_dict digest equal afterwards (scoring_pass.functions.G0_k)."""
    before = state_digest(gnn)
    gnn.message_passing = False
    try:
        s = gnn(batch)
    finally:
        gnn.message_passing = True
    if gnn.message_passing is not True or state_digest(gnn) != before:
        hard_stop("message_passing or the state_dict was not restored after a no-edge forward")
    return s


# ── per-query helpers ────────────────────────────────────────────────────────


def pool_z(s: np.ndarray) -> np.ndarray:
    """scoring_pass.z_scores: within the query over the full pool, float64; a constant score gives 0."""
    s = np.asarray(s, dtype=np.float64)
    sd = s.std()
    return np.zeros_like(s) if sd < 1e-12 else (s - s.mean()) / sd


def column_z(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """segment_zscore's rule for one query, in float64: a column constant within the query gives 0."""
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(0)
    z = (x - x.mean(0)) / np.maximum(sd, eps)
    z[:, sd < eps] = 0.0
    return z


def pool_rank(s: np.ndarray) -> np.ndarray:
    """1-based ranks under the frozen tie rule (score descending, then pool position)."""
    order = np.argsort(-np.asarray(s), kind="stable")
    r = np.empty(order.size, dtype=np.int64)
    r[order] = np.arange(1, order.size + 1)
    return r


def analysis_set(scores: list[np.ndarray], gold_local: np.ndarray, top: int = KEEP_TOP) -> np.ndarray:
    """U_q: the in-pool golds plus the top `top` of every function, as sorted local indices (pool order)."""
    keep = np.zeros(scores[0].size, dtype=bool)
    keep[np.asarray(gold_local, dtype=np.int64)] = True
    for s in scores:
        keep[np.argsort(-np.asarray(s), kind="stable")[:top]] = True
    return np.flatnonzero(keep)


def first_support(scalars: np.ndarray, gold_local: np.ndarray, idx: dict, family: str = "STRUCT") -> int:
    """D0.1's rule: the smallest t with has_h{t}_{family} > 0 at some in-pool gold; 0 when none, -1 without a gold."""
    if gold_local.size == 0:
        return -1
    return next((t for t in (1, 2, 3) if float(scalars[gold_local, idx[f"has_h{t}_{family}"]].max()) > 0), 0)


def to_f16(a: np.ndarray, what: str) -> np.ndarray:
    out = np.asarray(a, dtype=np.float32).astype(np.float16)
    if not np.isfinite(out).all():
        raise SystemExit(f"{what}: a value is outside the float16 range; stopping before anything is written")
    return out


# ── stage: score ─────────────────────────────────────────────────────────────


def load_stored(decl: dict, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """The pilot's stored per-query metrics of T_k and G_k over the whole eval population (both halves), with the
    stored halves and hops. Seed 0 from <name>.npz, seeds 1-2 from <name>__more_69caa3fe.npz."""
    arrays = decl["inputs"]["eval_arrays"][name]
    out = {}
    with np.load(ROOT / arrays["seed0"]["path"]) as z:
        half, hop = z["half"].astype(bool), z["hop"].astype(np.int64)
        for fam, tmpl in (("twin", TWIN), ("gnn", GNN)):
            out[f"{fam}0"] = np.stack([z[f"{tmpl.format(k=0)}/{m}"] for m in METRIC_NAMES], 1).astype(np.float64)
    with np.load(ROOT / arrays["seeds_1_2"]["path"]) as z:
        if not (np.array_equal(z["half"].astype(bool), half) and np.array_equal(z["hop"].astype(np.int64), hop)):
            hard_stop(f"{name}: the two stored eval files disagree on the halves or hops")
        for k in (1, 2):
            for fam, tmpl in (("twin", TWIN), ("gnn", GNN)):
                out[f"{fam}{k}"] = np.stack([z[f"{tmpl.format(k=k)}/{m}"] for m in METRIC_NAMES], 1).astype(np.float64)
    return half, hop, out


def load_models(decl: dict, U, inputs: dict, bank, selection: dict) -> dict:
    ckdir = ROOT / decl["inputs"]["checkpoint_dir"]
    models = {}
    for k in SEEDS:
        for fam, tmpl, arm in (("twin", TWIN, "u_mlp_v2_mix"), ("gnn", GNN, "u_gnn_v2_ef")):
            key = tmpl.format(k=k)
            rec = U.read_json(ckdir / f"{key}.json")
            if rec["arm"] != arm:
                hard_stop(f"{key}: fit record arm {rec['arm']} is not {arm}")
            m = U.make_model(rec["arm"], inputs, bank, selected_gnn=selection["gnn"]["arm"])
            m.load_state_dict(torch.load(ckdir / f"{key}.pt", map_location="cpu"))
            m.eval()
            models[f"{fam}{k}"] = m
    return models


def scored_rows(name: str, ids: list[str], half: np.ndarray) -> np.ndarray:
    """population: metaqa's stratified subsample of V2_GATE; every V2_GATE row of 2wiki and squad."""
    return metaqa_subsample(ids, half) if name == "metaqa" else np.flatnonzero(half).astype(np.int64)


def stage_score(decl: dict, name: str, log=print, limit: int | None = None, out_dir: Path | None = None) -> None:
    import universal_v2_run as U   # the frozen runner, imported unchanged

    torch.set_num_threads(SCORE_THREADS)   # the stored pilot eval records' thread count
    out_dir = out_dir or OUT / name
    if (out_dir / "meta.json").exists():
        log(f"{out_dir}/meta.json exists; not rescored")
        return
    verify_pins(decl)
    cfg, cfg_m3b, cfg_h = U.load_configs()
    inputs = U.model_inputs(cfg, cfg_m3b)
    n_sc = int(inputs["n_scalars"])
    selection = U.read_json(ROOT / decl["inputs"]["selection"]["path"])
    m3b_compile = U.M3B_RUN.load_script("m3b_compile")
    m3b_contract = U.M3B_RUN.load_script("m3b_contract")
    split = cfg_m3b["populations"]["eval_splits"][name]
    if "test" in str(split).lower():
        hard_stop(f"{name}: eval split {split} is a test split")
    contexts, handles, pkg, bank = U.open_contexts_v2(cfg_m3b, [name], m3b_compile)
    context, ds = contexts[name], handles[name]
    models = load_models(decl, U, inputs, bank, selection)
    for k in SEEDS:
        twin, gnn = models[f"twin{k}"], models[f"gnn{k}"]
        if twin.input.input_width != 2 * n_sc + N_VECTOR or gnn.steps != STEPS or gnn.message_passing is not True:
            hard_stop(f"seed {k}: not the declared architecture (input width {twin.input.input_width}, steps {gnn.steps})")
    half_stored, hop_stored, stored = load_stored(decl, name)
    m3a = pkg[0]
    _key, frozen = m3b_compile.frozen_contract(cfg_m3b)
    construction = frozen["per_dataset"][name]["construction"]
    declared = cfg["m3b_incumbents"]["eval_populations_reused_here"][name]
    positions = m3a.node_position_map(ds)
    pop = m3b_compile.population(ds, name, "eval", cfg_m3b, cfg_h, m3a, positions)
    del positions
    gc.collect()
    if pop.digest != declared["ids_sha256"] or pop.idx.size != int(declared["queries"]):
        hard_stop(f"{name}: not the M3B eval population", digest=pop.digest, queries=int(pop.idx.size))
    ids_file = json.loads((ROOT / decl["inputs"]["eval_arrays"][name]["query_ids"]["path"]).read_text(encoding="utf-8"))
    if list(pop.ids) != ids_file:
        hard_stop(f"{name}: the population ids are not the stored id list")
    half = U.half_labels(name, ds, split, pop.ids)
    if not np.array_equal(half, half_stored):
        hard_stop(f"{name}: the recomputed halves differ from the stored halves")
    if name == "metaqa" and not np.array_equal(np.asarray([hop_from_id(q) for q in pop.ids]), hop_stored):
        hard_stop("metaqa: the hops read from the ids differ from the stored hops")
    rows = scored_rows(name, pop.ids, half)
    if limit is not None:
        rows = rows[:limit]
    if not half[rows].all():
        hard_stop(f"{name}: a held row would be scored")
    all_ids = list(pop.ids)
    pop.ids, pop.idx, pop.golds = [all_ids[i] for i in rows], pop.idx[rows], [pop.golds[i] for i in rows]
    t_prep = time.time()
    prep = m3b_compile.prepare(ds, [pop], construction, cfg_h, context.stores, m3a, m3b_contract)[0]
    n = len(rows)
    sizes = np.asarray([p.size for p in prep.pools], dtype=np.int64)
    chunk = max(1, int(CHUNK_NODES // max(sizes.mean(), 1)))
    n_chunks = math.ceil(n / chunk)
    log(f"{name}: {n} queries scored ({int(half.sum())} V2_GATE of {half.size}), pools mean {sizes.mean():.0f}, "
        f"chunk {chunk} queries, {n_chunks} chunks, prepared in {time.time() - t_prep:.0f}s")
    columns = inputs["column_indices"]
    taps = {k: Taps(models[f"twin{k}"], models[f"gnn{k}"], 2 * n_sc) for k in SEEDS}
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    t0, done_here = time.time(), 0
    with torch.no_grad():
        for ci in range(n_chunks):
            idx = np.arange(ci * chunk, min((ci + 1) * chunk, n))
            path = chunks_dir / f"c{ci:05d}.npz"
            if path.exists():
                with np.load(path) as z:
                    if not np.array_equal(z["chunk_rows"], rows[idx]):
                        raise SystemExit(f"{path}: not this chunk's rows; delete {chunks_dir} to rescore")
                continue
            qds, gold_locals, xs, firsts = [], [], [], []
            for j in idx:
                E = context.nodes.read(prep.pools[j])
                inp = U.QueryInputs(prep.qemb[j], prep.dense_ids[j], prep.dense_scores[j], prep.splade_ids[j], prep.splade_scores[j])
                compiled = U.compile_query_v2(inp, prep.pools[j], prep.seeds[j], context.stores, context.nodes, context.rel_table, embeddings=E)
                gl = m3b_compile.gold_local_of(prep.pools[j], pop.golds[j])
                x = compiled.scalars[:, columns]
                qds.append({"pool": compiled.pool, "x": x, "seedw": compiled.seedw, "qemb": prep.qemb[j],
                            "seeds": compiled.seeds_local, "gold": gl, "gold_total": int(pop.golds[j].size), "emb": E})
                gold_locals.append(gl)
                xs.append(np.asarray(x, dtype=np.float32))
                firsts.append(first_support(compiled.scalars, gl, U.IDX))
                del compiled
            batch = U.pack_queries_v2(qds, context)
            del qds
            ptr = batch.qptr.numpy()
            scores = {}
            for k in SEEDS:
                twin, gnn, tap = models[f"twin{k}"], models[f"gnn{k}"], taps[k]
                tap.reset()
                tap.on = True
                scores[f"twin{k}"] = twin(U.arm_view(twin, batch, inputs)).numpy().astype(np.float64)
                scores[f"gnn{k}"] = gnn(U.arm_view(gnn, batch, inputs)).numpy().astype(np.float64)
                tap.on = False
                if len(tap.copies) != STEPS or len(tap.actual) != STEPS:
                    hard_stop(f"seed {k}, chunk {ci}: {len(tap.copies)} copied and {len(tap.actual)} cell messages, not {STEPS}")
                for t in range(STEPS):
                    try:
                        torch.testing.assert_close(tap.copies[t], tap.actual[t], atol=1e-5, rtol=1e-5)
                    except AssertionError as err:
                        hard_stop(f"message_split: seed {k}, chunk {ci}, step {t + 1}: the copied attention is not the cell's message",
                                  detail=str(err)[:2000])
                scores[f"noedge{k}"] = noedge_forward(gnn, U.arm_view(gnn, batch, inputs)).numpy().astype(np.float64)
            cap = {k: {"channels": taps[k].channels.numpy(), "body": taps[k].body.numpy(), "base_z": taps[k].base_z.numpy(),
                       "nbr": [m.numpy() for m in taps[k].nbr]} for k in SEEDS}
            per_row = {key: [] for key in ("query", "local", "is_gold", "x", "zx", "z", "rank")}
            for k in SEEDS:
                per_row.update({f"channels_{k}": [], f"body_{k}": [], f"base_z_{k}": [], f"mnbr_{k}": []})
            per_q = {key: [] for key in ("q_row", "q_hop", "q_gold_total", "q_pool_size", "q_uq_size", "q_fold", "q_first_support_STRUCT", "q_metrics")}
            for jj, j in enumerate(idx):
                a, b = int(ptr[jj]), int(ptr[jj + 1])
                if b - a != sizes[j]:
                    hard_stop(f"query {pop.ids[j]}: packed rows {b - a} != pool size {sizes[j]}")
                gl, gt, row = gold_locals[jj], int(pop.golds[j].size), int(rows[j])
                full = {f: rank_metrics(scores[f][a:b], gl, gt) for f in FUNCS}
                for f in STORED_FUNCS:
                    want = stored[f][row]
                    for mi, m in enumerate(METRIC_NAMES):
                        if full[f][m] != want[mi]:
                            hard_stop(f"integrity.stored: query {pop.ids[j]} (row {row}), {f}, {m}: forward {full[f][m]} != stored {want[mi]}",
                                      query=pop.ids[j], row=row, function=f, metric=m, forward=full[f][m], stored=float(want[mi]))
                loc = analysis_set([scores[f][a:b] for f in FUNCS], gl)
                gl_u = np.searchsorted(loc, gl)
                for f in FUNCS:
                    r_u = rank_metrics(scores[f][a:b][loc], gl_u, gt)
                    for m in UQ_EXACT:
                        if r_u[m] != full[f][m]:
                            hard_stop(f"integrity.analysis_set: query {pop.ids[j]} (row {row}), {f}, {m}: U_q {r_u[m]} != full pool {full[f][m]}",
                                      query=pop.ids[j], row=row, function=f, metric=m, uq=r_u[m], full=full[f][m])
                per_row["query"].append(np.full(loc.size, j, dtype=np.int32))
                per_row["local"].append(loc.astype(np.int32))
                per_row["is_gold"].append(np.isin(loc, gl))
                per_row["x"].append(xs[jj][loc])
                per_row["zx"].append(column_z(xs[jj])[loc].astype(np.float32))
                per_row["z"].append(np.stack([pool_z(scores[f][a:b])[loc] for f in FUNCS], 1))
                per_row["rank"].append(np.stack([pool_rank(scores[f][a:b])[loc] for f in FUNCS], 1).astype(np.int32))
                for k in SEEDS:
                    c = cap[k]
                    per_row[f"channels_{k}"].append(to_f16(c["channels"][a:b][loc], f"channels_{k}"))
                    per_row[f"body_{k}"].append(to_f16(c["body"][a:b][loc], f"body_{k}"))
                    per_row[f"base_z_{k}"].append(np.asarray(c["base_z"][a:b][loc], dtype=np.float32))
                    per_row[f"mnbr_{k}"].append(to_f16(np.stack([c["nbr"][t][a:b][loc] for t in range(STEPS)], 1), f"mnbr_{k}"))
                per_q["q_row"].append(row)
                per_q["q_hop"].append(int(hop_stored[row]))
                per_q["q_gold_total"].append(gt)
                per_q["q_pool_size"].append(b - a)
                per_q["q_uq_size"].append(int(loc.size))
                per_q["q_fold"].append(fold_of(pop.ids[j]))
                per_q["q_first_support_STRUCT"].append(firsts[jj])
                per_q["q_metrics"].append([[full[f][m] for m in METRIC_NAMES] for f in FUNCS])
            arrays = {key: np.concatenate(v) for key, v in per_row.items()}
            arrays.update({key: np.asarray(v, dtype=np.float64 if key == "q_metrics" else np.int64) for key, v in per_q.items()})
            arrays["chunk_rows"] = rows[idx]
            tmp = chunks_dir / f"c{ci:05d}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, path)
            del batch, scores, cap, arrays, per_row, per_q, xs
            for tap in taps.values():
                tap.reset()
            gc.collect()
            done_here += idx.size
            if ci % max(1, n_chunks // 25) == 0 or ci == n_chunks - 1:
                rate = (time.time() - t0) / done_here
                left = n - int(idx[-1]) - 1
                log(f"   {name}: chunk {ci + 1}/{n_chunks}, {int(idx[-1]) + 1}/{n} queries, {rate * 1000:.0f} ms/query, "
                    f"about {left * rate / 60:.0f} min left; integrity equal so far")
    for tap in taps.values():
        tap.remove()
    verify_pins(decl)   # again at the end
    meta = assemble(chunks_dir, out_dir, n_chunks)
    (out_dir / "qids.json").write_text(json.dumps(list(pop.ids)), encoding="utf-8")
    meta.update({"dataset": name, "utc": utc(), "git_head": git_head(), "declaration_lf_sha256": lf_sha256(CONFIG),
                 "queries": n, "population_rows_scored": "q_row", "limit": limit, "chunk_queries": chunk, "chunks": n_chunks,
                 "threads": torch.get_num_threads(), "seconds_this_process": round(time.time() - t0, 1),
                 "functions": list(FUNCS), "metric_names": list(METRIC_NAMES), "columns": list(inputs["columns"]),
                 "keep_top": KEEP_TOP, "steps": STEPS, "vector_channels": N_VECTOR, "mismatches": 0,
                 "integrity": ("rank_metrics of T_k and G_k on the full pool equal the stored per-query values of all 14 metrics on every "
                               "scored query; the copied attention equals the cell's message at every step of every chunk (atol 1e-5, "
                               "rtol 1e-5); within U_q the nine top-20 metrics and gold_in_pool of all nine functions equal their full-pool "
                               "values; message_passing and every state_dict were restored after each no-edge forward"),
                 "qids_sha256": sha256_file(out_dir / "qids.json")})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    shutil.rmtree(chunks_dir)
    log(f"{name}: scored {n} queries, {meta['uq_rows']} U_q rows, 0 mismatches")


def assemble(chunks_dir: Path, out_dir: Path, n_chunks: int) -> dict:
    """The chunks, concatenated one array at a time into <key>.npy, each with its sha256."""
    files = [chunks_dir / f"c{ci:05d}.npz" for ci in range(n_chunks)]
    with np.load(files[0]) as z:
        keys = [k for k in z.files if k != "chunk_rows"]
    shas, shapes = {}, {}
    for key in keys:
        parts = []
        for f in files:
            with np.load(f) as z:
                parts.append(z[key])
        arr = np.concatenate(parts)
        del parts
        tmp, path = out_dir / f"{key}.tmp.npy", out_dir / f"{key}.npy"
        np.save(tmp, arr)
        os.replace(tmp, path)
        shas[f"{key}.npy"] = sha256_file(path)
        shapes[key] = [int(s) for s in arr.shape] + [str(arr.dtype)]
        del arr
    return {"arrays_sha256": shas, "arrays_shape": shapes, "uq_rows": shapes["query"][0]}


# ── stage: probe ─────────────────────────────────────────────────────────────

LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
MLP_HIDDEN, MLP_LR, MLP_WD, MLP_BATCH_Q, MLP_EPOCHS, MLP_PATIENCE = 128, 1e-3, 1e-4, 64, 30, 3
CLIP, SD_FLOOR = 8.0, 1e-6
ROW_BLOCK = 16384
R_PROBES = ("B0-ridge", "B0-mlp", "B1-ridge", "B1-mlp", "B2-ridge")
E_PROBES = ("B0-ridge", "B1-ridge")
M_PROBES = ("B0-ridge", "B1-ridge")
MSG_WIDTH = 128


class Sidecar:
    """One dataset's U_q sidecar: arrays memory-mapped and checked against meta.json, rows grouped by query."""

    def __init__(self, d: Path, check: bool = True):
        self.dir = Path(d)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        if check:
            for f, digest in self.meta["arrays_sha256"].items():
                if sha256_file(self.dir / f) != digest:
                    raise SystemExit(f"{self.dir / f}: not the sha256 its meta.json records")
        self.qids = json.loads((self.dir / "qids.json").read_text(encoding="utf-8"))
        self.query = np.load(self.dir / "query.npy")
        n_q = len(self.qids)
        if self.query.size == 0 or np.any(np.diff(self.query) < 0):
            raise SystemExit(f"{self.dir}: rows are not grouped by query")
        self.sizes = np.bincount(self.query, minlength=n_q).astype(np.int64)
        if self.sizes.size != n_q or (self.sizes == 0).any():
            raise SystemExit(f"{self.dir}: a query has no U_q row")
        self.ptr = np.concatenate([[0], np.cumsum(self.sizes)]).astype(np.int64)
        self.n_q, self.n_rows = n_q, int(self.ptr[-1])
        self.fold = np.load(self.dir / "q_fold.npy")
        if not np.array_equal(self.fold, [fold_of(q) for q in self.qids]):
            raise SystemExit(f"{self.dir}: q_fold is not D0.2's fold rule")
        self.inner = np.asarray([is_inner(q) for q in self.qids], dtype=bool)

    def arr(self, key: str):
        return np.load(self.dir / f"{key}.npy", mmap_mode="r")

    def rows_of(self, qmask: np.ndarray) -> np.ndarray:
        return np.repeat(qmask, self.sizes)

    def rows_of_queries(self, qs: np.ndarray) -> np.ndarray:
        sizes = self.sizes[qs]
        return np.repeat(self.ptr[qs] - np.concatenate([[0], np.cumsum(sizes)[:-1]]), sizes) + np.arange(int(sizes.sum()))

    def chunks(self, qmask: np.ndarray, rows_per: int = ROW_BLOCK):
        """(queries, rows) of the selected queries, whole queries at a time, about rows_per rows per chunk."""
        qidx = np.flatnonzero(qmask)
        if qidx.size == 0:
            return
        cum = np.cumsum(self.sizes[qidx])
        start = 0
        while start < qidx.size:
            base = int(cum[start - 1]) if start else 0
            stop = max(int(np.searchsorted(cum, base + rows_per, side="right")), start + 1)
            qs = qidx[start:stop]
            yield qs, self.rows_of_queries(qs)
            start = stop


def centre_rows(a: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    """targets.centring: every row minus its query's mean over U_q (rows grouped by query as ptr gives)."""
    a = np.asarray(a)
    sizes = np.diff(ptr)
    means = np.add.reduceat(a, ptr[:-1], axis=0) / sizes.reshape((-1,) + (1,) * (a.ndim - 1))
    return a - np.repeat(means, sizes, axis=0)


def base_raw(sc: Sidecar, base: str, k: int, sel) -> np.ndarray:
    """bases: B0 = sign(x) log1p(|x|) with the within-query z-scores (258); B1 = B0 with T_k's 515 vector channels (773);
    B2 = T_k's body output with base_z (129). float32; sel is a slice or sorted row indices."""
    if base in ("B0", "B1"):
        x = np.asarray(sc.arr("x")[sel], dtype=np.float32)
        parts = [np.sign(x) * np.log1p(np.abs(x)), np.asarray(sc.arr("zx")[sel], dtype=np.float32)]
        if base == "B1":
            parts.append(np.asarray(sc.arr(f"channels_{k}")[sel], dtype=np.float32))
        return np.concatenate(parts, axis=1)
    if base == "B2":
        return np.concatenate([np.asarray(sc.arr(f"body_{k}")[sel], dtype=np.float32),
                               np.asarray(sc.arr(f"base_z_{k}")[sel], dtype=np.float32)[:, None]], axis=1)
    raise ValueError(base)


def standardised(sc: Sidecar, base: str, k: int, train_q: np.ndarray) -> tuple[np.ndarray, dict]:
    """bases.standardisation: mean and sd from the training rows only (float64, two passes); a column with sd < 1e-6
    becomes 0; values clipped to [-8, 8]. Returns every row, float32."""
    s, n = None, 0
    for _qs, sel in sc.chunks(train_q):
        B = base_raw(sc, base, k, sel)
        s = B.sum(0, dtype=np.float64) if s is None else s + B.sum(0, dtype=np.float64)
        n += B.shape[0]
    mu = s / n
    ss = np.zeros_like(mu)
    for _qs, sel in sc.chunks(train_q):
        D = base_raw(sc, base, k, sel).astype(np.float64) - mu
        ss += (D * D).sum(0)
    sd = np.sqrt(ss / n)
    live = sd >= SD_FLOOR
    scale = np.where(live, sd, 1.0)
    out = np.empty((sc.n_rows, mu.size), dtype=np.float32)
    for r0 in range(0, sc.n_rows, ROW_BLOCK):
        r1 = min(r0 + ROW_BLOCK, sc.n_rows)
        Z = (base_raw(sc, base, k, slice(r0, r1)).astype(np.float64) - mu) / scale
        Z[:, ~live] = 0.0
        out[r0:r1] = np.clip(Z, -CLIP, CLIP)
    return out, {"columns": int(mu.size), "dead_columns": int((~live).sum()), "train_rows": int(n)}


def centre_inplace(X: np.ndarray, sc: Sidecar) -> None:
    for qs, sel in sc.chunks(np.ones(sc.n_q, dtype=bool)):
        r0, r1 = int(sel[0]), int(sel[-1]) + 1
        X[r0:r1] = centre_rows(X[r0:r1], sc.ptr[qs[0]:qs[-1] + 2] - r0)


def gram(X: np.ndarray, sc: Sidecar, qmask: np.ndarray, targets):
    """Over the selected queries: X'X, X'Y, the column sums of Y*Y and the row count, float64, whole queries at a time."""
    d = X.shape[1]
    G, C, yy, n = np.zeros((d, d)), None, None, 0
    for _qs, sel in sc.chunks(qmask):
        Xb = X[sel].astype(np.float64)
        Yb = targets(sel)
        G += Xb.T @ Xb
        C = Xb.T @ Yb if C is None else C + Xb.T @ Yb
        yy = (Yb * Yb).sum(0) if yy is None else yy + (Yb * Yb).sum(0)
        n += sel.size
    return G, C, yy, n


def ridge_solve(G: np.ndarray, C: np.ndarray, n: int, lambdas) -> dict:
    """beta(lambda) = (G + lambda n I)^-1 C for every lambda, through one eigendecomposition."""
    w, V = np.linalg.eigh(G)
    w = np.clip(w, 0.0, None)
    A = V.T @ C
    return {lam: V @ (A / (w + lam * n)[:, None]) for lam in lambdas}


def fit_ridge(Xc: np.ndarray, sc: Sidecar, fold: int, targets, groups: list[tuple[str, slice]]):
    """probes.ridge for one fold: lambda per output group by inner-validation within-query MSE (penalty lambda * n_rows),
    then the refit on every training row. Xc is standardised and within-query centred; targets(rows) gives centred Y."""
    train_q = sc.fold != fold
    val_q, fit_q = train_q & sc.inner, train_q & ~sc.inner
    if not val_q.any() or not fit_q.any():
        raise SystemExit(f"fold {fold}: an empty inner split")
    G_fit, C_fit, _yy, n_fit = gram(Xc, sc, fit_q, targets)
    G_val, C_val, yy_val, n_val = gram(Xc, sc, val_q, targets)
    mse = {}
    for lam, b in ridge_solve(G_fit, C_fit, n_fit, LAMBDAS).items():
        mse[lam] = ((b * (G_val @ b)).sum(0) - 2.0 * (b * C_val).sum(0) + yy_val) / n_val
    chosen = {g: min(LAMBDAS, key=lambda lam: (float(mse[lam][cols].sum()), lam)) for g, cols in groups}
    G_all, C_all, n_all = G_fit + G_val, C_fit + C_val, n_fit + n_val
    betas = ridge_solve(G_all, C_all, n_all, sorted(set(chosen.values())))
    beta = np.zeros_like(C_all)
    for g, cols in groups:
        beta[:, cols] = betas[chosen[g]][:, cols]
    log = {"fit_rows": n_fit, "inner_val_rows": n_val, "lambda": chosen,
           "inner_mse": {g: [float(mse[lam][cols].sum()) for lam in LAMBDAS] for g, cols in groups}}
    return beta, log


class ProbeMLP(torch.nn.Module):
    """probes.mlp: Linear(d, 128), GELU, Linear(128, 128), GELU, Linear(128, 1)."""

    def __init__(self, d: int):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, MLP_HIDDEN), torch.nn.GELU(), torch.nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
                                       torch.nn.GELU(), torch.nn.Linear(MLP_HIDDEN, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(Xs: np.ndarray, y_c: np.ndarray, sc: Sidecar, fit_q: np.ndarray, val_q: np.ndarray, test_q: np.ndarray, seed: int):
    """probes.mlp for one fold and one target: AdamW, minibatches of 64 queries (all their U_q rows), the mean squared
    difference between the within-query-centred prediction and target, at most 30 epochs, stop after 3 epochs without
    an inner-validation improvement and restore the best epoch. Returns the centred predictions of test_q's rows."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(Xs)
    y = torch.from_numpy(np.asarray(y_c, dtype=np.float32))
    net = ProbeMLP(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    fit_idx, val_idx, test_idx = np.flatnonzero(fit_q), np.flatnonzero(val_q), np.flatnonzero(test_q)
    if fit_idx.size == 0 or val_idx.size == 0:
        raise SystemExit("an empty inner split")

    def batch(qs):
        rows = torch.from_numpy(sc.rows_of_queries(qs))
        seg = torch.from_numpy(np.repeat(np.arange(qs.size), sc.sizes[qs]))
        return rows, seg, torch.from_numpy(sc.sizes[qs].astype(np.float32))

    def centred(rows, seg, sizes):
        p = net(X[rows])
        return p - (torch.zeros(sizes.numel(), dtype=p.dtype).index_add_(0, seg, p) / sizes)[seg]

    def evaluate(qidx, want_pred=False):
        tot, n, preds = 0.0, 0, []
        with torch.no_grad():
            for b in range(0, qidx.size, 1024):
                rows, seg, sizes = batch(qidx[b:b + 1024])
                pc = centred(rows, seg, sizes)
                tot += float(((pc - y[rows]) ** 2).sum())
                n += rows.numel()
                if want_pred:
                    preds.append(pc.numpy().astype(np.float64))
        return tot / n, (np.concatenate(preds) if want_pred else None)

    best, best_state, best_epoch, bad, curve = math.inf, None, 0, 0, []
    for epoch in range(1, MLP_EPOCHS + 1):
        order = rng.permutation(fit_idx)
        for b in range(0, order.size, MLP_BATCH_Q):
            rows, seg, sizes = batch(order[b:b + MLP_BATCH_Q])
            loss = ((centred(rows, seg, sizes) - y[rows]) ** 2).mean()
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
            if bad >= MLP_PATIENCE:
                break
    if best_state is None:
        raise SystemExit(f"seed {seed}: no finite inner-validation loss")
    net.load_state_dict(best_state)
    _, pred = evaluate(test_idx, want_pred=True)
    return pred, {"seed": seed, "epochs_run": len(curve), "best_epoch": best_epoch, "inner_val_mse": [float(v) for v in curve]}


def message_stats(pred: np.ndarray, target: np.ndarray, local_ptr: np.ndarray) -> np.ndarray:
    """quantities.message, per query and step: [sum ||mhat_c - m_c||^2, sum ||m_c||^2, the sum of cos(mhat_c, m_c) over
    rows with ||m_c|| > 0 (0 where ||mhat_c|| = 0), that row count]. pred, target: (rows, STEPS, 128), centred."""
    err = ((pred - target) ** 2).sum(-1)
    ss = (target * target).sum(-1)
    tn, pn = np.sqrt(ss), np.sqrt((pred * pred).sum(-1))
    live = tn > 0
    ok = live & (pn > 0)
    cos = np.where(ok, (pred * target).sum(-1) / np.where(ok, tn * pn, 1.0), 0.0)
    starts = local_ptr[:-1]
    return np.stack([np.add.reduceat(err, starts, axis=0), np.add.reduceat(ss, starts, axis=0),
                     np.add.reduceat(cos, starts, axis=0), np.add.reduceat(live.astype(np.float64), starts, axis=0)], axis=-1)


def probe_targets(sc: Sidecar) -> tuple[np.ndarray, np.ndarray]:
    """targets: r_k = z(G_k) - z(T_k) and e_k = z(G_k) - z(G0_k), within-query centred, (rows, 3) each."""
    z = np.load(sc.dir / "z.npy")
    r = np.stack([z[:, 3 + k] - z[:, k] for k in SEEDS], 1)
    e = np.stack([z[:, 3 + k] - z[:, 6 + k] for k in SEEDS], 1)
    return centre_rows(r, sc.ptr), centre_rows(e, sc.ptr)


def message_targets(sc: Sidecar, ks):
    """targets.m_k_t of the seeds in ks, within-query centred, flattened to (rows, len(ks) * STEPS * 128); rows must be
    whole queries, sorted."""
    arrays = {k: sc.arr(f"mnbr_{k}") for k in ks}

    def fn(sel: np.ndarray) -> np.ndarray:
        qs = np.unique(sc.query[sel])
        if not np.array_equal(sc.rows_of_queries(qs), sel):
            raise ValueError("message targets are read whole queries at a time")
        local = np.concatenate([[0], np.cumsum(sc.sizes[qs])])
        return np.concatenate([centre_rows(np.asarray(arrays[k][sel], dtype=np.float64), local).reshape(sel.size, -1) for k in ks], axis=1)
    return fn


def stage_probe(name: str, log=print, out_dir: Path | None = None) -> None:
    """probes: the grid, cross-fitted over the five folds. One unit per (base, seed) -- B0 once for the three seeds --
    written atomically with its fit log and skipped on a restart."""
    torch.set_num_threads(PROBE_THREADS)
    d = out_dir or OUT / name
    sc = Sidecar(d)
    pdir = d / "probes"
    pdir.mkdir(parents=True, exist_ok=True)
    r_c, e_c = probe_targets(sc)
    log(f"{name}: {sc.n_q} queries, {sc.n_rows} U_q rows, fold sizes {np.bincount(sc.fold, minlength=FOLDS).tolist()}, "
        f"inner {int(sc.inner.sum())}, threads {torch.get_num_threads()}")
    t_all = time.time()
    for base, k in [("B0", None)] + [(b, kk) for b in ("B1", "B2") for kk in SEEDS]:
        tag = base if k is None else f"{base}_k{k}"
        unit = pdir / f"unit_{tag}.npz"
        if unit.exists():
            log(f"   {tag}: exists")
            continue
        t0 = time.time()
        ks = SEEDS if k is None else (k,)
        rich = base in ("B0", "B1")          # B0 and B1 carry the MLP on r and the ridge on e and m; B2 only the ridge on r
        n_s = len(ks) * (2 if rich else 1)   # the scalar target columns: r (and e), one per seed
        oof = {}
        for kk in ks:
            oof[f"r/{base}-ridge/{kk}"] = np.full(sc.n_rows, np.nan)
            if rich:
                oof[f"r/{base}-mlp/{kk}"] = np.full(sc.n_rows, np.nan)
                oof[f"e/{base}-ridge/{kk}"] = np.full(sc.n_rows, np.nan)
                oof[f"m/{base}-ridge/{kk}"] = np.full((sc.n_q, STEPS, 4), np.nan)
        scalar = np.concatenate([r_c[:, list(ks)]] + ([e_c[:, list(ks)]] if rich else []), axis=1)
        m_fn = message_targets(sc, ks) if rich else None

        def targets(sel):
            return scalar[sel] if m_fn is None else np.concatenate([scalar[sel], m_fn(sel)], axis=1)
        groups = [(f"r/{kk}", slice(i, i + 1)) for i, kk in enumerate(ks)]
        if rich:
            groups += [(f"e/{kk}", slice(len(ks) + i, len(ks) + i + 1)) for i, kk in enumerate(ks)]
            for i, kk in enumerate(ks):
                for t in range(STEPS):
                    c0 = n_s + (i * STEPS + t) * MSG_WIDTH
                    groups.append((f"m/{kk}/{t + 1}", slice(c0, c0 + MSG_WIDTH)))
        flog = []
        for fold in range(FOLDS):
            train_q = sc.fold != fold
            test_q = ~train_q
            Xs, st = standardised(sc, base, 0 if k is None else k, train_q)
            entry = {"fold": fold, "standardisation": st}
            if rich:
                entry["mlp"] = {}
                test_rows = sc.rows_of(test_q)
                for kk in ks:
                    pred, mlog = fit_mlp(Xs, r_c[:, kk], sc, train_q & ~sc.inner, train_q & sc.inner, test_q, 1000 + 10 * kk + fold)
                    oof[f"r/{base}-mlp/{kk}"][test_rows] = pred
                    entry["mlp"][str(kk)] = mlog
            centre_inplace(Xs, sc)
            beta, entry["ridge"] = fit_ridge(Xs, sc, fold, targets, groups)
            for qs, sel in sc.chunks(test_q, 8192):
                P = Xs[sel].astype(np.float64) @ beta
                for i, kk in enumerate(ks):
                    oof[f"r/{base}-ridge/{kk}"][sel] = P[:, i]
                    if rich:
                        oof[f"e/{base}-ridge/{kk}"][sel] = P[:, len(ks) + i]
                if rich:
                    Y = m_fn(sel)
                    local = np.concatenate([[0], np.cumsum(sc.sizes[qs])])
                    w = STEPS * MSG_WIDTH
                    for i, kk in enumerate(ks):
                        mp = P[:, n_s + i * w:n_s + (i + 1) * w].reshape(-1, STEPS, MSG_WIDTH)
                        mt = Y[:, i * w:(i + 1) * w].reshape(-1, STEPS, MSG_WIDTH)
                        oof[f"m/{base}-ridge/{kk}"][qs] = message_stats(mp, mt, local)
            del Xs
            gc.collect()
            flog.append(entry)
            log(f"   {tag}: fold {fold} done, {time.time() - t0:.0f}s")
        for key, v in oof.items():
            if np.isnan(v).any():
                raise SystemExit(f"{tag}: {key} has rows that no fold predicted")
        tmp = pdir / f"unit_{tag}.tmp.npz"
        np.savez(tmp, **{key.replace("/", "|"): v for key, v in oof.items()})
        os.replace(tmp, unit)
        (pdir / f"fitlog_{tag}.json").write_text(json.dumps({"seconds": round(time.time() - t0, 1), "folds": flog}, indent=1),
                                                  encoding="utf-8")
        log(f"   {tag}: written, {time.time() - t0:.0f}s")
    files = sorted(p.name for p in pdir.glob("unit_*.npz")) + sorted(p.name for p in pdir.glob("fitlog_*.json"))
    meta = {"dataset": name, "utc": utc(), "git_head": git_head(), "threads": torch.get_num_threads(),
            "seconds_this_process": round(time.time() - t_all, 1), "sidecar_meta_sha256": sha256_file(d / "meta.json"),
            "files_sha256": {f: sha256_file(pdir / f) for f in files}, "lambdas": list(LAMBDAS),
            "mlp": {"hidden": MLP_HIDDEN, "lr": MLP_LR, "weight_decay": MLP_WD, "batch_queries": MLP_BATCH_Q, "max_epochs": MLP_EPOCHS,
                    "patience": MLP_PATIENCE, "seed": "1000 + 10 k + fold"}}
    (pdir / "probe_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"{name}: probes done in {time.time() - t_all:.0f}s")


def load_probes(d: Path) -> dict:
    pdir = Path(d) / "probes"
    meta = json.loads((pdir / "probe_meta.json").read_text(encoding="utf-8"))
    for f, digest in meta["files_sha256"].items():
        if sha256_file(pdir / f) != digest:
            raise SystemExit(f"{pdir / f}: not the sha256 its probe_meta.json records")
    out = {}
    for p in sorted(pdir.glob("unit_*.npz")):
        with np.load(p) as z:
            for key in z.files:
                out[key.replace("|", "/")] = z[key]
    return out


# ── stage: read ──────────────────────────────────────────────────────────────

RETRIEVAL = ("recall@5", "full_coverage@5", "hit@1")
RESAMPLES, BOOT_SEED = 1000, 0
PRIMARY = "B0-mlp"
BASE_COL = {"r": 0, "e": 6}   # the reference each target is added to: z(T_k) for r, z(G0_k) for e
R_REFS = ("ref:twin", "ref:other_seed")
E_REFS = ("ref:no_edge", "ref:other_seed")


def boot_weights(n_q: int) -> np.ndarray:
    """statistics.bootstrap: queries resampled with replacement, 1000 resamples, default_rng(0); one matrix per dataset,
    as multiplicities (resamples x queries)."""
    draws = np.random.default_rng(BOOT_SEED).integers(0, n_q, size=(RESAMPLES, n_q))
    W = np.empty((RESAMPLES, n_q), dtype=np.float64)
    for b in range(RESAMPLES):
        W[b] = np.bincount(draws[b], minlength=n_q)
    return W


def ci(samples: np.ndarray) -> list[float]:
    """The 95% percentile interval over the resamples that define the quantity (NaN where none does)."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(np.asarray(samples, dtype=np.float64), [2.5, 97.5])
    return [float(lo), float(hi)]


def ratio(num: np.ndarray, den: np.ndarray, W: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """sum(num) / sum(den) over the queries (a stratum when mask is given), and the same per resample."""
    if mask is not None:
        num, den, W = num[mask], den[mask], W[:, mask]
    with np.errstate(all="ignore"):
        return float(num.sum() / den.sum()) if den.sum() != 0 else float("nan"), (W @ num) / (W @ den)


def seed_mean_ratio(num: np.ndarray, den: np.ndarray, W: np.ndarray, mask=None) -> tuple[float, np.ndarray]:
    """A ratio per seed (columns), averaged over the seeds inside each resample."""
    pts, boots = zip(*(ratio(num[:, k], den[:, k], W, mask) for k in range(num.shape[1])))
    return float(np.mean(pts)), np.mean(np.stack(boots), 0)


def band(point: float, interval: list[float], readable: bool) -> str:
    """readings.bands."""
    if not readable or not np.isfinite(point):
        return "NOT_READ"
    if point >= 0.75 and interval[0] >= 0.50:
        return "L0_HIGH"
    if point <= 0.25 and interval[1] <= 0.50:
        return "L0_LOW"
    return "L0_MID"


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman over one query's rows; None with fewer than 3 rows or a constant side."""
    from scipy.stats import rankdata
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else None


_TRIU: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def pair_counts(t: np.ndarray, g: np.ndarray, approx: list[np.ndarray], gold: np.ndarray) -> np.ndarray:
    """quantities.fit.DPR / gold_DPR for one query: the pairs T and G order with opposite strict signs, and those the
    approximation orders as G does (a tie is not recovered); columns [disc, recovered, disc one-gold, recovered one-gold],
    one row per approximation."""
    n = t.size
    if n not in _TRIU:
        _TRIU[n] = np.triu_indices(n, 1)
    i, j = _TRIU[n]
    dg = np.sign(g[i] - g[j])
    disc = np.sign(t[i] - t[j]) * dg < 0
    one = gold[i] != gold[j]
    out = np.empty((len(approx), 4))
    for a_i, a in enumerate(approx):
        rec = disc & (np.sign(a[i] - a[j]) == dg)
        out[a_i] = (disc.sum(), rec.sum(), (disc & one).sum(), (rec & one).sum())
    return out


def top_overlap(a: np.ndarray, g: np.ndarray, k: int = 5) -> float:
    """|topk(a) & topk(g)| / k within U_q, under the frozen tie rule (k = min(5, rows))."""
    k = min(k, a.size)
    return len(set(np.argsort(-a, kind="stable")[:k].tolist()) & set(np.argsort(-g, kind="stable")[:k].tolist())) / k


def per_query_measures(sc: Sidecar, z: np.ndarray, is_gold: np.ndarray, gold_total: np.ndarray, preds: dict, target: np.ndarray,
                       fam: str) -> dict:
    """For one target family: per probe and seed, per query, the three retrieval metrics of shat within U_q, the
    squared error and target energy (R2), Spearman, the discordant-pair counts and the top-5 overlap with G_k."""
    names = list(preds)
    n_q, S = sc.n_q, len(SEEDS)
    res = {p: {"M": np.zeros((n_q, S, len(RETRIEVAL))), "sse": np.zeros((n_q, S)), "ss": np.zeros((n_q, S)),
               "sp": np.zeros((n_q, S)), "sp_ok": np.zeros((n_q, S)), "pairs": np.zeros((n_q, S, 4)), "top5": np.zeros((n_q, S))}
           for p in names}
    base_top5 = np.zeros((n_q, S))
    for q in range(n_q):
        a, b = int(sc.ptr[q]), int(sc.ptr[q + 1])
        gold = is_gold[a:b]
        gl = np.flatnonzero(gold)
        for k in SEEDS:
            base = z[a:b, BASE_COL[fam] + k]
            g = z[a:b, 3 + k]
            y = target[a:b, k]
            base_top5[q, k] = top_overlap(base, g)
            approx = []
            for p in names:
                pr = preds[p][a:b, k]
                pr = pr - pr.mean()
                s_hat = base + pr
                approx.append(s_hat)
                m = rank_metrics(s_hat, gl, int(gold_total[q]))
                r = res[p]
                r["M"][q, k] = [m[x] for x in RETRIEVAL]
                r["sse"][q, k] = float(((pr - y) ** 2).sum())
                r["ss"][q, k] = float((y * y).sum())
                sp = spearman(pr, y)
                if sp is not None:
                    r["sp"][q, k], r["sp_ok"][q, k] = sp, 1.0
                r["top5"][q, k] = top_overlap(s_hat, g)
            counts = pair_counts(base, g, approx, gold)
            for p_i, p in enumerate(names):
                res[p]["pairs"][q, k] = counts[p_i]
    return {"probes": res, "base_top5": base_top5}


def read_family(fam: str, meas: dict, qm: np.ndarray, W: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """quantities for one target family over a set of queries: the denominators and their readability, and per probe
    the fit measures, rho per metric and rho_bar with intervals, and the band."""
    mi = {m: METRIC_NAMES.index(m) for m in RETRIEVAL}
    base_idx = [BASE_COL[fam] + k for k in SEEDS]
    g_idx = [3 + k for k in SEEDS]
    n = int(qm.shape[0] if mask is None else mask.sum())
    out = {"queries": n, "denominators": {}, "probes": {}}
    dens, readable = {}, []
    for m in RETRIEVAL:
        den_q = (qm[:, g_idx, mi[m]] - qm[:, base_idx, mi[m]]).mean(1)
        sel = np.ones(qm.shape[0], dtype=bool) if mask is None else mask
        point = float(den_q[sel].mean()) if n else float("nan")
        with np.errstate(all="ignore"):
            boot = (W[:, sel] @ den_q[sel]) / W[:, sel].sum(1)
        interval = ci(boot)
        ok = bool(n and interval[0] > 0)
        out["denominators"][m] = {"gap": point, "ci": interval, "readable": ok}
        dens[m] = den_q
        if ok:
            readable.append(m)
    out["readable_metrics"] = readable
    for p, r in meas["probes"].items():
        entry = {}
        pt, bt = seed_mean_ratio(r["sse"], r["ss"], W, mask)
        entry["R2"] = {"point": 1 - pt, "ci": sorted(ci(1 - bt))}
        pt, bt = seed_mean_ratio(r["sp"] * r["sp_ok"], r["sp_ok"], W, mask)
        entry["spearman"] = {"point": pt, "ci": ci(bt)}
        dpr_q = np.where(r["pairs"][:, :, 0] > 0, r["pairs"][:, :, 1] / np.maximum(r["pairs"][:, :, 0], 1), 0.0)
        pt, bt = seed_mean_ratio(dpr_q, (r["pairs"][:, :, 0] > 0).astype(float), W, mask)
        entry["DPR"] = {"point": pt, "ci": ci(bt)}
        gdpr_q = np.where(r["pairs"][:, :, 2] > 0, r["pairs"][:, :, 3] / np.maximum(r["pairs"][:, :, 2], 1), 0.0)
        pt, bt = seed_mean_ratio(gdpr_q, (r["pairs"][:, :, 2] > 0).astype(float), W, mask)
        entry["gold_DPR"] = {"point": pt, "ci": ci(bt)}
        pt, bt = seed_mean_ratio(r["top5"], np.ones_like(r["top5"]), W, mask)
        entry["top5_overlap"] = {"point": pt, "ci": ci(bt)}
        rho, rho_boot = {}, {}
        for m in RETRIEVAL:
            num_q = (r["M"][:, :, RETRIEVAL.index(m)] - qm[:, base_idx, mi[m]]).mean(1)
            pt, bt = ratio(num_q, dens[m], W, mask)
            rho[m] = {"point": pt, "ci": ci(bt), "readable": m in readable}
            rho_boot[m] = bt
        entry["rho"] = rho
        if readable:
            pt = float(np.mean([rho[m]["point"] for m in readable]))
            bt = np.mean(np.stack([rho_boot[m] for m in readable]), 0)
            entry["rho_bar"] = {"point": pt, "ci": ci(bt)}
            entry["_rho_bar_boot"] = bt
        else:
            entry["rho_bar"] = {"point": None, "ci": None}
        entry["band"] = band(entry["rho_bar"]["point"] if readable else float("nan"), entry["rho_bar"]["ci"] or [0, 0], bool(readable))
        out["probes"][p] = entry
    pt, bt = seed_mean_ratio(meas["base_top5"], np.ones_like(meas["base_top5"]), W, mask)
    out["reference_top5_overlap"] = {"point": pt, "ci": ci(bt)}
    return out


def reproducibility(target: np.ndarray, sc: Sidecar, W: np.ndarray) -> dict:
    """references.reproducibility: the within-query-centred Pearson correlation of the target across seed pairs, pooled
    over U_q rows, and its mean over the three pairs."""
    starts = sc.ptr[:-1]
    sums = {(a, b): np.add.reduceat(target[:, a] * target[:, b], starts) for a in SEEDS for b in SEEDS if a <= b}
    pairs, boots = {}, []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pt = float(sums[(a, b)].sum() / math.sqrt(sums[(a, a)].sum() * sums[(b, b)].sum()))
        bt = (W @ sums[(a, b)]) / np.sqrt((W @ sums[(a, a)]) * (W @ sums[(b, b)]))
        pairs[f"{a}-{b}"] = {"point": pt, "ci": ci(bt)}
        boots.append(bt)
    return {"pairs": pairs, "mean": {"point": float(np.mean([v["point"] for v in pairs.values()])), "ci": ci(np.mean(np.stack(boots), 0))}}


def read_messages(probes: dict, W: np.ndarray) -> dict:
    """quantities.message: R2_t and cosine_t per probe, averaged over the seeds inside each resample."""
    out = {}
    for p in M_PROBES:
        st = np.stack([probes[f"m/{p}/{k}"] for k in SEEDS], 1)   # (queries, seeds, steps, 4)
        entry = {"R2_t": [], "cosine_t": []}
        r2_boots = []
        for t in range(STEPS):
            pt, bt = seed_mean_ratio(st[:, :, t, 0], st[:, :, t, 1], W)
            entry["R2_t"].append({"point": 1 - pt, "ci": sorted(ci(1 - bt))})
            r2_boots.append(1 - bt)
            pt, bt = seed_mean_ratio(st[:, :, t, 2], st[:, :, t, 3], W)
            entry["cosine_t"].append({"point": pt, "ci": ci(bt)})
        entry["R2_mean"] = {"point": float(np.mean([x["point"] for x in entry["R2_t"]])), "ci": ci(np.mean(np.stack(r2_boots), 0))}
        out[p] = entry
    return out


def strata_masks(name: str, sc: Sidecar) -> dict:
    """quantities.strata: metaqa by hop, 2wiki by gold_total, every dataset by the first-support STRUCT bucket."""
    out = {}
    hop = np.load(sc.dir / "q_hop.npy")
    gt = np.load(sc.dir / "q_gold_total.npy")
    fs = np.load(sc.dir / "q_first_support_STRUCT.npy")
    if name == "metaqa":
        out.update({f"hop={h}": hop == h for h in (1, 2, 3)})
    if name == "2wiki":
        out.update({"gold_total=2": gt == 2, "gold_total>=3": gt >= 3})
    labels = {-1: "no_gold_in_pool", 0: "none", 1: "h1", 2: "h2", 3: "h3"}
    out.update({f"first_support_STRUCT={labels[v]}": fs == v for v in (-1, 0, 1, 2, 3) if (fs == v).any()})
    return out


def read_dataset(name: str, d: Path, log=print) -> dict:
    sc = Sidecar(d)
    probes = load_probes(d)
    z = np.load(d / "z.npy")
    is_gold = np.load(d / "is_gold.npy")
    qm = np.load(d / "q_metrics.npy")
    gold_total = np.load(d / "q_gold_total.npy")
    r_c, e_c = probe_targets(sc)
    W = boot_weights(sc.n_q)
    others = {k: [o for o in SEEDS if o != k] for k in SEEDS}
    preds = {"r": {p: np.stack([probes[f"r/{p}/{k}"] for k in SEEDS], 1) for p in R_PROBES},
             "e": {p: np.stack([probes[f"e/{p}/{k}"] for k in SEEDS], 1) for p in E_PROBES}}
    preds["r"]["ref:twin"] = np.zeros_like(r_c)
    preds["r"]["ref:other_seed"] = np.stack([r_c[:, others[k]].mean(1) for k in SEEDS], 1)
    preds["e"]["ref:no_edge"] = np.zeros_like(e_c)
    preds["e"]["ref:other_seed"] = np.stack([e_c[:, others[k]].mean(1) for k in SEEDS], 1)
    t0 = time.time()
    meas = {fam: per_query_measures(sc, z, is_gold, gold_total, preds[fam], {"r": r_c, "e": e_c}[fam], fam) for fam in ("r", "e")}
    log(f"   {name}: per-query measures in {time.time() - t0:.0f}s")
    out = {"queries": sc.n_q, "uq_rows": sc.n_rows, "mean_uq": float(sc.sizes.mean()),
           "r": read_family("r", meas["r"], qm, W), "e": read_family("e", meas["e"], qm, W),
           "reproducibility": {"r": reproducibility(r_c, sc, W), "e": reproducibility(e_c, sc, W)},
           "messages": read_messages(probes, W), "strata": {}}
    for s_name, mask in strata_masks(name, sc).items():
        fam_r = read_family("r", meas["r"], qm, W, mask)
        out["strata"][s_name] = {"queries": fam_r["queries"], "denominators": fam_r["denominators"],
                                 "probes": {p: {"rho": v["rho"], "rho_bar": v["rho_bar"], "band": v["band"]} for p, v in fam_r["probes"].items()}}
    rp = out["r"]["probes"]
    contrasts = {}
    for c_name, (a, b) in {"vector_channels": ("B1-mlp", "B0-mlp"), "nonlinearity": ("B0-mlp", "B0-ridge"),
                           "twin_representation": ("B2-ridge", "B1-mlp")}.items():
        if out["r"]["readable_metrics"]:
            contrasts[c_name] = {"of": f"rho_bar({a}) - rho_bar({b})", "point": rp[a]["rho_bar"]["point"] - rp[b]["rho_bar"]["point"],
                                 "ci": ci(rp[a]["_rho_bar_boot"] - rp[b]["_rho_bar_boot"])}
        else:
            contrasts[c_name] = {"of": f"rho_bar({a}) - rho_bar({b})", "point": None, "ci": None}
    out["contrasts"] = contrasts
    for fam in ("r", "e"):
        for v in out[fam]["probes"].values():
            v.pop("_rho_bar_boot", None)
    for s in out["strata"].values():
        for v in s["probes"].values():
            v.pop("_rho_bar_boot", None)
    out.update(readings(out))
    return out


def readings(out: dict) -> dict:
    """readings: the dataset's band (the primary probe's), the flags and the interpretation_map entries that apply."""
    rp = out["r"]["probes"]
    reading = rp[PRIMARY]["band"]
    flags = []
    for fam in ("r", "e"):
        for p, v in out[fam]["probes"].items():
            if not p.startswith("ref:") and v["R2"]["point"] >= 0.5 and v["band"] == "L0_LOW":
                flags.append(f"FIT_NOT_RANK ({fam}, {p})")
    if out["reproducibility"]["r"]["mean"]["point"] < 0.5:
        flags.append("SEED_BOUND")
    if out["messages"]["B1-ridge"]["R2_mean"]["point"] >= 0.5 and rp["B1-mlp"]["band"] == "L0_LOW":
        flags.append("MESSAGE_NOT_RETRIEVAL")
    interp = []
    if reading == "L0_HIGH":
        interp.append("primary_high")
    vc = out["contrasts"]["vector_channels"]
    if reading == "L0_LOW" and vc["ci"] is not None and vc["ci"][0] > 0:
        interp.append("primary_low_vectors_add")
    if rp["B1-mlp"]["band"] == "L0_LOW":
        interp.append("b1_low")
    tr = out["contrasts"]["twin_representation"]
    if tr["ci"] is not None and tr["ci"][1] >= 0 and rp["B2-ridge"]["band"] not in ("L0_LOW", "NOT_READ"):
        interp.append("twin_representation")
    if "MESSAGE_NOT_RETRIEVAL" in flags:
        interp.append("message_not_retrieval")
    return {"reading": reading, "flags": flags, "interpretation": interp}


def stage_read(log=print, datasets=DATASETS, out_root: Path | None = None) -> dict:
    root = out_root or OUT
    decl = load_declaration()
    rec = {"phase": decl["phase"], "utc": utc(), "git_head": git_head(), "declaration_lf_sha256": lf_sha256(CONFIG),
           "registered_question": decl["registered_question"], "primary_probe": f"{PRIMARY} on r_k", "datasets": {}}
    for name in datasets:
        t0 = time.time()
        rec["datasets"][name] = read_dataset(name, root / name, log)
        rec["datasets"][name]["sidecar_meta_sha256"] = sha256_file(root / name / "meta.json")
        rec["datasets"][name]["probe_meta_sha256"] = sha256_file(root / name / "probes" / "probe_meta.json")
        log(f"{name}: read in {time.time() - t0:.0f}s -> {rec['datasets'][name]['reading']}")
    rec = clean(rec)
    path = root / "record.json"
    tmp = root / "record.tmp.json"
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return rec


def clean(v):
    """NaN and infinities become null in the record."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# ── stage: doc ───────────────────────────────────────────────────────────────


POPULATION_NOTE = {"metaqa": "the declared subsample: 800 per hop of the 19,738", "2wiki": "all of them", "squad": "all of them; the control"}


def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def fci(d: dict | None) -> str:
    if not d or d.get("point") is None:
        return "n/a"
    lo, hi = d["ci"] if d.get("ci") else (None, None)
    return f"{d['point']:.3f} [{f3(lo)}, {f3(hi)}]"


def render_doc(rec: dict, decl: dict, metas: dict) -> str:
    L = []
    add = L.append
    add("# MP-Approx level 0 (MP-ORACLE): how much of the GNN's effect node-local compiled information predicts")
    add("")
    add(f"Declared in `configs/mp_approx_l0.yaml` (status RUN; terminal STOP_FOR_REVIEW). Record: `outputs/mp_approx_l0/record.json` "
        f"(git-ignored), written {rec['utc']} at `{rec['git_head'][:7]}`.")
    add("")
    add(f"Registered question: \"{rec['registered_question']}\"")
    add("")
    add("This is level 0 of the MP-Approx ladder, on the MP-ORACLE track. The trained GNN's own outputs are the targets, "
        "which the proposal allows \"only to measure approximation capacity\". No probe is a retriever, a teacher or a feature.")
    add("")
    add("## What was measured")
    add("")
    add("- **Scored functions** (pilot trio checkpoints, V2_GATE rows only): the twin `u_mlp_v2_mix__H128__s{k}` (T_k), the GNN "
        "`u_gnn_v2_ef__H128__s{k}` (G_k) and the GNN's no-edge counterfactual (G0_k: one forward with `message_passing = False`), k = 0, 1, 2.")
    add("- **Targets**, centred within the query: r_k = z(G_k) - z(T_k) (primary; the GNN's residual over its matched twin), "
        "e_k = z(G_k) - z(G0_k) (what the cell's edges add inside the trained GNN), and m_k_t, the cell's neighbour message at step t (128-d).")
    add("- **Bases** (node-local, nothing propagated): B0 = the 129 contract scalars (signed log1p plus within-query z; 258 columns); "
        "B1 = B0 plus the twin's 515 vector channels (semantic head, semantic_difference, the three hop-1 family prototypes, seed reach); "
        "B2 = the twin's own body output and base_z.")
    add("- **Probes**: ridge and a 2x128 MLP, cross-fitted over five query folds (D0.2's rule); every prediction is out-of-fold.")
    add("- **Recovery** rho_M = (M(z(T_k) + rhat_k) - M(T_k)) / (M(G_k) - M(T_k)), means over seeds and queries, for M in recall@5, "
        "full_coverage@5 and hit@1; read only where the GNN-twin gap's 95% interval lies above 0. rho_bar is their mean. "
        "rho is 0 for the twin itself and 1 for an approximation that ranks U_q as G_k does; it can exceed 1 (the "
        "approximation ranks the golds better than G_k within U_q) or fall below 0.")
    add("- For e_k every measure takes G0_k in T_k's place: the gap is M(G_k) - M(G0_k), and DPR counts the pairs G0_k and "
        "G_k order oppositely.")
    add("- **U_q**: the in-pool golds plus the top 20 of each of the nine functions. Every endpoint metric is exact within U_q "
        "(checked on every query); an approximation's metrics within U_q are an **upper bound** on its full-pool metrics, "
        "so a LOW reading is robust and a HIGH one is not a full-pool claim.")
    add("")
    add("Bands of rho_bar: L0_HIGH (>= 0.75, interval low >= 0.50), L0_LOW (<= 0.25, interval high <= 0.50), L0_MID otherwise, "
        "NOT_READ with no readable metric. The dataset's reading is the band of B0-mlp on r_k.")
    add("")
    add("## Readings")
    add("")
    add("| dataset | queries | reading (B0-mlp on r) | rho_bar [95% CI] | readable metrics | flags | interpretation |")
    add("|---|---:|---|---|---|---|---|")
    for name, ds in rec["datasets"].items():
        p = ds["r"]["probes"][PRIMARY]
        add(f"| {name} | {ds['queries']:,} | **{ds['reading']}** | {fci(p['rho_bar'])} | {', '.join(ds['r']['readable_metrics']) or 'none'} | "
            f"{'; '.join(ds['flags']) or 'none'} | {', '.join(ds['interpretation']) or 'none'} |")
    add("")
    imap = decl["readings"]["interpretation_map"]
    used = sorted({i for ds in rec["datasets"].values() for i in ds["interpretation"]})
    if used:
        add("What the interpretation entries license, as filed before any number:")
        add("")
        for i in used:
            add(f"- **{i}**: {' '.join(str(imap[i]).split())}")
        add("")
    for name, ds in rec["datasets"].items():
        meta = metas[name]
        add(f"## {name}")
        add("")
        add(f"{ds['queries']:,} V2_GATE queries ({POPULATION_NOTE.get(name, 'all of them')}), {ds['uq_rows']:,} U_q rows "
            f"(mean |U_q| {ds['mean_uq']:.1f}; mean pool {meta['pool_mean']:.0f}).")
        add("")
        for fam, label, base in (("r", "r_k = z(G_k) - z(T_k), the GNN over its twin", "T_k"),
                                 ("e", "e_k = z(G_k) - z(G0_k), the cell's edges inside the GNN (descriptive)", "G0_k")):
            F = ds[fam]
            add(f"### Target {label}")
            add("")
            add(f"Denominator: the gap M(G_k) - M({base}), seed mean, with its 95% interval.")
            add("")
            add("| metric | gap | 95% CI | readable |")
            add("|---|---:|---|---|")
            for m, v in F["denominators"].items():
                add(f"| {m} | {f3(v['gap'])} | [{f3(v['ci'][0])}, {f3(v['ci'][1])}] | {'yes' if v['readable'] else 'no'} |")
            add("")
            add("| probe or reference | R2 | Spearman | DPR | gold DPR | top-5 overlap | rho recall@5 | rho full_cov@5 | rho hit@1 | rho_bar [95% CI] | band |")
            add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
            for p, v in F["probes"].items():
                rho = v["rho"]
                cells = [f3(rho[m]["point"]) + ("" if rho[m]["readable"] else " (nr)") for m in RETRIEVAL]
                add(f"| {p} | {f3(v['R2']['point'])} | {f3(v['spearman']['point'])} | {f3(v['DPR']['point'])} | {f3(v['gold_DPR']['point'])} | "
                    f"{f3(v['top5_overlap']['point'])} | {' | '.join(cells)} | {fci(v['rho_bar'])} | {v['band']} |")
            add("")
            add(f"Top-5 overlap of {base} itself with G_k: {fci(F['reference_top5_overlap'])}. (nr) = metric not readable on this dataset.")
            add("")
        rep = ds["reproducibility"]
        add("### Seed reproducibility of the targets")
        add("")
        add(f"Within-query-centred correlation across GNN seeds (mean of pairs 0-1, 0-2, 1-2): r {fci(rep['r']['mean'])}; e {fci(rep['e']['mean'])}.")
        add("")
        add("### Neighbour messages (m_k_t)")
        add("")
        add("| probe | R2 step 1 | R2 step 2 | R2 step 3 | cosine step 1 | cosine step 2 | cosine step 3 |")
        add("|---|---:|---:|---:|---:|---:|---:|")
        for p, v in ds["messages"].items():
            add(f"| {p} | {' | '.join(f3(x['point']) for x in v['R2_t'])} | {' | '.join(f3(x['point']) for x in v['cosine_t'])} |")
        add("")
        add("### Contrasts of rho_bar (paired bootstrap)")
        add("")
        for c, v in ds["contrasts"].items():
            add(f"- {c}: {v['of']} = {fci(v) if v['point'] is not None else 'n/a (no readable metric)'}")
        add("")
        add("### Strata (descriptive; a stratum's rho is read only where its own gap interval lies above 0)")
        add("")
        add("| stratum | queries | readable | B0-mlp rho_bar | B1-mlp rho_bar | B2-ridge rho_bar | other-seed rho_bar |")
        add("|---|---:|---|---|---|---|---|")
        for s, v in ds["strata"].items():
            readable = [m for m, x in v["denominators"].items() if x["readable"]]
            cells = [fci(v["probes"][p]["rho_bar"]) for p in ("B0-mlp", "B1-mlp", "B2-ridge", "ref:other_seed")]
            add(f"| {s} | {v['queries']:,} | {', '.join(readable) or 'none'} | {' | '.join(cells)} |")
        add("")
    add("## What this does not say")
    add("")
    add("- Every probe is an oracle fit on the GNN's own outputs over V2_GATE queries of the same dataset. Recovery is a statement "
        "of capacity within U_q, never of a deployable model, and no probe output enters any retriever, feature, teacher or selection.")
    add("- No reading here says that message passing is unnecessary, that it is not needed, or that the MLP wins; low and high "
        "recovery are both results (readings.wording).")
    add("- Level 1 is not opened by any reading here; v2.1 and v2.2A stay closed as mechanisms and v2.2B is not opened.")
    add("")
    add("## Integrity and compute")
    add("")
    add("| dataset | queries | U_q rows | chunk (queries) | scoring threads | scoring (min) | probes (min) | stored-metric, message-split and U_q checks | sidecar meta sha256 |")
    add("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for name, ds in rec["datasets"].items():
        meta = metas[name]
        add(f"| {name} | {meta['queries']:,} | {meta['uq_rows']:,} | {meta['chunk_queries']} | {meta['threads']} | "
            f"{meta['seconds_this_process'] / 60:.0f} | {meta['probe_seconds'] / 60:.0f} | {meta['mismatches']} mismatches | "
            f"`{ds['sidecar_meta_sha256'][:16]}` |")
    add("")
    add("Placement: laptop CPU for every stage (placement in the declaration). Scoring at 6 threads (the stored pilot records' count), "
        "probes at 4 threads per process.")
    add("")
    return LF.join(L)


def stage_doc(log=print, out_root: Path | None = None, doc: Path | None = None) -> None:
    root = out_root or OUT
    rec = json.loads((root / "record.json").read_text(encoding="utf-8"))
    metas = {}
    for name in rec["datasets"]:
        m = json.loads((root / name / "meta.json").read_text(encoding="utf-8"))
        m["pool_mean"] = float(np.load(root / name / "q_pool_size.npy").mean())
        m["probe_seconds"] = sum(json.loads(f.read_text(encoding="utf-8"))["seconds"] for f in (root / name / "probes").glob("fitlog_*.json"))
        metas[name] = m
    target = doc or DOC
    target.write_text(render_doc(rec, load_declaration(), metas), encoding="utf-8")
    log(f"wrote {target}")


# ── stage: file ──────────────────────────────────────────────────────────────


def stage_file(date: str, log=print, extra: dict | None = None) -> None:
    """run_record_mp_approx_l0_<date> appended to the declaration; status DECLARED_NOT_RUN -> RUN."""
    text = CONFIG.read_text(encoding="utf-8")
    decl = yaml.safe_load(text)
    key = f"run_record_mp_approx_l0_{date}"
    if key in decl:
        raise SystemExit(f"{key} exists")
    if decl["status"] != "DECLARED_NOT_RUN":
        raise SystemExit(f"status is {decl['status']}, not DECLARED_NOT_RUN")
    rec = json.loads(RECORD.read_text(encoding="utf-8"))
    per = {}
    for name, ds in rec["datasets"].items():
        meta = json.loads((OUT / name / "meta.json").read_text(encoding="utf-8"))
        p = ds["r"]["probes"][PRIMARY]
        per[name] = {"reading": ds["reading"], "primary_rho_bar": p["rho_bar"], "readable_metrics": ds["r"]["readable_metrics"],
                     "flags": ds["flags"], "interpretation": ds["interpretation"],
                     "bands_r": {q: v["band"] for q, v in ds["r"]["probes"].items()},
                     "bands_e": {q: v["band"] for q, v in ds["e"]["probes"].items()},
                     "queries": ds["queries"], "uq_rows": ds["uq_rows"], "scoring_mismatches": meta["mismatches"],
                     "sidecar_meta_sha256": ds["sidecar_meta_sha256"], "probe_meta_sha256": ds["probe_meta_sha256"]}
    run = {"utc": utc(), "status_moves": "DECLARED_NOT_RUN -> RUN", "terminal": "STOP_FOR_REVIEW", "held_half_read": False,
           "checkpoints_updated": 0, "placement": "laptop CPU; scoring 6 threads in one lane, probes 4 threads per process",
           "record_sha256": sha256_file(RECORD), "document": str(DOC.relative_to(ROOT)).replace(chr(92), "/"),
           "document_sha256": lf_sha256(DOC), "datasets": per}
    if extra:
        run.update(extra)
    block = yaml.safe_dump(clean({key: run}), sort_keys=False, width=160, allow_unicode=True)
    text = text.replace("status: DECLARED_NOT_RUN", "status: RUN", 1)
    CONFIG.write_text(text.rstrip(LF) + LF + LF + block, encoding="utf-8")
    log(f"filed {key}")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("score", "probe", "read", "doc", "file"))
    ap.add_argument("--dataset", choices=DATASETS, help="score and probe: one dataset per process")
    ap.add_argument("--limit", type=int, default=None, help="smoke check only: the first N scored queries, written to --out")
    ap.add_argument("--out", type=Path, default=None, help="smoke check only: a directory outside outputs/mp_approx_l0")
    ap.add_argument("--date", help="file: the run record's date, e.g. 2026_09_29")
    ap.add_argument("--extra", type=Path, default=None, help="file: a JSON object of fields added to the run record")
    args = ap.parse_args()

    def log(s: str) -> None:
        print(f"[{utc()}] {s}", flush=True)

    decl = load_declaration()
    if args.stage in ("score", "probe") and args.dataset is None:
        ap.error(f"--stage {args.stage} needs --dataset")
    if args.stage in ("read", "doc", "file") and args.dataset is not None:
        ap.error(f"--stage {args.stage} reads every dataset")
    if args.stage == "score":
        if (args.limit is None) != (args.out is None):
            ap.error("--limit and --out go together (a smoke check)")
        if args.out is not None:
            out = args.out.resolve()
            if OUT.resolve() in (out, *out.parents):
                ap.error("a smoke check never writes under outputs/mp_approx_l0")
            HARD_STOP_DIR[0] = out
            stage_score(decl, args.dataset, log, limit=args.limit, out_dir=out / args.dataset)
        else:
            stage_score(decl, args.dataset, log)
    elif args.stage == "probe":
        stage_probe(args.dataset, log)
    elif args.stage == "read":
        stage_read(log)
    elif args.stage == "doc":
        stage_doc(log)
    else:
        if not args.date:
            ap.error("--stage file needs --date")
        extra = json.loads(args.extra.read_text(encoding="utf-8")) if args.extra else None
        stage_file(args.date, log, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
