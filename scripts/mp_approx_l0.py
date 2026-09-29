"""MP-Approx level 0, track MP-ORACLE: how much of the trained GNN's effect node-local compiled information predicts
(configs/mp_approx_l0.yaml).

    python scripts/mp_approx_l0.py --stage score --dataset metaqa   # compile + the nine scored functions, integrity-checked -> U_q sidecar
    python scripts/mp_approx_l0.py --stage probe --dataset metaqa   # cross-fitted probes -> out-of-fold predictions
    python scripts/mp_approx_l0.py --stage read                     # quantities, bootstrap, bands -> record.json
    python scripts/mp_approx_l0.py --stage doc
    python scripts/mp_approx_l0.py --stage file --date 2026_09_29

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


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("score",))
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--limit", type=int, default=None, help="smoke check only: the first N scored queries, written to --out")
    ap.add_argument("--out", type=Path, default=None, help="smoke check only: a directory outside outputs/mp_approx_l0")
    args = ap.parse_args()

    def log(s: str) -> None:
        print(f"[{utc()}] {s}", flush=True)

    decl = load_declaration()
    if args.stage == "score":
        if args.dataset is None:
            ap.error("--stage score needs --dataset")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
