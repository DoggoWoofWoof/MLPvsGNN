"""M3B training and evaluation (configs/m3b_controlled_comparison.yaml#training,
#measurement): the carve caches, the packed-batch assembly, the metrics from
raw ranks, the fit loop with dataset-balanced batches and early stopping on the
select carves, and the evaluation pass.

The cache holds what is expensive and deterministic (the scalar block per
candidate, the pool, the seeds, the seed-reach weights, the in-pool gold and
the query embedding); the pool-graph edges and the node embeddings are
re-derived per batch from the stores and the served embeddings by the same
functions the compiler used, so a cached and a freshly compiled query are the
same object.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from mp_retrieval.m3b_features import (
    COLUMNS, DIM, FAMILIES, MAX_SEEDS, N_COLUMNS, DenseNodes, RelationTable, pool_edges,
)
from mp_retrieval.m3b_models import N_EDGE_FEATURES, PackedBatch, listwise_loss

KS = (1, 5, 10, 20)
METRIC_NAMES = tuple([f"recall@{k}" for k in KS] + ["hit@1", "mrr", "ndcg@5", "ndcg@20", "full_coverage@5", "full_coverage@20", "first_gold_rank", "gold_in_pool", "gold_total", "pool_size"])


# ── metrics from raw ranks ───────────────────────────────────────────────────


def rank_metrics(scores: np.ndarray, gold_local: np.ndarray, gold_total: int) -> dict[str, float]:
    """All metrics of one query from its scores over the pool. Ties are broken
    by pool order (ascending position), deterministically. recall@k divides by
    the total gold count, so unretrieved gold counts against the model; MRR is
    1 / rank of the first gold in the pool, 0 when no gold is in the pool."""
    n = scores.shape[0]
    order = np.argsort(-scores, kind="stable")
    rank_of = np.empty(n, dtype=np.int64)
    rank_of[order] = np.arange(1, n + 1)
    gold_local = np.asarray(gold_local, dtype=np.int64)
    gold_ranks = np.sort(rank_of[gold_local]) if gold_local.size else np.empty(0, dtype=np.int64)
    first = int(gold_ranks[0]) if gold_ranks.size else 0
    out: dict[str, float] = {}
    total = max(int(gold_total), 1)
    for k in KS:
        out[f"recall@{k}"] = float((gold_ranks <= k).sum()) / total
    out["hit@1"] = float(first == 1)
    out["mrr"] = 1.0 / first if first else 0.0
    for k in (5, 20):
        hits = gold_ranks[gold_ranks <= k]
        dcg = float((1.0 / np.log2(hits + 1.0)).sum())
        ideal = float((1.0 / np.log2(np.arange(1, min(total, k) + 1) + 1.0)).sum())
        out[f"ndcg@{k}"] = dcg / ideal if ideal > 0 else 0.0
        out[f"full_coverage@{k}"] = float((gold_ranks <= k).sum() == total)
    out["first_gold_rank"] = float(first)
    out["gold_in_pool"] = float(gold_local.size)
    out["gold_total"] = float(gold_total)
    out["pool_size"] = float(n)
    return out


def mrr_audit(first_gold_rank: np.ndarray, mrr: np.ndarray) -> dict:
    """Recompute MRR from the stored first-gold ranks and compare."""
    first = np.asarray(first_gold_rank, dtype=np.float64)
    recomputed = np.where(first > 0, 1.0 / np.maximum(first, 1.0), 0.0)
    return {"max_abs_diff": float(np.abs(recomputed - np.asarray(mrr, dtype=np.float64)).max()) if first.size else 0.0,
            "mean_recomputed": float(recomputed.mean()) if first.size else 0.0,
            "mean_stored": float(np.asarray(mrr, dtype=np.float64).mean()) if first.size else 0.0,
            "queries": int(first.size)}


# ── the carve cache ──────────────────────────────────────────────────────────


class NpyAppender:
    """Writes one .npy file row-block by row-block without holding the array:
    a header for a placeholder row count is written first and rewritten with
    the final shape at close (padded to the same length, as the format allows)."""

    PLACEHOLDER_ROWS = 10**15

    def __init__(self, path: Path, dtype, columns: int | None):
        self.path, self.dtype, self.columns, self.rows = Path(path), np.dtype(dtype), columns, 0
        self.f = open(self.path, "wb")
        self.header_len = len(self._header(self.PLACEHOLDER_ROWS))
        self.f.write(self._header(self.PLACEHOLDER_ROWS))

    def _shape(self, rows: int) -> tuple:
        return (rows,) if self.columns is None else (rows, self.columns)

    def _header(self, rows: int) -> bytes:
        import io
        buf = io.BytesIO()
        np.lib.format.write_array_header_1_0(buf, {"descr": np.lib.format.dtype_to_descr(self.dtype), "fortran_order": False, "shape": self._shape(rows)})
        return buf.getvalue()

    def append(self, block: np.ndarray) -> None:
        block = np.ascontiguousarray(np.asarray(block, dtype=self.dtype))
        if block.shape[1:] != self._shape(0)[1:]:
            raise ValueError(f"{self.path.name}: block shape {block.shape} does not match {self._shape(0)}")
        self.f.write(block.tobytes())
        self.rows += int(block.shape[0])

    def close(self) -> None:
        header = self._header(self.rows)
        if len(header) > self.header_len:
            raise RuntimeError(f"{self.path.name}: final header longer than the placeholder")
        header = header[:-1] + b" " * (self.header_len - len(header)) + b"\n"   # the format pads its header with spaces before the newline
        self.f.seek(0)
        self.f.write(header)
        self.f.close()


class CacheWriter:
    """Accumulates compiled queries of one carve and writes the cache directory.
    The row-proportional arrays (pool, scalars, seedw) stream to disk as they
    are added; only the per-query arrays are held until the end."""

    STREAMED = ("pool", "scalars", "seedw")

    def __init__(self, out_dir: Path, meta: dict):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta)
        self.qrow: list[int] = []
        self.qemb: list[np.ndarray] = []
        self.pool_ptr: list[int] = [0]
        self.seeds: list[np.ndarray] = []
        self.gold: list[np.ndarray] = []
        self.gold_total: list[int] = []
        self.edge_counts: list[list[int]] = []
        self.query_ids: list[str] = []
        self.streams = {"pool": NpyAppender(self.out_dir / "pool.npy", np.int32, None),
                        "scalars": NpyAppender(self.out_dir / "scalars.npy", np.float16, N_COLUMNS),
                        "seedw": NpyAppender(self.out_dir / "seedw.npy", np.float16, MAX_SEEDS)}

    def add(self, query_id: str, qrow: int, qemb: np.ndarray, compiled, gold_local: np.ndarray, gold_total: int) -> None:
        self.query_ids.append(query_id)
        self.qrow.append(int(qrow))
        self.qemb.append(np.asarray(qemb, dtype=np.float16))
        self.streams["pool"].append(compiled.pool)
        self.streams["scalars"].append(compiled.scalars)
        self.streams["seedw"].append(compiled.seedw)
        self.pool_ptr.append(self.pool_ptr[-1] + int(compiled.pool.shape[0]))
        self.seeds.append(compiled.seeds_local.astype(np.int16))
        self.gold.append(np.asarray(gold_local, dtype=np.int16))
        self.gold_total.append(int(gold_total))
        self.edge_counts.append([compiled.n_edges[f] for f in FAMILIES])

    def write(self) -> dict:
        for s in self.streams.values():
            s.close()

        def ptr(chunks):
            return np.r_[0, np.cumsum([c.shape[0] for c in chunks])].astype(np.int64)

        arrays = {
            "qrow": np.asarray(self.qrow, dtype=np.int64), "qemb": np.stack(self.qemb) if self.qemb else np.empty((0, DIM), np.float16),
            "pool_ptr": np.asarray(self.pool_ptr, dtype=np.int64),
            "seeds_ptr": ptr(self.seeds), "seeds": np.concatenate(self.seeds) if self.seeds else np.empty(0, np.int16),
            "gold_ptr": ptr(self.gold), "gold": np.concatenate(self.gold) if self.gold else np.empty(0, np.int16),
            "gold_total": np.asarray(self.gold_total, dtype=np.int32),
            "edge_counts": np.asarray(self.edge_counts, dtype=np.int32).reshape(-1, len(FAMILIES)),
        }
        for name, arr in arrays.items():
            np.save(self.out_dir / f"{name}.npy", arr)
        (self.out_dir / "query_ids.json").write_text(json.dumps(self.query_ids), encoding="utf-8")
        n_rows = int(self.pool_ptr[-1])
        for name in self.STREAMED:   # the streamed files must read back with the row count the pointers imply
            shape = np.load(self.out_dir / f"{name}.npy", mmap_mode="r").shape
            if shape[0] != n_rows:
                raise RuntimeError(f"{name}.npy holds {shape[0]} rows, pool_ptr implies {n_rows}")
        meta = {**self.meta, "n_queries": len(self.qrow), "n_rows": n_rows,
                "candidates_mean": float(np.diff(arrays["pool_ptr"]).mean()) if self.qrow else 0.0,
                "edges_per_query_mean": {f: float(arrays["edge_counts"][:, i].mean()) if self.qrow else 0.0 for i, f in enumerate(FAMILIES)},
                "gold_in_pool_mean": float(np.diff(arrays["gold_ptr"]).mean()) if self.qrow else 0.0,
                "queries_with_no_gold_in_pool": int((np.diff(arrays["gold_ptr"]) == 0).sum()),
                "bytes": int(sum((self.out_dir / f"{n}.npy").stat().st_size for n in list(arrays) + list(self.STREAMED)))}
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return meta


@dataclass
class DatasetContext:
    """What a dataset needs at batch time besides the cache: its stores, node
    embeddings and relation table."""

    name: str
    stores: dict
    nodes: DenseNodes
    rel_table: RelationTable | None


class CarveData:
    """One cached carve, read lazily (the scalar block is memory-mapped)."""

    def __init__(self, cache_dir: Path, context: DatasetContext, columns: np.ndarray | None = None):
        d = Path(cache_dir)
        self.dir = d
        self.context = context
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        # the QLS_U_CORE_CONTRACT column subset (indices into the full layout); None reads every
        # compiled column. A trimmed cache stores a named subset (meta columns_stored), so the
        # requested layout indices are mapped through the stored names; a missing name refuses.
        stored = self.meta.get("columns_stored")
        if columns is None:
            self.columns = None
            if stored is not None:
                raise ValueError(f"{d}: trimmed to {len(stored)} columns; the full layout is not available")
        else:
            requested = np.asarray(columns, dtype=np.int64)
            if stored is None:
                self.columns = requested
            else:
                position = {name: k for k, name in enumerate(stored)}
                missing = [COLUMNS[c] for c in requested if COLUMNS[c] not in position]
                if missing:
                    raise ValueError(f"{d}: trimmed cache lacks {missing}")
                self.columns = np.asarray([position[COLUMNS[c]] for c in requested], dtype=np.int64)
        self.qrow = np.load(d / "qrow.npy")
        self.qemb = np.load(d / "qemb.npy", mmap_mode="r")
        self.pool_ptr = np.load(d / "pool_ptr.npy")
        self.pool = np.load(d / "pool.npy", mmap_mode="r")
        self.scalars = np.load(d / "scalars.npy", mmap_mode="r")
        self.seedw = np.load(d / "seedw.npy", mmap_mode="r")
        self.seeds_ptr = np.load(d / "seeds_ptr.npy")
        self.seeds = np.load(d / "seeds.npy")
        self.gold_ptr = np.load(d / "gold_ptr.npy")
        self.gold = np.load(d / "gold.npy")
        self.gold_total = np.load(d / "gold_total.npy")
        self.n_queries = int(self.qrow.size)
        self.trainable = np.flatnonzero(np.diff(self.gold_ptr) > 0)

    def query(self, i: int) -> dict:
        a, b = int(self.pool_ptr[i]), int(self.pool_ptr[i + 1])
        return {
            "pool": np.asarray(self.pool[a:b], dtype=np.int64),
            # float16 as stored (the column subset taken before any conversion); pack_queries converts once per batch
            "x": np.asarray(self.scalars[a:b]) if self.columns is None else np.asarray(self.scalars[a:b])[:, self.columns],
            "seedw": np.asarray(self.seedw[a:b], dtype=np.float32), "qemb": np.asarray(self.qemb[i], dtype=np.float32),
            "seeds": np.asarray(self.seeds[self.seeds_ptr[i]:self.seeds_ptr[i + 1]], dtype=np.int64),
            "gold": np.asarray(self.gold[self.gold_ptr[i]:self.gold_ptr[i + 1]], dtype=np.int64), "gold_total": int(self.gold_total[i]),
        }

    def pack(self, indices: np.ndarray, families: tuple[str, ...] = FAMILIES) -> PackedBatch:
        return pack_queries([self.query(int(i)) for i in indices], self.context, families)


def pack_queries(queries: list[dict], context: DatasetContext, families: tuple[str, ...] = FAMILIES) -> PackedBatch:
    """Assemble a PackedBatch; edges are re-derived with pool_edges and node
    embeddings gathered from the served store."""
    xs, embs, qembs, seedws, seed_nodes, golds, eis, eas = [], [], [], [], [], [], [], []
    ptr = [0]
    node_query = []
    for qi, qd in enumerate(queries):
        pool = qd["pool"]
        n = pool.size
        off = ptr[-1]
        xs.append(qd["x"])
        embs.append(qd["emb"] if "emb" in qd else context.nodes.read(pool, dtype=np.float16))   # the eval pass passes the rows it already gathered
        qembs.append(qd["qemb"])
        seedws.append(qd["seedw"])
        row = np.full(MAX_SEEDS, -1, dtype=np.int64)
        row[: qd["seeds"].size] = qd["seeds"][:MAX_SEEDS] + off
        seed_nodes.append(row)
        g = np.zeros(n, dtype=bool)
        g[qd["gold"]] = True
        golds.append(g)
        relcos = (context.rel_table.embeddings @ qd["qemb"]).astype(np.float32) if context.rel_table is not None else None
        edges, _ = pool_edges(pool, context.stores, relcos)
        for f_i, fam in enumerate(FAMILIES):
            if fam not in families:
                continue
            u, v, attr = edges[fam]
            if u.size == 0:
                continue
            eis.append(np.stack((u.astype(np.int64) + off, v.astype(np.int64) + off)))
            onehot = np.zeros((u.size, len(FAMILIES)), dtype=np.float32)
            onehot[:, f_i] = 1.0
            eas.append(np.concatenate((onehot, attr), axis=1))
        node_query.append(np.full(n, qi, dtype=np.int64))
        ptr.append(off + n)
    edge_index = np.concatenate(eis, axis=1) if eis else np.empty((2, 0), dtype=np.int64)
    edge_attr = np.concatenate(eas, axis=0) if eas else np.empty((0, N_EDGE_FEATURES), dtype=np.float32)
    # scalars and embeddings arrive as stored (float16) or as the caller gathered them; one
    # conversion to float32 per batch through torch (exact) replaces a per-query numpy astype
    return PackedBatch(
        x=torch.from_numpy(np.concatenate(xs)).to(torch.float32), qptr=torch.tensor(ptr, dtype=torch.long), node_query=torch.from_numpy(np.concatenate(node_query)),
        emb=torch.from_numpy(np.concatenate(embs)).to(torch.float32), qemb=torch.from_numpy(np.stack(qembs)), seedw=torch.from_numpy(np.concatenate(seedws)),
        seed_nodes=torch.from_numpy(np.stack(seed_nodes)), edge_index=torch.from_numpy(edge_index), edge_attr=torch.from_numpy(edge_attr),
        gold=torch.from_numpy(np.concatenate(golds)),
    )


def concat_batches(batches: list[PackedBatch]) -> PackedBatch:
    """Several packed batches as one: node and query indices are offset, absent
    seed slots (-1) stay absent. The per-query loss and metrics are unchanged by
    the concatenation, so a mixed-dataset batch is packed per dataset and joined."""
    if len(batches) == 1:
        return batches[0]
    node_off, query_off = 0, 0
    qptr = [torch.zeros(1, dtype=torch.long)]
    node_query, seed_nodes, edge_index = [], [], []
    for b in batches:
        qptr.append(b.qptr[1:] + node_off)
        node_query.append(b.node_query + query_off)
        seed_nodes.append(torch.where(b.seed_nodes >= 0, b.seed_nodes + node_off, b.seed_nodes))
        edge_index.append(b.edge_index + node_off)
        node_off += int(b.x.shape[0])
        query_off += b.n_queries
    return PackedBatch(
        x=torch.cat([b.x for b in batches]), qptr=torch.cat(qptr), node_query=torch.cat(node_query),
        emb=torch.cat([b.emb for b in batches]), qemb=torch.cat([b.qemb for b in batches]), seedw=torch.cat([b.seedw for b in batches]),
        seed_nodes=torch.cat(seed_nodes), edge_index=torch.cat(edge_index, dim=1), edge_attr=torch.cat([b.edge_attr for b in batches]),
        gold=torch.cat([b.gold for b in batches]),
    )


# ── evaluation ───────────────────────────────────────────────────────────────


def node_budgeted_batches(pool_ptr: np.ndarray, n_queries: int, batch_size: int, max_nodes: int) -> list[np.ndarray]:
    """Consecutive query index blocks of at most ``batch_size`` queries and, past
    the first query of a block, at most ``max_nodes`` candidate rows -- so a
    block of 2,000-candidate pools is a few queries and a block of 50-candidate
    pools is the full ``batch_size``. Scores do not depend on the blocking."""
    sizes = np.diff(pool_ptr[: n_queries + 1])
    blocks, start = [], 0
    while start < n_queries:
        end, nodes = start + 1, int(sizes[start])
        while end < n_queries and end - start < batch_size and nodes + int(sizes[end]) <= max_nodes:
            nodes += int(sizes[end])
            end += 1
        blocks.append(np.arange(start, end))
        start = end
    return blocks


@torch.no_grad()
def evaluate_carve(model: torch.nn.Module, data: CarveData, batch_size: int = 32, families: tuple[str, ...] = FAMILIES,
                   max_nodes: int = 24_000) -> dict[str, np.ndarray]:
    model.eval()
    rows: list[dict] = []
    for idx in node_budgeted_batches(data.pool_ptr, data.n_queries, batch_size, max_nodes):
        batch = data.pack(idx, families)
        scores = model(batch).cpu().numpy()
        ptr = batch.qptr.numpy()
        for j, i in enumerate(idx):
            qd = data.query(int(i))
            rows.append(rank_metrics(scores[ptr[j]:ptr[j + 1]], qd["gold"], qd["gold_total"]))
    return {name: np.asarray([r[name] for r in rows], dtype=np.float64) for name in METRIC_NAMES}


def macro_recall_at_5(per_dataset: dict[str, dict[str, np.ndarray]]) -> float:
    return float(np.mean([m["recall@5"].mean() for m in per_dataset.values()]))


# ── the fit loop ─────────────────────────────────────────────────────────────


class EpochTooLong(RuntimeError):
    """compute.abort_criteria: an epoch over the declared limit halts the screen."""


@dataclass
class FitRecord:
    arm: str
    config: dict
    seed: int
    epochs_run: int = 0
    best_epoch: int = -1
    best_select_macro_recall5: float = -1.0
    history: list = field(default_factory=list)
    seconds: float = 0.0
    parameters: int = 0
    steps: int = 0
    batches_skipped_no_gold: int = 0


def next_queries(fits: dict[str, CarveData], name: str, cursors: dict, rng: np.random.Generator, count: int) -> np.ndarray:
    """The next ``count`` trainable fit queries of a dataset, without replacement
    until the carve is exhausted, then a fresh permutation."""
    perm, pos = cursors[name]
    if pos + count > perm.size:
        perm = rng.permutation(fits[name].trainable)
        pos = 0
    cursors[name] = [perm, pos + count]
    return perm[pos:pos + count]


def draw_batch(fits: dict[str, CarveData], names: list[str], cursors: dict, rng: np.random.Generator, batch_size: int,
               families: tuple[str, ...], dataset_draw: str) -> PackedBatch:
    """One training batch under the declared sampler (see fit_model)."""
    if dataset_draw == "per_batch":
        name = names[int(rng.integers(len(names)))]
        return fits[name].pack(next_queries(fits, name, cursors, rng, batch_size), families)
    draws = rng.integers(len(names), size=batch_size)
    parts = []
    for k, name in enumerate(names):
        count = int((draws == k).sum())
        if count:
            parts.append(fits[name].pack(next_queries(fits, name, cursors, rng, count), families))
    return concat_batches(parts)


def fit_model(model: torch.nn.Module, fits: dict[str, CarveData], selects: dict[str, CarveData], *, seed: int, arm: str, config: dict,
              max_epochs: int = 6, batches_per_epoch: int = 2000, batch_size: int = 16, patience: int = 2, lr: float = 1e-3,
              weight_decay: float = 1e-4, clip: float = 1.0, families: tuple[str, ...] = FAMILIES, dataset_draw: str = "per_query",
              epoch_limit_s: float | None = None, log=print) -> tuple[torch.nn.Module, FitRecord]:
    """AdamW on dataset-balanced batches: for every slot of a batch the dataset
    is drawn uniformly and then a fit query of that dataset (``dataset_draw``
    "per_query", the declared sampler; "per_batch" draws one dataset for the
    whole batch); queries are drawn without replacement per dataset until its
    trainable fit carve is exhausted, then reshuffled. Early stopping on the
    macro select recall@5 with the declared patience; the best epoch's weights
    are returned. An epoch (training batches plus the select evaluation) longer
    than ``epoch_limit_s`` raises EpochTooLong: the declared abort criterion."""
    if dataset_draw not in ("per_query", "per_batch"):
        raise ValueError(f"dataset_draw must be per_query or per_batch, got {dataset_draw!r}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    names = sorted(fits)
    cursors = {n: [rng.permutation(fits[n].trainable), 0] for n in names}
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    record = FitRecord(arm=arm, config=dict(config), seed=seed, parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()
    bad = 0
    for epoch in range(max_epochs):
        model.train()
        t_epoch = time.time()
        losses = []
        for _ in range(batches_per_epoch):
            batch = draw_batch(fits, names, cursors, rng, batch_size, families, dataset_draw)
            if not bool(batch.gold.any()):
                record.batches_skipped_no_gold += 1
                continue
            optimiser.zero_grad(set_to_none=True)
            loss = listwise_loss(model(batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimiser.step()
            losses.append(float(loss.detach()))
            record.steps += 1
        per_dataset = {n: evaluate_carve(model, selects[n], families=families) for n in sorted(selects)}
        macro = macro_recall_at_5(per_dataset)
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None, "select_macro_recall@5": macro,
                 "select_recall@5": {n: float(m["recall@5"].mean()) for n, m in per_dataset.items()}, "seconds": round(time.time() - t_epoch, 1)}
        record.history.append(entry)
        record.epochs_run = epoch + 1
        log(f"      epoch {epoch}: loss {entry['train_loss']:.4f} select macro R@5 {macro:.4f} ({entry['seconds']}s) " +
            " ".join(f"{n}={v:.3f}" for n, v in entry["select_recall@5"].items()))
        if epoch_limit_s is not None and entry["seconds"] > epoch_limit_s:
            raise EpochTooLong(f"epoch {epoch} took {entry['seconds']}s, over the declared {epoch_limit_s:.0f}s: the screen halts; "
                               "file the fallback as an amendment before continuing")
        if macro > record.best_select_macro_recall5 + 1e-9:
            record.best_select_macro_recall5 = macro
            record.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    record.seconds = round(time.time() - t0, 1)
    return model, record
