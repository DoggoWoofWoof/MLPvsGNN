"""M3B pools: training carves, bounded N-hop expansion, family stores and the
pool graph (configs/m3b_controlled_comparison.yaml#populations, #candidate_contract,
#pool_graph).

Everything here is a deterministic function of the served substrate and the
declaration. The expansion generalises ``scripts/m3a_headroom.py::expand`` to
N hops and is byte-identical to it for N <= 2 (a test holds that); the family
store is the same ordered undirected CSR as ``FamilyCSR`` there, with the
weights, typed relations and stored direction kept beside the columns because
the pool graph reads them. Nothing in this module reads a gold label.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── training carves ──────────────────────────────────────────────────────────


def carve_ids(sorted_ids: list[str], select_cap: int = 1500, select_fraction: int = 5, fit_cap: int = 6000) -> tuple[list[str], list[str]]:
    """The declared stride carves of one dataset's training ids.

    ``select`` = ``sorted_ids[::s_sel]`` with ``s_sel = max(2, round(N / size))``
    and ``size = min(select_cap, N // select_fraction)``; ``fit`` = the remaining
    ids strided to at most ``fit_cap``. No seed, no shuffle: the carve is a
    function of the sorted id list alone.
    """

    n = len(sorted_ids)
    if n < 2 * select_fraction:
        raise ValueError(f"{n} ids is too few to carve")
    size = min(select_cap, n // select_fraction)
    s_sel = max(2, int(round(n / size)))
    select_idx = set(range(0, n, s_sel))
    select = [sorted_ids[i] for i in sorted(select_idx)]
    remaining = [sorted_ids[i] for i in range(n) if i not in select_idx]
    s_fit = max(1, math.ceil(len(remaining) / fit_cap))
    fit = remaining[::s_fit]
    return fit, select


def ids_digest(ids: list[str]) -> str:
    return hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest()


# ── family stores ────────────────────────────────────────────────────────────


def _first_occurrence_unique(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    _, first = np.unique(values, return_index=True)
    return values[np.sort(first)]


@dataclass
class FamilyStore:
    """One family as an ordered undirected CSR (``indptr``/``col``, the same
    order as the headroom's FamilyCSR) plus what the pool graph needs beside it:
    ``weight`` aligned with ``col`` for ner/knn; for structural the uncollapsed
    typed entries ``typed_col``/``typed_rel``/``typed_dir`` under ``typed_indptr``
    (dir 1 = the row node is the stored source, 2 = the stored destination), and
    ``rel_count`` = corpus count of stored edges per relation (for rel_ief)."""

    family: str
    n_nodes: int
    indptr: np.ndarray
    col: np.ndarray
    weight: np.ndarray | None = None
    typed_indptr: np.ndarray | None = None
    typed_col: np.ndarray | None = None
    typed_rel: np.ndarray | None = None
    typed_dir: np.ndarray | None = None
    rel_count: np.ndarray | None = None
    edges_stored: int = 0

    @classmethod
    def from_graph(cls, graph, family: str) -> "FamilyStore":
        n = int(graph.n_nodes)
        src = graph.src.astype(np.int32, copy=False)
        dst = graph.dst.astype(np.int32, copy=False)
        heads = np.concatenate((src, dst))
        tails = np.concatenate((dst, src))
        if graph.weight is not None:
            weight = graph.weight.astype(np.float32, copy=False)
            w2 = np.concatenate((weight, weight))
            order = np.lexsort((tails, -w2, heads))
            heads, tails, w2 = heads[order], tails[order], w2[order]
            del order
            counts = np.bincount(heads, minlength=n).astype(np.int64)
            indptr = np.zeros(n + 1, dtype=np.int64)
            np.cumsum(counts, out=indptr[1:])
            return cls(family, n, indptr, tails.astype(np.int32, copy=False), weight=w2.astype(np.float16), edges_stored=int(src.size))
        # structural: typed entries kept, adjacency collapsed
        rel = graph.rel.astype(np.int16, copy=False) if getattr(graph, "rel", None) is not None else np.zeros(src.size, dtype=np.int16)
        rel2 = np.concatenate((rel, rel))
        dir2 = np.concatenate((np.ones(src.size, dtype=np.int8), np.full(src.size, 2, dtype=np.int8)))
        order = np.lexsort((rel2, tails, heads))
        heads, tails, rel2, dir2 = heads[order], tails[order], rel2[order], dir2[order]
        del order
        t_counts = np.bincount(heads, minlength=n).astype(np.int64)
        typed_indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(t_counts, out=typed_indptr[1:])
        keep = np.ones(heads.size, dtype=bool)
        if heads.size:
            keep[1:] = (heads[1:] != heads[:-1]) | (tails[1:] != tails[:-1])
        c_heads, c_tails = heads[keep], tails[keep]
        counts = np.bincount(c_heads, minlength=n).astype(np.int64)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        n_rel = int(rel.max()) + 1 if rel.size else 0
        rel_count = np.bincount(rel.astype(np.int64), minlength=n_rel).astype(np.int64)
        return cls(family, n, indptr, c_tails.astype(np.int32, copy=False), typed_indptr=typed_indptr,
                   typed_col=tails.astype(np.int32, copy=False), typed_rel=rel2, typed_dir=dir2, rel_count=rel_count,
                   edges_stored=int(src.size))

    # the FamilyCSR interface the headroom helpers read
    def capped(self, nodes: np.ndarray, cap: int) -> np.ndarray:
        starts = self.indptr[nodes]
        lens = np.minimum(self.indptr[nodes + 1] - starts, cap)
        total = int(lens.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        offsets = np.repeat(np.cumsum(lens) - lens, lens)
        idx = np.repeat(starts, lens) + (np.arange(total, dtype=np.int64) - offsets)
        return self.col[idx].astype(np.int64, copy=False)

    def degree(self, nodes: np.ndarray) -> np.ndarray:
        return self.indptr[nodes + 1] - self.indptr[nodes]

    def save(self, path: Path) -> None:
        arrays = {"indptr": self.indptr, "col": self.col, "meta": np.asarray([self.n_nodes, self.edges_stored], dtype=np.int64)}
        for name in ("weight", "typed_indptr", "typed_col", "typed_rel", "typed_dir", "rel_count"):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, family: str) -> "FamilyStore":
        with np.load(path) as z:
            meta = z["meta"]
            kwargs = {name: z[name] for name in ("weight", "typed_indptr", "typed_col", "typed_rel", "typed_dir", "rel_count") if name in z.files}
            return cls(family, int(meta[0]), z["indptr"], z["col"], edges_stored=int(meta[1]), **kwargs)


def load_or_build_store(ds, family: str, cache_dir: Path) -> FamilyStore:
    path = cache_dir / f"{ds.name}_{family}.npz"
    if path.exists():
        return FamilyStore.load(path, family)
    graph = ds.graph(family)
    store = FamilyStore.from_graph(graph, family)
    del graph
    store.save(path)
    return store


# ── expansion ────────────────────────────────────────────────────────────────


def capped_multi(nodes: np.ndarray, families: list, cap: int) -> np.ndarray:
    """The first ``cap`` neighbours of every node in every family, node-major then
    family order then within-family rank -- the headroom's function, unchanged."""
    if nodes.size == 0:
        return np.empty(0, dtype=np.int64)
    cols, node_idx, fam_idx, rank = [], [], [], []
    for f_i, fam in enumerate(families):
        starts = fam.indptr[nodes]
        lens = np.minimum(fam.indptr[nodes + 1] - starts, cap)
        total = int(lens.sum())
        if total == 0:
            continue
        within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens)
        cols.append(fam.col[np.repeat(starts, lens) + within].astype(np.int64, copy=False))
        node_idx.append(np.repeat(np.arange(nodes.size, dtype=np.int64), lens))
        fam_idx.append(np.full(total, f_i, dtype=np.int8))
        rank.append(within)
    if not cols:
        return np.empty(0, dtype=np.int64)
    if len(cols) == 1:
        return cols[0]
    order = np.lexsort((np.concatenate(rank), np.concatenate(fam_idx), np.concatenate(node_idx)))
    return np.concatenate(cols)[order]


def expand_hops(seeds: np.ndarray, families: list, setting: dict) -> np.ndarray:
    """Bounded seed neighbourhood under one setting, over the union of the given
    families, for any number of hops. Hop 1 takes the first ``per_seed_cap``
    neighbours of each seed; every later hop expands the previous hop's fresh
    nodes in order of first appearance, each contributing its first
    ``per_frontier_cap`` neighbours per family, skipping visited nodes, until
    ``visited_cap`` nodes (seeds included) are visited. For hops <= 2 this is
    scripts/m3a_headroom.py::expand exactly."""
    cap = int(setting["per_seed_cap"])
    hop1 = _first_occurrence_unique(capped_multi(seeds, families, cap))
    hops = int(setting["hops"])
    if hops == 1:
        return hop1
    cap2 = int(setting["per_frontier_cap"])
    limit = int(setting["visited_cap"])
    visited = np.union1d(seeds, hop1)
    out = [hop1]
    frontier = hop1
    for _ in range(2, hops + 1):
        budget = limit - int(visited.size)
        if budget <= 0 or frontier.size == 0:
            break
        cand = capped_multi(frontier, families, cap2)
        if cand.size == 0:
            break
        cand = cand[np.isin(cand, visited, invert=True)]
        fresh = _first_occurrence_unique(cand)[:budget]
        if fresh.size == 0:
            break
        out.append(fresh)
        visited = np.union1d(visited, fresh)
        frontier = fresh
    return np.concatenate(out)


def seeds_of(dense_row: np.ndarray, splade_row: np.ndarray) -> np.ndarray:
    """dense top-5 then splade top-5, first occurrence kept (the frozen seed rule)."""
    merged = np.concatenate((dense_row[:5], splade_row[:5])).astype(np.int64)
    return _first_occurrence_unique(merged)
