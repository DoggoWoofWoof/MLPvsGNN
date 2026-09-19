"""Universal-v2 arms (configs/universal_v2.yaml#arms, innovations 2-6).

The M3B input block and residual readout are imported unchanged, so every arm
returns the fixed base score exactly at step 0. The twins are the M3B QLSU
under a v2 name (u_mlp_v2) and QLSU with a query-conditioned scalar gate per
depth-basis block before the input linear (u_mlp_v2_mix): no operator over
the pool edges after the input block. The U-GNN puts one shared
query-conditioned relation-aware cell (innovations 2, 3, 4) between the two,
applied T = 3 times with a per-node scalar update gate, and optionally the
evidence-flow state (innovation 6) propagated over the typed STRUCT edges.

Relation text enters the GNN without dataset ids: pack_queries_v2 widens the
M3B edge_attr by K_REL float columns holding the rows of the served
relation-text bank (the concatenated RelationTable embeddings of the datasets
in the process; -1 = empty slot) for the first K_REL stored relations of each
structural message edge; the model holds the bank as a non-persistent buffer
and projects it once per forward. The M3B input block reads edge_attr[:, :3]
only, so it is unaffected; the M3B control arm is packed 8 wide by the pinned
pack_queries over the same cache.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from mp_retrieval.m3b_features import DIM, EDGE_ATTR, FAMILIES, MAX_SEEDS, _rank_within, pool_edges
from mp_retrieval.m3b_models import (
    N_EDGE_FEATURES, PROJECTION_DIM, InputBlock, PackedBatch, QLSU, ResidualReadout, UniversalGAT, parameter_count, segment_mean_rows,
    segment_zscore,
)
from mp_retrieval.m2d_stage1_arms import semantic_difference_column
from mp_retrieval.m3b_train import CarveData, DatasetContext
from mp_retrieval.universal_v2_features import BLOCK_OF, BLOCKS, CONTRACT_NAME, COLUMNS as V2_COLUMNS, N_COLUMNS as N_V2_COLUMNS

K_REL = 4                                            # relation-text slots per structural message edge
N_EDGE_FEATURES_V2 = N_EDGE_FEATURES + K_REL         # family one-hot 3 | weight, rel_compat, rel_mask, dir_fwd, dir_bwd | K_REL bank rows
STRUCT_FAMILY = FAMILIES.index("structural")
ATTR = {name: len(FAMILIES) + i for i, name in enumerate(EDGE_ATTR)}     # column of each M3B edge attribute
SLOT0 = N_EDGE_FEATURES
# innovation 6: p^0 = W_p [rrf_norm, dense_cos, splade_score_norm, is_seed]; rrf_norm is rrf / max rrf within the
# query, taken in-model from the base column; a declared column the v2 core lacks is replaced by the earliest
# surviving member of its M3B duplicate pair (resolve_evidence_columns), and the substitution is recorded
EVIDENCE_COLUMNS = ("rrf", "dense_cos", "splade_score_norm", "is_seed")
ARMS = ("u_mlp_v2", "u_mlp_v2_mix", "u_gnn_v2", "u_gnn_v2_ef", "u_gnn_v2_core78", "gat_universal_v1_trio")
GNN_CANDIDATES = ("u_gnn_v2", "u_gnn_v2_ef")
TWIN_CANDIDATES = ("u_mlp_v2", "u_mlp_v2_mix")


# ── the served relation-text bank ────────────────────────────────────────────


@dataclass
class RelationBank:
    """The RelationTable embeddings of the datasets of a process, concatenated
    in sorted dataset order; ``offsets`` gives the first row of each dataset."""

    embeddings: torch.Tensor      # (R, DIM) float32 unit rows
    offsets: dict
    sha256: str

    @property
    def n_rows(self) -> int:
        return int(self.embeddings.shape[0])


@dataclass
class DatasetContextV2(DatasetContext):
    """DatasetContext plus the dataset's first row in the relation bank (-1: no typed relations)."""

    rel_offset: int = -1


def build_relation_bank(contexts: dict) -> tuple[RelationBank, dict]:
    """The bank of the given DatasetContexts and the DatasetContextV2 of each
    (same stores, nodes and relation table; the offset added)."""
    rows, offsets, off = [], {}, 0
    for name in sorted(contexts):
        rt = contexts[name].rel_table
        if rt is None:
            continue
        emb = np.ascontiguousarray(np.asarray(rt.embeddings, dtype=np.float32))
        offsets[name] = off
        rows.append(emb)
        off += int(emb.shape[0])
    emb = np.concatenate(rows) if rows else np.zeros((0, DIM), dtype=np.float32)
    digest = hashlib.sha256(np.ascontiguousarray(emb).tobytes()).hexdigest()
    bank = RelationBank(torch.from_numpy(emb), offsets, digest)
    v2 = {name: DatasetContextV2(c.name, c.stores, c.nodes, c.rel_table, offsets.get(name, -1)) for name, c in contexts.items()}
    return bank, v2


def relation_slots(pair_id: np.ndarray, erel: np.ndarray, n_pairs: int, offset: int) -> tuple[np.ndarray, int]:
    """(n_pairs, K_REL) bank rows of the first K_REL entries of every pair in
    entry order, -1 where a pair has fewer; and the number of pairs truncated."""
    slots = np.full((n_pairs, K_REL), -1.0, dtype=np.float32)
    if pair_id.size == 0:
        return slots, 0
    rank = _rank_within(pair_id)
    keep = rank < K_REL
    slots[pair_id[keep], rank[keep]] = erel[keep].astype(np.float32) + float(offset)
    truncated = int(np.unique(pair_id[~keep]).size)
    return slots, truncated


def relation_slot_stats(pair_id: np.ndarray) -> dict:
    """How many structural pairs of one query carry more than K_REL relations."""
    if pair_id.size == 0:
        return {"pairs": 0, "truncated": 0, "entries": 0}
    counts = np.bincount(pair_id)
    return {"pairs": int(counts.size), "truncated": int((counts > K_REL).sum()), "entries": int(pair_id.size)}


# ── packing: the M3B pack under a v2 name, plus the relation slots ───────────


def pack_queries_v2(queries: list, context: DatasetContextV2, families: tuple = FAMILIES) -> PackedBatch:
    """mp_retrieval.m3b_train.pack_queries copied under a v2 name: the same
    batch (x, embeddings, seeds, gold, edges with the M3B attributes) with
    edge_attr widened to N_EDGE_FEATURES_V2 -- K_REL bank rows per structural
    message edge, -1 elsewhere. A test holds every shared field equal to the
    pinned pack_queries."""
    xs, embs, qembs, seedws, seed_nodes, golds, eis, eas = [], [], [], [], [], [], [], []
    ptr = [0]
    node_query = []
    offset = int(getattr(context, "rel_offset", -1))
    for qi, qd in enumerate(queries):
        pool = qd["pool"]
        n = pool.size
        off = ptr[-1]
        xs.append(qd["x"])
        embs.append(qd["emb"] if "emb" in qd else context.nodes.read(pool, dtype=np.float16))
        qembs.append(qd["qemb"])
        seedws.append(qd["seedw"])
        row = np.full(MAX_SEEDS, -1, dtype=np.int64)
        row[: qd["seeds"].size] = qd["seeds"][:MAX_SEEDS] + off
        seed_nodes.append(row)
        g = np.zeros(n, dtype=bool)
        g[qd["gold"]] = True
        golds.append(g)
        relcos = (context.rel_table.embeddings @ qd["qemb"]).astype(np.float32) if context.rel_table is not None else None
        edges, typed = pool_edges(pool, context.stores, relcos)
        for f_i, fam in enumerate(FAMILIES):
            if fam not in families:
                continue
            u, v, attr = edges[fam]
            if u.size == 0:
                continue
            eis.append(np.stack((u.astype(np.int64) + off, v.astype(np.int64) + off)))
            onehot = np.zeros((u.size, len(FAMILIES)), dtype=np.float32)
            onehot[:, f_i] = 1.0
            slots = np.full((u.size, K_REL), -1.0, dtype=np.float32)
            if fam == "structural" and typed is not None and typed[0].size and context.rel_table is not None and offset >= 0:
                ev, eu, erel, _ = typed
                new_pair = np.r_[True, (ev[1:] != ev[:-1]) | (eu[1:] != eu[:-1])]
                pair_id = np.cumsum(new_pair) - 1
                if pair_id[-1] + 1 != u.size or not (np.array_equal(eu[new_pair], u) and np.array_equal(ev[new_pair], v)):
                    raise RuntimeError("typed entries do not run in the order of the structural message edges")
                slots, _ = relation_slots(pair_id, erel, u.size, offset)
            eas.append(np.concatenate((onehot, attr, slots), axis=1))
        node_query.append(np.full(n, qi, dtype=np.int64))
        ptr.append(off + n)
    edge_index = np.concatenate(eis, axis=1) if eis else np.empty((2, 0), dtype=np.int64)
    edge_attr = np.concatenate(eas, axis=0) if eas else np.empty((0, N_EDGE_FEATURES_V2), dtype=np.float32)
    return PackedBatch(
        x=torch.from_numpy(np.concatenate(xs)).to(torch.float32), qptr=torch.tensor(ptr, dtype=torch.long), node_query=torch.from_numpy(np.concatenate(node_query)),
        emb=torch.from_numpy(np.concatenate(embs)), qemb=torch.from_numpy(np.stack(qembs)), seedw=torch.from_numpy(np.concatenate(seedws)),
        seed_nodes=torch.from_numpy(np.stack(seed_nodes)), edge_index=torch.from_numpy(edge_index), edge_attr=torch.from_numpy(edge_attr),
        gold=torch.from_numpy(np.concatenate(golds)),
    )


class CarveDataV2(CarveData):
    """A v2 cache (UNIVERSAL_V2_FEATURE_CONTRACT, N_V2_COLUMNS stored) read as
    CarveData reads an M3B cache; ``columns`` are indices into the v2 layout;
    pack() builds the widened batch. A cache of another contract refuses."""

    def __init__(self, cache_dir: Path, context: DatasetContextV2, columns: np.ndarray | None = None):
        d = Path(cache_dir)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta.get("feature_contract") != CONTRACT_NAME or int(meta.get("n_columns", -1)) != N_V2_COLUMNS:
            raise ValueError(f"{d}: not a {CONTRACT_NAME} cache ({meta.get('feature_contract')}, {meta.get('n_columns')} columns)")
        stored = meta.get("columns_stored")
        if columns is None:
            if stored is not None:
                raise ValueError(f"{d}: trimmed to {len(stored)} columns; the full layout is not available")
            mapped = None
        else:
            requested = np.asarray(columns, dtype=np.int64)
            if stored is None:
                mapped = requested
            else:
                position = {name: k for k, name in enumerate(stored)}
                missing = [V2_COLUMNS[c] for c in requested if V2_COLUMNS[c] not in position]
                if missing:
                    raise ValueError(f"{d}: trimmed cache lacks {missing}")
                mapped = np.asarray([position[V2_COLUMNS[c]] for c in requested], dtype=np.int64)
        self._init_arrays(d, context, meta, mapped)

    def _init_arrays(self, d: Path, context, meta: dict, columns) -> None:
        self.dir = d
        self.context = context
        self.meta = meta
        self.columns = columns
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

    def pack(self, indices: np.ndarray, families: tuple = FAMILIES) -> PackedBatch:
        return pack_queries_v2([self.query(int(i)) for i in indices], self.context, families)


def batch_view(batch: PackedBatch, columns, edge_width: int = N_EDGE_FEATURES) -> PackedBatch:
    """The batch as a model of another layout reads it: a column subset of x
    (indices into the packed layout) and the first ``edge_width`` edge columns.
    Views share storage with the packed batch."""
    cols = torch.as_tensor(np.asarray(columns, dtype=np.int64))
    return PackedBatch(x=batch.x[:, cols], qptr=batch.qptr, node_query=batch.node_query, emb=batch.emb, qemb=batch.qemb, seedw=batch.seedw,
                       seed_nodes=batch.seed_nodes, edge_index=batch.edge_index, edge_attr=batch.edge_attr[:, :edge_width], gold=batch.gold)


# ── segment attention ────────────────────────────────────────────────────────


def segment_softmax(score: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of ``score`` (E, H) over the entries that share ``index`` (E,) in 0..n-1."""
    H = score.shape[1]
    smax = torch.full((n, H), -float("inf"), dtype=score.dtype, device=score.device)
    smax = smax.scatter_reduce(0, index.unsqueeze(1).expand(-1, H), score, reduce="amax", include_self=True)
    ex = (score - smax[index]).exp()
    denom = torch.zeros(n, H, dtype=score.dtype, device=score.device).index_add_(0, index, ex)
    return ex / denom[index].clamp_min(1e-30)


# ── innovation 2: the semantic relation encoder ──────────────────────────────


class RelationEncoder(nn.Module):
    """r_e = W_r e_text * rel_mask + E_family[family] + E_direction . (dir_fwd, dir_bwd)
    + w_weight * weight + w_compat * rel_compat * rel_mask; e_text is the mean
    of the served relation-text rows in the slots of the edge (a pair with
    several stored relations averages them). No relation id, no dataset id:
    the same parameters read MetaQA predicates, Freebase predicates,
    hyperlinks, NER co-mention and KNN edges. The self-loop of the cell has
    its own descriptor."""

    def __init__(self, d_r: int = 32):
        super().__init__()
        self.d_r = int(d_r)
        self.text = nn.Linear(DIM, d_r, bias=False)
        self.family = nn.Parameter(torch.randn(len(FAMILIES), d_r) * 0.05)
        self.direction = nn.Parameter(torch.randn(2, d_r) * 0.05)
        self.weight = nn.Parameter(torch.randn(d_r) * 0.05)
        self.compat = nn.Parameter(torch.randn(d_r) * 0.05)
        self.self_loop = nn.Parameter(torch.randn(d_r) * 0.05)

    def project_bank(self, bank: torch.Tensor) -> torch.Tensor:
        return self.text(bank) if bank.shape[0] else torch.zeros(0, self.d_r, dtype=self.text.weight.dtype, device=self.text.weight.device)

    def forward(self, edge_attr: torch.Tensor, bank_proj: torch.Tensor) -> torch.Tensor:
        E = edge_attr.shape[0]
        fam = edge_attr[:, : len(FAMILIES)]
        rel_mask = edge_attr[:, ATTR["rel_mask"]].unsqueeze(1)
        r = fam @ self.family
        r = r + edge_attr[:, ATTR["dir_fwd"]].unsqueeze(1) * self.direction[0] + edge_attr[:, ATTR["dir_bwd"]].unsqueeze(1) * self.direction[1]
        r = r + edge_attr[:, ATTR["weight"]].unsqueeze(1) * self.weight + (edge_attr[:, ATTR["rel_compat"]].unsqueeze(1) * rel_mask) * self.compat
        if E and edge_attr.shape[1] > SLOT0 and bank_proj.shape[0]:
            slots = edge_attr[:, SLOT0: SLOT0 + K_REL].long()
            valid = slots >= 0
            if bool(valid.any()):
                gathered = bank_proj[slots.clamp_min(0)] * valid.unsqueeze(-1).to(bank_proj.dtype)
                text = gathered.sum(1) / valid.sum(1).clamp_min(1).unsqueeze(1).to(bank_proj.dtype)
                r = r + text * rel_mask
        return r


# ── innovations 3 and 4: the shared query-conditioned relation-aware cell ────


class QueryRelationCell(nn.Module):
    """score_uv = a_k . leakyrelu(W_src h_u + W_dst h_v + (W_q q) * (W_r1 r_uv) + W_r2 r_uv),
    softmax over the in-edges of v plus a self-loop, ``heads`` heads;
    m_v = sum_u alpha_uv (W_mh h_u + W_mr r_uv); g_v = sigmoid(W_g [h_v | m_v | q]);
    h_v <- (1 - g_v) h_v + g_v LayerNorm(h_v + m_v). One instance is applied at
    every step, so its parameters are shared across steps by construction."""

    def __init__(self, hidden: int, d_r: int, heads: int = 4, q_dim: int = PROJECTION_DIM, dropout: float = 0.2):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden, self.heads = int(hidden), int(heads)
        self.src = nn.Linear(hidden, hidden)
        self.dst = nn.Linear(hidden, hidden, bias=False)
        self.query = nn.Linear(q_dim, hidden)
        self.rel1 = nn.Linear(d_r, hidden)
        self.rel2 = nn.Linear(d_r, hidden, bias=False)
        self.att = nn.Parameter(torch.randn(heads, hidden // heads) * (1.0 / (hidden // heads) ** 0.5))
        self.msg_h = nn.Linear(hidden, hidden)
        self.msg_r = nn.Linear(d_r, hidden, bias=False)
        self.gate = nn.Linear(2 * hidden + q_dim, 1)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, q_state: torch.Tensor, edge_index: torch.Tensor, r: torch.Tensor, self_r: torch.Tensor):
        N, H, K = h.shape[0], self.hidden, self.heads
        loops = torch.arange(N, dtype=torch.long, device=h.device)
        u = torch.cat([edge_index[0], loops])
        v = torch.cat([edge_index[1], loops])
        r_all = torch.cat([r, self_r.unsqueeze(0).expand(N, -1)])
        src, dst, qh = self.src(h), self.dst(h), self.query(q_state)
        e = Fn.leaky_relu(src[u] + dst[v] + qh[v] * self.rel1(r_all) + self.rel2(r_all), 0.2)
        score = (e.view(-1, K, H // K) * self.att).sum(-1)
        alpha = segment_softmax(score, v, N)
        msg = (self.msg_h(h)[u] + self.msg_r(r_all)).view(-1, K, H // K) * alpha.unsqueeze(-1)
        m = torch.zeros(N, K, H // K, dtype=h.dtype, device=h.device).index_add_(0, v, msg).view(N, H)
        m = self.dropout(m)
        g = torch.sigmoid(self.gate(torch.cat([h, m, q_state], dim=-1)))
        return (1.0 - g) * h + g * self.norm(h + m), g.squeeze(-1)


# ── innovation 6: the evidence-flow state ────────────────────────────────────


class EvidenceFlow(nn.Module):
    """p^0 = W_p [rrf_norm, dense_cos, splade_score_norm, is_seed];
    p_v <- (1 - g2_v) p_v + g2_v sum_u beta_uv W_pp p_u over the typed STRUCT
    in-edges of v, beta_uv = softmax_u(a2 . ((W_q2 q) * (W_r3 r_uv))),
    g2_v = sigmoid(W_g2 [p_v | m_v | q]); a node without a typed in-edge keeps
    its state (its gate is reported as 0). Nothing here reads h: evidence
    flows by query-relation compatibility, not by node semantics."""

    def __init__(self, d_p: int = 8, d_r: int = 32, q_dim: int = PROJECTION_DIM, d_att: int = 32, n_evidence: int = len(EVIDENCE_COLUMNS)):
        super().__init__()
        self.d_p = int(d_p)
        self.init = nn.Linear(n_evidence, d_p)
        self.prop = nn.Linear(d_p, d_p, bias=False)
        self.query = nn.Linear(q_dim, d_att)
        self.rel = nn.Linear(d_r, d_att, bias=False)
        self.att = nn.Parameter(torch.randn(d_att) * (1.0 / d_att ** 0.5))
        self.gate = nn.Linear(2 * d_p + q_dim, 1)

    def initial(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.init(evidence)

    def step(self, p: torch.Tensor, q_state: torch.Tensor, edge_index: torch.Tensor, r: torch.Tensor):
        N = p.shape[0]
        u, v = edge_index[0], edge_index[1]
        score = ((self.query(q_state)[v] * self.rel(r)) @ self.att).unsqueeze(1)
        beta = segment_softmax(score, v, N)
        m = torch.zeros(N, self.d_p, dtype=p.dtype, device=p.device).index_add_(0, v, beta * self.prop(p)[u])
        ones = torch.ones(v.numel(), dtype=p.dtype, device=p.device)
        has_in = torch.zeros(N, dtype=p.dtype, device=p.device).index_add_(0, v, ones) > 0
        g = torch.sigmoid(self.gate(torch.cat([p, m, q_state], dim=-1)))
        p_new = torch.where(has_in.unsqueeze(1), (1.0 - g) * p + g * m, p)
        return p_new, torch.where(has_in, g.squeeze(-1), torch.zeros_like(g.squeeze(-1)))


def evidence_features(x: torch.Tensor, node_query: torch.Tensor, n_queries: int, index: list) -> torch.Tensor:
    """[rrf / max rrf within the query, dense_cos, splade_score_norm (or its
    substitute), is_seed] from the input columns of the model; every one is a
    retrieval-time column of the contract (no gold, no split, no dataset)."""
    cols = x[:, torch.as_tensor(list(index), dtype=torch.long, device=x.device)]
    rrf = cols[:, 0]
    qmax = torch.full((n_queries,), -float("inf"), dtype=x.dtype, device=x.device).scatter_reduce(0, node_query, rrf, reduce="amax", include_self=True)
    rrf_norm = rrf / qmax[node_query].clamp_min(1e-12)
    return torch.cat([rrf_norm.unsqueeze(1), cols[:, 1:]], dim=1)


# ── u_mlp_v2_mix: the query-conditioned block gate before the input linear ───


class GatedInputBlock(InputBlock):
    """The M3B InputBlock with one addition before its linear: every depth-basis
    block B_{f,t} (both the raw and the z-scored copy of its columns) is
    multiplied by sigmoid(w_{f,t} . q_proj + b_{f,t}), q_proj the M3B 64-wide
    projected query. Zero-initialised gates (0.5 everywhere); with a large bias
    the block equals the parent exactly (test). forward() is the parent's
    code copied with the gate applied; nothing else changes."""

    def __init__(self, n_scalars: int, hidden: int, base_index: int, block_index, dropout: float = 0.2):
        super().__init__(n_scalars, hidden, base_index, dropout)
        block_index = torch.as_tensor(np.asarray(block_index, dtype=np.int64))
        if block_index.numel() != n_scalars:
            raise ValueError("block_index must give one block id (or -1) per scalar column")
        self.register_buffer("block_index", block_index)
        self.n_blocks = int(block_index.max().item()) + 1 if block_index.numel() and int(block_index.max()) >= 0 else 0
        self.gate_weight = nn.Parameter(torch.zeros(max(self.n_blocks, 1), PROJECTION_DIM))
        self.gate_bias = nn.Parameter(torch.zeros(max(self.n_blocks, 1)))
        self.last_block_gates = None

    def column_gates(self, q_state: torch.Tensor, node_query: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(q_state @ self.gate_weight.t() + self.gate_bias)          # (B, n_blocks)
        self.last_block_gates = gates.detach()
        ones = torch.ones(gates.shape[0], 1, dtype=gates.dtype, device=gates.device)
        table = torch.cat([ones, gates], dim=1)                                        # column -1 -> 1
        return table[:, self.block_index + 1][node_query]                              # (N, n_scalars)

    def forward(self, batch: PackedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        B = batch.n_queries
        N = batch.x.shape[0]
        P = self.semantic.projection_dim
        z = segment_zscore(batch.x, batch.node_query, B)
        base_z = z[:, self.base_index]
        pre_q = self.semantic.query_projection(batch.qemb)
        raw_q_query = Fn.gelu(pre_q)
        q_query = Fn.normalize(raw_q_query, dim=-1)
        gate_cols = self.column_gates(q_query, batch.node_query)
        raw_q = raw_q_query[batch.node_query]
        q_state = q_query[batch.node_query]
        emb = batch.emb if batch.emb.dtype == torch.float32 else batch.emb.to(torch.float32)
        pre_n = self.semantic.node_projection(emb)
        raw_n = Fn.gelu(pre_n)
        n_state = Fn.normalize(raw_n, dim=-1)
        semantic = torch.cat([
            q_state, n_state, q_state * n_state, (q_state - n_state).abs(),
            (q_state * n_state).sum(-1, keepdim=True), (raw_q * raw_n).sum(-1, keepdim=True) / raw_n.shape[-1] ** 0.5,
        ], dim=-1)
        difference = semantic_difference_column(batch.qemb[batch.node_query], emb, self.difference_weight).unsqueeze(1)
        del emb
        parts = [batch.x * gate_cols, z * gate_cols, semantic, difference]
        has_edges = batch.edge_attr.shape[0] > 0
        fam_of_edge = batch.edge_attr[:, : len(FAMILIES)].argmax(dim=1) if has_edges else None
        for f_i in range(len(FAMILIES)):
            if fam_of_edge is None:
                parts.append(torch.zeros(N, P, dtype=batch.x.dtype, device=batch.x.device))
                continue
            sel = fam_of_edge == f_i
            u, v = batch.edge_index[0, sel], batch.edge_index[1, sel]
            proto_pre, count = segment_mean_rows(pre_n[u], v, N)
            p_state = Fn.normalize(Fn.gelu(proto_pre), dim=-1)
            p_state = torch.where((count > 0).unsqueeze(1), p_state, torch.zeros_like(p_state))
            parts.append(p_state * q_state)
        seed_pre = torch.zeros(B, MAX_SEEDS, P, dtype=pre_n.dtype, device=pre_n.device)
        valid = batch.seed_nodes >= 0
        seed_pre[valid] = pre_n[batch.seed_nodes[valid]]
        w = batch.seedw
        reach_pre = torch.einsum("ns,nsp->np", w, seed_pre[batch.node_query]) / w.sum(1, keepdim=True).clamp_min(1e-12)
        r_state = Fn.normalize(Fn.gelu(reach_pre), dim=-1)
        r_state = torch.where((w.sum(1) > 0).unsqueeze(1), r_state, torch.zeros_like(r_state))
        parts.append(r_state * n_state)
        h = self.dropout(Fn.gelu(self.linear(torch.cat(parts, dim=-1))))
        return h, base_z


# ── the twins ────────────────────────────────────────────────────────────────


class UMLPv2(QLSU):
    """u_mlp_v2: the M3B QLSU, unchanged, over the v2 core. No edges beyond the input block."""

    arm = "u_mlp_v2"


class UMLPv2Mix(QLSU):
    """u_mlp_v2_mix: QLSU with the GatedInputBlock; body and readout as QLSU."""

    arm = "u_mlp_v2_mix"

    def __init__(self, n_scalars: int, hidden: int, base_index: int, block_index, dropout: float = 0.2):
        super().__init__(n_scalars, hidden, base_index, dropout)
        self.input = GatedInputBlock(n_scalars, hidden, base_index, block_index, dropout)

    @property
    def last_block_gates(self):
        return self.input.last_block_gates


# ── the U-GNN ────────────────────────────────────────────────────────────────


class UGNNv2(nn.Module):
    """u_gnn_v2 / u_gnn_v2_ef: input block -> h^0; the shared QueryRelationCell
    applied ``steps`` times over the FULL message edges (family_mask) with the
    scalar update gate; with ``evidence_index`` the EvidenceFlow state over the
    typed STRUCT edges beside it; residual readout on h^T (or [h^T | p^T]).
    ``message_passing=False`` removes every edge before the cell (the self-loop
    is the only message) -- the test of the propagation boundary, not an arm.
    The relation bank is a non-persistent buffer: served data, not a weight."""

    def __init__(self, n_scalars: int, hidden: int, base_index: int, dropout: float = 0.2, steps: int = 3, heads: int = 4,
                 d_r: int = 32, evidence_index=None, d_p: int = 8, message_passing: bool = True, families: tuple = FAMILIES):
        super().__init__()
        self.arm = "u_gnn_v2_ef" if evidence_index is not None else "u_gnn_v2"
        self.hidden, self.steps, self.message_passing = int(hidden), int(steps), bool(message_passing)
        self.family_mask = torch.tensor([f in families for f in FAMILIES])
        self.input = InputBlock(n_scalars, hidden, base_index, dropout)
        self.relations = RelationEncoder(d_r)
        self.cell = QueryRelationCell(hidden, d_r, heads, PROJECTION_DIM, dropout)
        self.evidence_index = list(int(i) for i in evidence_index) if evidence_index is not None else None
        self.evidence = EvidenceFlow(d_p, d_r, PROJECTION_DIM, n_evidence=len(self.evidence_index)) if evidence_index is not None else None
        self.readout = ResidualReadout(hidden + (int(d_p) if evidence_index is not None else 0))
        self.register_buffer("relation_bank", torch.zeros(0, DIM), persistent=False)
        self.last_gates = None
        self.last_gates2 = None
        self.last_base_z = None
        self.last_correction = None

    def set_relation_bank(self, bank) -> None:
        emb = bank.embeddings if isinstance(bank, RelationBank) else bank
        self.relation_bank = torch.as_tensor(emb, dtype=torch.float32).to(self.readout.out.weight.device)

    def edges(self, batch: PackedBatch, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.message_passing and batch.edge_attr.shape[0]:
            fam = batch.edge_attr[:, : len(FAMILIES)].argmax(dim=1)
            keep = self.family_mask.to(fam.device)[fam]
            return batch.edge_index[:, keep], batch.edge_attr[keep]
        return (torch.empty(2, 0, dtype=torch.long, device=h.device), torch.empty(0, N_EDGE_FEATURES_V2, dtype=h.dtype, device=h.device))

    def forward(self, batch: PackedBatch) -> torch.Tensor:
        h, base_z = self.input(batch)
        B, N = batch.n_queries, h.shape[0]
        q_state = Fn.normalize(Fn.gelu(self.input.semantic.query_projection(batch.qemb)), dim=-1)[batch.node_query]
        edge_index, edge_attr = self.edges(batch, h)
        bank_proj = self.relations.project_bank(self.relation_bank.to(h.dtype))
        r = self.relations(edge_attr, bank_proj)
        gates = []
        for _ in range(self.steps):
            h, g = self.cell(h, q_state, edge_index, r, self.relations.self_loop)
            gates.append(g)
        self.last_gates = torch.stack(gates).detach()
        out = h
        if self.evidence is not None:
            p = self.evidence.initial(evidence_features(batch.x, batch.node_query, B, self.evidence_index))
            if edge_attr.shape[0]:
                typed = (edge_attr[:, STRUCT_FAMILY] > 0.5) & (edge_attr[:, ATTR["rel_mask"]] > 0.5)
                t_index, t_r = edge_index[:, typed], r[typed]
            else:
                t_index, t_r = edge_index, r
            gates2 = []
            for _ in range(self.steps):
                p, g2 = self.evidence.step(p, q_state, t_index, t_r)
                gates2.append(g2)
            self.last_gates2 = torch.stack(gates2).detach()
            out = torch.cat([h, p], dim=-1)
        score = self.readout(out, base_z)
        self.last_base_z = base_z.detach()
        self.last_correction = (score - self.readout.base_weight * base_z).detach()
        return score


# ── arms from the declaration ────────────────────────────────────────────────

PARAMETER_BUDGET = {"gnn": 450_000, "twin": 350_000}
FAMILY_OF_ARM = {"u_mlp_v2": "twin", "u_mlp_v2_mix": "twin", "u_gnn_v2": "gnn", "u_gnn_v2_ef": "gnn", "u_gnn_v2_core78": "gnn",
                 "gat_universal_v1_trio": "gnn"}


def resolve_evidence_columns(core_columns: list, m3b_duplicate_pairs) -> tuple[list, dict]:
    """The evidence init columns inside a core contract: a declared column the
    core lacks is replaced by the earliest surviving member of its duplicate
    pair (the M3B screen record, fit-carve statistics only); the substitution
    is returned for the fit record. Refuses when no substitute exists."""
    present = set(core_columns)
    partner: dict = {}
    for pair in m3b_duplicate_pairs:                 # feature_screen.json rows {earlier, later, abs_spearman}
        a, b = (pair["earlier"], pair["later"]) if isinstance(pair, dict) else (pair[0], pair[1])
        partner.setdefault(b, []).append(a)
        partner.setdefault(a, []).append(b)
    names, substitutions = [], {}
    for name in EVIDENCE_COLUMNS:
        if name in present:
            names.append(name)
            continue
        candidates = [c for c in partner.get(name, []) if c in present]
        if not candidates:
            raise ValueError(f"evidence column {name} is not in the core and has no surviving duplicate partner")
        chosen = min(candidates, key=lambda c: core_columns.index(c))
        substitutions[name] = chosen
        names.append(chosen)
    return names, substitutions


def block_index_of(core_columns: list) -> np.ndarray:
    """One block id per core column (-1 for the M3B columns), blocks in BLOCKS order."""
    order = {b: i for i, b in enumerate(BLOCKS)}
    return np.asarray([order[BLOCK_OF[c]] if c in BLOCK_OF else -1 for c in core_columns], dtype=np.int64)


def build_arm(arm: str, inputs: dict, hidden: int = 128, dropout: float = 0.2, heads: int = 4, steps: int = 3, d_r: int = 32,
              d_p: int = 8, selected_gnn: str | None = None) -> nn.Module:
    """The model of an arm. ``inputs`` (scripts/universal_v2_run.py model_inputs)
    carries: columns (the core column names), base_local (rrf position),
    evidence_local (positions of the four evidence columns), core78_columns,
    core78_base_local and core78_evidence_local (the M3B core inside the same
    cache). No dataset id enters any arm."""
    cols = list(inputs["columns"])
    if arm == "u_mlp_v2":
        return UMLPv2(len(cols), hidden, int(inputs["base_local"]), dropout)
    if arm == "u_mlp_v2_mix":
        return UMLPv2Mix(len(cols), hidden, int(inputs["base_local"]), block_index_of(cols), dropout)
    if arm == "u_gnn_v2":
        return UGNNv2(len(cols), hidden, int(inputs["base_local"]), dropout, steps, heads, d_r, None, d_p)
    if arm == "u_gnn_v2_ef":
        return UGNNv2(len(cols), hidden, int(inputs["base_local"]), dropout, steps, heads, d_r, list(inputs["evidence_local"]), d_p)
    if arm == "u_gnn_v2_core78":
        if selected_gnn not in GNN_CANDIDATES:
            raise ValueError("u_gnn_v2_core78 takes the selected GNN architecture: selection.json must name it")
        c78 = list(inputs["core78_columns"])
        evidence = list(inputs["core78_evidence_local"]) if selected_gnn == "u_gnn_v2_ef" else None
        model = UGNNv2(len(c78), hidden, int(inputs["core78_base_local"]), dropout, steps, heads, d_r, evidence, d_p)
        model.arm = "u_gnn_v2_core78"
        return model
    if arm == "gat_universal_v1_trio":
        c78 = list(inputs["core78_columns"])
        model = UniversalGAT(len(c78), hidden, int(inputs["core78_base_local"]), layers=2, heads=heads, dropout=dropout, message_passing=True)
        model.arm = "gat_universal_v1_trio"
        return model
    raise ValueError(f"unknown arm {arm}")


def check_parameter_budget(arm: str, model: nn.Module) -> int:
    count = parameter_count(model)
    budget = PARAMETER_BUDGET[FAMILY_OF_ARM[arm]]
    if count > budget:
        raise ValueError(f"{arm}: {count:,} parameters over the declared budget {budget:,}")
    return count
