"""Design look (untracked; not a result and not filed): lean_mlp4's AW block made to read any graph's edge labels, with
a text relevance that needs no fitted alignment, an on-path token loss, and an EM edge typer whose label supervision is
annealed away, so the walks are weighted by a query-conditioned ranking of the edges.

lean_mlp4 (AW, its labels, codes and walk entries), lean_mlp5 (its config form), lean_mlp3 (the carves, store and
blocks), lean_mlp2 and lean_mlp are imported and called unchanged. This file adds:

KB labels. metaqa and webqsp label their structural edges with their own relation names. A relation's rank is its count
among the graph's stored structural edges (ties by relation id; rank >= K_LAB reads OTHER), and its text vector is
m3b's (outputs/m3b/relations/<ds>_rel_embeddings.npy: gte-Qwen2-1.5B in document mode, the anchor phrases' space). Its
vocabulary must equal the served manifest's relation_vocabulary. A pair that stores several relations reads its most
frequent one. Passage graphs keep lean_mlp4's anchor labels (lean_mlp4.Labels); squad has none (OTHER).

Arms (--arms; one fit each on the same rows, batches and seed; the block set is --set, plus AW for every arm but ctl):
  ctl   the set without AW
  aw    lean_mlp4's AW, l = <W_q q, P z(a) + D[d]> / sqrt(64) + b[d]; with no extra term it is lean_mlp4's model and
        fit bit for bit (the self-test checks both)
  awc   aw + kappa[d] cos(q, t(a)), t(a) the label's own text vector (0 for OTHER and for a dropped label): an unseen
        relation's relevance needs no learned alignment. kappa starts at 0, so awc starts as aw.
  aux   awc plus a token loss. A row's tokens are its walks' distinct (direction, label) pairs; a token is on-path when an
        edge carrying it lies on a seed walk of at most two edges that ends at a gold node (training only; no score
        reads it). Each token's AW logit is fitted to that, on- and off-path tokens at half the batch each.
  em    awc plus eta[d] log r(q, e) in the logit. r is an EM edge typer's query relevance:
          prior     pi(z | a, d) = softmax(f2 relu(f1 [z(a) has, onehot(d), has]))   (the label's text code, direction)
          relevance rho(q, z) = <LN(W_t q), E[z]> / sqrt(64) + c[z];  r = sum_z pi(z) sigmoid(rho(q, z))
          E-step    q_u(z) ~ pi(z) theta_g(a | z)^lambda_t Bern(o_u | sigmoid(rho(q, z)))^mu      (no gradient)
          M-step    theta_g: per training graph, a decayed count of q over the tokens' true labels (closed form);
                    loss omega (mean_u -sum_z q_u log pi + sum_u w_u -sum_z q_u log Bern(o_u | sigmoid(rho)))
          lambda_t = lambda0 max(0, 1 - ep / E_an): the labels pull each label's tokens to one type first, then are
          annealed away while the E and M steps go on, so the types end as what the on-path evidence and the text
          support. The prior's text is hidden with probability --aw-drop per token in the M step (the emission keeps
          the true label). The typer is fitted by EM alone: pi and rho reach the score detached.
  em0   em with lambda0 = 0 (no label emission)
Every fit keeps best (select-chosen), swa (the mean of the weights after epochs swa_from ..) and last, as lean_mlp5.

Non-MP, as lean_mlp4: learned weights of the query and an edge's own label and direction, applied to fixed seed-walk
counts. The typer reads an edge's label text and direction only; no node state, neighbour embedding or score is read.
At serving pi is an index-time table per (label, direction), and rho and cos are one query-side product each.

Reads: per model and --read carve, the ID read, the NR read (every label OTHER: no text, no cos), paired differences
against ctl (and, for aux/em/em0, against awc) at the same kind, and the edge ranking: per row, the AUC of each token
score against on-path over the row's tokens (rows with both), for the AW logit and, for em arms, the typer's r, its
query-blind sum_z pi sigmoid(c[z]) and r with every label hidden.

    python outputs/mp_unified/lean_host7.py lean_mlp7 --store pca256 --train 2wiki=x4,hotpotqa=fit,metaqa=fit \
        --select 2wiki=select,hotpotqa=select,metaqa=select --read 2wiki=x1,hotpotqa=x1,squad=x1,metaqa=x1f,webqsp=selectf \
        --arms ctl,aw,awc --threads 2 --save-models outputs/mp_unified/lean/l7-j3a_models.pt --out outputs/mp_unified/lean/l7-j3a.json
    python outputs/mp_unified/lean_mlp7.py --selftest
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import lean_mlp as LM  # noqa: E402
import lean_mlp2 as L2  # noqa: E402
import lean_mlp3 as L3  # noqa: E402
import lean_mlp4 as L4  # noqa: E402
import lean_mlp5 as L5  # noqa: E402

K_LAB, N_DIR, AW_W = L4.K_LAB, L4.N_DIR, L4.AW_W
KB_RELS = ROOT / "outputs" / "m3b" / "relations"
KBS = ("metaqa", "webqsp")
ARMS = ("ctl", "aw", "awc", "aux", "em", "em0")
EM_ARMS = ("em", "em0")
log = LM.log
_BATCH_OF = LM.batch_of


# ── labels ───────────────────────────────────────────────────────────────────


class KBLabels(L4.Labels):
    """A KB's relation rank per stored structural edge by global (src, dst), the pair's most frequent relation when it
    stores several, and its relation text table in rank order (rows past the vocabulary zero). lookup is
    lean_mlp4.Labels'."""

    def __init__(self, ds, served, canonical):
        self.ds = ds
        emb_p, voc_p = KB_RELS / f"{ds}_rel_embeddings.npy", KB_RELS / f"{ds}_rel_vocab.json"
        self.ok = ds in KBS and emb_p.exists() and voc_p.exists()
        self.seconds = 0.0
        if not self.ok:
            log(f"{ds}: no relation table; every AW label reads OTHER")
            return
        t = time.time()
        self.n_nodes = int(canonical.Dataset(ds, root=str(served)).n_nodes)
        gm = json.loads((Path(served) / ds / "graph" / "GRAPH_MANIFEST.json").read_text(encoding="utf-8"))
        st = gm["families"]["structural"]
        vocab = [str(r) for r in st["relation_vocabulary"]]
        voc = json.loads(voc_p.read_text(encoding="utf-8"))
        if vocab != voc["vocab"] or hashlib.sha256("\n".join(vocab).encode("utf-8")).hexdigest() != voc["sha256"]:
            raise SystemExit(f"{ds}: the served relation vocabulary is not m3b's")
        self.manifest_entry = {k: st.get(k) for k in ("n_edges", "directed", "npz_sha256", "directed_as_frozen")}
        with np.load(Path(served) / ds / "graph" / "structural.npz") as z:
            src, dst, rel = z["src"].astype(np.int64), z["dst"].astype(np.int64), z["rel"].astype(np.int64)
        if rel.size and (rel.min() < 0 or rel.max() >= len(vocab)):
            raise SystemExit(f"{ds}: relation ids outside the vocabulary")
        cnt = np.bincount(rel, minlength=len(vocab))
        by = np.lexsort((np.arange(cnt.size), -cnt))          # the most frequent first, ties by relation id
        rank_of = np.empty(cnt.size, np.int64)
        rank_of[by] = np.arange(cnt.size)
        self.vocab, self.rank_of = vocab, rank_of
        r_e = np.minimum(rank_of[rel], K_LAB)
        key = src * self.n_nodes + dst
        del src, dst, rel
        order = np.lexsort((r_e, key))                         # by pair, its most frequent relation first
        self.skey = key[order]
        self.rank = r_e[order].astype(np.int16)
        del key, order, r_e
        emb = np.load(emb_p).astype(np.float32)
        if emb.shape[0] != len(vocab):
            raise SystemExit(f"{ds}: {emb.shape[0]} relation vectors for {len(vocab)} relations")
        self.phrase = np.zeros((K_LAB, emb.shape[1]), np.float32)
        top = by[:K_LAB]
        self.phrase[:top.size] = emb[top]
        self.n_real = int(top.size)
        self.phrase_sha256 = hashlib.sha256(emb_p.read_bytes()).hexdigest()
        self.vocab_sha256 = voc["sha256"]
        u, c = np.unique(self.skey, return_counts=True)
        self.census = {"edges": int(self.skey.size), "pairs": int(u.size), "pairs_with_several": int((c > 1).sum()),
                       "relations": len(vocab), "labelled_relations": self.n_real,
                       "edges_other": int((self.rank >= K_LAB).sum()), "top": [vocab[i] for i in by[:5]]}
        self.seconds = time.time() - t
        log(f"{ds}: relation labels over {self.skey.size} structural edges, {len(vocab)} relations ({self.seconds:.0f}s); "
            f"{self.census}")


def labels_of(ds, served, canonical):
    return KBLabels(ds, served, canonical) if ds in KBS else L4.Labels(ds, served, canonical)


def n_real(labels):
    return 0 if labels is None or not labels.ok else int(getattr(labels, "n_real", K_LAB))


def text_table(labels):
    """Each label's unit text vector, row K_LAB (OTHER) and rows past a graph's labels zero."""
    T = np.zeros((K_LAB + 1, 1536), np.float32)
    if labels is not None and labels.ok:
        P = labels.phrase[:K_LAB].astype(np.float32)
        nrm = np.linalg.norm(P, axis=1, keepdims=True)
        T[:K_LAB] = np.where(nrm > 0, P / np.maximum(nrm, 1e-12), 0.0)
    return T


# ── walks and tokens ─────────────────────────────────────────────────────────


def walk_entries7(n, pool, eu, ev, efam, efwd, efwd_b, seeds, buckets, labels, checks, struct_only=False):
    """lean_mlp4.walk_entries (the same entries in the same order; the self-test checks it), plus each two-hop walk's
    parent, the index of its first edge among the one-hop walks. With struct_only, ner and knn edges are left out of
    both hops."""
    if struct_only:
        m = efam == 0
        eu, ev, efam, efwd, efwd_b = eu[m], ev[m], efam[m], efwd[m], efwd_b[m]
    u, v = eu.astype(np.int64), ev.astype(np.int64)
    fwd, bwd = efwd.astype(bool), efwd_b.astype(bool)
    d = L4.famdir(efam, fwd, bwd)
    lab = np.full(u.size, K_LAB, np.int32)
    t_lab = 0.0
    st = efam == 0
    if labels is not None and labels.ok and st.any():
        t = time.perf_counter()
        gu, gv = pool[u[st]].astype(np.int64), pool[v[st]].astype(np.int64)
        lf, lb = labels.lookup(gu, gv), labels.lookup(gv, gu)
        t_lab = time.perf_counter() - t
        f_s, b_s = fwd[st], bwd[st]
        checks["fwd_flag"] += int(f_s.sum())
        checks["fwd_found"] += int((f_s & (lf >= 0)).sum())
        checks["bwd_flag"] += int(b_s.sum())
        checks["bwd_found"] += int((b_s & (lb >= 0)).sum())
        a = np.where(d[st] == 1, lb, lf)
        checks["missing"] += int((a < 0).sum())
        lab[st] = np.where(a >= 0, a, K_LAB)
    valid = seeds >= 0
    S, Bk = seeds[valid].astype(np.int64), buckets[valid].astype(np.int64)
    sb = np.full(n, -1, np.int64)
    sb[S[::-1]] = Bk[::-1]
    first = np.flatnonzero(sb[u] >= 0)
    one = (v[first], d[first], lab[first], sb[u[first]])
    order = np.argsort(u, kind="stable")
    us = u[order]
    nodes = np.arange(n)
    start, end = np.searchsorted(us, nodes), np.searchsorted(us, nodes, side="right")
    mid = v[first]
    cnt = end[mid] - start[mid]
    rep = np.repeat(first, cnt)
    par = np.repeat(np.arange(first.size), cnt)
    pos = np.repeat(start[mid] - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(int(cnt.sum()))
    second = order[pos]
    keep = v[second] != u[rep]
    rep, second, par = rep[keep], second[keep], par[keep]
    two = (v[second], d[rep], lab[rep], d[second], lab[second], sb[u[rep]])
    return one, two, par, t_lab


def row_tokens(gold_row, one, two, par):
    """The row's distinct (direction, label) tokens and whether an edge carrying one lies on a seed walk of at most two
    edges that ends at a gold node: a one-hop walk whose node is gold or that a gold-ending two-hop walk continues, and
    a gold-ending two-hop walk's second edge (its first is its parent's)."""
    v1, d1, l1, _b1 = one
    v2, _d2a, _l2a, d2b, l2b, _b2 = two
    on2 = gold_row[v2] > 0
    on1 = gold_row[v1] > 0
    if par.size:
        on1 = on1 | (np.bincount(par[on2], minlength=v1.size)[:v1.size] > 0)
    tok = np.concatenate([d1.astype(np.int64) * (K_LAB + 1) + l1, d2b.astype(np.int64) * (K_LAB + 1) + l2b])
    o = np.concatenate([on1, on2]).astype(np.int8)
    if tok.size == 0:
        e = np.zeros(0, np.int64)
        return e, e, np.zeros(0, bool)
    u, inv = np.unique(tok, return_inverse=True)
    oo = np.zeros(u.size, np.int8)
    np.maximum.at(oo, inv, o)
    return u // (K_LAB + 1), u % (K_LAB + 1), oo.astype(bool)


def mask_walks(one, two, hold):
    """The walks with every held label read OTHER, and how many walk labels changed."""
    m1, m2a, m2b = np.isin(one[2], hold), np.isin(two[2], hold), np.isin(two[4], hold)
    return ((one[0], one[1], np.where(m1, K_LAB, one[2]), one[3]),
            (two[0], two[1], np.where(m2a, K_LAB, two[2]), two[3], np.where(m2b, K_LAB, two[4]), two[5]),
            int(m1.sum() + m2a.sum() + m2b.sum()))


class Carve7(L3.Carve3):
    """lean_mlp3's carve plus lean_mlp4's AW walks (walk_entries7), each walk edge's cos(q, label), and the rows' tokens
    with their on-path flags. With hold (label ranks) and hold_mode 'train' the held labels read OTHER from the start,
    tokens included; with 'read' they are kept, aw_mask shows them as OTHER in a batch, and held marks the rows with
    an on-path token of a held label."""

    def __init__(self, ds, carve, store=None, nodes=None, limit=None, root=LM.LOOK, labels=None, basis=None,
                 struct_only=False, hold=None, hold_mode=None):
        super().__init__(ds, carve, store, nodes, limit, root)
        one = {k: [] for k in ("node", "dir", "lab", "b")}
        two = {k: [] for k in ("node", "d1", "l1", "d2", "l2", "b")}
        tok = {k: [] for k in ("d", "lab", "o")}
        n1, n2, nt, per_q, lab_s = [], [], [], [], 0.0
        self.label_checks = {"fwd_flag": 0, "fwd_found": 0, "bwd_flag": 0, "bwd_found": 0, "missing": 0}
        if hold is not None and hold_mode not in ("train", "read"):
            raise SystemExit(f"hold_mode {hold_mode!r}")
        self.hold = None if hold is None else np.asarray(sorted(hold), np.int64)
        self.hold_mode = None if hold is None else hold_mode
        self.aw_mask = False
        n_masked, touched, held_on = 0, [], []
        k, cache = 0, None
        for arrays, i in L4.chunk_rows4(Path(root) / ds / carve, limit):
            if cache is None or cache[0] is not arrays:
                cache = (arrays, np.concatenate([[0], np.cumsum(arrays["q_pool_size"])]), np.concatenate([[0], np.cumsum(arrays["q_edges"])]))
            _, no, eo = cache
            a, b, ea, eb = no[i], no[i + 1], eo[i], eo[i + 1]
            n = int(arrays["q_pool_size"][i])
            assert n == int(self.n[k]), (ds, carve, k)
            t = time.perf_counter()
            o, w, par, t_lab = walk_entries7(n, arrays["pool"][a:b], arrays["e_u"][ea:eb], arrays["e_v"][ea:eb],
                                             arrays["e_fam"][ea:eb], arrays["e_fwd"][ea:eb], arrays["e_bwd"][ea:eb],
                                             arrays["q_seed_local"][i], arrays["q_seed_bucket"][i], labels, self.label_checks,
                                             struct_only)
            per_q.append(1e3 * (time.perf_counter() - t - t_lab))
            lab_s += t_lab
            if self.hold_mode == "train":
                o, w, nm = mask_walks(o, w, self.hold)
                n_masked += nm
                touched.append(nm > 0)
            td, tl, to = row_tokens(self.gold[self.off[k]:self.off[k + 1]], o, w, par)
            if self.hold_mode == "read":
                h = np.isin(tl, self.hold)
                touched.append(bool(h.any()))
                held_on.append(bool((h & to).any()))
            for key, arr in zip(one, o):
                one[key].append(arr)
            for key, arr in zip(two, w):
                two[key].append(arr)
            for key, arr in zip(tok, (td, tl, to)):
                tok[key].append(arr)
            n1.append(o[0].size)
            n2.append(w[0].size)
            nt.append(td.size)
            k += 1
        assert k == self.rows, (ds, carve, k, self.rows)
        cat = (lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt))
        self.aw1 = {"node": cat(one["node"], np.int64), "dir": cat(one["dir"], np.int64), "lab": cat(one["lab"], np.int64),
                    "b": cat(one["b"], np.int64)}
        self.aw2 = {"node": cat(two["node"], np.int64), "d1": cat(two["d1"], np.int64), "l1": cat(two["l1"], np.int64),
                    "d2": cat(two["d2"], np.int64), "l2": cat(two["l2"], np.int64), "b": cat(two["b"], np.int64)}
        self.tok = {"d": cat(tok["d"], np.int64), "lab": cat(tok["lab"], np.int64), "o": cat(tok["o"], bool)}
        self.aw_off1 = np.concatenate([[0], np.cumsum(n1)]).astype(np.int64)
        self.aw_off2 = np.concatenate([[0], np.cumsum(n2)]).astype(np.int64)
        self.tok_off = np.concatenate([[0], np.cumsum(nt)]).astype(np.int64)
        self.aw_Z = L4.codes_of(labels, basis)
        self.aw_nr = False
        self.struct_only = bool(struct_only)
        t = time.time()
        self.cos_of(text_table(labels))
        cos_s = time.time() - t
        self.widths["AW"] = AW_W
        self.lean_ms["AW"] = {"p50": float(np.percentile(per_q, 50)), "p95": float(np.percentile(per_q, 95)), "mean": float(np.mean(per_q))}
        nt_a = np.asarray(nt)
        on_rows = np.asarray([bool(self.tok["o"][self.tok_off[r]:self.tok_off[r + 1]].any()) for r in range(self.rows)])
        self.held = np.asarray(held_on, bool) if self.hold_mode == "read" else None
        hold_stats = None if self.hold is None else {
            "mode": self.hold_mode, "ranks": self.hold.tolist(), "walk_labels_masked": n_masked if self.hold_mode == "train" else None,
            "rows_touched": int(np.sum(touched)), "held_rows": None if self.held is None else int(self.held.sum())}
        self.aw_stats = {"one_hop_mean": float(np.mean(n1)), "two_hop_mean": float(np.mean(n2)), "two_hop_max": int(np.max(n2)),
                         "label_lookup_s": lab_s, "labelled": bool(labels is not None and labels.ok), "struct_only": self.struct_only,
                         "other_share_one_hop": float((self.aw1["lab"] == K_LAB).mean()) if self.aw1["lab"].size else None,
                         "tokens_mean": float(nt_a.mean()), "token_on_share": float(self.tok["o"].mean()) if self.tok["o"].size else None,
                         "rows_with_on_token": float(on_rows.mean()), "cos_s": cos_s, "checks": self.label_checks, "hold": hold_stats}
        if hold_stats is not None:
            log(f"{ds}={carve}: hold {hold_stats}")
        log(f"{ds}={carve}: AW walks one-hop {np.mean(n1):.1f}, two-hop {np.mean(n2):.1f} (max {np.max(n2)}) a row; tokens "
            f"{nt_a.mean():.1f} a row, on-path {self.aw_stats['token_on_share']}; entry ms p50 {self.lean_ms['AW']['p50']:.3f}; "
            f"label lookup {lab_s:.1f}s; cos {cos_s:.1f}s; checks {self.label_checks}")
        c = self.label_checks
        if labels is not None and labels.ok and (c["fwd_found"] != c["fwd_flag"] or c["bwd_found"] != c["bwd_flag"]):
            raise SystemExit(f"{ds}={carve}: the pool's direction flags do not match the stored edges: {c}")

    def cos_of(self, T):
        """cos(q, t(label)) of every walk edge and token (0 for OTHER), in blocks of rows."""
        Q = self.q_emb.astype(np.float32)
        Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
        q1 = np.repeat(np.arange(self.rows), np.diff(self.aw_off1))
        q2 = np.repeat(np.arange(self.rows), np.diff(self.aw_off2))
        qt = np.repeat(np.arange(self.rows), np.diff(self.tok_off))
        self.aw1["cos"] = np.zeros(q1.size, np.float32)
        self.aw2["c1"] = np.zeros(q2.size, np.float32)
        self.aw2["c2"] = np.zeros(q2.size, np.float32)
        self.tok["cos"] = np.zeros(qt.size, np.float32)
        for a in range(0, self.rows, 1024):
            b = min(self.rows, a + 1024)
            CQ = Q[a:b] @ T.T
            for arr, qq, lab, off in ((self.aw1["cos"], q1, self.aw1["lab"], self.aw_off1), (self.aw2["c1"], q2, self.aw2["l1"], self.aw_off2),
                                      (self.aw2["c2"], q2, self.aw2["l2"], self.aw_off2), (self.tok["cos"], qt, self.tok["lab"], self.tok_off)):
                s = slice(off[a], off[b])
                arr[s] = CQ[qq[s] - a, lab[s]]


def aw_batch7(carve, qs):
    """lean_mlp4.aw_batch's tensors (the self-test checks them) plus the walk edges' cos, the rows' tokens and the graph."""
    base = np.concatenate([[0], np.cumsum(carve.n[qs])])[:-1]
    s1 = [np.arange(carve.aw_off1[q], carve.aw_off1[q + 1]) for q in qs]
    s2 = [np.arange(carve.aw_off2[q], carve.aw_off2[q + 1]) for q in qs]
    st = [np.arange(carve.tok_off[q], carve.tok_off[q + 1]) for q in qs]
    i1, i2, it = np.concatenate(s1), np.concatenate(s2), np.concatenate(st)
    q1 = np.repeat(np.arange(len(qs)), [s.size for s in s1])
    q2 = np.repeat(np.arange(len(qs)), [s.size for s in s2])
    qt = np.repeat(np.arange(len(qs)), [s.size for s in st])
    t = torch.from_numpy
    out = {"qe": t(carve.q_emb[qs].astype(np.float32)), "Z": t(carve.aw_Z), "N": int(carve.n[qs].sum()),
           "n1": t(carve.aw1["node"][i1] + base[q1]), "q1": t(q1), "d1": t(carve.aw1["dir"][i1]), "l1": t(carve.aw1["lab"][i1]),
           "b1": t(carve.aw1["b"][i1]),
           "n2": t(carve.aw2["node"][i2] + base[q2]), "q2": t(q2), "d2a": t(carve.aw2["d1"][i2]), "l2a": t(carve.aw2["l1"][i2]),
           "d2b": t(carve.aw2["d2"][i2]), "l2b": t(carve.aw2["l2"][i2]), "b2": t(carve.aw2["b"][i2]),
           "c1": t(carve.aw1["cos"][i1]), "c2a": t(carve.aw2["c1"][i2]), "c2b": t(carve.aw2["c2"][i2]),
           "tq": t(qt), "td": t(carve.tok["d"][it]), "tl": t(carve.tok["lab"][it]), "to": t(carve.tok["o"][it]),
           "tc": t(carve.tok["cos"][it]), "g": carve.ds}
    if getattr(carve, "aw_mask", False):
        H = torch.from_numpy(carve.hold)
        for lk, ck in (("l1", "c1"), ("l2a", "c2a"), ("l2b", "c2b"), ("tl", "tc")):
            m = torch.isin(out[lk], H)
            out[lk] = torch.where(m, torch.full_like(out[lk], K_LAB), out[lk])
            out[ck] = torch.where(m, torch.zeros_like(out[ck]), out[ck])
    if carve.aw_nr:
        for k in ("l1", "l2a", "l2b", "tl"):
            out[k] = torch.full_like(out[k], K_LAB)
        for k in ("c1", "c2a", "c2b", "tc"):
            out[k] = torch.zeros_like(out[k])
    return out


def batch_of7(carve, qs, blocks):
    feats, nq, base_z, gold, idx = _BATCH_OF(carve, qs, [b for b in blocks if b != "AW"])
    if "AW" in blocks:
        feats["AW"] = aw_batch7(carve, qs)
    return feats, nq, base_z, gold, idx


# ── the model ────────────────────────────────────────────────────────────────


class Typer(nn.Module):
    """pi(z | a, d) from the label's text code and the direction; rho(q, z) from the query."""

    def __init__(self, r, k, D, g, hid=64):
        super().__init__()
        din = r + N_DIR + 1
        self.k, self.D = k, D
        self.f1 = nn.Parameter(torch.randn(din, hid, generator=g) / math.sqrt(din))
        self.b1 = nn.Parameter(torch.zeros(hid))
        self.f2 = nn.Parameter(torch.randn(hid, k, generator=g) / math.sqrt(hid))
        self.b2 = nn.Parameter(torch.zeros(k))
        self.Wt = nn.Parameter(torch.randn(1536, D, generator=g) / math.sqrt(1536))
        self.E = nn.Parameter(torch.randn(k, D, generator=g))
        self.c = nn.Parameter(torch.zeros(k))

    def prior_logits(self, Z, d, lab, hide=None):
        has = lab < K_LAB
        if hide is not None:
            has = has & ~hide
        hf = has.unsqueeze(1).to(Z.dtype)
        x = torch.cat([Z[lab.clamp(max=K_LAB)] * hf, Fn.one_hot(d, N_DIR).to(Z.dtype), hf], 1)
        return torch.relu(x @ self.f1 + self.b1) @ self.f2 + self.b2

    def rho(self, qe):
        h = Fn.layer_norm(qe @ self.Wt, (self.D,))
        return h @ self.E.T / math.sqrt(self.D) + self.c


class AWNet7(L4.AWNet):
    """lean_mlp4.AWNet with the arm's extra logit terms; with neither (arm aw) its forward is AWNet's, op for op."""

    def __init__(self, r, D=64, k_res=0, p_drop=0.25, seed=0, arm="aw", k_types=8):
        super().__init__(r, D, k_res, p_drop, seed)
        self.arm = arm
        self.kappa = nn.Parameter(torch.zeros(N_DIR)) if arm != "aw" else None
        self.typer = None
        if arm in EM_ARMS:
            self.typer = Typer(r, k_types, D, torch.Generator().manual_seed(seed + 13))
            self.eta = nn.Parameter(torch.zeros(N_DIR))       # 0: the arm starts as awc

    def typer_r(self, qe, Z, parts):
        """r(q, e) for each (query index, direction, label) list in parts, detached: the typer is fitted by EM alone."""
        with torch.no_grad():
            toks = [d * (K_LAB + 1) + lab.clamp(max=K_LAB) for _q, d, lab, _c in parts]
            u, inv = torch.unique(torch.cat(toks), return_inverse=True)
            pi = torch.softmax(self.typer.prior_logits(Z, u // (K_LAB + 1), u % (K_LAB + 1)), 1)
            sr = torch.sigmoid(self.typer.rho(qe))
            out, k = [], 0
            for (qi, _d, _l, _c), t in zip(parts, toks):
                iv = inv[k:k + t.numel()]
                k += t.numel()
                out.append((pi[iv] * sr[qi]).sum(1))
            return out

    def logit7(self, phi, qi, d, lab, cos, Z, r=None):
        lg = self.logit(phi, qi, d, lab, Z)
        if self.kappa is not None:
            lg = lg + self.kappa[d] * torch.where(lab < K_LAB, cos, torch.zeros_like(cos))
        if r is not None:
            lg = lg + self.eta[d] * torch.log(r.clamp_min(1e-6))
        return lg

    def forward(self, A):
        phi = A["qe"] @ self.Wq
        Z, N = A["Z"], A["N"]
        l1, l2a, l2b = A["l1"], A["l2a"], A["l2b"]
        if self.training and self.p_drop > 0:
            l1 = torch.where(torch.rand(l1.shape) < self.p_drop, torch.full_like(l1, K_LAB), l1)
            l2a = torch.where(torch.rand(l2a.shape) < self.p_drop, torch.full_like(l2a, K_LAB), l2a)
            l2b = torch.where(torch.rand(l2b.shape) < self.p_drop, torch.full_like(l2b, K_LAB), l2b)
        parts = [(A["q1"], A["d1"], l1, A.get("c1")), (A["q2"], A["d2a"], l2a, A.get("c2a")), (A["q2"], A["d2b"], l2b, A.get("c2b"))]
        rs = self.typer_r(A["qe"], Z, parts) if self.typer is not None else [None, None, None]
        la1, la2, lb2 = (self.logit7(phi, qi, d, lab, c, Z, r) for (qi, d, lab, c), r in zip(parts, rs))
        cols = []
        for node, la, lb, bk in ((A["n1"], la1, None, A["b1"]), (A["n2"], la2, lb2, A["b2"])):
            g = torch.sigmoid(la) if lb is None else torch.sigmoid(la) * torch.sigmoid(lb)
            sc = la if lb is None else la + lb
            s = torch.zeros(N).index_add(0, node, g)
            mx = torch.full((N,), float("-inf")).scatter_reduce(0, node, sc, reduce="amax", include_self=True)
            m0 = bk == 0
            s0 = torch.zeros(N).index_add(0, node[m0], g[m0])
            cols += [torch.log1p(s), torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx)), torch.log1p(s0)]
        return torch.stack(cols, 1)

    def unit_logit(self, A, lab):
        """The AW logit of each of the batch's tokens under the labels lab."""
        phi = A["qe"] @ self.Wq
        parts = [(A["tq"], A["td"], lab, A["tc"])]
        r = self.typer_r(A["qe"], A["Z"], parts)[0] if self.typer is not None else None
        return self.logit7(phi, A["tq"], A["td"], lab, A["tc"], A["Z"], r)

    def hide_draw(self, shape):
        if self.training and self.p_drop > 0:
            return torch.rand(shape) < self.p_drop
        return torch.zeros(shape, dtype=torch.bool)

    def aux_loss(self, A):
        """The token loss: each token's AW logit against on-path, on- and off-path tokens at half the batch each."""
        o = A["to"]
        w = balanced(o)
        if w is None:
            return None
        hide = self.hide_draw(o.shape)
        lab = torch.where(hide, torch.full_like(A["tl"], K_LAB), A["tl"])
        lg = self.unit_logit(A, lab)
        return (w * Fn.binary_cross_entropy_with_logits(lg, o.to(lg.dtype), reduction="none")).sum()

    def em_loss(self, A, em, lam):
        """One E step (no gradient) and the M step's loss on the batch's tokens; theta's M step in closed form."""
        o, tl, td, tq, Z = A["to"], A["tl"], A["td"], A["tq"], A["Z"]
        if o.numel() == 0:
            return None
        hide = self.hide_draw(o.shape)
        lp = torch.log_softmax(self.typer.prior_logits(Z, td, tl, hide), 1)
        rr = self.typer.rho(A["qe"])[tq]
        lb = torch.where(o.unsqueeze(1), Fn.logsigmoid(rr), Fn.logsigmoid(-rr))
        has = tl < K_LAB
        with torch.no_grad():
            post = lp + em.mu * lb
            if lam > 0 and bool(has.any()):
                post[has] = post[has] + lam * em.log_theta(A["g"])[tl[has]]
            q = torch.softmax(post, 1)
            em.observe(A["g"], tl[has], q[has], lp, q, o)
        loss = -(q * lp).sum(1).mean()
        w = balanced(o)
        if w is not None:
            loss = loss - (w.unsqueeze(1) * q * lb).sum()
        return em.omega * loss


def balanced(o):
    n_on = int(o.sum())
    n_off = int(o.numel()) - n_on
    if n_on == 0 or n_off == 0:
        return None
    return torch.where(o, torch.full(o.shape, 0.5 / n_on), torch.full(o.shape, 0.5 / n_off))


class EMState:
    """theta_g per training graph (a decayed count of the posterior over the tokens' true labels), the anneal and the
    epoch trace."""

    def __init__(self, k, graphs, lam0, an, mu=1.0, omega=1.0, decay=0.995, alpha=1e-2):
        self.k, self.lam0, self.an, self.mu, self.omega, self.decay, self.alpha = k, lam0, an, mu, omega, decay, alpha
        self.C = {g: torch.zeros(K_LAB, k, dtype=torch.float64) for g in graphs}
        self.tr = None
        self.reset()

    def lam(self, ep):
        return self.lam0 * max(0.0, 1.0 - ep / self.an) if self.an > 0 else 0.0

    def log_theta(self, g):
        C = self.C[g] + self.alpha
        return torch.log(C / C.sum(0, keepdim=True)).to(torch.float32)

    def observe(self, g, labs, q_has, lp, q, o):
        self.C[g].mul_(self.decay).index_add_(0, labs, q_has.to(torch.float64))
        p = lp.exp()
        t = self.tr
        t["units"] += int(o.numel())
        t["on"] += int(o.sum())
        t["H_post"] += float(-(q * torch.log(q.clamp_min(1e-12))).sum())
        t["H_prior"] += float(-(p * lp).sum())
        t["agree"] += int((q.argmax(1) == lp.argmax(1)).sum())

    def reset(self):
        self.tr = {"units": 0, "on": 0, "H_post": 0.0, "H_prior": 0.0, "agree": 0}

    def nmi(self, g):
        """NMI between the graph's labels and the types, from the decayed posterior counts."""
        C = self.C[g]
        tot = float(C.sum())
        if tot <= 0:
            return None
        P = C / tot
        pa, pz = P.sum(1, keepdim=True), P.sum(0, keepdim=True)
        nz = P > 0
        mi = float((P[nz] * torch.log(P[nz] / (pa @ pz)[nz])).sum())
        ha = float(-(pa[pa > 0] * torch.log(pa[pa > 0])).sum())
        hz = float(-(pz[pz > 0] * torch.log(pz[pz > 0])).sum())
        return mi / math.sqrt(ha * hz) if ha > 0 and hz > 0 else None

    def epoch_trace(self, ep, lam):
        t, n = self.tr, max(self.tr["units"], 1)
        rec = {"epoch": ep, "lambda": round(lam, 4), "units": t["units"], "on_share": round(t["on"] / n, 5),
               "H_post": round(t["H_post"] / n, 4), "H_prior": round(t["H_prior"] / n, 4), "agree": round(t["agree"] / n, 4),
               "nmi": {g: (None if self.nmi(g) is None else round(self.nmi(g), 4)) for g in self.C}}
        self.reset()
        return rec


class LeanMLP7(L3.LeanMLP3):
    """lean_mlp3's model; with AW in its blocks, an AWNet7 of its arm turns the batch's walks into AW's six columns."""

    cfg = {"r": 32, "D": 64, "k_res": 0, "p_drop": 0.25, "k_types": 8}

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="aw"):
        super().__init__(blocks, widths, hidden, dropout, seed)
        self.arm = arm
        if "AW" in self.blocks:
            c = LeanMLP7.cfg
            self.awn = AWNet7(c["r"], c["D"], c["k_res"], c["p_drop"], seed, arm, c.get("k_types", 8))

    def forward(self, feats, keep, nq, B, base_z):
        if "AW" in self.blocks and isinstance(feats.get("AW"), dict):
            feats = dict(feats)
            feats["AW"] = self.awn(feats["AW"])
        return super().forward(feats, keep, nq, B, base_z)


def build7(blocks, widths, hidden, state, arm):
    m = LeanMLP7(blocks, widths, hidden, arm=arm)
    m.load_state_dict(state)
    m.eval()
    return m


# ── the fit ──────────────────────────────────────────────────────────────────


def fit7(train, select, blocks, arm, cfg, seed, hidden, emc=None, aux_omega=1.0):
    """lean_mlp5.fit5's loop (Adam, the config's lr, wd and dropout; best, swa and last weights) on LeanMLP7, plus the
    arm's token loss or EM step. With arm aw and a base config the best weights are lean_mlp4.fit4's (mode fixed, rule
    rf) bit for bit."""
    if cfg["cos"] or cfg["adamw"] or cfg["drop"]:
        raise SystemExit("fit7 takes lr:wd:dropout:epochs:swa_from only")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    widths = {b: train[0].widths[b] for b in blocks}
    model = LeanMLP7(blocks, widths, hidden, dropout=cfg["dropout"], seed=seed, arm=arm)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    units = [(ci, q) for ci, c in enumerate(train) for q in range(c.rows)]
    aw = "AW" in blocks
    em = None
    if aw and arm in EM_ARMS:
        em = EMState(LeanMLP7.cfg.get("k_types", 8), sorted({c.ds for c in train}), emc["lam0"] if arm == "em" else 0.0,
                     emc["an"], emc["mu"], emc["omega"])
    keep_all = None
    best, best_state, best_ep, curve = -1.0, None, -1, []
    acc, n_acc = None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(len(units))
        t0 = time.time()
        tot, nb, tot_x, nx = 0.0, 0, 0.0, 0
        lam = em.lam(ep) if em is not None else 0.0
        for k in range(0, len(order), 32):
            chunk = [units[j] for j in order[k:k + 32]]
            for ci in sorted({c for c, _ in chunk}):
                qs = np.asarray([q for c, q in chunk if c == ci])
                feats, nq, base_z, gold, _ = batch_of7(train[ci], qs, blocks)
                B = qs.size
                if keep_all is None or keep_all.shape[0] != B:
                    keep_all = torch.ones((B, len(blocks)), dtype=torch.float32)
                s = model(feats, keep_all, nq, B, base_z)
                loss = LM.listwise(s, gold, nq, B)
                x = None
                if aw and arm == "aux":
                    x = model.awn.aux_loss(feats["AW"])
                    x = None if x is None else aux_omega * x
                elif em is not None:
                    x = model.awn.em_loss(feats["AW"], em, lam)
                if x is not None:
                    tot_x += float(x.item())
                    nx += 1
                    loss = loss + x
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
        q = LM.quality(model, select, blocks, {b: 1.0 for b in blocks})
        state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        rec = {"epoch": ep, "loss": tot / max(nb, 1), "extra": tot_x / max(nx, 1) if nx else None, "select": q,
               "seconds": time.time() - t0}
        if aw and model.awn.kappa is not None:
            rec["kappa"] = [round(float(v), 4) for v in model.awn.kappa.detach()]
        if em is not None:
            rec["em"] = em.epoch_trace(ep, lam)
            rec["eta"] = [round(float(v), 4) for v in model.awn.eta.detach()]
        curve.append(rec)
        log(f"  {arm} ep {ep}: loss {rec['loss']:.4f} extra {rec['extra']} select R@5/FC@5/hit@1 {[round(v, 4) for v in q]} "
            f"({rec['seconds']:.0f}s){' kappa ' + str(rec['kappa']) if 'kappa' in rec else ''}"
            f"{' em ' + json.dumps(rec['em']) + ' eta ' + str(rec['eta']) if 'em' in rec else ''}")
        score = 0.5 * (q[0] + q[1])
        if score > best:
            best, best_state, best_ep = score, state, ep
        if ep >= cfg["swa_from"]:
            if acc is None:
                acc = {k_: v.double().clone() for k_, v in state.items()}
            else:
                for k_, v in state.items():
                    acc[k_] += v.double()
            n_acc += 1
    swa_state = {k_: (v / n_acc).to(torch.float32) for k_, v in acc.items()}
    last_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
    return {"best": best_state, "swa": swa_state, "last": last_state, "best_epoch": best_ep, "curve": curve, "widths": widths,
            "swa_epochs": list(range(cfg["swa_from"], cfg["epochs"]))}


# ── reads ────────────────────────────────────────────────────────────────────


def avg_rank(x):
    """1-based ranks, ties sharing their mean rank."""
    o = np.argsort(x, kind="mergesort")
    xs = x[o]
    new = np.r_[True, xs[1:] != xs[:-1]]
    gid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], xs.size]
    r = np.empty(x.size)
    r[o] = ((starts + ends - 1) / 2.0 + 1.0)[gid]
    return r


def row_auc(score, o, off):
    """Per row, the AUC of score against o over the row's tokens, and the rows (with both kinds only)."""
    out, ids = [], []
    for i in range(off.size - 1):
        s, y = score[off[i]:off[i + 1]], o[off[i]:off[i + 1]]
        npos = int(y.sum())
        nneg = int(y.size) - npos
        if npos == 0 or nneg == 0:
            continue
        r = avg_rank(s)
        out.append((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))
        ids.append(i)
    return np.asarray(out), np.asarray(ids, np.int64)


@torch.no_grad()
def edge_diag(model, c):
    """The edge ranking on a carve's rows: AUC of the AW token logit (and the typer's r, its query-blind form and r
    with every label hidden) against on-path, per row; on a held graph's read rows also on the held rows alone."""
    aw = model.awn
    model.eval()
    keys = ["aw"] + (["r", "blind", "hidden"] if aw.typer is not None else [])
    sc = {k: [] for k in keys}
    for k in range(0, c.rows, 64):
        qs = np.arange(k, min(k + 64, c.rows))
        A = aw_batch7(c, qs)
        tl, td, tq = A["tl"], A["td"], A["tq"]
        sc["aw"].append(aw.unit_logit(A, tl).numpy())
        if aw.typer is not None:
            Z = A["Z"]
            pi = torch.softmax(aw.typer.prior_logits(Z, td, tl), 1)
            pih = torch.softmax(aw.typer.prior_logits(Z, td, torch.full_like(tl, K_LAB)), 1)
            sr = torch.sigmoid(aw.typer.rho(A["qe"]))[tq]
            sc["r"].append((pi * sr).sum(1).numpy())
            sc["hidden"].append((pih * sr).sum(1).numpy())
            sc["blind"].append((pi * torch.sigmoid(aw.typer.c)).sum(1).numpy())
    o = c.tok["o"]
    held = getattr(c, "held", None)
    out = {"tokens_mean": float(np.diff(c.tok_off).mean()) if c.rows else None, "on_share": float(o.mean()) if o.size else None}
    for key in keys:
        s = np.concatenate(sc[key]) if sc[key] else np.zeros(0, np.float32)
        v, ids = row_auc(s, o, c.tok_off)
        out[f"auc_{key}"] = float(v.mean()) if v.size else None
        out["rows_both"] = int(v.size)
        if held is not None:
            h = held[ids]
            out[f"auc_{key}_held"] = float(v[h].mean()) if h.any() else None
            out["rows_both_held"] = int(h.sum())
    return out


def held_summary(models, rows_of, mask_of, refs, hm, rng):
    """The held rows' reads: each model shown the held labels (REV, its ID read) and masked (MASK), against the twin,
    each other and ctl at the same kind."""
    out = {"rows": int(hm.sum())}
    if hm.sum() < 2:
        return out
    t, g = refs["twin0"][hm].mean(0), refs["gnn0"][hm].mean(0)
    out["twin0"], out["gnn0"] = t.tolist(), g.tolist()

    def rho(x):
        return [float((x[j] - t[j]) / (g[j] - t[j])) if abs(g[j] - t[j]) > 1e-9 else None for j in range(3)]

    for name in models:
        x = rows_of[name][hm]
        e = {"REV": x.mean(0).tolist(), "rho_REV": rho(x.mean(0)), "REV_minus_twin0": LM.boot_diff(x, refs["twin0"][hm], rng)}
        if name in mask_of:
            y = mask_of[name][hm]
            e.update({"MASK": y.mean(0).tolist(), "rho_MASK": rho(y.mean(0)), "REV_minus_MASK": LM.boot_diff(x, y, rng),
                      "MASK_minus_twin0": LM.boot_diff(y, refs["twin0"][hm], rng)})
        ctl = f"ctl@{name.split('@')[1]}"
        if ctl in rows_of and ctl != name:
            e["REV_minus_ctl"] = LM.boot_diff(x, rows_of[ctl][hm], rng)
        out[name] = e
    return out


def pts(d):
    return " ".join(f"{100 * p[0]:+.2f}[{100 * p[1][0]:+.2f},{100 * p[1][1]:+.2f}]" for p in d)


def read7(a, models, out, carve):
    """Each model on each --read carve: ID, NR (AW models), MASK and the held rows (a held graph), the paired
    differences and the edge ranking. The per-row metrics of every read (rows with gold, in carve order) and the
    twin's and the GNN's go to <out>.rows.npz, so arms of different jobs can be paired on the same rows."""
    rng = np.random.default_rng(20261003)
    reads, read_costs, read_lean_ms, read_rows, aw_stats, edge, held = {}, {}, {}, {}, {}, {}, {}
    store = {}
    for ds, cv in LM.parse_sets(a.read):
        key = f"{ds}={cv}"
        c = carve(ds, cv, "read")
        read_rows[key] = {"rows": c.rows, "chunks": c.chunks_read, "n_chunks": c.n_chunks, "carve_queries": c.carve_queries}
        aw_stats[key] = c.aw_stats
        (prof, share, lm), = L2.costs_of([c])
        read_lean_ms[key] = c.lean_ms
        read_costs[key] = {name: LM.cost_ms(bl, prof, share, lm) for name, (_m, bl, _k) in models.items()}
        log(f"read {key} ({c.rows} rows); approximate compile ms {({k: round(v, 2) for k, v in read_costs[key].items()})}")
        okm = c.gold_total > 0
        refs = {nm: LM.row_metrics(c.score[:, col], c.gold, c.off, c.gold_total)[okm] for nm, col in LM.SCORE_COL.items()}
        for nm, v in refs.items():
            store[f"{nm}@{key}"] = v.astype(np.float32)
        rows_of = {}
        for name, (m, bl, keep) in models.items():
            r, rows = LM.read_one(m, c, bl, keep, rng)
            rows_of[name] = rows
            store[f"{name}@{key}"] = rows.astype(np.float32)
            reads[f"{name}@{key}"] = r
        for name in models:
            arm, kind = name.split("@")
            r = reads[f"{name}@{key}"]
            for ref_arm in ("ctl", "awc"):
                ref = f"{ref_arm}@{kind}"
                if ref in rows_of and ref != name and (ref_arm == "ctl" or arm in ("aux", "em", "em0")):
                    r[f"minus_{ref_arm}"] = LM.boot_diff(rows_of[name], rows_of[ref], rng)
            extra = "".join(f" vs {k[6:]} {pts(r[k])}" for k in ("minus_ctl", "minus_awc") if k in r)
            log(f"  {name}@{key}: {LM.fmt(r)}{extra}")
        c.aw_nr = True
        for name, (m, bl, keep) in models.items():
            if "AW" not in bl:
                continue
            r, rows = LM.read_one(m, c, bl, keep, rng, rows_of[name])
            reads[f"{name}@{key}/NR"] = r
            store[f"{name}@{key}~NR"] = rows.astype(np.float32)
            log(f"  {name}@{key}/NR: {LM.fmt(r)} (vs full = vs its own ID read)")
        c.aw_nr = False
        if c.hold_mode == "read":
            mask_of = {}
            c.aw_mask = True
            for name, (m, bl, keep) in models.items():
                if "AW" not in bl:
                    continue
                r, rows = LM.read_one(m, c, bl, keep, rng, rows_of[name])
                reads[f"{name}@{key}/MASK"] = r
                mask_of[name] = rows
                store[f"{name}@{key}~MASK"] = rows.astype(np.float32)
                log(f"  {name}@{key}/MASK: {LM.fmt(r)} (vs full = vs its own ID read, the held labels shown)")
            c.aw_mask = False
            hm = c.held[okm]
            store[f"held@{key}"] = hm
            held[key] = held_summary(models, rows_of, mask_of, refs, hm, rng)
            for name, e in held[key].items():
                if isinstance(e, dict):
                    log(f"  held rows ({held[key]['rows']}) {name}@{key}: REV rho {e['rho_REV']} vs twin0 {pts(e['REV_minus_twin0'])}"
                        + (f"; MASK rho {e['rho_MASK']}; REV - MASK {pts(e['REV_minus_MASK'])}" if "REV_minus_MASK" in e else "")
                        + (f"; REV - ctl {pts(e['REV_minus_ctl'])}" if "REV_minus_ctl" in e else ""))
        for name, (m, bl, keep) in models.items():
            if "AW" in bl:
                edge[f"{name}@{key}"] = edge_diag(m, c)
                log(f"  edge ranking {name}@{key}: {edge[f'{name}@{key}']}")
                if c.hold_mode == "read":
                    c.aw_mask = True
                    edge[f"{name}@{key}/MASK"] = edge_diag(m, c)
                    c.aw_mask = False
                    log(f"  edge ranking {name}@{key}/MASK: {edge[f'{name}@{key}/MASK']}")
        del c
    out["read_costs_ms"] = read_costs
    out["read_lean_ms"] = read_lean_ms
    out["read_rows"] = read_rows
    out["read_aw_stats"] = aw_stats
    out["edge_rank"] = edge
    out["held"] = held
    out["reads"] = reads
    if a.out:
        p = Path(a.out).with_suffix(".rows.npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp.npz")
        np.savez_compressed(tmp, **store)
        import os
        os.replace(tmp, p)
        out["rows_file"] = {"path": p.name, "keys": len(store), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        log(f"wrote {p} ({len(store)} arrays)")


# ── main ─────────────────────────────────────────────────────────────────────


def shas():
    s = {f"{n}_sha256": hashlib.sha256((HERE / f"{n}.py").read_bytes()).hexdigest()
         for n in ("lean_mlp", "lean_mlp2", "lean_mlp3", "lean_mlp4", "lean_mlp5")}
    s["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return s


def code_phrases(src, labels):
    """The phrase rows the label code basis is fitted on: a loaded graph's labelled rows, or (a graph not loaded) its
    anchor table's first K_LAB rows, the rows lean_mlp4.Labels would give, read without its edges."""
    if src in labels:
        return labels[src].phrase[:n_real(labels[src])] if labels[src].ok else None
    d = L4.ANCHORS.get(src)
    if d is None or not (d / "phrase_emb_w2_K4096.npy").exists():
        return None
    return np.load(d / "phrase_emb_w2_K4096.npy").astype(np.float32)[:K_LAB]


def parse_hold(spec, labels, train_names):
    """--hold <kb>:mask:<relation>[+<relation>...] ('+' or ';' between names; a name not in the vocabulary as written
    is read with each '_' as a space): the held KB, its relation names and their label ranks."""
    parts = spec.split(":")
    if len(parts) != 3 or parts[1] != "mask":
        raise SystemExit(f"--hold {spec!r}: <kb>:mask:<relation>[+<relation>...] (mask only: the lean blocks are compiled "
                         f"over every edge, so an edge cannot be dropped from them)")
    g = parts[0]
    if g not in KBS or g not in labels or not labels[g].ok:
        raise SystemExit(f"--hold {g}: the held graph must be a KB with relation labels, one of {KBS}")
    if g not in train_names:
        raise SystemExit(f"--hold {g}: the held graph must be a training graph")
    vocab = labels[g].vocab
    names = [s if s in vocab else s.replace("_", " ") for s in parts[2].replace(";", "+").split("+") if s]
    if not names or len(set(names)) != len(names):
        raise SystemExit(f"--hold {spec!r}: name the held relations once each")
    bad = [s for s in names if s not in vocab]
    if bad:
        raise SystemExit(f"--hold {bad}: not in {g}'s relation vocabulary ({vocab[:12]} ...)")
    ranks = sorted(int(labels[g].rank_of[vocab.index(s)]) for s in names)
    if ranks[-1] >= K_LAB:
        raise SystemExit(f"--hold {names}: a relation past the {K_LAB} most frequent already reads OTHER")
    return {"graph": g, "mode": "mask", "names": names, "ranks": ranks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="pca256", choices=("rand128", "pca256"))
    ap.add_argument("--basis-from", default="")
    ap.add_argument("--fit-nodes", type=int, default=60000)
    ap.add_argument("--train", default="2wiki=x4")
    ap.add_argument("--select", default="2wiki=select")
    ap.add_argument("--read", default="2wiki=x1")
    ap.add_argument("--set", default="rank+SEMB+SEED+WALK+DISTS", help="the block set; every arm but ctl adds AW")
    ap.add_argument("--arms", default="ctl,aw,awc")
    ap.add_argument("--config", default="2e-3:1e-4:0.1:8:2", help="lr:wd:dropout:epochs:swa_from (lean_mlp5's form)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--aw-r", type=int, default=32)
    ap.add_argument("--aw-dim", type=int, default=64)
    ap.add_argument("--aw-drop", type=float, default=0.25)
    ap.add_argument("--aw-struct", action="store_true", help="walks over structural edges only")
    ap.add_argument("--em-k", type=int, default=8)
    ap.add_argument("--em-lam", type=float, default=1.0)
    ap.add_argument("--em-an", type=int, default=0, help="anneal epochs (0: ceil(epochs / 2))")
    ap.add_argument("--em-mu", type=float, default=1.0)
    ap.add_argument("--em-omega", type=float, default=1.0)
    ap.add_argument("--aux-omega", type=float, default=1.0)
    ap.add_argument("--hold", default="", help="<kb>:mask:<relation>[+<relation>...]: the KB's training and select carves show "
                    "the held relations as OTHER from the start (tokens included); its read carves are read with them shown "
                    "(REV, the ID read) and masked (MASK), on every row and on the held rows")
    ap.add_argument("--code-from", default="", help="the graph whose phrase table fits the label code basis (default: the first "
                    "training graph; a KB has too few labels)")
    ap.add_argument("--save-models")
    ap.add_argument("--load-models")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    LM.batch_of = batch_of7                   # lean_mlp's quality and read paths batch through this name
    blob = torch.load(a.load_models, weights_only=False) if a.load_models else None
    store_name = blob["store"] if blob is not None else a.store
    L3.LeanMLP3.dim = L3.STORE_DIM if store_name == "pca256" else LM.PROJ_DIM
    if blob is not None:
        LeanMLP7.cfg = dict(blob["aw_cfg"])
        struct_only = bool(blob["args"].get("aw_struct", False))
    else:
        LeanMLP7.cfg = {"r": a.aw_r, "D": a.aw_dim, "k_res": 0, "p_drop": a.aw_drop, "k_types": a.em_k}
        struct_only = a.aw_struct
    names = sorted({ds for spec in (a.read,) + (() if blob is not None else (a.train, a.select)) for ds, _cv in LM.parse_sets(spec)})
    served, canonical, freeze = L4.served_root()
    labels = {n: labels_of(n, served, canonical) for n in names}
    if blob is not None:
        if a.hold:
            raise SystemExit("--hold with --load-models: the loaded fit's hold applies")
        cbasis = blob["code_basis"]
        hold = blob.get("hold")
    else:
        hold = parse_hold(a.hold, labels, {ds for ds, _cv in LM.parse_sets(a.train)}) if a.hold else None
        src = a.code_from or LM.parse_sets(a.train)[0][0]
        P = code_phrases(src, labels)
        if P is None or P.shape[0] < 4 * a.aw_r:
            raise SystemExit(f"{src}: {0 if P is None else P.shape[0]} labels; the code basis needs a passage graph's phrase "
                             f"table (--code-from)")
        cbasis = L4.code_basis(P, a.aw_r)
        cbasis["from"] = src
        del P
    if hold is not None:
        log(f"hold {hold}")
    store, nodes, basis_info = None, {}, None
    if store_name == "pca256":
        if blob is not None:
            basis = blob["basis"]
            nodes, _fz = L3.open_nodes(names)
        else:
            src = a.basis_from or LM.parse_sets(a.train)[0][0]
            nodes, _fz = L3.open_nodes(sorted(set(names) | {src}))
            basis = L3.fit_basis(nodes[src], a.fit_nodes)
            basis["from"] = src
            log(f"basis from {src}: {basis['rows']} nodes, {basis['explained']:.3f} of the variance on {L3.STORE_K} axes")
        store = L3.Store(basis)
        basis_info = {k: v for k, v in basis.items() if k not in ("m", "V", "w")}

    def carve(ds, cv, role):
        """A carve as a training, select or read carve: the held KB's training and select carves mask its held labels
        from the start, its read carves keep them for REV and MASK."""
        h = hold if hold is not None and ds == hold["graph"] else None
        return Carve7(ds, cv, store, nodes.get(ds), a.limit, labels=labels[ds], basis=cbasis, struct_only=struct_only,
                      hold=None if h is None else h["ranks"], hold_mode=None if h is None else ("read" if role == "read" else "train"))

    label_info = {n: {"ok": l.ok, "seconds": l.seconds, "phrase_sha256": getattr(l, "phrase_sha256", None),
                      "census": getattr(l, "census", None), "manifest": getattr(l, "manifest_entry", None)} for n, l in labels.items()}
    if blob is not None:
        models = {name: (build7(d["blocks"], d["widths"], d["hidden"], d["state"], d["arm"]), d["blocks"], d["keep"])
                  for name, d in blob["models"].items()}
        out = {"look": "lean_mlp7", "mode": "read_only", "store": blob["store"], "basis": basis_info, "freeze": freeze, "args": vars(a),
               "loaded": a.load_models, "loaded_sha256": hashlib.sha256(Path(a.load_models).read_bytes()).hexdigest(),
               "fit_args": blob.get("args"), "aw_cfg": blob["aw_cfg"], "labels": label_info, "hold": hold, **shas()}
        log(f"loaded {list(models)} from {a.load_models} (store {blob['store']}, AW {blob['aw_cfg']})")
        read7(a, models, out, carve)
        out["seconds"] = time.time() - t0
        L2.write(a, out)
        log(f"done in {time.time() - t0:.1f}s")
        return 0
    arms = [x for x in a.arms.split(",") if x]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; arms are {ARMS}")
    cfg = L5.parse_configs("x=" + a.config)["x"]
    emc = {"lam0": a.em_lam, "an": a.em_an or math.ceil(cfg["epochs"] / 2), "mu": a.em_mu, "omega": a.em_omega}
    tr = [carve(ds, cv, "train") for ds, cv in LM.parse_sets(a.train)]
    se = [carve(ds, cv, "select") for ds, cv in LM.parse_sets(a.select)]
    log(f"loaded train {[(c.ds, c.carve, c.rows) for c in tr]} select {[(c.ds, c.carve, c.rows) for c in se]} ({time.time() - t0:.0f}s)")
    known = set(LM.COMPILED) | set(LM.LEAN) | set(L2.NEW)
    base = [b for b in a.set.split("+") if b]
    if [b for b in base if b not in known] or "AW" in base:
        raise SystemExit(f"--set: unknown blocks {[b for b in base if b not in known]} (AW is added by the arms)")
    dead = [b for b in base if b not in LM.LEAN and float(np.nanstd(np.concatenate(
        [c.block(b, np.arange(int(c.off[-1]))).astype(np.float32) for c in tr]))) == 0.0]
    base = [b for b in base if b not in dead]
    costs = L2.costs_of(tr)
    sets = {arm: base if arm == "ctl" else base + ["AW"] for arm in arms}
    out = {"look": "lean_mlp7", "store": store_name, "basis": basis_info, "freeze": freeze, "args": vars(a), "config": cfg, "em": emc,
           "sets": sets, "dead": dead, "set_costs_ms": {arm: LM.cost_multi(bl, costs) for arm, bl in sets.items()},
           "aw_cfg": LeanMLP7.cfg, "code_basis": {k: v for k, v in cbasis.items() if k not in ("mu", "V", "w")},
           "aw_stats": {f"{c.ds}={c.carve}": c.aw_stats for c in tr + se},
           "lean_ms": {f"{c.ds}={c.carve}": c.lean_ms for c in tr + se},
           "profiles": {c.ds: LM.profile(c.ds) for c in tr}, "struct_share": {f"{c.ds}={c.carve}": c.struct_share for c in tr},
           "labels": label_info, "hold": hold, "fits": {},
           "assumptions": ["lean_mlp4's cost assumptions; cos and the typer's pi are index-time tables per label, rho one "
                           "query-side product (not timed here)"], **shas()}
    log(f"sets {sets}; constant on the training rows (dropped) {dead}; config {cfg}; em {emc}")
    models = {}
    for arm in arms:
        bl = sets[arm]
        log(f"fit {arm} ({len(bl)} blocks, {out['set_costs_ms'][arm]:.2f} ms)")
        t = time.time()
        f = fit7(tr, se, bl, arm, cfg, a.seed, a.hidden, emc, a.aux_omega)
        rec = {"best_epoch": f["best_epoch"], "swa_epochs": f["swa_epochs"], "curve": f["curve"], "seconds": time.time() - t}
        for kind in ("best", "swa", "last"):
            m = build7(bl, f["widths"], a.hidden, f[kind], arm)
            rec[f"select_{kind}"] = LM.quality(m, se, bl, {b: 1.0 for b in bl})
            if kind != "last":
                models[f"{arm}@{kind}"] = (m, bl, {b: 1.0 for b in bl})
        log(f"  {arm}: select best (ep {f['best_epoch']}) {[round(v, 4) for v in rec['select_best']]}, swa "
            f"{[round(v, 4) for v in rec['select_swa']]}, last {[round(v, 4) for v in rec['select_last']]} ({rec['seconds']:.0f}s)")
        out["fits"][arm] = rec
    if a.save_models:
        Path(a.save_models).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"models": {name: {"state": m.state_dict(), "blocks": bl, "keep": keep, "widths": m.widths, "hidden": a.hidden,
                                      "arm": name.split("@")[0]} for name, (m, bl, keep) in models.items()},
                    "args": vars(a), "store": store_name, "basis": None if store is None else store.basis, "aw_cfg": LeanMLP7.cfg,
                    "code_basis": cbasis, "sets": sets, "config": cfg, "em": emc, "hold": hold}, a.save_models)
        log(f"saved {len(models)} models to {a.save_models}")
    del tr, se
    read7(a, models, out, carve)
    out["seconds"] = time.time() - t0
    L2.write(a, out)
    log(f"done in {time.time() - t0:.1f}s")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


class ToyCarve7(L5.ToyCarve):
    """lean_mlp5's toy carve plus random AW walks, cos values and tokens."""

    def __init__(self, rows, seed, r=8, n_lab=20):
        super().__init__(rows, seed)
        rng = np.random.default_rng(seed + 100)
        self.ds = f"toy{seed % 2}"
        labs = np.r_[np.arange(n_lab), K_LAB]
        one = {k: [] for k in ("node", "dir", "lab", "b", "cos")}
        two = {k: [] for k in ("node", "d1", "l1", "d2", "l2", "b", "c1", "c2")}
        tok = {k: [] for k in ("d", "lab", "o", "cos")}
        n1, n2, nt = [], [], []
        for i in range(rows):
            n = int(self.n[i])
            a1, a2 = int(rng.integers(0, 8)), int(rng.integers(0, 15))
            one["node"].append(rng.integers(0, n, a1)), one["dir"].append(rng.integers(0, N_DIR, a1))
            one["lab"].append(rng.choice(labs, a1)), one["b"].append(rng.integers(0, 2, a1))
            one["cos"].append(np.where(one["lab"][-1] < K_LAB, rng.uniform(-0.2, 0.6, a1), 0.0).astype(np.float32))
            two["node"].append(rng.integers(0, n, a2))
            for dk, lk, ck in (("d1", "l1", "c1"), ("d2", "l2", "c2")):
                two[dk].append(rng.integers(0, N_DIR, a2))
                two[lk].append(rng.choice(labs, a2))
                two[ck].append(np.where(two[lk][-1] < K_LAB, rng.uniform(-0.2, 0.6, a2), 0.0).astype(np.float32))
            two["b"].append(rng.integers(0, 2, a2))
            tk = np.unique(np.r_[one["dir"][-1] * (K_LAB + 1) + one["lab"][-1], two["d2"][-1] * (K_LAB + 1) + two["l2"][-1]])
            tok["d"].append(tk // (K_LAB + 1)), tok["lab"].append(tk % (K_LAB + 1)), tok["o"].append(rng.uniform(size=tk.size) < 0.3)
            tok["cos"].append(np.where(tok["lab"][-1] < K_LAB, rng.uniform(-0.2, 0.6, tk.size), 0.0).astype(np.float32))
            n1.append(a1), n2.append(a2), nt.append(tk.size)
        self.aw1 = {k: np.concatenate(v).astype(np.float32 if k == "cos" else np.int64) for k, v in one.items()}
        self.aw2 = {k: np.concatenate(v).astype(np.float32 if k in ("c1", "c2") else np.int64) for k, v in two.items()}
        self.tok = {k: np.concatenate(v).astype(np.float32 if k == "cos" else (bool if k == "o" else np.int64)) for k, v in tok.items()}
        self.aw_off1 = np.concatenate([[0], np.cumsum(n1)]).astype(np.int64)
        self.aw_off2 = np.concatenate([[0], np.cumsum(n2)]).astype(np.int64)
        self.tok_off = np.concatenate([[0], np.cumsum(nt)]).astype(np.int64)
        Z = rng.standard_normal((K_LAB + 1, r)).astype(np.float32)
        Z[K_LAB] = 0
        self.aw_Z = Z
        self.aw_nr = False
        self.widths["AW"] = AW_W


class _FakeLab:
    """Every stored pair labelled, by a fixed hash of its ends."""
    ok = True

    def lookup(self, gu, gv):
        return ((gu * 7 + gv * 3) % 6).astype(np.int32)


def _brute_tokens(n, eu, ev, efam, fwd, bwd, S, bucket_of, gold):
    d = L4.famdir(efam, fwd, bwd)
    on = {}
    for j in range(eu.size):
        if int(eu[j]) not in bucket_of:
            continue
        kids = [k for k in range(eu.size) if int(eu[k]) == int(ev[j]) and int(ev[k]) != int(eu[j])]
        t1 = (int(d[j]), K_LAB)
        on[t1] = on.get(t1, False) or bool(gold[ev[j]]) or any(bool(gold[ev[k]]) for k in kids)
        for k in kids:
            t2 = (int(d[k]), K_LAB)
            on[t2] = on.get(t2, False) or bool(gold[ev[k]])
    return sorted((a, b, o) for (a, b), o in on.items())


def selftest():
    import tempfile
    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    # 1. walk_entries7 = lean_mlp4.walk_entries; parents; struct_only = walk_entries on the structural edges; tokens
    for trial in range(40):
        n = int(rng.integers(3, 25))
        m = int(rng.integers(0, 70))
        eu, ev = rng.integers(0, n, m), rng.integers(0, n, m)
        ok = eu != ev
        eu, ev = eu[ok], ev[ok]
        fam = rng.integers(0, 3, eu.size).astype(np.int8)
        fwd = (rng.uniform(size=eu.size) < 0.6) & (fam == 0)
        bwd = ((rng.uniform(size=eu.size) < 0.6) | ~fwd) & (fam == 0)
        S = rng.choice(n, size=min(n, int(rng.integers(1, 5))), replace=False)
        seeds = np.r_[S, -np.ones(10 - S.size, np.int64)]
        buckets = np.r_[rng.integers(0, 2, S.size), -np.ones(10 - S.size, np.int64)]
        mk = (lambda: {"fwd_flag": 0, "fwd_found": 0, "bwd_flag": 0, "bwd_found": 0, "missing": 0})
        one4, two4, _t = L4.walk_entries(n, np.arange(n), eu, ev, fam, fwd, bwd, seeds, buckets, None, mk())
        one, two, par, _t = walk_entries7(n, np.arange(n), eu, ev, fam, fwd, bwd, seeds, buckets, None, mk())
        for x, y in zip(one4 + two4, one + two):
            assert np.array_equal(x, y), ("walk_entries7 differs", trial)
        assert par.size == two[0].size and (par.size == 0 or (np.all(np.diff(par) >= 0) and par.max() < one[0].size))
        assert np.array_equal(one[1][par], two[1]) and np.array_equal(one[2][par], two[2]) and np.array_equal(one[3][par], two[5])
        st = fam == 0
        o4s, t4s, _t = L4.walk_entries(n, np.arange(n), eu[st], ev[st], fam[st], fwd[st], bwd[st], seeds, buckets, None, mk())
        o7s, t7s, _p, _t = walk_entries7(n, np.arange(n), eu, ev, fam, fwd, bwd, seeds, buckets, None, mk(), struct_only=True)
        for x, y in zip(o4s + t4s, o7s + t7s):
            assert np.array_equal(x, y), ("struct_only", trial)
        gold = (rng.uniform(size=n) < 0.25).astype(np.int8)
        bucket_of = {}
        for s_, b_ in zip(S.tolist(), buckets[:S.size].tolist()):
            bucket_of.setdefault(s_, b_)
        td, tl, to = row_tokens(gold, one, two, par)
        assert sorted(zip(td.tolist(), tl.tolist(), to.tolist())) == _brute_tokens(n, eu, ev, fam, fwd, bwd, S, bucket_of, gold), trial
        # masking: the masked walks' tokens are the tokens with the held labels read OTHER, merged by OR
        oL, wL, pL, _t = walk_entries7(n, np.arange(n), eu, ev, fam, fwd, bwd, seeds, buckets, _FakeLab(), mk())
        hold = np.array([1, 4])
        om, wm, nm = mask_walks(oL, wL, hold)
        assert nm == int(np.isin(oL[2], hold).sum() + np.isin(wL[2], hold).sum() + np.isin(wL[4], hold).sum())
        assert not (np.isin(om[2], hold).any() or np.isin(wm[2], hold).any() or np.isin(wm[4], hold).any())
        assert all(np.array_equal(x, y) for j, (x, y) in enumerate(zip(oL, om)) if j != 2)
        assert all(np.array_equal(x, y) for j, (x, y) in enumerate(zip(wL, wm)) if j not in (2, 4))
        tdm, tlm, tom = row_tokens(gold, om, wm, pL)
        td0, tl0, to0 = row_tokens(gold, oL, wL, pL)
        exp = {}
        for d_, l_, o_ in zip(td0.tolist(), np.where(np.isin(tl0, hold), K_LAB, tl0).tolist(), to0.tolist()):
            exp[(d_, l_)] = exp.get((d_, l_), False) or o_
        assert sorted(zip(tdm.tolist(), tlm.tolist(), tom.tolist())) == sorted((k_[0], k_[1], v_) for k_, v_ in exp.items()), trial
    # 2. KB labels: ranks by count (ties by id), a pair's most frequent relation, misses, the text table, a wrong vocabulary
    global KB_RELS
    old_rels = KB_RELS
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        served, KB_RELS = tmp / "served", tmp / "rels"
        (served / "metaqa" / "graph").mkdir(parents=True)
        KB_RELS.mkdir()
        vocab = ["r0", "r1", "r2", "r3"]
        src = np.array([0, 0, 1, 2, 3, 3, 4, 5, 6], np.int32)
        dst = np.array([1, 1, 2, 3, 4, 4, 5, 6, 7], np.int32)
        rel = np.array([2, 1, 1, 1, 0, 2, 2, 3, 0], np.int16)   # counts: r0 2, r1 3, r2 3, r3 1 -> ranks r1 0, r2 1, r0 2, r3 3
        np.savez(served / "metaqa" / "graph" / "structural.npz", src=src, dst=dst, rel=rel)
        (served / "metaqa" / "graph" / "GRAPH_MANIFEST.json").write_text(json.dumps(
            {"families": {"structural": {"relation_vocabulary": vocab, "n_edges": 9}}}), encoding="utf-8")
        emb = rng.standard_normal((4, 1536)).astype(np.float16)
        np.save(KB_RELS / "metaqa_rel_embeddings.npy", emb)
        (KB_RELS / "metaqa_rel_vocab.json").write_text(json.dumps(
            {"vocab": vocab, "texts": vocab, "sha256": hashlib.sha256("\n".join(vocab).encode("utf-8")).hexdigest()}), encoding="utf-8")

        class _DS:
            def __init__(self, ds, root):
                self.n_nodes = 8

        class _Canon:
            Dataset = _DS

        L = KBLabels("metaqa", served, _Canon)
        got = L.lookup(np.array([0, 1, 2, 3, 4, 5, 6, 1, 7]), np.array([1, 2, 3, 4, 5, 6, 7, 0, 0]))
        # (0,1): r2 (rank 1) and r1 (rank 0) -> 0; (1,2) r1 -> 0; (2,3) r1 -> 0; (3,4): r0 (2), r2 (1) -> 1; (4,5) r2 -> 1;
        # (5,6) r3 -> 3; (6,7) r0 -> 2; (1,0) and (7,0) not stored -> -1
        assert got.tolist() == [0, 0, 0, 1, 1, 3, 2, -1, -1], got.tolist()
        assert L.n_real == 4 and np.allclose(L.phrase[:4], emb[[1, 2, 0, 3]].astype(np.float32)) and not L.phrase[4:].any()
        T = text_table(L)
        assert np.allclose(np.linalg.norm(T[:4], axis=1), 1.0, atol=1e-5) and not T[4:].any()
        assert L.census["pairs_with_several"] == 2
        assert not KBLabels("squad", served, _Canon).ok
        (KB_RELS / "metaqa_rel_vocab.json").write_text(json.dumps({"vocab": vocab[::-1], "texts": vocab, "sha256": "x"}), encoding="utf-8")
        try:
            KBLabels("metaqa", served, _Canon)
            raise AssertionError("a wrong vocabulary must be refused")
        except SystemExit:
            pass
    KB_RELS = old_rels
    # 3. aw_batch7 carries aw_batch's tensors; NR zeroes labels and cos
    tc = ToyCarve7(12, 3)
    qs = np.array([0, 2, 5, 11])
    A4, A7 = L4.aw_batch(tc, qs), aw_batch7(tc, qs)
    for k, v in A4.items():
        assert (A7[k] == v) if isinstance(v, int) else torch.equal(A7[k], v), k
    tc.aw_nr = True
    A7n = aw_batch7(tc, qs)
    assert bool((A7n["l1"] == K_LAB).all()) and float(A7n["c2b"].abs().sum()) == 0.0 and float(A7n["tc"].abs().sum()) == 0.0
    tc.aw_nr = False
    # aw_mask: the held labels read OTHER with cos 0, every other entry as it was
    tc.hold = np.array([0, 3])
    tc.aw_mask = True
    A7m = aw_batch7(tc, qs)
    tc.aw_mask = False
    n_held = 0
    for lk, ck in (("l1", "c1"), ("l2a", "c2a"), ("l2b", "c2b"), ("tl", "tc")):
        h = torch.isin(A7[lk], torch.tensor([0, 3]))
        n_held += int(h.sum())
        assert torch.equal(A7m[lk], torch.where(h, torch.full_like(A7[lk], K_LAB), A7[lk])), lk
        assert torch.equal(A7m[ck], torch.where(h, torch.zeros_like(A7[ck]), A7[ck])), ck
    assert n_held > 0
    for k, v in A7.items():
        if k not in ("l1", "c1", "l2a", "c2a", "l2b", "c2b", "tl", "tc"):
            assert (A7m[k] == v) if not torch.is_tensor(v) else torch.equal(A7m[k], v), k
    # 4. AWNet7 (aw) = AWNet op for op, in training (the same dropout draws) and in eval; awc at kappa 0 = aw
    A = aw_batch7(tc, np.arange(tc.rows))
    for train_mode in (True, False):
        n4, n7, n7c = L4.AWNet(8, 16, 0, 0.25, seed=3), AWNet7(8, 16, 0, 0.25, seed=3, arm="aw"), AWNet7(8, 16, 0, 0.25, seed=3, arm="awc")
        for net in (n4, n7, n7c):
            net.train(train_mode)
        torch.manual_seed(11)
        f4 = n4(A)
        torch.manual_seed(11)
        f7 = n7(A)
        torch.manual_seed(11)
        f7c = n7c(A)
        assert torch.equal(f4, f7) and torch.equal(f4, f7c), train_mode
    with torch.no_grad():
        n7c.kappa.fill_(2.0)
    assert not torch.equal(n7c(A), f4), "kappa must move the columns"
    # em at eta 0 = awc op for op (the typer has its own generator and draws nothing in the forward)
    for train_mode in (True, False):
        nc, ne = AWNet7(8, 16, 0, 0.25, seed=3, arm="awc"), AWNet7(8, 16, 0, 0.25, seed=3, arm="em", k_types=4)
        nc.train(train_mode)
        ne.train(train_mode)
        torch.manual_seed(11)
        fc = nc(A)
        torch.manual_seed(11)
        fe_ = ne(A)
        assert torch.equal(fc, fe_), ("em at eta 0 must be awc", train_mode)
    with torch.no_grad():
        ne.eta.fill_(0.5)
    assert not torch.equal(ne(A), fc), "eta must move the columns"
    # 5. the fit: arm aw with a base config = lean_mlp4.fit4 (fixed, rf) bit for bit; best/swa/last
    L3.LeanMLP3.dim = LM.PROJ_DIM
    L4.LeanMLP4.cfg = {"r": 8, "D": 16, "k_res": 0, "p_drop": 0.25}
    LeanMLP7.cfg = {"r": 8, "D": 16, "k_res": 0, "p_drop": 0.25, "k_types": 4}
    tr, se = [ToyCarve7(60, 1), ToyCarve7(40, 4)], [ToyCarve7(30, 2)]
    blocks = ["rank", "WALK", "AW"]
    old_batch = LM.batch_of
    LM.batch_of = L4.batch_of4
    ref, _curve = L4.fit4(tr, se, blocks, "fixed", 4, 3e-2, 0, 32, "rf")
    LM.batch_of = batch_of7
    base = L5.parse_configs("b=3e-2:1e-4:0.1:4:1")["b"]
    f = fit7(tr, se, blocks, "aw", base, 0, 32)
    for k, v in ref.state_dict().items():
        assert torch.equal(v, f["best"][k]), f"fit7 (aw) differs from lean_mlp4.fit4 on {k}"
    assert set(ref.state_dict()) == set(f["best"]), "arm aw adds no parameter"
    # 6. aux: the token loss reaches the AW scorer; em: the M step reaches the typer alone, the score reads it detached
    emc = {"lam0": 1.0, "an": 2, "mu": 1.0, "omega": 1.0}
    fa = fit7(tr, se, blocks, "aux", base, 0, 32)
    fe = fit7(tr, se, blocks, "em", base, 0, 32, emc)
    fe0 = fit7(tr, se, blocks, "em0", base, 0, 32, emc)
    assert all(rec["extra"] is not None for rec in fa["curve"]) and all("em" in rec for rec in fe["curve"])
    assert [rec["em"]["lambda"] for rec in fe["curve"]] == [1.0, 0.5, 0.0, 0.0] and all(rec["em"]["lambda"] == 0.0 for rec in fe0["curve"])
    net = AWNet7(8, 16, 0, 0.25, seed=0, arm="em", k_types=4)
    net.train()
    em = EMState(4, ["toy0", "toy1"], 1.0, 2)
    Ab = aw_batch7(tr[0], np.arange(20))
    loss = net.em_loss(Ab, em, 1.0)
    loss.backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in (net.typer.f1, net.typer.f2, net.typer.Wt, net.typer.E))
    assert all(p.grad is None for p in (net.P, net.Wq, net.dirv, net.bias, net.kappa, net.eta)), "the M step reached the scorer"
    assert abs(float(em.C[Ab["g"]].sum()) - float((Ab["tl"] < K_LAB).sum())) < 1e-6, "theta counts the labelled tokens' posteriors"
    assert float(em.C["toy0" if Ab["g"] == "toy1" else "toy1"].sum()) == 0.0, "theta is per graph"
    net.zero_grad()
    net(Ab).sum().backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in net.typer.parameters()), "the score reached the typer"
    assert net.eta.grad is not None and net.kappa.grad is not None
    assert em.lam(0) == 1.0 and em.lam(1) == 0.5 and em.lam(2) == 0.0 and em.lam(5) == 0.0
    lt = em.log_theta("toy0")
    assert torch.allclose(lt.exp().sum(0), torch.ones(4), atol=1e-4)
    na = AWNet7(8, 16, 0, 0.25, seed=0, arm="aux")
    na.train()
    la = na.aux_loss(Ab)
    la.backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in (na.P, na.Wq, na.dirv, na.bias, na.kappa))
    # 7. ranks, AUC and the edge ranking on a model
    x = np.array([3.0, 1.0, 3.0, 2.0, 1.0])
    assert avg_rank(x).tolist() == [4.5, 1.5, 4.5, 3.0, 1.5]
    s = np.array([0.9, 0.1, 0.5, 0.2, 0.3, 0.3, 0.8, 0.1])
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0], bool)
    off = np.array([0, 4, 8])
    a_, ids_ = row_auc(s, y, off)
    assert a_.tolist() == [1.0, 0.5] and ids_.tolist() == [0, 1], a_
    assert row_auc(s, np.zeros(8, bool), off)[0].size == 0
    m7 = build7(blocks, fe["widths"], 32, fe["swa"], "em")
    d = edge_diag(m7, se[0])
    assert all(0.0 <= d[k] <= 1.0 for k in ("auc_aw", "auc_r", "auc_blind", "auc_hidden")) and d["rows_both"] > 0, d
    assert "auc_aw_held" not in d
    se[0].held = np.arange(se[0].rows) % 2 == 0
    dh = edge_diag(m7, se[0])
    del se[0].held
    assert {k: dh[k] for k in d} == d and all(dh[f"auc_{k}_held"] is None or 0.0 <= dh[f"auc_{k}_held"] <= 1.0
                                              for k in ("aw", "r", "blind", "hidden")) and 0 < dh["rows_both_held"] <= d["rows_both"], dh
    # 8. the held rows' summary; --hold parsing; the code basis' phrase rows
    rg = np.random.default_rng(0)
    R = {nm: rg.uniform(size=(10, 3)) for nm in ("ctl@best", "aw@best")}
    refs = {"twin0": rg.uniform(size=(10, 3)), "gnn0": rg.uniform(size=(10, 3)) + 0.5}
    hs = held_summary({"ctl@best": None, "aw@best": None}, R, {"aw@best": 0.9 * R["aw@best"]}, refs, np.arange(10) < 6, rg)
    assert hs["rows"] == 6 and {"REV_minus_MASK", "MASK_minus_twin0", "REV_minus_ctl"} <= set(hs["aw@best"]) and "MASK" not in hs["ctl@best"]
    assert abs(hs["aw@best"]["REV"][0] - R["aw@best"][:6, 0].mean()) < 1e-12
    assert held_summary({"aw@best": None}, R, {}, refs, np.arange(10) < 1, rg) == {"rows": 1}

    class _KL:
        ok = True
        vocab = ["directed_by", "has_genre", "written_by", "has tags"]
        rank_of = np.array([2, 0, 3, 1])

    labs = {"metaqa": _KL()}
    assert parse_hold("metaqa:mask:written_by+has_genre", labs, {"metaqa"}) == {
        "graph": "metaqa", "mode": "mask", "names": ["written_by", "has_genre"], "ranks": [0, 3]}
    assert parse_hold("metaqa:mask:has_tags;directed_by", labs, {"metaqa"})["ranks"] == [1, 2]
    for bad, trn in (("metaqa:drop:has_genre", {"metaqa"}), ("metaqa:mask:nope", {"metaqa"}), ("metaqa:mask:has_genre+has_genre", {"metaqa"}),
                     ("webqsp:mask:has_genre", {"webqsp"}), ("metaqa:mask:has_genre", {"2wiki"}), ("metaqa:mask", {"metaqa"})):
        try:
            parse_hold(bad, labs, trn)
            raise AssertionError(f"--hold {bad} with training {trn} must be refused")
        except SystemExit:
            pass
    old_anchors = dict(L4.ANCHORS)
    with tempfile.TemporaryDirectory() as tmp:
        P = rng.standard_normal((K_LAB + 40, 1536)).astype(np.float32)
        np.save(Path(tmp) / "phrase_emb_w2_K4096.npy", P)
        L4.ANCHORS["toyg"] = Path(tmp)
        try:
            assert np.array_equal(code_phrases("toyg", {}), P[:K_LAB]) and code_phrases("nope", {}) is None
        finally:
            L4.ANCHORS.clear()
            L4.ANCHORS.update(old_anchors)
    LM.batch_of = old_batch
    print("selftest ok: walk_entries7 = lean_mlp4's walks (and its structural-only form), parents, tokens = brute force; "
          "masked walks' tokens = the tokens with held labels OTHER, merged; KB labels (count ranks, a pair's most frequent "
          "relation, misses, text table, a wrong vocabulary refused); aw_batch7 carries aw_batch's tensors, NR and the mask; "
          "AWNet7 aw/awc(kappa 0) = AWNet op for op, em(eta 0) = awc; fit7 aw = fit4 bit for bit; the token loss reaches "
          "the scorer; the EM step reaches the typer alone and the score reads it detached; lambda anneals; theta counts; "
          "ranks, AUC and the held split; the held summary; --hold parsing; the code basis' phrase rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
