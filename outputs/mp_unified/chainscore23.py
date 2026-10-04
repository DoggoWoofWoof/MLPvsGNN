"""Design look (untracked; not a result and not filed): S6 part 7 of the transfer plan, the relation graph.
Part 5 (chainscore21, read 19:55 on 4 Oct): the text-cluster pseudo-relations type passages well within passages but
do not carry to KBs, and webqsp zero-shot stays at 0.29 R@5 (the untyped walk 0.20; a model trained on webqsp about
0.60). ULTRA (Galkin et al., ICLR 2024) transfers across KGs with unseen relation vocabularies by representing each
relation by how it meets the others: a graph over the relations whose edges say that two relations share heads, share
tails, or that one's tails are the other's heads. Here each row gets that graph over its own relation slots, so what a
chain's types are is learned from their structure in the row, never from an id or a name. Part 6 (chainscore22, the
population maps in training) runs beside it on the same device; a carried map is combined in a later part.

The row's relation graph. Its nodes are the row's slots (chainscore20's companion: the types of its chains' first two
steps at every level: the none slot, the two dir slots, each tt<k> cluster and each exact relation, one slot per
direction). A slot's messages are the row's messages of its type (none: all; dir: those of its direction; tt<k> and
exact: chainscore20's message-type incidence); its heads are their sources and its tails their targets. Between two
slots of one level, four kinds of edge, each weighted by the number of nodes shared:
    hh  both slots have the node as a head        tt  both have it as a tail
    ht  the node is a head of the first and a tail of the second, th the reverse
Each slot keeps its 16 strongest edges of each kind (ties to the lower slot), weights normalised to sum to one; a slot
has no edge to itself. Slot features (13), from the row's nodes and messages only:
    lm          log(1 + its messages) / log(1 + the row's messages)
    src_seed    share of its messages from a bucket-0 seed; dst_seed: into one
    cq_dst      mean query similarity (cq) of its distinct tails; cq_dst_max their maximum; cq_src of its heads
    cs_dst      mean cs of its distinct tails
    fan_out     log(messages / distinct heads); fan_in: log(messages / distinct tails)
    k_none, k_dir, k_tt, k_exact    its level's kind
The four text columns (cq_dst, cq_dst_max, cq_src, cs_dst) are what the rgs arm leaves out (set to 0 after
standardising): the structure and the seeds alone.
The model is part 5's em-gr with lb (MLP f and g, chainscore19's EM coupling, T = 2) and one added term in g: a chain's
prior logit g(x_c) becomes g(x_c) + q([h(s_1), h(s_2)]), where s_1, s_2 are its first two steps' slots (a missing step
reads a zero vector) and h is a GNN over the row's relation graph:
    h_0 = MLP(x_s)  (13 -> 32 -> 32);  h_l = h_{l-1} + ReLU(W_l h_{l-1} + b_l + M_l [agg_hh, agg_tt, agg_ht, agg_th]),
    l = 1, 2, agg_k the weighted mean of h_{l-1} over the slot's kind-k edges;  q: 64 -> 32 -> 1, its last layer zero.
The base weights are em-gr's for the seed (made first, by chainscore20's make_model); h and q are made after under seed
+ 7919, and q starts at 0, so every arm starts where its control does. Slot features are standardised by each
build's own slot moments: every population by its own distribution (the direction part 6 tests on the node and
chain features), the read build's own (unlabeled) moments at reading. b-lb-rgf uses the first training build's
moments for every build, the usual way, so the two differ only in that.
Arms. Training as parts 5 and 6 (Adam 1e-3, weight decay 1e-4, 12 epochs of 5,960 examples in batches of 64, the epoch
chosen by R@5 on the select carve), three seeds (SEED + 0, 1, 2), every run on the host GPU (cs_dev's settings:
deterministic algorithms, TF32 off):
    b-lb-rg     metaqa's fit carve and its four transforms with the relation graph
    b-lb-rgs    the same without the text columns
    b-lb-rgf    b-lb-rg with every build's slot features standardised by the fit build's moments
    mp-lb-rg    and 2wiki's typed fit carve, the dataset uniform per example
    b-lb-r0     seed 0 through this file's loop with the relation graph off (R0)
Controls: part 6's b-lb and mp-lb runs (chainscore22, the map none; the same device, seeds and statements).
Reads: metaqa x1f, webqsp selectf and webqsp selectf + fit (the KB rule); 2wiki x1 and hotpotqa x1 (the passage rule).
musique x1f is not read here (its companion is 5.7 GB, and part 5's typing gained nothing on it).
Verdicts, fixed at 21:49 on 4 Oct before any arm was trained or read on a real build (a smoke ran on stand-in builds).
Per arm, each row's R@5 averaged over its three seeds; paired row bootstrap (BOOT 1000):
    G1 (primary) webqsp selectf + fit: b-lb-rg minus b-lb: ABOVE / AT / BELOW
    G2 metaqa x1f: the same
    G3 webqsp selectf + fit by seed-gold distance D (1, 2): b-lb-rg, b-lb-rgs and b-lb-rgf minus b-lb
    G4 webqsp selectf + fit and metaqa x1f: b-lb-rgs and b-lb-rgf minus b-lb; b-lb-rg minus b-lb-rgs (the text
       columns' share) and b-lb-rg minus b-lb-rgf (the per-population standardisation's share)
    G5 the three KB reads: mp-lb-rg minus mp-lb, and mp-lb-rg minus b-lb-rg (with the relation graph, does typed 2wiki
       in training still cost the KB read, part 5's Q1)
    G6 the two passage reads: each relation-graph arm minus its control; every arm minus s0+rrf with its share of the
       twin's lead over s0+rrf (HIGH at or above 0.5, LOW below 0.25, else MID)
    G7 webqsp selectf (305 rows): G1's difference; webqsp selectf + fit: every arm minus none/rd (the untyped walk)
    S  every difference above per seed, with its sign
    R0 b-lb-r0 equals part 6's b-lb seed 0 row for row (|diff| < 1e-6) on every read: REPRODUCES / DIFFERS
Decision. One of b-lb-rg, b-lb-rgs and b-lb-rgf carries (the largest webqsp selectf + fit mean difference if
several) when its difference from b-lb is ABOVE on webqsp selectf + fit and not BELOW on metaqa x1f (G1, G2, G4);
with b-lb-rg, G5's mp-lb-rg minus b-lb-rg not BELOW means passages in training no longer cost the KB read. If none
carries, the result goes to Swastik with part 6's. R0 DIFFERS: nothing is read until its cause is found. Train-split rows
throughout: a look, not a result.
    python outputs/mp_unified/chainscore23.py rgbuild --base outputs/mp_unified/lean/cs19-mq-fit-id.npz \
        --edges outputs/mp_unified/lean/cs20e-mq-fit-id.npz --out outputs/mp_unified/lean/cs23r-mq-fit-id.npz
    python outputs/mp_unified/chainscore23.py train --arm b-lb-rg --seed 0 --device cuda \
        --fit outputs/mp_unified/lean/cs19-mq-fit-id.npz --aug al4=outputs/mp_unified/lean/cs19-mq-fit-al4.npz (sp4, mg3,
        rf) --select outputs/mp_unified/lean/cs19-mq-select.npz (mp: --pfit outputs/mp_unified/lean/cs21-2w-fit.npz) \
        --read metaqa=... --read webqsp=... --read webqsp_sf=... --pread 2wiki=... --pread hotpotqa=... \
        --rg fit=outputs/mp_unified/lean/cs23r-mq-fit-id.npz (one for every build named) \
        --out outputs/mp_unified/lean/cs23-b-lb-rg-s0.json --rows-out outputs/mp_unified/lean/cs23-b-lb-rg-s0.rows.npz
    python outputs/mp_unified/chainscore23.py grade --run outputs/mp_unified/lean/cs23-b-lb-rg-s0.json (each run) \
        --control outputs/mp_unified/lean/cs22-b-lb-s0.json (b-lb and mp-lb, each seed) --read ... --pread ... \
        --out outputs/mp_unified/lean/cs23.json
    python outputs/mp_unified/chainscore23.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainscore19's count
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainpop17 as CP  # noqa: E402
import chainscore18 as CS  # noqa: E402
import chainscore19 as C19  # noqa: E402
import chainscore20 as C20  # noqa: E402
import chainscore21 as C21  # noqa: E402
import cs_dev as CD  # noqa: E402

SEED, BOOT, BATCH = C19.SEED, C19.BOOT, C21.BATCH
F = "em-gr"
TOPK = 16
KINDS = ("hh", "tt", "ht", "th")
SLOTF = ("lm", "src_seed", "dst_seed", "cq_dst", "cq_dst_max", "cq_src", "cs_dst", "fan_out", "fan_in",
         "k_none", "k_dir", "k_tt", "k_exact")
TEXTC = [SLOTF.index(c) for c in ("cq_dst", "cq_dst_max", "cq_src", "cs_dst")]
H, NL, RG_SEED = 32, 2, 7919
ARMS = ("b-lb-r0", "b-lb-rg", "b-lb-rgs", "b-lb-rgf", "mp-lb-rg")
RG_ARMS = ("b-lb-rg", "b-lb-rgs", "b-lb-rgf")
RUNS = [("b-lb-r0", 0)] + [(a, s) for a in ARMS[1:] for s in (0, 1, 2)]
CONTROLS = ("b-lb", "mp-lb")
SEEDS = (0, 1, 2)
KB_READS, P_READS = C21.KB_READS, ("2wiki", "hotpotqa")
CQ, CSI = C19.NODEF.index("cq"), C19.NODEF.index("cs")
BKEYS = ("n", "xn_off", "XN", "seed0", "rowoff")
EKEYS = ("n", "e_off", "e_src", "e_dst", "e_d", "s_off", "slots", "p_slot", "i_off", "i_e", "i_s", "sl_none",
         "sl_dir")
log, sha = C21.log, C21.sha
ABV = ("ABOVE", "BELOW", "AT")


def arm_parts(arm):
    """(chainscore21's arm, the variant): b-lb-rgs -> (b-lb, rgs)."""
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm}; one of {ARMS}")
    p = arm.split("-")
    return "-".join(p[:2]), p[2]


# ── the relation-graph companion ────────────────────────────────────────────────────────────────────────────────


def load_keys(p, keys):
    with np.load(p, allow_pickle=False) as z:
        miss = [k for k in keys if k not in z.files]
        if miss:
            raise SystemExit(f"{p}: no {miss}")
        d = {k: z[k] for k in keys}
        d["meta"] = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return d


def kind_of(name):
    return 0 if name == "none" else 1 if name == "dir" else 3 if name == "exact" else 2


def topk_rows(M, k):
    """Each row's k largest entries (ties to the lower column), the diagonal dropped: (row, col, value)."""
    M = M.tocoo()
    keep = M.row != M.col
    r, c, v = M.row[keep].astype(np.int64), M.col[keep].astype(np.int64), M.data[keep].astype(np.float64)
    if r.size == 0:
        return r, c, v
    o = np.lexsort((c, -v, r))
    r, c, v = r[o], c[o], v[o]
    first = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    pos = np.arange(r.size) - np.repeat(first, np.diff(np.r_[first, r.size]))
    m = pos < k
    return r[m], c[m], v[m]


def row_graph(b, c, slev, skind, i):
    """Row i's slot features (S x 13) and its edges per kind (local a, local b, normalised weight)."""
    n = int(b["n"][i])
    s0, s1 = int(c["s_off"][i]), int(c["s_off"][i + 1])
    S = s1 - s0
    X = np.zeros((S, len(SLOTF)))
    E = {k: (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)) for k in KINDS}
    if S == 0:
        return X, E
    e0, e1 = int(c["e_off"][i]), int(c["e_off"][i + 1])
    m = e1 - e0
    es = c["e_src"][e0:e1].astype(np.int64)
    ee = c["e_dst"][e0:e1].astype(np.int64)
    ed = c["e_d"][e0:e1].astype(np.int64)
    i0, i1 = int(c["i_off"][i]), int(c["i_off"][i + 1])
    ie, is_ = [c["i_e"][i0:i1].astype(np.int64)], [c["i_s"][i0:i1].astype(np.int64)]
    sn = int(c["sl_none"][i])
    if sn >= 0:
        ie.append(np.arange(m))
        is_.append(np.full(m, sn))
    for dd in (0, 1):
        sd = int(c["sl_dir"][i][dd])
        if sd >= 0:
            ix = np.flatnonzero(ed == dd)
            ie.append(ix)
            is_.append(np.full(ix.size, sd))
    ie, is_ = np.concatenate(ie), np.concatenate(is_)
    lev = slev[s0:s1]
    X[np.arange(S), 9 + skind[s0:s1]] = 1.0
    if ie.size == 0 or n == 0:
        return X, E
    Hk = np.unique(is_ * n + es[ie])
    Tk = np.unique(is_ * n + ee[ie])
    hs, hn, ts, tn = Hk // n, Hk % n, Tk // n, Tk % n
    Mh = sp.csr_matrix((np.ones(Hk.size), (hs, hn)), shape=(S, n))
    Mt = sp.csr_matrix((np.ones(Tk.size), (ts, tn)), shape=(S, n))
    for name, A, B in (("hh", Mh, Mh), ("tt", Mt, Mt), ("ht", Mh, Mt), ("th", Mt, Mh)):
        P_ = (A @ B.T).tocoo()
        ok = lev[P_.row] == lev[P_.col]
        r, cc, v = topk_rows(sp.coo_matrix((P_.data[ok], (P_.row[ok], P_.col[ok])), shape=(S, S)), TOPK)
        tot = np.bincount(r, weights=v, minlength=S)
        E[name] = (r, cc, v / np.maximum(tot[r], 1e-12))
    a = int(b["xn_off"][i])
    cq = b["XN"][a:a + n, CQ].astype(np.float64)
    cs = b["XN"][a:a + n, CSI].astype(np.float64)
    seed = b["seed0"][a:a + n].astype(np.float64)
    cnt = np.bincount(is_, minlength=S).astype(np.float64)
    nh = np.bincount(hs, minlength=S).astype(np.float64)
    nt = np.bincount(ts, minlength=S).astype(np.float64)
    X[:, 0] = np.log1p(cnt) / np.log1p(max(m, 1))
    X[:, 1] = np.bincount(is_, weights=seed[es[ie]], minlength=S) / np.maximum(cnt, 1)
    X[:, 2] = np.bincount(is_, weights=seed[ee[ie]], minlength=S) / np.maximum(cnt, 1)
    X[:, 3] = np.bincount(ts, weights=cq[tn], minlength=S) / np.maximum(nt, 1)
    mx = np.full(S, -np.inf)
    np.maximum.at(mx, ts, cq[tn])
    X[:, 4] = np.where(nt > 0, mx, 0.0)
    X[:, 5] = np.bincount(hs, weights=cq[hn], minlength=S) / np.maximum(nh, 1)
    X[:, 6] = np.bincount(ts, weights=cs[tn], minlength=S) / np.maximum(nt, 1)
    X[:, 7] = np.where(nh > 0, np.log(np.maximum(cnt, 1) / np.maximum(nh, 1)), 0.0)
    X[:, 8] = np.where(nt > 0, np.log(np.maximum(cnt, 1) / np.maximum(nt, 1)), 0.0)
    return X, E


def rgbuild_cmd(a):
    t0 = time.time()
    bsha, esha = sha(a.base), sha(a.edges)
    b = load_keys(a.base, BKEYS)
    c = load_keys(a.edges, EKEYS)
    if c["meta"].get("look") != "chainscore20-edges" or c["meta"].get("base_sha256") != bsha:
        raise SystemExit(f"{a.edges}: not chainscore20's companion of {a.base}")
    if not np.array_equal(c["n"].astype(np.int64), b["n"].astype(np.int64)):
        raise SystemExit("the companion's pool sizes are not the build's")
    P = int(b["rowoff"][-1])
    ps = c["p_slot"].astype(np.int64) - 1
    if ps.shape[0] != P:
        raise SystemExit(f"{ps.shape[0]} chains in the companion, the build's {P}")
    if ps.shape[1] < 2:
        ps = np.c_[ps, np.full((P, 2 - ps.shape[1]), -1, np.int64)]
    lev_off = np.asarray(c["meta"]["lev_off"], np.int64)
    names = [lv["name"] for lv in c["meta"]["levels"]]
    g = c["slots"].astype(np.int64)
    slev = np.searchsorted(lev_off, g, side="right") - 1
    skind = np.asarray([kind_of(x) for x in names], np.int64)[slev]
    N = int(b["n"].size)
    Xs, cnt = [], {k: [] for k in KINDS}
    Ea, Eb, Ew = ({k: [] for k in KINDS} for _ in range(3))
    for i in range(N):
        X, E = row_graph(b, c, slev, skind, i)
        Xs.append(X.astype(np.float32))
        for k in KINDS:
            Ea[k].append(E[k][0].astype(np.int32))
            Eb[k].append(E[k][1].astype(np.int32))
            Ew[k].append(E[k][2].astype(np.float32))
            cnt[k].append(E[k][0].size)
        if (i + 1) % 1000 == 0:
            log(f"  {i + 1}/{N} rows ({time.time() - t0:.0f}s)")
    arr = {"n": b["n"].astype(np.int32), "s_off": c["s_off"].astype(np.int64),
           "x_s": np.concatenate(Xs) if Xs else np.zeros((0, len(SLOTF)), np.float32),
           "p_slot": ps[:, :2].astype(np.int32)}
    for k in KINDS:
        arr[f"{k}_off"] = np.r_[0, np.cumsum(cnt[k])].astype(np.int64)
        arr[f"{k}_a"] = np.concatenate(Ea[k]) if N else np.zeros(0, np.int32)
        arr[f"{k}_b"] = np.concatenate(Eb[k]) if N else np.zeros(0, np.int32)
        arr[f"{k}_w"] = np.concatenate(Ew[k]) if N else np.zeros(0, np.float32)
    st = {"rows": N, "slots": int(arr["x_s"].shape[0]), "chains": P,
          "edges": {k: int(arr[f"{k}_a"].size) for k in KINDS},
          "slots_per_row": [round(float(x), 2) for x in np.percentile(np.diff(arr["s_off"]), [50, 90, 100])]
          if N else [], "seconds": round(time.time() - t0, 1)}
    meta = {"look": "chainscore23-rg", "base": a.base, "base_sha256": bsha, "edges": a.edges, "edges_sha256": esha,
            "script_sha256": sha(__file__), "slotf": list(SLOTF), "topk": TOPK, "levels": names,
            "lev_off": lev_off.tolist(), "stats": st}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.asarray(json.dumps(meta)), **arr)
    os.replace(tmp, p)
    log(f"rgbuild {a.out}: {st}")
    return meta


def load_rg(p, d_sha, d):
    """A relation-graph companion, checked against the build it was made on (d, its file's hash d_sha)."""
    r = CS.load(p)
    if r["meta"].get("look") != "chainscore23-rg":
        raise SystemExit(f"{p}: not a chainscore23 relation-graph companion")
    if r["meta"]["base_sha256"] != d_sha:
        raise SystemExit(f"{p}: made on a build with hash {r['meta']['base_sha256'][:12]}, not {d_sha[:12]}")
    if not np.array_equal(r["n"].astype(np.int64), d["n"].astype(np.int64)) or \
            r["p_slot"].shape[0] != int(d["rowoff"][-1]):
        raise SystemExit(f"{p}: its rows or chains are not the build's")
    return r


def rg_stats(r):
    X = r["x_s"].astype(np.float64)
    m, s = X.mean(0), X.std(0)
    s[s < 1e-6] = 1.0
    return {"m": m.astype(np.float32), "s": s.astype(np.float32)}


# ── the model ───────────────────────────────────────────────────────────────────────────────────────────────────


def attach(model, seed):
    """g(x) -> g(x) + q([h(s_1), h(s_2)]), h the relation-graph GNN; made after the base under seed + RG_SEED."""
    import torch
    nn = torch.nn

    class RG(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(len(SLOTF), H), nn.ReLU(), nn.Linear(H, H))
            self.W = nn.ModuleList([nn.Linear(H, H) for _ in range(NL)])
            self.M = nn.ModuleList([nn.Linear(H * len(KINDS), H, bias=False) for _ in range(NL)])

        def forward(self, ctx):
            h = self.inp(ctx["X"])
            for layer in range(NL):
                aggs = [torch.zeros_like(h).index_add(0, ea, torch.index_select(h, 0, eb) * ew[:, None])
                        for ea, eb, ew in ctx["E"]]
                h = h + torch.relu(self.W[layer](h) + self.M[layer](torch.cat(aggs, 1)))
            return h

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(2 * H, H)
            self.l2 = nn.Linear(H, 1)
            nn.init.zeros_(self.l2.weight)
            nn.init.zeros_(self.l2.bias)

        def forward(self, h, ps):
            hx = torch.cat([h, h.new_zeros((1, H))], 0)
            pair = torch.cat([torch.index_select(hx, 0, ps[:, 0]), torch.index_select(hx, 0, ps[:, 1])], 1)
            return self.l2(torch.relu(self.l1(pair)))

    class GWrap(nn.Module):
        def __init__(self, g0):
            super().__init__()
            self.g0, self.rg, self.head = g0, RG(), Head()
            self.ctx = None

        def forward(self, Xp):
            out = self.g0(Xp)
            if self.ctx is None:
                return out
            return out + self.head(self.rg(self.ctx), self.ctx["ps"])

    torch.manual_seed(seed + RG_SEED)
    model.g = GWrap(model.g)
    return model


def set_ctx(model, bt):
    if hasattr(model.g, "ctx"):
        model.g.ctx = bt["rg"]


def make_batch_rg(items, builds, rgs, LB, st, rsts, variant, train=True):
    """make_batch21's batch and the rows' relation graphs as flat slot rows: X (each build's rows by rsts[build];
    rgs zeroes the text columns), per kind (a, b, w) and each chain's two step slots (the sentinel S for a missing
    step), in Xp's order."""
    import torch
    bt = C21.make_batch21(items, builds, [None] * len(builds), LB, st, pairs=True, train=train)
    t = torch.from_numpy
    xs, ps = [], []
    E = {k: ([], [], []) for k in KINDS}
    soff = 0
    for bi, i, _ri in np.asarray(items, np.int64).reshape(-1, 3):
        r = rgs[bi]
        s0, s1 = int(r["s_off"][i]), int(r["s_off"][i + 1])
        xs.append((r["x_s"][s0:s1] - rsts[bi]["m"]) / rsts[bi]["s"])
        for k in KINDS:
            o0, o1 = int(r[f"{k}_off"][i]), int(r[f"{k}_off"][i + 1])
            E[k][0].append(r[f"{k}_a"][o0:o1].astype(np.int64) + soff)
            E[k][1].append(r[f"{k}_b"][o0:o1].astype(np.int64) + soff)
            E[k][2].append(r[f"{k}_w"][o0:o1])
        p0, p1 = int(builds[bi]["rowoff"][i]), int(builds[bi]["rowoff"][i + 1])
        pl = r["p_slot"][p0:p1].astype(np.int64)
        ps.append(np.where(pl >= 0, pl + soff, -1))
        soff += s1 - s0
    X = np.concatenate(xs).astype(np.float32) if xs else np.zeros((0, len(SLOTF)), np.float32)
    if variant == "rgs":
        X[:, TEXTC] = 0.0
    P = np.concatenate(ps) if ps else np.zeros((0, 2), np.int64)
    P[P < 0] = soff
    bt["rg"] = {"X": t(X), "ps": t(P),
                "E": [(t(np.concatenate(E[k][0])), t(np.concatenate(E[k][1])),
                       t(np.concatenate(E[k][2]).astype(np.float32))) for k in KINDS]}
    return bt


def read_rg(model, d, r, lb, st, rst, rule, variant, device=None, ri=0, batch=BATCH):
    """cs_dev's read_dev with the relation graph in the batch."""
    import torch
    rank = {"kb": C19.rank_row, "passage": C21.rank_row_p}[rule]
    N = d["n"].size
    out = np.zeros((N, 3))
    L = d["level_names"].size
    ml = np.zeros(L)
    plev = d["p_lev"].astype(np.int64)
    with torch.no_grad():
        for b0 in range(0, N, batch):
            rows = np.arange(b0, min(N, b0 + batch))
            items = np.c_[np.zeros(rows.size, np.int64), rows, np.full(rows.size, ri)]
            bt = make_batch_rg(items, [d], [r], None if lb is None else [lb], st, [rst], variant, train=False)
            if CD.placed(device):
                bt = CD.to_dev(bt, device)
                set_ctx(model, bt)
                with torch.device(device):
                    s, lp = C20.forward20(model, bt)
            else:
                set_ctx(model, bt)
                s, lp = C20.forward20(model, bt)
            s = s.double().cpu().numpy()
            Pm = None if lp is None else np.exp(lp.double().cpu().numpy())
            for j, i in enumerate(rows):
                out[i] = rank(d, i, s[j, :int(d["n"][i])])
                if Pm is not None:
                    p0, p1 = int(d["rowoff"][i]), int(d["rowoff"][i + 1])
                    if p1 > p0:
                        ml += np.bincount(plev[p0:p1], weights=Pm[j, :p1 - p0], minlength=L)
    return out, ml / max(N, 1)


def train_rg(arm, builds, rgs, LB, groups, Sd, Sr, Slb, srule, epochs, per_epoch, batch, seed, device=None,
             threads=1):
    """cs_dev's train_dev with the relation graph: the same statements, the batch made by make_batch_rg and the
    model's g given each batch's relation graphs. b-lb-r0 attaches nothing (train_dev's model and steps)."""
    import torch
    base, variant = arm_parts(arm)
    block = CD.settings(device, threads)
    spec = C21.arm_spec(base, F)
    st = C19.input_stats(builds[0])
    fit_rst = rg_stats(rgs[0])
    rsts = [fit_rst] * len(rgs) if variant == "rgf" else [rg_stats(r) for r in rgs]
    Srst = fit_rst if variant == "rgf" else rg_stats(Sr)
    model = C20.make_model(F, seed)
    if variant != "r0":
        attach(model, seed)
    if CD.placed(device):
        CD._dp().model_to(model, device)
    opt = torch.optim.Adam(model.parameters(), lr=C19.LR, weight_decay=C19.WD)
    rng = np.random.default_rng(seed)
    hist, best, best_state = [], None, None
    tb, nbt = 0.0, 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for ep in range(epochs):
            t1 = time.time()
            model.train()
            items = C21.draw21(rng, builds, groups, per_epoch)
            tot, tn, tc, nb = 0.0, 0.0, 0.0, 0
            for b0 in range(0, len(items), batch):
                t2 = time.time()
                bt = make_batch_rg(items[b0:b0 + batch], builds, rgs, LB, st, rsts, variant)
                if CD.placed(device):
                    bt = CD.to_dev(bt, device)
                    set_ctx(model, bt)
                    with torch.device(device):
                        s, lp = C20.forward20(model, bt)
                        loss, ln, lc = C19.objective(s, lp, bt)
                else:
                    set_ctx(model, bt)
                    s, lp = C20.forward20(model, bt)
                    loss, ln, lc = C19.objective(s, lp, bt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.detach())
                tn += ln
                tc += lc
                nb += 1
                tb += time.time() - t2
                nbt += 1
            model.eval()
            x, _ = read_rg(model, Sd, Sr, Slb, st, Srst, srule, variant, device)
            sr5 = float(x[:, 0].mean())
            hist.append({"epoch": ep, "loss": round(tot / max(nb, 1), 5), "node_loss": round(tn / max(nb, 1), 5),
                         "chain_loss": round(tc / max(nb, 1), 5), "select_r5": round(sr5, 5),
                         "by_build": np.bincount(items[:, 0], minlength=len(builds)).tolist(),
                         **C20.coupling20(model), "seconds": round(time.time() - t1, 1)})
            log(f"    {arm} ep {ep}: loss {tot / max(nb, 1):.4f} (node {tn / max(nb, 1):.4f}, chain "
                f"{tc / max(nb, 1):.4f}) select R@5 {sr5:.4f} {C20.coupling20(model)} {time.time() - t1:.0f}s")
            if best is None or sr5 > best[1]:
                best = (ep, sr5)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    det = sorted({str(w.message)[:300] for w in caught if "deterministic" in str(w.message).lower()})
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": hist, "best_epoch": best[0], "best_select_r5": round(best[1], 5), "stats": st,
                   "rg_stats": [{k: v.tolist() for k, v in x.items()} for x in rsts + [Srst]], "rst": fit_rst,
                   "coupling": C20.coupling20(model), "spec": spec, "placement": block,
                   "state_sha256": CD.state_sha(best_state), "seconds_per_batch": round(tb / max(nbt, 1), 4),
                   "deterministic_warnings": det}


# ── train (one arm, one seed) ───────────────────────────────────────────────────────────────────────────────────


def train_cmd(a):
    t0 = time.time()
    base, variant = arm_parts(a.arm)
    if a.arm == "b-lb-r0" and a.seed != 0:
        raise SystemExit("b-lb-r0 runs seed 0 only")
    spec = C21.arm_spec(base, F)
    data = spec["data"]
    dev = None if a.device == "cpu" else a.device
    res = {"look": "chainscore23", "arm": a.arm, "base": base, "variant": variant, "seed": a.seed, "f": F,
           "device": a.device, "args": dict(vars(a)), "script_sha256": sha(__file__),
           "chainscore21_sha256": sha(C21.__file__), "chainscore20_sha256": sha(C20.__file__),
           "chainscore19_sha256": sha(C19.__file__), "chainscore18_sha256": sha(CS.__file__),
           "cs_dev_sha256": sha(CD.__file__), "inputs": {}, "inputs_sha256": {}, "rg": {}, "rg_read_stats": {},
           "passage_types": {},
           "reads": {}, "rows_out": a.rows_out}
    need = {"b": ("fit", "aug", "select"), "mp": ("fit", "aug", "select", "pfit")}[data]
    given = {"fit": a.fit, "aug": a.aug or None, "select": a.select, "pfit": a.pfit}
    for k, v in given.items():
        if (k in need) != (v is not None):
            raise SystemExit(f"arm {a.arm} takes {', '.join('--' + x for x in need)} and no other training input")
    aug = dict(x.split("=", 1) for x in a.aug)
    if sorted(aug) != sorted(C19.TRANSFORMS[1:]):
        raise SystemExit(f"arm {a.arm} takes --aug for each of {C19.TRANSFORMS[1:]}")
    kb = [tuple(x.split("=", 1)) for x in a.read]
    ps = [tuple(x.split("=", 1)) for x in a.pread]
    if sorted(n for n, _ in kb) != sorted(KB_READS) or sorted(n for n, _ in ps) != sorted(P_READS):
        raise SystemExit(f"every arm reads {KB_READS} (--read) and {P_READS} (--pread)")
    train_in = [("fit", a.fit, False)] + [(t, aug[t], False) for t in C19.TRANSFORMS[1:]] + \
        ([("pfit", a.pfit, True)] if data == "mp" else [])
    sel = ("select", a.select, False)
    reads = [(n, p, False) for n, p in kb] + [(n, p, True) for n, p in ps]
    rgp = dict(x.split("=", 1) for x in a.rg)
    names = [n for n, _p, _q in train_in + [sel] + reads]
    if sorted(rgp) != sorted(names):
        raise SystemExit(f"--rg NAME=PATH for each of {sorted(names)}; given {sorted(rgp)}")
    for n, p, is_p in train_in + [sel] + reads:
        res["inputs_sha256"][p] = sha(p)
        res["inputs_sha256"][rgp[n]] = sha(rgp[n])
        if is_p:
            side = C21.check_passage(p, res["inputs_sha256"][p], None)
            res["passage_types"][n] = {"script_sha256": side["script_sha256"], **side["types"]}
    builds, rgs, LB = [], [], []
    for n, p, _is_p in train_in:
        d = C19.prep(CS.load(p))
        tf = d["meta"]["transform"]["name"]
        if tf != ("id" if n in ("fit", "pfit") else n):
            raise SystemExit(f"{p} holds transform {tf}, not {'id' if n in ('fit', 'pfit') else n}")
        res["inputs"][n] = d["meta"]
        r = load_rg(rgp[n], res["inputs_sha256"][p], d)
        res["rg"][n] = r["meta"]["stats"]
        builds.append(d)
        rgs.append(r)
        LB.append(C21.lb_offsets(d))
    groups = {"b": [list(range(5))], "mp": [list(range(5)), [5]]}[data]
    Sd = CS.load(sel[1])
    Sr = load_rg(rgp["select"], res["inputs_sha256"][sel[1]], Sd)
    Slb = C21.lb_offsets(Sd)
    res["inputs"]["select"] = Sd["meta"]
    res["train_rows"] = {n: int(b["train_rows"].size) for (n, _p, _q), b in zip(train_in, builds)}
    res["groups"] = groups
    model, info = train_rg(a.arm, builds, rgs, LB, groups, Sd, Sr, Slb, "kb", a.epochs, a.per_epoch, a.batch,
                           SEED + a.seed, device=dev)
    del builds, rgs, LB, Sd, Sr, Slb
    res.update({k: info[k] for k in ("history", "best_epoch", "best_select_r5", "coupling", "spec", "placement",
                                     "state_sha256", "seconds_per_batch", "deterministic_warnings", "rg_stats")})
    rows_out = {}
    for n, p, is_p in reads:
        t1 = time.time()
        d = CS.load(p)
        r = load_rg(rgp[n], res["inputs_sha256"][p], d)
        rule = "passage" if is_p else "kb"
        rst = info["rst"] if variant == "rgf" else rg_stats(r)
        res["rg_read_stats"][n] = {k: v.tolist() for k, v in rst.items()}
        xr, ml = read_rg(model, d, r, C21.lb_offsets(d), info["stats"], rst, rule, variant, dev,
                         batch=a.read_batch)
        rows_out[n] = xr.astype(np.float32)
        rows_out[f"{n}__D"] = d["D"]
        res["reads"][n] = dict(C19.summarise(xr, d["D"], ml, d["level_names"]), rule=rule)
        res["inputs"][n] = d["meta"]
        log(f"  {a.arm} s{a.seed} on {n} ({rule} rule): {res['reads'][n]['mean']} ({time.time() - t1:.0f}s)")
        del d, r
    p = Path(a.rows_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **rows_out)
    os.replace(tmp, p)
    if a.state_out:
        import torch
        sp_ = Path(a.state_out)
        torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "stats": info["stats"],
                    "rst": info["rst"], "spec": info["spec"], "arm": a.arm, "seed": a.seed},
                   sp_.with_name(sp_.name + ".tmp"))
        os.replace(sp_.with_name(sp_.name + ".tmp"), sp_)
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ── grade ───────────────────────────────────────────────────────────────────────────────────────────────────────


def grade_cmd(a):
    t0 = time.time()
    runs, rows = {}, {}

    def take(p_, look, key_of):
        js = json.loads(Path(p_).read_text(encoding="utf-8"))
        if js.get("look") != look:
            raise SystemExit(f"{p_} is not a {look} run")
        key = key_of(js)
        if key in runs:
            raise SystemExit(f"two runs of {key}")
        with np.load(js["rows_out"]) as z:
            rows[key] = {k: z[k] for k in z.files}
        runs[key] = js

    for p_ in a.run:
        take(p_, "chainscore23", lambda js: (js["arm"], int(js["seed"])))
    for p_ in a.control:
        take(p_, "chainscore22", lambda js: (js["arm"] if js.get("norm") == "none" else "?", int(js["seed"])))
    want = set(RUNS) | {(c, s) for c in CONTROLS for s in SEEDS}
    if set(runs) != want:
        raise SystemExit(f"the grade takes {sorted(want)}; missing {sorted(want - set(runs))}, "
                         f"extra {sorted(set(runs) - want)}")
    devs = {js["device"] for js in runs.values()}
    if len(devs) != 1:
        raise SystemExit(f"runs on several devices {devs}")
    if len({js["script_sha256"] for k, js in runs.items() if k[0] in ARMS}) != 1:
        raise SystemExit("part 7's runs come from several scripts")
    refs, Ds, rule = {}, {}, {}
    for flag, xs in (("kb", a.read), ("passage", a.pread)):
        for x in xs:
            nm_, p_ = x.split("=", 1)
            h = sha(p_)
            for k, js in runs.items():
                if js["inputs_sha256"].get(p_) != h:
                    raise SystemExit(f"run {k} read another {nm_} than {p_}")
            d = CS.load(p_)
            refs[nm_] = CS.reference_arms(d) if flag == "kb" else C21.passage_refs(d)
            Ds[nm_] = d["D"]
            rule[nm_] = flag
            del d
    if sorted(k for k in rule if rule[k] == "kb") != sorted(KB_READS) or \
            sorted(k for k in rule if rule[k] == "passage") != sorted(P_READS):
        raise SystemExit(f"the grade reads {KB_READS} (--read) and {P_READS} (--pread)")

    def r5(arm, nm_, s=None):
        if s is not None:
            return rows[(arm, s)][nm_][:, 0].astype(np.float64)
        return np.mean([rows[(arm, k)][nm_][:, 0].astype(np.float64) for k in SEEDS], 0)

    V = {k: {} for k in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "S", "R0")}

    def ci(dd, idx):
        c = CP.boot_mean(dd, idx)
        return {"diff": c, "verdict": C19.ci_label(c, ABV)}

    def pair(x_, y_, nm_, idx):
        Y = r5(y_, nm_) if isinstance(y_, str) else y_
        e = ci(r5(x_, nm_) - Y, idx)
        per = [round(float((r5(x_, nm_, s) - (r5(y_, nm_, s) if isinstance(y_, str) else y_)).mean()), 5)
               for s in SEEDS]
        V["S"].setdefault(nm_, {})[f"{x_} - {y_ if isinstance(y_, str) else 'ref'}"] = {
            "per_seed": per, "signs": [int(np.sign(v)) for v in per]}
        return e

    for nm_ in refs:
        N = refs[nm_]["twin0"].shape[0]
        for k, r in rows.items():
            if r[nm_].shape[0] != N:
                raise SystemExit(f"{k}'s {nm_} read has {r[nm_].shape[0]} rows, the build {N}")
        idx = np.random.default_rng(SEED).integers(0, N, size=(BOOT, N))
        if nm_ == "webqsp_sf":
            V["G1"]["b-lb-rg - b-lb"] = pair("b-lb-rg", "b-lb", nm_, idx)
            for dd in (1, 2):
                m = Ds[nm_] == dd
                if m.any():
                    ii = np.random.default_rng(SEED).integers(0, int(m.sum()), size=(BOOT, int(m.sum())))
                    V["G3"][f"D{dd}"] = {f"{x_} - b-lb": dict(ci((r5(x_, nm_) - r5("b-lb", nm_))[m], ii),
                                                              rows=int(m.sum())) for x_ in RG_ARMS}
            walk = refs[nm_]["none/rd"][:, 0]
            V["G7"]["webqsp_sf vs none/rd"] = {f"{x_} - none/rd": pair(x_, walk, nm_, idx)
                                                for x_ in ("b-lb",) + RG_ARMS + ("mp-lb", "mp-lb-rg")}
        if nm_ == "webqsp":
            V["G7"]["webqsp b-lb-rg - b-lb"] = pair("b-lb-rg", "b-lb", nm_, idx)
        if nm_ == "metaqa":
            V["G2"]["b-lb-rg - b-lb"] = pair("b-lb-rg", "b-lb", nm_, idx)
        if nm_ in ("webqsp_sf", "metaqa"):
            V["G4"][nm_] = {f"{x_} - {y_}": pair(x_, y_, nm_, idx)
                            for x_, y_ in (("b-lb-rgs", "b-lb"), ("b-lb-rgf", "b-lb"), ("b-lb-rg", "b-lb-rgs"),
                                           ("b-lb-rg", "b-lb-rgf"))}
        if rule[nm_] == "kb":
            V["G5"][nm_] = {f"{x_} - {y_}": pair(x_, y_, nm_, idx)
                            for x_, y_ in (("mp-lb-rg", "mp-lb"), ("mp-lb-rg", "b-lb-rg"))}
        else:
            s0r, tw = refs[nm_]["s0+rrf"][:, 0], refs[nm_]["twin0"][:, 0]
            e = {f"{x_} - {y_}": pair(x_, y_, nm_, idx) for x_, y_ in
                 (("b-lb-rg", "b-lb"), ("b-lb-rgs", "b-lb"), ("b-lb-rgf", "b-lb"), ("mp-lb-rg", "mp-lb"))}
            for x_ in ("b-lb",) + RG_ARMS + ("mp-lb", "mp-lb-rg"):
                X_ = r5(x_, nm_)
                sh = CP.share(X_, s0r, tw, idx)
                e[f"{x_} - s0+rrf"] = dict(ci(X_ - s0r, idx), share=sh,
                                           share_verdict="HIGH" if sh[1] >= 0.5 else "LOW" if sh[2] < 0.25 else "MID")
            V["G6"][nm_] = e
        dif = float(np.abs(rows[("b-lb-r0", 0)][nm_].astype(np.float64)
                           - rows[("b-lb", 0)][nm_].astype(np.float64)).max())
        V["R0"][nm_] = {"max_abs_diff": dif, "verdict": "REPRODUCES" if dif < 1e-6 else "DIFFERS"}
    arms_all = ("b-lb",) + RG_ARMS + ("mp-lb", "mp-lb-rg")
    res = {"look": "chainscore23", "args": dict(vars(a)), "script_sha256": sha(__file__), "device": devs.pop(),
           "runs": {f"{k[0]}#s{k[1]}": {"best_epoch": js["best_epoch"], "best_select_r5": js["best_select_r5"],
                                        "state_sha256": js.get("state_sha256"),
                                        "reads": {n: v["mean"] for n, v in js["reads"].items()}}
                    for k, js in sorted(runs.items())},
           "means": {nm_: {arm: [round(float(np.mean([rows[(arm, s)][nm_][:, c].astype(np.float64).mean()
                                                       for s in SEEDS])), 5) for c in range(3)] for arm in arms_all}
                     for nm_ in refs},
           "refs": {nm_: {k: [round(float(u), 5) for u in x.mean(0)] for k, x in r.items()
                          if k in ("rrf", "twin0", "none/rd", "s0+rrf", "s0+walk")} for nm_, r in refs.items()},
           "verdicts": V, "decision": decide(V)}
    res["seconds"] = round(time.time() - t0, 1)
    log(f"decision {res['decision']}")
    return res


def decide(V):
    r0 = [(V["R0"].get(k) or {}).get("verdict") for k in KB_READS + P_READS]
    if None in r0 or "DIFFERS" in r0:
        return {"carry": None, "why": "R0 incomplete or DIFFERS: nothing is read until its cause is found",
                "report": True}
    ok = []
    for x_ in RG_ARMS:
        w = V["G1"]["b-lb-rg - b-lb"] if x_ == "b-lb-rg" else V["G4"]["webqsp_sf"][f"{x_} - b-lb"]
        m = V["G2"]["b-lb-rg - b-lb"] if x_ == "b-lb-rg" else V["G4"]["metaqa"][f"{x_} - b-lb"]
        if w["verdict"] == "ABOVE" and m["verdict"] != "BELOW":
            ok.append((w["diff"][0], x_))
    if not ok:
        return {"carry": "none", "why": "no relation-graph arm is ABOVE on webqsp selectf + fit without BELOW on "
                                        "metaqa", "report": True}
    best = max(ok)[1]
    g5 = V["G5"]["webqsp_sf"]["mp-lb-rg - b-lb-rg"]["verdict"]
    return {"carry": best, "candidates": [m for _d, m in sorted(ok, reverse=True)],
            "passages_no_longer_cost": g5 != "BELOW" if best == "b-lb-rg" else None, "report": False,
            "why": f"ABOVE on webqsp selectf + fit and not BELOW on metaqa: {[m for _d, m in ok]}"}


# ── selftest ────────────────────────────────────────────────────────────────────────────────────────────────────


def selftest():
    rng = np.random.default_rng(0)
    # 1. a toy row against a brute-force count: slots 0 (none, all messages), 1-2 (dir), 3-5 (exact, one level)
    n = 7
    msgs = [(0, 0, 1), (1, 1, 0), (1, 0, 2), (2, 1, 1), (0, 0, 3), (3, 1, 0), (4, 0, 5), (5, 1, 4), (2, 0, 6),
            (6, 1, 2)]
    es = np.asarray([m[0] for m in msgs])
    ed = np.asarray([m[1] for m in msgs])
    ee = np.asarray([m[2] for m in msgs])
    typ = {3: [0, 4], 4: [2, 8], 5: [6, 1, 3, 5, 7, 9]}
    i_e = np.concatenate([np.asarray(v) for v in typ.values()])
    i_s = np.concatenate([np.full(len(v), k) for k, v in typ.items()])
    c = {"e_off": np.asarray([0, len(msgs)]), "e_src": es, "e_dst": ee, "e_d": ed, "s_off": np.asarray([0, 6]),
         "i_off": np.asarray([0, i_e.size]), "i_e": i_e, "i_s": i_s, "sl_none": np.asarray([0]),
         "sl_dir": np.asarray([[1, 2]])}
    XN = np.zeros((n, len(C19.NODEF)), np.float16)
    XN[:, CQ] = np.linspace(0, 0.6, n)
    XN[:, CSI] = np.linspace(0.3, 0.9, n)
    b = {"n": np.asarray([n]), "xn_off": np.asarray([0]), "XN": XN, "seed0": np.asarray([1, 0, 0, 0, 0, 0, 0],
                                                                                         bool)}
    slev = np.asarray([0, 1, 1, 2, 2, 2])
    skind = np.asarray([0, 1, 1, 3, 3, 3])
    X, E = row_graph(b, c, slev, skind, 0)
    members = {0: list(range(len(msgs))), 1: list(np.flatnonzero(ed == 0)), 2: list(np.flatnonzero(ed == 1)),
               **{k: v for k, v in typ.items()}}
    heads = {s: {msgs[j][0] for j in v} for s, v in members.items()}
    tails = {s: {msgs[j][2] for j in v} for s, v in members.items()}
    for kind, (A, B) in {"hh": (heads, heads), "tt": (tails, tails), "ht": (heads, tails), "th": (tails, heads)}.items():
        want = {}
        for s in range(6):
            cand = [(len(A[s] & B[t]), t) for t in range(6) if t != s and slev[t] == slev[s] and len(A[s] & B[t])]
            cand = sorted(cand, key=lambda x: (-x[0], x[1]))[:TOPK]
            tot = sum(v for v, _t in cand)
            for v, t in cand:
                want[(s, t)] = v / tot
        r, cc, w = E[kind]
        got = {(int(x), int(y)): float(z) for x, y, z in zip(r, cc, w)}
        assert got.keys() == want.keys() and all(abs(got[k] - want[k]) < 1e-12 for k in want), (kind, got, want)
    for s, v in members.items():
        assert abs(X[s, 0] - np.log1p(len(v)) / np.log1p(len(msgs))) < 1e-12
        assert abs(X[s, 1] - np.mean([msgs[j][0] == 0 for j in v])) < 1e-12
        cqv = XN[:, CQ].astype(np.float64)
        assert abs(X[s, 3] - np.mean([cqv[t] for t in tails[s]])) < 1e-12
        assert abs(X[s, 4] - max(cqv[t] for t in tails[s])) < 1e-12
        assert abs(X[s, 7] - np.log(len(v) / len(heads[s]))) < 1e-12
        assert X[s, 9 + skind[s]] == 1.0 and X[s, 9:].sum() == 1.0
    # 2. the toy's graph has the composition edge: slot 3 (0->1, 0->3) and slot 4 (1->2, 2->6) meet at node 1 (ht/th)
    assert (4, 3) in {(int(x), int(y)) for x, y in zip(E["ht"][0], E["ht"][1])}
    # 3. a top-k tie goes to the lower slot; the diagonal never enters
    M = sp.coo_matrix((np.asarray([3.0, 2, 2, 2, 5]), (np.asarray([0, 0, 0, 0, 0]), np.asarray([0, 3, 1, 2, 4]))),
                      shape=(1, 5))
    r, cc, v = topk_rows(M, 2)
    assert cc.tolist() == [4, 1] and v.tolist() == [5.0, 2.0]
    # 4. the model: q starts at 0 (the forward is the base's), the base weights are make_model's, and q learns
    import torch
    torch.use_deterministic_algorithms(True)
    m0 = C20.make_model(F, SEED)
    m1 = attach(C20.make_model(F, SEED), SEED)
    sd0, sd1 = m0.state_dict(), m1.state_dict()
    for k, v in sd0.items():
        k1 = k.replace("g.", "g.g0.", 1) if k.startswith("g.") else k
        assert torch.equal(v, sd1[k1]), k
    S, P = 9, 6
    ctx = {"X": torch.randn(S, len(SLOTF)), "ps": torch.from_numpy(np.asarray([[0, 1], [2, S], [3, 4], [5, 6],
                                                                                [7, 8], [S, S]])),
           "E": [(torch.from_numpy(rng.integers(0, S, 12)), torch.from_numpy(rng.integers(0, S, 12)),
                  torch.rand(12)) for _ in KINDS]}
    Xp = torch.randn(P, len(C19.PAIRF))
    m1.g.ctx = ctx
    assert torch.equal(m1.g(Xp), m0.g(Xp))
    out = m1.g(Xp).sum()
    out.backward()
    assert float(m1.g.head.l2.weight.grad.abs().sum()) > 0
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("rgbuild")
    r.add_argument("--base", required=True)
    r.add_argument("--edges", required=True)
    r.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=ARMS)
    t.add_argument("--seed", type=int, required=True, choices=SEEDS)
    t.add_argument("--fit", default=None)
    t.add_argument("--aug", action="append", default=[])
    t.add_argument("--select", default=None)
    t.add_argument("--pfit", default=None)
    t.add_argument("--read", action="append", default=[])
    t.add_argument("--pread", action="append", default=[])
    t.add_argument("--rg", action="append", default=[])
    t.add_argument("--epochs", type=int, default=C19.EPOCHS)
    t.add_argument("--per-epoch", type=int, default=C19.PER_EPOCH)
    t.add_argument("--batch", type=int, default=BATCH)
    t.add_argument("--read-batch", type=int, default=BATCH)
    t.add_argument("--device", default="cpu")
    t.add_argument("--out", required=True)
    t.add_argument("--rows-out", required=True)
    t.add_argument("--state-out")
    g = sub.add_parser("grade")
    g.add_argument("--run", action="append", required=True)
    g.add_argument("--control", action="append", required=True, help="part 6's b-lb and mp-lb runs (each seed)")
    g.add_argument("--read", action="append", required=True)
    g.add_argument("--pread", action="append", required=True)
    g.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "rgbuild":
        rgbuild_cmd(a)
        return 0
    if a.cmd in ("train", "grade"):
        res = train_cmd(a) if a.cmd == "train" else grade_cmd(a)
        res.pop("rst", None)
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                       encoding="utf-8")
        os.replace(tmp, p)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
