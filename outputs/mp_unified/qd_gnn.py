"""Design look (untracked; not a result and not filed): a query-conditioned relational GNN (NBFNet / GFM-RAG style),
fitted as a residual over the frozen twin's z on the anchor looks' rows, and, in the same loop on the same rows with
the same batches, loss, selection and reads, the non-message-passing walk models it is set against.

The GNN (QD). A row's graph is its pool with the look's edges (both orientations; famdir d = 0 S-fwd, 1 S-bwd,
2 S-both, 3 NER, 4 KNN) and, on a structural edge, its label: the anchor phrase on a passage graph or the relation on a
KB, as a rank in that graph's label table (anchor_walk.anchor_tokens' choice of stored direction). QD-TXT reads a
label below K through its text vector phi[rank] only, so one model runs on any graph; a label at or past K, a
non-structural edge and the NR read are 'other'. QD-T0 reads d only. QD-ID (one graph only) reads a learned vector per
rank. The state starts at the seeds and nowhere else:
    qn   = LN(Wq q)
    h0_s = qn * (1 + E_b[bucket]) + z_s e_z
and each of L layers runs, on every edge u -> v of the row,
    r_e  = E_fd[d] + M_d P phi[a]          (QD-TXT, a below K; otherwise E_fd[d] + E_oth[d]; QD-T0: E_fd[d])
    w_e  = R_l r_e * (1 + tanh(G_l qn))
    g_e  = 2 sigmoid(<A_l qn, r_e> + c_l[fam] w_e + b_l)
    m_e  = rms(h_u) * w_e * g_e
    h_v <- h_v + relu(W_l sum_e m_e / (1 + n_v)^gamma),   n_v the edges into v whose source has a state.
No bias reaches a node without a state, so a node more than L edges from every seed keeps h = 0, and the pool cut to
the seeds' (L-1)-edge frontier scores exactly as the whole pool does (--selftest checks it). The score is
    s_v = z_v + w_o . relu(W_r [rms(h_v), rms(h_v) * qn, z_v, reached_v])
with w_o zero at init, so an unfitted QD scores the twin's z exactly. Nothing reads a node's text, its 129 columns or
any model's state beyond the twin's z, which the walk models read too.

The walk arms (W:<variant>) are anchor_univ's: anchor_gen / anchor_gen3 models over anchor_gen.types_for's walk types,
packed and scored by l16_look_gate2 as anchor_univ.fit_joint does. With rule p and one walk arm the loop is
fit_joint's, draw for draw (--check-walk compares the two on the host).

Rule (--rule): np fits every arm on the whole pool and selects on the select carve's mean of recall@5,
full_coverage@5 and hit@1 read without protection; p (anchor_univ's) drops the twin's rank-1 node from the softmax
and the golds and selects on recall@5 and full_coverage@5 read with it placed first. Every arm is read both ways.
Batches, epochs, the graph balance and the joint selection (the mean over graphs) are anchor_univ's.

Reads on each training graph's x1 half B (anchor_gen.Reader: rho and the bootstrap against the twin and the GNN):
    ID/np, ID/p      the graph's own view, unprotected and with the twin's rank-1 placed first
    NR/np, NR/p      every label 'other' (label arms)
    SHUF/np          (a KB, text arms) every relation shown with the next relation's text
2wiki x2 / x3 and musique x3 are never loaded; webqsp is never a training graph.

    python outputs/mp_unified/qd_gnn.py --selftest
    python outputs/mp_unified/qd_gnn.py --train 2wiki=x4,hotpotqa=x4,metaqa=fit --arms QD-T0-L3,QD-TXT-L3,W:T0 \
        --rule np --out outputs/mp_unified/qd_j3_np.json
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AU_DIR = ROOT / "outputs" / "mp_approx_2wiki_anchor" / "host"
KB_DIR = ROOT / "outputs" / "mp_approx_kb_anchor" / "host"
PINS = {"anchor_univ": "54da7d219a690fc5cab75ab12fdb769bf2c19564f7eca676c7de07405429a505",
        "anchor_walk11": "e782089f33cae0016ad4bbf09f14e092b0da7eaf31932c5d437d42f274d0f99b",
        "l16_look_analyze": "997e1d69a7c7794320e6a0b48924ca3460f94020c960354bbfa8778623044611"}
R_TOP = 4096
READ_SEED = 20261002
QDIM = TDIM = 1536


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ── the frontier cut, any depth ──────────────────────────────────────────────


def frontier_mask(u, v, seeds, n, max_len):
    """The edges whose source is within max_len - 1 edges of a seed (anchor_walk11.keep_mask for max_len 1 and 2)."""
    src = np.zeros(n, dtype=bool)
    S = seeds[seeds >= 0]
    src[S] = True
    for _ in range(max(max_len, 1) - 1):
        reach = np.zeros(n, dtype=bool)
        reach[v[src[u]]] = True
        src |= reach
    return src[u]


def bind_frontier(AW11):
    """anchor_walk11.keep_mask, read at call time by its loader and anchor_gen_kb's, extended past two edges; for
    max_len 1 and 2 the original is called (and checked equal on a random graph)."""
    orig = AW11.keep_mask
    rng = np.random.default_rng(7)
    u, v = rng.integers(0, 60, 400), rng.integers(0, 60, 400)
    seeds = np.asarray([3, 9, -1, 17], dtype=np.int64)
    for ml in (1, 2):
        if not np.array_equal(orig(u, v, seeds, 60, ml), frontier_mask(u, v, seeds, 60, ml)):
            raise SystemExit(f"frontier_mask differs from anchor_walk11.keep_mask at max_len {ml}")

    def keep_mask(u, v, seeds, n, max_len):
        return orig(u, v, seeds, n, max_len) if max_len <= 2 else frontier_mask(u, v, seeds, n, max_len)

    AW11.keep_mask = keep_mask


# ── tokens and packing ───────────────────────────────────────────────────────


def famdir(q):
    """l16_look_analyze.famdir (checked equal on the loaded rows)."""
    f = q["fam"]
    return np.where(f == 1, 3, np.where(f == 2, 4, np.where(q["fwd"] & q["bwd"], 2, np.where(q["fwd"], 0, 1)))).astype(np.int64)


def edge_labels(q):
    """Per edge: famdir d and the label rank a (-1 off structural edges), anchor_walk.anchor_tokens' direction choice."""
    d = famdir(q)
    a = np.full(d.size, -1, dtype=np.int64)
    m = q["fam"] == 0
    if m.any():
        am = np.where(d[m] == 1, q["w2_b"], q["w2_f"]).astype(np.int64)
        if (am < 0).any():
            raise SystemExit("a structural pool edge has no stored label in its direction")
        a[m] = am
    return d.astype(np.int8), a.astype(np.int32)


class Pack:
    pass


def pack_qd(rows, Q, TOK, z_of, K, nr=False):
    """One batch (rows of one graph) as flat tensors: nodes in row order, edges with global endpoints."""
    P = Pack()
    ns = np.asarray([Q[i]["n"] for i in rows], dtype=np.int64)
    off = np.r_[0, np.cumsum(ns)[:-1]]
    P.B, P.N, P.nmax = len(rows), int(ns.sum()), int(ns.max())
    P.node_row = torch.from_numpy(np.repeat(np.arange(len(rows)), ns))
    P.local = torch.from_numpy(np.concatenate([np.arange(n) for n in ns]))
    P.z = torch.from_numpy(np.concatenate([z_of[i] for i in rows]).astype(np.float32))
    P.gold = torch.from_numpy(np.concatenate([Q[i]["gold"] for i in rows]).astype(np.float32))
    P.qemb = torch.from_numpy(np.stack([Q[i]["qemb"] for i in rows]).astype(np.float32))
    ne = np.asarray([Q[i]["u"].size for i in rows], dtype=np.int64)
    P.src = torch.from_numpy(np.concatenate([Q[i]["u"] + o for i, o in zip(rows, off)]).astype(np.int64))
    P.dst = torch.from_numpy(np.concatenate([Q[i]["v"] + o for i, o in zip(rows, off)]).astype(np.int64))
    P.e_row = torch.from_numpy(np.repeat(np.arange(len(rows)), ne))
    P.d = torch.from_numpy(np.concatenate([TOK[i][0] for i in rows]).astype(np.int64))
    a = np.concatenate([TOK[i][1] for i in rows]).astype(np.int64)
    a = np.where((a >= 0) & (a < (K or 0)) & (not nr), a, -1)
    P.a = torch.from_numpy(a.astype(np.int64))
    P.fam = torch.from_numpy(np.concatenate([Q[i]["fam"] for i in rows]).astype(np.int64))
    P.w = torch.from_numpy(np.concatenate([Q[i]["w"] for i in rows]).astype(np.float32))
    si, sb, sr = [], [], []
    for r, (i, o) in enumerate(zip(rows, off)):
        m = Q[i]["seeds"] >= 0
        si.append(Q[i]["seeds"][m] + o)
        sb.append(Q[i]["bucket"][m])
        sr.append(np.full(int(m.sum()), r))
    P.seed_idx = torch.from_numpy(np.concatenate(si).astype(np.int64))
    P.seed_b = torch.from_numpy(np.concatenate(sb).astype(np.int64))
    P.seed_row = torch.from_numpy(np.concatenate(sr).astype(np.int64))
    return P


def padded(P, s):
    """Flat node scores and golds as (B, nmax) with -inf past each row's pool."""
    out = torch.full((P.B, P.nmax), float("-inf"))
    out = out.index_put((P.node_row, P.local), s)
    gold = torch.zeros(P.B, P.nmax).index_put((P.node_row, P.local), P.gold)
    z = torch.full((P.B, P.nmax), float("-inf")).index_put((P.node_row, P.local), P.z)
    return out, gold, z


# ── the model ────────────────────────────────────────────────────────────────


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class QD(torch.nn.Module):
    def __init__(self, kind, d=64, layers=3, n_id=0):
        super().__init__()
        if kind not in ("T0", "TXT", "ID"):
            raise ValueError(kind)
        self.kind, self.dim, self.L = kind, d, layers
        self.Wq = torch.nn.Linear(QDIM, d)
        self.lnq = torch.nn.LayerNorm(d)
        self.E_b = torch.nn.Parameter(torch.zeros(2, d))
        self.e_z = torch.nn.Parameter(torch.zeros(d))
        self.E_fd = torch.nn.Parameter(torch.randn(5, d))
        if kind == "TXT":
            self.P = torch.nn.Linear(TDIM, d, bias=False)
            self.M = torch.nn.Parameter(torch.eye(d).repeat(3, 1, 1))
        if kind == "ID":
            self.E_id = torch.nn.Parameter(torch.randn(n_id, d) * 0.5)
        if kind != "T0":
            self.E_oth = torch.nn.Parameter(torch.zeros(5, d))
        self.R = torch.nn.ModuleList([torch.nn.Linear(d, d, bias=False) for _ in range(layers)])
        self.G = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(layers)])
        self.A = torch.nn.ModuleList([torch.nn.Linear(d, d, bias=False) for _ in range(layers)])
        self.W = torch.nn.ModuleList([torch.nn.Linear(d, d, bias=False) for _ in range(layers)])
        self.c_fam = torch.nn.Parameter(torch.zeros(layers, 3))
        self.b_gate = torch.nn.Parameter(torch.zeros(layers))
        self.gamma_raw = torch.nn.Parameter(torch.zeros(()))
        self.Wr = torch.nn.Linear(2 * d + 2, d)
        self.wo = torch.nn.Linear(d, 1)
        torch.nn.init.zeros_(self.wo.weight)
        torch.nn.init.zeros_(self.wo.bias)
        self.register_buffer("phi", torch.zeros(0, TDIM), persistent=False)

    def set_phi(self, phi, K):
        """The graph's label text table, its first K rows, unit-normalised (read only through P)."""
        if self.kind != "TXT":
            return
        t = torch.as_tensor(np.asarray(phi[:K], dtype=np.float32))
        self.phi = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12) * math.sqrt(TDIM)   # unit variance per coordinate

    def relations(self, P):
        r = self.E_fd[P.d]
        if self.kind == "T0":
            return r
        has = P.a >= 0
        lab = self.E_oth[P.d]
        if self.kind == "ID":
            lab = torch.where(has[:, None], self.E_id[P.a.clamp_min(0)], lab)
        else:
            T = self.P(self.phi)                                        # (K, d)
            TT = torch.einsum("kij,aj->kai", self.M, T)                 # (3, K, d): each stored direction's map
            t = TT[P.d.clamp(max=2), P.a.clamp_min(0)]
            lab = torch.where(has[:, None], t, lab)
        return r + lab

    def forward(self, P):
        qn = self.lnq(self.Wq(P.qemb))                                  # (B, d)
        h0 = qn[P.seed_row] * (1.0 + self.E_b[P.seed_b]) + P.z[P.seed_idx, None] * self.e_z
        h = torch.zeros(P.N, self.dim).index_add(0, P.seed_idx, h0)
        r = self.relations(P)
        gamma = torch.sigmoid(self.gamma_raw)
        fam = P.fam.clamp(max=2)
        for l in range(self.L):
            w = self.R[l](r) * (1.0 + torch.tanh(self.G[l](qn)))[P.e_row]
            logit = (self.A[l](qn)[P.e_row] * r).sum(-1) + self.c_fam[l, fam] * P.w + self.b_gate[l]
            g = 2.0 * torch.sigmoid(logit)
            hs = rms(h)
            m = hs[P.src] * w * g[:, None]
            active = (h != 0).any(-1).to(m.dtype)
            n_in = torch.zeros(P.N).index_add(0, P.dst, active[P.src])
            agg = torch.zeros(P.N, self.dim).index_add(0, P.dst, m) / (1.0 + n_in)[:, None].pow(gamma)
            h = h + torch.relu(self.W[l](agg))
        hn = rms(h)
        reached = (h != 0).any(-1).to(h.dtype)
        f = torch.cat([hn, hn * qn[P.node_row], P.z[:, None], reached[:, None]], 1)
        return P.z + self.wo(torch.relu(self.Wr(f))).squeeze(-1)


def parse_qd(name):
    m = re.fullmatch(r"QD-(T0|TXT|ID)(?:-L(\d))?(?:-d(\d+))?(?:-K(\d+))?", name)
    if not m:
        raise SystemExit(f"{name}: QD-(T0|TXT|ID)[-L<layers>][-d<dim>][-K<labels>]")
    kind = m.group(1)
    return {"family": "qd", "kind": kind, "L": int(m.group(2) or 3), "d": int(m.group(3) or 64),
            "K": (int(m.group(4) or 256) if kind != "T0" else 0), "max_len": int(m.group(2) or 3)}


# ── the shared loop ──────────────────────────────────────────────────────────


def step_loss(s, gold, z, rule):
    """anchor_univ.fit_joint's loss; rule p first drops the twin's rank-1 node from the softmax and the golds."""
    if rule == "p":
        first = torch.argmax(z, 1)
        ar = torch.arange(s.shape[0])
        s = s.clone()
        s[ar, first] = float("-inf")
        gold = gold.clone()
        gold[ar, first] = 0.0
    keep = gold.sum(1) > 0
    if not bool(keep.any()):
        return None
    ls = torch.log_softmax(s[keep], 1)
    gk = gold[keep]
    gw = gk / gk.sum(1, keepdim=True)
    return 1.0 * -(torch.where(gk > 0, ls, torch.zeros_like(ls)) * gw).sum(1).mean()


def metrics_rows(s, z, rows, Q, rule, metrics_of):
    if rule == "p":
        ar = torch.arange(len(rows))
        s = s.clone()
        s[ar, torch.argmax(z, 1)] = float("inf")
    return [metrics_of(s[bi, :Q[i]["n"]].numpy(), Q[i]["gold"], Q[i]["gt"]) for bi, i in enumerate(rows)]


def fit_shared(arm, names, tr, sel, rule, epochs, lr, wd, seed, balance, AU, A16):
    """anchor_univ.fit_joint's loop for any arm: arm.make(), arm.prepare(g), arm.forward(model, g, rows) -> (s, gold, z)
    padded, arm.read(model, g, rows, rule) -> metrics. The select score is rule p's (recall@5, full_coverage@5 read
    protected) or np's (recall@5, full_coverage@5, hit@1 read unprotected); the joint score is the mean over graphs."""
    torch.manual_seed(seed)
    model = arm.make()
    mats = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": mats, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}], lr=lr)
    rng = np.random.default_rng(seed)
    streams = {g: {"perm": None, "pos": 0} for g in names}
    best, best_state, best_ep, curve, by_graph = -1.0, None, -1, [], {g: [] for g in names}
    cols = slice(0, 2) if rule == "p" else slice(0, 3)
    for ep in range(epochs):
        t_ep = time.time()
        batches = AU.epoch_batches(rng, names, tr, balance, streams)
        model.train()
        for g, rows in batches:
            arm.prepare(model, g)
            s, gold, z = arm.forward(model, g, rows)
            loss = step_loss(s, gold, z, rule)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
        sc = {}
        for g in names:
            arm.prepare(model, g)
            sc[g] = float(arm.read(model, g, list(sel[g]), rule)[:, cols].mean())
            by_graph[g].append(round(sc[g], 4))
        score = sc[names[0]] if len(names) == 1 else float(np.mean([sc[g] for g in names]))
        curve.append(round(score, 4))
        if score > best:
            best, best_ep, best_state = score, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"  ep {ep}: select {round(score, 4)} " + " ".join(f"{g} {sc[g]:.4f}" for g in names) + f" ({time.time() - t_ep:.0f}s)")
    model.load_state_dict(best_state)
    return model, best_ep, curve, by_graph


class QDArm:
    def __init__(self, sp, Q, TOK, G, z_of, A16, n_id):
        self.sp, self.Q, self.TOK, self.G, self.z_of, self.A16, self.n_id = sp, Q, TOK, G, z_of, A16, n_id
        self.nr = False

    def make(self):
        return QD(self.sp["kind"], self.sp["d"], self.sp["L"], self.n_id)

    def prepare(self, model, g, phi=None):
        model.set_phi(self.G[g]["phi"] if phi is None else phi, self.sp["K"])

    def forward(self, model, g, rows):
        P = pack_qd(rows, self.Q, self.TOK, self.z_of, self.sp["K"], self.nr)
        return padded(P, model(P))

    def read(self, model, g, rows, rule):
        model.eval()
        out = []
        with torch.no_grad():
            for s0 in range(0, len(rows), 128):
                rr = rows[s0:s0 + 128]
                s, _gold, z = self.forward(model, g, rr)
                out += metrics_rows(s, z, rr, self.Q, rule, self.A16.metrics_of)
        model.train()
        return np.asarray(out)


class WalkArm:
    """anchor_univ's walk models, packed and scored as fit_joint does."""

    def __init__(self, sp, Q, TY, nt, G, z_of, AU, AG, G3, A16):
        self.sp, self.Q, self.TY, self.nt, self.G, self.z_of = sp, Q, TY, nt, G, z_of
        self.AU, self.AG, self.G3, self.A16 = AU, AG, G3, A16
        self.make_fn = None

    def make(self):
        return self.make_fn()

    def prepare(self, model, g, phi=None):
        self.AU.set_graph(model, self.sp, self.G[g]["phi"] if phi is None else phi)

    def forward(self, model, g, rows, TY=None):
        PX = self.AU.G2.LG.pack_gate(rows, self.Q, self.TY if TY is None else TY, self.z_of, self.nt)
        s, gold = self.AU.G2.scores(model, PX)
        return s, gold, PX[0][9]

    def read(self, model, g, rows, rule, TY=None):
        model.eval()
        out = []
        with torch.no_grad():
            for s0 in range(0, len(rows), 256):
                rr = rows[s0:s0 + 256]
                s, _gold, z = self.forward(model, g, rr, TY)
                out += metrics_rows(s, z, rr, self.Q, rule, self.A16.metrics_of)
        model.train()
        return np.asarray(out)


# ── the run ──────────────────────────────────────────────────────────────────


def run(a):
    out_path = Path(a.out)
    if a.rule not in ("np", "p"):
        raise SystemExit("--rule np|p")
    sys.path.insert(0, str(AU_DIR))
    sys.path.insert(0, str(KB_DIR))
    import anchor_univ as AU
    if sha(AU.__file__) != PINS["anchor_univ"]:
        raise SystemExit("anchor_univ.py is not the pinned file")
    AU.check_pins()
    AU.AC.bind()
    AG, G3, A16, AW11 = AU.AG, AU.G3, AU.A16, AU.AW11
    for mod, key in ((AW11, "anchor_walk11"), (A16, "l16_look_analyze")):
        if sha(mod.__file__) != PINS[key]:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    bind_frontier(AW11)
    trains = AU.parse_train(a.train)
    names = list(trains)
    arms = {}
    for nm in a.arms.split(","):
        if nm.startswith("QD-"):
            sp = parse_qd(nm)
            if sp["kind"] == "ID" and len(names) > 1:
                raise SystemExit(f"{nm}: a rank's meaning is one graph's; QD-ID fits one graph only")
            arms[nm] = sp
        elif nm.startswith("W:"):
            v = nm[2:]
            if ":" in v:
                raise SystemExit(f"{nm}: a variant names no train looks")
            vv = f"{AU.TAG}:{v}" + ("/lin" if v.count("/") == 0 else "") + ("" if v.count("/") >= 2 else f"/{a.rule}")
            sp = AG.parse(vv, (AU.TAG,)) if v.startswith("T0") else G3.parse(vv, (AU.TAG,))
            if sp["rule"] != a.rule:
                raise SystemExit(f"{nm}: rule {sp['rule']} under --rule {a.rule}")
            if sp.get("pca") or (sp["family"] == "free" and sp["tok"] != "T0"):
                raise SystemExit(f"{nm}: refused as anchor_univ refuses it")
            arms[nm] = {**sp, "family_arm": "walk"}
        else:
            raise SystemExit(f"{nm}: QD-... or W:<variant>")
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    max_len = max(sp["max_len"] for sp in arms.values())
    Q, part, G = AU.load_graphs(trains, max_len, t0, True)
    log(f"loaded {len(Q)} rows at max_len {max_len}: " + ", ".join(f"{g} {G[g]['info']['pruned_loader']}" for g in names))
    for i in range(0, len(Q), max(1, len(Q) // 200)):
        if not np.array_equal(famdir(Q[i]), A16.famdir(Q[i])):
            raise SystemExit("famdir differs from l16_look_analyze.famdir")
    TOK = [edge_labels(q) for q in Q]
    tr = {g: [i for lk in trains[g] for i in part[(g, lk)]] for g in names}
    sel = {g: part[(g, "select")] for g in names}
    B = {g: part[(g, "x1")][1::2] for g in names}
    RB = {g: AG.Reader(Q, B[g], READ_SEED) for g in names}
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    res = {"look": "qd_gnn", "train": trains, "rule": a.rule, "arms": list(arms), "script_sha256": sha(__file__), "pins": PINS,
           "epochs": a.epochs, "lr": a.lr, "wd": a.wd, "balance": a.balance, "threads": a.threads, "max_len": max_len,
           "graphs": {g: {"kind": G[g]["kind"], "train_rows": len(tr[g]), "select_rows": len(sel[g]), "B_rows": len(B[g]),
                          "info": G[g]["info"], "B": RB[g].base()} for g in names},
           "results": {}}
    per_row = {}
    model_dir = out_path.parent / (out_path.stem + "_models")
    model_dir.mkdir(parents=True, exist_ok=True)

    def save():
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, **{f"B_rows_{g}": np.asarray(B[g]) for g in names})

    for ai, (name, sp) in enumerate(arms.items()):
        t1 = time.time()
        if sp["family"] == "qd":
            n_id = sp["K"] if sp["kind"] == "ID" else 0
            arm = QDArm(sp, Q, TOK, G, z_of, A16, n_id)
            alt = {}
        else:
            K = sp["K"]
            TY, nt = [None] * len(Q), None
            for g in names:
                rows_g = sorted({i for lk in ["x1", "select"] + trains[g] for i in part[(g, lk)]})
                nt, ty = AG.types_for(Q, sp, None if sp["tok"] == "T0" else AG.identity_map(K), rows_g)
                for i in rows_g:
                    TY[i] = ty[i]
                del ty
            arm = WalkArm(sp, Q, TY, nt, G, z_of, AU, AG, G3, A16)
            arm.make_fn = G3.make_for(sp, nt, G[names[0]]["phi"], None)
            alt = {g: ({"NR": AG.types_for(Q, sp, np.full(R_TOP + 1, K, dtype=np.int64), B[g])[1]} if sp["tok"] == "A" else {}) for g in names}
        seeds_out = {}
        for sd in [int(x) for x in a.seeds.split(",")]:
            t2 = time.time()
            model, best_ep, curve, by_graph = fit_shared(arm, names, tr, sel, a.rule, a.epochs, a.lr, a.wd, sd, a.balance, AU, A16)
            rd = {}
            for g in names:
                R = RB[g]
                arm.prepare(model, g)
                e = {}
                for rl in ("np", "p"):
                    mID = arm.read(model, g, B[g], rl)
                    per_row[f"{ai}_{sd}_{g}_ID_{rl}"] = mID
                    e[f"ID/{rl}"] = R.record(mID)
                    label_arm = (sp["family"] == "qd" and sp["kind"] != "T0") or (sp["family"] != "qd" and sp["tok"] == "A")
                    if label_arm:
                        if sp["family"] == "qd":
                            arm.nr = True
                            m_alt = arm.read(model, g, B[g], rl)
                            arm.nr = False
                        else:
                            m_alt = arm.read(model, g, B[g], rl, alt[g]["NR"])
                        per_row[f"{ai}_{sd}_{g}_NR_{rl}"] = m_alt
                        e[f"NR/{rl}"], e[f"NR - ID/{rl}"] = R.record(m_alt, by_type=False), AU.AW3.boot_pair(m_alt - mID, R.W)
                    if rl == "np" and G[g]["kind"] == "kb" and ((sp["family"] == "qd" and sp["kind"] == "TXT") or
                                                                (sp["family"] != "qd" and sp["tok"] == "A" and sp["family"] != "free")):
                        phi = G[g]["phi"]
                        phi_s = phi.copy()
                        k_ = min(G[g]["n_rel"], R_TOP)
                        phi_s[:k_] = phi[(np.arange(k_) + 1) % k_]
                        arm.prepare(model, g, phi_s)
                        m_alt = arm.read(model, g, B[g], rl)
                        arm.prepare(model, g)
                        per_row[f"{ai}_{sd}_{g}_SHUF_{rl}"] = m_alt
                        e["SHUF/np"], e["SHUF - ID/np"] = R.record(m_alt, by_type=False), AU.AW3.boot_pair(m_alt - mID, R.W)
                rd[g] = e
            pt = model_dir / f"a{ai}_s{sd}.pt"
            torch.save({"arm": name, "spec": sp, "state_dict": model.state_dict(), "train": trains, "rule": a.rule,
                        "phi_sha": {g: G[g]["phi_sha"] for g in names}}, pt)
            seeds_out[str(sd)] = {"best_epoch": best_ep, "curve": curve, "curve_by_graph": by_graph, "reads": rd,
                                  "model_file": str(pt.relative_to(ROOT)) if pt.is_relative_to(ROOT) else str(pt),
                                  "seconds": round(time.time() - t2, 1)}
            log(f"{name}#{sd}: ep {best_ep}; " + "; ".join(
                f"{g} " + ", ".join(f"{k} {v['rho (R@5, FC@5, hit@1)']}" for k, v in rd[g].items() if isinstance(v, dict) and "rho (R@5, FC@5, hit@1)" in v)
                for g in names))
            res["results"][name] = {"spec": {k: v for k, v in sp.items() if k != "seeds"}, "params": int(sum(p.numel() for p in model.parameters())),
                                    "seeds_read": seeds_out, "seconds": round(time.time() - t1, 1)}
            save()
    res["seconds"] = round(time.time() - t0, 1)
    save()
    log(f"done in {res['seconds']}s")


# ── selftest (synthetic rows; laptop) ────────────────────────────────────────


def synthetic_rows(n_rows, rng, K=8):
    Q = []
    for _ in range(n_rows):
        n = int(rng.integers(20, 60))
        m = int(rng.integers(n, 4 * n))
        u0, v0 = rng.integers(0, n, m), rng.integers(0, n, m)
        ok = u0 != v0
        u0, v0 = u0[ok], v0[ok]
        key = np.unique(np.minimum(u0, v0) * n + np.maximum(u0, v0))
        a, b = key // n, key % n
        fam = rng.integers(0, 3, a.size)
        f = rng.random(a.size) < 0.6
        bb = (~f) | (rng.random(a.size) < 0.2)
        fam_struct = fam == 0
        f = np.where(fam_struct, f, True)
        bb = np.where(fam_struct, bb, True)
        lf, lb = rng.integers(0, 2 * K, a.size), rng.integers(0, 2 * K, a.size)
        u = np.r_[a, b]
        v = np.r_[b, a]
        famx = np.r_[fam, fam]
        fwd = np.r_[f, bb]
        bwd = np.r_[bb, f]
        w2f = np.r_[np.where(f, lf, -1), np.where(bb, lb, -1)]
        w2b = np.r_[np.where(bb, lb, -1), np.where(f, lf, -1)]
        sm = famx == 0
        seeds = np.full(10, -1, dtype=np.int64)
        k = int(rng.integers(1, 5))
        seeds[:k] = rng.choice(n, k, replace=False)
        bucket = np.zeros(10, dtype=np.int64)
        bucket[1:k] = 1
        gold = np.zeros(n, dtype=bool)
        gold[rng.choice(n, int(rng.integers(1, 3)), replace=False)] = True
        Q.append({"n": n, "u": u, "v": v, "fam": famx, "fwd": fwd, "bwd": bwd, "w": rng.random(u.size).astype(np.float32),
                  "w2_f": w2f[sm].astype(np.int32), "w2_b": w2b[sm].astype(np.int32), "seeds": seeds, "bucket": bucket,
                  "gold": gold, "gt": int(gold.sum()), "qemb": rng.standard_normal(QDIM).astype(np.float32),
                  "score": rng.standard_normal((n, 6)).astype(np.float32)})
    return Q


def selftest():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    K = 8
    Q = synthetic_rows(12, rng, K)
    z_of = [((q["score"][:, 0] - q["score"][:, 0].mean()) / q["score"][:, 0].std()).astype(np.float64) for q in Q]
    phi = rng.standard_normal((R_TOP, TDIM)).astype(np.float32)
    rows = list(range(len(Q)))
    for kind in ("T0", "TXT", "ID"):
        for L in (1, 2, 3):
            torch.manual_seed(1)
            model = QD(kind, 16, L, K)
            model.set_phi(phi, K)
            TOK = [edge_labels(q) for q in Q]
            P = pack_qd(rows, Q, TOK, z_of, K)
            s = model(P)
            assert torch.equal(s, P.z), "an unfitted QD does not score z"
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(0.3 * torch.randn_like(p))
            s_full = model(P)
            Qc = []
            for q in Q:
                mk = frontier_mask(q["u"], q["v"], q["seeds"], q["n"], L)
                qq = {**q, **{k: q[k][mk] for k in ("u", "v", "fam", "fwd", "bwd", "w")}}
                sm_full, sm_cut = q["fam"] == 0, qq["fam"] == 0
                idx = np.cumsum(sm_full) - 1
                keep_struct = mk[sm_full]
                qq["w2_f"], qq["w2_b"] = q["w2_f"][keep_struct], q["w2_b"][keep_struct]
                assert qq["w2_f"].size == int(sm_cut.sum())
                del idx
                Qc.append(qq)
            TOKc = [edge_labels(q) for q in Qc]
            s_cut = model(pack_qd(rows, Qc, TOKc, z_of, K))
            err = float((s_full - s_cut).abs().max())
            assert err < 1e-5, f"{kind} L{L}: the frontier cut changes the scores by {err}"
            Pd = pack_qd(rows, Q, TOK, z_of, K)
            far = []
            for bi, q in enumerate(Q):
                reach = np.zeros(q["n"], dtype=bool)
                reach[q["seeds"][q["seeds"] >= 0]] = True
                for _ in range(L):
                    nxt = reach.copy()
                    nxt[q["v"][reach[q["u"]]]] = True
                    reach = nxt
                far.append(~reach)
            far = torch.from_numpy(np.concatenate(far))
            h_far = None
            s_pad, gold, z = padded(Pd, s_full)
            loss = step_loss(s_pad, gold, z, "np")
            loss.backward()
            g = model.wo.weight.grad
            assert g is not None and float(g.abs().sum()) > 0, "no gradient reaches w_o"
            lp = step_loss(s_pad.detach(), gold, z, "p")
            assert lp is not None
            del h_far, far
            print(f"selftest {kind} L{L}: z at init, frontier cut exact (max err {err:.2e}), gradients flow; params {sum(p.numel() for p in model.parameters())}")
    # the rule p read places the twin's rank-1 node first
    s = torch.randn(3, 5)
    z = torch.randn(3, 5)
    Qr = [{"n": 5, "gold": np.eye(5, dtype=bool)[int(torch.argmax(z[i]))], "gt": 1} for i in range(3)]

    def m_of(score, gold, gt):
        return np.asarray([float(np.argmax(score) == np.flatnonzero(gold)[0])])
    assert all(x[0] == 1.0 for x in metrics_rows(s, z, [0, 1, 2], Qr, "p", m_of))
    print("selftest: rule p read places the twin's rank-1 first; all checks passed")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", default=None)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rule", default="np")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--balance", default="graph")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return
    if not (a.train and a.arms and a.out):
        raise SystemExit("--train, --arms and --out (or --selftest)")
    run(a)


if __name__ == "__main__":
    main()
