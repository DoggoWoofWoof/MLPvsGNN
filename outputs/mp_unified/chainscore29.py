"""Design look (untracked; not a result and not filed): S6 part 13, the same ideas built two ways. Swastik (5 Oct):
'the edge scorer is actually the message passing estimation but can be a better message passing using the rgcn
attention etc, we dont need to use it in the primitive form'; the MLP 'can do all the same things but it can never
pass any message, it is all static features ... without going back or reasoning further'; 'gnn and mlp improvements
both should happen but there is a distinction on how they are improving and the actual implementations of the same
ideas with the constraints of each model'. Part 4's posterior-typed GNN (ena-gr) used the chain posterior in its
primitive form: one scalar per message (the posterior mass of the chains whose step admits it), one temperature per
layer and direction, one shared transform. This part builds the GNN's message passing out (typed transforms, attention)
and gives the MLP the static counterpart (parameter-free propagated inputs), on part 9's training mix, beside the
incumbents. It is declared before any part 12 number.

Arms (lb and part 6's carried map pq on every arm; chainscore19's chain scorer g and EM coupling, T = 2):
    rga   the GNN f, relational with attention: two message layers over the row's graph, one channel per direction;
          a message's transform is a mix of four shared bases (W = sum_b c_kb V_b, RGCN's basis decomposition) chosen
          by its soft type k: untyped, and the chain posterior's mass on the none, dir and typed chains whose step
          admits it (part 4's a_l(e) split by level, never a relation id or name); attention over a node's in-edges
          is GATv2's (source and target states, and the message's typed mass); the posterior mass into a node by type
          enters as log1p (ena's u, per type). The GNN reruns on each EM round's posterior: it goes back.
    rgu   the same GNN with one soft type (untyped) and attention on the states only: no posterior; it runs once
    sg2   the MLP f over static propagated inputs: each node's standardised inputs, and their mean over its in-edges
          one and two steps back, per direction (SIGN's [x, P x, P^2 x]; no parameter, no posterior); the f scores
          stay fixed through the EM rounds: it never passes a message and never goes back
    c-rga, c-rgu, c-sg2   each f on webqsp fit-A: diagnostic oracles as in part 12 (its split, its epoch choice on
          fit-B, its reads), never zero-shot models and never in a zero-shot comparison as a model
Incumbents (not retrained; their rows files by sha256): ena-gr cs25-e-b-lb-pq-s0..2 (rga's), en-gr
cs25-n-b-lb-pq-s0..2 (rgu's), the MLP f cs24-k5-s0..2 (sg2's); part 12's c-ena, c-en and c-mlp for the oracles.
Training as part 9 for rga, rgu and sg2 (metaqa fit id and its four transforms, one group; the epoch by R@5 on
metaqa's select carve; the first build's input stats) and as part 12 for the oracles; Adam 1e-3, weight decay 1e-4,
12 epochs of 5,960 examples in batches of 64, seeds SEED + 0, 1, 2, on the host GPU with cs_dev's settings
(deterministic algorithms, TF32 off). Every arm reads the rows' messages from part 4's companions (cs20e-* and, for
passages, cs21e-*; checked against each build's sha256): sg2 needs them for its propagation. Reads in the run: the KB
rule on metaqa x1f, webqsp selectf and webqsp selectf + fit, the passage rule on 2wiki x1 and hotpotqa x1 (the
oracles: metaqa x1f and webqsp selectf). musique is not read in this part.
Grade, fixed at 10:02 on 5 Oct (this file's first write), before any part 13 run and before any part 12 number. Each
row's R@5 averaged over its three seeds; paired row bootstrap (BOOT 1000):
    H1 (primary) webqsp selectf + fit: rga minus ena-gr (the GNN f), sg2 minus the MLP f: ABOVE / AT / BELOW
    H2 metaqa x1f: the same differences
    H3 webqsp selectf, 2wiki x1 and hotpotqa x1: the same differences
    H4 every read: rgu minus en-gr (attention and bases without the posterior) and rga minus rgu (the posterior's
       types under them): HELPS / HURTS / SAME
    H5 static against message passing, on each KB read: the MLP's share of the GNN's lead over none/rd (the untyped
       walk), (MLP - walk) / (GNN - walk), for (sg2, rga) and for (the MLP f, ena-gr), with a row-bootstrap interval
    H6 webqsp selectf: each oracle minus its zero-shot arm (c-rga - rga, c-rgu - rgu, c-sg2 - sg2) beside part 12's
       (c-ena - ena-gr, c-en - en-gr, c-mlp - the MLP f) when those are in, and the share of the gap closed,
       1 - new gap / old gap (reported, not a verdict)
    S  every H1 to H4 difference per seed, with its sign
Decision. Per class, the candidate (rga for the GNN f, sg2 for the MLP f) carries if H1 is ABOVE and H2 not BELOW, or
H1 AT and H2 ABOVE; a candidate that carries is its class's model in the next part, and otherwise the incumbent stays.
If rga does not carry and rgu does by the same rule against en-gr, rgu is the GNN's next model and the posterior types
did not help under attention (H4 names it). Every label goes to Swastik beside part 12's (CAPACITY / SHARED /
CONFLICT): part 12 says where the zero-shot gap sits, this part whether each class's own build-out moves it.
Train-split rows throughout (webqsp's fit and selectf carves are the train split's): a look, not a result.
Smokes (--smoke: one short epoch, every read and the selection cut to --read-limit rows; never graded) gate the runs.
    python outputs/mp_unified/chainscore29.py train --arm rga --seed 0 --device cuda \
        --map-from outputs/mp_unified/lean/cs22.json --cache outputs/mp_unified/cache \
        --fit outputs/mp_unified/lean/cs19-mq-fit-id.npz --aug al4=... (sp4, mg3, rf) \
        --select outputs/mp_unified/lean/cs19-mq-select.npz --read metaqa=... --read webqsp=... --read webqsp_sf=... \
        --pread 2wiki=... --pread hotpotqa=... --edges fit=... (every input's companion) \
        --out X.json --rows-out X.rows.npz --state-out X.pt
    python outputs/mp_unified/chainscore29.py train --arm c-rga --seed 0 --device cuda --map-from ... --cache ... \
        --wqfit outputs/mp_unified/lean/cs19-wq-sf.npz --read metaqa=... --read webqsp=... --edges wqfit=... \
        (metaqa, webqsp) --out X.json --rows-out X.rows.npz --state-out X.pt
    python outputs/mp_unified/chainscore29.py grade --run X.json (every arm x seed) --z Z.json (every incumbent run
        and part 12's c-ena, c-en, c-mlp runs) --read metaqa=... --read webqsp=... --read webqsp_sf=... \
        --out outputs/mp_unified/lean/cs29.json
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainscore19's count
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainscore18 as CS  # noqa: E402
import chainscore19 as C19  # noqa: E402
import chainscore20 as C20  # noqa: E402
import chainscore21 as C21  # noqa: E402
import chainscore22 as C22  # noqa: E402
import chainscore24 as C24  # noqa: E402
import chainscore25 as C25  # noqa: E402
import chainscore28 as C28  # noqa: E402
import cs_cache as CC  # noqa: E402
import cs_dev as CD  # noqa: E402

SEED, BOOT = C19.SEED, C19.BOOT
BATCH = C21.BATCH
SEEDS = (0, 1, 2)
TF4 = C19.TRANSFORMS[1:]
HID, NLAYER, NEG = C19.HID, C20.NLAYER, C19.NEG
NB = 4              # shared bases per layer (both directions, every soft type)
HA = 16             # GATv2's attention width
LOGIT_B = 5.0       # attention logits bounded to (-5, 5) by 5 tanh(x / 5)
NSIGN = 5           # sg2's input blocks: x, and per direction P x and P^2 x
ARMS = {"rga": ("rga", "k5"), "rgu": ("rgu", "k5"), "sg2": ("sg2", "k5"),
        "c-rga": ("rga", "wq"), "c-rgu": ("rgu", "wq"), "c-sg2": ("sg2", "wq")}
KB_READS = C21.KB_READS             # metaqa, webqsp, webqsp_sf
P_RUN = ("2wiki", "hotpotqa")
WQ_READS = ("metaqa", "webqsp")
INC = {"rga": "cs25-e-b-lb-pq", "rgu": "cs25-n-b-lb-pq", "sg2": "cs24-k5"}
C_INC = {"c-rga": "c-ena", "c-rgu": "c-en", "c-sg2": "c-mlp"}
log, sha = C21.log, CC.sha


# ── the models ──────────────────────────────────────────────────────────────────────────────────────────────────


def make_rgnn(att):
    import torch
    nn = torch.nn
    K = 4 if att else 1

    class RGNN(nn.Module):
        """Two relational message layers, one channel per direction. A message's transform is sum_k tau_k(e) W_k with
        W_k = sum_b coef[k, b] V_b; tau = (1, a_none, a_dir, a_typed) with att, (1,) without. a_none and a_dir are
        constant over a row's messages of one direction, so their channels reuse the attended mean; a_typed varies
        per message and gets its own aggregate. Attention: GATv2 over each node's in-edges."""

        def __init__(self):
            super().__init__()
            self.att = att
            self.inp = nn.Sequential(nn.Linear(len(C19.NODEF), HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU())
            self.lin = nn.ModuleList([nn.Linear(HID, HID) for _ in range(NLAYER)])
            bound = 1.0 / math.sqrt(HID)
            self.V = nn.Parameter(torch.empty(NLAYER, NB, HID, HID).uniform_(-bound, bound))
            coef = torch.zeros(NLAYER, 2, K, NB)
            coef[:, :, 0, :] = torch.randn(NLAYER, 2, NB) / math.sqrt(NB)
            self.coef = nn.Parameter(coef)          # the typed rows start at 0: rga starts as rgu
            self.As = nn.Parameter(torch.randn(NLAYER, 2, HID, HA) * bound)
            self.Ad = nn.Parameter(torch.randn(NLAYER, 2, HID, HA) * bound)
            self.q = nn.Parameter(torch.zeros(NLAYER, 2, HA))     # uniform attention at the start: the mean
            if att:
                self.wt = nn.Parameter(torch.zeros(NLAYER, 2, HA))
                self.u = nn.Parameter(torch.zeros(NLAYER, 2, 3, HID))
            self.out = nn.Linear(HID, 1)

        def forward(self, X, E, A=None):
            F_ = torch.nn.functional
            h = self.inp(X)
            N = X.shape[0]
            for l_ in range(NLAYER):
                pre = self.lin[l_](h)
                for d in (0, 1):
                    src, dst, eix, _cnt = E[d]
                    if src.numel() == 0:
                        continue
                    zz = (h @ self.As[l_, d])[src] + (h @ self.Ad[l_, d])[dst]
                    a3 = None
                    if A is not None:
                        a3 = A[l_][eix]
                        zz = zz + a3[:, 2:3] * self.wt[l_, d]
                    lg = LOGIT_B * torch.tanh((F_.leaky_relu(zz, 0.2) @ self.q[l_, d]) / LOGIT_B)
                    ex = torch.exp(lg)
                    den = torch.zeros(N).index_add(0, dst, ex)
                    al = ex / den[dst].clamp_min(1e-12)
                    G = h[src]
                    W = torch.einsum("kb,bij->kij", self.coef[l_, d], self.V[l_])
                    if a3 is None:
                        agg0 = torch.zeros(N, HID).index_add(0, dst, al[:, None] * G)
                        msg = agg0 @ W[0].T
                    else:
                        w2 = torch.stack([al, al * a3[:, 2]], 1)
                        agg = torch.zeros(N, 2 * HID).index_add(0, dst, (w2[:, :, None] * G[:, None, :]).reshape(-1, 2 * HID))
                        agg0, aggt = agg[:, :HID], agg[:, HID:]
                        ab = torch.zeros(N, 2).index_add(0, dst, al[:, None] * a3[:, :2])
                        msg = (agg0 @ W[0].T + ab[:, 0:1] * (agg0 @ W[1].T) + ab[:, 1:2] * (agg0 @ W[2].T)
                               + aggt @ W[3].T)
                        mass = torch.zeros(N, 3).index_add(0, dst, a3)
                        pre = pre + torch.log1p(mass) @ self.u[l_, d]
                    pre = pre + msg
                h = h + torch.relu(pre)
            return self.out(h).squeeze(-1)

    return RGNN()


def make_model29(kind, seed):
    """sg2: chainscore20's MLP joint (em-gr, its g and coupling as made for that seed) with f over NSIGN input blocks.
    rga / rgu: chainscore20's GNN joint with this file's relational GNN."""
    import torch
    if kind == "sg2":
        m = C20.make_model("em-gr", seed)
        torch.manual_seed(seed + 1)
        m.f = torch.nn.Sequential(torch.nn.Linear(NSIGN * len(C19.NODEF), HID), torch.nn.ReLU(),
                                  torch.nn.Linear(HID, HID), torch.nn.ReLU(), torch.nn.Linear(HID, 1))
        m.kind, m.att = kind, False
        return m
    if kind not in ("rga", "rgu"):
        raise SystemExit(f"unknown f {kind}")
    T = C20.arm_spec("ena-gr")["T"]

    class Joint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.f = None
            self.gnn = make_rgnn(kind == "rga")
            self.g = torch.nn.Sequential(torch.nn.Linear(len(C19.PAIRF), HID), torch.nn.ReLU(),
                                         torch.nn.Linear(HID, HID), torch.nn.ReLU(), torch.nn.Linear(HID, 1))
            inv1 = math.log(math.e - 1.0)
            self.beta_r = torch.nn.Parameter(torch.tensor(inv1))
            self.gamma_r = torch.nn.Parameter(torch.tensor(inv1))
            self.leps = torch.nn.Parameter(torch.tensor(math.log(C19.EPS0)))
            self.T, self.att, self.kind = T, kind == "rga", kind

    torch.manual_seed(seed)
    return Joint()


def marginals3(lp, bt):
    """chainscore20's marginals split by level: per layer, (Ne x 3) the posterior mass of the none, dir and typed
    chains whose step admits each message."""
    import torch
    P = torch.exp(lp).reshape(-1)
    out = []
    for h in range(NLAYER):
        pi = torch.zeros(bt["S"] + 1)
        if bt["hp"][h].numel():
            pi = pi.index_add(0, bt["hs"][h], P[bt["hp"][h]])
        an, ad = pi[bt["xnone"]], pi[bt["xdir"]]
        at = torch.zeros_like(an)
        if bt["ie"].numel():
            at = at.index_add(0, bt["ie"], pi[bt["is"]])
        out.append(torch.stack([an, ad, at], 1))
    return out


def sign_inputs(X, E):
    """[x, P_0 x, P_0^2 x, P_1 x, P_1^2 x]: P_d the mean over a node's in-edges of direction d (no parameter)."""
    import torch
    out = [X]
    for d in (0, 1):
        src, dst, _eix, cnt = E[d]
        inv = (1.0 / cnt.clamp_min(1.0))[:, None]
        Z = X
        for _ in range(2):
            Z = torch.zeros_like(Z).index_add(0, dst, Z[src]) * inv if src.numel() else torch.zeros_like(Z)
            out.append(Z)
    return torch.cat(out, 1)


def forward29(model, bt):
    """s_T over each row's nodes and log p_T over its chains: chainscore19's forward over sg2's propagated inputs, or
    chainscore20's forward20 with this file's GNN and marginals3."""
    if model.kind == "sg2":
        b = dict(bt)
        b["Xn"] = sign_inputs(bt["Xn"], bt["E"])
        return C19.forward(model, b)
    import torch
    F_ = torch.nn.functional
    B, Ln, Lk = bt["B"], bt["Ln"], bt["Lk"]

    def fnode(A):
        h = model.gnn(bt["Xn"], bt["E"], A)
        return torch.zeros(B * Ln).index_copy(0, bt["npos"], h).reshape(B, Ln)

    z = torch.full((B * Lk,), NEG)
    if bt["Xp"] is not None:
        z = z.index_copy(0, bt["ppos"], model.g(bt["Xp"]).squeeze(-1))
    lp0 = C19.masked_logsoftmax(z.reshape(B, Lk) - bt["lc"], bt["pm"])
    f0 = fnode(marginals3(lp0, bt) if model.att else None)
    if bt["Xp"] is None:
        return f0 + F_.softplus(model.gamma_r) * model.leps, lp0
    gamma, eps = F_.softplus(model.gamma_r), torch.exp(model.leps)
    ip, inn, im = bt["ip"], bt["in"], bt["im"]

    def mstep(lp, f):
        agg = torch.zeros(B * Ln).index_add(0, inn, torch.exp(lp).reshape(-1)[ip] * im)
        return f + gamma * torch.log(eps + agg.reshape(B, Ln))

    if model.T == 0:
        return mstep(lp0, f0), lp0
    beta = F_.softplus(model.beta_r)
    s, lp = f0, lp0
    for _ in range(model.T):
        pi = torch.softmax(s.masked_fill(~bt["nm"], NEG), 1)
        ev = torch.zeros(B * Lk).index_add(0, ip, pi.reshape(-1)[inn] * im).reshape(B, Lk)
        lp = C19.masked_logsoftmax(lp0 + beta * torch.log(ev + 1e-8), bt["pm"])
        s = mstep(lp, fnode(marginals3(lp, bt)) if model.att else f0)
    return s, lp


def coupling29(model):
    out = C19.coupling(model)
    if model.kind != "sg2":
        g = model.gnn
        out["coef_norm"] = [[[round(float(v), 4) for v in g.coef.detach()[l_, d].norm(dim=-1).tolist()] for d in (0, 1)]
                            for l_ in range(NLAYER)]
        out["q_norm"] = [round(float(v), 4) for v in g.q.detach().norm(dim=-1).reshape(-1).tolist()]
        if model.att:
            out["wt_norm"] = [round(float(v), 4) for v in g.wt.detach().norm(dim=-1).reshape(-1).tolist()]
            out["u_norm"] = [round(float(v), 4) for v in g.u.detach().norm(dim=-1).reshape(-1).tolist()]
    return out


# ── batches, selection and reads ────────────────────────────────────────────────────────────────────────────────


def run_fwd(model, bt, device):
    import torch
    if CD.placed(device):
        with torch.device(device):
            return forward29(model, CD.to_dev(bt, device))
    return forward29(model, bt)


def row_batches29(d, c, lb, rows, st, att, batch=BATCH):
    """make_batch21's read batches (the KB rule's inputs, with the rows' messages) of the given rows, once, on the
    CPU."""
    out = []
    for b0 in range(0, rows.size, batch):
        r = rows[b0:b0 + batch]
        items = np.c_[np.zeros(r.size, np.int64), r, np.zeros(r.size, np.int64)]
        out.append((r, C21.make_batch21(items, [d], [c], None if lb is None else [lb], st, pairs=True, train=False,
                                        gnn=True, att=att)))
    return out


def select_r5(model, d, bl, device):
    import torch
    r5 = []
    with torch.no_grad():
        for rows, bt in bl:
            s = run_fwd(model, bt, device)[0].double().cpu().numpy()
            r5 += [C19.rank_row(d, i, s[j, :int(d["n"][i])])[0] for j, i in enumerate(rows)]
    return float(np.mean(r5))


def read29(model, d, c, lb, st, rule, device=None, batch=BATCH, limit=None):
    """cs_dev's read_dev through this file's forward (every f reads the rows' messages); limit: the first rows only
    (a smoke)."""
    import torch
    rank = {"kb": C19.rank_row, "passage": C21.rank_row_p}[rule]
    N = d["n"].size if limit is None else min(int(limit), int(d["n"].size))
    out = np.zeros((N, 3))
    L = d["level_names"].size
    ml = np.zeros(L)
    plev = d["p_lev"].astype(np.int64)
    with torch.no_grad():
        for b0 in range(0, N, batch):
            rows = np.arange(b0, min(N, b0 + batch))
            items = np.c_[np.zeros(rows.size, np.int64), rows, np.zeros(rows.size, np.int64)]
            bt = C21.make_batch21(items, [d], [c], None if lb is None else [lb], st, pairs=True, train=False, gnn=True,
                                  att=model.att)
            s, lp = run_fwd(model, bt, device)
            s = s.double().cpu().numpy()
            Pm = np.exp(lp.double().cpu().numpy())
            for j, i in enumerate(rows):
                out[i] = rank(d, i, s[j, :int(d["n"][i])])
                p0, p1 = int(d["rowoff"][i]), int(d["rowoff"][i + 1])
                if p1 > p0:
                    ml += np.bincount(plev[p0:p1], weights=Pm[j, :p1 - p0], minlength=L)
    return out, ml / max(N, 1)


# ── training ────────────────────────────────────────────────────────────────────────────────────────────────────


def train29(arm, kind, builds, comps, LB, groups, sels, st, epochs, per_epoch, batch, seed, device=None, threads=1):
    """chainscore28's train28 statements with this file's model, batch (every f gets the rows' messages) and
    forward; the epoch by the mean R@5 over the selection sets; the CUDA peak memory per epoch."""
    import torch
    block = CD.settings(device, threads)
    att = kind == "rga"
    model = make_model29(kind, seed)
    cuda = CD.placed(device) and str(device).startswith("cuda")
    if CD.placed(device):
        CD._dp().model_to(model, device)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.Adam(model.parameters(), lr=C19.LR, weight_decay=C19.WD)
    rng = np.random.default_rng(seed)
    sb = [(sd, row_batches29(sd, sc_, slb, rows, st, att)) for sd, sc_, slb, rows in sels]
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
                bt = C21.make_batch21(items[b0:b0 + batch], builds, comps, LB, st, pairs=True, gnn=True, att=att)
                if CD.placed(device):
                    bt = CD.to_dev(bt, device)
                    with torch.device(device):
                        s, lp = forward29(model, bt)
                        loss, ln, lc = C19.objective(s, lp, bt)
                else:
                    s, lp = forward29(model, bt)
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
            parts = [select_r5(model, sd, bl, device) for sd, bl in sb]
            sr5 = float(np.mean(parts))
            peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if cuda else None
            hist.append({"epoch": ep, "loss": round(tot / max(nb, 1), 5), "node_loss": round(tn / max(nb, 1), 5),
                         "chain_loss": round(tc / max(nb, 1), 5), "select_r5": round(sr5, 5),
                         "select_parts": [round(x, 5) for x in parts],
                         "by_build": np.bincount(items[:, 0], minlength=len(builds)).tolist(),
                         **coupling29(model), "cuda_peak_gb": peak, "seconds": round(time.time() - t1, 1)})
            log(f"    {arm} ({kind}) ep {ep}: loss {tot / max(nb, 1):.4f} (node {tn / max(nb, 1):.4f}, chain "
                f"{tc / max(nb, 1):.4f}) select R@5 {sr5:.4f} {[round(x, 4) for x in parts]} peak {peak} GB "
                f"{time.time() - t1:.0f}s")
            if best is None or sr5 > best[1]:
                best = (ep, sr5)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    det = sorted({str(w.message)[:300] for w in caught if "deterministic" in str(w.message).lower()})
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": hist, "best_epoch": best[0], "best_select_r5": round(best[1], 5), "stats": st,
                   "coupling": coupling29(model), "placement": block, "state_sha256": CD.state_sha(best_state),
                   "seconds_per_batch": round(tb / max(nbt, 1), 4), "deterministic_warnings": det,
                   "cuda_peak_gb": max([h["cuda_peak_gb"] or 0.0 for h in hist] or [0.0]),
                   "n_params": int(sum(p.numel() for p in model.parameters()))}


def train_cmd(a):
    t0 = time.time()
    if a.arm not in ARMS:
        raise SystemExit(f"unknown arm {a.arm}; one of {list(ARMS)}")
    kind, data = ARMS[a.arm]
    if a.map_from:
        minfo, norm = C25.carried_map(a.map_from), "pq"
    elif a.smoke:
        norm, minfo = a.map, {"map_given": a.map}
    else:
        raise SystemExit("give --map-from (part 6's grade; the runs) or --smoke --map (a smoke)")
    if a.read_limit is not None and not a.smoke:
        raise SystemExit("--read-limit is a smoke's")
    dev = None if a.device == "cpu" else a.device
    att = kind == "rga"
    res = {"look": "chainscore29", "arm": a.arm, "seed": a.seed, "f": kind, "data": data, "norm": norm,
           "device": a.device, "args": dict(vars(a)), "map_rule": minfo, "cache": a.cache, "smoke": bool(a.smoke),
           "state_out": a.state_out, "script_sha256": sha(__file__), "chainscore28_sha256": sha(C28.__file__),
           "cs_cache_sha256": sha(CC.__file__), "chainscore25_sha256": sha(C25.__file__),
           "chainscore24_sha256": sha(C24.__file__), "chainscore22_sha256": sha(C22.__file__),
           "chainscore21_sha256": sha(C21.__file__), "chainscore20_sha256": sha(C20.__file__),
           "chainscore19_sha256": sha(C19.__file__), "chainscore18_sha256": sha(CS.__file__),
           "cs_dev_sha256": sha(CD.__file__), "inputs": {}, "inputs_sha256": {}, "passage_types": {}, "maps": {},
           "edges": {}, "reads": {}, "rows_out": a.rows_out, "timing": {}}
    aug = dict(x.split("=", 1) for x in a.aug)
    kb = [tuple(x.split("=", 1)) for x in a.read]
    ps = [tuple(x.split("=", 1)) for x in a.pread]
    if data == "k5":
        if sorted(aug) != sorted(TF4) or not a.fit or not a.select or a.wqfit:
            raise SystemExit(f"a k5 arm takes --fit, --select and --aug for each of {TF4}, and no --wqfit")
        if not a.smoke and (sorted(n for n, _ in kb) != sorted(KB_READS) or sorted(n for n, _ in ps) != sorted(P_RUN)):
            raise SystemExit(f"a k5 arm reads {KB_READS} (--read) and {P_RUN} (--pread)")
        train_in = [("fit", a.fit)] + [(t, aug[t]) for t in TF4]
    else:
        if aug or a.fit or a.select or ps or not a.wqfit:
            raise SystemExit("an oracle trains on webqsp fit-A alone (--wqfit): no --fit, --aug, --select or --pread")
        if not a.smoke and sorted(n for n, _ in kb) != sorted(WQ_READS):
            raise SystemExit(f"an oracle reads {WQ_READS}")
        train_in = [("wqfit", a.wqfit)]
    reads = [(n, p, False) for n, p in kb] + [(n, p, True) for n, p in ps]
    path = dict(train_in)
    if data == "k5":
        path["select"] = a.select
    path.update({n: p for n, p, _ in reads})
    edges = dict(x.split("=", 1) for x in a.edges)
    if sorted(edges) != sorted(path):
        raise SystemExit(f"every f takes --edges for each of {sorted(path)}; given {sorted(edges)}")
    t1 = time.time()
    for n, p in path.items():
        res["inputs_sha256"][p] = sha(p)
        res["inputs_sha256"][edges[n]] = sha(edges[n])
    for n, p, is_p in reads:
        if is_p:
            side = C21.check_passage(p, res["inputs_sha256"][p], res["inputs_sha256"][edges[n]])
            res["passage_types"][n] = {"script_sha256": side["script_sha256"], **side["types"]}
    res["timing"]["hash_s"] = round(time.time() - t1, 1)

    def load(p, kind_):
        if a.cache:
            return C25.get(p, kind_, norm, a.cache)
        d, lb, info = CC.pipeline(p, kind_, norm)
        return d, lb, info, (C19.input_stats(d) if kind_ == "train" else None)

    def comp(n, d):
        return C25.load_comp(edges[n], d, res["inputs_sha256"][path[n]], att)

    lim = a.read_limit
    t1 = time.time()
    builds, comps, LB, st0, wq = [], [], [], None, None
    for n, p in train_in:
        d, lb, info, st = load(p, "train")
        tf = d["meta"]["transform"]["name"]
        if tf != (n if n in TF4 else "id"):
            raise SystemExit(f"{p} holds transform {tf}")
        if n == "wqfit":
            if d["meta"]["args"]["ds"] != "webqsp" and not a.smoke:
                raise SystemExit(f"{p} is not a webqsp build")
            if a.smoke and "fitf" not in list(d["meta"]["carves"]):
                rows = np.arange(d["n"].size, dtype=np.int64)       # a smoke on selectf: its rows stand in
                perm = rows[np.random.default_rng(SEED + 28).permutation(rows.size)]
                nb_ = int(round(C28.HOLD_B * rows.size))
                fa, fb = np.sort(perm[nb_:]), np.sort(perm[:nb_])
            else:
                fa, fb = C28.fit_split(d)
            d["train_rows"] = np.intersect1d(d["train_rows"], fa)
            res["wq_split"] = {"fit_a": int(fa.size), "fit_b": int(fb.size), "train_rows_a": int(d["train_rows"].size),
                               "fit_b_sha256": CC.arr_sha(fb)}
            wq = (d, lb, fb)
        if not builds:
            st0 = st
        res["inputs"][n] = d["meta"]
        res["maps"][n] = info
        builds.append(d)
        comps.append(comp(n, d))
        LB.append(lb)
        log(f"  {a.arm} s{a.seed}: {n} ready ({'cache' if a.cache else 'pipeline'}; {info})")
    if st0 is None:
        raise SystemExit("no input stats for the first build")
    groups = [list(range(len(builds)))]
    if data == "k5":
        Sd, Slb, sinfo, _ = load(a.select, "plain")
        srows = np.arange(Sd["n"].size, dtype=np.int64)
        sels = [(Sd, comp("select", Sd), Slb, srows[:lim] if lim else srows)]
        res["maps"]["select"] = sinfo
        res["inputs"]["select"] = Sd["meta"]
    else:
        wd, wlb, fb = wq
        sels = [(wd, comps[0], wlb, fb[:lim] if lim else fb)]
    res["groups"] = groups
    res["train_rows"] = {n: int(b["train_rows"].size) for (n, _p), b in zip(train_in, builds)}
    res["edges"] = {n: {"meta_check": c["meta"]["check"], "steps": c["meta"]["steps"]}
                    for (n, _p), c in zip(train_in, comps)}
    res["timing"]["load_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    model, info = train29(a.arm, kind, builds, comps, LB, groups, sels, st0, a.epochs, a.per_epoch, a.batch,
                          SEED + a.seed, device=dev, threads=a.threads)
    res["timing"]["train_s"] = round(time.time() - t1, 1)
    del builds, comps, LB, sels, wq
    res.update({k: info[k] for k in ("history", "best_epoch", "best_select_r5", "coupling", "placement",
                                     "state_sha256", "seconds_per_batch", "deterministic_warnings", "cuda_peak_gb",
                                     "n_params")})
    t1 = time.time()
    rows_out = {}
    for n, p, is_p in reads:
        t2 = time.time()
        d, lb, rinfo, _ = load(p, "plain")
        c = comp(n, d)
        res["maps"][n] = rinfo
        rule = "passage" if is_p else "kb"
        xr, ml = read29(model, d, c, lb, info["stats"], rule, dev, batch=a.read_batch, limit=lim)
        D = d["D"][:xr.shape[0]]
        rows_out[n] = xr.astype(np.float32)
        rows_out[f"{n}__D"] = D
        res["reads"][n] = dict(C19.summarise(xr, D, ml, d["level_names"]), rule=rule, seconds=round(time.time() - t2, 1))
        res["inputs"][n] = d["meta"]
        log(f"  {a.arm} s{a.seed} on {n} ({rule} rule): {res['reads'][n]['mean']} ({time.time() - t2:.0f}s)")
        del d, c
    res["timing"]["reads_s"] = round(time.time() - t1, 1)
    p = Path(a.rows_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **rows_out)
    os.replace(tmp, p)
    if a.state_out:
        import torch
        sp_ = Path(a.state_out)
        torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "stats": info["stats"],
                    "arm": a.arm, "seed": a.seed, "f": kind, "norm": norm}, sp_.with_name(sp_.name + ".tmp"))
        os.replace(sp_.with_name(sp_.name + ".tmp"), sp_)
    res["seconds"] = round(time.time() - t0, 1)
    log(f"  {a.arm} s{a.seed}: state {res['state_sha256'][:16]}, {res['seconds_per_batch']} s/batch, peak "
        f"{res['cuda_peak_gb']} GB, {res['n_params']} params, timing {res['timing']}, {res['seconds']}s")
    return res


# ── grade ───────────────────────────────────────────────────────────────────────────────────────────────────────


def stem_seed(js):
    look = js.get("look")
    if look == "chainscore29" and not js.get("smoke"):
        return js["arm"], int(js["seed"])
    if look == "chainscore28" and js.get("arm") in C_INC.values() and not js.get("smoke"):
        return js["arm"], int(js["seed"])
    if look == "chainscore24" and js.get("arm") == "k5":
        return "cs24-k5", int(js["seed"])
    if look == "chainscore25" and js.get("arm") in ("n-b-lb-pq", "e-b-lb-pq") and not js.get("via_dev"):
        return f"cs25-{js['arm']}", int(js["seed"])
    raise SystemExit(f"not a run this grade reads: {look} {js.get('arm')}")


def label(ci, pos="ABOVE", neg="BELOW", zero="AT"):
    return pos if ci[1] > 0 else neg if ci[2] < 0 else zero


def grade_cmd(a):
    t0 = time.time()
    runs, rows, inputs = {}, {}, {}
    for p_ in list(a.run) + list(a.z):
        js = json.loads(Path(p_).read_text(encoding="utf-8"))
        key = stem_seed(js)
        if js.get("norm") != "pq":
            raise SystemExit(f"{p_}: map {js.get('norm')}, not pq")
        if key in runs:
            raise SystemExit(f"two runs of {key}")
        runs[key] = {"path": p_, "sha256": sha(p_), "state_sha256": js.get("state_sha256"),
                     "best_epoch": js.get("best_epoch"), "rows_sha256": sha(js["rows_out"])}
        with np.load(js["rows_out"]) as R:
            rows[key] = {k: R[k][:, 0].astype(np.float64) for k in R.files if not k.endswith("__D")}
        inputs[key] = js.get("inputs_sha256", {})
    walk = {}
    for x in a.read:
        nm_, p_ = x.split("=", 1)
        h = sha(p_)
        for k in runs:
            if nm_ in rows[k] and inputs[k].get(p_) != h:
                raise SystemExit(f"run {k} read another {nm_} than {p_}")
        d = CS.load(p_)
        walk[nm_] = CS.reference_arms(d)["none/rd"][:, 0].astype(np.float64)
        del d
    rng = np.random.default_rng(SEED + 29)

    def have(stem, rd):
        return all((stem, s) in rows and rd in rows[(stem, s)] for s in SEEDS)

    def m(stem, rd):
        return np.mean([rows[(stem, s)][rd] for s in SEEDS], 0)

    out = {"look": "chainscore29", "script_sha256": sha(__file__),
           "runs": {f"{k[0]}-s{k[1]}": v for k, v in sorted(runs.items())}, "H": {}, "S": {}, "H5": {}, "H6": {},
           "means": {}, "decision": {}, "not_read": []}
    for stem in sorted({k[0] for k in runs}):
        out["means"][stem] = {rd: round(float(m(stem, rd).mean()), 5) for rd in KB_READS + P_RUN if have(stem, rd)}
    pairs = {"H_gnn": ("rga", INC["rga"]), "H_mlp": ("sg2", INC["sg2"]), "H4_ctl": ("rgu", INC["rgu"]),
             "H4_typ": ("rga", "rgu")}
    for lab_, (x, y) in pairs.items():
        out["H"][lab_], out["S"][lab_] = {}, {}
        for rd in KB_READS + P_RUN:
            if not (have(x, rd) and have(y, rd)):
                out["not_read"].append(f"{lab_}:{rd}")
                continue
            ci = boot_ci(m(x, rd) - m(y, rd), rng)
            lb_ = label(ci) if lab_ in ("H_gnn", "H_mlp") else label(ci, "HELPS", "HURTS", "SAME")
            out["H"][lab_][rd] = {"diff": ci, "label": lb_}
            out["S"][lab_][rd] = [round(float((rows[(x, s)][rd] - rows[(y, s)][rd]).mean()), 5) for s in SEEDS]
    for rd in KB_READS:
        if rd not in walk:
            continue
        w = walk[rd]
        out["H5"][rd] = {}
        for lab_, (mlp, gnn) in (("new", ("sg2", "rga")), ("old", (INC["sg2"], INC["rga"]))):
            if not (have(mlp, rd) and have(gnn, rd)):
                continue
            a_, b_ = m(mlp, rd) - w, m(gnn, rd) - w
            idx = rng.integers(a_.size, size=(BOOT, a_.size))
            num, den = a_[idx].mean(1), b_[idx].mean(1)
            ok = np.abs(den) > 1e-9
            r = num[ok] / den[ok]
            out["H5"][rd][lab_] = {"rho": round(float(a_.mean() / b_.mean()), 4) if abs(b_.mean()) > 1e-9 else None,
                                   "ci": [round(float(np.quantile(r, 0.025)), 4), round(float(np.quantile(r, 0.975)), 4)]
                                   if r.size else None, "mlp_lead": round(float(a_.mean()), 5),
                                   "gnn_lead": round(float(b_.mean()), 5)}
    for c_arm, z_arm in (("c-rga", "rga"), ("c-rgu", "rgu"), ("c-sg2", "sg2")):
        new = boot_ci(m(c_arm, "webqsp") - m(z_arm, "webqsp"), rng) if have(c_arm, "webqsp") and have(
            z_arm, "webqsp") else None
        oc, oz = C_INC[c_arm], INC[z_arm]
        old = boot_ci(m(oc, "webqsp") - m(oz, "webqsp"), rng) if have(oc, "webqsp") and have(oz, "webqsp") else None
        closed = (round(1.0 - new[0] / old[0], 4) if new is not None and old is not None and abs(old[0]) > 1e-9
                  else None)
        out["H6"][z_arm] = {"new_gap": new, "old_gap": old, "old_pair": [oc, oz], "share_closed": closed}
    for cls, (cand, h) in (("gnn", ("rga", "H_gnn")), ("mlp", ("sg2", "H_mlp"))):
        hh = out["H"][h]
        if "webqsp_sf" not in hh or "metaqa" not in hh:
            out["decision"][cls] = {"candidate": cand, "label": "NOT_READ"}
            continue
        h1, h2 = hh["webqsp_sf"]["label"], hh["metaqa"]["label"]
        carries = (h1 == "ABOVE" and h2 != "BELOW") or (h1 == "AT" and h2 == "ABOVE")
        out["decision"][cls] = {"candidate": cand, "H1": h1, "H2": h2, "carries": carries}
    ctl = out["H"]["H4_ctl"]
    if not out["decision"]["gnn"].get("carries") and "webqsp_sf" in ctl and "metaqa" in ctl:
        c1, c2 = ctl["webqsp_sf"]["label"], ctl["metaqa"]["label"]
        out["decision"]["gnn"]["rgu_carries"] = (c1 == "HELPS" and c2 != "HURTS") or (c1 == "SAME" and c2 == "HELPS")
    out["seconds"] = round(time.time() - t0, 1)
    return out


def boot_ci(x, rng):
    return C28.boot_ci(x, rng)


# ── main ────────────────────────────────────────────────────────────────────────────────────────────────────────


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu")
    t.add_argument("--threads", type=int, default=1)
    t.add_argument("--map-from")
    t.add_argument("--map", default="pq")
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--read-limit", type=int)
    t.add_argument("--cache")
    t.add_argument("--wqfit")
    t.add_argument("--fit")
    t.add_argument("--aug", action="append", default=[])
    t.add_argument("--select")
    t.add_argument("--read", action="append", default=[])
    t.add_argument("--pread", action="append", default=[])
    t.add_argument("--edges", action="append", default=[])
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--per-epoch", type=int, default=5960)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--read-batch", type=int, default=BATCH)
    t.add_argument("--out", required=True)
    t.add_argument("--rows-out", required=True)
    t.add_argument("--state-out")
    r = sub.add_parser("grade")
    r.add_argument("--run", action="append", required=True)
    r.add_argument("--z", action="append", required=True)
    r.add_argument("--read", action="append", default=[])
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"train": train_cmd, "grade": grade_cmd}[a.cmd](a)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                   encoding="utf-8")
    os.replace(tmp, p)
    log(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
