"""Design look (untracked; not a result and not filed): do learned relations and path operators carry to relations and
graphs the fit never saw?

Every anchor round so far reads a model on the graph and the phrase vocabulary it was fitted on. The best 2wiki form gives
each of the top 256 phrases a free vector (t1[k]): a phrase outside the fit's vocabulary, a relation no training row used,
or a graph whose edges carry no phrase at all has nothing to read but the 'other' vector. This look fits a model on one
dataset's carves and reads it under four conditions:
    ID      the fit's own graph and phrases (carve x1's half B, as every round)
    NR      no relations: every structural phrase replaced by 'other' (an untyped graph; family x direction kept)
    MASK / REV   with --hold h/F: the phrases whose sha256 fold (F folds of the phrase string) is h are hidden in
            training, selection and MASK (their edges typed 'other'), and shown at REV, so REV - MASK is what a model
            makes of relations it never saw; read on all of B and on the B rows where a gold sits one structural edge
            from a seed through a hidden phrase (the rows the hidden relations matter for)
    XD      (mode --xread) a saved model read on another dataset's carve x1 half B: zero-shot transfer to a new graph,
            its own phrases, its own twin and GNN; also XD-NR
The relation forms:
    free    lin (anchor_walk5's), the anchor_walk6 operators (trans rope ropet house affine gate): a vector or operator
            per phrase token. At XD a phrase is matched by its string to the fit's vocabulary, else 'other'.
    text    txt0, txtc (anchor_walk7's TxtMix, unchanged): a phrase token's vector is P phi(k) + D[family x direction],
            phi the phrase's gte-Qwen2 vector (txtc adds the zero-shot term gamma * cos(q, phi)).
    gen     g<op>[c], op in lin trans rope ropet house affine gate: a vocabulary-free path operator. A phrase token's
            code is h(k) = P phi(k) + D[f(k)] (P drawn at std 0.1, the free vectors' scale); 'other', ner and knn keep
            one code each (five codes, none tied to a phrase). Each operator's parameters are linear in the code:
              glin   e = bk[b] + h(a) + h'(c) + ln[L]            the additive form; h'(k) = W' P phi(k) + D'[f(k)] at position 2
              gtrans O_a(x) = x + h(a)                           commutative
              grope  O_a(x) = R(W h(a) + w0) x                   a rotation per 2-plane: commutative
              gropet O_a(x) = R(W h(a) + w0) x + h(a)            rotation then translation: order matters
              ghouse O_a(x) = H(u2(a)) H(u1(a)) x + h(a)         u_j = W_j h(a) + b_j: non-commutative, invertible
              gaffine O_a(x) = x + U(a) V(a)^T x + h(a)          U, V linear in h(a), rank 4: can be singular
              ggate  O_a(x) = sigmoid(W h(a) + 2) * x + h(a)     lossy: an earlier relation's trace shrinks
            and suffix c adds gamma1[f(a)] cos(q, phi(a)) + gamma2[f(c)] cos(q, phi(c)) to the logit (zero-shot match).
            A phrase the fit never saw gets its operator from its text, so REV and XD read it; with no phrase the model is
            a family x direction model.
The fit loop, the loss (gold cross-entropy, optionally the GNN teacher of anchor_walk8), the rules, the selection and the
reads are anchor_walk8.fit_read_kd and anchor_walk6.read_rule, imported unchanged; the rows load through anchor_walk11's
frontier-pruned loader (checked against the full rows on every look's first chunk). Nothing reads a neighbour's score or
state: a type is (seed bucket, its edges' relation tokens) and its bonus is the same for every node it reaches.

    <train>:<base>/<model>[/<rule>][@seeds]   base T0, T0-1, A{256,1024}-1 (one edge) or A{256,1024} (two edges)
    python outputs/mp_approx_2wiki_anchor/host/anchor_gen.py --dataset 2wiki --variants x4+x5+x6:A1024/ghouse [--hold 0/5]
    python outputs/mp_approx_2wiki_anchor/host/anchor_gen.py --xread --target hotpotqa --models a.pt,b.pt --out PATH
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import anchor_walk11 as AW11  # noqa: E402

AW10, AW8 = AW11.AW10, AW11.AW8
AW7, AW6, A16, AW = AW8.AW7, AW8.AW6, AW8.A16, AW8.AW
AW5, AW3, G2 = AW6.AW5, AW8.AW3, AW8.G2
PINS = {"anchor_walk11": "e782089f33cae0016ad4bbf09f14e092b0da7eaf31932c5d437d42f274d0f99b",
        "anchor_walk8": "7e4aa326324f44b3bce7ecee2afd7e79b09c74fda8fa692509f9adb5063b5768",
        "anchor_walk7": "e153df784112dd9dd66130dbd86a8ac7357956d94be938d2dc30c4876d22f969",
        "anchor_walk6": "441d68bd85212d0bb1264b8ec269b9f5d440328b7869614c4287c67ca3ab49de"}
DATASETS = ("2wiki", "hotpotqa", "musique")
GEN = ("lin", "trans", "rope", "ropet", "house", "affine", "gate")
RANK = 4
R_TOP = 4096          # the compact stores the top 4096 phrase strings; a rank at or above it is 'other' for every K
log = A16.log


def famdir_of_tokens(nt, K):
    return AW7.famdir_of_tokens(nt, K)


def phi_layout(nt, K, phi):
    """TxtMix's layout: the phrase table's first K rows at every structural direction's ranks, 0 elsewhere."""
    Phi = np.zeros((nt, phi.shape[1]), dtype=np.float32)
    pm = np.zeros(nt, dtype=bool)
    for dd in range(3):
        Phi[dd * (K + 1):dd * (K + 1) + K] = phi[:K]
        pm[dd * (K + 1):dd * (K + 1) + K] = True
    return torch.as_tensor(Phi), torch.as_tensor(pm)


class GenMix(A16.Mix):
    """A vocabulary-free path operator: each token's parameters are linear in its code h(k) (see the docstring)."""

    def __init__(self, nt, K, phi, kind, cos=False, d=64, qdim=1536):
        super().__init__(nt, d, qdim)   # A, t1 (the five non-phrase codes), t2 (glin's position 2), bk, ln, c, nu, c_null, kappa, beta
        if nt != 3 * (K + 1) + 2 or phi.shape[0] < K:
            raise SystemExit("GenMix: the token count or the phrase table does not match K")
        Phi, pm = phi_layout(nt, K, phi)
        self.register_buffer("Phi", Phi, persistent=False)
        self.register_buffer("pmask", pm, persistent=False)
        self.register_buffer("fdir", torch.as_tensor(famdir_of_tokens(nt, K)), persistent=False)
        self.kind, self.cos, self.d, self.K = kind, cos, d, K
        self.P = torch.nn.Linear(qdim, d, bias=False)
        torch.nn.init.normal_(self.P.weight, std=0.1)
        self.D = torch.nn.Parameter(torch.zeros(5, d))
        if kind == "lin":
            self.W2 = torch.nn.Linear(d, d, bias=False)    # position 2's code W2 P phi + D2[f]: d^2 more, not another qdim x d
            torch.nn.init.normal_(self.W2.weight, std=d ** -0.5)
            self.D2 = torch.nn.Parameter(torch.zeros(5, d))
        else:
            self.s0 = torch.nn.Parameter(torch.randn(2, d) * 0.1)
        if kind in ("rope", "ropet"):
            self.W_th = torch.nn.Linear(d, d // 2)
            torch.nn.init.normal_(self.W_th.weight, std=1.0)
            with torch.no_grad():
                self.W_th.bias.uniform_(-math.pi, math.pi)
            if kind == "rope":
                with torch.no_grad():
                    self.s0.normal_()
        if kind == "house":
            self.W_u = torch.nn.Linear(d, 2 * d)
            torch.nn.init.normal_(self.W_u.weight, std=1.0)
            torch.nn.init.normal_(self.W_u.bias, std=1.0)
        if kind == "affine":
            self.W_U = torch.nn.Linear(d, d * RANK)
            self.W_V = torch.nn.Linear(d, d * RANK)
            for lin in (self.W_U, self.W_V):
                torch.nn.init.normal_(lin.weight, std=0.125)
                torch.nn.init.zeros_(lin.bias)
        if kind == "gate":
            self.W_g = torch.nn.Linear(d, d)
            torch.nn.init.normal_(self.W_g.weight, std=0.125)
            torch.nn.init.constant_(self.W_g.bias, 2.0)
        if cos:
            self.gam = torch.nn.Parameter(torch.zeros(2, 5))

    def set_phi(self, phi):
        Phi, pm = phi_layout(self.nt, self.K, phi)
        self.Phi, self.pmask = Phi, pm

    def codes(self):
        base = self.P(self.Phi)
        h = torch.where(self.pmask[:, None], base + self.D[self.fdir], self.t1)
        if self.kind != "lin":
            return h, None
        h2 = torch.where(self.pmask[:, None], self.W2(base) + self.D2[self.fdir], self.t2[1:])
        return h, h2

    def op(self, x, k, h):
        kind, hk = self.kind, h[k]
        if kind == "trans":
            return x + hk
        if kind in ("rope", "ropet"):
            x = AW6.rot(x, self.W_th(hk))
            return x if kind == "rope" else x + hk
        if kind == "house":
            u = self.W_u(hk)
            for j in range(2):
                uj = u[..., j * self.d:(j + 1) * self.d]
                x = x - 2.0 * ((x * uj).sum(-1, keepdim=True) / ((uj * uj).sum(-1, keepdim=True) + 1e-8)) * uj
            return x + hk
        if kind == "affine":
            U = self.W_U(hk).unflatten(-1, (self.d, RANK))
            V = self.W_V(hk).unflatten(-1, (self.d, RANK))
            t = torch.einsum("btdr,btd->btr", V, x)
            return x + torch.einsum("btdr,btr->btd", U, t) + hk
        return torch.sigmoid(self.W_g(hk)) * x + hk

    def forward(self, qemb, tb, t1, t2, tmask, ent_q, ent_t, ent_node, z, nmask):
        aq = self.A(qemb)
        L = (t2 > 0).long()
        h, h2 = self.codes()
        k2 = (t2 - 1).clamp(min=0)
        if self.kind == "lin":
            e = self.bk[tb] + h[t1] + torch.where((t2 > 0)[..., None], h2[k2], torch.zeros_like(h2[k2])) + self.ln[L]
        else:
            x = self.op(self.s0[tb], t1, h)
            x = torch.where((t2 > 0)[..., None], self.op(x, k2, h), x)
            e = x + self.ln[L]
        w = (aq[:, None, :] * e).sum(-1) + self.c[tb, L]
        if self.cos:
            cosv = qemb @ self.Phi.T                    # (B, nt); both sides unit, 0 for every non-phrase token
            w = w + self.gam[0][self.fdir[t1]] * cosv.gather(1, t1)
            w = w + torch.where(t2 > 0, self.gam[1][self.fdir[k2]] * cosv.gather(1, k2), torch.zeros_like(w))
        w = w.masked_fill(~tmask, float("-inf"))
        null = aq @ self.nu + self.c_null
        return torch.softmax(torch.cat([w, null[:, None]], 1), 1)[:, :-1]

    def params_used(self):
        n = sum(p.numel() for name, p in self.named_parameters() if name not in ("t1", "t2", "bk") or (name == "bk" and self.kind == "lin"))
        n += 5 * self.d                      # the five non-phrase codes of t1
        if self.kind == "lin":
            n += 5 * self.d                  # and of t2
        return int(n)


def split_model(model):
    """(family, op, cos): family free / text / gen."""
    if model in AW7.TXT:
        return "text", model, False
    if model.startswith("g") and model != "gate":
        core = model[1:]
        cos = core.endswith("c") and core[:-1] in GEN
        core = core[:-1] if cos else core
        if core in GEN:
            return "gen", core, cos
    if model == "lin" or model in AW6.OPS:
        return "free", model, False
    raise SystemExit(f"unknown model {model}")


def parse(name, train_looks):
    head, _at, seeds = name.partition("@")
    train, sep, rest = head.partition(":")
    if not sep:
        raise SystemExit(f"{name}: no train set")
    looks = train.split("+")
    if not looks or any(lk not in train_looks for lk in looks) or len(set(looks)) != len(looks):
        raise SystemExit(f"{name}: train looks must be among {train_looks}")
    parts = rest.split("/")
    if len(parts) > 3:
        raise SystemExit(f"{name}: too many fields")
    base, model, rule = parts[0], (parts[1] if len(parts) > 1 else "lin"), (parts[2] if len(parts) > 2 else "p")
    if base in ("T0", "T0-1"):
        tok, K, max_len = "T0", None, (2 if base == "T0" else 1)
    elif base.startswith("A") and base[1:].removesuffix("-1").isdigit() and int(base[1:].removesuffix("-1")) in (256, 1024):
        tok, K, max_len = "A", int(base[1:].removesuffix("-1")), (1 if base.endswith("-1") else 2)
    else:
        raise SystemExit(f"{name}: unknown base {base}")
    fam, op, cos = split_model(model)
    if tok == "T0" and fam != "free":
        raise SystemExit(f"{name}: T0 carries no phrase, so only the free models read it")
    if fam == "text" and max_len != 1:
        raise SystemExit(f"{name}: the text models read one-edge bases only")
    if fam == "free" and op != "lin" and max_len != 2:
        raise SystemExit(f"{name}: the free operators read two-edge bases only")
    if fam == "gen" and op != "lin" and max_len != 2:
        raise SystemExit(f"{name}: the gen operators read two-edge bases only (on one edge they are a reparametrisation)")
    if rule not in AW6.RULES:
        raise SystemExit(f"{name}: unknown rule {rule}")
    sl = [int(s) for s in seeds.split("+")] if seeds else [0]
    if len(set(sl)) != len(sl):
        raise SystemExit(f"{name}: repeated seed")
    return {"train": looks, "base": base, "tok": tok, "K": K, "max_len": max_len, "model": model, "family": fam, "op": op, "cos": cos,
            "rule": rule, "seeds": sl}


def make_for(sp, nt, phi):
    """gen: GenMix; free and text: anchor_walk7.make_for, unchanged (what anchor_walk8 fits)."""
    if sp["family"] == "gen":
        return lambda: GenMix(nt, sp["K"], phi, sp["op"], cos=sp["cos"])
    return AW7.make_for(sp["model"], nt, sp["K"], phi)


def params_used(model, sp):
    """The parameters a model's forward reads (Mix's unused tables left out)."""
    if isinstance(model, GenMix):
        return model.params_used()
    n = sum(p.numel() for p in model.parameters())
    d = model.t1.shape[1]
    if sp["family"] == "text":
        n -= model.t2.numel() - d                 # one edge: t2 row 0 only
        if sp["op"] != "txt":
            n -= model.t1.numel() - 5 * d          # t1 only for the five non-phrase tokens
    elif sp["op"] != "lin":
        n -= model.bk.numel() + model.t2.numel()  # OpMix: bk and t2 unused
    elif sp["max_len"] == 1:
        n -= model.t2.numel() - d
    return int(n)


def set_phi(model, phi, K):
    """Point a text or gen model's phrase table at another dataset's (the parameters are untouched)."""
    if isinstance(model, GenMix):
        model.set_phi(phi)
    elif isinstance(model, AW7.TxtMix):
        Phi, _pm = phi_layout(model.nt, K, phi)
        model.Phi = Phi


def fold_of(s, F):
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big") % F


def identity_map(K):
    return np.minimum(np.arange(R_TOP + 1), K).astype(np.int64)


def hold_map(K, vocab_w2, h, F):
    """Ranks below K whose phrase string falls in fold h go to 'other' (K)."""
    rm = identity_map(K)
    held = np.zeros(R_TOP + 1, dtype=bool)
    for r in range(min(K, len(vocab_w2))):
        if fold_of(vocab_w2[r], F) == h:
            held[r] = True
    rm[held] = K
    return rm, held


def string_map(K, vocab_from, vocab_to):
    """A rank of the reading dataset (vocab_from) -> the fit's rank of the same phrase string if it is in the fit's top K."""
    idx = {s: r for r, s in enumerate(vocab_to[:K])}
    rm = np.full(R_TOP + 1, K, dtype=np.int64)
    for r, s in enumerate(vocab_from[:R_TOP]):
        rm[r] = idx.get(s, K)
    return rm


def edge_ranks(q):
    """Each structural pool edge's raw phrase rank in its walk direction (anchor_walk.anchor_tokens' choice)."""
    m = q["fam"] == 0
    d = A16.famdir(q)[m]
    a = np.where(d == 1, q["w2_b"], q["w2_f"])
    if (a < 0).any():
        raise SystemExit("a structural pool edge has no stored anchor in its direction")
    return m, d, a


def tokens_mapped(Q, K, rmap, rows=None):
    """anchor_walk.anchor_tokens with each raw rank sent through rmap (identity_map(K) gives anchor_tokens exactly)."""
    nt = 3 * (K + 1) + 2
    toks = [None] * len(Q)
    for i in (range(len(Q)) if rows is None else rows):
        q = Q[i]
        t = np.where(q["fam"] == 1, 3 * (K + 1), 3 * (K + 1) + 1).astype(np.int64)
        m, d, a = edge_ranks(q) if (q["fam"] == 0).any() else (q["fam"] == 0, None, None)
        if m.any():
            t[m] = d * (K + 1) + rmap[np.minimum(a, R_TOP)]
        toks[i] = t
    return nt, toks


def types_for(Q, sp, rmap, rows=None):
    rr = range(len(Q)) if rows is None else rows
    if sp["tok"] == "T0":
        nt = 5
        TY = [None] * len(Q)
        for i in rr:
            TY[i] = A16.walk_types(Q[i], A16.famdir(Q[i]), nt, sp["max_len"])
        return nt, TY
    nt, tk = tokens_mapped(Q, sp["K"], rmap, rows)
    TY = [None] * len(Q)
    for i in rr:
        TY[i] = A16.walk_types(Q[i], tk[i], nt, sp["max_len"])
    return nt, TY


def held_reach_rows(Q, rows, held, K):
    """The rows with a gold one structural edge from a seed through an edge whose phrase is held out."""
    out = []
    for i in rows:
        q = Q[i]
        m = q["fam"] == 0
        if not m.any():
            continue
        _m, _d, a = edge_ranks(q)
        hm = held[np.minimum(a, R_TOP)] & (a < K)
        if not hm.any():
            continue
        u, v = q["u"][m][hm], q["v"][m][hm]
        S = set(q["seeds"][q["seeds"] >= 0].tolist())
        g = set(np.flatnonzero(q["gold"]).tolist())
        if any(int(x) in S and int(y) in g for x, y in zip(u, v)):
            out.append(i)
    return out


def load_looks(names, max_len, t0):
    """The looks through anchor_walk11's pruned loader (no projections: no gate model is read here)."""
    AW11.STATE.update({"max_len": max_len, "proj": False, "rows": 0, "edges_in": 0, "edges_kept": 0, "checked_rows": 0})
    Q, part, rec_sha = [], {}, {}
    for lk in names:
        recs = sorted((AW3.LOOKS / lk).glob("record*.json"))
        if not recs:
            raise SystemExit(f"look {lk} has no record")
        for r in recs:
            rs = json.loads(r.read_text(encoding="utf-8"))["script_sha256"]
            if rs != AW3.LOOK_SCORE_SHA:
                raise SystemExit(f"{r} was not written by the pinned look scorer")
            rec_sha[str(r.relative_to(AW3.LOOKS))] = rs
        _ids, Ql = AW11.load_pruned(AW3.LOOKS / lk)
        part[lk] = list(range(len(Q), len(Q) + len(Ql)))
        Q.extend(Ql)
        log(f"look {lk}: {len(Ql)} rows, {time.time() - t0:.0f}s")
    return Q, part, rec_sha, dict(AW11.STATE)


def phrase_table(ds):
    pth, want = AW7.PHRASE[ds]
    got = AW.sha(pth)
    if got != want:
        raise SystemExit(f"{pth} is not the pinned phrase table")
    phi = np.load(pth).astype(np.float32)
    if phi.shape != (AW7.PHRASE_K, 1536):
        raise SystemExit("unexpected phrase table shape")
    return phi, got


def check_pins():
    for mod, key in ((AW11, "anchor_walk11"), (AW8, "anchor_walk8"), (AW7, "anchor_walk7"), (AW6, "anchor_walk6")):
        if AW.sha(Path(mod.__file__)) != PINS[key]:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    if AW.sha(Path(AW10.__file__)) != AW11.AW10_SHA:
        raise SystemExit("anchor_walk10.py is not the pinned file")
    for mod, want in ((AW3, AW6.PINS["anchor_walk3"]), (AW6.AW4, AW6.PINS["anchor_walk4"]), (AW5, AW6.PINS["anchor_walk5"]),
                      (AW, AW3.AW_SHA), (G2, AW3.G2_SHA), (G2.LG, AW3.G1_SHA), (A16, AW.A16_SHA)):
        if AW.sha(Path(mod.__file__)) != want:
            raise SystemExit(f"{mod.__file__} is not the pinned file")


class Reader:
    """rho, the bootstrap against the GNN and the twin, and by-type means on a fixed set of rows."""

    def __init__(self, Q, rows, seed):
        M = np.stack([Q[i]["metrics"] for i in rows])[:, :, A16.RI]
        self.rows = list(rows)
        self.t, self.g = M[:, 0], M[:, 3]
        self.gain = self.g.mean(0) - self.t.mean(0)
        self.W = np.random.default_rng(seed).poisson(1.0, (1000, len(rows))).astype(np.float64)
        ty = np.asarray([str(Q[i]["type"]) for i in rows])
        self.ty = ty
        self.types = sorted(set(ty.tolist())) if len(set(ty.tolist())) <= 12 else []

    def rho(self, m):
        return [round(float((m[:, j].mean() - self.t[:, j].mean()) / self.gain[j]), 3) if abs(self.gain[j]) > 1e-9 else None for j in range(3)]

    def record(self, m, by_type=True):
        r = {"fit": m.mean(0).round(4).tolist(), "rho (R@5, FC@5, hit@1)": self.rho(m), "minus_gnn0": AW3.boot_pair(m - self.g, self.W),
             "minus_twin0": AW3.boot_pair(m - self.t, self.W)}
        if by_type and self.types:
            r["by_type"] = {t: {"rows": int((self.ty == t).sum()), "fit": m[self.ty == t].mean(0).round(4).tolist(),
                                "twin0": self.t[self.ty == t].mean(0).round(4).tolist(), "gnn0": self.g[self.ty == t].mean(0).round(4).tolist()}
                            for t in self.types}
        return r

    def base(self):
        return {"rows": len(self.rows), "twin0": self.t.mean(0).round(4).tolist(), "gnn0": self.g.mean(0).round(4).tolist(),
                "gain": self.gain.round(4).tolist()}

    def subset(self, m, keep_rows):
        """rho and the raw deltas on a subset of these rows (m aligned to self.rows)."""
        pos = {i: j for j, i in enumerate(self.rows)}
        ix = np.asarray([pos[i] for i in keep_rows], dtype=np.int64)
        if ix.size == 0:
            return {"rows": 0}
        t, g, mm = self.t[ix], self.g[ix], m[ix]
        gain = g.mean(0) - t.mean(0)
        return {"rows": int(ix.size), "fit": mm.mean(0).round(4).tolist(), "twin0": t.mean(0).round(4).tolist(), "gnn0": g.mean(0).round(4).tolist(),
                "rho (R@5, FC@5, hit@1)": [round(float((mm[:, j].mean() - t[:, j].mean()) / gain[j]), 3) if abs(gain[j]) > 1e-9 else None for j in range(3)]}


def save_model(path, model, sp, nt, margin, ds, vocab_w2, phi_sha, extra):
    sd = {k: v for k, v in model.state_dict().items() if k not in ("Phi", "pmask", "fdir")}
    torch.save({"spec": sp, "nt": nt, "margin": margin, "dataset": ds, "vocab_w2": list(vocab_w2[:sp["K"]]) if sp["K"] else None,
                "phi_sha": phi_sha, "state_dict": sd, **extra}, path)


def load_model(path, phi):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sp, nt = ck["spec"], ck["nt"]
    model = make_for(sp, nt, phi)()
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    if unexpected or any(k not in ("Phi", "pmask", "fdir") for k in missing):
        raise SystemExit(f"{path}: state dict does not match ({missing}, {unexpected})")
    model.eval()
    return model, ck


def read(model, Q, TY, nt, rows, z_of, margin):
    return AW6.read_rule(model, Q, TY, nt, list(rows), z_of, margin)


def main_fit(a):
    AW6.rebind(a.dataset)
    out_path = Path(a.out) if a.out else HERE / f"anchor_gen_{a.dataset}.json"
    torch.set_num_threads(2)
    t0 = time.time()
    train_looks = AW6.TRAIN[a.dataset] if a.dataset != "2wiki" else ("fit", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12")
    specs = {name: parse(name, train_looks) for name in a.variants.split(",")}
    hold = None
    if a.hold:
        h, F = (int(x) for x in a.hold.split("/"))
        if not 0 <= h < F or F < 2:
            raise SystemExit("--hold h/F with 0 <= h < F, F >= 2")
        hold = (h, F)
        if any(sp["tok"] == "T0" for sp in specs.values()):
            raise SystemExit("--hold reads phrase bases only")
    phi, phi_sha = (None, None)
    if any(sp["family"] != "free" for sp in specs.values()):
        phi, phi_sha = phrase_table(a.dataset)
    looks = ["x1", "select"] + [lk for lk in train_looks if any(lk in sp["train"] for sp in specs.values())]
    Q, part, rec_sha, load_state = load_looks(looks, max(sp["max_len"] for sp in specs.values()), t0)
    checks, vocab = AW.anchor_tables(Q)
    vocab_w2 = vocab.get("w2", [])
    if any(sp["tok"] == "A" for sp in specs.values()) and len(vocab_w2) < max(sp["K"] or 0 for sp in specs.values()):
        raise SystemExit("the compact's phrase strings do not cover K")
    log(f"anchors attached to {len(Q)} rows: {checks}, {time.time() - t0:.0f}s")
    x1 = part["x1"]
    B_rows = x1[1::2]
    RB, RX = Reader(Q, B_rows, 20261002), Reader(Q, x1, 20261003)
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    zT_of = [None] * len(Q)
    if a.kd > 0:
        for i in sorted({i for sp in specs.values() for lk in sp["train"] for i in part[lk]}):
            zT_of[i] = AW8.teacher_of(Q[i], a.teacher)
    res = {"look": "anchor_gen", "mode": "fit", "dataset": a.dataset, "script_sha256": AW.sha(Path(__file__)),
           "pins": {**PINS, "anchor_walk10": AW11.AW10_SHA, **AW6.PINS, "compact": AW.COMPACT_SHA, "struct": AW.STRUCT_SHA, "phrase_table": phi_sha,
                    "look_score_records": rec_sha}, "flag_checks": checks, "pruned_loader": load_state,
           "looks": {lk: len(r) for lk, r in part.items()}, "epochs": a.epochs, "lr": a.lr, "wd": a.wd,
           "kd": {"ce": a.ce, "kd": a.kd, "T": a.T, "teacher": a.teacher} if a.kd > 0 else None, "hold": a.hold,
           "B": RB.base(), "x1": RX.base(), "variants": {}}
    per_row = {}
    model_dir = out_path.parent / (out_path.stem + "_models")
    model_dir.mkdir(parents=True, exist_ok=True)

    def save():
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, B_rows=np.asarray(B_rows))

    cache = {}
    for name, sp in specs.items():
        t1 = time.time()
        K = sp["K"]
        if sp["tok"] == "A":
            rm_train, held = hold_map(K, vocab_w2, *hold) if hold else (identity_map(K), None)
        else:
            rm_train, held = None, None
        key = (sp["tok"], K, sp["max_len"])
        if key not in cache:
            cache.clear()
            cache[key] = types_for(Q, sp, rm_train)
        nt, TY = cache[key]
        alt = {}
        if sp["tok"] == "A":
            alt["NR"] = types_for(Q, sp, np.full(R_TOP + 1, K, dtype=np.int64), B_rows)[1]
            if hold and sp["family"] != "free":    # a free model has no vector for a phrase it never saw: MASK is its read
                alt["REV"] = types_for(Q, sp, identity_map(K), B_rows)[1]
        tr = [i for lk in sp["train"] for i in part[lk]]
        make = make_for(sp, nt, phi)
        hrows = held_reach_rows(Q, B_rows, held, K) if hold else None
        models, seeds_out = [], {}
        for sd in sp["seeds"]:
            t2 = time.time()
            model, reads, best_ep, curve, margin, by_margin = AW8.fit_read_kd(Q, TY, nt, tr, part["select"], {"B": B_rows, "x1": x1}, z_of, zT_of,
                                                                              make, a.epochs, sd, sp["rule"], a.lr, a.wd, a.ce, a.kd, a.T)
            models.append(model)
            tag = f"{name}#{sd}"
            rd = {"ID" if not hold else "MASK": RB.record(reads["B"]), "ID_x1" if not hold else "MASK_x1": RX.record(reads["x1"], by_type=False)}
            per_row[f"{len(res['variants'])}_{sd}_{'ID' if not hold else 'MASK'}"] = reads["B"]
            for nm, TYa in alt.items():
                m_alt = read(model, Q, TYa, nt, B_rows, z_of, margin)
                rd[nm] = RB.record(m_alt)
                per_row[f"{len(res['variants'])}_{sd}_{nm}"] = m_alt
                rd[f"{nm} - {'ID' if not hold else 'MASK'}"] = AW3.boot_pair(m_alt - reads["B"], RB.W)
            if hold:
                rd["held_reach_rows"] = {"MASK": RB.subset(reads["B"], hrows)}
                if "REV" in alt:
                    rd["held_reach_rows"]["REV"] = RB.subset(per_row[f"{len(res['variants'])}_{sd}_REV"], hrows)
            pt = model_dir / f"v{len(res['variants'])}_s{sd}.pt"
            save_model(pt, model, sp, nt, margin, a.dataset, vocab_w2, phi_sha, {"hold": a.hold, "name": name, "seed": sd})
            fam = sp["family"]
            seeds_out[str(sd)] = {"best_epoch": best_ep, "curve": curve, "margin": None if margin is None else str(margin), "by_margin_select": by_margin,
                                  "kappa": float(torch.exp(model.log_kappa).item()), "beta": float(torch.nn.functional.softplus(model.beta_raw).item()),
                                  "gam": model.gam.detach().round(decimals=4).tolist() if hasattr(model, "gam") else None,
                                  "model_file": str(pt.relative_to(ROOT)) if pt.is_relative_to(ROOT) else str(pt), "reads": rd,
                                  "seconds": round(time.time() - t2, 1)}
            log(f"{tag} ({fam}): ep {best_ep}, " + ", ".join(f"{k} {v['rho (R@5, FC@5, hit@1)']}" for k, v in rd.items() if isinstance(v, dict) and "rho (R@5, FC@5, hit@1)" in v))
        m0 = models[0]
        v = {**sp, "train_rows": len(tr), "tokens": nt, "types_per_row": round(float(np.mean([len(TY[i]) for i in tr])), 1),
             "params": int(sum(p.numel() for p in m0.parameters())), "params_used": params_used(m0, sp),
             "seeds_read": seeds_out}
        if hold:
            Hn = int(held[:K].sum())
            ne = nh = 0
            for i in B_rows:
                m, _d, a_ = edge_ranks(Q[i]) if (Q[i]["fam"] == 0).any() else (None, None, np.zeros(0, np.int64))
                ne += int(a_.size)
                nh += int((held[np.minimum(a_, R_TOP)] & (a_ < K)).sum())
            v["held"] = {"phrases": Hn, "of_K": K, "B_struct_edges": ne, "B_held_edges": nh, "held_reach_rows": len(hrows)}
        if len(models) > 1:
            mg = 0.0 if sp["rule"] == "p" else (None if sp["rule"] == "np" else None)
            if sp["rule"] == "mp":
                best_v = -1.0
                for m_ in AW6.MARGINS:
                    val = float(AW7.read_ens(models, Q, TY, nt, list(part["select"]), z_of, m_).mean())
                    if val > best_v + 1e-12:
                        mg, best_v = m_, val
            ens = {"margin": None if mg is None else str(mg)}
            mB = AW7.read_ens(models, Q, TY, nt, list(B_rows), z_of, mg)
            ens["ID" if not hold else "MASK"] = RB.record(mB)
            for nm, TYa in alt.items():
                ens[nm] = RB.record(AW7.read_ens(models, Q, TYa, nt, list(B_rows), z_of, mg))
            v["ensemble"] = ens
        v["seconds"] = round(time.time() - t1, 1)
        res["variants"][name] = v
        save()
    res["seconds"] = round(time.time() - t0, 1)
    save()
    log(f"done in {res['seconds']}s")


def main_xread(a):
    """Read saved models on another dataset's carve x1: zero-shot transfer to its graph, phrases, twin and GNN."""
    if a.target not in DATASETS:
        raise SystemExit(f"--target among {DATASETS}")
    AW6.rebind(a.target)
    out_path = Path(a.out)
    torch.set_num_threads(2)
    t0 = time.time()
    paths = [Path(p) for p in a.models.split(",")]
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    max_len = max(ck["spec"]["max_len"] for ck in cks)
    Q, part, rec_sha, load_state = load_looks(["x1"], max_len, t0)
    checks, vocab = AW.anchor_tables(Q)
    vocab_w2 = vocab.get("w2", [])
    x1 = part["x1"]
    B_rows = x1[1::2]
    RB, RX = Reader(Q, B_rows, 20261002), Reader(Q, x1, 20261003)
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    need_phi = any(ck["spec"]["family"] != "free" for ck in cks)
    phi, phi_sha = phrase_table(a.target) if need_phi else (None, None)
    res = {"look": "anchor_gen", "mode": "xread", "target": a.target, "script_sha256": AW.sha(Path(__file__)),
           "pins": {**PINS, "compact": AW.COMPACT_SHA, "struct": AW.STRUCT_SHA, "phrase_table": phi_sha, "look_score_records": rec_sha},
           "flag_checks": checks, "pruned_loader": load_state, "B": RB.base(), "x1": RX.base(), "models": {}}
    per_row = {}
    for j, (p, ck) in enumerate(zip(paths, cks)):
        t1 = time.time()
        sp, K = ck["spec"], ck["spec"]["K"]
        if ck["dataset"] == a.target and not ck.get("hold"):
            log(f"{p.name}: fitted on {a.target} itself (an in-domain read)")
        model, _ck = load_model(p, phi if sp["family"] != "free" else None)
        if sp["family"] != "free":
            set_phi(model, phi, K)
        ent = {"from": ck["dataset"], "name": ck.get("name"), "seed": ck.get("seed"), "hold": ck.get("hold"), "spec": sp,
               "margin": ck["margin"], "model_file": str(p)}
        if sp["tok"] == "A":
            rm = identity_map(K) if sp["family"] != "free" else string_map(K, vocab_w2, ck["vocab_w2"])
            if sp["family"] == "free":
                ne = nm_ = 0
                for i in B_rows:
                    if (Q[i]["fam"] == 0).any():
                        _m, _d, a_ = edge_ranks(Q[i])
                        ne += int(a_.size)
                        nm_ += int((rm[np.minimum(a_, R_TOP)] < K).sum())
                ent["string_matched_edge_share_B"] = round(nm_ / max(ne, 1), 4)
            nt, TY = types_for(Q, sp, rm, x1)
            _nt, TYn = types_for(Q, sp, np.full(R_TOP + 1, K, dtype=np.int64), B_rows)
        else:
            nt, TY = types_for(Q, sp, None, x1)
            TYn = None
        mB = read(model, Q, TY, nt, B_rows, z_of, ck["margin"])
        mX = read(model, Q, TY, nt, x1, z_of, ck["margin"])
        per_row[f"{j}_XD"] = mB
        ent["XD"], ent["XD_x1"] = RB.record(mB), RX.record(mX, by_type=False)
        if TYn is not None:
            mN = read(model, Q, TYn, nt, B_rows, z_of, ck["margin"])
            per_row[f"{j}_XDNR"] = mN
            ent["XD_NR"] = RB.record(mN)
            ent["XD_NR - XD"] = AW3.boot_pair(mN - mB, RB.W)
        ent["seconds"] = round(time.time() - t1, 1)
        res["models"][p.name if p.name not in res["models"] else f"{p.parent.name}/{p.name}"] = ent
        log(f"{p.name} ({ck['dataset']} -> {a.target}, {sp['model']}): XD {ent['XD']['rho (R@5, FC@5, hit@1)']}"
            + (f", XD_NR {ent['XD_NR']['rho (R@5, FC@5, hit@1)']}" if "XD_NR" in ent else ""))
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, B_rows=np.asarray(B_rows))
    res["seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done in {res['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--variants", default="x4+x5+x6:A1024-1/glin")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--ce", type=float, default=1.0)
    ap.add_argument("--kd", type=float, default=0.0)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--teacher", default="gm", choices=AW8.TEACHERS)
    ap.add_argument("--hold", default=None)
    ap.add_argument("--xread", action="store_true")
    ap.add_argument("--target", default=None)
    ap.add_argument("--models", default=None)
    a = ap.parse_args(argv)
    if a.ce < 0 or a.kd < 0 or a.ce + a.kd <= 0 or a.T <= 0:
        raise SystemExit("ce and kd must be non-negative, not both 0, and T positive")
    check_pins()
    if a.xread:
        if not (a.target and a.models and a.out):
            raise SystemExit("--xread needs --target, --models and --out")
        main_xread(a)
    else:
        if a.dataset not in DATASETS:
            raise SystemExit(f"--dataset among {DATASETS}")
        main_fit(a)


if __name__ == "__main__":
    main()
