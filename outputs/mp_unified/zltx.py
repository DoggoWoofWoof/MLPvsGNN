"""LTX's stage (docs/LTX_CROSSFIT_NEXT_HOP_GNN.md): LTG's next-hop term (docs/LTG_NEXT_HOP_GNN.md) cross-fitted inside the
GNN of record. The term holds two of LTW's bil scorers on the store's 256 axes, one per fold. A question's fold is its
rows' one NEXT column (ltxcols.py: on a training carve the components of the questions joined by a shared gold node, so no
gold node is in both folds; on any other carve a draw). Every layer's edge score reads, for each question, the scorer of
the fold the question is not in, detached; the next-hop loss (LTG's, weight LAMBDA) trains the scorer of the question's own
fold. So no scorer a question's message passing reads has seen that question's golds, in training as at the read, and the
main loss trains no scorer: it trains the sender maps W_k, the gains and the summaries, all zero-started as LTG's, so a fit
starts as the GNN of record bit for bit. On PKO's roots, on the host's GPU: the screen at seed 0 on both WQ splits against
the base's seed-0 fits under LTG's in-domain rule (docs/LTG_NEXT_HOP_GNN.md section 4), DEP's sixteen-read rule beside.

zgnx.py, zltg.py, ltxcols.py and every module they import are imported and not edited (ltxcols only inside the checks and
the self-test, so a fit's threads are GNX's). For the length of an LTX call zgnx's names (its fits, declaration, stage,
stamp key, output root, fit and pair names, code, term record, NEXT block, model patch, arm import and stamp) are LTX's, its
block width is one column, and ltmcols' root, stage, declaration, column names and width are LTX's: zgnx's carve then loads
LTX's folds through zltm.load_next and refuses every other stage's columns. For the length of an LTX model zltg's layer
names are LTX's, so zltg's own layers run on LTX's parameters. Everything else in a fit is GNX's and LTG's: the carve, the
base, the start, the trainer, the read, the layers and the next-hop loss. Every patched name is put back afterwards, also on
a failure.

    python outputs/mp_unified/zltx.py check-cols                   (LTX's folds of every carve both splits load)
    python outputs/mp_unified/zltx.py fit --name scr-ltx-xf-s0     (train when needed, then read; kept when read before)
    python outputs/mp_unified/zltx.py compare-all                  (each LTX fit read against the base's fit of its seed)
    python outputs/mp_unified/zltx.py calls                        (outputs/ltx/calls.json and .md)
    python outputs/mp_unified/zltx.py s0stop-check                 (s0stop's check under LTX's entry)
    python outputs/mp_unified/zltx.py s0stop-run -- COMMAND ...    (s0stop's run under LTX's entry)
    python outputs/mp_unified/zltx.py gate --want ADOPT_FOR_FULL_RUN (s0stop's gate under LTX's entry)
    python outputs/mp_unified/zltx.py --selftest
"""
import argparse
import contextlib
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import zgnx as GNX  # noqa: E402  (GNX's arm, carve, stamp, compares and the pieces of its calls; not edited)
import s0stop as S0  # noqa: E402  (the seed-0 stop, with LTX's entry patched in for a call; not edited)

M1, RULE, ISO, DEP, LTG, LTM, LTMC = GNX.M1, GNX.RULE, GNX.ISO, GNX.DEP, GNX.LTG, GNX.LTM, GNX.LTMC
DECLARED = "docs/LTX_CROSSFIT_NEXT_HOP_GNN.md"
STAGE = "LTX"
KEY = "ltx"                                     # the stamp's key in an LTX fit's screen.json and read.json and LTX's compares
OUT = "outputs/ltx"
COLS_OUT = ROOT / OUT                           # ltxcols' root (its cols/ tree), checked against ltxcols in the self-test
TAG = "xf"
NEXT = GNX.NEXT
COLS = ("fold",)
WIDTH = len(COLS)                               # one column: the question's fold, 0.0 or 1.0 on every row
IN = WIDTH
FOLDS = 2                                       # one scorer per fold
RANK, D, FAMS, LAYERS, AXES = LTG.RANK, LTG.D, LTG.FAMS, LTG.LAYERS, LTG.AXES
TERM_LAYER = tuple(tuple(f"ltx_{k}_{p}" for p in ("w", "g", "em", "ea")) for k in range(LAYERS))
TERM_NAMES = ("ltx_V", "ltx_U") + tuple(n for lay in TERM_LAYER for n in lay)
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, ADOPT, INCOMPLETE = (
    GNX.ADOPT_FOR_FULL_RUN, GNX.NO_GAIN, GNX.NOT_ADOPTED, GNX.ADOPT, GNX.INCOMPLETE)
PATCHED = ("FITS", "DECLARED", "STAGE", "KEY", "OUT", "fit_name", "pair_name", "code_extra", "term_record", "next_block",
           "IN", "WIDTH", "on_model", "import_g", "stamp_with_next")
COLS_PATCHED = ("OUT", "STAGE", "DECLARED", "COLS", "WIDTH")
ORIG = {k: getattr(GNX, k) for k in ("code_extra", "next_block", "import_g", "stamp_with_next")}
GNX_FITS = dict(GNX.FITS)                       # zgnx's own fit names, taken before any patch
BESIDE = ("ltg", "gnx", "gnw")


def n_term(axes):
    return 2 * FOLDS * 3 * axes * RANK + LAYERS * (3 * RANK * D + FAMS + 2 * FAMS * D)


N_TERM = n_term(AXES)
assert (WIDTH, IN, FOLDS, NEXT, N_TERM, LTG.N_TERM) == (1, 1, 2, "NEXT", 117897, 68745)


def fit_name(seed, split):
    return f"scr-ltx-{TAG}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    return f"scr-ltx-{TAG}-vs-base-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (GNX.ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 6 and not set(FITS) & (set(M1.ALL_FITS) | set(ISO.FITS) | set(DEP.FITS) | set(LTG.FITS)
                                            | set(GNX.FITS))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zltx: {msg}", flush=True)


def own_shas():
    return {"zltx_sha256": M1.WQ.sha_src(__file__), "ltxcols_sha256": M1.WQ.sha_src(str(HERE / "ltxcols.py"))}


def code_extra():
    """GNX's code beside zml1s's (zgnx, zdep, zltg, zltm, ltmcols), and LTX's own: this file and ltxcols.py."""
    return {**ORIG["code_extra"](), **own_shas()}


def term_record(G):
    A = int(LTG.BASIS["store"].V.shape[1])
    return {"term": "LTX's cross-fitted next-hop term: two of LTW's bil scorers on the store's axes, one per fold; a "
                    "question's layers read the scorer of the fold it is not in, detached, and the next-hop loss trains the "
                    "scorer of its own fold",
            "columns": list(COLS), "inputs": IN, "folds": FOLDS, "rank": RANK, "axes": A, "basis": LTG.BASIS_NAME,
            "basis_npz_sha256": (LTG.BASIS["rec"] or {}).get("npz_sha256"), "lambda": LTG.LAMBDA, "clip": LTG.CLIP,
            "term_parameters": n_term(A), "parameters_h128": int(G.N_PARAMS + n_term(A)), "layers": int(G.LAYERS),
            "seed_offset": LTG.SEED_OFF}


def term_record_keys():
    return ("term_parameters", "parameters_h128", "inputs", "columns", "layers", "folds", "rank", "axes", "lambda")


def next_block(split):
    """GNX's NEXT block with LTX's declaration and the columns' root (inside on_ltx: LTX's)."""
    return {**ORIG["next_block"](split), "declared_in": DECLARED, "columns_root": M1.WQ.rel(Path(LTMC.OUT))}


def import_g():
    """zgnx's import_g (zg6r's arm with its three layers), LTG's widths checked, the store's basis loaded (zltg.load_basis:
    basis_2wiki, pinned to LTW A1's sha256) and the next-hop curve reset."""
    G = ORIG["import_g"]()
    if (G.LAYERS, G.D, G.FAMS) != (LAYERS, D, FAMS):
        raise SystemExit(f"zltx: zg6r is {(G.LAYERS, G.D, G.FAMS)}, not LTG's {(LAYERS, D, FAMS)}")
    LTG.load_basis(G)
    LTG.aux_reset()
    return G


def stamp_with_next(orig, call, rec):
    """zgnx's stamp (the term's record, the base, ISO's verdict and the NEXT block, refusing a train or read without every
    carve's columns); an LTX train's stamp also holds the next-hop curve."""
    inner = ORIG["stamp_with_next"](orig, call, rec)

    def stamp(name, env, what, extra=None):
        if name in FITS and what == "screen":
            extra = {**(extra or {}), "aux": LTG.aux_summary()}
        return inner(name, env, what, extra)
    return stamp


# ── the term ─────────────────────────────────────────────────────────────────

CUR = {}


def add_term(G, m, seed):
    """The term's parameters on m: V of both scorers from LTG's own generator (scorer 0's draw is LTG's), U and every
    layer's W_k, gains and summaries 0; the basis as buffers under LTG's names (not saved)."""
    torch, nn = G.torch, G.nn
    st = LTG.BASIS["store"]
    if st is None:
        raise RuntimeError("zltx: no basis loaded before a ZG6R was built (zltg.load_basis)")
    A = int(st.V.shape[1])
    g = torch.Generator().manual_seed(int(seed) + LTG.SEED_OFF)
    m.ltx_V = nn.Parameter(torch.stack([torch.randn(3, A, RANK, generator=g) for _ in range(FOLDS)]))
    m.ltx_U = nn.Parameter(torch.zeros(FOLDS, 3, A, RANK))
    for wn, gn, en, an in TERM_LAYER:
        setattr(m, wn, nn.Parameter(torch.zeros(3 * RANK, D)))
        setattr(m, gn, nn.Parameter(torch.zeros(FAMS)))
        setattr(m, en, nn.Parameter(torch.zeros(FAMS, D)))
        setattr(m, an, nn.Parameter(torch.zeros(FAMS, D)))
    m.register_buffer("ltg_bV", torch.from_numpy(np.ascontiguousarray(st.V, np.float32)), persistent=False)
    m.register_buffer("ltg_bm", torch.from_numpy(np.ascontiguousarray(st.m, np.float32)), persistent=False)
    m.register_buffer("ltg_bc", torch.from_numpy(np.ascontiguousarray(st.c, np.float32)), persistent=False)
    m.ltg_kappa = float(st.kappa)


def fold_of(torch, x, nq, B):
    """(B,) int64: each question's fold from its rows' fold column. A value other than 0 and 1, or a question whose rows
    hold both folds, stops (one sync); a question without rows is in fold 0 (it has no edge)."""
    nq = nq.to(torch.int64)
    f = x[:, 0].to(torch.float64)
    s = torch.zeros(B, dtype=torch.float64, device=f.device).index_add_(0, nq, f)
    n = torch.zeros(B, dtype=torch.float64, device=f.device).index_add_(0, nq, torch.ones_like(f))
    if bool(((f != 0) & (f != 1)).any() | ((s != 0) & (s != n)).any()):
        raise RuntimeError("zltx: the batch's fold column holds a value other than 0 and 1, or a question with both folds")
    return (s > 0).to(torch.int64)


def cross_edges(torch, nq, u, v):
    """Stops when an edge joins two questions' rows (counted on the device chunk by chunk, one sync)."""
    n = torch.zeros((), dtype=torch.int64, device=u.device)
    for a in range(0, u.numel(), LTG.CHUNK):
        b = a + LTG.CHUNK
        n += (nq[u[a:b]] != nq[v[a:b]]).sum()
    if int(n):
        raise RuntimeError(f"zltx: {int(n)} edges of the batch join two questions' rows")


def scorer_pq(torch, V, U, qk, x, c, nr, nq):
    """(P, Q) of one scorer, zltg.term_inputs' op for op: P (N, 3 RANK) each row as a receiver, [x U0, x U1, x U2]; Q
    (N, 3 RANK) each row as a sender, [qk V0, x V1, (qk - c x) V2 / nr]."""
    xm = x @ torch.cat([V[1], V[2], U[0], U[1], U[2]], 1)
    Q = torch.cat([(qk @ V[0])[nq], xm[:, :RANK],
                   ((qk @ V[2])[nq] - c.unsqueeze(1) * xm[:, RANK:2 * RANK]) / nr.unsqueeze(1)], 1)
    P = xm[:, 2 * RANK:].contiguous()
    return P, Q


def term_inputs(G, m, semb, keep, nq, u, v, N, fq):
    """(P, Q, lnu, kp, Po, Qo): P and Q for every row from the scorer of the fold its question is not in, detached (the
    layers' edge scores); lnu and kp as zltg.term_inputs'; Po and Qo from the scorer of the row's own fold, with their
    gradients (the next-hop loss's edge score), in training with gradients only, else None. An edge between two questions'
    rows stops."""
    torch = G.torch
    cross_edges(torch, nq, u, v)
    qk, x, c, nr = LTG.store_terms(G, m, semb, nq)
    own = m.training and torch.is_grad_enabled()
    with contextlib.nullcontext() if own else torch.no_grad():
        P0, Q0 = scorer_pq(torch, m.ltx_V[0], m.ltx_U[0], qk, x, c, nr, nq)
        P1, Q1 = scorer_pq(torch, m.ltx_V[1], m.ltx_U[1], qk, x, c, nr, nq)
    rf = fq[nq].view(-1, 1).bool()                      # the row's question is in fold 1: its layers read scorer 0
    P, Q = torch.where(rf, P0, P1).detach(), torch.where(rf, Q0, Q1).detach()
    Po = Qo = None
    if own:
        Po, Qo = torch.where(rf, P1, P0), torch.where(rf, Q1, Q0)
    with torch.no_grad():
        deg = torch.zeros(N, dtype=x.dtype, device=x.device).index_add_(
            0, u, torch.ones(u.numel(), dtype=x.dtype, device=x.device))
        lnu = deg.clamp_min(1.0).log()[u]
        kp = keep[nq[u], m.j_semb].to(x.dtype)
    return P, Q, lnu, kp, Po, Qo


def term_stack(G, m, H, xb, lk, gd, nq, B, u01, semb, keep, fq):
    """zltg.term_stack, op for op, with LTX's inputs: the layers (zltg's) read the other fold's scorer; in training (with
    gradients) the scores, the own fold's plain edge score Po[v] . Qo[u] and the kept edges are left for the next-hop
    loss."""
    torch, Fn = G.torch, G.Fn
    xb = xb.to(H.dtype)
    h0 = Fn.gelu(Fn.linear(torch.cat([H, xb], 1), m.g6_in_w, m.g6_in_b))
    s0 = Fn.linear(h0, m.g6_ctx_w).squeeze(-1)
    z0 = G.LG.seg_zscore8D(s0.unsqueeze(1), nq, B).squeeze(1).clamp(-G.ZCLIP, G.ZCLIP)
    w = G.G2.seg_softmax_rows(z0, nq, B).unsqueeze(1)
    u, v, f, ci = lk["u"], lk["v"], lk["f"], gd["ci"]
    if u01 is not None:
        ke = G.G2.family_keep(f, nq[v], B, u01)
        u, v, f, ci = u[ke], v[ke], f[ke], ci[ke]
    te = torch.nonzero(ci >= 0).view(-1)
    codes = cte = None
    if te.numel():
        if gd["rel"] is None:
            raise RuntimeError("zg6r: typed edges in the batch without their relation vectors")
        codes = G.rel_codes(m, gd["rel"], gd["crel"], gd["cdir"], H.dtype)
        cte = ci[te]
    N = h0.shape[0]
    P, Q, lnu, kp, Po, Qo = term_inputs(G, m, semb, keep, nq, u, v, N, fq)
    h = LTG.term_layers(G, m, h0, xb, w, u, v, f, nq, B, te, codes, cte, P, Q, lnu, kp)
    out = (Fn.linear(h, m.g6_out_w) + Fn.linear(h0, m.g6_skip_w) + m.g6_out_b).squeeze(-1)
    if Po is not None:
        LTG.CUR["aux"] = {"out": out, "sp": LTG.edge_dot(G).apply(Po, Qo, v, u, LTG.CHUNK), "u": u, "v": v, "n": N}
    return out


@contextlib.contextmanager
def on_term(G):
    """For the block: zltg's layer names are LTX's (zltg.term_step reads them by name), zg6r's g6r_stack is term_stack, a
    ZG6R built gets the term after zg6r's own __init__ (the basis loaded before), and ZG6R's forward hands its batch's store
    block, keep flags and questions' folds to the stack. All restored after, also on a failure."""
    orig_init, orig_fwd = G.ZG6R.__init__, G.ZG6R.forward

    def init(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        orig_init(self, blocks, widths, hidden, dropout, seed, arm, ctx)
        add_term(G, self, seed)

    def forward(self, feats, keep, nq, B, base_z):
        if "SEMB" not in feats:
            raise RuntimeError("zltx: no SEMB block in the batch (the scorers read the store's rows)")
        x = feats.get(NEXT)
        if x is None or x.shape[1] != WIDTH:
            raise RuntimeError(f"zltx: the batch holds no {NEXT} block of {WIDTH} column (LTX's carve adds the folds)")
        CUR["in"] = (feats["SEMB"], keep, fold_of(G.torch, x, nq, int(B)))
        try:
            return orig_fwd(self, feats, keep, nq, B, base_z)
        finally:
            CUR.pop("in", None)

    def stack(m, H, xb, lk, gd, nq, B, u01=None):
        if "in" not in CUR:
            raise RuntimeError("zltx: g6r_stack called outside ZG6R's forward")
        semb, keep, fq = CUR["in"]
        return term_stack(G, m, H, xb, lk, gd, nq, B, u01, semb, keep, fq)

    try:
        with M1.WQ.patch(LTG, TERM_LAYER=TERM_LAYER), M1.WQ.patch(G, g6r_stack=stack), \
                M1.WQ.patch(G.ZG6R, __init__=init, forward=forward):
            yield
    finally:
        CUR.clear()
        LTG.CUR.pop("aux", None)


@contextlib.contextmanager
def on_model(G):
    """The term (on_term) and LTG's next-hop loss in lean_gpu's listwiseD (zltg.on_aux); restored after."""
    with on_term(G), LTG.on_aux(G):
        yield


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def on_cols(out=None):
    """ltmcols' root, stage, declaration, column names and width LTX's (out: LTX's root, or a test's); restored after,
    also on a failure."""
    with M1.WQ.patch(LTMC, OUT=Path(COLS_OUT if out is None else out), STAGE=STAGE, DECLARED=DECLARED, COLS=COLS,
                     WIDTH=WIDTH):
        yield


@contextlib.contextmanager
def on_ltx(out=None):
    """zgnx with LTX's names, code, term record, NEXT block, width, model patch, arm import and stamp, and ltmcols LTX's
    (on_cols); restored after, also on a failure."""
    with on_cols(out), M1.WQ.patch(GNX, FITS=FITS, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, OUT=OUT,
                                   fit_name=fit_name, pair_name=pair_name, code_extra=code_extra,
                                   term_record=term_record, next_block=next_block, IN=IN, WIDTH=WIDTH,
                                   on_model=on_model, import_g=import_g, stamp_with_next=stamp_with_next):
        yield


@contextlib.contextmanager
def on_s0stop():
    """s0stop's stage table with LTX's entry (zltx, its root, its two seed-0 fits, LTG's rule, this declaration), in this
    process; restored after."""
    with M1.WQ.patch(S0, STAGES={**S0.STAGES, KEY: ("zltx", OUT, fit_name(0, S0.S0_SPLIT), fit_name(0, S0.HP_SPLIT),
                                                     "ltg", DECLARED)}):
        yield


def stop_record(base=ROOT):
    """LTX's current s0stop.json (s0stop.stop_record under LTX's entry), else None."""
    with on_s0stop():
        return S0.stop_record(KEY, base)


# ── the commands (zgnx's under LTX's names) ──────────────────────────────────


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zltx: {name} is not one of LTX's fits ({', '.join(FITS)})")
    p = ROOT / OUT / "cols.json"
    if not p.is_file() or not json.loads(p.read_text(encoding="utf-8")).get("pass"):
        raise SystemExit(f"zltx: {name}: the folds' check ({OUT}/cols.json) is missing or did not pass")
    with on_ltx():
        return GNX.fit(name)


def check_cols():
    with on_ltx():
        return GNX.check_cols()


def compare_all(base=ROOT):
    with on_ltx():
        return GNX.compare_all(base)


def fit_block(name, what, base=ROOT):
    """An LTX fit's stamp block (zgnx's fit_block under LTX's names)."""
    with on_ltx():
        return GNX.fit_block(name, what, base)


def fit_problems(call, base=ROOT):
    """zgnx.fit_problems with LTX's term, inside on_ltx: why an LTX fit that is read is not the declared one: its term (two
    folds' scorers of LTG's rank on the pinned basis's axes, LTG's lambda, three layers), base and ISO verdict; a train
    without a next-hop step; its NEXT block (the split, the fold column, one rule for train and read, every training carve
    in its train and every s1eval carve in its read); its start; its device; its trainer's code against the base's fit of
    its seed and split."""
    out = []
    want = {"term_parameters": N_TERM, "inputs": IN, "columns": list(LTMC.COLS), "layers": DEP.BASE_LAYERS,
            "folds": FOLDS, "rank": RANK, "axes": AXES, "lambda": LTG.LAMBDA, "basis_npz_sha256": LTG.BASIS_SHA}
    for name, (_a, sd, sp) in FITS.items():
        if not GNX.is_read(name, base):
            continue
        bse = DEP.base_fit(sd, sp, call)
        try:
            ks, kr = GNX.fit_block(name, "screen", base), GNX.fit_block(name, "read", base)
            kb = GNX.fit_block(bse, "screen", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        for what, k, need in (("train", ks, GNX.train_carves(sp)), ("read", kr, GNX.eval_carves())):
            got = {x: k.get(x) for x in want}
            if got != want or (k.get("base"), k.get("iso_verdict")) != (bse, call):
                out.append(f"{name}: its {what} names the term {got} on {k.get('base')} (ISO {k.get('iso_verdict')}), "
                           f"not {want} on {bse}")
            nk = k.get("next") or {}
            miss = sorted(set(need) - set(nk.get("carves") or {}))
            if (nk.get("split"), nk.get("columns")) != (sp, list(LTMC.COLS)) or miss:
                out.append(f"{name}: its {what}'s NEXT block is {nk.get('split')} {nk.get('columns')}, missing {miss}")
        if int((ks.get("aux") or {}).get("steps") or 0) <= 0:
            out.append(f"{name}: its train records no next-hop step")
        if (ks.get("next") or {}).get("fit_sha256") != (kr.get("next") or {}).get("fit_sha256"):
            out.append(f"{name}: its train and read read folds of different rules")
        enc, benc = ks.get("encoder") or {}, kb.get("encoder") or {}
        if call == ISO.REPLACE:
            ok = enc.get("start") == "own" and enc.get("copies") == 0
        else:
            ok = benc.get("fit") is not None and int(enc.get("copies") or 0) >= 1 and all(
                enc.get(x) == benc.get(x) for x in ("fit", "seed", "sha256"))
        if not ok:
            out.append(f"{name}: its encoder's start ({enc.get('start') or enc.get('fit')}, {enc.get('copies')} copies) "
                       f"is not {bse}'s (ISO {call})")
        if ks.get("device") != kb.get("device"):
            out.append(f"{name}: device {ks.get('device')}, {bse}'s {kb.get('device')}")
        cb, cn = kb.get("code") or {}, ks.get("code") or {}
        diff = sorted(k for k in cb if k not in GNX.CODE_SKIP and cn.get(k) != cb[k])
        if diff:
            out.append(f"{name}: code differs from {bse}'s in {diff}")
    return out


def ltx_problems(base=ROOT):
    """Why an LTX fit that is read did not read LTX's folds by this code: its train's and read's NEXT block must name LTX's
    declaration and ltxcols' rule, its train's term the one fold column and both scorers, and its train's code this
    zltx.py and ltxcols.py; and the folds' check (outputs/ltx/cols.json) must have passed with no NaN in any carve."""
    import ltxcols as LXC
    out, own, rule = [], own_shas(), LXC.rule_sha()
    with on_ltx():
        for name in FITS:
            if not GNX.is_read(name, base):
                continue
            try:
                ks, kr = GNX.fit_block(name, "screen", base), GNX.fit_block(name, "read", base)
            except SystemExit as e:
                out.append(str(e))
                continue
            for what, k in (("train", ks), ("read", kr)):
                nk = k.get("next") or {}
                if nk.get("declared_in") != DECLARED:
                    out.append(f"{name}: its {what}'s NEXT block names {nk.get('declared_in')}, not LTX's folds")
                if nk.get("fit_sha256") != rule:
                    out.append(f"{name}: its {what}'s folds come from the rule {str(nk.get('fit_sha256'))[:12]}, not "
                               f"ltxcols' {rule[:12]}")
            got = (ks.get("inputs"), ks.get("columns"), ks.get("term_parameters"), ks.get("folds"))
            if got != (IN, list(COLS), N_TERM, FOLDS):
                out.append(f"{name}: its train's term takes {got[0]} inputs of {got[1]} with {got[2]} parameters over "
                           f"{got[3]} folds, not LTX's")
            code = ks.get("code") or {}
            diff = sorted(k for k, v in own.items() if code.get(k) != v)
            if diff:
                out.append(f"{name}: its train's code is not this LTX's in {diff}")
        read = any(GNX.is_read(f, base) for f in FITS)
    c = Path(base) / OUT / "cols.json"
    if read:
        if not c.is_file():
            out.append(f"{M1.WQ.rel(c)}: the folds' check is missing")
        else:
            rec = json.loads(c.read_text(encoding="utf-8"))
            nan = sorted(f"{sp} {k}" for sp, by in (rec.get("splits") or {}).items() for k, r in by.items()
                         if int(r.get("nan") or 0))
            if not rec.get("pass") or nan:
                out.append(f"{M1.WQ.rel(c)}: the folds' check did not pass (NaN in {nan})")
    return out


def screen_beside(stage, base=ROOT):
    """A stage's filed screen and full-run verdicts and screen reads (outputs/<stage>/calls.json), else None."""
    p = Path(base) / "outputs" / stage / "calls.json"
    if not p.is_file():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    sc = rec.get("screen") or {}
    return {"screen": sc.get("verdict"), "full_run": (rec.get("full_run") or {}).get("verdict"),
            "reads": [{k: r.get(k) for k in ("split", "read", "delta", "call")} for r in sc.get("reads") or []]}


def next_form(screen, full):
    s = (screen or {}).get("verdict")
    v = (full or {}).get("verdict")
    if s == INCOMPLETE or v == INCOMPLETE:
        return "what is missing is fixed in an amendment, before any number is read"
    if s == ADOPT_FOR_FULL_RUN and v in (None, ADOPT):
        return ("the stack takes LTX's cross-fitted term by an amendment to docs/STK_GNN_STACK.md (LTX is not one of its "
                "members); LTX's own full run is held for the stack")
    return ("the cross-fitted next-hop term closes for the GNN; next form: LTG's cumulative residual "
            "(docs/LTG_NEXT_HOP_GNN.md section 6), in its own file")


def calls(base=ROOT):
    """LTX's screen (and its full run, when every fit is read) by LTG's in-domain rule, DEP's sixteen-read rule beside,
    through zgnx's reads, slices and checks under LTX's names; outputs/ltx/calls.json and .md."""
    with on_ltx():
        call = DEP.iso_call(base)
        new_r5, b0 = GNX.r5_gnx(call, base)
        base3 = DEP.base_seeds(call, base)
        bad = [f"{rd[0]} {rd[1]} seed {sd}: the base gives {v} in LTX's compare, {base3.get(rd, {}).get(sd)} in its own "
               f"stage's" for rd, by in sorted(b0.items()) for sd, v in sorted(by.items())
               if base3.get(rd, {}).get(sd) is None or abs(v - base3[rd][sd]) > GNX.R5_TOL]
        new_s, base_s, bad_s = GNX.slices(call, new_r5, base3, base)
        bad += bad_s
        new_all, base_all = {**new_r5, **new_s}, {**base3, **base_s}
        screen = LTG.decide_screen(new_all, base_all)
        full_ready = all(GNX.is_read(fit_name(sd, sp), base) for sd in RULE.SEEDS for sp in M1.SPLITS)
        full = LTG.decide_full(new_all, base_all) if full_ready else None
        dep_rule = {"screen": DEP.decide_screen(new_all, base_all)["verdict"],
                    "full_run": DEP.decide_full(new_r5, base3, new_s, base_s)["verdict"] if full_ready else None}
        bad += fit_problems(call, base)
        read = [f for f in FITS if GNX.is_read(f, base)]
        params = {f: {k: GNX.fit_block(f, "screen", base).get(k) for k in term_record_keys()} for f in read}
        aux = {}
        for f in read:
            a = GNX.fit_block(f, "screen", base).get("aux") or {}
            aux[f] = {"steps": a.get("steps"), "lambda": a.get("lambda"), "last": (a.get("curve") or [None])[-1]}
    bad += ltx_problems(base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": f"{GNX.ARM} with LTX's cross-fitted next-hop term",
           "iso_verdict": call, "base": "zg6r-m (ISO)" if call == ISO.REPLACE else "zg6r (ML1)", "screen": screen,
           "full_run": full, "dep_rule_beside": dep_rule, "next_form": next_form(screen, full),
           "beside": {st: screen_beside(st, base) for st in BESIDE}, "problems": bad, "parameters": params, "aux": aux,
           **own_shas(), "zgnx_sha256": M1.WQ.sha_src(GNX.__file__), "zltg_sha256": M1.WQ.sha_src(LTG.__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    md = [f"# LTX: {GNX.ARM} with the cross-fitted next-hop term against {rec['base']}, on PKO's pools", "",
          f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}",
          f"DEP's sixteen-read rule beside: screen {dep_rule['screen']}, full run {dep_rule['full_run']}",
          f"next form: {rec['next_form']}", ""]
    rows = screen["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in screen.get("beside", [])]
    if rows:
        md += ["| split | read | base s0, s1, s2 | LTX s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
        md += [f"| {r['split']} | {r['read']} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | "
               f"{r['delta']:+.4f} | {r['floor']:.4f} | {r['call']} |" for r in rows]
    if full is not None:
        md += ["", f"## Full run: {full['verdict']}", ""]
        frows = full["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in full.get("beside", [])]
        if frows:
            md += ["| split | read | d0 | d1 | d2 | floor | call |", "|---|---|---:|---:|---:|---:|---|"]
            for r in frows:
                d = r["delta"]
                md.append(f"| {r['split']} | {r['read']} | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} | {r['floor']:.4f} | "
                          f"{r['call']} |")
    if aux:
        md += ["", "next-hop loss (train, last window): " + "; ".join(
            f"{f} {a['steps']} steps, {json.dumps(a['last'])}" for f, a in aux.items())]
    for st in BESIDE:
        b = rec["beside"][st]
        if b:
            md += ["", f"{st.upper()} beside: screen {b['screen']}, full run {b['full_run']}"]
    if bad:
        md += ["", "problems:"] + [f"- {p}" for p in bad]
    (out / "calls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"screen {screen['verdict']} {json.dumps(screen.get('counts', {}))}"
        + (f"; full run {full['verdict']}" if full else "") + f"; DEP's rule beside {dep_rule}; {OUT}/calls.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def must_stop(fn, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} passed")


def snapshot(G=None):
    s = {**{f"gnx.{k}": getattr(GNX, k) for k in PATCHED}, **{f"ltmcols.{k}": getattr(LTMC, k) for k in COLS_PATCHED},
         **{f"m1.{k}": getattr(M1, k) for k in ("DECLARED", "STAGE", "KEY", "pko_fit_name", "stamp", "code_shas",
                                                 "load_encoder", "train", "read")}, "n_fits": len(M1.ALL_FITS),
         "ltg.term_layer": LTG.TERM_LAYER}
    if G is not None:
        s.update({"g.encoder_used": G.encoder_used, "g.init": G.ZG6R.__init__, "g.forward": G.ZG6R.forward,
                  "g.stack": G.g6r_stack, "g.carve_init": G.G6rLinkCarve.__init__, "g.carve_batch": G.G6rLinkCarve.batch,
                  "lg.listwise": G.LG.listwiseD, "lg.model": G.LG.LeanMLP8D})
    return s


def toy_folds(out, split, ds, carve, cache_root, rng, fit_sha, training=False):
    """LTX's layout for one toy carve: each question's fold drawn (the carve's first two questions 0 and 1) on every one of
    its rows; the part and carve records ltmcols' names give inside on_cols; the questions' folds and the columns."""
    import lean_cache as LC
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    cd = LTMC.cols_dir(out, split, ds, carve)
    ps, fs, xs = [], [], []
    for i, (p, r) in enumerate(zip(parts, recs)):
        n = np.load(p / "n.npy").astype(np.int64)
        k = int(r["rows"])
        assert int(n.sum()) == k and n.size == int(r["queries"]) and n.size >= 2
        f = rng.integers(0, FOLDS, n.size).astype(np.int64)
        if i == 0:
            f[:2] = (0, 1)
        C = np.repeat(f.astype(np.float16), n).reshape(-1, WIDTH)
        pd = cd / p.name
        pd.mkdir(parents=True, exist_ok=True)
        np.save(pd / "next.npy", C)
        sha = LC.sha_file(pd / "next.npy")
        M1.WQ.write_json(pd / "record.json", {
            "stage": LTMC.STAGE, "declared_in": LTMC.DECLARED, "split": split, "dataset": ds, "carve": carve,
            "part": p.name, "cache_record_sha256": LC.sha_file(p / "record.json"), "rows": k,
            "queries": int(r["queries"]), "columns": list(LTMC.COLS), "fit_sha256": fit_sha, "training_carve": training,
            "next_sha256": sha})
        ps.append({"part": p.name, "rows": k, "next_sha256": sha})
        fs.append(f)
        xs.append(C)
    M1.WQ.write_json(cd / "record.json", {
        "stage": LTMC.STAGE, "declared_in": LTMC.DECLARED, "split": split, "dataset": ds, "carve": carve,
        "training_carve": training, "rows": int(sum(q["rows"] for q in ps)), "parts": ps, "columns": list(LTMC.COLS),
        "fit_sha256": fit_sha, "identity": {"pass": True}})
    return np.concatenate(fs), np.concatenate(xs)


def toy_term(G):
    """LTX on zg6r's own toy carve (GNX's toy: zg6r's self-test steps 2 and 3) with LTX-layout toy folds and LTG's toy
    basis: the carve's columns are LTX's files and the batch's NEXT rows its questions' folds; the term's shapes, V's draws
    (scorer 0's LTG's) and zeros; at the start the eval and the training forward are zg6r's bit for bit; each scorer's P
    and Q are zltg.term_inputs' on a view of it; the layers read the other fold's scorer, detached, the next-hop loss the
    own fold's; with the two scorers equal the forwards are LTG's model's bit for bit; the main loss gives no scorer a
    gradient and every layer's term one; the next-hop loss on one fold's golds moves that fold's scorer alone; no NEXT, a
    wider NEXT, no SEMB, a fold other than 0 and 1, a question of two folds, an edge between questions and the stack
    outside the forward stop; checkpointing on and off agree; the strict load with and without the patch; a one-epoch
    lean_gpu.fit_variant with the loss hook firing; the patches' restoring."""
    import types
    import torch
    G6 = G.G6
    G.LG.bind_device_ops()
    out = {}
    sp = "WQ-musique"
    with tempfile.TemporaryDirectory(prefix="zltx_") as td:
        tmp = Path(td)
        rng = np.random.default_rng(43)
        G.RM.toy_roots(tmp, rng, chunks=(12, 12))
        G.RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                   rel_dir=tmp / "rel")
        G.ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        kw = dict(look_root=tmp / "look", cache_root=tmp / "cache", links_root=tmp / "lk", ch_root=tmp / "ch")

        class ToyIds(G6.ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        F0, X0 = toy_folds(tmp / "ltx", sp, "metaqa", "toy", tmp / "cache", rng, "a" * 64)
        LTG.toy_basis(G, tmp / "cache", rng)
        st, _rec = LTG.load_basis(G, tmp / "cache", sha=None)
        A = int(st.V.shape[1])
        out["axes"] = A
        with G.U2N.extra_carves(("metaqa", "toy")):
            G.U2N.build_typed("metaqa", "toy", tmp / "look", tmp / "typed", **kw)
            plain = G.ZL.link_carve(G.RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
            c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            G.Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
            cls = G.g6r_carve(G.Z3.kt_carve(plain, tmp / "kt"), tmp / "typed")
            init0, batch0 = cls.__init__, cls.batch
            with GNX.on_carve(G, cls):
                GNX.LOADED.clear()
                with GNX.on_split(sp, out=tmp / "ltx"):
                    ce = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
                    nx = GNX.next_block(sp)
                out["loaded"] = sorted(nx["carves"])
                out["next_block_ltx"] = (nx["declared_in"], nx["columns"], nx["fit_sha256"]) == (
                    DECLARED, list(COLS), "a" * 64)
                out["carve_cols_are_files"] = bool(torch.equal(ce.gnx_X, torch.from_numpy(X0)))
                blocks = ["rank", "SEMB"]
                qs = np.r_[np.arange(ce.rows)[::-1][:7], np.arange(ce.rows)[:5]].astype(np.int64)
                B = qs.size
                feats, nq, bz, gold0 = ce.batch(qs, blocks)
            out["carve_restored"] = cls.__init__ is init0 and cls.batch is batch0
            x = feats[NEXT]
            out["next_rows_are_folds"] = bool(torch.equal(
                x[:, 0], torch.from_numpy(np.repeat(F0[qs], ce.n_np[qs]).astype(np.float32))))
            fq = fold_of(torch, x, nq, B)
            out["fold_of"] = bool(torch.equal(fq, torch.from_numpy(F0[qs])))
            out["batch_folds"] = sorted(set(fq.tolist()))
            keep = torch.ones(B, len(blocks))
            lk, gd = feats[G.LK_KEY], feats[G.G6R_KEY]
            u_all, v_all = lk["u"], lk["v"]
            N = int(nq.numel())
            out["cross_edges"] = int((nq[u_all] != nq[v_all]).sum())
            ef = fq[nq[u_all]]
            out["fold_edges"] = [int(((ef == k) & (u_all != v_all)).sum()) for k in range(FOLDS)]

            def gold_in(folds, u_, v_):
                """Gold rows on a forward's kept edges, in the folds' questions alone: up to three senders of two or more
                kept out-edges each (drawn as zltg's toy draws) and one receiver of each; with, per fold, the gold
                senders whose out-edges are not all gold (the senders the next-hop loss has a gradient on)."""
                nz = u_ != v_
                deg = torch.zeros(N, dtype=torch.int64).index_add_(0, u_, torch.ones_like(u_))
                g_ = torch.zeros(N, dtype=torch.bool)
                for k in folds:
                    cand = torch.nonzero((fq[nq[u_]] == k) & nz & (deg[u_] >= 2)).view(-1)
                    cand = cand[torch.randperm(cand.numel(), generator=torch.Generator().manual_seed(4))]
                    took = set()
                    for e in cand.tolist():
                        if int(u_[e]) in took:
                            continue
                        took.add(int(u_[e]))
                        g_[u_[e]] = True
                        g_[v_[e]] = True
                        if len(took) == 3:
                            break
                pos = g_[u_] & g_[v_] & nz
                hp = torch.zeros(N, dtype=torch.bool)
                hp[u_[pos]] = True
                hn = torch.zeros(N, dtype=torch.bool)
                hn[u_[~pos]] = True
                return g_, [int((hp & hn & (fq[nq] == k)).sum()) for k in range(FOLDS)]
            r1 = fq[nq] == 1
            semb = feats["SEMB"]
            orig = (G.g6r_stack, G.ZG6R.__init__, G.ZG6R.forward, LTG.TERM_LAYER)
            orig_stack, orig_fwd = G.g6r_stack, G.ZG6R.forward
            names0 = tuple(G6.G6_NAMES) + tuple(G.G6R_NAMES)
            gen = torch.Generator().manual_seed(3)
            with on_term(G):
                torch.manual_seed(0)
                m = G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)
                out["term_shapes"] = {k: tuple(getattr(m, k).shape) for k in TERM_NAMES[:6]}
                out["term_params"] = sum(int(getattr(m, k).numel()) for k in TERM_NAMES)
                out["want_params"] = n_term(A)
                g = torch.Generator().manual_seed(0 + LTG.SEED_OFF)
                v0, v1 = torch.randn(3, A, RANK, generator=g), torch.randn(3, A, RANK, generator=g)
                out["v_draws"] = bool(torch.equal(m.ltx_V[0], v0)) and bool(torch.equal(m.ltx_V[1], v1)) and not bool(
                    torch.equal(v0, v1))
                out["rest_zero"] = all(bool((getattr(m, k) == 0).all()) for k in TERM_NAMES if k != "ltx_V")
                out["buffers_not_saved"] = not any(k.startswith("ltg_b") for k in m.state_dict())
                with torch.no_grad():
                    for k in names0:
                        getattr(m, k).copy_(torch.randn(getattr(m, k).shape, generator=gen) * 0.3)
                m.eval()
                with torch.no_grad():
                    H = m.encode(feats, keep, nq, B)
                    xb = torch.cat([feats[G.KT_KEY].to(H.dtype), m.chain(feats, keep, nq).to(H.dtype)], 1)
                    y0 = m(feats, keep, nq, B, bz)
                    want0 = orig_stack(m, H, xb, lk, gd, nq, B)
                out["eval_start_is_zg6r"] = bool(torch.equal(y0, want0))
                m.train()
                gs = m.g6_gen.get_state()
                torch.manual_seed(9)
                y1 = m(feats, keep, nq, B, bz)
                out["aux_left_in_training"] = "aux" in LTG.CUR
                LTG.CUR.pop("aux", None)
                with M1.WQ.patch(G, g6r_stack=orig_stack), M1.WQ.patch(G.ZG6R, forward=orig_fwd):
                    m.g6_gen.set_state(gs)
                    torch.manual_seed(9)
                    y2 = m(feats, keep, nq, B, bz)
                out["train_start_is_zg6r"] = bool(torch.equal(y1, y2))
                # random term parameters, the two scorers apart: each scorer is zltg's on a view; the layers read the other
                # fold's, detached, the next-hop loss the own fold's
                with torch.no_grad():
                    for k in TERM_NAMES:
                        getattr(m, k).copy_(torch.randn(getattr(m, k).shape, generator=gen) * 0.3)
                sd_rand = {k: t.clone() for k, t in m.state_dict().items()}
                m.eval()
                with torch.no_grad():
                    qk, xx, cc, nr = LTG.store_terms(G, m, semb, nq)
                    ref, ok = [], []
                    for s in range(FOLDS):
                        view = types.SimpleNamespace(ltg_V=m.ltx_V[s], ltg_U=m.ltx_U[s], ltg_bV=m.ltg_bV, ltg_bm=m.ltg_bm,
                                                     ltg_bc=m.ltg_bc, ltg_kappa=m.ltg_kappa, j_semb=m.j_semb)
                        ref.append(LTG.term_inputs(G, view, semb, keep, nq, u_all, N))
                        Ps, Qs = scorer_pq(torch, m.ltx_V[s], m.ltx_U[s], qk, xx, cc, nr, nq)
                        ok.append(bool(torch.equal(Ps, ref[s][0])) and bool(torch.equal(Qs, ref[s][1])))
                    out["scorers_are_ltg"] = ok
                    out["scorers_differ"] = not bool(torch.equal(ref[0][0], ref[1][0]))
                    P, Q, lnu, kp, Po, Qo = term_inputs(G, m, semb, keep, nq, u_all, v_all, N, fq)
                out["layers_read_other_fold"] = all(bool(torch.equal(t[r1], ref[0][j][r1])) and bool(
                    torch.equal(t[~r1], ref[1][j][~r1])) for j, t in ((0, P), (1, Q)))
                out["lnu_kp_are_ltg"] = bool(torch.equal(lnu, ref[0][2])) and bool(torch.equal(kp, ref[0][3]))
                out["own_none_in_eval"] = Po is None and Qo is None
                m.train()
                P, Q, lnu, kp, Po, Qo = term_inputs(G, m, semb, keep, nq, u_all, v_all, N, fq)
                out["layers_detached"] = not P.requires_grad and not Q.requires_grad
                out["own_reads_own_fold"] = Po.requires_grad and Qo.requires_grad and all(
                    bool(torch.equal(t.detach()[r1], ref[1][j][r1])) and bool(torch.equal(t.detach()[~r1], ref[0][j][~r1]))
                    for j, t in ((0, Po), (1, Qo)))
                # the main loss alone: no scorer gets a gradient, every layer's term does
                m.zero_grad()
                torch.manual_seed(9)
                s_ = m(feats, keep, nq, B, bz)
                LTG.CUR.pop("aux", None)
                G.LG.listwiseD(s_, gold0, nq, B).backward()
                out["main_scorer_grads"] = [k for k in ("ltx_V", "ltx_U") if getattr(m, k).grad is not None
                                            and float(getattr(m, k).grad.abs().sum()) > 0]
                out["main_layer_no_grad"] = [k for k in TERM_NAMES[2:] if getattr(m, k).grad is None
                                             or float(getattr(m, k).grad.abs().sum()) == 0]
                # the next-hop loss on one fold's golds: that fold's scorer alone moves
                for k in range(FOLDS):
                    LTG.aux_reset()
                    with LTG.on_aux(G):
                        m.zero_grad()
                        torch.manual_seed(9)
                        s_ = m(feats, keep, nq, B, bz)
                        g_, inf_ = gold_in([k], LTG.CUR["aux"]["u"], LTG.CUR["aux"]["v"])
                        G.LG.listwiseD(s_, g_, nq, B).backward()
                    out[f"aux_fold{k}"] = {"steps": LTG.AUX["steps"], "senders": inf_,
                                           "U": [float(m.ltx_U.grad[j].abs().sum()) for j in range(FOLDS)],
                                           "V": [float(m.ltx_V.grad[j].abs().sum()) for j in range(FOLDS)]}
                LTG.aux_reset()
                # what stops
                m.eval()
                x2 = x.clone()
                rr = int(next(r for r in range(N) if int((nq == nq[r]).sum()) > 1))
                x2[rr, 0] = 1.0 - x2[rr, 0]
                far = int(next(r for r in range(N) if int(nq[r]) != int(nq[u_all[0]])))
                v2 = v_all.clone()
                v2[0] = far
                with torch.no_grad():
                    for what, f_ in (("no_next", {k: t for k, t in feats.items() if k != NEXT}),
                                     ("wide_next", {**feats, NEXT: x.repeat(1, 2)}),
                                     ("no_semb", {k: t for k, t in feats.items() if k != "SEMB"}),
                                     ("half_fold", {**feats, NEXT: x * 0.5}),
                                     ("two_folds", {**feats, NEXT: x2}),
                                     ("cross_edge", {**feats, G.LK_KEY: {**lk, "v": v2}})):
                        try:
                            m(f_, keep, nq, B, bz)
                            out[what] = "passed"
                        except RuntimeError:
                            out[what] = "refused"
                    try:
                        G.g6r_stack(m, H, xb, lk, gd, nq, B)
                        out["stack_outside"] = "passed"
                    except RuntimeError:
                        out["stack_outside"] = "refused"
                    out["in_cleared"] = "in" not in CUR
                # the two scorers equal: the forwards to set beside LTG's model's
                with torch.no_grad():
                    m.ltx_V[1].copy_(m.ltx_V[0])
                    m.ltx_U[1].copy_(m.ltx_U[0])
                    ya = m(feats, keep, nq, B, bz)
                m.train()
                m.g6_gen.set_state(gs)
                torch.manual_seed(9)
                yb = m(feats, keep, nq, B, bz).detach()
                LTG.CUR.pop("aux", None)
                sd_eq = {k: t.clone() for k, t in m.state_dict().items()}
                # checkpointing on and off, with the next-hop loss on both folds' golds: every term parameter moves
                m.load_state_dict(sd_rand, strict=True)
                grads = {}
                for ck in (True, False):
                    G.CKPT["on"] = ck
                    try:
                        with LTG.on_aux(G):
                            m.zero_grad()
                            m.g6_gen.set_state(gs)
                            torch.manual_seed(9)
                            yt = m(feats, keep, nq, B, bz)
                            g_, inf_ = gold_in(range(FOLDS), LTG.CUR["aux"]["u"], LTG.CUR["aux"]["v"])
                            G.LG.listwiseD(yt, g_, nq, B).backward()
                        grads[ck] = (yt.detach().clone(), {k: getattr(m, k).grad.clone() for k in TERM_NAMES
                                                           if getattr(m, k).grad is not None}, inf_)
                    finally:
                        G.CKPT["on"] = True
                LTG.aux_reset()
                out["term_no_grad"] = [k for k in TERM_NAMES if k not in grads[True][1]
                                       or float(grads[True][1][k].abs().sum()) == 0]
                out["ckpt_senders"] = [grads[True][2], grads[False][2]]
                out["ckpt_both_scorers"] = all(float(grads[True][1][k][j].abs().sum()) > 0 for k in ("ltx_V", "ltx_U")
                                               for j in range(FOLDS))
                out["ckpt_same_score"] = bool(torch.equal(grads[True][0], grads[False][0]))
                out["ckpt_grad_max_abs"] = max(float((grads[True][1][k] - grads[False][1][k]).abs().max())
                                               for k in grads[True][1])
                sd = m.state_dict()
                G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0).load_state_dict(sd, strict=True)
                out["strict_load_patched"] = "passed"
            out["restored"] = (G.g6r_stack, G.ZG6R.__init__, G.ZG6R.forward, LTG.TERM_LAYER) == orig
            # LTG's model with the equal scorers' parameters (scorer 0 as LTG's), under LTG's own patches
            with LTG.on_term(G):
                mg = G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)
                sd_g = {}
                for k, t in sd_eq.items():
                    if k in ("ltx_V", "ltx_U"):
                        sd_g["ltg_" + k[4:]] = t[0]
                    elif k.startswith("ltx_"):
                        sd_g["ltg_" + k[4:]] = t
                    else:
                        sd_g[k] = t
                mg.load_state_dict(sd_g, strict=True)
                mg.eval()
                with torch.no_grad():
                    yc = mg(feats, keep, nq, B, bz)
                mg.train()
                mg.g6_gen.set_state(gs)
                torch.manual_seed(9)
                yd = mg(feats, keep, nq, B, bz).detach()
                LTG.CUR.pop("aux", None)
            out["equal_scorers_eval_is_ltg"] = bool(torch.equal(ya, yc))
            out["equal_scorers_train_is_ltg"] = bool(torch.equal(yb, yd))
            plain_m = G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)
            out["plain_has_term"] = any(hasattr(plain_m, k) for k in TERM_NAMES)
            try:
                plain_m.load_state_dict(sd, strict=True)
                out["strict_load_plain"] = "passed"
            except RuntimeError:
                out["strict_load_plain"] = "refused"
            # a one-epoch lean_gpu.fit_variant on the patched carve and model: the hook fires inside the trainer's loop
            cfg = {"lr": 2e-3, "wd": 1e-4, "dropout": 0.1, "epochs": 1, "swa_from": 0, "cos": 0, "adamw": 0, "drop": 0}
            LTG.aux_reset()
            with GNX.on_carve(G, cls), on_model(G), M1.WQ.patch(G.LG, LeanMLP8D=G.ZG6R):
                res = G.LG.fit_variant([ce], blocks, cfg, 0, 16, "none", "cpu")
            summ = LTG.aux_summary()
            out["fit_aux_steps"] = summ["steps"]
            out["fit_U_moved"] = [float(res["swa"]["ltx_U"][k].abs().sum()) > 0 for k in range(FOLDS)]
            out["fit_loss_finite"] = all(math.isfinite(c["loss"]) for c in res["curve"])
            out["fit_restored"] = (G.LG.listwiseD.__name__ == "listwiseD" and G.LG.LeanMLP8D is not G.ZG6R
                                   and cls.batch is batch0 and G.ZG6R.__init__ is orig[1])
            LTG.aux_reset()
        GNX.LOADED.clear()
    return out


def selftest():
    import ltxcols as LXC
    import zrtr as RTR
    import zira as IRA
    import zisb as ISB
    import zgna as GNA
    import zlta as ZLA
    import zgnp as GNP
    import zgnw as GNW
    # 1. names: LTX's own, none another stage's; the constants ltxcols' own
    assert fit_name(0, "WQ-musique") == "scr-ltx-xf-s0" and fit_name(2, "WQ-hotpotqa") == "scr-ltx-xf-s2-hp"
    assert pair_name(1, "WQ-hotpotqa") == "scr-ltx-xf-vs-base-s1-hp"
    assert not set(FITS) & (set(RTR.FITS) | set(IRA.FITS) | set(ISB.FITS) | set(LTM.FITS) | set(GNA.FITS)
                            | set(ZLA.FITS) | set(GNP.FITS) | set(GNW.FITS) | set(GNX_FITS))
    assert (STAGE, DECLARED, COLS, WIDTH, FOLDS, COLS_OUT) == (LXC.STAGE, LXC.DECLARED, LXC.COLS, LXC.WIDTH, LXC.K,
                                                               LXC.OUT)
    assert TERM_LAYER[0] == ("ltx_0_w", "ltx_0_g", "ltx_0_em", "ltx_0_ea") and len(TERM_NAMES) == 2 + 4 * LAYERS
    # 2. a batch's folds: each question's from its rows; a value other than 0 and 1 and a question of two folds stop; a
    #    question without rows is in fold 0
    import torch
    nq_ = torch.tensor([0, 0, 1, 2, 2, 2, 4])
    x_ = torch.tensor([[1.0], [1.0], [0.0], [1.0], [1.0], [1.0], [0.0]])
    assert fold_of(torch, x_, nq_, 5).tolist() == [1, 0, 1, 0, 0]
    for bad in ([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0]], [[1.0], [1.0], [0.5], [1.0], [1.0], [1.0], [0.0]]):
        try:
            fold_of(torch, torch.tensor(bad), nq_, 5)
            raise AssertionError("fold_of passed")
        except RuntimeError:
            pass
    try:
        cross_edges(torch, nq_, torch.tensor([0, 3]), torch.tensor([1, 2]))
        raise AssertionError("cross_edges passed")
    except RuntimeError:
        pass
    cross_edges(torch, nq_, torch.tensor([0, 3, 5]), torch.tensor([1, 5, 4]))
    # 3. the patches: zgnx's names, code, width, model patch, arm import and stamp LTX's and ltmcols' root, stage,
    #    declaration, columns and width LTX's inside; restored after, also on a failure
    s0 = snapshot()
    with on_ltx():
        assert GNX.FITS is FITS and (GNX.DECLARED, GNX.STAGE, GNX.KEY, GNX.OUT) == (DECLARED, STAGE, KEY, OUT)
        assert (GNX.IN, GNX.WIDTH, LTMC.WIDTH, tuple(LTMC.COLS)) == (1, 1, 1, COLS) and GNX.on_model is on_model
        assert GNX.import_g is import_g and GNX.stamp_with_next is stamp_with_next
        assert GNX.pairs(ISO.KEEP)[pair_name(0, "WQ-musique")] == ("scr-ltx-xf-s0", "scr-ml1-zg6r-s0")
        assert GNX.pairs(ISO.REPLACE)[pair_name(0, "WQ-hotpotqa")] == ("scr-ltx-xf-s0-hp", "scr-iso-zg6rm-s0-hp")
        assert (Path(LTMC.OUT), LTMC.STAGE, LTMC.DECLARED) == (COLS_OUT, "LTX", DECLARED)
        cs = GNX.code_extra()
        assert all(f"{k}_sha256" in cs for k in ("zltx", "ltxcols", "zgnx", "zdep", "zltg", "zltm", "ltmcols")), cs
        with GNX.on_names():
            assert (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY) and set(FITS) <= set(M1.ALL_FITS)
            assert not set(GNX_FITS) & set(M1.ALL_FITS)
            assert all(f"{k}_sha256" in M1.code_shas(GNX.ARM) for k in ("zltx", "ltxcols"))
    assert snapshot() == s0 and (GNX.IN, LTMC.WIDTH, LTMC.STAGE) == (15, 7, "LTM") and GNX.on_model is not on_model
    try:
        with on_ltx():
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert snapshot() == s0
    for other in ("scr-gnx-nx-s0", "scr-ltg-nh-s0", "scr-gnw-mw-s0"):
        must_stop(fit, other)
    st0 = dict(S0.STAGES)
    with on_s0stop():
        assert S0.STAGES[KEY] == ("zltx", OUT, "scr-ltx-xf-s0", "scr-ltx-xf-s0-hp", "ltg", DECLARED)
        assert S0.STAGES["ltg"] == st0["ltg"] and S0.deciding("ltg") == S0.deciding(S0.STAGES["gnx"][4])
    assert S0.STAGES == st0 and KEY not in S0.STAGES
    tmp = Path(tempfile.mkdtemp(prefix="zltx_"))
    try:                                         # LTX's stop record on made files: current only for the read it was made on
        rp = S0.read_path(fit_name(0, "WQ-musique"), tmp)
        rp.parent.mkdir(parents=True)
        rp.write_text("{}", encoding="utf-8")
        (tmp / OUT).mkdir(parents=True)
        M1.WQ.write_json(tmp / OUT / "s0stop.json", {"s0_fit": fit_name(0, "WQ-musique"), "hp_fit": fit_name(0, "WQ-hotpotqa"),
                                                     "s0_read_sha256": S0.sha_file(rp), "decision": S0.STOP})
        assert stop_record(tmp)["decision"] == S0.STOP and KEY not in S0.STAGES
        rp.write_text("{ }", encoding="utf-8")
        assert stop_record(tmp) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 4. the columns an LTX carve loads are ltxcols': they load inside on_ltx, from ltmcols' root there, and stop outside
    #    it; LTM's and GNW's columns stop inside it; the NEXT block names LTX's declaration, root and rule
    import lean_cache as LC
    import ltwdiag as LTW
    tmp = Path(tempfile.mkdtemp(prefix="zltx_"))
    try:
        rng = np.random.default_rng(29)
        M = LTW.unit(rng.normal(size=(200, 16))).astype(np.float32)
        LTW.toy_tree(tmp, "musique", "s1eval", M, rng, nq=6, n=20)
        parts, _recs = LC.part_dirs("musique", "s1eval", tmp / "cache")
        ps = {p.name: LC.sha_file(p / "record.json") for p in parts}
        sp = "WQ-hotpotqa"
        rec = LXC.cols("musique", "s1eval", tmp, tmp / "ltx")
        with GNW.on_cols(tmp / "gnw"):
            GNW.toy_cols15(tmp / "gnw", sp, "musique", "s1eval", tmp / "cache", rng, "d" * 64)
        Y = LTM.toy_cols(tmp / "ltm", sp, "musique", "s1eval", tmp / "cache", rng, "e" * 64)
        assert LTM.load_next("musique", "s1eval", sp, ps, tmp / "ltm")[1]["stage"] == "LTM" and Y.shape[1] == 7
        with on_ltx(tmp / "ltx"):
            X2, r2, _sha = LTM.load_next("musique", "s1eval", sp, ps)      # out None: ltmcols' OUT, LTX's here
            assert X2.shape == (rec["rows"], 1) and (r2["stage"], r2["declared_in"]) == ("LTX", DECLARED), r2
            assert set(np.unique(X2).tolist()) <= {0.0, 1.0}
            with GNX.on_split(sp, rec["fit_sha256"]):
                GNX.LOADED.append({"dataset": "musique", "carve": "s1eval", "record_sha256": "r"})
                nb = GNX.next_block(sp)
            GNX.LOADED.clear()
            assert (nb["declared_in"], Path(nb["columns_root"]).name, nb["fit_sha256"], nb["columns"]) == (
                DECLARED, "ltx", LXC.rule_sha(), list(COLS)), nb
            must_stop(LTM.load_next, "musique", "s1eval", sp, ps, tmp / "gnw")   # GNW's columns inside
            must_stop(LTM.load_next, "musique", "s1eval", sp, ps, tmp / "ltm")   # LTM's columns inside
        must_stop(LTM.load_next, "musique", "s1eval", sp, ps, tmp / "ltx")      # LTX's columns outside on_ltx
        assert snapshot() == s0
    finally:
        GNX.LOADED.clear()
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. LTX's toy term under LTX's patches
    G = GNX.import_g()
    sg = snapshot(G)
    with on_ltx():
        t = toy_term(G)
        rec = GNX.term_record(G)
    A = t["axes"]
    assert A == AXES and t["want_params"] == N_TERM and t["term_params"] == N_TERM, t
    assert t["loaded"] == ["metaqa/toy"] and t["next_block_ltx"] and t["carve_cols_are_files"] and t["carve_restored"], t
    assert t["next_rows_are_folds"] and t["fold_of"] and t["batch_folds"] == [0, 1] and t["cross_edges"] == 0, t
    assert min(t["fold_edges"]) > 0, t
    assert t["term_shapes"] == {"ltx_V": (2, 3, A, RANK), "ltx_U": (2, 3, A, RANK), "ltx_0_w": (3 * RANK, D),
                                "ltx_0_g": (FAMS,), "ltx_0_em": (FAMS, D), "ltx_0_ea": (FAMS, D)}, t["term_shapes"]
    assert t["v_draws"] and t["rest_zero"] and t["buffers_not_saved"] and not t["plain_has_term"], t
    assert t["eval_start_is_zg6r"] and t["train_start_is_zg6r"] and t["aux_left_in_training"], t
    assert t["scorers_are_ltg"] == [True, True] and t["scorers_differ"], t
    assert t["layers_read_other_fold"] and t["lnu_kp_are_ltg"] and t["own_none_in_eval"], t
    assert t["layers_detached"] and t["own_reads_own_fold"], t
    assert t["main_scorer_grads"] == [] and t["main_layer_no_grad"] == [], t
    for k in range(FOLDS):
        a = t[f"aux_fold{k}"]
        assert a["steps"] == 1 and a["senders"][k] > 0 and a["senders"][1 - k] == 0, (k, a)
        assert a["U"][k] > 0 and a["U"][1 - k] == 0 and a["V"][k] > 0 and a["V"][1 - k] == 0, (k, a)
    assert all(t[w] == "refused" for w in ("no_next", "wide_next", "no_semb", "half_fold", "two_folds", "cross_edge",
                                           "stack_outside")), t
    assert t["in_cleared"] and t["restored"], t
    assert t["equal_scorers_eval_is_ltg"] and t["equal_scorers_train_is_ltg"], t
    assert t["term_no_grad"] == [] and t["ckpt_same_score"] and t["ckpt_grad_max_abs"] < 1e-5, t
    assert t["ckpt_senders"][0] == t["ckpt_senders"][1] and min(t["ckpt_senders"][0]) > 0 and t["ckpt_both_scorers"], t
    assert t["strict_load_patched"] == "passed" and t["strict_load_plain"] == "refused", t
    assert t["fit_aux_steps"] > 0 and any(t["fit_U_moved"]) and t["fit_loss_finite"] and t["fit_restored"], t
    assert rec["term"].startswith("LTX's cross-fitted") and rec["term_parameters"] == N_TERM, rec
    assert rec["inputs"] == 1 and rec["parameters_h128"] == 180218 + 117897 == 298115 and rec["folds"] == 2, rec
    assert rec["columns"] == list(COLS) and set(term_record_keys()) <= set(rec), rec
    assert snapshot(G) == sg
    # 6. the stamp under LTX's names: the train's NEXT block is LTX's, with LTX's term and the next-hop curve; a read's
    #    holds no curve; one without the columns stops
    seen, orig_stamp = [], M1.stamp
    hp = fit_name(0, "WQ-hotpotqa")
    try:
        M1.stamp = lambda name, env, what, extra=None: seen.append((name, what, dict(extra or {})))
        with on_ltx(), GNX.on_gnx(G, ISO.KEEP):
            assert (M1.STAGE, M1.KEY) == (STAGE, KEY) and set(FITS) <= set(M1.ALL_FITS)
            assert G.ZG6R.__init__ is not sg["g.init"] and LTG.TERM_LAYER == TERM_LAYER
            with GNX.on_split("WQ-hotpotqa"):
                must_stop(M1.stamp, hp, {}, "screen", {"x": 1})
            with GNX.on_split("WQ-hotpotqa", "c" * 64):
                GNX.LOADED.extend({"dataset": k.split("/")[0], "carve": k.split("/")[1], "record_sha256": "r"}
                                  for k in GNX.train_carves("WQ-hotpotqa"))
                M1.stamp(hp, {}, "screen", {"x": 1})
            GNX.LOADED.clear()
            with GNX.on_split("WQ-hotpotqa", "c" * 64):
                GNX.LOADED.extend({"dataset": k.split("/")[0], "carve": k.split("/")[1], "record_sha256": "r"}
                                  for k in GNX.eval_carves())
                M1.stamp(hp, {}, "read", {"x": 2})
            GNX.LOADED.clear()
    finally:
        M1.stamp = orig_stamp
        M1._ACTIVE[0] = None
        GNX.LOADED.clear()
    assert [s[:2] for s in seen] == [(hp, "screen"), (hp, "read")], seen
    e = seen[0][2]
    assert (e["next"]["declared_in"], e["next"]["fit_sha256"], e["base"]) == (DECLARED, "c" * 64, "scr-ml1-zg6r-s0-hp"), e
    assert e["next"]["columns"] == list(COLS) and e["inputs"] == 1 and e["term_parameters"] == N_TERM, e
    assert e["term"].startswith("LTX's cross-fitted") and e["x"] == 1 and set(e["aux"]) >= {"steps", "curve"}, e
    assert "aux" not in seen[1][2] and seen[1][2]["x"] == 2, seen[1]
    assert snapshot(G) == sg and GNX.CUR == {"split": None, "fit_sha256": None, "out": None}
    # 7. the checks a read fit faces, on made stamp blocks: a declared fit gives none; LTG's term, LTM's columns, one fold,
    #    no next-hop step, another rule, other code, a missing or failed folds' check each give theirs; the calls' records
    tmp = Path(tempfile.mkdtemp(prefix="zltx_"))
    try:
        name = fit_name(0, "WQ-musique")
        bse = DEP.base_fit(0, "WQ-musique", ISO.KEEP)
        rule = LXC.rule_sha()

        def block(what, **over):
            need = GNX.train_carves("WQ-musique") if what == "screen" else GNX.eval_carves()
            k = {"term_parameters": N_TERM, "inputs": IN, "columns": list(COLS), "layers": DEP.BASE_LAYERS,
                 "folds": FOLDS, "rank": RANK, "axes": AXES, "lambda": LTG.LAMBDA, "basis_npz_sha256": LTG.BASIS_SHA,
                 "base": bse, "iso_verdict": ISO.KEEP, "device": {"gpu": "t"},
                 "aux": {"steps": 5, "lambda": LTG.LAMBDA, "every": LTG.LOG_EVERY, "curve": []},
                 "next": {"declared_in": DECLARED, "split": "WQ-musique", "columns": list(COLS),
                          "fit_sha256": rule, "carves": {c: "r" for c in need}},
                 "encoder": {"fit": "z", "seed": 0, "sha256": "s", "copies": 1}, "code": {**own_shas(), "a_sha256": "1"}}
            k.update(over)
            return k
        blocks_ = {}

        def fb(nm, what, base=ROOT):
            if nm == bse:
                return {"encoder": {"fit": "z", "seed": 0, "sha256": "s"}, "device": {"gpu": "t"},
                        "code": {"a_sha256": "1"}}
            return blocks_[what]
        cols_ok = {"pass": True, "splits": {"WQ-musique": {"musique/s1eval": {"nan": 0}}}}
        with M1.WQ.patch(GNX, is_read=lambda nm, base=ROOT: nm == name, fit_block=fb):
            for over, n_fit, n_ltx in (({}, 0, 0), ({"term_parameters": LTG.N_TERM}, 2, 1),
                                       ({"columns": list(LTMC.COLS)}, 2, 1), ({"folds": 1}, 2, 1),
                                       ({"aux": {"steps": 0}}, 1, 0),
                                       ({"code": {**own_shas(), "zltx_sha256": "0" * 64, "a_sha256": "1"}}, 0, 1)):
                blocks_.update(screen=block("screen", **over), read=block("read", **over))
                with on_ltx():
                    fp = fit_problems(ISO.KEEP, tmp)
                (tmp / OUT).mkdir(parents=True, exist_ok=True)
                M1.WQ.write_json(tmp / OUT / "cols.json", cols_ok)
                lp = ltx_problems(tmp)
                assert (len(fp), len(lp)) == (n_fit, n_ltx), (over, fp, lp)
            blocks_.update(screen=block("screen"), read=block("read", next={**block("read")["next"], "fit_sha256": "g" * 64}))
            with on_ltx():
                assert len(fit_problems(ISO.KEEP, tmp)) == 1
            assert len(ltx_problems(tmp)) == 1
            nx_other = {**block("read")["next"], "fit_sha256": "g" * 64}
            blocks_.update(screen=block("screen", next=nx_other), read=block("read", next=nx_other))
            assert len(ltx_problems(tmp)) == 2
            nx_gnw = {**block("read")["next"], "declared_in": GNW.DECLARED}
            blocks_.update(screen=block("screen", next=nx_gnw), read=block("read", next=nx_gnw))
            assert len(ltx_problems(tmp)) == 2
            blocks_.update(screen=block("screen"), read=block("read"))
            M1.WQ.write_json(tmp / OUT / "cols.json", {**cols_ok, "pass": False})
            assert len(ltx_problems(tmp)) == 1
            M1.WQ.write_json(tmp / OUT / "cols.json", {"pass": True, "splits": {"WQ-musique": {"x/s1eval": {"nan": 3}}}})
            assert len(ltx_problems(tmp)) == 1
            (tmp / OUT / "cols.json").unlink()
            assert len(ltx_problems(tmp)) == 1
        # the calls on made reads: LTX's s0 read and declared, every read the rule takes above the base's three seeds
        keys = list(DEP.read_keys())
        base3 = {k: {0: 0.50, 1: 0.51, 2: 0.49} for k in keys}
        new = {k: {0: 0.53} for k in keys}
        blocks_.update(screen=block("screen"), read=block("read"))

        def made_calls():
            with M1.WQ.patch(DEP, iso_call=lambda base=ROOT: ISO.KEEP, base_seeds=lambda call, base=ROOT: base3), \
                    M1.WQ.patch(GNX, is_read=lambda nm, base=ROOT: nm == name, fit_block=fb,
                                r5_gnx=lambda call, base=ROOT: (new, {k: {0: v[0]} for k, v in base3.items()}),
                                slices=lambda call, new_r5, b3, base=ROOT: ({}, {}, [])):
                calls(tmp)
            return (json.loads((tmp / OUT / "calls.json").read_text(encoding="utf-8")),
                    (tmp / OUT / "calls.md").read_text(encoding="utf-8"))
        cj, md = made_calls()                                            # no cols.json: INCOMPLETE
        assert cj["stage"] == STAGE and cj["screen"]["verdict"] == INCOMPLETE, cj["screen"]
        assert len(cj["problems"]) == 1 and "cols.json" in cj["problems"][0] and "## Screen: INCOMPLETE" in md, cj
        M1.WQ.write_json(tmp / OUT / "cols.json", cols_ok)
        cj, md = made_calls()
        assert cj["screen"]["verdict"] == ADOPT_FOR_FULL_RUN and not cj["problems"] and cj["full_run"] is None, cj
        assert cj["parameters"] == {name: {k: block("screen").get(k) for k in term_record_keys()}}, cj["parameters"]
        assert cj["aux"] == {name: {"steps": 5, "lambda": LTG.LAMBDA, "last": None}}, cj["aux"]
        assert cj["next_form"] == next_form(cj["screen"], None) and "| split | read |" in md, md
        assert cj["beside"] == {st: None for st in BESIDE}, cj["beside"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert snapshot(G) == sg
    # 8. what follows, by the verdicts
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, None).startswith("the stack takes LTX's cross-fitted term")
    assert next_form({"verdict": NO_GAIN}, None).startswith("the cross-fitted next-hop term closes")
    assert next_form({"verdict": NOT_ADOPTED}, None).startswith("the cross-fitted next-hop term closes")
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, {"verdict": NOT_ADOPTED}).startswith("the cross-fitted")
    assert next_form({"verdict": INCOMPLETE}, None).startswith("what is missing")
    log("selftest: names and ltxcols' constants; the batch's folds (a value other than 0 and 1 and a question of two folds "
        "stop) and the edges between questions; the patches and their restoring (also on a failure); ltxcols' folds load "
        "inside, from ltmcols' root there, and stop outside, LTM's and GNW's stop inside, the NEXT block names LTX's rule; "
        "LTX's toy term (V's draws, zeros; zg6r's forwards bit for bit at the start; each scorer zltg's on a view; the "
        "layers read the other fold's scorer, detached, the next-hop loss the own fold's; equal scorers give LTG's model's "
        "forwards bit for bit; the main loss gives no scorer a gradient; one fold's golds move that fold's scorer alone; "
        "seven stops; checkpointing on and off agree; the term trains in lean_gpu's loop); the stamp's NEXT block LTX's "
        "with the curve on the train alone and its refusal without columns; the read checks on made blocks; the calls on "
        "made reads; what follows")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fit", "compare-all", "calls", "check-cols", "s0stop-check", "s0stop-run",
                                               "gate"))
    ap.add_argument("--name")
    ap.add_argument("--want")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "fit":
        return fit(a.name)
    if a.cmd == "compare-all":
        return compare_all()
    if a.cmd == "calls":
        return calls()
    if a.cmd == "check-cols":
        return check_cols()
    if a.cmd == "s0stop-check":
        with on_s0stop():
            return S0.check(KEY, ROOT)
    if a.cmd == "s0stop-run":
        with on_s0stop():
            return S0.run(KEY, cmd)
    if a.cmd == "gate":
        if not a.want:
            ap.error("gate needs --want")
        with on_s0stop():
            return S0.gate(KEY, [w for w in a.want.split(",") if w])
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
