"""GNW's stage (docs/GNW_PAIR_WEIGHTS_IN_GNN.md): the GNN of record with one more message family in its encoder. Every pool
row's l1 state gains W_m applied to a weighted sum of the l1 states of the first stage's top five rows of its question
(itself left out). The weights are a softmax over those rows of wp * pair_k + wr * res_k + b_k, where pair_k and res_k
are the frozen encoder's question-conditional pair scores of the row against the k-th of the five (PFS A3's two scores,
written by gnwcols.py with the five's places). W_m starts at zero, so a fit starts as the GNN of record bit for bit. On
PKO's roots, on the host's GPU: the screen at seed 0 on both WQ splits against the base's seed-0 fits under LTG's
in-domain rule (docs/LTG_NEXT_HOP_GNN.md section 4), DEP's sixteen-read rule beside.

zgnx.py, zgnp.py, gnwcols.py, gnpcols.py and zpfp.py, and every module they import, are imported and not edited. For the
length of a GNW call zgnx's names (its fits, declaration, stage, stamp key, output root, fit and pair names, code, term
record, NEXT block and model patch) are GNW's, its block width is fifteen columns, and ltmcols' root, stage, declaration,
column names and width are GNW's: zgnx's carve then loads GNW's columns through zltm.load_next and refuses LTM's, LTA's
and GNP's. Everything else in a fit is GNX's: the carve, the base, the start, the trainer and the read. Every patched name
is put back afterwards, also on a failure.

    python outputs/mp_unified/zgnw.py check-cols                   (GNW's columns of every carve both splits load)
    python outputs/mp_unified/zgnw.py fit --name scr-gnw-mw-s0     (train when needed, then read; kept when read before)
    python outputs/mp_unified/zgnw.py compare-all                  (each GNW fit read against the base's fit of its seed)
    python outputs/mp_unified/zgnw.py calls                        (outputs/gnw/calls.json and .md)
    python outputs/mp_unified/zgnw.py s0stop-check                 (s0stop's check under GNW's entry)
    python outputs/mp_unified/zgnw.py s0stop-run -- COMMAND ...    (s0stop's run under GNW's entry)
    python outputs/mp_unified/zgnw.py gate --want ADOPT_FOR_FULL_RUN (s0stop's gate under GNW's entry)
    python outputs/mp_unified/zgnw.py --selftest
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
import zgnp as GNP  # noqa: E402  (GNP's beside reads; not edited)
import gnwcols as GNWC  # noqa: E402  (GNW's columns: their root, names and layout; not edited after its items run)
import s0stop as S0  # noqa: E402  (the seed-0 stop, with GNW's entry patched in for a call; not edited)

M1, RULE, ISO, DEP, LTG, LTM, LTMC = GNX.M1, GNX.RULE, GNX.ISO, GNX.DEP, GNX.LTG, GNX.LTM, GNX.LTMC
PP = GNWC.PP
DECLARED = GNWC.DECLARED
STAGE = GNWC.STAGE
KEY = "gnw"                                     # the stamp's key in a GNW fit's screen.json and read.json and GNW's compares
OUT = "outputs/gnw"
TAG = "mw"
NEXT = GNX.NEXT
TOP = GNWC.TOP
WIDTH = GNWC.WIDTH                              # fifteen columns: pair_k, res_k, anchor_k for the five places
IN = WIDTH                                      # the term reads the columns as they are
HIDDEN = GNX.HIDDEN
J_PAIR, J_RES, J_ANCHOR = GNWC.J_PAIR, GNWC.J_RES, GNWC.J_ANCHOR
TERM_NAMES = ("gnw_wm", "gnw_wp", "gnw_wr", "gnw_b")
TERM_PARAMS = HIDDEN * HIDDEN + 2 + TOP         # W_m (hidden x hidden), wp, wr and the five places' biases
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, ADOPT, INCOMPLETE = (
    GNX.ADOPT_FOR_FULL_RUN, GNX.NO_GAIN, GNX.NOT_ADOPTED, GNX.ADOPT, GNX.INCOMPLETE)
PATCHED = ("FITS", "DECLARED", "STAGE", "KEY", "OUT", "fit_name", "pair_name", "code_extra", "term_record", "next_block",
           "IN", "WIDTH", "on_model")
COLS_PATCHED = ("OUT", "STAGE", "DECLARED", "COLS", "WIDTH")
ORIG = {k: getattr(GNX, k) for k in ("code_extra", "next_block")}
GNX_FITS = dict(GNX.FITS)                       # zgnx's own fit names, taken before any patch
OWN_CODE = ("zgnw", "gnwcols", "gnpcols", "zpfp")
assert (WIDTH, IN, TOP, HIDDEN, TERM_PARAMS, NEXT) == (15, 15, 5, 128, 16391, "NEXT") and Path(GNWC.OUT) == ROOT / OUT


def fit_name(seed, split):
    return f"scr-gnw-{TAG}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    return f"scr-gnw-{TAG}-vs-base-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (GNX.ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 6 and not set(FITS) & (set(M1.ALL_FITS) | set(ISO.FITS) | set(DEP.FITS) | set(LTG.FITS)
                                            | set(GNX.FITS) | set(GNP.FITS))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zgnw: {msg}", flush=True)


def own_shas():
    return {"zgnw_sha256": M1.WQ.sha_src(__file__), "gnwcols_sha256": M1.WQ.sha_src(GNWC.__file__),
            "gnpcols_sha256": M1.WQ.sha_src(GNWC.GNPC.__file__), "zpfp_sha256": M1.WQ.sha_src(PP.__file__)}


def code_extra():
    """GNX's code beside zml1s's (zgnx, zdep, zltg, zltm, ltmcols), and GNW's own: this file, gnwcols.py, gnpcols.py and
    zpfp.py."""
    return {**ORIG["code_extra"](), **own_shas()}


def term_record(G):
    return {"term": "GNW's message weights: each row's l1 state gains W_m applied to the softmax over the first stage's top "
                    "five rows of its question (itself left out) of wp * pair_k + wr * res_k + b_k, weighting those rows' "
                    "l1 states (W_m zero-started)",
            "columns": list(LTMC.COLS), "inputs": IN, "term_parameters": TERM_PARAMS,
            "parameters_h128": int(G.N_PARAMS + TERM_PARAMS), "layers": int(G.LAYERS)}


def next_block(split):
    """GNX's NEXT block with GNW's declaration and the columns' root (inside on_gnw: GNW's)."""
    return {**ORIG["next_block"](split), "declared_in": DECLARED, "columns_root": M1.WQ.rel(Path(LTMC.OUT))}


# ── the term ─────────────────────────────────────────────────────────────────


def message(torch, h, x, nq, B, wp, wr, b):
    """(N, H): each row's message. Over the first stage's top five rows of the row's question (the batch's anchor_k
    columns name them, one row each at most), itself and any place the question lacks left out: the softmax of
    wp * pair_k + wr * res_k + b_k weighting those rows' states h. Zeros for a row with no other anchor. Deterministic:
    index_add_, index_select and bmm only."""
    N, H = h.shape
    dev = h.device
    nq = nq.to(torch.int64)
    A = (x[:, J_ANCHOR:J_ANCHOR + TOP] > 0.5).to(torch.float64)
    cnt = torch.zeros(B, TOP, dtype=torch.float64, device=dev).index_add_(0, nq, A)
    if bool((cnt > 1.5).any()):
        raise RuntimeError("zgnw: a question of the batch holds two rows for one of the first stage's top five")
    row1 = torch.arange(1, N + 1, dtype=torch.float64, device=dev)
    anc = torch.zeros(B, TOP, dtype=torch.float64, device=dev).index_add_(0, nq, A * row1[:, None]).to(torch.int64) - 1
    a_r = anc.index_select(0, nq)                                        # (N, 5): the question's places' rows, -1 none
    ok = (a_r >= 0) & (a_r != torch.arange(N, device=dev)[:, None])
    lg = wp * x[:, J_PAIR:J_PAIR + TOP] + wr * x[:, J_RES:J_RES + TOP] + b
    has = ok.any(1, keepdim=True)
    lg = lg.masked_fill(~ok, float("-inf")).masked_fill(~has, 0.0)
    alpha = (torch.softmax(lg, 1) * ok.to(lg.dtype)).to(h.dtype)
    S = h.index_select(0, anc.clamp(min=0).reshape(-1)).reshape(B, TOP * H)
    if torch.is_grad_enabled() and (alpha.requires_grad or S.requires_grad):
        from torch.utils.checkpoint import checkpoint      # the (N, 5, H) gather is made again in backward, not kept
        return checkpoint(gather_bmm, alpha, S, nq, use_reentrant=False, preserve_rng_state=False)
    return gather_bmm(alpha, S, nq)


def gather_bmm(alpha, S, nq):
    """(N, H): each row's alpha-weighted sum of its question's five places' states (S: (B, 5H), one row per question)."""
    N = alpha.shape[0]
    return alpha.unsqueeze(1).bmm(S.index_select(0, nq).reshape(N, TOP, -1)).squeeze(1)


def message_ref(h, x, nq, B, wp, wr, b):
    """message() by loops in float64 (the self-test's reference)."""
    N, H = h.shape
    anc = -np.ones((B, TOP), np.int64)
    for r in range(N):
        for k in range(TOP):
            if x[r, J_ANCHOR + k] > 0.5:
                assert anc[nq[r], k] < 0
                anc[nq[r], k] = r
    out = np.zeros((N, H))
    for r in range(N):
        ks = [k for k in range(TOP) if anc[nq[r], k] >= 0 and anc[nq[r], k] != r]
        if not ks:
            continue
        lg = np.array([wp * x[r, J_PAIR + k] + wr * x[r, J_RES + k] + b[k] for k in ks])
        w = np.exp(lg - lg.max())
        w /= w.sum()
        out[r] = sum(w[i] * h[anc[nq[r], k]] for i, k in enumerate(ks))
    return out


def l1_hook(G, m, out):
    """l1's output plus W_m applied to each row's message (the batch's NEXT columns set by ZG6R's patched forward)."""
    got = m.gnw_in
    if got is None:
        raise RuntimeError("zgnw: zg6r's l1 ran outside ZG6R's forward (no NEXT input set)")
    x, nq, B = got
    if out.shape[0] != x.shape[0]:
        raise RuntimeError(f"zgnw: l1 ran on {out.shape[0]} rows, the batch's NEXT block holds {x.shape[0]}")
    msg = message(G.torch, out, x, nq, B, m.gnw_wp, m.gnw_wr, m.gnw_b)
    return out + G.Fn.linear(msg, m.gnw_wm.to(out.dtype))


@contextlib.contextmanager
def on_model(G):
    """For the block: a ZG6R built gets W_m (hidden x hidden), wp, wr and the five biases (zeros, no draw) and l1's hook
    after zg6r's own __init__; ZG6R's forward sets the batch's NEXT columns for the hook and clears them after. Restored
    after, also on a failure."""
    torch, nn = G.torch, G.nn
    orig_init, orig_fwd = G.ZG6R.__init__, G.ZG6R.forward

    def init(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        orig_init(self, blocks, widths, hidden, dropout, seed, arm, ctx)
        h = int(self.l1.out_features)
        self.gnw_wm = nn.Parameter(torch.zeros(h, h))
        self.gnw_wp = nn.Parameter(torch.zeros(1))
        self.gnw_wr = nn.Parameter(torch.zeros(1))
        self.gnw_b = nn.Parameter(torch.zeros(TOP))
        self.gnw_in = None
        self.l1.register_forward_hook(lambda _mod, _inp, out, m=self: l1_hook(G, m, out))

    def forward(self, feats, keep, nq, B, base_z):
        x = feats.get(NEXT)
        if x is None or x.shape[1] != WIDTH:
            raise RuntimeError(f"zgnw: the batch holds no {NEXT} block of {WIDTH} columns (GNW's carve adds it)")
        self.gnw_in = (x, nq, int(B))
        try:
            return orig_fwd(self, feats, keep, nq, B, base_z)
        finally:
            self.gnw_in = None

    with M1.WQ.patch(G.ZG6R, __init__=init, forward=forward):
        yield


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def on_cols(out=None):
    """ltmcols' root, stage, declaration, column names and width GNW's (out: GNW's root, or a test's); restored after,
    also on a failure."""
    with M1.WQ.patch(LTMC, OUT=Path(GNWC.OUT if out is None else out), STAGE=STAGE, DECLARED=DECLARED,
                     COLS=GNWC.COLS, WIDTH=WIDTH):
        yield


@contextlib.contextmanager
def on_gnw(out=None):
    """zgnx with GNW's names, code, term record, NEXT block, width and model patch, and ltmcols GNW's (on_cols); restored
    after, also on a failure."""
    with on_cols(out), M1.WQ.patch(GNX, FITS=FITS, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, OUT=OUT,
                                   fit_name=fit_name, pair_name=pair_name, code_extra=code_extra,
                                   term_record=term_record, next_block=next_block, IN=IN, WIDTH=WIDTH,
                                   on_model=on_model):
        yield


@contextlib.contextmanager
def on_s0stop():
    """s0stop's stage table with GNW's entry (zgnw, its root, its two seed-0 fits, LTG's rule, this declaration), in this
    process; restored after."""
    with M1.WQ.patch(S0, STAGES={**S0.STAGES, KEY: ("zgnw", OUT, fit_name(0, S0.S0_SPLIT), fit_name(0, S0.HP_SPLIT),
                                                     "ltg", DECLARED)}):
        yield


def stop_record(base=ROOT):
    """GNW's current s0stop.json (s0stop.stop_record under GNW's entry), else None."""
    with on_s0stop():
        return S0.stop_record(KEY, base)


# ── the commands (zgnx's under GNW's names) ──────────────────────────────────


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zgnw: {name} is not one of GNW's fits ({', '.join(FITS)})")
    if not (ROOT / OUT / "match.json").is_file() or not json.loads(
            (ROOT / OUT / "match.json").read_text(encoding="utf-8")).get("pass"):
        raise SystemExit(f"zgnw: {name}: the columns' match ({OUT}/match.json) is missing or did not pass")
    with on_gnw():
        return GNX.fit(name)


def check_cols():
    with on_gnw():
        return GNX.check_cols()


def compare_all(base=ROOT):
    with on_gnw():
        return GNX.compare_all(base)


def fit_block(name, what, base=ROOT):
    """A GNW fit's stamp block (zgnx's fit_block under GNW's names)."""
    with on_gnw():
        return GNX.fit_block(name, what, base)


def fit_problems(call, base=ROOT):
    """zgnx.fit_problems with GNW's term (TERM_PARAMS parameters on the fifteen columns as they are), inside on_gnw: why a
    GNW fit that is read is not the declared one: its term, base and ISO verdict; its NEXT block (the split, the columns,
    one scorer for train and read, every training carve in its train and every s1eval carve in its read); its start; its
    device; its trainer's code against the base's fit of its seed and split."""
    out = []
    want = {"term_parameters": TERM_PARAMS, "inputs": IN, "columns": list(LTMC.COLS), "layers": DEP.BASE_LAYERS}
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
        if (ks.get("next") or {}).get("fit_sha256") != (kr.get("next") or {}).get("fit_sha256"):
            out.append(f"{name}: its train and read read columns of different scorers")
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


def gnw_problems(base=ROOT):
    """Why a GNW fit that is read did not read GNW's columns by this code: its train's and read's NEXT block must name
    GNW's declaration, its train's term the fifteen columns, and its train's code this zgnw.py, gnwcols.py, gnpcols.py and
    zpfp.py; and the columns' match against gnpcols (outputs/gnw/match.json) must have passed."""
    out, own = [], own_shas()
    with on_gnw():
        for name in FITS:
            if not GNX.is_read(name, base):
                continue
            try:
                ks, kr = GNX.fit_block(name, "screen", base), GNX.fit_block(name, "read", base)
            except SystemExit as e:
                out.append(str(e))
                continue
            for what, k in (("train", ks), ("read", kr)):
                if (k.get("next") or {}).get("declared_in") != DECLARED:
                    out.append(f"{name}: its {what}'s NEXT block names {(k.get('next') or {}).get('declared_in')}, "
                               f"not GNW's columns")
            if (ks.get("inputs"), ks.get("columns"), ks.get("term_parameters")) != (IN, list(GNWC.COLS), TERM_PARAMS):
                out.append(f"{name}: its train's term takes {ks.get('inputs')} inputs of {ks.get('columns')} with "
                           f"{ks.get('term_parameters')} parameters, not GNW's")
            code = ks.get("code") or {}
            diff = sorted(k for k, v in own.items() if code.get(k) != v)
            if diff:
                out.append(f"{name}: its train's code is not this GNW's in {diff}")
        read = any(GNX.is_read(f, base) for f in FITS)
    m = Path(base) / OUT / "match.json"
    if read:
        if not m.is_file():
            out.append(f"{M1.WQ.rel(m)}: the columns' match against gnpcols is missing")
        elif not json.loads(m.read_text(encoding="utf-8")).get("pass"):
            out.append(f"{M1.WQ.rel(m)}: the columns' match against gnpcols did not pass")
    return out


def next_form(screen, full):
    s = (screen or {}).get("verdict")
    v = (full or {}).get("verdict")
    if s == INCOMPLETE or v == INCOMPLETE:
        return "what is missing is fixed in an amendment, before any number is read"
    if s == ADOPT_FOR_FULL_RUN and v in (None, ADOPT):
        return ("the stack takes GNW's message family by an amendment to docs/STK_GNN_STACK.md (GNW is not one of A3's "
                "members); GNW's own full run is held for the stack; next form: learned attention to the top five, the "
                "pair scores as its bias")
    return ("the pair scores as fixed message weights close for the GNN; next form: learned attention to the top five, the "
            "pair scores as its bias")


def calls(base=ROOT):
    """GNW's screen (and its full run, when every fit is read) by LTG's in-domain rule, DEP's sixteen-read rule beside,
    through zgnx's reads, slices and checks under GNW's names; outputs/gnw/calls.json and .md."""
    with on_gnw():
        call = DEP.iso_call(base)
        new_r5, b0 = GNX.r5_gnx(call, base)
        base3 = DEP.base_seeds(call, base)
        bad = [f"{rd[0]} {rd[1]} seed {sd}: the base gives {v} in GNW's compare, {base3.get(rd, {}).get(sd)} in its own "
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
        params = {f: {k: GNX.fit_block(f, "screen", base).get(k) for k in GNX.term_record_keys()}
                  for f in FITS if GNX.is_read(f, base)}
    bad += gnw_problems(base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": f"{GNX.ARM} with GNW's message weights", "iso_verdict": call,
           "base": "zg6r-m (ISO)" if call == ISO.REPLACE else "zg6r (ML1)", "screen": screen, "full_run": full,
           "dep_rule_beside": dep_rule, "next_form": next_form(screen, full),
           "beside": {"pfs_a3": GNP.pfs_beside(base), "gnp": GNP.screen_beside("gnp", base),
                      "gnx": GNP.screen_beside("gnx", base), "gna": GNP.screen_beside("gna", base)},
           "problems": bad, "parameters": params, **own_shas(), "zgnx_sha256": M1.WQ.sha_src(GNX.__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    md = [f"# GNW: {GNX.ARM} with the pair scores as message weights against {rec['base']}, on PKO's pools", "",
          f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}",
          f"DEP's sixteen-read rule beside: screen {dep_rule['screen']}, full run {dep_rule['full_run']}",
          f"next form: {rec['next_form']}", ""]
    rows = screen["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in screen.get("beside", [])]
    if rows:
        md += ["| split | read | base s0, s1, s2 | GNW s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
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
    for st in ("gnp", "gnx", "gna"):
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


def snapshot(G=None):
    s = {**{f"gnx.{k}": getattr(GNX, k) for k in PATCHED}, **{f"ltmcols.{k}": getattr(LTMC, k) for k in COLS_PATCHED},
         **{f"m1.{k}": getattr(M1, k) for k in ("DECLARED", "STAGE", "KEY", "pko_fit_name", "stamp", "code_shas",
                                                 "load_encoder", "train", "read")}, "n_fits": len(M1.ALL_FITS),
         "message": globals()["message"]}
    if G is not None:
        s.update({"g.encoder_used": G.encoder_used, "g.init": G.ZG6R.__init__, "g.forward": G.ZG6R.forward,
                  "g.carve_init": G.G6rLinkCarve.__init__, "g.carve_batch": G.G6rLinkCarve.batch})
    return s


def toy_cols15(out, split, ds, carve, cache_root, rng, fit_sha, training=False):
    """GNW's layout for one toy carve: per question, its first stage's places on min(rows, 5) of its rows drawn at random
    (anchor_k 1 there), pair_k and res_k in [-1, 1] on every other row and 0 on the place's own row and for places the
    question lacks; the part and carve records ltmcols' names give inside on_cols; the columns."""
    import lean_cache as LC
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    cd = LTMC.cols_dir(out, split, ds, carve)
    ps, xs = [], []
    for p, r in zip(parts, recs):
        n = np.load(p / "n.npy").astype(np.int64)
        k = int(r["rows"])
        assert int(n.sum()) == k and n.size == int(r["queries"])
        C = np.zeros((k, WIDTH), np.float16)
        off = np.r_[0, np.cumsum(n)]
        for q in range(n.size):
            s, e = int(off[q]), int(off[q + 1])
            m = min(e - s, TOP)
            C[s:e, J_PAIR:J_PAIR + m] = rng.uniform(-1.0, 1.0, (e - s, m))
            C[s:e, J_RES:J_RES + m] = rng.uniform(-1.0, 1.0, (e - s, m))
            for j, a in enumerate(s + rng.permutation(e - s)[:m]):
                C[a, J_PAIR + j] = C[a, J_RES + j] = 0.0
                C[a, J_ANCHOR + j] = 1.0
            assert GNWC.anchors_ok(C[s:e].astype(np.float64))
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
        xs.append(C)
    M1.WQ.write_json(cd / "record.json", {
        "stage": LTMC.STAGE, "declared_in": LTMC.DECLARED, "split": split, "dataset": ds, "carve": carve,
        "training_carve": training, "rows": int(sum(q["rows"] for q in ps)), "parts": ps, "columns": list(LTMC.COLS),
        "fit_sha256": fit_sha, "identity": {"pass": True}})
    return np.concatenate(xs)


def toy_term(G):
    """GNW on zg6r's own toy carve (GNX's toy: zg6r's self-test steps 2 and 3) with GNW-layout toy columns: the carve's
    columns are GNW's files, float16 on the CPU, and the carve refuses no split, a missing split's columns and another
    scorer's; the batch's NEXT rows are its questions' carve rows (NaN 0); the term's parameters are zeros; at W_m = 0 the
    eval and the training forward are zg6r's with the same parameters bit for bit, W_m gets a gradient and wp, wr and the
    biases none; with the term set, the forward's message is message_ref's on the l1 states and NEXT columns the forward
    hands it, and the term moves the score and gives wp, wr and the biases gradients; l1 outside the forward, a batch
    without NEXT or with a narrower one, and two rows on one place stop; the strict load with and without the patch; a
    one-epoch lean_gpu.fit_variant that moves W_m; the patches' restoring."""
    import torch
    G6 = G.G6
    G.LG.bind_device_ops()
    out = {}
    sp = "WQ-musique"
    with tempfile.TemporaryDirectory(prefix="zgnw_") as td:
        tmp = Path(td)
        rng = np.random.default_rng(41)
        G.RM.toy_roots(tmp, rng, chunks=(12, 12))
        G.RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                   rel_dir=tmp / "rel")
        G.ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        kw = dict(look_root=tmp / "look", cache_root=tmp / "cache", links_root=tmp / "lk", ch_root=tmp / "ch")

        class ToyIds(G6.ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        X0 = toy_cols15(tmp / "gnw", sp, "metaqa", "toy", tmp / "cache", rng, "a" * 64)
        with G.U2N.extra_carves(("metaqa", "toy")):
            G.U2N.build_typed("metaqa", "toy", tmp / "look", tmp / "typed", **kw)
            plain = G.ZL.link_carve(G.RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
            c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            G.Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
            cls = G.g6r_carve(G.Z3.kt_carve(plain, tmp / "kt"), tmp / "typed")
            mk = lambda: cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")  # noqa: E731
            ce0 = mk()
            init0, batch0 = cls.__init__, cls.batch
            X = LTM.load_next("metaqa", "toy", sp, ce0.part_shas, tmp / "gnw")[0]
            out["files_are_toy_cols"] = bool(np.array_equal(X, X0)) and X.dtype == np.float16 and X.shape[1] == WIDTH
            with GNX.on_carve(G, cls):
                for args, what in (((None,), "no_split"), (("WQ-hotpotqa",), "other_split"),
                                   ((sp, "b" * 64), "other_scorer")):
                    try:
                        with GNX.on_split(*args, out=tmp / "gnw") if args[0] else contextlib.nullcontext():
                            mk()
                        out[what] = "passed"
                    except SystemExit:
                        out[what] = "refused"
                GNX.LOADED.clear()
                with GNX.on_split(sp, out=tmp / "gnw"):
                    ce = mk()
                    nx = GNX.next_block(sp)
                out["loaded"] = sorted(nx["carves"])
                out["scorer"] = nx["fit_sha256"]
                out["next_block_gnw"] = (nx["declared_in"], nx["columns"]) == (DECLARED, list(GNWC.COLS))
                out["carve_cols_are_files"] = bool(torch.equal(ce.gnx_X, torch.from_numpy(X)))
                out["carve_cols_on_cpu"] = ce.gnx_X.device.type == "cpu" and ce.gnx_X.dtype == torch.float16
                ce.gnx_X = ce.gnx_X.clone()            # NaN in a carve's columns is 0 in the batch (a NaN anchor: none)
                ce.gnx_X[torch.from_numpy(rng.random(tuple(ce.gnx_X.shape)) < 0.05)] = float("nan")
                out["nan_set"] = int(torch.isnan(ce.gnx_X).sum())
                blocks = ["rank", "SEMB"]
                qs = np.r_[np.arange(ce.rows)[::-1][:7], np.arange(ce.rows)[:5]].astype(np.int64)
                B = qs.size
                feats, nq, bz, gold = ce.batch(qs, blocks)
            out["carve_restored"] = cls.__init__ is init0 and cls.batch is batch0
            rows = np.concatenate([np.arange(ce.off_np[q], ce.off_np[q + 1]) for q in qs])
            out["rank_rows_are_carve_rows"] = bool(torch.equal(feats["rank"], ce.X[torch.from_numpy(rows)]))
            want = torch.nan_to_num(ce.gnx_X[torch.from_numpy(rows)].to(torch.float32), nan=0.0)
            out["next_rows_are_carve_rows"] = bool(torch.equal(feats[NEXT], want))
            out["next_nan_free"] = bool(torch.isfinite(feats[NEXT]).all())
            keep = torch.ones(B, len(blocks))
            orig_init, orig_fwd = G.ZG6R.__init__, G.ZG6R.forward
            names0 = tuple(G6.G6_NAMES) + tuple(G.G6R_NAMES)
            gen = torch.Generator().manual_seed(3)
            p0 = G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)    # zg6r's own: built outside the patch, no term
            out["plain_has_term"] = any(hasattr(p0, k) for k in TERM_NAMES)
            with on_model(G):
                torch.manual_seed(0)
                m = G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)
                out["term_shapes"] = {k: tuple(getattr(m, k).shape) for k in TERM_NAMES}
                out["term_zero"] = all(bool((getattr(m, k) == 0).all()) for k in TERM_NAMES)
                out["l1_out"] = int(m.l1.out_features)
                out["term_params"] = sum(int(getattr(m, k).numel()) for k in TERM_NAMES)
                with torch.no_grad():
                    for k in names0:
                        getattr(m, k).copy_(torch.randn(getattr(m, k).shape, generator=gen) * 0.3)
                p0.load_state_dict({k: v for k, v in m.state_dict().items() if k not in TERM_NAMES}, strict=True)
                m.eval()
                p0.eval()
                with torch.no_grad():
                    out["eval_start_is_zg6r"] = bool(torch.equal(m(feats, keep, nq, B, bz), p0(feats, keep, nq, B, bz)))
                m.train()
                p0.train()
                p0.g6_gen.set_state(m.g6_gen.get_state())
                torch.manual_seed(9)
                y1 = m(feats, keep, nq, B, bz)
                torch.manual_seed(9)
                y2 = p0(feats, keep, nq, B, bz)
                out["train_start_is_zg6r"] = bool(torch.equal(y1, y2))
                m.zero_grad()
                G.LG.listwiseD(y1, gold, nq, B).backward()
                gr = {k: float(getattr(m, k).grad.abs().sum()) if getattr(m, k).grad is not None else 0.0
                      for k in TERM_NAMES}
                out["wm_grad"] = gr["gnw_wm"]
                out["gate_grads_at_zero"] = gr["gnw_wp"] + gr["gnw_wr"] + gr["gnw_b"]
                with torch.no_grad():
                    m.gnw_wm.copy_(torch.randn(m.gnw_wm.shape, generator=gen) * 0.3)
                    m.gnw_wp.fill_(1.5)
                    m.gnw_wr.fill_(-0.7)
                    m.gnw_b.copy_(torch.randn(TOP, generator=gen))
                seen, msg0 = [], globals()["message"]

                def spy(torch_, h, x, nq_, B_, wp, wr, b):
                    r = msg0(torch_, h, x, nq_, B_, wp, wr, b)
                    seen.append((h.detach().clone(), x.detach().clone(), nq_.clone(), B_, r.detach().clone()))
                    return r
                m.eval()
                try:
                    globals()["message"] = spy
                    with torch.no_grad():
                        y3 = m(feats, keep, nq, B, bz)
                finally:
                    globals()["message"] = msg0
                with torch.no_grad():
                    out["term_moves_score"] = not bool(torch.equal(y3, p0(feats, keep, nq, B, bz)))
                out["message_calls"] = len(seen)
                h, x, nq_, B_, r = seen[0]
                out["message_x_is_next"] = bool(torch.equal(x, feats[NEXT])) and bool(torch.equal(nq_, nq)) and B_ == B
                ref = message_ref(h.double().numpy(), x.double().numpy(), nq.numpy(), B, 1.5, -0.7,
                                  m.gnw_b.detach().double().numpy())
                out["message_max_abs"] = float(np.abs(r.double().numpy() - ref).max())
                out["message_rows"] = int((np.abs(ref).sum(1) > 0).sum())
                m.train()
                m.zero_grad()
                torch.manual_seed(9)
                G.LG.listwiseD(m(feats, keep, nq, B, bz), gold, nq, B).backward()
                out["gate_grads_set"] = min(float(getattr(m, k).grad.abs().sum()) for k in ("gnw_wp", "gnw_wr", "gnw_b"))
                m.eval()
                with torch.no_grad():
                    out["gnw_in_cleared"] = m.gnw_in is None
                    for what, f in (("no_next", {k: v for k, v in feats.items() if k != NEXT}),
                                    ("narrow_next", {**feats, NEXT: feats[NEXT][:, :7]})):
                        try:
                            m(f, keep, nq, B, bz)
                            out[what] = "passed"
                        except RuntimeError:
                            out[what] = "refused"
                    try:
                        m.encode(feats, keep, nq, B)
                        out["l1_outside"] = "passed"
                    except RuntimeError:
                        out["l1_outside"] = "refused"
                    a1 = feats[NEXT][:, J_ANCHOR] > 0.5
                    b0 = next(q for q in range(B) if int(a1[nq == q].sum()) == 1 and int((nq == q).sum()) > 1)
                    x2 = feats[NEXT].clone()
                    x2[int(((nq == b0) & ~a1).nonzero().flatten()[0]), J_ANCHOR] = 1.0
                    try:
                        m({**feats, NEXT: x2}, keep, nq, B, bz)
                        out["two_anchors"] = "passed"
                    except RuntimeError:
                        out["two_anchors"] = "refused"
                G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0).load_state_dict(m.state_dict(), strict=True)
                out["strict_load_patched"] = "passed"
            out["model_restored"] = G.ZG6R.__init__ is orig_init and G.ZG6R.forward is orig_fwd
            try:
                G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0).load_state_dict(m.state_dict(), strict=True)
                out["strict_load_plain"] = "passed"
            except RuntimeError:
                out["strict_load_plain"] = "refused"
            # a one-epoch lean_gpu.fit_variant on the patched carve and model: the term trains inside the trainer's own loop
            cfg = {"lr": 2e-3, "wd": 1e-4, "dropout": 0.1, "epochs": 1, "swa_from": 0, "cos": 0, "adamw": 0, "drop": 0}
            with GNX.on_carve(G, cls), on_model(G), M1.WQ.patch(G.LG, LeanMLP8D=G.ZG6R):
                res = G.LG.fit_variant([ce], blocks, cfg, 0, 16, "none", "cpu")
            out["fit_wm_moved"] = float(res["swa"]["gnw_wm"].abs().sum()) > 0
            out["fit_loss_finite"] = all(math.isfinite(c["loss"]) for c in res["curve"])
            out["fit_restored"] = G.LG.LeanMLP8D is not G.ZG6R and G.ZG6R.__init__ is orig_init and cls.batch is batch0
        GNX.LOADED.clear()
    return out


def selftest():
    import torch
    must_stop = GNP.must_stop
    # 1. names: GNW's own, none another stage's
    assert fit_name(0, "WQ-musique") == "scr-gnw-mw-s0" and fit_name(2, "WQ-hotpotqa") == "scr-gnw-mw-s2-hp"
    assert pair_name(1, "WQ-hotpotqa") == "scr-gnw-mw-vs-base-s1-hp"
    import zrtr as RTR
    import zira as IRA
    import zisb as ISB
    import zgna as GNA
    import zlta as ZLA
    assert not set(FITS) & (set(RTR.FITS) | set(IRA.FITS) | set(ISB.FITS) | set(LTM.FITS) | set(GNA.FITS)
                            | set(ZLA.FITS) | set(GNP.FITS))
    # 2. the message by itself, against loops: questions of one to nine rows, rows out of question order, a place a
    #    question lacks, two rows on one place stop; a two-row question's rows each take the other's state whole
    rng = np.random.default_rng(5)
    sizes = [1, 2, 3, 5, 6, 9]
    B, N, H = len(sizes), sum(sizes), 8
    nq = np.repeat(np.arange(B), sizes)
    off = np.r_[0, np.cumsum(sizes)]
    x = np.zeros((N, WIDTH))
    x[:, :J_ANCHOR] = rng.uniform(-1.0, 1.0, (N, J_ANCHOR))
    for q, n in enumerate(sizes):
        for j, a in enumerate(off[q] + rng.permutation(n)[:min(n, TOP)]):
            x[a, J_PAIR + j] = x[a, J_RES + j] = 0.0
            x[a, J_ANCHOR + j] = 1.0
    x[(nq == 5) & (x[:, J_ANCHOR + 2] > 0.5), J_ANCHOR + 2] = 0.0          # the nine-row question lacks its third place
    h = rng.standard_normal((N, H))
    perm = rng.permutation(N)
    h, x, nq = h[perm], x[perm], nq[perm]
    wp, wr, b = 2.0, -1.3, rng.standard_normal(TOP)
    T = lambda a: torch.tensor(a, dtype=torch.float64)                    # noqa: E731
    with torch.no_grad():
        got = message(torch, T(h), T(x), torch.tensor(nq), B, T([wp]), T([wr]), T(b)).numpy()
    ref = message_ref(h, x, nq, B, wp, wr, b)
    assert got.shape == (N, H) and np.abs(got - ref).max() < 1e-12, np.abs(got - ref).max()
    #    with gradients: the checkpointed gather gives the same values and the same gradients as the plain one
    import torch.utils.checkpoint as TUC
    up = T(np.random.default_rng(9).standard_normal((N, H)))
    grads = []
    for ck in (True, False):
        ins = [T(h).requires_grad_(), T([wp]).requires_grad_(), T([wr]).requires_grad_(), T(b).requires_grad_()]
        with contextlib.ExitStack() as es:
            if not ck:
                es.enter_context(M1.WQ.patch(TUC, checkpoint=lambda fn, *a, **k: fn(*a)))
            out = message(torch, ins[0], T(x), torch.tensor(nq), B, *ins[1:])
        assert np.array_equal(out.detach().numpy(), got)
        (out * up).sum().backward()
        grads.append([t.grad.numpy().copy() for t in ins])
    assert all(np.array_equal(a, c) for a, c in zip(*grads)) and all(np.abs(g).sum() > 0 for g in grads[0])
    assert not got[nq == 0].any()                                         # one row: no other anchor
    r2 = np.flatnonzero(nq == 1)
    assert np.array_equal(got[r2[0]], h[r2[1]]) and np.array_equal(got[r2[1]], h[r2[0]])
    assert np.abs(got[nq == 5]).sum() > 0
    x2 = x.copy()
    x2[np.flatnonzero((nq == 4) & (x[:, J_ANCHOR] < 0.5))[0], J_ANCHOR] = 1.0
    must_stop_rt = False
    try:
        message(torch, T(h), T(x2), torch.tensor(nq), B, T([wp]), T([wr]), T(b))
    except RuntimeError:
        must_stop_rt = True
    assert must_stop_rt
    # 3. the patches: zgnx's names, code, width and model patch GNW's and ltmcols' root, stage, declaration, columns and
    #    width GNW's inside; restored after, also on a failure; zgnx's own names, pairs and code under them
    s0 = snapshot()
    with on_gnw():
        assert GNX.FITS is FITS and (GNX.DECLARED, GNX.STAGE, GNX.KEY, GNX.OUT) == (DECLARED, STAGE, KEY, OUT)
        assert (GNX.IN, GNX.WIDTH, LTMC.WIDTH, tuple(LTMC.COLS)) == (15, 15, 15, GNWC.COLS) and GNX.on_model is on_model
        assert GNX.pairs(ISO.KEEP)[pair_name(0, "WQ-musique")] == ("scr-gnw-mw-s0", "scr-ml1-zg6r-s0")
        assert GNX.pairs(ISO.REPLACE)[pair_name(0, "WQ-hotpotqa")] == ("scr-gnw-mw-s0-hp", "scr-iso-zg6rm-s0-hp")
        assert (Path(LTMC.OUT), LTMC.STAGE, LTMC.DECLARED) == (Path(GNWC.OUT), "GNW", "docs/GNW_PAIR_WEIGHTS_IN_GNN.md")
        cs = GNX.code_extra()
        assert all(f"{k}_sha256" in cs for k in OWN_CODE + ("zgnx", "zdep", "zltg", "zltm", "ltmcols")), cs
        with GNX.on_names():
            assert (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY) and set(FITS) <= set(M1.ALL_FITS)
            assert not set(GNX_FITS) & set(M1.ALL_FITS)                  # zgnx's table is GNW's here
            assert all(f"{k}_sha256" in M1.code_shas(GNX.ARM) for k in OWN_CODE)
    assert snapshot() == s0 and (GNX.IN, LTMC.WIDTH, LTMC.STAGE) == (15, 7, "LTM") and GNX.on_model is not on_model
    try:
        with on_gnw():
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert snapshot() == s0
    must_stop(fit, "scr-gnx-nx-s0")                                      # GNX's own fit is not GNW's
    must_stop(fit, "scr-gnp-px-s0")                                      # nor GNP's
    st0 = dict(S0.STAGES)
    with on_s0stop():
        assert S0.STAGES[KEY] == ("zgnw", OUT, "scr-gnw-mw-s0", "scr-gnw-mw-s0-hp", "ltg", DECLARED)
        assert S0.STAGES["gnx"] == st0["gnx"] and S0.deciding("ltg") == S0.deciding(S0.STAGES["gnx"][4])
    assert S0.STAGES == st0 and KEY not in S0.STAGES
    tmp = Path(tempfile.mkdtemp(prefix="zgnw_"))
    try:                                         # GNW's stop record on made files: current only for the read it was made on
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
    # 4. the columns a GNW carve loads are gnwcols': its columns load inside on_gnw, from ltmcols' root there, and stop
    #    outside it; LTM-stamped and GNP-stamped columns stop inside it; the NEXT block names GNW's declaration and root
    import lean_cache as LC
    import ltwdiag as LTW
    tmp = Path(tempfile.mkdtemp(prefix="zgnw_"))
    try:
        rng = np.random.default_rng(29)
        M = LTW.unit(rng.normal(size=(200, 16))).astype(np.float32)
        LTW.toy_tree(tmp, "musique", "s1eval", M, rng, nq=6, n=20)
        parts, _recs = LC.part_dirs("musique", "s1eval", tmp / "cache")
        ps = {p.name: LC.sha_file(p / "record.json") for p in parts}
        sp = "WQ-hotpotqa"
        rec = GNWC.cols("musique", "s1eval", tmp, tmp / "gnw", {"musique": PP.FakeNodes(M)})
        GNWC.GNPC.cols("musique", "s1eval", tmp, tmp / "gnp", {"musique": PP.FakeNodes(M)})
        with on_gnw(tmp / "gnw"):
            X2, r2, _sha = LTM.load_next("musique", "s1eval", sp, ps)      # out None: ltmcols' OUT, GNW's here
            assert X2.shape == (rec["rows"], 15) and (r2["stage"], r2["declared_in"]) == ("GNW", DECLARED), r2
            with GNX.on_split(sp, rec["fit_sha256"]):
                GNX.LOADED.append({"dataset": "musique", "carve": "s1eval", "record_sha256": "r"})
                nb = GNX.next_block(sp)
            GNX.LOADED.clear()
            assert (nb["declared_in"], Path(nb["columns_root"]).name, nb["fit_sha256"], nb["columns"]) == (
                DECLARED, "gnw", GNWC.rule_sha(), list(GNWC.COLS)), nb
            must_stop(LTM.load_next, "musique", "s1eval", sp, ps, tmp / "gnp")   # GNP's columns inside
        Y = LTM.toy_cols(tmp / "ltm", sp, "musique", "s1eval", tmp / "cache", rng, "e" * 64)
        assert LTM.load_next("musique", "s1eval", sp, ps, tmp / "ltm")[1]["stage"] == "LTM" and Y.shape[1] == 7
        must_stop(LTM.load_next, "musique", "s1eval", sp, ps, tmp / "gnw")  # GNW's columns outside on_gnw
        with on_gnw(tmp / "gnw"):
            must_stop(LTM.load_next, "musique", "s1eval", sp, ps, tmp / "ltm")   # LTM's columns inside
        assert snapshot() == s0
    finally:
        GNX.LOADED.clear()
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. GNW's toy term under GNW's patches: zg6r's toy carve loads GNW-stamped columns, the term is zeros, at W_m = 0 the
    #    forwards are zg6r's bit for bit, the forward's message is message_ref's, the term trains in lean_gpu's loop
    G = GNX.import_g()
    sg = snapshot(G)
    with on_gnw():
        t = toy_term(G)
        rec = GNX.term_record(G)
    assert (t["no_split"], t["other_split"], t["other_scorer"]) == ("refused",) * 3, t
    assert t["loaded"] == ["metaqa/toy"] and t["scorer"] == "a" * 64 and t["carve_restored"] and t["next_block_gnw"], t
    assert t["files_are_toy_cols"] and t["carve_cols_are_files"] and t["carve_cols_on_cpu"] and t["nan_set"] > 0, t
    assert t["rank_rows_are_carve_rows"] and t["next_rows_are_carve_rows"] and t["next_nan_free"], t
    assert t["term_zero"] and t["l1_out"] == 16 and t["term_params"] == 16 * 16 + 2 + TOP and not t["plain_has_term"], t
    assert t["term_shapes"] == {"gnw_wm": (16, 16), "gnw_wp": (1,), "gnw_wr": (1,), "gnw_b": (TOP,)}, t
    assert t["eval_start_is_zg6r"] and t["train_start_is_zg6r"] and t["wm_grad"] > 0 and t["gate_grads_at_zero"] == 0, t
    assert t["term_moves_score"] and t["message_calls"] == 1 and t["message_x_is_next"], t
    assert t["message_max_abs"] < 1e-5 and t["message_rows"] > 0 and t["gate_grads_set"] > 0, t
    assert (t["no_next"], t["narrow_next"], t["l1_outside"], t["two_anchors"]) == ("refused",) * 4, t
    assert t["gnw_in_cleared"] and t["model_restored"], t
    assert t["strict_load_patched"] == "passed" and t["strict_load_plain"] == "refused", t
    assert t["fit_wm_moved"] and t["fit_loss_finite"] and t["fit_restored"], t
    assert rec["term"].startswith("GNW's message weights") and rec["term_parameters"] == 128 * 128 + 7 == 16391, rec
    assert rec["inputs"] == 15 and rec["parameters_h128"] == 180218 + 16391 == 196609, rec
    assert rec["columns"] == list(GNWC.COLS) and set(rec) == set(GNX.term_record_keys()) | {"term"}, rec
    assert snapshot(G) == sg
    # 6. the stamp under GNW's names: the train's NEXT block is GNW's, with GNW's term; one without the columns stops
    seen, orig_stamp = [], M1.stamp
    hp = fit_name(0, "WQ-hotpotqa")
    try:
        M1.stamp = lambda name, env, what, extra=None: seen.append((name, what, dict(extra or {})))
        with on_gnw(), GNX.on_gnx(G, ISO.KEEP):
            assert (M1.STAGE, M1.KEY) == (STAGE, KEY) and set(FITS) <= set(M1.ALL_FITS)
            assert G.ZG6R.__init__ is not sg["g.init"]
            with GNX.on_split("WQ-hotpotqa"):
                must_stop(M1.stamp, hp, {}, "screen", {"x": 1})
            with GNX.on_split("WQ-hotpotqa", "c" * 64):
                GNX.LOADED.extend({"dataset": k.split("/")[0], "carve": k.split("/")[1], "record_sha256": "r"}
                                  for k in GNX.train_carves("WQ-hotpotqa"))
                M1.stamp(hp, {}, "screen", {"x": 1})
            GNX.LOADED.clear()
    finally:
        M1.stamp = orig_stamp
        M1._ACTIVE[0] = None
        GNX.LOADED.clear()
    assert [s[:2] for s in seen] == [(hp, "screen")], seen
    e = seen[0][2]
    assert (e["next"]["declared_in"], e["next"]["fit_sha256"], e["base"]) == (DECLARED, "c" * 64, "scr-ml1-zg6r-s0-hp"), e
    assert e["next"]["columns"] == list(GNWC.COLS) and e["inputs"] == 15 and e["term_parameters"] == 16391, e
    assert e["term"].startswith("GNW's message weights") and e["x"] == 1, e
    assert snapshot(G) == sg and GNX.CUR == {"split": None, "fit_sha256": None, "out": None}
    # 7. the checks a read fit faces, on made stamp blocks: a declared fit gives none; GNX's term, another scorer, GNP's
    #    columns, other code or a missing or failed match each give one; the calls' records on made reads
    tmp = Path(tempfile.mkdtemp(prefix="zgnw_"))
    try:
        name = fit_name(0, "WQ-musique")
        bse = DEP.base_fit(0, "WQ-musique", ISO.KEEP)

        def block(what, **over):
            need = GNX.train_carves("WQ-musique") if what == "screen" else GNX.eval_carves()
            k = {"term_parameters": TERM_PARAMS, "inputs": IN, "columns": list(GNWC.COLS), "layers": DEP.BASE_LAYERS,
                 "base": bse, "iso_verdict": ISO.KEEP, "device": {"gpu": "t"},
                 "next": {"declared_in": DECLARED, "split": "WQ-musique", "columns": list(GNWC.COLS),
                          "fit_sha256": "f" * 64, "carves": {c: "r" for c in need}},
                 "encoder": {"fit": "z", "seed": 0, "sha256": "s", "copies": 1}, "code": {**own_shas(), "a_sha256": "1"}}
            k.update(over)
            return k
        blocks_ = {}

        def fb(nm, what, base=ROOT):
            if nm == bse:
                return {"encoder": {"fit": "z", "seed": 0, "sha256": "s"}, "device": {"gpu": "t"},
                        "code": {"a_sha256": "1"}}
            return blocks_[what]
        with M1.WQ.patch(GNX, is_read=lambda nm, base=ROOT: nm == name, fit_block=fb):
            for over, n_fit, n_gnw in (({}, 0, 0), ({"term_parameters": 1920}, 2, 1),
                                       ({"columns": list(GNWC.GNPC.COLS)}, 2, 1),
                                       ({"code": {**own_shas(), "zgnw_sha256": "0" * 64, "a_sha256": "1"}}, 0, 1)):
                blocks_.update(screen=block("screen", **over), read=block("read", **over))
                with on_gnw():
                    fp = fit_problems(ISO.KEEP, tmp)
                (tmp / OUT).mkdir(parents=True, exist_ok=True)
                M1.WQ.write_json(tmp / OUT / "match.json", {"pass": True})
                gp = gnw_problems(tmp)
                assert (len(fp), len(gp)) == (n_fit, n_gnw), (over, fp, gp)
            blocks_.update(screen=block("screen"), read=block("read", next={**block("read")["next"], "fit_sha256": "g" * 64}))
            with on_gnw():
                assert len(fit_problems(ISO.KEEP, tmp)) == 1
            blocks_.update(screen=block("screen"), read=block("read"))
            M1.WQ.write_json(tmp / OUT / "match.json", {"pass": False})
            assert len(gnw_problems(tmp)) == 1
            (tmp / OUT / "match.json").unlink()
            assert len(gnw_problems(tmp)) == 1
        # the calls on made reads: GNW's s0 read and declared, every read the rule takes above the base's three seeds
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
        cj, md = made_calls()                                            # no match.json: INCOMPLETE
        assert cj["stage"] == STAGE and cj["screen"]["verdict"] == INCOMPLETE, cj["screen"]
        assert len(cj["problems"]) == 1 and "match" in cj["problems"][0] and "## Screen: INCOMPLETE" in md, cj["problems"]
        M1.WQ.write_json(tmp / OUT / "match.json", {"pass": True})
        cj, md = made_calls()
        assert cj["screen"]["verdict"] == ADOPT_FOR_FULL_RUN and not cj["problems"] and cj["full_run"] is None, cj
        assert cj["parameters"] == {name: {k: block("screen").get(k) for k in GNX.term_record_keys()}}, cj["parameters"]
        assert cj["next_form"] == next_form(cj["screen"], None) and "| split | read |" in md and cj["beside"]["gnp"] is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert snapshot(G) == sg
    # 8. what follows, by the verdicts
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, None).startswith("the stack takes GNW's message family")
    assert next_form({"verdict": NO_GAIN}, None).startswith("the pair scores as fixed message weights close")
    assert next_form({"verdict": NOT_ADOPTED}, None).startswith("the pair scores as fixed message weights close")
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, {"verdict": NOT_ADOPTED}).startswith("the pair scores")
    assert next_form({"verdict": INCOMPLETE}, None).startswith("what is missing")
    log("selftest: names; the message against loops (one to nine rows, rows out of order, a missing place, two rows on "
        "one place stop); the patches and their restoring (also on a failure); gnwcols' columns load inside, from "
        "ltmcols' root there, and stop outside, LTM's and GNP's stop inside, the NEXT block names GNW's; GNW's toy term "
        "(zeros; zg6r's forwards bit for bit at W_m = 0; W_m's gradient alone there; the forward's message is the loops'; "
        "the term moves the score; no NEXT, a narrower NEXT, l1 outside and two anchors stop; the term trains in "
        "lean_gpu's loop); the stamp's NEXT block GNW's with GNW's term and its refusal without columns; the read "
        "checks on made blocks; the calls on made reads; what follows")
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
