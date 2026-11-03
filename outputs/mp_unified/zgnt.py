"""GNT's stage (docs/GNT_LEARNED_ATTENTION_GNN.md): the GNN of record with learned attention to the first stage's top five
rows in its encoder. Every pool row's l1 state gains W_m applied to a weighted sum of the l1 states of the first stage's
top five rows of its question (itself left out). The weights are a softmax over those rows of u_c W_a u_k + wp * pair_k
+ wr * res_k + b_k, where u is an l1 state over its norm and pair_k and res_k are the frozen encoder's question-conditional
pair scores of the row against the k-th of the five: GNW's columns, read as they are. W_m and W_a start at zero, so a fit
starts as the GNN of record bit for bit. On PKO's roots, on the host's GPU: the screen at seed 0 on both WQ splits
against the base's seed-0 fits under LTG's in-domain rule (docs/LTG_NEXT_HOP_GNN.md section 4), DEP's sixteen-read rule
beside.

zgnw.py, zgnx.py, zgnp.py, gnwcols.py, gnpcols.py and zpfp.py, and every module they import, are imported and not edited.
For the length of a GNT call zgnx's names (its fits, declaration, stage, stamp key, output root, fit and pair names, code,
term record, NEXT block and model patch) are GNT's, its block width is fifteen columns, and ltmcols' root, stage,
declaration, column names and width are GNW's (zgnw.on_cols): zgnx's carve then loads GNW's columns through
zltm.load_next and refuses LTM's, LTA's and GNP's. Everything else in a fit is GNX's: the carve, the base, the start, the
trainer and the read. Every patched name is put back afterwards, also on a failure.

    python outputs/mp_unified/zgnt.py fit --name scr-gnt-at-s0     (train when needed, then read; kept when read before)
    python outputs/mp_unified/zgnt.py compare-all                  (each GNT fit read against the base's fit of its seed)
    python outputs/mp_unified/zgnt.py calls                        (outputs/gnt/calls.json and .md)
    python outputs/mp_unified/zgnt.py s0stop-check                 (s0stop's check under GNT's entry)
    python outputs/mp_unified/zgnt.py s0stop-run -- COMMAND ...    (s0stop's run under GNT's entry)
    python outputs/mp_unified/zgnt.py gate --want ADOPT_FOR_FULL_RUN (s0stop's gate under GNT's entry)
    python outputs/mp_unified/zgnt.py --selftest
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
import zgnw as GNW  # noqa: E402  (GNW's column names, places, fit checks and toy columns; not edited)
import zgnx as GNX  # noqa: E402  (GNX's arm, carve, stamp, compares and the pieces of its calls; not edited)
import zgnp as GNP  # noqa: E402  (GNP's beside reads; not edited)
import gnwcols as GNWC  # noqa: E402  (GNW's columns: their root, names and layout; not edited)
import s0stop as S0  # noqa: E402  (the seed-0 stop, with GNT's entry patched in for a call; not edited)

M1, RULE, ISO, DEP, LTG, LTM, LTMC = GNX.M1, GNX.RULE, GNX.ISO, GNX.DEP, GNX.LTG, GNX.LTM, GNX.LTMC
DECLARED = "docs/GNT_LEARNED_ATTENTION_GNN.md"
STAGE = "GNT"
KEY = "gnt"                                     # the stamp's key in a GNT fit's screen.json and read.json and GNT's compares
OUT = "outputs/gnt"
TAG = "at"
COLS_ROOT = GNW.OUT                             # GNW's root: its columns, match.json and cols.json, read only
NEXT = GNX.NEXT
TOP = GNWC.TOP
WIDTH = GNWC.WIDTH                              # GNW's fifteen columns: pair_k, res_k, anchor_k for the five places
IN = WIDTH
HIDDEN = GNX.HIDDEN
EPS = 1e-6                                      # a state's norm below this reads as this (a zero state: u = 0)
J_PAIR, J_RES, J_ANCHOR = GNWC.J_PAIR, GNWC.J_RES, GNWC.J_ANCHOR
TERM_NAMES = ("gnt_wm", "gnt_wa", "gnt_wp", "gnt_wr", "gnt_b")
TERM_PARAMS = 2 * HIDDEN * HIDDEN + 2 + TOP     # W_m and W_a (hidden x hidden), wp, wr and the five places' biases
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, ADOPT, INCOMPLETE = (
    GNX.ADOPT_FOR_FULL_RUN, GNX.NO_GAIN, GNX.NOT_ADOPTED, GNX.ADOPT, GNX.INCOMPLETE)
PATCHED = GNW.PATCHED
COLS_PATCHED = GNW.COLS_PATCHED
ORIG = {k: getattr(GNX, k) for k in ("code_extra", "next_block")}
GNX_FITS = dict(GNX.FITS)                       # zgnx's own fit names, taken before any patch
OWN_CODE = ("zgnt",) + GNW.OWN_CODE             # this file, then GNW's: zgnw, gnwcols, gnpcols, zpfp
assert (WIDTH, IN, TOP, HIDDEN, TERM_PARAMS, NEXT) == (15, 15, 5, 128, 32775, "NEXT") and Path(GNWC.OUT) == ROOT / COLS_ROOT
assert (J_PAIR, J_RES, J_ANCHOR) == (0, TOP, 2 * TOP) and ORIG["next_block"] is GNW.ORIG["next_block"]


def fit_name(seed, split):
    return f"scr-gnt-{TAG}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    return f"scr-gnt-{TAG}-vs-base-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (GNX.ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 6 and not set(FITS) & (set(M1.ALL_FITS) | set(ISO.FITS) | set(DEP.FITS) | set(LTG.FITS)
                                            | set(GNX.FITS) | set(GNP.FITS) | set(GNW.FITS))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zgnt: {msg}", flush=True)


def own_shas():
    return {"zgnt_sha256": M1.WQ.sha_src(__file__), **GNW.own_shas()}


def code_extra():
    """GNX's code beside zml1s's (zgnx, zdep, zltg, zltm, ltmcols), and GNT's own: this file, zgnw.py, gnwcols.py,
    gnpcols.py and zpfp.py."""
    return {**ORIG["code_extra"](), **own_shas()}


def term_record(G):
    return {"term": "GNT's learned attention: each row's l1 state gains W_m applied to the softmax over the first stage's "
                    "top five rows of its question (itself left out) of u_c W_a u_k + wp * pair_k + wr * res_k + b_k "
                    "(u: l1 states over their norms), weighting those rows' l1 states (W_m and W_a zero-started)",
            "columns": list(LTMC.COLS), "inputs": IN, "term_parameters": TERM_PARAMS,
            "parameters_h128": int(G.N_PARAMS + TERM_PARAMS), "layers": int(G.LAYERS)}


def next_block(split):
    """GNX's NEXT block with GNT's declaration, and the columns' root and declaration (inside on_gnt: GNW's)."""
    return {**ORIG["next_block"](split), "declared_in": DECLARED, "columns_root": M1.WQ.rel(Path(LTMC.OUT)),
            "columns_declared_in": LTMC.DECLARED}


# ── the term ─────────────────────────────────────────────────────────────────


def message(torch, h, x, nq, B, wa, wp, wr, b):
    """(N, H): each row's message. Over the first stage's top five rows of the row's question (the batch's anchor_k
    columns name them, one row each at most), itself and any place the question lacks left out: the softmax of
    u_c W_a u_k + wp * pair_k + wr * res_k + b_k weighting those rows' states h. Zeros for a row with no other anchor.
    The places' rows as zgnw.message finds them; attend() computes the rest, recomputed in backward when training."""
    N, H = h.shape
    dev = h.device
    nq = nq.to(torch.int64)
    A = (x[:, J_ANCHOR:J_ANCHOR + TOP] > 0.5).to(torch.float64)
    cnt = torch.zeros(B, TOP, dtype=torch.float64, device=dev).index_add_(0, nq, A)
    if bool((cnt > 1.5).any()):
        raise RuntimeError("zgnt: a question of the batch holds two rows for one of the first stage's top five")
    row1 = torch.arange(1, N + 1, dtype=torch.float64, device=dev)
    anc = torch.zeros(B, TOP, dtype=torch.float64, device=dev).index_add_(0, nq, A * row1[:, None]).to(torch.int64) - 1
    a_r = anc.index_select(0, nq)                                        # (N, 5): the question's places' rows, -1 none
    ok = (a_r >= 0) & (a_r != torch.arange(N, device=dev)[:, None])
    has = ok.any(1, keepdim=True)
    xs = x[:, J_PAIR:J_ANCHOR]                                           # pair_1..5, then res_1..5
    ins = (h, xs, nq, anc.clamp(min=0), ok, has, wa, wp, wr, b)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (h, wa, wp, wr, b)):
        from torch.utils.checkpoint import checkpoint      # the per-place gathers are made again in backward, not kept
        return checkpoint(attend, *ins, use_reentrant=False, preserve_rng_state=False)
    return attend(*ins)


def attend(h, xs, nq, ia, ok, has, wa, wp, wr, b):
    """(N, H): the message, one place at a time (index_select, a row-wise dot product, softmax, a weighted sum in place
    order). ia (B, 5): each question's places' rows (0 where none; ok masks them)."""
    import torch
    u = h / h.norm(dim=1, keepdim=True).clamp(min=EPS)
    q = u.mm(wa)
    qa = torch.stack([(q * u.index_select(0, ia[:, k]).index_select(0, nq)).sum(1) for k in range(TOP)], 1)
    lg = qa + wp * xs[:, :TOP] + wr * xs[:, TOP:] + b
    lg = lg.masked_fill(~ok, float("-inf")).masked_fill(~has, 0.0)
    alpha = (torch.softmax(lg, 1) * ok.to(lg.dtype)).to(h.dtype)
    m = alpha[:, :1] * h.index_select(0, ia[:, 0]).index_select(0, nq)
    for k in range(1, TOP):
        m = m + alpha[:, k:k + 1] * h.index_select(0, ia[:, k]).index_select(0, nq)
    return m


def message_ref(h, x, nq, B, wa, wp, wr, b):
    """message() by loops in float64 (the self-test's reference)."""
    N, H = h.shape
    anc = -np.ones((B, TOP), np.int64)
    for r in range(N):
        for k in range(TOP):
            if x[r, J_ANCHOR + k] > 0.5:
                assert anc[nq[r], k] < 0
                anc[nq[r], k] = r
    u = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), EPS)
    out = np.zeros((N, H))
    for r in range(N):
        ks = [k for k in range(TOP) if anc[nq[r], k] >= 0 and anc[nq[r], k] != r]
        if not ks:
            continue
        lg = np.array([u[r] @ wa @ u[anc[nq[r], k]] + wp * x[r, J_PAIR + k] + wr * x[r, J_RES + k] + b[k] for k in ks])
        w = np.exp(lg - lg.max())
        w /= w.sum()
        out[r] = sum(w[i] * h[anc[nq[r], k]] for i, k in enumerate(ks))
    return out


def l1_hook(G, m, out):
    """l1's output plus W_m applied to each row's message (the batch's NEXT columns set by ZG6R's patched forward)."""
    got = m.gnt_in
    if got is None:
        raise RuntimeError("zgnt: zg6r's l1 ran outside ZG6R's forward (no NEXT input set)")
    x, nq, B = got
    if out.shape[0] != x.shape[0]:
        raise RuntimeError(f"zgnt: l1 ran on {out.shape[0]} rows, the batch's NEXT block holds {x.shape[0]}")
    msg = message(G.torch, out, x, nq, B, m.gnt_wa.to(out.dtype), m.gnt_wp, m.gnt_wr, m.gnt_b)
    return out + G.Fn.linear(msg, m.gnt_wm.to(out.dtype))


@contextlib.contextmanager
def on_model(G):
    """For the block: a ZG6R built gets W_m and W_a (hidden x hidden), wp, wr and the five biases (zeros, no draw) and
    l1's hook after zg6r's own __init__; ZG6R's forward sets the batch's NEXT columns for the hook and clears them after.
    Restored after, also on a failure."""
    torch, nn = G.torch, G.nn
    orig_init, orig_fwd = G.ZG6R.__init__, G.ZG6R.forward

    def init(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        orig_init(self, blocks, widths, hidden, dropout, seed, arm, ctx)
        h = int(self.l1.out_features)
        self.gnt_wm = nn.Parameter(torch.zeros(h, h))
        self.gnt_wa = nn.Parameter(torch.zeros(h, h))
        self.gnt_wp = nn.Parameter(torch.zeros(1))
        self.gnt_wr = nn.Parameter(torch.zeros(1))
        self.gnt_b = nn.Parameter(torch.zeros(TOP))
        self.gnt_in = None
        self.l1.register_forward_hook(lambda _mod, _inp, out, m=self: l1_hook(G, m, out))

    def forward(self, feats, keep, nq, B, base_z):
        x = feats.get(NEXT)
        if x is None or x.shape[1] != WIDTH:
            raise RuntimeError(f"zgnt: the batch holds no {NEXT} block of {WIDTH} columns (GNT's carve adds it)")
        self.gnt_in = (x, nq, int(B))
        try:
            return orig_fwd(self, feats, keep, nq, B, base_z)
        finally:
            self.gnt_in = None

    with M1.WQ.patch(G.ZG6R, __init__=init, forward=forward):
        yield


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def on_gnt(out=None):
    """zgnx with GNT's names, code, term record, NEXT block, width and model patch, and ltmcols GNW's (zgnw.on_cols; out:
    GNW's column root, or a test's); restored after, also on a failure."""
    with GNW.on_cols(out), M1.WQ.patch(GNX, FITS=FITS, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, OUT=OUT,
                                       fit_name=fit_name, pair_name=pair_name, code_extra=code_extra,
                                       term_record=term_record, next_block=next_block, IN=IN, WIDTH=WIDTH,
                                       on_model=on_model):
        yield


@contextlib.contextmanager
def on_s0stop():
    """s0stop's stage table with GNT's entry (zgnt, its root, its two seed-0 fits, LTG's rule, this declaration), in this
    process; restored after."""
    with M1.WQ.patch(S0, STAGES={**S0.STAGES, KEY: ("zgnt", OUT, fit_name(0, S0.S0_SPLIT), fit_name(0, S0.HP_SPLIT),
                                                     "ltg", DECLARED)}):
        yield


def stop_record(base=ROOT):
    """GNT's current s0stop.json (s0stop.stop_record under GNT's entry), else None."""
    with on_s0stop():
        return S0.stop_record(KEY, base)


# ── the commands (zgnx's under GNT's names) ──────────────────────────────────


def gnw_files_problems(base=ROOT):
    """Why GNW's columns are not ready for a GNT fit: its match.json or cols.json missing or not passed."""
    out = []
    for f, what in (("match.json", "the columns' match against gnpcols"), ("cols.json", "the columns' check")):
        p = Path(base) / COLS_ROOT / f
        if not p.is_file():
            out.append(f"{M1.WQ.rel(p)}: {what} is missing")
        elif not json.loads(p.read_text(encoding="utf-8")).get("pass"):
            out.append(f"{M1.WQ.rel(p)}: {what} did not pass")
    return out


def fit(name, base=ROOT):
    if name not in FITS:
        raise SystemExit(f"zgnt: {name} is not one of GNT's fits ({', '.join(FITS)})")
    bad = gnw_files_problems(base)
    if bad:
        raise SystemExit(f"zgnt: {name}: " + "; ".join(bad))
    with on_gnt():
        return GNX.fit(name)


def compare_all(base=ROOT):
    with on_gnt():
        return GNX.compare_all(base)


def fit_block(name, what, base=ROOT):
    """A GNT fit's stamp block (zgnx's fit_block under GNT's names)."""
    with on_gnt():
        return GNX.fit_block(name, what, base)


def fit_problems(call, base=ROOT):
    """zgnw.fit_problems with GNT's fits and term (TERM_PARAMS parameters on GNW's fifteen columns), inside on_gnt: its
    term, base and ISO verdict; its NEXT block (the split, the columns, one scorer for train and read, every training
    carve in its train and every s1eval carve in its read); its start; its device; its trainer's code against the base's
    fit of its seed and split."""
    with M1.WQ.patch(GNW, FITS=FITS, TERM_PARAMS=TERM_PARAMS):
        return GNW.fit_problems(call, base)


def gnt_problems(base=ROOT):
    """Why a GNT fit that is read did not read GNW's columns by this code: its train's and read's NEXT block must name
    GNT's declaration, GNW's column root and GNW's declaration, its train's term GNW's fifteen columns with GNT's
    parameters, and its train's code this zgnt.py, zgnw.py, gnwcols.py, gnpcols.py and zpfp.py; and GNW's match.json and
    cols.json must have passed."""
    out, own = [], own_shas()
    want = (DECLARED, COLS_ROOT, GNWC.DECLARED)
    with on_gnt():
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
                got = (nk.get("declared_in"), nk.get("columns_root"), nk.get("columns_declared_in"))
                if got != want:
                    out.append(f"{name}: its {what}'s NEXT block names {got}, not GNT's on GNW's columns {want}")
            if (ks.get("inputs"), ks.get("columns"), ks.get("term_parameters")) != (IN, list(GNWC.COLS), TERM_PARAMS):
                out.append(f"{name}: its train's term takes {ks.get('inputs')} inputs of {ks.get('columns')} with "
                           f"{ks.get('term_parameters')} parameters, not GNT's")
            code = ks.get("code") or {}
            diff = sorted(k for k, v in own.items() if code.get(k) != v)
            if diff:
                out.append(f"{name}: its train's code is not this GNT's in {diff}")
        read = any(GNX.is_read(f, base) for f in FITS)
    if read:
        out += gnw_files_problems(base)
    return out


def next_form(screen, full):
    s = (screen or {}).get("verdict")
    v = (full or {}).get("verdict")
    if s == INCOMPLETE or v == INCOMPLETE:
        return "what is missing is fixed in an amendment, before any number is read"
    if s == ADOPT_FOR_FULL_RUN and v in (None, ADOPT):
        return ("the stack takes GNT's attention by an amendment to docs/STK_GNN_STACK.md (GNT is not one of its "
                "members; with GNW, the larger mean seed-0 delta over the ten deciding reads); GNT's own full run is held "
                "for the stack; next form: the same attention to the GNN's own top five after a first pass (GNR's)")
    return ("learned attention to the first stage's top five closes for the GNN; next form: the same attention to the "
            "GNN's own top five after a first pass (GNR's)")


def calls(base=ROOT):
    """GNT's screen (and its full run, when every fit is read) by LTG's in-domain rule, DEP's sixteen-read rule beside,
    through zgnx's reads, slices and checks under GNT's names; outputs/gnt/calls.json and .md."""
    with on_gnt():
        call = DEP.iso_call(base)
        new_r5, b0 = GNX.r5_gnx(call, base)
        base3 = DEP.base_seeds(call, base)
        bad = [f"{rd[0]} {rd[1]} seed {sd}: the base gives {v} in GNT's compare, {base3.get(rd, {}).get(sd)} in its own "
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
    bad += gnt_problems(base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": f"{GNX.ARM} with GNT's learned attention", "iso_verdict": call,
           "base": "zg6r-m (ISO)" if call == ISO.REPLACE else "zg6r (ML1)", "screen": screen, "full_run": full,
           "dep_rule_beside": dep_rule, "next_form": next_form(screen, full),
           "beside": {"pfs_a3": GNP.pfs_beside(base), "gnw": GNP.screen_beside("gnw", base),
                      "gnp": GNP.screen_beside("gnp", base), "gnx": GNP.screen_beside("gnx", base),
                      "gna": GNP.screen_beside("gna", base)},
           "problems": bad, "parameters": params, **own_shas(), "zgnx_sha256": M1.WQ.sha_src(GNX.__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    md = [f"# GNT: {GNX.ARM} with learned attention to the first stage's top five against {rec['base']}, on PKO's pools",
          "", f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}",
          f"DEP's sixteen-read rule beside: screen {dep_rule['screen']}, full run {dep_rule['full_run']}",
          f"next form: {rec['next_form']}", ""]
    rows = screen["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in screen.get("beside", [])]
    if rows:
        md += ["| split | read | base s0, s1, s2 | GNT s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
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
    for st in ("gnw", "gnp", "gnx", "gna"):
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
         "message": globals()["message"], "gnw.FITS": GNW.FITS, "gnw.TERM_PARAMS": GNW.TERM_PARAMS}
    if G is not None:
        s.update({"g.encoder_used": G.encoder_used, "g.init": G.ZG6R.__init__, "g.forward": G.ZG6R.forward,
                  "g.carve_init": G.G6rLinkCarve.__init__, "g.carve_batch": G.G6rLinkCarve.batch})
    return s


def toy_term(G):
    """GNT on zg6r's own toy carve (GNX's toy: zg6r's self-test steps 2 and 3) with GNW-layout toy columns (zgnw's
    toy_cols15): the carve's columns are GNW's files, and the carve refuses no split, a missing split's columns and another
    scorer's; the NEXT block names GNT's declaration on GNW's columns; the term's parameters are zeros; at W_m = 0 the
    eval and the training forward are zg6r's with the same parameters bit for bit, W_m gets a gradient and W_a, wp, wr and
    the biases none; with the term set, the forward's message is message_ref's on the l1 states and NEXT columns the
    forward hands it, the term moves the score and every term parameter has a gradient; l1 outside the forward, a batch
    without NEXT or with a narrower one, and two rows on one place stop; the strict load with and without the patch; a
    two-epoch lean_gpu.fit_variant that moves W_m and W_a; the patches' restoring."""
    import torch
    G6 = G.G6
    G.LG.bind_device_ops()
    out = {}
    sp = "WQ-musique"
    with tempfile.TemporaryDirectory(prefix="zgnt_") as td:
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

        X0 = GNW.toy_cols15(tmp / "gnw", sp, "metaqa", "toy", tmp / "cache", rng, "a" * 64)
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
                out["next_block_gnt"] = (nx["declared_in"], nx["columns"], nx["columns_declared_in"]) == (
                    DECLARED, list(GNWC.COLS), GNWC.DECLARED)
                out["carve_cols_are_files"] = bool(torch.equal(ce.gnx_X, torch.from_numpy(X)))
                blocks = ["rank", "SEMB"]
                qs = np.r_[np.arange(ce.rows)[::-1][:7], np.arange(ce.rows)[:5]].astype(np.int64)
                B = qs.size
                feats, nq, bz, gold = ce.batch(qs, blocks)
            out["carve_restored"] = cls.__init__ is init0 and cls.batch is batch0
            rows = np.concatenate([np.arange(ce.off_np[q], ce.off_np[q + 1]) for q in qs])
            want = torch.nan_to_num(ce.gnx_X[torch.from_numpy(rows)].to(torch.float32), nan=0.0)
            out["next_rows_are_carve_rows"] = bool(torch.equal(feats[NEXT], want))
            keep = torch.ones(B, len(blocks))
            orig_init, orig_fwd = G.ZG6R.__init__, G.ZG6R.forward
            names0 = tuple(G6.G6_NAMES) + tuple(G.G6R_NAMES)
            gen = torch.Generator().manual_seed(5)
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
                out["wm_grad"] = gr["gnt_wm"]
                out["other_grads_at_zero"] = gr["gnt_wa"] + gr["gnt_wp"] + gr["gnt_wr"] + gr["gnt_b"]
                with torch.no_grad():
                    m.gnt_wm.copy_(torch.randn(m.gnt_wm.shape, generator=gen) * 0.3)
                    m.gnt_wa.copy_(torch.randn(m.gnt_wa.shape, generator=gen) * 0.8)
                    m.gnt_wp.fill_(1.5)
                    m.gnt_wr.fill_(-0.7)
                    m.gnt_b.copy_(torch.randn(TOP, generator=gen))
                seen, msg0 = [], globals()["message"]

                def spy(torch_, h, x, nq_, B_, wa, wp, wr, b):
                    r = msg0(torch_, h, x, nq_, B_, wa, wp, wr, b)
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
                ref = message_ref(h.double().numpy(), x.double().numpy(), nq.numpy(), B,
                                  m.gnt_wa.detach().double().numpy(), 1.5, -0.7, m.gnt_b.detach().double().numpy())
                out["message_max_abs"] = float(np.abs(r.double().numpy() - ref).max())
                out["message_ref_max"] = float(np.abs(ref).max())       # float32 in the forward: relative to this
                out["message_rows"] = int((np.abs(ref).sum(1) > 0).sum())
                m.train()
                m.zero_grad()
                torch.manual_seed(9)
                G.LG.listwiseD(m(feats, keep, nq, B, bz), gold, nq, B).backward()
                out["term_grads_set"] = min(float(getattr(m, k).grad.abs().sum()) for k in TERM_NAMES)
                m.eval()
                with torch.no_grad():
                    out["gnt_in_cleared"] = m.gnt_in is None
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
            # a two-epoch lean_gpu.fit_variant on the patched carve and model: the term trains inside the trainer's own
            # loop, W_a once W_m has moved
            cfg = {"lr": 2e-3, "wd": 1e-4, "dropout": 0.1, "epochs": 2, "swa_from": 0, "cos": 0, "adamw": 0, "drop": 0}
            with GNX.on_carve(G, cls), on_model(G), M1.WQ.patch(G.LG, LeanMLP8D=G.ZG6R):
                res = G.LG.fit_variant([ce], blocks, cfg, 0, 16, "none", "cpu")
            out["fit_wm_moved"] = float(res["swa"]["gnt_wm"].abs().sum()) > 0
            out["fit_wa_moved"] = float(res["swa"]["gnt_wa"].abs().sum()) > 0
            out["fit_loss_finite"] = all(math.isfinite(c["loss"]) for c in res["curve"])
            out["fit_restored"] = G.LG.LeanMLP8D is not G.ZG6R and G.ZG6R.__init__ is orig_init and cls.batch is batch0
        GNX.LOADED.clear()
    return out


def selftest():
    import torch
    must_stop = GNP.must_stop
    # 1. names: GNT's own, none another stage's
    assert fit_name(0, "WQ-musique") == "scr-gnt-at-s0" and fit_name(2, "WQ-hotpotqa") == "scr-gnt-at-s2-hp"
    assert pair_name(1, "WQ-hotpotqa") == "scr-gnt-at-vs-base-s1-hp"
    import zrtr as RTR
    import zira as IRA
    import zisb as ISB
    import zgna as GNA
    import zlta as ZLA
    import zgnr as GNR
    import zltx as LTX
    assert not set(FITS) & (set(RTR.FITS) | set(IRA.FITS) | set(ISB.FITS) | set(LTM.FITS) | set(GNA.FITS)
                            | set(ZLA.FITS) | set(GNP.FITS) | set(GNW.FITS) | set(GNR.FITS) | set(LTX.FITS))
    # 2. the message by itself, against loops: questions of one to nine rows, rows out of question order, a place a
    #    question lacks, W_a, wp, wr and b set; two rows on one place stop; a two-row question's rows each take the
    #    other's state whole
    rng = np.random.default_rng(7)
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
    wa, wp, wr, b = rng.standard_normal((H, H)), 2.0, -1.3, rng.standard_normal(TOP)
    T = lambda a: torch.tensor(a, dtype=torch.float64)                    # noqa: E731
    with torch.no_grad():
        got = message(torch, T(h), T(x), torch.tensor(nq), B, T(wa), T([wp]), T([wr]), T(b)).numpy()
        got0 = message(torch, T(h), T(x), torch.tensor(nq), B, T(np.zeros((H, H))), T([wp]), T([wr]), T(b)).numpy()
    ref = message_ref(h, x, nq, B, wa, wp, wr, b)
    assert got.shape == (N, H) and np.abs(got - ref).max() < 1e-12, np.abs(got - ref).max()
    assert np.abs(got0 - message_ref(h, x, nq, B, np.zeros((H, H)), wp, wr, b)).max() < 1e-12
    #    at W_a = 0 the weights are zgnw's (the scores' alone)
    with torch.no_grad():
        gw = GNW.message(torch, T(h), T(x), torch.tensor(nq), B, T([wp]), T([wr]), T(b)).numpy()
    assert np.abs(got0 - gw).max() < 1e-12 and np.abs(got - got0).max() > 1e-3
    #    with gradients: the checkpointed message gives the same values and the same gradients as the plain one
    import torch.utils.checkpoint as TUC
    up = T(np.random.default_rng(9).standard_normal((N, H)))
    grads = []
    for ck in (True, False):
        ins = [T(h).requires_grad_(), T(wa).requires_grad_(), T([wp]).requires_grad_(), T([wr]).requires_grad_(),
               T(b).requires_grad_()]
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
    try:
        message(torch, T(h), T(x2), torch.tensor(nq), B, T(wa), T([wp]), T([wr]), T(b))
        raise AssertionError("two rows on one place passed")
    except RuntimeError:
        pass
    # 3. the patches: zgnx's names, code, width and model patch GNT's and ltmcols' root, stage, declaration, columns and
    #    width GNW's inside; restored after, also on a failure; zgnx's own names, pairs and code under them
    s0 = snapshot()
    with on_gnt():
        assert GNX.FITS is FITS and (GNX.DECLARED, GNX.STAGE, GNX.KEY, GNX.OUT) == (DECLARED, STAGE, KEY, OUT)
        assert (GNX.IN, GNX.WIDTH, LTMC.WIDTH, tuple(LTMC.COLS)) == (15, 15, 15, GNWC.COLS) and GNX.on_model is on_model
        assert GNX.pairs(ISO.KEEP)[pair_name(0, "WQ-musique")] == ("scr-gnt-at-s0", "scr-ml1-zg6r-s0")
        assert GNX.pairs(ISO.REPLACE)[pair_name(0, "WQ-hotpotqa")] == ("scr-gnt-at-s0-hp", "scr-iso-zg6rm-s0-hp")
        assert (Path(LTMC.OUT), LTMC.STAGE, LTMC.DECLARED) == (Path(GNWC.OUT), "GNW", GNWC.DECLARED)
        cs = GNX.code_extra()
        assert all(f"{k}_sha256" in cs for k in OWN_CODE + ("zgnx", "zdep", "zltg", "zltm", "ltmcols")), cs
        with GNX.on_names():
            assert (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY) and set(FITS) <= set(M1.ALL_FITS)
            assert not set(GNX_FITS) & set(M1.ALL_FITS) and not set(GNW.FITS) & set(M1.ALL_FITS)
            assert all(f"{k}_sha256" in M1.code_shas(GNX.ARM) for k in OWN_CODE)
        with M1.WQ.patch(GNW, FITS=FITS, TERM_PARAMS=TERM_PARAMS):
            assert GNW.FITS is FITS and GNW.TERM_PARAMS == 32775
    assert snapshot() == s0 and (GNX.IN, LTMC.WIDTH, LTMC.STAGE) == (15, 7, "LTM") and GNX.on_model is not on_model
    try:
        with on_gnt():
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert snapshot() == s0
    must_stop(fit, "scr-gnw-mw-s0")                                      # GNW's own fit is not GNT's
    must_stop(fit, "scr-gnx-nx-s0")                                      # nor GNX's
    tmp = Path(tempfile.mkdtemp(prefix="zgnt_"))
    try:                                                                 # GNW's match and check hold a fit
        must_stop(fit, fit_name(0, "WQ-musique"), tmp)
        (tmp / COLS_ROOT).mkdir(parents=True)
        M1.WQ.write_json(tmp / COLS_ROOT / "match.json", {"pass": True})
        assert len(gnw_files_problems(tmp)) == 1
        M1.WQ.write_json(tmp / COLS_ROOT / "cols.json", {"pass": False})
        assert len(gnw_files_problems(tmp)) == 1
        must_stop(fit, fit_name(0, "WQ-musique"), tmp)
        M1.WQ.write_json(tmp / COLS_ROOT / "cols.json", {"pass": True})
        assert gnw_files_problems(tmp) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    st0 = dict(S0.STAGES)
    with on_s0stop():
        assert S0.STAGES[KEY] == ("zgnt", OUT, "scr-gnt-at-s0", "scr-gnt-at-s0-hp", "ltg", DECLARED)
        assert S0.STAGES["gnx"] == st0["gnx"] and S0.deciding("ltg") == S0.deciding(S0.STAGES["gnx"][4])
    assert S0.STAGES == st0 and KEY not in S0.STAGES
    tmp = Path(tempfile.mkdtemp(prefix="zgnt_"))
    try:                                         # GNT's stop record on made files: current only for the read it was made on
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
    # 4. the columns a GNT carve loads are gnwcols': its columns load inside on_gnt, from ltmcols' root there, and stop
    #    outside it; LTM-stamped and GNP-stamped columns stop inside it; the NEXT block names GNT's declaration on GNW's
    #    root and declaration
    import lean_cache as LC
    import ltwdiag as LTW
    tmp = Path(tempfile.mkdtemp(prefix="zgnt_"))
    try:
        rng = np.random.default_rng(31)
        M = LTW.unit(rng.normal(size=(200, 16))).astype(np.float32)
        LTW.toy_tree(tmp, "musique", "s1eval", M, rng, nq=6, n=20)
        parts, _recs = LC.part_dirs("musique", "s1eval", tmp / "cache")
        ps = {p.name: LC.sha_file(p / "record.json") for p in parts}
        sp = "WQ-hotpotqa"
        rec = GNWC.cols("musique", "s1eval", tmp, tmp / "gnw", {"musique": GNWC.PP.FakeNodes(M)})
        GNWC.GNPC.cols("musique", "s1eval", tmp, tmp / "gnp", {"musique": GNWC.PP.FakeNodes(M)})
        with on_gnt(tmp / "gnw"):
            X2, r2, _sha = LTM.load_next("musique", "s1eval", sp, ps)      # out None: ltmcols' OUT, GNW's here
            assert X2.shape == (rec["rows"], 15) and (r2["stage"], r2["declared_in"]) == ("GNW", GNWC.DECLARED), r2
            with GNX.on_split(sp, rec["fit_sha256"]):
                GNX.LOADED.append({"dataset": "musique", "carve": "s1eval", "record_sha256": "r"})
                nb = GNX.next_block(sp)
            GNX.LOADED.clear()
            assert (nb["declared_in"], Path(nb["columns_root"]).name, nb["columns_declared_in"], nb["fit_sha256"],
                    nb["columns"]) == (DECLARED, "gnw", GNWC.DECLARED, GNWC.rule_sha(), list(GNWC.COLS)), nb
            must_stop(LTM.load_next, "musique", "s1eval", sp, ps, tmp / "gnp")   # GNP's columns inside
        Y = LTM.toy_cols(tmp / "ltm", sp, "musique", "s1eval", tmp / "cache", rng, "e" * 64)
        assert LTM.load_next("musique", "s1eval", sp, ps, tmp / "ltm")[1]["stage"] == "LTM" and Y.shape[1] == 7
        must_stop(LTM.load_next, "musique", "s1eval", sp, ps, tmp / "gnw")  # GNW's columns outside on_gnt
        with on_gnt(tmp / "gnw"):
            must_stop(LTM.load_next, "musique", "s1eval", sp, ps, tmp / "ltm")   # LTM's columns inside
        assert snapshot() == s0
    finally:
        GNX.LOADED.clear()
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. GNT's toy term under GNT's patches: zg6r's toy carve loads GNW-stamped columns, the term is zeros, at W_m = 0 the
    #    forwards are zg6r's bit for bit, the forward's message is message_ref's, the term trains in lean_gpu's loop
    G = GNX.import_g()
    sg = snapshot(G)
    with on_gnt():
        t = toy_term(G)
        rec = GNX.term_record(G)
    assert (t["no_split"], t["other_split"], t["other_scorer"]) == ("refused",) * 3, t
    assert t["loaded"] == ["metaqa/toy"] and t["scorer"] == "a" * 64 and t["carve_restored"] and t["next_block_gnt"], t
    assert t["files_are_toy_cols"] and t["carve_cols_are_files"] and t["next_rows_are_carve_rows"], t
    assert t["term_zero"] and t["l1_out"] == 16 and t["term_params"] == 2 * 16 * 16 + 2 + TOP and not t["plain_has_term"], t
    assert t["term_shapes"] == {"gnt_wm": (16, 16), "gnt_wa": (16, 16), "gnt_wp": (1,), "gnt_wr": (1,),
                                "gnt_b": (TOP,)}, t
    assert t["eval_start_is_zg6r"] and t["train_start_is_zg6r"] and t["wm_grad"] > 0 and t["other_grads_at_zero"] == 0, t
    assert t["term_moves_score"] and t["message_calls"] == 1 and t["message_x_is_next"], t
    assert t["message_max_abs"] < 1e-5 * max(1.0, t["message_ref_max"]) and t["message_rows"] > 0, t
    assert t["term_grads_set"] > 0, t
    assert (t["no_next"], t["narrow_next"], t["l1_outside"], t["two_anchors"]) == ("refused",) * 4, t
    assert t["gnt_in_cleared"] and t["model_restored"], t
    assert t["strict_load_patched"] == "passed" and t["strict_load_plain"] == "refused", t
    assert t["fit_wm_moved"] and t["fit_wa_moved"] and t["fit_loss_finite"] and t["fit_restored"], t
    assert rec["term"].startswith("GNT's learned attention") and rec["term_parameters"] == 2 * 128 * 128 + 7 == 32775, rec
    assert rec["inputs"] == 15 and rec["parameters_h128"] == 180218 + 32775 == 212993, rec
    assert rec["columns"] == list(GNWC.COLS) and set(rec) == set(GNX.term_record_keys()) | {"term"}, rec
    assert snapshot(G) == sg
    # 6. the stamp under GNT's names: the train's NEXT block is GNT's on GNW's columns, with GNT's term; one without the
    #    columns stops
    seen, orig_stamp = [], M1.stamp
    hp = fit_name(0, "WQ-hotpotqa")
    try:
        M1.stamp = lambda name, env, what, extra=None: seen.append((name, what, dict(extra or {})))
        with on_gnt(), GNX.on_gnx(G, ISO.KEEP):
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
    assert (e["next"]["columns_root"], e["next"]["columns_declared_in"]) == (COLS_ROOT, GNWC.DECLARED), e
    assert e["next"]["columns"] == list(GNWC.COLS) and e["inputs"] == 15 and e["term_parameters"] == 32775, e
    assert e["term"].startswith("GNT's learned attention") and e["x"] == 1, e
    assert snapshot(G) == sg and GNX.CUR == {"split": None, "fit_sha256": None, "out": None}
    # 7. the checks a read fit faces, on made stamp blocks: a declared fit gives none; GNW's term, GNP's columns, other
    #    code, GNW's NEXT block, another scorer, or a missing or failed GNW match or check each give problems; the calls'
    #    records on made reads
    tmp = Path(tempfile.mkdtemp(prefix="zgnt_"))
    try:
        name = fit_name(0, "WQ-musique")
        bse = DEP.base_fit(0, "WQ-musique", ISO.KEEP)

        def block(what, gnw_next=False, **over):
            need = GNX.train_carves("WQ-musique") if what == "screen" else GNX.eval_carves()
            k = {"term_parameters": TERM_PARAMS, "inputs": IN, "columns": list(GNWC.COLS), "layers": DEP.BASE_LAYERS,
                 "base": bse, "iso_verdict": ISO.KEEP, "device": {"gpu": "t"},
                 "next": {"declared_in": DECLARED, "columns_root": COLS_ROOT, "columns_declared_in": GNWC.DECLARED,
                          "split": "WQ-musique", "columns": list(GNWC.COLS), "fit_sha256": "f" * 64,
                          "carves": {c: "r" for c in need}},
                 "encoder": {"fit": "z", "seed": 0, "sha256": "s", "copies": 1}, "code": {**own_shas(), "a_sha256": "1"}}
            if gnw_next:                                                 # GNW's own NEXT block, on this block's carves
                k["next"] = {**k["next"], "declared_in": GNWC.DECLARED, "columns_declared_in": None}
            k.update(over)
            return k
        blocks_ = {}

        def fb(nm, what, base=ROOT):
            if nm == bse:
                return {"encoder": {"fit": "z", "seed": 0, "sha256": "s"}, "device": {"gpu": "t"},
                        "code": {"a_sha256": "1"}}
            return blocks_[what]
        (tmp / COLS_ROOT).mkdir(parents=True)
        for f in ("match.json", "cols.json"):
            M1.WQ.write_json(tmp / COLS_ROOT / f, {"pass": True})
        with M1.WQ.patch(GNX, is_read=lambda nm, base=ROOT: nm == name, fit_block=fb):
            for over, n_fit, n_gnt in (({}, 0, 0), ({"term_parameters": 16391}, 2, 1),
                                       ({"columns": list(GNWC.GNPC.COLS)}, 2, 1),
                                       ({"code": {**own_shas(), "zgnt_sha256": "0" * 64, "a_sha256": "1"}}, 0, 1),
                                       ({"code": {**own_shas(), "zgnw_sha256": "0" * 64, "a_sha256": "1"}}, 0, 1),
                                       ({"gnw_next": True}, 0, 2)):
                blocks_.update(screen=block("screen", **over), read=block("read", **over))
                with on_gnt():
                    fp = fit_problems(ISO.KEEP, tmp)
                gp = gnt_problems(tmp)
                assert (len(fp), len(gp)) == (n_fit, n_gnt), (over, fp, gp)
            blocks_.update(screen=block("screen"), read=block("read", next={**block("read")["next"], "fit_sha256": "g" * 64}))
            with on_gnt():
                assert len(fit_problems(ISO.KEEP, tmp)) == 1
            blocks_.update(screen=block("screen"), read=block("read"))
            for f in ("match.json", "cols.json"):
                M1.WQ.write_json(tmp / COLS_ROOT / f, {"pass": False})
                assert len(gnt_problems(tmp)) == 1
                (tmp / COLS_ROOT / f).unlink()
                assert len(gnt_problems(tmp)) == 1
                M1.WQ.write_json(tmp / COLS_ROOT / f, {"pass": True})
            assert gnt_problems(tmp) == []
        # the calls on made reads: GNT's s0 read and declared, every read the rule takes above the base's three seeds
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
        (tmp / COLS_ROOT / "cols.json").unlink()
        cj, md = made_calls()                                            # no GNW check: INCOMPLETE
        assert cj["stage"] == STAGE and cj["screen"]["verdict"] == INCOMPLETE, cj["screen"]
        assert len(cj["problems"]) == 1 and "check" in cj["problems"][0] and "## Screen: INCOMPLETE" in md, cj["problems"]
        M1.WQ.write_json(tmp / COLS_ROOT / "cols.json", {"pass": True})
        cj, md = made_calls()
        assert cj["screen"]["verdict"] == ADOPT_FOR_FULL_RUN and not cj["problems"] and cj["full_run"] is None, cj
        assert cj["parameters"] == {name: {k: block("screen").get(k) for k in GNX.term_record_keys()}}, cj["parameters"]
        assert cj["next_form"] == next_form(cj["screen"], None) and "| split | read |" in md and cj["beside"]["gnw"] is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert snapshot(G) == sg
    # 8. what follows, by the verdicts
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, None).startswith("the stack takes GNT's attention")
    assert next_form({"verdict": NO_GAIN}, None).startswith("learned attention to the first stage's top five closes")
    assert next_form({"verdict": NOT_ADOPTED}, None).startswith("learned attention to the first stage's top five closes")
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, {"verdict": NOT_ADOPTED}).startswith("learned attention")
    assert next_form({"verdict": INCOMPLETE}, None).startswith("what is missing")
    log("selftest: names; the message against loops (one to nine rows, rows out of order, a missing place, W_a set; at "
        "W_a = 0 zgnw's; two rows on one place stop; checkpointed = plain in values and gradients); the patches and their "
        "restoring (also on a failure); GNW's match and check hold a fit; gnwcols' columns load inside, from ltmcols' "
        "root there, and stop outside, LTM's and GNP's stop inside, the NEXT block names GNT's on GNW's; GNT's toy term "
        "(zeros; zg6r's forwards bit for bit at W_m = 0; W_m's gradient alone there; the forward's message is the loops'; "
        "the term moves the score and every term parameter gets a gradient; no NEXT, a narrower NEXT, l1 outside and two "
        "anchors stop; W_m and W_a train in lean_gpu's loop); the stamp's NEXT block GNT's with GNT's term and its "
        "refusal without columns; the read checks on made blocks; the calls on made reads; what follows")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fit", "compare-all", "calls", "s0stop-check", "s0stop-run", "gate"))
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
