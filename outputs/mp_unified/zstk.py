"""STK's stage (docs/STK_GNN_STACK.md, amendment A1 in section 10): the GNN forms whose screens pass (DEP, LTG, RTR, IRA,
GNX; the members, fixed by `members` under section 2) stacked on the GNN of record, zg6r, each entering as it was screened.
On PKO's roots, on the host's GPU: the stack's screen at seed 0 on both WQ splits against the base's seed-0 fit, read under
every rule its members use (DEP's sixteen reads, LTG's in-domain ten); the full run (seeds 1 and 2) when the screen asks
for it. With one member, that member's own full run runs instead; with none, nothing (section 3).

How the members compose (section 10). The model is zg6r's with each member's patch, entered in this order, each wrapping
what the one before left: IRA's model (or RTR's rounds, when IRA is not a member); LTG's term, in each layer (four with
DEP: zltg's per-layer names and counts are set to zg6r's layer count) and, under RTR's or IRA's stack, in each layer of
both rounds, with its inputs taken once a forward (they do not depend on the state) and its next-hop loss on the last
round's score; GNX's model and carve. With DEP, zg6r is imported with four layers (zdep.import_deep, a process of its
own). zml1s, zg6r, zdep, zltg, zrtr, zira, zgnx and s0stop are imported and not edited; every patched name is put back
afterwards, also on a failure.

Amendment A2 (section 11): the members wait on GNA's screen too (docs/GNA_ALL_ROWS_COLUMNS_IN_GNN.md: GNX's arm on LTA's
columns). The gnx slot passes when GNX's or GNA's screen passes, and reads LTA's columns (zlta.on_cols, with GNA's NEXT
block and term record) when GNA's passes and GNX's does not, or both pass and GNA's mean seed-0 delta over section 6's six
reads is larger; LTM's otherwise. The gnx slot's own full run is GNX's on LTM's columns and GNA's on LTA's
(own-gate --stage gna). zgna, zlta and ltacols are imported and not edited.

Amendment A3 (section 12): the members wait on GNP's screen too (docs/GNP_PAIR_COLUMNS_IN_GNN.md: GNX's arm on PFS A3's
six pair columns). GNP is a form of its own (FORMS: ORDER's and gnp) that enters through the gnx slot: when GNP's screen
passes, the slot's block is its NEXT columns (LTM's or LTA's, by A2) then the six pair columns, one block with one shared
1 (IN = 27), or the six alone (IN = 13, as GNP was screened) when neither GNX's nor GNA's screen passes. The stack runs
with two or more forms; GNP's own full run is released by own-gate --stage gnp. zgnp, gnpcols and zpfp are imported on
first use (a fit, the calls, the self-test; never the gates) and not edited.

Amendment A4 (section 13): the members wait on GNW's and LTX's screens too (docs/GNW_PAIR_WEIGHTS_IN_GNN.md,
docs/LTX_CROSSFIT_NEXT_HOP_GNN.md), each read as s0stop's gate reads a verdict under its own entry. A passing GNW or LTX
is a form and a member of its own (MEMBERS: ORDER's, then gnw, ltx), read under LTG's in-domain rule. This code composes
ORDER's members only (COMPOSED); how the later forms compose is amendment A6's (A4 named it A5), committed before any
stack fit. So with a member this code does not compose, stack-gate --wait looks every POLL seconds until the zstk.py on
disk does (a fresh process: `composes`), and at MEMBERS_BY it decides the stack does not run (outputs/stk/stack_gate.json),
which the own gates read as the stack's screen NOT_ADOPTED. GNW's and LTX's own full runs are released by own-gate
--stage gnw / ltx. zgnw and zltx are imported in the self-test alone (their names checked), never by the gates or a fit.

Amendment A5 (section 14): the members wait on GNR's, GNT's and GNS's screens too (docs/GNR_REANCHORED_NEXT_GNN.md,
docs/GNT_LEARNED_ATTENTION_GNN.md, docs/GNS_SELF_ANCHORED_ATTENTION_GNN.md), read as A4's are; each passing one is a form
and a member of its own (MEMBERS: A4's, then gnr, gnt, gns). GNR and GNS are never the pick: their columns are per
seed, so no own full run before the freeze (NO_OWN); GNT's own full run is released by own-gate --stage gnt. One
attention form at most (GNW, GNT, GNS): the largest mean seed-0 delta over LTG's ten deciding reads. RTR and IRA also run
when GNR's or GNS's screen is ADOPT_FOR_FULL_RUN (reopen-gate; docs/RTR_GO_BACK_ROUNDS.md section 13,
docs/IRA_ITERATIVE_ALIGNMENT.md section 14), and when DEP's screen passes the members wait on that call; DEP or the
rounds then, not both: the larger mean over section 6's six reads, a tie to DEP. zgnr, zgnt and zgns are imported in the
self-test alone.

    python outputs/mp_unified/zstk.py members --wait                 (fix the members: outputs/stk/members.json)
    python outputs/mp_unified/zstk.py stack-gate --wait               (exit 0 with two or more forms, A4: once composed)
    python outputs/mp_unified/zstk.py composes                        (A4: exit 0 when this code composes the members)
    python outputs/mp_unified/zstk.py own-gate --stage dep --wait     (exit 0 when the stage's own full run is released)
    python outputs/mp_unified/zstk.py reopen-gate --wait              (A5: exit 0 when RTR and IRA run, 1 when they do not)
    python outputs/mp_unified/zstk.py fit --name scr-stk-st-s0        (train when needed, then read; kept when read before)
    python outputs/mp_unified/zstk.py s0stop-check                    (s0stop's check, for the stack)
    python outputs/mp_unified/zstk.py s0stop-run -- COMMAND ...       (s0stop's run, for the stack)
    python outputs/mp_unified/zstk.py gate --want ADOPT_FOR_FULL_RUN  (s0stop's gate, for the stack)
    python outputs/mp_unified/zstk.py compare-all                     (each stack fit read against the base's of its seed)
    python outputs/mp_unified/zstk.py calls                           (outputs/stk/calls.json and .md)
    python outputs/mp_unified/zstk.py --selftest
"""
import argparse
import calendar
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import zml1 as RULE  # noqa: E402  (stdlib only)
import zml1s as M1  # noqa: E402
import ziso as ISO  # noqa: E402
import zdep as DEP  # noqa: E402  (the base after ISO, the four layers, the slices and DEP's rule)
import zltg as LTG  # noqa: E402  (the next-hop term and loss, LTG's in-domain rule)
import zrtr as RTR  # noqa: E402  (the rounds, RTR's condition on DEP's calls)
import zira as IRA  # noqa: E402  (the plan between the rounds)
import zgnx as GNX  # noqa: E402  (the NEXT input, its carve and split)
import zgna as GNA  # noqa: E402  (A2: GNA's verdict, its NEXT block and term record, LTA's columns through zlta)
import s0stop as S0  # noqa: E402  (the seed-0 stop and the stages' verdicts)
import ml1start as MS  # noqa: E402  (on ISO's KEEP, ML1's zspk start read under ML1's names)
import readswa as RS  # noqa: E402  (a fit's read keeps p@swa only)

DECLARED = "docs/STK_GNN_STACK.md"
STAGE = "STK"
KEY = "stk"                                     # the stamp's key in a stack fit's screen.json and read.json and STK's compares
OUT = "outputs/stk"
ARM = DEP.ARM
TAG = "st"
ORDER = ("dep", "ltg", "rtr", "ira", "gnx")     # section 6's order: the pick's tie and the members' listing
MOD = {"dep": DEP, "ltg": LTG, "rtr": RTR, "ira": IRA, "gnx": GNX}
RULE_OF = {"dep": "dep", "rtr": "dep", "ira": "dep", "ltg": "ltg", "gnx": "ltg"}     # section 4
ROUNDS_OF = ("ira", "rtr")                      # IRA's arm holds RTR's rounds: IRA is the member when both pass
MEMBERS_BY = "2026-11-05T06:30:00Z"             # 12:00 IST on 5 November (section 2)
FREEZE = "2026-11-06T18:30:00Z"                 # 00:00 IST on 7 November: the own-run gates stop waiting (A1)
POLL = 600                                      # seconds between a waiting gate's looks
PICK_READS = tuple(k for k in LTG.deciding_keys() if k[0] == LTG.SLICE_SPLIT)       # section 6: the six WQ-hotpotqa reads
CODE_SKIP = DEP.CODE_SKIP
R5_TOL = DEP.R5_TOL
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, ADOPT, INCOMPLETE = (
    DEP.ADOPT_FOR_FULL_RUN, DEP.NO_GAIN, DEP.NOT_ADOPTED, DEP.ADOPT, DEP.INCOMPLETE)
VERDICTS = (ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED)
PAIR = "PAIR"                                   # A3: GNP's six pair columns
COLUMNS = ("LTM", "LTA", PAIR, "LTM+PAIR", "LTA+PAIR")   # A2, A3: whose columns the gnx slot reads, in block order
LATE = ("gnw", "ltx")                           # A4: GNW's and LTX's forms, each a member of its own
A5 = ("gnr", "gnt", "gns")                      # A5: GNR's, GNT's and GNS's forms, each a member of its own
FORMS = ORDER + ("gnp",) + LATE + A5            # A3-A5: the forms the stack holds (the pick's tie); GNP's is in the gnx slot
MEMBERS = ORDER + LATE + A5                     # A4, A5: the members, in their listing order
LATE_NAMES = {"gnw": ("zgnw", "outputs/gnw", "mw", "docs/GNW_PAIR_WEIGHTS_IN_GNN.md"),       # A4: module, root, tag and
              "ltx": ("zltx", "outputs/ltx", "xf", "docs/LTX_CROSSFIT_NEXT_HOP_GNN.md"),     # declaration, checked on import
              "gnr": ("zgnr", "outputs/gnr", "nx", "docs/GNR_REANCHORED_NEXT_GNN.md"),       # A5: the same, for GNR, GNT
              "gnt": ("zgnt", "outputs/gnt", "at", "docs/GNT_LEARNED_ATTENTION_GNN.md"),     # and GNS
              "gns": ("zgns", "outputs/gns", "sa", "docs/GNS_SELF_ANCHORED_ATTENTION_GNN.md")}
LATE_RULE = {"gnw": "ltg", "ltx": "ltg",        # A4: each read under LTG's in-domain rule (its declaration's section 4)
             "gnr": "ltg", "gnt": "ltg", "gns": "ltg"}                                       # A5: the same
ATTN = ("gnw", "gnt", "gns")                    # A5: one form at most: the largest mean seed-0 delta over the ten reads
SECOND = ("gnr", "gns")                         # A5: either one's ADOPT_FOR_FULL_RUN opens RTR and IRA (their A5)
NO_OWN = ("gnr", "gns")                         # A5: columns per seed: no own full run before the freeze, never the pick
TEN_READS = tuple(LTG.deciding_keys())          # A5: LTG's ten deciding reads (the attention forms' choice)
COMPOSED = ORDER                                # A4: the members this code stacks; A6 declares the later forms'
GNP_KEY, GNP_OUT, GNP_TAG = "gnp", "outputs/gnp", "px"                # A3: zgnp's names, checked against it on import
GNP_DECLARED = "docs/GNP_PAIR_COLUMNS_IN_GNN.md"
PAIR_COLS = ("pair_max", "res_max", "pair_mean", "res_mean", "pair_1", "res_1")    # gnpcols' six, checked on import
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")     # gnpcols sets their defaults on import
GNP_CODE = ("zgnp", "gnpcols", "zpfp")          # zgnp.OWN_CODE, checked on import
LTM0 = {k: getattr(GNX.LTMC, k) for k in ("OUT", "STAGE", "DECLARED", "COLS", "WIDTH")}    # ltmcols' own, as imported
LOAD_NEXT0 = GNX.LTM.load_next                  # zltm's, as imported
GNX0 = {k: getattr(GNX, k) for k in ("code_extra", "term_record", "next_block")}         # zgnx's own, as zgnp takes them
assert len(PICK_READS) == 6 and set(MOD) == set(ORDER) == set(RULE_OF) == set(S0.STAGES) and GNP_KEY not in S0.STAGES
assert set(LATE + A5) == set(LATE_NAMES) == set(LATE_RULE) and not set(LATE + A5) & (set(S0.STAGES) | {GNP_KEY, "gna"})
assert len(TEN_READS) == len(set(TEN_READS)) == 10 and set(PICK_READS) < set(TEN_READS) and set(ATTN) < set(LATE + A5)
assert Path(RTR.REOPEN) == Path(OUT) / "reopen.json" and RTR.STK_DECLARED == DECLARED      # A5: the record RTR reads


def fit_name(seed, split):
    return f"scr-stk-{TAG}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    return f"scr-stk-{TAG}-vs-base-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 6 and not set(FITS) & (set(M1.ALL_FITS) | set(ISO.FITS) | {f for m in MOD.values() for f in m.FITS})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zstk: {msg}", flush=True)


def pairs(call):
    return {pair_name(sd, sp): (fit_name(sd, sp), DEP.base_fit(sd, sp, call)) for sd in RULE.SEEDS for sp in M1.SPLITS}


def utc_s(s):
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rules_of(mem):
    """The rules the stack is read under: each member's (section 4; A4: GNW's and LTX's LTG's), in the order dep, ltg."""
    rule = {**RULE_OF, **LATE_RULE}
    return [r for r in ("dep", "ltg") if any(rule[st] == r for st in mem)]


def layers_of(mem):
    """Section 10: four layers when DEP is a member (LTG's term in each), three otherwise."""
    return DEP.LAYERS if "dep" in mem else DEP.BASE_LAYERS


def ltg_terms(layers, axes=LTG.AXES):
    """zltg.n_term at a layer count (zltg's own formula, its LAYERS set for the call)."""
    with M1.WQ.patch(LTG, LAYERS=int(layers)):
        return LTG.n_term(axes)


assert ltg_terms(LTG.LAYERS) == LTG.N_TERM == 68745 and LTG.LAYERS == 3


# ── A3: GNP's names, and zgnp on first use ───────────────────────────────────

_GNP = []


def gnp():
    """zgnp, imported on first use (a fit, the calls, the self-test; never the gates): torch before it, and the thread
    variables gnpcols sets at its import put back after, so a stack process keeps the threads it had; its names, columns
    and width checked against A3's."""
    if not _GNP:
        import torch  # noqa: F401  (before gnpcols' thread defaults)
        saved = {v: os.environ.get(v) for v in THREAD_VARS}
        try:
            import zgnp
        finally:
            for v, x in saved.items():
                if x is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = x
        if ((zgnp.KEY, zgnp.OUT, zgnp.DECLARED, zgnp.TAG, tuple(zgnp.GNPC.COLS), zgnp.IN, zgnp.fit_name(0, S0.HP_SPLIT),
             tuple(zgnp.OWN_CODE)) != (GNP_KEY, GNP_OUT, GNP_DECLARED, GNP_TAG, PAIR_COLS, 2 * len(PAIR_COLS) + 1,
                                       gnp_fit_name(0, S0.HP_SPLIT), GNP_CODE)):
            raise SystemExit("zstk: zgnp's names, columns, width or code are not A3's")
        if zgnp.ORIG != GNX0 or zgnp.GNX_FITS != dict(GNX.FITS):
            raise SystemExit("zstk: zgnp was imported inside a patch of zgnx's (its ORIG or GNX_FITS are not zgnx's own)")
        _GNP.append(zgnp)
    return _GNP[0]


_LATE = {}


def late_mods():
    """A4: zgnw and zltx, for the self-test alone (never the gates, never a fit of A4's): imported after zgnp (zgnw imports
    it), outside every patch, with the thread variables gnwcols sets at its import put back; their names, fits and stop
    entries checked against A4's."""
    if not _LATE:
        gnp()
        saved = {v: os.environ.get(v) for v in THREAD_VARS}
        try:
            import zgnw
            import zltx
        finally:
            for v, x in saved.items():
                if x is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = x
        for st, m in (("gnw", zgnw), ("ltx", zltx)):
            late_check(st, m)
        _LATE.update(gnw=zgnw, ltx=zltx)
    return _LATE


def late_check(st, m):
    """A4, A5: a later form's module: its names, fits and stop entry the stack's (LATE_NAMES); imported outside zgnx's
    patches."""
    mod, out, tag, decl = LATE_NAMES[st]
    fits = {late_fit_name(st, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
    if (m.__name__, m.KEY, m.OUT, m.TAG, m.DECLARED, set(m.FITS)) != (mod, st, out, tag, decl, fits):
        raise SystemExit(f"zstk: {mod}'s names or fits are not the stack's")
    if m.ORIG != {k: getattr(GNX, k) for k in m.ORIG} or m.GNX_FITS != dict(GNX.FITS):
        raise SystemExit(f"zstk: {mod} was imported inside a patch of zgnx's")
    with m.on_s0stop():
        if S0.STAGES[st] != late_entry(st):
            raise SystemExit(f"zstk: {mod}'s seed-0 stop entry is {S0.STAGES[st]}, the stack's {late_entry(st)}")


_A5 = {}


def a5_mods():
    """A5: zgnr, zgnt and zgns, for the self-test alone (never the gates, never a fit), as late_mods takes A4's: imported
    after A4's (zgnt and zgns import zgnw), outside every patch, with the thread variables gnrcols and gnscols set at their
    import put back; their names, fits and stop entries checked (late_check)."""
    if not _A5:
        late_mods()
        saved = {v: os.environ.get(v) for v in THREAD_VARS}
        try:
            import zgnr
            import zgnt
            import zgns
        finally:
            for v, x in saved.items():
                if x is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = x
        for st, m in (("gnr", zgnr), ("gnt", zgnt), ("gns", zgns)):
            late_check(st, m)
        _A5.update(gnr=zgnr, gnt=zgnt, gns=zgns)
    return _A5


def gnp_fit_name(seed, split):
    """zgnp.fit_name, without importing zgnp."""
    return f"scr-gnp-{GNP_TAG}-s{seed}{M1.WQ.SUF[split]}"


def gnp_entry():
    """s0stop's entry for GNP, as zgnp.on_s0stop patches it in."""
    return ("zgnp", GNP_OUT, gnp_fit_name(0, S0.S0_SPLIT), gnp_fit_name(0, S0.HP_SPLIT), "ltg", GNP_DECLARED)


# ── the members (section 2) ──────────────────────────────────────────────────


def stage_verdict(st, base=ROOT):
    """A stage's screen verdict as s0stop's gate reads it: NOT_ADOPTED on a current STOP, else its calls.json's
    screen.verdict; None when that file is missing or not the stage's."""
    rec = S0.stop_record(st, base)
    if rec is not None and rec.get("decision") == S0.STOP:
        return NOT_ADOPTED
    p = Path(base) / MOD[st].OUT / "calls.json"
    c = M1.WQ.read_json(p) if p.is_file() else {}
    if c.get("declared_in") != MOD[st].DECLARED:
        return None
    return (c.get("screen") or {}).get("verdict")


def six_deltas(st, base=ROOT, cols="LTM"):
    """{read: seed-0 delta} of the stage's screen on section 6's six WQ-hotpotqa reads (its calls.json's rows); the gnx
    slot on LTA's columns reads GNA's (A2); the form gnp GNP's (A3); the forms gnw and ltx their own (A4)."""
    out = (GNP_OUT if st == "gnp" else LATE_NAMES[st][1] if st in LATE_NAMES
           else GNA.OUT if st == "gnx" and next_part(cols) == "LTA" else MOD[st].OUT)
    p = Path(base) / out / "calls.json"
    c = M1.WQ.read_json(p) if p.is_file() else {}
    rows = {(r.get("split"), r.get("read")): r for r in (c.get("screen") or {}).get("reads") or []}
    return {f"{sp} {rd}": rows[(sp, rd)]["delta"] for sp, rd in PICK_READS if (sp, rd) in rows}


def mean_delta(st, base=ROOT, cols="LTM"):
    """The mean seed-0 delta over the six reads, None when one is missing."""
    d = six_deltas(st, base, cols)
    return round(sum(d.values()) / len(d), 6) if len(d) == len(PICK_READS) else None


def ten_mean(st, base=ROOT):
    """A5: an attention form's (GNW's, GNT's or GNS's) mean seed-0 delta over LTG's ten deciding reads (its calls.json's
    rows), None when one is missing."""
    p = Path(base) / LATE_NAMES[st][1] / "calls.json"
    c = M1.WQ.read_json(p) if p.is_file() else {}
    rows = {(r.get("split"), r.get("read")): r for r in (c.get("screen") or {}).get("reads") or []}
    d = [rows[k]["delta"] for k in TEN_READS if k in rows]
    return round(sum(d) / len(d), 6) if len(d) == len(TEN_READS) else None


def one_attention(forms, base=ROOT):
    """A5: (the forms with one attention form at most, each passing attention form's mean over the ten reads): the
    largest mean is kept, a tie to the earlier in ATTN; a form missing one of the ten is kept only when none has all ten,
    and then the earliest."""
    att = [st for st in ATTN if st in forms]
    means = {st: ten_mean(st, base) for st in att}
    if len(att) < 2:
        return list(forms), means
    ok = [st for st in att if means[st] is not None]
    best = max((means[st] for st in ok), default=None)
    keep = next((st for st in ok if means[st] == best), att[0])
    return [st for st in forms if st not in att or st == keep], means


def dep_or_rounds(forms, base=ROOT, cols="LTM"):
    """A5: (the forms with DEP or the rounds, not both; the two means): when DEP and IRA (or RTR, IRA not passing) are
    both forms, the larger mean seed-0 delta over section 6's six reads is kept, a tie (or a missing mean) to DEP."""
    rnd = next((st for st in ROUNDS_OF if st in forms), None)
    if "dep" not in forms or rnd is None:
        return list(forms), {}
    md, mr = mean_delta("dep", base, cols), mean_delta(rnd, base, cols)
    keep = rnd if mr is not None and (md is None or mr > md) else "dep"
    return [st for st in forms if st not in ("dep", rnd) or st == keep], {"dep": md, rnd: mr}


def reopened(vl):
    """A5: (RTR and IRA open, the call is final) from GNR's and GNS's verdicts: open once either is ADOPT_FOR_FULL_RUN;
    final once open or once both are verdicts."""
    second = [vl.get(st) for st in SECOND]
    is_open = any(x == ADOPT_FOR_FULL_RUN for x in second)
    return is_open, is_open or all(x in VERDICTS for x in second)


def pick_of(passing, base=ROOT, cols="LTM"):
    """(the pick, each passing form's mean delta): the largest mean seed-0 delta over the six reads; a tie goes to the
    earlier form in FORMS; a form missing one of the six is not picked. The gnx slot's is its NEXT columns' (A2); GNP's
    its own (A3)."""
    means = {st: mean_delta(st, base, cols) for st in passing}
    ok = [st for st in FORMS if means.get(st) is not None]
    best = max((means[st] for st in ok), default=None)
    return next((st for st in ok if means[st] == best), None), means


def gna_verdict(base=ROOT):
    """A2: GNA's screen verdict as s0stop's gate reads it (GNA's entry patched in): NOT_ADOPTED on a current STOP, else its
    calls.json's screen.verdict; None when that file is missing or not GNA's."""
    rec = GNA.stop_record(base)
    if rec is not None and rec.get("decision") == S0.STOP:
        return NOT_ADOPTED
    p = Path(base) / GNA.OUT / "calls.json"
    c = M1.WQ.read_json(p) if p.is_file() else {}
    if c.get("declared_in") != GNA.DECLARED:
        return None
    return (c.get("screen") or {}).get("verdict")


def gnp_verdict(base=ROOT):
    """A3: GNP's screen verdict as s0stop's gate reads it (GNP's entry patched in, as zgnp.on_s0stop does; zgnp is not
    imported): NOT_ADOPTED on a current STOP, else its calls.json's screen.verdict; None when that file is missing or not
    GNP's."""
    with M1.WQ.patch(S0, STAGES={**S0.STAGES, GNP_KEY: gnp_entry()}):
        rec = S0.stop_record(GNP_KEY, base)
    if rec is not None and rec.get("decision") == S0.STOP:
        return NOT_ADOPTED
    p = Path(base) / GNP_OUT / "calls.json"
    c = M1.WQ.read_json(p) if p.is_file() else {}
    if c.get("declared_in") != GNP_DECLARED:
        return None
    return (c.get("screen") or {}).get("verdict")


def late_fit_name(st, seed, split):
    """A4: zgnw.fit_name and zltx.fit_name, without importing them."""
    return f"scr-{st}-{LATE_NAMES[st][2]}-s{seed}{M1.WQ.SUF[split]}"


def late_entry(st):
    """A4: s0stop's entry for GNW or LTX, as zgnw.on_s0stop and zltx.on_s0stop patch it in."""
    mod, out, _tag, decl = LATE_NAMES[st]
    return (mod, out, late_fit_name(st, 0, S0.S0_SPLIT), late_fit_name(st, 0, S0.HP_SPLIT), LATE_RULE[st], decl)


def late_verdict(st, base=ROOT):
    """A4: GNW's or LTX's screen verdict as s0stop's gate reads it (its entry patched in; its module not imported):
    NOT_ADOPTED on a current STOP, else its calls.json's screen.verdict; None when that file is missing or not the
    stage's."""
    with M1.WQ.patch(S0, STAGES={**S0.STAGES, st: late_entry(st)}):
        rec = S0.stop_record(st, base)
    if rec is not None and rec.get("decision") == S0.STOP:
        return NOT_ADOPTED
    _m, out, _t, decl = LATE_NAMES[st]
    p = Path(base) / out / "calls.json"
    c = M1.WQ.read_json(p) if p.is_file() else {}
    if c.get("declared_in") != decl:
        return None
    return (c.get("screen") or {}).get("verdict")


def fix_members(base=ROOT, now=None):
    """(done, record) by section 2: each stage's verdict; RTR and IRA will not run when DEP's screen passes; the members
    are the passing forms in ORDER, IRA standing for RTR when both pass; a stage with no verdict at MEMBERS_BY is left out.
    A2: GNA's verdict is waited on too; the gnx slot passes when GNX's or GNA's screen passes, and its NEXT columns are
    LTA's when GNA's passes and GNX's does not, or GNA's mean delta over the six reads is the larger (a tie: LTM's).
    A3: GNP's verdict is waited on too; when GNP's screen passes, GNP is a form, and the gnx slot is a member reading its
    NEXT columns then the six pair columns (the slot passing) or the six alone (it not passing). The pick is over the
    passing forms, a tie to the earlier in FORMS. A4: GNW's and LTX's verdicts are waited on too; each passing one is a
    form and a member of its own, after ORDER's. A5: GNR's, GNT's and GNS's verdicts are waited on too, each passing one a
    form and a member of its own, after A4's; one attention form at most (one_attention); RTR and IRA also run when GNR's
    or GNS's screen is ADOPT_FOR_FULL_RUN (reopened), and when DEP's passes they wait on that call; DEP or the rounds,
    not both (dep_or_rounds); GNR and GNS are never the pick (NO_OWN: their columns are per seed)."""
    now = time.time() if now is None else now
    late = now >= utc_s(MEMBERS_BY)
    v = {st: stage_verdict(st, base) for st in ORDER}
    vg, vp = gna_verdict(base), gnp_verdict(base)
    vl = {st: late_verdict(st, base) for st in LATE + A5}
    is_open, final = reopened(vl)
    wont = [st for st in ROUNDS_OF[::-1] if v["dep"] == ADOPT_FOR_FULL_RUN and final and not is_open]
    pending = ([st for st in ORDER if st not in wont and v[st] not in VERDICTS] + ([] if vg in VERDICTS else ["gna"])
               + ([] if vp in VERDICTS else ["gnp"]) + [st for st in LATE + A5 if vl[st] not in VERDICTS])
    rec = {"stage": STAGE, "declared_in": DECLARED, "verdicts": v, "gna_verdict": vg, "gnp_verdict": vp,
           **{f"{st}_verdict": vl[st] for st in LATE + A5}, "rounds_reopened": is_open, "will_not_run": wont,
           "pending": pending, "members_by": MEMBERS_BY}
    if pending and not late:
        return False, rec
    gnx_ok, gna_ok, gnp_ok = v["gnx"] == ADOPT_FOR_FULL_RUN, vg == ADOPT_FOR_FULL_RUN, vp == ADOPT_FOR_FULL_RUN
    m_ltm, m_lta = mean_delta("gnx", base, "LTM"), mean_delta("gnx", base, "LTA")
    nxt = "LTA" if gna_ok and (not gnx_ok or (m_lta is not None and (m_ltm is None or m_lta > m_ltm))) else "LTM"
    cols = (f"{nxt}+{PAIR}" if gnx_ok or gna_ok else PAIR) if gnp_ok else nxt
    passing = [st for st in ORDER if st not in wont and (v[st] == ADOPT_FOR_FULL_RUN or (st == "gnx" and gna_ok))]
    passing += ["gnp"] if gnp_ok else []
    passing += [st for st in LATE + A5 if vl[st] == ADOPT_FOR_FULL_RUN]
    forms = [st for st in passing if not (st == "rtr" and "ira" in passing)]
    forms, att_means = one_attention(forms, base)
    forms, dep_means = dep_or_rounds(forms, base, nxt)
    mem = [st for st in MEMBERS if st in forms or (st == "gnx" and "gnp" in forms)]
    pick, means = pick_of([st for st in passing if st not in NO_OWN], base, nxt)
    rec.update(left_out=pending, passing=passing, forms=forms, members=mem, layers=layers_of(mem), rules=rules_of(mem),
               pick=pick, pick_means=means, pick_reads=[f"{a} {b}" for a, b in PICK_READS], gnx_columns=cols,
               gnx_means={"LTM": m_ltm, "LTA": m_lta}, attention_means=att_means, dep_or_rounds_means=dep_means,
               utc=utc_now())
    return True, rec


def forms_of(rec):
    """A3: the fixed forms (a record from before A3: its members)."""
    rec = rec or {}
    return list(rec.get("forms", rec.get("members")) or [])


def members_record(base=ROOT):
    """outputs/stk/members.json once written (fixed), else None; None too when its forms, members and gnx columns do not
    agree (A3: GNP's form is the gnx slot with the pair columns; the slot's NEXT set is there when GNX's form is; A4:
    GNW's and LTX's forms are members of their own; A5: so are GNR's, GNT's and GNS's, with one attention form at most,
    and DEP never with the rounds)."""
    p = Path(base) / OUT / "members.json"
    rec = M1.WQ.read_json(p) if p.is_file() else {}
    mem, forms, cols = rec.get("members"), rec.get("forms", rec.get("members")), rec.get("gnx_columns", "LTM")
    if (rec.get("declared_in") != DECLARED or not isinstance(mem, list) or not isinstance(forms, list)
            or [s for s in mem if s not in MEMBERS] or [s for s in forms if s not in FORMS] or cols not in COLUMNS):
        return None
    if (sorted({"gnx" if s == "gnp" else s for s in forms}) != sorted(mem) or ("gnp" in forms) != has_pair(cols)
            or ("gnp" in forms and ("gnx" in forms) != (next_part(cols) is not None))):
        return None
    if len([s for s in forms if s in ATTN]) > 1 or ("dep" in forms and any(s in forms for s in ROUNDS_OF)):
        return None
    return rec


def gnx_columns(rec):
    """A2, A3: whose columns the fixed members' gnx slot reads."""
    return (rec or {}).get("gnx_columns", "LTM")


def col_parts(cols):
    """A3: the gnx slot's column sets, in block order."""
    return cols.split("+")


def next_part(cols):
    """A3: the slot's NEXT set (LTM's or LTA's); None for the pair columns alone."""
    p = col_parts(cols)[0]
    return p if p in ("LTM", "LTA") else None


def has_pair(cols):
    return col_parts(cols)[-1] == PAIR


def slot_columns(cols):
    """A3: the slot's column names in block order (LTA's carry LTM's names: zlta.on_cols keeps them)."""
    return [c for p in col_parts(cols) for c in (PAIR_COLS if p == PAIR else LTM0["COLS"])]


def slot_in(cols):
    """A3: the slot's input width, [raw, z] of each column and one shared 1."""
    return 2 * len(slot_columns(cols)) + 1


def next_declared(cols):
    """A2, A3: the declaration a gnx slot's NEXT block names: LTM's, LTA's or GNP's columns'; with two sets, both, in block
    order."""
    d = [{"LTM": GNX.LTM.DECLARED, "LTA": GNA.LTAC.DECLARED, PAIR: GNP_DECLARED}[p] for p in col_parts(cols)]
    return d[0] if len(d) == 1 else d


@contextlib.contextmanager
def on_set(part):
    """A3: ltmcols on one column set, so zltm.load_next reads it: LTM's as imported, LTA's (zlta.on_cols) or GNP's
    (zgnp.on_cols: its root, stage, declaration, six names and width); restored after, also on a failure."""
    with M1.WQ.patch(GNX.LTMC, **LTM0):
        if part == PAIR:
            with gnp().on_cols():
                yield
        elif part == "LTA":
            with GNA.ZLA.on_cols():
                yield
        else:
            yield


def set_root(part):
    with on_set(part):
        return M1.WQ.rel(Path(GNX.LTMC.OUT))


def both_loader(cols):
    """A3: zltm.load_next for a slot of two sets: each set's columns of the carve (under on_set, out[part] when out is
    given), checked by zltm's own load, joined in block order; one record naming both, the scorer the sets' joined."""
    parts = col_parts(cols)

    def load_next(ds, carve, split, part_shas, out=None, verify=True):
        got = []
        for part in parts:
            with on_set(part):
                got.append((part, *LOAD_NEXT0(ds, carve, split, part_shas, None if out is None else out[part], verify)))
        rows = {int(x.shape[0]) for _p, x, _r, _s in got}
        train = {bool(r["training_carve"]) for _p, _x, r, _s in got}
        if len(rows) != 1 or len(train) != 1:
            raise SystemExit(f"zstk: {ds}/{carve}'s {cols} columns hold {sorted(rows)} rows, training {sorted(train)}")
        X = np.concatenate([x for _p, x, _r, _s in got], axis=1)
        rec = {"stage": STAGE, "declared_in": next_declared(cols), "split": split, "dataset": ds, "carve": carve,
               "columns": slot_columns(cols), "training_carve": train.pop(), "rows": rows.pop(),
               "fit_sha256": "+".join(r["fit_sha256"] for _p, _x, r, _s in got),
               "sets": {p: {"declared_in": r["declared_in"], "fit_sha256": r["fit_sha256"], "record_sha256": s}
                        for p, _x, r, s in got}}
        return X, rec, "+".join(s for _p, _x, _r, s in got)
    return load_next


def both_block(cols):
    """A3: zgnx's NEXT block for a slot of two sets (inside on_split): both sets' declarations and roots."""
    def next_block(split):
        return {"declared_in": next_declared(cols), "split": split, "columns": slot_columns(cols),
                "fit_sha256": GNX.CUR["fit_sha256"],
                "carves": {f"{d['dataset']}/{d['carve']}": d["record_sha256"] for d in GNX.LOADED},
                "training_carves": GNX.train_carves(split), "columns_root": [set_root(p) for p in col_parts(cols)]}
    return next_block


def both_term(cols):
    """A3: zgnx's term record for a slot of two sets."""
    n = slot_in(cols)

    def term_record(G):
        return {"term": f"{next_part(cols)}'s NEXT columns then GNP's six pair columns, one block, into zg6r's encoder "
                        "([raw, z, 1] through gnx_w, zero-started, added to l1's output)",
                "columns": slot_columns(cols), "inputs": n, "term_parameters": GNX.HIDDEN * n,
                "parameters_h128": int(G.N_PARAMS + GNX.HIDDEN * n), "layers": int(G.LAYERS)}
    return term_record


@contextlib.contextmanager
def on_columns(cols):
    """A2: the gnx slot on its columns. LTM's: zgnx as it is. LTA's: ltmcols' root, stage and declaration LTA's
    (zlta.on_cols), and zgnx's NEXT block and term record GNA's. A3: GNP's alone: ltmcols GNP's (zgnp.on_cols), and zgnx's
    NEXT block, term record and width GNP's (as GNP was screened); two sets: ltmcols LTM's, zltm.load_next both sets'
    joined, and zgnx's NEXT block, term record and width the slot's (IN = 27). In this process; restored after, also on a
    failure."""
    if cols not in COLUMNS:
        raise SystemExit(f"zstk: the gnx slot's columns are {cols}, not one of {COLUMNS}")
    with contextlib.ExitStack() as es:
        if cols == "LTA":
            es.enter_context(GNA.ZLA.on_cols())
            es.enter_context(M1.WQ.patch(GNX, next_block=GNA.next_block, term_record=GNA.term_record))
        elif cols == PAIR:
            P = gnp()
            es.enter_context(P.on_cols())
            es.enter_context(M1.WQ.patch(GNX, next_block=P.next_block, term_record=P.term_record, IN=P.IN, WIDTH=P.WIDTH))
        elif has_pair(cols):
            gnp()
            w = len(slot_columns(cols))
            es.enter_context(M1.WQ.patch(GNX.LTMC, **LTM0))
            es.enter_context(M1.WQ.patch(GNX.LTM, load_next=both_loader(cols)))
            es.enter_context(M1.WQ.patch(GNX, next_block=both_block(cols), term_record=both_term(cols), IN=2 * w + 1,
                                         WIDTH=w))
        yield


def missing_columns(cols, split):
    """A3: the training and s1eval carves of the split some set of the slot's has no columns of (its record.json)."""
    miss = []
    for part in col_parts(cols):
        with on_set(part):
            miss += [f"{part} {k}" for k in GNX.train_carves(split) + GNX.eval_carves()
                     if not (GNX.LTMC.cols_dir(GNX.LTMC.OUT, split, *k.split("/")) / "record.json").is_file()]
    return miss


def members(wait=False, base=ROOT, now=None, sleep=time.sleep):
    """Fix the members once (outputs/stk/members.json is never rewritten). With wait, look again every POLL seconds."""
    seen = None
    while True:
        rec = members_record(base)
        if rec is not None:
            log(f"members: {rec['members']} (fixed {rec.get('utc')}; {OUT}/members.json)")
            return 0
        done, rec = fix_members(base, now)
        if done:
            o = Path(base) / OUT
            o.mkdir(parents=True, exist_ok=True)
            M1.WQ.write_json(o / "members.json", rec)
            log(f"members fixed: {rec['members']}; passing {rec['passing']}; will not run {rec['will_not_run']}; "
                f"left out {rec['left_out']}; pick {rec['pick']} {rec['pick_means']}")
            return 0
        if json.dumps(rec, sort_keys=True) != seen:
            seen = json.dumps(rec, sort_keys=True)
            log(f"members: waiting on {rec['pending']} ({rec['verdicts']})")
        if not wait:
            return 2
        sleep(POLL)


def composed(mem):
    """A4: every member is one this code stacks (COMPOSED)."""
    return all(st in COMPOSED for st in mem)


def composes_now(base=ROOT):
    """A4's `composes`: 0 when this code composes every fixed member, 1 when it does not, 2 when they are not fixed."""
    rec = members_record(base)
    if rec is None:
        log("composes: the members are not fixed")
        return 2
    miss = [st for st in rec["members"] if st not in COMPOSED]
    log(f"composes: members {rec['members']}; {f'not composed: {miss}' if miss else 'each one composed'} "
        f"(this code composes {list(COMPOSED)})")
    return 1 if miss else 0


def disk_composes_rc():
    """A4: the exit code of the zstk.py on disk's `composes`, in a process of its own (a waiting gate runs the code it
    started with)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    pr = subprocess.run([sys.executable, str(Path(__file__).resolve()), "composes"], capture_output=True, text=True,
                        timeout=1800, env=env, cwd=str(ROOT))
    log(f"composes, on disk: exit {pr.returncode} {(pr.stdout or '').strip().splitlines()[-1:]}")
    return pr.returncode


def composed_on_disk():
    """A4: whether the zstk.py on disk now composes the fixed members."""
    return disk_composes_rc() == 0


def gate_record(base=ROOT):
    """A4: outputs/stk/stack_gate.json, written once the stack gate has decided on members this code did not compose;
    None before."""
    p = Path(base) / OUT / "stack_gate.json"
    rec = M1.WQ.read_json(p) if p.is_file() else {}
    return rec if rec.get("declared_in") == DECLARED and isinstance(rec.get("runs"), bool) else None


def stack_gate(base=ROOT, wait=False, now=None, sleep=time.sleep, composes=None):
    """0 with two or more forms (the stack runs), 1 with fewer (no stack, section 3), 2 when the members are not fixed.
    A4: with a member this code does not compose, 0 once the code on disk composes it (a fresh `composes`), looked at
    every POLL seconds with wait, 2 without; 1 when it still does not at MEMBERS_BY. That decision is written once
    (outputs/stk/stack_gate.json), and the own gates read a stack that does not run as the stack's screen NOT_ADOPTED."""
    composes = composed_on_disk if composes is None else composes
    seen = False
    while True:
        rec = members_record(base)
        if rec is None:
            log("stack-gate: the members are not fixed")
            return 2
        ok = len(forms_of(rec)) >= 2
        if not ok or composed(rec["members"]):
            log(f"stack-gate: forms {forms_of(rec)}, members {rec['members']} (the gnx slot on {gnx_columns(rec)}'s "
                f"columns); {'the stack runs' if ok else 'no stack (section 3)'}")
            return 0 if ok else 1
        g = gate_record(base)
        if g is not None:
            log(f"stack-gate: decided at {g.get('utc')}: {'the stack runs' if g['runs'] else 'no stack'} ({g.get('why')})")
            return 0 if g["runs"] else 1
        miss = [st for st in rec["members"] if st not in COMPOSED]
        if composes():
            runs, why = True, f"the code on disk composes {miss}"
        elif (time.time() if now is None else now) >= utc_s(MEMBERS_BY):
            runs, why = False, f"{miss} not composed by the code on disk at {MEMBERS_BY}"
        else:
            if not seen:
                seen = True
                log(f"stack-gate: forms {forms_of(rec)}, members {rec['members']}: waiting for code that composes {miss}")
            if not wait:
                return 2
            sleep(POLL)
            continue
        o = Path(base) / OUT
        o.mkdir(parents=True, exist_ok=True)
        M1.WQ.write_json(o / "stack_gate.json", {"stage": STAGE, "declared_in": DECLARED, "members": rec["members"],
                                                 "forms": forms_of(rec), "runs": runs, "why": why, "utc": utc_now()})
        log(f"stack-gate: {'the stack runs' if runs else 'no stack'} ({why})")
        return 0 if runs else 1


@contextlib.contextmanager
def on_s0stop():
    """s0stop's stage table and rules with the stack's entry, in this process; restored after."""
    with M1.WQ.patch(S0, STAGES={**S0.STAGES, "stk": ("zstk", OUT, fit_name(0, S0.S0_SPLIT), fit_name(0, S0.HP_SPLIT),
                                                       "stk", DECLARED)},
                     RULES={**S0.RULES, "stk": "every rule the stack's members use (docs/STK_GNN_STACK.md section 4)"}):
        yield


def stk_keys(_rule, base=ROOT):
    """The stack's deciding reads of the WQ-musique split: those of every rule its members use."""
    rec = members_record(base)
    if rec is None:
        raise RuntimeError("the members are not fixed")
    keys = []
    for r in rules_of(rec["members"]):
        keys += [k for k in S0.deciding(r) if k not in keys]
    return keys


def stack_verdicts(base=ROOT):
    """(the stack's screen verdict, its full run's): NOT_ADOPTED on a current STOP; None for what is not filed."""
    with on_s0stop():
        rec = S0.stop_record("stk", base)
    if rec is not None and rec.get("decision") == S0.STOP:
        return NOT_ADOPTED, None
    p = Path(base) / OUT / "calls.json"
    c = M1.WQ.read_json(p) if p.is_file() else {}
    if c.get("declared_in") != DECLARED:
        return None, None
    return (c.get("screen") or {}).get("verdict"), (c.get("full_run") or {}).get("verdict")


def own_call(st, base=ROOT, now=None):
    """(done, exit code, why) for stage st's own full run, by sections 3 and 6: with one member, that member's; with two or
    more, the pick's when the stack's screen is NO_GAIN or NOT_ADOPTED or its full run NOT_ADOPTED; none when the stack's
    full run is ADOPT; none still undecided at FREEZE. A2: the gnx slot's own full run is GNX's on LTM's columns and GNA's
    (st "gna") on LTA's. A3: by forms; GNP's own full run (st "gnp") is its form's. A4: GNW's and LTX's are their forms';
    a stack the stack gate decided does not run (a member not composed at MEMBERS_BY) is read as its screen NOT_ADOPTED.
    A5: GNR and GNS have no own full run (NO_OWN), also as the one form; GNT's is its form's."""
    now = time.time() if now is None else now
    rec = members_record(base)
    if rec is None:
        return now >= utc_s(FREEZE), 1, "the members are not fixed"
    forms, slot = forms_of(rec), "gnx" if st == "gna" else st
    mine = slot != "gnx" or (st == "gna") == (next_part(gnx_columns(rec)) == "LTA")
    if len(forms) < 2:
        ok = forms == [slot] and mine and slot not in NO_OWN
        return True, 0 if ok else 1, (f"forms {forms} (the gnx slot on {gnx_columns(rec)}'s columns): "
                                      f"{'its own full run' if ok else 'not its own'}")
    sv, fv = stack_verdicts(base)
    g = gate_record(base)
    if sv is None and g is not None and not g["runs"]:
        sv = NOT_ADOPTED                        # A4: the stack gate decided the stack does not run
    if sv in (NO_GAIN, NOT_ADOPTED) or fv == NOT_ADOPTED:
        return True, 0 if rec.get("pick") == slot and mine else 1, (
            f"the stack's screen {sv}, full run {fv}: the pick is {rec.get('pick')} (the gnx slot on "
            f"{gnx_columns(rec)}'s columns)")
    if fv == ADOPT:
        return True, 1, "the stack's full run is ADOPT"
    if now >= utc_s(FREEZE):
        return True, 1, f"the stack's screen {sv}, full run {fv} at the freeze ({FREEZE})"
    return False, 1, f"the stack's screen {sv}, full run {fv}: waiting"


def own_gate(st, wait=False, base=ROOT, now=None, sleep=time.sleep):
    seen = None
    while True:
        done, rc, why = own_call(st, base, now)
        if done:
            log(f"own-gate {st}: {'released' if rc == 0 else 'not released'} ({why})")
            return rc
        if why != seen:
            seen = why
            log(f"own-gate {st}: {why}")
        if not wait:
            return 2
        sleep(POLL)


def reopen_call(base=ROOT, now=None):
    """A5: (done, exit code, why) for RTR's and IRA's opening (docs/RTR_GO_BACK_ROUNDS.md section 13,
    docs/IRA_ITERATIVE_ALIGNMENT.md section 14), from GNR's and GNS's verdicts as the members read them: 0 once either is
    ADOPT_FOR_FULL_RUN; 1 once both are other verdicts, or at MEMBERS_BY (the rounds' screens could not be read by then)."""
    now = time.time() if now is None else now
    vl = {st: late_verdict(st, base) for st in SECOND}
    is_open, final = reopened(vl)
    if final:
        return True, 0 if is_open else 1, f"GNR's and GNS's screens {vl}: {'open' if is_open else 'closed'}"
    if now >= utc_s(MEMBERS_BY):
        return True, 1, f"GNR's and GNS's screens {vl} at {MEMBERS_BY}: closed"
    return False, 2, f"GNR's and GNS's screens {vl}: waiting"


def reopen_gate(wait=False, base=ROOT, now=None, sleep=time.sleep):
    """A5: reopen_call until done (with wait), then outputs/stk/reopen.json, the record zrtr.dep_call reads (open: true
    lets RTR's and IRA's fits run on DEP's passing screen)."""
    seen = None
    while True:
        done, rc, why = reopen_call(base, now)
        if done:
            o = Path(base) / OUT
            o.mkdir(parents=True, exist_ok=True)
            M1.WQ.write_json(o / "reopen.json", {"stage": STAGE, "declared_in": DECLARED, "open": rc == 0, "why": why,
                                                 "verdicts": {st: late_verdict(st, base) for st in SECOND},
                                                 "utc": utc_now()})
            log(f"reopen-gate: RTR and IRA {'open' if rc == 0 else 'stay closed'} ({why})")
            return rc
        if why != seen:
            seen = why
            log(f"reopen-gate: {why}")
        if not wait:
            return 2
        sleep(POLL)


# ── how the members compose (section 10) ─────────────────────────────────────


@contextlib.contextmanager
def on_ltg_layers(G):
    """zltg's per-layer names and counts (LAYERS, TERM_LAYER, TERM_NAMES, N_TERM) at zg6r's layer count, read when a ZG6R is
    built and in each forward: four with DEP (the term in each of its layers), zltg's own three otherwise; restored after."""
    L = int(G.LAYERS)
    tl = tuple(tuple(f"ltg_{k}_{p}" for p in ("w", "g", "em", "ea")) for k in range(L))
    names = ("ltg_V", "ltg_U") + tuple(n for lay in tl for n in lay)
    with M1.WQ.patch(LTG, LAYERS=L, TERM_LAYER=tl, TERM_NAMES=names, N_TERM=ltg_terms(L)):
        if LTG.TERM_LAYER[:LTG_OWN] != LTG_TERM_LAYER0:
            raise SystemExit(f"zstk: zltg's per-layer names at {L} layers do not start with its own three")
        yield


LTG_OWN, LTG_TERM_LAYER0 = LTG.LAYERS, LTG.TERM_LAYER     # zltg's own, as imported


@contextlib.contextmanager
def on_term_rounds(G):
    """LTG's term under RTR's or IRA's stack (entered before): a ZG6R built gets the term after the inits before it
    (zltg.add_term); ZG6R's forward hands the store block and keep flags to the layers; zg6r's g6r_layers, which the rounds
    call by name once a round, is zltg.term_layers, with the term's inputs (zltg.term_inputs: P, Q, the out-degrees, the
    flags) taken once a forward on the kept edges, since none depends on the state; in training the stack leaves its last
    round's score, the plain edge score and the kept edges for the next-hop loss, as zltg.term_stack does. Restored after,
    also on a failure."""
    orig_init, orig_fwd, orig_stack = G.ZG6R.__init__, G.ZG6R.forward, G.g6r_stack
    if getattr(orig_stack, "__module__", None) not in ("zrtr", "zira"):
        raise SystemExit("zstk: LTG's term in the rounds needs RTR's or IRA's stack entered before it")
    TI = {}

    def init(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        orig_init(self, blocks, widths, hidden, dropout, seed, arm, ctx)
        LTG.add_term(G, self, seed)

    def forward(self, feats, keep, nq, B, base_z):
        if "SEMB" not in feats:
            raise RuntimeError("zstk: no SEMB block in the batch (LTG's term reads the store's rows)")
        LTG.CUR["in"] = (feats["SEMB"], keep)
        TI.clear()
        try:
            return orig_fwd(self, feats, keep, nq, B, base_z)
        finally:
            LTG.CUR.pop("in", None)
            TI.clear()

    def layers(m, h, xb, w, u, v, f, nq, B, te, codes, cte):
        if "in" not in LTG.CUR:
            raise RuntimeError("zstk: g6r_layers called outside ZG6R's forward")
        if TI.get("u") is not u:
            semb, keep = LTG.CUR["in"]
            TI.update(u=u, v=v, n=int(h.shape[0]), t=LTG.term_inputs(G, m, semb, keep, nq, u, h.shape[0]))
        return LTG.term_layers(G, m, h, xb, w, u, v, f, nq, B, te, codes, cte, *TI["t"])

    def stack(m, H, xb, lk, gd, nq, B, u01=None):
        out = orig_stack(m, H, xb, lk, gd, nq, B, u01)
        if m.training and G.torch.is_grad_enabled():
            if "t" not in TI:
                raise RuntimeError("zstk: the rounds ran no layer of the term")
            P, Q = TI["t"][0], TI["t"][1]
            LTG.CUR["aux"] = {"out": out, "sp": LTG.edge_dot(G).apply(P, Q, TI["v"], TI["u"], LTG.CHUNK), "u": TI["u"],
                              "v": TI["v"], "n": TI["n"]}
        return out

    try:
        with M1.WQ.patch(G, g6r_stack=stack, g6r_layers=layers), M1.WQ.patch(G.ZG6R, __init__=init, forward=forward):
            yield
    finally:
        LTG.CUR.clear()
        TI.clear()


@contextlib.contextmanager
def on_members(G, mem, carve=None):
    """The members' patches in section 10's order: IRA's model or RTR's rounds; LTG's layer count, term and next-hop loss;
    GNX's model and carve (carve: the carve class GNX patches, zg6r's G6rLinkCarve when None). DEP is zg6r's import."""
    rounds = [st for st in ROUNDS_OF if st in mem][:1]
    if rounds and "dep" in mem:
        raise SystemExit("zstk: RTR and IRA run only when DEP's screen fails (section 2); never in one stack with DEP")
    if int(G.LAYERS) != layers_of(mem):
        raise SystemExit(f"zstk: zg6r holds {G.LAYERS} layers, the stack of {list(mem)} {layers_of(mem)}")
    with contextlib.ExitStack() as es:
        if rounds:
            es.enter_context(IRA.on_model(G) if rounds == ["ira"] else RTR.on_rounds(G))
        if "ltg" in mem:
            es.enter_context(on_ltg_layers(G))
            es.enter_context(on_term_rounds(G) if rounds else LTG.on_term(G))
            es.enter_context(LTG.on_aux(G))
        if "gnx" in mem:
            es.enter_context(GNX.on_model(G))
            es.enter_context(GNX.on_carve(G, carve))
        yield


def member_names(G, st):
    """Each member's own parameters (DEP's: its fourth layer's), for the self-test's gradient check."""
    if st == "dep":
        return tuple(G.G6.G6_LAYER[-1]) + tuple(G.G6R_LAYER[-1])
    return {"ltg": LTG.TERM_NAMES, "rtr": ("rtr_e",), "ira": IRA.IRA_NAMES, "gnx": ("gnx_w",)}[st]


def stack_record(G, mem, dep=None):
    """The stamp's record of the stack: the members, the layer count, each member's own record (its module's, at the stack's
    layers), the parameter counts, and DEP's call under RTR's condition. Inside on_members (LTG's counts at the layers)."""
    parts, extra = {}, 0
    if "dep" in mem:
        parts["dep"] = DEP.depth_record()
    if "ltg" in mem:
        parts["ltg"] = LTG.term_record(G)
        extra += parts["ltg"]["term_parameters"]
    if "rtr" in mem:
        parts["rtr"] = RTR.rounds_record(G)
        extra += int(G.D)
    if "ira" in mem:
        parts["ira"] = IRA.ira_record(G)
        extra += IRA.n_ira(G)
    if "gnx" in mem:
        parts["gnx"] = GNX.term_record(G)
        extra += GNX.HIDDEN * GNX.IN
    return {"members": list(mem), "layers": int(G.LAYERS), "parts": parts, "stack_parameters": int(extra),
            "parameters_h128": int(G.N_PARAMS + extra), "dep_call": dep}


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def with_fits():
    """STK's names in zml1s's fit table (in place, so on_ml1's FITS sees them); removed afterwards."""
    M1.ALL_FITS.update(FITS)
    try:
        yield
    finally:
        for k in FITS:
            M1.ALL_FITS.pop(k, None)


def stk_fit_name(arm, seed, split):
    """zml1s.pko_fit_name with the stack's fit being run answering for its own (arm, seed, split)."""
    a = M1._ACTIVE[0]
    if a in FITS and FITS[a] == (arm, seed, split):
        return a
    return M1.fit_name(arm, seed, split)


def stamp_with_members(orig, call, mem, rec):
    """zml1s.stamp with the stack's record, the base and ISO's verdict; with LTG a train's stamp holds the next-hop curve;
    with GNX the NEXT block, and a train that did not load every training carve's columns, or a read every s1eval
    carve's, stops unstamped (GNX's condition)."""
    def stamp(name, env, what, extra=None):
        if name not in FITS:
            return orig(name, env, what, extra)
        _a, sd, sp = FITS[name]
        add = {**rec, "base": DEP.base_fit(sd, sp, call), "iso_verdict": call}
        if "gnx" in mem:
            nxt = GNX.next_block(sp)
            need = GNX.train_carves(sp) if what == "screen" else GNX.eval_carves()
            miss = sorted(set(need) - set(nxt["carves"]))
            if GNX.CUR["split"] != sp or miss or not nxt["fit_sha256"]:
                raise SystemExit(f"zstk: {name}'s {what} loaded no NEXT columns of {miss or need} under {sp}; unstamped")
            add["next"] = nxt
        if "ltg" in mem and what == "screen":
            add["aux"] = LTG.aux_summary()
        return orig(name, env, what, {**(extra or {}), **add})
    return stamp


def train_with(orig, mem):
    """zml1s.train for a stack fit: LTG's curve from empty; with GNX under the fit's split, the carves' loads from empty."""
    def train(name, env):
        if name not in FITS:
            return orig(name, env)
        if "ltg" in mem:
            LTG.aux_reset()
        if "gnx" not in mem:
            return orig(name, env)
        GNX.LOADED.clear()
        try:
            with GNX.on_split(FITS[name][2]):
                return orig(name, env)
        finally:
            GNX.LOADED.clear()
    return train


def read_with(orig, mem):
    """zml1s.read for a stack fit: with GNX under its split, every carve's columns from the scorer its train read."""
    def read(name, env):
        if name not in FITS or "gnx" not in mem:
            return orig(name, env)
        sp = FITS[name][2]
        nk = M1.fit_stamp(name)[1].get("next") or {}
        nb = GNX.next_block(sp)                 # A3: the slot's columns and declaration (LTM's, LTA's, GNP's or two sets')
        if ((nk.get("split"), nk.get("columns"), nk.get("declared_in")) != (sp, nb["columns"], nb["declared_in"])
                or not nk.get("fit_sha256")):
            raise SystemExit(f"zstk: {name} was not trained on these NEXT columns of {sp} ({nk})")
        GNX.LOADED.clear()
        try:
            with GNX.on_split(sp, nk["fit_sha256"]):
                return orig(name, env)
        finally:
            GNX.LOADED.clear()
    return read


def code_extra():
    """The stage's own code and every member's beside zml1s's, by fixed keys (a module run as a script is named __main__)."""
    mods = (("zdep", DEP), ("zltg", LTG), ("zrtr", RTR), ("zira", IRA), ("zgnx", GNX), ("zltm", GNX.LTM),
            ("ltmcols", GNX.LTMC), ("s0stop", S0), ("zgna", GNA), ("zlta", GNA.ZLA), ("ltacols", GNA.LTAC))
    return {"zstk_sha256": M1.WQ.sha_src(__file__), **{f"{k}_sha256": M1.WQ.sha_src(m.__file__) for k, m in mods},
            **{f"{k}_sha256": M1.WQ.sha_src(HERE / f"{k}.py") for k in GNP_CODE}}     # A3: by path, zgnp not imported


@contextlib.contextmanager
def on_names():
    """zml1s with STK's names, stamp key, stage and declaration; restored after. Enough for the compares and the calls."""
    extra, code = {**code_extra(), **MS.code(), **RS.code()}, M1.code_shas
    with with_fits(), M1.WQ.patch(M1, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, pko_fit_name=stk_fit_name,
                                  code_shas=lambda arm=None: {**code(arm), **extra}):
        yield


@contextlib.contextmanager
def on_stack(G, call, mem, dep=None, carve=None):
    """A stack fit's context: STK's names; the members' patches; the stamp, train and read under them; the start (on
    REPLACE ISO's own, on KEEP ML1's zspk start read under ML1's names); a read of p@swa alone. Restored after."""
    own = call == ISO.REPLACE
    with contextlib.ExitStack() as es:
        es.enter_context(on_names())
        es.enter_context(on_members(G, mem, carve))
        rec = stack_record(G, mem, dep)
        es.enter_context(M1.WQ.patch(M1, stamp=stamp_with_members(M1.stamp, call, mem, rec),
                                     load_encoder=ISO.own_load if own else MS.ml1_load,
                                     train=train_with(M1.train, mem), read=read_with(M1.read, mem)))
        if own:
            es.enter_context(M1.WQ.patch(G, encoder_used=ISO.own_used))
        es.enter_context(RS.on_read())
        yield rec


def import_g(mem):
    """zg6r, with four layers when DEP is a member (zdep.import_deep: a process of its own), checked for every member."""
    G = DEP.import_deep()[0] if "dep" in mem else M1.import_arm(ARM)[0]
    if G.ARMS.get(ARM) != (G.ZG6R, G.G6rLinkCarve) or G.LAYERS != layers_of(mem) or (G.D, G.FAMS) != (LTG.D, LTG.FAMS):
        raise SystemExit(f"zstk: zg6r's arm is {G.ARMS.get(ARM)} with {G.LAYERS} layers of {G.D} over {G.FAMS} families, "
                         f"not G6r's with {layers_of(mem)} of {LTG.D} over {LTG.FAMS}")
    if "gnx" in mem and not all(hasattr(G.G6.S3, f) for f in ("seg_zscore_ref", "retrieved")):
        raise SystemExit("zstk: zg6's lean_screen3 lacks seg_zscore_ref or retrieved")
    return G


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zstk: {name} is not one of STK's fits ({', '.join(FITS)})")
    rec = members_record()
    if rec is None or len(forms_of(rec)) < 2:
        raise SystemExit(f"zstk: the stack runs with two or more forms (section 3, A3); forms {rec and forms_of(rec)}")
    mem, cols = rec["members"], gnx_columns(rec)
    if not composed(mem):
        raise SystemExit(f"zstk: this code composes {list(COMPOSED)}, not the members {mem} (A4: amendment A5 declares "
                         f"how GNW and LTX compose)")
    _a, sd, sp = FITS[name]
    if "gnx" in mem and has_pair(cols):
        gnp()                                   # A3: before any patch (zgnp takes zgnx's own at its import)
    with on_columns(cols):
        if "gnx" in mem:
            miss = missing_columns(cols, sp)
            if miss:
                raise SystemExit(f"zstk: {name}: no {cols} columns of {miss} under {sp}")
        dep = RTR.dep_call() if any(st in mem for st in ROUNDS_OF) else None
        call = DEP.iso_call()
        G = import_g(mem)
        if "ltg" in mem:
            LTG.load_basis(G)
            LTG.aux_reset()
        with on_stack(G, call, mem, dep) as srec:
            log(f"{name}: {mem} on {DEP.base_fit(sd, sp, call)}'s start (ISO {call}); {srec['layers']} layers, "
                f"{srec['stack_parameters']} parameters beside zg6r's {G.N_PARAMS}; the gnx slot on {cols}'s columns; "
                f"{srec['parts']}")
            return M1.fit(name)


# ── the compares, the slices and the calls ───────────────────────────────────


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: STK's, or the base's (ISO's or ML1's)."""
    if name in FITS:
        with on_names():
            return M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]
    return DEP.fit_block(name, what, base)


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != M1.ROWS_CAP[ARM]:
        raise SystemExit(f"zstk: {name}'s train and read are not on one device with zg6r's rows cap")
    return ks["device"]


def is_read(name, base=ROOT):
    return (M1.screen_dir(base) / "fits" / name / "read.json").is_file()


def compare_all(base=ROOT):
    """Each stack fit that is read against the base's fit of its seed and split, through zwq's run_compare; a compare filed
    before is kept."""
    call = DEP.iso_call(base)
    done, ran = [], []
    for cname, (new, bse) in pairs(call).items():
        if not is_read(new, base):
            continue
        js = M1.screen_dir(base) / f"{cname}.json"
        if js.is_file() and (M1.WQ.read_json(js).get(KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, bse)}
        if len(dev) != 1:
            raise SystemExit(f"zstk: {new} and {bse} ran on different devices: {sorted(dev)}")
        with ISO.with_fits(), on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zstk: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "iso_verdict": call,
                    "device": json.loads(dev.pop()), "zstk_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def r5_stk(call, base=ROOT):
    """({(split, dataset): {seed: R@5}} of the stack's fits, the same of their bases) from STK's compares."""
    new_r5, base_r5 = {}, {}
    for cname, (new, bse) in pairs(call).items():
        p = M1.screen_dir(base) / f"{cname}.json"
        if not p.is_file():
            continue
        rec = M1.WQ.read_json(p)
        k = rec.get(KEY) or {}
        if (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, new, bse):
            continue
        if rec.get("candidate") != DEP.CAND or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
            continue
        _a, sd, sp = FITS[new]
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            new_r5.setdefault((sp, r["dataset"]), {})[sd] = r["new"][0]
            base_r5.setdefault((sp, r["dataset"]), {})[sd] = r["base"][0]
    return new_r5, base_r5


def slices(call, new_r5, base3, base=ROOT):
    """({(split, slice): {seed: R@5}} of the stack's read fits, the same of the base's three seeds, problems), by
    zdep.slice_r5. Each file's all-question R@5 must be its compare's musique R@5."""
    new_s, base_s, bad = {}, {}, []
    for sp in M1.SPLITS:
        for sd in RULE.SEEDS:
            for f, own, ref in ((fit_name(sd, sp), new_s, new_r5), (DEP.base_fit(sd, sp, call), base_s, base3)):
                if f in FITS and not is_read(f, base):
                    continue
                s = DEP.slice_r5(f, base)
                if s is None:
                    bad.append(f"{f}: no {DEP.SLICE_DS} read file")
                    continue
                want = ref.get((sp, DEP.SLICE_DS), {}).get(sd)
                if want is not None and abs(s["all"] - want) > R5_TOL:
                    bad.append(f"{f}: its read file gives {DEP.SLICE_DS} R@5 {s['all']:.5f}, its compare {want}")
                for sl in DEP.SLICES:
                    if s[sl] is not None:
                        own.setdefault((sp, sl), {})[sd] = s[sl]
    return new_s, base_s, bad


def combine_screen(blocks):
    """Section 4: ADOPT_FOR_FULL_RUN when each rule gives it, NOT_ADOPTED when one gives NOT_ADOPTED, NO_GAIN otherwise;
    INCOMPLETE when a rule is. The rows are the union of the rules' (their per-read call is one formula); beside, the
    rules' beside rows not among them."""
    vs = [b["verdict"] for b in blocks]
    rows, side, bad = union_rows(blocks)
    v = (INCOMPLETE if INCOMPLETE in vs or bad else ADOPT_FOR_FULL_RUN if all(x == ADOPT_FOR_FULL_RUN for x in vs)
         else NOT_ADOPTED if NOT_ADOPTED in vs else NO_GAIN)
    return {"verdict": v, "rules": vs, "problems": [p for b in blocks for p in b.get("problems") or []] + bad,
            "reads": rows, "beside": side,
            "counts": {c: sum(1 for r in rows if r["call"] == c) for c in (RULE.GAIN, RULE.LOSS, RULE.WITHIN)}}


def union_rows(blocks):
    """(the rules' rows, each read once in the rules' order; their beside rows not among them; problems): a read two rules
    both call must get one call from both (the per-read call is one formula in zdep and zltg)."""
    rows, seen, bad = [], {}, []
    for b in blocks:
        for r in b.get("reads") or []:
            k = (r["split"], r["read"])
            if k not in seen:
                seen[k] = r
                rows.append(r)
            elif json.dumps(seen[k], sort_keys=True, default=str) != json.dumps(r, sort_keys=True, default=str):
                bad.append(f"{k[0]} {k[1]}: the rules' rows differ ({seen[k]} and {r})")
    side, sk = [], set()
    for b in blocks:
        for r in b.get("beside") or []:
            k = (r["split"], r["read"])
            if k not in seen and k not in sk:
                sk.add(k)
                side.append(r)
    return rows, side, bad


def combine_full(blocks):
    """Section 4: ADOPT when each rule's full-run call is ADOPT, NOT_ADOPTED otherwise; INCOMPLETE when a rule is. Each
    rule's rows are kept under its name (zdep's and zltg's full-run rows are not one format)."""
    vs = [b["verdict"] for b in blocks]
    v = INCOMPLETE if INCOMPLETE in vs else ADOPT if all(x == ADOPT for x in vs) else NOT_ADOPTED
    return {"verdict": v, "rules": vs, "problems": [p for b in blocks for p in b.get("problems") or []]}


def part_problems(name, what, k, mem, cols="LTM"):
    """A stamp's members and parts against section 10: the members as fixed, the stack's layer count, each member's record
    at it (LTG's term at the stack's layers, RTR's and IRA's rounds, GNX's input: A3, the slot's columns and width)."""
    L, out = layers_of(mem), []
    parts = k.get("parts") or {}
    if (k.get("members"), k.get("layers"), sorted(parts)) != (list(mem), L, sorted(mem)):
        out.append(f"{name}: its {what} names members {k.get('members')} with {k.get('layers')} layers and parts "
                   f"{sorted(parts)}, not {list(mem)} with {L}")
        return out
    D, want = LTG.D, {}
    if "dep" in mem:
        want["dep"] = {"layers": DEP.LAYERS}
    if "ltg" in mem:
        want["ltg"] = {"term_parameters": ltg_terms(L), "rank": LTG.RANK, "axes": LTG.AXES, "basis_npz_sha256": LTG.BASIS_SHA,
                       "lambda": LTG.LAMBDA, "layers": L}
    if "rtr" in mem:
        want["rtr"] = {"rounds": RTR.ROUNDS, "rounds_parameters": D, "layers": L}
    if "ira" in mem:
        want["ira"] = {"rounds": IRA.ROUNDS, "slots": IRA.SLOTS, "sinkhorn_iters": IRA.SINK_ITERS,
                       "ira_parameters": D + 2 * IRA.SLOTS * D + 3, "layers": L}
    if "gnx" in mem:
        n = slot_in(cols)
        want["gnx"] = {"term_parameters": GNX.HIDDEN * n, "inputs": n, "columns": slot_columns(cols), "layers": L}
    for st, w in want.items():
        got = {x: parts[st].get(x) for x in w}
        if got != w:
            out.append(f"{name}: its {what}'s {st} part is {got}, not {w}")
    return out


# Section 10: the one member seed-0 fit trained before its module's last change, which touches the read alone. zltg.py
# before 24385d8 (a fit's read keeps p@swa only, readswa; training unchanged): ltg-nh-s0-b started before it.
EARLIER = {("ltg", "WQ-musique"): "4b1657818d6f6641085821e6e7f4323d13ff240751dee6c9d0c4c63ed562e2d9"}


def own_of(st, cols="LTM"):
    """[(module, the code keys its own fits must share with the stack's train)] for the forms a member holds: its own
    stage's; for the gnx slot, zgnx's on LTM's NEXT columns, zgna's (with zgna's, zlta's and ltacols') on LTA's (A2), and
    zgnp's (with zgnp's, gnpcols' and zpfp's) for the pair columns (A3)."""
    if st != "gnx":
        return [(MOD[st], (f"z{st}_sha256",))]
    out = []
    if next_part(cols) == "LTA":
        out.append((GNA, ("zgnx_sha256",) + tuple(f"{k}_sha256" for k in GNA.OWN_CODE)))
    elif next_part(cols) == "LTM":
        out.append((GNX, ("zgnx_sha256",)))
    if has_pair(cols):
        out.append((gnp(), ("zgnx_sha256",) + tuple(f"{k}_sha256" for k in GNP_CODE)))
    return out


def member_code(name, ks, mem, base=ROOT, cols="LTM"):
    """Section 2's 'as it was screened' (section 10): each member's module in the stack's train stamp is the one both its
    own seed-0 fits' train stamps name, but EARLIER's fit may name EARLIER's sha instead; both fits must be read. A2: the
    gnx slot on LTA's columns was screened as GNA, so its own fits are GNA's, and zgna's, zlta's and ltacols' shas count
    beside zgnx's; an own fit that names no sha for a key fails. A3: with the pair columns GNP's fits too (own_of)."""
    out, mine = [], ks.get("code") or {}
    for st, m, keys in [(st, m, keys) for st in mem for m, keys in own_of(st, cols)]:
        for sp in M1.SPLITS:
            f = m.fit_name(0, sp)
            if not is_read(f, base):
                out.append(f"{name}: {f}, {st}'s own seed-0 fit, is not read")
                continue
            try:
                theirs = m.fit_block(f, "screen", base).get("code") or {}
            except SystemExit as e:
                out.append(str(e))
                continue
            for key in keys:
                a, b = mine.get(key), theirs.get(key)
                if a is None or not (b == a or (b is not None and b == EARLIER.get((st, sp)))):
                    out.append(f"{name}: its {key} is {a}, {f}'s {b}")
    return out


def fit_problems(call, mem, base=ROOT, cols="LTM"):
    """Why a stack fit that is read is not the declared one: its members, layers and parts (part_problems), base and ISO
    verdict; with LTG its next-hop loss having run; with GNX its NEXT block (the split, the columns and their declaration,
    one scorer for train and read, every training carve in its train and every s1eval carve in its read); with RTR or IRA
    DEP's call; each member's code as screened; its start; its device; its trainer's code against the base's fit of its seed
    and split. A2: cols, whose columns the gnx slot reads."""
    out = []
    for name, (_a, sd, sp) in FITS.items():
        if not is_read(name, base):
            continue
        bse = DEP.base_fit(sd, sp, call)
        try:
            ks, kr, kb = fit_block(name, "screen", base), fit_block(name, "read", base), fit_block(bse, "screen", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        for what, k, need in (("train", ks, GNX.train_carves(sp)), ("read", kr, GNX.eval_carves())):
            out += part_problems(name, what, k, mem, cols)
            if (k.get("base"), k.get("iso_verdict")) != (bse, call):
                out.append(f"{name}: its {what} is on {k.get('base')} (ISO {k.get('iso_verdict')}), not {bse}")
            if any(st in mem for st in ROUNDS_OF) and not k.get("dep_call"):
                out.append(f"{name}: its {what} names no DEP call under RTR's condition")
            if "gnx" in mem:
                nk = k.get("next") or {}
                miss = sorted(set(need) - set(nk.get("carves") or {}))
                if ((nk.get("split"), nk.get("columns"), nk.get("declared_in"))
                        != (sp, slot_columns(cols), next_declared(cols)) or miss):
                    out.append(f"{name}: its {what}'s NEXT block is {nk.get('split')} {nk.get('columns')} of "
                               f"{nk.get('declared_in')}, not {cols}'s, missing {miss}")
        if "gnx" in mem and (ks.get("next") or {}).get("fit_sha256") != (kr.get("next") or {}).get("fit_sha256"):
            out.append(f"{name}: its train and read read columns of different scorers")
        if "ltg" in mem and int((ks.get("aux") or {}).get("steps") or 0) <= 0:
            out.append(f"{name}: its train ran no next-hop loss step")
        out += member_code(name, ks, mem, base, cols)
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
        diff = sorted(k for k in cb if k not in CODE_SKIP and cn.get(k) != cb[k])
        if diff:
            out.append(f"{name}: code differs from {bse}'s in {diff}")
    return out


def calls(base=ROOT):
    call = DEP.iso_call(base)
    mrec = members_record(base)
    if mrec is None or len(forms_of(mrec)) < 2:
        raise SystemExit(f"zstk: the stack's calls need two or more fixed forms ({mrec and forms_of(mrec)})")
    mem, cols, forms = mrec["members"], gnx_columns(mrec), forms_of(mrec)
    rules = rules_of(mem)
    new_r5, b0 = r5_stk(call, base)
    base3 = DEP.base_seeds(call, base)
    bad = [f"{rd[0]} {rd[1]} seed {sd}: the base gives {v} in STK's compare, {base3.get(rd, {}).get(sd)} in its own stage's"
           for rd, by in sorted(b0.items()) for sd, v in sorted(by.items())
           if base3.get(rd, {}).get(sd) is None or abs(v - base3[rd][sd]) > R5_TOL]
    new_s, base_s, bad_s = slices(call, new_r5, base3, base)
    bad += bad_s
    new_all, base_all = {**new_r5, **new_s}, {**base3, **base_s}
    full_ready = all(is_read(fit_name(sd, sp), base) for sd in RULE.SEEDS for sp in M1.SPLITS)
    by = {"dep": {"screen": DEP.decide_screen(new_all, base_all),
                  "full_run": DEP.decide_full(new_r5, base3, new_s, base_s) if full_ready else None},
          "ltg": {"screen": LTG.decide_screen(new_all, base_all),
                  "full_run": LTG.decide_full(new_all, base_all) if full_ready else None}}
    screen = combine_screen([by[r]["screen"] for r in rules])
    full = combine_full([by[r]["full_run"] for r in rules]) if full_ready else None
    bad += fit_problems(call, mem, base, cols)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    beside = {r: {"screen": by[r]["screen"]["verdict"], "full_run": by[r]["full_run"] and by[r]["full_run"]["verdict"]}
              for r in ("dep", "ltg") if r not in rules}
    own = {st: six_deltas(st, base, cols) for st in forms}
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": f"{ARM} with {', '.join(forms)}", "members": mem,
           "forms": forms, "rules": rules, "gnx_columns": cols if "gnx" in mem else None,
           "layers": layers_of(mem), "iso_verdict": call, "base": "zg6r-m (ISO)" if call == ISO.REPLACE else "zg6r (ML1)",
           "screen": screen, "full_run": full, "by_rule": {r: by[r] for r in rules}, "rules_beside": beside,
           "members_own_hp_deltas": own, "problems": bad,
           "parameters": {f: {k: fit_block(f, "screen", base).get(k) for k in ("stack_parameters", "parameters_h128")}
                          for f in FITS if is_read(f, base)},
           "aux": {f: fit_block(f, "screen", base).get("aux") for f in FITS if is_read(f, base)} if "ltg" in mem else {},
           "zstk_sha256": M1.WQ.sha_src(__file__), "utc": utc_now()}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    md = [f"# STK: {ARM} with {', '.join(forms)} against {rec['base']}, on PKO's pools", "",
          f"## Screen: {screen['verdict']} (rules {', '.join(f'{r} {v}' for r, v in zip(rules, screen['rules']))})", "",
          f"counts: {json.dumps(screen.get('counts', {}))}", f"rules beside: {json.dumps(beside)}", ""]
    rows = screen["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in screen.get("beside", [])]
    if rows:
        md += ["| split | read | base s0, s1, s2 | STK s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
        md += [f"| {r['split']} | {r['read']} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | "
               f"{r['delta']:+.4f} | {r['floor']:.4f} | {r['call']} |" for r in rows]
    md += ["", "members' own seed-0 deltas on the six WQ-hotpotqa reads:"]
    md += [f"- {st}: {json.dumps(d)}" for st, d in own.items()]
    if full is not None:
        md += ["", f"## Full run: {full['verdict']} (rules {', '.join(f'{r} {v}' for r, v in zip(rules, full['rules']))})"]
        for r in rules:
            fb = by[r]["full_run"]
            frows = [x for x in (fb.get("reads") or []) + (fb.get("slices") or []) + (fb.get("beside") or [])
                     if isinstance(x.get("delta"), dict) and all(k in x["delta"] for k in RULE.SEEDS)]
            if frows:
                md += ["", f"{r}'s rule:", "", "| split | read | d0 | d1 | d2 | floor | call |",
                       "|---|---|---:|---:|---:|---:|---|"]
                md += [f"| {x['split']} | {x['read']} | {x['delta'][0]:+.4f} | {x['delta'][1]:+.4f} | "
                       f"{x['delta'][2]:+.4f} | {x['floor']:.4f} | {x['call']} |" for x in frows]
    if bad:
        md += ["", "problems:"] + [f"- {p}" for p in bad]
    (out / "calls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"screen {screen['verdict']} {json.dumps(screen.get('counts', {}))}" + (f"; full run {full['verdict']}" if full else "")
        + f"; rules {rules}; beside {beside}; {OUT}/calls.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


ZERO_START = {"ltg": "ltg_U", "rtr": "rtr_e", "ira": "ira_E", "gnx": "gnx_w"}     # each member's zero-started parameter


def toy_stack(G, sets, big, cols="LTM"):
    """The stack on zg6r's own toy carve (zltg's toy: its rows, gold and links; zg6r's self-test steps 2 and 3) with
    LTM-format toy columns and a toy store basis; A3: with cols naming the pair columns, GNP-format toy columns too (from a
    draw of their own) and the gnx slot on cols throughout. For each member set: built under the set's patches with zg6r's and the
    rounds' parameters drawn at random (LTG's and GNX's at their starts), its eval and training forwards are those of the
    set without LTG and GNX bit for bit, and its state names are that set's plus LTG's and GNX's. For the set big, zg6r's
    and every member's parameters drawn at random: the listwise loss with the next-hop loss reaches every member's parameters; the strict load
    under the patches and its refusal without; a one-epoch lean_gpu.fit_variant that moves each member's zero-started
    parameter. LTG's term in the rounds refuses without the rounds; the stamp (stamp_check); the patches' restoring."""
    import tempfile
    import torch
    G6 = G.G6
    G.LG.bind_device_ops()
    names0 = tuple(G6.G6_NAMES) + tuple(G.G6R_NAMES)
    out = {"sets": {}}
    sp = "WQ-musique"
    basis0 = dict(LTG.BASIS)
    saved = (G.g6r_stack, G.g6r_layers, G.ZG6R.__init__, G.ZG6R.forward, G.LG.listwiseD, LTG.LAYERS, LTG.TERM_LAYER,
             LTG.TERM_NAMES, LTG.N_TERM)
    es_cols = contextlib.ExitStack()
    try:
        with tempfile.TemporaryDirectory(prefix="zstk_") as td:
            tmp = Path(td)
            rng = np.random.default_rng(31)                 # zltg's toy's draw: its rows, gold and links
            G.RM.toy_roots(tmp, rng, chunks=(12, 12))
            G.RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                       rel_dir=tmp / "rel")
            G.ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
            kw = dict(look_root=tmp / "look", cache_root=tmp / "cache", links_root=tmp / "lk", ch_root=tmp / "ch")

            class ToyIds(G6.ZM.ToyZ):
                def __init__(self, *a, **k):
                    super().__init__(*a, **k)
                    self.ids = [f"q{i}" for i in range(self.rows)]

            GNX.LTM.toy_cols(tmp / "ltm", sp, "metaqa", "toy", tmp / "cache", rng, "a" * 64)    # zgnx's order
            roots = {"LTM": tmp / "ltm", PAIR: tmp / "gnp"}
            if has_pair(cols):                      # A3: GNP's six, written under GNP's names, before the slot's patches
                P = gnp()
                with P.on_cols(tmp / "gnp"):
                    P.toy_cols6(tmp / "gnp", sp, "metaqa", "toy", tmp / "cache", np.random.default_rng(37), "b" * 64)
            es_cols.enter_context(on_columns(cols))
            gnx_out = roots[cols] if len(col_parts(cols)) == 1 else {p: roots[p] for p in col_parts(cols)}
            blocks = ["rank", "SEMB"]
            with G.U2N.extra_carves(("metaqa", "toy")):
                G.U2N.build_typed("metaqa", "toy", tmp / "look", tmp / "typed", **kw)
                plain = G.ZL.link_carve(G.RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
                c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
                G.Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
                cls = G.g6r_carve(G.Z3.kt_carve(plain, tmp / "kt"), tmp / "typed")
                cs = (cls.__init__, cls.batch)
                GNX.LOADED.clear()
                with GNX.on_carve(G, cls):
                    with GNX.on_split(sp, out=gnx_out):
                        ce = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
                    qs = np.r_[np.arange(ce.rows)[::-1][:7], np.arange(ce.rows)[:5]].astype(np.int64)
                    feats, nq, bz, _g = ce.batch(qs, blocks)
                GNX.LOADED.clear()
                B = qs.size
                keep = torch.ones(B, len(blocks))
                plain_feats = {k: v for k, v in feats.items() if k != GNX.NEXT}
                LTG.toy_basis(G, tmp / "cache", rng)
                LTG.load_basis(G, tmp / "cache", sha=None)
                lk = feats[G.LK_KEY]
                gold = torch.zeros(int(nq.numel()), dtype=torch.bool)
                pick = torch.randperm(lk["u"].numel(), generator=torch.Generator().manual_seed(4))[:6]
                gold[lk["u"][pick]] = True
                gold[lk["v"][pick]] = True
                gen = torch.Generator().manual_seed(3)

                def build(mem):
                    with on_members(G, mem, cls):
                        torch.manual_seed(0)
                        return G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)

                def run(m, mem, train, gs):
                    fd = feats if "gnx" in mem else plain_feats
                    with on_members(G, mem, cls):
                        if train:
                            m.train()
                            m.g6_gen.set_state(gs)
                            torch.manual_seed(9)
                            return m(fd, keep, nq, B, bz).detach()
                        m.eval()
                        with torch.no_grad():
                            return m(fd, keep, nq, B, bz)

                for mem in sets:
                    ref = tuple(st for st in mem if st not in ("ltg", "gnx"))
                    m, r = build(mem), build(ref)
                    rk, mk = set(r.state_dict()), set(m.state_dict())
                    with on_members(G, mem, cls):
                        added = {k for st in mem if st in ("ltg", "gnx") for k in member_names(G, st)}
                    with torch.no_grad():
                        for k in names0 + tuple(k for st in ref if st in ROUNDS_OF for k in member_names(G, st)):
                            getattr(m, k).copy_(torch.randn(getattr(m, k).shape, generator=gen) * 0.3)
                        if "ira" in mem:
                            m.ira_lam.fill_(0.5)
                    r.load_state_dict({k: v for k, v in m.state_dict().items() if k in rk}, strict=True)
                    gs = m.g6_gen.get_state()
                    out["sets"]["+".join(mem)] = {
                        "names_are_ref_plus_terms": mk == (rk | added) and not (rk & added),
                        "eval_start_is_ref": bool(torch.equal(run(m, mem, False, gs), run(r, ref, False, gs))),
                        "train_start_is_ref": bool(torch.equal(run(m, mem, True, gs), run(r, ref, True, gs)))}
                # the set big: zg6r's and the members' parameters at random (the encoder's at their start, as the members'
                # own toys: at random they saturate every attention logit at zg6r's clamp), the losses' gradients, the
                # strict loads, a one-epoch fit
                m = build(big)
                out["gnx_shape"] = tuple(m.gnx_w.shape) if "gnx" in big else None
                out["next_width"] = int(feats[GNX.NEXT].shape[1])
                with on_members(G, big, cls):
                    names = {st: member_names(G, st) for st in big}
                with torch.no_grad():
                    for k in names0 + tuple(k for st in big for k in names[st]):
                        getattr(m, k).copy_(torch.randn(getattr(m, k).shape, generator=gen) * 0.3)
                    if "ira" in big:
                        m.ira_lam.fill_(0.5)                # inside clip(0, 1): its gradient is not cut
                LTG.aux_reset()
                with on_members(G, big, cls):
                    m.train()
                    m.zero_grad()
                    s = m(feats if "gnx" in big else plain_feats, keep, nq, B, bz)
                    G.LG.listwiseD(s, gold, nq, B).backward()
                    out["aux_steps"] = LTG.AUX["steps"]
                    out["no_grad"] = [k for st in big for k in names[st]
                                      if getattr(m, k).grad is None or float(getattr(m, k).grad.abs().sum()) == 0]
                    out["train_finite"] = bool(torch.isfinite(s).all())
                    G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0).load_state_dict(m.state_dict(), strict=True)
                    out["strict_load_patched"] = "passed"
                try:
                    G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0).load_state_dict(m.state_dict(), strict=True)
                    out["strict_load_plain"] = "passed"
                except RuntimeError:
                    out["strict_load_plain"] = "refused"
                cfg = {"lr": 2e-3, "wd": 1e-4, "dropout": 0.1, "epochs": 1, "swa_from": 0, "cos": 0, "adamw": 0, "drop": 0}
                LTG.aux_reset()
                with on_members(G, big, cls), M1.WQ.patch(G.LG, LeanMLP8D=G.ZG6R):
                    fr = G.LG.fit_variant([ce], blocks, cfg, 0, 16, "none", "cpu")
                out["fit_aux_steps"] = LTG.AUX["steps"]
                out["fit_moved"] = {ZERO_START[st]: float(fr["swa"][ZERO_START[st]].abs().sum()) > 0
                                    for st in big if st in ZERO_START}
                out["fit_loss_finite"] = all(math.isfinite(c["loss"]) for c in fr["curve"])
            try:
                with on_term_rounds(G):
                    pass
                out["term_rounds_without_rounds"] = "passed"
            except SystemExit:
                out["term_rounds_without_rounds"] = "refused"
            out["restored"] = (G.g6r_stack, G.g6r_layers, G.ZG6R.__init__, G.ZG6R.forward, G.LG.listwiseD, LTG.LAYERS,
                               LTG.TERM_LAYER, LTG.TERM_NAMES, LTG.N_TERM) == saved and (cls.__init__, cls.batch) == cs
            out["cur_clear"] = not LTG.CUR and GNX.CUR == {"split": None, "fit_sha256": None, "out": None}
            out["stamp"] = stamp_check(G, cls, big)
            es_cols.close()
    finally:
        es_cols.close()
        LTG.aux_reset()
        GNX.LOADED.clear()
        LTG.BASIS.clear()
        LTG.BASIS.update(basis0)
    return out


def stamp_check(G, cls, mem):
    """on_stack with a recording stamp: the stack's record (members, layers, parts, parameters), the base, ISO's verdict,
    with GNX the NEXT block and its refusals, with LTG the next-hop curve; zml1s's names and every patch restored after,
    also on a failure."""
    out = {}
    saved = (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.stamp, M1.code_shas, M1.load_encoder, M1.train, M1.read)
    n0, seen, orig_stamp = len(M1.ALL_FITS), [], M1.stamp
    hp = fit_name(0, "WQ-hotpotqa")
    try:
        M1.stamp = lambda name, env, what, extra=None: seen.append((name, what, dict(extra or {})))
        with on_stack(G, ISO.KEEP, mem, "screen NO_GAIN" if any(st in mem for st in ROUNDS_OF) else None, cls) as rec:
            out["inside"] = (len(M1.ALL_FITS) == n0 + 6 and (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY)
                             and M1.load_encoder is MS.ml1_load and RS.installed())
            cs = M1.code_shas(ARM)
            out["code_keys"] = all(f"{k}_sha256" in cs for k in ("zstk", "zdep", "zltg", "zrtr", "zira", "zgnx", "zltm",
                                                                 "ltmcols", "s0stop", "ml1start", "readswa", "zgna", "zlta",
                                                                 "ltacols") + GNP_CODE)
            M1._ACTIVE[0] = hp
            out["names"] = (M1.pko_fit_name(ARM, 0, "WQ-hotpotqa"), M1.pko_fit_name(ARM, 1, "WQ-hotpotqa"))
            M1._ACTIVE[0] = None
            out["record"] = rec
            if "gnx" in mem:
                try:
                    with GNX.on_split("WQ-hotpotqa"):
                        M1.stamp(hp, {}, "screen", {"x": 1})
                    out["no_columns"] = "passed"
                except SystemExit:
                    out["no_columns"] = "refused"
            GNX.LOADED.clear()
            with GNX.on_split("WQ-hotpotqa", "c" * 64):
                GNX.LOADED.extend({"dataset": k.split("/")[0], "carve": k.split("/")[1], "record_sha256": "r"}
                                  for k in GNX.train_carves("WQ-hotpotqa") + GNX.eval_carves())
                M1.stamp(hp, {}, "screen", {"x": 1})
                M1.stamp(hp, {}, "read", {"x": 2})
            GNX.LOADED.clear()
        try:
            with on_stack(G, ISO.KEEP, mem, None, cls):
                raise RuntimeError("inside")
        except RuntimeError:
            pass
    finally:
        M1.stamp = orig_stamp
        M1._ACTIVE[0] = None
        GNX.LOADED.clear()
    out["seen"] = [(n, w, sorted(x)) for n, w, x in seen]
    out["screen_extra"] = seen[0][2] if seen else {}
    out["restored"] = (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.stamp, M1.code_shas, M1.load_encoder, M1.train,
                       M1.read) == saved and len(M1.ALL_FITS) == n0 and not RS.installed()
    return out


def check_toy(t, sets, big, layers, cols="LTM"):
    for mem in sets:
        r = t["sets"]["+".join(mem)]
        assert r["names_are_ref_plus_terms"] and r["eval_start_is_ref"] and r["train_start_is_ref"], (mem, r)
    assert not t["no_grad"] and t["train_finite"] and t["strict_load_patched"] == "passed", t
    assert t["strict_load_plain"] == "refused" and t["fit_loss_finite"] and all(t["fit_moved"].values()), t
    assert t["fit_moved"] and (t["aux_steps"] == 1 and t["fit_aux_steps"] > 0 if "ltg" in big else True), t
    assert t["term_rounds_without_rounds"] == "refused" and t["restored"] and t["cur_clear"], t
    s = t["stamp"]
    assert s["inside"] and s["code_keys"] and s["restored"], s
    assert s["names"] == (fit_name(0, "WQ-hotpotqa"), M1.fit_name(ARM, 1, "WQ-hotpotqa")), s
    rec = s["record"]
    assert rec["members"] == list(big) and rec["layers"] == layers and sorted(rec["parts"]) == sorted(big), rec
    for st in big:
        if st != "dep":
            assert rec["parts"][st]["layers"] == layers, rec
    if "ltg" in big:
        assert rec["parts"]["ltg"]["term_parameters"] == ltg_terms(layers, rec["parts"]["ltg"]["axes"]), rec
    assert [x[:2] for x in s["seen"]] == [(fit_name(0, "WQ-hotpotqa"), "screen"), (fit_name(0, "WQ-hotpotqa"), "read")], s
    ex = s["screen_extra"]
    assert ex["base"] == DEP.base_fit(0, "WQ-hotpotqa", ISO.KEEP) and ex["iso_verdict"] == ISO.KEEP and ex["members"] == list(big), ex
    assert ("next" in ex) == ("gnx" in big) and ("aux" in ex) == ("ltg" in big) and ex["x"] == 1, sorted(ex)
    if "gnx" in big:
        assert s["no_columns"] == "refused" and ex["next"]["fit_sha256"] == "c" * 64, s
        n, g = slot_in(cols), rec["parts"]["gnx"]                            # A3: the slot's columns and width
        assert (g["inputs"], g["columns"], g["term_parameters"]) == (n, slot_columns(cols), GNX.HIDDEN * n), (cols, g)
        assert (ex["next"]["declared_in"], ex["next"]["columns"]) == (next_declared(cols), slot_columns(cols)), ex["next"]
        assert t["gnx_shape"] == (16, n) and t["next_width"] == len(slot_columns(cols)), (cols, t["gnx_shape"])
        assert rec["stack_parameters"] >= GNX.HIDDEN * n, rec


def layers_check():
    """Run in a process of its own by the self-test: the four-layer import, LTG's counts at four layers, and the toy
    stacks with DEP."""
    G, _U2N = DEP.import_deep()
    with on_ltg_layers(G):
        rec = {"layers": int(G.LAYERS), "term_layer": len(LTG.TERM_LAYER), "term_names": len(LTG.TERM_NAMES),
               "n_term": LTG.N_TERM, "term_layer_last": list(LTG.TERM_LAYER[-1])}
    rec["after"] = (LTG.LAYERS, len(LTG.TERM_LAYER), LTG.N_TERM)
    sets = [("dep", "ltg"), ("dep", "gnx"), ("dep", "ltg", "gnx")]
    t = toy_stack(G, sets, ("dep", "ltg", "gnx"))
    check_toy(t, sets, ("dep", "ltg", "gnx"), DEP.LAYERS)
    rec["toy"] = {k: t[k] for k in ("sets", "no_grad", "fit_moved", "aux_steps", "fit_aux_steps")}
    print("LAYERS_CHECK " + json.dumps(rec), flush=True)
    return 0


def gates_check():
    """members, stack-gate, own-gate, the pick and the stack's s0stop names on made files."""
    import tempfile
    t0 = utc_s(MEMBERS_BY) - 3600
    with tempfile.TemporaryDirectory(prefix="zstk_g_") as td:
        base = Path(td)

        def calls_file(st, v, deltas=None, decl=None):
            o = base / MOD[st].OUT
            o.mkdir(parents=True, exist_ok=True)
            rows = [{"split": a, "read": b, "delta": (deltas or {}).get(b, 0.0), "call": "WITHIN"} for a, b in PICK_READS]
            M1.WQ.write_json(o / "calls.json", {"declared_in": decl or MOD[st].DECLARED, "screen": {"verdict": v,
                                                                                                    "reads": rows}})

        def stop_file(st):
            _m, out, s0, hp, _r, _d = S0.STAGES[st]
            rp = S0.read_path(s0, base)
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text('{"read": 1}', encoding="utf-8")
            (base / out).mkdir(parents=True, exist_ok=True)
            (base / out / "s0stop.json").write_text(json.dumps({"s0_fit": s0, "hp_fit": hp, "s0_read_sha256": S0.sha_file(rp),
                                                                "decision": S0.STOP}), encoding="utf-8")

        def side_file(out, decl0, v, deltas=None, decl=None):
            o = base / out
            o.mkdir(parents=True, exist_ok=True)
            rows = [{"split": a, "read": b, "delta": (deltas or {}).get(b, 0.0), "call": "WITHIN"} for a, b in PICK_READS]
            M1.WQ.write_json(o / "calls.json", {"declared_in": decl or decl0, "screen": {"verdict": v, "reads": rows}})

        def gna_file(v, deltas=None, decl=None):
            side_file(GNA.OUT, GNA.DECLARED, v, deltas, decl)

        def gnp_file(v, deltas=None, decl=None):
            side_file(GNP_OUT, GNP_DECLARED, v, deltas, decl)

        def late_file(st, v, deltas=None, decl=None):
            side_file(LATE_NAMES[st][1], LATE_NAMES[st][3], v, deltas, decl)

        def late_none():
            """A4, A5: the later forms not passing, so A1's to A3's cases read as they were written."""
            for st in LATE + A5:
                late_file(st, NO_GAIN)

        def fixed(now=t0):
            return fix_members(base, now)

        def reset(late=True):
            for p in sorted(base.rglob("*"), reverse=True):
                p.unlink() if p.is_file() else p.rmdir()
            if late:
                late_none()

        assert fixed()[0] is False and fixed()[1]["pending"] == list(ORDER) + ["gna", "gnp"] + list(LATE + A5)
        late_none()
        calls_file("dep", ADOPT_FOR_FULL_RUN, {"squad": 0.01})
        calls_file("ltg", ADOPT_FOR_FULL_RUN, {"squad": 0.02})
        calls_file("gnx", INCOMPLETE)
        done, rec = fixed()
        assert not done and rec["pending"] == ["gnx", "gna", "gnp"] and rec["will_not_run"] == ["rtr", "ira"], rec
        gna_file(NO_GAIN, decl="docs/OTHER.md")                     # not GNA's file: no verdict (A2)
        assert fixed()[1]["pending"] == ["gnx", "gna", "gnp"] and fixed()[1]["gna_verdict"] is None
        gna_file(NO_GAIN)
        assert fixed()[1]["pending"] == ["gnx", "gnp"] and fixed()[1]["gna_verdict"] == NO_GAIN
        gnp_file(NO_GAIN, decl="docs/OTHER.md")                     # not GNP's file: no verdict (A3)
        assert fixed()[1]["pending"] == ["gnx", "gnp"] and fixed()[1]["gnp_verdict"] is None
        gnp_file(NO_GAIN)
        assert fixed()[1]["pending"] == ["gnx"] and fixed()[1]["gnp_verdict"] == NO_GAIN
        stop_file("gnx")                                            # a STOP is NOT_ADOPTED
        done, rec = fixed()
        assert done and rec["members"] == ["dep", "ltg"] and rec["verdicts"]["gnx"] == NOT_ADOPTED, rec
        assert rec["pick"] == "ltg" and rec["layers"] == DEP.LAYERS and rec["rules"] == ["dep", "ltg"], rec
        assert rec["gnx_columns"] == "LTM" and rec["forms"] == ["dep", "ltg"], rec
        assert members(False, base, t0) == 0 and members_record(base)["members"] == ["dep", "ltg"]
        calls_file("ltg", NOT_ADOPTED)
        assert members(False, base, t0) == 0 and members_record(base)["members"] == ["dep", "ltg"]   # fixed: not rewritten
        assert stack_gate(base) == 0
        with on_s0stop():
            assert S0.STAGES["stk"][2:4] == (fit_name(0, "WQ-musique"), fit_name(0, "WQ-hotpotqa"))
        assert "stk" not in S0.STAGES and "stk" not in S0.RULES
        keys = stk_keys("stk", base)
        assert keys == S0.deciding("dep") and len(keys) == 8, keys
        # own: two members, the stack waiting, then each outcome
        assert own_call("dep", base, t0)[:2] == (False, 1)
        assert own_call("dep", base, utc_s(FREEZE))[:2] == (True, 1)
        o = base / OUT
        M1.WQ.write_json(o / "calls.json", {"declared_in": DECLARED, "screen": {"verdict": INCOMPLETE}, "full_run": None})
        assert own_call("ltg", base, t0)[:2] == (False, 1)
        M1.WQ.write_json(o / "calls.json", {"declared_in": DECLARED, "screen": {"verdict": NO_GAIN}, "full_run": None})
        assert own_call("ltg", base, t0)[:2] == (True, 0) and own_call("dep", base, t0)[:2] == (True, 1)
        assert own_call("ltg", base, utc_s(FREEZE) + 9)[:2] == (True, 0)
        M1.WQ.write_json(o / "calls.json", {"declared_in": DECLARED, "screen": {"verdict": ADOPT_FOR_FULL_RUN},
                                            "full_run": None})
        assert own_call("ltg", base, t0)[:2] == (False, 1)
        M1.WQ.write_json(o / "calls.json", {"declared_in": DECLARED, "screen": {"verdict": ADOPT_FOR_FULL_RUN},
                                            "full_run": {"verdict": ADOPT}})
        assert own_call("ltg", base, t0)[:2] == (True, 1) and own_call("dep", base, t0)[:2] == (True, 1)
        M1.WQ.write_json(o / "calls.json", {"declared_in": DECLARED, "screen": {"verdict": ADOPT_FOR_FULL_RUN},
                                            "full_run": {"verdict": NOT_ADOPTED}})
        assert own_call("ltg", base, t0)[:2] == (True, 0)
        (o / "calls.json").unlink()
        stk_rp = S0.read_path(fit_name(0, "WQ-musique"), base)
        stk_rp.parent.mkdir(parents=True, exist_ok=True)
        stk_rp.write_text('{"read": 1}', encoding="utf-8")
        M1.WQ.write_json(o / "s0stop.json", {"s0_fit": fit_name(0, "WQ-musique"), "hp_fit": fit_name(0, "WQ-hotpotqa"),
                                             "s0_read_sha256": S0.sha_file(stk_rp), "decision": S0.STOP})
        assert stack_verdicts(base) == (NOT_ADOPTED, None) and own_call("ltg", base, t0)[:2] == (True, 0)
        # DEP fails: RTR and IRA run; IRA stands for RTR; a tie goes to ORDER; late leaves the pending out
        reset()
        calls_file("dep", NO_GAIN)
        calls_file("ltg", ADOPT_FOR_FULL_RUN, {"squad": 0.012, "webqsp": 0.0})
        calls_file("rtr", ADOPT_FOR_FULL_RUN, {"squad": 0.006, "webqsp": 0.006})
        calls_file("ira", ADOPT_FOR_FULL_RUN, {"squad": 0.003})
        calls_file("gnx", NOT_ADOPTED, decl="docs/OTHER.md")          # not GNX's file: no verdict
        done, rec = fixed()
        assert not done and rec["pending"] == ["gnx", "gna", "gnp"] and rec["will_not_run"] == [], rec
        done, rec = fixed(utc_s(MEMBERS_BY))
        assert done and rec["members"] == ["ltg", "ira"] and rec["passing"] == ["ltg", "rtr", "ira"], rec
        assert rec["left_out"] == ["gnx", "gna", "gnp"] and rec["layers"] == DEP.BASE_LAYERS, rec
        assert rec["rules"] == ["dep", "ltg"] and rec["forms"] == ["ltg", "ira"], rec
        assert rec["pick"] == "ltg" and rec["pick_means"]["ltg"] == rec["pick_means"]["rtr"] == 0.002, rec
        # one member: its own full run, no stack
        reset()
        calls_file("dep", ADOPT_FOR_FULL_RUN)
        for st in ("ltg", "gnx"):
            calls_file(st, NO_GAIN)
        assert members(False, base, t0) == 2                         # A2: GNA's verdict is waited on
        gna_file(NO_GAIN)
        assert members(False, base, t0) == 2                         # A3: and GNP's
        gnp_file(NO_GAIN)
        assert members(False, base, t0) == 0 and members_record(base)["members"] == ["dep"]
        assert stack_gate(base) == 1 and own_call("dep", base, t0)[:2] == (True, 0) and own_call("ltg", base, t0)[:2] == (True, 1)
        assert own_call("gnx", base, t0)[:2] == own_call("gna", base, t0)[:2] == own_call("gnp", base, t0)[:2] == (True, 1)
        # none
        reset()
        for st in ORDER:
            calls_file(st, NOT_ADOPTED)
        gna_file(NOT_ADOPTED)
        gnp_file(NOT_ADOPTED)
        assert members(False, base, t0) == 0 and members_record(base)["members"] == []
        assert stack_gate(base) == 1 and all(own_call(st, base, t0)[:2] == (True, 1) for st in ORDER + ("gna", "gnp"))
        reset()
        assert members(False, base, t0) == 2 and stack_gate(base) == 2 and own_call("dep", base, t0)[:2] == (False, 1)
        slept = []
        calls_file("dep", NOT_ADOPTED)
        assert members(True, base, utc_s(MEMBERS_BY), sleep=slept.append) == 0 and not slept   # late: no waiting
        assert members_record(base)["members"] == [] and members_record(base)["left_out"] == ["ltg", "rtr", "ira", "gnx",
                                                                                               "gna", "gnp"]
        # A2: GNA passes and GNX does not: the gnx slot passes on LTA's columns; its mean and own run are GNA's
        reset()
        calls_file("dep", NO_GAIN)
        calls_file("ltg", ADOPT_FOR_FULL_RUN, {"squad": 0.006})
        for st in ("rtr", "ira"):
            calls_file(st, NOT_ADOPTED)
        calls_file("gnx", NO_GAIN, {"squad": 0.06})
        gna_file(ADOPT_FOR_FULL_RUN, {"squad": 0.018})
        gnp_file(NO_GAIN, {"squad": 0.6})
        done, rec = fixed()
        assert done and rec["members"] == ["ltg", "gnx"] and rec["gnx_columns"] == "LTA" and rec["pick"] == "gnx", rec
        assert rec["pick_means"] == {"ltg": 0.001, "gnx": 0.003} and rec["gnx_means"] == {"LTM": 0.01, "LTA": 0.003}, rec
        assert members(False, base, t0) == 0 and gnx_columns(members_record(base)) == "LTA"
        o = base / OUT
        M1.WQ.write_json(o / "calls.json", {"declared_in": DECLARED, "screen": {"verdict": NO_GAIN}, "full_run": None})
        assert own_call("gna", base, t0)[:2] == (True, 0) and own_call("gnx", base, t0)[:2] == (True, 1)
        assert own_call("ltg", base, t0)[:2] == own_call("gnp", base, t0)[:2] == (True, 1)
        # A2: both pass: LTA's columns when GNA's mean is the larger; a tie or a smaller one keeps LTM's
        for d_gna, cols in ((0.024, "LTA"), (0.012, "LTM"), (0.006, "LTM")):
            reset()
            calls_file("dep", NO_GAIN)
            for st in ("ltg", "rtr", "ira"):
                calls_file(st, NOT_ADOPTED)
            calls_file("gnx", ADOPT_FOR_FULL_RUN, {"squad": 0.012})
            gna_file(ADOPT_FOR_FULL_RUN, {"squad": d_gna})
            gnp_file(NOT_ADOPTED)
            done, rec = fixed()
            assert done and rec["members"] == ["gnx"] and rec["gnx_columns"] == cols, (d_gna, rec)
            assert members(False, base, t0) == 0 and stack_gate(base) == 1
            own = {st: own_call(st, base, t0)[:2] for st in ("gnx", "gna")}
            assert own == {"gnx": (True, 0 if cols == "LTM" else 1), "gna": (True, 0 if cols == "LTA" else 1)}, own
        # A2: GNA's seed-0 STOP is NOT_ADOPTED; GNA passing alone with GNX's file another's still waits on GNX
        reset()
        gna_file(ADOPT_FOR_FULL_RUN)
        with GNA.on_s0stop():
            stop_file("gna")
        assert "gna" not in S0.STAGES and gna_verdict(base) == NOT_ADOPTED
        # A2: an unknown column set is not a members record
        (base / OUT).mkdir(parents=True, exist_ok=True)
        M1.WQ.write_json(base / OUT / "members.json", {"declared_in": DECLARED, "members": ["gnx"], "gnx_columns": "X"})
        assert members_record(base) is None
        # A3: GNP passes beside DEP: the gnx slot on the six pair columns alone holds GNP's form; the pick is over the forms
        reset()
        calls_file("dep", ADOPT_FOR_FULL_RUN, {"squad": 0.006})
        for st in ("ltg", "gnx"):
            calls_file(st, NO_GAIN, {"squad": 0.06})
        gna_file(NOT_ADOPTED)
        gnp_file(ADOPT_FOR_FULL_RUN, {"squad": 0.012})
        done, rec = fixed()
        assert done and rec["members"] == ["dep", "gnx"] and rec["forms"] == ["dep", "gnp"], rec
        assert rec["passing"] == ["dep", "gnp"] and rec["gnx_columns"] == PAIR and rec["pick"] == "gnp", rec
        assert rec["pick_means"] == {"dep": 0.001, "gnp": 0.002} and rec["gnp_verdict"] == ADOPT_FOR_FULL_RUN, rec
        assert rec["layers"] == DEP.LAYERS and rec["rules"] == ["dep", "ltg"] and rec["will_not_run"] == ["rtr", "ira"], rec
        assert members(False, base, t0) == 0 and stack_gate(base) == 0
        assert own_call("dep", base, t0)[:2] == own_call("gnp", base, t0)[:2] == (False, 1)
        M1.WQ.write_json(base / OUT / "calls.json", {"declared_in": DECLARED, "screen": {"verdict": ADOPT_FOR_FULL_RUN},
                                                     "full_run": {"verdict": NOT_ADOPTED}})
        own = {st: own_call(st, base, t0)[:2] for st in ORDER + ("gna", "gnp")}
        assert own == {**{st: (True, 1) for st in ORDER + ("gna",)}, "gnp": (True, 0)}, own
        # A3: GNX and GNP pass: LTM's columns then the pair columns, one member, two forms; a tie in the pick goes to GNX
        for d_gnp, pick in ((0.012, "gnx"), (0.018, "gnp")):
            reset()
            calls_file("dep", NO_GAIN)
            for st in ("ltg", "rtr", "ira"):
                calls_file(st, NOT_ADOPTED)
            calls_file("gnx", ADOPT_FOR_FULL_RUN, {"squad": 0.012})
            gna_file(NO_GAIN)
            gnp_file(ADOPT_FOR_FULL_RUN, {"squad": d_gnp})
            done, rec = fixed()
            assert done and rec["members"] == ["gnx"] and rec["forms"] == ["gnx", "gnp"], (d_gnp, rec)
            assert rec["gnx_columns"] == "LTM+PAIR" and rec["pick"] == pick and rec["layers"] == DEP.BASE_LAYERS, rec
            assert rec["rules"] == ["ltg"], rec
            assert members(False, base, t0) == 0 and stack_gate(base) == 0
            M1.WQ.write_json(base / OUT / "calls.json", {"declared_in": DECLARED, "screen": {"verdict": NO_GAIN},
                                                         "full_run": None})
            own = {st: own_call(st, base, t0)[:2] for st in ("gnx", "gna", "gnp")}
            assert own == {"gnx": (True, 0 if pick == "gnx" else 1), "gna": (True, 1),
                           "gnp": (True, 0 if pick == "gnp" else 1)}, (d_gnp, own)
        # A3: GNA and GNP pass, GNX not: LTA's columns then the pair columns; the slot's own run is GNA's
        reset()
        calls_file("dep", NO_GAIN)
        for st in ("ltg", "rtr", "ira"):
            calls_file(st, NOT_ADOPTED)
        calls_file("gnx", NO_GAIN)
        gna_file(ADOPT_FOR_FULL_RUN, {"squad": 0.018})
        gnp_file(ADOPT_FOR_FULL_RUN, {"squad": 0.006})
        done, rec = fixed()
        assert done and rec["members"] == ["gnx"] and rec["forms"] == ["gnx", "gnp"], rec
        assert rec["gnx_columns"] == "LTA+PAIR" and rec["pick"] == "gnx" and rec["pick_means"] == {"gnx": 0.003,
                                                                                                   "gnp": 0.001}, rec
        assert members(False, base, t0) == 0 and stack_gate(base) == 0
        M1.WQ.write_json(base / OUT / "calls.json", {"declared_in": DECLARED, "screen": {"verdict": ADOPT_FOR_FULL_RUN},
                                                     "full_run": {"verdict": NOT_ADOPTED}})
        own = {st: own_call(st, base, t0)[:2] for st in ("gnx", "gna", "gnp")}
        assert own == {"gnx": (True, 1), "gna": (True, 0), "gnp": (True, 1)}, own
        # A3: GNP alone passes: one form (the gnx slot on the pair columns alone), no stack, GNP's own full run
        reset()
        calls_file("dep", NO_GAIN)
        for st in ("ltg", "rtr", "ira", "gnx"):
            calls_file(st, NOT_ADOPTED)
        gna_file(NOT_ADOPTED)
        gnp_file(ADOPT_FOR_FULL_RUN)
        assert members(False, base, t0) == 0
        mr = members_record(base)
        assert (mr["forms"], mr["members"], gnx_columns(mr)) == (["gnp"], ["gnx"], PAIR) and stack_gate(base) == 1, mr
        own = {st: own_call(st, base, t0)[:2] for st in ORDER + ("gna", "gnp")}
        assert own == {**{st: (True, 1) for st in ORDER + ("gna",)}, "gnp": (True, 0)}, own
        # A3: GNP's seed-0 STOP is NOT_ADOPTED, under zgnp's own entry; s0stop's table put back
        reset()
        gnp_file(ADOPT_FOR_FULL_RUN)
        st0 = dict(S0.STAGES)
        with gnp().on_s0stop():
            assert S0.STAGES[GNP_KEY] == gnp_entry()
            stop_file(GNP_KEY)
        assert S0.STAGES == st0 and gnp_verdict(base) == NOT_ADOPTED and gnp().stop_record(base)["decision"] == S0.STOP
        # A3: a record whose forms, members and columns do not agree is not a members record; one before A3's (no forms)
        # is read with its members as its forms
        (base / OUT).mkdir(parents=True, exist_ok=True)
        for forms, mem, cols in ((["dep", "gnp"], ["dep"], PAIR), (["dep", "gnp"], ["dep", "gnx"], "LTM"),
                                 (["dep", "gnx"], ["dep", "gnx"], "LTM+PAIR"), (["gnx", "gnp"], ["gnx"], PAIR),
                                 (["gnp"], ["gnx"], "LTM+PAIR"), (["dep", "gnq"], ["dep"], "LTM"),
                                 (["gnx", "gnp"], ["gnx"], "PAIR+LTM")):
            M1.WQ.write_json(base / OUT / "members.json", {"declared_in": DECLARED, "forms": forms, "members": mem,
                                                           "gnx_columns": cols})
            assert members_record(base) is None, (forms, mem, cols)
        M1.WQ.write_json(base / OUT / "members.json", {"declared_in": DECLARED, "forms": ["gnx", "gnp"],
                                                       "members": ["gnx"], "gnx_columns": "LTA+PAIR"})
        assert forms_of(members_record(base)) == ["gnx", "gnp"]
        M1.WQ.write_json(base / OUT / "members.json", {"declared_in": DECLARED, "members": ["dep", "gnx"],
                                                       "gnx_columns": "LTA"})
        assert forms_of(members_record(base)) == ["dep", "gnx"]
        # A4: GNW's and LTX's verdicts are waited on; a calls file declared by another file is not a verdict
        reset(late=False)
        assert fixed()[1]["pending"] == list(ORDER) + ["gna", "gnp"] + list(LATE + A5)
        late_file("gnw", NO_GAIN, decl="docs/OTHER.md")
        late_file("ltx", NO_GAIN)
        rec = fixed()[1]
        assert rec["gnw_verdict"] is None and rec["ltx_verdict"] == NO_GAIN and rec["pending"][-4:] == ["gnw", *A5], rec
        # A4: DEP and LTX pass: two forms and two members, at four layers. This code does not compose LTX, so the stack
        # gate waits (2 without wait, one look a call), decides once at MEMBERS_BY that the stack does not run, and the
        # own gates read that as the stack's screen NOT_ADOPTED: the pick's own full run is released
        reset()
        calls_file("dep", ADOPT_FOR_FULL_RUN, {"squad": 0.006})
        for st in ("ltg", "gnx"):
            calls_file(st, NO_GAIN)
        gna_file(NO_GAIN)
        gnp_file(NOT_ADOPTED)
        late_file("ltx", ADOPT_FOR_FULL_RUN, {"squad": 0.012})
        done, rec = fixed()
        assert done and rec["members"] == ["dep", "ltx"] == rec["forms"] and rec["pick"] == "ltx", rec
        assert rec["layers"] == DEP.LAYERS and rec["rules"] == ["dep", "ltg"] and rec["gnx_columns"] == "LTM", rec
        assert rec["pick_means"] == {"dep": 0.001, "ltx": 0.002} and rec["ltx_verdict"] == ADOPT_FOR_FULL_RUN, rec
        assert members(False, base, t0) == 0 and not composed(members_record(base)["members"])
        assert composes_now(base) == 1
        looks, slept = [], []
        assert stack_gate(base, False, t0, slept.append, lambda: looks.append(1) or False) == 2 and looks == [1]
        assert gate_record(base) is None and own_call("ltx", base, t0)[:2] == (False, 1) and not slept
        assert stack_gate(base, True, utc_s(MEMBERS_BY), slept.append, lambda: False) == 1 and not slept
        g = gate_record(base)
        assert g["runs"] is False and g["members"] == ["dep", "ltx"], g
        assert stack_gate(base, True, t0, slept.append, lambda: True) == 1 and not slept        # decided once
        own = {st: own_call(st, base, t0)[:2] for st in FORMS + ("gna",)}
        assert own == {**{st: (True, 1) for st in FORMS + ("gna",)}, "ltx": (True, 0)}, own
        # A4: the code on disk composes LTX before MEMBERS_BY (after one look that it does not): the stack runs, decided
        # once; the own gates wait on the stack's verdicts
        (base / OUT / "stack_gate.json").unlink()
        seq = iter([False, True])
        assert stack_gate(base, True, t0, slept.append, lambda: next(seq)) == 0 and slept == [POLL]
        assert gate_record(base)["runs"] is True
        assert stack_gate(base, True, utc_s(MEMBERS_BY), slept.append, lambda: False) == 0 and slept == [POLL]
        assert own_call("ltx", base, t0)[:2] == own_call("dep", base, t0)[:2] == (False, 1)
        # A4: GNX, GNW and LTX pass, DEP not: three forms, three members, three layers; a tie in the pick goes to GNX
        reset()
        calls_file("dep", NO_GAIN)
        for st in ("ltg", "rtr", "ira"):
            calls_file(st, NOT_ADOPTED)
        calls_file("gnx", ADOPT_FOR_FULL_RUN, {"squad": 0.012})
        gna_file(NO_GAIN)
        gnp_file(NO_GAIN)
        late_file("gnw", ADOPT_FOR_FULL_RUN, {"squad": 0.012})
        late_file("ltx", ADOPT_FOR_FULL_RUN, {"squad": 0.006})
        done, rec = fixed()
        assert done and rec["members"] == ["gnx", "gnw", "ltx"] == rec["forms"] and rec["pick"] == "gnx", rec
        assert rec["layers"] == DEP.BASE_LAYERS and rec["rules"] == ["ltg"] and rec["gnx_columns"] == "LTM", rec
        assert members(False, base, t0) == 0 and members_record(base)["members"] == ["gnx", "gnw", "ltx"]
        assert composes_now(base) == 1 and stk_keys("stk", base) == S0.deciding("ltg")
        # A4: GNW alone passes: one form, no stack (composition never looked at), GNW's own full run
        reset()
        calls_file("dep", NO_GAIN)
        for st in ("ltg", "rtr", "ira", "gnx"):
            calls_file(st, NOT_ADOPTED)
        gna_file(NOT_ADOPTED)
        gnp_file(NOT_ADOPTED)
        late_file("gnw", ADOPT_FOR_FULL_RUN)
        assert members(False, base, t0) == 0 and members_record(base)["members"] == ["gnw"]
        assert stack_gate(base, True, t0, slept.append, lambda: 1 / 0) == 1 and composes_now(base) == 1
        own = {st: own_call(st, base, t0)[:2] for st in FORMS + ("gna",)}
        assert own == {**{st: (True, 1) for st in FORMS + ("gna",)}, "gnw": (True, 0)}, own
        # A4: a members record of ORDER's members composes; none fixed is 2
        reset()
        assert composes_now(base) == 2
        calls_file("dep", ADOPT_FOR_FULL_RUN)
        calls_file("ltg", ADOPT_FOR_FULL_RUN)
        for st in ("gnx",):
            calls_file(st, NO_GAIN)
        gna_file(NO_GAIN)
        gnp_file(NO_GAIN)
        assert members(False, base, t0) == 0 and composes_now(base) == 0
        assert stack_gate(base, True, t0, slept.append, lambda: 1 / 0) == 0 and gate_record(base) is None
        # A4: GNW's and LTX's seed-0 STOPs are NOT_ADOPTED, under their modules' own entries; s0stop's table put back
        for st, m in {**late_mods(), **a5_mods()}.items():
            reset()
            late_file(st, ADOPT_FOR_FULL_RUN)
            st0 = dict(S0.STAGES)
            with m.on_s0stop():
                assert S0.STAGES[st] == late_entry(st)
                stop_file(st)
            assert S0.STAGES == st0 and late_verdict(st, base) == NOT_ADOPTED and m.stop_record(base)["decision"] == S0.STOP
        # A4: members records with GNW's or LTX's forms are read when the forms make the members, refused otherwise
        (base / OUT).mkdir(parents=True, exist_ok=True)
        for forms, mem, cols, ok in ((["dep", "gnw"], ["dep", "gnw"], "LTM", True),
                                     (["gnx", "gnp", "ltx"], ["gnx", "ltx"], "LTM+PAIR", True),
                                     (["dep", "ltx"], ["dep"], "LTM", False), (["dep", "ltw"], ["dep", "ltw"], "LTM", False),
                                     (["gnw"], ["gnx"], "LTM", False)):
            M1.WQ.write_json(base / OUT / "members.json", {"declared_in": DECLARED, "forms": forms, "members": mem,
                                                           "gnx_columns": cols})
            assert (members_record(base) is not None) == ok, (forms, mem, cols)

        def ten_file(st, v, mean, drop=None):
            """A5: a later form's calls with every one of the ten reads at one delta (one read left out: drop)."""
            o = base / LATE_NAMES[st][1]
            o.mkdir(parents=True, exist_ok=True)
            rows = [{"split": a, "read": b, "delta": mean, "call": "WITHIN"} for a, b in TEN_READS if (a, b) != drop]
            M1.WQ.write_json(o / "calls.json", {"declared_in": LATE_NAMES[st][3], "screen": {"verdict": v, "reads": rows}})

        def others(v=NOT_ADOPTED):
            calls_file("dep", NO_GAIN)
            for st in ("ltg", "rtr", "ira", "gnx"):
                calls_file(st, v)
            gna_file(v)
            gnp_file(v)

        # A5: with DEP's screen passing, RTR and IRA wait on GNR's and GNS's calls; both other verdicts (or MEMBERS_BY)
        # close them, as before A5; reopen-gate's exits; GNT passing is a form of its own
        reset(late=False)
        for st in LATE:
            late_file(st, NO_GAIN)
        calls_file("dep", ADOPT_FOR_FULL_RUN)
        for st in ("ltg", "gnx"):
            calls_file(st, NO_GAIN)
        gna_file(NO_GAIN)
        gnp_file(NO_GAIN)
        done, rec = fixed()
        assert not done and rec["pending"] == ["rtr", "ira", *A5] and rec["will_not_run"] == [], rec
        slept = []
        assert reopen_call(base, t0)[:2] == (False, 2) and reopen_gate(False, base, t0, slept.append) == 2 and not slept
        late_file("gnt", ADOPT_FOR_FULL_RUN)
        late_file("gnr", NO_GAIN)
        rec = fixed()[1]
        assert rec["pending"] == ["rtr", "ira", "gns"] and not rec["rounds_reopened"], rec
        assert reopen_call(base, t0)[:2] == (False, 2) and reopen_call(base, utc_s(MEMBERS_BY))[:2] == (True, 1)
        late_file("gns", NOT_ADOPTED)
        done, rec = fixed()
        assert done and rec["will_not_run"] == ["rtr", "ira"] and rec["forms"] == ["dep", "gnt"] == rec["members"], rec
        assert rec["pick"] == "dep" and rec["rules"] == ["dep", "ltg"] and rec["layers"] == DEP.LAYERS, rec
        assert reopen_call(base, t0)[:2] == (True, 1) and reopen_gate(True, base, t0, slept.append) == 1 and not slept
        assert M1.WQ.read_json(base / OUT / "reopen.json")["open"] is False and RTR.reopen_record(base) is None
        # A5: the waiting reopen gate looks again after POLL; GNS's screen passing opens RTR and IRA (exit 0)
        late_file("gns", INCOMPLETE)
        assert reopen_gate(True, base, t0, lambda s: slept.append(s) or late_file("gns", ADOPT_FOR_FULL_RUN)) == 0
        assert slept == [POLL] and RTR.reopen_record(base)["open"] is True
        assert RTR.dep_call(base) == f"screen {ADOPT_FOR_FULL_RUN}, {RTR.REOPENED}" == IRA.dep_call(base)
        # A5: reopened beside DEP: the members wait on RTR and IRA; with IRA passing, DEP or IRA by the six-read mean (a
        # tie to DEP), RTR standing behind IRA; the pick is over the passing forms GNS (NO_OWN) is not one of
        for d_ira, keep in ((0.012, "ira"), (0.006, "dep")):
            reset()
            calls_file("dep", ADOPT_FOR_FULL_RUN, {"squad": 0.006})
            for st in ("ltg", "gnx"):
                calls_file(st, NO_GAIN)
            gna_file(NO_GAIN)
            gnp_file(NO_GAIN)
            late_file("gns", ADOPT_FOR_FULL_RUN, {"squad": 0.6})
            done, rec = fixed()
            assert not done and rec["rounds_reopened"] and rec["pending"] == ["rtr", "ira"], rec
            assert rec["will_not_run"] == [] and reopen_call(base, t0)[:2] == (True, 0)
            calls_file("rtr", ADOPT_FOR_FULL_RUN, {"squad": 0.024})
            calls_file("ira", ADOPT_FOR_FULL_RUN, {"squad": d_ira})
            done, rec = fixed()
            assert done and rec["passing"] == ["dep", "rtr", "ira", "gns"], (d_ira, rec)
            assert rec["forms"] == [keep, "gns"] == rec["members"], (d_ira, rec)
            assert rec["dep_or_rounds_means"] == {"dep": 0.001, "ira": round(d_ira / 6, 6)}, rec
            assert rec["pick"] == "rtr" and "gns" not in rec["pick_means"], rec
            assert rec["layers"] == (DEP.LAYERS if keep == "dep" else DEP.BASE_LAYERS), rec
            assert members(False, base, t0) == 0 and forms_of(members_record(base)) == [keep, "gns"]
        # A5: GNW, GNT and GNS pass: one attention form, the largest mean over the ten reads (a tie to the earlier in
        # ATTN; one missing a read kept only when none has all ten); the one form's own full run is its own, none for GNS
        gap = next(k for k in TEN_READS if k not in PICK_READS)
        for means, keep, pick in (((0.004, 0.006, 0.005), "gnt", "gnt"), ((0.006, 0.006, 0.005), "gnw", "gnw"),
                                  ((0.002, None, 0.009), "gns", "gnw"), ((None, None, None), "gnw", "gnw")):
            reset()
            others()
            for st, m in zip(ATTN, means):
                ten_file(st, ADOPT_FOR_FULL_RUN, 0.001 if m is None else m, drop=gap if m is None else None)
            done, rec = fixed()
            assert done and rec["passing"] == list(ATTN) and rec["forms"] == [keep] == rec["members"], (means, rec)
            assert rec["attention_means"] == dict(zip(ATTN, means)) and rec["pick"] == pick, (means, rec)
            assert members(False, base, t0) == 0 and stack_gate(base) == 1
            own = {st: own_call(st, base, t0)[:2] for st in FORMS + ("gna",)}
            assert own == {**{st: (True, 1) for st in FORMS + ("gna",)}, **({} if keep in NO_OWN else {keep: (True, 0)})}, own
        # A5: GNR alone passes: one form, no stack and no own full run; RTR and IRA (their verdicts in) are not rerun
        reset()
        others()
        late_file("gnr", ADOPT_FOR_FULL_RUN, {"squad": 0.6})
        done, rec = fixed()
        assert done and rec["rounds_reopened"] and rec["forms"] == ["gnr"] == rec["members"] and rec["pick"] is None, rec
        assert members(False, base, t0) == 0 and stack_gate(base) == 1
        assert all(own_call(st, base, t0)[:2] == (True, 1) for st in FORMS + ("gna",))
        # A5: GNR's, GNT's and GNS's seed-0 STOPs are NOT_ADOPTED (above, under their modules' entries); members records
        # with two attention forms, or DEP with the rounds, are refused; A5's forms are members of their own
        (base / OUT).mkdir(parents=True, exist_ok=True)
        for forms, ok in ((["dep", "gnr"], True), (["ira", "gnt"], True), (["gnr", "gns"], True), (["gnw", "gnt"], False),
                          (["gnt", "gns"], False), (["dep", "ira"], False), (["dep", "rtr", "gnr"], False)):
            M1.WQ.write_json(base / OUT / "members.json", {"declared_in": DECLARED, "forms": forms, "members": forms,
                                                           "gnx_columns": "LTM"})
            assert (members_record(base) is not None) == ok, forms
    return True


def columns_check():
    """A2: on_columns on LTM's (nothing patched) and LTA's (ltmcols' root, stage and declaration, zgnx's NEXT block and term
    record), restored after, also on a failure; an unknown set refused; member_code on GNA's fits and its four shas.
    A3: on GNP's six alone (ltmcols GNP's, zgnx's block, term and width GNP's) and on two sets (ltmcols LTM's, the joined
    load, the slot's block, term and width); the joined load on made files; member_code on GNP's fits."""
    import shutil
    import tempfile
    import lean_cache as LC
    P = gnp()

    def state():
        L = GNX.LTMC
        return (L.OUT, L.STAGE, L.DECLARED, L.COLS, L.WIDTH, GNX.next_block, GNX.term_record, GNX.IN, GNX.WIDTH,
                GNX.LTM.load_next)

    before = state()
    assert before[:5] == tuple(LTM0.values()) and before[7:] == (15, 7, LOAD_NEXT0)
    with on_columns("LTM"):
        assert state() == before
        assert next_declared("LTM") == GNX.LTM.DECLARED == GNX.next_block("WQ-musique")["declared_in"]
    with on_columns("LTA"):
        assert Path(GNX.LTMC.OUT) == Path(GNA.LTAC.OUT) and GNX.LTMC.DECLARED == GNA.LTAC.DECLARED
        assert GNX.next_block is GNA.next_block and GNX.term_record is GNA.term_record
        assert next_declared("LTA") == GNA.LTAC.DECLARED == GNX.next_block("WQ-musique")["declared_in"]
        assert GNX.next_block("WQ-musique")["columns"] == list(GNX.LTMC.COLS)
    assert state() == before
    try:
        with on_columns("LTA"):
            raise KeyError("x")
    except KeyError:
        pass
    assert state() == before
    for bad in ("PAIR+LTM", "LTM+LTA", "X"):
        try:
            with on_columns(bad):
                raise AssertionError(f"an unknown column set {bad} passed")
        except SystemExit:
            pass
    assert state() == before
    with on_columns(PAIR):
        assert (Path(GNX.LTMC.OUT), GNX.LTMC.STAGE, GNX.LTMC.DECLARED) == (ROOT / GNP_OUT, "GNP", GNP_DECLARED)
        assert (tuple(GNX.LTMC.COLS), GNX.LTMC.WIDTH, GNX.IN, GNX.WIDTH) == (PAIR_COLS, 6, 13, 6)
        assert GNX.next_block is P.next_block and GNX.term_record is P.term_record and GNX.LTM.load_next is LOAD_NEXT0
        nb = GNX.next_block("WQ-musique")
        assert (nb["declared_in"], nb["columns"], Path(nb["columns_root"]).name) == (GNP_DECLARED, list(PAIR_COLS), "gnp")
        assert next_declared(PAIR) == GNP_DECLARED and slot_in(PAIR) == GNX.IN
    assert state() == before
    for cols, decl in (("LTM+PAIR", GNX.LTM.DECLARED), ("LTA+PAIR", GNA.LTAC.DECLARED)):
        with on_columns(cols):
            assert state()[:5] == before[:5] and (GNX.IN, GNX.WIDTH) == (27, 13) == (slot_in(cols), 13)
            assert GNX.LTM.load_next is not LOAD_NEXT0 and GNX.next_block is not before[5]
            nb = GNX.next_block("WQ-musique")
            assert nb["declared_in"] == [decl, GNP_DECLARED] == next_declared(cols), nb
            assert nb["columns"] == list(LTM0["COLS"]) + list(PAIR_COLS) == slot_columns(cols), nb
            assert [Path(r).name for r in nb["columns_root"]] == [cols.split("+")[0].lower(), "gnp"], nb
        assert state() == before
    try:
        with on_columns("LTA+PAIR"):
            raise KeyError("x")
    except KeyError:
        pass
    assert state() == before
    # A3: the joined load on made files: LTM's (or LTA's) seven then GNP's six of one carve, one float16 block in block
    # order, the scorer the sets' joined; a set's columns under the other's root stop it
    tmp = Path(tempfile.mkdtemp(prefix="zstk_c_"))
    try:
        rng = np.random.default_rng(41)
        LTW = P.PP.LTW
        LTW.toy_tree(tmp, "musique", "s1eval", LTW.unit(rng.normal(size=(200, 16))).astype(np.float32), rng, nq=6, n=20)
        parts, _recs = LC.part_dirs("musique", "s1eval", tmp / "cache")
        ps = {p.name: LC.sha_file(p / "record.json") for p in parts}
        sp = "WQ-hotpotqa"
        X = {"LTM": GNX.LTM.toy_cols(tmp / "ltm", sp, "musique", "s1eval", tmp / "cache", rng, "e" * 64)}
        with GNA.ZLA.on_cols(tmp / "lta"):
            X["LTA"] = GNX.LTM.toy_cols(tmp / "lta", sp, "musique", "s1eval", tmp / "cache", rng, "f" * 64)
        with P.on_cols(tmp / "gnp"):
            X[PAIR] = P.toy_cols6(tmp / "gnp", sp, "musique", "s1eval", tmp / "cache", rng, "b" * 64)
        roots = {"LTM": tmp / "ltm", "LTA": tmp / "lta", PAIR: tmp / "gnp"}
        assert state() == before
        for cols, sha in (("LTM+PAIR", "e" * 64), ("LTA+PAIR", "f" * 64)):
            nx = next_part(cols)
            with on_columns(cols):
                Y, rec, rsha = GNX.LTM.load_next("musique", "s1eval", sp, ps, {p: roots[p] for p in col_parts(cols)})
                assert Y.dtype == np.float16 and Y.shape == (X[nx].shape[0], 13), (Y.dtype, Y.shape)
                assert np.array_equal(Y[:, :7], X[nx], equal_nan=True) and np.array_equal(Y[:, 7:], X[PAIR], equal_nan=True)
                assert rec["fit_sha256"] == f"{sha}+{'b' * 64}" and rec["columns"] == slot_columns(cols), rec
                assert [rec["sets"][p]["declared_in"] for p in col_parts(cols)] == next_declared(cols), rec
                assert rec["training_carve"] is False and rec["rows"] == Y.shape[0] and rsha.count("+") == 1, rec
                P.must_stop(GNX.LTM.load_next, "musique", "s1eval", sp, ps, {nx: roots[PAIR], PAIR: roots[PAIR]})
                P.must_stop(GNX.LTM.load_next, "musique", "s1eval", sp, ps, {nx: roots[nx], PAIR: roots[nx]})
            assert state() == before
        with on_columns(PAIR):
            Z = GNX.LTM.load_next("musique", "s1eval", sp, ps, roots[PAIR])[0]
            assert np.array_equal(Z, X[PAIR], equal_nan=True)
            P.must_stop(GNX.LTM.load_next, "musique", "s1eval", sp, ps, roots["LTM"])
        assert state() == before
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    me, cur = sys.modules[__name__], "c" * 64
    keys = ("zgnx_sha256",) + tuple(f"{k}_sha256" for k in GNA.OWN_CODE)

    def code_of(theirs, mine):
        with M1.WQ.patch(me, is_read=lambda f, base=ROOT: f in GNA.FITS), \
                M1.WQ.patch(GNA, fit_block=lambda f, what, base=ROOT: {"code": dict(theirs)}):
            return member_code("s", {"code": dict(mine)}, ["gnx"], ROOT, "LTA")

    same = {k: cur for k in keys}
    assert code_of(same, same) == []
    assert len(code_of(same, dict(same, zlta_sha256="d" * 64))) == 2           # GNA's two seed-0 fits name another
    assert len(code_of(dict(same, ltacols_sha256=None), same)) == 2
    with M1.WQ.patch(me, is_read=lambda f, base=ROOT: f in GNX.FITS):          # GNX's fits read, GNA's not
        assert len(member_code("s", {"code": same}, ["gnx"], ROOT, "LTA")) == 2
    # A3: with the pair columns, GNP's two seed-0 fits, and zgnp's, gnpcols' and zpfp's shas beside zgnx's, count beside
    # the NEXT set's own fits
    pkeys = ("zgnx_sha256",) + tuple(f"{k}_sha256" for k in GNP_CODE)

    def pcode_of(theirs, mine, cols):
        with M1.WQ.patch(me, is_read=lambda f, base=ROOT: f in P.FITS or f in GNX.FITS), \
                M1.WQ.patch(P, fit_block=lambda f, what, base=ROOT: {"code": dict(theirs)}), \
                M1.WQ.patch(GNX, fit_block=lambda f, what, base=ROOT: {"code": dict(theirs)}):
            return member_code("s", {"code": dict(mine)}, ["gnx"], ROOT, cols)

    psame = {k: cur for k in pkeys}
    assert pcode_of(psame, psame, PAIR) == [] and pcode_of(psame, psame, "LTM+PAIR") == []
    assert len(pcode_of(psame, dict(psame, zpfp_sha256="d" * 64), PAIR)) == 2          # GNP's two fits name another
    assert len(pcode_of(dict(psame, gnpcols_sha256=None), psame, "LTM+PAIR")) == 2
    assert len(pcode_of(dict(psame, zgnx_sha256="d" * 64), psame, "LTM+PAIR")) == 4    # GNX's two and GNP's two
    with M1.WQ.patch(me, is_read=lambda f, base=ROOT: f in GNX.FITS):                  # GNX's fits read, GNP's not
        assert len(member_code("s", {"code": psame}, ["gnx"], ROOT, PAIR)) == 2
    assert [m for m, _k in own_of("gnx", "LTA+PAIR")] == [GNA, P] and own_of("dep") == [(DEP, ("zdep_sha256",))]
    return True


def rules_check():
    """The combination of the rules (section 4) on made blocks, and part_problems on made stamps."""
    def blk(v, rows=(), side=()):
        return {"verdict": v, "reads": list(rows), "beside": list(side), "problems": []}

    def row(sp, rd, c):
        return {"split": sp, "read": rd, "call": c}

    A, N, X, I = ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, INCOMPLETE
    assert combine_screen([blk(A), blk(A)])["verdict"] == A and combine_screen([blk(A), blk(N)])["verdict"] == N
    assert combine_screen([blk(N), blk(X)])["verdict"] == X and combine_screen([blk(A), blk(I)])["verdict"] == I
    assert combine_screen([blk(A)])["verdict"] == A and combine_screen([blk(X)])["verdict"] == X
    r1 = [row("WQ-musique", "webqsp", "WITHIN"), row("WQ-musique", "metaqa", "LOSS")]
    r2 = [row("WQ-musique", "webqsp", "WITHIN")]
    c = combine_screen([blk(X, r1), blk(A, r2, [row("WQ-musique", "metaqa", "LOSS (beside)"),
                                                row("WQ-hotpotqa", "hotpotqa", "WITHIN")])])
    assert [(r["read"]) for r in c["reads"]] == ["webqsp", "metaqa"] and [r["read"] for r in c["beside"]] == ["hotpotqa"], c
    assert c["counts"] == {RULE.GAIN: 0, RULE.LOSS: 1, RULE.WITHIN: 1}, c
    assert combine_full([blk(ADOPT), blk(ADOPT)])["verdict"] == ADOPT and combine_full([blk(ADOPT), blk(X)])["verdict"] == X
    assert combine_full([blk(ADOPT), blk(I)])["verdict"] == I
    assert rules_of(["ltg", "gnx"]) == ["ltg"] and rules_of(["dep", "gnx"]) == ["dep", "ltg"] and rules_of(["ira"]) == ["dep"]
    assert layers_of(["dep", "ltg"]) == 4 and layers_of(["ltg", "ira"]) == 3
    assert ltg_terms(4) == 75276 and ltg_terms(3) == 68745
    mem = ["dep", "ltg", "gnx"]
    k = {"members": mem, "layers": 4, "parts": {
        "dep": {"layers": 4}, "ltg": {"term_parameters": 75276, "rank": LTG.RANK, "axes": LTG.AXES,
                                      "basis_npz_sha256": LTG.BASIS_SHA, "lambda": LTG.LAMBDA, "layers": 4},
        "gnx": {"term_parameters": GNX.HIDDEN * GNX.IN, "inputs": GNX.IN, "columns": list(GNX.LTMC.COLS), "layers": 4}}}
    assert part_problems("f", "train", k, mem) == []
    bad = json.loads(json.dumps(k))
    bad["parts"]["ltg"]["term_parameters"] = LTG.N_TERM                          # LTG's three-layer count under DEP
    assert len(part_problems("f", "train", bad, mem)) == 1
    assert len(part_problems("f", "train", dict(k, layers=3), mem)) == 1
    assert len(part_problems("f", "train", dict(k, members=["dep", "ltg"]), mem)) == 1
    kp = json.loads(json.dumps(k))                                             # A3: the gnx part on two sets
    kp["parts"]["gnx"] = {"term_parameters": 3456, "inputs": 27, "columns": list(LTM0["COLS"]) + list(PAIR_COLS),
                          "layers": 4}
    assert part_problems("f", "train", kp, mem, "LTM+PAIR") == [] and len(part_problems("f", "train", kp, mem)) == 1
    assert len(part_problems("f", "train", k, mem, "LTM+PAIR")) == 1 and part_problems("f", "train", k, mem, "LTA") == []
    assert (slot_in(PAIR), slot_in("LTA+PAIR"), slot_in("LTM"), slot_in("LTA")) == (13, 27, 15, 15)
    assert GNX.HIDDEN * slot_in("LTM+PAIR") == 3456 and slot_columns("LTA") == list(GNX.LTMC.COLS)
    # zdep's and zltg's own screen calls on one made input: each read both rules call gets one row from both
    rng = np.random.default_rng(5)
    new_all, base_all = {}, {}
    for key in DEP.read_keys():
        b = 0.5 + 0.02 * rng.random(len(RULE.SEEDS))
        base_all[key] = {sd: float(b[i]) for i, sd in enumerate(RULE.SEEDS)}
        new_all[key] = {0: float(b.mean() + rng.normal(0.0, 0.012))}
    blocks = [DEP.decide_screen(new_all, base_all), LTG.decide_screen(new_all, base_all)]
    rows, side, bad = union_rows(blocks)
    assert not bad and len(rows) == len(set(DEP.read_keys())) == 16 and not side, (bad, len(rows), len(side))
    both = set(LTG.deciding_keys()) & {(r["split"], r["read"]) for r in blocks[0]["reads"]}
    assert len(both) == 10, both
    c = combine_screen(blocks)
    assert c["verdict"] in VERDICTS and not c["problems"] and sum(c["counts"].values()) == 16, c
    # member_code on made stamps: one sha in both own fits passes, another fails; EARLIER's passes on its own fit alone
    me, cur = sys.modules[__name__], "c" * 64

    def code_of(shas, mine=cur, read=True):
        with M1.WQ.patch(me, is_read=lambda f, base=ROOT: read), \
                M1.WQ.patch(LTG, fit_block=lambda f, what, base=ROOT: {"code": {"zltg_sha256": shas[f]}}), \
                M1.WQ.patch(DEP, fit_block=lambda f, what, base=ROOT: {"code": {"zdep_sha256": cur}}):
            return member_code("s", {"code": {"zltg_sha256": mine, "zdep_sha256": cur}}, ["dep", "ltg"])

    s0, hp, old = LTG.fit_name(0, "WQ-musique"), LTG.fit_name(0, "WQ-hotpotqa"), EARLIER[("ltg", "WQ-musique")]
    assert code_of({s0: cur, hp: cur}) == [] and code_of({s0: old, hp: cur}) == []
    assert len(code_of({s0: cur, hp: old})) == 1 and len(code_of({s0: "d" * 64, hp: cur})) == 1
    assert len(code_of({s0: cur, hp: cur}, mine=None)) == 2 and len(code_of({s0: cur, hp: cur}, read=False)) == 4
    assert is_read.__module__ == __name__ and LTG.fit_block.__module__ == "zltg"
    return True


def selftest():
    assert fit_name(0, "WQ-musique") == "scr-stk-st-s0" and fit_name(2, "WQ-hotpotqa") == "scr-stk-st-s2-hp"
    assert pair_name(1, "WQ-hotpotqa") == "scr-stk-st-vs-base-s1-hp"
    assert pairs(ISO.KEEP)[pair_name(0, "WQ-musique")] == ("scr-stk-st-s0", DEP.base_fit(0, "WQ-musique", ISO.KEEP))
    assert not set(FITS) & set(GNX.LTM.FITS)
    assert [f"{a} {b}" for a, b in PICK_READS] == [f"WQ-hotpotqa {d}" for d in ("squad", "musique", "2wiki", "webqsp",
                                                                                "musique-3hop", "musique-4hop")]
    assert utc_s(MEMBERS_BY) - utc_s("2026-11-05T00:00:00Z") == 6.5 * 3600 and utc_s(FREEZE) > utc_s(MEMBERS_BY)
    # 0. A3: zgnp on first use, before any patch: its names, columns, width and code A3's; the thread variables kept
    env0 = {v: os.environ.get(v) for v in THREAD_VARS}
    P = gnp()
    assert gnp() is P and {v: os.environ.get(v) for v in THREAD_VARS} == env0 and set(P.FITS) == {
        gnp_fit_name(sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
    assert gnp_entry()[2:4] == ("scr-gnp-px-s0", "scr-gnp-px-s0-hp") and GNP_KEY not in S0.STAGES
    assert set(code_extra()) >= {f"{k}_sha256" for k in GNP_CODE}
    # 0b. A4: zgnw and zltx after zgnp, outside every patch: their names, fits and stop entries A4's; the threads kept;
    # the code on disk's `composes` runs in a process of its own
    L = late_mods()
    assert late_mods() is L and set(L) == set(LATE) and {v: os.environ.get(v) for v in THREAD_VARS} == env0
    assert late_entry("ltx")[2:4] == ("scr-ltx-xf-s0", "scr-ltx-xf-s0-hp") and not set(LATE) & set(S0.STAGES)
    assert COMPOSED == ORDER and disk_composes_rc() in (0, 1, 2)
    # 1. the rules' combination and the stamp's parts; the gates on made files; A2's and A3's columns
    assert rules_check() and gates_check() and columns_check()
    # 2. zltg's names at three layers are its own
    G = M1.import_arm(ARM)[0]
    own = (LTG.LAYERS, LTG.TERM_LAYER, LTG.TERM_NAMES, LTG.N_TERM)
    with on_ltg_layers(G):
        assert (LTG.LAYERS, LTG.TERM_LAYER, LTG.TERM_NAMES, LTG.N_TERM) == own
    assert (LTG.LAYERS, LTG.TERM_LAYER, LTG.TERM_NAMES, LTG.N_TERM) == own
    # 3. the stacks without DEP on the toy carve
    sets = [("ltg", "gnx"), ("ltg", "rtr"), ("ltg", "ira"), ("rtr", "gnx"), ("ira", "gnx"), ("ltg", "rtr", "gnx"),
            ("ltg", "ira", "gnx")]
    t = toy_stack(G, sets, ("ltg", "ira", "gnx"))
    check_toy(t, sets, ("ltg", "ira", "gnx"), DEP.BASE_LAYERS)
    t2 = toy_stack(G, [("ltg", "rtr")], ("ltg", "rtr", "gnx"))
    check_toy(t2, [("ltg", "rtr")], ("ltg", "rtr", "gnx"), DEP.BASE_LAYERS)
    # 3b. A3: the gnx slot on LTM's columns then GNP's six (IN 27), and on GNP's six alone (IN 13)
    t3 = toy_stack(G, [("ltg", "gnx")], ("ltg", "gnx"), "LTM+PAIR")
    check_toy(t3, [("ltg", "gnx")], ("ltg", "gnx"), DEP.BASE_LAYERS, "LTM+PAIR")
    t4 = toy_stack(G, [], ("ltg", "gnx"), PAIR)
    check_toy(t4, [], ("ltg", "gnx"), DEP.BASE_LAYERS, PAIR)
    # 4. the stacks with DEP, in a process of its own (zg6 is imported here with three layers)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    pr = subprocess.run([sys.executable, str(Path(__file__)), "layers-check"], capture_output=True, text=True, timeout=1800,
                        env=env)
    line = [x for x in (pr.stdout or "").splitlines() if x.startswith("LAYERS_CHECK ")]
    if pr.returncode or not line:
        raise AssertionError(f"layers-check exited {pr.returncode}: {(pr.stderr or '')[-3000:]}")
    lc = json.loads(line[0][len("LAYERS_CHECK "):])
    assert (lc["layers"], lc["term_layer"], lc["term_names"], lc["n_term"]) == (4, 4, 2 + 16, 75276), lc
    assert lc["term_layer_last"] == ["ltg_3_w", "ltg_3_g", "ltg_3_em", "ltg_3_ea"] and lc["after"] == [3, 3, 68745], lc
    log(f"selftest: names, pairs and the pick's six reads; the rules' combination and the stamp's parts; the member-code "
        f"rule on made stamps (EARLIER's one fit); members, "
        f"stack-gate, own-gate and the stack's s0stop on made files; A2: GNA's verdict, the gnx slot's columns, its pick "
        f"mean and own run, on_columns and member_code on GNA's fits; A3: zgnp on first use (the threads kept), GNP's "
        f"verdict and STOP, the forms, the gnx slot's columns (two sets or the six alone), the pick over the forms, the "
        f"own runs by forms (GNP's own), the members records refused when they disagree, on_columns on GNP's six and on "
        f"two sets, the joined load on made files, member_code on GNP's fits; A4: zgnw's and zltx's names, fits and stop "
        f"entries, GNW's and LTX's verdicts and STOPs, their forms and members, the waiting stack gate (one look a call, "
        f"decided once, MEMBERS_BY), the own gates on a stack that does not run, composes on made files and on disk, the "
        f"members records with their forms; zltg's names at three layers; on the toy "
        f"carve, {len(sets) + 3} stacks without DEP (two of them on the pair columns) and {len(lc['toy']['sets'])} with "
        f"DEP (a process of its own, four layers) each "
        f"start as the set without LTG and GNX bit for bit (eval and training forwards); every member's parameters get the "
        f"listwise and next-hop losses' gradient; the strict loads; a one-epoch fit_variant moves each zero-started term; "
        f"the stamp's record and refusals; the patches restored (also on a failure)")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("verb", nargs="?", choices=("members", "stack-gate", "own-gate", "reopen-gate", "fit", "s0stop-check",
                                                "s0stop-run", "gate", "compare-all", "calls", "layers-check", "composes"))
    ap.add_argument("--name")
    ap.add_argument("--stage", choices=ORDER + ("gna", "gnp") + LATE + A5)
    ap.add_argument("--want")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.verb == "members":
        return members(a.wait)
    if a.verb == "stack-gate":
        return stack_gate(wait=a.wait)
    if a.verb == "composes":
        return composes_now()
    if a.verb == "own-gate":
        if not a.stage:
            ap.error("own-gate needs --stage")
        return own_gate(a.stage, a.wait)
    if a.verb == "reopen-gate":
        return reopen_gate(a.wait)
    if a.verb == "fit":
        return fit(a.name)
    if a.verb == "s0stop-check":
        with on_s0stop():
            return S0.check("stk", ROOT, S0.run_stage, stk_keys)
    if a.verb == "s0stop-run":
        with on_s0stop():
            return S0.run("stk", cmd)
    if a.verb == "gate":
        if not a.want:
            ap.error("gate needs --want")
        with on_s0stop():
            return S0.gate("stk", [w for w in a.want.split(",") if w])
    if a.verb == "compare-all":
        return compare_all()
    if a.verb == "calls":
        return calls()
    if a.verb == "layers-check":
        return layers_check()
    ap.error("a verb or --selftest")


if __name__ == "__main__":
    sys.exit(main())
