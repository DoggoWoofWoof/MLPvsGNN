"""R2EG (docs/R2EG_R2E_GNN_UNDER_WQ.md): R2e, the GNN half, under WQ on Kaggle's T4, against WQK's base set
(docs/WQK_WQ_ON_T4.md). zg6e is zg6 (zg6.py, imported unchanged: the free GNN, its encoder, boundary, context, three
layers and readout) with R2's bridge scores inside its hidden update, for the bridges linked to the row only. R2's logit
s_vk of the row v and its bridge k (zr2br.py, imported unchanged: the five rows with the highest first-stage RRF, the
weights U, V, W1, W2, o, and the EM term) is summed over the row's linked bridges, m(v), and enters each of zg6's three
layers as m(v) * e_l before layer_norm, e_l a learned direction of 64 numbers starting at 0, so zg6e starts as zg6 bit for
bit. link(v, k) is R2EW's flag (zr2ew.py, imported unchanged): 1 when zlink's built edges hold an edge between the
bridge's row and v, fixed per carve from the links arrays and the stored RRF column, never a learned value or a score.
R2's EM term trains the logits on the linked slots. zg6e is trained by zg6's train (the arm and fit names added in
process) with zr2br's EM term, at WQK's names and roots with WQK's mixed KT input (zwqk.py, imported unchanged), as WQK
trains zg6, and read on the device it was fitted on with the link statistics per carve kept. Each fit is compared with WQK's
seed-0 zg6 fit of its split (first: it decides) and with WQK's seed-0 zspk fit (beside), all on one device. No zg6 seed null
is filed under WQ, so zg6e's re-call against zg6 is read under WQK's zspk seed null with its seed-0 R@5 replaced by zg6's
(G6's form; docs/R2EG_R2E_GNN_UNDER_WQ.md section 4).

In the kernel (zr2eg_session.py runs each in its own process, after zwqk.py overlay and gate):

    python outputs/mp_unified/zr2eg.py fit --name scr-r2eg-zg6-hp --rows-cap 65536

On the laptop, after `kaggle kernels output` of the session into a new folder:

    python outputs/mp_unified/zr2eg.py take --from DIR   # each fit's records and reads, never its weights
    python outputs/mp_unified/zr2eg.py compare-all       # outputs/screen/scr-r2eg-zg6-vs-zg6[-hp].json
    python outputs/mp_unified/zr2eg.py verdict           # outputs/r2eg/screen-verdict (.json, .md)
    python outputs/mp_unified/zr2eg.py --selftest

This file imports the standard library, zwqk (with it zwq), zr2gw and zr2ew only; zr2br, zg6, zlink and torch are
imported by the commands that need them, outside zwq.on_wq. verdict exits 0 on ADOPT_FOR_FULL_RUN or NOT_ADOPTED and 2
after writing an INCOMPLETE record.
"""
import contextlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zr2ew as EW  # noqa: E402  (the standard library, zwq, zwqk and zr2gw at import)

GW, K, WQ = EW.GW, EW.K, EW.WQ
DECLARED = "docs/R2EG_R2E_GNN_UNDER_WQ.md"
STAGE = "R2EG"
KEY = "r2eg"                                  # the stamp's key in screen.json, read.json and R2EG's compares
ARM, BASE, BESIDE = "zg6e", "zg6", "zspk"
SEED = 0
SPLITS = tuple(GW.SPLITS)
READ_CAP = GW.READ_CAP                        # WQK's rows a read batch (whole questions, so the scores do not change)
NB = GW.NB                                    # R2's bridges per question
R2_CONST, R2_DECLARED = GW.R2_CONST, GW.R2_DECLARED
HEAD_PARAMS = ("r2_w1", "r2_b1", "r2_w2", "r2_b2")            # R2's head: zg6e has none
BRIDGE_PARAMS = ("r2_U", "r2_V", "r2_W1", "r2_W2", "r2_o")    # R2's bridge logits: 1536 x 32, 257 x 32, 2 x 32 x 32, 32
DIR_PARAM = "g6e_dir"                                         # the 3 layers' directions, 3 x 64
NEW_PARAMS = BRIDGE_PARAMS + (DIR_PARAM,)
BRIDGE_N, DIR_N = 59456, 192
NEW_N = BRIDGE_N + DIR_N                      # 59,648 numbers over zg6's own
assert GW.R2_N - BRIDGE_N == 81 and tuple(GW.R2_PARAMS) == BRIDGE_PARAMS + HEAD_PARAMS
FIT = {sp: f"scr-r2eg-{BASE}{WQ.SUF[sp]}" for sp in SPLITS}
FITS = {n: sp for sp, n in FIT.items()}       # name -> split
BASE_FIT = {sp: K.fit_name(BASE, 0, sp) for sp in SPLITS}         # WQK's seed-0 zg6: the deciding base
BESIDE_FIT = {sp: K.fit_name(BESIDE, 0, sp) for sp in SPLITS}     # WQK's seed-0 zspk: G6's base, beside
CMP = {sp: f"scr-r2eg-{BASE}-vs-{BASE}{WQ.SUF[sp]}" for sp in SPLITS}
COMPARES = {CMP[sp]: (FIT[sp], [BASE_FIT[sp], BESIDE_FIT[sp]]) for sp in SPLITS}
NEW = "outputs/r2eg/"
SESSIONS = NEW + "sessions"
VERDICT = NEW + "screen-verdict"
CALLS = {"zg6e": 0, "zg6e-vs-zspk": 1}                         # call -> the compare's base column
AGAINST = {"zg6e": "zg6", "zg6e-vs-zspk": "zspk"}
DECIDING = {"zg6e": True, "zg6e-vs-zspk": False}
PAIR = {c: f"{NEW}r2eg-{c}-pair" for c in CALLS}
RECALL = {c: f"{p}-recall" for c, p in PAIR.items()}
NULL_KIND = {"zg6e": "WQK's zspk seed null, its seed-0 R@5 replaced by WQK's zg6 read (G6's form: no zg6 seed null is filed)",
             "zg6e-vs-zspk": "WQK's zspk seed null"}
FLOORS = K.FLOORS[BESIDE]
TIMING_FILE = "r2eg_timing.json"
CODE = tuple(dict.fromkeys(EW.CODE + ("zr2eg",)))
LSTAT = EW.LSTAT
WHAT = "zg6 + R2's linked-bridge scores in its hidden update"
NAMING = ("zg6e is a GNN arm, called \"" + WHAT + "\"; the user's open ruling on zrckpb (docs/R2_BRIDGE_EM.md section 2: "
          "whether reading the five bridges' vectors keeps an arm inside the MLP track) is about the MLP track and does "
          "not apply to it")
R2E_TEXT = GW.R2E
R2EG = {"form": "zg6's hidden update gets m(v) * e_l in each of its three layers: pre_l = (zg6's pre-activation) + m(v) * e_l, "
                "then layer_norm and the residual as zg6's",
        "m": "m(v) = the sum over the slots k with M(v, k) and link(v, k) of s_vk, R2's bridge logit of the row v and the "
             "bridge k (r2_U, r2_V, r2_W1, r2_W2, r2_o; no head); 0 when the row has no linked bridge",
        "link": EW.R2E["link"],
        "fixed": "a function of the links arrays and the stored c_rrf column only, from all of the question's edges (not "
                 "the edges the family dropout leaves); no learned value, score or label",
        "dir": "e_l, one learned direction of 64 numbers per layer (a constant added to every channel would be removed by "
               "layer_norm)",
        "params_added": NEW_N, "params": {"bridge logits": BRIDGE_N, "directions e_l": DIR_N},
        "init": "e_l = 0: zg6e starts as zg6 bit for bit; with every flag 0 it is zg6 for any weights",
        "em": "R2's EM term (LAM 1) on the linked slots trains the logits, as R2EW trains them",
        "mask_read_by": ["m(v)", "R2's EM term and prior"]}
RULE = ("ADOPT_FOR_FULL_RUN when zg6e's re-call against WQK's seed-0 zg6 under WQK's zspk seed null, its seed-0 R@5 replaced "
        "by zg6's, has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when "
        "WQK's base verdict is not BASE_READY, a fit or compare is missing or not R2EG's, R2EG's fits and WQK's seed-0 zg6 "
        "and zspk fits are not on one device, a fit or read names a gate that is not a filed PASS gate of an R2EG session, "
        "a fit's R2 constants, parameters or EM term are not as declared, zg6e's message is not as declared or its read "
        "holds no link statistics on the six carves, WQK's floors are not their compares', or a re-call does not hold the "
        "twelve reads (docs/R2EG_R2E_GNN_UNDER_WQ.md section 5)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zg6e fitted again on PKO's pools (docs/PKO_PICK_O_POOLS_UNDER_WQ.md, outputs/pko), then its "
                              "full run under WQ beside zg6's, and zg6's own seed null (zg6 at seeds 1 and 2) declared "
                              "before the full run's calls, each its own declared file",
        "NOT_ADOPTED": "R2's GNN line closes: zg6m and zg6e are NOT_ADOPTED, and a verdict closes only its form (R2WS's "
                       "confirmation of zrckpb at the null seeds is the MLP line's, in its own file). The next declared "
                       "forms are those waiting on PKO's pools (U2n's zrckpt on U2s's pick, G6RW's zg6r on PKO's pools) "
                       "and the typed channels of docs/PROGRAM_2026_10_24.md section 4.3 item 5 (G6 with EM-typed "
                       "messages over EL, R and kNN), each its own declared file",
        "INCOMPLETE": "what is missing is supplied (WQK's base set, a fit, a compare, a session's gate or a re-call); "
                      "nothing else changes"}
assert len(FIT) == len(FITS) == 2 and not set(FITS) & (set(K.FITS) | set(GW.FITS) | set(EW.FITS))
assert all(CMP[sp] not in K.COMPARES and CMP[sp] not in GW.COMPARES and CMP[sp] not in EW.COMPARES for sp in SPLITS)
assert all(BASE_FIT[sp] in K.FITS and BESIDE_FIT[sp] in K.FITS for sp in SPLITS) and FLOORS == "outputs/wqk/floors-zspk"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zr2eg: {msg}", flush=True)


# ── zr2br and zg6, imported outside zwq.on_wq, and zg6e's classes ────────────

_G = {}


def import_g():
    """zr2ew's import (zr2br through zr2gw's: torch, zg6, zp1c3p and R2's arms in lean_screen's table), then zg6e's model
    and carve, built once and put in lean_screen's table. Returns {"R2", "G6", "np", "torch", "link_flags", "stats_add",
    "g6e_layers", "g6e_stack", "ZG6E", "g_carve", "GCarve"}."""
    if _G:
        return _G
    R = EW.import_r2()
    R2, np, torch = R["R2"], R["np"], R["torch"]
    link_flags, stats_add = R["link_flags"], R["stats_add"]
    import torch.nn as nn
    import torch.nn.functional as Fn
    G6 = R2.G6
    D, FAMS, LCLIP, ZCLIP, LAYERS = G6.D, G6.FAMS, G6.LCLIP, G6.ZCLIP, G6.LAYERS
    want = (64, 3, 3, 8.0)
    if (D, FAMS, LAYERS, LCLIP) != want or G6.G6_LAYER != tuple(tuple(f"g6_{k}_{p}" for p in (
            "msg", "att", "fb", "self", "mean", "agg", "x", "b")) for k in range(3)):
        raise SystemExit(f"zr2eg: zg6's layer is {(D, FAMS, LAYERS, LCLIP)}, not the one R2EG adds to {want}")

    def g6e_layers(m, h, xb, w, u, v, f, nq, B, mb):
        """G6.g6_layers, op for op, with mb (N,) the bridge message: the layer's pre-activation gets mb * e_l before
        layer_norm. mb * 0 adds 0, so with e_l = 0 the layers are zg6's bit for bit."""
        N = h.shape[0]
        dev, dt = h.device, h.dtype
        E = u.numel()
        at = v * FAMS + f
        one = torch.ones(E, dtype=dt, device=dev)
        deg = torch.zeros(N * FAMS, dtype=dt, device=dev).index_add_(0, at, one).view(N, FAMS, 1).clamp_min(1.0)
        qv = nq[v]
        mb = mb.to(dt).unsqueeze(1)
        for l, (wm, wa, fb, ws, wmean, wagg, wx, b) in enumerate((getattr(m, n) for n in lay) for lay in G6.G6_LAYER):
            c = torch.zeros(B, D, dtype=dt, device=dev).index_add_(0, nq, w * h)
            hm = Fn.linear(h, wm).view(N, FAMS, D)
            msg = hm[u, f]
            lg = ((Fn.linear(h, wa)[u] * c[qv]).sum(1) / math.sqrt(D) + fb[f]).clamp(-LCLIP, LCLIP)
            mean = torch.zeros(N * FAMS, D, dtype=dt, device=dev).index_add_(
                0, at, torch.sigmoid(lg).unsqueeze(1) * msg).view(N, FAMS, D) / deg
            ex = torch.exp(lg)
            num = torch.zeros(N, D, dtype=dt, device=dev).index_add_(0, v, ex.unsqueeze(1) * msg)
            den = torch.zeros(N, dtype=dt, device=dev).index_add_(0, v, ex)
            att = num / den.clamp_min(1e-30).unsqueeze(1)
            pre = Fn.linear(h, ws) + Fn.linear(mean.reshape(N, FAMS * D), wmean) + Fn.linear(att, wagg) + \
                Fn.linear(xb, wx) + b
            pre = pre + mb * getattr(m, DIR_PARAM)[l]
            h = h + Fn.gelu(Fn.layer_norm(pre, (D,)))
        return h

    def g6e_stack(m, H, xb, lk, nq, B, u01, mb):
        """G6.g6_stack with the bridge message mb (N,) in the layers."""
        xb = xb.to(H.dtype)
        h0 = Fn.gelu(Fn.linear(torch.cat([H, xb], 1), m.g6_in_w, m.g6_in_b))
        s0 = Fn.linear(h0, m.g6_ctx_w).squeeze(-1)
        z0 = G6.LG.seg_zscore8D(s0.unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP)
        w = G6.G2.seg_softmax_rows(z0, nq, B).unsqueeze(1)
        u, v, f = G6.edges_of(lk, nq, B, u01, H.device)
        h = g6e_layers(m, h0, xb, w, u, v, f, nq, B, mb)
        return (Fn.linear(h, m.g6_out_w) + Fn.linear(h0, m.g6_skip_w) + m.g6_out_b).squeeze(-1)

    class ZG6E(R2.R2Mix, G6.ZG6):
        """zg6e: zg6 with R2's linked-bridge scores in its hidden update (R2e, the GNN half). R2Mix gives R2's logit
        weights and the hook the EM term patches (r2_feats); its head is deleted, as zg6 deletes zrm's output layer."""

        def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
            super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
            for n in HEAD_PARAMS:
                delattr(self, n)
            setattr(self, DIR_PARAM, nn.Parameter(torch.zeros(LAYERS, D)))
            self._r2_last = None

        def r2_parts(self, r2, nq):
            """R2's logits S and the mask M(v, k) and link(v, k), and the prior over the linked slots (R2EW's mask)."""
            S_, M = R2.r2_logits(self, r2["eq"], r2["p"], r2["bri"], nq)
            lk = r2.get("lk")
            if lk is None:
                raise SystemExit("zr2eg: zg6e's batch holds no link flags (its carve is not R2EG's)")
            M = M & lk
            self._r2_last = (S_, M)
            return S_, M, R2.prior_logp(M, S_.dtype)

        def bridge_message(self, feats, nq):
            """m(v) (N,): the sum of the row's linked bridges' logits. R2's r2_feats is the hook the EM term patches (it
            keeps the batch's logits, mask and prior in training), so it runs once and r2_parts leaves S and M here."""
            r2 = feats.get(R2.R2_KEY)
            if r2 is None:
                raise SystemExit("zr2eg: zg6e's batch holds no bridges (its carve is not R2EG's)")
            self._r2_last = None
            self.r2_feats(r2, nq)
            S_, M = self._r2_last
            self._r2_last = None
            return torch.where(M, S_, torch.zeros_like(S_)).sum(1)

        def forward(self, feats, keep, nq, B, base_z):
            """The score; base_z is never read (zg6's)."""
            if G6.KT_KEY not in feats:
                raise RuntimeError("zg6e: no KT input in the batch (P1c3's carve adds it)")
            if G6.LK_KEY not in feats:
                raise RuntimeError("zg6e: no pool edges in the batch (zlink's carve adds them)")
            H = self.encode(feats, keep, nq, B)
            xb = torch.cat([feats[G6.KT_KEY].to(H.dtype), self.chain(feats, keep, nq).to(H.dtype)], 1)
            u01 = torch.rand(B, FAMS, generator=self.g6_gen) if self.training else None
            return g6e_stack(self, H, xb, feats[G6.LK_KEY], nq, B, u01, self.bridge_message(feats, nq))

    def g_carve(base):
        """base (an R2 link carve class: zlink's edges and R2's bridges) with the link flags beside the bridges: batch() adds
        them under R2's key as "lk", from the batch's own edges (all of them) and bridges; every block, row, side key and gold
        is base's, unchanged."""

        class GCarve(base):

            def __init__(self, ds, carve, basis, device, cache_root=R2.LC.OUT, verify=True, score2=False):
                super().__init__(ds, carve, basis, device, cache_root, verify, score2)
                self.r2e_stats = None

            def batch(self, qs, blocks):
                feats, nq, base_z, gold = super().batch(qs, blocks)
                r2 = feats[R2.R2_KEY]
                flags = link_flags(feats[G6.LK_KEY], r2["bri"], nq, int(nq.numel()))
                r2["lk"] = flags
                if self.r2e_stats is not None:
                    stats_add(self.r2e_stats, flags, nq, int(r2["bri"].shape[0]))
                return feats, nq, base_z, gold

        GCarve.__name__ = GCarve.__qualname__ = f"E{base.__name__}"
        return GCarve

    GCarve = g_carve(R2.R2KTLinkCarveBase)
    R2.S.ARMS[ARM] = (ZG6E, GCarve)
    _G.update(R2=R2, G6=G6, np=np, torch=torch, link_flags=link_flags, stats_add=stats_add, g6e_layers=g6e_layers,
              g6e_stack=g6e_stack, ZG6E=ZG6E, GCarve=GCarve, g_carve=g_carve)
    return _G


@contextlib.contextmanager
def on_arm(G6, R2):
    """zg6's tables with zg6e in them for the block (the arm and R2EG's fit names), as zg6's train and read look arms and
    fits up; restored after, on a failure too. zg6 is inside zwq.on_wq (zwqk.g6_on_wq), so its splits are WQ's."""
    with WQ.patch(G6, ARMS={**G6.ARMS, ARM: R2.S.ARMS[ARM]}, FIT={**G6.FIT, ARM: dict(FIT)}):
        yield G6


STATS = {}


def stats_reader(orig):
    """lean_gpu.read_carve that also counts the link flags of the carve it reads (zr2ew's, for R2EG's carve)."""

    def read_carve(c, models, *a, **kw):
        if not hasattr(c, "r2e_stats"):
            raise SystemExit(f"zr2eg: {c.ds}={c.carve} is not read on R2EG's carve")
        c.r2e_stats = {k: 0 for k in LSTAT}
        try:
            out = orig(c, models, *a, **kw)
            st = dict(c.r2e_stats)
        finally:
            c.r2e_stats = None
        q, rows = len(c.n_np), int(c.n_np.sum())
        if st["rows"] == 0 or st["questions"] == 0:
            raise SystemExit(f"zr2eg: {c.ds}={c.carve}: the read scored no batch ({st['questions']} questions of {q})")
        STATS[f"{c.ds}={c.carve}"] = {**st, "passes": round(st["rows"] / rows, 4), "rows_in_carve": rows,
                                      "questions_in_carve": q}
        return out

    return read_carve


@contextlib.contextmanager
def stats_reads(LG):
    """lean_gpu's read_carve counts the link flags while zg6e is read."""
    with WQ.patch(LG, read_carve=stats_reader(LG.read_carve)):
        yield LG


link_stats = EW.link_stats


# ── guards ───────────────────────────────────────────────────────────────────


def arm_split(name):
    if name not in FITS:
        raise SystemExit(f"zr2eg: {name} is not one of R2EG's fits ({', '.join(FIT.values())})")
    return ARM, FITS[name]


def where_ok(env, what):
    if not K.ok_where(env):
        raise SystemExit(f"zr2eg: R2EG {what}s in a Kaggle kernel on a {K.GPU} from the image {K.IMAGE_TAG} (section "
                         f"3), not with {K.device_of(env)} ({env.get('where')})")


def cap_ok(rows_cap):
    if rows_cap is None or int(rows_cap) != READ_CAP:
        raise SystemExit(f"zr2eg: R2EG reads with --rows-cap {READ_CAP} (WQK's, section 3), not {rows_cap}")


def wqk_ready(base=None):
    """WQK's verdict record, committed before the stub: BASE_READY under WQK's declaration (section 3)."""
    p = Path(base or WQ.ROOT) / (K.VERDICT + ".json")
    if not p.is_file():
        raise SystemExit(f"zr2eg: {K.VERDICT}.json missing; R2EG trains after WQK's BASE_READY (section 3)")
    rec = WQ.read_json(p)
    if (rec.get("stage"), rec.get("declared_in"), rec.get("verdict")) != (K.STAGE, K.DECLARED, "BASE_READY"):
        raise SystemExit(f"zr2eg: {K.VERDICT}.json is {rec.get('verdict')}, not WQK's BASE_READY")
    return rec


def term_ok(summ, name):
    """A fit whose training had no batch with R2's term is not R2EG's (section 4): it stops unstamped."""
    if not summ.get("batches_with_term"):
        raise SystemExit(f"zr2eg: no training batch of {name} carried R2's term ({summ.get('batches')} batches); unstamped")


# ── train and read in the kernel (section 4) ─────────────────────────────────


def code_shas():
    return {m: WQ.sha_src(HERE / f"{m}.py") for m in CODE}


def head_of():
    return {"r2eg": R2EG, "kt_root": K.ROOTS["kt"], "roots": dict(K.ROOTS), "read_rows": READ_CAP}


def stamp_fit(name, env, R2, summ, base=None, records=None):
    """The fit's record after zg6's train: the wq block at WQK's roots and declaration (zwqk.wq_block), and R2EG's block (the
    device, WQK's gate, overlay, check and verdict, the KT, chain and link records, R2's constants, parameters and EM term,
    the message's declaration, the code)."""
    base = Path(base or WQ.ROOT)
    arm, split = arm_split(name)
    sj = WQ.fit_dir(name, base) / "screen.json"
    rec = WQ.read_json(sj)
    records = GW.fit_records(split) if records is None else records
    with WQ.patch(WQ, ROOT=base):
        wq = K.wq_block(arm, SEED, split)
    rec.update({"zwq_sha256": WQ.sha_src(WQ.__file__), "zwqk_sha256": WQ.sha_src(K.__file__),
                "zr2br_sha256": WQ.sha_src(HERE / "zr2br.py"), "zr2ew_sha256": WQ.sha_src(HERE / "zr2ew.py"),
                "zr2eg_sha256": WQ.sha_src(__file__), "wq": wq})
    rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": arm, "what": WHAT, "seed": SEED, "split": split,
                "against": BASE_FIT[split], "beside": BESIDE_FIT[split], "device": K.device_of(env), "env": env,
                "gate": WQ.sha_file(base / K.RECORDS["GATE"]), "overlay": WQ.sha_file(base / K.RECORDS["OVERLAY"]),
                "check": WQ.sha_file(base / K.RECORDS["CHECK"]),
                "wqk_verdict": WQ.sha_file(base / (K.VERDICT + ".json")), "cache_root": K.ROOTS["cache"], **records,
                "r2": {"declared_in": R2.DECLARED, "constants": R2.constants(), "params": list(NEW_PARAMS),
                       "settings": R2.SETTINGS, "term": summ},
                "head": head_of(), "code": code_shas()}
    WQ.write_json(sj, rec)
    return rec


def train(name, env=None):
    """R2EG's fit (section 4): zg6's train for the arm under the table swap with R2's EM term (LAM 1), at WQK's names and
    roots with WQK's mixed KT input, as zwqk's train runs zg6; then the fit's stamps."""
    arm, split = arm_split(name)
    env = K.device_env() if env is None else env
    where_ok(env, "train")
    wqk_ready()
    R = import_g()
    R2 = R["R2"]
    with K.on_wqk():
        WQ.gate_passed()
    args = ["train", "--name", name, "--arm", arm, "--device", "cuda", "--cache-root", K.ROOTS["cache"]]
    with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W, K.g6_on_wq(W) as G6:
        GW.same_modules(R2, W)
        if G6 is not R["G6"]:
            raise SystemExit("zr2eg: zg6 is not the module zr2br trains on")
        with on_arm(G6, R2), R2.r2_em():
            rc = G6.train(args, split)
    summ = R2.summary()
    if rc == 0:
        term_ok(summ, name)
        stamp_fit(name, env, R2, summ)
        ep = summ.get("epochs") or []
        log(f"train {name}: done, stamped (R2's term on {summ['questions_with_term']} of {summ['questions']} questions"
            + (f"; L_br {ep[0]['br_nll_mean']:.2f} -> {ep[-1]['br_nll_mean']:.2f})" if ep else ")"))
    return rc


def fit_stamp(name, fits_root=None):
    """An R2EG fit's screen.json and its R2EG block, checked against the fit's name."""
    arm, split = arm_split(name)
    sj = Path(fits_root or WQ.fit_dir(name).parent) / name / "screen.json"
    rec = WQ.read_json(sj) if sj.is_file() else {}
    k = rec.get(KEY) or {}
    if (rec.get("arm"), k.get("declared_in"), k.get("fit"), k.get("arm"), k.get("seed"), k.get("split")) != \
            (arm, DECLARED, name, arm, SEED, split):
        raise SystemExit(f"zr2eg: {name} is not R2EG's {arm} fit of {split} (no R2EG stamp)")
    return rec, k


def stamp_read(name, env, kt, rows_cap, stats, base=None, zl_records=None):
    """read.json's R2EG block: the device, the gate, the KT and link records, and the link statistics per carve."""
    base = Path(base or WQ.ROOT)
    arm, split = arm_split(name)
    rj = WQ.fit_dir(name, base) / "read.json"
    rec = WQ.read_json(rj)
    rec["zr2eg_sha256"] = WQ.sha_src(__file__)
    rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": arm, "seed": SEED, "split": split,
                "device": K.device_of(env), "env": env, "gate": WQ.sha_file(base / K.RECORDS["GATE"]),
                "kt_records": kt, "zlink_records": zl_records, "rows_cap": int(rows_cap),
                "link_stats": {c: link_stats(st) for c, st in stats.items()}}
    WQ.write_json(rj, rec)
    return rec


def read(name, env=None, rows_cap=READ_CAP):
    """The fit read by zg6's read for its arm with p@swa on the six s1eval carves, --rows-cap 65536, on the device it was
    fitted on (zwqk's read, under the table swap); the link flags counted per carve. read.json is stamped when the read
    exits 0."""
    arm, split = arm_split(name)
    cap_ok(rows_cap)
    env = K.device_env() if env is None else env
    where_ok(env, "read")
    rec, k = fit_stamp(name)
    if K.device_of(env) != k.get("device"):
        raise SystemExit(f"zr2eg: {name} was fitted on {k.get('device')}; this read runs on {K.device_of(env)}")
    got = (k.get("r2") or {}).get("constants")
    if got != R2_CONST:
        raise SystemExit(f"zr2eg: {name} was fitted with R2 constants {got}, not R2's declared {R2_CONST}")
    R = import_g()
    R2 = R["R2"]
    with K.on_wqk():
        WQ.cache_root_ok(K.ROOTS["cache"])
        WQ.gate_passed()
    argv = ["--name", name, "--device", "cuda", "--cache-root", K.ROOTS["cache"], "--rows-cap", str(READ_CAP)]
    STATS.clear()
    rc = 1
    try:
        with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W, K.g6_on_wq(W) as G6:
            GW.same_modules(R2, W)
            with on_arm(G6, R2), stats_reads(W.LG):
                rc = G6.read(argv)
            kt = W.Z3.kt_records(W.Z3.eval_carves())
        rj = WQ.fit_dir(name) / "read.json"
        if rc == 0 and rj.is_file():
            stamp_read(name, env, kt, rows_cap, dict(STATS), zl_records=EW.link_records())
    finally:
        STATS.clear()
    return rc


def stamped(p, name):
    try:
        k = WQ.read_json(p).get(KEY) or {}
    except (OSError, ValueError):
        return False
    return k.get("declared_in") == DECLARED and k.get("fit") == name


def fit(name, rows_cap=READ_CAP):
    """One fit in the kernel (zwqk's fit, for R2EG): a read R2EG stamped is kept; a fit trained and stamped but not read is
    only read; otherwise the fit's files are unlinked and it trains and reads. r2eg_timing.json holds the seconds and the
    peak."""
    arm, _split = arm_split(name)
    cap_ok(rows_cap)
    fd = WQ.fit_dir(name)
    if stamped(fd / "read.json", name):
        log(f"fit {name}: read before, kept")
        return 0
    env = K.device_env()
    where_ok(env, "train")
    wqk_ready()
    t0 = time.time()
    trained = stamped(fd / "screen.json", name) and (fd / "train.json").is_file() and (fd / "models.pt").is_file()
    if not trained:
        K.clear(fd, "*")
        K.clear(fd / "reads", "*")
        rc = train(name, env)
        if rc:
            return rc
    t1 = time.time()
    K.clear(fd / "reads", "*")
    K.clear(fd, "read.json")
    rc = read(name, env, rows_cap)
    WQ.write_json(fd / TIMING_FILE, {"fit": name, "trained_here": not trained, "train_seconds": round(t1 - t0, 1),
                                     "read_seconds": round(time.time() - t1, 1), "peak_gb": K.peak_gb(),
                                     "rows_cap": int(rows_cap), "exit": rc, "declared_in": DECLARED})
    return rc


# ── on the laptop, after the sessions: take, compare-all, verdict (sections 4 and 5) ──


def fit_device(name, fits_root=None):
    """An R2EG fit's device: its screen.json's R2EG block, and its read.json's on the same device with --rows-cap 65536."""
    _rec, k = fit_stamp(name, fits_root)
    rj = Path(fits_root or WQ.fit_dir(name).parent) / name / "read.json"
    r = WQ.read_json(rj) if rj.is_file() else {}
    rk = r.get(KEY) or {}
    if (rk.get("declared_in"), rk.get("fit"), rk.get("device"), rk.get("rows_cap")) != \
            (DECLARED, name, k.get("device"), READ_CAP):
        raise SystemExit(f"zr2eg: {name} has no R2EG read on the device it was fitted on")
    return k["device"]


def any_device(name, fits_root):
    """A fit's device by whose it is: R2EG's through fit_device, WQK's through zwqk's."""
    return (fit_device if name in FITS else K.fit_device)(name, fits_root)


def pair_device(names, base):
    """The one device the fits of names were fitted and read on."""
    fr = K.fits_dir(base)
    devs = {n: any_device(n, fr) for n in names}
    if len({json.dumps(d, sort_keys=True) for d in devs.values()}) != 1:
        raise SystemExit(f"zr2eg: {names[0]} and its bases ran on different devices or images: {devs}")
    return devs[names[0]]


def split_device(sp, base):
    """The one device R2EG's fit of sp and WQK's seed-0 zg6 and zspk fits of sp ran on."""
    return pair_device([FIT[sp], BASE_FIT[sp], BESIDE_FIT[sp]], base)


def take(src, base=None):
    """A kernel's output (src): each R2EG fit copied to outputs/screen/fits/<name>, its records and reads, never its
    weights; the session's gate.json and overlay.json to outputs/r2eg/sessions/<the gate's sha256[:12]>. A fit without
    R2EG's stamp and read on one device is left (named); a folder not R2EG's, or a file already there with other bytes,
    stops the take. Returns (copied, left)."""
    base, src = Path(base or WQ.ROOT), Path(src)
    fits = src / "fits"
    g, o = src / "wqk" / "gate.json", src / "wqk" / "overlay.json"
    if not (g.is_file() and o.is_file()):
        raise SystemExit(f"zr2eg take: {src.name} holds no wqk/gate.json and wqk/overlay.json (the session's records)")
    names = sorted(p.name for p in fits.iterdir() if p.is_dir()) if fits.is_dir() else []
    bad = [n for n in names if n not in FITS]
    if bad:
        raise SystemExit(f"zr2eg take: {bad} are not R2EG's fits")
    done, left = [], []
    for n in names:
        try:
            fit_device(n, fits)
            done.append(n)
        except SystemExit as e:
            left.append(f"{n}: {e}")
    plan = [(f, K.fits_dir(base) / n / f.relative_to(fits / n)) for n in done
            for f in sorted(p for p in (fits / n).rglob("*") if p.is_file())
            if f.name not in K.NEVER and f.suffix not in (".tmp", ".part")]
    sess = base / SESSIONS / WQ.sha_file(g)[:12]
    plan += [(g, sess / "gate.json"), (o, sess / "overlay.json")]
    for f, t in plan:
        if t.exists() and WQ.sha_file(t) != WQ.sha_file(f):
            raise SystemExit(f"zr2eg take: {WQ.rel(t, base)} exists with other bytes")
    took = []
    for f, t in plan:
        if not t.exists():
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, t)
            took.append(WQ.rel(t, base))
    log(f"take: {len(done)} fits, {len(took)} files copied; the gate {sess.name}" + (f"; left: {left}" if left else ""))
    return took, left


def compare_all(base=None):
    """lean_screen's compare of each R2EG fit with WQK's seed-0 zg6 fit of its split (first: it decides) and with WQK's
    seed-0 zspk fit of it (beside), run from the repository's root by zwq's run_compare, every fit on one device; each
    stamped with R2EG's record and that device. A compare R2EG stamped from the same fits is kept."""
    base = Path(base or WQ.ROOT)
    import_g()
    done, ran = [], []
    for cname, (new, bases) in COMPARES.items():
        js = K.screen_dir(base) / f"{cname}.json"
        if js.is_file():
            k = WQ.read_json(js).get(KEY) or {}
            if (k.get("declared_in"), k.get("new"), k.get("bases")) == (DECLARED, new, list(bases)):
                done.append(cname)
                continue
        dev = pair_device([new] + list(bases), base)
        with K.on_wqk():
            for f in bases:
                why = WQ.wq_why(f, base)
                if why:
                    raise SystemExit(f"zr2eg: {why}")
            rc = WQ.run_compare(f"outputs/screen/fits/{new}", ",".join(f"outputs/screen/fits/{b}" for b in bases),
                                f"outputs/screen/{cname}")
        if rc or not js.is_file():
            raise SystemExit(f"zr2eg: the compare {cname} exited {rc}")
        rec = WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases), "device": dev,
                    "zr2eg_sha256": WQ.sha_src(__file__)}
        WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def cmp_rows(js, cname, dev=None):
    """An R2EG compare's rows {base fit: {dataset: row}}, after its checks; SystemExit naming what is wrong."""
    p = Path(js)
    if not p.is_file():
        raise SystemExit(f"outputs/screen/{cname}.json missing")
    rec = WQ.read_json(p)
    new, bases = COMPARES[cname]
    k = rec.get(KEY) or {}
    if (k.get("declared_in"), k.get("compare"), k.get("new"), k.get("bases")) != (DECLARED, cname, new, list(bases)):
        raise SystemExit(f"outputs/screen/{cname}.json is not R2EG's compare of {new} with {bases}")
    if dev is not None and k.get("device") != dev:
        raise SystemExit(f"outputs/screen/{cname}.json was made on {k.get('device')}, not on the fits' device {dev}")
    got = [WQ.parts_of(b)[-1:] for b in rec.get("bases") or []]
    if WQ.parts_of(rec.get("new", ""))[-1:] != (new,) or got != [(b,) for b in bases]:
        raise SystemExit(f"outputs/screen/{cname}.json compares {rec.get('new')} with {rec.get('bases')}")
    split = FITS[new]
    if sorted(rec.get("trained_on") or []) != sorted(WQ.SPLITS[split]):
        raise SystemExit(f"outputs/screen/{cname}.json trained on {rec.get('trained_on')}, not {split}'s")
    if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
        raise SystemExit(f"outputs/screen/{cname}.json reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on "
                         f"s1eval")
    out = {}
    for b, bp in zip(bases, rec["bases"]):
        rows = {r["dataset"]: r for r in rec["by_base"][bp]["rows"]}
        if sorted(rows) != sorted(WQ.EVAL_ORDER):
            raise SystemExit(f"outputs/screen/{cname}.json reads {sorted(rows)} against {b}, not the six datasets")
        out[b] = rows
    return rec, out


def filed_gates(base):
    """zwqk.filed_gates on R2EG's sessions: ({gate sha256: its record}, problems), each a WQK PASS gate on the reproduced
    check and its session's overlay."""
    with WQ.patch(K, SESSIONS=SESSIONS):
        return K.filed_gates(base)


def fit_problems(base, gates):
    """Why an R2EG fit is not section 4's: its train exit, its six reads, the gates its train, wq block and read name, its
    overlay, R2's constants, parameters and EM term, its message's declaration, and the link statistics of its read."""
    out = []
    fr = K.fits_dir(base)
    for n in FIT.values():
        sj, rj = fr / n / "screen.json", fr / n / "read.json"
        if not (sj.is_file() and rj.is_file()):
            continue                                  # fit_device names it
        s, r = WQ.read_json(sj), WQ.read_json(rj)
        k, rk = s.get(KEY) or {}, r.get(KEY) or {}
        if s.get("rc") != 0:
            out.append(f"{n}: its train did not exit 0")
        if set(r.get("carves") or {}) != set(K.SIX):
            out.append(f"{n}: not read on the six s1eval carves")
        for what, g in (("train", k.get("gate")), ("wq block", (s.get("wq") or {}).get("gate")), ("read", rk.get("gate"))):
            if g not in gates:
                out.append(f"{n}: its {what} names the gate {str(g)[:12]}, not a filed R2EG PASS gate")
        if k.get("gate") in gates and (gates[k["gate"]].get("records") or {}).get("overlay") != k.get("overlay"):
            out.append(f"{n}: its overlay is not its gate's")
        r2 = k.get("r2") or {}
        term = r2.get("term") or {}
        if (r2.get("declared_in"), r2.get("constants"), r2.get("params"), term.get("lam")) != \
                (R2_DECLARED, R2_CONST, list(NEW_PARAMS), 1.0) or not term.get("questions_with_term"):
            out.append(f"{n}: R2's constants, parameters or EM term are not as declared")
        if (k.get("head") or {}).get("r2eg") != R2EG:
            out.append(f"{n}: its message is not R2eg's as declared")
        ls = rk.get("link_stats") or {}
        if set(ls) != set(K.SIX) or not all(all(isinstance(v.get(f), int) and v[f] > 0 for f in ("rows", "questions"))
                                            and all(f in v for f in LSTAT) for v in ls.values()):
            out.append(f"{n}: its read holds no link statistics on the six carves")
    return out


def restamp(pair, rcl, c):
    """R2EG's fields on the pair and re-call screen_pair and screen_recall wrote; a header on their markdown."""
    for p in (pair, rcl):
        js = Path(str(p) + ".json")
        rec = WQ.read_json(js)
        rec.update({"stage": STAGE, "declared_in": DECLARED, "arm": ARM, "against": AGAINST[c],
                    "null_kind": NULL_KIND[c], "deciding": DECIDING[c], "zr2eg_sha256": WQ.sha_src(__file__)})
        WQ.write_json(js, rec)
        md = Path(str(p) + ".md")
        if md.is_file():
            head_ = (f"R2EG ({DECLARED}): {ARM} against {AGAINST[c]} under WQ on the T4"
                     + (", deciding." if DECIDING[c] else ", beside, deciding nothing.") + "\n\n")
            md.write_text(head_ + md.read_text(encoding="utf-8"), encoding="utf-8")


def recall(base, dev, c):
    """screen_pair over R2EG's two compares (column CALLS[c] of each: WQK's seed-0 zg6, or WQK's seed-0 zspk), then
    screen_recall under WQK's zspk seed null. For zg6e against zg6 the null's seed-0 R@5 is zg6's read (the compare's zg6
    column; G6's form, no zg6 null is filed); the floors stay zspk's."""
    import screen_pair as SP
    import screen_recall as SR
    base = Path(base)
    scr = K.screen_dir(base)
    col = CALLS[c]

    def load(fn):
        cn = next(x for x in COMPARES if Path(fn).stem == x)
        _rec, rows = cmp_rows(scr / f"{cn}.json", cn, dev)
        return FITS[COMPARES[cn][0]], [rows[COMPARES[cn][1][col]][ds] for ds in WQ.EVAL_ORDER]

    pair, rcl = K.at(base, PAIR[c]), K.at(base, RECALL[c])
    pair.parent.mkdir(parents=True, exist_ok=True)
    with WQ.patch(SP, load=load):
        SP.pair([K.at(base, f"outputs/screen/{CMP[sp]}.json") for sp in SPLITS], pair)
    null, sources = K.wqk_null(BESIDE, base, dev, SR.floor_of)
    if c == "zg6e":
        for sp in SPLITS:
            _r, kb = cmp_rows(scr / f"{CMP[sp]}.json", CMP[sp], dev)
            null[sp] = {ds: {**v, "seed0": kb[BASE_FIT[sp]][ds]["base"][0]} for ds, v in null[sp].items()}
            sources[f"WQK zg6 {sp}"] = f"outputs/screen/{CMP[sp]}.json"
    with WQ.patch(SR, load_null=lambda _files: (null, sources)):
        rec = SR.pair([], Path(str(pair) + ".json"), rcl)
    restamp(pair, rcl, c)
    return rec


def tag(r):
    return f"{r['split']} {r['dataset']}"


def f4(x, signed=False):
    return "n/a" if x is None else K.fmt(x, signed)


def verdict(base=None, out=None):
    """R2EG's verdict by the rule from zg6e's re-call against WQK's seed-0 zg6 under WQK's zspk null with zg6's seed-0 R@5
    (sections 4 and 5). Beside, deciding nothing: zg6e against zspk (G6's call), the link statistics per read, R@5 per
    read, R2's term per fit and seconds and memory; outputs/r2eg/screen-verdict."""
    import screen_recall as SR
    base = Path(base or WQ.ROOT)
    od = Path(out) if out else base / VERDICT
    problems = []
    try:
        wqk_ready(base)
    except SystemExit as e:
        problems.append(str(e))
    gates, gp = filed_gates(base)
    problems += gp
    devs = {}
    for sp in SPLITS:
        try:
            devs[sp] = split_device(sp, base)
        except SystemExit as e:
            problems.append(str(e))
    problems += fit_problems(base, gates)
    keys = {json.dumps(d, sort_keys=True) for d in devs.values()}
    if len(keys) > 1:
        problems.append(f"R2EG's fits ran on {len(keys)} devices: {sorted(keys)}")
    dev = json.loads(next(iter(keys))) if len(keys) == 1 else None
    rows = {}
    for cn in COMPARES:
        try:
            rows[cn] = cmp_rows(K.screen_dir(base) / f"{cn}.json", cn, dev)[1]
        except SystemExit as e:
            problems.append(str(e))
    problems += K.floors_problems(base, SR.floor_of)
    rc = {}
    if not problems:
        try:
            for c in CALLS:
                rc[c] = recall(base, dev, c)
        except (SystemExit, KeyError) as e:
            problems.append(WQ.clean(e, base))
    whole = sorted((sp, ds) for sp in SPLITS for ds in WQ.EVAL_ORDER)
    for c, r in rc.items():
        if sorted((x["split"], x["dataset"]) for x in r["rows"]) != whole:
            problems.append(f"the re-call of {c} against {AGAINST[c]} holds {len(r['rows'])} reads, not the twelve of "
                            f"{', '.join(SPLITS)}")
    rrows = rc["zg6e"]["rows"] if "zg6e" in rc and not problems else []
    gains = [tag(r) for r in rrows if r["call"] == "GAIN"]
    losses = [tag(r) for r in rrows if r["call"] == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = NEXT[v]
    wv = base / (K.VERDICT + ".json")
    rec = {"stage": STAGE, "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "base_arm": BASE,
           "fits": {ARM: dict(FIT)}, "against": dict(BASE_FIT), "beside_fits": dict(BESIDE_FIT),
           "floors": FLOORS + ".json", "null_kind": NULL_KIND["zg6e"], "device": dev, "gates": sorted(gates),
           "next_form": nxt, "gains": gains, "losses": losses, "problems": problems,
           "calls": {tag(r): r["call"] for r in rrows}, "delta_r5": {tag(r): r["delta"] for r in rrows},
           "params_added": NEW_N, "naming": NAMING, "wqk_verdict": WQ.sha_file(wv) if wv.is_file() else None,
           "zwqk_sha256": WQ.sha_src(K.__file__), "zr2eg_sha256": WQ.sha_src(__file__), "utc": WQ.utc()}
    md = [f"# R2EG, R2e (the GNN half) under WQ on Kaggle's T4: zg6e **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zg6e** ({WHAT}) against WQK's seed-0 zg6 on the T4, re-called under WQK's zspk seed null with zg6's "
          f"seed-0 R@5: {len(gains)} GAIN ({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ".",
          f"- **Next form** (fixed in section 7 before any number): {nxt}.", f"- Device: {json.dumps(dev)}.",
          f"- Gates: {', '.join(g[:12] for g in sorted(gates)) or 'none'}.", f"- Naming: {NAMING}.",
          f"- Numbers added to zg6: {NEW_N} ({BRIDGE_N} in R2's bridge logits, {DIR_N} in the three directions)."]
    beside = {}
    if v != "INCOMPLETE":
        fr = K.fits_dir(base)
        side = {x["split"] + " " + x["dataset"]: x["call"] for x in rc["zg6e-vs-zspk"]["rows"]}
        by = {"gains": [k_ for k_, x in side.items() if x == "GAIN"], "losses": [k_ for k_, x in side.items() if x == "LOSS"]}
        lstat = {sp: ((WQ.read_json(fr / FIT[sp] / "read.json").get(KEY) or {}).get("link_stats") or {}) for sp in SPLITS}
        r5 = {}
        for r in rrows:
            sp, ds = r["split"], r["dataset"]
            r5[tag(r)] = {"read": r["read"], "rrf": r["rrf"], "zspk": rows[CMP[sp]][BESIDE_FIT[sp]][ds]["base"][0],
                          "zg6": r["base"], "zg6e": r["new"], "floor": r["floor"], "call": r["call"],
                          "zg6e_vs_zspk": side[tag(r)], "links": lstat[sp].get(f"{ds}=s1eval")}
        term = {}
        for sp in SPLITS:
            t = ((WQ.read_json(fr / FIT[sp] / "screen.json").get(KEY) or {}).get("r2") or {}).get("term") or {}
            ep = t.get("epochs") or []
            term[FIT[sp]] = {"questions": t.get("questions"), "questions_with_term": t.get("questions_with_term"),
                             "share_with_term": t.get("share_with_term"),
                             "br_nll_first": ep[0].get("br_nll_mean") if ep else None,
                             "br_nll_last": ep[-1].get("br_nll_mean") if ep else None}
        tm = {sp: {"zg6e": K.timing(FIT[sp], fr, TIMING_FILE), "zg6": K.timing(BASE_FIT[sp], fr, "wqk_timing.json"),
                   "zspk": K.timing(BESIDE_FIT[sp], fr, "wqk_timing.json")} for sp in SPLITS}
        rp = Path(str(K.at(base, RECALL["zg6e"])) + ".json")
        beside = {"calls": {"zg6e-vs-zspk": by}, "r5": r5, "terms": term, "timing": tm,
                  "pair": PAIR["zg6e"] + ".json", "recall": RECALL["zg6e"] + ".json", "recall_sha256": WQ.sha_file(rp),
                  "pairs": {c: PAIR[c] + ".json" for c in CALLS}, "recalls": {c: RECALL[c] + ".json" for c in CALLS}}
        md += ["", "Beside, deciding nothing.", "",
               f"- zg6e against WQK's seed-0 zspk (G6's call, under WQK's zspk null): {len(by['gains'])} GAIN "
               f"({', '.join(by['gains']) or 'none'}), {len(by['losses'])} LOSS ({', '.join(by['losses']) or 'none'}).",
               "", "R@5 per read on the T4, and zg6e's link statistics on the read (share of rows with a linked bridge, "
                   "linked bridges per row, share of questions with a linked row):", "",
               "| read | rrf | zspk | zg6 | zg6e | floor | call | vs zspk | rows linked | flags per row "
               "| questions linked |", "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|"]
        for k_, t in r5.items():
            ls = t["links"] or {}
            md.append(f"| {k_} ({t['read']}) | {f4(t['rrf'])} | {f4(t['zspk'])} | {f4(t['zg6'])} | {f4(t['zg6e'])} "
                      f"| {f4(t['floor'])} | {t['call']} | {t['zg6e_vs_zspk']} | {f4(ls.get('share_rows_linked'))} "
                      f"| {f4(ls.get('flags_per_row'))} | {f4(ls.get('share_questions_linked'))} |")
        md += ["", "R2's EM term on each fit (the share of training questions it reached, L_br at the first and the last "
                   "epoch):", "", "| fit | questions | with the term | share | L_br first | L_br last |",
               "|---|---:|---:|---:|---:|---:|"]
        for n, t in term.items():
            md.append(f"| {n} | {t['questions']} | {t['questions_with_term']} | {f4(t['share_with_term'])} "
                      f"| {f4(t['br_nll_first'])} | {f4(t['br_nll_last'])} |")

        def s0(x):
            return f"{x:.0f}" if isinstance(x, (int, float)) else "n/a"
        md += ["", "Seconds and memory: zg6e, and in brackets WQK's zg6 and zspk of the same split, all on the T4:", "",
               "| split | train s | read s | peak GPU GB, train / read | peak RAM GB |", "|---|---:|---:|---|---:|"]
        for sp in SPLITS:
            a_, b_, c_ = tm[sp]["zg6e"], tm[sp]["zg6"], tm[sp]["zspk"]
            md.append(f"| {sp} | {s0(a_['train_seconds'])} ({s0(b_['train_seconds'])}, {s0(c_['train_seconds'])}) | "
                      f"{s0(a_['read_seconds'])} ({s0(b_['read_seconds'])}, {s0(c_['read_seconds'])}) | "
                      f"{f4(a_['train_peak_gpu_gb'])} / {f4(a_['read_peak_gpu_gb'])} ({f4(b_['train_peak_gpu_gb'])} / "
                      f"{f4(b_['read_peak_gpu_gb'])}, {f4(c_['train_peak_gpu_gb'])} / {f4(c_['read_peak_gpu_gb'])}) | "
                      f"{f4(a_['peak_gb'])} ({f4(b_['peak_gb'])}, {f4(c_['peak_gb'])}) |")
    rec["beside"] = beside
    WQ.write(od, rec, md)
    log(f"verdict: {v} ({len(gains)} GAIN, {len(losses)} LOSS{'; ' + '; '.join(problems) if problems else ''}); "
        f"next: {nxt}")
    return rec


def real_state():
    """What R2EG has under the real folders: its fits, its compares and everything under outputs/r2eg."""
    w = WQ.ROOT / NEW
    return (sorted(p.name for p in K.fits_dir(WQ.ROOT).glob("scr-r2eg-*")),
            sorted(p.name for p in K.screen_dir(WQ.ROOT).glob("scr-r2eg-*")),
            sorted((p.relative_to(w).as_posix(), p.stat().st_mtime_ns) for p in w.rglob("*")) if w.is_dir() else [])


# ── selftest ─────────────────────────────────────────────────────────────────

FAKE_COUNTS = dict(EW.FAKE_COUNTS)


def fake_fit(fits_root, name, gate_sha, overlay_sha, dev=None, read_=True, rows_cap=READ_CAP, r2=None, r2eg="declared",
             stats=True):
    """An R2EG fit as the kernel leaves it: screen.json (WQK's wq block, R2EG's block) and read.json with their stamps,
    train.json, r2eg_timing.json, a read and models.pt."""
    sp = FITS[name]
    d = Path(fits_root) / name
    (d / "reads").mkdir(parents=True, exist_ok=True)
    dev = dev or K.device_of(K.TOY_ENV)
    r2 = r2 or {"declared_in": R2_DECLARED, "constants": dict(R2_CONST), "params": list(NEW_PARAMS),
                "term": {"lam": 1.0, "questions": 100, "questions_with_term": 90, "share_with_term": 0.9,
                         "epochs": [{"br_nll_mean": 4.6}, {"br_nll_mean": 3.9}]}}
    k = {"declared_in": DECLARED, "fit": name, "arm": ARM, "seed": SEED, "split": sp, "device": dev, "gate": gate_sha,
         "overlay": overlay_sha, "r2": r2, "head": {"r2eg": R2EG if r2eg == "declared" else r2eg}}
    WQ.write_json(d / "screen.json", {"arm": ARM, "seed": SEED, "split": sp, "rc": 0, KEY: k,
                                      "wq": {"declared_in": K.DECLARED, "split": sp, "seed": SEED, "arm": ARM,
                                             "gate": gate_sha}})
    WQ.write_json(d / "train.json", {"seconds": 150.0, "peak_gpu_gb": 3.5, "data_gb": 2.0})
    WQ.write_json(d / TIMING_FILE, {"peak_gb": 9.0})
    (d / "models.pt").write_bytes(b"weights")
    (d / "reads" / "metaqa__s1eval.npz").write_bytes(b"read")
    if read_:
        WQ.write_json(d / "read.json", {
            "seconds": 12.0, "peak_gpu_gb": 4.5, "carves": {c: {} for c in K.SIX},
            KEY: {"declared_in": DECLARED, "fit": name, "arm": ARM, "seed": SEED, "split": sp, "device": dev,
                  "gate": gate_sha, "rows_cap": rows_cap,
                  "link_stats": {c: link_stats(FAKE_COUNTS) for c in K.SIX} if stats else None}})
    return d


def selftest():
    import tempfile
    import types
    t0 = time.time()
    before = real_state()
    stops, TOY = K.stops, K.TOY_ENV
    me = sys.modules[__name__]
    R = import_g()
    R2, G6, np, torch = R["R2"], R["G6"], R["np"], R["torch"]
    ZG6E, GCarve, g6e_stack, link_flags = R["ZG6E"], R["GCarve"], R["g6e_stack"], R["link_flags"]
    nn = torch.nn
    Fn = torch.nn.functional
    # 1. the names: two fits, two compares each with WQK's seed-0 zg6 and zspk, the modules the stamps hash, R2's text for
    #    R2e, the parameter counts
    assert FIT == {"WQ-musique": "scr-r2eg-zg6", "WQ-hotpotqa": "scr-r2eg-zg6-hp"} and len(FITS) == 2
    assert BASE_FIT == {"WQ-musique": "scr-wqk-zg6-s0", "WQ-hotpotqa": "scr-wqk-zg6-s0-hp"}
    assert BESIDE_FIT == {"WQ-musique": "scr-wqk-zspk-s0", "WQ-hotpotqa": "scr-wqk-zspk-s0-hp"}
    assert COMPARES == {"scr-r2eg-zg6-vs-zg6": ("scr-r2eg-zg6", ["scr-wqk-zg6-s0", "scr-wqk-zspk-s0"]),
                        "scr-r2eg-zg6-vs-zg6-hp": ("scr-r2eg-zg6-hp", ["scr-wqk-zg6-s0-hp", "scr-wqk-zspk-s0-hp"])}
    assert all((HERE / f"{m}.py").is_file() for m in CODE) and len(set(CODE)) == len(CODE)
    assert {"zlink", "zr2br", "zr2gw", "zr2ew", "zg6", "zr2eg"} <= set(CODE)
    assert set(NEXT) == {"ADOPT_FOR_FULL_RUN", "NOT_ADOPTED", "INCOMPLETE"} and "zg6e" in WHAT + ARM
    assert R2E_TEXT == R2.NEXT["NO_GAIN"] and "inside zg6's messages" in R2E_TEXT and R2.constants() == R2_CONST
    assert R2.n_params() == GW.R2_N == 59537 and R2.K == NB == 5 and (R2.Q_DIM, R2.P_DIM, R2.R_DIM) == (1536, 257, 32)
    assert BRIDGE_N == R2.Q_DIM * R2.R_DIM + R2.P_DIM * R2.R_DIM + 2 * R2.R_DIM ** 2 + R2.R_DIM == 59456
    assert DIR_N == G6.LAYERS * G6.D == 192 and NEW_N == 59648 == R2EG["params_added"]
    assert R2.S.ARMS[ARM] == (ZG6E, GCarve) and ZG6E.__mro__[1:3] == (R2.R2Mix, G6.ZG6) and not ZG6E.TWIN
    assert GCarve.__mro__[1] is R2.R2KTLinkCarveBase and GCarve.__name__ == "E" + R2.R2KTLinkCarveBase.__name__
    assert R2.S.ARMS["zg6m"] == R2.ARMS["zg6m"] and G6.ARMS["zg6"] == (G6.ZG6, G6.Z3.KTLinkCarveBase) and ARM not in G6.ARMS
    # 2. the model and the carve on a toy built by zlink's own build: the state is zg6's plus R2's bridge logits and the
    #    directions (no head), exactly 59,648 numbers; at the start zg6e is zg6 bit for bit; with every flag 0 it is zg6
    #    for any weights; the message is the by-hand sum over linked bridges; the layers add mb * e_l before layer_norm;
    #    the flags are the by-hand loop over the edge files for any question order and the same for any weights
    ZL, RM, ZM, Z3, LG = R2.ZL, R2.RM, R2.ZM, R2.Z3, R2.LG
    LG.bind_device_ops()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rng = np.random.default_rng(64)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look", rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")

        class ToyIds(ZM.ToyZ):
            def __init__(self, *a, **k_):
                super().__init__(*a, **k_)
                self.ids = [f"q{i}" for i in range(self.rows)]
                self.q_emb = self.q_emb / self.q_emb.norm(dim=1, keepdim=True)
                self.P = self.P * (0.9 / self.P.norm(dim=1, keepdim=True))

        ch_cls = RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel")
        lk_cls = ZL.link_carve(ch_cls, tmp / "lk")
        c0 = ch_cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
        r2_cls = R2.r2_carve(Z3.kt_carve(lk_cls, tmp / "kt"))
        ec_cls = R["g_carve"](r2_cls)                   # the arm's carve class, on the toy's own link carve
        ec = ec_cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        cm = r2_cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert ec_cls.__mro__[1] is r2_cls and ec_cls.__name__ == f"E{r2_cls.__name__}" and ec.r2e_stats is None
        assert np.array_equal(ec.r2_br, cm.r2_br) and ec.nbytes() == cm.nbytes()
        blocks = ["rank", "SEMB"]
        rows_ = ec.rows
        qs = np.r_[np.arange(rows_)[::-1][:7], np.arange(rows_)[:5]].astype(np.int64)
        B_ = qs.size
        fe, nqe, bze, gold_e = ec.batch(qs, blocks)
        keep = torch.ones(B_, len(blocks))
        r2x = fe[R2.R2_KEY]
        N_ = int(nqe.numel())
        assert r2x["lk"].dtype == torch.bool and r2x["lk"].shape == (N_, NB) and G6.LK_KEY in fe and G6.KT_KEY in fe
        # the flags, from the link files alone, by the by-hand loop, for any question order
        parts = sorted((tmp / "lk" / "metaqa" / "toy").glob("part_*"))
        raw = {k_: np.concatenate([np.load(p / f"{k_}.npy") for p in parts]) for k_ in ("q_ne", "e_u", "e_v")}
        whole = EW.flags_np(ec.links["q_ne"], ec.links["e_u"], ec.links["e_v"], ec.n_np, ec.r2_br)
        offs = np.r_[0, np.cumsum(ec.n_np)]
        ref_set = EW.flags_by_hand(raw["q_ne"], raw["e_u"], raw["e_v"], ec.n_np, ec.r2_br)
        assert {(r, k_) for r, k_ in zip(*np.nonzero(whole))} == ref_set and whole.any() and not whole.all(), whole.sum()
        for qq in (qs, np.arange(rows_), np.array([3]), np.array([rows_ - 1, 0, 2])):
            qq = qq.astype(np.int64)
            f_e, nq_e, bz_e, gd_e = ec.batch(qq, blocks)
            f_c, nq_c, bz_c, gd_c = cm.batch(qq, blocks)
            assert torch.equal(nq_e, nq_c) and torch.equal(gd_e, gd_c) and torch.equal(bz_e, bz_c)
            assert all(torch.equal(f_e[R2.R2_KEY][k_], f_c[R2.R2_KEY][k_]) for k_ in ("bri", "eq", "p"))
            assert set(f_e[R2.R2_KEY]) == set(f_c[R2.R2_KEY]) | {"lk"} and not f_e[R2.R2_KEY]["lk"].requires_grad
            ref = np.concatenate([whole[offs[q]:offs[q + 1]] for q in qq])
            assert np.array_equal(f_e[R2.R2_KEY]["lk"].numpy(), ref), qq
        # the state: zg6's plus R2's five bridge tensors and the directions, 59,648 numbers, no head
        torch.manual_seed(0)
        m_g = G6.ZG6(blocks, ec.widths, 16, dropout=0.1, seed=0)
        after_g = torch.rand(4)
        torch.manual_seed(0)
        m_e = ZG6E(blocks, ec.widths, 16, dropout=0.1, seed=0)
        after_e = torch.rand(4)
        assert torch.equal(after_g, after_e)                                  # the global generator moves as zg6's
        sg, se = m_g.state_dict(), m_e.state_dict()
        assert sorted(set(se) - set(sg)) == sorted(NEW_PARAMS) and set(sg) <= set(se)
        assert sum(int(se[k_].numel()) for k_ in NEW_PARAMS) == NEW_N and sum(int(se[k_].numel()) for k_ in BRIDGE_PARAMS) == BRIDGE_N
        assert not any(hasattr(m_e, n) for n in HEAD_PARAMS) and tuple(m_e.g6e_dir.shape) == (3, 64)
        assert not m_e.g6e_dir.any() and all(torch.equal(sg[k_], se[k_]) for k_ in sg)
        assert nn.Module.__call__ and not any(k_.startswith("r2_w") and k_ not in BRIDGE_PARAMS for k_ in se)
        # at the start zg6e is zg6, bit for bit, in evaluation and in training under the same seed
        for mode in (False, True):
            torch.manual_seed(0)
            a_ = G6.ZG6(blocks, ec.widths, 16, dropout=0.1, seed=0)
            torch.manual_seed(0)
            b_ = ZG6E(blocks, ec.widths, 16, dropout=0.1, seed=0)
            a_.train(mode), b_.train(mode)
            with torch.no_grad():
                torch.manual_seed(5)
                ya = a_(fe, keep, nqe, B_, bze)
                torch.manual_seed(5)
                yb = b_(fe, keep, nqe, B_, bze)
            assert torch.equal(ya, yb) and bool(torch.isfinite(yb).all()), mode
        # random weights: e_l = 0 is zg6 for any logit weights; every flag 0 is zg6 for any e_l; both on is not
        gen = torch.Generator().manual_seed(65)
        scale = {"r2_U": 1.0, "r2_V": 1.0, "r2_W1": 0.3 / R2.R_DIM ** 0.5, "r2_W2": 0.3 / R2.R_DIM ** 0.5, "r2_o": 0.3}
        m_e.load_state_dict({**sg, **{k_: se[k_] for k_ in NEW_PARAMS}})
        m_e.eval(), m_g.eval()
        with torch.no_grad():
            for k_ in BRIDGE_PARAMS:
                getattr(m_e, k_).copy_(torch.randn(getattr(m_e, k_).shape, generator=gen) * scale[k_])
            y_g = m_g(fe, keep, nqe, B_, bze)
            assert torch.equal(m_e(fe, keep, nqe, B_, bze), y_g)                  # e_l = 0
            m_e.g6e_dir.copy_(torch.randn(3, 64, generator=gen) * 0.3)
            y_e = m_e(fe, keep, nqe, B_, bze)
            assert not torch.equal(y_e, y_g)
            off_ = {**fe, R2.R2_KEY: {**r2x, "lk": torch.zeros_like(r2x["lk"])}}
            assert torch.equal(m_e(off_, keep, nqe, B_, bze), y_g)                # every flag 0
            # the message: the by-hand sum over the slots that are present, not the row, and linked (float64)
            S_, M = R2.r2_logits(m_e, r2x["eq"], r2x["p"], r2x["bri"], nqe)
            Mk = (M & r2x["lk"]).numpy()
            ref_mb = np.array([sum(float(S_[v, k_]) for k_ in range(NB) if Mk[v, k_]) for v in range(N_)])
            mb = m_e.bridge_message(fe, nqe)
            assert mb.shape == (N_,) and np.abs(mb.double().numpy() - ref_mb).max() < 1e-4 and Mk.any()
            assert (mb[torch.from_numpy(~Mk.any(1))] == 0).all() and m_e._r2_last is None
            # the layers: layer 0's pre-activation is zg6's plus mb * e_0, bit-close; the stack with mb = 0 is zg6's
            orig_ln = Fn.layer_norm
            cap = {"g6": [], "g6e": []}

            def tap(key):
                def ln(x, shape, *a, **k_):
                    cap[key].append(x.detach().clone())
                    return orig_ln(x, shape, *a, **k_)
                return ln
            Hh = m_e.encode(fe, keep, nqe, B_)
            xb = torch.cat([fe[G6.KT_KEY], m_e.chain(fe, keep, nqe)], 1)
            with WQ.patch(Fn, layer_norm=tap("g6")):
                ref_s = G6.g6_stack(m_e, Hh, xb, fe[G6.LK_KEY], nqe, B_, None)
            with WQ.patch(Fn, layer_norm=tap("g6e")):
                new_s = g6e_stack(m_e, Hh, xb, fe[G6.LK_KEY], nqe, B_, None, mb)
            assert len(cap["g6"]) == len(cap["g6e"]) == 3
            assert torch.allclose(cap["g6e"][0] - cap["g6"][0], mb.unsqueeze(1) * m_e.g6e_dir[0], atol=1e-5)
            assert not torch.equal(new_s, ref_s) and torch.equal(
                g6e_stack(m_e, Hh, xb, fe[G6.LK_KEY], nqe, B_, None, torch.zeros_like(mb)), ref_s)
            u01 = torch.rand(B_, 3, generator=gen)
            assert torch.equal(g6e_stack(m_e, Hh, xb, fe[G6.LK_KEY], nqe, B_, u01, torch.zeros_like(mb)),
                               G6.g6_stack(m_e, Hh, xb, fe[G6.LK_KEY], nqe, B_, u01))
        # the flags are fixed inputs: the family dropout's draws and the weights do not change them (the model reads
        # r2["lk"], which the carve made from all of the batch's edges)
        seen = []

        class Probe(ZG6E):
            def r2_parts(self, r2_, nq_):
                seen.append(r2_["lk"].clone())
                return super().r2_parts(r2_, nq_)
        p1, p2 = Probe(blocks, ec.widths, 16, dropout=0.1, seed=0), Probe(blocks, ec.widths, 16, dropout=0.1, seed=3)
        p1.train(), p2.train()
        with torch.no_grad():
            p1(fe, keep, nqe, B_, bze), p2(fe, keep, nqe, B_, bze), p1(fe, keep, nqe, B_, bze)
        assert len(seen) == 3 and all(torch.equal(s_, r2x["lk"]) for s_ in seen)
        # the gradients: at the start the listwise-type loss reaches the directions and not R2's logits (e_l = 0); with
        # e_l nonzero it reaches both; R2's EM term trains the logits on the linked slots from the start
        m_t = ZG6E(blocks, ec.widths, 16, dropout=0.1, seed=0)
        m_t.train()
        m_t(fe, keep, nqe, B_, bze).square().sum().backward()
        assert int(torch.count_nonzero(m_t.g6e_dir.grad)) > 0
        assert all(int(torch.count_nonzero(getattr(m_t, k_).grad)) == 0 for k_ in BRIDGE_PARAMS)
        m_t.zero_grad()
        with torch.no_grad():
            m_t.g6e_dir.copy_(torch.randn(3, 64, generator=gen) * 0.3)
        m_t(fe, keep, nqe, B_, bze).square().sum().backward()
        assert all(int(torch.count_nonzero(getattr(m_t, k_).grad)) > 0 for k_ in BRIDGE_PARAMS)
        m_t.zero_grad()
        lk_any = r2x["lk"].any(1)
        gold_t = torch.zeros(N_)
        gold_t[int(torch.nonzero(lk_any)[0])] = 1.0
        with R2.r2_em():
            m_t.train()
            R2.clear_stash()
            m_t(fe, keep, nqe, B_, bze)
            S_k, M_k, lpt_k = R2.STASH["S"], R2.STASH["M"], R2.STASH["lpt"]
            _S, M_r2 = R2.r2_logits(m_t, r2x["eq"], r2x["p"], r2x["bri"], nqe)
            assert torch.equal(M_k, M_r2 & r2x["lk"]) and M_k.sum() < M_r2.sum() and R2.STASH["N"] == N_
            assert torch.equal(lpt_k, R2.prior_logp(M_k, S_k.dtype))
            stops(m_t, fe, keep, nqe, B_, bze, has="kept and never scored")
            R2.clear_stash()
            m_t.zero_grad()
            m_t(fe, keep, nqe, B_, bze)
            nll, nqs, _nqg = R2.br_nll(R2.STASH["S"], R2.STASH["M"], R2.STASH["lpt"], nqe, B_, gold_t)
            R2.clear_stash()
            assert nqs == 1 and torch.isfinite(nll)
            nll.backward()
            assert all(int(torch.count_nonzero(getattr(m_t, k_).grad)) > 0 for k_ in ("r2_U", "r2_V", "r2_W1"))
        assert R2.STASH["S"] is None
        m_t.eval()
        with torch.no_grad():
            m_t(fe, keep, nqe, B_, bze)
        assert R2.STASH["S"] is None                                          # evaluation keeps nothing
        no_lk = {**fe, R2.R2_KEY: {k_: v for k_, v in r2x.items() if k_ != "lk"}}
        stops(m_t, no_lk, keep, nqe, B_, bze, has="holds no link flags")
        stops(m_t, {k_: v for k_, v in fe.items() if k_ != R2.R2_KEY}, keep, nqe, B_, bze, has="holds no bridges")
        # the link statistics of a read: the carve counts the flags of each batch it serves, reset per carve
        allq = np.arange(rows_, dtype=np.int64)

        def orig_read(c, models, *a, **kw):
            for _ in range(2):
                c.batch(allq, blocks)
            return ("out", a, kw)

        STATS.clear()
        assert stats_reader(orig_read)(ec, [], 65536, first_q=None) == ("out", (65536,), {"first_q": None})
        st = STATS["metaqa=toy"]
        assert st["rows"] == 2 * int(ec.n_np.sum()) and st["questions"] == 2 * rows_ and st["passes"] == 2.0
        assert st["flags"] == 2 * int(whole.sum()) and st["rows_linked"] == 2 * int(whole.any(1).sum()) and ec.r2e_stats is None
        STATS.clear()
        stops(stats_reader(orig_read), cm, [], has="is not read on R2EG's carve")
        stops(stats_reader(lambda c, m_, *a, **k_: None), ec, [], has="scored no batch")
        lg = types.SimpleNamespace(read_carve=orig_read)
        with stats_reads(lg):
            assert lg.read_carve is not orig_read
        assert lg.read_carve is orig_read and ec.r2e_stats is None
        STATS.clear()
    # 3. inside zwq.on_wq: zr2br trains on zwq's modules; zg6's tables get zg6e for the block (its arm and R2EG's fit
    #    names), zg6's own train checks them and refuses what is not R2EG's; restored after
    with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W, K.g6_on_wq(W) as G6w:
        GW.same_modules(R2, W)
        assert G6w is G6 and G6.SPLITS == tuple(WQ.SPLITS) and ARM not in G6.ARMS
        saved = (dict(G6.ARMS), {k_: dict(v) for k_, v in G6.FIT.items()})
        args = ["train", "--name", FIT["WQ-musique"], "--arm", ARM, "--device", "cuda", "--cache-root", K.ROOTS["cache"]]
        stops(G6.train, args, "WQ-musique", has="train takes --arm zg6 or zg6t")            # not R2EG's tables
        with on_arm(G6, R2) as G6a:
            assert G6a is G6 and G6.ARMS[ARM] == R2.S.ARMS[ARM] and G6.FIT[ARM] == FIT and G6.FIT["zg6"]["WQ-hotpotqa"] == \
                "scr-wqk-zg6-s0-hp"
            assert not G6.fixed_flags(args, G6.FIXED_TRAIN) and not G6.fixed_flags(["--name", "x", "--rows-cap", "65536"],
                                                                                   G6.FIXED_READ)
            stops(G6.train, args[:2] + ["scr-r2eg-other"] + args[3:], "WQ-musique", has="the zg6e fit of WQ-musique is")
            stops(G6.train, args, "L-musique", has="G6's screen fits")
            stops(G6.train, args + ["--seed", "1"], "WQ-musique", has="would change them")
        assert (dict(G6.ARMS), {k_: dict(v) for k_, v in G6.FIT.items()}) == saved and ARM not in G6.ARMS
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        B = td / "repo"
        # 4. where, the read cap, WQK's verdict: train, read and fit refuse before reading any data
        stops(arm_split, "scr-wqk-zg6-s0", has="not one of R2EG's fits")
        stops(arm_split, "scr-r2ew-zrckpbe", has="not one of R2EG's fits")
        stops(train, "scr-r2-zg6m", TOY, has="not one of R2EG's fits")
        for bad in ({"where": "elsewhere"}, {"gpu": "Tesla P100-PCIE-16GB"}, {"image": "kaggle-gpu:other"}):
            stops(where_ok, {**TOY, **bad}, "train", has="R2EG trains in a Kaggle kernel")
            stops(train, FIT["WQ-musique"], {**TOY, **bad}, has="R2EG trains in a Kaggle kernel")
            stops(read, FIT["WQ-hotpotqa"], {**TOY, **bad}, has="R2EG reads in a Kaggle kernel")
        where_ok(TOY, "read")
        cap_ok(READ_CAP)
        for cap in (0, None, 32768):
            stops(cap_ok, cap, has="--rows-cap 65536")
            stops(read, FIT["WQ-musique"], TOY, rows_cap=cap, has="--rows-cap 65536")
            stops(fit, FIT["WQ-musique"], rows_cap=cap, has="--rows-cap 65536")
        stops(fit, FIT["WQ-hotpotqa"], has="R2EG trains in a Kaggle kernel")                 # the laptop is not a T4
        stops(wqk_ready, B, has="base-verdict.json missing")
        wv = B / (K.VERDICT + ".json")
        with WQ.patch(WQ, ROOT=B):
            stops(train, FIT["WQ-musique"], TOY, has="base-verdict.json missing")
            for bad in ({"verdict": "INCOMPLETE"}, {"declared_in": WQ.DECLARED}, {"stage": "WQ"}):
                WQ.write_json(wv, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY", **bad})
                stops(wqk_ready, B, has="not WQK's BASE_READY")
                stops(train, FIT["WQ-hotpotqa"], TOY, has="not WQK's BASE_READY")
        WQ.write_json(wv, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
        assert wqk_ready(B)["verdict"] == "BASE_READY"
        shutil.rmtree(B / "outputs")
        stops(term_ok, {"batches": 7, "batches_with_term": 0}, "scr-r2eg-zg6", has="carried R2's term (7 batches)")
        term_ok({"batches": 7, "batches_with_term": 3}, "scr-r2eg-zg6")
        # 5. the stamps of both fits (WQK's wq block at the toy root, R2EG's block), fit_stamp, the read's stamp with the
        #    link statistics, fit_device; then the read's refusals on the stamped fits
        B7 = td / "stamp"
        for k_, r_ in (("GATE", {"verdict": "PASS"}), ("OVERLAY", {"status": "OK"}), ("CHECK", {"status": "REPRODUCED"})):
            WQ.write_json(B7 / K.RECORDS[k_], {"declared_in": K.DECLARED, **r_})
        WQ.write_json(B7 / (K.VERDICT + ".json"), {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
        summ = {"lam": 1.0, "batches": 10, "batches_with_term": 9, "questions": 100, "questions_with_term": 90,
                "share_with_term": 0.9, "epochs": [{"br_nll_mean": 4.6}, {"br_nll_mean": 3.9}]}
        fr7 = K.fits_dir(B7)
        for nm, sp_ in FITS.items():
            WQ.write_json(WQ.fit_dir(nm, B7) / "screen.json", {"arm": ARM, "seed": SEED, "split": sp_, "rc": 0})
            stops(fit_stamp, nm, fr7, has="no R2EG stamp")
            stamp_fit(nm, TOY, R2, summ, B7, {"kt_records": {"webqsp/fit": "k"}, "zlink_records": {"p": "z"}})
            s7, k7 = fit_stamp(nm, fr7)
            assert (k7["declared_in"], k7["device"], k7["against"], k7["beside"], k7["what"]) == (
                DECLARED, K.device_of(TOY), BASE_FIT[sp_], BESIDE_FIT[sp_], WHAT)
            assert (s7["wq"]["declared_in"], s7["wq"]["gate"], s7["wq"]["arm"]) == (
                K.DECLARED, WQ.sha_file(B7 / K.RECORDS["GATE"]), ARM)
            assert k7["gate"] == s7["wq"]["gate"] and k7["wqk_verdict"] == WQ.sha_file(B7 / (K.VERDICT + ".json"))
            assert (k7["r2"]["constants"], k7["r2"]["params"], k7["r2"]["term"]) == (R2_CONST, list(NEW_PARAMS), summ)
            assert k7["head"]["r2eg"] == R2EG and k7["head"]["read_rows"] == READ_CAP and k7["zlink_records"] == {"p": "z"}
            assert s7["zr2br_sha256"] == WQ.sha_src(HERE / "zr2br.py") and s7["zr2eg_sha256"] == WQ.sha_src(__file__)
            assert s7["zr2ew_sha256"] == WQ.sha_src(HERE / "zr2ew.py") and set(k7["code"]) == set(CODE)
        stops(fit_stamp, FIT["WQ-musique"], fr7 / "none", has="no R2EG stamp")
        n7 = FIT["WQ-hotpotqa"]
        WQ.write_json(WQ.fit_dir(n7, B7) / "read.json", {"seconds": 12.0, "carves": {c: {} for c in K.SIX}})
        stops(fit_device, n7, fr7, has="has no R2EG read on the device it was fitted on")
        r7 = stamp_read(n7, TOY, {"webqsp/fit": "k"}, READ_CAP, {"metaqa=s1eval": dict(FAKE_COUNTS)}, B7, {"p": "z"})[KEY]
        assert r7["link_stats"] == {"metaqa=s1eval": link_stats(FAKE_COUNTS)} and r7["zlink_records"] == {"p": "z"}
        assert link_stats(FAKE_COUNTS)["share_rows_linked"] == 0.3 and r7["rows_cap"] == READ_CAP
        assert fit_device(n7, fr7) == K.device_of(TOY)
        stamp_read(n7, TOY, {"webqsp/fit": "k"}, 32768, {}, B7)
        stops(fit_device, n7, fr7, has="has no R2EG read on the device it was fitted on")
        stamp_read(n7, TOY, {"webqsp/fit": "k"}, READ_CAP, {}, B7)
        assert stamped(WQ.fit_dir(n7, B7) / "read.json", n7) and not stamped(WQ.fit_dir(n7, B7) / "read.json", FIT["WQ-musique"])
        boom = types.SimpleNamespace(n=0)

        def gate_boom(base=None):
            boom.n += 1
            raise SystemExit("zwq: outputs/wqk/gate.json is FAIL, not WQ's PASS")

        with WQ.patch(WQ, fit_dir=lambda name, base=B7: Path(base) / "outputs" / "screen" / "fits" / name):
            stops(read, n7, {**TOY, "cudnn": 1}, has="this read runs on")
            sj7 = WQ.fit_dir(n7, B7) / "screen.json"
            keep7 = sj7.read_bytes()
            s7 = WQ.read_json(sj7)
            s7[KEY]["r2"]["constants"] = {**R2_CONST, "K": 6}
            WQ.write_json(sj7, s7)
            stops(read, n7, TOY, has="fitted with R2 constants")
            sj7.write_bytes(keep7)
            with WQ.patch(WQ, gate_passed=gate_boom):
                stops(read, n7, TOY, has="is FAIL")
            assert boom.n == 1
        # 6. take, compare-all and the verdict on a toy of WQK's base set (zwqk's toy, its fits, compares and floors) and
        #    an R2EG session: ADOPT_FOR_FULL_RUN, NOT_ADOPTED, and INCOMPLETE for each missing or wrong piece
        K.toy_build(B)
        gw, ow = K.fake_session(td / "wqk_out", B, tag="w")
        for nm in K.FITS:
            K.fake_fit(K.fits_dir(B), nm, gw, ow)

        def wqk_run(new, base_fit, out):                # zwqk's selftest's: WQK's compares on the toy
            new, b0, c_ = Path(new).name, Path(base_fit).name, Path(out).name
            arm, seed, sp = K.FITS[new]
            dd = {ds: (0.002, -0.002)[seed - 1] for ds in WQ.EVAL_ORDER} if arm != "zg6" else {}
            if (arm, sp) == ("zspk", "WQ-musique"):
                dd["musique"] = (0.02, -0.01)[seed - 1]
            K.fake_compare(B, c_, new, b0, dd, base_r5=0.59 if arm == "zrckp" else 0.6)
            return 0

        why = WQ.wq_why
        with WQ.patch(WQ, run_compare=wqk_run, wq_why=lambda name, base=B: why(name, base)):
            assert K.compare_all(B) == 0
        assert K.floors(B) == 0
        WQ.write_json(B / (K.VERDICT + ".json"), {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
        out_e = td / "r2eg_out"
        ge, oe = K.fake_session(out_e, B, tag="e")
        names = list(FITS)
        fake_fit(out_e / "fits", names[0], ge, oe)
        fake_fit(out_e / "fits", names[1], ge, oe, read_=False)
        took, left = take(out_e, B)
        assert len(took) == 5 + 2 and len(left) == 1 and "has no R2EG read" in left[0], (took, left)
        fr = K.fits_dir(B)
        assert not (fr / names[0] / "models.pt").exists() and (fr / names[0] / TIMING_FILE).is_file()
        assert (B / SESSIONS / ge[:12] / "gate.json").is_file() and not (fr / names[1]).exists()
        fake_fit(out_e / "fits", names[1], ge, oe)
        took, left = take(out_e, B)
        assert len(took) == 5 and not left and take(out_e, B) == ([], [])
        (fr / names[0] / "train.json").write_text("{}", encoding="utf-8")
        stops(take, out_e, B, has="exists with other bytes")
        shutil.copyfile(out_e / "fits" / names[0] / "train.json", fr / names[0] / "train.json")
        (out_e / "fits" / BASE_FIT["WQ-musique"]).mkdir()
        stops(take, out_e, B, has="are not R2EG's fits")
        (out_e / "fits" / BASE_FIT["WQ-musique"]).rmdir()
        stops(take, td / "wqk_out" / "wqk", B, has="holds no wqk/gate.json")
        cur = {}

        def r2eg_run(new, base_fit, out):               # lean_screen's compare with two bases, in its record's form
            import lean_screen as S
            n_, c_ = Path(new).name, Path(out).name
            sp = FITS[n_]
            bs = [Path(b).name for b in base_fit.split(",")]
            assert bs == COMPARES[c_][1] and c_ == CMP[sp], (bs, c_)
            by = {}
            for b in bs:
                rows = []
                for ds in WQ.EVAL_ORDER:
                    nw = round(0.6 + cur.get(("e", sp, ds), 0.0), 4)
                    r5 = 0.6 if b == BESIDE_FIT[sp] else round(0.6 + cur.get(("b", sp, ds), 0.0), 4)
                    dl = round(nw - r5, 4)
                    lo, hi = round(dl - 0.004, 4), round(dl + 0.004, 4)
                    rows.append({"dataset": ds, "read": "in-domain" if ds in WQ.SPLITS[sp] else "zero-shot",
                                 "questions": 100, "base": [r5, 0.5, 0.4], "new": [nw, 0.5, 0.4],
                                 "rrf": [0.4, 0.3, 0.2], "delta": [dl, 0, 0], "lo": [lo, 0, 0], "hi": [hi, 0, 0],
                                 "call": S.call(dl, lo, hi)})
                by[f"outputs/screen/fits/{b}"] = {"rows": rows, "verdict": S.verdict([r["call"] for r in rows]),
                                                  "decides": b == bs[0]}
            bp = list(by)
            WQ.write_json(K.screen_dir(B) / f"{c_}.json", {
                "new": f"outputs/screen/fits/{n_}", "bases": bp, "candidate": "p@swa", "carve": "s1eval",
                "trained_on": sorted(WQ.SPLITS[sp]), "by_base": by, "verdict": by[bp[0]]["verdict"]})
            return 0

        def run(deltas):
            cur.clear()
            cur.update(deltas)
            for c_ in COMPARES:
                (K.screen_dir(B) / f"{c_}.json").unlink(missing_ok=True)
            with WQ.patch(WQ, run_compare=r2eg_run):
                assert compare_all(B) == 0 and compare_all(B) == 0
            return verdict(B)

        v = run({("e", "WQ-hotpotqa", "squad"): 0.03, ("b", "WQ-musique", "metaqa"): 0.02,
                 ("e", "WQ-musique", "metaqa"): 0.02})          # zg6e equals zg6 on metaqa, and both are above zspk
        assert v["verdict"] == "ADOPT_FOR_FULL_RUN" and not v["problems"], v["problems"]
        assert (v["gains"], v["losses"], v["next_form"]) == (["WQ-hotpotqa squad"], [], NEXT["ADOPT_FOR_FULL_RUN"])
        assert v["gates"] == [ge] and v["device"] == K.device_of(TOY) and len(v["calls"]) == 12 and v["naming"] == NAMING
        assert v["fits"] == {ARM: FIT} and v["beside_fits"] == BESIDE_FIT and v["params_added"] == NEW_N
        k_ = WQ.read_json(K.screen_dir(B) / f"{CMP['WQ-hotpotqa']}.json")[KEY]
        assert (k_["new"], k_["bases"], k_["device"]) == (FIT["WQ-hotpotqa"], [BASE_FIT["WQ-hotpotqa"], BESIDE_FIT["WQ-hotpotqa"]],
                                                          K.device_of(TOY))
        b = v["beside"]
        assert sorted(b["calls"]["zg6e-vs-zspk"]["gains"]) == ["WQ-hotpotqa squad", "WQ-musique metaqa"] and             not b["calls"]["zg6e-vs-zspk"]["losses"]
        m5 = b["r5"]["WQ-musique metaqa"]
        assert {k_: m5[k_] for k_ in m5 if k_ != "floor"} == {
            "read": "zero-shot", "rrf": 0.4, "zspk": 0.6, "zg6": 0.62, "zg6e": 0.62, "call": "WITHIN",
            "zg6e_vs_zspk": "GAIN", "links": link_stats(FAKE_COUNTS)}, m5
        assert m5["floor"] == 0.0075, m5["floor"]                                  # zspk's floor on metaqa (the minimum)
        assert b["r5"]["WQ-musique musique"]["floor"] == round(2 * (((0.02 ** 2 + 0.01 ** 2) / 2) ** 0.5), 6)
        assert b["terms"][FIT["WQ-musique"]] == {"questions": 100, "questions_with_term": 90, "share_with_term": 0.9,
                                                 "br_nll_first": 4.6, "br_nll_last": 3.9}
        assert b["timing"]["WQ-musique"]["zg6e"] == {"train_seconds": 150.0, "read_seconds": 12.0, "train_peak_gpu_gb": 3.5,
                                                     "read_peak_gpu_gb": 4.5, "data_gb": 2.0, "peak_gb": 9.0}
        assert b["timing"]["WQ-hotpotqa"]["zg6"]["peak_gb"] == 8.0 and b["timing"]["WQ-hotpotqa"]["zspk"]["peak_gb"] == 8.0
        assert b["recall_sha256"] == WQ.sha_file(B / (RECALL["zg6e"] + ".json"))
        for c_ in CALLS:
            r_ = WQ.read_json(B / (RECALL[c_] + ".json"))
            assert (r_["stage"], r_["against"], r_["deciding"], len(r_["rows"])) == (STAGE, AGAINST[c_], DECIDING[c_], 12)
            assert (B / (PAIR[c_] + ".md")).read_text(encoding="utf-8").startswith(f"R2EG ({DECLARED})")
        r0 = WQ.read_json(B / (RECALL["zg6e"] + ".json"))
        assert {(r["split"], r["dataset"]): r["base"] for r in r0["rows"]}[("WQ-musique", "metaqa")] == 0.62
        r1 = WQ.read_json(B / (RECALL["zg6e-vs-zspk"] + ".json"))
        assert {(r["split"], r["dataset"]): r["base"] for r in r1["rows"]}[("WQ-musique", "metaqa")] == 0.6
        md = (B / (VERDICT + ".md")).read_text(encoding="utf-8")
        assert "zg6e **ADOPT_FOR_FULL_RUN**" in md and "| WQ-hotpotqa squad (in-domain) | 0.4000 | 0.6000 | 0.6000 |" in md
        v = run({("e", "WQ-hotpotqa", "squad"): 0.03, ("e", "WQ-musique", "webqsp"): -0.03})
        assert (v["verdict"], v["losses"], v["next_form"]) == ("NOT_ADOPTED", ["WQ-musique webqsp"], NEXT["NOT_ADOPTED"])
        v = run({("e", "WQ-musique", "musique"): 0.02})        # a filed GAIN within the raised floor (0.0316)
        assert (v["verdict"], v["gains"], v["losses"], v["next_form"]) == ("NOT_ADOPTED", [], [], NEXT["NOT_ADOPTED"])
        assert v["calls"]["WQ-musique musique"] == "WITHIN" and v["delta_r5"]["WQ-musique musique"] == 0.02
        r0 = WQ.read_json(B / (RECALL["zg6e"] + ".json"))
        assert [r["call_filed"] for r in r0["rows"] if (r["split"], r["dataset"]) == ("WQ-musique", "musique")] == ["GAIN"]

        def incomplete(has):
            r = verdict(B)
            assert r["verdict"] == "INCOMPLETE" and r["beside"] == {} and r["next_form"] == NEXT["INCOMPLETE"] and \
                any(has in p for p in r["problems"]), (has, r["problems"])

        def away(p, has):
            keep_ = p.read_bytes()
            p.unlink()
            incomplete(has)
            p.write_bytes(keep_)

        ce = K.screen_dir(B) / f"{CMP['WQ-hotpotqa']}.json"
        away(ce, f"{CMP['WQ-hotpotqa']}.json missing")
        away(B / (K.VERDICT + ".json"), "base-verdict.json missing")
        keep = ce.read_bytes()
        rc_ = WQ.read_json(ce)
        rc_[KEY]["new"] = "scr-r2eg-zg6"
        WQ.write_json(ce, rc_)
        incomplete("is not R2EG's compare of")
        ce.write_bytes(keep)
        n0 = FIT["WQ-musique"]
        bad_r2 = {"declared_in": R2_DECLARED, "constants": {**R2_CONST, "K": 6}, "params": list(NEW_PARAMS),
                  "term": {"lam": 1.0, "questions_with_term": 90}}
        old_params = {"declared_in": R2_DECLARED, "constants": dict(R2_CONST), "params": list(GW.R2_PARAMS),
                      "term": {"lam": 1.0, "questions_with_term": 90}}
        no_term = {"declared_in": R2_DECLARED, "constants": dict(R2_CONST), "params": list(NEW_PARAMS),
                   "term": {"lam": 1.0, "questions_with_term": 0}}
        for nm, kw, has in (
                (n0, {"dev": {**K.device_of(TOY), "gpu": "Tesla P100"}}, "ran on different devices"),
                (n0, {"gate_sha": "f" * 64}, "names the gate ffffffffffff, not a filed R2EG PASS gate"),
                (n0, {"r2": bad_r2}, f"{n0}: R2's constants, parameters or EM term are not as declared"),
                (n0, {"r2": old_params}, f"{n0}: R2's constants, parameters or EM term are not as declared"),
                (n0, {"r2": no_term}, f"{n0}: R2's constants, parameters or EM term are not as declared"),
                (n0, {"rows_cap": 32768}, "has no R2EG read on the device it was fitted on"),
                (n0, {"r2eg": None}, f"{n0}: its message is not R2eg's as declared"),
                (n0, {"r2eg": {**R2EG, "dir": "a scalar"}}, f"{n0}: its message is not R2eg's as declared"),
                (n0, {"stats": False}, f"{n0}: its read holds no link statistics on the six carves")):
            fake_fit(fr, nm, **{"gate_sha": ge, "overlay_sha": oe, **kw})
            incomplete(has)
            fake_fit(fr, nm, ge, oe)
        out2 = td / "r2eg_out2"
        g2_, _o2 = K.fake_session(out2, B, verdict_="FAIL", tag="h")
        take(out2, B)
        incomplete("FAIL, not WQK's PASS")
        shutil.rmtree(B / SESSIONS / g2_[:12])
        for kind, fname in (("zspk", "scr-wqk-zspk-s1.json"),):
            p = K.screen_dir(B) / fname
            keep = p.read_bytes()
            r = WQ.read_json(p)
            r["by_base"]["outputs/screen/fits/scr-wqk-zspk-s0"]["rows"][2]["delta"][0] = 0.05
            WQ.write_json(p, r)
            incomplete(f"floors-{kind}.json is not its compares' floors")
            p.write_bytes(keep)

        def short(base, dev, c_):
            rec = recall.__wrapped__(base, dev, c_)
            rec["rows"] = rec["rows"][:-1]
            return rec

        short_fn = types.FunctionType(short.__code__, {**globals(), "recall": types.SimpleNamespace(__wrapped__=recall)})
        with WQ.patch(me, recall=short_fn):
            incomplete("holds 11 reads, not the twelve")
        assert verdict(B)["verdict"] == "NOT_ADOPTED"
    # 7. nothing left patched in zr2br, zg6, lean_gpu or the carve, no stats kept, and nothing written under R2EG's real folders
    assert R2.STASH["S"] is None and not STATS and ARM not in G6.ARMS and ARM not in G6.FIT
    assert real_state() == before, (real_state(), before)
    print(f"zr2eg selftest: ok ({time.time() - t0:.1f}s)", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    import argparse
    ap = argparse.ArgumentParser(prog="zr2eg.py", allow_abbrev=False)
    ap.add_argument("cmd", choices=("fit", "take", "compare-all", "verdict"))
    ap.add_argument("--name")
    ap.add_argument("--rows-cap", type=int, default=READ_CAP)
    ap.add_argument("--from", dest="src")
    a = ap.parse_args(argv)
    src = Path(a.src).resolve() if a.src else None
    os.chdir(WQ.ROOT)
    if a.cmd == "fit":
        if not a.name:
            ap.error("fit takes --name")
        return fit(a.name, a.rows_cap)
    if a.cmd == "take":
        if src is None:
            ap.error("take takes --from")
        take(src)
        return 0
    if a.cmd == "compare-all":
        return compare_all()
    return 0 if verdict()["verdict"] != "INCOMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
