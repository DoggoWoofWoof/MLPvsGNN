"""R2EW (docs/R2EW_R2E_UNDER_WQ.md): R2e, the MLP half, under WQ on Kaggle's T4, against WQK's base set
(docs/WQK_WQ_ON_T4.md) and beside R2GW's zrckpb fits (zr2gw.py). zrckpbe is zrckpb (zr2br.py, imported unchanged: zrckp
plus R2's bridge head and its EM term) with R2's slot mask M(v, k) replaced by M(v, k) and link(v, k): a bridge scores a
row only when zlink's edges hold an edge between the bridge's row and that row. The flag is fixed per carve from zlink's
built edges (outputs/wqk/p1c2/links/<dataset>/<carve>/part_*: q_ne, e_u, e_v, e_fam, read by zlink.load_links) and the
stored first-stage RRF column; no learned value, score or label enters it, so the MLP stays MP-free. It is trained by
zp1c3p's train under an arm swap with zr2br's EM term, at WQK's names and roots with WQK's mixed KT input (zwqk.py,
imported unchanged), as WQK trains zrckp, and read on the device it was fitted on with the link statistics per carve
kept. Each fit is compared with WQK's seed-0 zrckp fit of its split and with R2GW's zrckpb fit of it, all on one device,
and zrckpbe's re-call against zrckp under WQK's zrckp seed null decides. The GNN half of R2e is a later file.

In the kernel (zr2ew_session.py runs each in its own process, after zwqk.py overlay and gate):

    python outputs/mp_unified/zr2ew.py fit --name scr-r2ew-zrckpbe-hp --rows-cap 65536

On the laptop, after `zr2gw.py take` of R2GW's zrckpb fits and `kaggle kernels output` of the session into a new folder:

    python outputs/mp_unified/zr2ew.py take --from DIR   # each fit's records and reads, never its weights
    python outputs/mp_unified/zr2ew.py compare-all       # outputs/screen/scr-r2ew-zrckpbe-vs-zrckp[-hp].json
    python outputs/mp_unified/zr2ew.py verdict           # outputs/r2ew/screen-verdict (.json, .md)
    python outputs/mp_unified/zr2ew.py --selftest

This file imports the standard library, zwqk (with it zwq) and zr2gw only; zr2br, zlink and torch are imported by the
commands that need them, outside zwq.on_wq. verdict exits 0 on ADOPT_FOR_FULL_RUN or NOT_ADOPTED and 2 after writing an
INCOMPLETE record.
"""
import contextlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zr2gw as GW  # noqa: E402  (the standard library, zwq and zwqk at import)

K, WQ = GW.K, GW.WQ
DECLARED = "docs/R2EW_R2E_UNDER_WQ.md"
STAGE = "R2EW"
KEY = "r2ew"                                  # the stamp's key in screen.json, read.json and R2EW's compares
ARM, BESIDE_ARM, BASE = "zrckpbe", GW.BESIDE_ARM, GW.BASE
SEED = 0
SPLITS = tuple(GW.SPLITS)
READ_CAP = GW.READ_CAP                        # WQK's rows a read batch (whole questions, so the scores do not change)
NB = GW.NB                                    # R2's bridges per question
R2_N, R2_CONST, R2_PARAMS, R2_DECLARED = GW.R2_N, GW.R2_CONST, GW.R2_PARAMS, GW.R2_DECLARED
FIT = {sp: f"scr-r2ew-{ARM}{WQ.SUF[sp]}" for sp in SPLITS}
FITS = {n: sp for sp, n in FIT.items()}       # name -> split
BASE_FIT = dict(GW.BASE_FIT)                  # WQK's seed-0 zrckp: the deciding base
BESIDE_FIT = {sp: GW.FIT[(BESIDE_ARM, sp)] for sp in SPLITS}      # R2GW's zrckpb: the second base, beside
ZSPK_CMP = dict(GW.ZSPK_CMP)
CMP = {sp: f"scr-r2ew-{ARM}-vs-zrckp{WQ.SUF[sp]}" for sp in SPLITS}
COMPARES = {CMP[sp]: (FIT[sp], [BASE_FIT[sp], BESIDE_FIT[sp]]) for sp in SPLITS}
NEW = "outputs/r2ew/"
SESSIONS = NEW + "sessions"
VERDICT = NEW + "screen-verdict"
CALLS = {"zrckpbe": 0, "zrckpbe-vs-zrckpb": 1}                  # call -> the compare's base column
AGAINST = {"zrckpbe": "zrckp", "zrckpbe-vs-zrckpb": "zrckpb"}
DECIDING = {"zrckpbe": True, "zrckpbe-vs-zrckpb": False}
PAIR = {c: f"{NEW}r2ew-{c}-pair" for c in CALLS}
RECALL = {c: f"{p}-recall" for c, p in PAIR.items()}
NULL_KIND = {"zrckpbe": "WQK's zrckp seed null",
             "zrckpbe-vs-zrckpb": "WQK's zrckp seed null, its seed-0 R@5 replaced by R2GW's zrckpb read (beside)"}
FLOORS = K.FLOORS[BASE]
TIMING_FILE = "r2ew_timing.json"
CODE = tuple(dict.fromkeys(GW.CODE + ("zlink", "zr2ew")))
LSTAT = ("rows", "rows_linked", "flags", "questions", "questions_linked")      # the link statistics a read counts
WHAT = "zrckp + R2's bridge head on linked bridges"
NAMING_R2T = GW.NAMING_R2T
NAMING = NAMING_R2T.replace("zrckpb is called \"zrckp + R2's bridge head\"", f"zrckpbe is called \"{WHAT}\"") + \
    "; the edges answer R2's flag 2 (graph-free) only, and no claim sets zrckpbe against the GNN as the MLP"
R2E_TEXT = GW.R2E
R2E = {"form": "R2's slot mask M(v, k) replaced by M(v, k) and link(v, k)",
       "link": "1 if the question's zlink edges hold an edge (u, v), in any family, with u the pool-local row of R2's "
               "bridge k and v the row; else 0 (the bridge row itself has no self-loop; absent bridges are 0)",
       "fixed": "a function of the links arrays and the stored c_rrf column only; no learned value, score or label",
       "links": "outputs/wqk/p1c2/links/<dataset>/<carve>/part_*: q_ne, e_u, e_v, e_fam (zlink.load_links)",
       "mask_read_by": ["prior over the row's slots", "a, m, f", "R2's EM term"], "params_added": 0,
       "init": "R2's last layer is 0: zrckpbe starts as zrckp bit for bit; with every flag 1 it is zrckpb bit for bit"}
RULE = ("ADOPT_FOR_FULL_RUN when zrckpbe's re-call against WQK's seed-0 zrckp under WQK's zrckp seed null has no LOSS on "
        "any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when WQK's base verdict is not "
        "BASE_READY, a fit or compare is missing or not R2EW's (for R2GW's zrckpb, not R2GW's), R2EW's fits, R2GW's "
        "zrckpb fits and WQK's seed-0 zrckp fits are not on one device, a fit or read names a gate that is not a filed "
        "PASS gate of its session, a fit's R2 constants or EM term are not as declared, zrckpbe's flag is not as "
        "declared or its read holds no link statistics on the six carves, WQK's floors are not their compares', or a "
        "re-call does not hold the twelve reads (docs/R2EW_R2E_UNDER_WQ.md section 5)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zrckpbe fitted again on PKO's pools (docs/PKO_PICK_O_POOLS_UNDER_WQ.md, outputs/pko, whose "
                              "links PKO builds), then its full run under WQ beside zrckp's, each its own declared file",
        "NOT_ADOPTED": "R2e's GNN half under WQ on the T4, its own declared file: bridge scores only for linked bridges "
                       "inside zg6's messages (R2's text: " + R2E_TEXT + "); the MLP half closes with this verdict",
        "INCOMPLETE": "what is missing is supplied (WQK's base set, R2GW's zrckpb fits, a fit, a compare, a session's "
                      "gate or a re-call); nothing else changes"}
assert len(FIT) == len(FITS) == 2 and not set(FITS) & (set(K.FITS) | set(GW.FITS))
assert all(CMP[sp] not in GW.COMPARES and CMP[sp] not in K.COMPARES for sp in SPLITS) and FLOORS == GW.FLOORS


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zr2ew: {msg}", flush=True)


# ── the flag (section 3) ─────────────────────────────────────────────────────


def flags_np(q_ne, e_u, e_v, n, br):
    """(N, 5) bool, the reference: row v of question q has flag k when q's links hold an edge (u, v) with u the pool-local
    row of bridge k (br, R2's (Q, 5) pool offsets, -1 absent). q_ne edges per question; e_u and e_v pool-local rows; n rows
    per question. Reads the links arrays and the bridges only. zlink stores each edge in both directions."""
    import numpy as np
    q_ne, n, br = np.asarray(q_ne, np.int64), np.asarray(n, np.int64), np.asarray(br, np.int64)
    u, v = np.asarray(e_u, np.int64), np.asarray(e_v, np.int64)
    Q = n.size
    if q_ne.shape != (Q,) or br.shape != (Q, NB) or u.ndim != 1 or u.shape != v.shape or int(q_ne.sum()) != u.size:
        raise SystemExit(f"zr2ew: links of {q_ne.shape} questions with {u.size} edges and bridges {br.shape} do not fit "
                         f"{Q} questions")
    off = np.r_[0, np.cumsum(n)]
    qi = np.repeat(np.arange(Q), q_ne)
    pool = n[qi]
    if ((u < 0) | (v < 0) | (u >= pool) | (v >= pool)).any():
        raise SystemExit("zr2ew: a link edge lies outside its question's pool")
    slots = br[qi]
    hit = (slots >= 0) & (slots == u[:, None])
    rows = off[qi] + v
    flags = np.zeros((int(off[-1]), NB), bool)
    for k in range(NB):
        flags[rows[hit[:, k]], k] = True
    return flags


# ── zr2br and zlink, imported outside zwq.on_wq, and zrckpbe's classes ───────

_R2 = {}


def import_r2():
    """zr2br (through zr2gw's import: torch, zg6, zp1c3p and R2's arms in lean_screen's table, R2 checked as R2GW's) and
    zrckpbe's model and carve, built once and put in lean_screen's table. Returns {"R2", "np", "torch", "link_flags",
    "stats_add", "edge_carve", "EdgeMix", "ZRCKPBE", "ECarve"}."""
    if _R2:
        return _R2
    R = GW.import_r2()
    R2, np, torch = R["R2"], R["np"], R["torch"]
    ZL = R2.ZL

    def link_flags(lk, bri, nq, N):
        """(N, 5) bool on the batch's device: row v has flag k when the batch's edges hold (u, v) with u the batch row of the
        bridge in slot k of u's question. lk is zlink.link_batch's {u, v, f} (batch rows); bri is R2's (B, 5) bridge
        batch rows (-1 absent); nq the row's question. A function of the edges and the bridges, never of a score."""
        u, v = lk["u"], lk["v"]
        flags = torch.zeros(N, bri.shape[1], dtype=torch.bool, device=u.device)
        for k in range(bri.shape[1]):
            b = bri[:, k][nq[u]]
            flags[v[(b >= 0) & (b == u)], k] = True
        return flags

    def stats_add(st, flags, nq, B):
        rowany = flags.any(1)
        qa = torch.zeros(B, dtype=torch.int64, device=flags.device).index_add_(0, nq, rowany.to(torch.int64))
        st["rows"] += int(flags.shape[0])
        st["rows_linked"] += int(rowany.sum())
        st["flags"] += int(flags.sum())
        st["questions"] += int(B)
        st["questions_linked"] += int((qa > 0).sum())

    def edge_carve(base, lk_root=None):
        """base (an R2 carve class) with zlink's pool edges loaded beside it: batch() adds the flags under R2's key as
        "lk"; every block, row, side key, gold and bridge is base's, unchanged."""

        class EdgeCarve(base):
            LK_ROOT = lk_root

            def __init__(self, ds, carve, basis, device, cache_root=R2.LC.OUT, verify=True, score2=False):
                super().__init__(ds, carve, basis, device, cache_root, verify, score2)
                root = type(self).LK_ROOT or ZL.LinkCarveBase.LK_ROOT
                self.links = ZL.load_links(self, root, cache_root, verify)
                self.r2e_stats = None

            def nbytes(self):
                return super().nbytes() + sum(int(self.links[k].nbytes) for k in ("q_ne", "e_u", "e_v", "e_fam", "e_off"))

            def batch(self, qs, blocks):
                feats, nq, base_z, gold = super().batch(qs, blocks)
                r2 = feats[R2.R2_KEY]
                flags = link_flags(ZL.link_batch(self, qs), r2["bri"], nq, int(nq.numel()))
                r2["lk"] = flags
                if self.r2e_stats is not None:
                    stats_add(self.r2e_stats, flags, nq, int(r2["bri"].shape[0]))
                return feats, nq, base_z, gold

        EdgeCarve.__name__ = EdgeCarve.__qualname__ = f"E{base.__name__}"
        return EdgeCarve

    class EdgeMix:
        """R2Mix's slot mask M(v, k) and link(v, k); the prior, [a, m, f] and the EM term all read this mask."""

        def r2_parts(self, r2, nq):
            S_, M = R2.r2_logits(self, r2["eq"], r2["p"], r2["bri"], nq)
            lk = r2.get("lk")
            if lk is None:
                raise SystemExit("zr2ew: zrckpbe's batch holds no link flags (its carve is not R2EW's)")
            M = M & lk
            return S_, M, R2.prior_logp(M, S_.dtype)

    class ZRCKPBE(EdgeMix, R2.ZRCKPB):
        """zrckpbe: zrckpb with the bridge slots restricted to the row's linked bridges (R2e, the MLP half)."""

    ECarve = edge_carve(R2.R2KTChainCarveZRC)
    R2.S.ARMS[ARM] = (ZRCKPBE, ECarve)
    _R2.update(R2=R2, np=np, torch=torch, link_flags=link_flags, stats_add=stats_add, edge_carve=edge_carve,
               EdgeMix=EdgeMix, ZRCKPBE=ZRCKPBE, ECarve=ECarve)
    return _R2


STATS = {}


def stats_reader(orig):
    """lean_gpu.read_carve that also counts the link flags of the carve it reads: the carve counts them in batch() while
    r2e_stats is a dict, so they are the flags the read's batches carried, whatever the number of passes."""

    def read_carve(c, models, *a, **kw):
        if not hasattr(c, "r2e_stats"):
            raise SystemExit(f"zr2ew: {c.ds}={c.carve} is not read on R2EW's carve")
        c.r2e_stats = {k: 0 for k in LSTAT}
        try:
            out = orig(c, models, *a, **kw)
            st = dict(c.r2e_stats)
        finally:
            c.r2e_stats = None
        q, rows = len(c.n_np), int(c.n_np.sum())
        if st["rows"] == 0 or st["questions"] == 0:
            raise SystemExit(f"zr2ew: {c.ds}={c.carve}: the read scored no batch ({st['questions']} questions of {q})")
        STATS[f"{c.ds}={c.carve}"] = {**st, "passes": round(st["rows"] / rows, 4), "rows_in_carve": rows,
                                      "questions_in_carve": q}
        return out

    return read_carve


@contextlib.contextmanager
def stats_reads(LG):
    """lean_gpu's read_carve counts the link flags while zrckpbe is read."""
    with WQ.patch(LG, read_carve=stats_reader(LG.read_carve)):
        yield LG


def link_stats(st):
    """One carve's counts and the shares the verdict reports."""
    rows, qs = st["rows"], st["questions"]
    return {**st, "share_rows_linked": round(st["rows_linked"] / rows, 6), "flags_per_row": round(st["flags"] / rows, 6),
            "share_questions_linked": round(st["questions_linked"] / qs, 6)}


# ── guards ───────────────────────────────────────────────────────────────────


def arm_split(name):
    if name not in FITS:
        raise SystemExit(f"zr2ew: {name} is not one of R2EW's fits ({', '.join(FIT.values())})")
    return ARM, FITS[name]


def where_ok(env, what):
    if not K.ok_where(env):
        raise SystemExit(f"zr2ew: R2EW {what}s in a Kaggle kernel on a {K.GPU} from the image {K.IMAGE_TAG} (section "
                         f"4), not with {K.device_of(env)} ({env.get('where')})")


def cap_ok(rows_cap):
    if rows_cap is None or int(rows_cap) != READ_CAP:
        raise SystemExit(f"zr2ew: R2EW reads with --rows-cap {READ_CAP} (WQK's, section 4), not {rows_cap}")


def wqk_ready(base=None):
    """WQK's verdict record, committed before the stub: BASE_READY under WQK's declaration (section 4)."""
    p = Path(base or WQ.ROOT) / (K.VERDICT + ".json")
    if not p.is_file():
        raise SystemExit(f"zr2ew: {K.VERDICT}.json missing; R2EW trains after WQK's BASE_READY (section 4)")
    rec = WQ.read_json(p)
    if (rec.get("stage"), rec.get("declared_in"), rec.get("verdict")) != (K.STAGE, K.DECLARED, "BASE_READY"):
        raise SystemExit(f"zr2ew: {K.VERDICT}.json is {rec.get('verdict')}, not WQK's BASE_READY")
    return rec


def term_ok(summ, name):
    """A fit whose training had no batch with R2's term is not R2EW's (section 4): it stops unstamped."""
    if not summ.get("batches_with_term"):
        raise SystemExit(f"zr2ew: no training batch of {name} carried R2's term ({summ.get('batches')} batches); unstamped")


# ── train and read in the kernel (section 4) ─────────────────────────────────


def code_shas():
    return {m: WQ.sha_src(HERE / f"{m}.py") for m in CODE}


def link_records():
    """The built links' record.json hashes under WQK's links root (zlink.built_records), as the fit and the read used them."""
    with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W, W.Z2.on_p1c2():
        return W.Z3.ZL.built_records()


def head_of():
    return {"r2e": R2E, "kt_root": K.ROOTS["kt"], "roots": dict(K.ROOTS), "read_rows": READ_CAP}


def stamp_fit(name, env, R2, summ, base=None, records=None):
    """The fit's record after zp1c3p's train: the wq block at WQK's roots and declaration (zwqk.wq_block), and R2EW's block
    (the device, WQK's gate, overlay, check and verdict, the KT, chain and link records, R2's constants, parameters and EM
    term, the flag's declaration, the code)."""
    base = Path(base or WQ.ROOT)
    arm, split = arm_split(name)
    sj = WQ.fit_dir(name, base) / "screen.json"
    rec = WQ.read_json(sj)
    records = GW.fit_records(split) if records is None else records
    with WQ.patch(WQ, ROOT=base):
        wq = K.wq_block(arm, SEED, split)
    rec.update({"zwq_sha256": WQ.sha_src(WQ.__file__), "zwqk_sha256": WQ.sha_src(K.__file__),
                "zr2br_sha256": WQ.sha_src(HERE / "zr2br.py"), "zr2gw_sha256": WQ.sha_src(HERE / "zr2gw.py"),
                "zr2ew_sha256": WQ.sha_src(__file__), "wq": wq})
    rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": arm, "what": WHAT, "seed": SEED, "split": split,
                "against": BASE_FIT[split], "beside": BESIDE_FIT[split], "device": K.device_of(env), "env": env,
                "gate": WQ.sha_file(base / K.RECORDS["GATE"]), "overlay": WQ.sha_file(base / K.RECORDS["OVERLAY"]),
                "check": WQ.sha_file(base / K.RECORDS["CHECK"]),
                "wqk_verdict": WQ.sha_file(base / (K.VERDICT + ".json")), "cache_root": K.ROOTS["cache"], **records,
                "r2": {"declared_in": R2.DECLARED, "constants": R2.constants(), "params": list(R2.R2_PARAMS),
                       "settings": R2.SETTINGS, "term": summ},
                "head": head_of(), "code": code_shas()}
    WQ.write_json(sj, rec)
    return rec


def train(name, env=None):
    """R2EW's fit (section 4): zp1c3p's train for the arm under the arm swap with R2's EM term (LAM 1), at WQK's names and
    roots with WQK's mixed KT input, as zwqk's train runs zrckp; then the fit's stamps."""
    arm, split = arm_split(name)
    env = K.device_env() if env is None else env
    where_ok(env, "train")
    wqk_ready()
    R2 = import_r2()["R2"]
    with K.on_wqk():
        WQ.gate_passed()
    args = ["train", "--name", name, "--arm", arm, "--seed", str(SEED), "--device", "cuda", "--cache-root",
            K.ROOTS["cache"]]
    with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W, GW.on_arm(W, arm):
        GW.same_modules(R2, W)
        with WQ.patch(W.Z3P, FIT={**W.Z3P.FIT, split: name}), R2.r2_em():
            rc = W.Z3P.train(args, split)
    summ = R2.summary()
    if rc == 0:
        term_ok(summ, name)
        stamp_fit(name, env, R2, summ)
        ep = summ.get("epochs") or []
        log(f"train {name}: done, stamped (R2's term on {summ['questions_with_term']} of {summ['questions']} questions"
            + (f"; L_br {ep[0]['br_nll_mean']:.2f} -> {ep[-1]['br_nll_mean']:.2f})" if ep else ")"))
    return rc


def fit_stamp(name, fits_root=None):
    """An R2EW fit's screen.json and its R2EW block, checked against the fit's name."""
    arm, split = arm_split(name)
    sj = Path(fits_root or WQ.fit_dir(name).parent) / name / "screen.json"
    rec = WQ.read_json(sj) if sj.is_file() else {}
    k = rec.get(KEY) or {}
    if (rec.get("arm"), k.get("declared_in"), k.get("fit"), k.get("arm"), k.get("seed"), k.get("split")) != \
            (arm, DECLARED, name, arm, SEED, split):
        raise SystemExit(f"zr2ew: {name} is not R2EW's {arm} fit of {split} (no R2EW stamp)")
    return rec, k


def stamp_read(name, env, kt, rows_cap, stats, base=None, zl_records=None):
    """read.json's R2EW block: the device, the gate, the KT and link records, and the link statistics per carve."""
    base = Path(base or WQ.ROOT)
    arm, split = arm_split(name)
    rj = WQ.fit_dir(name, base) / "read.json"
    rec = WQ.read_json(rj)
    rec["zr2ew_sha256"] = WQ.sha_src(__file__)
    rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": arm, "seed": SEED, "split": split,
                "device": K.device_of(env), "env": env, "gate": WQ.sha_file(base / K.RECORDS["GATE"]),
                "kt_records": kt, "zlink_records": zl_records, "rows_cap": int(rows_cap),
                "link_stats": {c: link_stats(st) for c, st in stats.items()}}
    WQ.write_json(rj, rec)
    return rec


def read(name, env=None, rows_cap=READ_CAP):
    """The fit read by zp1c3p's read for its arm with p@swa on the six s1eval carves, --rows-cap 65536, on the device it was
    fitted on (zwqk's read, under the arm swap); the link flags counted per carve. read.json is stamped when the read exits 0."""
    arm, split = arm_split(name)
    cap_ok(rows_cap)
    env = K.device_env() if env is None else env
    where_ok(env, "read")
    rec, k = fit_stamp(name)
    if K.device_of(env) != k.get("device"):
        raise SystemExit(f"zr2ew: {name} was fitted on {k.get('device')}; this read runs on {K.device_of(env)}")
    got = (k.get("r2") or {}).get("constants")
    if got != R2_CONST:
        raise SystemExit(f"zr2ew: {name} was fitted with R2 constants {got}, not R2's declared {R2_CONST}")
    R2 = import_r2()["R2"]
    with K.on_wqk():
        WQ.cache_root_ok(K.ROOTS["cache"])
        WQ.gate_passed()
    argv = ["--name", name, "--device", "cuda", "--cache-root", K.ROOTS["cache"], "--rows-cap", str(READ_CAP)]
    STATS.clear()
    rc = 1
    try:
        with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W, GW.on_arm(W, arm):
            GW.same_modules(R2, W)
            with stats_reads(W.LG):
                rc = W.Z3P.read(argv)
            kt = W.Z3.kt_records(W.Z3.eval_carves())
        rj = WQ.fit_dir(name) / "read.json"
        if rc == 0 and rj.is_file():
            stamp_read(name, env, kt, rows_cap, dict(STATS), zl_records=link_records())
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
    """One fit in the kernel (zwqk's fit, for R2EW): a read R2EW stamped is kept; a fit trained and stamped but not read is
    only read; otherwise the fit's files are unlinked and it trains and reads. r2ew_timing.json holds the seconds and the peak."""
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
    """An R2EW fit's device: its screen.json's R2EW block, and its read.json's on the same device with --rows-cap 65536."""
    _rec, k = fit_stamp(name, fits_root)
    rj = Path(fits_root or WQ.fit_dir(name).parent) / name / "read.json"
    r = WQ.read_json(rj) if rj.is_file() else {}
    rk = r.get(KEY) or {}
    if (rk.get("declared_in"), rk.get("fit"), rk.get("device"), rk.get("rows_cap")) != \
            (DECLARED, name, k.get("device"), READ_CAP):
        raise SystemExit(f"zr2ew: {name} has no R2EW read on the device it was fitted on")
    return k["device"]


def any_device(name, fits_root):
    """A fit's device by whose it is: R2EW's through fit_device, R2GW's zrckpb through zr2gw's, WQK's through zwqk's."""
    if name in FITS:
        return fit_device(name, fits_root)
    return (GW.fit_device if name in GW.FITS else K.fit_device)(name, fits_root)


def pair_device(names, base):
    """The one device the fits of names were fitted and read on."""
    fr = K.fits_dir(base)
    devs = {n: any_device(n, fr) for n in names}
    if len({json.dumps(d, sort_keys=True) for d in devs.values()}) != 1:
        raise SystemExit(f"zr2ew: {names[0]} and its bases ran on different devices or images: {devs}")
    return devs[names[0]]


def split_device(sp, base):
    """The one device R2EW's fit of sp, R2GW's zrckpb fit of sp and WQK's seed-0 zrckp fit of sp ran on."""
    return pair_device([FIT[sp], BESIDE_FIT[sp], BASE_FIT[sp]], base)


def take(src, base=None):
    """A kernel's output (src): each R2EW fit copied to outputs/screen/fits/<name>, its records and reads, never its
    weights; the session's gate.json and overlay.json to outputs/r2ew/sessions/<the gate's sha256[:12]>. A fit without
    R2EW's stamp and read on one device is left (named); a folder not R2EW's, or a file already there with other bytes,
    stops the take. Returns (copied, left)."""
    base, src = Path(base or WQ.ROOT), Path(src)
    fits = src / "fits"
    g, o = src / "wqk" / "gate.json", src / "wqk" / "overlay.json"
    if not (g.is_file() and o.is_file()):
        raise SystemExit(f"zr2ew take: {src.name} holds no wqk/gate.json and wqk/overlay.json (the session's records)")
    names = sorted(p.name for p in fits.iterdir() if p.is_dir()) if fits.is_dir() else []
    bad = [n for n in names if n not in FITS]
    if bad:
        raise SystemExit(f"zr2ew take: {bad} are not R2EW's fits")
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
            raise SystemExit(f"zr2ew take: {WQ.rel(t, base)} exists with other bytes")
    took = []
    for f, t in plan:
        if not t.exists():
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, t)
            took.append(WQ.rel(t, base))
    log(f"take: {len(done)} fits, {len(took)} files copied; the gate {sess.name}" + (f"; left: {left}" if left else ""))
    return took, left


def compare_all(base=None):
    """lean_screen's compare of each R2EW fit with WQK's seed-0 zrckp fit of its split (first: it decides) and with R2GW's
    zrckpb fit of it (beside), run from the repository's root by zwq's run_compare, every fit on one device; each stamped
    with R2EW's record and that device. A compare R2EW stamped from the same fits is kept."""
    base = Path(base or WQ.ROOT)
    import_r2()
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
                if f in K.FITS:
                    why = WQ.wq_why(f, base)
                    if why:
                        raise SystemExit(f"zr2ew: {why}")
            rc = WQ.run_compare(f"outputs/screen/fits/{new}", ",".join(f"outputs/screen/fits/{b}" for b in bases),
                                f"outputs/screen/{cname}")
        if rc or not js.is_file():
            raise SystemExit(f"zr2ew: the compare {cname} exited {rc}")
        rec = WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases), "device": dev,
                    "zr2ew_sha256": WQ.sha_src(__file__)}
        WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def cmp_rows(js, cname, dev=None):
    """An R2EW compare's rows {base fit: {dataset: row}}, after its checks; SystemExit naming what is wrong."""
    p = Path(js)
    if not p.is_file():
        raise SystemExit(f"outputs/screen/{cname}.json missing")
    rec = WQ.read_json(p)
    new, bases = COMPARES[cname]
    k = rec.get(KEY) or {}
    if (k.get("declared_in"), k.get("compare"), k.get("new"), k.get("bases")) != (DECLARED, cname, new, list(bases)):
        raise SystemExit(f"outputs/screen/{cname}.json is not R2EW's compare of {new} with {bases}")
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
    """zwqk.filed_gates on R2EW's sessions: ({gate sha256: its record}, problems), each a WQK PASS gate on the reproduced
    check and its session's overlay."""
    with WQ.patch(K, SESSIONS=SESSIONS):
        return K.filed_gates(base)


def fit_problems(base, gates, gw_gates):
    """Why an R2EW fit (or R2GW's zrckpb fit, the beside base) is not section 4's: its train exit, its six reads, the gates
    its train, wq block and read name, its overlay, R2's constants and EM term, and for zrckpbe its flag's declaration and
    the link statistics of its read."""
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
                out.append(f"{n}: its {what} names the gate {str(g)[:12]}, not a filed R2EW PASS gate")
        if k.get("gate") in gates and (gates[k["gate"]].get("records") or {}).get("overlay") != k.get("overlay"):
            out.append(f"{n}: its overlay is not its gate's")
        r2 = k.get("r2") or {}
        term = r2.get("term") or {}
        if (r2.get("declared_in"), r2.get("constants"), r2.get("params"), term.get("lam")) != \
                (R2_DECLARED, R2_CONST, list(R2_PARAMS), 1.0) or not term.get("questions_with_term"):
            out.append(f"{n}: R2's constants or EM term are not as declared")
        if (k.get("head") or {}).get("r2e") != R2E:
            out.append(f"{n}: its flag is not R2e's as declared")
        ls = rk.get("link_stats") or {}
        if set(ls) != set(K.SIX) or not all(all(isinstance(v.get(f), int) and v[f] > 0 for f in ("rows", "questions"))
                                            and all(f in v for f in LSTAT) for v in ls.values()):
            out.append(f"{n}: its read holds no link statistics on the six carves")
    mine = tuple(f"{n}:" for n in BESIDE_FIT.values())
    out += [p for p in GW.fit_problems(base, gw_gates) if p.startswith(mine)]
    return out


def restamp(pair, rcl, c):
    """R2EW's fields on the pair and re-call screen_pair and screen_recall wrote; a header on their markdown."""
    for p in (pair, rcl):
        js = Path(str(p) + ".json")
        rec = WQ.read_json(js)
        rec.update({"stage": STAGE, "declared_in": DECLARED, "arm": ARM, "against": AGAINST[c],
                    "null_kind": NULL_KIND[c], "deciding": DECIDING[c], "zr2ew_sha256": WQ.sha_src(__file__)})
        WQ.write_json(js, rec)
        md = Path(str(p) + ".md")
        if md.is_file():
            head_ = (f"R2EW ({DECLARED}): {ARM} against {AGAINST[c]} under WQ on the T4"
                     + (", deciding." if DECIDING[c] else ", beside, deciding nothing.") + "\n\n")
            md.write_text(head_ + md.read_text(encoding="utf-8"), encoding="utf-8")


def recall(base, dev, c):
    """screen_pair over R2EW's two compares (column CALLS[c] of each: WQK's seed-0 zrckp, or for zrckpbe-vs-zrckpb the
    split's zrckpb), then screen_recall under WQK's zrckp null. For zrckpbe-vs-zrckpb the null's seed-0 R@5 is R2GW's
    zrckpb read (the compare's zrckpb column); the floors stay zrckp's."""
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
    null, sources = K.wqk_null(BASE, base, dev, SR.floor_of)
    if c == "zrckpbe-vs-zrckpb":
        for sp in SPLITS:
            _r, kb = cmp_rows(scr / f"{CMP[sp]}.json", CMP[sp], dev)
            null[sp] = {ds: {**v, "seed0": kb[BESIDE_FIT[sp]][ds]["base"][0]} for ds, v in null[sp].items()}
            sources[f"R2GW zrckpb {sp}"] = f"outputs/screen/{CMP[sp]}.json"
    with WQ.patch(SR, load_null=lambda _files: (null, sources)):
        rec = SR.pair([], Path(str(pair) + ".json"), rcl)
    restamp(pair, rcl, c)
    return rec


def tag(r):
    return f"{r['split']} {r['dataset']}"


def f4(x, signed=False):
    return "n/a" if x is None else K.fmt(x, signed)


def verdict(base=None, out=None):
    """R2EW's verdict by the rule from zrckpbe's re-call against WQK's seed-0 zrckp under WQK's zrckp null (sections 4 and
    5). Beside, deciding nothing: zrckpbe against R2GW's zrckpb, the link statistics per read, R@5 per read, R2's term per
    fit and seconds and memory; outputs/r2ew/screen-verdict."""
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
    gw_gates, _gwp = GW.filed_gates(base)
    devs = {}
    for sp in SPLITS:
        try:
            devs[sp] = split_device(sp, base)
        except SystemExit as e:
            problems.append(str(e))
    problems += fit_problems(base, gates, gw_gates)
    keys = {json.dumps(d, sort_keys=True) for d in devs.values()}
    if len(keys) > 1:
        problems.append(f"R2EW's fits ran on {len(keys)} devices: {sorted(keys)}")
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
    rrows = rc["zrckpbe"]["rows"] if "zrckpbe" in rc and not problems else []
    gains = [tag(r) for r in rrows if r["call"] == "GAIN"]
    losses = [tag(r) for r in rrows if r["call"] == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = NEXT[v]
    wv = base / (K.VERDICT + ".json")
    rec = {"stage": STAGE, "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "base_arm": BASE,
           "fits": {ARM: dict(FIT), BESIDE_ARM: dict(BESIDE_FIT)}, "against": dict(BASE_FIT),
           "floors": FLOORS + ".json", "null_kind": NULL_KIND["zrckpbe"], "device": dev, "gates": sorted(gates),
           "next_form": nxt, "gains": gains, "losses": losses, "problems": problems,
           "calls": {tag(r): r["call"] for r in rrows}, "delta_r5": {tag(r): r["delta"] for r in rrows},
           "naming": NAMING, "wqk_verdict": WQ.sha_file(wv) if wv.is_file() else None,
           "zwqk_sha256": WQ.sha_src(K.__file__), "zr2ew_sha256": WQ.sha_src(__file__), "utc": WQ.utc()}
    md = [f"# R2EW, R2e (the MLP half) under WQ on Kaggle's T4: zrckpbe **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zrckpbe** ({WHAT}: R2's head on the bridges that zlink's edges link to the row) against WQK's seed-0 zrckp "
          f"on the T4, re-called under WQK's zrckp seed null: {len(gains)} GAIN ({', '.join(gains) or 'none'}), "
          f"{len(losses)} LOSS ({', '.join(losses) or 'none'})" + (f"; problems: {'; '.join(problems)}" if problems else "")
          + ".", f"- **Next form** (fixed in section 7 before any number): {nxt}.", f"- Device: {json.dumps(dev)}.",
          f"- Gates: {', '.join(g[:12] for g in sorted(gates)) or 'none'}.", f"- Naming: {NAMING}."]
    beside = {}
    if v != "INCOMPLETE":
        fr, scr = K.fits_dir(base), K.screen_dir(base)
        zspk = {}
        for sp in SPLITS:
            _r, zr = K.cmp_rows(scr / f"{ZSPK_CMP[sp]}.json", ZSPK_CMP[sp], dev)
            zspk[sp] = {ds: zr[ds]["base"][0] for ds in WQ.EVAL_ORDER}
        side = {x["split"] + " " + x["dataset"]: x["call"] for x in rc["zrckpbe-vs-zrckpb"]["rows"]}
        by = {"gains": [k_ for k_, x in side.items() if x == "GAIN"], "losses": [k_ for k_, x in side.items() if x == "LOSS"]}
        lstat = {sp: ((WQ.read_json(fr / FIT[sp] / "read.json").get(KEY) or {}).get("link_stats") or {}) for sp in SPLITS}
        r5 = {}
        for r in rrows:
            sp, ds = r["split"], r["dataset"]
            r5[tag(r)] = {"read": r["read"], "rrf": r["rrf"], "zspk": zspk[sp][ds], "zrckp": r["base"],
                          "zrckpb": rows[CMP[sp]][BESIDE_FIT[sp]][ds]["base"][0], "zrckpbe": r["new"],
                          "floor": r["floor"], "call": r["call"], "zrckpbe_vs_zrckpb": side[tag(r)],
                          "links": lstat[sp].get(f"{ds}=s1eval")}
        term = {}
        for sp in SPLITS:
            t = ((WQ.read_json(fr / FIT[sp] / "screen.json").get(KEY) or {}).get("r2") or {}).get("term") or {}
            ep = t.get("epochs") or []
            term[FIT[sp]] = {"questions": t.get("questions"), "questions_with_term": t.get("questions_with_term"),
                             "share_with_term": t.get("share_with_term"),
                             "br_nll_first": ep[0].get("br_nll_mean") if ep else None,
                             "br_nll_last": ep[-1].get("br_nll_mean") if ep else None}
        tm = {sp: {"zrckpbe": K.timing(FIT[sp], fr, TIMING_FILE), "zrckpb": K.timing(BESIDE_FIT[sp], fr, GW.TIMING_FILE),
                   "zrckp": K.timing(BASE_FIT[sp], fr, "wqk_timing.json")} for sp in SPLITS}
        rp = Path(str(K.at(base, RECALL["zrckpbe"])) + ".json")
        beside = {"calls": {"zrckpbe-vs-zrckpb": by}, "r5": r5, "terms": term, "timing": tm,
                  "pair": PAIR["zrckpbe"] + ".json", "recall": RECALL["zrckpbe"] + ".json", "recall_sha256": WQ.sha_file(rp),
                  "pairs": {c: PAIR[c] + ".json" for c in CALLS}, "recalls": {c: RECALL[c] + ".json" for c in CALLS}}
        md += ["", "Beside, deciding nothing.", "",
               f"- zrckpbe against R2GW's zrckpb (the null's seed-0 R@5 is R2GW's zrckpb read): {len(by['gains'])} GAIN "
               f"({', '.join(by['gains']) or 'none'}), {len(by['losses'])} LOSS ({', '.join(by['losses']) or 'none'}).",
               "", "R@5 per read on the T4, and zrckpbe's link statistics on the read (share of rows with a linked bridge, "
                   "linked bridges per row, share of questions with a linked row):", "",
               "| read | rrf | zspk | zrckp | zrckpb | zrckpbe | floor | call | vs zrckpb | rows linked | flags per row "
               "| questions linked |", "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|"]
        for k_, t in r5.items():
            ls = t["links"] or {}
            md.append(f"| {k_} ({t['read']}) | {f4(t['rrf'])} | {f4(t['zspk'])} | {f4(t['zrckp'])} | {f4(t['zrckpb'])} "
                      f"| {f4(t['zrckpbe'])} | {f4(t['floor'])} | {t['call']} | {t['zrckpbe_vs_zrckpb']} "
                      f"| {f4(ls.get('share_rows_linked'))} | {f4(ls.get('flags_per_row'))} "
                      f"| {f4(ls.get('share_questions_linked'))} |")
        md += ["", "R2's EM term on each fit (the share of training questions it reached, L_br at the first and the last "
                   "epoch):", "", "| fit | questions | with the term | share | L_br first | L_br last |",
               "|---|---:|---:|---:|---:|---:|"]
        for n, t in term.items():
            md.append(f"| {n} | {t['questions']} | {t['questions_with_term']} | {f4(t['share_with_term'])} "
                      f"| {f4(t['br_nll_first'])} | {f4(t['br_nll_last'])} |")

        def s0(x):
            return f"{x:.0f}" if isinstance(x, (int, float)) else "n/a"
        md += ["", "Seconds and memory: zrckpbe, and in brackets R2GW's zrckpb and WQK's seed-0 zrckp of the same split, "
                   "all on the T4:", "", "| split | train s | read s | peak GPU GB, train / read | peak RAM GB |",
               "|---|---:|---:|---|---:|"]
        for sp in SPLITS:
            a_, b_, c_ = tm[sp]["zrckpbe"], tm[sp]["zrckpb"], tm[sp]["zrckp"]
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
    """What R2EW has under the real folders: its fits, its compares and everything under outputs/r2ew."""
    w = WQ.ROOT / NEW
    return (sorted(p.name for p in K.fits_dir(WQ.ROOT).glob("scr-r2ew-*")),
            sorted(p.name for p in K.screen_dir(WQ.ROOT).glob("scr-r2ew-*")),
            sorted((p.relative_to(w).as_posix(), p.stat().st_mtime_ns) for p in w.rglob("*")) if w.is_dir() else [])


# ── selftest ─────────────────────────────────────────────────────────────────

FAKE_COUNTS = {"rows": 1000, "rows_linked": 300, "flags": 450, "questions": 100, "questions_linked": 80, "passes": 2.0,
               "rows_in_carve": 500, "questions_in_carve": 50}


def fake_fit(fits_root, name, gate_sha, overlay_sha, dev=None, read_=True, rows_cap=READ_CAP, r2=None, r2e="declared",
             stats=True):
    """An R2EW fit as the kernel leaves it: screen.json (WQK's wq block, R2EW's block) and read.json with their stamps,
    train.json, r2ew_timing.json, a read and models.pt."""
    sp = FITS[name]
    d = Path(fits_root) / name
    (d / "reads").mkdir(parents=True, exist_ok=True)
    dev = dev or K.device_of(K.TOY_ENV)
    r2 = r2 or {"declared_in": R2_DECLARED, "constants": dict(R2_CONST), "params": list(R2_PARAMS),
                "term": {"lam": 1.0, "questions": 100, "questions_with_term": 90, "share_with_term": 0.9,
                         "epochs": [{"br_nll_mean": 4.6}, {"br_nll_mean": 3.9}]}}
    k = {"declared_in": DECLARED, "fit": name, "arm": ARM, "seed": SEED, "split": sp, "device": dev, "gate": gate_sha,
         "overlay": overlay_sha, "r2": r2, "head": {"r2e": R2E if r2e == "declared" else r2e}}
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


def flags_by_hand(q_ne, e_u, e_v, n, br):
    """The same flags as a loop over the edges, the first reading of section 3 and nothing else."""
    out = set()
    pos = 0
    for q in range(len(n)):
        base_ = sum(int(x) for x in n[:q])
        for _ in range(int(q_ne[q])):
            u, v = int(e_u[pos]), int(e_v[pos])
            pos += 1
            for k in range(NB):
                if int(br[q][k]) >= 0 and int(br[q][k]) == u:
                    out.add((base_ + v, k))
    return out


def feats_ref(m, r2, nq, torch, np):
    """[a, m, f] of each row by hand (float64): over the slots that are present, not the row, and linked."""
    R2 = _R2["R2"]
    S_, M = R2.r2_logits(m, r2["eq"], r2["p"], r2["bri"], nq)
    S_, Mk = S_.double().numpy(), (M & r2["lk"]).numpy()
    x = np.zeros((S_.shape[0], 3))
    for v in range(S_.shape[0]):
        ks = [k for k in range(NB) if Mk[v, k]]
        if not ks:
            continue
        pi = np.array([1.0 / (R2.PRIOR_C + k + 1) for k in ks])
        z = np.log(pi / pi.sum()) + S_[v, ks]
        x[v] = [z.max() + np.log(np.exp(z - z.max()).sum()), S_[v, ks].max(), 1.0]
    return x


def selftest():
    import inspect
    import tempfile
    import types
    t0 = time.time()
    before = real_state()
    stops, TOY = K.stops, K.TOY_ENV
    me = sys.modules[__name__]
    R = import_r2()
    R2, np, torch = R["R2"], R["np"], R["torch"]
    EdgeMix, ZRCKPBE, ECarve = R["EdgeMix"], R["ZRCKPBE"], R["ECarve"]
    link_flags, edge_carve = R["link_flags"], R["edge_carve"]
    nn = torch.nn
    # 1. the names: two fits, two compares each with WQK's seed-0 zrckp and R2GW's zrckpb, the modules the stamps hash,
    #    R2's text for R2e, R2t's naming text
    assert FIT == {"WQ-musique": "scr-r2ew-zrckpbe", "WQ-hotpotqa": "scr-r2ew-zrckpbe-hp"} and len(FITS) == 2
    assert BESIDE_FIT == {"WQ-musique": "scr-r2gw-zrckpb", "WQ-hotpotqa": "scr-r2gw-zrckpb-hp"}
    assert COMPARES == {"scr-r2ew-zrckpbe-vs-zrckp": ("scr-r2ew-zrckpbe", ["scr-wqk-zrckp-s0", "scr-r2gw-zrckpb"]),
                        "scr-r2ew-zrckpbe-vs-zrckp-hp": ("scr-r2ew-zrckpbe-hp",
                                                         ["scr-wqk-zrckp-s0-hp", "scr-r2gw-zrckpb-hp"])}
    assert ZSPK_CMP["WQ-musique"] == "scr-wqk-zspk-s1" and BASE_FIT["WQ-hotpotqa"] == "scr-wqk-zrckp-s0-hp"
    assert all((HERE / f"{m}.py").is_file() for m in CODE) and len(set(CODE)) == len(CODE)
    assert {"zlink", "zr2br", "zr2gw", "zr2ew"} <= set(CODE) and set(NEXT) == {"ADOPT_FOR_FULL_RUN", "NOT_ADOPTED", "INCOMPLETE"}
    assert NAMING_R2T == K.source_const(HERE / "zr2t.py", "NOT_BESIDE")["zrckpb"] and WHAT in NAMING
    assert R2E_TEXT == R2.NEXT["NO_GAIN"] and R2E_TEXT in NEXT["NOT_ADOPTED"] and "GNN half" in NEXT["NOT_ADOPTED"]
    assert R2.constants() == R2_CONST and R2.n_params() == R2_N == 59537 and R2.K == NB == 5
    assert R2.S.ARMS[ARM] == (ZRCKPBE, ECarve) and R2.S.ARMS[BESIDE_ARM] == R2.ARMS[BESIDE_ARM]
    assert ZRCKPBE.__mro__[1:4] == (EdgeMix, R2.ZRCKPB, R2.R2Mix) and ECarve.__mro__[1] is R2.R2KTChainCarveZRC
    assert ECarve.__name__ == "ER2KTChainCarveZRC" and R2E["params_added"] == 0
    # 2. the flags by hand from the links arrays alone: an edge in one direction, two families, a bridge-to-bridge edge,
    #    an absent slot, a pool of one, an empty pool; the function takes no score
    assert list(inspect.signature(flags_np).parameters) == ["q_ne", "e_u", "e_v", "n", "br"]
    assert list(inspect.signature(link_flags).parameters) == ["lk", "bri", "nq", "N"]
    n = np.array([6, 3, 1, 0])
    br = np.array([[2, 0, 5, -1, -1], [1, 0, 2, -1, -1], [0, -1, -1, -1, -1], [-1] * 5])
    ed = [[(2, 3), (3, 2), (0, 1), (1, 0), (2, 5), (5, 2), (0, 1), (1, 0)], [(1, 0)], [], []]
    q_ne = np.array([len(e) for e in ed])
    e_u = np.array([u for e in ed for u, _v in e], np.int16)
    e_v = np.array([v for e in ed for _u, v in e], np.int16)
    fl = flags_np(q_ne, e_u, e_v, n, br)
    want = {(3, 0), (1, 1), (5, 0), (2, 2), (6, 0)}
    assert fl.dtype == bool and fl.shape == (10, NB) and {(r, k) for r, k in zip(*np.nonzero(fl))} == want
    assert flags_by_hand(q_ne, e_u, e_v, n, br) == want and not fl[0].any() and not fl[4].any() and not fl[9].any()
    own = {(int(np.r_[0, np.cumsum(n)][q] + br[q][k]), k) for q in range(4) for k in range(NB) if br[q][k] >= 0}
    assert not (want & own)                                        # a bridge is never flagged for its own slot
    bad = e_v.copy()
    bad[0] = 7
    stops(flags_np, q_ne, e_u, bad, n, br, has="outside its question's pool")
    stops(flags_np, q_ne[:-1], e_u, e_v, n, br, has="do not fit")
    assert not flags_np(np.zeros(0, np.int64), e_u[:0], e_v[:0], np.zeros(0, np.int64), br[:0]).size
    # 3. the same flags on a batch, on the device (link_flags): any question order, equal to the whole-carve flags' rows
    off = np.r_[0, np.cumsum(n)]
    e_off = np.r_[0, np.cumsum(q_ne)]

    def batch_of(qs):
        qs = np.asarray(qs)
        cnt = n[qs]
        seg = np.cumsum(cnt) - cnt
        ne = q_ne[qs]
        idx = np.repeat(e_off[qs] - (np.cumsum(ne) - ne), ne) + np.arange(int(ne.sum()))
        eq = np.repeat(np.arange(qs.size), ne)
        lk = {"u": torch.from_numpy(seg[eq] + e_u[idx].astype(np.int64)), "v": torch.from_numpy(seg[eq] + e_v[idx].astype(np.int64))}
        bri = torch.from_numpy(np.where(br[qs] >= 0, br[qs] + seg[:, None], -1))
        return lk, bri, torch.from_numpy(np.repeat(np.arange(qs.size), cnt)), int(cnt.sum())

    for qs in ([0, 1, 2, 3], [1, 0, 3, 2], [1, 1, 0], [3], [2, 0]):
        lk_, bri_, nq_, N_ = batch_of(qs)
        got = link_flags(lk_, bri_, nq_, N_).numpy()
        ref = np.concatenate([fl[off[q]:off[q + 1]] for q in qs]) if N_ else np.zeros((0, NB), bool)
        assert got.dtype == bool and np.array_equal(got, ref), qs
    # 4. the model on a toy base: the mask is M and the flags; zrckpbe starts as the base; with every flag 1 it is zrckpb;
    #    a flag off removes the slot from the prior, [a, m, f] and R2's term; a row with no linked bridge has x = 0; no new
    #    parameter; the flags are fixed inputs, the same for any weights
    class ToyBase(nn.Module):
        def __init__(self, blocks_, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([0.5, -0.25]))

        def forward(self, feats, keep, nq, B, base_z):
            return feats["x"] @ self.w + base_z

    class ToyB(R2.R2Mix, ToyBase):
        pass

    class ToyBE(EdgeMix, ToyB):
        pass

    feats, nq, nb, base_z = GW.toy_batch(R2, torch)
    r2 = feats[R2.R2_KEY]
    N = int(nq.numel())
    g = torch.Generator().manual_seed(7)
    lk = torch.rand(N, NB, generator=g) < 0.5
    lk[0] = False
    lk[5] = torch.tensor([True, False, False, False, False])
    r2["lk"] = lk
    ones = torch.ones(N, NB, dtype=torch.bool)
    mb, me_ = ToyB(None, None), ToyBE(None, None)
    scale = {"r2_U": 1.0, "r2_V": 1.0, "r2_W1": 0.3 / R2.R_DIM ** 0.5, "r2_W2": 0.3 / R2.R_DIM ** 0.5, "r2_o": 0.3,
             "r2_w1": 1.0, "r2_b1": 0.3, "r2_w2": 0.5, "r2_b2": 0.1}
    s0 = ToyB(None, None)
    assert set(ToyBE(None, None).state_dict()) == set(s0.state_dict()) == set(mb.state_dict())
    assert sum(p.numel() for k_, p in me_.named_parameters() if k_.startswith("r2")) == R2_N
    for mode in ("eval", "train"):
        getattr(mb, mode)(), getattr(me_, mode)()
        assert torch.equal(mb(feats, None, nq, nb, base_z), me_(feats, None, nq, nb, base_z)), mode   # last layer is 0
    gen = torch.Generator().manual_seed(8)
    with torch.no_grad():
        for k_ in R2_PARAMS:
            getattr(mb, k_).copy_(torch.randn(getattr(mb, k_).shape, generator=gen) * scale[k_])
    me_.load_state_dict(mb.state_dict())
    mb.eval(), me_.eval()
    with torch.no_grad():
        xb, xe = mb.r2_feats(r2, nq), me_.r2_feats(r2, nq)
        assert float(mb.r2_head(xb).abs().max()) > 0 and not torch.equal(xb, xe)
        ref = feats_ref(me_, r2, nq, torch, np)
        assert np.abs(xe.double().numpy() - ref).max() < 1e-5 and (xe[0] == 0).all() and (xe[lk.sum(1) == 0] == 0).all()
        assert (xe[5, 2] == 1) and abs(float(xe[5, 0]) - float(xe[5, 1])) < 1e-6          # one linked slot: a = m
        y_all = me_(dict(feats, **{R2.R2_KEY: {**r2, "lk": ones}}), None, nq, nb, base_z)
        assert torch.equal(y_all, mb(feats, None, nq, nb, base_z))                       # every flag 1: zrckpb
        none = me_(dict(feats, **{R2.R2_KEY: {**r2, "lk": torch.zeros_like(lk)}}), None, nq, nb, base_z)
        assert torch.equal(none, feats["x"] @ me_.w + base_z + me_.r2_head(torch.zeros(N, 3)))
        s_ = feats["x"] @ me_.w + base_z
        assert torch.allclose(me_(feats, None, nq, nb, base_z), s_ + me_.r2_head(xe)) and not torch.equal(y_all, me_(feats, None, nq, nb, base_z))
        w2 = ToyBE(None, None)
        w2.load_state_dict({k_: torch.randn_like(v) for k_, v in mb.state_dict().items()})
        l2 = {}

        class Probe(ToyBE):
            def r2_parts(self, r2_, nq_):
                l2[id(self)] = r2_["lk"].clone()
                return super().r2_parts(r2_, nq_)
        pa, pb = Probe(None, None), Probe(None, None)
        pb.load_state_dict({k_: torch.randn_like(v) for k_, v in pa.state_dict().items()})
        pa(feats, None, nq, nb, base_z), pb(feats, None, nq, nb, base_z)
        assert len(l2) == 2 and torch.equal(l2[id(pa)], lk) and torch.equal(l2[id(pb)], lk) and l2[id(pa)].dtype == torch.bool
    stops(me_, {"x": feats["x"], R2.R2_KEY: {k_: v for k_, v in r2.items() if k_ != "lk"}}, None, nq, nb, base_z,
          has="holds no link flags")
    assert torch.equal(me_({"x": feats["x"]}, None, nq, nb, base_z), s_)
    # 5. R2's EM term on the linked slots: the kept mask is M and the flags (a subset of R2's), a batch kept twice is
    #    refused, a gold row with no linked slot has no term, the gradient reaches R2's last layer at the start only
    me0 = ToyBE(None, None)
    with R2.r2_em():
        mb.train(), me_.train()
        R2.clear_stash()
        mb(feats, None, nq, nb, base_z)
        Mb = R2.STASH["M"].clone()
        R2.clear_stash()
        me_(feats, None, nq, nb, base_z)
        S_k, M_k, lpt_k = R2.STASH["S"], R2.STASH["M"], R2.STASH["lpt"]
        assert torch.equal(M_k, Mb & lk) and (M_k.sum() < Mb.sum()) and R2.STASH["N"] == N
        assert torch.equal(lpt_k, R2.prior_logp(M_k, S_k.dtype))
        stops(me_, feats, None, nq, nb, base_z, has="kept and never scored")
        R2.clear_stash()
        gold_off = torch.zeros(N)
        gold_off[0] = 1.0                                               # row 0 has no linked slot
        _l, nqs, nqg = R2.br_nll(S_k, M_k, lpt_k, nq, nb, gold_off)
        assert (nqs, nqg) == (0, 1)
        gold_on = torch.zeros(N)
        gold_on[5] = 1.0
        nll, nqs, nqg = R2.br_nll(S_k, M_k, lpt_k, nq, nb, gold_on)
        assert nqs == 1 and torch.isfinite(nll) and float(nll) >= 0
    assert R2.STASH["S"] is None
    me0.train()
    me0(feats, None, nq, nb, base_z).sum().backward()
    assert int(torch.count_nonzero(me0.r2_w2.grad)) > 0
    stops(term_ok, {"batches": 7, "batches_with_term": 0}, "scr-r2ew-zrckpbe", has="carried R2's term (7 batches)")
    term_ok({"batches": 7, "batches_with_term": 3}, "scr-r2ew-zrckpbe")
    # 6. the carve on a toy built by zlink's own build and read by its own load_links (no model scores anything): the
    #    batch's flags are the by-hand loop over the edge files for any question order; a links record that does not
    #    match its part is refused; the link statistics of a read are the flags' counts
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
        c0 = ch_cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
        r2_cls = R2.r2_carve(Z3.kt_carve(ch_cls, tmp / "kt"))
        ec_cls = edge_carve(r2_cls, tmp / "lk")
        ec = ec_cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        cm = r2_cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert ec_cls.__mro__[1] is r2_cls and ec_cls.__name__ == f"E{r2_cls.__name__}" and np.array_equal(ec.r2_br, cm.r2_br)
        assert ec.nbytes() == cm.nbytes() + sum(int(ec.links[k_].nbytes) for k_ in ("q_ne", "e_u", "e_v", "e_fam", "e_off"))
        parts = sorted((tmp / "lk" / "metaqa" / "toy").glob("part_*"))
        raw = {k_: np.concatenate([np.load(p / f"{k_}.npy") for p in parts]) for k_ in ("q_ne", "e_u", "e_v")}
        assert all(np.array_equal(raw[k_], ec.links[k_]) for k_ in raw)
        whole = flags_np(ec.links["q_ne"], ec.links["e_u"], ec.links["e_v"], ec.n_np, ec.r2_br)
        offs = np.r_[0, np.cumsum(ec.n_np)]
        ref_set = flags_by_hand(raw["q_ne"], raw["e_u"], raw["e_v"], ec.n_np, ec.r2_br)
        assert {(r, k_) for r, k_ in zip(*np.nonzero(whole))} == ref_set and whole.any() and not whole.all(), whole.sum()
        blocks = ["rank", "SEMB"]
        rows_ = ec.rows
        for qs in (np.r_[np.arange(rows_)[::-1][:7], np.arange(rows_)[:5]], np.arange(rows_), np.array([3]),
                   np.array([rows_ - 1, 0, 2])):
            qs = qs.astype(np.int64)
            feats_e, nq_e, bz_e, gold_e = ec.batch(qs, blocks)
            feats_c, nq_c, bz_c, gold_c = cm.batch(qs, blocks)
            r2e = feats_e[R2.R2_KEY]
            assert r2e["lk"].dtype == torch.bool and not r2e["lk"].requires_grad and torch.equal(nq_e, nq_c)
            assert all(torch.equal(r2e[k_], feats_c[R2.R2_KEY][k_]) for k_ in ("bri", "eq", "p"))
            assert set(r2e) == set(feats_c[R2.R2_KEY]) | {"lk"} and torch.equal(bz_e, bz_c) and torch.equal(gold_e, gold_c)
            ref = np.concatenate([whole[offs[q]:offs[q + 1]] for q in qs])
            assert np.array_equal(r2e["lk"].numpy(), ref), qs
        ec2 = ec_cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")           # LK_ROOT of the class
        with WQ.patch(ZL.LinkCarveBase, LK_ROOT=tmp / "lk"):
            ec3 = edge_carve(r2_cls)("metaqa", "toy", "2wiki", "cpu", tmp / "cache")      # LK_ROOT of zlink's base
        assert ec2.links["records"] == ec3.links["records"] == ec.links["records"]
        shutil.copytree(tmp / "lk", tmp / "lk2")
        pt = sorted((tmp / "lk2" / "metaqa" / "toy").glob("part_*"))[0] / "e_u.npy"
        eu = np.load(pt)
        eu[0] = (eu[0] + 1) % 2
        np.save(pt, eu)
        stops(edge_carve(r2_cls, tmp / "lk2"), "metaqa", "toy", "2wiki", "cpu", tmp / "cache", has="sha256 is not its record's")
        # the real model class on the real batch: at the start it is zrckp, with every flag 1 it is zrckpb
        torch.manual_seed(0)
        m_base = R2.Z3P.ZRCKP(blocks, ec.widths, 16, dropout=0.1, seed=0)
        torch.manual_seed(0)
        m_b = R2.ZRCKPB(blocks, ec.widths, 16, dropout=0.1, seed=0)
        torch.manual_seed(0)
        m_e = ZRCKPBE(blocks, ec.widths, 16, dropout=0.1, seed=0)
        assert set(m_e.state_dict()) == set(m_b.state_dict()) and sum(m_e.state_dict()[k_].numel() for k_ in R2_PARAMS) == R2_N
        qs = np.r_[np.arange(rows_)[::-1][:7], np.arange(rows_)[:5]].astype(np.int64)
        fe, nqe, bze, _gd = ec.batch(qs, blocks)
        B_ = qs.size
        keep = torch.ones(B_, len(blocks))
        for mode in (False, True):
            for m_ in (m_base, m_b, m_e):
                m_.train(mode)
            with torch.no_grad():
                torch.manual_seed(5)
                y0 = m_base(fe, keep, nqe, B_, bze)
                torch.manual_seed(5)
                assert torch.equal(y0, m_e(fe, keep, nqe, B_, bze)), mode
        gen = torch.Generator().manual_seed(65)
        with torch.no_grad():
            for k_ in R2_PARAMS:
                getattr(m_b, k_).copy_(torch.randn(getattr(m_b, k_).shape, generator=gen) * scale[k_])
            m_e.load_state_dict(m_b.state_dict())
            m_b.eval(), m_e.eval()
            r2x = fe[R2.R2_KEY]
            ones_f = {**fe, R2.R2_KEY: {**r2x, "lk": torch.ones_like(r2x["lk"])}}
            assert torch.equal(m_e(ones_f, keep, nqe, B_, bze), m_b(fe, keep, nqe, B_, bze))
            assert not torch.equal(m_e(fe, keep, nqe, B_, bze), m_b(fe, keep, nqe, B_, bze))
            assert np.abs(m_e.r2_feats(r2x, nqe).double().numpy() - feats_ref(m_e, r2x, nqe, torch, np)).max() < 1e-4
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
        assert st["flags"] == 2 * int(whole.sum()) and st["rows_linked"] == 2 * int(whole.any(1).sum())
        qlinked = sum(bool(whole[offs[q]:offs[q + 1]].any()) for q in range(rows_))
        assert st["questions_linked"] == 2 * qlinked and ec.r2e_stats is None
        ls = link_stats({k_: st[k_] for k_ in LSTAT})
        assert ls["share_rows_linked"] == round(whole.any(1).mean(), 6) and ls["flags_per_row"] == round(whole.sum() / len(whole), 6)
        STATS.clear()

        def boom_read(c, models, *a, **kw):
            c.batch(allq, blocks)
            raise RuntimeError("inside")

        try:
            stats_reader(boom_read)(ec, [])
            raise AssertionError("did not raise")
        except RuntimeError:
            pass
        assert ec.r2e_stats is None and not STATS
        stops(stats_reader(orig_read), cm, [], has="is not read on R2EW's carve")
        stops(stats_reader(lambda c, m_, *a, **k_: None), ec, [], has="scored no batch")
        lg = types.SimpleNamespace(read_carve=orig_read)
        with stats_reads(lg):
            assert lg.read_carve is not orig_read
        assert lg.read_carve is orig_read and ec.r2e_stats is None
    # 7. inside zwq.on_wq: zr2br trains on zwq's modules (zlink's too); the arm swap is zp1c3p's for the block, restored after
    with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W:
        GW.same_modules(R2, W)
        other = types.SimpleNamespace(Z3=W.Z3, Z2=W.Z2, LG=W.LG, S=W.S, S2=W.S2, LC=W.LC, Z3P=W.Z3P,
                                      RM=types.SimpleNamespace(), ZL=W.Z3.ZL)
        saved = (W.Z3P.ARM, dict(W.Z3P.ARMS))
        with GW.on_arm(W, ARM) as W2:
            assert W2 is W and W.Z3P.ARM == ARM and W.Z3P.ARMS == {ARM: R2.S.ARMS[ARM]}
        assert (W.Z3P.ARM, dict(W.Z3P.ARMS)) == saved
        assert other.ZL is R2.ZL
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        B = td / "repo"
        # 8. where, the read cap, WQK's verdict: train, read and fit refuse before reading any data
        stops(arm_split, "scr-wqk-zrckp-s0", has="not one of R2EW's fits")
        stops(arm_split, "scr-r2gw-zrckpb", has="not one of R2EW's fits")
        stops(train, "scr-r2-zrckpb", TOY, has="not one of R2EW's fits")
        for bad in ({"where": "elsewhere"}, {"gpu": "Tesla P100-PCIE-16GB"}, {"image": "kaggle-gpu:other"}):
            stops(where_ok, {**TOY, **bad}, "train", has="R2EW trains in a Kaggle kernel")
            stops(train, FIT["WQ-musique"], {**TOY, **bad}, has="R2EW trains in a Kaggle kernel")
            stops(read, FIT["WQ-hotpotqa"], {**TOY, **bad}, has="R2EW reads in a Kaggle kernel")
        where_ok(TOY, "read")
        cap_ok(READ_CAP)
        for cap in (0, None, 32768):
            stops(cap_ok, cap, has="--rows-cap 65536")
            stops(read, FIT["WQ-musique"], TOY, rows_cap=cap, has="--rows-cap 65536")
            stops(fit, FIT["WQ-musique"], rows_cap=cap, has="--rows-cap 65536")
        stops(fit, FIT["WQ-hotpotqa"], has="R2EW trains in a Kaggle kernel")                 # the laptop is not a T4
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
        # 9. the stamps of both fits (WQK's wq block at the toy root, R2EW's block), fit_stamp, the read's stamp with the
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
            stops(fit_stamp, nm, fr7, has="no R2EW stamp")
            stamp_fit(nm, TOY, R2, summ, B7, {"kt_records": {"webqsp/fit": "k"}, "zlink_records": {"p": "z"}})
            s7, k7 = fit_stamp(nm, fr7)
            assert (k7["declared_in"], k7["device"], k7["against"], k7["beside"], k7["what"]) == (
                DECLARED, K.device_of(TOY), BASE_FIT[sp_], BESIDE_FIT[sp_], WHAT)
            assert (s7["wq"]["declared_in"], s7["wq"]["gate"]) == (K.DECLARED, WQ.sha_file(B7 / K.RECORDS["GATE"]))
            assert k7["gate"] == s7["wq"]["gate"] and k7["wqk_verdict"] == WQ.sha_file(B7 / (K.VERDICT + ".json"))
            assert (k7["r2"]["constants"], k7["r2"]["params"], k7["r2"]["term"]) == (R2_CONST, list(R2_PARAMS), summ)
            assert k7["head"]["r2e"] == R2E and k7["head"]["read_rows"] == READ_CAP and k7["zlink_records"] == {"p": "z"}
            assert s7["zr2br_sha256"] == WQ.sha_src(HERE / "zr2br.py") and s7["zr2ew_sha256"] == WQ.sha_src(__file__)
            assert s7["zr2gw_sha256"] == WQ.sha_src(HERE / "zr2gw.py") and set(k7["code"]) == set(CODE)
        stops(fit_stamp, FIT["WQ-musique"], fr7 / "none", has="no R2EW stamp")
        n7 = FIT["WQ-hotpotqa"]
        WQ.write_json(WQ.fit_dir(n7, B7) / "read.json", {"seconds": 12.0, "carves": {c: {} for c in K.SIX}})
        stops(fit_device, n7, fr7, has="has no R2EW read on the device it was fitted on")
        r7 = stamp_read(n7, TOY, {"webqsp/fit": "k"}, READ_CAP, {"metaqa=s1eval": dict(FAKE_COUNTS)}, B7, {"p": "z"})[KEY]
        assert r7["link_stats"] == {"metaqa=s1eval": link_stats(FAKE_COUNTS)} and r7["zlink_records"] == {"p": "z"}
        assert link_stats(FAKE_COUNTS)["share_rows_linked"] == 0.3 and link_stats(FAKE_COUNTS)["flags_per_row"] == 0.45
        assert link_stats(FAKE_COUNTS)["share_questions_linked"] == 0.8 and r7["rows_cap"] == READ_CAP
        assert fit_device(n7, fr7) == K.device_of(TOY)
        stamp_read(n7, TOY, {"webqsp/fit": "k"}, 32768, {}, B7)
        stops(fit_device, n7, fr7, has="has no R2EW read on the device it was fitted on")
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
        # 10. take, compare-all and the verdict on a toy of WQK's base set (zwqk's toy, its fits, compares and floors), an
        #     R2GW session holding the two zrckpb fits and an R2EW session: ADOPT_FOR_FULL_RUN, NOT_ADOPTED both ways, and
        #     INCOMPLETE for each missing or wrong piece
        K.toy_build(B)
        gw, ow = K.fake_session(td / "wqk_out", B, tag="w")
        for nm in K.FITS:
            K.fake_fit(K.fits_dir(B), nm, gw, ow)

        def wqk_run(new, base_fit, out):                # zwqk's selftest's: WQK's compares on the toy
            new, b0, c_ = Path(new).name, Path(base_fit).name, Path(out).name
            arm, seed, sp = K.FITS[new]
            dd = {ds: (0.002, -0.002)[seed - 1] for ds in WQ.EVAL_ORDER} if arm != "zg6" else {}
            if (arm, sp) == ("zrckp", "WQ-musique"):
                dd["musique"] = (0.02, -0.01)[seed - 1]
            K.fake_compare(B, c_, new, b0, dd, base_r5=0.59 if arm == "zrckp" else 0.6)
            return 0

        why = WQ.wq_why
        with WQ.patch(WQ, run_compare=wqk_run, wq_why=lambda name, base=B: why(name, base)):
            assert K.compare_all(B) == 0
        assert K.floors(B) == 0
        WQ.write_json(B / (K.VERDICT + ".json"), {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
        out_g = td / "r2gw_out"
        gg, og = K.fake_session(out_g, B, tag="g")
        for sp in SPLITS:
            GW.fake_fit(out_g / "fits", BESIDE_FIT[sp], gg, og)
        assert len(GW.take(out_g, B)[0]) == 2 * 5 + 2
        out_e = td / "r2ew_out"
        ge, oe = K.fake_session(out_e, B, tag="e")
        names = list(FITS)
        fake_fit(out_e / "fits", names[0], ge, oe)
        fake_fit(out_e / "fits", names[1], ge, oe, read_=False)
        took, left = take(out_e, B)
        assert len(took) == 5 + 2 and len(left) == 1 and "has no R2EW read" in left[0], (took, left)
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
        stops(take, out_e, B, has="are not R2EW's fits")
        (out_e / "fits" / BASE_FIT["WQ-musique"]).rmdir()
        (out_e / "fits" / BESIDE_FIT["WQ-musique"]).mkdir()
        stops(take, out_e, B, has="are not R2EW's fits")                      # R2GW's fits are R2GW's to take
        (out_e / "fits" / BESIDE_FIT["WQ-musique"]).rmdir()
        stops(take, td / "wqk_out" / "wqk", B, has="holds no wqk/gate.json")
        cur = {}

        def r2ew_run(new, base_fit, out):               # lean_screen's compare with two bases, in its record's form
            import lean_screen as S
            n_, c_ = Path(new).name, Path(out).name
            sp = FITS[n_]
            bs = [Path(b).name for b in base_fit.split(",")]
            assert bs == COMPARES[c_][1] and c_ == CMP[sp], (bs, c_)
            by = {}
            for b in bs:
                rows = []
                for ds in WQ.EVAL_ORDER:
                    nw = round(0.59 + cur.get(("e", sp, ds), 0.0), 4)
                    r5 = 0.59 if b == BASE_FIT[sp] else round(0.59 + cur.get(("b", sp, ds), 0.0), 4)
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
            with WQ.patch(WQ, run_compare=r2ew_run):
                assert compare_all(B) == 0 and compare_all(B) == 0
            return verdict(B)

        v = run({("e", "WQ-hotpotqa", "squad"): 0.03, ("b", "WQ-musique", "metaqa"): 0.02})
        assert v["verdict"] == "ADOPT_FOR_FULL_RUN" and not v["problems"], v["problems"]
        assert (v["gains"], v["losses"], v["next_form"]) == (["WQ-hotpotqa squad"], [], NEXT["ADOPT_FOR_FULL_RUN"])
        assert v["gates"] == [ge] and v["device"] == K.device_of(TOY) and len(v["calls"]) == 12 and v["naming"] == NAMING
        assert v["fits"] == {ARM: FIT, BESIDE_ARM: BESIDE_FIT}
        k_ = WQ.read_json(K.screen_dir(B) / f"{CMP['WQ-hotpotqa']}.json")[KEY]
        assert (k_["new"], k_["bases"], k_["device"]) == (FIT["WQ-hotpotqa"], [BASE_FIT["WQ-hotpotqa"], BESIDE_FIT["WQ-hotpotqa"]],
                                                          K.device_of(TOY))
        b = v["beside"]
        assert b["calls"]["zrckpbe-vs-zrckpb"] == {"gains": ["WQ-hotpotqa squad"], "losses": ["WQ-musique metaqa"]}
        assert b["r5"]["WQ-musique metaqa"] == {
            "read": "zero-shot", "rrf": 0.4, "zspk": 0.6, "zrckp": 0.59, "zrckpb": 0.61, "zrckpbe": 0.59, "floor": 0.0075,
            "call": "WITHIN", "zrckpbe_vs_zrckpb": "LOSS", "links": link_stats(FAKE_COUNTS)}
        assert b["r5"]["WQ-musique musique"]["floor"] == round(2 * (((0.02 ** 2 + 0.01 ** 2) / 2) ** 0.5), 6)
        assert b["terms"][FIT["WQ-musique"]] == {"questions": 100, "questions_with_term": 90, "share_with_term": 0.9,
                                                 "br_nll_first": 4.6, "br_nll_last": 3.9}
        assert b["timing"]["WQ-musique"]["zrckpbe"] == {"train_seconds": 150.0, "read_seconds": 12.0, "train_peak_gpu_gb": 3.5,
                                                        "read_peak_gpu_gb": 4.5, "data_gb": 2.0, "peak_gb": 9.0}
        assert b["timing"]["WQ-hotpotqa"]["zrckp"]["peak_gb"] == 8.0 and b["timing"]["WQ-hotpotqa"]["zrckpb"]["peak_gb"] == 9.0
        assert b["recall_sha256"] == WQ.sha_file(B / (RECALL["zrckpbe"] + ".json"))
        for c_ in CALLS:
            r_ = WQ.read_json(B / (RECALL[c_] + ".json"))
            assert (r_["stage"], r_["against"], r_["deciding"], len(r_["rows"])) == (STAGE, AGAINST[c_], DECIDING[c_], 12)
            assert (B / (PAIR[c_] + ".md")).read_text(encoding="utf-8").startswith(f"R2EW ({DECLARED})")
        r1 = WQ.read_json(B / (RECALL["zrckpbe-vs-zrckpb"] + ".json"))
        assert {(r["split"], r["dataset"]): r["base"] for r in r1["rows"]}[("WQ-musique", "metaqa")] == 0.61
        md = (B / (VERDICT + ".md")).read_text(encoding="utf-8")
        assert "zrckpbe **ADOPT_FOR_FULL_RUN**" in md and "| WQ-hotpotqa squad (in-domain) | 0.4000 | 0.6000 | 0.5900 |" in md
        v = run({("e", "WQ-hotpotqa", "squad"): 0.03, ("e", "WQ-musique", "webqsp"): -0.03})
        assert (v["verdict"], v["losses"], v["next_form"]) == ("NOT_ADOPTED", ["WQ-musique webqsp"], NEXT["NOT_ADOPTED"])
        v = run({("e", "WQ-musique", "musique"): 0.02})        # a filed GAIN within the raised floor (0.0316)
        assert (v["verdict"], v["gains"], v["losses"], v["next_form"]) == ("NOT_ADOPTED", [], [], NEXT["NOT_ADOPTED"])
        assert v["calls"]["WQ-musique musique"] == "WITHIN" and v["delta_r5"]["WQ-musique musique"] == 0.02
        r0 = WQ.read_json(B / (RECALL["zrckpbe"] + ".json"))
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
        rc_[KEY]["new"] = "scr-r2ew-zrckpbe"
        WQ.write_json(ce, rc_)
        incomplete("is not R2EW's compare of")
        ce.write_bytes(keep)
        n0, nb0 = FIT["WQ-musique"], BESIDE_FIT["WQ-musique"]
        bad_r2 = {"declared_in": R2_DECLARED, "constants": {**R2_CONST, "K": 6}, "params": list(R2_PARAMS),
                  "term": {"lam": 1.0, "questions_with_term": 90}}
        no_term = {"declared_in": R2_DECLARED, "constants": dict(R2_CONST), "params": list(R2_PARAMS),
                   "term": {"lam": 1.0, "questions_with_term": 0}}
        for nm, kw, has in (
                (n0, {"dev": {**K.device_of(TOY), "gpu": "Tesla P100"}}, "ran on different devices"),
                (n0, {"gate_sha": "f" * 64}, "names the gate ffffffffffff, not a filed R2EW PASS gate"),
                (n0, {"r2": bad_r2}, f"{n0}: R2's constants or EM term are not as declared"),
                (n0, {"r2": no_term}, f"{n0}: R2's constants or EM term are not as declared"),
                (n0, {"rows_cap": 32768}, "has no R2EW read on the device it was fitted on"),
                (n0, {"r2e": None}, f"{n0}: its flag is not R2e's as declared"),
                (n0, {"r2e": {**R2E, "fixed": "learned"}}, f"{n0}: its flag is not R2e's as declared"),
                (n0, {"stats": False}, f"{n0}: its read holds no link statistics on the six carves")):
            fake_fit(fr, nm, **{"gate_sha": ge, "overlay_sha": oe, **kw})
            incomplete(has)
            fake_fit(fr, nm, ge, oe)
        for kw, has in (({"dev": {**K.device_of(TOY), "gpu": "Tesla P100"}}, "ran on different devices"),
                        ({"gate_sha": "f" * 64}, f"{nb0}: its train names the gate ffffffffffff, not a filed R2GW PASS gate"),
                        ({"r2": bad_r2}, f"{nb0}: R2's constants or EM term are not as declared"),
                        ({"r2": no_term}, f"{nb0}: R2's constants or EM term are not as declared"),
                        ({"r2g": GW.R2G}, f"{nb0}: zrckpb is gated"),
                        ({"rows_cap": 32768}, "has no R2GW read on the device it was fitted on")):
            GW.fake_fit(fr, nb0, **{"gate_sha": gg, "overlay_sha": og, **kw})
            incomplete(has)
            GW.fake_fit(fr, nb0, gg, og)
        bk = td / "zrckpb_backup"
        shutil.copytree(fr / nb0, bk)
        shutil.rmtree(fr / nb0)
        incomplete("no R2GW stamp")                                          # R2GW's zrckpb fit not taken
        shutil.copytree(bk, fr / nb0)
        assert verdict(B)["verdict"] == "NOT_ADOPTED"
        out2 = td / "r2ew_out2"
        g2_, _o2 = K.fake_session(out2, B, verdict_="FAIL", tag="h")
        take(out2, B)
        incomplete("FAIL, not WQK's PASS")
        shutil.rmtree(B / SESSIONS / g2_[:12])
        shutil.rmtree(B / GW.SESSIONS / gg[:12])
        incomplete(f"{nb0}: its train names the gate {gg[:12]}, not a filed R2GW PASS gate")     # R2GW's session not filed
        GW.take(out_g, B)
        p = K.screen_dir(B) / "scr-wqk-zrckp-s1.json"
        keep = p.read_bytes()
        r = WQ.read_json(p)
        r["by_base"]["outputs/screen/fits/scr-wqk-zrckp-s0"]["rows"][2]["delta"][0] = 0.05
        WQ.write_json(p, r)
        incomplete("floors-zrckp.json is not its compares' floors")
        p.write_bytes(keep)

        def short(base, dev, c_):
            rec = recall.__wrapped__(base, dev, c_)
            rec["rows"] = rec["rows"][:-1]
            return rec

        short_fn = types.FunctionType(short.__code__, {**globals(), "recall": types.SimpleNamespace(__wrapped__=recall)})
        with WQ.patch(me, recall=short_fn):
            incomplete("holds 11 reads, not the twelve")
        assert verdict(B)["verdict"] == "NOT_ADOPTED"
    # 11. nothing left patched in zr2br, lean_gpu or the carve, no stats kept, and nothing written under R2EW's real folders
    assert R2.STASH["S"] is None and not STATS and ec.r2e_stats is None
    assert real_state() == before, (real_state(), before)
    print(f"zr2ew selftest: ok ({time.time() - t0:.1f}s)", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    import argparse
    ap = argparse.ArgumentParser(prog="zr2ew.py", allow_abbrev=False)
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
