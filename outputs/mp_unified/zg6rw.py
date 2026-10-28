"""G6RW (docs/G6RW_G6R_UNDER_WQ.md): G6r (docs/G6R_RELATION_MESSAGES.md, held on 27 Oct before any fit, its section 10)
under WQ on Kaggle's T4, against WQK's base set (docs/WQK_WQ_ON_T4.md, section 9, item 1). zg6r (zg6r.py, imported
unchanged: its model, carve, arm, encoder copy and relation map) is trained by zrm's train and read by zrm's read at WQK's
names and roots with WQK's mixed KT input (zwqk.py, imported unchanged), as WQK runs zg6. Two inputs are G6RW's: webqsp's
fit carve's typed links, built on the laptop by U2n's code (zu2n.build_typed, unchanged) on WQK's look, cache, links and
rmatch chains of that carve; and the encoder's start, WQK's seed-0 zspk fit of the split (its models.pt, p@swa). Each fit
is read on the device it was fitted on, 32,768 rows a batch, compared with WQK's seed-0 zspk fit of its split (deciding)
and WQK's seed-0 zg6 fit (beside), all on one device, and re-called under WQK's zspk seed null by G6's rule.

On the laptop, before the session (the arrays and weights are never committed):

    python outputs/mp_unified/zg6rw.py typed                  # outputs/g6rw/typed/webqsp/fit and typed.json
    python outputs/mp_unified/zg6rw.py ship --enc-from DIR    # outputs/g6rw/enc/<split>.pt and ship.json

In the kernel (zg6rw_session.py runs each in its own process, after zwqk.py overlay and gate):

    python outputs/mp_unified/zg6rw.py fit --name scr-g6rw-zg6r-hp --rows-cap 32768

On the laptop, after `kaggle kernels output` into a new empty folder:

    python outputs/mp_unified/zg6rw.py take --from DIR   # each fit's records, reads and relation map, never its weights
    python outputs/mp_unified/zg6rw.py compare-all       # outputs/screen/scr-g6rw-zg6r-vs-zspk[-hp].json
    python outputs/mp_unified/zg6rw.py verdict           # outputs/g6rw/screen-verdict (.json, .md)
    python outputs/mp_unified/zg6rw.py --selftest

This file imports the standard library and zwqk (with it zwq) only; zg6r (torch, zg6b2, zu2n and KG's chain) and torch are
imported by the commands that need them, outside zwq.on_wq. verdict exits 0 on ADOPT_FOR_FULL_RUN or NOT_ADOPTED and 2
after writing an INCOMPLETE record.
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

import zwqk as K  # noqa: E402  (the standard library and zwq at import)

WQ = K.WQ
DECLARED = "docs/G6RW_G6R_UNDER_WQ.md"
STAGE = "G6RW"
KEY = "g6rw"                                  # the stamp's key in screen.json, read.json and G6RW's compares
ARM, SEED = "zg6r", 0                         # zg6r's, checked when it is imported
SPLITS = tuple(WQ.SPLITS)
READ_CAP = 32768                              # G6r's rows a read batch (whole questions, so the scores do not change)
FIT = {sp: f"scr-g6rw-{ARM}{WQ.SUF[sp]}" for sp in SPLITS}
SPLIT_OF = {n: sp for sp, n in FIT.items()}
ZSPK = {sp: K.fit_name("zspk", 0, sp) for sp in SPLITS}           # WQK's seed-0 zspk: the deciding base, the encoder
ZG6 = {sp: K.fit_name("zg6", 0, sp) for sp in SPLITS}             # WQK's seed-0 zg6: beside
CMP = {sp: f"scr-g6rw-{ARM}-vs-zspk{WQ.SUF[sp]}" for sp in SPLITS}
COMPARES = {CMP[sp]: (FIT[sp], [ZSPK[sp], ZG6[sp]]) for sp in SPLITS}
WQK_ZG6_CMP = {sp: f"scr-wqk-zg6-vs-zspk{WQ.SUF[sp]}" for sp in SPLITS}     # zg6's R@5 as WQK filed it
WQK_ZRCKP = {sp: K.fit_name("zrckp", 1, sp) for sp in SPLITS}              # its base column: zrckp's seed-0 R@5 (the MLP)
NEW = "outputs/g6rw/"
TYPED_ROOT = NEW + "typed"                    # webqsp's fit carve's typed links (U2n's record form)
TYPED_REC = NEW + "typed.json"                # G6RW's record of them
ENC_ROOT = NEW + "enc"                        # <split>.pt: WQK's seed-0 zspk fit's models.pt
SHIP = NEW + "ship.json"                      # the session's dataset: every file, its size and sha256
SESSIONS = NEW + "sessions"
VERDICT = NEW + "screen-verdict"
PAIR = {0: NEW + "g6rw-zg6r-pair", 1: NEW + "g6rw-zg6r-vs-zg6-pair"}
RECALL = {k: f"{p}-recall" for k, p in PAIR.items()}
AGAINST = {0: "zspk", 1: "zg6"}
NULL_KIND = {0: "WQK's zspk seed null",
             1: "WQK's zspk seed null, its seed-0 R@5 replaced by WQK's zg6 read (beside)"}
FLOORS = K.FLOORS["zspk"]
REL_FILE = "g6rw_rel.json"
TIMING_FILE = "g6rw_timing.json"
OWN_TYPED = (("webqsp", "fit"),)              # the typed carve G6RW builds; every other one is U2n's
ENC_SHA = {"WQ-musique": "30cc29613f948f01a4de0aec4e58d6e31126ab028fb8cef18d7f7e69d62fcf09",
           "WQ-hotpotqa": "25e940c2fc47e049ac740b3b5cdd2e27798c188496eeb17447084e91fc4b720e"}
ENC_NAMES = ("U", "V", "cm_a", "cm_b", "cm_d", "cm_kappa", "cm_dir", "cm_bias", "cm_pi_w", "cm_pi_b", "cm_beta",
             "l1.weight", "l1.bias", "l2.weight", "l2.bias")            # zg6r's: zrm's block encoder and chain match
ENC_ELEMENTS = 224737
CODE = K.CODE + ("zg6r", "zu2n", "zg6b2", "zg6b", "zkg", "zu2d", "zg6rw")
RULE = ("ADOPT_FOR_FULL_RUN when zg6r's re-call against WQK's seed-0 zspk under WQK's zspk seed null has no LOSS on any of "
        "its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when WQK's base verdict is not "
        "BASE_READY, G6b2's verdict is not NOT_ADOPTED with G6r next, a fit or compare is missing or not G6RW's, G6RW's fits "
        "and WQK's seed-0 zspk and zg6 fits are not on one device, a fit or read names a gate that is not a filed PASS "
        "gate of a G6RW session, a fit was not trained from WQK's zspk encoder and typed.json's typed links, WQK's floors "
        "are not their compares', or a re-call does not hold the twelve reads (G6's rule, docs/G6_FREE_GNN.md section 4, "
        "under WQ on the T4; docs/G6RW_G6R_UNDER_WQ.md section 5)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zg6r under WQ fitted again on the pools WQP carried (U2s's pick), then its full run under "
                              "WQ beside zspk's, then Claim 1 (P3c's intervals) and Claim 3 (cold latency, the C stages' "
                              "timing) measured again; each its own declared file",
        "GAIN_AND_LOSS": "zg6r under G6b's gate under WQ on the T4 (its edge pass and its no-edge pass mixed per question "
                         "by the question's edge-family shares, as zg6b mixes zg6's; the encoder from WQK's zspk); one "
                         "declared file",
        "NO_GAIN": "the two changes apart under WQ on the T4: the relation messages with zg6's own encoder (zg6r-m) and "
                   "zg6 with WQK's zspk encoder start (zg6-z), each one fit pair; one declared file",
        "INCOMPLETE": "what is missing is supplied (WQK's base set, G6b2's verdict, a fit, a compare, a session's gate or "
                      "a re-call); nothing else changes"}
assert len(set(FIT.values())) == len(FIT) == 2 and not set(FIT.values()) & set(K.FITS)
assert all(b in K.FITS for sp in SPLITS for b in (ZSPK[sp], ZG6[sp]))
assert all(c in K.COMPARES for sp in SPLITS for c in (WQK_ZG6_CMP[sp], WQK_ZRCKP[sp]))
assert FLOORS == "outputs/wqk/floors-zspk" and len(set(ENC_NAMES)) == len(ENC_NAMES)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zg6rw: {msg}", flush=True)


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


# ── zg6r, imported outside zwq.on_wq ─────────────────────────────────────────


def import_g6r():
    """zg6r (torch, zg6b2, zu2n and KG's chain; its arm in lean_screen's table), imported outside zwq.on_wq after zg6
    (zwqk.import_g6): both bind their own splits at import. Its arm, seed, read cap and encoder checked G6RW's."""
    K.import_g6()
    import zg6r as G6R
    got = (G6R.ARM, G6R.SEED, G6R.READ_CAP, tuple(G6R.ENC_NAMES), G6R.ENC_ELEMENTS, tuple(G6R.SPLITS))
    want = (ARM, SEED, READ_CAP, ENC_NAMES, ENC_ELEMENTS, ("L-musique", "L-hotpotqa"))
    if got != want:
        raise SystemExit(f"zg6rw: zg6r is {got}, not G6RW's {want}")
    G6R.check_arm()
    return G6R


def same_modules(G6R, W):
    """zg6r's modules are zwq's inside zwq.on_wq (W), so the names on_wq patches are the ones zg6r's carve and zrm read."""
    pairs = ((G6R.Z3, W.Z3), (G6R.Z2, W.Z2), (G6R.LG, W.LG), (G6R.S, W.S), (G6R.S2, W.S2), (G6R.LC, W.LC),
             (G6R.RM, W.Z3.RM), (G6R.ZL, W.Z3.ZL))
    bad = [b.__name__ for a, b in pairs if a is not b]
    if bad:
        raise SystemExit(f"zg6rw: zg6r does not train on zwq's {bad}")


# ── guards ───────────────────────────────────────────────────────────────────


def split_of(name):
    if name not in SPLIT_OF:
        raise SystemExit(f"zg6rw: {name} is not one of G6RW's fits ({', '.join(FIT.values())})")
    return SPLIT_OF[name]


def where_ok(env, what):
    if not K.ok_where(env):
        raise SystemExit(f"zg6rw: G6RW {what}s in a Kaggle kernel on a {K.GPU} from the image {K.IMAGE_TAG} (section "
                         f"5), not with {K.device_of(env)} ({env.get('where')})")


def cap_ok(rows_cap):
    if rows_cap is None or int(rows_cap) != READ_CAP:
        raise SystemExit(f"zg6rw: G6RW reads with --rows-cap {READ_CAP} (G6r's, section 5), not {rows_cap}")


def wqk_ready(base=None):
    """WQK's verdict record, committed before the stub: BASE_READY under WQK's declaration (section 5)."""
    p = Path(base or WQ.ROOT) / (K.VERDICT + ".json")
    if not p.is_file():
        raise SystemExit(f"zg6rw: {K.VERDICT}.json missing; G6RW trains after WQK's BASE_READY (section 5)")
    rec = WQ.read_json(p)
    if (rec.get("stage"), rec.get("declared_in"), rec.get("verdict")) != (K.STAGE, K.DECLARED, "BASE_READY"):
        raise SystemExit(f"zg6rw: {K.VERDICT}.json is {rec.get('verdict')}, not WQK's BASE_READY")
    return rec


# ── webqsp's fit carve's typed links (section 3) ─────────────────────────────


def own_typed(base=None):
    """typed.json, G6RW's record of webqsp's fit carve's typed links, and the carve record it pins (U2n's record form,
    U2n's code). Returns typed.json's sha256."""
    base = Path(base or WQ.ROOT)
    tj = base / TYPED_REC
    if not tj.is_file():
        raise SystemExit(f"zg6rw: {TYPED_REC} missing (zg6rw.py typed, section 3)")
    t = WQ.read_json(tj)
    if (t.get("stage"), t.get("declared_in")) != (STAGE, DECLARED):
        raise SystemExit(f"zg6rw: {TYPED_REC} is not G6RW's")
    for c, x in (t.get("carves") or {}).items():
        rp = base / TYPED_ROOT / c / "record.json"
        if not rp.is_file() or WQ.sha_file(rp) != x.get("record"):
            raise SystemExit(f"zg6rw: {TYPED_ROOT}/{c}/record.json is not the record {TYPED_REC} pins")
    if sorted(t.get("carves") or {}) != sorted(f"{ds}/{cv}" for ds, cv in OWN_TYPED):
        raise SystemExit(f"zg6rw: {TYPED_REC} pins {sorted(t.get('carves') or {})}, not {OWN_TYPED}")
    return WQ.sha_file(tj)


@contextlib.contextmanager
def on_typed(U2N, base=None):
    """U2n's typed carves with webqsp's fit carve added; zg6r's carve reads that carve's typed links from G6RW's root and
    every other carve's from U2n's root, as zg6r reads them. Restored after, on a failure too."""
    own = (Path(base or WQ.ROOT) / TYPED_ROOT).resolve()
    u2n_root = Path(U2N.TYPED_ROOT).resolve()
    orig = U2N.load_typed

    def load_typed(c, root, cache_root, lk, verify=True):
        if Path(root).resolve() != u2n_root:
            raise SystemExit(f"zg6rw: typed links asked from {root}, not U2n's root (zg6r's carve)")
        return orig(c, own if (c.ds, c.carve) in OWN_TYPED else root, cache_root, lk, verify)

    with U2N.extra_carves(*OWN_TYPED), WQ.patch(U2N, load_typed=load_typed):
        yield U2N


def typed_records(carves, U2N, base=None):
    """zu2n.typed_records: webqsp's fit carve's record at G6RW's root, every other KB carve's at U2n's root."""
    own = [tuple(c) for c in carves if tuple(c) in OWN_TYPED]
    out = U2N.typed_records([tuple(c) for c in carves if tuple(c) not in OWN_TYPED], U2N.TYPED_ROOT)
    out.update(U2N.typed_records(own, Path(base or WQ.ROOT) / TYPED_ROOT))
    return out


# ── the encoder's start: WQK's seed-0 zspk fit of the split (section 4) ──────


def load_encoder(split, base=None):
    """(the ENC_NAMES tensors of WQK's seed-0 zspk fit of split, its p@swa; their record) from outputs/g6rw/enc/<split>.pt
    (that fit's models.pt), whose sha256 must be the one pinned here, the one its train.json names."""
    import torch
    p = Path(base or WQ.ROOT) / ENC_ROOT / f"{split}.pt"
    if not p.is_file():
        raise SystemExit(f"zg6rw: {ENC_ROOT}/{split}.pt missing (WQK's {ZSPK[split]}; zg6rw.py ship, then the session's "
                         f"dataset)")
    h = WQ.sha_file(p)
    if h != ENC_SHA[split]:
        raise SystemExit(f"zg6rw: {ENC_ROOT}/{split}.pt: sha256 {h[:12]}, not WQK's {ZSPK[split]} "
                         f"({ENC_SHA[split][:12]})")
    blob = torch.load(p, map_location="cpu", weights_only=True)
    if (blob.get("fit"), blob.get("seed")) != (ZSPK[split], 0) or "p@swa" not in (blob.get("states") or {}):
        raise SystemExit(f"zg6rw: {ENC_ROOT}/{split}.pt is {blob.get('fit')} seed {blob.get('seed')}, not "
                         f"{ZSPK[split]} seed 0 with a p@swa")
    st = blob["states"]["p@swa"]
    lack = [n for n in ENC_NAMES if n not in st]
    if lack:
        raise SystemExit(f"zg6rw: {ENC_ROOT}/{split}.pt: zspk's p@swa lacks {lack}")
    n = sum(int(st[k].numel()) for k in ENC_NAMES)
    if n != ENC_ELEMENTS:
        raise SystemExit(f"zg6rw: {ENC_ROOT}/{split}.pt: the encoder holds {n} numbers, not {ENC_ELEMENTS}")
    info = {"path": f"{ENC_ROOT}/{split}.pt", "sha256": h, "elements": n, "candidate": "p@swa", "fit": blob["fit"],
            "seed": blob["seed"], "hidden": blob.get("hidden"), "basis": blob.get("basis"),
            "basis_sha256": blob.get("basis_sha256")}
    return {k: st[k].detach().clone() for k in ENC_NAMES}, info


@contextlib.contextmanager
def on_encoder(split, G6R, base=None):
    """WQK's zspk encoder active (zg6r.using_encoder): each zg6r built meanwhile starts from it."""
    st, info = load_encoder(split, base)
    with G6R.using_encoder(st, info):
        yield info


# ── train and read in the kernel (section 5) ─────────────────────────────────


def code_shas():
    return {m: WQ.sha_src(HERE / f"{m}.py") for m in CODE}


def head(G6R):
    """zg6r's settings, its encoder and typed carves worded for G6RW; the roots."""
    model = dict(G6R.SETTINGS)
    model["encoder"] = ("zrm's block encoder and chain match (ENC_NAMES, 224,737 numbers) copied from WQK's seed-0 zspk fit "
                        "of the split (p@swa) when zg6r is built, then trained")
    model["typed_carves"] = [list(x) for x in tuple(G6R.U2N.TYPED_CARVES) + OWN_TYPED]
    return {"model": model, "kt_root": K.ROOTS["kt"], "typed_roots": {"u2n": WQ.rel(Path(G6R.U2N.TYPED_ROOT)),
                                                                      "g6rw": TYPED_ROOT},
            "enc_root": ENC_ROOT, "roots": dict(K.ROOTS), "read_rows": READ_CAP}


def stamp_fit(name, env, G6R, records, enc, base=None):
    """The fit's record after zrm's train: the wq block at WQK's roots and declaration (zwqk.wq_block), zg6r's code
    shas, and G6RW's block (the device, WQK's gate, overlay, check and verdict, typed.json, the KT, typed and chain
    records, the encoder, the head and the code)."""
    base = Path(base or WQ.ROOT)
    split = split_of(name)
    sj = WQ.fit_dir(name, base) / "screen.json"
    rec = WQ.read_json(sj)
    with WQ.patch(WQ, ROOT=base):
        wq = K.wq_block(ARM, SEED, split)
    rec.update({"zwq_sha256": WQ.sha_src(WQ.__file__), "zwqk_sha256": WQ.sha_src(K.__file__),
                "zg6rw_sha256": WQ.sha_src(__file__), **G6R.code_shas(), "wq": wq})
    rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": ARM, "seed": SEED, "split": split,
                "wqk_bases": {"zspk": ZSPK[split], "zg6": ZG6[split]}, "device": K.device_of(env), "env": env,
                "gate": WQ.sha_file(base / K.RECORDS["GATE"]), "overlay": WQ.sha_file(base / K.RECORDS["OVERLAY"]),
                "check": WQ.sha_file(base / K.RECORDS["CHECK"]),
                "wqk_verdict": WQ.sha_file(base / (K.VERDICT + ".json")), "typed": WQ.sha_file(base / TYPED_REC),
                "cache_root": K.ROOTS["cache"], **records, "encoder": enc, "head": head(G6R), "code": code_shas()}
    WQ.write_json(sj, rec)
    return rec


def train(name, env=None):
    """G6RW's fit (section 5): zrm's train for zg6r at WQK's names and roots with WQK's mixed KT input, zg6r's carve
    reading webqsp's fit carve's typed links from G6RW's root, WQK's zspk encoder active; then the fit's stamps."""
    split = split_of(name)
    env = K.device_env() if env is None else env
    where_ok(env, "train")
    wqk_ready()
    G6R = import_g6r()
    own_typed()
    with K.on_wqk():
        WQ.gate_passed()
    argv = ["train", "--name", name, "--arm", ARM, "--device", "cuda", "--cache-root", K.ROOTS["cache"]]
    with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W, on_typed(G6R.U2N), on_encoder(split, G6R) as ei:
        same_modules(G6R, W)
        G6R.check_arm()
        W.Z2.cache_root_of(argv)
        argv = W.Z2.on_card(argv) + ["--seed", str(SEED)]
        W.Z2.gate_passed()
        records = {"kt_records": W.Z3.kt_records(W.Z3.train_carves(split)),
                   "typed_records": typed_records(W.Z3.train_carves(split), G6R.U2N)}
        with W.Z2.on_p1c2():
            rc = G6R.RM.train(argv, split)
            records.update({"rmatch_records": W.Z3.RM.built_records(), "zrc_records": W.Z2._BUILT(W.Z2.CH_ZRC),
                            "zlink_records": W.Z3.ZL.built_records()})
        enc = {**ei, "copies": G6R.ENC["copies"]}
    if rc == 0:
        G6R.encoder_used(name, enc, WQ.fit_dir(name).parent)
        stamp_fit(name, env, G6R, records, enc)
        log(f"train {name}: done, stamped; WQK's zspk encoder ({enc['fit']}, {enc['elements']} numbers) copied into "
            f"{enc['copies']} zg6r")
    return rc


def fit_stamp(name, fits_root=None):
    """A G6RW fit's screen.json and its G6RW block, checked against the fit's name."""
    split = split_of(name)
    sj = Path(fits_root or WQ.fit_dir(name).parent) / name / "screen.json"
    rec = WQ.read_json(sj) if sj.is_file() else {}
    k = rec.get(KEY) or {}
    if (rec.get("arm"), k.get("declared_in"), k.get("fit"), k.get("arm"), k.get("seed"), k.get("split")) != \
            (ARM, DECLARED, name, ARM, SEED, split):
        raise SystemExit(f"zg6rw: {name} is not G6RW's {ARM} fit of {split} (no G6RW stamp)")
    return rec, k


def read(name, env=None, rows_cap=READ_CAP):
    """The fit read by zrm's read with p@swa on the six s1eval carves, --rows-cap 32768, on the device it was fitted on,
    the KB carves' typed links U2n's. read.json is stamped when the read exits 0."""
    split = split_of(name)
    cap_ok(rows_cap)
    env = K.device_env() if env is None else env
    where_ok(env, "read")
    _rec, k = fit_stamp(name)
    if K.device_of(env) != k.get("device"):
        raise SystemExit(f"zg6rw: {name} was fitted on {k.get('device')}; this read runs on {K.device_of(env)}")
    G6R = import_g6r()
    with K.on_wqk():
        WQ.cache_root_ok(K.ROOTS["cache"])
        WQ.gate_passed()
    argv = ["--name", name, "--device", "cuda", "--cache-root", K.ROOTS["cache"], "--rows-cap", str(READ_CAP)]
    with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W, on_typed(G6R.U2N):
        same_modules(G6R, W)
        G6R.check_arm()
        W.Z2.cache_root_of(argv)
        argv = W.Z2.on_card(argv)
        kt = W.Z3.kt_records(W.Z3.eval_carves())
        ty = typed_records(W.Z3.eval_carves(), G6R.U2N)
        with W.Z2.on_p1c2():
            rc = G6R.RM.read(argv)
    rj = WQ.fit_dir(name) / "read.json"
    if rc == 0 and rj.is_file():
        rec = WQ.read_json(rj)
        rec["zg6rw_sha256"] = WQ.sha_src(__file__)
        rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": ARM, "seed": SEED, "split": split,
                    "device": K.device_of(env), "env": env, "gate": WQ.sha_file(WQ.ROOT / K.RECORDS["GATE"]),
                    "kt_records": kt, "typed_records": ty, "rows_cap": int(rows_cap)}
        WQ.write_json(rj, rec)
    return rc


def stamped(p, name):
    try:
        k = WQ.read_json(p).get(KEY) or {}
    except (OSError, ValueError):
        return False
    return k.get("declared_in") == DECLARED and k.get("fit") == name


def rel_record(name, G6R, base=None):
    """g6rw_rel.json in the fit's folder: zg6r's relation map of its p@swa state (zg6r.rel_stats, unchanged) with the
    encoder's drift from WQK's zspk encoder."""
    import torch
    split = split_of(name)
    fd = WQ.fit_dir(name, Path(base or WQ.ROOT))
    blob = torch.load(fd / "models.pt", map_location="cpu", weights_only=False)
    enc, _info = load_encoder(split, base)
    rec = {"stage": STAGE, "declared_in": DECLARED, "fit": name, "split": split, "candidate": "p@swa",
           **G6R.rel_stats(blob["states"]["p@swa"], None, G6R.REL_SHOW, enc)}
    WQ.write_json(fd / REL_FILE, rec)
    return rec


def fit(name, rows_cap=READ_CAP):
    """One fit in the kernel (zwqk's fit, for G6RW): a read G6RW stamped, with its relation map, is kept; a fit trained
    and stamped but not read is only read; otherwise the fit's files are unlinked and it trains and reads. Then the
    relation map; g6rw_timing.json holds the seconds and the peak."""
    split_of(name)
    cap_ok(rows_cap)
    fd = WQ.fit_dir(name)
    if stamped(fd / "read.json", name) and (fd / REL_FILE).is_file():
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
    rc = 0
    if not stamped(fd / "read.json", name):
        K.clear(fd / "reads", "*")
        K.clear(fd, "read.json")
        rc = read(name, env, rows_cap)
    t2 = time.time()
    if rc == 0:
        rel_record(name, import_g6r())
    WQ.write_json(fd / TIMING_FILE, {"fit": name, "trained_here": not trained, "train_seconds": round(t1 - t0, 1),
                                     "read_seconds": round(t2 - t1, 1), "rel_seconds": round(time.time() - t2, 1),
                                     "peak_gb": K.peak_gb(), "rows_cap": int(rows_cap), "exit": rc,
                                     "declared_in": DECLARED})
    return rc


# ── on the laptop, before the session: typed and ship (sections 3, 4 and 8) ──


def typed(base=None):
    """Section 3: webqsp's fit carve's typed links by U2n's code (zu2n.build_typed, unchanged; its records carry U2n's
    stage) on WQK's look, cache, links and rmatch chains of that carve, into outputs/g6rw/typed/webqsp/fit; each part's
    cache and link records the ones WQK's ship.json names; then typed.json."""
    base = Path(base or WQ.ROOT)
    t0 = time.time()
    G6R = import_g6r()
    U2N = G6R.U2N
    r = {k: base / v for k, v in K.ROOTS.items()}
    shipped = WQ.read_json(base / K.SHIP)["files"]
    carves = {}
    for ds, cv in OWN_TYPED:
        with U2N.extra_carves((ds, cv)):
            rec = U2N.build_typed(ds, cv, chunk_root=r["plook"], out_root=base / TYPED_ROOT, look_root=r["plook"],
                                  cache_root=r["cache"], links_root=r["links"], ch_root=r["rmatch"])
        out = base / TYPED_ROOT / ds / cv
        parts = {}
        for p, sha in rec["parts"].items():
            h = WQ.read_json(out / p / "record.json")
            want = tuple((shipped.get(f"{K.ROOTS[x]}/{ds}/{cv}/{p}/record.json") or {}).get("sha256")
                         for x in ("cache", "links"))
            if (h["step1_record_sha256"], h["link_record_sha256"]) != want or None in want:
                raise SystemExit(f"zg6rw: {TYPED_ROOT}/{ds}/{cv}/{p} was built on cache and link records "
                                 f"{h['step1_record_sha256'][:12]} and {h['link_record_sha256'][:12]}, not the ones "
                                 f"WQK's ship.json names ({[str(w)[:12] for w in want]})")
            parts[p] = {"record": sha, "cache_record": want[0], "link_record": want[1], "edges": h["edges"],
                        "typed": h["typed"], "family0": h["family0"], "empty_family0": h["empty_family0"]}
        carves[f"{ds}/{cv}"] = {"record": WQ.sha_file(out / "record.json"), "parts": parts,
                                **{k: rec.get(k) for k in ("n_sets", "edges", "typed", "family0", "empty_family0",
                                                           "union_sizes", "relations")}}
    t = {"stage": STAGE, "declared_in": DECLARED, "carves": carves,
         "built_by": "zu2n.build_typed, U2n's code unchanged: its records carry U2n's stage and declaration",
         "roots": {"look": K.ROOTS["plook"], "chunks": K.ROOTS["plook"], "cache": K.ROOTS["cache"],
                   "links": K.ROOTS["links"], "rmatch": K.ROOTS["rmatch"], "out": TYPED_ROOT},
         "wqk_ship": WQ.sha_file(base / K.SHIP), "zu2n_sha256": WQ.sha_src(U2N.__file__),
         "zg6rw_sha256": WQ.sha_src(__file__), "seconds": round(time.time() - t0, 1), "utc": WQ.utc()}
    WQ.write_json(base / TYPED_REC, t)
    said = ", ".join("%s %s typed of %s edges, %s sets" % (c, x["typed"], x["edges"], x["n_sets"])
                     for c, x in carves.items())
    log(f"typed: {said} ({t['seconds']}s)")
    return t


def ship(enc_from, base=None):
    """Section 8: each split's encoder file (WQK's seed-0 zspk fit's models.pt from enc_from, a WQK kernel output's fits
    folder; its sha256 the one WQK's filed train.json names) at outputs/g6rw/enc/<split>.pt, loaded as the kernel loads
    it; the typed arrays checked against their records; then ship.json, every file the session's dataset holds."""
    base, enc_from = Path(base or WQ.ROOT), Path(enc_from)
    tsha = own_typed(base)
    files = {}
    for sp in SPLITS:
        tj = K.fits_dir(base) / ZSPK[sp] / "train.json"
        K.fit_stamp(ZSPK[sp], K.fits_dir(base))
        if (WQ.read_json(tj) if tj.is_file() else {}).get("models_sha256") != ENC_SHA[sp]:
            raise SystemExit(f"zg6rw: WQK's filed {ZSPK[sp]}/train.json does not name the encoder {ENC_SHA[sp][:12]}")
        dst = base / ENC_ROOT / f"{sp}.pt"
        if not dst.is_file():
            src = enc_from / ZSPK[sp] / "models.pt"
            if not src.is_file() or WQ.sha_file(src) != ENC_SHA[sp]:
                raise SystemExit(f"zg6rw: {ZSPK[sp]}/models.pt under {enc_from.name} is missing or not {ENC_SHA[sp][:12]}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        load_encoder(sp, base)
        files[f"{ENC_ROOT}/{sp}.pt"] = ENC_SHA[sp]
    for ds, cv in OWN_TYPED:
        out = base / TYPED_ROOT / ds / cv
        rec = WQ.read_json(out / "record.json")
        files[f"{TYPED_ROOT}/{ds}/{cv}/sets.npy"] = rec["sets"]["sha256"]
        for p in rec["parts"]:
            h = WQ.read_json(out / p / "record.json")
            for k in ("t_set", "t_dir"):
                files[f"{TYPED_ROOT}/{ds}/{cv}/{p}/{k}.npy"] = h["arrays"][k]["sha256"]
    got = {}
    for path, want in files.items():
        h = WQ.sha_file(base / path) if (base / path).is_file() else None
        if h != want:
            raise SystemExit(f"zg6rw: {path}: sha256 {str(h)[:12]}, not its record's {want[:12]}")
        got[path] = {"size": (base / path).stat().st_size, "sha256": h}
    s = {"stage": STAGE, "declared_in": DECLARED, "files": got, "total_bytes": sum(x["size"] for x in got.values()),
         "typed": tsha, "zg6rw_sha256": WQ.sha_src(__file__), "utc": WQ.utc()}
    WQ.write_json(base / SHIP, s)
    log(f"ship: {len(got)} files, {s['total_bytes'] / 1e6:.1f} MB")
    return s


# ── on the laptop, after the session: take, compare-all, verdict (sections 5 and 6) ──


def fit_device(name, fits_root=None):
    """A G6RW fit's device: its screen.json's G6RW block, and its read.json's on the same device with --rows-cap 32768."""
    _rec, k = fit_stamp(name, fits_root)
    rj = Path(fits_root or WQ.fit_dir(name).parent) / name / "read.json"
    r = WQ.read_json(rj) if rj.is_file() else {}
    rk = r.get(KEY) or {}
    if (rk.get("declared_in"), rk.get("fit"), rk.get("device"), rk.get("rows_cap")) != \
            (DECLARED, name, k.get("device"), READ_CAP):
        raise SystemExit(f"zg6rw: {name} has no G6RW read on the device it was fitted on")
    return k["device"]


def one_device(split, base):
    """The one device G6RW's fit of split and WQK's seed-0 zspk and zg6 fits of split were fitted and read on."""
    fr = K.fits_dir(base)
    devs = {FIT[split]: fit_device(FIT[split], fr), **{b: K.fit_device(b, fr) for b in (ZSPK[split], ZG6[split])}}
    if len({json.dumps(d, sort_keys=True) for d in devs.values()}) != 1:
        raise SystemExit(f"zg6rw: {FIT[split]} and its bases ran on different devices or images: {devs}")
    return devs[FIT[split]]


def take(src, base=None):
    """A kernel's output (src): each G6RW fit copied to outputs/screen/fits/<name>, its records, reads and relation map,
    never its weights; the session's gate.json and overlay.json to outputs/g6rw/sessions/<the gate's sha256[:12]>. A fit
    without G6RW's stamp and read on one device is left (named); a folder not G6RW's, or a file already there with other
    bytes, stops the take. Returns (copied, left)."""
    base, src = Path(base or WQ.ROOT), Path(src)
    fits = src / "fits"
    g, o = src / "wqk" / "gate.json", src / "wqk" / "overlay.json"
    if not (g.is_file() and o.is_file()):
        raise SystemExit(f"zg6rw take: {src.name} holds no wqk/gate.json and wqk/overlay.json (the session's records)")
    names = sorted(p.name for p in fits.iterdir() if p.is_dir()) if fits.is_dir() else []
    bad = [n for n in names if n not in SPLIT_OF]
    if bad:
        raise SystemExit(f"zg6rw take: {bad} are not G6RW's fits")
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
            raise SystemExit(f"zg6rw take: {WQ.rel(t, base)} exists with other bytes")
    took = []
    for f, t in plan:
        if not t.exists():
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, t)
            took.append(WQ.rel(t, base))
    log(f"take: {len(done)} fits, {len(took)} files copied; the gate {sess.name}" + (f"; left: {left}" if left else ""))
    return took, left


def compare_all(base=None):
    """lean_screen's compare of each G6RW fit with WQK's seed-0 zspk (first: it decides) and zg6 fits of its split, run
    from the repository's root by zwq's run_compare, all three on one device; each stamped with G6RW's record and that
    device. A compare G6RW stamped from the same fits is kept."""
    base = Path(base or WQ.ROOT)
    import_g6r()
    done, ran = [], []
    for sp in SPLITS:
        cname = CMP[sp]
        new, bases = COMPARES[cname]
        js = K.screen_dir(base) / f"{cname}.json"
        if js.is_file():
            k = WQ.read_json(js).get(KEY) or {}
            if (k.get("declared_in"), k.get("new"), k.get("bases")) == (DECLARED, new, list(bases)):
                done.append(cname)
                continue
        dev = one_device(sp, base)
        with K.on_wqk():
            for f in bases:
                why = WQ.wq_why(f, base)
                if why:
                    raise SystemExit(f"zg6rw: {why}")
            rc = WQ.run_compare(f"outputs/screen/fits/{new}", ",".join(f"outputs/screen/fits/{b}" for b in bases),
                                f"outputs/screen/{cname}")
        if rc or not js.is_file():
            raise SystemExit(f"zg6rw: the compare {cname} exited {rc}")
        rec = WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases), "device": dev,
                    "zg6rw_sha256": WQ.sha_src(__file__)}
        WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def cmp_rows(js, cname, dev=None):
    """A G6RW compare's rows {base fit: {dataset: row}}, after its checks; SystemExit naming what is wrong."""
    p = Path(js)
    if not p.is_file():
        raise SystemExit(f"outputs/screen/{cname}.json missing")
    rec = WQ.read_json(p)
    new, bases = COMPARES[cname]
    k = rec.get(KEY) or {}
    if (k.get("declared_in"), k.get("compare"), k.get("new"), k.get("bases")) != (DECLARED, cname, new, list(bases)):
        raise SystemExit(f"outputs/screen/{cname}.json is not G6RW's compare of {new} with {bases}")
    if dev is not None and k.get("device") != dev:
        raise SystemExit(f"outputs/screen/{cname}.json was made on {k.get('device')}, not on the fits' device {dev}")
    got = [WQ.parts_of(b)[-1:] for b in rec.get("bases") or []]
    if WQ.parts_of(rec.get("new", ""))[-1:] != (new,) or got != [(b,) for b in bases]:
        raise SystemExit(f"outputs/screen/{cname}.json compares {rec.get('new')} with {rec.get('bases')}")
    split = SPLIT_OF[new]
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
    """zwqk.filed_gates on G6RW's sessions: ({gate sha256: its record}, problems), each a WQK PASS gate on the reproduced
    check and its session's overlay."""
    with WQ.patch(K, SESSIONS=SESSIONS):
        return K.filed_gates(base)


def fit_problems(base, gates):
    """Why a G6RW fit is not section 5's: its train exit, its six reads, the gates its train, wq block and read name,
    its overlay, WQK's zspk encoder and typed.json's typed links."""
    out = []
    fr = K.fits_dir(base)
    tj = Path(base) / TYPED_REC
    tsha = WQ.sha_file(tj) if tj.is_file() else None
    for sp, n in FIT.items():
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
                out.append(f"{n}: its {what} names the gate {str(g)[:12]}, not a filed G6RW PASS gate")
        if k.get("gate") in gates and (gates[k["gate"]].get("records") or {}).get("overlay") != k.get("overlay"):
            out.append(f"{n}: its overlay is not its gate's")
        enc = k.get("encoder") or {}
        if (enc.get("sha256"), enc.get("fit")) != (ENC_SHA[sp], ZSPK[sp]) or (enc.get("copies") or 0) < 1:
            out.append(f"{n}: not trained from WQK's {ZSPK[sp]} encoder")
        if tsha is None or k.get("typed") != tsha:
            out.append(f"{n}: not trained on the typed links {TYPED_REC} records")
    return out


def restamp(pair, rcl, which):
    """G6RW's fields on the pair and re-call screen_pair and screen_recall wrote; a header on their markdown."""
    for p in (pair, rcl):
        js = Path(str(p) + ".json")
        rec = WQ.read_json(js)
        rec.update({"stage": STAGE, "declared_in": DECLARED, "arm": ARM, "against": AGAINST[which],
                    "null_kind": NULL_KIND[which], "deciding": which == 0, "zg6rw_sha256": WQ.sha_src(__file__)})
        WQ.write_json(js, rec)
        md = Path(str(p) + ".md")
        if md.is_file():
            head_ = (f"G6RW ({DECLARED}): zg6r against WQK's seed-0 {AGAINST[which]} under WQ on the T4"
                     + (", deciding." if which == 0 else ", beside, deciding nothing.") + "\n\n")
            md.write_text(head_ + md.read_text(encoding="utf-8"), encoding="utf-8")


def recall(base, dev, which):
    """screen_pair over G6RW's two compares against base `which` (0: WQK's zspk, deciding; 1: WQK's zg6, beside), then
    screen_recall under WQK's zspk null. Beside, the null's seed-0 R@5 is zg6's read as WQK filed it (scr-wqk-zg6-vs-zspk's
    new column), which the compare's zg6 column must equal; the floors stay zspk's."""
    import screen_pair as SP
    import screen_recall as SR
    base = Path(base)
    scr = K.screen_dir(base)

    def load(fn):
        c = next(c for c in COMPARES if Path(fn).stem == c)
        _rec, rows = cmp_rows(scr / f"{c}.json", c, dev)
        return SPLIT_OF[COMPARES[c][0]], [rows[COMPARES[c][1][which]][ds] for ds in WQ.EVAL_ORDER]

    pair, rcl = K.at(base, PAIR[which]), K.at(base, RECALL[which])
    pair.parent.mkdir(parents=True, exist_ok=True)
    with WQ.patch(SP, load=load):
        SP.pair([K.at(base, f"outputs/screen/{CMP[sp]}.json") for sp in SPLITS], pair)
    null, sources = K.wqk_null("zspk", base, dev, SR.floor_of)
    if which == 1:
        for sp in SPLITS:
            _r, kz = K.cmp_rows(scr / f"{WQK_ZG6_CMP[sp]}.json", WQK_ZG6_CMP[sp], dev)
            null[sp] = {ds: {**v, "seed0": kz[ds]["new"][0]} for ds, v in null[sp].items()}
            sources[f"WQK zg6 {sp}"] = f"outputs/screen/{WQK_ZG6_CMP[sp]}.json"
    with WQ.patch(SR, load_null=lambda _files: (null, sources)):
        rec = SR.pair([], Path(str(pair) + ".json"), rcl)
    restamp(pair, rcl, which)
    return rec


def tag(r):
    return f"{r['split']} {r['dataset']}"


def timing(name, fits_root):
    t = K.timing(name, fits_root, TIMING_FILE)
    p = Path(fits_root) / name / TIMING_FILE
    t["rel_seconds"] = (WQ.read_json(p) if p.is_file() else {}).get("rel_seconds")
    return t


def verdict(base=None, out=None):
    """G6RW's verdict by G6's rule from zg6r's re-call against WQK's zspk under WQK's zspk null (sections 5 and 7).
    Beside, deciding nothing: zg6r against WQK's zg6 under zspk's floors, R@5 per read with zrckp's and the MLP share,
    the relation map of each fit, and seconds and memory beside WQK's zg6 and zspk; outputs/g6rw/screen-verdict."""
    import screen_recall as SR
    base = Path(base or WQ.ROOT)
    od = Path(out) if out else base / VERDICT
    G6R = import_g6r()
    problems = []
    try:
        wqk_ready(base)
    except SystemExit as e:
        problems.append(str(e))
    problems += G6R.g6b2_filed(base / G6R.GB2.OUT)
    gates, gp = filed_gates(base)
    problems += gp
    devs = {}
    for sp in SPLITS:
        try:
            devs[sp] = one_device(sp, base)
        except SystemExit as e:
            problems.append(str(e))
    problems += fit_problems(base, gates)
    keys = {json.dumps(d, sort_keys=True) for d in devs.values()}
    if len(keys) > 1:
        problems.append(f"G6RW's fits ran on {len(keys)} devices: {sorted(keys)}")
    dev = json.loads(next(iter(keys))) if len(keys) == 1 else None
    rows = {}
    for c in COMPARES:
        try:
            rows[c] = cmp_rows(K.screen_dir(base) / f"{c}.json", c, dev)[1]
        except SystemExit as e:
            problems.append(str(e))
    problems += K.floors_problems(base, SR.floor_of)
    rc = {}
    if not problems:
        try:
            for which in AGAINST:
                rc[which] = recall(base, dev, which)
        except (SystemExit, KeyError) as e:
            problems.append(WQ.clean(e, base))
    whole = sorted((sp, ds) for sp in SPLITS for ds in WQ.EVAL_ORDER)
    for which, r in rc.items():
        if sorted((x["split"], x["dataset"]) for x in r["rows"]) != whole:
            problems.append(f"the re-call against {AGAINST[which]} holds {len(r['rows'])} reads, not the twelve of "
                            f"{', '.join(SPLITS)}")
    rrows = rc[0]["rows"] if 0 in rc and not problems else []
    gains = [tag(r) for r in rrows if r["call"] == "GAIN"]
    losses = [tag(r) for r in rrows if r["call"] == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    rec = {"stage": STAGE, "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "base_arm": "zspk",
           "fits": dict(FIT), "against": dict(ZSPK), "beside_bases": dict(ZG6), "floors": FLOORS + ".json",
           "null_kind": NULL_KIND[0], "device": dev, "gates": sorted(gates), "next_form": nxt, "gains": gains,
           "losses": losses, "problems": problems, "calls": {tag(r): r["call"] for r in rrows},
           "delta_r5": {tag(r): r["delta"] for r in rrows},
           "wqk_verdict": WQ.sha_file(base / (K.VERDICT + ".json")) if (base / (K.VERDICT + ".json")).is_file() else None,
           "zwqk_sha256": WQ.sha_src(K.__file__), "zg6rw_sha256": WQ.sha_src(__file__), "utc": WQ.utc()}
    md = [f"# G6RW, G6r under WQ on Kaggle's T4: zg6r **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zg6r** (zg6 with each typed KB edge's message from its relations' shared low-rank map, its encoder "
          f"started from WQK's seed-0 zspk fit) against WQK's seed-0 zspk on the T4, re-called under WQK's zspk seed "
          f"null: {len(gains)} GAIN ({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ".",
          f"- **Next form** (fixed in section 7 before any number): {nxt}.", f"- Device: {json.dumps(dev)}.",
          f"- Gates: {', '.join(g[:12] for g in sorted(gates)) or 'none'}."]
    beside = {}
    if v != "INCOMPLETE":
        fr = K.fits_dir(base)
        zrckp = {}
        for sp in SPLITS:
            _r, zr = K.cmp_rows(K.screen_dir(base) / f"{WQK_ZRCKP[sp]}.json", WQK_ZRCKP[sp], dev)
            zrckp[sp] = {ds: zr[ds]["base"][0] for ds in WQ.EVAL_ORDER}
        cz = {tag(r): r["call"] for r in rc[1]["rows"]}
        gz = [k for k, x in cz.items() if x == "GAIN"]
        lz = [k for k, x in cz.items() if x == "LOSS"]
        r5, share = {}, {}
        for r in rrows:
            sp, ds = r["split"], r["dataset"]
            r5[tag(r)] = {"read": r["read"], "rrf": r["rrf"], "zspk": r["base"],
                          "zg6": rows[CMP[sp]][ZG6[sp]][ds]["base"][0], "zg6r": r["new"], "zrckp": zrckp[sp][ds],
                          "floor": r["floor"], "call": r["call"], "call_vs_zg6": cz[tag(r)]}
            share[tag(r)] = G6R.G6.mlp_share(zrckp[sp][ds], r["new"], r["rrf"])
        rel = {}
        for sp, n in FIT.items():
            p = fr / n / REL_FILE
            rel[sp] = WQ.read_json(p) if p.is_file() else None
        tm = {sp: {"zg6r": timing(n, fr), "zg6": K.timing(ZG6[sp], fr, "wqk_timing.json"),
                   "zspk": K.timing(ZSPK[sp], fr, "wqk_timing.json")} for sp, n in FIT.items()}
        beside = {"vs_zg6": {"gains": gz, "losses": lz}, "r5": r5, "mlp_share": share, "rel": rel, "timing": tm,
                  "pair": PAIR[0] + ".json", "recall": RECALL[0] + ".json",
                  "recall_sha256": WQ.sha_file(Path(str(K.at(base, RECALL[0])) + ".json"))}
        fmt = K.fmt
        md += ["", "Beside, deciding nothing.", "",
               f"- zg6r against WQK's seed-0 zg6 under zspk's floors: {len(gz)} GAIN ({', '.join(gz) or 'none'}), "
               f"{len(lz)} LOSS ({', '.join(lz) or 'none'}).",
               "", f"R@5 per read on the T4; the MLP share = (zrckp - rrf) / (zg6r - rrf) where zg6r's gain over rrf "
                   f"exceeds {G6R.SHARE_MIN}:", "",
               "| read | rrf | zspk | zg6 | zg6r | zrckp (MLP) | MLP share | floor | call | vs zg6 |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for k_, t in r5.items():
            md.append(f"| {k_} ({t['read']}) | {fmt(t['rrf'])} | {fmt(t['zspk'])} | {fmt(t['zg6'])} | {fmt(t['zg6r'])} "
                      f"| {fmt(t['zrckp'])} | {fmt(share[k_])} | {fmt(t['floor'])} | {t['call']} | {t['call_vs_zg6']} |")
        md += ["", "The relation map of each fit's p@swa (zg6r's table, unchanged): P's norms and top singular values, "
                   "the share of their norm in metaqa's centred relation span (under WQ the unseen KB's: P learns from "
                   "webqsp's relations only), each KB's codes |P_f r| / |P_b r|, the layers' delta and norms, and the "
                   "encoder's drift from WQK's zspk:", ""]
        for sp in SPLITS:
            md += G6R.rel_md(sp, rel[sp])
        md += ["", "Seconds and memory: zg6r, and in brackets WQK's seed-0 zg6 and zspk of the same split, all on the "
                   "T4:", "",
               "| split | train s | read s | peak GPU GB, train / read | peak RAM GB |", "|---|---:|---:|---|---:|"]

        def s0(x):
            return f"{x:.0f}" if isinstance(x, (int, float)) else "n/a"
        for sp in SPLITS:
            a_, b_, c_ = tm[sp]["zg6r"], tm[sp]["zg6"], tm[sp]["zspk"]
            md.append(f"| {sp} | {s0(a_['train_seconds'])} ({s0(b_['train_seconds'])}, {s0(c_['train_seconds'])}) | "
                      f"{s0(a_['read_seconds'])} ({s0(b_['read_seconds'])}, {s0(c_['read_seconds'])}) | "
                      f"{fmt(a_['train_peak_gpu_gb'])} / {fmt(a_['read_peak_gpu_gb'])} ({fmt(b_['train_peak_gpu_gb'])} / "
                      f"{fmt(b_['read_peak_gpu_gb'])}, {fmt(c_['train_peak_gpu_gb'])} / {fmt(c_['read_peak_gpu_gb'])}) | "
                      f"{fmt(a_['peak_gb'])} ({fmt(b_['peak_gb'])}, {fmt(c_['peak_gb'])}) |")
    rec["beside"] = beside
    WQ.write(od, rec, md)
    log(f"verdict: {v} ({len(gains)} GAIN, {len(losses)} LOSS{'; ' + '; '.join(problems) if problems else ''}); "
        f"next: {nxt}")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def toy_encoder(p, fit_, n=ENC_ELEMENTS, drop=None, seed=0):
    """A models.pt in zrm's form whose p@swa holds the ENC_NAMES tensors (n numbers) and one other; its sha256."""
    import torch
    g = torch.Generator().manual_seed(seed)
    st = {k: torch.randn(n - len(ENC_NAMES) + 1 if i == 0 else 1, generator=g) for i, k in enumerate(ENC_NAMES)}
    st["g6_in_w"] = torch.randn(3, generator=g)
    if drop:
        st.pop(drop)
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fit": fit_, "seed": 0, "hidden": 16, "basis": "2wiki", "basis_sha256": "b" * 64,
                "states": {"p@swa": st}, "candidates": ["p@swa"]}, p)
    return WQ.sha_file(p)


def fake_build_typed(on):
    """zu2n.build_typed in its records' form (a carve record over part records and arrays), each part built on the
    (cache, link) record sha256s on[part]."""
    import numpy as np

    def build_typed(ds, carve, chunk_root, out_root=None, look_root=None, cache_root=None, links_root=None,
                    ch_root=None):
        U2N = sys.modules["zu2n"]
        assert (ds, carve) in U2N.TYPED_CARVES and Path(chunk_root) == Path(look_root), (ds, carve)
        out = Path(out_root) / ds / carve
        parts = {}
        for i, (p, (s1, lk)) in enumerate(sorted(on.items())):
            od = out / p
            od.mkdir(parents=True, exist_ok=True)
            arrays = {"t_set": np.array([i, -1, 1], np.int16), "t_dir": np.array([1, 0, 2], np.int8)}
            for k, a in arrays.items():
                np.save(od / f"{k}.npy", a)
            WQ.write_json(od / "record.json", {
                "stage": U2N.STAGE, "declared_in": U2N.DECLARED, "dataset": ds, "carve": carve, "part": p,
                "step1_record_sha256": s1, "link_record_sha256": lk, "edges": 3, "typed": 2, "family0": 3,
                "empty_family0": 1, "arrays": {k: {"dtype": a.dtype.str, "shape": list(a.shape),
                                                   "sha256": WQ.sha_file(od / f"{k}.npy")} for k, a in arrays.items()}})
            parts[p] = WQ.sha_file(od / "record.json")
        sets = np.full((2, U2N.SLOTS), -1, np.int16)
        sets[:, 0] = (3, 5)
        np.save(out / "sets.npy", sets)
        rec = {"stage": U2N.STAGE, "declared_in": U2N.DECLARED, "dataset": ds, "carve": carve, "slots": U2N.SLOTS,
               "n_sets": 2, "sets": {"dtype": sets.dtype.str, "shape": list(sets.shape),
                                     "sha256": WQ.sha_file(out / "sets.npy")},
               "parts": parts, "relations": {"n_relations": 7058, "k_rel": 4}, "edges": 3 * len(on),
               "typed": 2 * len(on), "family0": 3 * len(on), "empty_family0": len(on),
               "union_sizes": [2 * len(on)] + [0] * (U2N.SLOTS - 1)}
        WQ.write_json(out / "record.json", rec)
        return rec

    return build_typed


def fake_fit(fits_root, name, gate_sha, overlay_sha, typed_sha, rel, dev=None, read_=True, enc_sha=None,
             rows_cap=READ_CAP):
    """A G6RW fit as the kernel leaves it: screen.json (WQK's wq block, G6RW's block) and read.json with their stamps,
    train.json, g6rw_timing.json, the relation map, a read and models.pt."""
    sp = SPLIT_OF[name]
    d = Path(fits_root) / name
    (d / "reads").mkdir(parents=True, exist_ok=True)
    dev = dev or K.device_of(K.TOY_ENV)
    k = {"declared_in": DECLARED, "fit": name, "arm": ARM, "seed": SEED, "split": sp, "device": dev, "gate": gate_sha,
         "overlay": overlay_sha, "encoder": {"sha256": enc_sha or ENC_SHA[sp], "fit": ZSPK[sp], "copies": 1},
         "typed": typed_sha}
    WQ.write_json(d / "screen.json", {"arm": ARM, "seed": SEED, "split": sp, "rc": 0, KEY: k,
                                      "wq": {"declared_in": K.DECLARED, "split": sp, "seed": SEED, "arm": ARM,
                                             "gate": gate_sha}})
    WQ.write_json(d / "train.json", {"seconds": 160.0, "peak_gpu_gb": 3.6, "data_gb": 2.0})
    WQ.write_json(d / TIMING_FILE, {"peak_gb": 9.5, "rel_seconds": 2.0})
    WQ.write_json(d / REL_FILE, {"stage": STAGE, "declared_in": DECLARED, "fit": name, "split": sp, **rel})
    (d / "models.pt").write_bytes(b"weights")
    (d / "reads" / "metaqa__s1eval.npz").write_bytes(b"read")
    if read_:
        WQ.write_json(d / "read.json", {"seconds": 13.0, "peak_gpu_gb": 4.6, "carves": {c: {} for c in K.SIX},
                                        KEY: {"declared_in": DECLARED, "fit": name, "arm": ARM, "seed": SEED,
                                              "split": sp, "device": dev, "gate": gate_sha, "rows_cap": rows_cap}})
    return d


def real_state():
    """What G6RW has under the real folders: its fits, its compares and everything under outputs/g6rw."""
    w = WQ.ROOT / NEW
    return (sorted(p.name for p in K.fits_dir(WQ.ROOT).glob("scr-g6rw-*")),
            sorted(p.name for p in K.screen_dir(WQ.ROOT).glob("scr-g6rw-*")),
            sorted((p.relative_to(w).as_posix(), p.stat().st_mtime_ns) for p in w.rglob("*")) if w.is_dir() else [])


def selftest():
    import tempfile
    import types
    t0 = time.time()
    before = real_state()
    stops, TOY = K.stops, K.TOY_ENV
    me = sys.modules[__name__]
    G6R = import_g6r()
    U2N = G6R.U2N
    saved_u2n = (U2N.load_typed, U2N.TYPED_CARVES, U2N.TYPED_ROOT, U2N.build_typed)
    saved_enc = dict(G6R.ENC)
    # 1. the names: G6RW's two fits and compares on WQK's seed-0 zspk and zg6 fits; zg6r's encoder names; every module
    #    the stamps hash
    assert FIT == {"WQ-musique": "scr-g6rw-zg6r", "WQ-hotpotqa": "scr-g6rw-zg6r-hp"}
    assert COMPARES["scr-g6rw-zg6r-vs-zspk-hp"] == ("scr-g6rw-zg6r-hp", ["scr-wqk-zspk-s0-hp", "scr-wqk-zg6-s0-hp"])
    assert WQK_ZRCKP["WQ-musique"] == "scr-wqk-zrckp-s1" and WQK_ZG6_CMP["WQ-hotpotqa"] == "scr-wqk-zg6-vs-zspk-hp"
    assert all((HERE / f"{m}.py").is_file() for m in CODE) and len(set(CODE)) == len(CODE)
    assert ENC_NAMES == tuple(G6R.ENC_NAMES) and set(NEXT) == {"ADOPT_FOR_FULL_RUN", "GAIN_AND_LOSS", "NO_GAIN",
                                                                "INCOMPLETE"}
    # 2. zg6r: G6RW's arm, seed, read cap and encoder, else refused; inside zwq.on_wq it trains on zwq's modules
    with WQ.patch(G6R, READ_CAP=65536):
        stops(import_g6r, has="not G6RW's")
    with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W:
        same_modules(G6R, W)
        other = types.SimpleNamespace(Z3=W.Z3, Z2=W.Z2, LG=json, S=W.S, S2=W.S2, LC=W.LC)
        stops(same_modules, G6R, other, has="does not train on zwq's ['json']")
        assert W.LG.FITS["WQ-hotpotqa"] == WQ.SPLITS["WQ-hotpotqa"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        B = td / "repo"
        # 3. the typed links: webqsp's fit carve from G6RW's root, every other carve from U2n's (zg6r's carve asks
        #    U2n's root); another root refused; U2n's carves and loader restored after, on a failure too
        seen = []

        def rec_load(c, root, cache_root, lk, verify=True):
            seen.append((c.ds, c.carve, Path(root).resolve(), tuple(U2N.TYPED_CARVES)))
            return c.ds, c.carve

        assert Path(G6R.G6rLinkCarve.G6R_TYPED_ROOT).resolve() == Path(U2N.TYPED_ROOT).resolve()
        with WQ.patch(U2N, load_typed=rec_load):
            with on_typed(U2N, B) as U:
                assert U is U2N and ("webqsp", "fit") in U2N.TYPED_CARVES and U2N.load_typed is not rec_load
                for cv in ("fit", "s1eval"):
                    c = types.SimpleNamespace(ds="webqsp", carve=cv)
                    assert U2N.load_typed(c, U2N.TYPED_ROOT, "c", {}) == ("webqsp", cv)
                stops(U2N.load_typed, types.SimpleNamespace(ds="metaqa", carve="fit"), B / TYPED_ROOT, "c", {},
                      has="not U2n's root")
            try:
                with on_typed(U2N, B):
                    raise RuntimeError("inside")
            except RuntimeError:
                pass
            assert U2N.load_typed is rec_load and U2N.TYPED_CARVES == saved_u2n[1]
        assert (U2N.load_typed, U2N.TYPED_CARVES) == saved_u2n[:2]
        assert [s[:3] for s in seen] == [("webqsp", "fit", (B / TYPED_ROOT).resolve()),
                                         ("webqsp", "s1eval", Path(U2N.TYPED_ROOT).resolve())]
        assert ("webqsp", "fit") in seen[0][3] and ("webqsp", "fit") in seen[1][3]
        # 4. the typed records a fit and a read stamp: webqsp's fit carve's at G6RW's root, the KB carves' at U2n's,
        #    none for a passage carve
        ur = td / "u2n_typed"
        for ds, cv in U2N.TYPED_CARVES:
            WQ.write_json(ur / ds / cv / "record.json", {"declared_in": U2N.DECLARED})
        WQ.write_json(B / TYPED_ROOT / "webqsp" / "fit" / "record.json", {"declared_in": U2N.DECLARED, "own": 1})
        with WQ.patch(U2N, TYPED_ROOT=ur):
            got = typed_records([("metaqa", "fit"), ["webqsp", "fit"], ("2wiki", "fit")], U2N, B)
            assert got == {"metaqa/fit": WQ.sha_file(ur / "metaqa" / "fit" / "record.json"),
                           "webqsp/fit": WQ.sha_file(B / TYPED_ROOT / "webqsp" / "fit" / "record.json")}
            stops(typed_records, [("webqsp", "fit")], U2N, td / "none", has="missing")
        shutil.rmtree(B / "outputs")
        # 5. where, the read cap, WQK's verdict and typed.json: train, read and fit refuse before reading any data
        stops(split_of, "scr-wqk-zg6-s0", has="not one of G6RW's fits")
        stops(train, "scr-g6r-zg6r", TOY, has="not one of G6RW's fits")
        for bad in ({"where": "elsewhere"}, {"gpu": "Tesla P100-PCIE-16GB"}, {"image": "kaggle-gpu:other"}):
            stops(where_ok, {**TOY, **bad}, "train", has="G6RW trains in a Kaggle kernel")
            stops(train, FIT["WQ-musique"], {**TOY, **bad}, has="G6RW trains in a Kaggle kernel")
            stops(read, FIT["WQ-hotpotqa"], {**TOY, **bad}, has="G6RW reads in a Kaggle kernel")
        where_ok(TOY, "read")
        cap_ok(READ_CAP)
        for cap in (0, None, 65536):
            stops(cap_ok, cap, has="--rows-cap 32768")
            stops(read, FIT["WQ-musique"], TOY, rows_cap=cap, has="--rows-cap 32768")
            stops(fit, FIT["WQ-musique"], rows_cap=cap, has="--rows-cap 32768")
        stops(fit, FIT["WQ-hotpotqa"], has="G6RW trains in a Kaggle kernel")       # the laptop is not a Kaggle T4
        stops(wqk_ready, B, has="base-verdict.json missing")
        wv = B / (K.VERDICT + ".json")
        for bad in ({"verdict": "INCOMPLETE"}, {"declared_in": WQ.DECLARED}, {"stage": "WQ"}):
            WQ.write_json(wv, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY", **bad})
            stops(wqk_ready, B, has="not WQK's BASE_READY")
        WQ.write_json(wv, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
        assert wqk_ready(B)["verdict"] == "BASE_READY"
        tj = B / TYPED_REC
        stops(own_typed, B, has="typed.json missing")
        WQ.write_json(tj, {"stage": "U2n", "declared_in": DECLARED, "carves": {}})
        stops(own_typed, B, has="is not G6RW's")
        mr = B / TYPED_ROOT / "metaqa" / "fit" / "record.json"
        WQ.write_json(mr, {"x": 1})
        WQ.write_json(tj, {"stage": STAGE, "declared_in": DECLARED, "carves": {"metaqa/fit": {"record": "0" * 64}}})
        stops(own_typed, B, has="is not the record")
        WQ.write_json(tj, {"stage": STAGE, "declared_in": DECLARED, "carves": {"metaqa/fit": {"record": WQ.sha_file(mr)}}})
        stops(own_typed, B, has="pins ['metaqa/fit']")
        shutil.rmtree(B / "outputs")
        # 6. the encoder's start: WQK's seed-0 zspk fit's models.pt, its sha256 pinned; its fit, p@swa, names and
        #    numbers checked; on_encoder makes it zg6r's while active and restores zg6r's state after
        E = td / "enc"
        stops(load_encoder, "WQ-musique", E, has="missing")
        shas = {sp: toy_encoder(E / ENC_ROOT / f"{sp}.pt", ZSPK[sp], seed=i) for i, sp in enumerate(SPLITS)}
        stops(load_encoder, "WQ-musique", E, has="not WQK's scr-wqk-zspk-s0")
        with WQ.patch(me, ENC_SHA=shas):
            st, info = load_encoder("WQ-hotpotqa", E)
            assert set(st) == set(ENC_NAMES) and sum(int(t.numel()) for t in st.values()) == ENC_ELEMENTS
            assert (info["fit"], info["sha256"], info["path"], info["elements"], info["candidate"]) == \
                (ZSPK["WQ-hotpotqa"], shas["WQ-hotpotqa"], f"{ENC_ROOT}/WQ-hotpotqa.pt", ENC_ELEMENTS, "p@swa")
            with on_encoder("WQ-musique", G6R, E) as ei:
                assert G6R.ENC["info"] is ei and set(G6R.ENC["state"]) == set(ENC_NAMES) and G6R.ENC["copies"] == 0
                assert ei["sha256"] == shas["WQ-musique"]
            assert G6R.ENC == saved_enc
            for kw, has in (({"fit_": "scr-kg-base-zspk"}, "seed 0 with a p@swa"), ({"drop": "cm_beta"}, "['cm_beta']"),
                            ({"n": ENC_ELEMENTS + 1}, f"holds {ENC_ELEMENTS + 1} numbers")):
                h = toy_encoder(E / "bad" / ENC_ROOT / "WQ-musique.pt", **{"fit_": ZSPK["WQ-musique"], **kw})
                with WQ.patch(me, ENC_SHA={**shas, "WQ-musique": h}):
                    stops(load_encoder, "WQ-musique", E / "bad", has=has)
        # 7. the fit's stamps after zrm's train: WQK's wq block at the toy root and G6RW's block (typed.json, the
        #    encoder, the records, the head); fit_stamp checks them against the name
        B7 = td / "stamp"
        for k_, r_ in (("GATE", {"verdict": "PASS"}), ("OVERLAY", {"status": "OK"}), ("CHECK", {"status": "REPRODUCED"})):
            WQ.write_json(B7 / K.RECORDS[k_], {"declared_in": K.DECLARED, **r_})
        WQ.write_json(B7 / (K.VERDICT + ".json"), {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
        WQ.write_json(B7 / TYPED_REC, {"stage": STAGE, "declared_in": DECLARED, "carves": {}})
        n7 = FIT["WQ-hotpotqa"]
        WQ.write_json(WQ.fit_dir(n7, B7) / "screen.json", {"arm": ARM, "seed": SEED, "split": "WQ-hotpotqa", "rc": 0})
        stops(fit_stamp, n7, K.fits_dir(B7), has="no G6RW stamp")
        enc7 = {**info, "copies": 2}
        stamp_fit(n7, TOY, G6R, {"kt_records": {"webqsp/fit": "k"}, "typed_records": {"webqsp/fit": "t"}}, enc7, B7)
        s7, k7 = fit_stamp(n7, K.fits_dir(B7))
        assert (k7["typed"], k7["encoder"], k7["device"]) == (WQ.sha_file(B7 / TYPED_REC), enc7, K.device_of(TOY))
        assert (s7["wq"]["declared_in"], s7["wq"]["gate"], s7["wq"]["roots"]) == \
            (K.DECLARED, WQ.sha_file(B7 / K.RECORDS["GATE"]), K.ROOTS)
        assert k7["typed_records"] == {"webqsp/fit": "t"} and k7["gate"] == s7["wq"]["gate"] and s7["rc"] == 0
        assert k7["head"]["typed_roots"] == {"u2n": "outputs/u2n/typed", "g6rw": TYPED_ROOT}
        assert k7["head"]["model"]["typed_carves"][-1] == ["webqsp", "fit"] and k7["head"]["read_rows"] == READ_CAP
        assert "zg6r_sha256" in s7 and s7["zg6rw_sha256"] == WQ.sha_src(__file__) and set(k7["code"]) == set(CODE)
        stops(fit_stamp, FIT["WQ-musique"], K.fits_dir(B7), has="no G6RW stamp")
        # 8. typed (U2n's build_typed, faked here) on the parts WQK's ship.json names, then ship: the encoders and the
        #    typed arrays, each checked against its record; a part on other records, a missing or other encoder, an
        #    array changed after its record, refused; then the relation map of a fit with its encoder's drift
        B8, ef = td / "ship", td / "enc_from"
        on = {p: (f"{i}c" * 32, f"{i}e" * 32) for i, p in enumerate(("part_0of2", "part_1of2"))}
        WQ.write_json(B8 / K.SHIP, {"files": {f"{K.ROOTS[x]}/webqsp/fit/{p}/record.json": {"sha256": on[p][j]}
                                              for p in on for j, x in enumerate(("cache", "links"))}})
        with WQ.patch(U2N, build_typed=fake_build_typed({**on, "part_1of2": (on["part_1of2"][0], "9" * 64)})):
            stops(typed, B8, has="not the ones WQK's ship.json names")
        assert not (B8 / TYPED_REC).exists() and U2N.TYPED_CARVES == saved_u2n[1]
        stops(ship, ef, B8, has="typed.json missing")
        with WQ.patch(U2N, build_typed=fake_build_typed(on)):
            t8 = typed(B8)
        c8 = t8["carves"]["webqsp/fit"]
        assert list(t8["carves"]) == ["webqsp/fit"] and c8["parts"]["part_1of2"]["link_record"] == on["part_1of2"][1]
        assert (c8["n_sets"], c8["edges"], c8["typed"]) == (2, 6, 4) and own_typed(B8) == WQ.sha_file(B8 / TYPED_REC)
        assert t8["roots"]["chunks"] == t8["roots"]["look"] == K.ROOTS["plook"] and t8["roots"]["out"] == TYPED_ROOT
        assert c8["record"] == WQ.sha_file(B8 / TYPED_ROOT / "webqsp" / "fit" / "record.json")
        for sp in SPLITS:
            K.fake_fit(K.fits_dir(B8), ZSPK[sp], "a" * 64, "b" * 64)
            tj8 = K.fits_dir(B8) / ZSPK[sp] / "train.json"
            WQ.write_json(tj8, {**WQ.read_json(tj8), "models_sha256": shas[sp]})
        with WQ.patch(me, ENC_SHA=shas):
            stops(ship, ef, B8, has="missing or not")
            for sp in SPLITS:
                (ef / ZSPK[sp]).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(E / ENC_ROOT / f"{sp}.pt", ef / ZSPK[sp] / "models.pt")
            tj8 = K.fits_dir(B8) / ZSPK["WQ-hotpotqa"] / "train.json"
            keep = tj8.read_bytes()
            WQ.write_json(tj8, {**WQ.read_json(tj8), "models_sha256": "0" * 64})
            stops(ship, ef, B8, has="does not name the encoder")
            tj8.write_bytes(keep)
            s8 = ship(ef, B8)
            want = ({f"{ENC_ROOT}/{sp}.pt" for sp in SPLITS} | {f"{TYPED_ROOT}/webqsp/fit/sets.npy"}
                    | {f"{TYPED_ROOT}/webqsp/fit/{p}/{k}.npy" for p in on for k in ("t_set", "t_dir")})
            assert set(s8["files"]) == want and s8["typed"] == own_typed(B8)
            assert s8["files"][f"{ENC_ROOT}/WQ-musique.pt"]["sha256"] == shas["WQ-musique"]
            assert s8["total_bytes"] == sum((B8 / p).stat().st_size for p in want)
            a8 = B8 / TYPED_ROOT / "webqsp" / "fit" / "part_0of2" / "t_dir.npy"
            keep = a8.read_bytes()
            a8.write_bytes(keep + b"\0")
            stops(ship, ef, B8, has="not its record's")
            a8.write_bytes(keep)
            assert ship(ef, B8)["files"] == s8["files"]
            import torch
            fd = WQ.fit_dir(FIT["WQ-musique"], B8)
            fd.mkdir(parents=True, exist_ok=True)
            torch.save({"states": {"p@swa": {"g6r_pf": torch.zeros(2, 3)}}}, fd / "models.pt")
            asked = []

            def fake_stats(st_, rel_dir=None, show=None, enc=None):
                asked.append((sorted(st_), rel_dir, show, sorted(enc)))
                return G6R.fake_rel()

            with WQ.patch(G6R, rel_stats=fake_stats):
                r8 = rel_record(FIT["WQ-musique"], G6R, B8)
            assert asked == [(["g6r_pf"], None, G6R.REL_SHOW, sorted(ENC_NAMES))]
            assert WQ.read_json(fd / REL_FILE) == r8 and (r8["stage"], r8["fit"], r8["split"]) == \
                (STAGE, FIT["WQ-musique"], "WQ-musique")
        assert U2N.build_typed is saved_u2n[3]
        # 9. take, compare-all and the verdict on a toy of WQK's base set (zwqk's toy, its fits, compares and floors)
        #    and a G6RW session: ADOPT_FOR_FULL_RUN, NOT_ADOPTED both ways, and INCOMPLETE for each missing or wrong piece
        K.toy_build(B)
        gw, ow = K.fake_session(td / "wqk_out", B, tag="w")
        for n in K.FITS:
            K.fake_fit(K.fits_dir(B), n, gw, ow)
        g6 = {("WQ-musique", "metaqa"): 0.02, ("WQ-hotpotqa", "webqsp"): -0.03}

        def wqk_run(new, base_fit, out):                # zwqk's selftest's: WQK's ten compares on the toy
            new, b0, c = Path(new).name, Path(base_fit).name, Path(out).name
            arm, seed, sp = K.FITS[new]
            if arm == "zg6":
                dd = {ds: d for (s_, ds), d in g6.items() if s_ == sp}
            else:
                dd = {ds: (0.002, -0.002)[seed - 1] for ds in WQ.EVAL_ORDER}
                if (arm, sp) == ("zspk", "WQ-musique"):
                    dd["musique"] = (0.02, -0.01)[seed - 1]
            K.fake_compare(B, c, new, b0, dd, base_r5=0.59 if arm == "zrckp" else 0.6)
            return 0

        why = WQ.wq_why
        with WQ.patch(WQ, run_compare=wqk_run, wq_why=lambda name, base=B: why(name, base)):
            assert K.compare_all(B) == 0
        assert K.floors(B) == 0
        WQ.write_json(B / (K.VERDICT + ".json"), {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
        GB2 = G6R.GB2
        gp = B / GB2.OUT / f"{GB2.VERDICT}.json"
        WQ.write_json(gp, {"stage": GB2.STAGE, "declared_in": GB2.DECLARED, "verdict": "NOT_ADOPTED",
                           "next_form": GB2.NEXT["NO_GAIN"]})
        WQ.write_json(B / TYPED_REC, {"stage": STAGE, "declared_in": DECLARED, "carves": {}})
        tsha, rel = WQ.sha_file(B / TYPED_REC), G6R.fake_rel()
        out_g = td / "g6rw_out"
        gg, og = K.fake_session(out_g, B, tag="g")
        fake_fit(out_g / "fits", FIT["WQ-musique"], gg, og, tsha, rel)
        fake_fit(out_g / "fits", FIT["WQ-hotpotqa"], gg, og, tsha, rel, read_=False)
        took, left = take(out_g, B)
        assert len(took) == 6 + 2 and len(left) == 1 and "has no G6RW read" in left[0], (took, left)
        fr = K.fits_dir(B)
        assert not (fr / FIT["WQ-musique"] / "models.pt").exists() and (fr / FIT["WQ-musique"] / REL_FILE).is_file()
        assert (B / SESSIONS / gg[:12] / "gate.json").is_file() and not (fr / FIT["WQ-hotpotqa"]).exists()
        fake_fit(out_g / "fits", FIT["WQ-hotpotqa"], gg, og, tsha, rel)
        took, left = take(out_g, B)
        assert len(took) == 6 and not left and take(out_g, B) == ([], [])
        (fr / FIT["WQ-musique"] / "train.json").write_text("{}", encoding="utf-8")
        stops(take, out_g, B, has="exists with other bytes")
        shutil.copyfile(out_g / "fits" / FIT["WQ-musique"] / "train.json", fr / FIT["WQ-musique"] / "train.json")
        (out_g / "fits" / ZG6["WQ-musique"]).mkdir()
        stops(take, out_g, B, has="are not G6RW's fits")
        (out_g / "fits" / ZG6["WQ-musique"]).rmdir()
        stops(take, ef, B, has="holds no wqk/gate.json")
        cur = {}

        def g6rw_run(new, base_fit, out):               # lean_screen's compare with two bases, in its record's form
            import lean_screen as S
            n_, c = Path(new).name, Path(out).name
            sp = SPLIT_OF[n_]
            bs = [Path(b).name for b in base_fit.split(",")]
            assert bs == [ZSPK[sp], ZG6[sp]] and c == CMP[sp], (bs, c)
            by = {}
            for b in bs:
                rows = []
                for ds in WQ.EVAL_ORDER:
                    nw = round(0.6 + cur.get((sp, ds), 0.0), 4)
                    r5 = 0.6 if b == ZSPK[sp] else round(0.6 + g6.get((sp, ds), 0.0), 4)
                    d = round(nw - r5, 4)
                    lo, hi = round(d - 0.004, 4), round(d + 0.004, 4)
                    rows.append({"dataset": ds, "read": "in-domain" if ds in WQ.SPLITS[sp] else "zero-shot",
                                 "questions": 100, "base": [r5, 0.5, 0.4], "new": [nw, 0.5, 0.4],
                                 "rrf": [0.4, 0.3, 0.2], "delta": [d, 0, 0], "lo": [lo, 0, 0], "hi": [hi, 0, 0],
                                 "call": S.call(d, lo, hi)})
                by[f"outputs/screen/fits/{b}"] = {"rows": rows, "verdict": S.verdict([r["call"] for r in rows]),
                                                  "decides": b == bs[0]}
            bp = list(by)
            WQ.write_json(K.screen_dir(B) / f"{c}.json", {
                "new": f"outputs/screen/fits/{n_}", "bases": bp, "candidate": "p@swa", "carve": "s1eval",
                "trained_on": sorted(WQ.SPLITS[sp]), "by_base": by, "verdict": by[bp[0]]["verdict"]})
            return 0

        def run(deltas):
            cur.clear()
            cur.update(deltas)
            for c in COMPARES:
                (K.screen_dir(B) / f"{c}.json").unlink(missing_ok=True)
            with WQ.patch(WQ, run_compare=g6rw_run):
                assert compare_all(B) == 0 and compare_all(B) == 0
            return verdict(B)

        v = run({("WQ-hotpotqa", "squad"): 0.03})
        assert v["verdict"] == "ADOPT_FOR_FULL_RUN" and not v["problems"], v["problems"]
        assert (v["gains"], v["losses"], v["next_form"]) == (["WQ-hotpotqa squad"], [], NEXT["ADOPT_FOR_FULL_RUN"])
        assert v["gates"] == [gg] and v["device"] == K.device_of(TOY) and len(v["calls"]) == 12
        k_ = WQ.read_json(K.screen_dir(B) / f"{CMP['WQ-hotpotqa']}.json")[KEY]
        assert (k_["new"], k_["bases"], k_["device"]) == (FIT["WQ-hotpotqa"], [ZSPK["WQ-hotpotqa"], ZG6["WQ-hotpotqa"]],
                                                          K.device_of(TOY))
        b = v["beside"]
        assert b["vs_zg6"] == {"gains": ["WQ-hotpotqa squad", "WQ-hotpotqa webqsp"], "losses": ["WQ-musique metaqa"]}
        assert b["mlp_share"]["WQ-hotpotqa squad"] == 0.8261 and b["mlp_share"]["WQ-musique 2wiki"] == 0.95
        assert b["r5"]["WQ-musique metaqa"] == {"read": "zero-shot", "rrf": 0.4, "zspk": 0.6, "zg6": 0.62, "zg6r": 0.6,
                                                "zrckp": 0.59, "floor": 0.0075, "call": "WITHIN", "call_vs_zg6": "LOSS"}
        assert b["r5"]["WQ-musique musique"]["floor"] == round(2 * (((0.02 ** 2 + 0.01 ** 2) / 2) ** 0.5), 6)
        assert b["rel"]["WQ-hotpotqa"]["fit"] == FIT["WQ-hotpotqa"] and b["timing"]["WQ-musique"]["zg6r"] == {
            "train_seconds": 160.0, "read_seconds": 13.0, "train_peak_gpu_gb": 3.6, "read_peak_gpu_gb": 4.6,
            "data_gb": 2.0, "peak_gb": 9.5, "rel_seconds": 2.0}
        assert b["timing"]["WQ-hotpotqa"]["zg6"]["peak_gb"] == 8.0
        assert b["recall_sha256"] == WQ.sha_file(B / (RECALL[0] + ".json"))
        for which in AGAINST:
            r_ = WQ.read_json(B / (RECALL[which] + ".json"))
            assert (r_["stage"], r_["against"], r_["deciding"], len(r_["rows"])) == \
                (STAGE, AGAINST[which], which == 0, 12)
            assert (B / (PAIR[which] + ".md")).read_text(encoding="utf-8").startswith(f"G6RW ({DECLARED})")
        r1 = WQ.read_json(B / (RECALL[1] + ".json"))
        assert {(r["split"], r["dataset"]): r["base"] for r in r1["rows"]}[("WQ-musique", "metaqa")] == 0.62
        md = (B / (VERDICT + ".md")).read_text(encoding="utf-8")
        assert "zg6r **ADOPT_FOR_FULL_RUN**" in md and "| WQ-hotpotqa squad (in-domain) | 0.4000 | 0.6000 |" in md
        v = run({("WQ-hotpotqa", "squad"): 0.03, ("WQ-musique", "webqsp"): -0.03})
        assert (v["verdict"], v["losses"], v["next_form"]) == ("NOT_ADOPTED", ["WQ-musique webqsp"],
                                                               NEXT["GAIN_AND_LOSS"])
        v = run({("WQ-musique", "musique"): 0.02})          # a filed GAIN within the raised floor (0.0316)
        assert (v["verdict"], v["gains"], v["losses"], v["next_form"]) == ("NOT_ADOPTED", [], [], NEXT["NO_GAIN"])
        assert v["calls"]["WQ-musique musique"] == "WITHIN" and v["delta_r5"]["WQ-musique musique"] == 0.02
        r0 = WQ.read_json(B / (RECALL[0] + ".json"))
        assert [r["call_filed"] for r in r0["rows"] if (r["split"], r["dataset"]) == ("WQ-musique", "musique")] == \
            ["GAIN"]

        def incomplete(has):
            r = verdict(B)
            assert r["verdict"] == "INCOMPLETE" and r["beside"] == {} and r["next_form"] == NEXT["INCOMPLETE"] and \
                any(has in p for p in r["problems"]), (has, r["problems"])

        def away(p, has):
            keep_ = p.read_bytes()
            p.unlink()
            incomplete(has)
            p.write_bytes(keep_)

        away(K.screen_dir(B) / f"{CMP['WQ-hotpotqa']}.json", f"{CMP['WQ-hotpotqa']}.json missing")
        away(B / (K.VERDICT + ".json"), "base-verdict.json missing")
        away(gp, "G6b2's verdict")
        keep = gp.read_bytes()
        WQ.write_json(gp, {**WQ.read_json(gp), "next_form": "G6b3"})
        incomplete("not NOT_ADOPTED with G6r next")
        gp.write_bytes(keep)
        n0 = FIT["WQ-musique"]
        for kw, has in (({"dev": {**K.device_of(TOY), "gpu": "Tesla P100"}}, "ran on different devices"),
                        ({"gate_sha": "f" * 64}, "names the gate ffffffffffff, not a filed G6RW PASS gate"),
                        ({"enc_sha": "0" * 64}, f"not trained from WQK's {ZSPK['WQ-musique']} encoder"),
                        ({"typed_sha": "1" * 64}, "not trained on the typed links"),
                        ({"rows_cap": 65536}, "has no G6RW read on the device it was fitted on")):
            fake_fit(fr, n0, **{"gate_sha": gg, "overlay_sha": og, "typed_sha": tsha, "rel": rel, **kw})
            incomplete(has)
        fake_fit(fr, n0, gg, og, tsha, rel)
        assert verdict(B)["verdict"] == "NOT_ADOPTED"
        out2 = td / "g6rw_out2"
        g2, _o2 = K.fake_session(out2, B, verdict_="FAIL", tag="h")
        take(out2, B)
        incomplete("FAIL, not WQK's PASS")
        shutil.rmtree(B / SESSIONS / g2[:12])
        p = K.screen_dir(B) / "scr-wqk-zspk-s1.json"
        keep = p.read_bytes()
        r = WQ.read_json(p)
        r["by_base"]["outputs/screen/fits/scr-wqk-zspk-s0"]["rows"][2]["delta"][0] = 0.05
        WQ.write_json(p, r)
        incomplete("floors-zspk.json is not its compares' floors")
        p.write_bytes(keep)
        assert verdict(B)["verdict"] == "NOT_ADOPTED"
    # 10. nothing left patched in zu2n or zg6r, and nothing written under G6RW's real folders
    assert (U2N.load_typed, U2N.TYPED_CARVES, U2N.TYPED_ROOT, U2N.build_typed) == saved_u2n and G6R.ENC == saved_enc
    assert real_state() == before, (real_state(), before)
    print(f"zg6rw selftest: ok ({time.time() - t0:.1f}s)", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    import argparse
    ap = argparse.ArgumentParser(prog="zg6rw.py", allow_abbrev=False)
    ap.add_argument("cmd", choices=("typed", "ship", "fit", "take", "compare-all", "verdict"))
    ap.add_argument("--name")
    ap.add_argument("--rows-cap", type=int, default=READ_CAP)
    ap.add_argument("--from", dest="src")
    ap.add_argument("--enc-from")
    a = ap.parse_args(argv)
    src = Path(a.src).resolve() if a.src else None
    enc_from = Path(a.enc_from).resolve() if a.enc_from else None
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
    if a.cmd == "ship":
        if enc_from is None:
            ap.error("ship takes --enc-from")
        ship(enc_from)
        return 0
    if a.cmd == "typed":
        typed()
        return 0
    if a.cmd == "compare-all":
        return compare_all()
    return 0 if verdict()["verdict"] != "INCOMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
