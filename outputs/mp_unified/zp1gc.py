"""P1g-c (docs/P1GC_ADOPTED_ON_P1GB.md): P1g-b's pools under the adopted models.

P1g-b refit the lean pair (zrckp, zspk) on P1g2's chosen pools (ALL >= 0.95 on every dataset) and was NOT_ADOPTED: the
MLP had a GAIN (L-musique webqsp) and no LOSS, the GNN a LOSS (L-hotpotqa musique). That closes only that form
(docs/SCREENS.md, 3b). Since then two model changes passed their screens on P1c2's pools: G5's zspx (zsp's step end to
end, G3's residual message, G4's second step) and R1's zrckpr (R1's nine residual columns through a pointwise head).
Both send their arm to P1g-b's pools next. P1g-c fits them there, plus zspxr, and makes two calls:

    pools   zspx and zrckpr on P1g-b's pools against the same arm on P1c2's pools (scr-g5-zspx, scr-r1-zrckpr and
            -hp), under P1g-b's joint rule: decides the pools for the adopted models
    input   zspxr against zspx, both on P1g-b's pools: decides whether R1's columns reach the GNN (F1's one input set)

and reports each arm against P1g-b's own lean-pair fit on the same pools beside. The arms, exactly as filed, on P1g-b's
roots (zp1gb.on_p1gb):

    zspx    zg5mix's model on G3's carve (zspk's with c and s beside), G3's builds on P1g-b's parts (outputs/p1gc/res)
    zrckpr  zr1res's model on R1's MLP carve (zrckp's with the columns beside), R1's builds on them (outputs/p1gc/cols)
    zspxr   zspx with R1's head added to the score before zspx's first step (zspkr's placement; MRO ZSPX, ZSPR, ZSPK,
            ZProp, R1Mix, KTMix, ZRM), on G3's carve with R1's columns beside as well

P1g-b's links, KT columns and chains (and the looks they come from) were cleared by the host's clean-up after its
verdict; its caches were kept. They are built again by P1g-b's own items, unchanged. Before that, `snap` copies every
record P1g-b's fits read (checked against the record sha256s those fits stamped) to outputs/p1gc/p1gb_records; after
it, `gate` opens only if every rebuilt record holds the same arrays (each array's sha256 and shape) and counts. The
builds, fits and reads refuse while it is shut, and refuse a record rebuilt after it opened.

    python outputs/mp_unified/zp1gc.py snap
    python outputs/mp_unified/zp1gc.py gate
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zp1gc.py build --what res --dataset musique --carve s1fit \\
        --part 0 --host
    python outputs/mp_unified/zp1gc.py build --what cols --dataset musique --carve s1fit --part 0 --host
    python outputs/mp_unified/zp1gc.py train --split L-musique --name scr-p1gc-zspx --arm zspx --device cuda --host \\
        --cache-root outputs/p1gb/cache
    python outputs/mp_unified/zp1gc.py read --name scr-p1gc-zspx --device cuda --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/zp1gc.py compare --new outputs/screen/fits/scr-p1gc-zspx \\
        --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-p1gc-pools-zspx
    python outputs/mp_unified/zp1gc.py pair --call pools --arm zspx --screens S1.json,S2.json --out ...
    python outputs/mp_unified/zp1gc.py recall --call pools --arm zspx --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/zp1gc.py verdict
    python outputs/mp_unified/zp1gc.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if sys.argv[1:2] == ["build"]:                            # one CPU per build item
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zp1gb as GB  # noqa: E402  (P1g-b's roots; through it zp1c3p, zp1c3, zp1c2)
import zg5mix as G5  # noqa: E402  (zspx; through it zg3res and zg4prop)
import zr1res as R1  # noqa: E402  (zrckpr, R1Mix, R1's carves and builds)

import numpy as np  # noqa: E402
import torch  # noqa: E402

G3, G4 = G5.G3, G5.G4
Z3, Z3P, Z2 = GB.Z3, GB.Z3P, GB.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/P1GC_ADOPTED_ON_P1GB.md"
OUT = ROOT / "outputs" / "p1gc"
RES, COLS = OUT / "res", OUT / "cols"
SNAP = OUT / "p1gb_records"
GATE = OUT / "gate"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = Z3.SPLITS
SX = {"L-musique": "", "L-hotpotqa": "-hp"}
ARM_NAMES = ("zspx", "zspxr", "zrckpr")
NEEDS = {"zspx": ("res",), "zspxr": ("res", "cols"), "zrckpr": ("cols",)}
WHAT = {"zspx": "G5's GNN as filed", "zspxr": "zspx with R1's head before its first step",
        "zrckpr": "R1's MLP as filed"}
FIT = {a: {sp: f"scr-p1gc-{a}{SX[sp]}" for sp in SPLITS} for a in ARM_NAMES}
# P1g-b's records, as its fits read them: each family's root and record pattern
RECS = {"links": (GB.LINKS, "*/*/part_*/record.json"), "rmatch": (GB.CH_RM, "*/*/part_*/record.json"),
        "zrc": (GB.CH_ZRC, "*/*/part_*/record.json"), "kt": (GB.KT_ROOT, "*/*/record.json")}
STAMP_KEY = {"links": "zlink_records", "rmatch": "rmatch_records", "zrc": "zrc_records", "kt": "kt_records"}
SAME_KEYS = ("arrays", "queries", "rows")
LEAN = {"zrckp": "zrckp", "zspk": "zspk"}
LEAN_FITS = tuple(f"scr-p1gb-{a}{SX[sp]}" for a in ("zrckp", "zspk") for sp in SPLITS)

# the two calls: (call, arm) -> the base arm, its fits and its base screens (each base fit against step 1's)
CALLS = {"pools": ("zspx", "zrckpr"), "input": ("zspxr",)}
P1C2_FIT = {"zspx": "scr-g5-zspx", "zrckpr": "scr-r1-zrckpr"}
BASE_ARM = {("pools", "zspx"): "zspx", ("pools", "zrckpr"): "zrckpr", ("input", "zspxr"): "zspx"}
CALL_FITS = {("pools", a): {sp: ("screen", "fits", f"{P1C2_FIT[a]}{SX[sp]}") for sp in SPLITS} for a in CALLS["pools"]}
CALL_FITS[("input", "zspxr")] = {sp: ("screen", "fits", FIT["zspx"][sp]) for sp in SPLITS}
CALL_SCREENS = {k: {sp: f"outputs/p1gc/base-{k[0]}-{BASE_ARM[k]}-{sp}.json" for sp in SPLITS} for k in CALL_FITS}
AGAINST = {("pools", "zspx"): "zspx's card fit of each split on P1c2's pools (G5's scr-g5-zspx, -hp)",
           ("pools", "zrckpr"): "zrckpr's card fit of each split on P1c2's pools (R1's scr-r1-zrckpr, -hp)",
           ("input", "zspxr"): "zspx's card fit of each split on P1g-b's pools (P1g-c's scr-p1gc-zspx, -hp)"}
PAIR = {k: SCREEN / f"scr-p1gc-{k[0]}-{k[1]}-pair" for k in CALL_FITS}
RECALL = {k: SCREEN / f"scr-p1gc-{k[0]}-{k[1]}-pair-recall" for k in CALL_FITS}
COMPARE = {k: {sp: SCREEN / f"scr-p1gc-{k[0]}-{k[1]}{SX[sp]}" for sp in SPLITS} for k in CALL_FITS}
LEAN_OF = {"zspx": "zspk", "zspxr": "zspk", "zrckpr": "zrckp"}
BESIDE_CMP = {a: {sp: SCREEN / f"scr-p1gc-lean-{a}{SX[sp]}" for sp in SPLITS} for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
CLAIM1_READ = ("L-hotpotqa", "musique")
POOLS_RULE = ("pools: ADOPT_FOR_FULL_RUN when neither zspx's nor zrckpr's pools re-call under the seed null has a LOSS on "
              "any of its twelve reads and the two together have at least one GAIN (P1g-b's rule, with the adopted arms); "
              "NOT_ADOPTED otherwise; INCOMPLETE when a re-call is missing, not P1g-c's pools re-call of that arm, or not "
              "twelve reads (docs/P1GC_ADOPTED_ON_P1GB.md, section 4)")
INPUT_RULE = ("input: ADOPT_FOR_FULL_RUN when zspxr's re-call against zspx has no LOSS on any of its twelve reads and at "
              "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not P1g-c's input re-call, or "
              "not twelve reads (docs/P1GC_ADOPTED_ON_P1GB.md, section 4)")
NEXT_POOLS = {"ADOPT_FOR_FULL_RUN": "the full run on P1g-b's pools (its own file, declared before its numbers): six splits "
                                    "per arm under grade-nullx, MLP zrckpr, GNN as the input call says",
              "NOT_ADOPTED": "the pools stay P1c2's for the adopted models; the full run of zrckpr and zspx on P1c2's "
                             "pools (its own file); and the pools' next form P1g-b3, each row's place in P1g-b's pool "
                             "(today's fill, the fill beyond it, the extra set) as an input to both models through R1's "
                             "pointwise head, decided as P1g-b was (its own file)"}
NEXT_INPUT = {"ADOPT_FOR_FULL_RUN": "R1's columns reach both models: the full run's GNN is zspxr where the pools are "
                                    "P1g-b's; on P1c2's pools zspxr's two screen fits come first in that file, decided "
                                    "against G5's zspx fits by this rule",
              "GAIN_AND_LOSS": "zspxrb: R1's head bounded, 0.2 tanh(h / 0.2), on zspx (one declared file)",
              "NO_GAIN": "zspxs: R1's nine columns read by zspx's first step head beside zprop's inputs, so the step "
                         "weighs a row's neighbours by its residual (one declared file)"}


# ── the third arm and its carve ──────────────────────────────────────────────


class ZSPXR(G5.ZSPX, R1.R1Mix):
    """zspxr: zspx whose first step reads the score with R1's head added (MRO ZSPX, ZSPR, ZSPK, ZProp, R1Mix, KTMix,
    ZRM: ZSPX.forward's super(ZProp) score is R1Mix's, the KT head's score plus R1's head)."""


R1ResKTLinkCarveBase = R1.r1_carve(G3.ResKTLinkCarveBase)
ARMS = {"zspx": G5.ARMS["zspx"], "zspxr": (ZSPXR, R1ResKTLinkCarveBase), "zrckpr": R1.ARMS["zrckpr"]}
S.ARMS["zspxr"] = ARMS["zspxr"]


# ── P1g-c's roots ────────────────────────────────────────────────────────────


@contextlib.contextmanager
def on_p1gc(res=RES, cols=COLS):
    """P1g-b's roots (zp1gb.on_p1gb), with G3's residual root and R1's column root on P1g-c's builds; all restored
    after."""
    saved = (G3.ResKTLinkCarveBase.RES_ROOT, R1.R1KTChainCarveZRC.R1_ROOT, R1ResKTLinkCarveBase.R1_ROOT)
    G3.ResKTLinkCarveBase.RES_ROOT = res
    R1.R1KTChainCarveZRC.R1_ROOT = R1ResKTLinkCarveBase.R1_ROOT = cols
    try:
        with GB.on_p1gb():
            yield
    finally:
        G3.ResKTLinkCarveBase.RES_ROOT, R1.R1KTChainCarveZRC.R1_ROOT, R1ResKTLinkCarveBase.R1_ROOT = saved


# ── P1g-b's records: the snapshot and the gate ───────────────────────────────


def current(roots=None):
    """{family: {ds/carve[/part]: the record's sha256}} under P1g-b's roots (or the roots given)."""
    roots = roots or {f: r for f, (r, _) in RECS.items()}
    out = {}
    for fam, (_, pat) in RECS.items():
        root = Path(roots[fam])
        out[fam] = {str(f.parent.relative_to(root)).replace("\\", "/"): LC.sha_file(f)
                    for f in sorted(root.glob(pat))}
    return out


def snap(out=None, roots=None, fits=None):
    """Copy every P1g-b record to outputs/p1gc/p1gb_records, after checking that they are the records P1g-b's four
    fits stamped (zlink's, rmatch's and zrc's in full, the KT records of each fit's training carves)."""
    out = Path(out or SNAP)
    roots = roots or {f: r for f, (r, _) in RECS.items()}
    fits = fits or [SCREEN / "fits" / f / "screen.json" for f in LEAN_FITS]
    cur = current(roots)
    probs = [f"no {fam} records under {roots[fam]}" for fam in RECS if not cur[fam]]
    for sj in fits:
        sj = Path(sj)
        st = json.loads(sj.read_text(encoding="utf-8")).get("p1gb") if sj.exists() else None
        if not st:
            probs.append(f"{sj.parent.name}: no P1g-b stamp")
            continue
        for fam, key in STAMP_KEY.items():
            got = st.get(key) or {}
            if fam == "kt":
                bad = sorted(k for k, v in got.items() if cur["kt"].get(k) != v)
                if not got or bad:
                    probs.append(f"{sj.parent.name}: KT records {bad or 'missing'} are not the ones on disk")
            elif got != cur[fam]:
                diff = sorted(set(got) ^ set(cur[fam]) | {k for k in got if cur[fam].get(k) != got[k]})
                probs.append(f"{sj.parent.name}: {len(diff)} {fam} records are not the ones it stamped ({diff[:3]})")
    if probs:
        raise SystemExit("zp1gc snap: " + "; ".join(probs))
    for fam, recs in cur.items():
        for rel in recs:
            dst = out / fam / rel / "record.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(roots[fam]) / rel / "record.json", dst)
    rec = {"stage": "P1gc", "declared_in": DECLARED, "records": cur,
           "checked_against": [Path(f).parent.name for f in fits], "zp1gc_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LC.write_json(out / "snap.json", rec)
    log(f"zp1gc snap: {sum(len(v) for v in cur.values())} records ({', '.join(f'{k} {len(v)}' for k, v in cur.items())})"
        f" checked against {len(fits)} fits' stamps and copied to {out}")
    return rec


def gate(out=None, snap_dir=None, roots=None):
    """0 (OPEN) when every record of the snapshot is on disk again with the same arrays and counts, and no other record
    is; 1 (SHUT) otherwise. The record (outputs/p1gc/gate.{json,md}) keeps each record's sha256 as rebuilt."""
    snap_dir = Path(snap_dir or SNAP)
    roots = roots or {f: r for f, (r, _) in RECS.items()}
    probs, same = [], {}
    sj = snap_dir / "snap.json"
    if not sj.exists():
        probs.append(f"{sj} missing (zp1gc.py snap)")
        old = {}
    else:
        old = json.loads(sj.read_text(encoding="utf-8"))["records"]
    cur = current(roots)
    for fam in RECS:
        o, c = old.get(fam, {}), cur[fam]
        extra = sorted(set(c) - set(o))
        if extra:
            probs.append(f"{len(extra)} {fam} records not in the snapshot ({extra[:3]})")
        n = 0
        for rel in o:
            if rel not in c:
                probs.append(f"{fam} {rel} not built again")
                continue
            a = json.loads((snap_dir / fam / rel / "record.json").read_text(encoding="utf-8"))
            b = json.loads((Path(roots[fam]) / rel / "record.json").read_text(encoding="utf-8"))
            bad = [k for k in SAME_KEYS if a.get(k) != b.get(k)]
            if bad:
                probs.append(f"{fam} {rel}: {', '.join(bad)} differ from P1g-b's")
            else:
                n += 1
        same[fam] = n
    g = "OPEN" if not probs else "SHUT"
    rec = {"stage": "P1gc", "declared_in": DECLARED, "gate": g, "problems": probs, "same": same, "now": cur,
           "snap_sha256": LC.sha_file(sj) if sj.exists() else None, "zp1gc_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# P1g-c ({DECLARED}): gate **{g}**", "",
          "P1g-b's links, KT columns and chains, built again, against the records P1g-b's fits read: "
          + ", ".join(f"{k} {v} of {len(old.get(k, {}))} the same" for k, v in same.items()) + "."]
    if probs:
        md += ["", "Shut: " + "; ".join(probs[:20]) + (f"; and {len(probs) - 20} more" if len(probs) > 20 else "") + "."]
    out = Path(out or GATE)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zp1gc gate: {g}" + (f" ({'; '.join(probs[:5])})" if probs else f" ({same})"))
    return 0 if g == "OPEN" else 1


def gate_open(gate_file=None, roots=None):
    """The gate's record, when it is OPEN and nothing was built again after it opened; stops otherwise."""
    gp = Path(gate_file or GATE).with_suffix(".json")
    if not gp.exists():
        raise SystemExit(f"zp1gc: {gp} missing (zp1gc.py gate); P1g-c neither builds, trains nor reads")
    rec = json.loads(gp.read_text(encoding="utf-8"))
    if rec.get("stage") != "P1gc" or rec.get("declared_in") != DECLARED or rec.get("gate") != "OPEN":
        raise SystemExit(f"zp1gc: the gate is {rec.get('gate')} ({rec.get('stage')}); P1g-c neither builds, trains nor "
                         f"reads")
    if current(roots) != rec.get("now"):
        raise SystemExit("zp1gc: P1g-b's records changed after the gate opened; P1g-c neither builds, trains nor reads")
    return rec


# ── the builds ───────────────────────────────────────────────────────────────


def build(what, ds, carve, part, host, gate_file=None, roots=None):
    """One step-1 part of P1g-b's caches: G3's c and s (what=res) or R1's nine columns from them (what=cols)."""
    if what not in ("res", "cols"):
        raise SystemExit(f"zp1gc: build --what res or cols, not {what}")
    if ds not in LG.EVAL_ORDER or part is None or int(part) < 0:
        raise SystemExit(f"zp1gc: build takes one of the six datasets and a part, not {ds} part {part}")
    gate_open(gate_file, roots)
    place = {"where": "host" if host else "laptop", "stage": "P1gc", "declared_in": DECLARED}
    with GB.on_p1gb():
        if what == "res":
            nodes, freeze = G3.open_nodes(ds, host)
            return G3.build(ds, carve, int(part), out_root=RES, nodes=nodes, freeze=freeze, placement=place)
        return R1.build(ds, carve, int(part), out_root=COLS, res_root=RES, placement=place)


def records(arm, carves):
    """The G3 and R1 records the arm's carves need, from P1g-c's roots on P1g-b's caches."""
    out = {}
    if "res" in NEEDS[arm]:
        out["res_records"] = G3.res_records(carves, RES, GB.CACHE)
    if "cols" in NEEDS[arm]:
        out["r1_records"] = R1.r1_records(carves, COLS, GB.CACHE)
    return out


# ── train, read, compare ─────────────────────────────────────────────────────


def stamp(argv, arm, split, g):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zp1gc_sha256": LC.sha_src(__file__), "zp1gb_sha256": LC.sha_src(GB.__file__),
                    "zg5mix_sha256": LC.sha_src(G5.__file__), "zr1res_sha256": LC.sha_src(R1.__file__),
                    "zg3res_sha256": LC.sha_src(G3.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__), "p1gc": dict({
                        "declared_in": DECLARED, "arm": arm, "what": WHAT[arm], "device": Z2.DEVICE,
                        "pools": "P1g-b's (outputs/p1gb/pools)", "cache_root": GB.CACHE.relative_to(ROOT).as_posix(),
                        "res_root": RES.relative_to(ROOT).as_posix(), "cols_root": COLS.relative_to(ROOT).as_posix(),
                        "p1gb_gate": LC.sha_file(GB.GATE), "gate": LC.sha_file(Path(GATE).with_suffix(".json")),
                        "snap": g.get("snap_sha256"), "kt_records": Z3.kt_records(Z3.train_carves(split), GB.KT_ROOT),
                        "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(GB.CH_ZRC),
                        "zlink_records": ZL.built_records()}, **records(arm, Z3.train_carves(split)))})
        LC.write_json(sj, rec)


def train(argv, split, gate_file=None, roots=None):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zp1gc: train takes --arm {', '.join(ARM_NAMES)}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"zp1gc: P1g-c fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"zp1gc: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    with on_p1gc():
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv)
        g = gate_open(gate_file, roots)
        GB.gate_passed()
        Z3.kt_records(Z3.train_carves(split), GB.KT_ROOT)
        records(arm, Z3.train_carves(split))
        rc = RM.train(argv, split)
        stamp(argv, arm, split, g)
    return rc


def read(argv, gate_file=None, roots=None):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") not in ARMS or "p1gc" not in rec:
        raise SystemExit(f"zp1gc: {name} is not a P1g-c fit (arm {rec.get('arm')})")
    with on_p1gc():
        Z2.cache_root_of(argv)
        gate_open(gate_file, roots)
        Z3.kt_records(Z3.eval_carves(), GB.KT_ROOT)
        records(rec["arm"], Z3.eval_carves())
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, per call and arm ────────────────────────────────


def key_of(call, arm):
    k = (call, arm)
    if k not in CALL_FITS:
        raise SystemExit(f"zp1gc: the calls are pools (zspx, zrckpr) and input (zspxr), not {call} {arm}")
    return k


@contextlib.contextmanager
def on_base(call, arm):
    k = key_of(call, arm)
    fits = CALL_FITS[k]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"zp1gc: no base fit of {BASE_ARM[k]} on {split} for the {call} call")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_ARM[k], fits, CALL_SCREENS[k], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def pools_of(k):
    return ("P1c2's", "P1g-b's") if k[0] == "pools" else ("P1g-b's", "P1g-b's")


def fix(k):
    b, f = BASE_ARM[k], CALL_FITS[k]["L-musique"][2]
    pb, pn = pools_of(k)
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({b}'s card fits {f} and {f}-hp, {pb} pools)"),
            ("(rel's screen fits)", f"({b}'s card fits on {pb} pools)"),
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 ({pb}) | {k[1]} R@5 ({pn}) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, k):
    out = Path(out)
    md = out.with_suffix(".md")
    pb, pn = pools_of(k)
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(k):
            t = t.replace(a, b)
        md.write_text(f"P1g-c ({DECLARED}), the {k[0]} call: {k[1]} ({WHAT[k[1]]}) on {pn} pools against "
                      f"{BASE_ARM[k]} on {pb} pools.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "P1gc", "call": k[0], "declared_in": DECLARED, "decided_against": AGAINST[k],
                    "device": Z2.DEVICE, "zp1gc_sha256": LC.sha_src(__file__)})
        rec.pop("round", None)
        LC.write_json(js, rec)


def pair(call, arm, screens, out):
    with on_base(call, arm):
        rec = Z.pair(screens, out)
    restamp(out, (call, arm))
    return rec


def recall(call, arm, null_files, pair_file, out, base_screens=None):
    with on_base(call, arm):
        rec = Z.recall(null_files, pair_file, out, base_screens)
    restamp(out, (call, arm))
    return rec


# ── the verdict ──────────────────────────────────────────────────────────────


def tag(r):
    return f"{r['split']} {r['dataset']}"


def read_recall(k, path):
    rp, problems, rows = Path(path), [], []
    if not rp.exists():
        problems.append(f"{rp.name} missing")
    else:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if (rec.get("stage") != "P1gc" or rec.get("declared_in") != DECLARED or rec.get("call") != k[0]
                or rec.get("arm") != k[1]):
            problems.append(f"{rp.name} is not P1g-c's {k[0]} re-call of {k[1]}")
        rows = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    return {"gains": [tag(r) for r in rows if r.get("call") == "GAIN"],
            "losses": [tag(r) for r in rows if r.get("call") == "LOSS"], "problems": problems,
            "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
            "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                       if r.get("dataset") in BESIDE},
            "recall_sha256": LC.sha_file(rp) if rp.exists() else None}


def deltas_of(path):
    """{dataset: R@5 new minus base} of a compare's first base, or {} when the compare is not there."""
    p = Path(path).with_suffix(".json")
    if not p.exists():
        return {}
    rec = json.loads(p.read_text(encoding="utf-8"))
    return {r["dataset"]: round(r["new"][0] - r["base"][0], 6) for r in rec["by_base"][rec["bases"][0]]["rows"]}


def full_run(pv, iv):
    """The full run P1g-c leads to, or None while a call is INCOMPLETE."""
    if "INCOMPLETE" in (pv, iv):
        return None
    if pv == "ADOPT_FOR_FULL_RUN":
        return {"pools": "P1g-b's", "mlp": "zrckpr", "gnn": "zspxr" if iv == "ADOPT_FOR_FULL_RUN" else "zspx"}
    return {"pools": "P1c2's", "mlp": "zrckpr", "gnn": "zspx",
            "zspxr": ("its two screen fits on P1c2's pools first, against G5's zspx fits" if iv == "ADOPT_FOR_FULL_RUN"
                      else "not carried")}


def verdict(recalls=None, lean=None, out=None):
    """P1g-c's verdict: the pools call (section 4), the input call, the full run it leads to (section 5), and beside:
    webqsp and metaqa, musique's in-domain read, and each arm against P1g-b's lean-pair fit on the same pools."""
    recalls = recalls or {k: RECALL[k].with_suffix(".json") for k in CALL_FITS}
    lean = lean or {a: {sp: BESIDE_CMP[a][sp] for sp in SPLITS} for a in ARM_NAMES}
    per = {f"{k[0]} {k[1]}": read_recall(k, recalls[k]) for k in CALL_FITS}
    pk = [f"pools {a}" for a in CALLS["pools"]]
    if any(per[x]["problems"] for x in pk):
        pv = "INCOMPLETE"
    elif any(per[x]["losses"] for x in pk) or not any(per[x]["gains"] for x in pk):
        pv = "NOT_ADOPTED"
    else:
        pv = "ADOPT_FOR_FULL_RUN"
    ip = per["input zspxr"]
    iv = ("INCOMPLETE" if ip["problems"] else "NOT_ADOPTED" if ip["losses"] or not ip["gains"]
          else "ADOPT_FOR_FULL_RUN")
    io = iv if iv != "NOT_ADOPTED" else ("GAIN_AND_LOSS" if ip["gains"] else "NO_GAIN")
    fr = full_run(pv, iv)
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {x: {"call": per[x]["calls"].get(ck), "delta": per[x]["delta_r5"].get(ck)} for x in per}
    lean_d = {a: {sp: deltas_of(lean[a][sp]) for sp in SPLITS} for a in ARM_NAMES}
    rec = {"stage": "P1gc", "declared_in": DECLARED, "pools_rule": POOLS_RULE, "input_rule": INPUT_RULE,
           "pools": {"verdict": pv, "next": NEXT_POOLS.get(pv, "the missing re-call is run")},
           "input": {"verdict": iv, "outcome": io, "next": NEXT_INPUT.get(io, "the missing re-call is run")},
           "calls": per, "full_run": fr, "musique_in_domain_beside": side, "against_p1gb_lean_r5": lean_d,
           "zp1gc_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    num = lambda x: f"{x:+.4f}" if isinstance(x, (int, float)) else "-"  # noqa: E731
    md = [f"# P1g-c's verdict: pools **{pv}**, input (zspxr) **{iv}**", "", f"Rules: {POOLS_RULE}; {INPUT_RULE}.", ""]
    for x, p in per.items():
        k = tuple(x.split())
        md.append(f"- **{x}** ({WHAT[k[1]]}) against {AGAINST[k]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
                  + (", ".join(f"{t} {b['call']} ({num(b['delta'])})" for t, b in p["beside"].items()) or "none") + ".")
    md += ["", f"Next, pools: {rec['pools']['next']}.", f"Next, input: {rec['input']['next']}.", "",
           "Full run: " + (", ".join(f"{a} {b}" for a, b in fr.items()) if fr else "none while a call is INCOMPLETE")
           + ".", "",
           f"Beside, deciding nothing: musique in-domain ({ck}): "
           + ", ".join(f"{x} {s['call']} ({num(s['delta'])})" for x, s in side.items()) + ". Against P1g-b's own fit "
           "of the lean arm on the same pools (R@5): "
           + "; ".join(f"{a} vs {LEAN_OF[a]} " + ", ".join(f"{sp} {d} {num(v)}" for sp in SPLITS
                                                         for d, v in lean_d[a][sp].items()) for a in ARM_NAMES
                       if any(lean_d[a].values())) + "."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zp1gc verdict: pools {pv}, input {iv}" + (f"; full run {fr}" if fr else ""))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, k, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.001 * j, "call": calls.get((sp, ds), "WITHIN")}
            for j, (sp, ds) in enumerate((sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER)]
    rec = {"stage": "P1gc", "declared_in": DECLARED, "call": k[0], "arm": k[1], "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def must_stop_with(fn, words, *args, **kw):
    """fn(*args) stops, and its message holds words (so a refusal is the one meant)."""
    try:
        fn(*args, **kw)
    except SystemExit as e:
        if words not in str(e):
            raise AssertionError(f"{fn.__name__} stopped with {e!s:.300}, not for {words!r}")
        return
    raise AssertionError(f"{fn.__name__}{args} must stop")


def toy_records(base):
    """Four toy record roots in P1g-b's layout, and the records' shas as P1g-b's fits stamp them."""
    roots = {f: base / f for f in RECS}
    k = 0
    for fam, (_, pat) in RECS.items():
        rels = ["metaqa/fit", "squad/fit"] if fam == "kt" else ["metaqa/fit/part_0of2", "metaqa/fit/part_1of2",
                                                                "webqsp/s1eval/part_0of1"]
        for rel in rels:
            k += 1
            LC.write_json(roots[fam] / rel / "record.json",
                          {"queries": k, "rows": 10 * k, "arrays": {"a": {"sha256": f"{k:064x}", "shape": [10 * k]}},
                           "seconds": 1.0, "utc": "t0"})
    return roots


def toy_stamps(base, roots, n=2):
    cur = current(roots)
    fits = []
    for i in range(n):
        st = {STAMP_KEY[f]: dict(cur[f]) for f in ("links", "rmatch", "zrc")}
        st["kt_records"] = {"metaqa/fit": cur["kt"]["metaqa/fit"]}
        LC.write_json(base / f"fit{i}" / "screen.json", {"arm": "x", "p1gb": st})
        fits.append(base / f"fit{i}" / "screen.json")
    return fits


def selftest():
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: zspx and zrckpr as filed; zspxr is zspx with R1Mix under its step, on G3's carve with R1's columns
    assert S.ARMS["zspx"] == (G5.ZSPX, G3.ResKTLinkCarveBase) == ARMS["zspx"]
    assert S.ARMS["zrckpr"] == (R1.ZRCKPR, R1.R1KTChainCarveZRC) == ARMS["zrckpr"]
    assert S.ARMS["zspxr"] == (ZSPXR, R1ResKTLinkCarveBase)
    assert ZSPXR.__mro__[1:8] == (G5.ZSPX, G3.ZSPR, Z3.ZSPK, ZP.ZProp, R1.R1Mix, Z3.KTMix, ZM.ZRM)
    assert ZSPXR.__mro__[8:] == ZM.ZRM.__mro__[1:]
    assert R1ResKTLinkCarveBase.__mro__[1:3] == (G3.ResKTLinkCarveBase, Z3.KTLinkCarveBase)
    assert "RES_ROOT" not in vars(R1ResKTLinkCarveBase) and vars(R1ResKTLinkCarveBase)["R1_ROOT"] == R1.COLS_DIR
    assert S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase) and S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC)
    assert S.ARMS["zspkr"] == (R1.ZSPKR, R1.R1KTLinkCarveBase)
    # 2. P1g-c's roots inside (P1g-b's, plus P1g-c's residual and column roots), everything restored after
    before = (G3.ResKTLinkCarveBase.RES_ROOT, R1.R1KTChainCarveZRC.R1_ROOT, R1ResKTLinkCarveBase.R1_ROOT,
              R1.R1KTLinkCarveBase.R1_ROOT, Z2.CACHE, Z2.LOOK, Z2.LINKS, Z3.KTLinkCarveBase.KT_ROOT, Z3.DECLARED)
    with on_p1gc():
        assert G3.ResKTLinkCarveBase.RES_ROOT == RES and R1ResKTLinkCarveBase.RES_ROOT == RES
        assert R1.R1KTChainCarveZRC.R1_ROOT == COLS == R1ResKTLinkCarveBase.R1_ROOT
        assert R1.R1KTLinkCarveBase.R1_ROOT == R1.COLS_DIR          # zspkr's, not P1g-c's
        assert (Z2.CACHE, Z2.LOOK, Z2.LINKS, Z2.GATE) == (GB.CACHE, GB.LOOK, GB.LINKS, GB.GATE)
        assert Z3.KTLinkCarveBase.KT_ROOT == GB.KT_ROOT and Z3.DECLARED == GB.DECLARED
        assert ZL.LinkCarveBase.LK_ROOT == GB.LINKS and RM.ChainCarveBase.CH_ROOT == GB.CH_RM
    assert (G3.ResKTLinkCarveBase.RES_ROOT, R1.R1KTChainCarveZRC.R1_ROOT, R1ResKTLinkCarveBase.R1_ROOT,
            R1.R1KTLinkCarveBase.R1_ROOT, Z2.CACHE, Z2.LOOK, Z2.LINKS, Z3.KTLinkCarveBase.KT_ROOT,
            Z3.DECLARED) == before
    assert all("p1gc" in p.as_posix() for p in (OUT, RES, COLS, SNAP, GATE, VERDICT))
    assert {f: r for f, (r, _) in RECS.items()} == {"links": GB.LINKS, "rmatch": GB.CH_RM, "zrc": GB.CH_ZRC,
                                                      "kt": GB.KT_ROOT}
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        # 3. the snapshot: P1g-b's records, checked against its fits' stamps, copied; refused when a stamp differs
        roots = toy_records(T_ / "p1gb")
        fits = toy_stamps(T_, roots)
        sn = snap(T_ / "snap", roots, fits)
        assert sorted(sn["records"]) == sorted(RECS) and len(sn["records"]["links"]) == 3
        assert len(sn["records"]["kt"]) == 2
        assert all((T_ / "snap" / f / rel / "record.json").read_bytes() == (roots[f] / rel / "record.json").read_bytes()
                   for f in RECS for rel in sn["records"][f])
        bad = json.loads(fits[1].read_text(encoding="utf-8"))
        bad["p1gb"]["zlink_records"]["metaqa/fit/part_0of2"] = "0" * 64
        LC.write_json(T_ / "badfit" / "screen.json", bad)
        must_stop_with(snap, "links records are not the ones it stamped", T_ / "s2", roots,
                       [fits[0], T_ / "badfit" / "screen.json"])
        bad = json.loads(fits[1].read_text(encoding="utf-8"))
        bad["p1gb"]["kt_records"] = {"metaqa/fit": "0" * 64}
        LC.write_json(T_ / "badfit" / "screen.json", bad)
        must_stop_with(snap, "KT records", T_ / "s2", roots, [T_ / "badfit" / "screen.json"])
        LC.write_json(T_ / "nostamp" / "screen.json", {"arm": "x"})
        must_stop_with(snap, "no P1g-b stamp", T_ / "s2", roots, [T_ / "nostamp" / "screen.json"])
        # 4. the gate: shut before the rebuild (records missing), open after it when only seconds and utc moved, shut on
        #    a changed array, a changed count, a missing record or an extra one; gate_open refuses a later rebuild
        g = T_ / "g"
        held = {}
        for f in RECS:
            for rel in sn["records"][f]:
                held[(f, rel)] = json.loads((roots[f] / rel / "record.json").read_text(encoding="utf-8"))
                (roots[f] / rel / "record.json").unlink()
        assert gate(g, T_ / "snap", roots) == 1
        gr = json.loads(g.with_suffix(".json").read_text(encoding="utf-8"))
        assert gr["gate"] == "SHUT" and any("not built again" in p for p in gr["problems"])
        must_stop_with(gate_open, "the gate is SHUT", g, roots)
        for (f, rel), r in held.items():
            LC.write_json(roots[f] / rel / "record.json", dict(r, seconds=2.5, utc="t1"))
        assert gate(g, T_ / "snap", roots) == 0
        gr = gate_open(g, roots)
        assert gr["gate"] == "OPEN" and gr["same"] == {"links": 3, "rmatch": 3, "zrc": 3, "kt": 2}
        assert gr["now"] != sn["records"]                       # the record files changed, their arrays did not
        r0 = held[("links", "metaqa/fit/part_1of2")]
        LC.write_json(roots["links"] / "metaqa/fit/part_1of2" / "record.json", dict(r0, utc="t2"))
        must_stop_with(gate_open, "changed after the gate opened", g, roots)
        for change, words in ((dict(r0, arrays={"a": {"sha256": "f" * 64, "shape": [20]}}), "arrays differ"),
                              (dict(r0, rows=999), "rows differ")):
            LC.write_json(roots["links"] / "metaqa/fit/part_1of2" / "record.json", change)
            assert gate(g, T_ / "snap", roots) == 1
            assert any(words in p for p in json.loads(g.with_suffix(".json").read_text(encoding="utf-8"))["problems"])
        LC.write_json(roots["links"] / "metaqa/fit/part_1of2" / "record.json", r0)
        LC.write_json(roots["zrc"] / "squad/fit/part_0of1" / "record.json", r0)
        assert gate(g, T_ / "snap", roots) == 1
        assert any("not in the snapshot" in p for p in json.loads(g.with_suffix(".json").read_text(
            encoding="utf-8"))["problems"])
        (roots["zrc"] / "squad/fit/part_0of1" / "record.json").unlink()
        assert gate(g, T_ / "snap", roots) == 0
        assert gate(T_ / "g2", T_ / "nosnap", roots) == 1
        LC.write_json(T_ / "other.json", dict(json.loads(g.with_suffix(".json").read_text(encoding="utf-8")),
                                              stage="P1gb"))
        must_stop_with(gate_open, "the gate is OPEN (P1gb)", T_ / "other", roots)
        # 5. the builds: G3's and R1's code on P1g-b's caches, looks and links, writing to P1g-c's roots, only while the
        #    gate is open
        calls = []
        saved_b = G3.build, G3.open_nodes, R1.build
        G3.open_nodes = lambda ds, host: (f"nodes {ds} {host}", "freeze")
        G3.build = lambda *a, **k: calls.append(("res", a, k, Z2.CACHE, Z2.LOOK, Z2.LINKS))
        R1.build = lambda *a, **k: calls.append(("cols", a, k, Z2.CACHE, Z2.LOOK, Z2.LINKS))
        try:
            build("res", "musique", "s1fit", 3, True, g, roots)
            build("cols", "webqsp", "s1eval", "0", False, g, roots)
            must_stop_with(build, "res or cols", "links", "musique", "s1fit", 0, True, g, roots)
            must_stop_with(build, "six datasets", "res", "nq", "s1fit", 0, True, g, roots)
            must_stop_with(build, "six datasets", "res", "musique", "s1fit", None, True, g, roots)
            must_stop_with(build, "the gate is OPEN (P1gb)", "res", "musique", "s1fit", 0, True, T_ / "other", roots)
            must_stop_with(build, "missing", "cols", "musique", "s1fit", 0, True, T_ / "nogate", roots)
        finally:
            G3.build, G3.open_nodes, R1.build = saved_b
        rts = (GB.CACHE, GB.LOOK, GB.LINKS)
        pl = {"stage": "P1gc", "declared_in": DECLARED}
        assert calls[0] == ("res", ("musique", "s1fit", 3), {"out_root": RES, "nodes": "nodes musique True",
                                                             "freeze": "freeze", "placement": dict(pl, where="host")}) + rts
        assert calls[1] == ("cols", ("webqsp", "s1eval", 0), {"out_root": COLS, "res_root": RES,
                                                              "placement": dict(pl, where="laptop")}) + rts
        assert len(calls) == 2 and (Z2.CACHE, Z2.LOOK, Z2.LINKS) == before[4:7]
        # 6. the refusals, each for its own reason (the gate before any fit or read)
        cr = ["--cache-root", "outputs/p1gb/cache"]
        good = ["train", "--arm", "zspx", "--name", FIT["zspx"]["L-musique"]] + cr
        must_stop_with(train, "train takes --arm", ["train", "--arm", "zspk", "--name", FIT["zspx"]["L-musique"]] + cr,
                       "L-musique", g, roots)
        must_stop_with(train, "train takes --arm", ["train", "--arm", "zspkr", "--name", "scr-p1gc-zspkr"] + cr,
                       "L-musique", g, roots)
        must_stop_with(train, "not L-2wiki", good, "L-2wiki", g, roots)
        must_stop_with(train, "zspx's fit of L-hotpotqa", good, "L-hotpotqa", g, roots)
        must_stop_with(train, "zspxr's fit of L-musique",
                       ["train", "--arm", "zspxr", "--name", FIT["zspx"]["L-musique"]] + cr, "L-musique", g, roots)
        SR.must_stop(train, ["train", "--arm", "zspx", "--name", FIT["zspx"]["L-musique"], "--cache-root",
                             "outputs/p1c2/cache"], "L-musique", g, roots)
        SR.must_stop(train, good + ["--device", "cpu"], "L-musique", g, roots)
        must_stop_with(train, "the gate is OPEN (P1gb)", good + ["--device", "cuda"], "L-musique", T_ / "other", roots)
        (T_ / "fits" / "y").mkdir(parents=True)
        for rec_ in ({"arm": "zspx", "g5": {}}, {"arm": "zrckpr", "r1": {}}, {"arm": "zspk", "p1gb": {}}):
            LC.write_json(T_ / "fits" / "y" / "screen.json", rec_)
            must_stop_with(read, "not a P1g-c fit", ["--name", "y", "--out-root", str(T_ / "fits")] + cr, g, roots)
        LC.write_json(T_ / "fits" / "y" / "screen.json", {"arm": "zspxr", "p1gc": {}})
        must_stop_with(read, "the gate is OPEN (P1gb)", ["--name", "y", "--out-root", str(T_ / "fits"), "--device",
                                                         "cuda"] + cr, T_ / "other", roots)
        # 7. relz's pair and re-call for each call and arm against its base arm's fits, stamped P1g-c's; relz restored
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        assert CALL_FITS[("pools", "zspx")]["L-hotpotqa"] == ("screen", "fits", "scr-g5-zspx-hp")
        assert CALL_FITS[("pools", "zrckpr")]["L-musique"] == ("screen", "fits", "scr-r1-zrckpr")
        assert CALL_FITS[("input", "zspxr")]["L-hotpotqa"] == ("screen", "fits", "scr-p1gc-zspx-hp")
        for k in CALL_FITS:
            call, arm = k
            fits_ = CALL_FITS[k]
            d = T_ / f"{call}-{arm}"
            null1 = SR.fake_null(d / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(d / fits_[sp][0], fits_[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=BASE_ARM[k])) for sp in fits_}
            zb = {sp: "/".join(("outputs",) + fits_[sp]) for sp in fits_}
            za = SR.fake_compare(d / "z", FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm, base_r5=0.6,
                                 base=zb["L-musique"])
            zh = SR.fake_compare(d / "z", FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            zw = SR.fake_compare(d / "z2", FIT[arm]["L-musique"], "L-musique", {}, arm=arm, base_r5=0.6,
                                 base="outputs/screen/fits/scr-p1gb-zspk")
            SR.must_stop(pair, call, arm, [zw, zh], d / "bad")
            pr = pair(call, arm, [za, zh], d / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_ARM[k], pr["verdict"]
            pj = json.loads((d / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "P1gc" and pj["call"] == call and pj["decided_against"] == AGAINST[k]
            r = recall(call, arm, null1, d / "pz.json", d / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (d / "rz.md").read_text(encoding="utf-8")
            pb, pn = pools_of(k)
            assert t.startswith(f"P1g-c ({DECLARED}), the {call} call") and \
                f"| {BASE_ARM[k]} R@5 ({pb}) | {arm} R@5 ({pn}) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        must_stop_with(pair, "the calls are", "pools", "zspxr", [], T_ / "x")
        must_stop_with(pair, "the calls are", "input", "zspx", [], T_ / "x")
        # 8. the verdict: the pools call (joint), the input call (zspxr), the full run, next forms and beside
        gm, mi = ("L-musique", "musique"), CLAIM1_READ
        lean = {a: {sp: SR.fake_compare(T_ / "lean" / a, FIT[a][sp], sp, {"musique": 0.01}, arm=a,
                                        base=f"outputs/screen/fits/scr-p1gb-{LEAN_OF[a]}{SX[sp]}")
                    for sp in SPLITS} for a in ARM_NAMES}

        def v(c_px, c_pm, c_ir, **over):
            rc = {("pools", "zspx"): fake_recall(T_ / "px.json", ("pools", "zspx"), c_px, **over.get("px", {})),
                  ("pools", "zrckpr"): fake_recall(T_ / "pm.json", ("pools", "zrckpr"), c_pm, **over.get("pm", {})),
                  ("input", "zspxr"): fake_recall(T_ / "ir.json", ("input", "zspxr"), c_ir, **over.get("ir", {}))}
            return verdict(rc, lean, T_ / "v")

        rec = v({mi: "GAIN"}, {}, {gm: "GAIN"})
        assert rec["pools"]["verdict"] == "ADOPT_FOR_FULL_RUN" and rec["input"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["full_run"] == {"pools": "P1g-b's", "mlp": "zrckpr", "gnn": "zspxr"}
        assert rec["pools"]["next"] == NEXT_POOLS["ADOPT_FOR_FULL_RUN"]
        assert rec["musique_in_domain_beside"]["pools zspx"]["call"] == "GAIN"
        assert sorted(rec["calls"]["pools zspx"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert rec["against_p1gb_lean_r5"]["zspx"]["L-musique"]["musique"] == 0.01
        rec = v({mi: "GAIN"}, {gm: "LOSS"}, {gm: "GAIN"})
        assert rec["pools"]["verdict"] == "NOT_ADOPTED" and rec["pools"]["next"] == NEXT_POOLS["NOT_ADOPTED"]
        assert rec["full_run"] == {"pools": "P1c2's", "mlp": "zrckpr", "gnn": "zspx",
                                   "zspxr": "its two screen fits on P1c2's pools first, against G5's zspx fits"}
        rec = v({}, {}, {})
        assert rec["pools"]["verdict"] == "NOT_ADOPTED" and rec["input"]["outcome"] == "NO_GAIN"
        assert rec["input"]["next"] == NEXT_INPUT["NO_GAIN"] and rec["full_run"]["zspxr"] == "not carried"
        rec = v({gm: "GAIN"}, {mi: "GAIN"}, {gm: "LOSS", mi: "GAIN"})
        assert rec["pools"]["verdict"] == "ADOPT_FOR_FULL_RUN" and rec["input"]["outcome"] == "GAIN_AND_LOSS"
        assert rec["full_run"]["gnn"] == "zspx" and rec["input"]["next"] == NEXT_INPUT["GAIN_AND_LOSS"]
        rec = v({gm: "GAIN"}, {}, {gm: "GAIN"}, pm={"call": "input"})
        assert rec["pools"]["verdict"] == "INCOMPLETE" and rec["full_run"] is None
        assert rec["pools"]["next"] == "the missing re-call is run"
        rec = v({gm: "GAIN"}, {}, {gm: "GAIN"}, ir={"arm": "zspx"})
        assert rec["input"]["verdict"] == "INCOMPLETE" and rec["full_run"] is None
        short = fake_recall(T_ / "s.json", ("pools", "zspx"), {gm: "GAIN"})
        sj = json.loads(short.read_text(encoding="utf-8"))
        sj["rows"] = sj["rows"][:11]
        LC.write_json(short, sj)
        rec = verdict({("pools", "zspx"): short, ("pools", "zrckpr"): T_ / "none.json",
                       ("input", "zspxr"): fake_recall(T_ / "ir.json", ("input", "zspxr"), {})}, lean, T_ / "v")
        assert rec["pools"]["verdict"] == "INCOMPLETE" and rec["input"]["verdict"] == "NOT_ADOPTED"
        assert len(rec["calls"]["pools zspx"]["problems"]) == 1 and "missing" in rec["calls"]["pools zrckpr"][
            "problems"][0]
    tmp = Path(tempfile.mkdtemp(prefix="zp1gc_"))
    try:
        rng = np.random.default_rng(57)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        T, _, _ = G3.toy_embed(tmp, rng)
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        for pt in (0, 1):
            G3.build("metaqa", "toy", pt, out_root=tmp / "res", cache_root=tmp / "cache", look_root=tmp / "look",
                     lk_root=tmp / "lk", nodes=G3.ToyNodes(T), freeze="toy")
        R1.build("metaqa", "toy", out_root=tmp / "r1", cache_root=tmp / "cache", lk_root=tmp / "lk",
                 res_root=tmp / "res")
        # 9. the records: each arm needs what its carve reads, from the roots given
        for a, keys in (("zspx", ["res_records"]), ("zspxr", ["r1_records", "res_records"]),
                        ("zrckpr", ["r1_records"])):
            saved_r = G3.res_records, R1.r1_records
            G3.res_records = lambda carves, root, cache_root: {"res": (carves, root, cache_root)}
            R1.r1_records = lambda carves, root, cache_root: {"cols": (carves, root, cache_root)}
            try:
                got = records(a, [("metaqa", "toy")])
            finally:
                G3.res_records, R1.r1_records = saved_r
            assert sorted(got) == keys, (a, got)
            for k in got:
                assert got[k] == ({"res": ([("metaqa", "toy")], RES, GB.CACHE)} if k == "res_records" else
                                  {"cols": ([("metaqa", "toy")], COLS, GB.CACHE)})
        cls_ch = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        cls_res = G3.res_carve(cls_lk, tmp / "res")
        c_x = R1.r1_carve(cls_res, tmp / "r1")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_res = cls_res("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c_x.rows)[::-1][:7], np.arange(c_x.rows)[:5]].astype(np.int64)
        B = qs.size
        f_, nq, bz, _gold = c_x.batch(qs, blocks)
        fr, nq_r, bz_r, _ = c_res.batch(qs, blocks)
        # 10. zspxr's carve is G3's with R1's columns beside: the batch is G3's plus the columns
        assert sorted(set(f_) - set(fr)) == [R1.R1_KEY] and torch.equal(nq, nq_r) and torch.equal(bz, bz_r)
        for k in fr:
            if k == ZL.LK_KEY or k == G3.RES_KEY:
                assert all(torch.equal(f_[k][j], fr[k][j]) for j in fr[k]), k
        lk, rs = f_[ZL.LK_KEY], f_[G3.RES_KEY]
        assert lk["u"].numel() > 0 and float(f_[R1.R1_KEY].abs().max()) > 0.05
        keep = torch.ones(B, len(blocks))
        # 11. zspxr starts as zspx bit for bit (eval and training, the same dropout draws); its state is zspx's plus
        #     R1's head as R1 draws it
        torch.manual_seed(0)
        mx = G5.ZSPX(blocks, c_x.widths, 16, dropout=0.1, seed=0)
        gx = torch.random.get_rng_state()
        torch.manual_seed(0)
        mr = ZSPXR(blocks, c_x.widths, 16, dropout=0.1, seed=0)
        assert torch.equal(gx, torch.random.get_rng_state())
        torch.manual_seed(0)
        mk = R1.ZSPKR(blocks, c_x.widths, 16, dropout=0.1, seed=0)
        sx, sr, sk = mx.state_dict(), mr.state_dict(), mk.state_dict()
        assert sorted(set(sr) - set(sx)) == sorted(R1.R1) and set(sx) <= set(sr)
        assert not LG.same_state(sx, {k: sr[k] for k in sx})
        assert not LG.same_state({k: sk[k] for k in R1.R1}, {k: sr[k] for k in R1.R1})
        assert sum(sr[k].numel() for k in R1.R1) == R1.N_R1
        for train_mode in (False, True):
            mx.train(train_mode)
            mr.train(train_mode)
            torch.manual_seed(3)
            y0 = mx(f_, keep, nq, B, bz)
            torch.manual_seed(3)
            assert torch.equal(y0, mr(f_, keep, nq, B, bz)), train_mode
        mx.eval()
        mr.eval()
        # 12. the placement: with every head moved, zspxr is zspx's three steps over (the KT score + R1's head);
        #     without the columns it is zspx; without links it is the KT score + R1's head
        with torch.no_grad():
            g_ = torch.Generator().manual_seed(9)
            w_sp, w_rs, w_p2 = (torch.randn(1, 32, generator=g_) for _ in range(3))
            for m in (mx, mr):
                m.sp_w2.copy_(w_sp)
                m.rs_w2.copy_(w_rs)
                m.p2_w2.copy_(w_p2)
            mr.r1_w2.copy_(torch.randn(1, R1.R1_HID, generator=g_))
            y0, y1 = mx(f_, keep, nq, B, bz), mr(f_, keep, nq, B, bz)
            head = mr.r1_head(f_[R1.R1_KEY])
            kt = Z3.KTMix.forward(mr, f_, keep, nq, B, bz)
            pre = super(ZP.ZProp, mr).forward(f_, keep, nq, B, bz)
            assert float(head.abs().max()) > 1e-3 and torch.equal(pre, kt + head)
            s1 = pre + mr.prop_head(G4.prop_inputs_grad(pre, lk, nq, B))
            s2 = s1 + mr.rs_head(G3.res_inputs(s1, lk, rs, nq, B))
            s3 = s2 + mr.p2_head(ZP.prop_inputs(s2, lk, nq, B))
            assert torch.equal(y1, s3) and float((y1 - y0 - head).abs().max()) > 1e-3
            assert torch.equal(kt, Z3.KTMix.forward(mx, f_, keep, nq, B, bz))
            bare = {k: x for k, x in f_.items() if k != R1.R1_KEY}
            assert torch.equal(mr(bare, keep, nq, B, bz), mx(bare, keep, nq, B, bz))
            no_lk = {k: x for k, x in f_.items() if k not in (ZL.LK_KEY, G3.RES_KEY)}
            assert torch.equal(mr(no_lk, keep, nq, B, bz), kt + head)
        # 13. toy fits through lean_gpu's loop: a repeat is bit for bit; R1's head learns; on G3's carve without the
        #     columns zspxr's fit is zspx's bit for bit on every shared key, R1's last layer at zero
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched("zspxr"):
            fit = LG.fit_variant([c_x], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxr")
            rep = LG.fit_variant([c_x], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxr repeat")
            bare = LG.fit_variant([c_res], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspxr, no columns")
        with S.patched("zspx"):
            base = LG.fit_variant([c_res], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx")
        assert all(not LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]], rep["states"] + [rep["swa"]]))
        assert float(fit["swa"]["r1_w2"].abs().max()) > 0
        for a, b in zip(bare["states"] + [bare["swa"]], base["states"] + [base["swa"]]):
            assert not LG.same_state({k: a[k] for k in b}, b)
            assert float(a["r1_w2"].abs().max()) == 0.0
        assert LG.same_state({k: fit["swa"][k] for k in base["swa"]}, base["swa"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"zp1gc selftest: zspx and zrckpr as filed and zspxr = zspx with R1Mix under its step, on G3's carve with R1's "
        f"columns; P1g-b's roots with P1g-c's residual and column roots inside, all restored after; the snapshot checks "
        f"P1g-b's records against its fits' stamps and copies them; the gate opens only on the same arrays and counts "
        f"and refuses a later rebuild; the builds are G3's and R1's on P1g-b's caches, looks and links into P1g-c's "
        f"roots, behind the gate; train and read refuse for their own reasons; relz's pair and re-call for the pools and "
        f"input calls, stamped P1g-c's; the verdict, its full run and next forms; each arm needs its own records; the "
        f"batch is G3's plus the columns; zspxr starts as zspx bit for bit, its steps read the score with R1's head, and "
        f"without the columns it is zspx; toy fits repeat, R1's head learns, and without the columns the fit is zspx's "
        f"({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="L-musique", choices=S2.SPLITS)
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "snap":
        snap()
        return 0
    if k.cmd == "gate":
        return gate()
    if k.cmd == "build":
        bp = argparse.ArgumentParser()
        bp.add_argument("--what", required=True, choices=("res", "cols"))
        bp.add_argument("--dataset", required=True, choices=LG.EVAL_ORDER)
        bp.add_argument("--carve", required=True)
        bp.add_argument("--part", type=int, required=True)
        bp.add_argument("--host", action="store_true")
        b = bp.parse_args(rest)
        build(b.what, b.dataset, b.carve, b.part, b.host)
        return 0
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "verdict":
        rec = verdict()
        return 0 if "INCOMPLETE" not in (rec["pools"]["verdict"], rec["input"]["verdict"]) else 1
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--call", required=True, choices=tuple(CALLS))
        gp.add_argument("--arm", required=True, choices=ARM_NAMES)
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        try:
            if k.cmd == "pair":
                pair(g.call, g.arm, [x for x in g.screens.split(",") if x], g.out)
            else:
                if len(null) != 4 or not g.pair:
                    gp.error("recall: --pair and the four --null files")
                recall(g.call, g.arm, null, g.pair, g.out)
        except SystemExit as e:
            log(f"zp1gc {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zp1gc: snap, gate, build, train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
