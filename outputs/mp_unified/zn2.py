"""N2 (docs/N2_FRESH_SEED_NULL.md): a fresh seed null on today's lean pair. The 7 October rule for a read's floor
(docs/SCREENS.md, section 2; screen_recall.py, unchanged) with its seed-only spread measured again on the two bases every
screen since 15 October is decided against: zrckp's card fits (scr-p1c3p-zrckp, -hp; zp1c3p.py) and zspk's
(scr-p1c3-zspk, -hp; zp1c3.py). Each card is fitted again with seeds 1 and 2, its settings otherwise the card's, read on
the six s1eval carves and compared with its card fit (seed 0) alone. For a read r (split x dataset), with D_s(r) seed
s's R@5 minus the card's,
    floor(r) = max(0.0075, 2 * sqrt((D_1(r)^2 + D_2(r)^2) / 2)).
Every screen arm filed since 15 October on a zrckp or zspk base (56 arms, P1g-b's pool arms aside) is called again from
its filed pair with its base's fresh floors, beside its filed call; the filed verdicts are never rewritten.

    python outputs/mp_unified/zn2.py train --split L-musique --name scr-n2-zrckp-s1 --arm zrckp --seed 1 \\
        --device cuda --host --cache-root outputs/p1c2/cache          (zp1c3p.py's train under N2's fit name)
    python outputs/mp_unified/zp1c3.py train --split L-musique --name scr-n2-zspk-s1 --arm zspk --seed 1 \\
        --device cuda --host --cache-root outputs/p1c2/cache          (zspk takes --seed as it is)
    python outputs/mp_unified/zp1c3p.py read --name scr-n2-zrckp-s1 --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zn2.py compare --new outputs/screen/fits/scr-n2-zrckp-s1 \\
        --base outputs/screen/fits/scr-p1c3p-zrckp --out outputs/screen/scr-n2-zrckp-s1
    python outputs/mp_unified/zn2.py floors --base zrckp               (outputs/n2/floors-zrckp.json, .md)
    python outputs/mp_unified/zn2.py repro                             (outputs/n2/repro.json, .md)
    python outputs/mp_unified/zn2.py recall-all                        (outputs/n2/recall-<stage>-<arm>, recall-all)
    python outputs/mp_unified/zn2.py summary                           (outputs/n2/summary.json, .md)
    python outputs/mp_unified/zn2.py gate --stage a5c [--arm zrckpc]   (exit 0 pass, 1 no pass, 2 not decidable)
    python outputs/mp_unified/zn2.py --selftest

The fresh call starts from the pair's calls (made at 0.0075) through screen_recall.pair, so a fresh floor lower than the
7 October one can restore a call the 7 October floors removed, and a higher one can only remove calls. repro calls every
filed arm again with the OLD null through the stages' own path (relz.recall's loader, the card fits as the base) and
must give the filed re-call exactly. floors, repro and recall-all exit 0 once their record is written (floors exits 2
after writing a REFUSED record); the gate's exit code carries the decision.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zp1c3p as Z3P  # noqa: E402  (zrckp; through it zp1c3's zspk, P1c2's roots, relz and screen_recall)

Z3 = Z3P.Z3
Z2 = Z3.Z2
Z, S2, S, LG, LC, SR, RM = Z3.Z, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR, Z3.RM
log = S.log

DECLARED = "docs/N2_FRESH_SEED_NULL.md"
STAGE = "N2"
SPLITS = Z3.SPLITS
SUF = {"L-musique": "", "L-hotpotqa": "-hp"}
SEEDS = SR.SEEDS
BASES = ("zrckp", "zspk")
CARD = {"zrckp": "scr-p1c3p-zrckp", "zspk": "scr-p1c3-zspk"}
STAMP = {"zrckp": ("p1c3p", Z3P.DECLARED), "zspk": ("p1c3", Z3.DECLARED)}
OLD_NULL = tuple(f"outputs/screen/scr-null-s{sd}{SUF[sp]}.json" for sp in SPLITS for sd in SEEDS)
OLD_FLOORS = "outputs/screen/scr-null-floors.json"
SINCE = "2026-10-15"
RULE = ("floor(r) = max(0.0075, 2 * sqrt((D_1(r)^2 + D_2(r)^2) / 2)), D_s(r) = seed s's R@5 minus the card fit's "
        "(seed 0) on read r: the 7 October rule (docs/SCREENS.md, section 2; screen_recall.py), unchanged, on N2's "
        "null; a read's call starts from its pair's call (made at 0.0075)")


def card_fit(base, split):
    return CARD[base] + SUF[split]


def n2_fit(base, seed, split):
    return f"scr-n2-{base}-s{seed}{SUF[split]}"


def base_of(arm):
    return "zrckp" if arm.startswith("zrckp") else "zspk"


# Every screen arm with a verdict filed since 15 October on a zrckp or zspk base, P1g-b's pool arms aside (section 1).
ARMS = (("G2", "zg2"), ("L1", "zrckps"), ("L1", "zspks"), ("L3", "zrckpo"), ("L3", "zspko"), ("T1", "zrckpt"),
        ("T1", "zspkt"), ("E1", "zrckpe"), ("E1", "zspke"), ("G3", "zspr"), ("G4", "zspk2"), ("G4", "zspkg"),
        ("G5", "zspx"), ("R1", "zrckpr"), ("R1", "zspkr"), ("A2", "zrckpa"), ("A2", "zspka"), ("A3", "zrckpv"),
        ("A3", "zspkv"), ("A4", "zrckpd"), ("A4", "zspkd"), ("A5", "zrckpk"), ("A5", "zspkk"), ("A6", "zrckpw"),
        ("A6", "zspkw"), ("A7", "zrckps"), ("A7", "zspks"), ("A8", "zrckpq"), ("A8", "zspkq"), ("A11", "zspk4"),
        ("A12", "zspkgs"), ("A13", "zspkb"), ("P5c", "zrckpo"), ("P5c", "zspko"), ("A4h", "zrckph"), ("A4h", "zspkh"),
        ("A4f", "zrckpf"), ("A4f", "zspkf"), ("A5c", "zrckpc"), ("A5c", "zspkc"), ("A6e", "zrckpn"), ("A6e", "zspkl"),
        ("A2h", "zrckpi"), ("A2h", "zrckpm"), ("A2h", "zspki"), ("A2h", "zspkm"), ("A3t", "zrckpu"), ("A3t", "zspku"),
        ("A2xm", "zrckpxm"), ("A2xm", "zspkxm"), ("A3v", "zrckpy"), ("A3v", "zspky"), ("A4hs", "zrckphs"),
        ("A4hs", "zspkhs"), ("A8e", "zrckpqe"), ("A8e", "zspkqe"))
ADOPTED = frozenset({("A3v", "zrckpy"), ("A5c", "zrckpc"), ("A5c", "zspkc"), ("A7", "zrckps"), ("A7", "zspks"),
                     ("G3", "zspr"), ("G4", "zspk2"), ("G4", "zspkg"), ("G5", "zspx"), ("R1", "zrckpr")})
GATED = {"A5c": ("A5cp", "docs/A5CP_P1GB_POOLS.md"), "A3v": ("A3vp", "docs/A3VP_P1GB_POOLS.md"),
         "A7": ("A7p", "docs/A7P_P1GB_POOLS.md")}
UNGATED = ("G3", "G4", "G5", "R1")
GATE_STAGES = {st.lower(): st for st in GATED}

# A fresh seed fit must be its card fit with only the seed changed: these keys (where the card holds them) must match.
CODE = {"script_sha256": S, "lean_gpu_sha256": LG, "lean_screen2_sha256": S2, "rmatch_sha256": RM,
        "zp1c2_sha256": Z2, "zp1c3_sha256": Z3, "zp1c3p_sha256": Z3P}
SAME_TOP = ("arm", "split", "train", "basis", "variants", "config", "hidden", "cache_root") + tuple(CODE) + ("rmatch",)
SAME_STAMP = ("declared_in", "arm", "base_arm", "device", "graph", "pools", "cache_root", "head")
BESIDE_TOP = ("rmatch_records",)                          # build records: reported beside, never refused
BESIDE_STAMP = ("gate", "kt_records", "rmatch_records", "zrc_records", "zlink_records")
RECALL_ADDED = ("call_filed", "covered", "floor", "null_d", "call", "near_floor")   # screen_recall.recall_row's keys


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def repo_rel(s):
    """A path's repository form: its parts from the first 'outputs' on, joined with '/' (None if it has none)."""
    if not s:
        return None
    parts = SR.parts_of(s)
    if "outputs" not in parts:
        return None
    return "/".join(parts[parts.index("outputs"):])


class Where:
    """N2's paths under a repository root (the real one, or a selftest's)."""

    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.screen = self.root / "outputs" / "screen"
        self.out = self.root / "outputs" / "n2"

    def p(self, rel):
        return self.root / Path(*SR.parts_of(rel))

    def rel(self, p):
        p = Path(p)
        try:
            return p.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return repo_rel(p) or p.name

    def pair(self, st, arm):
        return self.screen / f"scr-{st.lower()}-{arm}-pair.json"

    def filed(self, st, arm):
        return self.screen / f"scr-{st.lower()}-{arm}-pair-recall.json"

    def verdict(self, st):
        return self.root / "outputs" / st.lower() / "screen-verdict.json"

    def recall_out(self, st, arm):
        return self.out / f"recall-{st.lower()}-{arm}"

    def nulls(self, base):
        return [self.screen / f"{n2_fit(base, sd, sp)}.json" for sp in SPLITS for sd in SEEDS]

    def old_null(self):
        return [self.p(x) for x in OLD_NULL]

    def floors(self, base):
        return self.out / f"floors-{base}"

    def fit_dir(self, name):
        return self.screen / "fits" / name

    def card_record(self, base, split):
        return self.fit_dir(card_fit(base, split)) / "screen.json"


def clean(msg, w):
    """A message without the root's absolute path (records name repository paths only)."""
    s = str(msg)
    for r in sorted({str(w.root), w.root.as_posix(), str(w.root.resolve()), w.root.resolve().as_posix()}, key=len,
                    reverse=True):
        s = s.replace(r + "\\", "").replace(r + "/", "").replace(r, "")
    return s


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write(out, rec, md):
    SR.write(out, rec, md)


# ── the card and the fresh fits ──────────────────────────────────────────────


def card_rec(base, split, w):
    p = w.card_record(base, split)
    if not p.exists():
        raise SystemExit(f"zn2: {w.rel(p)} missing (the card fit's record)")
    rec = load(p)
    key, decl = STAMP[base]
    if rec.get("seed") != 0 or rec.get("arm") != base or rec.get("split") != split or \
            (rec.get(key) or {}).get("declared_in") != decl:
        raise SystemExit(f"zn2: {w.rel(p)} is arm {rec.get('arm')} with seed {rec.get('seed')} on {rec.get('split')} "
                         f"({key} {(rec.get(key) or {}).get('declared_in')}), not {base}'s card fit of {split}")
    return rec


def same_as_card(base, split, rec, w):
    """(differences, beside): the keys a fresh fit's record must share with its card fit's (only those the card holds),
    and the build records that differ (reported, never refused)."""
    card = card_rec(base, split, w)
    key = STAMP[base][0]
    cs, rs = card.get(key) or {}, rec.get(key) or {}
    diff = [k for k in SAME_TOP if k in card and rec.get(k) != card[k]]
    diff += [f"{key}.{k}" for k in SAME_STAMP if k in cs and rs.get(k) != cs[k]]
    beside = [k for k in BESIDE_TOP if k in card and rec.get(k) != card[k]]
    beside += [f"{key}.{k}" for k in BESIDE_STAMP if k in cs and rs.get(k) != cs[k]]
    return diff, beside


def fit_rec(w, new):
    p = Path(str(new))
    d = p if p.is_absolute() else w.p(new)
    sj = d / "screen.json"
    if not sj.exists():
        raise SystemExit(f"zn2: {w.rel(sj)} missing")
    return load(sj)


def stamp_ok(base, s, seed, split):
    """Why a fresh fit's record is not N2's fit of (base, seed, split), or None."""
    key, decl = STAMP[base]
    st = s.get(key) or {}
    if st.get("declared_in") != decl or st.get("arm") != base:
        return f"its {key} stamp is {st.get('declared_in')} ({st.get('arm')}), not {decl} ({base})"
    if base == "zrckp":
        n = s.get("n2") or {}
        if n.get("declared_in") != DECLARED or n.get("seed") != seed or n.get("split") != split:
            return f"its n2 stamp is {n or None}, not N2's seed {seed} on {split}"
    return None


def seed_fits(base, files, w):
    """N2's four comparisons of base -> ({(split, seed): {file, rows}}, sources, beside), or SystemExit naming what is
    wrong: each compares N2's fit of (base, seed, split) with the card fit of split alone, p@swa on s1eval, on the split's
    training sets and the six datasets, and the fit is the card's with only the seed changed."""
    files = list(files)
    if len(files) != len(SPLITS) * len(SEEDS):
        raise SystemExit(f"zn2: {base}'s null takes four comparisons, not {len(files)}")
    got, sources, beside = {}, {}, {}
    for fn in files:
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zn2: {w.rel(p)} missing")
        rec = load(p)
        bases = rec.get("bases") or []
        if len(bases) != 1:
            raise SystemExit(f"zn2: {w.rel(p)} is decided against {bases}, not the card fit alone")
        b0 = SR.parts_of(bases[0])
        split = next((sp for sp in SPLITS if b0[-3:] == ("screen", "fits", card_fit(base, sp))), None)
        if split is None:
            raise SystemExit(f"zn2: {w.rel(p)} is decided against {bases[0]}, not {base}'s card fit of "
                             f"{' or '.join(SPLITS)}")
        if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
            raise SystemExit(f"zn2: {w.rel(p)} reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on s1eval")
        if sorted(rec.get("trained_on") or []) != sorted(LG.FITS[split]):
            raise SystemExit(f"zn2: {w.rel(p)} trained on {rec.get('trained_on')}, not {split}'s "
                             f"{sorted(LG.FITS[split])}")
        s = fit_rec(w, rec["new"])
        if s.get("arm") != base or s.get("split") != split or s.get("seed") not in SEEDS:
            raise SystemExit(f"zn2: {w.rel(p)}'s fit is arm {s.get('arm')} with seed {s.get('seed')} on "
                             f"{s.get('split')}, not {base} with seed 1 or 2 on {split}")
        seed = s["seed"]
        if SR.parts_of(rec["new"])[-3:] != ("screen", "fits", n2_fit(base, seed, split)):
            raise SystemExit(f"zn2: {w.rel(p)} reads {rec['new']}, not N2's fit {n2_fit(base, seed, split)}")
        why = stamp_ok(base, s, seed, split)
        if why:
            raise SystemExit(f"zn2: {w.rel(p)}'s fit: {why}")
        if (split, seed) in got:
            raise SystemExit(f"zn2: {base} on {split} with seed {seed} appears twice")
        rows = rec["by_base"][bases[0]]["rows"]
        if sorted(r["dataset"] for r in rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"zn2: {w.rel(p)} reads {[r['dataset'] for r in rows]}, not the six datasets")
        diff, bes = same_as_card(base, split, s, w)
        if diff:
            raise SystemExit(f"zn2: {w.rel(p)}'s fit differs from the card fit {card_fit(base, split)} in {diff}")
        got[(split, seed)] = {"file": p, "rows": {r["dataset"]: r for r in rows}}
        sources[f"{split} seed {seed}"] = w.rel(p)
        if bes:
            beside[f"{split} seed {seed}"] = bes
    want = [(sp, sd) for sp in SPLITS for sd in SEEDS]
    if sorted(got) != sorted(want):
        raise SystemExit(f"zn2: {base}'s null holds {sorted(got)}, not {sorted(want)}")
    return got, {f"{sp} seed {sd}": sources[f"{sp} seed {sd}"] for sp, sd in want}, beside


def null_of(got):
    """screen_recall.load_null's arithmetic on {(split, seed): {rows}}: each read's seed 0, D1, D2, spread and floor."""
    null = {}
    for split in SPLITS:
        null[split] = {}
        for ds in LG.EVAL_ORDER:
            r1, r2 = got[(split, 1)]["rows"][ds], got[(split, 2)]["rows"][ds]
            if r1["base"][0] != r2["base"][0]:
                raise SystemExit(f"zn2: {split} {ds}: the two seeds are compared with different seed-0 R@5 "
                                 f"({r1['base'][0]}, {r2['base'][0]})")
            d1, d2 = r1["delta"][0], r2["delta"][0]
            null[split][ds] = {"read": r1["read"], "seed0": r1["base"][0], "d1": d1, "d2": d2,
                               "spread": math.sqrt((d1 * d1 + d2 * d2) / 2.0), "floor": SR.floor_of(d1, d2)}
    return null


def floor_reads(null):
    """screen_recall.floors' reads of a null (the same keys, rounding and order)."""
    return [{"split": sp, "dataset": ds, **{k: (round(x, 6) if isinstance(x, float) else x) for k, x in v.items()},
             "raised": v["floor"] > S.FLOOR} for sp in SPLITS for ds in LG.EVAL_ORDER for v in [null[sp][ds]]]


def load_null_n2(base, files, w):
    got, sources, _ = seed_fits(base, files, w)
    return null_of(got), sources


@contextlib.contextmanager
def null_loader(fn):
    saved = SR.load_null
    SR.load_null = fn
    try:
        yield
    finally:
        SR.load_null = saved


def call_pair(loader, null_files, pair_file):
    """screen_recall.pair (unchanged) with its null loaded by loader; nothing written."""
    with null_loader(loader), contextlib.redirect_stdout(io.StringIO()):
        return SR.pair(null_files, pair_file)


@contextlib.contextmanager
def on_card(base, arm, rel_screens):
    """relz's names as the stages set them for an arm decided against base's card fits (za5c.on_base's form)."""
    fits = {sp: ("screen", "fits", card_fit(base, sp)) for sp in SPLITS}

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"zn2: no card fit of {base} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, base, fits, rel_screens, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


# ── train and compare ────────────────────────────────────────────────────────


def stamp_n2(w, name, seed, split):
    sj = w.fit_dir(name) / "screen.json"
    if not sj.exists():
        raise SystemExit(f"zn2: {w.rel(sj)} missing after the fit")
    rec = load(sj)
    if rec.get("seed") != seed or rec.get("split") != split or rec.get("arm") != "zrckp" or "p1c3p" not in rec:
        raise SystemExit(f"zn2: {w.rel(sj)} is arm {rec.get('arm')} with seed {rec.get('seed')} on {rec.get('split')}, "
                         f"not P1c3p's zrckp with seed {seed} on {split}")
    rec["zn2_sha256"] = LC.sha_src(__file__)
    rec["n2"] = {"declared_in": DECLARED, "base_arm": "zrckp", "seed": seed, "split": split,
                 "card_fit": card_fit("zrckp", split)}
    LC.write_json(sj, rec)


def train(argv, w=None, _train=None):
    """zp1c3p.py's train (unchanged) with N2's fit name in place of the card's; the code must be the card fit's."""
    w = w or Where()
    ap = argparse.ArgumentParser(prog="zn2.py train")
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--cache-root", required=True)
    a = ap.parse_args(argv)
    if a.arm != "zrckp":
        raise SystemExit(f"zn2: train fits zrckp only, not {a.arm} (zspk's fits run zp1c3.py train with --seed)")
    if a.seed not in SEEDS:
        raise SystemExit(f"zn2: N2's seeds are {SEEDS}, not {a.seed}")
    if a.name != n2_fit("zrckp", a.seed, a.split):
        raise SystemExit(f"zn2: zrckp's seed {a.seed} fit of {a.split} is {n2_fit('zrckp', a.seed, a.split)}, "
                         f"not {a.name}")
    card = card_rec("zrckp", a.split, w)
    diff = [k for k, m in CODE.items() if k in card and card[k] != LC.sha_src(m.__file__)]
    if diff:
        raise SystemExit(f"zn2: the code differs from the card fit {card_fit('zrckp', a.split)}'s in {diff}")
    args = ["train", "--name", a.name, "--arm", "zrckp", "--seed", str(a.seed), "--device", a.device,
            "--cache-root", a.cache_root] + (["--host"] if a.host else [])
    saved = Z3P.FIT
    Z3P.FIT = {**saved, a.split: a.name}
    try:
        rc = (_train or Z3P.train)(args, a.split)
    finally:
        Z3P.FIT = saved
    if rc == 0:
        stamp_n2(w, a.name, a.seed, a.split)
    return rc


def compare(argv, w=None, run=None):
    """lean_screen's compare (through lean_screen2's main, unchanged) of one of N2's fits with its card fit alone."""
    w = w or Where()
    ap = argparse.ArgumentParser(prog="zn2.py compare")
    ap.add_argument("--new", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    names = {n2_fit(b, sd, sp): (b, sd, sp) for b in BASES for sd in SEEDS for sp in SPLITS}
    parts = SR.parts_of(a.new)
    if len(parts) != 4 or parts[:3] != ("outputs", "screen", "fits") or parts[3] not in names:
        raise SystemExit(f"zn2: compare --new takes outputs/screen/fits/ and one of N2's fits, not {a.new}")
    b, sd, sp = names[parts[3]]
    if SR.parts_of(a.base) != ("outputs", "screen", "fits", card_fit(b, sp)):
        raise SystemExit(f"zn2: {parts[3]} is compared with outputs/screen/fits/{card_fit(b, sp)} alone, not {a.base}")
    if SR.parts_of(a.out) != ("outputs", "screen", parts[3]):
        raise SystemExit(f"zn2: {parts[3]}'s comparison is outputs/screen/{parts[3]}, not {a.out}")
    s = fit_rec(w, a.new)
    if s.get("arm") != b or s.get("split") != sp or s.get("seed") != sd:
        raise SystemExit(f"zn2: {parts[3]} is arm {s.get('arm')} with seed {s.get('seed')} on {s.get('split')}, not "
                         f"{b} with seed {sd} on {sp}")
    why = stamp_ok(b, s, sd, sp)
    if why:
        raise SystemExit(f"zn2: {parts[3]}: {why}")
    if run is None and not os.path.samefile(Path.cwd(), w.root):
        raise SystemExit("zn2: compare runs from the repository's root")
    return (run or S2.main)(["compare", "--new", a.new, "--base", a.base, "--out", a.out])


# ── the fresh floors ─────────────────────────────────────────────────────────


def floors(w, base):
    """outputs/n2/floors-<base>.json and .md: N2's null of base, read by read beside the 7 October floors. Exit 0 with
    status OK; exit 2 after writing a REFUSED record."""
    out = w.floors(base)
    head = {"stage": STAGE, "declared_in": DECLARED, "base_arm": base,
            "card": {sp: card_fit(base, sp) for sp in SPLITS}}
    try:
        got, sources, beside = seed_fits(base, [str(x) for x in w.nulls(base)], w)
        null = null_of(got)
        old = load(w.p(OLD_FLOORS))
        o = {(r["split"], r["dataset"]): r for r in old["reads"]}
        reads = []
        for r in floor_reads(null):
            f7 = o[(r["split"], r["dataset"])]
            reads.append({**r, "floor_7oct": f7["floor"], "seed0_7oct": f7["seed0"],
                          "change": "up" if r["floor"] > f7["floor"] else "down" if r["floor"] < f7["floor"] else
                          "same"})
        raised = [f"{r['split']} {r['dataset']}" for r in reads if r["raised"]]
        rec = {**head, "status": "OK", "null": sources, "beside": beside, "rule": RULE, "floor_min": S.FLOOR,
               "reads": reads, "raised": raised, "raised_7oct": old["raised"], "floors_7oct": OLD_FLOORS,
               "zn2_sha256": LC.sha_src(__file__), "utc": utc()}
        md = [f"# N2's fresh floors on {base}", "",
              f"{base}'s card fit of each split ({card_fit(base, 'L-musique')}, {card_fit(base, 'L-hotpotqa')}) trained "
              "again with seeds 1 and 2, its settings otherwise the card's; each compared with the card fit (seed 0) "
              f"on the six s1eval carves (p@swa, R@5). {RULE}. The 7 Oct columns are step 1's null "
              f"({OLD_FLOORS}). Bold: a floor above 0.0075.", "",
              "| split | dataset | read | card R@5 | D1 (seed 1) | D2 (seed 2) | spread | floor | 7 Oct seed 0 | 7 Oct floor "
              "| change |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in reads:
            f = f"{r['floor']:.4f}"
            f7 = f"{r['floor_7oct']:.4f}"
            md.append(f"| {r['split']} | {r['dataset']} | {r['read']} | {r['seed0']:.4f} | {r['d1']:+.4f} | "
                      f"{r['d2']:+.4f} | {r['spread']:.4f} | {'**' + f + '**' if r['raised'] else f} | "
                      f"{r['seed0_7oct']:.4f} | {'**' + f7 + '**' if r['floor_7oct'] > S.FLOOR else f7} | "
                      f"{r['change']} |")
        md += ["", f"Floors above 0.0075 now: {len(raised)} of {len(reads)}" + (f" ({', '.join(raised)})." if raised
                                                                                 else "."),
               f"7 Oct: {len(old['raised'])} of 12 ({', '.join(old['raised'])})."]
        if beside:
            md += ["", "Build records that differ from the card fit's (reported, deciding nothing): "
                   + "; ".join(f"{k}: {', '.join(v)}" for k, v in beside.items()) + "."]
        rc = 0
        log(f"zn2 floors {base}: {len(raised)} of 12 raised now ({raised}); 7 Oct {old['raised']}")
    except SystemExit as e:
        rec = {**head, "status": "REFUSED", "reason": clean(e, w), "zn2_sha256": LC.sha_src(__file__), "utc": utc()}
        md = [f"# N2's fresh floors on {base}: REFUSED", "", clean(e, w)]
        rc = 2
        log(f"zn2 floors {base}: REFUSED: {clean(e, w)}")
    write(out, rec, md)
    return rc


# ── the fresh calls ──────────────────────────────────────────────────────────


def filed_of(vrec, st, arm):
    """The filed verdict and calls of (st, arm), from a verdict file in either form (per-arm, or one arm)."""
    if vrec.get("stage") != st:
        raise SystemExit(f"zn2: the verdict file is {vrec.get('stage')}'s, not {st}'s")
    if "arms" in vrec:
        v = (vrec.get("arms") or {}).get(arm)
        if v is None:
            raise SystemExit(f"zn2: {st}'s verdict has no arm {arm}")
    elif vrec.get("arm") == arm:
        v = vrec
    else:
        raise SystemExit(f"zn2: {st}'s verdict is {vrec.get('arm')}'s, not {arm}'s")
    return {"verdict": v.get("verdict"), "calls": v.get("calls") or {}}


def tag(r):
    return f"{r['split']} {r['dataset']}"


def recall_md(rec):
    st, arm, b = rec["screen_stage"], rec["arm"], rec["base_arm"]
    pf = "PASS_FRESH" if rec["pass_fresh"] else "NO_PASS_FRESH"
    md = [f"# {st} {arm} called under N2's fresh null: **{rec['verdict']}** ({pf})", "",
          f"N2 ({DECLARED}). R@5 of {arm}'s p@swa minus {b}'s card fit of the same split, on the six s1eval carves "
          f"(95% question bootstrap), each read called with its floor from N2's null on {b} (the card fit trained again "
          "with seeds 1 and 2, each against the card fit). The call starts from the pair's (0.0075); the 7 Oct columns "
          f"are the filed re-call's ({rec['filed_recall']}). Filed: {st}'s verdict {rec['stage_verdict']}, its re-call "
          f"{rec['verdict_7oct']}, the pair {rec['verdict_filed']}. Bold: a fresh call that differs from the 7 Oct call.",
          "", f"| split | dataset | read | {b} R@5 | {arm} R@5 | delta R@5 [95% CI] | D1, D2 (N2) | floor now | "
              "floor 7 Oct | pair call | 7 Oct call | fresh call |",
          "|---|---|---|---:|---:|---|---|---:|---:|---|---|---|"]
    for r in rec["rows"]:
        c = f"**{r['call']}**" if r["call"] != r["call_7oct"] else r["call"]
        md.append(f"| {r['split']} | {r['dataset']} | {r['read']} | {r['base']:.4f} | {r['new']:.4f} | "
                  f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['null_d'][0]:+.4f}, {r['null_d'][1]:+.4f} | "
                  f"{r['floor']:.4f} | {r['floor_7oct']:.4f} | {r['call_filed']} | {r['call_7oct']} | {c} |")
    g = [tag(r) for r in rec["rows"] if r["call"] == "GAIN"]
    lo = [tag(r) for r in rec["rows"] if r["call"] == "LOSS"]
    md += ["", f"Fresh: GAIN on {', '.join(g) or 'none'}; LOSS on {', '.join(lo) or 'none'}. PASS_FRESH needs at least "
               "one GAIN and no LOSS.",
           "Changed from the 7 Oct calls: " + ("; ".join(rec["changed_from_7oct"]) or "none") + ".",
           f"Within {SR.ROUNDING} of the floor (the call stands): " + ("; ".join(rec["near_floor"]) or "none") + ".",
           "The filed verdict is not rewritten (N2, section 5)."]
    return md


def recall_arm(w, st, arm):
    """One filed arm called from its pair with its base's fresh floors -> outputs/n2/recall-<stage>-<arm>.json, .md."""
    base = base_of(arm)
    vp, fp, pp = w.verdict(st), w.filed(st, arm), w.pair(st, arm)
    for p in (vp, fp, pp):
        if not p.exists():
            raise SystemExit(f"zn2: {w.rel(p)} missing")
    v = filed_of(load(vp), st, arm)
    adopted = (st, arm) in ADOPTED
    if (v["verdict"] == "ADOPT_FOR_FULL_RUN") != adopted:
        raise SystemExit(f"zn2: {st} {arm} is filed {v['verdict']}; N2 declared it "
                         f"{'adopted' if adopted else 'not adopted'}")
    filed = load(fp)
    if filed.get("stage") != st or filed.get("arm") != arm or filed.get("base_arm") != base:
        raise SystemExit(f"zn2: {w.rel(fp)} is {filed.get('stage')} {filed.get('arm')} on {filed.get('base_arm')}, not "
                         f"{st} {arm} on {base}")
    calls = {tag(r): r["call"] for r in filed.get("rows") or []}
    if calls != v["calls"]:
        raise SystemExit(f"zn2: {w.rel(fp)}'s calls are not {st}'s filed verdict's")
    if repo_rel(filed.get("pair")) != w.rel(pp):
        raise SystemExit(f"zn2: {w.rel(fp)} re-called {repo_rel(filed.get('pair'))}, not {w.rel(pp)}")
    fresh = call_pair(lambda fs: load_null_n2(base, fs, w), [str(x) for x in w.nulls(base)], str(pp))
    f7 = {(r["split"], r["dataset"]): r for r in filed["rows"]}
    if sorted(f7) != sorted((r["split"], r["dataset"]) for r in fresh["rows"]):
        raise SystemExit(f"zn2: {w.rel(fp)} and {w.rel(pp)} hold different reads")
    rows = [{**r, "call_7oct": f7[(r["split"], r["dataset"])]["call"],
             "floor_7oct": f7[(r["split"], r["dataset"])]["floor"]} for r in fresh["rows"]]
    rec = {"stage": STAGE, "declared_in": DECLARED, "screen_stage": st, "arm": arm, "base_arm": base,
           "verdict": fresh["verdict"], "verdict_filed": fresh["verdict_filed"], "verdict_7oct": filed.get("verdict"),
           "stage_verdict": v["verdict"], "pass_fresh": fresh["verdict"] == "PROMISING", "adopted": adopted,
           "pair": w.rel(pp), "null": fresh["null"], "rule": RULE, "floor_min": S.FLOOR, "fits": fresh["fits"],
           "rows": rows, "changed_from_pair": fresh["changed"],
           "changed_from_7oct": [f"{tag(r)} {r['call_7oct']} -> {r['call']}" for r in rows if r["call"] != r["call_7oct"]],
           "near_floor": fresh["near_floor"], "filed_recall": w.rel(fp), "filed_recall_sha256": LC.sha_file(fp),
           "zn2_sha256": LC.sha_src(__file__), "utc": utc()}
    write(w.recall_out(st, arm), rec, recall_md(rec))
    return rec


def floors_ok(w, base):
    p = w.floors(base).with_suffix(".json")
    if not p.exists():
        return None, f"{w.rel(p)} missing"
    r = load(p)
    if r.get("stage") != STAGE or r.get("declared_in") != DECLARED or r.get("base_arm") != base or \
            r.get("status") != "OK":
        return None, f"{w.rel(p)} is {r.get('stage')} {r.get('base_arm')} with status {r.get('status')}, not N2's OK"
    return r, None


def recall_all(w):
    """Every arm of ARMS called with its base's fresh floors; outputs/n2/recall-all.json, .md (exit 0 once written)."""
    top, ok = [], {}
    for b in BASES:
        _, why = floors_ok(w, b)
        ok[b] = why is None
        if why:
            top.append(why)
    per = []
    for st, arm in ARMS:
        b = base_of(arm)
        row = {"screen_stage": st, "arm": arm, "base_arm": b}
        if not ok[b]:
            row.update(status="NOT_RUN", problems=[f"{b}'s floors are not N2's OK record"])
        else:
            try:
                rec = recall_arm(w, st, arm)
                out = w.recall_out(st, arm).with_suffix(".json")
                row.update(status="OK", verdict=rec["verdict"], verdict_7oct=rec["verdict_7oct"],
                           verdict_filed=rec["verdict_filed"], stage_verdict=rec["stage_verdict"],
                           pass_fresh=rec["pass_fresh"], file=w.rel(out), sha256=LC.sha_file(out), problems=[])
            except SystemExit as e:
                row.update(status="FAILED", problems=[clean(e, w)])
            except (KeyError, ValueError, TypeError, OSError) as e:
                row.update(status="FAILED", problems=[f"{type(e).__name__}: {clean(e, w)}"])
        per.append(row)
    rec = {"stage": STAGE, "declared_in": DECLARED, "arms": per, "problems": top, "zn2_sha256": LC.sha_src(__file__),
           "utc": utc()}
    md = ["# N2: the 56 arms called under the fresh floors", "",
          f"Each filed arm's pair called with its base's fresh floors (outputs/n2/floors-zrckp.json, floors-zspk.json; "
          f"{DECLARED}). The filed verdicts are not rewritten.", "",
          "| stage | arm | base | status | pair (0.0075) | 7 Oct floors | fresh | filed verdict | PASS_FRESH |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in per:
        md.append(f"| {x['screen_stage']} | {x['arm']} | {x['base_arm']} | {x['status']} | {x.get('verdict_filed', '-')} "
                  f"| {x.get('verdict_7oct', '-')} | {x.get('verdict', '-')} | {x.get('stage_verdict', '-')} | "
                  f"{x.get('pass_fresh', '-')} |")
    bad = [f"{x['screen_stage']} {x['arm']}: {x['status']} ({'; '.join(x['problems'])})" for x in per if x["status"] != "OK"]
    md += ["", "Problems: " + ("; ".join(top + bad) or "none") + "."]
    write(w.out / "recall-all", rec, md)
    log(f"zn2 recall-all: {sum(x['status'] == 'OK' for x in per)} of {len(per)} called; "
        f"{sum(bool(x.get('pass_fresh')) for x in per)} PASS_FRESH; problems {len(top) + len(bad)}")
    return 0


# ── the reproduction check ───────────────────────────────────────────────────


def repro_calls(w, base, arm, filed, pair_file, rel_screens, old_null, check_sources=True):
    """The fields of a filed re-call that the stages' path (relz.rel_null under the card fits, screen_recall.pair) does
    not give back from the pair and the OLD null; [] when it reproduces."""
    with on_card(base, arm, rel_screens):
        got = call_pair(lambda fs: Z.rel_null(fs, rel_screens), [str(x) for x in old_null], str(pair_file))
    got = json.loads(json.dumps(got))
    bad = [k for k in ("verdict", "verdict_filed", "fits", "rows", "changed", "near_floor") if got.get(k) != filed.get(k)]
    if check_sources:
        a = {k: repo_rel(x) for k, x in (got.get("null") or {}).items()}
        b = {k: repo_rel(x) for k, x in (filed.get("null") or {}).items()}
        if a != b:
            bad.append("null")
    return bad


def repro_arm(w, st, arm):
    base = base_of(arm)
    fp, pp = w.filed(st, arm), w.pair(st, arm)
    if not fp.exists():
        return {"status": "NOT_CHECKED", "why": f"{w.rel(fp)} missing"}
    filed = load(fp)
    probs = []
    if filed.get("stage") != st or filed.get("arm") != arm or filed.get("base_arm") != base:
        probs.append(f"the re-call is {filed.get('stage')} {filed.get('arm')} on {filed.get('base_arm')}")
    nul = filed.get("null") or {}
    if [repo_rel(nul.get(f"{sp} seed {sd}")) for sp in SPLITS for sd in SEEDS] != list(OLD_NULL):
        probs.append("its null is not step 1's (7 Oct)")
    if repo_rel(filed.get("pair")) != w.rel(pp):
        probs.append(f"it re-called {repo_rel(filed.get('pair'))}")
    if probs:
        return {"status": "MISMATCH", "fields": probs}
    if not pp.exists():
        return {"status": "NOT_CHECKED", "why": f"{w.rel(pp)} missing"}
    rel = {}
    for sp in SPLITS:
        r = repo_rel(nul.get(f"rel {sp}"))
        if r is None or not w.p(r).exists():
            return {"status": "NOT_CHECKED", "why": f"{base}'s card comparison of {sp} ({r or 'no repository path'}) "
                                                     "missing"}
        rel[sp] = str(w.p(r))
    try:
        bad = repro_calls(w, base, arm, filed, pp, rel, w.old_null())
    except SystemExit as e:
        return {"status": "FAILED", "why": clean(e, w)}
    except (KeyError, ValueError, TypeError, OSError) as e:
        return {"status": "FAILED", "why": f"{type(e).__name__}: {clean(e, w)}"}
    return {"status": "MISMATCH", "fields": bad} if bad else {"status": "REPRODUCED"}


def repro(w):
    """Every filed arm called again with the OLD null through N2's path, and the 7 Oct floors from their compares;
    outputs/n2/repro.json, .md (exit 0 once written). REPRODUCED needs no MISMATCH or FAILED arm, at least one
    REPRODUCED arm per base and the 7 Oct floors reproduced."""
    per = []
    for st, arm in ARMS:
        per.append({"screen_stage": st, "arm": arm, "base_arm": base_of(arm), **repro_arm(w, st, arm)})
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            f = SR.floors([str(x) for x in w.old_null()])
        old = load(w.p(OLD_FLOORS))
        f7 = "REPRODUCED" if f["reads"] == old["reads"] and f["raised"] == old["raised"] else "MISMATCH"
    except (SystemExit, OSError, ValueError, KeyError) as e:
        f7 = f"FAILED: {clean(e, w)}"
    by_base = {b: sum(x["status"] == "REPRODUCED" and x["base_arm"] == b for x in per) for b in BASES}
    bad = [x for x in per if x["status"] in ("MISMATCH", "FAILED")]
    ok = not bad and all(by_base.values()) and f7 == "REPRODUCED"
    counts = {s: sum(x["status"] == s for x in per) for s in ("REPRODUCED", "NOT_CHECKED", "MISMATCH", "FAILED")}
    rec = {"stage": STAGE, "declared_in": DECLARED, "status": "REPRODUCED" if ok else "NOT_REPRODUCED",
           "floors_7oct": f7, "reproduced_by_base": by_base, "counts": counts, "arms": per,
           "zn2_sha256": LC.sha_src(__file__), "utc": utc()}
    md = [f"# N2's reproduction check: **{rec['status']}**", "",
          "Each filed arm's pair called again with the OLD null (step 1's, 7 Oct) through N2's path: relz.rel_null "
          "with the base's card comparisons and screen_recall.pair, unchanged. It must give the filed re-call's "
          "verdicts, rows (calls, floors, deltas), changes and near-floor reads exactly; NOT_CHECKED: an input is not "
          f"on this machine. 7 Oct floors from their compares: {f7}.", "",
          "| stage | arm | base | status | detail |", "|---|---|---|---|---|"]
    for x in per:
        md.append(f"| {x['screen_stage']} | {x['arm']} | {x['base_arm']} | {x['status']} | "
                  f"{', '.join(x.get('fields', [])) or x.get('why', '')} |")
    md += ["", f"Counts: {counts}. Reproduced per base: {by_base}."]
    write(w.out / "repro", rec, md)
    log(f"zn2 repro: {rec['status']}; {counts}; 7 Oct floors {f7}")
    return 0


# ── the summary and the gates ────────────────────────────────────────────────


def late_arms(w):
    """Arms filed since 15 October on a zrckp or zspk base that N2 did not declare (P1g-b's and N2's aside): listed."""
    out = []
    for p in sorted((w.root / "outputs").glob("*/screen-verdict.json")):
        try:
            rec = load(p)
        except (OSError, ValueError):
            continue
        st = rec.get("stage")
        if not st or rec.get("utc", "") < SINCE or st in ("P1gb", STAGE):
            continue
        arms = list((rec.get("arms") or {})) if "arms" in rec else [rec.get("arm")]
        for a in arms:
            if not a or (st, a) in ARMS:
                continue
            fp = w.filed(st, a)
            f = load(fp) if fp.exists() else {}
            if f.get("base_arm") in BASES:
                v = (rec.get("arms") or {}).get(a, {}).get("verdict") if "arms" in rec else rec.get("verdict")
                out.append({"screen_stage": st, "arm": a, "base_arm": f["base_arm"], "verdict": v, "file": w.rel(p)})
    return out


def summary(w):
    """outputs/n2/summary.json, .md: the floors then and now, the calls (pair, 7 Oct, fresh), the ten adopted arms
    PASS_FRESH or NO_PASS_FRESH, the filed NOT_ADOPTED arms that pass fresh, and section 5's routing; COMPLETE only when
    both floors, all 56 calls and the reproduction check are in (exit 0 once written)."""
    problems, fl = [], {}
    for b in BASES:
        r, why = floors_ok(w, b)
        if why:
            problems.append(why)
        else:
            fl[b] = {(x["split"], x["dataset"]): x for x in r["reads"]}
    per = {}
    ra = w.out / "recall-all.json"
    if not ra.exists():
        problems.append("outputs/n2/recall-all.json missing")
    else:
        rall = load(ra)
        if rall.get("stage") != STAGE or rall.get("declared_in") != DECLARED:
            problems.append("outputs/n2/recall-all.json is not N2's")
        if [(x.get("screen_stage"), x.get("arm")) for x in rall.get("arms", [])] != list(ARMS):
            problems.append("recall-all does not hold N2's 56 arms in order")
        problems += list(rall.get("problems") or [])
        for x in rall.get("arms", []):
            k = f"{x.get('screen_stage')} {x.get('arm')}"
            if x.get("status") != "OK":
                problems.append(f"{k}: {x.get('status')} ({'; '.join(x.get('problems') or [])})")
                continue
            fp = w.p(x["file"])
            if not fp.exists() or LC.sha_file(fp) != x.get("sha256"):
                problems.append(f"{k}: {x['file']} is not the file recall-all wrote")
                continue
            rec = load(fp)
            if rec.get("stage") != STAGE or rec.get("declared_in") != DECLARED or \
                    (rec.get("screen_stage"), rec.get("arm")) != (x["screen_stage"], x["arm"]):
                problems.append(f"{k}: {x['file']} is not N2's call of the arm")
                continue
            b = rec["base_arm"]
            if b in fl and any(r["floor"] != fl[b][(r["split"], r["dataset"])]["floor"] for r in rec["rows"]):
                problems.append(f"{k}: its floors are not outputs/n2/floors-{b}.json's")
                continue
            per[(x["screen_stage"], x["arm"])] = rec
    rp = w.out / "repro.json"
    rpr = load(rp) if rp.exists() else {}
    if rpr.get("stage") != STAGE or rpr.get("status") != "REPRODUCED":
        problems.append(f"the reproduction check is {rpr.get('status') or 'missing'}, not REPRODUCED")
    status = "INCOMPLETE" if problems else "COMPLETE"

    floors_tab = []
    for sp in SPLITS:
        for ds in LG.EVAL_ORDER:
            row = {"split": sp, "dataset": ds}
            for b in BASES:
                x = fl.get(b, {}).get((sp, ds))
                if x:
                    row.update({"seed0_7oct": x["seed0_7oct"], "floor_7oct": x["floor_7oct"], f"seed0_{b}": x["seed0"],
                                f"floor_{b}": x["floor"], f"d_{b}": [x["d1"], x["d2"]]})
            floors_tab.append(row)
    counts = []
    for b in BASES:
        for sp in SPLITS:
            for ds in LG.EVAL_ORDER:
                rows = [r for k, rec in per.items() if rec["base_arm"] == b for r in rec["rows"]
                        if (r["split"], r["dataset"]) == (sp, ds)]
                counts.append({"base_arm": b, "split": sp, "dataset": ds, "arms": len(rows),
                               **{f"{w_}_{c.lower()}": sum(r[f] == c for r in rows)
                                  for w_, f in (("pair", "call_filed"), ("oct7", "call_7oct"), ("fresh", "call"))
                                  for c in ("GAIN", "LOSS")}})
    totals = {b: {f"{w_}_{c.lower()}": sum(x[f"{w_}_{c.lower()}"] for x in counts if x["base_arm"] == b)
                  for w_ in ("pair", "oct7", "fresh") for c in ("GAIN", "LOSS")} for b in BASES}
    adopted, promoted = [], []
    for st, arm in ARMS:
        rec = per.get((st, arm))
        if rec is None:
            continue
        g = [tag(r) for r in rec["rows"] if r["call"] == "GAIN"]
        lo = [tag(r) for r in rec["rows"] if r["call"] == "LOSS"]
        if (st, arm) in ADOPTED:
            refit = GATED.get(st)
            adopted.append({"screen_stage": st, "arm": arm, "base_arm": rec["base_arm"], "pass_fresh": rec["pass_fresh"],
                            "call": "PASS_FRESH" if rec["pass_fresh"] else "NO_PASS_FRESH", "verdict": rec["verdict"],
                            "verdict_7oct": rec["verdict_7oct"], "gains": g, "losses": lo,
                            "refit_stage": refit[0] if refit else None, "refit_doc": refit[1] if refit else None,
                            "gated": refit is not None})
        elif rec["pass_fresh"]:
            promoted.append({"screen_stage": st, "arm": arm, "base_arm": rec["base_arm"], "verdict_7oct":
                             rec["verdict_7oct"], "stage_verdict": rec["stage_verdict"], "gains": g,
                             "next_form": f"a refit on P1g-b's pools in its own declared file "
                                          f"(docs/{st.upper()}P_P1GB_POOLS.md, stage {st}p; to be declared, not by N2)"})
    late = late_arms(w)
    routing = []
    if status == "COMPLETE":
        for st, (rst, doc) in GATED.items():
            arms = [a for a in adopted if a["screen_stage"] == st]
            ok = all(a["pass_fresh"] for a in arms)
            routing.append(f"{st}: " + ", ".join(f"{a['arm']} {a['call']}" for a in arms)
                           + f" -> n2-gate-{st.lower()} exits {0 if ok else 1}: {rst}'s refits ({doc}) "
                           + ("run." if ok else "do not run."))
            for a in arms:
                if not a["pass_fresh"]:
                    routing.append(f"{st} {a['arm']}: next form, a second-seed screen of {a['arm']} in its own declared "
                                   "file (never closed).")
                elif not ok:
                    routing.append(f"{st} {a['arm']} passes alone: its refit may run behind `zn2.py gate --stage "
                                   f"{st.lower()} --arm {a['arm']}` (a feeder edit, not N2's).")
        for a in adopted:
            if a["screen_stage"] in UNGATED:
                routing.append(f"{a['screen_stage']} {a['arm']}: {a['call']}" + (
                    "." if a["pass_fresh"] else f"; a caveat on every later stage built on {a['arm']}."))
        for a in promoted:
            routing.append(f"{a['screen_stage']} {a['arm']}: filed {a['stage_verdict']} (7 Oct re-call "
                           f"{a['verdict_7oct']}), PASS_FRESH -> {a['next_form']}.")
    later = ("Later screens on zrckp or zspk bases are called with N2's floors (outputs/n2/floors-zrckp.json, "
             "floors-zspk.json); zspx bases with zspk's and zrckpr bases with zrckp's, labelled as their parent's.")
    rec = {"stage": STAGE, "declared_in": DECLARED, "status": status, "problems": problems, "floors": floors_tab,
           "counts": counts, "totals": totals, "adopted": adopted, "promoted": promoted, "late_arms": late,
           "routing": routing, "later_floors": later, "filed_verdicts": "not rewritten",
           "todays_pools": "no model change is adopted on today's pools (13 October)",
           "reproduction": {k: rpr.get(k) for k in ("status", "floors_7oct", "reproduced_by_base", "counts")},
           "zn2_sha256": LC.sha_src(__file__), "utc": utc()}
    md = [f"# N2: the fresh seed null on today's lean pair: **{status}**", "", f"Declared in {DECLARED}. {RULE}.", ""]
    if problems:
        md += ["Problems (nothing is routed until each is resolved; section 5):"] + [f"- {x}" for x in problems] + [""]
    md += ["## Floors, 7 Oct and now", "",
           "| split | dataset | 7 Oct seed 0 | 7 Oct floor | zrckp card | zrckp D1, D2 | zrckp floor | zspk card | "
           "zspk D1, D2 | zspk floor |", "|---|---|---:|---:|---:|---|---:|---:|---|---:|"]
    num = lambda x, f="{:.4f}": f.format(x) if isinstance(x, (int, float)) else "-"   # noqa: E731
    pair_ = lambda x: f"{x[0]:+.4f}, {x[1]:+.4f}" if isinstance(x, list) else "-"     # noqa: E731
    for r in floors_tab:
        md.append(f"| {r['split']} | {r['dataset']} | {num(r.get('seed0_7oct'))} | {num(r.get('floor_7oct'))} | "
                  f"{num(r.get('seed0_zrckp'))} | {pair_(r.get('d_zrckp'))} | {num(r.get('floor_zrckp'))} | "
                  f"{num(r.get('seed0_zspk'))} | {pair_(r.get('d_zspk'))} | {num(r.get('floor_zspk'))} |")
    md += ["", "## Calls on the 56 arms, read by read (GAIN / LOSS)", "",
           "| base | split | dataset | arms | pair (0.0075) | 7 Oct floors | N2 floors |", "|---|---|---|---:|---|---|---|"]
    for x in counts:
        md.append(f"| {x['base_arm']} | {x['split']} | {x['dataset']} | {x['arms']} | {x['pair_gain']} / {x['pair_loss']} "
                  f"| {x['oct7_gain']} / {x['oct7_loss']} | {x['fresh_gain']} / {x['fresh_loss']} |")
    for b, t in totals.items():
        md.append(f"| {b} | all | all | | {t['pair_gain']} / {t['pair_loss']} | {t['oct7_gain']} / {t['oct7_loss']} | "
                  f"{t['fresh_gain']} / {t['fresh_loss']} |")
    md += ["", "## The ten adopted arms", "", "| stage | arm | base | 7 Oct | fresh | call | GAIN | LOSS | refit | gated |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for a in adopted:
        md.append(f"| {a['screen_stage']} | {a['arm']} | {a['base_arm']} | {a['verdict_7oct']} | {a['verdict']} | "
                  f"**{a['call']}** | {', '.join(a['gains']) or '-'} | {', '.join(a['losses']) or '-'} | "
                  f"{a['refit_stage'] or '-'} | {'yes' if a['gated'] else 'no'} |")
    md += ["", "## Filed NOT_ADOPTED arms that pass under N2's floors", ""]
    md += [f"- {a['screen_stage']} {a['arm']} ({a['base_arm']}): GAIN on {', '.join(a['gains'])}; next: {a['next_form']}."
           for a in promoted] or ["- none"]
    md += ["", "## Arms filed after N2 was declared (listed; they decide nothing)", ""]
    md += [f"- {a['screen_stage']} {a['arm']} ({a['base_arm']}): {a['verdict']}" for a in late] or ["- none"]
    md += ["", "## Routing (section 5)", ""]
    md += [f"- {x}" for x in routing] or ["- none: the summary is INCOMPLETE; the missing piece is run and nothing is "
                                          "routed."]
    md += ["", later, "Filed verdicts are not rewritten. No model change is adopted on today's pools (13 October)."]
    write(w.out / "summary", rec, md)
    log(f"zn2 summary: {status}; adopted "
        + ", ".join(f"{a['screen_stage']} {a['arm']} {a['call']}" for a in adopted)
        + (f"; problems {len(problems)}" if problems else ""))
    return 0


def gate(w, stage, arm=None):
    """Exit 0 when the stage's adopted arm(s) are PASS_FRESH in N2's COMPLETE summary, 1 when one is NO_PASS_FRESH, 2
    when it cannot be decided (not a gated stage or arm, no summary, or an INCOMPLETE one)."""
    st = GATE_STAGES.get(str(stage).lower())
    if st is None:
        log(f"zn2 gate: {stage} is not a gated stage ({', '.join(GATE_STAGES)})")
        return 2
    want = sorted(a for s, a in ADOPTED if s == st)
    if arm is not None and arm not in want:
        log(f"zn2 gate: {arm} is not an adopted arm of {st} ({', '.join(want)})")
        return 2
    p = w.out / "summary.json"
    if not p.exists():
        log("zn2 gate: outputs/n2/summary.json missing")
        return 2
    rec = load(p)
    if rec.get("stage") != STAGE or rec.get("declared_in") != DECLARED or rec.get("status") != "COMPLETE":
        log(f"zn2 gate: the summary is {rec.get('stage')} {rec.get('declared_in')} {rec.get('status')}, not N2's COMPLETE")
        return 2
    arms = [a for a in rec.get("adopted", []) if a.get("screen_stage") == st]
    if sorted(a.get("arm") for a in arms) != want:
        log(f"zn2 gate: the summary holds {st}'s adopted arms {[a.get('arm') for a in arms]}, not {want}")
        return 2
    if arm is not None:
        arms = [a for a in arms if a["arm"] == arm]
    ok = all(a.get("pass_fresh") is True for a in arms)
    log(f"zn2 gate {st}{' ' + arm if arm else ''}: " + ", ".join(f"{a['arm']} {a.get('call')}" for a in arms)
        + f" -> exit {0 if ok else 1}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def must_stop(fn, *args, **kw):
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            fn(*args, **kw)
    except SystemExit:
        return
    raise AssertionError(f"{fn.__name__} must stop")


@contextlib.contextmanager
def mutated(p, fn):
    """A JSON file changed by fn while the block runs, then restored byte for byte."""
    p = Path(p)
    raw = p.read_bytes()
    rec = json.loads(raw)
    fn(rec)
    p.write_text(json.dumps(rec), encoding="utf-8")
    try:
        yield
    finally:
        p.write_bytes(raw)


@contextlib.contextmanager
def moved_away(p):
    p = Path(p)
    tmp = p.with_name(p.name + ".away")
    p.replace(tmp)
    try:
        yield
    finally:
        tmp.replace(p)


@contextlib.contextmanager
def null_fit_records():
    """Selftest only: the 7 Oct null's fit records live on the host; a missing one named scr-null-s1, -s2, -s1-hp or
    -s2-hp stands in as step 1's base arm with the split and seed its name gives."""
    real = SR.fit_record
    pat = re.compile(r"^scr-null-s([12])(-hp)?$")

    def stand_in(new):
        parts = SR.parts_of(new)
        p = Path(str(new))
        d = p if p.is_absolute() else ROOT / Path(*parts)
        m = pat.match(parts[-1]) if parts else None
        if (d / "screen.json").exists() or not m:
            return real(new)
        return {"arm": "base", "split": "L-hotpotqa" if m.group(2) else "L-musique", "seed": int(m.group(1))}

    SR.fit_record = stand_in
    try:
        yield
    finally:
        SR.fit_record = real


def snapshot():
    """(size, mtime) of every file under the stage roots N2 reads; outputs/n2 must not exist."""
    roots = {ROOT / "outputs" / x for x in ("screen", "l1", "p1c2", "p1c3", "p1c3p")}
    roots |= {ROOT / "outputs" / st.lower() for st, _ in ARMS}
    out = {}
    for r in sorted(roots):
        if not r.exists():
            out[r.name] = None
            continue
        for p in r.rglob("*"):
            if p.is_file():
                s = p.stat()
                out[p.relative_to(ROOT).as_posix()] = (s.st_size, s.st_mtime_ns)
    return out


def toy_card(w, base, split):
    """A card fit's record in the toy root (P1c3p's or P1c3's stamp; zspk's card holds no zp1c3p_sha256)."""
    rec = {"name": card_fit(base, split), "arm": base, "split": split, "seed": 0, "train": "toy=fit", "basis": "toy",
           "variants": "p", "config": "toy", "hidden": 128, "cache_root": "outputs/p1c2/cache",
           **{k: LC.sha_src(m.__file__) for k, m in CODE.items()}, "rmatch": {"toy": 1}, "rmatch_records": {"a": "1"},
           STAMP[base][0]: {"declared_in": STAMP[base][1], "arm": base, "base_arm": "zrc" if base == "zrckp" else "zsp",
                            "device": "cuda", "graph": "today's", "pools": "toy pools", "cache_root": "outputs/p1c2/cache",
                            "head": {"toy": 1}, "gate": "g", "kt_records": {"k": "1"}, "rmatch_records": {"a": "1"},
                            "zrc_records": {"z": "1"}}}
    if base == "zspk":
        rec.pop("zp1c3p_sha256")
        rec[STAMP[base][0]]["zlink_records"] = {"l": "1"}
    LC.write_json(w.card_record(base, split), rec)
    return rec


def toy_null(w, base, big, r5=0.6):
    """N2's four comparisons of base in the toy root (big = {(split, dataset): (D1, D2)}, every other read +-0.002)."""
    files = []
    for sp in SPLITS:
        card = toy_card(w, base, sp)
        for sd in SEEDS:
            name = n2_fit(base, sd, sp)
            fit = {**json.loads(json.dumps(card)), "name": name, "seed": sd}
            if base == "zrckp":
                fit["n2"] = {"declared_in": DECLARED, "base_arm": "zrckp", "seed": sd, "split": sp,
                             "card_fit": card_fit(base, sp)}
            LC.write_json(w.fit_dir(name) / "screen.json", fit)
            rows = []
            for ds in LG.EVAL_ORDER:
                d = big.get((sp, ds), (0.002, -0.002))[sd - 1]
                lo, hi = round(d - 0.004, 4), round(d + 0.004, 4)
                rows.append({"dataset": ds, "read": "in-domain" if ds in LG.FITS[sp] else "zero-shot", "questions": 100,
                             "base": [r5, 0, 0], "new": [round(r5 + d, 4), 0, 0], "rrf": [0.4, 0, 0], "delta": [d, 0, 0],
                             "lo": [lo, 0, 0], "hi": [hi, 0, 0], "call": S.call(d, lo, hi)})
            b0 = f"outputs\\screen\\fits\\{card_fit(base, sp)}"
            v = S.verdict([r["call"] for r in rows])
            rec = {"new": f"outputs\\screen\\fits\\{name}", "bases": [b0], "candidate": "p@swa", "carve": "s1eval",
                   "floor": S.FLOOR, "trained_on": sorted(LG.FITS[sp]),
                   "by_base": {b0: {"rows": rows, "verdict": v, "decides": True}}, "verdict": v}
            p = w.screen / f"{name}.json"
            LC.write_json(p, rec)
            files.append(p)
    return files


def toy_pair(w, st, arm, deltas, r5=0.6):
    rows = []
    for sp in SPLITS:
        for ds in LG.EVAL_ORDER:
            d = deltas.get(sp, {}).get(ds, 0.0)
            lo, hi = round(d - 0.004, 4), round(d + 0.004, 4)
            rows.append({"split": sp, "dataset": ds, "read": "in-domain" if ds in LG.FITS[sp] else "zero-shot",
                         "base": r5, "new": round(r5 + d, 4), "rrf": 0.4, "delta": d, "lo": lo, "hi": hi,
                         "call": S.call(d, lo, hi)})
    fits = [{"file": f"outputs/screen/scr-{st.lower()}-{arm}{SUF[sp]}.json", "split": sp,
             "verdict": S.verdict([r["call"] for r in rows if r["split"] == sp])} for sp in SPLITS]
    LC.write_json(w.pair(st, arm), {"verdict": S.verdict([r["call"] for r in rows]), "floor": S.FLOOR, "fits": fits,
                                    "rows": rows, "stage": st, "arm": arm, "base_arm": base_of(arm)})


def toy_filed(w, st, arm, rel_screens):
    """A stage's filed re-call, made by the stages' own path (relz.recall under the card fits, as za5c.recall)."""
    out = w.filed(st, arm).with_suffix("")
    with on_card(base_of(arm), arm, rel_screens), contextlib.redirect_stdout(io.StringIO()):
        Z.recall([str(x) for x in w.old_null()], str(w.pair(st, arm)), out, rel_screens)
    js = out.with_suffix(".json")
    r = load(js)
    r.update({"stage": st, "declared_in": f"docs/{st.upper()}_TOY.md"})
    LC.write_json(js, r)
    return r


def toy_verdicts(w, filed, when="2026-10-17T00:00:00Z"):
    single = {"G2", "G3", "G5", "A11", "A12", "A13"}
    by_stage = {}
    for (st, arm), r in filed.items():
        calls = {tag(x): x["call"] for x in r["rows"]}
        g = [k for k, c in calls.items() if c == "GAIN"]
        lo = [k for k, c in calls.items() if c == "LOSS"]
        by_stage.setdefault(st, {})[arm] = {"verdict": "ADOPT_FOR_FULL_RUN" if g and not lo else "NOT_ADOPTED",
                                            "calls": calls, "delta_r5": {tag(x): x["delta"] for x in r["rows"]}}
    for st, per in by_stage.items():
        if st in single:
            (arm, v), = per.items()
            rec = {"stage": st, "arm": arm, **v, "utc": when}
        else:
            rec = {"stage": st, "arms": per, "utc": when}
        LC.write_json(w.verdict(st), rec)


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def selftest():
    t0 = time.time()
    snap0 = snapshot()
    assert not (ROOT / "outputs" / "n2").exists(), "outputs/n2 exists before N2 has run"

    # (0) the declared constants
    assert len(ARMS) == 56 and len(set(ARMS)) == 56
    assert sum(base_of(a) == "zrckp" for _, a in ARMS) == 24 and sum(base_of(a) == "zspk" for _, a in ARMS) == 32
    assert len(ADOPTED) == 10 and ADOPTED <= set(ARMS)
    assert tuple(SPLITS) == tuple(SR.NULL_SPLITS) and tuple(SEEDS) == (1, 2)
    assert set(GATED) | set(UNGATED) == {s for s, _ in ADOPTED} and not set(GATED) & set(UNGATED)
    assert RM.__name__ == "rmatch" and Z.__name__ == "relz" and SR.__name__ == "screen_recall"
    assert OLD_NULL == ("outputs/screen/scr-null-s1.json", "outputs/screen/scr-null-s2.json",
                        "outputs/screen/scr-null-s1-hp.json", "outputs/screen/scr-null-s2-hp.json")
    assert [n2_fit("zspk", 1, sp) for sp in SPLITS] == ["scr-n2-zspk-s1", "scr-n2-zspk-s1-hp"]
    assert repo_rel("C:\\a\\b\\outputs\\l1\\x.json") == "outputs/l1/x.json" and repo_rel("a/b") is None
    print("zn2 selftest (0): 56 arms (24 zrckp, 32 zspk), 10 adopted (A3v, A5c, A7 gated; G3, G4, G5, R1 not)",
          flush=True)

    # (a) the 7 Oct floors come back from the 7 Oct compares, by screen_recall.floors and by N2's arithmetic
    filed7 = load(ROOT / OLD_FLOORS)
    with null_fit_records():
        f = quiet(SR.floors, [str(ROOT / x) for x in OLD_NULL])
    assert f["reads"] == filed7["reads"] and f["raised"] == filed7["raised"], "screen_recall.floors: not the filed floors"
    got = {}
    for sp in SPLITS:
        for sd in SEEDS:
            rec = load(ROOT / f"outputs/screen/scr-null-s{sd}{SUF[sp]}.json")
            got[(sp, sd)] = {"rows": {r["dataset"]: r for r in rec["by_base"][rec["bases"][0]]["rows"]}}
    assert floor_reads(null_of(got)) == filed7["reads"], "N2's arithmetic: not the filed floors"
    print(f"zn2 selftest (a): the 7 Oct floors ({len(filed7['raised'])} of 12 raised: {', '.join(filed7['raised'])}) "
          "come back exactly from the 7 Oct compares, by screen_recall.floors and by N2's own arithmetic", flush=True)

    with tempfile.TemporaryDirectory() as td:
        T = Path(td)

        # (b) a toy N2 null and its refusals
        w = Where(T / "rootb")
        zr = [str(x) for x in toy_null(w, "zrckp", {})]
        null, src = load_null_n2("zrckp", zr, w)
        assert all(null[sp][ds]["floor"] == S.FLOOR and null[sp][ds]["seed0"] == 0.6 for sp in SPLITS
                   for ds in LG.EVAL_ORDER)
        assert src == {f"{sp} seed {sd}": f"outputs/screen/{n2_fit('zrckp', sd, sp)}.json" for sp in SPLITS
                       for sd in SEEDS}
        zs = [str(x) for x in toy_null(w, "zspk", {("L-hotpotqa", "musique"): (0.006, -0.006)})]
        null, _ = load_null_n2("zspk", zs, w)                              # zspk's fits carry no n2 stamp
        assert abs(null["L-hotpotqa"]["musique"]["floor"] - 0.012) < 1e-12
        assert null["L-musique"]["musique"]["floor"] == S.FLOOR
        fit = lambda sd, sp="L-musique": w.fit_dir(n2_fit("zrckp", sd, sp)) / "screen.json"      # noqa: E731
        cmp_ = lambda sd, sp="L-musique": w.screen / f"{n2_fit('zrckp', sd, sp)}.json"           # noqa: E731
        refused = []

        def stop(files=None, base="zrckp"):
            must_stop(load_null_n2, base, files or (zr if base == "zrckp" else zs), w)
            refused.append(1)

        stop(zr[:3] + [str(w.screen / "none.json")])                       # a file missing
        stop([zr[0], zr[0], zr[2], zr[3]])                                 # a seed twice
        stop(zr[:3])                                                       # three files
        stop(zr[:3] + [zs[3]])                                             # another base's comparison
        for fn in (lambda r: r.update(seed=3), lambda r: r.update(seed=0), lambda r: r.update(arm="zrckpc"),
                   lambda r: r.update(split="L-hotpotqa"), lambda r: r.pop("n2"),
                   lambda r: r["n2"].update(declared_in="docs/OTHER.md"), lambda r: r["n2"].update(seed=2),
                   lambda r: r["p1c3p"].update(declared_in="docs/OTHER.md"),
                   lambda r: r.update(lean_gpu_sha256="0" * 64), lambda r: r.update(hidden=64),
                   lambda r: r.update(train="other=fit"), lambda r: r["p1c3p"].update(pools="other pools"),
                   lambda r: r["p1c3p"].update(head={"other": 1})):
            with mutated(fit(1), fn):
                stop()
        LC.write_json(w.fit_dir("scr-n2-zrckp-s1-copy") / "screen.json", load(fit(1)))

        def rebase(r, b):
            r["by_base"][b] = r["by_base"].pop(r["bases"][0])
            r["bases"] = [b]

        for fn in (lambda r: rebase(r, "outputs\\step1\\fits\\L-musique"),                  # a step-1 base
                   lambda r: rebase(r, "outputs\\screen\\fits\\scr-p1c3-zspk"),             # the other base's card
                   lambda r: r["bases"].append("outputs\\step1\\fits\\L-musique"),          # two bases
                   lambda r: r.update(candidate="p@full"), lambda r: r.update(carve="s1sel"),
                   lambda r: r.update(trained_on=["metaqa"]),
                   lambda r: r["by_base"][r["bases"][0]]["rows"].pop(),                     # five datasets
                   lambda r: r.update(new="outputs\\screen\\fits\\scr-n2-zrckp-s1-copy")):  # a fit of another name
            with mutated(cmp_(1), fn):
                stop()
        with mutated(cmp_(2), lambda r: r["by_base"][r["bases"][0]]["rows"][0]["base"].__setitem__(0, 0.61)):
            stop()                                                         # the seeds' seed 0 differ
        with moved_away(w.card_record("zrckp", "L-musique")):
            stop()                                                         # the card fit's record missing
        with mutated(w.card_record("zrckp", "L-hotpotqa"), lambda r: r.update(seed=1)):
            stop()                                                         # a card record that is not seed 0
        with mutated(fit(2, "L-hotpotqa"), lambda r: r.update(rmatch_records={"other": "2"})):
            _, _, bes = seed_fits("zrckp", zr, w)                          # a build record: beside, not refused
            assert bes == {"L-hotpotqa seed 2": ["rmatch_records"]}, bes
        with mutated(w.fit_dir(n2_fit("zspk", 1, "L-musique")) / "screen.json",
                     lambda r: r.update(zp1c3p_sha256="0" * 64)):
            load_null_n2("zspk", zs, w)                                    # zspk's card holds no zp1c3p_sha256
        print(f"zn2 selftest (b): a toy N2 null loads (zrckp with its n2 stamp, zspk without); {len(refused)} refusals: "
              "a file missing or repeated, seeds 0 and 3, another arm, split, name or base, a step-1 or second base, the "
              "n2 or P1c3p stamp, a code sha or setting not the card's, another candidate, carve or training set, five "
              "datasets, the seeds' seed 0 apart, the card record missing or not seed 0; build records reported beside",
              flush=True)

        # (d) a toy end to end over all 56 arms
        w = Where(T / "root")
        for b in BASES:
            for sp in SPLITS:
                p = SR.fake_compare(w.screen, card_fit(b, sp), sp, {ds: 0.1 for ds in LG.EVAL_ORDER}, arm=b)
                dst = w.p(f"outputs/l1/base-{b}-{sp}.json")
                dst.parent.mkdir(parents=True, exist_ok=True)
                p.replace(dst)
                toy_card(w, b, sp)
        old = SR.fake_null(w.screen, "scr-null", {("L-musique", "musique"): (0.05, -0.04),
                                                   ("L-hotpotqa", "webqsp"): (0.010, 0.012)})
        assert [w.rel(x) for x in old] == list(OLD_NULL)
        quiet(SR.floors, [str(x) for x in old], w.p(OLD_FLOORS).with_suffix(""))
        toy_null(w, "zrckp", {("L-hotpotqa", "webqsp"): (0.02, -0.02)})            # fresh floor 0.04
        toy_null(w, "zspk", {("L-hotpotqa", "musique"): (0.006, -0.006)})          # fresh floor 0.012
        scen = {("A5c", "zrckpc"): {"L-musique": {"webqsp": 0.03}},               # GAIN then and now: PASS
                ("A5c", "zspkc"): {"L-hotpotqa": {"musique": 0.010}},             # 0.0075 then, 0.012 now: NO_PASS
                ("A3v", "zrckpy"): {"L-musique": {"webqsp": 0.02}},               # PASS
                ("A7", "zrckps"): {"L-hotpotqa": {"webqsp": 0.03}},               # 0.0221 then, 0.04 now: NO_PASS
                ("A7", "zspks"): {"L-musique": {"webqsp": 0.02}},                 # PASS
                ("A2", "zrckpa"): {"L-musique": {"musique": 0.05}},               # 0.0906 then, 0.0075 now: restored
                ("A4", "zspkd"): {"L-musique": {"2wiki": -0.02}}}                 # a LOSS then and now
        for k in (("G3", "zspr"), ("G4", "zspk2"), ("G4", "zspkg"), ("G5", "zspx"), ("R1", "zrckpr")):
            scen[k] = {"L-musique": {"hotpotqa": 0.02}}
        rels = {b: {sp: str(w.p(f"outputs/l1/base-{b}-{sp}.json")) for sp in SPLITS} for b in BASES}
        filed = {}
        for st, arm in ARMS:
            toy_pair(w, st, arm, scen.get((st, arm), {}))
            filed[(st, arm)] = toy_filed(w, st, arm, rels[base_of(arm)])
        toy_verdicts(w, filed)
        assert {k for k, r in filed.items() if r["verdict"] == "PROMISING"} == set(ADOPTED)
        toy_pair(w, "A99", "zrckpz", {"L-musique": {"webqsp": 0.02}})          # filed after N2: listed only
        filed_late = {("A99", "zrckpz"): toy_filed(w, "A99", "zrckpz", rels["zrckp"])}
        toy_verdicts(w, filed_late, "2026-10-20T09:00:00Z")
        toy_pair(w, "A98", "zrckpz", {})                                        # filed before 15 October: not listed
        toy_verdicts(w, {("A98", "zrckpz"): toy_filed(w, "A98", "zrckpz", rels["zrckp"])}, "2026-10-10T00:00:00Z")
        toy_pair(w, "P1gb", "zrckp", {})                                        # P1g-b's pool arm: not listed
        toy_verdicts(w, {("P1gb", "zrckp"): toy_filed(w, "P1gb", "zrckp", rels["zrckp"])})
        frozen = {p: p.read_bytes() for p in list(w.screen.glob("scr-*-pair-recall.json"))
                  + list((w.root / "outputs").glob("*/screen-verdict.json"))}
        assert quiet(gate, w, "a5c") == 2                                       # no summary yet
        assert quiet(floors, w, "zrckp") == 0 and quiet(floors, w, "zspk") == 0
        fz = load(w.floors("zrckp").with_suffix(".json"))
        assert fz["status"] == "OK" and fz["raised"] == ["L-hotpotqa webqsp"] and fz["raised_7oct"] == [
            "L-musique musique", "L-hotpotqa webqsp"]
        r_ = {(r["split"], r["dataset"]): r for r in fz["reads"]}
        assert r_[("L-musique", "musique")]["change"] == "down" and r_[("L-hotpotqa", "webqsp")]["change"] == "up"
        assert r_[("L-hotpotqa", "webqsp")]["floor"] == 0.04 and r_[("L-musique", "musique")]["seed0"] == 0.6
        assert quiet(repro, w) == 0
        rp = load(w.out / "repro.json")
        assert rp["status"] == "REPRODUCED" and rp["counts"]["REPRODUCED"] == 56, rp["counts"]
        assert quiet(recall_all, w) == 0
        ra = load(w.out / "recall-all.json")
        assert all(x["status"] == "OK" for x in ra["arms"]) and not ra["problems"]
        rc = lambda st, arm: load(w.recall_out(st, arm).with_suffix(".json"))   # noqa: E731
        want = {("A5c", "zrckpc"): True, ("A5c", "zspkc"): False, ("A3v", "zrckpy"): True, ("A7", "zrckps"): False,
                ("A7", "zspks"): True, ("G3", "zspr"): True, ("G4", "zspk2"): True, ("G4", "zspkg"): True,
                ("G5", "zspx"): True, ("R1", "zrckpr"): True}
        assert {k: rc(*k)["pass_fresh"] for k in want} == want
        a2 = rc("A2", "zrckpa")
        assert a2["verdict_7oct"] == "NO_GAIN" and a2["verdict"] == "PROMISING" and a2["stage_verdict"] == "NOT_ADOPTED"
        assert a2["changed_from_7oct"] == ["L-musique musique WITHIN -> GAIN"]
        a4 = rc("A4", "zspkd")
        assert a4["verdict"] == "NO_GAIN" and [tag(r) for r in a4["rows"] if r["call"] == "LOSS"] == ["L-musique 2wiki"]
        assert rc("A7", "zrckps")["changed_from_7oct"] == ["L-hotpotqa webqsp GAIN -> WITHIN"]
        assert quiet(summary, w) == 0
        sm = load(w.out / "summary.json")
        assert sm["status"] == "COMPLETE" and not sm["problems"], sm["problems"]
        assert [(a["screen_stage"], a["arm"]) for a in sm["promoted"]] == [("A2", "zrckpa")]
        assert sm["promoted"][0]["next_form"].startswith("a refit on P1g-b's pools") and \
            "docs/A2P_P1GB_POOLS.md" in sm["promoted"][0]["next_form"]
        assert [(a["screen_stage"], a["arm"]) for a in sm["late_arms"]] == [("A99", "zrckpz")]
        assert {(a["screen_stage"], a["arm"]): a["pass_fresh"] for a in sm["adopted"]} == want
        assert sm["totals"]["zspk"]["fresh_loss"] == 1 and sm["totals"]["zrckp"]["oct7_gain"] == 4
        assert any("a caveat" not in x and x.startswith("G5 zspx: PASS_FRESH") for x in sm["routing"])
        expect = {("a5c", None): 1, ("a5c", "zrckpc"): 0, ("a5c", "zspkc"): 1, ("a3v", None): 0, ("A7", None): 1,
                  ("a7", "zspks"): 0, ("a7", "zrckps"): 1, ("g5", None): 2, ("a3v", "zspky"): 2, ("a5cp", None): 2}
        assert {k: quiet(gate, w, *k) for k in expect} == expect
        # INCOMPLETE: the gates exit 2 and nothing is routed; each piece put back gives COMPLETE again
        fp = w.filed("A5c", "zrckpc")
        with mutated(fp, lambda r: r["rows"][0].update(call="LOSS")):          # a filed re-call off its verdict
            quiet(recall_all, w)
            quiet(summary, w)
            assert load(w.out / "summary.json")["status"] == "INCOMPLETE" and quiet(gate, w, "a3v") == 2
            assert not load(w.out / "summary.json")["routing"]
        quiet(recall_all, w)
        quiet(summary, w)
        assert load(w.out / "summary.json")["status"] == "COMPLETE" and quiet(gate, w, "a3v") == 0
        with moved_away(w.floors("zrckp").with_suffix(".json")):               # a floors record missing
            quiet(summary, w)
            assert load(w.out / "summary.json")["status"] == "INCOMPLETE" and quiet(gate, w, "a3v") == 2
        with mutated(w.recall_out("A3v", "zrckpy").with_suffix(".json"), lambda r: r.update(pass_fresh=False)):
            quiet(summary, w)                                                  # a call file not recall-all's
            assert load(w.out / "summary.json")["status"] == "INCOMPLETE"
        quiet(summary, w)
        with moved_away(w.out / "summary.json"):                               # no summary
            assert quiet(gate, w, "a3v") == 2
        with mutated(w.out / "summary.json", lambda r: r.update(declared_in="docs/OTHER.md")):
            assert quiet(gate, w, "a3v") == 2
        with mutated(w.out / "summary.json", lambda r: r.update(adopted=[a for a in r["adopted"]
                                                                         if a["arm"] != "zspkc"])):
            assert quiet(gate, w, "a5c", "zrckpc") == 2                        # the stage's arms not all there
        with mutated(w.filed("A4", "zspkd"), lambda r: r["rows"][1].update(floor=0.0123)):
            quiet(repro, w)                                                    # a filed floor N2's path does not give
            rp = load(w.out / "repro.json")
            assert rp["status"] == "NOT_REPRODUCED" and rp["counts"]["MISMATCH"] == 1
            quiet(summary, w)
            assert load(w.out / "summary.json")["status"] == "INCOMPLETE" and quiet(gate, w, "a3v") == 2
        quiet(repro, w)
        quiet(summary, w)
        assert load(w.out / "summary.json")["status"] == "COMPLETE" and quiet(gate, w, "a3v") == 0
        with mutated(w.p("outputs/l1/base-zrckp-L-musique.json"), lambda r: r.update(new="outputs\\screen\\fits\\x")):
            assert repro_arm(w, "A3v", "zrckpy")["status"] == "FAILED"         # a card comparison not the card's
        with mutated(w.pair("A3v", "zrckpy"), lambda r: [x.update(base=0.61) for x in r["rows"]]):
            must_stop(recall_arm, w, "A3v", "zrckpy")                          # a pair against another seed 0
        with mutated(w.verdict("A3v"), lambda r: r["arms"]["zrckpy"].update(verdict="NOT_ADOPTED")):
            must_stop(recall_arm, w, "A3v", "zrckpy")                          # a verdict N2 did not declare
        with moved_away(w.nulls("zspk")[0]):
            assert quiet(floors, w, "zspk") == 2
            assert load(w.floors("zspk").with_suffix(".json"))["status"] == "REFUSED"
            quiet(recall_all, w)
            assert all(x["status"] == "NOT_RUN" for x in load(w.out / "recall-all.json")["arms"]
                       if x["base_arm"] == "zspk")
        assert all(p.read_bytes() == raw for p, raw in frozen.items()), "a filed toy record changed"
        txt = "".join(p.read_text(encoding="utf-8") for p in w.out.glob("*.*"))
        for form in {str(T), T.as_posix(), str(T).replace("\\", "\\\\"), str(T.resolve()),
                     str(T.resolve()).replace("\\", "\\\\")}:
            assert form not in txt, "an absolute path in N2's records"
        print("zn2 selftest (d): a toy over all 56 arms: floors, the reproduction check (56 REPRODUCED through the "
              "stages' own path), the fresh calls and the summary; PASS_FRESH for A5c zrckpc, A3v, A7 zspks, G3, G4, "
              "G5, R1 and NO_PASS_FRESH for A5c zspkc, A7 zrckps; A2 zrckpa's GAIN restored by a lower floor (promoted); "
              "a late arm listed; gates a5c 1, a5c zrckpc 0, a3v 0, a7 1, a7 zspks 0, g5 2; INCOMPLETE (a filed call, a "
              "floors record, a call file, a repro mismatch, no summary) gives 2; filed records unchanged byte for byte",
              flush=True)

        # (e) train and compare: refusals, the fit name swapped and restored, the n2 stamp
        w = Where(T / "root")
        seen = []

        def stub(argv, split, rc=0, write=True):
            seen.append((list(argv), split, dict(Z3P.FIT)))
            name = argv[argv.index("--name") + 1]
            if write:
                rec = {**json.loads(json.dumps(load(w.card_record("zrckp", split)))), "name": name,
                       "seed": int(argv[argv.index("--seed") + 1])}
                LC.write_json(w.fit_dir(name) / "screen.json", rec)
            return rc

        base_args = ["--split", "L-musique", "--name", "scr-n2-zrckp-s1", "--arm", "zrckp", "--seed", "1",
                     "--device", "cuda", "--host", "--cache-root", "outputs/p1c2/cache"]

        def args(**kw):
            a = list(base_args)
            for k, v in kw.items():
                a[a.index("--" + k) + 1] = v
            return a

        fit0 = dict(Z3P.FIT)
        for bad in (args(arm="zspk"), args(arm="zrckpc"), args(seed="0"), args(seed="3"),
                    args(name="scr-n2-zrckp-s2"), args(name="scr-p1c3p-zrckp"), args(split="L-2wiki"),
                    [x for x in base_args if x not in ("--cache-root", "outputs/p1c2/cache")]):
            must_stop(train, bad, w, stub)
        with mutated(w.card_record("zrckp", "L-musique"), lambda r: r.update(lean_gpu_sha256="0" * 64)):
            must_stop(train, base_args, w, stub)                               # the code is not the card's
        with moved_away(w.card_record("zrckp", "L-musique")):
            must_stop(train, base_args, w, stub)
        assert not seen and Z3P.FIT == fit0
        assert train(base_args, w, stub) == 0 and Z3P.FIT == fit0
        argv, split, fit_then = seen[-1]
        assert argv == ["train", "--name", "scr-n2-zrckp-s1", "--arm", "zrckp", "--seed", "1", "--device", "cuda",
                        "--cache-root", "outputs/p1c2/cache", "--host"] and split == "L-musique"
        assert fit_then == {**fit0, "L-musique": "scr-n2-zrckp-s1"}
        s1 = load(w.fit_dir("scr-n2-zrckp-s1") / "screen.json")
        assert s1["n2"] == {"declared_in": DECLARED, "base_arm": "zrckp", "seed": 1, "split": "L-musique",
                            "card_fit": "scr-p1c3p-zrckp"} and s1["zn2_sha256"] == LC.sha_src(__file__)
        hp = args(split="L-hotpotqa", name="scr-n2-zrckp-s2-hp", seed="2")
        assert train(hp, w, lambda a, s: stub(a, s, rc=1)) == 1 and Z3P.FIT == fit0
        assert "n2" not in load(w.fit_dir("scr-n2-zrckp-s2-hp") / "screen.json")   # a failed fit is not stamped

        def boom(a, s):
            stub(a, s, write=False)
            raise SystemExit("toy")

        must_stop(train, hp, w, boom)
        assert Z3P.FIT == fit0
        with moved_away(w.fit_dir("scr-n2-zrckp-s1") / "screen.json"):        # rc 0 but no record: refused
            must_stop(train, base_args, w, lambda a, s: stub(a, s, write=False))
        assert Z3P.FIT == fit0
        calls = []
        run = lambda a: calls.append(list(a)) or 0                              # noqa: E731
        good = ["--new", "outputs/screen/fits/scr-n2-zrckp-s1", "--base", "outputs/screen/fits/scr-p1c3p-zrckp",
                "--out", "outputs/screen/scr-n2-zrckp-s1"]
        assert compare(good, w, run) == 0 and calls[-1] == ["compare"] + good
        for i, v in ((1, "outputs/screen/fits/scr-a5c-zrckpc"), (1, str(w.fit_dir("scr-n2-zrckp-s1"))),
                     (1, "outputs/screen/fits/scr-n2-zrckp-s3"), (3, "outputs/screen/fits/scr-p1c3-zspk"),
                     (3, "outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique"),
                     (3, "outputs/screen/fits/scr-p1c3p-zrckp-hp"), (5, "outputs/screen/scr-n2-zrckp-s2"),
                     (5, "outputs/n2/scr-n2-zrckp-s1")):
            bad = list(good)
            bad[i] = v
            must_stop(compare, bad, w, run)
        for fn in (lambda r: r.update(seed=2), lambda r: r.update(arm="zspk"), lambda r: r.pop("n2"),
                   lambda r: r.pop("p1c3p")):
            with mutated(w.fit_dir("scr-n2-zrckp-s1") / "screen.json", fn):
                must_stop(compare, good, w, run)
        assert len(calls) == 1
        zsg = ["--new", "outputs/screen/fits/scr-n2-zspk-s2-hp", "--base", "outputs/screen/fits/scr-p1c3-zspk-hp",
               "--out", "outputs/screen/scr-n2-zspk-s2-hp"]
        assert compare(zsg, w, run) == 0 and calls[-1] == ["compare"] + zsg
        print("zn2 selftest (e): train refuses another arm, seeds 0 and 3, another name or split, a code sha not the "
              "card's and a missing card record; it runs zp1c3p's train with N2's name swapped in and restored (also "
              "on a failure), stamps n2 only after a fit; compare refuses any other new, base or out and a fit record "
              "not N2's, and runs lean_screen's compare with exactly its arguments", flush=True)

    # (g) the filed re-calls whose pairs are here, called again through N2's path with the OLD null
    w0 = Where()
    done = []
    with null_fit_records():
        for st, arm in ARMS:
            if w0.pair(st, arm).exists() and w0.filed(st, arm).exists():
                r = repro_arm(w0, st, arm)
                assert r["status"] == "REPRODUCED", (st, arm, r)
                done.append(f"{st} {arm}")
    assert {base_of(x.split()[1]) for x in done} == set(BASES), done
    print(f"zn2 selftest (g): {len(done)} filed re-calls whose pairs are on this machine ({', '.join(done)}) "
          "reproduce exactly through N2's path with the 7 Oct null (both bases)", flush=True)

    # (g') every filed re-call here, from a pair rebuilt out of its rows (the pair's numbers and calls)
    n, subs, absent = 0, [], []
    with tempfile.TemporaryDirectory() as td2, null_fit_records():
        T2 = Path(td2)
        for st, arm in ARMS:
            fp = w0.filed(st, arm)
            if not fp.exists():
                absent.append(f"{st} {arm}")
                continue
            filed = load(fp)
            b = base_of(arm)
            assert filed.get("base_arm") == b and filed.get("stage") == st and filed.get("arm") == arm, (st, arm)
            v = filed_of(load(w0.verdict(st)), st, arm)                 # what recall_arm checks before it calls
            assert v["calls"] == {tag(r): r["call"] for r in filed["rows"]}, (st, arm, "calls")
            assert (v["verdict"] == "ADOPT_FOR_FULL_RUN") == ((st, arm) in ADOPTED), (st, arm, v["verdict"])
            assert repo_rel(filed.get("pair")) == w0.rel(w0.pair(st, arm)), (st, arm, "pair")
            pp = T2 / f"scr-{st.lower()}-{arm}-pair.json"
            pp.write_text(json.dumps({"verdict": filed["verdict_filed"],
                                      "rows": [{**{k: x for k, x in r.items() if k not in RECALL_ADDED},
                                                "call": r["call_filed"]} for r in filed["rows"]]}), encoding="utf-8")
            rel = {sp: repo_rel((filed.get("null") or {}).get(f"rel {sp}")) for sp in SPLITS}
            chk = all(r and w0.p(r).exists() for r in rel.values())
            if not chk:                      # G2's card comparisons are on the host: l1's of the same card stand in
                rel = {sp: f"outputs/l1/base-{b}-{sp}.json" for sp in SPLITS}
                subs.append(f"{st} {arm}")
            bad = repro_calls(w0, b, arm, filed, pp, {sp: str(w0.p(x)) for sp, x in rel.items()}, w0.old_null(),
                              check_sources=chk)
            assert not bad, (st, arm, bad)
            n += 1
    assert n >= 50 and subs == ["G2 zg2"], (n, subs)
    print(f"zn2 selftest (g'): {n} of 56 filed re-calls on this machine reproduce exactly (calls, floors, deltas, "
          f"verdicts) from their pair rows through N2's path with the 7 Oct null; {', '.join(subs)} with l1's "
          f"comparisons of the same card fit standing in; not here: {', '.join(absent) or 'none'}", flush=True)

    # (f) nothing written under the stage roots
    snap1 = snapshot()
    assert snap1 == snap0, ("a file under the stage roots changed: "
                            + ", ".join(sorted(k for k in set(snap0) | set(snap1) if snap0.get(k) != snap1.get(k))[:10]))
    assert not (ROOT / "outputs" / "n2").exists()
    print(f"zn2 selftest (f): nothing written under outputs/screen, l1, p1c2, p1c3, p1c3p or the 30 stage folders; "
          f"no outputs/n2 ({time.time() - t0:.0f} s)", flush=True)
    print("zn2 selftest: ok", flush=True)
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--selftest"]:
        return selftest()
    if not argv:
        raise SystemExit("zn2: train, compare, floors, repro, recall-all, summary, gate, or --selftest")
    cmd, rest = argv[0], argv[1:]
    if cmd == "train":
        return train(rest)
    if cmd == "compare":
        return compare(rest)
    w = Where()
    if cmd == "floors":
        ap = argparse.ArgumentParser(prog="zn2.py floors")
        ap.add_argument("--base", required=True, choices=BASES)
        return floors(w, ap.parse_args(rest).base)
    if cmd in ("repro", "recall-all", "summary"):
        argparse.ArgumentParser(prog=f"zn2.py {cmd}").parse_args(rest)
        return {"repro": repro, "recall-all": recall_all, "summary": summary}[cmd](w)
    if cmd == "gate":
        ap = argparse.ArgumentParser(prog="zn2.py gate")
        ap.add_argument("--stage", required=True)
        ap.add_argument("--arm")
        a = ap.parse_args(rest)
        return gate(w, a.stage, a.arm)
    raise SystemExit(f"zn2: {cmd} is not a command (train, compare, floors, repro, recall-all, summary, gate)")


if __name__ == "__main__":
    sys.exit(main())
