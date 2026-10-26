"""GL's part 2 (docs/GL_GPU_LINE_ON_L40S.md, sections 3, 4 and 6): R2's two arms and G6's twin on U2d's base build, on
Modal's L40S from U2d's image, on workspace 3 from 1 November.

The six fits, seed 0, U2d's card settings, every carve on U2d's base build (zu2d.py's roots), stamped with their device:
    scr-gl-zrckpb, -hp    zrckp + R2's bridge head (zr2br.py's arm, unchanged), trained with R2's EM term
    scr-gl-zg6m, -hp      zg6 + R2's bridge head on its final score (zr2br.py's arm, unchanged), with R2's EM term
    scr-gl-zg6t, -hp      G6's twin: zg6 with no edges reaching its layers (zg6.py's arm, unchanged)
Each fit is read with p@swa on the six s1eval carves of the base build, at its arm's rows cap, never the caller's:
zg6m's and zg6t's at 65,536 (R2's and G6's reads), zrckpb's at lean_gpu's 262,144 (R2's zrckpb reads and U2d's zrckp
reads). Batches hold whole questions, so the cap does not change a score.

The compares (lean_screen's), every fit of one on one device:
    scr-gl-zrckpb[-hp]        against scr-u2d-base-zrckp[-hp]    R2: decides zrckpb
    scr-gl-zg6m[-hp]          against scr-u2d-base-zg6[-hp]      R2: decides zg6m
    scr-gl-zg6t[-hp]          against scr-u2d-base-zspk[-hp]     G6's beside row
    scr-gl-zg6-vs-zg6t[-hp]   scr-u2d-base-zg6 against scr-gl-zg6t: what the edges add (beside)
The calls: screen_pair's pair of each arm's two compares (GL's loader), then screen_recall's re-call under U2d's null:
zrckpb under U2d's zrckp null; zg6m under U2d's zspk null moved onto zg6 through scr-u2d-base-zg6-vs-zspk[-hp]
(zu2d.rebase: D1, D2 and the floors stay zspk's); zg6t under U2d's zspk null (beside). R2's rule decides each R2 arm.
    out  outputs/screen/scr-gl-<arm>-pair, -pair-recall; outputs/gl/r2-verdict; outputs/gl/zg6t-beside

On Modal (zu2d_modal.py --stage gl-fits runs these in the container's process):
    python outputs/mp_unified/zgl2.py train --split L-musique --name scr-gl-zg6m --arm zg6m --device cuda \\
        --cache-root outputs/u2d/base/cache --modal
    python outputs/mp_unified/zgl2.py read --name scr-gl-zg6m --device cuda --cache-root outputs/u2d/base/cache --modal
On the laptop, from the fetched reads:
    python outputs/mp_unified/zgl2.py compare-all
    python outputs/mp_unified/zgl2.py calls        # the three pairs and their re-calls
    python outputs/mp_unified/zgl2.py verdict      # R2's verdict on the L40S and the twin's rows
    python outputs/mp_unified/zgl2.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zr2br as R2  # noqa: E402  (R2's arms, constants, EM term and next forms; it sets the BLAS threads before numpy)
import zu2d as Z  # noqa: E402  (U2d's roots, card, device check, null loader, re-base and fakes)

G6, SP = Z.G6, Z.SP
G2 = G6.G2
Z2, Z3, RM, ZL = Z.Z2, Z.Z3, Z.RM, Z.ZL
S, S2, LG, LC, SR, R = Z.S, Z.S2, Z.LG, Z.LC, Z.SR, Z.R
log = Z.log

DECLARED = "docs/GL_GPU_LINE_ON_L40S.md"
STAGE = "GL"
KEY = "gl"                                     # the stamp's key in screen.json, read.json and the compares
BUILD = "base"
SCREEN = Path("outputs") / "screen"            # relative: every laptop command runs from the repository's root
OUT = Path("outputs") / "gl"
SPLITS, SUF = Z.SPLITS, Z.SUF
R2_ARMS = ("zrckpb", "zg6m")
TWIN = "zg6t"
GL_ARMS = R2_ARMS + (TWIN,)
ARMS = {"zrckpb": R2.ARMS["zrckpb"], "zg6m": R2.ARMS["zg6m"], TWIN: G6.ARMS[TWIN]}
BASE_OF = {"zrckpb": "zrckp", "zg6m": "zg6", TWIN: "zspk"}       # each compare's base: U2d's seed-0 fit of this arm
NULL_ARM = {"zrckpb": "zrckp", "zg6m": "zspk", TWIN: "zspk"}
ROWS_CAP = {"zrckpb": LG.READ_ROWS, "zg6m": 65536, TWIN: 65536}
WHAT = {"zrckpb": "zrckp + R2's bridge head", "zg6m": "zg6 + R2's bridge head on its final score",
        TWIN: "G6's twin: zg6 with no edges reaching its layers"}
NULL_KIND = {"zrckpb": "U2d-zrckp", "zg6m": "U2d-zspk-on-zg6", TWIN: "U2d-zspk"}
NULL_IS = {"zrckpb": "U2d's zrckp null on the L40S: zrckp's base-build fit trained again with seeds 1 and 2, each "
                     "against its seed-0 fit (N2's design, on the device the arms ran on)",
           "zg6m": "U2d's zspk null on the L40S, each read's seed-0 R@5 moved to zg6's through U2d's compares of zg6 "
                   "with zspk on the base build (zu2d.py's re-base; D1, D2 and the floors stay zspk's)",
           TWIN: "U2d's zspk null on the L40S: zspk's base-build fit trained again with seeds 1 and 2, each against "
                 "its seed-0 fit"}
RULE = ("per arm (zrckpb, zg6m): ADOPT_FOR_FULL_RUN when its re-call under its U2d null on the L40S has no LOSS on any "
        "of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when that re-call is missing, is "
        "not GL's re-call of the arm under that null, does not hold the twelve reads, or U2d's null files fail "
        f"zu2d.py's null checks (docs/R2_BRIDGE_EM.md, section 4, with the files of {DECLARED}, section 3)")
NOT_BESIDE = {"oct7": "the 7 October re-calls are not computed: their re-base reads the host's fits, another device "
                      "(section 3)",
              "step1": "step 1's fit is not set beside either arm: it is from another device (section 3)",
              "zrckpb": "zrckpb is called \"zrckp + R2's bridge head\" until the user rules whether reading the five "
                        "bridges' vectors keeps it inside the MLP track (R2, section 2)"}
PAIR_FIX = "R@5 of the screen's p@swa minus step 1's p@swa of the same split"
RECALL_FIX = (("twice the seed-only spread of seeds 1 and 2 against seed 0",
               "twice the seed-only spread of {n}'s seeds 1 and 2 against its seed-0 fit on U2d's base build"),
              ("D1, D2: the null's seed-only differences on the read",
               "D1, D2: {n}'s seed-only differences on the read (U2d's {n} null)"))
TWIN_VS = "scr-gl-zg6-vs-zg6t"

assert {a: BASE_OF[a] for a in R2_ARMS} == R2.BASE_OF and tuple(SPLITS) == tuple(R2.SPLITS)
assert R2.LG is LG and R2.RM is RM and R2.Z2 is Z2 and R2.S is S
for _a in GL_ARMS:
    if S.ARMS.get(_a) != ARMS[_a]:
        raise SystemExit(f"zgl2: lean_screen's arm {_a} is {S.ARMS.get(_a)}, not {ARMS[_a]}")


def rel(p):
    p = Path(p)
    return p.relative_to(ROOT).as_posix() if p.is_absolute() and p.is_relative_to(ROOT) else p.as_posix()


def fit_name(arm, split):
    return f"scr-gl-{arm}{SUF[split]}"


def base_fit(arm, split):
    return Z.fit_name(BUILD, BASE_OF[arm], 0, split)


FITS = {fit_name(a, sp): (a, sp) for a in GL_ARMS for sp in SPLITS}
assert len(FITS) == 6


def compares():
    """The eight compares: each arm against its base's L40S fit alone (R2's two decide, the twin's is beside), and zg6
    against the twin (beside)."""
    out = {}
    for sp in SPLITS:
        for a in GL_ARMS:
            out[fit_name(a, sp)] = (fit_name(a, sp), [base_fit(a, sp)])
        out[f"{TWIN_VS}{SUF[sp]}"] = (Z.fit_name(BUILD, "zg6", 0, sp), [fit_name(TWIN, sp)])
    return out


COMPARES = compares()
assert len(COMPARES) == 8
AGAINST = {a: f"{BASE_OF[a]}'s seed-0 fit on U2d's base build ({base_fit(a, SPLITS[0])}, {base_fit(a, SPLITS[1])}) on "
              f"Modal's L40S: the same pools, KT input, card settings and device" for a in GL_ARMS}


# ── train and read (sections 3 and 4) ────────────────────────────────────────


def head_of(arm):
    if arm == TWIN:
        return {**Z.head_of("zg6", BUILD), "twin": ARMS[TWIN][0].TWIN}
    return {"base": Z.head_of(BASE_OF[arm], BUILD), "r2": R2.SETTINGS, "r2_constants": R2.constants(),
            "r2_params": list(R2.R2_PARAMS)}


def code_shas():
    return {**Z.code_shas(), "zgl2_sha256": LC.sha_src(__file__), "zr2br_sha256": LC.sha_src(R2.__file__)}


def where_ok(where, env, what):
    if not Z.ok_where(where, env):
        raise SystemExit(f"zgl2: GL {what}s on Modal's {Z.GPU} from the image {Z.IMAGE_TAG} (section 3), not on "
                         f"{where} with {Z.device_of(env)}")


def train(argv, split, where="laptop", env=None):
    """One of GL's six fits: its arm and split from its name, seed 0, U2d's card on the GPU, every carve on U2d's base
    build; R2's arms with R2's EM term in training (zr2br.r2_em, as zr2br.py trains them), the twin as zg6.py trains
    it; the base gate PASS; stamped with the device it ran on and, for R2's arms, the term per epoch."""
    arm = R.arm_of(argv)
    name, _ = S2.where(argv)
    if name not in FITS:
        raise SystemExit(f"zgl2: {name} is not one of GL's fits (scr-gl-<arm>[-hp], sections 3 and 4)")
    farm, fsplit = FITS[name]
    if arm != farm or split != fsplit:
        raise SystemExit(f"zgl2: {name} is {farm}'s fit of {fsplit}, not {arm}'s of {split}")
    if S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zgl2: the arm {arm} is {S.ARMS.get(arm)}, not GL's {ARMS[arm]}")
    bad = G6.fixed_flags(argv, Z.REFUSED_TRAIN)
    if bad:
        raise SystemExit(f"zgl2: GL trains on U2d's card (seed 0, hidden {LG.HIDDEN}, config {LG.CONFIG}, variant p, "
                         f"the split's fit carves and basis, threads 2, outputs/screen/fits); {bad} would change that")
    if arm in R2_ARMS:
        R2.check_constants()
    env = Z.device_env() if env is None else env
    where_ok(where, env, "train")
    summ = None
    with Z.on_u2d(BUILD) as r:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv) + ["--seed", "0"]
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.train_carves(split), r["kt"])
        if arm in R2_ARMS:
            with R2.r2_em():
                rc = RM.train(argv, split)
            summ = R2.summary()
            if summ["questions_with_term"] == 0:
                raise SystemExit(f"zgl2: no training batch of {name} carried R2's term ({summ['batches']} batches)")
        else:
            rc = RM.train(argv, split)
        stamp_fit(argv, name, env, g, kt, summ)
    if summ:
        ep = summ["epochs"]
        log(f"zgl2 {name}: R2's term on {summ['questions_with_term']} of {summ['questions']} training questions "
            f"({summ['share_with_term']:.4f}); L_br {ep[0]['br_nll_mean']:.4f} (epoch 0) to "
            f"{ep[-1]['br_nll_mean']:.4f} (last)")
    return rc


def stamp_fit(argv, name, env, g, kt, summ):
    """The fit's screen.json stamped with GL's record; called inside on_u2d (the chains' and links' records)."""
    arm, split = FITS[name]
    _, out_root = S2.where(argv)
    sj = Path(out_root) / name / "screen.json"
    if not sj.exists():
        return
    rec = json.loads(sj.read_text(encoding="utf-8"))
    r = Z.roots(BUILD)
    u = {"declared_in": DECLARED, "stage": STAGE, "fit": name, "build": BUILD, "arm": arm, "what": WHAT[arm],
         "base_arm": BASE_OF[arm], "seed": 0, "split": split, "device": Z.device_of(env), "env": env,
         "graph": Z.GRAPH[BUILD], "pools": g.get("construction"), "rule": Z.ZP.RULE[BUILD],
         "cache_root": Z.rel(r["cache"]), "gate": LC.sha_file(r["gate"]), "head": head_of(arm), "kt_records": kt,
         "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(r["ch_zrc"]),
         "zlink_records": ZL.built_records(), "rows_cap": ROWS_CAP[arm]}
    if arm in R2_ARMS:
        u["r2"] = {"declared_in": R2.DECLARED, "constants": R2.constants(), "params": list(R2.R2_PARAMS),
                   "term": summ}
    rec.update({**code_shas(), KEY: u})
    LC.write_json(sj, rec)


def fit_stamp(name, out_root=None):
    """A GL fit's screen.json and its gl record, checked against the fit's name."""
    arm, split = FITS[name]
    sj = Path(out_root or S.OUT / "fits") / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    u = rec.get(KEY) or {}
    if (rec.get("arm"), u.get("declared_in"), u.get("fit"), u.get("build"), u.get("arm"), u.get("seed"),
            u.get("split")) != (arm, DECLARED, name, BUILD, arm, 0, split):
        raise SystemExit(f"zgl2: {sj} is not GL's {arm} fit of {split}")
    return rec, u


def read(argv, where="laptop", env=None):
    """The fit read with p@swa on the six s1eval carves of the base build at its arm's rows cap, on the device it was
    fitted on; R2's arms only with R2's declared constants."""
    name, out_root = S2.where(argv)
    if name not in FITS:
        raise SystemExit(f"zgl2: {name} is not one of GL's fits")
    arm, split = FITS[name]
    bad = G6.fixed_flags(argv, Z.REFUSED_READ + ("--rows-cap",))
    if bad:
        raise SystemExit(f"zgl2: GL reads p's fits on the six s1eval carves of the base build, {arm}'s at --rows-cap "
                         f"{ROWS_CAP[arm]} (section 3); {bad} would change that")
    _rec, u = fit_stamp(name, out_root)
    if arm in R2_ARMS:
        R2.check_constants((u.get("r2") or {}).get("constants"), name)
    env = Z.device_env() if env is None else env
    where_ok(where, env, "read")
    if Z.device_of(env) != u.get("device"):
        raise SystemExit(f"zgl2: {name} was fitted on {u.get('device')}; this read runs on {Z.device_of(env)}")
    with Z.on_u2d(BUILD) as r:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv) + ["--rows-cap", str(ROWS_CAP[arm])]
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.eval_carves(), r["kt"])
        rc = RM.read(argv)
    rj = Path(out_root) / name / "read.json"
    if rj.exists():
        rec = json.loads(rj.read_text(encoding="utf-8"))
        if rec.get("rows_cap") != ROWS_CAP[arm]:
            raise SystemExit(f"zgl2: {rj} was read at --rows-cap {rec.get('rows_cap')}, not {arm}'s {ROWS_CAP[arm]}")
        rec.update({**code_shas(), KEY: {
            "declared_in": DECLARED, "stage": STAGE, "fit": name, "build": BUILD, "arm": arm, "seed": 0,
            "split": split, "device": Z.device_of(env), "env": env, "gate": LC.sha_file(Z.roots(BUILD)["gate"]),
            "pools": g.get("construction"), "kt_records": kt, "rows_cap": ROWS_CAP[arm]}})
        LC.write_json(rj, rec)
    return rc


# ── the compares ─────────────────────────────────────────────────────────────


def fit_device(name, out_root=None):
    """A fetched fit's device: a GL fit's from its gl stamps (read on the fit's device, at its arm's rows cap), U2d's
    through zu2d.py."""
    if name not in FITS:
        return Z.fit_device(name, out_root)
    _rec, u = fit_stamp(name, out_root)
    rj = Path(out_root or S.OUT / "fits") / name / "read.json"
    r = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
    ru = r.get(KEY) or {}
    if ru.get("declared_in") != DECLARED or ru.get("fit") != name or ru.get("device") != u.get("device"):
        raise SystemExit(f"zgl2: {rj} is not GL's read of {name} on the device it was fitted on")
    if r.get("rows_cap") != ROWS_CAP[u["arm"]] or ru.get("rows_cap") != ROWS_CAP[u["arm"]]:
        raise SystemExit(f"zgl2: {rj} was read at --rows-cap {r.get('rows_cap')}, not {u['arm']}'s "
                         f"{ROWS_CAP[u['arm']]}")
    return u["device"]


def one_device(names, out_root=None):
    """The device every fit in names ran on; stops when two differ (section 3: one device)."""
    devs = {n: fit_device(n, out_root) for n in names}
    if len({json.dumps(d, sort_keys=True) for d in devs.values()}) != 1:
        raise SystemExit(f"zgl2: the fits ran on different devices or images: {devs}")
    return next(iter(devs.values()))


def compare_cmd(cname, out_root=None, screen=None):
    """lean_screen's compare of the compare's new fit against its base, the fits from one device, the paths relative
    to the repository; the comparison stamped with GL's record."""
    new, bases = COMPARES[cname]
    dev = one_device([new, *bases], out_root)
    fits = Path(out_root or S.OUT / "fits")
    out = Path(screen or SCREEN) / cname
    with Z.at_root():
        rc = S2.main(["compare", "--new", rel(fits / new), "--base", ",".join(rel(fits / b) for b in bases),
                      "--out", rel(out)])
        js = out.with_suffix(".json")
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec[KEY] = {"declared_in": DECLARED, "stage": STAGE, "compare": cname, "new": new, "bases": list(bases),
                    "device": dev, "zgl2_sha256": LC.sha_src(__file__)}
        LC.write_json(js, rec)
    return rc


def compare_all(only=None, out_root=None, screen=None):
    """Every compare not yet filed (a filed one with the same fits and GL's record is kept); a compare whose fits are
    not fetched, or not of one device, is named and skipped. 1 when one is skipped."""
    done, ran, skipped = [], [], []
    for cname, (new, bases) in COMPARES.items():
        if only and cname not in only:
            continue
        js = Path(screen or SCREEN) / f"{cname}.json"
        if js.exists():
            u = json.loads(js.read_text(encoding="utf-8")).get(KEY) or {}
            if u.get("declared_in") == DECLARED and u.get("new") == new and u.get("bases") == list(bases):
                done.append(cname)
                continue
        try:
            compare_cmd(cname, out_root, screen)
            ran.append(cname)
        except SystemExit as e:
            skipped.append(f"{cname}: {e}")
    log(f"zgl2 compare-all: {len(ran)} run, {len(done)} filed before, {len(skipped)} skipped"
        + (f" ({'; '.join(skipped)})" if skipped else ""))
    return 1 if skipped else 0


# ── the pairs and the re-calls ───────────────────────────────────────────────


def call_fits(arm):
    """The fits an arm's call reads: its two, its base's two and its null's six (seed 0, then seeds 1 and 2; the seed-0
    fits are zg6m's re-base's base)."""
    na = NULL_ARM[arm]
    names = [*(fit_name(arm, sp) for sp in SPLITS), *(base_fit(arm, sp) for sp in SPLITS),
             *(Z.fit_name(BUILD, na, s, sp) for sp in SPLITS for s in (0, *SR.SEEDS))]
    return list(dict.fromkeys(names))


def pair_out(arm, screen=None):
    return Path(screen or SCREEN) / f"scr-gl-{arm}-pair"


def recall_out(arm, screen=None):
    return Path(screen or SCREEN) / f"scr-gl-{arm}-pair-recall"


def loader(arm, dev):
    """screen_pair.load for GL's compare of the arm with its base's L40S fit, of either split: (split, rows)."""

    def load(fn):
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zgl2: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        nn_ = SR.parts_of(rec.get("new", ""))[-1:]
        split = next((sp for sp in SPLITS if nn_ == (fit_name(arm, sp),)), None)
        if split is None:
            raise SystemExit(f"zgl2: {p} reads {rec.get('new')}, not GL's {arm} fit")
        cname, base = fit_name(arm, split), base_fit(arm, split)
        u = rec.get(KEY) or {}
        if (u.get("declared_in"), u.get("compare"), u.get("new"), u.get("bases")) != (DECLARED, cname, cname, [base]):
            raise SystemExit(f"zgl2: {p} is not GL's compare {cname} of {cname} with {base}")
        if SR.parts_of((rec.get("bases") or [""])[0])[-1:] != (base,):
            raise SystemExit(f"zgl2: {p} is decided against {(rec.get('bases') or [None])[0]}, not {base}")
        if u.get("device") != dev:
            raise SystemExit(f"zgl2: {p} was made on {u.get('device')}, not on the fits' device {dev}")
        if sorted(rec.get("trained_on", [])) != sorted(LG.FITS[split]):
            raise SystemExit(f"zgl2: {p} trained on {rec.get('trained_on')}, not {split}'s {sorted(LG.FITS[split])}")
        if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
            raise SystemExit(f"zgl2: {p} reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on s1eval")
        rows = rec["by_base"][rec["bases"][0]]["rows"]
        if sorted(r["dataset"] for r in rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"zgl2: {p} reads {[r['dataset'] for r in rows]}, not the six datasets")
        return split, rows

    return load


def restamp(out, arm, kind, dev, through=None):
    """The pair's or re-call's files stamped as GL's call of the arm; the md's wording says what was compared."""
    out = Path(out)
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": STAGE, "declared_in": DECLARED, "decides": "R2" if arm in R2_ARMS else None,
                    "r2_declared_in": R2.DECLARED if arm in R2_ARMS else None, "arm": arm, "what": WHAT[arm],
                    "base_arm": BASE_OF[arm], "against": AGAINST[arm], "device": dev,
                    "compares": [fit_name(arm, sp) for sp in SPLITS], "zgl2_sha256": LC.sha_src(__file__)})
        if kind == "recall":
            rec.update({"null_kind": NULL_KIND[arm], "null_is": NULL_IS[arm], "rebased_through": through or {},
                        "deciding": arm in R2_ARMS})
        LC.write_json(js, rec)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8").replace(
            PAIR_FIX, f"R@5 of {arm}'s p@swa minus {BASE_OF[arm]}'s p@swa of the same split, both on U2d's base build "
                      f"on the L40S")
        if kind == "recall":
            for a, b in RECALL_FIX:
                t = t.replace(a, b.format(n=NULL_ARM[arm]))
        head = f"GL ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]} on U2d's base build"
        if kind == "recall":
            head += f", re-called under {NULL_IS[arm]}"
            if through:
                head += " (re-based through " + ", ".join(Path(x).stem for x in through.values()) + ")"
        md.write_text(head + ".\n\n" + t, encoding="utf-8")


def pair(arm, screen=None, out_root=None):
    """screen_pair.pair over GL's two compares of the arm, through GL's loader, after the device check."""
    dev = one_device(call_fits(arm), out_root)
    scr = Path(screen or SCREEN)
    out = pair_out(arm, scr)
    saved = SP.load
    SP.load = loader(arm, dev)
    try:
        rec = SP.pair([scr / f"{fit_name(arm, sp)}.json" for sp in SPLITS], out)
    finally:
        SP.load = saved
    restamp(out, arm, "pair", dev)
    return rec


def recall(arm, screen=None, out_root=None):
    """screen_recall.pair of GL's pair of the arm under its U2d null, re-based where the pair's base is not the null's
    seed-0 fit (zg6m: through U2d's compares of zg6 with zspk); screen_recall checks each read's base R@5 against the
    null's seed 0."""
    dev = one_device(call_fits(arm), out_root)
    scr = Path(screen or SCREEN)
    na = NULL_ARM[arm]
    files = Z.null_files(na, scr)
    null, sources = Z.u2d_null(na, files)
    through = {}
    for sp in SPLITS:
        pb, s0 = base_fit(arm, sp), Z.fit_name(BUILD, na, 0, sp)
        if pb != s0:
            cf = scr / f"{Z.rebase_name(pb, s0)}.json"
            Z.rebase(null, sources, cf, pb, s0)
            through[sp] = rel(cf)
    pf = pair_out(arm, scr).with_suffix(".json")
    pr = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else {}
    if (pr.get("stage"), pr.get("declared_in"), pr.get("arm"), pr.get("device")) != (STAGE, DECLARED, arm, dev):
        raise SystemExit(f"zgl2: {pf} is not GL's pair of {arm} on the fits' device")
    out = recall_out(arm, scr)
    saved = SR.load_null
    SR.load_null = lambda _files: (null, sources)
    try:
        rec = SR.pair(files, pf, out)
    finally:
        SR.load_null = saved
    restamp(out, arm, "recall", dev, through)
    return rec


def calls(screen=None, out_root=None):
    """Each arm's pair and re-call; an arm whose files are not all there, or not of one device, is named and skipped.
    1 when one is skipped."""
    done, skipped = [], []
    for a in GL_ARMS:
        try:
            pair(a, screen, out_root)
            recall(a, screen, out_root)
            done.append(a)
        except SystemExit as e:
            skipped.append(f"{a}: {e}")
    log(f"zgl2 calls: {', '.join(done) or 'none'} called" + (f"; skipped {'; '.join(skipped)}" if skipped else ""))
    return 1 if skipped else 0


# ── R2's verdict on the L40S (section 3) and the twin's rows (section 4) ─────


def tag(r):
    return f"{r.get('split')} {r.get('dataset')}"


def fmt(x, signed=True):
    if not isinstance(x, (int, float)):
        return "n/a"
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def recall_rows(rp, arm, problems):
    rp = Path(rp)
    if not rp.exists():
        problems.append(f"{rp.name} missing")
        return [], {}
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if (rec.get("stage"), rec.get("declared_in"), rec.get("arm"), rec.get("base_arm"), rec.get("null_kind")) != (
            STAGE, DECLARED, arm, BASE_OF[arm], NULL_KIND[arm]):
        problems.append(f"{rp.name} is not GL's re-call of {arm} under {NULL_KIND[arm]}")
    rows = rec.get("rows", [])
    got = sorted((r.get("split"), r.get("dataset")) for r in rows)
    if got != sorted((sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
        problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six datasets")
    return rows, rec


def null_problem(arm, screen):
    """A problem when U2d's null files of the arm's null fail zu2d.py's null checks."""
    try:
        Z.u2d_null(NULL_ARM[arm], Z.null_files(NULL_ARM[arm], screen))
    except SystemExit as e:
        return f"U2d's {NULL_ARM[arm]} null fails zu2d.py's null checks: {e}"
    return None


def fit_terms(arm, out_root=None):
    """Per split, R2's term from the fit's gl stamp: L_br's mean per epoch and the share of training questions with a
    term; None where the fit or its stamp is missing."""
    out = {}
    for sp in SPLITS:
        sj = Path(out_root or S.OUT / "fits") / fit_name(arm, sp) / "screen.json"
        rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
        t = ((rec.get(KEY) or {}).get("r2") or {}).get("term")
        out[sp] = None if not t else {"br_nll_by_epoch": [round(e["br_nll_mean"], 4) for e in t.get("epochs", [])],
                                      "share_with_term": round(t["share_with_term"], 4),
                                      "questions_with_term": t["questions_with_term"], "questions": t["questions"]}
    return out


def minus_md(rows_by, label):
    md = []
    for k, rs in rows_by.items():
        if rs is None:
            md.append(f"| {k} | (no compare) | | | | | |")
            continue
        for r in rs:
            d, lo, hi = r["delta"], r["lo"], r["hi"]
            md.append(f"| {k} | {r['dataset']} | {r['read']} | {d[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | "
                      f"{d[1]:+.4f} | {d[2]:+.4f} | {r['call']} |")
    return [f"| {label} | dataset | read | delta R@5 [95% CI] | delta FC@5 | delta hit@1 | call (0.0075) |",
            "|---|---|---|---|---|---|---|"] + md


def verdict(screen=None, out=None, out_root=None):
    """R2's verdict per arm from its re-call under its U2d null (R2, section 4, with GL's files) and its next form
    (R2, section 5). Beside, deciding nothing: each arm minus its base per read (R@5 with its interval, FC@5, hit@1),
    the webqsp and metaqa reads, musique in-domain (L-hotpotqa), and R2's term per epoch."""
    scr = Path(screen or SCREEN)
    per = {}
    for a in R2_ARMS:
        problems = []
        rp = recall_out(a, scr).with_suffix(".json")
        rows, rrec = recall_rows(rp, a, problems)
        np_ = null_problem(a, scr)
        if np_:
            problems.append(np_)
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        per[a] = {"verdict": v, "next_form": R2.next_form(v, gains, losses), "what": WHAT[a], "against": AGAINST[a],
                  "null_kind": NULL_KIND[a], "null_is": NULL_IS[a], "device": rrec.get("device"),
                  "rebased_through": rrec.get("rebased_through"), "gains": gains, "losses": losses,
                  "problems": problems, "calls": {tag(r): r.get("call") for r in rows},
                  "calls_filed": {tag(r): r.get("call_filed") for r in rows},
                  "delta_r5": {tag(r): r.get("delta") for r in rows}, "floors": {tag(r): r.get("floor") for r in rows},
                  "recall": rel(rp), "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
                  "beside": {"reads": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                                       if r.get("dataset") in R2.BESIDE},
                             "minus_base": {sp: G2.beside_rows(scr / f"{fit_name(a, sp)}.json", base_fit(a, sp))
                                            for sp in SPLITS},
                             "term": fit_terms(a, out_root)}}
    ck = f"{R2.CLAIM1_READ[0]} {R2.CLAIM1_READ[1]}"
    side = {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)} for a in R2_ARMS}
    rec = {"stage": STAGE, "declared_in": DECLARED, "decides": "R2", "r2_declared_in": R2.DECLARED, "rule": RULE,
           "arms": per, "musique_in_domain_beside": side, "not_beside": NOT_BESIDE,
           "zgl2_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = ["# R2's verdict on the L40S (GL): " + ", ".join(f"{a} **{per[a]['verdict']}**" for a in R2_ARMS), "",
          f"Rule: {RULE}.", ""]
    for a in R2_ARMS:
        p = per[a]
        md.append(f"- **{a}** ({WHAT[a]}) against {AGAINST[a]}, after the re-call under {NULL_IS[a]}: "
                  f"{len(p['gains'])} GAIN ({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS "
                  f"({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "")
                  + f". Next form (fixed in R2's section 5 before any number): {p['next_form']}.")
    md += [f"- Device: {json.dumps(per[R2_ARMS[0]]['device'])}.", "", "Beside, deciding nothing.", ""]
    for a in R2_ARMS:
        b = per[a]["beside"]
        md.append(f"- {a}, webqsp and metaqa: " + (", ".join(f"{k} {x['call']} ({fmt(x['delta'])})"
                                                            for k, x in b["reads"].items()) or "none") + ".")
        for sp in SPLITS:
            t = b["term"][sp]
            md.append(f"  - {a} {sp}: " + ("no R2 term in its gl stamp" if t is None else
                                           f"L_br per epoch {t['br_nll_by_epoch']}; the term on "
                                           f"{t['questions_with_term']} of {t['questions']} training questions "
                                           f"({t['share_with_term']:.4f})") + ".")
    md += [f"- musique in-domain ({ck}): " + ", ".join(f"{a} {side[a]['call']} ({fmt(side[a]['delta'])})"
                                                      for a in R2_ARMS) + "."]
    md += [f"- {x[0].upper() + x[1:]}." for x in NOT_BESIDE.values()]
    md += ["", "Each arm minus its base's L40S fit: R@5 with its 95% question-bootstrap interval, FC@5 and hit@1 (GL's "
               "compares; the call at 0.0075, before the re-call):", ""]
    md += minus_md({f"{a} {sp}": per[a]["beside"]["minus_base"][sp] for a in R2_ARMS for sp in SPLITS}, "arm, split")
    od = Path(out or OUT)
    od.mkdir(parents=True, exist_ok=True)
    LC.write_json(od / "r2-verdict.json", rec)
    (od / "r2-verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("zgl2 R2 verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in R2_ARMS))
    return rec


def twin_beside(screen=None, out=None):
    """The twin's rows, deciding nothing (section 4): its re-call against zspk under U2d's zspk null, and zg6 minus
    zg6t per read (what the edges add to the same model)."""
    scr = Path(screen or SCREEN)
    problems = []
    rp = recall_out(TWIN, scr).with_suffix(".json")
    rows, rrec = recall_rows(rp, TWIN, problems)
    edges = {sp: G2.beside_rows(scr / f"{TWIN_VS}{SUF[sp]}.json", fit_name(TWIN, sp)) for sp in SPLITS}
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    rec = {"stage": STAGE, "declared_in": DECLARED, "decides": None, "beside": True, "arm": TWIN, "what": WHAT[TWIN],
           "against": AGAINST[TWIN], "null_kind": NULL_KIND[TWIN], "null_is": NULL_IS[TWIN],
           "device": rrec.get("device"), "problems": problems, "gains": gains, "losses": losses,
           "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
           "floors": {tag(r): r.get("floor") for r in rows}, "zg6_minus_zg6t": edges, "recall": rel(rp),
           "recall_sha256": LC.sha_file(rp) if rp.exists() else None, "zgl2_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = ["# G6's twin on the L40S (GL): beside, deciding nothing", "",
          f"- **zg6t** ({WHAT[TWIN]}) against {AGAINST[TWIN]}, re-called under {NULL_IS[TWIN]}: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ".",
          f"- Device: {json.dumps(rrec.get('device'))}.", "",
          "zg6 minus zg6t, what the edges add to the same model: R@5 with its 95% question-bootstrap interval, FC@5 "
          "and hit@1 (GL's compares; the call at 0.0075):", ""]
    md += minus_md(edges, "split")
    od = Path(out or OUT)
    od.mkdir(parents=True, exist_ok=True)
    LC.write_json(od / "zg6t-beside.json", rec)
    (od / "zg6t-beside.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zgl2 twin beside: {len(gains)} GAIN, {len(losses)} LOSS against zspk"
        + (f"; problems: {'; '.join(problems)}" if problems else ""))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def real_state():
    """GL's real folders and files, with their modification times."""
    paths = [ROOT / OUT, *(ROOT / SCREEN).glob("scr-gl-*"), *(ROOT / SCREEN / "fits").glob("scr-gl-*")]
    out = {}
    for p in paths:
        out[str(p)] = p.stat().st_mtime if p.exists() else None
        if p.is_dir():
            out.update({str(q): q.stat().st_mtime for q in p.rglob("*")})
    return out


def fake_u2d_fit(td, name, dev):
    """One of U2d's fits in td: screen.json and read.json with U2d's stamps (the fields zu2d's device check reads)."""
    b, arm, seed, split = Z.FITS[name]
    d = Path(td) / "fits" / name
    d.mkdir(parents=True, exist_ok=True)
    LC.write_json(d / "screen.json", {"arm": arm, "split": split, "seed": seed, "u2d": {
        "declared_in": Z.DECLARED, "fit": name, "build": b, "arm": arm, "seed": seed, "split": split, "device": dev}})
    LC.write_json(d / "read.json", {"rows_cap": LG.READ_ROWS, "u2d": {"declared_in": Z.DECLARED, "fit": name,
                                                                      "device": dev}})


def toy_term(n=10, with_term=7):
    return {"lam": R2.LAM, "batches": 2, "batches_with_term": 2, "questions": n, "questions_with_gold": n,
            "questions_with_term": with_term, "epochs": [{"br_nll_mean": 1.5, "share_with_term": with_term / n},
                                                         {"br_nll_mean": 1.2, "share_with_term": with_term / n}],
            "listwise_mean": 0.9, "br_nll_mean": 1.35, "share_with_term": with_term / n}


def fake_gl_fit(td, name, dev, read_dev=None, rows_cap=None):
    """One of GL's fits in td, stamped as GL's train and read stamp it (the fields this file reads)."""
    arm, split = FITS[name]
    d = Path(td) / "fits" / name
    d.mkdir(parents=True, exist_ok=True)
    u = {"declared_in": DECLARED, "stage": STAGE, "fit": name, "build": BUILD, "arm": arm, "seed": 0, "split": split,
         "device": dev, "rows_cap": ROWS_CAP[arm]}
    if arm in R2_ARMS:
        u["r2"] = {"declared_in": R2.DECLARED, "constants": R2.constants(), "term": toy_term()}
    LC.write_json(d / "screen.json", {"arm": arm, "split": split, "seed": 0, KEY: u})
    cap = ROWS_CAP[arm] if rows_cap is None else rows_cap
    LC.write_json(d / "read.json", {"rows_cap": cap, KEY: {"declared_in": DECLARED, "fit": name, "rows_cap": cap,
                                                           "device": read_dev or dev}})


def fake_cmp_gl(td, cname, deltas, base_r5=0.6, dev=None, stamp=True):
    """GL's compare cname in lean_screen's form (deltas {dataset: d}; base_r5 a number or {dataset: R@5})."""
    new, bases = COMPARES[cname]
    split = (FITS.get(new) or Z.FITS[new][1:])[-1]
    rows = []
    for ds in LG.EVAL_ORDER:
        d = deltas.get(ds, 0.0)
        b0 = base_r5.get(ds, 0.6) if isinstance(base_r5, dict) else base_r5
        lo, hi = round(d - 0.004, 4), round(d + 0.004, 4)
        rows.append({"dataset": ds, "read": "in-domain" if ds in LG.FITS[split] else "zero-shot", "questions": 100,
                     "base": [b0, 0.5, 0.4], "new": [round(b0 + d, 4), 0.5, 0.4], "rrf": [0.4, 0.3, 0.2],
                     "delta": [d, 0.001, -0.002], "lo": [lo, 0, -0.004], "hi": [hi, 0.002, 0],
                     "call": S.call(d, lo, hi)})
    bp = f"outputs/screen/fits/{bases[0]}"
    v = S.verdict([r["call"] for r in rows])
    rec = {"new": str(Path(td) / "fits" / new), "bases": [bp], "candidate": "p@swa", "carve": "s1eval",
           "trained_on": sorted(LG.FITS[split]), "by_base": {bp: {"rows": rows, "verdict": v, "decides": True}},
           "verdict": v}
    if stamp:
        rec[KEY] = {"declared_in": DECLARED, "stage": STAGE, "compare": cname, "new": new, "bases": list(bases),
                    "device": dev or Z.device_of(Z.TOY_ENV)}
    p = Path(td) / f"{cname}.json"
    LC.write_json(p, rec)
    return p


def fake_world(td, gl, zg6, big=None, dev=None):
    """Every file GL's calls read, in td: U2d's fourteen base-build fits, its zrckp and zspk nulls (big: zspk's
    {(split, dataset): (D1, D2)}), its compares of zg6 with zspk (zg6 {(split, dataset): d}), GL's six fits and
    eight compares (gl {arm: {(split, dataset): d}}; 'edges' for zg6 against the twin)."""
    dev = dev or Z.device_of(Z.TOY_ENV)
    Z.fake_null(td, "zspk", big, base_r5=0.6)
    Z.fake_null(td, "zrckp", base_r5=0.59)
    for sp in SPLITS:
        Z.fake_cmp(td, f"scr-u2d-base-zg6-vs-zspk{SUF[sp]}", base_fit("zg6m", sp), base_fit(TWIN, sp),
                   {ds: d for (s, ds), d in zg6.items() if s == sp}, base_r5=0.6)
    for n, f in Z.FITS.items():
        if f[0] == BUILD:
            fake_u2d_fit(td, n, dev)
    for n in FITS:
        fake_gl_fit(td, n, dev)
    r5 = {"zrckpb": 0.59, TWIN: 0.6}
    for sp in SPLITS:
        r5_zg6 = {ds: round(0.6 + zg6.get((sp, ds), 0.0), 4) for ds in LG.EVAL_ORDER}
        for a in GL_ARMS:
            fake_cmp_gl(td, fit_name(a, sp), {ds: d for (s, ds), d in gl.get(a, {}).items() if s == sp},
                        base_r5=r5_zg6 if a == "zg6m" else r5[a], dev=dev)
        fake_cmp_gl(td, f"{TWIN_VS}{SUF[sp]}", {ds: d for (s, ds), d in gl.get("edges", {}).items() if s == sp},
                    base_r5=0.58, dev=dev)


def run_calls(td):
    assert calls(td, td / "fits") == 0
    return verdict(td, td / "gl", td / "fits"), twin_beside(td, td / "gl")


class Patched:
    """Module attributes replaced inside a with block, restored after."""

    def __init__(self, *triples):
        self.triples = triples

    def __enter__(self):
        self.saved = [(m, k, getattr(m, k)) for m, k, _v in self.triples]
        for m, k, v in self.triples:
            setattr(m, k, v)
        return self

    def __exit__(self, *exc):
        for m, k, v in self.saved:
            setattr(m, k, v)
        return False


def selftest_fit_steps(t):
    """train and read on fakes of rmatch's train and read: the refusals before any data, R2's term around R2's arms
    only, the base build's roots inside, the stamps, the rows cap and the device."""
    import torch
    env = dict(Z.TOY_ENV)
    dev = Z.device_of(env)
    r = Z.roots(BUILD)
    cache = str(r["cache"])
    seen = []
    base_loss, base_feats, base_state = LG.listwiseD, R2.R2Mix.r2_feats, LG.cpu_state
    roots0 = (Z2.CACHE, Z3.KTChainCarveZRC.KT_ROOT, R2.R2KTLinkCarveBase.KT_ROOT)

    def fake_train(argv, split, term=7):
        em = LG.listwiseD is not base_loss
        seen.append({"argv": list(argv), "split": split, "em": em, "feats_em": R2.R2Mix.r2_feats is R2.r2_feats_em,
                     "cache": Path(Z2.CACHE) == Path(r["cache"]), "kt": Path(Z3.KTChainCarveZRC.KT_ROOT) == r["kt"],
                     "r2_kt": Path(R2.R2KTLinkCarveBase.KT_ROOT) == r["kt"]})
        name, out_root = S2.where(argv)
        d = Path(out_root) / name
        d.mkdir(parents=True, exist_ok=True)
        LC.write_json(d / "screen.json", {"arm": R.arm_of(argv), "split": split, "seed": 0})
        if em:
            for _ in range(2):
                ep = R2.STATS["_ep"]
                ep["batches"] += 1
                ep["questions"] += 10
                ep["questions_with_gold"] += 10
                ep["batches_with_term"] += 1 if term else 0
                ep["questions_with_term"] += term
                ep["br_sum"] += 1.5 * term
                LG.cpu_state(torch.nn.Linear(1, 1))
        return 0

    def fake_read(argv):
        name, out_root = S2.where(argv)
        caps = [argv[i + 1] for i, x in enumerate(argv) if x == "--rows-cap"]
        seen.append({"read": list(argv), "caps": caps})
        LC.write_json(Path(out_root) / name / "read.json", {"rows_cap": int(caps[-1]) if caps else LG.READ_ROWS})
        return 0

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        kt_calls = []

        def kt_records(carves, root):
            kt_calls.append((list(carves), Path(root)))
            return {f"{d}/{c}": "toy" for d, c in carves}

        def toy(root):
            return {"toy": rel(root)}

        with Patched((S, "OUT", td), (RM, "train", fake_train), (RM, "read", fake_read), (Z3, "kt_records", kt_records),
                     (Z2, "_BUILT", toy), (Z2, "_ZL_BUILT", toy)):
            ms, hp = SPLITS

            def args(name, arm, *extra):
                return ["train", "--name", name, "--arm", arm, "--device", "cuda", "--cache-root", cache, *extra]

            # the refusals, before any data: not GL's fit, another arm or split, a fixed flag, another device
            Z.stops(train, args("scr-r2-zg6m", "zg6m"), ms, "modal", env, has="not one of GL's fits")
            Z.stops(train, args("scr-gl-zg6m", "zrckpb"), ms, "modal", env, has="is zg6m's fit of L-musique")
            Z.stops(train, args("scr-gl-zg6m", "zg6m"), hp, "modal", env, has="is zg6m's fit of L-musique")
            for flag in (["--seed", "1"], ["--hidden", "64"], ["--threads", "4"], ["--out-root", str(td)],
                         ["--conf", "x"]):
                Z.stops(train, args("scr-gl-zg6m", "zg6m", *flag), ms, "modal", env, has="would change that")
            Z.stops(train, args("scr-gl-zg6m", "zg6m"), ms, "laptop", env, has="on Modal's L40S")
            Z.stops(train, args("scr-gl-zg6m", "zg6m"), ms, "modal", {**env, "gpu": "Tesla T4"}, has="on Modal's L40S")
            Z.stops(train, args("scr-gl-zg6m", "zg6m"), ms, "modal", {**env, "image": "other"}, has="on Modal's L40S")
            Z.stops(train, args("scr-gl-zg6m", "zg6m"), ms, "modal", {**env, "where": "elsewhere"},
                    has="on Modal's L40S")
            assert not seen and not kt_calls and not list(td.iterdir()), "nothing runs before the refusals pass"
            # R2's arms train inside R2's EM term, on the base build's roots, with U2d's card and seed 0
            for name in ("scr-gl-zrckpb", "scr-gl-zg6m-hp", "scr-gl-zg6t"):
                arm, split = FITS[name]
                assert train(args(name, arm), split, "modal", env) == 0
                s = seen[-1]
                assert s["em"] == s["feats_em"] == (arm in R2_ARMS), (name, s)
                assert s["cache"] and s["kt"] and s["r2_kt"], (name, s)
                assert s["argv"][-2:] == ["--seed", "0"] and "--device" in s["argv"] and s["split"] == split
                assert kt_calls[-1] == (Z3.train_carves(split), r["kt"])
                rec, u = fit_stamp(name, td / "fits")
                assert u["device"] == dev and u["rows_cap"] == ROWS_CAP[arm] and u["what"] == WHAT[arm]
                assert u["base_arm"] == BASE_OF[arm] and u["zrc_records"] == {"toy": rel(r["ch_zrc"])}
                assert rec["zgl2_sha256"] == LC.sha_src(__file__) and rec["zr2br_sha256"] == LC.sha_src(R2.__file__)
                if arm in R2_ARMS:
                    tm = u["r2"]["term"]
                    assert tm["questions_with_term"] == 14 and len(tm["epochs"]) == 2 and tm["questions"] == 20
                    assert u["r2"]["constants"] == R2.DECLARED_CONST and u["head"]["r2_constants"] == R2.DECLARED_CONST
                    assert u["head"]["base"]["kt_root"] == rel(r["kt"])
                else:
                    assert "r2" not in u and u["head"]["twin"] is True and u["head"]["model"] == G6.SETTINGS
            assert (LG.listwiseD, R2.R2Mix.r2_feats, LG.cpu_state) == (base_loss, base_feats, base_state)
            assert (Z2.CACHE, Z3.KTChainCarveZRC.KT_ROOT, R2.R2KTLinkCarveBase.KT_ROOT) == roots0, \
                "on_u2d's roots are restored"
            # a fit whose batches never carried R2's term stops unstamped
            with Patched((RM, "train", lambda argv, split: fake_train(argv, split, term=0))):
                Z.stops(train, args("scr-gl-zg6m", "zg6m"), ms, "modal", env, has="carried R2's term")
            assert KEY not in json.loads((td / "fits" / "scr-gl-zg6m" / "screen.json").read_text(encoding="utf-8"))
            assert (LG.listwiseD, R2.R2Mix.r2_feats, LG.cpu_state) == (base_loss, base_feats, base_state)
            # the reads: the arm's rows cap, never the caller's; on the fit's device; R2's constants as declared
            rd = ["--device", "cuda", "--cache-root", cache]
            for name in ("scr-gl-zrckpb", "scr-gl-zg6t"):
                arm, _sp = FITS[name]
                assert read(["--name", name, *rd], "modal", env) == 0
                assert seen[-1]["caps"] == [str(ROWS_CAP[arm])], seen[-1]
                assert kt_calls[-1] == (Z3.eval_carves(), r["kt"])
                ru = json.loads((td / "fits" / name / "read.json").read_text(encoding="utf-8"))
                assert ru["rows_cap"] == ru[KEY]["rows_cap"] == ROWS_CAP[arm] and ru[KEY]["device"] == dev
                assert fit_device(name, td / "fits") == dev
            n = len(seen)
            Z.stops(read, ["--name", "scr-gl-zg6t", "--rows-cap", "262144", *rd], "modal", env, has="--rows-cap 65536")
            Z.stops(read, ["--name", "scr-gl-zg6t", "--rows", "1", *rd], "modal", env, has="would change that")
            Z.stops(read, ["--name", "scr-gl-zg6t", *rd], "modal", {**env, "cudnn": 1}, has="was fitted on")
            Z.stops(read, ["--name", "scr-gl-zg6m", *rd], "modal", env, has="is not GL's zg6m fit")
            sj = td / "fits" / "scr-gl-zrckpb" / "screen.json"
            x = json.loads(sj.read_text(encoding="utf-8"))
            x[KEY]["r2"]["constants"] = {**R2.DECLARED_CONST, "K": 6}
            LC.write_json(sj, x)
            Z.stops(read, ["--name", "scr-gl-zrckpb", *rd], "modal", env, has="fitted with R2 constants")
            assert len(seen) == n, "a refused read runs nothing"
            # a read whose record holds another rows cap stops
            with Patched((RM, "read", lambda argv: (fake_read(argv), LC.write_json(
                    Path(S2.where(argv)[1]) / S2.where(argv)[0] / "read.json", {"rows_cap": 1}))[0])):
                Z.stops(read, ["--name", "scr-gl-zg6m-hp", *rd], "modal", env, has="not zg6m's 65536")
            # the command line: zu2d_modal.py's argv reaches train and read, and the laptop is refused
            n = len(seen)
            Z.stops(main, ["train", "--split", ms, "--name", "scr-gl-zg6m", "--arm", "zg6m", *rd],
                    has="on Modal's L40S")
            Z.stops(main, ["train", "--split", hp, "--name", "scr-gl-zg6m", "--arm", "zg6m", *rd],
                    has="is zg6m's fit of L-musique")
            Z.stops(main, ["read", "--name", "scr-gl-zg6t", *rd], has="GL reads on Modal's L40S")
            Z.stops(main, ["verdict", "--screen", str(td)], has="no options")
            assert len(seen) == n, "a refused command runs nothing"
            assert (Z2.CACHE, Z3.KTChainCarveZRC.KT_ROOT, R2.R2KTLinkCarveBase.KT_ROOT) == roots0
    t.append("train and read")


def selftest():
    t0 = time.time()
    W, GN, LS = "WITHIN", "GAIN", "LOSS"
    before = real_state()
    saved = SP.load, SR.load_null
    ms, hp = SPLITS
    done = []
    assert set(COMPARES) == {*FITS, f"{TWIN_VS}", f"{TWIN_VS}-hp"} and len(call_fits("zg6m")) == 10
    assert call_fits("zrckpb") == ["scr-gl-zrckpb", "scr-gl-zrckpb-hp", "scr-u2d-base-zrckp", "scr-u2d-base-zrckp-hp",
                                   "scr-u2d-base-zrckp-s1", "scr-u2d-base-zrckp-s2", "scr-u2d-base-zrckp-s1-hp",
                                   "scr-u2d-base-zrckp-s2-hp"]
    assert ROWS_CAP == {"zrckpb": 262144, "zg6m": 65536, "zg6t": 65536} and ARMS[TWIN][0].TWIN
    selftest_fit_steps(done)
    # zrckpb: a GAIN and no LOSS -> ADOPT; zg6m: a GAIN and a LOSS -> R2g, and L-hotpotqa 2wiki a GAIN at 0.0075 below
    # its raised floor (zspk's 0.0139); the twin: a LOSS against zspk; zg6 above the twin on L-musique musique
    zg6 = {(ms, "metaqa"): 0.005, (hp, "webqsp"): -0.05, (hp, "2wiki"): 0.003}
    gl = {"zrckpb": {(ms, "metaqa"): 0.0102},
          "zg6m": {(ms, "metaqa"): 0.012, (hp, "webqsp"): -0.03, (hp, "2wiki"): 0.0101},
          TWIN: {(ms, "musique"): -0.02}, "edges": {(ms, "musique"): 0.02}}
    big = {(hp, "2wiki"): (0.007, -0.0069)}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake_world(td, gl, zg6, big)
        rec, tw = run_calls(td)
        a, b = rec["arms"]["zrckpb"], rec["arms"]["zg6m"]
        assert a["verdict"] == "ADOPT_FOR_FULL_RUN" and a["next_form"] == R2.NEXT["ADOPT_FOR_FULL_RUN"], a
        assert a["gains"] == [f"{ms} metaqa"] and not a["losses"] and not a["problems"]
        assert b["verdict"] == "NOT_ADOPTED" and b["next_form"] == R2.NEXT["GAIN_AND_LOSS"], b
        assert b["gains"] == [f"{ms} metaqa"] and b["losses"] == [f"{hp} webqsp"]
        assert b["calls"][f"{hp} 2wiki"] == W and b["calls_filed"][f"{hp} 2wiki"] == GN
        assert abs(b["floors"][f"{hp} 2wiki"] - 0.0139) < 1e-4 and b["floors"][f"{ms} metaqa"] == S.FLOOR
        assert sorted(b["rebased_through"]) == sorted(SPLITS) and not a["rebased_through"]
        rr = json.loads((td / "scr-gl-zg6m-pair-recall.json").read_text(encoding="utf-8"))
        row = next(x for x in rr["rows"] if (x["split"], x["dataset"]) == (ms, "metaqa"))
        assert row["base"] == 0.605 and rr["null_kind"] == NULL_KIND["zg6m"] and rr["deciding"] is True
        assert "re-based L-musique" in rr["null"] and "U2d zspk L-hotpotqa seed 2" in rr["null"]
        md = (td / "scr-gl-zg6m-pair-recall.md").read_text(encoding="utf-8")
        assert md.startswith("GL (") and "zspk's seeds 1 and 2" in md and "step 1's p@swa" not in md
        assert "re-based through scr-u2d-base-zg6-vs-zspk, scr-u2d-base-zg6-vs-zspk-hp" in md
        assert "zrckp's seeds 1 and 2" in (td / "scr-gl-zrckpb-pair-recall.md").read_text(encoding="utf-8")
        assert a["device"] == b["device"] == Z.device_of(Z.TOY_ENV)
        assert a["beside"]["term"][ms]["questions_with_term"] == 7 and a["beside"]["term"][ms]["br_nll_by_epoch"] == [
            1.5, 1.2]
        assert set(a["beside"]["reads"]) == {f"{sp} {d}" for sp in SPLITS for d in R2.BESIDE}
        assert all(len(a["beside"]["minus_base"][sp]) == 6 for sp in SPLITS)
        assert rec["musique_in_domain_beside"]["zg6m"]["call"] == W
        mb = {x["dataset"]: x for x in b["beside"]["minus_base"][hp]}
        assert mb["webqsp"]["delta"][0] == -0.03 and mb["webqsp"]["call"] == LS
        vmd = (td / "gl" / "r2-verdict.md").read_text(encoding="utf-8")
        assert vmd.startswith("# R2's verdict on the L40S (GL): zrckpb **ADOPT_FOR_FULL_RUN**, zg6m **NOT_ADOPTED**")
        assert "zrckp + R2's bridge head" in vmd and "7 October re-calls are not computed" in vmd
        assert tw["losses"] == [f"{ms} musique"] and not tw["gains"] and not tw["problems"] and tw["decides"] is None
        e = {x["dataset"]: x for x in tw["zg6_minus_zg6t"][ms]}
        assert e["musique"]["delta"][0] == 0.02 and e["musique"]["call"] == GN and len(tw["zg6_minus_zg6t"][hp]) == 6
        assert (td / "gl" / "zg6t-beside.md").read_text(encoding="utf-8").startswith("# G6's twin on the L40S")
        tr = json.loads((td / "scr-gl-zg6t-pair-recall.json").read_text(encoding="utf-8"))
        assert tr["deciding"] is False and tr["decides"] is None and not tr["rebased_through"]
        # INCOMPLETE: a re-call that is not GL's, one without twelve reads, none, a U2d null file that fails its checks
        rp = td / "scr-gl-zrckpb-pair-recall.json"
        x = json.loads(rp.read_text(encoding="utf-8"))
        LC.write_json(rp, {**x, "stage": "R2"})
        assert verdict(td, td / "gl", td / "fits")["arms"]["zrckpb"]["verdict"] == "INCOMPLETE"
        LC.write_json(rp, {**x, "rows": x["rows"][:11]})
        v = verdict(td, td / "gl", td / "fits")["arms"]["zrckpb"]
        assert v["verdict"] == "INCOMPLETE" and v["next_form"] == R2.NEXT["INCOMPLETE"]
        rp.unlink()
        assert verdict(td, td / "gl", td / "fits")["arms"]["zrckpb"]["verdict"] == "INCOMPLETE"
        LC.write_json(rp, x)
        assert verdict(td, td / "gl", td / "fits")["arms"]["zrckpb"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        (td / "scr-u2d-null-zspk-s2-hp.json").unlink()
        v = verdict(td, td / "gl", td / "fits")["arms"]
        assert v["zg6m"]["verdict"] == "INCOMPLETE" and "zspk null fails" in v["zg6m"]["problems"][-1]
        assert v["zrckpb"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        # compare-all: a filed compare is kept, a missing fit's compare is named and skipped, lean_screen's compare runs
        # on the fits' relative paths and is stamped with GL's record and the device
        made = []

        def fake_compare(argv):
            k = dict(zip(argv[1::2], argv[2::2]))
            made.append(k)
            LC.write_json(Path(k["--out"]).with_suffix(".json"), {"new": k["--new"], "bases": k["--base"].split(",")})
            return 0

        for cn in ("scr-gl-zg6t", "scr-gl-zg6t-hp", f"{TWIN_VS}-hp"):
            (td / f"{cn}.json").unlink()
        import shutil
        shutil.rmtree(td / "fits" / "scr-gl-zg6t")
        with Patched((S2, "main", fake_compare)):
            assert compare_all(None, td / "fits", td) == 1
        assert [Path(k["--new"]).name for k in made] == ["scr-gl-zg6t-hp", "scr-u2d-base-zg6-hp"]
        c = json.loads((td / f"{TWIN_VS}-hp.json").read_text(encoding="utf-8"))
        assert c[KEY]["new"] == "scr-u2d-base-zg6-hp" and c[KEY]["bases"] == ["scr-gl-zg6t-hp"]
        assert c[KEY]["device"] == Z.device_of(Z.TOY_ENV) and c[KEY]["compare"] == f"{TWIN_VS}-hp"
    done.append("verdict A")
    # zg6m with no GAIN -> R2e; the twin's pair stops when its fits are not of one device
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake_world(td, {"zrckpb": {(hp, "squad"): 0.003}, "zg6m": {(ms, "squad"): 0.003}}, {})
        rec, _tw = run_calls(td)
        for arm in R2_ARMS:
            v = rec["arms"][arm]
            assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == R2.NEXT["NO_GAIN"] and not v["gains"], v
    done.append("verdict B")
    # the refusals: the loader, the devices, the rows cap, a pair that is not GL's
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fake_world(td, {"zrckpb": {(ms, "metaqa"): 0.0102}}, {})
        dev = Z.device_of(Z.TOY_ENV)
        cmp_ = td / "scr-gl-zrckpb.json"
        good = cmp_.read_text(encoding="utf-8")

        def bad(has, **edit):
            c = json.loads(good)
            for k, val in edit.items():
                if k == KEY:
                    c[KEY].update(val)
                else:
                    c[k] = val
            cmp_.write_text(json.dumps(c), encoding="utf-8")
            Z.stops(pair, "zrckpb", td, td / "fits", has=has)
            cmp_.write_text(good, encoding="utf-8")

        bad("is not GL's compare", **{KEY: {"declared_in": "docs/OTHER.md"}})
        bad("is not GL's compare", **{KEY: {"compare": "scr-r2-zrckpb"}})
        bad("is not GL's compare", **{KEY: {"bases": ["scr-u2d-base-zspk"]}})
        bad("not on the fits' device", **{KEY: {"device": {**dev, "gpu": "Tesla T4"}}})
        bad("is decided against", bases=["outputs/screen/fits/scr-u2d-base-zspk"])
        bad("not GL's zrckpb fit", new=str(td / "fits" / "scr-gl-zg6m"))
        bad("not p@swa on s1eval", candidate="p@last")
        bad("not p@swa on s1eval", carve="s1sel")
        bad("trained on", trained_on=["squad"])
        c = json.loads(good)
        b0 = c["bases"][0]
        c["by_base"][b0]["rows"] = c["by_base"][b0]["rows"][:5]
        cmp_.write_text(json.dumps(c), encoding="utf-8")
        Z.stops(pair, "zrckpb", td, td / "fits", has="six datasets")
        cmp_.write_text(good, encoding="utf-8")
        # a null fit on another image, a GL fit read on another device or at another rows cap: nothing is called
        fake_u2d_fit(td, Z.fit_name(BUILD, "zrckp", 2, hp), {**dev, "image": "other"})
        Z.stops(pair, "zrckpb", td, td / "fits", has="different devices")
        Z.stops(recall, "zrckpb", td, td / "fits", has="different devices")
        fake_u2d_fit(td, Z.fit_name(BUILD, "zrckp", 2, hp), dev)
        fake_gl_fit(td, "scr-gl-zrckpb-hp", dev, read_dev={**dev, "gpu": "Tesla T4"})
        Z.stops(pair, "zrckpb", td, td / "fits", has="on the device it was fitted on")
        fake_gl_fit(td, "scr-gl-zrckpb-hp", dev, rows_cap=65536)
        Z.stops(pair, "zrckpb", td, td / "fits", has="--rows-cap 65536")
        assert not pair_out("zrckpb", td).with_suffix(".json").exists()
        fake_gl_fit(td, "scr-gl-zrckpb-hp", dev)
        pair("zrckpb", td, td / "fits")
        x = json.loads(pair_out("zrckpb", td).with_suffix(".json").read_text(encoding="utf-8"))
        LC.write_json(pair_out("zrckpb", td).with_suffix(".json"), {**x, "stage": "R2"})
        Z.stops(recall, "zrckpb", td, td / "fits", has="not GL's pair")
        # zg6m's re-base reads U2d's compare of zg6 with zspk: without it, nothing is re-called
        pair("zg6m", td, td / "fits")
        (td / "scr-u2d-base-zg6-vs-zspk-hp.json").unlink()
        Z.stops(recall, "zg6m", td, td / "fits", has="the null's re-base")
        assert not recall_out("zg6m", td).with_suffix(".json").exists()
        # calls names a skipped arm and goes on with the others
        assert calls(td, td / "fits") == 1 and recall_out(TWIN, td).with_suffix(".json").exists()
    done.append("refusals")
    assert (SP.load, SR.load_null) == saved, "screen_pair's load and screen_recall's null are restored"
    assert real_state() == before, "nothing is written under GL's or the screens' real folders"
    print(f"zgl2 selftest: OK ({', '.join(done)}; {time.time() - t0:.1f}s)", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", choices=("train", "read", "compare-all", "calls", "verdict"))
    ap.add_argument("--split", default="L-musique", choices=SPLITS)
    ap.add_argument("--modal", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    where = "modal" if k.modal else "laptop"
    if k.cmd == "train":
        return train(["train"] + rest, k.split, where)
    if k.cmd == "read":
        return read(rest, where)
    if rest:
        raise SystemExit(f"zgl2 {k.cmd}: no options ({rest})")
    with Z.at_root():
        if k.cmd == "compare-all":
            return compare_all()
        if k.cmd == "calls":
            return calls()
        if k.cmd == "verdict":
            rec = verdict()
            twin_beside()
            return 2 if any(v["verdict"] == "INCOMPLETE" for v in rec["arms"].values()) else 0
    raise SystemExit("zgl2: train, read, compare-all, calls, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
