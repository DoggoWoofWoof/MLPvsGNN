"""U2g's cell Y (docs/U2G_GRAPH_AND_RULE.md): P1c2's pool rule on W2_S, the LLM-free links. U2d's code runs it, with a
third build, y, registered in memory (zu2d_pools' roots, construction, rule and carve opener; base and new unchanged):
    roots         outputs/u2g/y/ (pools, look, cache, links, kt, gate.json); /out/y on mpr-u2d
    construction  U2G:y:KIND_T; the rule P1c2's: B, then KIND_T's order outside B, its first |I_q| - |B| nodes
    carve opener  zu2d_pools.open_carve("new") (W2_S swapped in as U2d's new build swaps it, its description checked
                  against U2e's and the seeds against today's), then the question's branch set to P1c2's rule
Y builds squad, musique, hotpotqa and 2wiki. metaqa and webqsp are U2d's base build as filed (the KB on both graphs, and
P1c2's rule): every fit opens each carve on its dataset's build, under that build's declaration.

The stages (zu2d_modal.py's y stages run them on Modal):
    python outputs/mp_unified/zu2g.py pools D --carve C [--shard I/K] [--limit Q]
    python outputs/mp_unified/zu2g.py merge D [--new-manifest P]          (the merge, then same-as-new)
    python outputs/mp_unified/zu2g.py same-as-new D [--new-manifest P]
    python outputs/mp_unified/zu2g.py look --dataset D --carve C --full [--shard i/n] [--modal] [--dev Q]
    python outputs/mp_unified/zu2g.py copy-bases
    python outputs/mp_unified/zu2g.py links D --carve C [--modal]
    python outputs/mp_unified/zu2g.py kt D --carve C
    python outputs/mp_unified/zu2g.py gate
    python outputs/mp_unified/zu2g.py train --split L-hotpotqa --name scr-u2g-y-zspk-hp --arm zspk --device cuda \\
        --cache-root outputs/u2g/y/cache --modal
    python outputs/mp_unified/zu2g.py read --name scr-u2g-y-zspk-hp --device cuda --cache-root outputs/u2g/y/cache \\
        --modal
On the laptop, from the fetched reads (section 5's decisions, section 6's readings):
    python outputs/mp_unified/zu2g.py compare-all
    python outputs/mp_unified/zu2g.py decide
    python outputs/mp_unified/zu2g.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if __name__ == "__main__" and sys.argv[1:2] == ["look"]:
    import look_step1  # noqa: E402,F401  (look_x_six sets the thread variables before numpy and torch load)

import zu2d_pools as ZP  # noqa: E402  (U2d's pools, merge, KT and gate; numba-free at import)

import numpy as np  # noqa: E402

DECLARED = "docs/U2G_GRAPH_AND_RULE.md"
OUT = ROOT / "outputs" / "u2g"
SCREEN = ROOT / "outputs" / "screen"
BUILD = "y"
U2D_DECLARED, U2D_BUILDS = "docs/U2D_REFIT_ON_U2E.md", ("base", "new")
if (ZP.DECLARED, ZP.BUILDS) != (U2D_DECLARED, U2D_BUILDS):
    raise SystemExit(f"zu2g: zu2d_pools holds {ZP.DECLARED} {ZP.BUILDS}, not U2d's as filed")
OWN = ("squad", "musique", "hotpotqa", "2wiki")      # built on Y's roots
LINKED = ("metaqa", "webqsp")                         # U2d's base build, as filed
assert set(OWN) | set(LINKED) == set(ZP.SIX) and not set(OWN) & set(LINKED)
CARVES_Y = tuple((d, c) for d in OWN for c in ZP.PC2.carves_for(d))
assert len(CARVES_Y) == 8 and set(CARVES_Y) <= set(ZP.SCREEN)
assert ZP.PC2.CONSTRUCTION == "KIND_T"
RULE_Y = ZP.PC2.CONSTRUCTION
CONSTRUCTION_Y = f"U2G:{BUILD}:{RULE_Y}"
NEW_CONSTRUCTION = f"U2D:new:{ZP.PICK}"
NEW_MANIFESTS = {                                     # U2d's new manifests as committed (sha256 of their LF bytes)
    "squad": "19cc1eda865a3e6deef13706b7aa7a12ecd8849ec7d22a99467e221c419571dd",
    "musique": "3eb0a01e50a8df4cb75f0e125063a045bf552f7f15a8966123091558c151da28",
    "hotpotqa": "3b8e834db269b955a2ff4ec69a05bf171450de9b66ef43ab2061a57fd669cf22",
    "2wiki": "ea87a8e775a8fc4be1bfb6f305b03ae6b7eba3b6ed5030cf3744bd84b94a19e8"}
SAME_KEYS = ("questions", "carve_ids_sha256", "B_q", "pool_mean", "today_inc_mean", "shares", "I", "graph_detail")
FILL_TOL = 0.05                                       # new's fill_want_mean is rounded to one decimal
GRAPH_Y = "W2_S (U2b's chosen LLM-free links), swapped in on the passage datasets as U2d's new build swaps it"
log, utc = ZP.log, ZP.utc


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rel(p):
    p = Path(p)
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


# ── Y, registered in zu2d_pools (in memory) ──────────────────────────────────

_ORIG = getattr(ZP, "_U2G_ORIG", None) or {"roots": ZP.roots, "construction_of": ZP.construction_of,
                                            "open_carve": ZP.open_carve}
ZP._U2G_ORIG = _ORIG                     # one set of originals, however often this file is imported


def roots_y(build):
    """zu2d_pools' roots of a build, Y's under outputs/u2g/y."""
    if build != BUILD:
        return _ORIG["roots"](build)
    b = OUT / BUILD
    return {"pools": b / "pools", "parts": b / "pools" / "parts", "look": b / "look", "cache": b / "cache",
            "kt": b / "kt", "gate": b / "gate.json", "check": b / "check"}


def construction_y(build):
    return CONSTRUCTION_Y if build == BUILD else _ORIG["construction_of"](build)


def open_carve_y(build, name, carve, threads):
    """Y's carve: zu2d_pools.open_carve("new") (W2_S swapped in, its description checked against the one U2e's pick ran
    on, the seeds against today's), then zu2d_pools.question's branch set to P1c2's rule: KIND_T's order outside B, its
    first |I_q| - |B| nodes, I_q the frozen pool on W2_S (the new build's fill). No dense node reader: P1c2's rule reads
    no cosine."""
    if build != BUILD:
        return _ORIG["open_carve"](build, name, carve, threads)
    if name not in OWN:
        raise SystemExit(f"zu2g: Y builds {OWN}; {name} is U2d's base build, as filed")
    cx = _ORIG["open_carve"]("new", name, carve, threads)
    cx.build, cx.nodes = "base", None
    return cx


def register():
    """Y's roots, construction, rule and carve opener in zu2d_pools; base and new pass through unchanged.
    zu2d_pools.BUILDS stays U2d's (zu2d builds its compares from it at import); y_stage admits y where a check needs it."""
    ZP.roots, ZP.construction_of, ZP.open_carve = roots_y, construction_y, open_carve_y
    ZP.RULE[BUILD] = RULE_Y


register()


def _lazy(name):
    """zu2d or look_u2d, imported only outside a stage of Y: both copy zu2d_pools' declaration (zu2d also its builds)
    at import."""
    if name not in sys.modules and (ZP.DECLARED, ZP.BUILDS) != (U2D_DECLARED, U2D_BUILDS):
        raise SystemExit(f"zu2g: {name} would be imported inside a stage of Y ({ZP.DECLARED}, {ZP.BUILDS})")
    return importlib.import_module(name)


def Zm():
    """zu2d, under U2d's declaration and builds."""
    Z = _lazy("zu2d")
    if (Z.DECLARED, Z.BUILDS) != (U2D_DECLARED, U2D_BUILDS):
        raise SystemExit(f"zu2g: zu2d holds {Z.DECLARED} {Z.BUILDS}, not U2d's")
    return Z


def LUm():
    """look_u2d, under U2d's declaration."""
    LU = _lazy("look_u2d")
    if LU.DECLARED != U2D_DECLARED:
        raise SystemExit(f"zu2g: look_u2d holds {LU.DECLARED}, not U2d's")
    return LU


@contextlib.contextmanager
def y_stage(screen=False, look=False):
    """zu2d_pools (with look, also look_u2d) writing under U2g's declaration while a stage of Y runs; with screen the
    gate's carves are Y's eight; with look, look_u2d's build check admits y. Yields look_u2d (or None); restored after,
    also on an error."""
    LU = LUm() if look else None
    saved = (ZP.DECLARED, ZP.SCREEN, ZP.BUILDS, LU.DECLARED if LU is not None else None)
    ZP.DECLARED = DECLARED
    if screen:
        ZP.SCREEN = CARVES_Y
    if look:
        ZP.BUILDS = U2D_BUILDS + (BUILD,)
        LU.DECLARED = DECLARED
    try:
        yield LU
    finally:
        ZP.DECLARED, ZP.SCREEN, ZP.BUILDS = saved[:3]
        if LU is not None:
            LU.DECLARED = saved[3]


@contextlib.contextmanager
def on_y():
    """zu2d.on_u2d on Y's roots (looks, caches, links, KT and gate; chains under outputs/u2g/y/chains, which Y does not
    build), with P1c3's KT declaration U2g's: Y's KT records are written under it. Restored after."""
    Z = Zm()
    with Z.on_u2d(BUILD) as r:
        saved = Z.Z3.DECLARED
        Z.Z3.DECLARED = DECLARED
        try:
            yield r
        finally:
            Z.Z3.DECLARED = saved


def build_of(ds):
    if ds in OWN:
        return BUILD
    if ds in LINKED:
        return "base"
    raise SystemExit(f"zu2g: {ds} is not one of the six datasets")


@contextlib.contextmanager
def on_build(b):
    """A dataset's build: Y's roots under U2g's declaration, or U2d's base build under U2d's."""
    if b == BUILD:
        with on_y() as r:
            yield r
    elif b == "base":
        with Zm().on_u2d("base") as r:
            yield r
    else:
        raise SystemExit(f"zu2g: Y's carves come from y or base, not {b}")


# ── the stages ───────────────────────────────────────────────────────────────


def own(name, carve=None):
    if name not in OWN:
        raise SystemExit(f"zu2g: Y builds {OWN}, not {name} (U2d's base build, as filed)")
    if carve is not None and carve not in ZP.PC2.carves_for(name):
        raise SystemExit(f"zu2g: {name}'s carves are {ZP.PC2.carves_for(name)}, not {carve}")


def pools(name, carve, threads=2, limit=0, shard=None):
    """zu2d_pools.pools_stage on Y: the shard's questions on W2_S at P1c2's rule (its checks: B is step 4d's, the
    seeds are today's, P1c2's questions), parts under outputs/u2g/y/pools/parts."""
    own(name, carve)
    with y_stage():
        return ZP.pools_stage(BUILD, name, carve, threads, limit, shard)


def merge(name, new_manifest=None):
    """zu2d_pools.merge_stage on Y (when its manifest is not written), then same_as_new (when not filed)."""
    own(name)
    mp = roots_y(BUILD)["pools"] / name / "pools.json"
    if not mp.exists():
        with y_stage():
            ZP.merge_stage(BUILD, name)
    sp = mp.parent / "same_as_new.json"
    if sp.exists():
        v = json.loads(sp.read_text(encoding="utf-8")).get("verdict")
        log(f"zu2g merge {name}: same_as_new filed before: {v}")
        return 0 if v == "PASS" else 1
    return same_as_new(name, new_manifest)


def same_as_new(name, new_manifest=None, pins=None):
    """Section 3's check of Y's manifest against U2d's new one (committed, its sha256 pinned): per carve the questions,
    the carve ids, |I_q| - |B|, the mean frozen pool and today's, the entity and route shares, ALL and recall of I, the
    caps of I and the graph's description equal, and 12 x Y's mean expansion (its npz, by its sha256) equal to new's
    mean fill_want within its rounding. Writes outputs/u2g/y/pools/<D>/same_as_new.json, never over one; 0 on PASS."""
    own(name)
    pins = NEW_MANIFESTS if pins is None else pins
    yd = roots_y(BUILD)["pools"] / name
    out = yd / "same_as_new.json"
    if out.exists():
        raise SystemExit(f"zu2g: {rel(out)} exists; U2g never overwrites it")
    npath = Path(new_manifest) if new_manifest else _ORIG["roots"]("new")["pools"] / name / "pools.json"
    if not npath.is_file():
        raise SystemExit(f"zu2g: {npath} missing (U2d's new manifest of {name})")
    raw = npath.read_bytes().replace(b"\r\n", b"\n")
    nsha = hashlib.sha256(raw).hexdigest()
    if nsha != pins.get(name):
        raise SystemExit(f"zu2g: {npath} is not U2d's committed new manifest of {name} (sha256 {nsha[:12]}, pinned "
                         f"{str(pins.get(name))[:12]})")
    new = json.loads(raw.decode("utf-8"))
    ymp = yd / "pools.json"
    if not ymp.is_file():
        raise SystemExit(f"zu2g: {rel(ymp)} missing (zu2g.py merge {name})")
    y = json.loads(ymp.read_text(encoding="utf-8"))
    problems, carves = [], {}
    for who, m, want in (("Y", y, (DECLARED, BUILD, CONSTRUCTION_Y, name)),
                         ("new", new, (U2D_DECLARED, "new", NEW_CONSTRUCTION, name))):
        got = (m.get("declared_in"), m.get("build"), m.get("construction"), m.get("dataset"))
        if got != want:
            problems.append(f"{who}'s manifest is {got}, not {want}")
    keys = [*SAME_KEYS, "caps.I"]
    for cv in ZP.PC2.carves_for(name):
        key = f"{name}__{cv}"
        ye, ne = (y.get("files") or {}).get(key), (new.get("files") or {}).get(key)
        if ye is None or ne is None:
            problems.append(f"{key}: missing from {'Y' if ye is None else 'new'}'s manifest")
            continue
        if (ye.get("construction"), ye.get("rule")) != (CONSTRUCTION_Y, RULE_Y):
            problems.append(f"{key}: Y's entry is {ye.get('construction')} by {ye.get('rule')}")
        differ = [k for k in SAME_KEYS if ye.get(k) != ne.get(k)]
        if (ye.get("caps") or {}).get("I") != (ne.get("caps") or {}).get("I"):
            differ.append("caps.I")
        problems += [f"{key}: {k} differs from new's" for k in differ]
        f = yd / str(ye.get("file"))
        if not f.is_file() or ZP.sha_file(f) != ye.get("sha256"):
            problems.append(f"{key}: {f.name} is not the file Y's manifest names")
            continue
        with np.load(f) as z:
            fk = z["fill_k"].astype(np.float64)
        got = ZP.PICK_F * float(fk.mean()) if fk.size else 0.0
        want = ne.get("fill_want_mean")
        if not isinstance(want, (int, float)) or abs(got - want) > FILL_TOL + 1e-9:
            problems.append(f"{key}: 12 x Y's mean expansion is {got:.4f}, new's mean fill_want {want}")
        carves[cv] = {"equal": [k for k in keys if k not in differ], "differ": differ,
                      "fill": {"y_12x_mean_fill_k": round(got, 4), "new_fill_want_mean": want},
                      "rows_mean": {"y": (ye.get("rows") or {}).get("mean"), "new": (ne.get("rows") or {}).get("mean")},
                      "pool": {"y": ye.get("pool"), "new": ne.get("pool")}}
    rec = {"declared_in": DECLARED, "stage": "U2g", "check": "Y's pools against U2d's new pools on W2_S (section 3)",
           "dataset": name, "y_manifest_sha256": ZP.sha_file(ymp),
           "new_manifest": f"outputs/u2d/new/pools/{name}/pools.json", "new_manifest_sha256": nsha, "carves": carves,
           "problems": problems, "verdict": "PASS" if not problems else "FAIL", "zu2g_sha256": sha_src(__file__),
           "utc": utc()}
    ZP.write_json(out, rec)
    log(f"zu2g same-as-new {name}: {rec['verdict']}" + (f": {problems}" if problems else ""))
    return 0 if not problems else 1


def same_as_new_ok(name):
    """same_as_new.json's sha256 when it is U2g's PASS for the dataset; stops otherwise."""
    p = roots_y(BUILD)["pools"] / name / "same_as_new.json"
    rec = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    if (rec.get("verdict"), rec.get("declared_in"), rec.get("dataset")) != ("PASS", DECLARED, name):
        raise SystemExit(f"zu2g: {rel(p)} is {rec.get('verdict', 'missing')}, not U2g's same_as_new PASS for {name}")
    return ZP.sha_file(p)


def look(rest):
    """look_u2d's look of one of Y's carves (its checks: the graph is the manifest's, and the questions, changed and
    short are the manifest's), under U2g's declaration; only after same_as_new PASSed on the dataset."""
    if "--build" in rest:
        raise SystemExit("zu2g look: the build is y")
    if "--dataset" not in rest:
        raise SystemExit("zu2g look: --dataset D")
    name = rest[rest.index("--dataset") + 1]
    own(name)
    same_as_new_ok(name)
    with y_stage(look=True) as LU:
        return LU.main(["--build", BUILD, *rest])


def copy_bases():
    return ZP.copy_bases(BUILD)


def links(name, carve, where="laptop"):
    """zlink's build of one of Y's carves on Y's looks and caches (zu2d.build's call, on Y's roots)."""
    own(name, carve)
    Z = Zm()
    pl = Z.placement_of(where)
    with on_y() as r:
        Z.ZL.build(name, carve, r["links"], cache_root=r["cache"], look_root=r["look"], placement=pl)
    return 0


def kt(name, carve):
    own(name, carve)
    with y_stage():
        return ZP.kt_stage(BUILD, name, carve)


def gate():
    """zu2d_pools.gate over Y's eight carves, under U2g's declaration; refused (nothing written) until same_as_new has
    PASSed on all four datasets."""
    bad = []
    for d in OWN:
        try:
            same_as_new_ok(d)
        except SystemExit as e:
            bad.append(str(e))
    if bad:
        raise SystemExit("zu2g gate: same_as_new has not PASSed on every dataset; no gate is written: " + "; ".join(bad))
    with y_stage(screen=True):
        return ZP.gate(BUILD)


# ── the fits (section 4) ─────────────────────────────────────────────────────

SPLITS = ("L-musique", "L-hotpotqa")       # zu2d's (checked)
SUF = {"L-musique": "", "L-hotpotqa": "-hp"}
ARMS_Y = ("zrckp", "zspk", "zg6")          # in this order: zg6 last


def fit_name(arm, split):
    return f"scr-u2g-y-{arm}{SUF[split]}"


def u2d_fit(b, arm, seed, split):
    """zu2d.fit_name (checked in the self-test)."""
    return f"scr-u2d-{b}-{arm}{f'-s{seed}' if seed else ''}{SUF[split]}"


FITS = {fit_name(arm, sp): (arm, sp) for arm in ARMS_Y for sp in SPLITS}
assert len(FITS) == 6


def per_build(base):
    """base's carve class with each carve opened on its dataset's build: squad, musique, hotpotqa and 2wiki on Y's
    roots (the KT records under U2g's declaration), metaqa and webqsp on U2d's base build's (under U2d's). The fits pass
    Y's cache root; each carve takes its build's. Nothing else of base changes."""

    class PerBuild(base):
        def __init__(self, ds, carve, basis, device, cache_root=None, verify=True, score2=False):
            want = roots_y(BUILD)["cache"]
            if cache_root is None or Path(cache_root).resolve() != want.resolve():
                raise SystemExit(f"zu2g: the fits take --cache-root {rel(want)}, not {cache_root}")
            b = build_of(ds)
            with on_build(b) as r:
                super().__init__(ds, carve, basis, device, r["cache"], verify, score2)
            self.u2g_build = b

    PerBuild.__name__ = PerBuild.__qualname__ = f"U2G{base.__name__}"
    return PerBuild


_PER_BUILD = {}


def carve_class(arm):
    base = Zm().ARMS[arm][1]
    if base not in _PER_BUILD:
        _PER_BUILD[base] = per_build(base)
    return _PER_BUILD[base]


@contextlib.contextmanager
def as_arm(arm):
    """lean_screen's arm with its carve class opened per build while a fit trains or reads; restored after."""
    Z = Zm()
    if Z.S.ARMS.get(arm) != Z.ARMS[arm]:
        raise SystemExit(f"zu2g: the arm {arm} is {Z.S.ARMS.get(arm)}, not U2d's {Z.ARMS[arm]}")
    Z.S.ARMS[arm] = (Z.ARMS[arm][0], carve_class(arm))
    try:
        yield
    finally:
        Z.S.ARMS[arm] = Z.ARMS[arm]


def gate_y():
    g = roots_y(BUILD)["gate"]
    if not g.is_file():
        raise SystemExit(f"zu2g: {rel(g)} missing (zu2g.py gate)")
    rec = json.loads(g.read_text(encoding="utf-8"))
    got = (rec.get("verdict"), rec.get("declared_in"), rec.get("build"), rec.get("construction"))
    if got != ("PASS", DECLARED, BUILD, [CONSTRUCTION_Y]):
        raise SystemExit(f"zu2g: {rel(g)} is {got}, not U2g's PASS for Y")
    return rec


def bases_same():
    """Y's caches and base's hold the same step-1 bases: every fit reads carves of both builds, each carve normalised by
    the basis in its own cache root (lean_gpu's read stops on a carve whose basis is not the fit's). The two builds'
    bases_copied.json must name the same source and files."""
    Z = Zm()
    got = {}
    for b in (BUILD, "base"):
        p = Z.roots(b)["cache"] / "bases_copied.json"
        rec = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        if not rec.get("files"):
            raise SystemExit(f"zu2g: {rel(p)} missing or empty (copy-bases)")
        got[b] = {"source": rec.get("source"), "files": rec.get("files")}
    if got[BUILD] != got["base"]:
        raise SystemExit(f"zu2g: Y's step-1 bases are not base's: {got}")
    return got[BUILD]


def gates_passed():
    """Before any fit or read (section 3): Y's gate PASS (U2g's), base's PASS (U2d's), same_as_new PASS on all four,
    and the same step-1 bases in both builds' caches."""
    Z = Zm()
    gy, gb = gate_y(), Z.gate_passed("base")
    return {BUILD: {"gate_sha256": ZP.sha_file(roots_y(BUILD)["gate"]), "construction": gy.get("construction")},
            "base": {"gate_sha256": ZP.sha_file(Z.roots("base")["gate"]), "construction": gb.get("construction")},
            "same_as_new": {d: same_as_new_ok(d) for d in OWN}, "bases": bases_same()}


def kt_records_of(carves):
    """Each carve's KT record sha256 on its dataset's build (zp1c3.kt_records under that build's declaration)."""
    Z = Zm()
    out = {}
    for ds, cv in carves:
        b = build_of(ds)
        with on_build(b) as r:
            out[f"{ds}/{cv}"] = {"build": b, "sha256": Z.Z3.kt_records([(ds, cv)], r["kt"])[f"{ds}/{cv}"]}
    return out


def builds_detail():
    """Per build: its datasets, roots, and the chain and link records of those datasets."""
    Z = Zm()
    out = {}
    for b in (BUILD, "base"):
        mine = [d for d in ZP.SIX if build_of(d) == b]
        with on_build(b) as r:
            pick = lambda recs: {k: v for k, v in recs.items() if k.split("/")[0] in mine}  # noqa: E731
            out[b] = {"datasets": mine, "cache_root": rel(r["cache"]), "kt_root": rel(r["kt"]),
                      "links_root": rel(r["links"]), "rmatch_records": pick(Z.RM.built_records()),
                      "zrc_records": pick(Z.Z2._BUILT(r["ch_zrc"])), "zlink_records": pick(Z.ZL.built_records())}
    return out


def where_ok(where, env, what):
    Z = Zm()
    if not Z.ok_where(where, env):
        raise SystemExit(f"zu2g: U2g {what}s on Modal's {Z.GPU} from the image {Z.IMAGE_TAG} (section 4), not on "
                         f"{where} with {Z.device_of(env)}")


def train(argv, split, where="laptop", env=None):
    """One of U2g's six fits: its arm and split from its name, seed 0, U2d's card on the GPU; each carve on its dataset's
    build; the gates PASS; stamped with the device, the builds and their records."""
    Z = Zm()
    arm = Z.R.arm_of(argv)
    name, out_root = Z.S2.where(argv)
    if name not in FITS:
        raise SystemExit(f"zu2g: {name} is not one of U2g's fits (scr-u2g-y-<arm>[-hp], section 4)")
    farm, fsplit = FITS[name]
    if arm != farm or split != fsplit:
        raise SystemExit(f"zu2g: {name} is {farm}'s fit of {fsplit}, not {arm}'s of {split}")
    if tuple(Z.SPLITS) != SPLITS:
        raise SystemExit(f"zu2g: zu2d's splits are {Z.SPLITS}, not {SPLITS}")
    if Z.S.ARMS.get(arm) != Z.ARMS[arm]:
        raise SystemExit(f"zu2g: the arm {arm} is {Z.S.ARMS.get(arm)}, not U2d's {Z.ARMS[arm]}")
    bad = Z.G6.fixed_flags(argv, Z.REFUSED_TRAIN)
    if bad:
        raise SystemExit(f"zu2g: U2g trains on U2d's card (seed 0, hidden {Z.LG.HIDDEN}, config {Z.LG.CONFIG}, variant "
                         f"p, the split's fit carves and basis, threads 2, outputs/screen/fits); {bad} would change "
                         f"that")
    env = Z.device_env() if env is None else env
    where_ok(where, env, "train")
    g = gates_passed()
    kt_ = kt_records_of(Z.Z3.train_carves(split))
    with on_y(), as_arm(arm):
        Z.Z2.cache_root_of(argv)
        argv = Z.Z2.on_card(argv) + ["--seed", "0"]
        rc = Z.RM.train(argv, split)
    stamp_fit(name, out_root, env, g, kt_)
    return rc


def stamp_fit(name, out_root, env, g, kt_):
    Z = Zm()
    arm, split = FITS[name]
    sj = Path(out_root) / name / "screen.json"
    if not sj.exists():
        return
    rec = json.loads(sj.read_text(encoding="utf-8"))
    rec.update({**Z.code_shas(), "zu2g_sha256": sha_src(__file__), "u2g": {
        "declared_in": DECLARED, "fit": name, "arm": arm, "seed": 0, "split": split, "device": Z.device_of(env),
        "env": env, "builds": {d: build_of(d) for d in ZP.SIX}, "graph": {BUILD: GRAPH_Y, "base": Z.GRAPH["base"]},
        "pools": {b: g[b]["construction"] for b in (BUILD, "base")}, "rule": {BUILD: RULE_Y, "base": ZP.RULE["base"]},
        "gates": g, "head": {**Z.head_of(arm, BUILD), "kt_root_base": rel(Z.roots("base")["kt"])}, "kt_records": kt_,
        "per_build": builds_detail()}})
    Z.LC.write_json(sj, rec)


def fit_stamp(name, out_root=None):
    """A U2g fit's screen.json and its u2g record, checked against the fit's name."""
    Z = Zm()
    arm, split = FITS[name]
    sj = Path(out_root or Z.S.OUT / "fits") / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    u = rec.get("u2g") or {}
    if (rec.get("arm"), u.get("declared_in"), u.get("fit"), u.get("arm"), u.get("seed"), u.get("split")) != (
            arm, DECLARED, name, arm, 0, split):
        raise SystemExit(f"zu2g: {sj} is not U2g's {arm} fit of {split} on Y")
    return rec, u


def read(argv, where="laptop", env=None):
    """The fit read with p@swa on the six s1eval carves, each on its dataset's build, on the device it was fitted on."""
    Z = Zm()
    name, out_root = Z.S2.where(argv)
    if name not in FITS:
        raise SystemExit(f"zu2g: {name} is not one of U2g's fits")
    arm, split = FITS[name]
    bad = Z.G6.fixed_flags(argv, Z.REFUSED_READ)
    if bad:
        raise SystemExit(f"zu2g: U2g reads p's fits on the six s1eval carves, each on its dataset's build; {bad} would "
                         f"change that")
    _rec, u = fit_stamp(name, out_root)
    env = Z.device_env() if env is None else env
    where_ok(where, env, "read")
    if Z.device_of(env) != u.get("device"):
        raise SystemExit(f"zu2g: {name} was fitted on {u.get('device')}; this read runs on {Z.device_of(env)}")
    g = gates_passed()
    kt_ = kt_records_of(Z.Z3.eval_carves())
    with on_y(), as_arm(arm):
        Z.Z2.cache_root_of(argv)
        argv = Z.Z2.on_card(argv)
        rc = Z.RM.read(argv)
    rj = Path(out_root) / name / "read.json"
    if rj.exists():
        rec = json.loads(rj.read_text(encoding="utf-8"))
        rec.update({**Z.code_shas(), "zu2g_sha256": sha_src(__file__), "u2g": {
            "declared_in": DECLARED, "fit": name, "arm": arm, "seed": 0, "split": split, "device": Z.device_of(env),
            "env": env, "builds": {d: build_of(d) for d in ZP.SIX}, "gates": g, "kt_records": kt_}})
        Z.LC.write_json(rj, rec)
    return rc


# ── the compares ─────────────────────────────────────────────────────────────


def fit_device(name, out_root=None):
    """A fetched fit's device: a U2g fit's from its u2g stamps (the read on the fit's device), U2d's through zu2d."""
    Z = Zm()
    if name not in FITS:
        return Z.fit_device(name, out_root)
    _rec, u = fit_stamp(name, out_root)
    rj = Path(out_root or Z.S.OUT / "fits") / name / "read.json"
    r = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
    ru = r.get("u2g") or {}
    if ru.get("declared_in") != DECLARED or ru.get("fit") != name or ru.get("device") != u.get("device"):
        raise SystemExit(f"zu2g: {rj} is not U2g's read of {name} on the device it was fitted on")
    return u["device"]


def compares():
    """The 14 compares: Y against base for the decisions (dy, dyg), U2d's new against Y (the rule on W2_S, readings
    across workspaces) and zg6 against zspk on Y. The flag: one device required."""
    out = {}
    for sp in SPLITS:
        s = SUF[sp]
        for arm in ("zrckp", "zspk"):
            out[f"scr-u2g-dy-{arm}{s}"] = (fit_name(arm, sp), [u2d_fit("base", arm, 0, sp)], True)
        out[f"scr-u2g-dyg-zg6{s}"] = (fit_name("zg6", sp), [u2d_fit("base", "zg6", 0, sp)], True)
        for arm in ARMS_Y:
            out[f"scr-u2g-rule-{arm}{s}"] = (u2d_fit("new", arm, 0, sp), [fit_name(arm, sp)], False)
        out[f"scr-u2g-y-zg6-vs-zspk{s}"] = (fit_name("zg6", sp), [fit_name("zspk", sp)], True)
    return out


COMPARES = compares()
assert len(COMPARES) == 14


def devices_of(names, one=True, out_root=None):
    """Each fit's device and whether they are one; stops when one device is required and they differ."""
    devs = {n: fit_device(n, out_root) for n in names}
    same = len({json.dumps(d, sort_keys=True) for d in devs.values()}) == 1
    if one and not same:
        raise SystemExit(f"zu2g: the fits ran on different devices or images: {devs}")
    return devs, same


def compare_cmd(cname, new, bases, one=True, out_root=None, screen=None):
    """lean_screen's compare of fit new against bases (the first decides), the paths relative to the repository; the
    comparison stamped with U2g's record and the fits' devices."""
    Z = Zm()
    devs, same = devices_of([new, *bases], one, out_root)
    fits = Path(out_root or Z.S.OUT / "fits")
    out = Path(screen or SCREEN) / cname

    def r_(p):
        p = Path(p)
        return p.relative_to(ROOT).as_posix() if p.is_absolute() and p.is_relative_to(ROOT) else str(p)

    with Z.at_root():
        rc = Z.S2.main(["compare", "--new", r_(fits / new), "--base", ",".join(r_(fits / b) for b in bases),
                        "--out", r_(out)])
    js = out.with_suffix(".json")
    rec = json.loads(js.read_text(encoding="utf-8"))
    rec["u2g"] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases), "devices": devs,
                  "one_device": same, "zu2g_sha256": sha_src(__file__)}
    Z.LC.write_json(js, rec)
    return rc


def compare_all(only=None, out_root=None, screen=None):
    """Every compare not yet filed (a filed one with the same fits and U2g's record is kept); a compare whose fits are
    not fetched is named and skipped. 1 when one is skipped."""
    done, ran, skipped = [], [], []
    for cname, (new, bases, one) in COMPARES.items():
        if only and cname not in only:
            continue
        js = Path(screen or SCREEN) / f"{cname}.json"
        if js.exists():
            u = json.loads(js.read_text(encoding="utf-8")).get("u2g") or {}
            if u.get("declared_in") == DECLARED and u.get("new") == new and u.get("bases") == list(bases):
                done.append(cname)
                continue
        try:
            compare_cmd(cname, new, bases, one, out_root, screen)
            ran.append(cname)
        except SystemExit as e:
            skipped.append(f"{cname}: {e}")
    log(f"zu2g compare-all: {len(ran)} run, {len(done)} filed before, {len(skipped)} skipped"
        + (f" ({'; '.join(skipped)})" if skipped else ""))
    return 1 if skipped else 0


# ── the decisions (section 5) ────────────────────────────────────────────────

DECISIONS = {"Y": ("zrckp", "zspk"), "YG": ("zg6",)}
ASKS = {"Y": "W2_S at P1c2's rule, for the lean pair", "YG": "W2_S at P1c2's rule, for zg6"}
CMP = {"Y": "dy", "YG": "dyg"}
NULL_ARM = {"zrckp": "zrckp", "zspk": "zspk", "zg6": "zspk"}
RULE = ("ADOPT_FOR_FULL_RUN when no arm's re-call under U2d's seed null has a LOSS on any of its twelve reads and the "
        "arms together have at least one GAIN; NO_LOSS when no read is a LOSS and none a GAIN; NOT_ADOPTED when any read "
        "is a LOSS; INCOMPLETE when a re-call is missing, is not U2g's, does not hold twelve reads, or its fits ran on "
        "different devices (docs/U2G_GRAPH_AND_RULE.md, section 5)")
FULL = "the lean pair's full run on Y's build (W2_S, P1c2's rule), declared in its own file: six splits per arm, " \
       "under grade-nullx"
FULL_RUN = {"ADOPT_FOR_FULL_RUN": FULL, "NO_LOSS": FULL, "NOT_ADOPTED": "none: the lean pair stays on today's graph",
            "INCOMPLETE": "none until the missing re-call is run"}
ZG6_GRAPH = {"ADOPT_FOR_FULL_RUN": "W2_S at P1c2's rule (Y's build)", "NO_LOSS": "W2_S at P1c2's rule (Y's build)",
             "NOT_ADOPTED": "today's graph, as filed", "INCOMPLETE": "none until the missing re-call is run"}
NEXT = {"links": "the links: a link rule chosen by the lean pair's R@5 on this screen, not by coverage alone (U2b chose "
                 "W2_S by the coverage of B and X(k))",
        "X": "X: U2e's rule on today's graph (section 8)",
        "stop": "if new - Y also loses (the rule readings beside), the next form is the stop: U2f's learned cut, after "
                "the pools are final",
        "zg6": "zg6 stays on today's graph; G's declared next form (zg6v) stands",
        "always": "X runs when credit or the host allows",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def pair_base_of(arm, sp):
    return fit_name(arm, sp), u2d_fit("base", arm, 0, sp)


def decision_cmp(decision, arm, sp):
    return f"scr-u2g-{CMP[decision]}-{arm}{SUF[sp]}"


def against(decision, arm):
    return f"{arm} on Y (W2_S, P1c2's rule) against {arm} on U2d's base build (today's graph, P1c2's rule)"


def u2g_loader(decision, arm):
    """screen_pair.load's checks for one of a decision's compares: U2g's, its new fit the arm's Y fit and its deciding
    base U2d's base fit of the split."""
    Z = Zm()

    def load(fn):
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zu2g: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if (rec.get("u2g") or {}).get("declared_in") != DECLARED:
            raise SystemExit(f"zu2g: {p} is not U2g's compare")
        nn_ = Z.SR.parts_of(rec["new"])[-1]
        if nn_ not in FITS or FITS[nn_][0] != arm:
            raise SystemExit(f"zu2g: {p} reads {nn_}, not one of U2g's {arm} fits")
        split = FITS[nn_][1]
        want_new, want_base = pair_base_of(arm, split)
        if nn_ != want_new or Z.SR.parts_of(rec["bases"][0])[-1] != want_base:
            raise SystemExit(f"zu2g: {p} compares {nn_} with {rec['bases'][0]}, not decision {decision}'s {want_new} "
                             f"with {want_base}")
        if sorted(rec["trained_on"]) != sorted(Z.LG.FITS[split]):
            raise SystemExit(f"zu2g: {p} trained on {rec['trained_on']}, not {split}'s {sorted(Z.LG.FITS[split])}")
        if rec["candidate"] != "p@swa" or rec["carve"] != "s1eval":
            raise SystemExit(f"zu2g: {p} reads {rec['candidate']} on {rec['carve']}, not p@swa on s1eval")
        rows = rec["by_base"][rec["bases"][0]]["rows"]
        if sorted(r["dataset"] for r in rows) != sorted(Z.LG.EVAL_ORDER):
            raise SystemExit(f"zu2g: {p} reads {[r['dataset'] for r in rows]}, not the six datasets")
        return split, rows

    return load


def pair_out(decision, arm, screen=None):
    return Path(screen or SCREEN) / f"scr-u2g-{CMP[decision]}-{arm}-pair"


def recall_out(decision, arm, screen=None):
    p = pair_out(decision, arm, screen)
    return p.with_name(p.name + "-recall.json")


def restamp(out, decision, arm, kind, through=None):
    Z = Zm()
    out = Path(out)
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "U2g", "declared_in": DECLARED, "decision": decision, "asks": ASKS[decision], "arm": arm,
                    "against": against(decision, arm), "null_arm": NULL_ARM[arm],
                    "null": f"U2d's seed null ({U2D_DECLARED}, section 6)", "rebased_through": through or {},
                    "zu2g_sha256": sha_src(__file__)})
        Z.LC.write_json(js, rec)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in Z.FIX:
            t = t.replace(a, b)
        head = f"U2g ({DECLARED}), decision {decision} ({ASKS[decision]}): {against(decision, arm)}"
        if kind == "recall":
            head += f", re-called under U2d's {NULL_ARM[arm]} seed null"
            if through:
                head += " re-based through " + ", ".join(Path(x).stem for x in through.values())
        md.write_text(head + ".\n\n" + t, encoding="utf-8")


def pair(decision, arm, screens=None, out=None, screen=None):
    """screen_pair.pair over the decision's two compares of the arm (L-musique and L-hotpotqa)."""
    Z = Zm()
    screens = screens or [Path(screen or SCREEN) / f"{decision_cmp(decision, arm, sp)}.json" for sp in SPLITS]
    out = Path(out or pair_out(decision, arm, screen))
    saved = Z.SP.load
    Z.SP.load = u2g_loader(decision, arm)
    try:
        rec = Z.SP.pair(screens, out)
    finally:
        Z.SP.load = saved
    restamp(out, decision, arm, "pair")
    return rec


def recall(decision, arm, pair_file=None, out=None, nulls=None, rebases=None, screen=None):
    """screen_recall.pair of the decision's pair under U2d's seed null of the arm's family (zu2d.u2d_null), re-based
    through U2d's filed compare where the pair's base is not the null's seed-0 fit (zg6: scr-u2d-base-zg6-vs-zspk)."""
    Z = Zm()
    na = NULL_ARM[arm]
    files = nulls or Z.null_files(na, screen)
    null, sources = Z.u2d_null(na, files)
    through = {}
    for sp in SPLITS:
        _new, pb = pair_base_of(arm, sp)
        s0 = u2d_fit("base", na, 0, sp)
        if pb != s0:
            cf = Path((rebases or {}).get(sp) or Path(screen or SCREEN) / f"{Z.rebase_name(pb, s0)}.json")
            Z.rebase(null, sources, cf, pb, s0)
            through[sp] = str(cf)
    pf = Path(pair_file or pair_out(decision, arm, screen).with_suffix(".json"))
    pr = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else {}
    if pr.get("stage") != "U2g" or pr.get("decision") != decision or pr.get("arm") != arm:
        raise SystemExit(f"zu2g: {pf} is not U2g's pair of {arm} for decision {decision}")
    out = Path(out or recall_out(decision, arm, screen).with_suffix(""))
    saved = Z.SR.load_null
    Z.SR.load_null = lambda _files: (null, sources)
    try:
        rec = Z.SR.pair(files, pf, out)
    finally:
        Z.SR.load_null = saved
    restamp(out, decision, arm, "recall", through)
    return rec


def decision_fits(decision):
    """Every fit a decision's re-calls read: the pairs' fits and the null's (with the re-base's)."""
    names = set()
    for arm in DECISIONS[decision]:
        for sp in SPLITS:
            names |= set(pair_base_of(arm, sp))
            names |= {u2d_fit("base", NULL_ARM[arm], s, sp) for s in (0, 1, 2)}
    return sorted(names)


def decision_of(decision, recalls=None, devices=None, screen=None):
    """A decision's verdict from its arms' re-calls (section 5's rule); devices maps fit names to their devices (the
    fetched fits' stamps when None)."""
    Z = Zm()
    problems, rows = [], {}
    for arm in DECISIONS[decision]:
        rp = Path((recalls or {}).get(arm) or recall_out(decision, arm, screen))
        if not rp.exists():
            problems.append(f"{rp.name} missing")
            continue
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if (rec.get("stage"), rec.get("declared_in"), rec.get("decision"), rec.get("arm")) != (
                "U2g", DECLARED, decision, arm):
            problems.append(f"{rp.name} is not U2g's re-call of {arm} for decision {decision}")
        rs = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rs) != sorted(
                (sp, ds) for sp in SPLITS for ds in Z.LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rs)} reads, not the twelve of {', '.join(SPLITS)} on the six datasets")
        rows[arm] = rs
    try:
        devs = devices if devices is not None else {n: fit_device(n) for n in decision_fits(decision)}
        if len({json.dumps(d, sort_keys=True) for d in devs.values()}) > 1:
            problems.append(f"the fits ran on different devices: {devs}")
    except SystemExit as e:
        problems.append(str(e))
    gains = [f"{arm} {r['split']} {r['dataset']}" for arm, rs in rows.items() for r in rs if r.get("call") == "GAIN"]
    losses = [f"{arm} {r['split']} {r['dataset']}" for arm, rs in rows.items() for r in rs if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else "NOT_ADOPTED" if losses else "ADOPT_FOR_FULL_RUN" if gains else "NO_LOSS"
    return {"decision": decision, "asks": ASKS[decision], "verdict": v, "gains": gains, "losses": losses,
            "problems": problems, "arms": {a: against(decision, a) for a in DECISIONS[decision]},
            "calls": {f"{arm} {r['split']} {r['dataset']}": r.get("call") for arm, rs in rows.items() for r in rs},
            "delta_r5": {f"{arm} {r['split']} {r['dataset']}": r.get("delta") for arm, rs in rows.items() for r in rs}}


def next_forms(v):
    """Section 7, fixed before any number."""
    y, yg = v["Y"], v["YG"]
    out = []
    if y["verdict"] == "NOT_ADOPTED":
        hl = any(x.split()[-1] in ("hotpotqa", "2wiki") for x in y["losses"])
        out.append(f"{NEXT['links']}; then {NEXT['X']}" if hl else f"{NEXT['X']}; then {NEXT['links']}")
    elif y["verdict"] in ("ADOPT_FOR_FULL_RUN", "NO_LOSS"):
        out.append(f"{FULL}; then {NEXT['X']}; {NEXT['stop']}")
    if yg["verdict"] == "NOT_ADOPTED":
        out.append(NEXT["zg6"])
    out += [f"{d}: {NEXT['INCOMPLETE']}" for d in DECISIONS if v[d]["verdict"] == "INCOMPLETE"]
    return out + [NEXT["always"]]


# ── beside, deciding nothing (section 6) ─────────────────────────────────────


def build_of_fit(n):
    return BUILD if n in FITS else Zm().FITS[n][0]


def split_of_fit(n):
    return FITS[n][1] if n in FITS else Zm().FITS[n][3]


def fits_metrics(screen=None):
    """R@5, FC@5 and hit@1 per fit and read, and rrf per build and split, from U2g's compares and U2d's filed ones."""
    Z = Zm()
    fit, rrf0 = Z.metrics(screen)
    rrf = {f"{b} {sp}": v for (b, sp), v in rrf0.items()}
    rows = {c: Z.cmp_rows(c, screen) or [] for c in COMPARES}
    for cname, (new, _bases, _one) in COMPARES.items():          # a fit's reads where it is the compare's new fit
        for r in rows[cname]:
            fit.setdefault(new, {})[r["dataset"]] = r["new"]
            rrf.setdefault(f"{build_of_fit(new)} {split_of_fit(new)}", {})[r["dataset"]] = r["rrf"]
    for cname, (_new, bases, _one) in COMPARES.items():          # then where it is the base, if not read as new
        for r in rows[cname]:
            fit.setdefault(bases[0], {}).setdefault(r["dataset"], r["base"])
    return fit, rrf


def pools_all():
    """Each carve's questions, mean rows, ALL and recall (I and the pool) and caps on base, new (U2d's manifests) and
    Y (the fetched ones)."""
    out = Zm().pools_stats()
    for p in sorted(roots_y(BUILD)["pools"].glob("*/pools.json")):
        for k, e in json.loads(p.read_text(encoding="utf-8")).get("files", {}).items():
            out.setdefault(BUILD, {})[k] = {"questions": e.get("questions"), "rows_mean": (e.get("rows") or {}).get("mean"),
                                            "I": e.get("I"), "pool": e.get("pool"), "caps": e.get("caps")}
    return out


def rule_readings(screen=None):
    """U2d's new fits against Y's (the rule on W2_S), each read called under U2d's null of the arm's family: readings,
    not decisions (the fits come from different workspaces)."""
    Z = Zm()
    floors, out = {}, {}
    for na in sorted(set(NULL_ARM.values())):
        try:
            null, _src = Z.u2d_null(na, Z.null_files(na, screen))
            floors[na] = {(sp, ds): v["floor"] for sp in SPLITS for ds, v in null[sp].items()}
        except SystemExit as e:
            floors[na] = str(e)
    for arm in ARMS_Y:
        fl = floors[NULL_ARM[arm]]
        for sp in SPLITS:
            for r in Z.cmp_rows(f"scr-u2g-rule-{arm}{SUF[sp]}", screen) or []:
                f = fl.get((sp, r["dataset"])) if isinstance(fl, dict) else None
                out[f"{arm} {sp} {r['dataset']}"] = {
                    "delta": r["delta"][0], "lo": r["lo"][0], "hi": r["hi"][0], "call": r["call"],
                    "floor": round(f, 4) if f is not None else None,
                    "reading": Z.SR.recall(r["call"], r["delta"][0], f) if f is not None else None}
    return out, {k: v if isinstance(v, str) else "U2d's null" for k, v in floors.items()}


def decomposition(screen=None):
    """new - base = (Y - base) + (new - Y) on R@5 per read, against U2d's own new - base (A for the lean pair, G for
    zg6)."""
    Z = Zm()
    out = {}
    for arm in ARMS_Y:
        for sp in SPLITS:
            s = SUF[sp]
            got = []
            for cname in (f"scr-u2g-{'dyg' if arm == 'zg6' else 'dy'}-{arm}{s}", f"scr-u2g-rule-{arm}{s}",
                          f"scr-u2d-{'g' if arm == 'zg6' else 'a'}-{arm}{s}"):
                got.append({r["dataset"]: r["delta"][0] for r in Z.cmp_rows(cname, screen) or []})
            for ds in Z.LG.EVAL_ORDER:
                a, b, c = (x.get(ds) for x in got)
                out[f"{arm} {sp} {ds}"] = {"y_minus_base": a, "new_minus_y": b,
                                           "sum": round(a + b, 4) if a is not None and b is not None else None,
                                           "u2d_new_minus_base": c}
    return out


def beside(screen=None):
    Z = Zm()
    fit, rrf = fits_metrics(screen)
    caps = pools_all()
    share, head = {}, {}
    for sp in SPLITS:
        for ds in Z.LG.EVAL_ORDER:
            m = (fit.get(fit_name("zrckp", sp)) or {}).get(ds)
            rr = (rrf.get(f"{BUILD} {sp}") or {}).get(ds)
            for gnn in ("zspk", "zg6"):
                gg = (fit.get(fit_name(gnn, sp)) or {}).get(ds)
                share[f"y {sp} {ds} zrckp/{gnn}"] = Z.G6.mlp_share(m and m[0], gg and gg[0], rr and rr[0])
    for ds in LINKED:
        cv = (caps.get("base") or {}).get(f"{ds}__s1eval") or {}
        row = {"hit1_cap": ((cv.get("caps") or {}).get("pool") or {}).get("hit1"),
               "r5_cap": ((cv.get("caps") or {}).get("pool") or {}).get("r5")}
        for sp in SPLITS:
            for arm in ARMS_Y:
                x = (fit.get(fit_name(arm, sp)) or {}).get(ds)
                row[f"{arm} {sp}"] = {"r5": x[0], "hit1": x[2]} if x else None
        head[f"y {ds}"] = row
    readings, floors = rule_readings(screen)
    return {"metrics": fit, "rrf": rrf, "mlp_share": share, "share_min": Z.G6.SHARE_MIN, "headline_hit1": head,
            "pools": caps, "zg6_vs_zspk": {sp: Z.cmp_rows(f"scr-u2g-y-zg6-vs-zspk{SUF[sp]}", screen) for sp in SPLITS},
            "rule_readings": readings, "rule_floors": floors, "decomposition": decomposition(screen)}


def decide(screen=None, out=None):
    """The pairs and re-calls of both decisions' arms, the decisions, what follows and the beside report:
    outputs/u2g/screen-verdict.json and .md."""
    Z = Zm()
    fmt = Z.fmt
    ran = []
    for d, arms in DECISIONS.items():
        for arm in arms:
            try:
                pair(d, arm, screen=screen)
                recall(d, arm, screen=screen)
                ran.append(f"{d} {arm}")
            except SystemExit as e:
                log(f"zu2g decide {d} {arm}: {e}")
    v = {d: decision_of(d, screen=screen) for d in DECISIONS}
    verdicts = {d: v[d]["verdict"] for d in v}
    nxt = next_forms(v)
    bs = beside(screen)
    rec = {"stage": "U2g", "declared_in": DECLARED, "rule": RULE, "verdicts": verdicts,
           "full_run": FULL_RUN[verdicts["Y"]], "zg6_graph": ZG6_GRAPH[verdicts["YG"]], "next_forms": nxt,
           "decisions": v, "recalled": ran, "beside": bs, "zu2g_sha256": sha_src(__file__), "utc": utc()}
    md = [f"# U2g's screen verdict: Y **{verdicts['Y']}**, YG **{verdicts['YG']}**", "",
          f"The lean pair's full run: {FULL_RUN[verdicts['Y']]}. zg6's graph: {ZG6_GRAPH[verdicts['YG']]}.", "",
          f"Rule: {RULE}.", ""]
    for d in DECISIONS:
        x = v[d]
        md.append(f"- **{d}** ({ASKS[d]}; " + "; ".join(x["arms"].values()) + f"): **{x['verdict']}**, "
                  f"{len(x['gains'])} GAIN ({', '.join(x['gains']) or 'none'}), {len(x['losses'])} LOSS "
                  f"({', '.join(x['losses']) or 'none'})" + (f"; problems: {'; '.join(x['problems'])}"
                                                             if x["problems"] else "") + ".")
    md += ["", "What follows (section 7, fixed before any number):", ""] + [f"- {x}" for x in nxt]
    md += ["", "## Beside, deciding nothing", "",
           "R@5 / FC@5 / hit@1 per fit (seed 0) and read; rrf from each build's reads (Y's metaqa and webqsp carves are "
           "the base build's):", "", "| build | split | dataset | rrf | zrckp | zspk | zg6 |", "|---|---|---|---|---|---|---|"]
    for b in ("base", BUILD, "new"):
        for sp in SPLITS:
            for ds in Z.LG.EVAL_ORDER:
                cells = []
                for arm in ARMS_Y:
                    n = fit_name(arm, sp) if b == BUILD else u2d_fit(b, arm, 0, sp)
                    x = (bs["metrics"].get(n) or {}).get(ds)
                    cells.append(" / ".join(fmt(y) for y in x) if x else "")
                rr = (bs["rrf"].get(f"{b} {sp}") or {}).get(ds)
                md.append(f"| {b} | {sp} | {ds} | {' / '.join(fmt(y) for y in rr) if rr else ''} | " + " | ".join(cells)
                          + " |")
    md += ["", "U2d's A split along Y: (Y - base) + (new - Y) against U2d's new - base, R@5 (new - Y crosses "
               "workspaces):", "", "| arm | split | dataset | Y - base | new - Y | sum | U2d new - base |",
           "|---|---|---|---:|---:|---:|---:|"]
    for k, x in bs["decomposition"].items():
        arm, sp, ds = k.split()
        md.append(f"| {arm} | {sp} | {ds} | {fmt(x['y_minus_base'])} | {fmt(x['new_minus_y'])} | {fmt(x['sum'])} | "
                  f"{fmt(x['u2d_new_minus_base'])} |")
    md += ["", "The rule on W2_S: U2d's new fits against Y's, read under U2d's null (readings, not decisions):", "",
           "| arm | split | dataset | delta R@5 [95% CI] | floor | call at 0.0075 | reading |", "|---|---|---|---|---:|---|---|"]
    for k, x in bs["rule_readings"].items():
        arm, sp, ds = k.split()
        md.append(f"| {arm} | {sp} | {ds} | {x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}] | {fmt(x['floor'])} | "
                  f"{x['call']} | {x['reading']} |")
    md += ["", f"The MLP share on Y, (zrckp - rrf) / (GNN - rrf) on R@5, where the GNN's gain over rrf exceeds "
               f"{Z.G6.SHARE_MIN}:", "", "| read | share |", "|---|---:|"]
    md += [f"| {k} | {fmt(x)} |" for k, x in bs["mlp_share"].items()]
    md += ["", "metaqa and webqsp: hit@1 is the headline; R@5 against its cap (the base build's s1eval pools):", "",
           "| dataset | hit@1 cap | R@5 cap | fits (R@5, hit@1) |", "|---|---:|---:|---|"]
    for k, row in bs["headline_hit1"].items():
        fits_ = "; ".join(f"{a}: {fmt(x['r5'])}, {fmt(x['hit1'])}" for a, x in row.items()
                          if a not in ("hit1_cap", "r5_cap") and x)
        md.append(f"| {k} | {fmt(row['hit1_cap'])} | {fmt(row['r5_cap'])} | {fits_} |")
    md += ["", "Each carve's pool on base, Y and new (questions, mean rows, ALL and recall, caps):", "",
           "| build | carve | questions | rows | ALL I | ALL pool | recall pool | R@5 cap | hit@1 cap |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for b in ("base", BUILD, "new"):
        for k, c in (bs["pools"].get(b) or {}).items():
            cp = (c.get("caps") or {}).get("pool") or {}
            md.append(f"| {b} | {k} | {c.get('questions')} | {c.get('rows_mean')} | {fmt((c.get('I') or {}).get('ALL'))} | "
                      f"{fmt((c.get('pool') or {}).get('ALL'))} | {fmt((c.get('pool') or {}).get('recall'))} | "
                      f"{fmt(cp.get('r5'))} | {fmt(cp.get('hit1'))} |")
    md += ["", "zg6 against zspk on Y (delta R@5 [95% CI], FC@5, hit@1, call at 0.0075):", ""]
    for sp, rows in bs["zg6_vs_zspk"].items():
        if not rows:
            md.append(f"- {sp}: not compared")
            continue
        md.append(f"- {sp}: " + "; ".join(f"{r['dataset']} {r['delta'][0]:+.4f} [{r['lo'][0]:+.4f}, {r['hi'][0]:+.4f}] "
                                          f"{r['delta'][1]:+.4f} {r['delta'][2]:+.4f} {r['call']}" for r in rows))
    out = Path(out or OUT / "screen-verdict")
    out.parent.mkdir(parents=True, exist_ok=True)
    Z.LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zu2g decide: {verdicts}; full run: {FULL_RUN[verdicts['Y']]}; zg6: {ZG6_GRAPH[verdicts['YG']]}")
    return rec


# ── selftest (toy data only; run it in its own process) ──────────────────────


def stops(fn, *args, has=None, **kw):
    """fn(*args) must stop with SystemExit, its message holding has."""
    try:
        fn(*args, **kw)
    except SystemExit as e:
        assert has is None or has in str(e), (has, str(e))
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)}{args} did not stop")


@contextlib.contextmanager
def patched_globals(**kw):
    g = globals()
    saved = {k: g[k] for k in kw}
    g.update(kw)
    try:
        yield
    finally:
        g.update(saved)


def toy_manifests(T, name, cases=None):
    """A toy Y manifest (with its npz files under Y's pools root) and a toy new manifest in their files' form; returns
    (new manifest path, its pins)."""
    yd = roots_y(BUILD)["pools"] / name
    yd.mkdir(parents=True, exist_ok=True)
    y = {"declared_in": DECLARED, "stage": "U2d", "build": BUILD, "dataset": name, "construction": CONSTRUCTION_Y,
         "files": {}}
    new = {"declared_in": U2D_DECLARED, "stage": "U2d", "build": "new", "dataset": name,
           "construction": NEW_CONSTRUCTION, "files": {}}
    for i, cv in enumerate(ZP.PC2.carves_for(name)):
        fk = np.asarray([0, 3, 7, 5], np.int32) + i
        f = yd / f"{name}__{cv}.npz"
        np.savez_compressed(f, fill_k=fk)
        same = {"questions": 4, "carve_ids_sha256": f"ids{i}", "B_q": {"mean": 3.75, "max": 7 + i}, "pool_mean": 9.0,
                "today_inc_mean": 8.0, "shares": {"entity": 0.5, "route": 0.25, "entity_route": 0.25},
                "I": {"ALL": 0.75, "recall": 0.8}, "graph_detail": {"graph": "W2_S", "npz": "toy"}}
        capi = {"r5": 0.7, "hit1": 0.9}
        y["files"][f"{name}__{cv}"] = {"file": f.name, "sha256": ZP.sha_file(f), "construction": CONSTRUCTION_Y,
                                       "rule": RULE_Y, **same, "caps": {"I": dict(capi), "pool": {"r5": 0.71}},
                                       "rows": {"mean": 12.0}, "pool": {"ALL": 0.76}, "fill_want_mean": 3.9}
        new["files"][f"{name}__{cv}"] = {"file": f.name, "construction": NEW_CONSTRUCTION, **json.loads(json.dumps(same)),
                                         "caps": {"I": dict(capi), "pool": {"r5": 0.72}}, "rows": {"mean": 60.0},
                                         "pool": {"ALL": 0.78}, "fill_want_mean": round(12 * float(fk.mean()), 1)}
    for fn in (cases or ()):
        fn(y, new, yd)
    ZP.write_json(yd / "pools.json", y)
    npath = T / f"new_{name}_pools.json"
    lf = json.dumps(new, indent=1).encode("utf-8")
    npath.write_bytes(lf.replace(b"\n", b"\r\n"))         # as a CRLF checkout holds it; pinned on the LF bytes
    return npath, {name: hashlib.sha256(lf).hexdigest()}


def fake_cmp_y(td, cname, new, base, deltas, base_r5=0.6, stamp=True):
    """A compare in lean_screen's form of fit new against fit base, new's screen.json stamped (U2g's or U2d's fit)."""
    Z = Zm()
    td = Path(td)
    if new in FITS:
        arm, split = FITS[new]
        key, u = "u2g", {"declared_in": DECLARED, "fit": new, "arm": arm, "seed": 0, "split": split,
                         "device": Z.device_of(Z.TOY_ENV)}
    else:
        b, arm, seed, split = Z.FITS[new]
        key, u = "u2d", {"declared_in": U2D_DECLARED, "fit": new, "build": b, "arm": arm, "seed": seed,
                         "split": split, "device": Z.device_of(Z.TOY_ENV)}
    fit = td / "fits" / new
    fit.mkdir(parents=True, exist_ok=True)
    (fit / "screen.json").write_text(json.dumps({"arm": arm, "split": split, "seed": 0, key: u}), encoding="utf-8")
    rows = []
    for ds in Z.LG.EVAL_ORDER:
        d = deltas.get(ds, 0.0)
        lo, hi = round(d - 0.004, 4), round(d + 0.004, 4)
        rows.append({"dataset": ds, "read": "in-domain" if ds in Z.LG.FITS[split] else "zero-shot", "questions": 100,
                     "base": [base_r5, 0.5, 0.4], "new": [round(base_r5 + d, 4), 0.5, 0.4], "rrf": [0.4, 0.3, 0.2],
                     "delta": [d, 0, 0], "lo": [lo, 0, 0], "hi": [hi, 0, 0], "call": Z.S.call(d, lo, hi)})
    bp = f"outputs/screen/fits/{base}"
    v = Z.S.verdict([r["call"] for r in rows])
    rec = {"new": str(fit), "bases": [bp], "candidate": "p@swa", "carve": "s1eval",
           "trained_on": sorted(Z.LG.FITS[split]), "by_base": {bp: {"rows": rows, "verdict": v, "decides": True}},
           "verdict": v}
    if stamp:
        rec["u2g"] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": [base]}
    p = td / f"{cname}.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def fake_recall_y(path, decision, arm, calls, over=None):
    Z = Zm()
    rows = [{"split": sp, "dataset": ds, "read": "x", "base": 0.6, "new": 0.61, "rrf": 0.4, "delta": 0.01,
             "call": calls.get((sp, ds), "WITHIN")} for sp in SPLITS for ds in Z.LG.EVAL_ORDER]
    rec = {"stage": "U2g", "declared_in": DECLARED, "decision": decision, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over or {})
    Z.LC.write_json(path, rec)
    return path


def selftest():
    import shutil
    import tempfile
    from types import SimpleNamespace
    t0 = time.time()
    # 1. the registration: base and new pass through unchanged; Y's roots, construction and rule; BUILDS untouched
    for b in U2D_BUILDS:
        assert ZP.roots(b) == _ORIG["roots"](b) and ZP.construction_of(b) == _ORIG["construction_of"](b)
    assert ZP.construction_of("base") == "U2D:base:KIND_T" and ZP.construction_of("new") == NEW_CONSTRUCTION
    assert ZP.construction_of(BUILD) == CONSTRUCTION_Y == "U2G:y:KIND_T" and ZP.RULE[BUILD] == "KIND_T"
    ry = ZP.roots(BUILD)
    assert ry["pools"] == ROOT / "outputs" / "u2g" / "y" / "pools" and ry["gate"] == OUT / "y" / "gate.json"
    assert ry["parts"] == ry["pools"] / "parts" and set(ry) == set(_ORIG["roots"]("base"))
    assert ZP.BUILDS == U2D_BUILDS and ZP.DECLARED == U2D_DECLARED and ZP.open_carve is open_carve_y
    stops(ZP.roots, "old", has="--build")
    assert len(CARVES_Y) == 8 and {d for d, _c in CARVES_Y} == set(OWN)
    # 2. the stages' declaration, carves and builds, restored after (also on an error); zu2d and look_u2d are imported
    #    only outside a stage
    with y_stage():
        assert ZP.DECLARED == DECLARED and ZP.SCREEN == ZP.PC2.SCREEN and ZP.BUILDS == U2D_BUILDS
        stops(_lazy, "zu2d_h1", has="inside a stage of Y")
    with y_stage(screen=True):
        assert ZP.SCREEN == CARVES_Y
    try:
        with y_stage(screen=True):
            raise KeyError("inside")
    except KeyError:
        pass
    assert (ZP.DECLARED, ZP.SCREEN, ZP.BUILDS) == (U2D_DECLARED, ZP.PC2.SCREEN, U2D_BUILDS)
    with y_stage(look=True) as LU:
        assert LU.DECLARED == DECLARED and ZP.BUILDS == ("base", "new", BUILD) and ZP.DECLARED == DECLARED
        stops(LUm, has="not U2d's")
    assert LU.DECLARED == U2D_DECLARED and ZP.BUILDS == U2D_BUILDS
    Z = Zm()
    assert "zu2d_h1" not in sys.modules and len(Z.COMPARES) == 26 and LUm() is LU
    # 3. the carve opener: metaqa and webqsp refused; Y opens new's carve and sets P1c2's branch; base and new pass
    calls = []

    def fake_open(build, name, carve, threads):
        calls.append((build, name, carve, threads))
        return SimpleNamespace(build=build, nodes="dense", name=name)

    saved_open = _ORIG["open_carve"]
    _ORIG["open_carve"] = fake_open
    try:
        for ds in LINKED:
            stops(ZP.open_carve, BUILD, ds, "s1eval", 2, has="U2d's base build")
        assert not calls
        cx = ZP.open_carve(BUILD, "2wiki", "fit", 2)
        assert calls == [("new", "2wiki", "fit", 2)] and cx.build == "base" and cx.nodes is None
        assert ZP.open_carve("new", "musique", "s1eval", 2).build == "new" and calls[-1][0] == "new"
        assert ZP.open_carve("base", "metaqa", "fit", 2).build == "base" and calls[-1][0] == "base"
    finally:
        _ORIG["open_carve"] = saved_open
    # 4. P1c2's rule is new's fill; new's budget is 12 x |fill|
    full = np.arange(100, 130, dtype=np.int64)
    for inc, bs in ((40, 10), (10, 40), (35, 35), (80, 10)):
        exp, want = ZP.base_expansion(full, inc, bs)
        assert want == max(0, inc - bs) and np.array_equal(exp, full[:max(0, inc - bs)])
    for f in (0, 1, 7, 1988):
        assert int(ZP.U2E.budget_k("ADA", (ZP.PICK_F, ZP.PICK_K0), [f])[0]) == 12 * f
    # 5. same_as_new: a PASS (the new manifest checked out with CRLF), each refusal and each failure, never overwritten
    tmp = Path(tempfile.mkdtemp(prefix="zu2g_"))
    try:
        def case(i, *muts, pins=None, name="hotpotqa"):
            with patched_globals(OUT=tmp / f"c{i}"):
                npath, p = toy_manifests(tmp / f"c{i}", name, muts)
                rc = same_as_new(name, npath, p if pins is None else pins)
                rec = json.loads((roots_y(BUILD)["pools"] / name / "same_as_new.json").read_text(encoding="utf-8"))
                return rc, rec

        rc, rec = case(0)
        assert rc == 0 and rec["verdict"] == "PASS" and not rec["problems"], rec["problems"]
        assert rec["carves"]["fit"]["differ"] == [] and rec["carves"]["s1eval"]["fill"] == {
            "y_12x_mean_fill_k": 57.0, "new_fill_want_mean": 57.0}         # 12 x mean(1, 4, 8, 6)
        with patched_globals(OUT=tmp / "c0"):
            stops(same_as_new, "hotpotqa", tmp / "c0" / "new_hotpotqa_pools.json", has="never overwrites")
            assert merge("hotpotqa") == 0                       # filed before: reported, not run again

        def yset(cv, k, v):
            return lambda y, new, yd: y["files"][f"hotpotqa__{cv}"].__setitem__(k, v)

        def nset(cv, k, v):
            return lambda y, new, yd: new["files"][f"hotpotqa__{cv}"].__setitem__(k, v)

        for i, (mut, has) in enumerate((
                (yset("fit", "B_q", {"mean": 3.8, "max": 7}), "B_q differs"),
                (yset("s1eval", "graph_detail", {"graph": "today's"}), "graph_detail differs"),
                (yset("fit", "shares", {"entity": 0.4, "route": 0.25, "entity_route": 0.25}), "shares differs"),
                (yset("fit", "caps", {"I": {"r5": 0.69, "hit1": 0.9}}), "caps.I differs"),
                (yset("s1eval", "today_inc_mean", 8.1), "today_inc_mean differs"),
                (yset("fit", "questions", 5), "questions differs"),
                (nset("fit", "fill_want_mean", 45.06), "mean expansion"),
                (yset("fit", "construction", NEW_CONSTRUCTION), "Y's entry"),
                (lambda y, new, yd: y.__setitem__("declared_in", U2D_DECLARED), "Y's manifest"),
                (lambda y, new, yd: new.__setitem__("construction", "U2D:base:KIND_T"), "new's manifest"),
                (lambda y, new, yd: y["files"].pop("hotpotqa__s1eval"), "missing from Y"),
                (lambda y, new, yd: np.savez_compressed(yd / "hotpotqa__fit.npz", fill_k=np.zeros(4, np.int32)),
                 "is not the file"))):
            rc, rec = case(i + 1, mut)
            assert rc == 1 and rec["verdict"] == "FAIL" and any(has in p for p in rec["problems"]), (has, rec["problems"])
        rc, _rec = case(20, nset("fit", "fill_want_mean", 45.04))       # 12 x 3.75 = 45.0, within 0.05
        assert rc == 0
        with patched_globals(OUT=tmp / "c21"):
            npath, _p = toy_manifests(tmp / "c21", "hotpotqa")
            stops(same_as_new, "hotpotqa", npath, {"hotpotqa": "0" * 64}, has="not U2d's committed new manifest")
            stops(same_as_new, "metaqa", npath, has="Y builds")
            assert not (roots_y(BUILD)["pools"] / "hotpotqa" / "same_as_new.json").exists()
            # the gate is refused, writing nothing, until same_as_new has PASSed on all four datasets
            stops(gate, has="same_as_new has not PASSed")
            assert same_as_new("hotpotqa", npath, _p) == 0
            stops(gate, has="same_as_new has not PASSed")
            assert not roots_y(BUILD)["gate"].exists()
            stops(look, ["--dataset", "musique", "--carve", "fit", "--full"], has="same_as_new PASS")
            stops(look, ["--build", "y", "--dataset", "hotpotqa"], has="the build is y")
            stops(pools, "webqsp", "s1eval", has="Y builds")
            stops(kt, "squad", "s1fit", has="carves are")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. each carve on its dataset's build: Y's roots and U2g's declaration, or base's and U2d's; restored after,
    #    also nested in on_y; another cache root stops
    filed = ({k: getattr(Z.Z2, k) for k in Z.Z2_ROOTS}, Z.Z3.KTChainCarveZRC.KT_ROOT, Z.Z3.KTLinkCarveBase.KT_ROOT,
             Z.Z3.DECLARED, Z.RM.ChainCarveBase.CH_ROOT, Z.ZC.ChainCarveZRC.CH_ROOT, Z.ZL.LinkCarveBase.LK_ROOT,
             Z.RM.built_records, Z.ZL.built_records, {a: Z.S.ARMS[a] for a in ARMS_Y})

    def now():
        return ({k: getattr(Z.Z2, k) for k in Z.Z2_ROOTS}, Z.Z3.KTChainCarveZRC.KT_ROOT, Z.Z3.KTLinkCarveBase.KT_ROOT,
                Z.Z3.DECLARED, Z.RM.ChainCarveBase.CH_ROOT, Z.ZC.ChainCarveZRC.CH_ROOT, Z.ZL.LinkCarveBase.LK_ROOT,
                Z.RM.built_records, Z.ZL.built_records, {a: Z.S.ARMS[a] for a in ARMS_Y})

    seen = []

    class Probe(Z.Z3.KTLinkCarveBase):
        def __init__(self, ds, carve, basis, device, cache_root=None, verify=True, score2=False):  # noqa: D401
            seen.append({"ds": ds, "kt": type(self).KT_ROOT, "lk": type(self).LK_ROOT, "ch": type(self).CH_ROOT,
                         "cache": Z.Z2.CACHE, "declared": Z.Z3.DECLARED, "cache_root": Path(cache_root),
                         "verify": verify, "score2": score2})

    P = per_build(Probe)
    yr, br = Z.roots(BUILD), Z.roots("base")
    assert yr["kt"] == OUT / "y" / "kt" and yr["links"] == OUT / "y" / "links" and br["kt"] == ZP.OUT / "base" / "kt"
    c = P("squad", "fit", "2wiki", "cpu", roots_y(BUILD)["cache"], True, True)
    assert c.u2g_build == BUILD and seen[-1] == {"ds": "squad", "kt": yr["kt"], "lk": yr["links"], "ch": yr["ch_rm"],
                                                 "cache": yr["cache"], "declared": DECLARED,
                                                 "cache_root": yr["cache"], "verify": True, "score2": True}
    c = P("webqsp", "s1eval", "2wiki", "cpu", str(roots_y(BUILD)["cache"]))
    assert c.u2g_build == "base" and seen[-1]["kt"] == br["kt"] and seen[-1]["lk"] == br["links"]
    assert seen[-1]["ch"] == br["ch_rm"] and seen[-1]["cache"] == br["cache"] == seen[-1]["cache_root"]
    assert seen[-1]["declared"] == U2D_DECLARED
    assert now() == filed, "the filed roots are not restored"
    with on_y() as r:
        assert r == yr and Z.Z3.DECLARED == DECLARED and Z.Z2.CACHE == yr["cache"] and Z.Z2.GATE == yr["gate"]
        assert Z.ZL.LinkCarveBase.LK_ROOT == yr["links"] and Z.Z3.KTChainCarveZRC.KT_ROOT == yr["kt"]
        P("metaqa", "fit", "2wiki", "cpu", yr["cache"])
        assert seen[-1]["declared"] == U2D_DECLARED and seen[-1]["kt"] == br["kt"]
        assert Z.Z3.DECLARED == DECLARED and Z.Z2.CACHE == yr["cache"] and Z.Z3.KTLinkCarveBase.KT_ROOT == yr["kt"]
        assert Z.RM.built_records() == {}                      # Y builds no chains
    assert now() == filed
    stops(P, "squad", "fit", "2wiki", "cpu", br["cache"], has="--cache-root")
    stops(P, "squad", "fit", "2wiki", "cpu", None, has="--cache-root")
    try:
        with on_y():
            raise KeyError("inside")
    except KeyError:
        pass
    assert now() == filed, "restored on an error"
    # the arms: each arm's carve class per build while it runs, U2d's after; one class for zspk and zg6
    for arm in ARMS_Y:
        with as_arm(arm):
            m, cl = Z.S.ARMS[arm]
            assert m is Z.ARMS[arm][0] and cl.__mro__[1] is Z.ARMS[arm][1] and cl.__name__.startswith("U2G")
            assert "KT_ROOT" not in vars(cl) and "LK_ROOT" not in vars(cl) and "CH_ROOT" not in vars(cl)
        assert Z.S.ARMS[arm] == Z.ARMS[arm]
    assert carve_class("zspk") is carve_class("zg6") is not carve_class("zrckp")
    Z.S.ARMS["zspk"] = (None, None)
    try:
        stops(lambda: as_arm("zspk").__enter__(), has="not U2d's")
    finally:
        Z.S.ARMS["zspk"] = Z.ARMS["zspk"]
    assert now() == filed
    # the step-1 bases: Y's copy record must be base's (a fit reads carves of both builds)
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        saved_roots, bt = _ORIG["roots"], T / "base"
        _ORIG["roots"] = lambda b: ({"pools": bt / "pools", "parts": bt / "pools" / "parts", "look": bt / "look",
                                     "cache": bt / "cache", "kt": bt / "kt", "gate": bt / "gate.json",
                                     "check": bt / "check"} if b == "base" else saved_roots(b))
        try:
            with patched_globals(OUT=T / "u2g"):
                stops(bases_same, has="missing or empty")
                rec = {"source": "outputs/step1/cache", "files": {"basis_2wiki.npz": "a" * 64}, "utc": "x"}
                for b in (BUILD, "base"):
                    ZP.write_json(Z.roots(b)["cache"] / "bases_copied.json", {**rec, "utc": b})
                assert bases_same() == {"source": rec["source"], "files": rec["files"]}
                ZP.write_json(roots_y(BUILD)["cache"] / "bases_copied.json",
                              {**rec, "files": {"basis_2wiki.npz": "b" * 64}})
                stops(bases_same, has="not base's")
        finally:
            _ORIG["roots"] = saved_roots
    assert ZP.roots("base") == _ORIG["roots"]("base") and Z.roots("base")["cache"] == ZP.OUT / "base" / "cache"
    # 7. the fits and compares: six fits, fourteen compares, U2d's fit names, splits and nulls; every U2d fit named is one
    #    of U2d's
    assert SPLITS == tuple(Z.SPLITS) and SUF == Z.SUF
    assert all(u2d_fit(*f) == n for n, f in Z.FITS.items())
    assert list(FITS) == ["scr-u2g-y-zrckp", "scr-u2g-y-zrckp-hp", "scr-u2g-y-zspk", "scr-u2g-y-zspk-hp",
                          "scr-u2g-y-zg6", "scr-u2g-y-zg6-hp"]
    named = {n for new, bs, _o in COMPARES.values() for n in [new, *bs]} | {n for d in DECISIONS
                                                                           for n in decision_fits(d)}
    assert {n for n in named if n not in FITS} <= set(Z.FITS) and set(FITS) <= named
    assert all(NULL_ARM[a] == Z.NULL_ARM[a] for a in ARMS_Y)
    assert sum(1 for _n, _b, one in COMPARES.values() if not one) == 6
    assert COMPARES["scr-u2g-dyg-zg6-hp"] == ("scr-u2g-y-zg6-hp", ["scr-u2d-base-zg6-hp"], True)
    assert COMPARES["scr-u2g-rule-zspk"] == ("scr-u2d-new-zspk", ["scr-u2g-y-zspk"], False)
    assert {"scr-u2d-base-zspk-s1-hp", "scr-u2d-base-zg6-hp"} <= set(decision_fits("YG"))
    # 8. the decisions on toy re-calls: ADOPT_FOR_FULL_RUN, NO_LOSS, NOT_ADOPTED and each INCOMPLETE
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        one = {n: Z.device_of(Z.TOY_ENV) for n in decision_fits("Y") + decision_fits("YG")}
        G_, L_ = "GAIN", "LOSS"
        mk = lambda d, a, calls_, **o: fake_recall_y(T / f"rc-{d}-{a}.json", d, a, calls_, o)  # noqa: E731
        assert decision_of("Y", {"zrckp": mk("Y", "zrckp", {("L-musique", "musique"): G_}),
                                 "zspk": mk("Y", "zspk", {})}, one)["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert decision_of("Y", {"zrckp": mk("Y", "zrckp", {}), "zspk": mk("Y", "zspk", {})}, one)["verdict"] == \
            "NO_LOSS"
        nl = decision_of("Y", {"zrckp": mk("Y", "zrckp", {("L-musique", "musique"): G_}),
                               "zspk": mk("Y", "zspk", {("L-hotpotqa", "2wiki"): L_})}, one)
        assert nl["verdict"] == "NOT_ADOPTED" and nl["losses"] == ["zspk L-hotpotqa 2wiki"]
        assert decision_of("YG", {"zg6": T / "none.json"}, one)["verdict"] == "INCOMPLETE"
        assert decision_of("YG", {"zg6": mk("YG", "zg6", {}, stage="U2d")}, one)["verdict"] == "INCOMPLETE"
        assert decision_of("YG", {"zg6": mk("YG", "zg6", {}, decision="Y")}, one)["verdict"] == "INCOMPLETE"
        assert decision_of("YG", {"zg6": mk("YG", "zg6", {}, declared_in=U2D_DECLARED)}, one)["verdict"] == "INCOMPLETE"
        assert decision_of("Y", {"zrckp": mk("Y", "zrckp", {}), "zspk": mk("Y", "zspk", {}, rows=[])}, one)[
            "verdict"] == "INCOMPLETE"
        two = {**one, u2d_fit("base", "zspk", 2, "L-hotpotqa"): Z.device_of({**Z.TOY_ENV, "gpu": "NVIDIA A10G"})}
        assert decision_of("YG", {"zg6": mk("YG", "zg6", {})}, two)["verdict"] == "INCOMPLETE"
        assert decision_of("YG", {"zg6": mk("YG", "zg6", {})}, one)["verdict"] == "NO_LOSS"
        verdict = lambda vv, losses=(): {"verdict": vv, "losses": list(losses)}  # noqa: E731
        nf = next_forms({"Y": verdict("NOT_ADOPTED", ["zspk L-hotpotqa 2wiki"]), "YG": verdict("NOT_ADOPTED")})
        assert nf[0].startswith(NEXT["links"]) and nf[1] == NEXT["zg6"] and nf[-1] == NEXT["always"]
        nf = next_forms({"Y": verdict("NOT_ADOPTED", ["zrckp L-musique webqsp"]), "YG": verdict("NO_LOSS")})
        assert nf[0].startswith(NEXT["X"]) and len(nf) == 2
        nf = next_forms({"Y": verdict("NO_LOSS"), "YG": verdict("INCOMPLETE")})
        assert nf[0].startswith(FULL) and nf[1] == f"YG: {NEXT['INCOMPLETE']}"
        assert FULL_RUN["NO_LOSS"] == FULL_RUN["ADOPT_FOR_FULL_RUN"] and "today's graph" in FULL_RUN["NOT_ADOPTED"]
    # 9. train's and read's refusals: the fit's name, arm and split; the card's flags (and abbreviations); the place,
    #    GPU and image; the gates; a read of a fit that is not U2g's or on another device
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        saved_s = Z.S.OUT
        Z.S.OUT = T / "screen"
        try:
            with patched_globals(OUT=T / "u2g"):
                def argv_t(name, arm, *extra):
                    return ["train", "--name", name, "--arm", arm, "--device", "cuda", "--cache-root",
                            str(roots_y(BUILD)["cache"]), *extra]
                nm = fit_name("zspk", "L-musique")
                env = Z.TOY_ENV
                stops(train, argv_t("scr-x", "zspk"), "L-musique", "modal", env, has="not one of U2g's fits")
                stops(train, argv_t(u2d_fit("base", "zspk", 0, "L-musique"), "zspk"), "L-musique", "modal", env,
                      has="not one of U2g's fits")
                stops(train, argv_t(nm, "zrckp"), "L-musique", "modal", env, has="not zrckp's")
                stops(train, argv_t(nm, "zspk"), "L-hotpotqa", "modal", env, has="not zspk's of L-hotpotqa")
                for extra in (["--seed", "1"], ["--se", "1"], ["--hidden", "64"], ["--config", "x"],
                              ["--variants", "p,q"], ["--train", "metaqa=fit"], ["--basis", "x"], ["--repeat", "2"],
                              ["--no-verify"], ["--threads", "4"], ["--out-root", str(T)], ["--host"], ["--seed=1"]):
                    stops(train, argv_t(nm, "zspk", *extra), "L-musique", "modal", env, has="would change that")
                stops(train, argv_t(nm, "zspk"), "L-musique", "laptop", env, has="on Modal's")
                for bad_env in ({**env, "gpu": "NVIDIA H100"}, {**env, "image": "other"}, {**env, "where": "elsewhere"}):
                    stops(train, argv_t(nm, "zspk"), "L-musique", "modal", bad_env, has="on Modal's")
                stops(train, argv_t(nm, "zspk"), "L-musique", "modal", env, has="gate.json missing")
                g = roots_y(BUILD)["gate"]
                g.parent.mkdir(parents=True, exist_ok=True)
                for bad in ({"verdict": "PASS", "declared_in": U2D_DECLARED, "build": BUILD,
                             "construction": [CONSTRUCTION_Y]},
                            {"verdict": "FAIL", "declared_in": DECLARED, "build": BUILD, "construction": [CONSTRUCTION_Y]},
                            {"verdict": "PASS", "declared_in": DECLARED, "build": BUILD, "construction": ["x"]}):
                    ZP.write_json(g, bad)
                    stops(train, argv_t(nm, "zspk"), "L-musique", "modal", env, has="not U2g's PASS for Y")

                def argv_r(name, *extra):
                    return ["--name", name, "--device", "cuda", "--cache-root", str(roots_y(BUILD)["cache"]), *extra]
                stops(read, argv_r("scr-y"), "modal", env, has="not one of U2g's fits")
                stops(read, argv_r(nm), "modal", env, has="is not U2g's zspk fit")
                fd = T / "screen" / "fits" / nm
                fd.mkdir(parents=True)
                good = {"declared_in": DECLARED, "fit": nm, "arm": "zspk", "seed": 0, "split": "L-musique",
                        "device": Z.device_of(env)}
                for bad_u in ({**good, "split": "L-hotpotqa"}, {**good, "declared_in": U2D_DECLARED},
                              {**good, "fit": "scr-u2g-y-zg6"}, {**good, "seed": 1}, {**good, "arm": "zg6"}):
                    Z.LC.write_json(fd / "screen.json", {"arm": "zspk", "u2g": bad_u})
                    stops(read, argv_r(nm), "modal", env, has="is not U2g's zspk fit")
                Z.LC.write_json(fd / "screen.json", {"arm": "zspk", "u2g": good})
                for extra in (["--read", "metaqa=s1eval"], ["--threads", "4"], ["--out-root", str(T)], ["--host"],
                              ["--variants", "p,q"], ["--basis", "x"], ["--no-verify"], ["--rea", "x"]):
                    stops(read, argv_r(nm, *extra), "modal", env, has="would change that")
                stops(read, argv_r(nm), "laptop", env, has="on Modal's")
                other = {**env, "torch": "2.8.0+cu126", "cuda": "12.6"}
                stops(read, argv_r(nm), "modal", other, has="was fitted on")
                stops(read, argv_r(nm), "modal", env, has="not U2g's PASS for Y")
                # the devices: a read on the fit's device; one device where a decision needs it, both kept for readings
                stops(fit_device, nm, has="is not U2g's read")
                Z.LC.write_json(fd / "read.json", {"u2g": {"declared_in": DECLARED, "fit": nm, "device": Z.device_of(other)}})
                stops(fit_device, nm, has="is not U2g's read")
                Z.LC.write_json(fd / "read.json", {"u2g": {"declared_in": DECLARED, "fit": nm, "device": Z.device_of(env)}})
                assert fit_device(nm) == Z.device_of(env)
                b0 = u2d_fit("base", "zspk", 0, "L-musique")
                (T / "screen" / "fits" / b0).mkdir(parents=True)
                ub = {"declared_in": U2D_DECLARED, "fit": b0, "build": "base", "arm": "zspk", "seed": 0,
                      "split": "L-musique", "device": Z.device_of(other)}
                Z.LC.write_json(T / "screen" / "fits" / b0 / "screen.json", {"arm": "zspk", "u2d": ub})
                Z.LC.write_json(T / "screen" / "fits" / b0 / "read.json",
                                {"u2d": {"declared_in": U2D_DECLARED, "fit": b0, "device": Z.device_of(other)}})
                assert fit_device(b0) == Z.device_of(other)
                stops(devices_of, [nm, b0], has="different devices")
                stops(compare_cmd, "scr-u2g-dy-zspk", nm, [b0], True, None, T, has="different devices")
                devs, same = devices_of([nm, b0], one=False)
                assert not same and set(devs) == {nm, b0}
        finally:
            Z.S.OUT = saved_s
    # 10. the pairs and re-calls under U2d's null: Y's lean arms on their own seed-0 bases, zg6 re-based through U2d's
    #     scr-u2d-base-zg6-vs-zspk; the loaders' refusals; screen_pair's and screen_recall's loaders restored
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        nz = Z.fake_null(T / "z", "zspk", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        nr = Z.fake_null(T / "z", "zrckp")
        ym = fake_cmp_y(T / "z", "scr-u2g-dy-zspk", fit_name("zspk", "L-musique"), u2d_fit("base", "zspk", 0, "L-musique"),
                        {"musique": 0.02, "metaqa": -0.003})
        yh = fake_cmp_y(T / "z", "scr-u2g-dy-zspk-hp", fit_name("zspk", "L-hotpotqa"),
                        u2d_fit("base", "zspk", 0, "L-hotpotqa"), {"hotpotqa": -0.0147})
        stops(pair, "Y", "zrckp", [ym, yh], T / "bad", has="not one of U2g's zrckp fits")
        unst = fake_cmp_y(T / "u", "scr-u2g-dy-zspk", fit_name("zspk", "L-musique"),
                          u2d_fit("base", "zspk", 0, "L-musique"), {}, stamp=False)
        stops(pair, "Y", "zspk", [unst, yh], T / "bad", has="not U2g's compare")
        wrong = fake_cmp_y(T / "w", "scr-u2g-dy-zspk", fit_name("zspk", "L-musique"),
                           u2d_fit("new", "zspk", 0, "L-musique"), {})
        stops(pair, "Y", "zspk", [wrong, yh], T / "bad", has="not decision Y's")
        pr = pair("Y", "zspk", [ym, yh], T / "py")
        assert pr["verdict"] == "MIXED" and len(pr["rows"]) == 12
        pj = json.loads((T / "py.json").read_text(encoding="utf-8"))
        assert (pj["stage"], pj["decision"], pj["arm"], pj["null_arm"], pj["declared_in"]) == (
            "U2g", "Y", "zspk", "zspk", DECLARED)
        rc_ = recall("Y", "zspk", T / "py.json", T / "ry", nz)
        assert rc_["verdict"] == "PROMISING" and rc_["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], rc_["changed"]
        rj = json.loads((T / "ry.json").read_text(encoding="utf-8"))
        assert rj["stage"] == "U2g" and rj["decision"] == "Y" and rj["rebased_through"] == {}
        t = (T / "ry.md").read_text(encoding="utf-8")
        assert t.startswith(f"U2g ({DECLARED}), decision Y") and "U2d's own seeds 1 and 2" in t
        stops(recall, "Y", "zrckp", T / "py.json", T / "rz", nr, has="not U2g's pair of zrckp")
        assert Z.SP.load.__module__ == "screen_pair" and Z.SR.load_null.__module__ == "screen_recall"
        rb = {sp: Z.fake_cmp(T / "z", f"scr-u2d-base-zg6-vs-zspk{SUF[sp]}", u2d_fit("base", "zg6", 0, sp),
                             u2d_fit("base", "zspk", 0, sp), {ds: 0.01 for ds in Z.LG.EVAL_ORDER}) for sp in SPLITS}
        gm = fake_cmp_y(T / "z", "scr-u2g-dyg-zg6", fit_name("zg6", "L-musique"), u2d_fit("base", "zg6", 0, "L-musique"),
                        {"webqsp": 0.0215}, base_r5=0.61)
        gh = fake_cmp_y(T / "z", "scr-u2g-dyg-zg6-hp", fit_name("zg6", "L-hotpotqa"),
                        u2d_fit("base", "zg6", 0, "L-hotpotqa"), {}, base_r5=0.61)
        pair("YG", "zg6", [gm, gh], T / "pg")
        stops(recall, "YG", "zg6", T / "pg.json", T / "rg", nz, {}, T / "none", has="missing (the null's re-base)")
        rg = recall("YG", "zg6", T / "pg.json", T / "rg", nz, rb)
        assert rg["verdict"] == "PROMISING"
        rj = json.loads((T / "rg.json").read_text(encoding="utf-8"))
        assert rj["stage"] == "U2g" and rj["decision"] == "YG" and set(rj["rebased_through"]) == set(SPLITS)
        assert "re-based through scr-u2d-base-zg6-vs-zspk" in (T / "rg.md").read_text(encoding="utf-8")
        rbad = Z.fake_cmp(T / "w", "scr-u2d-base-zg6-vs-zspk", u2d_fit("base", "zg6", 0, "L-musique"),
                          u2d_fit("base", "zspk", 0, "L-musique"), {}, base_r5=0.55)
        stops(recall, "YG", "zg6", T / "pg.json", T / "rg2", nz, {"L-musique": rbad, "L-hotpotqa": rb["L-hotpotqa"]},
              has="compared with seed-0 R@5")
        assert Z.SP.load.__module__ == "screen_pair" and Z.SR.load_null.__module__ == "screen_recall"
        # 11. beside: the rule readings under U2d's null (a LOSS inside the floor reads WITHIN) and the decomposition
        #     of U2d's A along Y; the metrics keyed by each fit's build
        for sp, dd in (("L-musique", {"musique": -0.02}), ("L-hotpotqa", {"hotpotqa": -0.010, "2wiki": -0.03})):
            fake_cmp_y(T / "z", f"scr-u2g-rule-zspk{SUF[sp]}", u2d_fit("new", "zspk", 0, sp), fit_name("zspk", sp), dd)
            Z.fake_cmp(T / "z", f"scr-u2d-a-zspk{SUF[sp]}", u2d_fit("new", "zspk", 0, sp), u2d_fit("base", "zspk", 0, sp),
                       {ds: round(x + (0.02 if (sp, ds) == ("L-musique", "musique") else 0.0), 4)
                        for ds, x in {**dd, "hotpotqa": dd.get("hotpotqa", 0.0) - 0.0147 * (sp == "L-hotpotqa")}.items()})
        readings, floors = rule_readings(T / "z")
        assert readings["zspk L-hotpotqa hotpotqa"]["call"] == "LOSS"
        assert readings["zspk L-hotpotqa hotpotqa"]["reading"] == "WITHIN"
        assert readings["zspk L-hotpotqa 2wiki"]["reading"] == "LOSS" and floors["zspk"] == "U2d's null"
        assert readings["zspk L-musique musique"]["floor"] == 0.0075 and "zrckp L-musique metaqa" not in readings
        dec = decomposition(T / "z")
        x = dec["zspk L-musique musique"]
        assert (x["y_minus_base"], x["new_minus_y"], x["sum"], x["u2d_new_minus_base"]) == (0.02, -0.02, 0.0, 0.0)
        x = dec["zspk L-hotpotqa hotpotqa"]
        assert x["sum"] == x["u2d_new_minus_base"] == -0.0247 and dec["zg6 L-musique webqsp"]["new_minus_y"] is None
        fit, rrf = fits_metrics(T / "z")
        assert fit[fit_name("zspk", "L-musique")]["musique"][0] == 0.62 and "y L-musique" in rrf
        assert "new L-hotpotqa" in rrf and "base L-musique" in rrf
    log(f"zu2g selftest: Y registered beside base and new (theirs unchanged, BUILDS untouched); the stages' declaration, "
        f"carves and builds restored (also on an error), zu2d and look_u2d imported only outside them; the carve "
        f"opener (new's carve, P1c2's branch, metaqa and webqsp refused); P1c2's rule is new's fill and new's budget 12 x "
        f"|fill|; same_as_new's PASS on a CRLF checkout, its twelve failures, its refusals, never overwritten; the gate, "
        f"look and stages refused before it; each carve on its dataset's build and declaration, restored (nested, on an "
        f"error); the arms per build; the step-1 bases both builds share; 6 fits and 14 compares on U2d's names; the decisions (ADOPT, NO_LOSS, NOT_ADOPTED, "
        f"five INCOMPLETE) and section 7's next forms; train's and read's refusals; the devices; the pairs and re-calls "
        f"under U2d's null (zg6 re-based); the rule readings and the decomposition ({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    cmd, rest = (argv[0], argv[1:]) if argv else (None, [])
    if cmd in ("pools", "merge", "same-as-new", "kt", "links"):
        ap = argparse.ArgumentParser(prog=f"zu2g.py {cmd}", allow_abbrev=False)
        ap.add_argument("dataset", choices=OWN)
        if cmd in ("pools", "kt", "links"):
            ap.add_argument("--carve", required=True)
        if cmd == "pools":
            ap.add_argument("--shard")
            ap.add_argument("--threads", type=int, default=2)
            ap.add_argument("--limit", type=int, default=0)
        if cmd in ("merge", "same-as-new"):
            ap.add_argument("--new-manifest")
        if cmd == "links":
            ap.add_argument("--modal", action="store_true")
        a = ap.parse_args(rest)
        if cmd == "pools":
            shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
            return pools(a.dataset, a.carve, a.threads, a.limit, shard)
        if cmd == "merge":
            return merge(a.dataset, a.new_manifest)
        if cmd == "same-as-new":
            return same_as_new(a.dataset, a.new_manifest)
        if cmd == "kt":
            return kt(a.dataset, a.carve)
        return links(a.dataset, a.carve, "modal" if a.modal else "laptop")
    if cmd == "look":
        return look(rest)
    if cmd == "copy-bases" and not rest:
        return copy_bases()
    if cmd == "gate" and not rest:
        return gate()
    if cmd in ("train", "read"):
        ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        ap.add_argument("--split", default="L-musique", choices=SPLITS)
        ap.add_argument("--modal", action="store_true")
        k, rest = ap.parse_known_args(rest)
        where = "modal" if k.modal else "laptop"
        return train(["train"] + rest, k.split, where) if cmd == "train" else read(rest, where)
    if cmd == "compare-all" and not rest:
        return compare_all()
    if cmd == "decide" and not rest:
        decide()
        return 0
    raise SystemExit("zu2g: pools, merge, same-as-new, look, copy-bases, links, kt, gate, train, read, compare-all, "
                     "decide, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
