"""D91 (docs/D91_D83_WITHOUT_LABELS.md section 4): LTM's lists, fold fits and NEXT columns on D91's links, in the D91 world.

ltmcols.py and ltwdiag.py are the record's, not edited and not patched: the candidate features hold the record's three family bits.
Every step checks the record's three families are in force (no D83 label axis) and that every links part of the carve is D91's
(zd91.py: settings = zlink's, a d91 block naming the arm).

    python outputs/host_ops/d91world.py run --mode train -- python outputs/mp_unified/zd91m.py ltm lists --arm D91F --dataset musique --carve s1fit
    python outputs/host_ops/d91world.py run --mode train -- python outputs/mp_unified/zd91m.py ltm fit --split WQ-musique --host
    python outputs/host_ops/d91world.py run --mode train -- python outputs/mp_unified/zd91m.py ltm cols --arm D91F --split WQ-musique --dataset 2wiki --carve fit --host
    python outputs/host_ops/d91world.py run --mode train -- python outputs/mp_unified/zd91m.py b1 lists --dataset musique
    python outputs/host_ops/d91world.py run --mode train -- python outputs/mp_unified/zd91m.py b1 cols --split WQ-musique --dataset musique --host
    python outputs/mp_unified/zd91m.py selftest

D91F's and D91E's lists and columns are written to the world's own outputs/ltm (as D83's: the carves differ, fit/s1fit against
s1eval). D91B's (B1's three settings) go to outputs/abl/D91B/ltm through zb1h's lists and cols (zb1h is never edited; its CARVE,
OUT and LTM_OUT are bound to D91B's, as zd83b binds them to b1v's): the carve keeps b1v's name and files, its links are D91's.
(D91B's first name was AP2's carve folder: every D91 arm is under outputs/abl/D91*, zd91.claim.)
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DECLARED = "docs/D91_D83_WITHOUT_LABELS.md"
STAGE = "D91"
WORLD = "d91world"
ARMS = {"D91F": ROOT / "outputs" / "abl" / "D91F", "D91E": ROOT / "outputs" / "abl" / "D91E"}
B1_ARM = "D91B"
B1_ROOT = ROOT / "outputs" / "abl" / B1_ARM
B1_CARVE = "b1v"                        # D91B keeps b1v's carve name and files (zd91.CARVES)
B1_SETTINGS = ("2wiki", "hotpotqa", "musique")


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rel(p):
    try:
        return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def world_ok(where="D91M"):
    """A step that writes the D91 world's own folders runs there only: D91WORLD=1, a tree named d91world."""
    if os.environ.get("D91WORLD") != "1" or ROOT.name != WORLD:
        raise SystemExit(f"{where}: runs only in the D91 world (python outputs/host_ops/d91world.py run -- ...)")
    return True


def record_axis():
    """The record's three families in force: zlink unpatched, no D83 module in the process. Returns zlink."""
    import zlink as ZL
    if getattr(ZL, "D83_LABELS", False) or ZL.FAMS != 3 or ZL.SETTINGS.get("families") != 3 or "labels" in ZL.SETTINGS:
        raise SystemExit(f"D91M: zlink holds {ZL.FAMS} families ({ZL.SETTINGS.get('families')}); D91 reads the record's three")
    if "zd83m" in sys.modules:
        raise SystemExit("D91M: zd83m (D83's model changes) is imported in a D91 process")
    return ZL


def links_ok(lp, arm):
    """A links part zd91.py wrote for the arm: zlink's settings, the record's three families, a d91 block naming the arm."""
    ZL = record_axis()
    h = read_json(Path(lp) / "record.json")
    d = h.get("d91") or {}
    if h.get("settings") != ZL.SETTINGS or d.get("arm") != arm or d.get("stage") != STAGE:
        raise SystemExit(f"D91M: {rel(lp)} is not a D91 links part of {arm} (families {(h.get('settings') or {}).get('families')}, "
                         f"arm {d.get('arm')}, stage {d.get('stage')})")
    return h


def parts_ok(root, ds, carve, arm):
    import lean_cache as LC
    parts, _recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    if not parts:
        raise SystemExit(f"D91M: no cache parts of {ds}/{carve} under {rel(root)}")
    for p in parts:
        links_ok(Path(root) / "links" / ds / carve / p.name, arm)
    return len(parts)


def ltw_ok(LTMC):
    """ltwdiag as the record has it: no D83 patch on the candidate features."""
    if getattr(LTMC.LTW, "D83_BITS", False) or any(getattr(LTMC.LTW, k, None) is not None and getattr(getattr(LTMC.LTW, k), "d83", False)
                                                   for k in ("feature_rows", "low_fam", "summarise_a")):
        raise SystemExit("D91M: ltwdiag carries D83's label bits")
    return True


def ltm(argv):
    """LTM's lists, fold fits and columns (ltmcols, not edited) in the D91 world; lists and cols on the named arm's cache and
    links (every links part checked as the arm's D91 links first)."""
    import argparse
    import ltmcols as LTMC                                  # first: its BLAS thread environment before numpy
    ap = argparse.ArgumentParser(prog="zd91m.py ltm")
    ap.add_argument("cmd", choices=("lists", "fit", "cols"))
    ap.add_argument("--arm", choices=tuple(ARMS))
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--split", choices=tuple(LTMC.WQ.SPLITS))
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    world_ok("D91M ltm")
    record_axis()
    ltw_ok(LTMC)
    root = None
    if a.cmd in ("lists", "cols"):
        if not a.arm:
            raise SystemExit("D91M ltm: --arm")
        root = ARMS[a.arm]
        if (a.dataset, a.carve) not in LTMC.ALL_CARVES:
            raise SystemExit(f"D91M: {a.dataset}/{a.carve} is not one of LTM's carves")
        parts_ok(root, a.dataset, a.carve, a.arm)
    if a.cmd == "lists":
        LTMC.lists(a.dataset, a.carve, root=root)
        return 0
    if not a.split:
        raise SystemExit("D91M ltm: --split")
    if a.host:
        import lean_host as LH
        LH.substitute()
    if a.cmd == "fit":
        LTMC.fit(a.split, host=a.host)
        return 0
    LTMC.cols(a.split, a.dataset, a.carve, root=root, host=a.host)
    return 0


def bind_b1():
    """zb1h's globals that its lists and cols read (CARVE, OUT, LTM_OUT) set to D91B's; returns zb1h."""
    import zb1h as H
    H.CARVE, H.OUT, H.LTM_OUT = B1_CARVE, B1_ROOT, B1_ROOT / "ltm"
    H.DECLARED, H.STAGE = DECLARED, STAGE
    return H


def b1(argv):
    import argparse
    import ltmcols as LTMC                                  # first: its BLAS thread environment before numpy
    ap = argparse.ArgumentParser(prog="zd91m.py b1")
    ap.add_argument("cmd", choices=("lists", "cols"))
    ap.add_argument("--dataset", choices=B1_SETTINGS, required=True)
    ap.add_argument("--split", choices=tuple(LTMC.WQ.SPLITS))
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    world_ok("D91M b1")
    record_axis()
    ltw_ok(LTMC)
    parts_ok(B1_ROOT, a.dataset, B1_CARVE, B1_ARM)
    H = bind_b1()
    if a.cmd == "lists":
        return H.lists(a.dataset)
    if not a.split:
        raise SystemExit("D91M b1 cols: --split")
    return H.cols(a.split, a.dataset, a.host)


def selftest():
    import tempfile
    keep = dict(os.environ)
    try:
        os.environ.pop("D91WORLD", None)
        for f in (lambda: ltm(["lists", "--arm", "D91F", "--dataset", "musique", "--carve", "s1fit"]),
                  lambda: b1(["lists", "--dataset", "musique"])):
            try:
                f()
                raise AssertionError("must refuse outside the D91 world")
            except SystemExit as e:
                assert "D91 world" in str(e), e
    finally:
        os.environ.clear()
        os.environ.update(keep)
    ZL = record_axis()
    import ltmcols as LTMC
    ltw_ok(LTMC)
    assert ("musique", "s1fit") in LTMC.ALL_CARVES and all((d, "s1eval") in LTMC.ALL_CARVES for d in
                                                           ("squad", "hotpotqa", "2wiki", "musique", "metaqa", "webqsp"))
    tmp = Path(tempfile.mkdtemp(prefix="zd91m_t_"))
    try:
        good = {"settings": dict(ZL.SETTINGS), "d91": {"arm": "D91F", "stage": "D91"}}
        (tmp / "a").mkdir()
        (tmp / "a" / "record.json").write_text(json.dumps(good), encoding="utf-8")
        links_ok(tmp / "a", "D91F")
        for bad in ({**good, "d91": {"arm": "D91E", "stage": "D91"}}, {k: v for k, v in good.items() if k != "d91"},
                    {**good, "settings": {**ZL.SETTINGS, "families": 5, "labels": ["S"]}}):
            (tmp / "a" / "record.json").write_text(json.dumps(bad), encoding="utf-8")
            try:
                links_ok(tmp / "a", "D91F")
                raise AssertionError(f"must refuse {bad}")
            except SystemExit:
                pass
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    H = bind_b1()
    assert H.CARVE == "b1v" and H.OUT == B1_ROOT and H.LTM_OUT == B1_ROOT / "ltm" and B1_ROOT.parent.name == "abl"
    import inspect
    src_l, src_c = inspect.getsource(H.lists), inspect.getsource(H.cols)
    assert "CARVE" in src_l and "LTM_OUT" in src_l and "OUT" in src_c and "CARVE" in src_c
    assert "zd83m" not in sys.modules
    print("zd91m self-test: OK")
    return 0


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if a[:1] == ["selftest"]:
        return selftest()
    if a[:1] == ["ltm"]:
        return ltm(a[1:])
    if a[:1] == ["b1"]:
        return b1(a[1:])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
