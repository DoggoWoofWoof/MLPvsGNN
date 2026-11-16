"""D83 (docs/D83_UNIFIED_R_REFIT.md section 6): the training step of D83's 20 fits, run in the D83 world only.

    python outputs/host_ops/d83world.py run --mode train -- python outputs/mp_unified/zd83t.py train --name scr-ml1-zspk-s0
    python outputs/mp_unified/zd83t.py selftest

zswpt.py (SWP Part C's train step, not edited) with the D83 world's checks, D83's fit table (zspk at seed 0; zg6r, GNX and zrckpn at seeds 0, 1
and 2; both WQ splits: 20 fits) and D83's model changes in force for the fit (model_changes: the label axis and LAB). Each fit is the record's fit
of the same name (zml1s, zgnx, zltm tables: the same epochs, widths, optimiser, seed and candidate), trained with the D83 world's outputs/pko =
outputs/abl/D83F and not read there: zspk and zg6r get the stand-in read stamp that lets the fits behind them start (D83's reads are later, in
read mode). zswpt is imported first, before numpy or torch: it puts zltm's 6-thread, GPU-hidden environment in place for a zrckpn train from
sys.argv. Neither file imports zabl.

The start of zg6r (the split's seed-0 zspk) and of GNX (the same seed's zg6r, ml1start.LOAD): zml1s.load_encoder takes a start only when its
encoder holds zg6rw.ENC_ELEMENTS numbers. A D83 zspk start is taken with its own count when it is the D83 world's own fit and differs from the
base build at most in l1.weight's input columns (a block dead on D83F's training rows; section 6 records it per fit, not a gate failure).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zswpt as T  # noqa: E402  (first: the zrckpn thread environment; Part C's train step, not edited)
import zd83m as M  # noqa: E402  (D83's model changes; imports neither numpy nor torch at load)

ROOT = T.ROOT
DECLARED = "docs/D83_UNIFIED_R_REFIT.md"
STAGE = "D83"
ARM = "D83F"
WORLD = "d83world"
SEEDS = {"zspk": (0,), "zg6r": (0, 1, 2), "gnx": (0, 1, 2), "zrckpn": (0, 1, 2)}
BASES = {"zspk": "scr-ml1-zspk", "zg6r": "scr-ml1-zg6r", "gnx": "scr-gnx-nx", "zrckpn": "scr-ltm-zrckpn"}
FITS = tuple(f"{BASES[k]}-s{s}{suf}" for k in BASES for s in SEEDS[k] for suf in ("", "-hp"))
BASE_ELEMENTS = 224737                                         # zg6rw.ENC_ELEMENTS: the base build's blocks, l1.weight (128, 309)
T.ARM, T.STAGE = ARM, STAGE


def log(msg):
    T.log(f"D83: {msg}")


def fit_kind(name):
    for k, base in BASES.items():
        for s in SEEDS[k]:
            if name in (f"{base}-s{s}", f"{base}-s{s}-hp"):
                return k
    raise SystemExit(f"D83T: {name} is not one of D83's fits ({len(FITS)}: {', '.join(FITS[:4])}, ...)")


def in_world():
    """The D83 world in train mode and nothing else: d83world.py run --mode train set D83WORLD and SWORLD_MODE, this file sits in a tree
    named d83world, its outputs/pko is the D83F build, and the fit, LTM, screen and read folders are the tree's own."""
    if os.environ.get("D83WORLD") != "1" or os.environ.get("SWORLD_MODE") != "train" or ROOT.name != WORLD:
        raise SystemExit("D83T: a D83 fit trains only in the D83 world, in train mode (python outputs/host_ops/d83world.py run --mode train -- ...)")
    pko = Path(os.path.realpath(ROOT / "outputs" / "pko"))
    if pko.parts[-3:] != ("outputs", "abl", ARM):
        raise SystemExit(f"D83T: the D83 world's outputs/pko is {pko}, not {ARM}")
    for d in ("screen", "ml1", "ltm", "gnx", "d83"):
        p = ROOT / "outputs" / d
        if os.path.realpath(p) != os.path.join(os.path.realpath(ROOT), "outputs", d):
            raise SystemExit(f"D83T: {p} is not the D83 world's own folder")
    return True


def standin_read(name, env):
    """zswpt.standin_read with D83's note: zspk and zg6r start the fits behind them, and zml1s.load_encoder takes a start only when ML1's
    read stamp is on its read.json; a train-mode fit is not read here (D83 reads it later, in read mode)."""
    import zml1s as M1
    fd = M1.WQ.fit_dir(name)
    M1.WQ.write_json(fd / "read.json", {"d83_standin": True, "declared_in": DECLARED,
                                        "note": "no read: a train-mode fit of D83; the stamp lets the next fit start from it"})
    M1.stamp(name, env, "read")
    return 0


def start_count(split):
    """The encoder count of the split's seed-0 zspk start in this world, with its checks: the fit folder is inside the D83 world, its p@swa
    holds every encoder name, and only l1.weight's input columns may differ from the base build (every other tensor's shape is fixed by
    the hidden width). Returns (count, l1 shape)."""
    import torch
    import zg6rw as GW
    import zml1s as M1
    zname = M1.fit_name("zspk", 0, split)
    fd = Path(os.path.realpath(M1.WQ.fit_dir(zname)))
    if not str(fd).startswith(os.path.realpath(ROOT) + os.sep):
        raise SystemExit(f"D83T: {zname}'s fit folder {fd} is outside the D83 world")
    blob = torch.load(fd / "models.pt", map_location="cpu", weights_only=True)
    st = (blob.get("states") or {}).get("p@swa") or {}
    lack = [n for n in GW.ENC_NAMES if n not in st]
    if lack:
        raise SystemExit(f"D83T: {zname}'s p@swa lacks {lack}")
    l1 = tuple(st["l1.weight"].shape) if "l1.weight" in st else None
    n = sum(int(st[k].numel()) for k in GW.ENC_NAMES)
    if l1 is None or l1[0] != 128 or n - l1[0] * l1[1] != BASE_ELEMENTS - 128 * 309:
        raise SystemExit(f"D83T: {zname}'s encoder ({n} numbers, l1.weight {l1}) differs from the base build beyond l1's input columns")
    return n, l1


def d83_load(load):
    """load (zml1s.load_encoder, or ml1start.LOAD, the same function) with the D83 start's own count in force for the call (restored after,
    also on a refusal); the count and l1's shape are logged and kept on the wrapper for the fit's record."""
    def run(split):
        import zg6rw as GW
        import zml1s as M1
        if GW.ENC_ELEMENTS != BASE_ELEMENTS:
            raise SystemExit(f"D83T: zg6rw.ENC_ELEMENTS is {GW.ENC_ELEMENTS}, not the base build's {BASE_ELEMENTS}")
        n, l1 = start_count(split)
        with M1.WQ.patch(GW, ENC_ELEMENTS=n):
            enc, info = load(split)
        if info.get("elements") != n or tuple(enc["l1.weight"].shape) != l1:
            raise SystemExit(f"D83T: the start's encoder changed between the count and the load ({info.get('elements')} against {n})")
        run.seen.append({"split": split, "fit": info.get("fit"), "elements": n, "l1": list(l1), "base": n == BASE_ELEMENTS})
        log(f"start {info.get('fit')} ({split}): {n} numbers, l1.weight {l1}" + ("" if n == BASE_ELEMENTS else " (not the base build's)"))
        return enc, info
    run.d83 = load
    run.seen = []
    return run


def install_load():
    """zml1s.load_encoder and ml1start.LOAD replaced by the D83 start load (ml1start takes zml1s.load_encoder as LOAD at its import)."""
    import ml1start as MS
    import zml1s as M1
    if getattr(M1.load_encoder, "d83", None) is not None or getattr(MS.LOAD, "d83", None) is not None:
        return M1.load_encoder
    if MS.LOAD is not M1.load_encoder:
        raise SystemExit("D83T: ml1start.LOAD is not zml1s.load_encoder")
    M1.load_encoder = MS.LOAD = d83_load(M1.load_encoder)
    return M1.load_encoder


def fit_record_path(name):
    return ROOT / "outputs" / "d83" / "fits" / f"{name}.json"


def train(name):
    """One D83 fit: the label axis set before any model module is imported (zd83m.labels_on), D83's model changes for the
    fit's kind in force (zd83m.model_changes), the record's train step (zswpt), then the fit's D83 record: the label
    axis every imported module held, LAB's carves, the start's encoder count (outputs/d83/fits/<name>.json)."""
    in_world()
    kind = fit_kind(name)
    M.labels_on()                                              # before zml1s, zgnx or zltm is imported
    if kind in ("zg6r", "gnx"):
        install_load()
    with M.model_changes(kind):
        rc = T.train(name)
        rec = M.fit_record(kind)
    if rc == 0:
        import zml1s as M1
        rec.update(fit=name, stage=STAGE, arm=ARM, starts=list(getattr(M1.load_encoder, "seen", [])),
                   zd83m_sha256=M.sha_src(M.__file__), zd83t_sha256=M.sha_src(__file__),
                   utc=__import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()))
        M.write_json(fit_record_path(name), rec)
        log(f"{name}: D83 record written ({fit_record_path(name).relative_to(ROOT).as_posix()})")
    return rc


T.in_world = in_world                                          # train() and stamp_start() look these names up at call time
T.standin_read = standin_read
T.fit_kind = fit_kind
T.FITS = FITS


def selftest():
    assert len(FITS) == 20 and len(set(FITS)) == 20
    assert [fit_kind(n) for n in FITS].count("zspk") == 2 and [fit_kind(n) for n in FITS].count("gnx") == 6
    assert [T.split_of(n) for n in FITS[:2]] == list(T.SPLITS)
    for bad in ("scr-ml1-zspk-s1", "scr-ml1-zrckp-s0", "x"):
        try:
            fit_kind(bad)
            raise AssertionError(bad)
        except SystemExit:
            pass
    keep = dict(os.environ)
    try:
        for env in ({}, {"SWORLD": "1", "SWORLD_MODE": "train"}, {"NWORLD": "1", "SWORLD_MODE": "train"},
                    {"D83WORLD": "1", "SWORLD_MODE": "read"}, {"D83WORLD": "1", "SWORLD_MODE": "train"}):
            os.environ.clear()
            os.environ.update({k: v for k, v in keep.items() if k not in ("NWORLD", "SWORLD", "SWORLD_MODE", "D83WORLD")}, **env)
            for f in (in_world, lambda: train("scr-ml1-zspk-s0"), lambda: T.stamp_start("scr-ml1-zspk-s0")):
                try:
                    f()
                    raise AssertionError(f"must refuse outside the D83 world in train mode ({env}, tree {ROOT.name})")
                except SystemExit:
                    pass
    finally:
        os.environ.clear()
        os.environ.update(keep)
    assert "zabl" not in sys.modules and "zd83" not in sys.modules, "zd83t must not import zabl (the PKO environment)"
    M.labels_on()                                              # as train() does, before zml1s, zgnx and zltm
    import zgnx
    import zltm
    import zml1s
    assert set(FITS) <= set(zml1s.ALL_FITS) | set(zgnx.FITS) | set(zltm.FITS), "a D83 fit is in no module's fit table"
    for n in FITS:
        k = fit_kind(n)
        assert (n in zml1s.ALL_FITS) == (k in ("zspk", "zg6r")) and (n in zgnx.FITS) == (k == "gnx") and (n in zltm.FITS) == (k == "zrckpn"), n
    # the start load: the count is patched for the call only, and restored after a refusal
    import ml1start as MS
    import zg6rw as GW
    orig_m1, orig_ms, orig_sc = zml1s.load_encoder, MS.LOAD, start_count
    try:
        seen = []

        def fake(split):
            seen.append(GW.ENC_ELEMENTS)
            import torch
            return {"l1.weight": torch.zeros(128, 300)}, {"fit": "f", "elements": GW.ENC_ELEMENTS}

        globals()["start_count"] = lambda split: (BASE_ELEMENTS - 128 * 9, (128, 300))
        ld = d83_load(fake)
        enc, info = ld("WQ-musique")
        assert seen == [BASE_ELEMENTS - 1152] and GW.ENC_ELEMENTS == BASE_ELEMENTS and ld.seen[0]["base"] is False
        globals()["start_count"] = lambda split: (BASE_ELEMENTS, (128, 309))
        try:
            ld("WQ-musique")
            raise AssertionError("a start whose shape changed between count and load must be refused")
        except SystemExit:
            pass
        assert GW.ENC_ELEMENTS == BASE_ELEMENTS
    finally:
        globals()["start_count"] = orig_sc
        zml1s.load_encoder, MS.LOAD = orig_m1, orig_ms
    print("zd83t self-test: OK")
    return 0


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if a[:1] == ["selftest"]:
        return selftest()
    if len(a) == 3 and a[0] == "train" and a[1] == "--name":
        return train(a[2])
    if len(a) == 3 and a[0] == "stamp" and a[1] == "--name":
        return T.stamp_start(a[2])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
