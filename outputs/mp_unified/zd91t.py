"""D91 (docs/D91_D83_WITHOUT_LABELS.md section 2): the training step of D91's 20 fits, run in the D91 world only.

    python outputs/host_ops/d91world.py run --mode train -- python outputs/mp_unified/zd91t.py train --name scr-ml1-zspk-s0
    python outputs/mp_unified/zd91t.py selftest

zswpt.py (SWP Part C's train step, not edited) with the D91 world's checks and D83's fit table (zspk at seed 0; zg6r, GNX and zrckpn
at seeds 0, 1 and 2; both WQ splits: 20 fits). Each fit is the record's fit of the same name (zml1s, zgnx, zltm tables: the same
epochs, widths, optimiser, seed and candidate) with the record's model code and no D83 change: zlink's three families (no label
axis), LTM's three family bits, no LAB. It trains with the D91 world's outputs/pko = outputs/abl/D91F and is not read there: zspk
and zg6r get the stand-in read stamp that lets the fits behind them start (D91's reads are later, zd91r.py). zswpt is imported
first, before numpy or torch: it puts zltm's 6-thread, GPU-hidden environment in place for a zrckpn train from sys.argv. Neither
file imports zabl.

The starts are the record's rule: zml1s.load_encoder takes a start only when its encoder holds zg6rw.ENC_ELEMENTS numbers (the
base build). D83's own zspk starts held the base build's count (its fit records: 224,737, l1.weight (128, 309)), and D91's
inputs are the record's widths, so no start rule is changed here.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zswpt as T  # noqa: E402  (first: the zrckpn thread environment; Part C's train step, not edited)

ROOT = T.ROOT
DECLARED = "docs/D91_D83_WITHOUT_LABELS.md"
STAGE = "D91"
ARM = "D91F"
WORLD = "d91world"
OWN = ("screen", "ml1", "ltm", "gnx", "d91")
SEEDS = {"zspk": (0,), "zg6r": (0, 1, 2), "gnx": (0, 1, 2), "zrckpn": (0, 1, 2)}
BASES = {"zspk": "scr-ml1-zspk", "zg6r": "scr-ml1-zg6r", "gnx": "scr-gnx-nx", "zrckpn": "scr-ltm-zrckpn"}
FITS = tuple(f"{BASES[k]}-s{s}{suf}" for k in BASES for s in SEEDS[k] for suf in ("", "-hp"))
T.ARM, T.STAGE = ARM, STAGE


def log(msg):
    T.log(f"D91: {msg}")


def sha_src(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fit_kind(name):
    for k, base in BASES.items():
        for s in SEEDS[k]:
            if name in (f"{base}-s{s}", f"{base}-s{s}-hp"):
                return k
    raise SystemExit(f"D91T: {name} is not one of D91's fits ({len(FITS)}: {', '.join(FITS[:4])}, ...)")


def in_world():
    """The D91 world in train mode and nothing else: d91world.py run --mode train set D91WORLD and SWORLD_MODE, this file sits in a
    tree named d91world, its outputs/pko is the D91F build, and the fit, LTM, screen and d91 folders are the tree's own."""
    if os.environ.get("D91WORLD") != "1" or os.environ.get("SWORLD_MODE") != "train" or ROOT.name != WORLD:
        raise SystemExit("D91T: a D91 fit trains only in the D91 world, in train mode (python outputs/host_ops/d91world.py run --mode train -- ...)")
    pko = Path(os.path.realpath(ROOT / "outputs" / "pko"))
    if pko.parts[-3:] != ("outputs", "abl", ARM):
        raise SystemExit(f"D91T: the D91 world's outputs/pko is {pko}, not {ARM}")
    for d in OWN:
        p = ROOT / "outputs" / d
        if os.path.realpath(p) != os.path.join(os.path.realpath(ROOT), "outputs", d):
            raise SystemExit(f"D91T: {p} is not the D91 world's own folder")
    return True


def record_axis():
    """The record's three families in force (no D83 label axis anywhere in the process). Returns zlink."""
    import zlink as ZL
    if getattr(ZL, "D83_LABELS", False) or ZL.FAMS != 3 or ZL.SETTINGS.get("families") != 3 or "labels" in ZL.SETTINGS:
        raise SystemExit(f"D91T: zlink holds {ZL.FAMS} families ({ZL.SETTINGS.get('families')}); D91 trains on the record's three")
    if "zd83m" in sys.modules:
        raise SystemExit("D91T: zd83m (D83's model changes) is imported in a D91 process")
    return ZL


def standin_read(name, env):
    """zswpt.standin_read with D91's note: zspk and zg6r start the fits behind them, and zml1s.load_encoder takes a start only when
    ML1's read stamp is on its read.json; a train-mode fit is not read here (D91 reads it later, zd91r.py)."""
    import zml1s as M1
    fd = M1.WQ.fit_dir(name)
    M1.WQ.write_json(fd / "read.json", {"d91_standin": True, "declared_in": DECLARED,
                                        "note": "no read: a train-mode fit of D91; the stamp lets the next fit start from it"})
    M1.stamp(name, env, "read")
    return 0


def fit_record_path(name):
    return ROOT / "outputs" / "d91" / "fits" / f"{name}.json"


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


def train(name):
    """One D91 fit: the record's three families checked, the record's train step (zswpt), then the fit's D91 record
    (outputs/d91/fits/<name>.json: the fit's models.pt sha256, its start, zlink's settings)."""
    in_world()
    kind = fit_kind(name)
    ZL = record_axis()
    rc = T.train(name)
    record_axis()
    if rc:
        return rc
    import zml1s as M1
    fd = Path(os.path.realpath(M1.WQ.fit_dir(name)))
    if not (fd / "models.pt").is_file():
        raise SystemExit(f"D91T: {name} trained without a models.pt in {fd}")
    starts = []
    if kind in ("zg6r", "gnx"):
        import zg6rw as GW
        starts.append({"enc_elements": GW.ENC_ELEMENTS})
    rec = {"declared_in": DECLARED, "stage": STAGE, "arm": ARM, "fit": name, "kind": kind, "split": T.split_of(name),
           "models_sha256": sha_file(fd / "models.pt"), "zlink_fams": ZL.FAMS, "zlink_settings": ZL.SETTINGS, "starts": starts,
           "zd91t_sha256": sha_src(__file__), "zswpt_sha256": sha_src(T.__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(fit_record_path(name), rec)
    log(f"{name}: D91 record written ({fit_record_path(name).relative_to(ROOT).as_posix()})")
    return 0


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
        for env in ({}, {"SWORLD": "1", "SWORLD_MODE": "train"}, {"D83WORLD": "1", "SWORLD_MODE": "train"},
                    {"D91WORLD": "1", "SWORLD_MODE": "read"}, {"D91WORLD": "1", "SWORLD_MODE": "train"}):
            os.environ.clear()
            os.environ.update({k: v for k, v in keep.items() if k not in ("NWORLD", "SWORLD", "SWORLD_MODE", "D83WORLD", "D91WORLD")}, **env)
            for f in (in_world, lambda: train("scr-ml1-zspk-s0"), lambda: T.stamp_start("scr-ml1-zspk-s0")):
                try:
                    f()
                    raise AssertionError(f"must refuse outside the D91 world in train mode ({env}, tree {ROOT.name})")
                except SystemExit:
                    pass
    finally:
        os.environ.clear()
        os.environ.update(keep)
    assert "zabl" not in sys.modules and "zd83m" not in sys.modules, "zd91t must import neither zabl nor zd83m"
    record_axis()
    import zgnx
    import zltm
    import zml1s
    import zg6rw
    assert set(FITS) <= set(zml1s.ALL_FITS) | set(zgnx.FITS) | set(zltm.FITS), "a D91 fit is in no module's fit table"
    for n in FITS:
        k = fit_kind(n)
        assert (n in zml1s.ALL_FITS) == (k in ("zspk", "zg6r")) and (n in zgnx.FITS) == (k == "gnx") and (n in zltm.FITS) == (k == "zrckpn"), n
    assert zg6rw.ENC_ELEMENTS == 224737, zg6rw.ENC_ELEMENTS        # D83's zspk starts held this count (base build)
    assert zgnx.IN == 15 if hasattr(zgnx, "IN") else True          # the record's hook input: [raw, z, 1] over NEXT's 7
    assert not getattr(zgnx, "D83_LAB", False) and not getattr(zltm, "D83_LAB", False)
    record_axis()
    print("zd91t self-test: OK")
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
