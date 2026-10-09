"""A diagnostic, not a round (docs/DIAG_ZGF_HARM.md): rounds twenty-six (zgf) and twenty-seven (zgr) read the same harm
on the passage graphs in L-hotpotqa's fit (musique in-domain -0.060 and -0.047, hotpotqa read zero-shot -0.044 and
-0.026, squad -0.014 and -0.014, against zrm's card fit). Where does it come from?

zgf and zgr are zrm's score plus a space head added at read. Their training also moves zrm's own weights, through the
head's gradient and the edge (and chain) losses. This file reads each trained fit twice more on the CPU:

    off   the fit's own weights with the space head left out (zrm's forward on the fit's zrm weights)
    on    the same fit as trained, head included (the device control: the filed reads were on the card)

    head at read    off reads level with zrm's fit where the fit lost: the head's read-time score does the harm
    drift           off still loses: training moved zrm's own weights
    both            between the two

The head is left out by forward only: no weight is changed, and the fit's folder is never written. `stage` copies the
fit (its models.pt, screen.json and train.json; not its reads) to outputs/diag/zgfoff/fits/<name> and names the
copy's arm; `read` and `compare` are lean_screen's.

    python outputs/mp_unified/zgfoff.py stage --src outputs/screen/fits/scr-zgf-hp --name scr-zgf-hp-off --arm zgf_off
    python outputs/mp_unified/zgfoff.py read --name scr-zgf-hp-off --device cpu --threads 4 --host
    python outputs/mp_unified/zgfoff.py compare --new outputs/diag/zgfoff/fits/scr-zgf-hp-off \\
        --base outputs/screen/fits/scr-zrm-hp,outputs/screen/fits/scr-zgf-hp --out outputs/diag/zgfoff/scr-zgf-hp-off
    python outputs/mp_unified/zgfoff.py --selftest

No training, no new carve, graph, encoder or setting. The MLP track: nothing of a row's neighbours is read.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import zgf as GF  # noqa: E402
import zgr as GR  # noqa: E402

ZL, ZM, RM, S, S2, LC = GF.ZL, GF.ZM, GF.RM, GF.S, GF.S2, GF.LC
log = S.log

OUT = Path("outputs/diag/zgfoff")
COPY = ("models.pt", "screen.json", "train.json")


class ZGFOff(GF.ZGF):
    """zgf's fit read with its space head left out: zrm's forward on the fit's weights."""

    def forward(self, feats, keep, nq, B, base_z):
        return ZM.ZRM.forward(self, feats, keep, nq, B, base_z)


class ZGROff(GR.ZGR):
    """zgr's fit read with its space head (and relation inputs) left out: zrm's forward on the fit's weights."""

    def forward(self, feats, keep, nq, B, base_z):
        return ZM.ZRM.forward(self, feats, keep, nq, B, base_z)


OFF = {"zgf_off": (ZGFOff, ZL.LinkCarveBase, "zgf"), "zgr_off": (ZGROff, ZL.LinkCarveBase, "zgr"),
       "zgf_on": (GF.ZGF, ZL.LinkCarveBase, "zgf"), "zgr_on": (GR.ZGR, ZL.LinkCarveBase, "zgr")}
S.ARMS.update({k: v[:2] for k, v in OFF.items()})


def stage(src, name, arm, root=None):
    """Copy the fit's models.pt, screen.json and train.json to root/fits/name; the copy's arm is `arm`. The source must
    have been trained as the arm's source arm; an existing copy is refused."""
    src, root = Path(src), Path(root or OUT)
    if arm not in OFF:
        raise SystemExit(f"zgfoff: --arm one of {sorted(OFF)}, not {arm}")
    rec = json.loads((src / "screen.json").read_text(encoding="utf-8"))
    if rec["arm"] != OFF[arm][2]:
        raise SystemExit(f"zgfoff: {src} was trained as {rec['arm']}, not {OFF[arm][2]}")
    dst = root / "fits" / name
    if dst.exists():
        raise SystemExit(f"zgfoff: {dst} exists")
    dst.mkdir(parents=True)
    for f in COPY:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    rec.update({"arm": arm, "trained_arm": OFF[arm][2], "source_fit": src.as_posix(), "name": name,
                "source_models_sha256": LC.sha_file(src / "models.pt"), "zgfoff_sha256": LC.sha_src(__file__),
                "head": "off at read" if arm.endswith("_off") else "on (device control)", "message_passing": False})
    LC.write_json(dst / "screen.json", rec)
    assert LC.sha_file(dst / "models.pt") == rec["source_models_sha256"]
    log(f"zgfoff stage: {src} -> {dst} as {arm} (models {rec['source_models_sha256'][:12]})")
    return 0


def read(argv):
    if "--out-root" not in argv:
        argv = argv + ["--out-root", str(OUT / "fits")]
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm not in OFF:
        raise SystemExit(f"zgfoff: {name} is staged as {arm}, not one of {sorted(OFF)}")
    return RM.read(argv)


def selftest():
    """On a toy carve: the off arms score as zrm's forward on the same weights; the on arms as their fits; off differs
    from on once the head's last layer is not zero; stage copies bit for bit and refuses a wrong source arm."""
    GF.LG.bind_device_ops()
    tmp = Path(tempfile.mkdtemp(prefix="zgfoff_"))
    try:
        import numpy as np
        rng = np.random.default_rng(26)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        cls = ZL.link_carve(RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel"), tmp / "lk")
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.arange(min(9, c.rows), dtype=np.int64)
        feats, nq, bz, _g = c.batch(qs, blocks)
        B = qs.size
        keep = torch.ones(B, len(blocks))
        gen = torch.Generator().manual_seed(3)
        for on_cls, off_cls in ((GF.ZGF, ZGFOff), (GR.ZGR, ZGROff)):
            torch.manual_seed(0)
            on = on_cls(blocks, c.widths, 16, dropout=0.1, seed=0)
            with torch.no_grad():
                for k, v in on.state_dict().items():
                    if k.startswith(("gf_", "gr_")):
                        v.copy_(torch.randn(v.shape, generator=gen) * 0.3)
            off = off_cls(blocks, c.widths, 16, dropout=0.1, seed=0)
            off.load_state_dict(on.state_dict())
            zr = ZM.ZRM(blocks, c.widths, 16, dropout=0.1, seed=0)
            zr.load_state_dict({k: v for k, v in on.state_dict().items() if not k.startswith(("gf_", "gr_"))})
            for m in (on, off, zr):
                m.eval()
            with torch.no_grad():
                s_on, s_off, s_zr = (m(feats, keep, nq, B, bz) for m in (on, off, zr))
            assert torch.equal(s_off, s_zr), on_cls.__name__
            assert not torch.equal(s_on, s_off), on_cls.__name__
        assert S.ARMS["zgf_off"] == (ZGFOff, ZL.LinkCarveBase) and S.ARMS["zgr_off"] == (ZGROff, ZL.LinkCarveBase)
        # stage: a copy bit for bit, the arm renamed; a wrong source arm and an existing copy are refused
        src = tmp / "fit"
        src.mkdir()
        torch.save({"x": torch.arange(5)}, src / "models.pt")
        (src / "screen.json").write_text(json.dumps({"arm": "zgf", "name": "fit"}), encoding="utf-8")
        (src / "reads").mkdir()
        assert stage(src, "fit-off", "zgf_off", root=tmp / "o") == 0
        d = tmp / "o" / "fits" / "fit-off"
        assert LC.sha_file(d / "models.pt") == LC.sha_file(src / "models.pt") and not (d / "reads").exists()
        assert json.loads((d / "screen.json").read_text(encoding="utf-8"))["arm"] == "zgf_off"
        for args in ((src, "fit-off", "zgf_off"), (src, "fit-r", "zgr_off"), (src, "fit-x", "zgf")):
            try:
                stage(*args, root=tmp / "o")
                raise AssertionError(f"stage {args} should refuse")
            except SystemExit:
                pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log("zgfoff selftest: ok")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    cmd, rest = (argv[0], argv[1:]) if argv else ("", [])
    if cmd == "stage":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--src", required=True)
        ap.add_argument("--name", required=True)
        ap.add_argument("--arm", required=True)
        a = ap.parse_args(rest)
        return stage(a.src, a.name, a.arm)
    if cmd == "read":
        return read(rest)
    if cmd == "compare":
        return S2.main(["compare"] + rest)
    raise SystemExit("zgfoff: stage, read, compare or --selftest")


if __name__ == "__main__":
    sys.exit(main())
