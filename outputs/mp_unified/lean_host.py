"""Run a lean MLP look (lean_mlp3, lean_mlp4 or lean_mlp5) on the host, with the verified mirror in place of the package, in memory.

    python outputs/mp_unified/lean_host.py lean_mlp4 --store pca256 --train 2wiki=x4 ... (the look's own arguments)
    python outputs/mp_unified/lean_host.py --selftest

The substitution is mp_approx_six_base_score.host_mode, the one look_x_six --host uses: it checks the pinned VERIFIED
records of amendment 2 of configs/mp_approx_six_base.yaml, then wraps universal_v2_run.load_configs in this process so
substrate.package_root = the mirror root. No config file is edited and the looks are imported unchanged; their
open_nodes / served_root call load_configs through look_x_six's V2, which is the wrapped module. Hard stops land in
outputs/mp_unified (look_x_six's redirection), never under a level's or the six-base's outputs. The BLAS thread
variables default to the look's --threads before numpy loads (lean_mlp4 imports numpy before lean_mlp sets them).
After the look returns, its --out JSON gets the placement and this file's sha256.
"""
import os
import sys


def _threads(argv):
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--threads="):
            return a.split("=", 1)[1]
    return "2"


if __name__ == "__main__":
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, _threads(sys.argv))
sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LOOKS = ("lean_mlp3", "lean_mlp4", "lean_mlp5", "lean_score")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def log(msg):
    print(msg, flush=True)


def substitute():
    """The mirror in place of the package for this process; returns the placement look_x_six files with its records."""
    import look_x_six as LX   # imports SB and S6, then points every hard stop at outputs/mp_unified
    return dict(LX.S6.host_mode(LX.SB.load_declaration(), log))


def out_path(argv):
    for i, a in enumerate(argv):
        if a == "--out" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--out="):
            return Path(a.split("=", 1)[1])
    return None


def annotate(path, placement, look):
    if path is None or not path.exists():
        return
    res = json.loads(path.read_text(encoding="utf-8"))
    res["placement"] = placement
    res["host_wrapper"] = {"file": "outputs/mp_unified/lean_host.py", "sha256": sha(__file__), "look": look}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    log(f"lean_host: {path} carries the placement ({placement.get('where')})")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: lean_host.py lean_mlp3|lean_mlp4|lean_mlp5|lean_score ARGS... | --selftest")
    if sys.argv[1] == "--selftest":
        return selftest()
    look = sys.argv[1]
    if look not in LOOKS:
        raise SystemExit(f"lean_host.py: {look} is not one of {LOOKS}")
    rest = sys.argv[2:]
    mod = importlib.import_module(look)
    placement = substitute()
    log(f"lean_host: {look} ({sha(mod.__file__)[:12]}) on {placement.get('node')}, BLAS threads "
        f"{os.environ.get('OPENBLAS_NUM_THREADS')}")
    if hasattr(mod, "PLACEMENT"):
        mod.PLACEMENT.update(placement)       # lean_score records it beside each score file
    sys.argv = [mod.__file__] + rest
    mod.main()
    annotate(out_path(rest), placement, {"name": look, "sha256": sha(mod.__file__)})


def selftest():
    import tempfile
    assert _threads(["x", "--threads", "4"]) == "4" and _threads(["x", "--threads=3"]) == "3" and _threads(["x"]) == "2"
    assert out_path(["--out", "a/b.json"]) == Path("a/b.json") and out_path(["--out=c.json"]) == Path("c.json")
    assert out_path(["--save-models", "m.pt"]) is None
    import look_x_six as LX
    V2 = LX.V2
    _c, before, _h = V2.load_configs()
    placement = substitute()
    _c, after, _h = V2.load_configs()
    import yaml
    root = yaml.safe_load((ROOT / "configs" / "host_mirror_six.yaml").read_text(encoding="utf-8"))["host"]["mirror_root"]
    assert after["substrate"]["package_root"] == str(Path(root)), after["substrate"]["package_root"]
    assert before["substrate"]["package_root"] != after["substrate"]["package_root"]
    assert placement["where"] == "host" and placement["mirror_root"] == root
    assert LX.SB.OUT == HERE and LX.L0.HARD_STOP_DIR[0] == HERE, "hard stops must land in outputs/mp_unified"
    import lean_mlp3 as L3
    import lean_mlp4 as L4
    import lean_mlp5 as L5
    import lean_score as LS
    assert L3.open_nodes.__globals__["sys"] is sys and callable(L4.served_root) and callable(L5.fit5) and LS.PLACEMENT == {"where": "laptop"}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.json"
        p.write_text(json.dumps({"look": "x"}), encoding="utf-8")
        annotate(p, placement, {"name": "lean_mlp4"})
        r = json.loads(p.read_text(encoding="utf-8"))
        assert r["look"] == "x" and r["placement"]["where"] == "host" and r["host_wrapper"]["look"]["name"] == "lean_mlp4"
    try:
        sys.argv = ["lean_host.py", "lean_mlp9"]
        main()
        raise AssertionError("an unknown look must be refused")
    except SystemExit as e:
        assert "not one of" in str(e)
    print("selftest: thread and --out parsing; the substitution gives package_root = the mirror root in memory, the "
          "placement is the host's and hard stops land in outputs/mp_unified; the looks import; annotate adds the "
          "placement; an unknown look is refused. all checks passed")


if __name__ == "__main__":
    main()
