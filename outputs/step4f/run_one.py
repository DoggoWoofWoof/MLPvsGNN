"""Step 4f's coverage stage for one dataset, into its own file (outputs/step4f/budget_<dataset>.json), so datasets run
as parallel host jobs without sharing budget.json. The stage is scripts/step4f_pool_budget.py's, unchanged; only the
output path differs. The laptop merges the files into budget.json (merge), refusing a carve that differs from one it
already holds (hotpotqa, read on both machines, is the device control).

    python outputs/host_ops/pylib_run.py outputs/step4f/run_one.py DATASET [--threads 5] [--host]
    python outputs/step4f/run_one.py merge FILE...
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def on_the_mirror(F):
    """--host: the host mirror in place of the package, as every host look reads it. mp_approx_six_base_score's
    host_mode checks the declared mirror root and the pinned VERIFIED records (refusing otherwise). Then, in memory
    only: step 4's load_script hands back an m3b_compile whose open_package reads a config copy with package_root = that
    root (its freeze check still runs), and the opened config itself names the root."""
    import copy
    import mp_approx_six_base as SB
    import mp_approx_six_base_score as S6
    root = str(Path(S6.host_mode(SB.load_declaration(), print)["mirror_root"]))
    S4 = F.E.S4
    orig_load = S4.load_script

    def load_script(name, *args, **kwargs):
        m = orig_load(name, *args, **kwargs)
        if name == "m3b_compile" and not getattr(m, "_on_the_mirror", False):
            orig = m.open_package

            def open_package(cfg):
                c = copy.deepcopy(cfg)
                c["substrate"]["package_root"] = root
                return orig(c)
            m.open_package = open_package
            m._on_the_mirror = True
        return m
    S4.load_script = load_script
    base = S4.Opened

    class OpenedOnTheMirror(base):
        def __init__(self):
            super().__init__()
            self.cfg["substrate"]["package_root"] = root
    S4.Opened = OpenedOnTheMirror


def merge(files):
    import step4f_pool_budget as F
    rec = json.loads(F.BUDGET.read_text(encoding="utf-8"))
    for f in files:
        part = json.loads(Path(f).read_text(encoding="utf-8"))
        if part["script_sha256"] != rec["script_sha256"] or part["freeze_RECORD_SHA256"] != rec["freeze_RECORD_SHA256"]:
            raise SystemExit(f"{f}: another script or freeze")
        for c, ent in part["carves"].items():
            if c in rec["carves"]:
                a, b = rec["carves"][c]["orders"], ent["orders"]
                same = a == b
                print(f"{c}: on both machines, orders {'IDENTICAL' if same else 'DIFFER'}")
                if not same:
                    raise SystemExit(f"{c}: the host's curve differs from the laptop's")
                continue
            ent = dict(ent, machine="host")
            rec["carves"][c] = ent
    F.write_json(F.BUDGET, rec)
    print("merged:", sorted(rec["carves"]))
    return 0


def main():
    if sys.argv[1:2] == ["merge"]:
        return merge(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args()
    import step4f_pool_budget as F
    if a.host:
        on_the_mirror(F)
    F.BUDGET = F.OUT / f"budget_{a.dataset}.json"
    return F.coverage_stage([a.dataset], a.threads)


if __name__ == "__main__":
    sys.exit(main())
