"""Step 4f's coverage stage for one dataset, into its own file (outputs/step4f/budget_<dataset>.json), so datasets run
as parallel host jobs without sharing budget.json. The stage is scripts/step4f_pool_budget.py's, unchanged; only the
output path differs. The laptop merges the files into budget.json (merge), refusing a carve that differs from one it
already holds (hotpotqa, read on both machines, is the device control).

    python outputs/host_ops/pylib_run.py outputs/step4f/run_one.py DATASET [--threads 5]
    python outputs/step4f/run_one.py merge FILE...
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


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
    a = ap.parse_args()
    import step4f_pool_budget as F
    F.BUDGET = F.OUT / f"budget_{a.dataset}.json"
    return F.coverage_stage([a.dataset], a.threads)


if __name__ == "__main__":
    sys.exit(main())
