"""Step 1 looks (docs/STEP1_MATCHED_SELECTION.md, section 3): look_x_six's scoring pass, unchanged, on the carves of
outputs/step1/carves.json, so the lean MLP of the step can read them:
    s1sel   a training dataset's matched select carve (train split)
    s1fit   musique's regrouped fit carve (train split)
    s1eval  a dataset's eval read: its M3B eval population (webqsp's train_holdout included; metaqa every 4th question)
look_x_six.py is imported and run as it is. Two of its names are bound here before its main runs:
    P12.carve_ids_of     a step-1 carve name returns that carve's ids from carves.json, after checking their sha256
                         against the record's; any other name is mp_approx_l12's rule, as before
    carve_population     s1eval takes its questions' rows, cache rows and golds from m3b_compile.population's eval
                         population (checked against the digest carves.json recorded); a train-split carve is
                         look_x_six's own function
Everything else (the six pair's scores, the compiled columns, --full's edges and projections, the chunk layout, the
records) is look_x_six's. carves.json is checked against the sha256 pinned in the declaration before anything runs.

    python outputs/mp_unified/look_step1.py --dataset musique --carve s1eval --host --full [--shard i/n]
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import look_x_six as LX  # noqa: E402  (sets the thread variables before numpy and torch load)
import numpy as np  # noqa: E402

CARVES = ROOT / "outputs" / "step1" / "carves.json"
CARVES_SHA256 = "53cfb89f1e41a76257113719639f96318d14279f86b50a970f9a0bbf6686d746"   # docs/STEP1_MATCHED_SELECTION.md
KEYS = {"s1sel": "dselect", "s1fit": "fit", "s1eval": "eval"}
_ORIG_IDS = LX.P12.carve_ids_of
_ORIG_POP = LX.carve_population


def carves():
    raw = CARVES.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if CARVES_SHA256 is None or got != CARVES_SHA256:
        raise SystemExit(f"{CARVES}: sha256 {got} is not the declared {CARVES_SHA256}")
    return json.loads(raw.decode("utf-8"))


def step1_ids(rec, name, carve):
    entry = rec["per_dataset"].get(name, {}).get(KEYS[carve])
    if not entry or "ids" not in entry or entry.get("name") != carve:
        raise SystemExit(f"{name} has no step-1 carve {carve}")
    ids = list(entry["ids"])
    if hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest() != entry["sha256"] or len(ids) != entry["n"]:
        raise SystemExit(f"{name}/{carve}: the ids do not hash to the record's sha256")
    return ids


def install(rec, name):
    def carve_ids_of(carve, rule):
        if carve in KEYS:
            return step1_ids(rec, name, carve)
        return _ORIG_IDS(carve, rule)

    def carve_population(m3b_compile, ds, ids, kind, m3a, positions, ds_name):
        if kind != "s1eval":
            return _ORIG_POP(m3b_compile, ds, ids, kind, m3a, positions, ds_name)
        _cfg, cfg_m3b, cfg_h = LX.V2.load_configs()
        entry = rec["per_dataset"][ds_name]["eval"]
        split = cfg_m3b["populations"]["eval_splits"][ds_name]
        ev = m3b_compile.population(ds, ds_name, "eval", cfg_m3b, cfg_h, m3a, positions)
        if split != entry["split"] or ev.digest != entry["population_sha256"] or len(ev.ids) != entry["n_population"]:
            raise SystemExit(f"{ds_name}: M3B's eval population is not the one carves.json read")
        at = {q: i for i, q in enumerate(ev.ids)}
        sel = np.asarray([at[q] for q in ids], dtype=np.int64)
        row_of = {r["query_id"]: r for r in m3a.population_rows(ds, split, cfg_h)[1]}
        pop = m3b_compile.Population(ds_name, kind, list(ids), ev.idx[sel], [ev.golds[i] for i in sel], len(ids), 0,
                                     LX.m3b_pools.ids_digest(list(ids)))
        info = {q: {"type": str(row_of[q].get("type")), "level": str(row_of[q].get("level")),
                    "evidences": row_of[q].get("evidences") or [], "gold_refs": row_of[q].get("gold_refs") or []} for q in ids}
        return pop, info

    LX.P12.carve_ids_of = carve_ids_of
    LX.carve_population = carve_population


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    name = argv[argv.index("--dataset") + 1]
    carve = argv[argv.index("--carve") + 1]
    if carve not in KEYS:
        raise SystemExit(f"--carve {carve}: one of {sorted(KEYS)} (other carves are look_x_six's own)")
    if "--full" not in argv:
        raise SystemExit("a step-1 carve has no earlier look to join: run it with --full")
    install(carves(), name)
    return LX.main(argv)


if __name__ == "__main__":
    main()
