"""SWP part E, the report (docs/SWP_S_TO_R_SWAP.md section 5f): the pair trained with the slot EMPTY (N + K only; GNX and zrckpn, seed 0, both WQ
splits), read with nothing, R and S in the slot, beside the same pair trained with R in the slot (Parts A and B, the record) and with S (Part C).
Laptop only; reads the files fetched from the host. Nothing here trains or reads a model.

    python outputs/mp_unified/zswpe_rep.py files      # the host paths of every read the report needs, one per line (for rx fetch)
    python outputs/mp_unified/zswpe_rep.py report     # outputs/swpe/report.json and report.md
    python outputs/mp_unified/zswpe_rep.py selftest

Slots, arms, regimes, fits and the paired bootstrap are zswpcr.py's (Part C's report, not edited; its per-resample bootstrap, never zabl.compare).
N-trained reads: outputs/swpe/fits/<arm>/<fit>/reads/<ds>__<carve>.npz (swpe_collect.py). What is compared (b minus a, 95% interval, the call with
the direction named; no threshold, no adoption call):
  (a) the diagonal: R-trained read R minus N-trained read empty; S-trained read S minus N-trained read empty;
  (b) the training effect at the empty slot: N-trained minus R-trained (and minus S-trained), all read empty;
  (c) GNX minus zrckpn in every N-trained cell, and the gap change from N-trained/empty to each diagonal cell;
  (d) beside, no claim: N-trained read R or S minus N-trained read empty; R-trained read R minus N-trained read R; S likewise.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zswpcr as CR  # noqa: E402  (Part C's report: slots, fits, loader and bootstrap; not edited)

np = CR.np
DECLARED = "docs/SWP_S_TO_R_SWAP.md"
STAGE = "SWP-E-report"
OUT = ROOT / "outputs" / "swpe"
N, R_, S_ = "N-trained", "R-trained", "S-trained"
WORLDS = (N, R_, S_)
SLOTS = ("-", "R", "S")
GNX, MLP, MODELS = CR.GNX, CR.MLP, CR.MODELS
_part_c_read_file = CR.read_file


def read_file(world, arm, carve, fit, ds):
    """The local path of one read: N-trained from outputs/swpe (fetched from the N-world), the others where Part C's report finds them."""
    if world == N:
        return OUT / "fits" / arm / fit / "reads" / f"{ds}__{carve}.npz"
    return _part_c_read_file(world, arm, carve, fit, ds)


CR.read_file = read_file                                       # CR.load_cell looks it up at call time


def all_paths():
    seen, out = set(), []
    for ds in CR.DATASETS:
        for reg in CR.regimes(ds):
            for world in WORLDS:
                for _slot, (arm, carve) in CR.SLOT_ARM[ds].items():
                    for model in MODELS:
                        for f in CR.fits_of(model, ds, reg):
                            p = read_file(world, arm, carve, f, ds)
                            if p not in seen:
                                seen.add(p)
                                out.append(p)
    return out


def build(get, Z):
    rec = {"declared_in": DECLARED, "stage": STAGE, "boot": Z.BOOT, "fits": list(CR.PAIR), "worlds": list(WORLDS),
           "slot_arm": {ds: {s: list(v) for s, v in m.items()} for ds, m in CR.SLOT_ARM.items()}, "datasets": {}, "missing": []}
    for ds in CR.DATASETS:
        d = {}
        for regime in CR.regimes(ds):
            e = {"cells": {}, "diag": {}, "train_empty": {}, "gap": {}, "gap_change": {}, "transfer": {}}
            cells = {}
            for world in WORLDS:
                for slot, (arm, carve) in CR.SLOT_ARM[ds].items():
                    for model in MODELS:
                        c = get(world, arm, carve, ds, model, regime)
                        if c is None:
                            rec["missing"].append(f"{regime}/{ds}/{world}/{slot}/{arm}/{model}")
                            continue
                        cells[(world, slot, model)] = c
                        e["cells"][f"{world}/{slot}/{model}"] = {"arm": arm, "carve": carve, "R@5": round(CR.point(c), 4),
                                                                  "per_fit": [round(float(np.mean(x[c["gt"] > 0])), 4) for x in c["per_fit"]],
                                                                  "fits": c["fits"]}

            def step(key, block, a, b):
                if a in cells and b in cells:
                    e[block][key] = CR.cmp(Z, cells[a], cells[b])

            for m in MODELS:
                step(f"{m}/R-trained R - N-trained -", "diag", (N, "-", m), (R_, "R", m))                  # (a)
                step(f"{m}/S-trained S - N-trained -", "diag", (N, "-", m), (S_, "S", m))
                step(f"{m}/N-trained - - R-trained -", "train_empty", (R_, "-", m), (N, "-", m))           # (b)
                step(f"{m}/N-trained - - S-trained -", "train_empty", (S_, "-", m), (N, "-", m))
                step(f"{m}/N-trained R - N-trained -", "transfer", (N, "-", m), (N, "R", m))               # (d)
                step(f"{m}/N-trained S - N-trained -", "transfer", (N, "-", m), (N, "S", m))
                step(f"{m}/R-trained R - N-trained R", "transfer", (N, "R", m), (R_, "R", m))
                step(f"{m}/S-trained S - N-trained S", "transfer", (N, "S", m), (S_, "S", m))
            for world in WORLDS:                                                                        # (c)
                for slot in SLOTS:
                    step(f"{world}/{slot}", "gap", (world, slot, MLP), (world, slot, GNX))
            for world, slot in ((R_, "R"), (S_, "S")):
                ks = [(N, "-", GNX), (N, "-", MLP), (world, slot, GNX), (world, slot, MLP)]
                if all(k in cells for k in ks):
                    e["gap_change"][f"{world} {slot} vs N-trained -"] = CR.diff_of_diffs(Z, *[cells[k] for k in ks])
            d[regime] = e
        rec["datasets"][ds] = d
    return rec


# ── the report ───────────────────────────────────────────────────────────────

_p, _d = CR._p, CR._d


def write_md(rec, path):
    L = ["# SWP part E: the pair trained with the slot EMPTY (N and K only), read with nothing, R and S; beside the R-trained and S-trained pair", "",
         f"Declared in {DECLARED} (section 5f). R@5 in points, seed 0 only (GNX and zrckpn, the WQ-musique and WQ-hotpotqa fits; the in-domain "
         f"regime uses the fits whose split holds the dataset, the held-out regime the others), on each dataset's carve. Differences are b minus a "
         f"with the 95% interval of a paired bootstrap over the questions ({rec['boot']} resamples, one mean per resample); the direction is named in "
         f"every call. N-trained: the slot empty while the fits train (Part E). R-trained: the rule's links in the slot (Parts A and B; SQuAD's record "
         f"slot is R already). S-trained: the dataset's own S in the slot (Part C). N (rule NER links) and K (kNN) are in every cell, so 'empty' means "
         f"no structural slot, not no graph. MuSiQue and SQuAD: S is itself a rule (title mentions). No threshold, no adoption call.", "",
         "## Reads in each slot", "",
         "| dataset | regime | model | trained with | - | R | S |", "|---|---|---|---|---:|---:|---:|"]
    for ds, d in rec["datasets"].items():
        for regime, e in d.items():
            for model in MODELS:
                for world in WORLDS:
                    if any(f"{world}/{s}/{model}" in e["cells"] for s in SLOTS):
                        L.append(f"| {ds} | {regime} | {model} | {world} | " + " | ".join(_p(e["cells"].get(f"{world}/{s}/{model}")) for s in SLOTS) + " |")
    L += ["", "## What the slot adds when the model trains with it (the diagonal), and the read-time removal beside it", "",
          "| dataset | regime | model | N-trained, empty | R-trained, R | S-trained, S | R-trained R - N-trained empty | S-trained S - N-trained empty "
          "| N-trained empty - R-trained empty |", "|---|---|---|---:|---:|---:|---|---|---|"]
    for ds, d in rec["datasets"].items():
        for regime, e in d.items():
            for m in MODELS:
                c = e["cells"]
                L.append(f"| {ds} | {regime} | {m} | {_p(c.get(f'{N}/-/{m}'))} | {_p(c.get(f'{R_}/R/{m}'))} | {_p(c.get(f'{S_}/S/{m}'))} | "
                         f"{_d(e['diag'].get(f'{m}/R-trained R - N-trained -'), 'N-trained empty', 'R-trained R')} | "
                         f"{_d(e['diag'].get(f'{m}/S-trained S - N-trained -'), 'N-trained empty', 'S-trained S')} | "
                         f"{_d(e['train_empty'].get(f'{m}/N-trained - - R-trained -'), 'R-trained empty', 'N-trained empty')} |")
    L += ["", "## Do we need message passing? GNX minus zrckpn on the same inputs", "",
          "| dataset | regime | N-trained, empty | N-trained, R | N-trained, S | R-trained, R | S-trained, S | gap change R-trained R vs N-trained empty "
          "| gap change S-trained S vs N-trained empty |", "|---|---|---|---|---|---|---|---|---|"]
    for ds, d in rec["datasets"].items():
        for regime, e in d.items():
            g, gc = e["gap"], e["gap_change"]
            L.append(f"| {ds} | {regime} | " + " | ".join(_d(g.get(k), "zrckpn", "GNX") for k in (f"{N}/-", f"{N}/R", f"{N}/S", f"{R_}/R", f"{S_}/S"))
                     + f" | {_d(gc.get(f'{R_} R vs N-trained -'), 'N-trained empty', 'R-trained R')} | {_d(gc.get(f'{S_} S vs N-trained -'), 'N-trained empty', 'S-trained S')} |")
    L += ["", "## Beside, no claim: transfer to a slot the fit never trained with", "",
          "| dataset | regime | model | N-trained R - N-trained empty | N-trained S - N-trained empty | R-trained R - N-trained R | S-trained S - N-trained S |",
          "|---|---|---|---|---|---|---|"]
    for ds, d in rec["datasets"].items():
        for regime, e in d.items():
            for m in MODELS:
                t = e["transfer"]
                L.append(f"| {ds} | {regime} | {m} | {_d(t.get(f'{m}/N-trained R - N-trained -'), 'empty', 'R')} | "
                         f"{_d(t.get(f'{m}/N-trained S - N-trained -'), 'empty', 'S')} | {_d(t.get(f'{m}/R-trained R - N-trained R'), 'N-trained', 'R-trained')} | "
                         f"{_d(t.get(f'{m}/S-trained S - N-trained S'), 'N-trained', 'S-trained')} |")
    if rec["missing"]:
        L += ["", f"Missing cells: {len(rec['missing'])}"] + [f"- {m}" for m in rec["missing"][:60]]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def report():
    import zabl as Z
    rec = build(lambda w, a, c, ds, m, r: CR.load_cell(w, a, c, ds, m, r), Z)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    write_md(rec, OUT / "report.md")
    print(f"report: {OUT / 'report.md'}; missing {len(rec['missing'])}")
    return 0


def files():
    for p in all_paths():
        print(p.relative_to(ROOT).as_posix())
    return 0


def selftest():
    import zabl as Z
    assert CR.read_file is read_file and read_file(N, "B1T", "b1t", CR.PAIR[0], "musique") == OUT / "fits" / "B1T" / CR.PAIR[0] / "reads" / "musique__b1t.npz"
    assert read_file(S_, "B1", "b1", CR.PAIR[0], "musique") == CR.OUT / "fits" / "B1" / CR.PAIR[0] / "reads" / "musique__b1.npz"
    assert OUT != CR.OUT and len(all_paths()) > 0
    n = 500
    rng = np.random.default_rng(3)
    ids = np.array([f"q{i}" for i in range(n)])
    gt = np.full(n, 2)
    base = rng.random(n)
    lift = {"-": 0.0, "R": 0.06, "S": 0.08}

    def fake(w, a, c, ds, m, r):
        slot = next(s for s, (arm, _c) in CR.SLOT_ARM[ds].items() if arm == a)
        shift = lift[slot] + (0.02 if m == GNX else 0.0)
        if w == N:
            shift += 0.04 if slot == "-" else -0.02                # trained without the slot: better when read empty, worse with a slot it never saw
        x = np.clip(base + shift + rng.normal(0, 0.03, n), 0, 1)
        fits = CR.fits_of(m, ds, r)
        return {"ids": ids, "gt": gt, "per_fit": [x] * len(fits), "sha256": ["x"], "fits": fits}

    rec = build(fake, Z)
    assert not rec["missing"], rec["missing"][:3]
    e = rec["datasets"]["2wiki"]["in-domain"]
    assert e["train_empty"]["gnx/N-trained - - R-trained -"]["call"] == "ABOVE"
    assert e["diag"]["zrckpn/R-trained R - N-trained -"]["call"] == "ABOVE" and 0.0 < e["diag"]["zrckpn/R-trained R - N-trained -"]["diff"] < 0.06
    assert e["gap"][f"{N}/-"]["call"] == "ABOVE" and f"{R_} R vs N-trained -" in e["gap_change"]
    assert e["transfer"]["gnx/R-trained R - N-trained R"]["call"] == "ABOVE"
    assert set(rec["datasets"]["metaqa"]) == {"held-out"} and set(rec["datasets"]["musique"]) == {"in-domain", "held-out"}
    part = build(lambda w, a, c, ds, m, r: None if w == S_ else fake(w, a, c, ds, m, r), Z)   # Part C not filed yet: S-trained cells missing
    assert part["missing"] and all("/S-trained/" in x for x in part["missing"])
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        write_md(rec, Path(t) / "r.md")
        txt = (Path(t) / "r.md").read_text(encoding="utf-8")
        write_md(part, Path(t) / "p.md")
    assert "N-trained empty above R-trained empty" in txt and "R-trained R above N-trained empty" in txt and "GNX above zrckpn" in txt
    print("zswpe_rep self-test: OK")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"report": report, "files": files, "selftest": selftest}.get(cmd, lambda: print(__doc__) or 2)())
