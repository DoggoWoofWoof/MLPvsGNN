"""SWP part C, the report (docs/SWP_S_TO_R_SWAP.md section 5): the pair trained with S in the slot (GNX and zrckpn, seed 0, both WQ splits), read with
S, R and nothing in the slot, beside the same pair trained with R in the slot (Parts A and B), seed 0 only, so that the two worlds compare the same four
fits' seeds and the same questions. Laptop only; reads the files fetched from the host. Nothing here trains or reads a model.

    python outputs/mp_unified/zswpcr.py files      # the host paths of every read the report needs, one per line (for rx fetch)
    python outputs/mp_unified/zswpcr.py report     # outputs/swpc/report.json and report.md
    python outputs/mp_unified/zswpcr.py selftest

Slots and arms. SQuAD: S = SNK, R = REC (the record), - = NK. MetaQA and WebQSP: S = REC (the record), R = RK, - = NK. MuSiQue, 2Wiki, HotpotQA:
S = carve b1 (the dataset's own S), R = b1w (the rule's links), - = b1t (empty slot); N and K are in every cell. A call is the 95% interval of a
paired bootstrap over the questions (2,000 resamples, zabl.compare): ABOVE, LEVEL or BELOW, always with the direction of the subtraction in the
row. S-trained = S in the slot while the fits train; R-trained = the rule's links in the slot (Part A for the three b1 datasets, Part B for the
rest) except SQuAD, where the record's own slot is R already. No threshold, no adoption call.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

DECLARED = "docs/SWP_S_TO_R_SWAP.md"
STAGE = "SWP-C-report"
OUT = ROOT / "outputs" / "swpc"
PAIR = ("scr-gnx-nx-s0", "scr-gnx-nx-s0-hp", "scr-ltm-zrckpn-s0", "scr-ltm-zrckpn-s0-hp")
GNX, MLP = "gnx", "zrckpn"
MODELS = (GNX, MLP)
SPLITS = ("WQ-musique", "WQ-hotpotqa")
WORLDS = ("S-trained", "R-trained")
DATASETS = ("musique", "2wiki", "hotpotqa", "squad", "metaqa", "webqsp")
B1DS = ("musique", "2wiki", "hotpotqa")
# slot -> (arm, carve) per dataset
SLOT_ARM = {ds: {"S": ("B1", "b1"), "R": ("B1W", "b1w"), "-": ("B1T", "b1t")} for ds in B1DS}
SLOT_ARM["squad"] = {"S": ("SNK", "s1eval"), "R": ("REC", "s1eval"), "-": ("NK", "s1eval")}
SLOT_ARM["metaqa"] = {"S": ("REC", "s1eval"), "R": ("RK", "s1eval"), "-": ("NK", "s1eval")}
SLOT_ARM["webqsp"] = dict(SLOT_ARM["metaqa"])
ARMS = tuple(dict.fromkeys(a for m in SLOT_ARM.values() for a, _c in m.values()))
BOOT_SEED = 20260706                                           # zabl.compare's


def model_of(fit):
    return GNX if "gnx" in fit else MLP


def split_of(fit):
    return "WQ-hotpotqa" if fit.endswith("-hp") else "WQ-musique"


def in_domain_splits(ds):
    """zabl.in_domain_splits (checked in the selftest): a WQ split is in-domain for the datasets its training set holds. WQ-hotpotqa trains on
    musique (s1fit), 2wiki, squad, webqsp; WQ-musique trains on hotpotqa, 2wiki, squad, webqsp. So MuSiQue: WQ-hotpotqa; HotpotQA: WQ-musique;
    2Wiki, SQuAD, WebQSP: both; MetaQA: neither."""
    return {"musique": ("WQ-hotpotqa",), "2wiki": SPLITS, "hotpotqa": ("WQ-musique",), "squad": SPLITS, "webqsp": SPLITS, "metaqa": ()}[ds]


def regimes(ds):
    ind = in_domain_splits(ds)
    out = []
    for regime in ("in-domain", "held-out"):
        if [sp for sp in SPLITS if (sp in ind) == (regime == "in-domain")]:
            out.append(regime)
    return out


def fits_of(model, ds, regime):
    ind = in_domain_splits(ds)
    return [f for f in PAIR if model_of(f) == model and (split_of(f) in ind) == (regime == "in-domain")]


def read_file(world, arm, carve, fit, ds):
    """The local path of one read. S-trained: outputs/swpc (fetched from the S-world). R-trained: Part A's, B1H's, B1M's, B1P's or Part B's own."""
    if world == "S-trained":
        return OUT / "fits" / arm / fit / "reads" / f"{ds}__{carve}.npz"
    if world == "B1L-ref":                                      # B1L (568bf08): zrckpn of record on the LLM-built P edges (carve b1o)
        return ROOT / "outputs" / "b1l" / "fits" / "h" / fit / "reads" / f"{ds}__{carve}.npz"
    if arm in ("B1", "B1W", "B1T"):
        import zidpb as IB
        return IB.read_path(GNX if model_of(fit) == GNX else MLP, fit, carve, ds)
    return ROOT / "outputs" / "swp" / "fits" / arm / fit / "reads" / f"{ds}__{carve}.npz"


def all_paths():
    seen, out = set(), []
    for ds in DATASETS:
        for reg in regimes(ds):
            for world in WORLDS:
                for _slot, (arm, carve) in SLOT_ARM[ds].items():
                    for model in MODELS:
                        for f in fits_of(model, ds, reg):
                            p = read_file(world, arm, carve, f, ds)
                            if p not in seen:
                                seen.add(p)
                                out.append(p)
    return out


def load_cell(world, arm, carve, ds, model, regime):
    """The cell Z.compare takes: the fits' per-question R@5 (the read's own, via p3c_claim1.load_read), None when a read is missing."""
    import p3c_claim1 as P3
    got = []
    fits = fits_of(model, ds, regime)
    for f in fits:
        p = read_file(world, arm, carve, f, ds)
        if not p.exists():
            return None
        fd = p.parent.parent
        P3.CARVE = carve
        r = P3.load_read(fd.parent, fd.name, ds)
        got.append((r["ids"], r["gt"], r["m"][:, 0], r["sha256"]))
    if not fits:
        return None
    if any(not np.array_equal(g[0], got[0][0]) for g in got):
        raise SystemExit(f"SWPC: {world}/{arm}/{ds}: the fits' questions differ")
    return {"ids": got[0][0], "gt": got[0][1], "per_fit": [g[2] for g in got], "sha256": [g[3] for g in got], "fits": fits}


def point(c):
    ok = c["gt"] > 0
    return float(np.mean(np.stack(c["per_fit"]), 0)[ok].mean())


def cmp(Z, a, b):
    """b minus a on the questions with a gold, over the fits' mean: point, 95% interval (paired bootstrap over the questions, each resample
    the mean of its own question draw), the call, in how many fits b is below a. zabl.compare is NOT used: its bootstrap line averages
    every resample of a chunk of 100 into one number, so its 2,000 resamples are 20 values and its interval is about 10 times too
    narrow (found 8 Nov 18:45: musique, GNX, b1w - b1: zabl.compare gives +1.9 (+1.85, +1.93), the per-resample mean gives
    +1.9 (+1.03, +2.79))."""
    import p3c_claim1 as P3
    ok = a["gt"] > 0
    if not np.array_equal(a["ids"], b["ids"]):
        raise SystemExit("SWPC: two arms' questions differ")
    ma, mb = np.mean(np.stack(a["per_fit"]), 0)[ok], np.mean(np.stack(b["per_fit"]), 0)[ok]
    d = mb - ma
    boots = np.concatenate([d[idx].mean(1) for idx in P3.resamples(d.size, BOOT_SEED, Z.BOOT)])
    assert boots.size == Z.BOOT, boots.size
    lo, hi = np.quantile(boots, [0.025, 0.975])
    call = "BELOW" if hi < 0 else "ABOVE" if lo > 0 else "LEVEL"
    below = sum(int(np.mean((fb - fa)[ok]) < 0) for fa, fb in zip(a["per_fit"], b["per_fit"]))
    return {"a": float(ma.mean()), "b": float(mb.mean()), "diff": float(d.mean()), "ci": [float(lo), float(hi)], "call": call,
            "fits_below": below, "fits": len(a["per_fit"]), "questions": int(ok.sum())}


def mean_vec(c):
    ok = c["gt"] > 0
    return np.mean(np.stack(c["per_fit"]), 0)[ok]


def diff_of_diffs(Z, g1, m1, g2, m2):
    """(g2 - m2) - (g1 - m1) over the questions with a gold, the same paired bootstrap as Z.compare."""
    import p3c_claim1 as P3
    d = (mean_vec(g2) - mean_vec(m2)) - (mean_vec(g1) - mean_vec(m1))
    boots = np.concatenate([d[idx].mean(1) for idx in P3.resamples(d.size, BOOT_SEED, Z.BOOT)])
    assert boots.size == Z.BOOT, boots.size
    lo, hi = np.quantile(boots, [0.025, 0.975])
    call = "BELOW" if hi < 0 else "ABOVE" if lo > 0 else "LEVEL"
    return {"diff": float(d.mean()), "ci": [float(lo), float(hi)], "call": call, "questions": int(d.size)}


def build(get, Z):
    rec = {"declared_in": DECLARED, "stage": STAGE, "boot": Z.BOOT, "fits": list(PAIR),
           "slot_arm": {ds: {s: list(v) for s, v in m.items()} for ds, m in SLOT_ARM.items()}, "datasets": {}, "missing": []}
    for ds in DATASETS:
        d = {}
        for regime in regimes(ds):
            e = {"cells": {}, "swap": {}, "gap": {}, "gap_change": {}, "train_effect": {}}
            cells = {}
            for world in WORLDS:
                for slot, (arm, carve) in SLOT_ARM[ds].items():
                    for model in MODELS:
                        c = get(world, arm, carve, ds, model, regime)
                        if c is None:
                            rec["missing"].append(f"{regime}/{ds}/{world}/{slot}/{arm}/{model}")
                            continue
                        cells[(world, slot, model)] = c
                        e["cells"][f"{world}/{slot}/{model}"] = {"arm": arm, "carve": carve, "R@5": round(point(c), 4),
                                                                  "per_fit": [round(float(np.mean(x[c["gt"] > 0])), 4) for x in c["per_fit"]],
                                                                  "fits": c["fits"]}
            for world in WORLDS:
                for model in MODELS:
                    for a_, b_ in (("S", "R"), ("-", "R"), ("-", "S")):
                        if (world, a_, model) in cells and (world, b_, model) in cells:
                            e["swap"][f"{world}/{model}/{b_} - {a_}"] = cmp(Z, cells[(world, a_, model)], cells[(world, b_, model)])
                for slot in SLOT_ARM[ds]:
                    if (world, slot, GNX) in cells and (world, slot, MLP) in cells:
                        e["gap"][f"{world}/{slot}"] = cmp(Z, cells[(world, slot, MLP)], cells[(world, slot, GNX)])
                for a_, b_ in (("S", "R"), ("-", "S"), ("-", "R")):
                    ks = [(world, s, m) for s in (a_, b_) for m in MODELS]
                    if all(k in cells for k in ks):
                        e["gap_change"][f"{world}/{b_} vs {a_}"] = diff_of_diffs(
                            Z, cells[(world, a_, GNX)], cells[(world, a_, MLP)], cells[(world, b_, GNX)], cells[(world, b_, MLP)])
            for slot in SLOT_ARM[ds]:
                for model in MODELS:
                    ka, kb = ("R-trained", slot, model), ("S-trained", slot, model)
                    if ka in cells and kb in cells:
                        e["train_effect"][f"{slot}/{model}"] = cmp(Z, cells[ka], cells[kb])
            d[regime] = e
        rec["datasets"][ds] = d
    rec["b1l_reference"] = {}
    for ds in ("musique", "2wiki"):
        for regime in regimes(ds):
            cells = {}
            for tag, (world, arm, carve) in {"b1t": ("R-trained", "B1T", "b1t"), "b1w": ("R-trained", "B1W", "b1w"),
                                             "b1": ("R-trained", "B1", "b1"), "b1o": ("B1L-ref", "b1o", "b1o")}.items():
                c = get(world, arm, carve, ds, MLP, regime)
                if c is not None:
                    cells[tag] = c
            if "b1o" not in cells:
                continue
            e = {"R@5": {k: round(point(c), 4) for k, c in cells.items()}, "steps": {}}
            for a_ in ("b1t", "b1w", "b1"):
                if a_ in cells:
                    e["steps"][f"b1o - {a_}"] = cmp(Z, cells[a_], cells["b1o"])
            rec["b1l_reference"][f"{ds}/{regime}"] = e
    return rec


# ── the report ───────────────────────────────────────────────────────────────

CALL = {"ABOVE": "{b} above {a}", "LEVEL": "level", "BELOW": "{b} below {a}"}


def _p(c):
    return "--" if c is None else f"{100 * c['R@5']:.1f}"


def _d(c, a, b):
    """b minus a in points with its interval and the call in words that name the direction."""
    if c is None:
        return "--"
    return f"{100 * c['diff']:+.1f} ({100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}) {CALL[c['call']].format(a=a, b=b)}"


def write_md(rec, path):
    L = ["# SWP part C: the pair trained with S in the slot, read with S, R and nothing (N and K in every cell)", "",
         f"Declared in {DECLARED}. R@5 in points, seed 0 only (GNX and zrckpn, the WQ-musique and WQ-hotpotqa fits; the in-domain regime uses "
         f"the fits whose split holds the dataset, the held-out regime the others), on each dataset's 1,000-question carve. Differences are "
         f"b minus a with the 95% interval of a paired bootstrap over the questions ({rec['boot']} resamples); the direction is named in every "
         f"call. S-trained: S in the slot while the fits train. R-trained: the same fits trained with the rule's links in the slot (Part A for "
         f"MuSiQue, 2Wiki, HotpotQA; Part B for the rest; SQuAD's record slot is R already), seed 0, so the two worlds compare the same seeds. "
         f"MuSiQue and SQuAD: S is itself a rule (title mentions), so there the swap is rule against rule, not rule against structure. "
         f"No threshold, no adoption call.", "",
         "## Reads in each slot", "",
         "| dataset | regime | model | trained with | S | R | - | R vs S (R - S) | S vs none (S - none) | R vs none (R - none) |",
         "|---|---|---|---|---:|---:|---:|---|---|---|"]
    for ds, d in rec["datasets"].items():
        for regime, e in d.items():
            for model in MODELS:
                for world in WORLDS:
                    if not any(f"{world}/{s}/{model}" in e["cells"] for s in ("S", "R", "-")):
                        continue
                    sw = e["swap"]
                    L.append(f"| {ds} | {regime} | {model} | {world} | " + " | ".join(_p(e["cells"].get(f"{world}/{s}/{model}")) for s in ("S", "R", "-"))
                             + f" | {_d(sw.get(f'{world}/{model}/R - S'), 'S', 'R')} | {_d(sw.get(f'{world}/{model}/S - -'), 'none', 'S')}"
                             + f" | {_d(sw.get(f'{world}/{model}/R - -'), 'none', 'R')} |")
    L += ["", "## Does the S slot in training change what a read gains from S? Same slot, S-trained minus R-trained", "",
          "| dataset | regime | slot read | model | R-trained | S-trained | S-trained - R-trained |", "|---|---|---|---|---:|---:|---|"]
    for ds, d in rec["datasets"].items():
        for regime, e in d.items():
            for slot in ("S", "R", "-"):
                for model in MODELS:
                    t = e["train_effect"].get(f"{slot}/{model}")
                    if t is None:
                        continue
                    L.append(f"| {ds} | {regime} | {slot} | {model} | {100 * t['a']:.1f} | {100 * t['b']:.1f} | {_d(t, 'R-trained', 'S-trained')} |")
    L += ["", "## Do we need message passing? GNX minus zrckpn on the same inputs, in each slot", "",
          "| dataset | regime | trained with | S | R | - | gap change R vs S | gap change S vs none |", "|---|---|---|---|---|---|---|---|"]
    for ds, d in rec["datasets"].items():
        for regime, e in d.items():
            for world in WORLDS:
                if not any(k.startswith(world) for k in e["gap"]):
                    continue
                g, gc = e["gap"], e["gap_change"]
                L.append(f"| {ds} | {regime} | {world} | " + " | ".join(_d(g.get(f"{world}/{s}"), "zrckpn", "GNX") for s in ("S", "R", "-"))
                         + f" | {_d(gc.get(f'{world}/R vs S'), 'S slot', 'R slot')} | {_d(gc.get(f'{world}/S vs -'), 'none', 'S slot')} |")
    if rec.get("b1l_reference"):
        L += ["", "## Reference only: B1L's LLM-built P edges (carve b1o) beside the slots, zrckpn trained with R in the slot, seed 0", "",
              "B1L (568bf08) read the models of record on these edges; GNX was not read on them, so there is no S-trained or GNX cell. The b1o entries "
              "are NULL-heavy (most have no W2_S type) while the fits were trained on W2_S types, so b1o is a lower bound for an LLM-built graph. "
              "This block says nothing about which graph is better.", "",
              "| dataset | regime | b1t | b1w (R) | b1 (S) | b1o (LLM edges) | b1o vs none (b1o - b1t) | b1o vs R (b1o - b1w) | b1o vs S (b1o - b1) |", "|---|---|---:|---:|---:|---:|---|---|---|"]
        for k, e in rec["b1l_reference"].items():
            ds, regime = k.split("/")
            st = e["steps"]
            L.append(f"| {ds} | {regime} | " + " | ".join("--" if t_ not in e["R@5"] else f"{100 * e['R@5'][t_]:.1f}" for t_ in ("b1t", "b1w", "b1", "b1o"))
                     + f" | {_d(st.get('b1o - b1t'), 'b1t', 'b1o')} | {_d(st.get('b1o - b1w'), 'b1w', 'b1o')} | {_d(st.get('b1o - b1'), 'b1', 'b1o')} |")
    if rec["missing"]:
        L += ["", f"Missing cells: {len(rec['missing'])}"] + [f"- {m}" for m in rec["missing"][:40]]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def report():
    import zabl as Z
    rec = build(lambda w, a, c, ds, m, r: load_cell(w, a, c, ds, m, r), Z)
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
    assert len(PAIR) == 4 and set(map(model_of, PAIR)) == set(MODELS) and [split_of(f) for f in PAIR].count("WQ-hotpotqa") == 2
    assert regimes("metaqa") == ["held-out"] and regimes("musique") == ["in-domain", "held-out"] and regimes("squad") == ["in-domain"]
    assert regimes("2wiki") == ["in-domain"]
    assert fits_of(GNX, "musique", "in-domain") == ["scr-gnx-nx-s0-hp"] and fits_of(MLP, "hotpotqa", "in-domain") == ["scr-ltm-zrckpn-s0"]
    assert len(fits_of(GNX, "squad", "in-domain")) == 2 and len(fits_of(MLP, "metaqa", "held-out")) == 2 and len(fits_of(MLP, "2wiki", "in-domain")) == 2
    for ds in DATASETS:
        assert set(in_domain_splits(ds)) == set(Z.in_domain_splits(ds)), (ds, in_domain_splits(ds), Z.in_domain_splits(ds))
    for ds, m in SLOT_ARM.items():
        assert set(m) == {"S", "R", "-"} and len({a for a, _c in m.values()}) == 3, ds
    n = 400
    rng = np.random.default_rng(1)
    ids = np.array([f"q{i}" for i in range(n)])
    gt = np.full(n, 2)
    base = rng.random(n)

    def fake(w, a, c, ds, m, r):
        shift = {"B1": 0.10, "B1W": 0.05, "B1T": 0.0, "SNK": 0.1, "REC": 0.05, "RK": 0.05, "NK": 0.0, "b1o": 0.03}[a] + (0.02 if w == "S-trained" else 0.0)
        shift += 0.03 if m == GNX else 0.0
        x = np.clip(base + shift + rng.normal(0, 0.05, n), 0, 1)
        return {"ids": ids, "gt": gt, "per_fit": [x] * len(fits_of(m, ds, r)), "sha256": ["x"], "fits": fits_of(m, ds, r)}
    rec = build(fake, Z)
    assert not rec["missing"], rec["missing"][:3]
    e = rec["datasets"]["musique"]["in-domain"]
    assert e["swap"]["S-trained/gnx/R - S"]["call"] == "BELOW" and e["swap"]["S-trained/gnx/R - S"]["diff"] < 0
    assert e["gap"]["R-trained/S"]["call"] in ("ABOVE", "LEVEL") and "S-trained/R vs S" in e["gap_change"]
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        write_md(rec, Path(t) / "r.md")
        txt = (Path(t) / "r.md").read_text(encoding="utf-8")
    assert "R below S" in txt and "S-trained - R-trained" in txt and "metaqa" in txt and "Reference only" in txt
    assert set(rec["b1l_reference"]) == {"musique/in-domain", "musique/held-out", "2wiki/in-domain"}
    print("selftest OK")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"report": report, "files": files, "selftest": selftest}.get(cmd, lambda: print(__doc__) or 2)())
