"""PH1 (docs/PH1_PER_HOP_READS.md): R@5 and FC@5 of the filed reads by number of hops or question kind. Descriptive, post hoc:
no verdict, no call, nothing trains or reads anew.

    python outputs/mp_unified/zph1.py selftest
    python outputs/mp_unified/zph1.py run        -> outputs/ph1/report.json, report.md
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from b1b import BOOT, SEED, boot_ci  # noqa: E402  (2,000 resamples, seed 0)

BENCH = ROOT / "outputs" / "bench" / "hipporag2"
OUT = ROOT / "outputs" / "ph1"
CAND = "p@swa"
SETTINGS = ("musique", "2wiki", "hotpotqa")
CARVES = ("b1w", "b1")
SEEDS = (0, 1, 2)
SUF = {"WQ-musique": "", "WQ-hotpotqa": "-hp"}                     # zb1h.SUF
IN_DOMAIN = {"musique": ("WQ-hotpotqa",), "hotpotqa": ("WQ-musique",), "2wiki": ("WQ-musique", "WQ-hotpotqa")}   # zb1h.IN_DOMAIN
HOPS = {"2hop": "2 hops", "3hop1": "3 hops", "3hop2": "3 hops", "4hop1": "4 hops", "4hop2": "4 hops", "4hop3": "4 hops"}
SIZES = {"musique": {"2 hops": 518, "3 hops": 316, "4 hops": 166},
         "2wiki": {"compositional": 413, "inference": 108, "comparison": 244, "bridge_comparison": 235},
         "hotpotqa": {"bridge": 811, "comparison": 189}}
PREFIX = {"zg6r": "scr-ml1-zg6r", "zrckpn": "scr-ltm-zrckpn", "gnx": "scr-gnx-nx"}
ML_ROWS = ("zg6r", "zrckpn", "gnx")
J5_ROWS = ("zrc", "zsp")
ROLE = {"zg6r": "GNN zg6r", "zrckpn": "MLP zrckpn", "gnx": "GNN GNX (same inputs)", "zrc": "J5 zrc (MLP)", "zsp": "J5 zsp (MLP + one step)"}
REC = {"b1h": ROOT / "outputs/b1h/report.json", "b1m": ROOT / "outputs/b1m/report.json", "idpa": ROOT / "outputs/idpa/report.json",
       "idpb": ROOT / "outputs/idpb/report.json", "j5s": ROOT / "outputs/j5s/report.json"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def group_of(ds, kind):
    return HOPS[kind] if ds == "musique" else kind


def ml_fits(arm, ds):
    return [f"{PREFIX[arm]}-s{sd}{SUF[sp]}" for sp in IN_DOMAIN[ds] for sd in SEEDS]


def read_path(arm, carve, fit, ds):
    if arm in J5_ROWS:
        if fit == "J5-s0":
            return ROOT / ("outputs/b1b/fits" if carve == "b1" else "outputs/b1f/fits/w") / arm / "J5" / "reads" / f"{ds}__{carve}.npz"
        return ROOT / "outputs/j5s/fits" / carve / arm / fit / "reads" / f"{ds}__{carve}.npz"
    base = {("zg6r", "b1"): "b1h/fits/h", ("zrckpn", "b1"): "b1h/fits/h", ("zg6r", "b1w"): "b1p/fits/w", ("zrckpn", "b1w"): "b1m/fits/w",
            ("gnx", "b1w"): "idpa/fits/w", ("gnx", "b1"): "idpb/fits/h"}[(arm, carve)]
    return ROOT / "outputs" / base / fit / "reads" / f"{ds}__{carve}.npz"


def fits_of(arm, ds):
    return [f"J5-s{sd}" for sd in SEEDS] if arm in J5_ROWS else ml_fits(arm, ds)


def seed0(arm, fits):
    return [f for f in fits if f in ("J5-s0",) or f.rsplit("-s", 1)[-1] in ("0", "0-hp")]


def load_read(p, ids, gold):
    with np.load(p) as z:
        qid = [str(x) for x in z["ids"]]
        gt = z["gold_total"].astype(np.int64)
        cands = [str(c) for c in z["candidates"]]
        refs = [str(r) for r in z["refs"]]
        top = z["top"][cands.index(CAND)].astype(np.int64)
        rrf = z["ref_top"][refs.index("rrf")].astype(np.int64)
    if qid != ids or not np.array_equal(gt, gold):
        raise SystemExit(f"G1: {p.relative_to(ROOT)} does not hold B1a's questions in B1a's order with B1a's gold counts")
    return top, rrf


def metrics(top, gold):
    top = np.asarray(top, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.float64)
    return top / gold, (top == gold).astype(np.float64)


def summarize(r5, fc, r5_s0, fc_s0, rrf_r5, rrf_fc, idx):
    out = {"n": int(idx.size)}
    for name, x, x0, ref in (("R@5", r5, r5_s0, rrf_r5), ("FC@5", fc, fc_s0, rrf_fc)):
        out[name] = round(float(x[idx].mean()), 4)
        out[name + "_ci"] = boot_ci(x[idx])
        if x0 is not None:
            out[name + "_s0"] = round(float(x0[idx].mean()), 4)
        if ref is not None:
            d = x[idx] - ref[idx]
            out["minus_rrf_" + name] = round(float(d.mean()), 4)
            out["minus_rrf_" + name + "_ci"] = boot_ci(d)
    return out


def groups(ds, kinds):
    g = {}
    for i, k in enumerate(kinds):
        g.setdefault(group_of(ds, k), []).append(i)
    out = {k: np.asarray(v, dtype=np.int64) for k, v in g.items()}
    if ds == "musique":
        for k in HOPS:
            out["kind " + k] = np.asarray([i for i, x in enumerate(kinds) if x == k], dtype=np.int64)
    out["all"] = np.arange(len(kinds), dtype=np.int64)
    return out


def pins():
    """file_sha256 per 'arm/carve/fit' from the filed records (G2); several records holding a key must agree."""
    pin = {}
    for name in ("b1h", "b1m", "idpa"):
        rec = json.loads(REC[name].read_text(encoding="utf-8"))
        for ds in SETTINGS:
            for k, v in rec["settings"][ds].get("fits", {}).items():
                if "file_sha256" in v:
                    pin.setdefault((ds, k), set()).add(v["file_sha256"])
    j5 = json.loads(REC["j5s"].read_text(encoding="utf-8"))
    for ds in SETTINGS:
        for carve in CARVES:
            for arm in J5_ROWS:
                for sd, h in enumerate(j5["settings"][ds]["carves"][carve][arm]["files_sha256"]):
                    pin.setdefault((ds, f"{arm}/{carve}/J5-s{sd}"), set()).add(h)
    bad = [k for k, v in pin.items() if len(v) > 1]
    if bad:
        raise SystemExit(f"G2: filed records disagree on {bad[:5]}")
    return {k: next(iter(v)) for k, v in pin.items()}


def filed_cells(ds):
    """(row key) -> {'R@5', optional 'FC@5', optional 'per_fit'} from the filed records (G3)."""
    out = {}
    b1h = json.loads(REC["b1h"].read_text(encoding="utf-8"))["settings"][ds]["in_domain"]
    for arm in ("zg6r", "zrckpn"):
        for carve in CARVES:
            out[f"{arm}/{carve}"] = b1h[f"{arm}/{carve}"]
    out["gnx/b1w"] = json.loads(REC["idpa"].read_text(encoding="utf-8"))["settings"][ds]["in_domain"]["gnx/b1w"]
    out["gnx/b1"] = json.loads(REC["idpb"].read_text(encoding="utf-8"))["settings"][ds]["in_domain"]["gnx/b1"]
    j5 = json.loads(REC["j5s"].read_text(encoding="utf-8"))["settings"][ds]["carves"]
    for carve in CARVES:
        for arm in J5_ROWS:
            c = j5[carve][arm]
            out[f"{arm}/{carve}"] = {"R@5": c["R@5"], "FC@5": c["FC@5"], "per_fit": c["seeds_R@5"]}
    return out


def same4(a, b):
    return round(float(a), 4) == round(float(b), 4) or abs(float(a) - float(b)) <= 5e-5 + 1e-12


def run_setting(ds, pin, log):
    q = json.loads((BENCH / ds / "queries.json").read_text(encoding="utf-8"))
    ids = [str(x["id"]) for x in q]
    gold = np.asarray([len(x["golds"]) for x in q], dtype=np.int64)
    kinds = [str(x["kind"]) for x in q]
    g = groups(ds, kinds)
    sizes = {k: int(v.size) for k, v in g.items() if k in SIZES[ds]}
    if sizes != SIZES[ds] or sum(sizes.values()) != len(q) or len(q) != 1000:
        raise SystemExit(f"G1: {ds} group sizes {sizes} are not {SIZES[ds]}")
    files, rrf, rows, checks = {}, None, {}, []
    cells = filed_cells(ds)
    for arm in ML_ROWS + J5_ROWS:
        for carve in CARVES:
            fits = fits_of(arm, ds)
            tops = {}
            for fit in fits:
                p = read_path(arm, carve, fit, ds)
                h = sha(p)
                key = f"{arm}/{carve}/{fit}"
                want = pin.get((ds, key))
                if want is None and not (arm == "gnx" and carve == "b1"):
                    raise SystemExit(f"G2: no filed sha256 for {ds} {key}")
                if want is not None and want != h:
                    raise SystemExit(f"G2: {p.relative_to(ROOT)} is not the file the record pins ({h[:12]} != {want[:12]})")
                files[key] = {"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": h, "pinned": want is not None}
                top, r = load_read(p, ids, gold)
                if rrf is None:
                    rrf = r
                elif not np.array_equal(r, rrf):
                    raise SystemExit(f"G1: {key}'s RRF row differs from the first read's ({ds})")
                tops[fit] = top
            per = [metrics(tops[f], gold) for f in fits]
            r5 = np.mean([m[0] for m in per], axis=0)
            fc = np.mean([m[1] for m in per], axis=0)
            s0 = seed0(arm, fits)
            per0 = [metrics(tops[f], gold) for f in s0]
            r5_0 = np.mean([m[0] for m in per0], axis=0)
            fc_0 = np.mean([m[1] for m in per0], axis=0)
            row = f"{arm}/{carve}"
            cell = cells[row]
            ok = same4(r5.mean(), cell["R@5"]) and ("FC@5" not in cell or same4(fc.mean(), cell["FC@5"]))
            if "per_fit" in cell:      # each fit's R@5, compared as sorted lists (the records' order is not needed)
                mine = sorted(float(m[0].mean()) for m in per)
                ok = ok and len(mine) == len(cell["per_fit"]) and all(same4(a, b) for a, b in zip(mine, sorted(cell["per_fit"])))
            checks.append({"row": row, "R@5": round(float(r5.mean()), 4), "filed_R@5": cell["R@5"],
                           "FC@5": round(float(fc.mean()), 4), "filed_FC@5": cell.get("FC@5"), "equal": bool(ok)})
            if not ok:
                raise SystemExit(f"G3: {ds} {row} R@5 {r5.mean():.4f} FC@5 {fc.mean():.4f} against the record's {cell}")
            rows[row] = (r5, fc, r5_0, fc_0, len(fits))
    rrf_r5, rrf_fc = metrics(rrf, gold)
    table = {"rrf": {k: summarize(rrf_r5, rrf_fc, None, None, None, None, v) for k, v in g.items()}}
    for row, (r5, fc, r5_0, fc_0, nf) in rows.items():
        table[row] = {k: summarize(r5, fc, r5_0, fc_0, rrf_r5, rrf_fc, v) for k, v in g.items()}
        table[row]["fits"] = nf
    log(f"{ds}: {len(rows)} rows, {len(files)} reads, G1-G3 held")
    return {"questions": len(q), "groups": {k: int(v.size) for k, v in g.items()}, "rows": table, "checks": checks, "reads": files}


def pct(x):
    return f"{100 * x:.1f}"


def ci(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def dpts(x, c):
    return f"{100 * x:+.1f} [{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"


def write_md(rec):
    L = ["# PH1: R@5 and FC@5 by number of hops (descriptive, post hoc)", "",
         f"Declared in {rec['declared_in']}; zph1.py sha256 {rec['zph1_sha256'][:16]}; {rec['utc']}. Means over the in-domain fits "
         "at three seeds (J5: the one fit's three seeds), seed 0 in parentheses; 95% intervals of 2,000 question resamples (seed 0). "
         "FC@5 = every gold passage in the top 5. No call: a group differs from another in its questions as well as its hops.", ""]
    for ds in SETTINGS:
        s = rec["settings"][ds]
        order = [k for k in SIZES[ds]] + ["all"]
        for metric in ("R@5", "FC@5"):
            L += [f"## {ds}: {metric}", "", "| row | " + " | ".join(f"{k} (n={s['groups'][k]})" for k in order) + " |",
                  "| --- |" + " ---: |" * len(order)]
            for row, t in s["rows"].items():
                name = "RRF" if row == "rrf" else f"{ROLE[row.split('/')[0]]}, {row.split('/')[1]}"
                cells = []
                for k in order:
                    c = t[k]
                    v = f"{pct(c[metric])} {ci(c[metric + '_ci'])}"
                    if metric + "_s0" in c:
                        v += f" ({pct(c[metric + '_s0'])})"
                    cells.append(v)
                L.append(f"| {name} | " + " | ".join(cells) + " |")
            L += ["", f"Minus RRF ({metric}, points, paired):", "", "| row | " + " | ".join(order) + " |", "| --- |" + " ---: |" * len(order)]
            for row, t in s["rows"].items():
                if row == "rrf":
                    continue
                name = f"{ROLE[row.split('/')[0]]}, {row.split('/')[1]}"
                L.append(f"| {name} | " + " | ".join(dpts(t[k]["minus_rrf_" + metric], t[k]["minus_rrf_" + metric + "_ci"]) for k in order) + " |")
            L.append("")
        if ds == "musique":
            kk = [k for k in s["groups"] if k.startswith("kind ")]
            L += ["MuSiQue by kind (R@5, beside):", "", "| row | " + " | ".join(f"{k[5:]} (n={s['groups'][k]})" for k in kk) + " |",
                  "| --- |" + " ---: |" * len(kk)]
            for row, t in s["rows"].items():
                name = "RRF" if row == "rrf" else f"{ROLE[row.split('/')[0]]}, {row.split('/')[1]}"
                L.append(f"| {name} | " + " | ".join(pct(t[k]["R@5"]) for k in kk) + " |")
            L.append("")
    L += ["## Gates", "", f"G1 (sizes, question order, gold counts, one RRF row), G2 (read sha256 against the filed records; GNX on b1 recorded "
          f"here), G3 (all-question R@5 and FC@5 equal to the filed records): {rec['gates']}.", ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8", newline="\n")


def run():
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)
    pin = pins()
    rec = {"stage": "PH1", "declared_in": "docs/PH1_PER_HOP_READS.md", "candidate": CAND, "boot": {"resamples": BOOT, "seed": SEED},
           "in_domain": {k: list(v) for k, v in IN_DOMAIN.items()}, "zph1_sha256": sha(Path(__file__)),
           "records_sha256": {k: sha(v) for k, v in REC.items()}, "settings": {}}
    for ds in SETTINGS:
        rec["settings"][ds] = run_setting(ds, pin, log)
    rec["gates"] = "held"
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rec, indent=1), encoding="utf-8", newline="\n")
    write_md(rec)
    log(f"wrote {OUT.relative_to(ROOT)}/report.json and report.md")


def selftest():
    # metrics: 2 golds, top 1 -> 0.5 / 0; 3 of 3 -> 1 / 1
    r5, fc = metrics([1, 3, 0], [2, 3, 4])
    assert np.allclose(r5, [0.5, 1.0, 0.0]) and np.allclose(fc, [0, 1, 0])
    # groups: musique kinds to hops, kinds beside, all
    g = groups("musique", ["2hop", "3hop1", "3hop2", "4hop3", "2hop"])
    assert list(g["2 hops"]) == [0, 4] and list(g["3 hops"]) == [1, 2] and list(g["4 hops"]) == [3]
    assert list(g["kind 3hop2"]) == [2] and list(g["all"]) == [0, 1, 2, 3, 4]
    g2 = groups("hotpotqa", ["bridge", "comparison", "bridge"])
    assert list(g2["bridge"]) == [0, 2] and "kind 2hop" not in g2
    # fits and seed 0: musique reads the -hp fits, 2wiki both splits
    assert ml_fits("zg6r", "musique") == ["scr-ml1-zg6r-s0-hp", "scr-ml1-zg6r-s1-hp", "scr-ml1-zg6r-s2-hp"]
    f2 = ml_fits("gnx", "2wiki")
    assert len(f2) == 6 and seed0("gnx", f2) == ["scr-gnx-nx-s0", "scr-gnx-nx-s0-hp"]
    assert seed0("zrc", fits_of("zrc", "musique")) == ["J5-s0"]
    # a row: two fits, per-question mean, then group means and the paired difference to RRF
    gold = np.array([2, 2, 3, 4])
    a, b = metrics([2, 1, 3, 2], gold), metrics([1, 1, 3, 4], gold)
    r5 = (a[0] + b[0]) / 2
    fc = (a[1] + b[1]) / 2
    assert np.allclose(r5, [0.75, 0.5, 1.0, 0.75]) and np.allclose(fc, [0.5, 0, 1, 0.5])
    rr5, rfc = metrics([1, 0, 2, 2], gold)
    s = summarize(r5, fc, a[0], a[1], rr5, rfc, np.array([0, 2]))
    assert s["n"] == 2 and s["R@5"] == 0.875 and s["FC@5"] == 0.75 and s["R@5_s0"] == 1.0
    assert s["minus_rrf_R@5"] == round(((0.75 - 0.5) + (1.0 - 2 / 3)) / 2, 4) and s["minus_rrf_FC@5"] == 0.75
    c = boot_ci(np.full(10, 0.3))
    assert c == [0.3, 0.3]
    assert same4(0.66255, 0.6626) and not same4(0.6620, 0.6626)
    # paths follow the records' layout
    assert read_path("zrc", "b1", "J5-s0", "musique").as_posix().endswith("outputs/b1b/fits/zrc/J5/reads/musique__b1.npz")
    assert read_path("zsp", "b1w", "J5-s2", "2wiki").as_posix().endswith("outputs/j5s/fits/b1w/zsp/J5-s2/reads/2wiki__b1w.npz")
    assert read_path("gnx", "b1", "scr-gnx-nx-s1", "hotpotqa").as_posix().endswith("outputs/idpb/fits/h/scr-gnx-nx-s1/reads/hotpotqa__b1.npz")
    print("selftest ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        selftest()
    elif cmd == "run":
        run()
    else:
        sys.exit(__doc__)
