"""J5S: seeds 1 and 2 of B1b's J5 fits, zrc (the MLP) and zsp (the MLP with one propagation step), read on B1's three
settings exactly as B1b, U1a and B1F read seed 0. Each arm also gets a seed-0 control fit and a re-read of its filed fit
(docs/J5S_J5_SEEDS.md).

    python outputs/mp_unified/j5s.py rows --queue outputs/host_ops/hostfeed_items.txt    (prints the feeder block)
    python outputs/mp_unified/j5s.py fork --arm zrc --seed 1                               (after that fit)
    python outputs/mp_unified/j5s.py report                                                -> outputs/j5s/report.json, .md
    python outputs/mp_unified/j5s.py --selftest

The fits and reads are the filed rows' own commands (fzk-train-J5, fsp-train-J5, b1-read2-zrc-J5, b1-read2-zsp-J5). Only
the name, the output folder and the seed change, and for reads the carve; `rows` makes each substitution exactly once and
checks it. Reservations are sized from the filed runs' measured peaks (RES); the commands' GPU caps are the filed ones.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import b1b as BB  # noqa: E402  (B1b's per_question, boot_ci, call, HIPPORAG2, CAND, BOOT, SEED, sha_file, write_json)
import b1f as BF  # noqa: E402  (B1F's PUBLISHED rows and best_published)

DECLARED = "docs/J5S_J5_SEEDS.md"
ARMS = ("zrc", "zsp")
SETTINGS = ("musique", "2wiki", "hotpotqa")
CARVES = ("b1", "b1w", "b1t", "b1u")   # the corpus hyperlinks; W2_S's rule links; text links only; U1a's title rule
SEEDS = ("0", "1", "2")
CONTROL = "0c"
SOURCE = {"zrc": "outputs/full_zrct/fits/J5", "zsp": "outputs/full_zsp/fits/J5"}
FILED_MODELS = {"zrc": "38ad128de6f9a6e41f8871e1f8d9fe94cae6d9400f7b5f23420e0364e9971127",
                "zsp": "966796eaf22e4d086c7c5a8a08df316e12265c13541a0942ce59541c1e0ca2fa"}
FILED_READ = {"b1": "outputs/b1b/fits/{arm}/J5/reads/{ds}__b1.npz",
              "b1w": "outputs/b1f/fits/w/{arm}/J5/reads/{ds}__b1w.npz",
              "b1t": "outputs/u1a/fits/t/{arm}/J5/reads/{ds}__b1t.npz",
              "b1u": "outputs/u1a/fits/u/{arm}/J5/reads/{ds}__b1u.npz"}
# the filed fit's settings every new fit must share (screen.json); only the seed differs
SAME = {"split": "J5", "train": "metaqa=fit,squad=fit,musique=s1fit,hotpotqa=fit,2wiki=fit", "basis": "2wiki",
        "variants": "p", "config": "2e-3:1e-4:0.1:8:2", "hidden": 128}
TEMPLATES = {("fit", "zrc"): "fzk-train-J5", ("fit", "zsp"): "fsp-train-J5",
             ("read", "zrc"): "b1-read2-zrc-J5", ("read", "zsp"): "b1-read2-zsp-J5"}
# memory (GB) to reserve, from the filed runs' peaks: zrc fit 2.4 GB resident (5.4 GB on the card), zsp fit 9.0 GB,
# zrc reads 1.2 GB, zsp reads 3.6-4.0 GB. CPU and GPU shares stay the templates' (the GPU share bounds the command's cap).
RES = {("fit", "zrc"): "4", ("fit", "zsp"): "12", ("read", "zrc"): "3", ("read", "zsp"): "6"}
OWN_INPUTS = "outputs/mp_unified/j5s.py;outputs/mp_unified/b1b.py;outputs/mp_unified/b1f.py;outputs/mp_unified/u1a.py"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_src(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def seed_int(seed):
    return int(seed.rstrip("c"))


def train_dir(root, arm, seed):
    return Path(root) / "outputs" / "j5s" / "train" / arm / f"J5-s{seed}"


def fit_dir(root, carve, arm, seed):
    return Path(root) / "outputs" / "j5s" / "fits" / carve / arm / f"J5-s{seed}"


def read_file(root, carve, arm, seed, ds):
    return fit_dir(root, carve, arm, seed) / "reads" / f"{ds}__{carve}.npz"


def carves_of(seed):
    return ("b1",) if seed == CONTROL else CARVES


# -- fork: copy a fit into each read carve's folder (its reads and read.json left out), models.pt sha-checked --------


def check_fit(src, arm, seed):
    sc = json.loads((Path(src) / "screen.json").read_text(encoding="utf-8"))
    want = dict(SAME, arm=arm, seed=seed_int(seed), rc=0)
    bad = {k: (sc.get(k), v) for k, v in want.items() if sc.get(k) != v}
    if bad:
        raise SystemExit(f"{src}: not B1b's J5 fit at seed {seed}: {bad}")


def fork(arm, seed, root=ROOT, filed=FILED_MODELS):
    if arm not in ARMS or seed not in SEEDS + (CONTROL,):
        raise SystemExit(f"fork: --arm one of {ARMS}, --seed one of {SEEDS + (CONTROL,)}")
    root = Path(root)
    src = root / SOURCE[arm] if seed == "0" else train_dir(root, arm, seed)
    check_fit(src, arm, seed)
    want = BB.sha_file(src / "models.pt")
    if seed == "0" and want != filed[arm]:
        raise SystemExit(f"{src}: models.pt is not B1b's filed fit ({want[:12]}, filed {filed[arm][:12]})")
    for carve in carves_of(seed):
        dst = fit_dir(root, carve, arm, seed)
        if (dst / "models.pt").exists():
            if BB.sha_file(dst / "models.pt") != want:
                raise SystemExit(f"{dst}: holds other models than {src}")
            log(f"{dst}: already a copy of {src}")
            continue
        tmp = dst.parent / (dst.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, tmp, ignore=lambda d, names: [n for n in names if Path(d) == src and n in ("reads", "read.json")])
        if BB.sha_file(tmp / "models.pt") != want:
            raise SystemExit(f"{tmp}: the copy's models.pt is not the source's")
        os.replace(tmp, dst)
        BB.write_json(dst / "fork.json", {"declared_in": DECLARED, "arm": arm, "seed": seed, "carve": carve,
                                          "source": src.relative_to(root).as_posix(), "models_sha256": want,
                                          "script_sha256": sha_src(__file__), "utc": utc()})
        log(f"{src} -> {dst} (models {want[:12]})")
    return 0


# -- rows: the feeder block, from the filed rows ------------------------------------------------------------------------


def _sub(s, old, new):
    if s.count(old) != 1:
        raise SystemExit(f"rows: {old!r} occurs {s.count(old)} times in {s!r}")
    return s.replace(old, new)


def templates(queue):
    rows = {}
    for ln in Path(queue).read_text(encoding="utf-8").splitlines():
        f = ln.split("|")
        if f[0] == "line" and len(f) >= 8 and f[3] in TEMPLATES.values():
            if f[3] in rows:
                raise SystemExit(f"rows: {f[3]} twice in {queue}")
            rows[f[3]] = f
    if set(rows) != set(TEMPLATES.values()):
        raise SystemExit(f"rows: {queue} lacks {sorted(set(TEMPLATES.values()) - set(rows))}")
    return rows


def fit_cmd(t, arm, seed):
    out = {"zrc": "outputs/full_zrct/fits", "zsp": "outputs/full_zsp/fits"}[arm]
    c = _sub(t, " --name J5 ", f" --name J5-s{seed} ")
    c = _sub(c, f" --out-root {out} ", f" --out-root outputs/j5s/train/{arm} ")
    return _sub(c, f" --arm {arm} ", f" --arm {arm} --seed {seed_int(seed)} ")


def read_cmd(t, arm, seed, carve):
    c = _sub(t, " --name J5 ", f" --name J5-s{seed} ")
    c = _sub(c, f" --out-root outputs/b1b/fits/{arm} ", f" --out-root outputs/j5s/fits/{carve}/{arm} ")
    return _sub(c, " --read musique=b1,2wiki=b1,hotpotqa=b1 ", f" --read musique={carve},2wiki={carve},hotpotqa={carve} ")


def block(queue):
    """The rows, GPU fits first: selftest -> fits -> forks -> reads -> report."""
    T = templates(queue)
    rows, reads = [], []

    def row(deps, name, cpu, mem, inputs, cmd, gpu=()):
        rows.append("|".join(["line", "x", deps, name, cpu, mem, inputs, cmd] + list(gpu)))

    row("", "j5s-selftest", "1", "2", OWN_INPUTS, "python outputs/mp_unified/j5s.py --selftest")
    for arm in ("zrc", "zsp"):                     # zrc's fits use the card; zsp's run on the CPU, as filed
        t = T[TEMPLATES[("fit", arm)]]
        for seed in ("1", "2", CONTROL):
            row("j5s-selftest", f"j5s-fit-{arm}-s{seed}", t[4], RES[("fit", arm)], t[6], fit_cmd(t[7], arm, seed), t[8:9])
    for arm in ARMS:
        for seed in SEEDS + (CONTROL,):
            dep = "j5s-selftest" if seed == "0" else f"j5s-fit-{arm}-s{seed}"
            row(dep, f"j5s-fork-{arm}-s{seed}", "0.5", "1", OWN_INPUTS, f"python outputs/mp_unified/j5s.py fork --arm {arm} --seed {seed}")
    for arm in ARMS:
        t = T[TEMPLATES[("read", arm)]]
        for seed in SEEDS + (CONTROL,):
            for carve in carves_of(seed):
                name = f"j5s-rd-{carve}-{arm}-s{seed}"
                row(f"j5s-fork-{arm}-s{seed}", name, t[4], RES[("read", arm)], t[6], read_cmd(t[7], arm, seed, carve), t[8:9])
                reads.append(name)
    row(",".join(reads + [f"j5s-fit-{a}-s{CONTROL}" for a in ARMS]), "j5s-report", "1", "4", OWN_INPUTS,
        "python outputs/mp_unified/j5s.py report")
    names = [r.split("|")[3] for r in rows]
    if len(set(names)) != len(names):
        raise SystemExit("rows: duplicate names")
    return rows


def check_rows(rows, queue):
    T = templates(queue)
    names = [r.split("|")[3] for r in rows]
    assert names[0] == "j5s-selftest" and names[-1] == "j5s-report" and len(names) == 1 + 6 + 8 + 26 + 1, len(names)
    assert names[1:7] == [f"j5s-fit-{a}-s{s}" for a in ("zrc", "zsp") for s in ("1", "2", CONTROL)], names[1:7]
    for r in rows:
        f = r.split("|")
        assert len(f) in (8, 9) and f[0] == "line" and f[1] == "x" and "\r" not in r and "\n" not in r
        name, cmd = f[3], f[7]
        if name.startswith("j5s-fit-"):
            arm, seed = name.split("-")[2], name.split("-s")[-1]
            t = T[TEMPLATES[("fit", arm)]]
            assert f[4] == t[4] and f[5] == RES[("fit", arm)] and f[6] == t[6] and f[8:] == t[8:9] and f[2] == "j5s-selftest"
            back = cmd.replace(f" --seed {seed_int(seed)}", "", 1).replace(f" --name J5-s{seed} ", " --name J5 ", 1) \
                .replace(f" --out-root outputs/j5s/train/{arm} ",
                         {"zrc": " --out-root outputs/full_zrct/fits ", "zsp": " --out-root outputs/full_zsp/fits "}[arm], 1)
            assert back == t[7], (back, t[7])
            assert f" --seed {seed_int(seed)} " in cmd and cmd.count("--seed") == 1
        if name.startswith("j5s-rd-"):
            _j, _r, carve, arm, sd = name.split("-")
            t = T[TEMPLATES[("read", arm)]]
            assert f[4] == t[4] and f[5] == RES[("read", arm)] and f[6] == t[6] and f[8:] == t[8:9]
            assert f[2] == f"j5s-fork-{arm}-{sd}"
            back = cmd.replace(f" --read musique={carve},2wiki={carve},hotpotqa={carve} ",
                               " --read musique=b1,2wiki=b1,hotpotqa=b1 ", 1) \
                .replace(f" --name J5-{sd} ", " --name J5 ", 1) \
                .replace(f" --out-root outputs/j5s/fits/{carve}/{arm} ", f" --out-root outputs/b1b/fits/{arm} ", 1)
            assert back == t[7], (back, t[7])
    rep = rows[-1].split("|")
    assert set(rep[2].split(",")) == {n for n in names if n.startswith("j5s-rd-")} | {f"j5s-fit-{a}-s{CONTROL}" for a in ARMS}


# -- report ---------------------------------------------------------------------------------------------------------


def gold_counts(root, ds):
    qs = json.loads((Path(root) / "outputs" / "bench" / "hipporag2" / ds / "queries.json").read_text(encoding="utf-8"))
    return [len(q["golds"]) for q in qs]


def load(p, gc):
    with np.load(p) as z:
        return BB.per_question(z, gc), BB.per_question(z, gc, "rrf")["R@5"], {k: z[k] for k in z.files}


def same_arrays(a, b):
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)


def same_fit(p, q):
    """C1: two models.pt blobs hold the same fit: every candidate's state tensors equal and every other field equal, the
    fit's name aside (lean_gpu saves the name in the blob, so the files' sha256 differ whenever the names do)."""
    import torch
    a, b = (torch.load(x, map_location="cpu", weights_only=False) for x in (p, q))
    fields = sorted(k for k in set(a) | set(b) if k not in ("fit", "states") and repr(a.get(k)) != repr(b.get(k)))
    sa, sb = a.get("states") or {}, b.get("states") or {}
    states = sorted(n for n in set(sa) | set(sb) if n not in sa or n not in sb or set(sa[n]) != set(sb[n])
                    or any(not torch.equal(sa[n][k], sb[n][k]) for k in sa[n]))
    return {"call": "REPRODUCED" if not fields and not states else "NOT_REPRODUCED", "states_compared": len(sa),
            "differing_states": states, "differing_fields": fields, "names": [a.get("fit"), b.get("fit")]}


def curve(p):
    tr = json.loads(Path(p).read_text(encoding="utf-8"))
    return [(e["epoch"], e["loss"], e["steps"]) for e in tr["variants"]["p"]["curve"]]


def r4(x):
    return round(float(x), 4)


def report(root=ROOT, filed=FILED_MODELS, out=None):
    root = Path(root)
    out = Path(out) if out else root / "outputs" / "j5s"
    rec = {"declared_in": DECLARED, "candidate": BB.CAND, "boot": {"resamples": BB.BOOT, "seed": BB.SEED},
           "hipporag2": BB.HIPPORAG2, "published": {n: {"peer_reviewed": p, "R@5": v} for n, (p, v) in BF.PUBLISHED.items()},
           "controls": {}, "settings": {}, "missing": []}
    for arm in ARMS:                                                                 # C1: the control fits
        t, s = train_dir(root, arm, CONTROL), root / SOURCE[arm]
        c = {"filed_models_sha256": filed[arm]}
        if (t / "models.pt").exists() and (s / "models.pt").exists():
            c.update(same_fit(t / "models.pt", s / "models.pt"))
            c["models_sha256"] = BB.sha_file(t / "models.pt")
            a, b = curve(t / "train.json"), curve(s / "train.json")
            c["curve"] = {"epochs": [len(a), len(b)], "same_steps": [x[2] for x in a] == [x[2] for x in b],
                          "max_abs_loss_delta": max((abs(x[1] - y[1]) for x, y in zip(a, b)), default=None)}
        else:
            c["call"] = "MISSING"
            rec["missing"].append(t.relative_to(root).as_posix())
        rec["controls"][arm] = {"C1": c, "C2": {}, "C3": {}}
    for ds in SETTINGS:
        gc = gold_counts(root, ds)
        s = {"questions": len(gc), "carves": {}, "paired": {}, "share_of_hyperlink_gain": {}, "arms_paired": {}}
        mean3 = {}
        for carve in CARVES:
            s["carves"][carve] = {}
            for arm in ARMS:
                ms, rrfs = [], []
                for seed in SEEDS:
                    p = read_file(root, carve, arm, seed, ds)
                    if not p.exists():
                        rec["missing"].append(p.relative_to(root).as_posix())
                        continue
                    m, rr, _ar = load(p, gc)
                    ms.append(m), rrfs.append(rr)
                fp = root / FILED_READ[carve].format(arm=arm, ds=ds)                  # C2: seed 0 re-read = the filed read
                p0 = read_file(root, carve, arm, "0", ds)
                if p0.exists() and fp.exists():
                    m0, _rr, a0 = load(p0, gc)
                    fm, _rr, fa = load(fp, gc)
                    rec["controls"][arm]["C2"][f"{carve} {ds}"] = "EQUAL" if same_arrays(a0, fa) else \
                        f"DIFFERENT (R@5 {r4(m0['R@5'].mean() - fm['R@5'].mean()):+.4f})"
                else:
                    rec["controls"][arm]["C2"][f"{carve} {ds}"] = "MISSING"
                if carve == "b1":                                                     # C3: the control's read = the filed read
                    cp = read_file(root, "b1", arm, CONTROL, ds)
                    if cp.exists() and fp.exists():
                        cm, _rr, ca = load(cp, gc)
                        fm, _rr, fa = load(fp, gc)
                        rec["controls"][arm]["C3"][ds] = "EQUAL" if same_arrays(ca, fa) else \
                            f"DIFFERENT (R@5 {r4(cm['R@5'].mean() - fm['R@5'].mean()):+.4f})"
                    else:
                        rec["controls"][arm]["C3"][ds] = "MISSING"
                        if not cp.exists():
                            rec["missing"].append(cp.relative_to(root).as_posix())
                if len(ms) != len(SEEDS):
                    continue
                if not all(np.array_equal(rrfs[0], x) for x in rrfs[1:]):
                    raise SystemExit(f"{ds} {carve} {arm}: the seeds' RRF references differ")
                mm = {k: sum(m[k] for m in ms) / len(ms) for k in ms[0]}
                ci, ci0 = BB.boot_ci(mm["R@5"]), BB.boot_ci(ms[0]["R@5"])
                seeds = [r4(m["R@5"].mean()) for m in ms]
                pub = {n: v[ds] / 100 for n, (_peer, v) in BF.PUBLISHED.items() if v[ds] is not None}
                e = {"R@5": r4(mm["R@5"].mean()), "R5_ci": ci, "FC@5": r4(mm["FC@5"].mean()), "hit@1": r4(mm["hit@1"].mean()),
                     "seeds_R@5": seeds, "seed_spread": r4(max(seeds) - min(seeds)), "seed0_R5_ci": ci0,
                     "rrf_R@5": r4(rrfs[0].mean()),
                     "vs_hipporag2": BB.call(ci, BB.HIPPORAG2[ds]), "seed0_vs_hipporag2": BB.call(ci0, BB.HIPPORAG2[ds]),
                     "vs_published": {n: BB.call(ci, v) for n, v in pub.items()},
                     "seed0_vs_published": {n: BB.call(ci0, v) for n, v in pub.items()},
                     "files_sha256": [BB.sha_file(read_file(root, carve, arm, sd, ds)) for sd in SEEDS]}
                e["changed"] = {n: f"{e['seed0_vs_published'][n]} -> {e['vs_published'][n]}" for n in pub
                                if e["vs_published"][n] != e["seed0_vs_published"][n]}
                s["carves"][carve][arm] = e
                mean3[(carve, arm)] = mm["R@5"]
        for a, b in (("b1", "b1t"), ("b1w", "b1t"), ("b1u", "b1t"), ("b1", "b1w"), ("b1w", "b1u")):
            for arm in ARMS:
                if (a, arm) in mean3 and (b, arm) in mean3:
                    g = mean3[(a, arm)] - mean3[(b, arm)]
                    s["paired"][f"{a}-{b} {arm}"] = {"R@5": r4(g.mean()), "ci": BB.boot_ci(g)}
        for arm in ARMS:
            k = [mean3.get((cv, arm)) for cv in ("b1", "b1t", "b1u", "b1w")]
            if all(x is not None for x in k) and float((k[0] - k[1]).mean()) > 0:
                den = float((k[0] - k[1]).mean())
                s["share_of_hyperlink_gain"][arm] = {"b1u": round(float((k[2] - k[1]).mean()) / den, 3),
                                                     "b1w": round(float((k[3] - k[1]).mean()) / den, 3)}
        for carve in CARVES:
            if (carve, "zsp") in mean3 and (carve, "zrc") in mean3:
                g = mean3[(carve, "zsp")] - mean3[(carve, "zrc")]
                s["arms_paired"][carve] = {"R@5": r4(g.mean()), "ci": BB.boot_ci(g)}
        if all(arm in s["carves"]["b1"] for arm in ARMS):
            rrf = s["carves"]["b1"]["zrc"]["rrf_R@5"]
            d = s["carves"]["b1"]["zsp"]["R@5"] - rrf
            s["mlp_share_of_prop_gain_b1"] = round((s["carves"]["b1"]["zrc"]["R@5"] - rrf) / d, 3) if d > 0 else None
        s["best_published"] = {w: dict(zip(("R@5", "row"), BF.best_published(ds, peer))) for w, peer in
                               (("any", False), ("peer_reviewed", True))}
        rec["settings"][ds] = s
    rec["script_sha256"] = sha_src(__file__)
    rec["utc"] = utc()
    BB.write_json(out / "report.json", rec)
    (out / "report.md").write_text(markdown(rec), encoding="utf-8")
    log(f"report: {out / 'report.md'}; missing {len(rec['missing'])}")
    return 0


def markdown(rec):
    P = lambda x: f"{100 * x:.1f}"                                                  # noqa: E731
    L = [f"# J5S report (declared in {DECLARED})", "",
         "R@5 x100, candidate p@swa, 1,000 questions per setting. Three seeds: each question's R@5 is averaged over seeds 0, 1 "
         "and 2, and the 95% interval is B1b's bootstrap over the questions. Seed 0 is this stage's re-read of the filed fit. "
         "b1 = the corpus hyperlinks (musique: its title links), b1w = W2_S's rule links, b1t = text links only, b1u = U1a's "
         "title rule. Calls: ABOVE when the interval's low end is above the reference, BELOW when its high end is below, AT "
         "otherwise.", "", "## Controls", ""]
    for arm, c in rec["controls"].items():
        c1 = c["C1"]
        L.append(f"- **{arm}** C1 (seed-0 control fit against the filed fit, tensor by tensor): {c1['call']}"
                 + (f"; {c1['states_compared']} states compared, differing {c1['differing_states'][:4]}, fields "
                    f"{c1['differing_fields']}; curve {c1['curve']}" if "curve" in c1 else ""))
        bad2 = {k: v for k, v in c["C2"].items() if v != "EQUAL"}
        bad3 = {k: v for k, v in c["C3"].items() if v != "EQUAL"}
        L.append(f"  - C2 (seed-0 re-read = filed read): {len(c['C2']) - len(bad2)} of {len(c['C2'])} EQUAL {bad2 or ''}")
        L.append(f"  - C3 (control's b1 read = filed read): {len(c['C3']) - len(bad3)} of {len(c['C3'])} EQUAL {bad3 or ''}")
    L += ["", "## The rows on each carve", "",
          "| setting | carve | arm | s0 | s1 | s2 | mean [95% CI] | spread | vs HippoRAG 2 (s0 / 3 seeds) | calls changed |",
          "|---|---|---|---:|---:|---:|---|---:|---|---|"]
    for ds, s in rec["settings"].items():
        for carve, c in s["carves"].items():
            for arm, e in c.items():
                L.append(f"| {ds} | {carve} | {arm} | " + " | ".join(P(x) for x in e["seeds_R@5"])
                         + f" | {P(e['R@5'])} [{P(e['R5_ci'][0])}, {P(e['R5_ci'][1])}] | {P(e['seed_spread'])} | "
                         f"{e['seed0_vs_hipporag2']} / {e['vs_hipporag2']} | "
                         + ("; ".join(f"{n}: {v}" for n, v in e["changed"].items()) or "none") + " |")
    L += ["", "## Against each published row (three seeds, by the interval)", ""]
    for ds, s in rec["settings"].items():
        for arm, e in s["carves"].get("b1", {}).items():
            L.append(f"- {ds} b1 {arm}: " + ", ".join(f"{n} {v}" for n, v in e["vs_published"].items()))
    L += ["", "## Paired differences (three-seed means, per question)", "", "| setting | difference | R@5 | 95% CI |", "|---|---|---:|---|"]
    for ds, s in rec["settings"].items():
        for k, v in list(s["paired"].items()) + [(f"zsp-zrc on {c}", v) for c, v in s["arms_paired"].items()]:
            L.append(f"| {ds} | {k} | {100 * v['R@5']:+.1f} | [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}] |")
        for arm, v in s["share_of_hyperlink_gain"].items():
            L.append(f"| {ds} | share of b1-b1t, {arm} | b1u {v['b1u']}, b1w {v['b1w']} | |")
        if s.get("mlp_share_of_prop_gain_b1") is not None:
            L.append(f"| {ds} | (zrc - RRF) / (zsp - RRF) on b1 | {s['mlp_share_of_prop_gain_b1']} | |")
    L += ["", f"Missing: {len(rec['missing'])} {rec['missing'][:6]}", ""]
    return "\n".join(L)


# -- selftest -------------------------------------------------------------------------------------------------------


def _toy_read(p, gt, rng, shift=0, ids=None):
    q = gt.size
    top = np.minimum(gt[None, :], rng.integers(0, 3, (9, q)) + shift).astype(np.int16)
    hit = (rng.random((9, q)) < 0.6).astype(np.uint8)
    ref_top = np.minimum(gt[None, :], rng.integers(0, 2, (3, q))).astype(np.int16)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, candidates=np.array([f"p@ep{i}" for i in range(8)] + ["p@swa"]), ids=ids,
             gold_total=gt.astype(np.int32), refs=np.array(["rrf", "twin0", "gnn0"]), top=top, hit=hit, ref_top=ref_top,
             ref_hit=(rng.random((3, q)) < 0.5).astype(np.uint8), scores64=rng.random((9, 40)).astype(np.float32),
             n64=np.int64(40), rows64=np.int64(40))


def _toy_fit(d, arm, seed, name, weights_seed, loss_shift=0.0):
    """A fit folder with lean_gpu's layout: models.pt (a blob with the fit's name and one state per candidate),
    screen.json and train.json."""
    import torch
    d.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(weights_seed)
    states = {f"p@ep{e}": {"w": torch.rand(4, 3, generator=g), "b": torch.rand(4, generator=g)} for e in range(8)}
    states["p@swa"] = {"w": torch.rand(4, 3, generator=g), "b": torch.rand(4, generator=g)}
    blob = {"fit": name, "train": [("metaqa", "fit"), ("2wiki", "fit")], "basis": "2wiki", "config": {"epochs": 8},
            "seed": seed_int(seed), "hidden": 128, "states": states, "candidates": list(states)}
    torch.save(blob, d / "models.pt")
    BB.write_json(d / "screen.json", dict(SAME, arm=arm, seed=seed_int(seed), rc=0, name=name))
    BB.write_json(d / "train.json", {"variants": {"p": {"curve": [{"epoch": i, "loss": 2.0 - 0.01 * i + loss_shift,
                                                                    "steps": 4415, "seconds": 1.0} for i in range(8)]}}})


def selftest():
    ok = 0
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        filed = {}
        for arm in ARMS:     # the filed fit; a zrc control with the same weights under its own name; a zsp control that differs
            _toy_fit(root / SOURCE[arm], arm, "0", "J5", 100 + ARMS.index(arm))
            filed[arm] = BB.sha_file(root / SOURCE[arm] / "models.pt")
            for seed in ("1", "2"):
                _toy_fit(train_dir(root, arm, seed), arm, seed, f"J5-s{seed}", 10 * int(seed) + ARMS.index(arm))
        _toy_fit(train_dir(root, "zrc", CONTROL), "zrc", CONTROL, "J5-s0c", 100)
        _toy_fit(train_dir(root, "zsp", CONTROL), "zsp", CONTROL, "J5-s0c", 999, loss_shift=1e-3)
        assert BB.sha_file(train_dir(root, "zrc", CONTROL) / "models.pt") != filed["zrc"]     # the name alone changes the sha
        # fork: four carves for seeds 0-2, b1 only for the control; idempotent; a mismatched copy, a wrong filed sha, an
        # unknown seed and a fit at the wrong seed are refused
        for arm in ARMS:
            for seed in SEEDS + (CONTROL,):
                fork(arm, seed, root, filed)
                fork(arm, seed, root, filed)
        assert all((fit_dir(root, c, a, s) / "models.pt").exists() for a in ARMS for s in SEEDS for c in CARVES)
        assert not fit_dir(root, "b1w", "zrc", CONTROL).exists() and (fit_dir(root, "b1", "zrc", CONTROL) / "fork.json").exists()
        fj = json.loads((fit_dir(root, "b1t", "zsp", "1") / "fork.json").read_text(encoding="utf-8"))
        assert fj["source"] == "outputs/j5s/train/zsp/J5-s1" and fj["seed"] == "1" and fj["carve"] == "b1t"
        assert not (fit_dir(root, "b1", "zrc", "0") / "reads").exists()
        keep = (fit_dir(root, "b1u", "zrc", "2") / "models.pt").read_bytes()
        (fit_dir(root, "b1u", "zrc", "2") / "models.pt").write_bytes(b"other")
        for bad in (lambda: fork("zrc", "2", root, filed), lambda: fork("zrc", "0", root, {"zrc": "0" * 64, "zsp": ""}),
                    lambda: fork("zrc", "3", root, filed)):
            try:
                bad()
                raise AssertionError("not refused")
            except SystemExit:
                pass
        (fit_dir(root, "b1u", "zrc", "2") / "models.pt").write_bytes(keep)
        js = train_dir(root, "zsp", "2") / "screen.json"
        sc = json.loads(js.read_text(encoding="utf-8"))
        BB.write_json(js, dict(sc, seed=1))
        try:
            check_fit(train_dir(root, "zsp", "2"), "zsp", "2")
            raise AssertionError("a fit at the wrong seed passed")
        except SystemExit:
            pass
        BB.write_json(js, sc)
        ok += 1
        # reads with the real layout (9 candidates x Q, 3 refs, int16 tops, uint8 hits, scores64)
        q = 60
        gts = {}
        for ds in SETTINGS:
            gts[ds] = rng.integers(1, 4, q)
            d = root / "outputs" / "bench" / "hipporag2" / ds
            d.mkdir(parents=True, exist_ok=True)
            BB.write_json(d / "queries.json", [{"id": str(i), "golds": list(range(int(g)))} for i, g in enumerate(gts[ds])])
        ids = np.array([f"q{i}" for i in range(q)])
        for arm in ARMS:
            for carve in CARVES:
                for ds in SETTINGS:
                    for seed in SEEDS:
                        r = np.random.default_rng(1000 * ARMS.index(arm) + 100 * CARVES.index(carve)
                                                  + 10 * SETTINGS.index(ds) + int(seed))
                        _toy_read(read_file(root, carve, arm, seed, ds), gts[ds], r, shift=int(seed == "2"), ids=ids)
                    with np.load(read_file(root, carve, arm, "0", ds)) as z0:      # one RRF reference across the seeds
                        rt, rh = z0["ref_top"], z0["ref_hit"]
                    for seed in ("1", "2"):
                        p = read_file(root, carve, arm, seed, ds)
                        with np.load(p) as z:
                            a = {k: z[k] for k in z.files}
                        a["ref_top"], a["ref_hit"] = rt, rh
                        np.savez(p, **a)
                    fp = root / FILED_READ[carve].format(arm=arm, ds=ds)
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(read_file(root, carve, arm, "0", ds), fp)
                    if carve == "b1":
                        read_file(root, "b1", arm, CONTROL, ds).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(fp, read_file(root, "b1", arm, CONTROL, ds))
        p = root / FILED_READ["b1w"].format(arm="zsp", ds="hotpotqa")              # one planted C2 difference
        with np.load(p) as z:
            a = {k: z[k] for k in z.files}
        a["scores64"] = a["scores64"].copy()
        a["scores64"][0, 0] += 1e-3
        np.savez(p, **a)
        out = root / "out"
        report(root, filed, out)
        rec = json.loads((out / "report.json").read_text(encoding="utf-8"))
        c1z, c1s = rec["controls"]["zrc"]["C1"], rec["controls"]["zsp"]["C1"]
        assert c1z["call"] == "REPRODUCED" and c1z["states_compared"] == 9 and c1z["names"] == ["J5-s0c", "J5"], c1z
        assert c1z["models_sha256"] != filed["zrc"]
        assert c1z["curve"] == {"epochs": [8, 8], "same_steps": True, "max_abs_loss_delta": 0.0}
        assert c1s["call"] == "NOT_REPRODUCED" and len(c1s["differing_states"]) == 9 and c1s["differing_fields"] == []
        assert abs(c1s["curve"]["max_abs_loss_delta"] - 1e-3) < 1e-9
        c2 = rec["controls"]["zsp"]["C2"]
        assert c2["b1w hotpotqa"].startswith("DIFFERENT") and sum(v == "EQUAL" for v in c2.values()) == 11, c2
        assert all(v == "EQUAL" for v in rec["controls"]["zrc"]["C2"].values()) and len(rec["controls"]["zrc"]["C2"]) == 12
        assert all(v == "EQUAL" for a in ARMS for v in rec["controls"][a]["C3"].values())
        assert rec["missing"] == []
        for arm in ARMS:                                                            # the three-seed means, by hand
            for carve in CARVES:
                for ds in SETTINGS:
                    per = []
                    for seed in SEEDS:
                        with np.load(read_file(root, carve, arm, seed, ds)) as z:
                            per.append(z["top"][8].astype(np.float64) / gts[ds])
                    mm = (per[0] + per[1] + per[2]) / 3
                    e = rec["settings"][ds]["carves"][carve][arm]
                    assert e["R@5"] == round(float(mm.mean()), 4) and e["R5_ci"] == BB.boot_ci(mm)
                    assert e["seeds_R@5"] == [round(float(x.mean()), 4) for x in per]
                    assert e["seed0_R5_ci"] == BB.boot_ci(per[0])
                    assert e["vs_hipporag2"] == BB.call(BB.boot_ci(mm), BB.HIPPORAG2[ds])
                    for n, v in e["vs_published"].items():
                        assert v == BB.call(e["R5_ci"], BF.PUBLISHED[n][1][ds] / 100)
                    assert set(e["changed"]) == {n for n in e["vs_published"] if e["vs_published"][n] != e["seed0_vs_published"][n]}
                    assert set(e["vs_published"]) == {n for n, (_p, v) in BF.PUBLISHED.items() if v[ds] is not None}
        ok += 1
        assert BB.call([0.97, 0.99], 0.96) == "ABOVE" and BB.call([0.95, 0.9875], 0.9875) == "AT" \
            and BB.call([0.95, 0.98], 0.9875) == "BELOW"
        g = rec["settings"]["musique"]["paired"]["b1-b1t zsp"]
        assert len(g["ci"]) == 2 and g["ci"][0] - 1e-4 <= g["R@5"] <= g["ci"][1] + 1e-4
        assert set(rec["settings"]["2wiki"]["arms_paired"]) == set(CARVES)
        assert (out / "report.md").read_text(encoding="utf-8").startswith("# J5S report")
        ok += 1
        tq = root / "queue.txt"                                                     # rows: a toy queue with the real rows' shape
        tq.write_text("\r\n".join([
            "# a comment line|x|y|fzk-train-J5|1|1|x|y",
            "line|x|fzk-gate|fzk-train-J5|1.1|9|outputs/mp_unified/zrct.py;outputs/mp_unified/zrc.py|python outputs/host_ops/retry_cmd.py "
            "--tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.26 --gc 0.9 -- outputs/mp_unified/zrct.py train "
            "--split J5 --name J5 --arm zrc --out-root outputs/full_zrct/fits --device cuda --host|gpu:0.28",
            "line|x|fsp-gate|fsp-train-J5|6|16|outputs/mp_unified/zprop.py|python outputs/mp_unified/zprop.py train --split J5 --name J5 "
            "--arm zsp --out-root outputs/full_zsp/fits --device cpu --threads 6 --host",
            "line|x|b1-fork-zrc-J5|b1-read2-zrc-J5|1.1|6|outputs/mp_unified/zrct.py|python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 "
            "--cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.3 --gc 0.9 -- outputs/mp_unified/zrct.py read --name J5 --out-root "
            "outputs/b1b/fits/zrc --read musique=b1,2wiki=b1,hotpotqa=b1 --device cuda --host|gpu:0.32",
            "line|x|b1-fork-zsp-J5|b1-read2-zsp-J5|6|12|outputs/mp_unified/zprop.py|python outputs/mp_unified/zprop.py read --name J5 "
            "--out-root outputs/b1b/fits/zsp --read musique=b1,2wiki=b1,hotpotqa=b1 --device cpu --threads 6 --host", ""]), encoding="utf-8")
        rows = block(tq)
        check_rows(rows, tq)
        f = {r.split("|")[3]: r.split("|") for r in rows}
        assert f["j5s-fit-zrc-s1"][7].endswith("zrct.py train --split J5 --name J5-s1 --arm zrc --seed 1 --out-root "
                                               "outputs/j5s/train/zrc --device cuda --host") and f["j5s-fit-zrc-s1"][8] == "gpu:0.28"
        assert f["j5s-fit-zsp-s0c"][7] == ("python outputs/mp_unified/zprop.py train --split J5 --name J5-s0c --arm zsp --seed 0 "
                                           "--out-root outputs/j5s/train/zsp --device cpu --threads 6 --host") and len(f["j5s-fit-zsp-s0c"]) == 8
        assert f["j5s-rd-b1u-zsp-s2"][7] == ("python outputs/mp_unified/zprop.py read --name J5-s2 --out-root outputs/j5s/fits/b1u/zsp "
                                             "--read musique=b1u,2wiki=b1u,hotpotqa=b1u --device cpu --threads 6 --host")
        assert f["j5s-rd-b1u-zsp-s2"][2] == "j5s-fork-zsp-s2" and f["j5s-fork-zsp-s0"][2] == "j5s-selftest"
        assert [f[n][5] for n in ("j5s-fit-zrc-s1", "j5s-fit-zsp-s1", "j5s-rd-b1-zrc-s0", "j5s-rd-b1-zsp-s0")] == ["4", "12", "3", "6"]
        assert "j5s-rd-b1w-zrc-s0c" not in f and "j5s-rd-b1-zrc-s0c" in f
        ok += 1
    real = []                     # the real files, where this tree has them: the queue's rows, the filed reads and fits
    rq = ROOT / "outputs" / "host_ops" / "hostfeed_items.txt"
    if rq.exists():
        check_rows(block(rq), rq)
        real.append("queue rows")
    n = 0
    for arm in ARMS:
        for carve in CARVES:
            for ds in SETTINGS:
                fp = ROOT / FILED_READ[carve].format(arm=arm, ds=ds)
                if fp.exists():
                    m, rr, _a = load(fp, gold_counts(ROOT, ds))
                    assert m["R@5"].size == 1000 and rr.size == 1000
                    n += 1
    real.append(f"{n} filed reads")
    for arm in ARMS:
        src = ROOT / SOURCE[arm]
        if (src / "models.pt").exists():
            check_fit(src, arm, "0")
            assert BB.sha_file(src / "models.pt") == FILED_MODELS[arm], f"{src}: not the filed models"
            real.append(f"{arm}'s filed fit")
    log(f"j5s selftest: {ok} toy checks passed; real: {', '.join(real)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("rows", "fork", "report"))
    ap.add_argument("--arm")
    ap.add_argument("--seed")
    ap.add_argument("--queue", default=str(ROOT / "outputs" / "host_ops" / "hostfeed_items.txt"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "rows":
        rows = block(a.queue)
        check_rows(rows, a.queue)
        print("\n".join(rows))
        return 0
    if a.cmd == "fork":
        return fork(a.arm, a.seed)
    if a.cmd == "report":
        return report()
    ap.error("rows, fork, report or --selftest")


if __name__ == "__main__":
    sys.exit(main())
