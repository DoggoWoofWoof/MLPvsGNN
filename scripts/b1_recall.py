"""B1 re-call (docs/B1_RECALL.md): every filed screen and full-run decision, replayed from its fits' filed s1eval reads,
then called again without HippoRAG 2's benchmark questions (B1: 1,000 each of musique, hotpotqa and 2wiki).

    python scripts/b1_recall.py run [--limit N]
    python scripts/b1_recall.py --selftest

A decision is a screen pair's re-call under the seed null (outputs/screen/*-pair-recall.json) or a full run's re-grade
under the null over every split (outputs/*/grade-nullx.json). Each read is computed again as lean_screen's compare
computed it (pick, metrics_of, boot_ci and call, imported unchanged), each floor and call as screen_recall's (floor_of,
recall, adopt). The replay with every question must give the filed record (NOT_REPRODUCED otherwise); the second pass
drops B1's questions from both fits' reads and from the null's. No model is run. Writes outputs/b1r/b1r.{json,md}.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MU = ROOT / "outputs" / "mp_unified"
if str(MU) not in sys.path:
    sys.path.insert(0, str(MU))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import screen_recall as SR  # noqa: E402

S, LG, LC = SR.S, SR.LG, SR.LC
OUT = ROOT / "outputs" / "b1r"
B1_ROOT = ROOT / "outputs" / "bench" / "hipporag2"
B1_DATASETS = ("musique", "hotpotqa", "2wiki")
B1_SIZE = 1000
MODES = ("full", "noB1")
NULL_KEY = re.compile(r"^(\S+) seed ([12])$")
# The decisions behind the models we report (docs/B1_RECALL.md, section 4), by record name.
LINEAGE = ("full_zret", "scr-zrs", "full_zrs", "scr-zrm", "full_zrm", "zbase", "scr-zrct", "full_zrct", "zbase2",
           "scr-zsp", "full_zsp", "scr-p1c3-zrck", "full_zrck", "scr-p1c3-zspk", "full_zspk", "scr-p1c3p-zrckp",
           "full_zrckp")
log = LC.log


class Gone(Exception):
    """An input the replay needs is not on this machine (NOT_REPLAYABLE)."""


class Differs(Exception):
    """The filed inputs no longer agree with each other (NOT_REPRODUCED)."""


def where(p):
    """A filed path (absolute, or the repository's with either slash) on this machine."""
    q = Path(str(p))
    if q.is_absolute() and q.exists():
        return q
    parts = SR.parts_of(p)
    if "outputs" in parts:
        parts = parts[parts.index("outputs"):]
    return ROOT.joinpath(*parts)


def shown(p):
    q = where(p)
    try:
        return q.relative_to(ROOT).as_posix()
    except ValueError:
        return q.name


def load_json(p):
    q = where(p)
    if not q.exists():
        raise Gone(f"{shown(p)} missing")
    return json.loads(q.read_text(encoding="utf-8"))


def b1_ids(root=B1_ROOT):
    """B1's question ids per dataset (the package's query ids, as the reads hold them)."""
    out = {}
    for ds in B1_DATASETS:
        ids = [str(x["id"]) for x in json.loads((root / ds / "queries.json").read_text(encoding="utf-8"))]
        if len(ids) != B1_SIZE or len(set(ids)) != B1_SIZE:
            raise SystemExit(f"b1_recall: {ds}: {len(ids)} ids ({len(set(ids))} distinct), not {B1_SIZE}")
        out[ds] = np.array(sorted(ids))
    return out


# ── the replay ───────────────────────────────────────────────────────────────


class Replay:
    """Reads and rows, cached: a fit's read is loaded once, a comparison's read is bootstrapped once per mode."""

    def __init__(self, b1):
        self.b1 = b1
        self.ids = {}       # sha of a read's ids -> the ids
        self.reads = {}     # (fit, dataset, carve, candidate) -> (ids sha, gold_total, top, hit)
        self.masks = {}     # (ids sha, dataset) -> (keep, dropped)
        self.rows = {}

    def read(self, fit, ds, carve, cand):
        key = (str(where(fit)), ds, carve, cand)
        if key not in self.reads:
            p = where(fit) / "reads" / f"{ds}__{carve}.npz"
            if not p.exists():
                raise Gone(f"{shown(p)} missing")
            with np.load(p) as z:
                ids = np.asarray([str(x) for x in z["ids"]])
                top, hit = S.pick(z, cand)
                rec = (hashlib.sha256(ids.tobytes()).hexdigest(), np.asarray(z["gold_total"]), np.asarray(top),
                       np.asarray(hit))
            self.ids.setdefault(rec[0], ids)
            self.reads[key] = rec
        return self.reads[key]

    def mask(self, sha, ds, mode):
        if mode == "full" or ds not in self.b1:
            return None, 0
        if (sha, ds) not in self.masks:
            keep = ~np.isin(self.ids[sha], self.b1[ds])
            self.masks[(sha, ds)] = (keep, int((~keep).sum()))
        return self.masks[(sha, ds)]

    def row(self, new, base, ds, carve, cand, mode):
        """One read of one comparison, as lean_screen's compare computes it, on the mode's questions."""
        key = (str(where(new)), str(where(base)), ds, carve, cand, mode)
        if key in self.rows:
            return self.rows[key]
        sn, gt, tn, hn = self.read(new, ds, carve, cand)
        sb, gtb, tb, hb = self.read(base, ds, carve, cand)
        if sn != sb:
            raise Differs(f"{ds}: {shown(new)} and {shown(base)} hold different questions")
        if not np.array_equal(gt, gtb):
            raise Differs(f"{ds}: {shown(new)} and {shown(base)} hold different gold totals")
        keep, dropped = self.mask(sn, ds, mode)
        if keep is not None:
            gt, tn, hn, tb, hb = gt[keep], tn[keep], hn[keep], tb[keep], hb[keep]
        ok = gt > 0
        mn = LG.metrics_of(tn, hn, gt)[ok]
        mb = LG.metrics_of(tb, hb, gt)[ok]
        d, lo, hi = S.boot_ci(mn - mb, seed=LG.EVAL_ORDER.index(ds))
        r4 = lambda v: [round(float(x), 4) for x in v]  # noqa: E731
        out = {"questions": int(ok.sum()), "dropped": dropped, "base": r4(mb.mean(0)), "new": r4(mn.mean(0)),
               "delta": r4(d), "lo": r4(lo), "hi": r4(hi), "call": S.call(d[0], lo[0], hi[0])}
        self.rows[key] = out
        return out

    def compare(self, file, mode):
        """A filed compare record replayed: ({dataset: row}, [differences from its filed rows])."""
        rec = load_json(file)
        b0 = rec["bases"][0]
        out, diff = {}, []
        for f in rec["by_base"][b0]["rows"]:
            ds = f["dataset"]
            r = out[ds] = self.row(rec["new"], b0, ds, rec["carve"], rec["candidate"], mode)
            if mode == "full":
                diff += [f"{shown(file)} {ds} {k}: filed {f[k]}, replayed {r[k]}"
                         for k in ("questions", "base", "new", "delta", "lo", "hi", "call") if r[k] != f[k]]
        return out, diff


# ── the decisions ────────────────────────────────────────────────────────────


def decision_files(root=ROOT):
    pairs = sorted((root / "outputs" / "screen").glob("*-pair-recall.json"))
    grades = sorted((root / "outputs").glob("*/grade-nullx.json"))
    return [("screen", p) for p in pairs] + [("full", g) for g in grades]


def name_of(kind, p):
    p = Path(p)
    return p.name[:-len("-pair-recall.json")] if kind == "screen" else p.parent.name


def compares_of(kind, R):
    """{split: compare record} of a decision: the pair's fits, or the grade's sources. A pair record no longer on
    this machine falls back to the screens' names (NAME.json, NAME-hp.json); the replay must still give every row."""
    if kind == "full":
        return dict(load_json(R["grade"])["sources"]), []
    try:
        return {f["split"]: f["file"] for f in load_json(R["pair"])["fits"]}, []
    except Gone:
        stem = SR.parts_of(R["pair"])[-1][:-len("-pair.json")]
        d = where(R["pair"]).parent
        return ({"L-musique": str(d / f"{stem}.json"), "L-hotpotqa": str(d / f"{stem}-hp.json")},
                [f"{shown(R['pair'])} missing: its comparisons found by name"])


def null_of(R):
    """The null's seed comparisons a decision names: {(split, seed): file}; its base-arm comparisons are not needed."""
    out = {}
    for k, f in R["null"].items():
        m = NULL_KEY.match(k)
        if m:
            out[(m.group(1), int(m.group(2)))] = f
    return out


def decide(rp, kind, R, mode):
    """One decision's reads called in one mode: (rows, verdict, differences from the filed record, notes)."""
    comps, notes = compares_of(kind, R)
    diff, reads = [], {}
    for split, f in comps.items():
        reads[split], d = rp.compare(f, mode)
        diff += d
    null = {}
    for (split, seed), f in null_of(R).items():
        rows, d = rp.compare(f, mode)
        diff += d
        for ds, r in rows.items():
            null.setdefault((split, ds), {})[seed] = r["delta"][0]
    rows = []
    for f in R["rows"]:
        sp, ds = f["split"], f["dataset"]
        if ds not in reads.get(sp, {}):
            raise Gone(f"{sp} {ds}: no comparison of the decision holds the read")
        r, n = reads[sp][ds], null.get((sp, ds))
        if n is not None and sorted(n) != [1, 2]:
            raise Differs(f"{sp} {ds}: the null holds seeds {sorted(n)}, not 1 and 2")
        covered = n is not None
        floor = SR.floor_of(n[1], n[2]) if covered else S.FLOOR
        row = {"split": sp, "dataset": ds, "read": f["read"], "primary": bool(f.get("primary", False)),
               "questions": r["questions"], "dropped": r["dropped"], "base": r["base"][0], "new": r["new"][0],
               "delta": r["delta"][0], "lo": r["lo"][0], "hi": r["hi"][0], "call0": r["call"], "covered": covered,
               "null_d": [n[1], n[2]] if covered else None, "floor": round(floor, 6),
               "call": SR.recall(r["call"], r["delta"][0], floor) if covered else r["call"]}
        rows.append(row)
        if mode == "full":
            want = {"delta": f["delta"], "lo": f["lo"], "hi": f["hi"], "call0": f["call_filed"],
                    "covered": f["covered"], "floor": f["floor"], "call": f["call"]}
            if covered:
                want["null_d"] = f.get("null_d")
            diff += [f"{sp} {ds} {k}: filed {v}, replayed {row[k]}" for k, v in want.items() if row[k] != v]
    calls = [r["call"] for r in rows]
    v = S.verdict(calls) if kind == "screen" else "ADOPT" if SR.adopt(rows) else "NOT_ADOPTED"
    if mode == "full" and v != R["verdict"]:
        diff.append(f"verdict: filed {R['verdict']}, replayed {v}")
    return rows, v, diff, notes


def changes(full, nob1):
    out = []
    for a, b in zip(full, nob1):
        if a["call"] != b["call"]:
            out.append(f"{a['split']} {a['dataset']} {a['call']} -> {b['call']} ({a['delta']:+.4f} -> "
                       f"{b['delta']:+.4f}, floor {a['floor']:.4f} -> {b['floor']:.4f})")
    return out


def one(rp, kind, p, lineage):
    R = json.loads(Path(p).read_text(encoding="utf-8"))
    name = name_of(kind, p)
    res = {"decision": name, "kind": kind, "file": shown(p), "arm": R.get("arm"), "base_arm": R.get("base_arm"),
           "lineage": name in lineage, "filed": R.get("verdict")}
    t0 = time.time()
    try:
        known = ("PROMISING", "MIXED", "NO_GAIN") if kind == "screen" else ("ADOPT", "NOT_ADOPTED")
        if R.get("verdict") not in known:
            raise Gone(f"filed verdict {R.get('verdict')}")
        full, _, diff, notes = decide(rp, kind, R, "full")
        res["notes"] = notes
        if diff:
            res.update(status="NOT_REPRODUCED", n_differences=len(diff), differences=diff[:12])
        else:
            nob1, v, _, _ = decide(rp, kind, R, "noB1")
            short = sorted({f"{r['split']} {r['dataset']} drops {r['dropped']}" for r in nob1
                            if r["dataset"] in rp.b1 and r["dropped"] != len(rp.b1[r["dataset"]])})
            res.update(status="REPRODUCED", without_b1=v, changed=changes(full, nob1), b1_short=short,
                       rows=[{**a, "noB1": {k: b[k] for k in ("questions", "dropped", "delta", "lo", "hi", "null_d",
                                                                  "floor", "call")}} for a, b in zip(full, nob1)])
    except Gone as e:
        res.update(status="NOT_REPLAYABLE", reason=str(e))
    except Differs as e:
        res.update(status="NOT_REPRODUCED", n_differences=1, differences=[str(e)])
    res["seconds"] = round(time.time() - t0, 1)
    return res


def conclude(results, lineage):
    got = {r["decision"]: r for r in results}
    changed = [n for n in lineage if n in got and got[n]["status"] == "REPRODUCED"
               and got[n]["without_b1"] != got[n]["filed"]]
    unchecked = [n for n in lineage if n not in got or got[n]["status"] != "REPRODUCED"]
    return ("CHANGED" if changed else "NOT_ALL_CHECKED" if unchecked else "STANDS"), changed, unchecked


def run(files=None, b1=None, out=OUT, limit=None, lineage=LINEAGE):
    b1 = b1_ids() if b1 is None else b1
    files = decision_files() if files is None else files
    rp, results, t0 = Replay(b1), [], time.time()
    for i, (kind, p) in enumerate(files[:limit] if limit else files):
        r = one(rp, kind, p, lineage)
        results.append(r)
        log(f"b1r {i + 1}/{len(files)} {r['decision']} ({kind}{', lineage' if r['lineage'] else ''}): {r['status']}; "
            f"filed {r['filed']}" + (f", without B1 {r['without_b1']}" if r["status"] == "REPRODUCED" else "")
            + f" ({r['seconds']}s)")
    verdict, changed, unchecked = conclude(results, lineage)
    rec = {"verdict": verdict, "changed": changed, "unchecked": unchecked, "lineage": list(lineage),
           "b1": {ds: int(len(v)) for ds, v in b1.items()}, "decisions": results, "doc": "docs/B1_RECALL.md",
           "script_sha256": LC.sha_src(__file__), "screen_recall_sha256": LC.sha_src(SR.__file__),
           "lean_screen_sha256": LC.sha_src(S.__file__), "seconds": round(time.time() - t0, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        LC.write_json(out / "b1r.json", rec)
        (out / "b1r.md").write_text("\n".join(report_md(rec)) + "\n", encoding="utf-8")
    log(f"b1r: {verdict}; changed {changed or 'none'}; unchecked {unchecked or 'none'}")
    return rec


def report_md(rec):
    md = [f"# B1 re-call: **{rec['verdict']}**", "",
          "Every filed screen pair and full-run grade, replayed from its fits' filed s1eval reads, then called again "
          "with B1's questions dropped (" + ", ".join(f"{ds} {n}" for ds, n in rec["b1"].items())
          + "; docs/B1_RECALL.md). Filed: the decision's verdict under the seed null. Full: the replay with every "
          "question, which must give the filed record. Without B1: the same replay on the other questions, the "
          "null's floors included.", ""]
    for title, lin in (("The lineage", True), ("Beside (cannot change the verdict)", False)):
        md += [f"## {title}", "", "| decision | kind | arm (against) | filed | replay | without B1 | changed reads |",
               "|---|---|---|---|---|---|---|"]
        for r in rec["decisions"]:
            if r["lineage"] != lin:
                continue
            if r["status"] == "REPRODUCED":
                rp, nb = "reproduced", r["without_b1"] if r["without_b1"] == r["filed"] else f"**{r['without_b1']}**"
                ch = "; ".join(r["changed"]) or "none"
            else:
                rp, nb, ch = r["status"], "—", (r.get("reason") or "; ".join(r.get("differences", [])[:3]))
            md.append(f"| {r['decision']} | {r['kind']} | {r['arm']} ({r['base_arm'] or 'step 1'}) | {r['filed']} "
                      f"| {rp} | {nb} | {ch} |")
        md.append("")
    notes = [f"{r['decision']}: {x}" for r in rec["decisions"] for x in r.get("notes", []) + r.get("b1_short", [])]
    md += [f"Lineage decisions whose verdict changes without B1: {', '.join(rec['changed']) or 'none'}.",
           "Lineage decisions not checked (not reproduced or not replayable): "
           f"{', '.join(rec['unchecked']) or 'none'}.",
           "Notes: " + ("; ".join(notes) if notes else "none") + "."]
    return md


# ── selftest ─────────────────────────────────────────────────────────────────


N_Q = 120
CANDS = np.array(["p@swa", "p@last"])


def base_read(ds, seed):
    rng = np.random.default_rng(1000 * seed + LG.EVAL_ORDER.index(ds))
    gt = rng.integers(0, 4, N_Q).astype(np.int32)
    top = np.stack([rng.integers(0, gt + 1) for _ in CANDS]).astype(np.int16)
    hit = (rng.random((len(CANDS), N_Q)) < 0.5).astype(np.uint8)
    return gt, top, hit


def write_fit(root, name, split, seed, change=None, arm="x"):
    """A fit directory with s1eval reads on the six datasets: step 1's read of split (seed seed), changed by
    change(dataset, top, gold_total) on candidate p@swa."""
    d = Path(root) / name
    (d / "reads").mkdir(parents=True, exist_ok=True)
    (d / "train.json").write_text(json.dumps({"train": [{"dataset": x} for x in LG.FITS[split]]}), encoding="utf-8")
    (d / "screen.json").write_text(json.dumps({"arm": arm, "split": split, "seed": seed}), encoding="utf-8")
    for ds in LG.EVAL_ORDER:
        gt, top, hit = base_read(ds, LG.EVAL_ORDER.index(ds) + 7 * list(LG.FITS).index(split))
        if change:
            top[0] = np.clip(change(ds, top[0].astype(np.int64), gt), 0, gt).astype(np.int16)
        np.savez(d / "reads" / f"{ds}__s1eval.npz", candidates=CANDS, ids=np.array([f"{ds}-{i}" for i in range(N_Q)]),
                 gold_total=gt, refs=np.array(["rrf", "a", "b"]), top=top, hit=hit, ref_top=top[[0, 0, 0]] * 0,
                 ref_hit=hit[[0, 0, 0]] * 0)
    return d


def compare_to(T, new, base, out):
    S.compare_cmd(argparse.Namespace(new=str(new), base=str(base), candidate="p@swa", carve="s1eval", out=str(out)))
    return Path(out).with_suffix(".json")


def on_rows(dsx, rows, k):
    """A change: k more golds in the top 5 (or fewer, k < 0) on dataset dsx's questions in rows."""
    def f(ds, top, gt):
        if ds == dsx:
            top[rows] += k
        return top
    return f


def selftest():
    import nullx as NX
    import screen_pair as SP
    import lean_screen2 as S2
    t0 = time.time()
    B1T = {ds: np.array(sorted(f"{ds}-{i}" for i in range(40))) for ds in B1_DATASETS}
    b1rows = np.arange(40)
    every = np.arange(0, N_Q, 2)
    with tempfile.TemporaryDirectory() as td:
        T = Path(td) / "outputs"
        step1 = T / "step1" / "fits"
        for sp in LG.FITS:
            write_fit(step1, sp, sp, 0, arm="base")
        null = {}
        for sp in LG.FITS:          # seed-only noise; L-musique metaqa's is large enough to raise its floor
            for seed in (1, 2):
                big = on_rows("metaqa", np.arange(0, 30), 1 if seed == 1 else -1) if sp == "L-musique" else None
                f = write_fit(T / "null", f"{sp}-s{seed}", sp, seed, big or on_rows("squad", [seed], 1), arm="base")
                null[(sp, seed)] = compare_to(T, f, step1 / sp, T / "screen" / f"null-{sp}-s{seed}")
        null4 = [null[(sp, s)] for sp in SR.NULL_SPLITS for s in (1, 2)]
        null12 = list(null.values())

        # The filter: B1's rows leave every array of the read alike; other datasets are untouched.
        rp = Replay(B1T)
        g = write_fit(T / "fits", "g-J5", "J5", 0, on_rows("musique", b1rows, 1))
        r_full = rp.row(g, step1 / "J5", "musique", "s1eval", "p@swa", "full")
        r_cut = rp.row(g, step1 / "J5", "musique", "s1eval", "p@swa", "noB1")
        gt, top, hit = base_read("musique", LG.EVAL_ORDER.index("musique"))
        keep = np.arange(N_Q) >= 40
        assert r_cut["dropped"] == 40 and r_full["dropped"] == 0
        assert r_cut["questions"] == int((gt[keep] > 0).sum()) and r_full["questions"] == int((gt > 0).sum())
        assert r_cut["delta"][0] == 0.0 and r_full["delta"][0] > S.FLOOR and r_full["call"] == "GAIN"
        assert r_cut["base"][0] == round(float(LG.metrics_of(top[0][keep], hit[0][keep], gt[keep])[gt[keep] > 0, 0]
                                               .mean()), 4)
        assert rp.row(g, step1 / "J5", "squad", "s1eval", "p@swa", "noB1")["dropped"] == 0

        # A screen pair: a GAIN on musique's B1 rows only and a LOSS on hotpotqa's B1 rows only (MIXED); without B1
        # both are WITHIN (NO_GAIN). A second pair gains on 2wiki's other rows too: it stands.
        a = compare_to(T, write_fit(T / "fits", "scr-x", "L-musique", 0, on_rows("musique", b1rows, 1)),
                       step1 / "L-musique", T / "screen" / "scr-x")
        b = compare_to(T, write_fit(T / "fits", "scr-x-hp", "L-hotpotqa", 0, on_rows("hotpotqa", b1rows, -1)),
                       step1 / "L-hotpotqa", T / "screen" / "scr-x-hp")
        SP.pair([a, b], T / "screen" / "scr-x-pair")
        SR.pair(null4, T / "screen" / "scr-x-pair.json", T / "screen" / "scr-x-pair-recall")
        a = compare_to(T, write_fit(T / "fits", "scr-y", "L-musique", 0, on_rows("2wiki", every, 1)),
                       step1 / "L-musique", T / "screen" / "scr-y")
        b = compare_to(T, write_fit(T / "fits", "scr-y-hp", "L-hotpotqa", 0), step1 / "L-hotpotqa",
                       T / "screen" / "scr-y-hp")
        SP.pair([a, b], T / "screen" / "scr-y-pair")
        SR.pair(null4, T / "screen" / "scr-y-pair.json", T / "screen" / "scr-y-pair-recall")
        # A full run: J5 musique (primary) GAINs on B1's rows only: ADOPT; without B1, NOT_ADOPTED.
        full = T / "full_g"
        full.mkdir()
        for sp in LG.FITS:
            ch = on_rows("musique", b1rows, 1) if sp == "J5" else None
            compare_to(T, write_fit(T / "fits", f"g-{sp}", sp, 0, ch), step1 / sp, full / f"compare-{sp}")
        assert S2.grade(full, {}, "x", out=full / "grade", check_fits=False)["verdict"] == "ADOPT"
        assert NX.regrade(null12, full / "grade.json", full / "grade-nullx")["verdict"] == "ADOPT"
        x, y = T / "screen" / "scr-x-pair-recall.json", T / "screen" / "scr-y-pair-recall.json"
        gx = full / "grade-nullx.json"
        files = [("screen", x), ("screen", y), ("full", gx)]
        assert [name_of(k, p) for k, p in files] == ["scr-x", "scr-y", "full_g"]
        assert json.loads(x.read_text(encoding="utf-8"))["verdict"] == "MIXED"
        assert any(r["floor"] > S.FLOOR for r in json.loads(x.read_text(encoding="utf-8"))["rows"])

        rec = run(files, B1T, T / "b1r", lineage=("scr-x", "scr-y", "full_g"))
        got = {r["decision"]: r for r in rec["decisions"]}
        assert all(r["status"] == "REPRODUCED" for r in got.values()), [r.get("differences") for r in got.values()]
        assert got["scr-x"]["without_b1"] == "NO_GAIN" and got["full_g"]["without_b1"] == "NOT_ADOPTED"
        assert got["scr-y"]["without_b1"] == got["scr-y"]["filed"] == "PROMISING" and not got["scr-y"]["changed"]
        assert sorted(c.split(" (")[0] for c in got["scr-x"]["changed"]) == [
            "L-hotpotqa hotpotqa LOSS -> WITHIN", "L-musique musique GAIN -> WITHIN"], got["scr-x"]["changed"]
        assert [c.split(" (")[0] for c in got["full_g"]["changed"]] == ["J5 musique GAIN -> WITHIN"]
        assert rec["verdict"] == "CHANGED" and rec["changed"] == ["scr-x", "full_g"]
        assert all(r["noB1"]["dropped"] == 40 and r["dropped"] == 0
                   for r in got["full_g"]["rows"] if r["dataset"] in B1_DATASETS)
        assert all(r["noB1"]["dropped"] == 0 for r in got["full_g"]["rows"] if r["dataset"] not in B1_DATASETS)
        assert "**NO_GAIN**" in (T / "b1r" / "b1r.md").read_text(encoding="utf-8")
        assert run([("screen", y)], B1T, None, lineage=("scr-y",))["verdict"] == "STANDS"

        # A filed number the reads do not give, and a decision whose comparisons are gone.
        bad = json.loads(y.read_text(encoding="utf-8"))
        bad["rows"][3]["delta"] = round(bad["rows"][3]["delta"] + 0.0001, 4)
        (T / "screen" / "scr-z-pair-recall.json").write_text(json.dumps(bad), encoding="utf-8")
        gone = dict(bad, pair=str(T / "screen" / "scr-w-pair.json"))
        (T / "screen" / "scr-w-pair-recall.json").write_text(json.dumps(gone), encoding="utf-8")
        rec = run([("screen", T / "screen" / "scr-z-pair-recall.json"),
                   ("screen", T / "screen" / "scr-w-pair-recall.json"), ("screen", y)], B1T, None,
                  lineage=("scr-z", "scr-y", "scr-w"))
        got = {r["decision"]: r for r in rec["decisions"]}
        assert got["scr-z"]["status"] == "NOT_REPRODUCED" and got["scr-z"]["n_differences"] == 1
        assert got["scr-w"]["status"] == "NOT_REPLAYABLE" and "missing" in got["scr-w"]["reason"]
        assert rec["verdict"] == "NOT_ALL_CHECKED" and rec["unchecked"] == ["scr-z", "scr-w"]
        assert conclude([], ("a",)) == ("NOT_ALL_CHECKED", [], ["a"])
    print(f"b1_recall selftest: the filter cuts every array of a read alike; a pair and a grade written by "
          f"lean_screen, screen_pair, screen_recall and nullx replay exactly; gains and losses carried by B1's rows "
          f"turn WITHIN without them (CHANGED), others stand (STANDS); a changed number is NOT_REPRODUCED, a missing "
          f"record NOT_REPLAYABLE ({time.time() - t0:.1f}s): ok", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("run",))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd != "run":
        ap.error("run or --selftest")
    rec = run(out=a.out, limit=a.limit)
    return 0 if rec["verdict"] in ("STANDS", "CHANGED") else 1


if __name__ == "__main__":
    sys.exit(main())
