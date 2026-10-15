"""P1g2: P1g's candidates, and each one's extra set gated on KIND_T's own expansion (docs/P1G2_GATED_EXTRA.md).
Pools only: nothing trains or reads, and nothing is fit.

    python scripts/p1g2_gated.py verdict      section 3's choice, from P1g's filed s1sel arrays (no new pass)
    python outputs/host_ops/pylib_run.py scripts/p1g2_gated.py eval DATASET --host
                                          after the verdict: today's and the chosen rule on s1eval (beside), through
                                          P1g's run_stage, unchanged
    python scripts/p1g2_gated.py report       the verdict with s1eval beside
    python scripts/p1g2_gated.py --selftest

A gated candidate `g:f+X` is P1g's `f+X` on a question KIND_T expands (today's fill, |I_q| - |B|, above 0), and the
question's base and seeds B alone on a question it does not (there no fraction of an empty fill adds a node, so the
gate drops the extra set only). P1g's own 805 candidates stay in the choice as filed; the 798 gated ones (every extra
but none) follow them in P1g's order. Every number comes from P1g's arrays: a gated candidate's rows and held golds
are, per question, the ungated candidate's where the fill is above 0 and the same fraction's `none` where it is 0.
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import p1g_pool_beam as G  # noqa: E402

DECLARED = "docs/P1G2_GATED_EXTRA.md"
OUT = ROOT / "outputs" / "p1g2"
VERDICT = OUT / "verdict.json"
P1G_VERDICT = G.VERDICT
TRAIN5, SIX, TODAY = G.TRAIN5, G.SIX, G.TODAY
GATE = "g:"
log, utc, write_json, sha_file = G.log, G.utc, G.write_json, G.sha_file


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the candidates ───────────────────────────────────────────────────────────


def candidates():
    """[(name, P1g's candidate it reads, gated)]: P1g's 805 as filed, then the 798 gated ones in P1g's order."""
    base = G.candidates()
    return [(n, n, False) for n, _f, _ex in base] + [(GATE + n, n, True) for n, _f, ex in base if ex != "none"]


def none_of(name):
    """The same fill fraction's `none` candidate: f2+cut_cos_3000 -> f2+none."""
    return name.split("+")[0] + "+none"


def gate_arrays(a, names, cands):
    """P1g's per-carve arrays (rows, held, ptr, fill over the candidates `names`) -> the same arrays over `cands`."""
    col = {n: i for i, n in enumerate(names)}
    open_ = np.asarray(a["fill"]) > 0
    gq = np.repeat(open_, np.diff(a["ptr"]))
    rows = np.empty((a["rows"].shape[0], len(cands)), np.int32)
    held = np.empty((a["held"].shape[0], len(cands)), bool)
    for i, (_nm, src, gated) in enumerate(cands):
        r, h = a["rows"][:, col[src]], a["held"][:, col[src]]
        if gated:
            r = np.where(open_, r, a["rows"][:, col[none_of(src)]])
            h = np.where(gq, h, a["held"][:, col[none_of(src)]])
        rows[:, i], held[:, i] = r, h
    return {"rows": rows, "held": held, "ptr": a["ptr"], "fill": a["fill"]}


def summarize(a, cands):
    return G.summarize(a, [(nm, None, None) for nm, _s, _g in cands])


def load_arrays(name, carve, out_name="stats", root=None):
    """P1g's filed arrays for one carve, checked against its record (P1g's script, its file sha, its checks)."""
    root = root or G.OUT
    rec = json.loads((root / f"{out_name}_{name}.json").read_text(encoding="utf-8"))
    if rec.get("script_sha256") != G.script_sha256():
        raise SystemExit(f"{out_name}_{name}.json: not written by P1g's script")
    e = rec["carves"][f"{name}__{carve}"]
    p = root / f"{out_name}_{name}__{carve}.npz"
    if sha_file(p) != e["file_sha256"]:
        raise SystemExit(f"{p}: not the file the record names")
    if not all(e["checks"].values()):
        raise SystemExit(f"{p}: a check failed")
    z = np.load(p)
    a = {k: z[k] for k in z.files}
    names = rec["candidates"]
    want = G.summarize(a, [(n, None, None) for n in names])
    if want != e["summary"]:
        raise SystemExit(f"{p}: the arrays do not give the record's summary")
    return a, names, rec


# ── the verdict ──────────────────────────────────────────────────────────────


def verdict_stage():
    cands = candidates()
    order = [c[0] for c in cands]
    table, fills = {}, {}
    for d in TRAIN5:
        a, names, _rec = load_arrays(d, "s1sel")
        if names != [c[0] for c in G.candidates()]:
            raise SystemExit(f"stats_{d}.json: not P1g's 805 candidates in order")
        s = summarize(gate_arrays(a, names, cands), cands)
        for c in order:
            table.setdefault(c, {})[d] = s[c]
        f = np.asarray(a["fill"])
        fills[d] = {"questions": int(f.size), "fill_zero": int((f == 0).sum())}
    w = G.F.weights()
    verdict, pick, ceil, tgt, summ = G.choose(table, w)
    wa, wn, _ = load_arrays("webqsp", "s1eval")
    beside = summarize(gate_arrays(wa, wn, cands), cands)
    fills["webqsp (s1eval, beside)"] = {"questions": int(wa["fill"].size), "fill_zero": int((wa["fill"] == 0).sum())}
    p1g = json.loads(P1G_VERDICT.read_text(encoding="utf-8")) if P1G_VERDICT.exists() else None
    if p1g is not None and p1g.get("script_sha256") != G.script_sha256():
        raise SystemExit(f"{P1G_VERDICT}: not written by P1g's script")
    best_ungated = min((summ[c]["cost"], i, c) for i, c in enumerate(order)
                       if summ[c]["meets"] and not c.startswith(GATE)) if any(
        summ[c]["meets"] and not c.startswith(GATE) for c in order) else None
    rec = {"declared_in": DECLARED, "stage": "P1g2", "script_sha256": script_sha256(),
           "p1g_script_sha256": G.script_sha256(), "verdict": verdict, "chosen": pick,
           "chosen_reads": dict(zip(("candidate", "gated"), [c[1:] for c in cands if c[0] == pick][0])),
           "today": TODAY, "ceilings": ceil, "targets": tgt, "weights": w, "fill_zero_questions": fills,
           "chosen_s1sel": {d: table[pick][d] for d in TRAIN5}, "today_s1sel": {d: table[TODAY][d] for d in TRAIN5},
           "cost": {"chosen": summ[pick]["cost"], "today": summ[TODAY]["cost"]},
           "beside_p1g": None if p1g is None else {"verdict": p1g["verdict"], "chosen": p1g["chosen"],
                                                    "cost": p1g["cost"]["chosen"]},
           "beside_cheapest_ungated": None if best_ungated is None else {"candidate": best_ungated[2],
                                                                         "cost": best_ungated[0]},
           "beside_webqsp_s1eval": {"chosen": beside[pick], "today": beside[TODAY]},
           "summary": summ, "utc": utc()}
    write_json(VERDICT, rec)
    log(f"P1g2: {verdict}, chosen {pick} (cost {summ[pick]['cost']} against today's {summ[TODAY]['cost']})")
    return 0


def chosen_rule():
    """(P1g's candidate the chosen rule reads, gated) from P1g2's verdict, checked."""
    v = json.loads(VERDICT.read_text(encoding="utf-8"))
    if v.get("script_sha256") != script_sha256() or v.get("p1g_script_sha256") != G.script_sha256():
        raise SystemExit(f"{VERDICT}: not written by this script over P1g's")
    return v, v["chosen_reads"]["candidate"], bool(v["chosen_reads"]["gated"])


def eval_stage(name, limit=0, threads=2):
    """s1eval (beside): P1g's run_stage, unchanged, on today's and the chosen rule's ungated candidate (and its
    fraction's none), written under outputs/p1g2; the gate is applied from the arrays."""
    if name not in TRAIN5:
        raise SystemExit(f"eval reads the five training datasets' s1eval; webqsp's is in P1g's stats ({TRAIN5})")
    _v, src, gated = chosen_rule()
    want = [TODAY, src] + ([none_of(src)] if gated and none_of(src) not in (TODAY, src) else [])
    cands = [c for c in G.candidates() if c[0] in want]
    G.OUT = OUT
    rc = G.run_stage(name, ("s1eval",), cands, "eval", limit, threads)
    if limit or rc:
        return rc
    p = OUT / f"eval_{name}.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec.update({"stage": "P1g2 eval (P1g's run_stage, unchanged)", "declared_in": DECLARED,
                "p1g2_script_sha256": script_sha256()})
    write_json(p, rec)
    return 0


def eval_summary(name):
    v, src, gated = chosen_rule()
    a, names, rec = load_arrays(name, "s1eval", "eval", OUT)
    if rec.get("p1g2_script_sha256") != script_sha256():
        raise SystemExit(f"eval_{name}.json: not P1g2's")
    cs = [(TODAY, TODAY, False), (v["chosen"], src, gated)]
    return summarize(gate_arrays(a, names, cs), cs), v


def report_stage():
    v = json.loads(VERDICT.read_text(encoding="utf-8"))
    lines = [f"# P1g2's verdict: **{v['verdict']}**", "",
             f"Chosen on s1sel: `{v['chosen']}`; today's pool is `{v['today']}`. Targets: min(0.95, ceiling - 0.005).",
             "", "| dataset | carve | target | today ALL / rows | chosen ALL / rows |", "| --- | --- | ---: | --- | --- |"]
    for d in TRAIN5:
        t, c = v["today_s1sel"][d], v["chosen_s1sel"][d]
        lines.append(f"| {d} | s1sel | {v['targets'][d]:.4f} | {t['ALL']:.4f} / {t['rows']:,.0f} | "
                     f"{c['ALL']:.4f} / {c['rows']:,.0f} |")
    for d in TRAIN5:
        if (OUT / f"eval_{d}.json").exists():
            s, _ = eval_summary(d)
            lines.append(f"| {d} | s1eval (beside) | | {s[v['today']]['ALL']:.4f} / {s[v['today']]['rows']:,.0f} | "
                         f"{s[v['chosen']]['ALL']:.4f} / {s[v['chosen']]['rows']:,.0f} |")
    b = v["beside_webqsp_s1eval"]
    lines.append(f"| webqsp | s1eval (beside) | | {b['today']['ALL']:.4f} / {b['today']['rows']:,.0f} | "
                 f"{b['chosen']['ALL']:.4f} / {b['chosen']['rows']:,.0f} |")
    lines += ["", f"Cost (P1f's question-weighted mean rows): chosen {v['cost']['chosen']:,}, today "
              f"{v['cost']['today']:,}."]
    if v["beside_p1g"]:
        lines.append(f"P1g's own choice: `{v['beside_p1g']['chosen']}` ({v['beside_p1g']['verdict']}, cost "
                     f"{v['beside_p1g']['cost']:,}).")
    lines += ["", "Questions KIND_T does not expand (today's fill 0), where the gate drops the extra set:", "",
              "| dataset | questions | fill 0 |", "| --- | ---: | ---: |"]
    for d, e in v["fill_zero_questions"].items():
        lines.append(f"| {d} | {e['questions']:,} | {e['fill_zero']:,} |")
    (OUT / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    cs = candidates()
    assert len(cs) == 805 + 798 and len({c[0] for c in cs}) == len(cs)
    assert cs[:805] == [(n, n, False) for n, _f, _e in G.candidates()]
    assert all(g and n == GATE + s and not s.endswith("+none") for n, s, g in cs[805:])
    assert none_of("f2+cut_cos_3000") == "f2+none" and none_of("f0.25+beam1_res_P10_m1") == "f0.25+none"
    # three questions: fill 4, 0, 3; golds 2, 1, 1. Columns: f1+none, f1+cut, f2+none, f2+cut
    names = ["f1+none", "f1+cut_x", "f2+none", "f2+cut_x"]
    a = {"rows": np.array([[10, 15, 14, 18], [6, 9, 6, 9], [8, 8, 11, 12]], np.int32),
         "held": np.array([[True, True, True, True], [False, True, False, True],      # q0
                           [False, True, False, True],                                # q1: only the cut holds it
                           [False, False, True, True]]),                              # q2
         "ptr": np.array([0, 2, 3, 4]), "fill": np.array([4, 0, 3])}
    cands = [("f1+none", "f1+none", False), ("f1+cut_x", "f1+cut_x", False), ("g:f1+cut_x", "f1+cut_x", True),
             ("g:f2+cut_x", "f2+cut_x", True)]
    g = gate_arrays(a, names, cands)
    assert g["rows"].tolist() == [[10, 15, 15, 18], [6, 9, 6, 6], [8, 8, 8, 12]], g["rows"]
    assert g["held"].tolist() == [[True, True, True, True], [False, True, True, True], [False, True, False, False],
                                  [False, False, False, True]], g["held"]
    s = summarize(g, cands)
    assert s["f1+cut_x"]["ALL"] == round(2 / 3, 4) and s["g:f1+cut_x"]["ALL"] == round(1 / 3, 4)
    assert s["g:f1+cut_x"]["rows"] == round(29 / 3, 1) and s["g:f2+cut_x"]["ALL"] == round(2 / 3, 4)
    # an ungated candidate reads its own column unchanged
    assert g["rows"][:, 1].tolist() == a["rows"][:, 1].tolist()
    # with every fill above 0, a gated candidate is its ungated one
    b = dict(a, fill=np.array([4, 2, 3]))
    gb = gate_arrays(b, names, cands)
    assert np.array_equal(gb["rows"][:, 2], b["rows"][:, 1]) and np.array_equal(gb["held"][:, 2], b["held"][:, 1])
    # the choice is P1g's: a gated candidate wins only by cost, after every target
    w = {"a": 1, "b": 1}
    tab = {"f1+none": {"a": {"ALL": 0.96, "rows": 100}, "b": {"ALL": 0.90, "rows": 100}},
           "f1+cut": {"a": {"ALL": 0.99, "rows": 900}, "b": {"ALL": 0.96, "rows": 300}},
           "g:f1+cut": {"a": {"ALL": 0.96, "rows": 100}, "b": {"ALL": 0.96, "rows": 300}}}
    assert G.choose(tab, w, names=("a", "b"))[:2] == ("TARGET_MET", "g:f1+cut")
    tab["g:f1+cut"]["a"]["ALL"] = 0.94
    assert G.choose(tab, w, names=("a", "b"))[:2] == ("TARGET_MET", "f1+cut")
    log("p1g2 selftest: ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=("verdict", "eval", "report"))
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.stage == "eval":
        if a.host:                          # P1e's host mode, as P1g's stats and eval
            import p1e_anchor_kind as K
            K.G.on_the_mirror()
        return eval_stage(a.dataset, a.limit, a.threads)
    return {"verdict": verdict_stage, "report": report_stage}[a.stage]()


if __name__ == "__main__":
    sys.exit(main())
