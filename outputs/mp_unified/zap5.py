"""AP5: does a question's type change which of D83's link labels points from what the first stage found to what it missed?
(docs/AP5_TYPED_LABEL_PRECISION.md)

AP4's filed ledgers (outputs/ap4/ledger/<arm>/<ds>.npz, sha-checked by zap4.load_ledger) are split by the datasets' own
question types and read with zap4's measures. Nothing trains, no model is read, and no ledger, pool or link is written.
Outputs go only under outputs/ap5/.

    python outputs/mp_unified/zap5.py selftest      # synthetic ledgers (laptop or host)
    python outputs/mp_unified/zap5.py report        # the host (the ledgers' npz live there)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import zap4 as A4  # noqa: E402  (load_ledger, units, per_question, unit_block, qsel_of, boot, ratio, call; read only)

DECLARED = "docs/AP5_TYPED_LABEL_PRECISION.md"
STAGE = "AP5"
OUT = ROOT / "outputs" / "ap5"
LOOK = ROOT / "outputs" / "mp_unified" / "look"
DECIDE = "D83E"
READS = {"D83E": ("s1eval", ("hotpotqa", "2wiki", "musique", "metaqa")),
         "B1V": ("b1", ("hotpotqa", "2wiki", "musique"))}
TYPES = {"hotpotqa": ("bridge", "comparison"),
         "2wiki": ("compositional", "comparison", "bridge_comparison", "inference"),
         "musique": ("2", "3", "4"),
         "metaqa": ("1", "2", "3")}
MIN_Q = 100          # a type is read when it has at least this many questions at the top1 anchor
NEED = 2             # TYPED when at least this many of the deciding arm's four datasets FLIP


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── the types (G1, G2) ───────────────────────────────────────────────────────


def types_from(ds, ids, info=None):
    """Each question's type: the dataset's own field (hotpotqa, 2wiki), the hop count in the id (musique 'Nhop..',
    metaqa 'metaqa:Nhop:..'). Every type must be one of TYPES[ds] (G2)."""
    if ds in ("hotpotqa", "2wiki"):
        if info is None or len(info) != len(ids):
            raise SystemExit(f"AP5: {ds}: {0 if info is None else len(info)} info rows against {len(ids)} ids")
        t = [str(x.get("type")) for x in info]
    elif ds == "musique":
        t = [s[0] if len(s) > 4 and s[1:4] == "hop" else "?" for s in ids]
    elif ds == "metaqa":
        t = [s.split(":")[1][0] if s.startswith("metaqa:") and s.split(":")[1][1:] == "hop" else "?" for s in ids]
    else:
        raise SystemExit(f"AP5: no types for {ds}")
    bad = sorted({x for x in t if x not in TYPES[ds]})
    if bad:
        raise SystemExit(f"AP5: G2: {ds} has types outside {TYPES[ds]}: {bad[:5]}")
    return np.asarray(t)


def types_of(ds, carve, meta, z):
    """G1: the look's ids.json must hash to the ledger record's ids_sha256 (so position i is question i). On MuSiQue the
    types must equal the ledger's own hop column."""
    d = LOOK / ds / carve
    ids = [str(x) for x in json.loads((d / "ids.json").read_text(encoding="utf-8"))]
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != meta["ids_sha256"] or len(ids) != z["Q_n"].size:
        raise SystemExit(f"AP5: G1: {d}/ids.json is not the ledger's question list")
    info = json.loads((d / "info.json").read_text(encoding="utf-8")) if ds in ("hotpotqa", "2wiki") else None
    t = types_from(ds, ids, info)
    if ds == "musique" and not np.array_equal(t.astype(np.int64), z["Q_hop"]):
        raise SystemExit("AP5: G1: MuSiQue's id hops disagree with the ledger's hop column")
    return t, hashlib.sha256((d / "ids.json").read_bytes()).hexdigest()


# ── the measures ─────────────────────────────────────────────────────────────


def type_block(z, U, tmask):
    """One type's measures, per kind: each label with pairs (zap4.unit_block, with its precision against any link's),
    any link, the paired precision difference of every label pair, and the share of questions whose in-pool golds are
    all in rrf's top five (where no link can add a gold)."""
    labels = [u for u in U if u.startswith("label:")]
    rec = {}
    for kind, kn in enumerate(A4.KINDS):
        qs = A4.qsel_of(z, kind) & tmask
        anys, _ = A4.per_question(z, U["any"], kind)
        r = {"questions": int(qs.sum()), "units": {"any": A4.unit_block(z, U["any"], kind, qs)}}
        pq = {}
        for un in labels:
            b = A4.unit_block(z, U[un], kind, qs, anys)
            if b["pairs"] == 0:
                continue
            r["units"][un] = b
            pq[un], _ = A4.per_question(z, U[un], kind)
        names = list(pq)
        r["label_pairs"] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = pq[names[i]], pq[names[j]]
                d = A4.boot(lambda s: A4.ratio(s["na"], s["da"]) - A4.ratio(s["nb"], s["db"]),
                            {"na": a["num"][qs], "da": a["den"][qs], "nb": b["num"][qs], "db": b["den"][qs]})
                r["label_pairs"][f"{names[i][6:]} - {names[j][6:]}"] = {"diff": d, "call": A4.call(d)}
        inpool = z["Q_inpool"][qs] > 0
        r["all_inpool_golds_in_top5"] = round(float(((z["Q_missed"][qs] == 0) & inpool).sum() / max(int(inpool.sum()), 1)), 5)
        rec[kn] = r
    return rec


def flips(blocks, kind="top1", min_q=MIN_Q):
    """The label pairs whose paired precision difference is ABOVE in one read type and BELOW in another. A type is read
    when it has at least min_q questions at that kind."""
    read = {t: b[kind] for t, b in blocks.items() if b[kind]["questions"] >= min_q}
    out = []
    keys = sorted({k for b in read.values() for k in b["label_pairs"]})
    for k in keys:
        up = sorted(t for t, b in read.items() if (b["label_pairs"].get(k) or {}).get("call") == "ABOVE")
        dn = sorted(t for t, b in read.items() if (b["label_pairs"].get(k) or {}).get("call") == "BELOW")
        if up and dn:
            out.append({"pair": k, "ABOVE_in": up, "BELOW_in": dn})
    return {"types_read": sorted(read), "types_too_small": sorted(set(blocks) - set(read)), "flips": out,
            "call": "FLIP" if out else "NO_FLIP"}


def decide(per_ds):
    """TYPED when at least NEED of the deciding arm's datasets FLIP at the top1 anchor; NO_TYPE_SIGNAL otherwise."""
    n = sum(1 for r in per_ds.values() if r["top1"]["call"] == "FLIP")
    return {"flipping_datasets": n, "of": len(per_ds), "need": NEED, "verdict": "TYPED" if n >= NEED else "NO_TYPE_SIGNAL"}


def read_one(arm, ds):
    carve, _ = READS[arm]
    got = A4.load_ledger(arm, ds)
    if got is None:
        raise SystemExit(f"AP5: no filed ledger {arm}/{ds}")
    z, meta = got
    t, ids_sha = types_of(ds, carve, meta, z)
    U = A4.units(arm, ds)
    blocks = {}
    for ty in TYPES[ds]:
        m = t == ty
        log(f"{arm} {ds}: type {ty} ({int(m.sum())} questions)")
        blocks[ty] = type_block(z, U, m)
    return {"carve": carve, "ledger_npz_sha256": meta["npz_sha256"], "ids_json_sha256": ids_sha,
            "type_counts": {ty: int((t == ty).sum()) for ty in TYPES[ds]}, "types": blocks,
            "top1": flips(blocks, "top1"), "found": flips(blocks, "found")}


def report():
    t0 = time.time()
    rec = {"declared_in": DECLARED, "stage": STAGE, "zap5_sha256": hashlib.sha256((HERE / "zap5.py").read_bytes()).hexdigest(),
           "min_q": MIN_Q, "need": NEED, "reads": {}}
    for arm, (_carve, dss) in READS.items():
        for ds in dss:
            rec["reads"].setdefault(arm, {})[ds] = read_one(arm, ds)
    rec["verdict"] = decide({ds: r for ds, r in rec["reads"][DECIDE].items()})
    rec["beside"] = {arm: {ds: r["top1"]["call"] for ds, r in v.items()} for arm, v in rec["reads"].items() if arm != DECIDE}
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "report.json.tmp"
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    tmp.replace(OUT / "report.json")
    write_md(rec, OUT / "report.md")
    log(f"AP5: {rec['verdict']}; written {OUT}")
    return rec


def ci_txt(x, scale=100.0):
    if x is None:
        return "-"
    return f"{scale * x['pt']:+.1f} [{scale * x['ci'][0]:+.1f}, {scale * x['ci'][1]:+.1f}]"


def write_md(rec, path):
    L = [f"# AP5: label precision by question type ({rec['utc']})", "",
         f"Declared in `{rec['declared_in']}`. AP4's filed ledgers split by the datasets' own question types. Precision in %;",
         "label pairs are paired differences in points (95%, 2,000 resamples of the type's questions, seed 0).", "",
         f"**Verdict ({DECIDE} decides): {rec['verdict']['verdict']}** - {rec['verdict']['flipping_datasets']} of "
         f"{rec['verdict']['of']} datasets FLIP at the top1 anchor (need {rec['verdict']['need']}). Beside: {rec['beside']}.", ""]
    for arm, v in rec["reads"].items():
        for ds, r in v.items():
            L += [f"## {arm} {ds} ({r['carve']}): {r['type_counts']}", "",
                  f"top1: {r['top1']['call']} {r['top1']['flips']} | found: {r['found']['call']} {r['found']['flips']}", "",
                  "| type | kind | questions | all golds in top 5 | unit | pairs/anchor | precision | lift | vs any |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for ty, b in r["types"].items():
                for kn in A4.KINDS:
                    x = b[kn]
                    for un, u in x["units"].items():
                        va = (u.get("vs_any") or {})
                        L.append(f"| {ty} | {kn} | {x['questions']} | {x['all_inpool_golds_in_top5']} | {un} | {u['price']} | "
                                 f"{ci_txt(u['precision'])} | {ci_txt(u['lift'], 1.0)} | "
                                 f"{ci_txt(va.get('diff'))} {va.get('call') or ''} |")
            L += ["", "| type | kind | label pair | difference | call |", "|---|---|---|---|---|"]
            for ty, b in r["types"].items():
                for kn in A4.KINDS:
                    for k, d in b[kn]["label_pairs"].items():
                        L.append(f"| {ty} | {kn} | {k} | {ci_txt(d['diff'])} | {d['call']} |")
            L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ── the self-test ────────────────────────────────────────────────────────────


def toy_ledger(spec, rng):
    """Questions built so that, per question, the found gold at rrf's first row links to one missed gold and to a few
    non-golds, with label a or b as spec says: spec is a list of (count, label_of_gold_link, label_of_nongold_links)."""
    acc = A4.Acc()
    for count, lg, ln in spec:
        for _ in range(count):
            n = 20
            g = np.zeros(n, bool)
            g[0] = True
            g[10] = True
            rrf = np.linspace(1.0, 0.0, n)
            cos = rng.random(n)
            u = [0] + [0] * 4
            v = [10] + [12, 13, 14, 15]
            f = [lg] + [ln] * 4
            if rng.random() < 0.3:                      # some noise: a non-gold link of the gold's label
                u.append(0)
                v.append(16)
                f.append(lg)
            u, v, f = np.array(u), np.array(v), np.array(f)
            q, rows, _top = A4.question_pairs(n, g, int(g.sum()), rrf, cos, np.r_[u, v], np.r_[v, u], np.r_[f, f])
            acc.add(n, int(g.sum()), 0, q, rows)
    return acc.arrays()


def selftest():
    rng = np.random.default_rng(3)
    U = {"label:a": 1, "label:b": 2, "any": 3}
    # 1. a flip: a is the precise label in type X, b in type Y
    z = toy_ledger([(150, 0, 1), (150, 1, 0)], rng)
    t = np.array(["X"] * 150 + ["Y"] * 150)
    blocks = {ty: type_block(z, U, t == ty) for ty in ("X", "Y")}
    assert blocks["X"]["top1"]["label_pairs"]["a - b"]["call"] == "ABOVE"
    assert blocks["Y"]["top1"]["label_pairs"]["a - b"]["call"] == "BELOW"
    f = flips(blocks)
    assert f["call"] == "FLIP" and f["flips"][0]["pair"] == "a - b" and f["flips"][0]["ABOVE_in"] == ["X"]
    # the type split partitions the pairs and the questions
    whole = type_block(z, U, np.ones(300, bool))
    for kn in A4.KINDS:
        for un in ("any", "label:a", "label:b"):
            assert sum(blocks[ty][kn]["units"][un]["pairs"] for ty in blocks) == whole[kn]["units"][un]["pairs"]
        assert sum(blocks[ty][kn]["questions"] for ty in blocks) == whole[kn]["questions"]
    # 2. no flip: a is precise in both types
    z2 = toy_ledger([(150, 0, 1), (150, 0, 1)], rng)
    b2 = {ty: type_block(z2, U, t == ty) for ty in ("X", "Y")}
    assert flips(b2)["call"] == "NO_FLIP"
    # 3. a type below MIN_Q is not read
    t3 = np.array(["X"] * 150 + ["Y"] * 60 + ["X"] * 90)
    b3 = {ty: type_block(z, U, t3 == ty) for ty in ("X", "Y")}
    assert flips(b3)["types_too_small"] == ["Y"]
    # 4. the verdict
    mk = lambda c: {"top1": {"call": c}}  # noqa: E731
    assert decide({"a": mk("FLIP"), "b": mk("FLIP"), "c": mk("NO_FLIP"), "d": mk("NO_FLIP")})["verdict"] == "TYPED"
    assert decide({"a": mk("FLIP"), "b": mk("NO_FLIP"), "c": mk("NO_FLIP"), "d": mk("NO_FLIP")})["verdict"] == "NO_TYPE_SIGNAL"
    # 5. the types from ids and info (G2)
    assert types_from("musique", ["2hop__1_2", "3hop1__1", "4hop3__9"]).tolist() == ["2", "3", "4"]
    assert types_from("metaqa", ["metaqa:1hop:dev:0", "metaqa:3hop:dev:4"]).tolist() == ["1", "3"]
    assert types_from("hotpotqa", ["a", "b"], [{"type": "bridge"}, {"type": "comparison"}]).tolist() == ["bridge", "comparison"]
    for bad in (lambda: types_from("musique", ["xhop"]), lambda: types_from("hotpotqa", ["a"], [{"type": "None"}]),
                lambda: types_from("2wiki", ["a", "b"], [{"type": "inference"}])):
        try:
            bad()
        except SystemExit:
            pass
        else:
            raise AssertionError("G2 did not stop")
    # 6. the report's text on the toy blocks
    tmp = OUT / "selftest"
    tmp.mkdir(parents=True, exist_ok=True)
    rec = {"declared_in": DECLARED, "utc": "selftest", "verdict": decide({"toy": {"top1": f}}), "beside": {},
           "reads": {"TOY": {"toy": {"carve": "toy", "type_counts": {"X": 150, "Y": 150}, "types": blocks, "top1": f,
                                     "found": flips(blocks, "found")}}}}
    write_md(rec, tmp / "report.md")
    assert "a - b" in (tmp / "report.md").read_text(encoding="utf-8")
    print("AP5 selftest OK")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["selftest"]:
        selftest()
    elif argv[:1] == ["report"]:
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
