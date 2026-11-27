"""LEG1 amendment A7 (docs/LEG1_LEG_NECESSITY.md section 18): the cost side measured on one card, the same texts, one arm after another.

Run as:
    python -B -E -P outputs/mp_unified/zleg1t.py selftest      # laptop: the pure helpers (no card, no model)
    python -I -B   outputs/mp_unified/zleg1t.py time           # host card: gte, granite, C0 at l0, C0 at l5 on the same 8,192 nodes of each STaRK KB

What it does. For each of Prime, MAG and Amazon it reads the first 4,096 rows of docs shards 0 and 3 of the package's nodes.jsonl (the texts the dense fills
and the SPLADE legs read) and times four arms on them, in the paths production runs them and with no encoder code of its own: gte and granite through
`zstk1d.load_part` and `zenc.encode_texts` (float16, longest first, token budgets), C0 at lever none (`zstk1g.Pooler`, A6's path) and at lever l5 (A8's path).
Three repeats, the arm order rotated by one place each repeat; each model is loaded once and warmed on 64 rows; the timer covers the encode of one KB's 8,192
rows (tokenization, host to card, forward, pooling, the copy back) with `torch.cuda.synchronize()` at both ends. The non-zeros of the two C0 arms are counted in a
separate untimed pass on the first 512 rows of each KB. The ratios are the result; no call is made here. Nothing is written but `outputs/leg1/timing_card.json`.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import zstk1 as Z  # noqa: E402  (sets CUDA_VISIBLE_DEVICES to "" at import: undone in main for `time`, before any card call)
import zstk1d as D  # noqa: E402

E = Z.E
DOC = "docs/LEG1_LEG_NECESSITY.md section 18 (LEG1-A7)"
KBS = ("prime", "mag", "amazon")
SLICES = ((0, 0), (3, 0))            # (docs shard, first row in it): STK1 A7's gate slices
ROWS = 4096                          # rows per slice: one production block
REPEATS = 3
WARM = 64
NNZ_ROWS = 512
ARMS = ("gte", "granite", "c0_l0", "c0_l5")
OUT = Z.ROOT.parent / "leg1" / "timing_card.json"


def log(msg):
    print(f"[zleg1t {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"ZLEG1T: {msg}")


def rotation(arms, r):
    """The order of the arms in repeat r: the list turned by r places."""
    k = r % len(arms)
    return list(arms[k:]) + list(arms[:k])


def rate(nodes, seconds):
    return float(nodes) / float(seconds)


def pair_rate(g, h):
    """Nodes per second of the two parts of e3v made one after the other: 1 / (1 / g + 1 / h)."""
    return 1.0 / (1.0 / float(g) + 1.0 / float(h))


def med(xs):
    return float(np.median(np.asarray(xs, np.float64)))


def summarize(recs):
    """recs: list of dicts {kb, arm, repeat, nodes, seconds, tokens}. Per KB and pooled: rates per arm (median and range over the repeats), the pair rate,
    R_5 = C0 l5 over the pair and R_0 = C0 l0 over the pair (per repeat, then median and range)."""
    kbs = sorted({r["kb"] for r in recs}, key=lambda k: KBS.index(k))
    reps = sorted({r["repeat"] for r in recs})

    def get(kb, arm, rep, key):
        sel = [r for r in recs if r["arm"] == arm and r["repeat"] == rep and (kb is None or r["kb"] == kb)]
        if kb is None and len(sel) != len(kbs) or kb is not None and len(sel) != 1:
            stop(f"records missing for {kb} {arm} repeat {rep}")
        return sum(r[key] for r in sel)

    out = {"per_kb": {}, "pooled": {}}
    for scope in kbs + [None]:
        rates = {a: [rate(get(scope, a, rp, "nodes"), get(scope, a, rp, "seconds")) for rp in reps] for a in ARMS}
        toks = {a: [rate(get(scope, a, rp, "tokens"), get(scope, a, rp, "seconds")) for rp in reps] for a in ARMS}
        pair = [pair_rate(g, h) for g, h in zip(rates["gte"], rates["granite"])]
        r5 = [x / p for x, p in zip(rates["c0_l5"], pair)]
        r0 = [x / p for x, p in zip(rates["c0_l0"], pair)]
        rec = {"nodes_per_s": {a: {"median": round(med(v), 1), "min": round(min(v), 1), "max": round(max(v), 1)} for a, v in rates.items()},
               "tokens_per_s": {a: {"median": round(med(v)), "min": round(min(v)), "max": round(max(v))} for a, v in toks.items()},
               "pair_nodes_per_s": {"median": round(med(pair), 1), "min": round(min(pair), 1), "max": round(max(pair), 1)},
               "R_5": {"median": round(med(r5), 3), "min": round(min(r5), 3), "max": round(max(r5), 3), "per_repeat": [round(x, 3) for x in r5]},
               "R_0": {"median": round(med(r0), 3), "min": round(min(r0), 3), "max": round(max(r0), 3), "per_repeat": [round(x, 3) for x in r0]}}
        if scope is None:
            out["pooled"] = rec
        else:
            out["per_kb"][scope] = rec
    return out


def expectations(summ, gate_v=None):
    """T1 R_5 above 1 on every KB; T2 R_0 below 1 on every KB; T3 pooled R_5 between 1.5 and 3.0; T4 (information) Prime's gte and granite rates within 25 percent of G-V's."""
    kb = summ["per_kb"]
    t = {"T1": all(v["R_5"]["median"] > 1.0 for v in kb.values()),
         "T2": all(v["R_0"]["median"] < 1.0 for v in kb.values()),
         "T3": bool(1.5 <= summ["pooled"]["R_5"]["median"] <= 3.0)}
    if gate_v and "prime" in kb:
        g = {e: v["nodes_per_s"] for e, v in gate_v["parts"].items()}
        mine = kb["prime"]["nodes_per_s"]
        parts = list(g)
        t["T4"] = bool(abs(mine["gte"]["median"] / g[parts[0]] - 1) <= 0.25 and abs(mine["granite"]["median"] / g[parts[1]] - 1) <= 0.25)
    return t


def texts_of(kb):
    ds = Z.KB[kb]
    path = Z.PKG / ds / "nodes.jsonl"
    offs, n = D.shard_starts(path)
    out = []
    for s, r0 in SLICES:
        t = D.shard_texts(path, offs[s], n - s * E.SHARD)[r0:r0 + ROWS]
        if len(t) != ROWS:
            stop(f"{kb}: shard {s} gave {len(t)} rows, not {ROWS}")
        out.append(t)
    sha = hashlib.sha256("\n".join("\n".join(t) for t in out).encode("utf-8")).hexdigest()
    return out, sha, n


def card_procs():
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return [x.strip() for x in r.stdout.splitlines() if x.strip()]
    except Exception as ex:                                              # best effort: the record says why it is empty
        return [f"nvidia-smi failed: {ex}"]


def cmd_time(a):
    import torch
    if not torch.cuda.is_available():
        stop("no card is visible")
    import zstk1g as G
    Z3, V = D.zmods()
    parts = list(V.PARTS)
    t0 = time.time()
    before = card_procs()
    texts = {kb: texts_of(kb) for kb in KBS}
    log("texts: " + ", ".join(f"{kb} {ROWS * len(SLICES)} rows of {texts[kb][2]:,} nodes" for kb in KBS))
    models = {"gte": D.load_part(parts[0], Z3), "granite": D.load_part(parts[1], Z3)}
    _, tok, m32 = G.load_model("cuda", 2, "none")
    _, tok16, m16 = G.load_model("cuda", 2, "l5")
    pool = {"c0_l0": G.Pooler(torch, tok, m32, 16, "none"), "c0_l5": G.Pooler(torch, tok16, m16, 32, "l5")}
    spl_batch = G.SPL_BATCH

    def run_dense(arm, kb):
        e = parts[0] if arm == "gte" else parts[1]
        toks = 0
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for sl in texts[kb][0]:
            v, st = E.encode_texts(models[arm], e, sl, "docs")
            toks += int(st["tokens"])
        torch.cuda.synchronize()
        return time.perf_counter() - t1, toks

    def run_c0(arm, kb, nrows=None):
        P = pool[arm]
        toks, nnz, n = 0, 0, 0
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for sl in texts[kb][0]:
            sl = sl if nrows is None else sl[:nrows]
            for i in range(0, len(sl), spl_batch):
                v, t, _ = P(sl[i:i + spl_batch], sl[i + spl_batch:i + 2 * spl_batch])
                toks += t
                if nrows is not None:
                    nnz += int(np.count_nonzero(v))
                n += v.shape[0]
        torch.cuda.synchronize()
        return time.perf_counter() - t1, toks, nnz, n

    def run(arm, kb):
        if arm in ("gte", "granite"):
            return run_dense(arm, kb)
        s, t, _, _ = run_c0(arm, kb)
        return s, t

    for arm in ARMS:                                                    # warm-up: 64 rows, not counted
        w = texts["prime"][0][0][:WARM]
        if arm in ("gte", "granite"):
            E.encode_texts(models[arm], parts[0] if arm == "gte" else parts[1], w, "docs")
        else:
            pool[arm](w, None)
    torch.cuda.synchronize()
    recs, orders = [], []
    for rp in range(REPEATS):
        order = rotation(ARMS, rp)
        orders.append(order)
        for arm in order:
            for kb in KBS:
                s, t = run(arm, kb)
                recs.append({"kb": kb, "arm": arm, "repeat": rp, "nodes": ROWS * len(SLICES), "seconds": round(s, 4), "tokens": int(t)})
                log(f"repeat {rp} {arm:8s} {kb:7s}: {ROWS * len(SLICES) / s:8.1f} nodes/s, {t / s:9.0f} tokens/s ({s:.2f} s)")
    nnz = {}
    for arm in ("c0_l0", "c0_l5"):
        nnz[arm] = {}
        for kb in KBS:
            _, _, z, n = run_c0(arm, kb, NNZ_ROWS)
            nnz[arm][kb] = round(z / n, 1)
    after = card_procs()
    summ = summarize(recs)
    gate_path = D.OUT / "gate_v.json"
    gate_v = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else None
    exp = expectations(summ, gate_v)
    for kb, v in summ["per_kb"].items():
        log(f"{kb}: pair {v['pair_nodes_per_s']['median']} nodes/s; C0 l5 {v['nodes_per_s']['c0_l5']['median']} (R_5 {v['R_5']['median']} "
            f"[{v['R_5']['min']}, {v['R_5']['max']}]); C0 l0 {v['nodes_per_s']['c0_l0']['median']} (R_0 {v['R_0']['median']} [{v['R_0']['min']}, {v['R_0']['max']}])")
    p = summ["pooled"]
    log(f"pooled: pair {p['pair_nodes_per_s']['median']} nodes/s; R_5 {p['R_5']['median']} [{p['R_5']['min']}, {p['R_5']['max']}]; R_0 {p['R_0']['median']} "
        f"[{p['R_0']['min']}, {p['R_0']['max']}]; expectations {exp}")
    rec = {"stage": "LEG1 A7 timing", "declared_in": DOC, "kbs": list(KBS), "rows_per_slice": ROWS, "slices": [list(x) for x in SLICES], "repeats": REPEATS,
           "arm_order_per_repeat": orders, "parts": {"gte": parts[0], "granite": parts[1]}, "texts_sha256": {kb: texts[kb][1] for kb in KBS},
           "package_nodes": {kb: texts[kb][2] for kb in KBS}, "records": recs, "summary": summ, "nnz_per_node": nnz, "expectations": exp,
           "gate_v_reference": ({e: v["nodes_per_s"] for e, v in gate_v["parts"].items()} if gate_v else None),
           "card_compute_processes": {"before": before, "after": after}, "versions": {"torch": torch.__version__},
           "code_sha256": {"outputs/mp_unified/zleg1t.py": E.sha256(Path(__file__))}, "seconds": round(time.time() - t0, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    D.wjson(OUT, rec)
    log(f"wrote {OUT}")
    return 0


def selftest():
    bad = []

    def chk(name, ok):
        print(("PASS " if ok else "FAIL ") + name)
        if not ok:
            bad.append(name)

    chk("rotation turns the list by one place per repeat and is a permutation", rotation(ARMS, 0) == list(ARMS) and rotation(ARMS, 1)[0] == ARMS[1]
        and rotation(ARMS, 4) == list(ARMS) and all(sorted(rotation(ARMS, r)) == sorted(ARMS) for r in range(6)))
    chk("every arm is first once in four repeats", sorted(rotation(ARMS, r)[0] for r in range(4)) == sorted(ARMS))
    chk("pair rate: two equal parts halve the rate", abs(pair_rate(100.0, 100.0) - 50.0) < 1e-9)
    chk("pair rate: 750 and 1,240 give 467.3 (750 * 1240 / 1990)", abs(pair_rate(750.0, 1240.0) - 750.0 * 1240.0 / 1990.0) < 1e-9 and abs(pair_rate(750.0, 1240.0) - 467.3) < 0.1)
    recs = []
    base = {"gte": 800.0, "granite": 1200.0, "c0_l0": 200.0, "c0_l5": 1000.0}
    for rp in range(REPEATS):
        for kb in KBS:
            for arm in ARMS:
                f = 1.0 + 0.05 * rp
                recs.append({"kb": kb, "arm": arm, "repeat": rp, "nodes": 8192, "seconds": 8192 / (base[arm] / f), "tokens": 8192 * 40})
    s = summarize(recs)
    pr = pair_rate(800.0, 1200.0)
    chk("summarize: the pooled rate of equal-size KBs is their common rate", abs(s["pooled"]["nodes_per_s"]["c0_l5"]["median"] - 1000.0 / 1.05) < 0.2)
    chk("summarize: R_5 and R_0 are the C0 rate over the pair rate", abs(s["pooled"]["R_5"]["median"] - round(1000.0 / pr, 3)) < 0.002
        and abs(s["pooled"]["R_0"]["median"] - round(200.0 / pr, 3)) < 0.002)
    chk("summarize: a common slowdown of all four arms leaves the ratios unchanged across repeats", len({x for x in s["pooled"]["R_5"]["per_repeat"]}) == 1)
    chk("summarize: min and max bracket the median", s["pooled"]["nodes_per_s"]["gte"]["min"] <= s["pooled"]["nodes_per_s"]["gte"]["median"]
        <= s["pooled"]["nodes_per_s"]["gte"]["max"])
    e = expectations(s)
    chk("expectations on the synthetic ratios: T1 true, T2 true, T3 true", e == {"T1": True, "T2": True, "T3": True})
    slow = [dict(r, seconds=r["seconds"] * (5 if r["arm"] == "c0_l5" else 1)) for r in recs]
    chk("expectations flip when C0 l5 is five times slower", expectations(summarize(slow))["T1"] is False)
    miss = [r for r in recs if not (r["arm"] == "gte" and r["kb"] == "mag" and r["repeat"] == 1)]
    try:
        summarize(miss)
        chk("a missing record stops the summary", False)
    except SystemExit:
        chk("a missing record stops the summary", True)
    gv = {"parts": {"a": {"nodes_per_s": 749.0}, "b": {"nodes_per_s": 1240.0}}}
    chk("T4 compares Prime's rates with G-V's within 25 percent", expectations(s, gv)["T4"] is True and expectations(summarize(
        [dict(r, seconds=r["seconds"] * 2) if r["arm"] == "gte" else r for r in recs]), gv)["T4"] is False)
    chk("the slices and repeats are as declared", SLICES == ((0, 0), (3, 0)) and ROWS == 4096 and REPEATS == 3 and len(ARMS) == 4)
    print("selftest:", "PASS" if not bad else "FAIL " + "; ".join(bad))
    return 0 if not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("time")
    a = ap.parse_args(argv)
    if a.cmd == "time":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)                     # zstk1.py hid the card at import; this row is the card's
        return cmd_time(a)
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
