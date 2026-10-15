"""The story's worked example (docs/PROJECT_STORY.md, section 2.8; docs/PROGRAM_2026_10_15.md, section 1.7): real 2wiki
questions traced from their pool to their feature rows and the lean pair's two scores. Read only; decides nothing.

    python scripts/story_example.py run --host
    python scripts/story_example.py --selftest

Inputs, all filed: M1's scores of the J5 fits on 2wiki's s1eval carve (outputs/m1/scores/{zrckp,zspk}-J5/2wiki.npz:
each row's node, gold flag, rrf z-score and score; zspk-J5 also holds the pool links), P1c2's pools
(outputs/p1c2/pools/2wiki/2wiki__s1eval.npz), P1c2's feature cache read through zrckp's carve (the rows the models
read), and the package's dev records and node texts. The questions are chosen by fixed rules, in s1eval order:
    A  two golds, both in the pool; zspk's top five holds both and zrckp's one; the other gold is linked to it
    B  the same with the models swapped
    C  two golds, both in the pool, both models' top fives holding both
Each rule's count over the carve is reported beside (A against B is M1's net of message passing on such golds).
Writes outputs/story/example_2wiki.{json,md}.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts", ROOT / "outputs" / "mp_unified"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import m1_musique_hops as M1  # noqa: E402

DS = "2wiki"
OUT = ROOT / "outputs" / "story"
POOLS = ROOT / "outputs" / "p1c2" / "pools" / DS / f"{DS}__s1eval.npz"
FAMS = ("structural", "named-entity", "kNN")
SHOW = 8
NAMES = {"rank": ("dense_rr", "splade_rr", "rrf", "agreement", "is_seed"), "dense_cos": ("dense_cos",),
         "SEED": ("cos_seed_mean", "max_cos_seed", "max_cos_rank1_seed")}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ranks(s, n):
    """Each row's rank in its question (1 = highest score; ties by pool position, as the reads count them)."""
    out = np.empty(s.size, np.int64)
    off = 0
    for k in n:
        o = np.argsort(-s[off:off + k], kind="stable")
        out[off + o] = np.arange(1, k + 1)
        off += k
    return out


def linked(u, v, a, b):
    """The families of the links joining rows a and b (either direction)."""
    m = ((u == a) & (v == b)) | ((u == b) & (v == a))
    return m


def pick(d, rule):
    """The first question of rule (A, B or C) in s1eval order, and the rule's count: (index or None, count)."""
    n, gt, gold = d["n"], d["gt"], d["gold"]
    off = np.concatenate([[0], np.cumsum(n.astype(np.int64))])
    rk = {m: d["rank"][m] for m in ("zrckp", "zspk")}
    u, v = d["u"].astype(np.int64), d["v"].astype(np.int64)
    eq = np.searchsorted(off, u, side="right") - 1          # the links are stored question by question
    if eq.size and np.any(np.diff(eq) < 0):
        raise SystemExit("story: the links are not in question order")
    first, count = None, 0
    for q in range(n.size):
        g = np.flatnonzero(gold[off[q]:off[q + 1]]) + off[q]
        if gt[q] != 2 or g.size != 2:
            continue
        inm = {m: int((rk[m][g] <= 5).sum()) for m in rk}
        if rule == "C":
            ok = inm["zrckp"] == 2 and inm["zspk"] == 2
        else:
            hi, lo = ("zspk", "zrckp") if rule == "A" else ("zrckp", "zspk")
            ok = inm[hi] == 2 and inm[lo] == 1
            if ok:
                e0, e1 = np.searchsorted(eq, q, "left"), np.searchsorted(eq, q, "right")
                ok = bool(linked(u[e0:e1], v[e0:e1], g[0], g[1]).any())
        if ok:
            count += 1
            if first is None:
                first = q
    return first, count


def node_texts(nodes_jsonl, lines, ids):
    """{line: (node id, text)} for the wanted lines, and {node id: line} for the wanted ids, in one pass."""
    rx = re.compile(rb'"node_id": "([^"]+)"')
    lines, ids = set(int(x) for x in lines), set(ids)
    text, pos = {}, {}
    with open(nodes_jsonl, "rb") as fh:
        for i, line in enumerate(fh):
            m = rx.search(line)
            nid = m.group(1).decode("utf-8") if m else None
            if nid in ids:
                pos[nid] = i
            if i in lines:
                r = json.loads(line)
                text[i] = (r["node_id"], r.get("text", ""))
    return text, pos


def short(text, words=14):
    w = str(text).split()
    return " ".join(w[:words]) + (" ..." if len(w) > words else "")


def feature_rows(q, qid, rows):
    """{row in the question: {block: [(name, value)]}} from P1c2's cache through zrckp's carve (the MLP's rows), and
    every other per-row input the carve gives (the KT input among them)."""
    torch, Z2, S, LG, _LC = M1.mods()
    LG.bind_device_ops()
    with torch.no_grad(), Z2.on_p1c2(), S.patched("zrckp"):
        c = LG.CacheCarve(DS, M1.CARVE, "2wiki", "cpu", ROOT / M1.CACHE, True)
        blocks = [b for b in LG.SETS["pick"] if b in c.widths]
        if str(c.ids[q]) != qid:
            raise SystemExit(f"story: the cache's question {q} is {c.ids[q]}, not {qid}")
        feats, _nq, base_z, _gold = c.batch(np.array([q]), blocks)
        n = int(base_z.shape[0])
        extra = sorted(k for k in set(feats) - set(blocks)
                       if torch.is_tensor(feats[k]) and feats[k].dim() == 2 and feats[k].shape[0] == n)
        out = {}
        for r in rows:
            row = {}
            for b in blocks + extra:
                x = feats[b]
                if isinstance(x, (tuple, list)):        # SEMB: (question embedding, node projections)
                    row[b] = f"learned product of the question's embedding and the node's projection ({x[1].shape[1]})"
                    continue
                vals = x[r].float().cpu().numpy().tolist()
                names = NAMES.get(b, tuple(f"{b}[{j}]" for j in range(len(vals))))
                row[b] = [(nm, round(float(val), 4)) for nm, val in zip(names, vals)]
            row["rrf_z"] = round(float(base_z[r]), 4)
            out[int(r)] = row
    return out, blocks, extra


def describe(d, q, recs, texts, pools, rule):
    off = int(np.concatenate([[0], np.cumsum(d["n"].astype(np.int64))])[q])
    n = int(d["n"][q])
    sl = slice(off, off + n)
    qid = str(d["ids"][q])
    rec = recs[qid]
    node = d["node"][sl].astype(np.int64)
    gold = np.flatnonzero(d["gold"][sl])
    pi = pools["index"][qid]
    fill = pools["nodes"][pools["ptr"][pi]:pools["ptr"][pi + 1]]          # KIND_T's first B_q nodes outside bs
    bsn = pools["bs_nodes"][pools["bs_ptr"][pi]:pools["bs_ptr"][pi + 1]]  # base and seeds, kept apart in the file
    bs = int(pools["bs_size"][pi])
    if bsn.size != bs or bsn.size + fill.size != n:
        raise SystemExit(f"story: question {qid}'s pool file holds {bsn.size} + {fill.size} nodes, the read {n} rows")
    where = {**{int(x): f"KIND_T fill {k + 1}" for k, x in enumerate(fill)}, **{int(x): "base or seed" for x in bsn}}
    show = sorted(set(np.argsort(d["rank"]["zrckp"][sl])[:SHOW].tolist()) | set(np.argsort(d["rank"]["zspk"][sl])[:SHOW]
                                                                                 .tolist()) | set(gold.tolist()))
    title = dict(zip(rec["context_node_ids"], rec["context_titles"]))

    def row_of(r):
        nid, txt = texts[int(node[r])]
        k = where.get(int(node[r]))
        if k is None:
            raise SystemExit(f"story: row {r} of question {qid} is in neither part of the pool file")
        return {"row": int(r), "node": nid, "title": title.get(nid), "text": short(txt),
                "gold": bool(d["gold"][off + r]),
                "pool": k,
                "rrf_z": round(float(d["rrf"][off + r]), 4),
                **{m: {"score": round(float(d["s"][m][off + r]), 4), "rank": int(d["rank"][m][off + r])}
                   for m in ("zrckp", "zspk")}}

    u, v, f = d["u"].astype(np.int64) - off, d["v"].astype(np.int64) - off, d["f"]
    sel = (u >= 0) & (u < n) & (v >= 0) & (v < n)
    u, v, f = u[sel], v[sel], f[sel]
    keep = np.isin(u, show) & np.isin(v, show)
    links = sorted({(int(min(a, b)), int(max(a, b)), FAMS[int(c)]) for a, b, c in zip(u[keep], v[keep], f[keep])})
    deg = {int(r): int(((u == r) | (v == r)).sum()) for r in show}
    return {"rule": rule, "index": int(q), "id": qid, "question": rec["question"], "answer": rec.get("answer"),
            "evidences": rec.get("evidences"), "gold_titles": rec.get("gold_refs"),
            "pool": {"rows": n, "base_and_seeds": bs, "today_size": int(pools["inc_size"][pi]),
                     "entity_question": bool(pools["entity"][pi]), "route": bool(pools["route"][pi]),
                     "links_in_pool": int(sel.sum())},
            "rows": [dict(row_of(r), links=deg[int(r)]) for r in show],
            "links": [{"a": a, "b": b, "family": c} for a, b, c in links]}


def load(host):
    z = {m: np.load(M1.SCORES / f"{m}-J5" / f"{DS}.npz") for m in ("zrckp", "zspk")}
    for k in ("ids", "n", "gt", "gold", "node", "rrf"):
        if not np.array_equal(z["zrckp"][k], z["zspk"][k]):
            raise SystemExit(f"story: the two arms' {k} differ")
    b = z["zspk"]
    d = {k: b[k] for k in ("ids", "n", "gt", "gold", "node", "rrf", "u", "v", "f")}
    d["s"] = {m: z[m]["s"] for m in z}
    d["rank"] = {m: ranks(d["s"][m], d["n"]) for m in z}
    p = np.load(POOLS)
    pools = {k: p[k] for k in ("ids", "ptr", "nodes", "bs_ptr", "bs_nodes", "bs_size", "inc_size", "entity", "route")}
    pools["index"] = {str(x): i for i, x in enumerate(pools["ids"])}
    return d, pools


def run(host):
    t0 = time.time()
    d, pools = load(host)
    chosen, counts = {}, {}
    for rule in ("A", "B", "C"):
        chosen[rule], counts[rule] = pick(d, rule)
        log(f"rule {rule}: question {chosen[rule]}, {counts[rule]} on the carve")
    qs = [q for q in chosen.values() if q is not None]
    root = M1.package_root(host) / "data" / "final_canonical" / DS
    recs = M1.query_records(root / "queries", [str(d["ids"][q]) for q in qs])
    off = np.concatenate([[0], np.cumsum(d["n"].astype(np.int64))])
    lines = {int(x) for q in qs for x in d["node"][off[q]:off[q + 1]]}
    want = {g for r in recs.values() for g in r["gold_node_ids"]}
    texts, pos = node_texts(root / "nodes.jsonl", lines, want)
    out = {"dataset": DS, "carve": M1.CARVE, "fits": {"zrckp": M1.FITS[("zrckp", "J5")],
                                                      "zspk": M1.FITS[("zspk", "J5")]},
           "counts": counts, "questions": {}}
    for rule, q in chosen.items():
        if q is None:
            continue
        r = recs[str(d["ids"][q])]
        gl = {pos[g] for g in r["gold_node_ids"] if g in pos}
        rows = np.arange(off[q], off[q + 1])
        if set(d["node"][rows][d["gold"][rows]].tolist()) != gl:
            raise SystemExit(f"story: question {q}: the gold rows are not the record's gold nodes")
        e = describe(d, q, recs, texts, pools, rule)
        fr, blocks, extra = feature_rows(q, e["id"], [x["row"] for x in e["rows"] if x["gold"]]
                                         + [min((x for x in e["rows"] if not x["gold"]),
                                                key=lambda x: x[m]["rank"])["row"] for m in ("zrckp", "zspk")])
        for x in e["rows"]:
            if x["row"] in fr:
                x["features"] = fr[x["row"]]
                if abs(fr[x["row"]]["rrf_z"] - x["rrf_z"]) > 1e-3:
                    raise SystemExit(f"story: question {q} row {x['row']}: the cache's rrf z-score is not M1's")
        e["blocks"], e["other_inputs"] = blocks, extra
        out["questions"][rule] = e
    out.update(seconds=round(time.time() - t0, 1), script_sha256=M1.sha_src(__file__),
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    OUT.mkdir(parents=True, exist_ok=True)
    M1.write_json(OUT / "example_2wiki.json", out)
    (OUT / "example_2wiki.md").write_text("\n".join(markdown(out)) + "\n", encoding="utf-8")
    log(f"story: {', '.join(f'{k} q{v}' for k, v in chosen.items())}; {time.time() - t0:.0f}s")
    return out


def markdown(out):
    md = [f"# Worked examples on {out['dataset']} s1eval (J5 fits)", "",
          "Rules (in s1eval order): A, zspk's top five holds both golds and zrckp's one, the two golds linked; B, the "
          "reverse; C, both models hold both. Counts on the carve: " +
          ", ".join(f"{k} {v}" for k, v in out["counts"].items()) + ".", ""]
    for rule, e in out["questions"].items():
        md += [f"## Rule {rule}: {e['question']}", "",
               f"Answer: {e['answer']}. Golds: {', '.join(e['gold_titles'])}.",
               f"Pool: {e['pool']['rows']} rows ({e['pool']['base_and_seeds']} base and seeds, the rest KIND_T's "
               f"fill); entity question {e['pool']['entity_question']}; {e['pool']['links_in_pool']} links inside "
               "the pool.", "",
               "| row | title / text | gold | pool | rrf z | zrckp score (rank) | zspk score (rank) | links |",
               "|---:|---|---|---|---:|---:|---:|---:|"]
        for x in sorted(e["rows"], key=lambda x: x["zspk"]["rank"]):
            md.append(f"| {x['row']} | {x['title'] or x['text']} | {'yes' if x['gold'] else ''} | {x['pool']} | "
                      f"{x['rrf_z']:+.3f} | {x['zrckp']['score']:+.3f} ({x['zrckp']['rank']}) | "
                      f"{x['zspk']['score']:+.3f} ({x['zspk']['rank']}) | {x['links']} |")
        md += ["", "Links among the rows shown: " + ("; ".join(f"{k['a']}-{k['b']} {k['family']}" for k in e["links"])
                                                     or "none") + ".", ""]
        for x in e["rows"]:
            if "features" in x:
                md.append(f"Features of row {x['row']} ({'gold' if x['gold'] else 'not gold'}):")
                for b, vals in x["features"].items():
                    md.append(f"- {b}: {vals if isinstance(vals, str) else ', '.join(f'{k} {v}' for k, v in vals)}"
                              if b != "rrf_z" else f"- rrf z-score (the base score): {vals}")
                md.append("")
    return md


def selftest():
    n = np.array([3, 4])
    s = np.array([0.1, 0.9, 0.5, 2.0, 2.0, -1.0, 0.0])
    assert ranks(s, n).tolist() == [3, 1, 2, 1, 2, 4, 3]
    # Two questions of 7 rows, two golds each. q0: zspk holds both (rows 0, 1), zrckp only row 0; they are linked.
    gold = np.zeros(14, bool)
    gold[[0, 1, 7, 8]] = True
    d = {"n": np.array([7, 7]), "gt": np.array([2, 2]), "gold": gold,
         "u": np.array([0, 7]), "v": np.array([1, 9]),
         "rank": {"zspk": np.array([1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7]),
                  "zrckp": np.array([1, 6, 2, 3, 4, 5, 7, 1, 2, 3, 4, 5, 6, 7])}}
    assert pick(d, "A") == (0, 1) and pick(d, "B") == (None, 0) and pick(d, "C") == (1, 1)
    d["u"] = np.array([0, 7])
    d["v"] = np.array([2, 9])                                  # the golds of q0 no longer linked
    assert pick(d, "A") == (None, 0)
    assert short("a b c", 2) == "a b ..." and short("a b", 2) == "a b"
    # the pools file keeps base and seeds apart (bs_nodes) from KIND_T's fill (nodes); a row is labelled from both
    d = {"n": np.array([4]), "ids": np.array(["q"]), "node": np.array([30, 10, 20, 11]), "gold": np.array([0, 1, 0, 0]),
         "rrf": np.zeros(4), "u": np.array([0]), "v": np.array([1]), "f": np.array([2]),
         "s": {"zrckp": np.arange(4.0), "zspk": np.arange(4.0)}}
    d["rank"] = {m: ranks(d["s"][m], d["n"]) for m in d["s"]}
    pools = {"index": {"q": 0}, "ptr": np.array([0, 2]), "nodes": np.array([20, 30]), "bs_ptr": np.array([0, 2]),
             "bs_nodes": np.array([10, 11]), "bs_size": np.array([2]), "inc_size": np.array([4]),
             "entity": np.array([False]), "route": np.array([False])}
    recs = {"q": {"question": "?", "context_node_ids": [], "context_titles": []}}
    texts = {x: (f"n{x}", "t") for x in (10, 11, 20, 30)}
    got = {r["row"]: r["pool"] for r in describe(d, 0, recs, texts, pools, "A")["rows"]}
    assert got == {0: "KIND_T fill 2", 1: "base or seed", 2: "KIND_T fill 1", 3: "base or seed"}, got
    pools["bs_nodes"] = np.array([10, 12])                     # a row in neither part stops the script
    try:
        describe(d, 0, recs, texts, pools, "A")
        raise AssertionError("a row outside the pool file passed")
    except SystemExit:
        pass
    print("story_example selftest: ok", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("run",))
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd != "run":
        ap.error("run or --selftest")
    run(a.host)
    return 0


if __name__ == "__main__":
    sys.exit(main())
