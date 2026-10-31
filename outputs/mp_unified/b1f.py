"""B1f (docs/B1F_W2S_ON_B1.md): B1a's three settings with the structural family replaced by U2b's chosen link rule (W2_S), cut to
B1's nodes; the rule against the hyperlinks; fit copies for the reads; the report. Trains nothing and chooses nothing.

settings   (laptop) outputs/bench/hipporag2_w/<dataset>/: B1a's files byte for byte, graph_structural.npz = W2_S's undirected
           neighbours among the setting's nodes (rows = nodes.json package_row), both directions; build.json is B1a's with that
           file's sha replaced, a "u1a" block (variant "w", carve "b1w": look_b1u's checks read it) and a "b1f" block.
           outputs/bench/hipporag2_w/recovery.json: the graph against B1a's structural graph (the hyperlinks on 2wiki and
           hotpotqa, title mentions on musique) and, beside it, U1a's rule's recovery as filed.
fork       u1a.fork with the variant "w" and outputs/b1f as its root: copies a B1b fit source into outputs/b1f/fits/w/<arm>/<split>.
report     the reads of b1 (B1b), b1t and b1u (U1a) and b1w (here), per setting, arm and read, with the published rows beside.

    python outputs/mp_unified/b1f.py settings | fork --variant w --arm A --split S --src DIR | report | --selftest
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import u1a  # noqa: E402

ROOT = u1a.ROOT
DECLARED = "docs/B1F_W2S_ON_B1.md"
VAR_W = ROOT / "outputs" / "bench" / "hipporag2_w"
OUT = ROOT / "outputs" / "b1f"
U2B = ROOT / "outputs" / "u2b"
GRAPH = {"hotpotqa": "W2_S.npz", "2wiki": "W2_S.d.npz", "musique": "W2_S.npz"}      # 2wiki's file is W2_S (142,967,492 neighbours)
SETTINGS, FILES, STRUCT = u1a.SETTINGS, u1a.FILES, u1a.STRUCT
CARVES = ("b1", "b1t", "b1u", "b1w")
# the published rows (paper/www2027_v2/fact_sheet.md; R@5 x100 on HippoRAG 2's three subsets; None = not comparable or not reported)
# first field: peer-reviewed. Caveats: BridgeRAG's musique set is another set, SAG2 does not say its questions are HippoRAG 2's.
PUBLISHED = {"HippoRAG 2": (True, {"2wiki": 90.4, "hotpotqa": 96.3, "musique": 74.7}),
             "G-reasoner": (True, {"2wiki": 98.2, "hotpotqa": None, "musique": 74.9}),
             "PROPEX-RAG": (True, {"2wiki": 98.1, "hotpotqa": 97.1, "musique": None}),
             "CoopRAG": (True, {"2wiki": 97.2, "hotpotqa": 96.6, "musique": 75.7}),
             "PropRAG": (True, {"2wiki": 94.1, "hotpotqa": 97.4, "musique": 78.3}),
             "HGRAG": (True, {"2wiki": 93.0, "hotpotqa": 95.5, "musique": 74.1}),
             "EviReform": (False, {"2wiki": 98.45, "hotpotqa": 97.30, "musique": 78.93}),
             "BridgeRAG": (False, {"2wiki": 95.27, "hotpotqa": 98.75, "musique": None}),
             "SAG2": (False, {"2wiki": 97.35, "hotpotqa": 98.45, "musique": 80.68})}


def best_published(ds, peer_only):
    return max((v[ds], n) for n, (peer, v) in PUBLISHED.items() if v[ds] is not None and (peer or not peer_only))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] b1f: {m}", flush=True)


def cut(indptr, col, rows):
    """W2_S's neighbours among the setting's nodes as (src, dst) positions, both directions, sorted and distinct. rows[i] is
    node i's package row; two nodes may share a row (musique repeats two passages): each links to all of the other's neighbours."""
    rows = np.asarray(rows, np.int64)
    at = {}
    for i, r in enumerate(rows.tolist()):
        at.setdefault(r, []).append(i)
    inb = np.zeros(indptr.size - 1, bool)
    inb[rows] = True
    s, t = [], []
    for r, ps in at.items():
        nb = col[indptr[r]:indptr[r + 1]]
        for q in nb[inb[nb]].tolist():
            for i in ps:
                for j in at[q]:
                    if i != j:
                        s.append(i)
                        t.append(j)
    if not s:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    k = np.unique(np.asarray(s, np.int64) * (rows.size + 1) + np.asarray(t, np.int64))
    return (k // (rows.size + 1)).astype(np.int32), (k % (rows.size + 1)).astype(np.int32)


def settings():
    t0 = time.time()
    plain = json.loads((u1a.VAR["u"] / "recovery.json").read_text(encoding="utf-8"))
    rec = {"declared_in": DECLARED, "graph": "U2b W2_S (outputs/u2b/<dataset>/graphs), cut to the setting's nodes", "settings": {}}
    for name in SETTINGS:
        d = u1a.BENCH / name
        build = json.loads((d / "build.json").read_text(encoding="utf-8"))
        bad = [f for f in FILES if u1a.sha_file(d / f) != build["files"].get(f)]
        if bad:
            raise SystemExit(f"{name}: {bad} differ from B1a's build.json")
        nodes = json.loads((d / "nodes.json").read_text(encoding="utf-8"))
        rows = np.asarray([n["package_row"] for n in nodes], np.int64)
        gp = U2B / name / "graphs" / GRAPH[name]
        z = np.load(gp)
        ws, wd = cut(z["indptr"], z["col"], rows)
        with np.load(d / STRUCT) as h:
            hs, hd = h["src"].astype(np.int32), h["dst"].astype(np.int32)
        found, truth = u1a.row_pairs(rows, ws, wd), u1a.row_pairs(rows, hs, hd)
        s = {"passages": len(nodes), "b1": u1a.graph_stats(len(nodes), hs, hd), "b1w": u1a.graph_stats(len(nodes), ws, wd),
             "b1_structural_is": "title mentions (the package's graph)" if name == "musique" else "the corpus's hyperlinks",
             "w2s_vs_b1_undirected": u1a.pr(u1a.undirected(found), u1a.undirected(truth)),
             "plain_rule_vs_b1_undirected_as_filed": plain["settings"][name].get("recovery_undirected"),
             "graph_file": f"outputs/u2b/{name}/graphs/{gp.name}", "graph_file_sha256": u1a.sha_file(gp)}
        out = VAR_W / name
        out.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in FILES:
            if f != STRUCT:
                shutil.copyfile(d / f, out / f)
                if u1a.sha_file(out / f) != build["files"][f]:
                    raise SystemExit(f"{out / f}: the copy is not B1a's")
        u1a.save_npz(out / STRUCT, src=ws, dst=wd, rel=np.zeros(ws.size, np.int16))
        b = json.loads(json.dumps(build))
        b["files"][STRUCT] = u1a.sha_file(out / STRUCT)
        b["u1a"] = {"declared_in": DECLARED, "variant": "w", "carve": "b1w",
                    "structural": "U2b's W2_S (undirected, both directions) among the setting's nodes",
                    "b1a_structural_sha256": build["files"][STRUCT], "edges": int(ws.size),
                    "script_sha256": u1a.sha_src(__file__), "utc": u1a.utc()}
        b["b1f"] = {"declared_in": DECLARED, "graph_file_sha256": s["graph_file_sha256"], "edges": int(ws.size)}
        u1a.write_json(out / "build.json", b)
        rec["settings"][name] = s
        log(f"{name}: {json.dumps({k: s[k] for k in ('passages', 'b1w', 'w2s_vs_b1_undirected', 'plain_rule_vs_b1_undirected_as_filed')})}")
    rec["script_sha256"] = u1a.sha_src(__file__)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["utc"] = u1a.utc()
    u1a.write_json(VAR_W / "recovery.json", rec)
    log(f"settings written; {VAR_W / 'recovery.json'}")
    return 0


def fork(variant, arm, split, src):
    if variant != "w":
        raise SystemExit("b1f fork: --variant w")
    saved = (dict(u1a.VAR), u1a.OUT)
    u1a.VAR["w"], u1a.OUT = VAR_W, OUT
    try:
        return u1a.fork(variant, arm, split, src)
    finally:
        u1a.VAR.clear()
        u1a.VAR.update(saved[0])
        u1a.OUT = saved[1]


def read_path(carve, arm, split, ds):
    if carve == "b1w":
        return OUT / "fits" / "w" / arm / split / "reads" / f"{ds}__b1w.npz"
    return u1a.read_path(carve, arm, split, ds)


def report():
    import b1b as BB                                 # B1b's per_question, boot_ci, call and the HippoRAG 2 row
    rec = {"declared_in": DECLARED, "candidate": BB.CAND, "hipporag2": BB.HIPPORAG2,
           "boot": {"resamples": BB.BOOT, "seed": BB.SEED}, "settings": {}}
    rv = VAR_W / "recovery.json"
    rec["graph"] = json.loads(rv.read_text(encoding="utf-8")) if rv.exists() else None
    missing = []
    for ds in SETTINGS:
        qs = json.loads((u1a.BENCH / ds / "queries.json").read_text(encoding="utf-8"))
        gc = [len(q["golds"]) for q in qs]
        s = {"questions": len(qs), "carves": {}, "paired": {}, "share_of_hyperlink_gain": {}}
        perq = {}
        for carve in CARVES:
            c = {}
            for arm in u1a.ARMS:
                for split in ("J5", u1a.HELD[ds]):
                    p = read_path(carve, arm, split, ds)
                    if not p.exists():
                        missing.append(f"{carve}/{arm}/{split}/{ds}")
                        continue
                    with np.load(p) as z:
                        m = BB.per_question(z, gc)
                    ci = BB.boot_ci(m["R@5"])
                    c[f"{arm}/{split}"] = {**{k: round(float(v.mean()), 4) for k, v in m.items()}, "R5_ci": ci,
                                           "vs_hipporag2": BB.call(ci, BB.HIPPORAG2[ds]),
                                           "read": "in-domain" if split == "J5" else "zero-shot", "file_sha256": u1a.sha_file(p)}
                    perq[(carve, arm, split)] = m["R@5"]
            s["carves"][carve] = c
        for a, b in (("b1w", "b1u"), ("b1w", "b1t"), ("b1", "b1w"), ("b1u", "b1t")):
            for arm in u1a.ARMS:
                for split in ("J5", u1a.HELD[ds]):
                    if (a, arm, split) in perq and (b, arm, split) in perq:
                        g = perq[(a, arm, split)] - perq[(b, arm, split)]
                        s["paired"][f"{a}-{b} {arm}/{split}"] = {"R@5": round(float(g.mean()), 4), "ci": BB.boot_ci(g)}
        for arm in u1a.ARMS:
            for split in ("J5", u1a.HELD[ds]):
                k = [perq.get((cv, arm, split)) for cv in ("b1", "b1t", "b1u", "b1w")]
                if all(x is not None for x in k) and float((k[0] - k[1]).mean()) > 0:
                    den = float((k[0] - k[1]).mean())
                    s["share_of_hyperlink_gain"][f"{arm}/{split}"] = {"b1u": round(float((k[2] - k[1]).mean()) / den, 3),
                                                                      "b1w": round(float((k[3] - k[1]).mean()) / den, 3)}
        s["best_published"] = {}
        for which, peer_only in (("any", False), ("peer_reviewed", True)):
            v, name = best_published(ds, peer_only)
            s["best_published"][which] = {"row": name, "R@5": v}
        s["calls_vs_best_published"] = {f"{cv} {key}": {w: BB.call(c["R5_ci"], b["R@5"] / 100) for w, b in s["best_published"].items()}
                                        for cv in CARVES for key, c in s["carves"][cv].items() if c["read"] == "in-domain"}
        rec["settings"][ds] = s
    rec["missing"] = missing
    rec["script_sha256"] = u1a.sha_src(__file__)
    rec["utc"] = u1a.utc()
    OUT.mkdir(parents=True, exist_ok=True)
    u1a.write_json(OUT / "report.json", rec)
    L = [f"# B1f report (declared in {DECLARED})", "", "R@5 x100, candidate p@swa; b1 = hyperlinks (musique: title links), b1t = text only, "
         "b1u = U1a's title rule, b1w = W2_S.", "", "| setting | read | b1 | b1t | b1u | b1w | w-u (CI) | HippoRAG 2 |", "|---|---|---:|---:|---:|---:|---|---:|"]
    for ds, s in rec["settings"].items():
        for key in sorted({k for c in s["carves"].values() for k in c}):
            cells = [f"{100 * s['carves'][cv][key]['R@5']:.1f}" if key in s["carves"][cv] else "--" for cv in CARVES]
            pw = s["paired"].get(f"b1w-b1u {key}")
            L.append(f"| {ds} | {key} | " + " | ".join(cells) + f" | {100 * pw['R@5']:+.1f} ({100 * pw['ci'][0]:+.1f}, {100 * pw['ci'][1]:+.1f}) | "
                     f"{100 * BB.HIPPORAG2[ds]:.1f} |" if pw else f" | -- | {100 * BB.HIPPORAG2[ds]:.1f} |")
    L += ["", "In-domain reads (J5) against the best published row of the setting, by the 95% interval of R@5 (ABOVE, AT or BELOW the row's value).", "",
          "| setting | best row (any) | best peer-reviewed row | read | b1 | b1u | b1w | calls (any/peer) for b1, b1u, b1w |", "|---|---|---|---|---:|---:|---:|---|"]
    for ds, s in rec["settings"].items():
        ba = s["best_published"]
        for key in sorted({k for c in s["carves"].values() for k, x in c.items() if x["read"] == "in-domain"}):
            cells = [f"{100 * s['carves'][cv][key]['R@5']:.1f}" if key in s["carves"][cv] else "--" for cv in ("b1", "b1u", "b1w")]
            calls = ", ".join("/".join(s["calls_vs_best_published"].get(f"{cv} {key}", {}).get(w, "--") for w in ("any", "peer_reviewed"))
                              for cv in ("b1", "b1u", "b1w"))
            L.append(f"| {ds} | {ba['any']['row']} {ba['any']['R@5']:.1f} | {ba['peer_reviewed']['row']} {ba['peer_reviewed']['R@5']:.1f} | {key} | "
                     + " | ".join(cells) + f" | {calls} |")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"report: {OUT / 'report.md'}; missing {len(missing)}")
    return 0


def selftest():
    # nodes 0,1 share package row 5 (a repeated passage); row 9 is outside the setting
    rows = [5, 7, 5, 8]
    indptr = np.zeros(11, np.int64)
    nb = {5: [7, 9], 7: [5, 8], 8: [7], 9: [5]}
    col = []
    for r in range(10):
        indptr[r + 1] = indptr[r] + len(nb.get(r, []))
        col += nb.get(r, [])
    s, t = cut(indptr, np.asarray(col, np.int32), rows)
    got = sorted(zip(s.tolist(), t.tolist()))
    want = sorted([(0, 1), (2, 1), (1, 0), (1, 2), (1, 3), (3, 1)])
    assert got == want, (got, want)
    assert all(a != b for a, b in got)
    e = cut(indptr, np.asarray(col, np.int32), [0, 1])
    assert e[0].size == 0
    log("selftest: the cut keeps in-setting neighbours, both directions, repeated rows share them, no self links")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    if argv[:1] == ["settings"]:
        return settings()
    if argv[:1] == ["fork"]:
        g = lambda k: argv[argv.index(k) + 1]  # noqa: E731
        return fork(g("--variant"), g("--arm"), g("--split"), g("--src"))
    if argv[:1] == ["report"]:
        return report()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
