"""U1a (docs/U1A_B1_WITHOUT_HYPERLINKS.md): B1a's three settings with the structural family emptied (b1t) or replaced
by the title-mention rule (b1u), the rule against the hyperlinks, fit copies for the reads, and the report.

    python outputs/mp_unified/u1a.py settings                  (laptop) outputs/bench/hipporag2_{t,u}/<dataset>/,
                                                              outputs/bench/hipporag2_u/recovery.json
    python outputs/mp_unified/u1a.py fork --variant t --arm zsp --split J5 --src outputs/full_zsp/fits/J5
    python outputs/mp_unified/u1a.py report                   -> outputs/u1a/report.json, report.md
    python outputs/mp_unified/u1a.py --selftest

settings  every file of a B1a setting but graph_structural.npz is copied byte for byte (checked against B1a's
          build.json); the variant's build.json is B1a's with that one file's sha replaced and a "u1a" block. b1u's
          edges are c3_derived's title-mention rule (CRAG scratchpad/c3_derived.py) on the setting corpus's own titles
          and texts, in setting positions, untyped. musique's b1u graph must equal B1a's (the package's title-mention
          graph induced on the setting) as package-row pairs, outside passages whose text is whitespace-only different.
fork      copies a fit's folder (everything but its reads) to outputs/u1a/fits/<variant>/<arm>/<split>; the models must
          be B1b's copy's (fork.json), where one exists.
report    B1b's reads (carve b1, outputs/b1b/fits) and the copies' reads (b1t, b1u), per setting, arm and read, with
          paired per-question differences.
"""
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCH = ROOT / "outputs" / "bench" / "hipporag2"
RAW = BENCH / "raw"
VAR = {"t": ROOT / "outputs" / "bench" / "hipporag2_t", "u": ROOT / "outputs" / "bench" / "hipporag2_u"}
OUT = ROOT / "outputs" / "u1a"
B1B = ROOT / "outputs" / "b1b"
SETTINGS = ("musique", "2wiki", "hotpotqa")
FILES = ("first_stage.npz", "graph_structural.npz", "graph_ner.npz", "graph_knn.npz", "nodes.json", "queries.json")
STRUCT = "graph_structural.npz"
ARMS = ("zsp", "zrc")
SPLITS = ("J5", "L-musique", "L-2wiki", "L-hotpotqa")
HELD = {"musique": "L-musique", "2wiki": "L-2wiki", "hotpotqa": "L-hotpotqa"}
CARVES = ("b1", "b1t", "b1u")

# c3_derived's rule, unchanged
STOP = set("the a an of and or to in on at for with by from as is are was were be this that these those "
           "he she it they we you i his her its their our your".split())
WORD = re.compile(r"[A-Za-z0-9]+")
MIN_TITLE_LEN = 4


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def save_npz(p, **arrays):
    tmp = Path(str(p) + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, p)


# ── the rule ─────────────────────────────────────────────────────────────────


def title_mentions(docs, min_title_len=MIN_TITLE_LEN):
    """c3_derived.build's edges on docs [(title, text)], as (src, dst) positions: A -> B iff A's lowercased text
    contains B's stripped, lowercased title as a word-bounded phrase; titles under min_title_len never match; each title
    is indexed by its longest token that is not a stop word and is over 2 characters (else its longest token)."""
    tokidx = defaultdict(list)
    pat = {}
    for i, (title, _text) in enumerate(docs):
        tl = title.strip()
        if len(tl) < min_title_len:
            continue
        toks = [w.lower() for w in WORD.findall(tl)]
        cand = [w for w in toks if w not in STOP and len(w) > 2] or toks
        if not cand:
            continue
        tokidx[max(cand, key=len)].append(i)
        pat[i] = re.compile(r"\b" + re.escape(tl.lower()) + r"\b")
    src, dst = [], []
    for i, (_title, text) in enumerate(docs):
        tl = text.lower()
        emitted = set()
        for tok in set(WORD.findall(tl)):
            for j in tokidx.get(tok, ()):
                if j == i or j in emitted:
                    continue
                if pat[j].search(tl):
                    emitted.add(j)
        for j in sorted(emitted):
            src.append(i)
            dst.append(j)
    return np.asarray(src, dtype=np.int32), np.asarray(dst, dtype=np.int32)


def graph_stats(n, s, d):
    """B1a's stats for a directed graph read undirected (mean degree 2E/n), plus the out-degree maximum."""
    deg = np.bincount(np.concatenate((s, d)).astype(np.int64), minlength=n) if s.size else np.zeros(n, np.int64)
    out = np.bincount(s.astype(np.int64), minlength=n) if s.size else np.zeros(n, np.int64)
    return {"edges": int(s.size), "mean_degree": round(2 * s.size / n, 3), "isolated": int((deg == 0).sum()),
            "max_out_degree": int(out.max()) if n else 0}


def row_pairs(rows, s, d):
    """Edges as distinct package-row pairs, self-loops dropped (a repeated passage's edges land on its one row)."""
    a, b = rows[s.astype(np.int64)], rows[d.astype(np.int64)]
    keep = a != b
    return {(int(x), int(y)) for x, y in zip(a[keep], b[keep])}


def pr(found, truth):
    tp = len(found & truth)
    return {"rule_pairs": len(found), "hyperlink_pairs": len(truth), "both": tp,
            "precision": round(tp / len(found), 4) if found else None, "recall": round(tp / len(truth), 4) if truth else None}


def undirected(pairs):
    return {(min(a, b), max(a, b)) for a, b in pairs}


def settings():
    t0 = time.time()
    rec = {"declared_in": "docs/U1A_B1_WITHOUT_HYPERLINKS.md", "rule": {"source": "CRAG scratchpad/c3_derived.py",
           "min_title_len": MIN_TITLE_LEN, "directed": True, "relation": 0}, "settings": {}}
    for name in SETTINGS:
        d = BENCH / name
        build = json.loads((d / "build.json").read_text(encoding="utf-8"))
        bad = [f for f in FILES if sha_file(d / f) != build["files"].get(f)]
        if bad:
            raise SystemExit(f"{name}: {bad} differ from B1a's build.json")
        corpus_file = [f for f in build["source_files"] if f.endswith("_corpus.json")]
        if len(corpus_file) != 1:
            raise SystemExit(f"{name}: B1a's build names {list(build['source_files'])}")
        corpus = json.loads((RAW / corpus_file[0]).read_text(encoding="utf-8"))
        nodes = json.loads((d / "nodes.json").read_text(encoding="utf-8"))
        if len(corpus) != len(nodes) or any(c["title"] != n["title"] for c, n in zip(corpus, nodes)):
            raise SystemExit(f"{name}: the raw corpus is not B1a's setting")
        rows = np.asarray([n["package_row"] for n in nodes], dtype=np.int64)
        n = len(corpus)
        with np.load(d / STRUCT) as z:
            hs, hd = z["src"].astype(np.int32), z["dst"].astype(np.int32)
        us, ud = title_mentions([(c["title"], c["text"]) for c in corpus])
        s = {"passages": n, "b1": graph_stats(n, hs, hd), "b1u": graph_stats(n, us, ud),
             "b1_structural_is": "title mentions (the package's graph)" if name == "musique" else "the corpus's hyperlinks"}
        found, truth = row_pairs(rows, us, ud), row_pairs(rows, hs, hd)
        if name == "musique":
            ws = {int(r) for r, nd in zip(rows, nodes) if not nd["exact"]}
            off = {p for p in found ^ truth if p[0] not in ws}
            s["identity"] = {"rule_pairs": len(found), "b1_pairs": len(truth), "differing_pairs": len(found ^ truth),
                             "differing_outside_whitespace_only_sources": len(off), "whitespace_only_passages": len(ws)}
            if off:
                raise SystemExit(f"musique: b1u differs from B1a's title-mention graph on {len(off)} pairs from exact "
                                 f"passages ({sorted(off)[:5]})")
        else:
            s["recovery_directed"] = pr(found, truth)
            s["recovery_undirected"] = pr(undirected(found), undirected(truth))
        for v, (vs, vd) in (("t", (np.zeros(0, np.int32), np.zeros(0, np.int32))), ("u", (us, ud))):
            out = VAR[v] / name
            out.mkdir(parents=True, exist_ok=True)
            for f in FILES:      # the lists stay in B1a's folder (look_b1u reads them there)
                if f != STRUCT:
                    shutil.copyfile(d / f, out / f)
                    if sha_file(out / f) != build["files"][f]:
                        raise SystemExit(f"{out / f}: the copy is not B1a's")
            save_npz(out / STRUCT, src=vs, dst=vd, rel=np.zeros(vs.size, np.int16))
            b = json.loads(json.dumps(build))
            b["files"][STRUCT] = sha_file(out / STRUCT)
            b["u1a"] = {"declared_in": "docs/U1A_B1_WITHOUT_HYPERLINKS.md", "variant": v, "carve": f"b1{v}",
                        "structural": "empty" if v == "t" else "title mentions (c3_derived's rule) on the setting corpus",
                        "b1a_structural_sha256": build["files"][STRUCT], "edges": int(vs.size),
                        "script_sha256": sha_src(__file__), "utc": utc()}
            write_json(out / "build.json", b)
        rec["settings"][name] = s
        log(f"{name}: {json.dumps(s)}")
    rec["script_sha256"] = sha_src(__file__)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["utc"] = utc()
    write_json(VAR["u"] / "recovery.json", rec)
    log(f"settings written; {VAR['u'] / 'recovery.json'}")
    return 0


# ── fit copies ───────────────────────────────────────────────────────────────


def fork(variant, arm, split, src):
    src = Path(src)
    if variant not in VAR or arm not in ARMS or split not in SPLITS:
        raise SystemExit(f"fork: --variant one of {tuple(VAR)}, --arm one of {ARMS}, --split one of {SPLITS}")
    want = sha_file(src / "models.pt")
    b1b = B1B / "fits" / arm / split / "fork.json"
    if b1b.exists() and json.loads(b1b.read_text(encoding="utf-8"))["models_sha256"] != want:
        raise SystemExit(f"{src}: not the models B1b read")
    dst = OUT / "fits" / variant / arm / split
    if (dst / "models.pt").exists():
        if sha_file(dst / "models.pt") != want:
            raise SystemExit(f"{dst}: holds other models than {src}")
        log(f"{dst}: already a copy of {src}")
        return 0
    tmp = dst.parent / (dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp, ignore=lambda d, names: [n for n in names if Path(d) == src and n in ("reads", "read.json")])
    if sha_file(tmp / "models.pt") != want:
        raise SystemExit(f"{tmp}: the copy's models.pt is not the source's")
    os.replace(tmp, dst)
    rel = src.resolve().relative_to(ROOT.resolve()).as_posix() if src.resolve().is_relative_to(ROOT.resolve()) else src.name
    write_json(dst / "fork.json", {"declared_in": "docs/U1A_B1_WITHOUT_HYPERLINKS.md", "variant": variant, "arm": arm,
                                   "split": split, "source": rel, "models_sha256": want, "script_sha256": sha_src(__file__),
                                   "utc": utc()})
    log(f"{src} -> {dst} (models {want[:12]})")
    return 0


# ── report ───────────────────────────────────────────────────────────────────


def read_path(carve, arm, split, ds):
    if carve == "b1":
        return B1B / "fits" / arm / split / "reads" / f"{ds}__b1.npz"
    return OUT / "fits" / carve[-1] / arm / split / "reads" / f"{ds}__{carve}.npz"


def report():
    sys.path.insert(0, str(HERE))
    import b1b as BB  # noqa: E402  (B1b's per_question, boot_ci, call and published numbers)
    rec = {"declared_in": "docs/U1A_B1_WITHOUT_HYPERLINKS.md", "candidate": BB.CAND, "published": BB.HIPPORAG2,
           "boot": {"resamples": BB.BOOT, "seed": BB.SEED}, "settings": {}}
    rv = VAR["u"] / "recovery.json"
    rec["rule"] = json.loads(rv.read_text(encoding="utf-8")) if rv.exists() else None
    missing = []
    for ds in SETTINGS:
        qs = json.loads((BENCH / ds / "queries.json").read_text(encoding="utf-8"))
        gc = [len(q["golds"]) for q in qs]
        s = {"questions": len(qs), "hipporag2": BB.HIPPORAG2[ds], "carves": {}, "paired": {}}
        perq = {}
        for carve in CARVES:
            c = {"reads": {}}
            for arm in ARMS:
                for split in ("J5", HELD[ds]):
                    p = read_path(carve, arm, split, ds)
                    if not p.exists():
                        missing.append(f"{carve}/{arm}/{split}/{ds}")
                        continue
                    with np.load(p) as z:
                        m = BB.per_question(z, gc)
                        if "rrf" not in c:
                            r = BB.per_question(z, gc, "rrf")
                            c["rrf"] = {k: round(float(v.mean()), 4) for k, v in r.items()}
                    ci = BB.boot_ci(m["R@5"])
                    row = {k: round(float(v.mean()), 4) for k, v in m.items()}
                    row.update({"R5_ci": ci, "vs_hipporag2": BB.call(ci, BB.HIPPORAG2[ds]),
                                "read": "in-domain" if split == "J5" else "zero-shot", "file_sha256": sha_file(p)})
                    c["reads"][f"{arm}/{split}"] = row
                    perq[(carve, arm, split)] = m["R@5"]
            s["carves"][carve] = c
        for a, b in (("b1t", "b1"), ("b1u", "b1"), ("b1u", "b1t")):
            for arm in ARMS:
                for split in ("J5", HELD[ds]):
                    if (a, arm, split) in perq and (b, arm, split) in perq:
                        g = perq[(a, arm, split)] - perq[(b, arm, split)]
                        s["paired"][f"{a}-{b} {arm}/{split}"] = {"R@5": round(float(g.mean()), 4), "ci": BB.boot_ci(g)}
        rec["settings"][ds] = s
    rec["missing"] = missing
    rec["script_sha256"] = sha_src(__file__)
    rec["utc"] = utc()
    write_json(OUT / "report.json", rec)
    write_md(rec)
    log(f"report: {OUT / 'report.md'}; missing {len(missing)}")
    return 0


def write_md(rec):
    L = ["# U1a: B1 without the corpus hyperlinks (docs/U1A_B1_WITHOUT_HYPERLINKS.md)", "",
         f"R@5 x100 of the {rec['candidate']} candidate, 95% bootstrap intervals; b1 = B1b's graph (hyperlinks on 2wiki "
         "and hotpotqa), b1t = text only (ner + knn), b1u = text + the title-mention rule.", ""]
    rule = rec.get("rule")
    if rule:
        L += ["## The rule against the hyperlinks", "", "| setting | b1 edges | b1u edges | P (dir) | R (dir) | P (undir) | R (undir) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for ds, s in rule["settings"].items():
            if "recovery_directed" in s:
                a, b = s["recovery_directed"], s["recovery_undirected"]
                L.append(f"| {ds} | {s['b1']['edges']} | {s['b1u']['edges']} | {a['precision']} | {a['recall']} | "
                         f"{b['precision']} | {b['recall']} |")
            else:
                L.append(f"| {ds} | {s['b1']['edges']} | {s['b1u']['edges']} | identity: {s['identity']['differing_pairs']} "
                         "pairs differ | | | |")
        L.append("")
    L += ["## Reads", "", "| setting | read | b1 | b1t | b1u | HippoRAG 2 |", "| --- | --- | --- | --- | --- | ---: |"]
    for ds, s in rec["settings"].items():
        keys = sorted({k for c in s["carves"].values() for k in c["reads"]})
        for k in ["rrf"] + keys:
            cells = []
            for carve in CARVES:
                c = s["carves"][carve]
                if k == "rrf":
                    cells.append(f"{100 * c['rrf']['R@5']:.1f}" if "rrf" in c else "–")
                elif k in c["reads"]:
                    r = c["reads"][k]
                    cells.append(f"{100 * r['R@5']:.1f} [{100 * r['R5_ci'][0]:.1f}, {100 * r['R5_ci'][1]:.1f}] {r['vs_hipporag2']}")
                else:
                    cells.append("–")
            L.append(f"| {ds} | {k} | " + " | ".join(cells) + f" | {100 * s['hipporag2']:.1f} |")
    L += ["", "## Paired differences (R@5 x100)", "", "| setting | difference | mean | 95% interval |", "| --- | --- | ---: | --- |"]
    for ds, s in rec["settings"].items():
        for k, v in s["paired"].items():
            L.append(f"| {ds} | {k} | {100 * v['R@5']:+.1f} | [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}] |")
    if rec["missing"]:
        L += ["", f"Missing reads: {', '.join(rec['missing'])}"]
    p = OUT / "report.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text("\n".join(L) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def selftest():
    docs = [("Lionel Messi", "He played with Neymar at FC Barcelona."), ("Neymar", "Neymar joined lionel messi's club."),
            ("FC Barcelona", "A club. The Neymar transfer."), ("The", "the the"), ("Neymarx", "not neymar")]
    s, d = title_mentions(docs)
    got = set(zip(s.tolist(), d.tolist()))
    want = {(0, 1), (0, 2), (1, 0), (2, 1), (4, 1)}
    assert got == want, got
    assert graph_stats(5, s, d)["edges"] == 5
    assert pr({(1, 2), (2, 3)}, {(1, 2)}) == {"rule_pairs": 2, "hyperlink_pairs": 1, "both": 1, "precision": 0.5, "recall": 1.0}
    rows = np.array([10, 11, 11])
    assert row_pairs(rows, np.array([0, 1]), np.array([1, 2])) == {(10, 11)}
    print("u1a selftest ok")
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
