"""U1b (docs/U1B_UNIVERSAL_LINKS_SIX.md): CRAG scratchpad/c3_derived.py's title-mention rule on all six full corpora.

    python outputs/mp_unified/u1b.py selftest
    python outputs/mp_unified/u1b.py shard --dataset 2wiki --shard 3/16 [--host]          -> outputs/u1b/<ds>/shards/
    python outputs/mp_unified/u1b.py shard --dataset 2wiki --limit 20000                  -> outputs/u1b/<ds>/sample.json
    python outputs/mp_unified/u1b.py merge --dataset 2wiki --shards 16 [--host]           -> graph_structural_u.npz, build.json
    python outputs/mp_unified/u1b.py report                                               -> outputs/u1b/report.json, .md

The matcher gives c3_derived's edges exactly. A regex match puts the title's lowercased [A-Za-z0-9]+ runs into the
lowercased text as consecutive whole runs, so a lookup of the text's run n-grams finds a superset of c3_derived's
edges. Each candidate is kept iff c3_derived's own conditions hold: its index token is among the text's tokens, and
the title occurs bounded by \\b on both sides.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "u1b"
DATASETS = ("musique", "squad", "2wiki", "hotpotqa", "metaqa", "webqsp")
IDENTITY = ("musique", "squad")          # the package's structural family is c3_derived's output
HYPERLINK = ("2wiki", "hotpotqa")        # the package's structural family is the corpus hyperlinks (dropped)
KB = ("metaqa", "webqsp")                # the package's structural family is KB triples (kept)
MIN_TITLE_LEN = 4                        # c3_derived's
# c3_derived's stop list and token pattern, verbatim
STOP = set("the a an of and or to in on at for with by from as is are was were be this that these those "
           "he she it they we you i his her its their our your".split())
WORD = re.compile(r"[A-Za-z0-9]+")
CFG = ROOT / "configs" / "m3b_controlled_comparison.yaml"
MIRROR_CFG = ROOT / "configs" / "host_mirror_six.yaml"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the rule ────────────────────────────────────────────────────────────────


def c3_reference(docs, min_title_len=MIN_TITLE_LEN):
    """c3_derived.build's loop on docs [(title, text)], as a set of (src, dst) positions (the self-test's reference)."""
    tokidx = defaultdict(list)
    title_pat = {}
    for did, (title, _text) in enumerate(docs):
        tl = title.strip()
        if len(tl) < min_title_len:
            continue
        toks = [w.lower() for w in WORD.findall(tl)]
        cand = [w for w in toks if w not in STOP and len(w) > 2] or toks
        if not cand:
            continue
        key = max(cand, key=len)
        tokidx[key].append((tl.lower(), did))
        title_pat[did] = re.compile(r"\b" + re.escape(tl.lower()) + r"\b")
    out = set()
    for did, (_title, text) in enumerate(docs):
        tl = text.lower()
        toks = set(WORD.findall(tl))
        emitted = set()
        for tok in toks:
            for (_cand_l, cand_did) in tokidx.get(tok, ()):
                if cand_did == did or cand_did in emitted:
                    continue
                if title_pat[cand_did].search(tl):
                    emitted.add(cand_did)
                    out.add((did, cand_did))
    return out


def _isw(ch):
    """re's \\w on str: alphanumeric (str.isalnum) or the underscore."""
    return ch.isalnum() or ch == "_"


def bounded(tl, low):
    """re.search(r'\\b' + re.escape(low) + r'\\b', tl) is not None, without compiling a pattern per title."""
    n, m = len(low), len(tl)
    a, b = _isw(low[0]), _isw(low[-1])
    p = tl.find(low)
    while p >= 0:
        left = _isw(tl[p - 1]) if p > 0 else False
        right = _isw(tl[p + n]) if p + n < m else False
        if left != a and right != b:
            return True
        p = tl.find(low, p + 1)
    return False


class TitleTable:
    """Every title that c3_derived indexes: its index token, its lowercased form and its lowercased run tuple."""

    def __init__(self, titles, min_title_len=MIN_TITLE_LEN):
        self.key, self.low = {}, {}
        self.idx = {}                       # run tuple -> [positions]
        first = defaultdict(set)            # first run -> {tuple lengths}
        for j, title in enumerate(titles):
            tl = title.strip()
            if len(tl) < min_title_len:
                continue
            toks = [w.lower() for w in WORD.findall(tl)]
            cand = [w for w in toks if w not in STOP and len(w) > 2] or toks
            if not cand:
                continue
            low = tl.lower()
            runs = tuple(WORD.findall(low))
            if not runs:                    # lowering removed every run: no occurrence can hold its key... checked below
                runs = None
            self.key[j] = max(cand, key=len)
            self.low[j] = low
            if runs is None:
                self.idx.setdefault(None, []).append(j)
                continue
            self.idx.setdefault(runs, []).append(j)
            first[runs[0]].add(len(runs))
        self.first = {r: tuple(sorted(ls)) for r, ls in first.items()}
        self.n_indexed = len(self.key)
        if None in self.idx:
            raise SystemExit(f"{len(self.idx[None])} indexed titles have no run once lowercased; the matcher's "
                             f"superset argument does not cover them")

    def match(self, i, text):
        """c3_derived's targets of position i (sorted)."""
        tl = text.lower()
        runs = WORD.findall(tl)
        if not runs:
            return []
        nr = len(runs)
        cands = set()
        first, idx = self.first, self.idx
        for k, r in enumerate(runs):
            ls = first.get(r)
            if ls is None:
                continue
            for L in ls:
                if k + L > nr:
                    break
                js = idx.get(tuple(runs[k:k + L]))
                if js:
                    cands.update(js)
        if not cands:
            return []
        toks = set(runs)
        key, low = self.key, self.low
        return sorted(j for j in cands if j != i and key[j] in toks and bounded(tl, low[j]))


def mention_edges(docs):
    t = TitleTable([d[0] for d in docs])
    out = set()
    for i, (_title, text) in enumerate(docs):
        for j in t.match(i, text):
            out.add((i, j))
    return out


# ── inputs ──────────────────────────────────────────────────────────────────


def package_root(host):
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    root = Path(yaml.safe_load(MIRROR_CFG.read_text(encoding="utf-8"))["host"]["mirror_root"]) if host \
        else Path(cfg["substrate"]["package_root"])
    served = root / "data" / "final_canonical"
    freeze = json.loads((served / "CANONICAL_FREEZE.json").read_text(encoding="utf-8"))
    if freeze["RECORD_SHA256"] != cfg["substrate"]["freeze_RECORD_SHA256_expected"]:
        raise SystemExit(f"served freeze {freeze['RECORD_SHA256']} != declared; refusing to run")
    return served, freeze["RECORD_SHA256"]


def title_of(rec):
    """A node's title: its title field; a KB node has none, and its title is its text (the entity name)."""
    t = rec.get("title")
    return t if t is not None else rec["text"]


def read_nodes(served, name, lo=0, hi=None):
    """All titles, the texts of positions [lo, hi), and the file's sha256 checked against DATASET.json."""
    man = json.loads((served / name / "DATASET.json").read_text(encoding="utf-8"))
    want, n = man["nodes"]["sha256"], int(man["nodes"]["n"])
    hi = n if hi is None else min(hi, n)
    h = hashlib.sha256()
    titles, texts = [], []
    with open(served / name / "nodes.jsonl", "rb") as f:
        for i, raw in enumerate(f):
            h.update(raw)
            rec = json.loads(raw)
            titles.append(title_of(rec))
            if lo <= i < hi:
                texts.append(rec["text"])
    if len(titles) != n:
        raise SystemExit(f"{name}: nodes.jsonl has {len(titles)} lines, DATASET.json names {n}")
    if h.hexdigest() != want:
        raise SystemExit(f"{name}: nodes.jsonl sha256 {h.hexdigest()} != DATASET.json's {want}")
    return titles, texts, n, want


def package_structural(served, name):
    gm = json.loads((served / name / "graph" / "GRAPH_MANIFEST.json").read_text(encoding="utf-8"))
    fam = gm["families"]["structural"]
    p = served / name / "graph" / os.path.basename(fam["file"])
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    if sha != fam["npz_sha256"]:
        raise SystemExit(f"{name}: structural.npz sha256 {sha} != GRAPH_MANIFEST's {fam['npz_sha256']}")
    z = np.load(p)
    src, dst = z["src"].astype(np.int64), z["dst"].astype(np.int64)
    rel = z["rel"] if "rel" in z.files else np.zeros(src.size, np.int16)
    return src, dst, rel, fam


# ── shard / sample ──────────────────────────────────────────────────────────


def parse_shard(s):
    i, n = (int(x) for x in s.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"--shard {s}")
    return i, n


def cmd_shard(a):
    served, freeze = package_root(a.host)
    name = a.dataset
    t0 = time.time()
    man = json.loads((served / name / "DATASET.json").read_text(encoding="utf-8"))
    n = int(man["nodes"]["n"])
    if a.limit:
        lo, hi, tag = 0, min(a.limit, n), None
    else:
        i, k = parse_shard(a.shard)
        lo, hi, tag = i * n // k, (i + 1) * n // k, f"s{i}of{k}"
    titles, texts, n, nodes_sha = read_nodes(served, name, lo, hi)
    t1 = time.time()
    table = TitleTable(titles)
    del titles
    t2 = time.time()
    log(f"{name}: {n} nodes, {table.n_indexed} titles indexed; read {t1 - t0:.0f} s, table {t2 - t1:.0f} s")
    src, dst = [], []
    for off, text in enumerate(texts):
        i = lo + off
        for j in table.match(i, text):
            src.append(i)
            dst.append(j)
        if (off + 1) % 200000 == 0:
            log(f"  {off + 1}/{hi - lo} passages, {len(src)} edges, {time.time() - t2:.0f} s")
    t3 = time.time()
    rec = {"dataset": name, "lo": lo, "hi": hi, "n_nodes": n, "edges": len(src), "titles_indexed": table.n_indexed,
           "secs": {"read": round(t1 - t0, 1), "table": round(t2 - t1, 1), "match": round(t3 - t2, 1)},
           "nodes_sha256": nodes_sha, "freeze_RECORD_SHA256": freeze, "u1b_sha256": sha_src(__file__),
           "host": bool(a.host)}
    d = OUT / name
    if a.limit:
        per = len(src) / max(hi - lo, 1)
        rec.update({"kind": "cost sample (not a result)", "edges_per_passage": round(per, 3),
                    "est_edges": int(per * n), "est_bytes_npz": int(per * n * 10),
                    "match_s_per_1k": round(1000 * (t3 - t2) / max(hi - lo, 1), 3)})
        d.mkdir(parents=True, exist_ok=True)
        _write_json(d / "sample.json", rec)
        log(json.dumps(rec))
        return 0
    sd = d / "shards"
    sd.mkdir(parents=True, exist_ok=True)
    tmp = sd / f"{tag}.tmp.npz"
    np.savez(tmp, src=np.asarray(src, np.int32), dst=np.asarray(dst, np.int32))
    os.replace(tmp, sd / f"{tag}.npz")
    rec["npz_sha256"] = hashlib.sha256((sd / f"{tag}.npz").read_bytes()).hexdigest()
    _write_json(sd / f"{tag}.json", rec)
    log(f"{name} {tag}: {len(src)} edges over [{lo}, {hi}), {t3 - t0:.0f} s")
    return 0


def _write_json(p, obj):
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


# ── merge ───────────────────────────────────────────────────────────────────


def degree_stats(n, s, d):
    deg = np.bincount(np.concatenate((s, d)), minlength=n) if s.size else np.zeros(n, np.int64)
    outd = np.bincount(s, minlength=n) if s.size else np.zeros(n, np.int64)
    ind = np.bincount(d, minlength=n) if s.size else np.zeros(n, np.int64)
    return {"edges": int(s.size), "mean_degree": round(2 * s.size / n, 3), "isolated": int((deg == 0).sum()),
            "isolated_share": round(float((deg == 0).mean()), 4), "max_out_degree": int(outd.max()),
            "max_in_degree": int(ind.max())}, ind


def pair_keys(n, s, d, undirected=False):
    if undirected:
        s, d = np.minimum(s, d), np.maximum(s, d)
    return np.unique(s.astype(np.int64) * n + d.astype(np.int64))


def pr(u, h):
    both = np.intersect1d(u, h, assume_unique=True).size
    return {"rule": int(u.size), "reference": int(h.size), "both": int(both),
            "precision": round(both / max(u.size, 1), 4), "recall": round(both / max(h.size, 1), 4)}


def cmd_merge(a):
    served, freeze = package_root(a.host)
    name = a.dataset
    sd = OUT / name / "shards"
    recs = [json.loads((sd / f"s{i}of{a.shards}.json").read_text(encoding="utf-8")) for i in range(a.shards)]
    n = recs[0]["n_nodes"]
    if [r["lo"] for r in recs] != [i * n // a.shards for i in range(a.shards)] or recs[-1]["hi"] != n:
        raise SystemExit(f"{name}: the shards do not cover [0, {n}) in order")
    if len({(r["nodes_sha256"], r["u1b_sha256"], r["freeze_RECORD_SHA256"]) for r in recs}) != 1:
        raise SystemExit(f"{name}: the shards ran on different inputs or code")
    parts = []
    for i, r in enumerate(recs):
        p = sd / f"s{i}of{a.shards}.npz"
        if hashlib.sha256(p.read_bytes()).hexdigest() != r["npz_sha256"]:
            raise SystemExit(f"{p.name}: sha256 differs from its record")
        z = np.load(p)
        parts.append((z["src"], z["dst"]))
    ms = np.concatenate([p[0] for p in parts]).astype(np.int64)
    md = np.concatenate([p[1] for p in parts]).astype(np.int64)
    del parts
    if (ms == md).any():
        raise SystemExit(f"{name}: a self-loop")
    keys = ms * n + md
    if np.unique(keys).size != keys.size:
        raise SystemExit(f"{name}: a pair repeats")
    del keys
    ps, pd_, prel, fam = package_structural(served, name)
    rec = {"dataset": name, "n_nodes": n, "freeze_RECORD_SHA256": freeze, "nodes_sha256": recs[0]["nodes_sha256"],
           "u1b_sha256": recs[0]["u1b_sha256"], "shards": a.shards,
           "package_structural": {"subtype": fam.get("edge_subtype") or ("kb" if name in KB else None),
                                  "npz_sha256": fam["npz_sha256"]},
           "secs_shards": round(sum(sum(r["secs"].values()) for r in recs), 1)}
    rec["mentions"], ind = degree_stats(n, ms, md)
    rec["package"], _ = degree_stats(n, ps, pd_)
    titles = None
    if name in IDENTITY:
        u, h = pair_keys(n, ms, md), pair_keys(n, ps, pd_)
        only_u, only_h = np.setdiff1d(u, h, True), np.setdiff1d(h, u, True)
        rec["identity"] = {"equal": bool(only_u.size == 0 and only_h.size == 0), "only_rule": int(only_u.size),
                           "only_package": int(only_h.size),
                           "examples_only_rule": [[int(k // n), int(k % n)] for k in only_u[:20]],
                           "examples_only_package": [[int(k // n), int(k % n)] for k in only_h[:20]]}
        if not rec["identity"]["equal"]:
            _write_json(OUT / name / "build.json", rec)
            raise SystemExit(f"{name}: IDENTITY FAILS ({only_u.size} pairs only in the rule, {only_h.size} only in "
                             f"the package); U1b stops here")
    if name in HYPERLINK:
        rec["recovery"] = {"directed": pr(pair_keys(n, ms, md), pair_keys(n, ps, pd_)),
                           "unordered": pr(pair_keys(n, ms, md, True), pair_keys(n, ps, pd_, True))}
    if name in KB:
        mu, ku = pair_keys(n, ms, md, True), pair_keys(n, ps, pd_, True)
        both = np.intersect1d(mu, ku, assume_unique=True).size
        rec["kb"] = {"mention_pairs": int(mu.size), "kb_pairs": int(ku.size), "mention_pairs_in_kb": int(both),
                     "share": round(both / max(mu.size, 1), 4)}
        vocab = fam.get("relation_vocabulary") or []
        mrel = len(vocab)
        s = np.concatenate((ps, ms))
        d = np.concatenate((pd_, md))
        r = np.concatenate((prel.astype(np.int16), np.full(ms.size, mrel, np.int16)))
        rec["relations"] = {"kb": len(vocab), "title_mention": mrel}
    else:
        s, d, r = ms, md, np.zeros(ms.size, np.int16)
        rec["relations"] = {"title_mention": 0}
    o = np.lexsort((r, d, s))
    s, d, r = s[o].astype(np.int32), d[o].astype(np.int32), r[o]
    rec["structural_u"], _ = degree_stats(n, s.astype(np.int64), d.astype(np.int64))
    top = np.argsort(-ind, kind="stable")[:20]
    titles, _t, _n, _sha = read_nodes(served, name, 0, 0)
    rec["top_mentioned"] = [[int(j), titles[j][:80], int(ind[j])] for j in top if ind[j] > 0]
    out = OUT / name / "graph_structural_u.npz"
    tmp = out.with_name("graph_structural_u.tmp.npz")
    np.savez(tmp, src=s, dst=d, rel=r)
    os.replace(tmp, out)
    rec["graph_structural_u"] = {"file": f"outputs/u1b/{name}/graph_structural_u.npz",
                                 "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "bytes": out.stat().st_size}
    _write_json(OUT / name / "build.json", rec)
    log(json.dumps({k: v for k, v in rec.items() if k != "top_mentioned"}))
    return 0


# ── report ──────────────────────────────────────────────────────────────────


def cmd_report(_a):
    rows = {}
    for name in DATASETS:
        p = OUT / name / "build.json"
        if p.exists():
            rows[name] = json.loads(p.read_text(encoding="utf-8"))
    _write_json(OUT / "report.json", {"declared_in": "docs/U1B_UNIVERSAL_LINKS_SIX.md", "datasets": rows})
    L = ["# U1b: one link rule on all six full corpora", "", "## Hyperlink recovery", "",
         "| dataset | rule edges | hyperlinks | directed P | directed R | unordered P | unordered R |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in HYPERLINK:
        if name in rows and "recovery" in rows[name]:
            r = rows[name]["recovery"]
            L.append(f"| {name} | {r['directed']['rule']:,} | {r['directed']['reference']:,} | "
                     f"{r['directed']['precision']} | {r['directed']['recall']} | {r['unordered']['precision']} | "
                     f"{r['unordered']['recall']} |")
    L += ["", "## Graphs", "",
          "| dataset | nodes | package structural (edges / isolated) | mentions (edges / isolated / max in) | "
          "structural_U (edges / isolated) | check |", "| --- | ---: | --- | --- | --- | --- |"]
    for name, r in rows.items():
        chk = ("identity " + ("EQUAL" if r["identity"]["equal"] else "DIFFERS")) if "identity" in r else \
            (f"mention pairs already KB pairs {r['kb']['share']}" if "kb" in r else "hyperlinks dropped")
        L.append(f"| {name} | {r['n_nodes']:,} | {r['package']['edges']:,} / {r['package']['isolated_share']} | "
                 f"{r['mentions']['edges']:,} / {r['mentions']['isolated_share']} / {r['mentions']['max_in_degree']:,} | "
                 f"{r['structural_u']['edges']:,} / {r['structural_u']['isolated_share']} | {chk} |")
    L += ["", "## Most-mentioned titles", ""]
    for name, r in rows.items():
        L.append(f"- **{name}**: " + "; ".join(f"{t} ({k:,})" for _j, t, k in r.get("top_mentioned", [])[:10]))
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


# ── self-test ───────────────────────────────────────────────────────────────


def cmd_selftest(a):
    hard = [("Zürich", "He moved to zürich in 1990; Zürich's lake."), ("AC/DC", "The band ac/dc played. AC/DCX no."),
            ("'Allo 'Allo!", "watch 'allo 'allo! tonight and x'allo 'allo!y"), ("2008_Sichuan_earthquake",
            "the 2008_sichuan_earthquake and the 2008 sichuan earthquake"), ("İstanbul", "istanbul and i̇stanbul"),
            ("C++ (language)", "c++ (language) is used; c++ (language)x"), ("Mr. Smith", "mr. smith and mr.smith"),
            ("The Who", "the who sang; thewho"), ("Paris", "paris, parisian, _paris_, paris_"), ("1990", "in 1990."),
            ("Éclair", "an éclair and aéclair"), ("Smith", "mr. smith and smithson"), ("The The", "the the the band"),
            ("New York City", "new york city and new york  city"), ("A", "a"), ("Oslo", "oslo oslo")]
    rng = random.Random(20261010)
    alpha = list("abcdefghij éü_-'.()/ 0123") + ["İ", "ß", "ﬁ"]
    syn = []
    for _ in range(400):
        t = "".join(rng.choice(alpha) for _ in range(rng.randint(3, 9)))
        syn.append((t, " ".join("".join(rng.choice(alpha) for _ in range(rng.randint(1, 12))) for _ in range(30))))
    for k in range(400):    # texts that contain other titles at random offsets
        a_, b_ = syn[k], syn[rng.randrange(400)]
        syn[k] = (a_[0], a_[1][:rng.randint(0, 60)] + rng.choice(["", " ", "x", "_", "é"]) + b_[0].lower() +
                  rng.choice(["", " ", "x", "_", "."]) + a_[1][60:])
    hard = hard + [(f"Holder {k}", t) for k, (_t, t) in enumerate(hard)]   # each tricky title as another doc's target
    sets = {"hard": hard, "synthetic": syn}
    if a.dataset:
        served, _f = package_root(False)
        titles, texts, n, _sha = read_nodes(served, a.dataset, 0, a.limit)
        sets[a.dataset] = list(zip(titles[:a.limit], texts))
    ok = True
    for nm, docs in sets.items():
        ref, got = c3_reference(docs), mention_edges(docs)
        same = ref == got
        ok &= same
        log(f"{nm}: {len(docs)} docs, c3_derived {len(ref)} edges, matcher {len(got)}: {'EQUAL' if same else 'DIFFER'}"
            + ("" if same else f" only c3 {sorted(ref - got)[:5]} only matcher {sorted(got - ref)[:5]}"))
    for low, tl in [("ab", "xab"), ("ab", "ab"), ("(ab", "x(ab"), ("ab)", "ab)x"), ("ab", "_ab")]:
        assert bounded(tl, low) == (re.search(r"\b" + re.escape(low) + r"\b", tl) is not None), (low, tl)
    log("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--dataset", choices=DATASETS)
    s.add_argument("--limit", type=int, default=3000)
    s = sub.add_parser("shard")
    s.add_argument("--dataset", choices=DATASETS, required=True)
    s.add_argument("--shard", default="0/1")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--host", action="store_true")
    s = sub.add_parser("merge")
    s.add_argument("--dataset", choices=DATASETS, required=True)
    s.add_argument("--shards", type=int, required=True)
    s.add_argument("--host", action="store_true")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    return {"selftest": cmd_selftest, "shard": cmd_shard, "merge": cmd_merge, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
