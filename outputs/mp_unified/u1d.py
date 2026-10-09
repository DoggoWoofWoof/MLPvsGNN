"""U1d (docs/U1D_PRECISE_LINKS.md): one precise link rule for all six corpora, chosen on hyperlink recovery.

    python outputs/mp_unified/u1d.py selftest
    python outputs/mp_unified/u1d.py stat  --dataset D --shard i/k [--host]   -> outputs/u1d/<D>/stat/s<i>of<k>.npz
    python outputs/mp_unified/u1d.py statmerge --dataset D --shards k [--host] -> outputs/u1d/<D>/capstat.npz
    python outputs/mp_unified/u1d.py link  --dataset D --shard i/k [--host]   -> outputs/u1d/<D>/link/s<i>of<k>.npz
    python outputs/mp_unified/u1d.py score --dataset D --shards k [--host]    -> outputs/u1d/<D>/score.json
    python outputs/mp_unified/u1d.py choose                                   -> outputs/u1d/choice.json
    python outputs/mp_unified/u1d.py build --dataset D --shards k [--host]    -> outputs/u1d/<D>/graph_structural_u.npz
    python outputs/mp_unified/u1d.py report                                   -> outputs/u1d/report.json, .md

Surfaces of a title (its `title` field; a KB node's text): the title itself (kind 0); the title without a trailing
parenthetical and the text before its first comma (kind 1, alias); the last token of a multi-token title that starts
with a capital (kind 2, last). Surfaces shorter than 4 characters never match. A surface matches at an occurrence that is
word-bounded (u1b.bounded's test) and written as the surface is, except that its first letter may take either case.
The capitalization statistic of a lowercased surface: over the whole corpus, its word-bounded occurrences that do not
start a sentence, counted as capitalized (first letter upper) or lowercase (all lower). A surface whose lowercase count
exceeds its capitalized count is a common word and never links. A surface owned by more than AMBIG distinct titles never
links. A surface owned by several distinct titles links to the title whose frozen dense vector is closest to the source
passage's (the highest cosine over the title's nodes); `exact-first` variants take the exact titles alone when the
surface has one. An edge goes to every node of the chosen title (a title can span several passages).
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import u1b  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "u1d"
DATASETS = u1b.DATASETS
HYPERLINK = u1b.HYPERLINK
KB = u1b.KB
MIN_LEN = u1b.MIN_TITLE_LEN
AMBIG = 50                     # step 4e's linking guard: a name shared by more than 50 nodes is not linked
PAREN = re.compile(r"\s*\([^)]*\)\s*$")
SENT_END = set(".!?:;\"'")
# variant: (kinds of surface, exact-first); declared order = simpler first (ties go to the earlier)
VARIANTS = {"V2": ((0,), False), "V5e": ((0, 1), True), "V5": ((0, 1), False),
            "V6e": ((0, 1, 2), True), "V6": ((0, 1, 2), False)}
CHUNK = 2000
EMB_SHARD = 40000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def save_npz(p, **arrs):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, p)
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ── surfaces ────────────────────────────────────────────────────────────────


class Surfaces:
    """Every surface (as written), its owners (node, kind), its lowercase-group id, and the title groups."""

    def __init__(self, titles):
        own = {}
        for j, t in enumerate(titles):
            t = t.strip()
            forms = [(t, 0)]
            a = PAREN.sub("", t).strip()
            if a != t:
                forms.append((a, 1))
            if "," in t:
                b = t.split(",")[0].strip()
                if b != t:
                    forms.append((b, 1))
            toks = PAREN.sub("", t).split()
            if len(toks) >= 2 and toks[-1][:1].isupper():
                forms.append((toks[-1], 2))
            seen = set()
            for sf, kd in forms:
                if len(sf) < MIN_LEN or (sf, kd) in seen:
                    continue
                seen.add((sf, kd))
                own.setdefault(sf, []).append((j, kd))
        self.surf = sorted(own)                          # deterministic order
        ptr = [0]
        nodes, kinds = [], []
        for sf in self.surf:
            for j, kd in own[sf]:
                nodes.append(j)
                kinds.append(kd)
            ptr.append(len(nodes))
        del own
        self.ptr = np.asarray(ptr, np.int64)
        self.nodes = np.asarray(nodes, np.int64)
        self.kinds = np.asarray(kinds, np.int8)
        lows = {}
        self.lowid = np.asarray([lows.setdefault(s.lower(), len(lows)) for s in self.surf], np.int64)
        self.n_low = len(lows)
        # title group of each node: nodes with the same stripped title
        g = {}
        self.group = np.asarray([g.setdefault(t.strip(), len(g)) for t in titles], np.int64)
        order = np.argsort(self.group, kind="stable")
        self.g_nodes = order
        self.g_ptr = np.searchsorted(self.group[order], np.arange(len(g) + 1))
        self.table = u1b.TitleTable(self.surf)

    def owners(self, k):
        a, b = self.ptr[k], self.ptr[k + 1]
        return self.nodes[a:b], self.kinds[a:b]

    def group_nodes(self, gid):
        return self.g_nodes[self.g_ptr[gid]:self.g_ptr[gid + 1]]


def occurrences(tl, low):
    n, m = len(low), len(tl)
    a, b = u1b._isw(low[0]), u1b._isw(low[-1])
    p = tl.find(low)
    while p >= 0:
        left = u1b._isw(tl[p - 1]) if p > 0 else False
        right = u1b._isw(tl[p + n]) if p + n < m else False
        if left != a and right != b:
            yield p
        p = tl.find(low, p + 1)


def sent_start(text, p):
    q = text[:p].rstrip()
    return q == "" or q[-1] in SENT_END


def scan(S, text):
    """[(surface k, case_ok)] for the surfaces bounded in text, and {lowid: [cap, lower]} mid-sentence counts."""
    tl = text.lower()
    same = len(tl) == len(text)          # lowercasing kept positions (else: case is not checked, counted nowhere)
    out, stat = [], {}
    done_low = set()
    for k in S.table.match(-1, text):
        sf = S.surf[k]
        low = S.table.low[k]
        lid = int(S.lowid[k])
        ok = False
        count = lid not in done_low
        done_low.add(lid)
        for p in occurrences(tl, low):
            if not same:
                ok = True
                break
            sl = text[p:p + len(low)]
            if sl[1:] == sf[1:] and sl[0].lower() == sf[0].lower():
                ok = True
            if count and not sent_start(text, p):
                c = stat.setdefault(lid, [0, 0])
                if sl[0].isupper():
                    c[0] += 1
                elif sl == low:
                    c[1] += 1
        out.append((k, ok))
    return out, stat


# ── dense vectors ───────────────────────────────────────────────────────────


class Dense:
    def __init__(self, served, name, n):
        self.d = served / name / "embeddings" / "dense" / "docs"
        man = json.loads((self.d / "manifest.json").read_text(encoding="utf-8"))
        if int(man["n_rows"]) != n or int(man["shard_size"]) != EMB_SHARD:
            raise SystemExit(f"{name}: dense manifest rows {man['n_rows']} / shard {man['shard_size']}")
        self.man_sha = hashlib.sha256((self.d / "manifest.json").read_bytes()).hexdigest()
        self.shards = {s["shard"]: s for s in man["shards"]}
        self.mm = {}

    def get(self, rows):
        rows = np.asarray(rows, np.int64)
        out = np.empty((rows.size, 1536), np.float32)
        sh = rows // EMB_SHARD
        for s in np.unique(sh):
            if s not in self.mm:
                p = self.d / self.shards[int(s)]["file"]
                if p.stat().st_size != int(self.shards[int(s)]["bytes"]):
                    raise SystemExit(f"{p}: size differs from the manifest")
                self.mm[s] = np.load(p, mmap_mode="r")
            k = sh == s
            r = rows[k] % EMB_SHARD
            o = np.argsort(r, kind="stable")
            v = np.asarray(self.mm[s][r[o]], np.float32)
            tmp = np.empty_like(v)
            tmp[o] = v
            out[k] = tmp
        return out


# ── stat ────────────────────────────────────────────────────────────────────


def shard_range(n, spec):
    i, k = u1b.parse_shard(spec)
    return i * n // k, (i + 1) * n // k, f"s{i}of{k}"


def open_inputs(a, lo_hi=None):
    served, freeze = u1b.package_root(a.host)
    man = json.loads((served / a.dataset / "DATASET.json").read_text(encoding="utf-8"))
    n = int(man["nodes"]["n"])
    lo, hi, tag = shard_range(n, a.shard) if lo_hi is None else lo_hi
    titles, texts, n, nodes_sha = u1b.read_nodes(served, a.dataset, lo, hi)
    return served, freeze, titles, texts, n, nodes_sha, lo, hi, tag


def base_rec(a, freeze, n, nodes_sha, S):
    return {"dataset": a.dataset, "n_nodes": n, "nodes_sha256": nodes_sha, "freeze_RECORD_SHA256": freeze,
            "u1d_sha256": u1b.sha_src(__file__), "u1b_sha256": u1b.sha_src(u1b.__file__),
            "n_surfaces": len(S.surf), "n_low": S.n_low, "host": bool(a.host)}


def cmd_stat(a):
    t0 = time.time()
    served, freeze, titles, texts, n, nodes_sha, lo, hi, tag = open_inputs(a)
    S = Surfaces(titles)
    del titles
    log(f"{a.dataset} {tag}: {len(S.surf)} surfaces, {S.n_low} lowercase forms ({time.time() - t0:.0f} s)")
    cap = np.zeros(S.n_low, np.int64)
    low = np.zeros(S.n_low, np.int64)
    for off, text in enumerate(texts):
        _hits, st = scan(S, text)
        for lid, (c, l_) in st.items():
            cap[lid] += c
            low[lid] += l_
        if (off + 1) % 200000 == 0:
            log(f"  {off + 1}/{hi - lo} ({time.time() - t0:.0f} s)")
    d = OUT / a.dataset / "stat"
    rec = base_rec(a, freeze, n, nodes_sha, S)
    rec.update({"lo": lo, "hi": hi, "secs": round(time.time() - t0, 1)})
    rec["npz_sha256"] = save_npz(d / f"{tag}.npz", cap=cap.astype(np.int32), low=low.astype(np.int32))
    write_json(d / f"{tag}.json", rec)
    log(f"{a.dataset} {tag}: done ({time.time() - t0:.0f} s)")
    return 0


def check_shards(d, k, n, what):
    recs = [json.loads((d / f"s{i}of{k}.json").read_text(encoding="utf-8")) for i in range(k)]
    if [r["lo"] for r in recs] != [i * n // k for i in range(k)] or recs[-1]["hi"] != n:
        raise SystemExit(f"{d}: the {what} shards do not cover [0, {n})")
    if len({(r["nodes_sha256"], r["u1d_sha256"], r["freeze_RECORD_SHA256"]) for r in recs}) != 1:
        raise SystemExit(f"{d}: the {what} shards ran on different inputs or code")
    for i, r in enumerate(recs):
        if hashlib.sha256((d / f"s{i}of{k}.npz").read_bytes()).hexdigest() != r["npz_sha256"]:
            raise SystemExit(f"{d}/s{i}of{k}.npz: sha256 differs from its record")
    return recs


def cmd_statmerge(a):
    d = OUT / a.dataset / "stat"
    r0 = json.loads((d / f"s0of{a.shards}.json").read_text(encoding="utf-8"))
    recs = check_shards(d, a.shards, r0["n_nodes"], "stat")
    cap = low = None
    for i in range(a.shards):
        z = np.load(d / f"s{i}of{a.shards}.npz")
        cap = z["cap"].astype(np.int64) if cap is None else cap + z["cap"]
        low = z["low"].astype(np.int64) if low is None else low + z["low"]
    common = low > cap
    rec = {k: recs[0][k] for k in ("dataset", "n_nodes", "nodes_sha256", "freeze_RECORD_SHA256", "u1d_sha256",
                                   "n_surfaces", "n_low")}
    rec.update({"shards": a.shards, "common_forms": int(common.sum()), "forms_seen": int(((cap + low) > 0).sum())})
    rec["npz_sha256"] = save_npz(OUT / a.dataset / "capstat.npz", cap=cap.astype(np.int64), low=low.astype(np.int64))
    write_json(OUT / a.dataset / "capstat.json", rec)
    log(json.dumps(rec))
    return 0


# ── link ────────────────────────────────────────────────────────────────────


def choose_targets(S, i, hits, common, need):
    """Per variant, the chosen title group of each case-matched, non-common surface; ambiguous ones go to `need`."""
    res = {v: [] for v in VARIANTS}
    for k, ok in hits:
        if not ok or common[S.lowid[k]]:
            continue
        nodes, kinds = S.owners(k)
        keep = nodes != i
        nodes, kinds = nodes[keep], kinds[keep]
        if nodes.size == 0:
            continue
        for v, (vk, exact_first) in VARIANTS.items():
            m = np.isin(kinds, vk)
            if exact_first and (kinds[m] == 0).any():
                m &= kinds == 0
            if not m.any():
                continue
            gs = np.unique(S.group[nodes[m]])
            if gs.size > AMBIG:
                continue
            if gs.size == 1:
                res[v].append(int(gs[0]))
            else:
                need.append((i, v, tuple(gs.tolist())))
    return res


# Code versions whose stat stage is this file's, byte for byte (only cmd_link's cache changed since): their capstat
# merges are accepted by link. 836a25e0... is 94acfb9's u1d.py.
STAT_SAME = ("836a25e0d22dd833c858d617fdf29a141b6b14a202d19a0af0d3a99a5ce3c5c0",)


def cmd_link(a):
    t0 = time.time()
    served, freeze, titles, texts, n, nodes_sha, lo, hi, tag = open_inputs(a)
    S = Surfaces(titles)
    del titles
    cs = json.loads((OUT / a.dataset / "capstat.json").read_text(encoding="utf-8"))
    if cs["n_low"] != S.n_low or cs["nodes_sha256"] != nodes_sha or cs["u1d_sha256"] not in (u1b.sha_src(__file__),) + STAT_SAME:
        raise SystemExit(f"{a.dataset}: capstat.json is not this code's or these nodes'")
    z = np.load(OUT / a.dataset / "capstat.npz")
    if hashlib.sha256((OUT / a.dataset / "capstat.npz").read_bytes()).hexdigest() != cs["npz_sha256"]:
        raise SystemExit("capstat.npz: sha256 differs from its record")
    common = z["low"] > z["cap"]
    D = Dense(served, a.dataset, n)
    log(f"{a.dataset} {tag}: {len(S.surf)} surfaces, {int(common.sum())} common forms ({time.time() - t0:.0f} s)")
    edges = {v: ([], []) for v in VARIANTS}
    stats = {v: {"surface_links": 0, "ambiguous": 0} for v in VARIANTS}
    gcache = {}

    def group_vecs(g):
        v = gcache.get(g)
        if v is None:
            if len(gcache) >= 200000:          # bounded cache (fix 10 Oct 01:50: it was cleared after the insert)
                gcache.clear()
            v = gcache[g] = D.get(S.group_nodes(g))
        return v

    for c0 in range(0, len(texts), CHUNK):
        chosen = {}
        need = []
        for off in range(c0, min(c0 + CHUNK, len(texts))):
            i = lo + off
            hits, _st = scan(S, texts[off])
            chosen[i] = choose_targets(S, i, hits, common, need)
        if need:
            src = np.asarray(sorted({x[0] for x in need}), np.int64)
            sv = dict(zip(src.tolist(), D.get(src)))
            for i, v, gs in need:
                best, bg = -9.0, -1
                for g in gs:
                    s = float((group_vecs(g) @ sv[i]).max())
                    if s > best:
                        best, bg = s, g
                chosen[i][v].append(bg)
                stats[v]["ambiguous"] += 1
        for i, per in chosen.items():
            for v, gl in per.items():
                stats[v]["surface_links"] += len(gl)
                tg = set()
                for g in gl:
                    tg.update(S.group_nodes(g).tolist())
                tg.discard(i)
                for j in sorted(tg):
                    edges[v][0].append(i)
                    edges[v][1].append(j)
        if (c0 // CHUNK) % 50 == 0:
            log(f"  {min(c0 + CHUNK, len(texts))}/{hi - lo}: " +
                " ".join(f"{v} {len(e[0])}" for v, e in edges.items()) + f" ({time.time() - t0:.0f} s)")
    arrs = {}
    for v, (s_, d_) in edges.items():
        arrs[f"{v}_src"] = np.asarray(s_, np.int32)
        arrs[f"{v}_dst"] = np.asarray(d_, np.int32)
    d = OUT / a.dataset / "link"
    rec = base_rec(a, freeze, n, nodes_sha, S)
    rec.update({"lo": lo, "hi": hi, "secs": round(time.time() - t0, 1), "stats": stats,
                "edges": {v: len(e[0]) for v, e in edges.items()}, "capstat_sha256": cs["npz_sha256"],
                "dense_manifest_sha256": D.man_sha})
    rec["npz_sha256"] = save_npz(d / f"{tag}.npz", **arrs)
    write_json(d / f"{tag}.json", rec)
    log(f"{a.dataset} {tag}: done " + json.dumps(rec["edges"]) + f" ({time.time() - t0:.0f} s)")
    return 0


# ── score / choose / build / report ─────────────────────────────────────────


def load_variant(a, v):
    d = OUT / a.dataset / "link"
    r0 = json.loads((d / f"s0of{a.shards}.json").read_text(encoding="utf-8"))
    recs = check_shards(d, a.shards, r0["n_nodes"], "link")
    s_, d_ = [], []
    for i in range(a.shards):
        z = np.load(d / f"s{i}of{a.shards}.npz")
        s_.append(z[f"{v}_src"].astype(np.int64))
        d_.append(z[f"{v}_dst"].astype(np.int64))
    s, t = np.concatenate(s_), np.concatenate(d_)
    n = r0["n_nodes"]
    key = s * n + t
    if np.unique(key).size != key.size or (s == t).any():
        raise SystemExit(f"{a.dataset} {v}: a pair repeats or loops")
    return s, t, n, recs


def cmd_score(a):
    served, freeze = u1b.package_root(a.host)
    ps, pd_, _rel, fam = u1b.package_structural(served, a.dataset)
    out = {"dataset": a.dataset, "shards": a.shards, "package_structural": fam.get("edge_subtype"), "variants": {}}
    c3 = None
    b = ROOT / "outputs" / "u1b" / a.dataset / "build.json"
    if b.is_file():
        c3 = json.loads(b.read_text(encoding="utf-8"))
        out["V0_c3"] = {k: c3[k] for k in ("recovery", "mentions") if k in c3}
    for v in VARIANTS:
        s, t, n, recs = load_variant(a, v)
        ent = {"edges": int(s.size), **u1b.degree_stats(n, s, t)[0],
               "ambiguous": sum(r["stats"][v]["ambiguous"] for r in recs)}
        if a.dataset in HYPERLINK:
            ent["directed"] = u1b.pr(u1b.pair_keys(n, s, t), u1b.pair_keys(n, ps, pd_))
            ent["unordered"] = u1b.pr(u1b.pair_keys(n, s, t, True), u1b.pair_keys(n, ps, pd_, True))
            P, R = ent["directed"]["precision"], ent["directed"]["recall"]
            ent["directed"]["f1"] = round(2 * P * R / max(P + R, 1e-12), 4)
        if a.dataset in KB:
            ent["already_kb_pair"] = round(float(np.isin(u1b.pair_keys(n, s, t, True),
                                                         u1b.pair_keys(n, ps, pd_, True)).mean()), 4) if s.size else 0.0
        ind = np.bincount(t, minlength=n)
        top = np.argsort(-ind, kind="stable")[:20]
        ent["top_in"] = [[int(j), int(ind[j])] for j in top]
        out["variants"][v] = ent
        log(f"{a.dataset} {v}: {s.size} edges " + (json.dumps(ent.get("directed")) if "directed" in ent else ""))
    titles, _t, _n, _sha = u1b.read_nodes(served, a.dataset, 0, 0)
    for v in out["variants"].values():
        v["top_in"] = [[titles[j][:60], c] for j, c in v["top_in"]]
    write_json(OUT / a.dataset / "score.json", out)
    return 0


def cmd_choose(_a):
    sc = {d: json.loads((OUT / d / "score.json").read_text(encoding="utf-8")) for d in HYPERLINK}
    mean = {v: round(float(np.mean([sc[d]["variants"][v]["directed"]["f1"] for d in HYPERLINK])), 4) for v in VARIANTS}
    best = max(mean.values())
    chosen = next(v for v in VARIANTS if mean[v] >= best - 0.005)      # declared order: the simpler wins a near tie
    rec = {"rule": "highest mean directed F1 over 2wiki and hotpotqa; within 0.005 the earlier (simpler) variant",
           "mean_f1": mean, "chosen": chosen,
           "per_dataset": {d: {v: sc[d]["variants"][v]["directed"] for v in VARIANTS} for d in HYPERLINK}}
    write_json(OUT / "choice.json", rec)
    log(json.dumps(rec))
    return 0


def cmd_build(a):
    ch = json.loads((OUT / "choice.json").read_text(encoding="utf-8"))["chosen"]
    served, freeze = u1b.package_root(a.host)
    s, t, n, recs = load_variant(a, ch)
    ps, pd_, prel, fam = u1b.package_structural(served, a.dataset)
    if a.dataset in KB:
        vocab = json.loads((served / a.dataset / "graph" / "GRAPH_MANIFEST.json").read_text(encoding="utf-8"))
        nv = len(vocab["families"]["structural"].get("relation_vocabulary") or [])
        rel_m = max(nv, int(prel.max()) + 1 if prel.size else 0)
        S_ = np.concatenate([ps, s])
        D_ = np.concatenate([pd_, t])
        R_ = np.concatenate([prel.astype(np.int16), np.full(s.size, rel_m, np.int16)])
    else:
        S_, D_, R_ = s, t, np.zeros(s.size, np.int16)
        rel_m = 0
    o = np.lexsort((D_, S_))
    S_, D_, R_ = S_[o], D_[o], R_[o]
    sha = save_npz(OUT / a.dataset / "graph_structural_u.npz", src=S_.astype(np.int32), dst=D_.astype(np.int32), rel=R_)
    rec = {"dataset": a.dataset, "variant": ch, "n_nodes": n, "mention_edges": int(s.size),
           "kb_triples_kept": int(ps.size) if a.dataset in KB else 0, "mention_rel_id": rel_m,
           "structural_u": u1b.degree_stats(n, S_, D_)[0], "package_structural": u1b.degree_stats(n, ps, pd_)[0],
           "npz_sha256": sha, "u1d_sha256": recs[0]["u1d_sha256"], "nodes_sha256": recs[0]["nodes_sha256"],
           "freeze_RECORD_SHA256": freeze}
    write_json(OUT / a.dataset / "build.json", rec)
    log(json.dumps({k: rec[k] for k in ("dataset", "variant", "mention_edges")}))
    return 0


def cmd_report(_a):
    rows, md = {}, ["# U1d: precise links on all six", ""]
    ch = json.loads((OUT / "choice.json").read_text(encoding="utf-8")) if (OUT / "choice.json").is_file() else None
    if ch:
        md += [f"Chosen: **{ch['chosen']}** (mean directed F1 over 2wiki and hotpotqa: "
               + ", ".join(f"{v} {f}" for v, f in ch["mean_f1"].items()) + ")", ""]
    md += ["## Hyperlink recovery (directed)", "", "| dataset | variant | edges | P | R | F1 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for d in DATASETS:
        p = OUT / d / "score.json"
        if not p.is_file():
            continue
        sc = json.loads(p.read_text(encoding="utf-8"))
        rows[d] = sc
        if d in HYPERLINK:
            c3 = sc.get("V0_c3", {}).get("recovery", {}).get("directed")
            if c3:
                md.append(f"| {d} | V0 (U1b) | {c3['rule']:,} | {c3['precision']} | {c3['recall']} | "
                          f"{round(2 * c3['precision'] * c3['recall'] / (c3['precision'] + c3['recall']), 4)} |")
            for v, e in sc["variants"].items():
                md.append(f"| {d} | {v} | {e['edges']:,} | {e['directed']['precision']} | {e['directed']['recall']} | "
                          f"{e['directed']['f1']} |")
    md += ["", "## Graphs", "", "| dataset | variant | edges | isolated share | max in | top titles |",
           "| --- | --- | ---: | ---: | ---: | --- |"]
    for d, sc in rows.items():
        for v, e in sc["variants"].items():
            md.append(f"| {d} | {v} | {e['edges']:,} | {e['isolated_share']} | {e.get('max_in_degree', '')} | "
                      + "; ".join(f"{t} ({c:,})" for t, c in e["top_in"][:5]) + " |")
    write_json(OUT / "report.json", {"choice": ch, "scores": rows})
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


# ── self-test ───────────────────────────────────────────────────────────────


def cmd_selftest(_a):
    titles = ["Paris", "Paris (band)", "Springfield, Illinois", "Barack Obama", "That", "Love Story", "love",
              "John Smith", "Jane Smith", "United States", "United States"]
    S = Surfaces(titles)
    sid = {s: k for k, s in enumerate(S.surf)}
    assert ("Paris", 0) in [(S.surf[k], int(S.kinds[S.ptr[k] + q])) for k in [sid["Paris"]]
                            for q in range(S.ptr[k + 1] - S.ptr[k])]
    own_p = sorted(zip(*[x.tolist() for x in S.owners(sid["Paris"])]))
    assert own_p == [(0, 0), (1, 1)], own_p
    assert sorted(S.owners(sid["Springfield"])[0].tolist()) == [2]
    assert sorted(S.owners(sid["Smith"])[0].tolist()) == [7, 8]
    assert sorted(S.owners(sid["Obama"])[0].tolist()) == [3]
    assert S.group[9] == S.group[10] and sorted(S.group_nodes(int(S.group[9])).tolist()) == [9, 10]
    hits, st = scan(S, "He moved to Paris. That was in the united states, and so that he saw Obama.")
    got = {S.surf[k]: ok for k, ok in hits}
    assert got["Paris"] and got["That"] and not got["United States"] and got["Obama"], got
    lid_that = int(S.lowid[sid["That"]])
    assert st[lid_that] == [0, 1], st                     # "That" opens a sentence (not counted); "that" mid: lower
    lid_us = int(S.lowid[sid["United States"]])
    assert st[lid_us] == [0, 1], st
    common = np.zeros(S.n_low, bool)
    common[lid_that] = True
    need = []
    res = choose_targets(S, 99, hits, common, need)
    gp = int(S.group[0])
    assert gp in res["V2"] and int(S.group[4]) not in res["V2"]
    assert any(v == "V5" and len(gs) == 2 for _i, v, gs in need)                  # Paris vs Paris (band): ambiguous
    assert gp in res["V5e"]                                                        # exact-first: Paris
    log("selftest passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("stat", "link"):
        p = sub.add_parser(c)
        p.add_argument("--dataset", required=True, choices=DATASETS)
        p.add_argument("--shard", required=True)
        p.add_argument("--host", action="store_true")
    for c in ("statmerge", "score", "build"):
        p = sub.add_parser(c)
        p.add_argument("--dataset", required=True, choices=DATASETS)
        p.add_argument("--shards", type=int, required=True)
        p.add_argument("--host", action="store_true")
    sub.add_parser("choose")
    sub.add_parser("report")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    return {"stat": cmd_stat, "statmerge": cmd_statmerge, "link": cmd_link, "score": cmd_score, "choose": cmd_choose,
            "build": cmd_build, "report": cmd_report, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
