"""U2b's corpus stages on Modal (docs/U2B_LINK_RULE.md, section 5). Run from the repository root on the laptop:

    modal run outputs/mp_unified/zu2b_modal.py --dataset D --stage dense
    modal run outputs/mp_unified/zu2b_modal.py --dataset D --stage reread
    modal run --detach outputs/mp_unified/zu2b_modal.py --dataset D --stage corpus --rule new|u1d --shards K
    modal run --detach outputs/mp_unified/zu2b_modal.py --dataset D --stage grade --carve s1sel --graphs g1,g2,...

"corpus" runs stat -> merge -> link -> build from one driver container; "grade" grades each group of graphs (groups
separated by ";") in its own container. The package copy and every U2b output live on one new volume, mounted at /v;
records are fetched with `modal volume get`. Every stage is safe to run again: one whose json record exists (it is
written last) returns that record, and a new array never takes the name of one an interrupted attempt left behind. The old dense store is read from a second volume, mounted at /old, only
to rebuild the package's own dense shards, each kept only when its np.save bytes hash to the package manifest. No
stage overwrites a path: each refuses one that exists. The rule's functions are zu2b.py's, run unchanged.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import modal

VOL = os.environ.get("U2B_VOLUME", "mpr-u2-links")
OLD_VOL = os.environ.get("U2B_OLD_VOLUME", "crag-data-volume")
app = modal.App("mpr-u2b")
image = (modal.Image.debian_slim(python_version="3.13")
         .pip_install("numpy==2.3.2", "pyyaml==6.0.2")
         .env({"OPENBLAS_NUM_THREADS": "8", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
         .add_local_file("outputs/mp_unified/u1b.py", "/root/mpu/u1b.py")
         .add_local_file("outputs/mp_unified/u1d.py", "/root/mpu/u1d.py")
         .add_local_file("outputs/mp_unified/zu2b.py", "/root/mpu/zu2b.py")
         .add_local_file("outputs/mp_unified/zu2b_fast.py", "/root/mpu/zu2b_fast.py")
         .add_local_file("src/mp_retrieval/m3b_pools.py", "/root/mpu/m3b_pools.py"))
vol = modal.Volume.from_name(VOL)
# One input per container, retried: every stage returns its finished record first, so a retry redoes nothing done.
STAGE = {"single_use_containers": True,
         "retries": modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=2.0)}
# Orchestrators wait on stages; preempting one would cancel every stage input in flight.
ORCH = {"nonpreemptible": True, "retries": modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=2.0)}
old = modal.Volume.from_name(OLD_VOL)
OLD = {"hotpotqa": "/old/data/canonical/hotpotqa/encodings/dense/docs",
       "2wiki": "/old/data/canonical/2wiki_universe/encodings/dense/docs"}
OLD_SS = 40000
PKG = Path("/v/pkg")
U2B = Path("/v/u2b")


def _mods():
    if "/root/mpu" not in sys.path:
        sys.path.insert(0, "/root/mpu")
    import zu2b
    import u1b
    import u1d
    import m3b_pools
    return zu2b, u1b, u1d, m3b_pools


def _rule(zu2b, name):
    return {"new": zu2b.NEW_RULE, "u1d": zu2b.U1D_RULE}[name]


def _variants(zu2b, name):
    return zu2b.VARIANTS if name == "new" else {"V2": ((0,), False)}


def _new_path(p):
    p = Path(p)
    if p.exists():
        raise SystemExit(f"{p} exists: U2b never overwrites")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _free(p):
    """p, or the first free one of p's .b, .c, ... names: an array an interrupted attempt left behind is never
    overwritten, and the stage's json record, written last, names the array it holds."""
    p = Path(p)
    for c in "abcdefghijklmnopqrstuvwxyz":
        q = p if c == "a" else p.with_name(f"{p.stem}.{c}{p.suffix}")
        if not q.exists():
            q.parent.mkdir(parents=True, exist_ok=True)
            return q
    raise SystemExit(f"{p}: no free name")


def _tmp(p):
    return p.with_name(f"{p.name}.{os.getpid()}.{time.time_ns()}.part")


def _save_npz(p, **arrs):
    """Saved under p or its first free name; returns (file name, sha256)."""
    import numpy as np
    p = _free(p)
    tmp = _tmp(p)
    with open(tmp, "wb") as f:
        np.savez(f, **arrs)
    os.replace(tmp, p)
    return p.name, _sha(p)


def _write_json(p, obj):
    p = _new_path(p)
    tmp = _tmp(p)
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def _done(p):
    """A stage's json record when an earlier attempt finished the stage (it is written last), else None."""
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _reload():
    import gc
    gc.collect()
    vol.reload()


def _peak_gb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM"):
                return round(int(line.split()[1]) / 1048576, 2)
    except OSError:
        pass
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)
    except (ImportError, OSError):
        return None


class _Hasher:
    def __init__(self):
        self.h, self.n = hashlib.sha256(), 0

    def write(self, b):
        self.h.update(b)
        self.n += len(b)


def _shard(n, i, k):
    return i * n // k, (i + 1) * n // k


def _codes(zu2b, u1b, u1d):
    return {"zu2b_sha256": u1b.sha_src(zu2b.__file__), "u1d_sha256": u1b.sha_src(u1d.__file__),
            "u1b_sha256": u1b.sha_src(u1b.__file__)}


# ── the package's dense doc vectors on the volume ───────────────────────────


@app.function(image=image, volumes={"/v": vol, "/old": old}, memory=6144, cpu=1.0, timeout=3600, **STAGE)
def dense_shards(d, shard_ids):
    """hotpotqa and 2wiki: each final shard rebuilt from the old store through the package's pointer index (final row
    i = old row row[i]) and the patch of the rows the old store holds differently, written only when its np.save bytes
    have the manifest's sha256 and length. The others: the uploaded shard files, hashed."""
    import numpy as np
    _reload()
    base = PKG / d / "embeddings" / "dense" / "docs"
    man = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    n, ss = int(man["n_rows"]), int(man["shard_size"])
    by = {int(s["shard"]): s for s in man["shards"]}
    if d in OLD:
        z = np.load(PKG / d / "pointer_index_dense.npz")
        row = z["row"].astype(np.int64)
        if not (z["src"] == 0).all() or row.size != n:
            raise SystemExit(f"{d}: the pointer index is not one old store's rows")
        pz = np.load(PKG / d / "dense_patch.npz")
        ppos, pvec = pz["pos"].astype(np.int64), pz["vec"]
    out = []
    for s in shard_ids:
        rec, lo = by[s], s * ss
        hi = min(lo + ss, n)
        p = base / rec["file"]
        wrote = False
        if d in OLD:
            r = row[lo:hi]
            arr = np.empty((hi - lo, 1536), np.float16)
            osh = r // OLD_SS
            for o in np.unique(osh).tolist():
                k = np.flatnonzero(osh == o)
                a = np.load(f"{OLD[d]}/shard_{o:05d}.npy", mmap_mode="r")
                idx = r[k] % OLD_SS
                srt = np.argsort(idx, kind="stable")
                arr[k[srt]] = np.asarray(a[idx[srt]])
                del a
            m = (ppos >= lo) & (ppos < hi)
            arr[ppos[m] - lo] = pvec[m]
            h = _Hasher()
            np.save(h, arr)
            sha, nb = h.h.hexdigest(), h.n
            if sha == rec["sha256"] and nb == int(rec["bytes"]) and not p.exists():
                tmp = p.with_name(p.name + ".part")
                with open(tmp, "wb") as f:
                    np.save(f, arr)
                os.replace(tmp, p)
                wrote = True
        else:
            sha, nb = _sha(p), p.stat().st_size
        out.append({"shard": s, "ok": sha == rec["sha256"] and nb == int(rec["bytes"]), "sha256": sha, "bytes": nb,
                    "wrote": wrote})
    if any(x["wrote"] for x in out):
        vol.commit()
    return out


@app.function(image=image, volumes={"/v": vol}, memory=1024, cpu=0.25, timeout=600, **STAGE)
def dense_record(d, results, name="dense_verified.json"):
    _reload()
    base = PKG / d / "embeddings" / "dense" / "docs"
    man = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    want = sorted(int(s["shard"]) for s in man["shards"])
    got = sorted(x["shard"] for x in results if x["ok"])
    rec = {"dataset": d, "manifest_sha256": _sha(base / "manifest.json"), "n_rows": int(man["n_rows"]),
           "shards": sorted(results, key=lambda x: x["shard"]), "status": "PASS" if got == want else "FAIL"}
    _write_json(U2B / d / name, rec)
    vol.commit()
    return {k: rec[k] for k in ("dataset", "n_rows", "status")} | {"ok": len(got), "of": len(want)}


@app.function(image=image, volumes={"/v": vol}, memory=2048, cpu=1.0, timeout=3600, **STAGE)
def dense_reread(d, shard_ids):
    """Each committed shard file read back from the volume and hashed against the manifest."""
    _reload()
    base = PKG / d / "embeddings" / "dense" / "docs"
    man = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    by = {int(s["shard"]): s for s in man["shards"]}
    out = []
    for s in shard_ids:
        p = base / by[s]["file"]
        sha, nb = (_sha(p), p.stat().st_size) if p.exists() else (None, 0)
        out.append({"shard": s, "ok": sha == by[s]["sha256"] and nb == int(by[s]["bytes"]), "sha256": sha,
                    "bytes": nb})
    return out


@app.function(image=image, volumes={"/v": vol}, memory=1024, cpu=0.25, timeout=7200, **ORCH)
def dense_reread_all(d, per):
    _reload()
    man = json.loads((PKG / d / "embeddings" / "dense" / "docs" / "manifest.json").read_text(encoding="utf-8"))
    ids = sorted(int(s["shard"]) for s in man["shards"])
    res = [x for r in dense_reread.starmap([(d, ids[i:i + per]) for i in range(0, len(ids), per)]) for x in r]
    return dense_record.remote(d, res, "dense_reread.json")


def _dense_checked(d, n):
    """u1d.Dense over the volume's package copy, once dense_verified.json has passed every shard of this manifest."""
    zu2b, u1b, u1d, _ = _mods()
    base = PKG / d / "embeddings" / "dense" / "docs"
    msha = _sha(base / "manifest.json")
    for name in ("dense_verified.json", "dense_reread.json"):
        rec = json.loads((U2B / d / name).read_text(encoding="utf-8"))
        if rec["status"] != "PASS" or rec["manifest_sha256"] != msha or rec["n_rows"] != n:
            raise SystemExit(f"{d}: the dense shards are not verified for this manifest ({name})")
    return u1d.Dense(PKG, d, n)


def _dense_all(d, n):
    import numpy as np
    D = _dense_checked(d, n)
    arr = np.empty((n, 1536), np.float16)
    for s, rec in D.shards.items():
        lo = int(s) * 40000
        a = np.load(D.d / rec["file"])
        arr[lo:lo + a.shape[0]] = a
    return arr, D.man_sha


# ── the corpus stages ───────────────────────────────────────────────────────


@app.function(image=image, volumes={"/v": vol}, memory=(12288, 65536), cpu=1.0, timeout=10800, **STAGE)
def stat(d, rule_name, i, k):
    import numpy as np
    zu2b, u1b, u1d, _ = _mods()
    _reload()
    od = U2B / d / rule_name / "stat"
    if (done := _done(od / f"s{i}of{k}.json")) is not None:
        return done
    t0 = time.time()
    rule = _rule(zu2b, rule_name)
    man = json.loads((PKG / d / "DATASET.json").read_text(encoding="utf-8"))
    lo, hi = _shard(int(man["nodes"]["n"]), i, k)
    titles, texts, n, nodes_sha = u1b.read_nodes(PKG, d, lo, hi)
    S = zu2b.Surfaces(titles, rule)
    t1 = time.time()
    tl = titles[lo:hi]
    cap, low = zu2b.stat_pass(S, texts, tl, rule)
    desc = np.asarray([zu2b.is_descriptive(t, x) for t, x in zip(tl, texts)], bool)
    rec = {"dataset": d, "rule": rule_name, "lo": lo, "hi": hi, "n_nodes": n, "nodes_sha256": nodes_sha,
           "n_surfaces": len(S.surf), "n_low": S.n_low, **_codes(zu2b, u1b, u1d), "secs_setup": round(t1 - t0, 1),
           "secs": round(time.time() - t0, 1), "peak_gb": _peak_gb()}
    rec["npz"], rec["npz_sha256"] = _save_npz(od / f"s{i}of{k}.npz", cap=cap, low=low, desc=desc)
    _write_json(od / f"s{i}of{k}.json", rec)
    vol.commit()
    return rec


def _check_shards(od, k, what):
    recs = [json.loads((od / f"s{i}of{k}.json").read_text(encoding="utf-8")) for i in range(k)]
    n = recs[0]["n_nodes"]
    if [r["lo"] for r in recs] != [_shard(n, i, k)[0] for i in range(k)] or recs[-1]["hi"] != n:
        raise SystemExit(f"{od}: the {what} shards do not cover [0, {n})")
    keys = ("nodes_sha256", "zu2b_sha256", "u1d_sha256", "u1b_sha256", "n_low")
    if len({tuple(r[x] for x in keys) for r in recs}) != 1:
        raise SystemExit(f"{od}: the {what} shards ran on different inputs or code")
    for i, r in enumerate(recs):
        if _sha(od / r["npz"]) != r["npz_sha256"]:
            raise SystemExit(f"{od}/{r['npz']}: sha256 differs from its record")
    return recs


@app.function(image=image, volumes={"/v": vol}, memory=8192, cpu=1.0, timeout=3600, **STAGE)
def merge(d, rule_name, k):
    import numpy as np
    _reload()
    if (done := _done(U2B / d / rule_name / "capstat.json")) is not None:
        return done
    od = U2B / d / rule_name / "stat"
    recs = _check_shards(od, k, "stat")
    cap = low = None
    desc = []
    for i in range(k):
        z = np.load(od / recs[i]["npz"])
        cap = z["cap"].astype(np.int64) if cap is None else cap + z["cap"]
        low = z["low"].astype(np.int64) if low is None else low + z["low"]
        desc.append(z["desc"])
    desc = np.concatenate(desc)
    rec = {x: recs[0][x] for x in ("dataset", "rule", "n_nodes", "nodes_sha256", "n_surfaces", "n_low", "zu2b_sha256",
                                   "u1d_sha256", "u1b_sha256")}
    rec.update({"shards": k, "common_forms": int((low > cap).sum()), "forms_seen": int(((cap + low) > 0).sum()),
                "descriptive_nodes": int(desc.sum()), "stat_secs_max": max(r["secs"] for r in recs),
                "stat_peak_gb_max": max(r["peak_gb"] or 0 for r in recs)})
    rec["npz"], rec["npz_sha256"] = _save_npz(U2B / d / rule_name / "capstat.npz", cap=cap, low=low, desc=desc)
    _write_json(U2B / d / rule_name / "capstat.json", rec)
    vol.commit()
    return rec


@app.function(image=image, volumes={"/v": vol}, memory=(24576, 131072), cpu=1.0, timeout=21600, **STAGE)
def link(d, rule_name, i, k):
    """One link shard, through zu2b_fast.link_pass (zu2b.link_pass's outputs, checked equal: its selftest here, and
    outputs/u2b/check_fast_musique.json on real passages under both rules)."""
    import numpy as np
    zu2b, u1b, u1d, _ = _mods()
    import zu2b_fast
    zu2b_fast._selftest()
    _reload()
    od = U2B / d / rule_name / "link"
    if (done := _done(od / f"s{i}of{k}.json")) is not None:
        return done
    t0 = time.time()
    rule = _rule(zu2b, rule_name)
    variants = _variants(zu2b, rule_name)
    man = json.loads((PKG / d / "DATASET.json").read_text(encoding="utf-8"))
    lo, hi = _shard(int(man["nodes"]["n"]), i, k)
    titles, texts, n, nodes_sha = u1b.read_nodes(PKG, d, lo, hi)
    S = zu2b.Surfaces(titles, rule)
    cs = json.loads((U2B / d / rule_name / "capstat.json").read_text(encoding="utf-8"))
    cpath = U2B / d / rule_name / cs["npz"]
    if cs["n_low"] != S.n_low or cs["nodes_sha256"] != nodes_sha or _sha(cpath) != cs["npz_sha256"] or \
            cs["zu2b_sha256"] != u1b.sha_src(zu2b.__file__):
        raise SystemExit(f"{d}: capstat is not this code's or these nodes'")
    with np.load(cpath) as z:
        common = z["low"] > z["cap"]
    D = _dense_checked(d, n)
    t1 = time.time()
    edges, stats = zu2b_fast.link_pass(S, texts, titles[lo:hi], rule, common, D.get, variants, lo=lo)
    arrs = {}
    for v, (s_, d_) in edges.items():
        arrs[f"{v}_src"], arrs[f"{v}_dst"] = s_, d_
    rec = {"dataset": d, "rule": rule_name, "lo": lo, "hi": hi, "n_nodes": n, "nodes_sha256": nodes_sha,
           "n_low": S.n_low, **_codes(zu2b, u1b, u1d), "capstat_sha256": cs["npz_sha256"],
           "dense_manifest_sha256": D.man_sha, "link_pass": "zu2b_fast",
           "zu2b_fast_sha256": u1b.sha_src(zu2b_fast.__file__), "stats": stats,
           "edges": {v: int(e[0].size) for v, e in edges.items()},
           "secs_setup": round(t1 - t0, 1), "secs": round(time.time() - t0, 1), "peak_gb": _peak_gb()}
    rec["npz"], rec["npz_sha256"] = _save_npz(od / f"s{i}of{k}.npz", **arrs)
    _write_json(od / f"s{i}of{k}.json", rec)
    vol.commit()
    return rec


# The build makes every graph of a rule in one container, and a preempted one starts again from the first graph
# (2wiki's was preempted four times): it is non-preemptible.
@app.function(image=image, volumes={"/v": vol}, memory=(32768, 131072), cpu=2.0, timeout=10800, nonpreemptible=True,
              **STAGE)
def build(d, rule_name, k):
    """Each variant's graphs as undirected CSR: the mention links (M), damped (D); with the descriptive switch also with
    the sibling links (S) and both (DS)."""
    import numpy as np
    zu2b, u1b, u1d, _ = _mods()
    _reload()
    if (done := _done(U2B / d / "graphs" / f"build_{rule_name}.json")) is not None:
        return done
    t0 = time.time()
    rule = _rule(zu2b, rule_name)
    variants = _variants(zu2b, rule_name)
    od = U2B / d / rule_name / "link"
    recs = _check_shards(od, k, "link")
    n = recs[0]["n_nodes"]
    titles, _t, _n, nodes_sha = u1b.read_nodes(PKG, d, 0, 0)
    cs = json.loads((U2B / d / rule_name / "capstat.json").read_text(encoding="utf-8"))
    if nodes_sha != recs[0]["nodes_sha256"] or recs[0]["capstat_sha256"] != cs["npz_sha256"]:
        raise SystemExit(f"{d}: the link shards ran on other nodes or another capstat")
    with np.load(U2B / d / rule_name / cs["npz"]) as z:
        desc = z["desc"]
    g = {}
    group = np.asarray([g.setdefault(zu2b.norm_title(t, rule).strip(), len(g)) for t in titles], np.int64)
    del titles, g
    if rule.descriptive:
        sib_s, sib_d = zu2b.sibling_pairs(group, desc)
        gs = np.bincount(group[desc])
        sib = {"pairs": int(sib_s.size), "groups_with_siblings": int((gs > 1).sum()), "largest_group": int(gs.max())}
    else:
        sib = None
    out = {}
    for v in variants:
        zs = [np.load(od / r["npz"]) for r in recs]
        s_ = np.concatenate([z_[f"{v}_src"] for z_ in zs]).astype(np.int64)
        t_ = np.concatenate([z_[f"{v}_dst"] for z_ in zs]).astype(np.int64)
        del zs
        key = s_ * n + t_
        if np.unique(key).size != key.size or (s_ == t_).any():
            raise SystemExit(f"{d} {v}: a pair repeats or loops")
        del key
        ds_, dt_ = zu2b.damp(s_, t_, n)
        forms = {"M": (s_, t_), "D": (ds_, dt_)}
        if rule.descriptive:
            forms["S"] = (np.concatenate([s_, sib_s]), np.concatenate([t_, sib_d]))
            forms["DS"] = (np.concatenate([ds_, sib_s]), np.concatenate([dt_, sib_d]))
        for f, (a_, b_) in forms.items():
            ip, col = zu2b.undirected_csr(n, a_, b_)
            deg = np.diff(ip)
            name = f"{v}_{f}"
            rec = {"graph": name, "directed_edges": int(a_.size), "undirected_neighbours": int(col.size),
                   "mean_degree": round(float(deg.mean()), 2), "isolated_share": round(float((deg == 0).mean()), 4),
                   "max_degree": int(deg.max()) if deg.size else 0}
            rec["file"], rec["npz_sha256"] = _save_npz(U2B / d / "graphs" / f"{name}.npz", indptr=ip, col=col)
            out[name] = rec
            del ip, col, deg
        del forms, s_, t_, ds_, dt_
    summary = {"dataset": d, "rule": rule_name, "nodes_sha256": nodes_sha, **_codes(zu2b, u1b, u1d),
               "capstat_sha256": cs["npz_sha256"], "link_npz_sha256": [r["npz_sha256"] for r in recs],
               "link_secs_max": max(r["secs"] for r in recs), "link_peak_gb_max": max(r["peak_gb"] or 0 for r in recs),
               "link_stats": {v: {x: sum(r["stats"][v][x] for r in recs) for x in ("surface_links", "ambiguous")}
                              for v in variants},
               "siblings": sib, "descriptive_nodes": int(desc.sum()), "graphs": out, "secs": round(time.time() - t0, 1),
               "peak_gb": _peak_gb()}
    _write_json(U2B / d / "graphs" / f"build_{rule_name}.json", summary)
    vol.commit()
    return summary


@app.function(image=image, volumes={"/v": vol}, memory=2048, cpu=0.25, timeout=86400, **ORCH)
def corpus(d, rule_name, k, k_link=0):
    """stat -> merge -> link -> build, each stage after the last one's records are on the volume. The link stage has
    its own shard count (k_link, default k); the merged statistic does not depend on k."""
    t0 = time.time()
    kl = k_link or k
    st = list(stat.starmap([(d, rule_name, i, k) for i in range(k)]))
    mg = merge.remote(d, rule_name, k)
    lk = list(link.starmap([(d, rule_name, i, kl) for i in range(kl)]))
    bd = build.remote(d, rule_name, kl)
    return {"dataset": d, "rule": rule_name, "shards": k, "link_shards": kl, "common_forms": mg["common_forms"],
            "stat_secs_max": max(r["secs"] for r in st), "link_secs_max": max(r["secs"] for r in lk),
            "peak_gb": {"stat": max(r["peak_gb"] or 0 for r in st), "link": max(r["peak_gb"] or 0 for r in lk),
                        "build": bd["peak_gb"]},
            "graphs": {g: {x: e[x] for x in ("directed_edges", "mean_degree", "max_degree")}
                       for g, e in bd["graphs"].items()}, "secs": round(time.time() - t0, 1)}


# ── grading ─────────────────────────────────────────────────────────────────


@app.function(image=image, volumes={"/v": vol}, memory=(32768, 131072), cpu=2.0, timeout=43200, **STAGE)
def grade(d, carve, graphs):
    """Per graph: the frozen I over it (m3b_pools.expand_hops and the construction, unchanged) and the reach of B ∪ X(k)
    (zu2b.reach_curves, one set of scores per question for all the graphs)."""
    import numpy as np
    from types import SimpleNamespace
    zu2b, u1b, u1d, m3b_pools = _mods()
    _reload()
    tag = hashlib.sha1(",".join(graphs).encode()).hexdigest()[:8]
    if (done := _done(U2B / d / "grade" / f"{carve}_{tag}.json")) is not None:
        return done
    t0 = time.time()
    files = {}
    for rn in ("new", "u1d"):
        b = _done(U2B / d / "graphs" / f"build_{rn}.json")
        for g, e in (b or {"graphs": {}})["graphs"].items():
            files[g] = (U2B / d / "graphs" / e["file"], e["npz_sha256"])
    qp = U2B / d / f"questions_{carve}.npz"
    qrec = json.loads((U2B / d / "questions.json").read_text(encoding="utf-8"))
    if _sha(qp) != qrec["carves"][carve]["npz_sha256"]:
        raise SystemExit(f"{d}/{carve}: the questions file is not its record's")
    q = np.load(qp)
    base = zu2b.unragged(q["base_ptr"], q["base_col"])
    seeds = zu2b.unragged(q["seeds_ptr"], q["seeds_col"])
    golds = zu2b.unragged(q["gold_ptr"], q["gold_col"])
    qemb = q["qemb"]
    con = qrec["construction"]
    fams = [str(f) for f in q["fam_order"]]
    others = {f: SimpleNamespace(indptr=q[f"{f}_indptr"], col=q[f"{f}_col"]) for f in fams if f != "structural"}
    B = [np.union1d(b, s) for b, s in zip(base, seeds)]
    man = json.loads((PKG / d / "DATASET.json").read_text(encoding="utf-8"))
    arr, dsha = _dense_all(d, int(man["nodes"]["n"]))
    t1 = time.time()
    csr, res, same = {}, {}, {}
    first = {}
    for gname in graphs:
        if gname == "PKG":
            p, want = U2B / d / "pkg_structural.npz", qrec["pkg_structural_npz_sha256"]
        else:
            p, want = files[gname]
        sha = _sha(p)
        if sha != want:
            raise SystemExit(f"{d}: {p.name} is not its record's")
        if sha in first:                    # the same graph (e.g. S = M where no two passages share a title)
            same[gname] = first[sha]
            continue
        first[sha] = gname
        z = np.load(p)
        st = SimpleNamespace(indptr=z["indptr"].astype(np.int64), col=z["col"])
        csr[gname] = (st.indptr, st.col)
        ent = {"npz_sha256": sha, "I": None}
        if con["regime"] != "RETRIEVAL":
            fam_objs = [st if f == "structural" else others[f] for f in fams]
            a = rec_ = size = 0.0
            for j in range(len(golds)):
                sd = np.asarray(seeds[j], np.int64)
                exp = m3b_pools.expand_hops(sd, fam_objs, con["setting"])
                pool = np.union1d(np.union1d(np.asarray(base[j], np.int64), exp), sd)
                hit = np.isin(golds[j], pool)
                a += hit.all()
                rec_ += hit.mean()
                size += pool.size
            nq = len(golds)
            ent["I"] = {"ALL": round(a / nq, 4), "recall": round(rec_ / nq, 4), "pool": round(size / nq, 1)}
        res[gname] = ent
    t2 = time.time()
    curves = zu2b.reach_curves(csr, B, golds, qemb, lambda r: arr[np.asarray(r, np.int64)].astype(np.float32))
    for gname in csr:
        res[gname].update(curves[gname])
    for gname, g0 in same.items():
        res[gname] = dict(res[g0], same_as=g0)
    out = {"dataset": d, "carve": carve, "questions": len(golds), "dense_manifest_sha256": dsha,
           "questions_npz_sha256": qrec["carves"][carve]["npz_sha256"], **_codes(zu2b, u1b, u1d), "graphs": res,
           "secs_load": round(t1 - t0, 1), "secs_I": round(t2 - t1, 1), "secs_reach": round(time.time() - t2, 1),
           "peak_gb": _peak_gb()}
    _write_json(U2B / d / "grade" / f"{carve}_{tag}.json", out)
    vol.commit()
    return out


@app.local_entrypoint()
def main(dataset: str, stage: str, rule: str = "new", shards: int = 1, carve: str = "s1sel", graphs: str = "",
         per: int = 6, link_shards: int = 0):
    t0 = time.time()
    if stage == "dense":
        man_n = {"hotpotqa": 131, "2wiki": 150}.get(dataset)
        if man_n is None:
            res = dense_shards.remote(dataset, list(range(shards)))
        else:
            ids = list(range(man_n))
            res = [x for r in dense_shards.starmap([(dataset, ids[i:i + per]) for i in range(0, len(ids), per)])
                   for x in r]
        print(json.dumps(dense_record.remote(dataset, res)))
    elif stage == "reread":
        print(json.dumps(dense_reread_all.remote(dataset, per)))
    elif stage == "corpus":
        print(json.dumps(corpus.remote(dataset, rule, shards, link_shards)))
    elif stage == "grade":
        groups = [[g for g in grp.split(",") if g] for grp in graphs.split(";") if grp]
        for r in grade.starmap([(dataset, carve, grp) for grp in groups]):
            print(json.dumps({"secs": [r["secs_load"], r["secs_I"], r["secs_reach"]], "peak_gb": r["peak_gb"],
                              "graphs": {g: {"I": e["I"], "k500": e["curve"]["500"], "all": e["all_neighbours"]}
                                         for g, e in r["graphs"].items()}}))
    else:
        raise SystemExit(f"unknown stage {stage}")
    print(f"{stage} {dataset} {rule}: {time.time() - t0:.0f} s")
