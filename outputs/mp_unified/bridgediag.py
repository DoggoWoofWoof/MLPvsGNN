"""Diagnostic D1, the bridge (docs/DIAG_BRIDGE.md): does conditioning on the first hop reach the golds zrm misses?

zrm scores every pool row on its own. Where it finds part of a question's golds (most of musique's and 2wiki's lost
R@5), this asks whether the golds it misses are linked, in the question's own pool graph (the look's edges: structural,
NER and kNN, as built), to the golds it finds, and whether choosing rows next to its top pick would recover them. No
training: zrm's fitted weights are read on CPU through zrm's own read, on a copy of the fit's models.pt and screen.json
in a work folder (the sealed fit folders are never written), with lean_gpu.top_hit wrapped so the scores of p@swa
(and rrf's) are kept per row. The edges come from the look's chunks, row for row in the cache's order (ids, pool sizes
and gold flags are checked against the cache).

Per read (a fit and a carve), over questions with at least one gold:
  zrm      R@5 and FC@5 of zrm's top five (as filed: must agree with the fit's read.json within 0.002, or exit 1)
  C1       label-free: zrm's top three, then the two highest-scored neighbours of zrm's top-1 not yet chosen
           (then zrm's order fills up to five)
  C2       label-free: zrm's top three, the highest-scored neighbour of the top-1, then of the top-2 (then zrm's order)
  O1       oracle hop 1: as C1 with the anchor the highest-ranked gold in zrm's top five (zrm's top five when none)
  O2       reach bound: zrm's found golds plus every missed in-pool gold next to one, at most five
and, over questions zrm finds partly (at least one gold in its top five, at least one in-pool gold outside it):
  adj      the share of missed in-pool golds next to a found gold (any family; and per family 0 structural, 1 NER,
           2 kNN), and within two hops
  lift     adj over the same share among non-gold rows outside the top five
  nb_rank  a missed gold's rank, by zrm's score, among the found golds' neighbours outside the top five

    python outputs/mp_unified/bridgediag.py run --fit scr-zrm --fit-root outputs/screen/fits --out outputs/diag/bridge-scr-zrm \\
        --device cpu --threads 8 --host
    python outputs/mp_unified/bridgediag.py verdict --runs outputs/diag/bridge-scr-zrm.json,outputs/diag/bridge-scr-zrm-hp.json \\
        --out outputs/diag/bridge-zrm
    python outputs/mp_unified/bridgediag.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

ROOT = HERE.parents[1]
WORK = ROOT / "outputs" / "diag" / "bridge-work"
CAND = "p@swa"
TOL_FILED = 0.002
FAMS = (0, 1, 2)
# the declared rule (docs/DIAG_BRIDGE.md), fixed before any number
MULTI = ("musique", "2wiki")
ADJ_MIN, LIFT_MIN = 0.5, 2.0
ORACLE_MIN = 0.02
FREE_MIN, FREE_HARM = 0.0075, 0.0075


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── one question ─────────────────────────────────────────────────────────────


def order_of(s):
    """lean_gpu.top_hit's order: descending score, NaN last, ties by pool position, -0.0 ties +0.0."""
    s = np.asarray(s, np.float64)
    nan = np.isnan(s)
    key = np.where(nan, np.inf, -s) + 0.0
    return np.argsort(key, kind="stable")


def nbrs(n, sel, u, v):
    """Rows next to any row in sel (sel's own rows included when they are next to another)."""
    ins = np.zeros(n, bool)
    ins[sel] = True
    out = np.zeros(n, bool)
    out[v[ins[u]]] = True
    out[u[ins[v]]] = True
    return out


def pick(order, keep, extra_anchors, s, u, v, n, k=5):
    """keep (row ids, in order), then for each anchor its highest-scored neighbour not yet chosen (per anchor the count
    in extra_anchors), then zrm's order fills up to k."""
    chosen = list(keep)
    taken = np.zeros(n, bool)
    taken[chosen] = True
    for a, cnt in extra_anchors:
        nb = nbrs(n, [a], u, v) & ~taken
        idx = np.flatnonzero(nb)
        if idx.size:
            o = idx[order_of(s[idx])][:cnt]
            chosen += [int(x) for x in o]
            taken[o] = True
    for r in order:
        if len(chosen) >= k:
            break
        if not taken[r]:
            chosen.append(int(r))
            taken[r] = True
    return np.asarray(chosen[:k], np.int64)


def question(s, g, gt, u, v, fam):
    """The counts of one question (pool-local edges u, v with families fam)."""
    n = s.size
    o = order_of(s)
    top = o[:5]
    gold = np.flatnonzero(g)
    found = [int(r) for r in top if g[r]]
    in_top = np.zeros(n, bool)
    in_top[top] = True
    missed = np.flatnonzero(g & ~in_top)
    out = {"gt": int(gt), "zrm": len(found)}

    def hits(sel):
        return int(g[sel].sum())

    out["C1"] = hits(pick(o, o[:3], [(int(o[0]), 2)], s, u, v, n)) if n else 0
    out["C2"] = hits(pick(o, o[:3], [(int(o[0]), 1)] + ([(int(o[1]), 1)] if n > 1 else []), s, u, v, n)) if n else 0
    out["O1"] = hits(pick(o, o[:3], [(found[0], 2)], s, u, v, n)) if found else len(found)
    if found and missed.size:
        adj = nbrs(n, found, u, v)
        out["O2"] = min(5, len(found) + int(adj[missed].sum()), len(found) + missed.size)
        out["partial"] = 1
        out["missed"] = int(missed.size)
        out["missed_adj"] = int(adj[missed].sum())
        for f in FAMS:
            m = fam == f
            out[f"missed_adj{f}"] = int(nbrs(n, found, u[m], v[m])[missed].sum())
        two = adj.copy()
        two[found] = True
        two = nbrs(n, np.flatnonzero(two), u, v) | adj
        out["missed_adj2hop"] = int(two[missed].sum())
        neg = ~g & ~in_top
        out["neg"] = int(neg.sum())
        out["neg_adj"] = int((adj & neg).sum())
        cand = np.flatnonzero(adj & ~in_top)
        if cand.size:
            rk = np.empty(n, np.int64)
            rk[cand[order_of(s[cand])]] = np.arange(1, cand.size + 1)
            out["nb_ranks"] = [int(rk[m]) for m in missed if adj[m]]
            out["nb_size"] = int(cand.size)
    else:
        out["O2"] = len(found)
    out["pool_golds"] = int(gold.size)
    return out


def summarise(rows):
    """Means over questions with golds (R@5, FC@5 per rule) and the partly-found counts."""
    w = [r for r in rows if r["gt"] > 0]
    res = {"questions": len(w)}
    for k in ("zrm", "C1", "C2", "O1", "O2"):
        res[k] = {"R@5": float(np.mean([min(r[k], r["gt"]) / r["gt"] for r in w])) if w else None,
                  "FC@5": float(np.mean([r[k] >= r["gt"] for r in w])) if w else None}
    p = [r for r in w if r.get("partial")]
    res["partial_questions"] = len(p)
    mi = sum(r["missed"] for r in p)
    res["missed_in_pool"] = mi
    if mi:
        res["adj"] = sum(r["missed_adj"] for r in p) / mi
        for f in FAMS:
            res[f"adj_fam{f}"] = sum(r[f"missed_adj{f}"] for r in p) / mi
        res["adj_2hop"] = sum(r["missed_adj2hop"] for r in p) / mi
        ng = sum(r["neg"] for r in p)
        res["neg_adj"] = sum(r["neg_adj"] for r in p) / ng if ng else None
        res["lift"] = res["adj"] / res["neg_adj"] if res["neg_adj"] else None
        rk = [x for r in p for x in r.get("nb_ranks", [])]
        res["nb_rank_median"] = float(np.median(rk)) if rk else None
        res["nb_rank_le2"] = float(np.mean(np.asarray(rk) <= 2)) if rk else None
        sz = [r["nb_size"] for r in p if "nb_size" in r]
        res["nb_size_median"] = float(np.median(sz)) if sz else None
    for k in ("C1", "C2", "O1", "O2"):
        if res[k]["R@5"] is not None:
            res[k]["dR@5"] = res[k]["R@5"] - res["zrm"]["R@5"]
    return res


# ── the read, with the scores kept ───────────────────────────────────────────


class Capture:
    def __init__(self, LG):
        self.LG = LG
        self.orig_top_hit, self.orig_read_carve = LG.top_hit, LG.read_carve
        self.scores = None
        self.per = None
        self.i = 0
        self.want = {}
        self.carves = {}
        self.on_carve = None
        LG.top_hit = self.top_hit
        LG.read_carve = self.read_carve

    def top_hit(self, s, gold, B, seg, cnt):
        j = self.i % self.per if self.per else -1
        if j in self.want:
            self.want[j].append(s.detach().to("cpu").float().numpy().copy())
        self.i += 1
        return self.orig_top_hit(s, gold, B, seg, cnt)

    def read_carve(self, c, models, *args, **kw):
        names = [m[0] for m in models]
        if CAND not in names:
            raise SystemExit(f"bridge: {CAND} is not among the fit's candidates {names}")
        k = names.index(CAND)
        self.per, self.i = 4, 0                  # three refs, then p@swa alone (the other candidates are not read)
        self.want = {0: [], 3: []}
        r1 = self.orig_read_carve(c, [models[k]], *args, **kw)
        s = np.concatenate(self.want[3]) if self.want[3] else np.zeros(0, np.float32)
        base = np.concatenate(self.want[0]) if self.want[0] else np.zeros(0, np.float32)
        self.per, self.want = None, {}
        if s.size != int(c.off_np[-1]):
            raise SystemExit(f"bridge: {c.ds}/{c.carve}: kept {s.size} scores for {int(c.off_np[-1])} rows")
        r = dict(r1)                             # the work folder's read: p@swa in its place, zeros elsewhere
        for key in ("top", "hit", "scores64"):
            full = np.zeros((len(models),) + r1[key].shape[1:], r1[key].dtype)
            full[k] = r1[key][0]
            r[key] = full
        if self.on_carve:
            self.on_carve(c, s, base, r1["top"][0])
        return r


def analyse_carve(c, s, top_filed, look_root=None):
    import lean_cache as LC
    d, files, ids, _head = LC.look_chunks(c.ds, c.carve, look_root or LC.LM.LOOK)
    if list(ids) != list(c.ids):
        raise SystemExit(f"bridge: {c.ds}/{c.carve}: the look's ids are not the cache's")
    gold_c = c.gold.cpu().numpy() if hasattr(c.gold, "cpu") else np.asarray(c.gold)
    rows, qi, r_at = [], 0, 0
    for f in files:
        z = np.load(f)
        qps, qed = z["q_pool_size"].astype(np.int64), z["q_edges"].astype(np.int64)
        isg = z["is_gold"].astype(bool)
        eu, ev, ef = z["e_u"].astype(np.int64), z["e_v"].astype(np.int64), z["e_fam"].astype(np.int64)
        no = np.concatenate([[0], np.cumsum(qps)])
        eo = np.concatenate([[0], np.cumsum(qed)])
        for i in range(qps.size):
            n = int(qps[i])
            if n != int(c.n_np[qi]):
                raise SystemExit(f"bridge: {c.ds}/{c.carve} question {qi}: pool {n} in the look, {c.n_np[qi]} in the cache")
            g = isg[no[i]:no[i + 1]]
            if not np.array_equal(g, gold_c[r_at:r_at + n]):
                raise SystemExit(f"bridge: {c.ds}/{c.carve} question {qi}: gold flags differ between look and cache")
            q = question(s[r_at:r_at + n], g, int(c.gt_np[qi]), eu[eo[i]:eo[i + 1]], ev[eo[i]:eo[i + 1]],
                         ef[eo[i]:eo[i + 1]])
            if q["zrm"] != int(top_filed[qi]):
                raise SystemExit(f"bridge: {c.ds}/{c.carve} question {qi}: {q['zrm']} golds in the top five, the read "
                                 f"counted {int(top_filed[qi])}")
            rows.append(q)
            qi += 1
            r_at += n
    if qi != c.n_np.size or r_at != int(c.off_np[-1]):
        raise SystemExit(f"bridge: {c.ds}/{c.carve}: the look covers {qi} questions and {r_at} rows")
    return rows


def run_cmd(a):
    import torch
    torch.set_num_threads(a.threads)
    import zrm as ZR
    LG = ZR.LG
    t0 = time.time()
    src = Path(a.fit_root) / a.fit
    work = WORK / a.fit
    work.mkdir(parents=True, exist_ok=True)
    for fn in ("models.pt", "screen.json"):
        shutil.copyfile(src / fn, work / fn)
        if ZR.LC.sha_file(src / fn) != ZR.LC.sha_file(work / fn):
            raise SystemExit(f"bridge: the copy of {fn} differs")
    filed = json.loads((src / "read.json").read_text(encoding="utf-8"))
    res = {"fit": a.fit, "fit_root": Path(a.fit_root).as_posix(), "candidate": CAND,
           "models_sha256": ZR.LC.sha_file(src / "models.pt"), "script_sha256": ZR.LC.sha_src(__file__), "reads": {}}
    cap = Capture(LG)

    def on_carve(c, s, base, top_k):
        tc = time.time()
        rows = analyse_carve(c, s, top_k)
        sm = summarise(rows)
        key = f"{c.ds}={c.carve}"
        fr = filed["carves"].get(key, {}).get("swa_means", {}).get(CAND)
        sm["filed_R@5"] = fr[0] if fr else None
        sm["filed_ok"] = fr is not None and abs(sm["zrm"]["R@5"] - fr[0]) <= TOL_FILED
        sm["seconds"] = time.time() - tc
        res["reads"][key] = sm
        log(f"  {key}: zrm R@5 {sm['zrm']['R@5']:.4f} (filed {fr[0] if fr else None}); C1 {sm['C1']['dR@5']:+.4f} "
            f"C2 {sm['C2']['dR@5']:+.4f} O1 {sm['O1']['dR@5']:+.4f} O2 {sm['O2']['dR@5']:+.4f}; adj {sm.get('adj')} "
            f"lift {sm.get('lift')}")
    cap.on_carve = on_carve
    argv = ["read", "--name", a.fit, "--out-root", str(WORK), "--device", a.device, "--threads", str(a.threads)]
    if a.read:
        argv += ["--read", a.read]
    if a.host:
        argv.append("--host")
    rc = ZR.main(argv)
    if rc not in (0, None):
        raise SystemExit(f"bridge: zrm's read exited {rc}")
    res["seconds"] = time.time() - t0
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    bad = [k for k, v in res["reads"].items() if not v["filed_ok"]]
    res["identity"] = "PASS" if not bad else "FAIL"
    res["identity_fail"] = bad
    write(Path(a.out), res, table(res))
    if bad:
        log(f"bridge: {a.fit}: zrm's R@5 here differs from the filed read by more than {TOL_FILED} on {bad}")
        return 1
    log(f"bridge {a.fit}: {len(res['reads'])} reads in {res['seconds']:.0f}s")
    return 0


# ── the verdict ──────────────────────────────────────────────────────────────


def verdict(runs):
    """The declared rule over the runs (one per fit) on the multi-hop passage reads."""
    reach, oracle, free_ok = {}, {}, {}
    for r in runs:
        if r.get("identity") != "PASS":
            return {"verdict": "INCOMPLETE", "why": f"{r['fit']}: identity {r.get('identity')}"}
        rd = {k.split("=")[0]: v for k, v in r["reads"].items()}
        reach[r["fit"]] = {d: bool(rd[d].get("adj") is not None and rd[d]["adj"] >= ADJ_MIN
                                   and (rd[d].get("lift") or 0) >= LIFT_MIN) for d in MULTI if d in rd}
        oracle[r["fit"]] = {d: rd[d]["O1"]["dR@5"] >= ORACLE_MIN for d in MULTI if d in rd}
        free_ok[r["fit"]] = {}
        for rule in ("C1", "C2"):
            harm = any(v[rule]["dR@5"] < -FREE_HARM for v in rd.values())
            free_ok[r["fit"]][rule] = {d: (not harm) and rd[d][rule]["dR@5"] >= FREE_MIN for d in MULTI if d in rd}

    def every_fit_some(m):
        return all(any(v.values()) for v in m.values()) and len(m) == len(runs)
    R = every_fit_some(reach)
    O = every_fit_some(oracle)
    F = [rule for rule in ("C1", "C2") if every_fit_some({f: v[rule] for f, v in free_ok.items()})]
    if not R:
        v = "NOT_REACHABLE"
    elif F:
        v = "PARAMETER_FREE"
    elif O:
        v = "CONDITION"
    else:
        v = "REACHABLE_NOT_RANKED"
    return {"verdict": v, "reachable": reach, "oracle_pays": oracle, "label_free_pays": free_ok, "free_rules": F}


def table(res):
    lines = [f"| read | questions | zrm R@5 | filed | C1 | C2 | O1 | O2 | partly found | missed in pool | adj | struct | "
             "NER | kNN | 2-hop | non-gold adj | lift | nb rank median | nb rank <= 2 | nb median size |",
             "| --- |" + " ---: |" * 19]

    def f(x, p=4):
        return "" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))
    for k, v in res["reads"].items():
        lines.append(f"| {k} | {v['questions']} | {f(v['zrm']['R@5'])} | {f(v['filed_R@5'])} | {v['C1']['dR@5']:+.4f} | "
                     f"{v['C2']['dR@5']:+.4f} | {v['O1']['dR@5']:+.4f} | {v['O2']['dR@5']:+.4f} | "
                     f"{v['partial_questions']} | {v['missed_in_pool']} | {f(v.get('adj'), 3)} | "
                     f"{f(v.get('adj_fam0'), 3)} | {f(v.get('adj_fam1'), 3)} | {f(v.get('adj_fam2'), 3)} | "
                     f"{f(v.get('adj_2hop'), 3)} | {f(v.get('neg_adj'), 3)} | {f(v.get('lift'), 2)} | "
                     f"{f(v.get('nb_rank_median'), 1)} | {f(v.get('nb_rank_le2'), 3)} | {f(v.get('nb_size_median'), 1)} |")
    return "\n".join(lines) + "\n"


def write(out, rec, md):
    out.parent.mkdir(parents=True, exist_ok=True)
    for p, text in ((out.with_suffix(".json"), json.dumps(rec, indent=1, sort_keys=True) + "\n"), (out.with_suffix(".md"), md)):
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, p)


def verdict_cmd(a):
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in a.runs.split(",") if p]
    v = verdict(runs)
    v["runs"] = {r["fit"]: {"identity": r.get("identity"), "models_sha256": r["models_sha256"]} for r in runs}
    v["rule"] = {"multi": MULTI, "adj_min": ADJ_MIN, "lift_min": LIFT_MIN, "oracle_min": ORACLE_MIN,
                 "free_min": FREE_MIN, "free_harm": FREE_HARM}
    md = f"# Bridge diagnostic: {v['verdict']}\n\n" + "".join(f"## {r['fit']}\n\n" + table(r) + "\n" for r in runs)
    write(Path(a.out), v, md)
    log(f"bridge verdict: {v['verdict']}")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    # a chain 0-1-2-3-4-5-6-7 plus a kNN edge 0-7; golds 0 (found, top) and 7 (missed, next to 0 by kNN), 6 non-gold
    s = np.array([9, 8, 7, 6, 5, 1, 0.5, 2], np.float32)
    g = np.zeros(8, bool)
    g[[0, 7]] = True
    u = np.array([0, 1, 2, 3, 4, 5, 6, 0])
    v = np.array([1, 2, 3, 4, 5, 6, 7, 7])
    fam = np.array([0, 0, 0, 0, 0, 0, 0, 2])
    q = question(s, g, 2, u, v, fam)
    assert q["zrm"] == 1 and q["partial"] == 1 and q["missed"] == 1 and q["missed_adj"] == 1, q
    assert q["missed_adj0"] == 0 and q["missed_adj2"] == 1 and q["missed_adj2hop"] == 1, q
    # C1: top three 0,1,2 then neighbours of 0 not chosen: 7 (score 2) -> gold found
    assert q["C1"] == 2 and q["O1"] == 2 and q["O2"] == 2, q
    # C2: top three, best neighbour of 0 (7), best of 1 (none left: 0, 2 taken) -> fill 3
    assert q["C2"] == 2, q
    # non-gold rows outside the top five: 5, 6; next to found gold 0: neither
    assert q["neg"] == 2 and q["neg_adj"] == 0, q
    assert q["nb_ranks"] == [1] and q["nb_size"] == 1, q
    # order: NaN last, ties by position
    assert list(order_of(np.array([1.0, np.nan, 1.0, 2.0]))) == [3, 0, 2, 1]
    assert list(order_of(np.array([0.0, -0.0, 0.0]))) == [0, 1, 2]
    # no gold found: oracle keeps zrm's count; nothing partial
    g2 = np.zeros(8, bool)
    g2[6] = True
    q2 = question(s, g2, 1, u, v, fam)
    assert q2["zrm"] == 0 and q2["O1"] == 0 and "partial" not in q2 and q2["O2"] == 0, q2
    # gold outside the pool counts in gt only
    q3 = question(s, g, 3, u, v, fam)
    sm = summarise([q, q2, q3])
    assert abs(sm["zrm"]["R@5"] - (1 / 2 + 0 + 1 / 3) / 3) < 1e-12, sm
    assert abs(sm["C1"]["R@5"] - (1 + 0 + 2 / 3) / 3) < 1e-12, sm
    assert sm["partial_questions"] == 2 and sm["adj"] == 1.0 and sm["lift"] is None, sm
    # the verdict
    def run(fit, adj, lift, o1, c1, harm=0.0):
        rd = {}
        for d in ("musique", "2wiki", "squad"):
            rd[f"{d}=s1eval"] = {"adj": adj if d != "squad" else None, "lift": lift,
                                 "O1": {"dR@5": o1}, "C1": {"dR@5": c1 if d != "squad" else harm},
                                 "C2": {"dR@5": 0.0}}
        return {"fit": fit, "identity": "PASS", "reads": rd}
    assert verdict([run("a", 0.3, 3, 0.05, 0), run("b", 0.7, 3, 0.05, 0)])["verdict"] == "NOT_REACHABLE"
    assert verdict([run("a", 0.6, 3, 0.05, 0), run("b", 0.7, 3, 0.05, 0)])["verdict"] == "CONDITION"
    assert verdict([run("a", 0.6, 3, 0.01, 0), run("b", 0.7, 3, 0.05, 0)])["verdict"] == "REACHABLE_NOT_RANKED"
    assert verdict([run("a", 0.6, 3, 0.05, 0.01), run("b", 0.7, 3, 0.05, 0.01)])["verdict"] == "PARAMETER_FREE"
    assert verdict([run("a", 0.6, 3, 0.05, 0.01, -0.01), run("b", 0.7, 3, 0.05, 0.01)])["verdict"] == "CONDITION"
    assert verdict([run("a", 0.6, 1.5, 0.05, 0), run("b", 0.7, 3, 0.05, 0)])["verdict"] == "NOT_REACHABLE"
    bad = run("a", 0.6, 3, 0.05, 0)
    bad["identity"] = "FAIL"
    assert verdict([bad])["verdict"] == "INCOMPLETE"
    print("bridgediag selftest: ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("run", "verdict"))
    ap.add_argument("--fit")
    ap.add_argument("--fit-root", default="outputs/screen/fits")
    ap.add_argument("--read", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--runs", default="")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        return run_cmd(a)
    if a.cmd == "verdict":
        return verdict_cmd(a)
    ap.error("run, verdict or --selftest")


if __name__ == "__main__":
    sys.exit(main())
