"""Step 1 carves (docs/STEP1_MATCHED_SELECTION.md, section 2): for each training dataset a select carve matched to its
eval population, musique's fit regrouped so that its matched select carve can share no single-hop component with it,
and the eval reads. Deterministic. Reads the served package read-only (the loader of m3b_population_overlap.py) and
writes outputs/step1/carves.json. It counts and lists question ids only: no score, weight or model is read.

    python scripts/step1_carves.py            # writes outputs/step1/carves.json
    python scripts/step1_carves.py --check    # recomputes and compares with the filed outputs/step1/carves.json

The rules (the declaration's section 2):
  cell(q)     (o, t): o = 1 if one of q's gold nodes is a gold node of a question in the fit carve, t = the stratum
              (metaqa hop; squad answerable; musique hop; hotpotqa level/type; 2wiki type)
  target      the eval population's share of each cell (golds resolved and zero-gold questions out, as M3B's eval)
  size        n = the M3B select carve's size after zero-gold exclusion, so the pooled select weighs datasets as before
  quotas      largest-remainder rounding of n x share; a remainder tie goes to the cell named first
  candidates  train questions outside the fit carve and the reserved carves (2wiki x2 and x3, musique x3), zero-gold
              out; on musique only questions of the select side's groups (below). Ordered by sha256("<dataset>:<qid>");
              each cell takes its quota in that order; a cell short of candidates takes what there is, and the select
              carve is that much smaller (recorded, not refilled).
  musique     questions linked through a shared single-hop component (the question id's 2hop__A_B) form groups.
              Groups in the order of sha256("musique:group:<smallest qid>") go to the select side until its eligible
              questions (train outside x3, zero-gold out) reach m x the hop quotas (m from 1.25 in steps of 0.25, until
              the quotas fill after the fit-gold filter); every other group is the fit side. The regrouped fit takes,
              per hop, as many questions as the M3B fit has, in sha256 order, from the fit side's questions outside
              x3 and outside the M3B select carve (which stays the M arm's select carve, out of sample). The matched
              select carve then fills the cells from the select side.
  eval        the M3B eval population of each dataset (webqsp's train_holdout included); metaqa reads every 4th of its
              39,138 questions in population order, the others read all.
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from m3b_run import load_script, CONFIG, HEADROOM_CONFIG  # noqa: E402

OUT = ROOT / "outputs" / "step1" / "carves.json"
TRAIN = ("metaqa", "squad", "musique", "hotpotqa", "2wiki")
ALL = TRAIN + ("webqsp",)
RESERVED = {"2wiki": ("x2", "x3"), "musique": ("x3",)}
EVAL_STRIDE = {"metaqa": 4}
NAMES = {"fit": "fit", "mselect": "select", "dselect": "s1sel", "eval": "s1eval", "musique_fit": "s1fit"}
MARGIN0, MARGIN_STEP, MARGIN_MAX = 1.25, 0.25, 4.0


def digest(ids: list[str]) -> str:
    return hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest()


def order_key(name: str, qid: str) -> str:
    return hashlib.sha256(f"{name}:{qid}".encode("utf-8")).hexdigest()


def hop_of(qid: str) -> int:
    m = re.match(r"(\d)hop\d*__", qid)
    if not m:
        raise SystemExit(f"musique id {qid} names no hop count")
    return int(m.group(1))


def components(qid: str) -> frozenset:
    m = re.match(r"\dhop\d*__(.*)$", qid)
    if not m:
        raise SystemExit(f"musique id {qid} names no components")
    return frozenset(m.group(1).split("_"))


def stratum(name: str, row: dict) -> str:
    if name == "metaqa":
        return f"hop{row['hop']}"
    if name == "musique":
        return f"hop{hop_of(row['query_id'])}"
    if name == "squad":
        return f"answerable={row['answerable']}"
    if name == "hotpotqa":
        return f"{row['level']}/{row['type']}"
    if name == "2wiki":
        return str(row["type"])
    raise SystemExit(f"{name} has no stratum")


def quotas(counts: dict[str, int], n: int) -> dict[str, int]:
    total = sum(counts.values())
    raw = {c: n * k / total for c, k in counts.items()}
    q = {c: int(math.floor(v)) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: (-(raw[c] - q[c]), c))[: n - sum(q.values())]:
        q[c] += 1
    return dict(sorted(q.items()))


def mix(cells: list[str]) -> dict[str, float]:
    c = collections.Counter(cells)
    n = max(sum(c.values()), 1)
    return {k: round(v / n, 4) for k, v in sorted(c.items())}


def x_carves(source: list[str], select: list[str], fit: list[str], fit_cap: int) -> dict[str, list[str]]:
    """m3b_pools.carve_ids' remaining set and fit stride (mp_approx_l12.carve_rule's formulas); x_j = remaining[j::s_fit]."""
    sel = set(select)
    remaining = [q for q in source if q not in sel]
    s_fit = max(1, math.ceil(len(remaining) / fit_cap))
    if remaining[0::s_fit] != fit:
        raise SystemExit("the recomputed fit stride is not carves_for's fit")
    return {f"x{j}": remaining[j::s_fit] for j in range(1, s_fit)}


def fill(name: str, cand: list[str], cell_of: dict[str, str], q: dict[str, int]) -> tuple[list[str], dict, dict]:
    by_cell = collections.defaultdict(list)
    for qid in sorted(cand, key=lambda x: order_key(name, x)):
        by_cell[cell_of[qid]].append(qid)
    take, short = [], {}
    for c, k in q.items():
        got = by_cell.get(c, [])[:k]
        take += got
        if len(got) < k:
            short[c] = k - len(got)
    avail = {c: len(by_cell.get(c, [])) for c in q}
    return sorted(take), short, avail


def musique_regroup(train_rows, zero, select_ids, x3_ids, fit_hops, sel_quota_hops, fills):
    """The regrouped fit and the select side (the module docstring); fills(fit, side) says whether the select side's
    eligible questions fill the matched select carve's cells. Returns (fit ids, select-side eligible ids, record)."""
    ids = [r["query_id"] for r in train_rows]
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for qid in ids:
        cs = sorted(components(qid))
        for c in cs[1:]:
            ra, rb = find(cs[0]), find(c)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    group = {qid: find(sorted(components(qid))[0]) for qid in ids}
    members = collections.defaultdict(list)
    for qid in ids:
        members[group[qid]].append(qid)
    order = sorted(members, key=lambda g: hashlib.sha256(f"musique:group:{min(members[g])}".encode("utf-8")).hexdigest())
    x3, sel = set(x3_ids), set(select_ids)
    elig_s = {q for q in ids if q not in x3 and not zero[q]}
    elig_f = {q for q in elig_s if q not in sel}
    margin = MARGIN0
    while margin <= MARGIN_MAX + 1e-9:
        need = {h: math.ceil(margin * k) for h, k in sel_quota_hops.items()}
        have = collections.Counter()
        s_groups = []
        for g in order:
            if all(have[h] >= need[h] for h in need):
                break
            gh = collections.Counter(hop_of(q) for q in members[g] if q in elig_s)
            if any(gh[h] and have[h] < need[h] for h in need):
                s_groups.append(g)
                have.update(gh)
        s_side = {q for g in s_groups for q in members[g]}
        f_pool = sorted((q for q in elig_f if q not in s_side), key=lambda x: order_key("musique", x))
        fit, short_f = [], {}
        for h, k in sorted(fit_hops.items()):
            got = [q for q in f_pool if hop_of(q) == h][:k]
            fit += got
            if len(got) < k:
                short_f[h] = k - len(got)
        if short_f:
            raise SystemExit(f"musique: the fit side cannot refill the M3B fit's hop counts at margin {margin}: short {short_f}")
        s_ok = sorted(q for q in s_side if q in elig_s)
        if fills(fit, s_ok):
            fit = sorted(fit)
            fc = set(c for q in fit for c in components(q))
            if any(components(q) & fc for q in s_side):
                raise SystemExit("musique: a select-side question shares a component with the regrouped fit")
            sizes = sorted((len(v) for v in members.values()), reverse=True)
            rec = {"groups": len(members), "largest_groups": sizes[:5], "select_side_groups": len(s_groups),
                   "select_side_questions": len(s_side), "select_side_eligible_by_hop": dict(sorted(have.items())),
                   "margin": margin,
                   "fit_side_eligible": len(f_pool), "fit_by_hop": dict(sorted(collections.Counter(hop_of(q) for q in fit).items()))}
            return fit, s_ok, rec
        margin += MARGIN_STEP
    raise SystemExit("musique: no margin up to the cap fills the select quotas")


def build(names: list[str]) -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg_h = yaml.safe_load(HEADROOM_CONFIG.read_text(encoding="utf-8"))
    m3b_compile = load_script("m3b_compile")
    m3a, canonical, served, freeze = m3b_compile.open_package(cfg)
    fit_cap = int(cfg["populations"]["training_carves"]["fit"]["size_cap"])
    rec = {"declared_in": "docs/STEP1_MATCHED_SELECTION.md", "script": "scripts/step1_carves.py",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
           "freeze_RECORD_SHA256": freeze["RECORD_SHA256"], "carve_names": NAMES, "eval_stride": EVAL_STRIDE,
           "reserved": {k: list(v) for k, v in RESERVED.items()}, "per_dataset": {}}
    for name in names:
        t0 = time.time()
        ds = canonical.Dataset(name, root=str(served))
        positions = m3a.node_position_map(ds)
        ev = m3b_compile.population(ds, name, "eval", cfg, cfg_h, m3a, positions)
        stride = EVAL_STRIDE.get(name, 1)
        read = list(ev.ids[::stride])
        entry = {"eval": {"name": NAMES["eval"], "split": cfg["populations"]["eval_splits"][name], "n_population": len(ev.ids),
                          "population_sha256": ev.digest, "zero_gold_excluded": ev.zero_gold_excluded, "stride": stride,
                          "n": len(read), "sha256": digest(read), "ids": read}}
        if name not in TRAIN:
            entry["note"] = "never a training graph of the lean track: read zero-shot by every fit, no select carve"
            rec["per_dataset"][name] = entry
            log(f"{name}: eval {len(read)} of {len(ev.ids)} ({time.time() - t0:.0f}s)")
            continue
        fitp = m3b_compile.population(ds, name, "fit", cfg, cfg_h, m3a, positions)
        selp = m3b_compile.population(ds, name, "select", cfg, cfg_h, m3a, positions)
        source, fit_raw, sel_raw = m3b_compile.carves_for(ds, name, cfg)
        xs = x_carves(source, sel_raw, fit_raw, fit_cap)
        reserved = set(q for x in RESERVED.get(name, ()) for q in xs[x])
        train_rows = [r for r in ds.queries("train")]
        if name == "webqsp":
            raise SystemExit("webqsp is not a training dataset here")
        golds = m3a.resolve_gold(train_rows, positions, name)
        gold_of = {r["query_id"]: g for r, g in zip(train_rows, golds)}
        zero = {q: g.size == 0 for q, g in gold_of.items()}
        row_of = {r["query_id"]: r for r in train_rows}
        ev_rows = {r["query_id"]: r for r in m3a.population_rows(ds, entry["eval"]["split"], cfg_h)[1]}
        ev_gold = dict(zip(ev.ids, ev.golds))
        n_sel = len(selp.ids)
        def cells(fit_ids):
            fit_gold = set(int(g) for x in fit_ids for g in gold_of[x])
            hold = lambda gs: int(any(int(g) in fit_gold for g in gs))  # noqa: E731
            cell_of = {x: f"o{hold(gold_of[x])}|{stratum(name, row_of[x])}" for x in gold_of if not zero[x]}
            ev_cells = [f"o{hold(ev_gold[x])}|{stratum(name, ev_rows[x])}" for x in ev.ids]
            return cell_of, ev_cells, quotas(dict(collections.Counter(ev_cells)), n_sel)

        if name == "musique":
            ev_hops = collections.Counter(hop_of(x) for x in ev.ids)
            sel_quota_hops = {int(c[3:]): k for c, k in quotas({f"hop{h}": k for h, k in ev_hops.items()}, n_sel).items()}
            fit_hops = collections.Counter(hop_of(x) for x in fitp.ids)

            def fills(fit_ids, side):
                cell_of, _ev, q_ = cells(fit_ids)
                return not fill(name, side, cell_of, q_)[1]

            fit_ids, cand, grec = musique_regroup(train_rows, zero, selp.ids, xs["x3"], dict(fit_hops), sel_quota_hops, fills)
            entry["regroup"] = grec
            fit_name, fit_source = NAMES["musique_fit"], "regrouped"
        else:
            fit_ids, fit_name, fit_source = list(fitp.ids), NAMES["fit"], "m3b"
            fset = set(fit_ids)
            cand = [x for x in gold_of if x not in fset and x not in reserved and not zero[x]]
        cell_of, ev_cells, q = cells(fit_ids)
        dsel, short, avail = fill(name, cand, cell_of, q)
        if set(dsel) & set(fit_ids):
            raise SystemExit(f"{name}: the matched select carve meets the fit carve")
        entry["fit"] = {"name": fit_name, "source": fit_source, "n": len(fit_ids), "sha256": digest(fit_ids),
                        "m3b_fit_sha256": fitp.digest}
        if fit_source == "regrouped":
            entry["fit"]["ids"] = fit_ids
        entry["mselect"] = {"name": NAMES["mselect"], "n": n_sel, "sha256": selp.digest}
        entry["dselect"] = {"name": NAMES["dselect"], "n": len(dsel), "sha256": digest(dsel), "quotas": q, "short": short,
                            "candidates_by_cell": avail, "candidates": len(cand), "ids": dsel}
        sel_cells = [cell_of[x] for x in selp.ids]
        entry["mix"] = {"fit": mix([cell_of[x] for x in fit_ids]), "mselect": mix(sel_cells),
                        "dselect": mix([cell_of[x] for x in dsel]), "eval": mix(ev_cells)}
        entry["holds_a_fit_gold"] = {k: round(float(np.mean([c.startswith("o1") for c in v])), 4) for k, v in
                                     (("mselect", sel_cells), ("dselect", [cell_of[x] for x in dsel]), ("eval", ev_cells))}
        if name == "musique":
            fc = set(c for x in fit_ids for c in components(x))
            entry["shares_a_component_with_fit"] = {
                k: round(float(np.mean([bool(components(x) & fc) for x in ids])), 4)
                for k, ids in (("mselect", selp.ids), ("dselect", dsel), ("eval", ev.ids))}
        rec["per_dataset"][name] = entry
        log(f"{name}: fit {len(fit_ids)} ({fit_source}), select M {n_sel} / D {len(dsel)} (short {short}), eval {len(read)} of "
            f"{len(ev.ids)}; holds a fit gold M {entry['holds_a_fit_gold']['mselect']} D {entry['holds_a_fit_gold']['dselect']} "
            f"eval {entry['holds_a_fit_gold']['eval']} ({time.time() - t0:.0f}s)")
        del positions, train_rows, golds, gold_of, row_of, ev_rows
        gc.collect()
    return rec


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def strip(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in ("utc", "seconds")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="recompute and compare with the filed carves.json")
    ap.add_argument("--datasets", default=",".join(ALL))
    a = ap.parse_args()
    names = [n for n in a.datasets.split(",") if n]
    t0 = time.time()
    rec = build(names)
    if a.check:
        old = json.loads(OUT.read_text(encoding="utf-8"))
        bad = [n for n in names if json.dumps(old["per_dataset"].get(n), sort_keys=True) != json.dumps(rec["per_dataset"][n], sort_keys=True)]
        print("check:", "IDENTICAL" if not bad else f"DIFFERS on {bad}")
        return 1 if bad else 0
    rec["utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec["seconds"] = round(time.time() - t0, 1)
    if OUT.exists() and set(names) != set(ALL):
        old = json.loads(OUT.read_text(encoding="utf-8"))
        old["per_dataset"].update(rec["per_dataset"])
        rec["per_dataset"] = old["per_dataset"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, OUT)
    log(f"wrote {OUT} ({rec['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
