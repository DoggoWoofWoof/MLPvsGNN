"""A diagnosis (no training; development numbers that decide nothing): within the depth that holds a gold, what tells
the gold from a fit's wrong top-1.

hopdiag.py (docs/SCREENS.md, 'Hop depth of the top-1 errors', filed 8 October about 03:06) found that most of rel's
metaqa 3-hop misses sit at a depth that holds a gold: 0.263 of the questions, the GNN's 0.108. Round eleven's depth
prior cannot reach them. This asks whether the fit's own inputs tell the gold from the top-1 there, and which inputs
do: whether the next round should change what the relation inputs say, or how the fit is trained on them.

For each fit and each question whose top-1 is not gold but sits at a depth that holds a gold (an at-depth miss), with
g the gold at that depth the fit scores highest:
  same_own       g and the top-1 are equal on every raw column of the fit's own blocks but SEMB, as the fit reads them
                 (NaN as 0): only the node's text embedding can tell them apart
  same_rel       equal on rel's 26 relation columns (typed_rel, typed_v2, ordered): the relation inputs cannot tell them
                 apart
  same_all       equal on every non-SEMB column of step 1's nine blocks and rel's three (112 columns)
  favours_gold   per column, the share of at-depth misses where g's value is above the top-1's (ties count half);
                 qrel_max is the largest over the twelve question-relation match columns (QREL)
  gnn0, twin0    the share where the look's GNN (twin) scores g above the top-1 (ties count half)
by metaqa's hop label and over all questions, on each read carve. depth is hopdiag.py's class (0 a seed, 1 to 3 the
first structural hop from a seed, 4 unreached), the top-1 hopdiag.py's (the first row reaching the question's max).

favours_gold is read on the fit's own errors, so selection bends it: a column the fit scores higher tends to sit
higher on the top-1 it chose. For the question-relation match columns (a better match should score higher) that bias
works against the gold, so qrel_max >= 0.65 is a conservative sign. The reading (RULE, docs/SCREENS.md, 'Within the
depth that holds a gold', declared before its numbers), on rel's fit, metaqa s1eval, the 3-hop questions:
  INPUT_GAP     same_rel >= 0.5: on half the misses the relation inputs are identical; round twelve adds columns from
                the existing graph that tell relation chains apart
  TRAINING_GAP  else qrel_max >= 0.65: a match column favours the gold against the selection; round twelve is an
                objective (a within-depth contrast on every dataset)
  MATCH_GAP     else: the relation columns differ but no match favours the gold; round twelve learns the
                question-relation match from the frozen embeddings instead of the fixed cosine

    python outputs/mp_unified/chaindiag.py --fit-dirs outputs/step1/fits/J5,outputs/full_rel/fits/J5 \\
        --read metaqa=s1eval,webqsp=s1eval --device cpu --threads 6 --host --out outputs/diag/chaindiag-J5
    python outputs/mp_unified/chaindiag.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hopdiag as H  # noqa: E402  (registers the rel and gcs arms through relcols)

R = H.R
S = H.S
LG, LC, LM = S.LG, S.LC, S.LM
log = S.log
STEP1 = ("rank", "dense_cos", "topo_STRUCT", "depth_STRUCT", "SEED", "WALK", "DISTS", "WALKF")
REL = tuple(R.ARM_BLOCKS["rel"])
ALL = STEP1 + REL
QREL = ("typed_rel:relmax_seed", "typed_rel:relmax_in", "typed_rel:relmean_in", "typed_rel:relchain2_max",
        "typed_v2:qsupport_h2", "typed_v2:qsupport_h3", "typed_v2:relpath_max_h3", "typed_v2:relpath_mean_h3",
        "typed_v2:relpath_min_h3", "ordered:opath_h3_q1", "ordered:opath_h3_q2", "ordered:opath_h3_q3")
REFS = ("gnn0", "twin0")
TOP_COLS = 15
RULE = {"fit_arm": "rel", "carve": "metaqa=s1eval", "group": "3hop", "same_rel": 0.5, "qrel_max": 0.65}


def col_names(widths, blocks):
    out = []
    for b in blocks:
        if b in R.NAMES:
            if len(R.NAMES[b]) != widths[b]:
                raise SystemExit(f"{b} is {widths[b]} wide, relcols names {len(R.NAMES[b])}")
            out += [f"{b}:{n}" for n in R.NAMES[b]]
        else:
            out += [f"{b}:{k}" for k in range(widths[b])]
    return out


@torch.no_grad()
def confusions(c, scorers, fits, blocks=ALL, rows_cap=LG.READ_ROWS):
    """Each fit's at-depth misses: question, top-1 row and g row (carve rows), both rows' columns over `blocks`, and the
    reference scorers' scores of both rows."""
    out = {n: {"q": [], "top": [], "g": [], "xt": [], "xg": [], **{f"{k}_t": [] for k in REFS},
               **{f"{k}_g": [] for k in REFS}} for n in fits}
    widths = None
    for q0, q1 in LG.batches(c.n_np, rows_cap):
        qs = np.arange(q0, q1)
        r0, r1 = int(c.off_np[q0]), int(c.off_np[q1])
        feats, _nq, _bz, gold = c.batch(qs, list(blocks))
        if widths is None:
            widths = {b: int(feats[b].shape[1]) for b in blocks}
        X = torch.cat([feats[b] for b in blocks], 1).cpu().numpy()
        d = H.depth_class(feats["WALK"].cpu().numpy())
        g_np = gold.cpu().numpy().astype(bool)
        cnt = c.n_np[q0:q1]
        start = np.cumsum(cnt) - cnt
        cache = {}
        ref = {k: scorers[k](c, qs, r0, r1, cache)[0].cpu().numpy().astype(np.float64) for k in REFS if k in scorers}
        for n in fits:
            s = scorers[n](c, qs, r0, r1, cache)[0].cpu().numpy().astype(np.float64)
            s = np.where(np.isnan(s), -np.inf, s)
            o = out[n]
            for i in range(q1 - q0):
                a, k = int(start[i]), int(cnt[i])
                if k == 0:
                    continue
                gi = g_np[a:a + k]
                t = int(np.argmax(s[a:a + k]))
                if gi[t] or not gi.any():
                    continue
                di = d[a:a + k]
                at = np.flatnonzero(gi & (di == di[t]))
                if at.size == 0:
                    continue
                g = int(at[np.argmax(s[a:a + k][at])])
                o["q"].append(q0 + i)
                o["top"].append(r0 + a + t)
                o["g"].append(r0 + a + g)
                o["xt"].append(X[a + t])
                o["xg"].append(X[a + g])
                for kk in REFS:
                    if kk in ref:
                        o[f"{kk}_t"].append(ref[kk][a + t])
                        o[f"{kk}_g"].append(ref[kk][a + g])
    for n in fits:
        o = out[n]
        w = sum(widths.values()) if widths else 0
        for key in ("xt", "xg"):
            o[key] = np.stack(o[key]) if o[key] else np.zeros((0, w), np.float32)
        for key in ("q", "top", "g"):
            o[key] = np.asarray(o[key], np.int64)
        for kk in REFS:
            for side in ("t", "g"):
                o[f"{kk}_{side}"] = np.asarray(o[f"{kk}_{side}"], np.float64)
    return out, widths


def prefers(xg, xt):
    """The share where the gold's value is above the top-1's, ties counting half (per column for 2-D input)."""
    return (xg > xt).astype(np.float64) + 0.5 * (xg == xt).astype(np.float64)


def summarise(conf, fits, widths, blocks, hops, gt):
    names = col_names(widths, blocks)
    spans, at = {}, 0
    for b in blocks:
        spans[b] = (at, at + widths[b])
        at += widths[b]
    qrel = [names.index(x) for x in QREL if x in names]
    rel_idx = [j for b in REL if b in spans for j in range(*spans[b])]
    ok = gt > 0
    groups = {"all": ok}
    for h in sorted(set(hops[ok].tolist()) - {0}):
        groups[f"{h}hop"] = ok & (hops == h)
    rows = {}
    for gname, gm in groups.items():
        row = {"questions": int(gm.sum()), "fits": {}}
        for n, own in fits.items():
            o = conf[n]
            m = gm[o["q"]] if o["q"].size else np.zeros(0, bool)
            xt, xg = o["xt"][m], o["xg"][m]
            own_idx = [j for b in own if b in spans for j in range(*spans[b])]
            eq = xt == xg
            M = int(m.sum())
            v = {"at_depth_misses": M, "at_depth_share": round(M / max(row["questions"], 1), 4)}
            if M:
                fav = prefers(xg, xt).mean(0)
                v.update({
                    "same_own": round(float(eq[:, own_idx].all(1).mean()), 4),
                    "same_rel": round(float(eq[:, rel_idx].all(1).mean()), 4) if rel_idx else None,
                    "same_all": round(float(eq.all(1).mean()), 4),
                    "cols_differing_mean": round(float((~eq).sum(1).mean()), 2),
                    "qrel_max": round(float(fav[qrel].max()), 4) if qrel else None,
                    "qrel_argmax": names[qrel[int(np.argmax(fav[qrel]))]] if qrel else None,
                    "favours_gold": {nm: round(float(f), 4) for nm, f in zip(names, fav)}})
                for kk in REFS:
                    if o[f"{kk}_t"].size:
                        v[f"{kk}_prefers_gold"] = round(float(prefers(o[f"{kk}_g"][m], o[f"{kk}_t"][m]).mean()), 4)
            row["fits"][n] = v
        rows[gname] = row
    return rows


def md_rows(rows):
    md = []
    for gname, row in rows.items():
        md += [f"### {gname}: {row['questions']} questions", "",
               "| fit | at-depth misses (share) | same_own | same_rel | same_all | columns differing (mean) | "
               "qrel_max (column) | gnn0 prefers gold | twin0 prefers gold |", "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
        for n, v in row["fits"].items():
            if not v["at_depth_misses"]:
                md.append(f"| {n} | 0 | | | | | | | |")
                continue
            md.append(f"| {n} | {v['at_depth_misses']} ({v['at_depth_share']:.4f}) | {v['same_own']:.4f} | "
                      f"{v['same_rel']} | {v['same_all']:.4f} | {v['cols_differing_mean']:.2f} | {v['qrel_max']} ({v['qrel_argmax']}) | "
                      f"{v.get('gnn0_prefers_gold', '')} | {v.get('twin0_prefers_gold', '')} |")
        md.append("")
        for n, v in row["fits"].items():
            if not v["at_depth_misses"]:
                continue
            top = sorted(v["favours_gold"].items(), key=lambda kv: -abs(kv[1] - 0.5))[:TOP_COLS]
            md += [f"{n}, the {TOP_COLS} columns furthest from one half (favours_gold): " +
                   ", ".join(f"{k} {x:.3f}" for k, x in top), ""]
    return md


def reading(rows, key):
    """RULE on one fit's rows of the rule's carve."""
    v = rows.get(RULE["group"], {}).get("fits", {}).get(key)
    if not v or not v["at_depth_misses"]:
        return "NO_MISSES"
    if v["same_rel"] is not None and v["same_rel"] >= RULE["same_rel"]:
        return "INPUT_GAP"
    if v["qrel_max"] is not None and v["qrel_max"] >= RULE["qrel_max"]:
        return "TRAINING_GAP"
    return "MATCH_GAP"


def run(a):
    t0 = time.time()
    flags = LG.set_flags(a.device)
    LG.bind_device_ops()
    if a.threads:
        torch.set_num_threads(a.threads)
    placement = None
    if a.host:
        import lean_host as LH
        placement = LH.substitute()
    loaded, own, arms, basis = {}, {}, {}, None
    for fd in [Path(p) for p in a.fit_dirs.split(",") if p]:
        meta = json.loads((fd / "screen.json").read_text(encoding="utf-8")) if (fd / "screen.json").exists() else {}
        arm = meta.get("arm", "base")
        if arm not in ("base", "rel"):
            raise SystemExit(f"{fd}: arm {arm}; this diagnosis reads base and rel fits")
        blob = torch.load(fd / "models.pt", weights_only=False, map_location="cpu")
        if basis is None:
            basis = (blob["basis"], blob["basis_sha256"])
        elif (blob["basis"], blob["basis_sha256"]) != basis:
            raise SystemExit(f"{fd}: another basis than the first fit's")
        with S.patched(arm):
            models = {n: (m, bl) for n, m, bl in LG.load_models(blob, a.device)}
        if a.candidate not in models:
            raise SystemExit(f"{fd}: no candidate {a.candidate}")
        key = f"{fd.parent.parent.name}/{fd.name}@{a.candidate}"
        loaded[key] = models[a.candidate]
        own[key] = [b for b in models[a.candidate][1] if b != "SEMB"]
        arms[key] = arm
        if not set(own[key]) <= set(ALL):
            raise SystemExit(f"{fd}: blocks {own[key]} outside {ALL}")
    cache_root = Path(a.cache_root) if a.cache_root else LC.OUT
    rec = {"fits": list(loaded), "own_blocks": own, "candidate": a.candidate, "basis": basis[0], "flags": flags,
           "placement": placement, "columns": None, "qrel": list(QREL), "rule": RULE, "reading": None, "carves": {},
           "script_sha256": LC.sha_src(__file__), "hopdiag_sha256": LC.sha_src(H.__file__)}
    md = ["# Within the depth that holds a gold: what tells it from the top-1 (a diagnosis; no training, decides "
          "nothing)", "",
          "at-depth miss: the top-1 is not gold and sits at a depth that holds a gold; g the gold at that depth the fit "
          "scores highest. same_own: g and the top-1 equal on every non-SEMB column of the fit's blocks; same_all: on "
          "all 112 (step 1's nine blocks and rel's three). favours_gold: the share where g's value is above the "
          "top-1's, ties half. qrel_max: the largest over the twelve question-relation match columns.", ""]
    for ds, cv in LM.parse_sets(a.read):
        tc = time.time()
        with S.patched("rel"):
            c = LG.CacheCarve(ds, cv, basis[0], a.device, cache_root, not a.no_verify, score2=True)
        if c.basis_sha256 != basis[1]:
            raise SystemExit(f"{ds}/{cv}: the cache's basis is not the fits'")
        conf, widths = confusions(c, H.scorers_for(c, loaded), loaded, ALL, a.rows_cap)
        if rec["columns"] is None:
            rec["columns"] = col_names(widths, ALL)
        rows = summarise(conf, own, widths, ALL, H.hop_label(c.ids), c.gt_np)
        rec["carves"][f"{ds}={cv}"] = {"rows": rows, "seconds": round(time.time() - tc, 1)}
        md += [f"## {ds} {cv}", ""] + md_rows(rows)
        if f"{ds}={cv}" == RULE["carve"]:
            for key, arm in arms.items():
                if arm == RULE["fit_arm"]:
                    rec["reading"] = {"fit": key, "verdict": reading(rows, key)}
                    md[4:4] = [f"**Reading (declared before its numbers):** {rec['reading']['verdict']} ({key}, "
                               f"{RULE['carve']}, {RULE['group']}; INPUT_GAP if same_rel >= {RULE['same_rel']}, else "
                               f"TRAINING_GAP if qrel_max >= {RULE['qrel_max']}, else MATCH_GAP)", ""]
        log(f"chaindiag {ds}/{cv}: {c.rows} questions ({time.time() - tc:.0f}s)")
        del c
    rec["seconds"] = round(time.time() - t0, 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"chaindiag: {out.with_suffix('.md')} ({rec['seconds']:.0f}s)")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


class ToyCarve:
    """Three questions with pools of four, three and three rows; blocks WALK (16) and typed_v2 (6)."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.n_np = np.array([4, 3, 3], np.int64)
        self.off_np = np.array([0, 4, 7, 10], np.int64)
        self.rows = 3
        self.gt_np = np.array([1, 1, 1], np.int64)
        self.ids = np.array(["metaqa:3hop:dev:0", "metaqa:3hop:dev:1", "metaqa:2hop:dev:2"])
        depth = [0, 3, 3, 1, 0, 2, 3, 0, 3, 3]
        W = np.zeros((10, H.WALK_W), np.float32)
        for r, k in enumerate(depth):
            if k == 0:
                W[r, H.IS_SEED] = 1
            elif k <= 3:
                W[r, H.FIRST_HOP[k - 1]] = 1
        T = np.zeros((10, 6), np.float32)
        T[2, 4] = 0.9        # q0: the gold (row 2) is above the top-1 (row 1) on relpath_mean_h3 alone
        T[1, 4] = 0.2
        self.W, self.T = torch.from_numpy(W), torch.from_numpy(T)
        self.gold = torch.tensor([0, 0, 1, 0, 0, 0, 1, 0, 1, 0], dtype=torch.uint8)
        # the fit: q0 top-1 row 1 (depth 3 holds the gold row 2): an at-depth miss; q1 top-1 row 5 (depth 2, no gold
        # there): off-depth, not counted; q2 top-1 row 9 (depth 3 holds gold row 8, every column equal): at-depth
        self.fit = np.array([0, 5, 4, 1, 0, 6, 2, 0, 3, 7], np.float32)
        # gnn0 above the top-1 on q0's gold, tied on q2's; twin0 below on both
        self.score2 = np.array([[0, 0], [1, 3], [0, 5], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 2], [1, 2]],
                               np.float32)

    def batch(self, qs, blocks):
        qs = np.asarray(qs, np.int64)
        idx = np.concatenate([np.arange(self.off_np[q], self.off_np[q + 1]) for q in qs])
        nq = torch.from_numpy(np.repeat(np.arange(qs.size), self.n_np[qs]))
        feats = {"WALK": self.W[idx], "typed_v2": self.T[idx], "rank": torch.zeros((idx.size, 5))}
        return {b: feats[b] for b in blocks}, nq, torch.zeros(idx.size), self.gold[idx]


def selftest():
    t0 = time.time()
    c = ToyCarve()

    def fit_scorer(c_, qs, r0, r1, cache):
        return torch.from_numpy(c_.fit[r0:r1]), c_.gold[r0:r1]

    sc = H.scorers_for(c, {})
    sc["toy"] = fit_scorer
    blocks = ("WALK", "typed_v2")
    conf, widths = confusions(c, sc, {"toy": None}, blocks, rows_cap=100)
    o = conf["toy"]
    assert o["q"].tolist() == [0, 2] and o["top"].tolist() == [1, 9] and o["g"].tolist() == [2, 8], o
    assert widths == {"WALK": 16, "typed_v2": 6}
    rows = summarise(conf, {"toy": list(blocks)}, widths, blocks, H.hop_label(c.ids), c.gt_np)
    v = rows["all"]["fits"]["toy"]
    assert v["at_depth_misses"] == 2 and v["same_own"] == 0.5 and v["same_all"] == 0.5 and v["same_rel"] == 0.5, v
    assert v["favours_gold"]["typed_v2:relpath_mean_h3"] == 0.75 and v["favours_gold"]["WALK:0"] == 0.5
    assert v["qrel_max"] == 0.75 and v["qrel_argmax"] == "typed_v2:relpath_mean_h3"
    assert v["gnn0_prefers_gold"] == 0.75 and v["twin0_prefers_gold"] == 0.0
    assert rows["3hop"]["fits"]["toy"]["at_depth_misses"] == 1 and rows["3hop"]["fits"]["toy"]["same_own"] == 0.0
    assert rows["2hop"]["fits"]["toy"]["same_own"] == 1.0 and rows["all"]["questions"] == 3
    assert rows["3hop"]["fits"]["toy"]["same_rel"] == 0.0 and rows["2hop"]["fits"]["toy"]["same_rel"] == 1.0
    # the reading: 3hop holds q0 only (same_rel 0, qrel_max 1.0) -> TRAINING_GAP; then each branch by hand
    assert reading(rows, "toy") == "TRAINING_GAP" and reading(rows, "other") == "NO_MISSES"
    v3 = rows["3hop"]["fits"]["toy"]
    assert reading({"3hop": {"fits": {"toy": {**v3, "same_rel": 0.5}}}}, "toy") == "INPUT_GAP"
    assert reading({"3hop": {"fits": {"toy": {**v3, "qrel_max": 0.6499}}}}, "toy") == "MATCH_GAP"
    assert reading({"3hop": {"fits": {"toy": {**v3, "qrel_max": 0.65}}}}, "toy") == "TRAINING_GAP"
    # own blocks without typed_v2: the two rows of q0 are then equal on the fit's own columns
    rows2 = summarise(conf, {"toy": ["WALK"]}, widths, blocks, H.hop_label(c.ids), c.gt_np)
    assert rows2["all"]["fits"]["toy"]["same_own"] == 1.0 and rows2["all"]["fits"]["toy"]["same_all"] == 0.5
    md = md_rows(rows)
    assert any("typed_v2:relpath_mean_h3 0.750" in x for x in md)
    try:
        col_names({"typed_v2": 5}, ("typed_v2",))
        raise AssertionError("a width other than relcols' names must be refused")
    except SystemExit:
        pass
    print(f"chaindiag selftest: at-depth misses, g, same_own/same_rel/same_all, favours_gold, the reference scorers, the "
          f"hop groups and the reading on a toy carve ({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-dirs", default="")
    ap.add_argument("--candidate", default="p@swa")
    ap.add_argument("--read", default="metaqa=s1eval,webqsp=s1eval")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--rows-cap", type=int, default=LG.READ_ROWS)
    ap.add_argument("--cache-root")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--out", default="outputs/diag/chaindiag")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
