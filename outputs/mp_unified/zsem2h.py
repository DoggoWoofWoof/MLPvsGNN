"""SEM2 R1h: hubs (docs/SEM2_EDGE_OPERATORS.md section 8, declared 24 Nov 2026 before any R1h number).

    python outputs/mp_unified/zsem2h.py selftest [--threads 2]
    python outputs/mp_unified/zsem2h.py hubread
    python outputs/mp_unified/zsem2h.py r1h --ds D --arm e3v|frozen [--host] [--threads 8]
    python outputs/mp_unified/zsem2h.py report

Part A (hubread): R1's saved ranks grouped by (read group, target): the target-macro MRR (MRR_T) and call H0.
Part B: logit adjustment at query time, score - tau log(1 + deg_L(y)) with R1's tau, on the cosine and R1's rotation.
Part C: the hub-balanced fit (HB): R1's rotation with each training positive weighted 1 / deg_L(target), stopped on
dev MRR_T. Popularity (rank by deg_L) beside. Gate G-R: R1's samples and ranks are reproduced first.
zsem2 is imported and never edited. Outputs only under outputs/sem2/r1h/ (json and md committed; npz not).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zsem2 as Z  # noqa: E402  (zenc reads --threads from sys.argv before numpy loads)
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 8"
OUT = Z.ROOT / "outputs" / "sem2" / "r1h"
R1 = Z.OUT
HUB_EDGES = 10              # 8.2's descriptive hub cut (sampled edges into one target)
CALLS = {"ABOVE": "LEARNS", "LEVEL": "LEVEL", "BELOW": "LOSES"}
log, stop, wjson, sha_file = Z.log, Z.stop, Z.wjson, Z.sha_file


# ── target-macro MRR (8.1) ──────────────────────────────────────────────────


def gkeys(grp, d):
    return np.asarray(grp, np.int64) * (1 << 40) + np.asarray(d, np.int64)


def group_vals(rr, keys):
    u, inv = np.unique(keys, return_inverse=True)
    s = np.bincount(inv, weights=np.asarray(rr, float), minlength=u.size)
    c = np.bincount(inv, minlength=u.size)
    return s / c, u


def mrr_t(rank, keys):
    if rank.size == 0:
        return None
    return round(100 * float(group_vals(1.0 / rank, keys)[0].mean()), 2)


def paired_t(rank_x, rank_y, keys):
    """MRR_T(x) - MRR_T(y) in points, paired 95% bootstrap over (group, target) groups (2,000 resamples, seed 0)."""
    import zap1 as ZA
    vx, ux = group_vals(1.0 / rank_x, keys)
    vy, uy = group_vals(1.0 / rank_y, keys)
    if not np.array_equal(ux, uy):
        stop("paired_t: the two reads group differently")
    n = vx.size
    dlt, lo, hi = ZA.boot_diff(vx, vy, np.arange(n), n, Z.BOOT, Z.SEED)
    return {"diff": ZA.pct(dlt), "lo": ZA.pct(lo), "hi": ZA.pct(hi), "call": ZA.call(lo, hi), "groups": int(n)}


def read_both(rank, smp, names):
    keys = gkeys(smp["grp"], smp["d"])
    out = {"pooled": Z.measures(rank), "mrr_t": mrr_t(rank, keys), "per_group": {}}
    for g, nm in enumerate(names):
        m = smp["grp"] == g
        if m.any():
            out["per_group"][nm] = {"mrr": Z.measures(rank[m]).get("mrr"), "mrr_t": mrr_t(rank[m], keys[m])}
    return out


def hub_desc(grp, d, names):
    out = {}
    for g, nm in enumerate(names):
        m = grp == g
        if not m.any():
            continue
        _, c = np.unique(d[m], return_counts=True)
        out[nm] = {"edges": int(m.sum()), "targets": int(c.size), "largest_target_edges": int(c.max()),
                   "hub_edge_share": round(100 * float(c[c >= HUB_EDGES].sum()) / int(m.sum()), 2)}
    return out


def per_group_t(rx, ry, smp, names):
    keys = gkeys(smp["grp"], smp["d"])
    out = {}
    for g, nm in enumerate(names):
        m = smp["grp"] == g
        if len(np.unique(smp["d"][m])) > 1:
            out[nm] = paired_t(rx[m], ry[m], keys[m])
    return out


# ── part A: the hub read of R1's saved ranks (8.2) ──────────────────────────


def load_r1(d, arm, base=R1):
    pj, pz = base / d / f"{arm}.json", base / d / f"{arm}.npz"
    if not (pj.is_file() and pz.is_file()):
        return None, None
    r = json.loads(pj.read_text(encoding="utf-8"))
    if r.get("smoke"):
        stop(f"{pj} is a smoke run")
    if sha_file(pz) != r["npz_sha256"]:
        stop(f"{pz}: sha256 is not its record's")
    return r, np.load(pz)


def hub_row(r, z):
    names = list(r["read_groups"])
    smp = {"grp": z["test_grp"], "d": z["test_d"]}
    keys = gkeys(smp["grp"], smp["d"])
    forms = [k[5:] for k in z.files if k.startswith("rank_")]
    h0 = paired_t(z["rank_rotation"], z["rank_cosine"], keys)
    h0["call"] = CALLS[h0["call"]]
    return {"forms": {f: {"pooled_mrr": Z.measures(z[f"rank_{f}"]).get("mrr"), "mrr_t": mrr_t(z[f"rank_{f}"], keys)}
                      for f in forms},
            "H0": h0, "R1_pooled_call": r["call"],
            "per_group_H0": per_group_t(z["rank_rotation"], z["rank_cosine"], smp, names),
            "hubs": hub_desc(smp["grp"], smp["d"], names), "npz_sha256": r["npz_sha256"]}


def cmd_hubread(a):
    rows = {}
    for d in Z.SEVEN:
        for arm in Z.ARMS:
            r, z = load_r1(d, arm)
            if r is not None:
                rows[f"{d} {arm}"] = hub_row(r, z)
                log(f"hubread {d} {arm}: MRR_T cosine {rows[f'{d} {arm}']['forms']['cosine']['mrr_t']}, rotation "
                    f"{rows[f'{d} {arm}']['forms']['rotation']['mrr_t']}: H0 {rows[f'{d} {arm}']['H0']}")
    if not rows:
        stop("no R1 npz found")
    rep = {"stage": "SEM2 R1h part A (hub read of R1)", "declared_in": DECLARED, "rows": rows,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": sha_file(Path(__file__))}
    wjson(OUT / "hubread.json", rep)
    L = ["# SEM2 R1h part A: R1's ranks grouped by target", "",
         f"Declared in {DECLARED}. Written by zsem2h.py hubread at {rep['utc']}. MRR in points. MRR_T: each (read group, "
         "target) counts once. H0: rotation - cosine on MRR_T, paired bootstrap over groups.", "",
         "| dataset arm | cosine MRR / MRR_T | rotation MRR / MRR_T | pooled gain (R1's call) | MRR_T gain (H0) |",
         "| --- | --- | --- | --- | --- |"]
    for k, v in rows.items():
        c, ro, h, p = v["forms"]["cosine"], v["forms"]["rotation"], v["H0"], v["R1_pooled_call"]
        L.append(f"| {k} | {c['pooled_mrr']} / {c['mrr_t']} | {ro['pooled_mrr']} / {ro['mrr_t']} | "
                 f"{p['diff']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}] {p['call']} | "
                 f"{h['diff']:+.2f} [{h['lo']:+.2f}, {h['hi']:+.2f}] {h['call']} |")
    (OUT / "hubread.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return 0


# ── degrees (8.1) and popularity (8.4) ──────────────────────────────────────


class Deg:
    """deg_L(y) over training edges only: R labels as n-length counts, S relations as sorted (rel * n + dst) keys."""

    def __init__(self, G, split):
        n = G.n
        self.n, self.nR = n, G.nR
        self.r = []
        for keys in G.r_keys:
            c = np.zeros(n, np.int64)
            for i in range(0, keys.size, 1 << 24):
                a, b = Z.ends(keys[i:i + (1 << 24)], n)
                m = (split[a] == Z.TRAIN) & (split[b] == Z.TRAIN)
                c += np.bincount(a[m], minlength=n) + np.bincount(b[m], minlength=n)
            self.r.append(c)
        if G.s_key.size:
            self.trS = (split[G.s_src] == Z.TRAIN) & (split[G.s_dst] == Z.TRAIN)
            self.s_keys, self.s_cnt = np.unique(G.s_rel[self.trS] * n + G.s_dst[self.trS], return_counts=True)
        else:
            self.trS = np.zeros(0, bool)
            self.s_keys, self.s_cnt = np.zeros(0, np.int64), np.zeros(0, np.int64)
        self._dist = {}

    def of(self, lab, y):
        lab, y = np.asarray(lab, np.int64), np.asarray(y, np.int64)
        out = np.zeros(y.size, np.int64)
        for L in np.unique(lab[lab < self.nR]):
            m = lab == L
            out[m] = self.r[int(L)][y[m]]
        m = lab >= self.nR
        if m.any() and self.s_keys.size:
            k = (lab[m] - self.nR) * self.n + y[m]
            i = np.minimum(np.searchsorted(self.s_keys, k), self.s_keys.size - 1)
            out[m] = np.where(self.s_keys[i] == k, self.s_cnt[i], 0)
        return out

    def dist(self, L):
        """Label L's nonzero degrees, sorted, and how many nodes have degree 0."""
        if L not in self._dist:
            if L < self.nR:
                v = self.r[L]
                nz = np.sort(v[v > 0])
            else:
                rel = L - self.nR
                lo, hi = np.searchsorted(self.s_keys, [rel * self.n, (rel + 1) * self.n])
                nz = np.sort(self.s_cnt[lo:hi])
            self._dist[L] = (nz, self.n - nz.size)
        return self._dist[L]


def pop_rank(deg, smp, excl):
    """Each test edge's filtered rank when every node is scored by deg_L alone (ties half)."""
    d, lab = smp["d"], smp["lab"]
    Q = d.size
    dd = deg.of(lab, d)
    gt, eq = np.zeros(Q), np.zeros(Q)
    for L in np.unique(lab):
        Ii = np.flatnonzero(lab == L)
        nz, z0 = deg.dist(int(L))
        v = dd[Ii]
        gt[Ii] = nz.size - np.searchsorted(nz, v, "right")
        eq[Ii] = np.where(v == 0, z0, np.searchsorted(nz, v, "right") - np.searchsorted(nz, v, "left"))
    en, eqq = excl
    dx = deg.of(lab[eqq], en)
    gt -= np.bincount(eqq, weights=(dx > dd[eqq]).astype(float), minlength=Q)
    eq -= np.bincount(eqq, weights=(dx == dd[eqq]).astype(float), minlength=Q)
    return 1.0 + gt + 0.5 * eq


# ── part B: logit adjustment (8.3) ──────────────────────────────────────────


class Pen:
    """tau log(1 + deg_L(y)) in float32: dense over the nodes for an R label, (sorted nodes, values) for an S label. The
    same expression gives the true tail's value (at), so a block's score and the tail's score subtract equal numbers."""

    def __init__(self, deg, labs, tau):
        self.deg, self.tau = deg, tau
        self.dense, self.sparse = {}, {}
        for L in np.unique(labs).tolist():
            if L < deg.nR:
                self.dense[L] = (tau * np.log1p(deg.r[L].astype(np.float64))).astype(np.float32)
            else:
                rel = L - deg.nR
                lo, hi = np.searchsorted(deg.s_keys, [rel * deg.n, (rel + 1) * deg.n])
                self.sparse[L] = ((deg.s_keys[lo:hi] - rel * deg.n).astype(np.int64),
                                  (tau * np.log1p(deg.s_cnt[lo:hi].astype(np.float64))).astype(np.float32))

    def at(self, lab, y):
        return (self.tau * np.log1p(self.deg.of(lab, y).astype(np.float64))).astype(np.float32)


def rank_read_la(torch, sc, X, smp, excl, op_of, pen):
    """zsem2.rank_read with the adjusted score q . x_y - tau log(1 + deg_L(y))."""
    s, d, lab = smp["s"], smp["d"], smp["lab"]
    Q = s.size
    opn = np.maximum(op_of[lab], 0)
    rows = {L: np.flatnonzero(lab == L) for L in np.unique(lab).tolist()}
    rows_t = {L: torch.from_numpy(v) for L, v in rows.items()}
    en, eq = excl
    with torch.no_grad():
        q = sc.query(sc.embed(torch.from_numpy(X.rows(s))), torch.from_numpy(opn), torch.from_numpy(lab >= smp["nR"]))
        t = (q * sc.embed(torch.from_numpy(X.rows(d)))).sum(1) - torch.from_numpy(pen.at(lab, d))
        NB = max(1024, min(65536, int(1e8 // Q)))
        greater = torch.zeros(Q, dtype=torch.float64)
        ties = torch.zeros(Q, dtype=torch.float64)
        hi_t, lo_t = (t + Z.EPS)[:, None], (t - Z.EPS)[:, None]
        for b0 in range(0, X.n, NB):
            b1 = min(X.n, b0 + NB)
            S = q @ sc.embed(torch.from_numpy(X.rows(slice(b0, b1)))).T
            for L, ri in rows_t.items():
                if L in pen.dense:
                    S[ri] -= torch.from_numpy(pen.dense[L][b0:b1])[None, :]
                else:
                    nodes, val = pen.sparse[L]
                    lo, hi = np.searchsorted(nodes, [b0, b1])
                    if hi > lo:
                        S[ri[:, None], torch.from_numpy(nodes[lo:hi] - b0)[None, :]] -= torch.from_numpy(val[lo:hi])[None, :]
            lo, hi = np.searchsorted(en, [b0, b1])
            if hi > lo:
                S[torch.from_numpy(eq[lo:hi]), torch.from_numpy(en[lo:hi] - b0)] = float("-inf")
            g = S > hi_t
            greater += g.sum(1)
            ties += ((S >= lo_t) & ~g).sum(1)
    return 1.0 + greater.numpy() + 0.5 * ties.numpy()


def la_read(torch, sc, X, smp, excl, op_of, pen, t0, what):
    rk = rank_read_la(torch, sc, X, smp, excl, op_of, pen)
    log(f"{what} + adjustment: test {Z.measures(rk)} ({time.time() - t0:.0f} s)")
    return rk


# ── part C: the hub-balanced fit (8.4) ──────────────────────────────────────


def fit_hb(torch, G, X, split, groups, op_of, ops, dev, dev_ex, cos_dev, cfg, gates, deg):
    """zsem2.fit's rotation, with each positive weighted 1 / deg_L(target) (normalised in the batch), stopped on dev
    MRR_T. The same random streams as zsem2.fit, so the edges, the order and the negatives are R1's."""
    form = "rotation"
    D = X.D
    Dp = D + cfg["extra"]
    model = Z.make_model(torch, D, Dp, len(ops), form)
    train_nodes = np.flatnonzero(split == Z.TRAIN)
    dkeys = gkeys(dev["grp"], dev["d"])
    hist, t0 = [], time.time()
    r0, t0s = Z.rank_read(torch, model, X, dev, dev_ex, op_of)
    g0 = {"max_score_diff": float(np.abs(t0s - cos_dev[1]).max()) if t0s.size else 0.0,
          "mrr_diff": float(abs((1.0 / r0).mean() - (1.0 / cos_dev[0]).mean())) if r0.size else 0.0}
    g0["verdict"] = "PASS" if g0["max_score_diff"] <= Z.G0_SCORE and g0["mrr_diff"] <= Z.G0_MRR else "FAIL"
    gates["G-0 hb"] = g0
    if g0["verdict"] != "PASS":
        stop(f"gate G-0 (hb): {g0}")
    best = {"epoch": 0, "dev_mrr_t": mrr_t(r0, dkeys), "dev_mrr": Z.measures(r0).get("mrr", 0.0),
            "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    hist.append({"epoch": 0, "dev_mrr_t": best["dev_mrr_t"], "dev_mrr": best["dev_mrr"]})
    log(f"hb: D' {Dp}, {len(ops)} operators; epoch 0 dev MRR_T {best['dev_mrr_t']} (= cosine)")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    share = cfg["epoch"] // max(len(groups), 1)
    gs_ok, since, wmin = True, 0, np.inf
    for ep in range(1, cfg["max_epochs"] + 1):
        te = time.time()
        rng = np.random.default_rng(ep)
        tr = Z.take(groups, split, Z.is_train, share, rng)
        m = tr["s"].size
        gs_ok &= bool(((split[tr["s"]] == Z.TRAIN) & (split[tr["d"]] == Z.TRAIN)).all())
        dg = deg.of(tr["lab"], tr["d"])
        if (dg < 1).any():
            stop("a training edge's target has deg_L 0")
        wall = 1.0 / dg
        wmin = min(wmin, float(wall.min()))
        perm = rng.permutation(m)
        tot, nb = 0.0, 0
        model.train()
        for b0 in range(0, m, cfg["batch"]):
            Ii = perm[b0:b0 + cfg["batch"]]
            s, d, lab = tr["s"][Ii], tr["d"][Ii], tr["lab"][Ii]
            neg = train_nodes[rng.integers(0, train_nodes.size, cfg["neg"])]
            cand = np.concatenate([d, neg])
            mask = Z.true_mask(G, s, lab, cand)
            mask[np.arange(Ii.size), np.arange(Ii.size)] = False
            xs = model.embed(torch.from_numpy(X.rows(s)))
            xc = model.embed(torch.from_numpy(X.rows(cand)))
            q = model.query(xs, torch.from_numpy(np.maximum(op_of[lab], 0)), torch.from_numpy(lab >= G.nR))
            logits = (q @ xc.T) / cfg["tau"]
            logits = logits.masked_fill(torch.from_numpy(mask), float("-inf"))
            ce = torch.nn.functional.cross_entropy(logits, torch.arange(Ii.size), reduction="none")
            w = torch.from_numpy(wall[Ii].astype(np.float32))
            loss = (ce * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        model.eval()
        r, _ = Z.rank_read(torch, model, X, dev, dev_ex, op_of)
        mt, mp = mrr_t(r, dkeys), Z.measures(r).get("mrr", 0.0)
        hist.append({"epoch": ep, "edges": int(m), "loss": round(tot / max(nb, 1), 4), "dev_mrr_t": mt, "dev_mrr": mp,
                     "seconds": round(time.time() - te, 1)})
        log(f"hb: epoch {ep}: {m:,} edges, loss {tot / max(nb, 1):.4f}, dev MRR_T {mt}, dev MRR {mp} ({time.time() - te:.0f} s)")
        if mt > best["dev_mrr_t"]:
            best = {"epoch": ep, "dev_mrr_t": mt, "dev_mrr": mp,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            since = 0
        else:
            since += 1
            if since >= cfg["patience"]:
                break
    gates["G-S hb training"] = "PASS" if gs_ok else "FAIL"
    if not gs_ok:
        stop("gate G-S (hb): a training edge touches a test or dev node")
    model.load_state_dict(best["state"])
    model.eval()
    rec = {"D_prime": Dp, "history": hist, "best_epoch": best["epoch"], "best_dev_mrr_t": best["dev_mrr_t"],
           "smallest_weight": wmin, "seconds": round(time.time() - t0, 1)}
    return model, rec, {k: v.numpy() for k, v in best["state"].items()}


# ── the r1h row ─────────────────────────────────────────────────────────────


def r1h_core(torch, G, X, cfg, r1, z, gates):
    t0 = time.time()
    split = Z.node_split(G.name, G.n)
    labs, ops, op_of, groups = Z.plan(G, split, cfg)
    names = [g.name for g in groups]
    rng = np.random.default_rng(Z.SEED)
    test = Z.take(groups, split, Z.is_test, cfg["n_test"], rng)
    dev = Z.take(groups, split, Z.is_dev, cfg["n_dev"], rng)
    for smp in (test, dev):
        smp["nR"] = G.nR
    same = {k: bool(np.array_equal(test[k], z[f"test_{k}"])) for k in ("s", "d", "lab", "grp")}
    same["read groups"] = names == list(r1["read_groups"])
    same["operators"] = [o["name"] for o in ops] == list(r1["operators"])
    if not all(same.values()):
        stop(f"gate G-R: R1's samples are not reproduced: {same}")
    test_ex, dev_ex = Z.exclusions(G, test), Z.exclusions(G, dev)
    log(f"{G.name}: G-R samples PASS: {test['s'].size:,} test, {dev['s'].size:,} dev edges ({time.time() - t0:.0f} s)")
    D = X.D
    Dp = D + cfg["extra"]
    if r1["fits"]["rotation"]["D_prime"] != Dp:
        stop("R1's rotation D' differs")
    m1 = Z.make_model(torch, D, Dp, len(ops), "rotation")
    m1.load_state_dict({k: torch.from_numpy(np.asarray(z[f"rotation.{k}"])) for k in m1.state_dict()})
    m1.eval()
    r_rot, _ = Z.rank_read(torch, m1, X, test, test_ex, op_of)
    agree = float((r_rot == z["rank_rotation"]).mean())
    dm = abs(Z.measures(r_rot)["mrr"] - Z.measures(z["rank_rotation"])["mrr"])
    gates["G-R"] = {"samples": same, "rank_agreement": round(agree, 6), "mrr_diff": round(dm, 4),
                    "verdict": "PASS" if agree >= 0.999 and dm <= 0.01 else "FAIL"}
    if gates["G-R"]["verdict"] != "PASS":
        stop(f"gate G-R: {gates['G-R']}")
    log(f"{G.name}: G-R ranks PASS ({agree:.6f} identical) ({time.time() - t0:.0f} s)")
    r_cos = np.asarray(z["rank_cosine"], float)
    deg = Deg(G, split)
    r_pop = pop_rank(deg, test, test_ex)
    pen = Pen(deg, test["lab"], cfg["tau"])
    cos = Z.Cosine()
    r_cos_c = la_read(torch, cos, X, test, test_ex, op_of, pen, t0, "cosine")
    r_rot_c = la_read(torch, m1, X, test, test_ex, op_of, pen, t0, "R1 rotation")
    cos_dev = Z.rank_read(torch, cos, X, dev, dev_ex, op_of)
    mhb, fitrec, st = fit_hb(torch, G, X, split, groups, op_of, ops, dev, dev_ex, cos_dev, cfg, gates, deg)
    r_hb, _ = Z.rank_read(torch, mhb, X, test, test_ex, op_of)
    log(f"{G.name}: hb test {Z.measures(r_hb)} ({time.time() - t0:.0f} s)")
    r_hb_c = la_read(torch, mhb, X, test, test_ex, op_of, pen, t0, "hb")
    ranks = {"cosine": r_cos, "rotation": r_rot, "popularity": r_pop, "cosine_la": r_cos_c,
             "rotation_la": r_rot_c, "hb": r_hb, "hb_la": r_hb_c}
    keys = gkeys(test["grp"], test["d"])
    read = {k: read_both(v, test, names) for k, v in ranks.items()}
    cmp_ = {"H0 rotation - cosine": ("rotation", "cosine"), "H1 hb - rotation": ("hb", "rotation"),
            "H2 rotation_la - rotation": ("rotation_la", "rotation"), "hb_la - rotation": ("hb_la", "rotation"),
            "cosine_la - cosine": ("cosine_la", "cosine"), "rotation - popularity": ("rotation", "popularity"),
            "hb - popularity": ("hb", "popularity")}
    on_t = {k: paired_t(ranks[x], ranks[y], keys) for k, (x, y) in cmp_.items()}
    on_pooled = {k: Z.paired(1.0 / ranks[x], 1.0 / ranks[y]) for k, (x, y) in cmp_.items()}
    per_group = {k: per_group_t(ranks[x], ranks[y], test, names) for k, (x, y) in cmp_.items() if k[:2] in ("H1", "H2")}
    out = {"nodes": int(G.n), "operators": [o["name"] for o in ops], "read_groups": names,
           "hubs": hub_desc(test["grp"], test["d"], names),
           "adjustment": {"tau": cfg["tau"], "labels": len(pen.dense) + len(pen.sparse)},
           "calls_mrr_t": on_t, "beside_pooled": on_pooled, "per_group_mrr_t": per_group, "read": read,
           "fit_hb": fitrec}
    arrays = {"test_s": test["s"], "test_d": test["d"], "test_lab": test["lab"], "test_grp": test["grp"],
              **{f"rank_{k}": v for k, v in ranks.items()}, **{f"hb.{k}": v for k, v in st.items()}}
    return out, arrays


def cmd_r1h(a):
    torch = Z.torch_()
    t0 = time.time()
    cfg = dict(Z.CFG)
    r1, z = load_r1(a.ds, a.arm)
    if r1 is None:
        stop(f"R1's {a.ds} {a.arm} record or npz is missing")
    if a.ds == "multihoprag":
        G, V, inputs, ge = Z.load_mhr(a.arm)
    else:
        G, V, inputs, ge = Z.load_six(a.ds, a.arm, a.host)
    gates = {"G-E": {"families": ge, "verdict": "PASS"}}
    X = Z.Vecs(V)
    log(f"{a.ds} {a.arm}: {G.n:,} nodes, rows {V.shape} ({time.time() - t0:.0f} s)")
    out, arrays = r1h_core(torch, G, X, cfg, r1, z, gates)
    rec = {"stage": "SEM2 R1h", "declared_in": DECLARED, "dataset": a.ds, "arm": a.arm, "dim": int(X.D), "config": cfg,
           "gates": gates, **out, "r1_record_sha256": sha_file(R1 / a.ds / f"{a.arm}.json"),
           "r1_npz_sha256": r1["npz_sha256"], "inputs": inputs, "threads": Z.E.THREADS,
           "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "script_sha256": sha_file(Path(__file__)), "zsem2_sha256": sha_file(Path(Z.__file__))}
    od = OUT / a.ds
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f"{a.arm}.npz", **arrays)
    rec["npz_sha256"] = sha_file(od / f"{a.arm}.npz")
    wjson(od / f"{a.arm}.json", rec)
    c = rec["calls_mrr_t"]
    log(f"{a.ds} {a.arm}: MRR_T H0 {c['H0 rotation - cosine']}; H1 {c['H1 hb - rotation']}; H2 "
        f"{c['H2 rotation_la - rotation']}; written {od / (a.arm + '.json')} ({rec['seconds']} s)")
    return 0


# ── report (8.6) ────────────────────────────────────────────────────────────


def cmd_report(a):
    rows = {}
    for d in Z.SEVEN:
        for arm in Z.ARMS:
            p = OUT / d / f"{arm}.json"
            if p.is_file():
                rows[(d, arm)] = json.loads(p.read_text(encoding="utf-8"))
    carry = {}
    for ch, key in (("HB", "H1 hb - rotation"), ("adjustment", "H2 rotation_la - rotation")):
        calls = {d: rows[(d, "e3v")]["calls_mrr_t"][key]["call"] for d in Z.SEVEN if (d, "e3v") in rows}
        missing = [d for d in Z.SEVEN if d not in calls]
        above = sum(v == "ABOVE" for v in calls.values())
        below = sum(v == "BELOW" for v in calls.values())
        verdict = "INCOMPLETE" if missing else ("CARRIED" if above >= 4 and below == 0 else "NOT_CARRIED")
        carry[ch] = {"verdict": verdict, "above": above, "below": below, "calls": calls, "missing": missing}
    h0 = {f"{d} {arm}": CALLS[r["calls_mrr_t"]["H0 rotation - cosine"]["call"]] for (d, arm), r in rows.items()}
    x = {"X5": None, "X6": None}
    if ("webqsp", "e3v") in rows:
        w = rows[("webqsp", "e3v")]
        x["X6"] = w["calls_mrr_t"]["H2 rotation_la - rotation"]["call"] == "ABOVE"
    if all((d, "e3v") in rows for d in Z.SEVEN):
        n_learn = sum(h0[f"{d} e3v"] == "LEARNS" for d in Z.SEVEN)
        w = rows[("webqsp", "e3v")]
        x["X5"] = n_learn >= 4 and (w["calls_mrr_t"]["H0 rotation - cosine"]["diff"]
                                    < 0.5 * w["beside_pooled"]["H0 rotation - cosine"]["diff"])
    rep = {"stage": "SEM2 R1h report", "declared_in": DECLARED, "carry": carry, "H0": h0, "expectations": x,
           "rows": {f"{d} {arm}": {"calls_mrr_t": r["calls_mrr_t"], "mrr_t": {k: v["mrr_t"] for k, v in r["read"].items()},
                                   "pooled": {k: v["pooled"].get("mrr") for k, v in r["read"].items()},
                                   "sha256": sha_file(OUT / d / f"{arm}.json")} for (d, arm), r in rows.items()},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": sha_file(Path(__file__))}
    wjson(OUT / "report.json", rep)
    L = ["# SEM2 R1h: hubs", "", f"Declared in {DECLARED}. Written by zsem2h.py report at {rep['utc']}. MRR_T in points "
         "(each read group and target once); intervals are paired 95% bootstraps over those groups.", "",
         "| dataset | arm | cosine | rotation | rotation+adj | HB | HB+adj | popularity | H0 | H1 (HB) | H2 (adj) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (d, arm), r in rows.items():
        m = {k: v["mrr_t"] for k, v in r["read"].items()}
        c = r["calls_mrr_t"]
        f = lambda k: f"{c[k]['diff']:+.2f} [{c[k]['lo']:+.2f}, {c[k]['hi']:+.2f}] {c[k]['call']}"  # noqa: E731
        L.append(f"| {d} | {arm} | {m['cosine']} | {m['rotation']} | {m['rotation_la']} | {m['hb']} | {m['hb_la']} | "
                 f"{m['popularity']} | {f('H0 rotation - cosine')} | {f('H1 hb - rotation')} | "
                 f"{f('H2 rotation_la - rotation')} |")
    L += ["", "**Carry (8.6):** " + "; ".join(f"{k} {v['verdict']} ({v['above']} ABOVE, {v['below']} BELOW of 7)"
                                               for k, v in carry.items()), "",
          "**Expectations:** " + "; ".join(f"{k} {'held' if v else ('failed' if v is False else 'not readable yet')}"
                                            for k, v in x.items()), ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    log(f"report: {carry}; {x}")
    return 0


# ── self-test ───────────────────────────────────────────────────────────────


def brute_la(torch, sc, X, smp, i, G, op_of, pen):
    """One query's adjusted rank by a full sort (gate G-C)."""
    s, d, L = int(smp["s"][i]), int(smp["d"][i]), int(smp["lab"][i])
    with torch.no_grad():
        allx = sc.embed(torch.from_numpy(X.rows(slice(0, X.n))))
        q = sc.query(sc.embed(torch.from_numpy(X.rows(np.asarray([s])))), torch.tensor([max(int(op_of[L]), 0)]),
                     torch.tensor([L >= G.nR]))
        raw = (allx @ q[0]).numpy()
        t = float((q[0] * allx[d]).sum()) - float(pen.at([L], [d])[0])
    sc_all = (raw - pen.at(np.full(X.n, L), np.arange(X.n))).astype(np.float64)
    drop = {s}
    if L < G.nR:
        a, b = Z.ends(G.r_keys[L], G.n)
        drop |= set(b[a == s].tolist()) | set(a[b == s].tolist())
    else:
        m = (G.s_src == s) & (G.s_rel == L - G.nR)
        drop |= set(G.s_dst[m].tolist())
    keep = np.ones(X.n, bool)
    keep[list(drop)] = False
    v = sc_all[keep]
    return 1 + float((v > t + Z.EPS).sum()) + 0.5 * float(((v >= t - Z.EPS) & (v <= t + Z.EPS)).sum())


def brute_pop(deg, smp, i, G):
    s, d, L = int(smp["s"][i]), int(smp["d"][i]), int(smp["lab"][i])
    allv = deg.of(np.full(G.n, L), np.arange(G.n)).astype(float)
    t = allv[d]
    drop = {s}
    if L < G.nR:
        a, b = Z.ends(G.r_keys[L], G.n)
        drop |= set(b[a == s].tolist()) | set(a[b == s].tolist())
    else:
        m = (G.s_src == s) & (G.s_rel == L - G.nR)
        drop |= set(G.s_dst[m].tolist())
    keep = np.ones(G.n, bool)
    keep[list(drop)] = False
    v = allv[keep]
    return 1 + float((v > t).sum()) + 0.5 * float((v == t).sum())


def selftest(a):
    import tempfile
    torch = Z.torch_()
    torch.manual_seed(0)
    # MRR_T on a hand example: group A (2 edges, ranks 1 and 3), group B (1 edge, rank 2): (2/3 + 1/2) / 2
    k = gkeys(np.array([0, 0, 0]), np.array([7, 7, 9]))
    want = round(100 * ((1 + 1 / 3) / 2 + 1 / 2) / 2, 2)
    if mrr_t(np.array([1.0, 3.0, 2.0]), k) != want:
        stop("self-test: MRR_T")
    rng = np.random.default_rng(0)
    n, D = 5000, 32
    split = Z.node_split("toyh", n)
    V = rng.standard_normal((n, D))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)  # noqa: E731
    th = rng.uniform(1.0, 2.0, D // 2) * rng.choice([-1.0, 1.0], D // 2)
    src, dst = np.arange(0, 1000), np.arange(1000, 2000)
    V[dst] = unit(Z.rotate_np(V[src], th) + 0.2 * rng.standard_normal((1000, D)) / np.sqrt(D))
    hub_src = np.arange(2000, 3500)                                     # 1,500 random sources -> two hubs (train nodes)
    hubs = np.asarray([y for y in range(3500, 3600) if split[y] == Z.TRAIN][:2])
    hub_dst = hubs[np.arange(hub_src.size) % 2]
    w = np.repeat(rng.uniform(-1, 1, D // 2), 2)
    aa, bb = np.arange(3600, 4200), np.arange(4200, 4800)
    V[bb] = unit(V[aa] * w + 0.2 * rng.standard_normal((bb.size, D)) / np.sqrt(D))
    V = V.astype(np.float32)
    rk = np.unique(np.minimum(aa, bb).astype(np.int64) * n + np.maximum(aa, bb))
    S = np.unique(np.concatenate([src.astype(np.int64) * n + dst, hub_src.astype(np.int64) * n + hub_dst]))
    G = Z.Graph("toyh", n, [("R.toy", rk)], S, ["toy_rel"])
    X = Z.Vecs(V.astype(np.float16))
    cfg = dict(Z.CFG, batch=256, neg=512, epoch=40_000, max_epochs=25, patience=4, lr=0.02, n_test=300, n_dev=200,
               extra=8, min_op=100)
    gates = {}
    out1, arr1 = Z.run(torch, G, X, cfg, ("rotation",), gates)
    with tempfile.TemporaryDirectory() as td:
        rec1 = {"read_groups": out1["read_groups"], "operators": out1["operators"], "fits": out1["fits"]}
        np.savez(Path(td) / "r1.npz", **arr1)
        z = np.load(Path(td) / "r1.npz")
        gh = {}
        out, arrays = r1h_core(torch, G, X, cfg, rec1, z, gh)
        z.close()
    # G-C and the popularity read against full sorts
    labs, ops, op_of, groups = Z.plan(G, split, cfg)
    test = Z.take(groups, split, Z.is_test, cfg["n_test"], np.random.default_rng(Z.SEED))
    test["nR"] = G.nR
    ex = Z.exclusions(G, test)
    deg = Deg(G, split)
    if int(deg.of([G.nR], [hubs[0]])[0]) != int((split[hub_src[hub_dst == hubs[0]]] == Z.TRAIN).sum()):
        stop("self-test: deg_L of a hub")
    pen = Pen(deg, test["lab"], cfg["tau"])
    cos = Z.Cosine()
    rc = rank_read_la(torch, cos, X, test, ex, op_of, pen)
    chk = [abs(brute_la(torch, cos, X, test, i, G, op_of, pen) - rc[i]) for i in range(0, test["s"].size, 20)]
    rp = pop_rank(deg, test, ex)
    chp = [abs(brute_pop(deg, test, i, G) - rp[i]) for i in range(0, test["s"].size, 20)]
    if max(chk) > 0 or max(chp) > 0:
        stop(f"self-test: adjusted ranks off a full sort by {max(chk)}, popularity by {max(chp)}")
    gh["G-C"] = "PASS"
    c = out["calls_mrr_t"]
    rd = out["read"]
    log(f"self-test: MRR_T cosine {rd['cosine']['mrr_t']}, rotation {rd['rotation']['mrr_t']}, rotation+adj "
        f"{rd['rotation_la']['mrr_t']}, hb {rd['hb']['mrr_t']}, hb+adj {rd['hb_la']['mrr_t']}, popularity "
        f"{rd['popularity']['mrr_t']}; pooled rotation {rd['rotation']['pooled']['mrr']}, hb {rd['hb']['pooled']['mrr']}, "
        f"popularity {rd['popularity']['pooled']['mrr']}; per group {json.dumps(rd['hb']['per_group'])}")
    log(f"self-test: H0 {c['H0 rotation - cosine']}; H1 {c['H1 hb - rotation']}; H2 {c['H2 rotation_la - rotation']}; "
        f"hubs {out['hubs']}; gates {gh}")
    ok = (all(v == "PASS" or (isinstance(v, dict) and v.get("verdict") == "PASS") for v in gh.values())
          and c["H0 rotation - cosine"]["call"] == "ABOVE")
    if not ok:
        stop("self-test FAILED")
    log(f"self-test PASS: G-R, G-0, G-S and G-C; {len(chk)} adjusted and {len(chp)} popularity ranks equal a full sort")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("selftest", "hubread", "r1h", "report"))
    p.add_argument("--ds", choices=Z.SEVEN)
    p.add_argument("--arm", choices=Z.ARMS)
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return selftest(a)
    if a.cmd == "hubread":
        return cmd_hubread(a)
    if a.cmd == "report":
        return cmd_report(a)
    if not (a.ds and a.arm):
        stop("r1h needs --ds and --arm")
    return cmd_r1h(a)


if __name__ == "__main__":
    sys.exit(main())
