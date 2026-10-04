"""Design look (untracked; not a result and not filed): qd_gnn6.py, pinned and unchanged, with label-free transfer
controls for its QD arms and a few-label residual-scale calibration read (anchor_kgrid's 'off'-including grid carried
to the QD residual).

Arm-name options (QD arms only), appended to the qd_gnn2 / qd_gnn3 name in any order:
  -g1        gamma fixed at 1: each layer's aggregate is the mean over active in-neighbours (no growth with degree)
  -tb<T>     each node's residual bounded: s = z + t tanh((s0 - z) / t), t = T / 10 (z is z-scored per row)
  -rs        each row's residual softly standardised: r <- c (r - mean r) / sqrt(var r + 1) over the row's pool, with
             c = exp(log_c) learned (init 1): a row whose residual spread is below 1 keeps it, a wider one is pulled to c
  -cap<K>    every node keeps at most K in-edges (qd_tdiag.cap_row: the K of largest w, ties to the earlier edge), in
             training and in every read
  -ed<P>     training-time edge dropout: each batch row draws u ~ U(0, P/100) and drops each of its edges with
             probability u (reads and the select reads see every edge)
With no option an arm is qd_gnn6's draw for draw (the same classes and calls); g1 / tb / rs build QD7 (qd_gnn.QD's
forward, line for line, with the three switches), cap / ed act on the arm's pack.

--kalpha N1,N2,...  (with --kalpha-draws D, default 3) after the ID read of every graph with a pool disjoint from its
  read rows (x1's half A, minus any training row; webqsp, read on its whole select carve, has none): pick alpha in
  ALPHAS (0 = the twin exactly ... 1 = the fitted model) on N labelled pool rows (numpy seed READ_SEED + 1000 N + draw)
  by the mean of recall@5, full_coverage@5 and hit@1 (the loop's np select score), ties to the smaller alpha, under
  np and under the run's margin (mp), and read s = z + alpha (s0 - z) on the read rows. Per N: the alphas picked, the
  mean of draws' read (per-row metrics averaged over the draws) and its paired differences against alpha 0 ('k - a0':
  z, the twin's read, or the GNN's under --base gnn) and alpha 1 ('k - a1': the ID read); per graph the read at every
  alpha and the oracle alpha on the read rows.
  The calibration is a per-graph scalar from labels: it is reported as its own read, never as the one checkpoint.

    python outputs/mp_unified/qd_gnn7.py --selftest
    python outputs/mp_unified/qd_gnn7.py --train 2wiki=x4 --read hotpotqa,musique,metaqa,webqsp \
        --arms QD-T0-L2,QD-T0-L2-g1,QD-T0-L2-rs,QD-T0-L2-cap16 --rule both --kalpha 64,256 \
        --out outputs/mp_unified/qd/t7-2w.json
"""
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn6 as Q6x  # noqa: E402  (imports qd_gnn5 ... qd_gnn, which sets the BLAS thread counts before numpy loads)
import qd_tdiag as TD  # noqa: E402  (cap_row only)

import numpy as np  # noqa: E402
import torch  # noqa: E402

Q5 = Q6x.Q5
Q4, Q3, Q2, QG = Q5.Q4, Q5.Q3, Q5.Q2, Q5.QG
SHAS = {**Q6x.SHAS, "qd_tdiag": QG.sha(TD.__file__), "qd_gnn7": QG.sha(__file__)}   # at import: what ran
OPT = re.compile(r"-(g1|rs|tb\d+|cap\d+|ed\d+)$")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
KA = {"Ns": [], "draws": 3}
STATE7 = {"part": None, "trains": None, "pool": {}, "B": {}}
KAR = {}      # arm name -> one entry per (seed, graph) ID read
KROWS = {}    # per-row metrics of the kalpha reads (saved beside the run's npz)
log = QG.log


# ── arm options ──────────────────────────────────────────────────────────────


def split_opts(nm):
    """(the qd_gnn6 name, the qd_gnn7 options) of an arm name."""
    opts, base = {}, nm
    while True:
        m = OPT.search(base)
        if not m:
            break
        tok = m.group(1)
        if tok in ("g1", "rs"):
            key, val = tok, True
        else:
            key = re.match(r"[a-z]+", tok).group(0)
            n = int(tok[len(key):])
            val = {"tb": n / 10.0, "cap": n, "ed": n / 100.0}[key]
            if (key == "tb" and n <= 0) or (key == "ed" and not 0 < n < 100) or (key == "cap" and n < 1):
                raise SystemExit(f"{nm}: -{tok} out of range (tb > 0, 0 < ed < 100, cap >= 1)")
        if key in opts:
            raise SystemExit(f"{nm}: -{tok} given twice")
        opts[key] = val
        base = base[:m.start()]
    if opts and not base.startswith("QD-"):
        raise SystemExit(f"{nm}: the qd_gnn7 options are for QD arms only")
    return base, opts


_parse6 = Q6x.parse_arms


def parse_arms(spec, names, rule, AU, AG, G3):
    full = spec.split(",")
    if len(set(full)) != len(full):
        raise SystemExit(f"--arms {spec}: each arm once")
    split = [split_opts(nm) for nm in full]
    base = _parse6(",".join(dict.fromkeys(b for b, _o in split)), names, rule, AU, AG, G3)
    out = {}
    for nm, (b, o) in zip(full, split):
        out[nm] = {**base[b], "t7": o} if o else base[b]
        Q5.NAMES[id(out[nm])] = nm
    return out


# ── the model ────────────────────────────────────────────────────────────────


class QD7(QG.QD):
    """qd_gnn.QD with three switches: g1 (gamma 1), tb (bounded residual), rs (softly standardised row residual)."""

    def __init__(self, kind, d=64, layers=3, n_id=0, g1=False, tb=None, rs=False):
        super().__init__(kind, d, layers, n_id)
        self.g1, self.tb, self.rs = bool(g1), tb, bool(rs)
        if self.rs:
            self.log_c = torch.nn.Parameter(torch.zeros(()))

    def forward(self, P):
        rms = QG.rms
        qn = self.lnq(self.Wq(P.qemb))                                  # (B, d)
        h0 = qn[P.seed_row] * (1.0 + self.E_b[P.seed_b]) + P.z[P.seed_idx, None] * self.e_z
        h = torch.zeros(P.N, self.dim).index_add(0, P.seed_idx, h0)
        r = self.relations(P)
        gamma = torch.sigmoid(self.gamma_raw)
        fam = P.fam.clamp(max=2)
        for l in range(self.L):
            w = self.R[l](r) * (1.0 + torch.tanh(self.G[l](qn)))[P.e_row]
            logit = (self.A[l](qn)[P.e_row] * r).sum(-1) + self.c_fam[l, fam] * P.w + self.b_gate[l]
            g = 2.0 * torch.sigmoid(logit)
            hs = rms(h)
            m = hs[P.src] * w * g[:, None]
            active = (h != 0).any(-1).to(m.dtype)
            n_in = torch.zeros(P.N).index_add(0, P.dst, active[P.src])
            if self.g1:
                agg = torch.zeros(P.N, self.dim).index_add(0, P.dst, m) / (1.0 + n_in)[:, None]
            else:
                agg = torch.zeros(P.N, self.dim).index_add(0, P.dst, m) / (1.0 + n_in)[:, None].pow(gamma)
            h = h + torch.relu(self.W[l](agg))
        hn = rms(h)
        reached = (h != 0).any(-1).to(h.dtype)
        f = torch.cat([hn, hn * qn[P.node_row], P.z[:, None], reached[:, None]], 1)
        res = self.wo(torch.relu(self.Wr(f))).squeeze(-1)
        if self.rs:
            cnt = torch.zeros(P.B).index_add(0, P.node_row, torch.ones_like(res))
            mu = torch.zeros(P.B).index_add(0, P.node_row, res) / cnt
            dv = res - mu[P.node_row]
            var = torch.zeros(P.B).index_add(0, P.node_row, dv * dv) / cnt
            res = torch.exp(self.log_c) * dv * torch.rsqrt(var[P.node_row] + 1.0)
        if self.tb is not None:
            res = self.tb * torch.tanh(res / self.tb)
        return P.z + res


MODEL_OPTS = ("g1", "tb", "rs")


def edge_drop(P, rng, top):
    """P with each batch row's edges dropped at its own rate u ~ U(0, top) (in place; returns how many went)."""
    u = rng.random(P.B) * top
    keep = rng.random(P.src.numel()) >= u[P.e_row.numpy()]
    kt = torch.from_numpy(keep)
    for k in ("src", "dst", "e_row", "d", "a", "fam", "w"):
        setattr(P, k, getattr(P, k)[kt])
    return int((~keep).sum())


# ── the arms ─────────────────────────────────────────────────────────────────


class QDArm7(Q5.QDArm5):
    """qd_gnn5's arms with the qd_gnn7 options on QD arms; qd_gnn2.run builds every 'qd'-family arm as QG.QDArm(...)."""

    def __new__(cls, sp, *args, **kwargs):
        if cls is QDArm7:
            if sp.get("w4"):
                return object.__new__(W4Arm7)
            if sp.get("w3"):
                return object.__new__(W3Arm7)
        return object.__new__(cls)

    def __init__(self, sp, Q, TOK, G, z_of, A16, n_id):
        super().__init__(sp, Q, TOK, G, z_of, A16, n_id)
        self.own, self.capped, self.rng_ed = (Q, TOK), {}, None

    def make(self):
        o = self.sp.get("t7")
        if not o:
            return super().make()
        if any(o.get(k) for k in MODEL_OPTS):
            model = QD7(self.sp["kind"], self.sp["d"], self.sp["L"], self.n_id, g1=o.get("g1", False), tb=o.get("tb"),
                        rs=o.get("rs", False))
        else:
            model = QG.QD(self.sp["kind"], self.sp["d"], self.sp["L"], self.n_id)
        # qd_gnn3.QDArmDrop.make's label-dropout generator, drawn as it draws it, then the edge-dropout generator
        if self.p_row > 0 or self.p_edge > 0:
            self.rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        self.rng_ed = (np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item())) if o.get("ed") else None)
        return model

    def rows_for(self, rows):
        """(Q, TOK) to pack rows from: the arm's own, or with -cap<K> their capped copies (cached for the arm's own
        lists; a swapped view, as qd_gnn6's hold reads make, is capped on the fly)."""
        K = (self.sp.get("t7") or {}).get("cap")
        if K is None:
            return self.Q, self.TOK
        Qd, Td = {}, {}
        cache = self.capped if (self.Q is self.own[0] and self.TOK is self.own[1]) else None
        for i in rows:
            if cache is not None and i in cache:
                Qd[i], Td[i] = cache[i]
                continue
            nq, nt = TD.cap_row(self.Q[i], self.TOK[i], K)
            Qd[i], Td[i] = nq, nt
            if cache is not None:
                cache[i] = (nq, nt)
        return Qd, Td

    def forward(self, model, g, rows):
        o = self.sp.get("t7")
        if not o:
            return super().forward(model, g, rows)
        Qp, Tp = self.rows_for(rows)
        P = QG.pack_qd(rows, Qp, Tp, self.z_of, self.sp["K"], self.nr)
        if model.training and self.rng is not None:     # qd_gnn3.QDArmDrop.forward's label dropout, draw for draw
            drop = self.rng.random(P.B) < self.p_row
            de = drop[P.e_row.numpy()] | (self.rng.random(P.a.numel()) < self.p_edge)
            P.a = torch.where(torch.from_numpy(de), torch.full_like(P.a, -1), P.a)
        if model.training and self.rng_ed is not None:
            edge_drop(P, self.rng_ed, o["ed"])
        return QG.padded(P, model(P))


class W3Arm7(QDArm7, Q5.W3Arm5):
    """qd_gnn5.W3Arm5, unchanged (no qd_gnn7 option reaches it), as a QDArm7."""


class W4Arm7(QDArm7, Q5.W4Arm):
    """qd_gnn5.W4Arm, unchanged (no qd_gnn7 option reaches it), as a QDArm7."""


# ── the few-label residual-scale read ────────────────────────────────────────


_load6 = Q6x.load_all6


def load_all7(trains, reads, max_len, t0, AU):
    Q, part, G = _load6(trains, reads, max_len, t0, AU)
    STATE7.update(part=part, trains=trains, pool={}, B={})
    for g in list(trains) + list(reads):
        lk = Q2.READ[g]
        if g in Q2.WHOLE:
            STATE7["B"][g] = list(part[(g, lk)])
            STATE7["pool"][g] = []
            continue
        STATE7["B"][g] = part[(g, lk)][1::2]
        tr = {i for c in trains.get(g, []) for i in part[(g, c)]}
        STATE7["pool"][g] = [i for i in part[(g, lk)][0::2] if i not in tr]
    if KA["Ns"]:
        log("kalpha pools: " + ", ".join(f"{g} {len(v)}" for g, v in STATE7["pool"].items()))
    return Q, part, G


def best_alpha(sc):
    best, bv = None, -math.inf
    for x in ALPHAS:
        if sc[x] > bv + 1e-12:
            best, bv = x, sc[x]
    return best


def blend(SZ, x):
    """(z + x (s - z), z) per row; x 1 is s itself and x 0 is z itself (float rounding would otherwise move ties)."""
    if x == 1.0:
        return SZ
    if x == 0.0:
        return [(z, z) for _s, z in SZ]
    return [(z + np.float32(x) * (s - z), z) for s, z in SZ]


def kalpha_reads(arm, model, g, rows, SZ):
    AU = sys.modules["anchor_univ"]
    AG, AW3 = AU.AG, AU.AW3
    mo = arm.A16.metrics_of
    name = Q5.NAMES.get(id(arm.sp), "?")
    A = STATE7["pool"][g]
    SA = Q5._score_rows(arm, model, g, A)
    R = AG.Reader(arm.Q, rows, QG.READ_SEED)
    base = R.base()
    margin = Q6x.STATE["margin"]
    k = sum(1 for e in KAR.get(name, []) if e["graph"] == g)
    ent = {"graph": g, "read": k, "pool_rows": len(A), "margin": str(margin), "base_z": Q4.BASE["z"], "rules": {}}
    for rl, mg in (("np", None), ("mp", margin)):
        MA = {x: Q2.metrics_at(blend(SA, x), A, arm.Q, mg, mo) for x in ALPHAS}
        MB = {x: Q2.metrics_at(blend(SZ, x), rows, arm.Q, mg, mo) for x in ALPHAS}
        e = {"by_alpha": {str(x): Q2.with_abs(R.record(MB[x], by_type=False), base) for x in ALPHAS},
             "oracle": str(best_alpha({x: float(MB[x][:, :3].mean()) for x in ALPHAS}))}
        for N in KA["Ns"]:
            if N > len(A):
                e[str(N)] = {"skipped": f"pool has {len(A)} rows"}
                continue
            picks = []
            for d in range(KA["draws"]):
                rng = np.random.default_rng(QG.READ_SEED + 1000 * N + d)
                sub = rng.choice(len(A), size=N, replace=False)
                picks.append(best_alpha({x: float(MA[x][sub, :3].mean()) for x in ALPHAS}))
            m = np.mean([MB[x] for x in picks], axis=0)
            KROWS[f"{name}|{k}|{g}|{rl}|N{N}"] = m
            e[str(N)] = {"alphas": [str(x) for x in picks],
                         "mean_of_draws": Q2.with_abs(R.record(m, by_type=False), base),
                         "k - a0": AW3.boot_pair(m - MB[0.0], R.W), "k - a1": AW3.boot_pair(m - MB[1.0], R.W)}
        ent["rules"][rl] = e
    # alpha 0 is z: the twin's own read (rho 0) under --base twin, the GNN's (rho 1) under --base gnn
    z0 = ent["rules"]["np"]["by_alpha"]["0.0"]["rho (R@5, FC@5, hit@1)"]
    want = 0.0 if Q4.BASE["z"] == "twin" else 1.0
    ent["alpha0_is_base"] = all(x is not None and abs(x - want) < 1e-9 for x in z0)
    KAR.setdefault(name, []).append(ent)
    log(f"kalpha {name} {g}: pool {len(A)}, alpha0 rho {z0}, oracle np {ent['rules']['np']['oracle']}; " + "; ".join(
        f"N{N} np {ent['rules']['np'][str(N)].get('alphas')} rho {ent['rules']['np'][str(N)].get('mean_of_draws', {}).get('rho (R@5, FC@5, hit@1)')}"
        for N in KA["Ns"]))


_score6 = Q6x.score_rows


def score_rows(arm, model, g, rows, TY=None):
    SZ = _score6(arm, model, g, rows, TY)
    if (KA["Ns"] and Q5.PHASE["now"] == "read" and TY is None and not getattr(arm, "nr", False)
            and not getattr(arm, "shuf", False) and STATE7["pool"].get(g) and sorted(rows) == sorted(STATE7["B"][g])):
        try:
            kalpha_reads(arm, model, g, rows, SZ)
        except Exception as ex:  # noqa: BLE001 -- the run's own reads stand; the failure is recorded and logged
            KAR.setdefault("_failed", []).append({"arm": Q5.NAMES.get(id(arm.sp), "?"), "graph": g, "error": repr(ex)})
            log(f"kalpha reads failed: {ex!r}")
    return SZ


_bind6 = Q6x.bind


def bind():
    _bind6()
    QG.QDArm = QDArm7
    Q2.parse_arms = parse_arms
    Q2.load_all = load_all7
    Q2.score_rows = score_rows


def finish(out):
    res = json.loads(out.read_text(encoding="utf-8"))
    res["look"] = "qd_gnn7"
    res["pins"]["qd_gnn5"] = SHAS["qd_gnn5"]
    res["pins"]["qd_gnn6"] = SHAS["qd_gnn6"]
    res["pins"]["qd_tdiag"] = SHAS["qd_tdiag"]
    res["qd_gnn7_sha256"] = SHAS["qd_gnn7"]
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    import tempfile
    sys.path.insert(0, str(QG.AU_DIR))
    sys.path.insert(0, str(QG.KB_DIR))
    import anchor_univ as AU
    A16 = AU.A16
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    K = 8
    # names
    assert split_opts("QD-TXT-L2-dr25e10-g1-tb20") == ("QD-TXT-L2-dr25e10", {"tb": 2.0, "g1": True})
    assert split_opts("QD-T0-L3-cap16-ed50-rs") == ("QD-T0-L3", {"rs": True, "ed": 0.5, "cap": 16})
    assert split_opts("QD-TXT-L2-d16-K8") == ("QD-TXT-L2-d16-K8", {})
    assert split_opts("W3-TXT-L2-dr25e10") == ("W3-TXT-L2-dr25e10", {})
    for bad in ("QD-T0-L2-g1-g1", "W3-T0-L2-g1", "QD-T0-L2-ed0", "QD-T0-L2-tb0", "QD-T0-L2-cap0", "W:T0-rs"):
        try:
            split_opts(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
    print("selftest: arm-name options parse, repeat / range / non-QD uses are refused")
    # QD7 with every switch off is qd_gnn.QD's forward bit for bit, values and gradients
    Q = QG.synthetic_rows(24, rng, K)
    TOK = [QG.edge_labels(q) for q in Q]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    phi = rng.standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32)
    rows = list(range(24))
    P = QG.pack_qd(rows, Q, TOK, z_of, K)
    for kind in ("T0", "TXT", "ID"):
        torch.manual_seed(1)
        m0 = QG.QD(kind, 16, 2, K)
        torch.manual_seed(1)
        m7 = QD7(kind, 16, 2, K)
        for m in (m0, m7):
            m.set_phi(phi, K)
        with torch.no_grad():
            torch.manual_seed(2)
            m0.wo.weight.normal_(0, 0.5)
            m0.gamma_raw.fill_(-0.7)
        m7.load_state_dict(m0.state_dict())
        s0, s7 = m0(P), m7(P)
        assert torch.equal(s0, s7), kind
        s0.sum().backward()
        s7.sum().backward()
        for (n0, p0), (n7, p7) in zip(m0.named_parameters(), m7.named_parameters()):
            assert n0 == n7 and (p0.grad is None) == (p7.grad is None), n0
            assert p0.grad is None or torch.equal(p0.grad, p7.grad), n0
        # g1 is gamma -> 1
        m1 = QD7(kind, 16, 2, K, g1=True)
        m1.load_state_dict(m0.state_dict())
        m1.set_phi(phi, K)
        mref = QG.QD(kind, 16, 2, K)
        mref.load_state_dict(m0.state_dict())
        mref.set_phi(phi, K)
        with torch.no_grad():
            mref.gamma_raw.fill_(40.0)
            assert torch.allclose(m1(P), mref(P), atol=1e-6), kind
            # tb bounds every node's shift; rs pulls each row's residual spread below c
            mt = QD7(kind, 16, 2, K, tb=0.5)
            mt.load_state_dict(m0.state_dict())
            mt.set_phi(phi, K)
            d = (mt(P) - P.z).abs()
            assert float(d.max()) <= 0.5 + 1e-6 and float(d.max()) > 0
            mr = QD7(kind, 16, 2, K, rs=True)
            sd = dict(m0.state_dict())
            sd["log_c"] = torch.tensor(math.log(0.3))
            mr.load_state_dict(sd)
            mr.set_phi(phi, K)
            dr = mr(P) - P.z
            for b in range(P.B):
                v = dr[P.node_row == b]
                assert float(v.std(unbiased=False)) < 0.3 + 1e-6
            assert abs(float(dr.mean())) < 1e-5
    print("selftest: QD7 with no switch is QD bit for bit (values, gradients); g1 = gamma 1; tb and rs bound the residual")
    # the arms: no option = qd_gnn5's arm; cap packs at most K in-edges per node; ed drops edges only in training
    G = {"g": {"phi": phi, "kind": "passage"}}
    bind()
    sp = Q2.parse_arms("QD-TXT-L2-d16-K8", ["g"], "np", AU, AU.AG, AU.G3)["QD-TXT-L2-d16-K8"]
    armA, armB = Q5.QDArm5(sp, Q, TOK, G, z_of, A16, 0), QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
    assert type(armB) is QDArm7
    torch.manual_seed(4)
    mA = armA.make()
    torch.manual_seed(4)
    mB = armB.make()
    assert type(mA) is type(mB) is QG.QD
    for (a_, b_) in zip(mA.parameters(), mB.parameters()):
        assert torch.equal(a_, b_)
    with torch.no_grad():
        mA.wo.weight.normal_(0, 0.5)
    mB.load_state_dict(mA.state_dict())
    armA.prepare(mA, "g")
    armB.prepare(mB, "g")
    mA.eval()
    mB.eval()
    with torch.no_grad():
        assert torch.equal(armA.forward(mA, "g", rows)[0], armB.forward(mB, "g", rows)[0])
    for nm in ("W3-TXT-L2-d16-K8", "W4-T0-L2-d16"):
        spw = Q2.parse_arms(nm, ["g"], "np", AU, AU.AG, AU.G3)[nm]
        aw = QG.QDArm(spw, Q, TOK, G, z_of, A16, 0)
        assert isinstance(aw, QDArm7) and isinstance(aw, Q5.W3Arm5) and type(aw) is (W4Arm7 if spw.get("w4") else W3Arm7)
        a5 = Q5.QDArm5(spw, Q, TOK, G, z_of, A16, 0)
        torch.manual_seed(5)
        m5 = a5.make()
        torch.manual_seed(5)
        m7 = aw.make()
        assert type(m5) is type(m7)
        m7.load_state_dict(m5.state_dict())
        a5.prepare(m5, "g")
        aw.prepare(m7, "g")
        m5.eval()
        m7.eval()
        with torch.no_grad():
            assert torch.equal(a5.forward(m5, "g", rows)[0], aw.forward(m7, "g", rows)[0]), nm
    print("selftest: with no option QDArm7 and its W3 / W4 arms score as qd_gnn5's arms")
    spc = Q2.parse_arms("QD-T0-L2-d16-cap2-ed60", ["g"], "np", AU, AU.AG, AU.G3)["QD-T0-L2-d16-cap2-ed60"]
    assert spc["t7"] == {"ed": 0.6, "cap": 2}
    ac = QG.QDArm(spc, Q, TOK, G, z_of, A16, 0)
    torch.manual_seed(6)
    mc = ac.make()
    assert type(mc) is QG.QD and ac.rng_ed is not None
    Qp, Tp = ac.rows_for(rows)
    for i in rows:
        indeg = np.bincount(Qp[i]["v"], minlength=Qp[i]["n"])
        assert indeg.max() <= 2 and Qp[i]["u"].size <= Q[i]["u"].size
        assert Qp[i]["w2_f"].size == int((Qp[i]["fam"] == 0).sum()) and Tp[i][0].size == Qp[i]["u"].size
    assert ac.rows_for(rows)[0][0] is Qp[0], "the capped rows are not cached"
    Pc = QG.pack_qd(rows, Qp, Tp, z_of, 0)
    E = Pc.src.numel()
    mc.train()
    calls = []
    real_padded = QG.padded
    QG.padded = lambda P_, s_: (calls.append(P_.src.numel()), real_padded(P_, s_))[1]
    try:
        ac.forward(mc, "g", rows)
        mc.eval()
        ac.forward(mc, "g", rows)
    finally:
        QG.padded = real_padded
    assert calls[0] < E and calls[1] == E, (calls, E)
    print(f"selftest: cap2 keeps <= 2 in-edges per node (cached); ed60 drops {E - calls[0]} of {E} edges in training, none in reads")
    # the full loop through qd_gnn2.run on a synthetic KB, with --kalpha and with --hold beside it
    names = [f"rel {j}" for j in range(K)]
    real = Q4._load_all

    def fake(trains, reads, max_len, t0, AU_):
        Qf = Q6x.synthetic_kb(80, np.random.default_rng(5), K)
        part = {("metaqa", "x1"): list(range(0, 40)), ("metaqa", "select"): list(range(40, 52)),
                ("metaqa", "fit"): list(range(52, 80))}
        Gf = {"metaqa": {"kind": "kb", "role": "train", "phi": phi, "phi_sha": "synthetic", "n_rel": K, "vocab": names,
                         "info": {"pruned_loader": {"synthetic": True}}}}
        return Qf, part, Gf

    def call_main(argv):
        keep = (Q2.run, Q5.bind, sys.argv)
        sys.argv = ["qd_gnn7.py"] + argv
        try:
            main()
        finally:
            Q2.run, Q5.bind, sys.argv = keep

    Q4._load_all = fake
    tmp = Path(tempfile.mkdtemp(prefix="qd7_"))
    try:
        for hold in (None, "metaqa:drop:rel 1;rel 5"):
            Q6x.HOLD.clear()
            for key in ("full", "tok", "held_rows"):
                Q6x.STATE[key] = {}
            Q6x.STATE["out"] = None
            Q6x.RES.clear()
            Q6x.ROWS.clear()
            KAR.clear()
            KROWS.clear()
            KA["Ns"] = []
            out = tmp / f"k_{'hold' if hold else 'plain'}.json"
            arms = "QD-T0-L2-d16,QD-T0-L2-d16-g1-tb20,QD-TXT-L2-d16-K8-rs-cap3-ed40,W3-T0-L2-d16"
            argv = ["--train", "metaqa=fit", "--arms", arms, "--rule", "both", "--epochs", "2", "--out", str(out),
                    "--kalpha", "4,8,30", "--kalpha-draws", "2"]
            call_main(argv + (["--hold", hold] if hold else []))
            res = json.loads(out.read_text(encoding="utf-8"))
            assert res["look"] == "qd_gnn7" and res["qd_gnn7_sha256"] == SHAS["qd_gnn7"]
            assert "_failed" not in KAR, KAR.get("_failed")
            assert set(KAR) == set(arms.split(",")), set(KAR)
            for nm, ents in KAR.items():
                assert len(ents) == 1 and ents[0]["graph"] == "metaqa" and ents[0]["pool_rows"] == 20
                e = ents[0]["rules"]["np"]
                assert e["30"] == {"skipped": "pool has 20 rows"} and len(e["4"]["alphas"]) == 2
                # alpha 1 reads the run's own ID read; alpha 0 reads z itself
                run_np = res["results"][nm]["seeds_read"]["0"]["reads"]["metaqa"]["ID/np"]["fit"]
                assert e["by_alpha"]["1.0"]["fit"] == run_np, (nm, e["by_alpha"]["1.0"]["fit"], run_np)
            assert res["kalpha"]["Ns"] == [4, 8, 30] and res["kalpha"]["draws"] == 2
            assert res["results"]["QD-T0-L2-d16-g1-tb20"]["spec"]["t7"] == {"tb": 2.0, "g1": True}
            # the saved rs model loads back into QD7 and scores as the run's model
            ck = torch.load(out.parent / (out.stem + "_models") / "a2_s0.pt", weights_only=False)
            assert "log_c" in ck["state_dict"] and ck["spec"]["t7"]["rs"]
            if hold:
                assert res["hold"] and Q6x.RES and "_failed" not in Q6x.RES
                assert set(Q6x.RES) == set(arms.split(",")), set(Q6x.RES)
            assert Path(str(out.with_suffix("")) + "_kalpha.npz").exists()
            print(f"selftest e2e ({'hold' if hold else 'plain'}): kalpha reads for {sorted(KAR)}; alpha 1 = the run's ID read")
    finally:
        Q4._load_all = real
        Q6x.HOLD.clear()
        KA["Ns"] = []
    print("selftest: all checks passed")


def main():
    bind()
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--hold" in sys.argv:
        k = sys.argv.index("--hold")
        Q6x.HOLD.update(Q6x.parse_hold(sys.argv[k + 1]))
        del sys.argv[k:k + 2]
    if "--kalpha-draws" in sys.argv:
        k = sys.argv.index("--kalpha-draws")
        KA["draws"] = int(sys.argv[k + 1])
        if KA["draws"] < 1:
            raise SystemExit("--kalpha-draws >= 1")
        del sys.argv[k:k + 2]
    if "--kalpha" in sys.argv:
        k = sys.argv.index("--kalpha")
        KA["Ns"] = [int(x) for x in sys.argv[k + 1].split(",")]
        if not KA["Ns"] or min(KA["Ns"]) < 1 or len(set(KA["Ns"])) != len(KA["Ns"]):
            raise SystemExit("--kalpha N1,N2,...: distinct positive row counts")
        del sys.argv[k:k + 2]
    _run2 = Q2.run

    def run(a):
        Q6x.STATE["out"] = Path(a.out)
        _run2(a)
        out = Path(a.out)
        res = json.loads(out.read_text(encoding="utf-8"))
        res["hold"] = Q6x.hold_record() if Q6x.HOLD else None
        res["kalpha"] = ({"Ns": KA["Ns"], "draws": KA["draws"], "alphas": list(ALPHAS),
                          "pools": {g: len(v) for g, v in STATE7["pool"].items()}, "reads": KAR} if KA["Ns"] else None)
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        if Q6x.ROWS:
            np.savez(str(out.with_suffix("")) + "_hold.npz", **{k.replace("|", "__"): v for k, v in Q6x.ROWS.items()})
        if KROWS:
            np.savez(str(out.with_suffix("")) + "_kalpha.npz", **{k.replace("|", "__"): v for k, v in KROWS.items()})

    Q2.run = run
    Q5.bind = bind    # qd_gnn5.main binds first (its --cap-len / --base, its output fields), then runs Q2.main
    Q5.main()
    if Q6x.STATE["out"] is not None:
        finish(Q6x.STATE["out"])


if __name__ == "__main__":
    main()
