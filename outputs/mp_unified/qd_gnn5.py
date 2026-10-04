"""Design look (untracked; not a result and not filed): qd_gnn4.py, pinned and unchanged, with
(1) one more non-message-passing arm family, W4: W3 with a per-node readout over its channel boosts, and
(2) residual diagnostics on every scoring pass (qd_gnn2.score_rows): how far, and which way, each arm moves the twin's
z on each graph it reads -- the first look at why a residual fitted on one graph transfers badly to another.

W4-<T0|TXT>[-L<1..3>][-d<dim>][-K<labels>][-dr<R>e<E>] (W3's grammar) reads W3's compiled walk types, reach sets, chain
embeddings and type posterior p unchanged and splits W3's boost by the channel c = (bucket b, walk length k) of the
types that contain the node:
    B_c(v) = sum over the types t of channel c with v in R_t of p_t |R_t|^-beta        (W3's boost = sum_c B_c(v))
    f_v    = [z_v, B_1(v) .. B_2L(v), log(1 + n_v), 1{n_v > 0}]      n_v = the number of the row's types containing v
    s_v    = z_v + kappa sum_c B_c(v) + w_o . relu(W_r f_v + G_r qn)
with w_o zero at init, so an unfitted W4 scores as the unfitted W3 drawn from the same generator state does (the
selftest checks it). f_v reads only v's own z and the fixed sets that contain v; no node's score or state reaches
another node, so W4 is non-message-passing on W3's terms (its readout is an MLP over node-local features, as the
twin's own head is).

Diagnostics (residual_diag in the output, one list per arm in call order): every call of qd_gnn2.score_rows -- the
select carve's margin pass (phase select), then the reads (phase read: ID, NR, SHUF) -- records the graph, the view,
the rows and, as means over rows: ratio = std(s - z) / std(z) over the pool, moved = rank-1 differs from z's, big =
the share of pool nodes moved by more than one z unit, top5 = |s - z| at the top 5 by s, and lift = mean(s - z over
the golds) - mean(s - z over the rest), whose sign says whether the residual points at the golds on that graph.
A diagnostic never stops a run: a failure is recorded in place of the entry.

    python outputs/mp_unified/qd_gnn5.py --selftest
    python outputs/mp_unified/qd_gnn5.py --train 2wiki=x4 --arms W3-T0-L2,W4-T0-L2 --rule both \
        --out outputs/mp_unified/qd/x.json
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn4 as Q4  # noqa: E402  (imports qd_gnn3, qd_gnn2, qd_six and qd_gnn, which sets the BLAS thread counts)

import numpy as np  # noqa: E402
import torch  # noqa: E402

Q3, Q2, QG = Q4.Q3, Q4.Q2, Q4.QG
W4RE = re.compile(r"W4-(T0|TXT)(?:-L([123]))?(?:-d(\d+))?(?:-K(\d+))?(?:-dr(\d+)e(\d+))?")
SHAS = {**Q4.SHAS, "qd_gnn5": QG.sha(__file__)}   # at import: what ran
NAMES = {}   # id(spec) -> arm name (parse_arms)
DIAG = {}    # arm name -> per-call records
PHASE = {"now": "read"}


# ── the model ────────────────────────────────────────────────────────────────


class W4(Q4.W3):
    def __init__(self, kind, d=64, L=3, K=0, dr=32):
        super().__init__(kind, d, L, K)
        self.Wr = torch.nn.Linear(2 * L + 3, dr)
        self.Gr = torch.nn.Linear(d, dr, bias=False)
        self.wo = torch.nn.Linear(dr, 1)
        torch.nn.init.zeros_(self.wo.weight)
        torch.nn.init.zeros_(self.wo.bias)

    def parts(self, X):
        """qn, W3's boost, the channel boosts B (B, N, 2L) and n_v."""
        qn = self.lnq(self.Wq(X.qemb))
        e = self.chains(X)
        W = self.A(qn) @ e.T + self.c[X.b_u, (X.len_u - 1).clamp_min(0)][None, :]
        w = torch.gather(W, 1, X.tix.clamp_min(0)).masked_fill(~X.tmask, float("-inf"))
        null = qn @ self.nu + self.c_null
        p = torch.softmax(torch.cat([w, null[:, None]], 1), 1)[:, :-1]
        beta = torch.nn.functional.softplus(self.beta_raw)
        contrib = p[X.eq, X.et] * X.ew.pow(-beta)
        boost = torch.zeros_like(X.z).index_put((X.eq, X.en), contrib, accumulate=True)
        ch = (X.b_u * self.L + (X.len_u - 1).clamp_min(0))[X.tix[X.eq, X.et]]
        Bc = torch.zeros(X.z.shape + (2 * self.L,)).index_put((X.eq, X.en, ch), contrib, accumulate=True)
        n_v = torch.zeros_like(X.z).index_put((X.eq, X.en), torch.ones_like(contrib), accumulate=True)
        return qn, boost, Bc, n_v

    def forward(self, X):
        qn, boost, Bc, n_v = self.parts(X)
        fin = torch.isfinite(X.z)
        zf = torch.where(fin, X.z, torch.zeros_like(X.z))
        f = torch.cat([zf[..., None], Bc, torch.log1p(n_v)[..., None], (n_v > 0).to(zf.dtype)[..., None]], -1)
        r = self.wo(torch.relu(self.Wr(f) + self.Gr(qn)[:, None, :])).squeeze(-1)
        return torch.where(fin, X.z + torch.exp(self.log_kappa) * boost + r, X.z)


# ── the arms ─────────────────────────────────────────────────────────────────


class QDArm5(Q4.QDArmAny):
    """qd_gnn4's arms (QD, W3) and W4; qd_gnn2.run builds every 'qd'-family arm as QG.QDArm(...)."""

    def __new__(cls, sp, *args, **kwargs):
        if cls is QDArm5:
            if sp.get("w4"):
                return object.__new__(W4Arm)
            if sp.get("w3"):
                return object.__new__(W3Arm5)
        return object.__new__(cls)

    def prepare(self, model, g, phi=None):
        self.shuf = phi is not None    # the SHUF read prepares with another table
        return super().prepare(model, g, phi)


class W3Arm5(QDArm5, Q4.W3Arm):
    """qd_gnn4.W3Arm, unchanged, as a QDArm5 (so qd_gnn2.score_rows reads it in QD chunks)."""


class W4Arm(W3Arm5):
    def __init__(self, sp, Q, TOK, G, z_of, A16, n_id):
        super().__init__(sp, Q, TOK, G, z_of, A16, n_id)
        key = f"W4-{self.kind}-L{self.L}-K{self.K}"
        self.st = Q4.STATS.setdefault(key, {"rows": 0, "cached_rows": 0, "seconds": 0.0, "types": 0, "pairs": 0})

    def make(self):
        model = W4(self.kind, self.sp["d"], self.L, self.K)
        if self.p_row > 0 or self.p_edge > 0:
            self.rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        return model


def parse_w4(nm):
    if not W4RE.fullmatch(nm):
        raise SystemExit(f"{nm}: W4-(T0|TXT)[-L<1..3>][-d<dim>][-K<labels>][-dr<R>e<E>]")
    sp = Q4.parse_w3("W3-" + nm[3:])
    sp["w4"] = True
    return sp


_parse4 = Q4.parse_arms


def parse_arms(spec, names, rule, AU, AG, G3):
    names_all = spec.split(",")
    rest = [nm for nm in names_all if not nm.startswith("W4-")]
    base = _parse4(",".join(rest), names, rule, AU, AG, G3) if rest else {}
    out = {nm: (parse_w4(nm) if nm.startswith("W4-") else base[nm]) for nm in names_all}
    for nm, sp in out.items():
        NAMES[id(sp)] = nm
    return out


# ── diagnostics ──────────────────────────────────────────────────────────────


def row_diag(s, z, gold):
    d = s - z
    k = min(5, s.size)
    top = np.argpartition(-s, k - 1)[:k]
    g = gold > 0
    lift = float(d[g].mean() - d[~g].mean()) if g.any() and (~g).any() else None
    return (float(d.std() / max(float(z.std()), 1e-6)), float(np.argmax(s) != np.argmax(z)),
            float((np.abs(d) > 1.0).mean()), float(np.abs(d[top]).mean()), lift)


def diag(arm, g, rows, TY, SZ):
    name = NAMES.get(id(getattr(arm, "sp", None)), "?")
    view = "NR" if (getattr(arm, "nr", False) or TY is not None) else ("SHUF" if getattr(arm, "shuf", False) else "ID")
    R = [row_diag(s, z, arm.Q[i]["gold"]) for (s, z), i in zip(SZ, rows)]
    lifts = [r[4] for r in R if r[4] is not None]
    rec = {"graph": g, "view": view, "phase": PHASE["now"], "rows": len(rows),
           "ratio": round(float(np.mean([r[0] for r in R])), 4), "moved": round(float(np.mean([r[1] for r in R])), 4),
           "big": round(float(np.mean([r[2] for r in R])), 4), "top5": round(float(np.mean([r[3] for r in R])), 4),
           "lift": round(float(np.mean(lifts)), 4) if lifts else None, "lift_rows": len(lifts)}
    DIAG.setdefault(name, []).append(rec)
    if PHASE["now"] == "read":
        QG.log(f"diag {name} {g} {view}: ratio {rec['ratio']} moved {rec['moved']} big {rec['big']} "
               f"top5 {rec['top5']} lift {rec['lift']}")


_score_rows = Q2.score_rows
_choose_margin = Q2.choose_margin


def score_rows(arm, model, g, rows, TY=None):
    SZ = _score_rows(arm, model, g, rows, TY)
    try:
        diag(arm, g, rows, TY, SZ)
    except Exception as ex:  # noqa: BLE001 -- a diagnostic never stops a run
        DIAG.setdefault("_failed", []).append({"graph": g, "error": repr(ex)})
    return SZ


def choose_margin(*args, **kwargs):
    PHASE["now"] = "select"
    try:
        return _choose_margin(*args, **kwargs)
    finally:
        PHASE["now"] = "read"


def bind():
    QG.QDArm = QDArm5               # qd_gnn2.run builds every 'qd'-family arm as QG.QDArm(...)
    Q2.parse_arms = parse_arms      # parses the arm list through Q2.parse_arms
    Q2.load_all = Q4.load_all4      # loads the graphs through Q2.load_all
    Q2.score_rows = score_rows      # scores every read through Q2.score_rows
    Q2.choose_margin = choose_margin


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    sys.path.insert(0, str(QG.AU_DIR))
    sys.path.insert(0, str(QG.KB_DIR))
    import anchor_univ as AU
    A16, AG = AU.A16, AU.AG
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    K = 8
    Q = QG.synthetic_rows(60, rng, K)
    TOK = [QG.edge_labels(q) for q in Q]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    phi = rng.standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32)
    G = {"g": {"phi": phi, "kind": "passage"}}
    rows = list(range(60))
    bind()
    # an unfitted W4 scores as the unfitted W3 from the same generator state; its channels sum to W3's boost
    for nm in ("W3-T0-L3-d16", "W3-TXT-L2-d16-K8"):
        sp3 = Q4.parse_w3(nm)
        arm = QG.QDArm(sp3, Q, TOK, G, z_of, A16, 0)
        TYs = arm.types(rows[:16], False)
        X = Q4.pack_w3(rows[:16], Q, z_of, TYs, arm.nt, sp3["L"])
        torch.manual_seed(3)
        m3 = Q4.W3(sp3["kind"], 16, sp3["L"], sp3["K"])
        torch.manual_seed(3)
        m4 = W4(sp3["kind"], 16, sp3["L"], sp3["K"])
        m3.set_phi(phi, sp3["K"])
        m4.set_phi(phi, sp3["K"])
        with torch.no_grad():
            s3, s4 = m3(X), m4(X)
            _qn, boost, Bc, n_v = m4.parts(X)
        assert torch.equal(s3, s4), f"{nm}: an unfitted W4 differs from W3"
        assert float((Bc.sum(-1) - boost).abs().max()) < 1e-5, f"{nm}: the channels do not sum to the boost"
        nb = int((n_v > 0).sum())
        assert nb > 0 and Bc.shape[-1] == 2 * sp3["L"]
        print(f"selftest {nm}: unfitted W4 == W3, channels sum to the boost, {nb} reached nodes")
    # dispatch, fit, reproducibility, the readout learns, NR moves label arms only
    tr, sel = {"g": rows[:40]}, {"g": rows[40:]}
    for nm in ("W4-T0-L3-d16", "W4-TXT-L2-d16-K8-dr25e10", "W3-T0-L2-d16", "QD-T0-L2-d16"):
        sp = parse_w4(nm) if nm.startswith("W4-") else (Q4.parse_w3(nm) if nm.startswith("W3-") else QG.parse_qd(nm))
        want = W4Arm if nm.startswith("W4-") else (W3Arm5 if nm.startswith("W3-") else QDArm5)
        arm = QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
        assert type(arm) is want and isinstance(arm, QG.QDArm) and isinstance(arm, Q3.QDArmDrop), (nm, type(arm))
        m0, e0, c0, _ = Q2.fit_shared2(arm, ["g"], tr, sel, "both", 3, 2e-3, 1e-4, 0, "graph", AU)
        arm2 = QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
        m1, e1, c1, _ = Q2.fit_shared2(arm2, ["g"], tr, sel, "both", 3, 2e-3, 1e-4, 0, "graph", AU)
        assert c0 == c1 and all(torch.equal(m0.state_dict()[k], m1.state_dict()[k]) for k in m0.state_dict()), f"{nm}: not reproducible"
        if nm.startswith("W4-"):
            assert float(m0.wo.weight.abs().max()) > 0, f"{nm}: the readout did not move"
        arm.prepare(m0, "g")
        s_id = arm.forward(m0.eval(), "g", rows[:8])[0]
        fin = torch.isfinite(s_id)
        assert all(int(fin[bi].sum()) == Q[i]["n"] for bi, i in enumerate(rows[:8])), f"{nm}: a pool node is not scored"
        arm.nr = True
        s_nr = arm.forward(m0, "g", rows[:8])[0]
        arm.nr = False
        dnr = float((s_id[fin] - s_nr[fin]).abs().max())
        assert (dnr == 0.0) == (sp["kind"] == "T0"), f"{nm}: NR read {dnr}"
        NAMES[id(sp)] = nm
        DIAG.clear()
        SZ = Q2.score_rows(arm, m0, "g", rows[40:])
        assert len(SZ) == 20 and DIAG[nm][-1]["view"] == "ID" and DIAG[nm][-1]["rows"] == 20, DIAG
        arm.nr = True
        Q2.score_rows(arm, m0, "g", rows[40:])
        arm.nr = False
        arm.prepare(m0, "g", phi[::-1].copy())
        Q2.score_rows(arm, m0, "g", rows[40:])
        arm.prepare(m0, "g")
        assert [r["view"] for r in DIAG[nm]] == ["ID", "NR", "SHUF"], DIAG[nm]
        print(f"selftest {nm}: {type(arm).__name__}, fitted ({c0}), reproducible, NR moves {dnr:.3g}, "
              f"params {sum(p.numel() for p in m0.parameters())}; diag {DIAG[nm][0]}")
    # the diagnostic itself: a residual of +1 on the golds lifts by 1, moves no node past one z unit, and puts a gold
    # (z 0.45 + 1) above z's rank 1 (z 1.0)
    z = np.linspace(0.0, 1.0, 12)
    gold = np.zeros(12)
    gold[[2, 5]] = 1.0
    ratio, moved, big, top5, lift = row_diag(z + gold, z, gold)
    assert abs(lift - 1.0) < 1e-12 and big == 0.0 and moved == 1.0, (ratio, moved, big, top5, lift)
    assert row_diag(z + 0.5 * gold, z, gold)[1] == 0.0   # +0.5 leaves rank 1 where it was
    assert row_diag(z, z, gold)[0] == 0.0
    # the select pass is labelled select, and choose_margin's result is unchanged
    DIAG.clear()
    sp = QG.parse_qd("QD-T0-L2-d16")
    arm = QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
    NAMES[id(sp)] = "QD-T0-L2-d16"
    m0 = arm.make()
    got = Q2.choose_margin(arm, m0, ["g"], sel, Q, A16.metrics_of)
    ref = _choose_margin(arm, m0, ["g"], sel, Q, A16.metrics_of)
    assert got == ref and PHASE["now"] == "read" and DIAG["QD-T0-L2-d16"][0]["phase"] == "select"
    # the grammar, through the bound parser
    AU.check_pins()
    AU.AC.bind()
    arms = Q2.parse_arms("QD-T0-L2,W3-T0-L3,W4-T0-L3,W4-TXT-L3-dr25e10,QD-TXT-L2-dr25e10,W:T0", ["2wiki"], "both", AU, AG, AU.G3)
    assert arms["W4-TXT-L3-dr25e10"]["w4"] and arms["W4-TXT-L3-dr25e10"]["drop_row"] == 0.25 and not arms["W3-T0-L3"].get("w4")
    assert all(NAMES[id(sp)] == nm for nm, sp in arms.items())
    for bad in ("W4-T0-K8", "W4-T0-dr25e10", "W4-XX"):
        try:
            parse_w4(bad)
        except SystemExit:
            continue
        raise AssertionError(f"{bad} parsed")
    print({k: (v.get("kind", v.get("tok")), v.get("w3", False), v.get("w4", False), v.get("max_len"), v.get("drop_row"))
           for k, v in arms.items()})
    print("selftest: all checks passed")


def main():
    bind()
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--cap-len" in sys.argv:
        k = sys.argv.index("--cap-len")
        for kv in sys.argv[k + 1].split(","):
            g, m = kv.split("=")
            if g not in Q2.READ or int(m) < 1:
                raise SystemExit(f"--cap-len {kv}: graph=max_len, graph one of {list(Q2.READ)}")
            Q4.CAP[g] = int(m)
        del sys.argv[k:k + 2]
    if "--base" in sys.argv:
        k = sys.argv.index("--base")
        if sys.argv[k + 1] not in ("twin", "gnn"):
            raise SystemExit("--base twin|gnn")
        Q4.BASE["z"] = sys.argv[k + 1]
        del sys.argv[k:k + 2]
    _run = Q2.run

    def run(a):
        _run(a)
        out = Path(a.out)
        res = json.loads(out.read_text(encoding="utf-8"))
        res["look"] = "qd_gnn5"
        res["pins"].update({k: v for k, v in SHAS.items() if k != "qd_gnn5"})
        res["qd_gnn5_sha256"] = SHAS["qd_gnn5"]
        res["cap_len"], res["max_len_by_graph"], res["base_z"] = Q4.CAP, Q4.LOADED, Q4.BASE["z"]
        res["w3_compile"] = {k: {**v, "ms_per_row": round(1000 * v["seconds"] / max(1, v["rows"]), 3),
                                 "types_per_row": round(v["types"] / max(1, v["rows"]), 1),
                                 "pairs_per_row": round(v["pairs"] / max(1, v["rows"]), 1)} for k, v in Q4.STATS.items()}
        res["residual_diag"] = DIAG
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        QG.log(f"w3 compile: {res['w3_compile']}")

    Q2.run = run
    Q2.main()


if __name__ == "__main__":
    main()
