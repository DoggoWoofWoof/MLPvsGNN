"""Design look (untracked; not a result and not filed): qd_gnn7.py, pinned and unchanged, with two label-channel
options for QD-TXT arms, aimed at the joint fit's failure on metaqa. In q4-j4 (metaqa, 2wiki, hotpot and musique
trained together) QD-TXT reads rho 0.005 on metaqa and 0.112 with every label removed, while with 2wiki left out
(q4-lo-2w) the same arm reads 0.606 on metaqa (0.066 without labels): 2wiki's anchor phrases and metaqa's relation
names share P, M and A, and metaqa's few relation vectors sit close together in that shared text space.

Arm-name options (QD-TXT arms only), appended with qd_gnn7's in any order:
  -pc   per-graph label centring: after the arm prepares a graph, the model's label table (its first K rows, already
        unit-normalised by QD.set_phi) has its own mean taken off and each row is normalised again, so a graph's labels
        differ from each other and not from another graph's. It is computed from the graph's own label table, so it
        needs no dataset id and runs on any graph, a new one included.
  -lq   query-label cosine in the gate: each layer's gate logit adds kappa_l c_e, where c_e is the cosine of the row's
        query embedding and the edge's label text (both 1536-wide, the same embedder; 0 for 'other', a dropped label
        and the NR read). kappa starts at 0, so an unfitted -lq model scores as the same model without it, bit for bit.
        c_e reads the raw label text, never the centred one, and only the query and the label: no node, no neighbour.
With neither option an arm is qd_gnn7's, draw for draw (the same classes and calls). Both options keep QD a
message-passing model; they change what the label channel sees, not what the walks or the twin see.

    python outputs/mp_unified/qd_gnn10.py --selftest
    python outputs/mp_unified/qd_gnn10.py --train metaqa=fit,2wiki=x4 --read hotpotqa,webqsp \
        --arms QD-TXT-L3-dr25e10,QD-TXT-L3-dr25e10-pc,QD-TXT-L3-dr25e10-lq,QD-TXT-L3-dr25e10-pc-lq \
        --rule np --epochs 6 --kalpha 64,256 --out outputs/mp_unified/qd/pc-mq2w.json
"""
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn7 as Q7  # noqa: E402  (imports qd_gnn6 ... qd_gnn, which sets the BLAS thread counts before numpy loads)

import numpy as np  # noqa: E402
import torch  # noqa: E402

Q6x, Q5, Q4, Q2, QG = Q7.Q6x, Q7.Q5, Q7.Q4, Q7.Q2, Q7.QG
SHAS = {**Q7.SHAS, "qd_gnn10": QG.sha(__file__)}   # at import: what ran
OPT10 = re.compile(r"-(pc|lq)(?=-|$)")
log = QG.log


# ── arm options ──────────────────────────────────────────────────────────────


_split7 = Q7.split_opts


def split_opts(nm):
    """(the qd_gnn6 name, the qd_gnn7 + qd_gnn10 options) of an arm name; -pc / -lq may sit anywhere among the
    options, qd_gnn7's are parsed by qd_gnn7 itself."""
    found = OPT10.findall(nm)
    if len(found) != len(set(found)):
        raise SystemExit(f"{nm}: an option given twice")
    rest = OPT10.sub("", nm)
    base, opts = _split7(rest)
    if found:
        if not base.startswith("QD-TXT"):
            raise SystemExit(f"{nm}: -pc / -lq are for QD-TXT arms only")
        opts = {**opts, **{k: True for k in found}}
    return base, opts


# ── the model ────────────────────────────────────────────────────────────────


class QD10(Q7.QD7):
    """qd_gnn7.QD7 plus lq (see the module docstring); with lq off it is QD7, call for call."""

    def __init__(self, kind, d=64, layers=3, n_id=0, g1=False, tb=None, rs=False, lq=False):
        super().__init__(kind, d, layers, n_id, g1=g1, tb=tb, rs=rs)
        self.lq = bool(lq)
        if self.lq:
            if kind != "TXT":
                raise ValueError("lq reads the TXT label channel")
            self.kappa = torch.nn.Parameter(torch.zeros(layers))
        self.register_buffer("phi_u", torch.zeros(0, QG.TDIM), persistent=False)

    def set_phi(self, phi, K):
        super().set_phi(phi, K)
        if self.kind == "TXT":
            t = torch.as_tensor(np.asarray(phi[:K], dtype=np.float32))
            self.phi_u = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def label_cos(self, P):
        """c_e per edge: cos(query, label text) for a label below K, else 0."""
        has = P.a >= 0
        qu = P.qemb / P.qemb.norm(dim=1, keepdim=True).clamp_min(1e-12)
        C = qu @ self.phi_u.T                                          # (B, K)
        c = C[P.e_row, P.a.clamp_min(0)]
        return torch.where(has, c, torch.zeros_like(c))

    def forward(self, P):
        if not self.lq:
            return super().forward(P)
        rms = QG.rms
        qn = self.lnq(self.Wq(P.qemb))                                  # (B, d)
        h0 = qn[P.seed_row] * (1.0 + self.E_b[P.seed_b]) + P.z[P.seed_idx, None] * self.e_z
        h = torch.zeros(P.N, self.dim).index_add(0, P.seed_idx, h0)
        r = self.relations(P)
        c = self.label_cos(P)
        gamma = torch.sigmoid(self.gamma_raw)
        fam = P.fam.clamp(max=2)
        for l in range(self.L):
            w = self.R[l](r) * (1.0 + torch.tanh(self.G[l](qn)))[P.e_row]
            logit = (self.A[l](qn)[P.e_row] * r).sum(-1) + self.c_fam[l, fam] * P.w + self.b_gate[l]
            logit = logit + self.kappa[l] * c
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


def centre_phi(model):
    """-pc: the model's label table centred on its own mean, each row normalised again (in place)."""
    t = model.phi
    if t.numel() == 0:
        return
    t = t - t.mean(0, keepdim=True)
    model.phi = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12) * math.sqrt(t.shape[1])


# ── the arms ─────────────────────────────────────────────────────────────────


class QDArm10(Q7.QDArm7):
    """qd_gnn7's arms with -pc / -lq on QD-TXT arms; qd_gnn2.run builds every 'qd'-family arm as QG.QDArm(...)."""

    def __new__(cls, sp, *args, **kwargs):
        if cls is QDArm10:
            if sp.get("w4"):
                return object.__new__(W4Arm10)
            if sp.get("w3"):
                return object.__new__(W3Arm10)
        return object.__new__(cls)

    def make(self):
        o = self.sp.get("t7") or {}
        if not o.get("lq"):
            return super().make()
        model = QD10(self.sp["kind"], self.sp["d"], self.sp["L"], self.n_id, g1=o.get("g1", False), tb=o.get("tb"),
                     rs=o.get("rs", False), lq=True)
        # qd_gnn7.QDArm7.make's generators, drawn as it draws them (kappa is zeros: no draw at init)
        if self.p_row > 0 or self.p_edge > 0:
            self.rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        self.rng_ed = (np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item())) if o.get("ed") else None)
        return model

    def prepare(self, model, g, phi=None):
        out = super().prepare(model, g, phi)
        if (self.sp.get("t7") or {}).get("pc") and getattr(model, "kind", None) == "TXT":
            centre_phi(model)
        return out


class W3Arm10(QDArm10, Q7.W3Arm7):
    """qd_gnn7.W3Arm7, unchanged (no qd_gnn10 option reaches it), as a QDArm10."""


class W4Arm10(QDArm10, Q7.W4Arm7):
    """qd_gnn7.W4Arm7, unchanged (no qd_gnn10 option reaches it), as a QDArm10."""


# ── binding and output ───────────────────────────────────────────────────────


_bind7 = Q7.bind


def bind():
    _bind7()
    QG.QDArm = QDArm10
    Q7.split_opts = split_opts       # qd_gnn7.parse_arms (bound by _bind7) splits names through it


def finish(out):
    res = json.loads(out.read_text(encoding="utf-8"))
    res["look"] = "qd_gnn10"
    res["pins"]["qd_gnn7"] = SHAS["qd_gnn7"]
    res["qd_gnn10_sha256"] = SHAS["qd_gnn10"]
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
    bind()
    # names
    assert split_opts("QD-TXT-L3-dr25e10-pc") == ("QD-TXT-L3-dr25e10", {"pc": True})
    assert split_opts("QD-TXT-L3-dr25e10-pc-tb5-ed50") == ("QD-TXT-L3-dr25e10", {"tb": 0.5, "ed": 0.5, "pc": True})
    assert split_opts("QD-TXT-L3-dr25e10-tb5-lq-pc") == ("QD-TXT-L3-dr25e10", {"tb": 0.5, "lq": True, "pc": True})
    assert split_opts("QD-TXT-L3-dr25e10-tb5") == ("QD-TXT-L3-dr25e10", {"tb": 0.5})
    assert split_opts("W3-TXT-L2-dr25e10") == ("W3-TXT-L2-dr25e10", {})
    for bad in ("QD-T0-L3-pc", "W3-TXT-L3-pc", "QD-TXT-L3-pc-pc", "QD-ID-L3-lq", "W:T0-lq"):
        try:
            split_opts(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
    print("selftest: -pc / -lq parse anywhere among the options; repeats and non-QD-TXT uses are refused")
    # QD10: lq off is QD7; lq on with kappa 0 is QD7 bit for bit; kappa moves only labelled edges' gates
    Q = QG.synthetic_rows(24, rng, K)
    TOK = [QG.edge_labels(q) for q in Q]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    phi = rng.standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32)
    rows = list(range(24))
    P = QG.pack_qd(rows, Q, TOK, z_of, K)
    assert int((P.a >= 0).sum()) > 0 and int((P.a < 0).sum()) > 0
    torch.manual_seed(1)
    m7 = Q7.QD7("TXT", 16, 2, K, tb=2.0)
    torch.manual_seed(1)
    m10 = QD10("TXT", 16, 2, K, tb=2.0, lq=True)
    m7.set_phi(phi, K)
    m10.set_phi(phi, K)
    with torch.no_grad():
        torch.manual_seed(2)
        m7.wo.weight.normal_(0, 0.5)
    sd = dict(m7.state_dict())
    sd["kappa"] = torch.zeros(2)
    m10.load_state_dict(sd)
    assert torch.equal(m7(P), m10(P))
    c = m10.label_cos(P)
    assert bool((c[P.a < 0] == 0).all()) and float(c[P.a >= 0].abs().max()) > 0
    m10(P).sum().backward()
    assert m10.kappa.grad is not None and float(m10.kappa.grad.abs().max()) > 0
    with torch.no_grad():
        m10.kappa.fill_(3.0)
    assert not torch.equal(m7(P), m10(P))
    print("selftest: QD10 with kappa 0 scores as QD7 bit for bit; c_e is 0 on 'other' edges; kappa gets a gradient "
          "and moves the score")
    # the arms: dispatch, -pc centres the table, an arm without the options is qd_gnn7's
    G = {"g": {"phi": phi, "kind": "passage"}}
    for nm, cls in (("QD-TXT-L2-d16-K8", QDArm10), ("W3-T0-L2-d16", W3Arm10), ("W4-T0-L2-d16", W4Arm10)):
        s_ = Q2.parse_arms(nm, ["g"], "np", AU, AU.AG, AU.G3)[nm]
        a_ = QG.QDArm(s_, Q, TOK, G, z_of, A16, 0)
        assert type(a_) is cls and isinstance(a_, QG.QDArm) and isinstance(a_, Q7.QDArm7), nm
    sp0 = Q2.parse_arms("QD-TXT-L2-d16-K8", ["g"], "np", AU, AU.AG, AU.G3)["QD-TXT-L2-d16-K8"]
    spc = Q2.parse_arms("QD-TXT-L2-d16-K8-pc", ["g"], "np", AU, AU.AG, AU.G3)["QD-TXT-L2-d16-K8-pc"]
    assert spc["t7"] == {"pc": True}
    a0, ac = QG.QDArm(sp0, Q, TOK, G, z_of, A16, 0), QG.QDArm(spc, Q, TOK, G, z_of, A16, 0)
    torch.manual_seed(4)
    mA = a0.make()
    torch.manual_seed(4)
    mB = ac.make()
    assert type(mA) is type(mB) is QG.QD
    a0.prepare(mA, "g")
    ac.prepare(mB, "g")
    ref = torch.as_tensor(phi[:K])
    ref = ref / ref.norm(dim=1, keepdim=True) * math.sqrt(QG.TDIM)
    assert torch.equal(mA.phi, ref)
    # centred, then each row normalised again: the mean is near 0 against a row's length (exactly 0 before the norm)
    assert float(mB.phi.mean(0).norm()) / math.sqrt(QG.TDIM) < 0.05, float(mB.phi.mean(0).norm())
    assert torch.allclose(mB.phi.norm(dim=1), torch.full((K,), math.sqrt(QG.TDIM)), rtol=1e-5)
    a7 = Q7.QDArm7(sp0, Q, TOK, G, z_of, A16, 0)
    torch.manual_seed(4)
    m7a = a7.make()
    m7a.load_state_dict(mA.state_dict())
    a7.prepare(m7a, "g")
    with torch.no_grad():
        mA.wo.weight.normal_(0, 0.5)
    m7a.load_state_dict(mA.state_dict())
    mA.eval()
    m7a.eval()
    with torch.no_grad():
        assert torch.equal(a0.forward(mA, "g", rows)[0], a7.forward(m7a, "g", rows)[0])
    # bunched labels (one shared direction plus small noise) come apart under -pc
    base_v = rng.standard_normal(QG.TDIM).astype(np.float32)
    bunched = (base_v[None, :] * 4.0 + rng.standard_normal((K, QG.TDIM)).astype(np.float32)).astype(np.float32)
    mC = ac.make()
    ac.prepare(mC, "g", bunched)
    mD = a0.make()
    a0.prepare(mD, "g", bunched)

    def mean_cos(t):
        u = t / t.norm(dim=1, keepdim=True)
        cc = u @ u.T
        return float((cc.sum() - cc.diagonal().sum()) / (K * (K - 1)))
    assert mean_cos(mD.phi) > 0.8 and abs(mean_cos(mC.phi)) < 0.2, (mean_cos(mD.phi), mean_cos(mC.phi))
    print(f"selftest: -pc centres each graph's table (bunched labels: mean cosine {mean_cos(mD.phi):.2f} -> "
          f"{mean_cos(mC.phi):.2f}); without options the arm is qd_gnn7's")
    # the full loop through qd_gnn2.run on a synthetic KB
    names = [f"rel {j}" for j in range(K)]
    real = Q4._load_all

    def fake(trains, reads, max_len, t0, AU_):
        Qf = Q6x.synthetic_kb(80, np.random.default_rng(5), K)
        part = {("metaqa", "x1"): list(range(0, 40)), ("metaqa", "select"): list(range(40, 52)),
                ("metaqa", "fit"): list(range(52, 80))}
        Gf = {"metaqa": {"kind": "kb", "role": "train", "phi": phi, "phi_sha": "synthetic", "n_rel": K, "vocab": names,
                         "info": {"pruned_loader": {"synthetic": True}}}}
        return Qf, part, Gf

    Q4._load_all = fake
    tmp = Path(tempfile.mkdtemp(prefix="qd10_"))
    keep = (Q2.run, Q5.bind, sys.argv, Q7.bind)
    try:
        Q6x.HOLD.clear()
        for key in ("full", "tok", "held_rows"):
            Q6x.STATE[key] = {}
        Q6x.STATE["out"] = None
        Q6x.RES.clear()
        Q6x.ROWS.clear()
        Q7.KAR.clear()
        Q7.KROWS.clear()
        Q7.KA["Ns"] = []
        out = tmp / "pc.json"
        arms = "QD-TXT-L2-d16-K8,QD-TXT-L2-d16-K8-pc,QD-TXT-L2-d16-K8-lq,QD-TXT-L2-d16-K8-pc-lq-tb20,W3-T0-L2-d16"
        sys.argv = ["qd_gnn10.py", "--train", "metaqa=fit", "--arms", arms, "--rule", "both", "--epochs", "2",
                    "--out", str(out), "--kalpha", "4,8", "--kalpha-draws", "2"]
        main(selftest_run=True)
        res = json.loads(out.read_text(encoding="utf-8"))
        assert res["look"] == "qd_gnn10" and res["qd_gnn10_sha256"] == SHAS["qd_gnn10"]
        assert set(res["results"]) == set(arms.split(","))
        assert res["results"]["QD-TXT-L2-d16-K8-pc-lq-tb20"]["spec"]["t7"] == {"tb": 2.0, "pc": True, "lq": True}
        assert "_failed" not in Q7.KAR, Q7.KAR.get("_failed")
        mdir = out.parent / (out.stem + "_models")
        ck = torch.load(mdir / "a2_s0.pt", weights_only=False)
        # the synthetic fit keeps epoch 0 (a flat select), so kappa may still be 0 here; it is saved and loads back
        assert "kappa" in ck["state_dict"] and tuple(ck["state_dict"]["kappa"].shape) == (2,)
        assert ck["spec"]["t7"] == {"lq": True}
        r0 = res["results"]["QD-TXT-L2-d16-K8"]["seeds_read"]["0"]["reads"]["metaqa"]["ID/np"]["fit"]
        r1 = res["results"]["QD-TXT-L2-d16-K8-pc"]["seeds_read"]["0"]["reads"]["metaqa"]["ID/np"]["fit"]
        print(f"selftest e2e: five arms ran (the -lq arm's kappa learned {ck['state_dict']['kappa'].tolist()}); "
              f"ID/np plain {r0} vs -pc {r1}")
    finally:
        Q4._load_all = real
        Q2.run, Q5.bind, sys.argv, Q7.bind = keep
        Q6x.HOLD.clear()
        Q7.KA["Ns"] = []
    print("selftest: all checks passed")


def main(selftest_run=False):
    if "--selftest" in sys.argv and not selftest_run:
        selftest()
        return
    Q7.bind = bind          # qd_gnn7.main binds through its module's 'bind' (and hands it to qd_gnn5 as Q5.bind)
    Q7.finish = lambda out: (Q7.__dict__["_finish7"](out), finish(out))
    Q7.main()


Q7._finish7 = Q7.finish


if __name__ == "__main__":
    main()
