"""Design look (untracked; not a result and not filed): qd_gnn12.py, pinned and unchanged, with three arm-name options
for QD arms. Two put relation structure into the message itself, as R-GCN and relational attention do, where QD only
scales and gates an edge by its relation; the third carries the lean MLP's block dropout to the GNN's input graph.

  -rg<B>  relation-specific transforms by basis decomposition (R-GCN, Schlichtkrull et al. 2018). Each layer's message
          becomes
              m_e = (sum_b c_lb(r_e) (I + D_lb) rms(h_u)) * w_e * g_e,      c_l(r_e) = softmax(C_l r_e) over B bases,
          with w_e and g_e QD's own (the relation's diagonal transform under the query, and the query-relation gate).
          r_e is QD's relation vector: the famdir embedding plus, on a TXT arm, the label's text through P and M, so an
          unseen relation's transform is read from its text and one model runs on any graph. Every D_lb starts at zero
          (weight decay pulls it back there), so an unfitted -rg model is QD to rounding, and the bases separate only
          as far as the relations' coefficients pull them apart.
  -at     relational attention over each node's in-edges (GATv2, Brody et al. 2022; RGAT, Busbridge et al. 2019):
              e_e     = a_l . LeakyReLU(U_l^s rms(h_u) + U_l^d rms(h_v) + U_l^r r_e + U_l^q qn)
              alpha_e = n_v softmax(e_e) over v's in-edges whose source has a state     (mean one over them)
          multiplies each message. QD's degree normalisation (1 + n_v)^gamma is kept, so the sum keeps its count. a_l
          starts at zero, so alpha is 1 and an unfitted -at model is QD to rounding. Unlike QD's gate, alpha reads both
          endpoints' states, so an edge is weighed against the other edges into the same node.
  -fd<P>  edge-family dropout in training: each batch row drops each edge family it has (structural, NER, kNN) with
          probability P / 100, keeping its structural family (or, with none, its first family) when every family it
          has would go. The lean MLP's block dropout carried to the GNN's input graph: a graph whose NER or kNN edges
          are weak or absent (a KB has structural edges only) should not cost the model its reading of the rest.
          Reads see every edge.

With none of the three an arm is qd_gnn12's, draw for draw (the same classes and calls). Each option keeps QD a
message-passing model. The frontier cut stays exact, because an attention weight is taken only over in-edges from nodes
with a state (--selftest checks it).

    python outputs/mp_unified/qd_gnn13.py --selftest
    python outputs/mp_unified/qd_gnn13.py --train 2wiki=x4 --read hotpotqa,musique,metaqa,webqsp --rule both \
        --kalpha 64,256 --arms QD-TXT-L2-dr25e10-tb5-ed50,QD-TXT-L2-dr25e10-tb5-ed50-at,QD-TXT-L2-dr25e10-tb5-ed50-rg4 \
        --out outputs/mp_unified/qd/rg-2w.json
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn12 as Q12  # noqa: E402  (imports qd_gnn11 ... qd_gnn, which sets the BLAS thread counts before numpy loads)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

Q11, Q10, Q7, Q6x, Q5, Q4, Q2, QG = Q12.Q11, Q12.Q10, Q12.Q7, Q12.Q6x, Q12.Q5, Q12.Q4, Q12.Q2, Q12.QG
SHAS = {**Q12.SHAS, "qd_gnn13": QG.sha(__file__)}   # at import: what ran
OPT13 = re.compile(r"-(at|rg\d+|fd\d+)(?=-|$)")
log = QG.log
_split10 = Q10.split_opts


# ── arm options ──────────────────────────────────────────────────────────────


def split_opts(nm):
    """(the qd_gnn6 name, the qd_gnn7 + qd_gnn10 + qd_gnn13 options) of an arm name; -at / -rg<B> / -fd<P> may sit
    anywhere among the options, the others are parsed by qd_gnn10 (and so qd_gnn7) themselves."""
    found = OPT13.findall(nm)
    keys = [re.match(r"[a-z]+", t).group(0) for t in found]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"{nm}: an option given twice")
    base, opts = _split10(OPT13.sub("", nm))
    if not found:
        return base, opts
    if not base.startswith("QD-"):
        raise SystemExit(f"{nm}: -at / -rg / -fd are for QD arms only")
    opts = dict(opts)
    for tok, key in zip(found, keys):
        if key == "at":
            opts["at"] = True
            continue
        n = int(tok[2:])
        if key == "rg" and not 2 <= n <= 16:
            raise SystemExit(f"{nm}: -rg<B> takes 2 to 16 bases")
        if key == "fd" and not 0 < n < 100:
            raise SystemExit(f"{nm}: -fd<P> takes 0 < P < 100")
        opts[key] = n if key == "rg" else n / 100.0
    return base, opts


# ── the model ────────────────────────────────────────────────────────────────


class QD13(Q10.QD10):
    """qd_gnn10.QD10 (with qd_gnn7.QD7's switches) plus rg and at; with both off it is QD10, call for call."""

    def __init__(self, kind, d=64, layers=3, n_id=0, g1=False, tb=None, rs=False, lq=False, at=False, rg=0):
        super().__init__(kind, d, layers, n_id, g1=g1, tb=tb, rs=rs, lq=lq)
        self.at, self.rg = bool(at), int(rg or 0)
        if self.rg:
            self.C = torch.nn.ModuleList([torch.nn.Linear(d, self.rg) for _ in range(layers)])
            self.D = torch.nn.Parameter(torch.zeros(layers, self.rg, d, d))
        if self.at:
            self.Us = torch.nn.ModuleList([torch.nn.Linear(d, d, bias=False) for _ in range(layers)])
            self.Ud = torch.nn.ModuleList([torch.nn.Linear(d, d, bias=False) for _ in range(layers)])
            self.Ur = torch.nn.ModuleList([torch.nn.Linear(d, d, bias=False) for _ in range(layers)])
            self.Uq = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(layers)])
            self.a_at = torch.nn.Parameter(torch.zeros(layers, d))

    def attention(self, l, hs, r, qn, P, active, n_in):
        """alpha_e: n_v times the softmax of e_e over v's in-edges whose source has a state, 0 on the others. The
        masked logits are finite and the max is taken without gradient, so no inf - inf reaches the backward."""
        pre = self.Us[l](hs)[P.src] + self.Ud[l](hs)[P.dst] + self.Ur[l](r) + self.Uq[l](qn)[P.e_row]
        e = (F.leaky_relu(pre, 0.2) * self.a_at[l]).sum(-1)
        on = active[P.src] > 0
        e = torch.where(on, e, torch.full_like(e, -1e30))
        mx = torch.zeros(P.N).scatter_reduce(0, P.dst, e.detach(), "amax", include_self=False)
        ex = torch.exp(e - mx[P.dst]) * on.to(e.dtype)
        den = torch.zeros(P.N).index_add(0, P.dst, ex)
        return ex / den[P.dst].clamp_min(1e-30) * n_in[P.dst]

    def forward(self, P):
        if not (self.at or self.rg):
            return super().forward(P)
        rms = QG.rms
        qn = self.lnq(self.Wq(P.qemb))                                  # (B, d)
        h0 = qn[P.seed_row] * (1.0 + self.E_b[P.seed_b]) + P.z[P.seed_idx, None] * self.e_z
        h = torch.zeros(P.N, self.dim).index_add(0, P.seed_idx, h0)
        r = self.relations(P)
        c = self.label_cos(P) if self.lq else None
        gamma = torch.sigmoid(self.gamma_raw)
        fam = P.fam.clamp(max=2)
        eye = torch.eye(self.dim)
        for l in range(self.L):
            w = self.R[l](r) * (1.0 + torch.tanh(self.G[l](qn)))[P.e_row]
            logit = (self.A[l](qn)[P.e_row] * r).sum(-1) + self.c_fam[l, fam] * P.w + self.b_gate[l]
            if self.lq:
                logit = logit + self.kappa[l] * c
            g = 2.0 * torch.sigmoid(logit)
            hs = rms(h)
            if self.rg:
                coef = torch.softmax(self.C[l](r), -1)                     # (E, B)
                x = coef[:, 0:1] * (hs @ (eye + self.D[l, 0]).T)[P.src]
                for b in range(1, self.rg):
                    x = x + coef[:, b:b + 1] * (hs @ (eye + self.D[l, b]).T)[P.src]
            else:
                x = hs[P.src]
            m = x * w * g[:, None]
            active = (h != 0).any(-1).to(m.dtype)
            n_in = torch.zeros(P.N).index_add(0, P.dst, active[P.src])
            if self.at:
                m = m * self.attention(l, hs, r, qn, P, active, n_in)[:, None]
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


def family_drop(P, rng, p):
    """-fd: each batch row drops each edge family it has with probability p, keeping its structural family (or, with
    none, its first family) when every family it has would go. In place; returns how many edges went."""
    fam = np.minimum(P.fam.numpy(), 2)
    row = P.e_row.numpy()
    has = np.zeros((P.B, 3), dtype=bool)
    has[row, fam] = True
    drop = (rng.random((P.B, 3)) < p) & has
    for b in np.flatnonzero(has.any(1) & (drop == has).all(1)):
        drop[b, 0 if has[b, 0] else int(np.flatnonzero(has[b])[0])] = False
    kill = drop[row, fam]
    keep = torch.from_numpy(~kill)
    for k in ("src", "dst", "e_row", "d", "a", "fam", "w"):
        setattr(P, k, getattr(P, k)[keep])
    return int(kill.sum())


# ── the arms ─────────────────────────────────────────────────────────────────


class QDArm13(Q10.QDArm10):
    """qd_gnn10's arms with -at / -rg / -fd on QD arms; qd_gnn2.run builds every 'qd'-family arm as QG.QDArm(...)."""

    def __new__(cls, sp, *args, **kwargs):
        if cls is QDArm13:
            if sp.get("w4"):
                return object.__new__(W4Arm13)
            if sp.get("w3"):
                return object.__new__(W3Arm13)
        return object.__new__(cls)

    def __init__(self, sp, Q, TOK, G, z_of, A16, n_id):
        super().__init__(sp, Q, TOK, G, z_of, A16, n_id)
        self.rng_fd = None

    def make(self):
        o = self.sp.get("t7") or {}
        if not (o.get("at") or o.get("rg")):
            model = super().make()
        else:
            model = QD13(self.sp["kind"], self.sp["d"], self.sp["L"], self.n_id, g1=o.get("g1", False), tb=o.get("tb"),
                         rs=o.get("rs", False), lq=o.get("lq", False), at=o.get("at", False), rg=o.get("rg", 0))
            # qd_gnn7.QDArm7.make's generators, drawn as it draws them
            if self.p_row > 0 or self.p_edge > 0:
                self.rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
            self.rng_ed = (np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item())) if o.get("ed") else None)
        self.rng_fd = (np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item())) if o.get("fd") else None)
        return model

    def forward(self, model, g, rows):
        o = self.sp.get("t7") or {}
        if not o.get("fd"):
            return super().forward(model, g, rows)
        Qp, Tp = self.rows_for(rows)                     # qd_gnn7.QDArm7.forward, then the family dropout
        P = QG.pack_qd(rows, Qp, Tp, self.z_of, self.sp["K"], self.nr)
        if model.training and self.rng is not None:
            drop = self.rng.random(P.B) < self.p_row
            de = drop[P.e_row.numpy()] | (self.rng.random(P.a.numel()) < self.p_edge)
            P.a = torch.where(torch.from_numpy(de), torch.full_like(P.a, -1), P.a)
        if model.training and self.rng_ed is not None:
            Q7.edge_drop(P, self.rng_ed, o["ed"])
        if model.training:
            family_drop(P, self.rng_fd, o["fd"])
        return QG.padded(P, model(P))


class W3Arm13(QDArm13, Q10.W3Arm10):
    """qd_gnn10.W3Arm10, unchanged (no qd_gnn13 option reaches it), as a QDArm13."""


class W4Arm13(QDArm13, Q10.W4Arm10):
    """qd_gnn10.W4Arm10, unchanged (no qd_gnn13 option reaches it), as a QDArm13."""


# ── binding and output ───────────────────────────────────────────────────────


_bind10 = Q10.bind


def bind():
    _bind10()
    QG.QDArm = QDArm13
    Q7.split_opts = split_opts       # qd_gnn7.parse_arms splits names through it (qd_gnn10's bind set its own)


def finish(out):
    res = json.loads(out.read_text(encoding="utf-8"))
    res["look"] = "qd_gnn13"
    res["pins"]["qd_gnn12"] = SHAS["qd_gnn12"]
    res["qd_gnn13_sha256"] = SHAS["qd_gnn13"]
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def main(selftest_run=False):
    if "--selftest" in sys.argv and not selftest_run:
        selftest()
        return
    Q10.bind = bind                  # qd_gnn10.main hands its module's 'bind' to qd_gnn7 by name (Q7.bind = bind)
    _finish12 = Q12.__dict__["_finish12"]
    Q12.finish = lambda out: (_finish12(out), finish(out))   # qd_gnn12.main's finish lambda calls Q12.finish by name
    Q12.main(selftest_run=selftest_run)


Q12._finish12 = Q12.finish


# ── selftest ─────────────────────────────────────────────────────────────────


ARMS_NEW = "QD-T0-L2-d16-at-fd50,QD-TXT-L2-d16-K8-pc-rg3,QD-TXT-L2-d16-K8-at-rg2-tb5"


def selftest():
    import os
    import subprocess
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
    assert split_opts("QD-TXT-L2-dr25e10-tb5-ed50-at-rg4-fd25") == ("QD-TXT-L2-dr25e10", {"tb": 0.5, "ed": 0.5, "at": True,
                                                                                         "rg": 4, "fd": 0.25})
    assert split_opts("QD-TXT-L2-dr25e10-at-pc-tb5") == ("QD-TXT-L2-dr25e10", {"tb": 0.5, "pc": True, "at": True})
    assert split_opts("QD-T0-L2-fd30") == ("QD-T0-L2", {"fd": 0.3})
    assert split_opts("QD-TXT-L2-d16-K8") == ("QD-TXT-L2-d16-K8", {})
    assert split_opts("W3-TXT-L2-dr25e10") == ("W3-TXT-L2-dr25e10", {})
    for bad in ("QD-T0-L2-at-at", "W3-T0-L2-at", "W4-T0-L2-rg4", "QD-T0-L2-rg1", "QD-T0-L2-rg17", "QD-T0-L2-fd0",
                "QD-T0-L2-fd100", "QD-T0-L2-rg2-rg3", "W:T0-fd20"):
        try:
            split_opts(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
    print("selftest: -at / -rg<B> / -fd<P> parse beside qd_gnn7's and qd_gnn10's options; repeats, ranges and non-QD uses refused")
    # the model: unfitted it scores z; at init it is QD10 to rounding; fitted, the frontier cut is exact and gradients flow
    Q = QG.synthetic_rows(24, rng, K)
    TOK = [QG.edge_labels(q) for q in Q]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    phi = rng.standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32)
    rows = list(range(24))
    P = QG.pack_qd(rows, Q, TOK, z_of, K)
    for kind in ("T0", "TXT"):
        for L in (1, 2, 3):
            for kw in ({"at": True}, {"rg": 3}, {"at": True, "rg": 2}, {"at": True, "rg": 2, "lq": kind == "TXT", "g1": True}):
                torch.manual_seed(1)
                m13 = QD13(kind, 16, L, K, **kw)
                m13.set_phi(phi, K)
                assert torch.equal(m13(P), P.z), f"{kind} L{L} {kw}: an unfitted QD13 does not score z"
                ref = Q10.QD10(kind, 16, L, K, g1=kw.get("g1", False), lq=kw.get("lq", False))
                ref.load_state_dict(m13.state_dict(), strict=False)
                ref.set_phi(phi, K)
                with torch.no_grad():
                    torch.manual_seed(2)
                    for nm_, p in ref.named_parameters():
                        p.add_(0.3 * torch.randn_like(p))
                    m13.load_state_dict(ref.state_dict(), strict=False)
                    err0 = float((m13(P) - ref(P)).abs().max())
                    assert err0 < 1e-5, f"{kind} L{L} {kw}: at init QD13 is not QD10 ({err0})"
                    for nm_, p in m13.named_parameters():
                        p.add_(0.3 * torch.randn_like(p))
                s_full = m13(P)
                Qc = []
                for q in Q:
                    mk = QG.frontier_mask(q["u"], q["v"], q["seeds"], q["n"], L)
                    qq = {**q, **{k: q[k][mk] for k in ("u", "v", "fam", "fwd", "bwd", "w")}}
                    keep_struct = mk[q["fam"] == 0]
                    qq["w2_f"], qq["w2_b"] = q["w2_f"][keep_struct], q["w2_b"][keep_struct]
                    Qc.append(qq)
                TOKc = [QG.edge_labels(q) for q in Qc]
                with torch.no_grad():
                    s_cut = m13(QG.pack_qd(rows, Qc, TOKc, z_of, K))
                err = float((s_full - s_cut).abs().max())
                assert err < 1e-5, f"{kind} L{L} {kw}: the frontier cut changes the scores by {err}"
                s_pad, gold, z = QG.padded(P, s_full)
                loss = QG.step_loss(s_pad, gold, z, "np")
                loss.backward()
                for nm_, p in m13.named_parameters():
                    assert p.grad is None or bool(torch.isfinite(p.grad).all()), f"{kind} L{L} {kw}: {nm_} grad not finite"
                if kw.get("at"):
                    assert float(m13.a_at.grad.abs().sum()) > 0, "no gradient reaches a_at"
                if kw.get("rg"):
                    assert float(m13.D.grad.abs().sum()) > 0, "no gradient reaches D"
                print(f"selftest QD13 {kind} L{L} {kw}: z unfitted, QD10 at init ({err0:.1e}), frontier cut exact ({err:.1e}), "
                      f"finite gradients; params {sum(p.numel() for p in m13.parameters())}")
    # attention is mean one over a node's active in-edges, and a node whose in-edges all lack a state gets none
    torch.manual_seed(3)
    ma = QD13("T0", 16, 1, K, at=True)
    with torch.no_grad():
        ma.a_at.normal_(0, 1.0)
        qn = ma.lnq(ma.Wq(P.qemb))
        h = torch.zeros(P.N, 16).index_add(0, P.seed_idx, qn[P.seed_row])
        active = (h != 0).any(-1).float()
        n_in = torch.zeros(P.N).index_add(0, P.dst, active[P.src])
        al = ma.attention(0, QG.rms(h), ma.relations(P), qn, P, active, n_in)
        sums = torch.zeros(P.N).index_add(0, P.dst, al)
        assert torch.allclose(sums, n_in, atol=1e-4) and bool((al[active[P.src] == 0] == 0).all())
        assert float(al.std()) > 0.05, "attention stayed uniform"
    print(f"selftest: alpha sums to n_v over the active in-edges (spread {float(al.std()):.2f}), 0 on the rest")
    # the arms: no option = qd_gnn10's arm, draw for draw; -fd drops whole families in training only
    G = {"g": {"phi": phi, "kind": "passage"}}
    bind()
    for nm in ("QD-TXT-L2-d16-K8-dr25e10-tb5-ed50", "QD-T0-L2-d16", "QD-TXT-L2-d16-K8-pc"):
        sp = Q2.parse_arms(nm, ["g"], "np", AU, AU.AG, AU.G3)[nm]
        a10, a13 = Q10.QDArm10(sp, Q, TOK, G, z_of, A16, 0), QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
        assert type(a13) is QDArm13
        m10, e10, c10, b10 = Q12._FIT2(a10, ["g"], {"g": rows[:16]}, {"g": rows[16:]}, "both", 2, 2e-3, 1e-4, 0, "graph", AU)
        m13_, e13, c13, b13 = Q12._FIT2(a13, ["g"], {"g": rows[:16]}, {"g": rows[16:]}, "both", 2, 2e-3, 1e-4, 0, "graph", AU)
        assert (c10, e10, b10) == (c13, e13, b13) and type(m10) is type(m13_), nm
        assert all(torch.equal(x, y) for x, y in zip(m10.state_dict().values(), m13_.state_dict().values())), nm
    for nm in ("W3-TXT-L2-d16-K8", "W4-T0-L2-d16"):
        spw = Q2.parse_arms(nm, ["g"], "np", AU, AU.AG, AU.G3)[nm]
        aw = QG.QDArm(spw, Q, TOK, G, z_of, A16, 0)
        assert type(aw) is (W4Arm13 if spw.get("w4") else W3Arm13), nm
    print("selftest: with no qd_gnn13 option QDArm13 fits as qd_gnn10's arm, curve and weights bit for bit; W3 / W4 arms map")
    spf = Q2.parse_arms("QD-T0-L2-d16-fd60", ["g"], "np", AU, AU.AG, AU.G3)["QD-T0-L2-d16-fd60"]
    af = QG.QDArm(spf, Q, TOK, G, z_of, A16, 0)
    torch.manual_seed(6)
    mf = af.make()
    assert type(mf) is QG.QD and af.rng_fd is not None
    seen = []
    real_padded = QG.padded
    QG.padded = lambda P_, s_: (seen.append((P_.e_row.clone(), P_.fam.clone())), real_padded(P_, s_))[1]
    try:
        mf.train()
        af.forward(mf, "g", rows)
        mf.eval()
        af.forward(mf, "g", rows)
    finally:
        QG.padded = real_padded
    full = QG.pack_qd(rows, Q, TOK, z_of, 0)
    (er_t, fa_t), (er_r, fa_r) = seen
    assert er_r.numel() == full.src.numel(), "a read must see every edge"
    dropped_any = False
    for b in range(len(rows)):
        had = set(full.fam[full.e_row == b].clamp(max=2).tolist())
        kept = fa_t[er_t == b].clamp(max=2)
        kept_set = set(kept.tolist())
        assert kept_set and kept_set <= had, b
        for f_ in kept_set:      # a kept family keeps every edge it had
            assert int((kept == f_).sum()) == int(((full.e_row == b) & (full.fam.clamp(max=2) == f_)).sum()), b
        if 0 in had and len(kept_set) == 1 and len(had) > 1:
            assert kept_set == {0} or kept_set != had
        dropped_any |= kept_set != had
    assert dropped_any
    print("selftest: -fd60 drops whole families per row in training (never all of a row's edges), none in reads")
    # end to end, each in a fresh process: no qd_gnn13 option runs as qd_gnn12 bit for bit; the new options run through
    with tempfile.TemporaryDirectory() as d:
        res = {}
        for mode, script in (("plain12", Q12.__file__), ("plain", __file__), ("new", __file__)):
            out = Path(d) / f"{mode}.json"
            r = subprocess.run([sys.executable, script, "--selftest-run", "plain" if mode == "plain12" else mode, str(out)],
                               env=dict(os.environ), capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                print(r.stdout[-3000:], r.stderr[-3000:])
                raise AssertionError(f"selftest run {mode} failed")
            res[mode] = json.loads(out.read_text(encoding="utf-8"))

        def reads(r):
            return {a: {s: sv["reads"] for s, sv in v["seeds_read"].items()} for a, v in r["results"].items()}

        def states(mode):
            return [torch.load(f, weights_only=False)["state_dict"] for f in sorted((Path(d) / f"{mode}_models").glob("a*_s*.pt"))]

        def same(a, b):
            return len(a) == len(b) and all(x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x) for x, y in zip(a, b))
        assert reads(res["plain"]) == reads(res["plain12"]), "with no qd_gnn13 option the run must read as qd_gnn12's"
        assert same(states("plain"), states("plain12")) and len(states("plain")) == 2
        assert res["plain"]["look"] == "qd_gnn13" and res["plain"]["pins"]["qd_gnn12"] == SHAS["qd_gnn12"]
        assert set(res["new"]["results"]) == set(ARMS_NEW.split(",")), set(res["new"]["results"])
        st = states("new")
        assert len(st) == 3 and any("a_at" in s_ for s_ in st) and any("D" in s_ for s_ in st)
        a0 = ARMS_NEW.split(",")[0]
        print(f"selftest e2e: no option runs as qd_gnn12 bit for bit (reads and saved models); the new arms run "
              f"({a0} ID/np {res['new']['results'][a0]['seeds_read']['0']['reads']['metaqa']['ID/np']['fit']})")
    print("selftest ok")


def selftest_run(mode, out):
    """One end-to-end run on qd_gnn11's synthetic KB (a subprocess, so every bind starts fresh)."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    phi = np.random.default_rng(7).standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32)
    names = [f"rel {j}" for j in range(Q11.K_ST)]

    def fake(trains, reads, max_len, t0, AU_):
        Qf, part = Q11.synthetic()
        Gf = {"metaqa": {"kind": "kb", "role": "train", "phi": phi, "phi_sha": "synthetic", "n_rel": Q11.K_ST, "vocab": names,
                         "info": {"pruned_loader": {"synthetic": True}}}}
        return Qf, part, Gf

    Q4._load_all = fake
    sys.argv = ["qd_gnn13.py", "--train", "metaqa=fit", "--arms", Q11.ARMS_ST if mode == "plain" else ARMS_NEW, "--rule", "np",
                "--epochs", "2", "--out", out, "--kalpha", "4,8", "--kalpha-draws", "2"]
    main(selftest_run=True)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--selftest-run":
        selftest_run(sys.argv[2], sys.argv[3])
    else:
        main()
