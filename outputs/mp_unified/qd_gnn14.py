"""Design look (untracked; not a result and not filed): qd_gnn9.py, pinned and unchanged, with qd_gnn13's edge-family
dropout on its QDEM arms.

em9-2w-c's QDEM-z8-L2-dr25e10-tb2-ed50-bal10 is the most graceful EM arm so far (its types label-free, its no-label
read equal to the labelled one, only metaqa below the twin on R@5 off 2wiki), and rg-2w's -fd25 on a QD text arm is
the first one-graph message-passing arm with no unseen graph significantly below the twin on R@5. This look puts the
two on one arm:

  -fd<P>  in training, after qd_gnn8's label and edge dropout, each batch row drops each edge family it has (structural,
          NER, kNN) with probability P / 100, keeping its structural family (or, with none, its first family) when every
          family it has would go: qd_gnn13.family_drop, draw for draw, here also cutting the EM step's tensors (the true
          label, the degrees, the on-path target and the frontier mask) alike, as qd_gnn8.edge_drop8 does. The on-path
          target is computed on the full graph before any drop, as for the edge dropout. Reads see every edge.

The family generator is drawn in make() after qd_gnn8's generators and only on an -fd arm, so an arm without -fd is
qd_gnn9's, draw for draw (--selftest checks the fit bit for bit, and a whole run in fresh processes). QDEM is message
passing.

    python outputs/mp_unified/qd_gnn14.py --selftest
    python outputs/mp_unified/qd_gnn14.py --train 2wiki=x4 --read hotpotqa,musique,metaqa,webqsp --rule both \
        --kalpha 256 --arms QDEM-z8-L2-dr25e10-tb2-ed50-bal10-fd25,QDEM-z8-L2-dr25e10-tb5-ed50-bal10-fd25 \
        --out outputs/mp_unified/qd/em14-2w.json
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn9 as Q9  # noqa: E402  (imports qd_gnn8 ... qd_gnn, which sets the BLAS thread counts before numpy loads)

import numpy as np  # noqa: E402
import torch  # noqa: E402

Q8, Q7, Q6x, Q5, Q4, Q2, QG = Q9.Q8, Q9.Q7, Q9.Q6x, Q9.Q5, Q9.Q4, Q9.Q2, Q9.QG
SHAS = {**Q9.SHAS, "qd_gnn14": QG.sha(__file__)}   # at import: what ran
OPT14 = re.compile(r"fd(\d+)")
EDGE_KEYS = ("src", "dst", "e_row", "d", "a", "fam", "w")
log = QG.log
_PARSE9 = Q9.parse_em9
_QDEMArm8 = Q8.QDEMArm


# ── arm names ────────────────────────────────────────────────────────────────


def parse_em14(nm):
    """qd_gnn9.parse_em9 on the name without -fd<P>, then -fd<P> (QDEM arms only)."""
    m = Q8.NAMERE.fullmatch(nm)
    if not m:
        return _PARSE9(nm)      # refuses it with qd_gnn8's message
    keep, fd = [], None
    for tok in [t for t in m.group(5).split("-") if t]:
        mo = OPT14.fullmatch(tok)
        if not mo:
            keep.append(tok)
            continue
        if fd is not None:
            raise SystemExit(f"{nm}: -fd given twice")
        n = int(mo.group(1))
        if not 0 < n < 100:
            raise SystemExit(f"{nm}: -fd<P> takes 0 < P < 100")
        fd = n / 100.0
    sp = _PARSE9(nm[:m.start(5)] + "".join("-" + t for t in keep))
    if fd is not None:
        if sp["em8"]["arm"] != "qdem":
            raise SystemExit(f"{nm}: -fd is for QDEM arms only")
        sp["fd14"] = fd
    return sp


# ── the family dropout ───────────────────────────────────────────────────────


def family_drop14(P, rng, p):
    """qd_gnn13.family_drop, draw for draw, cutting qd_gnn8's EM tensors alike. In place; returns how many edges went."""
    fam = np.minimum(P.fam.numpy(), 2)
    row = P.e_row.numpy()
    has = np.zeros((P.B, 3), dtype=bool)
    has[row, fam] = True
    drop = (rng.random((P.B, 3)) < p) & has
    for b in np.flatnonzero(has.any(1) & (drop == has).all(1)):
        drop[b, 0 if has[b, 0] else int(np.flatnonzero(has[b])[0])] = False
    kill = drop[row, fam]
    keep = torch.from_numpy(~kill)
    for k in EDGE_KEYS + Q8.EXTRA:
        x = getattr(P, k, None)
        if x is not None:
            setattr(P, k, x[keep])
    return int(kill.sum())


# ── the arm ──────────────────────────────────────────────────────────────────


class QDEMArm14(_QDEMArm8):
    """qd_gnn8.QDEMArm with -fd; without it, the same arm call for call."""

    def __init__(self, sp, Q, TOK, G, z_of, A16, n_id):
        super().__init__(sp, Q, TOK, G, z_of, A16, n_id)
        self.rng_fd = None

    def make(self):
        model = super().make()
        self.rng_fd = (np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
                       if self.sp.get("fd14") else None)
        return model

    def forward(self, model, g, rows):
        fd = self.sp.get("fd14")
        if not fd:
            return super().forward(model, g, rows)
        # qd_gnn8.QDEMArm.forward, step for step, with the family dropout after the edge dropout
        o = self.sp.get("t7") or {}
        Qp, Tp = self.rows_for(rows)
        P = QG.pack_qd(rows, Qp, Tp, self.z_of, self.sp["K"], self.nr)
        P.a_true = P.a.clone()
        P.du = P.dv = None
        if self.em["deg"]:
            P.du = torch.bincount(P.src, minlength=P.N).to(torch.float32)[P.src]
            P.dv = torch.bincount(P.dst, minlength=P.N).to(torch.float32)[P.dst]
        training = model.training
        if training and self.em["em"]:
            P.em_on, P.em_mask = Q8.em_targets(self, rows, Qp)
        if training and self.rng is not None:     # qd_gnn7.QDArm7.forward's label dropout, draw for draw
            drop = self.rng.random(P.B) < self.p_row
            de = drop[P.e_row.numpy()] | (self.rng.random(P.a.numel()) < self.p_edge)
            P.a = torch.where(torch.from_numpy(de), torch.full_like(P.a, -1), P.a)
        if training and self.rng_ed is not None:
            Q8.edge_drop8(P, self.rng_ed, o["ed"])
        if training:
            family_drop14(P, self.rng_fd, fd)
        s = model(P)
        if training:
            self.emc.trained = True
            if self.em["em"]:
                Q8.PENDING["aux"] = Q8.em_step(self, model, g, P)
        return QG.padded(P, s)


# ── binding and output ───────────────────────────────────────────────────────


def bind():
    Q9.bind()
    Q8.parse_em = parse_em14     # qd_gnn8.parse_arms reads its module's parse_em
    Q8.QDEMArm = QDEMArm14       # qd_gnn8.QDArm8.__new__ builds a QDEM arm from its module's QDEMArm


def finish(out):
    Q9.finish(out)
    res = json.loads(out.read_text(encoding="utf-8"))
    res["look"] = "qd_gnn14"
    res["pins"]["qd_gnn9"] = SHAS["qd_gnn9"]
    res["qd_gnn14_sha256"] = SHAS["qd_gnn14"]
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    Q7.bind = bind    # qd_gnn7.main binds through its module's bind (and hands it to qd_gnn5)
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--epochs" in sys.argv:
        Q8.EMCFG["epochs"] = int(sys.argv[sys.argv.index("--epochs") + 1])
    Q7.main()
    if Q6x.STATE["out"] is not None:
        finish(Q6x.STATE["out"])


# ── selftest (synthetic rows; laptop) ────────────────────────────────────────


K_ST = 8
ARMS_PLAIN = "QDEM-z4-L2-d16-dr25e10-tb2-ed50-bal10"
ARMS_NEW = "QDEM-z4-L2-d16-dr25e10-tb2-ed50-bal10-fd50,QDEM-z4-L2-d16-fd30"


def selftest():
    import os
    import subprocess
    import tempfile
    sys.path.insert(0, str(QG.AU_DIR))
    sys.path.insert(0, str(QG.KB_DIR))
    import anchor_univ as AU
    import qd_gnn13 as Q13     # the QD arms' family dropout, the reference here (laptop only)
    A16 = AU.A16
    bind()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    K = 8
    # names: -fd anywhere among the options; without it parse_em9's spec; repeats, W3EM and ranges refused
    base = "QDEM-z8-L2-dr25e10-tb2-ed50-bal10"
    want = {**_PARSE9(base), "fd14": 0.25}
    assert parse_em14(base + "-fd25") == want and parse_em14("QDEM-z8-L2-fd25-dr25e10-tb2-ed50-bal10") == want
    for nm in (base, "QDEM-z4-L2-d16", "W3EM-z8-L2-pu-bal10", "QDEM-z8-L2-tb5-ed50-pu-lmin20-bal10-sk-deg"):
        assert parse_em14(nm) == _PARSE9(nm), nm
    for bad in ("QDEM-z8-fd25-fd30", "W3EM-z8-L2-fd25", "QDEM-z8-fd0", "QDEM-z8-fd100", "QDEM-z8-fdx", "QDEM-noem-pu-fd20"):
        try:
            parse_em14(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
    assert Q8.parse_em is parse_em14 and Q8.QDEMArm is QDEMArm14
    print("selftest: -fd<P> parses beside qd_gnn8's and qd_gnn9's options; repeats, W3EM arms and ranges are refused")
    # family_drop14 is qd_gnn13.family_drop draw for draw, and cuts the EM tensors alike
    Q = QG.synthetic_rows(60, rng, K)
    TOK = [QG.edge_labels(q) for q in Q]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    for p in (0.25, 0.6, 0.95):
        P13 = QG.pack_qd(list(range(24)), Q, TOK, z_of, K)
        P14 = QG.pack_qd(list(range(24)), Q, TOK, z_of, K)
        full = {k: getattr(P14, k).clone() for k in EDGE_KEYS}
        ne = P14.src.numel()
        P14.a_true = torch.arange(ne)                       # an edge id: what survives, and in what order
        P14.em_on = torch.from_numpy(rng.random(ne) < 0.1)
        P14.em_mask = torch.from_numpy(rng.random(ne) < 0.7)
        on0, mask0 = P14.em_on.clone(), P14.em_mask.clone()
        P14.du = P14.dv = None
        n13 = Q13.family_drop(P13, np.random.default_rng(9), p)
        n14 = family_drop14(P14, np.random.default_rng(9), p)
        assert n13 == n14 > 0 and all(torch.equal(getattr(P13, k), getattr(P14, k)) for k in EDGE_KEYS), p
        kid = P14.a_true
        assert all(torch.equal(getattr(P14, k), full[k][kid]) for k in EDGE_KEYS)
        assert torch.equal(P14.em_on, on0[kid]) and torch.equal(P14.em_mask, mask0[kid]) and P14.du is None
    print("selftest: family_drop14 keeps qd_gnn13.family_drop's edges draw for draw and cuts the EM tensors alike")
    # the arms: without -fd QDEMArm14 fits as qd_gnn8's QDEMArm under qd_gnn9's EM, bit for bit; -fd fits reproducibly
    G = {"g": {"phi": rng.standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32), "kind": "passage"}}
    rows = list(range(60))
    tr, sel = {"g": rows[:40]}, {"g": rows[40:]}
    Q7.STATE7["B"]["g"] = rows[40:]

    def fit(cls, nm):
        sp = parse_em14(nm)
        Q5.NAMES[id(sp)] = nm
        torch.manual_seed(0)
        a_ = cls(sp, Q, TOK, G, z_of, A16, 0)
        assert type(a_) is cls
        m_, e_, c_, _x = Q2.fit_shared2(a_, ["g"], tr, sel, "both", 3, 2e-3, 1e-4, 0, "graph", AU)
        return a_, m_, e_, c_

    def same_state(m0, m1):
        s0, s1 = m0.state_dict(), m1.state_dict()
        return s0.keys() == s1.keys() and all(torch.equal(s0[k], s1[k]) for k in s0)

    for nm in ("QDEM-z4-L2-d16-dr25e10-tb2-ed50-bal10", "QDEM-z4-L2-d16-deg-pu-sk"):
        a8, m8, e8, c8 = fit(_QDEMArm8, nm)
        a14, m14, e14, c14 = fit(QDEMArm14, nm)
        assert (e8, c8) == (e14, c14) and same_state(m8, m14) and a14.rng_fd is None, nm
    print("selftest: without -fd QDEMArm14 fits as qd_gnn8's QDEMArm (qd_gnn9's EM), curve and weights bit for bit")
    nmf = "QDEM-z4-L2-d16-dr25e10-tb2-ed50-bal10-fd50"
    af0, mf0, ef0, cf0 = fit(QDEMArm14, nmf)
    af1, mf1, ef1, cf1 = fit(QDEMArm14, nmf)
    assert (ef0, cf0) == (ef1, cf1) and same_state(mf0, mf1) and af0.rng_fd is not None, "an -fd fit is not reproducible"
    assert not same_state(mf0, m14), "-fd50 changed nothing"
    print(f"selftest: an -fd50 fit is reproducible ({cf0}) and differs from the arm without it")
    # one training step: whole families go per row (never all of a row's edges), the EM step sees the cut tensors;
    # a read sees every edge
    sp = parse_em14("QDEM-z4-L2-d16-fd60")
    Q5.NAMES[id(sp)] = "QDEM-z4-L2-d16-fd60"
    torch.manual_seed(6)
    af = QDEMArm14(sp, Q, TOK, G, z_of, A16, 0)
    mf = af.make()
    af.prepare(mf, "g")
    seen, step_seen = [], []
    real_padded, real_step = QG.padded, Q8.em_step
    QG.padded = lambda P_, s_: (seen.append((P_.e_row.clone(), P_.fam.clone(), P_.src.numel())), real_padded(P_, s_))[1]
    Q8.em_step = lambda arm_, model_, g_, E_: (step_seen.append(E_), real_step(arm_, model_, g_, E_))[1]
    try:
        rr = rows[:24]
        mf.train()
        af.forward(mf, "g", rr)
        aux = Q8.PENDING.pop("aux")
        mf.eval()
        with torch.no_grad():
            af.forward(mf, "g", rr)
    finally:
        QG.padded, Q8.em_step = real_padded, real_step
    assert aux is not None and bool(torch.isfinite(aux)), aux
    Qp, Tp = af.rows_for(rr)
    fullP = QG.pack_qd(rr, Qp, Tp, af.z_of, af.sp["K"], af.nr)
    (er_t, fa_t, n_t), (er_r, fa_r, n_r) = seen
    assert n_r == fullP.src.numel() > n_t, (n_r, n_t)
    E = step_seen[0]
    assert all(getattr(E, k).numel() == n_t for k in ("a_true", "em_on", "em_mask")), "the EM tensors were not cut alike"
    dropped_any = False
    for b in range(len(rr)):
        had = set(fullP.fam[fullP.e_row == b].clamp(max=2).tolist())
        kept = fa_t[er_t == b].clamp(max=2)
        kept_set = set(kept.tolist())
        assert kept_set and kept_set <= had, b
        for f_ in kept_set:      # a kept family keeps every edge it had
            assert int((kept == f_).sum()) == int(((fullP.e_row == b) & (fullP.fam.clamp(max=2) == f_)).sum()), b
        dropped_any |= kept_set != had
    assert dropped_any
    print(f"selftest: -fd60 drops whole families per row in training ({n_r - n_t} of {n_r} edges, never all of a row's), "
          f"the EM step runs on the cut tensors (loss {float(aux):.4f}), a read sees every edge")
    # end to end, each in a fresh process: no -fd runs as qd_gnn9 bit for bit; the -fd arms run through
    with tempfile.TemporaryDirectory() as d:
        res = {}
        for mode in ("ref9", "plain", "new"):
            out = Path(d) / f"{mode}.json"
            r = subprocess.run([sys.executable, __file__, "--selftest-run", mode, str(out)], env=dict(os.environ),
                               capture_output=True, text=True, timeout=1800)
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
        assert reads(res["plain"]) == reads(res["ref9"]), "with no -fd the run must read as qd_gnn9's"
        assert same(states("plain"), states("ref9")) and len(states("plain")) == 1
        assert res["ref9"]["look"] == "qd_gnn9" and res["plain"]["look"] == "qd_gnn14"
        assert res["plain"]["pins"]["qd_gnn9"] == SHAS["qd_gnn9"] and res["plain"]["qd_gnn14_sha256"] == SHAS["qd_gnn14"]
        assert res["plain"]["em"]["trace"].keys() == res["ref9"]["em"]["trace"].keys()
        assert set(res["new"]["results"]) == set(ARMS_NEW.split(",")), set(res["new"]["results"])
        for a in ARMS_NEW.split(","):
            assert res["new"]["results"][a]["spec"]["fd14"] in (0.5, 0.3) and a in res["new"]["em"]["trace"], a
        assert len(states("new")) == 2 and not same(states("new")[:1], states("plain"))
        a0 = ARMS_NEW.split(",")[0]
        print(f"selftest e2e: no -fd runs as qd_gnn9 bit for bit (reads and saved models); the -fd arms run "
              f"({a0} ID/np {res['new']['results'][a0]['seeds_read']['0']['reads']['metaqa']['ID/np']['fit']})")
    print("selftest: all checks passed")


def selftest_run(mode, out):
    """One end-to-end run on qd_gnn11's synthetic KB, built here (a subprocess, so every bind starts fresh). ref9 runs
    qd_gnn9's main; plain and new run this module's."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    phi = np.random.default_rng(7).standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32)
    names = [f"rel {j}" for j in range(K_ST)]

    def fake(trains, reads, max_len, t0, AU_):
        Qf = Q6x.synthetic_kb(80, np.random.default_rng(5), K_ST)
        for j, q in enumerate(Qf):
            q["pool"] = np.arange(q["n"]) + 1000 * j
            q["score"] = q["score"].astype(np.float32)
        part = {("metaqa", "x1"): list(range(0, 40)), ("metaqa", "select"): list(range(40, 52)),
                ("metaqa", "fit"): list(range(52, 80))}
        Gf = {"metaqa": {"kind": "kb", "role": "train", "phi": phi, "phi_sha": "synthetic", "n_rel": K_ST, "vocab": names,
                         "info": {"pruned_loader": {"synthetic": True}}}}
        return Qf, part, Gf

    Q4._load_all = fake
    sys.argv = ["qd_gnn14.py", "--train", "metaqa=fit", "--arms", ARMS_NEW if mode == "new" else ARMS_PLAIN, "--rule",
                "np", "--epochs", "2", "--out", out, "--kalpha", "4,8", "--kalpha-draws", "2"]
    if mode == "ref9":
        Q9.main()
    else:
        main()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--selftest-run":
        selftest_run(sys.argv[2], sys.argv[3])
    else:
        main()
