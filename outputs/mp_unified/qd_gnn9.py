"""Design look (untracked; not a result and not filed): qd_gnn8.py, pinned and unchanged, with four options on its EM
against the collapse em-2w showed. There, once lambda reached 0 the posterior followed the prior: 2-3 of 8 types held
the frontier edges, the NMI between labels and types fell from 0.41-0.52 to 0 within two epochs, and the query's edge
ranking read AUC 0.79 against the query-blind 0.78 on 2wiki (0.54 against 0.55 on hotpotqa). Only 0.33% of frontier
edges are on a path, so the evidence moves little and the M step pulls every prior toward the dominant types.

-lmin<x>  a floor on lambda: lambda_t = max(x / 100, lambda0 max(0, 1 - ep / E_an)) (x / 100 <= lambda0). The labels
          stay a weak anchor in the E step while the evidence moves the types off them where it disagrees. A held,
          unseen, dropped or NR label is still no label.
-pu       positive-unlabelled evidence: in the E step only the on-path edges carry Bern(o_e = 1 | sigmoid(rho(q, z)))^mu;
          an off-path edge is unlabelled (it may be of a relevant type that this row's walk did not use), so its
          posterior is its prior times the label term. The M step's balanced relevance loss is unchanged.
-bal<x>   information maximisation in the M step (RIM, Krause et al. 2010): beta = x / 10,
              loss += beta (log k - H(mean_e pi_e))     over the batch's frontier edges
          so the prior's marginal over types stays spread while each edge's prior follows its posterior.
-sk       a balanced E step (SwAV's Sinkhorn-Knopp equipartition, SK_ITERS iterations, over the batch's frontier
          edges): q = diag(r) exp(post) diag(c), each row sums to 1, each type's column to about n / k.

The per-epoch em trace adds the type usage of the posterior, the prior and the on-path posterior, each with its
perplexity (exp of the entropy: k when every type is used alike, 1 at collapse).

    python outputs/mp_unified/qd_gnn9.py --selftest
    python outputs/mp_unified/qd_gnn9.py --train 2wiki=x4 --read hotpotqa,musique,metaqa,webqsp --rule both \
        --arms QDEM-z8-L2-tb5-ed50-pu-lmin20-bal10 --out outputs/mp_unified/qd/em9-2w.json
"""
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn8 as Q8  # noqa: E402  (imports qd_gnn7 ... qd_gnn, which sets the BLAS thread counts before numpy loads)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

Q7, Q6x, Q5 = Q8.Q7, Q8.Q6x, Q8.Q5
Q4, Q2, QG = Q8.Q4, Q8.Q2, Q8.QG
SHAS = {**Q8.SHAS, "qd_gnn9": QG.sha(__file__)}   # at import: what ran
OPT9 = re.compile(r"(lmin|bal)(\d+)|(pu|sk)")
SK_ITERS = 3
log = QG.log
_PARSE8 = Q8.parse_em


# ── arm names ────────────────────────────────────────────────────────────────


def parse_em9(nm):
    """qd_gnn8.parse_em on the name without qd_gnn9's options, then those options."""
    m = Q8.NAMERE.fullmatch(nm)
    if not m:
        return _PARSE8(nm)      # refuses it with qd_gnn8's message
    keep, o9, seen = [], {"lmin": 0.0, "bal": 0.0, "pu": False, "sk": False}, set()
    for tok in [t for t in m.group(5).split("-") if t]:
        mo = OPT9.fullmatch(tok)
        if not mo:
            keep.append(tok)
            continue
        key = mo.group(1) or mo.group(3)
        if key in seen:
            raise SystemExit(f"{nm}: -{key} given twice")
        seen.add(key)
        if mo.group(1):
            n = int(mo.group(2))
            if n <= 0:
                raise SystemExit(f"{nm}: -{key} above 0")
            o9[key] = n / (100.0 if key == "lmin" else 10.0)
        else:
            o9[key] = True
    base = nm[:m.start(5)] + "".join("-" + t for t in keep)
    sp = _PARSE8(base)
    o = sp["em8"]
    if seen and not o["em"]:
        raise SystemExit(f"{nm}: -noem takes no EM option")
    if o9["lmin"] > o["lam"]:
        raise SystemExit(f"{nm}: -lmin above lambda0 ({o['lam']})")
    o.update(o9)
    return sp


# ── the EM ───────────────────────────────────────────────────────────────────


def perplexity(u):
    p = u / u.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()))


class EMCore9(Q8.EMCore):
    """qd_gnn8.EMCore with lambda's floor and the type usage in the trace."""

    def lam(self, o):
        return max(super().lam(o), o.get("lmin", 0.0))

    def add(self, g, q, lp, on, at):
        super().add(g, q, lp, on, at)
        A = self.acc
        A["u_post"] = A.get("u_post", 0.0) + q.sum(0).double()
        A["u_prior"] = A.get("u_prior", 0.0) + lp.exp().sum(0).double()
        ob = on > 0
        if bool(ob.any()):
            A["u_on"] = A.get("u_on", 0.0) + q[ob].sum(0).double()

    def tick(self, o):
        if not self.trained:
            return
        extra = {}
        for key in ("u_post", "u_prior", "u_on"):
            u = self.acc.get(key)
            if torch.is_tensor(u) and float(u.sum()) > 0:
                extra["usage_" + key[2:]] = [round(float(x), 4) for x in u / u.sum()]
                extra["perplexity_" + key[2:]] = round(perplexity(u), 3)
        n0 = len(self.trace)
        super().tick(o)
        if len(self.trace) > n0 and extra:
            self.trace[-1].update(extra)
            log(f"  em ep {self.trace[-1]['epoch']}: type perplexity post {extra.get('perplexity_post')} prior "
                f"{extra.get('perplexity_prior')} on-path {extra.get('perplexity_on')} (of {self.k})")


def sinkhorn(post, iters=SK_ITERS):
    """SwAV's equipartition of exp(post): rows sum to 1 and each type's column to about n / k."""
    n, k = post.shape
    Q = torch.exp(post - post.max())
    Q = Q / Q.sum()
    for _ in range(iters):
        Q = Q / Q.sum(0, keepdim=True).clamp_min(1e-30) / k
        Q = Q / Q.sum(1, keepdim=True).clamp_min(1e-30) / n
    return Q * n


def em_step9(arm, model, g, E):
    """qd_gnn8.em_step with -pu, -sk and -bal; without them, its computation step for step."""
    o = arm.em
    if E is None or not bool(E.em_mask.any()):
        return None
    ty, m = model.typer, E.em_mask
    du = E.du[m] if E.du is not None else None
    dv = E.dv[m] if E.dv is not None else None
    lp = F.log_softmax(ty.logits(E.d[m], E.a[m], E.w[m], du, dv), 1)
    rl = ty.rel(E.qemb)[E.e_row[m]]
    on = E.em_on[m].to(lp.dtype)
    n, npos = on.numel(), float(on.sum())
    lpos, lneg = on[:, None] * F.logsigmoid(rl), (1.0 - on)[:, None] * F.logsigmoid(-rl)
    if 0 < npos < n:    # the M step's relevance loss: on- and off-path edges at half the batch each
        ll = lpos * (0.5 * n / npos) + lneg * (0.5 * n / (n - npos))
    else:
        ll = torch.zeros_like(lpos)
    at = E.a_true[m]
    with torch.no_grad():
        ev = lpos if o.get("pu") else lpos + lneg     # -pu: an off-path edge carries no evidence
        post = lp + o["mu"] * ev
        lam = arm.emc.lam(o)
        has = at >= 0
        if lam > 0 and bool(has.any()):
            post[has] = post[has] + lam * arm.emc.log_theta(g)[:, at[has]].T
        q = sinkhorn(post) if o.get("sk") else torch.softmax(post, 1)
        if bool(has.any()):
            arm.emc.update(g, q[has], at[has])
        arm.emc.add(g, q, lp, on, at)
    loss = o["om"] * (-(q * lp).sum(1).mean() - (q * ll).sum(1).mean())
    if o.get("bal"):
        pbar = lp.exp().mean(0)
        loss = loss + o["bal"] * (math.log(arm.k) + (pbar * torch.log(pbar.clamp_min(1e-12))).sum())
    return loss


# ── binding and output ───────────────────────────────────────────────────────


def bind():
    Q8.bind()
    Q8.parse_em = parse_em9     # qd_gnn8.parse_arms reads its module's parse_em
    Q8.EMCore = EMCore9         # the EM arms build their EMCore from qd_gnn8's module
    Q8.em_step = em_step9       # and call its em_step


def finish(out):
    Q8.finish(out)
    res = json.loads(out.read_text(encoding="utf-8"))
    res["look"] = "qd_gnn9"
    res["pins"]["qd_gnn8"] = SHAS["qd_gnn8"]
    res["qd_gnn9_sha256"] = SHAS["qd_gnn9"]
    res["em"]["sk_iters"] = SK_ITERS
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


# ── selftest (synthetic rows; laptop) ────────────────────────────────────────


def selftest():
    sys.path.insert(0, str(QG.AU_DIR))
    sys.path.insert(0, str(QG.KB_DIR))
    import anchor_univ as AU
    A16 = AU.A16
    bind()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    K = 8
    # names: qd_gnn8's parse, plus the four options; misplaced, repeated and out-of-range ones refused
    sp = parse_em9("QDEM-z8-L2-tb5-ed50-pu-lmin20-bal10-sk-deg")
    o = sp["em8"]
    assert (o["pu"], o["sk"], o["lmin"], o["bal"], o["deg"], sp["t7"]) == (True, True, 0.2, 1.0, True, {"tb": 0.5, "ed": 0.5})
    o = parse_em9("W3EM-z8-L2")["em8"]
    assert (o["pu"], o["sk"], o["lmin"], o["bal"]) == (False, False, 0.0, 0.0)
    assert parse_em9("W3EM-z4-L3-lg-ki30-pu")["em8"]["ki"] == 3.0
    for bad in ("QDEM-z8-pu-pu", "QDEM-noem-pu", "W3EM-noem-bal10", "QDEM-lam5-lmin60", "QDEM-lmin0", "QDEM-z8-foo",
                "QDEM-bal0", "W3EM-tb5-pu"):
        try:
            parse_em9(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
    assert Q8.parse_arms is not None and Q8.parse_em is parse_em9
    print("selftest: qd_gnn9's options parse beside qd_gnn8's; repeated, misplaced and out-of-range ones are refused")
    # lambda's floor
    emc = EMCore9(4)
    lams = []
    for ep in range(5):
        emc.ep = ep
        lams.append(round(emc.lam(parse_em9("QDEM-z4-an2-lmin20")["em8"]), 6))
    assert lams == [1.0, 0.5, 0.2, 0.2, 0.2], lams
    emc.ep = 3
    assert emc.lam(parse_em9("QDEM-z4-an2")["em8"]) == 0.0
    print(f"selftest: lambda with -an2 -lmin20 {lams}; without -lmin it reaches 0")
    # Sinkhorn: rows sum to 1, the columns go to n / k, a balanced input is kept
    post = torch.log_softmax(torch.randn(600, 8) * 3 + torch.tensor([4.0, 0, 0, 0, 0, 0, 0, -3]), 1)
    q3 = sinkhorn(post)
    q50 = sinkhorn(post, 50)
    assert torch.allclose(q3.sum(1), torch.ones(600), atol=1e-5) and torch.allclose(q50.sum(1), torch.ones(600), atol=1e-5)
    col3, col50 = q3.sum(0) / 75.0, q50.sum(0) / 75.0
    raw = torch.softmax(post, 1).sum(0) / 75.0
    assert float((col50 - 1).abs().max()) < 1e-3, col50
    assert float((col3 - 1).abs().max()) < float((raw - 1).abs().max()), (col3, raw)
    flat = torch.full((80, 8), -math.log(8.0))
    assert torch.allclose(sinkhorn(flat), torch.full((80, 8), 1 / 8.0), atol=1e-6)
    print(f"selftest: Sinkhorn rows sum to 1; the column share's worst gap {float((raw - 1).abs().max()):.3f} raw, "
          f"{float((col3 - 1).abs().max()):.3f} at {SK_ITERS} iterations, {float((col50 - 1).abs().max()):.1e} at 50")
    # the E and M steps on synthetic rows: plain = qd_gnn8's step exactly, -pu keeps the off-path prior, -bal adds
    # RIM's term (0 at a uniform marginal), the trace gets the usage
    Q = QG.synthetic_rows(60, rng, K)
    TOK = [QG.edge_labels(q) for q in Q]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    phi = rng.standard_normal((QG.R_TOP, QG.TDIM)).astype(np.float32)
    G = {"g": {"phi": phi, "kind": "passage"}}
    rows = list(range(60))
    tr, sel = {"g": rows[:40]}, {"g": rows[40:]}
    Q7.STATE7["B"]["g"] = rows[40:]

    def step(nm, fn, ep=0):
        torch.manual_seed(0)
        arm = QG.QDArm(parse_em9(nm), Q, TOK, G, z_of, A16, 0)
        model = arm.make()
        arm.prepare(model, "g")
        model.train()
        arm.emc.ep = ep
        Qp, Tp = arm.rows_for(rows[:16])
        P = QG.pack_qd(rows[:16], Qp, Tp, arm.z_of, arm.sp["K"], arm.nr)
        P.a_true = P.a.clone()
        P.du = P.dv = None
        P.em_on, P.em_mask = Q8.em_targets(arm, rows[:16], Qp)
        return arm, model, P, fn(arm, model, "g", P)

    _a8, _m8, _P8, l8 = step("QDEM-z4-L2-d16", Q8_EM_STEP["f"])
    a9, m9, P9, l9 = step("QDEM-z4-L2-d16", em_step9)
    assert l8.item() == l9.item() and torch.equal(_a8.emc.N["g"], a9.emc.N["g"]), (l8.item(), l9.item())
    assert isinstance(a9.emc, EMCore9)
    print(f"selftest: without the options em_step9 is qd_gnn8.em_step to the bit (loss {float(l9):.6f})")
    cap = {}
    sk0 = globals()["sinkhorn"]

    def spy(post, iters=SK_ITERS):
        cap["post"] = post.clone()
        return sk0(post, iters)

    globals()["sinkhorn"] = spy
    try:
        apu, mpu, Ppu, _ = step("QDEM-z4-L2-d16-pu-sk-an1", em_step9, ep=5)     # lambda 0: post is the prior plus evidence
    finally:
        globals()["sinkhorn"] = sk0
    m_ = Ppu.em_mask
    lp = F.log_softmax(mpu.typer.logits(Ppu.d[m_], Ppu.a[m_], Ppu.w[m_], None, None), 1).detach()
    off = ~Ppu.em_on[m_]
    assert bool(off.any()) and bool((~off).any())
    assert torch.allclose(cap["post"][off], lp[off], atol=1e-6), "-pu: an off-path edge's posterior is not its prior"
    assert not torch.allclose(cap["post"][~off], lp[~off], atol=1e-3), "-pu: the on-path evidence did not move"
    print(f"selftest: -pu leaves the {int(off.sum())} off-path edges at their prior and moves the {int((~off).sum())} on-path ones")
    ab, mb, Pb, lb = step("QDEM-z4-L2-d16-bal10", em_step9)
    _a0, _m0, _P0, l0 = step("QDEM-z4-L2-d16", em_step9)
    m_ = Pb.em_mask
    pbar = F.softmax(mb.typer.logits(Pb.d[m_], Pb.a[m_], Pb.w[m_], None, None), 1).mean(0)
    want = math.log(4) + float((pbar * torch.log(pbar)).sum())
    assert abs((float(lb) - float(l0)) - want) < 1e-5, (float(lb) - float(l0), want)
    lb.backward()
    assert float(mb.typer.f2.weight.grad.abs().sum()) > 0 and mb.wo.weight.grad is None
    print(f"selftest: -bal10 adds log k - H(mean pi) = {want:.5f}; its gradient reaches the typer alone")
    # fits in qd_gnn2's loop: reproducible, with the usage in the trace
    for nm in ("QDEM-z4-L2-d16-tb5-ed50-pu-lmin20-bal10-an1", "QDEM-z4-L2-d16-sk-deg", "W3EM-z4-L2-d16-pu-lmin20-bal10-an1",
               "W3EM-z4-L2-d16-sk"):
        sp = parse_em9(nm)
        Q5.NAMES[id(sp)] = nm
        fits = []
        for _rep in range(2):
            a_ = QG.QDArm(sp, Q, TOK, G, z_of, A16, 0)
            fits.append((a_,) + tuple(Q2.fit_shared2(a_, ["g"], tr, sel, "both", 3, 2e-3, 1e-4, 0, "graph", AU)))
        (a0, m0, e0, c0, _), (a1, m1, e1, c1, _) = fits
        assert isinstance(a0.emc, EMCore9)
        assert c0 == c1 and all(torch.equal(m0.state_dict()[k], m1.state_dict()[k]) for k in m0.state_dict()), f"{nm}: not reproducible"
        tr_ = Q8.TRACE[nm][-1]
        assert len(tr_) == 3 and all("perplexity_post" in r and "usage_prior" in r for r in tr_), tr_
        if sp["em8"]["lmin"]:
            assert tr_[-1]["lambda"] == sp["em8"]["lmin"], tr_[-1]["lambda"]
        print(f"selftest {nm}: reproducible ({c0}); type perplexity post {[r['perplexity_post'] for r in tr_]}, "
              f"lambda {[r['lambda'] for r in tr_]}")
    print("selftest: all checks passed")


Q8_EM_STEP = {"f": Q8.em_step}


def main():
    Q7.bind = bind    # qd_gnn7.main binds through its module's bind
    if "--selftest" in sys.argv:
        selftest()
        return
    if "--epochs" in sys.argv:
        Q8.EMCFG["epochs"] = int(sys.argv[sys.argv.index("--epochs") + 1])
    Q7.main()
    if Q6x.STATE["out"] is not None:
        finish(Q6x.STATE["out"])


if __name__ == "__main__":
    main()
