"""E1 (docs/E1_LATENT_CHAIN_EM.md): the lean pair's chain head trained by EM over latent relation chains, in training
only. Both models read rmatch's chain match (rmatch.ChainMatch): each chain entry is a (question, row, typed chain c,
seed bucket, walk mass m_c(row)) and the head weighs a chain by w(c) = exp(lw(c)), lw(c) the sum of its steps' matches,
its length prior and its bucket's weight. The head learns only through lean_gpu's listwise loss on the rows, behind gates
that start at zero. E1 adds one term that asks the chain weights themselves to explain the golds:

    P(c | q) = softmax over the question's chains of lw(c)
    pi_c(G) = (mass of c's kept entries on gold rows) / (mass of all c's kept entries)
    L_path(q) = - log sum_c P(c | q) pi_c(G_q)

over the questions of a batch with at least one gold that a chain with mass reaches (the others take no term), and the
loss is lean_gpu's listwise loss + LAM * the batch mean of L_path. Its gradient in lw(c) is EM's: the posterior that
c explains the golds minus its prior, P(c | q, G) - P(c | q). The chains, masses and walk are the build's, unchanged;
pi uses no parameter. Only chain entries carry the term, so only metaqa's fit carve trains it (webqsp never trains, the
passage graphs have no entries). Reads, serving, the forward and every other carve are the arm's own, unchanged.

Two arms, each the lean pair's arm unchanged (model, carve, KT input, settings, seeds, batches) with the term added:
    zrckpe  zrckp's model and carve (outputs/mp_unified/zp1c3p.py)
    zspke   zspk's model and carve (outputs/mp_unified/zp1c3.py)
One card fit per arm on L-musique and L-hotpotqa, on P1c2's roots and P1c3's KT input, read on the six s1eval carves and
compared with the same arm's card fit of the split under lean_gpu's loss (zrckp: scr-p1c3p-zrckp, -hp; zspk:
scr-p1c3-zspk, -hp), which decides; step 1's fits beside. The pair and its re-call under the seed null are relz's, stamped
as E1's; the verdict is this file's.

    python outputs/mp_unified/ze1em.py train --split L-musique --name scr-e1-zrckpe --arm zrckpe --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/ze1em.py read --name scr-e1-zrckpe --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/ze1em.py compare --new outputs/screen/fits/scr-e1-zrckpe \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-e1-zrckpe
    python outputs/mp_unified/ze1em.py pair --arm zrckpe --screens S1.json,S2.json --out ...
    python outputs/mp_unified/ze1em.py recall --arm zrckpe --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/ze1em.py verdict
    python outputs/mp_unified/ze1em.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zp1c3p as Z3P  # noqa: E402  (zrckp; through it P1c3's zspk and KT carves, P1c2's roots, gate and card)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z3 = Z3P.Z3
Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/E1_LATENT_CHAIN_EM.md"
OUT = ROOT / "outputs" / "e1"
SCREEN = ROOT / "outputs" / "screen"
LAM = 1.0                                                 # the term's weight, fixed before any number
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpe": "zrckp", "zspke": "zspk"}
ARM_NAMES = tuple(BASE_OF)
ARMS = {"zrckpe": Z3P.ARMS["zrckp"], "zspke": Z3.ARMS["zspk"]}
S.ARMS.update(ARMS)
FIT = {a: {"L-musique": f"scr-e1-{a}", "L-hotpotqa": f"scr-e1-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpe": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspke": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split under lean_gpu's loss ({BASE_FITS[a]['L-musique'][2]})"
           for a in ARM_NAMES}
PAIR = {a: SCREEN / f"scr-e1-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-e1-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not E1's, or not twelve reads "
        "(docs/E1_LATENT_CHAIN_EM.md, section 3)")
TERM = ("lean_gpu's listwise loss + LAM * mean over the batch's questions with a reached gold of "
        "-log sum_c softmax(lw)(c) pi_c(G) (training only)")

STASH = {"lw": None, "ch": None, "N": None}
STATS = {}


def reset_stats():
    STATS.clear()
    STATS.update({"lam": LAM, "batches": 0, "batches_with_chains": 0, "questions": 0, "listwise_sum": 0.0,
                  "path_sum": 0.0, "epochs": [], "_ep": None})
    STATS["_ep"] = {"batches": 0, "batches_with_chains": 0, "questions": 0, "listwise_sum": 0.0, "path_sum": 0.0}


reset_stats()


# ── the term ─────────────────────────────────────────────────────────────────


def chain_feats_e1(self, feats, N):
    """rmatch.ChainMatch.chain_feats, its arithmetic unchanged; in training with gradients on, each entry's lw is kept
    (in entry order) for the loss of the same step."""
    ch = feats[RM.CH_KEY]
    q = Fn.normalize(feats["SEMB"][0], dim=1)
    lp = self.step_logp(q, ch)
    Kz = lp.shape[2]
    flat = lp.reshape(-1)
    lpi = Fn.log_softmax(q @ self.cm_pi_w + self.cm_pi_b, dim=1)
    dev = q.device
    k = torch.arange(RM.HOPS, device=dev)
    Sm = torch.zeros(N, dtype=q.dtype, device=dev)
    Sr = torch.zeros(N, dtype=q.dtype, device=dev)
    keep = self.training and torch.is_grad_enabled()
    lws = []
    for s in range(0, ch["eq"].size, RM.CHUNK):
        e = min(ch["eq"].size, s + RM.CHUNK)
        eq = torch.from_numpy(ch["eq"][s:e]).to(dev)
        row = torch.from_numpy(ch["row"][s:e]).to(dev)
        z = torch.from_numpy(ch["z"][s:e]).to(dev).long()
        b = torch.from_numpy(ch["b"][s:e]).to(dev).long()
        m = torch.from_numpy(ch["m"][s:e]).to(dev).to(q.dtype)
        valid = z >= 0
        idx = (eq[:, None] * RM.HOPS + k[None, :]) * Kz + z.clamp_min(0)
        lw = (flat[idx] * valid.to(q.dtype)).sum(1) + lpi[eq, valid.sum(1) - 1] + self.cm_beta[b]
        w = lw.exp()
        Sm = Sm.index_add(0, row, w * m)
        Sr = Sr.index_add(0, row, w)
        if keep:
            lws.append(lw)
    if keep:
        if STASH["lw"] is not None:
            raise SystemExit("ze1em: a batch's chain weights were kept and never scored")
        STASH.update(lw=torch.cat(lws) if lws else q.new_zeros(0), ch=ch, N=int(N))
    return torch.log1p(Sm / RM.TAU), torch.log1p(Sr)


def chains_of(ch, packed=True):
    """Per entry its chain (one per distinct question, typed steps and bucket), and per chain its first entry and its
    question; the chains come sorted by (question, steps, bucket). The key is packed into one int64 when it fits (the
    same order as the row-wise unique, which is the fallback)."""
    eq, z, b = (np.asarray(ch["eq"], np.int64), np.asarray(ch["z"], np.int64) + 1, np.asarray(ch["b"], np.int64))
    if eq.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    ke, kz, kb = (max(int(x.max()).bit_length(), 1) for x in (eq, z, b))
    if packed and ke + RM.HOPS * kz + kb <= 62:
        key = (eq << (RM.HOPS * kz + kb)) | b
        for j in range(RM.HOPS):
            key |= z[:, j] << ((RM.HOPS - 1 - j) * kz + kb)
        _u, first, inv = np.unique(key, return_index=True, return_inverse=True)
    else:
        _u, first, inv = np.unique(np.column_stack([eq, z, b]), axis=0, return_index=True, return_inverse=True)
    return inv.reshape(-1), first, eq[first]


def chain_pi(ch, gold_rows):
    """Per chain: pi_c(G) (0 when its kept mass is 0); the chains' first entries and questions (numpy, float64)."""
    inv, first, cq = chains_of(ch)
    C = first.size
    m = np.asarray(ch["m"], np.float64)
    g = gold_rows[np.asarray(ch["row"], np.int64)]
    M = np.bincount(inv, m, C)
    H = np.bincount(inv, m * g, C)
    pi = np.where(M > 0, H / np.where(M > 0, M, 1.0), 0.0)
    return pi, first, cq


def chain_nll(lwc, cq, pi):
    """lwc (C,) the chains' log weights (a tensor), cq (C,) their questions sorted, pi (C,) numpy: the sum of
    -log sum_c softmax(lwc)(c) pi_c over the questions with a chain of pi > 0, and their count."""
    take = np.unique(cq[pi > 0])
    if take.size == 0:
        return lwc.sum() * 0.0, 0
    sel = np.flatnonzero(np.isin(cq, take))
    q_s, pi_s = cq[sel], pi[sel]
    _u, start, cnt = np.unique(q_s, return_index=True, return_counts=True)
    Qn, W = start.size, int(cnt.max())
    slot = np.repeat(np.arange(Qn), cnt) * W + (np.arange(sel.size) - np.repeat(start, cnt))
    dev = lwc.device
    at = torch.from_numpy(slot).to(dev)
    x = lwc[torch.from_numpy(sel).to(dev)].to(torch.float64)
    lpi = torch.from_numpy(np.where(pi_s > 0, np.log(np.where(pi_s > 0, pi_s, 1.0)), -np.inf)).to(dev)
    neg = torch.full((Qn * W,), -math.inf, dtype=torch.float64, device=dev)
    logZ = torch.logsumexp(neg.index_put((at,), x).view(Qn, W), 1)
    logN = torch.logsumexp(neg.index_put((at,), x + lpi).view(Qn, W), 1)
    return (logZ - logN).sum(), Qn


def path_nll(lw, ch, gold):
    """The batch's L_path sum and the questions it covers, from the kept lw (entry order) and the batch's golds."""
    gold_rows = gold.detach().to("cpu").numpy() > 0
    pi, first, cq = chain_pi(ch, gold_rows)
    if first.size == 0:
        return lw.sum() * 0.0, 0
    lwc = lw[torch.from_numpy(first).to(lw.device)]
    return chain_nll(lwc, cq, pi)


def make_loss(base, lam):
    def listwise_e1(scores, gold, nq, B):
        loss = base(scores, gold, nq, B)
        ep = STATS["_ep"]
        ep["batches"] += 1
        ep["listwise_sum"] += float(loss.detach())
        if STASH["lw"] is None:
            return loss
        lw, ch, N = STASH["lw"], STASH["ch"], STASH["N"]
        STASH.update(lw=None, ch=None, N=None)
        if N != scores.numel():
            raise SystemExit(f"ze1em: the kept chain weights are of a batch of {N} rows, the loss's of {scores.numel()}")
        ep["batches_with_chains"] += 1
        nll, nqs = path_nll(lw, ch, gold)
        if nqs == 0:
            return loss
        ep["questions"] += nqs
        ep["path_sum"] += float(nll.detach())
        return loss + (lam * nll / nqs).to(loss.dtype)
    return listwise_e1


def close_epoch():
    ep = STATS["_ep"]
    STATS["epochs"].append({"batches": ep["batches"], "batches_with_chains": ep["batches_with_chains"],
                            "questions": ep["questions"],
                            "listwise_mean": ep["listwise_sum"] / max(ep["batches"], 1),
                            "path_nll_mean": ep["path_sum"] / max(ep["questions"], 1)})
    for k in ("batches", "batches_with_chains", "questions", "listwise_sum", "path_sum"):
        STATS[k] += ep[k]
        ep[k] = 0 if isinstance(ep[k], int) else 0.0


@contextlib.contextmanager
def e1_loss(lam=LAM):
    """lean_gpu's loop with the term added to its one loss: rmatch's chain_feats keeps each training batch's chain
    weights, the loss adds LAM * L_path, and each epoch's sums close when the loop keeps its state; all restored
    after, an error too."""
    saved = LG.listwiseD, RM.ChainMatch.chain_feats, LG.cpu_state
    base_state = saved[2]

    def cpu_state(model):
        close_epoch()
        return base_state(model)

    reset_stats()
    STATS["lam"] = lam
    STASH.update(lw=None, ch=None, N=None)
    LG.listwiseD, RM.ChainMatch.chain_feats, LG.cpu_state = make_loss(saved[0], lam), chain_feats_e1, cpu_state
    try:
        yield
    finally:
        LG.listwiseD, RM.ChainMatch.chain_feats, LG.cpu_state = saved
        STASH.update(lw=None, ch=None, N=None)


def summary():
    return {k: STATS[k] for k in ("lam", "batches", "batches_with_chains", "questions", "epochs")} | {
        "listwise_mean": STATS["listwise_sum"] / max(STATS["batches"], 1),
        "path_nll_mean": STATS["path_sum"] / max(STATS["questions"], 1)}


# ── train and read ───────────────────────────────────────────────────────────


def stamp(argv, arm, split, summ):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"ze1em_sha256": LC.sha_src(__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__), "e1": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "device": Z2.DEVICE,
                        "loss": TERM, "lam": LAM, "term": summ, "pools": "P1c2's (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zrc_records": Z2._BUILT(Z2.CH_ZRC), "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"ze1em: train takes --arm zrckpe or zspke, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"ze1em: E1's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"ze1em: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    with Z2.on_p1c2(), e1_loss():
        rc = RM.train(argv, split)
    summ = summary()
    if summ["questions"] == 0:
        raise SystemExit(f"ze1em: no training batch carried the term ({summ['batches']} batches)")
    with Z2.on_p1c2():
        stamp(argv, arm, split, summ)
    log(f"ze1em {arm} {split}: the term on {summ['questions']} question-batches of {summ['batches_with_chains']} "
        f"chain batches; L_path {summ['epochs'][0]['path_nll_mean']:.4f} (epoch 0) to "
        f"{summ['epochs'][-1]['path_nll_mean']:.4f} (last)")
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") not in ARMS or "e1" not in rec:
        raise SystemExit(f"ze1em: {name} is not an E1 fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fit ────────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"ze1em: no base fit of {BASE_OF[arm]} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_OF[arm], fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b, f = BASE_OF[arm], BASE_FITS[arm]["L-musique"][2]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({b}'s card fits {f} and {f}-hp, lean_gpu's loss)"),
            ("(rel's screen fits)", f"({b}'s card fits, lean_gpu's loss)"),
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (E1's chain EM) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"E1 ({DECLARED}): {arm} ({BASE_OF[arm]} trained with EM over latent chains) against "
                      f"{BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "E1", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "ze1em_sha256": LC.sha_src(__file__)})
        rec.pop("round", None)
        LC.write_json(js, rec)


def pair(arm, screens, out):
    with on_base(arm):
        rec = Z.pair(screens, out)
    restamp(out, arm)
    return rec


def recall(arm, null_files, pair_file, out, base_screens=None):
    with on_base(arm):
        rec = Z.recall(null_files, pair_file, out, base_screens)
    restamp(out, arm)
    return rec


# ── the verdict ──────────────────────────────────────────────────────────────


def tag(r):
    return f"{r['split']} {r['dataset']}"


def verdict(recalls=None, out=None):
    """E1's screen verdict, per arm, from its re-call (section 3); webqsp and metaqa beside."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems = Path(recalls[a]), []
        rows = []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "E1" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not E1's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems,
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                             if r.get("dataset") in BESIDE},
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    rec = {"stage": "E1", "declared_in": DECLARED, "rule": RULE, "arms": per, "ze1em_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# E1's screen verdict: zrckpe **{per['zrckpe']['verdict']}**, zspke **{per['zspke']['verdict']}**", "",
          f"Rule: {RULE}.", ""]
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
                  + (", ".join(f"{k} {v['call']} ({v['delta']:+.4f})" if isinstance(v["delta"], (int, float))
                               else f"{k} {v['call']}" for k, v in p["beside"].items()) or "none") + ".")
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("ze1em verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "E1", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def nll_by_hand(ch, lw_e, gold_rows):
    """The float64 reference, question by question and chain by chain: the L_path sum, the questions it covers, and
    d sum / d lw(c) = -(posterior - prior) per chain key."""
    chains = {}
    for i in range(len(ch["eq"])):
        key = (int(ch["eq"][i]), tuple(int(x) for x in ch["z"][i]), int(ch["b"][i]))
        c = chains.setdefault(key, {"lw": float(lw_e[i]), "M": 0.0, "H": 0.0})
        assert c["lw"] == float(lw_e[i])
        m = float(ch["m"][i])
        c["M"] += m
        c["H"] += m * float(gold_rows[int(ch["row"][i])])
    total, nqs, grad = 0.0, 0, {}
    for qi in sorted({k[0] for k in chains}):
        ks = [k for k in chains if k[0] == qi]
        lw = np.array([chains[k]["lw"] for k in ks])
        pi = np.array([chains[k]["H"] / chains[k]["M"] if chains[k]["M"] > 0 else 0.0 for k in ks])
        if not (pi > 0).any():
            continue
        prior = np.exp(lw - lw.max())
        prior /= prior.sum()
        post = prior * pi
        total += -math.log(post.sum())
        post /= post.sum()
        nqs += 1
        for k, a, b in zip(ks, post, prior):
            grad[k] = -(a - b)
    return total, nqs, grad


def selftest():
    import shutil
    import tempfile
    import zkind as ZK
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arms are the lean pair's, unchanged; the term is added in training only and restored after
    assert S.ARMS["zrckpe"] == S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC)
    assert S.ARMS["zspke"] == S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    saved = LG.listwiseD, RM.ChainMatch.chain_feats, LG.cpu_state
    with e1_loss():
        assert LG.listwiseD is not saved[0] and Z3P.ZRCKP.chain_feats is chain_feats_e1
        assert Z3.ZSPK.chain_feats is chain_feats_e1 and LG.cpu_state is not saved[2]
    assert (LG.listwiseD, RM.ChainMatch.chain_feats, LG.cpu_state) == saved
    try:
        with e1_loss():
            raise ValueError("x")
    except ValueError:
        pass
    assert (LG.listwiseD, RM.ChainMatch.chain_feats, LG.cpu_state) == saved
    tmp = Path(tempfile.mkdtemp(prefix="ze1em_"))
    try:
        rng = np.random.default_rng(23)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        c = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        u = ZK.Untyped(c, "2wiki")
        blocks = ["rank", "SEMB"]
        qs = np.arange(min(c.rows, 10), dtype=np.int64)
        B = qs.size
        keep = torch.ones(B, len(blocks))
        feats, nq, bz, gold = c.batch(qs, blocks)
        ch = feats[RM.CH_KEY]
        assert ch["eq"].size > 0 and float(gold.sum()) > 0
        for arm, cls in (("zrckpe", Z3P.ZRCKP), ("zspke", Z3.ZSPK)):
            torch.manual_seed(0)
            mdl = cls(blocks, c.widths, 16, dropout=0.1, seed=0)
            g = torch.Generator().manual_seed(31)
            with torch.no_grad():
                for p in mdl.parameters():
                    p.copy_(torch.randn(p.shape, generator=g) * 0.3)
            # 2. the kept chain_feats is rmatch's bit for bit, in eval and in training; it keeps lw only in training
            #    with gradients on; a forward is rmatch's; an untyped batch keeps nothing
            for mode in (False, True):
                mdl.train(mode)
                a = saved[1](mdl, feats, nq.numel())
                with e1_loss():
                    b = mdl.chain_feats(feats, nq.numel())
                    assert (STASH["lw"] is not None) == mode
                    STASH.update(lw=None, ch=None, N=None)
                    with torch.no_grad():
                        mdl.chain_feats(feats, nq.numel())
                    assert STASH["lw"] is None
                assert all(torch.equal(x, y) for x, y in zip(a, b)), (arm, mode)
                torch.manual_seed(5)
                f0 = mdl(feats, keep, nq, B, bz)
                with e1_loss():
                    torch.manual_seed(5)
                    f1 = mdl(feats, keep, nq, B, bz)
                    STASH.update(lw=None, ch=None, N=None)
                    fu = mdl(*u.batch(qs, blocks)[:1], keep, nq, B, bz)
                    assert STASH["lw"] is None
                assert torch.equal(f0, f1), (arm, mode)
            # 3. the term: the float64 reference's sum and question count; its gradient in each chain's lw is EM's,
            #    minus (posterior - prior); a second batch without a kept step is refused
            mdl.train(True)
            with e1_loss():
                mdl.chain_feats(feats, nq.numel())
                lw = STASH["lw"]
                STASH.update(lw=None, ch=None, N=None)
            gold_rows = gold.numpy() > 0
            ref, nref, gref = nll_by_hand(ch, lw.detach().double().numpy(), gold_rows)
            assert nref >= 2, nref
            leaf = lw.detach().double().clone().requires_grad_(True)
            tot, nqs = path_nll(leaf, ch, gold)
            assert nqs == nref and abs(float(tot) - ref) < 1e-9 * max(1.0, abs(ref)), (float(tot), ref)
            tot.backward()
            inv, first, cq = chains_of(ch)
            for x, y in zip((inv, first, cq), chains_of(ch, packed=False)):
                assert np.array_equal(x, y)
            per_chain = np.bincount(inv, leaf.grad.numpy(), first.size)
            for j, f in enumerate(first):
                key = (int(ch["eq"][f]), tuple(int(x) for x in ch["z"][f]), int(ch["b"][f]))
                assert abs(per_chain[j] - gref.get(key, 0.0)) < 1e-9, (key, per_chain[j], gref.get(key))
            assert any(abs(v) > 1e-6 for v in gref.values())
            # no reached gold: no term
            t0z, n0z = path_nll(leaf.detach(), ch, torch.zeros_like(gold))
            assert n0z == 0 and float(t0z) == 0.0
            with e1_loss():
                loss_fn = LG.listwiseD
                mdl.train(True)
                s = mdl(feats, keep, nq, B, bz)
                lst = saved[0](s, gold, nq, B)
                tot_l = loss_fn(s, gold, nq, B)
                assert abs(float(tot_l) - float(lst) - LAM * ref / nref) < 1e-5, (float(tot_l), float(lst), ref)
                assert STASH["lw"] is None and float(loss_fn(s, gold, nq, B)) == float(lst)
                s = mdl(feats, keep, nq, B, bz)
                SR.must_stop(mdl, feats, keep, nq, B, bz)
                STASH.update(lw=None, ch=None, N=None)
        # 4. toy fits through lean_gpu's loop (the typed carve and an untyped one): with LAM 0 every epoch's state and
        #    the SWA state are the plain loop's bit for bit; with LAM every state moves, the term runs on the typed
        #    carve's batches only and is counted per epoch; a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm in ARM_NAMES:
            with S.patched(arm):
                plain = LG.fit_variant([c, u], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm} plain")
                with e1_loss(0.0):
                    zero = LG.fit_variant([c, u], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm} lam 0")
                    sz = summary()
                with e1_loss():
                    fit = LG.fit_variant([c, u], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm} lam 1")
                    sf = summary()
                with e1_loss():
                    rep = LG.fit_variant([c, u], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-{arm} repeat")
            assert (LG.listwiseD, RM.ChainMatch.chain_feats, LG.cpu_state) == saved
            pairs = list(zip(plain["states"] + [plain["swa"]], zero["states"] + [zero["swa"]]))
            assert all(not LG.same_state(a, b) for a, b in pairs), arm
            assert all(LG.same_state(a, b) for a, b in zip(plain["states"] + [plain["swa"]],
                                                               fit["states"] + [fit["swa"]])), arm
            assert all(not LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]],
                                                           rep["states"] + [rep["swa"]])), arm
            assert len(sf["epochs"]) == cfg["epochs"] and sf["questions"] > 0 and sz["questions"] == sf["questions"]
            assert all(e["batches_with_chains"] > 0 and e["batches"] > e["batches_with_chains"] for e in sf["epochs"])
            assert sf["epochs"][-1]["path_nll_mean"] < sf["epochs"][0]["path_nll_mean"], sf["epochs"]
            log(f"  toy {arm}: L_path {sf['epochs'][0]['path_nll_mean']:.4f} -> {sf['epochs'][-1]['path_nll_mean']:.4f} "
                f"with the term, {sz['epochs'][0]['path_nll_mean']:.4f} -> {sz['epochs'][-1]['path_nll_mean']:.4f} at LAM 0")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. the refusals
    good = ["train", "--arm", "zrckpe", "--name", "scr-e1-zrckpe", "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", "scr-e1-zrckpe", "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, ["train", "--arm", "zrckpe", "--name", "scr-e1-zrckpe-hp", "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspke", "--name", "scr-e1-zspke", "--cache-root", "outputs/p1gb/cache"],
                 "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        (T / "fits" / "y").mkdir(parents=True)
        LC.write_json(T / "fits" / "y" / "screen.json", {"arm": "zrckp", "p1c3p": {}})
        SR.must_stop(read, ["--name", "y", "--out-root", str(T / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 6. relz's pair and re-call for each arm against its base arm's card fits, relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm in ARM_NAMES:
            fits = BASE_FITS[arm]
            null1 = SR.fake_null(T / arm / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(T / arm / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=BASE_OF[arm])) for sp in fits}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(T / arm / "z", FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(T / arm / "z", FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            pr = pair(arm, [za, zh], T / arm / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_OF[arm], pr["verdict"]
            pj = json.loads((T / arm / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "E1" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T / arm / "pz.json", T / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"E1 ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (E1's chain EM) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 7. the verdict, per arm, webqsp and metaqa beside
        gm = ("L-musique", "musique")
        rec = verdict({"zrckpe": fake_recall(T / "a.json", "zrckpe", {gm: "GAIN"}),
                       "zspke": fake_recall(T / "b.json", "zspke", {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"})},
                      T / "v")
        assert rec["arms"]["zrckpe"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zspke"]["verdict"] == "NOT_ADOPTED"
        assert sorted(rec["arms"]["zspke"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        rec = verdict({"zrckpe": fake_recall(T / "a.json", "zrckpe", {}),
                       "zspke": fake_recall(T / "b.json", "zspke", {gm: "GAIN"}, stage="T1")}, T / "v")
        assert rec["arms"]["zrckpe"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspke"]["verdict"] == "INCOMPLETE"
        rec = verdict({"zrckpe": T / "none.json", "zspke": fake_recall(T / "b.json", "zrckpe", {gm: "GAIN"})}, T / "v")
        assert rec["arms"]["zrckpe"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspke"]["verdict"] == "INCOMPLETE"
    log(f"ze1em selftest: the arms are the lean pair's; the kept chain_feats is rmatch's bit for bit and keeps lw in "
        f"training only; L_path matches the float64 reference and its gradient is EM's (posterior - prior); toy fits at "
        f"LAM 0 are the plain loop's bit for bit, at LAM {LAM} they move and L_path falls; train refuses another arm, "
        f"split, fit name, cache root or device; read refuses a fit that is not E1's; relz's pair and re-call against "
        f"each base arm's card fits, stamped E1's; the verdict per arm ({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="L-musique", choices=S2.SPLITS)
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "verdict":
        return 0 if all(v["verdict"] != "INCOMPLETE" for v in verdict()["arms"].values()) else 1
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=ARM_NAMES)
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        try:
            if k.cmd == "pair":
                pair(g.arm, [x for x in g.screens.split(",") if x], g.out)
            else:
                if len(null) != 4 or not g.pair:
                    gp.error("recall: --pair and the four --null files")
                recall(g.arm, null, g.pair, g.out)
        except SystemExit as e:
            log(f"ze1em {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("ze1em: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
