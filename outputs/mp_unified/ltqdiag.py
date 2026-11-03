"""LTQ, a question-gated next-hop scorer with relations and IRA's refinement, each gold in its own softmax
(docs/LTQ_QUESTION_GATED_NEXT_HOP.md): does a scorer trained on the golds lift the next hops the found passage has no
link to, once each target is its own softmax, the question's hop slots are matched to learned relations by IRA's plan and
refined over IRA's rounds, and a question gate chooses between LTA's scorer and that one?

LTA's column records rank musique's unlinked targets (every row outside rrf's top five, WQ-hotpotqa) at T2 0.096 (s1fit,
out of fold) and 0.084 (s1eval) by LTA's bil, against 0.144 and 0.113 by CHN's fixed pair. LTQ refits per split with
ltmcols' folds, stop and loop, in two forms that differ only in their lists: u on LTA's A lists (uniform negatives), h on
LTH's (negatives drawn 1/rrf rank). Each A case's anchor list over every row outside the top five is then ranked by the
fold fit its question draws (ltacols' question and A case) and by the free scores, whose T2s must equal LTA's column
records; bil's numbers are LTA's records'. ltmcols.py, ltacols.py, lthcols.py and ltwdiag.py are imported and not edited:
LTQ's scorer and loss are patched into ltwdiag, and its stage, root and lists into ltmcols, for the length of a fit or a
load, every name put back. LTQ writes only under outputs/ltq.

    python outputs/mp_unified/ltqdiag.py fit --form u --split WQ-hotpotqa --host
    python outputs/mp_unified/ltqdiag.py read --split WQ-hotpotqa --dataset musique --carve s1fit --host
    python outputs/mp_unified/ltqdiag.py report
    python outputs/mp_unified/ltqdiag.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("LTM_THREADS", "2"))
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import ltacols as LTA  # noqa: E402  (LTA's question, A case, chunk layout, lists and records; read only, not edited)
import lthcols as LTH  # noqa: E402  (LTH's lists: read only, not edited)

LTMC, LTW, CH, WQ = LTA.LTMC, LTA.LTW, LTA.CH, LTA.WQ
DECLARED = "docs/LTQ_QUESTION_GATED_NEXT_HOP.md"
STAGE = "LTQ"
OUT = ROOT / "outputs" / "ltq"
LISTS = {"u": LTA.OUT, "h": LTH.OUT}             # each form's A lists: LTA's (uniform negatives), LTH's (1/rrf rank)
FORMS = tuple(LISTS)
FORM = "ltq"                                     # ltmcols' FORM for the length of LTQ's fits and loads
PKO = LTMC.PKO
TOP, K = LTMC.TOP, LTMC.K
B_FEATS = ("cos", "res", "pair", "anchor_cos") + LTW.RANK_COLS + ("log1p_rank",)
B_IDX = tuple(LTW.FEATS.index(f) for f in B_FEATS)
REL_K = 16                                       # the relations: each a key (the slots' space) and an offset (the rows')
SLOTS = 4                                        # IRA's hop slots: one per hop of musique's longest questions
ROUNDS = 2                                       # IRA's rounds (RTR's two): the plan, then the slots moved toward it
SINK_ITERS = 20                                  # each a column step, then a row step (zira.plan's order)
MODELS = tuple(f"ltq_{f}" for f in FORMS)
SCORES = MODELS + tuple(LTW.FREE)                # ranked by the read; bil's numbers are LTA's column record's
READS = {"WQ-hotpotqa": (("musique", "s1fit"), ("musique", "s1eval"), ("2wiki", "fit"), ("2wiki", "s1eval"),
                         ("hotpotqa", "s1eval")),
         "WQ-musique": (("hotpotqa", "fit"), ("2wiki", "fit"), ("hotpotqa", "s1eval"), ("2wiki", "s1eval"),
                        ("musique", "s1eval"))}
DECIDE_SPLIT = "WQ-hotpotqa"
DECIDE = (("musique", "s1fit"), ("musique", "s1eval"))
PAIR_MARGIN, LINKED_SLACK, MIN_RANKED = 0.03, 0.02, 200
TOL = 1e-12
assert len(B_IDX) == 10 and "anc" not in B_FEATS and not any(f.startswith("fam_") for f in B_FEATS)
assert all(c in LTMC.col_carves(sp) for sp, cs in READS.items() for c in cs) and DECIDE == READS[DECIDE_SPLIT][:2]
assert set(LTW.FREE) == {"rrf", "cos", "anc", "res", "pair"} and FORM not in LTW.FORMS and REL_K >= SLOTS + 1

LTW_BUILD, LTW_LOSS = LTW.build_model, LTW.list_loss     # ltwdiag's own, captured before any patch


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] ltqdiag: {msg}", flush=True)


# ── the scorer and its loss (section 2) ──────────────────────────────────────


def plan(torch, h, key, bz, b0):
    """IRA's plan on the question's side, in zira.plan's form: Sinkhorn in log space, in float32, of the REL_K relations
    over the SLOTS hop slots and a dustbin, on slot s's logits bz h_s . key_j / sqrt(rank) and the dustbin's b0. Each
    relation's mass over the slots and the dustbin is 1; each slot's mass over the relations t = min(1, REL_K / (SLOTS +
    1)); the dustbin's REL_K - SLOTS t. SINK_ITERS iterations, each a column step and then a row step, so the rows hold
    exactly. Returns (B, REL_K, SLOTS)."""
    B, nk = h.shape[0], key.shape[0]
    lg = torch.cat([bz.float() * torch.einsum("bsr,kr->bks", h.float(), key.float()) / math.sqrt(key.shape[1]),
                    b0.float().view(1, 1, 1).expand(B, nk, 1)], 2)
    t = min(1.0, nk / (SLOTS + 1))
    tgt = torch.log(torch.tensor([t] * SLOTS + [nk - SLOTS * t], dtype=torch.float32)).view(1, 1, -1)
    u = torch.zeros(B, nk, 1)
    v = None
    for _i in range(SINK_ITERS):
        v = tgt - torch.logsumexp(lg + u, dim=1, keepdim=True)
        u = -torch.logsumexp(lg + v, dim=2, keepdim=True)
    return (lg + u + v).exp()[:, :, :SLOTS]


def build(form, nf, d):
    """ltwdiag's build_model with LTQ's scorer as the form ltq; any other form is ltwdiag's own. A is LTW's bil. B reads
    the ten features that are not the row's raw cosine to the anchor or its link bits, and x_v . p: p the next hop's
    vector from the question's hop slots, matched to the relations by IRA's plan over IRA's rounds. The gate reads q, x_a
    and q * x_a; the score is log(sigmoid(g) e^A + sigmoid(-g) e^B)."""
    if form != FORM:
        return LTW_BUILD(form, nf, d)
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    class Ltq(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = LTW_BUILD("bil", nf, d)                     # A: LTW's bil, seeded as ltwdiag seeds it
            torch.manual_seed(LTW.SEED + 7)
            hd, rk = LTW.HID, LTW.RANK
            self.b = nn.Sequential(nn.Linear(len(B_IDX), hd), nn.GELU(), nn.Linear(hd, hd), nn.GELU(), nn.Linear(hd, 1))
            g = torch.Generator().manual_seed(LTW.SEED + 8)
            self.rV = nn.Parameter(torch.randn(3, d, rk, generator=g))       # q, x_a and r into the slots' space
            self.slot = nn.Parameter(torch.randn(SLOTS, rk, generator=g))    # each hop slot's own start
            self.key = nn.Parameter(torch.randn(REL_K, rk, generator=g))     # each relation's key
            self.off = nn.Parameter(torch.zeros(REL_K, d))                   # each relation's offset; zero at the start
            self.Uo = nn.Parameter(torch.zeros(rk, d))                       # a slot's state into the rows' space; zero
            self.step = nn.Parameter(torch.zeros(ROUNDS))                    # each round's move toward the plan; zero
            self.bz = nn.Parameter(torch.ones(1))                            # the plan's logit scale (IRA's ira_bz)
            self.b0 = nn.Parameter(torch.zeros(1))                           # the dustbin's logit (IRA's ira_b0)
            self.ptr = nn.Parameter(torch.zeros(rk))                         # the next hop's slot; uniform at the start
            self.ptr_b = nn.Parameter(torch.zeros(SLOTS))
            self.gate = nn.Linear(3 * d, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            self.register_buffer("bi", torch.tensor(B_IDX, dtype=torch.long), persistent=False)

        def hop(self, q, xa, r):
            """The slots after IRA's rounds, the last plan, the pointer over the slots, the next hop's relation mass and
            its vector p."""
            h = torch.einsum("btd,tdk->bk", torch.stack((q, xa, r), 1), self.rV).unsqueeze(1) + self.slot.unsqueeze(0)
            for t in range(ROUNDS):
                h = h + self.step[t] * torch.einsum("bks,kr->bsr", plan(torch, h, self.key, self.bz, self.b0), self.key)
            P = plan(torch, h, self.key, self.bz, self.b0)
            w = torch.softmax(h @ self.ptr + self.ptr_b, dim=1)
            rel = torch.einsum("bks,bs->bk", P, w)
            p = torch.einsum("bs,bsr->br", w, h) @ self.Uo + rel @ self.off
            return {"P": P, "w": w, "rel": rel, "p": p}

        def parts(self, F, X, q, xa, r):
            sa = self.a(F, X, q, xa, r)
            sb = self.b(F.index_select(-1, self.bi)).squeeze(-1) + torch.einsum("bcd,bd->bc", X, self.hop(q, xa, r)["p"])
            return sa, sb, self.gate(torch.cat((q, xa, q * xa), -1))

        def forward(self, F, X, q, xa, r):
            sa, sb, gam = self.parts(F, X, q, xa, r)
            return torch.logaddexp(Fn.logsigmoid(gam) + sa, Fn.logsigmoid(-gam) + sb)

        def explain(self, q, xa, r):
            """What a list's score uses: the gate's weight on A, the next hop's relation mass and the pointer."""
            with torch.no_grad():
                hp = self.hop(q, xa, r)
                return {"gate_A": torch.sigmoid(self.gate(torch.cat((q, xa, q * xa), -1))).squeeze(-1), "rel": hp["rel"],
                        "w": hp["w"]}

    return Ltq()


def gold_loss(s, mask, pos, w):
    """Each target its own softmax against its list's negatives: softplus(logsumexp of the negatives - the target's
    score), averaged over a list's targets, the lists weighted as ltwdiag's list_loss weighs them."""
    import torch
    import torch.nn.functional as Fn
    lo = torch.finfo(s.dtype).min
    ln = torch.logsumexp(s.masked_fill(~(mask & ~pos), lo), 1, keepdim=True)
    per = (Fn.softplus(ln - s) * pos).sum(1) / pos.sum(1).clamp(min=1)
    return (per * w).sum() / w.numel()


def form_sha():
    """What a fit's numbers depend on beyond ltmcols and ltwdiag: LTQ's scorer, plan and loss and their settings."""
    src = "".join(inspect.getsource(f) for f in (plan, build, gold_loss))
    src += repr((B_FEATS, FORM, REL_K, SLOTS, ROUNDS, SINK_ITERS))
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


@contextlib.contextmanager
def on_ltq(form, out=OUT, lists=LISTS):
    """ltmcols with LTQ's stage, declaration, form and root (out/form), its lists read from the form's own root (LTA's or
    LTH's, read only), and ltwdiag's build_model and list_loss LTQ's; every name put back after, also on a failure."""
    src, lp0 = Path(lists[form]), LTMC.lists_path

    def lists_path(_out, ds, carve):
        return lp0(src, ds, carve)

    def no_lists(*_a, **_k):
        raise SystemExit("ltq: LTQ builds no lists; it reads LTA's and LTH's")
    with WQ.patch(LTMC, STAGE=STAGE, DECLARED=DECLARED, OUT=Path(out) / form, FORM=FORM, lists_path=lists_path,
                  lists=no_lists), WQ.patch(LTW, build_model=build, list_loss=gold_loss):
        yield


# ── the fits (section 4) ─────────────────────────────────────────────────────


def ltq_block(form, lists=LISTS):
    return {"declared_in": DECLARED, "form": form, "lists_root": LTA.rel(lists[form]), "form_sha256": form_sha(),
            "ltqdiag_sha256": WQ.sha_src(__file__), "b_features": list(B_FEATS), "gate_in": "q, x_a, q * x_a",
            "relations": REL_K, "slots": SLOTS, "rounds": ROUNDS, "sinkhorn_iters": SINK_ITERS,
            "loss": "each target its own softmax against its list's negatives"}


def fit(form, split, root=PKO, out=OUT, nodes=None, epochs=LTW.EPOCHS, host=False, min_lists=LTMC.MIN_LISTS,
        lists=LISTS):
    """ltmcols' five fold fits of the split on the form's A lists, LTQ's scorer and loss, written under out/form; lists
    built with a limit or another cap are refused."""
    src = Path(lists[form])
    for ds, cv in LTMC.train_carves(split):
        z = LTA.load_npz(LTMC.lists_path(src, ds, cv), lambda k: not k.startswith("A_"))
        if int(z["limit"]) or int(z["cap"]) != LTW.CAP:
            raise SystemExit(f"ltq fit {form} {split}: the lists of {ds}/{cv} were built with limit {int(z['limit'])}, "
                             f"cap {int(z['cap'])}")
    with on_ltq(form, out, lists):
        rec = LTMC.fit(split, root=root, out=Path(out) / form, nodes=nodes, epochs=epochs, host=host, min_lists=min_lists)
    rec["ltq"] = ltq_block(form, lists)
    LTW.write_json(Path(out) / form / f"fit_{split}.json", rec)
    log(f"fit {form} {split}: {rec['with_lists']} in the scorer; stop T2 "
        f"{json.dumps([f['best_stop_T2'] for f in rec['folds']])}")
    return rec


def load_ltq(form, split, out=OUT, lists=LISTS):
    """The form's fold fits of the split (ltmcols' load, under LTQ's names) and its fit record's sha; a fit by another
    scorer, plan or loss is refused."""
    fr = Path(out) / form / f"fit_{split}.json"
    if not fr.is_file():
        raise SystemExit(f"ltq: no fit of form {form} for {split} ({fr})")
    b = json.loads(fr.read_text(encoding="utf-8")).get("ltq") or {}
    if (b.get("declared_in"), b.get("form"), b.get("form_sha256")) != (DECLARED, form, form_sha()):
        raise SystemExit(f"ltq: {fr} is not LTQ's fit of form {form} by this scorer, plan and loss")
    with on_ltq(form, out, lists):
        M, fsha, _ = LTMC.load_folds(Path(out) / form, split)
    return M, fsha


# ── the reads (section 5) ────────────────────────────────────────────────────


class AChunk(LTA.Chunk):
    """ltacols' chunk with each question's A list alone: its A case's anchor, every row outside the top five."""

    def add(self, qi, qd, qe, gt):
        case = LTA.a_case(qd, gt)
        if case is None:
            return
        a, k, cand, _ct = case
        b = LTW.fam_bits(qd["n"], a, qd["u"], qd["v"], qd["f"])
        lq = len(self.gq)
        self.gq.append(qi)
        self.qe.append(qe)
        for key, x in (("q", lq), ("k", k), ("anode", qd["node"][a]), ("axc", qd["x6"][a]), ("c", cand),
                       ("cnode", qd["node"][cand]), ("ct", qd["g"][cand]), ("cfam", b[cand].astype(np.int8)),
                       ("cxc", qd["x6"][cand]), ("crank", qd["rk"][cand])):
            self.d[key].append(x)
        t = np.flatnonzero(qd["g"] & ~qd["in_top"])
        self.case[lq] = (k, t, b[t])
        self.rows += cand.size


def summary(rk, lk, hp):
    a, lk, hp = np.asarray(rk, np.float64), np.asarray(lk, bool), np.asarray(hp, np.int64)

    def t(m, k=2):
        return float(np.mean(a[m] <= k)) if m.any() else None
    every = np.ones(a.size, bool)
    return {"ranked": int(a.size), "T2": t(every), "T2_linked": t(lk), "T2_unlinked": t(~lk), "T10": t(every, 10),
            "T10_unlinked": t(~lk, 10), "MRR": float(np.mean(1.0 / a)) if a.size else None,
            "T2_unlinked_by_hop": {f"{k}hop": t(~lk & (hp == k)) for k in (2, 3, 4)}}


def explain_summary(e):
    if not e["gate_A"]:
        return None
    g, rel, w, u = (np.concatenate(e[k]) for k in ("gate_A", "rel", "w", "unl"))
    rc = np.clip(rel / np.maximum(rel.sum(1, keepdims=True), 1e-12), 1e-12, 1.0)
    top = rel.argmax(1)
    return {"lists": int(g.size), "gate_A_mean": float(g.mean()),
            "gate_A_mean_lists_with_unlinked": float(g[u].mean()) if u.any() else None,
            "gate_A_mean_lists_linked_only": float(g[~u].mean()) if (~u).any() else None,
            "relation_top_share": [float(np.mean(top == j)) for j in range(REL_K)],
            "relation_entropy_mean": float(-(rc * np.log(rc)).sum(1).mean()),
            "slot_top_share": [float(np.mean(w.argmax(1) == s)) for s in range(SLOTS)]}


def check_lta(lr, split, ds, carve, summ, ranked, linked_share):
    """The free scores' T2s, the A count and the linked share against LTA's column record of the same split and carve."""
    if not lr:
        return {"equal": False, "diffs": {"record": "absent"}}
    diffs = {}
    if (lr.get("stage"), lr.get("split"), lr.get("dataset"), lr.get("carve")) != (LTA.STAGE, split, ds, carve):
        diffs["record"] = [lr.get("stage"), lr.get("split"), lr.get("dataset"), lr.get("carve")]
    if lr.get("A_ranked") != ranked:
        diffs["A_ranked"] = [lr.get("A_ranked"), ranked]
    aa = lr.get("A_assigned") or {}
    pairs = [("A_linked_share", lr.get("A_linked_share"), linked_share)]
    pairs += [(f"{s}.{k}", (aa.get(s) or {}).get(k), summ[s][k]) for s in LTW.FREE for k in ("T2", "T2_linked", "T2_unlinked")]
    for name, x, y in pairs:
        if (x is None) != (y is None) or (x is not None and abs(x - y) > TOL):
            diffs[name] = [x, y]
    return {"equal": not diffs, "diffs": diffs}


def read_path(out, split, ds, carve):
    return Path(out) / "reads" / split / f"{ds}__{carve}.json"


def read(split, ds, carve, root=PKO, out=OUT, nodes=None, host=False, chunk_rows=LTA.CHUNK_ROWS, lta_out=LTA.OUT,
         lists=LISTS):
    """Each A case of the carve ranked among every row outside the top five by LTQ's forms (the fold fit its question
    draws) and the free scores; T2 of every target, the linked and the unlinked; bil's from LTA's column record, which
    the free scores are checked against; what each form's fit uses."""
    import torch
    import lean_cache as LC
    t0 = time.time()
    torch.set_num_threads(LTMC.threads())
    if (ds, carve) not in LTMC.col_carves(split):
        raise SystemExit(f"ltq: {ds}/{carve} is not a carve the split {split} reads")
    training = (ds, carve) in LTMC.train_carves(split)
    z = LTA.own_lists(lta_out, ds, carve, with_a=False)
    if int(z["limit"]):
        raise SystemExit(f"ltq: LTA's lists of {ds}/{carve} were built with a limit")
    models, fits = {}, {}
    for f in FORMS:
        models[f"ltq_{f}"], fits[f"ltq_{f}"] = load_ltq(f, split, out, lists)
    if nodes is None:
        nodes, _ = LTMC.open_nodes([ds])
    nd = nodes[ds]
    fold = z["Q_fold"] if training else z["Q_hfold"]
    nq = int(z["Q_n"].size)
    parts, recs = LC.part_dirs(ds, carve, Path(root) / "cache")
    if [p.name for p in parts] != [str(x) for x in z["P_name"]] or [LC.sha_file(p / "record.json") for p in parts] != [
            str(x) for x in z["P_sha"]]:
        raise SystemExit(f"ltq: {ds}/{carve}: the cache's parts are not the ones LTA's lists were built on")
    take = LTA.xc_take(recs[0]["look"], ds, carve)
    ranks = {s: [] for s in SCORES}
    expl = {s: {"gate_A": [], "rel": [], "w": [], "unl": []} for s in MODELS}
    link, hop, chk = [], [], [0, 0, 0.0]
    qi, nlists = 0, 0

    def flush(ch):
        nonlocal chk, nlists
        if not ch.gq:
            return
        zc = ch.arrays()
        sel = np.arange(zc["T_q"].size)
        b, c_ = LTW.read_chunk(zc, "T", nd, sel, [np.arange(zc["T_off"][i], zc["T_off"][i + 1]) for i in sel])
        chk = [chk[0] + c_[0], chk[1] + c_[1], max(chk[2], c_[2])]
        its = LTW.items_of(b, zc, "T", LTMC.DS_K[ds])
        got = {}
        for s, M in models.items():
            for j, (m, mu, sd) in enumerate(M):
                mine = [it for it in its if fold[ch.gq[int(zc["T_q"][it["list"]])]] == j]
                if not mine:
                    continue
                for it, sc in zip(mine, LTW.score_items(m, mine, mu, sd, batch=TOP)):
                    got.setdefault(it["list"], {})[s] = sc
                ex = m.explain(*(torch.from_numpy(np.stack([it[k] for it in mine]).astype(np.float32))
                                 for k in ("qv", "xa", "rv")))
                for k in ("gate_A", "rel", "w"):
                    expl[s][k].append(ex[k].numpy().astype(np.float64))
                expl[s]["unl"].append(np.asarray([bool((ch.case[int(zc["T_q"][it["list"]])][2] == 0).any())
                                                  for it in mine]))
        for it in its:
            lq = int(zc["T_q"][it["list"]])
            _k, t, bits = ch.case[lq]
            sc = {**got[it["list"]], **{s: it["free"][s] for s in LTW.FREE}}
            for s in SCORES:
                ranks[s] += CH.ranks_in(sc[s], it["loc"], t)
            link.extend(bool(x > 0) for x in bits)
            hop.extend([int(z["Q_hop"][ch.gq[lq]])] * t.size)
            nlists += 1

    with LTMC.on_ds_k():
        for pi, (p, r) in enumerate(zip(parts, recs)):
            Ar, E, lk_sha = LTW.part_arrays(p, r, Path(root) / "links")
            if lk_sha != str(z["P_links_sha"][pi]):
                raise SystemExit(f"ltq: {ds}/{carve}/{p.name}: its links are not the ones LTA's lists were built on")
            xc = np.asarray(Ar["xc"][:, take], np.float32)
            gold, node = np.asarray(Ar["gold"], bool), np.asarray(Ar["node"], np.int64)
            nqa, gtot = np.asarray(Ar["n"], np.int64), np.asarray(Ar["gold_total"], np.int64)
            eu, ev, ef = np.asarray(E["e_u"]), np.asarray(E["e_v"]), np.asarray(E["e_fam"])
            no = np.r_[0, np.cumsum(nqa)]
            eo = np.r_[0, np.cumsum(np.asarray(E["q_ne"], np.int64))]
            ch = AChunk()
            for i in range(nqa.size):
                qd = LTA.question(xc, gold, node, nqa, eu, ev, ef, no, eo, i)
                if qd["n"] != int(z["Q_n"][qi]) or pi != int(z["Q_part"][qi]):
                    raise SystemExit(f"ltq: {ds}/{carve}: question {qi} is not its lists' question")
                ch.add(qi, qd, z["qe"][qi], int(gtot[i]))
                qi += 1
                if ch.rows >= chunk_rows:
                    flush(ch)
                    ch = AChunk()
                if qi % 1000 == 0:
                    log(f"  read {split} {ds}/{carve}: {qi} questions, {nlists} A lists ({time.time() - t0:.0f}s)")
            flush(ch)
            del Ar, E, xc, gold, node, eu, ev, ef
    if qi != nq:
        raise SystemExit(f"ltq: {ds}/{carve}: read {qi} questions, LTA's lists hold {nq}")
    summ = {s: summary(ranks[s], link, hop) for s in SCORES}
    lrp = LTMC.cols_dir(lta_out, split, ds, carve) / "record.json"
    lr = json.loads(lrp.read_text(encoding="utf-8")) if lrp.is_file() else {}
    lb = (lr.get("A_assigned") or {}).get("bil") or {}
    summ["bil"] = {"T2": lb.get("T2"), "T2_linked": lb.get("T2_linked"), "T2_unlinked": lb.get("T2_unlinked"),
                   "MRR": lb.get("MRR"), "T2_by_hop": lb.get("T2_by_hop"), "from": "LTA's column record"}
    lshare = float(np.mean(link)) if link else None
    ident = LTMC.identity(chk)
    rec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "dataset": ds, "carve": carve,
           "training_carve": training, "questions": nq, "A_lists": nlists, "A_ranked": len(ranks[MODELS[0]]),
           "linked_share": lshare, "scores": summ,
           "check": check_lta(lr, split, ds, carve, summ, len(ranks[MODELS[0]]), lshare),
           "lta_record": LTA.rel(lrp), "lta_record_sha256": LC.sha_file(lrp) if lrp.is_file() else None,
           "explain": {s: explain_summary(expl[s]) for s in MODELS}, "identity": ident, "fits": fits,
           "form_sha256": form_sha(), "host": bool(host), "script_sha256": WQ.sha_src(__file__),
           "seconds": time.time() - t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = read_path(out, split, ds, carve)
    p.parent.mkdir(parents=True, exist_ok=True)
    LTW.write_json(p, rec)

    def t2(s, k):
        v = summ[s][k]
        return "-" if v is None else f"{v:.4f}"
    log(f"read {split} {ds}/{carve}: {rec['A_ranked']} targets ranked (linked {lshare}); unlinked T2 " +
        ", ".join(f"{s} {t2(s, 'T2_unlinked')}" for s in ("bil",) + SCORES) + "; linked T2 " +
        ", ".join(f"{s} {t2(s, 'T2_linked')}" for s in ("bil",) + MODELS) +
        f"; LTA's record {'equal' if rec['check']['equal'] else rec['check']['diffs']}; identity {ident['pass']} "
        f"({time.time() - t0:.0f}s)")
    return rec


# ── the rule and the report (section 6) ──────────────────────────────────────


def call(reads, min_ranked=MIN_RANKED):
    """Section 6 on the decisive reads (keyed 'split ds/carve'): (verdict, reading, decision form, problems)."""
    dec = [reads.get(f"{DECIDE_SPLIT} {ds}/{cv}") for ds, cv in DECIDE]
    probs = []
    for (ds, cv), r in zip(DECIDE, dec):
        if r is None:
            probs.append(f"{ds}/{cv}: no read")
            continue
        if not r["check"]["equal"]:
            probs.append(f"{ds}/{cv}: the free scores or counts are not LTA's record's: {r['check']['diffs']}")
        if not r["identity"]["pass"]:
            probs.append(f"{ds}/{cv}: identity {r['identity']}")
        if r["A_ranked"] < min_ranked:
            probs.append(f"{ds}/{cv}: {r['A_ranked']} targets ranked, under {min_ranked}")
        need = [(s, k) for s in ("bil", "pair") + MODELS for k in ("T2_linked", "T2_unlinked")]
        if any(r["scores"][s][k] is None for s, k in need):
            probs.append(f"{ds}/{cv}: a T2 the rule reads is empty")
    if probs:
        return "INCOMPLETE", None, None, probs
    u = {f: dec[0]["scores"][f"ltq_{f}"]["T2_unlinked"] for f in FORMS}
    d = max(FORMS, key=lambda f: (u[f], f == "u"))
    unl = all(r["scores"][f"ltq_{d}"]["T2_unlinked"] >= r["scores"]["pair"]["T2_unlinked"] + PAIR_MARGIN - TOL for r in dec)
    lnk = all(r["scores"][f"ltq_{d}"]["T2_linked"] >= r["scores"]["bil"]["T2_linked"] - LINKED_SLACK - TOL for r in dec)
    if unl and lnk:
        return "SIGNAL", None, d, []
    return "NO_SIGNAL", ("BELOW_PAIR" if not unl else "TRADE"), d, []


def report(out=OUT, min_ranked=MIN_RANKED):
    """Every read on disk, the fits' stop T2, what each form uses, and section 6's call (the decisive reads; the rest
    beside)."""
    reads, fits = {}, {}
    for sp, carves in READS.items():
        for ds, cv in carves:
            p = read_path(out, sp, ds, cv)
            if p.is_file():
                reads[f"{sp} {ds}/{cv}"] = json.loads(p.read_text(encoding="utf-8"))
        for f in FORMS:
            p = Path(out) / f / f"fit_{sp}.json"
            if p.is_file():
                fits.setdefault(sp, {})[f] = [x["best_stop_T2"] for x in json.loads(p.read_text(encoding="utf-8"))["folds"]]
    verdict, reading, form, probs = call(reads, min_ranked)
    rep = {"stage": STAGE, "declared_in": DECLARED, "verdict": verdict, "reading": reading, "decision_form": form,
           "problems": probs, "rule": {"decisive": [f"{DECIDE_SPLIT} {ds}/{cv}" for ds, cv in DECIDE],
                                       "pair_margin": PAIR_MARGIN, "linked_slack": LINKED_SLACK, "min_ranked": min_ranked},
           "fits": fits, "reads": {k: {x: r[x] for x in ("A_ranked", "linked_share", "scores", "check", "identity",
                                                        "training_carve", "explain")} for k, r in reads.items()},
           "script_sha256": WQ.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def t(v):
        return "-" if v is None else f"{v:.3f}"
    show = ("bil",) + MODELS + ("pair", "res", "rrf")
    L = [f"# LTQ: a question-gated next-hop scorer with relations and IRA's refinement (declared in {DECLARED})", "",
         f"**{verdict}{' (' + reading + ')' if reading else ''}**" + (f", decision form {form}" if form else "") +
         ". The rule reads WQ-hotpotqa musique/s1fit and musique/s1eval: the decision form's unlinked T2 at least pair's "
         f"+ {PAIR_MARGIN} and its linked T2 at least bil's - {LINKED_SLACK} on both.", ""]
    L += [f"- {x}" for x in probs] + ([""] if probs else [])
    L += ["## Unlinked targets' T2 (every row outside rrf's top five)", "",
          "| split | carve | targets | linked share | " + " | ".join(show) + " | LTA's record |",
          "|---|---|---:|---:|" + "---:|" * len(show) + "---|"]
    for k, r in reads.items():
        sp, cv = k.split(" ", 1)
        L.append(f"| {sp} | {cv} | {r['A_ranked']} | {t(r['linked_share'])} | " +
                 " | ".join(t(r["scores"][s]["T2_unlinked"]) for s in show) +
                 f" | {'equal' if r['check']['equal'] else 'NOT equal'} |")
    L += ["", "## Linked targets' T2", "", "| split | carve | " + " | ".join(show) + " |", "|---|---|" + "---:|" * len(show)]
    for k, r in reads.items():
        sp, cv = k.split(" ", 1)
        L.append(f"| {sp} | {cv} | " + " | ".join(t(r["scores"][s]["T2_linked"]) for s in show) + " |")
    L += ["", "## What each form's fit uses (decides nothing)", "",
          "| split | carve | form | gate on A | gate on A, lists with an unlinked target | top relation's share | "
          "relation entropy |", "|---|---|---|---:|---:|---:|---:|"]
    for k, r in reads.items():
        sp, cv = k.split(" ", 1)
        for s in MODELS:
            e = r["explain"].get(s)
            if e:
                L.append(f"| {sp} | {cv} | {s} | {t(e['gate_A_mean'])} | {t(e['gate_A_mean_lists_with_unlinked'])} | "
                         f"{t(max(e['relation_top_share']))} | {t(e['relation_entropy_mean'])} |")
    L += ["", "## Fold fits' stop T2", "", "| split | form | folds |", "|---|---|---|"]
    L += [f"| {sp} | {f} | {json.dumps(v)} |" for sp, d in fits.items() for f, v in d.items()]
    Path(out).mkdir(parents=True, exist_ok=True)
    LTW.write_json(Path(out) / "report.json", rep)
    (Path(out) / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"report: {verdict} {reading or ''} (form {form}); {len(reads)} reads; problems {probs}")
    return rep


# ── self-test (laptop or host, CPU) ──────────────────────────────────────────


def selftest():
    import torch
    import lean_cache as LC
    sp = "WQ-hotpotqa"
    saved = {k: getattr(LTMC, k) for k in ("STAGE", "DECLARED", "OUT", "FORM", "lists_path", "lists")}
    saved_w = {k: getattr(LTW, k) for k in ("build_model", "list_loss", "DS_K")}
    tmp = Path(tempfile.mkdtemp(prefix="ltq_"))
    try:
        # 1. the scorer: A is LTW's bil; B reads no anchor cosine and no link bit; p and the gate start at zero
        d, nf = 24, len(LTW.FEATS)
        m, bil = build(FORM, nf, d), LTW_BUILD("bil", nf, d)
        sa_ = {k[2:]: v for k, v in m.state_dict().items() if k.startswith("a.")}
        assert sa_.keys() == bil.state_dict().keys() and all(torch.equal(sa_[k], v) for k, v in bil.state_dict().items())
        assert build("bil", nf, d).state_dict().keys() == bil.state_dict().keys() and "bi" not in m.state_dict()
        g = torch.Generator().manual_seed(3)
        F, X = torch.randn(3, 7, nf, generator=g), torch.randn(3, 7, d, generator=g)
        q, xa, r = (torch.nn.functional.normalize(torch.randn(3, d, generator=g), dim=-1) for _ in range(3))
        sa0, sb0, gam0 = m.parts(F, X, q, xa, r)
        hp0 = m.hop(q, xa, r)
        assert torch.all(gam0 == 0) and torch.all(hp0["p"] == 0)
        assert torch.allclose(sb0, m.b(F[..., list(B_IDX)]).squeeze(-1), atol=1e-6)
        assert torch.allclose(m(F, X, q, xa, r), torch.logaddexp(sa0, sb0) + np.log(0.5), atol=1e-5)
        assert torch.allclose(hp0["w"], torch.full((3, SLOTS), 1.0 / SLOTS))
        # IRA's plan: each relation's mass over the slots and the dustbin is 1 (exactly), each slot's about 1
        h = torch.einsum("btd,tdk->bk", torch.stack((q, xa, r), 1), m.rV).unsqueeze(1) + m.slot.unsqueeze(0)
        P = plan(torch, h, m.key, m.bz, m.b0)
        lg = torch.cat([torch.einsum("bsr,kr->bks", h, m.key) / math.sqrt(LTW.RANK), torch.zeros(3, REL_K, 1)], 2)
        assert P.shape == (3, REL_K, SLOTS) and float(P.sum(2).max()) < 1 + 1e-6
        assert float((P.sum(1) - 1).abs().max()) < 0.02, P.sum(1)
        Pf = torch.exp(lg - lg.logsumexp(2, keepdim=True))                  # one row step alone: the slots do not hold
        assert float((Pf[:, :, :SLOTS].sum(1) - 1).abs().max()) > float((P.sum(1) - 1).abs().max())
        with torch.no_grad():
            for p_ in (m.Uo, m.off, m.a.U, m.gate.weight, m.step, m.ptr):
                p_.normal_(0.0, 0.1, generator=g)
        sa, sb, gam = m.parts(F, X, q, xa, r)
        F2 = F.clone()
        F2[..., [LTW.FEATS.index(k) for k in ("anc", "fam_struct", "fam_ner", "fam_knn")]] += 5.0
        sa2, sb2, _ = m.parts(F2, X, q, xa, r)
        assert torch.equal(sb, sb2) and not torch.allclose(sa, sa2) and not torch.all(gam == 0)
        assert not torch.all(m.hop(q, xa, r)["p"] == 0)
        ref = torch.log(torch.sigmoid(gam) * torch.exp(sa) + torch.sigmoid(-gam) * torch.exp(sb))
        assert torch.allclose(m(F, X, q, xa, r), ref, atol=1e-4)
        ex = m.explain(q, xa, r)
        assert ex["gate_A"].shape == (3,) and ex["rel"].shape == (3, REL_K)
        assert torch.all((ex["gate_A"] > 0) & (ex["gate_A"] < 1))
        # every one of B's learned parts gets a gradient once p is off zero
        m.zero_grad()
        m(F, X, q, xa, r).sum().backward()
        for nm in ("rV", "slot", "key", "off", "Uo", "step", "bz", "b0", "ptr", "ptr_b"):
            assert getattr(m, nm).grad is not None and float(getattr(m, nm).grad.abs().sum()) > 0, nm
        # 2. the loss: each target its own softmax; an unlinked target beside a dominant one keeps its gradient
        s = torch.randn(4, 6, generator=g)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
        pos = torch.tensor([[1, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
        w = torch.tensor([1.0, 2.0, 0.5, 1.0])
        sn, exp = s.numpy().astype(np.float64), 0.0
        for i in range(4):
            neg = [j for j in range(6) if mask[i, j] and not pos[i, j]]
            per = [np.log(np.exp(sn[i, t_]) + sum(np.exp(sn[i, j]) for j in neg)) - sn[i, t_]
                   for t_ in range(6) if pos[i, t_]]
            exp += float(w[i]) * float(np.mean(per))
        assert abs(float(gold_loss(s, mask, pos, w)) - exp / 4) < 1e-5
        s = torch.tensor([[10.0, 0.0, 1.0, 1.0]], requires_grad=True)
        mk, ps = torch.ones(1, 4, dtype=torch.bool), torch.tensor([[True, True, False, False]])
        LTW_LOSS(s, mk, ps, torch.ones(1)).backward()
        g_old, s.grad = float(s.grad[0, 1]), None
        gold_loss(s, mk, ps, torch.ones(1)).backward()
        assert abs(g_old) < 1e-3 and float(s.grad[0, 1]) < -0.2, (g_old, float(s.grad[0, 1]))
        # 3. toy carves, LTM's, LTA's and LTH's lists, LTA's fit and column records (ltacols' own self-test's toys)
        rng = np.random.default_rng(17)
        M = LTW.unit(rng.normal(size=(400, 24))).astype(np.float32)
        sizes = {"musique": [[40, 300, 23, 61, 40, 17, 33, 40, 270, 29], [52, 40, 23, 300, 36, 40, 18, 44]],
                 "2wiki": [[40, 23, 61, 33, 40, 29, 52, 40, 36, 44, 27, 31, 40, 38]],
                 "webqsp": [[40, 280, 23, 33, 40, 29], [61, 40, 300, 25]], "squad": [[12, 9, 14, 11, 10, 13]]}
        for ds, cv in LTMC.col_carves(sp):
            LTA.toy_carve(tmp, ds, cv, M, rng, sizes.get(ds, [[40, 23, 61, 33], [29, 300, 40]]) if cv != LTMC.EVAL_CARVE
                          else [[40, 23, 61, 33, 270], [29, 40, 36]])
        nodes = {ds: LTW.FakeNodes(M) for ds in LTMC.DS_K}
        ltm, lta, lth, out = tmp / "ltm", tmp / "lta", tmp / "lth", tmp / "ltq"
        L = {"u": lta, "h": lth}
        for ds, cv in LTMC.col_carves(sp):
            LTMC.lists(ds, cv, root=tmp, out=ltm)
            LTA.lists(ds, cv, root=tmp, out=lta, ltm_out=ltm)
            LTH.lists(ds, cv, root=tmp, out=lth, ltm_out=ltm)
        LTA.fit(sp, root=tmp, out=lta, nodes=nodes, epochs=2, min_lists=1, ltm_out=ltm)
        for ds, cv in DECIDE:
            LTA.cols(sp, ds, cv, root=tmp, out=lta, nodes=nodes, ltm_out=ltm)
        # 4. LTQ's fits: ltmcols' under LTQ's names, on each form's lists, the same folds; every name put back
        recs = {}
        for f in FORMS:
            recs[f] = fit(f, sp, root=tmp, out=out, nodes=nodes, epochs=2, min_lists=1, lists=L)
            rf = recs[f]
            assert (rf["stage"], rf["declared_in"], rf["form"], rf["ltq"]["form"]) == (STAGE, DECLARED, FORM, f)
            assert rf["ltq"]["form_sha256"] == form_sha() and rf["identity"]["pass"] and len(rf["folds"]) == K
            assert rf["carves"]["musique"]["lists_sha256"] == LC.sha_file(LTMC.lists_path(L[f], "musique", "s1fit"))
            assert {k: getattr(LTMC, k) for k in saved} == saved and {k: getattr(LTW, k) for k in saved_w} == saved_w
        assert recs["u"]["carves"]["musique"]["fold_lists"] == recs["h"]["carves"]["musique"]["fold_lists"]
        assert recs["u"]["folds"][0]["file_sha256"] != recs["h"]["folds"][0]["file_sha256"]
        assert recs["u"]["folds"][0]["parameters"] > json.loads((lta / f"fit_{sp}.json").read_text(
            encoding="utf-8"))["folds"][0]["parameters"]
        # 5. the reads: the free scores and counts meet LTA's column records; any chunk size gives the same ranks
        rs = {}
        for ds, cv in DECIDE:
            rd = read(sp, ds, cv, root=tmp, out=out, nodes=nodes, lta_out=lta, lists=L, chunk_rows=150)
            assert rd["check"]["equal"], rd["check"]
            assert rd["identity"]["pass"] and rd["A_ranked"] > 0 and set(rd["scores"]) == set(SCORES) | {"bil"}
            assert rd["training_carve"] == (cv == "s1fit") and rd["scores"]["bil"]["T2"] is not None
            assert all(rd["explain"][s]["lists"] == rd["A_lists"] for s in MODELS)
            rs[f"{sp} {ds}/{cv}"] = rd
        rd2 = read(sp, "musique", "s1eval", root=tmp, out=out, nodes=nodes, lta_out=lta, lists=L, chunk_rows=10 ** 9)
        assert rd2["scores"] == rs[f"{sp} musique/s1eval"]["scores"] and rd2["check"]["equal"]
        assert {k: getattr(LTMC, k) for k in saved} == saved and {k: getattr(LTW, k) for k in saved_w} == saved_w
        # 6. refusals: a fit by another scorer; a carve outside the reads; lists with a limit; LTA's record not met
        fj = out / "u" / f"fit_{sp}.json"
        fr = json.loads(fj.read_text(encoding="utf-8"))
        LTW.write_json(fj, {**fr, "ltq": {**fr["ltq"], "form_sha256": "0" * 64}})
        LTA.must_stop(read, sp, "musique", "s1fit", root=tmp, out=out, nodes=nodes, lta_out=lta, lists=L)
        LTW.write_json(fj, fr)
        LTA.must_stop(read, sp, "hotpotqa", "fit", root=tmp, out=out, nodes=nodes, lta_out=lta, lists=L)
        LTA.lists("squad", "fit", root=tmp, out=tmp / "lim", ltm_out=ltm, limit=3)
        lim = tmp / "lim2"
        shutil.copytree(lta / "work", lim / "work")
        shutil.copyfile(LTMC.lists_path(tmp / "lim", "squad", "fit"), LTMC.lists_path(lim, "squad", "fit"))
        LTA.must_stop(fit, "u", sp, root=tmp, out=tmp / "ltq_lim", nodes=nodes, epochs=1, min_lists=1,
                      lists={"u": lim, "h": lth})
        cr = LTMC.cols_dir(lta, sp, "musique", "s1eval") / "record.json"
        crec = json.loads(cr.read_text(encoding="utf-8"))
        LTW.write_json(cr, {**crec, "A_assigned": {**crec["A_assigned"],
                                                   "pair": {**crec["A_assigned"]["pair"], "T2_unlinked": -1.0}}})
        for f in FORMS:
            shutil.copytree(out / f, tmp / "ltq_bad" / f)
        rb = read(sp, "musique", "s1eval", root=tmp, out=tmp / "ltq_bad", nodes=nodes, lta_out=lta, lists=L)
        assert not rb["check"]["equal"] and set(rb["check"]["diffs"]) == {"pair.T2_unlinked"}, rb["check"]
        assert call({f"{sp} musique/s1fit": rs[f"{sp} musique/s1fit"], f"{sp} musique/s1eval": rb})[0] == "INCOMPLETE"
        LTW.write_json(cr, crec)
        # 7. the rule
        key = [f"{DECIDE_SPLIT} {ds}/{cv}" for ds, cv in DECIDE]

        def mk(u_unl, h_unl, pair_unl, lnk, bil_lnk, eq=True, n=500):
            sc = {x: {"T2_unlinked": 0.1, "T2_linked": 0.3} for x in SCORES + ("bil",)}
            sc.update({"ltq_u": {"T2_unlinked": u_unl, "T2_linked": lnk}, "ltq_h": {"T2_unlinked": h_unl, "T2_linked": lnk},
                       "pair": {"T2_unlinked": pair_unl, "T2_linked": 0.05}, "bil": {"T2_unlinked": 0.09, "T2_linked": bil_lnk}})
            return {"check": {"equal": eq, "diffs": {} if eq else {"x": 1}}, "identity": {"pass": True}, "A_ranked": n,
                    "scores": sc}
        assert call({key[0]: mk(.20, .18, .144, .33, .331), key[1]: mk(.15, .30, .113, .34, .346)})[:3] == ("SIGNAL", None, "u")
        assert call({key[0]: mk(.20, .18, .144, .33, .331), key[1]: mk(.14, .30, .113, .34, .346)})[:3] == \
            ("NO_SIGNAL", "BELOW_PAIR", "u")
        assert call({key[0]: mk(.20, .18, .144, .30, .331), key[1]: mk(.15, .30, .113, .34, .346)})[:3] == \
            ("NO_SIGNAL", "TRADE", "u")
        assert call({key[0]: mk(.20, .20, .144, .33, .331), key[1]: mk(.15, .10, .113, .34, .346)})[2] == "u"
        assert call({key[0]: mk(.10, .20, .144, .33, .331), key[1]: mk(.30, .15, .113, .34, .346)})[:3] == ("SIGNAL", None, "h")
        assert call({key[0]: mk(.20, .18, .144, .33, .331)})[0] == "INCOMPLETE"
        assert call({key[0]: mk(.20, .18, .144, .33, .331, eq=False), key[1]: mk(.15, .3, .113, .34, .346)})[0] == "INCOMPLETE"
        assert call({key[0]: mk(.20, .18, .144, .33, .331, n=199), key[1]: mk(.15, .3, .113, .34, .346)})[0] == "INCOMPLETE"
        rep = report(out, min_ranked=1)
        assert rep["verdict"] in ("SIGNAL", "NO_SIGNAL") and (out / "report.md").is_file() and set(rep["fits"][sp]) == set(FORMS)
        assert len(rep["reads"]) == 2 and report(out)["verdict"] == "INCOMPLETE"
    finally:
        for k, v in saved.items():
            setattr(LTMC, k, v)
        for k, v in saved_w.items():
            setattr(LTW, k, v)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ltqdiag selftest: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fit", "read", "report"))
    ap.add_argument("--form", choices=FORMS)
    ap.add_argument("--split", choices=tuple(READS))
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.cmd == "report":
        report()
        return 0
    if a.host:
        import lean_host as LH
        LH.substitute()                          # the verified mirror in place of the package, as ltmcols --host
    if a.cmd == "fit" and a.form and a.split:
        fit(a.form, a.split, host=a.host)
        return 0
    if a.cmd == "read" and a.split and (a.dataset, a.carve) in READS[a.split]:
        read(a.split, a.dataset, a.carve, host=a.host)
        return 0
    ap.error("fit --form --split, read --split --dataset --carve (a carve section 5 reads), report, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
