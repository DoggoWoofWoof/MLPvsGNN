"""ENC E2B (docs/ENC_ENCODER_COMPARISON.md section 16, declared 20 Nov 2026 before any E2B number): the best vectors per
encoder, before the encoder is chosen. E2S's parts (q-map, fuse, windows) and a pseudo-relevance step are combined per
encoder, each arm's combination is chosen on the s1sel carves, and it is read once on E0's s1eval carves (SQuAD,
MuSiQue). HotpotQA is read beside, with E2N's node memory.

    python outputs/mp_unified/zenc2b.py selftest [--real]
    python outputs/mp_unified/zenc2b.py winenc --enc ENC --device cuda   (host GPU; ENC: mpnetqa, qwen3e06 or frozen)
    python outputs/mp_unified/zenc2b.py maps --threads 8                  (host CPU: qwen3e06's and mpnetqa's q-maps)
    python outputs/mp_unified/zenc2b.py grid --arm ARM --threads 8        (host CPU; ARM: mpnet, qwen3e06 or frozen)
    python outputs/mp_unified/zenc2b.py hotpot --threads 8                (host CPU; beside, no call)
    python outputs/mp_unified/zenc2b.py report

Two recipes per arm (16.3): V keeps one vector per question and per document (q-map, fuse, prf), so the models can read
it; F adds the windows, which change the first-stage list but are not a vector. Nothing here changes an encoder's
weights, a model, a world, or anything outside outputs/enc/e2b/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import zenc as E  # noqa: E402  (first: sets the BLAS threads before numpy loads)
import zenc2m as M  # noqa: E402
import zenc2s as S  # noqa: E402
import zenc2n as N  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "enc" / "e2b"
E0 = ROOT / "outputs" / "enc" / "e0"
E2S = ROOT / "outputs" / "enc" / "e2s"
E2N = ROOT / "outputs" / "enc" / "e2n"
E2R = ROOT / "outputs" / "enc" / "e2" / "report.json"
STAGE = "ENC E2B"
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 16"
DS = ("squad", "musique")
HOT = "hotpotqa"
ARMS = {"mpnet": ("mpnet", "mpnetqa"), "qwen3e06": ("qwen3e06",), "frozen": ("frozen",)}
SMALL = ("mpnet", "qwen3e06")
NEW_WIN = ("mpnetqa", "qwen3e06", "frozen")
NEW_MAPS = ("qwen3e06", "mpnetqa")
HOT_ARMS = ("mpnet", "frozen")
GAMMAS = (0.0, 0.5, 1.0)
WINS = (64, 126)
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
PRFS = ((3, 0.25), (3, 0.5), (3, 1.0), (10, 0.25), (10, 0.5), (10, 1.0))
MAP_FORM, MAP_LR = "linear", 1e-4
G1_SHARE, G2_COS, G2_SHARE, G4_SHARE = 0.995, 0.99, 0.995, 0.995
K2 = 2000
QB = 128
MODELS = ("zrckpn", "gnx")


def log(msg):
    print(f"[zenc2b {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zenc2b: {msg}")


def code_shas():
    return {f"outputs/mp_unified/{Path(p).name}": E.sha256(Path(p))
            for p in (__file__, E.__file__, M.__file__, S.__file__, N.__file__)}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rjson(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ── q-maps ───────────────────────────────────────────────────────────────────

def map_source(enc):
    """(the map's npz, its sha256 of record)."""
    if enc in ("mpnet", "frozen"):
        rec = rjson(E2S / "adapt.json")["maps"]["adapt" if enc == "mpnet" else "adapt-frozen"]
        return E2S / "maps" / rec["npz"], rec["sha256"]
    rec = rjson(OUT / "maps.json")["maps"][enc]
    return OUT / "maps" / rec["npz"], rec["sha256"]


def load_map(enc):
    p, want = map_source(enc)
    if E.sha256(p) != want:
        stop(f"{p.name}: sha256 differs from its record")
    z = np.load(p, allow_pickle=False)
    return {k.replace("__", "."): np.asarray(z[k]) for k in z.files if k != "form"}, str(z["form"])


def qmap(mp, X, gamma, block=65536):
    """unit(x + gamma g(x)); gamma 0 returns X itself (the raw read, as stored)."""
    if gamma == 0.0:
        return X
    import torch
    state, form = mp
    g = S.make_g(form, X.shape[1])
    g.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in state.items()})
    out = np.empty(X.shape, np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], block):
            x = torch.from_numpy(np.ascontiguousarray(X[i:i + block], np.float32))
            out[i:i + block] = torch.nn.functional.normalize(x + float(gamma) * g(x), dim=1).numpy()
    return out


def save_map(enc, state, form):
    d = OUT / "maps"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{enc}.npz"
    np.savez(p, form=np.asarray(form), **{k.replace(".", "__"): v for k, v in state.items()})
    return {"npz": p.name, "sha256": E.sha256(p)}


# ── windows ──────────────────────────────────────────────────────────────────

def win_vec_path(enc, ds, W):
    return (E2S / "win" / f"{ds}_w{W}.npy") if enc == "mpnet" else (OUT / "win" / f"{enc}_{ds}_w{W}.npy")


def win_doc_path(ds, W):
    return E2S / "win" / f"{ds}_w{W}_doc.npy"


def load_win(enc, ds, W, nd):
    Wv = np.asarray(np.load(win_vec_path(enc, ds, W)), np.float32)
    doc = np.load(win_doc_path(ds, W)).astype(np.int64)
    if doc.size != Wv.shape[0] or np.any(np.diff(doc) < 0):
        stop(f"{enc} {ds} W={W}: the window index does not match the vectors")
    starts = np.flatnonzero(np.r_[True, doc[1:] != doc[:-1]])
    if starts.size != nd or not np.array_equal(doc[starts], np.arange(nd)):
        stop(f"{enc} {ds} W={W}: not every document has a window")
    return Wv, starts


def cmd_winenc(a):
    root = E.served_root()
    t0 = time.time()
    if a.enc not in NEW_WIN:
        stop(f"--enc must be one of {NEW_WIN}")
    if a.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            stop("no CUDA device")
    d = OUT / "win"
    rp = d / f"{a.enc}.json"
    if rp.exists():
        stop(f"{rp.name} exists: the windows are encoded once")
    d.mkdir(parents=True, exist_ok=True)
    tok = E.load_model("mpnet", "cpu", E.pinned("mpnet")).tokenizer      # E2S's spans: mpnet's tokenizer
    m = E.load_model(a.enc, a.device, E.pinned(a.enc))
    rec = {"stage": STAGE, "enc": a.enc, "declared_in": DECLARED, "device": a.device, "sets": {}, "g2": {}}
    for ds in DS:
        texts = [t for _, ch in E.doc_chunks(root, ds) for t in ch]
        own = M.load_all(root, a.enc, ds, "docs")
        if len(texts) != own.shape[0]:
            stop(f"{ds}: {len(texts)} texts, {own.shape[0]} vectors")
        for W in WINS:
            t = time.time()
            wt, doc = S.windows_of(tok, texts, W)
            if not np.array_equal(doc, np.load(win_doc_path(ds, W)).astype(np.int64)):
                stop(f"{ds} W={W}: the windows differ from E2S's")
            V, stt = E.encode_texts(m, a.enc, wt, "docs")
            np.save(win_vec_path(a.enc, ds, W), V)
            one = np.flatnonzero(np.bincount(doc, minlength=len(texts)) == 1)
            first = np.searchsorted(doc, one)
            cos = np.sum(M.unit(V[first].astype(np.float32)) * M.unit(own[one]), axis=1)
            share = float(np.mean(cos >= G2_COS)) if one.size else 1.0
            rec["g2"][f"{ds}_w{W}"] = {"one_window_docs": int(one.size), "share_ge": round(share, 6),
                                       "min_cos": round(float(cos.min()), 6) if one.size else None,
                                       "median_cos": round(float(np.median(cos)), 6) if one.size else None,
                                       "pass": share >= G2_SHARE}
            rec["sets"][f"{ds}_w{W}"] = {"windows": int(doc.size), "docs": len(texts), "stats": stt,
                                         "sha256": E.sha256(win_vec_path(a.enc, ds, W)),
                                         "seconds": round(time.time() - t, 1)}
            log(f"winenc {a.enc} {ds} W={W}: {doc.size} windows, G2 {json.dumps(rec['g2'][f'{ds}_w{W}'])}")
    rec.update(code_sha256=code_shas(), seconds=round(time.time() - t0, 1), utc=utc())
    E.wjson(rp, rec)
    return 0


def g2_of(enc):
    if enc == "mpnet":
        return rjson(E2S / "win" / "winenc.json")["g2"]
    return rjson(OUT / "win" / f"{enc}.json")["g2"]


# ── the maps for qwen3e06 and mpnetqa (E2S's recipe) ─────────────────────────

def cmd_maps(a):
    import torch
    torch.set_num_threads(E.THREADS)
    root = E.served_root()
    t0 = time.time()
    if (OUT / "maps.json").exists():
        stop("outputs/enc/e2b/maps.json exists: the maps are trained once")
    golds, cs, ce = {}, {}, {}
    for ds in DS:
        golds[ds], qids = S.gold_rows(root, ds)
        cs[ds], ce[ds] = S.sel_carve(ds, golds[ds], qids), M.carve(root, ds)
    rec = {"stage": STAGE, "declared_in": DECLARED, "form": MAP_FORM, "lr": MAP_LR, "maps": {}}
    for enc in NEW_MAPS:
        t = time.time()
        blocks, sels, info = {}, {}, {}
        for ds in DS:
            blocks[ds], _, info[ds] = S.adapt_data(root, enc, ds, golds[ds], cs[ds])
            if np.isin(blocks[ds]["pd"], ce[ds]["flat"]).any():
                stop(f"{enc} {ds}: a training pair's document is an s1eval gold")
            sels[ds] = (cs[ds], S.rows_of(root, enc, ds, "queries", cs[ds]["rows"]))
        curve, ep, st = S.train_map(blocks, MAP_FORM, MAP_LR, S.EPOCHS,
                                    S.sel_reader(sels, {ds: blocks[ds]["D"] for ds in DS}, MAP_FORM),
                                    score=lambda s: float(np.mean([s[d] for d in DS])))
        rec["maps"][enc] = dict(save_map(enc, st, MAP_FORM), best_epoch=ep, curve=curve, pairs=info,
                                seconds=round(time.time() - t, 1))
        log(f"maps {enc}: best epoch {ep}, s1sel mean R@5 {round(curve[ep]['score'], 6)} (epoch 0 "
            f"{round(curve[0]['score'], 6)})")
        del blocks
    rec.update(threads=E.THREADS, code_sha256=code_shas(), seconds=round(time.time() - t0, 1), utc=utc())
    E.wjson(OUT / "maps.json", rec)
    return 0


# ── scoring ──────────────────────────────────────────────────────────────────

def weights_of(arm, w):
    """[(encoder, weight)] with weight > 0."""
    if arm == "mpnet":
        return [(e, x) for e, x in (("mpnet", 1.0 - w), ("mpnetqa", w)) if x > 0]
    return [(ARMS[arm][0], 1.0)]


def enc_part(Q, P, beta, W):
    """(1 - beta) Q D^T + beta max over each document's windows, for one encoder's block P."""
    if beta == 0.0:
        return Q @ P["D"].T
    Wv, st = P["win"][W]
    mx = np.maximum.reduceat(Q @ Wv.T, st, axis=1)
    if beta == 1.0:
        return mx
    return np.float32(1.0 - beta) * (Q @ P["D"].T) + np.float32(beta) * mx


def mix(parts, wts):
    out = None
    for e, x in wts:
        y = parts[e] if x == 1.0 else np.float32(x) * parts[e]
        out = y if out is None else out + y
    return out


def top_ids(Sc, k):
    part = np.argpartition(-Sc, k - 1, axis=1)[:, :k]
    ps = np.take_along_axis(Sc, part, 1)
    return np.take_along_axis(part, np.lexsort((part, -ps), axis=1), 1)


def prf_q(Q, Dv, k, alpha):
    """unit(q + alpha unit(mean of the top k documents' vectors)); Dv (nq, k, dim)."""
    return M.unit(Q + np.float32(alpha) * M.unit(Dv[:, :k].mean(axis=1))).astype(np.float32)


def score_fn(arm, P, Qm, s):
    """The setting's score for question rows q0:q1; Qm[e] are the q-mapped questions (gamma applied)."""
    wts = weights_of(arm, s["w"])
    used = [e for e, _ in wts]

    def f(q0, q1):
        Qc = {e: Qm[e][q0:q1] for e in used}
        Sc = mix({e: enc_part(Qc[e], P[e], s["beta"], s["W"]) for e in used}, wts)
        if s["prf"] is None:
            return Sc
        k, alpha = s["prf"]
        first = top_ids(Sc, k)
        Qp = {e: prf_q(Qc[e], P[e]["D"][first], k, alpha) for e in used}
        return mix({e: enc_part(Qp[e], P[e], s["beta"], s["W"]) for e in used}, wts)
    return f


def setting(gamma=0.0, beta=0.0, W=None, w=0.0, prf=None):
    return {"gamma": gamma, "beta": beta, "W": W if beta > 0 else None, "w": w, "prf": None if prf is None else list(prf)}


def order_a(s):
    return (s["gamma"], s["beta"], s["w"], s["W"] or 0)


def order_b(s):
    return (0, 0.0, 0) if s["prf"] is None else (1, s["prf"][1], s["prf"][0])


def step_a(arm, data, maps):
    """Every (gamma, beta, W, w) without prf, on s1sel: [{setting, s1sel_r5 {ds}, mean}]. The document and window
    products are computed once per question block and gamma, then mixed per setting."""
    encs = ARMS[arm]
    ws = WEIGHTS if arm == "mpnet" else (0.0,)
    bw = [(b, W) for b in BETAS for W in ((None,) if b == 0.0 else WINS)]
    tops = {}
    for g in GAMMAS:
        for ds in DS:
            cs, _, P = data[ds]
            Qm = {e: qmap(maps[e], P[e]["Qs"], g) for e in encs}
            nq = len(cs["ids"])
            for q0 in range(0, nq, QB):
                q1 = min(nq, q0 + QB)
                Sd = {e: Qm[e][q0:q1] @ P[e]["D"].T for e in encs}
                Sw = {(e, W): np.maximum.reduceat(Qm[e][q0:q1] @ P[e]["win"][W][0].T, P[e]["win"][W][1], axis=1)
                      for e in encs for W in WINS}
                for b, W in bw:
                    part = {e: Sd[e] if b == 0.0 else (Sw[(e, W)] if b == 1.0 else
                                                       np.float32(1.0 - b) * Sd[e] + np.float32(b) * Sw[(e, W)])
                            for e in encs}
                    for w in ws:
                        key = (g, b, W, w, ds)
                        if key not in tops:
                            tops[key] = np.empty((nq, 5), np.int64)
                        tops[key][q0:q1] = top_ids(mix(part, weights_of(arm, w)), 5)
        log(f"grid {arm} step A: gamma={g} done")
    curve = []
    for g in GAMMAS:
        for b, W in bw:
            for w in ws:
                r = {ds: S.r5_of(tops[(g, b, W, w, ds)], data[ds][0]) for ds in DS}
                curve.append({"setting": setting(g, b, W, w), "s1sel_r5": {k: round(v, 6) for k, v in r.items()},
                              "mean": float(np.mean(list(r.values())))})
    return curve


def sel_r5(arm, data, maps, s):
    out = {}
    for ds in DS:
        cs, _, P = data[ds]
        Qm = {e: qmap(maps[e], P[e]["Qs"], s["gamma"]) for e in ARMS[arm]}
        out[ds] = S.r5_of(S.topk(score_fn(arm, P, Qm, s), len(cs["ids"]), 5, qb=QB)[0], cs)
    return out


def step_b(arm, data, maps, base):
    curve = []
    for prf in [None] + list(PRFS):
        s = dict(base, prf=None if prf is None else list(prf))
        r = sel_r5(arm, data, maps, s)
        curve.append({"setting": s, "s1sel_r5": {k: round(v, 6) for k, v in r.items()}, "mean": float(np.mean(list(r.values())))})
        log(f"grid {arm} step B {json.dumps(s)}: {json.dumps(curve[-1]['s1sel_r5'])}")
    return curve


def parts_on(s):
    return [p for p, on in (("gamma", s["gamma"] > 0), ("beta", s["beta"] > 0), ("w", s["w"] > 0),
                            ("prf", s["prf"] is not None)) if on]


def without(s, p):
    t = dict(s)
    if p == "beta":
        t.update(beta=0.0, W=None)
    else:
        t[p] = None if p == "prf" else 0.0
    return t


def save_read(name, ds, c, h, top):
    d = OUT / "reads"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}__{ds}.npz"
    np.savez(p, ids=np.asarray(c["ids"]), rows=c["rows"], gold_total=c["gt"], top20=top[:, :20].astype(np.int32),
             **{f"h_{k}": v.astype(np.int16) for k, v in h.items()})
    return {"npz": p.name, "sha256": E.sha256(p), "recall": {k: round(S.recall_share(v, c["gt"]), 6) for k, v in h.items()}}


def arm_data(root, arm):
    data = {}
    for ds in DS:
        golds, qids = S.gold_rows(root, ds)
        cs, ce = S.sel_carve(ds, golds, qids), M.carve(root, ds)
        P = {}
        for e in ARMS[arm]:
            D = M.load_all(root, e, ds, "docs")
            P[e] = {"D": D, "Qs": S.rows_of(root, e, ds, "queries", cs["rows"]),
                    "Qe": S.rows_of(root, e, ds, "queries", ce["rows"]), "win": {W: load_win(e, ds, W, D.shape[0]) for W in WINS}}
        data[ds] = (cs, ce, P)
    return data


def cmd_grid(a):
    root = E.served_root()
    t0 = time.time()
    if a.arm not in ARMS:
        stop(f"--arm must be one of {tuple(ARMS)}")
    rp = OUT / f"grid-{a.arm}.json"
    if rp.exists():
        stop(f"{rp.name} exists: the arm is read once")
    g2 = {e: g2_of(e) for e in ARMS[a.arm]}
    if not all(v["pass"] for x in g2.values() for v in x.values()):
        stop(f"G2 failed: {json.dumps(g2)}")
    maps = {e: load_map(e) for e in ARMS[a.arm]}
    data = arm_data(root, a.arm)
    log(f"grid {a.arm}: data loaded ({round(time.time() - t0, 1)} s)")
    ca = step_a(a.arm, data, maps)
    pick = lambda cv: S.pick(cv, key=lambda x: x["mean"], order=lambda x: order_a(x["setting"]))  # noqa: E731
    base = {"V": pick([x for x in ca if x["setting"]["beta"] == 0.0])["setting"], "F": pick(ca)["setting"]}
    log(f"grid {a.arm} step A: {json.dumps(base)}")
    cb, chosen = {}, {}
    for r in ("V", "F"):
        if r == "F" and base["F"] == base["V"]:
            cb["F"], chosen["F"] = cb["V"], chosen["V"]
            continue
        cb[r] = step_b(a.arm, data, maps, base[r])
        chosen[r] = S.pick(cb[r], key=lambda x: x["mean"], order=lambda x: order_b(x["setting"]))["setting"]
    log(f"grid {a.arm} chosen: {json.dumps(chosen)}")
    reads, g1 = {}, {}
    raws = [("raw", setting(), ARMS[a.arm][0])] + ([("raw-mpnetqa", setting(w=1.0), "mpnetqa")] if a.arm == "mpnet" else [])
    plan = {}
    for r in ("V", "F"):
        plan[f"{a.arm}-{r}"] = chosen[r]
        for p in parts_on(chosen[r]):
            plan[f"{a.arm}-{r}-no-{p}"] = without(chosen[r], p)
    for ds in DS:
        _, ce, P = data[ds]
        nq = len(ce["ids"])
        for tag, s, enc in raws:
            Qm = {e: qmap(maps[e], P[e]["Qe"], 0.0) for e in ARMS[a.arm]}
            top, _ = S.topk(score_fn(a.arm, P, Qm, s), nq, E.K, qb=QB)
            g1[f"{tag}__{ds}"] = S.g1_check(S.read_of(top, ce), enc, ds, ce)
        done = {}
        for name, s in plan.items():
            key = json.dumps(s, sort_keys=True)
            if key in done:
                reads[f"{name}__{ds}"] = dict(reads[f"{done[key]}__{ds}"], same_as=done[key])
                continue
            Qm = {e: qmap(maps[e], P[e]["Qe"], s["gamma"]) for e in ARMS[a.arm]}
            top, _ = S.topk(score_fn(a.arm, P, Qm, s), nq, E.K, qb=QB)
            reads[f"{name}__{ds}"] = dict(save_read(name, ds, ce, S.read_of(top, ce), top), setting=s)
            done[key] = name
            log(f"grid {a.arm} s1eval {name} {ds}: {json.dumps(reads[f'{name}__{ds}']['recall'])}")
    rec = {"stage": STAGE, "arm": a.arm, "encoders": list(ARMS[a.arm]), "declared_in": DECLARED, "step_a": ca,
           "base": base, "step_b": cb, "chosen": chosen, "g1": g1, "g2": g2,
           "maps": {e: {"npz": map_source(e)[0].name, "sha256": map_source(e)[1]} for e in ARMS[a.arm]},
           "reads": reads, "plan": plan, "threads": E.THREADS, "code_sha256": code_shas(),
           "seconds": round(time.time() - t0, 1), "utc": utc()}
    E.wjson(rp, rec)
    log(f"grid {a.arm} done in {rec['seconds']} s; G1 {json.dumps({k: v['pass'] for k, v in g1.items()})}")
    return 0


# ── HotpotQA, beside ─────────────────────────────────────────────────────────

def doc_vecs(root, enc, ids, Md=None, U=None):
    """(nq, k, dim) document vectors of ids (nq, k); documents in Md take their memory vectors U."""
    flat = ids.reshape(-1)
    u, inv = np.unique(flat, return_inverse=True)
    V = S.rows_of(root, enc, HOT, "docs", u)
    if Md is not None:
        pos = np.searchsorted(Md, u)
        hit = (pos < Md.size) & (Md[np.minimum(pos, Md.size - 1)] == u)
        V[hit] = U[pos[hit]]
    return V[inv].reshape(ids.shape + (V.shape[1],))


def cmd_hotpot(a):
    root = E.served_root()
    t0 = time.time()
    if (OUT / "hotpot.json").exists():
        stop("outputs/enc/e2b/hotpot.json exists: the beside reads run once")
    run = rjson(E2N / "run.json")
    c = M.carve(root, HOT)
    golds, qids = S.gold_rows(root, HOT)
    cs = S.sel_carve(HOT, golds, qids)
    keep = np.setdiff1d(M.split_rows(root, HOT)["train"], cs["rows"])
    if keep.size != run["feedback_questions"] or np.isin(c["rows"], keep).any():
        stop("the feedback questions differ from E2N's")
    qi, di = N.feedback_pairs("gold", keep, golds)
    rec = {"stage": STAGE, "declared_in": DECLARED, "encs": {}, "g1": {}, "g4": {}, "reads": {}}
    for arm in HOT_ARMS:
        t = time.time()
        enc = arm
        V = rjson(OUT / f"grid-{arm}.json")["chosen"]["V"]
        gamma, prf = V["gamma"], V["prf"]
        lam = run["encs"][enc]["gold"]["chosen"]
        paths = E.vec_shards(root, enc, HOT, "docs")
        Qe = S.rows_of(root, enc, HOT, "queries", c["rows"])
        r_i, r_s, _ = E.search(Qe, paths, K2)
        rec["g1"][enc] = S.g1_check(S.read_of(r_i[:, :E.K], c), enc, HOT, c)
        Qk = S.rows_of(root, enc, HOT, "queries", keep)
        Md, Mdir = N.memory_dirs(Qk, qi, di)
        U = N.updated(S.rows_of(root, enc, HOT, "docs", Md), Mdir, lam)
        del Qk
        top, _ = N.merged_top(r_i, r_s, Qe, Md, U, E.K)
        h = S.read_of(top, c)
        zn = np.load(E2N / "reads" / f"{enc}-gold__{HOT}.npz")
        same = {k: round(float(np.mean(h[k] == zn[f"h_{k}"])), 6) for k in h}
        rec["g4"][enc] = {"equal_share": same, "pass": all(v >= G4_SHARE for v in same.values())}
        log(f"hotpot {enc}: raw and memory reads ({round(time.time() - t, 1)} s), G1 {rec['g1'][enc]['pass']}, "
            f"G4 {rec['g4'][enc]['pass']}")
        Qg = qmap(load_map(enc), Qe, gamma)
        g_i, g_s = (r_i, r_s) if gamma == 0.0 else E.search(Qg, paths, K2)[:2]
        er = {"V": V, "applied": {"gamma": gamma, "prf": prf, "w": 0.0}, "lambda": lam, "changed_docs": int(Md.size)}
        # the recipe without memory
        if prf is None:
            top = g_i[:, :E.K]
        else:
            k, alpha = prf
            Qp = prf_q(Qg, doc_vecs(root, enc, g_i[:, :k]), k, alpha)
            top = E.search(Qp, paths, E.K)[0]
        rec["reads"][f"hot-{enc}-V"] = save_read(f"hot-{enc}-V", HOT, c, S.read_of(top, c), top)
        # the recipe with memory
        if prf is None:
            top, n_unch = N.merged_top(g_i, g_s, Qg, Md, U, E.K)
        else:
            first, _ = N.merged_top(g_i, g_s, Qg, Md, U, k)
            Qp = prf_q(Qg, doc_vecs(root, enc, first, Md, U), k, alpha)
            p_i, p_s, _ = E.search(Qp, paths, K2)
            top, n_unch = N.merged_top(p_i, p_s, Qp, Md, U, E.K)
        rec["reads"][f"hot-{enc}-V-mem"] = dict(save_read(f"hot-{enc}-V-mem", HOT, c, S.read_of(top, c), top),
                                                exact1000_share=round(float(np.mean(n_unch >= E.K)), 6))
        er["seconds"] = round(time.time() - t, 1)
        rec["encs"][enc] = er
        log(f"hotpot {enc}: V {json.dumps(rec['reads'][f'hot-{enc}-V']['recall'])}, V+memory "
            f"{json.dumps(rec['reads'][f'hot-{enc}-V-mem']['recall'])}")
    rec.update(threads=E.THREADS, code_sha256=code_shas(), seconds=round(time.time() - t0, 1), utc=utc())
    E.wjson(OUT / "hotpot.json", rec)
    return 0


# ── the report ───────────────────────────────────────────────────────────────

def bars_of(e2, arm):
    """13.5's bars for the arm: -1 point / (the model's E2 gap / the arm's first-stage gap), per carve."""
    out = {}
    for m in MODELS:
        out[m] = {}
        for ds in DS:
            g = e2["cells"][f"{arm} {m} {ds}"]["pts"]
            f = e2["beside"][f"{arm} {ds}"]["first_stage"]["pts"]
            out[m][ds] = round(-1.0 / (g / f), 2)
    return out


def gate_of(gap_pts, wb_lo, bars):
    """16.7 on V: ADVANCE when both carves' gaps to frozen clear one deciding model's bars; else PARTIAL when won back
    starts above 0 on both; else STOP."""
    for m in MODELS:
        if all(gap_pts[ds] >= bars[m][ds] for ds in DS):
            return "ADVANCE", m
    if all(wb_lo[ds] > 0 for ds in DS):
        return "PARTIAL", None
    return "STOP", None


def load_read(name, ds):
    return np.load(OUT / "reads" / f"{name}__{ds}.npz", allow_pickle=False)


def read_name(grid, name, ds):
    r = grid["reads"][f"{name}__{ds}"]
    return r.get("same_as", name)


def cmd_report(a):
    grids = {}
    for arm in ARMS:
        p = OUT / f"grid-{arm}.json"
        if not p.is_file():
            stop(f"{p.name} missing: every arm reads before the report")
        grids[arm] = rjson(p)
    hot = rjson(OUT / "hotpot.json")
    gates = [v["pass"] for g in grids.values() for v in g["g1"].values()]
    gates += [v["pass"] for g in grids.values() for x in g["g2"].values() for v in x.values()]
    gates += [v["pass"] for v in hot["g1"].values()] + [v["pass"] for v in hot["g4"].values()]
    if not all(gates):
        stop("a gate failed: " + json.dumps({a_: {"g1": g["g1"], "g2": g["g2"]} for a_, g in grids.items()})
             + json.dumps({"g1": hot["g1"], "g4": hot["g4"]}))
    e2 = rjson(E2R)
    zf = {ds: np.load(E0 / "frozen" / f"{ds}.npz") for ds in DS}
    rep = {"stage": STAGE, "declared_in": DECLARED, "chosen": {a_: g["chosen"] for a_, g in grids.items()},
           "cells": {}, "loo": {}, "calls": {}, "bars": {}, "hotpot": {}}
    lines = ["# ENC E2B: the best vectors per encoder (report)", "",
             f"Written by `outputs/mp_unified/zenc2b.py report` under {DECLARED}. s1eval R@k x100. Won back = the read "
             "minus E0's raw read of the arm's encoder; gap = minus E0's frozen read; points, paired 95% bootstrap over "
             "questions (2,000, seed 0). Settings chosen on s1sel. V = one vector per question and per document "
             "(q-map, fuse, prf); F = V's parts plus the windows.", ""]
    for a_, g in grids.items():
        lines.append(f"- {a_}: V {json.dumps(g['chosen']['V'])}; F {json.dumps(g['chosen']['F'])}")
    lines += ["", "| arm | recipe | carve | R@5 | won back | gap to frozen | gap to frozen's same recipe | R@20 gap | "
              "R@1000 gap | RRF@5 gap |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a_, g in grids.items():
        base = ARMS[a_][0]
        for r in ("V", "F"):
            cs = {}
            for ds in DS:
                z = load_read(read_name(g, f"{a_}-{r}", ds), ds)
                zb = np.load(E0 / base / f"{ds}.npz")
                if not S.same_ids(z, zb, zf[ds]):
                    stop(f"{a_}-{r} {ds}: the reads hold other questions")
                x = M.per_q(z, "d5")
                c = {"r5": round(100 * float(x.mean()), 2), "won_back": M.gap(x, M.per_q(zb, "d5")),
                     "gap": {k: M.gap(M.per_q(z, k), M.per_q(zf[ds], k)) for k in ("d5", "d20", "d1000", "rrf5")}}
                if a_ != "frozen":
                    zr = load_read(read_name(grids["frozen"], f"frozen-{r}", ds), ds)
                    c["gap_same_recipe"] = M.gap(x, M.per_q(zr, "d5"))
                cs[ds] = c
                rep["cells"][f"{a_}-{r}__{ds}"] = c
                loo = {}
                for p in parts_on(g["chosen"][r]):
                    zl = load_read(read_name(g, f"{a_}-{r}-no-{p}", ds), ds)
                    loo[p] = M.gap(x, M.per_q(zl, "d5"))
                rep["loo"][f"{a_}-{r}__{ds}"] = loo
                sr = c.get("gap_same_recipe")
                lines.append(f"| {a_} | {r} | {ds} | {c['r5']} | {c['won_back']['pts']} {c['won_back']['ci']} | "
                             f"{c['gap']['d5']['pts']} {c['gap']['d5']['ci']} | "
                             f"{'' if sr is None else str(sr['pts']) + ' ' + str(sr['ci'])} | {c['gap']['d20']['pts']} | "
                             f"{c['gap']['d1000']['pts']} | {c['gap']['rrf5']['pts']} |")
            rep["calls"][f"{a_}-{r}"] = {"recipe": S.call_of({ds: cs[ds]["won_back"] for ds in DS})}
            if a_ in SMALL:
                bars = rep["bars"].setdefault(a_, bars_of(e2, a_))
                gate, by = gate_of({ds: cs[ds]["gap"]["d5"]["pts"] for ds in DS},
                                   {ds: cs[ds]["won_back"]["ci"][0] for ds in DS}, bars)
                rep["calls"][f"{a_}-{r}"].update({"gate" if r == "V" else "gate_pointer": gate, "by": by})
    lines += ["", "Calls (16.7): " + json.dumps(rep["calls"]), "", "Bars (13.5's rule, per arm): " + json.dumps(rep["bars"]),
              "", "Leave one out (the recipe minus the recipe with that part set back to raw, R@5 points): "
              + json.dumps({k: {p: [v["pts"], v["ci"]] for p, v in d.items()} for k, d in rep["loo"].items()}), ""]
    lines += ["HotpotQA beside (V's gamma and prf; mpnet's w set to 0; memory = E2N's gold form):", "",
              "| encoder | read | R@5 | won back | gap to frozen (E0) | R@1000 exact share |", "|---|---|---|---|---|---|"]
    zhf = np.load(E0 / "frozen" / f"{HOT}.npz")
    for enc in HOT_ARMS:
        zb = np.load(E0 / enc / f"{HOT}.npz")
        for v in ("V", "V-mem"):
            name = f"hot-{enc}-{v}"
            z = load_read(name, HOT)
            if not S.same_ids(z, zb, zhf):
                stop(f"{name}: the reads hold other questions")
            x = M.per_q(z, "d5")
            c = {"r5": round(100 * float(x.mean()), 2), "won_back": M.gap(x, M.per_q(zb, "d5")),
                 "gap": M.gap(x, M.per_q(zhf, "d5")), "exact1000_share": hot["reads"][name].get("exact1000_share", 1.0)}
            rep["hotpot"][name] = c
            lines.append(f"| {enc} | {v} | {c['r5']} | {c['won_back']['pts']} {c['won_back']['ci']} | "
                         f"{c['gap']['pts']} {c['gap']['ci']} | {c['exact1000_share']} |")
    lines += ["", "One read per arm and recipe on s1eval; every setting was chosen on s1sel."]
    rep["utc"] = utc()
    E.wjson(OUT / "report.json", rep)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report: {json.dumps(rep['calls'])}")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def selftest_toy():
    import torch
    rng = np.random.default_rng(5)
    # qmap: gamma 1 is E2S's apply_map, gamma 0 the input itself, gamma 0.5 the half step
    X = M.unit(rng.standard_normal((9, 16)).astype(np.float32))
    g = S.make_g("linear", 16)
    with torch.no_grad():
        g.weight.normal_(0, 0.3)
        g.bias.normal_(0, 0.1)
    st = S.snapshot(g)
    assert np.allclose(qmap((st, "linear"), X, 1.0), S.apply_map(st, "linear", X), atol=1e-6)
    assert qmap((st, "linear"), X, 0.0) is X
    with torch.no_grad():
        want = torch.nn.functional.normalize(torch.from_numpy(X) + 0.5 * g(torch.from_numpy(X)), dim=1).numpy()
    assert np.allclose(qmap((st, "linear"), X, 0.5), want, atol=1e-6)
    # a toy arm with two encoders and windows: step A's mixed products equal score_fn's on every setting
    nd, nq = 60, 37
    data = {}
    for ds in DS:
        P = {}
        doc = np.sort(np.r_[np.arange(nd), rng.integers(0, nd, 90)])
        st_ = np.flatnonzero(np.r_[True, doc[1:] != doc[:-1]])
        for e in ARMS["mpnet"]:
            D = M.unit(rng.standard_normal((nd, 16)).astype(np.float32))
            win = {W: (M.unit(rng.standard_normal((doc.size, 16)).astype(np.float32)), st_) for W in WINS}
            P[e] = {"D": D, "Qs": M.unit(rng.standard_normal((nq, 16)).astype(np.float32)), "win": win}
        gl = [np.unique(rng.integers(0, nd, rng.integers(1, 3))) for _ in range(nq)]
        flat, off = S.ragged(gl)
        data[ds] = ({"ids": [str(i) for i in range(nq)], "flat": flat, "off": off, "gt": np.diff(off)}, None, P)
    maps = {e: (st, "linear") for e in ARMS["mpnet"]}
    global QB
    qb0, QB = QB, 8
    try:
        ca = step_a("mpnet", data, maps)
        assert len(ca) == len(GAMMAS) * 9 * len(WEIGHTS), len(ca)
        for x in ca[::7] + ca[-3:]:
            r = sel_r5("mpnet", data, maps, x["setting"])
            assert all(abs(r[ds] - x["s1sel_r5"][ds]) < 1e-6 for ds in DS), (x, r)
        # prf: the second pass uses q' = unit(q + alpha unit(mean of the first pass's top k))
        s = setting(0.5, 0.25, 64, 0.25, (3, 0.5))
        cs, _, P = data["squad"]
        Qm = {e: qmap(maps[e], P[e]["Qs"], 0.5) for e in ARMS["mpnet"]}
        got = score_fn("mpnet", P, Qm, s)(0, nq)
        first = top_ids(score_fn("mpnet", P, Qm, dict(s, prf=None))(0, nq), 3)
        Qp = {e: M.unit(Qm[e] + 0.5 * M.unit(P[e]["D"][first].mean(axis=1))) for e in ARMS["mpnet"]}
        want = 0.75 * enc_part(Qp["mpnet"], P["mpnet"], 0.25, 64) + 0.25 * enc_part(Qp["mpnetqa"], P["mpnetqa"], 0.25, 64)
        assert np.allclose(got, want, atol=1e-5)
        # the ends: w 0 is mpnet alone and w 1 mpnetqa alone, exactly
        assert np.array_equal(score_fn("mpnet", P, Qm, setting())(0, nq), Qm["mpnet"] @ P["mpnet"]["D"].T)
        assert np.array_equal(score_fn("mpnet", P, Qm, setting(w=1.0))(0, nq), Qm["mpnetqa"] @ P["mpnetqa"]["D"].T)
        # fuse as one vector: [sqrt(1 - w) v1 ; sqrt(w) v2] gives the fused score (16.3)
        w = 0.25
        cat_q = np.concatenate([np.sqrt(1 - w) * Qm["mpnet"], np.sqrt(w) * Qm["mpnetqa"]], 1)
        cat_d = np.concatenate([np.sqrt(1 - w) * P["mpnet"]["D"], np.sqrt(w) * P["mpnetqa"]["D"]], 1)
        assert np.allclose(cat_q @ cat_d.T, score_fn("mpnet", P, Qm, setting(w=w))(0, nq), atol=1e-5)
    finally:
        QB = qb0
    # ties and leave-one-out
    a1, a2 = setting(0.5, 0.0, None, 0.25), setting(0.0, 0.25, 64, 0.0)
    cv = [{"setting": a1, "mean": 1.0}, {"setting": a2, "mean": 1.0}]
    assert S.pick(cv, key=lambda x: x["mean"], order=lambda x: order_a(x["setting"]))["setting"] == a2
    b1, b2 = dict(a1, prf=[10, 0.25]), dict(a1, prf=None)
    assert S.pick([{"setting": b1, "mean": 1.0}, {"setting": b2, "mean": 1.0}], key=lambda x: x["mean"],
                  order=lambda x: order_b(x["setting"]))["setting"] == b2
    s = setting(1.0, 0.5, 126, 0.5, (3, 1.0))
    assert parts_on(s) == ["gamma", "beta", "w", "prf"]
    assert without(s, "beta") == dict(s, beta=0.0, W=None) and without(s, "prf")["prf"] is None
    assert parts_on(setting()) == []
    # the gate
    bars = {"zrckpn": {"squad": -5.0, "musique": -3.0}, "gnx": {"squad": -5.5, "musique": -2.0}}
    assert gate_of({"squad": -4.0, "musique": -2.9}, {"squad": 1, "musique": 1}, bars) == ("ADVANCE", "zrckpn")
    assert gate_of({"squad": -5.2, "musique": -1.9}, {"squad": 1, "musique": 1}, bars) == ("ADVANCE", "gnx")
    assert gate_of({"squad": -6.0, "musique": -1.0}, {"squad": 0.1, "musique": 0.2}, bars) == ("PARTIAL", None)
    assert gate_of({"squad": -6.0, "musique": -1.0}, {"squad": 0.1, "musique": -0.2}, bars) == ("STOP", None)
    # doc_vecs' memory swap
    Md = np.asarray([2, 5], np.int64)
    U = np.full((2, 4), 9.0, np.float32)
    ids = np.asarray([[5, 1], [2, 2]])
    pos = np.searchsorted(Md, ids.reshape(-1))
    assert pos.tolist() == [1, 0, 0, 0]
    return "toy OK"


def selftest_real():
    """Host (or laptop where the files are): E2S's maps by sha and against apply_map, the windows' index against E2S's,
    E2N's lambdas, 13.5's bars from E2's report, and the raw mpnet read on 64 s1eval questions against E0."""
    import torch
    torch.set_num_threads(E.THREADS)
    root = E.served_root()
    out = {}
    for enc in ("mpnet", "frozen"):
        mp = load_map(enc)
        X = S.rows_of(root, enc, "squad", "queries", np.arange(5))
        assert np.allclose(qmap(mp, X, 1.0), S.apply_map(mp[0], mp[1], X), atol=1e-6)
        out[f"map_{enc}"] = map_source(enc)[0].name
    run = rjson(E2N / "run.json")
    out["e2n_lambda"] = {e: run["encs"][e]["gold"]["chosen"] for e in HOT_ARMS}
    e2 = rjson(E2R)
    out["bars"] = {a_: bars_of(e2, a_) for a_ in SMALL}
    assert out["bars"]["mpnet"] == M.DECLARED_BARS, (out["bars"]["mpnet"], M.DECLARED_BARS)
    tok = E.load_model("mpnet", "cpu", E.pinned("mpnet")).tokenizer
    for ds in DS:
        texts = []
        for _, ch in E.doc_chunks(root, ds, want_shard=lambda s: s == 0):
            if ch is not None:
                texts = ch[:300]
                break
        for W in WINS:
            _, doc = S.windows_of(tok, texts, W)
            ref = np.load(win_doc_path(ds, W)).astype(np.int64)
            assert np.array_equal(doc, ref[:doc.size]) and (ref.size == doc.size or ref[doc.size] == len(texts)), (ds, W)
        ce = M.carve(root, ds)
        n = 64
        sub = {"ids": ce["ids"][:n], "rows": ce["rows"][:n], "spl": ce["spl"][:n], "gt": ce["gt"][:n],
               "flat": ce["flat"][:ce["off"][n]], "off": ce["off"][:n + 1]}
        P = {"mpnet": {"D": M.load_all(root, "mpnet", ds, "docs")}}
        Qm = {"mpnet": S.rows_of(root, "mpnet", ds, "queries", sub["rows"])}
        top, _ = S.topk(score_fn("mpnet", P, Qm, setting()), n, E.K, qb=QB)
        h = S.read_of(top, sub)
        z = np.load(E0 / "mpnet" / f"{ds}.npz")
        same = {k: float(np.mean(h[k] == z[f"h_{k}"][:n])) for k in h}
        assert all(v >= G1_SHARE for v in same.values()), (ds, same)
        out[f"g1_sample_{ds}"] = same
    return out


def cmd_selftest(a):
    print("zenc2b self-test:", selftest_toy(), flush=True)
    if a.real:
        print("zenc2b real:", json.dumps(selftest_real()), flush=True)
    print("zenc2b self-test: OK", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(prog="zenc2b.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("winenc")
    p.add_argument("--enc", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", default="8")
    p = sub.add_parser("maps")
    p.add_argument("--threads", default="8")
    p = sub.add_parser("grid")
    p.add_argument("--arm", required=True)
    p.add_argument("--threads", default="8")
    p = sub.add_parser("hotpot")
    p.add_argument("--threads", default="8")
    sub.add_parser("report")
    p = sub.add_parser("selftest")
    p.add_argument("--real", action="store_true")
    p.add_argument("--threads", default="8")
    a = ap.parse_args()
    return {"winenc": cmd_winenc, "maps": cmd_maps, "grid": cmd_grid, "hotpot": cmd_hotpot, "report": cmd_report,
            "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
