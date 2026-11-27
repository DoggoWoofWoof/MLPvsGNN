"""SPL1 amendment A11 (docs/SPL1_SPARSE_LEG_SCREEN.md section 8.11): C0's speed levers, read against the record's contract.

    python -E -P -B outputs/mp_unified/zspl1l.py selftest                       # laptop, toy model
    python -I -B    outputs/mp_unified/zspl1l.py bench --ds squad               # host card: every arm interleaved on the same blocks -> outputs/spl1/lev/bench_squad.json
    python -I -B    outputs/mp_unified/zspl1l.py call                           # E1 and E2 -> outputs/spl1/lev/pick.json
    python -I -B    outputs/mp_unified/zspl1l.py encode --arm pick|l0|..|l6 --ds D [--if-pick]   # one full arm run: top-1000 list, block rates, non-zeros
                                                                                # (--if-pick: skipped, row done, when the bench picked no arm)
    python -E -P -B outputs/mp_unified/zspl1l.py read                           # laptop: lists against the record's, calls, D2 -> outputs/spl1/lev/read.json
    python -I -B    outputs/mp_unified/zspl1l.py gate --wait FILE [--max-s N]   # host: a row that ends when FILE exists (D2 pushed), fails after N seconds

Run from the workspace root. Weights, tokenizer and the 256-token cut are C0's (outputs/spl1/models/c0, zspl1c.py); the record's pooling is zspl1c.pooled and is the
reference arm l0. Imports zspl1, zspl1c and zworld and edits none of them. Arms and rules: section 8.11.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import zspl1 as S  # noqa: E402
import zspl1c as C  # noqa: E402

LEV = S.OUT / "lev"
DOC = S.DOC + " section 8.11 (A11)"
ARMS = {
    "l0": {"record": True},
    "l1": {"dtype": "fp32", "chunk": 16},
    "l2": {"dtype": "tf32", "chunk": 16},
    "l3": {"dtype": "fp16", "chunk": 16},
    "l4": {"dtype": "bf16", "chunk": 16},
    "l5": {"dtype": "fp16", "chunk": 32, "prefetch": True},
    "l6": {"dtype": "fp16", "chunk": 32, "prefetch": True, "mask": True, "sort": True},     # information only: not the record's vector
}
PICKABLE = ("l1", "l2", "l3", "l4", "l5")
E1_MIN, COS_P1_MIN, NNZ_TOL, LEVEL_LEV = 1.5, 0.999, 0.01, -0.20
READS = ("rrf5", "seeds", "leg20", "leg1000")
DATASETS = S.SCREEN
log, stop = S.log, S.stop


# ── the arms ─────────────────────────────────────────────────────────────────────────────────────────────────

def tokenize(tok, batch):
    return tok([str(t) for t in batch], return_tensors="pt", truncation=True, max_length=C.LEN, padding=True)


def pool_batch(torch, mdl, enc, device, arm):
    """A lever arm's vectors for one padded batch: the maximum over positions of the raw logits, then log1p(relu(.)) on the pooled row (the function is
    non-decreasing, so this is the record's max of log1p(relu(.)) over positions). With arm['mask'] only real tokens are pooled."""
    parts = []
    n = enc["input_ids"].shape[0]
    with torch.no_grad():
        for i in range(0, n, arm["chunk"]):
            x = {k: v[i:i + arm["chunk"]].to(device) for k, v in enc.items()}
            lg = mdl(**x).logits
            if arm.get("mask"):
                lg = lg.masked_fill(~x["attention_mask"].bool().unsqueeze(-1), float("-inf"))
            parts.append(torch.log1p(torch.relu(lg.amax(dim=1).float())).cpu())
    return torch.cat(parts).numpy()


def run_encode(torch, tok, models, device, texts, arm_name):
    """The vectors of `texts` under an arm, as CSR in the order of `texts`; the seconds cover the tokenizer, the forward passes, the copy to the CPU and the
    non-zero count (A7's timing); the CSR conversion is outside. Returns (csr, seconds, non-zeros)."""
    import scipy.sparse as sp
    arm = ARMS[arm_name]
    n = len(texts)
    t0 = time.perf_counter()
    order = np.arange(n)
    if arm.get("sort"):
        lens = np.asarray([len(x) for x in tok([str(t) for t in texts], truncation=True, max_length=C.LEN)["input_ids"]])
        order = np.argsort(-lens, kind="stable")
    groups = [order[i:i + C.BATCH] for i in range(0, n, C.BATCH)]
    dense = []
    if arm.get("record"):
        torch.set_float32_matmul_precision("highest")
        for g in groups:
            dense.append(C.pooled(torch, tok, models["fp32"], [texts[j] for j in g], device)[0])
    else:
        torch.set_float32_matmul_precision("high" if arm["dtype"] == "tf32" else "highest")
        mdl = models["fp32" if arm["dtype"] == "tf32" else arm["dtype"]]
        if arm.get("prefetch"):
            with ThreadPoolExecutor(1) as ex:
                fut = ex.submit(tokenize, tok, [texts[j] for j in groups[0]])
                for k in range(len(groups)):
                    enc = fut.result()
                    if k + 1 < len(groups):
                        fut = ex.submit(tokenize, tok, [texts[j] for j in groups[k + 1]])
                    dense.append(pool_batch(torch, mdl, enc, device, arm))
        else:
            for g in groups:
                dense.append(pool_batch(torch, mdl, tokenize(tok, [texts[j] for j in g]), device, arm))
    nnz = sum(int(np.count_nonzero(v)) for v in dense)
    secs = time.perf_counter() - t0
    csr = sp.vstack([sp.csr_matrix(v) for v in dense]).tocsr()
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    return csr[inv], secs, nnz


def load_models(device, arms):
    import copy
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    if not (C.DEST / "model.safetensors").is_file():
        stop("no model.safetensors of C0 here: run zspl1c.py fetch and convert first")
    tok = AutoTokenizer.from_pretrained(str(C.DEST), local_files_only=True)
    base = AutoModelForMaskedLM.from_pretrained(str(C.DEST), local_files_only=True, use_safetensors=True).eval().to(device)
    models = {"fp32": base}
    need = {ARMS[a].get("dtype") for a in arms}
    if "fp16" in need:
        models["fp16"] = copy.deepcopy(base).half()
    if "bf16" in need:
        models["bf16"] = copy.deepcopy(base).bfloat16()
    return torch, tok, models, transformers.__version__


# ── measures ─────────────────────────────────────────────────────────────────────────────────────────────────

def cos_rows(A, B):
    """Cosine of row i of A with row i of B (two all-zero rows count as 1)."""
    num = np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float64)
    na = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel().astype(np.float64))
    nb = np.sqrt(np.asarray(B.multiply(B).sum(axis=1)).ravel().astype(np.float64))
    den = na * nb
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.where((na == 0) & (nb == 0), 1.0, 0.0))


def compare(A, B):
    """Arm vectors A against l0's B (same rows)."""
    c = cos_rows(A, B)
    same = np.asarray((A != B).sum(axis=1)).ravel() == 0
    return {"cos_min": float(c.min()), "cos_p1": float(np.percentile(c, 1)), "cos_median": float(np.median(c)), "bit_equal_share": float(same.mean()),
            "nnz_ratio": float(A.nnz / max(B.nnz, 1)), "rows": int(A.shape[0])}


def e2_pass(m):
    return all(m[s]["cos_p1"] >= COS_P1_MIN and abs(m[s]["nnz_ratio"] - 1.0) <= NNZ_TOL for s in ("docs", "questions"))


def pick_of(table):
    """The arm among l1..l5 with the largest paired ratio among those with E1 and E2; (None, why) if there is none."""
    ok = {a: t["ratio"] for a, t in table.items() if a in PICKABLE and t["ratio"] is not None and t["ratio"] >= E1_MIN and t["e2"]}
    if not ok:
        return None
    return max(ok, key=lambda a: (ok[a], a))


def call_lever(lo):
    return "BETTER" if lo > 0 else ("LEVEL" if lo >= LEVEL_LEV else "WORSE")


# ── bench ────────────────────────────────────────────────────────────────────────────────────────────────────

def cmd_bench(a):
    import scipy.sparse as sp
    import zworld
    t0 = time.time()
    device = a.device or "cuda"
    arms = list(ARMS)
    torch, tok, models, tfv = load_models(device, arms)
    w = zworld.load(a.ds)
    rows, _ = S.eval_rows(a.ds, w)
    qt = S.question_texts(w, rows[:a.questions])
    texts = []
    for _, t in w.iter_nodes():
        texts.append(t)
        if len(texts) >= a.nodes:
            break
    nb = len(texts) // a.block
    gpu = None
    if device == "cuda":
        fm, tm = torch.cuda.mem_get_info()
        gpu = {"name": torch.cuda.get_device_name(0), "free_mb_at_start": int(fm / 2**20), "total_mb": int(tm / 2**20)}
    log(f"bench {a.ds}: {len(texts)} nodes in {nb} blocks of {a.block}, {len(qt)} questions, arms {arms}, gpu {gpu}")
    for arm in arms:                                           # one untimed warm-up each
        run_encode(torch, tok, models, device, texts[:C.BATCH], arm)
    rates = {x: [] for x in arms}
    vec = {x: [] for x in arms}
    nnzs = {x: 0 for x in arms}
    peak = {x: 0 for x in arms}
    for b in range(nb):
        blk = texts[b * a.block:(b + 1) * a.block]
        k = b % len(arms)
        for arm in arms[k:] + arms[:k]:
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            csr, secs, nnz = run_encode(torch, tok, models, device, blk, arm)
            if device == "cuda":
                peak[arm] = max(peak[arm], int(torch.cuda.max_memory_allocated() / 2**20))
            rates[arm].append(len(blk) / secs)
            vec[arm].append(csr)
            nnzs[arm] += nnz
        log(f"  block {b + 1}/{nb}: " + "  ".join(f"{x} {rates[x][-1]:.0f}" for x in arms))
    qv, qs = {}, {}
    for arm in arms:
        qv[arm], qs[arm], _ = run_encode(torch, tok, models, device, qt, arm)
    D0, Q0 = sp.vstack(vec["l0"]).tocsr(), qv["l0"]
    res = {"stage": "SPL1 A11 bench", "declared_in": DOC, "dataset": a.ds, "nodes": len(texts), "block": a.block, "blocks": nb, "questions": len(qt), "gpu": gpu,
           "versions": {"torch": torch.__version__, "transformers": tfv}, "arms": {}}
    base = np.asarray(rates["l0"])
    for arm in arms:
        r = np.asarray(rates[arm])
        m = {"docs": compare(sp.vstack(vec[arm]).tocsr(), D0), "questions": compare(qv[arm], Q0)}
        res["arms"][arm] = {"spec": ARMS[arm], "rate_median": round(float(np.median(r)), 2), "rates": [round(float(x), 1) for x in r],
                            "ratio": round(float(np.median(r / base)), 4), "peak_mb": peak[arm], "doc_nnz_mean": round(nnzs[arm] / len(texts), 1),
                            "question_seconds": round(qs[arm], 3), "measures": m, "e2": bool(e2_pass(m)) if arm in PICKABLE else None}
    res["seconds"] = round(time.time() - t0, 1)
    S.wjson(LEV / f"bench_{a.ds}.json", res)
    for arm in arms:
        x = res["arms"][arm]
        log(f"  {arm}: rate {x['rate_median']} ratio {x['ratio']} E2 {x['e2']} cos_p1 docs {x['measures']['docs']['cos_p1']:.5f} questions {x['measures']['questions']['cos_p1']:.5f} "
            f"nnz {x['measures']['docs']['nnz_ratio']:.4f} bit-equal {x['measures']['docs']['bit_equal_share']:.4f} peak {x['peak_mb']} MB")


def cmd_call(a):
    p = LEV / "bench_squad.json"
    if not p.is_file():
        stop("no bench result: run bench first")
    b = json.loads(p.read_text(encoding="utf-8"))
    table = {arm: {"ratio": x["ratio"], "e2": bool(x["e2"]), "e1": x["ratio"] >= E1_MIN, "rate_median": x["rate_median"]} for arm, x in b["arms"].items() if arm in PICKABLE}
    pick = pick_of(table)
    out = {"stage": "SPL1 A11 call", "declared_in": DOC, "pick": pick, "table": table, "bench": "bench_squad.json", "bench_sha256": S.sha256(p),
           "decision": "PICK" if pick else "LEVERS_NOT_ADOPTED", "rule": f"E1 ratio >= {E1_MIN}, E2 cosine p1 >= {COS_P1_MIN} and non-zeros within {NNZ_TOL}", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    S.wjson(LEV / "pick.json", out)
    for arm, t in table.items():
        log(f"  {arm}: ratio {t['ratio']} E1 {t['e1']} E2 {t['e2']}")
    log(f"pick: {pick} ({out['decision']})")


# ── full encode of one arm ───────────────────────────────────────────────────────────────────────────────────

def cmd_encode(a):
    import scipy.sparse as sp
    import zworld
    t0 = time.time()
    arm_name = a.arm
    if a.if_pick:                                                # the full runs of A11 exist only if the bench picked an arm
        pk0 = json.loads((LEV / "pick.json").read_text(encoding="utf-8")) if (LEV / "pick.json").is_file() else None
        if pk0 is None or pk0["pick"] is None:
            log("no lever was picked (or no pick.json): this run is not needed")
            return
    if arm_name == "pick":
        pk = json.loads((LEV / "pick.json").read_text(encoding="utf-8")) if (LEV / "pick.json").is_file() else None
        if pk is None:
            stop("no pick.json: run call first")
        if pk["pick"] is None:
            log("no lever was picked: nothing to encode")
            return
        arm_name = pk["pick"]
    device = a.device or "cuda"
    torch, tok, models, tfv = load_models(device, [arm_name])
    w = zworld.load(a.ds)
    rows, _ = S.eval_rows(a.ds, w)
    qt = S.question_texts(w, rows)
    texts = [t for _, t in w.iter_nodes()]
    if a.smoke:
        texts = texts[:a.smoke]
    n, k = len(texts), min(S.TOPK, len(texts))
    d = LEV / arm_name / (a.ds + (f"_smoke_{device}" if a.smoke else ""))
    d.mkdir(parents=True, exist_ok=True)
    gpu = None
    if device == "cuda":
        fm, tm = torch.cuda.mem_get_info()
        gpu = {"name": torch.cuda.get_device_name(0), "free_mb_at_start": int(fm / 2**20), "total_mb": int(tm / 2**20)}
    Qs, qn = [], 0
    t1 = time.time()
    for i in range(0, len(qt), 1024):
        q, _, nz = run_encode(torch, tok, models, device, qt[i:i + 1024], arm_name)
        Qs.append(q)
        qn += nz
    Qm = sp.vstack(Qs).tocsr()
    log(f"{arm_name} {a.ds}: {len(qt)} questions in {time.time() - t1:.1f} s, query nnz mean {qn / max(len(qt), 1):.1f}, {n:,} nodes")
    st = d / "state.npz"
    top_s, top_i, done = np.full((len(qt), k), -np.inf, np.float32), np.full((len(qt), k), -1, np.int64), 0
    if st.is_file():
        with np.load(st, allow_pickle=False) as z:
            if int(z["n"]) == n and int(z["k"]) == k and int(z["nq"]) == len(qt) and str(z["arm"]) == arm_name:
                top_s, top_i, done = np.array(z["top_s"]), np.array(z["top_i"]), int(z["done"])
                log(f"resume at {done:,}")
    bl = d / "blocks.jsonl"
    if done == 0 and bl.is_file():
        bl.unlink()
    nnz_all, n0, t2 = 0, done, time.time()
    while done < n:
        end = min(n, done + S.BLOCK)
        Db, secs, nz = run_encode(torch, tok, models, device, texts[done:end], arm_name)
        nnz_all += nz
        top_s, top_i = S.merge_top(top_s, top_i, (Qm @ Db.T).toarray(), done, k)
        with open(bl, "a", encoding="utf-8") as f:
            f.write(json.dumps({"end": end, "nodes": end - done, "seconds": round(secs, 3), "t": round(time.time(), 1)}) + "\n")
        done = end
        tmp = st.with_name("state.tmp.npz")
        np.savez(tmp, top_s=top_s, top_i=top_i, done=done, n=n, k=k, nq=len(qt), arm=arm_name)
        S.replace_retry(tmp, st)
        log(f"  {done:,}/{n:,} nodes, {(done - n0) / max(time.time() - t2, 1e-9):.0f} nodes/s so far")
    el = time.time() - t2
    ids = np.where(np.isfinite(top_s), top_i, -1).astype(np.int32)
    np.savez(d / "top1000.npz", ids=ids, scores=np.where(np.isfinite(top_s), top_s, 0).astype(np.float32), rows=rows)
    brs, brm = S.block_rates([json.loads(x) for x in bl.read_text(encoding="utf-8").splitlines() if x.strip()])
    rec = {"stage": "SPL1 A11 encode", "declared_in": DOC, "arm": arm_name, "spec": ARMS[arm_name], "dataset": a.ds, "smoke": a.smoke, "nodes": n, "questions": len(qt), "k": k,
           "device": device, "nodes_per_s": round((n - n0) / max(el, 1e-9), 2), "block_rate_median": None if brm is None else round(brm, 2), "block_rates": [round(x, 1) for x in brs],
           "encode_seconds": round(el, 1), "seconds": round(time.time() - t0, 1), "doc_nnz_mean": round(nnz_all / max(n - n0, 1), 1),
           "query_nnz_mean": round(qn / max(len(qt), 1), 1), "gpu": gpu, "top1000_sha256": S.sha256(d / "top1000.npz"),
           "versions": {"torch": torch.__version__, "transformers": tfv}}
    S.wjson(d / "encode.json", rec)
    log(f"done {arm_name} {a.ds}: {rec['nodes_per_s']} nodes/s wall, block median {rec['block_rate_median']}, nnz {rec['doc_nnz_mean']}, {rec['seconds']} s")


# ── read ─────────────────────────────────────────────────────────────────────────────────────────────────────

def agreement(leg, rec, n):
    """Share of questions whose first min(1000, n) nodes are the same set in both lists, and whose first 20 are the same sequence."""
    kk = min(1000, n, leg.shape[1], rec.shape[1])
    same_set = float(np.mean(np.all(np.sort(leg[:, :kk], axis=1) == np.sort(rec[:, :kk], axis=1), axis=1)))
    same20 = float(np.mean(np.all(leg[:, :20] == rec[:, :20], axis=1)))
    return {"same_top_set_share": round(same_set, 5), "same_top20_sequence_share": round(same20, 5), "k": int(kk)}


def cmd_read(a):
    import zenc as E
    t0 = time.time()
    pk = json.loads((LEV / "pick.json").read_text(encoding="utf-8")) if (LEV / "pick.json").is_file() else None
    pick = pk["pick"] if pk else None
    arms = [x for x in ("l0", pick, "l6") if x]
    res = {"stage": "SPL1 A11 read", "declared_in": DOC, "pick": pick, "datasets": {}}
    for ds in DATASETS:
        if not any((LEV / x / ds / "top1000.npz").is_file() for x in arms):
            continue
        c = S.context(E, ds)
        n = c["w"].n
        base, _ = S.reads_of(E, c["dense"], c["c0"], c["flat"], c["off"], c["gt"], n)
        per, lists = {}, {}
        for x in arms:
            p = LEV / x / ds / "top1000.npz"
            if not p.is_file():
                continue
            z = np.load(p, allow_pickle=False)
            if not np.array_equal(z["rows"], c["rows"]):
                stop(f"{x} {ds}: the list's rows are not the screen's rows")
            leg = z["ids"].astype(np.int64)
            if leg.shape[1] < c["dense"].shape[1]:
                leg = np.pad(leg, ((0, 0), (0, c["dense"].shape[1] - leg.shape[1])), constant_values=-1)
            lists[x] = leg
            r, _ = S.reads_of(E, c["dense"], leg, c["flat"], c["off"], c["gt"], n)
            enc = json.loads((LEV / x / ds / "encode.json").read_text(encoding="utf-8")) if (LEV / x / ds / "encode.json").is_file() else {}
            e = {"level": {k_: round(float(v.mean() * 100), 3) for k_, v in r.items()}, "vs_record": {}, "vs_l0": {}, "agreement_with_record": agreement(leg, c["c0"], n),
                 "block_rate_median": enc.get("block_rate_median"), "nodes_per_s_wall": enc.get("nodes_per_s"), "doc_nnz_mean": enc.get("doc_nnz_mean")}
            for k_ in READS:
                if k_ in r and k_ in base:
                    dm, lo, hi = S.boot(r[k_] - base[k_])
                    e["vs_record"][k_] = {"diff": round(dm, 3), "lo": round(lo, 3), "hi": round(hi, 3), "call": call_lever(lo)}
            per[x] = (e, r)
        for x in per:
            if x != "l0" and "l0" in per:
                per[x][0]["agreement_with_l0"] = agreement(lists[x], lists["l0"], n)
                for k_ in READS:
                    if k_ in per[x][1] and k_ in per["l0"][1]:
                        dm, lo, hi = S.boot(per[x][1][k_] - per["l0"][1][k_])
                        per[x][0]["vs_l0"][k_] = {"diff": round(dm, 3), "lo": round(lo, 3), "hi": round(hi, 3), "call": call_lever(lo)}
        res["datasets"][ds] = {"questions": int(len(c["rows"])), "nodes": n, "record": {k_: round(float(v.mean() * 100), 3) for k_, v in base.items()}, "arms": {x: per[x][0] for x in per}}
        log(f"{ds}: read {list(per)}")
    d2 = None
    if pick and all(ds in res["datasets"] and pick in res["datasets"][ds]["arms"] for ds in DATASETS):
        calls = [v["call"] for ds in DATASETS for v in res["datasets"][ds]["arms"][pick]["vs_record"].values()]
        calls += [v["call"] for ds in ("squad", "multihoprag") for v in res["datasets"][ds]["arms"][pick].get("vs_l0", {}).values()]
        ok = all(x in ("BETTER", "LEVEL") for x in calls)
        d2 = {"decision": f"ADOPT_LEVER({pick})" if ok else "LEVERS_NOT_ADOPTED", "reads": len(calls), "worse": int(sum(x == "WORSE" for x in calls)),
              "agreement_below_99": [ds for ds in DATASETS if res["datasets"][ds]["arms"][pick]["agreement_with_record"]["same_top_set_share"] < 0.99]}
    elif not pick:
        d2 = {"decision": "LEVERS_NOT_ADOPTED", "reason": "no arm passed E1 and E2"}
    res["D2"] = d2
    res["seconds"] = round(time.time() - t0, 1)
    S.wjson(LEV / "read.json", res)
    if d2 is not None:                                           # the file the host's gate row waits for (pushed by hand after this read)
        S.wjson(LEV / "d2.json", {"stage": "SPL1 A11 D2", "declared_in": DOC, "pick": pick, "D2": d2, "read_sha256": S.sha256(LEV / "read.json")})
    print(json.dumps({"pick": pick, "D2": d2}, indent=1))


def cmd_gate(a):
    """A row that ends when a file exists (the decision D2, pushed to the host after the reads), and fails after --max-s seconds: the STaRK legs that follow
    it wait for D2 and are dropped by the feeder if the file never comes."""
    p = Path(a.wait)
    t0 = time.time()
    while not p.is_file():
        if time.time() - t0 > a.max_s:
            stop(f"{p} did not appear within {a.max_s} s")
        time.sleep(30)
    log(f"{p} is here after {time.time() - t0:.0f} s")


# ── self-test ────────────────────────────────────────────────────────────────────────────────────────────────

def selftest():
    import torch
    bad = 0

    def chk(name, ok):
        nonlocal bad
        log(f"  {'ok  ' if ok else 'FAIL'} {name}")
        bad += 0 if ok else 1

    class Out:
        def __init__(self, lg):
            self.logits = lg

    class Tok:                                                   # pads to the longest; id 0 is the pad; callable with or without tensors
        def __call__(self, texts, return_tensors=None, truncation=True, max_length=256, padding=False):
            ids = [[(ord(ch) % 7) + 1 for ch in t][:max_length] for t in texts]
            if return_tensors is None:
                return {"input_ids": ids}
            L = max(len(x) for x in ids)
            return {"input_ids": torch.tensor([x + [0] * (L - len(x)) for x in ids]),
                    "attention_mask": torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids])}

    class Mdl:
        def __init__(self, pad_boost=0.0):
            g = torch.Generator().manual_seed(0)
            self.W = torch.randn(8, 12, generator=g)
            self.W[0] += pad_boost                              # what a pad position adds to the logits

        def __call__(self, input_ids, attention_mask):
            return Out(self.W[input_ids] - 0.5)

    texts = ["abc", "abcdefghij", "xyz", "q", "hello world", "zz", "mm mm", "the end", "a", "bb", "ccc", "dddd"]
    tok = Tok()
    models = {"fp32": Mdl()}
    ARMS["t_mask"] = {"dtype": "fp32", "chunk": 5, "mask": True}
    ARMS["t_masksort"] = {"dtype": "fp32", "chunk": 3, "mask": True, "sort": True}
    ARMS["t_l1"] = {"dtype": "fp32", "chunk": 4}
    v0, _, _ = run_encode(torch, tok, models, "cpu", texts, "l0")
    v1, _, _ = run_encode(torch, tok, models, "cpu", texts, "t_l1")
    chk("max-then-activate gives the record's vectors bit for bit (l1 against l0)", np.array_equal(v0.toarray(), v1.toarray()))
    models["fp32"] = Mdl(pad_boost=5.0)
    vm, _, _ = run_encode(torch, tok, models, "cpu", texts, "t_mask")
    vr, _, _ = run_encode(torch, tok, models, "cpu", texts, "l0")
    import scipy.sparse as sp
    single = sp.vstack([run_encode(torch, tok, models, "cpu", [t], "l0")[0] for t in texts]).tocsr()     # a batch of one has no padding
    chk("a pad position that fires raises the record's vector (it is batch dependent) and the masked vector is the unpadded one",
        (not np.array_equal(vr.toarray(), single.toarray())) and bool((vr.toarray() >= vm.toarray()).all()) and np.array_equal(vm.toarray(), single.toarray()))
    vs, _, _ = run_encode(torch, tok, models, "cpu", texts, "t_masksort")
    chk("masked pooling with the nodes sorted by length returns the rows in the order of the texts, equal to each text alone", np.array_equal(vs.toarray(), single.toarray()))
    chk("the sorted arm's matrix has one row per text", vs.shape == (len(texts), 12))
    A, B = v0, v1
    c = cos_rows(A, B)
    chk("cosine of equal vectors is 1; the bit-equal share is 1", np.allclose(c, 1.0) and compare(A, B)["bit_equal_share"] == 1.0)
    chk("an arm with half the non-zeros fails E2 and the nnz tolerance", not e2_pass({"docs": {"cos_p1": 1.0, "nnz_ratio": 0.5}, "questions": {"cos_p1": 1.0, "nnz_ratio": 1.0}}))
    chk("an arm with cosine 0.9995 and 0.3 percent more non-zeros passes E2", e2_pass({"docs": {"cos_p1": 0.9995, "nnz_ratio": 1.003}, "questions": {"cos_p1": 0.9991, "nnz_ratio": 0.998}}))
    tab = {"l1": {"ratio": 1.2, "e2": True}, "l3": {"ratio": 2.4, "e2": True}, "l5": {"ratio": 3.1, "e2": False}, "l2": {"ratio": 1.6, "e2": True}}
    chk("the pick is the fastest arm with E1 and E2 (l3: l5 fails E2, l1 fails E1)", pick_of(tab) == "l3")
    chk("no arm with E1 and E2: no pick", pick_of({"l1": {"ratio": 1.2, "e2": True}, "l3": {"ratio": 3.0, "e2": False}}) is None)
    chk("calls: low end above 0 is BETTER, -0.20 is LEVEL, below is WORSE", (call_lever(0.01), call_lever(-0.20), call_lever(-0.21)) == ("BETTER", "LEVEL", "WORSE"))
    rec_l = np.array([[3, 1, 2, 0], [0, 1, 2, 3]])
    chk("agreement: a swapped top 20 keeps the set and breaks the sequence", agreement(np.array([[1, 3, 2, 0], [0, 1, 2, 3]]), rec_l, 4) == {"same_top_set_share": 1.0, "same_top20_sequence_share": 0.5, "k": 4})
    for k in ("t_mask", "t_masksort", "t_l1"):
        ARMS.pop(k)
    if bad:
        raise SystemExit(f"zspl1l: selftest FAILED ({bad})")
    log("selftest PASS (11 checks)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(fn=lambda a: selftest())
    b = sub.add_parser("bench")
    b.add_argument("--ds", default="squad", choices=S.SCREEN)
    b.add_argument("--nodes", type=int, default=4096)
    b.add_argument("--block", type=int, default=512)
    b.add_argument("--questions", type=int, default=256)
    b.add_argument("--device", default=None)
    b.set_defaults(fn=cmd_bench)
    sub.add_parser("call").set_defaults(fn=cmd_call)
    e = sub.add_parser("encode")
    e.add_argument("--arm", required=True)
    e.add_argument("--ds", required=True, choices=S.SCREEN)
    e.add_argument("--device", default=None)
    e.add_argument("--smoke", type=int, default=0)
    e.add_argument("--if-pick", dest="if_pick", action="store_true")
    e.set_defaults(fn=cmd_encode)
    sub.add_parser("read").set_defaults(fn=cmd_read)
    g = sub.add_parser("gate")
    g.add_argument("--wait", required=True)
    g.add_argument("--max-s", type=int, default=7200)
    g.set_defaults(fn=cmd_gate)
    a = ap.parse_args()
    if getattr(a, "arm", None) not in (None, "pick") and a.arm not in ARMS:
        stop(f"unknown arm {a.arm}")
    a.fn(a)


if __name__ == "__main__":
    main()
