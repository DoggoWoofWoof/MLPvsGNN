"""ENC E0N (docs/ENC_ENCODER_COMPARISON.md section 17): three retrieval-trained encoders at mpnet's size, read at the
first stage exactly as E0 was (zenc.py's encode and first, unchanged).

Subcommands:
  selftest [--real]     the call rule on toy reads; --real also checks the reference reads and bars on the host
  models                download the three at their pins (huggingface_hub), record every file's sha256, load each
                        offline, stop on any prompt, a short position table or a wrong width
  encode --enc ARM      GP (float32 against float16 on 256 + 256 rows per dataset), then zenc's encode on SQuAD and
                        MuSiQue (queries and documents)
  first --enc ARM --ds  zenc's first-stage read, written beside E0's (outputs/enc/e0/<arm>/<ds>.npz)
  report                17.5's calls, with the beside measures; outputs/enc/e0n/report.{json,md}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import zenc as E  # noqa: E402  (first: sets the BLAS threads before numpy loads)
import zenc2m as M  # noqa: E402
import zenc2b as B  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

STAGE = "ENC E0N"
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 17"
OUT = E.OUT / "e0n"
DS = ("squad", "musique")
LEN = 8192
GP_ROWS, GP_MIN_COS = 256, 0.99
NEW3 = {
    "gtemodern": {"repo": "Alibaba-NLP/gte-modernbert-base", "dim": 768, "budget": 65536,
                  "rev": "e7f32e3c00f91d699e8c43b53106206bcc72bb22"},
    "granite": {"repo": "ibm-granite/granite-embedding-english-r2", "dim": 768, "budget": 65536,
                "rev": "47ea694b257b703fee9253d75c2b1f2985180498"},
    "granitesm": {"repo": "ibm-granite/granite-embedding-small-english-r2", "dim": 384, "budget": 131072,
                  "rev": "2ab6fa8ea2d674564defd37171ae19079b864b33"},
}
for _k, _v in NEW3.items():
    E.ENCODERS[_k] = {"repo": _v["repo"], "dim": _v["dim"], "query": {}, "max_seq_length": LEN, "budget": _v["budget"]}
REFS = ("frozen", "mpnet", "qwen3e06")


def log(msg):
    print(f"[zenc3 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


_pinned = E.pinned


def pinned(enc):
    """The new arms' pins come from outputs/enc/e0n/models.json and must equal NEW3's; the others are zenc's."""
    if enc not in NEW3:
        return _pinned(enc)
    p = OUT / "models.json"
    if not p.is_file():
        E.stop("outputs/enc/e0n/models.json missing: run `zenc3.py models` first")
    rev = json.loads(p.read_text(encoding="utf-8"))[enc]["revision"]
    if rev != NEW3[enc]["rev"]:
        E.stop(f"{enc}: models.json pins {rev}, section 17.2 pins {NEW3[enc]['rev']}")
    return rev


E.pinned = pinned            # zenc's encode looks pinned up at call time


def code_sha():
    return {f"outputs/mp_unified/{Path(m.__file__).name}": E.sha256(Path(m.__file__)) for m in (E, M, B)} | {
        "outputs/mp_unified/zenc3.py": E.sha256(Path(__file__))}


# ── models ───────────────────────────────────────────────────────────────────

def cmd_models(a):
    from huggingface_hub import HfApi, snapshot_download
    api, rec = HfApi(), {}
    OUT.mkdir(parents=True, exist_ok=True)
    for enc, s in NEW3.items():
        names = [x.rfilename for x in (api.model_info(s["repo"], revision=s["rev"]).siblings or [])]
        allow = E.ALLOW + ([] if "model.safetensors" in names else ["pytorch_model.bin"])
        t0 = time.time()
        path = Path(snapshot_download(s["repo"], revision=s["rev"], allow_patterns=allow, ignore_patterns=E.IGNORE))
        files = {str(f.relative_to(path)).replace("\\", "/"): {"bytes": f.stat().st_size, "sha256": E.sha256(f)}
                 for f in sorted(path.rglob("*")) if f.is_file()}
        rec[enc] = {"repo": s["repo"], "revision": s["rev"], "seconds": round(time.time() - t0, 1),
                    "bytes": sum(v["bytes"] for v in files.values()), "files": files}
        log(f"{enc} {s['repo']} {s['rev'][:12]} {rec[enc]['bytes']} bytes")
    for enc in NEW3:
        m = E.load_model(enc, "cpu", rec[enc]["revision"])
        cfg = m[0].auto_model.config
        prompts = dict(m.prompts or {})
        rec[enc].update({
            "params": int(sum(t.numel() for t in m.parameters())), "max_seq_length": int(m.max_seq_length),
            "config_default": {"tokenizer_model_max_length": int(min(m.tokenizer.model_max_length, 10 ** 9)),
                               "max_position_embeddings": int(getattr(cfg, "max_position_embeddings", -1))},
            "model_type": getattr(cfg, "model_type", None), "modules": [type(x).__name__ for x in m],
            "pooling": m[1].get_config_dict() if len(m) > 1 and hasattr(m[1], "get_config_dict") else None,
            "dim": int(m.get_sentence_embedding_dimension()), "prompts": prompts,
            "default_prompt_name": m.default_prompt_name})
        if m.default_prompt_name or any(v for v in prompts.values()):
            E.stop(f"{enc}: a prompt in its config ({prompts}, default {m.default_prompt_name}); 17.2 says none")
        if rec[enc]["config_default"]["max_position_embeddings"] < LEN:
            E.stop(f"{enc}: {rec[enc]['config_default']['max_position_embeddings']} positions, under {LEN}")
        if rec[enc]["dim"] != NEW3[enc]["dim"]:
            E.stop(f"{enc}: width {rec[enc]['dim']}, 17.2 says {NEW3[enc]['dim']}")
        log(f"{enc} loads offline: {rec[enc]['params']} params, {rec[enc]['modules']}, length {m.max_seq_length}")
        del m
    E.wjson(OUT / "models.json", rec)
    log(f"wrote {OUT / 'models.json'}")
    return 0


# ── encode ───────────────────────────────────────────────────────────────────

def gp_check(enc, rev):
    """17.3 GP: float32 against float16 on 256 document and 256 query rows per dataset (rng seed 0)."""
    import torch
    from sentence_transformers import SentenceTransformer
    root = E.served_root()
    m16 = E.load_model(enc, "cuda", rev)
    m32 = SentenceTransformer(NEW3[enc]["repo"], device="cuda", revision=rev, local_files_only=True)
    m32.max_seq_length = LEN
    rng = np.random.default_rng(0)
    rec = {}
    for ds in DS:
        for kind in ("docs", "queries"):
            n = E.manifest(root, ds, kind)["n_rows"]
            rows = np.sort(rng.choice(n, GP_ROWS, replace=False))
            got = E.doc_texts(root, ds, rows) if kind == "docs" else E.query_texts(root, ds, rows)
            texts = [got[int(r)] for r in rows]
            a16, _ = E.encode_texts(m16, enc, texts, kind)
            a32, _ = E.encode_texts(m32, enc, texts, kind)
            x, y = a16.astype(np.float32), a32.astype(np.float32)
            cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
            rec[f"{ds} {kind}"] = {"rows": int(rows.size), "min_cos": round(float(cos.min()), 6),
                                   "mean_cos": round(float(cos.mean()), 6)}
            log(f"GP {enc} {ds} {kind}: min cos {cos.min():.5f}")
    del m16, m32
    torch.cuda.empty_cache()
    ok = all(v["min_cos"] >= GP_MIN_COS for v in rec.values())
    out = {"stage": STAGE, "declared_in": DECLARED, "enc": enc, "revision": rev, "min_cos_bar": GP_MIN_COS,
           "cells": rec, "pass": ok, "code_sha256": code_sha()}
    E.wjson(OUT / f"gp-{enc}.json", out)
    if not ok:
        E.stop(f"GP failed for {enc}: {rec}")


def cmd_encode(a):
    import torch
    if not torch.cuda.is_available():
        E.stop("no GPU")
    rev = pinned(a.enc)
    t0 = time.time()
    gp_check(a.enc, rev)
    log(f"GP passed for {a.enc} in {time.time() - t0:.0f} s")
    for ds in DS:
        E.cmd_encode(argparse.Namespace(enc=a.enc, ds=ds, kinds="queries,docs", cpu=False))
    log(f"{a.enc} encoded in {time.time() - t0:.0f} s")
    return 0


def cmd_first(a):
    pinned(a.enc)
    E.cmd_first(argparse.Namespace(enc=a.enc, ds=a.ds))
    return 0


# ── report ───────────────────────────────────────────────────────────────────

def load_reads(arms):
    """{(arm, ds): npz} for the new arms, E0's references and E2B's mpnet-V; GR: same ids in the same order."""
    z = {}
    for ds in DS:
        for arm in tuple(arms) + REFS:
            z[(arm, ds)] = np.load(E.OUT / "e0" / arm / f"{ds}.npz", allow_pickle=False)
        z[("mpnet-V", ds)] = np.load(E.OUT / "e2b" / "reads" / f"mpnet-V__{ds}.npz", allow_pickle=False)
        ids = z[("frozen", ds)]["ids"]
        for (arm, d), r in z.items():
            if d == ds and not (r["ids"].shape == ids.shape and (r["ids"] == ids).all()):
                E.stop(f"GR: {arm} {ds} holds other question ids than E0's frozen read")
    return z


def call_of(vs_mpnet, vs_v):
    """17.5: NO unless above raw mpnet on both carves; STRONG when also above E2B's mpnet V on both; else WORKABLE."""
    if not all(vs_mpnet[ds]["ci"][0] > 0 for ds in DS):
        return "NO"
    if all(vs_v[ds]["ci"][0] > 0 for ds in DS):
        return "STRONG"
    return "WORKABLE"


def order_share(za, zf):
    """1 - (frozen's golds minus the arm's at 1,000) / (the same at 5); None when the gap at 5 is not positive."""
    gt = zf["gold_total"]
    g = {k: int(np.minimum(zf[f"h_{k}"], gt).sum() - np.minimum(za[f"h_{k}"], gt).sum()) for k in ("d5", "d1000")}
    return {"gold_gap_5": g["d5"], "gold_gap_1000": g["d1000"],
            "order_share": None if g["d5"] <= 0 else round(1 - g["d1000"] / g["d5"], 3)}


def by_hops(z, gt):
    return {int(h): round(100 * float(np.mean(np.minimum(z["h_d5"][gt == h], h) / h)), 2) for h in sorted(set(gt.tolist()))}


def build_report(arms, z, models, bars, cost):
    rec = {"stage": STAGE, "declared_in": DECLARED, "arms": {}, "refs": {}, "bars": bars}
    for ref in REFS + ("mpnet-V",):
        rec["refs"][ref] = {ds: {k: round(100 * float(np.mean(M.per_q(z[(ref, ds)], k))), 2)
                                 for k in ("d5", "d20", "d1000", "rrf5")} for ds in DS}
    for arm in arms:
        vs = {ref: {ds: M.gap(M.per_q(z[(arm, ds)], "d5"), M.per_q(z[(ref, ds)], "d5")) for ds in DS}
              for ref in ("mpnet", "mpnet-V", "qwen3e06", "frozen")}
        r = {"call": call_of(vs["mpnet"], vs["mpnet-V"]), "vs": vs,
             "recall": {ds: {k: round(100 * float(np.mean(M.per_q(z[(arm, ds)], k))), 2)
                             for k in ("d5", "d20", "d1000", "rrf5")} for ds in DS},
             "order": {ds: order_share(z[(arm, ds)], z[("frozen", ds)]) for ds in DS},
             "hops_musique": by_hops(z[(arm, "musique")], z[("frozen", "musique")]["gold_total"]),
             "mean_gap_to_frozen": round(float(np.mean([vs["frozen"][ds]["pts"] for ds in DS])), 2),
             "clears_bars": {enc: {m: all(vs["frozen"][ds]["pts"] >= b[m][ds] for ds in DS) for m in b}
                             for enc, b in bars.items()},
             "cost": cost.get(arm), "params": models.get(arm, {}).get("params"), "dim": NEW3[arm]["dim"]}
        rec["arms"][arm] = r
    rec["refs_hops_musique"] = {ref: by_hops(z[(ref, "musique")], z[("frozen", "musique")]["gold_total"])
                                for ref in REFS + ("mpnet-V",)}
    rec["order_of_next"] = sorted([a for a in arms if rec["arms"][a]["call"] != "NO"],
                                  key=lambda a: -rec["arms"][a]["mean_gap_to_frozen"])
    return rec


def report_md(rec):
    L = [f"# {STAGE} report", "", f"Declared in {DECLARED}. Dense R@5 on E0's s1eval carves; points; paired 95% "
         "bootstrap over questions (2,000, seed 0).", ""]
    L += ["| arm | call | SQuAD R@5 | MuSiQue R@5 | vs mpnet (SQuAD / MuSiQue) | vs mpnet V | vs qwen3e06 | vs frozen |",
          "|---|---|---|---|---|---|---|---|"]
    f = lambda g: f"{g['pts']:+.2f} [{g['ci'][0]:+.2f}, {g['ci'][1]:+.2f}]"  # noqa: E731
    for arm, r in rec["arms"].items():
        L.append(f"| {arm} | **{r['call']}** | {r['recall']['squad']['d5']:.2f} | {r['recall']['musique']['d5']:.2f} | "
                 + " | ".join(f"{f(r['vs'][ref]['squad'])} / {f(r['vs'][ref]['musique'])}"
                              for ref in ("mpnet", "mpnet-V", "qwen3e06", "frozen")) + " |")
    L += ["", "References: " + "; ".join(f"{ref} {v['squad']['d5']:.2f} / {v['musique']['d5']:.2f}"
                                          for ref, v in rec["refs"].items()), ""]
    L += ["## Beside (no call)", "", "| arm | ds | R@20 | R@1000 | RRF@5 | gold gap at 5 / 1000 | order share |",
          "|---|---|---|---|---|---|---|"]
    for arm, r in rec["arms"].items():
        for ds in DS:
            o = r["order"][ds]
            L.append(f"| {arm} | {ds} | {r['recall'][ds]['d20']:.2f} | {r['recall'][ds]['d1000']:.2f} | "
                     f"{r['recall'][ds]['rrf5']:.2f} | {o['gold_gap_5']} / {o['gold_gap_1000']} | {o['order_share']} |")
    for ref in REFS + ("mpnet-V",):
        v = rec["refs"][ref]
        L.append(f"| {ref} (ref) | both | {v['squad']['d20']:.2f} / {v['musique']['d20']:.2f} | "
                 f"{v['squad']['d1000']:.2f} / {v['musique']['d1000']:.2f} | {v['squad']['rrf5']:.2f} / "
                 f"{v['musique']['rrf5']:.2f} | | |")
    L += ["", "MuSiQue R@5 by hop count: " + "; ".join(
        f"{a} {r['hops_musique']}" for a, r in rec["arms"].items()) + "; " + "; ".join(
        f"{a} {h}" for a, h in rec["refs_hops_musique"].items()), ""]
    L += ["Bars (pointers only): " + json.dumps(rec["bars"]), "",
          "Clears the bars (gap to frozen, both carves): " + json.dumps({a: r["clears_bars"] for a, r in rec["arms"].items()}),
          "", "Cost: " + json.dumps({a: {"params": r["params"], "dim": r["dim"], **(r["cost"] or {})}
                                     for a, r in rec["arms"].items()}), "",
          "Next, best first (17.6): " + (", ".join(rec["order_of_next"]) or "none"), ""]
    return "\n".join(L)


def arm_cost(arm):
    out = {"encode_seconds": 0.0, "cut": 0}
    for ds in DS:
        for kind in ("queries", "docs"):
            mf = json.loads((E.OUT / arm / ds / kind / "manifest.json").read_text(encoding="utf-8"))
            out["encode_seconds"] += mf["seconds"]
            out["cut"] += mf["cut"]
    out["encode_seconds"] = round(out["encode_seconds"], 1)
    return out


def ref_bars():
    e2 = json.loads(B.E2R.read_text(encoding="utf-8"))
    return {"mpnet": B.bars_of(e2, "mpnet"), "qwen3e06": B.bars_of(e2, "qwen3e06")}


def cmd_report(a):
    arms = tuple(NEW3)
    z = load_reads(arms)
    models = json.loads((OUT / "models.json").read_text(encoding="utf-8"))
    gp = {arm: json.loads((OUT / f"gp-{arm}.json").read_text(encoding="utf-8")) for arm in arms}
    if not all(g["pass"] for g in gp.values()):
        E.stop("a GP record did not pass")
    rec = build_report(arms, z, models, ref_bars(), {arm: arm_cost(arm) for arm in arms})
    rec["gp"] = {arm: g["cells"] for arm, g in gp.items()}
    rec["code_sha256"] = code_sha()
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    E.wjson(OUT / "report.json", rec)
    (OUT / "report.md").write_text(report_md(rec), encoding="utf-8")
    log("calls: " + json.dumps({arm: r["call"] for arm, r in rec["arms"].items()}))
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def toy_read(gt, h5, rng):
    n = gt.size
    h = {"d5": h5}
    h["d20"] = np.minimum(gt, h5 + rng.integers(0, 2, n))
    h["d1000"] = np.minimum(gt, h["d20"] + rng.integers(0, 2, n))
    h["rrf5"] = h5
    return {"ids": np.asarray([f"q{i}" for i in range(n)]), "gold_total": gt, **{f"h_{k}": v.astype(np.int16)
                                                                                for k, v in h.items()}}


def selftest_toy():
    rng = np.random.default_rng(0)
    n = 3000
    z = {}
    for ds in DS:
        gt = rng.integers(1, 4, n) if ds == "musique" else np.ones(n, np.int64)
        base = rng.random(n)
        lvl = {"frozen": 0.80, "qwen3e06": 0.70, "mpnet-V": 0.60, "mpnet": 0.50,
               "gtemodern": 0.68, "granite": 0.55, "granitesm": 0.50}
        for arm, p in lvl.items():
            z[(arm, ds)] = toy_read(gt, np.minimum(gt, (base < p).astype(np.int64) * gt), rng)
    bars = {"mpnet": M.DECLARED_BARS, "qwen3e06": M.DECLARED_BARS}
    rec = build_report(tuple(NEW3), z, {}, bars, {})
    calls = {a: r["call"] for a, r in rec["arms"].items()}
    assert calls == {"gtemodern": "STRONG", "granite": "WORKABLE", "granitesm": "NO"}, calls
    assert rec["order_of_next"] == ["gtemodern", "granite"], rec["order_of_next"]
    assert rec["arms"]["granitesm"]["vs"]["mpnet"]["squad"]["pts"] == 0.0
    o = rec["arms"]["granite"]["order"]["squad"]
    assert o["gold_gap_5"] > 0 and o["order_share"] is not None
    report_md(rec)
    for k in NEW3:                      # the registration zenc's code paths read
        assert E.ENCODERS[k]["max_seq_length"] == LEN and E.ENCODERS[k]["query"] == {}
    assert E.pinned is pinned
    print("zenc3 self-test: toy OK", flush=True)


def selftest_real():
    z = {}
    for ds in DS:
        for ref in REFS:
            z[ref, ds] = np.load(E.OUT / "e0" / ref / f"{ds}.npz", allow_pickle=False)
        z["mpnet-V", ds] = np.load(E.OUT / "e2b" / "reads" / f"mpnet-V__{ds}.npz", allow_pickle=False)
        ids = z["frozen", ds]["ids"]
        assert all((z[r, ds]["ids"] == ids).all() for r in REFS + ("mpnet-V",)), ds
    b = ref_bars()
    assert b["mpnet"] == M.DECLARED_BARS, b
    root = E.served_root()
    for ds in DS:
        for kind in ("queries", "docs"):
            assert E.manifest(root, ds, kind)["n_rows"] > 0
    print("zenc3 real: " + json.dumps({"bars": b, "refs_r5": {f"{r} {ds}": round(100 * float(np.mean(
        M.per_q(z[r, ds], "d5"))), 2) for r in REFS + ("mpnet-V",) for ds in DS}}), flush=True)


def cmd_selftest(a):
    selftest_toy()
    if a.real:
        selftest_real()
    print("zenc3 self-test: OK", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "models", "encode", "first", "report"])
    ap.add_argument("--enc", choices=sorted(NEW3))
    ap.add_argument("--ds", choices=DS)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.cmd in ("encode", "first") and not a.enc:
        E.stop(f"{a.cmd} needs --enc")
    if a.cmd == "first" and not a.ds:
        E.stop("first needs --ds")
    return {"selftest": cmd_selftest, "models": cmd_models, "encode": cmd_encode, "first": cmd_first,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
