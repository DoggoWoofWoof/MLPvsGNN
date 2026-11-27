"""SPL1 amendment A12 (docs/SPL1_SPARSE_LEG_SCREEN.md section 8.12): four further sparse models screened as section 5 screens C1 to C3, and the rule D3.

    python -E -P -B outputs/mp_unified/zspl1m.py selftest                    # laptop, no model
    python -I -B    outputs/mp_unified/zspl1m.py fetch --key c4|c5|c6|c7     # host: the pinned revision into outputs/spl1/models/cK (a new empty folder)
    python -I -B    outputs/mp_unified/zspl1m.py convert --key c5|c6         # host: the pickle read once with torch.load(weights_only=True), written as safetensors
    python -I -B    outputs/mp_unified/zspl1m.py check --key cK --device cpu # section 4's check on the fetched model
    python -I -B    outputs/mp_unified/zspl1m.py encode --key cK --ds D      # zspl1.py's encode: top-1000 lists, block rates, non-zeros
    python -E -P -B outputs/mp_unified/zspl1m.py read                        # laptop: H1, H2 and the leg's reads against the record's lists, D3 -> outputs/spl1/cand_read.json

Run from the workspace root. zspl1.py is imported and edited nowhere: its model table is extended in this process only.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import zspl1 as S  # noqa: E402

DOC = S.DOC + " section 8.12 (A12)"
MODELS_M = {
    "c4": {"repo": "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill", "rev": "babf71f3c48695e2e53a978208e8aba48335e3c0", "params_m": 67, "cap_mb": 300, "pickle": False},
    "c5": {"repo": "naver/splade-v3-distilbert", "rev": "2db06b86d65e316e2ca9907aa1aa8be6f8c4e739", "params_m": 67, "cap_mb": 300, "pickle": True},
    "c6": {"repo": "naver/splade-v3", "rev": "fdfeceb91d7b9de7985b38addd3ba9f53a59a355", "params_m": 110, "cap_mb": 480, "pickle": True},
    "c7": {"repo": "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte", "rev": "1646fef40807937e8e130c66d327a26421c408d5", "params_m": 137, "cap_mb": 600, "pickle": False},
}
for _k, _m in MODELS_M.items():                                  # zspl1.py's table, in this process only
    S.MODELS[_k] = {kk: vv for kk, vv in _m.items() if kk != "pickle"}
KEYS = tuple(MODELS_M)
SPEED_A, SPEED_B, GAIN_B, PRICE_C = 3.0, 1.0, 1.0, -1.5
log, stop = S.log, S.stop


def take_file(key, path):
    """The fetch rule of A12: zspl1.py's allow-list (json, txt, md, safetensors; no hidden file, no onnx or openvino folder) and, for the two pickle models, the one pickle."""
    return S.allowed_file(path) or (MODELS_M[key]["pickle"] and path == "pytorch_model.bin")


# ── fetch and convert ────────────────────────────────────────────────────────────────────────────────────────

def cmd_fetch(a):
    from huggingface_hub import HfApi, hf_hub_download
    m = MODELS_M[a.key]
    dest = S.OUT / "models" / a.key
    if (S.OUT / f"fetch_{a.key}.json").is_file():
        stop(f"{a.key} was fetched already (fetch_{a.key}.json): each model goes into its own folder once")
    dest.mkdir(parents=True, exist_ok=True)                      # a folder left by an interrupted fetch of this model is resumed: the hub client skips finished files
    files = [(x.path, int(x.size or 0)) for x in HfApi().list_repo_tree(m["repo"], revision=m["rev"], recursive=True, token=False) if hasattr(x, "size")]
    take = [(p, s) for p, s in files if take_file(a.key, p)]
    skip = [p for p, _ in files if not take_file(a.key, p)]
    total = sum(s for _, s in take)
    log(f"{m['repo']} @ {m['rev'][:8]}: {len(files)} files, {len(take)} taken, {total / 1e6:.1f} MB; skipped {skip}")
    for p, s in take:
        log(f"  {p}  {s / 1e6:.2f} MB")
    if total > m["cap_mb"] * 1e6:
        stop(f"the files are {total / 1e6:.0f} MB, over this model's cap of {m['cap_mb']} MB: nothing fetched")
    if not any(p.endswith(".safetensors") or p == "pytorch_model.bin" for p, _ in take):
        stop("no weights file in the list: nothing fetched")
    rec = {"stage": "SPL1 A12 fetch", "declared_in": DOC, "key": a.key, "repo": m["repo"], "revision": m["rev"], "files": {}, "skipped": skip}
    for p, s in take:
        got = hf_hub_download(repo_id=m["repo"], filename=p, revision=m["rev"], local_dir=str(dest), token=False)
        rec["files"][p] = {"bytes": os.path.getsize(got), "listed_bytes": s, "sha256": S.sha256(got)}
    rec["total_bytes"] = sum(v["bytes"] for v in rec["files"].values())
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    S.wjson(S.OUT / f"fetch_{a.key}.json", rec)
    log(f"fetched {len(take)} files, {rec['total_bytes'] / 1e6:.1f} MB into {dest}")


def cmd_convert(a):
    import torch
    from safetensors.torch import load_file, save_file
    if not MODELS_M[a.key]["pickle"]:
        stop(f"{a.key} has safetensors weights: nothing to convert")
    d = S.OUT / "models" / a.key
    pk, out = d / "pytorch_model.bin", d / "model.safetensors"
    if out.exists():
        stop(f"{out} exists")
    if not pk.is_file():
        stop(f"{pk} is not here: run fetch first")
    sd = torch.load(pk, map_location="cpu", weights_only=True)    # the restricted unpickler: tensors and plain containers only
    if not isinstance(sd, dict) or not sd or not all(isinstance(v, torch.Tensor) for v in sd.values()):
        stop("the pickle is not a plain dict of tensors: nothing written")
    sd = {k: v.detach().contiguous().clone() for k, v in sd.items()}
    save_file(sd, str(out), metadata={"format": "pt"})
    back = load_file(str(out))
    if set(back) != set(sd) or not all(torch.equal(back[k], sd[k]) for k in sd):
        stop("the safetensors file does not read back equal: the pickle is left in place")
    rec = {"stage": "SPL1 A12 convert", "declared_in": DOC, "key": a.key, "tensors": len(sd), "parameters": int(sum(v.numel() for v in sd.values())), "bytes": out.stat().st_size,
           "sha256": S.sha256(out), "torch": torch.__version__, "dtype_set": sorted({str(v.dtype) for v in sd.values()})}
    S.wjson(S.OUT / f"convert_{a.key}.json", rec)
    S.replace_retry(pk, d / "pytorch_model.bin.unused")           # the loader now finds only the safetensors file
    log(f"converted {rec['tensors']} tensors, {rec['parameters'] / 1e6:.1f}M parameters, {rec['bytes'] / 1e6:.1f} MB: {out}; the pickle is renamed pytorch_model.bin.unused")


# ── the read and D3 ──────────────────────────────────────────────────────────────────────────────────────────

def ref_speed():
    """R: C0 at the contract D2 leaves in force: the adopted lever's SQuAD median block rate, else A7's."""
    lev = S.OUT / "lev"
    p = lev / "d2.json"
    if p.is_file():
        j = json.loads(p.read_text(encoding="utf-8"))
        if str(j.get("D2", {}).get("decision", "")).startswith("ADOPT_LEVER") and j.get("pick"):
            e = lev / j["pick"] / "squad" / "encode.json"
            if e.is_file():
                return {"basis": f"C0 with lever {j['pick']} (D2), SQuAD block median", "rate": json.loads(e.read_text(encoding="utf-8"))["block_rate_median"]}
    e = S.OUT / "c0" / "squad" / "encode.json"
    return {"basis": "C0 as the record (A7), SQuAD block median", "rate": json.loads(e.read_text(encoding="utf-8"))["block_rate_median"]}


def d3_of(cells, ratio):
    """cells: {dataset: {'rrf5': (diff, lo), 'seeds': (diff, lo)}}; ratio: speed over R's. Returns (verdict, why). D3 of section 8.12."""
    calls = {(ds, k): S.call_of(v[1]) for ds, d in cells.items() for k, v in d.items()}
    diffs = {(ds, k): v[0] for ds, d in cells.items() for k, v in d.items()}
    ok = all(c != "WORSE" for c in calls.values())
    price_ok = all(x >= PRICE_C for x in diffs.values())
    better_all = all(c == "BETTER" for c in calls.values())
    gain_b = all(cells[ds]["rrf5"][0] >= GAIN_B for ds in cells)
    if ok and ratio >= SPEED_A:
        return "POOL_READ", "(a) first stage passes and the speed ratio is at least 3.0"
    if ok and better_all and gain_b and ratio >= SPEED_B:
        return "POOL_READ", "(b) BETTER on H1 and H2 everywhere, H1 up by at least 1.0 point on each dataset, speed ratio at least 1.0"
    if (not ok) and price_ok and ratio >= SPEED_A:
        return "POOL_READ", "(c) first stage loses at most 1.5 points at a speed ratio of at least 3.0"
    if (ok or price_ok) and ratio < SPEED_A:
        return "RETIME_FP16", "would meet (a) or (c) except for the ratio: re-time at fp16 (arm l3 on the same weights) and call again"
    return "C0_STAYS", "no rule of D3 is met"


def cmd_read(a):
    import zenc as E
    t0 = time.time()
    res = {"stage": "SPL1 A12 read", "declared_in": DOC, "datasets": {}, "reference_speed": ref_speed()}
    for ds in S.SCREEN:
        c = S.context(E, ds)
        n = c["w"].n
        base, r20_c0 = S.reads_of(E, c["dense"], c["c0"], c["flat"], c["off"], c["gt"], n)
        if c["e0_rrf5"] is not None:
            gate = {"c0_rrf5": round(float(base["rrf5"].mean()), 6), "e0_filed_rrf5": c["e0_rrf5"], "pass": abs(float(base["rrf5"].mean()) - c["e0_rrf5"]) <= 5e-6}
        else:
            L = np.load(S.WS / "outputs" / "mhr1" / "first" / "lists.npz", allow_pickle=False)
            share = float(np.mean(np.all(L["e3v_splade"][:, :5].astype(np.int64) == r20_c0[:, :5], axis=1)))
            gl, go = L["golds_flat"], L["golds_off"]
            same_g = all(set(c["flat"][c["off"][i]:c["off"][i + 1]].tolist()) == set(gl[go[i]:go[i + 1]].tolist()) for i in range(len(c["rows"])))
            gate = {"c0_rrf5_top5_equals_mhr1_lists_share": round(share, 6), "golds_equal_mhr1": same_g, "pass": share >= 0.999 and same_g}
        log(f"{ds}: G-S1 {json.dumps(gate)}")
        if not gate["pass"]:
            stop(f"G-S1 failed on {ds}: the reader does not reproduce the filed first stage; nothing is read")
        dres = {"questions": int(len(c["rows"])), "nodes": n, "gate_gs1": gate, "c0": {k: round(float(v.mean() * 100), 3) for k, v in base.items()}, "models": {}}
        for key in KEYS:
            p = S.OUT / key / ds / "top1000.npz"
            if not p.is_file():
                continue
            z = np.load(p, allow_pickle=False)
            if not np.array_equal(z["rows"], c["rows"]):
                stop(f"{key} {ds}: the rows of the list are not the screen's rows")
            leg = z["ids"].astype(np.int64)
            if leg.shape[1] < c["dense"].shape[1]:
                leg = np.pad(leg, ((0, 0), (0, c["dense"].shape[1] - leg.shape[1])), constant_values=-1)
            r, _ = S.reads_of(E, c["dense"], leg, c["flat"], c["off"], c["gt"], n)
            mr = {"level": {k: round(float(v.mean() * 100), 3) for k, v in r.items()}, "diff_vs_c0": {}}
            for cell in ("rrf5", "seeds", "rrf20", "leg20", "leg1000"):
                if cell in r and cell in base:
                    dm, lo, hi = S.boot(r[cell] - base[cell])
                    mr["diff_vs_c0"][cell] = {"diff": round(dm, 3), "lo": round(lo, 3), "hi": round(hi, 3), "call": S.call_of(lo)}
            e = S.OUT / key / ds / "encode.json"
            enc = json.loads(e.read_text(encoding="utf-8")) if e.is_file() else {}
            mr.update({"block_rate_median": enc.get("block_rate_median"), "nodes_per_s_wall": enc.get("nodes_per_s"), "doc_nnz_mean": enc.get("doc_nnz_mean"),
                       "query_nnz_mean": enc.get("query_nnz_mean")})
            dres["models"][key] = mr
        res["datasets"][ds] = dres
    res["D3"] = {}
    R = res["reference_speed"]["rate"]
    for key in KEYS:
        if not all(key in res["datasets"][ds]["models"] for ds in S.SCREEN):
            continue
        sq = res["datasets"]["squad"]["models"][key]
        rate = sq["block_rate_median"] or sq["nodes_per_s_wall"]
        ratio = (rate / R) if (rate and R) else None
        cells = {ds: {k: (res["datasets"][ds]["models"][key]["diff_vs_c0"][k]["diff"], res["datasets"][ds]["models"][key]["diff_vs_c0"][k]["lo"]) for k in ("rrf5", "seeds")} for ds in S.SCREEN}
        verdict, why = d3_of(cells, ratio) if ratio is not None else ("NO_SPEED", "no block rate of the SQuAD run")
        res["D3"][key] = {"params_m": MODELS_M[key]["params_m"], "ratio": None if ratio is None else round(ratio, 3), "first_stage_ok": all(S.call_of(v[1]) != "WORSE" for d in cells.values() for v in d.values()),
                          "h1_h2": {ds: {k: list(v) for k, v in d.items()} for ds, d in cells.items()}, "verdict": verdict, "because": why}
        log(f"{key}: ratio {res['D3'][key]['ratio']} first stage ok {res['D3'][key]['first_stage_ok']} -> {verdict} ({why})")
    res["seconds"] = round(time.time() - t0, 1)
    S.wjson(S.OUT / "cand_read.json", res)
    print(json.dumps(res["D3"], indent=1))


# ── self-test ────────────────────────────────────────────────────────────────────────────────────────────────

def selftest():
    bad = 0

    def chk(name, ok):
        nonlocal bad
        log(f"  {'ok  ' if ok else 'FAIL'} {name}")
        bad += 0 if ok else 1

    def cells(h1, h2, lo_h1=None, lo_h2=None):
        return {ds: {"rrf5": (h1, h1 - 0.5 if lo_h1 is None else lo_h1), "seeds": (h2, h2 - 0.5 if lo_h2 is None else lo_h2)} for ds in S.SCREEN}

    chk("the revisions are 40 hex digits and the keys are the four of A12", all(len(m["rev"]) == 40 and set(m["rev"]) <= set("0123456789abcdef") for m in MODELS_M.values()) and KEYS == ("c4", "c5", "c6", "c7"))
    chk("the fetch rule: the pickle only for c5 and c6; never a hidden file, a .py or an onnx folder",
        take_file("c5", "pytorch_model.bin") and take_file("c6", "pytorch_model.bin") and not take_file("c4", "pytorch_model.bin") and not take_file("c7", "pytorch_model.bin")
        and take_file("c4", "model.safetensors") and take_file("c5", "1_SpladePooling/config.json") and not take_file("c5", ".gitattributes")
        and not take_file("c4", "modeling.py") and not take_file("c4", "onnx/model.safetensors"))
    chk("zspl1.py's table carries the four keys with their revisions", all(S.MODELS[k]["rev"] == MODELS_M[k]["rev"] and S.MODELS[k]["repo"] == MODELS_M[k]["repo"] for k in KEYS))
    chk("D3 (a): first stage level and 3.2 times faster -> pool read", d3_of(cells(0.2, 0.1), 3.2)[0] == "POOL_READ")
    chk("D3 (b): BETTER by 1.3 points on H1 and H2 everywhere, ratio 1.0 -> pool read", d3_of(cells(1.3, 1.3, 0.4, 0.4), 1.0)[0] == "POOL_READ")
    chk("D3 (b) needs a ratio of 1.0: 1.3 points but a ratio of 0.9 is no pool read (it is re-timed at fp16, as every first-stage-OK candidate below 3.0)", d3_of(cells(1.3, 1.3, 0.4, 0.4), 0.9)[0] == "RETIME_FP16")
    chk("D3 (c): WORSE by 1.2 points at 3.5 times the speed -> pool read", d3_of(cells(-1.2, -1.2, -1.6, -1.6), 3.5)[0] == "POOL_READ")
    chk("D3: WORSE by 2.0 points at 5 times the speed -> C0 stays", d3_of(cells(-2.0, -2.0, -2.5, -2.5), 5.0)[0] == "C0_STAYS")
    chk("D3: level and 1.5 times faster -> re-time at fp16", d3_of(cells(0.1, 0.1), 1.5)[0] == "RETIME_FP16")
    chk("D3: level and slower than C0 -> re-time at fp16 (the ratio is below 3.0)", d3_of(cells(0.1, 0.1), 0.6)[0] == "RETIME_FP16")
    chk("the call is section 5's: -1.0 is LEVEL, -1.01 is WORSE", (S.call_of(-1.0), S.call_of(-1.01)) == ("LEVEL", "WORSE"))
    if bad:
        raise SystemExit(f"zspl1m: selftest FAILED ({bad})")
    log("selftest PASS (11 checks)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(fn=lambda a: selftest())
    f = sub.add_parser("fetch")
    f.add_argument("--key", required=True, choices=KEYS)
    f.set_defaults(fn=cmd_fetch)
    v = sub.add_parser("convert")
    v.add_argument("--key", required=True, choices=KEYS)
    v.set_defaults(fn=cmd_convert)
    k = sub.add_parser("check")
    k.add_argument("--key", required=True, choices=KEYS)
    k.add_argument("--device", default=None)
    k.set_defaults(fn=S.cmd_check)
    e = sub.add_parser("encode")
    e.add_argument("--key", required=True, choices=KEYS)
    e.add_argument("--ds", required=True, choices=S.SCREEN)
    e.add_argument("--device", default=None)
    e.add_argument("--smoke", type=int, default=0)
    e.add_argument("--batch", type=int, default=S.BATCH)
    e.add_argument("--threads", type=int, default=0)
    e.set_defaults(fn=S.cmd_encode)
    sub.add_parser("read").set_defaults(fn=cmd_read)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
