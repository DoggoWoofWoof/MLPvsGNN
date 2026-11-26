"""SPL1 amendment A7 (docs/SPL1_SPARSE_LEG_SCREEN.md section 8.7): the record's own sparse leg (C0) on the host card, for its costs only.

    python -B -E -P outputs/mp_unified/zspl1c.py selftest                    # laptop
    python -I -B    outputs/mp_unified/zspl1c.py fetch                      # host: C0's files into a new empty folder (one pickle, 438 MB)
    python -I -B    outputs/mp_unified/zspl1c.py convert                    # host: the pickle read with torch.load(weights_only=True), written as safetensors
    python -I -B    outputs/mp_unified/zspl1c.py encode --ds squad          # blocks of 2,048 nodes timed; non-zeros per node and per question
    python -I -B    outputs/mp_unified/zspl1c.py encode --ds squad --smoke 512 --device cpu --threads 4

Run from the workspace root. C0 = naver/splade-cocondenser-ensembledistil at the record's pinned revision, the record's contract (zmhr.py, zstk1s.py):
256 tokens, batches of 64 in file order padded to the longest in the batch, log1p(relu(logits)) max-pooled over every position, padding included.
The forward pass of a batch is split into chunks of 16 rows of the SAME padded tensor (same vectors, less card memory). Nothing is scored: C0's lists
are the record's (SPL1 gate G-S1); this module gives C0's encode time and vector sizes beside the candidates'. The fetched pickle is read once, only
with torch.load(weights_only=True), and the model is then loaded from the safetensors file alone (use_safetensors=True); nothing is executed from it.
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

REPO, REV = "naver/splade-cocondenser-ensembledistil", "49cf4c7b0db5b870a401ddf5e2669993ef3699c7"
LEN, BATCH, BLOCK, CHUNK = 256, 64, 2048, 16
DEST = S.OUT / "models" / "c0"
FILES_OK = {"config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "pytorch_model.bin"}
CAP_BIN_MB, CAP_ALL_MB = 460, 480
MHR_NNZ = 213.2                                       # the record's stored MultiHop-RAG documents (outputs/mhr1/package/.../splade/docs/manifest.json)
log, stop = S.log, S.stop


def cmd_fetch(a):
    from huggingface_hub import HfApi, hf_hub_download
    if DEST.exists() and any(DEST.iterdir()):
        stop(f"{DEST} is not empty: each model goes into its own new empty folder")
    DEST.mkdir(parents=True, exist_ok=True)
    files = [(x.path, int(x.size or 0)) for x in HfApi().list_repo_tree(REPO, revision=REV, recursive=True, token=False) if hasattr(x, "size")]
    take = [(p, s) for p, s in files if p in FILES_OK]
    skip = [p for p, _ in files if p not in FILES_OK]
    total = sum(s for _, s in take)
    log(f"{REPO} @ {REV[:8]}: {len(files)} files, {len(take)} taken, {total / 1e6:.1f} MB; skipped {skip}")
    for p, s in take:
        log(f"  {p}  {s / 1e6:.2f} MB")
    if not any(p == "pytorch_model.bin" for p, _ in take):
        stop("pytorch_model.bin is not in the listing: nothing fetched")
    if dict(take)["pytorch_model.bin"] > CAP_BIN_MB * 1e6 or total > CAP_ALL_MB * 1e6:
        stop(f"the pickle is over {CAP_BIN_MB} MB or the files over {CAP_ALL_MB} MB: nothing fetched")
    rec = {"stage": "SPL1 C0 fetch", "declared_in": S.DOC, "repo": REPO, "revision": REV, "files": {}, "skipped": skip}
    for p, s in take:
        got = hf_hub_download(repo_id=REPO, filename=p, revision=REV, local_dir=str(DEST), token=False)
        rec["files"][p] = {"bytes": os.path.getsize(got), "listed_bytes": s, "sha256": S.sha256(got)}
    rec["total_bytes"] = sum(v["bytes"] for v in rec["files"].values())
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    S.wjson(S.OUT / "fetch_c0.json", rec)
    log(f"fetched {len(take)} files, {rec['total_bytes'] / 1e6:.1f} MB into {DEST}")


def cmd_convert(a):
    import torch
    from safetensors.torch import save_file
    out = DEST / "model.safetensors"
    if out.exists():
        stop(f"{out} exists")
    sd = torch.load(DEST / "pytorch_model.bin", map_location="cpu", weights_only=True)    # the restricted unpickler: tensors and containers only
    if not isinstance(sd, dict) or not sd or not all(isinstance(v, torch.Tensor) for v in sd.values()):
        stop("the pickle is not a plain dict of tensors: nothing written")
    sd = {k: v.detach().contiguous().clone() for k, v in sd.items()}
    save_file(sd, str(out), metadata={"format": "pt"})
    rec = {"stage": "SPL1 C0 convert", "tensors": len(sd), "parameters": int(sum(v.numel() for v in sd.values())), "bytes": out.stat().st_size,
           "sha256": S.sha256(out), "torch": torch.__version__, "dtype_set": sorted({str(v.dtype) for v in sd.values()})}
    S.wjson(S.OUT / "convert_c0.json", rec)
    log(f"converted {rec['tensors']} tensors, {rec['parameters'] / 1e6:.1f}M parameters, {rec['bytes'] / 1e6:.1f} MB: {out}")


def pooled(torch, tok, mdl, batch, device, chunk=CHUNK):
    """The record's vector for one batch: padded to the longest in the batch, log1p(relu(logits)) max-pooled over every position. The forward pass is
    split in chunks of `chunk` rows of the same padded tensor, so the vectors are the unsplit batch's."""
    enc = tok([str(t) for t in batch], return_tensors="pt", truncation=True, max_length=LEN, padding=True)
    parts = []
    with torch.no_grad():
        for i in range(0, len(batch), chunk):
            x = {k: v[i:i + chunk].to(device) for k, v in enc.items()}
            parts.append(torch.log1p(torch.relu(mdl(**x).logits)).max(dim=1).values.float().cpu())
    am = enc["attention_mask"]
    return torch.cat(parts).numpy(), int(am.sum()), int((am.sum(dim=1) >= LEN).sum())


def cmd_encode(a):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    import zworld
    t0 = time.time()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and a.threads:
        torch.set_num_threads(a.threads)
    w = zworld.load(a.ds)
    rows, _ = S.eval_rows(a.ds, w)
    qt = S.question_texts(w, rows)
    texts = [t for _, t in w.iter_nodes()]
    if a.smoke:
        texts, qt = texts[:a.smoke], qt[:BATCH]                  # the CPU timing row: the first nodes only, a short question pass
    n = len(texts)
    if not (DEST / "model.safetensors").is_file():
        stop("no model.safetensors: run fetch and convert first")
    tok = AutoTokenizer.from_pretrained(str(DEST), local_files_only=True)
    mdl = AutoModelForMaskedLM.from_pretrained(str(DEST), local_files_only=True, use_safetensors=True).eval().to(device)
    d = S.OUT / "c0" / (a.ds + (f"_smoke_{device}" if a.smoke else ""))
    d.mkdir(parents=True, exist_ok=True)
    bl = d / "blocks.jsonl"
    if bl.is_file():
        bl.unlink()
    gpu = None
    if device == "cuda":
        fm, tm = torch.cuda.mem_get_info()
        gpu = {"name": torch.cuda.get_device_name(0), "free_mb_at_start": int(fm / 2**20), "total_mb": int(tm / 2**20)}
    t1 = time.time()
    qn = 0
    for i in range(0, len(qt), BATCH):
        v, _, _ = pooled(torch, tok, mdl, qt[i:i + BATCH], device)
        qn += int(np.count_nonzero(v))
    log(f"c0 {a.ds}: {len(qt)} questions encoded in {time.time() - t1:.1f} s on {device}, query nnz mean {qn / max(len(qt), 1):.1f}, {n:,} nodes")
    nnz = cut = done = 0
    t2 = time.time()
    while done < n:
        end = min(n, done + BLOCK)
        tb = time.time()
        for i in range(done, end, BATCH):
            v, _, c = pooled(torch, tok, mdl, texts[i:min(i + BATCH, end)], device)
            nnz += int(np.count_nonzero(v))
            cut += c
        dt = time.time() - tb
        with open(bl, "a", encoding="utf-8") as f:
            f.write(json.dumps({"end": end, "nodes": end - done, "seconds": round(dt, 3), "t": round(time.time(), 1)}) + "\n")
        done = end
        log(f"  {done:,}/{n:,} nodes, {done / max(time.time() - t2, 1e-9):.0f} nodes/s so far")
    el = time.time() - t2
    brs, brm = S.block_rates([json.loads(x) for x in bl.read_text(encoding="utf-8").splitlines() if x.strip()])
    rec = {"stage": "SPL1 C0 encode", "declared_in": S.DOC, "repo": REPO, "revision": REV, "dataset": a.ds, "smoke": a.smoke, "nodes": n,
           "questions": len(qt), "device": device, "threads": (a.threads or None) if device == "cpu" else None, "max_len": LEN, "batch": BATCH, "chunk": CHUNK,
           "nodes_per_s": round(n / max(el, 1e-9), 2), "block_rate_median": None if brm is None else round(brm, 2), "block_rates": [round(x, 1) for x in brs],
           "encode_seconds": round(el, 1), "seconds": round(time.time() - t0, 1), "doc_nnz_mean": round(nnz / max(n, 1), 1),
           "doc_bytes_per_node_at_8": round(8 * nnz / max(n, 1), 1), "query_nnz_mean": round(qn / max(len(qt), 1), 1), "gpu": gpu,
           "cut_at_256": cut, "cut_counted_on_nodes": n, "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
    if a.ds == "multihoprag" and not a.smoke:
        rec["G_C0"] = {"filed_nnz_mean": MHR_NNZ, "this_nnz_mean": rec["doc_nnz_mean"], "pass": abs(rec["doc_nnz_mean"] - MHR_NNZ) <= 1.0}
    S.wjson(d / "encode.json", rec)
    log(f"done c0 {a.ds}: {rec['nodes_per_s']} nodes/s wall, block median {rec['block_rate_median']}, nnz {rec['doc_nnz_mean']}, {rec['seconds']} s")


def selftest():
    import torch
    bad = 0

    def chk(name, ok):
        nonlocal bad
        log(f"  {'ok  ' if ok else 'FAIL'} {name}")
        bad += 0 if ok else 1

    class Tok:                                                      # a fixed-length-per-batch toy: pads to the longest
        def __call__(self, texts, return_tensors, truncation, max_length, padding):
            ids = [[(ord(c) % 7) + 1 for c in t][:max_length] for t in texts]
            L = max(len(x) for x in ids)
            return {"input_ids": torch.tensor([x + [0] * (L - len(x)) for x in ids]),
                    "attention_mask": torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids])}

    class Mdl:
        def __init__(self):
            g = torch.Generator().manual_seed(0)
            self.W = torch.randn(8, 12, generator=g)

        def __call__(self, input_ids, attention_mask):
            return type("O", (), {"logits": self.W[input_ids] - 0.5})()

    texts = ["abc", "abcdefghij", "xyz", "q", "hello world", "zz", "mm mm", "the end"]
    v1, t1, c1 = pooled(torch, Tok(), Mdl(), texts, "cpu", chunk=8)
    v2, t2, c2 = pooled(torch, Tok(), Mdl(), texts, "cpu", chunk=3)
    chk("chunks of any size give the unsplit batch's vectors", np.array_equal(v1, v2) and t1 == t2 and c1 == c2)
    chk("padding positions are pooled (the record's contract): a short text beside a long one gets the padding column's value",
        bool((v1[3] > 0).any()) and v1.shape == (8, 12) and (v1 >= 0).all())
    chk("the file allow-list has the pickle and no code", "pytorch_model.bin" in FILES_OK and not any(x.endswith(".py") for x in FILES_OK))
    if bad:
        raise SystemExit(f"zspl1c: selftest FAILED ({bad})")
    log(f"selftest PASS (3 checks)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(fn=lambda a: selftest())
    sub.add_parser("fetch").set_defaults(fn=cmd_fetch)
    sub.add_parser("convert").set_defaults(fn=cmd_convert)
    e = sub.add_parser("encode")
    e.add_argument("--ds", required=True, choices=S.SCREEN)
    e.add_argument("--device", default=None)
    e.add_argument("--smoke", type=int, default=0)
    e.add_argument("--threads", type=int, default=0)
    e.set_defaults(fn=cmd_encode)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
