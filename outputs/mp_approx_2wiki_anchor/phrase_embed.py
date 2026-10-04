"""Design look (untracked; not a result and not filed): text vectors for the anchor phrases of 2wiki, hotpotqa and musique.

Each dataset's anchors_compact.npz lists its w2 phrases by count (w2_top). The union of the top K phrases of the three
datasets goes through Alibaba-NLP/gte-Qwen2-1.5B-instruct exactly as scripts/m3b_relations.py runs it (document mode, no
instruction prefix, the pooling the model ships with, max_seq_length 64, bfloat16 weights, unit-L2, stored float16), so a
phrase vector sits in the space of the served query and node embeddings. Placeholders become words before encoding:
<C> -> "Name", <N> -> "number", <s> -> nothing (a phrase left empty becomes "start"). The model comes from the local HF
hub with no network access. Self-check: metaqa's 9 relation texts re-encoded against outputs/m3b/relations (cosine).

Writes outputs/mp_approx_2wiki_anchor/phrase_emb/union_K{K}.npz (texts, phrases, emb) and, per dataset, its host
directory's phrase_emb_w2_K{K}.npy: row r is w2_top[r]'s vector (float16, (K, 1536)). Resumable: chunks of 64 are kept
in union_K{K}.partial.npz.

    python outputs/mp_approx_2wiki_anchor/phrase_embed.py --K 1024
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
HOSTS = {"2wiki": ROOT / "outputs" / "mp_approx_2wiki_anchor" / "host",
         "hotpotqa": ROOT / "outputs" / "mp_approx_hotpot_anchor" / "host",
         "musique": ROOT / "outputs" / "mp_approx_musique_anchor" / "host"}
COMPACT_SHA = {"2wiki": "12815fa52ff673e1e149382d33ab15a947ead1a9c4e20ea963b58e1e348220ca", "hotpotqa": "cd8db96fd72d5d2c304c1934e734c70f45a0151bfc4c000856ed466d455830c4",
               "musique": "eb08ae611de798f76533108d7abf89d18a731e1aa930d7a81608c1d25d97c4d6"}
CHUNK = 64


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def text_of(phrase):
    t = " ".join(w for w in phrase.replace("<C>", "Name").replace("<N>", "number").replace("<s>", " ").split())
    return t or "start"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    tops, shas = {}, {}
    for ds, d in HOSTS.items():
        p = d / "anchors_compact.npz"
        shas[ds] = sha_file(p)
        if shas[ds] != COMPACT_SHA[ds]:
            raise SystemExit(f"{p} is not the pinned file")
        with np.load(p) as z:
            tops[ds] = [str(t) for t in z["w2_top"][:a.K]]
        if len(tops[ds]) != a.K:
            raise SystemExit(f"{ds}: fewer than {a.K} phrases")
    phrases = sorted(set().union(*tops.values()))
    texts = [text_of(p) for p in phrases]
    out_dir = HERE / "phrase_emb"
    out_dir.mkdir(exist_ok=True)
    part = out_dir / f"union_K{a.K}.partial.npz"
    emb = np.zeros((len(phrases), 1536), dtype=np.float32)
    done = 0
    if part.exists():
        with np.load(part) as z:
            if list(z["phrases"]) == phrases:
                done = int(z["done"])
                emb[:done] = z["emb"][:done]
    print(f"{len(phrases)} phrases (union of top {a.K}), {done} already encoded", flush=True)
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(a.threads)
    local = snapshot_download(MODEL, local_files_only=True)
    model = SentenceTransformer(local, device="cpu", trust_remote_code=True, local_files_only=True, model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = 64
    model[0].auto_model.config.use_cache = False
    ref = np.load(ROOT / "outputs" / "m3b" / "relations" / "metaqa_rel_embeddings.npy").astype(np.float32)
    voc = json.loads((ROOT / "outputs" / "m3b" / "relations" / "metaqa_rel_vocab.json").read_text(encoding="utf-8"))
    chk = model.encode(voc["texts"], batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    cos = [round(float(x @ y), 4) for x, y in zip(chk, ref)]
    print("self-check cosines", cos, flush=True)
    if min(cos) < 0.99:
        raise SystemExit("the encoder does not reproduce the stored relation vectors")
    for s in range(done, len(phrases), CHUNK):
        e = model.encode(texts[s:s + CHUNK], batch_size=CHUNK, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        emb[s:s + len(e)] = np.asarray(e, dtype=np.float32)
        done = s + len(e)
        tmp = out_dir / f"union_K{a.K}.partial.tmp.npz"
        np.savez(tmp, phrases=np.asarray(phrases), emb=emb, done=done)
        os.replace(tmp, part)
        print(f"  {done}/{len(phrases)} {time.time() - t0:.0f}s", flush=True)
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-2):
        raise SystemExit("unexpected norms")
    union = out_dir / f"union_K{a.K}.npz"
    np.savez(union, phrases=np.asarray(phrases), texts=np.asarray(texts), emb=emb.astype(np.float16))
    idx = {p: i for i, p in enumerate(phrases)}
    rec = {"model": MODEL, "mode": "document (no instruction prefix)", "max_seq_length": 64, "weights_dtype": "bfloat16", "K": a.K,
           "phrases": len(phrases), "self_check_cosines": cos, "compact_sha256": shas, "union_sha256": sha_file(union), "per_dataset": {}}
    for ds, top in tops.items():
        arr = emb[[idx[p] for p in top]].astype(np.float16)
        dst = HOSTS[ds] / f"phrase_emb_w2_K{a.K}.npy"
        np.save(dst, arr)
        rec["per_dataset"][ds] = {"file": str(dst.relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(dst), "rows": len(top)}
    rec["seconds"] = round(time.time() - t0, 1)
    rec["script_sha256"] = sha_file(Path(__file__))
    (out_dir / f"phrase_embed_K{a.K}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
