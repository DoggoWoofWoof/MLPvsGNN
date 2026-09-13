"""Relation text embeddings for the typed datasets
(configs/m3b_controlled_comparison.yaml#pool_graph.relation_text_embeddings).

    python scripts/m3b_relations.py                # metaqa (9) and webqsp (7,058)

The rel_vocab strings of the served structural graph, with dots and underscores
replaced by spaces, through Alibaba-NLP/gte-Qwen2-1.5B-instruct in document
mode (no instruction prefix, the sentence-transformers pooling the model ships
with), unit-L2, float16. Read-only against the package; the model is loaded
from the local HF hub with no network access. Outputs under outputs/m3b/relations/.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "m3b_controlled_comparison.yaml"
OUT = ROOT / "outputs" / "m3b" / "relations"
MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"


def relation_text(raw: str) -> str:
    return " ".join(raw.replace(".", " ").replace("_", " ").split())


def load_script(name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    package_root = Path(cfg["substrate"]["package_root"])
    served = package_root / "data" / "final_canonical"
    m3b_compile = load_script("m3b_compile")
    canonical = m3b_compile.import_loader_readonly(served)   # never writes under the root; reports a cache left by others
    freeze = json.loads((served / "CANONICAL_FREEZE.json").read_text(encoding="utf-8"))
    if freeze["RECORD_SHA256"] != cfg["substrate"]["freeze_RECORD_SHA256_expected"]:
        raise SystemExit("served freeze mismatch; refusing")
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(int(os.environ.get("M3B_THREADS", "6")))
    # bfloat16 weights: the float32 load exceeds this machine's commit limit while the
    # contract measurement runs; the outputs are cast to float32 before normalisation
    from huggingface_hub import snapshot_download
    local = snapshot_download(MODEL, local_files_only=True)   # the hub cache; no network
    model = SentenceTransformer(local, device="cpu", trust_remote_code=True, local_files_only=True,
                                model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = 64
    # the repo's remote code predates transformers 4.57's cache API; no KV cache is needed to embed
    model[0].auto_model.config.use_cache = False
    OUT.mkdir(parents=True, exist_ok=True)
    # stack self-check: three served node texts re-encoded in document mode against their served vectors
    check = {}
    probe = canonical.Dataset(cfg["pool_graph"]["relation_text_embeddings"]["datasets"][0], root=str(served))
    texts, rows = [], []
    for i, node in enumerate(probe.nodes()):
        if i in (0, 1000, 20000):
            texts.append(str(node.get("text") or node.get("name") or node.get("title") or ""))
            rows.append(i)
        if i > 20000:
            break
    served_vecs = np.asarray(probe.embeddings("dense", "docs").read(np.asarray(rows)), dtype=np.float32)
    ours = np.asarray(model.encode(texts, batch_size=4, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
    check["rows"] = rows
    check["cosine_with_served"] = [round(float(a @ b), 5) for a, b in zip(ours, served_vecs)]
    print("stack self-check cosines:", check["cosine_with_served"], flush=True)
    record = {"stack_self_check": check}
    for name in cfg["pool_graph"]["relation_text_embeddings"]["datasets"]:
        ds = canonical.Dataset(name, root=str(served))
        g = ds.graph("structural")
        vocab = [str(r) for r in g.rel_vocab]
        del g
        texts = [relation_text(r) for r in vocab]
        t0 = time.time()
        emb = model.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1)
        if emb.shape != (len(vocab), 1536) or not np.allclose(norms, 1.0, atol=1e-2):
            raise SystemExit(f"{name}: unexpected embedding shape or norms")
        np.save(OUT / f"{name}_rel_embeddings.npy", emb.astype(np.float16))
        vocab_sha = hashlib.sha256("\n".join(vocab).encode("utf-8")).hexdigest()
        (OUT / f"{name}_rel_vocab.json").write_text(json.dumps({"vocab": vocab, "texts": texts, "sha256": vocab_sha}, ensure_ascii=False), encoding="utf-8")
        record[name] = {"relations": len(vocab), "vocab_sha256": vocab_sha, "seconds": round(time.time() - t0, 1),
                        "embeddings_sha256": hashlib.sha256((OUT / f"{name}_rel_embeddings.npy").read_bytes()).hexdigest(),
                        "examples": texts[:3]}
        print(f"{name}: {len(vocab)} relations in {record[name]['seconds']}s", flush=True)
    record["_meta"] = {"model": MODEL, "mode": "document (no instruction prefix)", "max_seq_length": 64, "weights_dtype": "bfloat16", "normalised": True,
                       "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "freeze_RECORD_SHA256": freeze["RECORD_SHA256"]}
    (OUT / "RECORD.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
