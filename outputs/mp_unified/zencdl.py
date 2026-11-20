"""The frozen encoder (Alibaba-NLP/gte-Qwen2-1.5B-instruct) into the host's huggingface cache, at the revision the
record's G0 used (outputs/enc/g0.json), for E2B's frozen windows (docs/ENC_ENCODER_COMPARISON.md section 16).

The host never held the model: the frozen vectors are the record's, and G0 ran on the laptop. zenc2b loads the frozen
model with no revision and local files only, so the cache's main must be the record's revision: the download is of
main, and it stops unless main is that revision. Then the model loads offline on the CPU exactly as zenc2b loads it.
Writes outputs/enc/e2b/frozen_model.json.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import zenc as E  # noqa: E402  (first: sets the BLAS threads before numpy loads)

import json  # noqa: E402


def main():
    from huggingface_hub import HfApi, snapshot_download
    rev = json.loads((E.OUT / "g0.json").read_text(encoding="utf-8"))["revision"]
    repo = E.ENCODERS["frozen"]["repo"]
    head = HfApi().model_info(repo).sha
    if head != rev:
        E.stop(f"{repo}: main is {head}, the record's G0 used {rev}; not downloaded")
    t0 = time.time()
    path = Path(snapshot_download(repo, revision="main", allow_patterns=E.ALLOW, ignore_patterns=E.IGNORE))
    if path.name != rev:
        E.stop(f"{repo}: the cache's snapshot is {path.name}, want {rev}")
    files = {str(f.relative_to(path)).replace("\\", "/"): {"bytes": f.stat().st_size, "sha256": E.sha256(f)}
             for f in sorted(path.rglob("*")) if f.is_file()}
    dl_s = round(time.time() - t0, 1)
    print(f"downloaded {repo} {rev[:12]}: {sum(v['bytes'] for v in files.values())} bytes in {dl_s} s", flush=True)
    m = E.load_model("frozen", "cpu", E.pinned("frozen"))
    dim = int(m.get_sentence_embedding_dimension())
    if dim != E.ENCODERS["frozen"]["dim"]:
        E.stop(f"frozen: width {dim}, want {E.ENCODERS['frozen']['dim']}")
    rec = {"repo": repo, "revision": rev, "head_at_download": head, "seconds": dl_s,
           "bytes": sum(v["bytes"] for v in files.values()), "files": files, "dim": dim,
           "params": int(sum(t.numel() for t in m.parameters())), "loads_offline": True,
           "code_sha256": {"outputs/mp_unified/zencdl.py": E.sha256(Path(__file__)),
                           "outputs/mp_unified/zenc.py": E.sha256(Path(E.__file__))},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (E.OUT / "e2b").mkdir(parents=True, exist_ok=True)
    E.wjson(E.OUT / "e2b" / "frozen_model.json", rec)
    print(f"frozen loads offline: {rec['params']} params, width {dim}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
