"""Does packing the eval blocks depend on the BLAS thread count? The laptop reference was packed at 6 threads; the host
stages run at 8. Prepares the eval population at 8 threads (the host's count), then packs block 0 at 8, 6, 4 and 1
threads and compares each with the reference's block 0. Writes nothing under the repository."""
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "8"
REPO = r"C:\Users\Swastik\Desktop\message-passing-retrieval"
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import json  # noqa: E402

import deploy_ck_2wiki as D  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

print([(d["internal_api"], d.get("version"), d["num_threads"]) for d in threadpool_info()], flush=True)
decl = D.load_declaration()
cfg, cfg_m3b, cfg_h, inputs, contexts, handles, pkg, bank, m3b_compile = D.open_2wiki(decl, host=False, log=D.log_utc)
m3b_contract = D.V2.M3B_RUN.load_script("m3b_contract")
ref = json.loads(D.REFERENCE.read_text(encoding="utf-8"))
with threadpool_limits(limits=8, user_api="blas"):
    ev = D.eval_population(decl, cfg, cfg_m3b, cfg_h, contexts[D.NAME], handles[D.NAME], pkg, m3b_compile, m3b_contract,
                           shard=None, log=D.log_utc)
print("population", ev["n"], ev["chunk"], ev["pop"].digest == ref["eval"]["ids_sha256"], ev["chunk"] == ref["eval"]["chunk"], flush=True)
for k in (8, 6, 4, 1):
    t = time.time()
    with threadpool_limits(limits=k, user_api="blas"):
        got = D.eval_batch_pins(ev, contexts[D.NAME], inputs["column_indices"], m3b_compile, n=1)
    probs = D.batch_problems(ref["eval"]["batches"][:1], got)
    print(f"{k} threads: {'EQUAL' if not probs else probs} ({time.time() - t:.0f}s)", flush=True)
