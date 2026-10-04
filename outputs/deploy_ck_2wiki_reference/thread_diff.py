"""How large is the BLAS-thread-count difference in the packed eval features? Packs eval block 0 at 6 and at 8 threads
and compares x element by element: how many entries differ, by how much, and in which columns. Writes nothing under
the repository."""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "8"
REPO = r"C:\Users\Swastik\Desktop\message-passing-retrieval"
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402

import deploy_ck_2wiki as D  # noqa: E402
from mp_retrieval.universal_v2_features import IDX  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

decl = D.load_declaration()
cfg, cfg_m3b, cfg_h, inputs, contexts, handles, pkg, bank, m3b_compile = D.open_2wiki(decl, host=False, log=D.log_utc)
m3b_contract = D.V2.M3B_RUN.load_script("m3b_contract")
ev = D.eval_population(decl, cfg, cfg_m3b, cfg_h, contexts[D.NAME], handles[D.NAME], pkg, m3b_compile, m3b_contract,
                       shard=None, log=D.log_utc)
idx = D.eval_chunks(ev["n"], ev["chunk"])[0]
xs = {}
for k in (6, 8):
    with threadpool_limits(limits=k, user_api="blas"):
        batch = D.eval_chunk_batch(ev, contexts[D.NAME], inputs["column_indices"], idx, m3b_compile)
    xs[k] = batch.x.numpy().copy()
a, b = xs[6], xs[8]
diff = a != b
full = {i: n for n, i in IDX.items()}
kept = list(inputs["column_indices"])
names = {c: full.get(int(kept[c]), "?") for c in range(len(kept))}
print("x columns", a.shape[1], "kept feature columns", len(kept))
cols = np.nonzero(diff.any(axis=0))[0]
print("x", a.shape, "entries differing", int(diff.sum()), "of", a.size, "rows differing", int(diff.any(axis=1).sum()))
print("max abs diff", float(np.abs(a - b).max()), "max rel diff", float((np.abs(a - b) / np.maximum(np.abs(a), 1e-30))[diff].max()) if diff.any() else 0.0)
for c in cols:
    d = diff[:, c]
    print(f"col {c} ({names.get(c, '?')}): {int(d.sum())} differ, max |d| {float(np.abs(a[:, c] - b[:, c]).max()):.3g}")
