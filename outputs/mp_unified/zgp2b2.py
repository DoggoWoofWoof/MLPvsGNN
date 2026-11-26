"""GP2 family B2 (embedding windows) is COSTED here, never run (docs/GP2_UNIVERSAL_LINK_SCORE.md, section 4, row B2): how many 64-token mention windows a
window-similarity score would have to encode on the dev population, from GP1's link rows (kind, weight, mention level) with nothing encoded and no GPU.

    python outputs/mp_unified/zgp2b2.py            # every judged cell, parts q and r, whatever is present; writes outputs/gp2/b2_cost.json

A link row is a mention link when its stored mention level (exact) is above 0, the rows B1 gives a text window. Needed rows count 1 each; a sampled control row
counts its weight (the exact candidates it stands for), so the weighted sum estimates the candidate links of the dev population. One window per candidate
mention link and question is the upper bound on windows to encode (the same link in two questions' frontiers is one window when windows are stored per link).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import zgp2 as Z  # noqa: E402

GP, G1 = Z.GP, Z.G1
TOKENS = 64                       # E6's window length
BYTES_FP16, BYTES_INT8 = 1152 * 2, 1152         # e3v is 1152 wide; E5/E7: int8 keeps cosine order
GPU_S_PER_M_DOCS = (479, 3173)    # docs/ENC_ENCODER_COMPARISON.md section 8, mpnet and qwen3e06 (the frozen e3v is not in that table): bracket only


def main():
    Z.gate_pins()
    t0 = time.time()
    out = {"declared_in": Z.DECLARED, "zgp2_sha256": Z.script_sha256(), "tokens_per_window": TOKENS, "bytes_per_window_fp16": BYTES_FP16, "bytes_per_window_int8": BYTES_INT8,
           "gpu_seconds_per_million_documents_bracket": list(GPU_S_PER_M_DOCS), "cells": {}}
    tot = {"q": 0.0, "r": 0.0}
    for part in ("q", "r"):
        for d, v in Z.CELLS:
            key = f"{d}/{v}/{part}"
            T = GP.load_cell(d, v, part, cols=("q", "kind", "w", "exact"), gold=False)
            if T is None:
                continue
            if "not_read" in T:
                out["cells"][key] = {"status": "NOT_READ", "why": T["not_read"]}
                continue
            nq = int(np.unique(T["q"]).size)            # the part's questions (the record's G-G1 count is the whole dev population)
            kind, w, ex = T["kind"], T["w"].astype(np.float64), T["exact"]
            nd = kind == 1
            mention = ex > 0
            cand = float(nd.sum() + w[~nd].sum())
            cand_m = float((nd & mention).sum() + w[(~nd) & mention].sum())
            pairs = None
            gp = Z.OUT / d / f"gates_{part}.json"
            if gp.is_file():
                g = json.loads(gp.read_text(encoding="utf-8"))["cells"].get(v, {})
                pairs = g.get("mention_pairs")
            out["cells"][key] = {"status": "read", "questions": nq, "candidate_links_est": round(cand), "mention_links_est": round(cand_m),
                                 "mention_share": round(cand_m / cand, 4) if cand else None, "mention_links_per_question": round(cand_m / nq, 1) if nq else None,
                                 "mention_rows_read": int((mention).sum()), "distinct_mention_pairs_read": pairs,
                                 "tokens_upper": round(cand_m * TOKENS), "gb_fp16_upper": round(cand_m * BYTES_FP16 / 1e9, 2), "gb_int8_upper": round(cand_m * BYTES_INT8 / 1e9, 2),
                                 "gpu_hours_bracket": [round(cand_m / 1e6 * s / 3600, 2) for s in GPU_S_PER_M_DOCS]}
            tot[part] += cand_m
            print(key, json.dumps(out["cells"][key]))
            del T, kind, w, ex, nd, mention
    out["mention_links_est_total"] = {k: round(x) for k, x in tot.items()}
    out["gpu_hours_bracket_total"] = {k: [round(x / 1e6 * s / 3600, 2) for s in GPU_S_PER_M_DOCS] for k, x in tot.items()}
    out["seconds"] = round(time.time() - t0, 1)
    GP.wjson(Z.OUT / "b2_cost.json", out)
    print("total", json.dumps({"mention_links_est_total": out["mention_links_est_total"], "gpu_hours_bracket_total": out["gpu_hours_bracket_total"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
