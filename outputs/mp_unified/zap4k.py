"""AP4's open item (docs/AP4_LABEL_PRECISION_LEDGER.md, section 10): on 5 of 9 paired reads D83's K pairs at the anchors
are a strict subset of the record's. This reads the two filed ledgers of each pair (outputs/ap4/ledger/<arm>/<ds>.npz,
checked against their records' sha256 by zap4.load_ledger) and says where the missing pairs sit:

- in D83's ledger at all (another label holds the anchor-candidate pair) or not at all (no D83 link joins the two rows);
- the labels D83 and the record give them;
- how they spread over questions, and the pool sizes of the questions that hold them.

Reads only; writes outputs/ap4/kdiag.json. No measure of AP4 changes. Run on the host (the ledgers' npz live there):
    python outputs/mp_unified/zap4k.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import zap4 as A4  # noqa: E402  (load_ledger, units, ARMS, PAIR; read only)

OUT = A4.OUT / "kdiag.json"


def keys(z, big):
    return ((z["P_q"].astype(np.int64) * 2 + z["P_kind"]) * big + z["P_a"]) * big + z["P_c"]


def names_of(bits, names):
    return "+".join(nm for i, nm in enumerate(names) if (int(bits) >> i) & 1) or "-"


def one(arm, ds):
    rec_arm = A4.PAIR[arm]
    la, lb = A4.load_ledger(arm, ds), A4.load_ledger(rec_arm, ds)
    if la is None or lb is None:
        return None
    (za, _), (zb, _) = la, lb
    na, nb = A4.ARMS[arm][2], A4.ARMS[rec_arm][2]
    ka_m, kb_m = A4.units(arm, ds)["group:K"], A4.units(rec_arm, ds)["group:K"]
    big = int(max(za["Q_n"].max(), zb["Q_n"].max())) + 1
    kya, kyb = keys(za, big), keys(zb, big)
    sa = (za["P_bits"].astype(np.int64) & ka_m) != 0
    sb = (zb["P_bits"].astype(np.int64) & kb_m) != 0
    ka, kb = np.unique(kya[sa]), np.unique(kyb[sb])
    miss = np.setdiff1d(kb, ka)
    extra = np.setdiff1d(ka, kb)
    out = {"arm": arm, "record_arm": rec_arm, "dataset": ds, "k_pairs_d83": int(ka.size), "k_pairs_record": int(kb.size),
           "missing_in_d83": int(miss.size), "extra_in_d83": int(extra.size)}
    if miss.size:
        # the record's bits on the missing pairs (OR over duplicates), and D83's bits on the same anchor-candidate pairs
        ob = np.argsort(kyb, kind="stable")
        ib = np.searchsorted(kyb[ob], miss)
        rb_bits = np.zeros(miss.size, np.int64)
        for j, (lo, k) in enumerate(zip(ib, miss)):
            hi = lo
            while hi < ob.size and kyb[ob[hi]] == k:
                rb_bits[j] |= int(zb["P_bits"][ob[hi]])
                hi += 1
        oa = np.argsort(kya, kind="stable")
        ia = np.searchsorted(kya[oa], miss)
        da_bits = np.zeros(miss.size, np.int64)
        present = np.zeros(miss.size, bool)
        for j, (lo, k) in enumerate(zip(ia, miss)):
            hi = lo
            while hi < oa.size and kya[oa[hi]] == k:
                da_bits[j] |= int(za["P_bits"][oa[hi]])
                present[j] = True
                hi += 1
        q = miss // (2 * big * big)
        qn = za["Q_n"][q]
        kinds = (miss // (big * big)) % 2
        out.update({
            "present_in_d83_with_other_labels": int(present.sum()),
            "absent_from_d83": int((~present).sum()),
            "record_labels": dict(zip(*[x.tolist() for x in np.unique([names_of(b, nb) for b in rb_bits], return_counts=True)])),
            "d83_labels_when_present": dict(zip(*[x.tolist() for x in np.unique([names_of(b, na) for b in da_bits[present]] or ["-"],
                                                                                 return_counts=True)])) if present.any() else {},
            "by_kind": {"found": int((kinds == 0).sum()), "top1": int((kinds == 1).sum())},
            "questions_holding_them": int(np.unique(q).size),
            "most_in_one_question": int(np.bincount(q).max()),
            "pool_size_of_those_questions": {"median": float(np.median(qn)), "min": int(qn.min()), "max": int(qn.max())},
            "pool_size_all_questions": {"median": float(np.median(za["Q_n"])), "max": int(za["Q_n"].max())},
        })
    return out


def main():
    res = []
    for arm in A4.PAIR:
        for ds in A4.ARMS[arm][3]:
            r = one(arm, ds)
            if r is not None:
                res.append(r)
                print(json.dumps(r))
    OUT.write_text(json.dumps({"stage": "AP4 open item (section 10)", "reads": res}, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
