"""U1c looks, amendment 1 (docs/U1C_RETRAIN_ON_U.md): look_u1c with U1d's mention relation given no text.

U1d keeps every KB triple on metaqa and webqsp and adds its mention links as one new relation id after the KB vocabulary
(metaqa 9, webqsp 7,058; outputs/u1d/<D>/build.json, mention_rel_id). The relation table has no row for that id, so
look_u1c stops at the compile's relcos[erel] (IndexError). The amendment gives the mention relation no text:

    relation table   one zero row appended for the mention id: cos(q, e_r) = 0 for a mention entry (the fill an untyped
                     pair already gets), the ordered channel's adjacent-relation cosine 0 (the 1e-12 guard). Its ief is
                     M3B's formula on structural_U's count, log((1 + n) / (1 + count)); the KB rows' ief must equal
                     today's table bit for bit (structural_U keeps every triple), or the look refuses.
    relation slots   a mention entry takes no bank row: universal_v2_models.relation_slots sees only the entries of
                     KB relations (a pair's mention entry sorts last, so the KB slots are unchanged); a pair with only a
                     mention link has every slot -1, as an untyped pair has, and rmatch's and zrc's chains skip it.

The topology (pools, STRUCT columns, reach, depth, message edges) keeps every mention link. On a dataset without a
relation table (the four passage datasets) the amendment changes nothing, and look_u1c's looks stand.

    python outputs/mp_unified/look_u1c2.py --dataset metaqa --carve s1eval --host --full [--shard i/n]

Output: look_u1c's (outputs/u1c/look/<dataset>/<carve>) plus amend1<tag>.json beside each shard's pools<tag>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import look_u1c as LU  # noqa: E402

from mp_retrieval import universal_v2_models as UV2M  # noqa: E402  (look_u1c's imports put src on the path)
from mp_retrieval.m3b_features import RelationTable  # noqa: E402

KB = ("metaqa", "webqsp")
_ORIG_SLOTS = UV2M.relation_slots
_STATE = {"n_kb": None}


def extend_table(rt, counts, n_nodes, mention_id):
    """The KB table with one zero row for the mention id; the KB rows' ief unchanged bit for bit."""
    emb = np.asarray(rt.embeddings, dtype=np.float32)
    n_kb = emb.shape[0]
    if mention_id != n_kb or counts.shape[0] != n_kb + 1:
        raise SystemExit(f"mention id {mention_id}, {counts.shape[0]} counts: not one new id after {n_kb} KB relations")
    new = RelationTable.from_arrays(np.vstack([emb, np.zeros((1, emb.shape[1]), np.float32)]), counts, n_nodes)
    if not np.array_equal(new.ief[:n_kb], np.asarray(rt.ief, dtype=np.float32)):
        raise SystemExit("the KB relations' ief differs on structural_U: not every triple kept")
    return new


def kb_slots(pair_id, erel, n_pairs, offset):
    """relation_slots over the KB relations' entries only (mention entries take no bank row)."""
    n_kb = _STATE["n_kb"]
    if n_kb is None:
        return _ORIG_SLOTS(pair_id, erel, n_pairs, offset)
    keep = np.asarray(erel) < n_kb
    return _ORIG_SLOTS(np.asarray(pair_id)[keep], np.asarray(erel)[keep], n_pairs, offset)


def install(name, carve, filed, manifest, stats):
    """look_u1c.install, then the context's relation table extended and relation_slots restricted."""
    _LU_INSTALL(name, carve, filed, manifest, stats)
    lu_open = LU.LX.S6.pair_open

    def pair_open(*args, **kwargs):
        op = lu_open(*args, **kwargs)
        ctx = op.contexts[name]
        if ctx.rel_table is None:
            stats["amend1"] = {"note": "no relation table: unchanged"}
            return op
        b = json.loads((LU.U1D / name / "build.json").read_text(encoding="utf-8"))
        st = ctx.stores["structural"]
        rt = ctx.rel_table
        n_kb = int(np.asarray(rt.embeddings).shape[0])
        ctx.rel_table = extend_table(rt, np.asarray(st.rel_count), int(op.handles[name].n_nodes), int(b["mention_rel_id"]))
        _STATE["n_kb"] = n_kb
        UV2M.relation_slots = kb_slots
        stats["amend1"] = {"kb_relations": n_kb, "mention_rel_id": int(b["mention_rel_id"]),
                           "mention_edges": int(st.rel_count[n_kb]), "mention_ief": float(ctx.rel_table.ief[n_kb])}
        return op

    LU.LX.S6.pair_open = pair_open


_LU_INSTALL = LU.install


def selftest():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((5, 8)).astype(np.float32)
    counts = np.array([3, 1, 4, 1, 5], np.int64)
    rt = RelationTable.from_arrays(emb, counts, 100)
    new = extend_table(rt, np.r_[counts, 50], 100, 5)
    q = rng.standard_normal(8).astype(np.float32)
    rc = new.embeddings @ q
    assert rc[5] == 0.0 and np.array_equal(rc[:5], rt.embeddings @ q) and np.array_equal(new.ief[:5], rt.ief)
    assert np.isclose(new.ief[5], np.log(101 / 51))
    for bad in ((np.r_[counts, 50], 6), (counts, 5), (np.r_[counts[:4], 9, 50], 5)):
        try:
            extend_table(rt, bad[0], 100, bad[1])
        except SystemExit:
            continue
        raise AssertionError(f"accepted {bad}")
    # slots: entries sorted by (owner, neighbour, rel); the mention id (5) sorts last in a pair
    pair_id = np.array([0, 0, 1, 2, 2, 2, 2, 2, 3])
    erel = np.array([1, 5, 5, 0, 1, 2, 3, 5, 4])
    _STATE["n_kb"] = None
    ref = _ORIG_SLOTS(pair_id[erel < 5], erel[erel < 5], 4, 10)
    _STATE["n_kb"] = 5
    got = kb_slots(pair_id, erel, 4, 10)
    assert np.array_equal(got[0], ref[0]) and got[1] == ref[1]
    s = got[0]
    assert (s[1] == -1).all() and s[0, 0] == 11 and (s[0, 1:] == -1).all() and s[3, 0] == 14
    assert (s[s >= 0] < 10 + 5).all()
    # without the mention entries the slots are relation_slots' own
    keep = erel < 5
    assert np.array_equal(kb_slots(pair_id[keep], erel[keep], 4, 10)[0], _ORIG_SLOTS(pair_id[keep], erel[keep], 4, 10)[0])
    _STATE["n_kb"] = None
    print("look_u1c2 selftest: zero row, ief bit for bit, refusals, KB-only slots: ok")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return selftest()
    name = argv[argv.index("--dataset") + 1]
    if name not in KB:
        raise SystemExit(f"{name}: no relation table; look_u1c's look stands")
    LU.install = install
    rc = LU.main(argv)
    shard = argv[argv.index("--shard") + 1] if "--shard" in argv else None
    sh = LU.LX.L8.parse_shard(shard)
    tag = f"_{sh[0]}of{sh[1]}" if sh else ""
    carve = argv[argv.index("--carve") + 1]
    out = LU.LOOK / name / carve
    rec = json.loads((out / f"pools{tag}.json").read_text(encoding="utf-8"))
    if "amend1" not in rec:
        raise SystemExit("the amendment never ran")
    rec2 = {"look": "look_u1c2", "amendment": 1, "declared_in": "docs/U1C_RETRAIN_ON_U.md", "dataset": name,
            "carve": carve, "shard": list(sh) if sh else None, "amend1": rec["amend1"],
            "script_sha256": {"look_u1c2": LU.sha_src(__file__), **rec["script_sha256"]}, "utc": LU.S4E.utc()}
    LU.S4E.write_json(out / f"amend1{tag}.json", rec2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
