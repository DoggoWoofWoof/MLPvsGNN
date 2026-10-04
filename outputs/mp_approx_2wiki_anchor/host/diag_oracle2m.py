"""Design look (untracked; not a result and not filed): diag_oracle2.py on musique with anchor_walk11.py's frontier-pruned
look loader (edges out of the seeds and out of the nodes one edge from them; no projections), since musique's full rows
cost about 3 MB each in load's form. The walk-type schemes read exactly what they read on full rows (anchor_walk11 checks
the walk types on the first chunk of every look); the PPR schemes here diffuse over that frontier subgraph only (mass
reaches nodes up to two edges from the seeds and stops there), which the record names as ppr_graph.

    python outputs/mp_approx_2wiki_anchor/host/diag_oracle2m.py --dataset musique [--schemes ...] [--out PATH]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_walk11 as AW11  # noqa: E402
import diag_oracle2 as D  # noqa: E402

AW11_SHA = "e782089f33cae0016ad4bbf09f14e092b0da7eaf31932c5d437d42f274d0f99b"
D_SHA = "3ae30324891d8ae3844f17f17b9787077969301571134558a9b8aff4b97fd204"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if D.AW.sha(Path(AW11.__file__)) != AW11_SHA or D.AW.sha(Path(D.__file__)) != D_SHA:
        raise SystemExit("anchor_walk11.py or diag_oracle2.py is not the pinned file")
    if D.A16 is not AW11.A16:
        raise SystemExit("the two looks do not share l16_look_analyze")
    ds = argv[argv.index("--dataset") + 1] if "--dataset" in argv else "musique"
    if "--out" not in argv:
        argv += ["--out", str(HERE / f"diag_oracle2m_{ds}.json")]
    out = Path(argv[argv.index("--out") + 1])
    AW11.STATE.update({"max_len": 2, "proj": False, "rows": 0, "edges_in": 0, "edges_kept": 0, "checked_rows": 0})
    D.A16.load = AW11.load_pruned
    try:
        D.main(argv)
    finally:
        D.A16.load = AW11._load
    res = json.loads(out.read_text(encoding="utf-8"))
    res["look"] = "diag_oracle2m"
    res["pins"]["diag_oracle2"] = D_SHA
    res["pins"]["anchor_walk11"] = AW11_SHA
    res["pins"]["diag_oracle2_record_sha256_field"] = res["script_sha256"]
    res["script_sha256"] = D.AW.sha(Path(__file__))
    res["ppr_graph"] = "frontier: the edges out of the seeds and out of the nodes one edge from them"
    res["pruned_loader"] = {k: AW11.STATE[k] for k in ("max_len", "proj", "rows", "edges_in", "edges_kept", "checked_rows")}
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
