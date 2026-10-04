"""Design look (untracked; not a result and not filed): anchor-typed walks, round 10: the node gate on the anchor bases.

A type's bonus is the same for every node it reaches (p(tau | q) / |R_tau|^beta), so a reach set that holds the right
node and wrong ones lifts them alike. On 2wiki two thirds of the full_coverage@5 the best variant still leaves to the
GNN is on bridge_comparison rows (four golds, two of them one edge from the two compared entities), and on hotpotqa the
GNN's whole edge is hit@1, which a uniform bonus cannot reach without displacing the twin's right rank-1 node. The
node gate of l16_look_gate.py / l16_look_gate2.py scales each reached node's bonus by 2 sigmoid(g), g a bilinear match
of the query, the walk's type and the reached node's own fixed 128-d projection (and, for gatez, its own twin score):
    gate    l16_look_gate2's 'typed' (l16_look_gate.Gate(typed=True))
    gatek   'typed-qk': the gate and a query-conditioned kappa
    gatez   l16_look_gate2.GateZ: gatek plus w_z z(v), the reached node's own twin score, w_z starting at 0
The gate reads only the reached node's own attributes, never a neighbour's (no message), and starts at 2 sigmoid(0) = 1,
so each fit starts as lin's. Everything else is anchor_walk8.py's main, imported unchanged (the loss with or without the
GNN teacher, the rules, the selection, the reads): this file maps the three names onto l16_look_gate2.make_for and
anchor_walk7's parser. Its record is anchor_walk8's with this file's sha and pins added.

    <train>:<base>/<gate>[/<rule>][@seeds]       base T0 or A{256,1024}-1 or A256
    python outputs/mp_approx_2wiki_anchor/host/anchor_walk10.py --dataset 2wiki --variants x4+x5+x6:A256-1/gatez --kd 1
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_walk8 as AW8  # noqa: E402

AW7, G2 = AW8.AW7, AW8.G2
AW8_SHA = "7e4aa326324f44b3bce7ecee2afd7e79b09c74fda8fa692509f9adb5063b5768"
GATES = {"gate": "typed", "gatek": "typed-qk", "gatez": "typed-qk-z"}
_parse7, _make7 = AW7.parse, AW7.make_for


def parse(name, train_looks):
    head, at, seeds = name.partition("@")
    parts = head.split("/")
    if len(parts) >= 2 and parts[1] in GATES:
        g = parts[1]
        parts[1] = "lin"
        sp = _parse7("/".join(parts) + (at + seeds if at else ""), train_looks)
        sp["model"] = g
        return sp
    return _parse7(name, train_looks)


def make_for(model, ntp, K, phi):
    if model in GATES:
        return G2.make_for(GATES[model], False, ntp)
    return _make7(model, ntp, K, phi)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if AW8.AW.sha(Path(AW8.__file__)) != AW8_SHA:
        raise SystemExit("anchor_walk8.py is not the pinned file")
    ds = argv[argv.index("--dataset") + 1] if "--dataset" in argv else "2wiki"
    if "--out" not in argv:
        argv += ["--out", str(HERE / f"anchor_walk10_{ds}.json")]
    out = Path(argv[argv.index("--out") + 1])
    AW7.parse, AW7.make_for = parse, make_for
    try:
        AW8.main(argv)
    finally:
        AW7.parse, AW7.make_for = _parse7, _make7
    res = json.loads(out.read_text(encoding="utf-8"))
    res["look"] = "anchor_walk10"
    res["pins"]["anchor_walk8"] = AW8_SHA
    res["pins"]["anchor_walk8_record_sha256_field"] = res["script_sha256"]
    res["script_sha256"] = AW8.AW.sha(Path(__file__))
    res["gates"] = GATES
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
