"""SWP Part D (docs/SWP_S_TO_R_SWAP.md section 5b): the oracle multi-hop pairs of the B1 MuSiQue setting. A diagnostic upper bound, never a method: the
pairs are built from the gold supporting paragraphs and MuSiQue's own question decomposition.

    ORC (chain): for each question, an edge between the supporting paragraphs of a sub-question and of each earlier sub-question its text refers to
                 ("#1", "#2" in MuSiQue's decomposition: the answers it depends on); a sub-question that refers to none is a branch root and gets no edge,
                 and the build stops unless every question's hops form one connected chain through these dependencies;
    ORA (all gold pairs): an edge between every two supporting paragraphs of a question (ORC is a subset of it).

A hop's paragraph is matched to the setting's node by (title, text) of HippoRAG 2's corpus file (node i is corpus entry i; every (title, text) is unique), and the
set of a question's hop nodes must equal the golds B1a filed in queries.json, for all 1,000 questions, or the build stops.

    python outputs/mp_unified/oraclepairs.py check          # the real files: the match, the pair counts, per question kind
    python outputs/mp_unified/oraclepairs.py selftest
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCH = ROOT / "outputs" / "bench" / "hipporag2"
KINDS = ("ORC", "ORA")
REF_RE = re.compile(r"#(\d+)")


def load(name="musique", bench=BENCH):
    """(raw questions, corpus entries, B1a's nodes, B1a's queries) of the setting, read only."""
    raw = json.loads((Path(bench) / "raw" / f"{name}.json").read_text(encoding="utf-8"))
    corpus = json.loads((Path(bench) / "raw" / f"{name}_corpus.json").read_text(encoding="utf-8"))
    nodes = json.loads((Path(bench) / name / "nodes.json").read_text(encoding="utf-8"))
    queries = json.loads((Path(bench) / name / "queries.json").read_text(encoding="utf-8"))
    return raw, corpus, nodes, queries


def hop_nodes(q, by_text):
    """The question's hop positions in decomposition order, and the dependency pairs (j, i) with i < j between hops."""
    para = {int(p["idx"]): p for p in q["paragraphs"]}
    pos, deps, without = [], [], 0
    for j, sub in enumerate(q["question_decomposition"]):
        p = para[int(sub["paragraph_support_idx"])]
        hit = by_text.get((p["title"], p["paragraph_text"]))
        if hit is None or len(hit) != 1:
            raise SystemExit(f"oracle pairs: {q['id']} hop {j}: the paragraph {p['title']!r} is not exactly one corpus entry ({hit})")
        pos.append(hit[0])
        refs = sorted({int(m) - 1 for m in REF_RE.findall(sub["question"])})
        refs = [r for r in refs if 0 <= r < j]
        if j and not refs:
            without += 1                                         # a branch root (3hop2, 4hop2, 4hop3): it depends on no earlier answer, so it gets no edge
        deps += [(j, r) for r in refs]
    return pos, deps, without


def connected(n, deps):
    """The n hops joined by the dependency pairs form one component."""
    seen, todo = {0}, [0]
    while todo:
        x = todo.pop()
        for j, i in deps:
            for a, b in ((j, i), (i, j)):
                if a == x and b not in seen:
                    seen.add(b)
                    todo.append(b)
    return len(seen) == n


def build(raw, corpus, nodes, queries):
    """{kind: sorted (a < b) pairs over node positions}, per-question rows and the counts; stops unless the hop nodes equal B1a's golds."""
    if len(corpus) != len(nodes) or any(corpus[i]["title"] != nodes[i]["title"] for i in range(len(nodes))):
        raise SystemExit("oracle pairs: the corpus file is not B1a's node order")
    by_text = {}
    for i, c in enumerate(corpus):
        by_text.setdefault((c["title"], c["text"]), []).append(i)
    if len(raw) != len(queries):
        raise SystemExit(f"oracle pairs: {len(raw)} questions in the raw file, {len(queries)} in B1a's")
    pairs = {k: set() for k in KINDS}
    rows, no_ref = [], 0
    for q, g in zip(raw, queries):
        if str(q["id"]) != str(g["id"]):
            raise SystemExit(f"oracle pairs: question order differs at {q['id']} / {g['id']}")
        pos, deps, without = hop_nodes(q, by_text)
        no_ref += without
        if set(pos) != set(int(x) for x in g["golds"]):
            raise SystemExit(f"oracle pairs: {q['id']}: hop nodes {sorted(set(pos))} are not B1a's golds {sorted(g['golds'])}")
        if not connected(len(pos), deps):
            raise SystemExit(f"oracle pairs: {q['id']}: the hops' dependencies do not form one connected chain ({deps})")
        chain = {tuple(sorted((pos[j], pos[i]))) for j, i in deps if pos[j] != pos[i]}
        allp = {(a, b) for a in set(pos) for b in set(pos) if a < b}
        assert chain <= allp
        pairs["ORC"] |= chain
        pairs["ORA"] |= allp
        rows.append({"id": q["id"], "kind": g["kind"], "hops": len(pos), "orc": len(chain), "ora": len(allp)})
    out = {k: sorted(v) for k, v in pairs.items()}
    return out, rows, {"questions": len(rows), "hops_without_reference": no_ref, "pairs": {k: len(v) for k, v in out.items()}}


def by_kind(rows):
    agg = {}
    for r in rows:
        a = agg.setdefault(r["kind"], {"questions": 0, "orc": 0, "ora": 0})
        a["questions"] += 1
        a["orc"] += r["orc"]
        a["ora"] += r["ora"]
    return agg


def check(name="musique"):
    raw, corpus, nodes, queries = load(name)
    out, rows, meta = build(raw, corpus, nodes, queries)
    print(json.dumps(meta))
    for k, a in sorted(by_kind(rows).items()):
        print(f"  {k:6s} {a['questions']:4d} questions  ORC {a['orc']:5d} pair-slots ({a['orc'] / a['questions']:.2f} per question)  "
              f"ORA {a['ora']:5d} ({a['ora'] / a['questions']:.2f})")
    return 0


def selftest():
    corpus = [{"title": f"T{i}", "text": f"x{i}"} for i in range(8)]
    nodes = [{"title": c["title"]} for c in corpus]

    def q(id_, subs, golds):
        para = [{"idx": str(i), "title": corpus[i]["title"], "paragraph_text": corpus[i]["text"]} for i in range(8)]
        return ({"id": id_, "paragraphs": para,
                 "question_decomposition": [{"question": s, "paragraph_support_idx": p} for s, p in subs]},
                {"id": id_, "golds": golds, "kind": f"{len(subs)}hop"})

    a, ga = q("a", [("who", 1), ("when was #1 born", 2)], [1, 2])                                       # 2 hops
    b, gb = q("b", [("who", 3), ("where", 4), ("what links #1 and #2", 5)], [3, 4, 5])                  # 3 hops, the last depends on both
    c, gc = q("c", [("who", 1), ("when", 6), ("how #1 and #2", 7)], [1, 6, 7])                          # two branch roots joined by the last hop
    d, gd = q("d", [("first", 0), ("second of #1", 0)], [0])                                            # two hops on one paragraph: no self-edge
    out, rows, meta = build([a, b, c, d], corpus, nodes, [ga, gb, gc, gd])
    assert out["ORC"] == [(1, 2), (1, 7), (3, 5), (4, 5), (6, 7)], out["ORC"]
    assert set(out["ORC"]) <= set(out["ORA"])
    assert out["ORA"] == sorted({(1, 2), (3, 4), (3, 5), (4, 5), (1, 6), (1, 7), (6, 7)}), out["ORA"]
    assert meta["hops_without_reference"] == 2 and [r["orc"] for r in rows] == [1, 2, 2, 0] and [r["ora"] for r in rows] == [1, 3, 3, 0]
    lone, glone = q("e", [("who", 1), ("when", 2)], [1, 2])                                             # two roots, not joined: the build stops
    try:
        build([lone], corpus, nodes, [glone])
        raise AssertionError("a disconnected chain must stop the build")
    except SystemExit:
        pass
    bad = dict(ga, golds=[1, 3])
    try:
        build([a], corpus, nodes, [bad])
        raise AssertionError("hop nodes not equal to the golds must stop the build")
    except SystemExit:
        pass
    dup = [dict(corpus[0]), dict(corpus[0])] + corpus[2:]
    try:
        build([a], dup, [{"title": c["title"]} for c in dup], [ga])
        raise AssertionError("an ambiguous paragraph must stop the build")
    except SystemExit:
        pass
    print("oraclepairs self-test: OK")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        raise SystemExit(selftest())
    if cmd == "check":
        raise SystemExit(check())
    raise SystemExit(__doc__)
