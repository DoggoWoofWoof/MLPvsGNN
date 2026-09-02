#!/usr/bin/env python
"""Render the Phase -1 summary as the markdown tables the results doc carries.

``analyze_graph_substrate.py`` decides what the audit measured; this only
decides how it is laid out. Keeping the two apart means the results document
can be regenerated when the outstanding datasets land without anyone retyping a
number, and it means a reviewer can diff the rendered tables against the
summary rather than trusting prose.

Two labelling rules are enforced here, because getting them wrong is how a
report starts lying:

*   A column whose underlying field is a per-query **count** says so in its
    header. ``globally_connected_but_induced_disconnected`` is a count of gold
    targets, not a rate; the conditional rate a reader usually wants is
    ``bridge_loss_at_h``, which is reported separately and is a fraction.
*   The two retention aggregation levels get separate columns with their level
    named, never a single "retention" column. They have different denominators
    and they disagree.

Only ``complete`` audits are rendered. A partial audit is missing whole graph
families, and a table that silently omits them reads as though the dataset had
fewer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO_ROOT / "outputs" / "graph_substrate_audit" / "summary.json"

GRAPH_ORDER = ("dataset_default", "structural_only", "knn_only", "baseline_a_simple")
SHORT = {
    "dataset_default": "sealed A",
    "structural_only": "structural",
    "knn_only": "kNN only",
    "baseline_a_simple": "baseline A",
}
HOPS = (1, 2, 3)


def fmt(value: Any, places: int = 3) -> str:
    """A missing metric prints as a dash, never as zero."""

    if value is None:
        return "--"
    return format(float(value), "." + str(places) + "f")


def split_of(audit: dict, graph: str, split: str) -> dict | None:
    entry = audit.get("graphs", {}).get(graph)
    if not entry:
        return None
    return entry.get("splits", {}).get(split)


def table(title: str, header: list[str], rows: list[list[str]]) -> str:
    lines = ["", "### " + title, ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def rows_for(
    audits: list[dict], split: str, cells: Callable[[dict], list[str]]
) -> list[list[str]]:
    out = []
    for audit in audits:
        for graph in GRAPH_ORDER:
            payload = split_of(audit, graph, split)
            if payload is not None:
                out.append([audit["dataset"], SHORT[graph]] + cells(payload))
    return out


def connectivity(sp: dict) -> list[str]:
    c = sp["connectivity"]
    return [
        fmt(c.get("candidates"), 1),
        fmt(c.get("edges_directed_non_self"), 1),
        fmt(c.get("isolated_fraction")),
        fmt(c.get("degree_1_fraction")),
        fmt(c.get("degree_ge_2_fraction")),
    ]


def retention(sp: dict) -> list[str]:
    node = sp["retention"]["node_level_pooled_over_candidates"]
    query = sp["retention"]["query_level_mean_across_queries"]
    return [
        fmt(node.get("retention_mean")),
        fmt(node.get("retention_median")),
        fmt(node.get("global_degree_median"), 1),
        fmt(query.get("retention_zero_fraction")),
        fmt(query.get("retention_below_10pct_fraction")),
        fmt(query.get("retention_below_25pct_fraction")),
        fmt(query.get("boundary_cut_ratio")),
    ]


def receptive_field(sp: dict) -> list[str]:
    field = sp["receptive_field"]
    sym, flow = field["symmetrised"], field["message_flow"]
    coincide = field.get("notions_coincide")
    return (
        [fmt(sym.get("R" + str(h) + "_median"), 2) for h in HOPS]
        + [fmt(flow.get("R" + str(h) + "_median"), 2) for h in HOPS]
        + [
            "--" if coincide is None else ("yes" if coincide else "NO"),
            fmt(sym.get("R1_zero_fraction")),
        ]
    )


def message_load(sp: dict) -> list[str]:
    m = sp["operator_message_load"]
    return [
        fmt(m.get("unique_non_self_edges"), 1),
        fmt(m.get("stored_non_self_messages"), 1),
        fmt(m.get("duplicate_message_fraction")),
        # Protocol 4.4 lists stored and operator-inserted self-loops as separate
        # reported quantities, and 4.2 flags the case where they coexist: `gcn`
        # and `gat` insert their own, so a stored self-loop would be consumed
        # twice. Rendering only the inserted count would leave that unanswered.
        fmt(m.get("stored_self_loops"), 1),
        fmt(m.get("operator_inserted_self_loops"), 1),
        fmt(m.get("messages_consumed_by_operator"), 1),
    ]


def seed_reach(sp: dict) -> list[str]:
    reach = sp["seed_reachability"]
    induced, glob = reach["induced_symmetrised"], reach["global"]
    return [fmt(induced.get("reachable_at_" + str(h))) for h in HOPS] + [
        fmt(glob.get("reachable_at_" + str(h))) for h in HOPS
    ]


def paths(sp: dict) -> list[str]:
    p = sp["path_preservation"]["gold_path_preservation"]
    b = sp["path_preservation"]["gold_bridge_loss"]
    return [
        fmt(p.get("targets"), 2),
        fmt(p.get("connected_globally_fraction")),
        fmt(p.get("connected_induced_fraction")),
        fmt(p.get("globally_connected_but_induced_disconnected")),
        fmt(p.get("distance_inflated_fraction")),
    ] + [fmt(b.get("bridge_loss_at_" + str(h))) for h in HOPS]


def expansion(sp: dict) -> list[str]:
    headroom = sp["expansion_headroom"]
    s = headroom["symmetric"]
    return (
        [str(headroom.get("queries_measured", "--")), fmt(s.get("candidates"), 1)]
        + [fmt(s.get("U_seed_" + str(h) + "_expansion"), 2) for h in HOPS]
        + [fmt(s.get("U_target_" + str(h) + "_expansion"), 2) for h in HOPS]
    )


def orientation(audits: list[dict]) -> str:
    """How the frozen artifact stores its edges -- protocol 1.3, questions 1 and 2.

    Split-independent, because orientation is a property of the stored graph
    rather than of a query set. The ratio is the informative column: a symmetric
    graph with no duplicate edges stores exactly two directed edges per
    undirected one, so anything above 2.0 is stored multiplicity.
    """

    rows = []
    for audit in audits:
        for graph in GRAPH_ORDER:
            entry = audit.get("graphs", {}).get(graph)
            if not entry:
                continue
            stored = entry.get("stored_directed_edges")
            undirected = entry.get("undirected_edges")
            symmetric = entry.get("stored_graph_was_symmetric")
            ratio = (
                fmt(stored / undirected, 4)
                if stored is not None and undirected
                else "--"
            )
            rows.append([
                audit["dataset"],
                SHORT[graph],
                "--" if stored is None else str(stored),
                "--" if undirected is None else str(undirected),
                "--" if symmetric is None else ("yes" if symmetric else "no"),
                ratio,
            ])
    return table(
        "Storage orientation and multiplicity -- a property of the artifact",
        ["dataset", "graph", "stored directed", "undirected", "stored symmetric",
         "stored / undirected"],
        rows,
    )


def aliasing(audits: list[dict], split: str) -> str:
    """The one axis an aliased pair differs on, next to the fact they are aliased.

    Two families that hash to the same undirected edge key have identical
    structure everywhere else in this report, so the message-load column is
    what a reader needs in order to attribute any downstream difference.
    """

    rows = []
    for audit in audits:
        groups = (audit.get("provenance_aliases") or {}).get("aliased_groups", [])
        for group in groups:
            families = group["families"]
            rows.append([
                audit["dataset"],
                " + ".join("`" + name + "`" for name in families),
                "`" + group["undirected_edge_key_sha256"][:16] + "`",
                " / ".join(str(group["stored_directed_edges"][n]) for n in families),
                " / ".join(
                    "yes" if group["stored_graph_was_symmetric"][n] else "no"
                    for n in families
                ),
                " / ".join(
                    fmt(
                        group["operator_message_load"][n]
                        .get(split, {})
                        .get("messages_consumed_by_operator"),
                        1,
                    )
                    for n in families
                ),
            ])
    return table(
        "Provenance aliasing -- families that are the same undirected graph",
        [
            "dataset",
            "families",
            "undirected key",
            "stored directed edges",
            "stored symmetric",
            "messages consumed",
        ],
        rows,
    )


def render(summary: dict, split: str) -> str:
    audits = [a for a in summary.get("audits", []) if a.get("complete")]
    if not audits:
        raise SystemExit(
            "No complete audit in the summary; nothing to render. Partial audits "
            "are missing whole graph families, and a table that omits them reads "
            "as though the dataset had fewer."
        )
    if not any(
        split_of(audit, graph, split) is not None
        for audit in audits
        for graph in GRAPH_ORDER
    ):
        # Otherwise every table renders as a bare header, which reads like "we
        # measured this split and it was empty" rather than "this split was
        # never audited". The audit runs on the splits named in its config; a
        # typo here must not look like a finding.
        available = sorted(
            name
            for audit in audits
            for graph in GRAPH_ORDER
            for name in (audit.get("graphs", {}).get(graph, {}).get("splits") or {})
        )
        raise SystemExit(
            f"No complete audit recorded split {split!r}; nothing to render. "
            f"Audited splits: {sorted(set(available))}."
        )
    parts = [
        table(
            "Candidate-induced connectivity (" + split + " split)",
            ["dataset", "graph", "mean cand", "mean directed edges", "isolated",
             "degree 1", "degree 2+"],
            rows_for(audits, split, connectivity),
        ),
        table(
            "Global-neighbourhood retention",
            ["dataset", "graph", "mean (node-pooled)", "median (node-pooled)",
             "median global degree", "ret = 0 (query-mean)", "ret < 10%",
             "ret < 25%", "boundary cut"],
            rows_for(audits, split, retention),
        ),
        table(
            "Effective receptive field -- the two notions, kept apart",
            ["dataset", "graph", "sym R1", "sym R2", "sym R3", "flow R1", "flow R2",
             "flow R3", "coincide", "zero fraction"],
            rows_for(audits, split, receptive_field),
        ),
        table(
            "Operator message load",
            ["dataset", "graph", "unique edges", "stored messages",
             "duplicate fraction", "stored self-loops", "operator self-loops",
             "messages consumed"],
            rows_for(audits, split, message_load),
        ),
        table(
            "Seed reachability -- induced versus global",
            ["dataset", "graph", "induced @1", "induced @2", "induced @3",
             "global @1", "global @2", "global @3"],
            rows_for(audits, split, seed_reach),
        ),
        table(
            "Gold path preservation and bridge loss",
            ["dataset", "graph", "targets (count/query)", "connected globally",
             "connected induced", "lost by induction (count/query)",
             "distance inflated", "bridge loss @1", "@2", "@3"],
            rows_for(audits, split, paths),
        ),
        table(
            "Graph-expansion headroom -- ORACLE ONLY, admits nothing to any pool",
            ["dataset", "graph", "n", "mean cand", "U_seed H=1", "H=2", "H=3",
             "U_target H=1", "H=2", "H=3"],
            rows_for(audits, split, expansion),
        ),
        orientation(audits),
        aliasing(audits, split),
    ]
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    text = render(summary, args.split)
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
