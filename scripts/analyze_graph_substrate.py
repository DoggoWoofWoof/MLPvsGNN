#!/usr/bin/env python
"""Turn the Phase -1 substrate audits into the report the protocol asks for.

The audit measures; this reads what it measured and lays it out. It computes
nothing new and decides nothing: no threshold is applied, no verdict is issued,
and no metric from one connectivity notion is mixed with the other. Where the
protocol says a number must be reported alongside how many queries could report
it, that denominator is carried through rather than dropped.

Sections follow ``docs/GRAPH_SUBSTRATE_AUDIT_PROTOCOL.md``:

    2  candidate-induced connectivity
    3  global-neighbourhood retention
    4  effective receptive field, on both notions, plus operator message load
    5  seed reachability, induced versus global
    6  gold path preservation and bridge loss
    8  graph-expansion headroom (oracle-only; admits nothing to any pool)

Two rules from the protocol shape the output:

*   **The two connectivity notions are never merged.** The symmetrised view and
    the exact directed message-flow view are reported side by side, and their
    difference is reported as a difference rather than averaged away.
*   **There is no adequacy threshold.** Nothing here says a substrate is good
    enough or too shallow. It reports distributions and lets them be read.

Inputs are the audit files as ``run_graph_substrate_audit`` wrote them. Fetch
them from the volume with::

    MODAL_PROFILE=<workspace> python scripts/replicate_volume.py download \\
        --slice results --staging <dir>

Then::

    python scripts/analyze_graph_substrate.py --results-root <dir>/outputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

COMPLETE_STATUS = "GRAPH_SUBSTRATE_AUDIT_COMPLETE"
AUDIT_DIR = "graph_substrate_audit"
OUTPUT_DIR = REPO_ROOT / "outputs" / AUDIT_DIR

# Package B provenance families, in the order the protocol discusses them.
GRAPH_ORDER = ("dataset_default", "structural_only", "knn_only", "baseline_a_simple")

# The suffix the audit appends to record how many queries could report a value.
REPORTING = "__queries_reporting"

HOPS = (1, 2, 3)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def discover(root: Path) -> dict[str, Path]:
    """Find one audit file per dataset, under either layout.

    The Modal runner writes ``<root>/graph_substrate_audit/<dataset>/<prefix>/
    substrate.json``; the launcher's blocking fallback writes
    ``<root>/graph_substrate_audit/<dataset>.json``. Both are accepted so a
    report can be produced from whichever fetch happened.
    """

    base = root / AUDIT_DIR if (root / AUDIT_DIR).is_dir() else root
    found: dict[str, Path] = {}
    for path in sorted(base.glob("*/*/substrate.json")):
        found[path.parent.parent.name] = path
    for path in sorted(base.glob("*.json")):
        if path.stem not in found and path.name != "summary.json":
            found[path.stem] = path
    return found


def load_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "graphs" not in payload:
        raise ValueError(f"{path}: not a substrate audit")
    return payload


# ---------------------------------------------------------------------------
# Reading a measured value
# ---------------------------------------------------------------------------


def value(block: dict[str, Any], key: str) -> float | None:
    """One measured value, or None when the audit did not record it."""

    if not isinstance(block, dict):
        return None
    found = block.get(key)
    return float(found) if isinstance(found, (int, float)) else None


def reporting(block: dict[str, Any], key: str) -> int | None:
    """How many queries contributed to ``key``.

    A mean over 32 queries and a mean over 97,852 are not the same claim, and
    the audit records which it is. Dropping that here would quietly upgrade a
    capped measurement into a whole-split one.
    """

    found = value(block, key + REPORTING)
    return int(found) if found is not None else None


def row(block: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        measured = value(block, key)
        if measured is None:
            continue
        out[key] = measured
        denominator = reporting(block, key)
        if denominator is not None:
            out[key + REPORTING] = denominator
    return out


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def connectivity(split: dict[str, Any]) -> dict[str, Any]:
    """Section 2. What the induced subgraph looks like, per query."""

    block = split.get("query_level", {}).get("connectivity", {})
    return row(
        block,
        (
            "candidates",
            "edges_directed_non_self",
            "edges_undirected_non_self",
            "self_loops",
            "isolated_fraction",
            "degree_1_fraction",
            "degree_ge_2_fraction",
        ),
    )


def retention(split: dict[str, Any]) -> dict[str, Any]:
    """Section 3. How much of a candidate's true neighbourhood survives induction.

    Both aggregation levels are carried, and they are kept apart: the node-level
    figure is pooled over individual candidates, which is the level at which
    retention is defined, while the query-level figure is a mean across queries
    of a per-query summary. Reporting one as the other would misstate what was
    averaged.
    """

    node_level = split.get("node_level", {})
    query_level = split.get("query_level", {}).get("retention", {})
    distribution = row(
        node_level,
        (
            "retention_mean",
            "retention_p10",
            "retention_p25",
            "retention_median",
            "retention_p75",
            "retention_p90",
            "retention_p95",
            "retention_max",
            "global_degree_mean",
            "global_degree_p10",
            "global_degree_p25",
            "global_degree_median",
            "global_degree_p75",
            "global_degree_p90",
            "global_degree_p95",
        ),
    )
    pooled = value(node_level, "pooled_from_queries")
    if pooled is not None:
        distribution["pooled_from_queries"] = int(pooled)
    return {
        "node_level_pooled_over_candidates": distribution,
        "query_level_mean_across_queries": row(
            query_level,
            (
                "candidates_with_global_neighbors",
                "retention_zero_fraction",
                "retention_below_10pct_fraction",
                "retention_below_25pct_fraction",
                "boundary_cut_ratio",
                "retention_mean",
                "retention_p10",
                "retention_median",
                "retention_p90",
            ),
        ),
    }


def receptive_field(split: dict[str, Any]) -> dict[str, Any]:
    """Section 4. Reach at each hop, on both notions, never merged.

    ``symmetrised`` treats an edge as usable in either direction; ``message_flow``
    follows the direction the operator actually propagates along. They coincide
    on a stored-symmetric graph and diverge on a directed one, and that
    divergence is the point, so it is reported rather than reconciled.
    """

    query_level = split.get("query_level", {})
    symmetric = query_level.get("receptive_field", {})
    flow = query_level.get("message_flow_receptive_field", {})
    keys = tuple(
        f"R{hop}_{statistic}"
        for hop in HOPS
        for statistic in ("median", "mean", "zero_fraction")
    )
    flow_keys = tuple("flow_" + key for key in keys)
    symmetric_row = row(symmetric, keys)
    flow_row = {
        key.removeprefix("flow_"): measured
        for key, measured in row(flow, flow_keys).items()
    }
    divergence = {
        key: flow_row[key] - symmetric_row[key]
        for key in symmetric_row
        if key in flow_row and not key.endswith(REPORTING)
    }
    return {
        "symmetrised": symmetric_row,
        "message_flow": flow_row,
        "message_flow_minus_symmetrised": divergence,
        "notions_coincide": all(abs(delta) < 1e-12 for delta in divergence.values())
        if divergence
        else None,
    }


def message_load(split: dict[str, Any]) -> dict[str, Any]:
    """Section 4. Unique edges versus messages the operator actually consumes.

    An operator that sums over a duplicated edge list consumes more messages
    than the graph has edges, and three of the four frozen families do. The
    distinction is reported because "how much structure is there" and "how much
    the operator reads" are different questions.
    """

    return row(
        split.get("query_level", {}).get("operator_edge_load", {}),
        (
            "unique_non_self_edges",
            "stored_non_self_messages",
            "duplicate_messages",
            "duplicate_message_fraction",
            "stored_self_loops",
            "operator_inserted_self_loops",
            "messages_consumed_by_operator",
        ),
    )


def seed_reach(split: dict[str, Any]) -> dict[str, Any]:
    """Section 5. What a seed signal reaches, induced versus global.

    The gap between the induced and global columns is structural information
    that candidate induction destroyed. It is computed here as a subtraction of
    two measured fractions, not measured separately.
    """

    query_level = split.get("query_level", {})
    keys = tuple(f"reachable_at_{hop}" for hop in HOPS)
    induced = row(query_level.get("seed_reachability_induced", {}), keys)
    induced_flow = row(
        query_level.get("seed_reachability_induced_message_flow", {}), keys
    )
    global_ = row(query_level.get("seed_reachability_global", {}), keys)
    lost = {
        key: global_[key] - induced[key]
        for key in keys
        if key in induced and key in global_
    }
    return {
        "induced_symmetrised": induced,
        "induced_message_flow": induced_flow,
        "global": global_,
        "global_minus_induced": lost,
    }


def path_preservation(split: dict[str, Any]) -> dict[str, Any]:
    """Section 6. Whether gold-to-gold paths survive induction, and bridge loss."""

    query_level = split.get("query_level", {})
    preservation = row(
        query_level.get("gold_path_preservation", {}),
        (
            "targets",
            "connected_globally_fraction",
            "connected_induced_fraction",
            "globally_connected_but_induced_disconnected",
            "already_disconnected_globally",
            "distance_inflated_fraction",
            "mean_distance_inflation",
        ),
    )
    bridge = row(
        query_level.get("gold_bridge_loss", {}),
        tuple(
            f"{name}_at_{hop}" for hop in HOPS for name in ("bridge_loss", "eligible")
        ),
    )
    return {"gold_path_preservation": preservation, "gold_bridge_loss": bridge}


def expansion(split: dict[str, Any]) -> dict[str, Any]:
    """Section 8. Oracle-only headroom. Admits nothing to any candidate pool.

    ``U_seed(H) = Cq union N_H(Sq)`` and ``U_target(H) = Cq union N_H(Cq)`` are
    counts of what a wider substrate *would* contain. Nothing here changes a
    pool, a hash, or a scoring set; the protocol keeps this question separate
    from the ranking experiments for exactly that reason.
    """

    block = split.get("expansion", {})
    if not block:
        return {}
    keys = tuple(
        f"U_{which}_{hop}_{statistic}"
        for hop in HOPS
        for which in ("seed", "target")
        for statistic in ("nodes", "expansion")
    )
    out: dict[str, Any] = {
        "queries_measured": int(value(block, "queries_measured") or 0),
        "admits_nothing_to_the_pool": True,
    }
    for notion in ("symmetric", "message_flow"):
        measured = row(block.get(notion, {}), ("candidates",) + keys)
        if measured:
            out[notion] = measured
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def summarize_split(split: dict[str, Any]) -> dict[str, Any]:
    return {
        "queries": int(split.get("queries", 0)),
        "queries_without_retrieval_seeds": int(
            split.get("queries_without_retrieval_seeds", 0)
        ),
        "queries_without_gold_in_pool": int(split.get("queries_without_gold_in_pool", 0)),
        "connectivity": connectivity(split),
        "retention": retention(split),
        "receptive_field": receptive_field(split),
        "operator_message_load": message_load(split),
        "seed_reachability": seed_reach(split),
        "path_preservation": path_preservation(split),
        "expansion_headroom": expansion(split),
    }


def provenance_aliases(payload: dict[str, Any]) -> dict[str, Any]:
    """Group the provenance families by the undirected graph they actually are.

    The audit hashes each graph's undirected edge set, so two families that
    carry different names, different stored edge counts and different symmetry
    flags can still be the same graph once direction and duplication are
    removed. When they are, every downstream metric coincides -- not because
    the substrates behave alike, but because there is one substrate wearing two
    labels.

    This is reported rather than left for a reader to notice in two identical
    rows, because it changes what a comparison across families means.
    """

    keys: dict[str, list[str]] = {}
    for name, entry in payload["graphs"].items():
        key = entry.get("undirected_edge_key_sha256")
        if not key:
            continue
        keys.setdefault(str(key), []).append(name)
    groups = [
        {
            "undirected_edge_key_sha256": key,
            "families": sorted(names),
            "stored_directed_edges": {
                name: payload["graphs"][name].get("stored_directed_edges")
                for name in sorted(names)
            },
            "stored_graph_was_symmetric": {
                name: payload["graphs"][name].get("stored_graph_was_symmetric")
                for name in sorted(names)
            },
        }
        for key, names in sorted(keys.items())
        if len(names) > 1
    ]
    return {
        "distinct_undirected_graphs": len(keys),
        "families_audited": sum(len(names) for names in keys.values()),
        "aliased_groups": groups,
    }


def summarize_audit(payload: dict[str, Any]) -> dict[str, Any]:
    contract = payload.get("diagnostic_contract", {})
    graphs: dict[str, Any] = {}
    for name in GRAPH_ORDER:
        entry = payload["graphs"].get(name)
        if entry is None:
            continue
        graphs[name] = {
            "provenance": entry.get("provenance"),
            "stored_directed_edges": entry.get("stored_directed_edges"),
            "undirected_edges": entry.get("undirected_edges"),
            "stored_graph_was_symmetric": entry.get("stored_graph_was_symmetric"),
            "splits": {
                split_name: summarize_split(split)
                for split_name, split in entry.get("splits", {}).items()
            },
        }
    unknown = sorted(set(payload["graphs"]) - set(GRAPH_ORDER))
    return {
        "dataset": payload["dataset"],
        "status": payload.get("status"),
        "complete": payload.get("status") == COMPLETE_STATUS,
        "queries": payload.get("queries"),
        "num_nodes": payload.get("num_nodes"),
        "num_stored_directed_edges": payload.get("num_stored_directed_edges"),
        "operator_kind": contract.get("operator_kind"),
        "operator_edge_semantics": contract.get("operator_edge_semantics"),
        "message_flow": contract.get("message_flow"),
        "max_hops": contract.get("max_hops"),
        "pooled_query_cap": contract.get("pooled_query_cap"),
        "expansion_query_cap": contract.get("expansion_query_cap"),
        "read_only": contract.get("read_only"),
        "graphs": graphs,
        "graphs_not_in_the_reported_order": unknown,
        "provenance_aliases": provenance_aliases(payload),
    }


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def fmt(measured: float | None, places: int = 3) -> str:
    if measured is None:
        return "     -"
    if abs(measured) >= 1000:
        return f"{measured:,.0f}"
    return f"{measured:.{places}f}"


def print_report(summaries: list[dict[str, Any]], split_name: str) -> None:
    incomplete = [s["dataset"] for s in summaries if not s["complete"]]
    print(f"Phase -1 substrate audit -- {len(summaries)} dataset(s), split {split_name!r}")
    if incomplete:
        print(f"  still running (partial numbers below): {', '.join(incomplete)}")
    print()

    print("=== substrate size ===")
    print(f"{'dataset':16s}{'queries':>10s}{'nodes':>12s}{'stored edges':>15s}{'operator':>10s}")
    for summary in summaries:
        print(
            f"{summary['dataset']:16s}{summary['queries'] or 0:>10,}"
            f"{summary['num_nodes'] or 0:>12,}{summary['num_stored_directed_edges'] or 0:>15,}"
            f"{str(summary['operator_kind']):>10s}"
        )
    print()

    for section, title in (
        ("connectivity", "=== 2. candidate-induced connectivity (per query) ==="),
        ("receptive_field", "=== 4. effective receptive field ==="),
        ("seed_reachability", "=== 5. seed reach, induced vs global ==="),
        ("expansion_headroom", "=== 8. expansion headroom (oracle only) ==="),
    ):
        print(title)
        for summary in summaries:
            for graph_name, graph in summary["graphs"].items():
                split = graph["splits"].get(split_name)
                if not split:
                    continue
                label = f"{summary['dataset']}/{graph_name}"
                block = split[section]
                if section == "connectivity":
                    print(
                        f"  {label:38s} candidates {fmt(block.get('candidates'), 1):>9s}"
                        f"  isolated {fmt(block.get('isolated_fraction')):>7s}"
                        f"  deg1 {fmt(block.get('degree_1_fraction')):>7s}"
                        f"  deg>=2 {fmt(block.get('degree_ge_2_fraction')):>7s}"
                    )
                elif section == "receptive_field":
                    sym, flow = block["symmetrised"], block["message_flow"]
                    print(
                        f"  {label:38s} R1 {fmt(sym.get('R1_median'), 2):>7s}"
                        f"  R2 {fmt(sym.get('R2_median'), 2):>8s}"
                        f"  R3 {fmt(sym.get('R3_median'), 2):>9s}"
                        f"   flow R3 {fmt(flow.get('R3_median'), 2):>9s}"
                        f"   coincide {block['notions_coincide']}"
                    )
                elif section == "seed_reachability":
                    ind, glob = block["induced_symmetrised"], block["global"]
                    print(
                        f"  {label:38s} induced @1/@2/@3 "
                        f"{fmt(ind.get('reachable_at_1')):>6s}/{fmt(ind.get('reachable_at_2')):>6s}/"
                        f"{fmt(ind.get('reachable_at_3')):>6s}"
                        f"   global {fmt(glob.get('reachable_at_1')):>6s}/"
                        f"{fmt(glob.get('reachable_at_2')):>6s}/{fmt(glob.get('reachable_at_3')):>6s}"
                    )
                elif section == "expansion_headroom":
                    if not block:
                        continue
                    sym = block.get("symmetric", {})
                    print(
                        f"  {label:38s} n={block['queries_measured']:<5d}"
                        f" U_seed x1/x2/x3 {fmt(sym.get('U_seed_1_expansion'), 2):>8s}/"
                        f"{fmt(sym.get('U_seed_2_expansion'), 2):>9s}/"
                        f"{fmt(sym.get('U_seed_3_expansion'), 2):>10s}"
                        f"  U_target x1 {fmt(sym.get('U_target_1_expansion'), 2):>8s}"
                    )
        print()

    print("=== 3. retention, node-level pooled over candidates ===")
    for summary in summaries:
        for graph_name, graph in summary["graphs"].items():
            split = graph["splits"].get(split_name)
            if not split:
                continue
            node = split["retention"]["node_level_pooled_over_candidates"]
            print(
                f"  {summary['dataset']}/{graph_name:22s}"
                f" mean {fmt(node.get('retention_mean')):>6s}"
                f"  p10 {fmt(node.get('retention_p10')):>6s}"
                f"  median {fmt(node.get('retention_median')):>6s}"
                f"  p90 {fmt(node.get('retention_p90')):>6s}"
                f"  (pooled from {node.get('pooled_from_queries', 0):,} queries)"
            )
    print()

    print("=== 4. operator message load (per query) ===")
    for summary in summaries:
        for graph_name, graph in summary["graphs"].items():
            split = graph["splits"].get(split_name)
            if not split:
                continue
            load = split["operator_message_load"]
            print(
                f"  {summary['dataset']}/{graph_name:22s}"
                f" unique {fmt(load.get('unique_non_self_edges'), 1):>10s}"
                f"  consumed {fmt(load.get('messages_consumed_by_operator'), 1):>10s}"
                f"  duplicate frac {fmt(load.get('duplicate_message_fraction')):>6s}"
            )
    print()

    print("=== 6. gold path preservation and bridge loss ===")
    for summary in summaries:
        for graph_name, graph in summary["graphs"].items():
            split = graph["splits"].get(split_name)
            if not split:
                continue
            preserved = split["path_preservation"]["gold_path_preservation"]
            bridge = split["path_preservation"]["gold_bridge_loss"]
            print(
                f"  {summary['dataset']}/{graph_name:22s}"
                f" global {fmt(preserved.get('connected_globally_fraction')):>6s}"
                f"  induced {fmt(preserved.get('connected_induced_fraction')):>6s}"
                f"  lost {fmt(preserved.get('globally_connected_but_induced_disconnected')):>6s}"
                f"  bridge@3 {fmt(bridge.get('bridge_loss_at_3')):>6s}"
            )
    print()

    print("=== provenance families that are the same undirected graph ===")
    any_alias = False
    for summary in summaries:
        aliases = summary["provenance_aliases"]
        for group in aliases["aliased_groups"]:
            any_alias = True
            edges = "  ".join(
                f"{name}={count:,}"
                for name, count in group["stored_directed_edges"].items()
            )
            print(
                f"  {summary['dataset']:16s} {' == '.join(group['families'])}"
            )
            print(
                f"    same undirected edge set "
                f"{group['undirected_edge_key_sha256'][:16]}; stored directed "
                f"edges differ: {edges}"
            )
        print(
            f"  {summary['dataset']:16s} {aliases['families_audited']} families ->"
            f" {aliases['distinct_undirected_graphs']} distinct undirected graph(s)"
        )
    if not any_alias:
        print("  none; every family is a distinct undirected graph")
    print()
    print("No adequacy threshold is applied. These are measurements, not verdicts.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="directory holding graph_substrate_audit/",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="exit non-zero if any discovered audit is still in progress",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "summary.json")
    args = parser.parse_args(argv)

    found = discover(args.results_root)
    if not found:
        raise SystemExit(f"No substrate audit found under {args.results_root}")

    summaries = []
    misfiled: list[str] = []
    for name, path in found.items():
        summary = summarize_audit(load_audit(path))
        # The directory says one dataset and the file says another only if
        # a fetch wrote to the wrong place. Reporting the file's own name
        # under the directory's would attribute a measurement to the wrong
        # dataset, which is worse than refusing to read it.
        if summary["dataset"] != name:
            misfiled.append(f"{path}: recorded dataset {summary['dataset']!r}, filed under {name!r}")
        summaries.append(summary)
    if misfiled:
        raise SystemExit(
            "Audit filed under the wrong dataset:\n  " + "\n  ".join(misfiled)
        )
    summaries.sort(key=lambda item: item["dataset"])
    print_report(summaries, args.split)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "split": args.split,
                "datasets": len(summaries),
                "complete": [s["dataset"] for s in summaries if s["complete"]],
                "in_progress": [s["dataset"] for s in summaries if not s["complete"]],
                "audits": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")

    incomplete = [s["dataset"] for s in summaries if not s["complete"]]
    if args.require_complete and incomplete:
        print(f"still in progress: {', '.join(incomplete)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
