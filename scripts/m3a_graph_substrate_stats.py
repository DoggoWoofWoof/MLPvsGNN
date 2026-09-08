"""Measure the five-dataset graph substrate, per dataset and edge family.

This is the part of M3A-COMPILATION that needs no new compute. Blockers 1 and 2
mean no headroom, occupancy or redundancy table can exist yet -- there are no
queries on two datasets and no retrieval anywhere. But the graphs themselves
are already on disk, and they decide most of what Directions C and D cost
before a single vector is aggregated.

The number this exists for is the 2-hop fan-out. A fixed neighbourhood
prototype aggregates whatever the graph hands it; if a hub node hands it four
million neighbours, the direction is not affordable at that radius no matter
how good the idea is. Length-2 walk count is exact and comes free from the
degree vectors, so the knee can be found without touching a GPU.

Read-only against the package. Writes only to this repository's outputs/.
"""

from __future__ import annotations

import platform
import time

try:
    import resource
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "m3a"
JSON_PATH = OUT_DIR / "graph_substrate_stats.json"
DOC_PATH = ROOT / "docs" / "M3A_GRAPH_SUBSTRATE.md"
DEFAULT_ROOT = Path("C:/Users/Swastik/Desktop/CRAG")

# The five canonical development datasets. 2wiki's canonical graph lives under
# 2wiki_universe -- data/canonical/2wiki exists but is the superseded corpus,
# and putting it on the path here would quietly measure the wrong graph.
DATASETS: dict[str, str] = {
    "metaqa": "metaqa",
    "squad": "squad",
    "musique": "musique",
    "hotpotqa": "hotpotqa",
    "2wiki": "2wiki_universe",
}

FAMILIES = ("structural", "knn", "ner")

# Reciprocity and duplicate detection need the edge list resident as int64
# keys. Above this many edges that is several GiB, so the measurement is
# declined and said to be declined rather than being silently skipped or
# crashing the run.
PAIR_ANALYSIS_MAX_EDGES = 60_000_000

CHUNK_LINES = 4_000_000


def measured_cost(started: float) -> dict[str, float | str | None]:
    """What this run actually cost. Recorded by the script that spends it,
    because a wall time nobody wrote down cannot be recovered later."""

    usage = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    return {
        "wall_seconds": round(time.perf_counter() - started, 1),
        "cpu_seconds": round(usage.ru_utime + usage.ru_stime, 1) if usage else None,
        "peak_rss_bytes": getattr(usage, "ru_maxrss", None) if usage else None,
        "gpu_seconds": 0.0,
        "host": platform.platform(),
    }


def percentiles(counts: np.ndarray) -> dict[str, float]:
    """Degree summary. The tail is the part that matters: a mean of 6 over a
    max of 400,000 is not a graph where 'aggregate the neighbours' costs 6."""

    if counts.size == 0:
        return {}
    return {
        "mean": round(float(counts.mean()), 3),
        "median": float(np.median(counts)),
        "p90": float(np.percentile(counts, 90)),
        "p99": float(np.percentile(counts, 99)),
        "p99_9": float(np.percentile(counts, 99.9)),
        "max": int(counts.max()),
    }


def shannon(counts: np.ndarray) -> dict[str, float]:
    total = float(counts.sum())
    if total <= 0 or counts.size <= 1:
        return {"entropy_bits": 0.0, "normalised": 0.0}
    p = counts / total
    p = p[p > 0]
    bits = float(-(p * np.log2(p)).sum())
    return {"entropy_bits": round(bits, 4), "normalised": round(bits / math.log2(counts.size), 4)}


def read_edges(
    path: Path, family: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int], int]:
    """Stream the TSV into interned integer ids.

    The third column is interned too, not accumulated as strings: 2wiki's NER
    file alone is tens of millions of rows, and a Python str per row would cost
    more memory than the whole graph. Structural files get a label vocabulary;
    KNN and NER files get a float32 weight array.

    Node ids are interned as they are seen, so the vocabulary is the set of
    nodes the edge file actually touches -- which is not the dataset's node
    set, and is reported as its own quantity.
    """

    vocab: dict[str, int] = {}
    labels: dict[str, int] = {}
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    third_parts: list[np.ndarray] = []
    typed = family == "structural"
    malformed = 0

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        while True:
            lines = handle.readlines(CHUNK_LINES * 32)
            if not lines:
                break
            n = len(lines)
            s_chunk = np.empty(n, dtype=np.int64)
            d_chunk = np.empty(n, dtype=np.int64)
            t_chunk = np.empty(n, dtype=np.int32 if typed else np.float32)
            kept = 0
            for line in lines:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 3:
                    malformed += 1
                    continue
                u, v, w = parts[0], parts[1], parts[2]
                iu = vocab.get(u)
                if iu is None:
                    iu = vocab[u] = len(vocab)
                iv = vocab.get(v)
                if iv is None:
                    iv = vocab[v] = len(vocab)
                if typed:
                    iw = labels.get(w)
                    if iw is None:
                        iw = labels[w] = len(labels)
                    t_chunk[kept] = iw
                else:
                    try:
                        t_chunk[kept] = float(w)
                    except ValueError:
                        malformed += 1
                        continue
                s_chunk[kept] = iu
                d_chunk[kept] = iv
                kept += 1
            src_parts.append(s_chunk[:kept].copy())
            dst_parts.append(d_chunk[:kept].copy())
            third_parts.append(t_chunk[:kept].copy())

    empty = np.empty(0, dtype=np.int64)
    src = np.concatenate(src_parts) if src_parts else empty
    dst = np.concatenate(dst_parts) if dst_parts else empty
    third = np.concatenate(third_parts) if third_parts else empty
    label_names = sorted(labels, key=labels.__getitem__)
    return src, dst, third, label_names, vocab, malformed


def pair_analysis(
    src: np.ndarray, dst: np.ndarray, n_nodes: int, family: str,
    third: np.ndarray, n_labels: int,
) -> dict[str, Any]:
    """Repeated node pairs and reciprocity.

    Reciprocity decides what 'the neighbours of v' even means. If the file
    already stores both directions, an undirected prototype and a directed one
    aggregate the same vectors and the distinction is free; if it stores one,
    they are different features and Direction C has to say which it used.
    """

    edges = src.size
    if edges == 0:
        return {"status": "NO_EDGES"}
    if edges > PAIR_ANALYSIS_MAX_EDGES:
        return {
            "status": "DECLINED_TOO_LARGE",
            "edges": int(edges),
            "limit": PAIR_ANALYSIS_MAX_EDGES,
            "why": "keying both directions as int64 would need several GiB resident",
        }

    keys = src * n_nodes + dst
    keys.sort()
    distinct = int(np.count_nonzero(np.diff(keys)) + 1)

    if family == "structural" and n_labels > 0:
        # (pair, label) keyed into one int64. Guarded rather than trusted:
        # a wide vocabulary on a large node set would wrap silently and
        # invent a duplicate count that looks plausible.
        span = float(n_nodes) * float(n_nodes) * float(n_labels)
        if span >= float(np.iinfo(np.int64).max):
            repeat_kind = "parallel_typed_edges"
            exact_duplicates = None
        else:
            triple_keys = (src * n_nodes + dst) * n_labels + third.astype(np.int64)
            triple_keys.sort()
            n_triples = int(np.count_nonzero(np.diff(triple_keys)) + 1)
            repeat_kind = "parallel_typed_edges"
            exact_duplicates = int(edges - n_triples)
    else:
        repeat_kind = "duplicate_rows"
        exact_duplicates = int(edges - distinct)

    reverse = dst * n_nodes + src
    idx = np.searchsorted(keys, reverse)
    idx = np.clip(idx, 0, keys.size - 1)
    reciprocated = int(np.count_nonzero(keys[idx] == reverse))

    return {
        "status": "MEASURED",
        "distinct_directed_pairs": distinct,
        "repeated_node_pairs": int(edges - distinct),
        "repeat_kind": repeat_kind,
        "exact_duplicate_rows": exact_duplicates,
        "reciprocated_rows": reciprocated,
        "reciprocity": round(reciprocated / edges, 4),
    }


def measure_family(path: Path, family: str) -> dict[str, Any]:
    started = time.perf_counter()
    src, dst, third, label_names, vocab, malformed = read_edges(path, family)
    edges = int(src.size)
    n_nodes = max(len(vocab), 1)

    self_loops = int(np.count_nonzero(src == dst))
    out_deg = np.bincount(src, minlength=n_nodes)
    in_deg = np.bincount(dst, minlength=n_nodes)

    # What a prototype at v actually aggregates if it ignores direction. The
    # union of in- and out-neighbours is at most this; on a reciprocated file
    # it is about half.
    und_deg = out_deg + in_deg

    # Exact length-2 walk count. Upper bound on, and the aggregation cost of,
    # an unbounded 2-hop prototype -- distinct 2-hop nodes are fewer, but every
    # walk is a vector the fixed aggregator would touch.
    walks2_directed = int((in_deg.astype(np.int64) * out_deg.astype(np.int64)).sum())
    walks2_undirected = int(((und_deg.astype(np.int64)) ** 2).sum())

    record: dict[str, Any] = {
        "family": family,
        "file": path.name,
        "bytes": path.stat().st_size,
        "edge_rows": edges,
        "malformed_rows": malformed,
        "self_loops": self_loops,
        "nodes_touched": len(vocab),
        "distinct_sources": int(np.count_nonzero(out_deg)),
        "distinct_targets": int(np.count_nonzero(in_deg)),
        "out_degree": percentiles(out_deg[out_deg > 0]),
        "in_degree": percentiles(in_deg[in_deg > 0]),
        "undirected_degree": percentiles(und_deg[und_deg > 0]),
        "two_hop": {
            "walks_directed": walks2_directed,
            "walks_undirected": walks2_undirected,
            "mean_per_node_directed": round(walks2_directed / n_nodes, 1),
            "mean_per_node_undirected": round(walks2_undirected / n_nodes, 1),
            "max_single_node_undirected": int(und_deg.max()) ** 2 if edges else 0,
        },
        "pairs": pair_analysis(src, dst, n_nodes, family, third, len(label_names)),
    }

    if family == "structural":
        counts = np.bincount(third.astype(np.int64), minlength=len(label_names))
        labels = np.array(label_names, dtype=object)
        order = np.argsort(counts)[::-1]
        record["relations"] = {
            "distinct": int(labels.size),
            "singletons": int(np.count_nonzero(counts == 1)),
            "entropy": shannon(counts),
            "top": [
                {"relation": str(labels[i]), "edges": int(counts[i])} for i in order[:10]
            ],
        }
    else:
        weights = third.astype(np.float64)
        if weights.size:
            record["weights"] = {
                "min": round(float(weights.min()), 6),
                "median": round(float(np.median(weights)), 6),
                "mean": round(float(weights.mean()), 6),
                "max": round(float(weights.max()), 6),
                # A KNN or NER edge set is a knob, not a given. Direction C's
                # KNN prototype changes with this threshold, so the survival
                # curve is part of the substrate description.
                "surviving_threshold": {
                    str(t): int(np.count_nonzero(weights >= t))
                    for t in (0.1, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9)
                },
            }

    record["read_seconds"] = round(time.perf_counter() - started, 1)
    return record


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# M3A-COMPILATION -- graph substrate")
    add("")
    add(f"Generated by `{report['generated_by']}` on {report['generated']}.")
    add(f"Read-only against `{report['package_root']}`. No GPU. Nothing built.")
    add("")
    add("Measured because Directions C and D aggregate whatever the graph hands them,")
    add("and the cost of that is a property of the graph, not of the model. Blockers 1")
    add("and 2 still stand: no queries on metaqa or 2wiki, no retrieval anywhere, so")
    add("nothing here is a headroom number.")
    add("")

    add("## Edge families present")
    add("")
    add("| dataset | structural | knn | ner |")
    add("| --- | --- | --- | --- |")
    for name in report["datasets"]:
        row = report["datasets"][name]["families"]
        cells = [
            f"{row[f]['edge_rows']:,}" if f in row else "ABSENT" for f in FAMILIES
        ]
        add(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} |")
    add("")
    add("`FULL` therefore means a different union on every dataset. It must not be")
    add("averaged across them without saying so.")
    add("")

    add("## Degree, treating each family as undirected")
    add("")
    add("| dataset | family | nodes touched | edges | mean | median | p99 | p99.9 | max |")
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in report["datasets"]:
        for family, row in report["datasets"][name]["families"].items():
            d = row["undirected_degree"]
            add(
                f"| {name} | {family} | {row['nodes_touched']:,} | {row['edge_rows']:,} "
                f"| {d['mean']:,.1f} | {d['median']:,.0f} | {d['p99']:,.0f} "
                f"| {d['p99_9']:,.0f} | {d['max']:,} |"
            )
    add("")

    add("## What a 2-hop prototype would aggregate")
    add("")
    add("Exact length-2 walk counts from the degree vectors. Distinct 2-hop nodes are")
    add("fewer, but every walk is a vector a fixed aggregator touches.")
    add("")
    add("| dataset | family | 2-hop walks (undirected) | mean per node | worst single node |")
    add("| --- | --- | ---: | ---: | ---: |")
    for name in report["datasets"]:
        for family, row in report["datasets"][name]["families"].items():
            t = row["two_hop"]
            add(
                f"| {name} | {family} | {t['walks_undirected']:,} "
                f"| {t['mean_per_node_undirected']:,.1f} "
                f"| {t['max_single_node_undirected']:,} |"
            )
    add("")

    add("## Reciprocity")
    add("")
    add("Whether the file already stores both directions. If it does, a directed and")
    add("an undirected prototype aggregate the same vectors; if it does not, they are")
    add("different features and Direction C has to declare which it used.")
    add("")
    add("A repeated (u, v) on `structural` is a parallel typed edge, not a duplicate:")
    add("the same pair under two relations. `exact dups` is the column that would")
    add("indicate a real defect, and it is the one that should read zero.")
    add("")
    add("| dataset | family | reciprocity | repeated pairs | kind | exact dups | self loops |")
    add("| --- | --- | ---: | ---: | --- | ---: | ---: |")
    for name in report["datasets"]:
        for family, row in report["datasets"][name]["families"].items():
            p = row["pairs"]
            recip = f"{p['reciprocity']:.4f}" if p.get("status") == "MEASURED" else p["status"]
            measured = p.get("status") == "MEASURED"
            reps = f"{p['repeated_node_pairs']:,}" if measured else "-"
            kind = p["repeat_kind"] if measured else "-"
            dups = f"{p['exact_duplicate_rows']:,}" if measured else "-"
            add(
                f"| {name} | {family} | {recip} | {reps} | {kind} | {dups} "
                f"| {row['self_loops']:,} |"
            )
    add("")

    add("## Relation vocabulary")
    add("")
    add("| dataset | distinct | singletons | normalised entropy |")
    add("| --- | ---: | ---: | ---: |")
    for name in report["datasets"]:
        row = report["datasets"][name]["families"].get("structural", {})
        rel = row.get("relations")
        if rel:
            add(
                f"| {name} | {rel['distinct']:,} | {rel['singletons']:,} "
                f"| {rel['entropy']['normalised']:.4f} |"
            )
    add("")
    add("Only metaqa carries a typed vocabulary. Every other dataset has exactly one")
    add("relation, so Direction B is metaqa-only by necessity and not merely by")
    add("authorisation -- there is nothing to type anywhere else in this substrate.")
    add("That makes a metaqa null the *only* typed evidence the five datasets can")
    add("produce, which is precisely why it must not be generalised.")
    add("")

    add("## Edge weights on the derived families")
    add("")
    add("KNN and NER edges are a knob, not a given, so the survival curve is part of the")
    add("substrate description. The two families are **not on the same scale**: KNN")
    add("weights are cosines in [0, 1], while NER weights run past 1 (to 35.3 on 2wiki)")
    add("with a median near 0.07. A single numeric threshold applied across both would")
    add("be meaningless, and Direction C has to set them separately.")
    add("")
    add("| dataset | family | edges | min | median | max | >=0.5 | >=0.7 | >=0.9 |")
    add("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in report["datasets"]:
        for family, row in report["datasets"][name]["families"].items():
            w = row.get("weights")
            if not w:
                continue
            s = w["surviving_threshold"]
            add(
                f"| {name} | {family} | {row['edge_rows']:,} | {w['min']:.4f} "
                f"| {w['median']:.4f} | {w['max']:.4f} "
                f"| {s['0.5']:,} | {s['0.7']:,} | {s['0.9']:,} |"
            )
    add("")

    cost = report["measured_cost"]
    add("## Cost of this measurement")
    add("")
    add(f"- wall {cost['wall_seconds']:,.1f}s, GPU {cost['gpu_seconds']}s")
    add(f"- bytes read {report['bytes_read']:,} ({report['bytes_read'] / 2**30:.2f} GiB)")
    add("- package bytes modified: 0")
    add("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument(
        "--render-only",
        action="store_true",
        help=(
            "Redraw the document from the measurements already on disk. The 3.5 GiB "
            "pass is not repeated to add a table the existing rows already contain."
        ),
    )
    args = parser.parse_args()

    if args.render_only:
        report = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        DOC_PATH.write_text(render(report), encoding="utf-8")
        print(f"redrew {DOC_PATH} from {JSON_PATH} (nothing re-read)")
        return

    started = time.perf_counter()
    report: dict[str, Any] = {
        "format": "m3a_graph_substrate_stats/1",
        "phase": "M3A-COMPILATION",
        "generated_by": "scripts/m3a_graph_substrate_stats.py",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "package_root": str(args.root),
        "access_mode": "READ_ONLY",
        "decides": (
            "whether Directions C and D are affordable at 1 and 2 hops, and what "
            "FULL means per dataset"
        ),
        "not_a_headroom_number": (
            "No queries on metaqa or 2wiki and no retrieval on any dataset, so no "
            "candidate, occupancy or ceiling quantity can be derived from this."
        ),
        "datasets": {},
        "bytes_read": 0,
    }

    for name in args.datasets:
        directory = args.root / "data" / "canonical" / DATASETS[name]
        entry: dict[str, Any] = {"directory": str(directory), "families": {}}
        for family in FAMILIES:
            path = directory / f"graph_{family}.tsv"
            if not path.exists():
                continue
            print(f"[{name}/{family}] reading {path.stat().st_size / 2**20:,.1f} MiB ...", flush=True)
            entry["families"][family] = measure_family(path, family)
            report["bytes_read"] += path.stat().st_size
        entry["families_absent"] = [f for f in FAMILIES if f not in entry["families"]]
        report["datasets"][name] = entry

    report["measured_cost"] = measured_cost(started)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    DOC_PATH.write_text(render(report), encoding="utf-8")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
