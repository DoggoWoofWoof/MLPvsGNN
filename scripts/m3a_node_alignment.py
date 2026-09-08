"""M3A item 5: node-space alignment across the four id spaces.

Section E of the adoption contract. For every dataset the chain

    graph node id -> canonical document/entity id -> embedding row -> retrieval row

has to be deterministic, and one-to-one where the contract requires it. No model
result may exist before this is green, so the check reports counts in both
directions rather than a single coverage percentage: a bridge that covers every
canonical node can still leave graph endpoints stranded, and only one of those
two failures shows up in a coverage number.

Everything is read-only. The bridge is applied in memory; no rewritten copy of
any id space is produced.
"""

from __future__ import annotations

import platform
import time

try:  # not available on Windows; the wall clock still is
    import resource
except ImportError:  # pragma: no cover - platform dependent
    resource = None

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "m3a"
JSON_PATH = OUT_DIR / "node_alignment.json"
DOC_PATH = ROOT / "docs" / "M3A_NODE_ALIGNMENT.md"
DEFAULT_ROOT = Path("C:/Users/Swastik/Desktop/CRAG")

ENCODING_DIR = {
    "metaqa": "metaqa",
    "squad": "squad",
    "musique": "musique",
    "hotpotqa": "hotpotqa",
    "2wiki": "2wiki_universe",
}

# How each id space reduces to one comparable key. The bridge rules are the
# package's, restated here as parsers rather than as string rewrites, so that a
# malformed id raises instead of silently producing a key that matches nothing.
INTEGER_DATASETS = {"metaqa", "hotpotqa", "2wiki"}


def canonical_key(dataset: str, node_id: str) -> int | str:
    if dataset == "metaqa":
        return int(node_id.split(":e", 1)[1])
    if dataset == "hotpotqa":
        return int(node_id.split(":c", 1)[1])
    if dataset == "2wiki":
        return int(node_id.split(":c", 1)[1])
    return node_id


def encoding_keys(dataset: str, legacy_id: str, bridge: dict[str, Any]) -> list[int | str]:
    """One legacy row can serve more than one canonical node.

    musique has exactly one such row: the legacy id hashed the text alone while
    canonical hashes (title, text), so one paragraph carried by two titles
    becomes two canonical nodes sharing a single vector. That is correct -- the
    encoder input was the text -- but it means the bridge is one-to-many and a
    scalar return would silently drop the second node.
    """

    if dataset == "metaqa":
        return [int(legacy_id.rsplit("_", 1)[1])]
    if dataset == "hotpotqa":
        return [int(legacy_id.split("_", 1)[1])]
    if dataset == "2wiki":
        return [int(legacy_id.split(":", 1)[1])]
    mapped = bridge.get(legacy_id)
    if mapped is None:
        return []
    if isinstance(mapped, str):
        mapped = [mapped]
    return [canonical_key(dataset, node_id) for node_id in mapped]


def graph_keys(dataset: str, endpoint: str, bridge: dict[str, Any]) -> list[int | str]:
    """Graph endpoints live in the legacy id space, same as the encoding rows.

    So musique and squad endpoints go through the same emitted join table the
    embeddings use. Comparing a legacy endpoint straight to a canonical node id
    makes every endpoint look stranded and every node look edgeless, which is a
    total mismatch rather than a subtle one -- but it is a mismatch that reads
    like a real finding if the bridge is skipped.
    """

    if dataset == "metaqa":
        return [int(endpoint.rsplit("_", 1)[1])]
    if dataset == "hotpotqa":
        # hotpotqa graph endpoints are bare curids, unlike the encoding ids.
        return [int(endpoint)]
    if dataset == "2wiki":
        return [int(endpoint.split(":", 1)[1])]
    mapped = bridge.get(endpoint)
    if mapped is None:
        return []
    if isinstance(mapped, str):
        mapped = [mapped]
    return [canonical_key(dataset, node_id) for node_id in mapped]


def read_canonical(root: Path, dataset: str) -> tuple[set[Any], int, int]:
    path = root / "data" / "final_canonical" / dataset / "nodes.jsonl"
    keys: set[Any] = set()
    rows = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows += 1
            keys.add(canonical_key(dataset, json.loads(line)["node_id"]))
    return keys, rows, rows - len(keys)


def read_encoding_ids(root: Path, dataset: str, modality: str) -> Iterator[str]:
    base = root / "data" / "canonical" / ENCODING_DIR[dataset] / "encodings" / modality / "docs"
    index = json.loads((base / "index.json").read_text(encoding="utf-8"))
    for shard in index["shards"]:
        yield from json.loads((base / shard["ids"]).read_text(encoding="utf-8"))


def load_bridge(root: Path, dataset: str) -> dict[str, str]:
    path = root / "transfer" / "id_bridge" / f"{dataset}_legacy_to_canonical.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "map" in payload:
        payload = payload["map"]
    return payload


def read_graph_endpoints(
    root: Path, dataset: str, bridge: dict[str, Any]
) -> tuple[set[Any], int, int, int]:
    path = root / "data" / "canonical" / ENCODING_DIR[dataset] / "graph_structural.tsv"
    keys: set[Any] = set()
    rows = 0
    unparsed = 0
    unbridged = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                unparsed += 1
                continue
            try:
                for endpoint in (parts[0], parts[1]):
                    mapped = graph_keys(dataset, endpoint, bridge)
                    if not mapped:
                        unbridged += 1
                        continue
                    keys.update(mapped)
            except (ValueError, IndexError):
                unparsed += 1
    return keys, rows, unparsed, unbridged


def align(root: Path, dataset: str) -> dict[str, Any]:
    canonical, canonical_rows, canonical_dupes = read_canonical(root, dataset)
    bridge = load_bridge(root, dataset)

    modalities: dict[str, Any] = {}
    for modality in ("dense", "splade"):
        seen: set[Any] = set()
        rows = 0
        unbridged = 0
        fanned_out = 0
        for legacy in read_encoding_ids(root, dataset, modality):
            rows += 1
            keys = encoding_keys(dataset, legacy, bridge)
            if not keys:
                unbridged += 1
                continue
            fanned_out += len(keys) - 1
            seen.update(keys)
        modalities[modality] = {
            "rows": rows,
            "distinct_after_bridge": len(seen),
            "rows_serving_more_than_one_canonical_node": fanned_out,
            "duplicate_rows": rows + fanned_out - len(seen) - unbridged,
            "unbridged_rows": unbridged,
            "canonical_nodes_not_covered": len(canonical - seen),
            "embedding_rows_not_in_canonical": len(seen - canonical),
        }

    graph, graph_rows, graph_unparsed, graph_unbridged = read_graph_endpoints(
        root, dataset, bridge
    )
    dense_keys_missing = modalities["dense"]["canonical_nodes_not_covered"]

    return {
        "dataset": dataset,
        "encoding_dir": ENCODING_DIR[dataset],
        "bridge": "integer rewrite" if dataset in INTEGER_DATASETS else "emitted join table",
        "bridge_entries": len(bridge) or None,
        "counts": {
            "canonical_nodes": len(canonical),
            "canonical_rows": canonical_rows,
            "canonical_duplicate_rows": canonical_dupes,
            "graph_edges": graph_rows,
            "graph_unparsed_rows": graph_unparsed,
            "graph_endpoints_unbridged": graph_unbridged,
            "distinct_graph_endpoints": len(graph),
            "dense_rows": modalities["dense"]["rows"],
            "splade_rows": modalities["splade"]["rows"],
        },
        "missing": {
            "graph_endpoints_not_in_canonical": len(graph - canonical),
            "canonical_nodes_with_no_graph_endpoint": len(canonical - graph),
            "canonical_nodes_without_dense_row": dense_keys_missing,
            "canonical_nodes_without_splade_row": modalities["splade"][
                "canonical_nodes_not_covered"
            ],
            "dense_rows_not_in_canonical": modalities["dense"]["embedding_rows_not_in_canonical"],
            "splade_rows_not_in_canonical": modalities["splade"]["embedding_rows_not_in_canonical"],
        },
        "modalities": modalities,
        "green": (
            len(graph - canonical) == 0
            and dense_keys_missing == 0
            and modalities["splade"]["canonical_nodes_not_covered"] == 0
            and modalities["dense"]["embedding_rows_not_in_canonical"] == 0
            and modalities["splade"]["embedding_rows_not_in_canonical"] == 0
            and canonical_dupes == 0
            and graph_unparsed == 0
            and graph_unbridged == 0
        ),
    }


def align_webqsp(root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    v1 = root / "data" / "final_canonical" / "webqsp" / "v1"
    nodes = set(pq.read_table(v1 / "nodes.parquet", columns=["node_uid"]).column("node_uid").to_pylist())
    entity_text = pq.read_metadata(v1 / "entity_text.parquet").num_rows
    cvt_text = pq.read_metadata(v1 / "cvt_text.parquet").num_rows

    endpoints: set[int] = set()
    edges_file = pq.ParquetFile(v1 / "edges.parquet")
    rows = 0
    for batch in edges_file.iter_batches(batch_size=1 << 20, columns=["src_uid", "dst_uid"]):
        columns = batch.to_pydict()
        rows += len(columns["src_uid"])
        endpoints.update(columns["src_uid"])
        endpoints.update(columns["dst_uid"])

    return {
        "dataset": "webqsp",
        "encoding_dir": None,
        "bridge": "n/a -- v1 is a single self-consistent uid space",
        "counts": {
            "canonical_nodes": len(nodes),
            "graph_edges": rows,
            "distinct_graph_endpoints": len(endpoints),
            "entity_text_rows": entity_text,
            "cvt_text_rows": cvt_text,
            "text_rows_total": entity_text + cvt_text,
            "dense_rows": 0,
            "splade_rows": 0,
        },
        "missing": {
            "graph_endpoints_not_in_canonical": len(endpoints - nodes),
            "canonical_nodes_with_no_graph_endpoint": len(nodes - endpoints),
            "canonical_nodes_without_dense_row": len(nodes),
            "canonical_nodes_without_splade_row": len(nodes),
        },
        "text_partition_is_exact": entity_text + cvt_text == len(nodes),
        "green": False,
        "blocker": (
            "WebQSP v1 has no embeddings at all, so the chain stops at the text tables. "
            "The graph and text halves are internally consistent; the embedding and "
            "retrieval halves do not exist yet. This is what item 6 exists to fix, and "
            "until it is fixed no WebQSP row may be produced."
        ),
    }


def render(data: dict[str, Any]) -> str:
    rows = []
    for entry in data["alignments"]:
        counts = entry["counts"]
        missing = entry["missing"]
        rows.append(
            f"| {entry['dataset']} | {counts['canonical_nodes']:,} | "
            f"{counts['distinct_graph_endpoints']:,} | {counts['dense_rows']:,} | "
            f"{counts['splade_rows']:,} | {missing['graph_endpoints_not_in_canonical']:,} | "
            f"{missing['canonical_nodes_without_dense_row']:,} | "
            f"{'**green**' if entry['green'] else 'BLOCKED'} |"
        )

    orphan_rows = "\n".join(
        f"| {entry['dataset']} | {entry['missing']['canonical_nodes_with_no_graph_endpoint']:,} | "
        f"{entry['counts']['canonical_nodes']:,} | "
        f"{entry['missing']['canonical_nodes_with_no_graph_endpoint'] / entry['counts']['canonical_nodes']:.1%} |"
        for entry in data["alignments"]
    )

    return f"""# M3A node-space alignment

Generated by `{data['generated_by']}`. Package root `{data['package_root']}`,
read-only. Section E of the adoption contract: no model result may exist before
this is green.

The chain checked is

    graph node id -> canonical document/entity id -> embedding row -> retrieval row

reported in both directions, because a bridge can cover every canonical node and
still strand graph endpoints, and a single coverage number hides one of those.

| dataset | canonical | graph endpoints | dense | splade | endpoints not canonical | canonical w/o dense | state |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## Nodes the graph never mentions

Not an error, and not a blocker: a corpus node with no hyperlink or triple simply
has no edge. It is reported because it bounds what any graph feature can say --
for these nodes every structural feature is a masked default, and the mask, not
the zero, is what the model must see.

| dataset | nodes with no edge | of | share |
|---|---|---|---|
{orphan_rows}

## WebQSP

{data['webqsp_note']}

Machine-readable form: `outputs/m3a/node_alignment.json`.
"""


def measured_cost(started: float) -> dict[str, float | str]:
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

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=list(ENCODING_DIR))
    args = parser.parse_args()
    started = time.perf_counter()

    alignments = []
    for dataset in args.datasets:
        print(f"aligning {dataset} ...", flush=True)
        alignments.append(align(args.root, dataset))
    print("aligning webqsp ...", flush=True)
    webqsp = align_webqsp(args.root)
    alignments.append(webqsp)

    data = {
        "format": "m3a_node_alignment_v1",
        "phase": "M3A",
        "item": "5 -- six-dataset node/retrieval/embedding alignment",
        "generated_by": "scripts/m3a_node_alignment.py",
        "package_root": str(args.root).replace("\\", "/"),
        "access_mode": "READ_ONLY",
        "decides": "nothing; it reports whether the id spaces line up",
        "chain": "graph node id -> canonical id -> embedding row -> retrieval row",
        "alignments": alignments,
        "all_green": all(entry["green"] for entry in alignments),
        "webqsp_note": webqsp["blocker"],
        "measured_cost": measured_cost(started),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    DOC_PATH.write_text(render(data), encoding="utf-8", newline="\n")
    print(f"wrote {JSON_PATH.relative_to(ROOT)}")
    print(f"wrote {DOC_PATH.relative_to(ROOT)}")
    for entry in alignments:
        print(f"  {entry['dataset']:10s} {'green' if entry['green'] else 'BLOCKED'}")


if __name__ == "__main__":
    main()
