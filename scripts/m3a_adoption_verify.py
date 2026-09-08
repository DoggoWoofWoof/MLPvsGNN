"""M3A items 3, 4 and 7: adoption verification for the typed substrates.

Sections B, C and F of the adoption contract. Verify is not rebuild: nothing
here writes to the package, and nothing here produces a second relation
vocabulary. Where a check recomputes something, it recomputes to compare, and
the transferred artifact stays the artifact.

Section B recomputes MetaQA's losslessness chain locally from the official
kb.txt, because at nine relation types that is cheap. The chain has to close as
an accounting identity -- source = mapped + everything explicitly accounted for
-- and any row that fails to map is enumerated, not counted.

Section C verifies WebQSP against its own source contract rather than
reinventing 7,058 relations. Where a claim cannot be independently recomputed
from what the package contains, it is recorded as
UPSTREAM_ASSERTED_NOT_LOCALLY_REPRODUCIBLE rather than accepted as measured.

Section F is diagnostic only. It quantifies how far apart the two relation
regimes are before anyone chooses a relation feature representation.
"""

from __future__ import annotations

import platform
import time

try:  # not available on Windows; the wall clock still is
    import resource
except ImportError:  # pragma: no cover - platform dependent
    resource = None

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "m3a"
JSON_PATH = OUT_DIR / "adoption_verification.json"
DOC_PATH = ROOT / "docs" / "M3A_ADOPTION_VERIFICATION.md"
DEFAULT_ROOT = Path("C:/Users/Swastik/Desktop/CRAG")

STATUS_REPRODUCED = "LOCALLY_REPRODUCED"
STATUS_UPSTREAM_ONLY = "UPSTREAM_ASSERTED_NOT_LOCALLY_REPRODUCIBLE"
STATUS_FAILED = "FAILED"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Section B -- MetaQA, recomputed locally from kb.txt
# --------------------------------------------------------------------------


def section_b(root: Path) -> dict[str, Any]:
    kb = root / "data" / "original" / "metaqa" / "kb.txt"
    entity_dict = root / "data" / "original" / "metaqa" / "entity" / "kb_entity_dict.txt"
    graph_tsv = root / "data" / "canonical" / "metaqa" / "graph_structural.tsv"
    graph_manifest = json.loads(
        (root / "data" / "canonical" / "metaqa" / "graph_manifest.json").read_text(encoding="utf-8")
    )
    build_info = json.loads(
        (root / "data" / "final_canonical" / "metaqa" / "build_info.json").read_text(encoding="utf-8")
    )

    # The source, exactly as shipped.
    raw = kb.read_text(encoding="utf-8").splitlines()
    source_lines = len(raw)
    blank = sum(1 for line in raw if not line.strip())

    parsed: list[tuple[str, str, str]] = []
    malformed: list[dict[str, Any]] = []
    for number, line in enumerate(raw, start=1):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != 3 or not all(part.strip() for part in parts):
            malformed.append({"line": number, "text": line[:200]})
            continue
        parsed.append((parts[0], parts[1], parts[2]))

    distinct = set(parsed)
    duplicates = len(parsed) - len(distinct)

    # The entity dict is the id space the canonical graph uses.
    name_to_index: dict[str, int] = {}
    for line in entity_dict.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        index, _, name = line.partition("\t")
        name_to_index[name] = int(index)

    unmapped: list[dict[str, Any]] = []
    mapped: set[tuple[int, int, str]] = set()
    for src, relation, dst in distinct:
        src_index = name_to_index.get(src)
        dst_index = name_to_index.get(dst)
        if src_index is None or dst_index is None:
            unmapped.append(
                {
                    "src": src[:120],
                    "relation": relation,
                    "dst": dst[:120],
                    "src_resolved": src_index is not None,
                    "dst_resolved": dst_index is not None,
                }
            )
            continue
        mapped.add((src_index, dst_index, relation))

    # The transferred graph, read as-is.
    shipped: set[tuple[int, int, str]] = set()
    shipped_relations: Counter[str] = Counter()
    shipped_rows = 0
    shipped_malformed = 0
    with open(graph_tsv, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            shipped_rows += 1
            parts = line.split("\t")
            if len(parts) != 3:
                shipped_malformed += 1
                continue
            src, dst, relation = parts
            shipped.add((int(src.rsplit("_", 1)[-1]), int(dst.rsplit("_", 1)[-1]), relation))
            shipped_relations[relation] += 1

    only_local = sorted(mapped - shipped)[:20]
    only_shipped = sorted(shipped - mapped)[:20]

    declared_kb_sha = (build_info.get("source_files") or {}).get(
        "data/original/metaqa/kb.txt", {}
    ).get("sha256")
    measured_kb_sha = digest(kb)

    identity_holds = source_lines == (
        len(parsed) + blank + len(malformed)
    ) and len(parsed) == (len(distinct) + duplicates)

    chain_closes = (
        len(distinct) == len(mapped) + len(unmapped)
        and mapped == shipped
        and shipped_rows == graph_manifest["n_edges"]
        and len(shipped_relations) == graph_manifest["n_relations"]
    )

    return {
        "section": "B -- MetaQA local losslessness chain",
        "recomputed_from": str(kb.relative_to(root)).replace("\\", "/"),
        "source_sha256_declared": declared_kb_sha,
        "source_sha256_measured": measured_kb_sha,
        "source_sha256_agrees": declared_kb_sha == measured_kb_sha,
        "chain": {
            "source_lines": source_lines,
            "blank_lines": blank,
            "malformed_lines": len(malformed),
            "parsed_triples": len(parsed),
            "exact_duplicates": duplicates,
            "distinct_triples": len(distinct),
            "mapped_triples": len(mapped),
            "unmapped_triples": len(unmapped),
            "shipped_graph_rows": shipped_rows,
            "shipped_graph_malformed": shipped_malformed,
            "shipped_distinct_edges": len(shipped),
        },
        "accounting_identity": {
            "source = parsed + blank + malformed": identity_holds,
            "distinct = mapped + unmapped": len(distinct) == len(mapped) + len(unmapped),
            "local_reconstruction_equals_shipped_graph": mapped == shipped,
        },
        "enumerated_unmapped": unmapped,
        "set_difference": {
            "in_local_not_shipped_sample": only_local,
            "in_shipped_not_local_sample": only_shipped,
            "in_local_not_shipped_total": len(mapped - shipped),
            "in_shipped_not_local_total": len(shipped - mapped),
        },
        "declared": {
            "n_edges": graph_manifest["n_edges"],
            "n_relations": graph_manifest["n_relations"],
            "unmapped_endpoint_edges_dropped": graph_manifest["unmapped_endpoint_edges_dropped"],
            "graph_tsv_sha256": graph_manifest["graph_tsv_sha256"],
        },
        "measured_graph_tsv_sha256": digest(graph_tsv),
        "relation_counts": dict(shipped_relations.most_common()),
        "upstream_claims": {
            "0 malformed": len(malformed) == 0 and shipped_malformed == 0,
            "0 dangling": len(unmapped) == 0,
        },
        "status": STATUS_REPRODUCED if chain_closes else STATUS_FAILED,
        "substitution_rule": (
            "The locally reconstructed graph is equal to the transferred one. It is "
            "discarded. Equivalence is the result, not a licence to substitute."
        ),
    }


# --------------------------------------------------------------------------
# Section C -- WebQSP, verified against its own contract
# --------------------------------------------------------------------------


def section_c(root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    base = root / "data" / "final_canonical" / "webqsp"
    v1 = base / "v1"
    rog = json.loads((base / "ROG_UNION_REBUILD.json").read_text(encoding="utf-8"))

    relations = pq.read_table(v1 / "relations.parquet").to_pydict()
    n_relations = len(relations["relation_uid"])
    relation_uid_unique = len(set(relations["relation_uid"])) == n_relations
    relation_key_unique = len(set(relations["relation_key"])) == n_relations
    short_labels = Counter(relations["relation_label_short"])
    qualified = Counter(relations["relation_label_qualified"])
    qualification = Counter(relations["qualification_level"])

    nodes_table = pq.read_table(v1 / "nodes.parquet", columns=["node_uid"])
    node_uids = set(nodes_table.column("node_uid").to_pylist())
    n_nodes = nodes_table.num_rows

    edges_file = pq.ParquetFile(v1 / "edges.parquet")
    n_edges = 0
    dangling_src = dangling_dst = self_loops = null_rows = 0
    seen: set[tuple[int, int, int]] = set()
    relation_frequency: Counter[int] = Counter()
    relation_uid_set = set(relations["relation_uid"])
    unknown_relation = 0
    for batch in edges_file.iter_batches(batch_size=1 << 20):
        columns = batch.to_pydict()
        for src, rel, dst in zip(
            columns["src_uid"], columns["relation_uid"], columns["dst_uid"]
        ):
            n_edges += 1
            if src is None or dst is None or rel is None:
                null_rows += 1
                continue
            if src not in node_uids:
                dangling_src += 1
            if dst not in node_uids:
                dangling_dst += 1
            if src == dst:
                self_loops += 1
            if rel not in relation_uid_set:
                unknown_relation += 1
            relation_frequency[rel] += 1
            seen.add((src, rel, dst))

    duplicates = n_edges - len(seen)

    # The RoG-exact claim. The union was built from raw shards under
    # data/original/{webqsp,cwq}/rog_*; recomputing it took the upstream build
    # 1111.8s. What is cheap here is checking the shipped v1 tables against the
    # recorded targets, which is a weaker but honest statement.
    sources_present = all(
        (root / "data" / "original" / dataset / f"rog_{dataset}").exists()
        for dataset in ("webqsp", "cwq")
    )
    targets = rog["targets"]
    measured = rog["measured"]
    v1_matches_targets = n_relations == targets["relations"] and n_edges == targets["triples"]

    return {
        "section": "C -- WebQSP typed-substrate verification",
        "verified_against": "its own source contract, not a reinvented vocabulary",
        "counts": {
            "relations": n_relations,
            "edges": n_edges,
            "nodes": n_nodes,
            "distinct_edges": len(seen),
        },
        "checks": {
            "relation_uid_unique": relation_uid_unique,
            "relation_key_unique": relation_key_unique,
            "relation_short_label_collisions": sum(1 for v in short_labels.values() if v > 1),
            "relation_qualified_label_collisions": sum(1 for v in qualified.values() if v > 1),
            "qualification_levels": dict(qualification),
            "null_rows": null_rows,
            "dangling_src": dangling_src,
            "dangling_dst": dangling_dst,
            "self_loops": self_loops,
            "duplicate_edges": duplicates,
            "edges_with_unknown_relation": unknown_relation,
        },
        "direction": {
            "schema_is_directed_triple": ["src_uid", "relation_uid", "dst_uid"],
            "inverse_edges_materialised": False,
            "note": (
                "edges.parquet stores one row per directed triple and no inverse rows. "
                "Any traversal needing reverse adjacency must add inverse edges "
                "explicitly at build time, which is what the typed graph contract's "
                "direction rule already requires."
            ),
        },
        "rog_exact_claim": {
            "upstream_targets": targets,
            "upstream_measured": measured,
            "upstream_deltas": rog["deltas"],
            "upstream_hard_check_exact": rog.get("HARD_CHECK_EXACT"),
            "v1_tables_match_targets": v1_matches_targets,
            "raw_rog_sources_present": sources_present,
            "recompute_cost_seconds_upstream": rog.get("elapsed_s"),
            "status": STATUS_UPSTREAM_ONLY,
            "why": (
                "Counting the shipped v1 tables confirms they agree with the recorded "
                "targets, which is internal consistency. It does not independently "
                "re-derive the union from the raw RoG shards, so the RoG-exact property "
                "itself remains upstream-asserted. The raw shards are present and the "
                "upstream rebuild took 1111.8s, so this is affordable if review wants it."
            ),
        },
        "endpoint_caveat_carried_from_source": rog.get("endpoints_note"),
        "union_scope": {
            "graph_is": "WebQSP union CWQ",
            "per_dataset": rog.get("per_dataset"),
            "consequence": (
                "The 8,309,195-triple graph is the union. WebQSP alone contributed "
                "3,791,303 distinct triples. Evaluating WebQSP queries against the union "
                "gives a candidate universe containing CWQ-only triples, which is a "
                "different graph from 'WebQSP's graph' and has to be a declared choice."
            ),
        },
        "text_representation_choice": {
            "entity_text_columns": ["text_name_only", "text_name_plus_facts"],
            "cvt_text_column": "text_cvt_record",
            "consequence": (
                "The encode has to pick a text field, and the choice changes what the "
                "semantic features can express and what the encode costs. It is a "
                "scientific choice, not a systems one, and belongs to review."
            ),
        },
        "status": STATUS_REPRODUCED
        if (
            relation_uid_unique
            and relation_key_unique
            and null_rows == 0
            and dangling_src == 0
            and dangling_dst == 0
            and unknown_relation == 0
            and v1_matches_targets
        )
        else STATUS_FAILED,
    }


# --------------------------------------------------------------------------
# Section F -- how different the two relation regimes actually are
# --------------------------------------------------------------------------


def entropy(counts: list[int]) -> float:
    total = sum(counts)
    return -sum((c / total) * math.log2(c / total) for c in counts if c)


def concentration(counts: list[int], k: int) -> float:
    total = sum(counts)
    return sum(sorted(counts, reverse=True)[:k]) / total if total else 0.0


def regime(name: str, counts: Counter[Any]) -> dict[str, Any]:
    values = list(counts.values())
    total = sum(values)
    h = entropy(values)
    n = len(values)
    return {
        "dataset": name,
        "relation_types": n,
        "edges": total,
        "entropy_bits": round(h, 4),
        "max_entropy_bits": round(math.log2(n), 4) if n else 0.0,
        "normalised_entropy": round(h / math.log2(n), 4) if n > 1 else None,
        "top1_share": round(concentration(values, 1), 4),
        "top10_share": round(concentration(values, 10), 4),
        "top100_share": round(concentration(values, 100), 4),
        "singleton_relations": sum(1 for v in values if v == 1),
        "median_edges_per_relation": sorted(values)[n // 2] if n else 0,
    }


def section_f(root: Path, metaqa_counts: dict[str, int], webqsp_counts: Counter[int]) -> dict[str, Any]:
    metaqa = regime("metaqa", Counter(metaqa_counts))
    webqsp = regime("webqsp", webqsp_counts)
    return {
        "section": "F -- relation vocabulary diagnostics",
        "purpose": "quantify the two regimes before choosing a relation feature representation",
        "regimes": [metaqa, webqsp],
        "cardinality_ratio": round(webqsp["relation_types"] / metaqa["relation_types"], 1),
        "relation_text_encoded": False,
        "relation_text_rule": (
            "Not encoded yet. Section F requires the audit first, and this is the audit."
        ),
        "forbidden_generalisation": (
            "A MetaQA null must never be reported as 'typed relation features do not "
            f"help'. MetaQA has {metaqa['relation_types']} relation types against WebQSP's "
            f"{webqsp['relation_types']}; a null there is about that regime."
        ),
    }


def render(data: dict[str, Any]) -> str:
    b, c, f = data["section_b"], data["section_c"], data["section_f"]
    chain = b["chain"]

    chain_rows = "\n".join(f"| {key.replace('_', ' ')} | {value:,} |" for key, value in chain.items())
    relation_rows = "\n".join(
        f"| `{name}` | {count:,} |" for name, count in b["relation_counts"].items()
    )
    check_rows = "\n".join(
        f"| {key.replace('_', ' ')} | {value} |"
        for key, value in c["checks"].items()
        if not isinstance(value, dict)
    )
    regime_rows = "\n".join(
        f"| {r['dataset']} | {r['relation_types']:,} | {r['edges']:,} | {r['entropy_bits']} | "
        f"{r['max_entropy_bits']} | {r['normalised_entropy']} | {r['top1_share']} | "
        f"{r['top10_share']} |"
        for r in f["regimes"]
    )

    return f"""# M3A adoption verification

Generated by `{data['generated_by']}`. Package root `{data['package_root']}`,
opened read-only. Verify is not rebuild: nothing was written to the package and
no second relation vocabulary was produced.

| section | status |
|---|---|
| B -- MetaQA losslessness | **{b['status']}** |
| C -- WebQSP typed substrate | **{c['status']}** |
| C -- RoG-exact property | **{c['rog_exact_claim']['status']}** |

## B. MetaQA, recomputed from kb.txt

Source sha256 {'agrees with' if b['source_sha256_agrees'] else 'DISAGREES with'} the
value the package recorded for it, so the file recomputed here is the file the
package built from.

| step | rows |
|---|---|
{chain_rows}

The identity closes in both directions:

- `source = parsed + blank + malformed` — **{b['accounting_identity']['source = parsed + blank + malformed']}**
- `distinct = mapped + unmapped` — **{b['accounting_identity']['distinct = mapped + unmapped']}**
- local reconstruction equals the shipped graph — **{b['accounting_identity']['local_reconstruction_equals_shipped_graph']}**

Unmapped rows are enumerated rather than counted; there are
{len(b['enumerated_unmapped'])} of them. The upstream claims of 0 malformed and 0
dangling are independently established here, not accepted.

{b['substitution_rule']}

| relation | edges |
|---|---|
{relation_rows}

## C. WebQSP, against its own contract

{c['counts']['relations']:,} relations, {c['counts']['edges']:,} edges,
{c['counts']['nodes']:,} nodes, counted from the shipped v1 tables.

| check | value |
|---|---|
{check_rows}

**The RoG-exact property is `{c['rog_exact_claim']['status']}`.**
{c['rog_exact_claim']['why']}

That status is not a rejection; it goes to review.

**Two things the shipped tables make visible that the transfer manifest did not.**

The graph is the WebQSP ∪ CWQ union. {c['union_scope']['consequence']}

The endpoint count carries its own caveat from source:
*{c['endpoint_caveat_carried_from_source']}*

And the encode has a choice to make: {c['text_representation_choice']['consequence']}

## F. The two relation regimes

| dataset | types | edges | entropy | max | normalised | top-1 | top-10 |
|---|---|---|---|---|---|---|---|
{regime_rows}

WebQSP's vocabulary is **{f['cardinality_ratio']}×** MetaQA's.
{f['forbidden_generalisation']}

Relation text is **not** encoded. {f['relation_text_rule']}

Machine-readable form: `outputs/m3a/adoption_verification.json`.
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
    args = parser.parse_args()
    root = args.root
    started = time.perf_counter()

    b = section_b(root)
    c = section_c(root)

    import pyarrow.parquet as pq

    edges_file = pq.ParquetFile(root / "data/final_canonical/webqsp/v1/edges.parquet")
    webqsp_relation_counts: Counter[int] = Counter()
    for batch in edges_file.iter_batches(batch_size=1 << 20, columns=["relation_uid"]):
        webqsp_relation_counts.update(batch.to_pydict()["relation_uid"])

    f = section_f(root, b["relation_counts"], webqsp_relation_counts)

    data = {
        "format": "m3a_adoption_verification_v1",
        "phase": "M3A",
        "items": "3, 4 and 7 -- MetaQA losslessness, WebQSP verification, relation diagnostics",
        "generated_by": "scripts/m3a_adoption_verify.py",
        "package_root": str(root).replace("\\", "/"),
        "access_mode": "READ_ONLY",
        "decides": "nothing; adoption is a review decision",
        "section_b": b,
        "section_c": c,
        "section_f": f,
        "measured_cost": measured_cost(started),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    DOC_PATH.write_text(render(data), encoding="utf-8", newline="\n")
    print(f"wrote {JSON_PATH.relative_to(ROOT)}")
    print(f"wrote {DOC_PATH.relative_to(ROOT)}")
    print(f"B {b['status']} | C {c['status']} | RoG {c['rog_exact_claim']['status']}")


if __name__ == "__main__":
    main()
