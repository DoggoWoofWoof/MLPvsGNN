"""What would it cost to give the five datasets a retrieval view?

Blocker 2 of M3A-COMPILATION: no retrieval, top-K list or candidate pool exists
anywhere in the package, for any dataset. Every headroom, occupancy and
redundancy quantity the phase is supposed to produce sits behind that, so the
compute record has to exist before any of it is spent.

The good news is measured, not assumed: all five datasets already ship complete
dense and SPLADE *document* encodings. Nothing needs re-encoding on the corpus
side. What is missing is the scoring pass, and for exact brute-force scoring
its cost is fully determined by three numbers per dataset -- queries, documents
and dimension -- all of which are already on disk.

Exact search is priced rather than approximate. An ANN index is faster but adds
a recall parameter, and a candidate pool whose recall depends on an index knob
would confound the candidate-supply axis this phase is trying to separate from
ranking quality. Exact scoring costs little enough here that the knob is not
worth introducing.

Nothing is retrieved. This prices it. Read-only against the package.
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

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "m3a"
JSON_PATH = OUT_DIR / "retrieval_budget.json"
DOC_PATH = ROOT / "docs" / "M3A_RETRIEVAL_BUDGET.md"
DEFAULT_ROOT = Path("C:/Users/Swastik/Desktop/CRAG")

DATASETS: dict[str, str] = {
    "metaqa": "metaqa",
    "squad": "squad",
    "musique": "musique",
    "hotpotqa": "hotpotqa",
    "2wiki": "2wiki_universe",
}

# Query counts. Three are on disk; two do not exist yet and are carried from
# the upstream source counts, marked as projected so a built number and a
# hoped-for number never sit in the same column unlabelled.
PROJECTED_QUERIES = {"metaqa": 407_513, "2wiki": 192_606}

# Same effective-throughput band as the encode budget, so the two records are
# comparable. A dot product is 2 FLOPs per dimension.
EFFECTIVE_TFLOPS = {"low": 15.0, "high": 45.0}
ASSUMED_GPU_USD_PER_HOUR = 1.10
GPU_MEMORY_GIB = 24.0
FP16_BYTES = 2

# A10G / L4 class card and an ordinary NVMe volume. Both bands are wide on
# purpose: they are here to stop the matmul term being mistaken for the whole
# cost, not to predict a wall time to the second.
HBM_BYTES_PER_SECOND = {"low": 300e9, "high": 900e9}
DISK_BYTES_PER_SECOND = {"low": 200e6, "high": 2e9}


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


def dense_cost(queries: int, docs: int, dim: int) -> dict[str, Any]:
    """Exact scoring is one queries x docs x dim matmul.

    The residency term is the one that actually bites. A document matrix larger
    than the card has to be streamed in chunks, which is a systems detail --
    but it changes the wall time, so it is recorded rather than hidden.
    """

    flops = 2.0 * queries * docs * dim
    matmul = {
        "low": flops / (EFFECTIVE_TFLOPS["high"] * 1e12),
        "high": flops / (EFFECTIVE_TFLOPS["low"] * 1e12),
    }

    # The matmul is not the whole job and pretending it is would understate the
    # cost by an order of magnitude on the two large corpora. Two other terms:
    # every score has to be written and read back for top-K, which is bandwidth
    # bound rather than FLOP bound, and the document shards have to come off
    # disk at least once.
    score_bytes = float(queries) * docs * FP16_BYTES
    topk = {
        "low": score_bytes * 2 / (HBM_BYTES_PER_SECOND["high"]),
        "high": score_bytes * 2 / (HBM_BYTES_PER_SECOND["low"]),
    }
    shard_bytes = float(docs) * dim * FP16_BYTES
    disk = {
        "low": shard_bytes / DISK_BYTES_PER_SECOND["high"],
        "high": shard_bytes / DISK_BYTES_PER_SECOND["low"],
    }
    seconds = {k: matmul[k] + topk[k] + disk[k] for k in ("low", "high")}

    doc_matrix_gib = docs * dim * FP16_BYTES / 2**30
    # Residency has to hold the document block, a query block and a score
    # buffer at once, so the usable fraction is well under the card.
    usable = GPU_MEMORY_GIB * 0.55
    return {
        "flops": flops,
        "seconds_matmul": {k: round(v, 1) for k, v in matmul.items()},
        "seconds_topk_bandwidth": {k: round(v, 1) for k, v in topk.items()},
        "seconds_shard_read": {k: round(v, 1) for k, v in disk.items()},
        "gpu_seconds": {"low": round(seconds["low"], 1), "high": round(seconds["high"], 1)},
        "gpu_hours": {
            "low": round(seconds["low"] / 3600, 4),
            "high": round(seconds["high"] / 3600, 4),
        },
        "scores_computed": queries * docs,
        "doc_matrix_gib_fp16": round(doc_matrix_gib, 2),
        "fits_resident": doc_matrix_gib < usable,
        "chunks_needed": max(1, math.ceil(doc_matrix_gib / usable)),
    }


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# M3A-COMPILATION -- retrieval budget")
    add("")
    add(f"Generated by `{report['generated_by']}` on {report['generated']}.")
    add(f"Read-only against `{report['package_root']}`. No GPU. Nothing retrieved.")
    add("")
    add("Blocker 2: no retrieval, top-K or candidate pool exists in the package for any")
    add("dataset. This is the compute record that has to precede clearing it.")
    add("")

    add("## Document encodings already exist on all five")
    add("")
    add("| dataset | documents | dense | splade | complete |")
    add("| --- | ---: | ---: | ---: | --- |")
    for name, row in report["datasets"].items():
        e = row["encodings"]
        add(
            f"| {name} | {row['documents']:,} | {e['dense']:,} | {e['splade']:,} "
            f"| {'yes' if e['complete'] else 'NO'} |"
        )
    add("")
    add("Nothing needs re-encoding on the corpus side. That is the whole reason this")
    add("blocker is cheap and the encode blocker is not.")
    add("")

    add("## Exact dense scoring")
    add("")
    add("| dataset | queries | source | docs x dim | scores | GPU hours (low-high) | doc matrix | chunks |")
    add("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    total_low = total_high = 0.0
    for name, row in report["datasets"].items():
        d = row["dense_scoring"]
        h = d["gpu_hours"]
        total_low += h["low"]
        total_high += h["high"]
        add(
            f"| {name} | {row['queries']:,} | {row['queries_source']} "
            f"| {row['documents']:,} x {row['dim']:,} "
            f"| {d['scores_computed']:,} "
            f"| {h['low']:.3f}-{h['high']:.3f} | {d['doc_matrix_gib_fp16']:.2f} GiB "
            f"| {d['chunks_needed']} |"
        )
    add(f"| **all five** | | | | | **{total_low:.3f}-{total_high:.3f}** | | |")
    add("")
    add(f"At ${ASSUMED_GPU_USD_PER_HOUR:.2f}/GPU-hour that is at most "
        f"**${total_high * ASSUMED_GPU_USD_PER_HOUR:.2f}**.")
    add("")
    add("Each row is matmul + top-K bandwidth + one shard read. The matmul alone would")
    add("understate the two large corpora badly, so the other two terms are carried:")
    add("")
    add("| dataset | matmul s | top-K s | shard read s |")
    add("| --- | ---: | ---: | ---: |")
    for name, row in report["datasets"].items():
        d = row["dense_scoring"]
        add(
            f"| {name} | {d['seconds_matmul']['low']:,.1f}-{d['seconds_matmul']['high']:,.1f} "
            f"| {d['seconds_topk_bandwidth']['low']:,.1f}-{d['seconds_topk_bandwidth']['high']:,.1f} "
            f"| {d['seconds_shard_read']['low']:,.1f}-{d['seconds_shard_read']['high']:,.1f} |"
        )
    add("")
    add("`projected` queries do not exist yet -- they are the upstream source counts for")
    add("the two datasets blocker 1 covers, and clearing blocker 1 is a precondition for")
    add("those two rows. `on disk` rows could be run today.")
    add("")

    add("## Why exact rather than approximate")
    add("")
    add("An ANN index is faster but introduces a recall knob, and a candidate pool whose")
    add("recall depends on that knob would confound candidate supply with ranking")
    add("quality -- the one separation this phase is required to keep. At these sizes")
    add("exact scoring is cheap enough that the knob is not worth introducing.")
    add("")

    add("## Declared ceiling and abort criteria")
    add("")
    add(f"- hard ceiling: {report['hard_ceiling_gpu_hours']} GPU hours for all five datasets")
    add("- abort if measured throughput falls below the low band by more than 2x")
    add("- abort if any dataset's recall@K curve is flat in K, which would mean the")
    add("  scoring is wrong rather than the corpus hard")
    add("- SPLADE retrieval is not priced here: it is sparse, runs on CPU against an")
    add("  inverted index, and its cost model is not the one above. It needs its own")
    add("  record before it is spent.")
    add("")

    cost = report["measured_cost"]
    add("## Cost of this measurement")
    add("")
    add(f"- wall {cost['wall_seconds']:,.1f}s, GPU {cost['gpu_seconds']}s, USD 0")
    add("- package bytes modified: 0")
    add("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    started = time.perf_counter()
    report: dict[str, Any] = {
        "format": "m3a_retrieval_budget/1",
        "phase": "M3A-COMPILATION",
        "generated_by": "scripts/m3a_retrieval_budget.py",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "package_root": str(args.root),
        "access_mode": "READ_ONLY",
        "decides": "what clearing blocker 2 costs, before any of it is spent",
        "retrieves_nothing": True,
        "hard_ceiling_gpu_hours": 4.0,
        "datasets": {},
    }

    for name, directory_name in DATASETS.items():
        directory = args.root / "data" / "canonical" / directory_name
        dense = json.loads(
            (directory / "encodings" / "dense" / "docs" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        splade = json.loads(
            (directory / "encodings" / "splade" / "docs" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )

        query_manifest = directory / "query_manifest.json"
        if query_manifest.exists():
            queries = json.loads(query_manifest.read_text(encoding="utf-8"))["total"]
            source = "on disk"
        else:
            queries = PROJECTED_QUERIES[name]
            source = "projected"

        docs = dense["n_items"]
        dim = dense["dim"]
        report["datasets"][name] = {
            "documents": docs,
            "dim": dim,
            "queries": queries,
            "queries_source": source,
            "encodings": {
                "dense": dense["n_items"],
                "splade": splade["n_items"],
                "complete": bool(dense.get("complete") and splade.get("complete")),
            },
            "dense_scoring": dense_cost(queries, docs, dim),
        }

    report["measured_cost"] = measured_cost(started)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    DOC_PATH.write_text(render(report), encoding="utf-8")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
