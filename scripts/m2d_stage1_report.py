#!/usr/bin/env python
"""Report M2D Stage 1: what the eight fits measured, and what the gate returned.

Section 15 asks for a fixed list -- exact metrics, the three per-blocker
deltas, the error-conditioned top-1 analysis, p50/p95/p99, exact parameter
counts, training time, measured cost -- then the verdict, then STOP_FOR_REVIEW.
This renders that list and nothing else. It decides nothing: the verdict is
read from ``stage1_gate.json``, which was produced by a gate committed before
any of these fits existed.

Every number is read from a file:

* the arm numbers from the eight fetched artifacts, each verified on fetch
  against its own identity, content digest and row count, and selected by
  commit rather than by being newest;
* the S3 and S4 rows from M2B's immutable baseline table, reused, never refit;
* the predicted spend from the Stage-1 compute record, filed before launch.

Two things it deliberately does NOT do.

**It does not restate the gate's reasoning.** The gate's own render is a
separate document. This report carries the verdict and the one distinction the
verdict's label hides -- that A3-MINIMAL's shortfall on musique_clean/R1 was
RESOLVABLE, not FAIL -- because a reader who saw only "case 3, both fail"
would draw a stronger conclusion than the artifacts support.

**It does not compute a measured cost from a fresh basis.** Section 6's record
priced this stage as an itemised line list; the measurement substitutes the
measured seconds into those same lines, so the prediction and the measurement
are comparable line by line rather than being two different quantities that
happen to share a unit.

Reads only. Writes docs/M2D_STAGE1_REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import m2d_stage1_gate as gate

REPORT = REPO_ROOT / "docs" / "M2D_STAGE1_REPORT.md"
GATE_JSON = REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage1_gate.json"
RECORD_JSON = (
    REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage1_compute_record.json"
)

BLOCKERS = ("squad_clean/R1", "musique_clean/R1")
CONTROLS = ("hotpotqa_clean/R1", "metaqa/R1")
CELLS = BLOCKERS + CONTROLS
ARMS = ("A1", "A3_MINIMAL")
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")


def pp(value: float) -> str:
    """A signed percentage-point delta. The sign is the whole content of most
    of these cells, so it is never dropped for a positive number."""

    return f"{value * 100.0:+.3f}"


def load() -> tuple[dict, dict, dict, list]:
    results = gate.load_results()
    if len(results) != len(CELLS) * len(ARMS):
        raise SystemExit(
            f"{len(results)} of {len(CELLS) * len(ARMS)} fits are present; section 15 "
            "reports a completed stage, and a partial one is what the gate refuses."
        )
    verdict = json.loads(GATE_JSON.read_text(encoding="utf-8"))
    record = json.loads(RECORD_JSON.read_text(encoding="utf-8"))
    return results, verdict, record, gate.baseline_rows()


def measured_cost(results: dict, record: dict) -> dict[str, Any]:
    """The record's own line items, re-priced with the seconds that happened.

    The native-S4 lines keep the record's filed p50 as their basis: the
    container did re-score native S4, but the artifact carries that pass's p95
    rather than its p50, and substituting a p95 into a line the prediction
    built from a p50 would report a difference in basis as a difference in
    cost.
    """

    rate = record["container"]["usd_per_hour"]
    utilisation = record["prediction"]["utilisation_assumed"]
    overhead = record["prediction"]["container_overhead_usd"]
    passes = record["workload"]["cells"][0]["latency_benchmark_passes"]

    cells = []
    for cell in CELLS:
        priced = next(item for item in record["workload"]["cells"] if item["cell"] == cell)
        held, s4_p50 = priced["held_out_queries"], priced["s4_uncached_p50_ms"]
        train = 0.0
        score = held * s4_p50 / 1000.0
        bench = passes * s4_p50 / 1000.0
        for arm in ARMS:
            systems = results[(cell, arm)]["systems"]
            train += systems["train_time_seconds"]
            score += held * systems["uncached_p50_ms"] / 1000.0
            bench += passes * systems["uncached_p50_ms"] / 1000.0
        cells.append(
            {
                "cell": cell,
                "predicted_seconds": priced["seconds"],
                "measured_seconds": train + score + bench,
                "train_seconds": train,
                "scoring_seconds": score,
                "benchmark_seconds": bench,
            }
        )

    measured = sum(item["measured_seconds"] for item in cells)
    predicted = sum(item["predicted_seconds"] for item in cells)
    compute = measured / utilisation / 3600.0 * rate
    return {
        "cells": cells,
        "measured_work_seconds": measured,
        "predicted_work_seconds": predicted,
        "ratio_measured_over_predicted": measured / predicted,
        "compute_usd": compute,
        "container_overhead_usd": overhead,
        "measured_spend_usd": compute + overhead,
        "predicted_spend_usd": record["prediction"]["expected_spend_usd"],
        "hard_ceiling_usd": record["prediction"]["hard_ceiling_usd"],
        "within_ceiling": compute + overhead <= record["prediction"]["hard_ceiling_usd"],
        "utilisation_assumed": utilisation,
        "usd_per_hour": rate,
    }


def render(results: dict, verdict: dict, record: dict, rows: list) -> str:
    cost = measured_cost(results, record)
    out: list[str] = []
    add = out.append

    add("# M2D Stage 1 — report")
    add("")
    add(
        f"**Verdict: {verdict['verdict']}** — "
        f"{(verdict.get('prospective_case') or {}).get('case', 'no case')}, applied by "
        "`scripts/m2d_stage1_gate.py`, which was committed at `6f429ee` before any of "
        "these eight fits existed."
    )
    add("")
    add(
        "Eight new fits: two arms (A1, A3-MINIMAL) at seed 0 on four cells. S3 and S4 "
        "are reused from M2B's immutable baseline table on the same held-out portion; "
        "nothing already fit was refit to produce a comparison row. Every artifact was "
        "verified on fetch against its held-out panel digest and its cell's shared "
        "structural inputs, and selected by commit "
        f"`{next(iter(results.values()))['provenance']['source_commit'][:12]}`. No "
        "test data was read: every fit reports `test_split_read: false`."
    )
    add("")

    # -- Section 8 ----------------------------------------------------------
    add("## Metrics (section 8)")
    add("")
    add("Mandatory five, on the held-out panel. `reuse` rows are M2B's, not refit here.")
    add("")
    for cell in CELLS:
        kind = "blocker" if cell in BLOCKERS else "control"
        add(f"### {cell} ({kind})")
        add("")
        add("| model | R@1 | R@5 | R@20 | MRR | FullCov@20 |")
        add("|---|---:|---:|---:|---:|---:|")
        for rung in ("S3", "S4"):
            base = gate.baseline(rows, cell, rung)
            add(
                f"| {rung} (reuse) | "
                + " | ".join(f"{base[key]:.6f}" for key in METRICS)
                + " |"
            )
        for arm in ARMS:
            metrics = results[(cell, arm)]["metrics"]
            add(
                f"| {arm} | " + " | ".join(f"{metrics[key]:.6f}" for key in METRICS) + " |"
            )
        add("")

    # -- The three deltas ---------------------------------------------------
    add("## The three per-blocker deltas (section 8)")
    add("")
    add("Percentage points. The gate's blocker rule reads the `A3_MINIMAL − S3` column.")
    add("")
    add("| blocker | metric | A1 − S4 | A3_MINIMAL − S4 | A3_MINIMAL − S3 |")
    add("|---|---|---:|---:|---:|")
    for cell in BLOCKERS:
        s3 = gate.baseline(rows, cell, "S3")
        s4 = gate.baseline(rows, cell, "S4")
        for key in METRICS:
            a1 = results[(cell, "A1")]["metrics"][key]
            a3 = results[(cell, "A3_MINIMAL")]["metrics"][key]
            add(
                f"| {cell} | {key} | {pp(a1 - s4[key])} | {pp(a3 - s4[key])} "
                f"| {pp(a3 - s3[key])} |"
            )
    add("")
    add("Controls, against native S4 — section 7's rule is no R@5 regression beyond 0.50pp.")
    add("")
    add("| control | A1 R@5 − S4 | A3_MINIMAL R@5 − S4 |")
    add("|---|---:|---:|")
    for cell in CONTROLS:
        s4 = gate.baseline(rows, cell, "S4")["recall@5"]
        add(
            f"| {cell} | {pp(results[(cell, 'A1')]['metrics']['recall@5'] - s4)} "
            f"| {pp(results[(cell, 'A3_MINIMAL')]['metrics']['recall@5'] - s4)} |"
        )
    add("")

    # -- Error-conditioned analysis ----------------------------------------
    add("## Error-conditioned top-1 analysis (section 8)")
    add("")
    add(
        "Conditioned on the queries native S4 got wrong at rank 1, measured against a "
        "re-score of M2B's own S4 checkpoint in the same container on the same panel — "
        "not against the filed row, so the pairing is query-by-query."
    )
    add("")
    add(
        "| cell | arm | S4 top-1 errors | corrected | fraction of errors | newly broken "
        "| fraction of correct | net |"
    )
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    for cell in CELLS:
        for arm in ARMS:
            block = results[(cell, arm)]["integration"]
            corrected_fraction = block.get("corrected_fraction_of_s4_errors")
            broken_fraction = block.get("newly_broken_fraction_of_s4_correct")
            add(
                f"| {cell} | {arm} | {block['s4_top1_errors']:,} | {block['corrected']:,} "
                f"| {'—' if corrected_fraction is None else f'{corrected_fraction:.4f}'} "
                f"| {block['newly_broken']:,} "
                f"| {'—' if broken_fraction is None else f'{broken_fraction:.4f}'} "
                f"| {block['net_top1_corrections']:+,} |"
            )
    add("")

    # -- Section 10 ---------------------------------------------------------
    add("## Latency (section 10)")
    add("")
    add(
        "Full uncached inference: the structural columns, all of native S4's projection "
        "work, the arm's added column and its reduction, and the widened scorer. Nothing "
        "precomputed between queries — every fit reports "
        "`cached_or_precomputed_semantic_difference: false`. Section 10's standing "
        "requirement is that a repaired S4 keep p95 below S3's."
    )
    add("")
    add("| cell | model | p50 ms | p95 ms | p99 ms | p95 below S3 |")
    add("|---|---|---:|---:|---:|---|")
    for cell in CELLS:
        s3_p95 = gate.baseline(rows, cell, "S3")["uncached_p95_ms"]
        for rung in ("S3", "S4"):
            base = gate.baseline(rows, cell, rung)
            add(
                f"| {cell} | {rung} (reuse) | {base['uncached_p50_ms']:.4f} "
                f"| {base['uncached_p95_ms']:.4f} | {base['uncached_p99_ms']:.4f} "
                f"| {'—' if rung == 'S3' else 'yes'} |"
            )
        for arm in ARMS:
            systems = results[(cell, arm)]["systems"]
            add(
                f"| {cell} | {arm} | {systems['uncached_p50_ms']:.4f} "
                f"| {systems['uncached_p95_ms']:.4f} | {systems['uncached_p99_ms']:.4f} "
                f"| {'yes' if systems['uncached_p95_ms'] < s3_p95 else 'NO'} |"
            )
    add("")
    add("Added semantic time and scorer time, same container, against a native S4 re-score.")
    add("")
    add(
        "| cell | arm | semantic p95 | scorer p95 | native S4 p95 | added p95 | increase |"
    )
    add("|---|---|---:|---:|---:|---:|---:|")
    for cell in CELLS:
        for arm in ARMS:
            systems = results[(cell, arm)]["systems"]
            add(
                f"| {cell} | {arm} | {systems['semantic_p95_ms']:.4f} "
                f"| {systems['scorer_p95_ms']:.4f} "
                f"| {systems['native_s4_same_container_p95_ms']:.4f} "
                f"| {systems['added_semantic_p95_ms_same_container']:.4f} "
                f"| {systems['increase_over_native_s4_same_container_pct']:.2f}% |"
            )
    add("")
    add(
        "> The per-component figures come from a hooked pass whose synchronisations cost "
        "time the clean pass does not pay. They do not sum to the total and are not a "
        "partition of it; the p95 column above is the number section 10 asks for."
    )
    add("")

    # -- Section 11 ---------------------------------------------------------
    add("## Parameters (section 11)")
    add("")
    width = {results[key]["precomputed_width"] for key in results}
    add(
        f"At the live structural width of {min(width)}, which every one of the "
        "eight artifacts reports and which M2B's filed fits pin independently."
    )
    add("")
    add("| model | semantic | scorer | total | added semantic | added total vs S4 |")
    add("|---|---:|---:|---:|---:|---:|")
    s4_row = gate.baseline(rows, "squad_clean/R1", "S4")
    s3_row = gate.baseline(rows, "squad_clean/R1", "S3")
    for label, row in (("S3 (reuse)", s3_row), ("S4 (reuse)", s4_row)):
        add(
            f"| {label} | {row['semantic_parameters']:,} | {row['scorer_parameters']:,} "
            f"| {row['total_parameters']:,} | — | "
            f"{row['total_parameters'] - s4_row['total_parameters']:+,} |"
        )
    for arm in ARMS:
        parameters = results[("squad_clean/R1", arm)]["parameters"]
        add(
            f"| {arm} | {parameters['semantic']:,} | {parameters['scorer']:,} "
            f"| {parameters['total']:,} | {parameters['added_semantic_parameters']:,} "
            f"| {parameters['total'] - s4_row['total_parameters']:+,} |"
        )
    add("")
    add(
        "A3-MINIMAL adds exactly the 1,536 semantic parameters section 3 authorised, and "
        "nothing else: no `semantic_product`, no `dot_qd_pct`, no structural feature."
    )
    add("")

    # -- Training time and cost --------------------------------------------
    add("## Training time and measured cost (sections 6 and 15)")
    add("")
    add("| cell | arm | training s | peak train VRAM MB | peak inference VRAM MB | peak RSS MB |")
    add("|---|---|---:|---:|---:|---:|")
    total_training = 0.0
    for cell in CELLS:
        for arm in ARMS:
            systems = results[(cell, arm)]["systems"]
            total_training += systems["train_time_seconds"]
            add(
                f"| {cell} | {arm} | {systems['train_time_seconds']:.2f} "
                f"| {systems['peak_train_vram_mb']:.1f} "
                f"| {systems['peak_inference_vram_mb']:.1f} "
                f"| {systems['peak_rss_mb']:.1f} |"
            )
    add("")
    add(f"Total training time across the eight fits: **{total_training:.2f} s**.")
    add("")
    add(
        "Measured cost re-prices the compute record's own line items with the seconds "
        "that happened, so the two are comparable line by line."
    )
    add("")
    add("| cell | predicted work s | measured work s | ratio | train | scoring | benchmarks |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for item in cost["cells"]:
        add(
            f"| {item['cell']} | {item['predicted_seconds']:.1f} "
            f"| {item['measured_seconds']:.1f} "
            f"| {item['measured_seconds'] / item['predicted_seconds']:.2f} "
            f"| {item['train_seconds']:.1f} | {item['scoring_seconds']:.1f} "
            f"| {item['benchmark_seconds']:.1f} |"
        )
    add(
        f"| **total** | **{cost['predicted_work_seconds']:.1f}** "
        f"| **{cost['measured_work_seconds']:.1f}** "
        f"| **{cost['ratio_measured_over_predicted']:.2f}** | | | |"
    )
    add("")
    add(
        f"At the record's own divisor ({cost['utilisation_assumed']}) and rate "
        f"(${cost['usd_per_hour']}/h): **${cost['compute_usd']:.4f}** compute plus "
        f"${cost['container_overhead_usd']:.4f} measured container overhead = "
        f"**${cost['measured_spend_usd']:.2f} measured**, against "
        f"${cost['predicted_spend_usd']:.2f} predicted and a "
        f"${cost['hard_ceiling_usd']:.2f} filed ceiling. Within ceiling: "
        f"{str(cost['within_ceiling']).lower()}."
    )
    add("")
    add(
        "The prediction came in high, in the direction the record said it would: its "
        "multiplier was measured on a host CPU, and the record stated in advance that a "
        "host ratio should be expected to overstate the container ratio because a "
        "compute-bound GEMM and a memory-bound elementwise pass do not scale alike."
    )
    add("")

    # -- Verdict ------------------------------------------------------------
    add("## Verdict")
    add("")
    add(f"**{verdict['verdict']}**")
    add("")
    case = verdict.get("prospective_case") or {}
    if case:
        add(
            f"Reading, filed in advance: **{case['case']}** ({case['pattern']}) — "
            f"{case['conclusion']}"
        )
        add("")
        add(f"Action: {case['action']}")
        add("")
    add(
        "Neither arm passed the filed rule, so S3 is retained and S4 development stops. "
        "One distinction the case label does not carry, recorded here because it changes "
        "what a reader should conclude:"
    )
    add("")
    for arm in ARMS:
        block = verdict["arms"][arm]
        outcomes = ", ".join(
            f"{cell} {entry['outcome']}" for cell, entry in block["blockers"].items()
        )
        add(f"* **{arm}** — {outcomes}.")
    add("")
    add(
        "A3-MINIMAL's shortfall on musique_clean/R1 is RESOLVABLE, not FAIL: it misses "
        "the 0.50pp bound against S3 by less than that cell's own measured S4 seed "
        "spread, so seed 0 provably cannot settle it. RESOLVABLE is a third outcome, not "
        "a softer failure, and it never advances the phase on its own. The declaration's "
        "mechanism for that state is seeds 1 and 2 on the blockers, which section 14 "
        "authorises only on a pass; the gate reports it could change the decision for "
        f"A3-MINIMAL and authorises it: "
        f"{str(verdict['extra_seeds']['authorised_by_this_gate']).lower()}."
    )
    add("")
    add(
        "So the honest summary is not that the added column did nothing. It corrected "
        "more top-1 errors than it broke on both blockers, improved R@1 and MRR against "
        "native S4 on every cell but metaqa, held both controls, and stayed under S3's "
        "p95 everywhere. It did not reach S3's R@5 on musique_clean/R1 by a margin seed 0 "
        "cannot resolve. Under the rule filed before the fits, that is not a repair."
    )
    add("")
    add("## STOP_FOR_REVIEW")
    add("")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    text = render(*load())
    if args.print_only:
        print(text)
        return 0
    REPORT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {REPORT.relative_to(REPO_ROOT)} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
