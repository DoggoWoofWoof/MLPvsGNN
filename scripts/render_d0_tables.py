"""Render the Stage D0 relevance-discrimination tables from the result files.

One table per question, so a reader can disagree with a conclusion without
re-deriving the numbers:

    invariant     does TARGET_H1 preserve the two-paths the theorem says it does
    overall       does either context discriminate at all, and with what ties
    stratified    does the gain land on the candidates Phase -1 found starved

The stratified table is the one the advancement criterion is read from. A gain
that appears only on already-well-connected candidates is not the repair paying
off, and the isolated column is where a restored signal has to show up if the
repair is worth anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ("CAND", "TARGET_H1")
FEATURES = (
    "seed_distance",
    "distinct_seed_support",
    "two_hop_seed_support",
    "bridge_support",
)
BUCKETS = ("isolated", "degree_1", "low_degree", "ordinary")
BUCKET_LABELS = {
    "isolated": "isolated",
    "degree_1": "deg 1",
    "low_degree": "deg 2-4",
    "ordinary": "deg 5+",
}


def load(root: Path, split: str) -> dict[str, dict]:
    results = {}
    for path in sorted(root.glob("*_stage_d0.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if split in payload.get("splits", {}):
            results[payload["dataset"]] = payload
    return results


def _rows(lines: list[str], header: list[str], aligns: list[str]) -> None:
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(aligns) + "|")


def render(results: dict[str, dict], split: str) -> str:
    lines: list[str] = []
    datasets = sorted(results)
    lines.append(
        f"Split: `{split}`. Datasets: " + ", ".join(f"`{d}`" for d in datasets) + "."
    )
    lines.append("")

    lines.append("### Coverage")
    lines.append("")
    _rows(
        lines,
        ["dataset", "measured", "no seeds", "one class only", "median positives"],
        [":-------", "-------:", "-------:", "-------------:", "--------------:"],
    )
    for dataset in datasets:
        s = results[dataset]["splits"][split]
        lines.append(
            f"| {dataset} | {s['queries_measured']} | "
            f"{s['queries_skipped_without_seeds']} | "
            f"{s['queries_skipped_without_both_classes']} | "
            f"{s['positives_per_query_median']:.1f} |"
        )
    lines.append("")

    lines.append("### Two-path preservation, measured on real queries")
    lines.append("")
    lines.append(
        "The theorem says `TARGET_H1` keeps every directed `a -> v -> b` between "
        "candidates. Kept/total, summed over queries."
    )
    lines.append("")
    _rows(
        lines,
        ["dataset", "CAND candidate 2-paths", "TARGET_H1", "CAND seed 2-paths",
         "TARGET_H1", "common successor lost"],
        [":-------", "--------------------:", "--------:", "---------------:",
         "--------:", "-------------------:"],
    )
    for dataset in datasets:
        arms = results[dataset]["splits"][split]["arms"]
        row = [dataset]
        for key in ("directed_bridges", "seed_bridges"):
            for arm in ARMS:
                p = arms[arm]["two_path_preservation"]
                total = p[key]
                kept = p[f"{key}_in_context"]
                share = kept / total if total else float("nan")
                row.append(f"{kept:,}/{total:,} ({share:.3f})")
        p = arms["TARGET_H1"]["two_path_preservation"]
        row.append(
            f"{p['common_successor_bridges'] - p['common_successor_bridges_in_context']:,}"
        )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    for dataset in datasets:
        arms = results[dataset]["splits"][split]["arms"]
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("**Discrimination over all candidates.** Mean per-query rank AUC.")
        lines.append("")
        _rows(
            lines,
            ["quantity", "CAND", "TARGET_H1", "delta", "tie fraction", "latency p95 ms"],
            [":-------", "----:", "--------:", "----:", "-----------:", "------------:"],
        )
        for feature in FEATURES:
            cand = arms["CAND"]["auc"][feature]["mean"]
            target = arms["TARGET_H1"]["auc"][feature]["mean"]
            ties = arms["TARGET_H1"]["tie_fraction"][feature]
            lines.append(
                f"| {feature} | {cand:.4f} | {target:.4f} | {target - cand:+.4f} | "
                f"{ties:.3f} | |"
            )
        lines.append(
            f"| *context cost* | | | | | "
            f"{arms['CAND']['latency_ms']['p95']:.1f} -> "
            f"{arms['TARGET_H1']['latency_ms']['p95']:.1f} |"
        )
        lines.append("")
        lines.append(
            "**Where the discrimination lands.** Mean per-query rank AUC within each "
            "stratum, by the induced degree the candidate had in `G[Cq]`."
        )
        lines.append("")
        header = ["quantity"]
        aligns = [":-------"]
        for bucket in BUCKETS:
            header.append(f"{BUCKET_LABELS[bucket]} CAND")
            header.append("TARGET_H1")
            aligns.extend(["----:", "--------:"])
        _rows(lines, header, aligns)
        for feature in FEATURES:
            row = [feature]
            for bucket in BUCKETS:
                for arm in ARMS:
                    stat = arms[arm]["auc_by_prior_induced_degree"][feature][bucket]
                    row.append(
                        "--" if stat["queries_scored"] == 0 else f"{stat['mean']:.4f}"
                    )
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = load(args.root, args.split)
    if not results:
        raise SystemExit(f"no stage_d0 results with split {args.split!r} under {args.root}")
    if args.json:
        print(json.dumps({d: r["splits"][args.split] for d, r in results.items()}, indent=2))
        return
    text = render(results, args.split)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"{len(text.splitlines())} lines -> {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
