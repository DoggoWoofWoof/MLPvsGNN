"""Can metaqa and 2wiki be given a canonical query view, and what would it cost?

Blocker 1 of M3A-COMPILATION: those two datasets ship no canonical queries, no
query manifest and no query encodings, while squad, musique and hotpotqa ship
all three. The source material exists upstream under data/final_canonical, but
in a different id space -- upstream cites `metaqa:e36865` and `2wiki:c1933873`
where the canonical graphs use `metaqa_ent_36865` and `2wu:1933873`.

Two questions decide whether the blocker is worth clearing, and only one of
them is about money:

  1. Do the upstream gold node ids actually resolve into the canonical graph?
     A query whose gold nodes are not in the substrate cannot be scored against
     it at any budget, so this is measured before anything is priced. It is
     also the question that a prefix rewrite makes *look* trivial: the rewrite
     is deterministic, but whether the target exists is not.
  2. What would encoding the resulting queries cost?

Nothing is built and nothing is encoded. This prices and tests feasibility.
Read-only against the package.
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
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from m3a_webqsp_encode_budget import (  # noqa: E402
    ASSUMED_DENSE_TOKENS_PER_GPU_SECOND,
    ASSUMED_GPU_USD_PER_HOUR,
    ASSUMED_SPLADE_TOKENS_PER_GPU_SECOND,
    BATCH,
    CHARS_PER_TOKEN,
    DENSE_MAX_TOKENS,
    SPLADE_MAX_TOKENS,
    file_order_batch_slots,
    sorted_batch_slots,
)

OUT_DIR = ROOT / "outputs" / "m3a"
JSON_PATH = OUT_DIR / "query_view_feasibility.json"
DOC_PATH = ROOT / "docs" / "M3A_QUERY_VIEW_FEASIBILITY.md"
DEFAULT_ROOT = Path("C:/Users/Swastik/Desktop/CRAG")

SPLITS = ("train", "dev", "test")

# Where each blocked dataset's queries come from, which canonical graph they
# would be scored against, and how an upstream gold id becomes a canonical one.
# The rewrite is asserted here and then tested against the canonical node set;
# it is never assumed to have worked.
BLOCKED: dict[str, dict[str, Any]] = {
    "metaqa": {
        "source": "data/final_canonical/metaqa/queries",
        "canonical": "metaqa",
        "gold_field": "gold_node_ids",
        # Upstream zero-pads the entity index to five digits and the canonical
        # graph does not: `metaqa:e04210` is `metaqa_ent_4210`, not
        # `metaqa_ent_04210`. Taking the rewrite as a plain prefix swap made
        # 58% of gold ids miss and made metaqa look unusable. The padding is
        # stripped numerically rather than by lstrip("0"), which would turn a
        # legitimate `e0` into an empty string.
        "rewrite": ("metaqa:e", "metaqa_ent_"),
        "strip_zero_padding": True,
    },
    "2wiki": {
        "source": "data/final_canonical/2wiki/queries",
        "canonical": "2wiki_universe",
        "gold_field": "gold_node_ids",
        "fallback_gold_field": "context_node_ids",
        "rewrite": ("2wiki:c", "2wu:"),
        "strip_zero_padding": False,
    },
}

# Datasets that already have a query view. Read to confirm the target shape
# rather than inventing one, and to check the blocked pair against a working
# example.
PRESENT = ("squad", "musique", "hotpotqa")


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


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def canonical_node_ids(directory: Path) -> set[str]:
    """The node id set the graph actually uses, taken from documents.jsonl."""

    ids: set[str] = set()
    for record in read_jsonl(directory / "documents.jsonl"):
        ids.add(record["canonical_doc_id"])
    return ids


def measure_split(
    path: Path, spec: dict[str, Any], node_ids: set[str]
) -> dict[str, Any]:
    prefix, replacement = spec["rewrite"]
    strip_padding = spec.get("strip_zero_padding", False)
    primary = spec["gold_field"]
    fallback = spec.get("fallback_gold_field")

    def rewrite(raw: str) -> str:
        body = raw[len(prefix) :]
        if strip_padding and body.isdigit():
            body = str(int(body))
        return replacement + body

    rows = 0
    question_chars: list[int] = []
    gold_total = 0
    gold_resolved = 0
    gold_wrong_prefix = 0
    queries_with_gold = 0
    queries_fully_resolved = 0
    queries_with_no_resolved_gold = 0
    used_fallback = 0

    for record in read_jsonl(path):
        rows += 1
        question = record.get("question") or record.get("question_plain") or ""
        question_chars.append(len(question))

        gold = record.get(primary)
        if not gold and fallback:
            gold = record.get(fallback)
            if gold:
                used_fallback += 1
        gold = gold or []
        if not gold:
            continue

        queries_with_gold += 1
        resolved = 0
        for raw in gold:
            gold_total += 1
            if raw.startswith(prefix):
                mapped = rewrite(raw)
            else:
                gold_wrong_prefix += 1
                mapped = raw
            if mapped in node_ids:
                gold_resolved += 1
                resolved += 1
        if resolved == len(gold):
            queries_fully_resolved += 1
        if resolved == 0:
            queries_with_no_resolved_gold += 1

    chars = np.asarray(question_chars, dtype=np.int64)
    tokens = np.ceil(chars / CHARS_PER_TOKEN).astype(np.int64)

    return {
        "rows": rows,
        "queries_with_gold": queries_with_gold,
        "queries_using_fallback_field": used_fallback,
        "gold_ids_total": gold_total,
        "gold_ids_resolved": gold_resolved,
        "gold_ids_unresolved": gold_total - gold_resolved,
        "gold_ids_with_unexpected_prefix": gold_wrong_prefix,
        "gold_resolution_rate": round(gold_resolved / gold_total, 6) if gold_total else None,
        "queries_fully_resolved": queries_fully_resolved,
        "queries_with_no_resolved_gold": queries_with_no_resolved_gold,
        "question_tokens": {
            "total": int(tokens.sum()),
            "mean": round(float(tokens.mean()), 2) if tokens.size else 0.0,
            "max": int(tokens.max()) if tokens.size else 0,
        },
        "_tokens": tokens,
    }


def price(tokens: np.ndarray) -> dict[str, Any]:
    """Padded token slots, not rows. Queries are short and uniform, so padding
    overhead here is small -- which is itself worth recording, because it is
    what makes this budget unlike the WebQSP node budget."""

    dense = sorted_batch_slots(tokens, DENSE_MAX_TOKENS, BATCH)
    splade = file_order_batch_slots(tokens, SPLADE_MAX_TOKENS, BATCH)

    def hours(slots: int, rate: dict[str, float]) -> dict[str, float]:
        return {
            "low": round(slots / rate["high"] / 3600, 4),
            "high": round(slots / rate["low"] / 3600, 4),
        }

    dense_hours = hours(dense["padded_token_slots"], ASSUMED_DENSE_TOKENS_PER_GPU_SECOND)
    splade_hours = hours(splade["padded_token_slots"], ASSUMED_SPLADE_TOKENS_PER_GPU_SECOND)
    total_high = dense_hours["high"] + splade_hours["high"]

    return {
        "dense": {**dense, "gpu_hours": dense_hours},
        "splade": {**splade, "gpu_hours": splade_hours},
        "gpu_hours_total": {
            "low": round(dense_hours["low"] + splade_hours["low"], 4),
            "high": round(total_high, 4),
        },
        "usd_high": round(total_high * ASSUMED_GPU_USD_PER_HOUR, 2),
    }


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# M3A-COMPILATION -- can the two blocked datasets get a query view?")
    add("")
    add(f"Generated by `{report['generated_by']}` on {report['generated']}.")
    add(f"Read-only against `{report['package_root']}`. No GPU. Nothing built or encoded.")
    add("")
    add("## The id rewrite, tested rather than assumed")
    add("")
    add("Upstream cites gold nodes in its own id space. The rewrite is deterministic,")
    add("but whether the target exists in the canonical graph is a separate question,")
    add("and it is the one that decides whether the blocker is worth clearing.")
    add("")
    add("`unpadded` means upstream zero-pads the index and the canonical graph does")
    add("not. Read as a plain prefix swap, `metaqa:e04210` becomes `metaqa_ent_04210`,")
    add("which does not exist -- 58% of metaqa gold ids missed and the dataset looked")
    add("unusable. That was the rewrite being wrong, not the data.")
    add("")
    add("| dataset | rewrite | canonical nodes | gold ids | resolved | rate |")
    add("| --- | --- | ---: | ---: | ---: | ---: |")
    for name, entry in report["blocked"].items():
        t = entry["totals"]
        rewrite = f"`{entry['rewrite'][0]}` -> `{entry['rewrite'][1]}`"
        if entry["strip_zero_padding"]:
            rewrite += ", unpadded"
        rate = f"{t['gold_resolution_rate']:.4f}" if t["gold_resolution_rate"] is not None else "-"
        add(
            f"| {name} | {rewrite} | {entry['canonical_nodes']:,} | {t['gold_ids_total']:,} "
            f"| {t['gold_ids_resolved']:,} | {rate} |"
        )
    add("")

    add("## Per split")
    add("")
    add("| dataset | split | queries | with gold | fully resolved | no resolved gold |")
    add("| --- | --- | ---: | ---: | ---: | ---: |")
    for name, entry in report["blocked"].items():
        for split, row in entry["splits"].items():
            add(
                f"| {name} | {split} | {row['rows']:,} | {row['queries_with_gold']:,} "
                f"| {row['queries_fully_resolved']:,} "
                f"| {row['queries_with_no_resolved_gold']:,} |"
            )
    add("")
    add("`test` is listed because a budget has to cover every row that would be")
    add("encoded. No scientific result may be read from those rows.")
    add("")

    add("## What encoding the queries would cost")
    add("")
    add("Priced by padded token slots, not rows.")
    add("")
    add("| dataset | rows | dense slots | splade slots | GPU hours (low-high) | USD at high |")
    add("| --- | ---: | ---: | ---: | ---: | ---: |")
    grand_low = grand_high = 0.0
    for name, entry in report["blocked"].items():
        p = entry["encode_budget"]
        h = p["gpu_hours_total"]
        grand_low += h["low"]
        grand_high += h["high"]
        add(
            f"| {name} | {entry['totals']['rows']:,} "
            f"| {p['dense']['padded_token_slots']:,} "
            f"| {p['splade']['padded_token_slots']:,} "
            f"| {h['low']:.3f}-{h['high']:.3f} | ${p['usd_high']:.2f} |"
        )
    add(f"| **both** | | | | **{grand_low:.3f}-{grand_high:.3f}** "
        f"| **${grand_high * ASSUMED_GPU_USD_PER_HOUR:.2f}** |")
    add("")
    add(f"Hard ceiling to declare: {report['hard_ceiling_gpu_hours']} dense GPU hours.")
    add("Abort if the measured rate exceeds the high estimate by more than 2x.")
    add("")

    add("## For comparison, the three datasets that already have a query view")
    add("")
    add("| dataset | canonical queries | dense | splade |")
    add("| --- | ---: | ---: | ---: |")
    for name, row in report["present"].items():
        add(f"| {name} | {row['queries']:,} | {row['dense']:,} | {row['splade']:,} |")
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
        "format": "m3a_query_view_feasibility/1",
        "phase": "M3A-COMPILATION",
        "generated_by": "scripts/m3a_query_view_feasibility.py",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "package_root": str(args.root),
        "access_mode": "READ_ONLY",
        "decides": "whether blocker 1 can be cleared, and what clearing it would cost",
        "builds_nothing": True,
        "hard_ceiling_gpu_hours": 2.0,
        "blocked": {},
        "present": {},
    }

    for name, spec in BLOCKED.items():
        canonical_dir = args.root / "data" / "canonical" / spec["canonical"]
        print(f"[{name}] reading canonical node ids from {spec['canonical']} ...", flush=True)
        node_ids = canonical_node_ids(canonical_dir)

        splits: dict[str, Any] = {}
        token_parts: list[np.ndarray] = []
        for split in SPLITS:
            path = args.root / spec["source"] / f"{split}.jsonl"
            if not path.exists():
                continue
            print(f"[{name}/{split}] {path.stat().st_size / 2**20:,.1f} MiB ...", flush=True)
            row = measure_split(path, spec, node_ids)
            token_parts.append(row.pop("_tokens"))
            splits[split] = row

        totals = {
            key: sum(row[key] for row in splits.values())
            for key in (
                "rows",
                "queries_with_gold",
                "queries_using_fallback_field",
                "gold_ids_total",
                "gold_ids_resolved",
                "gold_ids_unresolved",
                "gold_ids_with_unexpected_prefix",
                "queries_fully_resolved",
                "queries_with_no_resolved_gold",
            )
        }
        totals["gold_resolution_rate"] = (
            round(totals["gold_ids_resolved"] / totals["gold_ids_total"], 6)
            if totals["gold_ids_total"]
            else None
        )

        all_tokens = np.concatenate(token_parts) if token_parts else np.zeros(0, dtype=np.int64)
        report["blocked"][name] = {
            "source": spec["source"],
            "canonical": spec["canonical"],
            "canonical_nodes": len(node_ids),
            "rewrite": list(spec["rewrite"]),
            "strip_zero_padding": bool(spec.get("strip_zero_padding", False)),
            "splits": splits,
            "totals": totals,
            "encode_budget": price(all_tokens),
        }

    for name in PRESENT:
        directory = args.root / "data" / "canonical" / name
        manifest = json.loads((directory / "query_manifest.json").read_text(encoding="utf-8"))
        dense = json.loads(
            (directory / "encodings" / "dense" / "queries" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        splade = json.loads(
            (directory / "encodings" / "splade" / "queries" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        report["present"][name] = {
            "queries": manifest["total"],
            "dense": dense["n_items"],
            "splade": splade["n_items"],
            "splits": {k: v["n"] for k, v in manifest["splits"].items()},
            # The convention the blocked pair would have to follow. hotpotqa
            # records gold_available=false on test rather than pretending the
            # split has gold, which is the same distinction this script draws
            # between a query with no gold and a query whose gold did not map.
            "gold_policy": manifest.get("gold_policy"),
        }

    report["measured_cost"] = measured_cost(started)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    DOC_PATH.write_text(render(report), encoding="utf-8")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
