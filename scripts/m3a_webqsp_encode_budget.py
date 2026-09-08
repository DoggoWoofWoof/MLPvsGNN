"""M3A item 6, first half: the compute record the WebQSP encode may not start without.

Section J of the adoption contract requires jobs, CPU hours, GPU hours,
walltime, storage, expected cost, a hard ceiling and abort criteria to be filed
before an expensive step runs. Decision 1 additionally requires the number of
missing embeddings, the encoder identity, dimension and batch size.

The workload half is measured here from the shipped parquet. It is measured in
PADDED TOKEN-SLOTS, not rows, because that is what the GPU actually spends:

  - dense goes through SentenceTransformer.encode, which sorts by length before
    batching, so padding waste is small but the longest documents end up in a
    batch TOGETHER -- the peak batch, not the mean, is what decides whether the
    job runs at all;
  - splade batches in file order with padding=True and a 256-token cap, so its
    per-batch cost is the running max within each consecutive group of 32.

Both are simulated below exactly as the encoder does them. Pricing by rows would
have made the two candidate text representations look identical, which they are
not: one is several times the token volume of the other and has a tail four
orders of magnitude longer.

The rate half cannot be measured on this box -- it is CPU-only and the encoder
is a 1.5B model -- so throughput is a declared assumption with its arithmetic
shown, and the protocol calibrates on one shard and re-derives the estimate
before the rest is released.

This script encodes nothing. It prices the work.
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
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "m3a"
JSON_PATH = OUT_DIR / "webqsp_encode_budget.json"
DOC_PATH = ROOT / "docs" / "M3A_WEBQSP_ENCODE_BUDGET.md"
DEFAULT_ROOT = Path("C:/Users/Swastik/Desktop/CRAG")

BATCH = 32
SHARD_SIZE = 40000
DENSE_DIM = 1536
CHARS_PER_TOKEN = 4.0

# canonical_encode.py passes max_seq=None for dense, so max_seq_length stays at
# the model's own default. Read from the cached sentence_bert_config.json, not
# assumed: 32768. That is the number that matters -- it is not 256, and the
# name_plus_facts tail runs well past it.
DENSE_MAX_TOKENS = 32768
SPLADE_MAX_TOKENS = 256

ENCODER = {
    "dense": {
        "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "params": 1_500_000_000,
        "precision": "FP16",
        "normalize_embeddings": True,
        "dim": DENSE_DIM,
        "stored_dtype": "float16",
        "document_prefix": "none",
        "query_prefix": "GTE_QINSTR, copied byte-exact from canonical_encode.py",
        "batch": BATCH,
        "max_seq_tokens": DENSE_MAX_TOKENS,
        "max_seq_source": "model default; canonical_encode.py passes max_seq=None",
        "batch_order": "length-sorted by SentenceTransformer.encode",
    },
    "splade": {
        "model": "naver/splade-cocondenser-ensembledistil",
        "params": 110_000_000,
        "pooling": "log(1 + relu(logits)) max-pooled to CSR",
        "vocab": 30522,
        "batch": BATCH,
        "max_seq_tokens": SPLADE_MAX_TOKENS,
        "batch_order": "file order, padding to the longest in each group of 32",
    },
    "shard_size": SHARD_SIZE,
}

# Declared, not measured. Derived from a forward pass costing about 2*params
# FLOPs per token, at an effective 15-45 TFLOP/s on an A10G-class card:
#   dense  2 * 1.5e9 = 3.0e9 FLOP/token -> 5k-15k tokens/s
#   splade 2 * 1.1e8 = 2.2e8 FLOP/token -> 70k-200k tokens/s
# The arithmetic is shown so the assumption can be argued with rather than
# merely believed, and the calibration shard replaces it with a measurement.
ASSUMED_DENSE_TOKENS_PER_GPU_SECOND = {"low": 5000.0, "high": 15000.0}
ASSUMED_SPLADE_TOKENS_PER_GPU_SECOND = {"low": 70000.0, "high": 200000.0}
ASSUMED_GPU_USD_PER_HOUR = 1.10

# A10G / L4 class, 24 GiB. Activation memory for one forward pass through a 28
# layer 1536-hidden FP16 model is roughly 2 bytes * hidden * layers * slots,
# which is the term that decides whether the peak batch fits.
GPU_MEMORY_GIB = 24.0
DENSE_HIDDEN = 1536
DENSE_LAYERS = 28


def token_lengths(path: Path, column: str) -> np.ndarray:
    """Per-row token estimate, as chars/4. Returned as an array so the batching
    the encoder performs can be simulated rather than approximated."""
    import pyarrow.parquet as pq

    chunks = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1 << 18, columns=[column]):
        lengths = [len(value or "") for value in batch.to_pydict()[column]]
        chunks.append(np.asarray(lengths, dtype=np.int64))
    chars = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
    return np.ceil(chars / CHARS_PER_TOKEN).astype(np.int64)


def sorted_batch_slots(tokens: np.ndarray, cap: int, batch: int) -> dict[str, Any]:
    """Dense: sentence-transformers sorts by length, then batches. Padded slots
    are therefore the sum over batches of (batch size * longest in that batch),
    and the peak batch is the largest of those -- the OOM predictor."""
    capped = np.minimum(tokens, cap)
    order = np.sort(capped)[::-1]
    n = order.size
    starts = np.arange(0, n, batch)
    per_batch = np.array(
        [order[start] * min(batch, n - start) for start in starts], dtype=np.int64
    )
    return {
        "padded_token_slots": int(per_batch.sum()),
        "peak_batch_slots": int(per_batch.max()) if per_batch.size else 0,
        "peak_batch_longest_tokens": int(order[0]) if n else 0,
        "n_batches": int(per_batch.size),
        "padding_overhead_ratio": round(float(per_batch.sum() / max(int(capped.sum()), 1)), 4),
    }


def file_order_batch_slots(tokens: np.ndarray, cap: int, batch: int) -> dict[str, Any]:
    """Splade: no sorting, padding=True per consecutive group of 32."""
    capped = np.minimum(tokens, cap)
    n = capped.size
    pad = (-n) % batch
    grid = np.concatenate([capped, np.zeros(pad, dtype=np.int64)]).reshape(-1, batch)
    per_batch = grid.max(axis=1) * batch
    return {
        "padded_token_slots": int(per_batch.sum()),
        "peak_batch_slots": int(per_batch.max()) if per_batch.size else 0,
        "n_batches": int(per_batch.size),
        "padding_overhead_ratio": round(float(per_batch.sum() / max(int(capped.sum()), 1)), 4),
    }


def describe(tokens: np.ndarray, cap: int) -> dict[str, Any]:
    capped = np.minimum(tokens, cap)
    return {
        "rows": int(tokens.size),
        "tokens_raw": int(tokens.sum()),
        "tokens_after_cap": int(capped.sum()),
        "mean_tokens": round(float(tokens.mean()), 1) if tokens.size else 0.0,
        "p50_tokens": int(np.percentile(tokens, 50)) if tokens.size else 0,
        "p99_tokens": int(np.percentile(tokens, 99)) if tokens.size else 0,
        "max_tokens": int(tokens.max()) if tokens.size else 0,
        "rows_over_cap": int((tokens > cap).sum()),
        "tokens_discarded_by_cap": int((tokens - capped).sum()),
    }


def activation_gib(slots: int) -> float:
    """Rough peak activation footprint for one forward pass, in GiB."""
    return slots * DENSE_HIDDEN * DENSE_LAYERS * 2 / 2**30


def price(slots: int, rate: dict[str, float]) -> dict[str, Any]:
    slow = slots / rate["low"]
    fast = slots / rate["high"]
    return {
        "gpu_seconds_low": round(fast, 1),
        "gpu_seconds_high": round(slow, 1),
        "gpu_hours_low": round(fast / 3600, 2),
        "gpu_hours_high": round(slow / 3600, 2),
        "usd_low": round(fast / 3600 * ASSUMED_GPU_USD_PER_HOUR, 2),
        "usd_high": round(slow / 3600 * ASSUMED_GPU_USD_PER_HOUR, 2),
    }


def build(root: Path) -> dict[str, Any]:
    v1 = root / "data" / "final_canonical" / "webqsp" / "v1"

    cvt = token_lengths(v1 / "cvt_text.parquet", "text_cvt_record")

    representations = {
        "name_only": ("text_name_only", "entity display name alone"),
        "name_plus_facts": (
            "text_name_plus_facts",
            "entity name plus verbalised incident facts",
        ),
    }

    options: dict[str, Any] = {}
    for name, (column, describes) in representations.items():
        entity = token_lengths(v1 / "entity_text.parquet", column)
        combined = np.concatenate([entity, cvt])

        dense_batches = sorted_batch_slots(combined, DENSE_MAX_TOKENS, BATCH)
        splade_batches = file_order_batch_slots(combined, SPLADE_MAX_TOKENS, BATCH)
        rows = int(combined.size)
        peak_gib = activation_gib(dense_batches["peak_batch_slots"])

        options[name] = {
            "describes": describes,
            "entity_column": column,
            "rows": rows,
            "entity_rows": int(entity.size),
            "cvt_rows": int(cvt.size),
            "dense_tokens": describe(combined, DENSE_MAX_TOKENS),
            "splade_tokens": describe(combined, SPLADE_MAX_TOKENS),
            "dense_batching": dense_batches,
            "splade_batching": splade_batches,
            "dense_peak_activation_gib": round(peak_gib, 1),
            "dense_peak_batch_fits_in_gpu": bool(peak_gib < GPU_MEMORY_GIB),
            "shards": -(-rows // SHARD_SIZE),
            "storage_dense_bytes": rows * DENSE_DIM * 2,
            "estimated_dense": price(
                dense_batches["padded_token_slots"], ASSUMED_DENSE_TOKENS_PER_GPU_SECOND
            ),
            "estimated_splade": price(
                splade_batches["padded_token_slots"], ASSUMED_SPLADE_TOKENS_PER_GPU_SECOND
            ),
        }

    worst = max(options.values(), key=lambda option: option["estimated_dense"]["gpu_hours_high"])
    ceiling = round(worst["estimated_dense"]["gpu_hours_high"] * 2, 1)

    hazards = []
    for name, option in options.items():
        if not option["dense_peak_batch_fits_in_gpu"]:
            hazards.append(
                f"{name}: the dense peak batch is "
                f"{option['dense_batching']['peak_batch_slots']:,} padded token-slots, about "
                f"{option['dense_peak_activation_gib']} GiB of activations against the "
                f"{GPU_MEMORY_GIB} GiB assumed card. SentenceTransformer sorts by length, so "
                f"the longest documents batch with each other and this fires on the FIRST "
                f"batch, deterministically, not at random late in the run. Encoding this "
                f"representation needs an explicit --max-seq or a smaller batch, and both "
                f"depart from the frozen contract, so both are review decisions."
            )

    return {
        "format": "m3a_webqsp_encode_budget_v1",
        "phase": "M3A",
        "item": "6 -- WebQSP missing embedding encode (compute record only; nothing encoded)",
        "generated_by": "scripts/m3a_webqsp_encode_budget.py",
        "package_root": str(root).replace("\\", "/"),
        "decides": "nothing; it prices the work and names what review must settle first",
        "encoder_contract": ENCODER,
        "encoder_contract_source": "src/experiments/canonical_encode.py in the package",
        "missing_embeddings": {
            "node_embeddings_missing": options["name_only"]["rows"],
            "node_embeddings_present": 0,
            "modalities_missing": ["dense", "splade"],
            "query_embeddings_missing": None,
            "query_set_exists": False,
            "query_blocker": (
                "WebQSP v1 ships no queries at all -- there is no queries/ directory, "
                "unlike every other dataset. So the number of missing query embeddings "
                "is not zero and is not a count: the query set does not exist in "
                "canonical form yet. The raw material is present (WebQSP.train.json, "
                "WebQSP.test.json, and RoG train/validation/test parquet shards), but "
                "turning it into a canonical query set with splits is a BUILD, not an "
                "encode, and 'verify is not rebuild' does not authorise it."
            ),
        },
        "text_representation_options": options,
        "representation_choice_is_open": True,
        "representation_note": (
            "The two options are not two prices for the same thing. name_plus_facts "
            "folds incident triples into the node's own text, which moves relational "
            "information into the SEMANTIC channel. QLS-U would then receive graph "
            "structure through its embeddings whether or not a structural feature is "
            "ever declared, and delta_MP would stop isolating what it is meant to "
            "isolate -- the GNN's advantage would already be inside the baseline's "
            "inputs. That is a scientific objection rather than a cost one, and it "
            "points at name_only, but the call belongs to review."
        ),
        "assumptions": {
            "dense_tokens_per_gpu_second": ASSUMED_DENSE_TOKENS_PER_GPU_SECOND,
            "splade_tokens_per_gpu_second": ASSUMED_SPLADE_TOKENS_PER_GPU_SECOND,
            "derivation": (
                "a forward pass costs about 2*params FLOPs per token; at an effective "
                "15-45 TFLOP/s on an A10G-class card that is 5k-15k tokens/s for the "
                "1.5B dense encoder and 70k-200k for the 110M splade encoder"
            ),
            "gpu_usd_per_hour": ASSUMED_GPU_USD_PER_HOUR,
            "chars_per_token": CHARS_PER_TOKEN,
            "gpu_memory_gib": GPU_MEMORY_GIB,
            "status": "DECLARED_NOT_MEASURED",
            "why": (
                "This box is CPU-only, so a local benchmark of a 1.5B encoder would not "
                "predict GPU throughput, and no shard status JSON in the package records "
                "a timing -- they carry rows, sha, dtype and dim only. The calibration "
                "shard replaces the assumption with a measurement before most of the "
                "cost is spent."
            ),
        },
        "hazards": hazards,
        "hard_ceiling": {
            "dense_gpu_hours": ceiling,
            "basis": "twice the slow end of the more expensive representation",
            "rule": "exceeding the ceiling aborts the job; it does not trigger a raise",
        },
        "calibration_protocol": [
            "encode shard 0 only, both modalities, under the frozen contract",
            "measure real padded-token throughput and real storage per shard",
            "re-derive the full estimate from the measurement",
            "file the re-derived estimate",
            "only then release the remaining shards",
        ],
        "abort_criteria": [
            "measured throughput below the low end of the declared range",
            "re-derived total above the hard ceiling",
            "any shard failing the finite/shape check canonical_encode.py already raises",
            "dimension not 1536, or stored dtype not float16",
            "row count of a finished shard not equal to its id list length",
            "an out-of-memory on the length-sorted peak batch",
        ],
        "execution_placement": {
            "where": "Modal, spawned server-side via scripts/spawn_modal_jobs.py",
            "never": "modal run --detach",
            "volume_write_rule": (
                "a write that overwrites an existing file on a result volume is silently "
                "discarded at commit -- unlink first, and check the fetched artifact's "
                "source_commit"
            ),
        },
        "blocking_on_review": [
            "the text representation choice (name_only vs name_plus_facts)",
            "whether building a canonical WebQSP query set is authorised, since it is a build",
            "whether the WebQSP union CWQ graph is the intended candidate universe",
            "any --max-seq or batch override, since both depart from the frozen contract",
        ],
    }


def render(data: dict[str, Any]) -> str:
    options = data["text_representation_options"]

    workload = []
    for name, option in options.items():
        tokens = option["dense_tokens"]
        dense = option["estimated_dense"]
        splade = option["estimated_splade"]
        workload.append(
            f"| `{name}` | {tokens['tokens_after_cap']:,} | "
            f"{option['dense_batching']['padded_token_slots']:,} | "
            f"{dense['gpu_hours_low']}-{dense['gpu_hours_high']} | "
            f"{splade['gpu_hours_low']}-{splade['gpu_hours_high']} | "
            f"${dense['usd_low'] + splade['usd_low']:.2f}-"
            f"${dense['usd_high'] + splade['usd_high']:.2f} |"
        )

    shape = []
    for name, option in options.items():
        tokens = option["dense_tokens"]
        shape.append(
            f"| `{name}` | {tokens['mean_tokens']} | {tokens['p50_tokens']:,} | "
            f"{tokens['p99_tokens']:,} | {tokens['max_tokens']:,} | "
            f"{tokens['rows_over_cap']:,} | "
            f"{option['dense_batching']['peak_batch_slots']:,} | "
            f"{option['dense_peak_activation_gib']} |"
        )

    missing = data["missing_embeddings"]
    ceiling = data["hard_ceiling"]
    assumptions = data["assumptions"]
    dense_rate = assumptions["dense_tokens_per_gpu_second"]
    splade_rate = assumptions["splade_tokens_per_gpu_second"]
    any_option = next(iter(options.values()))

    hazard_block = (
        "\n\n".join(f"> {hazard}" for hazard in data["hazards"])
        if data["hazards"]
        else "None. Every peak batch fits the assumed card."
    )

    return f"""# M3A WebQSP encode budget

Generated by `{data['generated_by']}`. Package root `{data['package_root']}`,
read-only. This document decides {data['decides']}. **Nothing was encoded.**

## What is missing

{missing['node_embeddings_missing']:,} node embeddings, {missing['node_embeddings_present']} present,
both modalities absent. Storage on completion
{any_option['storage_dense_bytes'] / 2**30:.1f} GiB dense across {any_option['shards']} shards of
{data['encoder_contract']['shard_size']:,}, plus a sparse SPLADE matrix.

**Queries are a different problem, not a smaller one.** {missing['query_blocker']}

## The workload, measured

Priced in padded token-slots rather than rows, because that is what the GPU
spends. Pricing by rows makes the two representations look identical; they are
not.

| representation | tokens after cap | padded slots (dense) | dense GPU h | splade GPU h | cost |
|---|---|---|---|---|---|
{chr(10).join(workload)}

### The shape of the tail

Dense truncates at {data['encoder_contract']['dense']['max_seq_tokens']:,} tokens -- the model
default, because `canonical_encode.py` passes `max_seq=None`. Not 256. That
matters, because `SentenceTransformer.encode` sorts by length before batching,
so the longest documents are batched *with each other*.

| representation | mean | p50 | p99 | max | rows over cap | peak batch slots | peak activations (GiB) |
|---|---|---|---|---|---|---|---|
{chr(10).join(shape)}

### Hazards

{hazard_block}

## Which representation

{data['representation_note']}

## The rate, assumed

{assumptions['why']}

Declared: dense {dense_rate['low']:,.0f}-{dense_rate['high']:,.0f} padded tokens/GPU-second,
SPLADE {splade_rate['low']:,.0f}-{splade_rate['high']:,.0f}, at
${assumptions['gpu_usd_per_hour']}/GPU-hour on a {assumptions['gpu_memory_gib']} GiB card.
Status **{assumptions['status']}**. Derivation: {assumptions['derivation']}.

Hard ceiling **{ceiling['dense_gpu_hours']} dense GPU hours** ({ceiling['basis']}).
{ceiling['rule'][0].upper() + ceiling['rule'][1:]}.

### Calibration before commitment

{chr(10).join(f'{i}. {step}' for i, step in enumerate(data['calibration_protocol'], 1))}

### Abort criteria

{chr(10).join(f'- {criterion}' for criterion in data['abort_criteria'])}

### Where it runs

{data['execution_placement']['where']}, never `{data['execution_placement']['never']}`.
{data['execution_placement']['volume_write_rule'][0].upper()
 + data['execution_placement']['volume_write_rule'][1:]}.

## Blocked on review

{chr(10).join(f'- {item}' for item in data['blocking_on_review'])}

Machine-readable form: `outputs/m3a/webqsp_encode_budget.json`.
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
    started = time.perf_counter()

    data = build(args.root)
    data["measured_cost"] = measured_cost(started)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    DOC_PATH.write_text(render(data), encoding="utf-8", newline="\n")
    print(f"wrote {JSON_PATH.relative_to(ROOT)}")
    print(f"wrote {DOC_PATH.relative_to(ROOT)}")
    for hazard in data["hazards"]:
        print(f"HAZARD: {hazard}")


if __name__ == "__main__":
    main()
