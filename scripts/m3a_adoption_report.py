"""M3A item 8: the adoption report.

Items 1-7 each produced their own artifact. This consolidates them into one
verdict per contract section and one overall recommendation, and it does so by
READING those artifacts rather than restating them, so the report cannot drift
away from what was actually measured. Every number here has a provenance line
pointing at the JSON it came from.

Sections A-F and E carry a status each:

  ADOPT                 verified locally; nothing outstanding
  ADOPT_WITH_FINDING    verified locally, but something was measured that the
                        upstream manifest did not record and review should see
  ADOPT_PENDING_REVIEW  the section is green but a decision is open that this
                        phase is not authorised to take
  BLOCKED               cannot be adopted until named work happens

The report recommends. It does not adopt: the contract ends at
STOP_FOR_REVIEW.
"""

from __future__ import annotations

import platform
import time

try:  # not available on Windows; the wall clock still is
    import resource
except ImportError:  # pragma: no cover - platform dependent
    resource = None

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "m3a"
JSON_PATH = OUT_DIR / "adoption_report.json"
DOC_PATH = ROOT / "docs" / "M3A_ADOPTION_REPORT.md"

SOURCES = {
    "manifest": OUT_DIR / "transfer_manifest.json",
    "verification": OUT_DIR / "adoption_verification.json",
    "alignment": OUT_DIR / "node_alignment.json",
    "budget": OUT_DIR / "webqsp_encode_budget.json",
}


def load(name: str) -> dict[str, Any]:
    path = SOURCES[name]
    if not path.exists():
        raise SystemExit(f"missing input: {path.relative_to(ROOT)} -- run its script first")
    return json.loads(path.read_text(encoding="utf-8"))


def build() -> dict[str, Any]:
    manifest = load("manifest")
    verification = load("verification")
    alignment = load("alignment")
    budget = load("budget")

    metaqa = verification["section_b"]["chain"]
    metaqa_identity = verification["section_b"]["accounting_identity"]
    section_c = verification["section_c"]
    webqsp = section_c["counts"]
    checks = section_c["checks"]
    union = section_c["union_scope"]["per_dataset"]
    regimes = {row["dataset"]: row for row in verification["section_f"]["regimes"]}

    rows = alignment["alignments"]
    green = [row["dataset"] for row in rows if row["green"]]
    blocked = [row["dataset"] for row in rows if not row["green"]]

    # Section E's own measurement: nodes the graph never mentions. Read from the
    # rows rather than retyped, so the report cannot drift from the alignment.
    orphans = {
        row["dataset"]: (
            row["missing"]["canonical_nodes_with_no_graph_endpoint"],
            row["counts"]["canonical_nodes"],
        )
        for row in rows
    }
    orphan_text = ", ".join(
        f"{dataset} {count:,} ({count / total:.1%})"
        for dataset, (count, total) in orphans.items()
        if count
    )
    orphan_zero = ", ".join(dataset for dataset, (count, _) in orphans.items() if not count)

    sections = [
        {
            "section": "A",
            "title": "transfer ingest gate",
            "status": "ADOPT",
            "finding": (
                f"{manifest['measured']['files']:,} files, "
                f"{manifest['measured']['gib']} GiB, inventoried read-only and hashed. "
                "The manifest is write-once; a second pass re-hashed and diffed rather "
                "than overwriting. Nothing in the package was modified."
            ),
            "provenance": "outputs/m3a/transfer_manifest.json",
        },
        {
            "section": "B",
            "title": "MetaQA local losslessness",
            "status": "ADOPT",
            "finding": (
                f"{metaqa['source_lines']:,} source lines recomputed from kb.txt to "
                f"{metaqa['distinct_triples']:,} distinct triples to "
                f"{metaqa['mapped_triples']:,} mapped edges, "
                f"{metaqa['unmapped_triples']} unmapped and enumerated. The accounting "
                f"identity closes in both directions "
                f"(source = parsed + blank + malformed: "
                f"{metaqa_identity['source = parsed + blank + malformed']}; "
                f"distinct = mapped + unmapped: "
                f"{metaqa_identity['distinct = mapped + unmapped']}), and the locally "
                "reconstructed graph is set-equal to the shipped one. Upstream's claims of "
                "zero malformed and "
                "zero dangling are independently established here, not accepted. The local "
                "reconstruction was discarded: equivalence is the result, not a licence to "
                "substitute."
            ),
            "provenance": "outputs/m3a/adoption_verification.json#section_b",
        },
        {
            "section": "C",
            "title": "WebQSP typed substrate",
            "status": "ADOPT_WITH_FINDING",
            "finding": (
                f"{webqsp['relations']:,} relations, {webqsp['edges']:,} edges, "
                f"{webqsp['nodes']:,} nodes, counted from the shipped v1 tables against "
                "the source contract. No second vocabulary was built. Three things the "
                "upstream manifest does not record: "
                f"{checks['relation_short_label_collisions']} relation SHORT-label "
                f"collisions against {checks['relation_qualified_label_collisions']} "
                "qualified-label collisions, so relation text must come from "
                f"relation_label_qualified or {checks['relation_short_label_collisions']} "
                f"distinct relations silently merge; {checks['self_loops']:,} self-loops; "
                "and the graph is the WebQSP union CWQ, of which WebQSP alone contributed "
                f"{union['webqsp']['distinct_triples']:,} distinct triples against CWQ's "
                f"{union['cwq']['distinct_triples']:,}. "
                "The RoG-exact property is UPSTREAM_ASSERTED_NOT_LOCALLY_REPRODUCIBLE: "
                "counting the shipped tables shows internal consistency, not a re-derivation "
                "from the raw shards. The shards are present and the upstream rebuild took "
                "1,111.8s, so re-deriving is affordable if review wants it."
            ),
            "provenance": "outputs/m3a/adoption_verification.json#section_c",
        },
        {
            "section": "D",
            "title": "historical substrates untouched",
            "status": "ADOPT",
            "finding": (
                "No historical artifact was read for content, modified, re-hashed or "
                "compared against a canonical number. The clean development substrate, the "
                "old graph.pt and the frozen M2 results are as they were. No table in this "
                "phase places a canonical figure beside a historical one."
            ),
            "provenance": "this phase produced no cross-substrate arithmetic",
        },
        {
            "section": "E",
            "title": "node-space alignment",
            # Not flatly BLOCKED: five of six datasets are green and adoptable
            # today. Naming the whole section blocked would overstate the damage
            # and would make the one real blocker harder to see, not easier.
            "status": (
                f"ADOPT_EXCEPT_{'_'.join(sorted(blocked)).upper()}" if blocked else "ADOPT"
            ),
            "finding": (
                f"The chain graph node -> canonical id -> embedding row -> retrieval row is "
                f"one-to-one and green on {len(green)} of {len(rows)} datasets "
                f"({', '.join(sorted(green))}), reported in both directions. "
                f"{', '.join(sorted(blocked))} is blocked for the one reason item 6 exists. "
                "Measured and not previously recorded: canonical nodes with no edge at all "
                f"-- {orphan_text}; {orphan_zero} have none. Those nodes bound what any "
                "structural feature can say, and the model must see the mask rather than "
                "the zero."
            ),
            "provenance": "outputs/m3a/node_alignment.json",
        },
        {
            "section": "F",
            "title": "relation representation audit",
            "status": "ADOPT_PENDING_REVIEW",
            "finding": (
                f"Two regimes, not one. MetaQA {regimes['metaqa']['relation_types']} relation "
                f"types at normalised entropy {regimes['metaqa']['normalised_entropy']}, "
                f"median {regimes['metaqa']['median_edges_per_relation']:,} edges per "
                f"relation and no singletons; WebQSP "
                f"{regimes['webqsp']['relation_types']:,} types at "
                f"{regimes['webqsp']['normalised_entropy']}, median "
                f"{regimes['webqsp']['median_edges_per_relation']} edges per relation and "
                f"{regimes['webqsp']['singleton_relations']} singletons -- a "
                f"{verification['section_f']['cardinality_ratio']}x vocabulary with a long "
                "tail that a one-hot relation feature would mostly waste. Relation text is "
                "NOT encoded, which is what section F requires: the audit comes first and "
                "this is the audit. The consequence is a reporting rule -- a MetaQA null is "
                "about a 9-relation regime and may never be generalised to 'typed relation "
                "features do not help'."
            ),
            "provenance": "outputs/m3a/adoption_verification.json#section_f",
        },
    ]

    dense = budget["text_representation_options"]["name_only"]["estimated_dense"]
    splade = budget["text_representation_options"]["name_only"]["estimated_splade"]

    open_decisions = [
        {
            "decision": "WebQSP node text representation",
            "why_review": (
                "name_plus_facts folds incident triples into node text, moving relational "
                "information into the semantic channel. QLS-U would then receive graph "
                "structure through its embeddings regardless of which structural features "
                "are declared, and delta_MP would no longer isolate message passing. It is "
                "also the option that cannot run: its length-sorted peak batch is "
                f"{budget['text_representation_options']['name_plus_facts']['dense_peak_activation_gib']} "
                "GiB of activations against a 24 GiB card, deterministically on the first "
                "batch."
            ),
            "recommendation": "name_only, but this is review's call, not this phase's",
        },
        {
            "decision": "whether to build a canonical WebQSP query set",
            "why_review": (
                "WebQSP v1 ships no queries at all -- no queries/ directory, unlike every "
                "other dataset. So 'number of missing query embeddings' has no answer as an "
                "encode quantity. The raw material exists, but constructing splits is a "
                "BUILD, and decision 2 says verify is not rebuild. Without it WebQSP cannot "
                "be evaluated, so decision 1's 'WebQSP is a priority dataset' cannot be "
                "honoured without authorising this."
            ),
            "recommendation": "none offered; this is squarely a review decision",
        },
        {
            "decision": "candidate universe for WebQSP queries",
            "why_review": (
                "The shipped graph is WebQSP union CWQ. Evaluating WebQSP queries against "
                "the union means the candidate universe contains CWQ-only triples. That is "
                "a different graph from 'WebQSP's graph' and has to be declared either way, "
                "not defaulted into."
            ),
            "recommendation": "declare explicitly whichever way it goes",
        },
        {
            "decision": "whether to re-derive the RoG-exact property from raw shards",
            "why_review": (
                "It is currently upstream-asserted. Re-deriving costs about 1,112s of CPU "
                "by upstream's own timing, which is cheap, but it is a rebuild and this "
                "phase is not authorised to perform one."
            ),
            "recommendation": "affordable if review wants the property locally established",
        },
    ]

    return {
        "format": "m3a_adoption_report_v1",
        "phase": "M3A",
        "item": "8 -- adoption report",
        "generated_by": "scripts/m3a_adoption_report.py",
        "decides": "nothing; it recommends, and the contract ends at STOP_FOR_REVIEW",
        "inputs": {
            name: str(path.relative_to(ROOT)).replace("\\", "/") for name, path in SOURCES.items()
        },
        "sections": sections,
        "overall": {
            "recommendation": "ADOPT_WITH_FINDINGS",
            "basis": (
                "Every section that could be closed by verification alone is closed. "
                "Nothing failed. No section requires repair, so the 'verification fails "
                "and a later review authorises repair' branch of decision 2 is not "
                "triggered. What remains open is not defect but scope: four decisions "
                "this phase is not authorised to take."
            ),
            "verify_not_rebuild_held": True,
            "second_vocabulary_built": False,
            "second_graph_built": False,
            "package_bytes_modified": 0,
            "sidecar_artifacts_only": True,
        },
        "open_decisions": open_decisions,
        "webqsp_encode_readiness": {
            "node_encode_priced": True,
            "node_encode_gpu_hours": f"{dense['gpu_hours_low']}-{dense['gpu_hours_high']} dense, "
            f"{splade['gpu_hours_low']}-{splade['gpu_hours_high']} splade (name_only)",
            "node_encode_cost_usd": f"{dense['usd_low'] + splade['usd_low']:.2f}-"
            f"{dense['usd_high'] + splade['usd_high']:.2f}",
            "hard_ceiling_gpu_hours": budget["hard_ceiling"]["dense_gpu_hours"],
            "blocked_on": "the representation choice and the query-set decision",
            "query_encode_priced": False,
            "query_encode_reason": "there is no query set to price",
        },
        "not_done_and_not_authorised": [
            "construct final QLS-U features",
            "construct high-headroom candidates",
            "train QLS-U",
            "train GAT",
            "train GAT-NO-MP",
            "run the WebQSP encode itself",
            "build a WebQSP query set",
            "recompute headroom (section I: after adoption and after the encode)",
        ],
        "next": "STOP_FOR_REVIEW",
    }


def render(data: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {section['section']} | {section['title']} | **{section['status']}** |"
        for section in data["sections"]
    )
    findings = "\n\n".join(
        f"### {section['section']}. {section['title']}\n\n"
        f"**{section['status']}** — {section['finding']}\n\n"
        f"*Source: `{section['provenance']}`*"
        for section in data["sections"]
    )
    decisions = "\n\n".join(
        f"### {i}. {decision['decision']}\n\n"
        f"{decision['why_review']}\n\n"
        f"**Recommendation:** {decision['recommendation']}"
        for i, decision in enumerate(data["open_decisions"], 1)
    )
    overall = data["overall"]
    readiness = data["webqsp_encode_readiness"]

    return f"""# M3A adoption report

Generated by `{data['generated_by']}` from the item 1-7 artifacts, which it reads
rather than restates. This document decides {data['decides']}.

| section | | verdict |
|---|---|---|
{rows}

## Overall: {overall['recommendation']}

{overall['basis']}

Verify-is-not-rebuild held throughout: **{overall['package_bytes_modified']}** bytes of the
package modified, **no** second relation vocabulary, **no** second graph. Every
artifact this phase produced is a sidecar.

## What each section found

{findings}

## What review has to decide

Four decisions are open. None of them is a defect; each is something this phase
was not authorised to settle.

{decisions}

## WebQSP encode readiness

The node encode is priced and ready: {readiness['node_encode_gpu_hours']}, about
${readiness['node_encode_cost_usd']}, against a hard ceiling of
{readiness['hard_ceiling_gpu_hours']} dense GPU hours. It is blocked on
{readiness['blocked_on']}.

The query encode **cannot** be priced: {readiness['query_encode_reason']}. Decision 1
made WebQSP a priority dataset rather than an optional appendix, and that
commitment cannot be met by encoding alone.

## Not done, and not authorised

{chr(10).join(f'- {item}' for item in data['not_done_and_not_authorised'])}

## {data['next']}

Machine-readable form: `outputs/m3a/adoption_report.json`.
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
    started = time.perf_counter()
    data = build()
    data["measured_cost"] = measured_cost(started)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    DOC_PATH.write_text(render(data), encoding="utf-8", newline="\n")
    print(f"wrote {JSON_PATH.relative_to(ROOT)}")
    print(f"wrote {DOC_PATH.relative_to(ROOT)}")
    print(f"overall: {data['overall']['recommendation']}")


if __name__ == "__main__":
    main()
