"""Render docs/M3A_HEADROOM_REPORT.md from outputs/m3a/headroom/<dataset>.json.

Every number in the report is copied from the runner's output; nothing is
computed here except the knee reading, which applies the rule declared in
configs/m3a_headroom.yaml#graph_regimes.knee_rule (smallest pool whose
fraction_of_attainable@5 is at least 0.90) and is labelled as a reading.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "m3a" / "headroom"
CONFIG = ROOT / "configs" / "m3a_headroom.yaml"
COMPILATION = ROOT / "configs" / "m3a_compilation.yaml"
REPORT = ROOT / "docs" / "M3A_HEADROOM_REPORT.md"
ORDER = ["metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp"]
KNEE = 0.90


def f4(x) -> str:
    return "–" if x is None else f"{x:.4f}"


def f1(x) -> str:
    return "–" if x is None else f"{x:.1f}"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def load() -> dict[str, dict]:
    data = {}
    for name in ORDER:
        p = OUT_DIR / f"{name}.json"
        if p.exists():
            data[name] = json.loads(p.read_text(encoding="utf-8"))
    return data


def population_table(data: dict[str, dict]) -> str:
    rows = []
    for name, d in data.items():
        ev = d["populations"]["eval"]
        cc = d["corpus_ceiling_column"]
        row = cc.get("eval_split_row") or {}
        legacy = d["populations"].get("legacy_continuity")
        legacy_txt = "–"
        if legacy:
            comp = ", ".join(f"{k} {v:,}" for k, v in legacy["by_split"].items())
            legacy_txt = f"{legacy['queries']:,} ({comp})"
        ref_level = row.get("reference_level_corpus_ceiling", row.get("reference_level_in_file"))
        query_level = row.get("query_level_corpus_ceiling", row.get("query_level_any_gold"))
        of_answerable = row.get("query_level_corpus_ceiling_of_answerable")
        classes = cc.get("gold_coverage_classes")
        note = "text corpus: gold resolves by construction" if cc.get("corpus_ceiling") == 1.0 else (
            f"KB corpus: {row.get('gold_refs_resolved', 0):,} of {row.get('gold_refs_upstream', 0):,} gold references resolve; "
            f"of answerable {f4(of_answerable)}; classes {json.dumps(classes)}" if classes else "")
        rows.append([
            name, d["eval_split"], f"{ev['queries']:,}", str(ev["zero_gold_excluded"]), f"{ev['queries'] - ev['zero_gold_excluded']:,}",
            f4(ref_level), f4(query_level), note, legacy_txt,
        ])
    return table(["dataset", "eval split", "queries in split", "zero-gold excluded", "queries scored", "corpus ceiling, reference level", "corpus ceiling, query level (any gold)", "note", "LEGACY_CONTINUITY rows (NOT a population)"], rows)


def depth_table(d: dict) -> str:
    rows_by_depth: dict[int, dict[str, float]] = {}
    for r in d["R4_depth_curve"]:
        if r["population"] != "eval":
            continue
        k = int(r["depth"])
        model = r["pool"].rsplit("_top", 1)[0]
        rows_by_depth.setdefault(k, {})[model] = r[f"recall_ceiling@{k}"]
    rows = []
    for k in sorted(rows_by_depth):
        m = rows_by_depth[k]
        rows.append([str(k), f4(m.get("dense")), f4(m.get("splade")), f4(m.get("equal_rrf"))])
    return table(["K (= prefix depth)", "dense recall@K", "splade recall@K", "equal-RRF recall@K"], rows)


def retrieval_table(d: dict, population: str) -> str:
    rows = []
    for r in d["rows"]:
        if r["population"] != population or r["regime"] != "RETRIEVAL":
            continue
        rows.append([
            r["pool"], f"{r['candidates_mean']:.1f} ({r['candidates_p95']:.0f})",
            f4(r["recall_ceiling@1"]), f4(r["recall_ceiling@5"]), f4(r["recall_ceiling@20"]),
            f4(r.get("fraction_of_attainable@5")), f4(r["any_gold_at_pool"]), f4(r["all_gold_at_pool"]), f4(r["full_coverage_ceiling@20"]),
        ])
    return table(["pool", "candidates mean (p95)", "pool_ceiling@1", "pool_ceiling@5", "pool_ceiling@20", "fraction of attainable@5", "ANY", "ALL", "FullCov ceiling@20"], rows)


def regime_table(d: dict, population: str) -> str:
    rows = []
    for r in d["rows"]:
        if r["population"] != population or r["regime"] == "RETRIEVAL":
            continue
        rows.append([
            r["regime"], r["setting"], r["base_pool"].replace("equal_rrf_budget_", "rrf@"),
            f"{r['candidates_mean']:.1f} ({r['candidates_p95']:.0f})",
            f4(r["recall_ceiling@5"]), f4(r["recall_ceiling@20"]), f4(r.get("fraction_of_attainable@5")),
            f4(r["any_gold_at_pool"]), f4(r["full_coverage_ceiling@20"]),
            str(r["movement_missing_golds_recovered"]), f4(r["movement_recovered_gold_per_added_candidate"]),
        ])
    return table(["regime", "setting", "base pool", "candidates mean (p95)", "pool_ceiling@5", "pool_ceiling@20", "fraction of attainable@5", "ANY", "FullCov ceiling@20", "missing golds recovered", "recovered per added candidate"], rows)


def knee_rows(d: dict, name: str, sota_substrate: dict) -> list[list[str]]:
    """Per regime: the smallest pool (by mean candidates) with fraction_of_attainable@5 >= KNEE,
    its gain over its own base pool, and the best cell of the regime. A reading of the declared rule."""
    out = []
    cells = [r for r in d["rows"] if r["population"] == "eval"]
    base_by_name = {r["pool"]: r for r in cells if r["regime"] == "RETRIEVAL"}
    for regime in ("RETRIEVAL", "STRUCT", "NER", "KNN", "FULL"):
        pool_cells = [r for r in cells if r["regime"] == regime]
        if not pool_cells:
            continue
        reaching = [r for r in pool_cells if (r.get("fraction_of_attainable@5") or 0.0) >= KNEE]
        best = max(pool_cells, key=lambda r: (r["recall_ceiling@5"], -r["candidates_mean"]))
        k = min(reaching, key=lambda r: r["candidates_mean"]) if reaching else None
        if k is not None:
            knee = f"{k['pool']} — {k['candidates_mean']:.0f} cand., ceiling@5 {k['recall_ceiling@5']:.4f} ({k['fraction_of_attainable@5']:.3f} of attainable)"
        else:
            knee = "not reached in the declared pools"
        gain, ratio = "–", "–"
        ref = k if k is not None else best
        if regime != "RETRIEVAL" and ref is not None and ref["base_pool"] in base_by_name:
            b = base_by_name[ref["base_pool"]]
            gain = f"{ref['recall_ceiling@5'] - b['recall_ceiling@5']:+.4f} over {b['pool'].replace('equal_rrf_budget_', 'rrf@')}"
            ratio = f"{ref['candidates_mean'] / max(b['candidates_mean'], 1e-9):.2f}×"
        best_txt = f"{best['pool']}: {best['recall_ceiling@5']:.4f} ({best['fraction_of_attainable@5']:.3f}) at {best['candidates_mean']:.0f} cand."
        tag = "SOTA-matched substrate" if sota_substrate.get(name) == regime else ""
        out.append([name, regime, knee, gain, ratio, best_txt, tag])
    return out


def reach_table(d: dict) -> str:
    rows = []
    for regime, r in (d.get("reachability") or {}).items():
        b = r["buckets"]
        rows.append([
            regime, r["population"], f"{r['queries_scanned']:,}", f"{r['queries_with_missing_gold']:,}", f"{r['missing_golds_total']:,}",
            f4(b["missing_golds_reachable_within_1_fraction"]), f4(b["missing_golds_reachable_within_2_fraction"]), f4(b["missing_golds_reachable_within_3_fraction"]),
            f4(b["missing_golds_beyond_3_hops_or_unreachable_fraction"]), str(r["frontier_capped_queries"]), f1(r["seconds"]),
        ])
    return table(["regime (undirected)", "population", "queries walked", "with missing gold", "missing golds", "within 1 hop", "within 2", "within 3", "beyond 3 / unreachable", "frontier-capped queries", "seconds"], rows)


def oracle_block(d: dict) -> str:
    o = d.get("oracle_topic_entity_exposure")
    if not o:
        return ""
    lines = [f"Status: **{o.get('status')}** — the exposure the KB-QA lineage had, measured on our served STRUCT graph; never a seed, never a pool."]
    scalar = [k for k in o if k not in ("status", "seconds") and not isinstance(o[k], (dict, list))]
    lines.append(table(["setting", "value"], [[k, f4(o[k]) if isinstance(o[k], float) else str(o[k])] for k in scalar]))
    hop_keys = [k for k in o if isinstance(o[k], dict) and k.startswith("within_")]
    rows = []
    for k in hop_keys:
        h = o[k]
        rows.append([
            k.replace("within_", "").replace("_hops", "").replace("_hop", ""),
            f"{h['queries']:,}",
            f4(h["gold_coverage_reference_level_macro"]),
            f4(h["queries_any_gold_covered"]),
            f4(h["queries_all_gold_covered"]),
            f"{h['subgraph_nodes_mean']:,.0f}",
            f"{h['subgraph_nodes_p50']:,.0f}",
            f"{h['subgraph_nodes_p95']:,.0f}",
            f"{h['subgraph_nodes_max']:,}",
        ])
    lines.append(table(
        ["hops from the assigned topic entities", "queries", "gold coverage (reference level, macro)", "queries with any gold covered", "queries with all gold covered", "subgraph nodes mean", "p50", "p95", "max"],
        rows,
    ))
    return "\n\n".join(lines)


def pick(d: dict, pool: str, population: str = "eval") -> dict:
    for r in d["rows"]:
        if r["population"] == population and r["pool"] == pool:
            return r
    raise KeyError(pool)


def reach_frac(d: dict, regime: str, hops: int) -> float | None:
    r = (d.get("reachability") or {}).get(regime)
    if not r:
        return None
    return r["buckets"][f"missing_golds_reachable_within_{hops}_fraction"]


def readings(data: dict[str, dict], sota_substrate: dict) -> list[str]:
    """Numbers-first readings, one paragraph per dataset. Every figure is pulled
    from the output; the prose only says which cell it is."""
    parts = ["\n## Readings (per dataset; ceilings, not results)\n"]
    parts.append(
        "Each paragraph names the cells behind it. `attainable` is `recall_ceiling_perfect_retrieval@5`, the most a K=5 "
        "cut-off can return given the gold count; `of attainable` is the fraction of that the pool reaches.\n"
    )
    for name, d in data.items():
        ret_best = max((r for r in d["rows"] if r["population"] == "eval" and r["regime"] == "RETRIEVAL"), key=lambda r: r["recall_ceiling@5"])
        graph_cells = [r for r in d["rows"] if r["population"] == "eval" and r["regime"] != "RETRIEVAL"]
        g_best = max(graph_cells, key=lambda r: r["recall_ceiling@5"])
        sub = sota_substrate.get(name)
        sub_cells = [r for r in graph_cells if r["regime"] == sub]
        sub_best = max(sub_cells, key=lambda r: r["recall_ceiling@5"]) if sub_cells else None
        rrf200 = pick(d, "equal_rrf_budget_200")
        dense200 = pick(d, "dense_top200")
        attain = ret_best["recall_ceiling_perfect_retrieval@5"]
        lines = [
            f"**{name}** (`{d['eval_split']}`, {d['populations']['eval']['queries'] - d['populations']['eval']['zero_gold_excluded']:,} queries scored; {ret_best['golds_per_query_mean']:.2f} golds per query, attainable@5 {attain:.4f}). "
            f"Retrieval alone: the best inherited pool is `{ret_best['pool']}` at pool_ceiling@5 {ret_best['recall_ceiling@5']:.4f} "
            f"({ret_best['fraction_of_attainable@5']:.3f} of attainable, {ret_best['candidates_mean']:.0f} candidates); "
            f"`dense_top200` {dense200['recall_ceiling@5']:.4f}, `equal_rrf_budget_200` {rrf200['recall_ceiling@5']:.4f}. "
            f"Best declared graph cell: `{g_best['pool']}` {g_best['recall_ceiling@5']:.4f} ({g_best['fraction_of_attainable@5']:.3f} of attainable) "
            f"at {g_best['candidates_mean']:.0f} candidates, {g_best['movement_missing_golds_recovered']:,} missing golds recovered over its base."
        ]
        if sub_best is not None:
            lines.append(
                f" On the SOTA-matched substrate ({sub}): best cell `{sub_best['pool']}` {sub_best['recall_ceiling@5']:.4f} "
                f"({sub_best['fraction_of_attainable@5']:.3f} of attainable) at {sub_best['candidates_mean']:.0f} candidates."
            )
        elif sub == "none":
            lines.append(" No graph-retrieval SOTA exists for this dataset; the retrieval pools are the comparison.")
        w2s, w3s = reach_frac(d, "STRUCT", 2), reach_frac(d, "STRUCT", 3)
        w2f, w3f = reach_frac(d, "FULL", 2), reach_frac(d, "FULL", 3)
        if w2s is not None:
            rf = d["reachability"]["FULL"]
            lines.append(
                f" Of the {rf['missing_golds_total']:,} golds the `frozen_union` pool misses ({rf['queries_with_missing_gold']:,} of {rf['queries_scanned']:,} walked queries, {rf['population']}), "
                f"{w2s:.3f} lie within 2 undirected STRUCT hops of the seeds and {w3s:.3f} within 3; on FULL {w2f:.3f} and {w3f:.3f}."
            )
        o = d.get("oracle_topic_entity_exposure")
        if o:
            o1, o2 = o["within_1_hops"], o["within_2_hops"]
            lines.append(
                f" Oracle topic-entity exposure (diagnostic): within 1 STRUCT hop of the assigned topic entities gold coverage is {o1['gold_coverage_reference_level_macro']:.3f} "
                f"(mean subgraph {o1['subgraph_nodes_mean']:,.0f} nodes, p50 {o1['subgraph_nodes_p50']:,.0f}); within 2 hops {o2['gold_coverage_reference_level_macro']:.3f} "
                f"(mean {o2['subgraph_nodes_mean']:,.0f} nodes, p50 {o2['subgraph_nodes_p50']:,.0f}, p95 {o2['subgraph_nodes_p95']:,.0f}). "
                f"That is the exposure the KB-QA lineage had; our seeds are retrieved, and the pools above never use it."
            )
        parts.append("".join(lines) + "\n")
    return parts


LINEAGE = {
    "metaqa": "KB-QA (NuTrea / ReaRev / GNN-RAG): assigned topic entity, subgraph of as many hops as the question (1-3)",
    "webqsp": "KB-QA (NuTrea / ReaRev / GNN-RAG): assigned topic entities, 2-hop subgraph",
    "hotpotqa": "GraphER: 200-candidate scope over a query-induced corpus (2,000 sampled dev queries), PR@K",
    "2wiki": "GraphER: 200-candidate scope over a query-induced corpus, PR@K",
    "musique": "GraphER: 200-candidate scope over a query-induced corpus, PR@K",
    "squad": "none (no graph-retrieval SOTA)",
}


def exposure_table(data: dict[str, dict], sota_substrate: dict) -> str:
    """The SOTA exposure beside our inference-safe counterpart, per dataset. For the KB lineage the
    SOTA exposure is measured on our STRUCT graph (oracle diagnostic); for GraphER it is the 200-candidate
    scope, and our counterpart is the SOTA-matched-substrate cell nearest 200 candidates."""
    rows = []
    for name, d in data.items():
        sub = sota_substrate.get(name)
        cells = [r for r in d["rows"] if r["population"] == "eval"]
        o = d.get("oracle_topic_entity_exposure")
        if o:
            o2 = o["within_2_hops"]
            sota_txt = (f"2-hop topic-entity subgraph on our STRUCT graph: gold coverage {o2['gold_coverage_reference_level_macro']:.3f} "
                        f"(all golds {o2['queries_all_gold_covered']:.3f}) at {o2['subgraph_nodes_mean']:,.0f} nodes mean, p50 {o2['subgraph_nodes_p50']:,.0f}")
        elif sub == "none":
            sota_txt = "–"
        else:
            sota_txt = "200 candidates per query over a corpus induced from 2,000 sampled dev queries (corpus subsetting by query -- forbidden for us)"
        sub_cells = [r for r in cells if r["regime"] == sub]
        if sub_cells:
            near = min(sub_cells, key=lambda r: (abs(r["candidates_mean"] - 200), -r["recall_ceiling@5"]))
            best = max(sub_cells, key=lambda r: r["recall_ceiling@5"])
            ours_near = f"`{near['pool']}`: {near['recall_ceiling@5']:.4f} ({near['fraction_of_attainable@5']:.3f}) at {near['candidates_mean']:.0f} cand."
            ours_best = f"`{best['pool']}`: {best['recall_ceiling@5']:.4f} ({best['fraction_of_attainable@5']:.3f}) at {best['candidates_mean']:.0f} cand."
        else:
            ret = max((r for r in cells if r["regime"] == "RETRIEVAL"), key=lambda r: r["recall_ceiling@5"])
            ours_near = f"`equal_rrf_budget_200`: {[r for r in cells if r['pool'] == 'equal_rrf_budget_200'][0]['recall_ceiling@5']:.4f} at 200 cand."
            ours_best = f"`{ret['pool']}`: {ret['recall_ceiling@5']:.4f} ({ret['fraction_of_attainable@5']:.3f}) at {ret['candidates_mean']:.0f} cand."
        full_best = max((r for r in cells if r["regime"] == "FULL"), key=lambda r: r["recall_ceiling@5"], default=None)
        full_txt = f"`{full_best['pool']}`: {full_best['recall_ceiling@5']:.4f} ({full_best['fraction_of_attainable@5']:.3f}) at {full_best['candidates_mean']:.0f} cand." if full_best else "–"
        rows.append([name, LINEAGE.get(name, "–"), sub or "–", sota_txt, ours_near, ours_best, full_txt])
    return table(["dataset", "SOTA lineage", "matched substrate", "the exposure the SOTA had", "ours nearest 200 candidates on that substrate: pool_ceiling@5 (of attainable)", "ours best on that substrate", "ours best on FULL (three families; exceeds the matched substrate)"], rows)


QUESTION = (
    "After matching candidate exposure and inference-time graph information to modern "
    "graph-retrieval/GNN systems, how much effectiveness remains attributable specifically to learned message passing?"
)


def render_head(data: dict[str, dict], amend: dict) -> list[str]:
    prov = {name: d["provenance"] for name, d in data.items()}
    freezes = sorted({p["freeze_RECORD_SHA256"] for p in prov.values()})
    commits = sorted({p["git_commit"] for p in prov.values()})
    runners = sorted({p["runner_sha256"] for p in prov.values()})
    config_shas = sorted({p["config_sha256"] for p in prov.values()})
    parts: list[str] = []
    parts.append("# M3A-COMPILATION — headroom over the served canonical substrate\n")
    parts.append(
        f"Registered question: *{QUESTION}*\n\n"
        "This is the headroom half of the compilation report: **ceilings of candidate pools, not results of any ranker.** "
        "No model is scored here; nothing here is a scientific result about message passing. Every number is a property of a "
        "pool of candidates on the served substrate, measured on a labelled non-test split, and is reported with its K, its pool "
        "provenance, its dataset and its split (K semantics R1–R6, `configs/m3a_compilation.yaml#amendment_1_2026_09_13.k_semantics_adopted`). "
        "`pool_ceiling@K` is the cache-bounded ceiling; the corpus ceiling is a column and is 1.0 by construction only for the five text corpora.\n"
    )
    parts.append("## Provenance\n")
    parts.append(table(["field", "value"], [
        ["declared under", "`configs/m3a_headroom.yaml` as run (sha256 " + ", ".join(f"`{c[:16]}…`" for c in config_shas) + ", the f2d15ac version; the file has since gained `run_record`, live sha256 `" + sha(CONFIG)[:16] + "…`), `configs/m3a_compilation.yaml#amendment_1_2026_09_13.headroom_amended`"],
        ["git commit of the rule", "`f2d15ac` (2026-09-12T23:53:48Z), before the first run"],
        ["runner", "`scripts/m3a_headroom.py` (sha256 " + ", ".join(f"`{r[:16]}…`" for r in runners) + ")"],
        ["served freeze RECORD_SHA256", ", ".join(f"`{f}`" for f in freezes)],
        ["git commit at run time", ", ".join(f"`{c}`" for c in commits)],
        ["run UTC", ", ".join(sorted({p["utc"] for p in prov.values()}))],
        ["package access", "READ_ONLY; outputs are a sidecar under `outputs/m3a/headroom/` (gitignored; sha256s filed in the config)"],
        ["GPU", "none"],
    ]))
    parts.append("\n## Populations\n")
    parts.append(
        "One labelled non-test split per dataset, taken whole. WebQSP is scored on the seed-free `train_holdout` carve "
        "(lexicographic stride 2 of train, 1,549 ids, sha pinned in the config); its test split is barred. The corpus ceiling is a "
        "**column**: on webqsp it is below one because the gold references of RoG resolve at the reference level to only part of the "
        "served corpus, so no webqsp ceiling may be averaged with a text-corpus ceiling. LEGACY_CONTINUITY rows are the only bridge "
        "to the M0A–M2D numbers and are **not an evaluation population** (train-derived on four of five datasets).\n"
    )
    parts.append(population_table(data))

    parts.append("\n## Retrieval depth curve (R4)\n")
    parts.append(
        "For a prefix pool, `pool_ceiling@K` at cut-off K equals the recall@K of the retriever itself, so this table is both the "
        "recall curve of each served cache and the ceiling of the pool it would define. Equal-RRF fuses the two top-200 lists and "
        "therefore stops at 400.\n"
    )
    for name, d in data.items():
        parts.append(f"\n### {name} — `{d['eval_split']}`, {d['populations']['eval']['queries'] - d['populations']['eval']['zero_gold_excluded']:,} queries scored\n")
        parts.append(depth_table(d))

    parts.append("\n## Retrieval pools (the inherited seven)\n")
    parts.append(
        "`dense_top200`, `splade_top200`, their union (`frozen_union`) and the equal-RRF budget pools, exactly as "
        "`configs/candidate_headroom.yaml` defined them. `fraction of attainable@5` divides `pool_ceiling@5` by the ceiling a perfect "
        "retriever would have at K=5 given the gold count (`recall_ceiling_perfect_retrieval@5`).\n"
    )
    for name, d in data.items():
        parts.append(f"\n### {name}\n")
        parts.append(retrieval_table(d, "eval"))

    parts.append("\n## Graph regimes: what a fixed, parameter-free expansion adds to a retrieval pool\n")
    parts.append(
        "Seeds are the dense top-5 then the splade top-5 of the served caches (never gold, never assigned topic entities). "
        "`h1_c25`/`h1_c100`: one undirected hop, the first 25/100 neighbours of each seed per family (weighted families by weight "
        "descending, structural by position). `h2_c25`: a second hop from the hop-1 nodes in order of first appearance, 25 per "
        "frontier node per family, stopping at 2,000 visited nodes. The expansion is a function of seeds and graph alone and is "
        "unioned with the base pool. `FULL` is the union of the three per-family expansions, so its exposure is up to three times "
        "that of a single family. `missing golds recovered` and `recovered per added candidate` are `pool_movement` against the base pool.\n"
    )
    for name, d in data.items():
        parts.append(f"\n### {name}\n")
        parts.append(regime_table(d, "eval"))

    parts.append("\n## Knee reading, beside the SOTA exposure target\n")
    parts.append(
        f"Rule (declared, not chosen here): the smallest pool whose fraction of attainable recall@5 is at least {KNEE:.2f}, per dataset "
        "and regime. The SOTA exposure targets are quoted from the amendment; the SOTA-matched substrate per dataset is the regime "
        "the comparison systems actually had. Match, do not exceed.\n"
    )
    knee = []
    for name, d in data.items():
        knee += knee_rows(d, name, amend["sota_matched_substrate"])
    parts.append(table(["dataset", "regime", "knee (smallest pool reaching the rule)", "gain over its base pool @5 (knee cell, else best)", "candidates ÷ base", "best cell in regime: ceiling@5 (fraction of attainable) at candidates", "note"], knee))
    parts.append("\n### Exposure match\n")
    parts.append(
        "What the comparison systems were given, beside what our inference-safe protocol gives on the same substrate. "
        "The KB-QA exposure is measured on our STRUCT graph from the assigned topic entities (a diagnostic; never our input); "
        "the GraphER exposure is a 200-candidate scope over a query-induced corpus, which the freeze forbids us to build, so our "
        "counterpart is the matched-substrate cell nearest 200 candidates over the full corpus.\n"
    )
    parts.append(exposure_table(data, amend["sota_matched_substrate"]))
    parts.append("\n**SOTA exposure targets (amendment 1):**\n")
    for k, v in amend["sota_exposure_targets"].items():
        parts.append(f"- `{k}`: {v}")
    return parts


def render_tail(data: dict[str, dict]) -> list[str]:
    parts: list[str] = []
    parts.append("\n## Reachability of the missing gold (inherited diagnostic)\n")
    parts.append(
        "For the gold nodes the `frozen_union` pool misses, how many undirected hops from the seeds they sit, per regime graph, "
        "up to three hops and two million visited nodes per query. Strided populations are labelled. A missing gold beyond three "
        "hops is not reachable by any bounded expansion from these seeds.\n"
    )
    for name, d in data.items():
        if d.get("reachability"):
            parts.append(f"\n### {name}\n")
            parts.append(reach_table(d))

    parts.append("\n## Oracle topic-entity exposure (diagnostic column only)\n")
    for name, d in data.items():
        block = oracle_block(d)
        if block:
            parts.append(f"\n### {name}\n")
            parts.append(block)

    parts.append("\n## LEGACY_CONTINUITY paired column (NOT an evaluation population)\n")
    parts.append(
        "The legacy 2,000-query sets mapped to canonical ids: squad/musique/2wiki are TRAIN questions, hotpotqa is 1,842 train + 158 "
        "validation, metaqa is 1,998 dev with 46 gold disagreements. Reported only so a reader can place the M0A–M2D numbers "
        "next to the served substrate; never averaged with the rows above.\n"
    )
    for name, d in data.items():
        if "legacy_continuity" not in d["populations"]:
            continue
        parts.append(f"\n### {name} — {d['populations']['legacy_continuity']['queries']:,} legacy queries\n")
        parts.append(retrieval_table(d, "legacy_continuity"))
        parts.append("")
        parts.append(regime_table(d, "legacy_continuity"))

    parts.append("\n## Served graph families as loaded\n")
    rows = []
    for name, d in data.items():
        for fam, g in d["graph_families"].items():
            rows.append([name, fam, f"{g['edges_stored']:,}", "yes" if g["directed_as_stored"] else "no (one row per pair)", f"{g['csr_entries']:,}", f1(g["build_seconds"])])
    parts.append(table(["dataset", "family", "edges stored", "directed as stored", "undirected CSR entries", "CSR build s"], rows))
    parts.append(
        "\nThese rows supersede, by citation, the stale figures in `docs/M3A_GRAPH_SUBSTRATE.md` and "
        "`outputs/m3a/graph_substrate_stats.json` (musique kNN 266,488 / max degree 1,475; 2wiki with no KNN; nothing stored "
        "reciprocally): the served musique kNN has 265,366 rows, 2wiki has a kNN family, and every NER/kNN family is stored one row "
        "per unordered pair and symmetrised in memory. The old files are not edited (`amendment_1_2026_09_13.stale_rows_withdrawn`).\n"
    )

    parts.append("\n## Cost\n")
    stage_names: list[str] = []
    for d in data.values():
        for k in d["seconds"]:
            if k not in stage_names:
                stage_names.append(k)
    rows = [
        [name, f"{d['provenance']['seconds_total']:.0f}"] + [f"{d['seconds'][k]:.1f}" if k in d["seconds"] else "–" for k in stage_names]
        for name, d in data.items()
    ]
    parts.append(table(["dataset", "wall s"] + [f"{k} s" for k in stage_names], rows))
    parts.append("\nGPU seconds: 0.\n")

    parts.append("\n## What this report does not contain\n")
    parts.append(
        "- No ranker, QLS-U or GNN number: the ceilings bound what any ranker over these pools can reach; they say nothing about which ranker reaches it.\n"
        "- No test-split number.\n"
        "- No pool larger than the exposure of the inherited protocol: the widest retrieval pool is the 400-max union of two top-200 lists; the widest graph pool is the declared two-hop expansion capped at 2,000 visited nodes.\n"
        "- No averaging across datasets, and no webqsp figure on the same footing as a text-corpus figure (corpus ceiling column).\n"
        "- No selection: the knee is a reading of the declared rule; the candidate contract freezes in the M3B declaration, after review.\n"
    )
    parts.append("\n**STOP_FOR_REVIEW.**\n")
    return parts


def main() -> int:
    comp = yaml.safe_load(COMPILATION.read_text(encoding="utf-8"))
    amend = comp["amendment_1_2026_09_13"]["headroom_amended"]
    data = load()
    if not data:
        raise SystemExit("no outputs to report")
    parts = render_head(data, amend) + readings(data, amend["sota_matched_substrate"]) + render_tail(data)
    REPORT.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    print(f"wrote {REPORT.relative_to(ROOT)} for {list(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
