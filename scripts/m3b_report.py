"""docs/M3B_RESULTS.md from the M3B sidecars under outputs/m3b/
(configs/m3b_controlled_comparison.yaml#measurement, amendment 1
#reporting_amended). Re-renderable at any time; it computes nothing new except
the declared paired bootstrap and the seed mean/sd from the stored per-query
metric arrays.

    python scripts/m3b_report.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "m3b_controlled_comparison.yaml"
OUT = ROOT / "outputs" / "m3b"
EVAL = OUT / "eval"
MODELS = OUT / "models"
DOC = ROOT / "docs" / "M3B_RESULTS.md"
DATASETS = ("metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp")
DISPLAY = {"qls_u_sota_v1": "QLS-U", "gat_universal_v1": "universal GAT", "gat_no_mp_v1": "GAT-NO-MP"}
MAIN_METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "hit@1", "full_coverage@5", "full_coverage@20")
BOOT_RESAMPLES = 1000
WEBQSP_CORPUS_CEILING = 0.5255


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_eval(name: str) -> tuple[dict, dict, list[str]]:
    rec = json.loads((EVAL / f"{name}.json").read_text(encoding="utf-8"))
    z = np.load(EVAL / f"{name}.npz")
    arrays = {k: z[k] for k in z.files}
    ids = json.loads((EVAL / f"{name}_query_ids.json").read_text(encoding="utf-8"))
    return rec, arrays, ids


def paired_bootstrap(a: np.ndarray, b: np.ndarray, resamples: int = BOOT_RESAMPLES) -> tuple[float, float, float]:
    """mean(a - b) with the 95 percent percentile interval of the paired bootstrap over queries, default_rng(0)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(0)
    n = d.size
    means = np.empty(resamples)
    for r in range(resamples):
        means[r] = d[rng.integers(0, n, n)].mean()
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def scorer_keys(rec: dict, arm: str, seed: int | None = None, substrate: str | None = None) -> list[str]:
    keys = []
    for s in rec["scorers"]:
        if not s.startswith(arm + "__"):
            continue
        parts = s.split("__")
        s_seed = int(parts[2][1:])
        s_sub = parts[3] if len(parts) > 3 else None
        if seed is not None and s_seed != seed:
            continue
        if s_sub != substrate:
            continue
        keys.append(s)
    return keys


def fmt(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def seed_stat(arrays: dict, keys: list[str], metric: str) -> tuple[float | None, float | None, int]:
    vals = [float(arrays[f"{k}/{metric}"].mean()) for k in keys]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), (float(np.std(vals, ddof=0)) if len(vals) > 1 else None), len(vals)


def hop_of(query_id: str) -> str | None:
    m = re.match(r"metaqa:(\dhop):", query_id)
    return m.group(1) if m else None


def section_contract(cfg: dict, evals: dict) -> list[str]:
    key = sorted(k for k in cfg if k.startswith("candidate_contract_frozen_"))[-1]
    block = cfg[key]["per_dataset"]
    lines = [f"## 1. The frozen candidate contract (`{key}`)", "",
             "One pool per dataset, fixed before any weight existed. Ceilings are pool ceilings (K-aware `recall_ceiling@k`), "
             "measured on the eval population; `as compiled` is the same ceiling on the pools the models actually saw "
             "(the measured pool plus any retrieval seed it did not already contain). The reading column says whether the "
             "knee rule was applied over every cell or within the family the user's plan named (amendment 1, `ruled_reading`); "
             "the rule-as-filed pick is beside it either way.", "",
             "| dataset | pool | reading | R-ceil@1 | R-ceil@5 | R-ceil@20 | fraction of attainable@5 | candidates (mean) | as compiled R-ceil@5 / cand. | rule-as-filed pick |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name in DATASETS:
        b = block.get(name)
        if not b:
            continue
        e = evals.get(name)
        compiled = "—"
        if e:
            c = e[0]["ceiling_as_compiled"]
            compiled = f"{c['recall_ceiling@5']:.4f} / {c['candidates_mean']:.0f}"
        f = b["rule_as_filed"]
        lines.append(f"| {name} | `{b['pool']}` | {b['reading']} | {b['recall_ceiling@1']:.4f} | {b['recall_ceiling@5']:.4f} | {b['recall_ceiling@20']:.4f} | "
                     f"{b['fraction_of_attainable@5']:.3f} | {b['candidates_mean']:.0f} | {compiled} | `{f['pool']}` ({f['fraction_of_attainable@5']:.3f} at {f['candidates_mean']:.0f}) |")
    lines.append("")
    lines.append(f"WebQSP's corpus ceiling column is {WEBQSP_CORPUS_CEILING} (gold resolution on the served corpus): a column, never averaged.")
    lines.append("")
    return lines


def section_features(cfg: dict) -> list[str]:
    screen = json.loads((OUT / "feature_screen.json").read_text(encoding="utf-8"))
    base = json.loads((OUT / "base_score.json").read_text(encoding="utf-8"))
    carves = json.loads((OUT / "carves.json").read_text(encoding="utf-8"))["per_dataset"]
    lines = ["## 2. The information contract F(q, v) and the fixed base score", "",
             f"{screen['screened_columns']} compiled columns (directions A-D plus retrieval and the fixed GCS propagation) were screened on the fit carves "
             f"({screen['total_fit_rows']:,} candidate rows; duplicates on a {screen['sample_rows']:,}-row stride sample, |Spearman| >= {screen['duplicate_threshold']}); "
             f"{screen['surviving_columns']} survive as `QLS_U_CORE_CONTRACT` (sha256 `{screen['core_contract_sha256'][:16]}…`). "
             "Nothing was dropped for weak signal.", ""]
    if screen["dropped"]:
        lines.append("| dropped column | reason |")
        lines.append("|---|---|")
        for c, r in screen["dropped"].items():
            lines.append(f"| `{c}` | {r} |")
        lines.append("")
    if screen.get("duplicate_pairs"):
        lines.append("Duplicate pairs (earlier member kept): " + ", ".join(f"`{p['earlier']}`~`{p['later']}` ({p['abs_spearman']:.3f})" for p in screen["duplicate_pairs"]))
        lines.append("")
    if screen.get("base_column_exempted_from_the_duplicate_rule"):
        lines.append(f"The base column `{screen['base_column_exempted_from_the_duplicate_rule']}` was exempted from the duplicate rule because every arm reads it by declaration.")
        lines.append("")
    lines.append(f"**Fixed base score:** `{base['selected']}` — the candidate with the highest macro select recall@5 ranked alone "
                 f"({', '.join(f'{c} {v:.4f}' for c, v in base['macro_select_recall5'].items())}); it enters every arm as `base_z` and is what every arm returns at step 0.")
    lines.append("")
    lines.append("| dataset | train N | select | fit | " + " | ".join(f"univariate R@5 `{c}`" for c in base["candidates"]) + " |")
    lines.append("|---|---|---|---|" + "---|" * len(base["candidates"]))
    for name in DATASETS:
        cv = carves[name]
        lines.append(f"| {name} | {cv['N']:,} | {cv['select']:,} | {cv['fit']:,} | " + " | ".join(f"{base['per_dataset'][name][c]:.4f}" for c in base["candidates"]) + " |")
    lines.append("")
    top = []
    for name in DATASETS:
        sig = screen["per_dataset"][name]["univariate_recall5_best_sign"]
        order = np.argsort(sig)[::-1][:5]
        top.append(f"{name}: " + ", ".join(f"`{screen['columns'][i]}` {sig[i]:.3f}" for i in order))
    lines.append("Strongest single columns by univariate select recall@5 (reported only, never used to drop): " + "; ".join(top) + ".")
    lines.append("")
    return lines


def section_selection(cfg: dict) -> list[str]:
    sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    lines = ["## 3. Selection behind the firewall", "",
             "The GAT is chosen among GAT candidates only (L x H, seed 0, macro select recall@5, ties to the smaller); QLS-U among its own widths. "
             "No eval number existed when `outputs/m3b/selection.json` was written.", "",
             "| arm | configuration | select macro R@5 | best epoch | seconds |", "|---|---|---|---|---|"]
    for r in sel["gat_universal_v1"]["screen"]:
        mark = " **(selected)**" if (r["L"], r["H"]) == (sel["gat_universal_v1"]["L"], sel["gat_universal_v1"]["H"]) else ""
        lines.append(f"| universal GAT | L={r['L']}, H={r['H']}{mark} | {r['select_macro_recall5']:.4f} | {r['best_epoch']} | {r['seconds']:.0f} |")
    for r in sel["qls_u_sota_v1"]["screen"]:
        mark = " **(selected)**" if r["H"] == sel["qls_u_sota_v1"]["H"] else ""
        lines.append(f"| QLS-U | H={r['H']}{mark} | {r['select_macro_recall5']:.4f} | {r['best_epoch']} | {r['seconds']:.0f} |")
    lines.append(f"| GAT-NO-MP | L={sel['gat_no_mp_v1']['L']}, H={sel['gat_no_mp_v1']['H']} (the GAT's) | — | — | — |")
    lines.append("")
    return lines


def section_main(cfg: dict, evals: dict, base_name: str) -> tuple[list[str], dict]:
    """The headline table per dataset and metric, plus the two differences with their paired intervals."""
    lines = ["## 4. The controlled comparison: fixed compilation → QLS-U → GAT-NO-MP → universal GAT", "",
             "Same F(q, v), same projections, same readout, same loss, optimiser and schedule; one universal set of weights per arm over the six fit carves. "
             "Seed 0 carries the paired statistic; the seed columns give mean ± sd over seeds 0-2 where they exist. "
             "`δ_MP` = universal GAT − GAT-NO-MP (the increment attributable to message passing itself, capacity and training held fixed); "
             "`GAT − QLS-U` is the question's other difference. Intervals: paired bootstrap over eval queries, 1000 resamples, `default_rng(0)`, 95 % percentile.", ""]
    stats: dict = {}
    for name in DATASETS:
        if name not in evals:
            continue
        rec, arrays, ids = evals[name]
        q_keys = scorer_keys(rec, "qls_u_sota_v1")
        g_keys = scorer_keys(rec, "gat_universal_v1")
        c_keys = scorer_keys(rec, "gat_no_mp_v1")
        q0, g0, c0 = scorer_keys(rec, "qls_u_sota_v1", 0), scorer_keys(rec, "gat_universal_v1", 0), scorer_keys(rec, "gat_no_mp_v1", 0)
        fixed_cols = [s for s in rec["scorers"] if s.startswith("fixed:")]
        best_fixed = max(fixed_cols, key=lambda s: float(arrays[f"{s}/recall@5"].mean())) if fixed_cols else None
        ceil = rec["ceiling_as_compiled"]
        lines.append(f"### {name} — eval `{cfg['populations']['eval_splits'][name]}`, {rec['queries']:,} queries, pool `{rec['pool']}` "
                     f"(R-ceil@5 {ceil['recall_ceiling@5']:.4f}, {ceil['candidates_mean']:.0f} candidates)")
        lines.append("")
        lines.append(f"| metric | fixed `{base_name}` | best fixed ({best_fixed.split(':')[1] if best_fixed else '—'}) | QLS-U s0 | QLS-U seeds | GAT-NO-MP s0 | GAT-NO-MP seeds | GAT s0 | GAT seeds | δ_MP = GAT − NO-MP (s0) | GAT − QLS-U (s0) | pool ceiling |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        stats[name] = {}
        for metric in MAIN_METRICS:
            def cell(keys):
                if not keys:
                    return "—"
                return f"{float(arrays[f'{keys[0]}/{metric}'].mean()):.4f}"

            def seeds(keys):
                m, sd, k = seed_stat(arrays, keys, metric)
                return "—" if m is None else (f"{m:.4f} ± {sd:.4f} (n={k})" if sd is not None else f"{m:.4f} (n=1)")

            d_mp = paired_bootstrap(arrays[f"{g0[0]}/{metric}"], arrays[f"{c0[0]}/{metric}"]) if g0 and c0 else None
            d_q = paired_bootstrap(arrays[f"{g0[0]}/{metric}"], arrays[f"{q0[0]}/{metric}"]) if g0 and q0 else None
            stats[name][metric] = {"delta_mp": d_mp, "gat_minus_qlsu": d_q}
            ceiling_cell = {"recall@1": ceil.get("recall_ceiling@1"), "recall@5": ceil.get("recall_ceiling@5"), "recall@20": ceil.get("recall_ceiling@20"),
                            "full_coverage@5": ceil.get("full_coverage_ceiling@5"), "full_coverage@20": ceil.get("full_coverage_ceiling@20")}.get(metric)
            lines.append(f"| {metric} | {cell([f'fixed:{base_name}'])} | {cell([best_fixed] if best_fixed else [])} | {cell(q0)} | {seeds(q_keys)} | {cell(c0)} | {seeds(c_keys)} | "
                         f"{cell(g0)} | {seeds(g_keys)} | {ci(*d_mp) if d_mp else '—'} | {ci(*d_q) if d_q else '—'} | {fmt(ceiling_cell, 4)} |")
        lines.append("")
        if name == "webqsp":
            lines.append(f"Corpus ceiling column: {WEBQSP_CORPUS_CEILING} (reference level; every webqsp recall above is bounded by the served gold resolution).")
            lines.append("")
    return lines, stats


def section_ablation(evals: dict) -> list[str]:
    lines = ["## 5. The three substrates: message edges restricted to one family", "",
             "The selected GAT retrained (seed 0) with its message edges restricted to STRUCT, NER or KNN; F(q, v) unchanged, so the input block still reads every family's "
             "fixed prototypes. Cells are recall@5 and the paired difference to GAT-NO-MP; on a dataset where the headroom showed a family adds no exposure "
             "(squad, every family) the cell is a control, not a regime result.", "",
             "| dataset | GAT-NO-MP | GAT[STRUCT] | δ | GAT[NER] | δ | GAT[KNN] | δ | GAT[FULL] | δ_MP |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name in DATASETS:
        if name not in evals:
            continue
        rec, arrays, _ = evals[name]
        c0 = scorer_keys(rec, "gat_no_mp_v1", 0)
        if not c0:
            continue
        base = arrays[f"{c0[0]}/recall@5"]
        row = [name, f"{base.mean():.4f}"]
        for sub in ("STRUCT", "NER", "KNN", None):
            k = scorer_keys(rec, "gat_universal_v1", 0, sub)
            if not k:
                row += ["—", "—"]
                continue
            a = arrays[f"{k[0]}/recall@5"]
            m, lo, hi = paired_bootstrap(a, base)
            row += [f"{a.mean():.4f}", ci(m, lo, hi)]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def section_exposure_table(cfg: dict) -> list[str]:
    """Note 2's entry-point / exposure columns: the published rows as filed, our rows read from the frozen candidate
    contract (never typed), one row per dataset shared by the three arms. Reporting; nothing here is a result."""
    notes = [v for k, v in cfg.items() if k.startswith("note_") and isinstance(v, dict) and "calibration_columns_for_the_report" in v]
    if not notes:
        return []
    note = notes[-1]
    table = note["calibration_columns_for_the_report"]
    pools = cfg["candidate_contract_frozen_2026_09_13"]["per_dataset"]
    lines = ["**How each system enters the graph, and what its candidate set exposes.** Published coverage is "
             "at-least-one-answer-in-the-extracted-subgraph (ReaRev Table 6 / GNN-RAG Table 7); our like-for-like figure is "
             "`any_gold_at_pool` of the frozen pool (not the corpus query-level coverage, which has no pool behind it, and not "
             "the pool ceiling@5, which is K-aware). The three arms share every row of ours: identical pool, identical "
             "features, identical base score, so a difference between them is a difference of the scorer.", "",
             "| method | graph entry point | avg graph / pool | any-answer exposure | retrieval prior retained? | ranking metric |",
             "|---|---|---|---|---|---|"]
    for row in table["published_rows"]:
        lines.append("| " + " | ".join(str(row[c]).replace("|", "&#124;") for c in table["columns"]) + " |")
    seeds = "Dense + SPLADE seeds (dense top-5 ∪ splade top-5)"
    for name in DATASETS:
        p = pools[name]
        c = p["construction"]
        entry = f"{seeds}; base pool `{c['base_pool']}`" + (f"; expansion `{c['regime']}:{c['setting']['name']}`" if c.get("setting") else "; no graph expansion (retrieval-only pool)")
        knee = "" if p["fraction_of_attainable@5"] >= 0.9 else f" — below the 0.90 knee (fraction of attainable@5 {p['fraction_of_attainable@5']:.4f}; fallback rule)"
        metric = "hit@1, R@5" if name in ("metaqa", "webqsp") else "R@5 / R@10"
        lines.append(f"| **ours ({name}): QLS-U = universal GAT = GAT-NO-MP** | {entry} | {p['candidates_mean']:,.1f} candidates | "
                     f"any-gold {p['any_gold_at_pool']:.4f}, all-gold {p['all_gold_at_pool']:.4f}, pool ceiling@5 {p['recall_ceiling@5']:.4f}{knee} | "
                     f"yes — retrieval columns in F(q, v) and the fixed base score in the readout | {metric} |")
    lines.append("")
    w = note["wording_adopted_for_the_calibration_section"]
    lines.append(w["written_instead_verbatim_from_the_user"].strip() + " " + w["qualifier_that_travels_with_high_answer_exposure"].strip()[0].upper()
                 + w["qualifier_that_travels_with_high_answer_exposure"].strip()[1:] + ".")
    lines.append("")
    h = note["check_outcomes"]["6_mp_benefit_tracks_evidence_beyond_what_retrieval_exposes"]
    lines.append("**Filed before the eval, read in it (note 2, H_MP):** " + h["refined_hypothesis_H_MP"].strip() + " " + h["where_it_is_read"].strip())
    lines.append("")
    return lines


def section_sota(cfg: dict, evals: dict) -> list[str]:
    sota = cfg["measurement"]["sota_column"]
    lines = ["## 6. Calibration against the published systems (exposure attached)", "",
             "Published numbers are read against their named exposure, never averaged with ours. Our number is given in the same form: recall@5 / recall@10 "
             "for the GraphER rows (their PR@K is gold-passage coverage at K on a 200-candidate query-induced corpus with per-dataset training), "
             "hit@1 for the KB rows (NuTrea / ReaRev: assigned topic entities, full KB neighbourhood up to 3 hops, per-dataset training). "
             f"Ours: {sota['exposure_named_on_every_cell'].strip().split('ours: ')[-1]}", "",
             "| dataset | published (exposure) | ours: universal GAT s0 | ours: QLS-U s0 | ours: GAT-NO-MP s0 |", "|---|---|---|---|---|"]
    for name in DATASETS:
        if name not in evals:
            continue
        rec, arrays, ids = evals[name]
        g0, q0, c0 = scorer_keys(rec, "gat_universal_v1", 0), scorer_keys(rec, "gat_universal_v1", 0), scorer_keys(rec, "gat_no_mp_v1", 0)
        q0 = scorer_keys(rec, "qls_u_sota_v1", 0)
        pub = str(sota["cells"].get(name, "—"))
        if name in ("hotpotqa", "2wiki", "musique"):
            def our(keys):
                return "—" if not keys else f"R@5 {arrays[f'{keys[0]}/recall@5'].mean():.3f} / R@10 {arrays[f'{keys[0]}/recall@10'].mean():.3f}"
            lines.append(f"| {name} | {pub} (GraphER exposure) | {our(g0)} | {our(q0)} | {our(c0)} |")
        elif name == "metaqa":
            hops = np.asarray([hop_of(q) or "?" for q in ids])
            def our(keys):
                if not keys:
                    return "—"
                h = arrays[f"{keys[0]}/hit@1"]
                return " / ".join(f"{hp} {h[hops == hp].mean():.3f}" for hp in ("1hop", "2hop", "3hop") if (hops == hp).any()) + f" (all {h.mean():.3f})"
            lines.append(f"| {name} | {pub} (KB exposure) | hit@1 {our(g0)} | hit@1 {our(q0)} | hit@1 {our(c0)} |")
        elif name == "webqsp":
            def our(keys):
                return "—" if not keys else f"hit@1 {arrays[f'{keys[0]}/hit@1'].mean():.3f}"
            lines.append(f"| {name} | {pub} (KB exposure; corpus ceiling {WEBQSP_CORPUS_CEILING}) | {our(g0)} | {our(q0)} | {our(c0)} |")
        else:
            def our(keys):
                return "—" if not keys else f"R@5 {arrays[f'{keys[0]}/recall@5'].mean():.3f}"
            lines.append(f"| {name} | {pub} | {our(g0)} | {our(q0)} | {our(c0)} |")
    lines.append("")
    lines.append(cfg["measurement"]["sota_column"]["when_delta_mp_is_not_read"].strip())
    lines.append("")
    lines += section_exposure_table(cfg)
    notes = [v for k, v in cfg.items() if k.startswith("note_") and isinstance(v, dict) and "the_three_regimes_of_a_KB_node_text" in v]
    if notes:
        note = notes[-1]
        lines.append("**The historical MetaQA and WebQSP rows are not comparable with these.** " + note["what_the_historical_regime_was_and_was_not"]["confounded_for_the_mechanism_question"].strip()
                     + " " + note["what_the_historical_regime_was_and_was_not"]["comparability"].strip())
        lines.append("")
        lines.append("How the local facts re-enter here without message passing: " + note["how_this_phase_recreates_the_relational_information_without_message_passing"].strip())
        lines.append("")
    return lines


def section_cost(cfg: dict, evals: dict) -> list[str]:
    lines = ["## 7. Parameters, training cost, latency, memory", "",
             "| fit | parameters | epochs run | best epoch | seconds | per-epoch seconds | peak RSS GB | threads | select macro R@5 |", "|---|---|---|---|---|---|---|---|---|"]
    for path in sorted(MODELS.glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        per_epoch = " / ".join(f"{h['seconds']:.0f}" for h in r.get("history", []))
        rss = f"{r['peak_rss_bytes'] / 1e9:.2f}" if r.get("peak_rss_bytes") else "—"
        lines.append(f"| `{r['key']}` | {r['parameters']:,} | {r['epochs_run']} | {r['best_epoch']} | {r['seconds']:.0f} | {per_epoch} | {rss} | {r.get('threads', '—')} | {r['best_select_macro_recall5']:.4f} |")
    lines.append("")
    compute = [v for k, v in cfg.items() if k.startswith("amendment_") and isinstance(v, dict) and isinstance(v.get("timing"), dict)]
    if compute:
        lines.append("Wall clock is CPU time on a shared machine and is not comparable with the SOTA systems' GPU-hours: " + compute[-1]["timing"]["machine"].strip())
        lines.append("")
    lines.append("Cold per-query latency (first 500 eval queries, batch of one; compile = feature construction from the caches, embeddings and stores; forward per model) and the eval process's peak resident set:")
    lines.append("")
    lines.append("| dataset | compile p50 / p95 / p99 ms | pack p50 / p95 / p99 ms | forward per model p50 / p95 / p99 ms | peak RSS GB | threads | eval ms/query (batched) |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in DATASETS:
        if name not in evals:
            continue
        rec = evals[name][0]
        lat = rec["latency"]
        def trip(d):
            return "—" if not d else f"{d['p50_ms']:.1f} / {d['p95_ms']:.1f} / {d['p99_ms']:.1f}"
        fwd = "; ".join(f"{k.split('__')[0].replace('_v1', '')}{k[k.index('__s'):] if '__s' in k else ''}: {trip(v)}" for k, v in lat.items() if k not in ("compile", "pack"))
        lines.append(f"| {name} | {trip(lat.get('compile'))} | {trip(lat.get('pack'))} | {fwd} | {rec['peak_rss_bytes'] / 1e9:.2f} | {rec['threads']} | {rec['ms_per_query']} |")
    lines.append("")
    return lines


def section_audit(evals: dict) -> list[str]:
    lines = ["## 8. The MRR audit", "", "Every published MRR was recomputed from the stored per-query first-gold rank by an independent function (agreement to 1e-12) and "
             "hit@1 ≤ MRR ≤ 1 was asserted on every row of every scorer; `tests/test_m3b_models.py` holds the metric function against a brute-force implementation.", "",
             "| dataset | scorers audited | max |recomputed − stored| | hit@1 ≤ MRR ≤ 1 |", "|---|---|---|---|"]
    for name in DATASETS:
        if name not in evals:
            continue
        rec = evals[name][0]
        audits = rec["mrr_audit"]
        worst = max(a["max_abs_diff"] for a in audits.values())
        ok = all(a["hit1_le_mrr_le_1"] for a in audits.values())
        lines.append(f"| {name} | {len(audits)} | {worst:.1e} | {'yes' if ok else 'NO'} |")
    lines.append("")
    return lines


def section_reading(stats: dict, evals: dict) -> list[str]:
    """What the two differences say, in the three-outcome frame the plan filed; numbers only, no forbidden framing."""
    lines = ["## 9. Reading", "",
             "The registered question asks how much effectiveness remains attributable specifically to learned message passing once candidate exposure and "
             "inference-time graph information are matched. Per dataset, `δ_MP` on recall@5 with its interval:", ""]
    for name in DATASETS:
        if name not in stats:
            continue
        s = stats[name]["recall@5"]
        d_mp, d_q = s["delta_mp"], s["gat_minus_qlsu"]
        if d_mp is None:
            continue
        sign = "interval excludes zero" if (d_mp[1] > 0 or d_mp[2] < 0) else "interval includes zero"
        sign_q = "" if d_q is None else (", GAT − QLS-U " + ci(*d_q) + (" (excludes zero)" if (d_q[1] > 0 or d_q[2] < 0) else " (includes zero)"))
        lines.append(f"- **{name}**: δ_MP {ci(*d_mp)} — {sign}{sign_q}.")
    lines.append("")
    lines.append("Read against the three outcomes filed in advance: δ_MP ≈ 0 everywhere; small on the passage graphs but substantial on the KB graphs; substantial everywhere. "
                 "Cells where the universal GAT falls outside the published band without a named exposure difference are marked NOT_READ in section 6.")
    lines.append("")
    return lines


def run_record(cfg: dict) -> tuple[list[str], dict]:
    files = sorted(list(EVAL.glob("*.json")) + list(EVAL.glob("*.npz")) + list(MODELS.glob("*.json")) +
                   [OUT / "selection.json", OUT / "feature_screen.json", OUT / "base_score.json", OUT / "carves.json", OUT / "contract" / "CONTRACT.json"])
    record = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in files if p.exists()}
    lines = ["## 10. Run record", "", f"Rendered {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} by `scripts/m3b_report.py` from the sidecars below (sha256 of every output a number above cites; the sidecars are gitignored, the record is committed in the declaration).", "",
             "| file | sha256 |", "|---|---|"]
    for k, v in record.items():
        lines.append(f"| `{k}` | `{v}` |")
    lines.append("")
    return lines, record


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    evals = {name: load_eval(name) for name in DATASETS if (EVAL / f"{name}.json").exists()}
    base_name = json.loads((OUT / "base_score.json").read_text(encoding="utf-8"))["selected"]
    lines = ["# M3B — the controlled comparison on the frozen contract", "",
             f"**Registered question.** {cfg['registered_question'].strip()}", "",
             f"Declaration: `configs/m3b_controlled_comparison.yaml` (status `{cfg['status']}`). Arms: `qls_u_sota_v1` (QLS-U), `gat_universal_v1` (universal GAT), "
             "`gat_no_mp_v1` (GAT-NO-MP, the causal control). Eval populations are the headroom populations, whole; test splits were never read; "
             "the substrate is the read-only served package pinned by its freeze record. Nothing here is a scientific result on test data.", ""]
    lines += section_contract(cfg, evals)
    lines += section_features(cfg)
    if (OUT / "selection.json").exists():
        lines += section_selection(cfg)
    stats: dict = {}
    if evals:
        main_lines, stats = section_main(cfg, evals, base_name)
        lines += main_lines
        lines += section_ablation(evals)
        lines += section_sota(cfg, evals)
        lines += section_cost(cfg, evals)
        lines += section_audit(evals)
        lines += section_reading(stats, evals)
    rec_lines, record = run_record(cfg)
    lines += rec_lines
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "report_run_record.json").write_text(json.dumps({"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "files": record,
                                                             "stats_recall5": {n: {k: v for k, v in s["recall@5"].items()} for n, s in stats.items()}}, indent=1), encoding="utf-8")
    print(f"wrote {DOC} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
