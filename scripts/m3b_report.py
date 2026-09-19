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
             f"{len(screen['columns'])} compiled columns (directions A-D plus retrieval and the fixed GCS propagation) were screened on the fit carves "
             f"({screen['total_fit_rows']:,} candidate rows; duplicates on a {screen['sample_rows']:,}-row stride sample, |Spearman| >= {screen['duplicate_threshold']}); "
             f"{len(screen['surviving'])} survive as `QLS_U_CORE_CONTRACT` (sha256 `{screen['core_contract_sha256'][:16]}…`). "
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


def section_select_vs_eval(evals: dict, base_name: str) -> list[str]:
    """Reporting only, from records already on disk: the select-carve recall@5 that chose the epoch (fit record, best
    epoch) beside the eval-population recall@5 of the same weights, with the parameter-free base score on both populations
    as the control for the population shift itself. No rule reads this table."""
    base = json.loads((OUT / "base_score.json").read_text(encoding="utf-8"))
    overlap_path = OUT / "population_overlap.json"
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))["per_dataset"] if overlap_path.exists() else {}
    arms = ("qls_u_sota_v1", "gat_no_mp_v1", "gat_universal_v1")
    lines = ["### 4b. Select carve → eval population, the fixed base as the population control", "",
             "The select carve is a slice of the train split held out from the fit carve; it chose the epoch and the configuration (section 3) and is "
             "not a result. For each arm (seed 0, the selected weights): recall@5 on the select carve at the best epoch (fit record) → on the eval population "
             f"(section 4), and the same pair for the fixed base `{base_name}`, which has no parameters and moves only with the population. "
             "`shift` = eval − select; `beyond base` = the arm's shift minus the base's shift, i.e. the part of the arm's select-carve advantage over the "
             "fixed base that does not carry to the eval population. The last column (`scripts/m3b_population_overlap.py`, golds resolved by the headroom's "
             "function) is the fraction of each population's queries that hold at least one gold node which is a gold node of some fit-carve query: where the "
             "select carve shares fit golds and the eval population does not, the select carve was not distribution-matched with the eval population and its "
             "learned lift is read as train-internal. Reporting only; no rule reads it.", "",
             f"| dataset | `{base_name}` select → eval (shift) | " + " | ".join(f"{DISPLAY[a]} select → eval (shift; beyond base)" for a in arms)
             + " | queries holding a fit gold: select / eval |",
             "|---|---|" + "---|" * len(arms) + "---|"]
    for name in DATASETS:
        if name not in evals:
            continue
        rec, arrays, _ = evals[name]
        b_sel = float(base["per_dataset"][name][base_name])
        b_eval = float(arrays[f"fixed:{base_name}/recall@5"].mean())
        row = [name, f"{b_sel:.4f} → {b_eval:.4f} ({b_eval - b_sel:+.3f})"]
        for arm in arms:
            keys = scorer_keys(rec, arm, 0)
            if not keys:
                row.append("—")
                continue
            fit = json.loads((MODELS / f"{keys[0]}.json").read_text(encoding="utf-8"))
            a_sel = float(fit["history"][fit["best_epoch"]]["select_recall@5"][name])
            a_eval = float(arrays[f"{keys[0]}/recall@5"].mean())
            row.append(f"{a_sel:.4f} → {a_eval:.4f} ({a_eval - a_sel:+.3f}; {(a_eval - a_sel) - (b_eval - b_sel):+.3f})")
        o = overlap.get(name)
        if o:
            extra = ""
            if "queries_sharing_a_single_hop_component_with_fit" in o["select"]:
                extra = (f" (single-hop components shared with a fit question: {o['select']['queries_sharing_a_single_hop_component_with_fit']:.3f} / "
                         f"{o['eval']['queries_sharing_a_single_hop_component_with_fit']:.3f})")
            row.append(f"{o['select']['queries_with_a_fit_gold']:.3f} / {o['eval']['queries_with_a_fit_gold']:.3f}{extra}")
        else:
            row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def section_ablation(evals: dict) -> list[str]:
    lines = ["## 5. The three substrates: message edges restricted to one family", "",
             "The selected GAT retrained (seed 0) with its message edges restricted to STRUCT, NER or KNN; F(q, v) unchanged, so the input block still reads every family's "
             "fixed prototypes. Cells are recall@5 and the paired difference to GAT-NO-MP; on a dataset where the headroom showed a family adds no exposure "
             "(squad, every family) the cell is a control, not a regime result. Each substrate cell is a single fit (seed 0): a δ smaller than the full GAT's "
             "seed-to-seed spread on that dataset (last column, sd over seeds 0-2 from section 4) is not read.", "",
             "| dataset | GAT-NO-MP | GAT[STRUCT] | δ | GAT[NER] | δ | GAT[KNN] | δ | GAT[FULL] | δ_MP | GAT seed sd |", "|---|---|---|---|---|---|---|---|---|---|---|"]
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
        _, sd, n = seed_stat(arrays, scorer_keys(rec, "gat_universal_v1"), "recall@5")
        row.append(f"{sd:.4f} (n={n})" if sd is not None else "—")
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
    qualifier = w["qualifier_that_travels_with_high_answer_exposure"].strip().rstrip(".")
    lines.append(w["written_instead_verbatim_from_the_user"].strip() + " " + qualifier[0].upper() + qualifier[1:] + ".")
    lines.append("")
    h = note["check_outcomes"]["6_mp_benefit_tracks_evidence_beyond_what_retrieval_exposes"]
    lines.append("**Filed before the eval, read in it (note 2, H_MP):** " + h["refined_hypothesis_H_MP"].strip() + " " + h["where_it_is_read"].strip())
    lines.append("")
    return lines


def published_band(cfg: dict, name: str) -> dict:
    """The lowest published number per cell, parsed from measurement.sota_column.cells (percent -> fraction)."""
    cell = str(cfg["measurement"]["sota_column"]["cells"].get(name, ""))
    if name in ("hotpotqa", "2wiki", "musique"):
        head = cell.split(";")[0]   # "GraphER PR@5 GCS 78.8 / GAT 78.0 / MLP 78.9"
        vals = [float(x) / 100 for x in re.findall(r"(\d+\.\d+)", head)]
        return {"metric": "recall@5", "low": min(vals) if vals else None, "high": max(vals) if vals else None}
    if name == "webqsp":
        vals = [float(x) / 100 for x in re.findall(r"(\d+\.\d+)", cell)]
        return {"metric": "hit@1", "low": min(vals) if vals else None, "high": max(vals) if vals else None}
    if name == "metaqa":
        hops = {}
        for hp, v in re.findall(r"(\d)-hop (\d+\.\d+)", cell):
            hops.setdefault(f"{hp}hop", []).append(float(v) / 100)
        return {"metric": "hit@1", "per_hop_low": {h: min(v) for h, v in hops.items()}}
    return {"metric": None}


def exposure_shortfall(cfg: dict, name: str) -> float | None:
    """Published any-answer coverage minus our any_gold_at_pool, where note 2 filed both (webqsp 94.9 %, metaqa-3 99.0 %)."""
    published = {"webqsp": 0.949, "metaqa": 0.990}
    if name not in published:
        return None
    ours = cfg["candidate_contract_frozen_2026_09_13"]["per_dataset"][name]["any_gold_at_pool"]
    return published[name] - ours


def band_verdict(cfg: dict, name: str, rec: dict, arrays: dict, ids: list[str]) -> tuple[str, str]:
    """(READ | NOT_READ | CONTROL, why). The filed rule: outside the band with the gap not explained by a named
    exposure difference -> delta_MP NOT_READ for that cell. Operationalised: inside when the universal GAT (seed 0)
    is at or above the lowest published number minus the measured exposure shortfall (published coverage minus our
    any_gold_at_pool); the differences that remain named but unmeasured here (training scale and universality, the
    fit cap, the node-text regime) do not lift a cell into the band."""
    band = published_band(cfg, name)
    g0 = scorer_keys(rec, "gat_universal_v1", 0)
    if not g0 or band["metric"] is None:
        return "CONTROL", "no published graph-retrieval number for this dataset; the retrieval ceiling row stands in"
    short = exposure_shortfall(cfg, name) or 0.0
    if name == "metaqa":
        hops = np.asarray([hop_of(q) or "?" for q in ids])
        h = arrays[f"{g0[0]}/hit@1"]
        parts, inside = [], True
        for hp, low in band["per_hop_low"].items():
            ours = float(h[hops == hp].mean()) if (hops == hp).any() else None
            if ours is None:
                continue
            ok = ours >= low - short
            inside &= ok
            parts.append(f"{hp} {ours:.3f} vs {low:.3f}{'' if ok else f' (below by {low - ours:.3f}; measured exposure shortfall {short:.3f})'}")
        return ("READ" if inside else "NOT_READ"), "; ".join(parts)
    ours = float(arrays[f"{g0[0]}/{band['metric']}"].mean())
    ok = ours >= band["low"] - short
    why = f"{band['metric']} {ours:.3f} vs published {band['low']:.3f}-{band['high']:.3f}"
    if not ok:
        why += f" (below by {band['low'] - ours:.3f}; measured exposure shortfall {short:.3f})"
    return ("READ" if ok else "NOT_READ"), why


def section_sota(cfg: dict, evals: dict) -> list[str]:
    sota = cfg["measurement"]["sota_column"]
    lines = ["## 6. Calibration against the published systems (exposure attached)", "",
             "Published numbers are read against their named exposure, never averaged with ours. Our number is given in the same form: recall@5 / recall@10 "
             "for the GraphER rows (their PR@K is gold-passage coverage at K on a 200-candidate query-induced corpus with per-dataset training), "
             "hit@1 for the KB rows (NuTrea / ReaRev: assigned topic entities, full KB neighbourhood up to 3 hops, per-dataset training). "
             f"Ours: {sota['exposure_named_on_every_cell'].strip().split('ours: ')[-1]}", "",
             "| dataset | published (exposure) | ours: universal GAT s0 | ours: QLS-U s0 | ours: GAT-NO-MP s0 | band check (universal GAT s0) | δ_MP |", "|---|---|---|---|---|---|---|"]
    for name in DATASETS:
        if name not in evals:
            continue
        rec, arrays, ids = evals[name]
        g0, c0 = scorer_keys(rec, "gat_universal_v1", 0), scorer_keys(rec, "gat_no_mp_v1", 0)
        q0 = scorer_keys(rec, "qls_u_sota_v1", 0)
        verdict, why = band_verdict(cfg, name, rec, arrays, ids)
        tail = f" {why} | {verdict} |"
        pub = str(sota["cells"].get(name, "—"))
        if name in ("hotpotqa", "2wiki", "musique"):
            def our(keys):
                return "—" if not keys else f"R@5 {arrays[f'{keys[0]}/recall@5'].mean():.3f} / R@10 {arrays[f'{keys[0]}/recall@10'].mean():.3f}"
            lines.append(f"| {name} | {pub} (GraphER exposure) | {our(g0)} | {our(q0)} | {our(c0)} |" + tail)
        elif name == "metaqa":
            hops = np.asarray([hop_of(q) or "?" for q in ids])
            def our(keys):
                if not keys:
                    return "—"
                h = arrays[f"{keys[0]}/hit@1"]
                return " / ".join(f"{hp} {h[hops == hp].mean():.3f}" for hp in ("1hop", "2hop", "3hop") if (hops == hp).any()) + f" (all {h.mean():.3f})"
            lines.append(f"| {name} | {pub} (KB exposure) | hit@1 {our(g0)} | hit@1 {our(q0)} | hit@1 {our(c0)} |" + tail)
        elif name == "webqsp":
            def our(keys):
                return "—" if not keys else f"hit@1 {arrays[f'{keys[0]}/hit@1'].mean():.3f}"
            lines.append(f"| {name} | {pub} (KB exposure; corpus ceiling {WEBQSP_CORPUS_CEILING}) | {our(g0)} | {our(q0)} | {our(c0)} |" + tail)
        else:
            def our(keys):
                return "—" if not keys else f"R@5 {arrays[f'{keys[0]}/recall@5'].mean():.3f}"
            lines.append(f"| {name} | {pub} | {our(g0)} | {our(q0)} | {our(c0)} |" + tail)
    lines.append("")
    lines.append(cfg["measurement"]["sota_column"]["when_delta_mp_is_not_read"].strip())
    lines.append("")
    lines.append("**How the band check is applied.** " + " ".join(x.strip() for x in band_verdict.__doc__.split("Operationalised: ")[1].split(chr(10))).strip()
                 + " READ: δ_MP is read for that cell. NOT_READ: the numbers stand and δ_MP is reported as a measurement on an arm below the published band, "
                 "not as the answer to the question for that cell; what would lift it is a per-dataset fit at the published training scale, which is outside this phase. "
                 "CONTROL: no published number to calibrate against.")
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
    lines.append("Cold per-query latency (first 500 eval queries, batch of one; compile = feature construction from the caches, embeddings and stores; "
                 "pack = the batch tensors; forward = one model on the packed query) and the eval process's peak resident set. "
                 "The batched column is the whole pass (compile once, every model and fixed column scored) per query, node-budgeted chunks, "
                 "on the shared machine with the other lane running.")
    lines.append("")
    lines.append("| dataset | compile p50 / p95 / p99 ms | pack p50 / p95 / p99 ms | peak RSS GB | threads | eval ms/query (batched, all models) |")
    lines.append("|---|---|---|---|---|---|")
    present = [name for name in DATASETS if name in evals]

    def trip(d):
        return "—" if not d else f"{d['p50_ms']:.1f} / {d['p95_ms']:.1f} / {d['p99_ms']:.1f}"

    for name in present:
        rec = evals[name][0]
        lat = rec["latency"]
        lines.append(f"| {name} | {trip(lat.get('compile'))} | {trip(lat.get('pack'))} | {rec['peak_rss_bytes'] / 1e9:.2f} | {rec['threads']} | {rec['ms_per_query']} |")
    lines.append("")
    lines.append("Forward pass per model, p50 / p95 / p99 ms (same 500 queries, batch of one):")
    lines.append("")
    lines.append("| model | " + " | ".join(present) + " |")
    lines.append("|---|" + "---|" * len(present))
    model_keys = [k for k in evals[present[0]][0]["latency"] if k not in ("compile", "pack")]
    for k in model_keys:
        lines.append(f"| `{k}` | " + " | ".join(trip(evals[name][0]["latency"].get(k)) for name in present) + " |")
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


def section_reading(cfg: dict, stats: dict, evals: dict) -> list[str]:
    """What the two differences say, in the three-outcome frame the plan filed; every number computed here from the
    stored arrays, the band verdicts from section 6, the population shift from section 4b; no forbidden framing."""
    base = json.loads((OUT / "base_score.json").read_text(encoding="utf-8"))
    base_name = base["selected"]
    overlap_path = OUT / "population_overlap.json"
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))["per_dataset"] if overlap_path.exists() else {}
    lines = ["## 9. Reading", "",
             "The registered question asks how much effectiveness remains attributable specifically to learned message passing once candidate exposure and "
             "inference-time graph information are matched. Per dataset, on recall@5 (seed 0 paired over eval queries; the seed column is GAT − GAT-NO-MP "
             "seed by seed):", ""]
    kb, passages = [], []
    for name in DATASETS:
        if name not in stats:
            continue
        rec, arrays, ids = evals[name]
        s = stats[name]["recall@5"]
        d_mp, d_q = s["delta_mp"], s["gat_minus_qlsu"]
        if d_mp is None:
            continue
        verdict, _ = band_verdict(cfg, name, rec, arrays, ids)
        per_seed = []
        for seed in (0, 1, 2):
            g, c = scorer_keys(rec, "gat_universal_v1", seed), scorer_keys(rec, "gat_no_mp_v1", seed)
            if g and c:
                per_seed.append(float(arrays[f"{g[0]}/recall@5"].mean() - arrays[f"{c[0]}/recall@5"].mean()))
        sign = "excludes zero" if (d_mp[1] > 0 or d_mp[2] < 0) else "includes zero"
        sub = []
        c0 = scorer_keys(rec, "gat_no_mp_v1", 0)
        for fam in ("STRUCT", "NER", "KNN"):
            k = scorer_keys(rec, "gat_universal_v1", 0, fam)
            if k and c0:
                sub.append((fam, float(arrays[f"{k[0]}/recall@5"].mean() - arrays[f"{c0[0]}/recall@5"].mean())))
        sub_txt = ""
        if sub:
            best = max(sub, key=lambda t: t[1])
            sub_txt = (" Single-family message edges (one fit each): " + ", ".join(f"{f} {v:+.3f}" for f, v in sub)
                       + f"; the best single family ({best[0]}) gives {best[1]:+.3f} against the full GAT's {d_mp[0]:+.3f}.")
        q_txt = "" if d_q is None else f" GAT − QLS-U {ci(*d_q)}."
        other = stats[name]
        o_txt = " On the other metrics, δ_MP: " + "; ".join(f"{m} {ci(*other[m]['delta_mp'])}" for m in ("hit@1", "mrr", "full_coverage@5") if other.get(m, {}).get("delta_mp")) + "."
        lines.append(f"- **{name}** ({verdict}): δ_MP {ci(*d_mp)} ({sign}); by seed " + ", ".join(f"{v:+.3f}" for v in per_seed) + f".{q_txt}{o_txt}{sub_txt}")
        (kb if name in ("metaqa", "webqsp") else passages).append((name, d_mp[0], verdict))
    lines.append("")

    def andjoin(names):
        names = list(names)
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]

    def span(items):
        vals = [d for _, d, _ in items]
        return f"spans {min(vals):+.3f} to {max(vals):+.3f}" if len(vals) > 1 else (f"is {vals[0]:+.3f}" if vals else "is —")

    read_p = [n for n, _, v in passages if v == "READ"]
    ctrl = [n for n, _, v in passages if v == "CONTROL"]
    frame = ("Read against the three outcomes filed in advance (δ_MP ≈ 0 everywhere; small on the passage graphs but substantial on the KB graphs; substantial everywhere): "
             f"on the passage graphs inside the published band ({', '.join(read_p) or '—'}) δ_MP on recall@5 {span([x for x in passages if x[2] == 'READ'])}; "
             f"on the KB graphs ({', '.join(n for n, _, _ in kb) or '—'}) it {span(kb)}")
    nr = [n for n, _, v in kb if v == "NOT_READ"]
    if nr:
        frame += (f", but {andjoin(nr)} {'is' if len(nr) == 1 else 'are'} NOT_READ against the published band (section 6): the universal GAT sits below the band by "
                  "more than the measured exposure shortfall, so that δ_MP is a measurement on an arm weaker than the published systems and is not read as the answer "
                  "for the cell")
    if ctrl:
        frame += (f"; {andjoin(ctrl)} is the control with no graph exposure in its pool (retrieval-only), where message passing over the pool graph is measured "
                  "as a cost, not a regime result")
    lines.append(frame + ".")
    lines.append("")
    cov, top, further, both, lower = [], [], [], [], []
    for name in DATASETS:
        if name in stats and stats[name].get("full_coverage@5", {}).get("delta_mp") and stats[name].get("hit@1", {}).get("delta_mp"):
            fc, h1 = stats[name]["full_coverage@5"]["delta_mp"], stats[name]["hit@1"]["delta_mp"]
            cov.append((name, fc[0]))
            top.append((name, h1[0]))
            if fc[1] > 0 and h1[1] > 0:
                both.append(name)
            elif fc[1] > 0:
                further.append(name)
            elif fc[2] < 0 and h1[2] < 0:
                lower.append(name)
    if cov:
        where = ("Where the increment sits: δ_MP on full_coverage@5 (every gold of the query in the top 5) is " + ", ".join(f"{n} {v:+.3f}" for n, v in cov)
                 + "; on hit@1 it is " + ", ".join(f"{n} {v:+.3f}" for n, v in top) + " (intervals in section 4).")
        parts = []
        if further:
            parts.append(f"on {andjoin(further)} message passing adds coverage of the further golds of a multi-gold query while the top rank is not lifted "
                         "(its hit@1 interval does not exclude zero on the positive side)")
        if both:
            parts.append(f"on {andjoin(both)} it lifts the top rank as well")
        if lower:
            parts.append(f"on {andjoin(lower)} it lowers both")
        if parts:
            where += " " + "; ".join(parts).capitalize() + "."
        lines.append(where)
        lines.append("")
    if "metaqa" in stats and "webqsp" in stats:
        m, w = stats["metaqa"]["recall@5"]["delta_mp"], stats["webqsp"]["recall@5"]["delta_mp"]
        pools = cfg["candidate_contract_frozen_2026_09_13"]["per_dataset"]
        order = "The ordering H_MP predicted is the ordering measured" if m[0] > w[0] else "The ordering H_MP predicted is not the ordering measured"
        lines.append("**H_MP (note 2, filed before the eval).** It predicted that learned message passing pays where the gold retrieval misses is reachable through paths that are "
                     "discriminative inside the pool (metaqa: 9 relation types, answers at a fixed hop) and adds little where the pool is large and the relation vocabulary wide "
                     f"(webqsp). Measured: metaqa δ_MP {ci(*m)} against webqsp {ci(*w)} (pool all-gold exposure {pools['metaqa']['all_gold_at_pool']:.3f} vs "
                     f"{pools['webqsp']['all_gold_at_pool']:.3f}; any-gold {pools['metaqa']['any_gold_at_pool']:.3f} vs {pools['webqsp']['any_gold_at_pool']:.3f}). "
                     + order + "; both KB cells carry the band verdict above.")
        lines.append("")
    shifted = []
    for name in DATASETS:
        if name not in evals or name not in overlap:
            continue
        o = overlap[name]
        if o["select"]["queries_with_a_fit_gold"] >= 0.5 and o["eval"]["queries_with_a_fit_gold"] < 0.1:
            rec, arrays, _ = evals[name]
            b_shift = float(arrays[f"fixed:{base_name}/recall@5"].mean()) - float(base["per_dataset"][name][base_name])
            arm_shifts = []
            for arm in ("qls_u_sota_v1", "gat_no_mp_v1", "gat_universal_v1"):
                k = scorer_keys(rec, arm, 0)
                fit = json.loads((MODELS / f"{k[0]}.json").read_text(encoding="utf-8"))
                arm_shifts.append(float(arrays[f"{k[0]}/recall@5"].mean()) - float(fit["history"][fit["best_epoch"]]["select_recall@5"][name]))
            shifted.append(f"{name} (select carve: {o['select']['queries_with_a_fit_gold']:.3f} of queries hold a fit gold; eval: {o['eval']['queries_with_a_fit_gold']:.3f}; "
                           f"fixed base shift {b_shift:+.3f}, arms {', '.join(f'{v:+.3f}' for v in arm_shifts)})")
    if shifted:
        lines.append("**Where the select carve was not distribution-matched with the eval population** (section 4b): " + "; ".join(shifted) + ". "
                     "The three arms lose alike, so the controlled differences survive the shift; the absolute learned lift on such a dataset is population-specific and the "
                     "select-carve numbers of section 3 are read as model selection only.")
        lines.append("")
    lines.append("Nothing here reads a negative or small δ_MP as a statement that message passing is unnecessary, and nothing reads a positive one beyond its interval; "
                 "the NOT_READ cells are the calibration verdicts of section 6, not results.")
    lines.append("")
    return lines


def run_record(cfg: dict) -> tuple[list[str], dict]:
    files = sorted(list(EVAL.glob("*.json")) + list(EVAL.glob("*.npz")) + list(MODELS.glob("*.json")) +
                   [OUT / "selection.json", OUT / "feature_screen.json", OUT / "base_score.json", OUT / "carves.json", OUT / "contract" / "CONTRACT.json",
                    OUT / "population_overlap.json"])
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
    glance_at = len(lines)   # the one-line summary is inserted here once the statistics exist
    lines += section_contract(cfg, evals)
    lines += section_features(cfg)
    if (OUT / "selection.json").exists():
        lines += section_selection(cfg)
    stats: dict = {}
    if evals:
        main_lines, stats = section_main(cfg, evals, base_name)
        lines += main_lines
        lines += section_select_vs_eval(evals, base_name)
        lines += section_ablation(evals)
        lines += section_sota(cfg, evals)
        lines += section_cost(cfg, evals)
        lines += section_audit(evals)
        lines += section_reading(cfg, stats, evals)
    if stats:
        glance = []
        for name in DATASETS:
            if name not in stats:
                continue
            rec, arrays, ids = evals[name]
            verdict, _ = band_verdict(cfg, name, rec, arrays, ids)
            d = stats[name]["recall@5"]["delta_mp"]
            if d:
                glance.append(f"{name} {ci(*d)} ({verdict})")
        lines[glance_at:glance_at] = ["**At a glance — δ_MP on recall@5, universal GAT − GAT-NO-MP, seed 0, paired 95 % interval, with the section-6 band verdict:** "
                                      + "; ".join(glance) + ". The reading is section 9; nothing above is a result on test data.", ""]
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
