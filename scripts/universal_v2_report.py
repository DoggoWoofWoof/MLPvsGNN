"""docs/UNIVERSAL_V2_PILOT.md from the sidecars under outputs/universal_v2/
(configs/universal_v2.yaml#outputs.code, #measurement, #check_1_the_halves,
order_of_operations 7). Two stages, in the declared order:

    python scripts/universal_v2_report.py --stage held   # V2_HELD_CONFIRMATION read once -> outputs/universal_v2/held_record.json
    python scripts/universal_v2_report.py --stage doc    # docs/UNIVERSAL_V2_PILOT.md, every number pinned by sha256
    python scripts/universal_v2_report.py --stage replication      # amendment 4: seeds 0-2 of the selected arms on V2_GATE, once;
                                                                   # the held half for a REPLICATION_PASS family only, once
    python scripts/universal_v2_report.py --stage replication_doc  # docs/UNIVERSAL_V2_REPLICATION.md from the two records

The held stage refuses before the gate is filed in the declaration
(pilot_gate_record_<date> pinning gate_record.json), refuses a second time,
and refuses while a passing arm's seeds 1-2 are unfitted or unscored; it
produces the held column of the phase once. The doc stage reads the held half
only through held_record.json: its own reads of the eval arrays are sliced to
V2_GATE as they are read (the seed confirmation, the gate-half seed table).
Nothing here computes a new number except the declared paired bootstrap and
the seed mean / sd from the stored per-query arrays. The doc is
re-renderable; the held record is not.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def load_script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = load_script("universal_v2_run")      # the run tooling: paths, the paired procedures, the gate cells, the readers (module attributes read at call time)
M3B_REPORT = load_script("m3b_report")   # published_band / exposure_shortfall: the M3B parsing of the published cells, reused unchanged

SUMMARY_METRICS = ("recall@1", "recall@5", "recall@10", "recall@20", "hit@1", "mrr", "ndcg@5", "ndcg@20", "full_coverage@5", "full_coverage@20")
TABLE_METRICS = ("recall@1", "recall@5", "recall@20", "hit@1", "mrr", "full_coverage@5")
SEED_METRICS = ("recall@5", "hit@1", "full_coverage@5", "mrr")
DISPLAY = {"u_mlp_v2": "U-MLP-v2", "u_mlp_v2_mix": "U-MLP-v2-mix", "u_gnn_v2": "U-GNN-v2", "u_gnn_v2_ef": "U-GNN-v2-EF",
           "u_gnn_v2_core78": "U-GNN-v2 on the M3B 78 (ablation)", "gat_universal_v1_trio": "universal GAT, trio (control)",
           "gat_universal_v1": "universal GAT (frozen M3B)", "gat_no_mp_v1": "GAT-NO-MP (frozen M3B)", "qls_u_sota_v1": "QLS-U (frozen M3B)",
           "fixed_rrf": "fixed rrf"}
HELD, GATE = "V2_HELD_CONFIRMATION", "V2_GATE"
LF = chr(10)


# ── readers: every array is sliced to one half as it is read ─────────────────


def is_fit_key(key: str) -> bool:
    return not key.startswith("fixed:")


def arm_of(key: str) -> str:
    return key.split("__")[0]


def seed_of(key: str) -> int:
    return int(key.split("__")[2][1:])


def eval_records(name: str) -> list[tuple[str, dict]]:
    """The seed-0 record of a population and every unsharded supplement record beside it, (file stem, record)."""
    base = R.EVAL / f"{name}.json"
    if not base.exists():
        raise SystemExit(f"{name}: no eval record under outputs/universal_v2/eval/")
    out = [(name, R.read_json(base))]
    for p in sorted(R.EVAL.glob(f"{name}__more_*.json")):
        if p.name.endswith("_query_ids.json") or "__shard" in p.name:
            continue
        out.append((p.stem, R.read_json(p)))
    return out


def pooled_rows(name: str, half_name: str) -> tuple[dict, np.ndarray, dict]:
    """One half of a population across its seed-0 record and its supplements: the mask is applied to every
    array as it is read, so the other half never enters the caller. The supplements are pinned to the seed-0
    record by the query-id list, the half labels and the fixed rrf per query on the rows read."""
    if half_name not in (GATE, HELD):
        raise ValueError(half_name)
    records = eval_records(name)
    ids = json.loads((R.EVAL / f"{name}_query_ids.json").read_text(encoding="utf-8"))
    with np.load(R.EVAL / f"{name}.npz") as z:
        half = z["half"].astype(bool)
        mask = half if half_name == GATE else ~half
        rows = {k: z[k][mask] for k in z.files if k != "half"}
    meta = {name: {"record_sha256": R.sha256_file(R.EVAL / f"{name}.json"), "arrays_sha256": R.sha256_file(R.EVAL / f"{name}.npz"),
                   "scorers": records[0][1]["scorers"], "supplement": None}}
    for stem, rec in records[1:]:
        if json.loads((R.EVAL / f"{stem}_query_ids.json").read_text(encoding="utf-8")) != ids:
            raise SystemExit(f"{stem}: not the query list of {name}; refusing")
        if rec["supplement"]["seed0_record_sha256"] != meta[name]["record_sha256"]:
            raise SystemExit(f"{stem}: pinned to a different seed-0 record than the one on disk; refusing")
        with np.load(R.EVAL / f"{stem}.npz") as z:
            if not np.array_equal(z["half"].astype(bool), half):
                raise SystemExit(f"{stem}: the half labels differ from {name}; refusing")
            if np.max(np.abs(z["fixed:rrf/recall@5"][mask] - rows["fixed:rrf/recall@5"])) > 1e-9:
                raise SystemExit(f"{stem}: the fixed rrf differs from {name} on the rows read; the pools differ; refusing")
            for key in rec["supplement"]["models"]:
                if f"{key}/recall@5" in rows:
                    raise SystemExit(f"{stem}: {key} is scored twice on {name}; a model is scored once")
                for a in z.files:
                    if a.startswith(key + "/"):
                        rows[a] = z[a][mask]
        meta[stem] = {"record_sha256": R.sha256_file(R.EVAL / f"{stem}.json"), "arrays_sha256": R.sha256_file(R.EVAL / f"{stem}.npz"),
                      "scorers": rec["scorers"], "supplement": rec["supplement"]}
    return rows, mask, meta


def m3b_rows(name: str, mask: np.ndarray) -> dict:
    """The frozen M3B arrays of the same population on the rows given (the query lists must agree exactly)."""
    return R.m3b_gate_rows(name, mask)


def paired_pairs(selected: dict) -> list:
    """measurement.paired_procedures.against_each_other and against_m3b, as the gate stage lists them, plus the
    ablation minus the control (hypotheses.H_operator_vs_basis: the operator alone against the ordinary GAT)."""
    g, t = R.fit_key(selected["gnn"], 0), R.fit_key(selected["twin"], 0)
    core78, trio = R.fit_key("u_gnn_v2_core78", 0), R.fit_key("gat_universal_v1_trio", 0)
    return [("selected_gnn_minus_selected_twin", g, t, False), ("selected_gnn_minus_u_gnn_v2_core78", g, core78, False),
            ("selected_gnn_minus_gat_universal_v1_trio", g, trio, False), ("u_gnn_v2_core78_minus_gat_universal_v1_trio", core78, trio, False),
            ("selected_gnn_minus_gat_universal_v1_frozen", g, R.M3B_REFERENCES["gat_universal_v1"], True),
            ("selected_twin_minus_qls_u_sota_v1_frozen", t, R.M3B_REFERENCES["qls_u_sota_v1"], True),
            ("selected_twin_minus_gat_no_mp_v1_frozen", t, R.M3B_REFERENCES["gat_no_mp_v1"], True)]


def summary_of(rows: dict) -> dict:
    return {k: {m: round(float(rows[f"{k}/{m}"].mean()), 4) for m in SUMMARY_METRICS if f"{k}/{m}" in rows} for k in R.scorer_keys(rows)}


def seed_stats(rows: dict) -> dict:
    """measurement.paired_procedures.seeds: mean and sd over the seeds present per arm, with the per-seed values."""
    out: dict = {}
    fit_keys = [k for k in R.scorer_keys(rows) if is_fit_key(k)]
    for arm in sorted({arm_of(k) for k in fit_keys}):
        keys = sorted((k for k in fit_keys if arm_of(k) == arm), key=seed_of)
        out[arm] = {"seeds": [seed_of(k) for k in keys]}
        for m in SEED_METRICS:
            vals = [float(rows[f"{k}/{m}"].mean()) for k in keys]
            out[arm][m] = {"per_seed": {str(seed_of(k)): round(v, 4) for k, v in zip(keys, vals)}, "mean": round(float(np.mean(vals)), 4),
                           "sd": round(float(np.std(vals)), 4) if len(vals) > 1 else None, "n": len(vals)}
    return out


# ── the held half, read once ─────────────────────────────────────────────────


def stage_held(cfg: dict, log=print) -> dict:
    """check_1_the_halves: V2_HELD_CONFIRMATION read once, in the final report of the phase, as the confirmatory
    column. Refused before gate_record.json is filed in the declaration (the review point of order_of_operations 6),
    refused a second time, refused while a passing arm's seeds 1-2 are unscored (order_of_operations 7). The gate
    cells are re-read on this half and labelled confirmatory: nothing here is a gate and nothing here advances an arm."""
    path = R.OUT / "held_record.json"
    if path.exists():
        raise SystemExit("held_record.json exists; V2_HELD_CONFIRMATION is read once (check_1_the_halves)")
    gate = R.read_json(R.OUT / "gate_record.json")
    if gate is None:
        raise SystemExit("no gate_record.json: the held half is read after the gate (order_of_operations 6 then 7)")
    filed = R.dated_blocks(cfg, "pilot_gate_record")
    if not filed:
        raise SystemExit("the gate record is not filed in the declaration (pilot_gate_record_<date>); the held half is read after that review point")
    gate_sha = R.sha256_file(R.OUT / "gate_record.json")
    if cfg[filed[-1]].get("output_sha256") != gate_sha:
        raise SystemExit(f"gate_record.json ({gate_sha[:12]}) is not the record filed as {filed[-1]}; refusing")
    if cfg.get("status") not in ("PILOT_GATE_READ", "RUN", "RUN_PILOT_FAILED"):
        raise SystemExit(f"status {cfg.get('status')}: the held half is read at PILOT_GATE_READ or later")
    passing = sorted(arm for arm, v in gate["outcome"].items() if v == "PASS")
    need = [R.fit_key(arm, s) for arm in passing for s in (1, 2)]
    rows, masks, refs, meta = {}, {}, {}, {}
    for name in R.PILOT:
        rows[name], masks[name], meta[name] = pooled_rows(name, HELD)
        missing = [k for k in need if f"{k}/recall@5" not in rows[name]]
        if missing:
            raise SystemExit(f"{name}: {missing} not scored; seeds 1-2 of a passing arm are fitted and scored before the held half is read "
                             "(pilot_gate.on_pass, order_of_operations 7)")
        refs[name] = m3b_rows(name, masks[name])
    selected = gate["selection"]
    cells = R.gate_cells(*R.gate_thresholds(cfg)[:2])
    confirmatory = {arm: {**R.evaluate_cells(R.fit_key(arm, 0), cells[family], rows, refs), "confirmatory_not_a_gate": True,
                          "read": "the gate cells re-read on V2_HELD_CONFIRMATION for the arm the gate selected; the verdict is the gate's, on V2_GATE"}
                    for family, arm in selected.items()}
    record = {"utc": R.utc(), "half": HELD, "read_once": True, "status_at_read": cfg["status"], "pilot_gate_record_block": filed[-1],
              "gate_record_sha256": gate_sha, "selection": selected, "outcome_on_V2_GATE": gate["outcome"], "passing_arms": passing,
              "queries": {name: int(masks[name].sum()) for name in R.PILOT},
              "scorers": {name: R.scorer_keys(rows[name]) for name in R.PILOT},
              "summary": {name: summary_of(rows[name]) for name in R.PILOT},
              "frozen_references": {name: {label: {m: round(float(refs[name][f"{key}/{m}"].mean()), 4) for m in ("recall@5", "hit@1", "full_coverage@5", "mrr")}
                                           for label, key in R.M3B_REFERENCES.items()} for name in R.PILOT},
              "gate_cells_confirmatory": confirmatory, "paired": R.paired_table(rows, refs, paired_pairs(selected)),
              "seeds": {name: seed_stats(rows[name]) for name in R.PILOT}, "slices": R.slices_on(rows), "mechanism_readouts": R.mechanism_on(rows),
              "bootstrap": R.BOOTSTRAP, "records_read": meta,
              "m3b_reference_arrays": {name: R.sha256_file(R.M3B_OUT / "eval" / f"{name}.npz") for name in R.PILOT},
              "what_this_is": "the reported column of the pilot (process_rules_kept); not previously unseen data, not an independent test set "
                              "(check_1_the_halves.what_it_is_not); no threshold, selection or checkpoint decision read it"}
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    for arm, v in confirmatory.items():
        log(f"held {arm}: " + "; ".join(f"{c['dataset']}/{c['metric']}/{c['slice']} {c['value']}" for c in v["cells"]) + " (confirmatory, not a gate)")
    log(f"wrote {path} ({sum(record['queries'].values())} held queries)")
    return record


# ── the document ─────────────────────────────────────────────────────────────


def fmt(x, digits: int = 3) -> str:
    return "—" if x is None else f"{float(x):.{digits}f}"


def ci(d: dict | None) -> str:
    return "—" if not d else f"{d['mean']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]"


def capital(text) -> str:
    """A config sentence with its first letter upper-cased (the yaml prose starts lower-case)."""
    text = str(text).strip()
    return text[:1].upper() + text[1:]


def label(key: str) -> str:
    if not is_fit_key(key):
        return DISPLAY.get(key.replace("fixed:", "fixed_"), key)
    return f"{DISPLAY.get(arm_of(key), arm_of(key))} s{seed_of(key)}"


def section_header(cfg: dict, held: dict, gate: dict, terminal: dict | None = None) -> list[str]:
    lines = ["# Universal-v2 pilot — the trio under UNIVERSAL_V2_CORE_CONTRACT", "",
             f"**Registered question of this phase.** {str(cfg['registered_question']).strip()}", "",
             f"**Registered question of the paper (unchanged).** {str(cfg['registered_question_of_the_paper_unchanged']).strip()}", "",
             f"Declaration: `configs/universal_v2.yaml` (status `{cfg['status']}`; gate filed as `{held['pilot_gate_record_block']}`). "
             "A pilot on the three M3B eval populations (metaqa dev, 2wiki dev, squad dev; never a test split), each split by the filed rule into "
             f"`{GATE}` (the gate half, read once for the selected GNN and the selected twin) and `{HELD}` (the reported column, read once by "
             "`scripts/universal_v2_report.py --stage held`). The held half is not previously unseen data and not an independent test set: M3B scored "
             "these populations whole, and the M3B numbers on both halves are on file. Select-carve numbers select and are never results. "
             "The substrate is the read-only served package pinned by its freeze record; M3B is pinned byte-for-byte and never re-run.", ""]
    fam = family_labels(held["selection"], gate)
    glance = [f"**{R.FAMILY_GATES['gnn']} {fam[R.FAMILY_GATES['gnn']]}** / **{R.FAMILY_GATES['twin']} {fam[R.FAMILY_GATES['twin']]}** / overall **{fam['overall']}** "
              "(amendment 2 family_status_vocabulary; a one-family pass is that family's pass, never a pass of the proposed universal pair)"]
    if terminal:
        ff = terminal["family_final"]
        glance.append(f"Terminal state **{terminal['terminal']}** (amendment 3 terminal_state_vocabulary: {R.FAMILY_GATES['gnn']} {ff[R.FAMILY_GATES['gnn']]}, "
                      f"{R.FAMILY_GATES['twin']} {ff[R.FAMILY_GATES['twin']]}; a confirmation of one family is never a confirmation of the pair)")
    for family, arm in held["selection"].items():
        v = held["gate_cells_confirmatory"][arm]
        cells = "; ".join(f"{c['dataset']}/{c['metric']}/{c['slice']} {c['value']:.3f} vs {c['threshold']:.3f}" for c in v["cells"])
        glance.append(f"{family} `{arm}` ({R.FAMILY_GATES[family]} {fam[R.FAMILY_GATES[family]]}): gate **{gate['outcome'][arm]}** on {GATE} (seed 0); on {HELD} {cells}")
    lines.append("**At a glance.** " + ". ".join(glance) + f". The held-half cells are confirmatory readings of the gate's arms, not a gate; "
                 "nothing here is a statement about message passing (the paper's question is answered by M3B) or a comparison to the published systems "
                 "(their exposure is named in section 10).")
    lines.append("")
    return lines


def family_labels(selection: dict, gate: dict) -> dict:
    """The family / overall labels of the gate record (pilot_gate_record_<date>.family_outcome), never recomputed
    from the cells here; the run tooling's function is the fallback for a record written before the vocabulary."""
    return gate.get("family_outcome") or R.family_outcome(selection, gate["outcome"])


def section_contract(cfg: dict, screen: dict) -> list[str]:
    key, block = R.frozen_contract_v2(cfg)
    hashes = block.get("hashes") or {}
    lines = [f"## 1. The frozen contract (`{key}`)", "",
             f"{block['screened_columns']} raw columns (the M3B 78 first, then the {block.get('v2_columns_screened', block['depth_basis_screened'])} v2 columns of "
             f"`information_contract_v2`: {block['depth_basis_screened']} depth-basis and {block.get('ordered_relation_path_screened', 0)} ordered relation-path columns, "
             f"pinned by `{block.get('raw_contract_pinned_in', 'the declaration')}`) were screened on the fit carves only, under the M3B rule verbatim; "
             f"{block['surviving_columns']} survive as `{block['name']}` (sha256 `{block['sha256_of_comma_joined_surviving_names'][:16]}…`). "
             "The M3B 78 are protected earlier members and always survive; the trio-only statistics that would have dropped one of them are recorded, not applied.", ""]
    if hashes:
        lines += ["| hashed at contract_frozen | sha256 |", "|---|---|"]
        for k in ("raw_contract_sha256", "screen_rule_sha256", "surviving_columns_sha256", "six_caches_combined_sha256"):
            if k in hashes:
                lines.append(f"| {k} | `{hashes[k]}` |")
        lines.append("")
    dropped = block.get("dropped", {})
    if dropped:
        lines += ["| dropped column | reason |", "|---|---|"]
        for c, why in dropped.items():
            lines.append(f"| `{c}` | {why} |")
        lines.append("")
    would = (screen.get("m3b_core") or {}).get("would_have_dropped_under_the_trio_statistics") or {}
    if would:
        lines.append("M3B columns the trio statistics alone would have dropped (protected, kept): " + ", ".join(f"`{c}` ({why})" for c, why in would.items()) + ".")
        lines.append("")
    pairs = block.get("duplicate_pairs") or []
    if pairs:
        lines.append("Duplicate pairs among the new columns (earlier member kept): " + ", ".join(f"`{p['earlier']}`~`{p['later']}` ({p['abs_spearman']:.3f})" for p in pairs) + ".")
        lines.append("")
    lines += ["| dataset | carve | queries | rows | candidates (mean) | compile ms/query | queries / s | wall s | peak RSS GB | cache GB | seeds added / query |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in R.PILOT:
        for kind in ("fit", "select"):
            m = block["compile_record"][name][kind]
            gb = m["cache_bytes"] / 2**30 if m.get("cache_bytes") is not None else None
            rss = m["peak_rss_bytes"] / 2**30 if m.get("peak_rss_bytes") is not None else None
            lines.append(f"| {name} | {kind} | {m['queries']:,} | {m['rows']:,} | {m['candidates_mean']:.1f} | {m['ms_per_query']} | {fmt(m.get('queries_per_second'), 2)} | "
                         f"{fmt(m.get('compile_seconds'), 0)} | {fmt(rss, 2)} | {fmt(gb, 2)} | {fmt(m.get('seeds_added_mean'), 2)} |")
    lines.append("")
    lines += section_compile_diagnostics(block)
    return lines


def section_compile_diagnostics(block: dict) -> list[str]:
    """amendment 2 k_rel_truncation_report_required: the K_REL slot truncation and the typed / ordered relation-path
    availability over every query of every carve, by dataset and (metaqa) by hop, as the compile record filed them."""
    rows_t, rows_a = [], []
    for name in R.PILOT:
        for kind in ("fit", "select"):
            m = block["compile_record"][name][kind]
            d = m.get("diagnostics")
            if not d:
                continue
            for hop, h in d["by_hop"].items():
                rs = h["relation_slots"]
                hist = rs.get("relations_per_pair_histogram") or {}
                over = sum(int(c) for k, c in hist.items() if int(k) > d["k_rel"])
                rows_t.append(f"| {name} | {kind} | {hop} | {h['queries']:,} | {h['typed_queries']:,} | {rs['pairs']:,} | {rs['entries']:,} | "
                              f"{rs['pairs_truncated']:,} | {rs['fraction_of_pairs_truncated']:.4f} | {rs['fraction_of_queries_with_any_truncated_pair']:.4f} | "
                              f"{rs['max_relations_per_pair']} | {over:,} |")
                w, o = h["typed_walk_availability"], h["ordered_path"]
                rows_a.append(f"| {name} | {kind} | {hop} | " + " | ".join(f"{w[f'h{t}']['rows']:.3f} / {w[f'h{t}']['gold_rows']:.3f} / {w[f'h{t}']['queries']:.3f}" for t in (1, 2, 3))
                              + " | " + " | ".join(f"{o[f'h{t}']['rows_with_walk']:,} / {o[f'h{t}']['fraction_of_walks_with_an_inverse_step']:.3f} / "
                                                   f"{o[f'h{t}']['fraction_of_walks_composing_different_relations']:.3f}" for t in (2, 3)) + " |")
    if not rows_t:
        return []
    lines = [f"K_REL = {block['compile_record'][R.PILOT[0]]['fit']['diagnostics']['k_rel']} relation-text slots per structural message edge: how often a "
             "structural pair carries more stored relations than the slots hold, over every query of every carve (amendment 2 "
             "`k_rel_truncation_report_required`; untyped datasets have no structural relation table and report zero pairs).", "",
             "| dataset | carve | hop | queries | typed queries | pairs | typed entries | pairs truncated | fraction of pairs | fraction of queries with any | max relations / pair | pairs beyond K_REL (histogram) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|", *rows_t, "",
             "Typed STRUCT relation-path availability (rows with a walk of length t / gold rows / queries with any such row) and the ordered channel "
             "(best walks at t = 2, 3: rows with a walk / fraction with an inverse step / fraction composing two different relations):", "",
             "| dataset | carve | hop | h1 rows / gold / queries | h2 | h3 | ordered h2 rows / inverse / heterogeneous | ordered h3 |", "|---|---|---|---|---|---|---|---|",
             *rows_a, ""]
    return lines


def section_cost(timing: dict | None, fits: dict, evals: dict) -> list[str]:
    lines = ["## 2. Cost: the timing run, the fits, the eval pass", ""]
    if timing:
        f = timing["full_epoch_u_gnn_v2_ef"]
        lines.append(f"Timing run (`timing.json`, before any fit): one epoch of `u_gnn_v2_ef` = {f['epoch_seconds']:.0f} s at {timing['threads']} threads "
                     f"({timing['batches_per_epoch']} batches of {timing['batch_size']}); fallback fires: {timing['fallback']['fires']}. Per arm on three mixed batches:")
        lines += ["", "| arm | parameters | s / batch | projected minutes / epoch |", "|---|---|---|---|"]
        for arm, a in timing["arms"].items():
            lines.append(f"| `{arm}` | {a['parameters']:,} | {a['mean_seconds_per_batch']:.2f} | {a['projected_epoch_minutes']} |")
        lines.append("")
    lines += ["| fit | parameters (budget) | epochs | best | seconds | s / epoch | peak RSS GB | threads | select macro R@5 (selection only) |", "|---|---|---|---|---|---|---|---|---|"]
    for key, r in sorted(fits.items()):
        lines.append(f"| `{key}` | {r['parameters']:,} ({r['parameter_budget']:,}) | {r['epochs_run']} | {r['best_epoch']} | {r['seconds']:.0f} | "
                     f"{r['seconds'] / max(r['epochs_run'], 1):.0f} | {r['peak_rss_bytes'] / 2**30:.2f} | {r['threads']} | {r['best_select_macro_recall5']:.4f} |")
    lines.append("")
    lines += ["| eval record | queries | ms / query | compile p50 / p95 ms | pack p50 / p95 ms | forward p50 ms per model | peak RSS GB |", "|---|---|---|---|---|---|---|"]
    for stem, r in sorted(evals.items()):
        lat = r.get("latency") or {}
        def pp(k):
            v = lat.get(k) or {}
            return f"{v['p50_ms']:.1f} / {v['p95_ms']:.1f}" if v else "—"
        fwd = "; ".join(f"`{k}` {v['p50_ms']:.1f}" for k, v in lat.items() if k not in ("compile", "pack") and v)
        lines.append(f"| `{stem}` | {r['queries']:,} | {r['ms_per_query']} | {pp('compile')} | {pp('pack')} | {fwd or '—'} | {r['peak_rss_bytes'] / 2**30:.2f} |")
    lines.append("")
    return lines


def section_selection(sel: dict) -> list[str]:
    lines = ["## 3. Selection behind the firewall", "",
             "Within each family the seed-0 candidate with the highest macro select recall@5 (ties to the fewer parameters); `selection.json` was "
             "written once, before any eval population was scored, and no number of the other family or of M3B entered it. The ablation "
             "`u_gnn_v2_core78` takes the selected GNN architecture on the M3B 78 and feeds no selection; `gat_universal_v1_trio` is the control.", "",
             "| family | candidate | select macro R@5 | parameters | best epoch | seconds |", "|---|---|---|---|---|---|"]
    for family in ("gnn", "twin"):
        f = sel[family]
        for arm, c in f["candidates"].items():
            mark = " **(selected)**" if arm == f["arm"] else ""
            lines.append(f"| {family} | `{arm}`{mark} | {c['select_macro_recall5']:.4f} | {c['parameters']:,} | {c['best_epoch']} | {c['seconds']:.0f} |")
    lines.append("")
    return lines


def cell_rows(v: dict) -> list[str]:
    out = []
    for c in v["cells"]:
        interval = f"{ci(c['interval'])} vs `{c['paired_vs']}`, excludes zero: {c['interval_excludes_zero_on_the_positive_side']}" if c.get("paired_vs") else "—"
        out.append(f"| {c['dataset']} | {c['metric']} | {c['slice']} | {c['queries']:,} | {c['value']:.4f} | {c['threshold']:.4f} | {interval} | {'holds' if c['holds'] else 'FAILS'} |")
    return out


def section_gate(cfg: dict, gate: dict) -> list[str]:
    lines = [f"## 4. The pilot gate ({GATE}, seed 0, read once)", "",
             f"Thresholds from `{gate['thresholds_from']}`, filed before any weight existed; paired intervals: bootstrap over the gate-half queries, "
             f"{gate['bootstrap']['resamples']} resamples, `default_rng({gate['bootstrap']['seed']})`, {gate['bootstrap']['level']} % percentile. "
             "The gate is read for the selected GNN and the selected twin only; a non-selected candidate is reported below and cannot advance. "
             f"What the gate is not: {str(cfg['pilot_gate']['what_the_gate_is_not']).strip()}", ""]
    for arm, v in gate["verdict"].items():
        lines += [f"**`{arm}` — {gate['outcome'][arm]}**", "", "| dataset | metric | slice | queries | value | threshold | paired interval | cell |", "|---|---|---|---|---|---|---|---|"]
        lines += cell_rows(v)
        lines.append("")
    if gate.get("reported_not_advancing"):
        lines.append("Reported, not advancing (the other candidate of each family, on the same cells):")
        lines.append("")
        for arm, v in gate["reported_not_advancing"].items():
            lines += [f"`{arm}` (would {'hold' if v['pass'] else 'fail'} the cells; {v['why']}):", "", "| dataset | metric | slice | queries | value | threshold | paired interval | cell |", "|---|---|---|---|---|---|---|---|"]
            lines += cell_rows(v)
            lines.append("")
    lines.append(f"On a pass: {str(cfg['pilot_gate']['on_pass']).strip()} On a fail: {str(cfg['pilot_gate']['on_fail']).strip()}")
    lines.append("")
    return lines


def seed_mean_cells(cells: list, arm: str, rows: dict, refs: dict) -> dict:
    """The declared seed aggregation on the rows given: per cell, the per-query mean over the seeds of the arm that
    are present on every population, its population mean against the threshold (a float tolerance of 1e-12 and
    nothing else), the sd over the per-seed population means, and for a paired cell the declared bootstrap of the
    per-query seed mean against the frozen reference; complete when the three seeds are present, confirmed when
    complete and every cell holds."""
    keys = [R.fit_key(arm, s) for s in (0, 1, 2)]
    present = [k for k in keys if all(f"{k}/recall@5" in rows[n] for n in R.PILOT)]
    entries, confirmed = [], len(present) == 3
    for name, metric, slice_, threshold, paired in cells:
        mask = R.slice_mask(rows[name], slice_)
        per_seed = {str(seed_of(k)): round(float(rows[name][f"{k}/{metric}"][mask].mean()), 4) for k in present}
        mean_q = np.mean([rows[name][f"{k}/{metric}"][mask] for k in present], axis=0) if present else np.zeros(int(mask.sum()))
        e = {"dataset": name, "metric": metric, "slice": slice_, "queries": int(mask.sum()), "per_seed": per_seed,
             "mean_over_seeds": round(float(mean_q.mean()) if mean_q.size else 0.0, 4),
             "sd_over_seeds": round(float(np.std(list(per_seed.values()))), 4) if per_seed else None,
             "threshold": threshold, "at_or_above_threshold": bool(mean_q.size and mean_q.mean() >= threshold - 1e-12)}
        holds = e["at_or_above_threshold"]
        if paired:
            e["paired_vs"] = paired
            e["interval"] = R.paired_bootstrap(mean_q, refs[name][f"{R.M3B_REFERENCES[paired]}/{metric}"][mask])
            e["interval_excludes_zero_on_the_positive_side"] = bool(e["interval"]["low"] > 0)
            holds = holds and e["interval_excludes_zero_on_the_positive_side"]
        e["holds"] = bool(holds)
        confirmed = confirmed and bool(holds)
        entries.append(e)
    return {"seeds_present": [seed_of(k) for k in present], "complete": len(present) == 3, "cells": entries, "confirmed": bool(confirmed)}


def seed_confirmation(cfg: dict, gate: dict, rows: dict, refs: dict, arms: list | None = None) -> dict:
    """pilot_gate.on_pass: the pass is confirmed if the mean over seeds 0-2 also holds every cell. The mean is taken
    per query over the seeds present, on V2_GATE (the cells are gate-half quantities); a paired cell's interval is the
    bootstrap of that per-query mean against the frozen reference. Without arms: the selected arms that passed their
    gate (the pilot); with arms (amendment 4): those selected arms whatever their gate outcome, the same aggregation."""
    cells = R.gate_cells(*R.gate_thresholds(cfg)[:2])
    out: dict = {}
    for family, arm in gate["selection"].items():
        if arms is None and gate["outcome"].get(arm) != "PASS":
            continue
        if arms is not None and arm not in arms:
            continue
        out[arm] = {"family": family, **seed_mean_cells(cells[family], arm, rows, refs), "half": GATE}
    return out


def section_confirmation(conf: dict, seeds_gate: dict) -> list[str]:
    lines = [f"## 5. Seed confirmation ({GATE}, seeds 0-2)", ""]
    if not conf:
        lines += ["No arm passed its gate; no seeds 1-2 were fitted (pilot_gate.on_fail).", ""]
    for arm, v in conf.items():
        status = "CONFIRMED_PASS" if v["confirmed"] else ("CONFIRMATION_FAIL" if v["complete"] else f"INCOMPLETE ({len(v['seeds_present'])} of 3 seeds)")
        lines += [f"**`{arm}` — {status}** (seeds present: {v['seeds_present']}; the mean is per query over the seeds, on {GATE})", "",
                  "| dataset | metric | slice | per seed | mean over seeds | sd | threshold | paired interval of the seed mean | cell |", "|---|---|---|---|---|---|---|---|---|"]
        for c in v["cells"]:
            per = ", ".join(f"s{s} {x:.4f}" for s, x in c["per_seed"].items())
            interval = f"{ci(c['interval'])} vs `{c['paired_vs']}`" if c.get("paired_vs") else "—"
            lines.append(f"| {c['dataset']} | {c['metric']} | {c['slice']} | {per} | {c['mean_over_seeds']:.4f} | {c['sd_over_seeds']:.4f} | {c['threshold']:.4f} | {interval} | {'holds' if c['holds'] else 'FAILS'} |")
        lines.append("")
    lines += [f"Seed mean ± sd on {GATE} for every arm with more than one seed:", "", "| dataset | arm | seeds | recall@5 | hit@1 | full_coverage@5 | mrr |", "|---|---|---|---|---|---|---|"]
    for name in R.PILOT:
        for arm, s in seeds_gate.get(name, {}).items():
            if len(s["seeds"]) < 2:
                continue
            lines.append(f"| {name} | `{arm}` | {s['seeds']} | " + " | ".join(f"{s[m]['mean']:.4f} ± {s[m]['sd']:.4f}" for m in SEED_METRICS) + " |")
    lines.append("")
    return lines


def section_held(held: dict, gate: dict) -> list[str]:
    lines = [f"## 6. {HELD} — the reported column (read once)", "",
             f"Read once at {held['utc']} (status `{held['status_at_read']}`, gate record `{held['gate_record_sha256'][:12]}…`), after the gate was "
             "filed; no selection, threshold, screen or checkpoint decision saw it. Every v2 arm, the fixed scorers and the frozen M3B references on "
             f"the same queries (the frozen `fixed:rrf` array equals the v2 row per query, section 10, and is listed once). The gate-half reads of "
             f"the same arms are section 4 ({GATE}, seed 0) and the gate record's summary; the counts of both halves are beside each population.", ""]
    for name in R.PILOT:
        n = held["queries"][name]
        lines += [f"**{name}** ({n:,} held queries; {gate['queries_V2_GATE'][name]:,} gate queries)", "",
                  "| scorer | " + " | ".join(TABLE_METRICS) + " |", "|---|" + "---|" * len(TABLE_METRICS)]
        for key, s in held["summary"][name].items():
            shown = f"`{key}`" if label(key) == key else f"{label(key)} `{key}`"
            lines.append(f"| {shown} | " + " | ".join(fmt(s.get(m), 4) for m in TABLE_METRICS) + " |")
        for ref, s in held["frozen_references"][name].items():
            if ref == "fixed_rrf":
                continue      # identical to the v2 fixed:rrf row per query (m3b_fixed_rrf_agreement); in the record, listed once here
            lines.append(f"| {DISPLAY.get(ref, ref)} `{R.M3B_REFERENCES[ref]}` | — | {s['recall@5']:.4f} | — | {s['hit@1']:.4f} | {s['mrr']:.4f} | {s['full_coverage@5']:.4f} |")
        lines.append("")
        seeds = held["seeds"].get(name, {})
        multi = {arm: s for arm, s in seeds.items() if len(s["seeds"]) > 1}
        if multi:
            lines += ["| arm | seeds | recall@5 mean ± sd | hit@1 | full_coverage@5 | mrr |", "|---|---|---|---|---|---|"]
            for arm, s in multi.items():
                lines.append(f"| `{arm}` | {s['seeds']} | " + " | ".join(f"{s[m]['mean']:.4f} ± {s[m]['sd']:.4f}" for m in SEED_METRICS) + " |")
            lines.append("")
    lines += ["The gate cells re-read on the held half for the arms the gate selected (confirmatory: the verdict is the gate's, on the gate half):", ""]
    for arm, v in held["gate_cells_confirmatory"].items():
        lines += [f"`{arm}` (gate outcome {gate['outcome'][arm]}):", "", "| dataset | metric | slice | queries | value | gate threshold | paired interval | at threshold |", "|---|---|---|---|---|---|---|---|"]
        lines += cell_rows(v)
        lines.append("")
    lines += [f"Paired differences on {HELD} (per query, bootstrap {held['bootstrap']['resamples']} resamples, `default_rng({held['bootstrap']['seed']})`, "
              f"{held['bootstrap']['level']} % percentile); the same pairs on {GATE} are in the gate record.", "",
              "| pair | " + " | ".join(f"{n} recall@5 / hit@1 / full_coverage@5" for n in R.PILOT) + " |", "|---|" + "---|" * len(R.PILOT)]
    for pair, per in held["paired"].items():
        cells = []
        for name in R.PILOT:
            d = per.get(name)
            cells.append("—" if not d else " / ".join(ci(d[m]) for m in ("recall@5", "hit@1", "full_coverage@5")))
        lines.append(f"| {pair.replace('_', ' ')} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def section_slices(held: dict, gate: dict) -> list[str]:
    lines = ["## 7. Slices (measurement.slices_reported)", ""]
    for half_name, src in ((HELD, held["slices"]), (GATE, gate["slices_V2_GATE"])):
        hop = src["metaqa_by_hop"]
        keys = sorted({k for h in hop.values() for k in h if k != "queries"})
        lines += [f"**metaqa by hop, hit@1 / recall@5, {half_name}**", "", "| scorer | " + " | ".join(f"{h} ({hop[h]['queries']:,})" for h in hop) + " |", "|---|" + "---|" * len(hop)]
        for k in keys:
            lines.append(f"| `{k}` | " + " | ".join(f"{hop[h][k]['hit@1']:.4f} / {hop[h][k]['recall@5']:.4f}" for h in hop) + " |")
        lines.append("")
        lines += [f"**recall@5 by the gold's STRUCT distance from the seeds, {half_name}** (dist0..dist3, unreached; queries in brackets)", ""]
        for name in R.PILOT:
            d = src["gold_distance_recall5"][name]
            if not d:
                continue
            ks = sorted({k for b in d.values() for k in b if k != "queries"})
            lines += [f"{name}:", "", "| scorer | " + " | ".join(f"{b} ({d[b]['queries']:,})" for b in d) + " |", "|---|" + "---|" * len(d)]
            for k in ks:
                lines.append(f"| `{k}` | " + " | ".join(f"{d[b][k]:.4f}" for b in d) + " |")
            lines.append("")
        lines += [f"**multi-gold queries (2+ in-pool golds), full_coverage@5 / recall@5, {half_name}**", "",
                  "| scorer | " + " | ".join(f"{n} ({src['multi_gold'][n]['queries']:,})" for n in R.PILOT) + " |", "|---|" + "---|" * len(R.PILOT)]
        ks = sorted({k for n in R.PILOT for k in src["multi_gold"][n] if k != "queries"})
        for k in ks:
            lines.append(f"| `{k}` | " + " | ".join(("—" if src["multi_gold"][n][k]["recall@5"] is None else f"{src['multi_gold'][n][k]['full_coverage@5']:.4f} / {src['multi_gold'][n][k]['recall@5']:.4f}") for n in R.PILOT) + " |")
        lines.append("")
    return lines


def section_mechanism(held: dict, gate: dict) -> list[str]:
    lines = ["## 8. Mechanism readouts (measurement.mechanism_readouts)", "",
             "`delta_ratio` = mean |delta_s| / mean |base_z| over the pool; `top1_changed` = fraction of queries whose top-1 leaves the fixed base; "
             "`gate_step{t}` = the update gate g per step (mean, q25 / q50 / q75), `gate2_step{t}` the evidence gate, `block_gate{b}` the mix twin's block gates.", ""]
    for half_name, src in ((HELD, held["mechanism_readouts"]), (GATE, gate["mechanism_readouts_V2_GATE"])):
        lines += [f"**{half_name}**", "", "| dataset | scorer | delta_ratio | top1_changed | gates (mean, q25 / q50 / q75) |", "|---|---|---|---|---|"]
        for name in R.PILOT:
            for key, e in src[name].items():
                gates = "; ".join(f"{g} {v['mean']:.3f} ({v['q25']:.2f} / {v['q50']:.2f} / {v['q75']:.2f})" for g, v in e.items() if isinstance(v, dict))
                lines.append(f"| {name} | `{key}` | {fmt(e.get('delta_ratio'), 4)} | {fmt(e.get('top1_changed'), 4)} | {gates or '—'} |")
        lines.append("")
    return lines


def section_calibration(cfg: dict, cfg_m3b: dict, held: dict) -> list[str]:
    """calibration_for_the_report and methodological_ruling_band_rule: the published cells with their exposure, our
    rows from the frozen M3B pools (identical in v2), and the band check on the selected GNN's held numbers, labelled
    diagnostic (its operational form was written mid-M3B-eval; it is never pre-registered and decides nothing)."""
    sota = cfg_m3b["measurement"]["sota_column"]
    gnn = R.fit_key(held["selection"]["gnn"], 0)
    lines = ["## 9. Calibration against the published systems (diagnostic; exposure attached)", "",
             f"{capital(cfg['methodological_ruling_band_rule']['ruling'])} Families: {str(cfg['calibration_for_the_report']['families']).strip()}. "
             f"Rule: {str(cfg['calibration_for_the_report']['rule']).strip()}.", "",
             f"| dataset | published (exposure) | selected GNN `{gnn}` on {HELD} | diagnostic band check |", "|---|---|---|---|"]
    for name in R.PILOT:
        pub = str(sota["cells"].get(name, "—"))
        band = M3B_REPORT.published_band(cfg_m3b, name)
        short = M3B_REPORT.exposure_shortfall(cfg_m3b, name) or 0.0
        if name == "metaqa":
            hops = held["slices"]["metaqa_by_hop"]
            ours = {h: hops[h][gnn]["hit@1"] for h in hops if gnn in hops[h]}
            parts, inside = [], True
            for h, low in band["per_hop_low"].items():
                if h in ours:
                    ok = ours[h] >= low - short
                    inside &= ok
                    parts.append(f"{h} {ours[h]:.3f} vs {low:.3f}" + ("" if ok else f" (below by {low - ours[h]:.3f}; exposure shortfall {short:.3f})"))
            verdict = "READ" if inside else "NOT_READ"
            lines.append(f"| metaqa | {pub} (KB exposure) | hit@1 " + " / ".join(f"{h} {v:.3f}" for h, v in ours.items()) + f" (all {held['summary']['metaqa'][gnn]['hit@1']:.3f}) | {verdict}: {'; '.join(parts)} |")
        elif band.get("metric"):
            ours = held["summary"][name][gnn][band["metric"]]
            ok = ours >= band["low"] - short
            why = f"{band['metric']} {ours:.3f} vs published {band['low']:.3f}-{band['high']:.3f}" + ("" if ok else f" (below by {band['low'] - ours:.3f})")
            lines.append(f"| {name} | {pub} | R@5 {held['summary'][name][gnn]['recall@5']:.3f} / R@10 {held['summary'][name][gnn]['recall@10']:.3f} | {'READ' if ok else 'NOT_READ'}: {why} |")
        else:
            lines.append(f"| {name} | {pub} | R@5 {held['summary'][name][gnn]['recall@5']:.3f} | CONTROL: no published graph-retrieval number; the fixed rrf row stands in |")
    lines.append("")
    notes = [v for k, v in cfg_m3b.items() if k.startswith("note_") and isinstance(v, dict) and "calibration_columns_for_the_report" in v]
    pools = cfg_m3b["candidate_contract_frozen_2026_09_13"]["per_dataset"]
    if notes:
        table = notes[-1]["calibration_columns_for_the_report"]
        lines += ["| method | graph entry point | avg graph / pool | any-answer exposure | retrieval prior retained? | ranking metric |", "|---|---|---|---|---|---|"]
        for row in table["published_rows"]:
            lines.append("| " + " | ".join(str(row[c]).replace("|", "&#124;") for c in table["columns"]) + " |")
        for name in R.PILOT:
            p = pools[name]
            c = p["construction"]
            entry = "Dense + SPLADE seeds; base pool `" + c["base_pool"] + "`" + (f"; expansion `{c['regime']}:{c['setting']['name']}`" if c.get("setting") else "; no graph expansion (retrieval-only pool)")
            lines.append(f"| **ours ({name}): every v2 arm and every frozen reference** | {entry} | {p['candidates_mean']:,.1f} candidates | any-gold {p['any_gold_at_pool']:.4f}, "
                         f"all-gold {p['all_gold_at_pool']:.4f}, pool ceiling@5 {p['recall_ceiling@5']:.4f} | yes — retrieval columns and the fixed base score | "
                         f"{'hit@1, R@5' if name == 'metaqa' else 'R@5 / R@10'} |")
        lines.append("")
    return lines


def section_audit(cfg: dict, evals: dict) -> list[str]:
    declared = cfg["m3b_incumbents"]["eval_populations_reused_here"]
    lines = ["## 10. Audit", "", "| record | population digest = M3B | halves (gate / held) = filed | MRR audit | fixed rrf = M3B per query (max abs diff) | ceiling@5 as compiled |", "|---|---|---|---|---|---|"]
    for stem, r in sorted(evals.items()):
        name = r["dataset"]
        counts = R.declared_half_counts(cfg, name) or {}
        halves_ok = r["halves"][GATE] == counts.get(GATE) and r["halves"][HELD] == counts.get(HELD)
        digest_ok = r["ids_sha256"] == declared[name]["ids_sha256"] and r["queries"] == declared[name]["queries"]
        audit_ok = all(v["ok"] for v in r["mrr_audit"].values())
        agree = r.get("m3b_fixed_rrf_agreement")
        if agree and "per_shard" in agree:
            diff = max(max(a["max_abs_diff"].values()) for a in agree["per_shard"] if a)
        else:
            diff = max(agree["max_abs_diff"].values()) if agree else None
        lines.append(f"| `{stem}` | {digest_ok} | {r['halves'][GATE]:,} / {r['halves'][HELD]:,}: {halves_ok} | {audit_ok} ({len(r['mrr_audit'])} scorers) | "
                     f"{'—' if diff is None else f'{diff:.1e}'} | {r['ceiling_as_compiled']['recall_ceiling@5']:.4f} |")
    lines.append("")
    lines.append("The CRAG package is read-only; the foreign `canonical.cpython-313.pyc` (37,614 B) is reported in every run's note and never removed. "
                 "No population statistic was estimated and stored; the input block sees raw columns and within-query z-scores only.")
    lines.append("")
    return lines


def read_hypotheses(cfg: dict, held: dict, gate: dict, conf: dict) -> dict:
    """The four pilot hypotheses filed before any fit (hypotheses), each read on its declared measurement: H_pilot on the
    gate (V2_GATE); the others on the reported column (V2_HELD_CONFIRMATION) with the gate-half reading beside."""
    hyp = cfg["hypotheses"]
    sel = held["selection"]
    gnn, ef, plain = R.fit_key(sel["gnn"], 0), R.fit_key("u_gnn_v2_ef", 0), R.fit_key("u_gnn_v2", 0)
    out = {}
    passed = all(gate["outcome"][arm] == "PASS" for arm in sel.values())
    confirmed = {arm: v["confirmed"] for arm, v in conf.items()}
    out["H_pilot"] = {"claim": hyp["H_pilot"]["claim"], "half": GATE, "holds": passed, "outcome": gate["outcome"], "seed_confirmation": confirmed}
    for half_name, paired in ((HELD, held["paired"]), (GATE, gate["paired_on_V2_GATE"])):
        a = paired.get("selected_gnn_minus_u_gnn_v2_core78", {}).get("metaqa")
        b = paired.get("u_gnn_v2_core78_minus_gat_universal_v1_trio", {}).get("metaqa")
        entry = {"claim": hyp["H_operator_vs_basis"]["claim"], "gnn_minus_core78_hit1": a["hit@1"] if a else None,
                 "core78_minus_trio_hit1": b["hit@1"] if b else None}
        entry["holds"] = bool(a and b and a["hit@1"]["low"] > 0 and b["hit@1"]["low"] > 0) if (a and b) else None
        out.setdefault("H_operator_vs_basis", {})[half_name] = entry
    for half_name, src in ((HELD, held), (GATE, gate)):
        slices = src["slices"] if half_name == HELD else src["slices_V2_GATE"]
        mech = src["mechanism_readouts"] if half_name == HELD else src["mechanism_readouts_V2_GATE"]
        hop3 = slices["metaqa_by_hop"].get("3hop", {})
        multi = slices["multi_gold"].get("2wiki", {})
        e = {"claim": hyp["H_evidence_flow"]["claim"]}
        if ef in hop3 and plain in hop3 and ef in multi and plain in multi:
            e["metaqa_3hop_hit1"] = {"u_gnn_v2_ef": hop3[ef]["hit@1"], "u_gnn_v2": hop3[plain]["hit@1"]}
            e["2wiki_multi_gold_full_coverage5"] = {"u_gnn_v2_ef": multi[ef]["full_coverage@5"], "u_gnn_v2": multi[plain]["full_coverage@5"]}
            g2 = {name: {k: v["mean"] for k, v in mech[name].get(ef, {}).items() if k.startswith("gate2_step")} for name in ("squad", "metaqa")}
            e["evidence_gate_g2_mean_per_step"] = g2
            lower = bool(g2["squad"]) and all(g2["squad"][k] < g2["metaqa"][k] for k in g2["squad"] if k in g2["metaqa"])
            e["holds"] = bool(hop3[ef]["hit@1"] > hop3[plain]["hit@1"] and (multi[ef]["full_coverage@5"] or 0) > (multi[plain]["full_coverage@5"] or 0) and lower)
        else:
            e["holds"] = None
            e["why"] = "u_gnn_v2_ef and u_gnn_v2 are not both scored on this half"
        out.setdefault("H_evidence_flow", {})[half_name] = e
        summary = src["summary"] if half_name == HELD else None
        d = {"claim": hyp["H_do_nothing"]["claim"], "per_arm": {}}
        for arm in sel.values():
            if gate["outcome"][arm] != "PASS":
                continue
            key = R.fit_key(arm, 0)
            ratios = {name: mech[name].get(key, {}).get("delta_ratio") for name in R.PILOT}
            entry = {"delta_ratio": ratios, "squad_smallest_of_the_trio": bool(ratios["squad"] is not None and all(ratios["squad"] <= (ratios[n] if ratios[n] is not None else ratios["squad"]) for n in R.PILOT))}
            if summary is not None:
                entry["squad_recall5"] = summary["squad"][key]["recall@5"]
                entry["fixed_rrf_recall5"] = summary["squad"]["fixed:rrf"]["recall@5"]
                entry["within_0.005_of_fixed_rrf"] = bool(entry["squad_recall5"] >= entry["fixed_rrf_recall5"] - 0.005)
            else:
                cell = next((c for c in gate["verdict"][arm]["cells"] if c["dataset"] == "squad"), None)
                entry["squad_recall5"] = cell["value"] if cell else None
                entry["within_0.005_of_fixed_rrf"] = cell["at_or_above_threshold"] if cell else None
            entry["holds"] = bool(entry["within_0.005_of_fixed_rrf"] and entry["squad_smallest_of_the_trio"]) if entry["within_0.005_of_fixed_rrf"] is not None else None
            d["per_arm"][arm] = entry
        d["holds"] = all(v["holds"] for v in d["per_arm"].values()) if d["per_arm"] else None
        out.setdefault("H_do_nothing", {})[half_name] = d
    return out


def section_reading(cfg: dict, held: dict, gate: dict, conf: dict, hyps: dict, terminal: dict | None = None) -> list[str]:
    lines = ["## 11. Reading", "", "Each hypothesis was filed before any fit (`hypotheses`) and is read on its declared measurement; "
             f"`holds` is a reading of the filed claim, nothing is re-thresholded. Forbidden framings are not used: {', '.join(repr(f) for f in cfg['forbidden_framings'])}.", ""]
    h = hyps["H_pilot"]
    lines.append(f"- **H_pilot** ({GATE}): {h['claim']} — {'holds' if h['holds'] else 'does not hold'}; outcome {h['outcome']}; seed confirmation {h['seed_confirmation'] or 'none (no pass)'}.")
    for name, key in (("H_operator_vs_basis", "H_operator_vs_basis"), ("H_evidence_flow", "H_evidence_flow"), ("H_do_nothing", "H_do_nothing")):
        parts = []
        for half_name in (HELD, GATE):
            e = hyps[key][half_name]
            verdict = "holds" if e["holds"] else ("does not hold" if e["holds"] is False else "not readable")
            if key == "H_operator_vs_basis":
                detail = f"GNN − core78 hit@1 {ci(e['gnn_minus_core78_hit1'])}; core78 − trio {ci(e['core78_minus_trio_hit1'])}"
            elif key == "H_evidence_flow":
                detail = (f"3hop hit@1 EF {e['metaqa_3hop_hit1']['u_gnn_v2_ef']:.4f} vs {e['metaqa_3hop_hit1']['u_gnn_v2']:.4f}; 2wiki multi-gold FC@5 "
                          f"{fmt(e['2wiki_multi_gold_full_coverage5']['u_gnn_v2_ef'], 4)} vs {fmt(e['2wiki_multi_gold_full_coverage5']['u_gnn_v2'], 4)}; g2 per step "
                          + "; ".join(f"{n} " + ", ".join(f"{v:.3f}" for v in g.values()) for n, g in e["evidence_gate_g2_mean_per_step"].items())) if "metaqa_3hop_hit1" in e else e.get("why", "")
            else:
                detail = "; ".join(f"`{arm}` squad R@5 {fmt(v['squad_recall5'], 4)} (within 0.005 of fixed rrf: {v['within_0.005_of_fixed_rrf']}), delta_ratio "
                                   + ", ".join(f"{n} {fmt(r, 4)}" for n, r in v["delta_ratio"].items()) + f" (squad smallest: {v['squad_smallest_of_the_trio']})" for arm, v in e["per_arm"].items()) or "no passing arm"
            parts.append(f"{half_name}: {verdict} ({detail})")
        lines.append(f"- **{name}**: {hyps[key][HELD]['claim']} — " + "; ".join(parts) + ".")
    lines.append("")
    fam = family_labels(held["selection"], gate)
    passed = [R.FAMILY_GATES[f] for f in ("gnn", "twin") if fam[R.FAMILY_GATES[f]] == "PASS"]
    if fam["overall"] == "BOTH_PASS":
        pair = "both families passed their gate at seed 0, so the proposed universal pair (one GNN and its non-message-passing twin over one contract) has passed the pilot gate"
    elif fam["overall"] == "PILOT_FAILED":
        pair = "neither family passed its gate at seed 0; the proposed universal pair has not passed the pilot gate"
    else:
        pair = (f"only {passed[0]} passed at seed 0; this is a pass of that family and is written as such -- the proposed universal pair has NOT passed the pilot gate, "
                "and no later document may describe this outcome as a pass of the pair")
    lines.append(f"**Family outcome** (amendment 2 family_status_vocabulary): {R.FAMILY_GATES['gnn']} {fam[R.FAMILY_GATES['gnn']]}, {R.FAMILY_GATES['twin']} "
                 f"{fam[R.FAMILY_GATES['twin']]}, overall **{fam['overall']}** -- {pair}.")
    lines.append("")
    if terminal:
        ff = terminal["family_final"]
        confirmed = [g for g in (R.FAMILY_GATES["gnn"], R.FAMILY_GATES["twin"]) if ff[g] == "CONFIRMED_PASS"]
        if terminal["terminal"] == "BOTH_CONFIRMED":
            sentence = "both families hold every cell on the mean over seeds 0-2, so the proposed universal pair is confirmed on V2_GATE"
        elif terminal["terminal"] == "PILOT_FAILED":
            sentence = "no family is confirmed; the proposed universal pair is not confirmed and no family is rescued"
        else:
            sentence = (f"only {confirmed[0]} is confirmed on the mean over seeds 0-2; this is a confirmation of that family and is written as such -- "
                        "the proposed universal pair is NOT confirmed")
        lines.append(f"**Terminal state** (amendment 3 terminal_state_vocabulary): {R.FAMILY_GATES['gnn']} {ff[R.FAMILY_GATES['gnn']]}, {R.FAMILY_GATES['twin']} "
                     f"{ff[R.FAMILY_GATES['twin']]}, terminal **{terminal['terminal']}** -- {sentence}.")
        lines.append("")
    lines.append(f"What this pilot is: {str(cfg['result_is']).strip().rstrip('.') if 'result_is' in cfg else 'the pilot of the declared innovations on the trio'}. "
                 "What it is not: a result on test data (none was read), a statement about message passing (M3B answers the paper's question), "
                 "a comparison to the published systems beyond the diagnostic calibration of section 9, or a six-dataset result (later_stages need their own dated authorization).")
    lines.append("")
    return lines


def run_record(files: list[Path]) -> tuple[list[str], dict]:
    record = {}
    for p in files:
        if p.exists():
            try:
                record[p.relative_to(ROOT).as_posix()] = R.sha256_file(p)
            except ValueError:
                record[p.as_posix()] = R.sha256_file(p)
    lines = ["## 12. Run record", "", f"Rendered {R.utc()} by `scripts/universal_v2_report.py --stage doc` from the files below (sha256 of every sidecar a number above "
             "cites; the sidecars are gitignored, the record is committed in the declaration's run_record block). The held record is read, never recomputed.", "",
             "| file | sha256 |", "|---|---|"]
    for k, v in record.items():
        lines.append(f"| `{k}` | `{v}` |")
    lines.append("")
    return lines, record


def stage_doc(cfg: dict, cfg_m3b: dict, log=print) -> Path:
    """docs/UNIVERSAL_V2_PILOT.md from the sidecars: the held column from held_record.json only; the gate-half reads
    (seed confirmation, seed table) sliced to V2_GATE as the arrays are read."""
    held = R.read_json(R.OUT / "held_record.json")
    if held is None:
        raise SystemExit("no held_record.json: the held column is produced once by --stage held before the document (order_of_operations 7)")
    gate = R.read_json(R.OUT / "gate_record.json")
    sel = R.read_json(R.OUT / "selection.json")
    if gate is None or sel is None:
        raise SystemExit("gate_record.json and selection.json are read by the document; one is missing")
    if held["gate_record_sha256"] != R.sha256_file(R.OUT / "gate_record.json"):
        raise SystemExit("gate_record.json changed after the held record was written; refusing")
    screen = R.read_json(R.OUT / "feature_screen.json") or {}
    timing = R.read_json(R.OUT / "timing.json")
    fits = {}
    for p in sorted(R.FITS.glob("*.json")) if R.FITS.exists() else []:
        r = R.read_json(p)
        fits[r["key"]] = r
    evals, rows, refs = {}, {}, {}
    for name in R.PILOT:
        for stem, rec in eval_records(name):
            evals[stem] = rec
        rows[name], mask, _ = pooled_rows(name, GATE)
        refs[name] = m3b_rows(name, mask)
    conf = seed_confirmation(cfg, gate, rows, refs)
    seeds_gate = {name: seed_stats(rows[name]) for name in R.PILOT}
    hyps = read_hypotheses(cfg, held, gate, conf)
    terminal = R.terminal_state(family_labels(held["selection"], gate), conf)
    lines = section_header(cfg, held, gate, terminal)
    lines += section_contract(cfg, screen)
    lines += section_cost(timing, fits, evals)
    lines += section_selection(sel)
    lines += section_gate(cfg, gate)
    lines += section_confirmation(conf, seeds_gate)
    lines += section_held(held, gate)
    lines += section_slices(held, gate)
    lines += section_mechanism(held, gate)
    lines += section_calibration(cfg, cfg_m3b, held)
    lines += section_audit(cfg, evals)
    lines += section_reading(cfg, held, gate, conf, hyps, terminal)
    files = [R.CONFIG, R.M3B_CONFIG, R.OUT / "held_record.json", R.OUT / "gate_record.json", R.OUT / "selection.json", R.OUT / "timing.json",
             R.OUT / "feature_screen.json", R.OUT / "feature_contract.json"]
    files += sorted(R.FITS.glob("*.json")) if R.FITS.exists() else []
    for stem in sorted(evals):
        files += [R.EVAL / f"{stem}.json", R.EVAL / f"{stem}.npz", R.EVAL / f"{stem}_query_ids.json"]
    files += [R.M3B_OUT / "eval" / f"{name}.npz" for name in R.PILOT]
    rec_lines, record = run_record(files)
    lines += rec_lines
    DOC = R.DOC
    DOC.parent.mkdir(parents=True, exist_ok=True)
    with open(DOC, "w", encoding="utf-8", newline=LF) as f:
        f.write(LF.join(lines) + LF)
    (R.OUT / "report_run_record.json").write_text(json.dumps({"utc": R.utc(), "doc": DOC.relative_to(ROOT).as_posix() if DOC.is_relative_to(ROOT) else DOC.as_posix(), "doc_sha256_lf": R.lf_sha256(DOC),
                                                             "files": record, "seed_confirmation": conf, "terminal_state": terminal, "hypotheses": hyps}, indent=1), encoding="utf-8")
    log(f"wrote {DOC} ({len(lines)} lines)")
    return DOC


# ── amendment 4: the post-pilot replication ──────────────────────────────────


def replication_premises(cfg: dict) -> dict:
    """What the replication stands on, verified before any number is read: the amendment, the status line it applies
    at, the pilot's run record with the terminal state it cites, the pinned sidecars unchanged, the selected arms one
    and the same in selection.json, gate_record.json and the amendment, the thresholds block the amendment names."""
    rep_key, rep = R.replication_amendment(cfg)
    if cfg.get("status") != rep["applies_at_status"]:
        raise SystemExit(f"status {cfg.get('status')}: the replication applies at {rep['applies_at_status']} (amendment 4)")
    runs = R.dated_blocks(cfg, "run_record")
    if not runs:
        raise SystemExit("no run_record_<date>: the replication follows the closed pilot")
    run = cfg[runs[-1]]
    terminal = (run.get("terminal_state") or {}).get("terminal")
    if terminal != rep["original_pilot_status"]:
        raise SystemExit(f"{runs[-1]} closed at {terminal}, not the {rep['original_pilot_status']} the amendment cites")
    pins = {}
    for name in ("selection.json", "gate_record.json", "held_record.json"):
        pinned = run.get(name.replace(".json", "") + "_sha256")
        got = R.sha256_file(R.OUT / name)
        if pinned != got:
            raise SystemExit(f"{name} ({got[:12]}) is not the file {runs[-1]} pins ({str(pinned)[:12]}); nothing of the pilot is edited")
        pins[name] = got
    gate, sel = R.read_json(R.OUT / "gate_record.json"), R.read_json(R.OUT / "selection.json")
    selected = {fam: sel[fam]["arm"] for fam in ("gnn", "twin")}
    if selected != dict(rep["selected_arms"]) or dict(gate["selection"]) != selected:
        raise SystemExit("the selected arms of selection.json, gate_record.json and the amendment differ; refusing")
    thresholds_from = R.gate_thresholds(cfg)[2]
    if not str(rep["thresholds"]).startswith(thresholds_from):
        raise SystemExit(f"the gate thresholds come from {thresholds_from}, not the block the amendment names ({rep['thresholds']})")
    return {"amendment": rep_key, "rep": rep, "run_record_block": runs[-1], "run": run, "pins": pins, "gate": gate, "selected": selected,
            "thresholds_from": thresholds_from}


def replication_fits(premises: dict) -> dict:
    """The three seeds of each selected arm: seed 0 as the run record pins it, seeds 1-2 as the amendment authorises
    them -- fitted under it, or under pilot_gate.on_pass for an arm that passed its gate; every weight file hashes
    to its record."""
    rep, run, runs = premises["rep"], premises["run"], premises["run_record_block"]
    fits = {}
    for fam, arm in premises["selected"].items():
        for seed in (0, 1, 2):
            k = R.fit_key(arm, seed)
            rec = R.read_json(R.FITS / f"{k}.json")
            if rec is None or not (R.FITS / f"{k}.pt").exists():
                raise SystemExit(f"{k}: no fit record and weights; the replication reads three seeds of each selected arm")
            state = R.sha256_file(R.FITS / f"{k}.pt")
            if state != rec["state_sha256"]:
                raise SystemExit(f"{k}: the weights do not hash to the record")
            if seed == 0 and ((run.get("fits") or {}).get(k) or {}).get("state_sha256") != state:
                raise SystemExit(f"{k}: the seed-0 weights differ from what {runs} pins")
            if seed != 0:
                if seed not in (rep.get("authorised_fits") or {}).get(arm, []):
                    raise SystemExit(f"{k}: not an authorised fit of {premises['amendment']}")
                passed = bool(premises["gate"]["verdict"].get(arm, {}).get("pass"))   # pilot_gate.on_pass fitted seeds 1-2 of a passing arm
                if not passed and (rec.get("replication") or {}).get("block") != premises["amendment"]:
                    raise SystemExit(f"{k}: not fitted under {premises['amendment']} (the arm did not pass its gate, so the pilot fitted no seeds 1-2)")
            fits[k] = {"arm": arm, "seed": seed, "family": fam, "record_sha256": R.sha256_file(R.FITS / f"{k}.json"), "state_sha256": state,
                       "seconds": rec["seconds"], "best_epoch": rec["best_epoch"], "epochs_run": rec["epochs_run"], "parameters": rec["parameters"],
                       "best_select_macro_recall5": rec["best_select_macro_recall5"], "peak_rss_bytes": rec["peak_rss_bytes"], "threads": rec["threads"],
                       "utc": rec["utc"], "authorised_by": (rec.get("replication") or {}).get("block"),
                       "checks_against_seed_0": (rec.get("replication") or {}).get("checks_against_seed_0")}
    return fits


def stage_replication(cfg: dict, log=print) -> dict:
    """amendment 4 post_pilot_replication: the V2_GATE reading of seeds 0-2 of the two selected arms by the declared
    aggregation (seed_mean_cells, the pilot's seed confirmation) against the original frozen cells, once, into
    replication_record.json with REPLICATION_PASS | REPLICATION_FAIL per family and the overall status; then, for a
    REPLICATION_PASS family only, the held half read once (replication_held). Refused a second time, at any other
    status, before the pilot's run record, on a changed pinned sidecar, on a missing seed or an unscored key."""
    path = R.OUT / "replication_record.json"
    if path.exists():
        raise SystemExit("replication_record.json exists; the replication is read once (amendment 4)")
    premises = replication_premises(cfg)
    rep, selected, gate = premises["rep"], premises["selected"], premises["gate"]
    fits = replication_fits(premises)
    rows, refs, meta, masks = {}, {}, {}, {}
    for name in R.PILOT:
        rows[name], masks[name], meta[name] = pooled_rows(name, GATE)
        refs[name] = m3b_rows(name, masks[name])
        missing = [k for k in fits if f"{k}/recall@5" not in rows[name]]
        if missing:
            raise SystemExit(f"{name}: {missing} not scored on the population; the supplement pass precedes the reading")
    conf = seed_confirmation(cfg, gate, rows, refs, arms=list(selected.values()))
    family_status, passing = {}, []
    for fam, arm in selected.items():
        c = conf[arm]
        for cell in c["cells"]:
            cell["margin_of_the_seed_mean"] = round(cell["mean_over_seeds"] - cell["threshold"], 4)
            cell["per_seed_margin"] = {s: round(v - cell["threshold"], 4) for s, v in cell["per_seed"].items()}
        ok = bool(c["complete"] and c["confirmed"])
        c["replication_status"] = R.REPLICATION_FAMILY[0] if ok else R.REPLICATION_FAMILY[1]
        family_status[R.FAMILY_GATES[fam]] = c["replication_status"]
        if ok:
            passing.append(fam)
    overall = R.REPLICATION_OVERALL[(family_status["GNN_GATE"] == R.REPLICATION_FAMILY[0], family_status["TWIN_GATE"] == R.REPLICATION_FAMILY[0])]
    keys = set(fits)
    seeds_gate = {name: seed_stats({k: v for k, v in rows[name].items() if "/" not in k or k.split("/")[0] in keys}) for name in R.PILOT}
    record = {"utc": R.utc(), "amendment": premises["amendment"], "half": GATE, "read_once": True, "status_at_read": cfg["status"],
              "original_pilot_status": rep["original_pilot_status"], "original_pilot_commit": rep["original_pilot_commit"],
              "run_record_block": premises["run_record_block"], "pins_verified": premises["pins"], "selected": selected,
              "outcome_on_V2_GATE_at_seed_0": gate["outcome"], "thresholds_from": premises["thresholds_from"],
              "aggregation": "scripts/universal_v2_report.py seed_mean_cells (the pilot's seed confirmation, pilot_gate.on_pass): per cell the per-query "
                             "mean over seeds 0-2 on V2_GATE against the frozen threshold; a paired cell by the declared bootstrap of that per-query mean",
              "family_status": family_status, "replication_status": overall, "passing_families": passing, "never": "PILOT_PASS",
              "confirmation": conf, "seeds": seeds_gate, "fits": fits, "records_read": meta, "bootstrap": R.BOOTSTRAP,
              "queries": {name: int(masks[name].sum()) for name in R.PILOT},
              "m3b_reference_arrays": {name: R.sha256_file(R.M3B_OUT / "eval" / f"{name}.npz") for name in R.PILOT},
              "interpretation": rep["interpretation_rule"][overall],
              "what_this_is": "the post-pilot replication of amendment 4 on V2_GATE: a separate status beside the original PILOT_FAILED, never a rename "
                              "of it; GNN_REPLICATION_ONLY and TWIN_REPLICATION_ONLY replicate that family and never the proposed universal pair"}
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    for arm, c in conf.items():
        log(f"replication {arm} ({c['family']}): {c['replication_status']}; " + "; ".join(
            f"{e['dataset']}/{e['metric']}/{e['slice']} mean {e['mean_over_seeds']} vs {e['threshold']} ({'holds' if e['holds'] else 'FAILS'})" for e in c["cells"]))
    log(f"ORIGINAL PILOT STATUS: {rep['original_pilot_status']}; POST-PILOT REPLICATION STATUS: {overall}; wrote {path}")
    if passing:
        replication_held(cfg, premises, passing, R.sha256_file(path), log=log)
    else:
        log("no family is REPLICATION_PASS: V2_HELD_CONFIRMATION is not read (amendment 4 held_confirmation.not_read_for)")
    return record


def replication_held(cfg: dict, premises: dict, passing: list, record_sha: str, log=print) -> dict:
    """amendment 4 held_confirmation: V2_HELD_CONFIRMATION read once for the REPLICATION_PASS families only, by the
    declared held procedure and nothing new (stage_held: the gate cells re-read on the held half, labelled confirmatory,
    no threshold of their own) -- for seeds 1 and 2 and for the per-query mean over seeds 0-2 by the same aggregation;
    the seed-0 held cells are those held_record.json already carries and are copied, not re-read. The arrays of a
    REPLICATION_FAIL family are dropped as they are read and no number of them is computed or written."""
    path = R.OUT / "replication_held_record.json"
    if path.exists():
        raise SystemExit("replication_held_record.json exists; the held half is read once under the replication (amendment 4)")
    selected = premises["selected"]
    keep = {R.fit_key(selected[fam], s) for fam in passing for s in (0, 1, 2)}
    dropped = {R.fit_key(selected[fam], s) for fam in selected if fam not in passing for s in (0, 1, 2)}
    rows, masks, refs = {}, {}, {}
    for name in R.PILOT:
        r, masks[name], _ = pooled_rows(name, HELD)
        rows[name] = {k: v for k, v in r.items() if "/" not in k or k.split("/")[0] in keep or k.startswith("fixed:")}
        del r
        refs[name] = m3b_rows(name, masks[name])
    pilot_held = R.read_json(R.OUT / "held_record.json")
    cells = R.gate_cells(*R.gate_thresholds(cfg)[:2])
    out = {}
    for fam in passing:
        arm = selected[fam]
        per_seed = {"0": {**pilot_held["gate_cells_confirmatory"][arm], "from": "held_record.json (the pilot's read-once column), copied, not re-read"}}
        for s in (1, 2):
            per_seed[str(s)] = {**R.evaluate_cells(R.fit_key(arm, s), cells[fam], rows, refs), "confirmatory_not_a_gate": True}
        mean = {**seed_mean_cells(cells[fam], arm, rows, refs), "confirmatory_not_a_gate": True, "half": HELD}
        for cell in mean["cells"]:
            cell["margin_of_the_seed_mean"] = round(cell["mean_over_seeds"] - cell["threshold"], 4)
        out[arm] = {"family": fam, "gate_half_status": R.REPLICATION_FAMILY[0], "per_seed": per_seed, "seed_mean": mean,
                    "seeds": {name: seed_stats({k: v for k, v in rows[name].items() if "/" not in k or k.split("/")[0] in keep and arm_of(k.split("/")[0]) == arm})
                              for name in R.PILOT},
                    "read": "the gate cells re-read on V2_HELD_CONFIRMATION for seeds 1-2 and for the per-query mean over seeds 0-2; confirmatory, "
                            "no threshold of its own; the replication status is the gate half's (amendment 4 held_confirmation)"}
    record = {"utc": R.utc(), "amendment": premises["amendment"], "half": HELD, "read_once": True, "read_for": passing,
              "not_read_for": [fam for fam in selected if fam not in passing], "arrays_dropped_unread": sorted(dropped),
              "replication_record_sha256": record_sha, "queries": {name: int(masks[name].sum()) for name in R.PILOT},
              "cells_confirmatory": out, "bootstrap": R.BOOTSTRAP,
              "m3b_reference_arrays": {name: R.sha256_file(R.M3B_OUT / "eval" / f"{name}.npz") for name in R.PILOT},
              "what_this_is": "a Universal-v2 architecture-selection holdout, not a globally unseen dataset (check_1_the_halves.what_it_is_not); "
                              "the confirmatory column of the replication for the families that passed on V2_GATE; nothing here is a gate"}
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    for arm, v in out.items():
        log(f"held {arm}: seed mean " + "; ".join(f"{c['dataset']}/{c['metric']}/{c['slice']} {c['mean_over_seeds']}" for c in v["seed_mean"]["cells"]) + " (confirmatory, not a gate)")
    log(f"wrote {path} (read for {passing}; not read for {record['not_read_for']})")
    return record


def replication_cell_rows(c: dict) -> str:
    per = {s: c["per_seed"].get(s) for s in ("0", "1", "2")}
    seeds = " | ".join("—" if per[s] is None else f"{per[s]:.4f}" for s in ("0", "1", "2"))
    interval = f"{ci(c['interval'])} vs `{c['paired_vs']}`, excludes zero: {c['interval_excludes_zero_on_the_positive_side']}" if c.get("paired_vs") else "—"
    sd = "—" if c.get("sd_over_seeds") is None else f"{c['sd_over_seeds']:.4f}"
    return (f"| {c['dataset']} | {c['metric']} | {c['slice']} | {c['queries']:,} | {seeds} | {c['mean_over_seeds']:.4f} | {sd} | {c['threshold']:.4f} | "
            f"{c.get('margin_of_the_seed_mean', c['mean_over_seeds'] - c['threshold']):+.4f} | {interval} | {'holds' if c['holds'] else 'FAILS'} |")


CELL_HEADER = ["| dataset | metric | slice | queries | seed 0 | seed 1 | seed 2 | mean over seeds | sd | threshold | margin | paired interval of the seed mean | cell |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def stage_replication_doc(cfg: dict, log=print) -> Path:
    """docs/UNIVERSAL_V2_REPLICATION.md from replication_record.json and, where it exists, replication_held_record.json:
    the two statuses stated separately, the seed-wise cells, the means, sds, thresholds, margins and intervals, the
    held reading where authorised, the compute, the incidents, the hashes and the commits. The pilot document is
    not re-rendered. Every number comes from the records; nothing is recomputed here."""
    rec = R.read_json(R.OUT / "replication_record.json")
    if rec is None:
        raise SystemExit("no replication_record.json: --stage replication precedes the document (amendment 4 execution)")
    held = R.read_json(R.OUT / "replication_held_record.json")
    if rec["passing_families"] and held is None:
        raise SystemExit("a REPLICATION_PASS family reads the held half once before the document is rendered")
    if held is not None and held["replication_record_sha256"] != R.sha256_file(R.OUT / "replication_record.json"):
        raise SystemExit("replication_record.json changed after the held record was written; refusing")
    rep_key, rep = R.replication_amendment(cfg)
    run = cfg[rec["run_record_block"]]
    selected = rec["selected"]
    fits = {k: R.read_json(R.FITS / f"{k}.json") for k in rec["fits"]}
    evals = {}
    for name in R.PILOT:
        for stem, r in eval_records(name):
            evals[stem] = r
    incidents = R.read_json(R.OUT / "replication_incidents.json") or []
    try:
        import subprocess
        out = subprocess.run(["git", "log", "--format=%h %s", f"{rep['original_pilot_commit']}..HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        commits = [line for line in out.stdout.splitlines() if line.strip()]
    except Exception as e:   # noqa: BLE001  (the sandbox names no real commit)
        commits = [f"not read: {e}"]
    fam_label = {fam: f"{R.FAMILY_GATES[fam]} family (`{arm}`) **{rec['family_status'][R.FAMILY_GATES[fam]]}**" for fam, arm in selected.items()}
    lines = ["# Universal-v2 post-pilot replication (amendment 4)", "",
             f"**ORIGINAL PILOT STATUS: {rec['original_pilot_status']}** (commit `{rec['original_pilot_commit']}`, `{rec['run_record_block']}`, "
             "`docs/UNIVERSAL_V2_PILOT.md` as filed -- not re-rendered, not reinterpreted).", "",
             f"**POST-PILOT REPLICATION STATUS: {rec['replication_status']}** -- {fam_label['gnn']} / {fam_label['twin']}.", "",
             f"The two statuses are separate: nothing here renames the original pilot's result, and a replication pass is never `{rec['never']}` "
             f"(`{rep_key}` status_vocabulary). GNN_REPLICATION_ONLY and TWIN_REPLICATION_ONLY replicate that family only, never the proposed universal pair. "
             f"Read on {rec['half']} once ({rec['utc']}, status line `{rec['status_at_read']}`, unchanged). "
             f"Interpretation rule for this outcome (verbatim from the amendment): {rec['interpretation']}", ""]
    lines += ["## 1. What was frozen and verified", "",
              f"- The amendment `{rep_key}` was filed and committed before any fit it authorises; it authorises {rep['authorised_fits']} and nothing else.",
              f"- The selected arms are those of `selection.json` and `gate_record.json` ({selected}); the pilot's sidecars hash to what `{rec['run_record_block']}` pins: "
              + ", ".join(f"`{k}` `{v[:12]}…`" for k, v in rec["pins_verified"].items()) + ".",
              f"- Thresholds from `{rec['thresholds_from']}`, unchanged; the aggregation is {rec['aggregation']}; paired intervals: {rec['bootstrap']['resamples']} resamples, "
              f"`default_rng({rec['bootstrap']['seed']})`, {rec['bootstrap']['level']} % percentile.",
              "- Every seed-1/2 fit refused to start unless its training rule, parameter count, columns, core sha256, evidence fields and relation bank equalled the "
              "seed-0 record of the same arm (the fit records carry `replication.checks_against_seed_0`); seed 0 was not retrained (its records exist and are not repeated).",
              f"- M3B pins verified at every stage; the M3B reference arrays: " + ", ".join(f"{n} `{v[:12]}…`" for n, v in rec["m3b_reference_arrays"].items()) + ".",
              f"- Queries on {rec['half']}: " + ", ".join(f"{n} {q:,}" for n, q in rec["queries"].items()) + ".", ""]
    lines += ["## 2. The fits", "", "| key | family | seed | parameters | best epoch | epochs run | select macro R@5 | hours | peak RSS GB | authorised by | state sha256 (checkpoint hash) | record sha256 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rep_hours, spent = 0.0, 0.0
    for p in sorted(R.FITS.glob("*.json")) if R.FITS.exists() else []:
        spent += float(R.read_json(p).get("seconds", 0.0)) / 3600
    for k, f in rec["fits"].items():
        if f["seed"] != 0:
            rep_hours += float(f["seconds"]) / 3600
        lines.append(f"| `{k}` | {f['family']} | {f['seed']} | {f['parameters']:,} | {f['best_epoch']} | {f['epochs_run']} | {f['best_select_macro_recall5']:.4f} | "
                     f"{float(f['seconds']) / 3600:.2f} | {f['peak_rss_bytes'] / 2**30:.2f} | {f['authorised_by'] or 'pilot (seed 0)'} | `{f['state_sha256']}` | `{f['record_sha256']}` |")
    lines += ["", f"Compute: the replication fits took {rep_hours:.2f} fit-hours (one lane, {next(iter(fits.values()))['threads']} threads); "
              f"{spent:.2f} fit-hours spent in total against the ceiling of {R.FIT_HOURS_CEILING:.0f} (amendment 3 compute_ceiling_guard, unchanged). "
              "The supplement eval pass is not a fit and does not count.", ""]
    lines += ["## 3. The supplement eval pass", "", "| record | dataset | models | queries | V2_GATE / V2_HELD_CONFIRMATION | lane seconds | ms / query | threads | shards | record sha256 | arrays sha256 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    lane_hours = 0.0
    for stem in sorted(evals):
        r = evals[stem]
        if not r.get("supplement"):
            continue
        lane_hours += float(r["seconds"]) / 3600
        shards = len(r["merged_from"]) if r.get("merged_from") else 1
        lines.append(f"| `{stem}` | {r['dataset']} | {', '.join(f'`{m}`' for m in r['supplement']['models'])} | {r['queries']:,} | {r['halves']['V2_GATE']:,} / {r['halves']['V2_HELD_CONFIRMATION']:,} | "
                     f"{r['seconds']:.0f} | {r['ms_per_query']:.1f} | {r['threads']} | {shards} | `{R.sha256_file(R.EVAL / f'{stem}.json')}` | `{R.sha256_file(R.EVAL / f'{stem}.npz')}` |")
    lines += ["", f"Lane time of the pass: {lane_hours:.2f} lane-hours. Each supplement record is pinned to its seed-0 record (query list, half labels, fixed rrf per query); "
              "both halves are labelled as written and the pass reads nothing.", ""]
    lines += [f"## 4. Replication on {rec['half']} (seeds 0-2, the declared aggregation)", ""]
    for arm, c in rec["confirmation"].items():
        lines += [f"**`{arm}` ({R.FAMILY_GATES[c['family']]}) — {c['replication_status']}** (seeds present {c['seeds_present']}; the mean is per query over the seeds; "
                  f"a family passes only if every cell holds)", "", *CELL_HEADER]
        lines += [replication_cell_rows(e) for e in c["cells"]]
        failed = [f"{e['dataset']} {e['metric']} ({e['slice']})" for e in c["cells"] if not e["holds"]]
        lines += ["", (f"Cells failing on the seed mean: {', '.join(failed)}." if failed else "Every cell holds on the seed mean."), ""]
    lines += [f"Seed mean ± sd on {rec['half']} (three seeds) for every metric reported:", "", "| dataset | arm | seeds | recall@5 | hit@1 | full_coverage@5 | mrr |", "|---|---|---|---|---|---|---|"]
    for name in R.PILOT:
        for arm, s in rec["seeds"].get(name, {}).items():
            lines.append(f"| {name} | `{arm}` | {s['seeds']} | " + " | ".join(f"{s[m]['mean']:.4f} ± {s[m]['sd']:.4f}" if s[m]['sd'] is not None else f"{s[m]['mean']:.4f}" for m in SEED_METRICS) + " |")
    lines.append("")
    lines += [f"## 5. {HELD} (read once, REPLICATION_PASS families only)", ""]
    if held is None:
        statuses = ", ".join(f"{R.FAMILY_GATES[f]} {rec['family_status'][R.FAMILY_GATES[f]]}" for f in selected)
        lines += [f"Not read: no family is REPLICATION_PASS ({statuses}); the held half stays unread under the replication (amendment 4 held_confirmation.not_read_for). "
                  "The seed-0 held cells of both selected arms stand in the pilot's `held_record.json` and are not re-read.", ""]
    else:
        lines += [f"Read {held['utc']} for {held['read_for']}; not read for {held['not_read_for']} (its arrays dropped unread: {len(held['arrays_dropped_unread'])} keys). "
                  f"{held['what_this_is']}.", ""]
        for arm, v in held["cells_confirmatory"].items():
            lines += [f"**`{arm}` ({R.FAMILY_GATES[v['family']]}) — confirmatory, not a gate** (gate-half status {v['gate_half_status']})", "", *CELL_HEADER]
            lines += [replication_cell_rows(e) for e in v["seed_mean"]["cells"]]
            lines += ["", "Per seed on the held half (seed 0 copied from the pilot's held record):", "", "| seed | " + " | ".join(f"{e['dataset']} {e['metric']} ({e['slice']})" for e in v["seed_mean"]["cells"]) + " |",
                      "|---|" + "---|" * len(v["seed_mean"]["cells"])]
            for s in ("0", "1", "2"):
                cells = v["per_seed"][s]["cells"]
                lines.append(f"| {s} | " + " | ".join(f"{e['value']:.4f}" for e in cells) + " |")
            lines.append("")
    lines += ["## 6. Reading", "",
              f"ORIGINAL PILOT STATUS: {rec['original_pilot_status']}. POST-PILOT REPLICATION STATUS: {rec['replication_status']}. {rec['interpretation']}.",
              "The original pilot is still reported as failed; this document adds a replication reading beside it and changes no threshold, feature, architecture or population. "
              "Forbidden framings are not used: the question of the paper is how much effectiveness remains attributable to learned message passing after matching exposure and information, "
              "and this replication reads only whether two frozen arms hold their frozen cells on three seeds.", ""]
    lines += ["## 7. Incidents", ""]
    lines += ([f"- {i}" for i in incidents] if incidents else ["None logged during the replication (fits, eval pass, readings)."]) + [""]
    lines += ["## 8. Commits since the pilot closed", ""] + ([f"- `{c}`" for c in commits] if commits else ["- none"]) + [""]
    files = [R.CONFIG, R.OUT / "replication_record.json", R.OUT / "replication_held_record.json", R.OUT / "held_record.json", R.OUT / "gate_record.json",
             R.OUT / "selection.json", R.OUT / "replication_incidents.json"]
    files += [R.FITS / f"{k}.json" for k in rec["fits"]] + [R.FITS / f"{k}.pt" for k in rec["fits"]]
    for stem in sorted(evals):
        files += [R.EVAL / f"{stem}.json", R.EVAL / f"{stem}.npz", R.EVAL / f"{stem}_query_ids.json"]
    files += [R.M3B_OUT / "eval" / f"{name}.npz" for name in R.PILOT]
    record = {}
    for p in files:
        if p.exists():
            try:
                record[p.relative_to(ROOT).as_posix()] = R.sha256_file(p)
            except ValueError:
                record[p.as_posix()] = R.sha256_file(p)
    lines += ["## 9. Run record", "", f"Rendered {R.utc()} by `scripts/universal_v2_report.py --stage replication_doc` from the files below (sha256 of every sidecar a number above cites; "
              "the sidecars are gitignored, the record is committed as replication_record_<date>).", "", "| file | sha256 |", "|---|---|"]
    lines += [f"| `{k}` | `{v}` |" for k, v in record.items()] + [""]
    DOC = R.REPLICATION_DOC
    DOC.parent.mkdir(parents=True, exist_ok=True)
    with open(DOC, "w", encoding="utf-8", newline=LF) as f:
        f.write(LF.join(lines) + LF)
    (R.OUT / "replication_report_record.json").write_text(json.dumps({
        "utc": R.utc(), "doc": DOC.relative_to(ROOT).as_posix() if DOC.is_relative_to(ROOT) else DOC.as_posix(), "doc_sha256_lf": R.lf_sha256(DOC),
        "replication_record_sha256": R.sha256_file(R.OUT / "replication_record.json"),
        "replication_held_record_sha256": R.sha256_file(R.OUT / "replication_held_record.json") if held is not None else None,
        "original_pilot_status": rec["original_pilot_status"], "replication_status": rec["replication_status"], "family_status": rec["family_status"],
        "files": record}, indent=1), encoding="utf-8")
    log(f"wrote {DOC} ({len(lines)} lines): ORIGINAL PILOT STATUS {rec['original_pilot_status']}; POST-PILOT REPLICATION STATUS {rec['replication_status']}")
    return DOC


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["held", "doc", "replication", "replication_doc"], required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    cfg, cfg_m3b, _ = R.load_configs()
    if args.stage == "held":
        stage_held(cfg)
    elif args.stage == "doc":
        stage_doc(cfg, cfg_m3b)
    elif args.stage == "replication":
        stage_replication(cfg)
    else:
        stage_replication_doc(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
