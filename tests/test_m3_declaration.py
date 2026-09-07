"""The M3 declaration must be filed, self-consistent, and authorise no fit.

M3's whole reason for existing as a separate phase is a firewall: the GNN is
selected among GNN candidates, without QLS's numbers in the room. The failure
modes worth guarding against are therefore different from M2B's:

* a QLS quantity leaking into the GNN selection rule, as a threshold, an anchor
  or a tie-break -- which would make M4 a comparison between QLS and the GNN
  that happened to lose to it least;
* an archaeology record that is prose rather than measurement, so that "the
  historical architecture" drifts from what the code actually builds;
* a screen justified by graph evidence but actually chosen some other way;
* a workload or compute number preserved from a draft rather than recomputed
  from the filed matrix;
* an estimate presented with the authority of a measurement;
* a declaration that quietly authorises a fit.

Parameter counts here are checked against LIVE models, not against the numbers
in the document, because the point of the archaeology was to verify by
instantiation. The graph-only screen table is checked against the audit
documents it claims to copy, and the passage-style pick is recomputed from that
table rather than read off the declaration.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

CONFIG_PATH = pathlib.Path("configs/m3_gnn_development.yaml")
PROTOCOL_PATH = pathlib.Path("docs/M3_GNN_DEVELOPMENT_PROTOCOL.md")
CONTRACT_PATH = pathlib.Path("docs/GRAPH_INFORMATION_FAIRNESS_CONTRACT.md")
M2B_CONFIG_PATH = pathlib.Path("configs/m2b_semantic_minimality.yaml")
BUDGET_PATH = pathlib.Path("configs/candidate_budget.yaml")
SUBSTRATE_AUDIT = pathlib.Path("docs/GRAPH_SUBSTRATE_AUDIT_RESULTS.md")
D0_RESULTS = pathlib.Path("docs/GRAPH_CONTEXT_D0_RESULTS.md")

SIX_DATASETS = {"squad_clean", "2wiki_clean", "hotpotqa_clean", "metaqa", "webqsp",
                "musique_clean"}


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def protocol() -> str:
    return PROTOCOL_PATH.read_text(encoding="utf-8")


def _walk(node, path=""):
    """Yield (path, key, value) for every scalar leaf and every mapping key."""

    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            yield here, str(key), value
            yield from _walk(value, here)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            here = f"{path}[{index}]"
            yield here, "", value
            yield from _walk(value, here)


# ---------------------------------------------------------------------------
# The declaration authorises nothing.
# ---------------------------------------------------------------------------


def test_the_declaration_is_filed_and_authorises_no_fit(declaration):
    assert declaration["phase"] == "M3_GNN_DEVELOPMENT"
    assert declaration["status"] == "M3_DECLARED_RECONNAISSANCE_COMPLETE_NO_FIT_AUTHORISED"
    authorises = declaration["this_file_authorises"]
    assert "no GNN fit" in authorises
    assert "no comparison" in authorises
    assert "no M4" in authorises
    assert declaration["stop_condition"]["and_then"] == "STOP_FOR_REVIEW"


def test_the_stop_list_names_every_thing_that_would_cost_money_or_break_ordering(declaration):
    not_authorised = "\n".join(declaration["stop_condition"]["not_authorised"]).lower()
    for forbidden in ("training any gnn", "comparing anything against qls", "opening m4",
                      "additional seeds", "canonical crag", "package f", "e2"):
        assert forbidden in not_authorised, forbidden
    # And the thing M2B closed stays closed.
    assert "semantic selection" in not_authorised


def test_the_unearned_gates_are_exactly_the_ones_that_would_spend_money(declaration):
    """Six gates are earned by paperwork; four are earned by doing the work.

    The four false ones are the whole remaining phase-step. A declaration that
    arrived with all ten true would be claiming a smoke it never ran.
    """

    gates = declaration["launch_authorization"]["gates"]
    earned = {name for name, state in gates.items() if state}
    unearned = {name for name, state in gates.items() if not state}
    assert unearned == {
        "runner_exists",
        "instrumentation_tests_pass",
        "smoke_measures_a_gnn_fit",
        "compute_ceiling_refiled_against_measurement",
    }
    assert "archaeology_recorded" in earned
    assert "selection_rule_frozen" in earned
    assert "truncation_declared" in earned
    earned_not_waived = declaration["launch_authorization"]["gates_are_earned_not_waived"]
    assert "never by deciding it does not apply" in earned_not_waived


# ---------------------------------------------------------------------------
# The firewall. This is the load-bearing test in the module.
# ---------------------------------------------------------------------------


NEGATORS = ("never", "no ", "not ", "must not", "cannot", "prohibit")


def test_no_qls_quantity_can_enter_the_gnn_selection_rule(declaration):
    """Mechanical, not editorial: scan the rule for any mention of QLS.

    Every occurrence that survives must sit in a clause that FORBIDS the thing
    it names. A future edit adding, say, ``qls_gap_tolerance_pp: 0.5`` trips
    this on the key alone, which is the point -- the guarantee should not
    depend on anyone noticing the addition in review.
    """

    offenders = []
    for path, key, value in _walk(declaration["selection_rule"]):
        text = f"{key} {value}" if not isinstance(value, (dict, list)) else key
        if "qls" not in text.lower():
            continue
        if not any(negator in text.lower() for negator in NEGATORS):
            offenders.append((path, text[:120]))
    assert offenders == [], offenders


def test_the_anchor_is_the_best_gnn_and_not_the_incumbent_or_the_comparator(declaration):
    rule = declaration["selection_rule"]
    assert rule["anchoring"] == "SYMMETRIC_BEST_ANCHORED"
    anchored_on = rule["anchored_on"].lower()
    assert "best gnn" in anchored_on
    # Anchoring on the historical GNN would make the phase capable only of
    # confirming the incumbent; anchoring on QLS would break the firewall.
    assert "never on qls" in anchored_on
    assert "never on the historical gnn" in anchored_on


def test_the_firewall_states_both_the_question_and_its_negation(declaration):
    firewall = declaration["firewall"]
    assert firewall["m3_answers"].startswith("Which GNN architecture is best among")
    assert firewall["m3_does_not_answer"] == "Which GNN architecture beats QLS?"
    ordering = " | ".join(firewall["ordering"])
    assert ordering.index("QLS FROZEN") < ordering.index("GNN FROZEN")
    assert ordering.index("GNN FROZEN") < ordering.index("M4")


def test_the_prohibitions_run_in_both_directions(declaration):
    prohibited = "\n".join(declaration["firewall"]["prohibited_in_both_directions"]).lower()
    for item in ("teacher", "distillation", "logits", "hidden representations",
                 "gold-derived"):
        assert item in prohibited, item
    # GNN -> QLS as well as QLS -> GNN.
    assert "as qls features" in prohibited or "as gnn features" in prohibited


# ---------------------------------------------------------------------------
# Thresholds were not chosen for this phase's data.
# ---------------------------------------------------------------------------


def test_the_thresholds_are_m2bs_and_therefore_predate_this_phase(declaration):
    """The strongest available statement about a threshold: it is not new.

    M2B filed 0.25 / 0.50 / 0.50 before its own numbers existed, and M3 reuses
    them unchanged. They cannot have been picked to suit GNN results that do
    not exist yet, and they were not picked to suit QLS results either.
    """

    m3 = declaration["selection_rule"]["effectiveness_admissible_iff_pp"]
    m2b = yaml.safe_load(M2B_CONFIG_PATH.read_text(encoding="utf-8"))
    m2b_rule = m2b["selection_rule"]["effectiveness_admissible_iff"]
    assert set(m3) == set(m2b_rule) == {"macro", "per_dataset", "per_cell"}
    for clause, tolerance in m3.items():
        # M2B states its clauses algebraically ("... >= macro_best - 0.25pp").
        filed = re.search(r"([\d.]+)pp", m2b_rule[clause])
        assert filed, clause
        assert float(filed.group(1)) == tolerance, clause
    assert (m3["macro"], m3["per_dataset"], m3["per_cell"]) == (0.25, 0.5, 0.5)
    frozen = declaration["selection_rule"]["thresholds_are_not_adjustable"]
    assert "moved after seeing the numbers it failed is not a threshold" in frozen


def test_the_tie_break_orders_systems_cost_and_not_publication_convenience(declaration):
    keys = declaration["selection_rule"]["if_multiple_survive"]["keys"]
    assert declaration["selection_rule"]["if_multiple_survive"]["ordering"] == (
        "LEXICOGRAPHIC_SYSTEMS_PARETO"
    )
    assert "p95" in keys[0]
    assert "parameters" in keys[1]
    assert "memory" in keys[2]
    assert len(keys) == 4 and "deterministic" in keys[3]


def test_a_pareto_conflict_stops_the_phase_rather_than_relaxing_it(declaration):
    outcome = declaration["selection_rule"]["if_none_survives"]
    assert "GNN_PARETO_CONFLICT" in outcome
    assert "Do not widen a tolerance" in outcome


def test_five_seeds_stay_prohibited_and_a_resolution_is_proposed_not_launched(declaration):
    policy = declaration["seed_policy"]
    assert policy["seed"] == 0 and policy["count"] == 1
    assert policy["five_seed_confirmation"] == "PROHIBITED"
    resolution = policy["three_seed_resolution"]
    assert "PROPOSED" in resolution
    assert "Not launched automatically" in resolution
    assert "frozen before the fits" in resolution


# ---------------------------------------------------------------------------
# Archaeology, checked against live models and frozen configs.
# ---------------------------------------------------------------------------


def test_the_parameter_counts_are_what_a_live_model_reports(declaration):
    """The archaeology claimed verification by instantiation. Verify it.

    Every count in the declaration is rebuilt here from the real factory. If
    the model changes shape, this fails rather than the document quietly
    becoming wrong about the object it describes.
    """

    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from mp_retrieval.operator_models import build_seed_aware_message_passing

    del torch
    measured = declaration["historical_gnn"]["measured_parameter_counts"]
    for depth_key, expected in (("depth_1", measured["depth_1"]),
                                ("depth_2", measured["depth_2"])):
        layers = int(depth_key.split("_")[1])
        for kind, claimed in expected.items():
            model = build_seed_aware_message_passing(kind, 1536, 64, layers=layers)
            live = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert live == claimed, (kind, layers, live, claimed)


def test_depth_is_nearly_free_which_is_why_no_width_variant_is_proposed(declaration):
    """The capacity argument in the declaration, recomputed from its own table.

    A depth candidate is cheap and a width candidate would mostly resize the
    1536-wide input projection -- which is not the mechanism under test. That
    is the stated reason for a 2-candidate set, so it should be arithmetic
    rather than assertion.
    """

    measured = declaration["historical_gnn"]["measured_parameter_counts"]
    for kind in measured["depth_1"]:
        added_by_a_layer = measured["depth_2"][kind] - measured["depth_1"][kind]
        assert added_by_a_layer / measured["depth_1"][kind] < 0.05, kind
    assert "no concrete capacity reason" in declaration["candidates"][
        "no_width_variant_is_proposed"
    ]
    sweep = declaration["candidates"]["no_cartesian_sweep"]
    assert "3 operators x 4 depths x 5 widths x 6 datasets" in sweep
    assert declaration["candidates"]["count"] == 2


def test_the_per_dataset_operator_map_matches_the_frozen_budget(declaration):
    """The first archaeology finding, checked against the file it came from."""

    budget = yaml.safe_load(BUDGET_PATH.read_text(encoding="utf-8"))
    frozen = {name: spec["selected_gnn"] for name, spec in budget["datasets"].items()}
    assert set(frozen) == SIX_DATASETS
    assert len(set(frozen.values())) > 1, "the finding is that there is no single operator"
    stated = declaration["historical_gnn"]["the_historical_operator_is_per_dataset"]
    for dataset, operator in frozen.items():
        assert f"{dataset} {operator}" in stated, (dataset, operator)


def test_the_single_anchor_is_an_output_of_the_screen_and_not_a_reason_for_it(declaration):
    """gat happens to cover both screen datasets. That must not read as a cause."""

    budget = yaml.safe_load(BUDGET_PATH.read_text(encoding="utf-8"))
    frozen = {name: spec["selected_gnn"] for name, spec in budget["datasets"].items()}
    screen = set(declaration["screen_selection"]["cells"])
    assert {frozen[dataset] for dataset in screen} == {"gat"}
    assert declaration["candidates"]["G0"]["operator"].startswith("gat")
    assert declaration["candidates"]["G1"]["operator"] == "gat"
    excuse = declaration["historical_gnn"]["why_that_does_not_block_the_screen"]
    assert "not a reason the derivation chose them" in excuse
    assert "hotpotqa_clean" in excuse  # the case where the anchor would not exist


def test_the_historical_graph_privilege_is_recorded_as_the_induced_candidate_graph(declaration):
    privilege = declaration["historical_gnn"]["historical_graph_privilege"]
    assert privilege["scored_set"] == "Cq"
    assert privilege["context_set"] == "Cq"
    assert "G[Cq]" in privilege["graph_passed_to_each_layer"]
    assert "candidate_index" in privilege["graph_passed_to_each_layer"]
    assert "induction-starved" in privilege["the_defect"]
    assert "Historical results stand" in privilege["not_rewritten"]


def test_the_retrieval_features_the_historical_gnn_never_saw_are_listed(declaration):
    """The second archaeology finding: G0 is not the historical model re-run."""

    missing = "\n".join(declaration["historical_gnn"]["inputs_it_did_not_receive"]).lower()
    for feature in ("dense retriever reciprocal rank", "splade", "agreement", "node_role"):
        assert feature in missing, feature
    # And SUPPORT/PATH are in that list for a different reason, stated as such.
    assert "which is correct and stays correct" in missing

    width = declaration["fairness"]["node_input_width"]
    assert width["historical"] == 1537
    assert width["m3_all_regimes"] == 1541
    assert width["informative_columns_under_R1_R2"] == 1540
    assert "not a byte-identical" in declaration["historical_gnn"]["why_that_matters"].lower()


def test_the_loss_and_optimiser_are_recorded_from_the_scripts_that_define_them(declaration):
    training = declaration["historical_gnn"]["training"]
    assert training["optimizer"] == "AdamW"
    assert training["learning_rate"] == 0.001
    assert training["weight_decay"] == 0.0001
    assert training["epochs"] == 3
    assert training["batch_size"] == 16
    assert "run_operator_screen.py:135" in training["loss"]
    budget = yaml.safe_load(BUDGET_PATH.read_text(encoding="utf-8"))["training"]
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay"):
        assert training[key] == budget[key], key
    assert declaration["historical_gnn"]["architecture"]["depth_layers"] == budget["layers"]


# ---------------------------------------------------------------------------
# Graph privilege and the truncation the fairness contract forces us to declare.
# ---------------------------------------------------------------------------


def test_the_regimes_are_the_frozen_ones_and_r3_full_stays_prohibited(declaration):
    privilege = declaration["graph_privilege"]
    assert privilege["R1"] == {"scored": "Cq", "context": "Cq",
                               "radius_supplied": "0 beyond Cq"}
    assert "TARGET_H1" in privilege["R2"]["context"]
    assert privilege["R3"]["scored"] == "C3 = Cq u A64"
    assert privilege["r3_full_reexpansion"] == "PROHIBITED"
    assert "defines no new context" in privilege["regimes_are_already_frozen"]


def test_depth_two_is_truncated_under_every_frozen_regime(declaration):
    """The third finding. Declared before the fit, per fairness contract 9.2."""

    clause = declaration["graph_privilege"]["depth_and_radius_are_not_independent"]
    assert clause["contract_clause"].endswith("section 9.2")
    for regime in ("R1", "R2", "R3"):
        assert "truncated" in clause["what_that_means_here"][regime].lower(), regime
    consequence = clause["consequence_for_this_screen"]
    assert "no un-truncated condition for depth 2" in consequence
    assert "cannot be read as a clean depth effect" in consequence
    # And the fix that is deliberately not taken is named, with its cost.
    assert "TARGET_H2" in consequence and "NOT proposed" in consequence
    # The candidate carries the same warning where a reader will meet it.
    assert "Truncated under every frozen regime" in declaration["candidates"]["G1"]["truncation"]


def test_the_clean_contrast_the_screen_does_supply_is_named(declaration):
    """R1 vs R2 at depth 1 is the un-truncated measurement. Say so up front."""

    clause = declaration["graph_privilege"]["depth_and_radius_are_not_independent"]
    clean = clause["the_clean_contrast_this_screen_does_supply"]
    assert "G0 under R1 against G0 under R2" in clean
    assert "un-truncated on the R2 side" in clean


def test_the_fairness_contract_it_binds_to_exists_and_says_what_is_cited(declaration):
    assert declaration["fairness"]["binding_document"] == str(CONTRACT_PATH).replace("\\", "/")
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    assert "TARGET_H1" in contract and "9.2" in contract
    assert "adds no clause" in declaration["fairness"]["not_restated_here"]


def test_the_gnn_is_not_handed_qls_summary_mechanism(declaration):
    equal = declaration["fairness"]["equal_privilege_not_equal_representation"]
    both = "\n".join(equal["both_sides_receive"]).lower()
    for shared in ("query", "scored set", "context node", "dense rr", "splade rr",
                   "agreement", "node_role"):
        assert shared in both, shared
    # SUPPORT and PATH are QLS's summary mechanism and are named in upper case
    # wherever they are meant as feature blocks -- "one code path" is not one.
    for item in equal["both_sides_receive"]:
        assert "SUPPORT" not in item and "PATH" not in item, item

    withheld = equal["the_gnn_does_not_receive_SUPPORT_or_PATH"]
    assert withheld["decision"] == "correct, and preserved"
    assert "not extra information" in withheld["why"]
    assert "different and narrower experiment" in withheld["why"]
    assert "NOT authorised in M3" in withheld["the_control_this_forgoes"]


def test_the_tuning_budget_asymmetry_is_declared_rather_than_argued_away(declaration):
    asymmetry = declaration["fairness"]["the_tuning_budget_asymmetry_this_screen_creates"]
    assert "section 5" in asymmetry["contract_clause"].lower()
    assert "that is a smaller budget" in asymmetry["the_asymmetry"]
    handled = asymmetry["how_it_is_handled"]
    assert "BEFORE any fit" in handled
    assert "never to widen it afterwards" in handled
    # Parameter count is a finding, not a handicap to spend in either direction.
    assert "is not a tuning budget" in asymmetry["what_is_not_a_defence"]


# ---------------------------------------------------------------------------
# The screen was derived, not chosen. Recompute it.
# ---------------------------------------------------------------------------


COLUMNS = ["style", "isolated_fraction", "boundary_cut", "median_retention",
           "isolated_gold_queries", "validation_queries"]


def _screen_table(declaration) -> dict[str, dict]:
    section = declaration["screen_selection"]
    assert section["graph_only_table"]["columns"] == COLUMNS
    return {
        name: dict(zip(COLUMNS, row))
        for name, row in section["graph_only_table"].items()
        if name != "columns"
    }


def test_the_graph_only_table_is_a_faithful_copy_of_the_audit(declaration):
    """Every number in the derivation is checked against the doc it came from.

    A screen "derived mechanically from the frozen graph audit" is only as good
    as the transcription. This reads the audit's own sealed-A rows and the D0
    prevalence line and compares them to what the declaration tabulated.
    """

    table = _screen_table(declaration)
    assert set(table) == SIX_DATASETS

    audit = SUBSTRATE_AUDIT.read_text(encoding="utf-8")
    for dataset, row in table.items():
        pattern = rf"\|\s*{re.escape(dataset)}\s*\|\s*sealed A\s*\|\s*(\d+)\s*\|[^|]+\|[^|]+\|\s*([\d.]+)\s*\|"
        match = re.search(pattern, audit)
        assert match, dataset
        assert int(match.group(1)) == row["validation_queries"], dataset
        assert float(match.group(2)) == row["isolated_fraction"], dataset

    for dataset, row in table.items():
        pattern = (
            rf"\|\s*{re.escape(dataset)}\s*\|\s*sealed A\s*\|[^|]+\|\s*([\d.]+)\s*\|"
            r"[^|]+\|[^|]+\|[^|]+\|[^|]+\|\s*([\d.]+)\s*\|"
        )
        match = re.search(pattern, audit)
        assert match, dataset
        assert float(match.group(1)) == row["median_retention"], dataset
        assert float(match.group(2)) == row["boundary_cut"], dataset

    d0 = D0_RESULTS.read_text(encoding="utf-8")
    for dataset, row in table.items():
        percent = re.escape(f"{row['isolated_gold_queries'] * 100:.1f}")
        assert re.search(rf"\({percent}%\)", d0), dataset


def test_the_passage_pick_is_recomputed_from_the_table_not_read_off_it(declaration):
    """2wiki_clean must be the argmax on the criteria the declaration names.

    If a future edit changes the table or the reasoning, the stated pick and
    the derived pick part company here rather than in review.
    """

    table = _screen_table(declaration)
    passage = {name: row for name, row in table.items() if row["style"] == "passage"}
    assert set(passage) == {"2wiki_clean", "hotpotqa_clean", "musique_clean", "squad_clean"}

    most_damaged = max(passage, key=lambda name: passage[name]["isolated_fraction"])
    most_headroom = max(passage, key=lambda name: passage[name]["isolated_gold_queries"])
    assert most_damaged == most_headroom == "2wiki_clean"
    # And it is the argmax of isolated-gold prevalence across all six, not only
    # the passage four -- which is what "the most to act on" claims.
    assert max(table, key=lambda name: table[name]["isolated_gold_queries"]) == "2wiki_clean"
    assert declaration["screen_selection"]["passage_style_selection"]["selected"] == "2wiki_clean"


def test_webqsp_is_excluded_by_arithmetic_rather_than_by_preference(declaration):
    """One query is worth 1.587pp on a 63-query holdout. The tolerance is 0.50pp.

    That is not a judgement call: a panel whose resolution is coarser than the
    clause it must decide cannot decide it. Recomputed here so the exclusion
    stands on the number rather than on the sentence about the number.
    """

    tolerance = declaration["selection_rule"]["effectiveness_admissible_iff_pp"]["per_cell"]
    one_query_pp = 100.0 / 63
    assert one_query_pp == pytest.approx(1.587, abs=0.001)
    assert one_query_pp > 3 * tolerance

    table = _screen_table(declaration)
    assert table["webqsp"]["validation_queries"] == 315
    # It is also the least damaged of the six, so it loses on both counts.
    assert table["webqsp"]["boundary_cut"] == min(
        row["boundary_cut"] for row in table.values()
    )
    excluded = declaration["screen_selection"]["kb_style_selection"]["why_not_webqsp"]
    assert excluded["excluded_on"] == "panel size, mechanically"
    assert "1.587pp" in excluded["detail"]


def test_metaqa_is_the_most_damaged_kb_graph_and_its_limitation_is_preregistered(declaration):
    table = _screen_table(declaration)
    kb = {name: row for name, row in table.items() if row["style"] == "KB"}
    assert set(kb) == {"metaqa", "webqsp"}
    assert max(table, key=lambda name: table[name]["isolated_fraction"]) == "metaqa"
    assert max(table, key=lambda name: table[name]["boundary_cut"]) == "metaqa"

    kb_choice = declaration["screen_selection"]["kb_style_selection"]
    assert kb_choice["selected"] == "metaqa"
    limitation = kb_choice["the_limitation_this_choice_carries"]
    assert table["metaqa"]["isolated_gold_queries"] == 0.0
    assert "0.0%" in limitation["what"]
    assert "weak evidence" in limitation["consequence"]
    # The misreading is named in advance so it cannot be adopted afterwards.
    assert "does not help on KB graphs" in limitation["what_would_be_wrong"]
    assert "metaqa cannot distinguish that" in limitation["what_would_be_wrong"]


def test_the_screen_is_not_justified_by_qls_results_anywhere(declaration):
    section = declaration["screen_selection"]
    assert "predate" in section["rule"]
    disclaimer = section["not_chosen_because_of_qls"]
    assert "consequence of the derivation, not an input to it" in disclaimer
    # Same mechanical scan as the selection rule: no QLS quantity in the screen.
    offenders = []
    for path, key, value in _walk(section):
        text = f"{key} {value}" if not isinstance(value, (dict, list)) else key
        if "qls" in text.lower() and not any(n in text.lower() for n in NEGATORS):
            offenders.append((path, text[:120]))
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# Workload and compute, recomputed rather than carried forward.
# ---------------------------------------------------------------------------


def test_the_workload_is_derived_from_the_filed_matrix(declaration):
    """Counts must follow from the cells, not from an earlier draft's number."""

    cells = declaration["screen_selection"]["cells"]
    derived_cells = sum(len(regimes) for regimes in cells.values())
    assert derived_cells == declaration["screen_selection"]["cell_count"] == 6

    workload = declaration["workload"]
    assert workload["candidates"] == declaration["candidates"]["count"] == 2
    assert workload["cells"] == derived_cells
    assert workload["seeds"] == declaration["seed_policy"]["count"] == 1
    assert workload["fits"] == workload["candidates"] * workload["cells"] * workload["seeds"]
    assert declaration["screen_selection"]["logical_matrix"] == workload["fits"]
    assert workload["containers"] == len(cells)
    assert workload["features_rebuilt"] == 0


def test_the_compute_estimate_recomputes_from_the_measured_basis(declaration):
    """Every derived cost is rebuilt from the measured seconds and the rates."""

    compute = declaration["compute"]
    seconds = compute["measured_s4_seconds_on_the_screen_cells"]
    cells = declaration["screen_selection"]["cells"]
    assert set(seconds) - {"total"} == set(cells)
    total = sum(
        value
        for dataset, regimes in seconds.items()
        if dataset != "total"
        for value in regimes.values()
    )
    assert total == pytest.approx(seconds["total"], abs=0.01)
    assert seconds["total"] == pytest.approx(462.07, abs=0.01)

    multipliers = compute["provisional_multipliers"]
    estimated = compute["estimated_training_seconds"]
    for candidate, key in (("G0", "G0_depth_1"), ("G1", "G1_depth_2")):
        assert estimated[candidate] == pytest.approx(total * multipliers[key], abs=0.5)
    assert estimated["total"] == pytest.approx(estimated["G0"] + estimated["G1"], abs=0.5)

    conservative = estimated["total"] * compute["scale_headroom"]
    assert compute["conservative_training_seconds"] == pytest.approx(conservative, abs=1.0)

    fit_usd = conservative / 3600 * compute["gpu_usd_per_hour"]
    assert compute["estimated_fit_usd"] == pytest.approx(fit_usd, abs=0.01)

    inference = compute["inference_benchmarking"]
    assert inference["fits"] == declaration["workload"]["fits"]
    inference_usd = (
        inference["fits"] * inference["seconds_per_fit"] / 3600 * compute["gpu_usd_per_hour"]
    )
    assert inference["usd"] == pytest.approx(inference_usd, abs=0.01)

    assert compute["estimated_total_usd"] == pytest.approx(
        compute["estimated_fit_usd"] + inference["usd"] + compute["container_overhead_usd"],
        abs=0.01,
    )
    assert compute["estimated_total_usd"] < compute["proposed_ceiling_usd"]


def test_the_multipliers_are_labelled_as_guesses_and_the_ceiling_is_not_filed(declaration):
    """M2B guessed 2.5x for S4 and the smoke disagreed. Do not repeat it silently."""

    compute = declaration["compute"]
    assert compute["status"].startswith("ESTIMATED_PROVISIONAL")
    assert "guesses" in compute["why_provisional"]
    assert "smoke must replace these" in compute["why_provisional"]
    assert "M2B" in compute["why_provisional"]
    assert "PROPOSED, not filed" in compute["ceiling_status"]
    assert "refile" in compute["ceiling_status"]
    assert compute["ledger_line"].startswith("m3_gnn_development")
    assert "separate from every M2B line" in compute["ledger_line"]


# ---------------------------------------------------------------------------
# Systems and persistence.
# ---------------------------------------------------------------------------


def test_the_timed_span_is_the_whole_cold_path(declaration):
    systems = declaration["systems_contract"]
    span = " ".join(systems["the_span_that_must_be_timed"]).lower()
    for stage in ("candidate and context resolution", "context edge gathering",
                  "node and query input construction", "message passing",
                  "candidate readout"):
        assert stage in span, stage
    forbidden = systems["what_must_not_be_reported_as_cold_latency"]
    assert "Pre-built graph tensors" in forbidden
    assert "measures the GPU kernel" in forbidden
    components = systems["components_persisted_separately"]
    assert set(components) == {
        "context_materialization_ms", "edge_gather_ms", "message_passing_ms",
        "readout_ms", "total_uncached_ms",
    }
    assert systems["cached_forward_latency"] == "secondary diagnostic only"


def test_every_fit_persists_enough_to_rebuild_its_own_aggregate(declaration):
    required = "\n".join(declaration["instrumentation_required_per_fit"]).lower()
    for item in ("checkpoint", "per-query rows", "edge fingerprint", "contract hash",
                 "source commit", "parameter count", "peak training vram",
                 "peak inference vram", "p50/p95/p99"):
        assert item in required, item
    assert "rebuilt from them and matching exactly" in required


def test_the_contract_hash_covers_science_and_not_declaration_prose(declaration):
    design = declaration["contract_hash_design"]
    assert design["reused_from"] == "scripts/feature_build_contract.py"
    assert pathlib.Path(design["reused_from"]).exists()
    assert "NOT the declaration prose" in design["the_rule"]
    assert "Hashing this file" in design["what_would_be_wrong"]


# ---------------------------------------------------------------------------
# The two documents describe the same phase.
# ---------------------------------------------------------------------------


def test_the_protocol_and_the_declaration_agree(declaration, protocol):
    assert declaration["status"] in protocol
    assert "configs/m3_gnn_development.yaml" in protocol
    assert "**This phase-step trains nothing.**" in protocol

    for dataset in declaration["screen_selection"]["cells"]:
        assert dataset in protocol
    for value in ("0.25pp", "0.50pp", "462.07", "213,568", "1537", "1541"):
        assert value in protocol, value
    assert str(declaration["workload"]["fits"]) in protocol
    assert "GNN_PARETO_CONFLICT" in protocol
    assert "SYMMETRIC_BEST_ANCHORED" in protocol
    assert "LEXICOGRAPHIC_SYSTEMS_PARETO" in protocol


def test_the_protocol_keeps_the_estimate_labelled_as_an_estimate(protocol):
    assert "**The multipliers are guesses and are labelled as such.**" in protocol
    assert "proposed, not filed" in protocol
    assert "$2.50" in protocol


def test_the_protocol_ends_by_stopping(protocol):
    tail = protocol[protocol.index("## 11. Stop"):]
    assert "Not authorised by this document" in tail
    for forbidden in ("training any GNN", "opening M4", "canonical CRAG", "Package F", "E2"):
        assert forbidden in tail, forbidden
