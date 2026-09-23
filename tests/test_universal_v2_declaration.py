"""Tests for the Universal-v2 declaration.

configs/universal_v2.yaml is committed before any v2 column is compiled, any
v2 weight is fitted and any v2 number exists. These tests hold the shape of
that commitment: the registered question of the phase and of the paper are
verbatim, the forbidden framings are listed, the authorising ruling is filed
verbatim, M3B is pinned read-only by content hash ("Do not alter or rerun
M3B" fails loudly on a byte change), the pilot is the three anchor datasets on
development populations only, the gate thresholds are numbers above their
frozen references, the gate / held split is recomputable from the M3B query
ids, and every block appended later must be dated. Amendment 2 pins the raw
contract as an exact ordered list: the code contract, the declaration and,
once filed, the frozen contract must be one and the same list.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
CONFIG = ROOT / "configs" / "universal_v2.yaml"
M3B = ROOT / "configs" / "m3b_controlled_comparison.yaml"
M3B_EVAL = ROOT / "outputs" / "m3b" / "eval"
M3B_CARVES = ROOT / "outputs" / "m3b" / "carves.json"

QUESTION_V2 = (
    "Can depth-preserving compiled reasoning and a query-conditioned "
    "semantic-relation operator close the MetaQA weakness of the current "
    "universal models without sacrificing their behaviour on text retrieval?"
)
QUESTION_PAPER = (
    "After matching candidate exposure and inference-time graph information to "
    "modern graph-retrieval/GNN systems, how much effectiveness remains "
    "attributable specifically to learned message passing?"
)
CONTRIBUTION_2 = (
    "A single retrieval-seeded architecture can operate across typed KGs and "
    "text graphs, with either learned propagation or deterministic graph "
    "compilation."
)
TRIO = ["metaqa", "2wiki", "squad"]
CANDIDATES = ["u_mlp_v2", "u_mlp_v2_mix", "u_gnn_v2", "u_gnn_v2_ef"]
ARMS = CANDIDATES + ["u_gnn_v2_core78", "gat_universal_v1_trio"]
OVERALL = ("BOTH_PASS", "GNN_ONLY_PASS", "TWIN_ONLY_PASS", "PILOT_FAILED")     # amendment 2 family_status_vocabulary (the gate record)
TERMINAL = ("BOTH_CONFIRMED", "GNN_ONLY_CONFIRMED", "TWIN_ONLY_CONFIRMED", "PILOT_FAILED")   # amendment 3 terminal_state_vocabulary (the run record)
FAMILY_FINAL = ("GATE_FAIL", "CONFIRMATION_FAIL", "CONFIRMED_PASS")
STATUSES = {"DECLARED_NOT_RUN", "PILOT_GATE_READ"} | {f"RUN_{t}" for t in TERMINAL}
DATED = re.compile(
    r"^(contract_frozen|timing|amendment_[0-9]+|pilot_gate_record|run_record|replication_record|hard_stop|authorization_stage_[0-9])_[0-9]{4}_[0-9]{2}_[0-9]{2}$"
)
# the stage-2 blocks carry the stage-2 suffix and sit outside the pilot convention on purpose
# (amendment_5_2026_09_23.blocks_this_stage_will_append.naming_note)
STAGE2_DATED = re.compile(r"^(compile_record|timing|run_record|hard_stop)_stage_2_[0-9]{4}_[0-9]{2}_[0-9]{2}$")
NUM = re.compile(r"[0-9]+[.][0-9]+")


def lf_sha256(path: Path) -> str:
    data = path.read_bytes().replace(bytes([13, 10]), bytes([10]))
    return hashlib.sha256(data).hexdigest()


def squash(text: str) -> str:
    return " ".join(str(text).split())


def numbers(text: str) -> list[float]:
    return [float(x) for x in NUM.findall(str(text))]


@pytest.fixture(scope="module")
def decl() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def m3b() -> dict:
    return yaml.safe_load(M3B.read_text(encoding="utf-8"))


def test_status_phase_and_date(decl):
    assert decl["experiment"] == "universal_v2"
    assert decl["phase"] == "UNIVERSAL_V2"
    assert decl["status"] in STATUSES
    assert str(decl["declared"]) == "2026-09-19"
    assert "configs/m3b_controlled_comparison.yaml" in " ".join(decl["declared_under"])


def test_the_registered_questions_and_the_contribution_are_verbatim(decl):
    assert squash(decl["registered_question"]) == QUESTION_V2
    assert squash(decl["registered_question_of_the_paper_unchanged"]) == QUESTION_PAPER
    assert squash(decl["contribution_2_as_registered"]) == CONTRIBUTION_2


def test_the_forbidden_framings_are_listed(decl):
    forbidden = decl["forbidden_framings"]
    assert "prove message passing is unnecessary" in forbidden
    assert "show that we do not need message passing" in forbidden
    assert "demonstrate that the MLP wins" in forbidden
    joined = " ".join(forbidden)
    assert "pre-registered" in joined and "band" in joined
    assert "architecture shopping" in joined


def test_the_authorising_ruling_is_filed_verbatim(decl):
    auth = decl["authorization_2026_09_19"]
    verbatim = auth["ruling_verbatim"]
    assert "So yes: authorize Universal-v2 now." in verbatim
    assert "But start with the declaration and the 3-dataset pilot, not another giant run." in verbatim
    assert "Do not alter or rerun M3B." in verbatim
    assert "Keep this much smaller than M3B initially. Use development populations only." in verbatim
    still_out = " ".join(auth["still_out_of_scope"])
    assert "distillation" in still_out and "test split" in still_out and "M3B" in still_out
    assert "no Modal spend" in auth["lifts"] and "no test split" in auth["lifts"]
    assert auth["on_completion_of_each_stop"].startswith("STOP_FOR_REVIEW")


def test_m3b_is_pinned_read_only_by_content_hash(decl, m3b):
    """The ruling "Do not alter or rerun M3B": every pinned file hashes to the
    value filed here (CRLF folded to LF so the pin survives autocrlf)."""
    inc = decl["m3b_incumbents"]
    assert str(inc["closed_at_commit"]) == "0bbb3c4"
    pins = inc["pinned_files_sha256_lf_normalised"]
    assert set(pins) >= {
        "configs/m3b_controlled_comparison.yaml",
        "docs/M3B_RESULTS.md",
        "src/mp_retrieval/m3b_features.py",
        "src/mp_retrieval/m3b_models.py",
        "src/mp_retrieval/m3b_train.py",
        "scripts/m3b_compile.py",
        "scripts/m3b_run.py",
    }
    for rel, expected in pins.items():
        path = ROOT / rel
        assert path.exists(), rel
        assert lf_sha256(path) == expected, f"{rel} changed after the v2 declaration pinned it"
    assert m3b["status"] == "RUN"
    assert "run_record_2026_09_19" in m3b
    assert "edit, rerun, re-render or re-hash" in inc["must_not"]


def test_the_pilot_is_the_anchor_trio_on_development_populations_only(decl):
    assert decl["training"]["pilot_datasets"] == TRIO
    pops = decl["m3b_incumbents"]["eval_populations_reused_here"]
    assert set(pops) == set(TRIO)
    for name, cell in pops.items():
        assert cell["split"] == "dev", name
        assert len(cell["ids_sha256"]) == 64
        assert cell["queries"] > 0
    assert "test" not in json.dumps(pops)
    populations_text = squash(decl["measurement"]["populations"])
    assert "sha256 must match" in populations_text
    carves = decl["m3b_incumbents"]["training_carves_reused_here"]
    assert set(carves) == set(TRIO)
    for cell in carves.values():
        assert cell["select"] < cell["fit"] < cell["N"]
        assert len(cell["select_sha256"]) == 64 and len(cell["fit_sha256"]) == 64


@pytest.mark.skipif(not M3B_CARVES.exists(), reason="M3B carve sidecar not present")
def test_the_training_carves_are_the_m3b_carves(decl):
    filed = json.loads(M3B_CARVES.read_text(encoding="utf-8"))["per_dataset"]
    for name, cell in decl["m3b_incumbents"]["training_carves_reused_here"].items():
        src = filed[name]
        for key in ("N", "select", "fit", "select_sha256", "fit_sha256"):
            assert src[key] == cell[key], (name, key)


def test_selection_is_behind_the_firewall_and_written_before_any_eval(decl):
    sel = decl["arms"]["selection_behind_the_firewall"]
    assert "between u_gnn_v2 and u_gnn_v2_ef" in sel["gnn"]
    assert "before any eval population is scored" in sel["gnn"]
    assert "u_mlp_v2 and u_mlp_v2_mix" in sel["twin"]
    assert "no eval number" in sel["rule"] and "no number of the other family" in sel["rule"]
    order = decl["order_of_operations"]
    assert "STOP_FOR_REVIEW" in order["1"] and "DECLARED_NOT_RUN" in order["1"]
    assert "selection.json written once" in order["6"] and "STOP_FOR_REVIEW" in order["6"]
    assert "u_gnn_v2_core78" in order["6"]
    assert "dated authorization" in order["8"]


def test_the_arms_are_the_declared_six_and_nothing_carries_a_dataset_identity(decl):
    arms = decl["arms"]
    assert set(ARMS) <= set(arms)
    for name in CANDIDATES:
        assert arms[name]["role"].startswith("candidate"), name
    assert arms["u_gnn_v2_core78"]["role"].startswith("ablation")
    assert arms["gat_universal_v1_trio"]["role"].startswith("control")
    assert "no edges beyond the input block" in arms["u_mlp_v2"]["what"]
    assert "No evidence-flow state" in squash(arms["u_gnn_v2"]["what"])
    assert "typed STRUCT edges only" in arms["u_gnn_v2_ef"]["what"]
    shared = arms["shared"]
    assert shared["hidden"] == 128
    assert "zero-initialised" in shared["readout"]
    assert "no arm receives a dataset id" in shared["no_dataset_identity"]
    budgets = numbers(shared["parameter_accounting"].replace(",", ""))
    assert 450000.0 not in budgets  # the budgets are integers with thousands separators
    assert "450,000" in shared["parameter_accounting"] and "350,000" in shared["parameter_accounting"]
    frozen = arms["frozen_references_not_refitted"]["what"]
    for ref in ("gat_universal_v1", "gat_no_mp_v1", "qls_u_sota_v1"):
        assert ref in frozen


def test_the_twin_has_no_learned_propagation_and_the_guard_is_named(decl):
    lvf = decl["learned_vs_fixed_propagation"]
    assert "tests/test_universal_v2_models.py" in lvf["guard"]
    assert "may not contain a per-step operator over the pool edges" in lvf["the_twin_may"]
    assert "trainable parameter" in squash(lvf["learned_propagation"])


def test_the_contract_is_the_m3b_core_plus_the_depth_basis(decl):
    c = decl["information_contract_v2"]
    assert c["name"] == "UNIVERSAL_V2_FEATURE_CONTRACT"
    assert "ce584194a751" in c["base"]
    assert len(c["depth_basis_columns"]) == 13
    count = c["count"]
    assert "= 64" in count and "142" in count   # as filed; amendment 1 adds the 9 mask columns, amendment 2 the 13 ordered columns and the pin
    amended = decl["amendment_1_2026_09_19"]["check_3_firewalls_made_explicit"]["availability_masks"]["count_amended"]
    assert "64 + 9 = 73" in amended and "151 raw" in amended
    assert decl["amendment_2_2026_09_19"]["raw_contract_pinned"]["screen_input_columns"] == 164   # the count that stands (test_the_raw_contract_is_pinned...)
    prohibitions = " ".join(c["prohibitions"])
    for word in ("gold label", "test split", "dataset identity", "never edited"):
        assert word in prohibitions, word
    assert "|Spearman| >= 0.98" in squash(c["screen"])
    assert "before any fit" in squash(c["screen"])


def test_the_training_rule_is_m3b_unchanged(decl):
    tr = decl["training"]
    rule = squash(tr["rule"])
    assert "configs/m3b_controlled_comparison.yaml#training" in rule
    assert "epoch 2000 batches" in rule and "max 6 epochs" in rule and "patience 2" in rule
    assert "mp_retrieval.m3b_train unchanged" in squash(tr["training_code"])
    assert tr["seeds"]["pilot_seed0"].endswith("seed 0")
    assert "seeds 1 and 2" in tr["seeds"]["confirmation"]


def test_the_band_rule_stays_diagnostic(decl):
    ruling = squash(decl["methodological_ruling_band_rule"]["ruling"])
    assert "diagnostic" in ruling
    assert "2026-09-19 at 13:36 IST" in ruling
    assert "never described as pre-registered" in ruling


def test_the_later_stages_open_only_on_dated_authorization(decl):
    later = decl["later_stages"]
    for key in ("stage_2_six_dataset_joint", "stage_3_per_dataset_copies", "stage_4_leave_one_dataset_out"):
        assert "dated authorization block" in later[key]["opens_when"], key
    placement = squash(later["compute_placement"])
    assert "scripts/spawn_modal_jobs.py" in placement and "never modal run --detach" in placement


def test_the_compute_record_precedes_the_expensive_step(decl):
    comp = decl["compute"]
    assert comp["gpu_seconds"] == 0
    assert "150 fit-hours" in comp["hard_ceiling"]
    assert len(comp["abort_criteria"]) >= 3
    assert "BEFORE the seed-0 matrix" in squash(comp["timing_run"])
    assert "No architectural quantity" in squash(comp["timing_run"])


def test_every_block_appended_after_the_declaration_is_dated(decl):
    keys = list(decl)
    after = keys[keys.index("process_rules_kept") + 1:]
    declared = " ".join(decl["amendment_5_2026_09_23"]["blocks_this_stage_will_append"]["blocks"])
    for key in after:
        if STAGE2_DATED.match(key):
            # dated, but under the stage-2 naming: it must be a block amendment 5 said this stage would append
            assert key[:-11] + "_<date>" in declared, f"stage-2 block not declared in amendment 5: {key}"
            continue
        assert DATED.match(key), f"undated block appended to the declaration: {key}"
    rules = " ".join(decl["process_rules_kept"])
    assert "no scientific result from test data" in rules
    assert "Do not alter or rerun M3B" in rules
    assert "systems convenience never edits science" in rules


# ── amendment 1 (2026-09-19): the review's three checks ──────────────────────

SPLIT_AUDIT = ROOT / "outputs" / "universal_v2" / "split_audit.json"


@pytest.fixture(scope="module")
def amendment(decl) -> dict:
    return decl["amendment_1_2026_09_19"]


def test_the_halves_are_named_and_the_held_claim_is_the_narrow_one(amendment):
    halves = amendment["check_1_the_halves"]
    assert halves["names"] == {"gate": "V2_GATE", "held": "V2_HELD_CONFIRMATION"}
    held = squash(halves["what_V2_HELD_CONFIRMATION_is"])
    for phrase in ("architecture selection", "feature screening", "threshold decisions", "checkpoint selection", "seed-0 advance decision"):
        assert phrase in held, phrase
    not_claimed = squash(halves["what_it_is_not"])
    assert "not previously unseen" in not_claimed and "not an independent test set" in not_claimed
    assert "V2_HELD_CONFIRMATION was not used" in " ".join(amendment["ruling_verbatim"])


def test_the_amended_split_hashes_the_problem_family_where_one_exists(amendment):
    rule = amendment["check_1_the_halves"]["split_rule_amended"]
    assert rule["key"]["metaqa"] == "query_id"
    assert "sorted(gold_node_ids)" in rule["key"]["2wiki"] and "sorted(gold_node_ids)" in rule["key"]["squad"]
    assert "is even" in rule["rule"] and "frozen M3B arrays" in rule["rule"]
    audit = amendment["check_1_the_halves"]["group_audit"]["under_the_as_filed_query_id_parity_split"]
    assert audit["metaqa"]["topic_entity_x_qtype_crossing"] < 0.01          # negligible: metaqa keeps the id split
    assert audit["squad"]["gold_set_paragraph_crossing"] > 0.9              # the paragraph family: hashed
    assert audit["2wiki"]["gold_set_crossing"] > 0.01                        # the gold-pair family: hashed
    crossing = rule["problem_family_crossing_under_the_amended_split"]
    assert crossing["2wiki"] == 0.0 and crossing["squad"] == 0.0 and crossing["metaqa"] < 0.01
    counts = rule["counts"]
    pops = {"metaqa": 39138, "2wiki": 12576, "squad": 11873}
    for name, n in pops.items():
        assert counts[name]["V2_GATE"] + counts[name]["V2_HELD_CONFIRMATION"] == n, name


@pytest.mark.skipif(not M3B_EVAL.exists(), reason="M3B eval sidecars not present")
def test_the_metaqa_split_is_recomputable_from_the_m3b_ids(decl, amendment):
    ids_file = M3B_EVAL / "metaqa_query_ids.json"
    record = M3B_EVAL / "metaqa.json"
    if not (ids_file.exists() and record.exists()):
        pytest.skip("metaqa eval record not present")
    pops = decl["m3b_incumbents"]["eval_populations_reused_here"]
    rec = json.loads(record.read_text(encoding="utf-8"))
    assert rec["ids_sha256"] == pops["metaqa"]["ids_sha256"]
    ids = json.loads(ids_file.read_text(encoding="utf-8"))
    assert len(ids) == pops["metaqa"]["queries"]
    gate = sum(1 for q in ids if int(hashlib.sha256(str(q).encode("utf-8")).hexdigest(), 16) % 2 == 0)
    counts = amendment["check_1_the_halves"]["split_rule_amended"]["counts"]["metaqa"]
    assert gate == counts["V2_GATE"] and len(ids) - gate == counts["V2_HELD_CONFIRMATION"]


@pytest.mark.skipif(not SPLIT_AUDIT.exists(), reason="split audit sidecar not present")
def test_the_filed_split_counts_and_references_are_the_audit_sidecar(amendment):
    audit = json.loads(SPLIT_AUDIT.read_text(encoding="utf-8"))["per_dataset"]
    counts = amendment["check_1_the_halves"]["split_rule_amended"]["counts"]
    refs = amendment["pilot_gate_amended"]["reference_numbers_on_V2_GATE_seed0"]
    for name in TRIO:
        split = audit[name]["amended_split"]
        assert split["V2_GATE"] == counts[name]["V2_GATE"] and split["V2_HELD_CONFIRMATION"] == counts[name]["V2_HELD_CONFIRMATION"], name
        gat = audit[name]["frozen_m3b_references"]["gat_universal_v1_s0"]
        filed = refs[name]["gat_universal_v1"]
        for metric, key in (("recall@5", "recall5"), ("hit@1", "hit1"), ("full_coverage@5", "full_coverage5")):
            if key in filed:
                assert abs(gat[metric]["V2_GATE"] - filed[key]) < 1e-9, (name, key)
    metaqa = audit["metaqa"]
    assert abs(metaqa["ceilings_frozen_pool"]["hit_ceiling_any_gold_in_pool"]["V2_GATE"] - refs["metaqa"]["hit_ceiling"]) < 1e-9
    assert abs(metaqa["frozen_m3b_references"]["by_hop_V2_GATE"]["3hop"]["hit_ceiling"] - refs["metaqa"]["hit_ceiling_3hop"]) < 1e-9
    assert abs(metaqa["frozen_m3b_references"]["by_hop_V2_GATE"]["3hop"]["gat_universal_v1_s0/hit@1"] - refs["metaqa"]["gat_universal_v1"]["hit1_3hop"]) < 1e-9
    assert abs(audit["squad"]["frozen_m3b_references"]["fixed_rrf"]["recall@5"]["V2_GATE"] - refs["squad"]["fixed_rrf"]["recall5"]) < 1e-9
    assert abs(audit["2wiki"]["frozen_m3b_references"]["qls_u_sota_v1_s0"]["recall@5"]["V2_GATE"] - refs["2wiki"]["qls_u_sota_v1"]["recall5"]) < 1e-9


def test_the_amended_gate_thresholds_follow_their_internal_rules(amendment):
    refs = amendment["pilot_gate_amended"]["reference_numbers_on_V2_GATE_seed0"]
    gnn = amendment["pilot_gate_amended"]["gnn_gate"]
    twin = amendment["pilot_gate_amended"]["twin_gate"]
    check2 = amendment["check_2_metaqa_threshold_anchored_internally"]
    gat = refs["metaqa"]["gat_universal_v1"]
    # metaqa: one third of the gap to the frozen pool's own hit ceiling, not the published band
    assert abs(gnn["metaqa"]["hit1_all_hops"] - round(gat["hit1"] + (refs["metaqa"]["hit_ceiling"] - gat["hit1"]) / 3, 4)) < 1e-9
    assert abs(gnn["metaqa"]["hit1_3hop"] - round(gat["hit1_3hop"] + (refs["metaqa"]["hit_ceiling_3hop"] - gat["hit1_3hop"]) / 3, 4)) < 1e-9
    assert check2["all_hops"]["threshold_hit1"] == gnn["metaqa"]["hit1_all_hops"]
    assert check2["3hop"]["threshold_hit1"] == gnn["metaqa"]["hit1_3hop"]
    assert "published" in squash(check2["rule"]) and "calibration table only" in squash(check2["rule"])
    assert "0.989" not in json.dumps(gnn)
    # 2wiki: the GAT minus 0.010; squad: the fixed rrf minus 0.005
    assert abs(gnn["2wiki"]["recall5"] - round(refs["2wiki"]["gat_universal_v1"]["recall5"] - 0.010, 4)) < 1e-9
    assert abs(gnn["2wiki"]["full_coverage5"] - round(refs["2wiki"]["gat_universal_v1"]["full_coverage5"] - 0.010, 4)) < 1e-9
    assert abs(gnn["squad"]["recall5"] - round(refs["squad"]["fixed_rrf"]["recall5"] - 0.005, 4)) < 1e-9
    assert gnn["squad"]["recall5"] > max(refs["squad"][k]["recall5"] for k in ("gat_universal_v1", "gat_no_mp_v1", "qls_u_sota_v1"))
    # twin: 40 percent of the residual to the GAT; the same do-nothing test
    qls = refs["metaqa"]["qls_u_sota_v1"]["hit1"]
    assert abs(twin["metaqa"]["hit1_all_hops"] - round(qls + 0.4 * (gat["hit1"] - qls), 4)) < 1e-9
    q2 = refs["2wiki"]["qls_u_sota_v1"]["recall5"]
    assert abs(twin["2wiki"]["recall5"] - round(q2 + 0.4 * (refs["2wiki"]["gat_universal_v1"]["recall5"] - q2), 4)) < 1e-9
    assert twin["squad"]["recall5"] == gnn["squad"]["recall5"]
    assert "all three cells hold" in gnn["pass"] and "all three cells hold" in twin["pass"]
    assert set(amendment["supersedes"]) >= {"pilot_gate.gnn_gate", "pilot_gate.twin_gate"}


def test_the_screen_and_mixture_firewalls_are_explicit(amendment):
    fw = amendment["check_3_firewalls_made_explicit"]
    seq = [squash(s) for s in fw["screen_sequence"]]
    keys = ("fit carves", "screen on the fit carves only", "frozen", "fit and select", "V2_GATE read once", "advance or stop", "V2_HELD_CONFIRMATION read once")
    order = [next(i for i, s in enumerate(seq) if key in s) for key in keys]
    assert order == sorted(order) and len(set(order)) == 7
    required = {"which of the 151 columns survive", "the redundancy threshold", "any normalisation constant", "block inclusion",
                "relation-path inclusion", "the architecture", "the checkpoint epoch"}
    assert set(fw["neither_half_may_determine"]) >= required
    refusals = squash(fw["refusals_in_code"])
    assert "fit only" in refusals and "not compiled until contract_frozen_<date>" in refusals and "refuses to read V2_HELD_CONFIRMATION" in refusals
    masks = fw["availability_masks"]["added"]
    assert set(masks) == {"ring_n_h{t}_{f}", "typed_walks_h{t}"}
    mix = fw["data_mixture_frozen"]
    assert "mp_retrieval.m3b_train" in mix["sampler"] and "uniformly at random" in mix["sampler"]
    assert mix["balance"].startswith("balanced, not proportional")
    assert "16 query draws" in mix["batch"]
    for word in ("per-dataset loss weights", "curriculum", "over-sampling of metaqa"):
        assert word in mix["not_allowed"], word
    assert "byte-for-byte" in mix["control_conditions"]


def test_step_2_is_authorised_without_any_fit(amendment):
    step = amendment["step_2_authorised"]
    for path in ("src/mp_retrieval/universal_v2_features.py", "src/mp_retrieval/universal_v2_models.py", "scripts/universal_v2_run.py"):
        assert path in step["what"]
    assert "no fit" in step["not"] and "no timing run" in step["not"] and "no compilation" in step["not"]
    assert len(step["tests_required_beyond_shapes"]) >= 14
    assert amendment["status_after"] == "DECLARED_NOT_RUN"


# ── amendment 2 (2026-09-19): the step-2 review; the raw contract pinned ─────


@pytest.fixture(scope="module")
def amendment2(decl) -> dict:
    return decl["amendment_2_2026_09_19"]


def sha_of_names(names) -> str:
    return hashlib.sha256(",".join(names).encode("utf-8")).hexdigest()


def test_the_step_3_ruling_is_filed_verbatim_and_its_corrections_are_named(amendment2):
    ruling = " ".join(amendment2["ruling_verbatim"])
    for phrase in ("GO for step 3, but make two corrections before the real caches are compiled/frozen",
                   "pin the exact ordered feature-name list plus total count",
                   "code contract == declaration contract == frozen contract",
                   "add a bounded ordered, direction-sensitive relation-path channel for typed STRUCT paths at t=2/3",
                   "no dataset-specific relation-ID table",
                   "the step-3 compile report must show how often truncation actually occurs by dataset/hop",
                   "GNN_GATE = PASS / FAIL; TWIN_GATE = PASS / FAIL; overall = BOTH_PASS / GNN_ONLY_PASS / TWIN_ONLY_PASS / PILOT_FAILED",
                   "AUTHORIZE STEP 3 ONLY", "No model fitting", "No V2_GATE or V2_HELD reads", "STOP_FOR_REVIEW before the one-epoch timing run",
                   "MetaQA must not later receive extra training weight"):
        assert phrase in ruling, phrase
    assert amendment2["status_after"] == "DECLARED_NOT_RUN" and "bfdc4b2" in amendment2["reviewed"]
    decided = squash(amendment2["correction_1_the_raw_contract_pinned"]["decided"])
    assert "73 is intended" in decided and "78 + 86 = 164" in decided and "197 columns" in decided


def test_the_raw_contract_is_pinned_and_equals_the_code_contract(decl, m3b, amendment2):
    """Correction 1: code contract == declaration contract. The pinned ordered list is the module's V2_COLUMNS,
    the counts and both shas agree, and the M3B 78 are the frozen M3B core in its order."""
    from mp_retrieval import universal_v2_features as V
    pin = amendment2["raw_contract_pinned"]
    core78 = list(m3b["qls_u_core_contract_2026_09_13"]["surviving"])
    assert pin["m3b_core"] == 78 == len(core78)
    assert pin["m3b_core_sha256"] == sha_of_names(core78) == m3b["qls_u_core_contract_2026_09_13"]["sha256_of_comma_joined_surviving_names"]
    assert list(pin["v2_column_names"]) == list(V.V2_COLUMNS)
    assert pin["v2_column_names_sha256"] == sha_of_names(V.V2_COLUMNS)
    assert pin["screen_input_sha256"] == sha_of_names(core78 + list(V.V2_COLUMNS)) == sha_of_names(V.screen_input_columns(core78))
    assert (pin["depth_basis_columns"], pin["ordered_relation_path_columns"], pin["v2_columns"]) == (V.N_DEPTH, V.N_ORDERED, V.N_V2) == (73, 13, 86)
    assert pin["screen_input_columns"] == 78 + 86 == 164 == len(V.screen_input_columns(core78))
    assert pin["cache_layout_columns"] == V.N_COLUMNS == 111 + 86 == 197
    assert len(set(pin["v2_column_names"])) == 86 and not set(pin["v2_column_names"]) & set(core78)
    ordered = [c for c in pin["v2_column_names"] if c.startswith("opath_")]
    assert len(ordered) == 13 and ordered == list(V.ORDERED_COLUMNS) and pin["v2_column_names"][-13:] == ordered


def test_the_frozen_contract_when_filed_is_the_pinned_raw_contract_minus_the_recorded_drops(decl, m3b, amendment2):
    """Correction 1: declaration contract == frozen contract, once contract_frozen_<date> exists; until then the
    declaration carries no frozen block (step 3 is where it appears)."""
    keys = [k for k in decl if k.startswith("contract_frozen_")]
    pin = amendment2["raw_contract_pinned"]
    core78 = list(m3b["qls_u_core_contract_2026_09_13"]["surviving"])
    raw = core78 + list(pin["v2_column_names"])
    rule_sha = hashlib.sha256(str(decl["information_contract_v2"]["screen"]).strip().encode("utf-8")).hexdigest()
    for key in keys:
        frozen = decl[key]
        assert frozen["raw_contract_sha256"] == pin["screen_input_sha256"] and frozen["screened_columns"] == 164
        assert frozen["raw_contract_pinned_in"] == "amendment_2_2026_09_19"
        assert frozen["surviving"][:78] == core78 and not set(frozen["dropped"]) & set(core78)
        assert [c for c in raw if c not in frozen["surviving"]] == list(frozen["dropped"]) and set(frozen["surviving"]) <= set(raw)
        assert frozen["sha256_of_comma_joined_surviving_names"] == sha_of_names(frozen["surviving"]) == frozen["hashes"]["surviving_columns_sha256"]
        assert frozen["hashes"]["raw_contract_sha256"] == pin["screen_input_sha256"]
        assert frozen["hashes"]["screen_rule_sha256"] == frozen["screen_rule_sha256"] == rule_sha
        assert len(frozen["hashes"]["six_caches_combined_sha256"]) == 64
        for name in TRIO:
            for kind in ("fit", "select"):
                rec = frozen["compile_record"][name][kind]
                assert len(rec["cache_combined_sha256"]) == 64 and rec["diagnostics"]["k_rel"] == 4 and "all" in rec["diagnostics"]["by_hop"]
                assert rec["peak_rss_bytes"] > 0 and rec["queries_per_second"] > 0 and rec["cache_bytes"] > 0
        assert set(frozen["compile_record"]["metaqa"]["fit"]["diagnostics"]["by_hop"]) >= {"1hop", "2hop", "3hop", "all"}


def test_the_ordered_relation_path_channel_is_declared_bounded_semantic_and_directional(amendment2):
    c2 = amendment2["correction_2_ordered_relation_path_channel"]
    assert set(c2["columns"]) == {"opath_h{t}_q{k}", "opath_h{t}_dir{k}", "opath_h{t}_adj{k}{k+1}"}
    best = squash(c2["best_walk"])
    assert "relpath_max_h{t}" in best and "back-pointer" in best and "(owner, neighbour, relation)" in best
    direction = squash(c2["direction_convention"])
    assert "dir 2" in direction and "head -> tail (+1)" in direction and "dir 1" in direction and "(-1)" in direction
    props = c2["properties"]
    assert "one walk per node and depth" in props["bounded"] and "13 scalars" in props["fixed_dimensional"]
    assert "no relation id" in props["semantic"] and "no per-dataset table" in props["semantic"]
    assert "equals relpath_max_h{t}" in props["identity"]
    budgets = c2["parameter_budgets_at_164_columns"]
    assert budgets["holds"] is True
    assert max(budgets[a] for a in ("u_mlp_v2", "u_mlp_v2_mix")) <= budgets["twin_budget"] == 350000
    assert max(budgets[a] for a in ("u_gnn_v2", "u_gnn_v2_ef", "u_gnn_v2_core78")) <= budgets["gnn_budget"] == 450000
    assert budgets["gat_universal_v1_trio"] == 353410
    assert len(c2["tests_added"]) >= 4


def test_the_family_status_vocabulary_and_the_one_family_rule(amendment2):
    fam = amendment2["family_status_vocabulary"]
    assert set(fam["per_family"]) == {"GNN_GATE", "TWIN_GATE"} and set(fam["overall"]) == set(OVERALL)
    assert fam["statuses"]["after_the_gate"] == "PILOT_GATE_READ"
    assert set(fam["statuses"]["after_the_run_record"].split(" | ")) == {f"RUN_{o}" for o in OVERALL}   # superseded by amendment 3 (RUN_<terminal>)
    rule = squash(fam["reporting_rule"])
    assert "has passed only under BOTH_PASS" in rule and "no sentence describes GNN_ONLY_PASS or TWIN_ONLY_PASS as a pass of the pair" in rule
    assert "continues on GNN_ONLY_PASS or TWIN_ONLY_PASS" in squash(fam["continuation"])


def test_the_training_mixture_is_pinned_balanced_and_metaqa_gets_no_extra_weight(amendment2):
    mix = amendment2["training_mixture_pinned"]
    assert "mp_retrieval.m3b_train.draw_indices" in mix["sampler"] and "per_query" in mix["sampler"]
    assert "uniformly among the three pilot datasets" in mix["sampler"] and "16 query draws" in mix["sampler"] and "without replacement" in mix["sampler"]
    assert mix["balance"].startswith("balanced, not proportional") and "5,960" in mix["balance"] and "5,928" in mix["balance"] and "5,856" in mix["balance"]
    assert "mp_retrieval.m3b_models.listwise_loss" in mix["loss"] and "unweighted mean over the drawn queries" in mix["loss"] and "no per-dataset weight" in mix["loss"]
    assert "receives no extra weight" in mix["metaqa"] and "because it is the weak dataset" in mix["metaqa"]
    for word in ("per-dataset loss weights", "per-dataset batch quotas", "curriculum", "over-sampling of any dataset", "between arms"):
        assert word in mix["not_allowed"], word
    assert "6 epochs of 2,000 batches" in mix["optimiser_and_budget"] and "gat_universal_v1_trio" in mix["optimiser_and_budget"]


def test_step_3_is_authorised_without_any_fit_or_gate_read(amendment2):
    step = amendment2["step_3_authorised"]
    what = " ".join(step["what"])
    for phrase in ("--stage compile", "K_REL truncation by dataset and hop", "ordered relation-path availability", "peak RSS", "pre-registered screen only",
                   "UNIVERSAL_V2_CORE_CONTRACT", "six compiled cache directories", "STOP_FOR_REVIEW before the one-epoch timing run"):
        assert phrase in what, phrase
    not_ = " ".join(step["not"])
    assert "no model fitting" in not_ and "no timing run" in not_ and "no V2_GATE or V2_HELD_CONFIRMATION read" in not_ and "no eval population compiled" in not_
    assert "committed before the first real cache is compiled" in step["order"]
    k = amendment2["k_rel_truncation_report_required"]
    assert "K_REL = 4" in k["what"] and "by dataset and (metaqa) by hop" in k["what"] and "none is pre-registered" in k["decision_rule"]
    assert "CompileDiagnostics" in k["where"]
    sup = " ".join(amendment2["supersedes"])
    assert "64 / 142 / 151" in sup and "RUN / RUN_PILOT_FAILED" in sup and "86 / 164" in sup


# ── amendment 3: the continuous execution authorization ──────────────────────


@pytest.fixture(scope="module")
def amendment3(decl) -> dict:
    return decl["amendment_3_2026_09_19"]


def test_the_continuous_execution_ruling_is_filed_verbatim(amendment3):
    r = amendment3["ruling_verbatim"]
    assert r.startswith("CONTINUOUS EXECUTION AUTHORIZATION") and r.rstrip().endswith("Do not stop for intermediate review.")
    for phrase in ("The scientific contract MUST NOT change as a consequence of observed" + chr(10) + "results.",
                   "M3B is frozen and MUST NOT be altered or rerun.", "No architecture shopping.", "No new arm may be added.", "No threshold may be relaxed.",
                   "No test split may be read.", "selection.json is immutable.", "GNN_GATE = PASS / FAIL", "TWIN_GATE = PASS / FAIL",
                   "mark family CONFIRMATION_FAIL", "mark family CONFIRMED_PASS.", "Do not stop for intermediate review.",
                   "11. a genuinely new scientific decision not already covered by the", "Routine success is NOT a stop condition.",
                   "HARD_STOP_WITH_REASON", "Respect the filed compute ceiling."):
        assert phrase in r, phrase
    assert amendment3["status_after"] == "DECLARED_NOT_RUN" and "0881d4b" in amendment3["reviewed"]
    assert "you're doing" in amendment3["gloss_verbatim"] and "inventing Universal-v3" in amendment3["gloss_verbatim"]
    assert "Autonomy applies to execution, not to scientific redesign" in squash(amendment3["what_this_is"])


def test_the_terminal_state_vocabulary_and_the_statuses(amendment3):
    v = amendment3["terminal_state_vocabulary"]
    assert set(v["per_family_final"]) == set(FAMILY_FINAL) and set(v["terminal"]) == set(TERMINAL) | {"HARD_STOP_WITH_REASON"}
    assert v["statuses"]["after_the_gate"] == "PILOT_GATE_READ"
    assert set(v["statuses"]["after_the_run_record"].split(" | ")) == {f"RUN_{t}" for t in TERMINAL} == STATUSES - {"DECLARED_NOT_RUN", "PILOT_GATE_READ"}
    computed = squash(v["computed_from"])
    assert "pilot_gate.on_pass" in computed and "seed_confirmation" in computed and "report_run_record.json" in computed and "RUN_<terminal>" in computed
    assert "confirmed only under BOTH_CONFIRMED" in squash(v["gate_level_labels_unchanged"])
    assert "the status line is not moved" in v["terminal"]["HARD_STOP_WITH_REASON"]
    # the run script carries the same vocabulary
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("universal_v2_run_decl_check", ROOT / "scripts" / "universal_v2_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.TERMINAL.values()) == set(TERMINAL) and module.FAMILY_FINAL == FAMILY_FINAL and module.HARD_STOP == "HARD_STOP_WITH_REASON"
    assert module.FIT_HOURS_CEILING == 150.0 and module.DATED.pattern == DATED.pattern


def test_the_reconciliations_name_the_declaration_as_governing(amendment3):
    rec = amendment3["reconciliations_where_the_ruling_and_the_declaration_differ"]
    assert "no eval cache is written" in squash(rec["evaluation_carves"]) and "check_3" in squash(rec["evaluation_carves"])
    assert "patience 4" in squash(rec["timing_fallback"]) and "no third option" in squash(rec["timing_fallback"])
    assert "H=128" in rec["h_128"] and "unchanged" in rec["h_128"]
    assert "are not opened by this run" in squash(rec["later_stages"]) and "compute.hard_ceiling covers the pilot only" in squash(rec["later_stages"])
    assert "terminal state" in squash(rec["eligible_family_fails"])
    guard = squash(amendment3["compute_ceiling_guard"])
    assert "150 fit-hours" in guard and "ceiling_breach.json" in guard and "hard_stop_<date>" in guard and "the timing run is not a fit" in guard
    assert "the status line is not moved" in squash(amendment3["hard_stop_procedure"])
    assert "resumes from its epoch checkpoint" in squash(amendment3["crash_and_reboot"])
    sup = " ".join(amendment3["supersedes"])
    assert "order_of_operations 6" in sup and "RUN_<terminal>" in sup and "STOP_FINAL" in sup


# ── amendment 4: the post-pilot replication of the frozen selected arms ──────


@pytest.fixture(scope="module")
def amendment4(decl) -> dict:
    return decl["amendment_4_2026_09_21"]


def test_the_replication_ruling_is_filed_verbatim_after_the_closed_pilot(decl, amendment4):
    r = amendment4["ruling_verbatim"]
    assert r.startswith("AMENDMENT " + chr(0x2014) + " POST-PILOT REPLICATION OF FROZEN UNIVERSAL-v2 ARMS")
    assert r.rstrip().endswith("Otherwise continue autonomously to terminal replication status.")
    for phrase in ("This amendment is prospective with respect to every fit authorised below.",
                   "Nothing in this amendment changes, reinterprets, overrides, or retroactively" + chr(10) + "passes the original seed-0 pilot.",
                   "Do not retrain seed 0.", "Do not train any other Universal-v2 arm.", "Do not rerun model selection.",
                   "No threshold is changed.", "No tolerance band is added.", "No epsilon is introduced.",
                   "only if its THREE-SEED MEAN satisfies EVERY original frozen gate cell for", "Not merely the cell that failed seed 0.",
                   "Never rename a replication pass to:", "do NOT read the held half for that family.", "Do not introduce a new held threshold.",
                   "NO ARCHITECTURE REPAIR INSIDE THIS AMENDMENT", "ORIGINAL PILOT STATUS: PILOT_FAILED", "POST-PILOT REPLICATION STATUS: <status>",
                   "Crashes/reboots are resumable operational incidents and do not require" + chr(10) + "review.",
                   "This closes Universal-v2 without rescue."):
        assert phrase in r, phrase
    for status in ("REPLICATION_PASS", "REPLICATION_FAIL", "BOTH_REPLICATION_PASS", "GNN_REPLICATION_ONLY", "TWIN_REPLICATION_ONLY", "BOTH_REPLICATION_FAIL", "PILOT_PASS"):
        assert status in r, status
    assert amendment4["status_after"] == "RUN_PILOT_FAILED" and "d2850ab" in amendment4["reviewed"]
    assert "not a retroactive rescue" in amendment4["gloss_verbatim"] and "cheapest and most defensible next experiment" in amendment4["gloss_verbatim"]
    assert "nothing is redesigned inside it" in squash(amendment4["what_this_is"])
    # filed after the pilot closed, in the order of the file: the declaration is chronological and append-only
    keys = list(decl)
    assert keys.index("run_record_2026_09_21") < keys.index("amendment_4_2026_09_21") and DATED.match(keys[-1])
    original = amendment4["original_pilot"]
    assert decl["run_record_2026_09_21"]["terminal_state"]["terminal"] == "PILOT_FAILED" == original["status"]
    assert decl["status"] == "RUN_PILOT_FAILED" == original["status_line"] and original["commit"] == "d2850ab"
    assert original["document"].startswith("docs/UNIVERSAL_V2_PILOT.md") and decl["run_record_2026_09_21"]["doc_sha256"] in original["document"]
    assert "refuses on a mismatch" in squash(original["untouched"])


def test_the_authorised_fits_are_the_selected_arms_seeds_1_2_and_nothing_else(decl, amendment4):
    rep = amendment4["post_pilot_replication"]
    assert rep["authorised_fits"] == {"u_gnn_v2_ef": [1, 2], "u_mlp_v2_mix": [1, 2]}
    gate = decl["pilot_gate_record_2026_09_21"]
    assert rep["selected_arms"] == gate["selection"] == {"gnn": "u_gnn_v2_ef", "twin": "u_mlp_v2_mix"}
    assert gate["outcome"] == {"u_gnn_v2_ef": "FAIL", "u_mlp_v2_mix": "FAIL"}   # the replication is of the failed pilot's selected arms
    assert set(rep["authorised_fits"]) == set(rep["selected_arms"].values()) and all(v == [1, 2] for v in rep["authorised_fits"].values())
    assert rep["applies_at_status"] == "RUN_PILOT_FAILED" and rep["original_pilot_status"] == "PILOT_FAILED" and rep["original_pilot_commit"] == "d2850ab"
    assert "seed 0 is never retrained" in squash(rep["selected_arms_are"])
    assert "refuses a rule that differs" in squash(rep["frozen_training_protocol"]) and "max_epochs 6" in rep["frozen_training_protocol"]
    assert "parameter count" in squash(rep["frozen_architecture"]) and "seed-0 fit record" in squash(rep["frozen_architecture"])
    assert rep["thresholds"].startswith("amendment_1_2026_09_19.pilot_gate_amended")
    assert "no epsilon" in rep["thresholds"] and "no averaging across datasets" in rep["thresholds"]
    # the motivation cites the filed gate record's seed-0 cells, which stand unchanged: one failed cell per family
    cells = {(c["dataset"], c["metric"], c["slice"]): c for c in gate["verdict"]["u_gnn_v2_ef"]["cells"]}
    failed = [c for c in cells.values() if not c["holds"]]
    assert len(failed) == 1 and (failed[0]["value"], failed[0]["threshold"]) == (0.8988, 0.8996) and failed[0]["dataset"] == "squad"
    cells = {(c["dataset"], c["metric"], c["slice"]): c for c in gate["verdict"]["u_mlp_v2_mix"]["cells"]}
    failed = [c for c in cells.values() if not c["holds"]]
    assert len(failed) == 1 and (failed[0]["value"], failed[0]["threshold"]) == (0.8545, 0.8578) and failed[0]["dataset"] == "2wiki"
    assert "0.8988" in amendment4["ruling_verbatim"] and "0.8545" in amendment4["ruling_verbatim"]
    rec = amendment4["reconciliations_where_the_ruling_and_the_declaration_differ"]
    assert "0.8988 against 0.8996" in rec["seed_0_of_the_gnn_and_the_twin"] and "0.8545 against 0.8578" in rec["seed_0_of_the_gnn_and_the_twin"]


def test_the_replication_vocabulary_the_pass_rule_and_the_held_rule(amendment4):
    rep = amendment4["post_pilot_replication"]
    v = rep["status_vocabulary"]
    assert v["per_family"] == ["REPLICATION_PASS", "REPLICATION_FAIL"] and v["never"] == "PILOT_PASS"
    assert set(v["overall"]) == {"BOTH_REPLICATION_PASS", "GNN_REPLICATION_ONLY", "TWIN_REPLICATION_ONLY", "BOTH_REPLICATION_FAIL"}
    assert "stays RUN_PILOT_FAILED" in squash(v["status_line"]) and "replication_record_<date>" in squash(v["status_line"])
    assert "never of the proposed universal pair" in squash(v["one_family_rule"])
    assert "EVERY original frozen gate cell" in squash(rep["pass_rule"]) and "no other cell may compensate" in squash(rep["pass_rule"])
    assert "per-query mean over seeds 0, 1, 2 on V2_GATE" in squash(rep["aggregation"]) and "1e-12" in rep["aggregation"]
    held = rep["held_confirmation"]
    assert held["read_for"].startswith("REPLICATION_PASS families only")
    assert "no threshold of their own" in squash(held["procedure"]) and "stage_held" in held["procedure"]
    assert "dropped as the arrays are read" in squash(held["not_read_for"])
    assert "not a globally unseen dataset" in squash(held["what_the_held_half_is"])
    assert set(rep["interpretation_rule"]) == set(v["overall"])
    assert "closes Universal-v2 without rescue" in rep["interpretation_rule"]["BOTH_REPLICATION_FAIL"]
    assert "original pilot is still reported as failed" in rep["interpretation_rule"]["BOTH_REPLICATION_PASS"]
    assert "no fixed-score gating mechanism" in squash(rep["no_architecture_repair"])
    assert set(rep["reporting"]) >= {"seed-wise gate numbers", "three-seed means", "seed standard deviations", "original thresholds", "margins to threshold",
                                     "bootstrap intervals", "family replication statuses", "held-confirmation results where authorised", "compute time",
                                     "incidents", "artifact hashes", "checkpoint hashes", "commit IDs"}
    stops = rep["hard_stops"]
    assert len([k for k in stops if k != "procedure"]) == 8 and all("amendment_3 condition" in str(val) for k, val in stops.items() if k != "procedure")
    assert "the status line is not moved" in stops["procedure"]
    assert rep["execution"][:2] == ["file amendment", "commit before fitting"] and rep["execution"][-1] == "STOP" and len(rep["execution"]) == 11
    assert "150 fit-hours" in squash(rep["compute"]) and "is not a fit and does not count" in squash(rep["compute"])
    assert "checkpoint resumes the fit" in squash(rep["crash_and_reboot"])
    assert rep["outputs"]["declaration_block"].startswith("replication_record_<date>") and "the status line is not moved" in rep["outputs"]["declaration_block"]
    rec = amendment4["reconciliations_where_the_ruling_and_the_declaration_differ"]
    assert "not an ensemble" in squash(rec["ensemble_wording"]) and "stays RUN_PILOT_FAILED" in squash(rec["status_line"])
    assert "a REPLICATION_FAIL family gets none" in squash(rec["held_half_of_the_pilot"])
    # the run script carries the vocabulary and the dated prefix of the replication record
    import importlib.util
    spec = importlib.util.spec_from_file_location("universal_v2_run_decl_check_4", ROOT / "scripts" / "universal_v2_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.REPLICATION_FAMILY == tuple(v["per_family"]) and set(module.REPLICATION_OVERALL.values()) == set(v["overall"])
    assert module.REPLICATION_OVERALL[(True, True)] == "BOTH_REPLICATION_PASS" and module.REPLICATION_OVERALL[(False, False)] == "BOTH_REPLICATION_FAIL"
    assert module.DATED.pattern == DATED.pattern and DATED.match("replication_record_2026_09_22") and not DATED.match("replication_2026_09_22")


# ── the stage-2 authorization (2026-09-22): the six-dataset checkpoint of the replicated GNN ──


@pytest.fixture(scope="module")
def stage2(decl) -> dict:
    return decl["authorization_stage_2_2026_09_22"]


def test_the_stage_2_authorization_is_dated_last_and_moves_no_status_line(decl, stage2):
    dated = [k for k in decl if DATED.match(k)]
    assert dated.index("replication_record_2026_09_22") < dated.index("authorization_stage_2_2026_09_22")
    # only the amendments that carry the stage follow it, in the order they were filed
    assert dated[dated.index("authorization_stage_2_2026_09_22") + 1:] == [
        "amendment_5_2026_09_23", "amendment_6_2026_09_23", "amendment_7_2026_09_23", "amendment_8_2026_09_23"]
    assert decl["status"] == "RUN_PILOT_FAILED"
    assert stage2["stage_2_status"] == "DECLARED_NOT_RUN"
    lines = stage2["status_lines_not_moved"]
    assert lines == {"file_status": "RUN_PILOT_FAILED", "original_pilot_status": "PILOT_FAILED",
                     "post_pilot_replication_status": "GNN_REPLICATION_ONLY"}
    assert squash(stage2["filed_before"]) == "any stage-2 cache, fit, evaluation or number exists"


def test_the_stage_2_ruling_is_filed_verbatim_with_its_six_ordered_items(stage2):
    ruling = stage2["ruling_verbatim"]
    assert "UNIVERSAL_GNN_V2_SELECTED" in ruling and "u_gnn_v2_ef" in ruling
    for item in ("1. **Freeze", "2. File a dated authorization", "3. Evaluate all six",
                 "4. Separately run", "5. Then declare", "6. Do not modify M3B or Universal-v2 records"):
        assert item in ruling, item
    assert "The Universal-v2 GNN works. The Universal-v2 MLP idea is promising but not yet universal." in ruling
    assert r"\Delta_{\text{MetaQA Hit@1}}" in ruling      # the LaTeX of the ruling survives the filing byte for byte
    assert "as the universal MLP." in ruling


def test_the_freeze_is_one_arm_pinned_to_the_trio_checkpoints_without_a_warm_start(stage2):
    frozen = stage2["the_freeze"]
    assert frozen["name"] == "UNIVERSAL_GNN_V2_SELECTED" and frozen["arm"] == "u_gnn_v2_ef"
    what = squash(frozen["what_is_frozen"])
    for word in ("hidden 128", "K_REL 4", "420,932", "8d1da88b14df", "129 columns", "evidence-flow"):
        assert word in what, word
    pins = frozen["trio_checkpoints_pinned_as_history"]
    assert pins["seed_0_state_sha256"].startswith("9f749ad3") and pins["seed_1_state_sha256"].startswith("9b3a5521")
    assert pins["seed_2_state_sha256"].startswith("4575c418") and all(len(v) == 64 for v in pins.values())
    warm = squash(frozen["no_warm_start"])
    assert "from scratch" in warm and "No trio checkpoint is loaded, fine-tuned or used as an initialization" in warm


def test_the_opening_condition_is_the_replication_pass_and_renames_nothing(stage2):
    rec = stage2["opening_condition_reconciliation"]
    assert "at least one arm has a confirmed pass" in squash(rec["declared_rule"])
    happened = squash(rec["what_actually_happened"])
    assert "both arms failed" in happened and "PILOT_FAILED is terminal and is not reopened" in happened
    assert "REPLICATION_PASS on all five cells" in happened and "TWIN_GATE REPLICATION_FAIL" in happened
    assert "No threshold, cell, population, feature or architecture was changed" in squash(rec["the_ruling"])
    never = squash(rec["what_this_does_not_do"])
    for word in ("does not rename PILOT_FAILED", "does not move any status line", "PILOT_PASS",
                 "selection_behind_the_firewall", "opens nothing for u_mlp_v2_mix"):
        assert word in never, word


def test_stage_2_authorises_five_items_and_bars_the_other_stages_and_the_mlp(stage2):
    work = stage2["authorised_work"]
    assert len(work) == 5
    assert "compile the three added datasets" in work[0] and "before any fit" in work[0]
    assert "one epoch" in work[1] and "three fits of u_gnn_v2_ef" in work[2] and "seeds 0, 1, 2" in work[2]
    assert "one evaluation pass" in work[3] and "docs/UNIVERSAL_GNN_SIX.md" in work[4]
    barred = " ".join(stage2["not_authorised"])
    for word in ("stage_3_per_dataset_copies", "stage_4_leave_one_dataset_out", "Universal-MLP work",
                 "any test split", "u_mlp_v2_mix", "gat_universal_v1_trio", "any threshold, gate or selection"):
        assert word in barred, word
    other = stage2["the_other_track_is_not_opened_here"]
    assert "one file per phase" in squash(other["where"]) and "stays REPLICATION_FAIL as filed" in squash(other["bar"])


def test_the_six_populations_and_carves_are_the_m3b_ones_and_no_test_split(stage2):
    pops = stage2["populations_and_splits"]
    assert "no test split anywhere in this stage" in squash(pops["rule"])
    expect = {"metaqa": ("dev", 39138), "2wiki": ("dev", 12576), "squad": ("dev", 11873),
              "hotpotqa": ("validation", 7405), "musique": ("dev", 2417), "webqsp": ("train_holdout", 1503)}
    for name, (split, queries) in expect.items():
        row = pops[name]
        assert row["split"] == split and row["queries"] == queries and len(row["ids_sha256"]) == 64, name
        assert "test" not in row["split"]
    assert pops["webqsp"]["zero_gold_excluded"] == 46
    halves = squash(pops["halves"])
    assert "reported on each separately" in halves and "reported whole" in halves
    carves = stage2["training_carves"]
    m3b = json.loads(M3B_CARVES.read_text(encoding="utf-8"))["per_dataset"]
    for name in expect:
        assert carves[name]["fit"] == m3b[name]["fit"] and carves[name]["fit_sha256"] == m3b[name]["fit_sha256"], name
        assert carves[name]["select"] == m3b[name]["select"] and carves[name]["N"] == m3b[name]["N"], name
    assert carves["webqsp"]["source_rule"] == "sorted(train)[1::2]"
    assert "0042c258bd7b" in carves["refusal"]


def test_the_contract_does_not_move_for_the_added_datasets(stage2):
    tr = stage2["contract_transfer_to_the_added_datasets"]
    rule = squash(tr["rule"])
    assert "same 129 columns in the same order" in rule and "screen is NOT re-run" in rule
    bad = squash(tr["a_column_that_misbehaves_on_an_added_dataset"])
    assert "REPORTED in the compile record" in bad and "kept" in bad and "hard stop" in bad
    diag = squash(tr["diagnostics_required_before_any_fit"])
    for word in ("K_REL 4", "truncat", "typed-walk", "queries_with_no_gold_in_pool"):
        assert word in diag, word


def test_the_sampler_is_uniform_over_six_with_no_dataset_identity(stage2):
    tr = stage2["training"]
    assert "max_epochs 6" in squash(tr["rule"]) and "patience 2" in squash(tr["rule"])
    sampler = squash(tr["sampler"])
    for word in ("uniformly at random among the SIX", "No per-dataset loss weight", "no curriculum",
                 "no dataset identity feature", "no per-dataset head", "no router", "one checkpoint"):
        assert word in sampler, word
    assert "macro select recall@5 over the SIX select carves" in squash(tr["early_stopping"])
    assert tr["seeds"] == [0, 1, 2]
    timing = stage2["timing_before_the_schedule"]
    assert "before the three fits" in squash(timing["rule"])
    over = squash(timing["if_the_projection_exceeds_the_ceiling"])
    assert "BEFORE any full fit" in over and "spawn_modal_jobs.py" in over


def test_stage_2_reads_paired_against_m3b_has_no_gate_and_keeps_the_band_verdicts(stage2):
    ev = stage2["evaluation_and_reading"]
    assert "gat_universal_v1" in squash(ev["references"]) and "not refitted" in squash(ev["references"])
    paired = squash(ev["paired_procedure"])
    assert "1000 resamples" in paired and "default_rng(0)" in paired and "gate half and the held half separately" in paired
    assert "three-seed mean and the standard deviation" in squash(ev["seeds_reported"])
    gate = stage2["this_stage_has_no_gate"]
    assert "no threshold, no pass or fail" in squash(gate["rule"]) and "never repaired inside this stage" in squash(gate["rule"])
    cal = stage2["calibration_against_published_systems"]
    assert cal["verdicts"] == {"metaqa": "NOT_READ", "webqsp": "NOT_READ", "musique": "READ",
                               "hotpotqa": "READ", "2wiki": "READ", "squad": "CONTROL"}
    care = squash(cal["care"])
    assert "PR@K" in care and "set coverage, not recall" in care and "any_gold_at_pool" in care and "never an average" in care
    assert "0.5255" in care and "train_holdout" in care


def test_stage_2_compute_hard_stops_and_execution_order(stage2):
    comp = stage2["compute"]
    assert comp["ceiling_fit_hours"] == 150 and comp["spent_fit_hours"] == 46.36
    assert round(comp["spent_fit_hours"] + comp["remaining_fit_hours"], 2) == 150.0
    proj = comp["projection_filed_before_the_run"]
    assert "24 to 33" in str(proj["three_seeds_hours"]) and "inside the remaining ceiling" in str(proj["three_seeds_hours"])
    stops = stage2["hard_stops"]
    assert len(stops["conditions"]) == 7 and "hard_stop_<date>" in squash(stops["rule"])
    assert any("column would have to be added, dropped, reordered or replaced" in c for c in stops["conditions"])
    assert any("pass 150" in c for c in stops["conditions"])
    ex = stage2["execution"]
    assert ex["order"][0].startswith("file this block and commit it before any stage-2 work")
    stop = squash(ex["stop_after_this_block"])
    assert "go-ahead" in stop and "amendment_3 continuous execution applies within" in stop
    assert stage2["outputs"]["record"].startswith("run_record_stage_2_<date>") and "moving no status line" in stage2["outputs"]["record"]
    assert len(stage2["tests"]) >= 9

# ── amendment 5 (2026-09-23): the stage-2 execution go-ahead and its one systems bound ──


@pytest.fixture(scope="module")
def amd5(decl) -> dict:
    return decl["amendment_5_2026_09_23"]


def test_amendment_5_is_dated_last_and_moves_no_status_line(decl, amd5):
    dated = [k for k in decl if DATED.match(k)]
    assert dated[-4:] == ["amendment_5_2026_09_23", "amendment_6_2026_09_23",
                          "amendment_7_2026_09_23", "amendment_8_2026_09_23"]
    assert dated.index("authorization_stage_2_2026_09_22") < dated.index("amendment_5_2026_09_23")
    assert decl["status"] == "RUN_PILOT_FAILED"
    lines = amd5["status_lines_not_moved"]
    assert lines["file_status"] == "RUN_PILOT_FAILED" and lines["original_pilot_status"] == "PILOT_FAILED"
    assert lines["post_pilot_replication_status"] == "GNN_REPLICATION_ONLY"
    assert "DECLARED_NOT_RUN until run_record_stage_2_" in lines["stage_2_status"]
    assert squash(amd5["filed_before"]) == "any stage-2 cache, fit, evaluation or number exists"
    assert "00aabc9" in amd5["authorises"]


def test_the_go_ahead_is_filed_verbatim_with_its_nine_steps_and_seven_hard_stops(amd5):
    ruling = amd5["ruling_verbatim"]
    assert ruling.startswith("GO.") and "STAGE-2 EXECUTION AUTHORIZATION" in ruling
    for step in ("1. Compile the added canonical carves", "2. Run the declared measured one-epoch timing pass",
                 "3. Train from scratch", "4. Preserve exactly", "5. Evaluate the three resulting universal checkpoints",
                 "6. Compare against the frozen M3B incumbents", "7. Produce", "8. Render", "9. Run tests and commit"):
        assert step in ruling, step
    for stop in ("frozen-contract/hash mismatch", "M3B pin drift", "test-data access", "column/order change required",
                 "architecture/scientific change required", "unrecoverable corruption", "compute ceiling violation"):
        assert stop in ruling, stop
    assert "no dataset identity" in ruling and "no router" in ruling and "WebQSP remains train_holdout" in ruling
    read = amd5["reading_of_the_ruling"]
    assert "420,932-parameter architecture" in squash(read["item_4_is_the_freeze"])
    assert "Nothing new is measured." in squash(read["item_7_report_columns"])
    assert "no threshold" in squash(read["the_four_properties_the_user_will_read"])
    assert "no stop after the compile" in squash(read["continuous_execution"])
    assert "train_holdout" in squash(read["what_does_not_change"]) and "no test split" in squash(read["what_does_not_change"])


def test_the_only_change_is_the_cache_bound_and_it_is_measured_and_declared_first(amd5):
    sysc = amd5["systems_only_change"]
    what = squash(sysc["what"])
    assert "12 GB" in what and "raised to" in what and "20 GB" in what and "free-disk floor of 8 GB is unchanged" in what
    why = sysc["why_measured_not_guessed"]
    assert "16,193,190 rows" in why["trio_cache"] and "422.3 bytes per row" in why["trio_cache"]
    assert "16,724,483 rows" in why["added_carve_rows_from_the_m3b_compile_records"]
    assert str(why["projected_six_dataset_cache_gb"]).startswith("13.9") and float(why["free_disk_at_filing_gb"]) > 8
    assert "compute.abort_criteria is not edited" in squash(sysc["enforced_by"])
    assert "systems convenience never edits science" in squash(sysc["it_changes_no_science"])
    assert "12 GB" in " ".join(str(x) for x in decl_compute_abort())


def decl_compute_abort():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return cfg["compute"]["abort_criteria"]


def test_the_go_ahead_keeps_the_band_verdicts_the_no_gate_rule_and_names_its_blocks(amd5):
    read = amd5["reading_of_the_ruling"]
    care = squash(read["pr_at_k_care"])
    assert "set coverage, not recall" in care
    for word in ("metaqa NOT_READ", "webqsp NOT_READ", "musique READ", "hotpotqa READ", "2wiki READ", "squad CONTROL"):
        assert word in care, word
    stops = squash(read["hard_stops"])
    assert "hard_stop_<date>" in stops and "leave every status line where it is" in stops
    blocks = amd5["blocks_this_stage_will_append"]["blocks"]
    assert [b.split(" --")[0] for b in blocks] == ["compile_record_stage_2_<date>", "timing_stage_2_<date>", "run_record_stage_2_<date>"]
    assert "no status line moves" in squash(amd5["blocks_this_stage_will_append"]["naming_note"])
    assert "the compile of the three added" in squash(amd5["execution"])


# ── amendment 6: the trio checkpoint reprinted as context ────────────────────


@pytest.fixture(scope="module")
def amd6(decl) -> dict:
    return decl["amendment_6_2026_09_23"]


def test_amendment_6_is_dated_last_and_moves_no_status_line(decl, amd6):
    dated = [k for k in decl if DATED.match(k)]
    assert dated.index("amendment_6_2026_09_23") == dated.index("amendment_5_2026_09_23") + 1
    assert dated.index("amendment_7_2026_09_23") == dated.index("amendment_6_2026_09_23") + 1
    assert decl["status"] == "RUN_PILOT_FAILED"
    lines = amd6["status_lines_not_moved"]
    assert lines["file_status"] == "RUN_PILOT_FAILED" and lines["original_pilot_status"] == "PILOT_FAILED"
    assert lines["post_pilot_replication_status"] == "GNN_REPLICATION_ONLY"
    assert squash(amd6["filed_before"]) == "any stage-2 fit, evaluation or model number exists"


def test_amendment_6_adds_one_column_and_pins_the_files_it_reads(amd6):
    added = amd6["what_is_added"]
    assert added["context_arm"] == "u_gnn_v2_ef__H128__s0"
    assert sorted(added["datasets"]) == ["2wiki", "metaqa", "squad"]
    for name, src in added["source_files"].items():
        path = ROOT / src["path"]
        assert path.exists() and len(src["sha256"]) == 64
        assert hashlib.sha256(path.read_bytes()).hexdigest() == src["sha256"], f"{name}: the pilot arrays moved under the pin"
    text = squash(str(added["how_it_is_labelled"]))
    assert "different training set" in text and "not a seed comparison" in text


def test_amendment_6_authorises_no_fit_no_gate_and_no_substitute_column(amd6):
    bars = " ".join(squash(str(b)) for b in amd6["what_this_does_not_authorise"])
    for phrase in ("no fit, no re-fit and no warm start", "no threshold, no gate, no pass or fail",
                   "not a replication claim", "no context column on webqsp, hotpotqa or musique",
                   "no change to evaluation_and_reading.references", "no change to the architecture"):
        assert phrase in bars, phrase
    assert "GNN_REPLICATION_ONLY stands as filed" in bars


@pytest.fixture(scope="module")
def amd7(decl) -> dict:
    return decl["amendment_7_2026_09_23"]


def test_amendment_7_is_dated_last_and_moves_no_status_line(decl, amd7):
    dated = [k for k in decl if DATED.match(k)]
    assert dated[-2] == "amendment_7_2026_09_23" and dated[-1] == "amendment_8_2026_09_23"
    assert dated.index("amendment_6_2026_09_23") < dated.index("amendment_7_2026_09_23")
    assert decl["status"] == "RUN_PILOT_FAILED"
    lines = amd7["status_lines_not_moved"]
    assert lines["file_status"] == "RUN_PILOT_FAILED" and lines["original_pilot_status"] == "PILOT_FAILED"
    assert lines["post_pilot_replication_status"] == "GNN_REPLICATION_ONLY"
    assert squash(amd7["filed_before"]).startswith("any stage-2 fit exists")


def test_amendment_7_answers_a_measured_epoch_that_is_over_three_hours(decl, amd7):
    timing = decl["timing_stage_2_2026_09_23"]
    assert timing["epoch_over_three_hours"] is True
    measured = amd7["what_was_measured"]
    assert measured["filed_as"] == "timing_stage_2_2026_09_23"
    assert measured["epoch_seconds"] == timing["epoch_seconds"] == 20282.9
    assert measured["threads_used"] == timing["threads"] == 6
    # the rule asks for 8 threads: the deviation is recorded, and the trigger is shown to fire regardless
    off = squash(str(amd7["the_measurement_was_taken_off_the_declared_placement"]))
    assert "the rule asks for the epoch wall clock at 8 threads" in off
    assert "thread scaling is at best linear" in off and "4.226 hours" in off
    assert timing["epoch_seconds"] * 6 / 8 / 3600 > 3.0


def test_amendment_7_files_a_placement_and_no_reduced_schedule(decl, amd7):
    frozen = squash(decl["authorization_stage_2_2026_09_22"]["training"]["rule"])
    kept = squash(amd7["what_does_not_change"]["training_rule"])
    for number in ("max_epochs 6", "2000 batches", "patience 2", "clip 1.0", "per_query", "28800"):
        assert number in frozen and number in kept, number
    assert "No reduced schedule is filed" in kept
    unchanged = amd7["what_does_not_change"]
    assert squash(unchanged["seeds"]) == "three, 0 and 1 and 2, none dropped"
    assert "420932 parameters" in squash(unchanged["architecture"]) and "129-column" in squash(unchanged["architecture"])
    assert "train_holdout" in squash(unchanged["populations"]) and "no test split" in squash(unchanged["populations"])
    assert "150 fit-hours" in squash(unchanged["the_ceiling"])


def test_amendment_7_fit_placement_is_the_one_every_pilot_fit_used(amd7):
    assert "8 threads" in squash(amd7["the_placement_that_is_filed"]["fits"])
    fits = sorted((ROOT / "outputs" / "universal_v2" / "fits").glob("*.json"))
    assert fits, "the pilot fit records are the evidence for the filed placement"
    threads = {json.loads(p.read_text(encoding="utf-8"))["threads"] for p in fits}
    assert threads == {8}, f"the pilot did not run at one placement: {threads}"


def test_amendment_7_authorises_no_schedule_no_gate_and_no_second_timing(amd7):
    bars = " ".join(squash(str(b)) for b in amd7["what_this_does_not_authorise"])
    for phrase in ("no reduced schedule", "not a dropped seed", "no architecture, contract, column, K_REL or parameter change",
                   "no change to the populations", "no gate", "no Modal placement and no GPU", "no second timing block"):
        assert phrase in bars, phrase
    ceiling = amd7["the_ceiling_arithmetic"]
    assert "147.76" in squash(ceiling["worst_case"]) and "150" in squash(ceiling["worst_case"])
    assert "fit_hours_guard_six refuses the fit before it starts" in squash(ceiling["what_happens_if_it_binds"])
    assert "does not enter fit_hours_spent" in squash(ceiling["not_counted"])


@pytest.fixture(scope="module")
def amd8(decl) -> dict:
    return decl["amendment_8_2026_09_23"]


def test_amendment_8_is_dated_last_and_moves_no_status_line(decl, amd8):
    dated = [k for k in decl if DATED.match(k)]
    assert dated[-1] == "amendment_8_2026_09_23"
    assert dated.index("amendment_7_2026_09_23") < dated.index("amendment_8_2026_09_23")
    assert decl["status"] == "RUN_PILOT_FAILED"
    lines = amd8["status_lines_not_moved"]
    assert lines["file_status"] == "RUN_PILOT_FAILED" and lines["original_pilot_status"] == "PILOT_FAILED"
    assert squash(amd8["filed_before"]) == "any stage-2 fit record, checkpoint hash or evaluation number exists"


def test_amendment_8_corrects_the_cache_total_to_what_is_on_disk(amd8):
    """The corrected figure has to agree with the compile record AND with the caches themselves, and the trio
    and added shares must sum to it -- the original error was adding the trio to a total that contained it."""
    per = amd8["the_number_that_was_wrong"]["measured_per_dataset_gb"]
    total = per["total"]
    assert round(sum(v for k, v in per.items() if k != "total"), 2) == total == 13.88
    record = json.loads((ROOT / "outputs" / "universal_v2" / "six" / "compile_record.json").read_text(encoding="utf-8"))
    assert round(record["cache_bytes_total"] / 1e9, 2) == total
    trio = sum(v["bytes"] for v in record["trio_caches_untouched"].values())
    added = sum(v["bytes"] for v in record["cache_hashes"].values())
    assert round(trio / 1e9, 2) == 6.84 and round(added / 1e9, 2) == 7.04
    assert round((trio + added) / 1e9, 2) == total, "the trio and added shares must sum to the total, not exceed it"
    said = squash(amd8["the_number_that_was_wrong"]["what_is_true"])
    assert "13.88 GB, not 20.7 GB" in said


def test_amendment_8_quotes_amendment_7_without_editing_it(decl, amd8):
    wrong = amd8["the_number_that_was_wrong"]
    where = wrong["where"].split(".")
    node = decl[where[0]]
    for part in where[1:]:
        node = node[part]
    # the quoted sentence must still be the one amendment 7 carries: quoted, never re-filed
    assert squash(wrong["what_it_said"]) in squash(str(node))
    bars = " ".join(squash(str(b)) for b in amd8["what_this_does_not_authorise"])
    assert "no edit to amendment_7_2026_09_23" in bars


def test_amendment_8_keeps_the_placement_and_files_no_schedule(amd8):
    place = amd8["the_placement_is_not_changed"]
    assert squash(place["decision"]) == "the three fits stay at 8 threads"
    assert "EpochTooLong" in squash(place["what_would_change_it"])
    bars = " ".join(squash(str(b)) for b in amd8["what_this_does_not_authorise"])
    for phrase in ("no reduced schedule, and no new placement", "no change to the training rule",
                   "no gate, no threshold", "no change to the runner while a fit is running"):
        assert phrase in bars, phrase


def test_amendment_8_records_that_the_filed_placement_did_not_help(decl, amd8):
    measured = amd8["what_has_since_been_measured"]
    at_8 = squash(measured["the_filed_placement_did_not_help"])
    assert "23008.5 s" in at_8 and "20282.9 s" in at_8
    assert decl["timing_stage_2_2026_09_23"]["epoch_seconds"] == 20282.9
    assert "3.72 of the 8 threads" in squash(measured["the_constraint_is_memory_not_cores"])
    assert "not falling" in squash(measured["disk_is_stable"])
    assert "SixDiskGuard is enforced in the compile stage only" in squash(measured["a_gap_that_is_recorded_not_fixed"])
