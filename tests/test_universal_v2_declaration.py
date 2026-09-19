"""Tests for the Universal-v2 declaration.

configs/universal_v2.yaml is committed before any v2 column is compiled, any
v2 weight is fitted and any v2 number exists. These tests hold the shape of
that commitment: the registered question of the phase and of the paper are
verbatim, the forbidden framings are listed, the authorising ruling is filed
verbatim, M3B is pinned read-only by content hash ("Do not alter or rerun
M3B" fails loudly on a byte change), the pilot is the three anchor datasets on
development populations only, the gate thresholds are numbers above their
frozen references, the gate / held split is recomputable from the M3B query
ids, and every block appended later must be dated.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
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
STATUSES = {"DECLARED_NOT_RUN", "PILOT_GATE_READ", "RUN", "RUN_PILOT_FAILED"}
DATED = re.compile(
    r"^(contract_frozen|timing|amendment_[0-9]+|pilot_gate_record|run_record|authorization_stage_[0-9])_[0-9]{4}_[0-9]{2}_[0-9]{2}$"
)
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


@pytest.mark.skipif(not M3B_EVAL.exists(), reason="M3B eval sidecars not present")
def test_the_eval_populations_and_the_gate_split_are_recomputable_from_the_m3b_ids(decl):
    pops = decl["m3b_incumbents"]["eval_populations_reused_here"]
    counts = decl["measurement"]["gate_and_held_halves"]["counts_from_the_m3b_ids"]
    for name in TRIO:
        record = M3B_EVAL / f"{name}.json"
        ids_file = M3B_EVAL / f"{name}_query_ids.json"
        if not (record.exists() and ids_file.exists()):
            pytest.skip(f"{name} eval record not present")
        rec = json.loads(record.read_text(encoding="utf-8"))
        assert rec["ids_sha256"] == pops[name]["ids_sha256"], name
        assert rec["queries"] == pops[name]["queries"], name
        ids = json.loads(ids_file.read_text(encoding="utf-8"))
        assert len(ids) == pops[name]["queries"], name
        gate = sum(1 for q in ids if int(hashlib.sha256(str(q).encode("utf-8")).hexdigest(), 16) % 2 == 0)
        assert gate == counts[name]["gate"], name
        assert len(ids) - gate == counts[name]["held"], name


def test_the_gate_split_rule_is_the_even_sha256_rule(decl):
    rule = squash(decl["measurement"]["gate_and_held_halves"]["rule"])
    assert "sha256(query_id utf-8)" in rule and "is even" in rule
    assert "gate half only" in rule and "held half is read once" in rule
    counts = decl["measurement"]["gate_and_held_halves"]["counts_from_the_m3b_ids"]
    pops = decl["m3b_incumbents"]["eval_populations_reused_here"]
    for name in TRIO:
        assert counts[name]["gate"] + counts[name]["held"] == pops[name]["queries"], name


def test_the_gate_thresholds_are_numbers_above_their_frozen_references(decl):
    gate = decl["pilot_gate"]
    ref = gate["reference_numbers_from_m3b_on_the_gate_half_seed0"]
    gnn = gate["gnn_gate"]
    all_hops, three_hop = numbers(gnn["metaqa"]["materially_close_the_kb_gap"])[:2]
    assert all_hops == 0.88 and three_hop == 0.82
    incumbent = ref["metaqa"]["gat_universal_v1"]
    band = ref["metaqa"]["published_kb_band_hit1"]["3hop"]
    assert incumbent["hit1"] < all_hops < band
    assert incumbent["hit1_3hop"] < three_hop < band
    # one third of the gap, filed as the derivation
    assert abs(all_hops - (incumbent["hit1"] + (band - incumbent["hit1"]) / 3)) < 0.006
    assert abs(three_hop - (incumbent["hit1_3hop"] + (band - incumbent["hit1_3hop"]) / 3)) < 0.006
    r5, fc5 = numbers(gnn["2wiki"]["no_regression_vs_the_universal_gat"])[:2]
    assert r5 == 0.873 and fc5 == 0.742
    assert abs(r5 - (ref["2wiki"]["gat_universal_v1"]["recall5"] - 0.010)) < 1e-9
    assert abs(fc5 - (ref["2wiki"]["gat_universal_v1"]["full_coverage5"] - 0.010)) < 1e-9
    squad = numbers(gnn["squad"]["no_meaningful_regression_from_fixed_retrieval"])[0]
    assert squad == 0.899
    assert ref["squad"]["fixed_rrf"]["recall5"] - 0.005 <= squad
    assert squad > max(ref["squad"][k]["recall5"] for k in ("gat_universal_v1", "gat_no_mp_v1", "qls_u_sota_v1"))
    twin = gate["twin_gate"]
    t_metaqa = numbers(twin["metaqa"])[0]
    t_2wiki = numbers(twin["2wiki"])[0]
    t_squad = numbers(twin["squad"])[0]
    assert t_metaqa == 0.70 and t_2wiki == 0.855 and t_squad == squad
    assert ref["metaqa"]["qls_u_sota_v1"]["hit1"] < t_metaqa < incumbent["hit1"]
    assert ref["2wiki"]["qls_u_sota_v1"]["recall5"] < t_2wiki < ref["2wiki"]["gat_universal_v1"]["recall5"]
    assert "all three cells hold" in gnn["pass"] and "all three cells hold" in twin["pass"]
    on_fail = squash(gate["on_fail"])
    assert "no threshold is moved" in on_fail and "no candidate is added" in on_fail


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


def test_the_contract_is_the_m3b_core_plus_a_64_column_depth_basis(decl):
    c = decl["information_contract_v2"]
    assert c["name"] == "UNIVERSAL_V2_FEATURE_CONTRACT"
    assert "ce584194a751" in c["base"]
    assert len(c["depth_basis_columns"]) == 13
    count = c["count"]
    assert "= 64" in count and "142" in count
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
    for key in after:
        assert DATED.match(key), f"undated block appended to the declaration: {key}"
    rules = " ".join(decl["process_rules_kept"])
    assert "no scientific result from test data" in rules
    assert "Do not alter or rerun M3B" in rules
    assert "systems convenience never edits science" in rules
