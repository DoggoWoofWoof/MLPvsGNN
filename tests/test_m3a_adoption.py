"""Tests for the M3A adoption scripts: ingest, verify, align, budget, report, cost.

These scripts read an 81 GiB package that lives outside the repository, so the
tests are split by what they can honestly assert without it:

  - pure logic (exclusion matching, batch simulation, pricing, status derivation)
    is tested directly on the functions, with no package and no artifacts;
  - the artifacts, when present, are checked for internal consistency and for
    the contract properties that must hold whatever the numbers turn out to be;
  - the package itself is never required. Every artifact-dependent test skips
    when the artifact is absent, so a fresh clone still runs green.

The point of the second group is that the derived numbers cannot drift away from
the measured ones: a report that restates a figure instead of reading it would
fail here the moment the measurement changed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "m3a"


def module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def artifact(name: str) -> dict:
    path = OUT_DIR / name
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not generated in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


# ── ingest: the exclusion bug that silently dropped a 28.9M-edge graph ────────


class TestExclusionMatching:
    """`data/canonical/2wiki` must not swallow `data/canonical/2wiki_universe`.

    A startswith() prefix test did exactly that on the first run and cut the
    inventory from 2,841 files to 1,522 without erroring. Segment matching is
    the fix, and these cases are the regression.
    """

    @pytest.fixture(scope="class")
    def ingest(self):
        return module("m3a_transfer_ingest")

    @pytest.mark.parametrize(
        "relative",
        [
            "data/canonical/2wiki/graph.tsv",
            "data/canonical/2wiki",
            "data/final_canonical/_work/scratch.json",
            "data/final_canonical/freebase_v3/anything.parquet",
            "data/final_canonical/_superseded_textualization_rev1/x.jsonl",
        ],
    )
    def test_the_excluded_trees_really_are_excluded(self, ingest, relative):
        assert ingest.is_excluded(relative) is True

    @pytest.mark.parametrize(
        "relative",
        [
            "data/canonical/2wiki_universe/graph.tsv",
            "data/canonical/2wiki_universe/dense/docs/shard_00000.npy",
            "data/final_canonical/2wiki/nodes.jsonl",
            "data/final_canonical/webqsp/v1/edges.parquet",
            "data/canonical/metaqa/graph.tsv",
        ],
    )
    def test_the_required_trees_survive(self, ingest, relative):
        assert ingest.is_excluded(relative) is False

    def test_2wiki_universe_is_the_encoding_dir_and_2wiki_is_not(self, ingest):
        assert ingest.ENCODING_DIR["2wiki"] == "2wiki_universe"
        assert "data/canonical/2wiki" in ingest.EXCLUDED_TREES


# ── budget: the batch simulation the price depends on ────────────────────────


class TestBatchSimulation:
    """Dense sorts by length before batching and splade does not. Pricing that
    difference wrong is what made the two text representations look identical."""

    @pytest.fixture(scope="class")
    def budget(self):
        return module("m3a_webqsp_encode_budget")

    def test_length_sorting_puts_the_longest_documents_in_one_batch(self, budget):
        # One very long document among short ones. Sorted batching pays
        # batch_size * longest once; file-order batching would smear it.
        tokens = np.array([1000] + [1] * 63, dtype=np.int64)
        result = budget.sorted_batch_slots(tokens, cap=32768, batch=32)
        assert result["peak_batch_slots"] == 1000 * 32
        assert result["n_batches"] == 2

    def test_the_cap_is_applied_before_batching_not_after(self, budget):
        tokens = np.array([10**6] * 32, dtype=np.int64)
        result = budget.sorted_batch_slots(tokens, cap=100, batch=32)
        assert result["peak_batch_slots"] == 100 * 32

    def test_padded_slots_never_undercount_real_tokens(self, budget):
        rng = np.random.default_rng(0)
        tokens = rng.integers(1, 500, size=1000)
        for simulate in (budget.sorted_batch_slots, budget.file_order_batch_slots):
            result = simulate(tokens, cap=32768, batch=32)
            assert result["padded_token_slots"] >= int(tokens.sum())
            assert result["padding_overhead_ratio"] >= 1.0

    def test_sorting_costs_no_more_than_file_order(self, budget):
        """Sorting exists to reduce padding. If the simulation ever says
        otherwise, the simulation is wrong."""
        rng = np.random.default_rng(1)
        tokens = rng.integers(1, 500, size=4096)
        sorted_slots = budget.sorted_batch_slots(tokens, 32768, 32)["padded_token_slots"]
        file_slots = budget.file_order_batch_slots(tokens, 32768, 32)["padded_token_slots"]
        assert sorted_slots <= file_slots

    def test_a_ragged_final_batch_is_not_padded_to_a_full_one(self, budget):
        tokens = np.array([10] * 33, dtype=np.int64)
        result = budget.sorted_batch_slots(tokens, 32768, 32)
        assert result["padded_token_slots"] == 10 * 32 + 10 * 1

    def test_the_dense_cap_is_the_model_default_not_the_splade_cap(self, budget):
        # 32768 is what canonical_encode.py leaves in place by passing
        # max_seq=None. Reading it as 256 understates the tail by 128x.
        assert budget.DENSE_MAX_TOKENS == 32768
        assert budget.SPLADE_MAX_TOKENS == 256

    def test_pricing_is_monotone_in_work(self, budget):
        # Priced at realistic magnitudes: gpu_hours is rounded to two places,
        # so a few thousand tokens is genuinely 0.0 hours and would compare
        # equal for a reason that has nothing to do with the pricing.
        cheap = budget.price(10**7, budget.ASSUMED_DENSE_TOKENS_PER_GPU_SECOND)
        dear = budget.price(10**8, budget.ASSUMED_DENSE_TOKENS_PER_GPU_SECOND)
        assert dear["gpu_seconds_low"] > cheap["gpu_seconds_low"]
        assert dear["gpu_hours_low"] > cheap["gpu_hours_low"]
        assert dear["usd_high"] > cheap["usd_high"]

    def test_the_low_estimate_is_never_above_the_high_estimate(self, budget):
        result = budget.price(10**7, budget.ASSUMED_DENSE_TOKENS_PER_GPU_SECOND)
        assert result["gpu_hours_low"] <= result["gpu_hours_high"]
        assert result["usd_low"] <= result["usd_high"]


# ── the artifacts, when they exist ───────────────────────────────────────────


class TestTransferManifest:
    @pytest.fixture(scope="class")
    def manifest(self):
        return artifact("transfer_manifest.json")

    def test_the_package_is_outside_this_repository_and_read_only(self, manifest):
        assert manifest["root_is_outside_this_repository"] is True
        assert manifest["access_mode"] == "READ_ONLY"
        assert not str(ROOT) in manifest["package_root"].replace("/", "\\")

    def test_no_excluded_path_reached_the_inventory(self, manifest):
        ingest = module("m3a_transfer_ingest")
        offenders = [
            row["relative_path"]
            for row in manifest["files"]
            if ingest.is_excluded(row["relative_path"])
        ]
        assert offenders == []

    def test_the_2wiki_universe_graph_is_present(self, manifest):
        """The file the prefix bug dropped. Its absence was invisible."""
        paths = {row["relative_path"] for row in manifest["files"]}
        assert any(path.startswith("data/canonical/2wiki_universe/") for path in paths)

    def test_every_row_carries_a_size_and_a_full_sha256(self, manifest):
        for row in manifest["files"]:
            assert row["bytes"] >= 0
            assert len(row["sha256"]) == 64
            assert set(row["sha256"]) <= set("0123456789abcdef")

    def test_the_totals_are_the_sum_of_the_rows(self, manifest):
        assert manifest["measured"]["files"] == len(manifest["files"])
        assert manifest["measured"]["bytes"] == sum(r["bytes"] for r in manifest["files"])

    def test_paths_are_unique(self, manifest):
        paths = [row["relative_path"] for row in manifest["files"]]
        assert len(paths) == len(set(paths))

    def test_the_webqsp_identity_reads_real_keys_and_not_none(self, manifest):
        """An earlier draft guessed n_relations/n_triples, which do not exist,
        so the whole column rendered as '?'. A missing key reads as absent
        rather than as a wrong lookup, which is why this needs asserting."""
        webqsp = next(r for r in manifest["declared_identity"] if r["dataset"] == "webqsp")
        relation_space = webqsp["relation_space_identity"]
        assert relation_space["measured_relations"] == 7058
        assert relation_space["measured_triples"] == 8309195
        assert relation_space["hard_check_exact"] is True

    def test_the_union_scope_is_recorded_rather_than_left_implicit(self, manifest):
        webqsp = next(r for r in manifest["declared_identity"] if r["dataset"] == "webqsp")
        assert "union" in webqsp["corpus_identity"]["scope"].lower()


class TestAdoptionVerification:
    @pytest.fixture(scope="class")
    def verification(self):
        return artifact("adoption_verification.json")

    def test_the_metaqa_accounting_identity_closes(self, verification):
        identity = verification["section_b"]["accounting_identity"]
        assert all(identity.values())

    def test_the_metaqa_chain_adds_up_arithmetically(self, verification):
        """Not just the recorded booleans -- recompute from the counts, so a
        true flag over inconsistent numbers cannot pass."""
        chain = verification["section_b"]["chain"]
        assert (
            chain["source_lines"]
            == chain["parsed_triples"] + chain["blank_lines"] + chain["malformed_lines"]
        )
        assert chain["distinct_triples"] == chain["mapped_triples"] + chain["unmapped_triples"]
        assert chain["parsed_triples"] - chain["exact_duplicates"] == chain["distinct_triples"]
        assert chain["mapped_triples"] == chain["shipped_distinct_edges"]

    def test_unmapped_rows_are_enumerated_not_merely_counted(self, verification):
        section = verification["section_b"]
        assert len(section["enumerated_unmapped"]) == section["chain"]["unmapped_triples"]

    def test_the_source_hash_was_checked_against_the_declared_one(self, verification):
        section = verification["section_b"]
        assert section["source_sha256_agrees"] is True
        assert section["source_sha256_measured"] == section["source_sha256_declared"]

    def test_the_rog_property_is_not_silently_accepted_as_measured(self, verification):
        status = verification["section_c"]["rog_exact_claim"]["status"]
        assert status == "UPSTREAM_ASSERTED_NOT_LOCALLY_REPRODUCIBLE"

    def test_qualified_labels_are_collision_free_and_short_ones_are_not(self, verification):
        """This is why relation text must come from relation_label_qualified.
        If short labels ever became collision-free the guidance would change,
        so the asymmetry is asserted rather than assumed."""
        checks = verification["section_c"]["checks"]
        assert checks["relation_qualified_label_collisions"] == 0
        assert checks["relation_short_label_collisions"] > 0

    def test_the_graph_has_no_dangling_endpoints_or_unknown_relations(self, verification):
        checks = verification["section_c"]["checks"]
        assert checks["dangling_src"] == 0
        assert checks["dangling_dst"] == 0
        assert checks["edges_with_unknown_relation"] == 0
        assert checks["null_rows"] == 0

    def test_relation_text_is_not_encoded_before_the_audit_authorises_it(self, verification):
        assert verification["section_f"]["relation_text_encoded"] is False

    def test_the_two_relation_regimes_are_genuinely_different(self, verification):
        regimes = {row["dataset"]: row for row in verification["section_f"]["regimes"]}
        ratio = regimes["webqsp"]["relation_types"] / regimes["metaqa"]["relation_types"]
        assert ratio == pytest.approx(verification["section_f"]["cardinality_ratio"], rel=1e-3)
        assert ratio > 100

    def test_the_forbidden_generalisation_is_recorded(self, verification):
        forbidden = verification["section_f"]["forbidden_generalisation"].lower()
        assert "metaqa" in forbidden
        assert "never" in forbidden

    def test_entropy_never_exceeds_its_own_maximum(self, verification):
        for row in verification["section_f"]["regimes"]:
            assert row["entropy_bits"] <= row["max_entropy_bits"]
            assert 0.0 <= row["normalised_entropy"] <= 1.0

    def test_nothing_in_the_package_was_written(self, verification):
        assert verification["access_mode"] == "READ_ONLY"


class TestNodeAlignment:
    @pytest.fixture(scope="class")
    def alignment(self):
        return artifact("node_alignment.json")

    # A canonical node with no hyperlink or triple simply has no edge. That is
    # a property of the corpus, not a defect in the alignment, so `green`
    # excludes it -- and that exclusion is the one judgement call in the flag.
    ORPHANS = "canonical_nodes_with_no_graph_endpoint"

    def test_green_means_every_blocking_count_is_zero(self, alignment):
        """`green` is a derived flag. Recompute it from the counts, so a row
        cannot be green while stranding nodes it should not strand."""
        for row in alignment["alignments"]:
            blocking = {
                key: value for key, value in row["missing"].items() if key != self.ORPHANS
            }
            assert row["green"] == all(value == 0 for value in blocking.values()), row["dataset"]

    def test_orphan_nodes_do_not_block_and_that_is_deliberate(self, alignment):
        """If this ever passed vacuously -- no dataset having orphans -- the
        exclusion above would be untested rather than confirmed."""
        with_orphans = [
            row["dataset"]
            for row in alignment["alignments"]
            if row["missing"][self.ORPHANS] and row["green"]
        ]
        assert with_orphans, "no green dataset has orphans; the exclusion is untested"

    def test_the_five_embedded_datasets_are_green(self, alignment):
        green = {row["dataset"] for row in alignment["alignments"] if row["green"]}
        assert {"metaqa", "squad", "musique", "hotpotqa", "2wiki"} <= green

    def test_webqsp_is_the_only_blocked_dataset_and_for_the_stated_reason(self, alignment):
        blocked = [row["dataset"] for row in alignment["alignments"] if not row["green"]]
        assert blocked == ["webqsp"]
        assert "embedding" in alignment["webqsp_note"].lower()

    def test_alignment_is_reported_in_both_directions(self, alignment):
        """A bridge can cover every canonical node and still strand graph
        endpoints. One coverage number hides that; both directions do not.

        webqsp reports the graph and canonical directions only, because the
        embedding side does not exist yet -- there is no row set to report the
        reverse direction of."""
        for row in alignment["alignments"]:
            missing = row["missing"]
            assert "graph_endpoints_not_in_canonical" in missing
            assert "canonical_nodes_with_no_graph_endpoint" in missing
            assert "canonical_nodes_without_dense_row" in missing
            if row["dataset"] == "webqsp":
                assert "dense_rows_not_in_canonical" not in missing
            else:
                assert "dense_rows_not_in_canonical" in missing
                assert "splade_rows_not_in_canonical" in missing

    def test_musique_has_exactly_one_row_serving_two_canonical_nodes(self, alignment):
        """The off-by-one that made 117,533 embedding rows cover 117,534
        canonical nodes. It is legitimate, and it is the reason the alignment
        counts distinct canonical keys rather than rows."""
        musique = next(r for r in alignment["alignments"] if r["dataset"] == "musique")
        assert musique["counts"]["canonical_nodes"] == 117534
        assert musique["modalities"]["dense"]["rows"] == 117533
        assert musique["modalities"]["dense"]["rows_serving_more_than_one_canonical_node"] == 1

    def test_orphan_nodes_never_exceed_the_node_count(self, alignment):
        for row in alignment["alignments"]:
            orphans = row["missing"]["canonical_nodes_with_no_graph_endpoint"]
            assert 0 <= orphans <= row["counts"]["canonical_nodes"]

    def test_metaqa_and_webqsp_are_the_kb_regimes_with_no_orphans(self, alignment):
        """Every node in a KB graph is an endpoint by construction. A corpus
        dataset has documents nothing links to. If a KB dataset ever grew
        orphans, the graph and the node table would have drifted apart."""
        for dataset in ("metaqa", "webqsp"):
            row = next(r for r in alignment["alignments"] if r["dataset"] == dataset)
            assert row["missing"]["canonical_nodes_with_no_graph_endpoint"] == 0


class TestEncodeBudget:
    @pytest.fixture(scope="class")
    def budget(self):
        return artifact("webqsp_encode_budget.json")

    def test_nothing_was_encoded(self, budget):
        assert budget["missing_embeddings"]["node_embeddings_present"] == 0
        assert "compute record only" in budget["item"]

    def test_the_query_count_is_absent_rather_than_zero(self, budget):
        """Zero missing queries would mean the queries exist and are encoded.
        None means the set does not exist. Collapsing those two would hide the
        blocker completely."""
        missing = budget["missing_embeddings"]
        assert missing["query_embeddings_missing"] is None
        assert missing["query_set_exists"] is False

    def test_the_encoder_contract_matches_the_frozen_one(self, budget):
        dense = budget["encoder_contract"]["dense"]
        assert dense["model"] == "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        assert dense["dim"] == 1536
        assert dense["stored_dtype"] == "float16"
        assert dense["normalize_embeddings"] is True
        assert dense["document_prefix"] == "none"
        assert budget["encoder_contract"]["shard_size"] == 40000
        assert budget["encoder_contract"]["splade"]["vocab"] == 30522

    def test_the_throughput_assumption_is_labelled_as_an_assumption(self, budget):
        assert budget["assumptions"]["status"] == "DECLARED_NOT_MEASURED"

    def test_a_calibration_step_precedes_the_full_run(self, budget):
        assert budget["calibration_protocol"][0].startswith("encode shard 0 only")
        assert any("re-derive" in step for step in budget["calibration_protocol"])

    def test_the_ceiling_is_above_the_worst_estimate_and_aborts_rather_than_raises(self, budget):
        worst = max(
            option["estimated_dense"]["gpu_hours_high"]
            for option in budget["text_representation_options"].values()
        )
        assert budget["hard_ceiling"]["dense_gpu_hours"] >= worst
        assert "abort" in budget["hard_ceiling"]["rule"]

    def test_storage_is_rows_times_dim_times_two_bytes(self, budget):
        for option in budget["text_representation_options"].values():
            assert option["storage_dense_bytes"] == option["rows"] * 1536 * 2

    def test_both_representations_cover_exactly_the_node_count(self, budget):
        for option in budget["text_representation_options"].values():
            assert option["entity_rows"] + option["cvt_rows"] == option["rows"]
            assert option["rows"] == 2592894

    def test_name_plus_facts_is_measurably_more_expensive_not_merely_asserted(self, budget):
        """Pricing by rows made these identical. If that regression returns,
        the whole representation argument silently loses its cost half."""
        options = budget["text_representation_options"]
        plain = options["name_only"]["dense_batching"]["padded_token_slots"]
        facts = options["name_plus_facts"]["dense_batching"]["padded_token_slots"]
        assert facts > plain * 2

    def test_the_oom_hazard_is_recorded_when_a_peak_batch_does_not_fit(self, budget):
        for name, option in budget["text_representation_options"].items():
            if not option["dense_peak_batch_fits_in_gpu"]:
                assert any(name in hazard for hazard in budget["hazards"])

    def test_the_representation_choice_is_left_open_for_review(self, budget):
        assert budget["representation_choice_is_open"] is True
        assert len(budget["blocking_on_review"]) >= 3

    def test_the_execution_placement_forbids_detached_modal_runs(self, budget):
        assert budget["execution_placement"]["never"] == "modal run --detach"
        assert "spawn_modal_jobs" in budget["execution_placement"]["where"]


class TestAdoptionReport:
    @pytest.fixture(scope="class")
    def report(self):
        return artifact("adoption_report.json")

    def test_the_report_adopts_nothing_itself(self, report):
        assert "STOP_FOR_REVIEW" == report["next"]
        assert "recommends" in report["decides"]

    def test_verify_was_not_rebuild(self, report):
        overall = report["overall"]
        assert overall["verify_not_rebuild_held"] is True
        assert overall["second_vocabulary_built"] is False
        assert overall["second_graph_built"] is False
        assert overall["package_bytes_modified"] == 0
        assert overall["sidecar_artifacts_only"] is True

    def test_every_contract_section_has_a_verdict(self, report):
        covered = {row["section"] for row in report["sections"]}
        assert {"A", "B", "C", "D", "E", "F"} <= covered

    def test_no_section_failed(self, report):
        for row in report["sections"]:
            assert not row["status"].startswith("REJECT")
            assert "FAIL" not in row["status"]

    def test_every_section_names_where_its_numbers_came_from(self, report):
        for row in report["sections"]:
            assert row["provenance"]

    def test_the_training_steps_remain_unauthorised(self, report):
        blocked = " ".join(report["not_done_and_not_authorised"]).lower()
        for forbidden in ("train qls-u", "train gat", "train gat-no-mp"):
            assert forbidden in blocked

    def test_the_query_encode_is_not_priced_because_it_cannot_be(self, report):
        readiness = report["webqsp_encode_readiness"]
        assert readiness["query_encode_priced"] is False
        assert readiness["node_encode_priced"] is True

    def test_the_report_numbers_agree_with_the_artifacts_they_came_from(self, report):
        """The report reads its inputs rather than restating them. This is the
        assertion that keeps that true: if a figure were ever retyped, it would
        drift from the measurement here first."""
        verification = artifact("adoption_verification.json")
        section_c = next(r for r in report["sections"] if r["section"] == "C")
        counts = verification["section_c"]["counts"]
        assert f"{counts['relations']:,}" in section_c["finding"]
        assert f"{counts['edges']:,}" in section_c["finding"]

        alignment = artifact("node_alignment.json")
        section_e = next(r for r in report["sections"] if r["section"] == "E")
        green = sum(1 for row in alignment["alignments"] if row["green"])
        assert f"green on {green} of {len(alignment['alignments'])}" in section_e["finding"]

    def test_the_open_decisions_are_decisions_and_not_defects(self, report):
        assert len(report["open_decisions"]) >= 4
        for decision in report["open_decisions"]:
            assert decision["why_review"]
            assert decision["recommendation"]


class TestComputeReport:
    @pytest.fixture(scope="class")
    def compute(self):
        return artifact("compute_report.json")

    def test_this_phase_spent_no_gpu_time_and_no_money(self, compute):
        assert compute["spent"]["gpu_seconds"] == 0.0
        assert compute["spent"]["usd"] == 0.0

    def test_no_package_byte_was_modified(self, compute):
        assert compute["written"]["package_bytes_modified"] == 0

    def test_the_per_item_gpu_seconds_are_all_zero(self, compute):
        for row in compute["per_item"]:
            assert row["cost"]["gpu_seconds"] == 0.0

    def test_section_j_claim_is_tested_against_a_real_reconstruction_cost(self, compute):
        comparisons = compute["section_j_claim"]["comparisons"]
        held = [row for row in comparisons if row["verdict"] == "CLAIM_HOLDS"]
        assert held, "no comparison actually tests the claim"
        for row in held:
            assert row["verification_seconds"] < row["reconstruction_seconds"]
            assert row["ratio"] > 1

    def test_the_metaqa_row_admits_it_does_not_test_the_claim(self, compute):
        """Section B reconstructs MetaQA in full, so verification and
        reconstruction cost the same there. Reporting that as a win would be
        the easy dishonesty."""
        metaqa = next(
            row
            for row in compute["section_j_claim"]["comparisons"]
            if "MetaQA" in row["artifact"]
        )
        assert metaqa["verdict"] == "NOT_A_TEST_OF_THE_CLAIM"

    def test_the_projected_encode_is_marked_as_not_run(self, compute):
        projected = compute["projected_not_spent"]
        assert projected["webqsp_node_encode"]["status"] == "NOT_AUTHORISED_TO_RUN"
        assert projected["webqsp_query_encode"]["status"] == "CANNOT_BE_PRICED"

    def test_every_rerun_carries_its_reason(self, compute):
        assert compute["reruns_performed"]
        for entry in compute["reruns_performed"]:
            assert entry["why"]

    def test_the_wall_time_is_the_sum_of_the_instrumented_items(self, compute):
        recorded = [
            row["cost"]["wall_seconds"]
            for row in compute["per_item"]
            if row["cost"]["wall_seconds"]
        ]
        assert compute["spent"]["wall_seconds"] == pytest.approx(sum(recorded), abs=0.5)


# ── properties every M3A artifact shares ─────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "transfer_manifest.json",
        "adoption_verification.json",
        "node_alignment.json",
        "webqsp_encode_budget.json",
        "adoption_report.json",
        "compute_report.json",
    ],
)
class TestEveryArtifact:
    def test_it_names_the_script_that_generated_it(self, name):
        data = artifact(name)
        generated_by = data["generated_by"]
        assert (ROOT / generated_by).exists(), generated_by

    def test_it_says_what_it_decides(self, name):
        data = artifact(name)
        assert "nothing" in data["decides"].lower()

    def test_it_belongs_to_this_phase(self, name):
        assert artifact(name)["phase"] == "M3A"

    def test_it_never_points_at_a_path_inside_this_repository(self, name):
        """The package is external and read-only. An artifact that recorded a
        repo-internal package root would mean something got copied in."""
        data = artifact(name)
        root = data.get("package_root")
        if root:
            assert not Path(root).is_relative_to(ROOT)


# ── the declaration's own record of what happened ────────────────────────────


class TestExecutionRecord:
    """Amendment 3 is the rule; this block is the outcome filed against it.

    The tests that matter here are the ones that stop the record from being
    more generous than the artifacts it summarises -- a record claiming an
    item completed when its artifact says otherwise is worse than no record.
    """

    @pytest.fixture(scope="class")
    def record(self):
        import yaml

        config = ROOT / "configs" / "m3a_sota_information_contract.yaml"
        declaration = yaml.safe_load(config.read_text(encoding="utf-8"))
        return declaration["adoption_execution_record_2026_09_08"]

    def test_it_stops_for_review(self, record):
        assert record["next"] == "STOP_FOR_REVIEW"
        assert record["status"] == "ADOPTION_COMPLETE_PENDING_REVIEW"

    def test_every_authorised_item_has_a_recorded_outcome(self, record):
        import yaml

        config = ROOT / "configs" / "m3a_sota_information_contract.yaml"
        declaration = yaml.safe_load(config.read_text(encoding="utf-8"))
        authorised = declaration["amendment_3_2026_09_08"]["current_authorization"]["authorised"]
        assert len(record["items"]) == len(authorised)

    def test_the_encode_is_recorded_as_not_run(self, record):
        """Item 6 was authorised. It did not run, and the record has to say so
        in the status rather than only in a note."""
        item = record["items"]["6_webqsp_encode"]
        assert item["status"] == "BUDGET_FILED_ENCODE_NOT_RUN"
        assert len(item["blockers"]) == 2

    def test_the_training_steps_are_still_unauthorised(self, record):
        blocked = " ".join(record["still_not_authorised"]).lower()
        for forbidden in ("train qls-u", "train gat", "train gat-no-mp"):
            assert forbidden in blocked

    def test_verify_was_not_rebuild_and_the_metaqa_exception_is_explained(self, record):
        verify = record["verify_not_rebuild"]
        assert verify["held"] is True
        assert verify["second_relation_vocabulary_built"] is False
        assert verify["second_graph_built"] is False
        # Section B does rebuild MetaQA. An unqualified "held: true" would be
        # false; the note is what makes it true.
        assert "discarded" in verify["note"]

    def test_no_package_byte_was_modified(self, record):
        assert record["package_bytes_modified"] == 0
        assert record["access_mode"] == "READ_ONLY"

    def test_the_record_agrees_with_the_compute_artifact(self, record):
        compute = artifact("compute_report.json")
        item = record["items"]["9_compute_report"]
        assert item["gpu_seconds_spent"] == compute["spent"]["gpu_seconds"]
        assert item["usd_spent"] == compute["spent"]["usd"]
        assert item["wall_seconds"] == compute["spent"]["wall_seconds"]

    def test_the_record_agrees_with_the_adoption_report(self, record):
        report = artifact("adoption_report.json")
        assert record["overall_recommendation"] == report["overall"]["recommendation"]
        assert len(record["open_for_review"]) == len(report["open_decisions"])

    def test_the_record_agrees_with_the_inventory(self, record):
        manifest = artifact("transfer_manifest.json")
        measured = record["items"]["1_transfer_inventory"]["measured"]
        assert measured["files"] == manifest["measured"]["files"]
        assert measured["gib"] == manifest["measured"]["gib"]

    def test_the_rog_property_is_not_upgraded_by_the_summary(self, record):
        """The easy dishonesty: a summary that quietly promotes an
        upstream-asserted property to a verified one."""
        verification = artifact("adoption_verification.json")
        assert (
            record["items"]["4_webqsp_verification"]["rog_exact_property"]
            == verification["section_c"]["rog_exact_claim"]["status"]
        )

    def test_the_alignment_record_names_webqsp_as_the_only_blocker(self, record):
        alignment = artifact("node_alignment.json")
        blocked = [row["dataset"] for row in alignment["alignments"] if not row["green"]]
        assert record["items"]["5_node_alignment"]["blocked"] == blocked
