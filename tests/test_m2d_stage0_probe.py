"""The Stage-0 probe's pieces, and its arms against the file that declared them.

The probe's end-to-end run needs the sealed cell masters and M2B's checkpoints,
which live on the volume; the same is true of the M2C probe and this file
follows its convention of testing the parts that do not. What it adds is a set
of checks that the ARMS the container will run are the arms the declaration
named -- a launcher and a declaration that disagree produce results filed under
a protocol that did not describe them, and nothing downstream would notice.

The strict-load check is worth singling out. It is the difference between
comparing S4 against M2B's S3 and comparing it against a partly randomised
model wearing S3's name, and it runs here because the scorer is pure torch and
needs no data.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval import m2d_semantic_fusion as fusion
from mp_retrieval import run_artifacts
from scripts import run_m2d_stage0_probe as probe

DECLARATION = yaml.safe_load(
    (REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml").read_text(encoding="utf-8")
)
BASELINE_JSON = (
    REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "m2b_baseline_table.json"
)

PRECOMPUTED_WIDTH = 7
EMBEDDING_DIM = 32


def build(rung: str) -> torch.nn.Module:
    import scripts.run_m1a_feature_screen as _m1a
    import scripts.run_m2b_semantic_minimality as _m2b

    return _m1a.build_m1a_model(
        precomputed_width=PRECOMPUTED_WIDTH,
        semantic_rung=rung,
        dropout=0.2,
        temperature=0.07,
        embedding_dim=EMBEDDING_DIM,
        semantic_head=_m2b.build_semantic_head(rung, EMBEDDING_DIM),
    )


# ---------------------------------------------------------------------------
# The arms are the declaration's arms
# ---------------------------------------------------------------------------


def test_the_fusion_arms_are_exactly_the_five_the_declaration_names():
    declared = DECLARATION["stage_0"]["b_rank_fusion"]["arms"]
    assert sorted(declared) == ["Z0", "Z1", "Z2", "Z3", "Z4"]
    assert [name.split("_")[0] for name in probe.FUSION_ARMS] == sorted(declared)


@pytest.mark.parametrize(
    ("arm", "sources"),
    [
        ("Z0", ("S4",)),
        ("Z1", ("S4", "Dense")),
        ("Z2", ("S4", "SPLADE")),
        ("Z3", ("S4", "Dense", "SPLADE")),
        ("Z4", ("S4", "S3")),
    ],
)
def test_each_arm_fuses_the_sources_the_declaration_says_it_does(arm, sources):
    """Read off the declaration's own prose rather than restated here, so an
    arm cannot be quietly given an extra source."""

    described = DECLARATION["stage_0"]["b_rank_fusion"]["arms"][arm]
    implemented = next(v for k, v in probe.FUSION_ARMS.items() if k.startswith(f"{arm}_"))
    assert implemented == sources
    for source in sources:
        assert source.lower() in described.lower(), (
            f"{arm} fuses {source} and the declaration's description does not mention it"
        )


def test_the_context_arms_are_the_two_the_declaration_calls_cheap_plus_s3_alone():
    declared = DECLARATION["stage_0"]["b_rank_fusion"]["also_if_cheap_from_stored_ranks"]
    assert declared == ["RRF(S3, Dense)", "RRF(S3, SPLADE)"]
    assert probe.CONTEXT_ARMS["C1_S3_DENSE"] == ("S3", "Dense")
    assert probe.CONTEXT_ARMS["C2_S3_SPLADE"] == ("S3", "SPLADE")
    # S3 alone is the reference those two are read against; without it the
    # context arms say nothing about whether the incumbent moved.
    assert probe.CONTEXT_ARMS["C0_S3"] == ("S3",)


def test_the_s3_plus_s4_arm_is_named_diagnostic_in_the_arm_name_itself():
    """The label has to travel with the number. A key called "Z4" alone would
    be one careless copy away from a results table."""

    assert probe.DIAGNOSTIC_ONLY_ARMS == ("Z4_S4_S3_DIAGNOSTIC_ONLY",)
    assert "DIAGNOSTIC_ONLY" in probe.DIAGNOSTIC_ONLY_ARMS[0]
    assert (
        DECLARATION["stage_0"]["b_rank_fusion"]["z4_is_diagnostic_only"][
            "eligible_as_a_final_model"
        ]
        is False
    )


def test_no_context_arm_and_no_diagnostic_arm_can_be_reported_as_a_candidate():
    for name in (*probe.CONTEXT_ARMS, *probe.DIAGNOSTIC_ONLY_ARMS):
        eligible = name not in probe.DIAGNOSTIC_ONLY_ARMS and name not in probe.CONTEXT_ARMS
        assert eligible is False, f"{name} would be reported as an eligible model"
    for name in ("Z1_S4_DENSE", "Z2_S4_SPLADE", "Z3_S4_DENSE_SPLADE"):
        eligible = name not in probe.DIAGNOSTIC_ONLY_ARMS and name not in probe.CONTEXT_ARMS
        assert eligible is True


def test_every_arm_draws_only_on_rankers_the_probe_actually_builds():
    for sources in (*probe.FUSION_ARMS.values(), *probe.CONTEXT_ARMS.values()):
        assert set(sources) <= set(probe.RANKERS)


def test_the_coverage_split_partitions_the_rankers():
    assert set(probe.FULL_COVERAGE) | set(probe.PARTIAL_COVERAGE) == set(probe.RANKERS)
    assert not set(probe.FULL_COVERAGE) & set(probe.PARTIAL_COVERAGE)
    # Which side a ranker is on decides whether an absence is scored, so it is
    # not a cosmetic grouping: S4 and S3 score the pool, the stored lists do not.
    assert set(probe.PARTIAL_COVERAGE) == {"Dense", "SPLADE"}


# ---------------------------------------------------------------------------
# The family split is M2C's, not a new one
# ---------------------------------------------------------------------------


def test_the_family_split_matches_the_one_the_declaration_froze():
    for dataset in ("2wiki_clean", "hotpotqa_clean", "musique_clean", "squad_clean"):
        assert probe._family(dataset) == "passage"
    for dataset in ("metaqa", "webqsp"):
        assert probe._family(dataset) == "kb"


def test_every_dataset_in_the_baseline_table_is_assigned_a_family():
    if not BASELINE_JSON.exists():
        pytest.skip("the immutable M2B baseline table is not on this machine")
    import json

    table = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    datasets = {row["dataset"] for row in table["rows"]}
    families = {probe._family(dataset) for dataset in datasets}
    assert families == {"passage", "kb"}
    counted = sum(1 for dataset in datasets if probe._family(dataset) == "passage")
    assert counted == 4, "the passage family should hold four datasets"


def test_the_declared_stage_0_cells_split_two_and_two():
    cells = DECLARATION["stage_0"]["cells"]
    named = [*cells["failure_cells"], cells["passage_control"], cells["kb_control"]]
    families = [probe._family(cell.split("/")[0]) for cell in named]
    assert families.count("passage") == 3
    assert families.count("kb") == 1


# ---------------------------------------------------------------------------
# Loading a checkpoint strictly
# ---------------------------------------------------------------------------


def test_a_checkpoint_from_the_wrong_rung_is_refused_rather_than_partly_loaded(tmp_path):
    """The whole comparison rests on both sides being M2B's own fits."""

    s3 = build("S3")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(s3.state_dict(), checkpoint)

    with pytest.raises(RuntimeError):
        probe._load_model(
            checkpoint,
            rung="S4",
            precomputed_width=PRECOMPUTED_WIDTH,
            embedding_dim=EMBEDDING_DIM,
            dropout=0.2,
            temperature=0.07,
            device=torch.device("cpu"),
        )


def test_a_matching_checkpoint_loads_and_comes_back_in_eval_mode(tmp_path):
    s3 = build("S3")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(s3.state_dict(), checkpoint)

    loaded = probe._load_model(
        checkpoint,
        rung="S3",
        precomputed_width=PRECOMPUTED_WIDTH,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.2,
        temperature=0.07,
        device=torch.device("cpu"),
    )
    assert loaded.training is False
    for (name, mine), (other_name, theirs) in zip(
        sorted(loaded.state_dict().items()), sorted(s3.state_dict().items()), strict=True
    ):
        assert name == other_name
        assert torch.equal(mine, theirs)


def test_dropout_being_off_is_what_makes_the_ranking_reproducible(tmp_path):
    """A model left in train mode would give a different ranking on every pass,
    and the fusion arms would be measuring dropout noise."""

    s3 = build("S3")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(s3.state_dict(), checkpoint)
    loaded = probe._load_model(
        checkpoint,
        rung="S3",
        precomputed_width=PRECOMPUTED_WIDTH,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.9,
        temperature=0.07,
        device=torch.device("cpu"),
    )
    nodes = torch.randn(6, EMBEDDING_DIM)
    queries = torch.randn(1, EMBEDDING_DIM)
    batch_index = torch.zeros(6, dtype=torch.long)
    structural = torch.randn(6, PRECOMPUTED_WIDTH)
    with torch.no_grad():
        first = loaded.forward_explicit(nodes, queries, batch_index, structural)
        second = loaded.forward_explicit(nodes, queries, batch_index, structural)
    assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# The complementarity summary
# ---------------------------------------------------------------------------


POOL = np.array([10, 20, 30, 40], dtype=np.int64)


def row(order_s4: list[int], order_s3: list[int], gold: list[int]) -> dict:
    position = {int(node): index for index, node in enumerate(POOL.tolist())}

    def ranks(order: list[int]) -> np.ndarray:
        out = np.empty(POOL.size, dtype=np.int64)
        for rank, node in enumerate(order, start=1):
            out[position[node]] = rank
        return out

    relevant = np.isin(POOL, np.asarray(gold, dtype=np.int64))
    return fusion.complementarity(ranks(order_s4), ranks(order_s3), relevant, POOL)


def test_the_summary_counts_both_directions_and_reports_the_ratio():
    rows = [
        row([10, 20, 30, 40], [20, 10, 30, 40], gold=[20]),  # S4 wrong, S3 right
        row([10, 20, 30, 40], [20, 10, 30, 40], gold=[20]),  # again
        row([30, 20, 10, 40], [10, 20, 30, 40], gold=[30]),  # S4 right, S3 wrong
        row([10, 20, 30, 40], [10, 20, 30, 40], gold=[10]),  # both right
    ]
    summary = probe._summarise_complementarity(rows)
    assert summary["queries"] == 4
    assert summary["counts"]["s4_wrong_s3_right"] == 2
    assert summary["counts"]["s3_wrong_s4_right"] == 1
    assert summary["counts"]["both_right_at_1"] == 1
    assert summary["disagreement_at_1"]["total"] == 3
    assert summary["disagreement_at_1"]["s4_wrong_s3_right_share_of_disagreements"] == 2 / 3


def test_a_dominated_cell_reads_as_one_sided():
    """The shape the declaration says means there is nothing to fuse."""

    rows = [row([10, 20, 30, 40], [20, 10, 30, 40], gold=[20]) for _ in range(10)]
    summary = probe._summarise_complementarity(rows)
    assert summary["disagreement_at_1"]["s4_wrong_s3_right_share_of_disagreements"] == 1.0
    assert summary["counts"]["s3_wrong_s4_right"] == 0


def test_the_cross_rank_distributions_skip_queries_with_no_gold():
    """A query with no relevant candidate has no 'top relevant candidate' to
    locate under the other model, and averaging a zero in would drag the
    distribution toward a rank that was never measured."""

    rows = [
        row([10, 20, 30, 40], [20, 10, 30, 40], gold=[20]),
        row([10, 20, 30, 40], [20, 10, 30, 40], gold=[]),
    ]
    summary = probe._summarise_complementarity(rows)
    assert summary["queries"] == 2
    assert summary["queries_with_a_relevant_candidate_in_the_pool"] == 1
    assert summary["cross_ranks"]["s3_best_relevant_rank_under_s4"]["mean"] == 2.0


def test_a_panel_where_no_query_has_a_gold_returns_none_rather_than_a_number():
    rows = [row([10, 20, 30, 40], [20, 10, 30, 40], gold=[])]
    summary = probe._summarise_complementarity(rows)
    assert summary["cross_ranks"]["s3_best_relevant_rank_under_s4"] is None
    assert summary["same_best_relevant_node_share"] == 0.0


def test_an_empty_panel_does_not_divide_by_zero():
    summary = probe._summarise_complementarity([])
    assert summary["queries"] == 0
    assert summary["disagreement_at_1"]["share_of_queries"] == 0.0
    assert summary["top_k_overlap"]["1"]["mean"] == 0.0


def test_top_k_overlap_is_reported_as_a_share_of_the_cutoff():
    """Two orderings that agree on the top 5 have overlap 5 at k=5; the share
    is what makes k=1 and k=20 readable side by side."""

    rows = [row([10, 20, 30, 40], [10, 20, 30, 40], gold=[10])]
    summary = probe._summarise_complementarity(rows)
    assert summary["top_k_overlap"]["1"]["mean_share"] == 1.0
    assert summary["top_k_overlap"]["5"]["mean"] == 4.0  # the pool holds only four


# ---------------------------------------------------------------------------
# The probe's own contract
# ---------------------------------------------------------------------------


def test_the_probe_cannot_be_run_without_an_identity_for_its_result():
    """Every field that makes the artifact path unique is required, because a
    result that cannot say which commit and run produced it is the failure the
    persistence work exists to prevent."""

    parser = probe.build_parser()
    required = {
        action.dest for action in parser._actions if getattr(action, "required", False)
    }
    assert {"source_commit", "config_fingerprint", "output_root", "dataset", "regime"} <= required


def test_the_probe_has_no_flag_that_could_train_or_sweep_anything():
    parser = probe.build_parser()
    flags = {action.dest for action in parser._actions}
    for forbidden in ("epochs", "learning_rate", "lr", "weight", "sweep", "rrf_constant"):
        assert forbidden not in flags, f"--{forbidden} would make this not a zero-training probe"


def test_the_probe_declares_the_phase_and_arm_the_artifact_path_will_carry():
    assert probe.PHASE == "m2d"
    assert probe.ARM == "stage0_diagnostic"
    assert probe.COMPLETE_STATUS == "M2D_STAGE0_COMPLETE"


def test_the_passage_family_constant_is_the_set_the_declaration_named():
    """The split is inherited from M2C, not chosen here.

    The probe reports its diagnostics by family and the compute record prices
    the panels by family, and both read this one constant. If it drifted from
    the declared definition, every family-conditioned number in the phase would
    be reported under a split nobody declared.
    """

    definition = DECLARATION["failure_shape"][
        "family_split_seed_0_mean_s4_minus_s3_pp"
    ]["definition"]
    head, _, tail = definition.partition("passage-graph =")
    assert head.strip() == "", definition
    named = {item.strip() for item in tail.split("(")[0].split(",")}
    assert named == set(probe.PASSAGE_FAMILY)

    kb_text = definition.split("KB =")[1].split("(")[0]
    kb = {item.strip() for item in kb_text.split(",")}
    assert kb.isdisjoint(probe.PASSAGE_FAMILY)
    for dataset in kb:
        assert probe._family(dataset) == "kb"
    for dataset in named:
        assert probe._family(dataset) == "passage"


# --------------------------------------------------------------------------
# The artifact the container will try to write
# --------------------------------------------------------------------------

PROBE_SOURCE = (REPO_ROOT / "scripts" / "run_m2d_stage0_probe.py").read_text(encoding="utf-8")
PROBE_TREE = ast.parse(PROBE_SOURCE)


def _declared_rows_at() -> str:
    """The rows_at the probe passes to write_artifact, read from the call."""

    for node in ast.walk(PROBE_TREE):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if name != "write_artifact":
            continue
        for keyword in node.keywords:
            if keyword.arg == "rows_at":
                assert isinstance(keyword.value, ast.Constant), "rows_at must be a literal"
                return keyword.value.value
    raise AssertionError("the probe does not call write_artifact")


def _payload_node(dotted: str) -> ast.AST:
    """Walk the payload literal along a dotted rows_at path."""

    literal = None
    for node in ast.walk(PROBE_TREE):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "result"
            and isinstance(node.value, ast.Dict)
        ):
            literal = node.value
    assert literal is not None, "the probe's payload is no longer a dict literal named result"

    current: ast.AST = literal
    for key in dotted.split("."):
        assert isinstance(current, ast.Dict), (
            f"rows_at={dotted!r} descends into a {type(current).__name__}, whose "
            f"contents are computed rather than written out, so nothing here can "
            f"establish that {key!r} is a list. Point rows_at at a literal list."
        )
        match = [
            value
            for name, value in zip(current.keys, current.values, strict=True)
            if isinstance(name, ast.Constant) and name.value == key
        ]
        assert match, f"the payload has no key {key!r} on the path {dotted!r}"
        current = match[0]
    return current


def test_the_row_count_counts_something_countable() -> None:
    """Reproduces a failure that cost four containers their last step.

    write_artifact refuses a rows_at that resolves to anything but a list -- a
    row count over a dict counts nothing and would let a truncated artifact
    read back as a valid one. The probe pointed it at ``rankers.S4``, which is
    a metrics dict, and every cell died at the write, after the whole panel had
    been scored and paid for.

    Nothing about that needed a container to discover: the payload is a literal
    in this file, so the type of what rows_at names can be read off the source.
    """

    node = _payload_node(_declared_rows_at())
    produces_a_list = isinstance(node, (ast.List, ast.ListComp)) or (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("sorted", "list")
    )
    assert produces_a_list, (
        f"rows_at={_declared_rows_at()!r} names a "
        f"{type(node).__name__}, and write_artifact refuses anything but a list"
    )


def test_the_counted_rows_are_the_arms_the_declaration_names() -> None:
    """A count is only a check if it counts the thing that could go missing."""

    assert _declared_rows_at() == "arms_scored"
    node = _payload_node("arms_scored")
    assert isinstance(node, ast.Call) and node.func.id == "sorted"
    assert isinstance(node.args[0], ast.Name) and node.args[0].id == "per_arm"


def _ranks(order: list[int]) -> np.ndarray:
    position = {int(node): index for index, node in enumerate(POOL.tolist())}
    out = np.empty(POOL.size, dtype=np.int64)
    for rank, node in enumerate(order, start=1):
        out[position[node]] = rank
    return out


def test_every_block_the_payload_is_built_from_survives_the_write() -> None:
    """The last runtime-only failure class on this path, closed here.

    The payload is assembled from numpy arrays. A block that hands back a
    numpy scalar instead of a float, or a NaN from an empty mean, is valid
    Python and fails only at write_artifact -- which is to say after the whole
    panel has been scored, in a container, on the last line. That has now
    happened twice on this probe for two different reasons, so each builder is
    pushed through the real digest here, on synthetic ranks. content_digest
    round-trips through JSON with allow_nan=False, so it refuses exactly what
    the container would refuse.
    """

    relevant = np.isin(POOL, np.asarray([20], dtype=np.int64))
    ranks = {
        "S4": _ranks([10, 20, 30, 40]),
        "S3": _ranks([20, 10, 30, 40]),
        "Dense": _ranks([30, 20, 10, 40]),
        "SPLADE": _ranks([40, 20, 30, 10]),
    }

    per_query = [fusion.metrics(ranks[name], relevant) for name in ranks]
    fused = fusion.fuse_ranks(
        [fusion.source_contribution(POOL, np.asarray([20, 10], dtype=np.int64))], POOL
    )
    rescue = fusion.rescue_row(
        ranks["S4"], {name: ranks[name] for name in ("Dense", "SPLADE", "S3")}, relevant
    )
    assert rescue is not None, "S4 is wrong at rank 1 here, so this query is in the population"

    blocks = {
        "metrics": per_query,
        "mean_metrics": fusion.mean_metrics(per_query),
        "mean_metrics_of_nothing": fusion.mean_metrics([]),
        "fused_metrics": fusion.metrics(fused, relevant),
        "complementarity": probe._summarise_complementarity(
            [fusion.complementarity(ranks["S4"], ranks["S3"], relevant, POOL)]
        ),
        "complementarity_of_nothing": probe._summarise_complementarity([]),
        "rescue": fusion.rescue_table(
            [rescue],
            names=("Dense", "SPLADE", "S3"),
            excluded={"no_relevant_in_pool": 1, "s4_top1_already_right": 2},
        ),
        "rescue_of_nothing": fusion.rescue_table(
            [], names=("Dense", "SPLADE", "S3"), excluded={"no_relevant_in_pool": 0}
        ),
    }
    for name, value in blocks.items():
        try:
            run_artifacts.content_digest(value)
        except (TypeError, ValueError) as error:
            raise AssertionError(f"the {name} block cannot be written: {error}") from error


def test_the_empty_cases_do_not_produce_a_nan_the_writer_would_refuse() -> None:
    """allow_nan=False is what makes the check above bite, so it is asserted."""

    with pytest.raises(ValueError):
        run_artifacts.content_digest({"mrr": float("nan")})
