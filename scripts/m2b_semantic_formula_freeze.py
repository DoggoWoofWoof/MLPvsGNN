"""Every semantic output column M2B varies, written out and then checked.

M2B's whole claim is that S2, S3 and S4 differ only in how a query and a
candidate are compared, and that the three sit on a real capacity ladder:
fixed statistics, two learned vectors, a learned projection. A declaration that
records only the widths -- 3, 5, 258 -- and the parameter counts -- 0, 3072,
196608 -- does not say what any of those columns IS, and 258 in particular is a
number nobody should have to reverse-engineer into ``4 * 64 + 2``.

So this script writes the formula for every column, and then refuses to believe
its own transcription. Each formula is implemented a second time here, straight
from the notation the declaration will carry, and required to reproduce what
the live module emits on a deterministic probe. A formula that is stated
correctly and a formula that is stated plausibly look identical in a YAML file;
they do not survive this.

The three heads are instantiated at the frozen payload width, so every count
here is ``numel()`` on a real module and never ``2 * dim * 64`` evaluated in
prose. That distinction already cost this track once: the same expression at
768 gives 98,304, which is half of what the control actually costs at 1536.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.m1a_screen import M1AScorer  # noqa: E402
from mp_retrieval.m2b_semantic_control import ProjectionSemanticHead  # noqa: E402
from mp_retrieval.qls_v2_semantic import EPS, SemanticHead  # noqa: E402

OUTPUT_PATH = (
    REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "semantic_formula_freeze.json"
)

#: The payload M1A, M1B and M2 all ran against, and the width every count below
#: is measured at. Not qls_v2_semantic.EMBEDDING_DIM, which is 768.
FROZEN_EMBEDDING_DIM = 1536
PRECOMPUTED_WIDTH = 9
#: M2's frozen scorer settings, transcribed here and asserted against
#: qls_universal.hyperparameters by tests/test_m2b_semantic_formula_freeze.py.
#: Neither touches a parameter count -- dropout is inert under eval() and the
#: temperature scales a loss -- but building the scorer under different numbers
#: than M2 froze would make this a description of a different model.
SCORER_DROPOUT = 0.2
SCORER_TEMPERATURE = 0.07

PROBE_SEED = 20260907
PROBE_CANDIDATES = 17

#: Reference and module differ only by floating-point association order -- the
#: reference computes a column at a time, the module a batched matmul -- so an
#: exactly-zero requirement would fail for a correct formula. The comparison is
#: relative because S3's learned columns sum 1536 float32 terms and land near
#: magnitude 40, where 1e-5 of absolute drift is 3e-7 of relative drift; a
#: mis-stated formula misses by a factor, not by an ulp. Both the absolute
#: difference and the scale it is measured against are reported.
TOLERANCE = 1e-5


def _agrees(emitted: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    difference = float((emitted - expected).abs().max())
    scale = max(float(expected.abs().max()), 1.0)
    return {
        "max_absolute_difference_from_its_own_formula": difference,
        "scale": scale,
        "relative_difference": difference / scale,
        "formula_reproduces_the_module": difference <= TOLERANCE * scale,
    }


def _probe(dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(PROBE_SEED)
    query = torch.randn(dim, generator=generator)
    candidates = torch.randn(PROBE_CANDIDATES, dim, generator=generator)
    return query, candidates


# --- the formulas, restated independently of the modules ----------------------
#
# Each function below implements exactly what the ``formula`` string beside it
# says, using nothing from the module it checks. If the string is wrong, the
# implementation is wrong the same way and the comparison fails.


def _cosine(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    unit_q = query / max(float(query.norm()), EPS)
    return torch.stack([
        row @ unit_q / max(float(row.norm()), EPS) for row in candidates
    ])


def _dot_percentile(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    dots = [float(row @ query) for row in candidates]
    count = len(dots)
    if count <= 1:
        return torch.full((count,), 0.5)
    ranks = []
    for value in dots:
        below = sum(1 for other in dots if other < value)
        tied = sum(1 for other in dots if other == value)
        # average of the ordinal ranks the tied group spans
        ranks.append(below + (tied - 1) / 2)
    return torch.tensor([rank / (count - 1) for rank in ranks], dtype=candidates.dtype)


def _mean_abs_diff(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    return torch.stack([(row - query).abs().mean() for row in candidates])


def _weighted_product(weight: torch.Tensor) -> Callable[..., torch.Tensor]:
    def compute(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return torch.stack([((row * query) * weight).sum() for row in candidates])

    return compute


def _weighted_absolute_difference(weight: torch.Tensor) -> Callable[..., torch.Tensor]:
    def compute(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return torch.stack([((row - query).abs() * weight).sum() for row in candidates])

    return compute


def _projection_states(head: ProjectionSemanticHead, query, candidates):
    """GELU, then L2 normalise, on each of the two bias-free projections."""

    raw_query = F.gelu(head.query_projection.weight @ query)
    raw_nodes = torch.stack([F.gelu(head.node_projection.weight @ row) for row in candidates])
    unit_query = raw_query / raw_query.norm().clamp_min(1e-12)
    unit_nodes = torch.stack([row / row.norm().clamp_min(1e-12) for row in raw_nodes])
    return raw_query, raw_nodes, unit_query, unit_nodes


def _with_nonzero_weights(head: SemanticHead) -> SemanticHead:
    """Give S3's learned vectors values before checking their formulas.

    ``difference_weight`` initialises to zero, so ``sum_i v_i * |q_i - d_i|`` is
    identically zero at init and ANY stated formula would reproduce it. The
    weights are the thing S3 learns; verifying its columns at initialisation
    would verify nothing about them.
    """

    generator = torch.Generator().manual_seed(PROBE_SEED + 1)
    with torch.no_grad():
        head.product_weight.copy_(torch.randn(head.dim, generator=generator))
        head.difference_weight.copy_(torch.randn(head.dim, generator=generator))
    return head


# --- the frozen description ---------------------------------------------------


def s2_columns() -> list[dict[str, Any]]:
    return [
        {
            "name": "cosine_qd",
            "formula": "cos(q, d) = (q / max(||q||_2, eps)) . (d / max(||d||_2, eps))",
            "eps": EPS,
            "parameters": 0,
            "reference": _cosine,
        },
        {
            "name": "dot_qd_pct",
            "formula": (
                "rank of <q, d> among the candidates of this query, ties taking the "
                "average of the ordinal ranks they span, divided by (n - 1); exactly "
                "0.5 when n <= 1"
            ),
            "why_average_ties": (
                "ordinal ranking resolves ties by position, so permuting the candidate "
                "list would change the feature; averaging makes it a function of the "
                "multiset alone"
            ),
            "parameters": 0,
            "reference": _dot_percentile,
        },
        {
            "name": "mean_abs_diff",
            "formula": "(1 / dim) * sum_i |d_i - q_i|",
            "parameters": 0,
            "reference": _mean_abs_diff,
        },
    ]


def s3_columns(head: SemanticHead) -> list[dict[str, Any]]:
    return s2_columns() + [
        {
            "name": "semantic_product",
            "formula": "sum_i w_i * q_i * d_i,  w in R^dim, initialised w_i = 1/dim",
            "what_it_is": (
                "the diagonal restriction of the bilinear form q^T W d that a rank-64 "
                "projection approximates -- strictly less expressive there"
            ),
            "parameters": int(head.product_weight.numel()),
            "reference": _weighted_product(head.product_weight.detach()),
        },
        {
            "name": "semantic_difference",
            "formula": "sum_i v_i * |q_i - d_i|,  v in R^dim, initialised v_i = 0",
            "what_it_is": (
                "a learned weighted L1. |q - d| is not bilinear in q and d, so no "
                "projection could express it: new capacity, not only cheaper capacity"
            ),
            "parameters": int(head.difference_weight.numel()),
            "reference": _weighted_absolute_difference(head.difference_weight.detach()),
        },
    ]


def s4_blocks(head: ProjectionSemanticHead) -> dict[str, Any]:
    """The projection control, block by block, with nothing derived from 258."""

    projection = head.projection_dim
    return {
        "step_1_query_projection": {
            "formula": "g = GELU(W_q q),  W_q in R^{P x dim}, bias=False",
            "shape": [projection],
            "parameters": int(head.query_projection.weight.numel()),
            "why_no_bias": (
                "operator_models.OperatorModel, where the historical projection lives, "
                "is bias-free; a bias here would be a quiet 128-parameter departure "
                "from the thing this controls for"
            ),
        },
        "step_2_candidate_projection": {
            "formula": "h = GELU(W_d d),  W_d in R^{P x dim}, bias=False",
            "shape": [projection],
            "parameters": int(head.node_projection.weight.numel()),
        },
        "step_3_normalisation": {
            "formula": "qhat = g / max(||g||_2, 1e-12);  dhat = h / max(||h||_2, 1e-12)",
            "note": "torch.nn.functional.normalize over the last dimension",
        },
        "output_blocks": [
            {"name": "query_state", "formula": "qhat, broadcast over the candidates",
             "width": projection},
            {"name": "node_state", "formula": "dhat", "width": projection},
            {"name": "state_product", "formula": "qhat * dhat (elementwise)",
             "width": projection},
            {"name": "state_absolute_difference", "formula": "|qhat - dhat| (elementwise)",
             "width": projection},
        ],
        "output_scalars": [
            {"name": "normalized_state_dot", "formula": "sum_k qhat_k * dhat_k", "width": 1},
            {"name": "raw_projection_dot_scaled",
             "formula": "(sum_k g_k * h_k) / sqrt(P)", "width": 1,
             "note": "the RAW projections, before normalisation"},
        ],
        "width_derivation": {
            "blocks": 4,
            "projection_dim": projection,
            "scalars": 2,
            "expression": "4 * P + 2",
            "width": 4 * projection + 2,
        },
        "parameter_derivation": {
            "expression": "2 * dim * P",
            "measured": int(sum(p.numel() for p in head.parameters())),
        },
    }


# --- checking the transcription ------------------------------------------------


def verify_scalar_columns(head, columns: list[dict[str, Any]], dim: int) -> list[dict[str, Any]]:
    """Each named column against an implementation of its own formula text."""

    query, candidates = _probe(dim)
    with torch.no_grad():
        emitted = head(query, candidates)
    if emitted.shape[1] != len(columns):
        raise SystemExit(
            f"{head.rung} emits {emitted.shape[1]} columns but {len(columns)} are described"
        )
    if tuple(head.feature_names) != tuple(column["name"] for column in columns):
        raise SystemExit(
            f"{head.rung} names its columns {tuple(head.feature_names)}, the description "
            f"names {tuple(column['name'] for column in columns)}"
        )

    results = []
    for index, column in enumerate(columns):
        with torch.no_grad():
            expected = column["reference"](query, candidates)
        results.append(
            {key: value for key, value in column.items() if key != "reference"}
            | {"column_index": index}
            | _agrees(emitted[:, index], expected)
        )
    return results


def verify_projection_blocks(head: ProjectionSemanticHead, dim: int) -> dict[str, Any]:
    """Every S4 block against the formula text, sliced out by position."""

    query, candidates = _probe(dim)
    with torch.no_grad():
        emitted = head(query, candidates)
        raw_query, raw_nodes, unit_query, unit_nodes = _projection_states(
            head, query, candidates
        )
        projection = head.projection_dim
        expected = {
            "query_state": unit_query.expand(PROBE_CANDIDATES, projection),
            "node_state": unit_nodes,
            "state_product": unit_query * unit_nodes,
            "state_absolute_difference": (unit_query - unit_nodes).abs(),
            "normalized_state_dot": (unit_query * unit_nodes).sum(dim=-1, keepdim=True),
            "raw_projection_dot_scaled": (raw_query * raw_nodes).sum(dim=-1, keepdim=True)
            / projection**0.5,
        }

    per_block, cursor = {}, 0
    for name, block in expected.items():
        width = block.shape[1]
        per_block[name] = {"columns": [cursor, cursor + width]} | _agrees(
            emitted[:, cursor: cursor + width], block
        )
        cursor += width

    return {
        "emitted_width": int(emitted.shape[1]),
        "described_width": cursor,
        "widths_agree": int(emitted.shape[1]) == cursor,
        "per_block": per_block,
    }


def _scorer_facts(semantic_head, dim: int) -> dict[str, Any]:
    """What the frozen scorer becomes at this rung, measured on a real model."""

    model = M1AScorer(
        precomputed_width=PRECOMPUTED_WIDTH,
        semantic_rung=semantic_head.rung,
        embedding_dim=dim,
        semantic_head=semantic_head,
        dropout=SCORER_DROPOUT,
        temperature=SCORER_TEMPERATURE,
    )
    semantic = sum(p.numel() for p in model.semantic_head.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {
        "input_width": int(model.scorer[0].in_features),
        "precomputed_width": PRECOMPUTED_WIDTH,
        "semantic_width": len(semantic_head.feature_names),
        "architecture": "nn.Sequential(Linear(W, 32), GELU, Dropout(0.2), Linear(32, 1))",
        "scorer_parameters": total - semantic,
        "scorer_parameter_formula": "32 * W + 65",
        "scorer_parameters_from_formula": 32 * int(model.scorer[0].in_features) + 65,
        "semantic_parameters": semantic,
        "total_parameters": total,
    }


def build_report(dim: int = FROZEN_EMBEDDING_DIM) -> dict[str, Any]:
    s2 = SemanticHead(rung="S2", dim=dim)
    s3 = _with_nonzero_weights(SemanticHead(rung="S3", dim=dim))
    s4 = ProjectionSemanticHead(dim=dim)

    rungs = {
        "S2": {
            "what_it_is": "fixed semantic statistics; no learned semantic parameter at all",
            "module": "src/mp_retrieval/qls_v2_semantic.py::SemanticHead(rung='S2')",
            "columns": verify_scalar_columns(s2, s2_columns(), dim),
            "scorer": _scorer_facts(s2, dim),
        },
        "S3": {
            "what_it_is": "S2's three statistics plus two learned vectors over the payload",
            "module": "src/mp_retrieval/qls_v2_semantic.py::SemanticHead(rung='S3')",
            "weights_were_randomised_before_checking": (
                "difference_weight initialises to zero, so its column would reproduce any "
                "stated formula at init; both S3 vectors carry random values here."
            ),
            "initialisation_is_part_of_the_design": (
                "w_i = 1/dim makes semantic_product the mean elementwise product, a "
                "monotone function of the raw dot already available at S2, and v_i = 0 "
                "makes semantic_difference identically zero. S3 therefore starts as S2 "
                "plus one redundant channel and can only depart from it by learning, so "
                "an S3-over-S2 gain is attributable to the weights rather than to the "
                "channels existing."
            ),
            "columns": verify_scalar_columns(s3, s3_columns(s3), dim),
            "scorer": _scorer_facts(s3, dim),
        },
        "S4": {
            "what_it_is": "a learned projection of query and candidate into P dimensions",
            "module": "src/mp_retrieval/m2b_semantic_control.py::ProjectionSemanticHead",
            "blocks": s4_blocks(s4),
            "verification": verify_projection_blocks(s4, dim),
            "scorer": _scorer_facts(s4, dim),
        },
    }

    failures: list[str] = []
    for rung in ("S2", "S3"):
        for column in rungs[rung]["columns"]:
            if not column["formula_reproduces_the_module"]:
                failures.append(f"{rung}.{column['name']} does not match its stated formula")
    verification = rungs["S4"]["verification"]
    if not verification["widths_agree"]:
        failures.append("S4's described blocks do not span the width it emits")
    for name, block in verification["per_block"].items():
        if not block["formula_reproduces_the_module"]:
            failures.append(f"S4.{name} does not match its stated formula")
    for rung, entry in rungs.items():
        scorer = entry["scorer"]
        if scorer["scorer_parameters"] != scorer["scorer_parameters_from_formula"]:
            failures.append(f"{rung}: 32 * W + 65 does not give the scorer's real size")
        if scorer["input_width"] != scorer["precomputed_width"] + scorer["semantic_width"]:
            failures.append(f"{rung}: the scorer's input width is not 9 + the semantic width")

    return {
        "status": "M2B_SEMANTIC_FORMULA_FREEZE_COMPLETE",
        "why_this_exists": (
            "The declaration must say what each semantic column is, not only how many "
            "there are. 258 in particular is 4 * 64 + 2 and nobody should have to "
            "reverse-engineer that from the number."
        ),
        "frozen_embedding_dim": dim,
        "why_that_width": (
            "The payload every M1A/M1B/M2 fit ran against. qls_v2_semantic.EMBEDDING_DIM "
            "is 768, and 2 * 768 * 64 = 98,304 is the number this track repeated for "
            "months; the same projection on the real payload costs 196,608."
        ),
        "how_each_formula_was_checked": (
            "Implemented a second time in this script from the formula text beside it, "
            f"and required to reproduce the live module within a relative {TOLERANCE} on a "
            f"{PROBE_CANDIDATES}-candidate probe at seed {PROBE_SEED}. The two differ "
            "only in floating-point association order."
        ),
        "the_ladder": {
            "S2": "fixed semantic statistics",
            "S3": "tiny learned semantic weighting",
            "S4": "high-capacity learned projection representation",
            "the_question_this_makes_askable": (
                "How much learned semantic capacity does explicit graph structure need?"
            ),
        },
        "rungs": rungs,
        "measured_ladder": {
            rung: {
                "semantic_columns": entry["scorer"]["semantic_width"],
                "semantic_parameters": entry["scorer"]["semantic_parameters"],
                "scorer_input_width": entry["scorer"]["input_width"],
                "total_parameters": entry["scorer"]["total_parameters"],
            }
            for rung, entry in rungs.items()
        },
        "failed_checks": failures,
        "verdict": "FORMULAS_FROZEN" if not failures else "FORMULA_TRANSCRIPTION_WRONG",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    report = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "measured_ladder": report["measured_ladder"],
        "failed_checks": report["failed_checks"],
        "verdict": report["verdict"],
    }, indent=2))
    return 0 if not report["failed_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
