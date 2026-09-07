#!/usr/bin/env python
"""Prove the M1AScorer injection left S3 behaviourally identical, or forbid reuse.

M2B reuses 14 completed S3 fits rather than re-running them. Those fits were
produced by a version of ``m1a_screen.py`` that no longer exists in the working
tree: M2B added an optional ``semantic_head`` parameter so a different semantic
branch could be injected. The change is meant to be inert for every existing
caller, and the full suite passing is consistent with that -- but "the tests
still pass" is not the same claim as "an S3 model built today is the same
function as the S3 model that produced those numbers".

This script makes the stronger claim checkable. It extracts the exact
``m1a_screen.py`` the M2 fits ran under from git, loads it alongside the current
one as a separate module, and compares live objects:

* the state dict's keys, shapes and dtypes;
* the semantic branch's output tensor;
* the concatenated matrix the scorer actually receives, captured by a forward
  hook rather than recomputed;
* the forward scores, required bit-exact under identical weights and input.

Where a real M2 checkpoint is available it goes further and loads those trained
weights into the current implementation with ``strict=True``, then requires the
same equalities. That is the only check that exercises the actual numbers being
reused.

The verdict is a gate, not a report. ``S3_REUSE_PERMITTED`` means the 14 fits
may stand. Anything else means they may not, and M2B's workload becomes 42 new
fits rather than 28 -- a cost decision rather than a paperwork one, so the
script exits non-zero rather than letting a caller miss it.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.m1a_screen import M1AScorer  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "s3_reuse_proof.json"

#: The commit whose containers ran M2's 15 fits. Not the declaration commit and
#: not HEAD: the question is what produced the numbers being reused.
M2_FIT_COMMIT = "eae453a"
BASELINE_SOURCE = "src/mp_retrieval/m1a_screen.py"

#: M2's frozen construction. Transcribed here and asserted against
#: qls_universal by tests/test_m2b_s3_reuse_proof.py rather than trusted.
FROZEN = {
    "precomputed_width": 9,
    "semantic_rung": "S3",
    "dropout": 0.2,
    "temperature": 0.07,
    "embedding_dim": 1536,
}

PROBE_SEED = 20260907
PROBE_QUERIES = 3
PROBE_CANDIDATES = 11

VERDICT_PERMITTED = "S3_REUSE_PERMITTED"
VERDICT_FORBIDDEN = "S3_REUSE_FORBIDDEN"


def load_baseline_module(commit: str):
    """Import the pre-injection m1a_screen as a live module, straight from git.

    Loaded under a name inside the ``mp_retrieval`` package so its relative
    imports resolve against the real package: the baseline scorer must be built
    on the same SemanticHead the current one uses, or this would be comparing
    two semantic modules rather than two scorers.
    """

    source = subprocess.run(
        ["git", "show", f"{commit}:{BASELINE_SOURCE}"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout

    import mp_retrieval  # noqa: F401  (the parent package must already be live)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "m1a_screen_baseline.py"
        path.write_text(source, encoding="utf-8")
        name = "mp_retrieval._m2b_s3_baseline"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "mp_retrieval"
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module, source


def build(scorer_class, **overrides) -> torch.nn.Module:
    torch.manual_seed(PROBE_SEED)
    model = scorer_class(**{**FROZEN, **overrides})
    model.eval()
    return model


def capture(model: torch.nn.Module, nodes, queries, batch_index, structural):
    """Forward once, capturing what the scorer's first layer really receives."""

    seen: dict[str, torch.Tensor] = {}

    def hook(_module, inputs, _output):
        seen["scorer_input"] = inputs[0].detach().clone()

    handle = model.scorer[0].register_forward_hook(hook)
    try:
        with torch.no_grad():
            scores = model.forward_explicit(nodes, queries, batch_index, structural)
    finally:
        handle.remove()

    with torch.no_grad():
        semantic = torch.cat(
            [model.semantic_head(queries[position], nodes[batch_index == position])
             for position in range(queries.shape[0])],
            dim=0,
        )
    return scores.detach(), seen["scorer_input"], semantic


def probe_inputs(dim: int, precomputed_width: int):
    generator = torch.Generator().manual_seed(PROBE_SEED)
    batch_index = torch.repeat_interleave(torch.arange(PROBE_QUERIES), PROBE_CANDIDATES)
    nodes = torch.randn(PROBE_QUERIES * PROBE_CANDIDATES, dim, generator=generator)
    queries = torch.randn(PROBE_QUERIES, dim, generator=generator)
    structural = torch.randn(
        PROBE_QUERIES * PROBE_CANDIDATES, precomputed_width, generator=generator
    )
    return nodes, queries, batch_index, structural


def compare(baseline: torch.nn.Module, current: torch.nn.Module) -> dict[str, Any]:
    """Every equality the reuse rests on, reported one by one."""

    old_state, new_state = baseline.state_dict(), current.state_dict()
    keys_match = list(old_state) == list(new_state)
    shapes_match = keys_match and all(
        old_state[key].shape == new_state[key].shape for key in old_state
    )
    dtypes_match = keys_match and all(
        old_state[key].dtype == new_state[key].dtype for key in old_state
    )
    # Same seed, same construction order -> the same RNG draws. Stronger than
    # shape equality: it catches a reordered or extra parameter allocation that
    # happens to preserve every shape.
    same_init = keys_match and shapes_match and all(
        torch.equal(old_state[key], new_state[key]) for key in old_state
    )

    # From here both models hold the baseline's weights, so any output
    # difference is a difference in the function, not in initialisation.
    current.load_state_dict(old_state, strict=True)

    nodes, queries, batch_index, structural = probe_inputs(
        FROZEN["embedding_dim"], FROZEN["precomputed_width"]
    )
    old_scores, old_input, old_semantic = capture(
        baseline, nodes, queries, batch_index, structural
    )
    new_scores, new_input, new_semantic = capture(
        current, nodes, queries, batch_index, structural
    )

    return {
        "state_dict_keys": sorted(old_state),
        "state_dict_keys_identical": keys_match,
        "parameter_shapes_identical": shapes_match,
        "parameter_dtypes_identical": dtypes_match,
        "same_seed_yields_identical_weights": same_init,
        "parameter_count_identical": (
            sum(p.numel() for p in baseline.parameters())
            == sum(p.numel() for p in current.parameters())
        ),
        "parameter_count": sum(p.numel() for p in current.parameters()),
        "semantic_output_identical": torch.equal(old_semantic, new_semantic),
        "semantic_output_shape": list(new_semantic.shape),
        "scorer_input_identical": torch.equal(old_input, new_input),
        "scorer_input_shape": list(new_input.shape),
        "scorer_input_captured_by": "forward hook on scorer[0], not recomputed",
        "forward_scores_identical": torch.equal(old_scores, new_scores),
        "max_absolute_score_difference": float((old_scores - new_scores).abs().max()),
        "scores_shape": list(new_scores.shape),
    }


def extract_state(payload: Any) -> dict[str, torch.Tensor]:
    """Find the weights inside whatever shape the checkpoint was saved in."""

    if not isinstance(payload, dict):
        raise SystemExit(f"checkpoint is a {type(payload).__name__}, not a mapping")
    for key in ("model_state_dict", "state_dict", "model"):
        if key in payload and isinstance(payload[key], dict):
            return payload[key]
    if all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload
    raise SystemExit(
        f"cannot find a state dict in a checkpoint with keys {sorted(payload)}"
    )


def check_checkpoint(path: Path, baseline_class) -> dict[str, Any]:
    """Load real trained M2 weights into the current implementation.

    ``strict=True`` on purpose: a silently ignored or silently missing key is
    exactly the failure that would let a reused number come from a different
    model than the one that produced it.
    """

    state = extract_state(torch.load(path, map_location="cpu", weights_only=True))

    current, baseline = build(M1AScorer), build(baseline_class)
    current.load_state_dict(state, strict=True)
    baseline.load_state_dict(state, strict=True)

    nodes, queries, batch_index, structural = probe_inputs(
        FROZEN["embedding_dim"], FROZEN["precomputed_width"]
    )
    old_scores, old_input, old_semantic = capture(
        baseline, nodes, queries, batch_index, structural
    )
    new_scores, new_input, new_semantic = capture(
        current, nodes, queries, batch_index, structural
    )

    return {
        "checkpoint": path.name,
        "loaded_strict": True,
        "trained_parameter_count": sum(int(value.numel()) for value in state.values()),
        "keys": sorted(state),
        "semantic_output_identical": torch.equal(old_semantic, new_semantic),
        "scorer_input_identical": torch.equal(old_input, new_input),
        "forward_scores_identical": torch.equal(old_scores, new_scores),
        "max_absolute_score_difference": float((old_scores - new_scores).abs().max()),
        "why_this_is_the_check_that_matters": (
            "These are the weights one of the 14 reused fits actually produced. Every other "
            "equality here uses weights this proof invented."
        ),
    }


def build_report(commit: str, checkpoint: Path | None) -> dict[str, Any]:
    baseline_module, _source = load_baseline_module(commit)
    baseline_class = baseline_module.M1AScorer

    # Asked of the live class, not of the source text: "semantic_head" also
    # appears in the baseline as the attribute assignment, so a string match
    # says nothing about whether the parameter exists.
    injection_absent = "semantic_head" not in inspect.signature(
        baseline_class.__init__
    ).parameters

    equalities = compare(build(baseline_class), build(M1AScorer))

    required = [
        "state_dict_keys_identical", "parameter_shapes_identical",
        "parameter_dtypes_identical", "same_seed_yields_identical_weights",
        "parameter_count_identical", "semantic_output_identical",
        "scorer_input_identical", "forward_scores_identical",
    ]
    failures = [name for name in required if not equalities[name]]
    if not injection_absent:
        failures.append("baseline_predates_the_injection")

    report: dict[str, Any] = {
        "status": "M2B_S3_REUSE_PROOF_COMPLETE",
        "question": (
            "Is an S3 model built by the current code the same function as the S3 model that "
            "produced M2's 14 reused fits?"
        ),
        "baseline_commit": commit,
        "baseline_source": BASELINE_SOURCE,
        "baseline_predates_the_injection": injection_absent,
        "frozen_construction": FROZEN,
        "probe": {
            "seed": PROBE_SEED,
            "queries": PROBE_QUERIES,
            "candidates_per_query": PROBE_CANDIDATES,
            "why_multi_query": (
                "forward_explicit loops over batch positions and concatenates, so a "
                "single-query probe would never exercise that concatenation."
            ),
        },
        "equalities": equalities,
    }

    if checkpoint is not None:
        if not checkpoint.is_file():
            raise SystemExit(f"{checkpoint} does not exist")
        trained = check_checkpoint(checkpoint, baseline_class)
        report["trained_checkpoint"] = trained
        failures += [
            f"trained_checkpoint.{name}"
            for name in ("semantic_output_identical", "scorer_input_identical",
                         "forward_scores_identical")
            if not trained[name]
        ]
    else:
        report["trained_checkpoint"] = {
            "checked": False,
            "why_not": (
                "No M2 checkpoint was supplied. The equalities above are proved on invented "
                "weights, which establishes the function is the same but never touches the "
                "actual reused numbers. Pass --checkpoint to close that gap."
            ),
        }

    report["failed_checks"] = failures
    report["verdict"] = VERDICT_PERMITTED if not failures else VERDICT_FORBIDDEN
    report["what_the_verdict_means"] = (
        "The 14 M2 S3 fits may stand as M2B's S3 row; new fits = 28."
        if not failures else
        "The 14 M2 S3 fits may NOT be reused. M2B's workload becomes 42 new fits and the "
        "compute estimate must be recomputed before anything launches."
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=M2_FIT_COMMIT)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="a real M2 S3 checkpoint.pt pulled from the volume")
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    report = build_report(args.commit, args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == VERDICT_PERMITTED else 1


if __name__ == "__main__":
    raise SystemExit(main())
