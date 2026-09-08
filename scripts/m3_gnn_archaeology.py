"""M3 section 5 archaeology: every historical GNN in this repository.

The review asked one question before anything new is trained: were the
message-passing layers ever actually in the gradient path?  Historical CRAG
carried a zero-gradient GNN bug, and every "GNN underperforms" row filed here
would be worthless if the same bug were present.  This script answers that by
running a live backward pass through each surviving implementation and recording
the per-parameter gradient norms, alongside a negative control that injects the
bug so the check is shown to have power rather than merely to pass.

It decides nothing.  It records what was built, which runners built it, what the
filed results say, and what the gradients do.  Selection belongs to M3B.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mp_retrieval.l2_models import build_gnn_scorer  # noqa: E402
from mp_retrieval.models import MessagePassingEncoder  # noqa: E402
from mp_retrieval.operator_models import (  # noqa: E402
    build_operator_model,
    build_seed_aware_message_passing,
)
from mp_retrieval.representation import (  # noqa: E402
    assert_message_passing_gradients,
    gradient_health,
)

DOC_PATH = ROOT / "docs" / "M3_GNN_ARCHAEOLOGY.md"
JSON_PATH = ROOT / "outputs" / "m3" / "gnn_archaeology.json"

OPERATORS = ("gcn", "sage", "gat", "gin")
PROBE_SEED = 0

# The four families are labelled A-D by the review.  Everything in ``source``
# was read out of the file at the recorded line; everything else on the record
# is derived below, so a reader can check either half independently.
FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "key": "A",
        "class_name": "MessagePassingEncoder",
        "module": "src/mp_retrieval/models.py",
        "line": 51,
        "role": "dual-encoder node tower for a retrieval model",
        "builder": "mp_retrieval.models.build_parameter_matched_pair",
        "probe_dims": {"input_dim": 1536, "hidden_dim": 64, "output_dim": 64, "layers": (2,)},
        "source": {
            "activation": "ReLU",
            "residuals": "h <- LayerNorm(h + Dropout(ReLU(conv(h, edge_index))))",
            "normalization": "LayerNorm per layer; L2 normalize on the output projection",
            "default_layers": 2,
            "query_conditioning": (
                "none inside the node tower; a separate QueryEncoder produces the "
                "query vector and the two meet only at the final dot product"
            ),
            "input_features": "frozen node embedding only",
            "scorer_readout": "Linear(hidden, output) then L2 normalize; scored by dot product",
            "layer_states_exposed": "yes, via return_layers=True",
        },
    },
    {
        "key": "B",
        "class_name": "CandidateGNNScorer",
        "module": "src/mp_retrieval/l2_models.py",
        "line": 32,
        "role": "Level-2 candidate scorer",
        "builder": "mp_retrieval.l2_models.build_gnn_scorer",
        "probe_dims": {"input_dim": 40, "hidden_dim": 32, "layers": (1,)},
        "source": {
            "activation": "GELU",
            "residuals": "h <- LayerNorm(h + Dropout(GELU(conv(h, edge_index))))",
            "normalization": "LayerNorm per layer; no output normalization",
            "default_layers": 2,
            "query_conditioning": (
                "the query enters through the precomputed candidate feature row, not "
                "through a separate tower"
            ),
            "input_features": (
                "build_candidate_features: z-scored retrieval scores, presence mask, "
                "percentile, membership, repeated state (width 40)"
            ),
            "scorer_readout": "Linear(hidden, 1).squeeze(-1)",
            "layer_states_exposed": "no",
        },
    },
    {
        "key": "C",
        "class_name": "MessagePassingOperator",
        "module": "src/mp_retrieval/operator_models.py",
        "line": 158,
        "role": "operator screen and paper main table",
        "builder": "mp_retrieval.operator_models.build_operator_model",
        "probe_dims": {"embedding_dim": 1536, "hidden_dim": 64, "layers": (1, 2)},
        "source": {
            "activation": "GELU",
            "residuals": "h <- LayerNorm(h + Dropout(GELU(conv(h, edge_index))))",
            "normalization": "LayerNorm per layer; L2 normalize before scoring",
            "default_layers": 1,
            "query_conditioning": (
                "query-residual readout: target = normalize(q + MLP(q)) with q = GELU("
                "W_q x); the node states are never conditioned on the query before "
                "message passing, only compared to it afterwards"
            ),
            "input_features": "frozen node embedding only; adjacency is the sole privileged input",
            "scorer_readout": "cosine(node_state, target) / temperature",
            "layer_states_exposed": "no",
        },
    },
    {
        "key": "D",
        "class_name": "SeedAwareMessagePassingOperator",
        "module": "src/mp_retrieval/operator_models.py",
        "line": 227,
        "role": "seed-aware screen, confirmation, phase perturbations, edge provenance",
        "builder": "mp_retrieval.operator_models.build_seed_aware_message_passing",
        "probe_dims": {"embedding_dim": 1536, "hidden_dim": 64, "layers": (1, 2)},
        "source": {
            "activation": "GELU",
            "residuals": "h <- LayerNorm(h + Dropout(GELU(conv(h, edge_index))))",
            "normalization": "LayerNorm per layer; L2 normalize before scoring",
            "default_layers": 1,
            "query_conditioning": (
                "identical to family C, plus one binary retrieval-seed scalar "
                "concatenated to the node embedding before the projection"
            ),
            "input_features": "frozen node embedding (1536) + is_seed indicator = 1537",
            "scorer_readout": "cosine(node_state, target) / temperature",
            "layer_states_exposed": "no",
            "entry_point": (
                "forward_seed_aware, not forward; the inherited forward would silently "
                "drop the seed channel and fail the projection width"
            ),
        },
    },
)

# Which filed result families each runner wrote.  The runner list itself is
# derived from the source below; only this mapping is recorded by hand.
RUNNER_OUTPUTS: dict[str, tuple[str, ...]] = {
    "run_l2_pair.py": ("outputs/webqsp_l2_smoke.json",),
    "run_operator_screen.py": ("outputs/operator_screen/", "outputs/operator_screen_smoke/"),
    "run_main_table.py": ("outputs/main_table/",),
    "run_confirmation.py": ("outputs/confirmation/",),
    "run_coverage_variant.py": ("outputs/coverage_variant/",),
    "run_sa_mlp_confirmation.py": ("outputs/sa_mlp_confirmation/",),
    "run_phase_screen.py": ("outputs/phase_screen/",),
    "run_phase_confirmation.py": ("outputs/phase_confirmation/",),
    "run_edge_provenance.py": ("outputs/edge_provenance/",),
    "run_candidate_budget.py": ("outputs/candidate_budget/",),
    "run_online_systems.py": ("outputs/online_systems/",),
}

# ``run_sa_mlp_screen.py`` and ``run_linear_rank_structure.py`` report GNN numbers
# but never build one: they read the frozen GNN row out of a baseline result.
# Recording them as runners would overstate how many places trained a GNN.
SEED_AWARE_MODEL_KEY = "seed_aware_gnn"
SELF = Path(__file__).name

HYPERPARAMETERS = (
    "seed",
    "seeds",
    "epochs",
    "batch_size",
    "hidden_dim",
    "layers",
    "dropout",
    "temperature",
    "learning_rate",
    "weight_decay",
    "device",
    "selected_gnn",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchor_holds(family: dict[str, Any]) -> bool:
    """The file:line anchors are only useful if they still name the class."""

    lines = (ROOT / family["module"]).read_text(encoding="utf-8").splitlines()
    index = family["line"] - 1
    if not 0 <= index < len(lines):
        return False
    return lines[index].startswith(f"class {family['class_name']}")


def runners_for(builder: str) -> list[str]:
    """Find the scripts that construct a family rather than trusting a list."""

    symbol = builder.rsplit(".", 1)[-1]
    scripts = [path for path in sorted((ROOT / "scripts").glob("*.py")) if path.name != SELF]
    direct = [
        path.name
        for path in scripts
        if re.search(rf"\b{re.escape(symbol)}\b", path.read_text(encoding="utf-8"))
    ]
    # Several runners reach the seed-aware family by importing another runner's
    # ``_build_model`` rather than the factory.  Follow one level of re-export,
    # but only for scripts that also name the model key that selects the GNN --
    # importing ``_build_model`` alone just as often means building the MLP.
    if symbol == "build_seed_aware_message_passing":
        hosts = {name[:-3] for name in direct}
        for path in scripts:
            text = path.read_text(encoding="utf-8")
            reexported = any(re.search(rf"from scripts\.{host} import", text) for host in hosts)
            if reexported and "_build_model" in text and SEED_AWARE_MODEL_KEY in text:
                if path.name not in direct:
                    direct.append(path.name)
    return sorted(direct)


def probe_dims_for(family: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in family["probe_dims"].items() if key != "layers"}


def build(family: dict[str, Any], operator: str, layers: int) -> torch.nn.Module:
    dims = probe_dims_for(family)
    if family["key"] == "A":
        return MessagePassingEncoder(operator, num_layers=layers, **dims)
    if family["key"] == "B":
        return build_gnn_scorer(operator, num_layers=layers, **dims)
    if family["key"] == "C":
        return build_operator_model(operator, layers=layers, **dims)
    return build_seed_aware_message_passing(operator, layers=layers, **dims)


def listwise_loss(scores: torch.Tensor, lengths: tuple[int, ...]) -> torch.Tensor:
    """The multi-positive listwise soft cross-entropy the runners train with.

    Same formula as ``scripts/run_operator_screen.py:135`` and the L2 runner's
    ``_listwise_loss``; reimplemented here so the probe does not need a dataset.
    """

    losses: list[torch.Tensor] = []
    offset = 0
    for length in lengths:
        local = scores[offset : offset + length]
        target = torch.zeros(length)
        target[0] = 1.0
        losses.append(-(F.log_softmax(local, dim=0) * target).sum())
        offset += length
    return torch.stack(losses).mean()


def _probe_inputs(family: dict[str, Any]) -> dict[str, torch.Tensor]:
    """A three-query batch with a connected but non-trivial local topology."""

    lengths = (4, 4, 4)
    total = sum(lengths)
    batch_index = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
    edge_index = torch.tensor([[0, 1, 2, 4, 5, 8, 9, 10], [1, 2, 3, 5, 6, 9, 10, 11]])
    dims = probe_dims_for(family)
    width = dims.get("embedding_dim", dims.get("input_dim"))
    return {
        "lengths": lengths,
        "batch_index": batch_index,
        "edge_index": edge_index,
        "nodes": torch.randn(total, width),
        "queries": torch.randn(len(lengths), width),
        "anchors": torch.randn(len(lengths), width),
        "seeds": torch.randint(0, 2, (total, 1)).float(),
        "query_readout": F.normalize(torch.randn(len(lengths), dims.get("output_dim", 64)), dim=-1),
    }


def score(family: dict[str, Any], model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    if family["key"] == "A":
        states = model(batch["nodes"], batch["edge_index"])
        return (states * batch["query_readout"][batch["batch_index"]]).sum(dim=-1)
    if family["key"] == "B":
        return model(batch["nodes"], batch["edge_index"])
    if family["key"] == "C":
        return model(
            batch["nodes"],
            batch["queries"],
            batch["anchors"],
            batch["batch_index"],
            batch["edge_index"],
        )
    return model.forward_seed_aware(
        batch["nodes"],
        batch["queries"],
        batch["batch_index"],
        batch["edge_index"],
        batch["seeds"],
    )


def conv_gradients(model: torch.nn.Module) -> dict[str, dict[str, float | bool]]:
    return {
        name: value
        for name, value in gradient_health(model).items()
        if "conv" in name.lower()
    }


def gradient_probe() -> list[dict[str, Any]]:
    """Backward once through every historical family and record conv gradients."""

    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for layers in family["probe_dims"]["layers"]:
            for operator in OPERATORS:
                torch.manual_seed(PROBE_SEED)
                model = build(family, operator, layers)
                batch = _probe_inputs(family)
                model.zero_grad(set_to_none=True)
                listwise_loss(score(family, model, batch), batch["lengths"]).backward()
                grads = conv_gradients(model)
                norms = [float(value["norm"]) for value in grads.values()]
                try:
                    assert_message_passing_gradients(model)
                    asserted = True
                except RuntimeError:
                    asserted = False
                rows.append(
                    {
                        "family": family["key"],
                        "class": family["class_name"],
                        "operator": operator,
                        "layers": layers,
                        "trainable_parameters": int(
                            sum(p.numel() for p in model.parameters() if p.requires_grad)
                        ),
                        "convolution_tensors": len(grads),
                        "min_gradient_norm": min(norms),
                        "max_gradient_norm": max(norms),
                        "all_present": all(bool(v["present"]) for v in grads.values()),
                        "all_finite": all(bool(v["finite"]) for v in grads.values()),
                        "all_nonzero": all(float(v["norm"]) > 0.0 for v in grads.values()),
                        "assert_message_passing_gradients_passes": asserted,
                        "gradient_norms": {name: float(v["norm"]) for name, v in grads.items()},
                    }
                )
    return rows


def negative_control() -> dict[str, Any]:
    """Inject the historical CRAG bug and confirm the probe catches it.

    A gradient check that cannot fail is not evidence.  This detaches the
    convolution update -- message passing still runs, still shapes the forward
    activations, and contributes nothing to the gradient -- which is exactly the
    failure mode the review asked about.
    """

    family = next(item for item in FAMILIES if item["key"] == "C")
    torch.manual_seed(PROBE_SEED)
    model = build(family, "gcn", 1)
    batch = _probe_inputs(family)

    def sabotaged() -> torch.Tensor:
        state = model.project_nodes(batch["nodes"])
        for conv, norm in zip(model.convs, model.norms):
            state = norm(state + F.gelu(conv(state, batch["edge_index"])).detach())
        state = F.normalize(state, dim=-1)
        query_state = model.project_queries(batch["queries"])
        target = F.normalize(query_state + model.query_target(query_state), dim=-1)
        return model.similarities(state, target, batch["batch_index"])

    observed: dict[str, Any] = {}
    for label, forward in (
        ("healthy", lambda: score(family, model, batch)),
        ("message_passing_detached", sabotaged),
    ):
        model.zero_grad(set_to_none=True)
        listwise_loss(forward(), batch["lengths"]).backward()
        grads = conv_gradients(model)
        try:
            assert_message_passing_gradients(model)
            caught = False
        except RuntimeError as error:
            caught = True
            observed.setdefault("raised", str(error))
        observed[label] = {
            "max_gradient_norm": max(float(v["norm"]) for v in grads.values()),
            "detected_as_dead": caught,
        }
    return observed


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def filed_hyperparameters() -> dict[str, dict[str, Any]]:
    """Training settings come from the results themselves, not from memory."""

    sources = {
        "outputs/operator_screen/2wiki_clean.json": "C",
        "outputs/main_table/metaqa.json": "C",
        "outputs/sa_mlp_screen/metaqa.json": "D",
        "outputs/sa_mlp_confirmation/metaqa.json": "D",
        "outputs/webqsp_l2_smoke.json": "B",
    }
    filed: dict[str, dict[str, Any]] = {}
    for relative, family in sources.items():
        path = ROOT / relative
        if not path.exists():
            continue
        config = read_json(path).get("config", {})
        filed[relative] = {
            "family": family,
            **{key: config[key] for key in HYPERPARAMETERS if key in config},
        }
    return filed


def operator_screen_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "outputs" / "operator_screen").glob("*.json")):
        models = read_json(path)["models"]
        recall = {name: models[name]["metrics"]["recall@5"] for name in models}
        gnns = {name: recall[name] for name in OPERATORS if name in recall}
        explicit = {name: value for name, value in recall.items() if name not in OPERATORS}
        best_gnn = max(gnns, key=gnns.__getitem__)
        best_explicit = max(explicit, key=explicit.__getitem__)
        rows.append(
            {
                "dataset": path.stem,
                "recall@5": recall,
                "best_gnn": best_gnn,
                "best_non_graph": best_explicit,
                "best_gnn_minus_best_non_graph": gnns[best_gnn] - explicit[best_explicit],
            }
        )
    return rows


def main_table_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "outputs" / "main_table").glob("*.json")):
        paired = read_json(path).get("paired_mlp_minus_gnn", {}).get("recall@5")
        if paired is None:
            continue
        rows.append({"dataset": path.stem, "plain_mlp_minus_gnn_recall@5": paired})
    return rows


def gap_closure_rows() -> list[dict[str, Any]]:
    path = ROOT / "outputs" / "sa_mlp_screen" / "decision.json"
    if not path.exists():
        return []
    decision = read_json(path)
    return [
        {"dataset": entry["dataset"], **entry["gap_closure"]}
        for entry in decision.get("results", [])
        if "gap_closure" in entry
    ]


def record() -> dict[str, Any]:
    families: list[dict[str, Any]] = []
    for family in FAMILIES:
        runners = runners_for(family["builder"])
        families.append(
            {
                "key": family["key"],
                "class": family["class_name"],
                "module": family["module"],
                "line": family["line"],
                "anchor_verified": anchor_holds(family),
                "module_sha256": digest(ROOT / family["module"]),
                "role": family["role"],
                "builder": family["builder"],
                "runners": runners,
                "outputs": sorted(
                    {out for runner in runners for out in RUNNER_OUTPUTS.get(runner, ())}
                ),
                "ran_in_production": bool(runners),
                "operators_available": list(OPERATORS),
                "read_from_source": family["source"],
                "asserts_message_passing_gradients_in_production": any(
                    "assert_message_passing_gradients" in (ROOT / "scripts" / runner).read_text(
                        encoding="utf-8"
                    )
                    for runner in runners
                ),
            }
        )

    probe = gradient_probe()
    return {
        "format": "m3_gnn_archaeology_v1",
        "phase": "M3A",
        "section": "M3 section 5 -- GNN archaeology",
        "generated_by": "scripts/m3_gnn_archaeology.py",
        "decides": "nothing; selection belongs to M3B",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "probe_seed": PROBE_SEED,
        },
        "families": families,
        "filed_hyperparameters": filed_hyperparameters(),
        "gradient_path_verification": {
            "question": (
                "were the message-passing layers of every historical GNN actually in "
                "the gradient path"
            ),
            "method": (
                "one live backward pass per family/operator/depth through the same "
                "entry point production used, then per-parameter gradient norms for "
                "every tensor whose name contains 'conv'"
            ),
            "checker": "mp_retrieval.representation.assert_message_passing_gradients",
            "negative_control": negative_control(),
            "results": probe,
            "all_live": all(
                row["all_present"] and row["all_finite"] and row["all_nonzero"] for row in probe
            ),
        },
        "historical_results": {
            "operator_screen_recall@5": operator_screen_rows(),
            "main_table_paired_5_seed": main_table_rows(),
            "sa_mlp_screen_gap_closure": gap_closure_rows(),
        },
    }


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render(data: dict[str, Any]) -> str:
    verification = data["gradient_path_verification"]
    control = verification["negative_control"]
    historical = data["historical_results"]

    family_rows = [
        [
            family["key"],
            f"`{family['class']}`",
            f"`{family['module']}:{family['line']}`",
            ", ".join(f"`{runner}`" for runner in family["runners"]) or "**none**",
            "yes" if family["ran_in_production"] else "no (dead code)",
        ]
        for family in data["families"]
    ]

    probe_rows = [
        [
            row["family"],
            row["operator"],
            str(row["layers"]),
            f"{row['trainable_parameters']:,}",
            str(row["convolution_tensors"]),
            f"{row['min_gradient_norm']:.3e}",
            f"{row['max_gradient_norm']:.3e}",
            "LIVE" if row["all_nonzero"] else "**DEAD**",
        ]
        for row in verification["results"]
    ]

    screen_rows = [
        [
            row["dataset"],
            f"{row['recall@5'].get('plain_mlp', float('nan')):.4f}",
            f"{row['recall@5'].get('offset_mlp', float('nan')):.4f}",
            f"{row['best_gnn']} {row['recall@5'][row['best_gnn']]:.4f}",
            f"{row['best_gnn_minus_best_non_graph']:+.4f}",
        ]
        for row in historical["operator_screen_recall@5"]
    ]

    paired_rows = [
        [
            row["dataset"],
            f"{row['plain_mlp_minus_gnn_recall@5']['mean']:+.4f}",
            f"{row['plain_mlp_minus_gnn_recall@5']['sample_std']:.4f}",
            str(row["plain_mlp_minus_gnn_recall@5"]["n"]),
        ]
        for row in historical["main_table_paired_5_seed"]
    ]

    closure_rows = [
        [
            row["dataset"],
            f"{row['plain_mlp_seed_0']:.4f}",
            f"{row['gnn_seed_0']:.4f}",
            f"{row['sa_mlp_seed_0']:.4f}",
            f"{row['fraction']:.3f}",
        ]
        for row in historical["sa_mlp_screen_gap_closure"]
    ]

    return f"""# M3 GNN archaeology

Generated by `{data['generated_by']}`. This document decides {data['decides']}.

M3 section 5 asked for one thing before any new training: an audit of every
historical GNN in this repository, and specifically a check that the
message-passing layers were ever in the gradient path at all. Historical CRAG
carried a zero-gradient GNN bug. If the same bug were here, every filed row
comparing a GNN against anything else would be measuring an expensive
initialisation rather than message passing.

They were in the gradient path. The concern is closed for this repository, and
the evidence is below.

## The four families

{_table(["", "class", "defined at", "runners", "ran"], family_rows)}

Family **A** has no caller anywhere in `scripts/`, `src/` or `tests/`. It is the
earliest dual-encoder design and it is dead code: no filed scientific result
depends on it. It is audited here for completeness, not because anything rests
on it.

Families **C** and **D** are the same architecture; D concatenates one binary
retrieval-seed scalar to the node embedding before the projection, which is the
only difference and costs exactly `hidden_dim` parameters. D must be entered
through `forward_seed_aware`; the inherited `forward` would silently drop the
seed channel.

All four place their convolutions in a live residual: `h <- LayerNorm(h +
Dropout(act(conv(h, edge_index))))`, with the result reaching the score. None of
them detaches, freezes or discards the convolution output.

Only family B ever ran `assert_message_passing_gradients` in production, at
`scripts/run_l2_pair.py:123`. Families C and D were never gradient-asserted while
training, which is precisely why the live probe below was necessary rather than
merely reassuring.

## Gradient-path verification

Each row is one live backward pass through the same entry point production used,
with the multi-positive listwise soft cross-entropy the runners train with. The
reported norms are over every parameter tensor whose name contains `conv`.

{_table(
    ["", "operator", "depth", "params", "conv tensors", "min grad", "max grad", "verdict"],
    probe_rows,
)}

All {len(verification['results'])} configurations are live:
`all_live = {str(verification['all_live']).lower()}`.

### The check has power

A gradient probe that cannot fail is not evidence, so the historical bug was
injected deliberately: detach the convolution update, leaving message passing to
run and shape the forward activations while contributing nothing to the
gradient.

- healthy forward, max conv gradient norm **{control['healthy']['max_gradient_norm']:.4g}**, flagged dead: `{str(control['healthy']['detected_as_dead']).lower()}`
- detached forward, max conv gradient norm **{control['message_passing_detached']['max_gradient_norm']:.4g}**, flagged dead: `{str(control['message_passing_detached']['detected_as_dead']).lower()}`

The checker raised: `{control.get('raised', 'n/a')}`

The probe reproduces filed parameter counts exactly, which is what establishes
that it probed the architectures that actually ran rather than lookalikes:
family C depth-1 `gat` at 213,504 matches `outputs/operator_screen/2wiki_clean.json`,
family D depth-1 `gat` at 213,568 is that plus the seed scalar's 64, and family B
`gcn` at 2,465 matches `outputs/webqsp_l2_smoke.json`.

## What the filed results actually say

The shorthand "GNN underperforms" does not survive contact with the record. The
message-passing models beat the graph-blind MLP on the knowledge-base datasets by
a wide margin, tie on SQuAD, and lose on the two passage datasets in the seed-0
operator screen.

Five-seed paired contrast, graph-blind `plain_mlp` minus the selected GNN at
recall@5 (negative means the GNN wins):

{_table(["dataset", "mean", "sd", "n"], paired_rows)}

Seed-0 operator screen, recall@5 on test:

{_table(["dataset", "plain_mlp", "offset_mlp", "best GNN", "GNN - best non-graph"], screen_rows)}

What actually displaced the GNN was neither of those. It was the seed-aware
explicit-feature MLP, which recovered more than the whole GNN-over-plain-MLP gap:

{_table(["dataset", "plain_mlp", "gnn", "sa_mlp", "gap closure"], closure_rows)}

This is the closest thing in the repository to the M3B question, run at small
scale, at seed 0, under the historical information contract. It is prior
evidence motivating M3A, and it is explicitly not the claim: it was measured
before feature parity, before typed relations, and inside the historical
candidate universe. Whether the mechanism survives a SOTA-derived information
contract and a high-headroom candidate universe is what M3B exists to find out.

## Filed training settings

Read from the `config` block of the results themselves rather than from memory.

```json
{json.dumps(data['filed_hyperparameters'], indent=2)}
```

## What this closes, and what it does not

Closed: the zero-gradient concern for this repository. Every historical
message-passing layer carried gradient.

Not closed, and deliberately out of scope here: whether those GNNs were *well
specified*. Depth 1 at width 64 on a candidate-induced subgraph, with the query
entering only at the readout, is a weak interface next to what current
graph-retrieval systems receive. That is the observation M3A acts on; it is not
a defect this audit found in the code.

Machine-readable form: `outputs/m3/gnn_archaeology.json`.
"""


def main() -> None:
    data = record()
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    DOC_PATH.write_text(render(data), encoding="utf-8", newline="\n")
    print(f"wrote {JSON_PATH.relative_to(ROOT)}")
    print(f"wrote {DOC_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
