"""What a persisted feature store IS, as distinct from what authorised it.

M2 identified each persisted cell master by, among seven real fields, the
SHA-256 of the entire declaration YAML. That worked for exactly as long as the
declaration never changed. The moment M2B amended the file to record a
selection -- a statement about authorship and statistics that cannot move a
single float in a structural feature tensor -- ``load_cell_for_fit`` began
refusing all fourteen stores, and the only cheap fix on the table was to pin
the old hash. Pinning would work once and then rot: M3, M4 and the canonical
migration would each hit it again, and each would be tempted to pin again until
the identity check meant nothing.

The distinction this module draws:

    the scientific identity of a feature store
        != the identity of the mutable declaration file that authorised it

So ``feature_build_contract`` hashes only inputs that can change the cell
tensor -- the data, the candidate set, the regime and its context construction,
the A64 budget, the query split, the feature formulas and their constants, the
normalisation, and the stored dtype/schema. Amendments about authorisation,
compute ceilings, statistics or wording do not appear in it, so they cannot
invalidate a store; a changed damping constant or a changed column layout does
appear, and does.

Two hashes are kept side by side, never one instead of the other:

``original_full_config_sha256``
    the whole-file hash of the declaration that authorised the build, carried
    forward verbatim as provenance. Never recomputed, never rewritten.

``feature_build_contract_sha256``
    the contract hash above, which is what governs whether a store may be
    reused.

The formula half of the contract is a version string, and a version string is
worth nothing unless something forces it to move when the formulas move. So
``measure_formula_behaviour`` runs the real ``_cell_master_local`` over a
deterministic toy cell in all three regimes and digests the tensors it emits;
``FORMULA_BEHAVIOUR_SHA256`` pins that digest per version, and
``tests/test_feature_build_contract.py`` fails if the live digest and the pin
disagree. Changing a formula therefore forces a new version, which changes
every contract hash, which correctly invalidates every store built under the
old formulas.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import run_m1a_feature_screen as _m1a  # noqa: E402

#: Bumped only when the SHAPE of the contract changes -- a field added, removed
#: or renamed. Every existing store's hash moves when this moves, which is the
#: intended consequence: a contract with different fields is a different claim.
CONTRACT_VERSION = "feature_build_contract_v1"

#: Bumped when any formula that produces the master block changes. Defended by
#: FORMULA_BEHAVIOUR_SHA256 below, not by discipline.
FEATURE_FORMULA_VERSION = "m1a_master_block_v1"

#: version -> digest of the master blocks the real builder emits on the toy
#: cell, rounded to FORMULA_BEHAVIOUR_DECIMALS. Measured, then pinned.
FORMULA_BEHAVIOUR_SHA256: dict[str, str] = {
    "m1a_master_block_v1": "bca6972317df6c367d98fb6b4371298a102d5d9329eb01a688d2819476bd4708",
}

#: float32 carries ~7 significant decimal digits, and the PPR iteration in
#: qls_local_features goes through BLAS, so the last bit or two of a feature can
#: differ between machines. Rounding here means the pin travels between
#: platforms; it is far tighter than any real formula edit, and the constants
#: a sub-1e-6 edit would move (damping, iterations) are named fields of the
#: contract in their own right.
FORMULA_BEHAVIOUR_DECIMALS = 6

#: How the queries were chosen. Defined here rather than in a runner because
#: the contract is what these strings mean; run_m2_qls_v2_freeze imports them
#: and emits them, so the artifact text and the hashed value cannot diverge.
QUERY_SPLIT = "validation"
QUERY_SELECTION = "deterministic_prefix_of_the_split_order"

#: The regimes the behaviour digest covers. All three, because R2's and R3's
#: context construction decides which rows exist at all.
BEHAVIOUR_REGIMES = ("R1", "R2", "R3")


def formula_identity() -> dict[str, Any]:
    """The formula half of the contract, read from the live modules.

    Read rather than restated: a constant transcribed here could drift away
    from the constant the builder actually uses, and the contract would then
    certify a build it never described.
    """

    return {
        "feature_formula_version": FEATURE_FORMULA_VERSION,
        "feature_builder": "scripts/run_m1a_feature_screen.py::_cell_master_local",
        "context_arm": _m1a.CONTEXT_ARM,
        "feature_damping": float(_m1a.FEATURE_DAMPING),
        "feature_ppr_iterations": int(_m1a.FEATURE_PPR_ITERATIONS),
        # qls_local_features is called with normalisation="candidate"; this is
        # the candidate normalisation the declaration means by "normalization".
        "feature_normalisation": "candidate",
        "master_column_layout": {
            name: [int(span.start), int(span.stop)]
            for name, span in _m1a.MASTER_COLUMNS.items()
        },
        "master_column_count": max(
            int(span.stop) for span in _m1a.MASTER_COLUMNS.values()
        ),
    }


def feature_build_contract(
    *,
    dataset: str,
    data_fingerprint_sha256: str,
    regime: str,
    queries: int,
    query_split: str,
    query_selection: str,
    candidate_contract_sha256: str,
    candidate_id_order_sha256: str,
    per_seed_cap: int,
    neighbour_scan_cap_per_seed: int,
    a64_mainline_family: str | None,
    store_format: str,
    master_columns: int,
    master_dtype: str,
) -> dict[str, Any]:
    """Everything that can change the cell tensor, and nothing else.

    ``a64_mainline_family`` is recorded as null outside R3 rather than omitted,
    matching ``cell_build_key``: a field that appears and disappears would make
    two contracts differ for a reason that is not scientific.
    """

    if regime not in BEHAVIOUR_REGIMES:
        raise ValueError(f"unknown regime {regime!r}; declared regimes are {BEHAVIOUR_REGIMES}")
    contract: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        # --- the data ---
        "dataset": dataset,
        "data_fingerprint_sha256": data_fingerprint_sha256,
        # --- the candidate set ---
        "candidate_contract_sha256": candidate_contract_sha256,
        "candidate_id_order_sha256": candidate_id_order_sha256,
        # --- which queries, in which order ---
        "queries": int(queries),
        "query_split": query_split,
        "query_selection": query_selection,
        # --- the regime and its context construction ---
        "regime": regime,
        "per_seed_cap": int(per_seed_cap),
        "neighbour_scan_cap_per_seed": int(neighbour_scan_cap_per_seed),
        "a64_mainline_family": a64_mainline_family if regime == "R3" else None,
        # --- what is written, and how ---
        "store_format": store_format,
        "master_columns": int(master_columns),
        "master_dtype": master_dtype,
    }
    contract.update(formula_identity())
    return contract


def contract_sha256(contract: dict[str, Any]) -> str:
    """Canonical JSON, so key order in the caller can never change the hash."""

    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def contract_from_runner_args(
    args: Any,
    regime: str,
    *,
    candidate_contract_sha256: str,
    candidate_id_order_sha256: str,
    store_format: str,
    master_columns: int,
    master_dtype: str,
) -> dict[str, Any]:
    """The contract a build running RIGHT NOW would write for this cell.

    The forward direction. ``reconstruct_m2_contract`` is the backward one, off
    a store persisted before contracts existed, and the two agreeing on all
    fourteen cells is what licenses reusing those stores.

    The two candidate hashes are parameters rather than fields of ``args``
    because they are measurements of the data, produced by
    ``validate_candidate_contract`` against the real corpus. A container that
    has the data passes what it just measured; a local audit that does not
    passes what the build recorded, and says so.
    """

    return feature_build_contract(
        dataset=args.dataset,
        data_fingerprint_sha256=args.data_fingerprint_sha256,
        regime=regime,
        queries=int(args.queries),
        query_split=QUERY_SPLIT,
        query_selection=QUERY_SELECTION,
        candidate_contract_sha256=candidate_contract_sha256,
        candidate_id_order_sha256=candidate_id_order_sha256,
        per_seed_cap=int(args.per_seed_cap),
        neighbour_scan_cap_per_seed=int(args.neighbour_scan_cap_per_seed),
        a64_mainline_family=args.a64_mainline_family,
        store_format=store_format,
        master_columns=master_columns,
        master_dtype=master_dtype,
    )


# --- defending the formula version -------------------------------------------


def measure_formula_behaviour(fixture: dict[str, Any], module: Any = _m1a) -> dict[str, str]:
    """Digest the tensors the real builder emits on a deterministic toy cell.

    Both digests are returned. ``rounded`` is what gets pinned, because it
    survives a different BLAS; ``exact`` is recorded so a bit-level difference
    is still visible to anyone investigating one.
    """

    rounded = hashlib.sha256()
    exact = hashlib.sha256()
    for regime in BEHAVIOUR_REGIMES:
        scored_sets, master_blocks, _latencies = module._cell_master_local(
            regime=regime,
            views=fixture["views"],
            queries=fixture["queries"],
            dense=fixture["dense"],
            splade=fixture["splade"],
            rowptr=fixture["rowptr"],
            col=fixture["col"],
            num_nodes=fixture["num_nodes"],
            operators=fixture["operators"],
            family_rowptr=fixture["family_rowptr"] if regime == "R3" else None,
            family_col=fixture["family_col"] if regime == "R3" else None,
            node_embeddings=fixture["node_embeddings"],
            budget=fixture["budget"],
        )
        for digest in (rounded, exact):
            digest.update(regime.encode("utf-8"))
        for scored, master in zip(scored_sets, master_blocks, strict=True):
            # Candidate ids are integers and cannot drift, so both digests take
            # them exactly: a changed candidate set is a changed formula.
            ids = np.ascontiguousarray(scored, dtype=np.int64).tobytes()
            values = np.ascontiguousarray(master, dtype=np.float32)
            rounded.update(ids)
            rounded.update(
                np.ascontiguousarray(
                    np.round(values, FORMULA_BEHAVIOUR_DECIMALS), dtype=np.float32
                ).tobytes()
            )
            exact.update(ids)
            exact.update(values.tobytes())
    return {"rounded": rounded.hexdigest(), "exact": exact.hexdigest()}


def pinned_formula_behaviour(version: str = FEATURE_FORMULA_VERSION) -> str:
    pinned = FORMULA_BEHAVIOUR_SHA256.get(version)
    if pinned is None:
        raise KeyError(
            f"no behaviour digest is pinned for feature formula version {version!r}; "
            "measure it with measure_formula_behaviour and add it to "
            "FORMULA_BEHAVIOUR_SHA256 in the same commit that changed the formulas"
        )
    return pinned


# --- reconstructing the contract for stores built before it existed -----------


def _module_constant(commit: str, relpath: str, name: str) -> Any:
    """One module-level literal assignment, read out of git without executing it."""

    source = subprocess.run(
        ["git", "show", f"{commit}:{relpath}"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    for node in ast.parse(source).body:
        targets = (
            node.targets if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign) and node.value
            else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise KeyError(f"{relpath} at {commit} has no module-level literal {name!r}")


#: Where each live formula constant was defined at build time, so "unchanged
#: since the build" is checked against the source that ran rather than assumed.
FORMULA_CONSTANT_SOURCES: dict[str, tuple[str, str]] = {
    "CONTEXT_ARM": ("scripts/run_m0b_regime_map.py", "CONTEXT_ARM"),
    "FEATURE_DAMPING": ("scripts/run_m0b_webqsp_probe.py", "FEATURE_DAMPING"),
    "FEATURE_PPR_ITERATIONS": ("scripts/run_m0b_webqsp_probe.py", "FEATURE_PPR_ITERATIONS"),
    "MASTER_COLUMNS": ("scripts/run_m1a_feature_screen.py", "MASTER_COLUMNS"),
}


def formula_constants_unchanged_since(commit: str) -> dict[str, Any]:
    """Were the formula constants at ``commit`` the ones in the tree now?

    Reconstructing an old store's contract uses today's formula identity for
    the fields the store never recorded. That substitution is only honest if
    those values were the same when the store was built, so this checks each
    one against the source at the build commit instead of assuming it.

    ``MASTER_COLUMNS`` is compared through the same ``[start, stop]``
    normalisation the contract uses, since a slice object is not literal-eval
    friendly and its repr is not a stable comparison surface.
    """

    live = {
        "CONTEXT_ARM": _m1a.CONTEXT_ARM,
        "FEATURE_DAMPING": float(_m1a.FEATURE_DAMPING),
        "FEATURE_PPR_ITERATIONS": int(_m1a.FEATURE_PPR_ITERATIONS),
        "MASTER_COLUMNS": formula_identity()["master_column_layout"],
    }
    per_constant: dict[str, Any] = {}
    for name, (relpath, symbol) in FORMULA_CONSTANT_SOURCES.items():
        if name == "MASTER_COLUMNS":
            source = subprocess.run(
                ["git", "show", f"{commit}:{relpath}"],
                cwd=REPO_ROOT, capture_output=True, text=True, check=True,
            ).stdout
            historical = _master_columns_from_source(source)
        else:
            historical = _module_constant(commit, relpath, symbol)
            if isinstance(live[name], float):
                historical = float(historical)
            elif isinstance(live[name], int):
                historical = int(historical)
        per_constant[name] = {
            "defined_in": relpath,
            "at_build_commit": historical,
            "in_the_tree_now": live[name],
            "unchanged": historical == live[name],
        }
    return {
        "build_commit": commit,
        "per_constant": per_constant,
        "all_unchanged": all(entry["unchanged"] for entry in per_constant.values()),
    }


def _master_columns_from_source(source: str) -> dict[str, list[int]]:
    """``MASTER_COLUMNS`` out of a source string, as ``{name: [start, stop]}``.

    Its values are ``slice(...)`` calls, which ``ast.literal_eval`` refuses, so
    the two integer arguments are read off the call node directly.
    """

    for node in ast.parse(source).body:
        targets = (
            node.targets if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign) and node.value
            else []
        )
        if not any(isinstance(t, ast.Name) and t.id == "MASTER_COLUMNS" for t in targets):
            continue
        layout: dict[str, list[int]] = {}
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            call = value
            if not (isinstance(call, ast.Call) and getattr(call.func, "id", None) == "slice"):
                raise ValueError(f"MASTER_COLUMNS[{key!r}] is not a slice(...) call")
            layout[ast.literal_eval(key)] = [ast.literal_eval(arg) for arg in call.args]
        return layout
    raise KeyError("no module-level MASTER_COLUMNS assignment")


def reconstruct_m2_contract(
    *, store_metadata: dict[str, Any], build_artifact: dict[str, Any]
) -> dict[str, Any]:
    """The contract for a store persisted before contracts existed.

    Every field comes from what that build actually recorded -- the store's own
    ``build_key`` and dtype/schema block, and the build artifact's candidate
    contract and split -- except the formula identity, which the build never
    wrote down and which ``formula_constants_unchanged_since`` checks against
    the build commit separately.

    The store's ``config_sha256`` is read but not reused: it is provenance, and
    ``m2_store_identity`` carries it forward under its own name. Nothing here
    writes back to the store.
    """

    build_key = store_metadata["build_key"]
    provenance = build_artifact["provenance"]
    if build_key["dataset"] != build_artifact["dataset"]:
        raise ValueError(
            f"store says dataset {build_key['dataset']!r}, artifact says "
            f"{build_artifact['dataset']!r} -- these are not the same build"
        )
    if build_key["data_fingerprint_sha256"] != provenance["dataset_fingerprint_sha256"]:
        raise ValueError(
            "store and artifact disagree about the dataset fingerprint; refusing to "
            "reconstruct a contract from two different builds"
        )
    return feature_build_contract(
        dataset=build_key["dataset"],
        data_fingerprint_sha256=build_key["data_fingerprint_sha256"],
        regime=build_key["regime"],
        queries=build_key["queries"],
        query_split=build_artifact["split"],
        query_selection=build_artifact["selection"],
        candidate_contract_sha256=provenance["candidate_contract_sha256"],
        candidate_id_order_sha256=provenance["candidate_id_order_sha256"],
        per_seed_cap=build_key["per_seed_cap"],
        neighbour_scan_cap_per_seed=build_key["neighbour_scan_cap_per_seed"],
        a64_mainline_family=build_key["a64_mainline_family"],
        store_format=store_metadata["format"],
        master_columns=store_metadata["master_columns"],
        master_dtype=store_metadata["master_dtype"],
    )


#: The fields of M2's ``cell_build_key`` that can change the cell tensor. The
#: eighth, ``config_sha256``, is the paperwork hash this module exists to
#: demote; it is still read, still reported, and never compared.
SCIENTIFIC_BUILD_KEY_FIELDS = (
    "dataset",
    "data_fingerprint_sha256",
    "regime",
    "queries",
    "per_seed_cap",
    "neighbour_scan_cap_per_seed",
    "a64_mainline_family",
)


def verify_store_identity(
    *,
    store_metadata: dict[str, Any],
    expected_build_key: dict[str, Any],
    forward_contract: dict[str, Any],
) -> dict[str, Any]:
    """May a fit load this persisted store? The decision, with its reasons.

    Three regimes of evidence, and the caller is told which one it got:

    ``recorded``
        The store was built after contracts existed and wrote its own
        ``feature_build_contract_sha256``. That value is compared against the
        contract the loading process just built from its own arguments and its
        own measurement of the candidate set. A real equality.

    ``reconstructed``
        A store from before contracts existed, such as M2's fourteen. Its
        contract is rebuilt from what it did record plus today's formula
        identity, so comparing that against ``forward_contract`` would be
        circular and is not claimed as evidence. What carries the load instead
        is the scientific build key matching field by field, and
        ``formula_constants_unchanged_since`` proving -- against the source at
        the store's own build commit -- that the formula identity substituted
        into the reconstruction is the one that actually ran.

    ``refused``
        Anything else.

    The store's ``config_sha256`` is reported as drift, never as a refusal.
    That is the whole point: an amendment about authorisation or statistics
    cannot invalidate a structural tensor.
    """

    recorded_key = store_metadata.get("build_key") or {}
    key_differences = sorted(
        field for field in SCIENTIFIC_BUILD_KEY_FIELDS
        if recorded_key.get(field) != expected_build_key.get(field)
    )

    recorded_contract = store_metadata.get("feature_build_contract_sha256")
    forward_sha = contract_sha256(forward_contract)
    evidence = "recorded" if recorded_contract else "reconstructed"

    constants: dict[str, Any] | None = None
    if evidence == "reconstructed":
        commit = store_metadata.get("source_commit")
        if not commit:
            return {
                "admitted": False,
                "evidence": "refused",
                "why": (
                    "the store records neither a feature_build_contract_sha256 nor a "
                    "source_commit, so there is no way to establish which formulas built it"
                ),
                "scientific_key_differences": key_differences,
            }
        constants = formula_constants_unchanged_since(commit)

    contract_matches = (
        recorded_contract == forward_sha if evidence == "recorded" else True
    )
    constants_ok = constants is None or constants["all_unchanged"]
    admitted = not key_differences and contract_matches and constants_ok

    reasons = []
    if key_differences:
        reasons.append(
            f"the persisted store was built for a different cell -- {key_differences} differ"
        )
    if not contract_matches:
        reasons.append(
            f"the store's recorded feature build contract {recorded_contract} is not the "
            f"contract this process would build ({forward_sha})"
        )
    if not constants_ok:
        changed = sorted(
            name for name, entry in constants["per_constant"].items()
            if not entry["unchanged"]
        )
        reasons.append(
            f"{changed} changed since the store was built at "
            f"{store_metadata.get('source_commit')}, so its tensors were produced by "
            "formulas this process no longer implements"
        )

    return {
        "admitted": admitted,
        "evidence": evidence if admitted else "refused",
        "why": "; ".join(reasons) if reasons else "",
        "scientific_key_differences": key_differences,
        "feature_build_contract_sha256": forward_sha,
        "recorded_feature_build_contract_sha256": recorded_contract,
        "original_full_config_sha256": recorded_key.get("config_sha256"),
        "declaration_hash_drift_is_not_a_refusal": (
            "The store's config_sha256 is carried forward as provenance. It is not "
            "compared, because the declaration it hashes can be amended for reasons "
            "that cannot move a float in a structural feature tensor."
        ),
        "formula_constants": constants,
    }


def m2_store_identity(
    *, store_metadata: dict[str, Any], build_artifact: dict[str, Any]
) -> dict[str, Any]:
    """Both hashes, side by side, for one existing M2 store."""

    contract = reconstruct_m2_contract(
        store_metadata=store_metadata, build_artifact=build_artifact
    )
    return {
        "dataset": contract["dataset"],
        "regime": contract["regime"],
        "feature_build_contract_sha256": contract_sha256(contract),
        "original_full_config_sha256": store_metadata["build_key"]["config_sha256"],
        "store_fingerprint_sha256": store_metadata["fingerprint_sha256"],
        "source_commit": store_metadata.get("source_commit"),
        "contract": contract,
        "note": (
            "original_full_config_sha256 is the declaration hash this store was built "
            "under, carried forward unchanged as provenance. It is not compared, and "
            "nothing rewrites it on the volume."
        ),
    }
