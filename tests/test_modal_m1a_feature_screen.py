"""Static regression test for the M1A Modal launcher's image mounts.

Root cause this guards against: scripts/modal_m1a_feature_screen.py reads
CONFIRMATION_CONFIG_PATH (configs/sa_mlp_confirmation.yaml) via
yaml.safe_load(...read_text()) at module import time, but the image build
never shipped that file into the container with add_local_file. Every
remote invocation crash-looped on FileNotFoundError before running any real
code, observed directly on the first real step-4 smoke submission.

This walks the launcher's own source AST rather than importing the module
and introspecting the built modal.Image object: modal.Image exposes no
public API for listing the local files/dirs a build has mounted (confirmed
by direct dir(image) introspection), so a live-object check isn't available.
A source-level check has the advantage of generalising: it does not name
CONFIRMATION_CONFIG_PATH specifically, so it also catches a future *_PATH
variable read the same way and never mounted.
"""

from __future__ import annotations

import ast
from pathlib import Path

LAUNCHER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "modal_m1a_feature_screen.py"
)


class _ModuleLevelReadTextVisitor(ast.NodeVisitor):
    """Collects X.read_text() targets, but never descends into a def/class body.

    A name read inside a function (e.g. a host-side helper opening a result
    artifact after the run) executes only when that function is called, not
    at import time, so it carries no image-mount obligation -- only names
    read directly in the module's own top-level statements do.
    """

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        pass

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_text"
            and isinstance(node.func.value, ast.Name)
        ):
            self.names.add(node.func.value.id)
        self.generic_visit(node)


def _names_read_as_text_at_module_level(tree: ast.Module) -> set[str]:
    visitor = _ModuleLevelReadTextVisitor()
    visitor.visit(tree)
    return visitor.names


def _names_mounted_via_add_local_file(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_local_file"
            and node.args
        ):
            continue
        arg = node.args[0]
        # add_local_file(str(X), ...)
        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "str":
            arg = arg.args[0] if arg.args else None
        if isinstance(arg, ast.Name):
            names.add(arg.id)
    return names


def test_every_path_read_at_module_import_time_is_mounted_into_the_image() -> None:
    tree = ast.parse(LAUNCHER_PATH.read_text(encoding="utf-8"), filename=str(LAUNCHER_PATH))

    read_at_import_time = _names_read_as_text_at_module_level(tree)
    mounted = _names_mounted_via_add_local_file(tree)

    # Sanity check the walk itself is finding real constants, not vacuously
    # passing because the AST shapes above stopped matching this file.
    assert "CONFIG_PATH" in read_at_import_time
    assert "CONFIRMATION_CONFIG_PATH" in read_at_import_time

    unmounted = read_at_import_time - mounted
    assert not unmounted, (
        f"{sorted(unmounted)} are read via .read_text() at module import time "
        "but never passed to add_local_file -- every remote container will "
        "crash with FileNotFoundError before running any real code"
    )


def _function_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no top-level def {name!r} found in {LAUNCHER_PATH}")


def _calls_function_named(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name
        for call in ast.walk(node)
    )


def test_run_feature_screen_headline_actually_runs_instead_of_refusing() -> None:
    """Regression test for the step-5 authorisation (2026-09-04).

    Before the step-5 amendment to configs/m1a_feature_screen.yaml,
    run_feature_screen_headline's entire body was an unconditional
    ``raise RuntimeError(...)`` -- step 5 was declared but not authorised,
    so the function existed only to refuse. The amendment removed
    the_real_m1a_screen_launch_step_5 from does_not_authorise; this
    launcher must now actually delegate to the shared _run helper (the same
    one run_feature_screen_smoke already uses), not still refuse.
    """
    tree = ast.parse(LAUNCHER_PATH.read_text(encoding="utf-8"), filename=str(LAUNCHER_PATH))
    headline = _function_def(tree, "run_feature_screen_headline")

    body_raises_unconditionally = any(isinstance(stmt, ast.Raise) for stmt in headline.body)
    assert not body_raises_unconditionally, (
        "run_feature_screen_headline still unconditionally raises -- step 5 was authorised "
        "by the 2026-09-04 amendment to configs/m1a_feature_screen.yaml; this function "
        "should delegate to _run(job, stage='headline') like run_feature_screen_smoke does"
    )
    assert _calls_function_named(headline, "_run"), (
        "run_feature_screen_headline no longer calls the shared _run helper -- "
        "it should reuse the same stage-dispatch _run(job, stage='headline') that "
        "run_feature_screen_smoke uses via _run(job, stage='smoke')"
    )
