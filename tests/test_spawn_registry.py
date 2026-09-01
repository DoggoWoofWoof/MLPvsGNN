"""The spawn registry must cover every long package that needs to outlive a client.

``modal run --detach`` does not survive client teardown, so any package that runs
for hours must be submitted through ``spawn_modal_jobs.py``. A package missing
from that registry is only discovered when someone tries to launch it, which is
exactly the moment a gate opens.
"""

import importlib

import pytest

from scripts.spawn_modal_jobs import PACKAGES

# Long GPU packages. pilot3 is excluded: it is a short smoke run and is not
# volume-bound, so it does not need a persistent server-side call.
LONG_PACKAGES = (
    "edge-provenance",
    "candidate-budget",
    "phase-screen",
    "candidate-headroom",
    "online-systems",
)


@pytest.mark.parametrize("package", LONG_PACKAGES)
def test_every_long_package_can_be_spawned_persistently(package: str) -> None:
    assert package in PACKAGES, (
        f"{package} cannot be launched persistently; it would need "
        "modal run --detach, which dies with the client"
    )


@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_each_registered_package_names_a_real_module_and_function(package: str) -> None:
    module_name, stages = PACKAGES[package]
    module = importlib.import_module(module_name)
    assert hasattr(module, "app"), f"{module_name} exposes no Modal app"
    assert hasattr(module, "_jobs"), f"{module_name} exposes no _jobs()"
    for stage, function_name in stages.items():
        assert hasattr(module, function_name), (
            f"{module_name} has no {function_name!r} for stage {stage!r}; "
            "Modal functions are module-level objects, not app attributes"
        )
