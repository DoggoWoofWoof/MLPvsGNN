# CPU–GPU equivalence

Declaration: `configs/cpu_gpu_equivalence.yaml` (declared at `1ee9e48`). Script: `scripts/cpu_gpu_equivalence.py`; device path: `src/mp_retrieval/device_placement.py`. Sidecars: `outputs/cpu_gpu_equivalence/` (git-ignored).

> On inputs that are bit-identical by construction, do the lab host's CPU and GPU compute the frozen universal-v2 models' per-query scores and training gradients within the repository's own tolerance of the laptop CPU, the machine that produced every filed number?

This decides only **where** later stages may run. Nothing was fitted or selected, and no retrieval number was read: the metrics below are computed on fit-carve queries only to compare one placement with another.

## Verdicts

| candidate | verdict | failing cells | cells checked | bit-identical to the reference | largest excess over the tolerance |
|---|---|---:|---:|---|---:|
| `host_cpu_t8` | **EQUIVALENT_WITHIN_TOLERANCE** | 0 | 2,937 | no | — |
| `host_gpu_det` | **NOT_EQUIVALENT** | 2 | 3,993 | no | 2.9e-06 |
| `host_gpu_default` | **NOT_EQUIVALENT** | 245 | 3,993 | no | 2.4e-04 |

The floor, `laptop_cpu_t4` against `laptop_cpu_t8` (the same laptop at 4 threads), fails 0 of 2,937 cells and is not bit-identical to the reference. A candidate's failing cell counts against it only where the floor passes that cell (readings.NOT_EQUIVALENT, INCONCLUSIVE_FLOOR).

Cells: T1, one per (model, query draw), the centred scores under `assert_close(atol=1e-5, rtol=1e-5)`; T2, one per (model, batch) for the loss (`|L - L_ref| <= 1e-5|L_ref| + 1e-6`) and the global gradient (`||g - g_ref|| <= 1e-4||g_ref||`), and one per (model, batch, parameter tensor) (`||g_p - g_ref_p|| <= 1e-4||g_ref_p|| + 1e-6||g_ref||`). A candidate's `_r2` repeat is held to the T1 gates too.

**Wording.** A pass says the placement computes these models' scores and gradients within the filed tolerance on these inputs. It never says that a fit on the placement reproduces a laptop fit (placement_rules.new_seeds).

**What a pass opens.** EQUIVALENT_BIT_IDENTICAL or EQUIVALENT_WITHIN_TOLERANCE makes the placement NAMEABLE as the compute placement in a later dated authorization block of a declared stage. That block quotes the tested settings verbatim (host, env name@hash, torch and CUDA versions, the determinism mode that passed, TF32 off, float32, 8 threads for host CPU work) and runs mirror_verification first. A verdict opens no stage by itself.

**What a fail keeps.** NOT_EQUIVALENT or INCONCLUSIVE_FLOOR keeps the placement barred. No tolerance is relaxed and no setting is changed under this file; the only routes are a new arm (placement_arms_added_later) or a new declaration.

## The device-path gate

`tests/test_cpu_gpu_equivalence.py` passed on the laptop (16 passed, 2 skipped: the two CUDA tests, no CUDA device) and on the host GPU in `mpr-cu128` under the `host_gpu_det` settings (18 passed, 0 skipped) before any candidate arm ran. On the CPU the device-path copies give the pinned fit and evaluation exactly (weights `torch.equal`, records equal apart from wall-clock fields, every metric array equal). On CUDA a synthetic GAT fit wrote CPU tensors to its checkpoint and resumed to the uninterrupted fit bit for bit (weights `torch.equal`), and the CUDA evaluation gave the CPU evaluation's metrics on every compared synthetic query (34 of 52 compared; 18 with a near tie at twice the forward tolerance were counted and left out). Determinism warnings: none.

## The probe

11 batches of the frozen stage-2 draw for seed 0 (batch 16, `dataset_draw per_query`), 176 query draws from the six fit carves: 2wiki 30, hotpotqa 28, metaqa 24, musique 42, squad 26, webqsp 26. Packed once on the laptop by the frozen code and shipped as a 0.97 GB bundle (manifest sha256 `26c61bf5e9764c54…`). Every arm checked every bundle file and every reconstructed field against the manifest, the six checkpoints and records against the declaration, the six-dataset relation bank (7,067 rows), the frozen contract (129 columns) and the twelve frozen-code files before it computed anything.

## The arms

| arm | host | env | torch | device | threads | settings | seconds | warnings |
|---|---|---|---|---|---:|---|---:|---:|
| `laptop_cpu_t8` | Inspiron-14 | `Python313` | 2.8.0+cpu | cpu | 8 | — | 423 | 0 |
| `laptop_cpu_t8_r2` | Inspiron-14 | `Python313` | 2.8.0+cpu | cpu | 8 | — | 54 | 0 |
| `laptop_cpu_t4` | Inspiron-14 | `Python313` | 2.8.0+cpu | cpu | 4 | — | 459 | 0 |
| `host_cpu_t8` | DESKTOP-SLQMEQH | `mpr-cpu@31803e6457ab` | 2.8.0+cpu | cpu | 8 | — | 239 | 0 |
| `host_gpu_det` | DESKTOP-SLQMEQH | `mpr-cu128@62fc45e9e1ba` | 2.8.0+cu128 | NVIDIA RTX 4500 Ada Generation (driver 596.71) | 8 | mode det; TF32 off; deterministic on (warn_only) | 29 | 0 |
| `host_gpu_det_r2` | DESKTOP-SLQMEQH | `mpr-cu128@62fc45e9e1ba` | 2.8.0+cu128 | NVIDIA RTX 4500 Ada Generation (driver 596.71) | 8 | mode det; TF32 off; deterministic on (warn_only) | 6 | 0 |
| `host_gpu_default` | DESKTOP-SLQMEQH | `mpr-cu128@62fc45e9e1ba` | 2.8.0+cu128 | NVIDIA RTX 4500 Ada Generation (driver 596.71) | 8 | mode default; TF32 off; deterministic off | 26 | 0 |
| `host_gpu_default_r2` | DESKTOP-SLQMEQH | `mpr-cu128@62fc45e9e1ba` | 2.8.0+cu128 | NVIDIA RTX 4500 Ada Generation (driver 596.71) | 8 | mode default; TF32 off; deterministic off | 5 | 0 |

## Reported beside, never gating

### Forward (T1): largest |centred difference| and largest ratio to the tolerance

| model | `laptop_cpu_t4` | `host_cpu_t8` | `host_gpu_det` | `host_gpu_default` |
|---|---:|---:|---:|---:|
| GNN six__s0 | 7.6e-06 (0.55) | 0 (0.00) | 1.5e-05 (0.69) | 7.5e-05 (2.44) |
| GNN six__s1 | 9.5e-06 (0.71) | 0 (0.00) | 1.6e-05 (0.97) | 2.4e-04 (13.02) |
| GNN six__s2 | 7.6e-06 (0.48) | 0 (0.00) | 1.3e-05 (0.64) | 9.5e-05 (3.79) |
| twin__s0 | 7.6e-06 (0.47) | 0 (0.00) | 1.1e-05 (0.98) | 2.1e-04 (4.53) |
| twin__s1 | 1.1e-05 (0.74) | 0 (0.00) | 2.3e-05 (1.28) | 1.7e-04 (4.68) |
| twin__s2 | 9.5e-06 (0.50) | 0 (0.00) | 1.3e-05 (0.74) | 1.8e-04 (4.82) |

### Gradient (T2): loss and global-gradient relative errors, largest (median) over batches

| model | `laptop_cpu_t4` loss | `host_cpu_t8` loss | `host_gpu_det` loss | `host_gpu_default` loss | `laptop_cpu_t4` gradient | `host_cpu_t8` gradient | `host_gpu_det` gradient | `host_gpu_default` gradient |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GNN six__s0 | 1.9e-07 (0) | 0 (0) | 2.7e-06 (1.0e-06) | 7.4e-07 (4.1e-07) | 1.8e-06 (1.2e-06) | 1.6e-07 (9.7e-08) | 7.4e-06 (3.6e-06) | 1.2e-05 (5.0e-06) |
| GNN six__s1 | 1.9e-07 (7.7e-08) | 0 (0) | 3.2e-06 (1.1e-06) | 2.0e-06 (4.8e-07) | 2.2e-06 (1.3e-06) | 1.8e-07 (1.0e-07) | 7.2e-06 (3.2e-06) | 1.8e-05 (6.2e-06) |
| GNN six__s2 | 2.3e-07 (9.6e-08) | 0 (0) | 2.0e-06 (9.4e-07) | 1.3e-06 (3.2e-07) | 2.4e-06 (1.3e-06) | 1.3e-07 (9.5e-08) | 6.1e-06 (2.8e-06) | 2.0e-04 (4.6e-06) |
| twin__s0 | 2.0e-07 (0) | 0 (0) | 3.7e-07 (1.5e-07) | 2.5e-06 (4.6e-07) | 1.9e-06 (4.0e-07) | 7.4e-08 (1.1e-08) | 5.4e-06 (2.7e-06) | 1.1e-05 (5.2e-06) |
| twin__s1 | 1.1e-07 (0) | 0 (0) | 3.3e-07 (1.6e-07) | 2.6e-06 (4.3e-07) | 9.4e-07 (5.5e-07) | 4.2e-08 (6.9e-09) | 3.3e-06 (2.1e-06) | 1.2e-05 (4.5e-06) |
| twin__s2 | 1.2e-07 (0) | 0 (0) | 3.4e-07 (1.5e-07) | 2.4e-06 (4.4e-07) | 1.7e-06 (4.2e-07) | 6.2e-08 (1.2e-08) | 4.0e-06 (2.5e-06) | 8.4e-06 (4.6e-06) |

### Decision agreement

Query draws (of 176) whose 14 metrics differ from the reference in any place; in brackets, how many of those hold the forward tolerance and sit on a near tie of the reference (two candidates within twice the tolerance).

| model | `laptop_cpu_t4` | `host_cpu_t8` | `host_gpu_det` | `host_gpu_default` |
|---|---:|---:|---:|---:|
| GNN six__s0 | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| GNN six__s1 | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| GNN six__s2 | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| twin__s0 | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| twin__s1 | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| twin__s2 | 0 (0) | 0 (0) | 0 (0) | 0 (0) |

### Determinism: is each `_r2` run bit-identical to its first run?

| repeat | GNN six__s0 | GNN six__s1 | GNN six__s2 | twin__s0 | twin__s1 | twin__s2 |
|---|---|---|---|---|---|---|
| `laptop_cpu_t8_r2` | yes | yes | yes | yes | yes | yes |
| `host_gpu_det_r2` | yes | yes | yes | yes | yes | yes |
| `host_gpu_default_r2` | no | no | no | no | no | no |

Warnings raised by the determinism request: none.

### Trajectory (T3, descriptive): largest |centred difference| from the reference's own trajectory

Seed 0 of each family, dropout off, a fresh AdamW (lr 1e-3, weight decay 1e-4, clip 1.0) over the probe batches as consecutive steps; how fast rounding differences grow, beside the same growth for the floor. Context for `placement_rules.new_seeds` only.

| arm | model | after 1 | after 2 | after 4 | after 8 |
|---|---|---:|---:|---:|---:|
| `laptop_cpu_t4` | GNN six__s0 | 2.5e-05 | 3.2e-05 | 2.6e-05 | 4.4e-05 |
| `laptop_cpu_t4` | twin__s0 | 1.3e-05 | 1.8e-05 | 2.3e-05 | 3.1e-05 |
| `host_cpu_t8` | GNN six__s0 | 6.7e-06 | 9.5e-06 | 2.0e-05 | 3.3e-05 |
| `host_cpu_t8` | twin__s0 | 4.8e-06 | 1.3e-05 | 1.1e-05 | 1.7e-05 |
| `host_gpu_det` | GNN six__s0 | 4.6e-05 | 7.2e-05 | 5.1e-05 | 1.2e-04 |
| `host_gpu_det` | twin__s0 | 3.6e-05 | 5.9e-05 | 4.4e-05 | 4.9e-05 |
| `host_gpu_default` | GNN six__s0 | 7.7e-05 | 9.6e-05 | 9.6e-05 | 1.6e-04 |
| `host_gpu_default` | twin__s0 | 1.5e-04 | 1.1e-04 | 7.7e-05 | 1.0e-04 |

### Clock (T4, systems)

Milliseconds per probe batch, timed on the T1 forward pass and the T2 gradient pass themselves (no separate pass), packing excluded, CUDA synchronised around each batch: the median over the family's three checkpoints of each checkpoint's median over the probe batches, and the first batch (warm-up included). Peak GPU memory is the gradient pass's, with the probe batches resident on the device.

| arm | family | forward median | forward first | forward+backward median | peak GPU MB |
|---|---|---:|---:|---:|---:|
| `laptop_cpu_t8` | GNN | 1487.9 | 1644.2 | 5841.6 | — |
| `laptop_cpu_t8` | twin | 226.6 | 226.6 | 562.2 | — |
| `laptop_cpu_t4` | GNN | 1625.4 | 1993.9 | 5743.4 | — |
| `laptop_cpu_t4` | twin | 233.8 | 243.7 | 611.4 | — |
| `host_cpu_t8` | GNN | 1080.5 | 1449.8 | 3130.9 | — |
| `host_cpu_t8` | twin | 149.5 | 156.5 | 284.6 | — |
| `host_gpu_det` | GNN | 69.4 | 94.1 | 358.2 | 6608 |
| `host_gpu_det` | twin | 12.2 | 11.9 | 24.1 | 1785 |
| `host_gpu_default` | GNN | 59.2 | 83.3 | 337.7 | 6560 |
| `host_gpu_default` | twin | 7.5 | 8.8 | 17.1 | 1736 |

## Placement rules (filed with the declaration, binding every stage placed off the laptop)

- **one_placement_per_comparison**: arms whose numbers a stage compares are fitted and evaluated on one placement, so a device is never a hidden difference between an MP arm and a non-MP arm. A cross-placement reading is allowed only if the stage's block names it in advance, and then only on seed means (never seed-paired), with each placement named beside its number.
- **new_seeds**: a GPU fit is a new draw of the frozen rule (its dropout masks come from the device's generator): it never replaces, pairs with, or is averaged as the same seed as a laptop fit with the same seed number. A host CPU fit is treated the same way, even after EQUIVALENT_BIT_IDENTICAL, unless a later block declares a replica check and it passes.
- **no_regeneration**: filed arrays and checkpoints (M2-M3B, universal-v2, UMLP) are never re-produced on another placement to confirm or replace them
- **record_the_placement**: every fit and eval record names the host, env name@hash, device and driver, torch and CUDA versions, determinism and TF32 flags, and threads. Weights are saved as CPU tensors and pinned by state_sha256.
- **passed_settings_only**: a mode that did not pass is not used, even when another mode on the same GPU passed
- **packing_on_cpu**: batches are packed on the host CPU by the frozen code; the device sees only packed tensors
- **reference**: the laptop CPU stays the reference placement

## Deviations

None recorded by the arms.
