"""Controlled topology perturbations used to build the phase diagram."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _edge_pairs(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    return list(zip(edge_index[0].cpu().tolist(), edge_index[1].cpu().tolist()))


def coalesce_edges(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Remove duplicate directed edges while retaining the first edge type."""

    seen: dict[tuple[int, int], int] = {}
    keep: list[int] = []
    for idx, pair in enumerate(_edge_pairs(edge_index)):
        if pair not in seen:
            seen[pair] = idx
            keep.append(idx)
    keep_tensor = torch.tensor(keep, dtype=torch.long)
    types = None if edge_type is None else edge_type.cpu()[keep_tensor]
    return edge_index.cpu()[:, keep_tensor], types


def drop_edges(
    edge_index: torch.Tensor,
    rate: float,
    *,
    seed: int,
    edge_type: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Uniformly remove a fraction of directed edges."""

    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    num_edges = edge_index.shape[1]
    keep_count = num_edges - round(rate * num_edges)
    order = torch.randperm(num_edges, generator=_generator(seed))[:keep_count]
    order = order.sort().values
    types = None if edge_type is None else edge_type.cpu()[order]
    return edge_index.cpu()[:, order], types


def add_random_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    rate: float,
    *,
    seed: int,
    edge_type: torch.Tensor | None = None,
    new_edge_type: int = -1,
    allow_self_loops: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Add ``round(rate * |E|)`` unique uniformly random directed edges."""

    if rate < 0.0:
        raise ValueError("rate must be non-negative")
    requested = round(rate * edge_index.shape[1])
    existing = set(_edge_pairs(edge_index))
    additions: list[tuple[int, int]] = []
    gen = _generator(seed)
    max_attempts = max(100, requested * 30)
    attempts = 0
    while len(additions) < requested and attempts < max_attempts:
        batch_size = min(max(64, (requested - len(additions)) * 2), 100_000)
        sources = torch.randint(num_nodes, (batch_size,), generator=gen).tolist()
        targets = torch.randint(num_nodes, (batch_size,), generator=gen).tolist()
        for pair in zip(sources, targets):
            attempts += 1
            if (allow_self_loops or pair[0] != pair[1]) and pair not in existing:
                existing.add(pair)
                additions.append(pair)
                if len(additions) == requested:
                    break
            if attempts >= max_attempts:
                break
    if len(additions) != requested:
        raise RuntimeError("Could not sample enough unique random edges")
    if additions:
        extra = torch.tensor(additions, dtype=torch.long).T.contiguous()
        perturbed = torch.cat([edge_index.cpu(), extra], dim=1)
    else:
        perturbed = edge_index.cpu().clone()
    if edge_type is None:
        types = None
    else:
        extra_types = torch.full((len(additions),), new_edge_type, dtype=edge_type.dtype)
        types = torch.cat([edge_type.cpu(), extra_types])
    return perturbed, types


def degree_preserving_rewire(
    edge_index: torch.Tensor,
    rate: float,
    *,
    seed: int,
    allow_self_loops: bool = False,
) -> torch.Tensor:
    """Swap edge destinations, preserving directed in- and out-degree exactly.

    This operator is intended for directed artifacts. For an undirected graph,
    store both directions and interpret the result as a degree-controlled null;
    reciprocity is not guaranteed after rewiring and should be measured.
    """

    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    pairs = _edge_pairs(edge_index)
    occupied = Counter(pairs)
    target_swaps = round(rate * len(pairs) / 2)
    gen = _generator(seed)
    swaps = 0
    used_positions: set[int] = set()
    attempts = 0
    max_attempts = max(100, target_swaps * 50)
    while swaps < target_swaps and attempts < max_attempts and len(pairs) >= 2:
        i, j = torch.randint(len(pairs), (2,), generator=gen).tolist()
        attempts += 1
        if i == j or i in used_positions or j in used_positions:
            continue
        a, b = pairs[i]
        c, d = pairs[j]
        left, right = (a, d), (c, b)
        old_left, old_right = pairs[i], pairs[j]
        if Counter((old_left, old_right)) == Counter((left, right)):
            continue
        if not allow_self_loops and (a == d or c == b):
            continue
        occupied[old_left] -= 1
        if occupied[old_left] == 0:
            del occupied[old_left]
        occupied[old_right] -= 1
        if occupied[old_right] == 0:
            del occupied[old_right]
        invalid = left in occupied or right in occupied or left == right
        if invalid:
            occupied[old_left] += 1
            occupied[old_right] += 1
            continue
        occupied[left] += 1
        occupied[right] += 1
        pairs[i], pairs[j] = left, right
        used_positions.update((i, j))
        swaps += 1
    if swaps < target_swaps:
        raise RuntimeError(f"Completed {swaps}/{target_swaps} requested degree-preserving swaps")
    return torch.tensor(pairs, dtype=torch.long).T.contiguous()


def corrupt_edge_types(edge_type: torch.Tensor, rate: float, *, seed: int) -> torch.Tensor:
    """Permute a selected subset of type labels, preserving the type histogram."""

    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    result = edge_type.cpu().clone()
    count = round(rate * result.numel())
    if count < 2:
        return result
    gen = _generator(seed)
    selected = torch.randperm(result.numel(), generator=gen)[:count]
    permutation = selected[torch.randperm(count, generator=gen)]
    result[selected] = result[permutation]
    return result


def inject_hubs(
    edge_index: torch.Tensor,
    num_nodes: int,
    rate: float,
    *,
    seed: int,
    num_hubs: int = 1,
    allow_self_loops: bool = False,
) -> torch.Tensor:
    """Redirect a fraction of edge destinations to high-degree nodes.

    The edge count is preserved. Existing highest-degree nodes are used as hubs
    so this intervention isolates hub amplification rather than adding nodes or
    parameters. Duplicate proposals are deterministically resampled.
    """

    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    if num_hubs < 1 or num_hubs > num_nodes:
        raise ValueError("num_hubs must be between one and num_nodes")
    pairs = _edge_pairs(edge_index)
    if not pairs or rate == 0.0:
        return edge_index.cpu().clone()
    source, target = edge_index.cpu()
    degree = torch.bincount(source, minlength=num_nodes) + torch.bincount(
        target, minlength=num_nodes
    )
    hubs = torch.argsort(degree, descending=True)[:num_hubs].tolist()
    count = round(rate * len(pairs))
    selected = torch.randperm(len(pairs), generator=_generator(seed))[:count].tolist()
    occupied = Counter(pairs)
    gen = _generator(seed + 1)
    for edge_pos in selected:
        old = pairs[edge_pos]
        occupied[old] -= 1
        if occupied[old] == 0:
            del occupied[old]
        source_node = old[0]
        hub_order = torch.randperm(len(hubs), generator=gen).tolist()
        replacement = next(
            (
                (source_node, hubs[idx])
                for idx in hub_order
                if (allow_self_loops or source_node != hubs[idx])
                and (source_node, hubs[idx]) not in occupied
            ),
            old,
        )
        pairs[edge_pos] = replacement
        occupied[replacement] += 1
    return torch.tensor(pairs, dtype=torch.long).T.contiguous()


def degrade_features(
    features: torch.Tensor,
    rate: float,
    *,
    seed: int,
    mode: str = "mask",
) -> torch.Tensor:
    """Apply identical controlled feature degradation before either model.

    ``mask`` zeros a fraction of scalar entries. ``gaussian`` adds standard
    normal noise scaled by each feature column's empirical standard deviation.
    """

    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    result = features.detach().cpu().clone()
    gen = _generator(seed)
    if mode == "mask":
        mask = torch.rand(result.shape, generator=gen) < rate
        result[mask] = 0.0
        return result
    if mode == "gaussian":
        scale = result.float().std(dim=0, unbiased=False).clamp_min(1e-6)
        noise = torch.randn(result.shape, generator=gen, dtype=result.dtype)
        return result + rate * noise * scale
    raise ValueError(f"Unknown feature degradation mode: {mode}")


def remove_edge_types(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    removed_types: set[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove all edges whose relation type is in ``removed_types``."""

    if edge_type.ndim != 1 or edge_type.numel() != edge_index.shape[1]:
        raise ValueError("edge_type must align with edge_index")
    keep = torch.tensor(
        [int(value) not in removed_types for value in edge_type.cpu().tolist()],
        dtype=torch.bool,
    )
    return edge_index.cpu()[:, keep], edge_type.cpu()[keep]


@dataclass(frozen=True)
class TopologyPerturbation:
    """Serializable description of one controlled topology intervention."""

    kind: str
    rate: float
    seed: int

    def apply(
        self,
        edge_index: torch.Tensor,
        *,
        num_nodes: int,
        edge_type: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.kind == "drop":
            return drop_edges(edge_index, self.rate, seed=self.seed, edge_type=edge_type)
        if self.kind == "add_random":
            return add_random_edges(
                edge_index,
                num_nodes,
                self.rate,
                seed=self.seed,
                edge_type=edge_type,
            )
        if self.kind == "degree_rewire":
            return degree_preserving_rewire(edge_index, self.rate, seed=self.seed), edge_type
        if self.kind == "hub_injection":
            return inject_hubs(
                edge_index,
                num_nodes,
                self.rate,
                seed=self.seed,
            ), edge_type
        if self.kind == "type_corruption":
            if edge_type is None:
                raise ValueError("type_corruption requires edge_type")
            return edge_index.cpu().clone(), corrupt_edge_types(edge_type, self.rate, seed=self.seed)
        raise ValueError(f"Unknown perturbation kind: {self.kind}")
