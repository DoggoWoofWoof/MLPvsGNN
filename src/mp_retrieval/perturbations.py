"""Controlled topology perturbations used to build the phase diagram."""

from __future__ import annotations

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
    occupied = set(pairs)
    target_swaps = round(rate * len(pairs) / 2)
    gen = _generator(seed)
    swaps = 0
    attempts = 0
    max_attempts = max(100, target_swaps * 50)
    while swaps < target_swaps and attempts < max_attempts and len(pairs) >= 2:
        i, j = torch.randint(len(pairs), (2,), generator=gen).tolist()
        attempts += 1
        if i == j:
            continue
        a, b = pairs[i]
        c, d = pairs[j]
        left, right = (a, d), (c, b)
        if not allow_self_loops and (a == d or c == b):
            continue
        if left in occupied or right in occupied or left == right:
            continue
        occupied.remove((a, b))
        occupied.remove((c, d))
        occupied.add(left)
        occupied.add(right)
        pairs[i], pairs[j] = left, right
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
        if self.kind == "type_corruption":
            if edge_type is None:
                raise ValueError("type_corruption requires edge_type")
            return edge_index.cpu().clone(), corrupt_edge_types(edge_type, self.rate, seed=self.seed)
        raise ValueError(f"Unknown perturbation kind: {self.kind}")

