"""The deployable 2wiki compiled-kernel arm (configs/deploy_ck_2wiki.yaml): u_mlp_v2_mix with one hop of the
MP-Approx factorised kernel (levels 5 and 7) over the pool's message edges.

    kappa_e^h = phi_h(v)^T psi_h(u, e),  phi = softplus(W_phi [h_v | q_state_v]),  psi = softplus(W_psi s_ue)
    val_e = W_val s_ue,  alpha = kappa over its sum across v's in-edges and v's self entry, per head
    m_v = sum_e alpha_e val_e,  h'_v = h_v + W_msg m_v (W_msg zero-initialised),  score = readout(body(h'), base_z)

4 heads of rank 32 and value width 16, as at levels 5 and 7. h and base_z are the GatedInputBlock's, and the body and the
residual readout are QLSU's: with W_msg at zero the arm is u_mlp_v2_mix exactly (a test holds it). The entries are the
message edges the GNN reads (universal_v2_models.pack_queries_v2: the first 64 in-pool neighbours per family, in store
order), plus one self entry per node.

s_ue, the neighbour side, by ``side``:
    qi    [X_u R | entry family one-hot (structural, ner, knn, self) | rel_mask, dir_fwd, dir_bwd]   -- query-free
    qw    qi plus the pool-relative weight                                                           -- depends on the pool
    full  [h_u | entry family one-hot | weight, rel_compat, rel_mask, dir_fwd, dir_bwd]              -- depends on the query
X_u R is the node's served embedding times level 1's JL matrix (default_rng(20260929), 1536 x 64 over sqrt 64), a fixed
buffer. Under qi nothing psi or the values read depends on the query or the pool, so psi_e and val_e are compiled once
per edge and C_v = sum psi val^T, c_v = sum psi are sums of them over v's in-pool entries; m_v = phi^T C_v / phi^T c_v
(``compiled``). With form MEAN alpha is uniform over v's entries (no kernel, no phi, no psi). With form SELF (side qi
only) the entries are the self entries alone, so m_v = W_val [X_v R | self one-hot]: the kernel's node-local path with
the message edges removed, the control that separates what the edges carry from what the self entry adds.

No dataset id and no relation text enter the kernel. By configs/universal_v2.yaml's definition this is learned
propagation (an operator over the pool edges whose message weights depend on trainable parameters): an MP arm, not QLS-U.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from mp_retrieval.m3b_features import DIM, FAMILIES
from mp_retrieval.m3b_models import PROJECTION_DIM, PackedBatch
from mp_retrieval.universal_v2_models import ATTR, UMLPv2Mix, block_index_of

HEADS, RANK, HEAD_WIDTH = 4, 32, 16
MSG_WIDTH = HEADS * HEAD_WIDTH                      # 64
JL_SEED, JL_DIM = 20260929, 64                       # level 1's the_projection
ENTRY_FAMILIES = FAMILIES + ("self",)                # the entry one-hot
SIDES = ("qi", "qw", "full")
FORMS = ("KERN", "MEAN", "SELF")
COMBOS = tuple((s, f) for s in SIDES for f in FORMS if f != "SELF" or s == "qi")   # SELF reads the query-free side only
SIDE_ATTRS = {"qi": ("rel_mask", "dir_fwd", "dir_bwd"),
              "qw": ("weight", "rel_mask", "dir_fwd", "dir_bwd"),
              "full": ("weight", "rel_compat", "rel_mask", "dir_fwd", "dir_bwd")}


def jl_matrix() -> np.ndarray:
    """Level 1's R: standard_normal((1536, 64)) / sqrt(64) from default_rng(20260929), float64 (a test holds it equal)."""
    return np.random.default_rng(JL_SEED).standard_normal((DIM, JL_DIM)) / np.sqrt(float(JL_DIM))


def entry_width(side: str, hidden: int) -> int:
    node = hidden if side == "full" else JL_DIM
    return node + len(ENTRY_FAMILIES) + len(SIDE_ATTRS[side])


class OneHopKernel(nn.Module):
    """The kernel: m_v per node from the batch's message edges and self entries (module docstring)."""

    def __init__(self, hidden: int, side: str = "qi", form: str = "KERN"):
        super().__init__()
        if (side, form) not in COMBOS:
            raise ValueError(f"side {side!r} / form {form!r}: one of {COMBOS}")
        self.hidden, self.side, self.form = int(hidden), side, form
        self.n_node = self.hidden if side == "full" else JL_DIM
        width = entry_width(side, self.hidden)
        self.register_buffer("jl", torch.as_tensor(jl_matrix(), dtype=torch.float32), persistent=False)
        self.register_buffer("attr_cols", torch.tensor([ATTR[a] for a in SIDE_ATTRS[side]], dtype=torch.long), persistent=False)
        if form == "KERN":
            self.w_phi = nn.Linear(self.hidden + PROJECTION_DIM, HEADS * RANK)
            self.w_psi = nn.Linear(width, HEADS * RANK)
        self.w_val = nn.Linear(width, MSG_WIDTH)
        self.w_msg = nn.Linear(MSG_WIDTH, self.hidden)
        nn.init.zeros_(self.w_msg.weight)
        nn.init.zeros_(self.w_msg.bias)

    def node_rows(self, batch: PackedBatch, h: torch.Tensor) -> torch.Tensor:
        """The neighbour's row of s_ue: h_u (full) or X_u R (qi, qw), per pool node."""
        if self.side == "full":
            return h
        emb = batch.emb if batch.emb.dtype == torch.float32 else batch.emb.to(torch.float32)
        return emb @ self.jl

    def entries(self, batch: PackedBatch, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(u, v, edge side) over the message edges and then one self entry per node; the edge side is the entry
        family one-hot and the side's attributes (zero on a self entry). Form SELF keeps the self entries only."""
        ea = batch.edge_attr
        m = 0 if self.form == "SELF" else ea.shape[0]
        loops = torch.arange(n, dtype=torch.long, device=ea.device)
        u = torch.cat([batch.edge_index[0][:m], loops])
        v = torch.cat([batch.edge_index[1][:m], loops])
        side = torch.zeros(m + n, len(ENTRY_FAMILIES) + self.attr_cols.numel(), dtype=torch.float32, device=ea.device)
        if m:
            side[:m, : len(FAMILIES)] = ea[:m, : len(FAMILIES)]
            side[:m, len(ENTRY_FAMILIES):] = ea[:m][:, self.attr_cols]
        side[m:, len(FAMILIES)] = 1.0
        return u, v, side

    @staticmethod
    def _lin(lin: nn.Linear, rows: torch.Tensor, side: torch.Tensor, u: torch.Tensor, n_node: int) -> torch.Tensor:
        """lin([rows[u] | side]) with the node part applied per node before the gather (the same function)."""
        W = lin.weight
        return Fn.linear(rows, W[:, :n_node])[u] + Fn.linear(side, W[:, n_node:], lin.bias)

    def parts(self, batch: PackedBatch, h: torch.Tensor, q_state: torch.Tensor):
        """(v, phi or None, psi or None, val) over the entries."""
        n = h.shape[0]
        u, v, side = self.entries(batch, n)
        rows = self.node_rows(batch, h)
        val = self._lin(self.w_val, rows, side, u, self.n_node).view(-1, HEADS, HEAD_WIDTH)
        if self.form != "KERN":
            return v, None, None, val
        phi = Fn.softplus(self.w_phi(torch.cat([h, q_state], dim=1))).view(n, HEADS, RANK)
        psi = Fn.softplus(self._lin(self.w_psi, rows, side, u, self.n_node)).view(-1, HEADS, RANK)
        return v, phi, psi, val

    def weights(self, v: torch.Tensor, phi, psi, n: int, dtype, device) -> torch.Tensor:
        """alpha per entry and head: kappa over its per-receiver sum (KERN), or 1 over the receiver's entry count (MEAN;
        SELF, where the count is 1)."""
        if self.form != "KERN":
            ones = torch.ones(v.numel(), dtype=dtype, device=device)
            cnt = torch.zeros(n, dtype=dtype, device=device).index_add_(0, v, ones)
            return (1.0 / cnt[v]).unsqueeze(1).expand(-1, HEADS)
        kappa = (phi[v] * psi).sum(-1)
        den = torch.zeros(n, HEADS, dtype=kappa.dtype, device=kappa.device).index_add_(0, v, kappa)
        return kappa / den[v]

    def forward(self, batch: PackedBatch, h: torch.Tensor, q_state: torch.Tensor, return_alpha: bool = False):
        n = h.shape[0]
        v, phi, psi, val = self.parts(batch, h, q_state)
        alpha = self.weights(v, phi, psi, n, val.dtype, val.device)
        m = torch.zeros(n, HEADS, HEAD_WIDTH, dtype=val.dtype, device=val.device).index_add_(0, v, val * alpha.unsqueeze(-1))
        m = m.view(n, MSG_WIDTH)
        return (m, alpha) if return_alpha else m

    def moments(self, batch: PackedBatch, h: torch.Tensor, q_state: torch.Tensor):
        """KERN: C_v = sum psi val^T (n, H, R, W) and c_v = sum psi (n, H, R) with phi; MEAN, SELF: the value sum and the count."""
        n = h.shape[0]
        v, phi, psi, val = self.parts(batch, h, q_state)
        if self.form != "KERN":
            ones = torch.ones(v.numel(), dtype=val.dtype, device=val.device)
            S = torch.zeros(n, HEADS, HEAD_WIDTH, dtype=val.dtype, device=val.device).index_add_(0, v, val)
            return S, torch.zeros(n, dtype=val.dtype, device=val.device).index_add_(0, v, ones), None
        C = torch.zeros(n, HEADS, RANK, HEAD_WIDTH, dtype=val.dtype, device=val.device).index_add_(0, v, psi.unsqueeze(-1) * val.unsqueeze(-2))
        c = torch.zeros(n, HEADS, RANK, dtype=psi.dtype, device=psi.device).index_add_(0, v, psi)
        return C, c, phi

    def compiled(self, batch: PackedBatch, h: torch.Tensor, q_state: torch.Tensor) -> torch.Tensor:
        """m_v from the moments: phi^T C_v / phi^T c_v (KERN) or S_v / count_v (MEAN, SELF); equal to forward up to float order."""
        a, b, phi = self.moments(batch, h, q_state)
        n = h.shape[0]
        if self.form != "KERN":
            return (a / b.view(n, 1, 1)).reshape(n, MSG_WIDTH)
        return (torch.einsum("nhr,nhrw->nhw", phi, a) / torch.einsum("nhr,nhr->nh", phi, b).unsqueeze(-1)).reshape(n, MSG_WIDTH)


class CKv2(UMLPv2Mix):
    """u_mlp_v2_mix plus the one-hop kernel's message, added to the input block's h before the body."""

    def __init__(self, n_scalars: int, hidden: int, base_index: int, block_index, dropout: float = 0.2, side: str = "qi",
                 form: str = "KERN"):
        super().__init__(n_scalars, hidden, base_index, block_index, dropout)
        self.arm = {"KERN": f"ck_{side}", "MEAN": f"ck_{side}_mean", "SELF": "ck_self"}[form]
        self.kernel = OneHopKernel(hidden, side, form)

    def q_state(self, batch: PackedBatch) -> torch.Tensor:
        """The GatedInputBlock's q_state (normalize(gelu(query_projection(qemb)))) per node."""
        return Fn.normalize(Fn.gelu(self.input.semantic.query_projection(batch.qemb)), dim=-1)[batch.node_query]

    def forward(self, batch: PackedBatch) -> torch.Tensor:
        h, base_z = self.input(batch)
        m = self.kernel(batch, h, self.q_state(batch))
        return self.readout(self.body(h + self.kernel.w_msg(m)), base_z)

    def forward_compiled(self, batch: PackedBatch) -> torch.Tensor:
        """The same score with m_v taken from the moments (the deployable form)."""
        h, base_z = self.input(batch)
        m = self.kernel.compiled(batch, h, self.q_state(batch))
        return self.readout(self.body(h + self.kernel.w_msg(m)), base_z)


def build_ck(inputs: dict, hidden: int = 128, dropout: float = 0.2, side: str = "qi", form: str = "KERN") -> CKv2:
    """The CK arm over the v2 core (universal_v2_run.model_inputs), as build_arm builds u_mlp_v2_mix."""
    cols = list(inputs["columns"])
    return CKv2(len(cols), hidden, int(inputs["base_local"]), block_index_of(cols), dropout, side, form)
