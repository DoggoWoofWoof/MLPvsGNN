"""Design look (untracked; not a result and not filed): a relation model that reads a relation only through its text.

Every learned relation model so far gives a relation a code it fits (a free vector, P phi + D, or an operator linear in
it). Read on another graph, 2wiki's codes lose to the untyped T0 on hotpotqa at every kappa (ksweep.py: T0 +0.41, the
best typed model +0.14 on its BASE read and +0.28 with its labels removed), and on musique every model is harmful at its
fitted kappa. This look asks how far a relation model gets that cannot fit a relation at all: a walk type's logit is a
few numbers times the match between the question and the edge label's gte-Qwen2 vector, so an unseen relation (a held
phrase, another dataset's anchor, a KB relation name) is scored by its text as a training relation is, and an edge with
no label reads its direction's bias:
    w(t) = c[b, L] + sum over the walk's positions p of  bo_p[f] + pm(k_p) bl_p[f] + gam_p[f] m_p(q, phi(k_p))
           (f: the edge's family x direction; pm: the token carries a phrase; 'other', ner and knn carry none)
    m_p   cos       cos(q, phi)                                    (the whole relation model: 37 numbers)
          cosd      <q * u_p, phi>, u_p in R^1536 from 1            (a diagonal metric per position)
          cosrot    <R(theta_p) q, phi>, theta_p in R^768 from 0     (a 2-plane rotation of the question per position)
          cosr<r>   cos(q, phi) + <U_p^T q, V_p^T phi>, rank r       (a low-rank metric per position; V_p from 0)
Every match is multiplied by a fixed S = 20 (a contrastive temperature of 0.05): the vectors are unit and a question's
cosines to the phrases spread by a few hundredths (2wiki's phrase-phrase cosines: mean 0.37, std 0.08), so without it
gam would have to grow to tens before a relation moved a logit (anchor_gen's 'c' models stopped at |gam| <= 0.57).
The two positions have their own weights, so a two-edge walk r1 -> r2 does not score as r2 -> r1: order matters (cos
with position-tied weights would be commutative). gam starts at 0, so every fit starts as a family x direction model.
The null token's logit is c_null (with suffix n, + <a, q>, a from 0). Everything else is anchor_gen3's: the bases
A{256,1024,4096}[-1], hold, relation dropout, the fit loop, the rules, selection and reads. The model is a GenMix
subclass and its spec records 'pca', so anchor_gen3's loader, and through it every reader (anchor_ens, anchor_nn,
anchor_gen_kb, ksweep, ksweep_kb), reads it once this module has bound it into anchor_gen3; the subcommands below run
those readers in-process after binding, their files unchanged. Nothing reads a neighbour's score or state.

    python anchor_cos.py fit --dataset 2wiki --variants x4+x5+x6:A4096-1/cos,... [--hold 0/5] [anchor_gen3's fit flags]
    python anchor_cos.py ens --target hotpotqa --groups "a=x.pt;..." --out .../anchor_ens_cos_....json
    python anchor_cos.py kb --target metaqa --models a.pt,b.pt --out .../anchor_gen_kb_cos_....json
    python anchor_cos.py ksweep --target musique --models a.pt --deltas=-3,-2,-1,0 --out .../anchor_ens_ksweep_cos_....json
    python anchor_cos.py ksweep_kb --target metaqa --models a.pt --deltas=-3,-2,-1,0 --out .../anchor_gen_kb_ksweep_cos_....json
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import re  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "outputs" / "mp_approx_kb_anchor" / "host"))
import anchor_gen3 as G3  # noqa: E402

AG, AW6, AW = G3.AG, G3.AW6, G3.AW
ANCHOR_GEN3_SHA = "ee5f76dc7d391649e04ab2b3575d4fdd30dd62cd2a34b1b7dbd114d56f7b14a7"
COS_RE = re.compile(r"(cos|cosd|cosrot|cosr(\d+))(n?)")
RANKS = (4, 8, 16)
S = 20.0
_G3_PARSE, _G3_MAKE_FOR = G3.parse, G3.make_for


class CosMix(AG.GenMix):
    """A walk type's logit from the question's match to its edges' label vectors (see the docstring)."""

    def __init__(self, nt, K, phi, metric, rank=None, qnull=False, qdim=1536):
        super().__init__(nt, K, phi, "trans", cos=False, d=4, qdim=qdim)   # GenMix's buffers: Phi, pmask, fdir
        for nm in ("A", "bk", "t1", "t2", "ln", "nu", "P", "D", "s0"):
            delattr(self, nm)
        self.metric, self.rank, self.qnull = metric, rank, qnull
        self.bo = torch.nn.Parameter(torch.zeros(2, 5))
        self.bl = torch.nn.Parameter(torch.zeros(2, 5))
        self.gam = torch.nn.Parameter(torch.zeros(2, 5))
        if metric == "cosd":
            self.u = torch.nn.Parameter(torch.ones(2, qdim))
        elif metric == "cosrot":
            self.theta = torch.nn.Parameter(torch.zeros(2, qdim // 2))
        elif metric == "cosr":
            self.U = torch.nn.Parameter(torch.randn(2, qdim, rank) * qdim ** -0.5)
            self.V = torch.nn.Parameter(torch.zeros(2, qdim, rank))
        if qnull:
            self.a_null = torch.nn.Parameter(torch.zeros(qdim))

    def match(self, qemb, pos):
        """(B, nt): position pos's match of the question to every token's label vector, times S (0 for a token with none)."""
        if self.metric == "cos":
            m = qemb @ self.Phi.T
        elif self.metric == "cosd":
            m = (qemb * self.u[pos]) @ self.Phi.T
        elif self.metric == "cosrot":
            m = AW6.rot(qemb, self.theta[pos]) @ self.Phi.T
        else:
            m = qemb @ self.Phi.T + (qemb @ self.U[pos]) @ (self.Phi @ self.V[pos]).T
        return S * m

    def forward(self, qemb, tb, t1, t2, tmask, ent_q, ent_t, ent_node, z, nmask):
        L = (t2 > 0).long()
        k2 = (t2 - 1).clamp(min=0)
        pm = self.pmask.to(qemb.dtype)
        m1 = self.match(qemb, 0)
        m2 = m1 if self.metric == "cos" else self.match(qemb, 1)
        f1, f2 = self.fdir[t1], self.fdir[k2]
        w = self.c[tb, L] + self.bo[0][f1] + pm[t1] * self.bl[0][f1] + self.gam[0][f1] * m1.gather(1, t1)
        w2 = self.bo[1][f2] + pm[k2] * self.bl[1][f2] + self.gam[1][f2] * m2.gather(1, k2)
        w = w + torch.where(t2 > 0, w2, torch.zeros_like(w2))
        w = w.masked_fill(~tmask, float("-inf"))
        null = self.c_null + (qemb @ self.a_null if self.qnull else torch.zeros(qemb.shape[0]))
        return torch.softmax(torch.cat([w, null[:, None]], 1), 1)[:, :-1]

    def params_used(self):
        return int(sum(p.numel() for p in self.parameters()))


def parse(name, train_looks):
    """anchor_gen3.parse, with the cos models (a glin proxy gives the common fields)."""
    head, at, seeds = name.partition("@")
    train, sep, rest = head.partition(":")
    parts = rest.split("/")
    mm = COS_RE.fullmatch(parts[1]) if len(parts) > 1 else None
    if not mm:
        return _G3_PARSE(name, train_looks)
    if parts[0].startswith("T0"):
        raise SystemExit(f"{name}: a cos model reads phrase bases")
    rank = int(mm.group(2)) if mm.group(2) else None
    if rank is not None and rank not in RANKS:
        raise SystemExit(f"{name}: cosr rank among {RANKS}")
    sp = _G3_PARSE(f"{train}:{'/'.join([parts[0], 'glin'] + parts[2:])}{at}{seeds}", train_looks)
    sp.update({"family": "cos", "op": "cosr" if rank else mm.group(1), "cos": True, "model": parts[1], "rank": rank,
               "qnull": mm.group(3) == "n"})
    return sp


def make_for(sp, nt, phi, basis=None):
    if sp["family"] == "cos":
        return lambda: CosMix(nt, sp["K"], phi, sp["op"], rank=sp["rank"], qnull=sp["qnull"])
    return _G3_MAKE_FOR(sp, nt, phi, basis)


def bind():
    """Bind the cos models into anchor_gen3 (its parse and make_for; its loader reads make_for at call time)."""
    if AW.sha(Path(G3.__file__)) != ANCHOR_GEN3_SHA:
        raise SystemExit("anchor_gen3.py is not the pinned file")
    G3.parse, G3.make_for = parse, make_for


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("fit", "ens", "kb", "ksweep", "ksweep_kb"):
        raise SystemExit("anchor_cos.py fit|ens|kb|ksweep|ksweep_kb ...")
    cmd, rest = argv[0], argv[1:]
    bind()
    if cmd == "fit":
        if "--xread" in rest:
            raise SystemExit("fit: cross reads go through ens / kb")
        G3.main(rest)
    elif cmd == "ens":
        import anchor_ens as EN
        EN.main(rest)
    elif cmd == "kb":
        import anchor_gen_kb as KB
        KB.main(rest)
    elif cmd == "ksweep":
        import ksweep as KS
        KS.main(rest)
    else:
        import ksweep_kb as KK
        KK.main(rest)


if __name__ == "__main__":
    main()
