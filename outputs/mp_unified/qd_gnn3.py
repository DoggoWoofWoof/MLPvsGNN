"""Design look (untracked; not a result and not filed): qd_gnn2.py, pinned and unchanged, with label dropout in
training so a label arm learns to fall back to structure when its labels are missing (the NR read, an unseen or
deleted relation) instead of falling below the relation-free model.

An arm name may end in -dr<R>e<E> (QD arms only): while the model trains (model.training), each batch row has all its
labels shown as 'other' with probability R/100 and every remaining labelled edge with probability E/100; reads,
the select reads and the margin choice see every label. The draws come from their own generator, seeded once per
fit from the torch generator after the model is built, so the batches, the initialisation and the rest of the loop
are qd_gnn2's draw for draw (with R = E = 0 the fit is qd_gnn2's exactly).

    python outputs/mp_unified/qd_gnn3.py --train 2wiki=x4 --arms QD-T0-L2,QD-TXT-L2,QD-TXT-L2-dr25e10 --rule both \
        --out outputs/mp_unified/qd/x.json
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn2 as Q2  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

QG = Q2.QG
Q2_SHA = None   # the qd_gnn2.py this run imports (recorded in the output)
DROP = re.compile(r"^(QD-.*?)-dr(\d+)e(\d+)$")


class QDArmDrop(QG.QDArm):
    """qd_gnn.QDArm with training-time label dropout read from the spec (drop_row, drop_edge)."""

    def __init__(self, sp, Q, TOK, G, z_of, A16, n_id):
        super().__init__(sp, Q, TOK, G, z_of, A16, n_id)
        self.p_row, self.p_edge = float(sp.get("drop_row", 0.0)), float(sp.get("drop_edge", 0.0))
        self.rng = None

    def make(self):
        model = super().make()
        if self.p_row > 0 or self.p_edge > 0:
            self.rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        return model

    def forward(self, model, g, rows):
        P = QG.pack_qd(rows, self.Q, self.TOK, self.z_of, self.sp["K"], self.nr)
        if model.training and self.rng is not None:
            drop = self.rng.random(P.B) < self.p_row
            de = drop[P.e_row.numpy()] | (self.rng.random(P.a.numel()) < self.p_edge)
            P.a = torch.where(torch.from_numpy(de), torch.full_like(P.a, -1), P.a)
        return QG.padded(P, model(P))


_parse_arms = Q2.parse_arms


def parse_arms(spec, names, rule, AU, AG, G3):
    plain, drops = [], {}
    for nm in spec.split(","):
        m = DROP.match(nm)
        if m:
            if m.group(1).startswith("QD-T0"):
                raise SystemExit(f"{nm}: QD-T0 reads no label to drop")
            drops[nm] = (m.group(1), int(m.group(2)) / 100.0, int(m.group(3)) / 100.0)
            plain.append(m.group(1))
        else:
            plain.append(nm)
    base = _parse_arms(",".join(dict.fromkeys(plain)), names, rule, AU, AG, G3)
    out = {}
    for nm in spec.split(","):
        if nm in drops:
            b, pr, pe = drops[nm]
            out[nm] = {**base[b], "drop_row": pr, "drop_edge": pe}
        else:
            out[nm] = base[nm]
    return out


def main():
    global Q2_SHA
    Q2_SHA = QG.sha(Q2.__file__)
    QG.QDArm = QDArmDrop          # qd_gnn2.run builds every QD arm as QG.QDArm(...)
    Q2.parse_arms = parse_arms    # and parses the arm list through Q2.parse_arms
    _run = Q2.run

    def run(a):
        _run(a)
        import json
        out = Path(a.out)
        res = json.loads(out.read_text(encoding="utf-8"))
        res["look"] = "qd_gnn3"
        res["pins"]["qd_gnn2"] = Q2_SHA
        res["qd_gnn3_sha256"] = QG.sha(__file__)
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    Q2.run = run
    Q2.main()


if __name__ == "__main__":
    main()
