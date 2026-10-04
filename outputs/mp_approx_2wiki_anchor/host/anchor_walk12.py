"""Design look (untracked; not a result and not filed): anchor-typed walks, round 12: a learned temperature on the twin's score.

Every round so far scores a pool node s = z + kappa * bonus (z the twin's z-scored score) and trains on the golds'
cross-entropy over log_softmax(s). The ranking depends only on the bonus's size against z, but the loss also depends on
how sharp the softmax is, and z's scale is fixed at 1: the loss can sharpen the softmax on the golds only by raising the
bonus, which lifts every node a type reaches, the rivals of the twin's right rank-1 node with them. On hotpotqa every
fit at 3e-3 peaked at epoch 0 and fell below the twin as it trained, while its reach sets hold most of the GNN's lifts
(diag_oracle2.py: rho 2 to 4 under the oracle). A model named with the suffix 'zt' scores
    s = exp(log_tz) * z + kappa * bonus        log_tz one scalar starting at 0, so each fit starts as its base model's,
so the loss can sharpen z itself and the bonus's size against z is learned apart from the softmax's temperature. The
rule's margin and the gates still read z itself. Everything else is anchor_walk11.py's main (the frontier-pruned loader,
anchor_walk10's gates, anchor_walk8's loss, the rules, the selection, the reads), imported unchanged. Its record is
anchor_walk11's with this file's sha and pins added, and every 'zt' fit's learned exp(log_tz) in fit order.

    <train>:<base>/<model>zt[/<rule>][@seeds]   model lin, an anchor_walk6 operator or an anchor_walk10 gate (not text)
    python outputs/mp_approx_2wiki_anchor/host/anchor_walk12.py --dataset hotpotqa --variants x4+x5+x6:A256-1/linzt/mp --kd 0
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_walk11 as AW11  # noqa: E402

AW10, AW8, AW7, AW6, A16, AW = AW11.AW10, AW11.AW8, AW11.AW7, AW11.AW6, AW11.A16, AW11.AW
G2 = AW8.G2
AW11_SHA = "e782089f33cae0016ad4bbf09f14e092b0da7eaf31932c5d437d42f274d0f99b"
_parse10, _make10, _needs11, _scores = AW10.parse, AW10.make_for, AW11.needs, G2.scores
MADE = []


def split_zt(model):
    return (model[:-2], True) if model.endswith("zt") and len(model) > 2 else (model, False)


def parse(name, train_looks):
    """anchor_walk10.parse; a model field ending in 'zt' is parsed as its base model and the suffix put back."""
    head, at, seeds = name.partition("@")
    train, sep, rest = head.partition(":")
    parts = rest.split("/")
    if sep and len(parts) >= 2:
        base, zt = split_zt(parts[1])
        if zt:
            if base.removesuffix("sc") in AW7.TXT:
                raise SystemExit(f"{name}: zt is not read with the text models")
            parts[1] = base
            sp = _parse10(train + ":" + "/".join(parts) + (at + seeds if at else ""), train_looks)
            sp["model"] = sp["model"] + "zt"
            return sp
    return _parse10(name, train_looks)


def make_for(model, ntp, K, phi):
    base, zt = split_zt(model)
    if not zt:
        return _make10(model, ntp, K, phi)
    mk = _make10(base, ntp, K, phi)

    def make():
        m = mk()
        m.log_tz = torch.nn.Parameter(torch.zeros(()))
        MADE.append((model, m))
        return m
    return make


def needs(argv):
    """anchor_walk11.needs with each model's 'zt' suffix taken off before the gate test."""
    ds = argv[argv.index("--dataset") + 1] if "--dataset" in argv else "2wiki"
    vs = argv[argv.index("--variants") + 1].split(",")
    specs = [parse(v, AW6.TRAIN[ds]) for v in vs]
    return max(s["max_len"] for s in specs), any(split_zt(s["model"])[0] in AW10.GATES for s in specs)


def scores(model, PX):
    """l16_look_gate2.scores, plus (exp(log_tz) - 1) z on the finite pool nodes for a model with log_tz."""
    s, gold = _scores(model, PX)
    lt = getattr(model, "log_tz", None)
    if lt is None:
        return s, gold
    z = PX[0][9]
    fin = torch.isfinite(z)
    return torch.where(fin, s + (torch.exp(lt) - 1.0) * torch.where(fin, z, torch.zeros_like(z)), s), gold


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if AW.sha(Path(AW11.__file__)) != AW11_SHA:
        raise SystemExit("anchor_walk11.py is not the pinned file")
    ds = argv[argv.index("--dataset") + 1] if "--dataset" in argv else "2wiki"
    if "--out" not in argv:
        argv += ["--out", str(HERE / f"anchor_walk12_{ds}.json")]
    out = Path(argv[argv.index("--out") + 1])
    MADE.clear()
    AW10.parse, AW10.make_for, AW11.needs, G2.scores = parse, make_for, needs, scores
    try:
        AW11.main(argv)
    finally:
        AW10.parse, AW10.make_for, AW11.needs, G2.scores = _parse10, _make10, _needs11, _scores
    res = json.loads(out.read_text(encoding="utf-8"))
    res["look"] = "anchor_walk12"
    res["pins"]["anchor_walk11"] = AW11_SHA
    res["pins"]["anchor_walk11_record_sha256_field"] = res["script_sha256"]
    res["script_sha256"] = AW.sha(Path(__file__))
    res["z_temperature_in_fit_order"] = [{"model": nm, "tz": round(float(torch.exp(m.log_tz).item()), 4)} for nm, m in MADE]
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
