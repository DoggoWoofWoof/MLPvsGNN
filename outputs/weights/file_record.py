"""File the weight counts of the two models of record: reads the JSON that count_weights.py printed (its stdout, saved
as a file) and writes weights.json and weights.md beside this script. Reads nothing else; trains and scores nothing.

    python -I outputs/weights/file_record.py <count_weights stdout file>
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# zg6r's encoder: zrm's block encoder and chain match, copied from the frozen zspk fit (zg6rw.ENC_NAMES, 224,737 numbers)
ENC_NAMES = ("U", "V", "cm_a", "cm_b", "cm_d", "cm_kappa", "cm_dir", "cm_bias", "cm_pi_w", "cm_pi_b", "cm_beta",
             "l1.weight", "l1.bias", "l2.weight", "l2.bias")
ENC_ELEMENTS = 224737       # zg6rw.ENC_ELEMENTS
GNN_OWN_FILED = 180218      # docs/G6RW_G6R_UNDER_WQ.md section 2: "180,218 parameters at hidden 128"
MLP_FITS = [f"scr-ltm-zrckpn-s{s}{h}" for s in (0, 1, 2) for h in ("", "-hp")]
GNN_FITS = [f"scr-ml1-zg6r-s{s}{h}" for s in (0, 1, 2) for h in ("", "-hp")]


def load(path):
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    return json.loads(raw[raw.index("["):raw.rindex("]") + 1])


def group_mlp(per):
    """zrckpn's tensors by part."""
    parts = {"lookup tables U, V": ["U", "V"],
             "chain match (cm_*)": [k for k in per if k.startswith("cm_")],
             "scoring layers l1, l2, out": ["l1.weight", "l1.bias", "l2.weight", "l2.bias", "out.weight", "out.bias"],
             "pointwise KT head (kt_*)": [k for k in per if k.startswith("kt_")],
             "scale (base_w)": ["base_w"]}
    out = {name: sum(per[k][1] for k in ks) for name, ks in parts.items()}
    assert sum(out.values()) == sum(v[1] for v in per.values()), "zrckpn: a tensor is in no part"
    return out


def group_gnn(per):
    """zg6r's tensors: the copied encoder and its own stack."""
    enc = sum(per[k][1] for k in ENC_NAMES)
    own = sum(v[1] for k, v in per.items() if k not in ENC_NAMES)
    assert enc == ENC_ELEMENTS and own == GNN_OWN_FILED, (enc, own)
    return {"encoder (15 tensors, copied from the zspk fit)": enc,
            "own message-passing stack (g6_*, g6r_*)": own,
            "own tensors": sum(1 for k in per if k not in ENC_NAMES)}


def main(path):
    recs = {r["fit"]: r for r in load(path)}
    assert sorted(recs) == sorted(MLP_FITS + GNN_FITS), sorted(recs)
    out = {"read": "5 Nov 2026, the saved p@swa state of each fit's models.pt (read-only, on the host)",
           "script": "outputs/weights/count_weights.py", "script_sha256": hashlib.sha256((HERE / "count_weights.py").read_bytes()).hexdigest(),
           "weights": "the elements of every tensor in the saved state (each tensor is named with its shape in per_tensor)"}
    summary = {}
    for model, fits, grp in (("zrckpn", MLP_FITS, group_mlp), ("zg6r", GNN_FITS, group_gnn)):
        totals = {recs[f]["elements"] for f in fits}
        shapes = {json.dumps(recs[f]["per_tensor"], sort_keys=True) for f in fits}
        assert len(totals) == 1 and len(shapes) == 1, (model, totals)
        r0 = recs[fits[0]]
        summary[model] = {"elements": r0["elements"], "tensors": r0["tensors"], "hidden": r0["hidden"],
                          "same_in_all_fits": len(fits), "parts": grp(r0["per_tensor"]), "fits": fits}
    out["summary"] = summary
    out["ratio_gnn_over_mlp"] = round(summary["zg6r"]["elements"] / summary["zrckpn"]["elements"], 3)
    out["fits"] = [recs[f] for f in MLP_FITS + GNN_FITS]
    (HERE / "weights.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    m, g = summary["zrckpn"], summary["zg6r"]
    lines = ["# Weight counts of the two models of record", "",
             "Read on 5 Nov 2026 from each fit's saved `p@swa` state (`outputs/weights/count_weights.py`, read-only); "
             "the six fits of each model (seeds 0 to 2, both training sets) hold the same tensors, so the count is one number.",
             "", "| model | weights | tensors | parts |", "|---|---:|---:|---|",
             f"| zrckpn (the MLP of record) | **{m['elements']:,}** | {m['tensors']} | "
             + "; ".join(f"{k} {v:,}" for k, v in m["parts"].items()) + " |",
             f"| zg6r (the GNN of record) | **{g['elements']:,}** | {g['tensors']} | "
             + "; ".join(f"{k} {v:,}" for k, v in g["parts"].items() if k != "own tensors") + " |",
             "", f"The GNN's saved model holds {out['ratio_gnn_over_mlp']:.2f} times the MLP's weights. "
             f"Its own stack alone ({GNN_OWN_FILED:,}, the count in docs/G6RW_G6R_UNDER_WQ.md) is "
             f"{GNN_OWN_FILED / m['elements']:.2f} times the MLP's; the rest of its state is an encoder of the MLP's shape "
             "(lookup tables, chain match, two layers), copied from the zspk fit and carried into every read.",
             "Quote the totals (226,950 and 404,955) when the two sizes are set side by side; the 180,218 alone leaves the "
             "GNN's encoder out.", ""]
    (HERE / "weights.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("filed", {k: v["elements"] for k, v in summary.items()}, "ratio", out["ratio_gnn_over_mlp"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
