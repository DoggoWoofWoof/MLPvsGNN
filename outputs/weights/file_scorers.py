"""File the weight count of the next-hop scorer behind zrckpn's NEXT columns: reads the JSON that count_scorers.py printed
(its stdout, saved as a file) and weights.json (the two models' counts, filed earlier) and writes scorers.json and scorers.md
beside this script. Reads nothing else; trains and scores nothing.

    python -I outputs/weights/file_scorers.py <count_scorers stdout file>
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = ("WQ-musique", "WQ-hotpotqa")
FOLDS = 5
PER_SCORER = 296481         # the saved "bil" form (ltwdiag.build_model), rank 32 at d 1536: net 1,569 + V 147,456 + U 147,456
COLUMNS = ["next_1", "next_2", "next_3", "next_4", "next_5", "next_max", "next_n"]
FITS = {f"scr-ltm-zrckpn-s{s}{h}": ("WQ-hotpotqa" if h else "WQ-musique") for s in (0, 1, 2) for h in ("", "-hp")}
GNN_FIT = "scr-ml1-zg6r-s0"
GNN_TRAIN_WIDTHS = ("rank", "dense_cos", "topo_STRUCT", "depth_STRUCT", "SEMB", "SEED", "WALK", "DISTS", "WALKF")


def main(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    assert sorted(raw["splits"]) == sorted(SPLITS) and sorted(raw["fits"]) == sorted(FITS), "unexpected splits or fits"
    shapes = set()
    split_out = {}
    for split in SPLITS:
        s = raw["splits"][split]
        assert s["form"] == "bil" and [f["fold"] for f in s["folds"]] == list(range(FOLDS)), split
        for f in s["folds"]:
            assert f["form"] == "bil" and f["d"] == 1536 and f["tensors"] == 8, (split, f["fold"])
            assert f["elements"] == f["recorded_parameters"] == PER_SCORER, (split, f["fold"], f["elements"])
            assert f["file_sha256"] == f["recorded_sha256"], (split, f["fold"], "file differs from its fit record")
            assert f["mu_len"] == f["sd_len"] == 14, (split, f["fold"])
            assert sum(n for _, n in f["per_tensor"].values()) == PER_SCORER
            shapes.add(json.dumps(f["per_tensor"], sort_keys=True))
        split_out[split] = {"fit_record_sha256": s["fit_record_sha256"], "form": s["form"],
                            "fold_file_sha256": [f["file_sha256"] for f in s["folds"]]}
    assert len(shapes) == 1, "the ten fold files do not share one set of shapes"
    per = json.loads(next(iter(shapes)))
    parts = {"net (14 -> 32 -> 32 -> 1, with biases)": sum(n for k, (_, n) in per.items() if k.startswith("lin.")),
             "V (3 x 1536 x 32)": per["V"][1], "U (3 x 1536 x 32)": per["U"][1]}
    assert sum(parts.values()) == PER_SCORER and parts["net (14 -> 32 -> 32 -> 1, with biases)"] == 1569
    for name, split in FITS.items():
        f = raw["fits"][name]
        assert f["split"] == split and f["fit_sha256"] == split_out[split]["fit_record_sha256"], (name, f["split"])
        assert f["columns"] == COLUMNS and f["train_width_NEXT"] == len(COLUMNS), name
    w = json.loads((HERE / "weights.json").read_text(encoding="utf-8"))
    mlp, gnn = w["summary"]["zrckpn"]["elements"], w["summary"]["zg6r"]["elements"]
    l1 = {r["fit"]: r["per_tensor"]["l1.weight"][0] for r in w["fits"]}
    assert (mlp, gnn) == (226950, 404955)
    assert {l1[n][1] for n in FITS} == {324} and {l1[n][1] for n in l1 if n not in FITS} == {309}, "first-layer widths changed"
    one, five = mlp + PER_SCORER, mlp + FOLDS * PER_SCORER
    assert (one, five) == (523431, 1709355)
    out = {"read": "5 Nov 2026, the saved fold files of the next-hop scorer (read-only, on the host)",
           "script": "outputs/weights/count_scorers.py",
           "script_sha256": hashlib.sha256((HERE / "count_scorers.py").read_bytes()).hexdigest(),
           "per_scorer": {"elements": PER_SCORER, "tensors": 8, "parts": parts, "shapes": per,
                          "constants_not_weights": "feature mean and spread, 14 numbers each, stored with every fold file"},
           "splits": split_out, "folds_per_split": FOLDS, "scorer_files": FOLDS * len(SPLITS),
           "fits": {n: {"split": raw["fits"][n]["split"], "fit_record_sha256": raw["fits"][n]["fit_sha256"]} for n in FITS},
           "next_columns": COLUMNS,
           "gnn_of_record_next_block": {"fit_read": GNN_FIT, "widths": list(GNN_TRAIN_WIDTHS),
                                        "read_from": "outputs/screen/fits/scr-ml1-zg6r-s0/train.json, variants.p.widths"},
           "sizes": {"zrckpn": mlp, "zrckpn_plus_one_scorer": one, "zrckpn_plus_five_fold_fits_of_one_split": five,
                     "zg6r": gnn, "zrckpn_plus_one_scorer_over_zg6r": round(one / gnn, 3)}}
    (HERE / "scorers.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    p = list(parts.items())
    lines = ["# Weight count of the next-hop scorer behind zrckpn's NEXT columns", "",
             "Read on 5 Nov 2026 from the saved fold files of the scorer (`outputs/weights/count_scorers.py`, read-only). "
             "The two models' own counts are in weights.md (same folder), unchanged.", "",
             "| part | weights |", "|---|---:|",
             f"| {p[0][0]} | {p[0][1]:,} |", f"| {p[1][0]} | {p[1][1]:,} |", f"| {p[2][0]} | {p[2][1]:,} |",
             f"| **one scorer** | **{PER_SCORER:,}** |", "",
             "The feature mean and spread stored with each fold file (14 numbers each) are constants, not weights.", "",
             "## Which scorers a fit reads", "",
             f"- Each split has {FOLDS} fold fits of this one shape ({FOLDS * len(SPLITS)} files in all, one set of shapes). "
             f"WQ-musique's fit record is {split_out['WQ-musique']['fit_record_sha256'][:12]}, WQ-hotpotqa's is "
             f"{split_out['WQ-hotpotqa']['fit_record_sha256'][:12]}; every fold file matches the sha256 its record names.",
             "- A question reads exactly one scorer, never an average of the five (docs/LTM_NEXT_HOP_MLP.md section 2). "
             "A training question's columns come from the fold fit that never saw its component. An evaluation question's "
             f"columns come from the fold fit its index draws, uniform over the {FOLDS}. Over a whole fit, all {FOLDS} fold "
             "fits of its split are used.",
             "- The six zrckpn fits are the same in this respect. Seeds 0, 1 and 2 on WQ-musique read WQ-musique's five; "
             "the three `-hp` fits read WQ-hotpotqa's five. Each fit's read record names its split's fit record by sha256, "
             "and it matches the files read here. Each fit's NEXT block is 7 columns "
             f"({', '.join(COLUMNS)}).", "",
             "## Sizes side by side", "",
             "| system | weights |", "|---|---:|",
             f"| zrckpn alone (weights.md) | {mlp:,} |",
             f"| zrckpn and the one scorer a question reads | {one:,} |",
             f"| zrckpn and all {FOLDS} fold fits of one split | {five:,} |",
             f"| zg6r (weights.md) | {gnn:,} |", "",
             f"weights.md's \"1.78 times\" counts the MLP alone. With the one scorer its NEXT columns need, the MLP's side "
             f"holds {one / gnn:.2f} times the GNN's weights ({one:,} against {gnn:,}). The GNN of record reads no NEXT block "
             "(its saved input widths are rank, dense_cos, topo_STRUCT, depth_STRUCT, SEMB, SEED, WALK, DISTS and WALKF, "
             "zrckpn's list without NEXT), so it has no scorer to add. NEXT widens zrckpn's first layer by 15 inputs "
             f"(docs/LTM_NEXT_HOP_MLP.md section 9): its first layer is 324 wide in all six fits against {309} in the six "
             "zg6r fits; those first-layer weights are already inside the 226,950. Neither count includes the frozen text "
             "encoder.", ""]
    (HERE / "scorers.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("filed per scorer", PER_SCORER, "| one", one, "| five", five, "| ratio", out["sizes"]["zrckpn_plus_one_scorer_over_zg6r"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
