"""Reporting sidecar for the Universal-v2 V2_GATE / V2_HELD_CONFIRMATION split.

The review of the declaration (amendment 1) asked, before any v2 code: do
query families cross the two halves of an eval population under the declared
parity split, and if meaningful families exist, hash the family instead of the
query id. This script measures it and files the reference numbers of the
frozen M3B arrays on the amended halves. Nothing here touches a score, a weight
or a rule; it reads the served package read-only (same loader as M3B) and the
frozen M3B eval records.

    python scripts/universal_v2_split_audit.py            # writes outputs/universal_v2/split_audit.json

Family keys measured per dataset: the normalised question text (exact
paraphrase duplicates); the dataset-native problem family (metaqa: topic entity
x question type; 2wiki / squad: the sorted gold node set); the coarse family
(metaqa question type, 2wiki type, squad article) which is a template, not a
problem, and is reported for context only.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from m3b_run import load_script, CONFIG, OUT as M3B_OUT  # noqa: E402

OUT = ROOT / "outputs" / "universal_v2"
TRIO = ("metaqa", "2wiki", "squad")
SCORERS = {"gat_universal_v1_s0": "gat_universal_v1__H128_L2__s0", "gat_universal_v1_s1": "gat_universal_v1__H128_L2__s1",
           "gat_universal_v1_s2": "gat_universal_v1__H128_L2__s2", "gat_no_mp_v1_s0": "gat_no_mp_v1__H128_L2__s0",
           "qls_u_sota_v1_s0": "qls_u_sota_v1__H128__s0", "gat_universal_v1_s0_STRUCT": "gat_universal_v1__H128_L2__s0__STRUCT",
           "fixed_rrf": "fixed:rrf"}
METRICS = ("recall@5", "hit@1", "full_coverage@5")


def parity_even(key: str) -> bool:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 2 == 0


def family_key(dataset: str, row: dict) -> str:
    """The amended split key: metaqa keeps the query id (its problem families
    were measured negligible); the passage graphs hash the sorted gold set."""
    if dataset in ("metaqa", "webqsp"):
        return row["query_id"]
    return dataset + ":" + ",".join(sorted(str(g) for g in row["gold_node_ids"]))


def in_v2_gate(dataset: str, row: dict) -> bool:
    return parity_even(family_key(dataset, row))


def normalised_text(row: dict) -> str:
    text = row.get("question_plain") or row["question"]
    return " ".join(re.sub("[^a-z0-9 ]", " ", text.lower()).split())


def crossing(ids: list[str], key_of, gate: dict) -> dict:
    members = defaultdict(list)
    for q in ids:
        members[key_of(q)].append(q)
    cross = [g for g, qs in members.items() if len({gate[q] for q in qs}) == 2]
    n_cross = sum(len(members[g]) for g in cross)
    return {"groups": len(members), "multi_member_groups": sum(1 for qs in members.values() if len(qs) > 1),
            "groups_crossing_halves": len(cross), "queries_in_crossing_groups": n_cross,
            "fraction_of_queries_in_crossing_groups": round(n_cross / len(ids), 4),
            "max_group_size": max(len(qs) for qs in members.values())}


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    m3b_compile = load_script("m3b_compile")
    m3a, canonical, served, freeze = m3b_compile.open_package(cfg)
    record = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "freeze_RECORD_SHA256": freeze["RECORD_SHA256"],
              "what": "query-family crossing under the as-declared query-id parity split, the amended family split, "
                      "and the frozen M3B reference numbers on the amended halves; reporting only",
              "per_dataset": {}}
    for name in TRIO:
        ds = canonical.Dataset(name, root=str(served))
        by = {r["query_id"]: r for r in ds.queries("dev")}
        ids = json.loads((M3B_OUT / "eval" / f"{name}_query_ids.json").read_text(encoding="utf-8"))
        z = np.load(M3B_OUT / "eval" / f"{name}.npz")
        id_gate = {q: parity_even(q) for q in ids}
        text = {q: normalised_text(by[q]) for q in ids}
        gold = {q: name + ":" + ",".join(sorted(str(g) for g in by[q]["gold_node_ids"])) for q in ids}
        families = {"normalised_text": lambda q: text[q], "gold_set": lambda q: gold[q]}
        if name == "metaqa":
            families["topic_entity_x_qtype"] = lambda q: str(by[q]["topic_entity_node_id"]) + "|" + str(by[q]["qtype"])
            families["qtype_template_context_only"] = lambda q: str(by[q]["qtype"])
        elif name == "2wiki":
            families["type_context_only"] = lambda q: str(by[q]["type"])
        else:
            families["article_context_only"] = lambda q: str(by[q]["title"])
        audit = {f: crossing(ids, key_of, id_gate) for f, key_of in families.items()}
        gate = np.asarray([in_v2_gate(name, by[q]) for q in ids], dtype=bool)
        gate_of = {q: bool(gate[i]) for i, q in enumerate(ids)}
        problem_family = families.get("topic_entity_x_qtype", families["gold_set"])
        amended = {"split_key": "query_id" if name == "metaqa" else "dataset:sorted gold_node_ids", "V2_GATE": int(gate.sum()),
                   "V2_HELD_CONFIRMATION": int((~gate).sum()),
                   "agreement_with_query_id_parity": round(float(np.mean([gate_of[q] == id_gate[q] for q in ids])), 4),
                   "problem_family_crossing_under_the_amended_split": crossing(ids, problem_family, gate_of)}
        gip = z["fixed:rrf/gold_in_pool"].astype(float)
        gt = z["fixed:rrf/gold_total"].astype(float)
        r5c = np.minimum(gip, 5) / np.maximum(gt, 1)
        ceilings = {"hit_ceiling_any_gold_in_pool": {"V2_GATE": round(float((gip[gate] >= 1).mean()), 4),
                                                     "V2_HELD_CONFIRMATION": round(float((gip[~gate] >= 1).mean()), 4)},
                    "recall_ceiling@5": {"V2_GATE": round(float(r5c[gate].mean()), 4), "V2_HELD_CONFIRMATION": round(float(r5c[~gate].mean()), 4)}}
        refs = {}
        for label, scorer in SCORERS.items():
            refs[label] = {m: {"V2_GATE": round(float(z[f"{scorer}/{m}"][gate].mean()), 4),
                               "V2_HELD_CONFIRMATION": round(float(z[f"{scorer}/{m}"][~gate].mean()), 4)} for m in METRICS}
        if name == "metaqa":
            hop = np.asarray([q.split(":")[1] for q in ids])
            refs["by_hop_V2_GATE"] = {}
            for h in ("1hop", "2hop", "3hop"):
                m = gate & (hop == h)
                cell = {"queries": int(m.sum()), "hit_ceiling": round(float((gip[m] >= 1).mean()), 4)}
                for label, scorer in SCORERS.items():
                    if label != "fixed_rrf":
                        cell[label + "/hit@1"] = round(float(z[f"{scorer}/hit@1"][m].mean()), 4)
                refs["by_hop_V2_GATE"][h] = cell
        record["per_dataset"][name] = {"queries": len(ids), "ids_sha256": hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest(),
                                       "crossing_under_query_id_parity": audit, "amended_split": amended,
                                       "ceilings_frozen_pool": ceilings, "frozen_m3b_references": refs}
        print(f"== {name}: {len(ids)} queries; amended split {amended['V2_GATE']} / {amended['V2_HELD_CONFIRMATION']} ({amended['split_key']})", flush=True)
        for f, a in audit.items():
            print(f"   {f}: {a['groups']} groups, {a['fraction_of_queries_in_crossing_groups']:.4f} of queries in groups crossing the id-parity halves", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "split_audit.json"
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"wrote {path} sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
