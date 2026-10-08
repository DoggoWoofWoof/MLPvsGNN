"""Screens, seventeenth round (docs/SCREENS.md): zrm with rmatch's chain entries kept by contribution. rmatch's build
(rmatch.question_entries) keeps, per row and seed bucket, the 64 entries with the most walk mass, and mass is set by the
graph's degrees, not by the question. On webqsp that keeps 0.132 of the match's starting sum on gold rows and 0.305 on
the others (outputs/diag/chaincov-webqsp-s1eval). zrc keeps, per row and bucket, the 64 entries with the largest
w0(c) m_c(v), w0 the match at its start:
    w0(c) = (1/3) prod over the chain's steps of sigmoid(zcos_q(r_k))
(rmatch.ChainMatch with kappa 1 and every other match parameter 0; zcos_q in float64 from the look's question embedding,
as outputs/mp_unified/chaincov.py computes it). The walk, its chain cap, the renormalised mass, the arrays and the model
are rmatch's and zrm's, unchanged; only which entries a row keeps changes. The diagnosis: on webqsp's s1eval carve R@5 of
the rows ranked by the starting match alone is 0.2109 with the build's entries, 0.2138 with every entry and 0.2493 with
these; on metaqa's no cap binds (0.2938 for all three).

Why no training: webqsp never trains, and metaqa is the only typed graph a fit trains on. When zrc's build of metaqa's fit
carve is rmatch's array for array (`identity`), zrc's training is zrm's bit for bit (the same arrays, initialisation and
batches), so zrm's fits are zrc's: each is copied (`fork`: its models and training records unchanged, its arm zrc) and
read on zrc's carves. A read of a graph without typed relations must then be zrm's bit for bit, and so must metaqa's when
its s1eval build is identical too (`compare` files that check beside the comparison). If the identity fails, nothing is
forked or read.

    python outputs/mp_unified/zrc.py build --dataset webqsp --carve s1eval --host
    python outputs/mp_unified/zrc.py identity --carves metaqa=fit --out outputs/zrc/identity
    python outputs/mp_unified/zrc.py fork --src outputs/screen/fits/scr-zrm --name scr-zrc
    python outputs/mp_unified/zrc.py read --name scr-zrc --device cuda --host
    python outputs/mp_unified/zrc.py compare --new outputs/screen/fits/scr-zrc \\
        --base outputs/screen/fits/scr-zrm,outputs/screen/fits/scr-zret,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zrc
    python outputs/mp_unified/zrc.py pair --screens outputs/screen/scr-zrc.json,outputs/screen/scr-zrc-hp.json \\
        --out outputs/screen/scr-zrc-pair
    python outputs/mp_unified/zrc.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zrc-pair.json \\
        --out outputs/screen/scr-zrc-pair-recall
    python outputs/mp_unified/zrc.py grade --full-root outputs/full_zrc \\
        --reuse L-musique=outputs/screen/scr-zrc.json,L-hotpotqa=outputs/screen/scr-zrc-hp.json
    python outputs/mp_unified/zrc.py --selftest
Each read is decided against zrm's fit of its split: relz.py's pair, re-call and grade, run under zrc's name with zrm's
fits in place of rel's (zrm's screen fits scr-zrm and scr-zrm-hp, its full run's fits on the other splits). The
re-call's base R@5 and the full run's nullx.py re-grade take zrm's fits compared with step 1's
(outputs/zrc/base-zrm-<split>.json, `compare` with step 1's fit as the only base).

Speed: the rule changes which entries a row keeps, at build time, from the question's embedding and its pool's
relations; it is as cold as zrm's (8216ffe): any latency figure times each question from scratch, the walk, the
selection and the forward included.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import chaincov as CC  # noqa: E402
import rmatch as RM  # noqa: E402
import zrm as ZM  # noqa: E402

Z, R = RM.Z, RM.R
S2, S = RM.S2, RM.S
LG, LC, LM = S.LG, S.LC, S.LM
SR = Z.SR
log = S.log
ARM, BASE_ARM = "zrc", "zrm"
CH_OUT = RM.ROOT / "outputs" / "zrc" / "cache"
RULE = "per row and seed bucket the k_row entries with the largest w0(c) m_c(v), ties to the earlier chain"
PRIOR = ("w0(c) = (1/3) prod_k sigmoid(zcos_q(r_k)): rmatch.ChainMatch at its start, zcos in float64 from the look's "
         "q_emb (chaincov.zcos_of)")

ChainCarveZRC = RM.chain_carve(S.ARMS["base"][1], CH_OUT)
ChainCarveZRC.__name__ = ChainCarveZRC.__qualname__ = "ChainCarveZRC"
S.ARMS.update({ARM: (ZM.ZRM, ChainCarveZRC)})


# ── the entries kept by contribution ─────────────────────────────────────────


def step_of(q, U, pool_rels):
    """(2 n_rel,): log sigmoid(zcos_q(z // 2)) for every typed step z = 2 r + d."""
    return CC.log_sigmoid(np.repeat(CC.zcos_of(q, U, pool_rels), 2))


def contrib_entries(n, hl, tl, sl, seeds_by_bucket, kz, step, hops=RM.HOPS, cap=RM.MAX_CHAINS, k_row=RM.K_ROW):
    """rmatch.question_entries with its row cap by contribution: per row and bucket the k_row entries with the largest
    w0(c) m_c(v) (ties to the earlier chain), in chain order, w0(c) = (1/3) exp(sum over the chain's steps of step[z]).
    The walk, its cap, the renormalised mass and the statistics are question_entries'."""
    g = RM.typed_graph(n, hl, tl, sl, kz)
    rows, zs, bs, ms, stats = [], [], [], [], []
    for b, seeds in enumerate(seeds_by_bucket):
        st = {"seeds": int(seeds.size), "capped": 0, "chains": 0, "before_cap": 0, "entries": 0, "mass": 0.0,
              "dropped_mass": 0.0}
        levels, capped = RM.walk(n, g, seeds, hops, cap)
        st["capped"] = int(capped)
        J = sum(t.shape[0] for t, _o, _v, _m in levels)
        if J:
            zc = np.full((J, hops), -1, np.int64)
            at = 0
            for t, _o, _v, _m in levels:
                zc[at:at + t.shape[0], :t.shape[1]] = t
                at += t.shape[0]
            cid = np.repeat(np.arange(J), np.concatenate([np.diff(o) for _t, o, _v, _m in levels]))
            node = np.concatenate([v for _t, _o, v, _m in levels])
            mass = np.concatenate([m for _t, _o, _v, m in levels])
            seed = np.zeros(n, bool)
            seed[seeds] = True
            keep = ~seed[node]
            ck = cid[keep]
            live = np.bincount(ck, minlength=J) > 0
            row = node[keep]
            mass = mass[keep] / np.bincount(ck, weights=mass[keep], minlength=J)[ck]
            cid = (np.cumsum(live) - 1)[ck]
            zc = zc[live]
            st["chains"] = int(live.sum())
        if J and row.size:
            lw = np.where(zc >= 0, step[np.maximum(zc, 0)], 0.0).sum(1) + np.log(1.0 / 3.0)
            key = np.exp(lw)[cid] * mass                      # chaincov.question_scores' contrib, bit for bit
            order = np.lexsort((cid, -key, row))
            rs = row[order]
            first = np.r_[True, rs[1:] != rs[:-1]]
            start = np.maximum.accumulate(np.where(first, np.arange(rs.size), 0))
            keep = np.zeros(row.size, bool)
            keep[order[np.arange(rs.size) - start < k_row]] = True
            st.update({"before_cap": int(row.size), "entries": int(keep.sum()), "mass": float(mass.sum()),
                       "dropped_mass": float(mass[~keep].sum())})
            rows.append(row[keep])
            zs.append(zc[cid[keep]])
            bs.append(np.full(int(keep.sum()), b, np.int64))
            ms.append(mass[keep])
        stats.append(st)
    if not rows:
        return (np.zeros(0, np.int64), np.zeros((0, hops), np.int64), np.zeros(0, np.int64), np.zeros(0)), stats
    return (np.concatenate(rows), np.concatenate(zs), np.concatenate(bs), np.concatenate(ms)), stats


# ── build (rmatch.build with contrib_entries) ────────────────────────────────


def against_rmatch(ds, carve, part, step1_sha, shas, rm_root=RM.CH_OUT):
    """Each array against rmatch's build of the same part: IDENTICAL or DIFFERENT; None when rmatch's record is not on
    disk here."""
    rp = Path(rm_root) / ds / carve / part / "record.json"
    if not rp.exists():
        return None
    rr = json.loads(rp.read_text(encoding="utf-8"))
    same = rr.get("step1_record_sha256") == step1_sha
    return {k: "IDENTICAL" if same and rr["arrays"].get(k, {}).get("sha256") == v else "DIFFERENT"
            for k, v in shas.items()}


def build(ds, carve, out_root=CH_OUT, cache_root=LC.OUT, look_root=LM.LOOK, rel_dir=RM.REL_DIR, placement=None,
          rm_root=RM.CH_OUT, k_row=RM.K_ROW):
    """rmatch.build with contrib_entries in place of question_entries, each question's steps from its embedding and
    its pool's relations. The arrays and record are rmatch's, with the rule, the prior and each array against rmatch's
    build of the same part."""
    t0 = time.time()
    if ds not in RM.TYPED:
        raise SystemExit(f"zrc build: {ds} has no typed relations (the arm builds {RM.TYPED})")
    d, files, _ids, head = LC.look_chunks(ds, carve, look_root)
    rel = RM.look_relations(d)
    n_rel, k_rel = int(rel["n_relations"]), int(rel["k_rel"])
    ef = Path(rel_dir) / f"{ds}_rel_embeddings.npy"
    E = np.load(ef).astype(np.float64)
    if E.shape != (n_rel, RM.Q_DIM):
        raise SystemExit(f"{ef}: {E.shape}, the look has {n_rel} relations")
    U = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    relations = {"n_relations": n_rel, "kz": 2 * n_rel, "offset": rel.get("offset"), "k_rel": k_rel,
                 "embeddings": ef.name, "embeddings_sha256": LC.sha_file(ef)}
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    summary = {}
    for p, r in zip(parts, recs):
        if r["look"]["carve_ids_sha256"] != head["carve_ids_sha256"] or r["look"]["records"] != head["records"]:
            raise SystemExit(f"{p}: built from another look than {d}")
        tp = time.time()
        lo, hi = r["chunks"]
        acc = {k: [] for k in RM.ARRAYS}
        n_all = []
        tot = [{"questions_with_seeds": 0, "capped": 0, "no_chain": 0, "chains": 0, "before_cap": 0, "entries": 0,
                "mass": 0.0, "dropped_mass": 0.0} for _ in range(RM.BUCKETS)]
        for ch in range(lo, hi):
            with np.load(files[ch]) as zf:
                c = {k: zf[k] for k in RM.KEYS + ("q_emb",)}
            qps = c["q_pool_size"].astype(np.int64)
            eo = np.r_[0, np.cumsum(c["q_edges"].astype(np.int64))]
            if int(eo[-1]) != c["e_u"].size or c["e_rel"].shape != (c["e_u"].size, k_rel):
                raise SystemExit(f"{files[ch]}: {c['e_u'].size} edges and e_rel {c['e_rel'].shape} against q_edges "
                                 f"{int(eo[-1])} and k_rel {k_rel}")
            if c["q_emb"].shape != (qps.size, RM.Q_DIM):
                raise SystemExit(f"{files[ch]}: q_emb {c['q_emb'].shape} for {qps.size} questions")
            for i in range(qps.size):
                n = int(qps[i])
                if n > np.iinfo(np.int16).max:
                    raise SystemExit(f"{files[ch]}: question {i} has a pool of {n} rows, above int16")
                hl, tl, sl = RM.row_triples(c, int(eo[i]), int(eo[i + 1]))
                if sl.size and int(sl.max()) >= n_rel:
                    raise SystemExit(f"{files[ch]}: question {i} has relation {int(sl.max())} of {n_rel}")
                sl0, bk = c["q_seed_local"][i], c["q_seed_bucket"][i]
                ok = sl0 >= 0
                if bool(((bk[ok] < 0) | (bk[ok] >= RM.BUCKETS)).any()) or bool((sl0[ok] >= n).any()):
                    raise SystemExit(f"{files[ch]}: question {i} has seeds {sl0[ok]} in buckets {bk[ok]}")
                seeds = [np.unique(sl0[ok & (bk == b)]).astype(np.int64) for b in range(RM.BUCKETS)]
                qr = np.unique(sl[sl >= 0])
                step = step_of(c["q_emb"][i], U, qr)
                (row, zz, bb, mm), qst = contrib_entries(n, hl, tl, sl, seeds, 2 * n_rel, step, k_row=k_row)
                acc["q_ent"].append(row.size)
                acc["ent_row"].append(row.astype(np.int16))
                acc["ent_z"].append(zz.astype(np.int16))
                acc["ent_b"].append(bb.astype(np.int8))
                acc["ent_m"].append(mm.astype(np.float16))
                acc["q_nrel"].append(qr.size)
                acc["q_rel"].append(qr.astype(np.int16))
                for b, s_ in enumerate(qst):
                    t_ = tot[b]
                    t_["questions_with_seeds"] += int(s_["seeds"] > 0)
                    t_["no_chain"] += int(s_["seeds"] > 0 and s_["chains"] == 0)
                    for k in ("capped", "chains", "before_cap", "entries", "mass", "dropped_mass"):
                        t_[k] += s_[k]
            n_all.append(qps)
        arrays = {"q_ent": np.asarray(acc["q_ent"], np.int64), "q_nrel": np.asarray(acc["q_nrel"], np.int64)}
        for k in ("ent_row", "ent_z", "ent_b", "ent_m", "q_rel"):
            arrays[k] = np.concatenate(acc[k])
        n_all = np.concatenate(n_all)
        if n_all.size != r["queries"] or int(n_all.sum()) != r["rows"]:
            raise SystemExit(f"{p}: {n_all.size} queries and {int(n_all.sum())} rows against {r['queries']} and {r['rows']}")
        if bool((arrays["ent_row"].astype(np.int64) >= np.repeat(n_all, arrays["q_ent"])).any()):
            raise SystemExit(f"{p}: an entry's row is outside its pool")
        if (p / "n.npy").exists():
            if not np.array_equal(np.asarray(LC.load_array(p, "n", r, verify=True)).astype(np.int64), n_all):
                raise SystemExit(f"{p}: the chunks' pool sizes are not step 1's cached n")
            against = "IDENTICAL"
        else:
            against = "step 1's arrays are not on disk here"
        out = Path(out_root) / ds / carve / p.name
        out.mkdir(parents=True, exist_ok=True)
        shas = {k: R.save_npy(out, k, v) for k, v in arrays.items()}
        for t_ in tot:
            t_["dropped_mass_share"] = round(t_["dropped_mass"] / t_["mass"], 6) if t_["mass"] else 0.0
            t_["mass"], t_["dropped_mass"] = round(t_["mass"], 3), round(t_["dropped_mass"], 3)
        qe = arrays["q_ent"]
        s1sha = LC.sha_file(p / "record.json")
        vs = against_rmatch(ds, carve, p.name, s1sha, shas, rm_root)
        rec = {"dataset": ds, "carve": carve, "part": p.name, "step1_record_sha256": s1sha,
               "chunks": [lo, hi], "queries": int(n_all.size), "rows": r["rows"], "entries": int(qe.sum()),
               "rel_entries": int(arrays["q_nrel"].sum()), "look": head["records"],
               "carve_ids_sha256": head["carve_ids_sha256"], "relations": relations,
               "walk": {"hops": RM.HOPS, "max_chains": RM.MAX_CHAINS, "k_row": k_row, "buckets": RM.BUCKETS},
               "rule": RULE, "prior": PRIOR, "buckets": tot,
               "entries_per_question": {"mean": round(float(qe.mean()), 1) if qe.size else 0.0,
                                        "p50": int(np.percentile(qe, 50)) if qe.size else 0,
                                        "p90": int(np.percentile(qe, 90)) if qe.size else 0,
                                        "max": int(qe.max()) if qe.size else 0},
               "n_against_step1": against, "against_rmatch": vs,
               "arrays": {k: {"dtype": v.dtype.str, "shape": list(v.shape), "sha256": shas[k]} for k, v in arrays.items()},
               "placement": placement or {"where": "laptop"}, "script_sha256": LC.sha_src(__file__),
               "rmatch_sha256": LC.sha_src(RM.__file__), "chaincov_sha256": LC.sha_src(CC.__file__),
               "seconds": round(time.time() - tp, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        R.write_json(out / "record.json", rec)
        same = None if vs is None else all(v == "IDENTICAL" for v in vs.values())
        summary[p.name] = {"queries": int(n_all.size), "entries": int(qe.sum()), "identical_to_rmatch": same}
        log(f"  {ds}/{carve} {p.name}: {n_all.size} queries, {int(qe.sum())} entries; against rmatch's build "
            f"{'not on disk' if vs is None else 'IDENTICAL' if same else vs}; bucket 0 dropped mass "
            f"{tot[0]['dropped_mass_share']}, bucket 1 {tot[1]['dropped_mass_share']} ({time.time() - t0:.0f}s)")
    log(f"zrc build {ds}/{carve}: {len(parts)} part(s) ({time.time() - t0:.0f}s)")
    return summary


# ── the identity gate and the fork ───────────────────────────────────────────


def identity(carves, out, root=CH_OUT, cache_root=LC.OUT):
    """IDENTICAL when every step-1 part of every carve has zrc's build and each of its arrays is rmatch's, else
    DIFFERENT (or MISSING); exit 0 only on IDENTICAL."""
    rows, verdict = [], "IDENTICAL"
    for ds, cv in carves:
        parts, _recs = LC.part_dirs(ds, cv, cache_root)
        for p in parts:
            f = Path(root) / ds / cv / p.name / "record.json"
            if not f.exists():
                rows.append({"carve": f"{ds}={cv}", "part": p.name, "status": "MISSING"})
                verdict = "MISSING"
                continue
            h = json.loads(f.read_text(encoding="utf-8"))
            vs = h.get("against_rmatch")
            st = ("MISSING" if vs is None or h.get("step1_record_sha256") != LC.sha_file(p / "record.json") else
                  "IDENTICAL" if all(v == "IDENTICAL" for v in vs.values()) else "DIFFERENT")
            rows.append({"carve": f"{ds}={cv}", "part": p.name, "status": st, "arrays": vs, "queries": h.get("queries"),
                         "record_sha256": LC.sha_file(f)})
            if st != "IDENTICAL" and verdict == "IDENTICAL":
                verdict = st
            elif st == "MISSING":
                verdict = "MISSING"
    rec = {"verdict": verdict, "carves": [f"{d}={c}" for d, c in carves], "rows": rows, "rule": RULE,
           "script_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    md = [f"# zrc's chain entries against rmatch's: **{verdict}**", "",
          "Each step-1 part of each carve: zrc's build (entries kept by contribution) against rmatch's (kept by mass), "
          "array for array. IDENTICAL on metaqa's fit carve means zrc trains as zrm, bit for bit.", "",
          "| carve | part | questions | status |", "|---|---|---:|---|"]
    md += [f"| {r['carve']} | {r['part']} | {r.get('queries', '')} | {r['status']} |" for r in rows]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zrc identity {rec['carves']}: {verdict}")
    return rec


def rel_path(p):
    p = Path(p)
    try:
        p = p.resolve().relative_to(RM.ROOT)
    except ValueError:
        pass
    return str(p).replace("\\", "/")


def fork(src, name, out_root=None, identity_file=CH_OUT.parent / "identity.json"):
    """zrm's fit src copied to out_root/name as zrc's: models.pt and train.json unchanged, screen.json zrm's with the
    arm zrc and where it came from. Refused unless the identity gate says IDENTICAL and src is a zrm fit; a fork that
    exists already passes only with src's models."""
    src = Path(src)
    out_root = Path(out_root) if out_root else src.parent
    idf = Path(identity_file)
    v = json.loads(idf.read_text(encoding="utf-8")).get("verdict") if idf.exists() else None
    if v != "IDENTICAL":
        raise SystemExit(f"zrc fork: the identity gate {idf} is {v}, not IDENTICAL; zrm's fits are not zrc's")
    sj = json.loads((src / "screen.json").read_text(encoding="utf-8"))
    if sj.get("arm") != BASE_ARM:
        raise SystemExit(f"zrc fork: {src} is arm {sj.get('arm')}, not {BASE_ARM}")
    msha = LC.sha_file(src / "models.pt")
    dst = out_root / name
    if dst.exists():
        if (dst / "models.pt").exists() and LC.sha_file(dst / "models.pt") == msha and \
                json.loads((dst / "screen.json").read_text(encoding="utf-8")).get("arm") == ARM:
            log(f"zrc fork: {dst} is {src}'s already")
            return 0
        raise SystemExit(f"zrc fork: {dst} exists and is not {src}'s fork")
    tmp = out_root / f"{name}.forking"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copy2(src / "models.pt", tmp / "models.pt")
    shutil.copy2(src / "train.json", tmp / "train.json")
    if LC.sha_file(tmp / "models.pt") != msha:
        raise SystemExit(f"zrc fork: {tmp / 'models.pt'} is not {src}'s")
    rec = dict(sj)
    rec.update({"arm": ARM, "name": name, "zrc_sha256": LC.sha_src(__file__),
                "zrc": {"forked_from": rel_path(src), "models_sha256": msha,
                        "screen_sha256": LC.sha_file(src / "screen.json"), "train_sha256": LC.sha_file(src / "train.json"),
                        "identity": rel_path(idf), "identity_sha256": LC.sha_file(idf), "cache": rel_path(CH_OUT),
                        "rule": RULE, "model": "zrm.ZRM (rmatch.ChainMatch over lean_screen3.ZRet)",
                        "carve": "rmatch.chain_carve over lean_gpu's carve, entries from zrc's cache"}})
    LC.write_json(tmp / "screen.json", rec)
    os.replace(tmp, dst)
    log(f"zrc fork: {src} -> {dst} (models {msha[:12]})")
    return 0


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zrc: {name} is arm {arm}, not {ARM}")
    return RM.read(argv)


# ── compare, and the reads that must be zrm's ────────────────────────────────


def same(new, base, out):
    """Each s1eval read of the fork against zrm's: IDENTICAL when every array is equal, else the arrays that differ and
    the questions whose p@swa top-5 count differs."""
    new, base = Path(new), Path(base)
    rows = []
    for f in sorted((new / "reads").glob("*__s1eval.npz")):
        N, B = np.load(f), np.load(base / "reads" / f.name)
        keys = sorted(set(N.files) | set(B.files))
        diff = [k for k in keys if k not in N.files or k not in B.files or not np.array_equal(N[k], B[k])]
        row = {"dataset": f.name.split("__")[0], "status": "IDENTICAL" if not diff else "DIFFERENT", "differ": diff}
        if diff and "top" in N.files and "top" in B.files and list(N["candidates"]) == list(B["candidates"]):
            k = [str(x) for x in N["candidates"]].index("p@swa")
            row["questions_top_differs"] = int((N["top"][k] != B["top"][k]).sum())
            row["questions"] = int(N["top"].shape[1])
        rows.append(row)
    rec = {"new": rel_path(new), "base": rel_path(base), "rows": rows, "script_sha256": LC.sha_src(__file__)}
    out = Path(out)
    LC.write_json(out.with_suffix(".json"), rec)
    md = [f"# {new.name}'s reads against {base.name}'s", "", "| dataset | status | arrays that differ | "
          "questions whose p@swa top-5 golds differ |", "|---|---|---|---:|"]
    md += [f"| {r['dataset']} | {r['status']} | {', '.join(r['differ'])} | {r.get('questions_top_differs', '')} |"
           for r in rows]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zrc same {new.name} vs {base.name}: " + "; ".join(f"{r['dataset']} {r['status']}" for r in rows))
    return rec


def compare(rest):
    """lean_screen's compare; when the new fit is zrc's and the first base zrm's, the reads checked against zrm's too
    (<out>-same.json)."""
    rc = S2.main(["compare"] + rest)
    cp = argparse.ArgumentParser(add_help=False)
    cp.add_argument("--new")
    cp.add_argument("--base")
    cp.add_argument("--out")
    a, _ = cp.parse_known_args(rest)
    b0 = Path(a.base.split(",")[0])
    arm = lambda p: json.loads((Path(p) / "screen.json").read_text(encoding="utf-8")).get("arm")  # noqa: E731
    if rc == 0 and a.out and arm(a.new) == ARM and arm(b0) == BASE_ARM:
        same(a.new, b0, f"{a.out}-same")
    return rc


# ── relz.py's pair, re-call and grade, decided against zrm's fits ────────────


ZRM_FITS = {"L-musique": ("screen", "fits", "scr-zrm"), "L-hotpotqa": ("screen", "fits", "scr-zrm-hp")}
ZRM_SCREENS = {sp: f"outputs/zrc/base-zrm-{sp}.json" for sp in ZRM_FITS}


def zrm_fit(split):
    """zrm's fit of a split: its screen fits on L-musique and L-hotpotqa, its full run's on every other split."""
    return ZRM_FITS.get(split, ("full_zrm", "fits", split))


def zrm_compares():
    """zrm's six fits compared with step 1's (nullx.py regrade's --base-compares)."""
    return {sp: f"outputs/zrc/base-zrm-{sp}.json" for sp in S2.SPLITS}


@contextlib.contextmanager
def on_zrm():
    """relz.py's records under zrc's name, each read decided against zrm's fit of its split; relz restored after."""
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, ZRM_FITS, ZRM_SCREENS, zrm_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrm's screen fits scr-zrm and scr-zrm-hp)"),
       ("(rel's screen fits)", "(zrm's screen fits)"),
       ("| rel R@5 | relz R@5 |", "| zrm R@5 | zrc R@5 |"),
       ("section 2 and the tenth round", "section 2 and the seventeenth round"),
       ("docs/FULL_ROUND10.md", "docs/FULL_ROUND17.md"))


def restamp(out):
    """A record relz.py wrote under zrc's name: zrm named as the base in its md, this file's sha added."""
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec["decided_against"] = "zrm's fit of each split"
        rec["zrc_sha256"] = LC.sha_src(__file__)
        LC.write_json(js, rec)


def pair(screens, out):
    with on_zrm():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, zrm_screens=None):
    with on_zrm():
        rec = Z.recall(null_files, pair_file, out, zrm_screens)
    restamp(out)
    return rec


def grade(full_root, reuse, out, check_fits=True):
    with on_zrm():
        rec = Z.grade(full_root, reuse, out, check_fits)
    restamp(out)
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def add_q_emb(tmp, rng):
    """The toy look's chunks with a question embedding each (rmatch's toy has none)."""
    for f in sorted((tmp / "look" / "metaqa" / "toy" / "chunks").glob("c*.npz")):
        with np.load(f) as zf:
            A = {k: zf[k] for k in zf.files}
        A["q_emb"] = rng.standard_normal((A["q_pool_size"].size, RM.Q_DIM)).astype(np.float16)
        np.savez(f, **A)


def selftest():
    t0 = time.time()
    rng = np.random.default_rng(17)
    # 1. with no row cap binding, contrib_entries is question_entries, array for array; with one, each row and bucket
    #    keeps the k_row largest w0 m (a brute force over every entry), in chain order
    binds = 0
    for t in range(40):
        q = RM.toy_question(rng, int(rng.integers(4, 12)), 4, rng.choice(4, int(rng.integers(2, 5)), replace=False))
        hl, tl, sl = RM.row_triples(q, 0, q["q_edges"])
        ok = q["q_seed_local"] >= 0
        seeds = [np.unique(q["q_seed_local"][ok & (q["q_seed_bucket"] == b)]) for b in range(RM.BUCKETS)]
        step = step_of(rng.standard_normal(16), rng.standard_normal((4, 16)), np.unique(sl[sl >= 0]))
        n = q["q_pool_size"]
        a, sa = RM.question_entries(n, hl, tl, sl, seeds, 8, k_row=10 ** 6)
        b_, sb = contrib_entries(n, hl, tl, sl, seeds, 8, step, k_row=10 ** 6)
        assert all(np.array_equal(x, y) for x, y in zip(a, b_)) and sa == sb
        row, zz, bb, mm = b_
        w0 = np.exp(np.where(zz >= 0, step[np.maximum(zz, 0)], 0.0).sum(1) + np.log(1.0 / 3.0))
        for k_row in (1, 2, 3):
            kr, kz_, kb, km = contrib_entries(n, hl, tl, sl, seeds, 8, step, k_row=k_row)[0]
            want = []
            for e in range(row.size):
                grp = np.flatnonzero((row == row[e]) & (bb == bb[e]))
                key = w0[grp] * mm[grp]
                rank = int(((key > key[grp == e]) | ((key == key[grp == e]) & (grp < e))).sum())
                if rank < k_row:
                    want.append(e)
            want = np.asarray(want, np.int64)
            binds += int(want.size < row.size)
            assert np.array_equal(kr, row[want]) and np.array_equal(kz_, zz[want]) and np.array_equal(kb, bb[want])
            assert np.array_equal(km, mm[want])
    assert binds > 0
    # 2. contrib_entries' key is chaincov's contrib, so its S0 over the kept entries is chaincov's contrib set
    q = RM.toy_question(rng, 9, 4, np.arange(4))
    hl, tl, sl = RM.row_triples(q, 0, q["q_edges"])
    ok = q["q_seed_local"] >= 0
    seeds = [np.unique(q["q_seed_local"][ok & (q["q_seed_bucket"] == b)]) for b in range(RM.BUCKETS)]
    zc = rng.standard_normal(4)
    S0, _cp = CC.question_scores(9, hl, tl, sl, seeds, 8, zc, k_row=2)
    step = CC.log_sigmoid(np.repeat(zc, 2))
    row, zz, _bb, mm = contrib_entries(9, hl, tl, sl, seeds, 8, step, k_row=2)[0]
    w0 = np.exp(np.where(zz >= 0, step[np.maximum(zz, 0)], 0.0).sum(1) + np.log(1.0 / 3.0))
    assert np.allclose(np.bincount(row, weights=w0 * mm, minlength=9), S0["contrib"])
    tmp = Path(tempfile.mkdtemp(prefix="zrc_"))
    try:
        # 3. the toy look (its rows exceed 64 chains, so both builds run with no row cap): rmatch's build and zrc's
        #    are the same arrays (IDENTICAL) and the identity gate passes; at k_row 1 the arrays differ and it fails;
        #    rmatch's chain carve reads zrc's cache
        RM.toy_roots(tmp, rng)
        add_q_emb(tmp, rng)
        roots = {"cache_root": tmp / "cache", "look_root": tmp / "look", "rel_dir": tmp / "rel"}
        saved_defaults = RM.question_entries.__defaults__
        RM.question_entries.__defaults__ = saved_defaults[:-1] + (10 ** 6,)
        try:
            RM.build("metaqa", "toy", out_root=tmp / "rm", **roots)
        finally:
            RM.question_entries.__defaults__ = saved_defaults
        assert RM.question_entries.__defaults__[-1] == RM.K_ROW
        build("metaqa", "toy", out_root=tmp / "zrc", rm_root=tmp / "rm", k_row=10 ** 6, **roots)
        idn = identity([("metaqa", "toy")], tmp / "id", root=tmp / "zrc", cache_root=tmp / "cache")
        assert idn["verdict"] == "IDENTICAL" and len(idn["rows"]) == 2, idn
        for p in sorted((tmp / "rm" / "metaqa" / "toy").glob("part_*")):
            for k in RM.ARRAYS:
                assert np.array_equal(np.load(p / f"{k}.npy"), np.load(tmp / "zrc" / "metaqa" / "toy" / p.name /
                                                                       f"{k}.npy"))
        build("metaqa", "toy", out_root=tmp / "zrc1", rm_root=tmp / "rm", k_row=1, **roots)
        id1 = identity([("metaqa", "toy")], tmp / "id1", root=tmp / "zrc1", cache_root=tmp / "cache")
        assert id1["verdict"] == "DIFFERENT", id1
        assert identity([("metaqa", "toy")], tmp / "id2", root=tmp / "none", cache_root=tmp / "cache")["verdict"] == \
            "MISSING"
        c = RM.chain_carve(RM.ToyBase, tmp / "zrc1", tmp / "rel")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c0 = RM.chain_carve(RM.ToyBase, tmp / "rm", tmp / "rel")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert c.chains["ent_row"].size < c0.chains["ent_row"].size
        # 4. the arm: zrm's model on rmatch's chain carve over zrc's cache; zrm's and rmatch's arms unchanged
        assert S.ARMS[ARM] == (ZM.ZRM, ChainCarveZRC) and ChainCarveZRC.CH_ROOT == CH_OUT
        assert S.ARMS[BASE_ARM] == (ZM.ZRM, RM.ChainCarveBase) and RM.ChainCarveBase.CH_ROOT == RM.CH_OUT
        assert ChainCarveZRC.__mro__[1] is S.ARMS["base"][1]
        # 5. the fork: refused without an IDENTICAL gate or from another arm's fit; a copy of the models and the
        #    training record, the arm zrc; a second fork passes; read refuses a zrm fit
        src = tmp / "fits" / "scr-zrm"
        src.mkdir(parents=True)
        torch.save({"x": torch.arange(3)}, src / "models.pt")
        LC.write_json(src / "train.json", {"train": [{"dataset": "metaqa"}]})
        LC.write_json(src / "screen.json", {"arm": BASE_ARM, "split": "L-musique", "seed": 0})
        for bad_id, bad_src in ((tmp / "id1.json", src), (tmp / "nothing.json", src)):
            try:
                fork(bad_src, "scr-zrc", identity_file=bad_id)
                raise AssertionError("a fork was accepted")
            except SystemExit as e:
                assert "not IDENTICAL" in str(e)
        other = tmp / "fits" / "scr-x"
        shutil.copytree(src, other)
        LC.write_json(other / "screen.json", {"arm": "zret"})
        try:
            fork(other, "scr-zrc", identity_file=tmp / "id.json")
            raise AssertionError("a zret fit was forked")
        except SystemExit as e:
            assert "not zrm" in str(e)
        assert fork(src, "scr-zrc", identity_file=tmp / "id.json") == 0
        dst = tmp / "fits" / "scr-zrc"
        fj = json.loads((dst / "screen.json").read_text(encoding="utf-8"))
        assert fj["arm"] == ARM and fj["split"] == "L-musique" and fj["zrc"]["models_sha256"] == \
            LC.sha_file(src / "models.pt") == LC.sha_file(dst / "models.pt")
        assert (dst / "train.json").read_bytes() == (src / "train.json").read_bytes()
        assert not (tmp / "fits" / "scr-zrc.forking").exists()
        assert fork(src, "scr-zrc", identity_file=tmp / "id.json") == 0
        src2 = tmp / "fits" / "scr-zrm2"
        shutil.copytree(src, src2)
        torch.save({"x": torch.arange(4)}, src2 / "models.pt")
        try:
            fork(src2, "scr-zrc", identity_file=tmp / "id.json")
            raise AssertionError("a fork over another fit's fork was accepted")
        except SystemExit as e:
            assert "is not" in str(e) and "fork" in str(e)
        try:
            read(["--name", "scr-zrm", "--out-root", str(tmp / "fits")])
            raise AssertionError("zrc read a zrm fit")
        except SystemExit as e:
            assert "not zrc" in str(e)
        # 6. same: identical reads, then one array changed
        for f in (src, dst):
            (f / "reads").mkdir(exist_ok=True)
            for ds in ("metaqa", "webqsp"):
                top = np.array([[1, 2, 0], [2, 2, 1]], np.int16)
                if f == dst and ds == "webqsp":
                    top = top.copy()
                    top[1, 0] = 0
                np.savez(f / "reads" / f"{ds}__s1eval.npz", candidates=np.array(["p@ep0", "p@swa"]), top=top,
                         ids=np.array(["a", "b", "c"]))
        sm = same(dst, src, tmp / "same")
        st = {r["dataset"]: r for r in sm["rows"]}
        assert st["metaqa"]["status"] == "IDENTICAL" and st["webqsp"]["status"] == "DIFFERENT"
        assert st["webqsp"]["differ"] == ["top"] and st["webqsp"]["questions_top_differs"] == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 7. relz's pair, re-call and grade under zrc's name, decided against zrm's fits; relz restored after (an error too);
    #    restamp names zrm and this round
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    with on_zrm():
        assert Z.tag({})["arm"] == ARM and Z.BASE_ARM == BASE_ARM and Z.rel_fit("J5") == ("full_zrm", "fits", "J5")
        assert Z.rel_fit("L-hotpotqa") == ("screen", "fits", "scr-zrm-hp")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    try:
        with on_zrm():
            raise KeyError("x")
    except KeyError:
        pass
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    assert sorted(zrm_compares()) == sorted(S2.SPLITS) and set(ZRM_SCREENS.values()) <= set(zrm_compares().values())
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})   # floor 0.0221
        zs = {sp: str(SR.fake_compare(T / zrm_fit(sp)[0], zrm_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_ARM))
              for sp in ZRM_FITS}                                            # zrm's fits against step 1's (0.5)
        zb = {sp: "/".join(("outputs",) + zrm_fit(sp)) for sp in ZRM_FITS}
        za = SR.fake_compare(T / "z", "scr-zrc", "L-musique", {"webqsp": 0.0468}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zrc-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM and pr["base_arm"] == BASE_ARM
        t = (T / "pz.md").read_text(encoding="utf-8")
        assert "scr-zrm" in t and "rel's" not in t
        assert json.loads((T / "pz.json").read_text(encoding="utf-8"))["decided_against"] == "zrm's fit of each split"
        rc = recall(null1, T / "pz.json", T / "rz", zs)
        assert rc["verdict"] == "PROMISING" and rc["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], rc["changed"]
        assert all(r["base"] == 0.6 for r in rc["rows"])
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "| zrm R@5 | zrc R@5 |" in t and "seventeenth round" in t and "rel's" not in t
        st1 = SR.fake_compare(T / "z2", "scr-zrc", "L-musique", {}, arm=ARM)        # decided against step 1's fit
        SR.must_stop(pair, [st1, zh], T / "bad")
        full = T / "full"
        full.mkdir()
        for sp in S2.SPLITS:
            if sp not in ZRM_FITS:
                d = {"webqsp": -0.02} if sp == "J5" else {}
                SR.fake_compare(full / "src", f"fit-{sp}", sp, d, arm=ARM,
                                base="/".join(("outputs",) + zrm_fit(sp))).replace(full / f"compare-{sp}.json")
        g = grade(full, {"L-musique": str(za), "L-hotpotqa": str(zh)}, full / "grade", check_fits=False)
        assert g["verdict"] == "NOT_ADOPTED" and g["losses"] == 2 and g["arm"] == ARM, (g["verdict"], g["losses"])
        assert "docs/FULL_ROUND17.md" in (full / "grade.md").read_text(encoding="utf-8")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zrc selftest: with no cap binding the contribution rule is rmatch's, array for array; with one, each row "
        f"and bucket keeps the k_row largest w0 m (brute force), chaincov's contrib set; with no row cap the toy build is "
        f"rmatch's (IDENTICAL) and at k_row 1 it is not, the identity gate says so, rmatch's carve reads zrc's cache; the arm is "
        f"zrm's model over zrc's cache; the fork refuses without the gate or from another arm, copies models and "
        f"training record; same marks a changed read; relz's pair, re-call and grade run under zrc's name against "
        f"zrm's fits, name zrm and this round, and relz is restored ({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "build":
        bp = argparse.ArgumentParser()
        bp.add_argument("--dataset", required=True)
        bp.add_argument("--carve", required=True)
        bp.add_argument("--host", action="store_true")
        b = bp.parse_args(rest)
        build(b.dataset, b.carve, placement={"where": "host" if b.host else "laptop"})
        return 0
    if k.cmd == "identity":
        ip = argparse.ArgumentParser()
        ip.add_argument("--carves", required=True)
        ip.add_argument("--out", default=str(CH_OUT.parent / "identity"))
        i = ip.parse_args(rest)
        rec = identity(LM.parse_sets(i.carves), i.out)
        return 0 if rec["verdict"] == "IDENTICAL" else 1
    if k.cmd == "fork":
        fp = argparse.ArgumentParser()
        fp.add_argument("--src", required=True)
        fp.add_argument("--name", required=True)
        fp.add_argument("--out-root")
        f = fp.parse_args(rest)
        return fork(f.src, f.name, f.out_root)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd in ("pair", "recall", "grade"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--full-root")
        gp.add_argument("--reuse", default="")
        gp.add_argument("--out")
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        need = {"pair": g.screens and g.out, "recall": g.pair and len(null) == 4 and g.out, "grade": g.full_root}[k.cmd]
        if not need:
            gp.error(f"{k.cmd}: pair needs --screens A,B and --out; recall --pair, the four --null files and --out; "
                     "grade --full-root")
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            elif k.cmd == "recall":
                recall(null, g.pair, g.out)
            else:
                reuse = dict(x.split("=", 1) for x in g.reuse.split(",") if x)
                v = grade(g.full_root, reuse, g.out or str(Path(g.full_root) / "grade"))["verdict"]
                return 1 if v == "INCOMPLETE" else 0
        except SystemExit as e:
            log(f"zrc {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zrc: build, identity, fork, read, compare, pair, recall, grade, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
