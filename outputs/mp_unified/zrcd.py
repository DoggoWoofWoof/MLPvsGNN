"""A diagnosis (decides nothing about any arm; docs/SCREENS.md, 'zrcd: zrm's screen fits read with zrc's entries'):
zrm's two screen fits read with zrc's chain entries. Round seventeen's gate found zrc's build of metaqa's fit carve
DIFFERENT from rmatch's on two of its eight parts, so zrm's fits are not zrc's training, and the round stopped. This
asks two things before any fit is trained with zrc's entries:
- `count`: how much of metaqa's fit carve the rule changes (questions, rows and entries whose kept set differs, and
  their walk mass), from the two builds' arrays;
- `fork`, `read`, `compare`, `pair`, `recall`: whether zrm's trained match gains when a row's entries are kept by
  contribution at read time only. A fork copies zrm's fit (models and training record unchanged) into a new folder
  under the arm zrcd, with no identity gate; its record says it trained on rmatch's entries. zrm's folders are not
  written. Each comparison checks every read against zrm's (`<out>-same.md`): the untyped reads must be zrm's. The
  pair and its re-call are relz.py's, decided against zrm's screen fits, under zrcd's name.

    python outputs/mp_unified/zrcd.py count --dataset metaqa --carve fit --out outputs/diag/zrcd-count-metaqa-fit
    python outputs/mp_unified/zrcd.py fork --src outputs/screen/fits/scr-zrm --name zrcd --out-root outputs/diag/fits
    python outputs/mp_unified/zrcd.py read --name zrcd --out-root outputs/diag/fits --device cuda --host
    python outputs/mp_unified/zrcd.py compare --new outputs/diag/fits/zrcd \\
        --base outputs/screen/fits/scr-zrm,outputs/screen/fits/scr-zret,outputs/step1/fits/L-musique \\
        --out outputs/diag/zrcd-L-musique
    python outputs/mp_unified/zrcd.py pair --screens outputs/diag/zrcd-L-musique.json,outputs/diag/zrcd-L-hotpotqa.json \\
        --out outputs/diag/zrcd-pair
    python outputs/mp_unified/zrcd.py recall --null N1,N2,N3,N4 --pair outputs/diag/zrcd-pair.json \\
        --out outputs/diag/zrcd-pair-recall
    python outputs/mp_unified/zrcd.py --selftest
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

import zrc as ZC  # noqa: E402

RM, ZM, S, S2, LC, LG = ZC.RM, ZC.ZM, ZC.S, ZC.S2, ZC.LC, ZC.LG
Z, SR = ZC.Z, ZC.SR
log = ZC.log
ARM = "zrcd"
NOTE = ("trained on rmatch's entries (zrm's fit), read with zrc's: not zrc's training (the identity gate is not "
        "IDENTICAL); a diagnosis that decides nothing about any arm")
LEAD = ("zrm's screen fits read with zrc's chain entries: trained on rmatch's entries, read with zrc's. A diagnosis "
        "(docs/SCREENS.md, 'zrcd'); it decides nothing about any arm.")

S.ARMS.update({ARM: (ZM.ZRM, ZC.ChainCarveZRC)})


# ── how much of a carve the rule changes ─────────────────────────────────────


def radix(builds):
    """Mixed-radix bases (rows, buckets, typed steps + 1) over every build compared; refused past 2^62."""
    nq = max(int(A["q_ent"].size) for A in builds)
    nr = max(int(A["ent_row"].max()) + 1 if A["ent_row"].size else 1 for A in builds)
    nb = max(int(A["ent_b"].max()) + 1 if A["ent_b"].size else 1 for A in builds)
    nz = max(int(A["ent_z"].max()) + 2 if A["ent_z"].size else 1 for A in builds)
    hops = builds[0]["ent_z"].shape[1]
    if any(A["ent_z"].size and int(A["ent_z"].min()) < -1 for A in builds) or \
            any(A["ent_row"].size and int(A["ent_row"].min()) < 0 for A in builds):
        raise SystemExit("zrcd count: a row below 0 or a chain step below -1")
    if max(nq, 1) * nr * nb * nz ** hops >= 2 ** 62:
        raise SystemExit(f"zrcd count: {nq} questions x {nr} rows x {nb} buckets x {nz}^{hops} steps overflow a key")
    return nr, nb, nz


def keys(A, nr, nb, nz):
    """One int64 per entry: question (in its part), row, bucket and the chain's typed steps, in that order."""
    k = np.repeat(np.arange(A["q_ent"].size, dtype=np.int64), A["q_ent"].astype(np.int64))
    k = k * nr + A["ent_row"].astype(np.int64)
    k = k * nb + A["ent_b"].astype(np.int64)
    z = A["ent_z"].astype(np.int64) + 1
    for h in range(z.shape[1]):
        k = k * nz + z[:, h]
    return k


def count(ds, carve, out, zrc_root=ZC.CH_OUT, rm_root=RM.CH_OUT, cache_root=LC.OUT):
    """Per step-1 part, the questions, rows and entries whose kept chains differ between rmatch's build and zrc's,
    and the walk mass each keeps only on its side."""
    parts, _recs = LC.part_dirs(ds, carve, cache_root)
    names = ("q_ent", "ent_row", "ent_z", "ent_b", "ent_m")
    rows = []
    for p in parts:
        A = {k: np.load(Path(rm_root) / ds / carve / p.name / f"{k}.npy") for k in names}
        B = {k: np.load(Path(zrc_root) / ds / carve / p.name / f"{k}.npy") for k in names}
        nr, nb, nz = radix([A, B])
        per_row = nb * nz ** A["ent_z"].shape[1]
        ka, kb = keys(A, nr, nb, nz), keys(B, nr, nb, nz)
        if np.unique(ka).size != ka.size or np.unique(kb).size != kb.size:
            raise SystemExit(f"zrcd count: {p.name} has two entries with one question, row, bucket and chain")
        oa, ob = ~np.isin(ka, kb), ~np.isin(kb, ka)
        d = np.concatenate([ka[oa], kb[ob]])
        ma, mb = A["ent_m"].astype(np.float64), B["ent_m"].astype(np.float64)
        rows.append({"part": p.name, "questions": int(A["q_ent"].size), "entries_rmatch": int(ka.size),
                     "entries_zrc": int(kb.size), "q_ent_identical": bool(np.array_equal(A["q_ent"], B["q_ent"])),
                     "questions_differ": int(np.unique(d // (nr * per_row)).size),
                     "rows_differ": int(np.unique(d // per_row).size),
                     "entries_only_rmatch": int(oa.sum()), "entries_only_zrc": int(ob.sum()),
                     "mass_only_rmatch": float(ma[oa].sum()), "mass_only_zrc": float(mb[ob].sum()),
                     "mass_rmatch": float(ma.sum()), "mass_zrc": float(mb.sum())})
    tot = {k: sum(r[k] for r in rows) for k in ("questions", "entries_rmatch", "entries_zrc", "questions_differ",
                                                 "rows_differ", "entries_only_rmatch", "entries_only_zrc")}
    for k in ("mass_only_rmatch", "mass_only_zrc", "mass_rmatch", "mass_zrc"):
        tot[k] = float(sum(r[k] for r in rows))
    rec = {"dataset": ds, "carve": carve, "rows": rows, "total": tot, "rule": ZC.RULE,
           "rmatch_cache": ZC.rel_path(rm_root), "zrc_cache": ZC.rel_path(zrc_root),
           "script_sha256": LC.sha_src(__file__), "zrc_sha256": LC.sha_src(ZC.__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    share = lambda a, b: f"{a / b:.4f}" if b else "-"  # noqa: E731
    md = [f"# {ds}/{carve}: rmatch's chain entries (kept by mass) against zrc's (kept by contribution)", "",
          "A diagnosis; it decides nothing. Per step-1 part: the questions and (question, row) pairs whose kept "
          "entries differ, the entries each build keeps that the other does not, and their walk mass (ent_m, each "
          "chain's mass renormalised over its rows).", "",
          f"In all: {tot['questions_differ']} of {tot['questions']} questions ({share(tot['questions_differ'], tot['questions'])}) "
          f"and {tot['rows_differ']} rows keep a different set; {tot['entries_only_rmatch']} of rmatch's "
          f"{tot['entries_rmatch']} entries ({share(tot['entries_only_rmatch'], tot['entries_rmatch'])}) are not zrc's, "
          f"holding {share(tot['mass_only_rmatch'], tot['mass_rmatch'])} of rmatch's kept mass.", "",
          "| part | questions | entries (rmatch's) | questions that differ | rows that differ | entries only rmatch's | "
          "only zrc's | mass only rmatch's | only zrc's |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    md += [f"| {r['part']} | {r['questions']} | {r['entries_rmatch']} | {r['questions_differ']} | {r['rows_differ']} | "
           f"{r['entries_only_rmatch']} | {r['entries_only_zrc']} | {share(r['mass_only_rmatch'], r['mass_rmatch'])} | "
           f"{share(r['mass_only_zrc'], r['mass_zrc'])} |" for r in rows]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zrcd count {ds}/{carve}: {tot['questions_differ']} of {tot['questions']} questions, {tot['rows_differ']} rows, "
        f"{tot['entries_only_rmatch']} of {tot['entries_rmatch']} entries differ")
    return rec


# ── zrm's fits read with zrc's entries ───────────────────────────────────────


def fork(src, name, out_root, identity_file=ZC.CH_OUT.parent / "identity.json"):
    """zrm's fit src copied to out_root/name under the arm zrcd: models.pt and train.json unchanged, screen.json zrm's
    with the arm, where it came from and the identity gate's verdict (no gate). Refused from any other arm's fit; a fork
    that exists already passes only with src's models."""
    src, out_root, idf = Path(src), Path(out_root), Path(identity_file)
    sj = json.loads((src / "screen.json").read_text(encoding="utf-8"))
    if sj.get("arm") != ZC.BASE_ARM:
        raise SystemExit(f"zrcd fork: {src} is arm {sj.get('arm')}, not {ZC.BASE_ARM}")
    msha = LC.sha_file(src / "models.pt")
    dst = out_root / name
    if dst.exists():
        if (dst / "models.pt").exists() and LC.sha_file(dst / "models.pt") == msha and \
                json.loads((dst / "screen.json").read_text(encoding="utf-8")).get("arm") == ARM:
            log(f"zrcd fork: {dst} is {src}'s already")
            return 0
        raise SystemExit(f"zrcd fork: {dst} exists and is not {src}'s fork")
    out_root.mkdir(parents=True, exist_ok=True)
    tmp = out_root / f"{name}.forking"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    shutil.copy2(src / "models.pt", tmp / "models.pt")
    shutil.copy2(src / "train.json", tmp / "train.json")
    if LC.sha_file(tmp / "models.pt") != msha:
        raise SystemExit(f"zrcd fork: {tmp / 'models.pt'} is not {src}'s")
    idv = json.loads(idf.read_text(encoding="utf-8")).get("verdict") if idf.exists() else None
    rec = dict(sj)
    rec.update({"arm": ARM, "name": name, "zrcd_sha256": LC.sha_src(__file__),
                "zrcd": {"forked_from": ZC.rel_path(src), "models_sha256": msha,
                         "screen_sha256": LC.sha_file(src / "screen.json"),
                         "train_sha256": LC.sha_file(src / "train.json"), "identity": ZC.rel_path(idf),
                         "identity_verdict": idv, "identity_sha256": LC.sha_file(idf) if idf.exists() else None,
                         "cache": ZC.rel_path(ZC.CH_OUT), "rule": ZC.RULE, "note": NOTE}})
    LC.write_json(tmp / "screen.json", rec)
    os.replace(tmp, dst)
    log(f"zrcd fork: {src} -> {dst} (models {msha[:12]}; the gate is {idv})")
    return 0


def arm_of(fit):
    return json.loads((Path(fit) / "screen.json").read_text(encoding="utf-8")).get("arm")


def read(argv):
    name, out_root = S2.where(argv)
    if arm_of(out_root / name) != ARM:
        raise SystemExit(f"zrcd: {name} is arm {arm_of(out_root / name)}, not {ARM}")
    return RM.read(argv)


def compare(rest):
    """lean_screen's compare; when the new fit is zrcd's and the first base zrm's, every read checked against zrm's
    too (<out>-same)."""
    rc = S2.main(["compare"] + rest)
    cp = argparse.ArgumentParser(add_help=False)
    cp.add_argument("--new")
    cp.add_argument("--base")
    cp.add_argument("--out")
    a, _ = cp.parse_known_args(rest)
    if rc == 0 and a.out and arm_of(a.new) == ARM and arm_of(a.base.split(",")[0]) == ZC.BASE_ARM:
        ZC.same(a.new, a.base.split(",")[0], f"{a.out}-same")
    return rc


# ── relz.py's pair and re-call, decided against zrm's screen fits ────────────


@contextlib.contextmanager
def on_zrm():
    """relz.py's records under zrcd's name, each read decided against zrm's fit of its split; relz restored after."""
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, ZC.BASE_ARM, ZC.ZRM_FITS, ZC.ZRM_SCREENS, ZC.zrm_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrm's screen fits scr-zrm and scr-zrm-hp)"),
       ("(rel's screen fits)", "(zrm's screen fits)"),
       ("| rel R@5 | relz R@5 |", "| zrm R@5 | zrcd R@5 |"),
       ("section 2 and the tenth round", "section 2, and the diagnosis zrcd"))


def restamp(out):
    """A record relz.py wrote under zrcd's name: zrm named as the base and the diagnosis's note in its md, this file's
    sha and the note in its json."""
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        head, sep, body = t.partition("\n\n")
        md.write_text(head + "\n\n" + LEAD + "\n\n" + body if sep else t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"decided_against": "zrm's fit of each split", "zrcd_sha256": LC.sha_src(__file__), "note": NOTE})
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


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    rng = np.random.default_rng(18)
    tmp = Path(tempfile.mkdtemp(prefix="zrcd_"))
    try:
        # 1. count: rmatch's toy build with no row cap against zrc's at no cap (nothing differs) and at k_row 1
        #    (every difference found, checked against a direct comparison of the entry sets)
        RM.toy_roots(tmp, rng)
        ZC.add_q_emb(tmp, rng)
        roots = {"cache_root": tmp / "cache", "look_root": tmp / "look", "rel_dir": tmp / "rel"}
        saved = RM.question_entries.__defaults__
        RM.question_entries.__defaults__ = saved[:-1] + (10 ** 6,)
        try:
            RM.build("metaqa", "toy", out_root=tmp / "rm", **roots)
        finally:
            RM.question_entries.__defaults__ = saved
        assert RM.question_entries.__defaults__[-1] == RM.K_ROW
        ZC.build("metaqa", "toy", out_root=tmp / "zrc", rm_root=tmp / "rm", k_row=10 ** 6, **roots)
        ZC.build("metaqa", "toy", out_root=tmp / "zrc1", rm_root=tmp / "rm", k_row=1, **roots)
        c0 = count("metaqa", "toy", tmp / "c0", zrc_root=tmp / "zrc", rm_root=tmp / "rm", cache_root=tmp / "cache")
        assert c0["total"]["questions_differ"] == 0 and c0["total"]["entries_only_zrc"] == 0, c0["total"]
        assert c0["total"]["entries_only_rmatch"] == 0 and c0["total"]["mass_only_rmatch"] == 0.0
        c1 = count("metaqa", "toy", tmp / "c1", zrc_root=tmp / "zrc1", rm_root=tmp / "rm", cache_root=tmp / "cache")
        t1 = c1["total"]
        assert t1["entries_only_rmatch"] > 0 and t1["entries_only_zrc"] == 0 and t1["questions_differ"] > 0, t1
        want_q = want_r = want_e = 0
        for p in sorted((tmp / "rm" / "metaqa" / "toy").glob("part_*")):
            A = {k: np.load(p / f"{k}.npy") for k in ("q_ent", "ent_row", "ent_z", "ent_b")}
            B = {k: np.load(tmp / "zrc1" / "metaqa" / "toy" / p.name / f"{k}.npy") for k in A}
            sets = []
            for E in (A, B):
                off = np.concatenate([[0], np.cumsum(E["q_ent"])])
                sets.append([{(int(E["ent_row"][e]), int(E["ent_b"][e])) + tuple(int(x) for x in E["ent_z"][e])
                              for e in range(off[qi], off[qi + 1])} for qi in range(E["q_ent"].size)])
            for x, y in zip(*sets):
                want_q += bool(x ^ y)
                want_r += len({e[0] for e in x ^ y})
                want_e += len(x - y)
        assert (t1["questions_differ"], t1["rows_differ"], t1["entries_only_rmatch"]) == (want_q, want_r, want_e), \
            (t1, want_q, want_r, want_e)
        assert t1["mass_only_rmatch"] > 0 and abs(t1["mass_rmatch"] - c0["total"]["mass_rmatch"]) < 1e-9
        assert "questions that differ" in (tmp / "c1.md").read_text(encoding="utf-8")
        # the key keeps steps past 63 (webqsp's relations) apart
        big = {"q_ent": np.array([2, 1]), "ent_row": np.array([3, 3, 0], np.int16), "ent_b": np.array([0, 0, 1], np.int8),
               "ent_z": np.array([[700, -1, -1], [701, -1, -1], [700, 5, -1]], np.int16)}
        nr, nb, nz = radix([big])
        kk = keys(big, nr, nb, nz)
        assert (nr, nb, nz) == (4, 2, 703) and np.unique(kk).size == 3
        assert kk[0] // (nb * nz ** 3) == kk[1] // (nb * nz ** 3) != kk[2] // (nb * nz ** 3)
        # 2. the fork: no gate needed, its verdict recorded; refused from another arm's fit or over another fork
        src = tmp / "fits" / "scr-zrm"
        src.mkdir(parents=True)
        torch.save({"x": torch.arange(3)}, src / "models.pt")
        LC.write_json(src / "train.json", {"train": [{"dataset": "metaqa"}]})
        LC.write_json(src / "screen.json", {"arm": ZC.BASE_ARM, "split": "L-musique", "seed": 0})
        LC.write_json(tmp / "identity.json", {"verdict": "DIFFERENT"})
        other = tmp / "fits" / "scr-x"
        shutil.copytree(src, other)
        LC.write_json(other / "screen.json", {"arm": "zrc"})
        try:
            fork(other, "zrcd", tmp / "dfits", tmp / "identity.json")
            raise AssertionError("a zrc fit was forked")
        except SystemExit as e:
            assert "not zrm" in str(e)
        assert fork(src, "zrcd", tmp / "dfits", tmp / "identity.json") == 0
        dst = tmp / "dfits" / "zrcd"
        fj = json.loads((dst / "screen.json").read_text(encoding="utf-8"))
        assert fj["arm"] == ARM and fj["split"] == "L-musique" and fj["zrcd"]["identity_verdict"] == "DIFFERENT"
        assert fj["zrcd"]["models_sha256"] == LC.sha_file(src / "models.pt") == LC.sha_file(dst / "models.pt")
        assert (dst / "train.json").read_bytes() == (src / "train.json").read_bytes()
        assert not (tmp / "dfits" / "zrcd.forking").exists()
        assert fork(src, "zrcd", tmp / "dfits", tmp / "identity.json") == 0
        src2 = tmp / "fits" / "scr-zrm2"
        shutil.copytree(src, src2)
        torch.save({"x": torch.arange(4)}, src2 / "models.pt")
        try:
            fork(src2, "zrcd", tmp / "dfits", tmp / "identity.json")
            raise AssertionError("a fork over another fit's fork was accepted")
        except SystemExit as e:
            assert "is not" in str(e) and "fork" in str(e)
        # 3. read refuses a fit of another arm; the arm is zrm's model over zrc's cache; zrc's and zrm's unchanged
        try:
            read(["--name", "scr-zrm", "--out-root", str(tmp / "fits")])
            raise AssertionError("zrcd read a zrm fit")
        except SystemExit as e:
            assert "not zrcd" in str(e)
        assert S.ARMS[ARM] == (ZM.ZRM, ZC.ChainCarveZRC) and S.ARMS[ZC.ARM] == (ZM.ZRM, ZC.ChainCarveZRC)
        assert S.ARMS[ZC.BASE_ARM] == (ZM.ZRM, RM.ChainCarveBase)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 4. relz's pair and re-call under zrcd's name, decided against zrm's screen fits; relz restored after (an error
    #    too); the records name zrm and carry the diagnosis's note
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    try:
        with on_zrm():
            assert Z.tag({})["arm"] == ARM and Z.rel_fit("L-hotpotqa") == ("screen", "fits", "scr-zrm-hp")
            raise KeyError("x")
    except KeyError:
        pass
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})   # floor 0.0221
        zs = {sp: str(SR.fake_compare(T / ZC.zrm_fit(sp)[0], ZC.zrm_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=ZC.BASE_ARM)) for sp in ZC.ZRM_FITS}
        zb = {sp: "/".join(("outputs",) + ZC.zrm_fit(sp)) for sp in ZC.ZRM_FITS}
        za = SR.fake_compare(T / "d", "zrcd", "L-musique", {"webqsp": 0.0468}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "d", "zrcd-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM and pr["base_arm"] == ZC.BASE_ARM
        t = (T / "pz.md").read_text(encoding="utf-8")
        assert "scr-zrm" in t and "rel's" not in t and LEAD in t and t.index(LEAD) > t.index("**MIXED**")
        pj = json.loads((T / "pz.json").read_text(encoding="utf-8"))
        assert pj["decided_against"] == "zrm's fit of each split" and pj["note"] == NOTE
        rc = recall(null1, T / "pz.json", T / "rz", zs)
        assert rc["verdict"] == "PROMISING" and rc["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], rc["changed"]
        assert all(r["base"] == 0.6 for r in rc["rows"])
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "| zrm R@5 | zrcd R@5 |" in t and "the diagnosis zrcd" in t and "rel's" not in t and LEAD in t
        st1 = SR.fake_compare(T / "d2", "zrcd", "L-musique", {}, arm=ARM)       # decided against step 1's fit
        SR.must_stop(pair, [st1, zh], T / "bad")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zrcd selftest: count finds no difference between equal builds and every differing question, row and entry "
        f"at k_row 1 (a direct set comparison), and its key keeps steps past 63 apart; the fork needs no gate, records "
        f"its verdict, copies models and training record, refuses another arm's fit and an existing fork of another "
        f"fit; read refuses another arm; relz's pair and re-call run under zrcd's name against zrm's screen fits, carry "
        f"the note, and relz is restored ({time.time() - t0:.1f}s): ok")
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
    if k.cmd == "count":
        cp = argparse.ArgumentParser()
        cp.add_argument("--dataset", required=True)
        cp.add_argument("--carve", required=True)
        cp.add_argument("--out", required=True)
        c = cp.parse_args(rest)
        count(c.dataset, c.carve, c.out)
        return 0
    if k.cmd == "fork":
        fp = argparse.ArgumentParser()
        fp.add_argument("--src", required=True)
        fp.add_argument("--name", required=True)
        fp.add_argument("--out-root", required=True)
        f = fp.parse_args(rest)
        return fork(f.src, f.name, f.out_root)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        if (k.cmd == "pair" and not g.screens) or (k.cmd == "recall" and not (g.pair and len(null) == 4)):
            gp.error("pair needs --screens A,B; recall --pair and the four --null files")
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            else:
                recall(null, g.pair, g.out)
        except SystemExit as e:
            log(f"zrcd {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zrcd: count, fork, read, compare, pair, recall, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
