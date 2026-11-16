"""D83's B1 read carve b1v (docs/D83_UNIFIED_R_REFIT.md sections 5, 7 and 8): HippoRAG 2's three 1,000-question settings
(2wiki, hotpotqa, musique) with D83's three changes, on which the D83 fits are read against the record's reads on b1w
(section 7's deciding cells). The carve is named b1v: U1a's b1u (look_b1u.py, outputs/bench/hipporag2_u) is another graph.

zb1h's machinery (zb1h.py is never edited; its module globals are rebound here, as zap2.py does for AP2's b1n):

    bench     the laptop: b1n's folder (AP2: W2_S in the structural slot, F0T = AP1b's L0 + L0T as the NER family, B1a's kNN)
              with the kNN pairs that W2_S or F0T hold removed (S > R > K; W2_S and F0T are both R), and the five labels' key
              sets read BEFORE the dedup, as package-row keys min*n + max (look_b1.setting_store's rows: a repeated passage's
              edges land on its one row, self-loops dropped): R.mention = W2_S, R.name_comention = L0, R.title_comention =
              F0T minus L0, K.knn = B1a's kNN; S is empty on a passage setting
    settings  zb1h's: the hyperlink CSR and its entry types, on the folder
    link      B1P's pools of record of b1w (A2: never rebuilt) hard-linked as the b1v carve's (sha256 checked), the
              manifest naming them, its carve b1v and a d83 block naming the record's manifest and carve
    chain     the link, the KT pass (zd83.kt_pass and kt_merge: the pools' KT rows recomputed on b1v's stores, under
              zb1h's setting context) once G1b has passed on the setting, then zb1h's chain on the filed pools (look,
              bases, cache, links: one link per label (zd83.labelled_build), kt, record), each its own process
    gate      PKO's gate on the carve
    g1b       G1 on B1 (section 8): b1w's own folder (outputs/bench/hipporag2_w) through b1v's plumbing (zb1h's settings and
              setting context, the pools link under the b1v stem, the KT pass and merge) under outputs/b1v/g1b: its KT rows
              must equal B1P's filed KT rows of b1w, array for array. A b1v KT merge refuses until its setting passed.
    lab       zd83m's LAB on the carve (outputs/b1v/lab)
    lists     LTM's lists with the five label bits (zd83m.labels_on, ltm_on), every links part checked as b1v's labelled
    cols      links; the columns read D83's fold fits (the D83 world's outputs/ltm): both run in the D83 world

    python outputs/mp_unified/zd83b.py selftest
    python outputs/mp_unified/zd83b.py bench --dataset D                     # the laptop (AP1b's levels are there)
    python outputs/mp_unified/zd83b.py g1b --dataset D --host                # settings, link, KT pass and merge on b1w's folder; verdict
    python outputs/mp_unified/zd83b.py settings --dataset D --host
    python outputs/mp_unified/zd83b.py chain --dataset D --host
    python outputs/mp_unified/zd83b.py gate --tag D --host
    python outputs/mp_unified/zd83b.py lab --dataset D
    python outputs/host_ops/d83world.py run --mode train -- python outputs/mp_unified/zd83b.py lists --dataset D
    python outputs/host_ops/d83world.py run --mode train -- python outputs/mp_unified/zd83b.py cols --split S --dataset D --host
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zb1h as H  # noqa: E402  (its header sets BLAS from sys.argv[1:2] before numpy loads)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = H.ROOT
DECLARED = "docs/D83_UNIFIED_R_REFIT.md"
STAGE = "D83"
ARM = "b1v"                                                     # the label sets' and the links' d83 arm
CARVE = "b1v"
OUT = ROOT / "outputs" / "b1v"
LTM_OUT = OUT / "ltm"
B1N = ROOT / "outputs" / "b1n" / "bench" / "hipporag2_n"        # AP2's folder: W2_S, F0T, B1a's kNN
B1N_SETTINGS = ROOT / "outputs" / "b1n" / "settings"            # its settings records (the package's node count)
B1W = ROOT / "outputs" / "bench" / "hipporag2_w"                # b1w's folder (B1f): G1b's bench
B1A_REAL = ROOT / "outputs" / "bench" / "hipporag2"             # B1a's own folder: its lists and the files they name
B1P = ROOT / "outputs" / "b1p"                                  # B1P's roots: the pools of record of b1w
RECORD_CARVE = "b1w"
BENCH_V = OUT / "bench" / "hipporag2_v"
AP1B = ROOT / "outputs" / "ap1b"
G1B = OUT / "g1b"
STRUCT, NER, KNN = "graph_structural.npz", "graph_ner.npz", "graph_knn.npz"
SETTINGS = ("2wiki", "hotpotqa", "musique")
TAGS = {ds: (ds,) for ds in SETTINGS}
CARVES = ("b1t", "b1w", CARVE)
LABELS = ("S", "R.mention", "R.name_comention", "R.title_comention", "K.knn")   # zd83.LABELS (checked in the self-test)
KT_SHARDS = 1                                                   # 1,000 questions: one KT shard per setting
GRAPH_TEXT = ("D83 b1v: b1w's structural family (W2_S, R), F0T (AP1b's L0 + L0T, R) as the NER family, and B1a's kNN family "
              "without the pairs W2_S or F0T hold (S > R > K); every link labelled by the rules that made its pair")
G1B_TEXT = "D83 G1b: b1w's own folder (W2_S, B1a's NER and kNN) through b1v's plumbing"
NO_STEP1 = {"state": f"not applicable: B1's settings have no step-1 cache (D83, {DECLARED})"}
STEPS = ("look", "bases", "cache", "links", "kt", "record", "ktpass", "ktmerge")
_ON_B1H = H.on_b1h
_H_FILE = H.__file__


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zd83b: {msg}", flush=True)


sha_file, sha_src, rel, read_json, write_json, utc = H.sha_file, H.sha_src, H.rel, H.read_json, H.write_json, H.utc


def zd83():
    """zd83 (D83's chain code: the label sets, labelled links, the KT pass), imported when a step needs it."""
    import zd83 as D
    return D


# ── keys over the package's rows ─────────────────────────────────────────────

def row_keys(src, dst, rows, n):
    """Sorted unique keys min*n + max of a setting graph's pairs as look_b1.setting_store stores them: each setting position
    mapped to its package row, self-loops dropped (a repeated passage's edges land on its one row)."""
    a = np.asarray(rows, np.int64)[np.asarray(src, np.int64)]
    b = np.asarray(rows, np.int64)[np.asarray(dst, np.int64)]
    keep = a != b
    a, b = a[keep], b[keep]
    if a.size and (int(min(a.min(), b.min())) < 0 or int(max(a.max(), b.max())) >= int(n)):
        raise SystemExit(f"D83 b1v: a pair outside the package's {n} nodes")
    return np.unique(np.minimum(a, b) * int(n) + np.maximum(a, b))


def entry_row_keys(src, dst, rows, n):
    """The key of every entry (self-loops: -1), in entry order."""
    a = np.asarray(rows, np.int64)[np.asarray(src, np.int64)]
    b = np.asarray(rows, np.int64)[np.asarray(dst, np.int64)]
    k = np.minimum(a, b) * int(n) + np.maximum(a, b)
    k[a == b] = -1
    return k


def pos_keys(src, dst, m):
    """Sorted unique keys over setting positions (a < b checked) of a table of m positions."""
    a, b = np.asarray(src, np.int64), np.asarray(dst, np.int64)
    if a.size and (not (a < b).all() or int(a.min()) < 0 or int(b.max()) >= int(m)):
        raise SystemExit(f"D83 b1v: a table holds a pair that is not a < b inside the setting's {m} passages")
    return np.unique(a * int(m) + b)


def bench_arrays(st, ner, knn, l0, l0t, rows, n):
    """The kNN entries kept (no pair W2_S or F0T holds, keyed by package rows), the five labels' key sets (before the
    dedup) and the counts. st, ner, knn: the setting files' arrays; l0, l0t: AP1b's tables (src, dst) over positions."""
    m = len(rows)
    f0t_pos = pos_keys(ner["src"], ner["dst"], m)
    if not np.array_equal(f0t_pos, np.union1d(pos_keys(l0[0], l0[1], m), pos_keys(l0t[0], l0t[1], m))):
        raise SystemExit("D83 b1v G2: the folder's NER pairs are not AP1b's L0 united with L0T (F0T)")
    st_k = row_keys(st["src"], st["dst"], rows, n)
    f0k = row_keys(l0[0], l0[1], rows, n)
    f0t_k = row_keys(ner["src"], ner["dst"], rows, n)
    knn_k = row_keys(knn["src"], knn["dst"], rows, n)
    held = np.union1d(st_k, f0t_k)
    ek = entry_row_keys(knn["src"], knn["dst"], rows, n)
    keep = ~np.isin(ek, held)
    kept_k = row_keys(np.asarray(knn["src"])[keep], np.asarray(knn["dst"])[keep], rows, n)
    if np.intersect1d(kept_k, held).size:
        raise SystemExit("D83 b1v G4: a kept kNN pair is held by R")
    sets = {"S": np.zeros(0, np.int64), "R.mention": st_k, "R.name_comention": f0k,
            "R.title_comention": np.setdiff1d(f0t_k, f0k), "K.knn": knn_k}
    counts = {"structural_pairs": int(st_k.size), "ner_pairs": int(f0t_k.size), "f0_pairs": int(f0k.size),
              "knn_pairs_in": int(knn_k.size), "knn_entries_in": int(ek.size), "knn_entries_removed_in_R": int((~keep).sum()),
              "knn_pairs_removed_in_R": int(np.intersect1d(knn_k, held).size), "knn_pairs": int(kept_k.size)}
    if counts["knn_pairs"] != counts["knn_pairs_in"] - counts["knn_pairs_removed_in_R"]:
        raise SystemExit("D83 b1v: the kNN dedup's pair counts do not add up")
    return keep, sets, counts


@contextlib.contextmanager
def labels_here(base):
    """zd83's label sets under `base`/labels for ARM (zd83.labels_dir is looked up at call time); every other arm's are zd83's."""
    D = zd83()
    orig = D.labels_dir

    def labels_dir(arm, name):
        return Path(base) / "labels" / name if arm == ARM else orig(arm, name)

    D.labels_dir = labels_dir
    try:
        yield D
    finally:
        D.labels_dir = orig


def bench(name, out_root=None, src_root=None, levels_root=None, settings_root=None):
    """b1v's folder of one setting from b1n's, and the label sets (the laptop: AP1b's levels are read here)."""
    import u1a
    t0 = time.time()
    out_root = BENCH_V if out_root is None else Path(out_root)
    src_root = B1N if src_root is None else Path(src_root)
    levels_root = AP1B if levels_root is None else Path(levels_root)
    settings_root = B1N_SETTINGS if settings_root is None else Path(settings_root)
    src, out = src_root / name, out_root / name
    build = read_json(src / "build.json")
    bad = [f for f in u1a.FILES if sha_file(src / f) != build["files"].get(f)]
    if bad:
        raise SystemExit(f"D83 b1v {name}: {bad} differ from b1n's build.json")
    ap2 = build.get("ap2") or {}
    if ap2.get("form") != "F0T" or ap2.get("tables") != ["L0", "L0T"]:
        raise SystemExit(f"D83 b1v {name}: b1n's NER family is not F0T (L0 + L0T): {ap2.get('form')} {ap2.get('tables')}")
    lv = levels_root / name / "levels.npz"
    if sha_file(lv) != ap2["levels_npz_sha256"]:
        raise SystemExit(f"D83 b1v {name}: {rel(lv)} is not the levels file b1n's F0T was built from (G2)")
    real = read_json(B1A_REAL / name / "build.json") if (B1A_REAL / name / "build.json").exists() else None
    if real is not None and real["files"][KNN] != build["files"][KNN]:
        raise SystemExit(f"D83 b1v {name}: b1n's kNN file is not B1a's")
    nodes = read_json(src / "nodes.json")
    rows = np.asarray([x["package_row"] for x in nodes], np.int64)
    sj = read_json(settings_root / f"{name}.json")
    n = int(sj["n_nodes"])
    with np.load(src / STRUCT) as z:
        st = {k: z[k] for k in z.files}
    with np.load(src / NER) as z:
        ner = {k: z[k] for k in z.files}
    with np.load(src / KNN) as z:
        knn = {k: z[k] for k in z.files}
    with np.load(lv) as z:
        l0, l0t = (z["L0_src"], z["L0_dst"]), (z["L0T_src"], z["L0T_dst"])
    keep, sets, counts = bench_arrays(st, ner, knn, l0, l0t, rows, n)
    lab_base = OUT if out_root == BENCH_V else out_root.parent.parent     # <carve root>/bench/<folder>: labels beside bench
    out.mkdir(parents=True, exist_ok=True)
    for f in u1a.FILES:
        if f != KNN:
            shutil.copyfile(src / f, out / f)
            if sha_file(out / f) != build["files"][f]:
                raise SystemExit(f"D83 b1v {rel(out / f)}: the copy is not b1n's")
    u1a.save_npz(out / KNN, **{k: np.asarray(v)[keep] for k, v in knn.items()})
    with labels_here(lab_base) as D:
        lab = D.write_label_sets(ARM, name, n, sets)
    lab_dir = lab_base / "labels" / name
    b = json.loads(json.dumps(build))
    b["files"][KNN] = sha_file(out / KNN)
    b["d83"] = {"declared_in": DECLARED, "stage": STAGE, "carve": CARVE, "source": rel(src),
                "source_build_sha256": sha_file(src / "build.json"), "b1a_knn_sha256": build["files"][KNN],
                "package_nodes": n, "package_nodes_from": rel(settings_root / f"{name}.json"), "counts": counts,
                "labels": lab["pairs"], "labels_record_sha256": sha_file(lab_dir / "record.json"),
                "dedup": "kNN pairs W2_S (R.mention) or F0T (R's NER store) hold removed; NER keeps the pairs W2_S holds (both R)",
                "script_sha256": sha_src(__file__), "utc": utc()}
    write_json(out / "build.json", b)
    shas = {f: sha_file(out / f) for f in u1a.FILES}
    if shas != b["files"]:
        raise SystemExit(f"D83 b1v {name}: the folder as written is not its build.json's")
    check_lists(read_json(B1A_REAL / name / "b1_lists.json")["setting_files_sha256"], shas, b)
    log(f"bench {name}: {counts}; labels {lab['pairs']} ({time.time() - t0:.0f} s)")
    return 0


# ── the lists check: B1a's lists name B1a's files; b1v replaced three of them ─

def check_lists(named, shas, build):
    """B1a's lists record (setting_files_sha256 `named`) against b1v's file shas: every file but the structural, NER and kNN
    ones equal; each of those the sha B1a's file has (u1a.b1a_structural_sha256, ap2.b1a_ner_sha256, d83.b1a_knn_sha256)."""
    swapped = {STRUCT: build["u1a"]["b1a_structural_sha256"], NER: build["ap2"]["b1a_ner_sha256"],
               KNN: build["d83"]["b1a_knn_sha256"]}
    if {f: s for f, s in named.items() if f not in swapped} != {f: s for f, s in shas.items() if f not in swapped}:
        raise SystemExit("D83 b1v: B1a's lists were built on other setting files than this carve's")
    for f, s in swapped.items():
        if named.get(f) != s:
            raise SystemExit(f"D83 b1v: B1a's lists name another {f} than the one this carve replaced")
    return named


def lists_of(M, st):
    """look_b1u.load_lists with the NER and kNN files checked as the structural one is: B1a's lists, from B1a's folder."""
    rec = json.loads((M.LBU.B1A / st.name / "b1_lists.json").read_text(encoding="utf-8"))
    named = check_lists(rec["setting_files_sha256"], st.shas, st.build)
    here = M.LB.BENCH
    M.LB.BENCH = M.LBU.B1A
    try:
        return M.LBU._ORIG_LISTS(SimpleNamespace(**{**vars(st), "shas": named}))
    finally:
        M.LB.BENCH = here


@contextlib.contextmanager
def on_b1v(M, name, stats, look=False):
    """zb1h.on_b1h with b1v's lists (B1a's, checked by check_lists)."""
    P_ = H.zb().Patches()
    try:
        P_.set(M.LB, load_lists=lambda st: lists_of(M, st))
        with _ON_B1H(M, name, stats, look) as B:
            yield B
    finally:
        P_.restore()


@contextlib.contextmanager
def on_g1b(M, name, stats, look=False):
    """zb1h.on_b1h on b1w's own folder: look_b1u's variant check of B1a's lists (the structural file replaced)."""
    P_ = H.zb().Patches()
    try:
        P_.set(M.LB, load_lists=M.LBU.load_lists)
        with _ON_B1H(M, name, stats, look) as B:
            yield B
    finally:
        P_.restore()


def refuse_pools(*_a, **_k):
    raise SystemExit("D83 b1v: the pools are B1P's of record (A2); the pools step never runs")


def read_roots():
    d = f"outputs/{CARVE}"
    return {"pools": f"{d}/pools", "plook": f"{d}/look", "cache": f"{d}/cache", "rmatch": f"{d}/chains/rmatch", "zrc": f"{d}/chains/zrc",
            "links": f"{d}/links", "kt": f"{d}/kt"}


def bind(g1b=False):
    """zb1h's module globals set to b1v's (or G1b's: b1w's folder under outputs/b1v/g1b); its functions read them at call time."""
    out = G1B if g1b else OUT
    for k, v in (("DECLARED", DECLARED), ("STAGE", STAGE), ("OUT", out), ("LTM_OUT", out / "ltm"), ("B1A", B1W if g1b else BENCH_V),
                 ("CARVE", CARVE), ("CARVES", CARVES), ("SETTINGS", SETTINGS), ("TAGS", TAGS),
                 ("GRAPH_TEXT", G1B_TEXT if g1b else GRAPH_TEXT), ("NO_STEP1", NO_STEP1), ("read_roots", read_roots),
                 ("on_b1h", on_g1b if g1b else on_b1v), ("pools", refuse_pools)):
        setattr(H, k, v)
    return out


# ── settings, the pools link, the KT pass ────────────────────────────────────

def settings(name, host=False, g1b=False):
    """zb1h.settings on the folder; the record's desc_b1 names D83's carve."""
    bind(g1b)
    orig = H.write_json

    def labelled(p, obj):
        if isinstance(obj, dict) and "desc_b1" in obj:
            obj["desc_b1"]["variant"] = "g1b" if g1b else "v"
            obj["d83"] = {"script_sha256": sha_src(__file__), "zb1h_sha256": sha_src(_H_FILE), "g1b": bool(g1b)}
        return orig(p, obj)

    H.write_json = labelled
    try:
        return H.settings(host, name)
    finally:
        H.write_json = orig


def link_pools(name, host=False, g1b=False):
    """B1P's pools file and KT rows of b1w hard-linked under the b1v stem (zwq.place: sha256 checked, nothing overwritten);
    the manifest written beside them names them, carve b1v, and a d83 block naming the record's manifest and carve."""
    import zwq as WQ
    out_base = bind(g1b)
    M = H.zb().mods(host)
    PK = M.PK
    src_R, R = PK.roots(B1P), PK.roots(out_base)
    man_src = src_R.pools / f"{PK.stem(name, RECORD_CARVE)}.json"
    ent = read_json(man_src)
    if ent.get("carve") != RECORD_CARVE or ent.get("dataset") != name:
        raise SystemExit(f"D83 b1v: {rel(man_src)} is not B1P's {name}/{RECORD_CARVE} manifest")
    st = PK.stem(name, CARVE)
    out = R.pools / f"{st}.json"
    files = {"file": (PK.resolve(ent["file"]), R.pools / f"{st}.npz", ent["sha256"]),
             "kt": (PK.resolve(ent["kt"]["file"]), R.pools / f"{st}.kt.npy", ent["kt"]["sha256"])}
    placed = {k: WQ.place(f, dst, sha) for k, (f, dst, sha) in files.items()}
    if out.exists():
        old = read_json(out)
        old_kt = old.get("d83", {}).get("record_kt", old.get("kt", {}))
        if old.get("sha256") != ent["sha256"] or old_kt.get("sha256") != ent["kt"]["sha256"] or old.get("carve") != CARVE:
            raise SystemExit(f"D83 b1v: {rel(out)} is present and names other pools than B1P's")
        return 0
    new = json.loads(json.dumps(ent))
    new["carve"] = CARVE
    new["file"] = rel(files["file"][1])
    new["kt"] = dict(ent["kt"], file=rel(files["kt"][1]))
    new["d83"] = {"declared_in": DECLARED, "stage": STAGE, "arm": ARM, "g1b": bool(g1b), "linked_from": rel(man_src),
                  "record_manifest_sha256": sha_file(man_src), "record_carve": RECORD_CARVE, "placed": placed, "utc": utc()}
    write_json(out, new)
    log(f"link {'G1b ' if g1b else ''}{name}/{CARVE}: B1P's pools of {RECORD_CARVE} {placed}")
    return 0


@contextlib.contextmanager
def kt_context(g1b=False):
    """zd83.kt_pass and kt_merge on b1v's roots and carve, under zb1h's setting context (zabl's names they read, put back)."""
    D = zd83()
    out_base = bind(g1b)
    Z = D.Z
    P_ = Z.B.Patches()
    try:
        P_.set(Z, roots_of=lambda M, _arm: M.PK.roots(out_base), CARVE=CARVE,
               on_arm=lambda M, name, _arm: H.on_b1h(M, name, {}))
        P_.set(D, kt_shards=lambda _name: KT_SHARDS)
        yield D
    finally:
        P_.restore()


def kt_pass(name, index, host=False, g1b=False):
    with kt_context(g1b) as D:
        return D.kt_pass(ARM, name, index, host)


def g1b_path():
    return OUT / "d83b_g1b.json"


def g1b_passed(name):
    p = g1b_path()
    return p.exists() and (read_json(p).get("settings", {}).get(name) or {}).get("verdict") == "PASS"


def gates_passed(name):
    """Section 8: no b1v KT pass, labelled link or read before G1 (zd83 g1) and the setting's G1b passed."""
    if not zd83().g1_passed():
        raise SystemExit("D83 b1v: G1 has not passed (outputs/abl/D83I/d83_g1.json); b1v waits")
    if not g1b_passed(name):
        raise SystemExit(f"D83 b1v: G1b has not passed on {name} ({rel(g1b_path())}); b1v waits")
    return True


def kt_merge(name, host=False, g1b=False):
    """zd83.kt_merge on b1v's roots; on b1v (not G1b) only once G1 and the setting's G1b passed."""
    if not g1b:
        gates_passed(name)
    with kt_context(g1b) as D:
        return D.kt_merge(ARM, name, host)


def g1b(name, host=False):
    """G1b on one setting: b1w's own folder through b1v's settings, pools link, KT pass and merge; PASS when the KT rows
    equal B1P's filed KT rows of b1w. The verdict is added to outputs/b1v/d83b_g1b.json."""
    t0 = time.time()
    js = G1B / "settings" / f"{name}.json"
    if not js.exists():
        settings(name, host, g1b=True)
    link_pools(name, host, g1b=True)
    kt_pass(name, (0, KT_SHARDS), host, g1b=True)
    kt_merge(name, host, g1b=True)
    M = H.zb().mods(host)
    PK = M.PK
    st = PK.stem(name, CARVE)
    man = PK.roots(G1B).pools / f"{st}.json"
    ent = read_json(man)
    d = ent["d83"]
    rec_kt = PK.resolve(d["record_kt"]["file"])
    same = bool(d.get("kt_equal_to_record")) and sha_file(rec_kt) == d["record_kt"]["sha256"]
    new = np.load(PK.resolve(ent["kt"]["file"]), allow_pickle=False)
    old = np.load(rec_kt, allow_pickle=False)
    same = same and bool(np.array_equal(new, old))
    p = g1b_path()
    rec = read_json(p) if p.exists() else {"declared_in": DECLARED, "stage": STAGE, "gate": "G1b", "settings": {}}
    rec["settings"][name] = {"verdict": "PASS" if same else "FAIL", "rows": int(new.shape[0]), "kt_sha256": ent["kt"]["sha256"],
                             "record_kt": d["record_kt"], "record_manifest_sha256": d["record_manifest_sha256"],
                             "kind_changed": d.get("kind_changed"), "manifest": rel(man), "manifest_sha256": sha_file(man),
                             "settings_sha256": sha_file(js), "zd83b_sha256": sha_src(__file__), "seconds": round(time.time() - t0, 1),
                             "utc": utc()}
    write_json(p, rec)
    log(f"G1b {name}: {rec['settings'][name]['verdict']} ({new.shape[0]} KT rows against B1P's {old.shape[0]})")
    return 0 if same else 1


# ── zb1h's chain on b1v ──────────────────────────────────────────────────────

@contextlib.contextmanager
def step_ctx(kind, name):
    """The links step builds labelled links on b1v's label sets (zd83.step_patches, labels under outputs/b1v/labels), the
    label record's node count checked against the settings' first."""
    if kind != "links":
        yield
        return
    with labels_here(OUT) as D:
        _keys, n, _sha = D.load_label_sets(ARM, name)
        sj = H.load_settings(name)[0]
        if int(sj["n_nodes"]) != int(n):
            raise SystemExit(f"D83 b1v {name}: the label sets are keyed on {n} nodes, the settings on {sj['n_nodes']}")
        with D.step_patches("links", ARM, name):
            yield


def run_step(kind, name, index=None, host=False):
    bind()
    if kind == "ktpass":
        gates_passed(name)
        return kt_pass(name, index or (0, KT_SHARDS), host)
    if kind == "ktmerge":
        return kt_merge(name, host)
    if kind not in ("look", "bases", "cache", "links", "kt", "record"):
        raise SystemExit(f"D83 b1v: step {kind}: one of {STEPS}")
    with step_ctx(kind, name):
        return H.run_step(kind, name, index, host, None)


def chain(name, host=False):
    """The pools link, the KT pass and merge (each its own process; G1 and G1b first), then zb1h's chain on the
    filed pools: its pools and merge are skipped, the look, bases, cache, labelled links, kt and record run."""
    gates_passed(name)
    link_pools(name, host)
    bind()
    M = H.zb().mods(host)
    PK = M.PK
    R = PK.roots(OUT)
    base = ([sys.executable, str(H.PYLIB_RUN)] if host else [sys.executable]) + [str(Path(__file__).resolve())]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               NUMBA_NUM_THREADS="2")
    st = PK.stem(name, CARVE)
    if not read_json(R.pools / f"{st}.json").get("d83", {}).get("kt_recomputed"):
        for i in range(KT_SHARDS):
            if not (OUT / "d83kt" / f"{st}__{i}of{KT_SHARDS}.json").exists():
                cmd = base + ["step", "ktpass", "--dataset", name, "--index", f"{i}/{KT_SHARDS}"] + (["--host"] if host else [])
                if subprocess.run(cmd, cwd=str(ROOT), env=env).returncode:
                    raise SystemExit(f"D83 b1v chain {name}: the KT pass {i}/{KT_SHARDS} failed")
        cmd = base + ["step", "ktmerge", "--dataset", name] + (["--host"] if host else [])
        if subprocess.run(cmd, cwd=str(ROOT), env=env).returncode:
            raise SystemExit(f"D83 b1v chain {name}: the KT merge failed")
    H.__file__ = str(Path(__file__).resolve())                  # zb1h's chain starts its steps as this file's
    try:
        return H.chain(name, host, None)
    finally:
        H.__file__ = _H_FILE


def gate(tag, host=False):
    bind()
    ZB = H.zb()
    M = ZB.mods(host)
    R = copy.copy(H.roots_of(M))
    R.gate = H.gate_path(tag)
    carves = [(ds, CARVE) for ds in TAGS[tag]]
    with ZB.checks(M):
        verdict = M.PK.gate(R, carves=carves)
    write_json(OUT / f"d83b_gate_{tag}.json", {
        "declared_in": DECLARED, "stage": STAGE, "tag": tag, "pko_gate": rel(R.gate), "pko_gate_sha256": sha_file(R.gate),
        "verdict": verdict, "carves": carves, "settings": {ds: sha_file(H.settings_paths(ds)[1]) for ds in TAGS[tag]},
        "g1b": {ds: read_json(g1b_path())["settings"][ds]["verdict"] for ds in TAGS[tag]},
        "zb1h_sha256": sha_src(_H_FILE), "zd83b_sha256": sha_src(__file__), "utc": utc()})
    return 0


# ── LAB, LTM's lists and columns ─────────────────────────────────────────────

def lab(name):
    import zd83m as M83
    M83.lab(ARM, name, CARVE, root=OUT)
    return 0


def ltm_ready(name):
    """In the D83 world: the label axis and LTM's five bits on, every links part of the carve b1v's labelled links."""
    import ltmcols as LTMC                                      # first: its BLAS thread environment before numpy
    import zd83m as M83
    import lean_cache as LC
    M83.world_ok("D83 b1v ltm")
    M83.labels_on()
    M83.ltm_on(LTMC.LTW)
    parts, _recs = LC.part_dirs(name, CARVE, OUT / "cache")
    for p in parts:
        M83.links_ok(OUT / "links" / name / CARVE / p.name, ARM)
    return LTMC


def lists(name):
    ltm_ready(name)
    bind()
    return H.lists(name)


def cols(split, name, host=False):
    ltm_ready(name)
    bind()
    return H.cols(split, name, host)


# ── the self-test ────────────────────────────────────────────────────────────

def selftest():
    D = zd83()
    assert tuple(D.LABELS) == LABELS and CARVE not in ("b1u", "b1w", "b1t", "b1n", "b1x", "b1o", "b1")
    rng = np.random.default_rng(0)
    # row_keys is setting_store's pair set: a fake m3b_pools hands back the graph setting_store builds
    import look_b1 as LB
    m, n = 40, 1000
    rows = rng.choice(np.arange(n), size=m, replace=False).astype(np.int64)
    rows[7] = rows[3]                                           # a repeated passage: one row
    s = rng.integers(0, m, 300)
    d = rng.integers(0, m, 300)
    w = rng.random(300).astype(np.float32)
    fake = SimpleNamespace(FamilyStore=SimpleNamespace(from_graph=lambda g, fam: g))
    st_ = SimpleNamespace(rows=rows, graphs={"knn": {"src": s.astype(np.int32), "dst": d.astype(np.int32), "weight": w}})
    g = LB.setting_store(fake, st_, "knn", n)
    want = np.unique(np.minimum(g.src, g.dst).astype(np.int64) * n + np.maximum(g.src, g.dst))
    assert np.array_equal(row_keys(s, d, rows, n), want)
    # bench_arrays: F0T = L0 + L0T; the kNN pairs R holds removed; labels before the dedup
    def tab(k):
        a, b = rng.integers(0, m, k), rng.integers(0, m, k)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        u = np.unique(lo[lo < hi] * m + hi[lo < hi])
        return (u // m).astype(np.int32), (u % m).astype(np.int32)
    l0, l0t = tab(60), tab(30)
    f0t = np.union1d(l0[0].astype(np.int64) * m + l0[1], l0t[0].astype(np.int64) * m + l0t[1])
    ner = {"src": (f0t // m).astype(np.int32), "dst": (f0t % m).astype(np.int32), "weight": np.ones(f0t.size, np.float32)}
    sa, sb = tab(20)
    stg = {"src": np.r_[sa, sb].astype(np.int32), "dst": np.r_[sb, sa].astype(np.int32), "rel": np.zeros(2 * sa.size, np.int16)}
    ka, kb = tab(80)
    ka, kb = np.r_[ka, l0[0][:5], sa[:3]].astype(np.int32), np.r_[kb, l0[1][:5], sb[:3]].astype(np.int32)   # some kNN pairs R holds
    knn = {"src": ka, "dst": kb, "weight": rng.random(ka.size).astype(np.float32)}
    keep, sets, counts = bench_arrays(stg, ner, knn, l0, l0t, rows, n)
    held = np.union1d(row_keys(stg["src"], stg["dst"], rows, n), row_keys(ner["src"], ner["dst"], rows, n))
    assert not np.isin(entry_row_keys(ka[keep], kb[keep], rows, n), held).any()
    assert np.isin(entry_row_keys(ka[~keep], kb[~keep], rows, n), held).all() and (~keep).sum() >= 8
    assert np.array_equal(sets["K.knn"], row_keys(ka, kb, rows, n)) and sets["S"].size == 0
    assert np.array_equal(sets["R.title_comention"], np.setdiff1d(row_keys(ner["src"], ner["dst"], rows, n), sets["R.name_comention"]))
    assert np.array_equal(sets["R.mention"], row_keys(stg["src"], stg["dst"], rows, n))
    bad_ner = dict(ner, src=ner["src"][1:], dst=ner["dst"][1:])
    try:
        bench_arrays(stg, bad_ner, knn, l0, l0t, rows, n)
        raise AssertionError("an NER file other than L0 + L0T must be refused")
    except SystemExit:
        pass
    # the lists check: three swapped files accepted, a fourth refused
    b = {"u1a": {"b1a_structural_sha256": "s0"}, "ap2": {"b1a_ner_sha256": "n0"}, "d83": {"b1a_knn_sha256": "k0"}}
    named = {STRUCT: "s0", NER: "n0", KNN: "k0", "nodes.json": "x", "queries.json": "q", "first_stage.npz": "f"}
    check_lists(named, dict(named, **{STRUCT: "s1", NER: "n1", KNN: "k1"}), b)
    for bad in ({"nodes.json": "y"}, {KNN: "k1", "queries.json": "r"}):
        try:
            check_lists(named, dict(named, **bad), b)
            raise AssertionError("a fourth file replaced must be refused")
        except SystemExit:
            pass
    try:
        check_lists(dict(named, **{KNN: "kz"}), dict(named, **{KNN: "k1"}), b)
        raise AssertionError("lists naming another kNN file must be refused")
    except SystemExit:
        pass
    # the label sets under b1v's root (zd83's writer and reader), and every other arm's left at zd83's
    tmp = Path(tempfile.mkdtemp(prefix="zd83b_t_"))
    try:
        with labels_here(tmp) as D2:
            D2.write_label_sets(ARM, "2wiki", n, sets)
            keys, nn, _sha = D2.load_label_sets(ARM, "2wiki")
            assert nn == n and all(np.array_equal(k, sets[lab]) for k, lab in zip(keys, LABELS))
            assert D2.labels_dir("D83F", "2wiki") == D2.Z.arm_dir("D83F") / "labels" / "2wiki"
        assert (tmp / "labels" / "2wiki" / "record.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert D.labels_dir(ARM, "2wiki") == D.Z.arm_dir(ARM) / "labels" / "2wiki"           # put back
    # bind: zb1h's globals are b1v's (or G1b's), the pools step refused
    saved = {k: getattr(H, k) for k in ("OUT", "B1A", "CARVE", "on_b1h", "pools", "GRAPH_TEXT")}
    try:
        assert bind() == OUT and H.CARVE == CARVE and H.B1A == BENCH_V and H.on_b1h is on_b1v and H.settings_paths("2wiki")[1] == OUT / "settings" / "2wiki.json"
        assert bind(g1b=True) == G1B and H.B1A == B1W and H.on_b1h is on_g1b and H.settings_paths("2wiki")[1] == G1B / "settings" / "2wiki.json"
        try:
            H.pools("2wiki")
            raise AssertionError("the pools step must be refused")
        except SystemExit:
            pass
    finally:
        for k, v in saved.items():
            setattr(H, k, v)
    # the KT context: zabl's names point at b1v's roots and carve inside, and are put back
    Z = D.Z
    keep_z = (Z.roots_of, Z.CARVE, Z.on_arm, D.kt_shards)
    with kt_context() as D3:
        assert Z.CARVE == CARVE and D3.kt_shards("2wiki") == KT_SHARDS
        assert Z.roots_of(SimpleNamespace(PK=SimpleNamespace(roots=lambda b: b)), "x") == OUT
    assert (Z.roots_of, Z.CARVE, Z.on_arm, D.kt_shards) == keep_z
    for k, v in saved.items():
        setattr(H, k, v)
    # the label axis is still free in a fresh process after this module loads (lists and cols turn it on)
    code = ("import sys; sys.path.insert(0, r'%s'); import zd83b, zd83m as M; M.labels_on(); import zlink as ZL; "
            "assert ZL.FAMS == 5 and ZL.SETTINGS['families'] == 5" % str(HERE))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr[-2000:]
    print("zd83b self-test: OK")
    return 0


def parse_index(s):
    return H.parse_index(s) if s else None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zd83b.py")
    ap.add_argument("cmd", choices=("selftest", "bench", "settings", "link", "g1b", "chain", "step", "gate", "lab", "lists", "cols"))
    ap.add_argument("kind", nargs="?")
    ap.add_argument("--dataset", choices=SETTINGS)
    ap.add_argument("--tag", choices=tuple(TAGS))
    ap.add_argument("--split")
    ap.add_argument("--index")
    ap.add_argument("--root")
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    if a.root and Path(a.root).resolve() != OUT.resolve():
        raise SystemExit("D83 b1v: the carve's roots are outputs/b1v")
    need = {"bench": ("dataset",), "settings": ("dataset",), "link": ("dataset",), "g1b": ("dataset",), "chain": ("dataset",),
            "step": ("dataset", "kind"), "gate": ("tag",), "lab": ("dataset",), "lists": ("dataset",), "cols": ("dataset", "split")}
    miss = [k for k in need.get(a.cmd, ()) if getattr(a, k) is None]
    if miss:
        raise SystemExit(f"D83 b1v {a.cmd}: needs {miss}")
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "bench":
        return bench(a.dataset)
    if a.cmd == "settings":
        return settings(a.dataset, a.host)
    if a.cmd == "link":
        return link_pools(a.dataset, a.host)
    if a.cmd == "g1b":
        return g1b(a.dataset, a.host)
    if a.cmd == "chain":
        return chain(a.dataset, a.host)
    if a.cmd == "step":
        return run_step(a.kind, a.dataset, parse_index(a.index), a.host)
    if a.cmd == "gate":
        return gate(a.tag, a.host)
    if a.cmd == "lab":
        return lab(a.dataset)
    if a.cmd == "lists":
        return lists(a.dataset)
    return cols(a.split, a.dataset, a.host)


if __name__ == "__main__":
    raise SystemExit(main())
