"""Design look (untracked; not a result and not filed): S6 part 11 of the transfer plan, which granularity transforms
carry the KB-to-KB transfer, and three new ones. Part 3 (chainscore19) found that the transforms of metaqa's fit carve
are what carries its scorer to webqsp zero-shot (em-gr over em +10.7 R@5 on webqsp selectf + fit, -4.8 on metaqa).
Parts 8 to 10 keep that KB group fixed (metaqa's fit carve and al4 sp4 mg3 rf: "k5") and vary what is trained beside
it; part 8 (chainscore24, graded early on 5 Oct) found no passage mix ABOVE k5 for the MLP f on webqsp. This part
varies the KB group itself, for both f's: which of the four transforms carries its weight (leave one out), and whether
three more, each a trait of webqsp's Freebase graph that metaqa's lacks, add to it. Swastik (4 Oct): train 'in
multiple regimes getting from coarser to fine grained labels'; 'more distributions should work'.
    rfk   rf that keeps the fact. rf's half of the present relations (rng SEED + 5, the same draw) gets, beside each
          (h, r, t), a mediator path (h, n_rel + r, m), (m, 2 n_rel + r, t): m one node per distinct (h, r, t) of the
          row, its text n(p_h + p_t), its rrf max(rrf_h, rrf_t), never gold, as rf's. Freebase holds many facts both
          as a direct edge and through a CVT; rf removes the direct edge, so every reified fact is two steps away.
          Every part keeps r's name
    inv   inverse edges. A random half of the present relations (rng SEED + 6) gets (t, n_rel + r, h) beside each
          (h, r, t); no new node. Freebase pairs many relations with a reverse one (film.film.directed_by,
          film.director.film); metaqa has none. The inverse keeps r's name
    al16  r -> 16 r + a hash of (global head, global tail, r) mod 16: al4 with 16 aliases. webqsp's graph has 4,347
          relation ids and metaqa's 9; al4 reaches 36, al16 144. Each alias keeps r's name
rfk and inv rewrite each row's structure (chainscore19's reify_row path: the row's walks, levels, node features and
chainscore20's companion messages are rebuilt on the rewritten row, the union graph's endpoint descriptors on the
rewritten union graph); al16 relabels (its structure, none level and node features are id's). This file's Transform
is chainscore19's with the three added (the five old ones are chainscore19's own code), patched into chainscore19 for
the build and into chainscore20's companion; everything else is their statements. Built on metaqa's fit carve as the
old transforms were: rfk hops 3, ks 2,4, regimes 1, 0.05, 0 on the even rows (rf's settings and rows, so that rfk and
rf differ only in the direct edges kept); inv the same on the odd rows; al16 hops 2, ks 2,4,16, regimes 1, 0.05, 0 on
the odd rows (al4's settings, sp4's rows). Each build's chainscore20 companion; cs_cache train entries under pq.

Sets (the transforms beside metaqa's fit carve, id; one group: chainscore19's draw, the build uniform among the set's):
    k5        al4 sp4 mg3 rf (parts 8 to 10's KB group): not retrained but for seed 0 (R0)
    lo-al4, lo-sp4, lo-mg3, lo-rf        k5 less one
    p-rfk, p-inv, p-al16                 k5 plus one
    sw-rfk    al4 sp4 mg3 rfk (rf swapped for rfk)
    k8t       k5 plus all three
for both f's under part 6's carried map pq: the GNN f (ena-gr with g and lb: part 9's e-b-lb-pq) and the MLP f (em-gr
with g and lb: part 8's k5). k5's three seeds are part 9's e-b-lb-pq runs (GNN f) and part 8's k5 runs (MLP f), read
from their rows files by sha256, on the same device and map; each f's k5 seed 0 is rerun here (R0).
Training as parts 8 and 9: Adam 1e-3, weight decay 1e-4, 12 epochs of 5,960 examples in batches of 64, the epoch
chosen by R@5 on metaqa's select carve (the KB rule), seeds SEED + 0, 1, 2, inputs standardised on the fit build, the
host GPU with cs_dev's settings (deterministic algorithms, TF32 off): no number here is set beside a CPU fit's. The
GNN f's statements are part 9's train25, the MLP f's part 8's train24 (lam 0), each called with the set's builds as
one group; every input comes from cs_cache under pq, each companion must be its build's (chainscore20's load_comp),
each passage read's sidecar must name its companion (chainscore21's check_passage).
Reads. The KB rule on metaqa x1f, webqsp selectf and webqsp selectf + fit (webqsp is never trained on); the passage
rule on 2wiki x1 and hotpotqa x1 in the run, and musique x1f: in the run for the MLP f (part 8's form), by its own
job from the run's saved weights for the GNN f (pread). Until every musique read is in, the grade marks musique
NOT_READ for the GNN f and a second grade fills it; the decision does not use musique.
Verdicts, fixed at 02:05 on 5 Oct (this file's first write), before any build of rfk, inv or al16 and before any run
of this part, part 9's grade or part 10's runs. Per f and set, each row's R@5 averaged over the three seeds; paired
row bootstrap (BOOT 1000):
    T1 (primary) webqsp selectf + fit: each set minus k5: ABOVE / AT / BELOW
    T2 metaqa x1f: T1's differences
    T3 leave one out, on every read: k5 minus lo-X: X CARRIES (above 0) / COSTS (below) / NEUTRAL
    T4 webqsp selectf + fit by seed-gold distance D (1, 2): T1's differences
    T5 each passage read (zero-shot for every set): each set minus k5, and every set minus s0+rrf with its share of
       the twin's lead over s0+rrf (HIGH at or above 0.5, LOW below 0.25, else MID)
    T6 the GNN f minus the MLP f, each set, on every read graded for both
    T7 each KB read: every set minus none/rd (the untyped walk)
    S  every difference above per seed (seed k's rows minus seed k's), with its sign
    R0 per f: k5 seed 0 through this file equals part 9's e-b-lb-pq seed 0 (GNN f) or part 8's k5 seed 0 (MLP f) row
       for row (|diff| < 1e-6) on every read in the run, and weight for weight (state_sha256): REPRODUCES / DIFFERS
Decision, per f. A set ABOVE k5 on webqsp selectf + fit, with each of its three seeds' differences there above 0, and
not BELOW on metaqa carries (the largest webqsp selectf + fit mean difference if several): the next parts train that
f on that set. Nine sets are set against k5, so an ABOVE also needs its seeds to agree. If none carries, k5 stays and
the result goes to Swastik. The GNN f's decision is the headline (part 4: ena-gr carries to webqsp); an f's R0 DIFFERS
blocks its decision until the cause is found. Train-split rows throughout: a look, not a result.
    python outputs/mp_unified/chainscore27.py build --transform rfk --out outputs/mp_unified/lean/cs27-mq-fit-rfk.npz
    python outputs/mp_unified/chainscore27.py edges --base outputs/mp_unified/lean/cs27-mq-fit-rfk.npz \
        --out outputs/mp_unified/lean/cs27e-mq-fit-rfk.npz
    python outputs/mp_unified/cs_cache.py build --kind train --map pq --src outputs/mp_unified/lean/cs27-mq-fit-rfk.npz
    python outputs/mp_unified/chainscore27.py train --f gnn --set p-rfk --seed 0 --device cuda \
        --map-from outputs/mp_unified/lean/cs22.json --cache outputs/mp_unified/cache \
        --fit outputs/mp_unified/lean/cs19-mq-fit-id.npz --aug al4=outputs/mp_unified/lean/cs19-mq-fit-al4.npz \
        (sp4, mg3, rf: cs19-mq-fit-*; rfk: cs27-mq-fit-rfk) --select outputs/mp_unified/lean/cs19-mq-select.npz \
        --read metaqa=... --read webqsp=... --read webqsp_sf=... --pread 2wiki=... --pread hotpotqa=... \
        [the MLP f: --pread musique=...] [the GNN f: --edges NAME=COMPANION for fit, each aug, select and each read] \
        --out X.json --rows-out X.rows.npz --state-out X.pt
    python outputs/mp_unified/chainscore27.py pread --run X.json \
        --pread musique=outputs/mp_unified/lean/cs21-mu-x1f.npz --edges musique=outputs/mp_unified/lean/cs21e-mu-x1f.npz \
        --cache outputs/mp_unified/cache --device cuda --out X.mu.json --rows-out X.mu.rows.npz
    python outputs/mp_unified/chainscore27.py grade --run X.json (both f's: every set x seed, and k5 seed 0) \
        --ref9 outputs/mp_unified/lean/cs25-e-b-lb-pq-s0.json (s1, s2) --ref8 outputs/mp_unified/lean/cs24-k5-s0.json \
        (s1, s2) [--mu-run X.mu.json ... --ref9-mu outputs/mp_unified/lean/cs25-e-b-lb-pq-s0.mu.json ...] \
        --read ... --pread 2wiki=... --pread hotpotqa=... --pread musique=... --out outputs/mp_unified/lean/cs27.json
    python outputs/mp_unified/chainscore27.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainscore19's count
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainpop17 as CP  # noqa: E402
import chainscore18 as CS  # noqa: E402
import chainscore19 as C19  # noqa: E402
import chainscore20 as C20  # noqa: E402
import chainscore21 as C21  # noqa: E402
import chainscore22 as C22  # noqa: E402
import chainscore24 as C24  # noqa: E402
import chainscore25 as C25  # noqa: E402
import cs_cache as CC  # noqa: E402
import cs_dev as CD  # noqa: E402

SEED, BOOT = C19.SEED, C19.BOOT
BATCH = C21.BATCH
NORM = "pq"
SEEDS = (0, 1, 2)
TF4 = C19.TRANSFORMS[1:]
NEW = ("rfk", "inv", "al16")
ORDER = TF4 + NEW                       # a set's builds in this order, after the fit carve's (id)
SETS = {"k5": TF4,
        "lo-al4": ("sp4", "mg3", "rf"), "lo-sp4": ("al4", "mg3", "rf"), "lo-mg3": ("al4", "sp4", "rf"),
        "lo-rf": ("al4", "sp4", "mg3"),
        "p-rfk": TF4 + ("rfk",), "p-inv": TF4 + ("inv",), "p-al16": TF4 + ("al16",),
        "sw-rfk": ("al4", "sp4", "mg3", "rfk"), "k8t": TF4 + NEW}
CANDS = tuple(s for s in SETS if s != "k5")         # every set is set against k5; k5 runs here only as R0
LOTO = {"lo-al4": "al4", "lo-sp4": "sp4", "lo-mg3": "mg3", "lo-rf": "rf"}
FS = {"gnn": "ena-gr", "mlp": "em-gr"}
REF9, REF8 = "e-b-lb-pq", "k5"          # part 9's GNN-f arm, part 8's MLP-f arm: k5's seeds
KB_READS = C21.KB_READS
P_RUN = {"gnn": ("2wiki", "hotpotqa"), "mlp": ("2wiki", "hotpotqa", "musique")}    # passage reads in the run
P_READS = ("2wiki", "hotpotqa", "musique")
P_PREAD = ("musique",)                  # the GNN f's, by pread
BUILD = {"rfk": {"hops": 3, "ks": "2,4", "regimes": "1,0.05,0", "rows": "even"},
         "inv": {"hops": 3, "ks": "2,4", "regimes": "1,0.05,0", "rows": "odd"},
         "al16": {"hops": 2, "ks": "2,4,16", "regimes": "1,0.05,0", "rows": "odd"}}
ABV = ("ABOVE", "BELOW", "AT")
LOV = ("CARRIES", "COSTS", "NEUTRAL")
log, sha = C21.log, CC.sha
unit = C19.unit


# ── the transforms ──────────────────────────────────────────────────────────────────────────────────────────────


class Transform(C19.Transform):
    """chainscore19's Transform with rfk, inv and al16; the five old ones are chainscore19's code, unchanged. rfk and
    inv keep the name "rf" in .name (chainscore19's collect19 and chainscore20's collect_edges call reify_row on "rf");
    .info["name"] is the transform's own name (the build's meta, checked by chainscore20's edges and the runs)."""

    def __init__(self, name, n_rel, G, Pn, present):
        if name in C19.TRANSFORMS:
            super().__init__(name, n_rel, G, Pn, present)
            return
        if name not in NEW:
            raise SystemExit(f"unknown transform {name}")
        self.n_rel0, self.nodes = n_rel, G["nodes"]
        self.info = {"name": name}
        if name == "al16":
            self.name, self.n_rel = "al16", 16 * n_rel
        else:
            self.name = "rf"
            idx = np.flatnonzero(present)
            pick = np.random.default_rng(SEED + (5 if name == "rfk" else 6)).choice(idx, size=idx.size // 2,
                                                                                    replace=False)
            self.reif = np.zeros(n_rel, bool)
            self.reif[pick] = True
            self.n_rel = (3 if name == "rfk" else 2) * n_rel
            self.info["reified" if name == "rfk" else "inverted"] = sorted(int(x) for x in pick)
        self.info["n_rel"] = self.n_rel

    def relabel(self, S, hg, tg):
        if self.info["name"] != "al16":
            return super().relabel(S, hg, tg)
        ok = S >= 0
        out = np.full(S.shape, -1, np.int64)
        r = S[ok].astype(np.int64)
        tri = np.nonzero(ok)[0]
        out[ok] = 16 * r + C19.alias(hg[tri], tg[tri], r, 16)
        return out

    def names(self, name_vec):
        nm = self.info["name"]
        if name_vec is None or nm in C19.TRANSFORMS:
            return super().names(name_vec)
        if nm == "al16":
            return np.repeat(name_vec, 16, axis=0)
        return np.concatenate([name_vec] * (self.n_rel // self.n_rel0))

    def descriptors(self, G, Pn):
        """chainpop17's endpoint descriptors on the transformed union graph (chainscore19's, with the new rewrites)."""
        nm = self.info["name"]
        if nm in C19.TRANSFORMS:
            return super().descriptors(G, Pn)
        S = G["slots"]
        ok = S >= 0
        tri = np.repeat(np.arange(S.shape[0]), ok.sum(1))
        nn = Pn.shape[0]
        h, t = G["H"][tri], G["T"][tri]

        def acc(ty, v):
            return np.asarray(sp.csr_matrix((np.ones(ty.size), (ty, v)), shape=(self.n_rel, nn)) @ Pn,
                              dtype=np.float64)

        if nm == "al16":
            ty = self.relabel(S, self.nodes[G["H"]], self.nodes[G["T"]])[ok].astype(np.int64)
            Dt, Dh = acc(ty, t), acc(ty, h)
            present = np.bincount(ty, minlength=self.n_rel) > 0
        else:
            n0 = self.n_rel0
            ty = S[ok].astype(np.int64)
            Dt, Dh = acc(ty, t), acc(ty, h)         # every fact keeps its direct edge
            m = self.reif[ty]
            hm, tm, rm = h[m], t[m], ty[m]
            if nm == "rfk":
                med = unit(Pn[hm] + Pn[tm]).astype(np.float64)
                np.add.at(Dt, n0 + rm, med)
                np.add.at(Dh, n0 + rm, Pn[hm].astype(np.float64))
                np.add.at(Dt, 2 * n0 + rm, Pn[tm].astype(np.float64))
                np.add.at(Dh, 2 * n0 + rm, med)
                present = np.bincount(np.r_[ty, n0 + rm, 2 * n0 + rm], minlength=self.n_rel) > 0
            else:
                np.add.at(Dt, n0 + rm, Pn[hm].astype(np.float64))
                np.add.at(Dh, n0 + rm, Pn[tm].astype(np.float64))
                present = np.bincount(np.r_[ty, n0 + rm], minlength=self.n_rel) > 0
        tt = np.c_[unit(Dt), unit(Dh)] / math.sqrt(2)
        return tt, present

    def reify_row(self, n, hl, tl, sl, pn, rrf):
        """rf, rfk or inv on one row: the distinct (h, t, r) of the row's triples, rewritten. Returns the new pool size,
        triples (one slot each), texts and rrf (mediators appended after the pool's nodes)."""
        nm = self.info["name"]
        if nm == "rf":
            return super().reify_row(n, hl, tl, sl, pn, rrf)
        if nm not in ("rfk", "inv"):
            raise SystemExit(f"{nm} relabels; it does not rewrite a row's structure")
        n0 = self.n_rel0
        ok = sl >= 0
        k = ok.sum(1)
        key = np.unique((np.repeat(hl, k) * n + np.repeat(tl, k)) * n0 + sl[ok].astype(np.int64))
        r = key % n0
        ht = key // n0
        h, t = ht // n, ht % n
        m = self.reif[r]
        if nm == "inv":
            return n, np.r_[h, t[m]], np.r_[t, h[m]], np.r_[r, n0 + r[m]][:, None], pn, rrf
        M = int(m.sum())
        med = n + np.arange(M, dtype=np.int64)
        hl2 = np.r_[h, h[m], med]
        tl2 = np.r_[t, med, t[m]]
        sl2 = np.r_[r, n0 + r[m], 2 * n0 + r[m]][:, None]
        pn2 = np.r_[pn, unit(pn[h[m]] + pn[t[m]])].astype(np.float32)
        rrf2 = np.r_[rrf, np.maximum(rrf[h[m]], rrf[t[m]])]
        return n + M, hl2, tl2, sl2, pn2, rrf2


_C19_TRANSFORM = C19.Transform


def patch(on=True):
    """chainscore19's build and chainscore20's companion construct C19.Transform: this file's, or chainscore19's."""
    C19.Transform = Transform if on else _C19_TRANSFORM


# ── build and companion (chainscore19's and chainscore20's statements, this file's Transform) ──────────────────


def build27(a, root=C19.LOOK, rel_dir=C19.REL):
    if a.transform not in NEW:
        raise SystemExit(f"this file builds {NEW}; the old transforms' builds are chainscore19's")
    a.chainscore27_sha256 = sha(__file__)        # the build's meta args record the transform's code
    patch()
    try:
        return C19.build19(a, root=root, rel_dir=rel_dir)
    finally:
        patch(False)


def build_cmd(a):
    b = SimpleNamespace(ds=a.ds, carve=a.carve, transform=a.transform, max_chains=a.max_chains, limit=a.limit,
                        out=a.out, **BUILD[a.transform])
    return build27(b)


def edges_cmd(a, root=C19.LOOK):
    with np.load(a.base, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    nm = (meta.get("transform") or {}).get("name")
    if meta.get("look") != "chainscore19" or nm not in NEW or "chainscore27_sha256" not in meta.get("args", {}):
        raise SystemExit(f"{a.base}: not a build of {NEW} by this file (an old build's companion is chainscore20's)")
    a.chainscore27_sha256 = sha(__file__)
    patch()
    try:
        return C20.edges_cmd(a, root=root)
    finally:
        patch(False)


# ── train (one f, one set, one seed) ────────────────────────────────────────────────────────────────────────────


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_rows(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def train_cmd(a):
    t0 = time.time()
    if a.set == "k5" and a.seed != 0:
        raise SystemExit("k5 runs here only as R0 (seed 0); its seeds are part 9's and part 8's runs")
    f, gnn = FS[a.f], a.f == "gnn"
    minfo = C25.carried_map(a.map_from)
    if not a.cache:
        raise SystemExit("the runs load their inputs from cs_cache (--cache)")
    dev = None if a.device == "cpu" else a.device
    spec = C21.arm_spec("b-lb", f)
    res = {"look": "chainscore27", "fk": a.f, "f": f, "arm": a.set, "set": list(SETS[a.set]), "seed": a.seed,
           "norm": NORM, "device": a.device, "args": dict(vars(a)), "map_rule": minfo, "cache": a.cache,
           "state_out": a.state_out, "script_sha256": sha(__file__), "chainscore25_sha256": sha(C25.__file__),
           "chainscore24_sha256": sha(C24.__file__), "cs_cache_sha256": sha(CC.__file__),
           "chainscore22_sha256": sha(C22.__file__), "chainscore21_sha256": sha(C21.__file__),
           "chainscore20_sha256": sha(C20.__file__), "chainscore19_sha256": sha(C19.__file__),
           "chainscore18_sha256": sha(CS.__file__), "cs_dev_sha256": sha(CD.__file__), "inputs": {},
           "inputs_sha256": {}, "passage_types": {}, "maps": {}, "edges": {}, "reads": {}, "rows_out": a.rows_out,
           "timing": {}}
    aug = dict(x.split("=", 1) for x in a.aug)
    if sorted(aug) != sorted(SETS[a.set]):
        raise SystemExit(f"set {a.set} takes --aug for exactly {SETS[a.set]}")
    if not a.fit or not a.select:
        raise SystemExit("every run takes --fit and --select")
    kb = [tuple(x.split("=", 1)) for x in a.read]
    ps = [tuple(x.split("=", 1)) for x in a.pread]
    if sorted(n for n, _ in kb) != sorted(KB_READS) or sorted(n for n, _ in ps) != sorted(P_RUN[a.f]):
        raise SystemExit(f"the {a.f} f reads {KB_READS} (--read) and {P_RUN[a.f]} (--pread)")
    train_in = [("fit", a.fit)] + [(t, aug[t]) for t in ORDER if t in aug]
    reads = [(n, p, False) for n, p in kb] + [(n, p, True) for n, p in ps]
    path = dict(train_in)
    path["select"] = a.select
    path.update({n: p for n, p, _ in reads})
    edges = dict(x.split("=", 1) for x in a.edges)
    if gnn and sorted(edges) != sorted(path):
        raise SystemExit(f"the GNN f takes --edges for each of {sorted(path)}; given {sorted(edges)}")
    if not gnn and edges:
        raise SystemExit("the MLP f takes no companions")
    t1 = time.time()
    for n, p in path.items():
        res["inputs_sha256"][p] = sha(p)
        if gnn:
            res["inputs_sha256"][edges[n]] = sha(edges[n])
    for n, p, is_p in reads:
        if is_p:
            side = C21.check_passage(p, res["inputs_sha256"][p], res["inputs_sha256"][edges[n]] if gnn else None)
            res["passage_types"][n] = {"script_sha256": side["script_sha256"], **side["types"]}
    res["timing"]["hash_s"] = round(time.time() - t1, 1)

    def comp(n, d):
        return C25.load_comp(edges[n], d, res["inputs_sha256"][path[n]], spec["att"]) if gnn else None

    t1 = time.time()
    builds, comps, LB, st0 = [], [], [], None
    for n, p in train_in:
        d, lb, info, st = C25.get(p, "train", NORM, a.cache)
        tf = d["meta"]["transform"]["name"]
        want = "id" if n == "fit" else n
        if tf != want:
            raise SystemExit(f"{p} holds transform {tf}, not {want}")
        if n in NEW and "chainscore27_sha256" not in d["meta"]["args"]:
            raise SystemExit(f"{p}: a {n} build not made by this file")
        if not builds:
            st0 = st
        res["inputs"][n] = d["meta"]
        res["maps"][n] = info
        builds.append(d)
        comps.append(comp(n, d))
        LB.append(lb)
        log(f"  {a.f} {a.set} s{a.seed}: {n} ready (cache; {info})")
    groups = [list(range(len(builds)))]
    Sd, Slb, sinfo, _ = C25.get(a.select, "plain", NORM, a.cache)
    res["maps"]["select"] = sinfo
    res["inputs"]["select"] = Sd["meta"]
    res["train_rows"] = {n: int(b["train_rows"].size) for (n, _p), b in zip(train_in, builds)}
    res["groups"] = groups
    if gnn:
        res["edges"] = {n: {"meta_check": c["meta"]["check"], "steps": c["meta"]["steps"]}
                        for (n, _p), c in zip(train_in, comps)}
    res["timing"]["load_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    if gnn:
        Sc = comp("select", Sd)
        model, info = C25.train25(a.set, f, builds, comps, LB, groups, Sd, Sc, Slb, st0, a.epochs, a.per_epoch,
                                  a.batch, SEED + a.seed, device=dev)
        del Sc
    else:
        model, info = C24.train24(a.set, builds, LB, groups, [0] * len(builds), Sd, Slb, st0, a.epochs, a.per_epoch,
                                  a.batch, SEED + a.seed, 0.0, device=dev)
    res["timing"]["train_s"] = round(time.time() - t1, 1)
    del builds, comps, LB, Sd, Slb
    res.update({k: info[k] for k in ("history", "best_epoch", "best_select_r5", "coupling", "spec", "placement",
                                     "state_sha256", "seconds_per_batch", "deterministic_warnings")})
    t1 = time.time()
    rows_out = {}
    for n, p, is_p in reads:
        t2 = time.time()
        d, lb, rinfo, _ = C25.get(p, "plain", NORM, a.cache)
        c = comp(n, d)
        res["maps"][n] = rinfo
        rule = "passage" if is_p else "kb"
        xr, ml = CD.read_dev(model, d, c, lb, info["stats"], rule, dev, batch=a.read_batch)
        rows_out[n] = xr.astype(np.float32)
        rows_out[f"{n}__D"] = d["D"]
        res["reads"][n] = dict(C19.summarise(xr, d["D"], ml, d["level_names"]), rule=rule,
                               seconds=round(time.time() - t2, 1))
        res["inputs"][n] = d["meta"]
        log(f"  {a.f} {a.set} s{a.seed} on {n} ({rule} rule): {res['reads'][n]['mean']} ({time.time() - t2:.0f}s)")
        del d, c
    res["timing"]["reads_s"] = round(time.time() - t1, 1)
    p = Path(a.rows_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **rows_out)
    os.replace(tmp, p)
    if a.state_out:
        import torch
        sp_ = Path(a.state_out)
        torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "stats": info["stats"],
                    "spec": info["spec"], "fk": a.f, "f": f, "arm": a.set, "seed": a.seed, "norm": NORM,
                    "groups": groups}, sp_.with_name(sp_.name + ".tmp"))
        os.replace(sp_.with_name(sp_.name + ".tmp"), sp_)
    res["seconds"] = round(time.time() - t0, 1)
    log(f"  {a.f} {a.set} s{a.seed}: state {res['state_sha256'][:16]}, {res['seconds_per_batch']} s/batch, timing "
        f"{res['timing']}, {res['seconds']}s")
    return res


# ── pread: the GNN f's musique read from a run's saved weights ──────────────────────────────────────────────────


def pread_cmd(a):
    import torch
    t0 = time.time()
    run = load_json(a.run)
    if run.get("look") != "chainscore27" or run.get("fk") != "gnn" or not run.get("state_out"):
        raise SystemExit(f"{a.run}: not a chainscore27 GNN-f run with saved weights")
    if a.device != run["device"]:
        raise SystemExit(f"a read runs on its run's device ({run['device']}), not {a.device}")
    n, p = a.pread.split("=", 1)
    ne, pe = a.edges.split("=", 1)
    if n != ne or n not in P_PREAD:
        raise SystemExit(f"pread reads {P_PREAD} with its own --edges")
    dev = None if a.device == "cpu" else a.device
    block = CD.settings(dev, 1)
    S = torch.load(run["state_out"], map_location="cpu", weights_only=False)
    got = CD.state_sha(S["state"])
    if got != run["state_sha256"]:
        raise SystemExit(f"{run['state_out']}: weights {got[:16]}, the run's {run['state_sha256'][:16]}")
    if (S["fk"], S["arm"], int(S["seed"]), S["norm"]) != ("gnn", run["arm"], int(run["seed"]), NORM):
        raise SystemExit(f"{run['state_out']} holds {S['fk']} {S['arm']} seed {S['seed']} ({S['norm']})")
    f = FS["gnn"]
    spec = C21.arm_spec("b-lb", f)
    model = C20.make_model(f, SEED + int(run["seed"]))
    model.load_state_dict(S["state"])
    if CD.placed(dev):
        CD._dp().model_to(model, dev)
    model.eval()
    sp_, se = sha(p), sha(pe)
    side = C21.check_passage(p, sp_, se)
    d, lb, rinfo, _ = C25.get(p, "plain", NORM, a.cache)
    c = C25.load_comp(pe, d, sp_, spec["att"])
    t1 = time.time()
    xr, ml = CD.read_dev(model, d, c, lb, S["stats"], "passage", dev, batch=a.read_batch)
    rows = {n: xr.astype(np.float32), f"{n}__D": d["D"]}
    q = Path(a.rows_out)
    q.parent.mkdir(parents=True, exist_ok=True)
    tmp = q.with_name(q.stem + ".tmp.npz")
    np.savez_compressed(tmp, **rows)
    os.replace(tmp, q)
    res = {"look": "chainscore27-pread", "run": a.run, "fk": "gnn", "arm": run["arm"], "seed": int(run["seed"]),
           "f": f, "norm": NORM, "device": a.device, "placement": block, "state_sha256": got, "read": n,
           "inputs_sha256": {p: sp_, pe: se}, "passage_types": {n: {"script_sha256": side["script_sha256"],
                                                                    **side["types"]}},
           "maps": {n: rinfo}, "inputs": {n: d["meta"]}, "script_sha256": sha(__file__),
           "reads": {n: dict(C19.summarise(xr, d["D"], ml, d["level_names"]), rule="passage",
                             seconds=round(time.time() - t1, 1))}, "rows_out": a.rows_out}
    res["seconds"] = round(time.time() - t0, 1)
    log(f"  gnn {run['arm']} s{run['seed']} on {n} (passage rule): {res['reads'][n]['mean']} ({res['seconds']}s)")
    return res


# ── grade ───────────────────────────────────────────────────────────────────────────────────────────────────────


def grade_cmd(a):
    t0 = time.time()
    runs, rows, src, r0 = {}, {}, {}, {}       # key (fk, set, seed); src[key]: the json naming each read's file
    for p_ in a.run:
        js = load_json(p_)
        if js.get("look") != "chainscore27":
            raise SystemExit(f"{p_} is not a chainscore27 run")
        key = (js["fk"], js["arm"], int(js["seed"]))
        if js["arm"] == "k5":
            if js["fk"] in r0:
                raise SystemExit(f"two R0 runs of the {js['fk']} f")
            r0[js["fk"]] = (p_, js, load_rows(js["rows_out"]))
            continue
        if key in runs:
            raise SystemExit(f"two runs of {key}")
        runs[key], rows[key], src[key] = js, load_rows(js["rows_out"]), {"run": js}
    want = {(fk, s_, sd) for fk in FS for s_ in CANDS for sd in SEEDS}
    if set(runs) != want or set(r0) != set(FS):
        raise SystemExit(f"the grade takes both f's every set x seed and each f's k5 seed 0 (R0); missing "
                         f"{sorted(want - set(runs))}, R0 {sorted(r0)}, extra {sorted(set(runs) - want)}")
    devs = {js["device"] for js in runs.values()} | {v[1]["device"] for v in r0.values()}
    shas = {js["script_sha256"] for js in runs.values()} | {v[1]["script_sha256"] for v in r0.values()}
    if len(devs) != 1 or len(shas) != 1:
        raise SystemExit(f"runs on several devices {devs} or scripts {[s[:12] for s in shas]}")
    dev = next(iter(devs))
    for fk, xs, look, arm in (("gnn", a.ref9, "chainscore25", REF9), ("mlp", a.ref8, "chainscore24", REF8)):
        for p_ in xs:
            js = load_json(p_)
            if js.get("look") != look or js.get("via_dev") or js.get("arm") != arm:
                raise SystemExit(f"{p_} is not {look}'s {arm} run")
            key = (fk, "k5", int(js["seed"]))
            if key in rows:
                raise SystemExit(f"two runs of {key}")
            if js["device"] != dev or js.get("norm") != NORM:
                raise SystemExit(f"{p_} ran on {js['device']} under {js.get('norm')}; the runs on {dev} under {NORM}")
            runs[key], rows[key], src[key] = js, load_rows(js["rows_out"]), {"run": js}
    full = {(fk, s_, sd) for fk in FS for s_ in SETS for sd in SEEDS}
    if set(rows) != full:
        raise SystemExit(f"missing {sorted(full - set(rows))} (part 9's --ref9, part 8's --ref8)")
    mu = {}
    for flag, xs in (("mu", a.mu_run), ("ref9", a.ref9_mu)):
        for p_ in xs:
            js = load_json(p_)
            look = "chainscore27-pread" if flag == "mu" else "chainscore25-pread"
            if js.get("look") != look or js.get("read") not in P_PREAD:
                raise SystemExit(f"{p_} is not a {look} of {P_PREAD}")
            if flag == "ref9" and js["arm"] != REF9:
                raise SystemExit(f"{p_} reads part 9's {js['arm']}, not {REF9}")
            key = ("gnn", js["arm"] if flag == "mu" else "k5", int(js["seed"]))
            if key not in runs or js["state_sha256"] != runs[key]["state_sha256"] or js["device"] != dev:
                raise SystemExit(f"{p_}: not a read of the graded run {key} on {dev}")
            if (key, js["read"]) in mu:
                raise SystemExit(f"two preads of {key} on {js['read']}")
            mu[(key, js["read"])] = js
            rows[key].update(load_rows(js["rows_out"]))
            src[key][js["read"]] = js
    refs, Ds, rule = {}, {}, {}
    for flag, xs in (("kb", a.read), ("passage", a.pread)):
        for x in xs:
            nm_, p_ = x.split("=", 1)
            h = sha(p_)
            for k in full:
                if nm_ not in rows[k]:
                    continue
                js = src[k].get(nm_, src[k]["run"])
                if js["inputs_sha256"].get(p_) != h:
                    raise SystemExit(f"{k} read another {nm_} than {p_}")
            d = CS.load(p_)
            refs[nm_] = CS.reference_arms(d) if flag == "kb" else C21.passage_refs(d)
            Ds[nm_] = d["D"]
            rule[nm_] = flag
            del d
    if sorted(k for k in rule if rule[k] == "kb") != sorted(KB_READS) or \
            sorted(k for k in rule if rule[k] == "passage") != sorted(P_READS):
        raise SystemExit(f"the grade reads {KB_READS} (--read) and {P_READS} (--pread)")
    graded, not_read = {fk: [] for fk in FS}, {fk: [] for fk in FS}
    for nm_ in refs:
        for fk in FS:
            ks_ = [k for k in full if k[0] == fk]
            n_in = sum(nm_ in rows[k] for k in ks_)
            if n_in == len(ks_):
                graded[fk].append(nm_)
            elif n_in == 0:
                not_read[fk].append(nm_)
            else:
                raise SystemExit(f"{nm_} is read by {n_in} of the {fk} f's {len(ks_)} runs: give every pread or none")
        N = refs[nm_]["twin0"].shape[0]
        for k in full:
            if nm_ in rows[k] and rows[k][nm_].shape[0] != N:
                raise SystemExit(f"{k}'s {nm_} read has {rows[k][nm_].shape[0]} rows, the build {N}")

    def R(fk, s_, nm_, sd=None):
        if sd is not None:
            return rows[(fk, s_, sd)][nm_][:, 0].astype(np.float64)
        return np.mean([rows[(fk, s_, k)][nm_][:, 0].astype(np.float64) for k in SEEDS], 0)

    def ci(dd, labels_, idx):
        c = CP.boot_mean(dd, idx)
        return {"diff": c, "verdict": C19.ci_label(c, labels_)}

    def cmp(x, y, nm_, idx, S, key, labels_=ABV):
        """x, y: (fk, set), or y a reference arm's rows. S[nm_][key]: the per-seed differences and signs."""
        Y = R(*y, nm_) if isinstance(y, tuple) else y
        e = ci(R(*x, nm_) - Y, labels_, idx)
        per = [round(float((R(*x, nm_, sd) - (R(*y, nm_, sd) if isinstance(y, tuple) else y)).mean()), 5)
               for sd in SEEDS]
        S.setdefault(nm_, {})[key] = {"per_seed": per, "signs": [int(np.sign(v)) for v in per]}
        return e

    V = {fk: {k: {} for k in ("T1", "T2", "T3", "T4", "T5", "T7", "S", "R0")} for fk in FS}
    V["T6"] = {"S": {}}
    V["not_read"] = not_read
    for fk in FS:
        Vf = V[fk]
        for nm_ in graded[fk]:
            N = refs[nm_]["twin0"].shape[0]
            idx = np.random.default_rng(SEED).integers(0, N, size=(BOOT, N))
            if nm_ == "webqsp_sf":
                Vf["T1"] = {f"{c} - k5": cmp((fk, c), (fk, "k5"), nm_, idx, Vf["S"], f"{c} - k5") for c in CANDS}
                for dd in (1, 2):
                    msk = Ds[nm_] == dd
                    if msk.any():
                        ii = np.random.default_rng(SEED).integers(0, int(msk.sum()), size=(BOOT, int(msk.sum())))
                        Vf["T4"][f"D{dd}"] = {f"{c} - k5": dict(ci((R(fk, c, nm_) - R(fk, "k5", nm_))[msk], ABV, ii),
                                                               rows=int(msk.sum())) for c in CANDS}
            if nm_ == "metaqa":
                Vf["T2"] = {f"{c} - k5": cmp((fk, c), (fk, "k5"), nm_, idx, Vf["S"], f"{c} - k5") for c in CANDS}
            Vf["T3"][nm_] = {f"k5 - {lo}": dict(cmp((fk, "k5"), (fk, lo), nm_, idx, Vf["S"], f"k5 - {lo}", LOV),
                                                transform=t) for lo, t in LOTO.items()}
            if rule[nm_] == "kb":
                walk = refs[nm_]["none/rd"][:, 0]
                Vf["T7"][nm_] = {f"{s_} - none/rd": cmp((fk, s_), walk, nm_, idx, Vf["S"], f"{s_} - none/rd")
                                 for s_ in SETS}
            else:
                e = {f"{c} - k5": cmp((fk, c), (fk, "k5"), nm_, idx, Vf["S"], f"{c} - k5") for c in CANDS}
                s0r, tw = refs[nm_]["s0+rrf"][:, 0], refs[nm_]["twin0"][:, 0]
                for s_ in SETS:
                    X_ = R(fk, s_, nm_)
                    sh = CP.share(X_, s0r, tw, idx)
                    e[f"{s_} - s0+rrf"] = dict(ci(X_ - s0r, ABV, idx), share=sh,
                                               share_verdict="HIGH" if sh[1] >= 0.5 else "LOW" if sh[2] < 0.25
                                               else "MID")
                Vf["T5"][nm_] = e
        p_, js, rr = r0[fk]
        for nm_ in KB_READS + P_RUN[fk]:
            dif = float(np.abs(rows[(fk, "k5", 0)][nm_].astype(np.float64) - rr[nm_].astype(np.float64)).max())
            Vf["R0"][nm_] = {"max_abs_diff": dif, "verdict": "REPRODUCES" if dif < 1e-6 else "DIFFERS"}
        same_w = js.get("state_sha256") == runs[(fk, "k5", 0)].get("state_sha256")
        Vf["R0"]["weights"] = {"state_sha256": [runs[(fk, "k5", 0)].get("state_sha256"), js.get("state_sha256")],
                               "verdict": "REPRODUCES" if same_w else "DIFFERS"}
        Vf["R0"]["run"] = p_
    for nm_ in [x for x in graded["gnn"] if x in graded["mlp"]]:
        N = refs[nm_]["twin0"].shape[0]
        idx = np.random.default_rng(SEED).integers(0, N, size=(BOOT, N))
        V["T6"][nm_] = {f"gnn - mlp {s_}": cmp(("gnn", s_), ("mlp", s_), nm_, idx, V["T6"]["S"], f"gnn - mlp {s_}")
                        for s_ in SETS}
    res = {"look": "chainscore27", "args": dict(vars(a)), "script_sha256": sha(__file__), "run_script": shas.pop(),
           "device": dev, "norm": NORM, "graded": graded, "not_read": not_read,
           "runs": {f"{k[0]}:{k[1]}#s{k[2]}": {"best_epoch": js_["best_epoch"], "best_select_r5": js_["best_select_r5"],
                                              "state_sha256": js_.get("state_sha256"), "timing": js_.get("timing"),
                                              "seconds_per_batch": js_.get("seconds_per_batch"),
                                              "reads": {n: v["mean"] for n, v in js_["reads"].items()}}
                    for k, js_ in sorted(runs.items())},
           "preads": {f"{k[0]}:{k[1]}#s{k[2]}:{n}": js_["reads"][n]["mean"] for (k, n), js_ in sorted(mu.items())},
           "means": {fk: {nm_: {s_: [round(float(np.mean([rows[(fk, s_, sd)][nm_][:, c].astype(np.float64).mean()
                                                           for sd in SEEDS])), 5) for c in range(3)] for s_ in SETS}
                          for nm_ in graded[fk]} for fk in FS},
           "refs": {nm_: {k: [round(float(u), 5) for u in x.mean(0)] for k, x in r.items()
                          if k in ("rrf", "twin0", "none/rd", "s0+rrf", "s0+walk")} for nm_, r in refs.items()},
           "verdicts": V, "decision": decide(V)}
    res["seconds"] = round(time.time() - t0, 1)
    log(f"decision {res['decision']}")
    return res


def decide_f(Vf, reads):
    r0 = [(Vf.get("R0", {}).get(k) or {}).get("verdict") for k in reads + ("weights",)]
    if None in r0:
        return {"carry": None, "why": "R0 incomplete", "report": True}
    if "DIFFERS" in r0:
        return {"carry": None, "why": "R0 DIFFERS: nothing is read until its cause is found", "report": True}
    ok = []
    for c in CANDS:
        w = (Vf.get("T1") or {}).get(f"{c} - k5") or {}
        q = (Vf.get("T2") or {}).get(f"{c} - k5") or {}
        sg = ((Vf.get("S") or {}).get("webqsp_sf") or {}).get(f"{c} - k5", {}).get("signs") or []
        if w.get("verdict") == "ABOVE" and q.get("verdict") != "BELOW" and len(sg) == len(SEEDS) and min(sg) > 0:
            ok.append((w["diff"][0], c))
    if not ok:
        return {"carry": "k5", "set_carries": False, "report": True,
                "why": "no set is ABOVE k5 on webqsp selectf + fit with every seed above 0 and not BELOW on metaqa: "
                       "k5 stays"}
    return {"carry": max(ok)[1], "set_carries": True, "candidates": [c for _d, c in sorted(ok, reverse=True)],
            "report": False, "why": f"ABOVE k5 on webqsp selectf + fit, seeds agreeing, not BELOW on metaqa: "
                                    f"{[c for _d, c in ok]}"}


def decide(V):
    out = {fk: decide_f(V.get(fk) or {}, KB_READS + P_RUN[fk]) for fk in FS}
    out["headline"] = "gnn"
    out["report"] = bool(out["gnn"]["report"])
    return out


# ── selftest ────────────────────────────────────────────────────────────────────────────────────────────────────


def _brute_descriptors(tf, G, Pn):
    """The new transforms' descriptors fact by fact (the vectorised form's check)."""
    n0, nm = tf.n_rel0, tf.info["name"]
    Dt, Dh = np.zeros((tf.n_rel, Pn.shape[1])), np.zeros((tf.n_rel, Pn.shape[1]))
    pres = np.zeros(tf.n_rel, bool)

    def add(ty, hv, tv):
        Dh[ty] += hv
        Dt[ty] += tv
        pres[ty] = True

    for j in range(len(G["H"])):
        h, t = int(G["H"][j]), int(G["T"][j])
        for r in G["slots"][j]:
            if r < 0:
                continue
            if nm == "al16":
                a_ = C19.alias(G["nodes"][[h]], G["nodes"][[t]], np.array([r]), 16)[0]
                add(16 * int(r) + int(a_), Pn[h], Pn[t])
                continue
            add(int(r), Pn[h], Pn[t])
            if tf.reif[r]:
                if nm == "rfk":
                    med = unit((Pn[h] + Pn[t])[None])[0]
                    add(n0 + int(r), Pn[h], med)
                    add(2 * n0 + int(r), med, Pn[t])
                else:
                    add(n0 + int(r), Pn[t], Pn[h])
    return np.c_[unit(Dt), unit(Dh)] / math.sqrt(2), pres


def selftest():
    import inspect
    import tempfile
    # 1. the set table: canonical order, leave-one-out sets, the f's and their k5 arms
    assert SETS["k5"] == TF4 and not set(NEW) & set(C19.TRANSFORMS) and set(CANDS) == set(SETS) - {"k5"}
    for s_, tfs in SETS.items():
        assert list(tfs) == [t for t in ORDER if t in tfs] and len(set(tfs)) == len(tfs), s_
    for lo, t in LOTO.items():
        assert set(SETS[lo]) == set(TF4) - {t}, lo
    assert set(SETS["k8t"]) == set(ORDER) and set(SETS["sw-rfk"]) == set(TF4) - {"rf"} | {"rfk"}
    sg, sm = C21.arm_spec("b-lb", FS["gnn"]), C21.arm_spec("b-lb", FS["mlp"])
    assert sg["gnn"] and sg["att"] and sg["g"] and sg["lb"] and not sm["gnn"] and sm["g"] and sm["lb"]
    assert C25.ARMS[REF9] == (FS["gnn"], NORM) and FS["mlp"] == C24.F
    assert C24.ARMS[REF8] == {"kb": TF4, "p": (), "lam": 0.0}
    assert set(P_RUN["gnn"]) | set(P_PREAD) == set(P_READS) == set(P_RUN["mlp"]) == set(C21.P_READS)
    for t in NEW:
        assert set(BUILD[t]) == {"hops", "ks", "regimes", "rows"} and BUILD[t]["regimes"].startswith("1,")
    # the train statements are part 9's and part 8's, sha-pinned: the draw by groups, the f's spec
    assert "C21.draw21(rng, builds, groups, per_epoch)" in inspect.getsource(C25.train25)
    assert "C21.draw21(rng, builds, groups, per_epoch)" in inspect.getsource(C24.train24)
    # 2. the transforms on a hand-made union graph and row
    rng = np.random.default_rng(7)
    G = {"nodes": np.arange(6, dtype=np.int64) * 10 + 3, "H": np.array([0, 1, 2, 3, 4, 0]),
         "T": np.array([1, 2, 3, 4, 5, 5]), "slots": np.array([[0, -1], [1, 2], [0, 3], [3, -1], [2, -1], [1, 0]])}
    Pn = unit(rng.normal(size=(6, 8)).astype(np.float32))
    present = np.ones(4, bool)
    for nm in C19.TRANSFORMS:          # the old five: chainscore19's own, unchanged
        a_, b_ = _C19_TRANSFORM(nm, 4, G, Pn, present), Transform(nm, 4, G, Pn, present)
        assert a_.info == b_.info and a_.n_rel == b_.n_rel and a_.name == b_.name, nm
        x1, p1 = a_.descriptors(G, Pn)
        x2, p2 = b_.descriptors(G, Pn)
        assert np.array_equal(x1, x2) and np.array_equal(p1, p2), nm
        S = G["slots"]
        assert np.array_equal(a_.relabel(S, G["nodes"][G["H"]], G["nodes"][G["T"]]),
                              b_.relabel(S, G["nodes"][G["H"]], G["nodes"][G["T"]])), nm
        nv = rng.normal(size=(4, 5))
        assert np.array_equal(a_.names(nv), b_.names(nv)), nm
    rf_ = Transform("rf", 4, G, Pn, present)
    T = {nm: Transform(nm, 4, G, Pn, present) for nm in NEW}
    assert T["rfk"].info["reified"] == rf_.info["reified"] and T["rfk"].n_rel == 12 and T["rfk"].name == "rf"
    assert T["inv"].n_rel == 8 and T["inv"].name == "rf" and len(T["inv"].info["inverted"]) == 2
    assert T["al16"].n_rel == 64 and T["al16"].name == "al16"
    for nm, tf in T.items():
        x1, p1 = tf.descriptors(G, Pn)
        x2, p2 = _brute_descriptors(tf, G, Pn)
        assert x1.shape == (tf.n_rel, 16) and np.allclose(x1, x2, atol=1e-6) and np.array_equal(p1, p2), nm
        nv = rng.normal(size=(4, 5))
        nn_ = tf.names(nv)
        assert nn_.shape == (tf.n_rel, 5), nm
        for r in range(tf.n_rel):
            assert np.array_equal(nn_[r], nv[r // 16 if nm == "al16" else r % 4]), (nm, r)
    S = np.array([[0, -1], [3, 1]])
    out = T["al16"].relabel(S, np.array([13, 23]), np.array([33, 43]))
    assert out[0, 1] == -1 and (out[S >= 0] // 16 == S[S >= 0]).all()
    assert out[1, 0] == 48 + C19.alias(np.array([23]), np.array([43]), np.array([3]), 16)[0]
    # a row: (0, 1, r0), (1, 2, r1) twice, (1, 2, r2); r0 and r2 rewritten
    hl, tl = np.array([0, 1, 1]), np.array([1, 2, 2])
    sl = np.array([[0, -1], [1, 2], [1, -1]])
    pn = unit(rng.normal(size=(4, 8)).astype(np.float32))
    rrf = np.array([0.5, 0.25, 0.125, 0.1])
    for tf in (T["rfk"], T["inv"]):
        tf.reif = np.array([True, False, True, False])
    n2, h2, t2, s2, pn2, rrf2 = T["rfk"].reify_row(4, hl, tl, sl, pn, rrf)
    assert n2 == 6 and h2.tolist() == [0, 1, 1, 0, 1, 4, 5] and t2.tolist() == [1, 2, 2, 4, 5, 1, 2]
    assert s2[:, 0].tolist() == [0, 1, 2, 4, 6, 8, 10] and s2.shape == (7, 1)
    assert np.allclose(pn2[4], unit((pn[0] + pn[1])[None])[0]) and np.allclose(pn2[5], unit((pn[1] + pn[2])[None])[0])
    assert np.array_equal(pn2[:4], pn) and rrf2.tolist() == [0.5, 0.25, 0.125, 0.1, 0.5, 0.25]
    n2, h2, t2, s2, pn2, rrf2 = T["inv"].reify_row(4, hl, tl, sl, pn, rrf)
    assert n2 == 4 and h2.tolist() == [0, 1, 1, 1, 2] and t2.tolist() == [1, 2, 2, 0, 1]
    assert s2[:, 0].tolist() == [0, 1, 2, 4, 6] and pn2 is pn and rrf2 is rrf
    print("selftest: the old transforms are chainscore19's; rfk, inv and al16 rewrite rows and descriptors as defined")
    # 3. toy builds: the patch leaves the old builds unchanged; the new builds and their companions
    P = C19.RT.proj_matrix()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        roots = {}
        for nm_, sd, nrow in (("train", 11, 96), ("select", 12, 48)):
            root = td / nm_
            reld = td / f"{nm_}_rel"
            reld.mkdir(parents=True)
            _truth, names = CP.toy_world(root, np.random.default_rng(sd), P, rows=nrow)
            np.save(reld / "toy_rel_embeddings.npy", names)
            roots[nm_] = (root, reld)

        def args(**kw):
            base = dict(ds="toy", carve="read", hops=2, ks="2", limit=None, max_chains=CP.MAX_CHAINS, regimes="1,0",
                        transform="id", rows="all")
            base.update(kw)
            return SimpleNamespace(**base)

        root, reld = roots["train"]
        for tfn in ("id", "rf", "al4"):
            C19.build19(args(transform=tfn, out=str(td / f"o-{tfn}.npz")), root=root, rel_dir=reld)
            patch()
            try:
                C19.build19(args(transform=tfn, out=str(td / f"p-{tfn}.npz")), root=root, rel_dir=reld)
            finally:
                patch(False)
            d1, d2 = CS.load(td / f"o-{tfn}.npz"), CS.load(td / f"p-{tfn}.npz")
            assert d1["meta"]["transform"] == d2["meta"]["transform"] and sorted(d1) == sorted(d2), tfn
            for k in d1:
                if k != "meta":
                    assert np.array_equal(d1[k], d2[k], equal_nan=d1[k].dtype.kind == "f"), (tfn, k)
        assert C19.Transform is _C19_TRANSFORM
        did, drf = CS.load(td / "o-id.npz"), CS.load(td / "o-rf.npz")
        deg = C19.NODEF.index("deg")
        built = {}
        for tfn in NEW:
            meta = build27(args(transform=tfn, out=str(td / f"train-{tfn}.npz")), root=root, rel_dir=reld)
            assert C19.Transform is _C19_TRANSFORM
            d = CS.load(td / f"train-{tfn}.npz")
            built[tfn] = d
            assert d["meta"]["transform"]["name"] == tfn and "chainscore27_sha256" in d["meta"]["args"]
            assert np.array_equal(d["D"], did["D"]), tfn                 # every fact keeps its direct edge
            nid, nb = did["n"].astype(np.int64), d["n"].astype(np.int64)
            for i in range(nb.size):
                assert (d["gold"][d["gold_off"][i]:d["gold_off"][i + 1]] < nid[i]).all(), (tfn, i)
            if tfn == "rfk":
                assert meta["transform"]["n_rel"] == 12 and meta["transform"]["reified"] == \
                    drf["meta"]["transform"]["reified"]
                assert meta["mediators"] == drf["meta"]["mediators"] > 0 and np.array_equal(nb, drf["n"])
                for i in range(nb.size):
                    a_ = int(d["xn_off"][i])
                    Xm = d["XN"][a_ + nid[i]:a_ + nb[i]].astype(np.float32)
                    assert np.allclose(Xm[:, deg], math.log(3), atol=2e-3), i
            elif tfn == "inv":
                assert meta["transform"]["n_rel"] == 8 and meta["mediators"] == 0 and np.array_equal(nb, nid)
                assert np.array_equal(d["XN"][:, deg], did["XN"][:, deg])      # the same neighbours
            else:
                assert meta["transform"]["n_rel"] == 64 and meta["levels"][-1]["Kz"] == 128
                for k in ("XN", "seed0", "n", "gold", "gold_off", "D", "rank", "gt", "rrf_s", "xn_off"):
                    assert np.array_equal(d[k], did[k]), (tfn, k)
            e = edges_cmd(SimpleNamespace(base=str(td / f"train-{tfn}.npz"), out=str(td / f"train-{tfn}.e.npz"),
                                          no_node_check=False), root=root)
            assert C19.Transform is _C19_TRANSFORM and e["check"]["nodes"] and e["check"]["rows"] == nb.size
            print(f"toy {tfn}: n_rel {meta['transform']['n_rel']}, {meta['mediators']} mediators, levels "
                  f"{[(x['name'], x['vocab']) for x in meta['levels']]}; companion checked "
                  f"({e['check']['nodes_checked']} end nodes)")
        try:
            edges_cmd(SimpleNamespace(base=str(td / "o-rf.npz"), out=str(td / "x.npz"), no_node_check=False),
                      root=root)
            raise AssertionError("an old build's companion was made here")
        except SystemExit:
            pass
        # 4. both f's train on the new builds (one epoch, CPU) and read: the pipeline's pq map, the groups, lb
        C20.edges_cmd(SimpleNamespace(base=str(td / "o-id.npz"), out=str(td / "o-id.e.npz"), no_node_check=False),
                      root=root)
        sroot, sreld = roots["select"]
        C19.build19(args(regimes="1", out=str(td / "select.npz")), root=sroot, rel_dir=sreld)
        C20.edges_cmd(SimpleNamespace(base=str(td / "select.npz"), out=str(td / "select.e.npz"), no_node_check=False),
                      root=sroot)
        srcs = [("fit", td / "o-id.npz", td / "o-id.e.npz")] + [
            (t, td / f"train-{t}.npz", td / f"train-{t}.e.npz") for t in NEW]
        builds, comps, LB = [], [], []
        for n, p, pe in srcs:
            d, lb, info, _st = C25.get(str(p), "train", NORM, None)
            assert info["norm"] == NORM, info
            builds.append(d)
            comps.append(C25.load_comp(str(pe), d, sha(p), sg["att"]))
            LB.append(lb)
        Sd, Slb, _i, _s = C25.get(str(td / "select.npz"), "plain", NORM, None)
        Sc = C25.load_comp(str(td / "select.e.npz"), Sd, sha(td / "select.npz"), sg["att"])
        groups = [list(range(len(builds)))]
        mg, ig = C25.train25("toy", FS["gnn"], builds, comps, LB, groups, Sd, Sc, Slb, None, 1, 64, 32, SEED)
        mm, im = C24.train24("toy", builds, LB, groups, [0] * len(builds), Sd, Slb, None, 1, 64, 32, SEED, 0.0)
        for info in (ig, im):
            bb = info["history"][0]["by_build"]
            assert sum(bb) == 64 and min(bb) > 0, bb
        xg, _ml = CD.read_dev(mg, Sd, Sc, Slb, ig["stats"], "kb", None, batch=32)
        xm, _ml = CD.read_dev(mm, Sd, None, Slb, im["stats"], "kb", None, batch=32)
        assert xg.shape == xm.shape == (Sd["n"].size, 3) and np.isfinite(xg).all() and np.isfinite(xm).all()
        print(f"toy training: GNN f select R@5 {ig['best_select_r5']}, MLP f {im['best_select_r5']} (one epoch of 64)")
    # 5. decide

    def v(verdict, d=0.0):
        return {"verdict": verdict, "diff": [d, d - 0.01, d + 0.01]}

    def table(w, m, signs=None):
        signs = signs or {}
        out = {}
        for fk in FS:
            out[fk] = {"R0": {k: {"verdict": "REPRODUCES"} for k in KB_READS + P_RUN[fk] + ("weights",)},
                       "T1": {f"{c} - k5": v(*w[c]) for c in CANDS}, "T2": {f"{c} - k5": v(*m[c]) for c in CANDS},
                       "S": {"webqsp_sf": {f"{c} - k5": {"signs": signs.get(c, [1, 1, 1])} for c in CANDS}}}
        return out

    at = {c: ("AT",) for c in CANDS}
    D = decide(table(at, at))
    assert D["gnn"]["carry"] == "k5" and D["report"] and not D["mlp"]["set_carries"]
    W = {**at, "p-rfk": ("ABOVE", 0.02), "k8t": ("ABOVE", 0.03)}
    D = decide(table(W, at))
    assert D["gnn"]["carry"] == "k8t" and not D["report"] and D["gnn"]["candidates"] == ["k8t", "p-rfk"]
    D = decide(table(W, {**at, "k8t": ("BELOW", -0.02)}))
    assert D["gnn"]["carry"] == "p-rfk"
    D = decide(table(W, at, {"k8t": [1, -1, 1]}))
    assert D["gnn"]["carry"] == "p-rfk"
    D = decide(table({**at, "lo-mg3": ("ABOVE", 0.01)}, at))
    assert D["mlp"]["carry"] == "lo-mg3"
    V = table(W, at)
    V["gnn"]["R0"]["weights"] = {"verdict": "DIFFERS"}
    D = decide(V)
    assert D["gnn"]["carry"] is None and D["mlp"]["carry"] == "k8t" and D["report"]
    del V["mlp"]["R0"]["musique"]
    assert decide(V)["mlp"]["carry"] is None
    # 6. the carried map must be pq (part 9's rule)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cs22.json"
        for carry, ok in (("pq", True), ("pz", False), ("none", False)):
            p.write_text(json.dumps({"look": "chainscore22", "decision": {"carry": carry}}), encoding="utf-8")
            try:
                C25.carried_map(p)
                assert ok, carry
            except SystemExit:
                assert not ok, carry
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--transform", required=True, choices=NEW)
    b.add_argument("--ds", default="metaqa")
    b.add_argument("--carve", default="fit")
    b.add_argument("--max-chains", type=int, default=CP.MAX_CHAINS)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--out", required=True)
    e = sub.add_parser("edges")
    e.add_argument("--base", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--no-node-check", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--f", required=True, choices=list(FS))
    t.add_argument("--set", required=True, choices=list(SETS))
    t.add_argument("--seed", type=int, required=True, choices=SEEDS)
    t.add_argument("--map-from", required=True)
    t.add_argument("--cache", default=None)
    t.add_argument("--fit", default=None)
    t.add_argument("--aug", action="append", default=[])
    t.add_argument("--select", default=None)
    t.add_argument("--read", action="append", default=[])
    t.add_argument("--pread", action="append", default=[])
    t.add_argument("--edges", action="append", default=[])
    t.add_argument("--epochs", type=int, default=C19.EPOCHS)
    t.add_argument("--per-epoch", type=int, default=C19.PER_EPOCH)
    t.add_argument("--batch", type=int, default=BATCH)
    t.add_argument("--read-batch", type=int, default=BATCH)
    t.add_argument("--device", default="cpu")
    t.add_argument("--out", required=True)
    t.add_argument("--rows-out", required=True)
    t.add_argument("--state-out")
    r = sub.add_parser("pread")
    r.add_argument("--run", required=True)
    r.add_argument("--pread", required=True)
    r.add_argument("--edges", required=True)
    r.add_argument("--cache", required=True)
    r.add_argument("--read-batch", type=int, default=BATCH)
    r.add_argument("--device", default="cpu")
    r.add_argument("--out", required=True)
    r.add_argument("--rows-out", required=True)
    g = sub.add_parser("grade")
    g.add_argument("--run", action="append", required=True)
    g.add_argument("--ref9", action="append", required=True)
    g.add_argument("--ref8", action="append", required=True)
    g.add_argument("--ref9-mu", action="append", default=[])
    g.add_argument("--mu-run", action="append", default=[])
    g.add_argument("--read", action="append", required=True)
    g.add_argument("--pread", action="append", required=True)
    g.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "build":
        build_cmd(a)
        return 0
    if a.cmd == "edges":
        edges_cmd(a)
        return 0
    if a.cmd in ("train", "pread", "grade"):
        res = {"train": train_cmd, "pread": pread_cmd, "grade": grade_cmd}[a.cmd](a)
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                       encoding="utf-8")
        os.replace(tmp, p)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
