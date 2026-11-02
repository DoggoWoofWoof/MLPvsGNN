"""GNX's stage (docs/GNX_NEXT_COLUMNS_IN_GNN.md): the GNN of record with LTM's seven NEXT columns (docs/LTM_NEXT_HOP_MLP.md,
ADOPT 7874e4d: LTW's next-hop score of each pool row against the first stage's top five rows, cross-fitted) as one more
input of its encoder. Each row's [raw, z against its question's retrieved rows, 1] (15 numbers, as zrckpn's encoder takes
the block) goes through gnx_w (hidden x 15, zeros at the start) and is added to the output of zg6r's l1, so a fit starts as
the GNN of record bit for bit and zg6r's l1.weight stays the one it copies from zspk. On PKO's roots, on the host's GPU:
the screen at seed 0 on both WQ splits against the base's seed-0 fit under LTG's in-domain rule (docs/LTG_NEXT_HOP_GNN.md
section 4), DEP's sixteen-read rule beside; the full run (seeds 1 and 2) when the screen asks for it.

The base is the GNN of record after ISO (zdep.iso_call, zdep.base_fit). zml1s.py, zg6r.py, zg6.py, ziso.py, zdep.py,
zltg.py, zltm.py and ltmcols.py are imported and not edited. For the length of a GNX fit: ZG6R's __init__ adds gnx_w after
zg6r's own parameters (no draw, so every other start is zg6r's) and a forward hook on l1; ZG6R's forward forms the batch's
NEXT input for that hook; the arm's carve (zg6r's G6rLinkCarve, patched in place, so zg6r's arm table still holds it)
loads the fit's split's columns for its carve through zltm.load_next (LTM's files, read only, kept on the CPU) and its
batch adds them for the batch's rows; zml1s's train and read run with the split set, and the stamp refuses a train that
did not load every training carve's columns or a read that did not load every s1eval carve's. zml1s runs with GNX's
names, stamp key, stage and declaration, and on REPLACE with ISO's own-start patches. Every patched name is put back
afterwards, also on a failure.

    python outputs/mp_unified/zgnx.py check-cols                   (LTM's columns of every carve both splits load)
    python outputs/mp_unified/zgnx.py fit --name scr-gnx-nx-s0     (train when needed, then read; kept when read before)
    python outputs/mp_unified/zgnx.py compare-all                  (each GNX fit read against the base's fit of its seed)
    python outputs/mp_unified/zgnx.py calls                        (outputs/gnx/calls.json and .md)
    python outputs/mp_unified/zgnx.py --selftest
"""
import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import zml1 as RULE  # noqa: E402  (stdlib only)
import zml1s as M1  # noqa: E402
import ziso as ISO  # noqa: E402
import zdep as DEP  # noqa: E402  (the base after ISO, the slices and DEP's rule; DEP's fits are not touched)
import zltg as LTG  # noqa: E402  (LTG's in-domain rule: its keys and per-read calls, pure; LTG's fits are not touched)
import zltm as LTM  # noqa: E402  (load_next: LTM's columns of a carve, checked against its cache parts)
import ltmcols as LTMC  # noqa: E402  (the columns' layout and carves)
import ml1start as MS  # noqa: E402  (on ISO's KEEP, ML1's zspk start read under ML1's names: LTG amendment A1)
import readswa as RS  # noqa: E402  (a fit's read keeps p@swa only: LTG amendment A2)

DECLARED = "docs/GNX_NEXT_COLUMNS_IN_GNN.md"
STAGE = "GNX"
KEY = "gnx"                                     # the stamp's key in a GNX fit's screen.json and read.json and GNX's compares
OUT = "outputs/gnx"
ARM = DEP.ARM                                   # zrm's arm: the model is zg6r's with the NEXT input
TAG = "nx"
NEXT = LTM.NEXT                                 # the batch key of the columns
WIDTH = LTMC.WIDTH                              # seven columns
IN = 2 * WIDTH + 1                              # raw, z, 1: what zrckpn's encoder takes of the block
HIDDEN = 128                                    # zg6r's hidden size in every fit
CODE_SKIP = DEP.CODE_SKIP
R5_TOL = DEP.R5_TOL
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, ADOPT, INCOMPLETE = (
    DEP.ADOPT_FOR_FULL_RUN, DEP.NO_GAIN, DEP.NOT_ADOPTED, DEP.ADOPT, DEP.INCOMPLETE)
assert (WIDTH, IN, NEXT, LTMC.EVAL_CARVE) == (7, 15, "NEXT", "s1eval")


def fit_name(seed, split):
    return f"scr-gnx-{TAG}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    return f"scr-gnx-{TAG}-vs-base-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 6 and not set(FITS) & (set(M1.ALL_FITS) | set(ISO.FITS) | set(DEP.FITS) | set(LTG.FITS))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zgnx: {msg}", flush=True)


def pairs(call):
    return {pair_name(sd, sp): (fit_name(sd, sp), DEP.base_fit(sd, sp, call)) for sd in RULE.SEEDS for sp in M1.SPLITS}


def eval_carves():
    return [f"{ds}/{LTMC.EVAL_CARVE}" for ds in M1.WQ.EVAL_ORDER]


def train_carves(split):
    return [f"{ds}/{c}" for ds, c in LTMC.train_carves(split)]


# ── the columns, beside the carve ────────────────────────────────────────────

CUR = {"split": None, "fit_sha256": None, "out": None}   # the split whose columns the carves built now load
LOADED = []                                     # per carve loaded: dataset, carve, its column record's sha256, the scorer


@contextlib.contextmanager
def on_split(split, fit_sha256=None, out=None):
    """The split whose columns the carves built in the block load (and the scorer they must share, when given); put back
    after, also on a failure."""
    if split not in M1.SPLITS:
        raise SystemExit(f"zgnx: {split} is not one of WQ's splits")
    saved = dict(CUR)
    CUR.update(split=split, fit_sha256=fit_sha256, out=out)
    try:
        yield
    finally:
        CUR.clear()
        CUR.update(saved)


def next_block(split):
    """A fit's NEXT block, inside on_split: the split, the columns, the scorer and each carve's column record it read."""
    return {"declared_in": LTM.DECLARED, "split": split, "columns": list(LTMC.COLS), "fit_sha256": CUR["fit_sha256"],
            "carves": {f"{d['dataset']}/{d['carve']}": d["record_sha256"] for d in LOADED},
            "training_carves": train_carves(split)}


def next_rows(c, qs, nq):
    """The batch's NEXT rows (float32, NaN and infinities 0, as zrckpn's carve hands them): each question's rows in the
    carve's own order, questions in qs's order (lean_gpu's batch order), taken from the CPU copy and put on nq's device."""
    import torch
    qs = np.asarray(qs, np.int64)
    cnt = c.n_np[qs]
    idx = np.repeat(c.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
    if idx.size != nq.numel():
        raise RuntimeError(f"zgnx: {c.ds}/{c.carve}'s batch holds {nq.numel()} rows, its questions {idx.size}")
    x = c.gnx_X[torch.from_numpy(idx)].to(device=nq.device, dtype=torch.float32)
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


@contextlib.contextmanager
def on_carve(G, C=None):
    """For the block: the arm's carve class (zg6r's G6rLinkCarve unless C is given) loads its carve's columns under CUR's
    split after its own __init__, and its batch adds them under NEXT; patched in place and restored after."""
    import torch
    if C is None:
        C = G.G6rLinkCarve
        if G.ARMS[ARM][1] is not C:
            raise SystemExit(f"zgnx: zg6r's arm carve is {G.ARMS[ARM][1]}, not its G6rLinkCarve")
    orig_init, orig_batch = C.__init__, C.batch

    def init(self, ds, carve, basis, device, cache_root=G.LC.OUT, verify=True, score2=False):
        orig_init(self, ds, carve, basis, device, cache_root, verify, score2)
        sp = CUR["split"]
        if sp is None:
            raise SystemExit(f"zgnx: {ds}/{carve} built for GNX outside a GNX split")
        X, rec, sha = LTM.load_next(ds, carve, sp, self.part_shas, CUR["out"], verify)
        if X.shape[0] != int(self.off_np[-1]):
            raise SystemExit(f"zgnx: {ds}/{carve}'s columns hold {X.shape[0]} rows, its carve {int(self.off_np[-1])}")
        if CUR["fit_sha256"] is None:
            CUR["fit_sha256"] = rec["fit_sha256"]
        elif rec["fit_sha256"] != CUR["fit_sha256"]:
            raise SystemExit(f"zgnx: {ds}/{carve}'s columns come from the scorer {rec['fit_sha256'][:12]}, this fit's "
                             f"from {CUR['fit_sha256'][:12]}")
        self.gnx_X = torch.from_numpy(X)          # float16 on the CPU: each batch's rows go to the device
        self.gnx_record = {"dataset": ds, "carve": carve, "record_sha256": sha, "fit_sha256": rec["fit_sha256"],
                           "rows": int(X.shape[0]), "training_carve": rec["training_carve"]}
        LOADED.append(self.gnx_record)
        log(f"{ds}/{carve}: NEXT columns of {sp} loaded ({X.shape[0]} rows, {X.nbytes / 1e9:.2f} GB on the CPU)")

    def batch(self, qs, blocks):
        feats, nq, base_z, gold = orig_batch(self, qs, blocks)
        if NEXT in feats:
            raise RuntimeError(f"zgnx: {self.ds}/{self.carve}'s batch already holds a {NEXT} block")
        feats[NEXT] = next_rows(self, qs, nq)
        return feats, nq, base_z, gold

    with M1.WQ.patch(C, __init__=init, batch=batch):
        yield


# ── the term ─────────────────────────────────────────────────────────────────


def next_input(G, feats, nq, B):
    """(N, 15): the block as zg6's encode forms any block, [raw m, z m, m] with m = 1 (lean_gpu's keep is all ones):
    raw the seven columns, z their z-score against the question's retrieved rows (lean_screen3.seg_zscore_ref)."""
    S3 = G.G6.S3
    raw = feats[NEXT]
    z = S3.seg_zscore_ref(raw, nq, B, S3.retrieved(feats))
    return G.torch.cat([raw, z, G.torch.ones_like(raw[:, :1])], 1)


def l1_hook(G, m, out):
    """l1's output plus gnx_w applied to the batch's NEXT input (set by ZG6R's patched forward)."""
    x = m.gnx_in
    if x is None:
        raise RuntimeError("zgnx: zg6r's l1 ran outside ZG6R's forward (no NEXT input set)")
    return out + G.Fn.linear(x.to(out.dtype), m.gnx_w.to(out.dtype))


@contextlib.contextmanager
def on_model(G):
    """For the block: a ZG6R built gets gnx_w (zeros) and l1's hook after zg6r's own __init__; ZG6R's forward sets the
    batch's NEXT input for the hook and clears it after. Restored after, also on a failure."""
    torch, nn = G.torch, G.nn
    orig_init, orig_fwd = G.ZG6R.__init__, G.ZG6R.forward

    def init(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        orig_init(self, blocks, widths, hidden, dropout, seed, arm, ctx)
        self.gnx_w = nn.Parameter(torch.zeros(int(self.l1.out_features), IN))
        self.gnx_in = None
        self.l1.register_forward_hook(lambda _mod, _inp, out, m=self: l1_hook(G, m, out))

    def forward(self, feats, keep, nq, B, base_z):
        if NEXT not in feats:
            raise RuntimeError(f"zgnx: no {NEXT} block in the batch (GNX's carve adds it)")
        self.gnx_in = next_input(G, feats, nq, B)
        try:
            return orig_fwd(self, feats, keep, nq, B, base_z)
        finally:
            self.gnx_in = None

    with M1.WQ.patch(G.ZG6R, __init__=init, forward=forward):
        yield


def term_record(G):
    return {"term": "LTM's NEXT columns into zg6r's encoder ([raw, z, 1] through gnx_w, zero-started, added to l1's output)",
            "columns": list(LTMC.COLS), "inputs": IN, "term_parameters": HIDDEN * IN,
            "parameters_h128": int(G.N_PARAMS + HIDDEN * IN), "layers": int(G.LAYERS)}


def term_record_keys():
    return ("term_parameters", "parameters_h128", "inputs", "columns", "layers")


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def with_fits():
    """GNX's names in zml1s's fit table (in place, so on_ml1's FITS sees them); removed afterwards."""
    M1.ALL_FITS.update(FITS)
    try:
        yield
    finally:
        for k in FITS:
            M1.ALL_FITS.pop(k, None)


def gnx_fit_name(arm, seed, split):
    """zml1s.pko_fit_name with the GNX fit being run answering for its own (arm, seed, split)."""
    a = M1._ACTIVE[0]
    if a in FITS and FITS[a] == (arm, seed, split):
        return a
    return M1.fit_name(arm, seed, split)


def stamp_with_next(orig, call, rec):
    """zml1s.stamp with the term's record, the base, ISO's verdict and the NEXT block; a train that did not load every
    training carve's columns, or a read every s1eval carve's, stops unstamped."""
    def stamp(name, env, what, extra=None):
        if name not in FITS:
            return orig(name, env, what, extra)
        _a, sd, sp = FITS[name]
        nxt = next_block(sp)
        need = train_carves(sp) if what == "screen" else eval_carves()
        miss = sorted(set(need) - set(nxt["carves"]))
        if CUR["split"] != sp or miss or not nxt["fit_sha256"]:
            raise SystemExit(f"zgnx: {name}'s {what} loaded no columns of {miss or need} under {sp}; unstamped")
        add = {**rec, "base": DEP.base_fit(sd, sp, call), "iso_verdict": call, "next": nxt}
        return orig(name, env, what, {**(extra or {}), **add})
    return stamp


def train_with(orig):
    """zml1s.train for a GNX fit under its split, the carves' loads counted from empty."""
    def train(name, env):
        if name not in FITS:
            return orig(name, env)
        LOADED.clear()
        try:
            with on_split(FITS[name][2]):
                return orig(name, env)
        finally:
            LOADED.clear()
    return train


def read_with(orig):
    """zml1s.read for a GNX fit under its split, every carve's columns from the scorer its train read."""
    def read(name, env):
        if name not in FITS:
            return orig(name, env)
        sp = FITS[name][2]
        nk = M1.fit_stamp(name)[1].get("next") or {}
        if (nk.get("split"), nk.get("columns")) != (sp, list(LTMC.COLS)) or not nk.get("fit_sha256"):
            raise SystemExit(f"zgnx: {name} was not trained on LTM's columns of {sp} ({nk})")
        LOADED.clear()
        try:
            with on_split(sp, nk["fit_sha256"]):
                return orig(name, env)
        finally:
            LOADED.clear()
    return read


def code_extra():
    """The stage's own code beside zml1s's, by fixed keys (a module run as a script is named __main__)."""
    return {"zgnx_sha256": M1.WQ.sha_src(__file__),
            **{f"{k}_sha256": M1.WQ.sha_src(m.__file__) for k, m in
               (("zdep", DEP), ("zltg", LTG), ("zltm", LTM), ("ltmcols", LTMC))}}


@contextlib.contextmanager
def on_names():
    """zml1s with GNX's names, stamp key, stage and declaration; restored after. Enough for the compares and the calls."""
    extra, code = {**code_extra(), **MS.code(), **RS.code()}, M1.code_shas
    with with_fits(), M1.WQ.patch(M1, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, pko_fit_name=gnx_fit_name,
                                  code_shas=lambda arm=None: {**code(arm), **extra}):
        yield


@contextlib.contextmanager
def on_gnx(G, call):
    """A GNX fit's context: GNX's names and stamp, zml1s's train and read under the fit's split, the model and carve
    patches, the start (on REPLACE ISO's own; on KEEP ML1's zspk start read under ML1's names) and a read of p@swa alone;
    restored after."""
    own = call == ISO.REPLACE
    with on_names(), M1.WQ.patch(M1, stamp=stamp_with_next(M1.stamp, call, term_record(G)),
                                 load_encoder=ISO.own_load if own else MS.ml1_load,
                                 train=train_with(M1.train), read=read_with(M1.read)), \
            (M1.WQ.patch(G, encoder_used=ISO.own_used) if own else contextlib.nullcontext()), on_model(G), \
            on_carve(G), RS.on_read():
        yield


def import_g():
    G, _U2N = M1.import_arm(ARM)
    if G.ARMS.get(ARM) != (G.ZG6R, G.G6rLinkCarve) or G.LAYERS != DEP.BASE_LAYERS:
        raise SystemExit(f"zgnx: zg6r's arm is {G.ARMS.get(ARM)} with {G.LAYERS} layers, not G6r's with {DEP.BASE_LAYERS}")
    if not all(hasattr(G.G6.S3, f) for f in ("seg_zscore_ref", "retrieved")):
        raise SystemExit("zgnx: zg6's lean_screen3 lacks seg_zscore_ref or retrieved")
    return G


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zgnx: {name} is not one of GNX's fits ({', '.join(FITS)})")
    _a, sd, sp = FITS[name]
    miss = [k for k in train_carves(sp) + eval_carves()
            if not (LTMC.cols_dir(LTMC.OUT, sp, *k.split("/")) / "record.json").is_file()]
    if miss:                                     # before any import or train: the fit would stop at its first carve
        raise SystemExit(f"zgnx: {name}: no LTM columns of {miss} under {sp}")
    call = DEP.iso_call()
    G = import_g()
    log(f"{name}: NEXT into {DEP.base_fit(sd, sp, call)}'s model and start (ISO {call}); {term_record(G)}")
    with on_gnx(G, call):
        return M1.fit(name)


def check_cols(cache_root=None, out=None):
    """Every carve a GNX fit of either split loads (its training carves, the six s1eval carves): LTM's column record
    and files against the carve's cache parts (zltm.load_next, sha256 checked), one scorer per split; outputs/gnx/cols.json."""
    import lean_cache as LC
    root = Path(cache_root or ROOT / M1.ROOTS["cache"])
    rec, bad = {"declared_in": DECLARED, "cache_root": M1.WQ.rel(root), "splits": {}}, []
    for sp in M1.SPLITS:
        got, scorers = {}, set()
        for key in train_carves(sp) + eval_carves():
            ds, carve = key.split("/")
            try:
                parts, _r = LC.part_dirs(ds, carve, root)
                X, r, sha = LTM.load_next(ds, carve, sp, {p.name: LC.sha_file(p / "record.json") for p in parts}, out)
            except SystemExit as e:
                bad.append(f"{sp} {key}: {e}")
                continue
            got[key] = {"rows": int(X.shape[0]), "record_sha256": sha, "fit_sha256": r["fit_sha256"],
                        "training_carve": r["training_carve"], "nan": int(np.isnan(X).sum())}
            scorers.add(r["fit_sha256"])
        if len(scorers) > 1:
            bad.append(f"{sp}: its carves' columns come from {len(scorers)} scorers")
        rec["splits"][sp] = got
    rec["problems"] = bad
    rec["pass"] = not bad
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    o = ROOT / OUT
    o.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(o / "cols.json", rec)
    log(f"check-cols: {'PASS' if not bad else 'FAIL'}; {sum(len(v) for v in rec['splits'].values())} carves; {bad}")
    return 0 if not bad else 1


# ── the compares, the slices and the calls ───────────────────────────────────


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: GNX's, ISO's or ML1's."""
    if name in FITS:
        with on_names():
            return M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]
    return DEP.fit_block(name, what, base)


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != M1.ROWS_CAP[ARM]:
        raise SystemExit(f"zgnx: {name}'s train and read are not on one device with zg6r's rows cap")
    return ks["device"]


def is_read(name, base=ROOT):
    return (M1.screen_dir(base) / "fits" / name / "read.json").is_file()


def compare_all(base=ROOT):
    """Each GNX fit that is read against the base's fit of its seed and split, through zwq's run_compare; a compare filed
    before is kept."""
    call = DEP.iso_call(base)
    done, ran = [], []
    for cname, (new, bse) in pairs(call).items():
        if not is_read(new, base):
            continue
        js = M1.screen_dir(base) / f"{cname}.json"
        if js.is_file() and (M1.WQ.read_json(js).get(KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, bse)}
        if len(dev) != 1:
            raise SystemExit(f"zgnx: {new} and {bse} ran on different devices: {sorted(dev)}")
        with ISO.with_fits(), on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zgnx: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "iso_verdict": call,
                    "device": json.loads(dev.pop()), "zgnx_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def r5_gnx(call, base=ROOT):
    """({(split, dataset): {seed: R@5}} of GNX's fits, the same of their bases) from GNX's compares."""
    new_r5, base_r5 = {}, {}
    for cname, (new, bse) in pairs(call).items():
        p = M1.screen_dir(base) / f"{cname}.json"
        if not p.is_file():
            continue
        rec = M1.WQ.read_json(p)
        k = rec.get(KEY) or {}
        if (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, new, bse):
            continue
        if rec.get("candidate") != DEP.CAND or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
            continue
        _a, sd, sp = FITS[new]
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            new_r5.setdefault((sp, r["dataset"]), {})[sd] = r["new"][0]
            base_r5.setdefault((sp, r["dataset"]), {})[sd] = r["base"][0]
    return new_r5, base_r5


def slices(call, new_r5, base3, base=ROOT):
    """({(split, slice): {seed: R@5}} of GNX's read fits, the same of the base's three seeds, problems), by zdep.slice_r5.
    Each file's all-question R@5 must be its compare's musique R@5."""
    new_s, base_s, bad = {}, {}, []
    for sp in M1.SPLITS:
        for sd in RULE.SEEDS:
            for f, own, ref in ((fit_name(sd, sp), new_s, new_r5), (DEP.base_fit(sd, sp, call), base_s, base3)):
                if f in FITS and not is_read(f, base):
                    continue
                s = DEP.slice_r5(f, base)
                if s is None:
                    bad.append(f"{f}: no {DEP.SLICE_DS} read file")
                    continue
                want = ref.get((sp, DEP.SLICE_DS), {}).get(sd)
                if want is not None and abs(s["all"] - want) > R5_TOL:
                    bad.append(f"{f}: its read file gives {DEP.SLICE_DS} R@5 {s['all']:.5f}, its compare {want}")
                for sl in DEP.SLICES:
                    if s[sl] is not None:
                        own.setdefault((sp, sl), {})[sd] = s[sl]
    return new_s, base_s, bad


def fit_problems(call, base=ROOT):
    """Why a GNX fit that is read is not the declared one: its term, base and ISO verdict; its NEXT block (the split, the
    columns, one scorer for train and read, every training carve in its train and every s1eval carve in its read); its
    start; its device; its trainer's code against the base's fit of its seed and split."""
    out = []
    want = {"term_parameters": HIDDEN * IN, "inputs": IN, "columns": list(LTMC.COLS), "layers": DEP.BASE_LAYERS}
    for name, (_a, sd, sp) in FITS.items():
        if not is_read(name, base):
            continue
        bse = DEP.base_fit(sd, sp, call)
        try:
            ks, kr, kb = fit_block(name, "screen", base), fit_block(name, "read", base), fit_block(bse, "screen", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        for what, k, need in (("train", ks, train_carves(sp)), ("read", kr, eval_carves())):
            got = {x: k.get(x) for x in want}
            if got != want or (k.get("base"), k.get("iso_verdict")) != (bse, call):
                out.append(f"{name}: its {what} names the term {got} on {k.get('base')} (ISO {k.get('iso_verdict')}), "
                           f"not {want} on {bse}")
            nk = k.get("next") or {}
            miss = sorted(set(need) - set(nk.get("carves") or {}))
            if (nk.get("split"), nk.get("columns")) != (sp, list(LTMC.COLS)) or miss:
                out.append(f"{name}: its {what}'s NEXT block is {nk.get('split')} {nk.get('columns')}, missing {miss}")
        if (ks.get("next") or {}).get("fit_sha256") != (kr.get("next") or {}).get("fit_sha256"):
            out.append(f"{name}: its train and read read columns of different scorers")
        enc, benc = ks.get("encoder") or {}, kb.get("encoder") or {}
        if call == ISO.REPLACE:
            ok = enc.get("start") == "own" and enc.get("copies") == 0
        else:
            ok = benc.get("fit") is not None and int(enc.get("copies") or 0) >= 1 and all(
                enc.get(x) == benc.get(x) for x in ("fit", "seed", "sha256"))
        if not ok:
            out.append(f"{name}: its encoder's start ({enc.get('start') or enc.get('fit')}, {enc.get('copies')} copies) "
                       f"is not {bse}'s (ISO {call})")
        if ks.get("device") != kb.get("device"):
            out.append(f"{name}: device {ks.get('device')}, {bse}'s {kb.get('device')}")
        cb, cn = kb.get("code") or {}, ks.get("code") or {}
        diff = sorted(k for k in cb if k not in CODE_SKIP and cn.get(k) != cb[k])
        if diff:
            out.append(f"{name}: code differs from {bse}'s in {diff}")
    return out


def calls(base=ROOT):
    call = DEP.iso_call(base)
    new_r5, b0 = r5_gnx(call, base)
    base3 = DEP.base_seeds(call, base)
    bad = [f"{rd[0]} {rd[1]} seed {sd}: the base gives {v} in GNX's compare, {base3.get(rd, {}).get(sd)} in its own stage's"
           for rd, by in sorted(b0.items()) for sd, v in sorted(by.items())
           if base3.get(rd, {}).get(sd) is None or abs(v - base3[rd][sd]) > R5_TOL]
    new_s, base_s, bad_s = slices(call, new_r5, base3, base)
    bad += bad_s
    new_all, base_all = {**new_r5, **new_s}, {**base3, **base_s}
    screen = LTG.decide_screen(new_all, base_all)
    full_ready = all(is_read(fit_name(sd, sp), base) for sd in RULE.SEEDS for sp in M1.SPLITS)
    full = LTG.decide_full(new_all, base_all) if full_ready else None
    dep_rule = {"screen": DEP.decide_screen(new_all, base_all)["verdict"],
                "full_run": DEP.decide_full(new_r5, base3, new_s, base_s)["verdict"] if full_ready else None}
    bad += fit_problems(call, base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": f"{ARM} with LTM's NEXT input", "iso_verdict": call,
           "base": "zg6r-m (ISO)" if call == ISO.REPLACE else "zg6r (ML1)", "screen": screen, "full_run": full,
           "dep_rule_beside": dep_rule, "problems": bad,
           "parameters": {f: {k: fit_block(f, "screen", base).get(k) for k in term_record_keys()}
                          for f in FITS if is_read(f, base)},
           "zgnx_sha256": M1.WQ.sha_src(__file__), "zltg_sha256": M1.WQ.sha_src(LTG.__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    md = [f"# GNX: {ARM} with LTM's NEXT input against {rec['base']}, on PKO's pools", "",
          f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}",
          f"DEP's sixteen-read rule beside: screen {dep_rule['screen']}, full run {dep_rule['full_run']}", ""]
    rows = screen["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in screen.get("beside", [])]
    if rows:
        md += ["| split | read | base s0, s1, s2 | GNX s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
        md += [f"| {r['split']} | {r['read']} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | "
               f"{r['delta']:+.4f} | {r['floor']:.4f} | {r['call']} |" for r in rows]
    if full is not None:
        md += ["", f"## Full run: {full['verdict']}", ""]
        frows = full["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in full.get("beside", [])]
        if frows:
            md += ["| split | read | d0 | d1 | d2 | floor | call |", "|---|---|---:|---:|---:|---:|---|"]
            for r in frows:
                d = r["delta"]
                md.append(f"| {r['split']} | {r['read']} | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} | {r['floor']:.4f} | "
                          f"{r['call']} |")
    if bad:
        md += ["", "problems:"] + [f"- {p}" for p in bad]
    (out / "calls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"screen {screen['verdict']} {json.dumps(screen.get('counts', {}))}"
        + (f"; full run {full['verdict']}" if full else "") + f"; DEP's rule beside {dep_rule}; {OUT}/calls.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def toy_term(G):
    """GNX on zg6r's own toy carve (LTG's toy: zg6r's self-test steps 2 and 3) with LTM-format toy columns: the batch's
    NEXT rows are the carve's own rows of its questions (against the rank block's rows, questions out of order, NaN 0);
    the carve's columns are LTM's files, float16 on the CPU; the carve refuses no split, a missing split's columns,
    another scorer's and a batch that already holds NEXT; gnx_w is (hidden x 15) zeros; at gnx_w = 0 the eval and the training
    forward are zg6r's with the same parameters bit for bit; gnx_w gets a gradient and moves the score; l1 outside the
    forward and a batch without NEXT stop; the strict load with and without the patch; a one-epoch lean_gpu.fit_variant
    that moves gnx_w; the patches' restoring."""
    import tempfile
    import torch
    G6 = G.G6
    G.LG.bind_device_ops()
    out = {}
    sp = "WQ-musique"
    with tempfile.TemporaryDirectory(prefix="zgnx_") as td:
        tmp = Path(td)
        rng = np.random.default_rng(37)
        G.RM.toy_roots(tmp, rng, chunks=(12, 12))
        G.RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                   rel_dir=tmp / "rel")
        G.ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        kw = dict(look_root=tmp / "look", cache_root=tmp / "cache", links_root=tmp / "lk", ch_root=tmp / "ch")

        class ToyIds(G6.ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        X0 = LTM.toy_cols(tmp / "ltm", sp, "metaqa", "toy", tmp / "cache", rng, "a" * 64)
        with G.U2N.extra_carves(("metaqa", "toy")):
            G.U2N.build_typed("metaqa", "toy", tmp / "look", tmp / "typed", **kw)
            plain = G.ZL.link_carve(G.RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
            c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            G.Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
            cls = G.g6r_carve(G.Z3.kt_carve(plain, tmp / "kt"), tmp / "typed")
            mk = lambda: cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")  # noqa: E731
            ce0 = mk()
            init0, batch0 = cls.__init__, cls.batch
            X = LTM.load_next("metaqa", "toy", sp, ce0.part_shas, tmp / "ltm")[0]
            out["files_are_toy_cols"] = bool(np.array_equal(X, X0)) and X.dtype == np.float16
            with on_carve(G, cls):
                for args, what in (((None,), "no_split"), (("WQ-hotpotqa",), "other_split"),
                                   ((sp, "b" * 64), "other_scorer")):
                    try:
                        with on_split(*args, out=tmp / "ltm") if args[0] else contextlib.nullcontext():
                            mk()
                        out[what] = "passed"
                    except SystemExit:
                        out[what] = "refused"
                LOADED.clear()
                with on_split(sp, out=tmp / "ltm"):
                    ce = mk()
                    nx = next_block(sp)
                out["loaded"] = sorted(nx["carves"])
                out["scorer"] = nx["fit_sha256"]
                out["carve_cols_are_files"] = bool(torch.equal(ce.gnx_X, torch.from_numpy(X)))
                out["carve_cols_on_cpu"] = ce.gnx_X.device.type == "cpu" and ce.gnx_X.dtype == torch.float16
                ce.gnx_X = ce.gnx_X.clone()            # NaN in a carve's columns is 0 in the batch
                ce.gnx_X[torch.from_numpy(rng.random(tuple(ce.gnx_X.shape)) < 0.05)] = float("nan")
                out["nan_set"] = int(torch.isnan(ce.gnx_X).sum())
                blocks = ["rank", "SEMB"]
                qs = np.r_[np.arange(ce.rows)[::-1][:7], np.arange(ce.rows)[:5]].astype(np.int64)
                B = qs.size
                feats, nq, bz, gold = ce.batch(qs, blocks)
                try:
                    with M1.WQ.patch(cls, batch=lambda self, q, b: (dict(feats), nq, bz, gold)):
                        with on_carve(G, cls):
                            ce.batch(qs, blocks)
                    out["next_twice"] = "passed"
                except RuntimeError:
                    out["next_twice"] = "refused"
            out["carve_restored"] = cls.__init__ is init0 and cls.batch is batch0
            rows = np.concatenate([np.arange(ce.off_np[q], ce.off_np[q + 1]) for q in qs])
            out["rank_rows_are_carve_rows"] = bool(torch.equal(feats["rank"], ce.X[torch.from_numpy(rows)]))
            want = torch.nan_to_num(ce.gnx_X[torch.from_numpy(rows)].to(torch.float32), nan=0.0)
            out["next_rows_are_carve_rows"] = bool(torch.equal(feats[NEXT], want))
            out["next_nan_free"] = bool(torch.isfinite(feats[NEXT]).all()) and bool(torch.isnan(want).sum() == 0)
            keep = torch.ones(B, len(blocks))
            orig_init, orig_fwd = G.ZG6R.__init__, G.ZG6R.forward
            names0 = tuple(G6.G6_NAMES) + tuple(G.G6R_NAMES)
            gen = torch.Generator().manual_seed(3)
            p0 = G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)    # zg6r's own: built outside the patch, no term
            out["plain_has_term"] = hasattr(p0, "gnx_w")
            with on_model(G):
                torch.manual_seed(0)
                m = G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)
                out["gnx_shape"] = tuple(m.gnx_w.shape)
                out["gnx_zero"] = bool((m.gnx_w == 0).all())
                out["l1_out"] = int(m.l1.out_features)
                out["term_params"] = int(m.gnx_w.numel())
                with torch.no_grad():
                    for k in names0:
                        getattr(m, k).copy_(torch.randn(getattr(m, k).shape, generator=gen) * 0.3)
                p0.load_state_dict({k: v for k, v in m.state_dict().items() if k != "gnx_w"}, strict=True)
                m.eval()
                p0.eval()
                with torch.no_grad():
                    out["eval_start_is_zg6r"] = bool(torch.equal(m(feats, keep, nq, B, bz), p0(feats, keep, nq, B, bz)))
                m.train()
                p0.train()
                p0.g6_gen.set_state(m.g6_gen.get_state())
                torch.manual_seed(9)
                y1 = m(feats, keep, nq, B, bz)
                torch.manual_seed(9)
                y2 = p0(feats, keep, nq, B, bz)
                out["train_start_is_zg6r"] = bool(torch.equal(y1, y2))
                m.zero_grad()
                G.LG.listwiseD(y1, gold, nq, B).backward()
                out["gnx_grad"] = float(m.gnx_w.grad.abs().sum()) if m.gnx_w.grad is not None else 0.0
                with torch.no_grad():
                    m.gnx_w.copy_(torch.randn(m.gnx_w.shape, generator=gen) * 0.3)
                m.eval()
                with torch.no_grad():
                    out["term_moves_score"] = not bool(torch.equal(m(feats, keep, nq, B, bz), p0(feats, keep, nq, B, bz)))
                    out["gnx_in_cleared"] = m.gnx_in is None
                    try:
                        m.encode(feats, keep, nq, B)
                        out["l1_outside"] = "passed"
                    except RuntimeError:
                        out["l1_outside"] = "refused"
                    try:
                        m({k: v for k, v in feats.items() if k != NEXT}, keep, nq, B, bz)
                        out["no_next"] = "passed"
                    except RuntimeError:
                        out["no_next"] = "refused"
                G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0).load_state_dict(m.state_dict(), strict=True)
                out["strict_load_patched"] = "passed"
            out["model_restored"] = G.ZG6R.__init__ is orig_init and G.ZG6R.forward is orig_fwd
            try:
                G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0).load_state_dict(m.state_dict(), strict=True)
                out["strict_load_plain"] = "passed"
            except RuntimeError:
                out["strict_load_plain"] = "refused"
            # a one-epoch lean_gpu.fit_variant on the patched carve and model: the term trains inside the trainer's own loop
            cfg = {"lr": 2e-3, "wd": 1e-4, "dropout": 0.1, "epochs": 1, "swa_from": 0, "cos": 0, "adamw": 0, "drop": 0}
            with on_carve(G, cls), on_model(G), M1.WQ.patch(G.LG, LeanMLP8D=G.ZG6R):
                res = G.LG.fit_variant([ce], blocks, cfg, 0, 16, "none", "cpu")
            out["fit_gnx_moved"] = float(res["swa"]["gnx_w"].abs().sum()) > 0
            out["fit_loss_finite"] = all(math.isfinite(c["loss"]) for c in res["curve"])
            out["fit_restored"] = G.LG.LeanMLP8D is not G.ZG6R and G.ZG6R.__init__ is orig_init and cls.batch is batch0
        LOADED.clear()
    return out


def selftest():
    assert fit_name(0, "WQ-musique") == "scr-gnx-nx-s0" and fit_name(2, "WQ-hotpotqa") == "scr-gnx-nx-s2-hp"
    assert pair_name(1, "WQ-hotpotqa") == "scr-gnx-nx-vs-base-s1-hp"
    assert pairs(ISO.KEEP)[pair_name(0, "WQ-musique")] == ("scr-gnx-nx-s0", "scr-ml1-zg6r-s0")
    assert pairs(ISO.REPLACE)[pair_name(0, "WQ-hotpotqa")] == ("scr-gnx-nx-s0-hp", "scr-iso-zg6rm-s0-hp")
    import zrtr as RTR
    import zira as IRA
    import zisb as ISB
    assert not set(FITS) & (set(RTR.FITS) | set(IRA.FITS) | set(ISB.FITS) | set(LTM.FITS))
    assert set(train_carves("WQ-hotpotqa")) == {"webqsp/fit", "squad/fit", "musique/s1fit", "2wiki/fit"}
    assert len(eval_carves()) == 6 and "musique/s1eval" in eval_carves()
    # 1. GNX on zg6r's toy carve
    G = import_g()
    t = toy_term(G)
    assert (t["no_split"], t["other_split"], t["other_scorer"]) == ("refused",) * 3, t
    assert t["loaded"] == ["metaqa/toy"] and t["scorer"] == "a" * 64 and t["carve_restored"], t
    assert t["files_are_toy_cols"] and t["carve_cols_are_files"] and t["carve_cols_on_cpu"], t
    assert t["nan_set"] > 0 and t["next_twice"] == "refused", t
    assert t["rank_rows_are_carve_rows"] and t["next_rows_are_carve_rows"] and t["next_nan_free"], t
    assert t["gnx_zero"] and t["gnx_shape"] == (16, IN) and t["l1_out"] == 16 and t["term_params"] == 16 * IN, t
    assert not t["plain_has_term"] and t["gnx_in_cleared"], t
    assert t["eval_start_is_zg6r"] and t["train_start_is_zg6r"], t
    assert t["gnx_grad"] > 0 and t["term_moves_score"], t
    assert t["l1_outside"] == "refused" and t["no_next"] == "refused" and t["model_restored"], t
    assert t["strict_load_patched"] == "passed" and t["strict_load_plain"] == "refused", t
    assert t["fit_gnx_moved"] and t["fit_loss_finite"] and t["fit_restored"], t
    # 2. the record, the patches and their restoring (also on a failure); the stamp's refusals
    rec = term_record(G)
    assert rec["term_parameters"] == 1920 and rec["parameters_h128"] == 180218 + 1920 == 182138, rec
    n0 = len(M1.ALL_FITS)
    saved = (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.stamp, M1.code_shas, M1.load_encoder, M1.train, M1.read)
    gsaved = (G.encoder_used, G.ZG6R.__init__, G.ZG6R.forward, G.G6rLinkCarve.__init__, G.G6rLinkCarve.batch)
    seen = []
    orig_stamp = M1.stamp
    hp = fit_name(0, "WQ-hotpotqa")
    try:
        M1.stamp = lambda name, env, what, extra=None: seen.append((name, what, dict(extra or {})))
        with on_gnx(G, ISO.KEEP):
            assert len(M1.ALL_FITS) == n0 + 6 and (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY)
            assert M1.load_encoder is MS.ml1_load and G.encoder_used is gsaved[0] and RS.installed()
            assert M1.train is not saved[7] and M1.read is not saved[8]
            assert G.ZG6R.__init__ is not gsaved[1] and G.G6rLinkCarve.batch is not gsaved[4]
            cs = M1.code_shas(ARM)
            assert all(f"{k}_sha256" in cs for k in ("zgnx", "zdep", "zltg", "zltm", "ltmcols", "ml1start", "readswa"))
            M1._ACTIVE[0] = hp
            assert M1.pko_fit_name(ARM, 0, "WQ-hotpotqa") == hp
            assert M1.pko_fit_name(ARM, 1, "WQ-hotpotqa") == "scr-ml1-zg6r-s1-hp"
            M1._ACTIVE[0] = None
            LOADED.clear()
            for what, args in (("screen", ("WQ-musique",)), ("screen", ("WQ-hotpotqa",)), ("read", ("WQ-hotpotqa",))):
                try:
                    with on_split(*args):
                        M1.stamp(hp, {}, what, {"x": 1})
                    raise AssertionError(f"the {what} stamp passed with no columns under {args[0]}")
                except SystemExit:
                    pass
            with on_split("WQ-hotpotqa", "c" * 64):
                LOADED.extend({"dataset": k.split("/")[0], "carve": k.split("/")[1], "record_sha256": "r"}
                              for k in train_carves("WQ-hotpotqa"))
                M1.stamp(hp, {}, "screen", {"x": 1})
                try:
                    M1.stamp(hp, {}, "read", {"x": 2})
                    raise AssertionError("the read stamp passed without the s1eval carves' columns")
                except SystemExit:
                    pass
                LOADED.extend({"dataset": k.split("/")[0], "carve": k.split("/")[1], "record_sha256": "r"}
                              for k in eval_carves())
                M1.stamp(hp, {}, "read", {"x": 2})
            LOADED.clear()
        assert [s[:2] for s in seen] == [(hp, "screen"), (hp, "read")], seen
        nk = seen[0][2]["next"]
        assert nk["split"] == "WQ-hotpotqa" and nk["fit_sha256"] == "c" * 64 and seen[0][2]["base"] == "scr-ml1-zg6r-s0-hp"
        assert seen[0][2]["iso_verdict"] == ISO.KEEP and seen[0][2]["term_parameters"] == 1920 and seen[0][2]["x"] == 1
        assert sorted(seen[1][2]["next"]["carves"]) == sorted(set(train_carves("WQ-hotpotqa")) | set(eval_carves()))
        assert (G.encoder_used, G.ZG6R.__init__, G.ZG6R.forward, G.G6rLinkCarve.__init__, G.G6rLinkCarve.batch) == gsaved
        assert len(M1.ALL_FITS) == n0 and not RS.installed()
        with on_gnx(G, ISO.REPLACE):
            assert M1.load_encoder is ISO.own_load and G.encoder_used is ISO.own_used
        try:
            with on_gnx(G, ISO.KEEP):
                raise RuntimeError("inside")
        except RuntimeError:
            pass
    finally:
        M1.stamp = orig_stamp
        M1._ACTIVE[0] = None
        LOADED.clear()
    assert (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.stamp, M1.code_shas, M1.load_encoder, M1.train,
            M1.read) == saved
    assert (G.encoder_used, G.ZG6R.__init__, G.ZG6R.forward, G.G6rLinkCarve.__init__, G.G6rLinkCarve.batch) == gsaved
    assert len(M1.ALL_FITS) == n0 and CUR == {"split": None, "fit_sha256": None, "out": None} and not RS.installed()
    # 3. the rule is LTG's (its keys and calls, tested in zltg's self-test)
    assert LTG.deciding_keys() and set(LTG.deciding_keys()) | set(LTG.beside_keys()) == set(DEP.read_keys())
    log("selftest: names and pairs; GNX on zg6r's toy carve (the carve's columns are LTM's files, float16 on the CPU; "
        "the batch's NEXT rows are its questions' carve rows, NaN 0; no split, another split's and another scorer's "
        "columns and a second NEXT block refused; gnx_w hidden x 15 zeros; at gnx_w = 0 the eval and training forwards "
        "zg6r's bit for bit; gnx_w's gradient and its move of the score; l1 outside the forward and a batch without "
        "NEXT stop; the strict load under the patch and its refusal without; a one-epoch fit_variant that moves gnx_w); "
        "the patches and their restoring (also on a failure); the stamp's refusals without the columns; LTG's rule")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fit", "compare-all", "calls", "check-cols"))
    ap.add_argument("--name")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "fit":
        return fit(a.name)
    if a.cmd == "compare-all":
        return compare_all()
    if a.cmd == "calls":
        return calls()
    if a.cmd == "check-cols":
        return check_cols()
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
