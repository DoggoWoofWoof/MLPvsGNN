"""Design look (untracked; not a result and not filed): a text map that cannot memorise, and the 4096-phrase vocabulary.

anchor_gen.py (sha256 bdc938f5...) hides one fold of 2wiki's phrases in training and shows them at REV. The text model
A1024-1/txt0 reads REV - MASK +0.0000 [-0.0015, +0.0014] on R@5, and on the 227 rows where a gold sits one hidden-phrase
edge from a seed REV reads below MASK. A phrase's code P phi(k) is a 1536 x 64 map fitted on about 820 phrases: it can
give every training phrase any vector it likes (a free table with extra steps), so a phrase it never saw gets an arbitrary
code. This look reads the phrase through a fixed whitened PCA of the phrase table:
    z(k) = Lambda_r^{-1/2} V_r^T (phi(k) - mu)          r components (8, 16, 32, 64), from the fit's own top-K phrase
                                                        vectors with the held fold left out (label-free; no row is read)
    h(k) = P_r z(k) + D[f(k)]                           P_r is r x 64: at most 4096 numbers, for any vocabulary size
so a code is a smooth function of a few directions of the phrase's text, shared by every phrase. Everything else is
GenMix's (anchor_gen): the operators g<op>, the cos suffix, the free codes of 'other', ner and knn. The model's size does
not grow with K, so this look also reads K = 4096 (every phrase string the compact stores). The phrase table is the
4096-row gte-Qwen2 table of each dataset (phrase_emb_w2_K4096.npy, pinned below; its first 1024 rows differ from the
1024-row table by at most 0.03 per entry, batching), for every K. Relation dropout (anchor_gen2) is available too.

    <train>:A{256,1024,4096}[-1]/<model>[/<rule>][@seeds]   model: lin (free), txt0, g<op>[p<r>][c]
    python outputs/mp_approx_2wiki_anchor/host/anchor_gen3.py --dataset 2wiki --variants x4+x5+x6:A4096-1/glinp32 [--hold 0/5]
    python outputs/mp_approx_2wiki_anchor/host/anchor_gen3.py --xread --target hotpotqa --models a.pt,b.pt --out PATH
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import anchor_gen as AG  # noqa: E402
import anchor_gen2 as G2N  # noqa: E402

AW11, AW8, AW7, AW6, A16, AW, AW3 = AG.AW11, AG.AW8, AG.AW7, AG.AW6, AG.A16, AG.AW, AG.AW3
ANCHOR_GEN_SHA = G2N.ANCHOR_GEN_SHA
ANCHOR_GEN2_SHA = "cbb976408648e2da0d5398df7310b2695a471e246f7def05413f74cec84711c6"
R_TOP = AG.R_TOP
KS = (256, 1024, 4096)
RS = (8, 16, 32, 64)
PHRASE4096 = {"2wiki": (ROOT / "outputs" / "mp_approx_2wiki_anchor" / "host" / "phrase_emb_w2_K4096.npy",
                        "1cf77680b826f1ad8dc5657b13972905101aef2e8b04bbd9f22bb7c635136cf8"),
              "hotpotqa": (ROOT / "outputs" / "mp_approx_hotpot_anchor" / "host" / "phrase_emb_w2_K4096.npy",
                           "860de3b25b991adef01ae6d9e9cff0556676fb4d51dee65ce4c6da8660a14873"),
              "musique": (ROOT / "outputs" / "mp_approx_musique_anchor" / "host" / "phrase_emb_w2_K4096.npy",
                          "77cca00ef4691c9843c8326e97963659984a7d57c2574509391534d6990280bb")}
log = AG.log


def phrase_table(ds):
    pth, want = PHRASE4096[ds]
    got = AW.sha(pth)
    if got != want:
        raise SystemExit(f"{pth} is not the pinned 4096-phrase table")
    phi = np.load(pth).astype(np.float32)
    if phi.shape != (R_TOP, 1536):
        raise SystemExit("unexpected phrase table shape")
    return phi, got


def pca_basis(rows, r):
    """(mu, V (1536 x r), lam (r)) of the rows: the top r principal directions, each signed so its largest entry is
    positive; lam the variances along them."""
    X = np.asarray(rows, dtype=np.float64)
    mu = X.mean(0)
    _U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    V = Vt[:r].T.copy()
    sgn = np.sign(V[np.abs(V).argmax(0), np.arange(r)])
    V *= sgn
    lam = (S[:r] ** 2) / max(X.shape[0] - 1, 1)
    explained = float((S[:r] ** 2).sum() / (S ** 2).sum())
    return mu.astype(np.float32), V.astype(np.float32), lam.astype(np.float32), explained


class GenMixP(AG.GenMix):
    """GenMix with the phrase code read through a fixed whitened PCA (see the docstring): h(k) = P_r z(k) + D[f(k)]."""

    def __init__(self, nt, K, phi, kind, r, cos=False, d=64, qdim=1536, basis=None):
        super().__init__(nt, K, phi, kind, cos=cos, d=d, qdim=qdim)
        del self.P
        if basis is None:
            basis = pca_basis(phi[:K], r)[:3]
        mu, V, lam = basis
        self.r = r
        self.register_buffer("mu", torch.as_tensor(np.asarray(mu, dtype=np.float32)))
        self.register_buffer("V", torch.as_tensor(np.asarray(V, dtype=np.float32)))
        self.register_buffer("lam", torch.as_tensor(np.asarray(lam, dtype=np.float32)))
        self.Pr = torch.nn.Linear(r, d, bias=False)
        torch.nn.init.normal_(self.Pr.weight, std=0.1 / math.sqrt(r))
        self.register_buffer("Z", self._z(), persistent=False)

    def _z(self):
        Z = ((self.Phi - self.mu) @ self.V) / torch.sqrt(self.lam)
        return torch.where(self.pmask[:, None], Z, torch.zeros_like(Z))

    def set_phi(self, phi):
        super().set_phi(phi)
        self.Z = self._z()

    def codes(self):
        base = self.Pr(self.Z)
        h = torch.where(self.pmask[:, None], base + self.D[self.fdir], self.t1)
        if self.kind != "lin":
            return h, None
        h2 = torch.where(self.pmask[:, None], self.W2(base) + self.D2[self.fdir], self.t2[1:])
        return h, h2


def parse(name, train_looks):
    """anchor_gen.parse with K 4096 and the PCA suffix p<r> on gen models."""
    head, at, seeds = name.partition("@")
    train, sep, rest = head.partition(":")
    parts = rest.split("/")
    base = parts[0]
    m = re.fullmatch(r"A(\d+)(-1)?", base)
    if not m or int(m.group(1)) not in KS:
        raise SystemExit(f"{name}: base A{{256,1024,4096}}[-1]")
    K = int(m.group(1))
    model = parts[1] if len(parts) > 1 else "lin"
    pca = None
    mm = re.fullmatch(r"g([a-z]+?)p(\d+)(c?)", model)
    if mm and mm.group(1) in AG.GEN:
        pca = int(mm.group(2))
        if pca not in RS:
            raise SystemExit(f"{name}: PCA r among {RS}")
        model = f"g{mm.group(1)}{mm.group(3)}"
    proxy = "/".join([f"A1024{m.group(2) or ''}", model] + parts[2:])
    sp = AG.parse(f"{train}:{proxy}{at}{seeds}", train_looks)
    sp.update({"base": base, "K": K, "pca": pca, "model": parts[1] if len(parts) > 1 else "lin"})
    if pca and sp["family"] != "gen":
        raise SystemExit(f"{name}: the PCA code is a gen model's")
    return sp


def make_for(sp, nt, phi, basis=None):
    if sp["family"] == "gen" and sp["pca"]:
        return lambda: GenMixP(nt, sp["K"], phi, sp["op"], sp["pca"], cos=sp["cos"], basis=basis)
    if sp["family"] == "gen":
        return lambda: AG.GenMix(nt, sp["K"], phi, sp["op"], cos=sp["cos"])
    return AW7.make_for(sp["model"], nt, sp["K"], phi)


def set_phi(model, phi, K):
    if isinstance(model, AG.GenMix):
        model.set_phi(phi)
    else:
        AG.set_phi(model, phi, K)


def load_model(path, phi):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sp, nt, sd = ck["spec"], ck["nt"], ck["state_dict"]
    basis = (sd["mu"].numpy(), sd["V"].numpy(), sd["lam"].numpy()) if sp.get("pca") else None
    model = make_for(sp, nt, phi, basis)()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected or any(k not in ("Phi", "pmask", "fdir", "Z") for k in missing):
        raise SystemExit(f"{path}: state dict does not match ({missing}, {unexpected})")
    if isinstance(model, GenMixP):
        model.Z = model._z()
    model.eval()
    return model, ck


def save_model(path, model, sp, nt, margin, ds, vocab_w2, phi_sha, extra):
    sd = {k: v for k, v in model.state_dict().items() if k not in ("Phi", "pmask", "fdir", "Z")}
    torch.save({"spec": sp, "nt": nt, "margin": margin, "dataset": ds, "vocab_w2": list(vocab_w2[:sp["K"]]),
                "phi_sha": phi_sha, "state_dict": sd, **extra}, path)


def main_fit(a):
    AW6.rebind(a.dataset)
    out_path = Path(a.out) if a.out else HERE / f"anchor_gen3_{a.dataset}.json"
    torch.set_num_threads(2)
    t0 = time.time()
    train_looks = AW6.TRAIN[a.dataset] if a.dataset != "2wiki" else ("fit", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12")
    specs = {name: parse(name, train_looks) for name in a.variants.split(",")}
    hold = None
    if a.hold:
        h, F = (int(x) for x in a.hold.split("/"))
        if not 0 <= h < F or F < 2:
            raise SystemExit("--hold h/F with 0 <= h < F, F >= 2")
        hold = (h, F)
    phi, phi_sha = (None, None)
    if any(sp["family"] != "free" for sp in specs.values()):
        phi, phi_sha = phrase_table(a.dataset)
    looks = ["x1", "select"] + [lk for lk in train_looks if any(lk in sp["train"] for sp in specs.values())]
    Q, part, rec_sha, load_state = AG.load_looks(looks, max(sp["max_len"] for sp in specs.values()), t0)
    checks, vocab = AW.anchor_tables(Q)
    vocab_w2 = vocab.get("w2", [])
    if len(vocab_w2) < max(sp["K"] for sp in specs.values()):
        raise SystemExit("the compact's phrase strings do not cover K")
    log(f"anchors attached to {len(Q)} rows: {checks}, {time.time() - t0:.0f}s")
    x1 = part["x1"]
    B_rows = x1[1::2]
    RB, RX = AG.Reader(Q, B_rows, 20261002), AG.Reader(Q, x1, 20261003)
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    zT_of = [None] * len(Q)
    if a.kd > 0:
        for i in sorted({i for sp in specs.values() for lk in sp["train"] for i in part[lk]}):
            zT_of[i] = AW8.teacher_of(Q[i], a.teacher)
    rdrop = {"row": a.rdrop_row, "edge": a.rdrop_edge, "unit": a.rdrop_unit, "seed": G2N.DROP_SEED}
    res = {"look": "anchor_gen3", "mode": "fit", "dataset": a.dataset, "script_sha256": AW.sha(Path(__file__)),
           "pins": {"anchor_gen": ANCHOR_GEN_SHA, "anchor_gen2": ANCHOR_GEN2_SHA, **AG.PINS, "anchor_walk10": AW11.AW10_SHA, **AW6.PINS,
                    "compact": AW.COMPACT_SHA, "struct": AW.STRUCT_SHA, "phrase_table": phi_sha, "look_score_records": rec_sha},
           "flag_checks": checks, "pruned_loader": load_state, "looks": {lk: len(r) for lk, r in part.items()}, "epochs": a.epochs,
           "lr": a.lr, "wd": a.wd, "kd": {"ce": a.ce, "kd": a.kd, "T": a.T, "teacher": a.teacher} if a.kd > 0 else None, "hold": a.hold,
           "rdrop": rdrop, "B": RB.base(), "x1": RX.base(), "variants": {}}
    per_row = {}
    model_dir = out_path.parent / (out_path.stem + "_models")
    model_dir.mkdir(parents=True, exist_ok=True)

    def save():
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, B_rows=np.asarray(B_rows))

    cache = {}
    for name, sp in specs.items():
        t1 = time.time()
        K = sp["K"]
        rm_train, held = AG.hold_map(K, vocab_w2, *hold) if hold else (AG.identity_map(K), None)
        tr = [i for lk in sp["train"] for i in part[lk]]
        key = (K, sp["max_len"], tuple(sp["train"]))
        if key not in cache:
            cache.clear()
            drops, whole = G2N.drop_masks(Q, tr, K, a.rdrop_row, a.rdrop_edge, a.rdrop_unit, G2N.DROP_SEED)
            nt, TY = G2N.types_dropped(Q, sp, rm_train, drops)
            cache[key] = (nt, TY, {"rows_whole": whole, "rows_partial": len(drops) - whole,
                                   "edges_dropped": int(sum(int(dm.sum()) for dm in drops.values()))})
        nt, TY, stats = cache[key]
        alt = {"NR": AG.types_for(Q, sp, np.full(R_TOP + 1, K, dtype=np.int64), B_rows)[1]}
        if hold and sp["family"] != "free":
            alt["REV"] = AG.types_for(Q, sp, AG.identity_map(K), B_rows)[1]
        basis, pca_rec = None, None
        if sp["pca"]:
            keep = np.ones(K, dtype=bool) if held is None else ~held[:K]
            mu, V, lam, expl = pca_basis(phi[:K][keep], sp["pca"])
            basis, pca_rec = (mu, V, lam), {"r": sp["pca"], "phrases": int(keep.sum()), "explained": round(expl, 4)}
        make = make_for(sp, nt, phi, basis)
        hrows = AG.held_reach_rows(Q, B_rows, held, K) if hold else None
        models, seeds_out = [], {}
        main_key = "ID" if not hold else "MASK"
        for sd in sp["seeds"]:
            t2 = time.time()
            model, reads, best_ep, curve, margin, by_margin = AW8.fit_read_kd(Q, TY, nt, tr, part["select"], {"B": B_rows, "x1": x1}, z_of, zT_of,
                                                                              make, a.epochs, sd, sp["rule"], a.lr, a.wd, a.ce, a.kd, a.T)
            models.append(model)
            vi = len(res["variants"])
            rd = {main_key: RB.record(reads["B"]), main_key + "_x1": RX.record(reads["x1"], by_type=False)}
            per_row[f"{vi}_{sd}_{main_key}"] = reads["B"]
            for nm, TYa in alt.items():
                m_alt = AG.read(model, Q, TYa, nt, B_rows, z_of, margin)
                rd[nm] = RB.record(m_alt)
                per_row[f"{vi}_{sd}_{nm}"] = m_alt
                rd[f"{nm} - {main_key}"] = AW3.boot_pair(m_alt - reads["B"], RB.W)
            if hold:
                rd["held_reach_rows"] = {"MASK": RB.subset(reads["B"], hrows)}
                if "REV" in alt and hrows:
                    pos = {i: j for j, i in enumerate(B_rows)}
                    hx = np.asarray([pos[i] for i in hrows], dtype=np.int64)
                    rd["held_reach_rows"]["REV"] = RB.subset(per_row[f"{vi}_{sd}_REV"], hrows)
                    rd["held_reach_rows"]["REV - MASK"] = AW3.boot_pair((per_row[f"{vi}_{sd}_REV"] - reads["B"])[hx], RB.W[:, hx])
            pt = model_dir / f"v{vi}_s{sd}.pt"
            save_model(pt, model, sp, nt, margin, a.dataset, vocab_w2, phi_sha, {"hold": a.hold, "name": name, "seed": sd, "rdrop": rdrop})
            seeds_out[str(sd)] = {"best_epoch": best_ep, "curve": curve, "margin": None if margin is None else str(margin), "by_margin_select": by_margin,
                                  "kappa": float(torch.exp(model.log_kappa).item()), "beta": float(torch.nn.functional.softplus(model.beta_raw).item()),
                                  "gam": model.gam.detach().round(decimals=4).tolist() if hasattr(model, "gam") else None,
                                  "model_file": str(pt.relative_to(ROOT)) if pt.is_relative_to(ROOT) else str(pt), "reads": rd,
                                  "seconds": round(time.time() - t2, 1)}
            log(f"{name}#{sd} ({sp['family']}): ep {best_ep}, " + ", ".join(f"{k} {v['rho (R@5, FC@5, hit@1)']}" for k, v in rd.items()
                                                                       if isinstance(v, dict) and "rho (R@5, FC@5, hit@1)" in v))
        m0 = models[0]
        v = {**sp, "train_rows": len(tr), "tokens": nt, "types_per_row": round(float(np.mean([len(TY[i]) for i in tr])), 1),
             "params": int(sum(p.numel() for p in m0.parameters())), "params_used": AG.params_used(m0, sp), "pca_basis": pca_rec,
             "rdrop_stats": stats, "seeds_read": seeds_out}
        if hold:
            v["held"] = {"phrases": int(held[:K].sum()), "of_K": K, "held_reach_rows": len(hrows)}
        v["seconds"] = round(time.time() - t1, 1)
        res["variants"][name] = v
        save()
    res["seconds"] = round(time.time() - t0, 1)
    save()
    log(f"done in {res['seconds']}s")


def main_xread(a):
    """anchor_gen.main_xread with this look's models and the 4096-phrase tables."""
    if a.target not in AG.DATASETS:
        raise SystemExit(f"--target among {AG.DATASETS}")
    AW6.rebind(a.target)
    out_path = Path(a.out)
    torch.set_num_threads(2)
    t0 = time.time()
    paths = [Path(p) for p in a.models.split(",")]
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    Q, part, rec_sha, load_state = AG.load_looks(["x1"], max(ck["spec"]["max_len"] for ck in cks), t0)
    checks, vocab = AW.anchor_tables(Q)
    vocab_w2 = vocab.get("w2", [])
    x1 = part["x1"]
    B_rows = x1[1::2]
    RB, RX = AG.Reader(Q, B_rows, 20261002), AG.Reader(Q, x1, 20261003)
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    phi, phi_sha = phrase_table(a.target) if any(ck["spec"]["family"] != "free" for ck in cks) else (None, None)
    res = {"look": "anchor_gen3", "mode": "xread", "target": a.target, "script_sha256": AW.sha(Path(__file__)),
           "pins": {"anchor_gen": ANCHOR_GEN_SHA, "anchor_gen2": ANCHOR_GEN2_SHA, "compact": AW.COMPACT_SHA, "struct": AW.STRUCT_SHA,
                    "phrase_table": phi_sha, "look_score_records": rec_sha},
           "flag_checks": checks, "pruned_loader": load_state, "B": RB.base(), "x1": RX.base(), "models": {}}
    per_row = {}
    for j, (p, ck) in enumerate(zip(paths, cks)):
        t1 = time.time()
        sp, K = ck["spec"], ck["spec"]["K"]
        model, _ck = load_model(p, phi if sp["family"] != "free" else None)
        if sp["family"] != "free":
            set_phi(model, phi, K)
        ent = {"from": ck["dataset"], "name": ck.get("name"), "seed": ck.get("seed"), "hold": ck.get("hold"), "spec": sp,
               "margin": ck["margin"], "model_file": str(p)}
        rm = AG.identity_map(K) if sp["family"] != "free" else AG.string_map(K, vocab_w2, ck["vocab_w2"])
        if sp["family"] == "free":
            ne = nm_ = 0
            for i in B_rows:
                if (Q[i]["fam"] == 0).any():
                    _m, _d, a_ = AG.edge_ranks(Q[i])
                    ne += int(a_.size)
                    nm_ += int((rm[np.minimum(a_, R_TOP)] < K).sum())
            ent["string_matched_edge_share_B"] = round(nm_ / max(ne, 1), 4)
        nt, TY = AG.types_for(Q, sp, rm, x1)
        _nt, TYn = AG.types_for(Q, sp, np.full(R_TOP + 1, K, dtype=np.int64), B_rows)
        mB = AG.read(model, Q, TY, nt, B_rows, z_of, ck["margin"])
        mX = AG.read(model, Q, TY, nt, x1, z_of, ck["margin"])
        mN = AG.read(model, Q, TYn, nt, B_rows, z_of, ck["margin"])
        per_row[f"{j}_XD"], per_row[f"{j}_XDNR"] = mB, mN
        ent["XD"], ent["XD_x1"], ent["XD_NR"] = RB.record(mB), RX.record(mX, by_type=False), RB.record(mN)
        ent["XD_NR - XD"] = AW3.boot_pair(mN - mB, RB.W)
        ent["seconds"] = round(time.time() - t1, 1)
        res["models"][p.name if p.name not in res["models"] else f"{p.parent.name}/{p.name}"] = ent
        log(f"{p.name} ({ck['dataset']} -> {a.target}, {sp['model']}): XD {ent['XD']['rho (R@5, FC@5, hit@1)']}, XD_NR {ent['XD_NR']['rho (R@5, FC@5, hit@1)']}")
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez(out_path.with_suffix(".npz"), **per_row, B_rows=np.asarray(B_rows))
    res["seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"done in {res['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--variants", default="x4+x5+x6:A1024-1/glinp32")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--ce", type=float, default=1.0)
    ap.add_argument("--kd", type=float, default=0.0)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--teacher", default="gm", choices=AW8.TEACHERS)
    ap.add_argument("--hold", default=None)
    ap.add_argument("--rdrop_row", type=float, default=0.0)
    ap.add_argument("--rdrop_edge", type=float, default=0.0)
    ap.add_argument("--rdrop_unit", default="edge", choices=("edge", "phrase"))
    ap.add_argument("--xread", action="store_true")
    ap.add_argument("--target", default=None)
    ap.add_argument("--models", default=None)
    a = ap.parse_args(argv)
    if a.ce < 0 or a.kd < 0 or a.ce + a.kd <= 0 or a.T <= 0:
        raise SystemExit("ce and kd must be non-negative, not both 0, and T positive")
    if AW.sha(Path(AG.__file__)) != ANCHOR_GEN_SHA or AW.sha(Path(G2N.__file__)) != ANCHOR_GEN2_SHA:
        raise SystemExit("anchor_gen.py or anchor_gen2.py is not the pinned file")
    AG.check_pins()
    if a.xread:
        if not (a.target and a.models and a.out):
            raise SystemExit("--xread needs --target, --models and --out")
        main_xread(a)
    else:
        if a.dataset not in AG.DATASETS:
            raise SystemExit(f"--dataset among {AG.DATASETS}")
        main_fit(a)


if __name__ == "__main__":
    main()
