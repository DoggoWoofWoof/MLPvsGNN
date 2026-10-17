"""A5 (docs/A5_KIND_HEAD_BACKOFF.md): zck's kind head on the lean pair, its gradient kept off the shared weights.
zck (round 28, 9 Oct) gave zret's output layer and rrf's base weight learned offsets on a batch from a graph with typed
relations (a batch carrying chain entries: metaqa, webqsp); webqsp zero-shot gained +0.0160 and +0.0402, and 2wiki in
domain lost -0.0082 on batches the offsets never act on, through the weights shared in joint training. A5 keeps the head
and takes that path away:

    on a typed batch the score is s + kd_base z(rrf) + kd_w . h + kd_b, with h zret's last hidden units, detached, and
    z(rrf) zret's base term; kd_* zero at the start; any other batch's score is s
    training minimises L(s) + L(s.detach() + head): the first term trains every shared weight exactly as the base model
    trains (the base's loss on the base's score), the second trains the head alone

So the shared weights, and with them every read of a graph without typed relations, are the base model's bit for bit;
only the typed graphs' reads move, by the head. The head is placed at the top of each model (after zrckp's KT head; after
zspk's KT head and propagation) and reads zret's own h and z(rrf), stashed by A5Stash just above zret's forward. No score
read, nothing normalised over the question's scores; the graph's kind is the graph's own property.

    zrckpk  zrckp's model with the kind head (MRO: A5Top, ZRCKP, KTMixP, ZRM, ChainMatch, A5Stash, ZRet)
    zspkk   zspk's model with the kind head (MRO: A5Top, ZSPK, ZProp, KTMix, ZRM, ChainMatch, A5Stash, ZRet)

    python outputs/mp_unified/za5kind.py train --split L-musique --name scr-a5-zrckpk --arm zrckpk --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za5kind.py read --name scr-a5-zrckpk --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za5kind.py compare --new outputs/screen/fits/scr-a5-zrckpk \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a5-zrckpk
    python outputs/mp_unified/za5kind.py pair --arm zrckpk --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za5kind.py recall --arm zrckpk --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za5kind.py verdict
    python outputs/mp_unified/za5kind.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zp1c3 as Z3  # noqa: E402  (zspk and its KT carves; through it P1c2's roots, gate and card, zprop, zlink, rmatch)
import zp1c3p as Z3P  # noqa: E402  (zrckp)
import lean_screen3 as S3  # noqa: E402  (zret)
import zkind as ZK  # noqa: E402  (zck's head, read for its offsets' shapes; its Untyped toy carve)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
LM = LG.LM
log = S.log

DECLARED = "docs/A5_KIND_HEAD_BACKOFF.md"
OUT = ROOT / "outputs" / "a5"
SCREEN = ROOT / "outputs" / "screen"
KD = ("kd_w", "kd_b", "kd_base")
SETTINGS = {"typed": "a batch carrying chain entries (rmatch's CH_KEY: metaqa, webqsp)",
            "head": "kd_base z(rrf) + kd_w . h.detach() + kd_b, zret's h and base term, zero at the start",
            "where": "the top of the model", "loss": "L(s) + L(s.detach() + head) on a typed batch, L(s) otherwise",
            "shared_weights": "the base model's gradients bit for bit", "score_input": False}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpk": "zrckp", "zspkk": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a5-{a}", "L-hotpotqa": f"scr-a5-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpk": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspkk": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpk": "zrckp with zck's kind head on typed graphs, its gradient kept off the shared weights",
        "zspkk": "zspk with zck's kind head on typed graphs, its gradient kept off the shared weights"}
PAIR = {a: SCREEN / f"scr-a5-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a5-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
TYPED = ("metaqa", "webqsp")
BESIDE = ("metaqa", "webqsp", "2wiki")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A5's re-call of the arm, or "
        "not twelve reads (docs/A5_KIND_HEAD_BACKOFF.md, section 4)")
CUR = {}                                                     # the last typed training forward: its output, s and head
LAST = {"typed_forwards": 0, "untyped_forwards": 0}           # the last fit's counts


# ── the head ─────────────────────────────────────────────────────────────────


class A5Stash(S3.ZRet):
    """zret's forward, its last hidden units and base term kept on the model for A5Top (the score unchanged)."""

    def forward(self, feats, keep, nq, B, base_z):
        ref = S3.retrieved(feats)
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = S.raw_of(self, feats, b, nq)
            parts += [raw * m, S3.seg_zscore_ref(raw, nq, B, ref) * m, m]
        h = self.drop(Fn.gelu(self.l1(torch.cat(parts, 1))))
        h = self.drop(Fn.gelu(self.l2(h)))
        bz = S3.seg_zscore_ref(feats["rank"][:, S3.I_RRF:S3.I_RRF + 1], nq, B, ref).squeeze(1)
        self.a5_h, self.a5_bz = h, bz
        return self.base_w * bz + self.out(h).squeeze(-1)


class A5Top(S3.ZRet):
    """At the top of the arm's MRO: on a typed batch, the model's score plus the kind head over zret's detached hidden
    units and base term (zck's offsets on the output layer and base weight, as one head)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.kd_w = nn.Parameter(torch.zeros_like(self.out.weight))
        self.kd_b = nn.Parameter(torch.zeros_like(self.out.bias))
        self.kd_base = nn.Parameter(torch.zeros_like(self.base_w))
        LAST.update(typed_forwards=0, untyped_forwards=0)

    def forward(self, feats, keep, nq, B, base_z):
        CUR.clear()
        s = super().forward(feats, keep, nq, B, base_z)
        if RM.CH_KEY not in feats:
            if self.training:
                LAST["untyped_forwards"] += 1
            return s
        head = self.kd_base * self.a5_bz + Fn.linear(self.a5_h.detach(), self.kd_w, self.kd_b).squeeze(-1)
        out = s + head
        if self.training:
            LAST["typed_forwards"] += 1
            CUR.update(out=out, s=s, head=head)
        return out


@contextlib.contextmanager
def a5_loss():
    """lean_gpu's listwise loss, for the output of a typed training forward of an A5 arm: L(s) + L(s.detach() + head).
    Any other scores take the loss as it is."""
    orig = LG.listwiseD

    def loss(scores, gold, nq, B):
        if CUR.get("out") is scores:
            s, head = CUR["s"], CUR["head"]
            CUR.clear()
            return orig(s, gold, nq, B) + orig(s.detach() + head, gold, nq, B)
        return orig(scores, gold, nq, B)

    LG.listwiseD = loss
    try:
        yield
    finally:
        LG.listwiseD = orig
        CUR.clear()


class ZRCKPK(A5Top, Z3P.ZRCKP, A5Stash):
    """zrckpk: zrckp with the kind head (MRO: A5Top, ZRCKP, KTMixP, ZRM, ChainMatch, A5Stash, ZRet)."""


class ZSPKK(A5Top, Z3.ZSPK, A5Stash):
    """zspkk: zspk with the kind head after its propagation (MRO: A5Top, ZSPK, ZProp, KTMix, ZRM, ChainMatch, A5Stash,
    ZRet)."""


ARMS = {"zrckpk": (ZRCKPK, Z3.KTChainCarveZRC), "zspkk": (ZSPKK, Z3.KTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def split_of_fit(name):
    for a in ARM_NAMES:
        for sp, n in FIT[a].items():
            if n == name:
                return a, sp
    return None, None


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"za5kind_sha256": LC.sha_src(__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__),
                    "lean_screen3_sha256": LC.sha_src(S3.__file__), "zkind_sha256": LC.sha_src(ZK.__file__),
                    "a5": {"declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                           "device": Z2.DEVICE, "settings": SETTINGS, "last_fit": dict(LAST),
                           "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                           "gate": LC.sha_file(Z2.GATE), "kt_records": Z3.kt_records(Z3.train_carves(split)),
                           "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                           "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)
        log(f"  a5 {arm} {split}: {LAST['typed_forwards']} typed and {LAST['untyped_forwards']} untyped training "
            f"forwards in the last fit")


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za5kind: train takes --arm zrckpk or zspkk, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za5kind: A5's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za5kind: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    with Z2.on_p1c2(), a5_loss():
        rc = RM.train(argv, split)
        stamp(argv, arm, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    arm, split = split_of_fit(name)
    if rec.get("arm") not in ARMS or "a5" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za5kind: {name} is not an A5 fit (arm {rec.get('arm')})")
    if rec["a5"].get("split") != split:
        raise SystemExit(f"za5kind: {name} is {split}'s fit, its record says {rec['a5'].get('split')}")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fits ───────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"za5kind: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za5kind: no base fit of {BASE_OF[arm]} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_OF[arm], fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b, f = BASE_OF[arm], BASE_FITS[arm]["L-musique"][2]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({b}'s card fits {f} and {f}-hp)"),
            ("(rel's screen fits)", f"({b}'s card fits)"),
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A5's kind head) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A5 ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A5", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za5kind_sha256": LC.sha_src(__file__)})
        rec.pop("round", None)
        LC.write_json(js, rec)


def pair(arm, screens, out):
    with on_base(arm):
        rec = Z.pair(screens, out)
    restamp(out, arm)
    return rec


def recall(arm, null_files, pair_file, out, base_screens=None):
    with on_base(arm):
        rec = Z.recall(null_files, pair_file, out, base_screens)
    restamp(out, arm)
    return rec


# ── the verdict ──────────────────────────────────────────────────────────────


def tag(r):
    return f"{r['split']} {r['dataset']}"


NEXT = {"ADOPT_FOR_FULL_RUN": "the arm fitted again on P1g-b's pools, then its full run, its own declared file",
        "GAIN_AND_LOSS": "the narrow head, A5w: the kind's own level and base weight only (kd_b and kd_base, two "
                         "parameters per model), the shared weights kept off as here, its own declared file",
        "NO_GAIN": "A5c, zck's coupled form on the lean pair: the offsets inside the output layer and base weight with "
                   "the shared weights trained jointly (zck as filed), 2wiki in domain beside as its known risk, its "
                   "own declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains else NEXT["NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A5's screen verdict, per arm, from its re-call (section 4); metaqa, webqsp and 2wiki beside, the largest move on a
    graph without typed relations (0 by construction, up to the card's training noise), musique's in-domain read of
    both arms side by side, and the next form each outcome names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A5" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A5's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        untyped = [abs(r["delta"]) for r in rows
                   if r.get("dataset") not in TYPED and isinstance(r.get("delta"), (int, float))]
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems,
                  "next_form": next_form(v, gains, losses),
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                             if r.get("dataset") in BESIDE},
                  "untyped_max_abs_delta": max(untyped) if untyped else None,
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)} for a in ARM_NAMES}
    rec = {"stage": "A5", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za5kind_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# A5's screen verdict: zrckpk **{per['zrckpk']['verdict']}**, zspkk **{per['zspkk']['verdict']}**", "",
          f"Rule: {RULE}.", ""]
    num = lambda x: f"{x:+.4f}" if isinstance(x, (int, float)) else "-"
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** ({WHAT[a]}) against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
                  + (", ".join(f"{k} {b['call']} ({num(b['delta'])})" for k, b in p["beside"].items()) or "none")
                  + f"; the largest move on a graph without typed relations {num(p['untyped_max_abs_delta'])} "
                    f"(the shared weights are the base's). Next: {p['next_form']}.")
    md += ["", f"Beside, deciding nothing: musique in-domain ({ck}): "
           + ", ".join(f"{a} {side[a]['call']} ({num(side[a]['delta'])})" for a in ARM_NAMES) + ".",
           "", "This verdict closes only this form (zck's kind head on typed graphs over zret's detached hidden units, "
               "at the top of the model, its gradient kept off the shared weights); docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za5kind verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A5", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def trunk(state):
    return {k: v for k, v in state.items() if k not in KD}


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: A5Top at the top and A5Stash just above zret in zrckp's and zspk's models; the base arms unchanged;
    #    the head's parameters are zck's offsets' shapes
    assert S.ARMS["zrckpk"] == (ZRCKPK, Z3.KTChainCarveZRC) and S.ARMS["zspkk"] == (ZSPKK, Z3.KTLinkCarveBase)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert ZRCKPK.__mro__[1:8] == (A5Top, Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch, A5Stash, S3.ZRet)
    assert ZSPKK.__mro__[1:9] == (A5Top, Z3.ZSPK, ZP.ZProp, Z3.KTMix, ZM.ZRM, RM.ChainMatch, A5Stash, S3.ZRet)
    assert KD == ZK.KD and ZK.RM is RM
    rng = np.random.default_rng(85)
    tmp = Path(tempfile.mkdtemp(prefix="za5_"))
    try:
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        cls_ch = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        c_m = cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_g = cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c_m.rows)[::-1][:9], np.arange(c_m.rows)[:7]].astype(np.int64)
        B = qs.size
        keep_ = torch.ones(B, len(blocks))
        keep_[3, 1] = 0.0
        # 2. each arm against its base: from the same seed, the base's state plus three zero parameters and no draw; at
        #    zero it is the base bit for bit, typed or not, eval or train; set, a typed batch is the base plus the head
        #    over zret's own h and base term, an untyped batch the base bit for bit
        for arm, cls, base, cv in (("zrckpk", ZRCKPK, Z3P.ZRCKP, c_m), ("zspkk", ZSPKK, Z3.ZSPK, c_g)):
            bt = cv.batch(qs, blocks)
            bu = ZK.Untyped(cv, "2wiki").batch(qs, blocks)
            assert RM.CH_KEY in bt[0] and RM.CH_KEY not in bu[0]
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            r0 = torch.get_rng_state()
            torch.manual_seed(0)
            m1_ = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r0)
            s0, s1 = m0.state_dict(), m1_.state_dict()
            assert sorted(set(s1) - set(s0)) == sorted(KD) and not LG.same_state(s0, trunk(s1))
            assert all(float(s1[k].abs().max()) == 0 for k in KD)
            with torch.no_grad():
                for m_ in (m0, m1_):
                    for p_ in m_.out.parameters():
                        p_.copy_(torch.linspace(-0.3, 0.3, p_.numel()).reshape(p_.shape))
                    m_.cm_gate.copy_(torch.tensor([0.7, -0.4]))
                    if arm == "zspkk":
                        m_.sp_w2.copy_(torch.randn(m_.sp_w2.shape, generator=torch.Generator().manual_seed(8)))
            for mode in ("eval", "train"):
                for f2, nq2, bz2, _ in (bt, bu):
                    getattr(m0, mode)()
                    getattr(m1_, mode)()
                    torch.manual_seed(3)
                    y0 = m0(f2, keep_, nq2, B, bz2)
                    torch.manual_seed(3)
                    y1 = m1_(f2, keep_, nq2, B, bz2)
                    assert torch.equal(y0, y1), (arm, mode, RM.CH_KEY in f2)
            with torch.no_grad():
                m1_.kd_w.copy_(torch.linspace(-0.5, 0.4, m1_.kd_w.numel()).reshape(m1_.kd_w.shape))
                m1_.kd_b.fill_(0.25)
                m1_.kd_base.fill_(-0.6)
            m0.eval()
            m1_.eval()
            f2, nq2, bz2, _ = bt
            y0 = m0(f2, keep_, nq2, B, bz2)
            y1 = m1_(f2, keep_, nq2, B, bz2)
            ref = S3.retrieved(f2)
            bz_ref = S3.seg_zscore_ref(f2["rank"][:, S3.I_RRF:S3.I_RRF + 1], nq2, B, ref).squeeze(1)
            assert torch.equal(m1_.a5_bz, bz_ref)
            want = y0 + (-0.6) * bz_ref + m1_.a5_h @ m1_.kd_w.detach().squeeze(0) + 0.25
            assert torch.allclose(y1, want, atol=1e-5) and float((y1 - y0).abs().max()) > 1e-3
            f2, nq2, bz2, _ = bu
            assert torch.equal(m1_(f2, keep_, nq2, B, bz2), m0(f2, keep_, nq2, B, bz2))
            # 3. the gradients under A5's loss, the head set: every shared weight's gradient is the base's on its own
            #    loss, bit for bit, on a typed and an untyped batch; the head takes a gradient on the typed batch only
            m0.train()
            m1_.train()
            for f2, nq2, bz2, gold in (bt, bu):
                typed = RM.CH_KEY in f2
                m0.zero_grad()
                m1_.zero_grad()
                torch.manual_seed(4)
                LG.listwiseD(m0(f2, keep_, nq2, B, bz2), gold, nq2, B).backward()
                with a5_loss():
                    torch.manual_seed(4)
                    LG.listwiseD(m1_(f2, keep_, nq2, B, bz2), gold, nq2, B).backward()
                assert not CUR
                p1 = dict(m1_.named_parameters())
                for n_, p0 in m0.named_parameters():
                    g0, g1 = p0.grad, p1[n_].grad
                    assert (g0 is None and g1 is None) or torch.equal(g0, g1), (arm, typed, n_)
                assert all((p1[k].grad is not None and float(p1[k].grad.abs().sum()) > 0) == typed for k in KD), \
                    (arm, typed, {k: p1[k].grad for k in KD})
            # without A5's loss the joint loss reaches the shared weights through the head's score
            m0.zero_grad()
            m1_.zero_grad()
            f2, nq2, bz2, gold = bt
            torch.manual_seed(4)
            LG.listwiseD(m0(f2, keep_, nq2, B, bz2), gold, nq2, B).backward()
            torch.manual_seed(4)
            LG.listwiseD(m1_(f2, keep_, nq2, B, bz2), gold, nq2, B).backward()
            CUR.clear()
            p1 = dict(m1_.named_parameters())
            assert any(p0.grad is not None and not torch.equal(p0.grad, p1[n_].grad)
                       for n_, p0 in m0.named_parameters())
        # 4. toy fits through lean_gpu's loop under A5's loss: every shared weight of every epoch's state and of the SWA
        #    state is the base fit's bit for bit; the head moves (the toy graph is typed); a repeat is bit-identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv in (("zrckpk", c_m), ("zspkk", c_g)):
            with S.patched(BASE_OF[arm]):
                bfit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            with S.patched(arm), a5_loss():
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                n1 = dict(LAST)
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
            assert dict(LAST) == n1 and n1["typed_forwards"] > 0, n1
            for a_, b_ in zip(fit["states"] + [fit["swa"]], bfit["states"] + [bfit["swa"]]):
                assert not LG.same_state(trunk(a_), b_) and sorted(set(a_) - set(b_)) == sorted(KD)
            assert float(fit["swa"]["kd_w"].abs().sum()) > 0 and float(fit["swa"]["kd_base"].abs().sum()) > 0
            assert not any(LG.same_state(a_, b_) for a_, b_ in zip(fit["states"] + [fit["swa"]],
                                                                   rep["states"] + [rep["swa"]]))
            assert not issubclass(S.ARMS[BASE_OF[arm]][0], A5Top)
            assert LG.listwiseD.__name__ == "listwiseD" and not CUR
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. the refusals; the fit names give the split
    assert split_of_fit("scr-a5-zspkk-hp") == ("zspkk", "L-hotpotqa") and split_of_fit("scr-a4-zspkd") == (None, None)
    good = ["train", "--arm", "zrckpk", "--name", FIT["zrckpk"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", FIT["zrckpk"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspkk", "--name", FIT["zrckpk"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspkk", "--name", FIT["zspkk"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    assert LG.listwiseD.__name__ == "listwiseD"
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a5-zspkk", {"arm": "zspkd", "a4": {}}),
                        ("scr-a5-zspkk-hp", {"arm": "zspkk", "a5": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 6. relz's pair and re-call for each arm against its base arm's card fits, stamped A5's; relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm in ARM_NAMES:
            fits = BASE_FITS[arm]
            null1 = SR.fake_null(T_ / arm / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(T_ / arm / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=BASE_OF[arm])) for sp in fits}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(T_ / arm / "z", FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(T_ / arm / "z", FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            pr = pair(arm, [za, zh], T_ / arm / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_OF[arm], pr["verdict"]
            pj = json.loads((T_ / arm / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "A5" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A5 ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A5's kind head) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 7. the verdict, per arm, with the next form of section 5: gains with a loss -> A5w, no gain -> A5c
        gw, mi = ("L-musique", "webqsp"), CLAIM1_READ
        rec = verdict({"zrckpk": fake_recall(T_ / "a.json", "zrckpk", {gw: "GAIN"}),
                       "zspkk": fake_recall(T_ / "b.json", "zspkk", {gw: "GAIN", ("L-hotpotqa", "metaqa"): "LOSS"})},
                      T_ / "v")
        assert rec["arms"]["zrckpk"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zrckpk"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["arms"]["zrckpk"]["untyped_max_abs_delta"] == 0.0
        assert rec["arms"]["zspkk"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspkk"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["musique_in_domain_beside"] == {"zrckpk": {"call": "WITHIN", "delta": 0.0},
                                                   "zspkk": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspkk"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckpk": fake_recall(T_ / "a.json", "zrckpk", {}),
                       "zspkk": fake_recall(T_ / "b.json", "zspkk", {gw: "GAIN"}, stage="A4")}, T_ / "v")
        assert rec["arms"]["zrckpk"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspkk"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckpk"]["next_form"] == NEXT["NO_GAIN"]
        assert rec["arms"]["zspkk"]["next_form"] == NEXT["INCOMPLETE"]
        rec = verdict({"zrckpk": T_ / "none.json", "zspkk": fake_recall(T_ / "b.json", "zrckpk", {gw: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zrckpk"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspkk"]["verdict"] == "INCOMPLETE"
    log(f"za5kind selftest: A5Top at the top and A5Stash above zret in both models; from the same seed each arm is its "
        f"base's state plus three zero parameters, its base bit for bit at zero (typed or not, eval or train); set, a "
        f"typed batch is the base plus the head over zret's own h and base term, an untyped one the base; under A5's "
        f"loss every shared weight's gradient is the base's bit for bit and only a typed batch moves the head (without "
        f"it, the shared weights move); toy fits keep every shared weight the base fit's bit for bit, the head moves, a "
        f"repeat is identical; the refusals; relz's pair and re-call per arm, stamped A5's; the verdict's outcomes and "
        f"next forms ({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="L-musique", choices=S2.SPLITS)
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "verdict":
        return 0 if all(v["verdict"] != "INCOMPLETE" for v in verdict()["arms"].values()) else 1
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=ARM_NAMES)
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        try:
            if k.cmd == "pair":
                pair(g.arm, [x for x in g.screens.split(",") if x], g.out)
            else:
                if len(null) != 4 or not g.pair:
                    gp.error("recall: --pair and the four --null files")
                recall(g.arm, null, g.pair, g.out)
        except SystemExit as e:
            log(f"za5kind {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za5kind: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
