"""SWP Part E (docs/SWP_S_TO_R_SWAP.md section 5f, amendment of 11 Nov): zswpe_t.py's train step with the start-encoder count of the build
trained with the slot EMPTY. It runs in the N-world only, like zswpe_t.py:

    python outputs/host_ops/nworld.py run --mode train -- python outputs/mp_unified/zswpe_tn.py train --name scr-ml1-zg6r-s0
    python outputs/host_ops/nworld.py run --mode train -- python outputs/mp_unified/zswpe_tn.py check      (read only: both splits' starts)
    python outputs/mp_unified/zswpe_tn.py selftest

Why. zg6r, and GNX under ISO's KEEP (through ml1start.LOAD), start from ML1's seed-0 zspk fit of their split. They load it with
zml1s.load_encoder, which takes the start only when its encoder (zg6rw.ENC_NAMES) holds zg6rw.ENC_ELEMENTS = 224,737 numbers: zrm's block
encoder at hidden 128 on the base build's blocks. Part E's zspk fits trained with the slot empty, and lean_gpu.dead_blocks drops every block
whose values do not vary over the training rows. The slot's blocks are constant there, so l1.weight is (128, 243), not (128, 309): 66 block
columns fewer, 216,289 numbers. Every other encoder tensor has the base shape (both zspk fits' p@swa read on the host, 11 Nov). zg6r and GNX
in the N-world build their l1 on the same training rows, and zg6r's copy_encoder checks every name and shape of the start against the model
it builds. Both Part E zg6r fits stopped at this count on 11 Nov 05:44-05:56 (rc 1 in 3 s; no fit trained).

What changes. For a zg6r or GNX fit only, each start load runs with zg6rw.ENC_ELEMENTS = 216,289 for the call (restored after, also on a
refusal), and the start it returns must have l1.weight (128, 243). Nothing else changes. zswpe_t.py, zswpt.py, zml1s.py, ml1start.py,
zg6rw.py and zg6r.py are not edited (the S- and N-world pin them). A zspk or zrckpn fit runs exactly as through zswpe_t.py.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zswpe_t as E  # noqa: E402  (first: zswpt's thread environment, then the N-world's checks; Part E's train step, not edited)

T = E.T
BASE_ELEMENTS = 224737                                         # zg6rw.ENC_ELEMENTS: the base build's blocks, l1.weight (128, 309)
NKF_ELEMENTS = 216289                                          # the slot-empty build: l1.weight (128, 243), every other tensor the same
L1_BASE, L1_NKF = (128, 309), (128, 243)
STARTS = ("zg6r", "gnx")                                       # the fits that load a zspk start (GNX: ISO's KEEP)
assert BASE_ELEMENTS - NKF_ELEMENTS == L1_BASE[0] * (L1_BASE[1] - L1_NKF[1]) == 8448


def nkf_load(load):
    """load (zml1s.load_encoder, or ml1start.LOAD, which is the same function) with the slot-empty count in force for the call; the start it
    returns checked: l1.weight (128, 243) and 216,289 numbers in all."""
    def run(split):
        import zg6rw as GW
        import zml1s as M1
        if GW.ENC_ELEMENTS != BASE_ELEMENTS:
            raise SystemExit(f"SWPE-TN: zg6rw.ENC_ELEMENTS is {GW.ENC_ELEMENTS}, not the base build's {BASE_ELEMENTS}")
        with M1.WQ.patch(GW, ENC_ELEMENTS=NKF_ELEMENTS):
            enc, info = load(split)
        l1 = tuple(enc["l1.weight"].shape) if enc is not None and "l1.weight" in enc else None
        if l1 != L1_NKF or info.get("elements") != NKF_ELEMENTS:
            raise SystemExit(f"SWPE-TN: {info.get('fit')}'s start has l1.weight {l1} and {info.get('elements')} numbers, "
                             f"not {L1_NKF} and {NKF_ELEMENTS} (the slot-empty build)")
        T.log(f"start {info.get('fit')} ({split}): {NKF_ELEMENTS} numbers, l1.weight {L1_NKF} (the slot-empty build; SWPE-TN)")
        return enc, info
    run.nkf = load
    return run


def install():
    """zml1s.load_encoder and ml1start.LOAD (GNX's start under ISO's KEEP) replaced by the slot-empty load. ml1start is imported first: at its
    import it takes zml1s.load_encoder as LOAD and asserts that it is zml1s's own."""
    import zml1s as M1
    import ml1start as MS
    if getattr(M1.load_encoder, "nkf", None) is not None or getattr(MS.LOAD, "nkf", None) is not None:
        raise SystemExit("SWPE-TN: the slot-empty start load is installed already")
    if MS.LOAD is not M1.load_encoder:
        raise SystemExit("SWPE-TN: ml1start.LOAD is not zml1s.load_encoder")
    M1.load_encoder = MS.LOAD = nkf_load(M1.load_encoder)
    return M1, MS


def check():
    """Read only, in the N-world in train mode: both splits' zspk starts through the slot-empty load, in zml1s's train contexts."""
    E.in_world()
    M1, _MS = install()
    for split in T.SPLITS:
        with M1.on_ml1(), M1.WQ.on_wq():
            enc, info = M1.load_encoder(split)
        print(f"SWPE-TN check {split}: {info['fit']} seed {info['seed']}, {info['elements']} numbers, l1.weight "
              f"{tuple(enc['l1.weight'].shape)}, models.pt sha256 {info['sha256'][:16]}", flush=True)
    print("SWPE-TN check: OK", flush=True)
    return 0


class _Toy:
    """A stand-in tensor for the self-test: a shape and its numel."""
    def __init__(self, *shape):
        self.shape = shape

    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n


def selftest():
    import zg6rw as GW
    import zml1s as M1
    import ml1start as MS
    assert GW.ENC_ELEMENTS == BASE_ELEMENTS and M1.load_encoder is MS.LOAD
    seen = []

    def stub(l1, elements, fail=False):
        def load(split):
            seen.append((split, GW.ENC_ELEMENTS))
            if fail:
                raise SystemExit("stub refusal")
            return {"l1.weight": _Toy(*l1)}, {"fit": "scr-ml1-zspk-s0", "elements": elements}
        return load

    enc, info = nkf_load(stub(L1_NKF, NKF_ELEMENTS))("WQ-musique")
    assert seen[-1] == ("WQ-musique", NKF_ELEMENTS) and GW.ENC_ELEMENTS == BASE_ELEMENTS and info["elements"] == NKF_ELEMENTS
    for bad in (stub(L1_BASE, NKF_ELEMENTS), stub(L1_NKF, BASE_ELEMENTS), stub(L1_BASE, BASE_ELEMENTS), stub(L1_NKF, NKF_ELEMENTS, fail=True)):
        try:
            nkf_load(bad)("WQ-hotpotqa")
            raise AssertionError("must refuse")
        except SystemExit:
            pass
        assert GW.ENC_ELEMENTS == BASE_ELEMENTS, "the count is restored after a refusal"
    orig = M1.load_encoder
    try:
        M1.load_encoder = MS.LOAD = orig
        install()
        assert M1.load_encoder is MS.LOAD and M1.load_encoder.nkf is orig
        try:
            install()
            raise AssertionError("must refuse a second install")
        except SystemExit:
            pass
    finally:
        M1.load_encoder = MS.LOAD = orig
    keep = dict(E.os.environ)
    try:
        for var in ("NWORLD", "SWORLD", "SWORLD_MODE"):
            E.os.environ.pop(var, None)
        for f in (check, lambda: main(["train", "--name", "scr-ml1-zg6r-s0"])):
            try:
                f()
                raise AssertionError("must refuse outside the N-world")
            except SystemExit:
                pass
        assert M1.load_encoder is orig and MS.LOAD is orig, "a refusal before the install leaves the loaders alone"
    finally:
        E.os.environ.clear()
        E.os.environ.update(keep)
    assert [T.fit_kind(n) in STARTS for n in T.FITS] == [False, False, True, True, True, True, False, False]
    E.selftest()                                                   # Part E's own checks (and Part C's fit table), unchanged
    print("zswpe_tn self-test: OK")
    return 0


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if a[:1] == ["selftest"]:
        return selftest()
    if a[:1] == ["check"]:
        return check()
    if len(a) == 3 and a[0] == "train" and a[1] == "--name" and T.fit_kind(a[2]) in STARTS:
        E.in_world()                                               # refuse outside the N-world before anything is replaced
        install()
    return E.main(a)


if __name__ == "__main__":
    raise SystemExit(main())
