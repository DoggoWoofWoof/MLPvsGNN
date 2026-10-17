# G5: the three passing GNN changes in one model

Declared 17 October 2026, after G3's and G4's verdicts and before any number of G5. Code:
`outputs/mp_unified/zg5mix.py`. One arm, the GNN's. The MLP takes no message (the user's rule of 8 October).

## 1. Why this, now

Three changes to zspk's message passing each passed their screen against zspk's card fits, each with one GAIN and no
LOSS among twelve reads:

| change | screen | its GAIN |
| --- | --- | --- |
| G3's bridge-residual message (zspr) | docs/G3_BRIDGE_RESIDUAL.md | L-hotpotqa's musique read |
| G4's end-to-end step (zspkg) | docs/G4_DEEPER_PROPAGATION.md | L-hotpotqa's musique read |
| G4's second step (zspk2) | docs/G4_DEEPER_PROPAGATION.md | L-hotpotqa's musique read |

They change different things: what a message says (G3), whether the scorer learns from its neighbours' scores (zspkg),
and how far a score travels (zspk2). G4's section 3 says that if both of its arms pass, they are combined in the refit's
file. G5 combines all three now, on P1c2's pools, while P1g-b's pools are still being built and the card would
otherwise sit idle. The question is whether the three add up, or whether they find the same golds (all three gain on the
same read).

## 2. The arm

    s0 = zspk's score before its propagation (zrc's scorer and the KT head)
    s1 = s0 + h_sp(prop_inputs_grad(s0))        zspk's step, its inputs with the gradient kept (zspkg)
    s2 = s1 + h_rs(res_inputs(s1, c, s))        G3's residual message over the stepped score (zspr), no gradient
    s3 = s2 + h_p2(prop_inputs(s2))             G4's second step over everything before it (zspk2), no gradient

- Each piece is exactly as it was screened: zprop's head (seed + 2301), G3's (seed + 3401) and G4's (seed + 4401),
  385 parameters each. Each head's last layer starts at zero, so zspx starts as zspk bit for bit.
- The carve is G3's: zspk's with the per-row cosine c and per-edge cosine s beside, from G3's builds on P1c2's parts
  (on the host).
- Everything else is zspk's, unchanged: the settings, seeds, batches, KT input, loss and loop.

| arm | model | carve | its base |
| --- | --- | --- | --- |
| zspx | `ZSPX` (zspr's model plus zspk2's head, zspk's step end to end) | `ResKTLinkCarveBase` (G3's) | `scr-p1c3-zspk`, `-hp` |

## 3. The fits and the call

- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-g5-zspx`, `-hp`), on P1c2's roots, P1c3's KT input
  and G3's builds.
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against zspk's card fit of the same split, which decides. Step 1's fit is reported beside.
  The re-call's base screens are L1's (`outputs/l1/base-zspk-<split>.json`).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

- **ADOPT_FOR_FULL_RUN:** zspx's re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not G5's re-call of zspx, or does not hold the twelve reads.

**What follows.** No model change is adopted on today's pools (the user's rule of 13 October). The GNN that goes on
to P1g-b's pools is zspx if it passes. If it does not, each single change that passed goes on to P1g-b's pools on its
own. Either way the refit is its own file, declared before its numbers.

**Reported beside, deciding nothing:**
- webqsp (zero-shot in both splits) and metaqa (in-domain), with their calls and differences;
- zspx's R@5 difference against zspk per read beside each single change's, from their own re-calls;
- hit@1 and FC@5 per read, as the compares give them.

## 4. How it runs

- **Self-test** (`python outputs/mp_unified/zg5mix.py --selftest`, passes before this commit):
  - the arm is zspr's model plus zspk2's head, on G3's carve. zspk's, zspr's, zspk2's and zspkg's arms are unchanged;
  - zspx starts as zspk bit for bit, and its weights are zspr's and zspk2's, each drawn as in its own arm;
  - with only G3's head moved, zspx is zspr bit for bit; with only the second head moved, it is zspk2 bit for bit;
    with both, it adds the second head over zspr's score. Without c and s it is zspk2's chain; without links it is
    zspk's scorer;
  - the scorer's gradient runs through zspk's step as zspkg's does (equal to zspkg's while G3's and the second head
    are at zero, and different from zspr's);
  - toy fits through lean_gpu's loop: a repeat is bit-identical, both new heads learn, and without links the fit is
    zspk's bit for bit on every shared key, with the new heads at their start;
  - train refuses another arm, split, fit name, cache root or device. read refuses a fit that is not G5's;
  - relz's pair and re-call stamped G5's; the verdict's outcomes, with the single changes' calls beside.
- **Host items** (`g5-*`, 9 in all, after G4's verdict and ahead of P1g-b's fits):
  - on the card: two trains, then two reads (`--frac 0.40`, gpu share 0.42, both at once);
  - on the CPU: two compares, the pair, the re-call and the verdict.
- **Disk:** about 0.05 GB.
- **ETA:** G3's and G4's fits took 47 to 54 minutes each on the card, and their reads about 5 minutes. Both fits at
  once: about 1 hour 15 minutes, so the verdict is due about 10:45 on 17 October. P1g-b's fits are not ready before
  then (metaqa's pools are being rebuilt after a file-lock failure).
- The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer.

## 5. Results

Not run yet.
