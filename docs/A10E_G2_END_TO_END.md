# A10e: G2's hidden-state layers on the adopted GNN, end to end

Declared 19 October 2026, before any number. The next form A10's verdict named (zg2x NOT_ADOPTED with no GAIN;
outputs/a10/screen-verdict.json, fixed in docs/A10_G2_ON_ZSPX.md section 5). Under docs/SCREENS.md section 3b.
Code: `outputs/mp_unified/za10e.py` (the arm, the screen and the call). Filed 27 October:
NOT_ADOPTED with no GAIN and no LOSS; the next form is A10p (section 5).

## 1. Why this, now

**What A10 found.** A10 put G2's stack, unchanged, over zspx's full score (zg2x). After the seed-null re-call it had
0 GAIN and 0 LOSS (outputs/screen/scr-a10-zg2x-pair-recall.json). All twelve reads were WITHIN as filed, too. Three
reads moved with a 95% CI clear of zero, all inside the null's floor:

| split | read | zspx R@5 | zg2x R@5 | delta [95% CI] | floor | call |
| --- | --- | ---: | ---: | --- | ---: | --- |
| L-musique | metaqa (in domain) | 0.8030 | 0.8041 | +0.0012 [+0.0002, +0.0021] | 0.0075 | WITHIN |
| L-musique | 2wiki (in domain) | 0.9351 | 0.9294 | -0.0056 [-0.0071, -0.0041] | 0.0075 | WITHIN |
| L-hotpotqa | 2wiki (in domain) | 0.9312 | 0.9366 | +0.0054 [+0.0039, +0.0068] | 0.0075 | WITHIN |

The other nine reads' CIs held zero:
- webqsp zero-shot: +0.0050 [-0.0015, +0.0119] from L-musique (floor 0.0102), -0.0062 [-0.0144, +0.0022] from
  L-hotpotqa (floor 0.0233);
- musique zero-shot (L-musique): +0.0011 [-0.0042, +0.0063] (floor 0.0720);
- hotpotqa zero-shot (L-hotpotqa): -0.0014 [-0.0043, +0.0016] (floor 0.0200);
- musique in domain (L-hotpotqa, Claim 1's read): -0.0033 [-0.0082, +0.0016].

So the stack moved 2wiki, one way on each split, and nothing past a floor.

**Why that closes only its form.**
- In zg2x the stack read zspx's score through a detached z-score, as in G2. The stack learned on top of zspx's score,
  but zspx's scorer and steps never learned from how their score moves the stack.
- G4 found that this matters for a step. zspkg is zspk with its one step's inputs computed with the gradient kept. It
  had 1 GAIN and 0 LOSS: musique in domain on L-hotpotqa, 0.5754 to 0.5847, +0.0093 [+0.0042, +0.0141]
  (outputs/screen/scr-g4-zspkg-pair-recall.json).
- zspx carries that step and adds two more. Its two GAINs over zspk were both on L-hotpotqa: musique in domain,
  +0.0147 [+0.0095, +0.0198], and webqsp zero-shot, +0.0235 [+0.0133, +0.0327]
  (outputs/screen/scr-g5-zspx-pair-recall.json).
- The stack trained end to end is the form A10 named for no GAIN, and it has not been tried.

**The base is zspx,** decided against zspx's card fits (scr-g5-zspx, scr-g5-zspx-hp), as in A9 and A10.

## 2. The form

zg2xe is zg2x (outputs/mp_unified/za10g2x.py) with one change: the stack's z-score of zspx's score keeps the
gradient.

    s3  = zspx's score: zspk's scorer and KT head, its step with the gradient kept, G3's residual message, G4's
          second step
    z   = s3's z-score over the question's rows, clipped to +-8, with the gradient kept (zg2x: detached)
    out = s3 + 0.5 tanh(r / 0.5),   r = G2's stack reading z

**What changes.** z enters the stack in two places, and the loss now flows back through both into s3:
- the boundary [z, KT], fed at the input layer and at each of the three layers;
- the question's context weights, the softmax of z over the question's rows, which weight each layer's context
  vector.

From s3 the gradient reaches zspx's three step heads, its KT head and its scorer. zspx's own paths stay as they are:
its first step's inputs keep the gradient (zspkg's), and its second and third steps' inputs do not.

**What does not change.** Everything else is zg2x's, which is G2's stack over zspx:
- the forward: for the same weights, zg2xe's output is zg2x's bit for bit; only the training signal changes;
- the stack: 64-wide states from the encoder's hidden state (the output layer's input), three layers, a gated mean
  per link family and an attention over all edges, logits clipped at 8, readout bound 0.5;
- its draws: weights from its own generator (seed + 3501), the readout at zero at the start, edge-family dropout
  0.25 in training from another generator (seed + 3502);
- the parameters: G2's stack, 121,418 at hidden 128, nothing added over zg2x;
- zspx's settings, seeds, batches, KT input, carve, loss and loop.

**How it is built.** `g2_stack_e` is zg2.g2_stack line for line, with its one `s.detach()` read as `s`.
`ZG2XE.forward` is G2Mix.forward line for line, calling `g2_stack_e`. The self-test compares their syntax trees with
zg2's, so no other line can differ. No existing file changes.

**Why the gradient is safe.** lean_gpu's `seg_zscore8D` guards its square root with a double `where`, divides by a
standard deviation floored at 1e-6, and zeroes a question whose rows all score alike, so its backward stays finite.
zspx's first step already trains through it on the card (zspkg's inputs). z is clipped at 8, so the softmax cannot
overflow.

**Bit-for-bit anchors.** From one seed, zg2xe's weights are zg2x's. Its forward is zg2x's. At the start (readout
zero) every gradient is zg2x's, so a fit's first step is zg2x's. Without links it is zspx's scorer.

**GNN only.** The MLP takes no message.

## 3. The arm

| arm | the model | the carve |
| --- | --- | --- |
| zg2xe | zg2x (zspx, then G2's stack over its score), with the stack's z-score of the score keeping the gradient | zspx's (G3's: zspk's with c and s beside) |

## 4. The fits and the call

- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a10e-zg2xe`, `-hp`), on P1c2's roots with P1c3's
  KT input. Card fit settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128 (zspx's).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against zspx's card fit of the same split (scr-g5-zspx, scr-g5-zspx-hp), which decides.
  Step 1's fit is reported beside.
- **The re-call's base screens:** A9's, zspx's card fits compared with step 1's of the same split
  (outputs/a9/base-zspx-<split>.json).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

- **ADOPT_FOR_FULL_RUN:** zg2xe's re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A10e's re-call of zg2xe, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp and metaqa, both splits (the KB graphs: webqsp is the zero-shot KB);
- musique in domain (L-hotpotqa): Claim 1's read;
- hit@1 and FC@5 per read;
- zg2x's delta on each read (A10's re-call) and zg2xe's delta less it: what the gradient through the z-score adds
  to A10's stack. Both are compares against the same zspx fits; their difference is not a test.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | zg2xe fitted again on P1g-b's pools, then its full run beside zspx's (its own declared file) |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A10eb, the back-off form: zg2xe with the stack's readout bounded at 0.2 instead of 0.5 (the MP-unified track's 0.2 tanh), so the layers can move each score less (one declared file) |
| NOT_ADOPTED, with no GAIN | A10p, the stack before the steps: zg2xe's stack (its z-score with the gradient kept) reading zspx's pre-step score (zspk's scorer and KT head), its bounded readout added there, so zspx's three steps act on the stack's correction (one declared file) |
| INCOMPLETE | the missing re-call is run; nothing else changes |

Why A10p and not a close on no GAIN: G2, A10 and A10e all put the stack after the steps, where nothing propagates
its correction. G4 found that depth pays when the step acts on the score, so a stack whose output the steps carry is
a form not yet tried. The stack cannot reach the MLP as an input (it passes messages and reads the score's z-score),
so its idea can only close on the GNN's side.

No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-test, passed before this commit.** `python outputs/mp_unified/za10e.py --selftest` (8 s) checks, on a toy
carve with G3's residuals and the KT input:
- **The arm.** zg2xe is zg2x's class with its own forward, on G3's carve (class order ZG2XE, ZG2X, G2Mix, ZSPX).
  zg2x's, zspx's, zg2's and zspk's arms are unchanged. A9's base screens are the re-call's.
- **The code.** g2_stack_e's syntax tree is zg2.g2_stack's with its one `s.detach()` read as `s`, and no other line
  changed. ZG2XE.forward's is G2Mix.forward's, calling g2_stack_e (its `super()` spelled out).
- **The weights.** From one seed, zg2xe's state is zg2x's bit for bit, and the global generator ends where zg2x
  leaves it. The stack holds G2's parameter count.
- **The forward,** with zspx's three step heads moved so that all its steps act:
  - at the start, zg2xe is zg2x and zspx bit for bit;
  - with the readout moved, it is zg2x bit for bit and moves off zspx by at most 0.5;
  - without links, it is zspx's scorer;
  - in training, with the same global seed and the stack's own generator, it is zg2x bit for bit.
- **The gradient.**
  - At the start (readout zero), every parameter's gradient is zg2x's bit for bit.
  - With the readout moved, the stack's own gradients are still zg2x's bit for bit.
  - Each of zspx's parameters takes zg2x's gradient plus the gradient through the z-score, within 1e-6. That second
    part is built by hand: a detached copy of the score enters the stack only as its z-score, and its gradient is
    carried back through zspx's score.
  - That extra gradient reaches zspx's scorer and all three step heads.
- **Toy fits** through lean_gpu's loop. A repeat is bit for bit. The readout learns. zspx's weights move off zg2x's own
  toy fit (from the second step, once the readout has moved). Without links the fit is zspx's bit for bit on every
  shared key, the readout staying at zero.
- **The refusals.** train refuses zspx's, zg2x's and zg2's arms, another split, the wrong fit name (A10's included),
  another cache root and the CPU. read refuses zspx's fit, an unstamped zg2xe fit, zg2x's A10 fit and a zg2x fit
  stamped A10e's.
- **The rest.** relz's pair and re-call stamped A10e's, with relz restored after. The verdict's outcomes and next
  forms, with INCOMPLETE for a missing re-call, A10's re-call, or fewer than twelve reads. zg2x's deltas beside.

**Host items** (`a10e-*`, 9 in all), each starting when the GPU frees:
- 2 trains and 2 reads, sized as A10's: `--frac 0.48` (share 0.50). G2's stack takes most of the memory (G2's
  measured `--frac 0.45`, with A10's margin for zspx's steps), and the gradient through the z-score keeps only a few
  more per-row vectors. Memory: trains 6 GB (-hp 15 GB), reads 6 GB;
- the trains wait on the rebuild of G3's residual arrays for their split's fit carves (`n19-res-*`, by A9's lists),
  the reads on the six s1eval carves';
- 2 compares, the pair, the re-call (after A9's base compares, `a9-base-zspx` and `-hp`) and the verdict.

No build: zg2xe reads zspx's carves.

The commands (the -hp fit is the same with `--split L-hotpotqa` and `-hp` names):

    python outputs/mp_unified/za10e.py train --split L-musique --name scr-a10e-zg2xe --arm zg2xe --device cuda --host \
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za10e.py read --name scr-a10e-zg2xe --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za10e.py compare --new outputs/screen/fits/scr-a10e-zg2xe \
        --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a10e-zg2xe
    python outputs/mp_unified/za10e.py pair --screens outputs/screen/scr-a10e-zg2xe.json,outputs/screen/scr-a10e-zg2xe-hp.json \
        --out outputs/screen/scr-a10e-zg2xe-pair
    python outputs/mp_unified/za10e.py recall --null outputs/screen/scr-null-s1.json,outputs/screen/scr-null-s2.json,\
    outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json --pair outputs/screen/scr-a10e-zg2xe-pair.json \
        --out outputs/screen/scr-a10e-zg2xe-pair-recall
    python outputs/mp_unified/za10e.py verdict

The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer, and no new graph.

## 7. Results

Filed 27 October 2026 from the host's records (`outputs/a10e/screen-verdict.json`, written 2026-10-20T07:17:32Z; the
re-call `outputs/screen/scr-a10e-zg2xe-pair-recall.json`, host paths made relative). **NOT_ADOPTED, with no GAIN and no
LOSS:** with G2's stack trained end to end (its z-score of the score keeping the gradient), zspx moves no read past its
floor, and the one filed LOSS, musique zero-shot on L-musique, is inside its floor under the re-call.

- **zg2xe: 0 GAIN, 0 LOSS.** musique zero-shot on L-musique falls -0.0169 [-0.0225, -0.0114], a LOSS by its pair's
  interval that is WITHIN under its 0.0720 floor, with FC@5 -0.0186 and hit@1 -0.0410. Four more intervals exclude 0,
  all inside their floors: on L-musique, squad +0.0024 [+0.0005, +0.0042] and 2wiki -0.0054 [-0.0068, -0.0039]; on
  L-hotpotqa, metaqa -0.0012 [-0.0020, -0.0005] and 2wiki +0.0034 [+0.0020, +0.0049].
- **Beside: the gradient through the z-score turned musique zero-shot on L-musique down.** With the z-score detached
  (A10's zg2x), that read moved +0.0011; end to end it moves -0.0169. The 2wiki moves are A10's again: -0.0054 against
  -0.0056 on L-musique, +0.0034 against +0.0054 on L-hotpotqa.

Not adopted. This closes only this form: G2's stack after zspx's three steps, trained end to end (docs/SCREENS.md
section 3b). The next form, fixed in section 5 and recorded in the verdict, is A10p, the stack before the steps: zg2xe's
stack (its z-score with the gradient kept) reading zspx's pre-step score (zspk's scorer and KT head), its bounded
readout added there, so zspx's three steps act on the stack's correction (one declared file).

The floors are the seed null of 7 October (docs/SCREENS.md section 2: `outputs/screen/scr-null-s1.json`,
`scr-null-s2.json` and their `-hp` files), not N2's fresh floors (docs/N2_FRESH_SEED_NULL.md).

Records, filed with this section: the verdict `outputs/a10e/screen-verdict.json` and `.md`; for the arm `zg2xe`, the
compares `outputs/screen/scr-a10e-zg2xe.json` and `-hp.json`, the pair `-pair.json` and its re-call `-pair-recall.json`,
each with its `.md`. The host paths were made relative in each re-call's two `rel` entries and nowhere else, so the
`recall_sha256` each verdict holds is the host file's, from before that change. The re-calls' nulls also read the base
screens `outputs/a9/base-zspx-L-musique.json` and `outputs/a9/base-zspx-L-hotpotqa.json`, fetched from the host with
this filing (A9's own filing had left them there).

The twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's, zspx's card fit of the split; a
call is shown with its filed call where the re-call changed it):

| split | dataset | read | zspx R@5 | zg2xe: delta [95% CI], call |
| --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8030 | +0.0001 [-0.0007, +0.0009], WITHIN |
| L-musique | squad | in-domain | 0.9092 | +0.0024 [+0.0005, +0.0042], WITHIN |
| L-musique | musique | zero-shot | 0.5073 | -0.0169 [-0.0225, -0.0114], WITHIN (filed LOSS) |
| L-musique | hotpotqa | in-domain | 0.9074 | +0.0011 [-0.0010, +0.0031], WITHIN |
| L-musique | 2wiki | in-domain | 0.9351 | -0.0054 [-0.0068, -0.0039], WITHIN |
| L-musique | webqsp | zero-shot | 0.3204 | -0.0037 [-0.0116, +0.0040], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8039 | -0.0012 [-0.0020, -0.0005], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9129 | +0.0008 [-0.0010, +0.0028], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5901 | -0.0041 [-0.0088, +0.0008], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8547 | -0.0010 [-0.0035, +0.0016], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9312 | +0.0034 [+0.0020, +0.0049], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2914 | -0.0080 [-0.0158, +0.0002], WITHIN |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a10e-zg2xe.json` and `-hp.json`); and A10's R@5 deltas with the stack's z-score detached (zg2x, its
own re-call `outputs/screen/scr-a10-zg2x-pair-recall.json`), on the same read (a call is shown where it is not WITHIN):

| split | dataset | zg2xe FC@5 | zg2xe hit@1 | zg2x (A10) R@5 |
| --- | --- | ---: | ---: | ---: |
| L-musique | metaqa | +0.0000 | +0.0009 | +0.0012 |
| L-musique | squad | +0.0024 | +0.0009 | +0.0019 |
| L-musique | musique | -0.0186 | -0.0410 | +0.0011 |
| L-musique | hotpotqa | +0.0030 | -0.0062 | -0.0018 |
| L-musique | 2wiki | -0.0147 | +0.0015 | -0.0056 |
| L-musique | webqsp | -0.0013 | -0.0013 | +0.0050 |
| L-hotpotqa | metaqa | -0.0024 | -0.0056 | +0.0001 |
| L-hotpotqa | squad | +0.0008 | -0.0035 | +0.0013 |
| L-hotpotqa | musique | -0.0091 | -0.0062 | -0.0033 |
| L-hotpotqa | hotpotqa | -0.0030 | +0.0047 | -0.0014 |
| L-hotpotqa | 2wiki | +0.0082 | -0.0013 | +0.0054 |
| L-hotpotqa | webqsp | -0.0027 | -0.0033 | -0.0062 |
