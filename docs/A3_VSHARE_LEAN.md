# A3: the share augmentation of zret's reference, in both models' training

Declared 17 October 2026, before any number. The third reopening of docs/VERDICT_AUDIT_2026_10_17.md (section 2,
row 3), under docs/SCREENS.md section 3b. Code: `outputs/mp_unified/za3vshare.py`. Not run yet.

## 1. Why this, now

vshare (8 October, `lean_screen7.py`) reweighted a training pool's statistics. Its full run was not adopted on one
in-domain read:

| | read | R@5 against step 1 |
| --- | --- | --- |
| GAIN | musique zero-shot (L-musique) | +0.0891 [+0.079, +0.099] |
| GAIN | metaqa zero-shot (L-metaqa) | +0.0180 |
| GAIN | squad zero-shot (L-squad) | +0.0120 |
| LOSS | metaqa in domain (L-squad's fit) | −0.0094, against a floor of 0.0075; seeds 1 and 2 moved the same read −0.0037 and −0.0010 |

It ran on step 1's model and caches. It was never tried on zret, the reference both of today's models use, and it
never reached the GNN. Under section 3b a verdict closes only the form it tested, so the idea is still open.

## 2. Why the literal port does nothing, and the form

**vshare's form.** Every z-score was taken over the whole pool, with row weights: retrieval's ranked rows weigh 1,
the other rows b = r(1 − t) / (t u), where r is the number of ranked rows and u the number of others. The ranked rows
then hold a share t of the pool's weight. The rule was two-sided. A pool whose own ranked share is at least 0.375
(the small passage pools) was thinned to t in [0.08, 0.20]. A pool below it (the big pools) was enriched to t in
[0.55, 0.95]. The aim was that a pool's share would no longer tell the model which kind of graph it came from.

**zret.** Every z-score both of today's models take (each block's, SEMB's, and rrf's base term) is against the
pool's retrieved rows only (rrf > 0; `lean_screen3.seg_zscore_ref`). Multiplying that 0/1 reference by vshare's
weights leaves the retrieved rows at 1 and the other rows at 0, so the z-scores do not change. **The literal port
is the identity on zret.** A screen of it would only be a seed-null read of the base. The self-test checks this.

**What zret already did to vshare's channel.** zret's reference is the retrieved rows on every kind of pool. A
pool's share therefore no longer enters its statistics at all, so the kind-naming that vshare removed is already
gone. What remains of vshare under zret is the augmentation itself: training statistics perturbed around the
statistics the model reads.

**A3's form.** It moves the share into zret's reference.

- **When:** in a training forward only, for each question that has at least two retrieved rows and at least one
  other row, with probability 0.5.
- **The draw:** t ~ U[0.55, 0.95].
- **The reference:** the retrieved rows weigh 1 and the other rows weigh r(1 − t) / (t u), so the retrieved rows
  hold a share t of the reference.
- **What changes:** every z-score zret takes is taken against this reference.
- **Reads** are never reweighted: t = 1, zret's reference.

The range is vshare's RICH range, the side on which retrieval stays the majority of the reference, so the training
statistics stay near the read's. The THIN range is the named next form if this one finds no gain (section 5).

- **Why at least two retrieved rows:** with one, zret already takes the whole pool's z, and the weighted
  reference would weigh 1/t < 2, which falls back the same way.
- **Pools left unchanged:** a pool with no other row (squad's) is never reweighted.

**Worked example.** These are the same six rows as A2's example: one column, dense_cos, with v1 to v4 retrieved,
and t = 0.75, so b = 4 × 0.25 / (0.75 × 2) = 0.667.

| row | retrieved | dense_cos | weight | z (zret's reference) | z (A3's reference, t = 0.75) |
| --- | --- | ---: | ---: | ---: | ---: |
| v1 | yes | 0.81 | 1 | 1.11 | 1.17 |
| v2 | yes | 0.78 | 1 | 0.83 | 1.00 |
| v3 | yes | 0.62 | 1 | −0.65 | 0.08 |
| v4 | yes | 0.55 | 1 | −1.29 | −0.32 |
| v5 | no | 0.40 | 0.667 | −2.68 | −1.18 |
| v6 | no | 0.31 | 0.667 | −3.51 | −1.70 |

zret's reference has mean 0.690 and sd 0.108. A3's has mean 0.606 and sd 0.174, with the retrieved rows holding
0.75 of its weight.

**Nothing else changes.** No row is added or removed, the raw inputs and the listwise softmax are the pool's own,
and there is no new parameter. The draws come from the model's own generator (seed + 5801), which is made with the
model, so each fit restarts it. The global stream, dropout's masks included, is the base's.

**The MLP rule.** A3 reads no score and normalises nothing over the question's scores. It changes, in training only,
the reference of within-pool statistics of input columns. The rule of 8 and 14 October holds.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckpv | zrckp's model (`zp1c3p.ZRCKP`) with A3Mix just above zret's forward (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A3Mix, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) |
| zspkv | zspk's model (`zp1c3.ZSPK`) with A3Mix in the same place (MRO: ZSPK, ZProp, KTMix, ZRM, ChainMatch, A3Mix, ZRet); zsp's propagation reads the score A3 changes | zspk's (`zp1c3.KTLinkCarveBase`) |

Both models get the same augmentation in the same place. zprop's and KTMix's own z-scores of the score are
whole-pool statistics of the score, not zret's, so A3 leaves them as they are. No build is needed.

## 4. The fits and the call

- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a3-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input. These use the card fits' settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128.
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split, which decides:
  - zrckp: scr-p1c3p-zrckp and -hp;
  - zspk: scr-p1c3-zspk and -hp.

  Step 1's fit is reported beside. The re-call's base screens are L1's (`outputs/l1/base-<base arm>-<split>.json`).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm. GAIN,
  LOSS and WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A3's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- metaqa, both splits and arms: vshare's one LOSS was metaqa in domain;
- musique, both splits and arms: musique zero-shot (L-musique) was vshare's largest gain, and musique in domain
  (L-hotpotqa) is Claim 1's read;
- squad and webqsp, both zero-shot gains or reads of vshare's;
- hit@1 and FC@5 per read, as the compares give them;
- per fit, the share of training questions reweighted and the mean t (stamped in each fit's screen.json).

**Serving cost: none.** Reads are zret's forward, unchanged, so both models serve exactly as their bases do. In
training the extra cost is a few segment sums per batch.

## 5. What each outcome leads to (fixed now)

Per arm, the next form, each in its own file declared before its numbers:

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run |
| NOT_ADOPTED, with at least one GAIN and a LOSS | the back-off form, A3s: the same augmentation with the reference's retrieved share t as an input column (1 at every read). The model then learns what the reweighting changes, and it reads the unweighted reference as a case of its own |
| NOT_ADOPTED, with no GAIN | A3t, vshare's literal range: t in [0.08, 0.20], a reference made mostly of the pool's other rows, as step 1's statistics were on the big pools, on both models |
| INCOMPLETE | the missing re-call is run; nothing else changes |

- **If both arms pass**, the augmentation is the lean pair's training, in both models.
- **If only one passes**, its full run is declared. The next file decides whether F1's one-input-set rule needs the
  other model to train the same way.
- **What does not happen.** No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za3vshare.py --selftest` passes before this commit (3.9 s). It checks:
- the arms' MROs, with A3Mix just above zret's forward in both models, and the base arms unchanged. vshare's
  settings are read from `lean_screen7`, which this file never runs;
- the weights on a hand-made batch of five pools:
  - an applied pool's retrieved rows hold exactly t of the weight;
  - pools not drawn, pools with one retrieved row, and pools with no other row keep zret's reference;
  - z-scores against the weights match a float64 reference by hand, fallbacks included;
  - with nothing applied the weights are zret's 0/1 reference bit for bit, which is why the literal port is the
    identity;
- each arm against its base:
  - in eval mode bit for bit, and a read draws nothing from A3's generator;
  - in training at P_APPLY 0 bit for bit: scores, every gradient, and the global stream;
  - applied, the scores move on the eligible questions only and equal zret's forward on A3's reference, for zspkv
    through its propagation;
- toy fits through lean_gpu's loop repeat bit for bit, equal the base arm's fit at P_APPLY 0, and differ with A3;
- train refuses another arm, split, fit name, cache root or device; read refuses a fit that is not A3's or not its
  split's;
- relz's pair and re-call per arm, against its base's card fits and stamped A3's, with relz restored after;
- the verdict's outcomes and next forms.

**Host items** (17, `a3-*`), placed just after A2's verdict:
- four trains (`--frac 0.32`, GPU share 0.34);
- four reads;
- four compares, two pairs, two re-calls, and the verdict.

A2's compares, pairs, re-calls and verdict move up beside A2's reads in the same commit, so that no light item waits
behind P1g-b's looks.

Memory requests are A2's measured-to-date sizes:
- trains 4.5 GB, with zspkv-hp at 7 GB;
- reads 6 GB;
- compares 4 GB;
- pairs, re-calls and the verdict 2 GB.
