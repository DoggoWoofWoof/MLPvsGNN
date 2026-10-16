# E1: the lean pair's chain head trained by EM over latent relation chains

Declared 16 October 2026, before any number of this stage. Code: `outputs/mp_unified/ze1em.py`. Numbers here are
development numbers. Status: declared.

## 1. Why this, now

The user (15 October): "what about the EM training for better representations and better learning for
transformations". EM was used only in side tracks: S6's chain scorer and the type-factorized track. Both treated the
path to the gold as a latent variable. The lean models train with a listwise loss on gold labels only, with no latent
path (docs/PROGRAM_2026_10_15.md, section 1.6).

**What changed since the program was written.** The program put E1 on G2's gates ("the E-step infers which edges
carried each gold, the M-step trains the gates on those paths"). G2 was filed NOT_ADOPTED (f048f03), and zspk has no
edge gates: its propagation step averages the neighbours' z-scores with no weight per edge. The one part of both models
that weighs paths is **rmatch's chain head**, which both zrckp and zspk carry, so E1 goes there.

**What the chain head does today.** Each chain entry is a question, a pool row, a typed chain c (up to 3 relation steps,
each with a direction), a seed bucket and the walk's mass m_c(row) from the question's seeds along c. The head weighs a
chain by w(c) = exp(lw(c)), where lw(c) sums three things:
- each step's match to the question (`step_logp`: the question's z-scored cosine with the relation's name, a low-rank
  term on the centred relation table, the direction and a bias);
- the chain length's prior;
- the bucket's weight.

A row's two chain features are log(1 + Σ_c w(c) m_c(row) / TAU) and log(1 + Σ_c w(c)). Two gates add them to the
score, and both gates start at zero.

The head learns only through the listwise loss on the rows, and only as far as the gates let the loss through. Nothing
asks the chain weights themselves which chains lead to the golds.

## 2. The term

Per question q with chain entries:

| quantity | definition |
| --- | --- |
| P(c \| q) | softmax over the question's chains of lw(c), each chain one distinct (typed steps, bucket) among its entries |
| π_c(G) | the mass of c's kept entries on gold rows over the mass of all of c's kept entries (0 when the latter is 0) |
| L_path(q) | −log Σ_c P(c \| q) π_c(G_q) |

- **Which questions:** only those with at least one gold that a chain with mass reaches. Every other question takes no
  term.
- **The loss:** lean_gpu's listwise loss + **LAM = 1.0** × the batch mean of L_path over those questions. LAM is fixed
  here, before any number. No other value is screened.
- **Why it is EM:** the gradient of L_path in lw(c) is −(P(c | q, G) − P(c | q)). The E-step's posterior (which chain
  explains the golds) minus the prior (the head's current weights) is exactly what the M-step follows. The chain head's
  match parameters, length prior and bucket weights learn from it. The gates and the rest of each model learn from the
  listwise loss, as today.
- **What is unchanged:**
  - The chains, walks and masses are the build's (P1c2's roots), and π reads no parameter.
  - The forward is rmatch's, bit for bit: the term reads the chain weights the forward has already computed.
  - Reads and serving are unchanged. Nothing is added at read time, so latency is unchanged.
  - Both models, their parameters, settings, seeds and batches are unchanged.
  - P1c2's pools and caches, P1c3's KT input and zspk's propagation edges are unchanged.
- **Where it acts:** only chain entries carry the term, and in training only metaqa's fit carve has them. webqsp never
  trains. The passage graphs have no entries, so their batches train exactly as today.
- **The MLP rule holds.** zrckpe scores each row from inputs computed before any row is scored. Only its training loss
  changes.

## 3. The fits and the call

| arm | model and carve | its base |
| --- | --- | --- |
| zrckpe | zrckp (zrc + the pointwise KT head, no message passing) | `scr-p1c3p-zrckp`, `-hp` |
| zspke | zspk (zsp's propagation head + the KT head) | `scr-p1c3-zspk`, `-hp` |

- **Fits:** one card fit per arm on L-musique and L-hotpotqa (`scr-e1-zrckpe`, `-hp`; `scr-e1-zspke`, `-hp`). Both
  splits train metaqa. Each fit is read on the six s1eval carves at p@swa, on P1c2's roots and P1c3's KT input, as the
  bases were.
- **Compares:** each fit against its base arm's card fit of the same split, which decides. Step 1's fit is reported
  beside. The re-call's base screens are L1's (`outputs/l1/base-<arm>-<split>.json`): the same base fits compared with
  step 1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.
- **A train item stops** if no training batch carried the term.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not E1's re-call of that arm, or does not hold the twelve reads.

The two arms are decided separately. Under the user's rule of 13 October, no model change is adopted on today's pools:
an arm that passes is fitted again on P1g2's chosen pools before its full run (six splits under grade-nullx, its own
file).

**Reported beside, deciding nothing:**
- webqsp (zero-shot in both splits, the typed graph the term never sees) and metaqa (in-domain, the carve it trains
  on), per split, with their calls and differences;
- each fit's term: L_path's mean per epoch, the question-batches it covered, and the listwise loss beside it
  (`screen.json`, key `e1`);
- hit@1 and FC@5 per read, as the compares give them.

## 4. How it runs

- **Self-test** (`python outputs/mp_unified/ze1em.py --selftest`, passes before this commit):
  - the arms are the lean pair's classes. The term is installed only inside its context and restored after, even on an
    error;
  - on rmatch's toy typed carve, with every parameter drawn at random, the chain features are rmatch's bit for bit,
    in eval and in training. lw is kept only in training with gradients on. A forward is the plain forward bit for
    bit, and an untyped batch keeps nothing;
  - L_path matches a float64 reference built chain by chain. Its gradient in each chain's lw is EM's −(posterior −
    prior) to 1e-9. A batch with no reached gold takes no term. The loss is the listwise loss plus LAM × the batch
    mean. A second forward before the loss is refused;
  - toy fits through lean_gpu's loop, on the typed carve plus an untyped one:
    - at LAM 0, every epoch's state and the SWA state equal the plain loop's bit for bit;
    - at LAM 1, every state moves, a repeat is identical, and L_path falls (1.03 to 0.45 over three epochs, against
      1.03 to 0.98 at LAM 0). The term runs only on the typed carve's batches and is counted per epoch;
  - train refuses another arm, split, fit name, cache root or device. read refuses a fit that is not E1's;
  - relz's pair and re-call against each base arm's card fits, stamped E1's; the verdict's three outcomes per arm.
- **Host items** (`e1-*`, 17 in all; L1's items with zl1sep's command and arms swapped for E1's):
  - on the card, as L1's fits ran (`--frac 0.30`, gpu share 0.32): four trains, then four reads;
  - on the CPU: four compares, two pairs, two re-calls and the verdict.
- **Disk:** four fit folders of a few MB each. No new chain, cache or pool.
- **ETA:** about 2 hours on the card after L1's, L3's and T1's items. The term adds one sort of each metaqa batch's
  entries on the CPU and a few small device ops per step.
- The encoder stays frozen. webqsp never trains. Test splits are never read.

## 5. Results

Not run yet.
