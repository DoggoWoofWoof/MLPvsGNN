# RWT: question weights in zrckpn's loss, on PKO's pools, on the host's CPU

Declared 3 November 2026, before any RWT fit. This file fixes the forms, the fits, the rule and what follows. The stage
module (`outputs/mp_unified/zrwt.py`) and its self-test come in the same commit.

## 1. The question

On 3 November the user asked: "should we reweight and fix the class imbalance for the multihop datasets with more questions
and help it learn those better".

How zrckpn's trainer weighs questions now (`lean_gpu.fit_variant` and `lean_gpu.listwiseD`, unchanged since ML1):

- It draws 32 training questions at a time from all of the split's training carves.
- It takes one step for each carve among those 32, on that carve's questions alone.
- A step's loss is the mean, over its questions with a gold row in the pool, of the question's mean -log p over its gold
  rows.

So the datasets are already close to level by steps. Each carve among the 32 gets one step at the same scale, however
many of its questions are in it. Weights per dataset would mostly rescale steps that Adam normalizes, and they would be
per dataset.

What is not level is inside a step. Each question counts once, whether its pool holds one gold row or four, and whether
the model already ranks it or not.

Musique's multi-hop questions are where zrckpn is weakest. LTM's zrckpn at seed 0 on WQ-hotpotqa reads R@5:

- 0.5806 on musique: 0.5351 on its 3-hop questions and 0.3574 on its 4-hop ones;
- 0.9135 on 2wiki, 0.9132 on squad and 0.6469 on webqsp.

A musique question's gold rows are its supporting passages, so its gold count rises with its hop count.

RWT asks: does weighting questions inside a step, by one rule for every dataset, move zrckpn?

## 2. The forms (fixed now)

Each form is one function of a question's own pool scores and gold labels, the same on every carve. Neither reads a
dataset name, a hop count or the question's text.

| form | weight of question q in a step |
| --- | --- |
| unit | 1: the trainer's loss now, checked bit for bit |
| F (focal on the question's gold mass) | (1 - P_q) / mean, where P_q is the softmax mass of q's pool on its gold rows. If the step's mean of 1 - P_q is below 1e-6, the step is unweighted. |
| G (gold rows, capped at 5) | min(n_q, 5) / mean, where n_q is the number of gold rows in q's pool. The cap is R@5's K. |

- **The mean** is over the step's questions with a gold row. So the weights average to 1, and a step keeps the trainer's
  scale.
- **No gradient** passes through the weights: they are taken from the step's own scores and labels with the gradient off.
- **A step's loss** is the mean, over its questions, of the weight times the question's loss now.
- **What each form does.** F puts a step's weight on the questions the model does not yet rank. G counts a question once
  per gold row, up to five. So a 4-hop musique question counts twice a 2-hop one, and a squad question (one gold row)
  counts as it does now.

Everything else is zrckpn's, as LTM fitted it:

- the arm and LTM's NEXT columns;
- PKO's pools and the WQ carves;
- the optimizer, the epochs and SWA;
- the read of p@swa at zrckp's cap;
- DOB's CPU device at 6 threads.

## 3. The fits (fixed now)

| fit | form |
| --- | --- |
| `scr-rwt-zrckpnf-s0-hp` | F |
| `scr-rwt-zrckpng-s0-hp` | G |

- **What they are.** Both are zrckpn at seed 0 on WQ-hotpotqa, the split musique trains on, on the host's CPU. They read
  LTM's NEXT columns of WQ-hotpotqa without writing them.
- **The base.** Each is against LTM's `scr-ltm-zrckpn-s0-hp`, with the same arm, seed, split, columns and code. LTM's
  zrckpn fits at seeds 0, 1 and 2 on WQ-hotpotqa give the base's spread.
- **The code.** No file that LTM's zrckpn fits name in their stamp has changed since they ran. The last change to any of
  them is 3aa0b58, LTM's declaration. `zrwt.py` is added to the stamp.
- **Cost.** Each fit trains for about 2.4 hours and reads for about 1 hour, on 6 CPUs with 38 GB, as LTM's WQ-hotpotqa fits
  did.

## 4. The rule (fixed now)

Each form is called with LTG's per-read screen call (`zltg.screen_row`):

- **The delta** is the form's seed-0 R@5 minus LTM's zrckpn at seed 0.
- **The floor** is max(0.0075, 2 x zrckpn's spread). The spread is zml1's: the root mean square of seeds 1 and 2's
  differences from seed 0.
- **The reads** are the six WQ-hotpotqa reads that LTG's in-domain rule decides on:
  - webqsp, squad, musique and 2wiki;
  - musique's 3-hop and 4-hop slices.

Per form:

- **NOT_ADOPTED:** a LOSS on any of the six.
- **ADOPT_FOR_FULL_RUN:** no LOSS and at least one GAIN.
- **NO_GAIN:** otherwise.
- **INCOMPLETE:** any of these:
  - a read is missing;
  - the compare's base R@5 is not LTM's zrckpn seed-0 R@5 (zml1's tolerance);
  - a slice file's all-question R@5 is not its compare's;
  - a fit's stamps do not name its form, LTM's columns as LTM's fit read them, PKO's present gate, DOB's device and
    threads, LTM's pick, and LTM's fit's code;
  - the fit's trainer never weighted a step.

## 5. Beside, deciding nothing

- The two zero-shot reads of WQ-hotpotqa, hotpotqa and metaqa, called the same way.
- Each form's mean delta over the six reads.
- For each fit:
  - its training curve (the weighted loss per epoch);
  - the steps its form weighted, and the largest and smallest weights;
  - its train and read seconds.

## 6. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| one form ADOPT_FOR_FULL_RUN | Its full run, in an amendment that adds items only. That is zrckpn with the form at seed 0 on WQ-musique and at seeds 1 and 2 on both splits: five fits, against LTM's zrckpn at the same seeds, under LTM's full-run rule (`zltm.decide_full`). On ADOPT, zrckpn with the form is the MLP candidate. Claim 1 is then read again with it (C1R), and the form is screened on the GNN in its own file. |
| both ADOPT_FOR_FULL_RUN | The form with the larger mean delta over the six reads gets its full run first; a tie goes to F. The product of the two weights, normalized the same way, is screened next. |
| neither | These two forms close for zrckpn. The next MLP form follows LTC's verdict. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

**The freeze.** A full run not read by 6-7 November does not change the MLP of record for the frozen results. Its reads are
filed when they land.

## 7. Code (`outputs/mp_unified/zrwt.py`)

- `python outputs/mp_unified/zrwt.py fit --name <fit>` runs zltm's fit for zrckpn under RWT's names. That means RWT's stamp
  key (`rwt`), stage and declaration, with `zrwt` added to the code stamp.
  - For the length of the train, `lean_gpu.listwiseD` is the form's weighted loss. `fit_variant` reads it by its module
    name.
  - Every patched name is put back afterwards, also on a failure.
  - zltm, zml1s, zltg, zdep and lean_gpu are imported, not edited.
- `compare-all` compares each read RWT fit against `scr-ltm-zrckpn-s0-hp` through zwq's compare. The outputs are
  `outputs/screen/scr-rwt-zrckpnf-vs-zrckpn-s0-hp` and `outputs/screen/scr-rwt-zrckpng-vs-zrckpn-s0-hp`.
- `calls` writes `outputs/rwt/calls.json` and `.md`.
- `--selftest` checks:
  - the unit form is `lean_gpu.listwiseD` bit for bit, in its loss and its gradient;
  - F's and G's weights by hand in float64: their mean, G's cap, F's flat case, and that no gradient passes through them;
  - the swap and its restoring;
  - a toy fit through lean_gpu's loop: under unit it gives the trainer's own states bit for bit, and under F and G the states
    move;
  - the name patches and their restoring;
  - the stamp;
  - the compares' reading;
  - the rule on made numbers.

## 8. The queue (host)

- `rwt-selftest`: 1 CPU, 4 GB.
- `rwt-zrckpnf-s0-hp` and `rwt-zrckpng-s0-hp`: 6 CPUs and 38 GB each. They wait for `pkoh-gate`, `rwt-selftest` and
  `ltm-calls-full`. They sit just before DOB's x-fits that are still pending, after every small check, so they never hold
  one up.
- `rwt-compare` and `rwt-calls`: 1 CPU each.

RWT writes only under `outputs/rwt` and its own fit and compare names. It reads LTM's fits, compares and columns without
writing them.
