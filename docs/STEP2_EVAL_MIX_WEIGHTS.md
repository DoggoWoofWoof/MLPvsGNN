# Step 2: a fit weighted to each eval's mix, on all six datasets

Declared 7 October 2026, before any number of this step exists. It is step 2 of docs/MUSIQUE_DIAGNOSIS.md section 5
(a difficulty-balanced fit), under that section's rule: a change to the training of every dataset, graded on every
dataset, zero-shot included. Numbers here are development numbers; the paper's numbers come from one declared
confirmation run.

## 1. Question

Each fit carve has a different mix of question types than the population it is read on. musique's fit is 72% two-hop
where dev is 52%, and 6% four-hop where dev is 17%. hotpotqa's fit is 17% `hard` where its validation questions are all
`hard`. 2wiki's fit is 3% `inference` where dev is 12%. The loss then spends most of its weight on the easier types.

Does weighting each training dataset's fit questions to its eval population's mix give models that read at least as
well as step 1's unweighted fits on every dataset, and better on some, in-domain and zero-shot?

The fits, carves, candidates and picks are step 1's (docs/STEP1_MATCHED_SELECTION.md sections 4 to 6). Only the loss
weights change, so a difference is the weighting's.

## 2. Weights (filed before any training)

For a training dataset d and a stratum t (step 1's: metaqa hop, squad answerable, musique hop, hotpotqa level/type,
2wiki type; `step1_carves.stratum`):

- p_fit(t): the share of t in d's fit carve (musique: F'), and p_eval(t): its share in d's eval read. Both come from
  `outputs/step1/carves.json` (sha256 `53cfb89f…`).
- r(t) = p_eval(t) / p_fit(t), clipped to [1/4, 4]. The clip keeps every type in the fit and bounds any one type's
  weight.
- w(q) = r(t(q)) divided by the mean of r over d's fit carve. Each dataset's mean weight is then 1, so its share of
  the loss stays as in step 1.

`scripts/step2_weights.py` writes `outputs/step2/weights.json`. It holds every fit question's stratum and weight, in
fit-carve order, and checks the ids against carves.json and the fit carve's question ids against the package. The
weights:

| dataset | stratum: weight (fit share -> weighted share; eval share) |
| --- | --- |
| metaqa | hop1 0.874 (0.292 -> 0.255; 0.255), hop2 1.052 (0.361 -> 0.380; 0.380), hop3 1.052 (0.347 -> 0.365; 0.365) |
| squad | unanswerable 1.500 (0.334 -> 0.501; 0.501), answerable 0.749 (0.666 -> 0.499; 0.499) |
| musique | hop2 0.718 (0.721 -> 0.518; 0.518), hop3 1.429 (0.220 -> 0.314; 0.314), hop4 2.846 (0.059 -> 0.168; 0.168) |
| hotpotqa | hard 4.460 (0.173 -> 0.769; 1.000), easy and medium 0.279 (0.827 -> 0.231; 0) |
| 2wiki | inference 4.037 (0.029 -> 0.115; 0.123), bridge_comparison 1.115, comparison 0.790, compositional 0.904 |

The clip binds on hotpotqa and on 2wiki's inference questions, so there the weighted mix moves toward the eval mix without reaching it (2wiki's other types land within 0.004 of theirs). On metaqa, squad and musique the weighted mix equals the eval mix to rounding.
webqsp never trains.

## 3. Trainer

`outputs/mp_unified/lean_gpu2.py` is lean_gpu's trainer with one change. Each question's listwise term is multiplied
by its weight before the step's mean over its questions with gold:

- step 1: mean of l(q);
- step 2: mean of w(q) l(q).

Everything else is lean_gpu's:

- unit order, chunks of 32 split by carve, Adam, dropout, the step guard and SWA;
- deterministic algorithms, with TF32 off;
- the eight epoch states and the SWA state are saved;
- reads and the CPU read check (1e-4) are lean_gpu's own.

Checks before any fit is read:

- **CPU identity**: with every weight 1, lean_gpu2's states equal lean_gpu's bit for bit, after two epochs on the 2wiki
  fit carve, for p and pf (laptop).
- **Weights**: each fit's train record holds weights.json's sha256. Every question's weight is the file's, matched by
  question id against the cache's ids.

## 4. Fits and candidates

The six seed-0 fits are step 1's: J5 and leave-one-out L-metaqa, L-squad, L-musique, L-hotpotqa and L-2wiki. They
use the same carves, basis, variants (p, pf, n, nf), config `2e-3:1e-4:0.1:8:2` and hidden 128. Each fit has 36
candidates. Every candidate reads step 1's carves: the fit's M select carves, its D select carves, and all six eval
reads.

## 5. Grading

Each fit is picked twice by step 1's rule: once on the M select carves and once on the D select carves.

- **Primary pick rule**: the one step 1 adopts. That is D if step 1's verdict is ADOPT, and M otherwise. Step 1's grade
  fixes this before any number of this step exists. The other rule's reads are reported, not graded.
- **A read** is one fit on one eval read, using the primary rule. Per question, take step 2's pick minus step 1's
  pick. The paired bootstrap runs over questions: 2,000 resamples, seed 20261007, percentile 95% interval.
- **Labels**: ABOVE / BELOW when the R@5 interval lies above / below 0; AT when it holds 0. Two different trainings
  can never share a pick, so there is no SAME. FC@5 and hit@1 are reported beside R@5.
- **Primary reads** are step 1's eleven: J5 on its five training datasets, each leave-out fit on its held-out dataset,
  and J5 on webqsp.

**Verdict** (seed 0):

- **ADOPT**: no primary read BELOW, and at least one ABOVE.
- **NOT_ADOPTED**: any primary read BELOW. A loss on one dataset blocks adoption.
- **NO_EFFECT**: every primary read AT.

After ADOPT, seeds 1 and 2 run under the same rules. The verdict is then read again on the three seeds' mean
difference per question, and later steps train with these weights.

## 6. Diagnostics (reported, not graded)

- Each primary read split by stratum, including the up-weighted ones: musique three and four hops, metaqa three hops,
  hotpotqa `hard`, 2wiki `inference`, and squad's unanswerable questions.
- The other pick rule's reads.
- Each candidate's eval R@5 and the eval-best candidate. The eval-best candidate is an oracle, never a result.
- Each pick's variant and state.

## 7. Order of work and ETAs (7 October)

1. Commit this file, `scripts/step2_weights.py`, `outputs/step2/weights.json`, `lean_gpu2.py` (after its CPU identity
   passes) and `scripts/step2_grade.py` (after its selftest passes). Push.
2. On the host: six weighted GPU fits on step 1's feature caches, queued right behind step 1's GPU items. Then their
   reads and CPU checks.
3. The grade, once step 1's grade exists. Expected one to two hours after step 1's verdict.

Queued 04:27, 7 October (076cf5c): a 2wiki smoke on the card first, then the six weighted fits at step 1's
shares as step 1's fits free the card, then reads and checks. Expected verdict: about 2 to 3 pm, 7 October.
Revised 05:15: step 2's L-musique fit ran out of GPU memory at its first context-statistics pass and restarted with
no cap (docs/STEP1_MATCHED_SELECTION.md section 9). Expected verdict: about 2 to 2:30 pm.

## 8. What this does not do

- No new carves. The selection is step 1's.
- The encoder and the substrate embeddings stay frozen.
- No GNN. Step 3 (depth) and step 4 (pool reach) each need their own file.
- Test splits are never read.

## Results

None yet.
