# U2h's full run: the lean pair with KT_W on all six splits

Declared 23 October 2026, after U2h's verdict (ADOPT_FOR_FULL_RUN, docs/U2H_W2S_BESIDE.md, section 9) and before any
number of this run. Code: `outputs/mp_unified/zu2hf.py`, and its Modal stages added to
`outputs/mp_unified/zu2d_modal.py` in a follow-up commit, as U2h's were. Everything else is imported unchanged: zu2h.py,
zu2g.py, zu2d.py, zu2d_pools.py, zg6.py, zp1c3, zp1c3p, screen_recall.py, lean_screen2.py and lean_cache. Numbers here
are development numbers.

## 1. The question

U2h's section 7, fixed before its numbers: "the lean pair's full run with KT_W (six splits per arm, under grade-nullx),
declared in its own file, with zg6 on the same input (zg6w) beside it". This is that file.

U2h read two splits (L-musique, L-hotpotqa) on twelve reads per arm. A full run reads every split: J5 and the five
leave-outs, 36 reads per arm. Musique, squad, metaqa and 2wiki get read in-domain on J5 and zero-shot on their own
leave-out.

The user's ruling, 23 October: "use the $19.5 we have, write the full-run declaration see smaller gpus if possible".
The lab host may be lost, so the run is on Modal, on the credit of workspaces 3, 4 and 5 (section 7).

## 2. What runs

Everything is on U2d's base build (today's graph, P1c2's pools), as U2h was. KT_W is U2h's filed input
(outputs/u2h/ktw, the records under it), unchanged.

**The arms.** zrckpw and zspkw, exactly U2h's (zu2h.py's classes and carves), seed 0, U2d's card.

**The splits.** lean_screen2's six: J5, L-metaqa, L-squad, L-musique, L-hotpotqa, L-2wiki.
- **L-musique and L-hotpotqa** are U2h's screen splits. Their fits and compares are reused as filed:
  - the arms' fits (scr-u2h-zrckpw, scr-u2h-zspkw and their -hp fits);
  - the bases' seed-0 fits and the seed null (U2d's scr-u2d-base-* fits and scr-u2d-null-* compares);
  - the decision compares (scr-u2h-dh-*).
  
  All of these ran on Modal's L40S.
- **J5, L-metaqa, L-squad and L-2wiki** are new. Each gets fresh fits, all on one GPU type D (section 3).

**The fits** (zu2hf.py's FITS; suffixes -j5, -lmetaqa, -lsquad, -l2wiki):

| group | fits | what |
|---|---:|---|
| probe | 2 | zrckp, seed 0, J5, on the base build: scr-u2hf-probe-l4 and scr-u2hf-probe-t4 (section 3) |
| arms | 8 | scr-u2hf-zrckpw-*, scr-u2hf-zspkw-* on the four new splits |
| base | 7 | scr-u2hf-base-zrckp-*, scr-u2hf-base-zspk-* (seed 0) on the four new splits. zrckp's J5 seed-0 fit is the probe on D |
| fallback | 1 | scr-u2hf-base-zrckp-j5, run only when D falls back to L40S (section 3) |
| null | 16 | the same bases with seeds 1 and 2 (scr-u2hf-base-<arm>-s1-*, -s2-*) on the four new splits |
| zg6w | 6 | zg6w on all six splits (beside, section 5) |
| zg6 | 6 | zg6 on all six splits, on D (beside) |

Each fit is read with p@swa on the six s1eval carves of the base build, on the GPU it was fitted on. Each comparison
below is between fits on one device. The two device types never meet inside a comparison.

**zg6w.** zg6 (docs/G6_FREE_GNN.md) with KT_W added through new zero-initialised weights. The free GNN's boundary
input xb (the eight KT columns and the two chain features) enters at two places: zg6's input layer h0 and every one of
its three layers. KT_W enters at both places, as one more term after zg6's own:
`h0 = gelu(W_in [H, xb] + b_in + W_kw,in KT_W)` and `pre_l = ... + W_x,l xb + b_l + W_kw,l KT_W`.
- **The new weights:** 64 × 8 × 4 = 2,048 parameters, zero at the start.
- **Nothing else changes:** zg6's draws, its seed offset, its generator, its readout and its edges are zg6's.
- **The check:** zg6w's score at the start is zg6's, bit for bit (the selftest checks it in eval and training mode).

## 3. The device (fixed before any number)

U2h's fits ran on an L40S ($1.95 per GPU-hour). The run looks overhead-bound: about 27 ms per step and 4-6 GB of GPU
memory. A cheaper GPU may cost less per fit. The rule picks the GPU by measured cost, not by any result.

**The probe.** zrckp's seed-0 J5 fit on the base build, trained and read twice in workspace 3:
- on an L4 (scr-u2hf-probe-l4);
- on a T4 (scr-u2hf-probe-t4).

Both use U2h's image (u2d-gpu:py3.13:torch2.8.0+cu128:numpy2.3.2) and U2h's container settings (4 CPUs, 16 GiB
requested); only the GPU differs. Each container writes timing.json beside the fit: setup, train, read and total
seconds, from the function's start to the read's end.

**The rule.**
- `cost(g) = total_seconds / 3600 × (price(g) + 4 × 0.047 + 16 × 0.008)`, at Modal's list prices of 23 October 2026:
  L4 $0.80, T4 $0.59 and L40S $1.95 per GPU-hour, $0.047 per core-hour, $0.008 per GiB-hour.
- **D** is the probe GPU with the lower cost, among the probes that trained and read. Within 1% of each other, D is
  L4, which has more memory.
- If neither probe trains and reads, D is L40S, U2h's device.
- The probe on D is the full run's zrckp J5 seed-0 base fit. The other probe is reported beside as device noise (the
  same fit on two GPUs), deciding nothing.
- `zu2hf.py device` files D in outputs/u2hf/device.json, committed before any other fit runs. Every later fit and read
  refuses to run on another GPU.

## 4. The decision

**The reads.** Per arm, 36: six splits × six datasets. Each is R@5 of the arm's p@swa minus its base arm's seed-0
p@swa of the same split (zrckpw against zrckp, zspkw against zspk), with a 95% question-bootstrap interval:
- on L-musique and L-hotpotqa, U2h's filed decision compares;
- on the four new splits, the new compares scr-u2hf-dh-<arm>-<split>.

The primary reads are step 1's eleven (lean_screen2.primary): J5 on all six, and each leave-out on its held-out
dataset.

**The floors (grade-nullx's rule, on this build).** Each read's floor is
`max(0.0075, 2 × sqrt((D1² + D2²) / 2))`. D1 and D2 are the base arm's seed-1 and seed-2 R@5 minus its seed-0 R@5 on
that read:
- on L-musique and L-hotpotqa, U2d's filed null (scr-u2d-null-*);
- on the four new splits, the new null compares scr-u2hf-null-<arm>-s<seed>-<split>, on D.

The base arm's own seed-0 fit is both the read's base and the null's seed 0, so nothing is re-based. Each read is
called by screen_recall.recall: a GAIN or LOSS stands only when |delta| reaches its floor.

**The rule (P1c3's pair rule, docs/P1C3_FULL_RUN.md, section 5):**
- **ADOPT:** neither arm has a LOSS on any of its 36 reads, and the two arms together have at least one primary GAIN.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a compare or null file is missing or not the declared one, a compare's fits ran on two devices, or
  an arm does not have 36 reads (11 primary).

Each arm's own verdict (a primary GAIN and no LOSS) is reported beside and decides nothing.

## 5. Reported beside, deciding nothing

- **zg6w against zg6** on each split, both on D: delta R@5, FC@5 and hit@1.
- **The MLP's share of the GNN's gain.** zrckpw's R@5 gain over rrf divided by zg6w's, where zg6w's gain exceeds
  0.002. On L-musique and L-hotpotqa, zrckpw ran on L40S and zg6w on D, so those shares cross devices and are marked.
- **Device noise:** the two probes' R@5 per dataset.
- **Each arm's FC@5 and hit@1 deltas** on its 36 reads.
- **The cost:** each probe's cost per fit, and the run's total by workspace.

## 6. What follows (fixed now)

- **ADOPT:** the lean pair becomes (zrckpw, zspkw). Next, each in its own file:
  - KT_W's cold serving cost (P4c's measure, on W2_S's walk);
  - Claim 1's intervals for the new pair;
  - zg6w's own full run against zg6 under zg6's seed null, if its readings here show a gain.
- **NOT_ADOPTED with a LOSS:** a per-question gate on the KT_W head, its weight per question from inputs computed
  before any row is scored, in its own file (SCREENS 3b's back-off form).
- **NOT_ADOPTED with no LOSS and no primary GAIN:** KT_W stays out of the lean pair. W2_S's signal goes on through the
  link-precision features (the hyperlink-gap plan's step 1), beside KT_W as inputs to both models, in its own file.
- **INCOMPLETE:** the missing fits and compares run when credit allows (November's credit, or credit the user adds);
  nothing else changes.
- **Always:** the hyperlink-gap plan runs beside this full run, each step in its own declared file:
  1. link precision;
  2. a hyperlink scorer learned on 2wiki and applied unchanged to the other datasets;
  3. corpus aliases.

## 7. How it runs, and the credit

From the laptop:

    modal run outputs/mp_unified/zu2d_modal.py --stage upload
    modal run --detach outputs/mp_unified/zu2d_modal.py --stage hf-fits --fits scr-u2hf-probe-l4,scr-u2hf-probe-t4
    modal run outputs/mp_unified/zu2d_modal.py --stage fetch-hf-fits --fits ...
    python outputs/mp_unified/zu2hf.py device          # files D; committed before the other fits
    python outputs/mp_unified/zu2hf.py compare-all
    python outputs/mp_unified/zu2hf.py decide
    python outputs/mp_unified/zu2hf.py --selftest

Each command runs with the workspace's profile set for that command only.

**Credit at 19:00 IST, 23 October** (each workspace's $30 for October): workspace 3 $4.54, workspace 4 $6.56,
workspace 5 $8.50; $19.60 in all.

**Where each group runs:**
- **Workspace 4:** the arms, then zg6w. It is the only workspace with KT_W.
- **Workspace 3:** the probes, then the base fits, then zg6 as credit allows.
- **Workspace 5:** the null, then zg6, once its copy of the base build matches the build's records file by file.

Fits may move between workspaces 3 and 5, which both hold the base build. Every array is checked against its record
when a fit loads it, so a wrong copy stops a fit before it trains.

**Priority:** (1) the arms and the base fits, (2) the null, (3) zg6w and zg6, the four new splits first.

**The credit rule.** Before a group launches in a workspace, n × cost(D) × 1.15 must fit within that workspace's
remaining credit, less $0.30 (from `modal billing report`). If it doesn't, fits are cut from the end of the priority
list. The cut fits are named in the filing. A run without the arms, the bases or the null is INCOMPLETE until credit
returns.

**The estimate,** if an L4 fit takes about 1.3 times an L40S fit (U2h measured $0.56 per fit on the L40S):

| item | cost |
|---|---:|
| lean fit | about $0.33 |
| zg6 fit | about $0.50 |
| probes | $0.75 |
| arms and bases (15) | $5.0 |
| null (16) | $5.3 |
| zg6w and zg6 (12) | $6.0 |
| total | about $17 of $19.60 |

## 8. Results

Filed after the run: outputs/u2hf (device.json, grade-zrckpw, grade-zspkw, verdict) and outputs/screen/scr-u2hf-*.
