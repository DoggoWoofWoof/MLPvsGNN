# P5b3: a learned back-off gate between H2 and P5b2's g (a diagnostic)

Declared 14 October 2026, about 07:45, before any number of the stage. P5b2's filed result says that any further
reranker form gets its own file ([P5B2_HELD_OUT_STOP.md](P5B2_HELD_OUT_STOP.md), section 7), and this is that file.
It builds on P5b2's filed files (004c31c).

Nothing here is adopted, and no ranking model trains or reads.

## 1. The question

P5b2's F fuses g with H2 at a fixed half weight. It finds more of the golds zsp leaves low on metaqa, hotpotqa and
2wiki, including golds no fit question had (2wiki 0.316 against H2's 0.156 at N = 10). It finds fewer on musique.
There g adds nothing (G alone finds 0.011 of the new golds at N = 10, H2 0.159), and the fixed half weight costs H2.
P5b2's record put it this way: a fixed fusion cannot back off where g knows nothing.

**The question:** a gate can set, per question, how much the fusion trusts H2 and how much it trusts g, from what is
known before any gold. With that gate, does the order find more of the low golds on some dataset and fewer on none?

**Beside it:**
- what weight the gate gives each dataset;
- whether one weight for every question (no inputs) does as well.

- The user's rule (13 October, about 10:40) stands. The pools are final and parameter-free; g and the gate are
  rerankers on them.
- **Only the fusion changes.** These are all P5b2's, unchanged:
  - g: P5b2's filed G (epoch 4, sha256 559323e5...);
  - the candidates, H2 and the two reads;
  - the grid and the call.

## 2. What changes: the gate

Take a question q whose candidates rank r_H2(v) in H2 and r_G(v) in G (1 is first). The gated order **FG** sorts them
by

    s_q(v) = λ_q / (60 + r_H2(v)) + (1 − λ_q) / (60 + r_G(v)),

highest first, ties by node id. Three settings give known orders:
- λ_q = ½ gives P5b's F exactly, since halving is exact in floating point;
- λ_q = 1 gives H2;
- λ_q = 0 gives G.

**The gate:** λ_q = sigmoid(b + w · z_q). Here z_q is the question's ten inputs, each scaled by its mean and spread on
the fit questions (section 3). An input whose spread is below 1e-6 is left unscaled.

**The ten inputs** use nothing about golds:
- log n, where n is the number of candidates;
- ring1 and ring2: the shares of the candidates one and two links from a seed (P5a's rings over the three families);
- the log of the number of seeds;
- g's entropy over the candidates, divided by log n (0 when n = 1);
- g's probability mass on ring 1, and on ring 2;
- pi_max: g's largest weight π over the (seed, head) pairs;
- delta_rms: g's RMS |Δ| over the (seed, head) pairs (pi_max and delta_rms are P5b's beside measures);
- overlap25: the share of G's first 25 candidates that are also in H2's first 25 (min(25, n) where n is smaller).

That makes 11 weights in all: the bias and ten.

## 3. The gate's fit

- **Data: P5b2's held-out fit questions.**
  - What they are: per training dataset, the 20% of P5b's filed base that P5b2's split holds out.
  - g never trained on them, though its stop chose epoch 4 on them.
  - Only their new golds count: golds that no train question of the same dataset had.
  - Questions with a new gold: metaqa 195, squad 21, musique 140, hotpotqa 181, 2wiki 224.
- **Why these questions.** On s1eval, g meets questions it never trained on. Among the fit questions, only the
  held-out ones put g in that same position; on its own train questions, g has seen the golds.
- **The objective.** The fit maximizes

      J = Σ_q ω_q · mean over q's new golds of log(λ_q / (60 + r_H2) + (1 − λ_q) / (60 + r_G)) − 0.001 |w|².

  - The bias b carries no penalty.
  - ω_q = 1 / (D · |Q_d|), so every dataset weighs the same.
  - Each term is the log of a mixture of two rank distributions that share one normalizer, so the normalizer drops out.
- **The fit method:**
  - L-BFGS-B (scipy), starting from all weights zero (λ = ½, which is P5b's F), with the exact gradient;
  - at most 5,000 iterations;
  - tolerances of 1e-8 on the gradient and 1e-12 on J.
- **Beside:**
  - a constant gate: b alone, one weight for every question, giving the order **FC**;
  - J at λ = ½, 1 and 0, which is J for F, H2 and G.
- **Caveats:**
  - squad has 21 such questions;
  - the gate is fit on the questions that chose g's epoch, so J on them is not a held-out number (s1eval is);
  - the fit side has no low read, because zsp's top 5 are filed for s1eval only. So the gate fits on every new gold of
    a held-out question, while the call reads s1eval's low golds;
  - webqsp never fits, so its gate comes from the five training datasets (zero-shot).

## 4. The read

**The fit side (the five training datasets).** On the host, the held-out questions come from P5b's filed base, as
P5b2's split gives them. Checks:
- the split's counts equal P5b2's train record;
- g's held-out loss on these questions equals P5b2's at its kept epoch, within 1e-5. This holds over all golds and over
  new golds;
- on today's carve, each question's seeds and candidates (its final pool less its seeds) equal the base's, and its
  frame vector is within 1e-4 of the base's;
- today's pools file is the one P5b's base ran on.

What it stores:
- per candidate: its rank in H2 (rings, then the frozen cosine: P5b's `h2_low`), its rank in G, its ring, and whether
  it is a gold and a new gold;
- per question: the ten inputs.

**The s1eval side (all six).** This rebuilds P5b2's read the same way: the same sample, pools, seeds and counted golds
(low and pool), with every check of P5b's section 5. What it stores:
- per candidate: its ranks in H2 and G, its ring, and whether it is a low gold, a pool gold, and a seen gold;
- per question: the ten inputs.

s1eval chooses nothing.

**The orders:** FG (the call), F, H2, G and FC.

**The reproduction check.** At λ = ½, 1 and 0, the gated order must give P5b2's filed F, H2 and G:
- the same golds and the same found(N) on every question of both reads, by ring and for seen and new golds apart;
- pi_max and delta_rms equal to P5b2's, within 1e-5.

No question may differ.

## 5. The call, fixed before any number

The checks come first. Any failure stops the stage:
- P5b2's script, train record, g and six read records are the filed ones, by sha256, and so are P5b's script and its
  five base records;
- kappa equals P5b2's;
- every check of section 4 holds;
- every read ran on one script and one g.

**The call is P5b's, applied to FG against H2.** For each dataset (all six, with webqsp zero-shot), at N = 5, 10, 25,
50 and 100:
- Delta(N) = found_FG(N) − found_H2(N) on the low read;
- its 95% interval comes from a question bootstrap (2,000 resamples, seed 0);
- a point counts when |Delta| > 0.002 and its interval is clear of 0.

The labels are P5a's: MORE, FEWER, MIXED and TIE.
- FG **FINDS_MORE** when it is MORE on at least one dataset and FEWER on none.
- The stage **GOES_FURTHER** if FG FINDS_MORE, and otherwise **STOPS**.

**Reported beside, deciding nothing:**
- FG against F, F against H2, G against H2 and FC against H2, labelled the same way;
- the gate's weights and the constant gate's weight;
- J on the fit questions for the gate, the constant gate, ½, 1 and 0;
- λ on the fit questions and on s1eval, per dataset (its mean, and its 10th, 50th and 90th percentiles);
- found(N) by ring, and for seen and new golds apart;
- the pool read.

## 6. What follows

- **GOES_FURTHER:** P5c, in its own file. P5c makes the gated scores inputs to the models (the reranking named in
  P5b's section 6), with g's scores out of fold.
- **STOPS:** the learned offsets close.
  - For each dataset where FG is not MORE or TIE, the record says where it fell short: its label, the mean λ, G alone,
    and the constant gate.
  - The order's next stage is P1c2, which waits for clean-up 15, and then P4.
- The encoder stays frozen, nothing is finer than a node, and test splits are never read.

## 7. How it runs

- **Code:** `scripts/p5b3_gate.py`. It imports P5b2's script unchanged, and through it P5b's and P5a's.
- **The self-test passes:**
  - the gated order at ½ equals P5b's fusion (either argument order), and at 1 and 0 equals the two orders. This holds
    with ties, and with positions standing in for node ids;
  - ranks round-trip;
  - the ten inputs match a hand example;
  - J's gradient matches finite differences, with inputs and for the bias alone;
  - the fit recovers a planted gate: λ is above 0.9 or below 0.1 where the planted input says so, and J beats ½, 1,
    0 and the constant;
  - counts from the stored ranks equal counts along the explicit orders on node ids, by ring and for seen and new;
  - gate and verdict on made-up reads, where g leads only on 2wiki and metaqa questions marked by an input: those two
    come out MORE, the rest TIE, and the call GOES_FURTHER;
  - a changed P5b2 count is refused.
- **Development runs, no number.** `read DATASET --limit Q` runs on the first Q sampled questions and writes nothing.
  It logs only that every step ran, and the checks. Where P5b2's g, P5a's tables or P5b's base are not on disk, it
  works in memory, on P5b's development path:
  - g at its start stands in for g;
  - this dataset's base is built in memory.

  Ran before this file was committed, 40 questions each: musique, metaqa and webqsp (webqsp has no fit side). Every
  step ran and every check held. musique and webqsp ran on P1c2's own files; metaqa ran on stand-in pools. zsp's top 5
  (P5a's read1) are on the host only, so the checks against them wait for the host.
- **Host items** (light lane, all on the card at 0.1; 2 CPUs; no dependencies, since P5b2's files are on the host):

  | item | what | memory (P5b2's measured read peak, plus room for the fit carve) |
  | --- | --- | --- |
  | `p5b3-read-<dataset>` (six) | the fit side (training datasets) and the s1eval side | metaqa 13, squad 3, musique 35, hotpotqa 8, 2wiki 9, webqsp 22 GB |
  | `gate`, `verdict` | run locally, on the fetched files, beside P5b2's fetched reads | |

  - Records go to `outputs/p5b3/`. The read records, `gate.json`, `verdict.json` and `verdict.md` are tracked.
  - The fit and eval arrays are fetched for the gate and the verdict, and are never tracked.
  - They write under 0.2 GB in all.
- **ETA:**
  - on a normal host, each read takes about 1 to 6 minutes (P5b's reads took 35 seconds to 4.5 minutes; the fit side
    adds one more carve);
  - the gate and the verdict take about 2 minutes on the laptop.

  The call should land about 20 minutes after the items start. The host's link has been dropping since about 07:05
  and has been down since about 07:30, so the items wait in the feeder until it returns.

## 8. Results (filed 14 October 2026, about 09:50): STOPS

`python scripts/p5b3_gate.py gate`, then `verdict`, on the fetched records (`outputs/p5b3/gate.json`, `verdict.json`
and `verdict.md`). Every check held:
- P5b2's script, train record, g and six read records, and P5b's script and five base records, are the filed ones.
  kappa is P5b2's;
- **the fit side:**
  - each split's counts are P5b2's;
  - g's held-out loss equals P5b2's at its kept epoch exactly, over all golds and over new golds;
  - every held-out question's seeds and candidates are the base's, and so is its frame vector (difference 0);
  - each fit carve's pools file is the one P5b's base ran on;
- **the s1eval side:** on all 2,000 sampled questions (all 1,503 on webqsp), today's pool and seeds equal read1's, and
  each final pool is P1c2's KIND_T file's;
- **at λ = ½, 1 and 0, the gated order gives P5b2's F, H2 and G on every question of all six datasets,** on both reads
  (0 differ). pi_max and delta_rms equal P5b2's (difference 0).

The host ran the six reads between 09:37 and 09:42, 16 seconds to 4.4 minutes each (peak memory 2.5 to 33.4 GB).

**The gate.** The fit converged in 20 iterations on 761 questions (metaqa 195, squad 21, musique 140, hotpotqa 181,
2wiki 224).

| J on the fit questions (higher is better) | all | metaqa | squad | musique | hotpotqa | 2wiki |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| the gate | **−4.751** | −5.259 | −4.310 | −5.496 | −4.319 | −4.368 |
| one weight for every question (0.739 on H2) | −4.799 | −5.386 | −4.325 | −5.552 | −4.330 | −4.404 |
| F (½) | −4.816 | −5.352 | −4.346 | −5.653 | −4.343 | −4.388 |
| H2 (1) | −4.838 | −5.608 | −4.306 | −5.526 | −4.321 | −4.427 |
| G (0) | −5.003 | −5.622 | −4.403 | −6.235 | −4.388 | −4.370 |

- **Its weights** (inputs scaled on the fit questions): the bias is +1.40.
  - Toward H2: g's mass on ring 2 (+1.44), g's entropy (+1.15) and the top-25 overlap (+0.68).
  - Toward g: the pool's ring-2 share (−0.89) and g's offset size (−0.83).
  - The other five are within ±0.17.
- **Its mean weight on H2:**
  - on s1eval: metaqa 0.32, squad 0.85, musique 0.90, hotpotqa 0.76, 2wiki 0.60, webqsp 0.72 (zero-shot);
  - on the fit questions: 0.47, 0.92, 0.89, 0.76 and 0.46 for the five training datasets.

**The call: STOPS.** FG finds more of the low golds than H2 on metaqa, hotpotqa, 2wiki and webqsp. It ties on squad
and finds fewer on musique.

| low read, against H2 | metaqa | squad | musique | hotpotqa | 2wiki | webqsp (zero-shot) |
| --- | --- | --- | --- | --- | --- | --- |
| **FG (the call)** | **MORE** | TIE | **FEWER** | **MORE** | **MORE** | **MORE** |
| F (P5b2's call) | MORE | TIE | FEWER | MORE | MORE | MIXED |
| FC (one weight) | MORE | FEWER | TIE | MORE | MORE | MORE |
| G | MORE | FEWER | FEWER | FEWER | MORE | FEWER |
| FG against F | MORE | TIE | MORE | TIE | TIE | MIXED |
| mean weight on H2 | 0.32 | 0.85 | 0.90 | 0.76 | 0.60 | 0.72 |

FG against H2, Delta at N = 5, 10, 25, 50 and 100 (n.c.: not counted):
- metaqa: +0.031, +0.067, +0.142, +0.210, +0.285;
- 2wiki: +0.128, +0.202, +0.169, +0.092, +0.016 (n.c.);
- hotpotqa: +0.046, +0.025 (n.c.), +0.034 (n.c.), +0.029 (n.c.), +0.004 (n.c.). Its MORE rests on N = 5;
- webqsp: +0.004, +0.005 (n.c.), +0.001 (n.c.), +0.003 (n.c.), +0.000 (n.c.). Its MORE rests on N = 5;
- musique: −0.005 (n.c.), −0.006, −0.005 (n.c.), −0.011, −0.009;
- squad: no point counted (79 golds).

The pool read gives the same six labels.

**What the reads say.**
- **The gate backs off where g knows nothing, and leans on g where it reaches.**
  - On metaqa it puts 0.32 on H2. FG then finds about as much as G alone: 0.076 of the low golds at N = 10, against G's
    0.080 and H2's 0.009.
  - On musique (0.90) and squad (0.85) it stays close to H2.
  - On webqsp, zero-shot, it turns P5b2's MIXED into a small MORE (one point, +0.004 at N = 5).
- **musique: the back-off recovers most of F's loss, but not all of it.**
  - F lost 0.037 to 0.092 against H2. FG loses 0.005 to 0.011, counted at N = 10, 50 and 100.
  - The losses are golds one link from a seed (1,155 of 1,959). At N = 25, FG finds 0.450 of them and H2 finds 0.460.
  - Below a weight of 1, g's rank still moves nodes in H2's order: at 0.95 on H2, it can move a node three or four
    places near rank 10. On musique, g's rank is noise: G alone finds 0.011 of the low golds at N = 10.
- **One weight for every question (FC, beside):** it ties on musique and finds more on the other four datasets, but it
  finds fewer on squad (−0.076 at N = 5, on 79 golds). It decides nothing, and no stage is built on it after this read.
- **New golds** (golds that no fit question had):
  - metaqa: FG finds 0.024 of the new low golds at N = 10 (H2 0.007, G 0.028);
  - 2wiki: FG finds 0.293 (H2 0.156, G 0.290);
  - hotpotqa: FG ties H2 on the new golds alone (0.312 against 0.295).

**What follows (section 6, STOPS).** The learned offsets close, and P5c does not start. The record says where FG fell
short: musique, FEWER at a mean weight on H2 of 0.90. There G alone is FEWER, and one weight for every question ties.

**P5b, P5b2 and P5b3 together:**
- A learned target g(question, seed, graph features) reaches golds two or more links away on 2wiki and metaqa,
  including golds no fit question had.
- A per-question gate keeps those gains and keeps most of H2's order on musique, but not all of it.
- The reason: a rank fusion moves H2's near ranks at any weight below 1. musique's low golds sit one link from the
  seeds, where H2's cosine orders them best.

The order's next stage is P1c2 ([P1C2_POOLS_KIND_T.md](P1C2_POOLS_KIND_T.md)), which waits for clean-up 15, and then
P4. Nothing here is adopted.
