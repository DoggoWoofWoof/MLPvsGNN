# P1f: one pool-size rule for every dataset

Declared 15 October 2026, before any P1f number exists. Stage 3 of docs/PROGRAM_2026_10_15.md ("a common rule for
the dataset pools"). Code: `scripts/p1f_pool_size.py`. Status: **NONE_MEETS** (section 7).

## 1. Why

P1c2's pools (docs/P1C2_POOLS_KIND_T.md) take KIND_T's order, P1e's universal order, but they cut it at each
question's frozen size |I_q|. Those sizes come from step 1's per-dataset recipes, and those recipes were chosen on
s1eval golds. That is the audit's HIGH item: the order is one rule for all six datasets, but the size is not. Today's
mean pool sizes are metaqa 2018, squad 50, musique 2092, hotpotqa 94, 2wiki 106 and webqsp 2120.

P1f replaces the size with one label-free rule, the same for all six datasets. It does not change the content: the
pool is still the question's base and seeds (step 1's `bs`), then KIND_T's nodes outside them, in KIND_T's order. The
only thing a rule sets is how many of those nodes the pool takes.

## 2. Inputs (read only)

- P1e's walks, kinds and orders, imported unchanged (`scripts/p1e_anchor_kind.py`, through it P1d's five sources and
  step 4e's carves, linking and graphs), on the host mirror.
- Carves: s1sel on the five training datasets chooses the rule. s1eval on all six is computed and reported beside the
  choice; the choice never reads it.
- The cost weights: each dataset's fit and s1eval question counts, from P1c2's pool manifests
  (`outputs/p1c2/pools/{ds}/pools.json`; musique's fit carve is s1fit). These are metaqa 15,745, squad 17,729,
  musique 7,018, hotpotqa 13,335 and 2wiki 18,504.

## 3. What is measured, per question

- **Gold depth:** 0 for a gold in `bs`. Otherwise, one plus its index among KIND_T's nodes outside `bs`. A gold that
  KIND_T's 7,000 nodes do not hold has no depth (never covered).
- **`bs`'s size, today's size |I_q|, and how many KIND_T nodes lie outside `bs`.**
- **Label-free signals:**
  - the question's kind (entity or passage) and ROUTE's condition (both P1e's);
  - |L| and |L*|;
  - for each of P1d's five sources (A1st, A1all, Kst, Kall, R0) and for `max4` (the largest of the four walks):
    - the nucleus size n(τ), the fewest nodes, in the source's order, whose mass reaches τ of the source's total;
    - the perplexity exp(H) of its masses.
  - An empty source has n(τ) = 0 and perplexity 0.

**Checks**, on every carve, or the script stops:
- KIND_T's gold positions equal P1e's filed `pos_{ds}__{carve}.npz`, question for question.
- On s1eval, today's sizes reproduce P1c2's filed pool coverage (ALL and recall, to four decimals).

## 4. The rules (fixed before any number)

A rule gives each question a total size N_q. The pool is `bs`, then the first max(0, N_q − |bs|) KIND_T nodes outside
`bs`. Rows are |bs| plus the nodes taken (fewer when KIND_T runs out). Every N_q is rounded and clipped to [0, 7000].
`SIZES` = 50, 75, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000.

| family | N_q | grid |
| --- | --- | --- |
| FIX(N) | N | N in SIZES (13 rules) |
| KIND(N_e, N_p) | N_e for an entity question, N_p for a passage question | both in SIZES (169) |
| NUC(s, τ, c) | c · n_s(τ) | s in A1st, A1all, Kst, Kall, R0, max4; τ in 0.5, 0.8, 0.9, 0.95, 0.99; c in 0.25, 0.5, 1, 2, 4 (150) |
| PPL(s, c) | c · exp(H_s) | s as above; c in 0.5, 1, 2, 4, 8, 16, 32 (42) |

**Not in any rule:**
- a dataset's name or any dataset-level statistic (corpus size, mean degree);
- a gold, an answer, or a score from a trained model.

The signals belong to the question and its walk.

## 5. The choice (s1sel only)

For a rule r, on each training dataset d's s1sel:
- ALL_d(r): the share of questions whose every gold is in the pool. A question with no golds counts 0, as in P1a.
- rows_d(r): the mean pool size.
- The cost of r is the question-weighted mean of rows_d(r), with weights w_d = fit + s1eval questions (section 2).
- "Today" is |I_q|, evaluated the same way on s1sel.

1. **Budget.** cost(r) ≤ cost(today).
2. **Floor.** ALL_d(r) ≥ ALL_d(today) − 0.01 on each of the five datasets.
3. **Pick.** Among the rules that pass both: the highest equal-weight mean of ALL_d. Ties within 0.001 go to the
   lower cost, then to the earlier rule in the order of section 4.

The verdicts:
- **`ADOPT_RULE`:** a rule passes both. It becomes P1f-b's pool size, and P1f-b is declared in its own file before any
  pool is rebuilt. Rebuilding means pools, looks, caches, KT, links and the lean pair's refits; P1f-b's call is made
  on the models.
- **`NONE_MEETS`:** no rule passes. Today's sizes stay, and the HIGH item stays open, said so in the program. The
  report gives the frontier (each family's best mean ALL at or under the budget, and each dataset's shortfall) so the
  next family is chosen from it.

## 6. Reported beside (never read by the choice)

- The chosen rule (or each family's best) on s1eval: all six datasets' ALL, recall and rows, today's beside.
- webqsp is zero-shot: its row count under the rule is what the lean pair would pay there.
- Each dataset's s1sel recall, and the share of questions at the 7,000 cap.

## 7. Results

**`NONE_MEETS`** (`outputs/p1f/verdict.{json,md}`, filed 15 October about 15:43). Of the 374 rules, none passes both
the budget and the floor. Today's sizes stay, and the HIGH item stays open.

**The checks passed on every carve:**
- KIND_T's gold positions equal P1e's filed ones;
- on s1eval, today's sizes reproduce P1c2's pool coverage.

**The frontier on s1sel** (each family's best mean ALL within today's cost of 699.7 rows; cells are ALL, then the mean
rows):

| rule | metaqa | squad | musique | hotpotqa | 2wiki | mean ALL | cost |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| today | 0.9519 (2022) | 0.9793 (50) | 0.8846 (2094) | 0.9556 (94) | 0.9525 (105) | 0.9448 | 699.7 |
| FIX(500) | 0.8497 (500) | 0.9953 (500) | 0.7679 (500) | 0.9894 (500) | 0.9773 (500) | 0.9159 | 500.0 |
| KIND(1000,500) | 0.9125 (1000) | 0.9953 (500) | 0.7679 (500) | 0.9894 (500) | 0.9773 (500) | 0.9285 | 608.8 |
| NUC(R0,0.9,0.5) | 0.8864 (686) | 0.9953 (537) | 0.7855 (593) | 0.9927 (624) | 0.9813 (661) | 0.9283 | 622.7 |
| PPL(max4,1) | 0.8958 (756) | 0.9960 (640) | 0.7934 (622) | 0.9920 (573) | 0.9799 (604) | 0.9314 | 641.7 |

**What binds: metaqa and musique.**
- Today these two take about 2,000 rows a question, and squad, hotpotqa and 2wiki take 50 to 105.
- Every family's best within the budget gains on squad, hotpotqa and 2wiki (+0.016 to +0.037 ALL), but loses on
  metaqa (−0.039 to −0.102) and musique (−0.078 to −0.117).
- The smallest worst-case shortfall within the budget is PPL(max4,1)'s: musique −0.091.
- The signals do not tell a musique question that needs 2,000 rows from a hotpotqa or 2wiki question that needs 100.
  All three are passage questions, and their walks' nucleus and perplexity overlap.

**Beside, past the budget (not a choice).** 70 rules meet the floor at a higher cost. The cheapest is the simplest:
**FIX(2000)**, one size for every question, at 2,000 rows (2.86 times today's cost). Against today it gives:

| metaqa | squad | musique | hotpotqa | 2wiki |
| ---: | ---: | ---: | ---: | ---: |
| −0.0013 | +0.0187 | −0.0046 | +0.0398 | +0.0334 |

FIX(1500) misses the floor on musique (−0.018) and metaqa (−0.013). On hotpotqa and 2wiki, FIX(2000) is about 20
times today's rows.

**s1eval, beside (never read by the choice):**

| rule | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| today | 0.9606 (2018) | 0.9798 (50) | 0.8999 (2092) | 0.9594 (94) | 0.9587 (106) | 0.8410 (2120) |
| FIX(500) | 0.8566 (500) | 0.9965 (500) | 0.8060 (500) | 0.9891 (500) | 0.9792 (500) | 0.7166 (500) |
| KIND(1000,500) | 0.9187 (1000) | 0.9965 (500) | 0.8060 (500) | 0.9891 (500) | 0.9792 (500) | 0.7944 (983) |
| NUC(R0,0.9,0.5) | 0.8898 (686) | 0.9965 (542) | 0.8188 (576) | 0.9896 (628) | 0.9816 (656) | 0.7332 (582) |
| PPL(max4,1) | 0.8986 (756) | 0.9980 (642) | 0.8271 (610) | 0.9900 (568) | 0.9808 (596) | 0.7006 (446) |

**What it means:**
- **The order is one rule, but the size is not.** Today's per-dataset sizes stay, and so does the HIGH item: they
  come from step 1's recipes, which were chosen on s1eval golds. Every claim that reads these pools says so.
- **A common size exists only at about three times the rows.** FIX(2000) meets the floor everywhere. That costs
  about 20 times the rows on hotpotqa and 2wiki, which touches Claim 3's latency there and gives the models 20 times
  more wrong rows to rank. Whether to pay that is a separate stage (P1g), declared in its own file if taken, and
  decided on the models, not on coverage.
- **The pools stay final for the model stages:** G2, the ledger and T1 run on today's pools.
