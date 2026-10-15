# P1g2: P1g's pool rules, with the extra set gated on KIND_T's own expansion

Declared 15 October 2026. P1g's s1sel stats for metaqa, squad and musique were filed at that point; those for
hotpotqa, 2wiki and webqsp, and P1g's verdict, were not. Code: `scripts/p1g2_gated.py` (P1g's
`scripts/p1g_pool_beam.py` imported unchanged). Pools only: nothing trains or reads, and nothing is fit. Status:
declared.

## 1. Why

The user's target (15 October): **ALL ≥ 0.95 on every dataset, coverage first, then the fewest rows.** P1g
(docs/P1G_POOL_BEAM.md) prices 805 rules of one form: today's pool, a fraction f of KIND_T's fill, and one extra set X
(a per-parent beam or a cut of the base's one-link neighbours). Each rule is the same on every dataset.

What P1g's filed stats show on the three datasets in so far:
- **musique decides.** Today's pool holds every gold on 0.8846 of s1sel's questions (2,094 rows). KIND_T's fill alone
  reaches 0.9498 at f = 4 (6,895 rows), just short. 100 of the 805 rules reach 0.95 there: 66 with a neighbour cut,
  24 with a one-step beam and 10 with a two-step beam. Every one of them needs an extra set. The cheapest on musique
  alone is f2+cut_rrf_3000: 0.9505 at 5,612 rows.
- **An extra set adds nodes to every question, including those KIND_T never expands.** squad's questions are those:
  KIND_T gives all 1,498 of its s1sel questions an empty fill, so its pool is the 50 base rows, and that already
  holds every gold on 0.979 of them. Under the 100 rules that meet musique, squad's pools hold 83 to 1,871 rows: 1.7
  to 37 times today's, for golds squad does not need to reach 0.95. Every cut of 3,000 or more puts it above 1,800.
- **Every row costs twice.** P1g-b builds features for every row, and the card fits hold every training row on the
  card (zspk's fits peaked at 5.2 GB on today's pools). Rows that add no needed gold make the refit slower and can
  push it off the card.

KIND_T's size rule already decides, per question and without labels, whether a question needs expanding at all. P1g2
lets that same decision gate the extra set.

## 2. The candidates

- **P1g's 805 rules,** as filed.
- **Their gated forms** `g:f+X`, one for every X except none (798). On a question whose fill today is above 0
  (|I_q| − |B| > 0, P1c2's sizes), a gated rule is P1g's f+X. On a question whose fill is 0, it is the base and seeds
  B alone. There no fraction of an empty fill adds a node, so the gate drops only X.
- **1,603 in all:** P1g's first, then the gated ones in P1g's order.

**No new pass.** A gated rule's rows and held golds come per question from P1g's filed arrays: the ungated rule's
column where the fill is above 0, and the same fraction's `none` column where it is 0. Before any use, the arrays are
checked against P1g's records: P1g's script, the file's sha256, P1g's checks, and the summary recomputed from the
arrays equal to the filed one.

## 3. The rule, fixed before any of its numbers (P1g's section 5, unchanged)

- **Ceilings, targets, cost and the choice** are P1g's, over all 1,603 rules:
  - target t_d = min(0.95, C_d − 0.005);
  - cost = P1f's question-weighted mean rows;
  - the cheapest rule meeting every target wins, with ties going to the earlier rule in the order above.

  Gating never adds a gold, so the ceilings are P1g's.
- **TARGET_MET, CEILING_MET and NONE_MEETS** are P1g's.
- **P1g2's choice is the one P1g-b builds.** It is the cheapest rule in a set that holds P1g's 805. It equals P1g's
  own choice unless a gated rule is cheaper. A gated rule that ties loses, because P1g's rules come first.

**Reported beside, deciding nothing:**
- P1g's own choice and its cost, and the cheapest ungated rule meeting every target;
- webqsp's s1eval under the chosen rule and today's (from P1g's stats arrays);
- the five training datasets' s1eval under the chosen rule and today's (`eval`);
- per dataset, how many questions have an empty fill (where the gate acts).

## 4. What follows

- **TARGET_MET or CEILING_MET:** P1g-b (docs/P1GB_POOLS_REFIT.md) builds its pools under P1g2's choice. Before the
  pools, `eval` runs on each training dataset's s1eval through P1g's `run_stage`, unchanged, on three rules: today's,
  the chosen rule's ungated form, and, when the rule is gated, the same fraction's `none`. The record is written
  under `outputs/p1g2`. P1g-b's pools check their s1eval ALL, recall and mean rows against it.
- **NONE_MEETS** can happen only when P1g's own verdict is NONE_MEETS. Then nothing is adopted, as in P1g.

## 5. Running it

- `python scripts/p1g2_gated.py --selftest` passes before this commit. It checks:
  - the 1,603 rules and their order;
  - a gated rule's rows and held golds per question, with and without empty fills;
  - that an ungated rule reads its own column unchanged;
  - that P1g's choice, imported, picks a gated rule only by cost, after every target.
- `python scripts/p1g2_gated.py verdict`: on the laptop, in seconds, after P1g's six stats records are fetched.
- `eval DATASET --host` (five host items, 2 CPUs each, as P1g's stats). Each covers one carve and two or three rules,
  so each is shorter than P1g's stats item for the same dataset.
- `python scripts/p1g2_gated.py report` writes `outputs/p1g2/verdict.md`.

## 6. Results

Not run yet.
