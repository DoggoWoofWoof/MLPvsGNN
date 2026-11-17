# AP5: does a question's type change which link label is precise? (the query-side typed rules, scouting)

Declared 17 November 2026, before any AP5 number. Code: `outputs/mp_unified/zap5.py`, new. It imports `zap4.py` and
edits nothing. Outputs go only under `outputs/ap5/`.

It follows AP4 ([AP4_LABEL_PRECISION_LEDGER.md](AP4_LABEL_PRECISION_LEDGER.md), filed c72372d and bfff7fd). AP4's
section 6 named the query-side typed rules as the next scouting stage, after its ledger was filed. The user approved it
on 17 Nov ("yeah this is fine"), and ruled that links and encoder work come before the hyperlink reads and the baselines.

**What AP5 does not touch:**
- Nothing trains and no model is read.
- No ledger, pool, link, column or fit is written.
- It decides nothing about D83, D91 or any refit.

## 1. Why

AP4 found that D83's labels differ widely in how often a link from what rrf found leads to a gold it missed:
- R.title_comention is 3 to 13 times as precise as R.mention, at under one pair per anchor;
- K.knn POINTS everywhere on the passage datasets.

Those are averages over all questions. A query-side typed rule would let the question's words choose which labels to
follow. Such a rule is worth building only if the question's type changes which label is the more precise one. If one
label order holds for every type, type-free label weights (AP4's, D92) say the same thing.

## 2. What it reads

- **The ledgers.** AP4's filed ledgers, unchanged: `outputs/ap4/ledger/<arm>/<ds>.npz`, sha-checked against their
  records by `zap4.load_ledger`.
  - **Deciding:** D83E, the record's s1eval carves, on HotpotQA (7,405 questions), 2Wiki (12,576), MuSiQue (2,417) and
    MetaQA (9,785).
  - **Beside, deciding nothing:** B1V, the 1,000-question B1 carves of HotpotQA, 2Wiki and MuSiQue.
- **The types.** The datasets' own types, as analysis strata (no rule reads them at query time):

  | dataset | types | source |
  |---|---|---|
  | HotpotQA | bridge, comparison | the dataset's `type` field |
  | 2Wiki | compositional, comparison, bridge_comparison, inference | the dataset's `type` field |
  | MuSiQue | 2, 3 and 4 hops | the question id (`2hop__`, `3hop1__`, ...) |
  | MetaQA | 1, 2 and 3 hops | the question id (`metaqa:1hop:...`) |

  The `type` field comes from `outputs/mp_unified/look/<ds>/<carve>/info.json`, by position.
- **Not read:** SQuAD and WebQSP, which have no types, and the found-gold-free questions (as in AP4).

## 3. Measures (zap4's, per type)

For each read, dataset, type and anchor kind (top1, found), over the type's questions only:
- each label's price, precision and lift, with its precision against any link's (zap4.unit_block);
- any link's measures;
- **the label pairs:** for every two labels with pairs, the paired difference of their precisions over the type's
  questions, with ABOVE, LEVEL or BELOW (zap4.boot: 2,000 resamples, seed 0);
- beside: the share of questions whose in-pool golds are all in rrf's top five, where no link can add a gold.

## 4. Gates

- **G1, alignment.** The look's `ids.json` must hash to the ledger record's `ids_sha256`, so position i is question i.
  On MuSiQue the id hops must equal the ledger's own hop column.
- **G2, the types.** Every question's type must be one of section 2's. Otherwise AP5 exits.

## 5. The rule (fixed before any number)

- **A dataset FLIPs** when some label pair is ABOVE in one type and BELOW in another, at the top1 anchor.
  - Only types with at least 100 questions at that anchor are read.
  - The top1 anchor decides because it is what a model has without knowing the golds. The found anchor is reported beside.
- **Verdict on D83E:**
  - **TYPED:** at least two of the four datasets FLIP;
  - **NO_TYPE_SIGNAL:** otherwise.
- B1V's calls are reported beside and decide nothing.

## 6. What follows

- **TYPED.** It names AP5 part 2, declared in its own file before its numbers:
  - a parameter-free surface rule that recovers the types from the question text (fixed patterns, graded against the
    datasets' types on the training carves);
  - type-conditioned label preferences, chosen on the training carves and graded on s1eval, as an input for the one next
    refit (never a refit of its own).
- **NO_TYPE_SIGNAL.** The query-side typed rules close for D83's labels. Label weights stay type-free (D92 unchanged).
  The verdict closes only these labels and types.
- **Either way,** the per-type tables go to the journal's analysis beside AP4's ledger.

## 7. Rows

- **`ap5-rep`:** `python outputs/mp_unified/zap5.py report` on the host, 1 CPU and 6 GB. It needs no other row, since
  AP4's ledgers are filed. It writes `outputs/ap5/report.json` and `report.md`.
- **Then the laptop:** fetch, then a results section here.

## 8. Development log (disclosed before any AP5 number)

- **The self-test** (`zap5.py selftest`, laptop) passes on synthetic ledgers built with zap4's question_pairs:
  - a constructed flip is called FLIP, with its pair and its types;
  - a constructed non-flip is called NO_FLIP;
  - the per-type pairs and questions sum to the whole;
  - a type under 100 questions is not read;
  - the verdict, G2 refusals and the report's text.
- **G1 and G2 were checked on the laptop** against the seven reads' ids and info files. This printed only the
  per-type question counts (above) and that every ids list matches its ledger. No ledger was opened, and no measure
  was looked at.
