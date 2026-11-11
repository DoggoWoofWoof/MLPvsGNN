# AP1b: sub-span names and the passage's own title (graph level, before any number)

Declared 12 November, before any number of this stage exists (script `outputs/mp_unified/zap1b.py`, committed with this
file). It continues [AP1](AP1_RULE_NAMES_AND_MUSIQUE_REACH.md): AP1 §11 (filed 981dab4) found that U2r's CN_all replaces
spaCy's co-mention family at the graph level, and that a shared-name rule alone is capped by how the run rule cuts a name.
AP1b asks whether reading the names inside a longer capitalized run, and the passage's own title, closes part of that gap.
Graph level only: no model reads a vector, nothing trains, no filed graph or record changes, and nothing here is an R@5.

## 1. Why, and what was read before this declaration

- AP1 §11: R + CN_all + K links 1,146 of MuSiQue's 1,648 chain slots (69.5%). A scratch read of the 502 it leaves unlinked
  (the answer read as a label; scripts not in the repository, figures not filed) found 223 where the bridge name is a name in
  both passages at a df above 25 (hubs), 163 where the string is neither a name nor a sub-span of one in at least one
  passage, and 116 where it is a name in one passage and a proper sub-span of a longer capitalized run in the other, or a
  sub-span in both (67 of them with the sub-span key's df in 2-25, 49 above 25). Examples: "Chief Justice of Delta State"
  against "women of Delta State"; "St Andrew's Birmingham City" against "Birmingham City"; "King of Aragon" against
  "Zaragoza, Aragon, Spain". In 16 of 54 slots whose answer is in the earlier passage only, the later passage's own title
  contains it ("Bill Maher" in "Real Time with Bill Maher").
- AP1 Part C: of MuSiQue's 78 GP slots that N_sp's union links and CN_w's does not, 36 share no rule name at any df.
- So the hypothesis is: U2r's run rule takes a neighbouring word into the run (a role word before a name, a place after it),
  and a period or a comma splits a run; reading every sub-span of a run as a name of the passage links what the run hides.
  The same hypothesis for the title: a passage's own title is a name its text may not repeat.
- **What this read is.** No rule below reads a label, and none has a fitted parameter. But the forms were chosen after the
  scratch read above, which used the answers of these 1,000 MuSiQue questions: the CH grade on MuSiQue is a dev read.
  2Wiki's and HotpotQA's GP slots, where nobody looked at the unlinked slots, are the check that a form does not cost
  where it was not designed; AP2's model reads and a held-out carve decide anything about models.

## 2. The questions

- **Q1.** Does a sub-span form link more of MuSiQue's chain slots than CN_all, at what price in pairs, and without losing
  gold pairs on 2Wiki or HotpotQA?
- **Q2.** Does the passage's own title, read as a further name of the passage, link slots the texts do not?
- **Q3.** What does the carried form leave unlinked on MuSiQue's chain slots (the classes of §1, rebuilt for it)?

## 3. The forms

U2r's tokens, capitalization statistic, `usual`, connector set (`CONN`) and `norm_key` are used unchanged. A **run** is one of
U2r's capitalized runs with key words w_1 ... w_L. For L >= 2 a **sub-span** is w_i ... w_j (not the whole run) whose first and
last word are not connector words; **end-anchored** when i = 1 or j = L, **infix** otherwise. A one-word sub-span must have
more than two characters and be `usual`, as a one-word run must. The **levels**:

| level | names of a passage |
|---|---|
| 0 | its runs (CN_all's names) |
| 1 | level 0 and its end-anchored sub-spans of two or more words |
| 2 | level 1 and its infix sub-spans of two or more words |
| 3 | level 2 and its one-word sub-spans |

**T**: the passage's title, with a parenthetical and the comma tail removed (underscores read as spaces), is read like a text
by U2r's tokens and runs, and its runs and their sub-spans up to the level are further names of the passage; only for passages
whose text is read (`zu2b.is_descriptive`), as in U2r.

A level's pairs L_k are co-mention pairs over its (passage, key) entries with U2r's df cap [2, 25] (a key's df is the number
of distinct passages holding it as a run, a sub-span or, with T, a title run); the weight is the sum of 1/df, as CN_all's.
The forms are unions, so that each contains the one before it (a larger df at a deeper level cannot drop a pair):

- F0 = L_0, which must equal CN_all; F_k = F_{k-1} union L_k for k = 1, 2, 3.
- F_kT = F_k union F_{k-1}T union L_kT (L_kT: level k with T), with F_{-1}T empty.

Eight forms: F0, F1, F2, F3, F0T, F1T, F2T, F3T. The union graph of a form is R + K + the form (the R-world only: R = the
W2_S pairs of b1w, K = b1's 3-NN, both unchanged); no S-world and no by-kind table here.

## 4. Slots and grading

As AP1 §5: GP (every pair of a question's gold passages) on musique, 2wiki and hotpotqa, and CH (MuSiQue's 1,648 chain slots)
on musique. The share of slots the union links; the difference to CN_all (the reference here) and to N_sp, as a ratio
estimator over questions with the 95% interval of 2,000 question resamples (seed 0); calls ABOVE / LEVEL / BELOW by whether the
interval clears 0. **Price:** union pairs added over a reference union per slot added: over F0's, over the union without a
NER slot, and (S1) over the carried form's. N_sp's price (new union pairs per slot it adds over the union without a NER slot)
is recomputed in the run.

## 5. Gates (the run stops on any failure)

G1 F0's pairs equal AP1's filed CN_all pairs (`outputs/ap1/<d>/rules.npz`, whose sha256 is the one in AP1's `report.json`).
G2 F0 is contained in F1, F1 in F2, F2 in F3; F_k in F_kT; F_{k-1}T in F_kT. G3 N_sp's price on MuSiQue's CH rounds to AP1's
filed 128.6. G4 the union R + K + CN_all links the number of slots AP1 filed, for every slot kind of every setting. G5 the runs
rebuilt here equal U2r's names, passage by passage. The self-test (toy corpus: the sub-spans of a run, the title strip, the
nesting, the decision walk on synthetic coverage) passes before the run.

## 6. What AP1b decides

- **S1, the carried form.** F0 is carried at the start. The forms F1, F2, F3 are tried in this order; then the T counterpart of
  the form carried after F3 (F_kT for the carried F_k; for F0 it is F0T). A form replaces the carried one when, against the
  carried form in the R-world: (a) its call on MuSiQue's CH is ABOVE; (b) its call on GP is not BELOW on any of the three
  settings; (c) its price over the carried form on MuSiQue's CH is at most N_sp's price. The form carried at the end is the
  NER slot AP2 reads (its pairs and weights are saved); if it is F0, AP2 carries CN_all and the loss class stays open for AP3.
- **S2.** The report states, for the carried form, the share of chain slots linked and what Q3 finds; it also states each
  step's three conditions, so that a step that failed names the condition.
- Every number is a graph-level reach. A call here moves nothing in a model of record; the paper keeps the spaCy-built
  numbers; changing the models of record and a refit stay the user's call.

## 7. Outputs

`outputs/ap1b/report.json` and `report.md` (committed); `outputs/ap1b/<d>/levels.npz` (every L_k and L_kT: source, target and
weight; uncommitted). The report records the sha256 of this script and of `zap1.py` and the script sha256 AP1's report carries;
a failed gate ends the run, so a report exists only when all five passed.

## 8. Decisions (the user's; my default first)

- **D75.** AP1b is graph level, on the laptop, with the forms and rules above, and its MuSiQue CH grade is a dev read.
  Default yes. Reverses no ruling.
- **D76.** AP2 (the models of record read on b1-style carves whose NER slot is the carried form) carries S1's pick, declared
  in its own file after this report is filed. Default yes; the alternative is CN_all whatever S1 picks. Triples and more
  per-dataset rules stay AP3 (D73).

## 9. Results (filed 12 November; `outputs/ap1b/report.json`, `report.md`; `zap1b.py` as committed in 309a133)

**The run.** It completed on all three settings, so none of the five gates stopped it (a failed gate ends the run with
SystemExit; `outputs/ap1b/run.log`). Nothing below is an R@5: each figure is the share of slots whose pair is an edge of
R + K + the form (R-world). The first column is CN_all's share; the other columns are the difference to CN_all in points,
with the call from the paired 95% interval over 1,000 questions.

| setting, slots | F0 = CN_all | F1 | F2 | F3 | F0T | F1T | F2T | F3T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| musique CH (1,648) | 69.54% | +2.73 ABOVE | +2.91 ABOVE | +6.13 ABOVE | +1.64 ABOVE | +4.25 ABOVE | +4.49 ABOVE | +7.65 ABOVE |
| musique GP (2,462) | 49.15% | +1.99 ABOVE | +2.15 ABOVE | +4.75 ABOVE | +1.14 ABOVE | +3.01 ABOVE | +3.21 ABOVE | +5.77 ABOVE |
| 2wiki GP (2,175) | 46.25% | +0.09 LEVEL | +0.09 LEVEL | +0.87 ABOVE | +0.28 ABOVE | +0.32 ABOVE | +0.32 ABOVE | +0.97 ABOVE |
| hotpotqa GP (1,000) | 79.10% | +0.40 ABOVE | +0.40 ABOVE | +1.90 ABOVE | +0.50 ABOVE | +0.80 ABOVE | +0.80 ABOVE | +2.00 ABOVE |
| musique union pairs | 141,965 | 159,325 | 162,323 | 266,886 | 144,684 | 162,661 | 165,645 | 273,379 |

The intervals are in `report.md` (MuSiQue CH: F1 +2.73 [+1.86, +3.72], F3 +6.13 [+4.85, +7.52], F0T +1.64 [+1.05, +2.24],
F3T +7.65 [+6.31, +9.10]; 2Wiki GP: F0T +0.28 [+0.09, +0.51]; HotpotQA GP: F0T +0.50 [+0.10, +0.90]). N_sp's MuSiQue union
holds 125,662 pairs.

**S1, the walk** (each step against the carried form; price = new union pairs per chain slot added over it; N_sp's price is
128.6):

| form | against | (a) CH diff [95%] | (b) GP calls, musique / 2wiki / hotpotqa | chain slots added | (c) price | step |
| --- | --- | --- | --- | ---: | ---: | --- |
| F1 | F0 | +2.73 [+1.86, +3.72] ABOVE | ABOVE / LEVEL / ABOVE | 45 | 385.8 | refused by (c) |
| F2 | F0 | +2.91 [+2.02, +3.96] ABOVE | ABOVE / LEVEL / ABOVE | 48 | 424.1 | refused by (c) |
| F3 | F0 | +6.13 [+4.85, +7.52] ABOVE | ABOVE / ABOVE / ABOVE | 101 | 1,236.8 | refused by (c) |
| F0T | F0 | +1.64 [+1.05, +2.24] ABOVE | ABOVE / ABOVE / ABOVE | 27 | 100.7 | carried |

**Carried form: F0T**, CN_all's names plus the passage's own title (trailing parenthetical and comma tail removed) read as a
further name of the passage. It holds 98,845 pairs (CN_all 92,847) and links 1,173 of the 1,648 chain slots (71.18%; CN_all
69.54%, N_sp 68.26%) for 2,719 more union pairs than CN_all, which is 100.7 per added slot against N_sp's 128.6. It is ABOVE
CN_all on GP in all three settings. Against N_sp, spaCy's family, F0T is ABOVE on MuSiQue (CH +2.91 [+1.17, +4.54]; GP
+2.23 [+0.96, +3.39]) and on HotpotQA (GP +2.10 [+1.10, +3.20]) and LEVEL on 2Wiki (GP +0.28 [+0.00, +0.57]), at 1.15, 1.12
and 1.19 times N_sp's union pairs (144,684 against 125,662; 51,308 against 45,980; 99,315 against 83,356). So at the graph
level a rule-built NER slot replaces spaCy's without a loss of reach on any of the three settings, and gains on two; no model
has read it.

**Q1, sub-spans: reach rises with every level, and so does the price.** Every sub-span form is ABOVE CN_all on MuSiQue (chain
slots and GP) and none is BELOW it on 2Wiki or HotpotQA; F3 and F3T are ABOVE on all three settings (F3: +0.87 on 2Wiki,
+1.90 on HotpotQA). But a chain slot added over F0 costs 385.8 (F1), 424.1 (F2) and 1,236.8 (F3) new union pairs, 3.0, 3.3
and 9.6 times N_sp's price; F3's one-word sub-spans alone add 107,175 form pairs over F2 on MuSiQue (104,563 union pairs) for
53 more chain slots (1,247 against 1,194). S1's condition (c) refuses all three, as declared.

**Q2, the title as a name.** The title is the cheap gain: F0T is ABOVE CN_all in all four grades and below N_sp's price. R's
title rule (W2, kind 0, the title as written with its parenthetical) never links a mention of "Robert Abel" to the passage
titled "Robert Abel (animator)"; the title read as a name does, through the same df cap [2, 25] (the title's key counts the
passage itself).

**Q3, what F0T leaves** (the answer read as a label, diagnostic only, not a grade). 475 chain slots stay unlinked: 148 where
the bridge name is neither a name nor a sub-span in at least one of the two passages; 264 where it is a name or a sub-span in
both at a df above 25 (the hubs CN_w's price refuses); 63 where it is one in both at a df of 2-25, the class the sub-span
levels were built for; none at a df below 2.

**Correction to AP1 section 11.** Its scratch read counted 67 slots whose answer label is a sub-span at a df of 2-25 and
wrote "at most 67 more slots at the cap". The measured reach over CN_all is 45 (F1), 48 (F2) and 101 (F3) slots: a form
links a slot through any name the two passages share, not only through the answer label's key, so the 67 is a count by the
label and not a bound on a form's reach. F3 reaches more slots than that count, at 1,236.8 pairs per added slot.

**What this decides.** The rule-built NER slot of the carried form is F0T (S1). The forms were chosen after AP1 section 11's
scratch read, so the MuSiQue chain grade is a dev read (D75); 2Wiki and HotpotQA, where no one looked at the unlinked slots,
show no form BELOW CN_all. Nothing about models, pools or R@5 is decided; AP2's model reads are declared after this filing.

**D77 (opened by this report; the user's).** S1's price condition refuses F1, F2 and F3; F3T reaches 99 more chain slots
than F0T (1,272 against 1,173 of 1,648) at 2.18 times N_sp's union pairs (273,379 against 125,662). The rule stays as
declared and is not reinterpreted after the numbers. Whether the extra reach moves a model is a model question. Default:
AP2 carries F0T (D76) and, on MuSiQue only, also reads F3T as a second, disclosed carve ("refused by S1's price"), declared
in AP2's own file; the alternative is F0T alone.
