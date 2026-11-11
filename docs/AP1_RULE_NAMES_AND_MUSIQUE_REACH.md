# AP1: rule-built names in place of spaCy's NER, and the reach of the name rules on MuSiQue's chains

Declared 12 November 2026, before zap1.py's first real run. The user, 11 November about 23:00 IST (verbatim): "the next
in order of business would be the after paper items like removing spacy, finding the musique ruleset, seeing if we can have
more rules to improve our results based on the datasets, hyperlinks, triplets etc". This is taken as the go of D19
([after-paper roadmap, 5 Nov]) for those items; it follows the 5 Nov ruling (drop spaCy NER, cover it with rules), the
10 Nov ruling (R must improve, MuSiQue's oracle links show where) and the canonical layout
([CANONICAL_LINK_AND_POOL_LAYOUT.md](CANONICAL_LINK_AND_POOL_LAYOUT.md): R = W2_S rule links plus the NER store, whose
spaCy build is the part to replace).

Code: `outputs/mp_unified/zap1.py` (new), which imports oraclepairs.py, zu2r.py, zu2l_el.py and zu2b.py unchanged.
Outputs go only under `outputs/ap1/`. **Nothing trains, no model reads a vector, and no filed graph, pool, fit or record
changes. AP1 decides nothing about models, the paper or the models of record.** It is the graph-level read that says
which model reads (AP2, declared separately) are worth their host hours.

## 1. Why, and what was read before this declaration

U2r (25 Oct) built a model-free name family (N1 = IS + CN_u, N2 = IS + CN_all) and graded it on the six datasets' full
corpora; it never met B1's settings, which the paper's models read, and it never met a pool test. R's rule part (W2_S,
U2b's variant W2) is the scan U2l's IS uses, so what replaces spaCy inside R is the co-mention part: CN_u or CN_all.
SWP Part D (docs/SWP_S_TO_R_SWAP.md section 5b) showed that the oracle chain edges lift
MuSiQue's b1 read by about 11 points of R@5; the question for "the MuSiQue ruleset" is how many of those chain pairs a
rule can link and at what price in edges.

**Scratch reads on the laptop before this declaration (deciding nothing; scripts under the session scratchpad, not
filed):** MuSiQue's B1 setting has 1,648 chain-pair slots (a slot = a question and one pair of its supporting passages
joined by a sub-question's dependency; 1,090 distinct pairs). Family reach on the slots: S 444 (26.9%), R 475 (28.8%),
spaCy's N 905 (54.9%), K 343 (20.8%); the b1w graph (R, N, K) links 1,125 (68.3%) and misses 523. The intermediate
answer, the bridge, appears in the earlier passage in 99.6% of slots and in the later passage in 85.7%. U2r's name rule,
run on the 11,656 passages, finds the bridge as a name in both passages in 1,185 slots (71.9%), of which 870 at df 2-25
(the cap), 288 at df 26-100, 26 at df 101-300 and 1 above. With the cap at 50, 100 and 200 the bridge-in-both slots
within it are 1,060, 1,158 and 1,184. Of the 523 slots b1w misses, the rule finds the bridge in both passages in 317
(df 2-25: 90, 26-100: 201, 101-300: 25, above 300: 1), in one passage in 157 and in neither in 49. So (i) a shared-name
rule can reach about 72% of the chain slots at most (b1w's graph reaches 68%), and (ii) the cap is what cuts the rest of the
reachable share: cliques over every name with df up to 100 would hold up to 373k pairs (before pairs shared by several
names merge) against B1a's 75k NER pairs.

## 2. The questions

- **Q1 (spaCy out).** Does a rule-built name family, rebuilt from the passage texts alone, link as many of a
  question's gold pairs as spaCy's family (B1a's `graph_ner.npz`), inside the union a model reads, on the three passage
  settings (musique, 2wiki, hotpotqa)? At what edge cost?
- **Q2 (the MuSiQue reach).** What share of MuSiQue's chain slots can any shared-name rule link (the bridge name read as a
  label, as a diagnostic only), and does CN_w (section 4), which lets names above the cap link pairs that share enough of
  them, add reach at a price below the spaCy family's?
- **Q3 (what is lost).** Which slots does the union with spaCy's family link that the union with the rule does not, and
  why (a name shared at any df, or none)?

## 3. What is read

For each setting d in {musique, 2wiki, hotpotqa}: B1a's files under `outputs/bench/hipporag2/` (the corpus
`raw/<name>_corpus.json`, `nodes.json`, `queries.json`, `graph_structural.npz` = S, `graph_ner.npz` = N_sp,
`graph_knn.npz` = K) and `outputs/bench/hipporag2_w/<d>/graph_structural.npz` = R. Each file's sha256 is checked against its
build record, and b1w's N and K must be b1's bytes (gate G5), so the R-world differs from b1 in the structural family only.
Node i is corpus entry i. A name is read from a passage's text only (as B1a's spaCy run read `corpus[i]["text"]`), not
from its title; a passage whose text is its title is blanked (U2r's is_descriptive).

## 4. The families

| family | what it holds |
| --- | --- |
| S, R, K, N_sp | the filed graphs of the setting (S = B1a's structural family; R = W2_S pairs; K = 3 nearest neighbours; N_sp = spaCy's co-mention pairs at df [2, 25], weight sum of 1/df) |
| **CN_all** | U2r's, unchanged: nodes u, w that both hold a name, df of the name in [2, 25] |
| **CN_u** | U2r's, unchanged: the same over names that are not a title's key (a titled name links through IS, which R already holds) |
| **CN_w** (new) | nodes u, w whose shared names, any df >= 2, carry sum of 1/df >= 1/25 (tolerance 1e-9): one name at the cap carries exactly 1/25, so CN_w contains CN_all and equals it when no name exceeds the cap; a name above the cap links a pair only with enough other shared names. No parameter beyond the filed cap. |
| **CN_wu** (new) | CN_w over the names that are not a title's key (CN_u's counterpart) |

A name is U2r's: a run of capitalized tokens with the corpus's own capitalization statistic (docs/U2R_NAME_RULE.md
section 2). The two worlds: R-world = R + K + the NER form (the canonical R + K domain for text graphs); S-world = S + K +
the NER form (B1a's b1). NER forms: none, N_sp, CN_u, CN_all, CN_wu, CN_w. The reference is N_sp in the same world.

## 5. Slots and grading

- **GP:** every unordered pair of distinct gold passages of a question (B1a's `golds`), one slot each, per question.
- **CH (MuSiQue only):** oraclepairs' ORC slots (the pairs a sub-question's dependency joins), one slot per question and
  pair; the bridge answer of each slot is read for Part B only.
- A slot is linked when its pair is an edge of the union. The share is linked slots over slots. The difference between two
  forms is the share difference by the ratio estimator over questions, with a 95% interval of 2,000 question resamples
  (seed 0). The call is ABOVE when the interval lies above 0, BELOW when it lies below 0, LEVEL otherwise.
- **Price:** the pairs a form adds to the world's union without a NER slot, per slot it adds; the marginal price of CN_w
  over CN_all (and CN_wu over CN_u) is the extra pairs per extra slot.
- **Part B (MuSiQue):** per chain slot, whether the rule finds the answer-name in both passages, in one or in neither, and its
  df band; which of R+N_sp+K, R+CN_all+K and R+CN_w+K link the slot; the ceiling of any shared-name rule.
- **Part C (each setting):** the loss class (linked by R+N_sp+K, not by R+CN_w+K), the gain class, and up to 12 loss
  examples with the names the two passages share by the rule.

## 6. Gates (the run stops on any failure)

G1 MuSiQue's chain slots are 1,648 and its distinct ORC pairs 1,090 (the scratch read's counts), and equal oraclepairs' own
build. G2 the ORC pairs each filed family links equal `oraclepairs.coverage`'s (S, R, N, K). G3 CN_w contains CN_all and
CN_wu contains CN_u. G4 the names rebuilt here give U2r's own CN_all and CN_u pairs. G5 b1w's N and K are b1's, and every
input file is its build record's. The self-test (toy corpus with two hub names, the interval and calls, the gold slots)
passes before the run.

## 7. What AP1 decides

- **R1, the replacement candidates.** In the R-world, a form is a candidate to replace spaCy's family when its call
  against N_sp on GP is not BELOW on any of the three settings and its call on MuSiQue's CH is not BELOW. Among the
  candidates the one with the most GP slots linked summed over the three settings (ties: fewer pairs) is carried to AP2 as
  the NER slot of a new carve. If there is none, AP2's first work is the loss class.
- **R2, CN_w's own read.** CN_w gets its own model read in AP2 only if, on MuSiQue's CH in the R-world, it is ABOVE CN_all
  and its marginal price over CN_all is at most N_sp's price over the union without a NER slot.
- **R3.** Part C is read, not graded: it names what to add next (a rule for the loss class) or says the loss is not a
  naming problem.
- Every number is a graph-level reach, not an R@5. A call here moves nothing in a model of record; AP2 (the models of
  record read on carves with the chosen NER slot, a read-time swap, not a refit) is declared after this report is filed,
  and a refit is the user's call.

## 8. Outputs

`outputs/ap1/report.json` and `report.md` (committed), `outputs/ap1/<d>/rules.npz` (the rule edges, uncommitted). The
report records the script's, zu2r's and zu2l_el's sha256, the seed and the gates' results. The scratch reads of section 1
are not filed numbers.

## 9. Decisions (the user's; my default first)

- **D71.** AP1 is graph-level, on the laptop, on the three passage settings. Default yes. The six full corpora's `ner` store
  (the package's spaCy build that ML1, PKO and ABL read) is not touched here; U2r's `names.npz` for the six exist and are
  AP2's or a later stage's input.
- **D72.** The best replacement form goes to AP2 as model reads (b1-style carves read by zrckpn and zg6r, queued behind
  J5S), declared in its own file after this report. Default yes. It reverses no ruling: the paper keeps the spaCy-built
  numbers, and changing the models of record stays the user's call.
- **D73.** "Triplets" (typed subject-relation-object edges from text without a parser; today's P2c triples use spaCy's
  parser) and more per-dataset rules (SQuAD's question links, the two-name list of typed KBs) are AP3, declared after AP2.
  Default: not before AP2's declaration.

## 10. Amendment A1 (12 November, after the first run's tables were read; declared before the second run)

The first run (fc73b37's script) passed all five gates; its tables are the report's main tables and the second run must
reproduce them unchanged (checked against the first run's `report.json`). The second run adds:

- **(i) GP by question kind**, from the `kind` field of B1a's queries (2Wiki: compositional, comparison,
  bridge_comparison, inference; HotpotQA: bridge, comparison; MuSiQue: 2hop, 3hop1, ...). The two gold passages of a
  comparison question need no link between them, so the question "does the graph link the pair" belongs to the kinds that
  chain. Descriptive; changes no call and no rule of section 7.
- **(ii) Reach against NER pairs kept**, R-world: N_sp's and CN_all's pairs ordered by weight (descending, ties by pair),
  the top fraction f in {0.1, 0.2, 0.3, 0.5, 0.7, 1.0} kept, unioned with R and K: the union's pairs and its share of GP and
  CH. Descriptive (is every NER pair needed; the 2 Nov "cut useless NER rules later" item); changes no call.
- **(iii) A correction to section 1.** The scratch reads matched each slot's two passages by title, which counts a bridge
  name found in any passage of a repeated title; the filed Part B matches by node, and its figures replace the scratch's
  bridge-in-both counts (1,185 and 317 there).
- **(iv) How R1 and R2 are read together.** R1 as written carries the candidate with the most GP slots, which the first run
  shows can be CN_w (3,014 against CN_all's 3,007 slots over the three settings), while R2 refuses CN_w its own read
  (marginal price 1,356.5 pairs per slot against N_sp's 128.6 on MuSiQue's CH). Written down after seeing that: the two are
  read as "R2 limits what R1 may carry", so a form R2 refuses is not carried. This is open as **D74** (default CN_all; the
  alternative is CN_w, R1 read alone: 7 more GP slots for 6.7k more NER pairs).

## 11. Results (filed 12 November; `outputs/ap1/report.json`, `report.md`; the second run, `zap1.py` as of ad1bfed)

**The run.** It completed on all three settings, so none of the five gates stopped it (a failed gate ends the run with
SystemExit; `outputs/ap1/run2.log` shows each setting done and the report written). Against the first run's `report.json`
every value of the tables, prices and Part C (GP, CH) the first run filed is identical in the second; the script's sha256
and the seconds differ, and the by-kind tables and the weight curve are new. Nothing below is an R@5: each figure is the
share of slots whose pair is an edge of the union.

**Q1, spaCy out: U2r's CN_all replaces spaCy's family at the graph level; CN_u does not.** R-world (R + K + the NER form),
GP share of the form and its difference to N_sp (95% interval, paired over 1,000 questions):

| setting | N_sp | CN_u | CN_all | CN_w |
| --- | ---: | ---: | ---: | ---: |
| musique GP | 48.05% | -3.25 [-4.51, -2.07] BELOW | +1.10 [-0.08, +2.22] LEVEL | +1.34 [+0.16, +2.45] ABOVE |
| musique CH (chain slots) | 68.26% | -5.10 [-6.76, -3.43] BELOW | +1.27 [-0.36, +2.84] LEVEL | +1.52 [-0.12, +3.10] LEVEL |
| 2wiki GP | 46.25% | -1.38 [-1.93, -0.86] BELOW | +0.00 [-0.18, +0.19] LEVEL | +0.00 [-0.18, +0.19] LEVEL |
| hotpotqa GP | 77.50% | -0.10 [-1.10, +0.90] LEVEL | +1.60 [+0.70, +2.60] ABOVE | +1.70 [+0.80, +2.70] ABOVE |

Pairs: the NER family alone holds 75,002 (N_sp), 92,847 (CN_all) and 99,554 (CN_w) pairs on MuSiQue; 36,147, 41,593 and
43,254 on 2Wiki; 68,698, 85,369 and 90,411 on HotpotQA. CN_all's union with R and K has 1.13, 1.11 and 1.18 times the pairs
of N_sp's. On MuSiQue's chain slots the NER slot adds 128.6 (N_sp), 155.5 (CN_all) and 165.0 (CN_w) new pairs per slot
linked over the union without it. In the S-world (S + K + the form) the calls are the same except on HotpotQA, where CN_all
(+0.40 [+0.00, +0.90]) and CN_w (+0.50 [+0.00, +1.00]) are LEVEL, not ABOVE.

**R1:** CN_all and CN_w are candidates (CN_u and CN_wu are BELOW on MuSiQue and 2Wiki); summed over the three settings CN_w
links 3,014 GP slots and CN_all 3,007. **R2:** CN_w minus CN_all on MuSiQue's CH is +0.24 [+0.06, +0.49] ABOVE (1,150 against
1,146 slots: 4 slots) at a marginal price of 1,356.5 pairs per slot against N_sp's 128.6, so R2 refuses CN_w its own read.
**Read together (section 10 (iv)), the form AP2 carries is CN_all**, the NER slot of U2r's N2 (IS + CN_all); CN_u, the
untitled names only (N1 = IS + CN_u), is the one that falls BELOW. D74 stays open (default CN_all).

**Q2, the MuSiQue reach.** By node (Part B, the answer read as a label, diagnostic only) the rule finds the bridge name in
both passages in 1,133 of the 1,648 chain slots (68.75%), in one passage in 398 and in neither in 117; at df at most 100 the
count is 1,106. The unions link 1,125 (R + N_sp + K), 1,146 (R + CN_all + K) and 1,150 (R + CN_w + K) slots, 68.3%, 69.5% and
69.8%. Of the 523 slots R + N_sp + K misses, the bridge name is in both passages in 304 (df 2-25: 82; 26-50: 112; 51-100:
84; above 100: 26) and not in both in 219. So a shared-name rule alone is capped at 1,133 slots, and CN_w's hub names add 4
slots over CN_all for 5,426 more union pairs: the df cap is not what limits the reach.

By question kind (R-world, descriptive): 2Wiki's compositional pairs are 96.1% linked by R + K alone and 98.8% with N_sp or
CN_all, its inference pairs 96.3% and 100.0%; its comparison pairs 4.9% and 6.6% (the two gold passages of a comparison
question need no link, A1 (i)) and its bridge_comparison pairs 31.4% and 33.6%. HotpotQA's bridge pairs are 88.0% linked
without a NER family, 92.8% with N_sp and 94.3% with CN_all (ABOVE). On MuSiQue CN_all is LEVEL with N_sp in four kinds and
ABOVE in two (4hop1 43.1% against 39.7%; 4hop3 35.0% against 30.1%); CN_u is BELOW in three (3hop1, 3hop2, 4hop2) and ABOVE
in one (4hop3).

**Reach against pairs kept (R-world, descriptive).** MuSiQue CH: the union without a NER family links 38.96%; with the top
10% of N_sp's pairs by weight (7,501 pairs, weight at least 1/3) 43.81%, with half 57.28%, with all 68.26%; every band of
weights adds reach, so weight pruning is not free on MuSiQue. 2Wiki GP: 43.95% with none, 45.20% with the top 10%, 46.25%
with all, so the lower 90% of N_sp's pairs add 1.05 points. HotpotQA GP: 73.00% with none, 74.40% at 10%, 76.50% at half,
77.50% with all. CN_all's curve lies above N_sp's at every fraction of its own pairs on MuSiQue and HotpotQA (on 2Wiki the
two stay within 0.14 points); CN_all has 1.24, 1.15 and 1.24 times N_sp's pairs, so the comparison is at equal fractions, not
at equal pair counts.

**Q3, what is lost (Part C, CN_w against N_sp, R-world; read, not graded).** Slots N_sp's union links and CN_w's does not
(loss), and the reverse (gain): MuSiQue GP 78 and 111, CH 71 and 96; 2Wiki GP 2 and 2; HotpotQA GP 3 and 20. Of MuSiQue's 78
GP loss slots 42 have a rule name the two passages share at some df and 36 have none. Where a name is shared it is a demonym,
a month, or a place or event name that many passages mention (italian, american; march, july, october; china, indonesia,
vietnam; world series, super bowl, warner bros, academy awards): the kind of pair the weight threshold refuses. The 36 with no
shared name are pairs spaCy's family links through a span the capitalized-run rule does not hold as such (one of them,
Jessie Woodrow Wilson Sayre with Treaty of Versailles, is the "Woodrow Wilson" case below). R3's reading: the loss is partly
a naming problem, of the run rule's shape, not of the cap.

**What this leaves (a scratch read after the report; not filed, decides nothing).** Of the 502 chain slots R + CN_all + K does
not link (1,648 - 1,146), read with the answer as a label (three scratch scripts, not in the repository; a key's df counts the
passages where it is a name or a proper sub-span of one): in 223 the bridge name is a name in both passages at a df above 25
(Mississippi River 28, South Carolina 53, NBC 61, New Zealand 62, Senate 70: the hubs CN_w's price refuses); in 163 the string
is neither a name nor a sub-span of a name in at least one of the two passages; in 116 it is a name in one passage and a
proper sub-span of a longer capitalized run in the other, or a sub-span in both: 67 of them with the sub-span key's df in 2-25
and 49 above 25. There the run rule takes a neighbouring word into the run: "Chief Justice of Delta State" against "women of
Delta State", "St Andrew's Birmingham City" against "Birmingham City", "King of Aragon" against "Zaragoza, Aragon, Spain",
"Woodrow Wilson" inside longer runs in both passages of Jessie Woodrow Wilson Sayre and Treaty of Versailles. A period or a
comma also splits a run ("U.S. Navy" against "United States Navy SEALs"; the label "Texas," carries its comma). In 16 of the
54 slots where the answer is in the earlier passage and nothing (32) or one word (22) of it is in the later passage's text,
the later passage's own title contains it ("Bill Maher" in "Real Time with Bill Maher"; "Robert Abel" in "Robert Abel
(animator)"). So the second class is a shape gap of the run rule, not a cap gap: sub-span names could reach at most 67 more
slots at the cap, 4.1% of the chain slots, at a price in pairs nobody has measured. That is the candidate for AP1b, which is
declared before it runs.

**Correction (12 November, from AP1b, `docs/AP1B_SUBSPAN_NAMES.md` section 9).** The "at most 67" above counts slots by the
answer label's key. A form links a slot through any name the two passages share, and AP1b's forms reached 45 (F1), 48 (F2)
and 101 (F3) more chain slots than CN_all, so the 67 is a count by the label, not a bound on a form's reach.
