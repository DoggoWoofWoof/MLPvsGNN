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
