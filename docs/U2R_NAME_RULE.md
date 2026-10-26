# U2r: names without a model, one entity rule in place of spaCy's NER

Declared 25 October 2026, before any U2r code or graded number. It follows U2l
([U2L_ENTITY_MATCH.md](U2L_ENTITY_MATCH.md)) and the user's rulings:

> i think ner and links do the same thing which is entity match and can be unified into a better ner itself, and cant
> we use this on the KB too if it needs to be a universal rule (24 October)

> yeah i mean unify it properly so we have a strong claim (25 October)

Code: `outputs/mp_unified/zu2r.py` (new), which imports zu2l_el.py, zu2b.py, zu2i.py, u1b.py and u1d.py unchanged.
Outputs go only under `outputs/u2r/`. Nothing trains and nothing is fit. No filed graph, pool, fit or record changes.

## 1. Why

U2l chose E1, the title links alone (IS). Its three-family union scores 0.7265 against 0.7125 for E3, the title links
with spaCy's co-mention edges (CO_s), so the pools' entity family already has no NER model. But E3 is above E1 on two
datasets: musique (+0.0129) and squad (+0.0007). spaCy finds names that have no node, and a title dictionary cannot.
On musique, 152 golds are one link from B through CO_s alone (U2l's diagnostic).

The claim needs one rule with no trained model that keeps E1's gains and also covers names without a node. U2r
declares that rule and grades it as U2l graded EL, with spaCy's edges as the reference.

**A diagnostic read before this declaration** (scratch, deciding nothing). On musique's s1sel, the rule of section 2,
fixed in the diagnostic's script before it read a gold, reached 122 of the 152 golds and scored 0.6960 on its
three-family union, against E3's 0.6940 and E1's 0.6811. 70% of CO_s's pairs were among its pairs. The rule below is
that rule, unchanged. musique's numbers in this stage are recomputed by zu2r.py and must equal the diagnostic's
(section 7).

## 2. The names

The texts are the ones U2l's scan reads: on a passage dataset the descriptive passages' texts (zu2b.is_descriptive);
on a KB every node's name, read as its text, except webqsp's CVT mediators, which neither mention nor own (U2l,
section 2).

- **Tokens** are the matches of `\w+(?:['’\-.]\w+)*`. A token is *capitalized* when its first character is an
  uppercase letter. Two tokens are *adjacent* when only white space lies between them.
- **The capitalization statistic** is the corpus's own, as u1d defines it for titles: per lowercased token, its
  occurrences that do not start a sentence (`u1d.sent_start`), counted as capitalized or as all lowercase. A token is
  *usually capitalized* unless its lowercase count exceeds its capitalized count (u1d's common rule).
- **A run** starts at a capitalized token and extends over adjacent capitalized tokens. A connector (zu2i's CONNECT
  without the coordinators and, &, y and e) joins the run when it sits between two capitalized tokens, all three
  adjacent. A token that starts a sentence starts a run only if it is usually capitalized.
- **A name** is a run's tokens lowercased and joined by one space, with a leading "the" and a trailing possessive
  ('s) dropped, longer than 2 characters. A one-token name must be usually capitalized. Each node's names form a set.
- **A title's key** is the node's title with underscores read as spaces, lowercased, with a leading "the" dropped,
  with and without its trailing parenthetical (u1d's PAREN). A name equal to some title's key is *titled*: it has a
  node.

No list from outside the corpus, no label and no model enters. The frozen encoder is not read by the names.

## 3. The edges

| kind | between | kept | weight |
| --- | --- | --- | --- |
| **CN_u** | u and w both hold a name that is not titled | df in [2, 25] | Σ over shared names of 1/df |
| **CN_all** | u and w both hold a name | df in [2, 25] | Σ over shared names of 1/df |

df is the number of nodes that hold the name. The cap and the weight are NER's (CRAG's `ner_edges`, U2l's CO_t).
CN_u is the unified rule: a name with a node links to that node (IS, U2l's, unchanged), and a name without a node
links the nodes that hold it. CN_all keeps the co-mentions of titled names as well.

## 4. The candidates

| form | edges |
| --- | --- |
| E1 | IS (U2l's chosen form) |
| N1 | IS ∪ CN_u |
| N2 | IS ∪ CN_all |

Each is graded alone, with R and with R and kNN, as in U2l (section 4 there): R is the sibling links on a passage
dataset and the stored triples on a KB; kNN is the package's. The references, beside: E3 (IS ∪ CO_s) in the same
three unions, W2_S ∪ NER ∪ kNN and PKG ∪ NER ∪ kNN.

## 5. The grading

U2l's section 5, unchanged: U2b's grading of B ∪ X(k), X ranked by the question's cosine, through
`zu2l_el.reach_unions` on the families. U2l's families are read as filed (`zu2l_el.families`), and CN_u and CN_all are
added beside them. s1sel on musique, squad, hotpotqa, 2wiki and metaqa; webqsp's s1eval beside. No test split is read.
The frozen I is not read.

## 6. The choice

- **Score:** per dataset, U2b's choice score of the form's three-family union (E ∪ R ∪ kNN); the choice reads the
  mean over the five s1sel datasets.
- **Order:** E1, N1, N2. The pick is the earliest form within 0.002 of the best mean.
- **Outcome:** `CHOSEN <form>` when the gates hold on all six; otherwise `REFUSED`, naming what is missing.
  `CHOSEN E1` means the name edges add nothing to the mean, and U2l's choice stands.
- **The claim read (beside; it does not move the pick).** Per dataset, the pick's three-family score against E3's.
  `NER_MATCHED_ALL` when the pick is at or above E3 − 0.002 on all five s1sel datasets; otherwise `NER_MATCHED_NOT_ALL`
  with the datasets below. webqsp's s1eval is filed beside and never counts.

## 7. Gates

Any failure stops the dataset, and none of its numbers is read.
1. **Selftest** (`zu2r.py --selftest`), before the commit that adds the code:
   - the names of toy sentences equal names worked by hand: a sentence-initial article, a connector between names,
     a coordinator between two names, punctuation inside a run, a possessive, a one-token common word, a title with
     a parenthetical;
   - the statistic equals a count by hand on toy texts;
   - CN_u and CN_all equal a brute-force count of pairs and weights, and CN_u holds no titled name.
2. **The families.** U2l's records and arrays are the filed ones (`zu2l_el.families` checks their sha256), and on
   every dataset the E1 and E3 unions' scores and their all-of-X reads equal U2l's filed grade record exactly.
3. **musique.** The names, mentions, CN_u and CN_all pair counts equal the diagnostic's (392,380 names, 1,027,176
   mentions, 931,069 and 766,887 pairs), checked in a development run before the code's commit. It reads no gold.
4. **Exactness.** Names are kept as strings, not hashed: two different names are never one name.

## 8. What follows (fixed now)

- **CHOSEN N1 or N2:** U2m reads the chosen form in its entity slot in place of E1, in a dated section of
  docs/U2M_POOLS_THREE_FAMILIES.md declared before its code (it amends section 19's E1, which has not run).
- **CHOSEN E1:** U2l's choice stands, and the entity family has no model either way.
- Either way, U2n's models may read CN_u and CN_all as edge kinds beside IS, CO_t and CO_s.
- The paper names the name rule, not NER: "title-mention entity links, and co-mention links over names found by the
  corpus's own capitalization statistic".

A verdict closes only the form tested (docs/SCREENS.md, section 3b): capitalized runs with these connectors, the
corpus's statistic, NER's df cap [2, 25], and these unions. A name merged into a title group by its prefix, another
df cap, and names as entity nodes of their own are different forms, and none is closed.

## 9. How it runs (laptop)

- `zu2r.py names --dataset D`: the two passes (the statistic, then the names) and the edges, with their record.
  Each pass streams the nodes file, whose sha256 is checked against DATASET.json.
- `zu2r.py grade --dataset D [--carve C]`: section 5, with gate 2.
- `zu2r.py choose`, then `zu2r.py report`.
- Order: musique, squad and metaqa first; then webqsp, hotpotqa and 2wiki, after U2q's laptop runs (chain lapS), or
  on the lab host if it is reachable first.
- Outputs: `outputs/u2r/<D>/names.json`, `grade_<carve>.json`, `outputs/u2r/choice.json` and `report.md`. The arrays
  (`names.npz`) stay uncommitted.

## 10. Results (filed 26 October 2026)

**`CHOSEN E1`. The claim read: `NER_MATCHED_NOT_ALL` (below E3: musique).** U2l's choice stands. The pools' entity
family is the title links alone, with no NER model, as before.

**Gates.** The selftest and gate 3 passed before the code's commit (e645d79). Gate 2 passed on all six datasets: the
eight U2l scores (E1 and E3 in the three unions, W2_S ∪ NER ∪ kNN, PKG ∪ NER ∪ kNN) equal U2l's filed grade record
exactly, with nothing differing. Names are kept as strings.

### The three-family scores (s1sel; webqsp is s1eval, beside)

| form | musique | squad | hotpotqa | 2wiki | metaqa | mean of five | webqsp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 | 0.6811 | 0.9908 | **0.9025** | **0.7671** | **0.2912** | **0.7265** | **0.5168** |
| N1 | **0.6960** | **0.9915** | 0.9004 | 0.7445 | 0.2903 | 0.7245 | 0.5148 |
| N2 | 0.6953 | **0.9915** | 0.8954 | 0.6953 | 0.2874 | 0.7130 | 0.5108 |
| E3 (reference) | 0.6940 | **0.9915** | 0.8920 | 0.6948 | 0.2900 | 0.7125 | 0.5071 |

E1 has the best mean and comes first in the order, so it is the pick. N1 sits 0.0020 below it.

### The claim read

E1 against E3 per dataset: musique −0.0129, squad −0.0007 (within the 0.002), hotpotqa +0.0105, 2wiki +0.0723 and
metaqa +0.0012; webqsp's s1eval +0.0097, beside. The chosen form is at or above spaCy's on four of the five datasets
and below it on musique, where the names without a node matter (section 1).

### Beside, deciding nothing

- **The name rule against spaCy.** N1 (IS ∪ CN_u) is at or above E3 on all six: musique +0.0020, squad 0.0000,
  hotpotqa +0.0084, 2wiki +0.0497, metaqa +0.0003, webqsp +0.0077. A rule with no model and no outside list matches
  spaCy's co-mention edges wherever they help and costs less where they hurt.
- **Why N1 is not the pick.** Against E1, N1 gains musique (+0.0149) and squad (+0.0007) but loses 2wiki (−0.0226),
  hotpotqa (−0.0021) and metaqa (−0.0009); webqsp −0.0020. On 2wiki the name edges dilute X's order as the kNN family
  does: E1 alone 0.8191 and N1 alone 0.7862, falling to 0.7671 and 0.7445 with R and kNN. 2wiki's CN_u holds
  12,133,314 pairs.
- **The names.**

| dataset | names | mentions | titled | CN_u pairs | CN_all pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| musique | 392,380 | 1,027,176 | 51,025 | 766,887 | 931,069 |
| squad | 70,103 | 171,727 | 254 | 165,521 | 173,645 |
| hotpotqa | 6,691,279 | 31,412,080 | 2,504,599 | 7,884,128 | 23,218,108 |
| 2wiki | 9,529,415 | 49,225,075 | 2,883,336 | 12,133,314 | 38,242,751 |
| metaqa | 37,914 | 41,894 | 33,194 | 3,287 | 12,605 |
| webqsp | 675,583 | 1,133,354 | 494,714 | 286,248 | 995,434 |

The names pass took 1,581 s on hotpotqa and 2,594 s on 2wiki (laptop), and grading took 508 s and 1,154 s.

### What follows

- Section 8's `CHOSEN E1` branch: U2l's choice stands, and U2m's section 19 (E1) runs as declared, with no amendment.
- U2n's models may read CN_u and CN_all as edge kinds beside IS, CO_t and CO_s. N1's musique gain is signal, so it
  reaches both models as an input before the idea closes (docs/SCREENS.md, section 3b).
- This verdict closes only the forms tested: these runs, connectors, statistic and df cap [2, 25] in these unions.
  Open, each in its own file: a tighter df cap on CN_u (2wiki's loss comes with its pair count), a name merged into a
  title group by its prefix, and names as entity nodes of their own.
