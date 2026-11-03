# NMD: do the passages' own names link musique's unlinked next hop, and do they single it out?

Declared 3 November 2026, before any number. Script: `outputs/mp_unified/nmddiag.py`. Training-free, host CPU, fit
carves only.

- **Module:** `nmddiag.py` (new). It imports `l2ddiag` (sym_edges, depths, auc, order_ranks, strata_of, share, ratio,
  at, bfs, adj_of, thin_links), and through it `lthcols` (question, a_case, xc_take, toy_carve), `ltwdiag`
  (fam_bits, part_arrays, write_json, unit), `zwq` (sha_file, sha_src), `lean_cache` (part_dirs, sha_file) and
  `u1b` (package_root). None of them is edited.
- **Reads, all read only:**
  - PKO's cache and links parts of musique/s1fit, hotpotqa/fit and 2wiki/fit.
  - `outputs/lta/report.json`, LTA's census of the same A cases, checked.
  - U2r's names: `outputs/u2r/<dataset>/names.json` and `names.npz`. The npz is checked against its record's sha256,
    and the record's nodes against the served package's (`DATASET.json`'s nodes sha256 and count, and the freeze's
    RECORD_SHA256).
- **Writes:** `outputs/nmd` only: one census per carve (`<dataset>__<carve>.json`, a run with `--limit` under
  `outputs/nmd/dev`), plus `report.json` and `report.md`.

## 1. The question

The user, 3 Nov, about musique's unlinked next hop: use the names in the passages without the encoder, as the link
rule does.

The filed facts behind it:

- **Two thirds of musique's next-hop golds have no link to the found gold.** On musique/s1fit, 1,262 of the 3,798 A
  targets (0.332) are linked to their anchor by PKO's links (LTA's census; L2D 5f8da44).
- **Two links reach part of the rest.** 0.569 of the unlinked targets are two links from the anchor (L2D's SIGNAL,
  matched lift 2.42). That is LT2's term, now training. About 0.29 of all targets are three or more links away, or not
  linked at all, which no link-shaped term reaches.
- **The vectors do not hold that hop.** LTA's learned next hop, a bilinear alignment of the frozen vectors trained for
  this, lifts linked targets (T2 0.33 to 0.36 on musique/s1eval) but leaves unlinked targets at rrf's level (0.067 to
  0.096, against 0.066 to 0.067). LTH's harder negatives did not change this. A learned offset of the same frozen
  vectors has the same inputs: it can re-arrange what they hold, not add what they lack.
- **So the next lever is a new input that is not the encoder's.** The passages' names are one. They are read from the
  text alone, with no model and no new encoder call (the 24 Oct rule allows derived representations).
- **U2r built exactly these names, model-free** (14a48ee). A name is a run of capitalized tokens, found with the
  corpus's own capitalization statistic.
  - CN_u joins two passages that share an untitled name with df in [2, 25], with weight the sum of 1/df over the names
    they share. CN_all does the same for any name.
  - U2r graded them as pool links. On musique, N1 (IS + CN_u) with R_kNN reached 0.6960, against E1's 0.6811 and
    spaCy's 0.6940, and CN reached 122 of the 152 golds that only spaCy's names reached.
  - N1 lost 2wiki's pools by 0.0226, so E1 was chosen and CN_u is not in PKO's links.
  - Nobody has read CN_u at the next hop.

NMD asks whether a name the anchor shares with a row reaches the unlinked targets, and whether it singles them out
from the rows of the same rrf rank.

- If it does, a names input for the next-hop term of both models has something to carry, and it is built.
- If it does not, nothing is built, and we know which reading closed it.

## 2. What it reads

- **The A cases are L2D's** (LTA's and LTH's: `lthcols.question` and `a_case`).
  - The anchor is the first gold among the rrf top five.
  - The candidates are every row outside the top five.
  - The targets are the golds among the candidates.
- **Unlinked means L2D's depth from the anchor is 2 or more.** Depth is taken over the question's PKO links, in either
  direction, every family. Depth 2 and depth 3 or farther are kept apart.
- **The names are U2r's CN_u (the rule's) and CN_all (beside),** looked up by each row's node. Two weights are read:
  - from the anchor: the weight between the anchor's node and the row's (the rule's);
  - from the top five: the weight summed over the five top rows (beside; at serving time the anchor is not known).
- **These checks stop the census (exit 1) on any difference:**
  - Per carve, the questions, rows, A cases, targets and linked targets must equal LTA's census.
  - On every case, a row's depth 1 from the anchor must be LTW's link to it (L2D's check).
  - `names.npz` must be its `names.json`'s (sha256). The names' nodes must be the served package's: nodes sha256,
    node count and the freeze's RECORD_SHA256.
  - Every row's node must lie within the names' nodes.
  - Any package pin in PKO's cache records must be the names' freeze.
- **Carves:** musique/s1fit decides; hotpotqa/fit and 2wiki/fit are read beside. Fit carves only, never s1eval or test.
- **Nothing else:** no vectors, no fits, and no labels beyond the gold flags the A cases already use.

## 3. What it measures

Per carve, pooled over its A cases:

- **Depth counts** of the candidates and of the targets from the anchor (1, 2, 3, far), as L2D's. Beside them, the
  same counts for the rows named by each form: CN_u or CN_all, from the anchor or from the top five.
- **share:** of the unlinked targets, the share named from the anchor by CN_u. Beside:
  - the candidates' share and the unmatched lift (targets' share over candidates');
  - the share among the targets two links away, and among those beyond two links (what LT2 cannot reach);
  - the share of the linked targets also named;
  - the reach with two links: the unlinked targets at depth 2, or named beyond it.
- **Matched lift**, as L2D's: within each case, the unlinked candidates are cut into rrf deciles.
  - The expected count of named targets is the sum, over deciles, of the decile's targets times its named share.
  - The matched lift is observed over expected. It is read for CN_u from the anchor (the rule's), CN_u from the top
    five, and CN_all from the anchor.
- **AUC** of the CN_u weight from the anchor, among the unlinked candidates (targets against the rest): over the
  case, and within the rrf deciles. rrf's own AUC is beside.
- **Weight bins** of the named unlinked rows (CN_u from the anchor): below 0.1, 0.1 to 0.5, 0.5 and more (one name of
  df 2 weighs 0.5).
- **Per case:** candidates, unlinked candidates, and the unlinked candidates named from the anchor and from the top
  five (median, p90, max).
- **Fixed orders** of the unlinked candidates, read as the unlinked targets' ranks (T2, T10, median, MRR):
  - rrf;
  - named first, then rrf;
  - by the CN_u weight, then rrf.
- **musique by hop** (2, 3 and 4 hops): A cases, linked share, and the named share and lift of the unlinked targets.

## 4. The rule (fixed now)

The call is made on musique/s1fit, from the anchor, with CN_u:

- **INCOMPLETE** if any of these holds:
  - there is no census made by this script, or the deciding census is a limited run;
  - LTA's check is not "equal";
  - the names were not checked against the package;
  - there are fewer than 200 A cases.
- **SIGNAL** iff share >= 0.10 and matched lift >= 2.0.
- **NO_SIGNAL** otherwise. The reading is RARE if share < 0.10, else NOT_SELECTIVE.

hotpotqa/fit and 2wiki/fit are reported beside and decide nothing.

**Why these bars:**

- **share >= 0.10** is about 250 of musique/s1fit's 2,536 unlinked targets, a fifth as many as the 1,262 linked targets
  the next-hop terms already lift. Fewer than that would not move a read the WWW rule decides on.
- **Matched lift >= 2.0:** a name shared with the anchor has to mark a target at twice the rate of the unlinked rows of
  the same rrf rank. L2D's two-link rows reached 2.42 and were built (LT2). A names input sits beside LT2's term, so
  it has to single out at about that rate, and above L2D's 1.5 bar, because one name link reaches far fewer rows than
  two links.
- **The unmatched reads are reported but not used,** because rrf alone can produce them: rows near the top of rrf
  share more names.
- **CN_u decides; CN_all is read beside.** CN_u is the form U2r declared beside IS. CN_all adds co-mentions of titled
  names, which already have their own node and IS links to it.

## 5. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| SIGNAL | The names term is declared in its own file now. Its input is the CN_u name weight between each rrf top-five row and each row outside the top five: read from the passages' text, no encoder, and fixed before any row is scored. It feeds the next-hop term of both models, the MLP's and the GNN's, with the same rule on every dataset. Its columns are built on the host's CPU, so its screens can start before the freeze. |
| NO_SIGNAL, RARE | No names term is built. U2r's untitled names do not reach musique's unlinked next hop. The names line moves to its open forms (prefix-merged titled names, names as entity nodes), each read by this census before anything is built from it. |
| NO_SIGNAL, NOT_SELECTIVE | No names term is built. The names reach the unlinked targets, but no more often than they reach the other rows of the same rrf rank: a name shared with the anchor does not single out the next hop. The next-hop line keeps PKO's links and LT2's two-link rows. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

- **The MLP rule.** A SIGNAL's input stays message-passing-free for the MLP. The names come from the text, before any
  row is scored, and the top five are rrf's fixed beam: no other row's score enters.
- **Timing.** The censuses run tonight. A SIGNAL's term is declared on 4 Nov, its columns are built on the CPU, and
  its screens run before the freeze (00:00 IST, 7 Nov).
- No running stage waits on NMD, and no other stage reads its outputs.

## 6. Risks, stated before any number

- **One numbering.** The names index the served package's nodes, and PKO's rows carry the same node ids. NMD checks the
  names against the package and every row's node against the names' range. It reads PKO's records' package pin where
  they hold one.
- **U2r's df cap is fixed, not tuned here.** A name in more than 25 passages joins nothing. The weight bins show where
  the signal sits.
- **The weight is a sum.** CN_u's weight cannot tell one rare shared name from several common ones; the bins show both.
- **Pooled shares weight large pools,** as in L2D (musique's candidates per case: median 5,068.5, p90 7,032.6). The
  per-case sizes are reported.
- **Deciles are coarse.** Rank still varies within an rrf decile, so the matched reads may keep a little of rrf's
  signal.
- **One anchor.** The rule reads the anchor (a gold, unknown when serving). The top-five read beside is what a serving
  input sees; a SIGNAL's term reads each top-five row as LTA's every-row term does.
- **A fixed order is not a learned term.** The fixed orders show what the names alone do; they are not a forecast of a
  screen.
- **Time and memory.** NMD reads the parts as L2D's census did, plus two name lookups per top-five row per case (binary
  search). The name tables are the largest part: 2wiki's CN_all has 38.2 million pairs, about 1 GB as a symmetric
  table and about 4 GB while it is built.

## 7. Items and commands

| item | CPU, memory | command |
| --- | --- | --- |
| `nmd-selftest` | 1, 2 GB | `python outputs/mp_unified/nmddiag.py --selftest` |
| `nmd-census-musique-s1fit` | 1, 4 GB | `python outputs/mp_unified/nmddiag.py census --dataset musique --carve s1fit --host` |
| `nmd-census-hotpotqa-fit`, `nmd-census-2wiki-fit` | 1, 8 GB | `python outputs/mp_unified/nmddiag.py census --dataset D --carve fit --host` |
| `nmd-report` | 1, 1 GB | `python outputs/mp_unified/nmddiag.py report` |

U2r's three `names.npz` were pushed to the host for these items, each checked against its `names.json` there by the
census itself.

**The self-test** checks:

- the symmetric name table against a dict on random pairs, and that reversed or repeated pairs stop it;
- PKO-shaped toy carves (LTH's `toy_carve`, links thinned as L2D's) with U2r-shaped toy names. Every count, sum,
  per-case size and order is checked against a by-hand recount (BFS with a queue, names as a dict, deciles by
  position).
- that another npz, another package, rows beyond the names' nodes, PKO on another package, and A cases that are not
  LTA's each stop the census;
- that a limited run writes under `dev` only;
- the report: INCOMPLETE below 200 cases, or on a stale, limited, unchecked or LTA-absent census;
- the rule on fixed reads.

Committed: this file, `nmddiag.py`, and later `outputs/nmd/report.json` and `report.md` (masked). The census JSONs
stay on the host and the laptop.
