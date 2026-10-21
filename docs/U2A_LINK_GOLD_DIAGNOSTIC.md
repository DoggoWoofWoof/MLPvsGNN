# U2a: which golds the LLM-free links lose, and why

Declared 21 October 2026, before any number. Program: [PROGRAM_2026_10_21.md](PROGRAM_2026_10_21.md), section 3
item 4 and section 5. Code: `outputs/mp_unified/zu2a.py`. Outputs: only under `outputs/u2a/`.

U2a is a diagnostic. Nothing trains, nothing is chosen, and no filed graph, pool, fit or record changes.

## 1. Why

On the full corpora, the base pool loses golds when the hyperlinks are replaced by the LLM-free links. Step 4h and U1e
filed these I-stage means (ALL, recall, mean pool):

| dataset / carve | G0 (hyperlinks) | GU (U1d, rule V2) | GU_1024 (U1e, damped) |
| --- | --- | --- | --- |
| hotpotqa s1sel | 0.9682, 0.9791, 94.1 | 0.8959, 0.9426, 119.5 | 0.8879, 0.9387, 93.1 |
| hotpotqa s1eval | 0.9684, 0.9800, 93.7 | 0.8858, 0.9386, 118.7 | 0.8774, 0.9342, 92.6 |
| 2wiki s1sel | 0.8964, 0.9579, 105.1 | 0.8369, 0.9347, 162.3 | 0.8596, 0.9437, 119.7 |
| 2wiki s1eval | 0.9059, 0.9637, 105.7 | 0.8507, 0.9416, 163.5 | 0.8728, 0.9508, 120.4 |

U1c and U1f were not adopted because of this lost reach (PROGRAM_2026_10_21 section 1.2). U2a says where each lost gold
goes, so that U2b's fixes are aimed rather than guessed.

## 2. Objects

- **Datasets and carves:** hotpotqa and 2wiki, carves s1sel and s1eval. These are step 1's carves, read the way step
  4h reads them (`step4c_walk_pools.populations`). musique is used only for the re-derivation check in section 3.
- **The base pool I^G:** the frozen construction, through `m3b_compile.prepare` unchanged. It is the RRF base (budget
  50), the seeds (dense top 5 and SPLADE top 5), and the first 25 neighbours of each seed over G (`h1_c25`). A seed's
  neighbours are its undirected link neighbours in node-id order (`m3b_pools.FamilyStore`).
- **The graphs:**
  - **G0:** the package hyperlinks, M3B's local store.
  - **GU:** U1d's rule V2.
  - **GU_1024:** GU without the mention links whose target has more than 1,024 mention links in (U1e). U1f retrained
    on this graph.
- **A lost gold:** a gold that is in I^G0 and not in I^GU_1024. The same table is also kept against I^GU.

## 3. The re-derivation on the laptop, and its checks

The host has been unreachable since about 06:00 on 20 October. GU and GU_1024 live only there. The laptop holds the
whole package, so U2a re-derives the seeds' GU rows locally, with `u1d.py`'s own functions imported unchanged:
`Surfaces`' surface forms, `scan` (bounded match, case gate, capitalization counts) and `choose_targets` (rule V2).

- **A seed's GU row** is the union of:
  - **out:** the titles its own text names;
  - **in:** the passages whose text names its title.

  Each link must pass the case gate and the corpus capitalization statistic (a lowercase form written more often in
  lowercase than capitalized, mid-sentence, is a common word and never links). A link goes to every node of the named
  title's group, and never to the passage itself.
- **The capitalization statistic and the in-degrees** are counted over the whole corpus, for every lowercase form
  involved:
  - the forms of the seeds' and golds' titles, of all three surface kinds;
  - every title that a seed's or gold's text names.
- **GU_1024** drops a mention link when its target's in-degree is above 1,024.

The checks, all of which must pass before any cause is read:

1. **musique:** the re-derived out- and in-sets of every examined node equal U1d's filed V2 edges
   (`outputs/u1d/musique/link/s0of1.npz`), and the re-derived capitalization counts equal the filed `capstat.npz` for
   every form involved.
2. **hotpotqa and 2wiki:** I^GU and I^GU_1024, rebuilt from the re-derived seed rows, reproduce the filed I means in
   the table above to four decimals, on both carves. I^G0 must reproduce step 4h's G0 I in the same way.
3. **The restricted surface table:** with every form kept, it equals `u1d.Surfaces` exactly on musique (surfaces,
   owners, kinds, lowercase ids, title groups).

If any check fails, U2a files REFUSED with the difference, and no cause is read.

## 4. What is recorded for each lost gold

**The parents.** These are the seeds whose first 25 G0 neighbours hold the gold. For each parent, U2a also records the
hyperlink's direction: seed to gold, gold to seed, or both.

**One cause per graph,** the first in this order that applies:

1. **DAMPED** (GU_1024 only): some seed has a GU link to the gold, and U1e's cap removed every such link.
2. **CAP_CUT:** some seed has a link to the gold in this graph, but the gold falls outside that seed's first 25
   neighbours. The neighbours are taken in node-id order.
3. **No link from any seed.** The cause is classified from each parent's hyperlink: its source text, and its target's
   title. The most advanced step over the parents and directions counts, in this order:
   - **COMMON:** the title occurs and passes the case gate, but its lowercase form is a common word.
   - **CASE:** the title occurs (word-bounded, any case), but never written as the title is, apart from the first
     letter.
   - **ALIAS:** the full title does not occur, but an alias surface of U1d's kinds 1 or 2 does, with the case gate.
     Kind 1 is the title without its trailing parenthetical, or the text before its first comma. Kind 2 is the last
     token. For this cause U2a also records:
     - the number of title groups that own the alias (more than 50 trips the AMBIG guard);
     - whether the alias's lowercase form is common;
     - whether the dense context choice (the highest cosine to the source passage over each owner group's nodes)
       picks the gold's title.
   - **ABSENT:** neither the title nor an alias occurs. The anchor text is not derivable from the title.
   - **SHORT:** the title is under 4 characters.

**Beside each lost gold.** None of these changes a pool.
- whether a ner or knn neighbour of any seed is the gold, with no cap and within the first 25;
- the gold's rank in the dense and SPLADE top-1000 lists;
- **the question-guided hop** from each parent p. The residual r = q - (q·p̂)p̂ scores cos(r, v). For each parent, two
  ranks are recorded:
  - the gold's rank over p's GU neighbours, all of them, where the link exists;
  - the gold's rank over the whole corpus (the graph-free hop).

  The gold's rank under the plain question cosine over the whole corpus is recorded beside it.

## 5. Pool-level reach

These are measurements only. A later pool stage chooses, on all six datasets. For each of G0, GU and GU_1024, U2a
records ALL, recall and the mean pool for three pools:
- **I:** the filed object (cap 25, node-id order);
- **I_nocap:** every neighbour of every seed;
- **I_out:** cap 25 again, but over the seed's own out-links first, then its in-links, each in node-id order.

## 6. Outputs

- `outputs/u2a/musique/check.json`: check 1 and check 3.
- `outputs/u2a/<D>/summary.json`: check 2, the pool-level table, and cause counts per carve and graph, with the share
  of lost golds and the share of all golds.
- `outputs/u2a/<D>/lost_<carve>.jsonl`: one record per question, lost gold and graph.
- `outputs/u2a/report.md`.

The results are appended to this file as a section 7. No verdict is filed: U2a only reads.

## 7. What follows

**U2b** declares one universal change for each cause class that holds a material share of the lost golds (at least
0.10 of them on either dataset), and is chosen on s1sel by I's gold reach. A cause and its change are paired as
follows:

| cause | the change U2b declares |
| --- | --- |
| CAP_CUT | the neighbour order (a pool rule for all six, declared as its own pool stage) |
| DAMPED | the damping |
| COMMON or CASE | the gate |
| ALIAS | the alias kinds, with the AMBIG guard and the context choice as recorded |
| ABSENT | no title rule can reach these golds; they are left to the question-guided or graph-free hop if section 4's ranks support it |

U2c rebuilds on the result. Everything is new files beside today's.
