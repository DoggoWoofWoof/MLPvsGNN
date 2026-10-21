# U2b: one LLM-free link rule for all six datasets, chosen by the reach of the pools it serves

Declared 22 October 2026, before any number. Program: [PROGRAM_2026_10_21.md](PROGRAM_2026_10_21.md), section 5 (the
user's ruling of 21 October), and the user's instruction of 22 October: "find improved links improve all the datasets,
and hence improve and compress the pools and eliminate all upstream failures, use modal if required for speed".
Follows [U2A_LINK_GOLD_DIAGNOSTIC.md](U2A_LINK_GOLD_DIAGNOSTIC.md), section 7. Code: `outputs/mp_unified/zu2b.py` (the
rule, the grading, the choice) and `outputs/mp_unified/zu2b_modal.py` (the corpus stages on Modal). Outputs: only under
`outputs/u2b/` and new paths on a new Modal volume.

Nothing trains. No filed graph, pool, fit or record changes: every graph here is new, beside today's.

## 1. Why

U1d's rule V2 ([U1D_PRECISE_LINKS.md](U1D_PRECISE_LINKS.md)) replaced the hotpotqa and 2wiki hyperlinks and cost
reach (U2A section 1). U2a's hotpotqa record (`outputs/u2a/hotpotqa/summary.json`) sorts the golds the frozen I keeps
over the hyperlinks and loses over GU:

| cause (hotpotqa, GU) | s1sel share of the lost | s1eval share of the lost |
| --- | --- | --- |
| ALIAS: a text names the gold, under a form V2 does not link | 0.58 | 0.56 |
| ABSENT: no V2 form of the gold's title occurs in a linking text | 0.27 | 0.33 |
| COMMON: the form is gated as a common word | 0.08 | 0.06 |
| CAP_CUT: the link exists, the node-id cap of 25 cuts it | 0.05 | 0.04 |
| CASE, SHORT | 0.02 | 0.02 |

Against GU_1024, DAMPED holds 0.11 of the lost on s1sel. 2wiki's U2a run is still going. U2A section 7 asks for one
universal change per cause class that holds at least 0.10 on either dataset. This declaration covers every class
(including COMMON and CAP_CUT, below that bar on hotpotqa) so that 2wiki's shares cannot leave one out.

A probe of U2a's hotpotqa lost-gold records looked at the texts that link to each lost gold over the hyperlinks. It is
not a stage, and it chose nothing; the choice below is made on s1sel only. On s1sel's 105 lost golds outside CAP_CUT,
it found the gold's title, or one of u1d's forms of it, in a linking text for 73 under u1d's boundary test, 76 with
the boundary fixed, and 95 with the title prefixes added. Two defects in u1d's matching showed up:

- **The boundary test.** `u1b.bounded` copies re's `\b`: a surface edge that is itself punctuation must touch a word
  character. A title ending in "Jr.", "F.C.", "Inc.", "Bros." or "!" therefore matches almost nowhere ("Louis Gossett
  Jr.", "D.R.I. (band)", "Just Shoot Me!", "Ashland Inc.", "Where Is the Love?", "The Venture Bros.").
- **The forms.** Most ABSENT golds are named in the linking text by the first words of their title ("Leeds United" for
  "Leeds United F.C.", "Siouxsie" for "Siouxsie and the Banshees"), which no u1d kind produces.

## 2. The rule: u1d's, with four switches, each the same on every dataset

U1d's surfaces, case gate, capitalization statistic, AMBIG guard (50) and context choice (the highest cosine of a title
group's nodes to the source passage, the frozen dense vectors) are imported unchanged. The switches:

1. **Boundary "look":** an occurrence counts when the characters just outside it are not word characters (`str.isalnum`
   or "_"), whatever the surface's own edge characters are.
2. **Underscores:** a title's underscores are read as spaces (squad stores its titles in URL form: "New_York_City").
3. **Prefixes (kind 3):** for a title of several tokens (parenthetical and comma tail removed), each first-j-token
   prefix whose last token starts with a capital, trailing " ,:;-" stripped. u1d's floor of 4 characters applies.
4. **Descriptive sources:** a node whose text is its own title (a KB entity name) names nothing, and its text is not
   read. On metaqa and webqsp every node is such a node, so they get no mention links: their graph stays the KB
   triples, today's.

With the four switches off and u1d's variants, the code reproduces u1d's filed musique surfaces, capitalization counts
and the edges of all five u1d variants exactly (gate 1, section 6).

**Variants** (which forms link; "exact-first": where a surface has a full-title owner, only those owners):

| variant | kinds | exact-first | gate |
| --- | --- | --- | --- |
| W2 | 0 (the title) | no | u1d's |
| W5e | 0, 1 (title without parenthetical or comma tail) | yes | u1d's |
| W6e | 0, 1, 2 (the last capitalized token) | yes | u1d's |
| W3e | 0, 1, 3 (prefixes) | yes | u1d's |
| W7e | 0, 1, 2, 3 | yes | u1d's |
| W7 | 0, 1, 2, 3 | no | u1d's |
| W3eg | 0, 1, 3 | yes | capmid |
| W7eg | 0, 1, 2, 3 | yes | capmid |

The **capmid gate** (for COMMON): a form gated as common still links where the text writes it exactly as titled,
capitalized, mid-sentence ("... and Apple sued").

**Graph forms** of each variant, all undirected and collapsed, rows in node-id order like `m3b_pools.FamilyStore`:

- **M:** the mention links;
- **D:** without the links into a node with more than 1,024 mention links in (U1e's damping, for DAMPED);
- **S:** M plus sibling links, between descriptive nodes that share a title (passages of one article);
- **DS:** D plus the sibling links.

That makes 32 candidates (variants in the order above, forms M, D, S, DS within each). The references, beside:

- **PKG:** the package's own structural family (M3B's store). That is the hyperlinks on hotpotqa and 2wiki, c3's title
  links on musique and squad, and the KB on metaqa and webqsp.
- **V2_M and V2_D:** u1d's rule V2, rebuilt on Modal with u1d's switches, undamped and damped.

## 3. Grading: the pool each graph serves

For each question, B is the frozen construction's base pool together with its seeds (`m3b_compile.prepare`, unchanged;
per dataset as `candidate_contract_frozen_2026_09_13`). X is B's one-link neighbours in the graph, outside B, ranked by
cos(q, v) with the frozen dense vectors, ties broken by node id. The pool at k is B ∪ X[:k], the order P1g2 used. For
k in 25, 50, 100, 250, 500, 1000, 2000 and 5000, U2b records:

- ALL (all of a question's golds in the pool);
- recall (per-question means, as the frozen I's tally takes them);
- rows (|B| + min(k, |X|)).

Beside these: all of X (one link's upstream ceiling), B alone, and the frozen I over the graph (`m3b_pools.expand_hops`
with the dataset's setting and its other families unchanged). A question's scores are computed once, over the union of
the X's of the graphs graded together, so every graph ranks by the same numbers.

**Amending U2A section 7's criterion.** U2A said U2b is "chosen on s1sel by I's gold reach". I's first hop takes each
seed's first 25 neighbours in node-id order, so on a denser graph I's reach is cut by an order that has nothing to do
with the question: that is CAP_CUT itself, which U2A pairs with "the neighbour order". On top of that, squad's I reads no
links. U2b therefore grades the pool a graph serves under the question's order, and reports I beside, unchanged.

## 4. The choice

- **Score:** the mean, over the four passage datasets (musique, squad, hotpotqa, 2wiki), of the mean ALL of B ∪ X(k)
  over k in 100, 250, 500, 1000 and 2000, on s1sel.
- **Pick:** among the 32 candidates, the earliest in declared order whose score is within 0.002 of the best.
- **Universal:** the same candidate is the graph on every passage dataset. A per-dataset best is reported beside, as an
  upper bound only.
- **Outcome:** `CHOSEN <graph>` when gate 2 holds and every candidate has been graded on all four datasets; otherwise
  `REFUSED`, naming what is missing.
- **Beside:** s1eval is reported for the chosen graph, V2_M, V2_D and PKG. metaqa and webqsp are graded on PKG, which
  is their graph under the descriptive switch, as U2c's starting point.

A pick closes only the forms graded here. Should no candidate reach the hyperlinks' pools, the next forms are named in
section 7.

## 5. Where it runs

The corpus passes (5.2 and 6.0 million passages for hotpotqa and 2wiki) run on Modal, which the user authorized on 22
October. The stages are stat, merge, link and build, one container per shard. They use a new volume holding a copy of
the package's nodes and dense vectors plus all U2b outputs; no existing volume's data is written. The question side (B,
seeds, golds, query vectors, the frozen I's other families) is computed on the laptop by `zu2b.py questions`, which
slices the frozen top-1000 lists as `m3b_compile.prepare` does. It is then uploaded with its sha256 record, and every
Modal stage checks the sha256 of what it reads. musique's, squad's and metaqa's question records name an earlier draft
of `zu2b.py` (sha256 `e355fe1d…`): that draft's `questions` command is the one committed here, unchanged, and the
other three datasets' records are written by the committed file.

The dense vectors on Modal are exact. A probe found that an older store on Modal holds some rows differently from the
package: 49,418 of 5,233,329 on hotpotqa and 150,902 of 5,989,847 on 2wiki. Each final shard is rebuilt from that store
through the package's own pointer index, with a patch of exactly those rows taken from the local package. A rebuilt
shard is kept only if its np.save bytes have the package manifest's sha256 and length. musique's and squad's shards are
uploaded and hashed. All four pass on every shard (131/131, 150/150, 3/3, 1/1).

## 6. Gates

1. **Gate 1** (laptop, before this commit): with every switch off and u1d's variants, zu2b's surfaces, capitalization
   counts and edges equal u1d's filed musique records (`outputs/u2b/check_u1d_musique.json`, status PASS). It reads no
   gold.
2. **Gate 2** (Modal, before the choice): over PKG, the frozen I must equal step 4h's filed G0 I, and over the
   Modal-built V2_M, the filed GU I. This holds on musique, hotpotqa and 2wiki, on each carve graded. The check reads the
   question arrays, the graph and the frozen construction; if it fails on a dataset, that dataset's U2b numbers are
   REFUSED.

## 7. Outputs and what follows

Outputs:

- `outputs/u2b/<D>/grade/<carve>_<tag>.json`;
- `outputs/u2b/<D>/graphs/build_<rule>.json` and `outputs/u2b/<D>/<rule>/capstat.json`, fetched from the volume;
- `outputs/u2b/<D>/questions*.json`;
- `outputs/u2b/choice.json` and `outputs/u2b/report.md`.

Arrays (npz) stay uncommitted.

**U2c** rebuilds today's pools on the chosen graph for all six datasets. It covers the X order, the k, compression to
the fewest rows with ALL at least 0.95 on every dataset including webqsp, the KT input and the lean pair's inputs. Its
rule is declared on its own. It also carries the **upstream-failure ledger**: every s1sel gold outside the chosen graph's
one-link reach, by cause. That is the list of golds no pool built on one link can hold.

If the chosen graph does not reach the hyperlinks' pools on hotpotqa or 2wiki, the next forms are a second link from
the question's best X rows, and the question-guided hop of U2A section 7. They come in that order, each declared before
its numbers.
