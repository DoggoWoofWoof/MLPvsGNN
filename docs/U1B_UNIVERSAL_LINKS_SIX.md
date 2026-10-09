# U1b: one link rule on all six full corpora

Declared 10 October 2026, before any number. The second of three stages (U1a in docs/U1A_B1_WITHOUT_HYPERLINKS.md):

1. U1a: B1b's fits read on B1's settings without hyperlinks, and with the title-mention rule;
2. **U1b**: the same rule on all six full corpora, scored first on how well it recovers the hyperlinks;
3. U1c: retrain and grade on all six, zero-shot included.

U1b trains nothing, reads no question and chooses nothing. It builds one graph per dataset and describes it. U1c is the
stage that uses the graphs; it gets its own file.

## Why

The user, 9 October: the headline must not depend on the 2wiki and hotpotqa corpus hyperlinks, and the links have to
be universal, "irrespective of dataset". Today the six `structural` families come from four different sources:

| dataset | structural today | made by |
| --- | --- | --- |
| musique, squad | title mentions | CRAG scratchpad/c3_derived.py (the rule below) |
| 2wiki, hotpotqa | Wikipedia hyperlinks | the corpus's authors |
| metaqa, webqsp | KB triples | the dataset's KB |

## The construction (one for all six)

`structural_U` = the dataset's own structure that is not a hyperlink + the title-mention edges.

- **Own structure.** KB triples on metaqa and webqsp, kept as they are (relation ids unchanged). None on the four
  passage corpora: their hyperlinks are dropped, and musique's and squad's structural family is itself the rule's
  output.
- **Title-mention edges.** c3_derived's rule, unchanged:
  - an edge A → B iff A's lowercased text contains B's stripped, lowercased title as a phrase, bounded by `\b` on
    both sides (Python `re`, Unicode);
  - titles shorter than 4 characters never match;
  - B's index token is the longest of its title's `[A-Za-z0-9]+` tokens that is not a stop word and is over 2
    characters (else its longest token). An edge needs that token among A's text tokens. A title with no such token
    never matches;
  - A ≠ B, and a pair is kept once.
- **A node's title** is its `title` field. A KB node has none; its title is its text, the entity name.
- **Relation ids.** On the passage corpora every edge is relation 0, as the package's title-mention graphs are. On a
  KB, mention edges take one new relation id after the KB's vocabulary (`title_mention`).

There is no parameter fitted to anything: the 4-character floor and the stop list are c3_derived's.

**The matcher.** c3_derived scans every title indexed under each token of a text. That took 15 minutes on musique's
117,534 passages and does not scale to 2wiki's 5,989,847. U1b finds the same edges another way:
- A regex match always puts the title's lowercased `[A-Za-z0-9]+` runs into the lowercased text as consecutive whole
  runs. The `\b` at each end and the title's own separators bound every run.
- So a lookup of the text's run n-grams in a table of the titles' run tuples yields a superset of c3_derived's edges.
- Each candidate is then kept iff c3_derived's two conditions hold:
  - the `\b`-bounded occurrence exists, checked at every occurrence with `\w` = `str.isalnum()` or `_`, as `re`
    defines it;
  - B's index token is among A's text tokens.

The result is c3_derived's edge set exactly. Two checks guard this:
- a self-test runs c3_derived's loop and the matcher on the same documents, including accented, underscored and
  punctuated titles, and requires equal sets;
- the identity checks below.

## Checks, before any number is reported

- **Identity: musique and squad.** The rule on the package's own nodes must give the package's `structural` edge set
  exactly (as sets of directed pairs). The package built them with c3_derived from the same titles and texts. A
  difference stops U1b, and the differing pairs are reported.
- **Inputs.** The package freeze's RECORD_SHA256 must be the declared one (configs/m3b_controlled_comparison.yaml).
  Each `structural.npz` read must have its GRAPH_MANIFEST sha256. `nodes.jsonl` is recorded by size and by the line
  count the manifest names.

## What U1b reports, per dataset

1. **Hyperlink recovery** (2wiki and hotpotqa, reported first): precision and recall of the mention edges against the
   package's hyperlinks, directed and as unordered pairs.
2. The graph: edges, mean degree, isolated nodes, the largest in- and out-degree, and the 20 most-mentioned titles
   with their in-degrees.
3. KBs: mention edges added, and the share of them whose unordered pair is already a KB triple.
4. Against today's `structural`: edges and isolated nodes before and after.

A dense hub (a short common title matched by a large share of passages) is reported as found. U1b does not cap or
filter it. Any cap would be a parameter. It would come with its own declaration, be the same for all six, and be judged
by U1c, not by U1b's numbers.

## How it runs

- **Laptop:** the self-test; then musique and squad in full (the identity checks); then a cost sample on the first
  20,000 passages of 2wiki and of hotpotqa against their full title tables. The sample sizes the shards and the disk;
  its edge counts are a cost estimate, not a result.
- **Host:** the declared mirror (configs/host_mirror_six.yaml `host.mirror_root`, substituted for
  `substrate.package_root` in memory only). Every dataset is split into passage-range shards. Each shard reads all
  titles and its own texts, and writes its edges. A merge step per dataset joins the shards, checks that no pair
  repeats, writes the graph and reports.
- Before shards are queued, the cost sample's edges per passage set the expected bytes. They are held until at least
  twice that is free above the 100 GB floor.

## Files

- `outputs/mp_unified/u1b.py` (`selftest`, `shard`, `merge`, `report`)
- `outputs/u1b/<dataset>/shards/`: the shards' edges and records (untracked)
- `outputs/u1b/<dataset>/graph_structural_u.npz`: `src`, `dst` int32, `rel` int16, sorted by (src, dst)
  (untracked)
- `outputs/u1b/<dataset>/build.json`, `outputs/u1b/report.json`, `outputs/u1b/report.md` (committed)

## Results

Run 10 October 2026, 00:00-00:50: musique and squad on the laptop, the other four on the host (25 shards, 4 merges).
Records: outputs/u1b/<dataset>/build.json, outputs/u1b/report.json, outputs/u1b/report.md.

**Checks.** The self-test passed. Identity is EQUAL on musique (2,744,076 edges) and squad (874,190): the matcher gives
c3_derived's edge set exactly, in 31 s against c3_derived's 920 s on musique. Every input sha256 matched.

**Hyperlink recovery** (directed; unordered is within 0.01):

| dataset | rule edges | hyperlinks | precision | recall |
| --- | ---: | ---: | ---: | ---: |
| 2wiki | 205,604,282 | 28,963,600 | 0.099 | 0.705 |
| hotpotqa | 76,327,595 | 15,367,541 | 0.138 | 0.688 |

**The graphs:**

| dataset | structural today (edges) | structural_U (edges) | isolated nodes, today -> U |
| --- | ---: | ---: | --- |
| musique | 2,744,076 | 2,744,076 (identity) | 0.084 -> 0.084 |
| squad | 874,190 | 874,190 (identity) | 0.314 -> 0.314 |
| 2wiki | 28,963,600 | 205,604,282 | 0.086 -> 0.002 |
| hotpotqa | 15,367,541 | 76,327,595 | 0.105 -> 0.001 |
| metaqa | 133,582 | 160,520 (+26,938 mentions; 4.2% already KB pairs) | 0 -> 0 |
| webqsp | 8,309,195 | 330,247,304 (+321,938,109 mentions; 1.4% already KB pairs) | 0 -> 0 |

What this says:

- **The rule finds about 70% of the hyperlinks on the full corpora**, as on B1's settings (U1a: 0.79-0.81), but at
  a precision of 0.10-0.14. On B1's small settings it was 0.66-0.76: a title table of 5-6 million titles holds
  thousands of common words and phrases.
- **The hubs are common words.** The most-mentioned titles are "From", "With", "Also", "That", "Born", "Which",
  "It Was" (2wiki: 1.6 million in-edges each), "That", "Which", "County", "District" (hotpotqa). On webqsp they are
  Freebase schema words that the KB's node texts repeat ("Record" 1.6 million, "Topic", "Film", "Notable for"). Mention
  edges are 39 times webqsp's KB.
- **So structural_U as built here is not usable on 2wiki, hotpotqa and webqsp.** It is 7 times (2wiki), 5 times
  (hotpotqa) and 40 times (webqsp) today's structural family, and mostly hub edges.

**Next: U1d** (its own file) keeps the recall and removes the common-word hubs with one rule for all six. A design
look at the first 30,000 passages, against their hyperlinks only (no question read), shows:
- a case-sensitive match plus a corpus statistic (a surface the corpus writes lowercase more often than capitalized
  mid-sentence is a common word, not a name) halves the edges and doubles the precision on hotpotqa at the same
  recall;
- on 2wiki the statistic, estimated on the sample alone, lost recall. U1d estimates it on the full corpus.
