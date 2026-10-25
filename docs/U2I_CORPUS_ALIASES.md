# U2i: corpus aliases (step 3 of the hyperlink-gap plan)

Declared 23 October 2026, before any number. It follows [U2H_FULL_RUN.md](U2H_FULL_RUN.md), section 6 ("Always: the
hyperlink-gap plan runs beside this full run, each step in its own declared file"), and the user's instruction of 23
October: "yeah do the 3 steps to fix". Code: `outputs/mp_unified/zu2i.py`. Outputs: only under `outputs/u2i/`.

Nothing trains. No filed graph, pool, fit or record changes: every graph here is new, beside today's and W2_S.

## 1. The question

U2b's LLM-free rule (W2_S) reaches the package's own links on musique and squad, but not the hyperlinks of hotpotqa
and 2wiki (docs/U2B_LINK_RULE.md, section 8):

| dataset | s1sel, W2_S | s1sel, hyperlinks |
| --- | --- | --- |
| hotpotqa: ALL of B ∪ X, all of X | 0.921 | 0.989 |
| hotpotqa: the choice score | 0.909 | 0.972 |
| 2wiki: ALL of B ∪ X, all of X | 0.901 | 0.930 |
| 2wiki: the choice score | 0.819 | 0.848 |

With W2_S in place of the hyperlinks, the lean pair's R@5 falls by 0.048 to 0.069 on hotpotqa and by 0.008 to 0.027
on 2wiki (docs/U2G_GRAPH_AND_RULE.md, section 10). U2g sets at most 0.024 of hotpotqa's loss down to reach (the R@5 cap)
and the rest to ranking: the second gold, the bridge, drops out of the top five.

U2a (docs/U2A_LINK_GOLD_DIAGNOSTIC.md, section 7) sorts the golds the hyperlinks keep and the LLM-free links lose:

| cause | hotpotqa s1eval | 2wiki s1eval |
| --- | --- | --- |
| ALIAS: a text names the gold by a shorter form of its title | 0.56 | 0.64 |
| ABSENT: the text names the gold by a form not derivable from its title | 0.33 | 0.04 |
| CAP_CUT: the link exists; the node-id cap of 25 cuts it | 0.04 | 0.31 |

The plan has three steps, each declared in its own file:
1. link precision (per-link features for both models);
2. a hyperlink scorer learned on 2wiki and applied unchanged to the other datasets;
3. corpus aliases.

This file is step 3, which goes first because it changes the graph that steps 1 and 2 score. A Wikipedia-style passage
often states its subject's other names in its own first sentence: "Leeds United Football Club ...", "..., also known
as ...", "North Atlantic Treaty Organization (NATO)". U2i reads those names from the corpus and links them like
titles. The rule:
- reads only each passage's title and its first sentence, so it carries to any titled corpus;
- reads no gold, no hyperlink, no question and no external knowledge base;
- is the same on every dataset.

## 2. The aliases

**The first sentence** of a passage is its text up to the first ".", "!" or "?" that is followed by white space or the
end of the text and preceded by a lowercase letter, a digit, ")", "]" or a closing quote. It is cut at 600 characters.
Only descriptive passages are read (U2b's switch: a node whose text is its own title names nothing).

**Tokens** are the white-space-separated words. A token is *capitalized* when its first character is an uppercase
letter or a digit. The *connectors* are: of, the, and, de, du, des, la, le, les, von, van, der, den, del, della, di,
da, y, e, &, al, el, bin, ibn, for, upon.

**Three cues**, each read from the first sentence. Each alias belongs to the passage's title group (U2b's group: the
title with underscores read as spaces, stripped).

- **K4a, the restatement.** The opening span is the text before the first of " (", " is ", " was ", " are ", " were ",
  " has ", " had ", ", ", " – ", " — ", " - ", "; " or ": ". There is no alias if no separator occurs. With its
  quotes stripped, the span is an alias when all of these hold:
  - it has 1 to 8 tokens;
  - its first token is capitalized and, lowercased, is not a stop word other than "the" ("In 1990, ..." gives
    none);
  - no token is a month name ("On April 18, 1988, ..." gives none);
  - every token is capitalized or a connector;
  - at least one token has 3 or more letters and is not a stop word (u1b.STOP);
  - it shares a non-stop word of 3 or more letters with the title (lowercased, parenthetical removed), **or** the
    title group has exactly one descriptive passage. A single passage is the article's lead, so its opening span
    names the subject even when no word is shared ("Samuel Langhorne Clemens ..." under "Mark Twain").
- **K4b, the cue phrases.** After each occurrence of one of these phrases, alone or followed by ":" or ",":
  - also known as, better known as, commonly known as, otherwise known as, formerly known as;
  - known professionally as, professionally known as, known as;
  - also called, commonly called, often called, or simply, nicknamed;
  - stage name, pen name, born, née, nee, formerly, officially, short for;
  - stylized as, stylised as, abbreviated as, abbreviated.

  A phrase counts only where it does not follow a letter, a digit or a hyphen ("Polish-born" holds no cue). The alias
  is a quoted string (opened by a double quote, “ or ‘), or else the longest run of tokens that starts with a token
  whose first character is an uppercase letter, not a month name ("born 12 July 1955" gives no alias), and goes on
  through capitalized tokens and connectors. A lowercase "the" right after the cue is skipped ("also called the
  North Atlantic Alliance"). Trailing connectors are dropped, and a token ending in ",",
  ";", ":", ")" or "." ends the run (its punctuation stripped). Up to three aliases are read after one cue, joined by
  " or " or ", or ".
- **K4c, the bracket acronym.** A token inside a parenthetical of the first sentence, of 2 to 12 characters, starting
  with an uppercase letter and holding at least two uppercase letters. Its uppercase letters must equal the initials
  of the title's tokens (parenthetical removed), or of K4a's opening span, skipping connectors: "(NATO)" after "North
  Atlantic Treaty Organization", "(OECD)" after "Organisation for Economic Co-operation and Development".

**Every alias also** must:
- be at least 4 characters (U1d's MIN_LEN), at most 80 characters and at most 8 tokens;
- start with a capitalized token;
- differ from its group's title and from that title's U1d kind-1 forms (the title without its parenthetical, and the
  text before its first comma), with or without a leading "The ". Those forms are already the title's own.

## 3. The links

An alias is a surface of a new kind, 4, owned by the groups that state it. It links under W2's gates, U2b's rule
unchanged:
- the look boundary;
- the case gate (the occurrence is written as the alias is, apart from its first letter);
- the capitalization statistic: a lowercase form written more often in lowercase than capitalized mid-sentence, over
  the whole corpus, is common and never links;
- the AMBIG guard: a surface owned by more than 50 groups does not link;
- exact-first: an alias that is any group's full title is a title, and W2 already links it, so it adds nothing;
- the context choice: among 2 to 50 owner groups, the group whose nodes have the highest cosine to the source passage,
  with the frozen dense vectors (u1d's choice; ties go to the first group in id order).

A link goes to every node of the chosen group except the source itself. A group whose only node is the source is not
an owner.

**The graphs**, all undirected and collapsed, rows in node-id order as U2b's:

| graph | links |
| --- | --- |
| W2_S | W2's title links and the sibling links, rebuilt by this pass (gate 1) |
| A_c | W2_S and K4c's links |
| A_b | W2_S and K4b's links |
| A_a | W2_S and K4a's links |
| A_abc | W2_S and all three cues' links |

PKG, the package's own structural family (the hyperlinks on hotpotqa and 2wiki), is graded beside them as the
reference. metaqa and webqsp have no descriptive passages, so U2b's switch leaves them their KB, and no cue applies. The
four candidates equal today's graph there by construction, and nothing is graded.

## 4. The pass and its gates (laptop)

One dataset at a time, on the laptop, which holds the whole package. The lab host may be lost (23 October), and the
Modal credit goes to U2h's full run.
1. **Stream 1** reads every title and every descriptive passage's first sentence, and mines the aliases (section 2).
2. **Stream 2** scans every descriptive passage once against one table of the titles (kind 0) and the aliases (kind 4).
   The scan is zu2b.scan's code:
   - U2b's u1b.TitleTable matching and the look boundary;
   - the case gate, the capitalization counts and the capmid set.

   For each occurrence it also records the evidence of section 7. The scan writes its case-passing hits in chunks of
   200,000 passages, each chunk with its own capitalization counts, so an interrupted pass resumes at its first missing
   chunk.
3. **The links.** After the pass, the counts give the common forms (lower > capital, U2b's rule), and the hits give the
   links of section 3.

The gates:
- **Selftest** (`zu2i.py --selftest`), on toy passages:
  - zu2i's scan returns zu2b.scan's hits, counts and capmid set;
  - zu2i's W2 links equal zu2b.link_pass's W2 edges;
  - the cues give the aliases written into the test.
- **Gate 1 (before any grade).** On each passage dataset, the rebuilt W2_S equals the filed W2_S, array for array
  (indptr and col). The filed graph comes from the U2b links volume (`/u2b/<D>/graphs/`, the file named in
  `outputs/u2b/<D>/graphs/build_new.json`). It is fetched to `outputs/u2i/<D>/ref/` and checked against that record's
  sha256. A dataset that fails gate 1 is REFUSED, and none of its numbers is read.

## 5. The grading

For each passage dataset, on s1sel, the graphs W2_S, A_c, A_b, A_a, A_abc and PKG are graded with U2b's grading,
imported unchanged:
- **The frozen I over the graph:** m3b_pools.expand_hops with the dataset's construction and its other families, as
  zu2b_modal.grade runs it. It gives ALL, recall and the mean pool.
- **The reach of B ∪ X(k):** zu2b.reach_curves, B = the base rows and the seeds, X = B's one-link neighbours ranked
  by cos(q, v). It gives ALL, recall and rows for k in 25 to 5,000, and all of X.
- **The choice score:** zu2b.choice_score, the mean ALL of B ∪ X(k) over k from 100 to 2,000.

The question records are U2b's (`outputs/u2b/<D>/questions_<carve>.npz`), checked against their sha256. The dense
vectors are the local package's, read through u1d.Dense.

**The gap share.** On hotpotqa and 2wiki, for each measure m (I's ALL, all-of-X ALL, the choice score):
`share = (m(graph) − m(W2_S)) / (m(PKG) − m(W2_S))`. A share is reported only where the denominator is at least 0.005.
On musique and squad, W2_S already reaches PKG (U2b, section 8), so there is no gap to share. Their measures are
reported as changes from W2_S.

s1eval is graded beside, for W2_S, PKG and the chosen graph, after the choice.

## 6. The choice

- **Score:** U2b's: the mean over the four passage datasets of the choice score, on s1sel.
- **Best:** the candidate with the highest score; among candidates within 0.002 of it, the first in the declared order
  A_c, A_b, A_a, A_abc.
- **ADOPT_FOR_POOLS** when, for the best:
  - its score is at least W2_S's plus 0.002;
  - no passage dataset's choice score falls below W2_S's by more than 0.002;
  - no passage dataset's I-ALL falls below W2_S's by more than 0.002.
- **NO_GAIN** otherwise. It closes only these three cues as graph links (SCREENS 3b), and section 8 names the next
  forms.
- **REFUSED** when gate 1 fails on a passage dataset, or a candidate is missing a grade.

## 7. Recorded for steps 1 and 2 (deciding nothing here)

The pass writes, per dataset (arrays uncommitted, their sha256 in the record):
- **The aliases:** the surface, the group, the cue (a, b or c).
- **Every case-passing hit:**
  - the source passage and the surface;
  - the first occurrence's character offset and the number of occurrences;
  - the index of the sentence holding the first occurrence (0 = the first sentence);
  - the capmid flag;
  - for an alias, the cue.
- **The capitalization counts** of every surface's lowercase form, and each node's text length.

From these, steps 1 and 2 compute every link's features without another corpus pass:
- exact title or alias;
- the first mention, and whether it is in the lead sentence;
- the mention count;
- the title's rarity (its mention links in, and its common-word statistic).

Steps 1 and 2 declare their own grading before reading them.

## 8. What follows (fixed now)

- **ADOPT_FOR_POOLS:** U2e's pool rule (kt_cos:ada_f12_k0@0.85) is rebuilt on the chosen graph. Its coverage is
  graded against U2e's, and the model cell (the lean pair and zg6 on the new graph) follows when GPU credit allows.
  Each step is declared in its own file.
- **NO_GAIN:** the alias links stay out of the graph. Next, in its own file: they become their own edge family beside
  W2_S, an input to both models. Step 2's scorer also reads them as candidates.
- **Always:** step 1 (link precision, U2j) and step 2 (the learned hyperlink scorer, U2k) read section 7's records and
  are declared before their numbers.

## 9. How it runs

    python outputs/mp_unified/zu2i.py --selftest
    MODAL_PROFILE=<workspace 3> modal volume get mpr-u2-links /u2b/<D>/graphs/<file> outputs/u2i/<D>/ref/
    python outputs/mp_unified/zu2i.py pass --dataset D        # streams 1 and 2, the links, gate 1
    python outputs/mp_unified/zu2i.py grade --dataset D --carve s1sel
    python outputs/mp_unified/zu2i.py choose
    python outputs/mp_unified/zu2i.py grade --dataset D --carve s1eval --graphs W2_S,PKG,<chosen>
    python outputs/mp_unified/zu2i.py report

Laptop timing, from a probe of the kind-0 scan: 0.75 ms per passage and 4.1 GB resident on hotpotqa. That is about 70
minutes for hotpotqa's pass and somewhat more for 2wiki's.

## 10. Results

Filed after the run: `outputs/u2i/<D>/pass.json`, `gate1.json`, `grade_<carve>.json`; `outputs/u2i/choice.json` and
`outputs/u2i/report.md`.

**Filed 25 October 2026: NO_GAIN, best A_c** (outputs/u2i/choice.json and report.md). Gate 1 passed on all four
passage datasets. The score is the mean choice score over the four, on s1sel:

| graph | score |
| --- | ---: |
| W2_S | 0.8461 |
| A_c (K4c, bracket acronyms) | 0.8461 |
| A_a (K4a, restatement) | 0.8449 |
| A_b (K4b, cue phrases) | 0.8431 |
| A_abc (all three) | 0.8427 |

A_c ties W2_S, and no candidate reaches W2_S plus 0.002.

- **The aliases reach golds but cost the ranking.** On every dataset the alias links raise what all of X holds, and
  they lower the I and the choice score.
  - musique (A_abc): all of X goes from 0.7542 to 0.7705, its rows from 5,588 to 8,088, and the I's ALL falls
    from 0.7751 to 0.7575.
  - 2wiki (A_a): the share of the hyperlink gap closed is +0.068 in all of X, but −0.143 in the choice score.
  - hotpotqa (A_abc): +0.059 in all of X, −0.123 in the I.
  The new neighbours sit in X's cosine ranking and the I's 25-per-seed cap beside the golds already reached, and
  they push some of them out.
- **A_c adds few links**: 24,721 on musique, 4,169 on squad, 320,710 on hotpotqa and 442,765 on 2wiki. It changes
  nothing that is graded.
- **s1eval, beside** (choice score, W2_S against PKG): musique 0.7052 against 0.7008; squad 0.9888 against 0.9850;
  hotpotqa 0.9019 against 0.9740; 2wiki 0.8474 against 0.8775. The hyperlink gap holds on s1eval: 0.072 on hotpotqa
  (0.063 on s1sel) and 0.030 on 2wiki (0.029).

**Next, as section 8 fixes:**
- The alias links stay out of the graph.
- In their own file, they become their own edge family beside W2_S, an input to both models, when GPU credit allows.
- Step 2's scorer (U2k, filed 2ee7446) did not read them as candidates; U2k left the alias feature out because it
  is 0 on every W2 link. Reading the alias links as U2k candidates is that file's next form.

A verdict closes only the form tested (docs/SCREENS.md, section 3b): here, the three cues as graph links.
