# The project, explained from scratch

Written 14 October 2026 for a reader who has never seen the project. It explains every idea we use with small worked
examples, then what we did with it, what we found, and why. Examples marked *(illustration)* are made up to show how
something works; every other number comes from a filed record, and each part ends with the docs that hold the
detail. All numbers are development numbers: the paper's come from one confirmation run later.

The same work, date by date, is in [PROJECT_TIMELINE.md](PROJECT_TIMELINE.md). The latest numbers are in
[WHERE_WE_STAND.md](WHERE_WE_STAND.md).

**Parts:** 1 the problem · 2 the building blocks · 3 what we set out to show · 4 how we decide · 5 pools · 6 links ·
7 models · 8 generalization · 9 speed · 10 where we stand · 11 glossary.

## 1. The problem, in one example

Take a library of about six million Wikipedia-style passages and this question *(illustration)*:

> "Who is the mother of the director of the film *The Long Road*?"

No single passage answers it. It takes two:
1. the film's passage, which names its director, say Ana Ruiz;
2. Ana Ruiz's passage, which names her mother.

A search engine compares the question with every passage. It finds the film's passage easily, because the title is
in the question. It rarely finds Ana Ruiz's passage, because nothing in the question mentions her. But the film's
passage **links** to hers. So if we start from what the search finds and follow the links, we can reach the second
passage. The passages a question needs are its **golds**, and a system is scored on how many golds it puts in its top
five.

So the project has three steps, and every part of this document is about one of them:

```
question -> 1. search (the "first stage")   -> a few hundred to a few thousand candidates
         -> 2. the graph (links between candidates) -> grow and trim them into a "pool"
         -> 3. a model reranks the pool      -> the top 5 we report
```

**The research question is about step 3.** There are two ways a model can use the links while it ranks:

- **A GNN (graph neural network)** lets candidates "talk" to their linked neighbours while it scores them. A
  candidate's score is updated from its neighbours' at question time. This is called **message passing**. It is
  powerful, but every question pays for that extra work over the graph.
- **An MLP (a plain neural network)** scores each candidate on its own, from one row of numbers that describes it. The
  links can still help, but only as numbers computed beforehand ("this candidate is one link from the top search
  result"), never by candidates exchanging information while they are scored. It is simpler, and should be cheaper.

**The question:** how much of the GNN's benefit can an MLP keep without message passing, and is it really faster? If
an MLP keeps nearly all of the gain at a fraction of the cost, anyone building a retrieval system can drop the GNN.

## 2. The building blocks

### 2.1 Six datasets, two kinds

Four datasets are **passage corpora**: the answer is written in Wikipedia-style passages, and a candidate is a
passage. Two are **knowledge bases (KBs)**: the answer is an entity (a film, a person), entities are joined by typed
facts such as *directed_by*, and a candidate is an entity.

| dataset | kind | what a question needs | questions we score |
| --- | --- | --- | ---: |
| squad | passages | one passage, which contains the answer | 11,873 |
| hotpotqa | passages (5.23 million) | two passages: a bridge, or two things compared | 7,405 |
| 2wiki | passages (5.99 million) | two or more passages, chained or compared | 12,576 |
| musique | passages | two to four passages, each hop built on the last | 2,417 |
| metaqa | KB (movies) | a chain of typed facts from the question's entity | 9,785 |
| webqsp | KB (2.59 million entities) | facts around the question's entity | 1,503 |

The corpora run from about 20 thousand to 6 million nodes. "Questions we score" is the fixed evaluation set (called
s1eval) that every model since 7 October is read on, counting only questions with at least one gold in the corpus.

Two of the six behave differently, and you will see them again and again:
- **squad** is easy for the first stage (it already gets 0.905 of the golds into the top five), so a graph has little
  to add.
- **webqsp never trains** in the lean models: none of them learns from it, so each reads it **zero-shot** (seeing that
  dataset for the first time). It is the honest test of whether a model carries over to new data. (The universal pair,
  2.7, is the exception: it trains on all six, so it has no zero-shot read.) Its gold list is also
  incomplete: only 0.5255 of its gold references exist in its corpus, and the 46 questions with none are left out.

### 2.2 Step 1, the first stage: two searches, fused by rank

Two searchers each rank the whole corpus for the question:
- **Dense:** a frozen text encoder turns the question and every passage into a vector, and ranks passages by how
  close their vectors are (similar meaning).
- **SPLADE:** ranks by weighted word matches (shared words, with learned weights).

They are good at different things, and their scores live on different scales, so we fuse them by **rank** alone with
**reciprocal rank fusion (RRF)**: a candidate scores 1/(60 + its rank) in each list, added up.

*Worked example (illustration).* Dense ranks P1, P7, P3; SPLADE ranks P3, P1, P9.

| candidate | dense rank | SPLADE rank | RRF score | RRF order |
| --- | ---: | ---: | --- | ---: |
| P1 | 1 | 2 | 1/61 + 1/62 = 0.03252 | 1 |
| P3 | 3 | 1 | 1/63 + 1/61 = 0.03227 | 2 |
| P7 | 2 | (absent) | 1/62 = 0.01613 | 3 |
| P9 | (absent) | 3 | 1/63 = 0.01587 | 4 |

A candidate both lists like beats one only a single list likes. The 60 damps the top ranks, so rank 1 is not worth
much more than rank 5. The **seeds** are the dense top 5 plus the SPLADE top 5: where the walks in section 2.5 start.

**Why RRF:** it needs no training and no tuning to put two different score scales together, so it is the honest
"before" line. Every gain in this project is measured over it.

### 2.3 How we score a ranking: R@5, FC@5 and hit@1

*Worked example (illustration).* A question has two golds, G1 and G2. Two systems return these top fives:

| system | top five | R@5 | FC@5 | hit@1 |
| --- | --- | ---: | ---: | ---: |
| A | G1, x, x, x, x | 1/2 = 0.5 | 0 | 1 |
| B | x, G1, x, G2, x | 2/2 = 1.0 | 1 | 0 |

- **R@5 (recall at 5):** the share of the question's golds that are in the top five. This is the main measure, and
  Claim 1 is written on it.
- **FC@5 (full coverage at 5):** 1 only if *every* gold is in the top five. For a two-hop question, a reader needs
  both passages, so this is the measure that says the question can be answered. (GraphER calls it PR@5.)
- **hit@1:** 1 if the very first result is a gold. It matters when only one result is shown.

Each is averaged over the questions. A system can be good on one and bad on another: B finds everything but puts a
non-gold first. That is exactly what the project's "hit@1 gap" on hotpotqa is about (section 7).

### 2.4 Step 2, the graph: three kinds of links

The nodes are the corpus's passages or entities. A **link** (edge) joins two nodes. We use three families, all built
without an LLM:
- **Structural links**, which the data already has: KB facts on metaqa and webqsp (*The Long Road* →directed_by→
  *Ana Ruiz*), Wikipedia hyperlinks on 2wiki and hotpotqa, and title mentions on musique and squad (a passage that
  writes another passage's title).
- **Named-entity links:** two passages that mention the same named entity, such as the same person or place.
- **Nearest-neighbour (kNN) links:** each passage linked to the passages whose vectors are closest to its own.

The state of the art builds its graph differently. HippoRAG 2 asks an LLM to read every passage and write out facts,
about 11.2 million LLM passes on 2wiki and hotpotqa. Ours uses links that are in the data or come from simple rules.
Section 6 tells how we chose them and what they buy.

### 2.5 Walks: how the graph finds what the search missed

A **walk** is a random walker on the graph. It starts at the seeds, and at every step it either jumps back to a
seed (with probability 0.5, the **restart**) or follows a random link from where it stands. Nodes it visits often get
a high score. This is **personalised PageRank (PPR)**; "personalised" means the restarts go to this question's seeds,
not to random nodes.

*Worked example (illustration).* The seed is the film's passage F. F links to Ana Ruiz's passage A and to a passage
about the film's studio S. A links to her mother's passage M.

```
   F (seed) ---> A (director) ---> M (mother)
      \
       `-------> S (studio)
```

Half the walker's moves are restarts to F, and from F it goes to A or S with equal odds. So A and S get visited often,
and M (two links away) less, but much more than any of the millions of nodes the walker never comes near. Ranked by
visits: F, then A and S, then M, then everything else at 0. A was nowhere in the search's list, and now it is near
the top. That is the whole point of the graph.

Our real walks do the arithmetic exactly instead of simulating a walker. A node keeps half of the mass it receives
(the restart share) and passes the other half to its neighbours. This **push** repeats until what is left at every
node is below 1e-7 times its number of links. *In numbers (illustration):* F keeps 0.5 and passes 0.25 to each of A
and S. A keeps 0.125 and passes 0.0625 to M and 0.0625 back to F, so M keeps about 0.031. Each hop halves what is
left, so a two-hop gold gets about a quarter of a one-hop node's share. On a walk over several link families, each
step first picks a family at random, so on three families only a third of the flow takes any one family's links.
Section 9 explains why this step is the slow part.

### 2.6 The pool

The **pool** is the shortlist a model reranks: the first stage's candidates plus nodes the walks or links add. A gold
outside the pool can never be found, whatever the model does. Two measures of a pool:
- **ALL:** the share of questions with every gold in the pool.
- **Gold recall:** the share of all golds that are in the pool.

**A limit no pool can lift.** R@5 can never pass min(golds in the pool, 5) ÷ golds. *Example (illustration):* a
question with 10 golds, 8 of them in the pool, has gold recall 0.8, but no ranker can get its R@5 above 5/10 = 0.5.
metaqa has up to 246 golds on one question (7.69 on average) and webqsp up to 222 (5.93). So on these two the top
five sets the limit before the pool does: 0.170 of metaqa's R@5 and 0.114 of webqsp's is out of every model's reach.
Mixing up the two limits invents headroom no model could reach.

A bigger pool holds more golds but costs more time, and it gives the model more wrong answers to get confused by.
The user's rule for the final pool (13 October) was: **the most golds in the fewest nodes, one parameter-free rule
for all six datasets, nothing fitted to golds.** Section 5 is the story of how we got there.

### 2.7 Step 3, the rerankers: MLP and GNN

Every candidate in the pool gets a **row** of numbers (its **features**, or columns). For example *(illustration)*:
its dense and SPLADE ranks, whether it is a seed, how many links join it to the seeds, its walk score, how well the
relation names on its path match the question's words.

**The MLP** reads one row and outputs one score. Each candidate is scored alone; the top five scores are the answer.

**The GNN** reads the same rows, then lets each candidate update itself from its linked neighbours, one or more
times, before its final score. This step is **message passing**.

*Worked example (illustration), one propagation step.* Suppose the scores after reading the rows are F 0.9, A 0.2,
S 0.3, and the step adds 0.5 times the best linked neighbour's score:

| candidate | own score | best linked neighbour | after one step |
| --- | ---: | --- | ---: |
| F | 0.9 | A (0.2) | 0.9 + 0.5 × 0.2 = 1.00 |
| A | 0.2 | F (0.9) | 0.2 + 0.5 × 0.9 = 0.65 |
| S | 0.3 | F (0.9) | 0.3 + 0.5 × 0.9 = 0.75 |

A rises because its neighbour is strong. The real step is learned (which neighbours count and how much), but the idea
is the same: a candidate borrows strength from its neighbours **at question time**. The MLP cannot do that. It can
only read a number computed *before* scoring, such as "A is one link from a seed".

**The MLP rule (8 October), with the example that taught it.** An MLP input may never come from another candidate's
score. On 14 October one of our MLP heads read its row's score *z-scored over the question's rows*, that is (own score
− the mean of all the rows' scores) / their spread. No links are involved, but the mean and the spread come from
every other candidate's score: if you change B's score, A's input changes. Every row then "hears" every other row,
which is message passing over a fully connected graph. That model (zrck) was set aside, and a version that reads only
its own row (zrckp) replaced it (section 7).

**The twin and δ_MP.** To measure what message passing alone buys, we train a **twin**: an MLP given exactly the
GNN's rows, training questions and loss, with the message passing taken out. **δ_MP** = GNN minus twin. Anything
else that differs between them would muddy the comparison.

**Two families of models:**
- **Universal** (twin0 the MLP, gnn0 the GNN): the same 129 columns for both, chosen from the 197 the feature compile
  builds by screens on the training questions only (they dropped only near-duplicate columns, those with a rank
  correlation of 0.98 or more with a column kept). A clean apples-to-apples pair.
- **Lean** (today zrckp the MLP, zspk the GNN): each model with the inputs that serve it best, built for accuracy
  and speed. Section 7 tells how each came about.

**Training, in one paragraph.** A model learns its weights from training questions whose golds are known. The loss
rewards scoring golds above non-golds in the same pool. A lean model is trained on five datasets together (webqsp never
trains); the universal pair trains on all six. Either way one model must find rules that work everywhere, not tricks
for one dataset.

*Worked example (illustration), the loss.* The loss turns a pool's scores into probabilities with a softmax (e to the
score, divided by the sum of e to the score over the whole pool) and charges −log of each gold's probability. Scores
2, 1 and 0 give probabilities 0.665, 0.245 and 0.090. If the gold is the first row, the loss is −ln 0.665 = 0.41; if it
is the last, −ln 0.090 = 2.41. To lower it, training must raise the gold above the rest of *its own pool*, which is why
it is called **listwise**. With several golds, the loss is their average.

## 3. What we set out to show: three claims

The claims were fixed on 9 October ([PAPER_CLAIMS_2026_10_09.md](PAPER_CLAIMS_2026_10_09.md)), before the numbers
that test them.

**Claim 1: the MLP keeps at least 90% of the GNN's gain.** The gain is measured over the first stage (RRF), on R@5:

    share = (MLP − RRF) / (GNN − RRF)

*Worked example, real numbers (2wiki, the lean pair, 14 October):* RRF 0.6079, GNN 0.9351, MLP 0.9304. The GNN gains
0.3272 and the MLP 0.3225, so the share is 0.3225 / 0.3272 = 0.986: the MLP keeps 98.6% of the GNN's gain. The claim
asks for this on each dataset in-domain (on datasets it trained on), with a 95% interval (section 4.4) that stays at
or above 0.90.

It also has a **zero-shot clause:** on a dataset a model never trained on, the MLP must not fall further below the
GNN than it does in-domain. *Example (illustration):* if the MLP is 0.005 below the GNN in-domain, then 0.012 below on
the held-out read is an excess of 0.007, a failure if the interval is clear of 0.

**Claim 2: a GNN built with no LLM anywhere is at the state of the art.** The bar is HippoRAG 2 on its own benchmark
(called B1), and on our full corpora the best of eight LLM-free rankers, five published ranking rules rebuilt on our
graphs plus the dense, SPLADE and RRF lists.

**Claim 3: the MLP is faster, measured cold.** Cold means each question from scratch: nothing cached, nothing warmed
up, one thread. The user's target: "mlp should be at minimum 2-3x faster than gnn across all families". Section 9
explains why this is the hardest claim.

## 4. How we decide: the method

Most of what we did is only as good as the way we decided it. These are the rules, with examples.

### 4.1 Development numbers and one confirmation run

Every number in this document is a **development number**: chosen on development questions, often after trying many
versions. Picking the best of many versions flatters it (some of its lead is luck). So the paper's numbers come from
**one declared confirmation run** of the final models, on questions never used for any choice. The datasets' official
test splits are never read.

Within development, questions are split into **carves**: a **fit** carve the model learns from, a **select** carve
for choices such as when to stop training, and the evaluation carve (**s1eval**) every comparison reads. A model
never trains on the questions it is scored on.

*Real example of a carve that misled us (musique, early October).* MuSiQue builds its training questions out of shared
single-hop questions, and its dev split uses none of them. So 0.945 of the questions in the old select carve held a
passage that was gold for some training question, against none in dev. Learned models scored R@5 0.745 to 0.761 on
that select carve (the first stage 0.538), but only 0.491 to 0.521 on dev (the first stage 0.473). About 0.18 of the
select carve's R@5 did not carry to dev, beyond the first stage's own drop: memory, not skill.

Step 1 (7 October) built select carves shaped like the evaluation (on musique, sharing no single-hop question with the
fit) and compared the two ways of picking: NOT_ADOPTED. The pick rarely moved, and neither carve predicted a held-out
dataset. Even an oracle that picked the best of 36 saved states on the evaluation itself reached only 0.148 on metaqa
and 0.312 on musique held out. The conclusion: "The zero-shot gap is not a selection problem." Since then every fit is
read at its averaged final weights, with no pick at all.

### 4.2 J5 and the leave-one-out fits

- **J5** trains on five datasets at once (everything but webqsp).
- **L-x** ("leave x out") trains on four, leaving dataset x out. Reading x on L-x is **zero-shot**.

*Example:* L-musique trains on metaqa, squad, hotpotqa and 2wiki. Its musique read says how a model does on a dataset
it has never seen; its hotpotqa read is in-domain. With J5 plus five leave-outs, there are 6 fits × 6 datasets = 36
reads. Every dataset but webqsp is read both trained and zero-shot; webqsp is zero-shot on all six fits.

### 4.3 Screens and full runs

Training one fit takes about 25 minutes; a full set of six takes hours. So an idea goes through two gates:
- **A screen:** two fits (L-musique and L-hotpotqa), each read on all six datasets: 12 reads. If it passes it is
  **ADOPT_FOR_FULL_RUN** (earlier, PROMISING).
- **A full run:** all six fits, 36 reads, of which 11 are **primary**: J5 on each trained dataset and on webqsp, and
  each L-x on x. A change is **ADOPT**ed if at least one primary read gains and none of the 36 loses.

Why two fits per screen: in early October, two ideas passed one-fit screens and then failed their full runs on reads
the single fit could not see. A second fit catches more.

### 4.4 The bootstrap: how sure is a number?

A difference between two models measured on a few thousand questions could be luck. The **bootstrap** asks: if we
had drawn a different set of questions, how much would the number move?

*Worked example (illustration).* Five questions, and the MLP's R@5 on each:

| question | q1 | q2 | q3 | q4 | q5 | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R@5 | 1.0 | 0.5 | 0.0 | 1.0 | 1.0 | 0.70 |
| one resample: q2, q2, q5, q1, q3 | 0.5 | 0.5 | 1.0 | 1.0 | 0.0 | 0.60 |

A resample draws five questions *with replacement* (q2 twice, q4 not at all) and recomputes the mean. Repeat 2,000
times and sort the 2,000 means. The value 2.5% of the way up (about the 50th from the bottom) and the value 97.5% of
the way up (about the 50th from the top) make the **95% interval**. To compare two models, both are read on the
*same* resamples, so the luck of the draw cancels in their difference.

*Paired (illustration):* the new model minus the old on the same five questions is +0.5, 0, +0.5, 0, −0.5, a mean of
+0.1. Resampling questions 1, 3, 3, 5, 2 gives +0.2; resampling 4, 4, 2, 5, 1 gives 0. With five questions the
interval would straddle 0, so the +0.1 could be luck. Real reads have 1,503 to 12,576 questions, and their intervals
are narrow.

*Real example (P3c, 14 October):* on squad, the GNN gains only +0.0045 over RRF (interval +0.0012 to +0.0077). The
MLP's share of a gain that small is a ratio of two small numbers, and its interval runs from 0.86 to 2.82. Even
though the MLP's mean is above the GNN's there, "at least 0.90" cannot be shown.

### 4.5 The seed null: how much does luck in training move a number?

Training is random too: the starting weights and the order of examples depend on a **training seed**. 0.0075 of R@5
was the lean models' measured one-seed noise, in-domain. But in early October every full run was dying on a "LOSS" of
0.008 to 0.014, and four different ideas all "lost" the same read, hotpotqa zero-shot. If the ideas did nothing, four
losses in four would happen about one time in sixteen. More likely, the base's own seed had drawn high on that read.

So we ran the **seed null:** the unchanged base retrained with seeds 1 and 2 on every fit, giving two seed-only moves,
Δ1 and Δ2, on each of the 36 reads. Each read's **floor** is twice the typical seed-only move, and never below 0.0075:

    floor = max(0.0075, 2 × √((Δ1² + Δ2²) / 2))

*Real examples:*
- musique zero-shot (L-musique): reseeding moved R@5 by +0.0503 and −0.0079. √((0.0503² + 0.0079²) / 2) = 0.036, so
  the floor is 0.072.
- squad in-domain (J5): −0.0024 and −0.0021, a typical move of 0.0023, so the floor stays 0.0075.

17 of the 36 floors came out above 0.0075: 9 of the 11 zero-shot reads, against 8 of the 25 in-domain ones. A model is
much less stable on data it never saw. A change smaller than reseeding is not a change, and the floor works both
ways: it turns a small GAIN into WITHIN as surely as a small LOSS. Its price: on a noisy read such as musique
zero-shot, a real change below 0.072 cannot be seen either.

### 4.6 The verdicts

Each read of a change against its base is called one of:
- **GAIN:** up by at least the floor, with its 95% interval above 0;
- **LOSS:** down by at least the floor, with its interval below 0;
- **WITHIN:** anything else.

One LOSS anywhere blocks adoption, even with gains elsewhere. That is deliberate: a change that helps four datasets
and hurts the fifth is not general. Stages that test something other than a model have their own calls, fixed in
their docs: HOLDS / NOT_SHOWN / MISSES for a claim, STOP for a direction that closes, NO_SIGNAL for a diagnosis that
finds nothing.

*Real case (zrckp, 14 October):* held out from musique, zrckp read −0.0114 [−0.0198, −0.0033] against its base. At
the plain floor of 0.0075 that is a LOSS. Under that read's seed-null floor (0.072) it is WITHIN. The full run was
ADOPT: 5 of the 11 primary reads gained, and none of the 36 lost.

### 4.7 Declared before the numbers, and the same for every dataset

- **Every stage has its own file, committed before any of its numbers exist.** It fixes what is read, the rule, and
  what each outcome means. Then the numbers cannot bend the rule. *Real example (P4c, 14 October):* the first timing
  ran while the laptop was busy. A rule to re-time all six datasets idle was committed at 20:28, before any number
  was read, and the re-timed numbers were filed. The two passes agreed within 3.3%.
- **Every change is universal:** the same change on all six datasets, graded on all six, including zero-shot. No
  dataset gets its own switch in the ranking rule or the models. Three per-dataset choices remain in the pipeline
  (the pool sizes, squad's question links and a two-name list of typed KBs); section 5.6 lists them with their fixes.
- **Fixed for good:** the text encoder is never changed or fine-tuned; no LLMs or transformers inside our models;
  the MLP has no message passing.

### 4.8 Where evaluation questions touched our choices

A rule is only as good as its exceptions, so here is every place where the questions we report on also shaped a
choice:
- **The pool sizes (HIGH).** They were set on the evaluation questions' golds, and the final pools keep them (5.6).
- **A second read (LOW).** KIND_T's rule (5.4) was designed after an earlier rule's call had read s1eval, so KIND_T's
  call there is a second read of the same questions. Its choice was still made on the select carve alone.
- **Step 4e (LOW)** read all six datasets' evaluation questions to choose its arm, webqsp's included.
- **B1's questions are inside s1eval (MEDIUM; found on 14 October, while writing this).** HippoRAG 2's benchmark
  questions (Claim 2) are rows of the dev splits that s1eval holds whole: 1,000 of musique's 2,417 questions, 1,000 of
  hotpotqa's 7,405 and 1,000 of 2wiki's 12,576. No model trained on them, and nothing was chosen on the B1 runs
  themselves. But every screen and full run that chose our models read them, on our own corpora, and so did earlier
  gates read on the whole dev splits (the universal pilot's 2wiki cells, for one). The claims doc
  planned to answer "tuned on the test questions" with an even/odd split of B1's questions, and that split was never
  applied. **The proposed check:** re-call every adoption decision behind the models we report without B1's
  questions. If every call stands, our choices did not depend on them. Until then, the B1 numbers carry this caveat.
- **August's screens (Era 1, past).** They were scored on the test split of the older, smaller copy of the data.
  Nothing from that era serves a claim today, and the rule since then is that no test split is read.

## 5. Pools: getting the golds onto the shortlist

**Why it matters.** A gold that is not in the pool is lost, whatever the model does. A pool that is too big costs
time and gives the model more wrong answers to choose from. So the goal is **the most golds in the fewest nodes**.

### 5.1 August: search lists alone fail on knowledge bases

The first pool was the dense top 200 plus the SPLADE top 200. On the passage corpora that was fine. On the two KBs it
was not, and longer lists barely helped. Growing the budget from 50 to 400 rows (about 340 to 370 distinct candidates)
raised the share of golds in the pool only from 21.4% to 33.5% on metaqa, and from 31.6% to 49.1% on webqsp. On
today's full corpora, even the dense list's top 1,000 holds only 0.154 of metaqa's golds (SPLADE's 0.293), against
0.776 to 0.998 on the passage corpora.

*Why (illustration):* "Which films did the director of *The Long Road* also direct?" The answers are other film
titles, and nothing in the question's words resembles them. A KB answer sits a few facts away from the question's
entity, and a search for similar words cannot see facts. Longer lists did not help.

### 5.2 September: follow the right links, and cap the expansion

A graph audit gave the project its founding fact: **"kNN edges supply local density, structural edges supply
reach."** Through structural links, a path from the seeds reaches every metaqa and webqsp gold. Through kNN links,
only 0.515 and 0.338 of them. Growing the pool along structural links raised the reachable R@5 by 35 points on metaqa
and 15 on 2wiki; growing it along kNN links recovered no gold at all. A capped expansion recovered the same golds
with about as much text as before, where an uncapped one needed up to 56 times as much.

**"Today's pools" (13 September)** froze one pool per dataset: the first stage's top N plus a few hops along links,
with the smallest pool reaching 0.90 of the best top-5 recall reachable (the "knee rule"). For example:

| dataset | today's pool | rows |
| --- | --- | ---: |
| hotpotqa | RRF top 50 + 1 structural hop | 94 |
| metaqa | RRF top 50 + 3 structural hops | 2,017 |
| webqsp | dense top 200 + 3 hops on every family (no setting reached the knee) | 2,120 |

(webqsp's base is the dense list alone: there, fusing it with SPLADE lowered the ceiling, 0.316 against 0.358.)

ALL ran from 0.688 (webqsp) to 0.980 (squad). Two problems remained. A fixed hop treats every neighbour alike, a
gold and a dead end the same. And each dataset got its own recipe, sized on the evaluation questions' golds. That is
still the audit's one HIGH item: the final pool keeps these sizes (5.6).

### 5.3 Early October: walks instead of hops

- **Step 4 (7 October): one walk over every link family mixed together.** It diluted the structural hop: ALL fell on
  metaqa (0.895 → 0.820), hotpotqa (0.968 → 0.897) and 2wiki (0.906 → 0.824). On hotpotqa and 2wiki, 0.91 and 0.88 of
  the golds the search missed were one structural hop from the seeds, but the walk sent two thirds of its flow down
  the other two families (2.5). **Fix (4b):** walk each dataset's own expansion graph, the families its pool already
  used. ADOPT: musique +0.032 and webqsp +0.094 of ALL, the rest level.
- **Step 4e (8 to 9 October): the published recipes.** Three restarts were tried. **A1** is HippoRAG 2's: half the
  restart goes to the seeds, the other half to the first stage's top 1,000, shared by RRF weight. **A2** adds
  HippoRAG's down-weighting of hub nodes (it added nothing). **A3** adds PullNet-style exact name linking: a node whose
  name appears in the question, such as "tom hanks", becomes a start node too. A3 was chosen, with a fill of twice the
  pool's budget. ALL rose to 0.909-0.992 (webqsp 0.688 → 0.909, musique 0.803 → 0.937), but the pools were about 3.5
  times bigger on metaqa, musique and webqsp.
- **Step 4f** tried to cut them back and could not. No single ranked order held the same golds in half the nodes:
  "the coverage lives in the union itself, not near the top of any one list". **Step 4g** found where the remaining
  misses were: 78 to 89% sat one hop outside the pool on metaqa, musique and webqsp, and on 2wiki and hotpotqa every
  first hop into a missed gold was a hyperlink.
- **P1a (13 October): one order, and a bar to beat.** A pool should be one ranked order cut at a size, so that pools
  can be compared fairly at the same size. P1a ranked each question's nodes by our order and by LLM-free rebuilds of
  the published ones: A1, A2, the walk from linked entities that KB question-answering systems use (**LINKED**), and
  the plain lists. The best of them at each size is the **SOTA pool bar**. Kept whole, our step 4e recipe had looked
  best. As one order cut at today's pool sizes it was BELOW the bar on five datasets: ALL 0.857 against 0.973 on
  hotpotqa, 0.772 against 0.911 on 2wiki, 0.800 against 0.961 on metaqa. *Why:* on passages, the linked names took over
  the head of the order (on hotpotqa at 100 nodes, the walk from the linked names alone held 0.617, against 0.947 from
  the seeds alone). On the KBs, the lists took half the restart, and they rank look-alike names, not answers.

### 5.4 13 October: the final rule, KIND_T

The user set the rule at 10:40: pools and links must be **"dataset agnostic and parameter free and should have the
most golds in least nodes"**, and nothing is learned until the pools are final. We had built gates that learned, per
question, how to mix the walks (called MX and RC). RC reached the bar on four datasets, and an oracle choosing among
its blends passed all six. But both were fitted to the very golds they are judged on, so under the rule they became
reference lines only. MX also taught a lesson: it was fitted to put walk mass on the golds, while a pool is judged by
the golds' rank, and mass is not rank.

KIND_T is built in five steps, with nothing fitted:
1. **Name links, but only when corroborated.** The question is linked to nodes whose names it mentions, but a link is
   trusted only if the first stage also retrieved that node (it is in the dense or SPLADE top 100). *Why:* linking is
   precise on the KBs and loose on passages. A median metaqa question links 2 nodes, and 98% of the links are in the
   lists' top 100. A median hotpotqa question links 24, and only 15.5% are. Untrusted names flooded the walk (P1a).
2. **Decide the question's kind from the data.** A node is *name-only* when its text is just its name (a KB entity
   such as "Tom Hanks"; a passage is not). If every start node is name-only, it is an entity question; otherwise a
   passage question. No dataset name is read. On these six the test acts almost as a KB switch: every metaqa node is
   name-only, 38.8% of webqsp's, about 2% of hotpotqa's and 2wiki's, and none of musique's or squad's. But a seventh
   dataset made of entity records would be routed the same way without being named.
3. **Entity questions** whose name links are all corroborated ("on the route") walk from those links on the
   structural links, topped up by HippoRAG 2's A1 walk where that walk stops short. The others use the A1 walk on the
   structural links. *Why:* on a KB the links name the question's own entities, so the walk from them should lead.
4. **Passage questions** fuse five orders with RRF (2.2): the first stage's own order plus four walks (the A1 walk and
   the walk from the corroborated links, each on the structural links and on all link families). This fusion is
   called **RRF5**; with the steer of step 5 it is **RRF5T** (T for steered).
5. **The steer.** A plain walk splits a node's mass equally over its neighbours in each link family. The steered walk
   splits it in proportion to each neighbour's first-stage RRF weight, w(v) = 1/(60 + dense rank) + 1/(60 + SPLADE
   rank). A neighbour neither list holds gets the floor, as if ranked just past the lists' end (1,001st).

*Worked example of the steer (illustration).* Node F passes mass to two neighbours. A is ranked 30th by dense and 90th
by SPLADE; S is in neither top 1,000.
- w(A) = 1/90 + 1/150 = 0.0111 + 0.0067 = 0.0178
- w(S) = 1/1061 + 1/1061 = 0.0019
- A gets 0.0178 / (0.0178 + 0.0019) = 90% of F's mass, S gets 10%, instead of 50/50.

The walk now prefers neighbours the search also found a little relevant. *Real effect:* musique splits an article into
paragraphs that share its title (seven paragraphs titled "Paris", seven "World War II"), and a title mention links a
passage to all of them, so an unsteered walk floods them. On musique's fit carve, the walk from the corroborated links
held every gold for 0.418 of questions at 25 nodes steered, against 0.173 unsteered.

**No new constant was introduced:** the 60
is the first stage's, and the restart (0.5), the push threshold (1e-7) and the cut at 7,000 nodes are step 4e's.
With every weight equal, the steered walk is step 4e's walk, bit for bit (checked).

**The result** (ALL at today's pool sizes; RRF5 is the same fusion with no kind test and no steer):

| dataset | today's pools | RRF5 | KIND_T |
| --- | ---: | ---: | ---: |
| metaqa | 0.895 | 0.910 | 0.962 |
| squad | 0.980 | 0.979 | 0.979 |
| musique | 0.803 | 0.899 | 0.900 |
| hotpotqa | 0.968 | 0.963 | 0.965 |
| 2wiki | 0.906 | 0.961 | 0.960 |
| webqsp | 0.688 | 0.876 | 0.832 |

- **Most golds in fewest nodes:** KIND_T reaches today's coverage with about 37% of the nodes on metaqa, a quarter on
  musique and webqsp, and half on 2wiki. On hotpotqa it needs 150 rows where today's pools use 94. squad cannot change
  (its pool has no room to fill).
- **Against the state of the art:** the bar is the best LLM-free reproduction of a published pool order at each pool
  size. KIND_T is at or above it on squad, musique, 2wiki and webqsp, and just below on metaqa (by up to 0.005) and
  hotpotqa (by up to 0.007). The filed verdict is NOT_ALL_SIX. Against the like-for-like bar (the same link family),
  webqsp is below too, by up to 0.051.
- **Why we stopped there:** metaqa and webqsp pull the entity branch in opposite directions. It lifts metaqa by 0.052
  over RRF5 and costs webqsp 0.044. No further fixed rule was declared, and a learned one would break the 10:40 rule.

### 5.5 Did the better pools help the models?

Not at first. **P1c2** retrained both lean models on KIND_T's pools. There were real gains (metaqa +0.020 to +0.021,
2wiki +0.039 to +0.045 on both models), but one LOSS (zsp on webqsp zero-shot, −0.0167), so it was NOT_ADOPTED.

The record showed why. On musique and webqsp the new pools held more golds (gold recall +0.042 and +0.108), but the
extra golds were **ranked out** of the top five. Held out from musique, both models even ranked KIND_T's pool below
KIND_T's own order used as a ranker with no model at all (0.4789 and 0.4792, against 0.5025). The models had no way
to know which candidates the walks liked. The fix was to tell them, as eight extra columns (section 7.4).

### 5.6 What is still not right with the pools

From the timeline's honest audit (section 5 there):
- **The pool sizes (HIGH):** KIND_T keeps each question's size from today's pools (50 rows on squad up to 2,120 on
  webqsp), and those sizes were chosen per dataset on evaluation golds. Proposed fix: one size rule set on training
  carves only.
- **squad's question links are switched off by its name (MEDIUM):** squad's "titles" are article titles, shared by
  every paragraph of the article, so name links there are noise. Proposed fix: a data test, for example link a title
  only when it names exactly one node.
- **"Is this a typed KB" is a list of two names (MEDIUM):** the feature compile already works it out from the data;
  the rest should too.

Details: [P1E_ANCHOR_KIND.md](P1E_ANCHOR_KIND.md), [P1C2_POOLS_KIND_T.md](P1C2_POOLS_KIND_T.md),
[P4A_POOL_COLD.md](P4A_POOL_COLD.md), and the timeline's eras 2, 3, 6 and 8.

## 6. Links: which edges, and the hyperlink question

### 6.1 Three families, and why all three stay

- **Structural links carry the reach.** Without them, ALL falls by 0.540 on metaqa and 0.310 on 2wiki (P2a). Edges
  that encode real relations beat kNN edges on five of six datasets as early as August (metaqa 30.4 against 23.7).
- **kNN links add density, not reach:** they join passages that say similar things, which helps a model compare
  candidates but rarely leads to a new gold.
- **Named-entity links** join passages about the same person or place, which structural links can miss.
- **No one set of families suits all six** (P2a). Each family helps somewhere (on musique all three do), but adding the
  named-entity and kNN links to the structural ones hurts on metaqa, squad and hotpotqa. Three times (step 4, G1a,
  P2a) a walk over all families at once diluted the structural hop. So all three families stay, each is walked on its
  own, and the results are combined by a fixed rule (RRF5T).

### 6.2 The hyperlink question: is our lead just Wikipedia's links?

On 2wiki and hotpotqa our structural links are the corpus's own Wikipedia hyperlinks, which HippoRAG 2 does not
use. A fair reader would ask whether the hyperlinks, not our models, beat HippoRAG 2. We tested it on HippoRAG 2's
own benchmark (U1a to U1f):

| zsp on B1 (recall@5, %) | 2wiki | hotpotqa |
| --- | ---: | ---: |
| HippoRAG 2 | 90.4 | 96.3 |
| ours, with the hyperlinks | 97.2 | 98.2 |
| ours, with no links at all | 77.3 | 90.0 |
| ours, with one LLM-free title rule in place of the hyperlinks | 95.9 | 95.0 |

**The title rule:** a passage links to any passage whose title it mentions. *(Illustration:* the film's passage says
"directed by Ana Ruiz", and "Ana Ruiz" is another passage's title, so the two are linked.*)* It rebuilds the real
hyperlinks with precision 0.76 and recall 0.81 on 2wiki (0.66 and 0.79 on hotpotqa), and wins back 94% of their gain
on 2wiki and 61% on hotpotqa.

**What went wrong first:** common words that are also titles. A passage titled "From" or "County" got linked from
almost everything. Against the real hyperlinks the rule's precision fell to 0.10 to 0.14 on the full corpora, and on
webqsp it built 330 million edges, 40 times the KB (U1b). **Fix (U1d, then U1e):**
- match titles case-sensitively;
- never link a title that the corpus writes in lowercase more often than capitalised, away from sentence starts (so
  "From" never links: the corpus mostly writes "from");
- drop every mention link into a node that receives more than 1,024 of them (one cap for all six datasets).

On a 30,000-passage sample of hotpotqa, read against its hyperlinks only (a design look, no question read), the first
two steps doubled precision (0.138 to 0.285) at almost the same recall (0.676 to 0.669).

**The decision:** retraining our models on the title-rule graph was NOT_ADOPTED twice (U1c, U1f). Using it in place
of the hyperlinks costs R@5 on our corpora: 0.035 to 0.065 on hotpotqa and 0.010 to 0.020 on 2wiki. So the
hyperlinks stay, and every SOTA claim names them.
The fair headline: without any LLM, **a few precise edges** are what the models need. With no links the models fall
well below HippoRAG 2. With the title rule, 2wiki stays 5.5 points above it.

### 6.3 Can simple rules replace an LLM-built graph?

HippoRAG 2 asks an LLM to write facts (triples such as *Ana Ruiz — directed — The Long Road*) out of every passage.
We extracted triples with a standard parser (spaCy) and fixed rules instead (P2c). On B1 they match or beat the
LLM-built graphs in 7 of 8 cells, and on B1's musique they add more as a pool walk (+8.4 R@5, against +3.7 to +4.8
for the LLM-built graphs). But as a fourth link family on our full corpora they gave **NO_GAIN**, so the links stay
three families.

### 6.4 Do the words around a link say which way to go? (D4)

On musique, the gold a model misses is often a neighbour of a gold it found, among about 45 neighbours. Could the
words near each link (a title mention, a shared entity, a KB relation name) pick the right one? **NO_SIGNAL:** ranked
by those words fused with the model's own order, the missed gold lands in the top two 0.261 of the time, against
0.225 for the model alone (0.308 against 0.291 in a second fit). The bar, declared first, was at least +0.05 and at
least 1.25 times as often, in both fits. The fused order reached +0.036 (1.16 times) and +0.018 (1.06 times). So
musique's gap is reported as the first stage's: our list starts 13.3 points below NV-Embed-v2's on B1 (56.4 against
69.7 R@5), and our models already add about twice HippoRAG 2's lift (+9.9 against +5.0).

Details: [P2C_LLM_FREE_TRIPLES.md](P2C_LLM_FREE_TRIPLES.md),
[U1A_B1_WITHOUT_HYPERLINKS.md](U1A_B1_WITHOUT_HYPERLINKS.md), [DIAG_LINK_WORDS.md](DIAG_LINK_WORDS.md).

## 7. Models: from a fair comparison to the lean pair

### 7.1 August: the first lesson was fairness

The first MLP (QLS-MLP) read fixed structure summaries (distances, path counts, PageRank) plus the search seeds, and
passed its screen 3 of 3. Then we saw that only the MLP had been shown the seeds. Given the same inputs, the two
were about level (2wiki 68.40 against 69.85; webqsp 33.37 against 33.09; percentages, on the older, smaller copy of
the data). **Lesson:** an MLP-against-GNN comparison
means nothing unless both read the same inputs. That is why every later pair shares pools, carves and seeds, and the
universal pair shares every column.

### 7.2 September: where does the GNN's gain live?

- **M3B isolated message passing's own effect (δ_MP), 19 September.** Three arms shared the same 78 columns,
  projections, readout, loss and schedule: an MLP, a GAT (an attention GNN, two layers) and **GAT-NO-MP**, the
  identical GAT with every message edge replaced by a self-loop. GAT minus GAT-NO-MP is then message passing's own
  effect, with nothing else changed. *Real example, 2wiki (that day's pools, 106 candidates):*

  | measure | first stage | MLP | GAT-NO-MP | GAT | δ_MP |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | R@5 | 0.6079 | 0.8377 | 0.8335 | 0.8850 | +0.052 |
  | FC@5 | 0.2537 | 0.6444 | 0.6306 | 0.7564 | +0.126 |
  | hit@1 | 0.8754 | 0.8719 | 0.9037 | 0.8832 | −0.021 |

  Message passing helps get **all** of a question's golds in (FC@5 +0.126), not the first one (hit@1 −0.021). On R@5
  it adds +0.052 on 2wiki, +0.030 on hotpotqa and +0.018 on musique, and −0.011 on squad, the control.
- **The universal pair (21 to 28 September).** gnn0 is one GNN for all six datasets with no dataset identity: one
  relation-aware message-passing cell applied three times (420,932 weights), plus an "evidence flow" state that moves
  only along typed KB links. Its twin, twin0 (330,955 weights), reads the same 129 columns, embeddings and seeds. Its
  input block also reads the average of each link family's neighbours' frozen embeddings. That average is fixed before
  any row is scored (the learned projection comes after it), so no neighbour's score or learned state enters, and after
  it nothing passes between rows. Both train on all six datasets at once. Three things worth knowing:
  - The pilot's gate failed for both at seed 0: the GNN by a hair (squad 0.8988 against a bar of 0.8996), the twin on
    2wiki (0.8545 against 0.8578). It was filed as failed, with the bar unchanged. A three-seed replication of the GNN
    then held every cell, and that opened the six-dataset stage.
  - The evidence flow reads exactly 0 on all four passage graphs: they have no typed links for it to move along.
  - Against M3B's GAT (three seeds), gnn0 gained on metaqa (hit@1 +0.068), webqsp (hit@1 +0.037) and 2wiki (R@5
    +0.005), and was level on the rest.
- **Probes found the shape of the gain** (MP-Approx, 30 September and 2 October). A probe is a small model fitted to
  the difference between the GNN's scores and its twin's. If adding the probe's output to the twin wins back most of
  the GNN's gain, the probe's form is the form of the gain.
  - On passages the gain is a **one-hop, question-dependent reweighting of neighbours**. L3, a probe that does just
    that (attention over each row's direct neighbours, steered by the question), wins back 0.799 [0.759, 0.838] of the
    GNN's gain over its twin on 2wiki. The probe itself passes messages, so it locates the gain; it is not a model we
    serve.
  - On metaqa the same probe wins back nothing (−0.076): the gain there is **typed chains of 2 to 3 hops**. L15, a
    typed-walk model with no message passing, trained on the same labels, wins back 0.929 [0.902, 0.957] of it (read
    on 11,920 metaqa training-split questions that no model had trained on) and still sits 0.006 below the GNN on
    R@5.

**This shaped everything after:** if the gain is "one hop, weighted by the question" on passages and "typed chains" on
KBs, then an MLP can get most of it from inputs that describe those hops and chains, computed before scoring.

### 7.3 October: the lean pair, built screen by screen

About 30 one-fit screen rounds (section 4.3), mostly on the MLP. The ones that paid:
- **zret:** each feature block is normalised against the question's retrieved rows. *(Illustration:* a dense
  similarity of 0.62 means little alone; if the question's retrieved rows average 0.55 with a spread of 0.05, the row
  reads (0.62 − 0.55) / 0.05 = 1.4, "1.4 spreads above a typical retrieved candidate", which means the same thing on
  every dataset.*)* musique zero-shot rose from 0.270 to 0.390.
- **rmatch:** a learned match between the question and the relation names along a KB chain. *(Illustration:* for
  "Who directed the films Tom Hanks starred in?", the chain Tom Hanks ←starred_actors— film —directed_by→ person
  matches "starred" and "directed" in the question.*)* The relation names are read as text through the frozen
  encoder, so the match carries to relations never seen in training. metaqa rose 0.644 → 0.779, and webqsp zero-shot
  0.117 → 0.280, learned on metaqa's relations and carried to webqsp's. On its own it was blocked by one LOSS: 2wiki
  read zero-shot fell 0.0136, past its floor of 0.0112 (its hit@1 held; the loss was in ranks 2 to 5).
- **zrm** is rmatch trained together with zret's model, and it was adopted.
- **zrc, the MLP base:** zrm with a better choice of chain entries. For each row and seed group, rmatch keeps 64 of
  the chains that reach the row. zrm kept the 64 carrying the most walk mass; zrc keeps the 64 with the largest mass
  times the untrained match (how well the chain's relation names fit the question, from the frozen encoder, before any
  training). No new weight. Full run: ADOPT, no LOSS in 36 reads, webqsp zero-shot +0.051 on J5, and every other read
  within 0.006 of zrm.
- **zsp, the GNN base:** zrm plus **one propagation step** over the pool's links. For each link family, a row reads
  three summaries of its neighbours' z-scored zrm scores (their mean, a soft maximum, and how many there are), plus its
  own z-score: 10 numbers into a 385-weight head whose output is added to its score. *(Illustration:* three neighbours
  at 2.1, 0.3 and −0.5 give a mean of 0.63, a soft maximum of ln(e^2.1 + e^0.3 + e^−0.5) = 2.32, a little above the
  largest, and ln(1 + 3) = 1.39 for the count. Change one neighbour's score and this row's inputs change: that is what
  makes zsp message passing.*)* Full run: ADOPT, no LOSS in 36 reads; metaqa held out +0.030, 2wiki zero-shot +0.014.

**zret is allowed in an MLP; zrck was not.** zret compares a row's *inputs* (fixed numbers such as a dense
similarity) with the other retrieved rows' inputs. Those are known before the model scores anything, like a rank in
a list. zrck compared a row's *score* (the model's own output) with the other rows' scores, so each row's final
answer depended on what the model said about the others. That is message passing (section 2.7).

**How much does structure matter to the MLP?** A diagnostic MLP with every link-derived input removed (zn) keeps only
13 to 73% of the GNN's gain. So the MLP needs the graph, but as numbers computed before scoring.

### 7.4 14 October: the KT input, and the rule that caught a mistake

After P1c2 (5.5), both models were given **eight columns from KIND_T** per candidate (P1c3):
- six **positions**: its place in KIND_T's order, and in five walks: HippoRAG 2's A1 walk (on the structural links,
  and on all families), the walk from the corroborated links (on the structural links), and the steered walk from
  the links (on the structural links, and on all families);
- two **flags**: whether the pool's fill added it (rather than the first stage's base or the seeds), and whether the
  question is an entity question (the same on every row of the question).

The first stage's own order is left out, because the models already read the lists' ranks. Places count from 0, and
place p is scaled to log(1 + p) / log(1 + 7,000). So the first node reads 0, the 11th 0.27, the 101st 0.52 and the
1,001st 0.78, and a candidate the walk never reached reads 1.0. Small numbers mean "the walks like it". Half of the
range covers the first 101 places, so differences near the top count most.

- **P1c3's screen:** held out from musique, the columns won back about 0.6 to 0.7 of what KIND_T's pools had cost.
  On webqsp more of the pools' extra golds reached the top five: the first MLP version (zrck, below) kept 0.34 and
  0.76 of them in the two fits, against 0.08 and 0.26 without the columns.
- **P1c3's full run: ADOPT** for both models, with no LOSS in 36 reads each.
- **But the MLP's head broke the MLP rule:** besides the eight columns, it read the row's score z-scored over the
  question's rows (2.7). The user caught it: "That uses no edges, but it does use the mean and spread of the other
  rows' scores." So zrck serves no claim.
- **The fix, zrckp:** a pointwise head that reads only the row's own eight columns: 8 inputs, 16 hidden units and 1
  output, so 8×16 + 16 + 16 + 1 = 161 weights, and no score input. Its last layer starts at zero, so before training
  it is zrc exactly. A self-test checks that changing the other rows' columns leaves a row's output unchanged. Screen
  ADOPT_FOR_FULL_RUN; full run **ADOPT**: 20 GAINs and no LOSS in 36 reads, 2wiki +0.060 to +0.064 on every split,
  metaqa +0.108 when held out (0.142 → 0.249), webqsp zero-shot up on all six splits. The side costs: hotpotqa's
  hit@1 falls on every split (−0.002 to −0.034), and musique's FC@5 held out falls 0.026.
- **What keeping the rule cost:** against zrck, zrckp is level on 31 of 36 reads, above on 3 (metaqa held out
  +0.046) and below on 2 (webqsp zero-shot −0.013 and −0.021). Little, in short.
- **The lean pair is now (zrckp, zspk).** zspk's head may read the z-scored score, because it is the GNN.

### 7.5 What did not pay, and why we stopped

Each of these was a declared screen or stage, graded by the same rules (4.5, 4.6):

| idea | what it did | result |
| --- | --- | --- |
| Bigger model (S4) | 205,217 weights against S3's 3,585 | lost on musique at every repair; S3 kept |
| Deeper message passing (zg1, round 31) | a 4-layer learned state over the pool's links, on top of zsp (17,025 more weights) | 2wiki +0.006 and +0.007, just under its floor of 0.0075; webqsp zero-shot −0.016, a LOSS. NO_GAIN |
| Feature selection (rounds 24-25) | every one of the 128 subsets of seven candidate feature blocks | the best gained +0.0007: "The MLP-to-GNN gap is not in these columns." |
| Training longer (round 33) | twice the steps | nothing past reseeding noise; both models keep 8 epochs |
| Less data (D3) | half the training questions | lost 4 of 12 reads, but only 1 at the same number of steps: the steps mattered, not the questions |
| A learned space (zgf, round 26) | a head over a learned 64-wide space | musique −0.060, hotpotqa zero-shot −0.044. Leaving the head out at read time brought every lost read back, so the head was the harm |
| An offset as an input (zof, round 32) | an offset from the first stage's top five, set by the question | 7 of 12 reads lost (musique −0.087). NO_GAIN |

*Real example of a trade (bound1).* Keeping the model's correction within one spread of the first stage's score won
back most of musique read zero-shot (0.270 → 0.412, with the first stage alone at 0.473) but sank metaqa by 0.62, to
the first stage's level (0.031), because
metaqa's golds are the nodes the first stage ranks lowest. One bound cannot suit both a dataset where the first stage
is nearly right and one where it is nearly always wrong. bound1 was not adopted, and the lean models' correction
stays unbounded.

**Learned offsets** (the user's idea: reach a multi-hop answer by jumping from a seed by a learned shift, instead of
walking):
- **August:** offsets sharpened rank 1 (2wiki R@1 +12.5) but lost coverage on questions with several answers (R@20
  −10.1; percentages, older data).
- **P5b (October):** g(question, seed, graph features), 200,465 weights, is fused with a plain order H2 (the seeds'
  direct neighbours, then their neighbours, then the rest, each by cosine). Among the golds H2 ranks low, g's own top
  10 found 0.322 on 2wiki against H2's 0.136, but 0.007 on musique against 0.159, about chance (0.005). It had
  **memorized**: on its own training questions the loss fell to 0.88 (the golds took about 0.42 of its probability),
  and nothing checked it on held-out questions.
- *Real example of memorizing (P5b2):* with a held-out stop, the loss on held-out questions' new golds fell from 5.58
  to 5.19 by epoch 4, then climbed to 10.16 by epoch 30. The stop kept epoch 4. On new golds the fused order then found
  0.316 of 2wiki's against H2's 0.156, but 0.093 of musique's against 0.159.
- **P5b3:** a learned back-off gate (11 weights) decides, per question, how much to lean on H2 instead of g, from ten
  label-free signals such as how spread out g's choices are and how much g's top 25 overlaps H2's. It still lost
  musique by 0.005 to 0.011. The track closed: it found golds no training question had, but not on every dataset.

Details: [P1C3_KT_INPUT.md](P1C3_KT_INPUT.md), [P1C3P_FULL_RUN.md](P1C3P_FULL_RUN.md), [SCREENS.md](SCREENS.md),
[ZN_LEAN_NO_MP.md](ZN_LEAN_NO_MP.md), [P3C_CLAIM1_INTERVALS.md](P3C_CLAIM1_INTERVALS.md).

## 8. Generalization: does it work on data it never saw?

### 8.1 How we test it

Every model is trained six times: J5 and the five leave-outs (4.2). A dataset's leave-out read is its zero-shot read,
and webqsp is zero-shot on every fit. The rule since early October: **every change is graded on all six, zero-shot
included**, and one zero-shot LOSS blocks it. The yardstick for a zero-shot read is the first stage, not twin0 or
gnn0: those trained on the dataset, so they are in-domain references.

### 8.2 The numbers (R@5)

| dataset | zrckp trained on it | zrckp held out | zrc held out (before the KT input) | first stage alone |
| --- | ---: | ---: | ---: | ---: |
| metaqa | 0.7989 | 0.2494 | 0.1417 | 0.0047 |
| squad | 0.9112 | 0.8910 | 0.8912 | 0.9054 |
| musique | 0.5728 | 0.5121 | 0.5235 | 0.4734 |
| hotpotqa | 0.9065 | 0.8513 | 0.8451 | 0.6846 |
| 2wiki | 0.9304 | 0.8541 | 0.7937 | 0.6079 |
| webqsp | never trains | 0.1139 to 0.3686 | 0.0997 to 0.3384 | 0.0535 |

(zrckp reads KIND_T's pools with the KT input; zrc reads today's pools.)

How to read it:
- **The Wikipedia-style corpora carry over.** Held out, hotpotqa keeps 0.851 and 2wiki 0.854, far above the first
  stage (0.685, 0.608). What the model learned about hops and links on other corpora works on these.
- **squad held out lands just below its first stage** (0.891 against 0.905). There is little to gain on squad. Our
  reading, not a tested cause: a model trained only on multi-hop data gives weight to neighbours that squad's
  one-passage questions do not need.
- **musique held out is 0.512,** above the first stage but well below in-domain, and 0.011 below zrc held out: inside
  that read's 0.072 floor (4.5).
- **The knowledge bases are the hard case.** metaqa falls from 0.799 to 0.249 when held out. With no KB in training,
  no relation match is learned: in L-metaqa's fit "the match takes no gradient and its gates stay at zero". webqsp
  shows the same: 0.35 to 0.37 on the five fits that train on metaqa, but 0.114 on L-metaqa's, which has no KB in
  training. The KT input still lifted held-out metaqa from 0.142 to 0.249. Our reading: a walk's position means the
  same on any graph, so what the model learned about it on the passage corpora carries over.
- **webqsp's misses are mostly ranking.** Of the lean pair's webqsp R@5, 0.43 to 0.47 is lost to golds that are in the
  pool but ranked out, against 0.12 to 0.15 for the universal pair, which trains on webqsp. It is the largest such
  share in the gap ledger.

### 8.3 What helps a model carry over

- **Relation names as text** (rmatch, 7.3): a relation the model never saw still has a name, and a name has meaning.
  That is how webqsp zero-shot went from 0.117 to 0.280.
- **Normalising inputs per question** (zret): a feature means the same thing on every dataset.
- **Bounding message passing (the QD track's design looks, early October):** a message-passing term trained on one
  graph harmed graphs it had not seen. Two changes fixed that. The GNN's change to the base score is bounded,
  s = z + 0.2·tanh((s0 − z) / 0.2), with z the base score and s0 the unbounded one. And whole link families are dropped
  at random during training (the structural family always stays; reads see every link). *(Illustration:* a change of
  +0.05 stays about +0.049, but +1.5 is squashed to +0.200. Message passing can still settle close calls, but it
  cannot overturn the base order on a graph it never saw.*)* That arm was the first message-passing arm trained on one
  graph with no unseen graph below the twin on R@5. These were design looks, not graded results, and the lean pair
  does not use the bound.
- **Transfer without labels (S1 to S6, 4 to 5 October):** train on metaqa's 9 relations, read webqsp's thousands,
  with no webqsp label. Label-free rules that shift a model to a new KB failed: statistics alone cannot tell which new
  relation plays which old role. Only text, or which relations exist, can tie a learned cluster to a relation. What
  did carry was training on varied relation granularity. *(Illustration:* "directed_by" becomes four random aliases,
  so a scorer cannot memorise one relation id; or film → director becomes film → m → director, the way Freebase
  stores facts through mediator nodes.*)* An EM-trained chain scorer trained this way beat the untyped walk by 6.43
  R@5 on webqsp zero-shot, and the same scorer trained without the transforms by 10.7, at a cost of 4.8 points on
  metaqa in-domain. These were design looks on training-split rows, not graded results.

### 8.4 Would it work on a seventh dataset?

**The models would; the pipeline needs edits first** (the timeline's audit). A new dataset needs its package (nodes,
questions, frozen vectors, SPLADE, the three link families), a pool size (today set with gold labels), and entries in
a few lists that name the six datasets. Then KIND_T, the features and the models run unchanged. The universal pair has
never been tested on a held-out dataset; that is an open item.

**Nothing is copied.** Published methods (HippoRAG 2's restart, PullNet-style linking) are reimplemented from their
papers and named, and third-party data is used only to compare, never committed.

### 8.5 Case study: why musique is the hardest

musique chains two to four single-hop questions, so a question has two to four golds (2.65 on average), and each hop
builds on the last. Each of these was measured:
- **The mix shifts.** Dev is 52% two-hop, 31% three-hop and 17% four-hop questions; training is 72%, 22% and 6%.
- **The first passage is found, the later ones are not.** In the 5 October diagnosis, hit@1 was 0.58 to 0.68 at every
  hop count, but FC@5 (every gold in the top five) was 0.411 on two-hop, 0.034 on three-hop and 0.000 on four-hop
  questions. Message passing helped only the two-hop questions (δ_MP +0.046 R@5), not the three- and four-hop ones
  (−0.018 and −0.004).
- **Reach.** The old pools held every gold for only 48% of four-hop questions. Golds outside today's pools still cost
  0.079 of R@5, and 0.198 on four-hop questions.
- **Ranking.** Most misses are in the pool, near a found gold: 0.38 are its direct neighbours and 0.77 are within two
  hops. But the model ranks the missed gold 5th to 9th among about 45 neighbours, and the words near the links do not
  pick it out either (D4, 6.4).
- **The first stage.** On HippoRAG 2's musique setting our pools hold 98.2% of the golds, yet R@5 is 66.3 against
  HippoRAG 2's 74.7. Our first stage starts 13.3 points below HippoRAG 2's 7B encoder (56.4 against 69.7). Our models
  add 9.9 over it, twice HippoRAG 2's lift over its own encoder (5.0), which wins back 4.9 of the 13.3. 752 of the 990
  golds zsp misses there are in three- and four-hop questions. The encoder stays frozen, so this gap is reported as the
  first stage's.

This is also where Claim 1 needs work: the MLP keeps 93% of the GNN's gain on musique, but the low end of its interval
is 88%.

## 9. Speed: is the MLP really faster?

### 9.1 What latency is, and why the headline is cold

**Latency** is the time to answer one new question, from the first stage's lists to the top five. It has three
parts:
- **Index:** built once per corpus, before any question: the graph in a compact form, the node tables, and so on.
- **Query-local compile:** everything built for this question: the pool, the walks, the features of every row.
- **Forward:** the model's own arithmetic.

**Warm** times the forward alone, on inputs already built. **Cold** builds everything from scratch for each question,
with no cache and no warm-up. A real new question has no cache, so the headline is cold; a warm number may sit
beside it, labelled.

*Real example, musique, the lean GNN zsp (cold, median, ms):*

| pool | compile | edges | lean inputs | links | forward | total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.32 | 17.54 | 0.32 | 11.92 | 5.54 | 6.15 | 42.19 |

(Stage medians do not add up exactly to the total's median.) The forward, where message passing lives, is about 15%
of the time. Most of a question's cost is building its inputs, and both models pay for that.

**The lesson that set this rule, learned twice:**
- **September (Package D):** warm, the early MLP's operator was 4 to 8 times cheaper than the GNN's (0.113-0.139 ms
  against 0.446-0.953 ms). Rebuilt for each question, it was *slower* in 10 of 12 cells, because its own structure
  summaries had to be built per question too. The verdict: "the QLS advantage is a property of warm caching, not of
  the method."
- **10 October (C1):** an often-quoted "MLP 6 to 9 times faster" turned out to be warm, timing the forward only. Cold
  over the whole path, zrc took 31.0 ms on 2wiki against gnn0's 30.3, and 7.9 against 7.2 on squad: "Claim 3, as the
  models are served today, does not hold cold." The shared feature compile and the MLP's own walk inputs ate the
  lead.

### 9.2 How a number is made, and how the machine is kept honest

- **The questions:** the first 200 s1eval questions per dataset, one question at a time (batch 1), one thread.
- **The number:** the median question's time (p50). A ratio is GNN time ÷ MLP time at the median, so above 1 means
  the MLP is faster. Its 95% interval is a paired bootstrap (4.4): resample the 200 questions, with the same draw for
  both models, so a hard question counts against both.
- **Rotation:** the models take turns going first on each question, so no model always pays for a cold memory.
- **Pinning:** the laptop's processor has 2 fast cores and 8 slow ones, and Windows moves a background process
  between them and throttles it. One unpinned run swung 3 to 10 times. So the timing process pins itself to one fast
  core, opts out of power throttling and raises its own priority. It changes only itself, and the settings end with
  it.
- **Quiet:** the load is checked before each dataset, and nothing else runs during a timed pass. Twice this went
  wrong. C9's laptop was at 74% load (times 1.5-1.9 times too slow, so only its ratios were read). P4c's first pass
  overlapped commands from this session, so all six datasets were re-timed idle under a rule committed first; the
  passes agreed within 3.3%.
- **Memory:** 2wiki's node table is a 17.14 GiB memory map on a 16.8 GB laptop, so parts of it sit on disk. The
  model that ran first on a question paid for loading them (twin0's compile: 54.5 ms first, 2.5 ms otherwise), which
  is why the rotation matters.

### 9.3 Exact forms: faster, but the same answers

Every speed-up must give the same scores (within 1e-4, the same top five), and most are bit for bit identical. Nothing
is approximated: no pruning, no rounding of inputs, no cache across questions.

**Why "within 1e-4" and not always "bit for bit":** computer arithmetic on decimals is not exactly associative.
*Example:* in float32, 2^24 + 1 rounds back to 16,777,216. So (2^24 + 1) + 1 = 16,777,216, but 2^24 + (1 + 1) =
16,777,218. A faster form that adds the same numbers in another order can differ in the last digits. Even the
reference moves: the matrix library numpy calls (OpenBLAS) splits long sums into blocks by thread count, so the same
features differ slightly between one thread and eight. References are built at the host's count (8).

**Fair to both families.** C2 sped up only the MLP, and musique's ratio reached 4.89. The rule then became "no path
gets an optimisation the others are denied". C3 gave the GNN its fastest exact form too, and musique fell to 3.39. A
fast MLP against a slow GNN would have been a false win.

### 9.4 What made things faster

| change | what it did | result |
| --- | --- | --- |
| Fast feature compile (numba, 30 Sep) | the 197-column feature build rewritten as compiled loops, in the reference's exact order | 3.9 to 16.2 times faster, bit for bit (metaqa 877 → 54 ms) |
| Galloping merge (C4) | a hub node's neighbour list is searched by doubling-then-halving steps instead of read entry by entry | 2wiki's compile 2.59 → 1.14 ms (MLP), 4.53 → 2.71 ms (GNN) |
| Exact chain walk (C6) | webqsp's typed chains (87,299 entries per question) built by a compiled walk, with no global sort | webqsp's MLP went from slower than the GNN (0.84) to 1.33 times faster |
| Both families' fastest forms (C11) | one input code for twin0 and gnn0; fused steps | every model 1.16 to 1.74 times faster at the same scores |
| Compile kernels (C12) | seed similarities made once per pattern; a CPU instruction for half-precision reads | 1.02 to 1.10 times; small, but exact |
| Memory layout for the walks (P4b) | each node's hot data in one 16-byte record, neighbours merged, no reset pass after a walk | 1.14 to 3.75 times faster walks, every pool the same |

*Example of the galloping merge (illustration):* a hub row holds 80,000 entries and the pool has 104 nodes. Reading
the row costs 80,000 lookups. Searching it for each pool node by doubling-then-binary steps costs about 20 steps
each, about 2,000 in all.

*Example of "no reset pass" (P4b):* a walk marks the nodes it touches. Instead of clearing 300,000 marks after each
walk, every walk stamps its own number, and walk 8 treats any stamp from an earlier walk as "untouched".

### 9.5 What we refused to do: cut inputs

A faster MLP is easy if it reads less. zn, the lean MLP with every link-derived input removed, is 2.65 to 9.61 times
faster than zsp, but it keeps only 13 to 73% of the gain. F1 checked every input family with a **permutation test:**
shuffle one family's values across a question's rows and see how much R@5 drops. *(Illustration:* if the walk score
were noise, giving candidate A candidate B's walk score would not change the top five.*)* Every family is used by
both models: shuffling the first-stage ranks drops R@5 by up to 0.54, the chain inputs on metaqa by 0.72, and even
the least-used family (graph distances) by 0.007, over the 0.002 bar. So nothing is cut. The user's rule: "we are
not removing stuff to make it faster we are making faster keeping the wins".

### 9.6 The numbers, and why 2x is out of reach within a family

Cold, median ms per question on today's pools (C12), and the ratios:

| dataset | twin0 | gnn0 | zrc | zsp | gnn0 / twin0 | zsp / zrc | gnn0 / zrc |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2wiki | 4.11 | 6.00 | 2.52 | 2.61 | 1.46 | 1.04 | 2.38 |
| hotpotqa | 3.91 | 5.88 | 2.36 | 2.48 | 1.50 | 1.05 | 2.49 |
| squad | 2.67 | 4.24 | 1.65 | 1.73 | 1.59 | 1.05 | 2.57 |
| musique | 75.66 | 143.26 | 35.87 | 42.19 | 1.89 | 1.18 | 3.99 |
| metaqa | 49.47 | 77.45 | 26.42 | 28.07 | 1.57 | 1.06 | 2.93 |
| webqsp | 69.87 | 107.26 | 82.65 | 79.62 | 1.54 | 0.96 | 1.30 |

zsp is the fastest GNN on all six. The target was an MLP 2 to 3 times faster than it, and none is. The reason is
arithmetic. A ratio is (shared + GNN-only) ÷ (shared + MLP-only). The shared part pulls it toward 1, and even if
the shared part cost nothing, it could not pass GNN-only ÷ MLP-only.

- **The lean pair (real, musique):** zsp's own parts are its links (5.54 ms) and its forward (6.15); zrc's is its
  forward (5.50). Even with everything else free, (5.54 + 6.15) / 5.50 = 2.12, and on the other five datasets 1.16 to
  1.34. zsp is zrc plus one small step that costs 3 to 17% of zrc's time. That is the honest finding: a well-built GNN
  of this kind costs barely more than its MLP.
- **The universal pair (real, 2wiki):** the feature compile both share is 58 to 71% of twin0's time. Without it,
  gnn0 is 0.09 + 0.19 + 3.34 = 3.62 ms and twin0 is 0.09 + 1.58 = 1.67 ms, so the ratio tops out at 2.17. For 2x the
  compile would have to get 1.18 times faster on musique, 3 to 3.7 times on squad, metaqa and webqsp, and 8 to 9
  times on 2wiki and hotpotqa.
- **Across families:** the lean MLP against the universal GNN is 2.38 to 3.99 times faster on five datasets (1.30 on
  webqsp), for the model stages.

### 9.7 The final pool's build: a cost every model pays

KIND_T's build per question (P4a, cold, median):

| dataset | build | walks' share |
| --- | ---: | ---: |
| squad | 0.19 ms | no walk |
| metaqa | 10.4 ms | 88% |
| webqsp | 61.3 ms | 96% |
| musique | 76.0 ms | 92% |
| 2wiki | 198.9 ms | 97% |
| hotpotqa | 286.2 ms | 97% |

**Why the walks are slow:** an A1 walk on hotpotqa or 2wiki touches 270,000 to 330,000 nodes and reads about a
million stored entries, at 67 to 142 ns each, mostly random reads from main memory. A million reads at about 100 ns is
0.1 s. The processor spends its time waiting for memory, not computing.

**P4b's fix,** the same walks step for step with their data laid out for the memory (9.4), measured on fresh
questions (201-400) after tuning on the first 200:

| dataset | before | after | faster by |
| --- | ---: | ---: | --- |
| metaqa | 7.24 ms | 6.36 ms | 1.14x |
| musique | 48.93 ms | 34.40 ms | 1.42x |
| webqsp | 34.58 ms | 13.20 ms | 2.62x |
| hotpotqa | 171.05 ms | 47.72 ms | 3.58x |
| 2wiki | 188.47 ms | 50.30 ms | 3.75x |

The bigger the graph, the bigger the gain. All 5,322 walks gave the same pools. Even so, the build is still about 20
times the lean MLP's whole time on hotpotqa and 2wiki, so every model-to-model ratio there is 1.00 to 1.08 end to end.
Running a question's walks on several threads would cut it further; that is the user's call.

**The KT input's cost (P4c).** The new columns read six walk orders, and KIND_T's build skips some of them. Serving
them adds, per question: about 4 ms on musique, 7 ms on hotpotqa and 2wiki (KIND_T already walks there), 22 ms on
squad, 32 ms on webqsp and 39 ms on metaqa. The columns and the head cost under 1.4 ms of that. Both lean models pay
it, so their ratio stays 0.98 to 1.08.

### 9.8 Where Claim 3 stands

Not met as the user set it ("mlp should be at minimum 2-3x faster than gnn across all families"). What holds: the lean
MLP is 2.4 to 4 times as fast as the universal GNN for the model stages on five datasets, and twin0 is 1.46 to 1.89
times as fast as gnn0. Within the lean family the two cost about the same, by design. Whether the claim is stated on
the model stages with the pool build beside it, or end to end, and whether the walks may use threads, are the user's
calls.

**Lessons worth keeping:** name the boundary (warm or cold); time the whole path and split it; give every speed-up to
both families; exact forms only; pin, use one thread, keep the machine quiet, and trust paired ratios over absolute
times; memory layout beats arithmetic tweaks; cutting inputs is not a speed-up.

Details: [C1_COLD_COST.md](C1_COLD_COST.md), [C3_FASTEST_FORMS.md](C3_FASTEST_FORMS.md),
[C11_BOTH_FAMILIES_FAST.md](C11_BOTH_FAMILIES_FAST.md), [C12_SHARED_COMPILE.md](C12_SHARED_COMPILE.md),
[F1_ONE_INPUT_SET.md](F1_ONE_INPUT_SET.md), [P4A_POOL_COLD.md](P4A_POOL_COLD.md),
[P4B_FAST_WALKS.md](P4B_FAST_WALKS.md), [P4C_KT_SERVING.md](P4C_KT_SERVING.md).

## 10. Where we stand (14 October, 21:30)

**Claim 1 (the MLP keeps at least 90% of the GNN's gain).** For the lean pair (zrckp, zspk), with 95% intervals
([P3C_CLAIM1_INTERVALS.md](P3C_CLAIM1_INTERVALS.md)):

| dataset | GNN's gain over RRF | MLP's share [95%] | in-domain | zero-shot |
| --- | ---: | --- | --- | --- |
| metaqa | +0.796 | 0.997 [0.996, 0.998] | holds | holds |
| squad | +0.0045 | 1.302 [0.859, 2.824] | not shown | holds |
| musique | +0.107 | 0.933 [0.881, 0.983] | not shown | holds |
| hotpotqa | +0.220 | 1.010 [0.999, 1.020] | holds | holds |
| 2wiki | +0.327 | 0.986 [0.981, 0.991] | holds | holds |
| webqsp | (never trains) | | | holds |

- On squad there is almost no gain to share: the first stage is already at 0.905.
- On musique the MLP keeps 93% at the point, but as little as 88% on resamples. This is the one the models can move.
- On HippoRAG 2's benchmark the MLP keeps 0.97 to 0.99 of the GNN's gain.
- The universal pair falls short: twin0 keeps 0.852 of gnn0's gain on 2wiki and 0.596 on musique (point estimates).

**Claim 2 (an LLM-free GNN at the state of the art).**
- On HippoRAG 2's benchmark: above it on 2wiki (97.2 against 90.4) and hotpotqa (98.2 against 96.3), below on musique
  (66.3 against 74.7). Caveat: on 2wiki and hotpotqa our structural links are the corpus's own hyperlinks, which
  HippoRAG 2 does not use. With one LLM-free link rule in their place, 2wiki stays above (95.9) and hotpotqa falls
  just below (95.0).
- Our lift over our own first stage is larger than HippoRAG 2's lift over its 7B encoder on all three settings:
  +9.9 / +25.9 / +13.2 points against +5.0 / +13.9 / +1.8 (musique / 2wiki / hotpotqa). So the wins come from the
  features and models, not the encoder.
- On our full corpora, both GNNs are at or above the best of eight LLM-free rankers on all six datasets (P3a).
- Two caveats. B1's questions are inside s1eval, the questions our choices were made on (4.8). And P3a read zsp and
  gnn0, so the new lean GNN, zspk, has not been read against the bar yet.

**Claim 3 (the MLP is faster, cold).** Not met as the user set it. No MLP is 2x faster than the fastest GNN (zsp)
on any dataset. Within the lean family the MLP is 0.96 to 1.18 times as fast as the GNN (zrc against zsp). Within
the universal family it is 1.46 to 1.89 times as fast. Across families, the lean MLP is 2.4 to 4 times as fast as the
universal GNN for the model stages on five datasets. On hotpotqa and 2wiki, building the pool takes most of every
model's time (97 to 99% before P4b's faster walks, still about 20 times the lean MLP's own time after), so every
end-to-end ratio there is near 1.

**What is final:** the pools (KIND_T) and the links (three families, hyperlinks kept).

**What is open:**
- musique's ranking (three- and four-hop questions);
- hotpotqa's hit@1 under the new input;
- zero-shot on knowledge bases (webqsp stays below the universal pair, which trains on it);
- one pool-size rule for all datasets, chosen without evaluation golds;
- squad's question-link switch, which should become a test on the data;
- the adoption decisions re-called without B1's questions (4.8), and zspk read against the SOTA bar;
- Claim 3's wording, and threads for the walks (the user's calls);
- one confirmation run of everything final.

## 11. Glossary

- **A1:** HippoRAG 2's walk restart: the seeds plus the first stage's top 1,000.
- **ADOPT / NOT_ADOPTED:** a full run's verdict: at least one primary GAIN and no LOSS in 36 reads, or not.
- **ALL:** the share of questions with every gold in the pool.
- **B1:** HippoRAG 2's own benchmark (1,000 questions each on musique, 2wiki and hotpotqa). Nothing was chosen on its
  runs, but its questions are inside s1eval (4.8).
- **Bootstrap:** resample the questions with replacement 2,000 times to see how much a number could move (4.4).
- **BRIDGE / RANKED / POOL:** where a missed gold is. BRIDGE: in the pool, one link from a gold the model found.
  RANKED: in the pool, ranked out, with no such link. POOL: outside the pool.
- **Carve:** a fixed slice of questions: fit (learn), select (choices such as a pool rule; no training epoch has been
  picked on it since 7 October), s1eval (score).
- **Cold:** timed from scratch for each question, with nothing cached or warmed up, on one pinned thread.
- **Corroborated link (L\*):** a node the question names that the first stage also retrieved (dense or SPLADE top
  100) (5.4).
- **δ_MP:** what message passing alone adds: the GNN minus its twin. In M3B the twin was the GNN itself with every
  message edge replaced by a self-loop (GAT-NO-MP) (7.2).
- **Dense:** the search that ranks by vector closeness (meaning), from a frozen text encoder.
- **Entity question / passage question:** every start node is name-only (its text is just its name), or not (5.4).
- **FC@5:** 1 if every gold is in the top five (example in 2.3).
- **Fill:** the nodes KIND_T adds to a question's base and seeds, up to its size in today's pools (5.5, 5.6).
- **First stage:** RRF of the dense and SPLADE lists; the "before" line of every gain.
- **Floor:** the smallest change that counts on a read: max(0.0075, twice the typical seed-only move) (4.5).
- **Full run / screen:** 6 fits and 36 reads / 2 fits and 12 reads (4.3).
- **GAIN / LOSS / WITHIN:** a read's verdict against the base (4.6).
- **GNN:** a model whose candidates update from their linked neighbours at question time (message passing).
- **Gold:** a correct passage or entity. **Gold recall:** the share of golds in the pool.
- **hit@1:** the top result is a gold.
- **HippoRAG 2:** the state-of-the-art graph retriever we compare with. It builds its graph with an LLM.
- **In-domain / zero-shot:** read on a dataset the fit trained on / never saw.
- **J5 / L-x:** the fit trained on five datasets / the fit that leaves dataset x out.
- **KIND_T:** the final pool rule: one parameter-free recipe for all six datasets (section 5).
- **KT input:** eight numbers per candidate from KIND_T: six positions and two flags (section 7).
- **Lean pair / universal pair:** (zrckp, zspk) with their own best inputs / (twin0, gnn0) with one shared set of
  129 columns.
- **LINKED:** a walk from the question's linked entities, the way KB question-answering systems enter the graph; one of
  the orders behind the SOTA pool bar (5.3).
- **Link families:** structural (KB facts, hyperlinks, title mentions), named-entity, nearest-neighbour (2.4).
- **Listwise loss:** −log of each gold's softmax probability over its own pool, averaged over the golds (2.7).
- **Memorizing / held-out stop:** a fit that learns its training questions' answers instead of a rule that carries
  to new questions; the stop keeps the epoch where the loss on held-out questions is lowest (7.5).
- **Message passing:** a candidate's score or representation updated from other candidates' at question time.
- **MLP:** a model that scores each candidate from its own row.
- **MLP rule:** an MLP input never comes from another candidate's score, not even through a mean or a spread.
- **Pool:** the shortlist a model reranks. A gold outside it cannot be found.
- **PPR / walk:** personalised PageRank, a random walker that restarts at the seeds with probability 0.5 (2.5).
- **Primary reads:** the 11 of 36 that decide a full run: J5 on each trained dataset and on webqsp, each L-x on x.
- **Probe:** a small model fitted to the gap between a GNN's scores and its twin's, to see what form the gain takes
  (7.2).
- **R@5:** the share of a question's golds in the top five. Claim 1's measure.
- **RRF:** reciprocal rank fusion: a candidate scores the sum of 1/(60 + rank) over the lists (2.2).
- **RRF5 / RRF5T:** RRF over the first stage's order and four walks; RRF5T steers the walks from the links (5.4).
- **s1eval:** the fixed evaluation carve every comparison since 7 October reads.
- **Seed (two meanings):** a *seed node* is a top-5 dense or SPLADE result, where walks start; a *training seed* is
  the random number that fixes a training run's luck.
- **Seed null:** the unchanged model retrained on other training seeds, to measure luck (4.5).
- **Share:** (MLP − RRF) / (GNN − RRF), the part of the GNN's gain the MLP keeps.
- **SOTA:** state of the art: the best published method.
- **SOTA pool bar:** at each pool size, the best LLM-free rebuild of a published pool order (5.3).
- **SPLADE:** the search that ranks by weighted word matches.
- **Steer:** a walk that splits its mass by the neighbours' first-stage RRF weights instead of equally (5.4).
- **Today's pools:** the per-dataset pools frozen in September, before KIND_T.
- **Twin:** an MLP trained on exactly its GNN's rows, questions and loss.
- **twin0 / gnn0:** the universal MLP and GNN.
- **zrc / zsp:** the lean MLP and GNN of 9 to 14 October, on today's pools.
- **zrckp / zspk:** the lean MLP and GNN since 14 October, on KIND_T's pools with the KT input.
