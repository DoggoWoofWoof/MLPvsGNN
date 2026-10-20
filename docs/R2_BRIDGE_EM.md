# R2: learned bridge transforms trained with EM, on both models

Declared 20 October 2026, before any number. The user's direction of 20 October. Code: `outputs/mp_unified/zr2br.py`.
Numbers here are development numbers. Not run yet.

## 1. Why this, now

**The user's words** (20 October): "we need to improve our ranking try transformations, attentions offsets etc using
EM to improve the vectors as well". R2 does this inside both ranking models. It learns a transform of the frozen
vectors, set by the question and by a row the first stage already ranks high, and trains it inside each model's own fit
with an EM term.

**The bridge is there.** The first stage ranks each question's pool by RRF: the reciprocal rank fusion of the frozen
dense encoder's list and SPLADE's. Three filed findings say the golds the models miss sit next to rows they find (R@5
is the share of a question's golds in the top 5):
- **B1d** (docs/B1D_BRIDGE_SLOTS.md; HippoRAG 2's three settings; nothing trained).
  - 86% to 99.5% of the misses have a sibling gold in the top 5, and most are one edge from it (B1c's finding).
  - One fixed rule keeps the top 3 and fills slots 4 and 5 with the top 2's best document-link neighbours, by cosine
    with the question. Its R@5 (×100, with the odd-half gain over the first stage) is ABOVE on 2wiki (92.8,
    +21.2 [+19.0, +23.3]) and on hotpotqa (95.2, +10.0 [+8.0, +11.9]), and SAME on musique (57.0, +1.1 [-0.7, +2.9]).
  - Adding the found passage's vector to the question's did not help: 0 to −3 points on the structural and NER rules.
    Its record: "A fixed vector sum is not the bridge; the edge is. Any offset must be learned (G1)."
- **D1** (docs/DIAG_BRIDGE.md, 8 October; on zrm's screen fits, zrm being the MLP's base then): REACHABLE_NOT_RANKED.
  - Take the questions the model finds only partly. On 2wiki and hotpotqa, 0.77 to 0.93 of the golds it misses sit
    next to a found gold. On musique the share is 0.38, and 0.77 within two hops.
  - Fixed re-ranks do not use this. Anchored on a found gold, the model's score ranks the missed gold 5th to 9th
    (median) of the anchor's about 45 neighbours on musique.
- **D2** (the same file): FEATURE_LIMIT.
  - Boosted trees on zrm's inputs plus zrm's own score add at most +0.0059 R@5 on all eight in-domain reads.
  - What the per-row inputs lack is relational: the missed golds are tied to the rows the model ranks high.

**What was tried.** docs/SCREENS.md section 3b: a verdict closes only the form it tested.
- **zbr** (docs/SCREENS.md, round 29, 9 October): the first stage's leading rows, given to zrc (the lean MLP's base)
  as 13 per-row inputs from the pool's edges. NO_GAIN, and all 12 reads WITHIN.
- **zof** (round 32, 10 October) is the nearest form.
  - It scored each row against the first stage's top 5 rows by a question-conditioned offset on zrc, in DistMult's
    form (4 channels of 16 each).
  - It had 25 inputs in rank order and learned only from the listwise loss.
  - NO_GAIN, and it harmed: 7 of 12 reads are a LOSS. The largest is musique in-domain on L-hotpotqa (the screen fit
    that holds hotpotqa out of training), −0.087 [−0.096, −0.078]. Its record: "It costs most where the anchors are
    least often gold (musique, webqsp)."
  - The audit of 17 October (docs/VERDICT_AUDIT_2026_10_17.md) closed that end-to-end form, and only that form.
- **P5b** (docs/P5B_LEARNED_OFFSETS.md, 13 October).
  - Offsets g (200,465 weights), fitted on their own by online generalized EM over the latent (seed, head), on 5,783
    questions for 30 epochs.
  - On 2wiki g worked: at N = 10, g's order found 0.322 of the low golds (neither seeds nor in the top 5 of zsp, the
    lean GNN then). The hops-then-cosine order found 0.136.
  - On musique g failed. Its top 10 held 0.007 of the low golds, against 0.159. Its fit loss fell to 0.88 nats per
    gold, and its offsets were 5 to 10 times the median distance between two nodes. STOPS.
- **P5c** (docs/P5C_OFFSET_INPUT.md, 19 October): P5b's offsets, cross-fitted, given to both lean models as a fixed
  input through a 97-parameter head. Both NOT_ADOPTED:
  - the MLP: 0 GAIN, 0 LOSS (musique in-domain +0.0062 [+0.0016, +0.0112], WITHIN);
  - the GNN: 0 GAIN, 1 LOSS (2wiki in-domain on L-hotpotqa, −0.0111 [−0.0126, −0.0094]).
- **E1** (docs/E1_LATENT_CHAIN_EM.md, 17 October): EM over the chain head's latent relation chains, in both lean
  models. Its term acted only where chains exist, and in training only metaqa's fit carve has them.
  - the MLP: 0 GAIN, 0 LOSS;
  - the GNN: 0 GAIN, 1 LOSS (2wiki in-domain on L-hotpotqa, −0.0192 [−0.0210, −0.0173]).

**What R2 changes.**
- **Against zof.** The transform gets its own training signal. For each gold, an EM term asks which of the top 5 led to
  it, so the transform does not learn only through a head that starts at zero.
  - The head sees three inputs pooled over the bridges, in place of 25 slots in rank order.
  - The translation is set jointly by the question and the bridge, in one shared 32-wide space.
  - It runs on both models, against today's card fits (the GPU fits the screens are decided against). It is called
    under N2's fresh floors: the seed-noise floors, measured again on today's models.
- **Against P5b and P5c.** Nothing is fitted apart from the ranker. The transform trains inside each model's own fit, on
  its batches, so it learns what the ranker can use. There is no build, no cross-fitting and no fixed column.
- **Against E1.** The term reaches every training question with a gold in its pool, on all four of a fit's training
  datasets: passage graphs and KB alike.

## 2. The form

**R2 reads no edge and no score: each row is scored from its own vector, the question's vector and the question's 5
first-stage bridges' frozen vectors (the same 5 for every row, picked by the first stage's RRF before any row is
scored); the only normalisation over rows is inside the training loss.**

**Fixed before any row is scored**, per question q:
- **The bridges.** The 5 rows of q's pool with the highest first-stage RRF score, in descending order. R2 reads the
  carve's `c_rrf` column as stored.
  - A tie goes to the earlier pool position (lean_gpu's order), and a missing value (NaN) goes last.
  - A pool of n_q < 5 rows has n_q bridges: K_q = min(5, n_q).
  - The bridges are computed once per carve, when it loads.
- **The prior.** Bridge k (k = 1 to K_q, in that order) has weight π_k ∝ 1/(60 + k), RRF's own form. For each row v,
  π is renormalised over the bridges unmasked for v; that gives π̃_k(v).

**Learned**, at r = 32. Every draw comes from R2's own generator, at seed + 5502.

| quantity | definition |
| --- | --- |
| h_q | U_bᵀ e_q, where e_q is the question's 1536-wide vector from the frozen encoder |
| h_v | V_bᵀ p_v, where p_v is the row's stored vector (257 wide: the encoder's vector on the graph's 256 principal axes, plus one constant column), as the models' semantic block (SEMB) reads it |
| h_k | V_bᵀ p for bridge k: the same V_b maps the bridges and every row |
| t_k | h_q + W_1 h_k + W_2 (h_q ∗ h_k) + o: the point bridge k leads to, given the question (∗ is the elementwise product) |
| s_vk | ⟨t_k, h_v⟩ / √32; masked where v is bridge k itself, or q has no bridge k |
| a_v | log Σ_k π̃_k(v) exp(s_vk) over the unmasked k: a soft maximum under the prior |
| m_v | the maximum of s_vk over the unmasked k |
| f_v | 1 when some bridge is unmasked for v; when none is, a_v = m_v = f_v = 0 |

- **The head.** Linear(3, 16), GELU, Linear(16, 1), over x_v = [a_v, m_v, f_v]. Its output is added to the arm's final
  score. The last layer starts at zero, so each arm starts as its base, bit for bit.
- **The start.**
  - U_b and V_b ~ N(0, 1). This scale was chosen so the bridge logits start near unit size, from a read-only look at
    step 1's s1eval carves' inputs (the RRF column, the question and row vectors, and the logits' spread at this
    start). No gold, label, score or model output was read, and the look is not a result.
  - W_1 ~ N(0, 1) × 0.1/√32.
  - W_2 = 0 and o = 0.
  - The head's first layer is uniform in ±1/√3.
  - The base's own draws, and the global generator, stay where the base leaves them.
- **59,537 parameters per arm:** U_b 49,152; V_b 8,224; W_1 and W_2 2,048; o 32; the head 81.

**The EM term, in training only.** For question q, gold row v and bridge k:

| quantity | definition |
| --- | --- |
| p_k(v) | the softmax of s_·k over q's rows unmasked for bridge k: how likely bridge k leads to v |
| L_br(q) | the mean over q's golds v of −log Σ_k π̃_k(v) p_k(v) |
| the loss | lean_gpu's listwise loss, plus LAM × the batch mean of L_br over the batch's questions with a term |

**LAM = 1**, fixed here before any number. No other value is screened.

- **Why it is EM.** The gradient of L_br in s_v′k is −r_k(v) (1[v′ = v] − p_k(v′)).
  - r_k(v) = π̃_k(v) p_k(v) / Σ_j π̃_j(v) p_j(v) is the posterior that bridge k led to gold v.
  - The E-step's posterior weighs the M-step. For the bridge that explains the gold, the gold rises over the
    question's other rows.
  - This is the online form P5b and E1 used.
- **What it trains.** s reads only U_b, V_b, W_1, W_2 and o; e_q and p are frozen. So the term trains only the
  transform. The head and the base learn from the listwise loss, as before.
- **Which questions.** Every training question with a gold in its pool and at least two rows, on all four of the fit's
  training datasets. A gold that is itself a bridge is explained by the other bridges. webqsp never trains.
- **Reads and serving have no term.** The softmax over rows is never formed at read time.

**Unchanged.**
- Each base's carve, blocks, settings, seed 0, batches and training loop.
- P1c2's pools and caches (today's pools), P1c3's KT input (eight columns of each row's walk positions in the pool's
  build) and zg6's edges.
- The encoder.
- There is no build, and no new graph, edge, column or text.

**Against the MLP rule.** docs/SCREENS.md's label of rounds 22 and 23 sets the rule: "A row's inputs are computed
before any row is scored, from the question, the row, the frozen features and the graph's fixed structure, and never
from another row's score." R2 carries three flags against it:
1. **Other rows' vectors, as context.** R2 reads other rows' frozen vectors (the five bridges') as fixed question-level
   context. So do the ctx block's pool statistics (lean_mlp8's per-question means, spreads and maxima of the input
   columns) and zbr's leaders (the first stage's top rows). No other row's score enters, and the bridges are the same
   for every row.
2. **Graph-free.** R2 reads no edge. Its bridge is a learned translation in vector space, not a link. The edge form,
   R2e, is the next form for no GAIN (section 5).
3. **The user rules on it.** Whether reading the bridges' vectors keeps zrckpb inside the MLP track is the user's
   ruling.
   - Until then, every record calls zrckpb "zrckp + R2's bridge head", and no claim sets it against the GNN as the MLP.
   - Its rule and its verdict do not change with the ruling.

On zg6m the flags change nothing. zg6 already reads the pool's edges and other rows' states, and R2's head adds no
edge of its own.

**A worked toy example.** The numbers are made up, to show the arithmetic. Here r = 2 (R2 uses 32), and the toy sets
W_1 = I and W_2 = o = 0. One question has three bridges, b1, b2 and b3, and two more rows: v (the gold) and w.

**The vectors.**

| vector | value |
| --- | --- |
| h_q | (1, 0) |
| h_b1, h_b2, h_b3 | (0, 1), (1, 1), (−1, −1) |
| h_v, h_w | (1, 2), (2, −1) |
| t_1, t_2, t_3 | (1, 1), (2, 1), (0, −1) |

**Each row's inputs.**

| row | s (bridges 1, 2, 3) | π̃ | a | m | f |
| --- | --- | --- | --- | --- | --- |
| v | (2.1213, 2.8284, −1.4142) | (0.3387, 0.3333, 0.3280) | 2.1453 | 2.8284 | 1 |
| w | (0.7071, 2.1213, 0.7071) | (0.3387, 0.3333, 0.3280) | 1.4189 | 2.1213 | 1 |
| b1 (a bridge) | (masked, 0.7071, −0.7071) | (masked, 0.5040, 0.4960) | 0.2364 | 0.7071 | 1 |

- π̃ is 1/61, 1/62 and 1/63, renormalised over the slots unmasked for the row.
- At the start the head's last layer is zero. Every row adds 0, so the score is the base's.

**The EM term for gold v.** Each softmax runs over the four rows unmasked for its bridge.
- p_1(v) = 0.5665, p_2(v) = 0.6173 and p_3(v) = 0.0746.
- Σ_k π̃_k(v) p_k(v) = 0.4221, so L_br = 0.8625.
- The posterior is r = (0.4546, 0.4874, 0.0580): bridges 1 and 2 explain v, and bridge 3 hardly does.
- The gradient raises s_v1 and s_v2 (−0.1971 and −0.1865 in L_br). It lowers w most under bridge 2 (+0.1483).

**Speed.**
- One sort of RRF per carve, at load.
- Per question, one 1536 × 32 map.
- Per row, one 257 × 32 map and five products of width 32.

All of it is query-local. Any latency figure for R2 is cold, timed in its own declared stage.

## 3. The arms

| arm | what it is | carve | decided against (card fits) | the null that decides |
| --- | --- | --- | --- | --- |
| zrckpb | zrckp (the lean MLP: zrc plus P1c3p's pointwise head on the eight KT columns, docs/P1C3P_POINTWISE_HEAD.md) plus R2's head | zrckp's KT chain carve, with the bridges beside | zrckp's P1c3p fits `scr-p1c3p-zrckp` and `-hp` | N2's fresh zrckp null |
| zg6m | zg6 (G6's free GNN, docs/G6_FREE_GNN.md) plus R2's head on its final score | zg6's KT link carve, with the bridges beside | zg6's G6 fits `scr-g6-zg6` and `-hp` | N2's fresh zspk null, moved onto zg6's card fits through G6's comparisons |

- **Each arm is its base unchanged, plus R2's head.** It keeps the base's carve, settings, seed 0 and batches. From
  the same seed, each starts as its base, bit for bit.
- **zg6m waits for G6.** Its base fits are G6's, and its memory is sized from G6's measured runs. It is decided against
  zg6, whatever G6's own verdict.

## 4. The fits and the call

- **Fits.** One card fit per arm and split:
  - `scr-r2-zrckpb` and `scr-r2-zg6m` on L-musique, where musique is held out;
  - the `-hp` fits on L-hotpotqa, where hotpotqa is held out.

  All four run on P1c2's roots, with P1c3's KT input and the base's card settings. Each fit's record keeps R2's
  constants and the EM term for each epoch.
- **Reads.** p@swa (the fit's weight-averaged state) on the six s1eval carves (the screens' evaluation questions, one
  carve per dataset). zg6m's reads use `--rows-cap 65536`, as G6's do.
- **Compares.** Each fit against its base's card fit of the split, with step 1's fit beside.
- **Call.** relz's pair and its re-calls, as in every screen since the tenth round, stamped as R2's. GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5 (the share of a question's golds in the top 5).

### The rule, fixed before any number

**The null that decides is N2's** (docs/N2_FRESH_SEED_NULL.md). Each base's card fit is trained again with seeds
1 and 2. Let D1 and D2 be the two seeds' R@5 differences on a read. That read's floor is
max(0.0075, 2 · sqrt((D1² + D2²) / 2)). A filed GAIN stays a GAIN only when its difference reaches the floor, and the
same holds for a LOSS.
- **zrckpb:** N2's zrckp null (`outputs/screen/scr-n2-zrckp-s1`, `-s2`, `-s1-hp`, `-s2-hp`).
- **zg6m:** N2's zspk null (`outputs/screen/scr-n2-zspk-*`). Each read's seed-0 R@5 is moved to zg6's card fit
  through G6's comparisons of zg6 with zspk (`outputs/screen/scr-g6-zg6`, `-hp`). D1, D2 and the floors stay zspk's.
  This is the stand-in G6 itself uses.

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call under N2's null has no LOSS on any of its twelve reads, and at least one GAIN.
  The twelve are six datasets × two fits; webqsp is read zero-shot in both, and so is each fit's held-out dataset.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** any of these:
  - the re-call is missing;
  - it is not R2's re-call of the arm under N2's null;
  - it does not hold the twelve reads;
  - N2's floors record of the arm's null (`outputs/n2/floors-zrckp.json`, `floors-zspk.json`) is missing or not OK.

**Beside, deciding nothing:**
- Each arm's re-call under the 7 October null (step 1's seeds), re-based to the card fits.
  - For zrckpb, through L1's comparisons `outputs/l1/base-zrckp-<split>.json`.
  - For zg6m, through R2's own `outputs/r2/base-zg6-<split>.json`, which compares zg6's card fit with step 1's.
- The webqsp and metaqa reads of each fit, and musique in-domain (L-hotpotqa) for both arms.
- Each arm minus its base on each read: R@5 with its 95% question-bootstrap interval, FC@5 (all golds in the top 5)
  and hit@1.
- The EM term for each epoch: L_br's mean, and the share of training questions with a term.

**Two limits, disclosed.**
- N2 measures zspk's seed spread, not zg6's or zg6m's, and a model trained from scratch may vary more between seeds.
  That is why an ADOPT leads to a refit and a full run, not to a claim.
- On the KB datasets the first stage's top 5 rarely holds a gold. RRF's R@5 is 0.0047 on metaqa and 0.0535 on webqsp
  (the table in docs/G6_FREE_GNN.md, from L1's compares). There the bridges are seldom golds themselves, and the
  transform has to learn the step from them to the golds. zof cost most where its anchors were least often gold.

## 5. What each outcome leads to (fixed now)

| an arm's re-call under N2's null | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run on all six splits beside its base's, in its own declared file (no model change is adopted on today's pools) |
| NOT_ADOPTED, with at least one GAIN and a LOSS | R2g, a learned per-question back-off: R2's head times 2 · sigmoid(a · c_q + b), where c_q holds the question's own embedding projection and its first-stage top-5 RRF margins; the gate starts at 1 (one declared file) |
| NOT_ADOPTED, with no GAIN | R2e, the edge form: bridge scores only for bridges linked to the row by zlink's edges (B1d: the edge is the bridge). They enter as fixed per-row link flags for the MLP, and inside zg6's messages for the GNN (one declared file) |
| INCOMPLETE | the missing re-call or floors record is run; nothing else changes |

- When only one arm passes, the other goes to its own next form. Neither is closed.
- A zg6m refit puts R2's head on zg6, or on G6's next form if G6 is not adopted. That refit's own file fixes which.
- No verdict here closes the idea (docs/SCREENS.md section 3b). Each outcome names its next form.

## 6. How it runs

**Self-test, passed before this commit.** `python outputs/mp_unified/zr2br.py --selftest` takes about 25 s. On a toy
carve it checks:
- **The arms.** zrckpb and zg6m run on their bases' carves with the bridges beside, and the earlier arms are unchanged.
  R2 has 59,537 parameters at r = 32.
- **The bridges.** They match a by-hand order, including:
  - ties by pool position, NaN last, and −0.0;
  - pools under five rows, of one row, and empty;
  - any slice.

  A batch's bridges are its own questions', and e_q and p are the ones SEMB reads.
- **The start.** Each arm starts as its base, bit for bit: the state, the forward in eval and in training, and the
  global generator.
- **The head's inputs** match a float64 computation by hand, and no row is its own bridge.
- **The MLP rule.** Changing a row that is not a bridge moves only that row's score. Changing a bridge moves only its
  own question's rows.
- **The EM term.**
  - It matches float64 by hand, and its gradient is EM's to 1e-9.
  - A question with no gold takes no term.
  - The logits the loss reads are the forward's, kept in training only. The training loop is restored after.
- **Toy fits.** With LAM = 0 a fit is the plain loop, bit for bit. With LAM = 1 it lowers L_br. A repeat is
  identical. Without the bridges, each fit is its base's.
- **The refusals.** train and read refuse what is not R2's, constants included.
- **The call.**
  - relz's pair and re-calls under N2's and the 7 October nulls, stamped R2's. The loaders' refusals are checked, and
    relz is restored after.
  - The verdict's outcomes (a missing or refused floors record among them), its next forms and its beside rows.
  - Nothing is written under R2's output roots.

**No development run on real data.** P1c2's caches and P1c3's KT files are on the host only.

**Host items** (`r2-*`, 21 in all). Each line is made from its base counterpart: P1c3p's zrckp items for zrckpb, G6's
zg6 items for zg6m.

| items | what | waits on | memory (GB) |
| --- | --- | --- | --- |
| `r2-train-zrckpb-hp`, `r2-train-zrckpb` | zrckpb's fits on the card (`--frac 0.30`, share 0.32, as P1c3p's) | nothing | 3.0 each |
| `r2-train-zg6m-hp`, `r2-train-zg6m` | zg6m's fits on the card (`--frac 0.45`, share 0.47, as G6's) | nothing (sized by G6's measured fits) | 5.3, 3.5 (estimates) |
| `r2-read-zrckpb-hp`, `r2-read-zrckpb` | the reads | its fit | 5.3, 5.1 |
| `r2-read-zg6m-hp`, `r2-read-zg6m` | the reads, at `--rows-cap 65536` | its fit | 4.9, 5.0 (estimates) |
| `r2-base-zg6`, `-hp` | zg6's card fit against step 1's (the 7 October re-base) | G6's read | 0.6 (an estimate) |
| `r2-compare-zrckpb`, `-hp` | against zrckp's card fit, with step 1's beside | its read | 0.9 |
| `r2-compare-zg6m`, `-hp` | against zg6's card fit, with step 1's beside | its read and G6's | 0.6 (an estimate) |
| `r2-pair-zrckpb`, `r2-pair-zg6m` | the pairs | both compares | 0.6 |
| `r2-recall-zrckpb` | the deciding re-call | the pair, and N2's four zrckp compares | 0.6 |
| `r2-recall-zg6m` | the deciding re-call | the pair, N2's four zspk compares, and G6's two compares | 0.6 |
| `r2-recall-zrckpb-oct7`, `r2-recall-zg6m-oct7` | the 7 October re-calls, beside | the pair, and L1's base compares (zrckpb) or R2's (zg6m) | 0.6 |
| `r2-verdict` | the verdict | the four re-calls, the four compares, and N2's floors items (`n2-recall-floors-zrckp`, `-zspk`) | 0.6 |

- **Memory.** Each item's REF is its counterpart, plus 0.5 GB, times 1.15.
  - zrckpb's counterparts have run, so its sizes are measured.
  - zg6m's wait until G6's runs are measured: the feeder holds an item until its REF is measured. The figures above are
    estimates from G2's measured peaks.
- **Inputs.**
  - G6's 36 (34 code files and the two relation-embedding arrays), plus `zp1c3p.py`, `zn2.py` and `zr2br.py`.
  - The re-calls' null files. zg6m's deciding re-call also lists G6's two comparisons.
  - zr2br.py's whole import closure (32 repository files) is among them.
- **A refused floors record.** When N2's floors item refuses (zn2.py floors exits 2), the feeder drops the verdict. Run
  by hand, the verdict then says INCOMPLETE.
- **Committing.** `outputs/mp_unified/zr2br.py` is under the repository's ignore rule for `outputs/`, so it is
  committed with `git add -f`. rx push sends only tracked files.

**The commands.** The L-hotpotqa items are the same with `--split L-hotpotqa` and `-hp` names. zg6m's items use
`--arm zg6m`, zg6's card fits as the base and `--rows-cap 65536` on the reads.

    python outputs/mp_unified/zr2br.py train --split L-musique --name scr-r2-zrckpb --arm zrckpb --device cuda --host \
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zr2br.py read --name scr-r2-zrckpb --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zr2br.py compare --new outputs/screen/fits/scr-r2-zrckpb \
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-r2-zrckpb
    python outputs/mp_unified/zr2br.py compare --new outputs/screen/fits/scr-g6-zg6 \
        --base outputs/step1/fits/L-musique --out outputs/r2/base-zg6-L-musique
    python outputs/mp_unified/zr2br.py pair --arm zrckpb \
        --screens outputs/screen/scr-r2-zrckpb.json,outputs/screen/scr-r2-zrckpb-hp.json \
        --out outputs/screen/scr-r2-zrckpb-pair
    python outputs/mp_unified/zr2br.py recall --arm zrckpb --null-kind N2 --null outputs/screen/scr-n2-zrckp-s1.json,\
    outputs/screen/scr-n2-zrckp-s2.json,outputs/screen/scr-n2-zrckp-s1-hp.json,outputs/screen/scr-n2-zrckp-s2-hp.json \
        --pair outputs/screen/scr-r2-zrckpb-pair.json --out outputs/screen/scr-r2-zrckpb-pair-recall
    python outputs/mp_unified/zr2br.py recall --arm zrckpb --null-kind oct7 --null outputs/screen/scr-null-s1.json,\
    outputs/screen/scr-null-s2.json,outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json \
        --pair outputs/screen/scr-r2-zrckpb-pair.json --out outputs/screen/scr-r2-zrckpb-pair-recall-oct7
    python outputs/mp_unified/zr2br.py verdict

zg6m's deciding re-call takes N2's zspk files (`scr-n2-zspk-s1.json` to `-s2-hp.json`) and moves them onto zg6's card
fits through G6's comparisons by itself.

**Disk** (an estimate):
- zrckpb's fits: about 10.3 MB each, zrckp's 8.2 MB plus R2's nine states of 59,537 parameters (2.1 MB).
- zg6m's fits: zg6's size plus 2.1 MB.
- Each read: about 13 to 14 MB (six s1eval score files).
- Everything else: under 0.1 MB per item.

About 0.1 GB in all.

**ETA** (an estimate: the bases' measured runs plus 15%):
- **zrckpb.** Fits take about 30 minutes each (P1c3p's zrckp fits took 26 and 27 minutes), and reads about 4 minutes.
  With both fits at once on a free card, it takes about 35 minutes from the first fit to the last read.
- **zg6m.** Fits take about 65 minutes (L-musique) and 100 minutes (L-hotpotqa), and reads about 18 minutes at the
  cap. These come from G2's measured runs: fits of 56 and 89 minutes, reads of 15.5 minutes at the cap. G6's own runs
  are measured before zg6m's start. Two at a time, it takes about 2 hours.
- **In all,** about 4.5 card-hours of item time. The deciding re-calls then wait for N2's compares; zg6m's also waits
  for G6's.

The text encoder stays frozen. webqsp never trains. Test splits are never read. There is no LLM or transformer, and no
new graph.

## 7. Results

Not run yet.
