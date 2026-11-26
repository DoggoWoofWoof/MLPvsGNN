# UL5: a label-free depth signal as a constrained program (can one universal rule spend nodes by question and reach 95 percent ALL near 2,500?)

Declared 26 November 2026, 04:20 IST (22:50 UTC the day before), before any UL5 number exists. **Nothing here trains, no model reads anything, no LLM or spaCy is used, no filed graph, pool, link, fit or record changes, and no link is removed from any store.** It builds on records that are already filed (PL6's picks and PL2 to PL7's dev grids) and adds two things: one new extraction of the same engine's pools at finer cuts (dev first, eval once), and a constrained program on the laptop. It carries nothing downstream; it says whether a depth signal that reads no label exists, and what it costs.

## 0. The user's words, and the rulings that bind

- 26 Nov, ~03:3x: "No universal one-hop rule exists. we need to make it universal, tell the bounds or costs for that to happen i told you my limits, 95+ all coverage, as close as possible to 2.5k nodes".
- 26 Nov (latest): "do both parallely and wrk on r7 as well based on the 5 pints yu told me, keep working autnmuslly, and instead f linear programming do as a constrained programming problem". **This page is the "constrained programming" form of UL4's depth layer.** Its pair (the gold-link patterns) is the user's other half and runs as GP1 in its own session (docs/GP1_GOLD_LINK_PATTERNS.md).
- Earlier and still binding: "we need to have universal rules" (one rule on all seven, carried everywhere or nowhere); pools first and parameter-free (no LLM, no spaCy, no learning); declare before any number; selection on dev, grading on eval, subset first; the hub line is CLOSED (see section 3 for the one flag); 25 Nov: "coverage target is 95+ and as close to 100 as possible for the gold pools at the smallest node counts"; the 2,500-node bound is liftable only under one rule on all seven, and "we still need to start restricting it and bring towards 2.5k".

## 1. What I had read before this page

UL4's results (section 10 of docs/UL4_UNIVERSAL_LP.md): the oracle's least mean nodes at ALL 0.95 per question on the stored ladder (MuSiQue 3,382, MetaQA 2,804, WebQSP 10,250; the four small datasets inside 2,500) and the uniform picks (15,400, 4,984, 39,572). PL6's picks and PL7's call (NOT_CARRIED: no hard cut wins on dev at any ceiling). **I have seen no per-question statistic of this page and no relation of one to a question's depth.** The only run of the new code before this page is an eight-question smoke test of the extraction on MuSiQue's first dev questions (pass flags only; the file was deleted, nothing was read from it but the range of `agree`, 0 to 5, printed by the harness).

## 2. The question, and the program

**Question.** UL4 found that a per-question depth rule is where the factor lies (an oracle that reads the golds needs 1.12 to 4.1 times 2,500 nodes on MetaQA, MuSiQue and WebQSP) and that its ladders stop above 2,500 on the two big ones. Does any signal that reads **no label** choose the depth well enough, with the same parameters on all seven datasets?

**Variables** (all shared by the seven datasets: universality is the constraint that there is one copy of each).
- a statistic s from the five of section 3, and a sign sigma in {+1, -1} (deeper for larger, or smaller, values);
- k - 1 breakpoints tau_1 < ... < tau_{k-1} on the signed statistic, chosen from the statistic's nine pooled deciles (every dataset weighted equally: a number, not a per-dataset calibration), k in {2, 3, 4};
- one level j_b of the universal ladder per bin b (y[b, j] in {0, 1}, sum_j y[b, j] = 1).

**The universal ladder.** M walk nodes kept beyond the base: M in (0, 100, 200, 400, 800, 1,200, 1,600, 2,000, 2,500, 3,500, 5,000, 7,500, 10,000, 15,000, 20,000, 30,000) and the uncut pool (17 levels; M = 0 is the base alone). The base and the walk are each dataset's own PL6 pick (a fixed name: MuSiQue `rrf1000_b400+FULL:h3_bf_v20000~cs-fu`, MetaQA `rrf1000_b25+STRUCT:h3_c100_v5000`, WebQSP `rrf1000_b400+FULL:h3_bf_v40000~cs-vr`, 2Wiki `rrf1000_b400+FULL:h1_c100~s20`, HotpotQA `rrf1000_b25+STRUCT:h1_c100~cs`, MultiHop-RAG `rrf1000_b25+FULL:h1_c100`, SQuAD `rrf1000_b15`, which has no walk and takes its one pool at every level). A level is a prefix of the walk's own ranked list, so the ladder is nested. **M is a node count, the same number for every dataset.**

**Constraints and objective.**
- **P1 (the call).** minimise t subject to: ALL_d >= 0.95 for every dataset d (ALL = the share of questions whose pool holds every gold, on dev); mean nodes_d <= t for every dataset d; the levels non-decreasing in the bin index (a monotone map; it is a constraint, not a tuning choice); one level per bin. t is the worst dataset's mean nodes; the secondary term (a factor 1e-7 on the sum of the datasets' mean nodes) only breaks ties.
- **P2 (beside).** maximise r subject to ALL_d >= r for every d and mean nodes_d <= N for every d, N in {2,500; 3,500; 5,000}.
**Solver, and what is exact.** For fixed (s, sigma, k, breakpoints) P1 is a small mixed-integer program (k times 17 binaries and t) solved to proven optimality by branch and bound (scipy `milp`, HiGHS). The breakpoints are **enumerated exhaustively** over the declared grid: 5 statistics times 2 signs times (9 + 36 + 84) breakpoint sets for k = 2, 3, 4 = 1,290 programs (fewer for a statistic with fewer distinct deciles), each preceded by its LP relaxation, which prunes a candidate that cannot beat the best so far (valid for a minimum). **The answer is therefore the exact optimum over the declared policy class**; it says nothing about policies outside it (a combined statistic, other breakpoints, a different level set). The LP relaxation and its gap are kept as sensitivity (the dual view: how much of t is the integrality of the levels), and the uniform rule (k = 1) is solved by enumeration over the 17 levels. UL4's LP is a lower bound on every program of this kind; **constraining it can only worsen it, so UL4's NOT_EXISTS on the link rule stands and this page does not revisit it.**

**The null (selection is on 500 dev questions and a policy class of about 1,300).** The whole search is repeated 20 times with the statistic shuffled within every dataset (seed 20261126), which keeps every marginal and breaks the signal. A real saving counts only if it beats the best saving of every one of the 20 shuffles.

**The oracle and the uniform, beside.** The oracle is UL4's layer Q on the new ladder (the least mean nodes per dataset at ALL 0.95 when a question may take the cheapest covering level; it reads the golds and is a bound, never a method). The uniform is the one level that every dataset takes (k = 1). **Closure** phi_d = (uniform nodes_d - policy nodes_d) / (uniform nodes_d - oracle nodes_d) per dataset, printed where the denominator exceeds 1 node.

## 3. The statistics (all label-free; declared now)

All five read the pool sizes or the stored first-stage lists and vectors of the question; none reads a label, a gold or a link weight.

1. **log10_pool**: log10 of the uncut pool's size (base plus the whole walk).
2. **growth**: the uncut pool's size divided by the base's size.
3. **cos1**: the float64 inner product of the question's stored vector with the stored vector of the dense first-stage list's first node.
4. **gap15**: cos1 minus the same product for the list's fifth node.
5. **agree**: how many of the dense list's first five nodes are among the splade list's first five (0 to 5).

**Flag (the hub line is closed, and statistics 1 and 2 are the nearest thing to it).** They are walk-size counts: the number of nodes the question's seeds reach. They are **not a hub correction**: they change no link, no weight, no rank and no pool member; they only choose how many of the walk's ranked nodes a question keeps. A large pool can mean seeds on hubs, and the program may find that. If the user reads this as a hub step, statistics 1 and 2 are dropped and the program is re-read on 3 to 5 alone; each statistic's best is printed beside, so the user can see what each carries.

Not declared, named: the question's token count, the number of corpus titles matched in the question text (needs the R.mention rule applied to a question, a new code path), the first-stage score of the splade list, two-statistic combinations.

## 4. Data, and the two stages

**Stage D (dev).** `zul5st.py extract` evaluates, per dataset, the 17 cells of the ladder (103 cells over the seven; the declared list's sha256 is `37ad6402a67a528051d1866e44ff546f29aa2e4b6f4e475dfce6a2946daa09da`, pinned in the code) on the dev carve (PL6's s1sel carve, the whole carve of each dataset: the three parts of MuSiQue and WebQSP stack to its full dev count; first stage frozen) using the same engine as PL2 to PL6 (it imports zpl6g.py and edits nothing), and writes the three extracted statistics per question. Cells that PL2 to PL7's dev grids already hold (the base, the uncut pool and any cut at the same M) must equal them element for element (G5-1). The solve (`zul5cp.py solve`) runs on dev only and writes `outputs/ul5/dev_solve.json`, **including the call of section 7, before any eval extraction exists.**

**Stage E (eval, once).** Only if the dev call is not NO_SIGNAL: the same extraction on the eval carve, and `zul5cp.py grade` applies the dev-chosen policy (the same statistic, breakpoints and levels, nothing refitted) to it and writes `eval_grade.json` once (the script refuses a second run).

## 5. Expectations, stated before any number (a failed one is reported, not explained away)

- **E1 (oracle on the finer ladder).** MuSiQue's oracle falls below UL4's 3,382 (finer cuts) but stays above 2,500; WebQSP's stays above 3 times 2,500 (its family reaches 0.95 on dev only at V = 40,000); MetaQA's stays within 15 percent of 2,804. If MuSiQue's or WebQSP's oracle falls to 2,500 or below, the 2,500 bound is a depth-allocation problem for that dataset and not a walk problem.
- **E2 (the signal is weak).** The best policy closes less than half of the gap from the uniform to the oracle on MuSiQue and on WebQSP (phi < 0.5), and the worst dataset's t stays above 2,500.
- **E3 (which statistic).** The best statistic is one of the two walk-size counts (1, 2), not a first-stage score statistic (3 to 5).
- **E4 (cost of universality).** The same program with private levels per dataset (same breakpoints) is at least 10 percent cheaper on the worst dataset than the universal one. (Computed beside for the chosen policy.)
- **E5 (shrink).** On eval the chosen policy's ALL is below its dev ALL by 0.01 or more on at least three datasets.
- **E6 (null).** The best real saving beats the best saving of all 20 shuffles. **What would surprise me:** a universal policy with worst-dataset t at or below 2,500 on dev (the bound would then be a signal problem, solved), or the real best saving inside the null's range (no signal at all, with no label to blame).

## 6. Gates (a failure stops that dataset's reading; none of its numbers is read)

- **G5-1 (data).** `zul5cp.py gates`: per dataset the dev extraction has the same questions in the same order with the same gold counts as the filed dev grids; every cell it shares with a filed dev grid (at least the base and the uncut pool) equals it element for element (present and size); the ladder is nested (presence and size do not fall as M grows); every head carries one code sha256 (the extraction's and PL6's pinned engine); the cells' sha256 equals the declared one.
- **G5-2 (cells).** `zul5st.py cells`: 103 cells and the declared sha256 (the code stops otherwise).
- **G5-3 (labels unread).** `zul5st.py selftest` on two datasets: the three extracted statistics are identical with the source's labels removed, and the functions that compute them contain no read of a label.
- **G5-4 (nested).** The same selftest: presence and size do not fall along the ladder on the first questions.
- **G5-5 (solver).** `zul5cp.py selftest`: P1 and P2 equal brute force on toy instances (every monotone level assignment); the LP relaxation is a lower bound; a deliberately wrong objective is caught; the monotone constraint cannot lower t; the uniform equals P1 with one bin; the deciles are invariant to a within-dataset shuffle; a planted signal is found; the parsimonious choice does not pick more bins than the best.
- **G5-6 (universality).** One y per program, applied to all seven datasets: true by construction of the program and checked in the report (the same levels and breakpoints are printed once).

## 7. Calls, fixed before any number

**Dev call (written by `solve`).** Let t_U be the uniform's worst-dataset mean nodes (k = 1) and t* the chosen policy's. **The choice** among the best policy per (statistic, sign, k): the smallest k whose t is within 3 percent of the best t over k <= 4, then the least t.
- **NO_SIGNAL** if the chosen policy saves less than 10 percent over the uniform (1 - t*/t_U < 0.10), or the best real saving does not beat the best saving of all 20 shuffles.
- **SIGNAL_AT_BOUND** if it is not NO_SIGNAL and t* <= 2,500.
- **SIGNAL_PARTIAL** otherwise (a real saving and t* above 2,500: the call prints t*, the per-dataset nodes and ALL, phi_d, which datasets sit above 2,500, and what each statistic's own best reaches).

**Eval call (written by `grade`, once).** The chosen policy on eval: per dataset ALL with its Wilson 95 percent interval and mean nodes, beside the uniform's. **EVAL_HOLDS** iff every dataset's ALL is at least 0.93 (0.95 less a selection shrink of 0.02, fixed now) and the worst dataset's mean nodes is at most 1.10 times its dev t*. **AT_BOUND** is printed separately (worst nodes at most 2,500). A NO_SIGNAL dev call grades nothing on eval.

Effect sizes are printed beside every call (savings in nodes and in percent of the uniform, per dataset).

## 8. Code, rows, outputs

`outputs/mp_unified/zul5st.py` (new, mine; imports zpl6g.py and edits nothing; numpy only) and `outputs/mp_unified/zul5cp.py` (new, mine; numpy and scipy; imports zul4lp.py for the oracle). Host rows: one `extract --carve dev` per dataset (MuSiQue and WebQSP in three parts), `--host`, run as `python outputs/host_ops/pylib_run.py outputs/mp_unified/zul5st.py ... --host`. Outputs only under `outputs/ul5/` (npz stays uncommitted; the heads, `gates.json`, `selftest*.json`, `dev_solve.json`, `eval_grade.json` and the report are committed). The pool code reads no label and nothing here edits a pinned file.

## 9. What follows

The dev call and, if graded, the eval call go into a results section of this page, then to the user with the bounds and the costs. **Nothing is carried.** If a policy is EVAL_HOLDS on all seven, a per-question depth step is a candidate for PL8's pool rule, **declared on a new page and only after the user says the link set is final** (PL8 is not declared). If NO_SIGNAL, the oracle bound stands as the statement of what a depth rule could buy, and the remaining levers are the link rules (GP1's patterns) and a different signal (the named, not declared, ones of section 3).

## 10. Disclosed

- The five statistics were chosen from what the engine already computes at no extra scoring cost; the first-stage score statistics were chosen because the lists and vectors are stored. I had read UL4's oracle depths and PL6's tables (which says which datasets need depth), and had not seen how any statistic relates to depth.
- The ladder's M values and the deciles' grid were fixed before any number; the ladder's finer cuts below 2,500 on MuSiQue and WebQSP are new (PL2 to PL7's grids hold none below 5,400 and 10,399 nodes there), so UL4's "infeasible at 2,500" on those two was conditional on the cuts that existed.
- The policy class is one statistic, monotone, at most four bins. An exact answer on it is a bound on that class, not on depth rules in general. The null controls the selection over the class on this dev population (not eval's).
- Dev ALL is measured on the same questions the policy is chosen on; the 0.95 constraint is therefore met on dev by construction, and the eval call allows the declared shrink of 0.02.
- The oracle and phi read the golds. They are bounds, labelled so wherever they are printed.

## 11. Dev results (26 November 2026, 08:10 IST; 02:40 UTC)

Read by `zul5cp.py solve` from `outputs/ul5/<d>/ext_dev*.npz` (the host's ten rows; MultiHop-RAG ran on the laptop because its package files are laptop-only on the host, and its two failed host rows are superseded by that run), after the gates and the solver's self-test passed. **Call: NO_SIGNAL.** Nothing is carried; eval is not extracted and not graded (declared, section 7).

### 11.1 Gates

- **G5-1 (data identity with the filed dev grids):** MultiHop-RAG PASS (7 of 7 shared cells equal, 224 questions); SQuAD PASS (1 of 1 shared cells equal, 1,498 questions); HotpotQA PASS (7 of 7 shared cells equal, 1,508 questions); 2Wiki PASS (7 of 7 shared cells equal, 1,496 questions); MuSiQue PASS (5 of 5 shared cells equal, 1,534 questions); MetaQA PASS (7 of 7 shared cells equal, 1,497 questions); WebQSP PASS (5 of 5 shared cells equal, 1,507 questions).
- **G5-3 and G5-4 (statistics read no label; the ladder is nested)** on the host, first 24 questions of MuSiQue and MetaQA: MuSiQue PASS, MetaQA PASS (17 cells each; one code sha 4a0c37d8). The other five datasets were not self-tested separately: the statistics are one code path (`question_stats`) for all seven, and G5-1's nesting and one-code-sha checks cover every dataset.
- **G5-5 (solver self-test):** PASS (9 checks: P1 and P2 equal brute force, a wrong objective is caught, levels stay monotone, k = 1 equals the enumeration, deciles are invariant to a shuffle, a planted signal is found, the choice is parsimonious).

### 11.2 The dev call

Dev questions: MultiHop-RAG 224, SQuAD 1,498, HotpotQA 1,508, 2Wiki 1,496, MuSiQue 1,534, MetaQA 1,497, WebQSP 1,507. A level is M walk nodes kept beyond the base; the nodes below are mean nodes per question.

**Uniform (k = 1): the one level every dataset takes to reach ALL 0.95 is the uncut pool; the worst dataset (WebQSP) has 39,572 mean nodes there at ALL 0.956.** The next level down (M = 30,000) gives WebQSP ALL 0.823, so the uniform has no cheaper step.

**Chosen policy:** statistic `cos1`, sign +, k = 2, one breakpoint at 0.5690 (questions whose first dense score is at most that get M = 15,000, the rest the uncut pool). Worst-dataset mean nodes t* = 39,190 (LP relaxation 38,939). **Saving over the uniform: 1.0 percent** (the best policy of any statistic is this one). 1,060 programs solved to proven optimality, 0 pruned by their LP relaxation.

| dataset | uniform ALL | uniform nodes | policy ALL | policy nodes | oracle nodes (golds read) | closure phi | share of questions in bin 1 / bin 2 |
|---|---|---|---|---|---|---|---|
| MultiHop-RAG | 0.996 | 255 | 0.996 | 255 | 35 | 0.000 | 0.25 / 0.75 |
| SQuAD | 0.952 | 15 | 0.952 | 15 | 15 | - | 0.31 / 0.69 |
| HotpotQA | 0.968 | 85 | 0.968 | 85 | 41 | 0.000 | 0.02 / 0.98 |
| 2Wiki | 0.955 | 911 | 0.955 | 911 | 541 | 0.000 | 0.15 / 0.85 |
| MuSiQue | 0.963 | 20,038 | 0.956 | 18,096 | 1,831 | 0.107 | 0.42 / 0.58 |
| MetaQA | 0.961 | 4,984 | 0.961 | 4,984 | 2,745 | 0.000 | 0.24 / 0.76 |
| WebQSP | 0.956 | 39,572 | 0.952 | 39,190 | 9,301 | 0.013 | 0.02 / 0.98 |

**Cost of universality.** The same bins with a private level per dataset: MultiHop-RAG 125 (levels 100, 100), SQuAD 15 (levels 0, uncut), HotpotQA 76 (levels 0, 100), 2Wiki 908 (levels 800, 800), MuSiQue 15,400 (levels 15,000, 15,000), MetaQA 4,984 (levels 5,000, uncut), WebQSP 39,190 (levels 15,000, uncut). On the worst dataset the private program costs 39,190 against the universal 39,190: the cost is zero there, because the worst dataset is WebQSP in both. (Read after the fact, not an expectation: on MuSiQue the private level costs 15,400 against 18,096, 14.9 percent less.)

### 11.3 The best policy of each statistic (worst-dataset mean nodes at ALL 0.95; saving over the uniform in parentheses)

| statistic | sign | k | breakpoints | levels (M) | t |
|---|---|---|---|---|---|
| agree | + | 2 | 1 | uncut, uncut | 39,572 (0.0 percent) |
| agree | + | 3 | 1, 2 | uncut, uncut, uncut | 39,572 (0.0 percent) |
| agree | + | 4 | 1, 2, 3 | uncut, uncut, uncut, uncut | 39,572 (0.0 percent) |
| agree | - | 2 | -4 | uncut, uncut | 39,572 (0.0 percent) |
| agree | - | 3 | -4, -3 | uncut, uncut, uncut | 39,572 (0.0 percent) |
| agree | - | 4 | -4, -3, -2 | uncut, uncut, uncut, uncut | 39,572 (0.0 percent) |
| cos1 | + | 2 | 0.569 | 15,000, uncut | 39,190 (1.0 percent) |
| cos1 | + | 3 | 0.5355, 0.569 | 15,000, 15,000, uncut | 39,190 (1.0 percent) |
| cos1 | + | 4 | 0.5355, 0.569, 0.5945 | 15,000, 15,000, uncut, uncut | 39,190 (1.0 percent) |
| cos1 | - | 2 | -0.7346 | uncut, uncut | 39,572 (0.0 percent) |
| cos1 | - | 3 | -0.7346, -0.7002 | uncut, uncut, uncut | 39,572 (0.0 percent) |
| cos1 | - | 4 | -0.7346, -0.7002, -0.6754 | uncut, uncut, uncut, uncut | 39,572 (0.0 percent) |
| gap15 | + | 2 | 0.02877 | uncut, uncut | 39,572 (0.0 percent) |
| gap15 | + | 3 | 0.02877, 0.04278 | uncut, uncut, uncut | 39,572 (0.0 percent) |
| gap15 | + | 4 | 0.02877, 0.04278, 0.05577 | uncut, uncut, uncut, uncut | 39,572 (0.0 percent) |
| gap15 | - | 2 | -0.1846 | 10,000, uncut | 39,244 (0.8 percent) |
| gap15 | - | 3 | -0.1846, -0.15 | 10,000, uncut, uncut | 39,244 (0.8 percent) |
| gap15 | - | 4 | -0.1846, -0.15, -0.1274 | 10,000, uncut, uncut, uncut | 39,244 (0.8 percent) |
| growth | + | 2 | 50.03 | 5,000, uncut | 39,533 (0.1 percent) |
| growth | + | 3 | 1, 50.03 | 0, 5,000, uncut | 39,533 (0.1 percent) |
| growth | + | 4 | 1, 1.962, 50.03 | 0, 400, 5,000, uncut | 39,533 (0.1 percent) |
| growth | - | 2 | -200.5 | 5,000, uncut | 39,572 (0.0 percent) |
| growth | - | 3 | -200.5, -100.3 | 5,000, uncut, uncut | 39,572 (0.0 percent) |
| growth | - | 4 | -200.5, -100.3, -50.19 | 5,000, uncut, uncut, uncut | 39,572 (0.0 percent) |
| log10_pool | + | 2 | 4.302 | 15,000, uncut | 39,568 (0.0 percent) |
| log10_pool | + | 3 | 1.176, 4.302 | 0, 15,000, uncut | 39,568 (0.0 percent) |
| log10_pool | + | 4 | 1.176, 1.799, 4.302 | 0, 100, 15,000, uncut | 39,568 (0.0 percent) |
| log10_pool | - | 2 | -4.603 | uncut, uncut | 39,572 (0.0 percent) |
| log10_pool | - | 3 | -4.603, -4.302 | uncut, uncut, uncut | 39,572 (0.0 percent) |
| log10_pool | - | 4 | -4.603, -4.302, -3.7 | uncut, uncut, uncut, uncut | 39,572 (0.0 percent) |

Per statistic, the best over signs and k: `cos1` 39,190, `gap15` 39,244, `growth` 39,533, `log10_pool` 39,568, `agree` 39,572. The five are within 1.0 percent of one another.

### 11.4 P2 beside: the most the worst dataset's ALL can reach at a mean-node cap (no null was run for P2: these numbers claim no signal)

| cap (every dataset's mean nodes) | uniform: level and worst ALL | best policy | its worst ALL |
|---|---|---|---|
| 2,500 | M = 2,000, 0.388 | `gap15`, sign +, k 3 | 0.494 |
| 3,500 | M = 2,500, 0.467 | `gap15`, sign +, k 3 | 0.581 |
| 5,000 | M = 3,500, 0.601 | `agree`, sign +, k 3 | 0.629 |

At 2,500 the uniform's worst dataset (MetaQA) reaches 0.388 and the best policy's worst 0.494; neither is near 0.95.

### 11.5 The null (20 within-dataset shuffles of the statistic, the whole search repeated)

Best t of each shuffle: 38,535, 38,649, 38,670, 38,773, 38,925, 38,957, 39,028, 39,042, 39,071, 39,078, 39,094, 39,102, 39,173, 39,178, 39,179, 39,180, 39,184, 39,186, 39,186, 39,187. Their savings over the uniform: 1.0 to 2.6 percent (mean 1.4). **The real best (saving 1.0 percent, t = 39,190) does not beat the shuffles:** it is worse than every one of the 20. A shuffled statistic finds a 1 to 2.6 percent saving on this population by chance, which is the size of everything the five real statistics find.

### 11.6 Expectations against the outcome (a failed one is reported, not explained away)

- **E1 (oracle on the finer ladder): MuSiQue clause FAILED, WebQSP and MetaQA clauses held.** MuSiQue's oracle is 1,831, below 2,500 (UL4: 3,382); WebQSP's 9,301 is 3.7 times 2,500 (above the 3 times); MetaQA's 2,745 is 2.1 percent under UL4's 2,804 (inside the 15 percent). The small four: MultiHop-RAG 35, SQuAD 15, HotpotQA 41, 2Wiki 541. As the declaration says for a failed MuSiQue clause: the 2,500 bound on MuSiQue is a depth-allocation problem (an oracle that picks the depth per question meets it), not a walk problem.
- **E2 (the signal is weak): held.** Closure phi on MuSiQue 0.107, WebQSP 0.013; t* = 39,190, far above 2,500.
- **E3 (the best statistic is a walk-size count): FAILED.** The chosen statistic is `cos1` (a first-stage score); the best count reaches 39,533 and the best score 39,190, 0.9 percent apart, which the null's range covers.
- **E4 (private levels at least 10 percent cheaper on the worst dataset): FAILED.** 0.0 percent (the worst dataset is WebQSP in both programs).
- **E5 (eval shrink): not read** (a NO_SIGNAL dev call grades nothing on eval).
- **E6 (the best real saving beats all 20 shuffles): FAILED.**
- **The declared surprise: the real best saving inside the null's range.** It happened: the real best (39,190) is not below any shuffle's best (38,535 to 39,187). There is no label to blame: the statistics read none.

### 11.7 What this says, and what it does not

1. **None of the five label-free statistics tells a question how deep its walk must go, on this dev population and in this policy class.** The best saving over the uniform is 1.0 percent, the shuffled statistics find 1.0 to 2.6 percent, and the closure of the gap to the oracle is 0.107 on MuSiQue and 0.013 on WebQSP.
2. **The worst dataset fixes the answer.** P1 is a minimax, and WebQSP's family reaches ALL 0.95 only on the uncut pool (39,572 nodes, ALL 0.956; at M = 30,000 its ALL is 0.823), so the uniform's t, every policy's t and every shuffle's t sit within 3 percent of 39,572. Even the oracle, which reads the golds, needs 9,301 on WebQSP (3.7 times 2,500): **no per-question depth rule, with or without labels, brings WebQSP's family to 2,500; its lever is the links and paths the walk follows, not the depth.**
3. **MuSiQue is a different case.** The oracle needs 1,831 there, under 2,500, against 20,038 for the uniform: a depth signal that existed would pay on MuSiQue; these five do not (phi 0.107). MetaQA's oracle (2,745) sits just above 2,500.
4. **What it does not say.** It is five statistics, one statistic per policy, at most four monotone bins, one dev population (the null controls the selection on that population, not eval's). It does not say no label-free signal exists: the named, not declared, ones (question token count, title matches in the question text, the sparse score, two-statistic combinations) were not tried. The first two statistics are walk-size counts and read no hub attribute; they are not a hub correction.

### 11.8 What follows, and disclosed

- **No eval extraction and no grade** (`zul5cp.py grade` refuses a NO_SIGNAL call). The oracle bound stands as the statement of what a depth rule could buy: WebQSP 9,301, MetaQA 2,745, MuSiQue 1,831 mean nodes, the other four at most 541.
- **PL8 is not declared** (the user has not said the link set is final). The bound the user set (ALL 0.95 at close to 2,500 nodes) is, for WebQSP, a links-and-paths problem (GP1's gold-link patterns are the lever); for MuSiQue the oracle meets it, so it is a depth-signal problem (none found here); for MetaQA the oracle sits about 10 percent above it.
- **Disclosed:** MultiHop-RAG's dev extraction ran on the laptop (3.5 s) because the host has no copy of its package; its two host rows failed with a file-not-found and stay in the feed record as failed. One sentence of section 4 was corrected before any number (83f7f71e: the dev carve is the whole carve, not one part). An 8-question smoke run on MultiHop-RAG preceded the declaration (smoke numbers decide nothing). The host rows' timings: MuSiQue and WebQSP about 27 to 31 minutes per part, the other four under 3 minutes.
- **Files:** `outputs/ul5/dev_solve.json`, `gates.json`, `selftest.json`, `selftest_musique_host.json`, `selftest_metaqa_host.json`, the `ext_dev*.json` heads (npz uncommitted); results table sha in `dev_solve.json`.

