"""The hand-written sentences of PL5 section 14, written after the numbers were read (the builder's hook). Everything numeric is read from the records; the sentences below only
say what the tables say. Re-read after every re-filing: a sentence that stops being true must be changed here, not left."""
from pathlib import Path

import pl5_tables as T

ORDER, LABEL = T.ORDER, T.LABEL


def hook(ctx):
    # the builder's own globals (the hook receives only build()'s locals): read them the same way
    C = T.call()
    READ = [d for d in ORDER if C["datasets"][d]["status"] != "NOT_READ"]
    NR, COMPLETE = len(READ), len(READ) == 7
    D = lambda d: C["datasets"][d]  # noqa: E731
    out = {}
    # guards: the sentences below quote these numbers; a re-filing that moves one stops the build instead of leaving a false sentence
    w, m = D("webqsp"), D("musique")
    assert (round(w["all_diff"] * 100, 2), round(w["nodes_ratio"], 4), w["status"]) == (9.85, 0.8486, "ABOVE")
    assert (round(m["all_diff"] * 100, 2), round(m["nodes_ratio"], 4), m["status"]) == (0.25, 1.0718, "LEVEL")
    assert [round(w["frontier"][b]["all_diff"] * 100, 2) for b in ("500", "1000", "1500", "2000", "2500")] == [3.46, 5.46, 10.11, 7.39, 9.85]
    assert [round(m["frontier"][b]["all_diff"] * 100, 2) for b in ("1500", "2000", "2500")] == [2.23, 1.37, 0.25]
    assert (m["frontier_status"], w["frontier_status"]) == ("ABOVE", "ABOVE")
    done = f" (all seven read: {C['call']}; frontier reading {C['call_frontier']})" if COMPLETE else ""
    # what decides the old call, clause by clause
    out["above_note"] = (
        "\n- **What decides each status.** WebQSP is ABOVE through clauses 2 and 3: NEW (`dense_top200+STRUCT:h3_bf_v2000~cs-fu`, a budget-following walk of PL5's own cells) is +9.85 points [+8.18, +11.65] of ALL "
        "over PL4's pick and uses 0.85 times its nodes [0.83, 0.87]; neither reaches 0.95 on eval (0.8789 and 0.7804). MuSiQue is LEVEL: +0.25 points [-0.37, +0.83] with 1.07 times the nodes [1.0705, 1.0729], an interval that includes 0 bought with more nodes. "
        "On the other datasets read NEW is PL4's own cell (difference exactly 0, ratio exactly 1)" + (", SQuAD included (`rrf1000_b15`, the same cell on dev)" if COMPLETE else "") + ". "
        "**Every chosen cell is inside 2,500 on eval frozen lists** (the largest is MuSiQue's, 2,438.2); no clause fires BELOW." + done + "\n")
    out["frontier_note"] = (
        " The two ABOVE readings are MuSiQue (+2.23 points [+1.53, +2.98] at 1,500 nodes and +1.37 [+0.54, +2.23] at 2,000, LEVEL at 500, 1,000 and 2,500: +0.25 [-0.29, +0.83]) and WebQSP (ABOVE at all five budgets, +3.46 to +10.11). "
        "HotpotQA's two budgets with a different cell read -0.07 [-0.24, +0.11] and -0.14 [-0.30, +0.01]: LEVEL, both intervals include 0.")
    out["bought"] = (
        "\nRead from `outputs/pl5/report.md` (\"What the PL5 cells bought\" and \"What each option bought\", section 14.10). **The dev frontier of all cells differs from PL2 to PL4's on three datasets only.** "
        "WebQSP: +4.6 points at 200 nodes, +3.9 to +4.3 from 300 to 750, +7.1 at 1,000, then +9.8, +9.2 and +10.3 at 1,500, 2,000 and 2,500 nodes (0.7731 to 0.8759 at 2,500); above the bound +9.1, +8.0, +6.2 and +7.2 at 3,000 to 6,000. "
        "MuSiQue: +2.09 at 1,500, +1.17 at 2,000 and 2,500, +0.98 at 3,000, +0.52 at 4,000, +0.20 at 5,000 and nothing from 6,000. HotpotQA: +0.07 at 200, 300, 400 and 1,500 and +0.13 at 2,000, one or two questions of 1,508. "
        "MultiHop-RAG, SQuAD, 2Wiki and MetaQA: nothing at any of the 19 budgets. On eval the WebQSP gain reads +9.85 points [+8.18, +11.65] at 2,500 (ABOVE), MuSiQue's +2.23 and +1.37 at 1,500 and 2,000 (ABOVE) and +0.25 at 2,500 (LEVEL), HotpotQA's -0.07 and -0.14 (LEVEL).\n"
        "**By option, on the three datasets whose PL4 cell is SHORT inside the bound** (R95 of the option's own cells, the highest ALL where none reaches; dev): every one of the seven options beats PL4's cell on WebQSP (0.7810 to 0.8759 against 0.7731; the budget-following cap with any key and the fused order give 0.8759 at 2,094.2, the cosine order 0.8666 at 2,499.5); "
        "four of seven beat it on MuSiQue (the budget-following cap with any key, the cosine order and the families-take-turns order 0.8898 at 2,464.3; seeds of 10 and 20 0.8827; PL4's is 0.8781; the fused order 0.8761 and the one-hop whole-window walks 0.8449 are lower, the 4,096-entry window ties); "
        "and none beats it on MetaQA (the best option cell is 0.7882 at 2,276.8, PL4's 0.8978). On the four datasets that reach 0.95 inside the bound no option makes a smaller cell reach it, so their picks stay PL4's.\n")
    out["short"] = (
        "\n**Read** (the same 300 dev questions as PL4's D1 and D2, at PL5's chosen cells): in the three SHORT datasets no missing gold is ranked out (R 0) and only two lie beyond the uncapped three-hop reach (U: MetaQA's 2 of 375); "
        "every other missing gold is cut by a cap or a budget inside a reach the walk could take. "
        "**WebQSP**: 31 of 300 questions miss a gold, 81 golds (2.6 per such question; PL4's cell missed 245 C golds on the same questions); **42 are cut by a cap** (hop 1: 2, hop 2: 35, hop 3: 5), **35 by the visited budget** (hop 2: 19, hop 3: 16) and 4 are upstream. "
        "PL4's D2 had put 223 of its 245 at the per-seed and per-frontier caps and none at the budget: the budget-following cell has moved most of the cut from the caps to the budget and removed two thirds of the missing golds at 2,094 nodes. "
        "**MuSiQue**: 36 of 300 questions miss a gold, 39 golds (about one per such question; PL4's cell 37), 18 at the cap at hop 2, 9 at the budget at hop 2, 12 upstream: no one stage dominates, as at PL4's cell. "
        "**MetaQA** (unchanged cell): 33 questions, 375 golds (11.4 per such question), 264 of the 373 C golds cut by the budget at hop 3. "
        "Upstream means the gold is no neighbour of any node the walk kept, so every route to it passes through a node an earlier hop dropped (PL4 section 10).\n")
    out["price"] = (
        "\n**What the any-size reading says** (eval, frozen lists; section 14.5): the 0.95 floor's cell is PL4's own on six of seven datasets (difference exactly 0) and WebQSP's moves: the PL5 cells reach 0.9175 at 6,079.5 nodes where PL2 to PL4's reach 0.8390 at 6,029.5 (+7.85 points [+6.39, +9.45], "
        "nodes ratio 1.0083: LEVEL by the rule, ABOVE by the ALL interval), and neither reaches 0.95 on dev or eval. MetaQA reaches 0.95 only above the bound (0.9582 at 4,972.8 nodes, the same cell as PL4's); MuSiQue stays at 0.9222 at 5,742.4 (SHORT on dev, best dev 0.9355), "
        "2Wiki's 0.98 floor is never reached (0.9785 at 6,275.0 on eval). PL6 (`docs/PL6_COVERAGE_ROUND5.md`) reads the same question with a ladder of visited budgets.\n")
    out["meaning"] = (
        f"\n- **At 7 of 7 read the old rule is {C['call']} and the frontier reading is {C['call_frontier']}**, " + (
            "so the two agree: neither finds a BELOW; the old rule's call rests on one ABOVE (WebQSP), the frontier reading's on two (MuSiQue at 1,500 and 2,000 nodes, WebQSP at all five budgets). "
            "The page still does not choose between them (section 6): which governs is the user's decision.\n" if COMPLETE else "the page is incomplete.\n"))
    out["meaning"] += (
        "- **A reading of what the datasets say.** On five of the seven the PL5 pick is PL4's (MultiHop-RAG, SQuAD, HotpotQA, 2Wiki and MetaQA: the same cell, difference exactly 0). "
        "MuSiQue's moves by +0.25 points with 7% more nodes (LEVEL), and **WebQSP's moves by +9.85 points with 15% fewer nodes**, from a budget-following walk (PL5's own cells, `bf`): the one change of this round that the old rule credits. "
        "It is also the change D2 pointed at (PL4's D2 put 223 of WebQSP's 245 attributed C golds at the per-seed and per-frontier caps). "
        "**Eval, frozen lists: four datasets meet 0.95 inside the bound (MultiHop-RAG 0.9690, SQuAD 0.9554, HotpotQA 0.9634, 2Wiki 0.9609); three do not (MuSiQue 0.8846, MetaQA 0.8968, WebQSP 0.8789), short by 6.54, 5.32 and 7.11 points.**\n"
        "- **What it leaves unread.** (a) The e3v lists were read for MultiHop-RAG, HotpotQA and 2Wiki only (none declared for the other four); no chosen cell exceeds the bound there. (b) The three SHORT datasets are short *inside* 2,500 nodes: PL6 reads what lifting the bound costs, and PL7 what restricting back toward 2,500 costs. "
        "(c) Every number here is on the current links; the link-tightening stage and one pool re-run are declared after the link set is final.\n")
    out["surprises"] = (
        "\n- **The WebQSP pick is smaller than the one it replaces** (2,094.6 nodes against 2,468.1) and 9.85 points higher; the old rule's price clause 2 (ratio upper end below 1) and clause 3 (ALL lower end above 0 with the ratio upper end at or below 1) both fire.\n"
        "- **The two readings differ on MuSiQue**: the old rule says LEVEL (+0.25 points, ratio 1.07, an interval that includes 0) and the frontier reading says ABOVE (two budgets, 1,500 and 2,000, with the lower ends above 0); both are filed under their own names.\n"
        "- **No PL5 cell is competitive on MetaQA**: the best option cell inside the bound reads 0.7882 on dev against PL4's 0.8978; MetaQA's D2 (section 14.6: 264 of its 373 attributed C golds cut by the hop-3 budget) is the account of what its pool still misses.\n"
        "- **SQuAD's pick is 15 nodes and stands 0.54 points above the floor on eval** (0.9554 on 11,873 questions; dev 0.9519): the smallest cell that reaches 0.95 sits close to it by construction, so a 15-node pool is at the floor, not above it with room.\n")
    # the next changes: names only
    out["next"] = (
        "\nFrom this page's section 12, unchanged by the results: a parent-conditioned rank (a node's cosine plus its best parent's), aimed at the hop-3 budget cut of MetaQA and MuSiQue; a hop-weighted schedule; a window wider than 4,096 "
        "if it paid on MuSiQue and WebQSP; the weight values of NER and KNN rows as a key; label-ordered caps once a UL2 layout exists on every dataset. "
        "**Declared separately, before any number: PL6 (`docs/PL6_COVERAGE_ROUND5.md`, 744811e6)** reads the user's ruling of 25 Nov (\"i do not mind just rules should be universal\") as one selection rule on all seven with no node bound "
        "and a ladder of visited budgets (10,000 to 80,000) for the datasets that stay SHORT; it reads this round's selects as its rung 0. "
        "The user's later ruling (relayed, 25 Nov) made that an analysis (\"by lifting that bound i asked you to do analysis, we still need to start restricting it and bring towards 2.5k\"); "
        "**PL7 (`docs/PL7_COVERAGE_RESTRICTION.md`, 24f4892e)**, declared before any number, restricts the pool back toward 2,500 nodes under one rule on all seven, on the current links (provisional until the link set is final).\n")
    out["disclosed"] = (
        "- **The laptop's driver was re-ordered once, mid-run (25 Nov 15:39).** `run_chain.sh` (MuSiQue's G-R, then MultiHop-RAG, MetaQA, WebQSP, MuSiQue, SQuAD) was stopped while WebQSP's three grid parts were running (the parts were left running, "
        "not restarted) and replaced by `outputs/pl5/run_chain2.sh`, at the coordinator's request, so that MuSiQue's dev grid and select exist before WebQSP's frozen confirmation and the host's PL6 rows can start earlier. The commands, the rule, the cells, "
        "the seeds and the carves are the same; only their order changed (`outputs/pl5/run_chain.log` records both).\n"
        "- **WebQSP's G-R ran twice**: PASSED at 15:01 (1,763 s) and, because `run_chain2.sh` called `run_ds.sh webqsp` without `no-gr`, again at 17:48 (3,265 s, the machine paging), PASS with 0 present and 0 size mismatches both times; "
        "`gr_webqsp.json` is the second record (same code, same cells, same 400 + 150 questions). The repeat cost about 55 minutes of laptop time and changed nothing.\n"
        "- **SQuAD's G-R also ran twice**: PASSED at 14:03 (1,058.0 s; the dev grid and select of the first chain were filed at 14:47) and again at 22:32 (3,087.5 s), when `run_chain2.sh` called `run_ds.sh squad no-confirm` without `no-gr` "
        "(the dev grid parts existed and were not rebuilt); PASS with 0 present and 0 size mismatches both times, `gr_squad.json` is the second record. The repeat cost about 31 minutes of laptop time and changed nothing.\n"
        "- **SQuAD's frozen confirmation ran as four question slices** (`zpl5r.py confirm-par --ds squad --slices 4`, 11,873 eval questions, PL4's method) from 22:34 and was merged at 00:05 on 26 Nov; the slices ran at roughly 5 to 10 minutes per 300 questions beside other work, so their seconds are upper bounds.\n"
        "- **D1 and D2 for MuSiQue and WebQSP ran together** (`zpl5r.py miss`, one process each, after SQuAD's slices ended; MetaQA's ran earlier, at 13:57).\n"
        "- The laptop's WebQSP grid ran at 7.8 to 8 s per question per part with the machine paging at about 4,000 pages a second (free memory under 1 GB beside three 4 to 5 GB processes): its seconds are upper bounds, as PL4's were.\n"
        "- 6,078 cells are selected over on 1.5k-question dev carves (224 for MultiHop-RAG): a winner's dev ALL is optimistic, so every call is a paired interval on eval; the dev numbers of section 14.3 are the selection's, not a result.\n"
        "- The bound is not changed by this page: the price at any size (14.5) is beside the calls, and a result that needs more than 2,500 nodes is filed as such, not carried. PL6 (744811e6) reads the user's ruling on the bound.\n"
        "- D3 (section 1.2) was computed before the declaration commit on PL4's chosen cells; D1 and D2 here are run at PL5's chosen cells of the SHORT datasets, post-hoc, and decide nothing.\n"
        "- The grid arrays (`*.npz`) are not committed (as PL2's to PL4's); the heads, selects, confirmations, the call, the diagnostics and the report are.\n")
    return out
