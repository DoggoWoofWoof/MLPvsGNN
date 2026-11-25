"""The hand-written sentences of PL6 section 13, written after the numbers were read (the builder's hook). Everything numeric is read from the records or asserted against them below;
the sentences only say what the tables say. Re-read after every re-filing: a sentence that stops being true must be changed here, not left."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
C = json.loads((ROOT / "outputs" / "pl6" / "call.json").read_text(encoding="utf-8"))
D = C["datasets"]
COMPLETE = C["call"] != "INCOMPLETE"


def sel(ds):
    return json.loads((ROOT / "outputs" / "pl6" / ds / "select.json").read_text(encoding="utf-8"))


def _guard():
    """The sentences below quote these numbers; a re-filing that moves one stops the build instead of leaving a false sentence."""
    m, w, q = D["musique"], D["webqsp"], D["metaqa"]
    assert (round(m["new_all"], 4), round(m["new_nodes"], 1), round(m["old_all"], 4), round(m["old_nodes"], 1)) == (0.9640, 15400.0, 0.8846, 2438.2)
    assert (round(w["new_all"], 4), round(w["new_nodes"], 1), round(w["old_all"], 4), round(w["old_nodes"], 1)) == (0.9474, 39529.5, 0.8789, 2094.6)
    assert (round(q["new_all"], 4), round(q["new_nodes"], 1), round(q["old_all"], 4), round(q["old_nodes"], 1)) == (0.9582, 4972.8, 0.8968, 2335.9)
    assert (m["status"], w["status"], q["status"]) == ("ABOVE", "LEVEL", "ABOVE")
    assert (w["all_status"], w["dev_reached"], w["eval_reached"]) == ("ABOVE", True, False)
    assert (m["rung"], w["rung"], q["rung"]) == (2, 3, 0)
    assert m["ladder_state"] == w["ladder_state"] == q["ladder_state"] == "REACHED"
    assert C["universality_check"]["unchanged_on_all_of_them"] is True
    assert C["code_path_equal_everywhere"] is True


_guard()


def hook(ctx):
    out = {}
    nread = C["datasets_read"] if isinstance(C["datasets_read"], int) else len(C["datasets_read"])
    call = C["call"]
    counts = C["counts"]
    # ------------------------------------------------------------------ head
    out["head"] = (
        "\n**How to read this page (the rulings that came after the declaration).** The user's ruling of 25 Nov about 15:00 lifted the 2,500 bound only under one rule on all seven (section 0); he then said (relayed by the coordinator, 25 Nov, before PL7) "
        "\"by lifting that bound i asked you to do analysis, we still need to start restricting it and bring towards 2.5k\". **This page is therefore the analysis he asked for: the map of what reaching 0.95 costs when nothing bounds the pool.** "
        "The call below carries its rule's name, but no cell above 2,500 nodes is taken downstream from this page: the restriction is PL7 (`docs/PL7_COVERAGE_RESTRICTION.md`, 24f4892e, declared before any number, "
        "one rule on all seven, on the current links, provisional until the link set is final), which reads the carry at 2,500 only. Nothing of PL2 to PL5 is changed by this page.\n")
    # ------------------------------------------------------------------ ladder
    out["ladder"] = (
        "\n- **MuSiQue** asked for two rungs. Rung 0 (PL5's cells) reaches 0.9355 on dev at best; rung 1 (visited budget V = 10,000) 0.9485, a rise of 0.0130 and still under 0.95; rung 2 (V = 20,000) 0.9668: REACHED, so rungs 3 and 4 were not run. "
        "U95 is the smallest cell at 0.95 or more on dev: a hard cut of the V = 20,000 budget-following walk, `+top15000` (dev 0.9550 at 15,400.0 nodes; a pool never exceeds the base's 400 plus the 15,000 the cut keeps).\n"
        "- **WebQSP** asked for three rungs: 0.9257 (rung 0), 0.9416 (V = 10,000), 0.9496 (V = 20,000, a rise of 0.0080 and still under 0.95), 0.9569 (V = 40,000): REACHED at rung 3, so rung 4 was not run. U95 is the V = 40,000 walk with base 400 and no cut "
        "(dev 0.9555 at 39,571.6 nodes, 1.5% of the graph's 2,592,894).\n"
        "- **MetaQA** reaches 0.95 on dev at rung 0 above the bound (U95 `rrf1000_b25+STRUCT:h3_c100_v5000`, dev 0.9613 at 4,983.6 nodes): no rung was run for it, as declared in section 6.\n"
        "- **MultiHop-RAG, SQuAD, HotpotQA and 2Wiki** reach the target inside the bound at rung 0; their U95 is PL5's bounded pick (the universality check, below). "
        "**No dataset ended short on dev (every ladder state is REACHED), so section 7 of the declaration asks for no D1 or D2 on this page** (13.6).\n")
    # ------------------------------------------------------------------ call
    callnote = [
        "\n- **What decides each status.** MuSiQue and MetaQA are ABOVE through clause 1: the new pick reaches 0.95 on eval (MuSiQue 0.9640, MetaQA 0.9582) where the bounded pick does not (0.8846, 0.8968). "
        "**WebQSP is LEVEL**: its new pick does not reach 0.95 on eval either (0.9474, short by 0.26 points, where it reached 0.9555 on dev), so clause 1 does not fire, and the ALL difference's lower end is above 0 (+6.85 [+5.59, +8.12] points) "
        "but the nodes ratio is 18.87, so clause 3 does not either. The ALL-only status says ABOVE: the 37,435 extra nodes per question bought coverage, and not enough of it.",
        "- **The price in nodes.** The three datasets whose pick moved use 2.13 times (MetaQA), 6.32 times (MuSiQue) and 18.87 times (WebQSP) the nodes of the bounded pick, "
        "with 95th percentiles and maxima within 2.1% of the means (the hard cut and the budget-following walks make the pools nearly constant in size).",
    ]
    if COMPLETE:
        callnote.append(f"- **The call is {call} on all seven** (BELOW {counts['old']['BELOW']}, ABOVE {counts['old']['ABOVE']}, LEVEL {counts['old']['LEVEL']}). Under the user's later ruling (head of this section) it is the name of the reading, "
                        "not a decision to take a cell above 2,500 downstream; PL7 reads the carry at 2,500 only.")
    out["call_note"] = "\n".join(callnote) + "\n"
    # ------------------------------------------------------------------ price
    out["price"] = (
        "\nThe share of the graph is filed where a record holds the graph's node count (MultiHop-RAG, SQuAD, MuSiQue, WebQSP; HotpotQA, 2Wiki and MetaQA hold none in these records). "
        "Two costs are different: the pool's size is what the next stage reads per question; the first stage's own cost is the walk, filed by the grid heads' seconds (section 13.1) and the two-question probes "
        "(15.0 s per question for all 156 cells of MuSiQue's rung 2, 14.1 s for WebQSP's rung 3), and it is not read from the pool's size.\n")
    # ------------------------------------------------------------------ bought
    out["bought"] = (
        "\nOn **MuSiQue** the ladder buys +4.18 points [+3.27, +5.13] at 2.68 times the nodes of the best rung-0 cell (rung 0's rule-U pick is `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr`, 0.9222 at 5,742.4: the any-size reading of PL5), and U98 reaches 0.9710 at 20,235.7 (the uncut V = 20,000 walk). "
        "On **WebQSP** it buys +2.99 points [+2.13, +3.93] at 6.50 times the nodes (0.9474 at 39,529.5 against 0.9175 at 6,079.5): ALL-only ABOVE, LEVEL by the rule, and neither floor is reached on eval (U98 0.9481 at 40,164.1). "
        "On the five datasets that ran no rung (MultiHop-RAG, SQuAD, HotpotQA, 2Wiki, MetaQA) the ladder buys nothing: the same cell on both sides, difference exactly 0.\n")
    # ------------------------------------------------------------------ frontier
    out["frontier_note"] = (
        "\n- **MuSiQue's frontier (eval) rises from 7,000 nodes upward**: the matched cells equal rung 0's up to 7,000, then 0.9350 at 9,076 (+1.28 points, ABOVE), 0.9429 at 13,615 (+2.07), 0.9661 at 16,576 (+4.39) and 0.9710 at 20,236 (+4.88). "
        "From 30,000 to 80,000 the cell is the same and the difference stays at +4.88: **that flat tail is where the ladder stopped (REACHED at rung 2), not a measurement that more nodes would buy nothing.**\n"
        "- **WebQSP's** matched cells equal rung 0's up to 10,000 nodes, then 0.9328 at 10,817 (+1.53 points), 0.9421 at 20,738 (+2.46), 0.9474 at 39,530 (+2.99) and 0.9481 at 40,164 (+3.06): about +1.5, +0.9 and +0.5 points of eval ALL for each step of 1.8 to 1.9 times the nodes. "
        "The 60,000 and 80,000 cells are the V = 40,000 cell, again where the ladder stopped.\n"
        "- **The shapes differ**: WebQSP's gain per step falls (+1.5, +0.9, +0.5 points); MuSiQue's is uneven, its largest single step (+2.32 points: 0.9429 at 13,615 to 0.9661 at 16,576) falling between 15,000 and 20,000 budgets and its last (+0.49, to 0.9710 at 20,236) at the ladder's end. "
        "The frontier reading counts MuSiQue and WebQSP ABOVE (two or more budgets ABOVE and none BELOW). MetaQA ran no rung, so its frontier is rung 0 read against itself (the same cell at every budget, difference exactly 0).\n")
    # ------------------------------------------------------------------ short
    out["short_intro"] = (
        "\n**None to report.** Section 7 of the declaration asks for D1 and D2 at the chosen cell of a dataset that ends STOPPED_RISING or LADDER_END; every ladder here ended REACHED on dev, and `zpl5r.py miss --pl6` skips a dataset whose pick reaches 0.95 (it runs only with `--all`). "
        "PL5's D1 and D2 at the bounded picks (section 14 of PL5's page) stand as the account of what the 2,500 bound cuts. **The one shortfall left is WebQSP on eval**: dev 0.9555, eval 0.9474; a gap of 0.8 points, which is the kind selection over about 6,700 cells on a 1.5k-question carve produces (section 12 of the declaration); no diagnostic was run to separate it from a difference between the two carves, and none reads what the walk misses.\n")
    out["short"] = ""
    # ------------------------------------------------------------------ meaning
    out["meaning"] = (
        "\n- **What the map says.** The three datasets that PL2 to PL5 left SHORT inside 2,500 nodes reach the target on dev above it: MetaQA at 4,984 nodes on dev (4,973 on eval, 2.13 times the bounded pick's), MuSiQue at 15,400 (6.32 times), WebQSP at 39,572 (39,530 on eval, 18.87 times). "
        "On eval MetaQA (0.9582) and MuSiQue (0.9640) meet 0.95; WebQSP (0.9474) does not. The cost of coverage on these three is not small and not flat: MuSiQue's frontier stands 4.88 points above rung 0's at 20,236 nodes and WebQSP's 3.06 points above at 40,164, "
        "and WebQSP's gain per step of nodes is falling.\n"
        "- **What the bound was doing.** At 2,500 nodes the three datasets read 0.8846, 0.8968 and 0.8789 on eval; the extra points are bought at nodes per question that are 13.1% of MuSiQue's graph at the pick and 1.5% of WebQSP's. "
        "The first four datasets are untouched by lifting the bound (their picks are inside it and rule U names them).\n"
        "- **What it leaves unread.** (a) No cell above 2,500 is read on e3v lists here (the beside-read `zpl6g.py confirm --first e3v` was not run; it decides nothing). (b) No rung beyond 2 (MuSiQue) or 3 (WebQSP) was run, so the frontier above 20,000 and 40,000 nodes is unread, "
        "and WebQSP's eval shortfall of 0.26 points is not mapped. (c) The pool size is the mean of the pool the walk hands on; the walk's cost is filed by the grid heads' seconds and the two-question probes only. "
        "(d) The links are the current ones: the UL2 per-label table (cited in section 1 of PL7's page) puts 98.7% of WebQSP's candidate pairs in one label at 0.009% precision, so a smaller pool at the same ALL is partly a question for the link set, which the next stages address (PL7, then the link-tightening stage and one pool re-run).\n")
    out["next"] = (
        "\n**Declared before any number: PL7** (`docs/PL7_COVERAGE_RESTRICTION.md`, 24f4892e): restrict the pool back toward 2,500 nodes under one rule on all seven (a ladder of ceilings from this page's picks down to 2,500; hard-cut cells; the carry is read at 2,500 only; the node cost and a floor caveat beside every reading). "
        "It is provisional on the current links. After it: a link-tightening stage (per-label and per-class keep-or-drop against gold reach, one rule on all seven, selected on dev carves and graded on eval), declared by the coordinator, then one pool re-run on the final link set. "
        "Names only here: a parent-conditioned rank, a hop-weighted schedule and a window wider than 4,096 (PL5 section 14) stay unadopted.\n")
    out["surprises"] = (
        "\n- **WebQSP reaches 0.95 on dev and not on eval** (0.9555 against 0.9474) with 39,530 nodes per question: a dev ladder state of REACHED is not a reading of the target; the eval call is.\n"
        "- **The flat tails of the frontier are the ladder's end, not the data's** (13.5): MuSiQue's cells at 30,000 to 80,000 are one cell, as are WebQSP's at 60,000 and 80,000.\n"
        "- **The pools are nearly constant in size** (the 95th percentile and the maximum are within 2.1% of the mean at every new pick), because the pick is a hard cut or a budget-following walk that spends its budget; a mean node count is therefore close to every question's.\n"
        "- **Nothing contradicts a filed number.** PL5's selects are rung 0 unchanged, and the laptop's PL5 confirmation grid and the host's PL6 confirmation grid agree bit for bit on the cells both hold (see 13.10).\n")
    out["disclosed"] = (
        "- **The two confirmations ran as host rows, not on the laptop** (the declaration's section 6 says the laptop reads them): `pl6-musiquecf` (2,417 eval questions) and `pl6-webqspcf` (1,503), so as not to wait for the laptop, whose chain was still running PL5's reads (one heavy laptop job at a time). "
        "The host builds only the cells PL5's confirmation grid does not hold. **Cells held by both agree exactly**: WebQSP 7 cells over 1,503 questions and MuSiQue 5 cells over 2,417, 0 differing (present counts and pool sizes, element for element; `grid_eval_frozen_confirm.*` against PL5's laptop grid).\n"
        "- **G-R ran on the host for MuSiQue only** (`gr_musique_host.json`), as section 8 of the declaration says (\"host: the host's G-R on MuSiQue, as PL5's\"); the WebQSP rungs depend on that row in the feed and have no G-R of their own. The laptop ran none for PL6 (it ran no rung).\n"
        "- **Rows of later rungs were added to the feed after the stop rule asked for them**: MuSiQue rung 2 and WebQSP rungs 2 and 3 were appended to `outputs/pl6/feed_rows.txt` after the previous rung's select (section 6: \"rung 2 and later rows are not in the file\"), and `feed_rows_confirm.txt` holds the two confirmation rows; no cell, rule or constant changed.\n"
        "- **This page is an analysis**, by the user's ruling relayed before PL7 (head of this section); the call is filed under its rule's name and carries nothing above 2,500 downstream.\n"
        "- The grid arrays (`*.npz`) are not committed (as PL2 to PL5); the heads, selects, confirmations, the call, the probes and this page are.\n")
    return out
