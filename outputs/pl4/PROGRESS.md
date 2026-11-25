# PL4 progress (laptop side), 25 Nov 2026 15:30 local

Declaration: c71f0d7 (docs/PL4_COVERAGE_ROUND3.md sections 0-13; engine outputs/mp_unified/zpl4g.py, pinned, unchanged since).
Results: section 14 first filed for five of seven datasets (c25eddf, INCOMPLETE) and RE-FILED for all seven after the host rows returned
(HotpotQA: grid, select, frozen and e3v confirmations; 2Wiki: grid, select, frozen and e3v confirmations; SQuAD e3v). Old rule CARRY_PL4 (BELOW 0, ABOVE 1 = MuSiQue by the
price clause, LEVEL 6); frontier reading NOT_CARRIED (BELOW 0, ABOVE 0, LEVEL 7). The five laptop datasets' call records are identical to the first filing.

## Laptop reads (frozen lists): all done
MultiHop-RAG, SQuAD, MuSiQue, MetaQA, WebQSP: G-R PASS, dev grid, select, frozen confirmation, call. D1/D2 (`zpl4r.py miss`) on the SHORT three. SQuAD's frozen confirmation ran as
`zpl4r.py confirm-par --ds squad --first frozen --slices 4` (arrays checked equal to the single-process confirm on MultiHop-RAG).

## To re-file again (about 5 minutes of laptop work; no host record is missing now)
1. `python outputs/mp_unified/zpl4g.py call`  (writes outputs/pl4/call.json; reads confirm_frozen + select + grid_eval_frozen_confirm.npz per dataset)
2. `python outputs/mp_unified/zpl4r.py report`  (outputs/pl4/report.md and report.json)
3. `python outputs/pl4/filing/build_section14.py`  (rewrites section 14 of docs/PL4_COVERAGE_ROUND3.md from the records; the hand-written sentences of 14.4 and 14.7 are in the builder and
   written for seven datasets; read the generated text once)
4. Commit: git add -f the page, zpl4r.py and the json/md under outputs/pl4 (never the npz), `git commit -F msg -- paths`.

## Not mine: outputs/host_ops/hostfeed_items.txt, the host, pushes.

## What the seven say
- MultiHop-RAG, SQuAD, HotpotQA, 2Wiki, MetaQA: PL4 adds nothing at R95 (the same cells). WebQSP: a name tie, identical ALL and sizes on every eval question.
- MuSiQue: R95 = `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq`, eval 0.8821 at 2,274.9 against PL3's 0.8788 at 2,455.1: old rule ABOVE by the price clause alone (ALL +0.33 [-0.33, +0.99]); frontier reading LEVEL at all budgets.
- HotpotQA: no PL4 cell moves the dev frontier at any of 14 budgets. 2Wiki: one budget (2,500: +0.20 on dev, -0.01 [-0.14, +0.13] on eval, LEVEL).
- D2: WebQSP's C golds are cut by the per-seed cap at hop 1 (126) and the per-frontier cap at hop 2 (97); MetaQA's by the hop-3 budget (264 of 373); MuSiQue's spread. Next change: a cap that follows the budget (PL5).
