# PL4 progress (laptop side), 25 Nov 2026 12:30 local

Declaration: c71f0d7 (docs/PL4_COVERAGE_ROUND3.md sections 0-13; engine outputs/mp_unified/zpl4g.py, pinned, unchanged since).
Results: section 14 filed for FIVE of seven datasets (call INCOMPLETE) in the first results commit; it is re-filed when HotpotQA and 2Wiki return.

## Laptop reads (frozen lists): all done
MultiHop-RAG, SQuAD, MuSiQue, MetaQA, WebQSP: G-R PASS, dev grid, select, frozen confirmation, call. D1/D2 (`zpl4r.py miss`) on the SHORT three. SQuAD's frozen confirmation ran as
`zpl4r.py confirm-par --ds squad --first frozen --slices 4` (arrays checked equal to the single-process confirm on MultiHop-RAG).

## What is back from the host (fetched by the coordinator) and what is still needed
Back: gr_{musique,hotpotqa,2wiki}_host.json (PASS); HotpotQA dev grid parts + select.json; e3v confirmations of MuSiQue, MetaQA, WebQSP.
Needed: HotpotQA confirm_frozen.json + grid_eval_frozen_confirm.{json,npz}, confirm_e3v.json; 2Wiki dev grid parts, select.json, confirm_frozen, confirm_e3v; SQuAD confirm_e3v.json.

## To re-file when they land (about 5 minutes of laptop work)
1. `python outputs/mp_unified/zpl4g.py call`  (writes outputs/pl4/call.json; reads confirm_frozen + select + grid_eval_frozen_confirm.npz per dataset)
2. `python outputs/mp_unified/zpl4r.py report`  (outputs/pl4/report.md and report.json)
3. `python <scratchpad>/build_pl4_results.py`  (rewrites section 14 of docs/PL4_COVERAGE_ROUND3.md from the records; the script is in the session scratchpad, not in the repo)
4. If HotpotQA or 2Wiki is SHORT on dev: `python outputs/mp_unified/zpl4r.py miss --ds <d> --host` is a host job; on the laptop it needs the dev vectors (not here).
5. Commit: git add -f the page, zpl4r.py and the json/md under outputs/pl4 (never the npz), `git commit -F msg -- paths`.

## If this run is cut off
Everything above is on disk; nothing running now needs a restart. Not mine: outputs/host_ops/hostfeed_items.txt, the host, pushes.

## What the five read say
- MultiHop-RAG, SQuAD, MetaQA: PL4 adds nothing at R95 (the same cells). WebQSP: a name tie, identical ALL and sizes on every eval question.
- MuSiQue: R95 = `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq`, eval 0.8821 at 2,274.9 against PL3's 0.8788 at 2,455.1: old rule ABOVE by the price clause alone (ALL +0.33 [-0.33, +0.99]); frontier reading LEVEL at all budgets.
- Frontier call and old call: INCOMPLETE (BELOW 0, ABOVE 1 old / 0 frontier).
- D2: WebQSP's C golds are cut by the per-seed cap at hop 1 (126) and the per-frontier cap at hop 2 (97); MetaQA's by the hop-3 budget (264 of 373); MuSiQue's spread. Next change: a cap that follows the budget.
