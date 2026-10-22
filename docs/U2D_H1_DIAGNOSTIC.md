# U2d's hit@1 diagnostic: where the gold sits when hit@1 misses

Declared 22 October 2026, before any U2d fit was trained or read. Code: `outputs/mp_unified/zu2d_h1.py`, and two stages
added to `outputs/mp_unified/zu2d_modal.py` (`h1` and `fetch-h1`). Everything else is imported unchanged. It decides
nothing. Its numbers are development numbers.

## 1. Why

U2d fixed it in advance (docs/U2D_REFIT_ON_U2E.md, section 9): "In every case: the hit@1 diagnostic on metaqa and
webqsp (where the gold sits in the order when hit@1 misses)." On these two datasets R@5 is capped by the metric, since
many questions have more than five golds. So hit@1 is their headline (U2d, section 8).

A read keeps only two things per question: the golds among the top five, and whether the top row is gold. It does not
keep where the first gold sits. So the diagnostic needs its own pass.

## 2. What it computes

**The fits.** U2d's sixteen seed-0 fits:
- 6 on the base build: zrckp, zspk and zg6, on L-musique and L-hotpotqa;
- 10 on the new build: zrckp, zspk, zg6, zrckpv and zspkv, on both splits.

**The carves.** metaqa's and webqsp's s1eval carves of each fit's own build.

**The pass.** Each fit is scored with p@swa on the device it was fitted on. The pass repeats the fit's own read and
enters it through the same wrappers as `zu2d.py read`:
- the build's roots (`on_u2d`), P1c2's card and cache root, and the gate;
- rmatch's read;
- lean_screen's arm patch (the arm's model and carve classes).

The read's own flags and batches (the fit's read rows cap) are used. While the pass runs, only `lean_gpu.read_cmd` is
swapped for the diagnostic's version. Nothing the fit or its read wrote is opened for writing.

**The four orders.** RRF (the base z-score), twin0 and gnn0 (the read's three references), and the fit's p@swa. Each
order is `lean_gpu.top_hit`'s: descending score, NaN last, ties broken by pool position.

**Per question:**
- the pool's rows;
- the golds the pool holds;
- the question's golds (R@5's denominator);
- in each order, the rank of the first gold (1 is a hit@1; 0 when the pool holds no gold);
- RRF's rank of the model's top row;
- RRF's rank of the model's first gold.

## 3. The checks, before any number is reported

**REPRODUCED.** In each carve, the following must all equal the fit's read file (`reads/<dataset>__s1eval.npz`):
- each question's golds in the top five, and whether its top row is gold, for p@swa and the three references;
- the question ids and the gold totals;
- the scores the read kept for the first 64 questions, bit for bit.

Otherwise the carve is DIFFERS, and its summary is not reported.

**Inside the pass.** For every question and order, rank 1 must match `top_hit`'s hit, and the golds ranked 1 to 5
must match `top_hit`'s count.

**The fit.** The fit must be U2d's (`zu2d.fit_stamp`), and its read must be stamped by U2d on the same device. The pass
runs on Modal's L40S from the pinned image, as the reads did.

## 4. What is reported (deciding nothing)

Per fit and carve, over the questions with a gold (the questions R@5 grades):
- hit@1, RRF's hit@1, and the cap (the share of questions whose pool holds a gold);
- pool misses (no gold in the pool) and rank misses (a gold in the pool, but not at rank 1);
- for the rank misses, where the first gold sits: rank 2, 3–5, 6–10, 11–20, 21–50, or beyond 50, with the median;
- for the rank misses, two shares:
  - how often the model's top row is RRF's top row;
  - how often RRF's top row was gold;
- hit@1 against RRF's: both, the model only, RRF only, neither;
- R@5 and its cap, beside.

The same distribution for RRF's own rank misses is reported beside.

The s1eval carves are the ones the compares read. No decision or selection reads this diagnostic. It says where the
next forms should aim: the order near the top, the order deep in the pool, or the pool.

## 5. How it runs

- **On Modal.** It runs in the workspace that holds the fit: `zu2d_modal.py --stage h1 [--build B] [--fits N,...]`,
  one GPU container per fit, after that fit's read.
- **What it writes.** The fit's files are read only. The diagnostic writes only `/u2d/out/h1/<fit>/`, linked as
  `outputs/u2d/h1/<fit>/`. That folder holds one `<dataset>__s1eval.npz` per carve and `h1.json`. A filed `h1.json` is
  never rewritten.
- **Fetching.** `--stage fetch-h1` copies those files to the laptop.
- **The summary.** `python outputs/mp_unified/zu2d_h1.py summary` writes `outputs/u2d/h1/summary.json` and `.md` from
  the fetched files. Each npz must match the sha256 its `h1.json` names.

The selftest checks the following on toy pools, with ties, NaN scores, empty pools and questions without a gold:
- each order's ranks equal a direct sort of the rows;
- the ranks agree with `top_hit`;
- the summary's shares are correct;
- a changed read makes the check DIFFERS.

The encoder stays frozen. webqsp never trains. Test splits are never read.

## 6. Results

Not run yet.
