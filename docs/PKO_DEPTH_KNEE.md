# PKO depth knee: stage 3 of the 30 October program (R@5 against pool depth, with the models that exist)

Declared 31 October 2026, before any cut read. Stage 3 of [PROGRAM_2026_10_30.md](PROGRAM_2026_10_30.md) section 7. Code:
`outputs/mp_unified/zknee.py` (self-test passes). Outputs only under `outputs/knee/`. Nothing is fit; no sealed folder is written.

## 1. The question

The user, 30 October: "why hasnt better pools given better ranking till now". The program's answer so far (section 1 there): where reach
was already about 0.96 (hotpotqa, 2wiki) the room is in the ranking, and a pool helps a ranker only where reach binds and the extra rows
are not noise to it. That is a reading of old filings. This stage measures it on PKO's pools: take the two sealed ML1 fits that read PKO's
eleven carves, cut each question's pool to a depth, score once, and read R@5 and the pool's reach at every depth. The knee is the smallest
depth at which R@5 is within a tolerance of the uncut read. It says how much of the pool the rankers use, which decides whether stages 5 to 7
(a learned cut, EM) are worth their GPU time.

## 2. The cut

A pool is B (the base and the seeds) and then the expansion, `kt_cos[:keep]`, in the file's order (PKO_PICK_O_POOLS_UNDER_WQ.md section 2).
Depth D keeps B whole and the first max(0, D - |B|) rows of the expansion; a question whose pool is no longer than D is not cut. Depths:
**500, 1,000, 1,500, 2,000, 3,300 rows**, and no cut. The scores come from one pass over the full pool; the cut decides which rows may take
one of the five places (and which count toward reach). It does not rebuild the pool.

## 3. The reads

- Fits: `scr-ml1-zspk-s0` (the GNN, zspk) and `scr-ml1-zrckp-s0` (the MLP, zrckp), ML1's sealed folders, copied to `outputs/knee/fits/`
  with their shas checked. Both are WQ-musique's seed 0: trained on webqsp, squad, hotpotqa and 2wiki; musique and metaqa are read zero-shot.
- Carves: all six s1eval carves with PKO's pools, read through ML1's own route (`zml1s`), with `lean_gpu.read_carve` replaced by a copy that
  also scores p@swa and the three references (rrf, twin0, gnn0) at every depth from the one set of scores.
- Check: with no cut the copy is compared with ML1's own read file of the same fit, question by question. A carve whose no-cut R@5 differs
  from ML1's by more than 0.002, or whose file is missing, makes the fit NOT_REPRODUCED and its cuts are not read. The device must be the one
  ML1 read on (the host's GPU).
- Placement: the host's GPU, two items (one per fit), ahead of ML1's unsent fits; each is a read, with the cap and share of ML1's own fits.

## 4. What this is not

- **The features are the full pool's.** The cached features (rank, z-scores, pooled statistics) were built on the full pool, so a cut pool's
  rows keep the context of the uncut one. A ranker trained and used on a cut pool would see other statistics. This stage measures how much
  of the uncut pool's ranking the top rows carry, not the R@5 of a rebuilt smaller pool.
- **Two fits, one seed, one split.** ML1's other fits (seeds 1 and 2, the hotpotqa split where musique is in-domain, zg6r, zrckpt) are not
  read here; reading them is an amendment. Numbers are development numbers.
- **Reach** is the share of a question's golds inside the kept rows. It is reported, not decided.

## 5. Reading, fixed before the numbers

Per fit and carve, from `p@swa`, R@5 over questions with a gold; EPS = 0.01:

- **knee(eps)**: the smallest of the five depths whose R@5 is at least R@5(no cut) - eps (None when only the uncut pool reaches it); reported
  at 0.01 and 0.02. **common knee**: the smallest depth at which no carve of the fit loses more than eps.
- **DEPTH_FLAT** (for a fit): the common knee at 0.01 is 2,000 rows or fewer. The rankers use no more than 2,000 rows' worth of the pool;
  reach beyond it is not converted by these two rankers.
- **DEPTH_PAYS** (for a carve): R@5(no cut) exceeds R@5(2,000) by more than 0.01. Rows beyond 2,000 buy R@5 there.
- **NOISE_AT_DEPTH** (for a carve): R@5 at some depth below no cut exceeds R@5(no cut) by more than 0.01. The extra rows cost the ranker.
- **What follows.** DEPTH_FLAT on both fits and DEPTH_PAYS on no carve: a pool of about 2,000 rows suffices for R@5 with these rankers, the
  stage-5 selector has no recall to protect above it, and the route from better pools to better ranking runs through the ranker (ML1's refits,
  U2u). DEPTH_PAYS on a carve: that carve's reach beyond 2,000 is being used, so a cut costs R@5 there, and stage 5 must protect it (a depth
  that depends on the question). NOISE_AT_DEPTH on any carve: a ranker trained on cut pools is the next form, not a larger pool.
- The three can hold on different carves; the report lists each per carve and per fit and makes no overall call beyond this list.

## 6. Files

    outputs/mp_unified/zknee.py              run (host GPU), report (CPU), --selftest
    outputs/knee/fits/<fit>/                 the copied fit, reads/<ds>__<carve>.npz, knee_check.json
    outputs/knee/knee.json, knee.md          the report

## 7. Results

(to be filed after the reads)

## 8. Amendment A1, 31 October 2026, before any cut was read

The user, 31 October: improving the GNN is the priority and the GNN must be its own model, not built over the MLP. zspk is the MLP plus one
propagation step (G6, docs/G6_FREE_GNN.md section 1); the GNN of record is zg6r. So the read of zspk (`knee-zspk-s0`) is withdrawn from the
queue before it ran, and the report lists that fit NOT_RUN. Sections 3 and 5 read as "the fits read" (zrckp now). zg6r's reads go through zrm,
not lean_gpu, so its depth read needs its own patched read; it is declared as A2 with that code, the same reading rule (section 5) and the same
check against ML1's file, once the code is written and self-tested. The GPU order changed with it: ML1's zg6r items now run ahead of the lean
pair's. No rule, depth or tolerance changes.
