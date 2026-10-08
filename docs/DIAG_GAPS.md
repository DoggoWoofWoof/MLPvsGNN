# Gap diagnostics: GNN against MLP, and both against the pool and the published numbers

Declared 9 October 2026 at about 01:40, before any of its numbers. The user: "keep the diagnostics ready of gnn vs mlp
and sota vs gnn what we are missing". This is a diagnostic. It decides nothing, trains nothing and changes no model,
pool or rule. Its numbers are development numbers.

## 1. Questions

1. **GNN against MLP.** On the same splits and questions, where does the GNN track's base gain or lose against the
   MLP's base, and which questions carry the difference?
2. **Where every model misses.** Of each model's R@5 miss, how much is outside the pool (no ranking can recover it)
   and how much is in the pool but ranked below five?
3. **Against the published numbers.** How do our reads sit beside the filed published numbers, metric by metric? What
   would step 4e's pools change, and what does each published setup have that ours does not?

## 2. What it reads

- **The two bases.**
  - GNN track: zsp, zrm plus one propagation step (message passing; docs/FULL_ROUND23.md, ADOPT).
  - MLP: zrm.
  - Both are CPU fits on all six splits: round twenty-three's full run (`outputs/full_zsp/fits/S` and
    `outputs/full_zsp/zrm/S` for J5, L-metaqa, L-squad and L-2wiki) and its screen (`scr-zsp` and `scr-zrm-cpu` on
    L-musique; `scr-zsp-hp` and `scr-zrm-cpu-hp` on L-hotpotqa).
- **The reads.** Each fit's reads on the six s1eval carves, candidate p@swa (lean_screen's `reads/<ds>__s1eval.npz`):
  per question, the gold count, the golds in the top five, hit@1, and rrf's.
- **The pools** the reads ran on, from step 1's cache (`outputs/step1/cache/<ds>/s1eval`). Per question:
  - its pool size;
  - its golds in the pool;
  - the walk depth class of its deepest in-pool gold (qdepth's classes: a seed, the first hop at 1, 2 or 3, or
    unreached).
- **Context only.** Step 4e's coverage on s1eval (`outputs/step4e/coverage.json`, the chosen A3 at k = 2).
- **The published numbers,** as filed in docs/MUSIQUE_DIAGNOSIS.md and docs/M3A_SOTA_ARCHAEOLOGY.md, each with its own
  metric and setup.

## 3. What it reports (`outputs/mp_unified/gapdiag.py run`, into `outputs/gapdiag/zsp-zrm/gaps.{json,md}`)

1. **Per split and dataset:**
   - rrf's, the MLP's and the GNN's R@5;
   - the GNN-minus-MLP delta on R@5, FC@5 and hit@1, with a 2,000-resample question bootstrap interval;
   - the questions each side wins on R@5.
2. **The delta by bucket, and each bucket's share of the summed delta.** Buckets:
   - gold count (1, 2, 3+);
   - golds in the pool (all, some, none);
   - the deepest in-pool gold's depth;
   - rrf's own R@5 (zero, partial, full);
   - pool-size quartile;
   - hop count (musique only, from its ids).
3. **Each model's miss, split in two:** outside the pool, 1 − ceiling, where ceiling = min(golds in the pool, 5) /
   golds; and in the pool but ranked out, ceiling − R@5. Also the share of questions where FC@5 is reachable at all.
4. **Beside the published numbers:**
   - the GNN track's J5 read and its zero-shot read, as R@5 / FC@5 / hit@1;
   - today's ceiling and step 4e's coverage;
   - each published row, with its metric and setup.

   The traps stand. GraphER's PR@K is set coverage (our FC@5, never R@5). NuTrea's Hit@1 is answer accuracy with topic
   entities assigned, an oracle we refuse. HippoRAG 2's R@5 comes from a 7B encoder over a corpus a tenth of ours.
   Nothing in that table is a direct comparison.

## 4. Order and later runs

- **One CPU item** (`gd-zsp-zrm`, 1 CPU, 8 GB), ahead of step 4e's looks, about ten minutes. Nothing waits on it.
- **The same script reruns on any later pair of bases** (`--pair SPLIT=GNN_FIT,MLP_FIT`) into its own folder:
  - zgn against zfs, round twenty-five's same-feature pair, when its reads land;
  - a step 4e screen's fits against today's (with `--cache-root outputs/step4e/cache`), when they land.

  Each rerun is a diagnostic too.
- **No MLP-against-GNN number from here is cited before M4** (the M3 firewall). These tables steer the two tracks'
  next screens.

## Results

**First run (9 October 01:48, `outputs/gapdiag/zsp-zrm/gaps.{md,json}`).** zsp (the GNN track) against zrm (the MLP),
on the six splits and the six s1eval carves. A diagnostic; it decides nothing.

1. **GNN against MLP.**
   - In-domain, zsp adds +0.000 to +0.010 R@5. Most of it comes from 2wiki and hotpotqa: questions with three or more
     golds, and partly-found questions (2wiki partial +0.008 in-domain, +0.022 zero-shot).
   - Zero-shot, zsp adds more: metaqa +0.030 and 2wiki +0.014, with webqsp +0.009 to +0.027.
     - metaqa's gain sits on questions whose golds are one hop from the seeds (+0.140); its two-hop golds lose
       (−0.022).
     - 2wiki's gain sits on questions with three or more golds (+0.040).
   - zsp loses on hotpotqa zero-shot (−0.007) and on musique in J5's fit (−0.007).
   - **Message passing changes neither track's musique.** By hop count, both models lose the same questions. Their
     ranked-out miss is 0.35 to 0.41 of R@5 on every split, whichever model reads.
2. **Where every model misses.**
   - **metaqa and webqsp lose most outside the pool.** Today's ceiling is 0.811 and 0.765, so 0.189 and 0.235 of R@5
     no ranking can recover. Step 4e's pools hold every gold for 0.953 and 0.909 of s1eval's questions (0.988 and
     0.966 of golds).
   - **musique loses most inside the pool.** It has 0.079 outside the pool, against 0.35 ranked out. Its 4-hop
     questions read R@5 0.33 against a ceiling of 0.80.
   - **squad, hotpotqa and 2wiki in-domain** lose 0.02 to 0.04 outside the pool and 0.07 to 0.10 ranked out.
   - **Zero-shot, the ranked-out share grows.** metaqa grows 0.03 → 0.62 to 0.65, 2wiki 0.09 → 0.17 to 0.18, and
     hotpotqa 0.08 → 0.13 to 0.14. The models do not carry their ranking to an unseen graph; the pool is unchanged.
3. **Against the published numbers** (never direct; each setup differs, and the traps stand).
   - **FC@5 against GraphER's PR@5:** hotpotqa 0.849 against 0.78 to 0.79; 2wiki 0.721 against 0.425 to 0.441;
     musique 0.259 against 0.216 to 0.256.
   - **musique R@5 0.566 against HippoRAG 2's 0.747** and NV-Embed-v2's 0.697. Both use a 7B encoder over a corpus a
     tenth of ours; this is the one published row we sit well below.
   - **KB.** metaqa hit@1 is 0.888 in J5's fit, against NuTrea's answer Hit@1 of 0.99 with topic entities assigned (an
     oracle we refuse). webqsp is read only zero-shot.
