# U1f: U1c's screen on U1e's damped links (GU_1024)

Declared 10 October 2026, about 09:10, before any number of the stage. U1e (docs/U1E_DAMPED_LINKS.md) chose c = 1,024:
the mention links of a name are kept only while the name links at most 1,024 nodes. That graph, GU_1024, is not ∞ (U1d's
GU), so U1c's screen, which runs on GU, is run again on GU_1024. U1c's screen keeps running; U1f is read beside it.

## What runs

U1c's screen (docs/U1C_RETRAIN_ON_U.md, with its amendment 1), unchanged, on another graph. This includes its models, settings, carves,
bases, reads, calls and rule. Only these differ:

- **The graph.** Each look's `structural` family is GU_1024 (outputs/u1e/c1024/<D>/graph_structural_u.npz, checked
  against the sha256 of U1e's build.json) on musique, 2wiki, hotpotqa and webqsp.
- **The pools.** Step 4h's chosen configuration (outputs/step4h/choice.json, W1), unchanged, built on GU_1024. Each
  pools file must equal U1e's coverage record of that graph (outputs/u1e/c1024/coverage_<D>.json), as U1c's must equal
  step 4h's.
- **The roots.** Everything is written under outputs/u1f (pools, looks, caches, chains, links, gate).
- **metaqa and squad.** GU_1024 is GU on these two: U1e removed no metaqa link (its build record: `same_as_gu`), and
  squad has no graph. Their pools, looks and caches are U1c's, byte for byte. They are hard-linked into outputs/u1f
  (`u1f.py adopt`, a copy where a link fails), which costs no disk. Each is adopted only after U1e's build record and U1d's
  sha256 agree. U1f's gate then checks them as it checks the rest. Their chain and link builds run again under U1f's roots.
- **The fits** are scr-u1f-zrc, scr-u1f-zrc-hp, scr-u1f-zsp and scr-u1f-zsp-hp, on the host's card under U1c's caps.

Each fit is decided against the same bases as U1c's: scr-zrct(-hp) for zrc, scr-zspg(-hp) for zsp (today's graph and
pools, the same device). The calls are relz's pair and its re-call under the seed null.

All of it runs through outputs/mp_unified/u1f.py. The wrapper moves U1c's module roots (scripts/u1c_host.py,
look_u1c.py, look_u1c2.py, zu1c.py) and calls their code unchanged. Its selftest checks every moved root, that step 4h's
W1 is still the configuration, and the adoption's refusals.

## The rule, fixed before any number

U1c's, for U1f on its own:
- **ADOPT_FOR_FULL_RUN:** for each arm, the re-call has no LOSS on metaqa, musique, webqsp or squad (in-domain or
  zero-shot), and the two arms together have at least one GAIN anywhere.
- **NOT_ADOPTED:** otherwise.

The 2wiki and hotpotqa reads are reported in full. A LOSS there is the measured price of the hyperlink-free headline
and goes to the gap ledger.

**Which graph the full run uses**, also fixed now:
- **Only U1f passes:** GU_1024.
- **Only U1c passes:** GU.
- **Both pass:** GU_1024, U1e's choice. It was chosen on s1sel coverage, before either screen, and is the smaller graph
  (pools 0.735 of step 4e's against 0.787).
- **Neither passes:** no full run on either graph. The reads go to the gap ledger.

The screens' numbers do not choose between the graphs when both pass. Choosing by them would choose on the eval carves.

## Disk

- **Estimate.** U1f's own bytes are the looks, caches, chains and links of musique, 2wiki, hotpotqa and webqsp: at
  most about 25 GB. That is U1c's for those datasets, scaled by the pool ratio.
- **The gate.** The looks wait on `outputs/host_ops/wait_disk.py --need-gb 150` (item u1f-disk): twice the estimate
  above the feeder's 100 GB floor.
- **Before the gate opens.** Clean-up 12 (U1d's link shards and U1e's unchosen graphs, 22 GB, the user's command)
  comes first. Everything before the looks (pools, adoption) is small and runs at once.
- **Amended 12 October 2026, about 13:05 (corrected to the commit time), before any number of U1f.** The user's disk rule: 90 GB free is the point
  where a clean-up survey happens, never a reason to hold jobs. The 100 GB hold had idled the host from 10 October.
  - The feeder now holds heavy writers only when the drive is near full: below 30 GB free it sends only light items,
    and below 15 GB nothing.
  - So u1f-disk waits for 60 GB: the estimate of 25 GB above that 30 GB line, with margin.
  - Nothing else in the stage changes.

## What this does not do

- No new model, column, objective, setting, seed or carve. The encoder and the substrate embeddings stay frozen.
- Nothing per dataset. One graph rule (c = 1,024) for all six.
- U1c's screen, its records and its rule are unchanged. U1f writes nothing into outputs/u1c.
- webqsp never trains. Test splits are never read. B1 is never used to choose.

## Results

Filed 13 October 2026, about 00:30. **NOT_ADOPTED.**
- Both arms are NO_GAIN, filed and under the seed null: no GAIN in 24 reads.
- Each arm also has a LOSS on a dataset the rule protects:
  - zrc on musique, in-domain on L-hotpotqa's fit (−0.0098 against a 0.0075 floor);
  - zsp on webqsp zero-shot, L-musique's fit (−0.043).
- U1c did not pass either, so by the rule no full run happens on either graph.

Files: `outputs/screen/scr-u1f-{zrc,zsp}{,-hp}.{md,json}`, `scr-u1f-{zrc,zsp}-pair{,-recall}.{md,json}` (host paths
cut from the re-call records).

delta R@5 against each arm's base on today's graph and pools, with the re-call (W = WITHIN; zs = zero-shot):

| arm | fit | metaqa | squad | musique | webqsp (zs) | hotpotqa | 2wiki |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zrc | L-musique | +0.0074 W | +0.0013 W | −0.0259 W (zs) | +0.0037 W | −0.0586 LOSS | −0.0098 LOSS |
| zrc | L-hotpotqa | +0.0059 W | +0.0023 W | **−0.0098 LOSS** | +0.0142 W | −0.0452 LOSS (zs) | −0.0108 LOSS |
| zsp | L-musique | +0.0040 W | +0.0008 W | −0.0133 W (zs) | **−0.0427 LOSS** | −0.0648 LOSS | −0.0100 LOSS |
| zsp | L-hotpotqa | +0.0065 W | +0.0026 W | +0.0038 W | +0.0152 W | −0.0353 LOSS (zs) | −0.0127 LOSS |

The re-call changed one call per arm: L-musique's musique zero-shot read, filed LOSS, is WITHIN its 0.072 floor.

What this says, against U1c (docs/U1C_RETRAIN_ON_U.md):
- **Damping the links changes little.** Capping each mention target's in-degree at 1,024 moves no read to a GAIN.
  - metaqa still leans up (+0.004 to +0.007), under its floor.
  - zsp's webqsp zero-shot loss shrinks (−0.056 to −0.043) but stays a LOSS.
  - zrc picks up a musique loss on L-hotpotqa's fit (U1c: −0.001).
- **The hyperlink price stands (to the gap ledger).** Both arms still lose hotpotqa (−0.035 to −0.065, U1c −0.040 to
  −0.063) and 2wiki (−0.010 to −0.013, U1c −0.011 to −0.020).
  - The LLM-free links recover the hyperlinks as edges (docs/U1A_B1_WITHOUT_HYPERLINKS.md), on both U1d's and U1e's
    graphs.
  - The trained models do not get the hyperlinks' value back from them.

What follows, by the rule: no full run on either graph. Both bases keep today's graph and pools. The U1 reads go to
the gap ledger. The ranking gap on musique is taken up by D4 (docs/DIAG_LINK_WORDS.md), on today's graph.
