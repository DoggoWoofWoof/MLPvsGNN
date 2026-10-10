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

## What this does not do

- No new model, column, objective, setting, seed or carve. The encoder and the substrate embeddings stay frozen.
- Nothing per dataset. One graph rule (c = 1,024) for all six.
- U1c's screen, its records and its rule are unchanged. U1f writes nothing into outputs/u1c.
- webqsp never trains. Test splits are never read. B1 is never used to choose.

## Results

Not yet run.
