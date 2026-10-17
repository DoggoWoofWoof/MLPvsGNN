# Verdict audit, 17 October 2026

The user, 17 October, after the learned offsets were closed on a diagnostic: "make sure we dont reach a premature
verdict like this ever", then "based on this rule check all the stuff we did". The rule is docs/SCREENS.md, section 3b:
a verdict closes only the form it tested, an idea closes only when its signal has reached both models as an input, a
diagnostic never closes an idea, and gains with a loss call for a form that can back off.

This file checks every closed verdict in the repository's records against that rule. **No filed number, record or
verdict changes.** Sealed records stay as filed. This file says what each verdict closes, and which ideas are reopened.

## 1. How each verdict was checked

Four questions, answered from the verdict's own record (the doc named in each row, and its `outputs/` files):
1. **What form was tested?** An input, a training change, a loss, a model change, or a pool or link rule; on which
   model (step 1's, zret's, zrm's, zrc's, zsp's, or today's pair: the MLP zrckp and the GNN zspk).
2. **What kind of read was it?** A model screen or full run (a ranking model trained and read), or a diagnostic (no
   ranking model trained).
3. **Was there a signal?** A GAIN past the seed null's floor on any read, or a diagnostic lift in the right direction.
4. **Has the signal reached both of today's models?** As a fixed input, or as a training change tried on them.

Each verdict then gets one of three statuses:
- **Form closed:** no signal survived the seed null, or the signal already reaches both of today's models.
- **Superseded:** a later form carries the signal into today's models. The row names it.
- **REOPEN:** a signal, closed on a small loss or a diagnostic, that never reached both of today's models. The row
  names the next form. Section 2 lists them in the order they will be declared.

Sections 3 to 5 cover every closed verdict in 68 rows; a row groups the verdicts on one idea. **12 ideas are
reopened** (section 2), and one more is conditional on R1. Two closed
ideas had already been reopened before the rule and both passed: zdp became G4's zspk2, and P1c2's KIND_T pools became
P1c3's input. They are what the rule asks for.

## 2. Reopened, in the order they will be declared

Each is its own declared file, committed before any of its numbers, screened by docs/SCREENS.md's rule (each arm
against its own model's card fits: no LOSS of twelve reads, at least one GAIN, after the seed-null re-call). Inputs go
to both models through the same pointwise head (R1's template), and each arm is decided against its own model, never
against the other. The MLP rules hold: no message passing, no input from another row's score, nothing normalised over
the question's rows from a score. Ranks or z-scores of first-stage retrieval values within the pool are computed before
any row is scored, as zret's z-scores are.

Screens run on today's pools, on the card. A form that passes is fitted again on the final pools (P1g-b's, if adopted)
in its own file before its full run, as G5's and R1's winners are.

| # | idea | closed by | the signal | why the close was premature | next form | file |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | learned offsets g(q, s) in the node space | P5b, P5b2, P5b3 (13 to 14 Oct): STOPS | g finds about twice H2's share of the missed golds on 2wiki and about eight times on metaqa; the gated form finds more on 4 of 6 datasets, webqsp zero-shot among them | a diagnostic (no ranking model trained), closed by "fewer on none" on musique's −0.005 to −0.011 | g's pointwise columns (logsumexp over seeds and heads of log π + ℓ, max ℓ) as a fixed input to both models; g refit per split without the held-out dataset, cross-fitted in domain | docs/P5C_OFFSET_INPUT.md |
| 2 | pool ranks and clipped pool statistics | prank, pret, ztop50 (7 to 8 Oct), relz (8 Oct): MIXED or NO_GAIN, dropped | musique zero-shot +0.139 (ztop50) and +0.202 (prank) | each REPLACED the z-scores, so metaqa in domain lost what the z-scores carry (−0.018 to −0.032); never tried beside them, never on the GNN | the rank and top-50 columns beside zret's z-scores, both models | docs/A2_RANKS_BESIDE_Z.md |
| 3 | share augmentation of the pool statistics (vshare) | full run NOT_ADOPTED under the null (8 Oct, 05:14) | three primary GAINs: metaqa zero-shot +0.0180, squad zero-shot +0.0120, musique zero-shot +0.0891 | one LOSS, metaqa in domain in L-squad's fit, −0.0094 against a floor of 0.0075 (seeds 1 and 2 moved it −0.0037 and −0.0010); never tried on zret or later | the augmentation in the lean pair's training, both models | docs/A3_VSHARE_LEAN.md |
| 4 | block dropout in training (bdrop20; padbd20 with it) | full run NOT_ADOPTED on its first LOSS (7 Oct, 15:45); three fits cancelled | musique zero-shot +0.134 on step 1's model; padbd20 PROMISING (+0.1095) with no full run | the LOSS (webqsp zero-shot −0.0081) was called at the pre-null floor 0.0075; the null's webqsp zero-shot floors are 0.0102 to 0.0295, and the run never got a complete grade to re-grade | block dropout 0.2 in the lean pair's training, both models | docs/A4_BLOCK_DROPOUT_LEAN.md |
| 5 | a head for typed-relation entries on zrc (zck) | round 28 (9 Oct): MIXED, no full run | webqsp zero-shot +0.0160 and +0.0402 | one LOSS, 2wiki −0.0082, through weights shared in joint training on batches the head never acts on; the record says a narrower screen "would be declared", and none was. L3's frozen per-kind offsets were another form | the kind head with its gradient kept off the shared weights on batches without chain entries, on zrckp and zspk | docs/A5_KIND_HEAD_BACKOFF.md |
| 6 | the words around a link (D4) | NO_SIGNAL (13 Oct): "no ranking screen follows" | fused le2 on musique +0.036 (scr-zrct) and +0.018 (scr-zrct-hp) | a diagnostic with a bar (+0.050 and 1.25 times) set for a fused score, not for a model input | per row, the link-word match to the question over its pool neighbours, as a fixed input to both models | docs/A6_LINK_WORDS_INPUT.md |
| 7 | granularity transforms (S6) | T1 (17 Oct): NOT_ADOPTED | S6's standalone scorer, webqsp zero-shot +6.43 points (significant) | T1 tried only a metaqa training augmentation (MLP 2 LOSS on webqsp zero-shot, GNN 0 and 0); the scorer's output never reached the models | S6's scorer output per row as a fixed input to both models | docs/A7_S6_SCORER_INPUT.md |
| 8 | a question-to-depth prior (qdepth) | round 11 (8 Oct): MIXED, re-call MIXED, dropped at its gate | musique zero-shot 0.270 to 0.380 on step 1's model | MIXED on step 1's base, MLP only; never on zrc or the GNN | the prior's columns as a fixed input to both models | docs/A8_QDEPTH_INPUT.md |
| 9 | edge-family dropout for message passing | the MP-unified track (early October; docs/STATUS_2026_10_05.md), carried only inside G2's form | edge-family dropout was the first one-graph MP arm with no unseen graph below the twin on R@5 | zsp, zspk and G3 to G5 train without it; G2 had it only together with its hidden-state layers (0 GAIN, 0 LOSS), so it was never tried alone on zspk's step | zspk (zspx if G5 passes) trained with edge-family dropout; GNN only, the MLP has no edges | docs/A9_EDGE_FAMILY_DROPOUT.md |
| 10 | GNN hidden-state layers per link family (G2; zg1 before it) | G2 (15 Oct): 0 GAIN, 0 LOSS; zg1 (10 Oct): NO_GAIN | filed GAINs on musique zero-shot (+0.0099) and webqsp zero-shot (+0.0210), within the floors | tried over zspk's one step; G4 later found that depth pays when the step acts on the score | G2's layers on zspx, after G5's verdict | docs/A10_G2_ON_ZSPX.md |
| 11 | our LLM-free triples as a fourth link family (P2c2) | NO_GAIN (13 Oct) on pool coverage | none on pools: −0.007 to +0.003 | a pool diagnostic; the family never reached the GNN's links | the fourth family as a link family in zspk's propagation | docs/A11_FOURTH_FAMILY_LINKS.md |
| 12 | gradient surgery (gsurg) | ADOPT on step 1 (8 Oct); zrg on zrm NOT_ADOPTED (9 Oct): 0 primary GAIN, 0 LOSS | on zrm, non-primary GAINs: webqsp zero-shot +0.0273, musique +0.0087 | tried on the MLP's line (zrm), never on the GNN | zspk trained with gsurg's loop | docs/A12_GSURG_GNN.md |

**One more, conditional.** G1a's training-free bridge rule was ABOVE on four datasets. Its quantities reached only the
MLP as inputs (round 29, zbr: NO_GAIN). R1's residual carries the bridge signal to both models now. If R1 is
NOT_ADOPTED, zbr's columns on zspk are its next form (docs/A13_BRIDGE_COLUMNS_GNN.md).

## 3. Era 6, the screens: rounds 1 to 33 (7 to 12 October)

| verdict | the form tested | read | status |
| --- | --- | --- | --- |
| step 2, eval-mix weights: NO_GAIN | fit questions weighted to each eval's mix | screens | form closed: no read gained |
| zonly: NO_GAIN | z-scores alone, raw values dropped | screen | form closed: the raw values carry information in domain |
| dnorm: NO_GAIN; step 1's pf, n, nf: NO_GAIN | normalisations of step 1's inputs | screens | form closed: no gain |
| bound1: MIXED | the score kept within one pool sd of rrf | screen | superseded by zret's pool z-scores (ADOPT), which carry the same pool-relative signal to both models |
| pad: MIXED; padbd20: PROMISING, no full run | padding the pools | screens | goes with #4 |
| bdrop20: NOT_ADOPTED | block dropout 0.2 | full run, stopped | **REOPEN #4** |
| scr-pools: MIXED | step 4c's P_F pools | screen | superseded: KIND_T, then P1g2's pools |
| zret: ADOPT (re-grade) | pool z-scores over retrieved rows | full run | in both models |
| ztop50, prank, pret: MIXED, MIXED, NO_GAIN | ranks or top-50 statistics in place of the z-scores | screens | **REOPEN #2** |
| gsurg: ADOPT (re-grade); zgs: PROMISING; zrg: NOT_ADOPTED | gradient surgery, on step 1, zret, zrm | screens and full runs | MLP: form closed (zrg, nothing on primaries). GNN: **REOPEN #12** |
| hubwalk: NO_GAIN | hub-discounted walks | screen | form closed: no gain survived the null |
| vshare: NOT_ADOPTED (re-grade) | share augmentation | full run | **REOPEN #3** |
| rel: NOT_ADOPTED; rmatch: NOT_ADOPTED; zrm: ADOPT | question-relation inputs, then a learned match, then the match on zret | full runs | superseded: the relation signal is in zrm, so in both models (zsp = zrm + one step). The rule followed |
| gcs: MIXED, re-call NO_GAIN | GraphER's retrieval propagation | screen | form closed: no gain survived; the GNN propagates its own score |
| relz: MIXED | ztop50 on rel | screen | goes with #2 |
| qdepth: MIXED | a question-to-depth prior | screen | **REOPEN #8** |
| zrs: ADOPT | rmatch's match on zret's frozen fits | full run | form closed: BASE_ZRM_ZRS chose zrm, which carries the same match |
| round 17: stopped at its identity gate | zrc's entries read on zrm's fits | full run | superseded: zrcd diagnosed it, round 18's zrc ADOPT |
| zsep: NO_GAIN (re-call) | each gold against the non-gold rows | screen | form closed: L1 tried the loss on today's pair, 0 and 0 on both |
| zkind: ADOPT (re-grade) | a head for typed relations on zrm | full run | BASE_ZRC_ZKIND chose zrc; the combination is zck |
| zck: MIXED | zkind's head on zrc | screen | **REOPEN #5** |
| zrk: NO_GAIN | smooth recall at five in the loss | screen | form closed: 0 GAIN, 0 LOSS |
| zlk: NOT_ADOPTED | a head over each row's links to zrm's top rows | full run | superseded: the GNN line went on as zsp (ADOPT) |
| zsp: ADOPT | one propagation step on zrm | full run | today's GNN line (zspk) |
| rounds 24 and 25: NO_SELECTION | a joint selection over seven feature blocks | selection | form closed: no subset gained on both models; F1 continues the input-set work |
| zgf: NO_GAIN, 3 LOSS | a graph-shaped space read without the graph | screen | form closed (DIAG_ZGF_HARM: the head harms at read). The node-space idea goes on in #1 |
| zgr: NO_GAIN, 4 LOSS | KB relation offsets from relation-name vectors | screen | form closed. Offsets go on in #1 |
| zbr (G1b): NO_GAIN | G1a's bridge quantities as MLP inputs | screen | MLP: form closed. GNN: conditional on R1 (section 2) |
| zdp: NO_GAIN | a second propagation step | screen | already reopened: G4's zspk2, ADOPT_FOR_FULL_RUN |
| zg1: NO_GAIN | a learned four-layer state over the pool's edges | screen | reopened as G2 (0 and 0), then G4 (passes); next form **#10** |
| zof: NO_GAIN, harms | a query-conditioned offset trained end to end on zrc | screen | form closed: the end-to-end form harms. The fixed-input form is #1 |
| zep, zspx (16 epochs): NOT_ADOPTED (re-grade) | twice the training steps | full runs | form closed: 0 primary GAIN on both; zrc's webqsp zero-shot direction is in the gap ledger |

## 4. 7 to 17 October: pools, links, latency and the lean pair

| verdict | the form tested | read | status |
| --- | --- | --- | --- |
| step 1: NOT_ADOPTED | D-pick against M-pick selection carves | selection | form closed: a selection protocol; the M-pick stays |
| step 4: union walk NOT_ADOPTED; 4b ADOPT; 4c mixed; 4d to 4h | pool reach rules | pool reads | superseded: KIND_T, then P1g2's rule (TARGET_MET) |
| U1c, U1f: NOT_ADOPTED | the models refit on graphs without the hyperlinks | full screens | form closed: the losses are the hyperlinks' price, the framing's measured cost, not an idea's |
| U1a to U1e | LLM-free link rules | link reads | superseded: zlink's links |
| P1a to P1f; P1c2: NOT_ADOPTED | pool rules; KIND_T's pools refit | pool reads; screens | superseded: P1c2's signal reached both models as P1c3's input (ADOPT), and P1g2's rule met the target. The rule followed |
| P1c3: ADOPT; P1c3p: ADOPT | KIND_T's walk positions as an input; zrckp's pointwise head | full runs | today's pair |
| P2a, P2b (REGIME_STAYS), P2c | link families, a family gate, LLM-free triples | link and pool reads | form closed as pool rules, superseded by P1g2; P2c's part 2 is #11 |
| P2c2: NO_GAIN | the fourth family in the pool walks | pool read | **REOPEN #11** |
| P3a, P3b, P3c | SOTA rankers, the gap ledger, Claim 1's intervals | measurements | not verdicts on ideas |
| P4a, P4b, P4c | cold pool and KT serving cost | timings | not verdicts on ideas |
| P5a: STOPS | the node space as a reader | diagnostic | carried by P5b's learned g; goes on in #1 |
| P5b, P5b2, P5b3: STOPS | learned offsets read alone, with a held-out stop, with a back-off gate | diagnostics | **REOPEN #1** |
| D1: REACHABLE_NOT_RANKED; D2: FEATURE_LIMIT | where the missed golds sit; what zrm's inputs allow | diagnostics | routed new inputs (rounds 29 to 31, G3, R1); closed nothing |
| D4: NO_SIGNAL | the words around a link | diagnostic | **REOPEN #6** |
| D5, D5b; M1 | gold reach, neighbour orders; musique's hops | diagnostics | routed P1g and G3; closed nothing |
| DIAG_DATA_SCALE: NOT_DATA_LIMITED | more training questions | diagnostic | routes away from data scale; closed nothing |
| B1, B1b, B1c, B1d; B1 re-call: STANDS | HippoRAG 2's setting, our models on it, walks, bridge slots | measurements | not verdicts on ideas; B1d routed G1a |
| G1a: NOT UNIVERSAL | a training-free bridge rule | rule read | ABOVE on four datasets; carried by zbr (MLP) and R1 (both); conditional item in section 2 |
| C1 to C12, F1 | cold latency, fastest exact forms, one input set | timings and design | not verdicts on ideas |
| L1: NOT_ADOPTED | zsep's loss on today's pair | screens | form closed: 0 GAIN, 0 LOSS on both models |
| L3: NOT_ADOPTED | per-kind offsets, frozen | screens | form closed (MLP 0 GAIN, 1 LOSS; GNN 0 and 0). The kind head is #5, the offsets #1 |
| T1: NOT_ADOPTED | S6's transforms as a training augmentation | screens | **REOPEN #7** |
| E1: NOT_ADOPTED | EM over latent relation chains, on the shared chain head, metaqa only | screens | form closed: 0 GAIN on both (GNN 1 LOSS). EM goes on through g's EM fit in #1 |
| G2: NOT_ADOPTED | hidden-state layers per link family on zspk | screen | **REOPEN #10** |
| G3, G4: ADOPT_FOR_FULL_RUN | the bridge-residual message; a second step; the step end to end | screens | going on: G5 combines them, R1 gives both models the residual |

## 5. Eras 1 to 5 (August to early October)

Every model of these eras is superseded by the lean pair, on other pools and inputs. What matters here is whether a
signal they found was dropped, not whether their models stand.

| era and verdict | the signal | status |
| --- | --- | --- |
| Era 1, the Offset-MLP screen (25 Aug): mixed | 2wiki R@1 +12.5 and MRR +18.7 points over the selected GAT, with R@5 losses | superseded: the offset idea goes on in #1 (P5c reads R@5 and hit@1) |
| QLS, M0 to M2; M2C and M2D: STOP_S4_DEVELOPMENT_CONFIRMED | S3 retained over S4 | form closed: S4's repair failed on both routes; the pipeline was replaced |
| Universal-v2: PILOT_FAILED (squad, by 0.0008); REPLICATION_PASS for the GNN family | the GNN family replicated | superseded: the models went on as gnn0 and twin0, then the screens' line. The near miss closed no idea |
| UMLP v2.1 and v2.2A: STAGE_A_FAIL | a coverage objective; a linear offset | form closed for the linear form; offsets go on in #1 |
| MP-Approx L0 to L15: BELOW_GNN | non-MP approximations of the GNN | superseded: the lean MLP (zrckp) carries Claim 1 on three of five datasets |
| the lean MLP track (lean_mlp to lean_mlp8) | in domain at the GNN on 2wiki and hotpotqa | superseded by zrc's line |
| the MP-unified track | edge-family dropout: no unseen graph below the twin on R@5 | carried only inside G2's form: **REOPEN #9** |
| the type-factorized track | per-type counts times a learned w(q, type) | superseded: zkind's head and rmatch's match carry typed relations |
| relation generalization | KB in training makes relation channels transfer; no back-off for unseen relation names | carried: metaqa trains in every split's fit; the back-off finding is the rule's own reason |
| transfer without labels, S1 to S6 | label-free shift rules fail; S6 granularity transforms transfer (webqsp zero-shot +6.43) | S1 to S5 form closed; S6: **REOPEN #7** |
| S6's two models, parts 12 to 15 | learned MP above static summaries; no MLP group drop carries | carried: the GNN line keeps MP; drops are not wanted (the user's rule of 12 October) |

## 6. What follows

- **The declarations follow section 2's order**, each in its own file, committed before its numbers, with its items
  queued behind G5's and R1's on the card and P1g-b's chain. They fill the CPU and GPU backlog.
- **Every verdict record from today on names the next form** of its idea, or says why none is left (SCREENS.md 3b).
- **Each reopened row here gets its verdict filled in when its file lands.** A reopened idea that fails as a fixed
  input to both models closes, with that file named as the close.
