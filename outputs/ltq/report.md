# LTQ: a question-gated next-hop scorer with relations and IRA's refinement (declared in docs/LTQ_QUESTION_GATED_NEXT_HOP.md)

**NO_SIGNAL (BELOW_PAIR)**, decision form u. The rule reads WQ-hotpotqa musique/s1fit and musique/s1eval: the decision form's unlinked T2 at least pair's + 0.03 and its linked T2 at least bil's - 0.02 on both.

## Unlinked targets' T2 (every row outside rrf's top five)

| split | carve | targets | linked share | bil | ltq_u | ltq_h | pair | res | rrf | LTA's record |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| WQ-hotpotqa | musique/s1fit | 3798 | 0.332 | 0.096 | 0.089 | 0.043 | 0.144 | 0.128 | 0.079 | equal |
| WQ-hotpotqa | musique/s1eval | 2547 | 0.362 | 0.084 | 0.086 | 0.054 | 0.113 | 0.104 | 0.070 | equal |
| WQ-hotpotqa | 2wiki/fit | 5661 | 0.612 | 0.191 | 0.194 | 0.123 | 0.155 | 0.170 | 0.096 | equal |
| WQ-hotpotqa | hotpotqa/s1eval | 3632 | 0.771 | 0.116 | 0.126 | 0.096 | 0.269 | 0.236 | 0.144 | equal |

## Linked targets' T2

| split | carve | bil | ltq_u | ltq_h | pair | res | rrf |
|---|---|---:|---:|---:|---:|---:|---:|
| WQ-hotpotqa | musique/s1fit | 0.331 | 0.342 | 0.285 | 0.042 | 0.029 | 0.040 |
| WQ-hotpotqa | musique/s1eval | 0.346 | 0.343 | 0.300 | 0.048 | 0.025 | 0.062 |
| WQ-hotpotqa | 2wiki/fit | 0.827 | 0.805 | 0.812 | 0.001 | 0.001 | 0.014 |
| WQ-hotpotqa | hotpotqa/s1eval | 0.579 | 0.582 | 0.481 | 0.045 | 0.027 | 0.081 |

## What each form's fit uses (decides nothing)

| split | carve | form | gate on A | gate on A, lists with an unlinked target | top relation's share | relation entropy |
|---|---|---|---:|---:|---:|---:|
| WQ-hotpotqa | musique/s1fit | ltq_u | 0.701 | 0.701 | 0.342 | 2.642 |
| WQ-hotpotqa | musique/s1fit | ltq_h | 0.611 | 0.605 | 0.385 | 2.116 |
| WQ-hotpotqa | musique/s1eval | ltq_u | 0.697 | 0.698 | 0.332 | 2.635 |
| WQ-hotpotqa | musique/s1eval | ltq_h | 0.606 | 0.596 | 0.378 | 2.103 |
| WQ-hotpotqa | 2wiki/fit | ltq_u | 0.708 | 0.708 | 0.370 | 2.634 |
| WQ-hotpotqa | 2wiki/fit | ltq_h | 0.669 | 0.659 | 0.423 | 2.086 |
| WQ-hotpotqa | hotpotqa/s1eval | ltq_u | 0.689 | 0.687 | 0.243 | 2.635 |
| WQ-hotpotqa | hotpotqa/s1eval | ltq_h | 0.617 | 0.612 | 0.336 | 2.145 |

## Fold fits' stop T2

| split | form | folds |
|---|---|---|
| WQ-hotpotqa | u | [{"musique": 0.7023255813953488, "2wiki": 0.7090069284064665}, {"musique": 0.43558282208588955, "2wiki": 0.6893617021276596}, {"musique": 0.6735751295336787, "2wiki": 0.6909975669099757}, {"musique": 0.6381578947368421, "2wiki": 0.6877729257641921}, {"musique": 0.5461538461538461, "2wiki": 0.6701208981001727}] |
| WQ-hotpotqa | h | [{"musique": 0.3581395348837209, "2wiki": 0.7182448036951501}, {"musique": 0.2167689161554192, "2wiki": 0.7042553191489361}, {"musique": 0.36787564766839376, "2wiki": 0.7274939172749392}, {"musique": 0.42105263157894735, "2wiki": 0.6899563318777293}, {"musique": 0.34615384615384615, "2wiki": 0.6735751295336787}] |
| WQ-musique | u | [{"hotpotqa": 0.7652173913043478, "2wiki": 0.7297921478060047}, {"hotpotqa": 0.7647058823529411, "2wiki": 0.725531914893617}, {"hotpotqa": 0.7847533632286996, "2wiki": 0.732360097323601}, {"hotpotqa": 0.7763157894736842, "2wiki": 0.7161572052401747}, {"hotpotqa": 0.7244444444444444, "2wiki": 0.7063903281519862}] |
