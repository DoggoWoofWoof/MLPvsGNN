# U2k report (docs/U2K_LEARNED_LINK_SCORER.md)

Outcome: **ADOPT_FOR_POOLS**, best K1

Score (mean I's ALL, musique, hotpotqa, 2wiki): R0 0.8368, K1 0.8499, K2 0.8312, K3 0.8325

Scorer A (fit on 2wiki): weights lead +0.087, pos +0.184, count +0.125, indeg -0.784, outdeg -0.260, caps +0.189, capmid +0.370, tokens +0.820, mutual +0.461; tau -0.644; in-domain {'auc': 0.8397, 'ap': 0.6441, 'precision_at_tau': 0.6065, 'recall_at_tau': 0.6065}

Scorer H (fit on hotpotqa): weights lead +0.217, pos +0.424, count +0.151, indeg -0.777, outdeg -0.193, caps +0.311, capmid +0.490, tokens +0.784, mutual +0.404; tau -0.403; in-domain {'auc': 0.8287, 'ap': 0.6997, 'precision_at_tau': 0.639, 'recall_at_tau': 0.639}

## musique (scorer A, 1534 questions)

Kept 479,104 of 2,304,416 directed links at tau.

| form | I's ALL | bridge@5 | bridge@25 | all of X: ALL | rows | choice | gap share (I, choice, bridge@25) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 0.7751 | 0.1558 | 0.1991 | 0.7542 | 5588.3 | 0.6676 | None, None, None |
| K1 | 0.7816 | 0.1926 | 0.2251 | 0.7542 | 5588.3 | 0.6676 | None, None, None |
| K2 | 0.7647 | 0.1732 | 0.2013 | 0.6695 | 1585.0 | 0.6408 | None, None, None |
| K3 | 0.766 | 0.1786 | 0.2013 | 0.6695 | 1585.0 | 0.6408 | None, None, None |
| PKG | 0.7679 | 0.145 | 0.1926 | 0.7562 | 5865.9 | 0.6683 | None, None, None |

Beside (musique, 0.9991 of W2 links hyperlinked): A {'fit_on': '2wiki', 'in_domain': False, 'auc': 0.1514, 'ap': 0.9976, 'kept_share': 0.2079, 'precision': 0.9967, 'recall': 0.2074}; H {'fit_on': 'hotpotqa', 'in_domain': False, 'auc': 0.1981, 'ap': 0.9978, 'kept_share': 0.3127, 'precision': 0.9976, 'recall': 0.3122}

## squad (scorer A, 1498 questions)

Kept 184,374 of 685,845 directed links at tau.

| form | I's ALL | bridge@5 | bridge@25 | all of X: ALL | rows | choice | gap share (I, choice, bridge@25) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | None | 0.0 | 0.0 | 0.9887 | 1673.9 | 0.9883 | None, None, None |
| K1 | None | 0.0 | 0.0 | 0.9887 | 1673.9 | 0.9883 | None, None, None |
| K2 | None | 0.0 | 0.0 | 0.9853 | 1082.2 | 0.9853 | None, None, None |
| K3 | None | 0.0 | 0.0 | 0.9853 | 1082.2 | 0.9853 | None, None, None |
| PKG | None | 0.0 | 0.0 | 0.9866 | 1429.1 | 0.9861 | None, None, None |

Beside (squad, 0.726 of W2 links hyperlinked): A {'fit_on': '2wiki', 'in_domain': False, 'auc': 0.094, 'ap': 0.5298, 'kept_share': 0.2688, 'precision': 0.2919, 'recall': 0.1081}; H {'fit_on': 'hotpotqa', 'in_domain': False, 'auc': 0.1052, 'ap': 0.5327, 'kept_share': 0.3531, 'precision': 0.3832, 'recall': 0.1864}

## hotpotqa (scorer A, 1508 questions)

Kept 7,561,948 of 32,334,355 directed links at tau.

| form | I's ALL | bridge@5 | bridge@25 | all of X: ALL | rows | choice | gap share (I, choice, bridge@25) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 0.8985 | 0.3283 | 0.6698 | 0.9211 | 792.0 | 0.9094 | None, None, None |
| K1 | 0.9012 | 0.5579 | 0.6774 | 0.9211 | 792.0 | 0.9094 | 0.039, 0.0, 0.033 |
| K2 | 0.8654 | 0.5104 | 0.5655 | 0.8846 | 299.8 | 0.8793 | -0.475, -0.479, -0.451 |
| K3 | 0.868 | 0.5104 | 0.5712 | 0.8846 | 299.8 | 0.8793 | -0.438, -0.479, -0.426 |
| PKG | 0.9682 | 0.7116 | 0.9013 | 0.9887 | 589.6 | 0.9723 | None, None, None |

Beside (hotpotqa, 0.3236 of W2 links hyperlinked): A {'fit_on': '2wiki', 'in_domain': False, 'auc': 0.8247, 'ap': 0.6959, 'kept_share': 0.2339, 'precision': 0.7287, 'recall': 0.5266}; H {'fit_on': 'hotpotqa', 'in_domain': True, 'auc': 0.8287, 'ap': 0.6997, 'kept_share': 0.3236, 'precision': 0.639, 'recall': 0.639}

## 2wiki (scorer H, 1496 questions)

Kept 23,033,838 of 71,913,824 directed links at tau.

| form | I's ALL | bridge@5 | bridge@25 | all of X: ALL | rows | choice | gap share (I, choice, bridge@25) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 0.8369 | 0.3181 | 0.8133 | 0.9011 | 941.6 | 0.8191 | None, None, None |
| K1 | 0.867 | 0.7003 | 0.8566 | 0.9011 | 941.6 | 0.8191 | 0.506, 0.0, 0.569 |
| K2 | 0.8636 | 0.6635 | 0.8526 | 0.8971 | 491.2 | 0.8297 | 0.449, 0.371, 0.516 |
| K3 | 0.8636 | 0.6995 | 0.8534 | 0.8971 | 491.2 | 0.8297 | 0.449, 0.371, 0.527 |
| PKG | 0.8964 | 0.6987 | 0.8894 | 0.9305 | 510.0 | 0.8477 | None, None, None |

Beside (2wiki, 0.2305 of W2 links hyperlinked): A {'fit_on': '2wiki', 'in_domain': True, 'auc': 0.8397, 'ap': 0.6441, 'kept_share': 0.2305, 'precision': 0.6065, 'recall': 0.6065}; H {'fit_on': 'hotpotqa', 'in_domain': False, 'auc': 0.8323, 'ap': 0.635, 'kept_share': 0.3203, 'precision': 0.5015, 'recall': 0.6968}
