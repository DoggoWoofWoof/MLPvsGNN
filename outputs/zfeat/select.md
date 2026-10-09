# Rounds twenty-four and twenty-five: the common selection, NO_SELECTION

Chosen for both models: none. Mean R@5 gain over every candidate masked: zfs +0.0000 (lowest read +0.0000), zgn +0.0000 (lowest +0.0000); ten select reads per model, two fits.

Each model alone would choose: zfs depth_FULL+seedcond+typed_rel+ordered (+0.0012), zgn none (+0.0000).

| subset | smaller mean gain | zfs mean | zfs lowest | zgn mean | zgn lowest | admissible |
| --- | --- | --- | --- | --- | --- | --- |
| topo_KNN+depth_FULL+typed_rel+ordered | +0.0008 | +0.0009 | -0.0013 | +0.0008 | -0.0020 | no |
| topo_KNN+depth_FULL+seedcond+typed_rel+ordered | +0.0007 | +0.0007 | -0.0018 | +0.0008 | -0.0010 | yes |
| topo_KNN+depth_FULL+ordered | +0.0006 | +0.0009 | -0.0013 | +0.0006 | -0.0020 | no |
| topo_KNN+depth_FULL+typed_rel | +0.0005 | +0.0008 | -0.0013 | +0.0005 | -0.0020 | no |
| topo_NER+topo_KNN+depth_FULL+typed_rel+ordered | +0.0005 | +0.0005 | -0.0050 | +0.0005 | -0.0027 | no |
| depth_FULL+ordered | +0.0004 | +0.0010 | -0.0020 | +0.0004 | -0.0020 | no |
| topo_KNN+depth_FULL+seedcond+ordered | +0.0004 | +0.0005 | -0.0012 | +0.0004 | -0.0017 | yes |
| topo_KNN+depth_FULL | +0.0004 | +0.0008 | -0.0013 | +0.0004 | -0.0020 | no |
| depth_FULL+typed_rel+ordered | +0.0004 | +0.0013 | -0.0020 | +0.0004 | -0.0020 | no |
| depth_FULL+typed_rel | +0.0004 | +0.0009 | -0.0013 | +0.0004 | -0.0020 | no |
| topo_KNN+depth_FULL+seedcond+typed_rel | +0.0003 | +0.0005 | -0.0013 | +0.0003 | -0.0013 | yes |
| depth_FULL | +0.0003 | +0.0005 | -0.0013 | +0.0003 | -0.0027 | no |
| ordered | +0.0003 | +0.0003 | -0.0011 | +0.0004 | -0.0010 | yes |
| typed_rel | +0.0002 | +0.0003 | -0.0007 | +0.0002 | -0.0012 | yes |
| typed_rel+ordered | +0.0002 | +0.0004 | -0.0026 | +0.0002 | -0.0017 | no |

Each block alone:

| block | zfs mean | zfs lowest | zgn mean | zgn lowest |
| --- | --- | --- | --- | --- |
| topo_NER | -0.0013 | -0.0046 | -0.0001 | -0.0017 |
| topo_KNN | -0.0000 | -0.0023 | -0.0008 | -0.0033 |
| topo_FULL | -0.0009 | -0.0018 | -0.0007 | -0.0040 |
| depth_FULL | +0.0005 | -0.0013 | +0.0003 | -0.0027 |
| seedcond | +0.0004 | -0.0008 | -0.0009 | -0.0032 |
| typed_rel | +0.0003 | -0.0007 | +0.0002 | -0.0012 |
| ordered | +0.0003 | -0.0011 | +0.0004 | -0.0010 |

| model | fit | read | zero-shot | R@5 none | R@5 all |
| --- | --- | --- | --- | --- | --- |
| zfs | scr-zfs | metaqa=select |  | 0.7838 | 0.7869 |
| zfs | scr-zfs | squad=select |  | 0.9192 | 0.9186 |
| zfs | scr-zfs | musique=select | yes | 0.5912 | 0.5902 |
| zfs | scr-zfs | hotpotqa=select |  | 0.9108 | 0.9105 |
| zfs | scr-zfs | 2wiki=select |  | 0.8929 | 0.8909 |
| zfs | scr-zfs-hp | metaqa=select |  | 0.7827 | 0.7870 |
| zfs | scr-zfs-hp | squad=select |  | 0.9246 | 0.9239 |
| zfs | scr-zfs-hp | musique=select |  | 0.6502 | 0.6528 |
| zfs | scr-zfs-hp | hotpotqa=select | yes | 0.8607 | 0.8521 |
| zfs | scr-zfs-hp | 2wiki=select |  | 0.8847 | 0.8844 |
| zgn | scr-zgn | metaqa=select |  | 0.7869 | 0.7884 |
| zgn | scr-zgn | squad=select |  | 0.9212 | 0.9219 |
| zgn | scr-zgn | musique=select | yes | 0.5971 | 0.6012 |
| zgn | scr-zgn | hotpotqa=select |  | 0.9158 | 0.9168 |
| zgn | scr-zgn | 2wiki=select |  | 0.9056 | 0.9074 |
| zgn | scr-zgn-hp | metaqa=select |  | 0.7876 | 0.7889 |
| zgn | scr-zgn-hp | squad=select |  | 0.9226 | 0.9199 |
| zgn | scr-zgn-hp | musique=select |  | 0.6479 | 0.6443 |
| zgn | scr-zgn-hp | hotpotqa=select | yes | 0.8528 | 0.8455 |
| zgn | scr-zgn-hp | 2wiki=select |  | 0.9046 | 0.9034 |
