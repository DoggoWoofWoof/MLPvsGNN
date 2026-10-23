# U2h's full run: NOT_ADOPTED

The lean pair with KT_W (zrckpw, zspkw) against (zrckp, zspk) on U2d's base build, six splits per arm; the new splits on T4, L-musique and L-hotpotqa as filed on L40S (docs/U2H_FULL_RUN.md).

- Primary GAINs: zspkw L-metaqa metaqa
- LOSSes: zspkw L-squad musique, zspkw L-squad 2wiki
- Each arm's own verdict (beside): zrckpw NOT_ADOPTED, zspkw NOT_ADOPTED
- Next: a per-question gate on the KT_W head, its weight per question from inputs computed before any row is scored, in its own file (SCREENS 3b's back-off form)
- Always: the hyperlink-gap plan (link precision; a hyperlink scorer learned on 2wiki and applied unchanged elsewhere; corpus aliases), each step in its own declared file, beside this run

Device noise (zrckp J5 seed 0, L4 minus T4): metaqa -0.0013, squad +0.0002, musique +0.0046, hotpotqa +0.0008, 2wiki -0.0006, webqsp -0.0140
