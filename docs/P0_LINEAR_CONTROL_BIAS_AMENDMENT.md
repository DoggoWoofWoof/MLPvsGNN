# A3 Pre-Test Bias Amendment

Status: **frozen before implementation and before A3 test evaluation**.

Protocol v1 described an affine scorer with 19 weights and one shared bias.
Under the registered query-wise listwise loss, adding the same scalar to every
candidate score leaves both the softmax and ranking unchanged:

`softmax(s + b) = softmax(s)`.

The bias is therefore unidentifiable, receives no useful ranking gradient, and
must not be presented as effective capacity. A3 uses the bias-free form
`score(q,c) = w^T x(q,c)`, with exactly **19 trainable parameters**. No feature,
selection rule, optimizer setting, split, or evaluation rule changed. Protocol
v1 remains tagged as an immutable audit artifact; v2 is the operative protocol.
