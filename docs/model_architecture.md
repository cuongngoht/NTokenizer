# Model Architecture

*[Bản tiếng Việt](model_architecture.vi.md)*

A deep dive into the decoder-only Transformer implemented in
[`src/ntokenizer/model.py`](../src/ntokenizer/model.py). This document explains **what** each
component does and **why** it was chosen. For hyperparameter defaults and CLI
usage, see the [DEVELOPMENT.md](DEVELOPMENT.md#step-6--model-architecture).

---

## Table of contents

1. [Overview](#overview)
2. [Configuration (`GPTConfig`)](#configuration-gptconfig)
3. [Token embedding & weight tying](#token-embedding--weight-tying)
4. [RMSNorm](#rmsnorm)
5. [Rotary Positional Embeddings (RoPE)](#rotary-positional-embeddings-rope)
6. [Grouped Query Attention (GQA)](#grouped-query-attention-gqa)
7. [Causal self-attention](#causal-self-attention)
8. [KV Cache](#kv-cache)
9. [SwiGLU MLP](#swiglu-mlp)
10. [Transformer Block](#transformer-block)
11. [The full `GPT` module](#the-full-gpt-module)
12. [Generation (`generate`)](#generation-generate)
13. [Parameter count & sanity check](#parameter-count--sanity-check)
14. [GPT-2 vs this v2 — summary](#gpt-2-vs-this-v2--summary)

---

## Overview

The model is a **decoder-only Transformer** — the same family as GPT-2, LLaMA,
and Mistral — implemented in ~320 lines of plain PyTorch with no external
dependencies beyond `torch` itself. It takes a batch of token IDs and predicts
the next token at every position.

```
input_ids  [B, T]
    │
    └─ Token embedding  wte  [vocab_size, C]  →  [B, T, C]
         │                 (no separate position table — RoPE handles it)
         ├─ Block 0:
         │    pre-RMSNorm → GQA Attention (RoPE) → residual
         │    pre-RMSNorm → SwiGLU MLP           → residual
         ├─ Block 1 … N-1: (same)
         │
         └─ Final RMSNorm  →  [B, T, C]
              │
              └─ LM head  Linear(C, vocab_size, bias=False)  →  [B, T, vocab_size]
                          (weight-tied to wte)
```

`B` = batch size · `T` = sequence length · `C` = `n_embd`

Compared to vanilla GPT-2, four components have been swapped for their modern
equivalents (RoPE, RMSNorm, SwiGLU, GQA), and generation adds a KV cache plus
richer sampling (top-k, top-p, repetition penalty). Each is covered in its own
section below.

---

## Configuration (`GPTConfig`)

All hyperparameters live in one dataclass, `GPTConfig`, so a model is fully
described by a single object:

| Field | Meaning |
|---|---|
| `vocab_size` | Number of distinct tokens the model can emit/consume — must match the tokenizer. |
| `block_size` | Maximum context length in tokens. Bounds both training sequences and the RoPE frequency table. |
| `n_layer` | Number of stacked Transformer blocks. |
| `n_head` | Number of query attention heads. |
| `n_kv_head` | Number of key/value heads (`n_head` must be divisible by it — see [GQA](#grouped-query-attention-gqa)). |
| `n_embd` | Model width `C` — the size of every token's hidden vector. |
| `dropout` | Dropout probability applied in attention and MLP; set to `0` at inference. |
| `rope_theta` | Base frequency for RoPE's rotation angles. |

Everything downstream — head dimension, MLP hidden size, KV cache shape — is
derived from these few numbers.

---

## Token embedding & weight tying

The input token IDs are looked up in a single embedding table `wte` of shape
`[vocab_size, n_embd]`. Unlike vanilla GPT-2, **there is no separate learned
position table (`wpe`)** — position information is injected later, inside
attention, via RoPE.

The output projection (`lm_head`, `[n_embd, vocab_size]`) shares its weight
matrix with `wte`:

```python
self.transformer.wte.weight = self.lm_head.weight
```

This is **weight tying**. Intuitively, the same matrix that maps a token ID to
a vector also maps a hidden vector back to token-ID scores — both are doing
"translate between token space and embedding space," just in opposite
directions. Tying them halves the parameter count spent on embeddings and acts
as a mild regularizer.

---

## RMSNorm

Every sub-layer's input is normalized first (see [Transformer
Block](#transformer-block)). Instead of `LayerNorm`, the model uses **RMSNorm**:

```
RMSNorm(x) = x / RMS(x) * weight        RMS(x) = sqrt(mean(x²) + ε)
```

The difference from LayerNorm: no mean subtraction, and no bias term — only a
per-channel learned scale (`weight`). Concretely, in `RMSNorm.forward`, the
computation is done in `float32` for numerical stability and cast back to the
input dtype afterward.

**Why:** dropping the mean-centering step removes one reduction and a bias
parameter per normalization layer, at no measurable quality cost at model
scales up to billions of parameters. It's simpler and marginally faster, which
is why it's used in LLaMA, Mistral, Falcon, and most modern open LLMs.

---

## Rotary Positional Embeddings (RoPE)

### The problem it solves

Vanilla GPT-2 adds a **learned positional embedding** `wpe[t]` to the token
embedding at every position `t`. That table has one fixed vector per position
up to `block_size` — it has no way to represent a position it never saw during
training, so the model cannot generalize to longer sequences.

### How RoPE works

RoPE instead **rotates** the query and key vectors by an angle proportional to
their position, before the attention dot-product:

```
q_rotated = q * e^{i * t * θ}     (complex multiplication, per head-dim pair)
```

Each pair of dimensions in a head is treated as one complex number; rotating
Q and K by their respective positions means the dot product `q·k` naturally
depends on the *relative* distance `t_q - t_k`, not their absolute positions.

In code, `precompute_freqs_cis` builds a `[max_seq_len, head_dim // 2]` table
of unit complex numbers once (stored as a non-learnable buffer), and
`apply_rotary_emb` applies it to Q and K on every forward pass:

```
q, k  [B, H, T, head_dim]
    │  view last dim as (head_dim/2) complex pairs
    ▼
q * freqs_cis[t]   ← rotates each pair by angle t·θ_k
    │
    ▼
q_rotated, k_rotated  (same shape, same dtype)
```

Because Q and K share `head_dim`, the same `freqs_cis` slice applies to both,
even though GQA gives them different head *counts*.

**Why:** no extra learnable parameters, attention becomes naturally
relative-position-aware, and the model generalizes better to context lengths
beyond what it was trained on. This is the positional scheme behind LLaMA,
Mistral, Qwen, and DeepSeek.

---

## Grouped Query Attention (GQA)

### The problem it solves

In standard Multi-Head Attention (MHA), there are as many key/value heads as
query heads. During generation, the KV cache must store one K and one V vector
per layer, per head, per past token — memory that grows as
`O(T × n_head × head_dim)`. With many heads, this cache becomes the dominant
memory cost at long context lengths.

### How GQA works

GQA decouples the head counts: `n_head` query heads share `n_kv_head` key/value
heads, in groups of size `n_rep = n_head // n_kv_head`.

```
n_head = 8 (Q heads)     n_kv_head = 2 (KV heads)
                                      → 4 Q heads share 1 KV pair

Q heads:  [Q0 Q1 Q2 Q3] [Q4 Q5 Q6 Q7]
                │               │
KV heads: [ K0/V0 ]     [ K1/V1 ]
```

In `CausalSelfAttention`, the K/V projections output only `n_kv_head *
head_dim` channels (smaller than Q's `n_head * head_dim`), and before the
attention dot-product each KV head is repeated `n_rep` times
(`repeat_interleave`) to line up with its group of Q heads.

Two special cases fall out of the same code path:
- `n_kv_head == n_head` → standard MHA (`n_rep == 1`, no repeating needed).
- `n_kv_head == 1` → Multi-Query Attention (MQA), the most aggressive setting.

**Why:** the KV cache shrinks by exactly `n_head / n_kv_head`×, with
negligible quality loss once `n_kv_head ≥ 2`. This is standard in LLaMA 2/3,
Mistral, and Gemma.

---

## Causal self-attention

`CausalSelfAttention` ties RoPE, GQA, and the KV cache together:

1. Project `x` to Q, K, V with separate bias-free `Linear` layers (Q sized for
   `n_head`, K/V sized for the smaller `n_kv_head`).
2. Apply RoPE to the newly-computed Q and K (not V — rotation only encodes
   position for the attention score, not the values being aggregated).
3. If a KV cache prefix exists, prepend it to the new K/V (see [KV
   Cache](#kv-cache)).
4. Repeat KV heads `n_rep`× to match Q's head count (GQA expansion).
5. Compute scaled dot-product attention with a **causal mask** — each query
   position may only attend to key positions at or before it.
6. Concatenate heads and project back to `n_embd` with `o_proj`.

The causal mask needs care when a KV cache is in play: a single new query at
absolute position `T_k - 1` must still be allowed to see the *entire* cached
prefix, not just itself. The code handles this by offsetting the triangular
mask by `T_k - T` (or, when PyTorch's fused
`scaled_dot_product_attention` is available, by only passing `is_causal=True`
when there's no cache prefix — otherwise no mask is needed at all, since one
query attending to all past keys is exactly what "causal" means in that case).

This whole block sits inside a **pre-norm residual** connection (see
[Transformer Block](#transformer-block)): `x = x + Attention(RMSNorm(x))`.

---

## KV Cache

### The problem it solves

Naively, generating token `N+1` requires re-running the full forward pass over
all `T` tokens seen so far — recomputing attention is `O(T²)` per new token,
so generating `N` tokens costs `O(N × T²)` overall.

### How it works

Keys and values are deterministic functions of past tokens alone (they don't
depend on future tokens due to causality), so they can be **computed once and
reused**. `CausalSelfAttention.forward` accepts an optional `past_kv =
(past_K, past_V)`, concatenates it with the newly computed K/V, and returns the
extended cache for the caller to store:

```
Step 0 (prefill):  forward the whole prompt   → cache K,V for every position
Step 1:            forward only the new token → append its K,V to the cache
Step 2:            forward only the new token → append again
   ...
```

Each generation step after the prefill now costs `O(T)` instead of `O(T²)`,
since only one new token's Q/K/V needs computing while the rest are read from
cache. `GPT.generate` manages this across layers (a list of `(K, V)` per
layer) and evicts the cache — recomputing from the most recent `block_size -
1` tokens — if the accumulated length would exceed `block_size`.

**Why:** for a 256-token context generating 200 new tokens, this gives roughly
a 50× reduction in attention FLOPs, since the model repeatedly avoids
recomputing the O(T²) attention over the growing prefix at every step.

---

## SwiGLU MLP

The feed-forward sub-layer is a **gated** MLP instead of the classic GELU MLP.

Vanilla GPT-2:
```
MLP(x) = W₂ · GELU(W₁ · x)          hidden_dim = 4 × n_embd
```

SwiGLU:
```
SwiGLU(x) = W_down · ( SiLU(W_gate · x) ⊙ W_up · x )     hidden_dim ≈ 8/3 × n_embd
```

`W_gate` and `W_up` both project `x` up to the hidden dimension; the gate
branch is passed through SiLU and multiplied element-wise (⊙) against the up
branch before being projected back down. The `SiLU(W_gate · x)` term acts as a
learned, data-dependent filter over the up-projected features — some channels
get passed through, others get suppressed.

The hidden dimension is deliberately set to `8/3 × n_embd` (rounded up to a
multiple of 64 for hardware efficiency) rather than `4×`, because the gated
version has three weight matrices instead of two — this keeps the total
parameter count comparable to the plain GELU MLP while adding the gate.

**Why:** empirically, gating consistently produces lower loss than a plain
GELU MLP at the same parameter budget. Used in LLaMA, PaLM, and Gemma.

---

## Transformer Block

Each `Block` is two residual sub-layers, both using **pre-normalization**
(normalize *before* the sub-layer, not after):

```
x = x + Attention( RMSNorm(x) )
x = x + MLP( RMSNorm(x) )
```

Pre-norm keeps gradients flowing cleanly through the residual stream across
many stacked blocks — it's the standard choice in nearly every modern
Transformer, as opposed to GPT-1-style post-norm.

`Block.forward` returns both the updated hidden state and the new `(K, V)`
pair for that layer, so `GPT.forward` can thread the KV cache through all
layers uniformly.

---

## The full `GPT` module

`GPT` stitches everything together:

- `transformer.wte` — token embedding (weight-tied to `lm_head`, see
  [above](#token-embedding--weight-tying)).
- `transformer.h` — a `ModuleList` of `n_layer` `Block`s.
- `transformer.ln_f` — one final RMSNorm before the output projection.
- `freqs_cis` — the RoPE table, precomputed once for `block_size` positions
  and registered as a non-learnable buffer (so it moves with `.to(device)`
  but isn't touched by the optimizer).

**Initialization** follows GPT-2's scheme: all `Linear`/`Embedding` weights
are drawn from `Normal(0, 0.02)`. On top of that, every projection that feeds
directly back into the residual stream (`o_proj` in attention, `down` in the
MLP) is re-initialized with a smaller std, `0.02 / sqrt(2 * n_layer)` — this
keeps the variance of the residual stream from growing unboundedly as more
blocks are stacked, which matters more as `n_layer` increases.

`forward(input_ids, targets=None, past_kvs=None)` returns a 3-tuple
`(logits, loss, new_past_kvs)`:
- With `targets` provided (training), it computes logits for **every**
  position and cross-entropy loss against the targets.
- Without `targets` (inference), it only projects the **last** position to
  logits (`x[:, [-1], :]`) — there's no reason to compute logits for tokens
  that aren't being sampled from, so this saves a large matmul over the
  `vocab_size` dimension.

---

## Generation (`generate`)

`GPT.generate` is an autoregressive sampling loop built around the KV cache:

```
prefill:      forward the full prompt (cropped to block_size) → cache K,V
for each of max_new_tokens:
    forward only the latest token, using the cache        (O(T) per step)
    take logits for the last position
    repetition_penalty → temperature → top_k → top_p → softmax → sample
    append the sampled token, extend the cache
```

Sampling controls are applied in this order, and can be combined freely:

1. **Repetition penalty** — logits for tokens already present in the
   generated sequence are pulled toward zero (positive logits divided,
   negative logits multiplied by the penalty), discouraging loops.
2. **Temperature** — logits divided by `T` before softmax; `<1` sharpens the
   distribution (more deterministic), `>1` flattens it (more random).
3. **Top-k** — logits outside the `k` highest values are masked to `-inf`.
4. **Top-p (nucleus)** — after sorting by probability, keep the smallest
   prefix of tokens whose cumulative probability reaches `p`; mask the rest.
5. A final `softmax` + `torch.multinomial` draws the next token.

If the cache would grow past `block_size`, `generate` drops it and recomputes
from the most recent `block_size - 1` tokens — a simple sliding-window
strategy rather than a more elaborate cache-compaction scheme.

---

## Parameter count & sanity check

This snippet exercises the whole architecture end-to-end — paste it into a
`python` shell (with the package installed via `pip install -e .`) to try it
yourself:

```python
import math
import torch
from ntokenizer.config import GPTConfig
from ntokenizer.model import GPT

config = GPTConfig(vocab_size=32000, block_size=512, n_layer=8, n_head=8, n_kv_head=2, n_embd=512, dropout=0.0)
model = GPT(config)
print(f"Parameters : {model.count_parameters():,}")

B, T = 2, 64
ids = torch.randint(0, config.vocab_size, (B, T))
targets = torch.randint(0, config.vocab_size, (B, T))

logits, loss, kvs = model(ids, targets)
print(f"logits : {list(logits.shape)}")
print(f"loss   : {loss.item():.4f}  (expected ≈ {math.log(config.vocab_size):.4f} = ln(vocab_size))")
print(f"KV cache layers : {len(kvs)}  shapes : K{list(kvs[0][0].shape)} V{list(kvs[0][1].shape)}")

model.eval()
seed = torch.zeros((1, 1), dtype=torch.long)
out = model.generate(seed, max_new_tokens=30, temperature=0.8, top_k=50, top_p=0.9)
print(f"generated token IDs : {out[0].tolist()}")
```

This builds a small `GPTConfig`, reports the parameter count
(`model.count_parameters()`), runs one training-mode forward pass, and checks
that the loss is close to `ln(vocab_size)` — the expected cross-entropy loss
of an untrained model guessing uniformly over the vocabulary. It then runs
`generate()` for a few tokens and prints the resulting KV cache shapes,
confirming the whole prefill → incremental-decode path works. The equivalent
assertions are covered automatically in `tests/test_model.py` and
`tests/test_generation.py`.

---

## GPT-2 vs this v2 — summary

| Component | GPT-2 (original) | This implementation | Benefit |
|---|---|---|---|
| Positional encoding | Learned `wpe` table | **RoPE** | Generalizes beyond training length; no extra parameters |
| Normalization | `LayerNorm` (with bias) | **RMSNorm** (no bias) | Simpler, slightly faster, equally stable |
| MLP activation | `GELU` with 4× expansion | **SwiGLU** with 8/3× expansion | Better gradient flow; consistently lower loss |
| Attention heads | Multi-Head Attention | **Grouped Query Attention** | Fewer KV heads → smaller KV cache, faster inference |
| Generation | Recomputes full context each step | **KV Cache** | O(T) per step instead of O(T²) |
| Sampling | Top-k only | **Top-k + Top-p + Repetition penalty** | More natural, less repetitive output |

For the corresponding CLI flags, default hyperparameters, and training
pipeline, see the [DEVELOPMENT.md](DEVELOPMENT.md#step-6--model-architecture).
