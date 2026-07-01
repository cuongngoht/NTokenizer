"""
Vietnamese GPT — improved decoder-only Transformer.

Improvements over the original:
  • RMSNorm   — replaces LayerNorm (no bias, slightly faster, equally stable)
  • RoPE      — rotary positional embeddings replace the learned wpe table;
                generalises beyond the training context length
  • SwiGLU    — gated MLP (silu(gate) ⊙ up → down) replaces GELU MLP;
                converges faster and often reaches a lower loss
  • GQA       — Grouped Query Attention: fewer KV heads than Q heads;
                same quality, lower memory, faster inference
  • KV Cache  — generate() stores past K/V; each new token costs O(T)
                instead of O(T²)
  • Top-p     — nucleus sampling alongside existing top-k
  • Rep. pen. — repetition penalty to reduce looping output

Architecture:
  input_ids [B, T]
    → token embedding [B, T, C]          (wte — no separate wpe)
    → N × Transformer Block
         pre-RMSNorm → GQA-Attention (RoPE) → residual
         pre-RMSNorm → SwiGLU MLP          → residual
    → final RMSNorm [B, T, C]
    → LM head (weight-tied) [B, T, vocab_size]

Notation:
  B   = batch size
  T   = query sequence length (≤ block_size)
  C   = n_embd  (model width)
  H   = n_head  (query heads)
  Hkv = n_kv_head (KV heads; H % Hkv == 0)
  h   = C // H  (head dimension, shared by Q/K/V)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ntokenizer.config import GPTConfig


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler than LayerNorm: no mean subtraction, no bias.
    Computes in float32 for numerical stability then casts back.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).type_as(x)


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_freqs_cis(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    Precompute complex rotation frequencies for RoPE.

    Returns a [max_seq_len, head_dim // 2] complex64 tensor.
    Each row t holds the unit-complex numbers e^{i * m * θ_k} for that position.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)                        # [S, head_dim // 2]
    return torch.polar(torch.ones_like(freqs), freqs)    # complex64


def apply_rotary_emb(
    q: torch.Tensor,          # [B, H,   T, head_dim]
    k: torch.Tensor,          # [B, Hkv, T, head_dim]
    freqs_cis: torch.Tensor,  # [T, head_dim // 2]  complex
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to Q and K in-place (complex multiplication).

    Q and K may have different numbers of heads (GQA), but share the same
    head_dim, so the same freqs_cis applies to both.
    """
    def rotate(x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        xc = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
        out = torch.view_as_real(xc * freqs_cis[None, None]).flatten(3)
        return out.type_as(x)

    return rotate(q), rotate(k)


# ---------------------------------------------------------------------------
# Repetition penalty (used by GPT.generate)
# ---------------------------------------------------------------------------

def apply_repetition_penalty(
    logits: torch.Tensor,  # [B, vocab_size]
    idx:    torch.Tensor,  # [B, T_so_far]
    penalty: float,
) -> torch.Tensor:
    """
    Down-weight the logits of tokens already present in idx, discouraging
    the model from repeating itself. penalty > 1.0 strengthens the effect;
    1.0 is a no-op.
    """
    score = logits.gather(1, idx)   # [B, T_so_far]
    score = torch.where(
        score < 0,
        score * penalty,
        score / penalty,
    )
    logits = logits.clone()
    logits.scatter_(1, idx, score)
    return logits


# ---------------------------------------------------------------------------
# Causal Self-Attention with GQA, RoPE, and KV Cache
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with three upgrades:

    GQA — n_kv_head ≤ n_head: Q heads are split into groups that share
          one pair of KV heads. When n_kv_head == n_head this is standard MHA;
          when n_kv_head == 1 this is Multi-Query Attention (MQA).

    RoPE — position information is encoded by rotating Q and K in complex
           space. The rotation is applied after projection, so there is no
           separate positional embedding table.

    KV Cache — forward() accepts and returns past_kv = (past_K, past_V).
               During generation the caller accumulates these across steps
               so that only the new token's Q/K/V are computed each time.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        assert config.n_head % config.n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep     = config.n_head // config.n_kv_head  # GQA expansion factor
        self.head_dim  = config.n_embd // config.n_head
        self.dropout   = config.dropout

        # Separate Q/K/V projections — no bias (standard in modern LLMs)
        self.q_proj = nn.Linear(config.n_embd, config.n_head    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd,                    bias=False)

        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self._use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x:         torch.Tensor,
        freqs_cis: torch.Tensor,                                      # [T, head_dim//2]
        past_kv:   tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to the new Q and K tokens
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # KV cache: prepend cached keys/values from previous steps
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)  # caller stores this for the next step

        T_k = k.shape[2]  # total key/value sequence length

        # Expand KV heads to match Q heads for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if self._use_flash:
            dropout_p = self.dropout if self.training else 0.0
            # is_causal=True is valid only when T_q == T_k (no cache prefixed).
            # With KV cache (T=1, T_k>1) the single query attends to all keys —
            # no masking needed, so is_causal=False is correct.
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=(T == T_k),
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (q @ k.transpose(-2, -1)) * scale  # [B, H, T, T_k]
            # Causal mask: Q[i] (at absolute position T_k-T+i) may attend to
            # K[j] only when j ≤ T_k-T+i  →  tril with diagonal offset T_k-T.
            mask = torch.ones(T, T_k, device=q.device, dtype=torch.bool).tril(
                diagonal=T_k - T
            )
            scores = scores.masked_fill(~mask[None, None], float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_drop(weights)
            y = weights @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.o_proj(y)), new_kv


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    SwiGLU feed-forward network: down( SiLU(gate(x)) ⊙ up(x) )

    Uses hidden_dim = ⌈8/3 × n_embd⌉ rounded to the next multiple of 64.
    This keeps total parameters comparable to the classic 4× GELU MLP while
    benefiting from the gating mechanism's better gradient flow.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = int(config.n_embd * 8 / 3)
        hidden = ((hidden + 63) // 64) * 64  # align to 64 for efficiency

        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.up   = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    Pre-norm Transformer block using RMSNorm + GQA + SwiGLU:

      x = x + Attention(RMSNorm(x))
      x = x + MLP(RMSNorm(x))

    Returns (output, new_kv) so the caller can accumulate the KV cache.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1  = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2  = RMSNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(
        self,
        x:         torch.Tensor,
        freqs_cis: torch.Tensor,
        past_kv:   tuple | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        attn_out, new_kv = self.attn(self.ln1(x), freqs_cis, past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    Decoder-only GPT with all upgrades applied.

    forward() signature change: now returns a 3-tuple
      (logits, loss, past_kvs)
    so callers must unpack accordingly:
      logits, loss, _ = model(x, targets)
      logits, _, kvs  = model(x, past_kvs=kvs)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        head_dim = config.n_embd // config.n_head

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: token embedding and LM head share the same matrix
        self.transformer.wte.weight = self.lm_head.weight

        # RoPE frequency table — not a learnable parameter
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, config.block_size, config.rope_theta),
        )

        # GPT-2-style weight initialisation
        self.apply(self._init_weights)
        # Scale down residual path projections for stable training at depth
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,             # [B, T]
        targets:   torch.Tensor | None = None,  # [B, T]
        past_kvs:  list | None = None,       # list of (K, V) per layer
    ) -> tuple[torch.Tensor, torch.Tensor | None, list]:
        """
        Returns (logits, loss, new_past_kvs).
          - loss is None when targets is not provided (inference mode).
          - new_past_kvs is a list of (K, V) tensors, one per layer.
        """
        T = input_ids.size(1)
        start_pos = 0 if past_kvs is None else past_kvs[0][0].shape[2]

        assert start_pos + T <= self.config.block_size, (
            f"Total context {start_pos + T} exceeds block_size {self.config.block_size}. "
            "Truncate the input or reset the KV cache."
        )

        x = self.transformer.drop(self.transformer.wte(input_ids))
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]

        new_kvs: list = []
        for i, block in enumerate(self.transformer.h):
            past_kv = None if past_kvs is None else past_kvs[i]
            x, new_kv = block(x, freqs_cis, past_kv)
            new_kvs.append(new_kv)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)                         # [B, T, vocab_size]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        else:
            logits = self.lm_head(x[:, [-1], :])             # [B, 1, vocab_size]
            loss = None

        return logits, loss, new_kvs

    @torch.no_grad()
    def generate(
        self,
        idx:                torch.Tensor,        # [B, T_prompt]
        max_new_tokens:     int,
        temperature:        float = 1.0,
        top_k:              int | None = None,
        top_p:              float | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Auto-regressive token generation with KV cache.

        Sampling supports top-k, top-p (nucleus), and repetition penalty —
        these can be combined freely.

        KV cache strategy:
          Step 0 (prefill): forward the full prompt, cache all K/V.
          Steps 1-N: forward only the latest single token; O(T) per step.
          When accumulated length would exceed block_size: evict all but the
          last (block_size - 1) entries by recomputing from context.
        """
        past_kvs = None

        for _ in range(max_new_tokens):
            cached_len = 0 if past_kvs is None else past_kvs[0][0].shape[2]

            if past_kvs is None:
                # Prefill: process entire prompt (crop if longer than block_size)
                idx_cond = (
                    idx if idx.size(1) <= self.config.block_size
                    else idx[:, -self.config.block_size:]
                )
            elif cached_len + 1 > self.config.block_size:
                # Cache full: reset and recompute from last (block_size - 1) tokens
                past_kvs = None
                idx_cond = idx[:, -(self.config.block_size - 1):]
            else:
                # Cached step: only the single new token
                idx_cond = idx[:, -1:]

            logits, _, past_kvs = self(idx_cond, past_kvs=past_kvs)
            logits = logits[:, -1, :]   # [B, vocab_size]

            if repetition_penalty != 1.0:
                logits = apply_repetition_penalty(logits, idx, repetition_penalty)

            logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            # Top-p (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
