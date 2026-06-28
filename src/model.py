"""
Tiny Vietnamese GPT — decoder-only Transformer architecture.

This is a minimal, readable implementation of a GPT-style language model.
It is intentionally small (~5 M parameters) so it can be trained on a
single MacBook or a free Colab GPU.

Architecture overview
---------------------
  input_ids [B, T]
    → token embedding  [B, T, C]   (lookup table: each token ID → C-dim vector)
    → + positional emb [B, T, C]   (each position 0..T-1 → C-dim vector)
    → N × Transformer Block
         pre-LayerNorm → CausalSelfAttention → residual add
         pre-LayerNorm → MLP                 → residual add
    → final LayerNorm  [B, T, C]
    → LM head (Linear) [B, T, vocab_size]    (logits over the vocabulary)

  training:  loss = cross_entropy(logits, targets)
  inference: sample next token from softmax(logits / temperature)

Notation used in comments:
  B = batch size
  T = sequence length (≤ block_size)
  C = n_embd  (embedding / model dimension)
  H = n_head  (number of attention heads)
  h = C // H  (head dimension)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """All hyperparameters in one place.  Change here → affects the whole model."""
    vocab_size: int = 32000  # number of tokens in the tokenizer vocabulary
    block_size: int = 256    # maximum context length (sequence length)
    n_layer:    int = 4      # number of Transformer blocks stacked
    n_head:     int = 4      # number of attention heads per block
    n_embd:     int = 256    # embedding dimension (model width)
    dropout:    float = 0.1  # dropout probability (0 = disabled)
    bias:       bool = True  # use bias in Linear layers and LayerNorm?


# ---------------------------------------------------------------------------
# Causal Self-Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal (autoregressive) mask.

    "Causal" means token at position t can only attend to positions 0..t.
    This is enforced by a lower-triangular mask that sets future positions
    to -inf before softmax, making their attention weights effectively zero.

    We project Q, K, V from the input with a single Linear(C, 3C) for
    efficiency, then split.  After attention, we project back to C.

    If PyTorch ≥ 2.0 is available we use F.scaled_dot_product_attention
    (which calls Flash Attention under the hood on CUDA / MPS).
    Otherwise we fall back to an explicit manual implementation.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            "n_embd must be divisible by n_head"

        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head  # dimension per head

        # Single projection for Q, K, V together (3× faster than 3 separate Linears)
        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection: after concatenating heads, map back to C
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask — lower-triangular matrix of ones.
        # register_buffer: saved in state_dict but not a learnable parameter.
        # Shape [1, 1, T, T] so it broadcasts over [B, H, T, T].
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size),
        )

        # Check whether we can use Flash Attention (PyTorch ≥ 2.0)
        self._use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # 1. Compute Q, K, V from input
        qkv = self.c_attn(x)               # [B, T, 3C]
        q, k, v = qkv.split(self.n_embd, dim=2)  # each [B, T, C]

        # 2. Reshape into multiple heads: [B, T, C] → [B, H, T, h]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Compute attention
        if self._use_flash:
            # Flash Attention: fused CUDA kernel, handles masking internally.
            # is_causal=True tells it to apply the causal mask automatically.
            dropout_p = self.dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            # Manual attention:
            # scores = Q·Kᵀ / sqrt(head_dim)   [B, H, T, T]
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (q @ k.transpose(-2, -1)) * scale

            # Apply causal mask: set future positions to -inf so softmax → 0
            scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

            # Softmax over the key dimension to get attention weights
            weights = F.softmax(scores, dim=-1)        # [B, H, T, T]
            weights = self.attn_dropout(weights)

            # Weighted sum of values
            y = weights @ v                            # [B, H, T, h]

        # 4. Reassemble heads: [B, H, T, h] → [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. Output projection + dropout
        return self.resid_dropout(self.c_proj(y))


# ---------------------------------------------------------------------------
# Feed-Forward Network (MLP)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Position-wise feed-forward network applied after attention.

    Each token is processed independently:
      C  →  4C  (expand — gives the model more capacity to learn)
      4C →  C   (project back)

    GELU activation is used (smoother than ReLU, standard in GPT-2+).
    The 4× expansion factor comes from the original Transformer paper.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, C] → [B, T, 4C] → [B, T, C]
        return self.drop(self.proj(self.gelu(self.fc(x))))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    One Transformer block: Attention sublayer + MLP sublayer.

    Uses the pre-LayerNorm variant (GPT-2 style):
      x = x + Attention(LayerNorm(x))
      x = x + MLP(LayerNorm(x))

    Pre-norm is more stable during training than the original post-norm
    because each sublayer receives a normalized input regardless of depth.

    The "+" is the residual (skip) connection — it lets gradients flow
    directly back to early layers without vanishing through many non-linearities.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # attention sublayer with residual
        x = x + self.mlp(self.ln2(x))    # MLP sublayer with residual
        return x


# ---------------------------------------------------------------------------
# GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    Decoder-only GPT language model.

    The model has two embedding tables (token + position), a stack of
    Transformer blocks, a final LayerNorm, and a linear LM head that
    projects embeddings to logits over the vocabulary.

    Weight tying: lm_head.weight shares the same tensor as wte.weight.
    This saves ~2 M parameters and is standard in GPT-2.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Token embedding: vocab_size × C lookup table
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            # Positional embedding: block_size × C lookup table
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # Stack of n_layer Transformer blocks
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final layer norm before the LM head
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        # LM head: project C-dim embedding → vocab_size logits
        # bias=False is standard (the embedding already has its own bias-like offset)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: token embedding and LM head share the same matrix.
        # Intuition: the same representation used to encode a token as input
        # should also score it as a candidate output.
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights (GPT-2 style)
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(n_layer) for stable training
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """Standard GPT-2 weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,            # [B, T]  — token IDs
        targets:   torch.Tensor | None = None,  # [B, T]  — next-token labels
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.

        Returns:
          logits : [B, T, vocab_size]  — raw (unnormalized) scores for each token
          loss   : scalar or None      — cross-entropy loss if targets are given
        """
        B, T = input_ids.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}. "
            "Truncate the input before calling forward()."
        )

        device = input_ids.device

        # 1. Embeddings -------------------------------------------------------
        # Token embedding: each ID → C-dim vector  [B, T, C]
        tok_emb = self.transformer.wte(input_ids)

        # Positional embedding: positions 0, 1, …, T-1  [T, C] → broadcast [B, T, C]
        positions = torch.arange(T, device=device)         # [T]
        pos_emb = self.transformer.wpe(positions)          # [T, C]

        # Add token + position embeddings, then apply dropout
        x = self.transformer.drop(tok_emb + pos_emb)      # [B, T, C]

        # 2. Transformer blocks -----------------------------------------------
        for block in self.transformer.h:
            x = block(x)                                   # [B, T, C]

        # 3. Final layer norm -------------------------------------------------
        x = self.transformer.ln_f(x)                       # [B, T, C]

        # 4. LM head: project to vocabulary logits ----------------------------
        if targets is not None:
            # Training: compute logits for all positions and calculate loss.
            logits = self.lm_head(x)                       # [B, T, vocab_size]

            # Reshape for cross-entropy: expects [N, C] and [N]
            # Each (batch, position) pair is treated as an independent prediction.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),          # [B*T, vocab_size]
                targets.view(-1),                          # [B*T]
            )
        else:
            # Inference: only compute logits for the last token position
            # (saves memory — we only care about what comes next).
            logits = self.lm_head(x[:, [-1], :])          # [B, 1, vocab_size]
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,   # [B, T]  — initial token sequence (seed)
        max_new_tokens: int,
        temperature:    float = 1.0,    # > 1 = more random; < 1 = more focused
        top_k:          int | None = None,  # if set, sample only from top-k tokens
    ) -> torch.Tensor:
        """
        Auto-regressively generate max_new_tokens new tokens.

        At each step:
          1. Crop context to block_size (model has a fixed attention window)
          2. Forward pass → logits for the last position
          3. Apply temperature scaling
          4. Optionally restrict to top-k logits (nucleus filtering)
          5. Softmax → probability distribution
          6. Sample one token
          7. Append to sequence and repeat
        """
        for _ in range(max_new_tokens):
            # Crop: model cannot attend beyond block_size tokens
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                           else idx[:, -self.config.block_size:]

            # Forward → logits [B, 1, vocab_size], squeeze to [B, vocab_size]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                      # [B, vocab_size]

            # Scale by temperature: low temp → peaky distribution (less creative)
            logits = logits / temperature

            # Top-k filtering: zero out all logits outside the top-k
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                # Get the k-th largest value as threshold
                threshold = torch.topk(logits, k).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            # Convert logits → probabilities
            probs = F.softmax(logits, dim=-1)              # [B, vocab_size]

            # Sample one token per batch element
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # Append to the running sequence
            idx = torch.cat([idx, next_token], dim=1)      # [B, T+1]

        return idx

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Tiny Vietnamese GPT — Architecture Sanity Check")
    print("=" * 55)

    # Build model with default (small) config
    config = GPTConfig(
        vocab_size = 32000,
        block_size = 1024,
        n_layer    = 8,
        n_head     = 8,
        n_embd     = 512,
        dropout    = 0.1,
        bias       = True,
    )
    model = GPT(config)
    print(f"\n  Config         : {config}")
    print(f"  Parameters     : {model.count_parameters():,}")

    # Use MPS (Apple Silicon) if available, else CPU
    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    model = model.to(device)
    print(f"  Device         : {device}")

    # --- Forward pass test ---
    print("\n[1] Forward pass (with loss)")
    B, T = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
    targets   = torch.randint(0, config.vocab_size, (B, T), device=device)

    logits, loss = model(input_ids, targets)

    print(f"  input_ids shape : {list(input_ids.shape)}")
    print(f"  logits shape    : {list(logits.shape)}")
    print(f"  loss            : {loss.item():.4f}  (expected ≈ {math.log(config.vocab_size):.4f} = ln({config.vocab_size}))")
    # A freshly initialized model should have a loss close to ln(vocab_size)
    # because the weights are random → roughly uniform distribution over vocab.

    # --- Inference / generation test ---
    print("\n[2] Generation test (20 new tokens)")
    seed = torch.zeros((1, 1), dtype=torch.long, device=device)  # seed = token 0
    generated = model.generate(seed, max_new_tokens=20, temperature=1.0, top_k=50)

    print(f"  seed shape      : {list(seed.shape)}")
    print(f"  generated shape : {list(generated.shape)}")
    print(f"  generated IDs   : {generated[0].tolist()}")

    print("\n" + "=" * 55)
    print("  All checks passed.  Ready for training loop.")
    print("=" * 55)
