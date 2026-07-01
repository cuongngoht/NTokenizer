# Vietnamese GPT v2

Build a Vietnamese language model entirely from scratch — no pre-trained weights,
no Hugging Face abstractions.  Every step is a plain Python script you can read,
modify, and learn from.

```
Wikipedia XML  →  clean text  →  BPE tokenizer (32k)  →  binary dataset
                                                                ↓
                                                  GPT v2 (PyTorch)
                                                  RoPE · GQA · SwiGLU · RMSNorm
                                                                ↓
                                                         training loop
                                                                ↓
                                                        generate text
                                               (KV cache · top-p · rep. penalty)
```

> **Want to build the model right now?** The full command-by-command pipeline
> — from downloading Wikipedia to generating text — lives in
> [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md). This README covers the project
> layout, install, and the *why* behind the architecture.

---

## Table of contents

1. [Project structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Design decisions](#design-decisions)

---

## Project structure

```
NTokenizer/
├── pyproject.toml                              # installable package + dependencies
│
├── src/ntokenizer/                             # the installable ntokenizer package
│   ├── paths.py                                # every path constant, in one place
│   ├── config.py                                # GPTConfig, TrainConfig
│   ├── model.py                                 # GPT v2 Transformer
│   ├── corpus.py                                # Wikipedia text cleaning
│   ├── tokenizer.py                             # SentencePiece loading helpers
│   ├── dataset.py                               # binary dataset read/write
│   ├── optim.py                                 # LR schedule + AdamW setup
│   ├── training.py                              # training loop
│   └── cli/                                     # one module per pipeline step
│
├── scripts/                                     # thin CLI entry points — run these
│   ├── build_corpus.py                          # extract + clean Wikipedia text
│   ├── train_tokenizer_spm.py                   # train BPE tokenizer
│   ├── test_tokenizer.py                        # verify encode/decode round-trip
│   ├── inspect_vocab.py                         # explore the vocabulary
│   ├── prepare_dataset.py                       # encode corpus → binary files
│   ├── train.py                                 # training loop (AdamW + cosine LR)
│   └── sample.py                                # load checkpoint, generate text
│
├── tests/                                       # pytest suite (model, tokenizer, data pipeline)
│
├── docs/
│   ├── DEVELOPMENT.md                           # command-by-command build pipeline
│   ├── model_architecture.md                    # deep dive into the model
│   └── model_architecture.vi.md                 # bản tiếng Việt
│
├── data/                                         # gitignored — all raw/intermediate/processed data
│   ├── raw/
│   │   └── viwiki-latest-pages-articles.xml.bz2   # Wikipedia XML dump (~1.1 GB)
│   ├── interim/
│   │   ├── extracted/                             # wikiextractor output — JSON Lines files
│   │   └── corpus.txt                             # cleaned plain text (~1.3 GB)
│   └── processed/
│       ├── train.bin                              # 90% of tokens as uint16
│       ├── val.bin                                 # 10% of tokens as uint16
│       └── meta.json                               # vocab_size, token counts, tokenizer path
│
└── artifacts/
    ├── tokenizer/
    │   ├── viwiki_bpe_32k.model                    # trained SentencePiece model (default, gitignored)
    │   ├── viwiki_bpe_32k.vocab                    # vocabulary: token + log-prob score (gitignored)
    │   ├── viwiki_bpe_8k.model                     # legacy 8k model (kept for reference, git-tracked)
    │   └── viwiki_bpe_8k.vocab
    └── checkpoints/
        └── ckpt.pt                                 # best checkpoint (created by scripts/train.py)
```

---

## Prerequisites

| Requirement | Recommended |
|---|---|
| Python | 3.10 or newer |
| RAM | 8 GB minimum, 16 GB recommended |
| Disk | ~5 GB free (raw dump + extracted + corpus + binaries) |
| Hardware | MacBook M-series (MPS), any CUDA GPU, or CPU (slow) |

No GPU is strictly required.  On an Apple M-series chip training runs on the
Metal Performance Shaders (MPS) backend.  On CPU a 5 000-step run takes several
hours; on MPS or a mid-range GPU it takes 30–60 minutes.

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd NTokenizer

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install the package (editable) + dev tools (pytest)
pip install -e ".[dev]"

# Also need to build the corpus from a Wikipedia dump? Add the wikiextractor extra:
pip install -e ".[dev,wikiextract]"
```

`pyproject.toml` declares:

| Package | Used for |
|---|---|
| `sentencepiece` | BPE tokenizer training and encoding |
| `numpy` | Binary dataset storage and batch sampling |
| `torch` | Model definition and training |
| `tqdm` | Progress bars |
| `pytest` (`dev` extra) | Running the test suite |
| `wikiextractor` (`wikiextract` extra) | Converting Wikipedia XML to plain text — only needed for Step 1 |

Installing in editable mode makes `ntokenizer` importable from anywhere
(`from ntokenizer.model import GPT`) and exposes convenience console scripts
(`ntok-train`, `ntok-sample`, …) — but `scripts/*.py` remain the primary,
documented way to run each pipeline step, since they're plain files you can
open and read top to bottom.

Next: [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) walks through every command
to actually build the corpus, tokenizer, dataset, and model.

---

## Design decisions

### Why BPE?

BPE (Byte Pair Encoding) builds a vocabulary by iteratively merging the most
frequent adjacent byte-pairs.  For Vietnamese it strikes a good balance:

- Common syllables (`Việt`, `Nam`, `thủ`, `đô`, …) become single tokens.
- Rare or foreign words are split into smaller known pieces.
- No information is lost — every string is encodable.

BPE is the algorithm behind GPT-2, RoBERTa, LLaMA, and most modern LLMs.

### Why vocab_size = 32 000?

32 000 tokens gives Vietnamese much better coverage than smaller vocabularies:

- ~7 000 distinct Vietnamese syllables as single tokens.
- Thousands of common multi-syllable words and phrases as merged tokens.
- Room for digits, punctuation, code, and foreign words without byte-level
  fragmentation.

Use `--vocab_size 8000` if you need a smaller, faster model for experimentation.

### Why preserve Vietnamese diacritics?

Vietnamese diacritics carry **lexical and tonal meaning** — they are not
decorative.  Removing them or lowercasing collapses distinct words:

| Word | Meaning |
|---|---|
| `ma` | ghost |
| `má` | cheek / mother (Southern) |
| `mà` | but / yet |
| `mả` | tomb |
| `mã` | code / horse (Sino-Vietnamese) |
| `mạ` | rice seedling |

### Why RoPE instead of learned positional embeddings?

Learned positional embeddings (`wpe`) assign a fixed vector to each absolute
position.  They cannot generalise beyond the `block_size` they were trained on.

Rotary Positional Embeddings (RoPE) encode position by rotating the Q and K
vectors in complex space.  The rotation at position `t` is:

```
q_rotated = q * e^{i * t * θ}   (complex multiplication, applied per head-dim pair)
```

Benefits:
- **No extra parameters** — the frequencies are computed once and stored as a buffer.
- **Relative attention** — dot-product `q · k` naturally captures the *relative*
  distance between positions, not just absolute indices.
- **Length generalisation** — the model can attend to positions it never saw
  during training (up to a point), useful when generating long sequences.

RoPE is used in LLaMA, Mistral, Qwen, DeepSeek, and virtually all modern open LLMs.

### Why RMSNorm instead of LayerNorm?

Standard LayerNorm subtracts the mean, divides by the standard deviation, then
applies a learned scale (`weight`) and bias.  RMSNorm skips the mean subtraction
and bias:

```
RMSNorm(x) = x / RMS(x) * weight       RMS(x) = sqrt(mean(x²) + ε)
```

Two practical advantages:
- **Fewer parameters** — no bias vector.
- **Slightly faster** — one fewer statistic to compute per forward pass.

Empirically, removing the mean subtraction has no measurable quality cost at
model sizes up to billions of parameters.  Used in LLaMA, Mistral, Falcon, etc.

### Why SwiGLU instead of GELU?

The original GPT-2 MLP is:

```
MLP(x) = W₂ · GELU(W₁ · x)     hidden_dim = 4 × n_embd
```

SwiGLU introduces a gate:

```
SwiGLU(x) = W_down · (SiLU(W_gate · x) ⊙ W_up · x)    hidden_dim ≈ 8/3 × n_embd
```

The gate (`SiLU(W_gate · x)`) acts as a learned filter over the up-projected
features.  In practice this consistently produces lower loss at the same
parameter count.  The 8/3 expansion factor (instead of 4×) keeps total
parameters similar while adding the gate.

Used in LLaMA, PaLM, Gemma, and most post-2023 open LLMs.

### Why Grouped Query Attention (GQA)?

Standard Multi-Head Attention (MHA) has H query heads **and** H key/value heads.
During generation, the model must store one K and one V matrix per layer per head
in the KV cache — this grows as `O(T × H × head_dim)`.

GQA splits Q heads into groups that share a single pair of KV heads:

```
n_head = 8 (Q heads)    n_kv_head = 2 (KV heads)    → 4 Q heads share 1 KV pair
```

The KV cache shrinks by `n_head / n_kv_head` × — here 4×.  Quality loss is
negligible when `n_kv_head ≥ 2`.  Setting `n_kv_head = n_head` recovers standard MHA.

Used in LLaMA 2/3, Mistral, Gemma, Falcon, etc.

### Why KV Cache?

Without a cache, generating each new token requires a full forward pass through
the entire context.  If the context has T tokens:
- Each attention layer computes Q, K, V for all T tokens: **O(T²)** per step.
- Total cost for N new tokens: **O(N × T²)**.

With the KV cache, keys and values from past tokens are stored and reused:
- Only the new token's Q, K, V are computed: **O(T)** per step.
- Past K and V are read from cache — no recomputation.
- Total cost: **O(T + N × T)** ≈ **O(N × T)**.

For a 256-token context generating 200 new tokens, the cache gives a ~50× speedup
in attention computation.

### Why uint16 for the binary dataset?

Each token ID fits in a 16-bit unsigned integer (range 0–65 535).  At
`vocab_size = 32 000` this is well within the limit.  `uint16` uses half the
memory of `int32` and loads faster from disk — important when the dataset is
hundreds of millions of tokens.

### Why a decoder-only Transformer?

Encoder-decoder models (T5, BART) excel at translation and summarisation where
input and output are clearly separated.  Decoder-only models (GPT family) are
simpler, train faster on raw text, and are the dominant architecture for
language modelling.  For the goal of learning how LLMs work from scratch, the
decoder-only design is the clearest to implement and understand.
