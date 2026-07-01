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

---

## Table of contents

1. [Project structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Step 1 — Build corpus](#step-1--build-corpus)
5. [Step 2 — Train BPE tokenizer](#step-2--train-bpe-tokenizer)
6. [Step 3 — Test tokenizer](#step-3--test-tokenizer)
7. [Step 4 — Inspect vocabulary](#step-4--inspect-vocabulary)
8. [Step 5 — Prepare binary dataset](#step-5--prepare-binary-dataset)
9. [Step 6 — Model architecture](#step-6--model-architecture)
10. [Step 7 — Training loop](#step-7--training-loop)
11. [Step 8 — Generate text](#step-8--generate-text)
12. [Design decisions](#design-decisions)
13. [How the pieces fit together](#how-the-pieces-fit-together)
14. [Training tips](#training-tips)
15. [Troubleshooting](#troubleshooting)

---

## Project structure

```
NTokenizer/
├── raw/
│   └── viwiki-latest-pages-articles.xml.bz2   # Wikipedia XML dump (~1.1 GB)
│
├── extracted/                                  # wikiextractor output (gitignored)
│   └── AA/ AB/ ...                             # JSON Lines files
│
├── clean/
│   └── corpus.txt                              # Cleaned plain text (gitignored, ~1.3 GB)
│
├── tokenizer/
│   ├── viwiki_bpe_32k.model                    # Trained SentencePiece model (default)
│   ├── viwiki_bpe_32k.vocab                    # Vocabulary: token + log-prob score
│   ├── viwiki_bpe_8k.model                     # Legacy 8k model (kept for reference)
│   └── viwiki_bpe_8k.vocab
│
├── data/
│   ├── train.bin                               # 90% of tokens as uint16 (gitignored)
│   ├── val.bin                                 # 10% of tokens as uint16 (gitignored)
│   └── meta.json                               # vocab_size, token counts, tokenizer path
│
├── scripts/
│   ├── build_corpus.py                         # Step 1 – extract + clean Wikipedia text
│   ├── train_tokenizer_spm.py                  # Step 2 – train BPE tokenizer
│   ├── test_tokenizer.py                       # Step 3 – verify encode/decode round-trip
│   ├── inspect_vocab.py                        # Step 4 – explore the vocabulary
│   └── prepare_dataset.py                      # Step 5 – encode corpus → binary files
│
├── src/
│   ├── model.py                                # Step 6 – GPT v2 Transformer
│   ├── train.py                                # Step 7 – training loop (AdamW + cosine LR)
│   └── sample.py                               # Step 8 – load checkpoint, generate text
│
├── out/
│   └── ckpt.pt                                 # Best checkpoint (created by train.py)
│
└── requirements.txt
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

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` installs:

| Package | Used for |
|---|---|
| `sentencepiece` | BPE tokenizer training and encoding |
| `numpy` | Binary dataset storage and batch sampling |
| `torch` | Model definition and training |
| `tqdm` | Progress bars |
| `wikiextractor` | Converting Wikipedia XML to plain text |

---

## Step 1 — Build corpus

```bash
python scripts/build_corpus.py
```

**What it does**

Reads the JSON Lines files produced by `wikiextractor` in `extracted/`, cleans
each paragraph, and writes one paragraph per line to `clean/corpus.txt`.

Cleaning steps applied to every line:
- Strip HTML tags
- Decode HTML entities (`&amp;` → `&`, etc.)
- Collapse multiple spaces
- Skip lines shorter than 10 characters
- Skip Wikipedia section headers (`== History ==`, etc.)
- Skip stub-only articles

**Output**

```
clean/corpus.txt   ~1.3 GB   ~1.2 million paragraphs
```

**How to get the Wikipedia dump**

```bash
# Download the latest Vietnamese Wikipedia XML dump (~450 MB compressed)
wget https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2 \
     -P raw/

# Extract with wikiextractor (installed by requirements.txt)
wikiextractor raw/viwiki-latest-pages-articles.xml.bz2 \
              --output extracted/ --json --quiet
```

---

## Step 2 — Train BPE tokenizer

```bash
# Train 32k tokenizer (default / recommended)
python scripts/train_tokenizer_spm.py --vocab_size 32000

# Train 8k tokenizer (smaller, faster)
python scripts/train_tokenizer_spm.py --vocab_size 8000

# Custom corpus or output directory
python scripts/train_tokenizer_spm.py --vocab_size 32000 \
    --input clean/corpus.txt \
    --output_dir tokenizer/
```

**What it does**

Trains a Byte Pair Encoding (BPE) tokenizer on up to 1 million sentences
sampled from `clean/corpus.txt` using the `sentencepiece` library.

Training parameters:

| Parameter | Value | Why |
|---|---|---|
| `vocab_size` | 32 000 | Good coverage of Vietnamese syllables + subwords + multilingual tokens |
| `model_type` | `bpe` | Iterative merge — good balance for Vietnamese morphology |
| `character_coverage` | 0.9995 | Keeps rare characters as byte-level fallbacks |
| Special tokens | `<unk>` `<pad>` `<bos>` `<eos>` | Required by the training pipeline |

Takes 10–20 minutes on a modern laptop for 32k vocab.

**Output**

```
tokenizer/viwiki_bpe_32k.model   ~1.2 MB   binary SentencePiece model
tokenizer/viwiki_bpe_32k.vocab   ~400 KB   TSV: token + log-probability
```

---

## Step 3 — Test tokenizer

```bash
# Test the 32k tokenizer (default)
python scripts/test_tokenizer.py

# Test a specific model
python scripts/test_tokenizer.py --model tokenizer/viwiki_bpe_32k.model
python scripts/test_tokenizer.py --model tokenizer/viwiki_bpe_8k.model
```

Encodes 5 sample Vietnamese sentences, decodes them back, and verifies that
Vietnamese diacritics are preserved in the round trip.

Example output:

```
Input  : "Hà Nội là thủ đô của Việt Nam."
Pieces : ['▁Hà', '▁Nội', '▁là', '▁thủ', '▁đô', '▁của', '▁Việt', '▁Nam', '.']
IDs    : [1064, 1525, 283, 656, 897, 123, 987, 234, 5]
Decoded: "Hà Nội là thủ đô của Việt Nam."
Match  : ✓
```

---

## Step 4 — Inspect vocabulary

```bash
# Inspect the 32k vocab (default)
python scripts/inspect_vocab.py

# Inspect a specific vocab file
python scripts/inspect_vocab.py --vocab tokenizer/viwiki_bpe_32k.vocab
python scripts/inspect_vocab.py --vocab tokenizer/viwiki_bpe_8k.vocab
```

Prints the first 100 tokens (special tokens + most frequent subwords) and
searches the vocabulary for specific Vietnamese strings such as `Hà`, `Nội`,
`Việt`, `Nam`, `thủ`, `đô`.

Useful for verifying that common Vietnamese syllables are represented as single
tokens rather than being split into byte-level fragments.

---

## Step 5 — Prepare binary dataset

```bash
# Encode with the 32k tokenizer (default)
python scripts/prepare_dataset.py

# Encode with a specific model
python scripts/prepare_dataset.py --model tokenizer/viwiki_bpe_32k.model
```

**What it does**

1. Loads the trained BPE model.
2. Reads `clean/corpus.txt` line by line, skipping blank lines.
3. Encodes each line to a list of integer token IDs.
4. Appends the EOS token (`id=3`) after every line so the model learns sentence
   boundaries.
5. Concatenates all IDs into one flat array.
6. Stores as `numpy uint16` (2 bytes per token — fits vocab ≤ 65 535).
7. Splits 90 % / 10 % into `train.bin` / `val.bin`.
8. Writes `meta.json` with dataset statistics.
9. Verifies by reloading `train.bin` with `np.memmap` and decoding the first
   100 tokens.

**Output**

```
data/train.bin    ~90% of tokens   uint16 flat array
data/val.bin      ~10% of tokens   uint16 flat array
data/meta.json    metadata
```

`meta.json` looks like:

```json
{
  "vocab_size": 32000,
  "dtype": "uint16",
  "total_tokens": 180000000,
  "train_tokens": 162000000,
  "val_tokens":   18000000,
  "tokenizer": "tokenizer/viwiki_bpe_32k.model"
}
```

**Reload the binary data in your own scripts**

```python
import numpy as np
train = np.memmap("data/train.bin", dtype="uint16", mode="r")
# train[i] is token ID at position i
```

---

## Step 6 — Model architecture

```bash
python src/model.py     # runs a forward pass + generation sanity check
```

**Architecture overview**

A decoder-only Transformer with four modern upgrades over vanilla GPT-2,
implemented in ~320 lines of PyTorch with no external dependencies.

> For a full explanation of every component — RoPE, RMSNorm, SwiGLU, GQA, and
> the KV cache — see [`docs/model_architecture.md`](docs/model_architecture.md)
> ([bản tiếng Việt](docs/model_architecture.vi.md)).

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

**Upgrades vs the original**

| Component | Original | v2 | Benefit |
|---|---|---|---|
| Positional encoding | Learned `wpe` table | **RoPE** | Generalises beyond training length; no extra parameters |
| Normalization | `LayerNorm` (with bias) | **RMSNorm** (no bias) | Simpler, slightly faster, equally stable |
| MLP activation | `GELU` with 4× expansion | **SwiGLU** with 8/3× expansion | Better gradient flow; consistently lower loss |
| Attention heads | Multi-Head Attention | **Grouped Query Attention** | Fewer KV heads → smaller KV cache, faster inference |
| Generation | Re-computes full context each step | **KV Cache** | O(T) per step instead of O(T²) |
| Sampling | Top-k only | **Top-k + Top-p + Rep. penalty** | More natural, less repetitive output |

**Default hyperparameters**

| Parameter | Value | Description |
|---|---|---|
| `vocab_size` | 32 000 | Matches the BPE tokenizer |
| `block_size` | 256 | Maximum context length in tokens |
| `n_layer` | 4 | Number of Transformer blocks |
| `n_head` | 4 | Number of query heads per block |
| `n_kv_head` | 4 | Number of KV heads (= `n_head` → standard MHA by default) |
| `n_embd` | 256 | Embedding / model width |
| `dropout` | 0.1 | Dropout probability (set to 0 at inference) |
| `rope_theta` | 10 000 | RoPE base frequency |

**Example larger config (Apple M-series, ~40 M params)**

```bash
python src/train.py \
    --n_layer 8 --n_head 8 --n_kv_head 2 \
    --n_embd 512 --block_size 512 \
    --batch_size 16
```

**Sanity check output**

```
Parameters : ~8.5 M  (default config)
logits     : [2, 64, 32000]
loss       : ~10.37  (expected ≈ ln(32000) for random weights)
KV cache   : 4 layers, K[2, 4, 64, 64]
```

---

## Step 7 — Training loop

```bash
python src/train.py
```

**What it does**

1. Reads random batches from `data/train.bin` using `np.memmap`.
2. Runs forward pass → computes cross-entropy loss.
3. Runs backward pass → computes gradients.
4. Clips gradient norm to 1.0 to prevent instability.
5. Updates weights with AdamW (weight decay applied to matrices only, not
   RMSNorm scales).
6. Evaluates on `data/val.bin` every `eval_interval` steps.
7. Saves the best checkpoint (lowest val loss) to `out/ckpt.pt`.

**Options**

| Flag | Default | Notes |
|---|---|---|
| `--max_iters` | 5 000 | Total training steps |
| `--batch_size` | 32 | Sequences per step |
| `--block_size` | 256 | Tokens per sequence |
| `--n_layer` | 4 | Transformer blocks |
| `--n_head` | 4 | Query attention heads |
| `--n_kv_head` | 4 | KV heads (set < `n_head` to enable GQA) |
| `--n_embd` | 256 | Model width |
| `--rope_theta` | 10000 | RoPE base frequency |
| `--learning_rate` | 3e-4 | Peak LR (cosine-decays to `min_lr`) |
| `--min_lr` | 3e-5 | Floor LR (1/10 of peak) |
| `--warmup_iters` | 100 | Steps of linear LR warm-up |
| `--weight_decay` | 0.1 | AdamW weight decay |
| `--grad_clip` | 1.0 | Max gradient norm |
| `--eval_interval` | 500 | Evaluate + maybe checkpoint every N steps |
| `--eval_iters` | 100 | Batches to average for a stable eval loss |
| `--device` | auto | Force `cpu` / `cuda` / `mps` |
| `--out_dir` | `out/` | Where to save checkpoints |

**Learning rate schedule**

```
  LR
  3e-4 ┤         ╭─── peak ─────────────────╮
       │        /                             \
  3e-5 ┤───────╯ warmup                   cosine decay ╰─── min
       └──────────────────────────────────────────────────── step
              0   100                                  5000
```

**Expected loss progression**

| Steps | Val loss | Notes |
|---|---|---|
| 0 | ~10.4 | Random weights — uniform over vocab |
| 500 | ~5–6 | Model picks up common words |
| 1 000 | ~4–5 | Vietnamese patterns emerging |
| 5 000 | ~3–4 | Coherent subword sequences |
| 20 000+ | ~2–3 | Readable Vietnamese sentences |

**Resuming training**

```bash
# Just re-run — if out/ckpt.pt exists it resumes automatically
python src/train.py
```

**Quick smoke test (100 steps)**

```bash
python src/train.py --max_iters 100 --eval_interval 50 --eval_iters 10
```

---

## Step 8 — Generate text

```bash
python src/sample.py
```

**Options**

| Flag | Default | Effect |
|---|---|---|
| `--prompt` | `""` | Seed text (empty = start from BOS token) |
| `--max_new_tokens` | 200 | Number of tokens to generate |
| `--temperature` | 0.8 | `< 1` = focused; `> 1` = creative / random |
| `--top_k` | 50 | Keep only the top-k tokens before sampling (`0` = disabled) |
| `--top_p` | 0.95 | Nucleus: keep smallest set whose cumulative prob ≥ p (`0` = disabled) |
| `--repetition_penalty` | 1.1 | Divide logits of already-seen tokens (`1.0` = disabled) |
| `--num_samples` | 1 | Generate N independent completions |
| `--ckpt` | `out/ckpt.pt` | Path to checkpoint |
| `--tokenizer` | `tokenizer/viwiki_bpe_32k.model` | SentencePiece model |
| `--device` | auto | Force `cpu` / `cuda` / `mps` |

**Examples**

```bash
# Default — no prompt, 200 tokens
python src/sample.py

# Seed with Vietnamese text
python src/sample.py --prompt "Hà Nội là thủ đô"

# More tokens, nucleus sampling, repetition penalty
python src/sample.py --prompt "Lịch sử Việt Nam" \
    --max_new_tokens 300 --temperature 0.9 \
    --top_k 50 --top_p 0.95 --repetition_penalty 1.15

# Compare 3 different continuations
python src/sample.py --prompt "Việt Nam" --num_samples 3

# Focused / near-deterministic output
python src/sample.py --prompt "Ngôn ngữ" --temperature 0.5 --top_k 20
```

**Understanding the sampling parameters**

The model outputs a probability distribution over all 32 000 tokens at each step.
Three controls shape how you sample from it — they can be combined:

- **temperature** divides logits before softmax.
  `0.5` → sharper (more predictable). `1.0` → raw distribution. `1.5` → flatter (more surprising).

- **top_k** restricts sampling to the k most probable tokens.
  `top_k=1` is greedy decoding. `top_k=50` is a good general default.

- **top_p** (nucleus) keeps the smallest set of tokens whose cumulative
  probability exceeds p, then samples from that set only.
  `top_p=0.95` is more adaptive than a fixed k — it narrows automatically when
  the model is confident and widens when it is uncertain.

- **repetition_penalty** > 1 down-weights tokens that appear earlier in the
  context, discouraging the model from looping on the same phrase.

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

---

## How the pieces fit together

```
Wikipedia XML dump
        │
        ▼  wikiextractor
extracted/ (JSON Lines)
        │
        ▼  scripts/build_corpus.py
clean/corpus.txt        ← ~1.3 GB plain text, 1 paragraph per line
        │
        ├──▶  scripts/train_tokenizer_spm.py
        │             │
        │             ▼
        │     tokenizer/viwiki_bpe_32k.model  ← maps text ↔ integer IDs
        │
        ▼  scripts/prepare_dataset.py
data/train.bin          ← flat array of uint16 token IDs (90 %)
data/val.bin            ← flat array of uint16 token IDs (10 %)
data/meta.json
        │
        ▼  src/train.py
out/ckpt.pt             ← trained model weights
        │
        ▼  src/sample.py (with KV cache)
"Hà Nội là thủ đô của nước Cộng hòa…"
```

---

## Training tips

**Monitor val loss, not train loss.**
Train loss measures how well the model memorises the current batch.
Val loss measures generalisation.  If train loss keeps falling but val loss
plateaus or rises, the model is overfitting — stop or increase dropout.

**When to stop.**
For an ~8 M-parameter model on this dataset, diminishing returns set in around
10 000–20 000 steps.  A val loss of ~3.0 produces recognisable Vietnamese
subword sequences.  A val loss below 2.5 produces mostly coherent sentences.

**Out of memory.**
Reduce `--batch_size` (try 16 or 8) or `--block_size` (try 128).
Memory scales roughly as `batch_size × block_size × n_embd × 4 bytes`.

**Enable GQA to save memory.**
```bash
# Use 2 KV heads instead of 8 — 4× smaller KV cache
python src/train.py --n_head 8 --n_kv_head 2 --n_embd 512
```

**Faster training on Apple Silicon.**
The MPS backend is auto-detected.  If you see it is not being used:
```bash
python src/train.py --device mps
```

**Longer training for better quality.**
```bash
python src/train.py --max_iters 20000 --eval_interval 1000
```

**Larger model (~40 M params, Apple M-series).**
```bash
python src/train.py \
    --n_layer 8 --n_head 8 --n_kv_head 2 \
    --n_embd 512 --block_size 512 \
    --batch_size 16 --max_iters 20000
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'X'`**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**`ERROR: file not found: clean/corpus.txt`**
```bash
python scripts/build_corpus.py
```

**`ERROR: file not found: tokenizer/viwiki_bpe_32k.model`**
```bash
python scripts/train_tokenizer_spm.py --vocab_size 32000
```

**`ERROR: file not found: data/train.bin`**
```bash
python scripts/prepare_dataset.py
```

**`ERROR: checkpoint not found: out/ckpt.pt`**
```bash
python src/train.py     # must train before sampling
```

**Loss is NaN after a few steps.**
The learning rate is too high — gradients are exploding.  Try:
```bash
python src/train.py --learning_rate 1e-4 --grad_clip 0.5
```

**MPS backend crashes or gives wrong results.**
Fall back to CPU:
```bash
python src/train.py --device cpu
python src/sample.py --device cpu
```

**Generated text looks like garbage.**
The model needs more training.  Check `val_loss` in the checkpoint:

| Val loss | Quality |
|---|---|
| > 4.0 | Random-looking subwords |
| ~3.0 | Recognisable Vietnamese words, poor coherence |
| ~2.5 | Sentences start to make sense |
| < 2.0 | Mostly fluent Vietnamese (requires long training) |
