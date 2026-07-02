# Development guide — build the LLM step by step

*Project overview, folder structure, and install instructions live in the
[README](../README.md). This file is the command-by-command build pipeline.*

---

## Table of contents

1. [Quickstart](#quickstart--run-every-command-in-order)
2. [Step 1 — Build corpus](#step-1--build-corpus)
3. [Step 2 — Train BPE tokenizer](#step-2--train-bpe-tokenizer)
4. [Step 3 — Test tokenizer](#step-3--test-tokenizer)
5. [Step 4 — Inspect vocabulary](#step-4--inspect-vocabulary)
6. [Step 5 — Prepare binary dataset](#step-5--prepare-binary-dataset)
7. [Step 6 — Model architecture](#step-6--model-architecture)
8. [Step 7 — Training loop](#step-7--training-loop)
9. [Step 8 — Generate text](#step-8--generate-text)
10. [How the pieces fit together](#how-the-pieces-fit-together)
11. [Training tips](#training-tips)
12. [Troubleshooting](#troubleshooting)
13. [Running the tests](#running-the-tests)

---

## Quickstart — run every command in order

Copy-paste block to build the whole pipeline end to end. Each command is
explained in detail in its own [step](#step-1--build-corpus) further down —
this is just the fast path. See the [README](../README.md#setup) if you
haven't installed the package yet.

```bash
# 0. Setup
python3 -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -e ".[dev,wikiextract]"

# 1. Get the Wikipedia dump and extract plain text
wget https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2 \
     -P data/raw/
wikiextractor data/raw/viwiki-latest-pages-articles.xml.bz2 \
              --output data/interim/extracted/ --json --quiet

# 2. Build the clean corpus
python scripts/build_corpus.py

# 3. Train the BPE tokenizer (vocab_size auto-picked from corpus size)
python scripts/train_tokenizer_spm.py

# 4. Sanity-check the tokenizer (optional)
python scripts/test_tokenizer.py
python scripts/inspect_vocab.py

# 5. Encode the corpus into binary training data
python scripts/prepare_dataset.py

# 6. Train the model
python scripts/train.py

# 7. Generate text from the trained checkpoint
python scripts/sample.py --prompt "Hà Nội là thủ đô"
```

Resuming a partial run: skip any step whose output already exists — every
script checks for its inputs and tells you which earlier step to (re)run if
something's missing (see [Troubleshooting](#troubleshooting)).

---

## Step 1 — Build corpus

```bash
python scripts/build_corpus.py
```

**What it does**

Reads the JSON Lines files produced by `wikiextractor` in `data/interim/extracted/`,
cleans each paragraph, and writes one paragraph per line to `data/interim/corpus.txt`.

Cleaning steps applied to every line:
- Strip HTML tags
- Decode HTML entities (`&amp;` → `&`, etc.)
- Collapse multiple spaces
- Skip lines shorter than 10 characters
- Skip Wikipedia section headers (`== History ==`, etc.)
- Skip stub-only articles

**Output**

```
data/interim/corpus.txt   ~1.3 GB   ~1.2 million paragraphs
```

**How to get the Wikipedia dump**

```bash
# Download the latest Vietnamese Wikipedia XML dump (~450 MB compressed)
wget https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2 \
     -P data/raw/

# Extract with wikiextractor (installed via the `wikiextract` extra)
wikiextractor data/raw/viwiki-latest-pages-articles.xml.bz2 \
              --output data/interim/extracted/ --json --quiet
```

---

## Step 2 — Train BPE tokenizer

```bash
# Auto-picks vocab_size from your corpus file size (recommended default)
python scripts/train_tokenizer_spm.py

# Force a specific vocab size instead
python scripts/train_tokenizer_spm.py --vocab_size 32000
python scripts/train_tokenizer_spm.py --vocab_size 8000

# Custom corpus or output directory
python scripts/train_tokenizer_spm.py --vocab_size 32000 \
    --input data/interim/corpus.txt \
    --output_dir artifacts/tokenizer/
```

**What it does**

Trains a Byte Pair Encoding (BPE) tokenizer on up to 1 million sentences
sampled from `data/interim/corpus.txt` using the `sentencepiece` library.

**Auto-sized `vocab_size`**

If you don't pass `--vocab_size`, it's picked from your corpus file size
(`ntokenizer.tokenizer.estimate_vocab_size`) instead of a hardcoded number —
a bigger corpus supports (and benefits from) a bigger vocabulary:

| Corpus size | vocab_size |
|---|---|
| < 5 MB | 4 000 |
| 5–50 MB | 8 000 |
| 50–200 MB | 16 000 |
| 200 MB – 1 GB | 32 000 |
| > 1 GB | 48 000 |

The ceiling (48 000) stays comfortably under 65 535 — the binary dataset
format ([Step 5](#step-5--prepare-binary-dataset)) stores token IDs as
`uint16`, which tops out at 65 535. Pass `--vocab_size` explicitly to
override the automatic choice.

The output model is always named after whatever `vocab_size` was actually
used (`viwiki_bpe_{N}k.model`). The later steps' default `--model` /
`--tokenizer` flags point at `viwiki_bpe_32k.model` — if your corpus
auto-picked a different size, pass `--model artifacts/tokenizer/viwiki_bpe_Nk.model`
(and `--tokenizer` at [Step 8](#step-8--generate-text)) explicitly in every
following step.

Training parameters:

| Parameter | Value | Why |
|---|---|---|
| `vocab_size` | auto (see above), or your `--vocab_size` | Good coverage of Vietnamese syllables + subwords + multilingual tokens |
| `model_type` | `bpe` | Iterative merge — good balance for Vietnamese morphology |
| `character_coverage` | 0.9995 | Keeps rare characters as byte-level fallbacks |
| Special tokens | `<unk>` `<pad>` `<bos>` `<eos>` | Required by the training pipeline |

Takes 10–20 minutes on a modern laptop for 32k vocab.

**Output**

```
artifacts/tokenizer/viwiki_bpe_32k.model   ~1.2 MB   binary SentencePiece model
artifacts/tokenizer/viwiki_bpe_32k.vocab   ~400 KB   TSV: token + log-probability
```

---

## Step 3 — Test tokenizer

```bash
# Test the 32k tokenizer (default)
python scripts/test_tokenizer.py

# Test a specific model
python scripts/test_tokenizer.py --model artifacts/tokenizer/viwiki_bpe_32k.model
python scripts/test_tokenizer.py --model artifacts/tokenizer/viwiki_bpe_8k.model
```

Encodes 5 sample Vietnamese sentences, decodes them back, and verifies that
Vietnamese diacritics are preserved in the round trip. This is a manual demo —
see [Running the tests](#running-the-tests) for the automated pytest coverage.

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
python scripts/inspect_vocab.py --vocab artifacts/tokenizer/viwiki_bpe_32k.vocab
python scripts/inspect_vocab.py --vocab artifacts/tokenizer/viwiki_bpe_8k.vocab
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
python scripts/prepare_dataset.py --model artifacts/tokenizer/viwiki_bpe_32k.model
```

**What it does**

1. Loads the trained BPE model.
2. Reads `data/interim/corpus.txt` line by line, skipping blank lines.
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
data/processed/train.bin    ~90% of tokens   uint16 flat array
data/processed/val.bin      ~10% of tokens   uint16 flat array
data/processed/meta.json    metadata
```

`meta.json` looks like:

```json
{
  "vocab_size": 32000,
  "dtype": "uint16",
  "total_tokens": 180000000,
  "train_tokens": 162000000,
  "val_tokens":   18000000,
  "tokenizer": "artifacts/tokenizer/viwiki_bpe_32k.model"
}
```

**Reload the binary data in your own scripts**

```python
import numpy as np
train = np.memmap("data/processed/train.bin", dtype="uint16", mode="r")
# train[i] is token ID at position i
```

---

## Step 6 — Model architecture

**Architecture overview**

A decoder-only Transformer with four modern upgrades over vanilla GPT-2,
implemented in `src/ntokenizer/model.py` with no external dependencies beyond PyTorch.

> For a full explanation of every component — RoPE, RMSNorm, SwiGLU, GQA, and
> the KV cache — see [`model_architecture.md`](model_architecture.md)
> ([bản tiếng Việt](model_architecture.vi.md)), which also includes a
> copy-pasteable sanity-check snippet.

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
python scripts/train.py \
    --n_layer 8 --n_head 8 --n_kv_head 2 \
    --n_embd 512 --block_size 512 \
    --batch_size 16
```

---

## Step 7 — Training loop

```bash
python scripts/train.py
```

**What it does**

1. Reads random batches from `data/processed/train.bin` using `np.memmap`.
2. Runs forward pass → computes cross-entropy loss.
3. Runs backward pass → computes gradients.
4. Clips gradient norm to 1.0 to prevent instability.
5. Updates weights with AdamW (weight decay applied to matrices only, not
   RMSNorm scales).
6. Evaluates on `data/processed/val.bin` every `eval_interval` steps.
7. Saves the best checkpoint (lowest val loss) to `artifacts/checkpoints/ckpt.pt`.

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
| `--out_dir` | `artifacts/checkpoints/` | Where to save checkpoints |
| `--data_dir` | `data/processed/` | Where to read train/val/meta from |

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
# Just re-run — if artifacts/checkpoints/ckpt.pt exists it resumes automatically
python scripts/train.py
```

**Quick smoke test (100 steps)**

```bash
python scripts/train.py --max_iters 100 --eval_interval 50 --eval_iters 10
```

---

## Step 8 — Generate text

```bash
python scripts/sample.py
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
| `--ckpt` | `artifacts/checkpoints/ckpt.pt` | Path to checkpoint |
| `--tokenizer` | `artifacts/tokenizer/viwiki_bpe_32k.model` | SentencePiece model |
| `--device` | auto | Force `cpu` / `cuda` / `mps` |

**Examples**

```bash
# Default — no prompt, 200 tokens
python scripts/sample.py

# Seed with Vietnamese text
python scripts/sample.py --prompt "Hà Nội là thủ đô"

# More tokens, nucleus sampling, repetition penalty
python scripts/sample.py --prompt "Lịch sử Việt Nam" \
    --max_new_tokens 300 --temperature 0.9 \
    --top_k 50 --top_p 0.95 --repetition_penalty 1.15

# Compare 3 different continuations
python scripts/sample.py --prompt "Việt Nam" --num_samples 3

# Focused / near-deterministic output
python scripts/sample.py --prompt "Ngôn ngữ" --temperature 0.5 --top_k 20
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

## How the pieces fit together

```
Wikipedia XML dump
        │
        ▼  wikiextractor
data/interim/extracted/ (JSON Lines)
        │
        ▼  scripts/build_corpus.py
data/interim/corpus.txt   ← ~1.3 GB plain text, 1 paragraph per line
        │
        ├──▶  scripts/train_tokenizer_spm.py
        │             │
        │             ▼
        │     artifacts/tokenizer/viwiki_bpe_32k.model  ← maps text ↔ integer IDs
        │
        ▼  scripts/prepare_dataset.py
data/processed/train.bin   ← flat array of uint16 token IDs (90 %)
data/processed/val.bin     ← flat array of uint16 token IDs (10 %)
data/processed/meta.json
        │
        ▼  scripts/train.py
artifacts/checkpoints/ckpt.pt   ← trained model weights
        │
        ▼  scripts/sample.py (with KV cache)
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
python scripts/train.py --n_head 8 --n_kv_head 2 --n_embd 512
```

**Faster training on Apple Silicon.**
The MPS backend is auto-detected.  If you see it is not being used:
```bash
python scripts/train.py --device mps
```

**Longer training for better quality.**
```bash
python scripts/train.py --max_iters 20000 --eval_interval 1000
```

**Larger model (~40 M params, Apple M-series).**
```bash
python scripts/train.py \
    --n_layer 8 --n_head 8 --n_kv_head 2 \
    --n_embd 512 --block_size 512 \
    --batch_size 16 --max_iters 20000
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'ntokenizer'`**
```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

**`ERROR: file not found: data/interim/corpus.txt`**
```bash
python scripts/build_corpus.py
```

**`ERROR: file not found: artifacts/tokenizer/viwiki_bpe_32k.model`**
```bash
python scripts/train_tokenizer_spm.py --vocab_size 32000
```

**`ERROR: file not found: data/processed/train.bin`**
```bash
python scripts/prepare_dataset.py
```

**`ERROR: checkpoint not found: artifacts/checkpoints/ckpt.pt`**
```bash
python scripts/train.py     # must train before sampling
```

**Loss is NaN after a few steps.**
The learning rate is too high — gradients are exploding.  Try:
```bash
python scripts/train.py --learning_rate 1e-4 --grad_clip 0.5
```

**MPS backend crashes or gives wrong results.**
Fall back to CPU:
```bash
python scripts/train.py --device cpu
python scripts/sample.py --device cpu
```

**Generated text looks like garbage.**
The model needs more training.  Check `val_loss` in the checkpoint:

| Val loss | Quality |
|---|---|
| > 4.0 | Random-looking subwords |
| ~3.0 | Recognisable Vietnamese words, poor coherence |
| ~2.5 | Sentences start to make sense |
| < 2.0 | Mostly fluent Vietnamese (requires long training) |

---

## Running the tests

The `tests/` directory has automated pytest coverage for the model, tokenizer,
and data pipeline — separate from the manual demo scripts (`scripts/test_tokenizer.py`)
described above. Runs in well under a minute on CPU, with no real Wikipedia data
or GPU required.

```bash
pip install -e ".[dev]"
pytest
```
