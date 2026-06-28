# Tiny Vietnamese GPT

Build a small Vietnamese language model entirely from scratch — no pre-trained
weights, no Hugging Face abstractions.  Every step is a plain Python script you
can read, modify, and learn from.

```
Wikipedia XML  →  clean text  →  BPE tokenizer  →  binary dataset
                                                         ↓
                                              GPT architecture (PyTorch)
                                                         ↓
                                                   training loop
                                                         ↓
                                                  generate text
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
9. [Step 6 — GPT model architecture](#step-6--gpt-model-architecture)
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
│   ├── viwiki_bpe_8k.model                     # Trained SentencePiece model
│   └── viwiki_bpe_8k.vocab                     # Vocabulary: token + log-prob score
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
│   ├── model.py                                # Step 6 – GPT decoder-only Transformer
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
python scripts/train_tokenizer_spm.py
```

**What it does**

Trains a Byte Pair Encoding (BPE) tokenizer on up to 1 million sentences
sampled from `clean/corpus.txt` using the `sentencepiece` library.

Training parameters:

| Parameter | Value | Why |
|---|---|---|
| `vocab_size` | 8 000 | Covers ~7 000 Vietnamese syllables with room for subwords |
| `model_type` | `bpe` | Iterative merge — good balance for Vietnamese morphology |
| `character_coverage` | 0.9995 | Keeps rare characters as byte-level fallbacks |
| Special tokens | `<unk>` `<pad>` `<bos>` `<eos>` | Required by the training pipeline |

Takes 5–10 minutes on a modern laptop.

**Output**

```
tokenizer/viwiki_bpe_8k.model   ~360 KB   binary SentencePiece model
tokenizer/viwiki_bpe_8k.vocab   ~100 KB   TSV: token + log-probability
```

---

## Step 3 — Test tokenizer

```bash
python scripts/test_tokenizer.py
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
python scripts/inspect_vocab.py
```

Prints the first 100 tokens (special tokens + most frequent subwords) and
searches the vocabulary for specific Vietnamese strings such as `Hà`, `Nội`,
`Việt`, `Nam`, `thủ`, `đô`.

Useful for verifying that common Vietnamese syllables are represented as single
tokens rather than being split into byte-level fragments.

---

## Step 5 — Prepare binary dataset

```bash
python scripts/prepare_dataset.py
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
  "vocab_size": 8000,
  "dtype": "uint16",
  "total_tokens": 180000000,
  "train_tokens": 162000000,
  "val_tokens":   18000000,
  "tokenizer": "tokenizer/viwiki_bpe_8k.model"
}
```

**Reload the binary data in your own scripts**

```python
import numpy as np
train = np.memmap("data/train.bin", dtype="uint16", mode="r")
# train[i] is token ID at position i
```

---

## Step 6 — GPT model architecture

```bash
python src/model.py     # runs a sanity check
```

**Architecture**

A decoder-only Transformer (same family as GPT-2), implemented from scratch in
~250 lines of PyTorch.

```
input_ids  [B, T]
    │
    ├─ Token embedding   wte  [vocab_size, C]  →  [B, T, C]
    ├─ Position embedding wpe  [block_size, C]  →  [B, T, C]
    └─ x = tok_emb + pos_emb  + dropout        →  [B, T, C]
         │
         ├─ Block 0:  LayerNorm → CausalSelfAttention → residual
         │            LayerNorm → MLP                 → residual
         ├─ Block 1–3: (same)
         │
         └─ Final LayerNorm  →  [B, T, C]
              │
              └─ LM head  Linear(C, vocab_size)  →  logits  [B, T, vocab_size]
```

`B` = batch size, `T` = sequence length, `C` = `n_embd` = 256.

**Default hyperparameters**

| Parameter | Value | Description |
|---|---|---|
| `vocab_size` | 8 000 | Matches the BPE tokenizer |
| `block_size` | 256 | Maximum context length in tokens |
| `n_layer` | 4 | Number of Transformer blocks |
| `n_head` | 4 | Attention heads per block |
| `n_embd` | 256 | Embedding / model dimension |
| `dropout` | 0.1 | Dropout probability (0 at inference) |
| Total params | ~5.3 M | Fits comfortably on CPU or laptop GPU |

**Key implementation details**

- **Pre-norm** (LayerNorm before each sublayer) — more stable than post-norm.
- **Weight tying** — `lm_head` and `wte` share the same weight matrix, saving
  ~2 M parameters.
- **Flash Attention** — uses `F.scaled_dot_product_attention` on PyTorch ≥ 2.0
  for faster attention on MPS/CUDA; falls back to manual attention on older
  versions.
- **Causal mask** — lower-triangular mask ensures token `t` cannot attend to
  positions `t+1, t+2, …` (no peeking at future tokens).

**Sanity check output**

```
Parameters     : 5,273,088
logits shape   : [2, 64, 8000]
loss           : ~9.05  (expected ≈ ln(8000) = 8.99 for random weights)
generated IDs  : [0, 3421, 7102, ...]   shape [1, 21]
```

A freshly initialised model predicts roughly uniformly over all 8 000 tokens,
so the loss starts near `ln(8000) ≈ 8.99`.  Training drives it down.

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
5. Updates weights with AdamW.
6. Evaluates on `data/val.bin` every `eval_interval` steps.
7. Saves the best checkpoint (lowest val loss) to `out/ckpt.pt`.

**Options**

| Flag | Default | Notes |
|---|---|---|
| `--max_iters` | 5 000 | Total training steps |
| `--batch_size` | 32 | Sequences per step |
| `--block_size` | 256 | Tokens per sequence |
| `--learning_rate` | 3e-4 | Peak LR (cosine-decays to `min_lr`) |
| `--min_lr` | 3e-5 | Floor LR (1/10 of peak) |
| `--warmup_iters` | 100 | Steps of linear LR warm-up |
| `--weight_decay` | 0.1 | AdamW weight decay (matrices only) |
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

| Steps | Train loss | Val loss | Notes |
|---|---|---|---|
| 0 | ~9.0 | ~9.0 | Random weights — uniform over vocab |
| 500 | ~5–6 | ~5–6 | Model starts picking up common words |
| 1 000 | ~4–5 | ~4–5 | Vietnamese patterns emerging |
| 5 000 | ~3–4 | ~3–4 | Coherent subword sequences |
| 20 000+ | ~2–3 | ~2.5–3 | Readable Vietnamese text |

**Resuming training**

```bash
# Just re-run the same command — if out/ckpt.pt exists it resumes automatically
python src/train.py
```

**Running a quick smoke test (100 steps)**

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
| `--max_new_tokens` | 200 | How many tokens to generate |
| `--temperature` | 0.8 | `< 1` = focused / repetitive; `> 1` = creative / random |
| `--top_k` | 50 | Sample only from the top-k most likely tokens (`0` = no limit) |
| `--num_samples` | 1 | Generate N independent completions |
| `--ckpt` | `out/ckpt.pt` | Path to checkpoint |
| `--device` | auto | Force `cpu` / `cuda` / `mps` |

**Examples**

```bash
# Default — no prompt, 200 tokens
python src/sample.py

# Seed with Vietnamese text
python src/sample.py --prompt "Hà Nội là thủ đô"

# More tokens, more creative
python src/sample.py --prompt "Lịch sử Việt Nam" \
    --max_new_tokens 300 --temperature 1.0 --top_k 100

# Compare 3 different continuations from the same prompt
python src/sample.py --prompt "Việt Nam" --num_samples 3

# Focused / deterministic output
python src/sample.py --prompt "Ngôn ngữ" --temperature 0.5 --top_k 20
```

**Understanding temperature and top-k**

The model outputs a probability distribution over all 8 000 tokens.
Two settings control how you sample from it:

- **temperature** divides the logits before softmax.
  - `temperature=1.0` — sample from the raw distribution.
  - `temperature=0.5` — sharpen: high-probability tokens dominate, output is
    more predictable and repetitive.
  - `temperature=1.5` — flatten: low-probability tokens get a chance, output is
    more surprising and may be incoherent.

- **top_k** restricts sampling to the `k` most probable tokens and ignores the
  rest.
  - `top_k=1` — always pick the single most likely token (greedy decoding).
  - `top_k=50` — sample from the top 50 (good default).
  - `top_k=0` — no restriction; sample from the full vocabulary.

---

## Design decisions

### Why BPE?

BPE (Byte Pair Encoding) builds a vocabulary by iteratively merging the most
frequent adjacent byte-pairs.  For Vietnamese it strikes a good balance:

- Common syllables (`Việt`, `Nam`, `thủ`, `đô`, …) become single tokens.
- Rare or foreign words are split into smaller known pieces.
- No information is lost — every string is encodable.

BPE is the algorithm behind GPT-2, RoBERTa, LLaMA, and most modern LLMs.

### Why vocab_size = 8 000?

Vietnamese is a tonal, monosyllabic language with ~7 000 distinct syllables in
everyday use.  A vocabulary of 8 000 gives near-complete syllable coverage
while keeping the embedding matrix small enough for a tiny model.  Larger
vocabularies (GPT-2 uses 50 257; BERT uses 30 000) would be wasteful here.

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

Any tokenizer that strips diacritics would make these six words identical,
producing a model that cannot distinguish them.

### Why uint16 for the binary dataset?

Each token ID fits in a 16-bit unsigned integer (range 0–65 535).  At
`vocab_size = 8 000` this is well within the limit.  `uint16` uses half the
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
        │     tokenizer/viwiki_bpe_8k.model   ← maps text ↔ integer IDs
        │
        ▼  scripts/prepare_dataset.py
data/train.bin          ← flat array of uint16 token IDs (90 %)
data/val.bin            ← flat array of uint16 token IDs (10 %)
data/meta.json
        │
        ▼  src/train.py
out/ckpt.pt             ← trained model weights
        │
        ▼  src/sample.py
"Hà Nội là thủ đô của nước Cộng hòa…"
```

---

## Training tips

**Monitor val loss, not train loss.**
Train loss measures how well the model memorises the current batch.
Val loss measures how well it generalises.  If train loss keeps falling but val
loss plateaus or rises, the model is overfitting — stop or add dropout.

**When to stop.**
For a 5 M-parameter model on this dataset, diminishing returns set in around
10 000–20 000 steps.  A val loss of ~3.0 produces recognisable Vietnamese
subword sequences.  A val loss below 2.5 should produce mostly coherent
sentences.

**Out of memory.**
Reduce `--batch_size` (try 16 or 8) or `--block_size` (try 128).
Memory scales roughly as `batch_size × block_size × n_embd × 4 bytes`.

**Faster training on Apple Silicon.**
The MPS backend is auto-detected.  If you see it is not being used:
```bash
python src/train.py --device mps
```

**Longer training for better quality.**
```bash
python src/train.py --max_iters 20000 --eval_interval 1000
```

**Larger model (if you have more GPU memory).**
```bash
python src/train.py --n_layer 6 --n_head 6 --n_embd 384 --batch_size 16
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

**`ERROR: file not found: tokenizer/viwiki_bpe_8k.model`**
```bash
python scripts/train_tokenizer_spm.py
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
A learning rate that is too high causes gradients to explode.  Try:
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
`loss > 4` → needs significantly more steps.
`loss ~3` → basic Vietnamese subwords.
`loss ~2.5` → sentences start to make sense.
