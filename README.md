# NTokenizer — Tiny Vietnamese GPT

Build a small Vietnamese language model from scratch:
Wikipedia corpus → BPE tokenizer → binary dataset → GPT architecture → training → inference.

## Project structure

```
NTokenizer/
  raw/                        # Raw Wikipedia XML dump (.xml.bz2)
  extracted/                  # wikiextractor output (JSON Lines, ignored by git)
  clean/
    corpus.txt                # Cleaned plain-text corpus (ignored by git)
  tokenizer/
    viwiki_bpe_8k.model       # Trained SentencePiece BPE model
    viwiki_bpe_8k.vocab       # Vocabulary file (token + log-prob score)
  data/
    train.bin                 # Encoded training tokens (uint16, ignored by git)
    val.bin                   # Encoded validation tokens (uint16, ignored by git)
    meta.json                 # Dataset metadata (vocab size, token counts)
  scripts/
    build_corpus.py           # Step 1 – extract + clean Wikipedia text
    train_tokenizer_spm.py    # Step 2 – train BPE tokenizer
    test_tokenizer.py         # Step 3 – verify encode/decode
    inspect_vocab.py          # Step 4 – explore the vocabulary
    prepare_dataset.py        # Step 5 – encode corpus → train.bin / val.bin
  src/
    model.py                  # Step 6 – GPT model architecture (PyTorch)
    train.py                  # Step 7 – training loop
  requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1 – Build corpus (skip if `clean/corpus.txt` already exists)

```bash
python scripts/build_corpus.py
```

Reads `extracted/AA–AE/wiki_*` JSON Lines files, cleans the text, and writes
`clean/corpus.txt` (~435 MB, ~1.2 M paragraphs).

## Step 2 – Train tokenizer

```bash
python scripts/train_tokenizer_spm.py
```

Trains a BPE model on up to 1 million sentences from `clean/corpus.txt`.
Takes roughly 5–10 minutes. Outputs:
- `tokenizer/viwiki_bpe_8k.model`
- `tokenizer/viwiki_bpe_8k.vocab`

## Step 3 – Test tokenizer

```bash
python scripts/test_tokenizer.py
```

Encodes and decodes 5 Vietnamese sentences, prints pieces + IDs, and checks
that Vietnamese diacritics are preserved.

## Step 4 – Inspect vocab

```bash
python scripts/inspect_vocab.py
```

Prints the first 100 tokens and searches for Vietnamese subwords such as
`Việt`, `Hà`, `Nội`, `Nam`, `thủ`, `đô`.

## Step 5 – Prepare binary dataset

```bash
python scripts/prepare_dataset.py
```

Encodes the full `clean/corpus.txt` with the trained BPE tokenizer and writes:
- `data/train.bin` — 90 % of all tokens, stored as `uint16` (2 bytes per token)
- `data/val.bin`   — remaining 10 %
- `data/meta.json` — vocab size, token counts, tokenizer path

Tokens are stored as raw `uint16` arrays with no headers.  Reload with:

```python
import numpy as np
data = np.memmap("data/train.bin", dtype="uint16", mode="r")
```

## Step 6 – GPT model architecture

```bash
python src/model.py
```

Defines a decoder-only Transformer language model (~5 M parameters):

| Hyperparameter | Value |
|---|---|
| `vocab_size` | 8 000 |
| `block_size` | 256 tokens |
| `n_layer` | 4 |
| `n_head` | 4 |
| `n_embd` | 256 |
| `dropout` | 0.1 |

Running the file directly executes a sanity check: forward pass, loss
verification (expected ≈ `ln(8000) ≈ 8.99` for random weights), and
a 20-token generation sample.

---

## Design decisions

**Why BPE?**
BPE (Byte Pair Encoding) builds a vocabulary by iteratively merging the most
frequent character pairs. For Vietnamese it strikes a good balance: common
syllables and morphemes become single tokens while rare or foreign words are
split into smaller pieces. It is the algorithm used by GPT-2, RoBERTa, and
most modern LLMs.

**Why vocab_size = 8000?**
Vietnamese uses a tonal, syllabic writing system. Common syllables (~7 000)
cover the vast majority of everyday text. A vocab of 8 000 gives good
coverage without making the embedding matrix too large for a small model.
For comparison, GPT-2 uses 50 257 and BERT uses 30 000 — we are deliberately
small to match the Tiny GPT training goal.

**Why keep Vietnamese diacritics (no lowercase, no accent removal)?**
Vietnamese diacritics carry **lexical and tonal meaning**. Removing them
makes words ambiguous: `ma` (ghost), `má` (cheek), `mà` (but), `mả` (tomb),
`mã` (code), `mạ` (rice seedling) are six completely different words.
Lowercasing has the same problem for the tonal marks. Any tokenizer that
strips diacritics would produce a model that cannot distinguish these words.

---

## Step 7 – Train the model

```bash
python src/train.py
# override defaults
python src/train.py --max_iters 5000 --batch_size 32 --device mps
```

Trains the GPT model on `data/train.bin`, evaluates on `data/val.bin` every
500 steps, and saves the best checkpoint to `out/ckpt.pt`.

| Setting | Default | Notes |
|---|---|---|
| `max_iters` | 5 000 | increase for better quality |
| `batch_size` | 32 | reduce if GPU OOM |
| `learning_rate` | 3e-4 | cosine decay to 3e-5 |
| `warmup_iters` | 100 | linear LR ramp-up |
| `block_size` | 256 | tokens per sequence |
| `grad_clip` | 1.0 | gradient norm clipping |

Expected loss progression (random init → trained):
- Step 0 (random weights): ~9.0 ≈ ln(8 000)
- Step 1 000: ~4–5
- Step 5 000: ~3–4

Training is resumable — re-run the same command to continue from `out/ckpt.pt`.

## Next step — inference

With a checkpoint in `out/ckpt.pt`, run `src/sample.py` to generate text.
