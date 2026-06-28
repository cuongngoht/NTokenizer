# NTokenizer — Vietnamese SentencePiece BPE Tokenizer

Train a Vietnamese subword tokenizer from Vietnamese Wikipedia, as a first step toward training a small Vietnamese language model (Tiny GPT).

## Project structure

```
NTokenizer/
  raw/                        # Raw Wikipedia XML dump (.xml.bz2)
  extracted/                  # wikiextractor output (JSON Lines, ignored by git)
  clean/
    corpus.txt                # Cleaned plain-text corpus (ignored by git)
  tokenizer/
    viwiki_bpe_8k.model       # Trained SentencePiece model
    viwiki_bpe_8k.vocab       # Vocabulary file (token + log-prob score)
  scripts/
    build_corpus.py           # Step 1 – extract + clean Wikipedia text
    train_tokenizer_spm.py    # Step 2 – train BPE tokenizer
    test_tokenizer.py         # Step 3 – verify encode/decode
    inspect_vocab.py          # Step 4 – explore the vocabulary
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

## Next step — tokenize for Tiny GPT

Once the tokenizer is trained, encode the full corpus into integer arrays and
split into `train.bin` / `val.bin`:

```python
import numpy as np
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("tokenizer/viwiki_bpe_8k.model")

ids = []
with open("clean/corpus.txt") as f:
    for line in f:
        ids.extend(sp.encode(line.strip()))

ids = np.array(ids, dtype=np.uint16)
n = int(len(ids) * 0.9)
ids[:n].tofile("data/train.bin")
ids[n:].tofile("data/val.bin")
```

The resulting `.bin` files can be fed directly into a nanoGPT-style training
loop using `np.frombuffer(open("data/train.bin","rb").read(), dtype=np.uint16)`.
