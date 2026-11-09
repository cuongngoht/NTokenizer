# Hardware — what machine you need to train

*[Bản tiếng Việt](HARDWARE.vi.md)*

There are three levels, depending on the goal. **Level 1 runs on any normal
laptop** and is the recommended starting point.

| Level | Goal | Machine | Time |
|---|---|---|---|
| **1 — Learn & experiment** | Default model (4 layers, 256 dim, ~5–19M params), 5,000 steps ≈ 41M tokens, Vietnamese Wikipedia | Any laptop with 8 GB RAM and ~6 GB free disk; Apple M-series (MPS) or any GPU is faster | ~2.5–3 h on a 4-core CPU (measured below); less on MPS/GPU |
| 2 — ~100M model on Wikipedia | ~2.5B tokens (≈ 20 tokens/param) | One GPU with 12–24 GB (RTX 3060 12 GB, 4070, 3090, 4090) | Not feasible on CPU (~60 days at the measured 480 tokens/s); about half a day or more on a consumer GPU (unmeasured) |
| 3 — Wikipedia + news | ≥ 10B tokens | One 24 GB GPU or a rented A100/H100 | Many hours to days (unmeasured for this repo) |

## Measurements

Measured on a 4-core cloud CPU with 15 GB RAM (2026-11-09), using this repo's
code (fp32, no `torch.compile`):

| What | Result |
|---|---|
| Training speed, default model (5.3M params with the 8k tokenizer) | ~6,250 tokens/s in a micro-benchmark; ~1.7–2.0 s per step (batch 32 × 256 tokens) in `train.py`, including evaluation |
| Training speed, 12 layers / 768 dim / 32k vocab (100M params) | ~480 tokens/s |
| `prepare_dataset.py` on a 200 MB corpus (38.6M tokens) | 603 MB peak RAM, 10 s (streaming encoder) |
| End-to-end level-1 smoke run (synthetic corpus, own tokenizer trained first) | tokenizer on 200 MB / 937K lines in 6 s; 200 training steps in 5.7 min, val loss 9.02 → 4.31; `sample.py` generates text |

At ~1.8 s per step, the default 5,000 steps take about **2.5 hours** on such a
CPU. Apple M-series machines run the same code on MPS, which is reported to be
about 2–3× faster than CPU for nanoGPT-style models (not measured here).

## Level 1 — step by step

Requirements: Python ≥ 3.10, 8 GB RAM, ~6 GB free disk (the Wikipedia dump is
~1.1 GB, the extracted text and corpus ~3 GB, the token files a few hundred MB).

```bash
# 0. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,wikiextract]"

# 1. Download and extract Vietnamese Wikipedia (JSON lines)
wget https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2 -P data/raw/
wikiextractor data/raw/viwiki-latest-pages-articles.xml.bz2 \
              --output data/interim/extracted/ --json --quiet
python scripts/build_corpus.py

# 2. Train the tokenizer ON YOUR TEXT first: it must learn from the corpus the
#    model will train on, and every later step must use this same tokenizer.
#    8k vocab keeps the model small and fast for level 1. --name avoids
#    overwriting the git-tracked viwiki_bpe_8k.model shipped with the repo.
python scripts/train_tokenizer_spm.py --vocab_size 8000 --name level1_bpe
#    On an 8 GB machine, cap the sentences SentencePiece holds in RAM:
#    ... --max_sentences 3000000
#    Check it: python scripts/count_tokens.py "Hà Nội là thủ đô của Việt Nam." \
#        --model artifacts/tokenizer/level1_bpe_8k.model --show-pieces

# 3. Encode (streaming, < 1 GB RAM)
python scripts/prepare_dataset.py --model artifacts/tokenizer/level1_bpe_8k.model

# 4. Train the default model (auto-detects CUDA, MPS or CPU)
python scripts/train.py --out_dir artifacts/checkpoints/level1

# 5. Generate text
python scripts/sample.py --ckpt artifacts/checkpoints/level1/ckpt.pt \
    --tokenizer artifacts/tokenizer/level1_bpe_8k.model --prompt "Hà Nội là"
```

Tips for level 1:

- **Try the pipeline first** with a short run: `--max_iters 200 --eval_interval 100`
  takes a few minutes and should show the validation loss dropping.
- **Training is resumable**: re-running `train.py` with the same `--out_dir`
  continues from `ckpt.pt`.
- **Short of RAM during training**: lower `--batch_size` (e.g. 16).
- Expect fluent-looking but often factually wrong Vietnamese at this size;
  that is normal (see [why_llms_are_accurate.md](why_llms_are_accurate.md)).
- Level 1 is enough for the interpretability experiments planned in the
  epics (logit lens, entropy, grokking, frequency lab).
