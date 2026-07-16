# Fine-tuning — adapting the model to your own data

*[Bản tiếng Việt](FINETUNING.vi.md)*

*This document describes how to fine-tune an already-pretrained checkpoint
(e.g. `artifacts/checkpoints/gpt_8k_research/ckpt.pt`, trained on Wikipedia)
on a smaller, domain-specific corpus — for example historical text or
question/answer data. See [`DEVELOPMENT.md`](DEVELOPMENT.md) for the
from-scratch pretraining pipeline.*

---

## Table of contents

1. [Overview](#overview)
2. [Hard requirement: the tokenizer must match the checkpoint](#hard-requirement-the-tokenizer-must-match-the-checkpoint)
3. [General pipeline — 3 steps](#general-pipeline--3-steps)
4. [Approach 1 — Fine-tune on plain narrative text](#approach-1--fine-tune-on-plain-narrative-text)
5. [Approach 2 — Fine-tune on condensed Q&A (mixed with the source text)](#approach-2--fine-tune-on-condensed-qa-mixed-with-the-source-text)
6. [Approach 3 — Fine-tune on full Alpaca-block data](#approach-3--fine-tune-on-full-alpaca-block-data)
7. [Important limitation: fixed block_size](#important-limitation-fixed-block_size)
8. [Testing the model after fine-tuning](#testing-the-model-after-fine-tuning)
9. [CLI flag reference](#cli-flag-reference)
10. [Troubleshooting](#troubleshooting)

---

## Overview

Fine-tuning reuses the exact same `scripts/train.py` and
`scripts/prepare_dataset.py` from the pretraining pipeline, with two
additions:

| Component | Change |
|---|---|
| `ntokenizer.config.TrainConfig` | added field `init_from: str = ""` |
| `ntokenizer.training.train()` | if `out_dir` has **no** `ckpt.pt` yet and `init_from` is set → load *model weights only* from that checkpoint (fresh optimizer, step starts at 0) |
| `scripts/train.py` | added flag `--init_from <path/to/ckpt.pt>` |
| `scripts/prepare_dataset.py` | added flags `--corpus` and `--output-dir` to encode any corpus, not just the default `data/interim/corpus.txt` |

If `out_dir` **already has** a `ckpt.pt` (an interrupted fine-tuning run),
`train()` always resumes from it and **ignores** `--init_from` — this
behavior is covered by `tests/test_finetune.py`.

---

## Hard requirement: the tokenizer must match the checkpoint

A checkpoint's embedding table size is baked in at the `vocab_size` of the
tokenizer used during pretraining. For example, `gpt_8k_research/ckpt.pt`
was trained with `artifacts/tokenizer/viwiki_bpe_8k.model`
(`vocab_size=8000`). When fine-tuning, you **must** encode the new corpus
with that same tokenizer — using the `32k` or `48k` model by mistake will
produce a mismatched embedding shape, and `load_state_dict` will error out
as soon as it tries to load the checkpoint.

Quick way to check a checkpoint's vocab_size:

```bash
.venv/bin/python -c "
import torch
ckpt = torch.load('artifacts/checkpoints/gpt_8k_research/ckpt.pt', map_location='cpu', weights_only=True)
print(ckpt['config'])
"
```

---

## General pipeline — 3 steps

```
new corpus (.txt, one paragraph per line)
        │
        ▼  scripts/prepare_dataset.py --corpus ... --output-dir ...
data/processed_xxx/{train.bin, val.bin, meta.json}
        │
        ▼  scripts/train.py --data_dir ... --out_dir ... --init_from <base_ckpt>
artifacts/checkpoints/xxx/ckpt.pt   ← fine-tuned checkpoint
        │
        ▼  scripts/sample.py --ckpt ... --tokenizer ...
"generated text"
```

`--output-dir` and `--out_dir` **must be new directories**, distinct from
the base checkpoint's directory — if you point at an `out_dir` that already
has a `ckpt.pt`, `train()` will resume (continue training it further)
instead of fine-tuning from scratch with `init_from`.

---

## Approach 1 — Fine-tune on plain narrative text

Use this when you only have raw text (no question/answer structure) — for
example a book or a domain-specific document. The model picks up extra
vocabulary and style for that domain, but it's still just a **continuation
LM** — it keeps writing text, it doesn't answer questions directly.

```bash
# 1. Encode the corpus (one paragraph per line, using the matching 8k tokenizer)
.venv/bin/python scripts/prepare_dataset.py \
  --corpus data/raw/history/vn_su_luoc_for_training_clean_v2.txt \
  --output-dir data/processed_history \
  --model artifacts/tokenizer/viwiki_bpe_8k.model

# 2. Fine-tune from the base checkpoint
.venv/bin/python scripts/train.py \
  --data_dir data/processed_history \
  --out_dir artifacts/checkpoints/gpt_8k_history_finetune \
  --init_from artifacts/checkpoints/gpt_8k_research/ckpt.pt \
  --max_iters 500 --batch_size 16 \
  --learning_rate 3e-5 --min_lr 3e-6 --warmup_iters 10 \
  --eval_interval 25 --eval_iters 20 --device cpu
```

**Actual result** (2,223 lines, ~239K tokens, the "Việt Nam Sử Lược" dataset):
val loss dropped from **4.92 → 4.19** after 500 steps. The generated text
uses the right historical vocabulary (Gia Định, quân Pháp, huyện/tỉnh/phủ...)
but doesn't "answer" a direct question.

---

## Approach 2 — Fine-tune on condensed Q&A (mixed with the source text)

Use this when you have data shaped like `{"instruction": ..., "output": ...}`
(dropping the source `input` excerpt if it's already covered by the plain
narrative corpus) — keeping each example short so it fits inside the
`block_size=256` token window teaches the model the right behavior: "see the
cue 'Trả lời:' → stop narrating and answer."

```bash
# 1. Concatenate the narrative corpus + Q&A into one file, one example per line
.venv/bin/python -c "
import json
qa_path = 'data/raw/history/history_qa_all.jsonl'
narrative_path = 'data/raw/history/vn_su_luoc_for_training_clean_v2.txt'
out_path = 'data/interim/history_narrative_qa_corpus.txt'

lines = [l.strip() for l in open(narrative_path, encoding='utf-8') if l.strip()]
with open(qa_path, encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        instr = ' '.join(obj['instruction'].split())
        out = ' '.join(obj['output'].split())
        lines.append(f'{instr} Trả lời: {out}')

open(out_path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print(len(lines), 'lines')
"

# 2. Encode + fine-tune (from the BASE checkpoint, NOT the Approach-1 checkpoint)
.venv/bin/python scripts/prepare_dataset.py \
  --corpus data/interim/history_narrative_qa_corpus.txt \
  --output-dir data/processed_history_qa \
  --model artifacts/tokenizer/viwiki_bpe_8k.model

.venv/bin/python scripts/train.py \
  --data_dir data/processed_history_qa \
  --out_dir artifacts/checkpoints/gpt_8k_history_qa_finetune \
  --init_from artifacts/checkpoints/gpt_8k_research/ckpt.pt \
  --max_iters 1500 --batch_size 16 \
  --learning_rate 3e-5 --min_lr 3e-6 --warmup_iters 30 \
  --eval_interval 100 --eval_iters 30 --device cpu
```

**Actual result** (4,403 lines — 2,223 narrative + 2,180 Q&A, ~640K tokens):
val loss dropped from **5.04 → 2.80** after 1,500 steps — clearly better
than Approach 1, because the condensed format means most
instruction+answer examples fit entirely within a single 256-token window
(measured ~93% of examples ≤ 256 tokens). The model started responding in
the right "question → Trả lời:" structure, occasionally even producing
bullet points matching the `key_points` style seen in training.

> Factual accuracy (which king, which year, who fought whom) is still
> spotty — the model is only 5.2M parameters, not enough capacity to
> memorize precise historical detail. The Q&A format teaches the model the
> *structure* of an answer, it doesn't automatically guarantee correct
> content.

---

## Approach 3 — Fine-tune on full Alpaca-block data

Use this when you already have a standard Alpaca-style instruction-tuning
dataset — each example is a multi-line block:

```
### Instruction:
{instruction/question}

### Input:
{source excerpt}

### Response:
{answer}
```

This format is **not** compatible with `encode_corpus()` (which encodes
line by line, not block by block) — it needs a dedicated encoder that
splits on `"\n\n### Instruction:"` and wraps each block with BOS/EOS. If
your data already ships with a `prepare_instruction_bin.py` script like
this (e.g. bundled with a downloaded Q&A dataset), reuse it directly — just
point `--tokenizer` at the tokenizer matching the base checkpoint, and
write `meta.json` yourself afterward (this kind of script usually doesn't
write one, but `scripts/train.py` requires it):

```bash
# 1. Encode the Alpaca blocks (using the script shipped with the data, fixing --tokenizer to match the checkpoint)
.venv/bin/python data/raw/history/vn_su_luoc_qa_finetune/prepare_instruction_bin.py \
  --tokenizer artifacts/tokenizer/viwiki_bpe_8k.model \
  --train_txt data/raw/history/vn_su_luoc_qa_finetune/history_qa_train_corpus.txt \
  --val_txt data/raw/history/vn_su_luoc_qa_finetune/history_qa_val_corpus.txt \
  --out_dir data/processed_history_qa_v2

# 2. Write meta.json (required by scripts/train.py — the external encoder doesn't write one)
.venv/bin/python scripts/make_meta.py \
  --out_dir data/processed_history_qa_v2 \
  --vocab_size 8000 \
  --tokenizer artifacts/tokenizer/viwiki_bpe_8k.model

# 3. Fine-tune from the base checkpoint
.venv/bin/python scripts/train.py \
  --data_dir data/processed_history_qa_v2 \
  --out_dir artifacts/checkpoints/gpt_8k_history_qa_finetune_v2 \
  --init_from artifacts/checkpoints/gpt_8k_research/ckpt.pt \
  --max_iters 1000 --batch_size 16 \
  --learning_rate 5e-5 --min_lr 5e-6 --warmup_iters 30 \
  --eval_interval 100 --eval_iters 50 --device cpu
```

Because each block includes the full `### Input:` excerpt, most examples'
total token count exceeds 256 — they get truncated across several sampling
windows during training (see below). At test time, a prompt with only
`### Instruction:` + `### Response:` (skipping `### Input:`) diverges from
the training format — the model may tend to generate an `### Input:`
section of its own before answering, instead of answering right away.

---

## Important limitation: fixed block_size

`GPTConfig.block_size` isn't just a training hyperparameter — it determines
the size of the RoPE buffer (`freqs_cis`, see `src/ntokenizer/model.py`),
and that buffer **is saved inside the checkpoint** (via `register_buffer`,
without `persistent=False`). As a result:

- You **cannot** increase `--block_size` when fine-tuning from a pretrained
  checkpoint — `load_state_dict` will raise a shape mismatch on
  `freqs_cis`.
- Every fine-tuning command above keeps `block_size=256` (the pretraining
  value).
- Examples longer than 256 tokens (e.g. Approach 3) still train fine — the
  random sampling window over the concatenated token stream (nanoGPT-style)
  just sees a fragment of the long example, no error, but it learns less
  efficiently than a right-sized example (see the Approach 1/2 vs 3
  comparison above).

---

## Testing the model after fine-tuning

```bash
.venv/bin/python scripts/sample.py \
  --ckpt artifacts/checkpoints/<checkpoint_name>/ckpt.pt \
  --tokenizer artifacts/tokenizer/viwiki_bpe_8k.model \
  --prompt "Gia Long là ai? Trả lời:" \
  --max_new_tokens 80 --device cpu
```

The prompt must match the "template" used by the fine-tuning corpus (e.g.
contain the cue `Trả lời:` or `### Response:`) — otherwise the model falls
back to plain continuation behavior since it doesn't recognize the learned
cue.

The smaller the model (5.2M parameters here) and the fewer fine-tuning
steps/data, the more the output tends to get facts wrong — good enough to
demonstrate the pipeline works, but don't expect the accuracy of a much
larger model.

---

## CLI flag reference

**`scripts/prepare_dataset.py`**

| Flag | Default | Notes |
|---|---|---|
| `--corpus` | `data/interim/corpus.txt` | Source corpus — one paragraph per line |
| `--output-dir` | `data/processed/` | Where to write `train.bin`/`val.bin`/`meta.json` |
| `--model` | `viwiki_bpe_32k.model` | Tokenizer — must match the checkpoint you'll fine-tune |

**`scripts/train.py`** (only the fine-tuning-relevant flags — see the full list in [`DEVELOPMENT.md`](DEVELOPMENT.md#step-7--training-loop))

| Flag | Default | Notes |
|---|---|---|
| `--init_from` | `""` | Pretrained checkpoint to initialize weights from. **Ignored** if `--out_dir` already has a `ckpt.pt` (resume takes priority) |
| `--data_dir` | `data/processed/` | Point at the output directory from the `prepare_dataset.py` step above |
| `--out_dir` | `artifacts/checkpoints/` | Directory to write the fine-tuned checkpoint — **should be different** from the base checkpoint's directory |

**`scripts/make_meta.py`** (only needed when using an external encoder that doesn't write its own `meta.json`)

| Flag | Required | Notes |
|---|---|---|
| `--out_dir` | yes | Directory already containing `train.bin`/`val.bin` |
| `--vocab_size` | yes | Must match the checkpoint you'll fine-tune |
| `--tokenizer` | yes | Path to the tokenizer used to encode the corpus |

---

## Troubleshooting

**`RuntimeError: Error(s) in loading state_dict ... size mismatch`**
The tokenizer used to encode the fine-tuning corpus doesn't match the
checkpoint's `vocab_size`. Double-check using
[Hard requirement: the tokenizer must match the checkpoint](#hard-requirement-the-tokenizer-must-match-the-checkpoint).

**`ERROR: data/processed_xxx/meta.json not found`**
If you used a custom encoder (not this repo's `scripts/prepare_dataset.py`)
— e.g. one bundled with a downloaded dataset — it may not write
`meta.json` itself. Use `scripts/make_meta.py` as shown in
[Approach 3, step 2](#approach-3--fine-tune-on-full-alpaca-block-data).

**Fine-tuning isn't using `init_from`, val loss starts as high as a random init**
`--out_dir` already has a `ckpt.pt` from a previous run — `train()`
prioritizes resuming from it and ignores `--init_from`. Point at a new
`--out_dir`.

**Initial (step 0) val loss is higher than the base checkpoint's final val loss**
This is expected — the base checkpoint was evaluated on the Wikipedia
corpus's val split (a general domain), while fine-tuning evaluates on the
new corpus's val split (a narrower domain) that the base model never saw.
What matters is that this higher starting loss then decreases over the
fine-tuning steps.
