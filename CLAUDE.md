# CLAUDE.md

Guidance for Claude Code (and other AI agents) working in this repository.

## Project

**NTokenizer** — a Vietnamese GPT built from scratch: Wikipedia corpus →
SentencePiece BPE tokenizer → binary dataset → GPT v2 (RoPE, GQA, SwiGLU,
RMSNorm) → training loop → sampling (KV cache, top-k/top-p, repetition
penalty). Plain PyTorch, no Hugging Face abstractions.

## Layout

- `src/ntokenizer/` — installable package (`model.py`, `training.py`,
  `tokenizer.py`, `dataset.py`, `corpus.py`, `optim.py`, `config.py`,
  `paths.py`, `device.py`); `cli/` holds one module per pipeline step.
- `scripts/` — thin CLI entry points (primary way to run each step).
- `tests/` — pytest suite.
- `docs/` — guides. English file `X.md` with Vietnamese twin `X.vi.md`.
- `data/` — gitignored raw/interim/processed data.
- `artifacts/` — tokenizer models and checkpoints (mostly gitignored).
- `_bmad/`, `.claude/skills/` — BMAD Method install (v6, modules `core`+`bmm`).
  Do not hand-edit; reinstall/update with `npx bmad-method install`.
- `_bmad-output/` — BMAD artifacts (research runs, brainstorming sessions).

## Commands

```bash
pip install -e ".[dev]"   # install (add ,wikiextract for corpus building)
pytest                    # run tests
python scripts/train.py   # train; see docs/DEVELOPMENT.md for full pipeline
python scripts/sample.py  # generate text from a checkpoint
```

## Conventions

- **Language:** code, comments, commit messages and `CLAUDE.md` are in
  English. User-facing docs are bilingual (`.md` English + `.vi.md`
  Vietnamese) where both exist; BMAD outputs default to Vietnamese unless a
  session asks for English.
- Keep `scripts/*.py` thin; put logic in `src/ntokenizer/`.
- The tokenizer used for training and sampling must match the checkpoint
  (see `docs/FINETUNING.md`).
- Run `pytest` before committing changes to `src/`.

## Research docs

- `docs/why_llms_are_accurate.vi.md` — why next-token predictors answer
  accurately (Vietnamese).
- Deep-research runs live in `_bmad-output/planning-artifacts/research/`.
- Brainstorming sessions live in `_bmad-output/brainstorming/`.
