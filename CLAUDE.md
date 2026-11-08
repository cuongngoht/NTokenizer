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

- **Language:** every document is written in 100% English: code, comments,
  commit messages, `CLAUDE.md`, and all BMAD outputs (research reports,
  brainstorms, epics). Only the chat with the user is in Vietnamese
  (`communication_language = Vietnamese`, `document_output_language =
  English` in `_bmad/config.toml`). Guides in `docs/` may keep a Vietnamese
  twin (`X.vi.md`) next to the English original `X.md`.
- Keep `scripts/*.py` thin; put logic in `src/ntokenizer/`.
- The tokenizer used for training and sampling must match the checkpoint
  (see `docs/FINETUNING.md`).
- Run `pytest` before committing changes to `src/`.

## Research docs

- `docs/why_llms_are_accurate.md` (English) and `docs/why_llms_are_accurate.vi.md`
  (Vietnamese twin) — why next-token predictors answer accurately.
- Deep-research runs live in `_bmad-output/planning-artifacts/research/`.
- Brainstorming sessions live in `_bmad-output/brainstorming/`.
