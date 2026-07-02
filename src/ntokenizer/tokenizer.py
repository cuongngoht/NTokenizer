"""Thin helpers around the SentencePiece tokenizer used throughout the pipeline."""

import sys
from pathlib import Path

import sentencepiece as spm

# uint16 can hold values 0-65535; vocab IDs must fit within this range
# (the binary dataset format in ntokenizer.dataset stores tokens as uint16).
UINT16_MAX = 65535


def load_tokenizer(model_path: Path) -> tuple[spm.SentencePieceProcessor, int, int]:
    """
    Load a SentencePiece model and return (sp, vocab_size, eos_id).

    eos_id is -1 if the model has no EOS token, in which case no boundary
    token should be appended between lines when binarizing a corpus.
    """
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = sp.get_piece_size()

    # uint16 tops out at 65535 — raise early rather than silently corrupt data
    if vocab_size > UINT16_MAX:
        print(
            f"ERROR: vocab_size={vocab_size} exceeds uint16 max ({UINT16_MAX}).",
            file=sys.stderr,
        )
        print(
            "       Change the dtype to uint32 before encoding a larger vocabulary.",
            file=sys.stderr,
        )
        sys.exit(1)

    eos_id = sp.eos_id()

    return sp, vocab_size, eos_id


def encode_prompt(sp: spm.SentencePieceProcessor, text: str) -> list[int]:
    """Encode a text prompt to token IDs. Returns at least [bos_id] if empty."""
    if not text.strip():
        # Empty prompt: start from BOS token (or token 0 as fallback)
        bos = sp.bos_id()
        return [bos if bos >= 0 else 0]
    return sp.encode(text, out_type=int)


# Corpus size (MB) -> vocab_size. Larger corpora support (and benefit from) a
# larger vocabulary; the ceiling stays well under UINT16_MAX since the binary
# dataset format stores token IDs as uint16.
_VOCAB_SIZE_THRESHOLDS = [
    (5, 4_000),
    (50, 8_000),
    (200, 16_000),
    (1024, 32_000),
]
_VOCAB_SIZE_ABOVE_MAX_THRESHOLD = 48_000


def estimate_vocab_size(corpus_path: Path) -> int:
    """
    Pick a BPE vocab size from the corpus file size, so you don't have to
    guess a fixed number regardless of how much data you actually have.
    """
    size_mb = corpus_path.stat().st_size / (1024 ** 2)
    for threshold_mb, vocab_size in _VOCAB_SIZE_THRESHOLDS:
        if size_mb < threshold_mb:
            return vocab_size
    return _VOCAB_SIZE_ABOVE_MAX_THRESHOLD


def load_vocab_tsv(vocab_path: Path) -> list[tuple[str, float]]:
    """Read a tokenizer .vocab file (TSV: token<TAB>score) into memory."""
    tokens = []
    with open(vocab_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            token = parts[0]
            score = float(parts[1]) if len(parts) > 1 else 0.0
            tokens.append((token, score))
    return tokens
