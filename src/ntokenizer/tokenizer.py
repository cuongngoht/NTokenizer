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
