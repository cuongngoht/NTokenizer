"""
CLI: encode the clean corpus into token IDs and save as binary training data.

Input : data/interim/corpus.txt              (one paragraph per line, UTF-8)
        artifacts/tokenizer/viwiki_bpe_32k.model  (SentencePiece BPE model)
Output: data/processed/train.bin              (90% of tokens, numpy uint16)
        data/processed/val.bin                (10% of tokens, numpy uint16)
        data/processed/meta.json              (vocab size, token counts, paths)
"""

import argparse
import sys
import time
from pathlib import Path

from ntokenizer.dataset import (
    TRAIN_SPLIT,
    encode_corpus,
    save_bins,
    save_meta,
    verify_dataset,
)
from ntokenizer.paths import CORPUS_PATH, DEFAULT_TOKENIZER_MODEL, PROCESSED_DIR
from ntokenizer.tokenizer import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode corpus into binary training data")
    parser.add_argument("--model", type=str,
                        default=str(DEFAULT_TOKENIZER_MODEL),
                        help=f"Path to SentencePiece .model file (default: {DEFAULT_TOKENIZER_MODEL})")
    args = parser.parse_args()

    MODEL = Path(args.model)

    # ------------------------------------------------------------------
    # Guard: both input files must exist before we start
    # ------------------------------------------------------------------
    for path, hint in [
        (CORPUS_PATH, "Run scripts/build_corpus.py first."),
        (MODEL, "Run scripts/train_tokenizer_spm.py first."),
    ]:
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            print(f"       {hint}", file=sys.stderr)
            sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print("=" * 55)
    print("  Dataset Preparation")
    print("=" * 55)
    print(f"  Corpus        : {CORPUS_PATH}")
    print(f"  Corpus size   : {CORPUS_PATH.stat().st_size / 1024**2:.0f} MB")
    print(f"  Tokenizer     : {MODEL}")
    print(f"  Output dir    : {PROCESSED_DIR}")
    print(f"  Train split   : {TRAIN_SPLIT:.0%} train / {1 - TRAIN_SPLIT:.0%} val")
    print()

    # ------------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------------
    print("Step 1 — Load tokenizer")
    sp, vocab_size, eos_id = load_tokenizer(MODEL)
    print(f"  vocab_size    : {vocab_size}")
    print(f"  eos_id        : {eos_id}  {'(will append after each line)' if eos_id >= 0 else '(no EOS — skipping)'}")
    print(f"  dtype         : uint16")
    print()

    # ------------------------------------------------------------------
    # 2. Encode the full corpus
    # ------------------------------------------------------------------
    print("Step 2 — Encode corpus")
    t_encode = time.monotonic()
    all_ids = encode_corpus(CORPUS_PATH, sp, eos_id)
    print(f"  Encoding wall time: {time.monotonic() - t_encode:.1f}s")
    print()

    # ------------------------------------------------------------------
    # 3. Split and save binary files
    # ------------------------------------------------------------------
    print("Step 3 — Save train.bin / val.bin")
    n_train, n_val = save_bins(all_ids, PROCESSED_DIR, TRAIN_SPLIT)
    print(f"  train tokens  : {n_train:,}")
    print(f"  val tokens    : {n_val:,}")
    print()

    # ------------------------------------------------------------------
    # 4. Save metadata
    # ------------------------------------------------------------------
    print("Step 4 — Save meta.json")
    save_meta(PROCESSED_DIR, vocab_size, n_train, n_val, MODEL)
    print()

    # ------------------------------------------------------------------
    # 5. Verify
    # ------------------------------------------------------------------
    print("Step 5 — Verify output")
    verify_dataset(PROCESSED_DIR, sp)
    print()

    print("=" * 55)
    print("  Done.  Run scripts/train.py to start LLM training.")
    print("=" * 55)


if __name__ == "__main__":
    main()
