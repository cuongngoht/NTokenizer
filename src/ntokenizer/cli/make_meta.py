"""
CLI: write meta.json for a train.bin/val.bin pair produced by an external
encoder that doesn't emit one itself (e.g. an Alpaca-style instruction
corpus encoder from outside this repo).

Usage:
    python scripts/make_meta.py --out_dir data/processed_history_qa_v2 \
        --vocab_size 8000 --tokenizer artifacts/tokenizer/viwiki_bpe_8k.model
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from ntokenizer.dataset import save_meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write meta.json for an existing train.bin/val.bin pair"
    )
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory containing train.bin and val.bin")
    parser.add_argument("--vocab_size", type=int, required=True,
                        help="Tokenizer vocab size (must match the checkpoint you'll fine-tune)")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to the SentencePiece .model used to encode the corpus")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    for name in ("train.bin", "val.bin"):
        if not (out_dir / name).exists():
            print(f"ERROR: {out_dir / name} not found.", file=sys.stderr)
            sys.exit(1)

    n_train = int(np.memmap(out_dir / "train.bin", dtype="uint16", mode="r").size)
    n_val = int(np.memmap(out_dir / "val.bin", dtype="uint16", mode="r").size)

    save_meta(out_dir, args.vocab_size, n_train, n_val, Path(args.tokenizer))
    print(f"  train tokens : {n_train:,}")
    print(f"  val tokens   : {n_val:,}")


if __name__ == "__main__":
    main()
