"""
Train a Vietnamese SentencePiece BPE tokenizer from clean corpus.

Input : clean/corpus.txt  (one paragraph per line, UTF-8)
Output: tokenizer/viwiki_bpe_8k.model
        tokenizer/viwiki_bpe_8k.vocab
"""

import argparse
import sys
import time
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).parent.parent
INPUT = ROOT / "clean" / "corpus.txt"
OUTPUT_DIR = ROOT / "tokenizer"


def train(input_path: Path, model_prefix: str, vocab_size: int) -> None:
    train_params = dict(
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        input_sentence_size=1_000_000,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
        unk_id=0,  unk_piece="<unk>",
        pad_id=1,  pad_piece="<pad>",
        bos_id=2,  bos_piece="<bos>",
        eos_id=3,  eos_piece="<eos>",
    )

    print("=" * 55)
    print("  SentencePiece BPE Tokenizer Training")
    print("=" * 55)
    print(f"  Input file   : {input_path}")
    print(f"  Input size   : {input_path.stat().st_size / (1024**2):.1f} MB")
    print(f"  Model prefix : {model_prefix}")
    print(f"  vocab_size   : {vocab_size}")
    print(f"  model_type   : bpe")
    print(f"  coverage     : 0.9995")
    print(f"  max_sentences: {train_params['input_sentence_size']:,}")
    print()

    t0 = time.monotonic()

    try:
        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=model_prefix,
            byte_fallback=True,
            **train_params,
        )
    except TypeError:
        # byte_fallback not supported by this SPM version — train without it
        print("WARNING: byte_fallback not supported, training without it.", flush=True)
        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=model_prefix,
            **train_params,
        )

    elapsed = time.monotonic() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")


def verify_output(model_prefix: str) -> None:
    model_file = Path(model_prefix + ".model")
    vocab_file = Path(model_prefix + ".vocab")
    ok = True
    for f in (model_file, vocab_file):
        if f.exists():
            print(f"  [OK] {f}  ({f.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [MISSING] {f}", file=sys.stderr)
            ok = False
    if not ok:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Vietnamese SentencePiece BPE tokenizer")
    parser.add_argument("--vocab_size", type=int, default=8000,
                        help="Vocabulary size (default: 8000)")
    parser.add_argument("--input", type=str, default=str(INPUT),
                        help="Path to corpus text file")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="Directory to save model files")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    model_prefix = str(output_dir / f"viwiki_bpe_{args.vocab_size // 1000}k")

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        print("Run scripts/build_corpus.py first to generate the corpus.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    train(input_path, model_prefix, args.vocab_size)

    print("\nOutput files:")
    verify_output(model_prefix)
    print("\nDone. Run scripts/test_tokenizer.py to verify the model.")


if __name__ == "__main__":
    main()
