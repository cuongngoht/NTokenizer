"""
Train a Vietnamese SentencePiece BPE tokenizer from clean corpus.

Input : clean/corpus.txt  (one paragraph per line, UTF-8)
Output: tokenizer/viwiki_bpe_8k.model
        tokenizer/viwiki_bpe_8k.vocab
"""

import sys
import time
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).parent.parent
INPUT = ROOT / "clean" / "corpus.txt"
OUTPUT_DIR = ROOT / "tokenizer"
MODEL_PREFIX = str(OUTPUT_DIR / "viwiki_bpe_8k")

TRAIN_PARAMS = dict(
    vocab_size=8000,
    model_type="bpe",
    character_coverage=0.9995,
    input_sentence_size=1_000_000,
    shuffle_input_sentence=True,
    hard_vocab_limit=False,
    unk_id=0,
    unk_piece="<unk>",
    pad_id=1,
    pad_piece="<pad>",
    bos_id=2,
    bos_piece="<bos>",
    eos_id=3,
    eos_piece="<eos>",
)


def train(input_path: Path, model_prefix: str) -> None:
    print("=" * 55)
    print("  SentencePiece BPE Tokenizer Training")
    print("=" * 55)
    print(f"  Input file   : {input_path}")
    print(f"  Input size   : {input_path.stat().st_size / (1024**2):.1f} MB")
    print(f"  Model prefix : {model_prefix}")
    print(f"  vocab_size   : {TRAIN_PARAMS['vocab_size']}")
    print(f"  model_type   : {TRAIN_PARAMS['model_type']}")
    print(f"  coverage     : {TRAIN_PARAMS['character_coverage']}")
    print(f"  max_sentences: {TRAIN_PARAMS['input_sentence_size']:,}")
    print()

    t0 = time.monotonic()

    try:
        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=model_prefix,
            byte_fallback=True,
            **TRAIN_PARAMS,
        )
    except TypeError:
        # byte_fallback not supported by this SPM version — train without it
        print("WARNING: byte_fallback not supported, training without it.", flush=True)
        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=model_prefix,
            **TRAIN_PARAMS,
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
    if not INPUT.exists():
        print(f"ERROR: input file not found: {INPUT}", file=sys.stderr)
        print("Run scripts/build_corpus.py first to generate the corpus.", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train(INPUT, MODEL_PREFIX)

    print("\nOutput files:")
    verify_output(MODEL_PREFIX)
    print("\nDone. Run scripts/test_tokenizer.py to verify the model.")


if __name__ == "__main__":
    main()
