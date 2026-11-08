"""
CLI: train a Vietnamese SentencePiece BPE tokenizer from one or more corpora.

Input : data/interim/corpus.txt by default (one paragraph per line, UTF-8);
        pass several files to train on a mix (e.g. Wikipedia + FineWeb-2).
Output: artifacts/tokenizer/{name}_{N}k.model
        artifacts/tokenizer/{name}_{N}k.vocab

A tokenizer should see as much of the training data as memory allows: by
default every line of every input is used (--max_sentences 0). SentencePiece
holds the sampled sentences in RAM, so on small machines cap it with
--max_sentences; the script reports what fraction of the corpus was used.
"""

import argparse
import sys
import time
from pathlib import Path

import sentencepiece as spm

from ntokenizer.paths import CORPUS_PATH, TOKENIZER_DIR
from ntokenizer.tokenizer import estimate_vocab_size

# SentencePiece silently skips lines longer than this many bytes (its default
# is 4192). Wikipedia paragraphs are often longer, so raise the limit.
DEFAULT_MAX_SENTENCE_LENGTH = 16_384

# Above this many sampled lines, enable SentencePiece's large-corpus mode.
LARGE_CORPUS_LINES = 10_000_000


def count_lines(paths: list[Path]) -> int:
    """Count non-empty lines across all input files (streamed, constant memory)."""
    total = 0
    for path in paths:
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    total += 1
    return total


def build_train_params(
    vocab_size: int,
    max_sentences: int,
    max_sentence_length: int,
    corpus_lines: int,
) -> dict:
    """
    SentencePiece trainer options. max_sentences=0 means "use every line";
    large-corpus mode is switched on automatically when many lines are used.
    """
    used_lines = corpus_lines if max_sentences == 0 else min(max_sentences, corpus_lines)
    return dict(
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        input_sentence_size=max_sentences,
        shuffle_input_sentence=True,
        max_sentence_length=max_sentence_length,
        train_extremely_large_corpus=used_lines > LARGE_CORPUS_LINES,
        hard_vocab_limit=False,
        unk_id=0,  unk_piece="<unk>",
        pad_id=1,  pad_piece="<pad>",
        bos_id=2,  bos_piece="<bos>",
        eos_id=3,  eos_piece="<eos>",
    )


def train(
    input_paths: list[Path],
    model_prefix: str,
    vocab_size: int,
    max_sentences: int = 0,
    max_sentence_length: int = DEFAULT_MAX_SENTENCE_LENGTH,
) -> None:
    corpus_lines = count_lines(input_paths)
    train_params = build_train_params(vocab_size, max_sentences, max_sentence_length, corpus_lines)
    used_lines = corpus_lines if max_sentences == 0 else min(max_sentences, corpus_lines)
    fraction = used_lines / corpus_lines if corpus_lines else 0.0
    total_mb = sum(p.stat().st_size for p in input_paths) / (1024 ** 2)

    print("=" * 55)
    print("  SentencePiece BPE Tokenizer Training")
    print("=" * 55)
    for p in input_paths:
        print(f"  Input file   : {p}  ({p.stat().st_size / (1024**2):.1f} MB)")
    print(f"  Total size   : {total_mb:.1f} MB, {corpus_lines:,} lines")
    print(f"  Lines used   : {used_lines:,} ({fraction:.0%} of the corpus)")
    print(f"  Model prefix : {model_prefix}")
    print(f"  vocab_size   : {vocab_size}")
    print(f"  max_line_len : {max_sentence_length:,} bytes")
    print(f"  large corpus : {train_params['train_extremely_large_corpus']}")
    if fraction < 0.5:
        print(f"  WARNING: the tokenizer sees only {fraction:.0%} of the corpus; "
              "pass --max_sentences 0 to use all of it if you have the RAM.")
    print()

    t0 = time.monotonic()
    inputs = ",".join(str(p) for p in input_paths)

    try:
        spm.SentencePieceTrainer.train(
            input=inputs,
            model_prefix=model_prefix,
            byte_fallback=True,
            **train_params,
        )
    except TypeError:
        # byte_fallback not supported by this SPM version — train without it
        print("WARNING: byte_fallback not supported, training without it.", flush=True)
        spm.SentencePieceTrainer.train(
            input=inputs,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Vietnamese SentencePiece BPE tokenizer")
    parser.add_argument("--vocab_size", type=int, default=None,
                        help="Vocabulary size (default: auto-picked from total corpus size)")
    parser.add_argument("--input", type=str, nargs="+", default=[str(CORPUS_PATH)],
                        help="One or more corpus text files (one paragraph per line)")
    parser.add_argument("--output_dir", type=str, default=str(TOKENIZER_DIR),
                        help="Directory to save model files")
    parser.add_argument("--name", type=str, default="viwiki_bpe",
                        help="Model file name prefix (default: viwiki_bpe)")
    parser.add_argument("--max_sentences", type=int, default=0,
                        help="Max lines sampled for training; 0 = use every line (default)")
    parser.add_argument("--max_sentence_length", type=int, default=DEFAULT_MAX_SENTENCE_LENGTH,
                        help="Lines longer than this many bytes are skipped by SentencePiece")
    args = parser.parse_args(argv)

    input_paths = [Path(p) for p in args.input]
    output_dir = Path(args.output_dir)

    missing = [p for p in input_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: input file not found: {p}", file=sys.stderr)
        print("Run scripts/build_corpus.py first to generate the corpus.", file=sys.stderr)
        sys.exit(1)

    vocab_size = args.vocab_size
    if vocab_size is None:
        vocab_size = estimate_vocab_size(input_paths)
        size_mb = sum(p.stat().st_size for p in input_paths) / (1024 ** 2)
        print(f"--vocab_size not given: picked {vocab_size} for a {size_mb:.0f} MB corpus "
              f"(pass --vocab_size to override).\n")

    model_prefix = str(output_dir / f"{args.name}_{vocab_size // 1000}k")

    output_dir.mkdir(parents=True, exist_ok=True)

    train(input_paths, model_prefix, vocab_size,
          max_sentences=args.max_sentences,
          max_sentence_length=args.max_sentence_length)

    print("\nOutput files:")
    verify_output(model_prefix)
    print("\nDone. Run scripts/test_tokenizer.py to verify the model, and")
    print("scripts/count_tokens.py \"<prompt>\" to count the tokens of a prompt.")


if __name__ == "__main__":
    main()
