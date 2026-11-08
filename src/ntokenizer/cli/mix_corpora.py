"""
CLI: interleave several corpora (e.g. Wikipedia + news) into one training file.

prepare_dataset.py uses the last 10% of tokens as validation, so simply
concatenating files would make validation all-news. This mixes blocks at
random positions over the whole file instead.

Example:
    python scripts/mix_corpora.py \\
        --input data/interim/corpus.txt data/interim/news_corpus.txt \\
        --repeat 2 1 --output data/interim/corpus_mix.txt
"""

import argparse
import sys
from pathlib import Path

from ntokenizer.news import mix_corpora
from ntokenizer.paths import INTERIM_DIR

DEFAULT_OUTPUT = INTERIM_DIR / "corpus_mix.txt"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Interleave corpora into one training file")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Corpus files")
    parser.add_argument("--repeat", type=int, nargs="+", default=None,
                        help="Times to read each input (e.g. 2 1 upsamples the first); default 1 each")
    parser.add_argument("--chunk-lines", type=int, default=50,
                        help="Block size for files without blank-line document separators")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output corpus (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args(argv)

    missing = [p for p in args.input if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: input not found: {p}", file=sys.stderr)
        sys.exit(1)
    if args.repeat is not None and len(args.repeat) != len(args.input):
        sys.exit("ERROR: --repeat needs one number per --input")
    if args.output.resolve() in {p.resolve() for p in args.input}:
        sys.exit("ERROR: --output must differ from every --input")

    written = mix_corpora(args.input, args.output, args.repeat, args.chunk_lines, args.seed)
    total = sum(written.values())
    print("Blocks written per source:")
    for path, n in written.items():
        print(f"  {path}: {n:,} ({n / total:.0%})")
    print(f"Output: {args.output}  ({args.output.stat().st_size / 1024**2:.1f} MB)")
    print("Next: train the tokenizer on it (scripts/train_tokenizer_spm.py --input ...) "
          "and run scripts/prepare_dataset.py --corpus ...")


if __name__ == "__main__":
    main()
