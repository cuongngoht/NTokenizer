"""
CLI: merge wikiextractor output into a single tokenizer corpus.

Usage:
    python scripts/build_corpus.py
    python scripts/build_corpus.py --extracted-dir data/interim/extracted --output data/interim/corpus.txt
"""

import argparse
import sys
import time
from pathlib import Path

from ntokenizer.corpus import collect_input_files, print_stats, write_corpus
from ntokenizer.paths import CORPUS_PATH, EXTRACTED_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge wikiextractor output into a single tokenizer corpus."
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=EXTRACTED_DIR,
        help=f"Directory containing AA–AE subdirs (default: {EXTRACTED_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CORPUS_PATH,
        help=f"Output corpus file (default: {CORPUS_PATH})",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=10,
        help="Minimum character length for a line to be kept (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count lines without writing output file",
    )
    args = parser.parse_args()

    files = collect_input_files(args.extracted_dir)
    if not files:
        print(f"ERROR: no wiki_* files found under {args.extracted_dir}", file=sys.stderr)
        sys.exit(1)

    t0 = time.monotonic()
    stats = write_corpus(files, args.output, min_len=args.min_len, dry_run=args.dry_run)
    elapsed = time.monotonic() - t0

    print_stats(stats, args.output, elapsed, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
