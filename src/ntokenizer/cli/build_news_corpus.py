"""
CLI: clean locally downloaded Vietnamese news dumps into a training corpus.

Input : one or more news files (.jsonl, .jsonl.gz, .json, .parquet, .txt),
        e.g. local copies of BKAINewsCorpus, binhvq mirrors, VietNews
Output: data/interim/news_corpus.txt  (one paragraph per line, blank line
        between articles)

News text is for local training only; keep it under data/ (gitignored).

Example:
    python scripts/build_news_corpus.py --input data/raw/news/*.parquet --limit 2000000
"""

import argparse
import sys
import time
from pathlib import Path

from ntokenizer.news import write_news_corpus
from ntokenizer.paths import INTERIM_DIR

DEFAULT_OUTPUT = INTERIM_DIR / "news_corpus.txt"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clean news dumps into a training corpus")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="News files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output corpus (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--text-field", default=None,
                        help="Record field with the article body (default: auto-detect "
                             "content/text/body/article)")
    parser.add_argument("--title-field", default=None,
                        help="Record field with the title (default: auto-detect title/headline)")
    parser.add_argument("--min-line-len", type=int, default=10,
                        help="Drop lines shorter than this (captions, bylines)")
    parser.add_argument("--min-chars", type=int, default=200,
                        help="Drop articles whose body is shorter than this")
    parser.add_argument("--no-dedup", action="store_true", help="Keep exact duplicate articles")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N articles written")
    args = parser.parse_args(argv)

    missing = [p for p in args.input if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: input not found: {p}", file=sys.stderr)
        sys.exit(1)

    t0 = time.monotonic()
    stats = write_news_corpus(
        args.input, args.output,
        text_field=args.text_field, title_field=args.title_field,
        min_line_len=args.min_line_len, min_chars=args.min_chars,
        dedup=not args.no_dedup, limit=args.limit,
    )
    print("=" * 50)
    print("  Build News Corpus Complete")
    print("=" * 50)
    print(f"  Articles seen        : {stats['articles_seen']:>12,}")
    print(f"  Too short            : {stats['too_short']:>12,}")
    print(f"  Duplicates removed   : {stats['duplicates']:>12,}")
    print(f"  Articles written     : {stats['articles_written']:>12,}")
    print(f"  Lines written        : {stats['lines_written']:>12,}")
    print(f"  Output               : {args.output}  ({stats['output_bytes'] / 1024**2:.1f} MB)")
    print(f"  Time                 : {time.monotonic() - t0:.1f}s")
    if stats["articles_written"] == 0:
        print("WARNING: nothing written — check --text-field.", file=sys.stderr)


if __name__ == "__main__":
    main()
