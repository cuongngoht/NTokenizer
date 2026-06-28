"""
Build a plain-text corpus from wikiextractor JSON Lines output.

Input : extracted/AA–AE/wiki_* (JSON Lines, field "text")
Output: clean/corpus.txt  (UTF-8, one paragraph per line)
"""

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def collect_input_files(extracted_dir: Path) -> list[Path]:
    files = sorted(extracted_dir.glob("*/wiki_*"))
    print(f"Found {len(files)} input files in {extracted_dir}", flush=True)
    return files


# ---------------------------------------------------------------------------
# JSON Lines reader
# ---------------------------------------------------------------------------

def iter_articles(file_path: Path, stats: dict):
    with open(file_path, encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                print(
                    f"WARNING: skipping malformed JSON at {file_path.name}:{lineno}",
                    file=sys.stderr,
                )
                continue
            yield obj.get("id", ""), obj.get("title", ""), obj.get("text", "")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_INLINE_TAG_RE = re.compile(r"<[^>]*?>")
_MULTI_SPACE_RE = re.compile(r"  +")

# Section-header pattern: ≤ 45 chars, ends with '.', ≤ 4 spaces (≤ 5 words)
_SECTION_HEADER_RE = re.compile(r"^.{1,45}\.$")


def normalize_line(line: str) -> str:
    line = line.strip()
    line = line.replace("\xa0", " ")
    line = html.unescape(line)          # decode &lt; &gt; &amp; etc.
    line = _INLINE_TAG_RE.sub("", line)
    line = _MULTI_SPACE_RE.sub(" ", line).strip()
    return line


def is_noise_line(line: str, min_len: int = 10) -> bool:
    stripped = line.strip()

    # Empty
    if not stripped:
        return True

    # Bare HTML tag — literal or entity-encoded (templatestyles, stray <ref>, etc.)
    if stripped.startswith("<") or stripped.startswith("&lt;"):
        return True

    # Wikimedia Commons navigation stub
    if "Wikimedia Commons" in stripped and len(stripped) < 60:
        return True

    # Section header: short phrase ending with period, few words
    if _SECTION_HEADER_RE.match(stripped) and stripped.count(" ") <= 4:
        return True

    # Too short after normalization
    normalized = normalize_line(stripped)
    if len(normalized) < min_len:
        return True

    return False


def clean_text(text: str, min_len: int = 10) -> list[str]:
    result = []
    for line in text.split("\n"):
        if is_noise_line(line, min_len):
            continue
        normalized = normalize_line(line)
        if len(normalized) >= min_len:
            result.append(normalized)
    return result


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _progress_wrap(items):
    if HAS_TQDM:
        return tqdm(items, desc="Processing files", unit="file")
    return items


def _log_progress(file_idx: int, total: int, stats: dict):
    if not HAS_TQDM and (file_idx % 50 == 0 or file_idx == total):
        print(
            f"[{file_idx:4d}/{total}] "
            f"articles: {stats['articles_seen']:,}  "
            f"skipped: {stats['articles_skipped']:,}  "
            f"lines out: {stats['lines_written']:,}",
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main writer
# ---------------------------------------------------------------------------

def write_corpus(
    files: list[Path],
    output_path: Path,
    min_len: int = 10,
    dry_run: bool = False,
) -> dict:
    stats = {
        "articles_seen": 0,
        "articles_skipped": 0,
        "lines_considered": 0,
        "lines_written": 0,
        "json_errors": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ctx = open(output_path, "w", encoding="utf-8", buffering=8 * 1024 * 1024) if not dry_run else None

    try:
        for i, file_path in enumerate(_progress_wrap(files), 1):
            for _art_id, _title, text in iter_articles(file_path, stats):
                stats["articles_seen"] += 1
                if not text.strip():
                    stats["articles_skipped"] += 1
                    continue
                lines = clean_text(text, min_len)
                stats["lines_considered"] += len(text.split("\n"))
                stats["lines_written"] += len(lines)
                if ctx is not None:
                    for line in lines:
                        ctx.write(line + "\n")
            _log_progress(i, len(files), stats)
    finally:
        if ctx is not None:
            ctx.close()

    if not dry_run and output_path.exists():
        stats["output_bytes"] = output_path.stat().st_size
    else:
        stats["output_bytes"] = 0

    return stats


# ---------------------------------------------------------------------------
# Stats printer
# ---------------------------------------------------------------------------

def print_stats(stats: dict, output_path: Path, elapsed: float, dry_run: bool):
    articles_total = stats["articles_seen"]
    skipped = stats["articles_skipped"]
    skip_pct = 100.0 * skipped / articles_total if articles_total else 0.0
    lines_in = stats["lines_considered"]
    lines_out = stats["lines_written"]
    retain_pct = 100.0 * lines_out / lines_in if lines_in else 0.0
    size_mb = stats["output_bytes"] / (1024 * 1024)

    print()
    print("=" * 50)
    print("  Build Corpus Complete" + ("  [DRY RUN]" if dry_run else ""))
    print("=" * 50)
    print(f"  Articles seen        : {articles_total:>12,}")
    print(f"  Articles skipped     : {skipped:>12,}  ({skip_pct:.1f}% empty text)")
    print(f"  Lines considered     : {lines_in:>12,}")
    print(f"  Lines written        : {lines_out:>12,}  ({retain_pct:.1f}% retained)")
    if stats["json_errors"]:
        print(f"  JSON errors          : {stats['json_errors']:>12,}")
    if not dry_run:
        print(f"  Output file          : {output_path}")
        print(f"  Output size          : {size_mb:>11.1f} MB")
    print(f"  Elapsed time         : {elapsed:>11.1f} s")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Merge wikiextractor output into a single tokenizer corpus."
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=root / "extracted",
        help="Directory containing AA–AE subdirs (default: <root>/extracted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "clean" / "corpus.txt",
        help="Output corpus file (default: <root>/clean/corpus.txt)",
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
