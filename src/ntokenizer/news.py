"""
Turn locally downloaded Vietnamese news dumps into a training corpus, and
mix several corpora into one file.

Corpus format (same as data/interim/corpus.txt): UTF-8, one paragraph per
line. Documents are separated by a blank line; blank lines are ignored when
the corpus is encoded (ntokenizer.dataset.encode_corpus) and when the
tokenizer is trained, so the separator only matters for mixing.

Supported news inputs: .jsonl, .jsonl.gz, .json (array), .parquet (needs
pyarrow), and .txt (documents separated by blank lines). Typical sources are
local copies of BKAINewsCorpus, binhvq news-corpus mirrors, or VietNews.
News text is for local training only: it lives under data/ (gitignored) and
must not be committed or redistributed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path

from ntokenizer.corpus import normalize_line

# Photo/video credits and source lines that news pages put between
# paragraphs, e.g. "Ảnh: TTXVN", "Ảnh minh họa: ...", "Nguồn: VTV".
_CREDIT_RE = re.compile(r"^(Ảnh|Video|Clip|Nguồn)(\s+minh\s+h(?:o[ạa]|ọa))?\s*:", re.IGNORECASE)
_CREDIT_MAX_LEN = 80

TEXT_FIELDS = ("content", "text", "body", "article", "news_content", "document")
TITLE_FIELDS = ("title", "headline")


# ---------------------------------------------------------------------------
# Reading news dumps
# ---------------------------------------------------------------------------

def _pick(record: dict, explicit: str | None, candidates: tuple[str, ...]) -> str:
    """Return the first non-empty string field (the explicit one if given)."""
    keys = (explicit,) if explicit else candidates
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _iter_raw_records(path: Path) -> Iterator[dict | str]:
    name = path.name.lower()
    if name.endswith(".jsonl.gz") or name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
    elif name.endswith(".jsonl"):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
    elif name.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        yield from (data if isinstance(data, list) else [data])
    elif name.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit("ERROR: reading .parquet needs pyarrow: pip install pyarrow") from exc
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    elif name.endswith(".txt"):
        yield from iter_blocks(path)
    else:
        raise SystemExit(f"ERROR: unsupported news file type: {path}")


def iter_news_articles(
    paths: list[Path],
    text_field: str | None = None,
    title_field: str | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield (title, text) for every record in the given files, streamed."""
    for path in paths:
        for record in _iter_raw_records(path):
            if isinstance(record, str):
                yield "", record
            elif isinstance(record, dict):
                yield _pick(record, title_field, TITLE_FIELDS), _pick(record, text_field, TEXT_FIELDS)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_article(title: str, text: str, min_line_len: int = 10) -> list[str]:
    """
    Normalize an article into paragraphs: NFC Unicode (one code-point form for
    every Vietnamese letter), HTML entities/tags removed, whitespace collapsed,
    short lines and photo/source credits ("Ảnh: ...", "Nguồn: ...") dropped. The title, if any,
    becomes the first line.
    """
    lines = []
    title = normalize_line(unicodedata.normalize("NFC", title))
    if title:
        lines.append(title)
    for raw in unicodedata.normalize("NFC", text).split("\n"):
        line = normalize_line(raw)
        if len(line) < _CREDIT_MAX_LEN and _CREDIT_RE.match(line):
            continue
        if len(line) >= min_line_len and line != title:
            lines.append(line)
    return lines


def _fingerprint(lines: list[str]) -> int:
    body = " ".join(lines).lower()
    return int.from_bytes(hashlib.blake2b(body.encode("utf-8"), digest_size=8).digest(), "big")


def write_news_corpus(
    paths: list[Path],
    output_path: Path,
    text_field: str | None = None,
    title_field: str | None = None,
    min_line_len: int = 10,
    min_chars: int = 200,
    dedup: bool = True,
    limit: int | None = None,
) -> dict:
    """
    Clean news articles and write them in corpus format. Exact duplicates
    (same normalized text, common with syndicated news) are dropped using an
    8-byte fingerprint per article, so memory grows with the number of
    articles, not their size.
    """
    stats = {"articles_seen": 0, "too_short": 0, "duplicates": 0,
             "articles_written": 0, "lines_written": 0}
    seen: set[int] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", buffering=8 * 1024 * 1024) as out:
        for title, text in iter_news_articles(paths, text_field, title_field):
            if limit is not None and stats["articles_written"] >= limit:
                break
            stats["articles_seen"] += 1
            lines = clean_article(title, text, min_line_len)
            has_title = bool(normalize_line(unicodedata.normalize("NFC", title)))
            body_chars = sum(len(line) for line in lines[1 if has_title else 0:])
            if body_chars < min_chars:
                stats["too_short"] += 1
                continue
            if dedup:
                fp = _fingerprint(lines)
                if fp in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(fp)
            out.write("\n".join(lines) + "\n\n")
            stats["articles_written"] += 1
            stats["lines_written"] += len(lines)

    stats["output_bytes"] = output_path.stat().st_size
    return stats


# ---------------------------------------------------------------------------
# Mixing corpora
# ---------------------------------------------------------------------------

def iter_blocks(path: Path, chunk_lines: int = 50) -> Iterator[str]:
    """
    Yield the documents of a corpus file as text blocks. Documents are
    separated by blank lines; a file without blank lines (e.g. the Wikipedia
    corpus.txt) is cut into blocks of `chunk_lines` lines so neighbouring
    paragraphs stay together.
    """
    block: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                if block:
                    yield "\n".join(block)
                    block = []
                continue
            block.append(line)
            if len(block) >= chunk_lines:
                yield "\n".join(block)
                block = []
    if block:
        yield "\n".join(block)


def count_blocks(path: Path, chunk_lines: int = 50) -> int:
    return sum(1 for _ in iter_blocks(path, chunk_lines))


def mix_corpora(
    paths: list[Path],
    output_path: Path,
    repeats: list[int] | None = None,
    chunk_lines: int = 50,
    seed: int = 0,
) -> dict[str, int]:
    """
    Interleave the blocks of several corpora at random positions across the
    whole output, so the last 10% (the validation split in prepare_dataset)
    contains every source in proportion. Each source is read `repeats[i]`
    times (e.g. 2 to upsample Wikipedia). Memory use is constant: only block
    counts are kept, files are streamed.
    """
    repeats = repeats or [1] * len(paths)
    if len(repeats) != len(paths):
        raise ValueError("repeats must have one entry per input")

    def stream(i: int) -> Iterator[str]:
        for _ in range(repeats[i]):
            yield from iter_blocks(paths[i], chunk_lines)

    remaining = [count_blocks(p, chunk_lines) * r for p, r in zip(paths, repeats)]
    written = {str(p): 0 for p in paths}
    streams = [stream(i) for i in range(len(paths))]
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", buffering=8 * 1024 * 1024) as out:
        while any(remaining):
            # Pick a source with probability proportional to its blocks left,
            # which spreads every source uniformly over the output.
            i = rng.choices(range(len(paths)), weights=remaining)[0]
            out.write(next(streams[i]) + "\n\n")
            remaining[i] -= 1
            written[str(paths[i])] += 1
    return written
