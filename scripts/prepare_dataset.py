"""
Encode the clean corpus into token IDs and save as binary training data.

Input : clean/corpus.txt              (one paragraph per line, UTF-8)
         tokenizer/viwiki_bpe_8k.model (SentencePiece BPE model, vocab=8000)
Output: data/train.bin                (90% of tokens, numpy uint16)
        data/val.bin                  (10% of tokens, numpy uint16)
        data/meta.json                (vocab size, token counts, paths)

How the binary format works
---------------------------
Each token ID is stored as a 2-byte unsigned integer (uint16).
To reload: np.memmap("train.bin", dtype="uint16", mode="r")
The first index in the flat array is the first token of the corpus.
There are no separators between lines — the EOS token serves as the
sentence boundary if the tokenizer supports it.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
CORPUS = ROOT / "clean" / "corpus.txt"
MODEL = ROOT / "tokenizer" / "viwiki_bpe_8k.model"
OUT_DIR = ROOT / "data"

# 90% train, 10% validation — standard split for language model pre-training
TRAIN_SPLIT = 0.9

# uint16 can hold values 0–65535; vocab IDs must fit within this range
UINT16_MAX = 65535

# ---------------------------------------------------------------------------
# Try to import tqdm for a nice progress bar; fall back to simple counter
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# Step 1: Load the SentencePiece tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(model_path: Path) -> tuple:
    """
    Load SentencePiece model and return (sp, vocab_size, eos_id).

    eos_id is -1 if the model has no EOS token, in which case no boundary
    token is appended between lines.
    """
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = sp.get_piece_size()

    # uint16 tops out at 65535 — raise early rather than silently corrupt data
    if vocab_size > UINT16_MAX:
        print(
            f"ERROR: vocab_size={vocab_size} exceeds uint16 max ({UINT16_MAX}).",
            file=sys.stderr,
        )
        print(
            "       Change the dtype to uint32 before encoding a larger vocabulary.",
            file=sys.stderr,
        )
        sys.exit(1)

    # sp.eos_id() returns -1 when no EOS piece was configured during training
    eos_id = sp.eos_id()

    return sp, vocab_size, eos_id


# ---------------------------------------------------------------------------
# Step 2: Encode the entire corpus
# ---------------------------------------------------------------------------

def encode_corpus(corpus_path: Path, sp, eos_id: int) -> list:
    """
    Read corpus line-by-line, encode each line to token IDs, and
    concatenate everything into one flat list.

    Empty lines are skipped.  If eos_id >= 0, it is appended after
    each non-empty line so the model can learn sentence boundaries.
    """
    all_ids: list[int] = []
    n_lines = 0       # total non-empty lines processed
    n_skipped = 0     # blank lines ignored

    print("  Encoding corpus …")
    t0 = time.monotonic()

    with open(corpus_path, encoding="utf-8") as fh:
        # Wrap with tqdm if available for a real-time progress bar
        if _HAS_TQDM:
            lines = tqdm(fh, desc="  Lines", unit=" lines", dynamic_ncols=True)
        else:
            lines = fh

        for raw_line in lines:
            line = raw_line.strip()

            # Skip blank lines — they carry no linguistic content
            if not line:
                n_skipped += 1
                continue

            ids = sp.encode(line, out_type=int)

            # Append EOS so the model sees a clear sentence boundary
            if eos_id >= 0:
                ids.append(eos_id)

            all_ids.extend(ids)
            n_lines += 1

            # Periodic progress report when tqdm is not available
            if not _HAS_TQDM and n_lines % 100_000 == 0:
                elapsed = time.monotonic() - t0
                print(
                    f"    {n_lines:,} lines  |  {len(all_ids):,} tokens  |  {elapsed:.0f}s",
                    flush=True,
                )

    elapsed = time.monotonic() - t0
    print(f"\n  Encoding complete in {elapsed:.1f}s")
    print(f"  Lines processed : {n_lines:,}")
    print(f"  Lines skipped   : {n_skipped:,}")
    print(f"  Total tokens    : {len(all_ids):,}")

    return all_ids


# ---------------------------------------------------------------------------
# Step 3: Split and save as binary numpy arrays
# ---------------------------------------------------------------------------

def save_bins(
    all_ids: list, out_dir: Path, train_split: float
) -> tuple[int, int]:
    """
    Convert token ID list to uint16, split into train/val, and write .bin files.

    Returns (n_train, n_val) token counts.
    """
    arr = np.array(all_ids, dtype=np.uint16)

    split_idx = int(len(arr) * train_split)
    train_arr = arr[:split_idx]
    val_arr = arr[split_idx:]

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    print(f"\n  Saved {train_path}  ({train_path.stat().st_size / 1024**2:.1f} MB)")
    print(f"  Saved {val_path}   ({val_path.stat().st_size / 1024**2:.1f} MB)")

    return len(train_arr), len(val_arr)


# ---------------------------------------------------------------------------
# Step 4: Write metadata JSON
# ---------------------------------------------------------------------------

def save_meta(
    out_dir: Path,
    vocab_size: int,
    n_train: int,
    n_val: int,
    model_path: Path,
) -> None:
    """
    Write a small JSON file that records everything a training script needs
    to know about this dataset without reading the binary files.
    """
    meta = {
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "total_tokens": n_train + n_val,
        "train_tokens": n_train,
        "val_tokens": n_val,
        # Store a relative path so the project stays portable
        "tokenizer": str(model_path.relative_to(ROOT)),
    }

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(f"\n  Saved {meta_path}")


# ---------------------------------------------------------------------------
# Step 5: Verify the output by reading train.bin back
# ---------------------------------------------------------------------------

def verify_output(out_dir: Path, sp) -> None:
    """
    Read train.bin with np.memmap and decode a small prefix to confirm
    the data round-trips correctly.
    """
    train_path = out_dir / "train.bin"
    data = np.memmap(train_path, dtype="uint16", mode="r")

    first_20 = data[:20].tolist()
    decoded_100 = sp.decode(data[:100].tolist())

    print("\n  --- Verification (reading train.bin via np.memmap) ---")
    print(f"  First 20 token IDs : {first_20}")
    print(f"  Decoded 100 tokens :")
    print(f"    {decoded_100}")
    print("  ---")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Guard: both input files must exist before we start
    # ------------------------------------------------------------------
    for path, hint in [
        (CORPUS, "Run scripts/build_corpus.py first."),
        (MODEL, "Run scripts/train_tokenizer_spm.py first."),
    ]:
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            print(f"       {hint}", file=sys.stderr)
            sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print("=" * 55)
    print("  Dataset Preparation")
    print("=" * 55)
    print(f"  Corpus        : {CORPUS}")
    print(f"  Corpus size   : {CORPUS.stat().st_size / 1024**2:.0f} MB")
    print(f"  Tokenizer     : {MODEL}")
    print(f"  Output dir    : {OUT_DIR}")
    print(f"  Train split   : {TRAIN_SPLIT:.0%} train / {1 - TRAIN_SPLIT:.0%} val")
    print()

    # ------------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------------
    print("Step 1 — Load tokenizer")
    sp, vocab_size, eos_id = load_tokenizer(MODEL)
    print(f"  vocab_size    : {vocab_size}")
    print(f"  eos_id        : {eos_id}  {'(will append after each line)' if eos_id >= 0 else '(no EOS — skipping)'}")
    print(f"  dtype         : uint16  (max token id fits in 0–{UINT16_MAX})")
    print()

    # ------------------------------------------------------------------
    # 2. Encode the full corpus
    # ------------------------------------------------------------------
    print("Step 2 — Encode corpus")
    t_encode = time.monotonic()
    all_ids = encode_corpus(CORPUS, sp, eos_id)
    print(f"  Encoding wall time: {time.monotonic() - t_encode:.1f}s")
    print()

    # ------------------------------------------------------------------
    # 3. Split and save binary files
    # ------------------------------------------------------------------
    print("Step 3 — Save train.bin / val.bin")
    n_train, n_val = save_bins(all_ids, OUT_DIR, TRAIN_SPLIT)
    print(f"  train tokens  : {n_train:,}")
    print(f"  val tokens    : {n_val:,}")
    print()

    # ------------------------------------------------------------------
    # 4. Save metadata
    # ------------------------------------------------------------------
    print("Step 4 — Save meta.json")
    save_meta(OUT_DIR, vocab_size, n_train, n_val, MODEL)
    print()

    # ------------------------------------------------------------------
    # 5. Verify
    # ------------------------------------------------------------------
    print("Step 5 — Verify output")
    verify_output(OUT_DIR, sp)
    print()

    print("=" * 55)
    print("  Done.  Run scripts/train.py to start LLM training.")
    print("=" * 55)


if __name__ == "__main__":
    main()
