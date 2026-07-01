"""
Binary dataset format shared by the corpus-preparation and training steps.

On-disk format
--------------
Each token ID is stored as a 2-byte unsigned integer (uint16).
To reload: np.memmap("train.bin", dtype="uint16", mode="r")
The first index in the flat array is the first token of the corpus.
There are no separators between lines — the EOS token serves as the
sentence boundary if the tokenizer supports it.
"""

import json
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from ntokenizer.paths import PROJECT_ROOT

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# 90% train, 10% validation — standard split for language model pre-training
TRAIN_SPLIT = 0.9


# ---------------------------------------------------------------------------
# Writing: encode a corpus into binary train/val files
# ---------------------------------------------------------------------------

def encode_corpus(corpus_path: Path, sp: spm.SentencePieceProcessor, eos_id: int) -> list:
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
        lines = tqdm(fh, desc="  Lines", unit=" lines", dynamic_ncols=True) if _HAS_TQDM else fh

        for raw_line in lines:
            line = raw_line.strip()

            if not line:
                n_skipped += 1
                continue

            ids = sp.encode(line, out_type=int)

            if eos_id >= 0:
                ids.append(eos_id)

            all_ids.extend(ids)
            n_lines += 1

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


def save_bins(all_ids: list, out_dir: Path, train_split: float = TRAIN_SPLIT) -> tuple[int, int]:
    """
    Convert token ID list to uint16, split into train/val, and write .bin files.

    Returns (n_train, n_val) token counts.
    """
    arr = np.array(all_ids, dtype=np.uint16)

    split_idx = int(len(arr) * train_split)
    train_arr = arr[:split_idx]
    val_arr = arr[split_idx:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    print(f"\n  Saved {train_path}  ({train_path.stat().st_size / 1024**2:.1f} MB)")
    print(f"  Saved {val_path}   ({val_path.stat().st_size / 1024**2:.1f} MB)")

    return len(train_arr), len(val_arr)


def save_meta(
    out_dir: Path,
    vocab_size: int,
    n_train: int,
    n_val: int,
    model_path: Path,
    root: Path = PROJECT_ROOT,
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
        "tokenizer": str(model_path.resolve().relative_to(root.resolve())),
    }

    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(f"\n  Saved {meta_path}")


def verify_dataset(out_dir: Path, sp: spm.SentencePieceProcessor) -> None:
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
# Reading: memory-map the binary files and sample training batches
# ---------------------------------------------------------------------------

def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Memory-map train.bin and val.bin.

    np.memmap reads directly from disk without loading the whole file into RAM.
    For a ~1 GB train.bin this is critical on a MacBook with limited RAM.
    """
    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data   = np.memmap(data_dir / "val.bin",   dtype=np.uint16, mode="r")
    return train_data, val_data


def get_batch(
    data:       np.ndarray,
    block_size: int,
    batch_size: int,
    device:     torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch of (input, target) pairs from a token array.

    For language modelling the target is the input shifted by 1:
      input  = tokens[i   : i + block_size]
      target = tokens[i+1 : i + block_size + 1]

    At each position t the model predicts token t+1 given tokens 0..t.
    This gives block_size training examples per sequence in the batch.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i     : i + block_size].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64))
        for i in ix
    ])
    return x.to(device), y.to(device)
