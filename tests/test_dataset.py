import numpy as np
import pytest
import torch

from ntokenizer.dataset import get_batch, save_bins, save_meta
from ntokenizer.tokenizer import UINT16_MAX, load_tokenizer


def test_uint16_packing_roundtrip(tmp_path):
    ids = [0, 1, 2, 12345, 65000, 3, 7, 999]
    save_bins(ids, tmp_path, train_split=1.0)

    data = np.memmap(tmp_path / "train.bin", dtype="uint16", mode="r")
    assert data.tolist() == ids


def test_train_val_split_ratio(tmp_path):
    ids = list(range(100))
    n_train, n_val = save_bins(ids, tmp_path, train_split=0.9)

    assert n_train == 90
    assert n_val == 10
    assert n_train + n_val == len(ids)


def test_vocab_size_exceeds_uint16_raises(tmp_path, monkeypatch):
    class _FakeProcessor:
        def __init__(self, model_file=None):
            pass

        def get_piece_size(self):
            return UINT16_MAX + 1

        def eos_id(self):
            return 3

    monkeypatch.setattr("ntokenizer.tokenizer.spm.SentencePieceProcessor", _FakeProcessor)

    with pytest.raises(SystemExit):
        load_tokenizer(tmp_path / "fake.model")


def test_meta_json_schema(tmp_path):
    root = tmp_path
    model_dir = root / "artifacts" / "tokenizer"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "viwiki_bpe_32k.model"
    model_path.write_bytes(b"")

    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True)

    save_meta(out_dir, vocab_size=32000, n_train=900, n_val=100, model_path=model_path, root=root)

    import json
    meta = json.loads((out_dir / "meta.json").read_text())

    assert set(meta) == {"vocab_size", "dtype", "total_tokens", "train_tokens", "val_tokens", "tokenizer"}
    assert meta["dtype"] == "uint16"
    assert meta["train_tokens"] + meta["val_tokens"] == meta["total_tokens"]
    assert meta["tokenizer"] == "artifacts/tokenizer/viwiki_bpe_32k.model"


def test_get_batch_shapes_and_offset():
    torch.manual_seed(0)
    data = np.arange(1000, dtype=np.uint16)
    block_size, batch_size = 8, 4

    x, y = get_batch(data, block_size, batch_size, torch.device("cpu"))

    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    # targets are the inputs shifted by exactly one position
    assert torch.equal(y[:, :-1], x[:, 1:])
