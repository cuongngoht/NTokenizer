import re

import sentencepiece as spm

from ntokenizer.tokenizer import estimate_vocab_size

_VI_DIACRITIC_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
    r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]"
)


def _load(tiny_sentencepiece_model):
    sp = spm.SentencePieceProcessor()
    sp.load(tiny_sentencepiece_model)
    return sp


def test_encode_decode_roundtrip(tiny_sentencepiece_model):
    sp = _load(tiny_sentencepiece_model)

    for sentence in ["Hà Nội là thủ đô của Việt Nam.", "Tôi muốn học lập trình Python."]:
        ids = sp.encode(sentence, out_type=int)
        decoded = sp.decode(ids)
        assert decoded.strip() == sentence.strip()


def test_diacritics_preserved(tiny_sentencepiece_model):
    sp = _load(tiny_sentencepiece_model)

    sentence = "Hà Nội là thủ đô của Việt Nam."
    ids = sp.encode(sentence, out_type=int)
    decoded = sp.decode(ids)

    assert _VI_DIACRITIC_RE.search(decoded)


def test_vocab_size_matches_requested(tiny_sentencepiece_model):
    sp = _load(tiny_sentencepiece_model)
    assert sp.get_piece_size() == 500


def test_special_token_ids(tiny_sentencepiece_model):
    sp = _load(tiny_sentencepiece_model)
    assert sp.unk_id() == 0
    assert sp.pad_id() == 1
    assert sp.bos_id() == 2
    assert sp.eos_id() == 3


def _make_file_of_size(path, size_bytes):
    """Create a sparse file of an exact byte size without writing real data."""
    with open(path, "wb") as f:
        if size_bytes > 0:
            f.seek(size_bytes - 1)
            f.write(b"\0")


def test_estimate_vocab_size_scales_with_corpus_size(tmp_path):
    small = tmp_path / "small.txt"
    _make_file_of_size(small, 1 * 1024 * 1024)             # 1 MB
    medium = tmp_path / "medium.txt"
    _make_file_of_size(medium, 100 * 1024 * 1024)          # 100 MB
    large = tmp_path / "large.txt"
    _make_file_of_size(large, 2 * 1024 * 1024 * 1024)      # 2 GB (sparse)

    small_vocab = estimate_vocab_size(small)
    medium_vocab = estimate_vocab_size(medium)
    large_vocab = estimate_vocab_size(large)

    assert small_vocab < medium_vocab < large_vocab
    assert large_vocab < 65535   # must stay under the uint16 dataset format ceiling


def test_estimate_vocab_size_thresholds():
    from ntokenizer.tokenizer import _VOCAB_SIZE_THRESHOLDS

    assert _VOCAB_SIZE_THRESHOLDS == [
        (5, 4_000),
        (50, 8_000),
        (200, 16_000),
        (1024, 32_000),
    ]
