import re

import sentencepiece as spm

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
