"""
Test the trained Vietnamese SentencePiece BPE tokenizer.

Loads a SentencePiece model and runs encode/decode on sample Vietnamese
sentences, printing pieces, IDs, decoded text, token count, and round-trip
match status.
"""

import argparse
import re
import sys
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).parent.parent

TEST_SENTENCES = [
    "Hà Nội là thủ đô của Việt Nam.",
    "Hà Thành là một tên gọi khác của Thăng Long - Hà Nội.",
    "Có nhiều trường đại học lớn tại Thành phố Hồ Chí Minh.",
    "Tôi muốn tự huấn luyện mô hình ngôn ngữ tiếng Việt.",
    "OpenAI, ChatGPT và Python 3.12 có gì khác nhau?",
]

_VI_DIACRITIC_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
    r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]"
)


def has_vietnamese_diacritics(text: str) -> bool:
    return bool(_VI_DIACRITIC_RE.search(text))


def test_sentence(sp: spm.SentencePieceProcessor, sentence: str) -> bool:
    pieces = sp.encode(sentence, out_type=str)
    ids = sp.encode(sentence, out_type=int)
    decoded = sp.decode(ids)

    match = decoded.strip() == sentence.strip()
    match_symbol = "✓" if match else "✗"

    print(f"Input    : {sentence}")
    print(f"Pieces   : {pieces}")
    print(f"IDs      : {ids}")
    print(f"Decoded  : {decoded}")
    print(f"Tokens   : {len(pieces)}")
    print(f"Match    : {match_symbol}")

    if has_vietnamese_diacritics(sentence) and not has_vietnamese_diacritics(decoded):
        print("WARNING  : Vietnamese diacritics lost in decoded output!")

    return match


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Vietnamese SentencePiece tokenizer")
    parser.add_argument("--model", type=str,
                        default=str(ROOT / "tokenizer" / "viwiki_bpe_32k.model"),
                        help="Path to .model file (default: tokenizer/viwiki_bpe_32k.model)")
    args = parser.parse_args()

    MODEL_PATH = Path(args.model)

    if not MODEL_PATH.exists():
        print(f"ERROR: model not found: {MODEL_PATH}", file=sys.stderr)
        print("Run scripts/train_tokenizer_spm.py first.", file=sys.stderr)
        sys.exit(1)

    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL_PATH))

    print("=" * 60)
    print(f"  Model    : {MODEL_PATH.name}")
    print(f"  Vocab    : {sp.get_piece_size()} tokens")
    print(f"  unk_id   : {sp.unk_id()}")
    print(f"  pad_id   : {sp.pad_id()}")
    print(f"  bos_id   : {sp.bos_id()}")
    print(f"  eos_id   : {sp.eos_id()}")
    print("=" * 60)

    passed = 0
    for i, sentence in enumerate(TEST_SENTENCES, 1):
        print(f"\n[{i}/{len(TEST_SENTENCES)}]")
        ok = test_sentence(sp, sentence)
        if ok:
            passed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{len(TEST_SENTENCES)} sentences matched exactly")
    if passed < len(TEST_SENTENCES):
        print("  Note: mismatches are usually whitespace normalisation by SPM.")
    print("=" * 60)


if __name__ == "__main__":
    main()
