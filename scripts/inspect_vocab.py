"""
Inspect the vocabulary of the trained Vietnamese SentencePiece tokenizer.

Reads a tokenizer .vocab file (TSV: token<TAB>score) and:
  - Prints the first 100 tokens (IDs 0–99)
  - Searches for tokens containing specific Vietnamese substrings
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

SEARCH_TERMS = ["Hà", "Nội", "Việt", "Nam", "thủ", "đô", "Có"]


def load_vocab(vocab_path: Path) -> list[tuple[str, float]]:
    tokens = []
    with open(vocab_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            token = parts[0]
            score = float(parts[1]) if len(parts) > 1 else 0.0
            tokens.append((token, score))
    return tokens


def print_first_n(tokens: list[tuple[str, float]], n: int = 100) -> None:
    print(f"First {n} tokens:")
    print("-" * 40)
    for i, (token, score) in enumerate(tokens[:n]):
        print(f"  [{i:4d}] {token:<25s} {score:.4f}")
    print()


def search_vocab(tokens: list[tuple[str, float]], terms: list[str]) -> None:
    print("Vocabulary search:")
    print("-" * 40)
    for term in terms:
        matches = [
            (i, tok, score)
            for i, (tok, score) in enumerate(tokens)
            if term in tok
        ]
        print(f'\nTokens containing "{term}": {len(matches)} found')
        for idx, tok, score in matches[:20]:
            print(f"    [{idx:5d}] {tok:<25s} {score:.4f}")
        if len(matches) > 20:
            print(f"    ... and {len(matches) - 20} more")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Vietnamese SentencePiece vocabulary")
    parser.add_argument("--vocab", type=str,
                        default=str(ROOT / "tokenizer" / "viwiki_bpe_32k.vocab"),
                        help="Path to .vocab file (default: tokenizer/viwiki_bpe_32k.vocab)")
    args = parser.parse_args()

    VOCAB_PATH = Path(args.vocab)

    if not VOCAB_PATH.exists():
        print(f"ERROR: vocab file not found: {VOCAB_PATH}", file=sys.stderr)
        print("Run scripts/train_tokenizer_spm.py first.", file=sys.stderr)
        sys.exit(1)

    tokens = load_vocab(VOCAB_PATH)

    print("=" * 55)
    print(f"  Vocab file  : {VOCAB_PATH.name}")
    print(f"  Total tokens: {len(tokens)}")
    print("=" * 55)
    print()

    print_first_n(tokens, n=100)
    search_vocab(tokens, SEARCH_TERMS)


if __name__ == "__main__":
    main()
