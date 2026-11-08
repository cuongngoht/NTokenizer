"""
CLI: count how many tokens a prompt uses with a trained tokenizer.

The prompt is encoded exactly as the model sees it, so the count is the
number of context positions it occupies. Also reports fertility
(tokens per word) and compression (bytes per token).

Examples:
    python scripts/count_tokens.py "Hà Nội là thủ đô của Việt Nam."
    python scripts/count_tokens.py --file prompt.txt --show-pieces
    echo "Xin chào" | python scripts/count_tokens.py --block-size 512
"""

import argparse
import sys
from pathlib import Path

import sentencepiece as spm

from ntokenizer.paths import DEFAULT_TOKENIZER_MODEL
from ntokenizer.tokenizer import count_tokens


def read_prompt(text: str | None, file: str | None) -> str:
    if text is not None and file is not None:
        raise SystemExit("ERROR: give either a text argument or --file, not both.")
    if file is not None:
        return Path(file).read_text(encoding="utf-8")
    if text is not None:
        return text
    if sys.stdin.isatty():
        raise SystemExit("ERROR: no prompt given (pass text, --file, or pipe via stdin).")
    return sys.stdin.read()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Count tokens of a prompt")
    parser.add_argument("text", nargs="?", default=None, help="Prompt text")
    parser.add_argument("--file", type=str, default=None, help="Read the prompt from a UTF-8 file")
    parser.add_argument("--model", type=str, default=str(DEFAULT_TOKENIZER_MODEL),
                        help=f"SentencePiece .model file (default: {DEFAULT_TOKENIZER_MODEL})")
    parser.add_argument("--block-size", type=int, default=None,
                        help="Model context length; report whether the prompt fits")
    parser.add_argument("--show-pieces", action="store_true", help="Print every token piece")
    args = parser.parse_args(argv)

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"ERROR: tokenizer model not found: {model_path}\n"
                         "Run scripts/train_tokenizer_spm.py first.")

    prompt = read_prompt(args.text, args.file)
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    stats = count_tokens(sp, prompt)

    print(f"Tokenizer       : {model_path.name} (vocab {sp.get_piece_size():,})")
    print(f"Tokens          : {stats.n_tokens:,}")
    print(f"Characters      : {stats.n_chars:,}")
    print(f"Words           : {stats.n_words:,}")
    print(f"Tokens per word : {stats.tokens_per_word:.2f}")
    print(f"Bytes per token : {stats.bytes_per_token:.2f}")
    if stats.n_unk:
        print(f"Unknown tokens  : {stats.n_unk:,}  (text the tokenizer cannot represent)")
    if args.block_size is not None:
        left = args.block_size - stats.n_tokens
        if left >= 0:
            print(f"Context         : fits in {args.block_size:,} ({left:,} tokens left for generation)")
        else:
            print(f"Context         : exceeds {args.block_size:,} by {-left:,} tokens "
                  "(the oldest tokens will be cropped)")
    if args.show_pieces:
        print("Pieces          : " + " | ".join(stats.pieces))


if __name__ == "__main__":
    main()
