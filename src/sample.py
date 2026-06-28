"""
Generate Vietnamese text from a trained Tiny GPT checkpoint.

Input : out/ckpt.pt                    (trained model, from src/train.py)
        tokenizer/viwiki_bpe_32k.model (SentencePiece BPE tokenizer)

Usage:
    python src/sample.py
    python src/sample.py --prompt "Hà Nội là"
    python src/sample.py --prompt "Việt Nam" --max_new_tokens 200 --temperature 0.8
    python src/sample.py --num_samples 3 --top_k 50
    python src/sample.py --ckpt out/ckpt.pt --prompt "Lịch sử"
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import sentencepiece as spm

from Users.cuongn.NTokenizer.src.model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device) -> tuple[GPT, dict]:
    """Load model weights from a checkpoint saved by train.py."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    cfg_dict  = ckpt["config"]
    model_cfg = GPTConfig(
        vocab_size = cfg_dict.get("vocab_size", 32000),  # may not be in older ckpts
        block_size = cfg_dict["block_size"],
        n_layer    = cfg_dict["n_layer"],
        n_head     = cfg_dict["n_head"],
        n_embd     = cfg_dict["n_embd"],
        dropout    = 0.0,    # always disable dropout at inference
        bias       = cfg_dict["bias"],
    )

    # GPTConfig doesn't store vocab_size in TrainConfig — read from meta if missing
    if "vocab_size" not in cfg_dict:
        meta_path = ROOT / "data" / "meta.json"
        if meta_path.exists():
            import json
            with open(meta_path) as fh:
                model_cfg.vocab_size = json.load(fh)["vocab_size"]

    model = GPT(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return model, ckpt


def encode_prompt(sp: spm.SentencePieceProcessor, text: str) -> list[int]:
    """Encode a text prompt to token IDs. Returns at least [bos_id] if empty."""
    if not text.strip():
        # Empty prompt: start from BOS token (or token 0 as fallback)
        bos = sp.bos_id()
        return [bos if bos >= 0 else 0]
    return sp.encode(text, out_type=int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from Tiny Vietnamese GPT")
    parser.add_argument("--ckpt",           type=str,   default=str(ROOT / "out" / "ckpt.pt"),
                        help="Path to checkpoint file")
    parser.add_argument("--tokenizer",      type=str,   default=str(ROOT / "tokenizer" / "viwiki_bpe_32k.model"),
                        help="Path to SentencePiece model")
    parser.add_argument("--prompt",         type=str,   default="",
                        help="Seed text (empty = start from BOS token)")
    parser.add_argument("--max_new_tokens", type=int,   default=200,
                        help="Number of new tokens to generate")
    parser.add_argument("--temperature",    type=float, default=0.8,
                        help="> 1 = more random, < 1 = more focused (default: 0.8)")
    parser.add_argument("--top_k",          type=int,   default=50,
                        help="Sample from top-k tokens only (0 = disabled)")
    parser.add_argument("--num_samples",    type=int,   default=1,
                        help="Number of independent samples to generate")
    parser.add_argument("--device",         type=str,   default="",
                        help="Force device: cpu / cuda / mps")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    tok_path  = Path(args.tokenizer)

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        print("       Run src/train.py first.", file=sys.stderr)
        sys.exit(1)
    if not tok_path.exists():
        print(f"ERROR: tokenizer not found: {tok_path}", file=sys.stderr)
        print("       Run scripts/train_tokenizer_spm.py first.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available()          else
            "cpu"
        )

    # ------------------------------------------------------------------
    # Load tokenizer + model
    # ------------------------------------------------------------------
    sp    = spm.SentencePieceProcessor(model_file=str(tok_path))
    model, ckpt = load_model(ckpt_path, device)

    print("=" * 55)
    print("  Tiny Vietnamese GPT — Text Generation")
    print("=" * 55)
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  Trained steps  : {ckpt.get('step', '?')}")
    print(f"  Val loss       : {ckpt.get('val_loss', '?')}")
    print(f"  Device         : {device}")
    print(f"  Parameters     : {model.count_parameters():,}")
    print(f"  max_new_tokens : {args.max_new_tokens}")
    print(f"  temperature    : {args.temperature}")
    print(f"  top_k          : {args.top_k if args.top_k > 0 else 'disabled'}")
    print()

    # ------------------------------------------------------------------
    # Encode prompt
    # ------------------------------------------------------------------
    prompt_ids = encode_prompt(sp, args.prompt)
    prompt_text = sp.decode(prompt_ids)

    print(f"  Prompt         : \"{prompt_text}\"")
    print(f"  Prompt tokens  : {prompt_ids}")
    print()

    # Convert to tensor: [1, T_prompt]
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    top_k = args.top_k if args.top_k > 0 else None

    with torch.no_grad():
        for i in range(args.num_samples):
            if args.num_samples > 1:
                print(f"  --- Sample {i + 1} ---")

            out_ids = model.generate(
                idx,
                max_new_tokens = args.max_new_tokens,
                temperature    = args.temperature,
                top_k          = top_k,
            )

            # Decode only the newly generated tokens (after the prompt)
            new_ids   = out_ids[0, len(prompt_ids):].tolist()
            full_ids  = out_ids[0].tolist()
            generated = sp.decode(new_ids)
            full_text = sp.decode(full_ids)

            print(f"  {full_text}")
            print()

    print("=" * 55)


if __name__ == "__main__":
    main()
