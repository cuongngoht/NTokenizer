"""
CLI: generate Vietnamese text from a trained Tiny GPT checkpoint.

Input : artifacts/checkpoints/ckpt.pt          (trained model, from scripts/train.py)
        artifacts/tokenizer/viwiki_bpe_32k.model (SentencePiece BPE tokenizer)

Usage:
    python scripts/sample.py
    python scripts/sample.py --prompt "Hà Nội là"
    python scripts/sample.py --prompt "Việt Nam" --max_new_tokens 200 --temperature 0.8
    python scripts/sample.py --num_samples 3 --top_k 50
"""

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm
import torch

from ntokenizer.config import GPTConfig
from ntokenizer.device import auto_device
from ntokenizer.model import GPT
from ntokenizer.paths import DEFAULT_CHECKPOINT, DEFAULT_TOKENIZER_MODEL, PROCESSED_DIR
from ntokenizer.tokenizer import encode_prompt


def load_model(ckpt_path: Path, device: torch.device) -> tuple[GPT, dict]:
    """Load model weights from a checkpoint saved by scripts/train.py."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    cfg_dict  = ckpt["config"]
    n_head = cfg_dict["n_head"]
    model_cfg = GPTConfig(
        vocab_size  = cfg_dict.get("vocab_size", 32000),
        block_size  = cfg_dict["block_size"],
        n_layer     = cfg_dict["n_layer"],
        n_head      = n_head,
        n_kv_head   = cfg_dict.get("n_kv_head", n_head),  # fallback: standard MHA
        n_embd      = cfg_dict["n_embd"],
        dropout     = 0.0,
        rope_theta  = cfg_dict.get("rope_theta", 10000.0),
    )

    # GPTConfig doesn't store vocab_size in TrainConfig — read from meta if missing
    if "vocab_size" not in cfg_dict:
        meta_path = PROCESSED_DIR / "meta.json"
        if meta_path.exists():
            with open(meta_path) as fh:
                model_cfg.vocab_size = json.load(fh)["vocab_size"]

    model = GPT(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return model, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from Tiny Vietnamese GPT")
    parser.add_argument("--ckpt",           type=str,   default=str(DEFAULT_CHECKPOINT),
                        help="Path to checkpoint file")
    parser.add_argument("--tokenizer",      type=str,   default=str(DEFAULT_TOKENIZER_MODEL),
                        help="Path to SentencePiece model")
    parser.add_argument("--prompt",         type=str,   default="",
                        help="Seed text (empty = start from BOS token)")
    parser.add_argument("--max_new_tokens", type=int,   default=200,
                        help="Number of new tokens to generate")
    parser.add_argument("--temperature",    type=float, default=0.8,
                        help="> 1 = more random, < 1 = more focused (default: 0.8)")
    parser.add_argument("--top_k",          type=int,   default=50,
                        help="Sample from top-k tokens only (0 = disabled)")
    parser.add_argument("--top_p",          type=float, default=0.95,
                        help="Nucleus sampling threshold (0 = disabled, default: 0.95)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Penalty for repeating tokens (1.0 = disabled, default: 1.1)")
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
        print("       Run scripts/train.py first.", file=sys.stderr)
        sys.exit(1)
    if not tok_path.exists():
        print(f"ERROR: tokenizer not found: {tok_path}", file=sys.stderr)
        print("       Run scripts/train_tokenizer_spm.py first.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = auto_device(args.device)

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
    print(f"  top_p          : {args.top_p if args.top_p > 0 else 'disabled'}")
    print(f"  rep. penalty   : {args.repetition_penalty}")
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
    top_p = args.top_p if args.top_p > 0 else None

    with torch.no_grad():
        for i in range(args.num_samples):
            if args.num_samples > 1:
                print(f"  --- Sample {i + 1} ---")

            out_ids = model.generate(
                idx,
                max_new_tokens      = args.max_new_tokens,
                temperature         = args.temperature,
                top_k               = top_k,
                top_p               = top_p,
                repetition_penalty  = args.repetition_penalty,
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
