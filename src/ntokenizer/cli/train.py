"""
CLI: train Tiny Vietnamese GPT.

Usage:
    python scripts/train.py
    python scripts/train.py --max_iters 5000 --batch_size 32
    python scripts/train.py --device cpu

    # Fine-tune a pretrained checkpoint on a new dataset (new out_dir/data_dir):
    python scripts/train.py \
        --data_dir data/processed_history \
        --out_dir artifacts/checkpoints/gpt_8k_history_finetune \
        --init_from artifacts/checkpoints/gpt_8k_research/ckpt.pt \
        --max_iters 500 --learning_rate 3e-5 --min_lr 3e-6 --warmup_iters 20
"""

import argparse

from ntokenizer.config import TrainConfig
from ntokenizer.paths import CHECKPOINTS_DIR, PROCESSED_DIR
from ntokenizer.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tiny Vietnamese GPT")
    parser.add_argument("--max_iters",       type=int,   default=5000)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--learning_rate",   type=float, default=3e-4)
    parser.add_argument("--min_lr",          type=float, default=3e-5)
    parser.add_argument("--warmup_iters",    type=int,   default=100)
    parser.add_argument("--weight_decay",    type=float, default=0.1)
    parser.add_argument("--grad_clip",       type=float, default=1.0)
    parser.add_argument("--n_layer",         type=int,   default=4)
    parser.add_argument("--n_head",          type=int,   default=4)
    parser.add_argument("--n_kv_head",       type=int,   default=4)
    parser.add_argument("--n_embd",          type=int,   default=256)
    parser.add_argument("--block_size",      type=int,   default=256)
    parser.add_argument("--rope_theta",      type=float, default=10000.0)
    parser.add_argument("--eval_interval",   type=int,   default=500)
    parser.add_argument("--eval_iters",      type=int,   default=100)
    parser.add_argument("--log_interval",    type=int,   default=50)
    parser.add_argument("--device",          type=str,   default="")
    parser.add_argument("--out_dir",         type=str,   default=str(CHECKPOINTS_DIR))
    parser.add_argument("--data_dir",        type=str,   default=str(PROCESSED_DIR))
    parser.add_argument("--init_from",       type=str,   default="",
                        help="Path to a pretrained ckpt.pt to fine-tune from "
                             "(ignored if out_dir already has a ckpt.pt to resume)")
    args = parser.parse_args()

    cfg = TrainConfig(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
