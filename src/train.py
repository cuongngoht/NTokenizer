"""
Training loop for Tiny Vietnamese GPT.

Input : data/train.bin   (uint16 token IDs, from scripts/prepare_dataset.py)
        data/val.bin     (uint16 token IDs)
        data/meta.json   (vocab_size, etc.)
Output: out/ckpt.pt      (best checkpoint by val loss)

Usage:
    python src/train.py
    python src/train.py --max_iters 5000 --batch_size 32
    python src/train.py --device cpu
"""

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# src/ is the package root; add project root so we can import model.py
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from Users.cuongn.NTokenizer.src.model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # ── Paths ────────────────────────────────────────────────────────────
    data_dir:            str   = str(ROOT / "data")
    out_dir:             str   = str(ROOT / "out")

    # ── Model (must match GPTConfig defaults) ────────────────────────────
    block_size:          int   = 256
    n_layer:             int   = 4
    n_head:              int   = 4
    n_embd:              int   = 256
    dropout:             float = 0.1
    bias:                bool  = True

    # ── Optimization ─────────────────────────────────────────────────────
    batch_size:          int   = 32
    max_iters:           int   = 5000
    learning_rate:       float = 3e-4    # peak LR
    min_lr:              float = 3e-5    # floor (1/10 of peak)
    warmup_iters:        int   = 100     # steps of linear LR warm-up
    weight_decay:        float = 0.1
    grad_clip:           float = 1.0     # max gradient norm (0 = disabled)

    # ── Logging & checkpointing ───────────────────────────────────────────
    eval_interval:       int   = 500     # evaluate every N steps
    eval_iters:          int   = 100     # batches to average for eval loss
    log_interval:        int   = 50      # print train loss every N steps
    checkpoint_interval: int   = 1000    # save checkpoint every N steps

    # ── Device (empty = auto-detect) ─────────────────────────────────────
    device:              str   = ""


# ---------------------------------------------------------------------------
# Data loading
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


# ---------------------------------------------------------------------------
# Learning rate schedule: linear warm-up → cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainConfig) -> float:
    """
    Return the learning rate for the current step.

    Phase 1 (0 → warmup_iters):   linear ramp from 0 to learning_rate.
    Phase 2 (warmup → max_iters): cosine decay from learning_rate to min_lr.
    Phase 3 (> max_iters):        constant min_lr.

    Why cosine decay?  It smoothly reduces the step size as training converges,
    letting the optimizer fine-tune rather than overshoot the minimum.
    """
    # Linear warm-up
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / cfg.warmup_iters

    # After training ends
    if step > cfg.max_iters:
        return cfg.min_lr

    # Cosine decay
    progress = (step - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1.0 → 0.0
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def configure_optimizer(
    model:         GPT,
    weight_decay:  float,
    learning_rate: float,
    device:        torch.device,
) -> torch.optim.AdamW:
    """
    AdamW with weight decay applied only to weight matrices, not to biases
    or LayerNorm parameters (they are 1-D tensors).

    Why separate groups?  Weight decay is a regularizer that shrinks weights
    toward zero.  Applying it to biases or layer-norm scales is usually
    harmful because those parameters work best near their learned value, not
    near zero.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)       # weight matrices — apply decay
        else:
            no_decay.append(param)    # biases, LayerNorm — no decay

    n_decay    = sum(p.numel() for p in decay)
    n_no_decay = sum(p.numel() for p in no_decay)
    print(f"  Optimizer      : AdamW  "
          f"(decay={n_decay:,} params, no-decay={n_no_decay:,} params)")

    # fused=True uses a CUDA-optimized kernel; not available on MPS/CPU
    use_fused = device.type == "cuda"
    return torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        fused=use_fused,
    )


# ---------------------------------------------------------------------------
# Loss estimation (evaluation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(
    model:      GPT,
    train_data: np.ndarray,
    val_data:   np.ndarray,
    cfg:        TrainConfig,
    device:     torch.device,
) -> dict[str, float]:
    """
    Compute average loss over eval_iters random batches for each split.

    We use many batches (not just one) because a single batch has high
    variance — averaging gives a stable estimate of the true loss.

    model.eval() disables dropout so the evaluation is deterministic.
    """
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.block_size, cfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:     GPT,
    optimizer: torch.optim.AdamW,
    cfg:       TrainConfig,
    step:      int,
    val_loss:  float,
    out_dir:   Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config":    asdict(cfg),
            "step":      step,
            "val_loss":  val_loss,
        },
        out_dir / "ckpt.pt",
    )
    print(f"    → checkpoint saved  (step {step}, val_loss {val_loss:.4f})")


def load_checkpoint(
    path:      Path,
    model:     GPT,
    optimizer: torch.optim.AdamW,
) -> int:
    """Load model + optimizer state. Returns step number."""
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  Resumed from   : {path}  (step {ckpt['step']}, val_loss {ckpt['val_loss']:.4f})")
    return ckpt["step"]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    data_dir = Path(cfg.data_dir)
    out_dir  = Path(cfg.out_dir)

    # ------------------------------------------------------------------
    # Guard: input files must exist
    # ------------------------------------------------------------------
    for f, hint in [
        (data_dir / "train.bin",  "scripts/prepare_dataset.py"),
        (data_dir / "val.bin",    "scripts/prepare_dataset.py"),
        (data_dir / "meta.json",  "scripts/prepare_dataset.py"),
    ]:
        if not f.exists():
            print(f"ERROR: {f} not found. Run {hint} first.", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if cfg.device:
        device = torch.device(cfg.device)
    else:
        device = torch.device(
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available()          else
            "cpu"
        )

    # Seed for reproducibility
    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # Read vocab_size from meta.json (set by prepare_dataset.py)
    # ------------------------------------------------------------------
    with open(data_dir / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    vocab_size = meta["vocab_size"]

    # ------------------------------------------------------------------
    # Print header
    # ------------------------------------------------------------------
    tokens_per_iter = cfg.batch_size * cfg.block_size
    print("=" * 55)
    print("  Tiny Vietnamese GPT — Training")
    print("=" * 55)
    print(f"  Device         : {device}")
    print(f"  vocab_size     : {vocab_size}")
    print(f"  block_size     : {cfg.block_size}")
    print(f"  batch_size     : {cfg.batch_size}  ({tokens_per_iter:,} tokens/iter)")
    print(f"  n_layer/head   : {cfg.n_layer} / {cfg.n_head}")
    print(f"  n_embd         : {cfg.n_embd}")
    print(f"  max_iters      : {cfg.max_iters}")
    print(f"  lr             : {cfg.learning_rate} → {cfg.min_lr}  (cosine)")
    print(f"  out_dir        : {out_dir}")
    print()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_data, val_data = load_data(data_dir)
    print(f"  train tokens   : {len(train_data):,}")
    print(f"  val tokens     : {len(val_data):,}")
    print()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg = GPTConfig(
        vocab_size = vocab_size,
        block_size = cfg.block_size,
        n_layer    = cfg.n_layer,
        n_head     = cfg.n_head,
        n_embd     = cfg.n_embd,
        dropout    = cfg.dropout,
        bias       = cfg.bias,
    )
    model = GPT(model_cfg).to(device)
    print(f"  Parameters     : {model.count_parameters():,}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = configure_optimizer(
        model, cfg.weight_decay, cfg.learning_rate, device
    )

    # ------------------------------------------------------------------
    # Resume from checkpoint if one exists
    # ------------------------------------------------------------------
    start_step = 0
    ckpt_path  = out_dir / "ckpt.pt"
    if ckpt_path.exists():
        print()
        start_step = load_checkpoint(ckpt_path, model, optimizer)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print()
    print(f"  {'Step':>6}  {'Train':>8}  {'Val':>8}  {'LR':>9}  {'ms/iter':>8}")
    print("  " + "-" * 48)

    model.train()
    best_val_loss = float("inf")
    t0 = time.monotonic()
    t_iter = time.monotonic()

    for step in range(start_step, cfg.max_iters + 1):

        # ── Update learning rate ────────────────────────────────────────
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ── Evaluate & checkpoint ───────────────────────────────────────
        if step % cfg.eval_interval == 0:
            losses   = estimate_loss(model, train_data, val_data, cfg, device)
            iter_ms  = (time.monotonic() - t0) / max(step, 1) * 1000
            print(
                f"  {step:>6}  {losses['train']:>8.4f}  {losses['val']:>8.4f}"
                f"  {lr:>9.2e}  {iter_ms:>7.1f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(model, optimizer, cfg, step, losses["val"], out_dir)

        # ── Last step: nothing left to train ────────────────────────────
        if step == cfg.max_iters:
            break

        # ── Forward + backward ──────────────────────────────────────────
        x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        _, loss = model(x, y)

        # Zero gradients (set_to_none=True is faster than zeroing in-place)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping: prevents a single bad batch from blowing up weights
        if cfg.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        # ── Light log every log_interval steps ─────────────────────────
        if step % cfg.log_interval == 0 and step % cfg.eval_interval != 0:
            iter_ms = (time.monotonic() - t_iter) / cfg.log_interval * 1000
            t_iter  = time.monotonic()
            print(
                f"  {step:>6}  {loss.item():>8.4f}  {'--':>8}"
                f"  {lr:>9.2e}  {iter_ms:>7.1f}"
            )

        # ── Periodic checkpoint (not tied to eval) ──────────────────────
        if step > 0 and step % cfg.checkpoint_interval == 0:
            if step % cfg.eval_interval != 0:   # avoid double-save
                losses = estimate_loss(model, train_data, val_data, cfg, device)
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    save_checkpoint(
                        model, optimizer, cfg, step, losses["val"], out_dir
                    )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    total = time.monotonic() - t0
    print()
    print(f"  Training complete in {total / 60:.1f} min")
    print(f"  Best val loss  : {best_val_loss:.4f}")
    print(f"  Checkpoint     : {out_dir / 'ckpt.pt'}")
    print()
    print("  Next: python src/sample.py")
    print("=" * 55)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

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
    parser.add_argument("--n_embd",          type=int,   default=256)
    parser.add_argument("--block_size",      type=int,   default=256)
    parser.add_argument("--eval_interval",   type=int,   default=500)
    parser.add_argument("--eval_iters",      type=int,   default=100)
    parser.add_argument("--log_interval",    type=int,   default=50)
    parser.add_argument("--device",          type=str,   default="")
    parser.add_argument("--out_dir",         type=str,   default=str(ROOT / "out"))
    parser.add_argument("--data_dir",        type=str,   default=str(ROOT / "data"))
    args = parser.parse_args()

    cfg = TrainConfig(**{
        k: v for k, v in vars(args).items()
    })
    train(cfg)


if __name__ == "__main__":
    main()
