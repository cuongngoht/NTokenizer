"""
Training loop for Tiny Vietnamese GPT.

Input : data/processed/train.bin   (uint16 token IDs, from ntokenizer.dataset)
        data/processed/val.bin
        data/processed/meta.json   (vocab_size, etc.)
Output: artifacts/checkpoints/ckpt.pt   (best checkpoint by val loss)
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from ntokenizer.config import GPTConfig, TrainConfig
from ntokenizer.dataset import get_batch, load_data
from ntokenizer.device import auto_device
from ntokenizer.model import GPT
from ntokenizer.optim import configure_optimizer, get_lr


# ---------------------------------------------------------------------------
# Loss estimation (evaluation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(
    model:      GPT,
    train_data,
    val_data,
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
            _, loss, _ = model(x, y)
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


def load_pretrained_weights(path: Path, model: GPT) -> None:
    """
    Load only model weights from another run's checkpoint (e.g. a base
    pretraining checkpoint), leaving the optimizer fresh and the step
    counter at 0. Used to start a fine-tuning run on a new dataset.
    """
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model"])
    print(
        f"  Init from      : {path}  "
        f"(pretrained step {ckpt['step']}, val_loss {ckpt['val_loss']:.4f})"
    )


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
    device = auto_device(cfg.device)

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
    ckpt_path   = out_dir / "ckpt.pt"
    ckpt_exists = ckpt_path.exists()

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
    if cfg.init_from and not ckpt_exists:
        print(f"  init_from      : {cfg.init_from}")
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
        vocab_size  = vocab_size,
        block_size  = cfg.block_size,
        n_layer     = cfg.n_layer,
        n_head      = cfg.n_head,
        n_kv_head   = cfg.n_kv_head,
        n_embd      = cfg.n_embd,
        dropout     = cfg.dropout,
        rope_theta  = cfg.rope_theta,
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
    if ckpt_exists:
        print()
        start_step = load_checkpoint(ckpt_path, model, optimizer)
    elif cfg.init_from:
        print()
        load_pretrained_weights(Path(cfg.init_from), model)

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
        _, loss, _ = model(x, y)

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
    print("  Next: python scripts/sample.py")
    print("=" * 55)
