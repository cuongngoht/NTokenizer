"""Learning-rate schedule and optimizer configuration for training."""

import math

import torch

from ntokenizer.config import TrainConfig
from ntokenizer.model import GPT


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
