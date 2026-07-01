"""All hyperparameter and path configuration dataclasses in one place."""

from dataclasses import dataclass

from ntokenizer.paths import CHECKPOINTS_DIR, PROCESSED_DIR


@dataclass
class GPTConfig:
    """Model architecture hyperparameters."""
    vocab_size: int   = 32000
    block_size: int   = 256
    n_layer:    int   = 4
    n_head:     int   = 4
    n_kv_head:  int   = 4       # GQA: n_head must be divisible by n_kv_head
    n_embd:     int   = 256
    dropout:    float = 0.1
    rope_theta: float = 10000.0  # RoPE base frequency


@dataclass
class TrainConfig:
    # ── Paths ────────────────────────────────────────────────────────────
    data_dir:            str   = str(PROCESSED_DIR)
    out_dir:             str   = str(CHECKPOINTS_DIR)

    # ── Model (must match GPTConfig defaults) ────────────────────────────
    block_size:          int   = 256
    n_layer:             int   = 4
    n_head:              int   = 4
    n_kv_head:           int   = 4      # GQA: must divide n_head evenly
    n_embd:              int   = 256
    dropout:             float = 0.1
    rope_theta:          float = 10000.0

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
