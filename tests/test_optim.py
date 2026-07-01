import math

from ntokenizer.config import GPTConfig, TrainConfig
from ntokenizer.model import GPT
from ntokenizer.optim import configure_optimizer, get_lr


def _cfg(**overrides) -> TrainConfig:
    base = dict(learning_rate=1e-3, min_lr=1e-4, warmup_iters=100, max_iters=1000)
    base.update(overrides)
    return TrainConfig(**base)


def test_lr_warmup_linear():
    cfg = _cfg()
    assert get_lr(0, cfg) == 0.0
    assert math.isclose(get_lr(50, cfg), cfg.learning_rate * 50 / cfg.warmup_iters)
    assert math.isclose(get_lr(99, cfg), cfg.learning_rate * 99 / cfg.warmup_iters)


def test_lr_peak_at_warmup_end():
    cfg = _cfg()
    assert math.isclose(get_lr(cfg.warmup_iters, cfg), cfg.learning_rate, rel_tol=1e-9)


def test_lr_cosine_decay_monotonic():
    cfg = _cfg()
    steps = list(range(cfg.warmup_iters, cfg.max_iters + 1, 50))
    lrs = [get_lr(s, cfg) for s in steps]
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))


def test_lr_floor_after_max_iters():
    cfg = _cfg()
    assert get_lr(cfg.max_iters + 100, cfg) == cfg.min_lr


def test_configure_optimizer_param_groups(device):
    model_cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_kv_head=1, n_embd=16, dropout=0.0)
    model = GPT(model_cfg)

    optimizer = configure_optimizer(model, weight_decay=0.1, learning_rate=1e-3, device=device)

    assert len(optimizer.param_groups) == 2
    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    for p in decay_group["params"]:
        assert p.dim() >= 2
    for p in no_decay_group["params"]:
        assert p.dim() < 2

    # RMSNorm weights are 1-D and must land in the no-decay group
    rmsnorm_params = [p for name, p in model.named_parameters() if "ln" in name and name.endswith("weight")]
    no_decay_ids = {id(p) for p in no_decay_group["params"]}
    assert all(id(p) in no_decay_ids for p in rmsnorm_params)
