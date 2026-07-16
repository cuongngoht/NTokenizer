"""
Tests for fine-tuning support: initializing a fresh training run from a
pretrained checkpoint (--init_from), while an interrupted fine-tuning run
still resumes from its own out_dir first.
"""

import json

import torch

from ntokenizer.config import GPTConfig, TrainConfig
from ntokenizer.dataset import save_bins
from ntokenizer.model import GPT
from ntokenizer.optim import configure_optimizer
from ntokenizer.training import load_pretrained_weights, save_checkpoint, train

TINY_VOCAB = 64


def _tiny_train_cfg(data_dir, out_dir, init_from="", max_iters=0) -> TrainConfig:
    return TrainConfig(
        data_dir=str(data_dir),
        out_dir=str(out_dir),
        block_size=16,
        n_layer=2,
        n_head=2,
        n_kv_head=1,
        n_embd=16,
        dropout=0.0,
        batch_size=4,
        max_iters=max_iters,
        eval_interval=1,
        eval_iters=2,
        log_interval=1,
        checkpoint_interval=1,
        device="cpu",
        init_from=str(init_from),
    )


def _make_dataset(data_dir):
    """Tiny synthetic train/val split, tokens kept within TINY_VOCAB."""
    ids = [i % TINY_VOCAB for i in range(200)]
    n_train, n_val = save_bins(ids, data_dir, train_split=0.9)
    meta = {"vocab_size": TINY_VOCAB, "dtype": "uint16",
            "total_tokens": n_train + n_val, "train_tokens": n_train,
            "val_tokens": n_val, "tokenizer": "fake.model"}
    (data_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _make_checkpoint(cfg: TrainConfig, out_dir, step: int, val_loss: float, seed: int) -> GPT:
    """Build a model with deterministic-but-distinct weights and save it as a checkpoint."""
    torch.manual_seed(seed)
    model_cfg = GPTConfig(
        vocab_size=TINY_VOCAB, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_kv_head=cfg.n_kv_head, n_embd=cfg.n_embd, dropout=cfg.dropout,
    )
    model = GPT(model_cfg)
    optimizer = configure_optimizer(model, cfg.weight_decay, cfg.learning_rate, torch.device("cpu"))
    save_checkpoint(model, optimizer, cfg, step, val_loss, out_dir)
    return model


def _state_dicts_equal(a, b) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


def test_default_init_from_is_empty():
    assert TrainConfig().init_from == ""


def test_load_pretrained_weights_copies_model_only(tmp_path):
    torch.manual_seed(0)
    src_cfg = GPTConfig(vocab_size=TINY_VOCAB, block_size=16, n_layer=2, n_head=2, n_kv_head=1, n_embd=16, dropout=0.0)
    src_model = GPT(src_cfg)
    optimizer = configure_optimizer(src_model, 0.1, 1e-3, torch.device("cpu"))

    out_dir = tmp_path / "pretrained"
    save_checkpoint(src_model, optimizer, TrainConfig(), step=42, val_loss=1.23, out_dir=out_dir)

    torch.manual_seed(1)  # different init, must differ from src_model before loading
    dst_model = GPT(src_cfg)
    assert not _state_dicts_equal(src_model.state_dict(), dst_model.state_dict())

    load_pretrained_weights(out_dir / "ckpt.pt", dst_model)
    assert _state_dicts_equal(src_model.state_dict(), dst_model.state_dict())


def test_train_initializes_from_pretrained_when_out_dir_is_fresh(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_dataset(data_dir)

    pretrained_dir = tmp_path / "pretrained"
    cfg_for_ckpt = _tiny_train_cfg(data_dir, pretrained_dir)
    pretrained_model = _make_checkpoint(cfg_for_ckpt, pretrained_dir, step=999, val_loss=9.9, seed=0)

    out_dir = tmp_path / "finetune_out"
    cfg = _tiny_train_cfg(data_dir, out_dir, init_from=pretrained_dir / "ckpt.pt", max_iters=0)
    train(cfg)

    saved = torch.load(out_dir / "ckpt.pt", weights_only=True)
    assert _state_dicts_equal(saved["model"], pretrained_model.state_dict())


def test_train_resume_takes_priority_over_init_from(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_dataset(data_dir)

    pretrained_dir = tmp_path / "pretrained"
    cfg_for_pretrained = _tiny_train_cfg(data_dir, pretrained_dir)
    _make_checkpoint(cfg_for_pretrained, pretrained_dir, step=999, val_loss=9.9, seed=0)

    # out_dir already has its own (interrupted) fine-tuning checkpoint, with
    # weights distinct from the pretrained one above.
    out_dir = tmp_path / "finetune_out"
    cfg_for_resume = _tiny_train_cfg(data_dir, out_dir, max_iters=5)
    resumed_model = _make_checkpoint(cfg_for_resume, out_dir, step=5, val_loss=5.5, seed=7)

    cfg = _tiny_train_cfg(data_dir, out_dir, init_from=pretrained_dir / "ckpt.pt", max_iters=5)
    train(cfg)

    saved = torch.load(out_dir / "ckpt.pt", weights_only=True)
    assert _state_dicts_equal(saved["model"], resumed_model.state_dict())
