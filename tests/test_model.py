import math

import pytest
import torch

from ntokenizer.config import GPTConfig
from ntokenizer.model import GPT


def test_forward_pass_shapes(tiny_gpt_config, device):
    model = GPT(tiny_gpt_config).to(device)
    B, T = 2, 8
    ids = torch.randint(0, tiny_gpt_config.vocab_size, (B, T), device=device)
    targets = torch.randint(0, tiny_gpt_config.vocab_size, (B, T), device=device)

    logits, loss, kvs = model(ids, targets)

    assert logits.shape == (B, T, tiny_gpt_config.vocab_size)
    assert loss.dim() == 0
    assert len(kvs) == tiny_gpt_config.n_layer
    head_dim = tiny_gpt_config.n_embd // tiny_gpt_config.n_head
    for k, v in kvs:
        assert k.shape == (B, tiny_gpt_config.n_kv_head, T, head_dim)
        assert v.shape == (B, tiny_gpt_config.n_kv_head, T, head_dim)


def test_untrained_loss_near_uniform(tiny_gpt_config, device):
    model = GPT(tiny_gpt_config).to(device)
    model.eval()

    expected = math.log(tiny_gpt_config.vocab_size)
    losses = []
    with torch.no_grad():
        for _ in range(10):
            ids = torch.randint(0, tiny_gpt_config.vocab_size, (4, tiny_gpt_config.block_size), device=device)
            targets = torch.randint(0, tiny_gpt_config.vocab_size, (4, tiny_gpt_config.block_size), device=device)
            _, loss, _ = model(ids, targets)
            losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    assert abs(mean_loss - expected) < 0.5


def test_weight_tying(tiny_gpt_config):
    model = GPT(tiny_gpt_config)
    assert model.transformer.wte.weight is model.lm_head.weight


def test_gqa_shapes():
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=4, n_kv_head=2, n_embd=16, dropout=0.0)
    model = GPT(cfg)
    attn = model.transformer.h[0].attn

    assert attn.n_rep == 2
    head_dim = cfg.n_embd // cfg.n_head
    assert attn.k_proj.out_features == cfg.n_kv_head * head_dim
    assert attn.v_proj.out_features == cfg.n_kv_head * head_dim
    assert attn.q_proj.out_features == cfg.n_head * head_dim


def test_block_size_assertion(tiny_gpt_config, device):
    model = GPT(tiny_gpt_config).to(device)
    too_long = tiny_gpt_config.block_size + 1
    ids = torch.randint(0, tiny_gpt_config.vocab_size, (1, too_long), device=device)

    with pytest.raises(AssertionError):
        model(ids)
