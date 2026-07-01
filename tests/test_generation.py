import torch

from ntokenizer.model import GPT, apply_repetition_penalty


@torch.no_grad()
def _greedy_no_cache(model: GPT, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Reference implementation: full forward pass every step, no KV cache."""
    for _ in range(max_new_tokens):
        logits, _, _ = model(idx, past_kvs=None)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    return idx


def test_kv_cache_matches_no_cache_reference(tiny_gpt_config, device):
    model = GPT(tiny_gpt_config).to(device)
    model.eval()

    prompt = torch.randint(0, tiny_gpt_config.vocab_size, (1, 3), device=device)

    # top_k=1 makes multinomial sampling deterministic (equivalent to argmax),
    # isolating cache-vs-no-cache numerics from RNG-driven sampling.
    cached = model.generate(
        prompt.clone(), max_new_tokens=5,
        temperature=1.0, top_k=1, top_p=None, repetition_penalty=1.0,
    )
    reference = _greedy_no_cache(model, prompt.clone(), max_new_tokens=5)

    assert torch.equal(cached, reference)


def test_generate_respects_max_new_tokens(tiny_gpt_config, device):
    model = GPT(tiny_gpt_config).to(device)
    model.eval()
    prompt = torch.randint(0, tiny_gpt_config.vocab_size, (1, 4), device=device)

    out = model.generate(prompt, max_new_tokens=6, temperature=0.8, top_k=10)

    assert out.shape == (1, 4 + 6)


def test_generate_kv_cache_eviction(tiny_gpt_config, device):
    model = GPT(tiny_gpt_config).to(device)
    model.eval()
    # prompt + max_new_tokens exceeds block_size, forcing the cache-eviction branch
    prompt = torch.randint(0, tiny_gpt_config.vocab_size, (1, tiny_gpt_config.block_size - 2), device=device)

    out = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=10)

    assert out.shape == (1, prompt.shape[1] + 10)


def test_repetition_penalty_reduces_score_for_seen_tokens():
    # Token 2 appears in the context and has a positive score, token 5 does not
    # appear and must be left untouched.
    logits = torch.tensor([[1.0, 1.0, 4.0, 1.0, 1.0, -3.0]])
    idx = torch.tensor([[2, 2, 0]])

    penalized = apply_repetition_penalty(logits, idx, penalty=2.0)

    assert penalized[0, 2].item() == 2.0   # 4.0 / 2.0 (positive score divided)
    assert penalized[0, 0].item() == 0.5   # 1.0 / 2.0
    assert penalized[0, 5].item() == -3.0  # untouched — token 5 not in context
    assert penalized[0, 3].item() == 1.0   # untouched — token 3 not in context


def test_repetition_penalty_amplifies_negative_scores():
    # Negative scores get pushed further negative (multiplied, not divided),
    # so an already-unlikely repeated token becomes even less likely.
    logits = torch.tensor([[-2.0]])
    idx = torch.tensor([[0]])

    penalized = apply_repetition_penalty(logits, idx, penalty=2.0)

    assert penalized[0, 0].item() == -4.0


def test_repetition_penalty_noop_does_not_mutate_input():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    idx = torch.tensor([[1]])

    penalized = apply_repetition_penalty(logits, idx, penalty=2.0)

    assert logits[0, 1].item() == 2.0   # original tensor left untouched
    assert penalized[0, 1].item() == 1.0
