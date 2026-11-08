# D2 — Small-model training recipes (round 1, worker 1)

Accessed 2026-11-08 for all sources. Web sources only (research firewall respected). huggingface.co was unreachable from this session (DNS blocked), so HF model cards and blogs were seen only through search snippets and mirrors. arxiv was not fetched; paper claims come from search snippets of the paper or proceedings pages.

## Claims

1. SmolLM2-135M was pretrained on 2T tokens, about 15,000 tokens per parameter, far past "compute-optimal". | SmolLM2 135M model card, https://huggingface.co/HuggingFaceTB/SmolLM2-135M (search snippet) | Hugging Face | 2024-11 | accessed 2026-11-08 | medium (official card, but seen only as a snippet) | size/scale
2. SmolLM2-1.7B was pretrained on 11T tokens using 256 H100s. The 360M token count was NOT confirmed this run. | SmolLM2 model card mirror (search result); paper arXiv 2502.02737 | Hugging Face | 2025-02 | accessed 2026-11-08 | medium-low (mirror only) | size/scale
3. SmolLM3-3B: about 11.2T pretraining tokens in 3 stages. Stage I was about 8T tokens at 85% web / 12% code / 3% math. Then about 100B tokens for long context, and about 24 days wall-clock at about 14K tok/s per GPU and 29.4% MFU (per a blog mirror). | SmolLM3 blog via mirror hn.jakechampion.name/reader/https://huggingface.co/blog/smollm3; learnopencv.com/smollm3-explained | Hugging Face / LearnOpenCV | 2025-07 | accessed 2026-11-08 | medium-low (secondary) | size/scale
4. Llama 3.2 1B was pretrained on "up to 9 trillion tokens", about 7,000+ tokens per parameter. NVIDIA NeMo docs list 15T, which is likely a copy error. | Azure AI catalog Llama-3.2-1B, https://ai.azure.com/catalog/models/Llama-3.2-1B; Mozilla llamafile card | Microsoft / Mozilla (redistributing the Meta card) | 2024-09 | accessed 2026-11-08 | medium | size/scale
5. MobileLLM: for sub-billion LLMs "depth is more important than width". A deep-thin design plus embedding sharing plus GQA gives +2.7% / +4.3% average accuracy over prior 125M/350M SOTA. Block-wise weight sharing (MobileLLM-LS) adds +0.7% / +0.8%. | MobileLLM, ICML 2024, https://proceedings.mlr.press/v235/liu24ce.html (arXiv 2402.14905) | Meta / PMLR | 2024-02 | accessed 2026-11-08 | high | performance
6. MobileLLM: on a 30-layer 125M model, tying input/output embeddings saves 16M params (about 11.8%) for only a 0.2-point drop in average accuracy. This is directly relevant because the vocab embedding is a large share of a small model. | same | Meta / PMLR | 2024-02 | accessed 2026-11-08 | high | practice
7. Data repetition: for a fixed compute budget, training up to about 4 epochs on repeated data gives loss differences that "tend to be insignificant" versus unique data, with no downstream difference. Beyond that, returns diminish toward zero. Basis: 400+ runs, 10M to 9B params, up to 1,500 epochs. | Muennighoff et al., Scaling Data-Constrained Language Models, NeurIPS 2023 / JMLR 2025, https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract.html | NeurIPS / JMLR | 2023-05 (JMLR 2025) | accessed 2026-11-08 | high | practice
8. llm.c GPT-2 124M reproduction details:
   - Model: 12 layers, 12 heads, d=768.
   - Data: 10B FineWeb tokens (about 80 tokens/param), 18,865 steps at a 524,288-token batch.
   - Speed and cost: about 90 min on one 8xA100 80GB node, about $20 (at about $14/hr Lambda), up to about 60% MFU.
   - One GPU takes "~4–24 hours depending on the GPU". A single A100 40GB PCIe log shows 177,968 tok/s at 49% MFU.
   | Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20, https://github.com/karpathy/llm.c/discussions/481 | Karpathy / GitHub | 2024-05 | accessed 2026-11-08 | high | performance
9. nanoGPT README:
   - GPT-2 124M on OpenWebText: about 4 days on 8xA100 40GB, reaching loss about 2.85.
   - Shakespeare-char (6 layers, 6 heads, 384-d, ctx 256): about 3 min on one A100, val loss 1.4697.
   - CPU mini config (4L/4H/128d, ctx 64, 2000 iters): about 3 min, loss 1.88.
   - `--device=mps` gives "2-3X" over CPU.
   - The repo is now "very old and deprecated", with nanochat as its successor.
   | karpathy/nanoGPT README, https://github.com/karpathy/nanoGPT | Karpathy / GitHub | 2022-12 (deprecation note later) | accessed 2026-11-08 | high | practice/version
10. nanochat: one `--depth` dial sets width, heads, LR, horizon and weight decay automatically. GPT-2 capability (CORE > 0.2565) is reached at about d24–d26 for about $48 (about 2 h on 8xH100, about $15 on spot). It uses an AdamW + Muon optimizer and bf16 on CUDA, but fp32 on MPS/CPU. A single GPU works and is "8 times longer". It needs a smaller `--device-batch-size` on GPUs under 80GB. A d12 (GPT-1-sized) model works for about 5-min experiments. | karpathy/nanochat README, https://github.com/karpathy/nanochat | Karpathy / GitHub | 2025-10 (leaderboard to 2026-03) | accessed 2026-11-08 | high | practice/performance
11. nanochat leaderboard: from 3.04 h (Jan 29 2026) to 1.65 h (Mar 14 2026) to reach GPT-2 CORE on 8xH100. The largest single gain (2.76 h to 2.02 h) came from switching the dataset to NVIDIA ClimbMix, i.e. data quality beat architecture tweaks. A batch of 1M tokens and fp8 also helped. | same | Karpathy / GitHub | 2026-03 | accessed 2026-11-08 | high | performance
12. modded-nanogpt (Track 1) target is ≤3.28 val loss on FineWeb with a 124M-class model on 8xH100. The README reports record #92 at 0.665 min (2026-08-30) using under 330M tokens, versus 10B for llm.c. Core transferable tricks are listed as: Muon optimizer (with Polar Express), rotary embeddings, QK-Norm, ReLU², value embeddings, skip connections, and sliding-window FlashAttention-3. Record-specific tricks (FP8 MLP, hashed n-gram embedding with 84.6M rows, "ANVIL" optimizer stack, sampled softmax, CUDA graphs) are H100-specific or exotic. | KellerJordan/modded-nanogpt README, https://github.com/KellerJordan/modded-nanogpt | Keller Jordan / GitHub | 2026-08 | accessed 2026-11-08 | medium (fetched via summarizer; record figures are surprising and not cross-checked) | performance/practice
13. TinyStories: GPT-3.5/4-generated stories with a small-child vocabulary let models of a few million to about 33M params (checkpoints 1M, 3M, 9M, 28M, 33M) produce fluent, multi-paragraph, grammatical, consistent stories with some reasoning. The paper contrasts this with ~125M GPT-Neo/GPT-2 small, which "rarely" stay coherent beyond a few words on broad web data. Implication (inference, not from the source): narrowing and simplifying the distribution substitutes for scale. A Wikipedia-only corpus is narrower than web data but not nearly as simple as TinyStories. | Eldan & Li, TinyStories, https://www.microsoft.com/en-us/research/?p=973236 (arXiv 2305.07759) | Microsoft Research | 2023-05 | accessed 2026-11-08 | high (for findings); inference flagged | performance
14. Apple Silicon: Simon Willison on an M2 MacBook Pro reports CPU is about 3x slower than MPS for nanoGPT Shakespeare. He did 6,184 iterations in about 3.5 min on MPS and hit an MPS error during generation (older PyTorch). | Running nanoGPT on a MacBook M2..., https://til.simonwillison.net/llms/nanogpt-shakespeare-m2 | Simon Willison TIL | 2023-02 (approx.) | accessed 2026-11-08 | medium | performance
15. Consumer GPU throughput: no reliable primary RTX 3090/4090 benchmark was found. A community log shows stock nanoGPT 124M on an RTX 4070 at about 9.3 s/iter for a 524,288-token batch (about 56k tok/s, about 15% MFU), and that author estimates about 59 h for 10B tokens. An AI-generated page claims about 160k tok/s on a 4090, unverified. | lzwjava.github.io/nano-gpt-training-time-estimate-en | personal blog (AI-generated) | 2025 (approx.) | accessed 2026-11-08 | low | performance

### Derived estimates (arithmetic on retrieved figures; not sourced claims)
- **6·N·D:** 124M params × 10B tokens ≈ 7.4e18 FLOPs (llm.c run). The nanochat speedrun model is described as a "4e19 FLOPs capability model".
- **RTX 4090 at 124M, 1B tokens:** At about 100–180k tok/s (bracketed by the A100 177,968 tok/s figure and the 4070 figure of 56k tok/s), 1B tokens takes roughly 1.5–3 h. 10B tokens takes about 15–30 h. This is an extrapolation, not measured.
- **Mac MPS:** Expect throughput in the low tens of thousands of tok/s or less (fp32 per nanochat; 2–3x over CPU per nanoGPT). This is an unverified belief; no direct tok/s figure was retrieved.
- **Vietnamese Wikipedia scale:** If the corpus is roughly a few hundred million tokens (corpus size is D1's to verify), then 4 epochs (Claim 7) gives about 1–2B effective tokens. Under ~20 tok/param that is compute-optimal for about 50–100M params, and the claim-1/4 over-training trend favors smaller models (10–60M) trained longer.

### Unverified beliefs (not evidenced this run)
- **Chinchilla ratio:** The ~20 tokens/param rule (Hoffmann et al. 2022) is well known, but no source was retrieved this run.
- **SmolLM2-360M token count:** Believed to be 4T, but not confirmed.

## Leads
- **Official HF pages:** The SmolLM3 blog (huggingface.co/blog/smollm3) and "The Smol Training Playbook" (HF, late 2025) are probably the best primary source on small-model ablations. Both need huggingface.co allowed in the network.
- **nanochat "Beating GPT-2 for <<$100" guide (2026-02-01):** Likely has per-depth param counts, tokens/param ratio (nanochat reportedly uses a fixed ratio), and the Muon LR settings.
- **modded-nanogpt records log:** Worth checking to see which tricks (Muon, QK-norm, ReLU², value embeddings) contribute most at 124M on a single GPU. Also check the community "4090 speedrun" forks.
- **Other leads:**
  - Muennighoff datablations repo (github.com/huggingface/datablations): the data-constrained scaling formula, for an exact epoch budget.
  - MobileLLM repo (github.com/facebookresearch/MobileLLM): exact 125M config (30 layers, dim ~576?), to verify.
  - PleIAs / Vietnamese small-model efforts, and the TinyStories-Vietnamese analogues: not searched.

## Looked for but not found
- A primary benchmark of tokens/sec for 10M–125M GPT training on RTX 3090/4090 (only community/AI-generated estimates).
- A primary tok/s figure for PyTorch MPS on M-series for a ~124M model.
- Official SmolLM2-360M token count and official Llama 3.2 1B Meta card (huggingface.co blocked).
- A direct source for the Chinchilla 20 tokens/param figure (not searched due to budget).
- An independent replication of MobileLLM's depth-vs-width finding.
