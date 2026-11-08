# D3 — Vietnamese specifics (round 1, digest 1)

Accessed: 2026-11-08. Retrieval note: huggingface.co, arxiv.org, research.vinai.io and blog.arcee.ai did not resolve from this session (network allowlist). Claims marked "via search snippet" come from WebSearch result summaries of those primary sources, not from a direct fetch; confidence is lowered accordingly. github.com fetches worked directly.

## Claims

Format: claim | source title + URL | publisher | pub_date | accessed | confidence | class

### Existing Vietnamese LLMs

1. PhoGPT-4B base has 3.7B parameters and was pre-trained **from scratch** on a 102B-token Vietnamese corpus, 8192 context, "2 training epochs on 482GB of texts". | PhoGPT README, https://github.com/VinAIResearch/PhoGPT | VinAI Research | 2023-11 | 2026-11-08 | high (direct fetch) | size/scale
2. PhoGPT uses a vocabulary of 20K token types (20,480 per the tech report; Vietnamese-specific byte-level BPE trained with the HF `tokenizers` library). | README (above, "20K") + "PhoGPT: Generative Pre-training for Vietnamese", https://arxiv.org/pdf/2311.02945 (exact 20,480 and byte-level BPE via search snippet) | VinAI Research | 2023-11 | 2026-11-08 | high for ~20K; medium for 20,480/byte-level BPE | practice
3. PhoGPT's corpus includes about 1GB of Vietnamese Wikipedia (dump 20/05/2023). Wikipedia is therefore about 0.2% of a 482GB corpus. | arXiv 2311.02945 v2, https://arxiv.org/pdf/2311.02945v2.pdf (via search snippet) | VinAI Research | 2023-11 | 2026-11-08 | medium | size/scale
4. The PhoGPT repo code is BSD-3-Clause. The README states no separate weights license. | https://github.com/VinAIResearch/PhoGPT | VinAI Research | 2023-11 | 2026-11-08 | medium (weights license unverified) | license
5. PhoGPT-4B-Chat is fine-tuned on 70K instruction-response pairs plus 290K conversations. | https://github.com/VinAIResearch/PhoGPT | VinAI Research | 2023-11 | 2026-11-08 | high | practice
6. Vistral-7B is Mistral-7B with an **extended tokenizer**, then continual pretraining and instruction tuning on Vietnamese. The exact count of added tokens was not retrieved. | Vistral-7B-Chat model card, https://huggingface.co/Viet-Mistral/Vistral-7B-Chat (via search snippet; gated) | Viet-Mistral (UONLP x Ontocord) | 2024-01 (approx.) | 2026-11-08 | medium | practice
7. vi-mistral-x (a related but separate project) added 8,096 tokens to Mistral-7B's vocabulary, initialized the new embeddings as the mean of existing embeddings, and continued pretraining on CulturaX/vi (~8.3B tokens). | "Vi-Mistral-X", https://arxiv.org/html/2403.15470v1 (via search snippet) | arXiv | 2024-03 | 2026-11-08 | medium | practice
8. VinaLLaMA-7B is LLaMA-2 with an improved Vietnamese tokenizer and an additional 800B trained tokens. The public part is about 330B tokens: a CulturaX Vietnamese subset plus 100B English tokens. The in-house part is more than 500B synthetic Vietnamese tokens generated with GPT-4 and Vietcuna-3B-v2. A 2.7B variant was made by structured pruning. | "VinaLLaMA: LLaMA-based Vietnamese Foundation Model", https://arxiv.org/abs/2312.11011 (via search snippet) | arXiv / VILM | 2023-12 | 2026-11-08 | medium | size/scale
9. SeaLLM v1 (Llama-2 base) added about 16K SEA-language tokens to Llama-2's 32K vocabulary. Later versions moved to other bases: v2 on Mistral-7B (2024-02), v2.5 on Gemma-7B (2024-04), and v3 (2024-07). | SeaLLMs README, https://github.com/DAMO-NLP-SG/SeaLLMs | DAMO-NLP-SG (Alibaba) | 2023-12 to 2024-07 | 2026-11-08 | high | practice
10. SeaLLMs are released under custom "SeaLLMs Terms Of Use", not a standard OSS license. | https://github.com/DAMO-NLP-SG/SeaLLMs | DAMO-NLP-SG | 2023-12 | 2026-11-08 | medium | license
11. The URA-LLaMa family has 7B, 13B and 70B models fine-tuned from LLaMA-2, plus MixSUra (Mixtral) and GemSUra 7B (Gemma). They were trained on Vietnamese Wikipedia (1GB) and a Vietnamese news corpus (22GB). | "Crossing Linguistic Horizons", https://arxiv.org/html/2403.02715v2 + https://www.comp.nus.edu.sg/~nqduc/projects/vietnamese_llm (via search snippet) | HCMUT / Stanford / NUS | 2024-03 | 2026-11-08 | medium | size/scale
12. Arcee-VyLinh (~3B) is built from Qwen2.5-3B-Instruct; its mergekit config names that as the merge base. Its license is unclear: an aggregator lists Apache-2.0, while the Qwen2.5-3B lineage carries a qwen-research licence. | https://huggingface.co/arcee-ai/Arcee-VyLinh + https://featherless.ai/models/arcee-ai/Arcee-VyLinh (via search snippet) | Arcee AI | 2024-11 (approx.) | 2026-11-08 | low | license
13. In practice, almost every Vietnamese LLM is continued pretraining or fine-tuning of an English-centric base (Llama-2, Mistral, Gemma, Qwen). The VMLU leaderboard lists only two "from-scratch" models: ViGPT-1.6B-v1 (Vin BigData, 42.34) and PhoGPT-7B5-Instruct (24.01). | VMLU README leaderboard, https://github.com/ZaloAI-Jaist/VMLU | Zalo AI / JAIST | 2024-01 to 2024-05 (eval dates) | 2026-11-08 | high | practice

### Tokenization

14. Vietnamese compression ratio (tokens relative to a baseline, lower is better) is 4.41 for ChatGPT's tokenizer, 3.46 for Llama's and 1.48 for SeaLLM's extended vocabulary. English-centric tokenizers therefore use about 2-3x more tokens on Vietnamese. | https://github.com/DAMO-NLP-SG/SeaLLMs | DAMO-NLP-SG | 2023-12 | 2026-11-08 | high (figures direct; exact metric definition not retrieved) | practice
15. Across six 7-9B open models, Vietnamese/English fertility ratios range from 0.83x (VinaLLaMA, Vietnamese-native vocabulary) to 1.93x (Mistral 7B). Vocabulary composition matters as much as size: a 65K English-centric vocabulary did worse on Vietnamese than a 32K one. | "The Tokenization Tax: How Vietnamese Typology Exposes Vocabulary Design Failures in Open-Source LLMs", https://neurips.cc/virtual/2026/164760 (abstract via search snippet) | NeurIPS 2026 | 2026 | 2026-11-08 | low-medium (abstract only) | practice
16. Fertility alone can hide over-fragmentation. The Single Token Retention Rate (STRR, the share of words kept as one token) is proposed as a complementary metric. | "Beyond Fertility: Analyzing STRR...", https://arxiv.org/html/2510.09947v2 (via search snippet) | arXiv | 2025-10 | 2026-11-08 | medium | practice
17. Word segmentation: PhoBERT (an encoder) requires VnCoreNLP RDRSegmenter word-segmented input, with BPE applied after segmentation. Its authors argue syllable-level models are weaker on word-level tasks. This is an encoder/word-level-task argument. Generative PhoGPT uses a byte-level BPE over raw text (claim 2), with no evidence retrieved that it word-segments its input. | PhoBERT, https://github.com/VinAIResearch/PhoBERT mirror / https://arxiv.org/pdf/2003.00744v2 (via search snippet) | VinAI | 2020-03 | 2026-11-08 | medium | practice

### Evaluation

18. VMLU has 10,880 four-option multiple-choice questions across 58 subjects in STEM, Social Science, Humanities and Other, from elementary to professional level. It reports zero-shot and five-shot accuracy, with submissions made through vmlu.ai. The repo sidebar says MIT, but the README licence section says "TBU". | https://github.com/ZaloAI-Jaist/VMLU | Zalo AI / JAIST | 2024-01 (approx.) | 2026-11-08 | high | benchmark
19. VMLU zero-shot scores for the models above: Vistral-7B-Chat 50.07, vietcuna-3b 34.79, ura-llama-7b 32.18, vinallama-7b-chat 30.64, PhoGPT-7B5-Instruct 24.01. Random chance is 25%, so a 3.7-7.5B from-scratch model scored close to chance. | https://github.com/ZaloAI-Jaist/VMLU | Zalo AI / JAIST | 2024 | 2026-11-08 | high | benchmark
20. ViLLM-Eval was built for the VLSP 2023 Vietnamese LLM challenge. It contains multiple-choice tasks and **predict-next-word** tasks across disciplines, and its data and code are published under the HF org `vlsp-2023-vllm`. | "ViLLM-Eval", https://huggingface.co/papers/2404.11086 / https://arxiv.org/abs/2404.11086 (via search snippet) | VLSP / arXiv | 2024-04 | 2026-11-08 | medium | benchmark
21. "Crossing Linguistic Horizons" evaluates Vietnamese LLMs on 10 scenarios and 31 metrics. It finds training and fine-tuning data quality to be the key performance factor. | https://arxiv.org/html/2403.02715v2 (via search snippet) | arXiv | 2024-03 | 2026-11-08 | medium | benchmark
22. SeaLLM reports M3Exam-Vietnamese 3-shot accuracy: Llama-2-7b 29.82, SeaLLM-7b 38.74, SeaLLM-13b-5L 49.13 and ChatGPT 58.64. | https://github.com/DAMO-NLP-SG/SeaLLMs | DAMO-NLP-SG | 2023-12 | 2026-11-08 | high | benchmark

### Inference for a tiny (<200M) model (analysis, not a sourced claim)
- Claim 19 shows that even a 3.7-7.5B from-scratch Vietnamese model sits near chance on VMLU. A model under 200M parameters will almost certainly be at chance on VMLU-style MCQ. Usable evals are held-out perplexity or bits-per-byte/character, plus cloze or next-word tasks such as ViLLM-Eval's predict-next-word component (claim 20). Use a tokenizer-independent metric (bits per byte) when comparing against other tokenizers.
- Wikipedia is small relative to these corpora: ~1GB in PhoGPT's 482GB (claim 3) and in URA-LLaMa's ~23GB (claim 11).

## Leads
- PhoGPT tech report, sec. 2.1: full corpus breakdown and tokenizer training details (arxiv 2311.02945; blocked here).
- Vistral paper/model card: the exact number of added tokens and their initialization (huggingface.co gated; blocked).
- VinaLLaMA tokenizer vocab size (arxiv 2312.11011).
- NeurIPS 2026 "Tokenization Tax" full tables: per-model Vietnamese fertility, including Llama-3, Qwen2.5 and GPT-4o.
- `vlsp-2023-vllm` HF org: the actual predict-next-word (LAMBADA-like) dataset, a candidate eval for a tiny model.
- ViGPT-1.6B (Vin BigData): a from-scratch 1.6B model at 42.34 VMLU, much higher than PhoGPT. Its data/token count is worth looking up.
- Qwen2.5/Qwen3 and Llama-3 official Vietnamese support claims (not searched this round).

## Looked for but not found
- Exact Vistral vocabulary extension size.
- Arcee-VyLinh training recipe, data and authoritative licence (blog.arcee.ai and HF blocked).
- Any evidence that a generative Vietnamese BPE LLM uses word segmentation (VnCoreNLP/underthesea) before tokenization. None found. PhoGPT appears not to.
- ViGLUE details and the VLSP shared-task list (not searched; budget).
- Explicit licence text for PhoGPT weights and Vistral.
- Direct Vietnamese fertility numbers for Llama-3/GPT-4o tokenizers.
