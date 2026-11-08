---
title: 'academic-lit research: Why LLMs answer accurately despite only predicting the next token'
type: 'academic-lit'
topic: 'Why LLMs answer accurately despite next-token prediction'
decision: 'What to keep, fix, and add in docs/why_llms_are_accurate.vi.md'
source: 'native run (4 web-search assistants, 1 round each)'
status: complete
preset: 'standard'
validation: 'normal'
created: '2026-11-08'
updated: '2026-11-08'
claims_verified: 9
claims_unverified: 7
claims_disputed: 1
---

# Literature review: Why LLMs answer accurately despite only predicting the next token

**Decision this research serves:** what to keep, fix, and add in
`docs/why_llms_are_accurate.vi.md`.

> **Limits of this run:** the session's network blocked `arxiv.org` and
> `transformer-circuits.pub`. Most figures come from proceedings pages,
> publisher pages, and search-result excerpts of the original pages, not from
> reading the full PDFs. Those figures are marked with *medium* confidence. Each
> branch ran only 1 round, and no independent assistant has cross-checked them.

## Executive summary

**Conclusion:** the document's main thesis holds. Predicting tokens well forces
the model to learn internal structure, and this has empirical evidence. But
many places need an **added condition** or a **fix** to avoid overstating.

Three findings dominate the conclusion:

1. **The evidence for internal structure is real, but incomplete.**
   Othello, chess, modular addition, and the circuits in Claude 3.5 Haiku all
   show that the model computes rather than merely looks things up
   [12][14][17][24]. On the other hand, a model can pass probe-based tests while
   its internal world model remains inconsistent [15]. Anthropic's method also
   captures only "a part" of the full computation [24].
2. **Post-training and test-time compute are the main source of accuracy in
   modern models.** Examples: InstructGPT 1.3B is preferred over GPT-3 175B
   [26]; DeepSeek-R1 (Nature 2025) learned self-verification from RL with
   correct/incorrect rewards alone [27]. **However**, whether RL teaches *new*
   reasoning or merely re-selects what the base model already has is a debate
   that remains **open** [29][30].
3. **Some statements in the document are overstated.** "LLMs compress images
   better than PNG" holds only when *model size is not counted* [1]. "Transformer
   ≈ TC0" must state the assumption about numerical precision [32][33].
   Chain-of-thought raises accuracy, but the CoT text *does not faithfully
   reflect* how the model actually computes (only about 25–39%) [37].

**Biggest caveat:** many results on how the model works internally come from a
single lab, on a single model, and are not peer-reviewed (Anthropic)
[23][24][25]. Treat them as strong evidence that has not yet been independently
replicated.

## D1 — Theoretical foundations

- **Prediction is equivalent to compression.** Under arithmetic coding, the
  log-loss is exactly the compressed code length. Delétang et al. (ICLR 2024,
  peer-reviewed): Chinchilla 70B compresses ImageNet images to 43.4% (PNG
  58.5%) and LibriSpeech audio to 16.4% (FLAC 30.3%) [1]. **Condition:** these
  are "raw" ratios that do not count the model size. If the model is counted,
  the LLM compresses much worse. Confidence: high for the figures, medium for
  the condition.
- **Compression correlates with capability.** Across about 30 LLMs, the average
  benchmark score is nearly linear in compression efficiency (r ≈ −0.95; Huang
  et al., COLM 2024) [2]. This is a correlation and does not prove
  "intelligence". Confidence: medium.
- **Scaling.** Chinchilla (NeurIPS 2022): 70B parameters, 1.4T tokens, i.e.
  about 20 tokens per parameter. The figure "20" is a ratio inferred from the
  paper, not a law the paper states. Epoch AI replicated it (preprint 2024) and
  found an error in the way the 3rd fit was done, but the ratio of about 20
  still stands [3]. Important updates:
  - Training beyond Chinchilla is deliberate, to reduce inference cost (Sardana
    et al., ICML 2024) [4].
  - Repeating data up to about 4 epochs loses almost nothing (Muennighoff et
    al., NeurIPS 2023) [5].
  - Test-time compute lets a small model beat a model 14 times larger, but
    **only on easy and medium questions**. On the hardest questions,
    pretraining still wins (Snell et al., ICLR 2025 oral) [6].
- **Emergence: the debate is unsettled.**
  - Wei et al. (TMLR 2022) defined emergent abilities [7].
  - Schaeffer et al. (NeurIPS 2023, award) argue that the "suddenness" comes
    from nonlinear metrics [8]. The award category comes from a secondary
    source, confidence medium.
  - Du et al. (NeurIPS 2024) push back: in terms of pre-training loss, some
    abilities appear only once the loss crosses a threshold, even with
    continuous metrics [9].
- **Theory on the power of prediction.**
  - Malach proves that a next-token predictor is a "universal learner" **when
    trained on chain-of-thought data**, not on pure web pretraining [10]. The
    venue (ICML 2024?) is not confirmed.
  - Bachmann & Nagarajan is the counterpoint: teacher forcing can fail even on
    simple lookahead problems [11].

## D2 — Evidence from looking inside the model

- **Othello-GPT** (Li et al., ICLR 2023): it has a representation of the board
  state, and intervening on it changes the move [12]. **Fix:** the original
  paper found that *nonlinear* probes work better. The representation is
  *linear* only when squares are encoded as "của tôi / của đối thủ" (Nanda, Lee
  & Wattenberg, BlackboxNLP 2023) [13].
- **Chess** (Karvonen 2024, preprint): the board can be read with a linear probe
  at accuracy up to about 99.6%. Adding a "skill vector" raises the win rate by
  up to 2.6 times [14].
- **Counterpoint:** a model can pass probes and next-token validity checks while
  its implicit world model remains inconsistent (Vafa et al., NeurIPS 2024)
  [15]. *This is an important addition to the document.*
- **Grokking:** Power et al. (2022) is an ICLR 2021 workshop paper with a
  preprint, not a main-track paper [16]. Nanda et al. (ICLR 2023) confirm that a
  Fourier algorithm and trigonometric identities implement modular addition
  [17].
- **Space and time** (Gurnee & Tegmark, ICLR 2024): they confirm linear probes
  on Llama-2. The authors themselves say this is **not yet** a causal world
  model [18].
- **Induction heads** (Olsson et al. 2022, preprint): causal evidence exists
  only in small attention-only models. For large models it is only correlational
  [19].
- **MLP as key–value memory** (Geva et al., EMNLP 2021) [20]. **ROME** (Meng et
  al., NeurIPS 2022) [21]. **Superposition** (Elhage et al. 2022, preprint)
  [22]. **SAE** with about 34 million features on Claude 3 Sonnet (2024,
  preprint) [23].
- **Claude 3.5 Haiku** (Anthropic, 3/2025) [24]:
  - Two-step reasoning Dallas → Texas → Austin. Replacing "Texas" with
    "California" yields Sacramento.
  - It picks the rhyming word "rabbit" in advance and only then writes the line
    leading up to it.
  - It adds 36+59 through parallel paths: one path estimates the magnitude, one
    computes the last digit exactly. But when asked, the model **describes** the
    carry method in a textbook style, which does not match how it actually
    computes.
  - Self-stated limits: the method captures only part of the computation, one
    model, not peer-reviewed.
- **Introspection 2026** (Lindsey, preprint): the model sometimes detects a
  concept injected into its activations (about 20% with Opus 4.1). The author
  calls this ability "highly unreliable" [25].

## D3 — Post-training and reasoning

- **InstructGPT** (NeurIPS 2022): raters prefer the 1.3B model over GPT-3 175B
  [26]. The frequently cited figure "71.4%" is unverified.
- **DeepSeek-R1** was published in **Nature 645:633–638 (9/2025)**. Pure RL with
  correct/incorrect rewards, with no human-written reasoning traces, still
  produces self-verification and reflection [27].
- **o1** (2024): accuracy rises roughly log-linearly with both train-time RL
  compute and answer-time compute [28]. Secondary source, confidence medium.
- **Does RLVR teach new reasoning? Open debate.**
  - Yue et al. (NeurIPS 2025 oral): RL raises pass@1, but the base model wins at
    pass@k with large k. That means RL mostly re-selects existing capability
    [29].
  - ProRL (NeurIPS 2025): prolonged RL pushes pass@k beyond the base model [30].
  - The pass@k metric with large k is itself also disputed.
- **Theory of CoT:**
  - Feng et al. (NeurIPS 2023) [31].
  - Merrill & Sabharwal (ICLR 2024): without CoT it is bounded above by TC0; with
    O(log n) steps, by L; with O(n) steps it recognizes all regular languages;
    with poly(n) steps it is exactly **P** [32].
  - Li et al. (ICLR 2024): with constant precision, without CoT it gets only
    **AC0**. TC0 is the upper bound when precision is logarithmic. With T CoT
    steps it can solve circuits of size T [33].
- **Calibration:**
  - Kadavath et al. (preprint 2022): models "mostly know what they know" [34].
  - GPT-4 report: the base model is well calibrated, while RLHF *reduces*
    calibration [35]. *Needs to be added to the document.*
- **Tools:** Toolformer (NeurIPS 2023) [36].
- **CoT faithfulness** (Chen et al. 2025, Anthropic, preprint): a reasoning model
  mentions a hint it actually used in only about 25% (Claude 3.7 Sonnet) to 39%
  (DeepSeek-R1) of cases [37].

## D4 — Limits and debates

- **Hallucination** (Kalai et al. 2025, OpenAI, preprint) [38]:
  - The hallucination rate has a lower bound of roughly the "singleton rate",
    i.e. the fraction of facts that appear exactly once in the training data.
  - Benchmarks scored 0/1 reward guessing over abstaining. Proposal: set a
    confidence threshold t, and penalize a wrong answer by t/(1−t).
  - The lower bound applies only to facts with no regularity, so hallucination
    is **not inevitable** in general.
  - Kalai & Vempala (STOC 2024): a calibrated model is forced to hallucinate on
    this kind of fact [39].
- **Reversal curse** (Berglund et al., ICLR 2024): GPT-4 is right 79% of the time
  in the forward direction, 33% in the reverse direction [40]. "Reverse
  training" (Golovneva et al. 2024) reduces or eliminates the phenomenon, so it
  is **not a fundamental limit** [41].
- **Embers of autoregression** (McCoy et al., **PNAS 2024**): GPT-4 decodes a
  cipher correctly 51% of the time when the answer is a high-probability
  sentence, only 13% when the probability is low [42]. With o1 the effect shrinks
  but does not disappear (preprint) [43].
- **Faith and Fate** (Dziri et al., NeurIPS 2023 spotlight): confirms the
  "linearized subgraph matching" finding [44].
- **Unfaithful CoT:**
  - Turpin et al. (NeurIPS 2023): accuracy drops by up to 36% when a bias skews
    the answer without the CoT mentioning it [45].
  - Lanham et al. (preprint): the larger the model, the less faithful its CoT
    tends to be [46].
- **The "just pattern matching" debate:**
  - GSM-Symbolic: accuracy drops by up to 65% when an irrelevant clause is added
    (GSM-NoOp) [47]. The ICLR 2025 venue is not confirmed. A 2026 preprint
    rebuts the statistics [48].
  - "Illusion of Thinking" (Apple 2025) is rebutted by a non-peer-reviewed
    comment: output token limits, and some River Crossing instances have no
    solution [49].
- **Tokenization:** the evidence is mixed. CharBench finds that letter-counting
  errors depend on how many times the character occurs more than on
  tokenization [50][51]. Write "a contributing factor", not "the cause".

## Cross-dimension insights

1. **"Prediction" never stands alone.** The strongest theory on the power of
   next-token prediction (Malach [10], CoT reaching P [32]) all *requires
   intermediate tokens*. The strongest experiments (R1 [27], Snell [6]) all *add
   RL or test-time compute*. So the full answer is: prediction, plus context as
   scratch paper, plus optimization toward the correct answer.
2. **The same mechanism explains both right and wrong.** Accuracy tracks the
   probability in the data (embers [42], singleton rate [38]). That explains both
   why common facts are answered correctly and why rare facts are made up.
3. **The model's own explanation is not evidence.** Mechanistic interpretation
   (36+59 [24]) and measurements of CoT faithfulness [37][45] give the same
   conclusion: what the model *says* about how it works differs from what it
   *does*.

## Recommendations for `docs/why_llms_are_accurate.vi.md`

| # | Change | Basis | Confidence |
|---|---|---|---|
| 1 | Add the condition "not counting model size" to the comparison with PNG/FLAC | [1] | high |
| 2 | Othello: note the "của tôi / của đối thủ" encoding; add the Vafa et al. counterpoint | [13][15] | high |
| 3 | Grokking: Power et al. is a workshop paper | [16] | medium |
| 4 | Induction heads: causal evidence only in small models | [19] | medium |
| 5 | Add the 36+59 example and the story of the model misdescribing how it computes | [24] | medium (one lab) |
| 6 | Emergence: add Du et al. 2024, state clearly that it is still debated | [9] | high |
| 7 | Chinchilla: "about 20" is an inferred ratio; add over-training and a note that test-time compute wins only on easy/medium questions | [3][4][6] | high |
| 8 | State that R1 was published in Nature; add the RLVR debate (Yue vs ProRL) | [27][29][30] | high |
| 9 | Fix the TC0 paragraph: it needs an assumption about precision; poly CoT steps = P | [32][33] | high |
| 10 | Add: RLHF reduces calibration | [35] | medium |
| 11 | Add a section on CoT faithfulness (25–39%) | [37][45] | medium |
| 12 | Hallucination: add the singleton rate and the t/(1−t) penalty proposal; the reversal curse can be reduced by reverse training | [38][41] | medium |
| 13 | Embers: state PNAS 2024 and the 51% / 13% figures | [42] | high |
| 14 | Tokenization: change "cause" to "contributing factor" | [50][51] | medium |

## Open questions

- The "adjusted" ratio (counting the model) of Delétang et al. and the 2048-byte
  data chunk size: need to read the PDF (requires opening access to arxiv.org).
- The venues of Malach, Bachmann & Nagarajan, Karvonen, and GSM-Symbolic.
- The venues of Wei 2022 (CoT), Kojima 2022, self-consistency, and RAG were not
  checked in this run.
- Anthropic's results have not been replicated by any independent lab.
- RLVR: awaiting further pass@k results after ProRL.

## Source appendix

| [n] | Supports | Publisher / venue | Date | Accessed | Confidence |
|---|---|---|---|---|---|
| [1] | Image/audio compression vs PNG/FLAC | [ICLR 2024 — Delétang et al.](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3cbf627fa24fb6cb576e04e689b9428b-Abstract-Conference.html) | 2024-05 | 2026-11-08 | high |
| [2] | Compression correlates linearly with benchmarks | [arXiv / COLM 2024 — Huang et al.](https://arxiv.org/abs/2404.09937) | 2024-04 | 2026-11-08 | medium |
| [3] | Chinchilla replication | [Epoch AI — Besiroglu et al.](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt) | 2024-04 | 2026-11-08 | medium |
| [4] | Over-training for inference cost | [arXiv / ICML 2024 — Sardana et al.](https://arxiv.org/abs/2401.00448) | 2023-12 | 2026-11-08 | medium |
| [5] | Repeating data for about 4 epochs | [NeurIPS 2023 — Muennighoff et al.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract.html) | 2023-12 | 2026-11-08 | high |
| [6] | Test-time compute vs a 14x larger model | [ICLR 2025 — Snell et al.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html) | 2025-04 | 2026-11-08 | high |
| [7] | Definition of emergence | [TMLR — Wei et al.](https://openreview.net/forum?id=yzkSU5zdwD) | 2022-08 | 2026-11-08 | high |
| [8] | "Mirage" | [The Decoder (secondary) — Schaeffer et al.](https://the-decoder.com/neurips-2023-these-are-the-top-papers-and-award-winners-at-the-largest-ai-conference/) | 2023-12 | 2026-11-08 | medium |
| [9] | Emergence in terms of loss | [NeurIPS 2024 — Du et al.](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5f1eee2509599faeeb3570a887016a64-Abstract.html) | 2024-12 | 2026-11-08 | high |
| [10] | Universal learner with CoT data | [arXiv — Malach](https://arxiv.org/abs/2309.06979) | 2023-09 | 2026-11-08 | medium |
| [11] | Drawbacks of next-token prediction | [arXiv — Bachmann & Nagarajan](https://arxiv.org/abs/2403.06963) | 2024-03 | 2026-11-08 | medium |
| [12] | Othello-GPT | [arXiv / ICLR 2023 — Li et al.](https://arxiv.org/abs/2210.13382) | 2022-10 | 2026-11-08 | high |
| [13] | Linear representation "của tôi / của đối thủ" | [BlackboxNLP 2023 — Nanda, Lee, Wattenberg](https://aclanthology.org/2023.blackboxnlp-1.2) | 2023-12 | 2026-11-08 | high |
| [14] | Chess, linear probe | [arXiv — Karvonen](https://arxiv.org/abs/2403.15498) | 2024-03 | 2026-11-08 | medium |
| [15] | Inconsistent world model | [arXiv / NeurIPS 2024 — Vafa et al.](https://arxiv.org/abs/2406.03689) | 2024-06 | 2026-11-08 | high |
| [16] | Grokking | [arXiv (ICLR 2021 workshop) — Power et al.](https://arxiv.org/abs/2201.02177) | 2022-01 | 2026-11-08 | medium |
| [17] | Fourier algorithm for modular addition | [arXiv / ICLR 2023 — Nanda et al.](https://arxiv.org/abs/2301.05217) | 2023-01 | 2026-11-08 | high |
| [18] | Space and time | [arXiv / ICLR 2024 — Gurnee & Tegmark](https://arxiv.org/abs/2310.02207) | 2023-10 | 2026-11-08 | high |
| [19] | Induction heads | [arXiv — Olsson et al.](https://arxiv.org/abs/2209.11895) | 2022-09 | 2026-11-08 | medium |
| [20] | MLP as key–value memory | [arXiv / EMNLP 2021 — Geva et al.](https://arxiv.org/abs/2012.14913) | 2020-12 | 2026-11-08 | high |
| [21] | ROME | [arXiv / NeurIPS 2022 — Meng et al.](https://arxiv.org/abs/2202.05262) | 2022-02 | 2026-11-08 | high |
| [22] | Superposition | [arXiv — Elhage et al.](https://arxiv.org/abs/2209.10652) | 2022-09 | 2026-11-08 | medium |
| [23] | SAE with about 34 million features | [Transformer Circuits — Templeton et al.](https://transformer-circuits.pub/2024/scaling-monosemanticity/) | 2024-05 | 2026-11-08 | medium |
| [24] | Circuits in Claude 3.5 Haiku | [Anthropic — Tracing the thoughts of a LLM](https://www.anthropic.com/research/tracing-thoughts-language-model) | 2025-03 | 2026-11-08 | medium |
| [25] | Introspection | [arXiv — Lindsey](https://arxiv.org/abs/2601.01828) | 2026-01 | 2026-11-08 | medium |
| [26] | InstructGPT 1.3B > GPT-3 175B | [arXiv / NeurIPS 2022 — Ouyang et al.](https://arxiv.org/abs/2203.02155) | 2022-03 | 2026-11-08 | high |
| [27] | DeepSeek-R1 in Nature | [Nature — DeepSeek-AI](https://www.nature.com/nature-index/article/10.1038/s41586-025-09422-z) | 2025-09 | 2026-11-08 | high |
| [28] | o1, scaling with compute | [OpenAI — Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) | 2024-09 | 2026-11-08 | medium |
| [29] | RLVR mostly re-selects base capability | [arXiv / NeurIPS 2025 — Yue et al.](https://arxiv.org/abs/2504.13837) | 2025-04 | 2026-11-08 | high |
| [30] | ProRL beats base at pass@k | [arXiv / NeurIPS 2025 — Liu et al.](https://arxiv.org/abs/2505.24864) | 2025-05 | 2026-11-08 | medium |
| [31] | Theory of CoT | [NeurIPS 2023 — Feng et al.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract.html) | 2023-12 | 2026-11-08 | high |
| [32] | CoT: TC0 → L → regular → P | [ICLR 2024 — Merrill & Sabharwal](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1f59721c106ea80f613299039112f651-Abstract-Conference.html) | 2024-05 | 2026-11-08 | high |
| [33] | AC0 vs TC0, CoT solves sequential problems | [ICLR 2024 — Li et al.](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3309b4112c9f04a993f2bbdd0274bba1-Abstract-Conference.html) | 2024-05 | 2026-11-08 | high |
| [34] | Models know what they know | [arXiv — Kadavath et al.](https://arxiv.org/abs/2207.05221) | 2022-07 | 2026-11-08 | medium |
| [35] | RLHF reduces calibration | [arXiv — OpenAI GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) | 2023-03 | 2026-11-08 | medium |
| [36] | Toolformer | [NeurIPS 2023 — Schick et al.](https://proceedings.neurips.cc/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract.html) | 2023-12 | 2026-11-08 | high |
| [37] | CoT faithfulness 25–39% | [Anthropic — Chen et al.](https://www.anthropic.com/research/reasoning-models-dont-say-think) | 2025-04 | 2026-11-08 | medium |
| [38] | Singleton rate, t/(1−t) penalty | [arXiv — Kalai et al. (OpenAI)](https://arxiv.org/abs/2509.04664) | 2025-09 | 2026-11-08 | medium |
| [39] | Calibrated models must hallucinate | [arXiv / STOC 2024 — Kalai & Vempala](https://arxiv.org/abs/2311.14648) | 2023-11 | 2026-11-08 | high |
| [40] | Reversal curse 79% / 33% | [arXiv / ICLR 2024 — Berglund et al.](https://arxiv.org/abs/2309.12288) | 2023-09 | 2026-11-08 | high |
| [41] | Reverse training | [arXiv — Golovneva et al.](https://arxiv.org/abs/2403.13799) | 2024-03 | 2026-11-08 | medium |
| [42] | Embers, 51% / 13% | [PNAS — McCoy et al.](https://doi.org/10.1073/pnas.2322420121) | 2024-10 | 2026-11-08 | high |
| [43] | Embers with o1 | [arXiv — McCoy et al.](https://arxiv.org/abs/2410.01792) | 2024-10 | 2026-11-08 | medium |
| [44] | Linearized subgraph matching | [NeurIPS 2023 — Dziri et al.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/deb3c28192f979302c157cb653c15e90-Abstract.html) | 2023-12 | 2026-11-08 | high |
| [45] | Unfaithful CoT, drop of up to 36% | [arXiv / NeurIPS 2023 — Turpin et al.](https://arxiv.org/abs/2305.04388) | 2023-05 | 2026-11-08 | high |
| [46] | Larger model, less faithful CoT | [arXiv — Lanham et al.](https://arxiv.org/abs/2307.13702) | 2023-07 | 2026-11-08 | medium |
| [47] | GSM-Symbolic, drop of up to 65% | [arXiv — Mirzadeh et al.](https://arxiv.org/abs/2410.05229) | 2024-10 | 2026-11-08 | medium |
| [48] | Statistical rebuttal of GSM-Symbolic | [arXiv — 2026 preprint](https://arxiv.org/abs/2605.28700) | 2026-05 | 2026-11-08 | low |
| [49] | Rebuttal of "Illusion of Thinking" | [arXiv — Lawsen](https://arxiv.org/abs/2506.09250) | 2025-06 | 2026-11-08 | low |
| [50] | Why LLMs count letters poorly | [arXiv — Fu et al.](https://arxiv.org/abs/2412.18626) | 2024-12 | 2026-11-08 | medium |
| [51] | CharBench | [arXiv — CharBench](https://arxiv.org/abs/2508.02591) | 2025-08 | 2026-11-08 | medium |

## Staleness map

Time windows per pack: SOTA results in ML and ongoing debates should be rechecked
after 6 months; theory after 24 months; foundational work has no expiry.
Results computed by script are in `staleness.json`.

| Claim | Date | Recheck | Stale? |
|---|---|---|---|
| Claude 3.5 Haiku circuits [24] | 2025-03 | 2025-09 | yes |
| RLVR only re-selects capability [29] | 2025-04 | 2025-10 | yes |
| Test-time compute vs 14x model [6] | 2025-04 | 2025-10 | yes |
| CoT faithfulness 25–39% [37] | 2025-04 | 2025-10 | yes |
| DeepSeek-R1 Nature [27] | 2025-09 | 2026-03 | yes |
| GSM-Symbolic rebuttal [48] | 2026-05 | 2026-11 | soon |
| Hallucination lower bound [38] | 2025-09 | 2027-09 | no |

**The earliest milestone (2025-09) has passed:** by the ML 6-month standard, most
SOTA claims were already "stale" at the time of the run. Run a focused
**Refresh** on D2 and D3 (2026 mechanistic interpretation, RLVR after ProRL),
ideally after opening access to arxiv.org.
