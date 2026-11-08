# Why do LLMs answer correctly even though they are "just predicting the next token"?

*[Bản tiếng Việt](why_llms_are_accurate.vi.md)*

*Research document. Directly tied to the code in this repo:
[`src/ntokenizer/model.py`](../src/ntokenizer/model.py) (the `forward` and `generate` functions).*

---

## Table of contents

1. [The question](#1-the-question)
2. [What the algorithm actually is](#2-what-the-algorithm-actually-is)
3. [Core claim: good prediction = understanding structure](#3-core-claim-good-prediction--modelling-the-structure-that-generates-the-data)
4. [Empirical evidence: what is inside the model](#4-empirical-evidence-what-is-inside-the-model)
5. [Why "probability" yields confident answers](#5-why-probability-yields-confident-answers)
6. [Scale: why bigger models are more "correct"](#6-scale-why-bigger-models-are-more-correct)
7. [From "predicting text" to "answering correctly": post-training](#7-from-predicting-text-to-answering-correctly-post-training)
8. [Multi-step reasoning: chain-of-thought as a "scratchpad"](#8-multi-step-reasoning-chain-of-thought-as-a-scratchpad)
9. [The flip side: when and why LLMs are wrong](#9-the-flip-side-when-and-why-llms-are-wrong)
10. [Experiments you can run with this repo](#10-experiments-you-can-run-with-this-repo)
11. [Summary](#11-summary)
12. [References](#12-references)

---

## 1. The question

An LLM is trained with **a single objective**: given a sequence of tokens,
guess the next token. There is no "logic" module, no database of facts,
no math solver. So why does it answer "Thủ đô của Việt Nam là
**Hà Nội**" (The capital of Vietnam is **Hanoi**) correctly, write code that
runs, and solve math problems?

The short answer: **"just prediction" describes the _interface_ (input →
probability distribution), not _what happens inside_.** To predict the next
token with low error over trillions of tokens of diverse text, the cheapest
way (in parameters) that gradient descent finds is to build internal
representations of grammar, facts, relations, and even algorithms. The rest
of this document explains and gives evidence for this claim.

---

## 2. What the algorithm actually is

In this repo, all of the "intelligence" is trained by exactly one line
(`model.py`, in `forward`):

```python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

That is, minimizing

$$\mathcal{L} = -\frac{1}{N}\sum_{t} \log p_\theta(x_t \mid x_{<t})$$

And when generating text (`generate`), the model just repeats: compute the
`logits` for the last position → (optionally) divide by `temperature`, filter
with `top_k`/`top_p` → `softmax` → sample one token → append it to the
sequence → repeat.

Two important points that are often misunderstood:

| Misconception | Reality |
|---|---|
| "Prediction = guessing / n-gram statistics" | The distribution $p(x_t\mid x_{<t})$ is computed by a deep network (many attention + MLP blocks) that looks at the **entire** context, not by counting phrase frequencies. |
| "Probability = uncertainty" | For questions with a clear answer, the distribution is usually very peaked: the correct token can hold >99% of the probability mass. Sampling from such a distribution is nearly deterministic. |

---

## 3. Core claim: good prediction ⇒ modelling the structure that generates the data

### 3.1. Prediction = compression = understanding (in the information-theoretic sense)

Shannon (1948, 1951) showed that a model that predicts well can be used to
**compress** data (arithmetic coding), and the optimal code length is exactly
the cross-entropy. Delétang et al. (2023, *Language Modeling Is Compression*)
showed that LLMs are extremely strong general-purpose compressors: Chinchilla
70B compresses ImageNet images to 43.4% (PNG 58.5%) and audio to 16.4%
(FLAC 30.3%) even though it was trained mainly on text. **Note:** these
ratios *do not count the model size*; once you count hundreds of GB of
weights, the LLM compresses much worse. Huang et al. (COLM 2024) also found,
across ~30 LLMs, that benchmark scores correlate almost linearly with
compression efficiency. That is a correlation, not a proof.

Consequence: to reduce the loss, the model must discover the **regularities**
that explain the data. Memorizing each sentence is poor compression;
understanding the regularities (grammar, addition, geographic relations) is
good compression.

### 3.2. An intuitive example

Consider sentences in the corpus:

```
"... 17 + 25 = 42 ..."
"Trong ván cờ, sau khi Trắng đi Mã f3, Đen ..."
"Hàm này trả về None vì biến x chưa được ..."
"Kết luận: kẻ sát nhân chính là ..."
```

(In order: "In the chess game, after White plays Knight f3, Black ...";
"This function returns None because the variable x has not been ...";
"Conclusion: the murderer is ...")

To predict the next token **correctly** in these sentences, the model needs
to (partly) know how to add, know the state of the chessboard, understand
code semantics, and follow the thread of reasoning in a story. There is no
"statistical shortcut" that gives low loss on **all** such cases. Ilya
Sutskever summarized this idea: *predicting the next token well enough means
having to understand the reality that generated that token*.

### 3.3. A simple objective ≠ a simple capability

A simple training objective does not limit the complexity of the solution.
Compare with evolution: the objective is "survive and reproduce", but the
solutions are eyes, brains, language. Next-token prediction is a **meta**
problem: it contains millions of sub-problems (translation, summarization,
addition, classification...) because text on the Internet contains them.

---

## 4. Empirical evidence: what is inside the model

This is the most important part — not just theory: people have **opened up
the models and seen** these structures (the field of *mechanistic
interpretability*).

| Finding | Study | Meaning |
|---|---|---|
| **World model**: GPT trained only on sequences of Othello moves (never seeing the board) nevertheless has an internal representation of the board state; intervening on that representation changes the predicted move. The original paper needed *nonlinear* probes; the representation is linear only when squares are encoded as "mine / theirs". Chess models also have a board readable by linear probes (~99.6%). | Li et al. (ICLR 2023); Nanda, Lee & Wattenberg (BlackboxNLP 2023); Karvonen 2024 | Sequence prediction ⇒ building the hidden state of the "world" on its own. **Counterpoint:** a model can pass probes while its world model is still inconsistent (Vafa et al., NeurIPS 2024). |
| **Real algorithms**: transformers learn modular addition using the Fourier transform + trigonometric identities, which can be read out of the weights. | Nanda et al. 2023 (*Progress measures for grokking*) | The model does not merely memorize a lookup table; it learns a **general algorithm**. |
| **Grokking**: at first the model memorizes the training set (poor test performance); with continued training it suddenly generalizes. | Power et al. 2022 (ICLR 2021 workshop + preprint) | Regularization + gradients favor the "compact" (algorithmic) solution over the "memorizing" one. |
| **Space & time**: Llama-2 has linear directions encoding the longitude/latitude of places and the year of events. | Gurnee & Tegmark 2023 | Knowledge is organized with structure, not as a list of disconnected items. |
| **Induction heads**: attention circuits that implement "I saw A B earlier, now I see A → predict B"; the main mechanism of in-context learning. | Olsson et al. 2022 (preprint) | Explains why the model can learn from examples in the prompt. Causal evidence exists only in small models; in large models it is correlational. |
| **Facts live in the MLP**: feed-forward layers act as key–value memories; a specific fact can be located and edited. | Geva et al. 2021; Meng et al. 2022 (*ROME*) | Corresponds to the `MLP`/SwiGLU in `model.py`. |
| **Internal multi-step reasoning & planning**: Claude 3.5 Haiku computes "the capital of the state containing Dallas" via an intermediate step "Texas" internally (swap Texas for California and the output becomes Sacramento); when writing poetry, it picks the rhyming word **first** and then writes the line leading to it. | Anthropic 2025 (*On the Biology of a Large Language Model*; not peer-reviewed, one model) | Even though output is token by token, internally there is "lookahead" over multiple tokens. The method captures only *part* of the computation. |
| **The model does not know how it computes**: for 36+59, Claude 3.5 Haiku runs in parallel a magnitude-estimation path and a last-digit path; but when asked, it describes the textbook "carry the 1" method. | Anthropic 2025 | A model's own explanation of itself is not evidence about its internal mechanism. |
| **Superposition**: the model compresses more concepts than it has dimensions by superimposing nearly orthogonal features. | Elhage et al. 2022 | Explains why a few billion parameters can hold a huge amount of knowledge. |

**Conclusion of section 4:** "Token prediction" is just the outer shell.
Inside are learned computational circuits: world representations, algorithms,
fact memories, context-copying circuits, planning.

---

## 5. Why "probability" yields confident answers

For example, with the prompt `"Thủ đô của Việt Nam là"` ("The capital of
Vietnam is"), the logits after softmax might look like this (illustrative):

| Token | Probability |
|---|---|
| ` Hà` (Ha, as in Hanoi) | 0.97 |
| ` thành` ("city") | 0.015 |
| ` TP` ("City", abbreviated) | 0.006 |
| ...(the remaining 32k tokens) | ~0.009 |

- The correct answer is **not luck**: training on thousands of sentences
  asserting the same fact has pushed the probability of the correct token
  close to 1.
- `temperature < 1` or greedy decoding (argmax) makes the distribution even
  more peaked. In `generate`, `logits / temperature` with `temperature → 0`
  approaches argmax.
- `top_k`/`top_p` cut off the tail of the distribution, removing
  low-probability tokens (where errors tend to arise).
- **Calibration**: pretrained models usually "know what they know" — the
  probabilities they assign correlate well with real-world accuracy
  (Kadavath et al. 2022). When the distribution is flat, that is a signal
  the model is unsure. Note: the GPT-4 report shows the base model is well
  calibrated but RLHF *reduces* calibration, so the probabilities of a chat
  model are no longer as trustworthy.

In other words: *randomness* only enters at the sampling step, and with solid
knowledge the distribution is almost a single peak.

---

## 6. Scale: why bigger models are more "correct"

- **Scaling laws** (Kaplan et al. 2020; Hoffmann et al. 2022 — *Chinchilla*):
  loss falls as a power law as parameters, data, and compute increase.
  Chinchilla uses ~20 tokens of data per parameter (70B parameters, 1.4T
  tokens); this ratio is inferred from the paper and was replicated by Epoch
  AI (2024). In practice today, models are deliberately trained *beyond* this
  ratio so that smaller models are cheaper at inference time (Sardana et al.,
  ICML 2024).
- Each small drop in loss corresponds to modelling one more layer of
  structure: first character frequencies → words → grammar → common facts →
  rare facts → multi-step reasoning.
- **"Emergent abilities"** (Wei et al. 2022): some abilities appear to arise
  suddenly at large scale. Schaeffer et al. 2023 showed that most of the
  "suddenness" is due to the metric (exact-match) — the loss actually
  decreases smoothly, but a task only counts as "correct" when every token is
  correct. Du et al. (NeurIPS 2024) partly rebut this: in terms of
  pre-training loss, some abilities only appear once the loss crosses a
  threshold, even with continuous metrics. The debate remains open.

With this repo (a model of a few tens of millions of parameters, trained on
Vietnamese Wikipedia), you will see the model produce fluent, grammatical
text that is **often factually wrong** — exactly as the scaling laws predict:
grammar is learned first, while knowledge and reasoning need far more
parameters and data.

---

## 7. From "predicting text" to "answering correctly": post-training

A purely pretrained model predicts Internet text — which includes wrong
statements, jokes, and questions with no answer. Commercial LLMs go through
several further steps to **shift the distribution** toward correct and
helpful answers:

1. **Supervised fine-tuning (SFT / instruction tuning)**: continue training
   on (question, good answer) pairs. It is still exactly the same
   cross-entropy — see [`FINETUNING.md`](FINETUNING.md), Method 2 and Method 3
   of this repo.
2. **RLHF / RLAIF** (Ouyang et al. 2022 — *InstructGPT*): optimize against a
   reward score from human/AI judgments, not just imitating text.
3. **RL with verifiable rewards** (RLVR — e.g. DeepSeek-R1, published in
   *Nature* in 9/2025; "reasoning models"): reward when a math answer is
   correct or code passes tests. The model teaches itself to **self-check,
   backtrack, and correct errors** in its chain of thought without
   human-written reasoning traces. The objective is no longer "resemble
   Internet text" but "**get the right result**". *Open debate:* Yue et al.
   (NeurIPS 2025) show that RL raises pass@1 but the base model wins at
   pass@k for large k, i.e. RL mostly *re-selects* existing capabilities;
   ProRL (NeurIPS 2025) shows that prolonged RL can surpass the base model.
4. **Tools & retrieval**: RAG (Lewis et al. 2020), calling a calculator,
   running code, searching the web — the model predicts a *tool call*, while
   facts/numbers come from a trusted source.

So "an LLM just predicts" is true about the generation mechanism, but **what
is being predicted has been directly optimized for correctness**, not merely
for "resembling text".

---

## 8. Multi-step reasoning: chain-of-thought as a "scratchpad"

A single forward pass through a transformer has fixed depth (`n_layer`
blocks). In theory, this limits the amount of sequential computation within
**one** token (Feng et al., NeurIPS 2023): without CoT, a constant-depth
transformer is upper-bounded by the TC0 circuit class (logarithmic
precision), and even only AC0 if the precision is constant (Li et al., ICLR
2024). With poly(n) steps of CoT, the transformer obtains **exactly the class
P** (Merrill & Sabharwal, ICLR 2024).

**Chain-of-thought** (Wei et al. 2022; Kojima et al. 2022 — "Let's think step
by step") breaks that limit: each generated token is appended back into the
context (exactly the line `idx = torch.cat([idx, next_token], dim=1)` in
`generate`) and becomes **intermediate memory** for later steps. With enough
intermediate tokens, a transformer can simulate general sequential
computation. This is why:

- Writing out the steps → accuracy rises sharply on math and logic.
- **Self-consistency** (Wang et al. 2022): generate several solutions and
  pick the most frequent answer → random errors cancel out.
- Reasoning models "think longer" (more tokens) before answering → more
  accurate (test-time compute scaling). Snell et al. (ICLR 2025): a small
  model using extra compute at answer time can beat a model 14 times larger,
  but **only on easy and medium questions**; the hardest questions still need
  stronger pretraining.
- **Caveat:** CoT improves accuracy, but the CoT *text* does not faithfully
  reflect how the model computes. Reasoning models mention a hint they
  actually used in only ~25% (Claude 3.7 Sonnet) to ~39% (DeepSeek-R1) of
  cases (Chen et al. 2025); Turpin et al. (NeurIPS 2023) also found that CoT
  omits biases that skew the answer.

---

## 9. The flip side: when and why LLMs are wrong

Understanding why they are right also helps explain why they are wrong — and
these errors are all **traces of the "prediction" objective**:

| Phenomenon | Root cause |
|---|---|
| **Hallucination** (fabricating facts, citations) | The model is trained to generate *plausible* text; for rare facts it still produces fluent sentences. Kalai et al. 2025 (*Why Language Models Hallucinate*, preprint): the fabrication rate has a lower bound of roughly the fraction of facts that appear only **once** in the data (the "singleton rate"), and 0/1-graded benchmarks reward guessing over saying "I don't know"; they propose penalizing wrong answers by t/(1−t) with a confidence threshold t. |
| **Reversal curse**: knows "A is the mother of B" but cannot answer "Whose child is B?" | Learning in the left→right direction; Berglund et al. (ICLR 2024): GPT-4 is correct 79% in the forward direction, 33% in the reverse. "Reverse training" (Golovneva et al. 2024) reduces it considerably → not a fundamental limit. |
| **Frequency sensitivity** ("embers of autoregression"): counting/reversing letters and decoding do worse on rarely seen strings | McCoy et al. (PNAS 2024): GPT-4 decodes ciphers correctly 51% of the time when the answer is a high-probability sentence, only 13% when it is low-probability. With o1 the effect shrinks but does not vanish. |
| **Long multi-step computation** (multiplying very large numbers) without CoT | Fixed-depth limit; Dziri et al. 2023 (*Faith and Fate*): models often stitch together local patterns rather than follow a systematic algorithm. |
| **Tokenizer errors** (counting letters in a word, digit arithmetic) | The model sees BPE tokens, not characters — see `scripts/inspect_vocab.py` to see how a Vietnamese word is split. Evidence is mixed (CharBench 2025): the tokenizer is *a contributing factor*, not the sole cause. |
| **Reasoning robustness** | GSM-Symbolic (Apple 2024): adding one irrelevant clause drops accuracy by up to 65%. The conclusion "cannot reason" is still being challenged on statistical grounds. |
| **Error accumulation** | One wrong token is appended to the context and every later token is conditioned on it (autoregressive). |

So a more accurate answer to the original question is: **LLMs are right where
data + scale + post-training have made the true structure the best way to
predict; LLMs are wrong where a "plausible-sounding" solution is cheaper than
the correct one.**

---

## 10. Experiments you can run with this repo

1. **Observe the probability distribution**: temporarily modify `generate`
   (or write a script) to print the `top-5` of `F.softmax(logits, -1)` at each
   step. Compare a prompt with a clear answer (e.g. a common phrase on
   Wikipedia) against an ambiguous prompt → see a peaked vs. a flat
   distribution.
2. **Temperature**: run `scripts/sample.py` with `temperature` 0.1, 0.7, 1.5.
   Low → repetitive, safe; high → creative but with many factual errors.
3. **Loss across training steps**: plot val loss across checkpoints, and
   generate a few sentences from each checkpoint. You will see the order of
   learning: characters/syllables → words → grammar → some common facts.
4. **Miniature grokking**: create a text dataset of `a + b = c (mod 97)`,
   train a very small `GPTConfig` (2 layers) with large weight decay, and
   track train acc vs. val acc — reproducing the model's switch from
   "memorizing" to "algorithm".
5. **Fine-tune Q&A** ([`FINETUNING.md`](FINETUNING.md)): compare the answers
   before/after SFT for the same question — illustrating section 7.

---

## 11. Summary

1. The **interface** of an LLM is token prediction, but the **internal
   mechanism** is learned computational circuits: world representations,
   fact memory, algorithms, in-context learning.
2. Good prediction is equivalent to good compression, and good compression
   requires grasping the regularities that generate the data — so optimizing
   cross-entropy on huge data **forces** the model to learn real structure.
3. With solid knowledge, the probability distribution is very peaked →
   "prediction" is nearly deterministic; temperature/top-p control the
   remaining randomness.
4. Scale (parameters, data, compute) determines which structures get learned.
5. Post-training (SFT, RLHF, RL with verifiable rewards) shifts the objective
   from "resemble text" to "correct and helpful".
6. Chain-of-thought turns the context into a scratchpad, extending reasoning
   capability beyond the limit of a single forward pass.
7. Errors (hallucination, reversal curse, tokenizer errors) also stem from
   the very nature of prediction — so combine retrieval, tools, and
   verification when high reliability is needed.

---

## 12. References

*Verified and updated by a research run dated 2026-11-08 (51 sources, with
links and peer-review status):
[`_bmad-output/planning-artifacts/research/academic-lit-why-llms-answer-accurately-despite-next-2026-11-08/research.md`](../_bmad-output/planning-artifacts/research/academic-lit-why-llms-answer-accurately-despite-next-2026-11-08/research.md).*

- Shannon, C. (1948). *A Mathematical Theory of Communication*; (1951). *Prediction and Entropy of Printed English*.
- Bengio, Y. et al. (2003). *A Neural Probabilistic Language Model*. JMLR.
- Vaswani, A. et al. (2017). *Attention Is All You Need*.
- Kaplan, J. et al. (2020). *Scaling Laws for Neural Language Models*. arXiv:2001.08361.
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.
- Geva, M. et al. (2021). *Transformer Feed-Forward Layers Are Key-Value Memories*.
- Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla).
- Power, A. et al. (2022). *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*.
- Olsson, C. et al. (2022). *In-context Learning and Induction Heads*. Anthropic.
- Elhage, N. et al. (2022). *Toy Models of Superposition*. Anthropic.
- Meng, K. et al. (2022). *Locating and Editing Factual Associations in GPT* (ROME).
- Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*; *Emergent Abilities of Large Language Models*.
- Kojima, T. et al. (2022). *Large Language Models are Zero-Shot Reasoners*.
- Wang, X. et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning*.
- Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback* (InstructGPT).
- Kadavath, S. et al. (2022). *Language Models (Mostly) Know What They Know*.
- Li, K. et al. (2023). *Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task* (Othello-GPT).
- Nanda, N. et al. (2023). *Progress Measures for Grokking via Mechanistic Interpretability*; *Emergent Linear Representations in World Models of Self-Supervised Sequence Models*.
- Delétang, G. et al. (2023). *Language Modeling Is Compression*.
- Gurnee, W. & Tegmark, M. (2023). *Language Models Represent Space and Time*.
- Schaeffer, R. et al. (2023). *Are Emergent Abilities of Large Language Models a Mirage?*
- Feng, G. et al. (2023). *Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective*.
- Merrill, W. & Sabharwal, A. (2023/2024). *The Expressive Power of Transformers with Chain of Thought*.
- Berglund, L. et al. (2023, ICLR 2024). *The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"*.
- McCoy, R. T. et al. (2023/2024). *Embers of Autoregression*. PNAS 2024.
- Dziri, N. et al. (2023). *Faith and Fate: Limits of Transformers on Compositionality*.
- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. Nature 645.
- Lindsey, J. et al. (2025). *On the Biology of a Large Language Model*. Anthropic (Transformer Circuits).
- Kalai, A. T. et al. (2025). *Why Language Models Hallucinate*. OpenAI.
- Kalai, A. T. & Vempala, S. (2024). *Calibrated Language Models Must Hallucinate*. STOC 2024.
- Huang, Y. et al. (2024). *Compression Represents Intelligence Linearly*. COLM 2024.
- Du, Z. et al. (2024). *Understanding Emergent Abilities of Language Models from the Loss Perspective*. NeurIPS 2024.
- Vafa, K. et al. (2024). *Evaluating the World Model Implicit in a Generative Model*. NeurIPS 2024.
- Karvonen, A. (2024). *Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models*.
- Li, Z. et al. (2024). *Chain of Thought Empowers Transformers to Solve Inherently Serial Problems*. ICLR 2024.
- Snell, C. et al. (2025). *Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters*. ICLR 2025.
- Yue, Y. et al. (2025). *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* NeurIPS 2025.
- Chen, Y. et al. (2025). *Reasoning Models Don't Always Say What They Think*. Anthropic.
- Turpin, M. et al. (2023). *Language Models Don't Always Say What They Think*. NeurIPS 2023.
- Golovneva, O. et al. (2024). *Reverse Training to Nurse the Reversal Curse*.
- Mirzadeh, I. et al. (2024). *GSM-Symbolic*. Apple.
