---
status: draft
stepsCompleted: [1, 2, 3, 4]
inputDocuments:
  - _bmad-output/planning-artifacts/research/technical-build-vietnamese-llm-from-wikipedia-data-2026-11-08/research.md
  - _bmad-output/brainstorming/brainstorm-how-llms-predict-next-token-2026-11-08/.memlog.md
  - README.md
  - docs/DEVELOPMENT.md
  - docs/FINETUNING.md
  - docs/why_llms_are_accurate.md
---

# NTokenizer - Epic Breakdown

## Overview

This document breaks the requirements down into epics and stories for
NTokenizer: a Vietnamese GPT built from scratch in plain PyTorch (currently
trained on Vietnamese Wikipedia).

**There is no PRD or architecture document.** The requirements are drawn from
three sources: the technical research report (recommendations R1–R8), the
115-idea brainstorming session, and the existing source code and
documentation. If a PRD is created later, these requirements should be
reconciled against it.

**Primary user:** the project owner (a researcher/learner) who wants to (a)
understand the token-prediction mechanism on their own model, and (b) upgrade
the model from "only knows Wikipedia" to a more useful Vietnamese LM.

**Hardware assumption:** unknown. Every training-time requirement is measured
on the real machine (Epic 4) rather than promised in advance.

## Requirements Inventory

### Functional Requirements

**Evaluation**
- FR1: A script computes bits-per-byte (BPB) on a fixed validation set, so that different tokenizers can be compared.
- FR2: A script evaluates Vietnamese cloze/next-word prediction (in the style of `lambada_vi`) and reports accuracy.
- FR3: A tool checks n-gram overlap (8-gram and 13-gram) between the training data and the evaluation sets.
- FR4: Each checkpoint has an evaluation results table (BPB, cloze) that is saved so results can be compared over time.

**Data**
- FR5: The pipeline normalizes Unicode to NFC and unifies the position of tone marks ("hoà"/"hòa", i.e. old-style vs. new-style tone mark placement).
- FR6: The pipeline removes duplicates with MinHash and filters out stub/too-short articles.
- FR7: A configurable subset of FineWeb-2 `vie_Latn` (by token count) can be loaded without downloading the whole dataset.
- FR8: The mixing ratio between sources (Wikipedia, FineWeb-2) is configurable, and the provenance/license of each source is recorded in `meta.json`.

**Tokenizer**
- FR9: A tool measures fertility (tokens per word) and compression ratio on Vietnamese text for any tokenizer.
- FR10: Tokenizers with different vocab sizes (16k/20k/32k) can be trained and compared on the new data mix.

**Training**
- FR11: An option in `GPTConfig` ties the input and output embeddings (weight tying).
- FR12: Model size can be set from a single depth parameter (a deep-and-thin preset), instead of hand-tuning many coefficients.
- FR13: A token budget planning tool computes, from the parameter count and the maximum number of epochs, the number of tokens required, and warns when data would be repeated for more than 4 epochs.
- FR14: A script measures throughput (tokens/second) per device and records it in the documentation.
- FR15: (Experimental) QK-Norm and Muon optimizer options, toggled by flags.

**Conversation (SFT)**
- FR16: Vietnamese instruction data (vi-alpaca, Aya) can be converted to a training format with a chat template.
- FR17: SFT supports an option to mask or not mask the prompt when computing the loss.
- FR18: An interactive chat CLI is available for the SFT-tuned model.

**Prediction microscope**
- FR19: `sample.py` shows the top-5 probabilities at each generation step.
- FR20: A logit lens tool projects each layer's activations through `ln_f` and `lm_head` and plots the rank of the correct token by layer.
- FR21: A tool colors each token of a passage by its entropy.
- FR22: A reliability diagram compares predicted probabilities with the actual fraction correct.

**Mechanism lab**
- FR23: A text-format grokking dataset `a + b = c (mod 97)` and a 2-layer model configuration can be created for training, logging train/val accuracy per step.
- FR24: A "fake world" dataset can be created with facts repeated 1, 2, 5, 10, and 50 times, measuring recall accuracy by frequency.
- FR25: A Vietnamese reversal curse test is available (forward and reverse sentences).
- FR26: A tool detects induction heads and runs a per-head ablation sweep.

**Documentation**
- FR27: A model card lists data sources, licenses (CC BY-SA, ODC-By), and limitations.
- FR28: An English explainer `docs/how_llms_predict.md` has one runnable experiment per chapter.

### NonFunctional Requirements

- NFR1: All new tools live in `src/ntokenizer/` and are invoked only through thin scripts in `scripts/`.
- NFR2: Every change in `src/` has pytest tests; the existing test suite must still pass.
- NFR3: Measurements must be reproducible: fix the seed and record the configuration/commit in the result file.
- NFR4: The data pipeline must not require loading the entire corpus into RAM (it must run on 8–16 GB machines).
- NFR5: The tokenizer used for training and sampling must match the checkpoint; any tokenizer change must come with a warning that old checkpoints are no longer usable.
- NFR6: Every data source must have a record of its origin, license, and download date.
- NFR7: Tools must run on CPU, CUDA, and MPS; heavy features must have a small mode for trying them out.
- NFR8: Wording in documentation and results describes "representation/computation at small scale" and does not claim the model "understands".

### Additional Requirements

- No starter template; the repo already has its skeleton (`pyproject.toml`, `src/ntokenizer`, `scripts`, `tests`).
- The cloud environment's network blocks `huggingface.co` and `arxiv.org`. The data-loading story must run on the user's machine and have a way to run a trial with small sample data.
- Every probe must be paired with an intervention (patching or ablation) before drawing conclusions (the brainstorm's "Critic" rule).
- Open legal questions: whether CC BY-SA applies to model weights has no answer yet, so the model card must state this explicitly.
- Do not use VMLU as the primary metric for models under 200M parameters (PhoGPT-7B5-Instruct scores only 24.01, on par with random guessing).
- Throughput figures on consumer GPUs and Macs have no reliable source; they must be measured directly.

### UX Design Requirements

Not applicable: the project has no user interface. The tools are CLIs and static image/HTML files.

### FR Coverage Map

FR1: Epic 1 - Bits-per-byte on a fixed validation set
FR2: Epic 1 - Vietnamese cloze evaluation
FR3: Epic 1 - N-gram overlap check
FR4: Epic 1 - Results table per checkpoint
FR5: Epic 2 - NFC normalization and tone mark position
FR6: Epic 2 - MinHash deduplication and stub filtering
FR7: Epic 2 - Loading a FineWeb-2 subset
FR8: Epic 2 - Source mixing and provenance recording
FR9: Epic 3 - Measuring fertility and compression ratio
FR10: Epic 3 - Comparing vocab sizes
FR11: Epic 4 - Weight tying
FR12: Epic 4 - Depth-based preset
FR13: Epic 4 - Token budget planning (Story 4.4)
FR14: Epic 4 - Throughput measurement (Story 4.3)
FR15: Epic 4 - QK-Norm and Muon (experimental)
FR16: Epic 5 - SFT data preparation
FR17: Epic 5 - Masking or not masking the prompt
FR18: Epic 5 - Chat CLI
FR19: Epic 6 - Top-5 probabilities
FR20: Epic 6 - Logit lens
FR21: Epic 6 - Entropy heatmap
FR22: Epic 6 - Reliability diagram
FR23: Epic 7 - Grokking mod 97
FR24: Epic 7 - Fake world by frequency
FR25: Epic 7 - Vietnamese reversal curse
FR26: Epic 7 - Induction heads and ablation
FR27: Epic 8 - Model card and licenses
FR28: Epic 8 - English explainer

## Epic List

### Epic 1: Knowing whether the model is getting better or worse (evaluation suite)
The project owner runs one command to see the BPB and cloze accuracy of any checkpoint, and to know whether the evaluation data has leaked into the training data.
**FRs covered:** FR1, FR2, FR3, FR4

### Epic 2: Clean and sufficiently large Vietnamese data
The project owner has a training dataset made of Wikipedia and a portion of FineWeb-2, normalized, deduplicated, with provenance and license recorded.
**FRs covered:** FR5, FR6, FR7, FR8

### Epic 3: A tokenizer optimized for Vietnamese
The project owner chooses the vocab size based on measurements (fertility, compression, BPB) instead of guessing.
**FRs covered:** FR9, FR10

### Epic 4: A compact training recipe with real measurements
The project owner can plan training (size, tokens, time) that fits their machine, and can use architecture improvements that have been validated.
**FRs covered:** FR11, FR12, FR13, FR14, FR15

### Epic 5: A model that can chat in Vietnamese (SFT)
The project owner fine-tunes a pretrained checkpoint into a question-answering model, and can compare SFT choices using the evaluation suite.
**FRs covered:** FR16, FR17, FR18

### Epic 6: The prediction microscope
The project owner sees what the model is "thinking" at each token: the probabilities, which layer decides, where it is uncertain, and whether the probabilities can be trusted.
**FRs covered:** FR19, FR20, FR21, FR22

### Epic 7: A mechanism verification lab
The project owner reproduces, at small scale, the phenomena described in the research literature: grokking, fact memorization by frequency, the reversal curse, and induction heads.
**FRs covered:** FR23, FR24, FR25, FR26

### Epic 8: Publication and explanation
An outside reader can read a clear model card about the data and licenses, and an English explainer tied to runnable experiments.
**FRs covered:** FR27, FR28

## Epic 1: Knowing whether the model is getting better or worse (evaluation suite)

The project owner runs one command to see the BPB and cloze accuracy of any checkpoint, and to know whether the evaluation data has leaked into the training data.

### Story 1.1: Compute bits-per-byte on the validation set

As a project owner,
I want to run one command that computes the bits-per-byte of a checkpoint on a fixed validation set,
So that I can fairly compare models that use different tokenizers.

**Acceptance Criteria:**

**Given** a checkpoint and a valid `val.bin` with `meta.json`
**When** I run `scripts/eval_bpb.py --ckpt <path>`
**Then** the script prints the loss (nat/token), the number of UTF-8 bytes of the corresponding text (counted by decoding the tokens of `val.bin`), and the BPB
**And** BPB = loss × (number of tokens / number of bytes) / ln 2, with a test on synthetic data giving a known result.

**Given** the tokenizer in `meta.json` does not match the checkpoint
**When** I run the script
**Then** the script stops with a clear error message instead of printing a wrong number.

### Story 1.2: Vietnamese cloze evaluation

As a project owner,
I want to measure the accuracy with which the model predicts the final word of a sentence,
So that I have a "usable" metric for small models that VMLU cannot measure.

**Acceptance Criteria:**

**Given** a JSONL file of `{context, target}` items that I provide (at least 3 sample items ship with the repo for trying it out)
**When** I run `scripts/eval_cloze.py --ckpt <path> --data <file>`
**Then** the script reports the fraction of items where greedy decoding reproduces `target`, and prints each incorrect item when `--verbose` is given
**And** rerunning with the same seed gives identical results.

**Given** an item whose `target` consists of multiple tokens
**When** evaluating
**Then** the item counts as correct only if all tokens of `target` match.

### Story 1.3: Check n-gram overlap between train and evaluation sets

As a project owner,
I want to know how many evaluation items have n-grams that appear in the training data,
So that I do not report scores inflated by data contamination.

**Acceptance Criteria:**

**Given** a training corpus and an evaluation file
**When** I run `scripts/check_contamination.py --train <corpus> --eval <file> --n 13`
**Then** the script prints the count and fraction of evaluation items with at least one overlapping 13-gram, and saves the list of contaminated items to a file
**And** the `--n 8` flag gives the result for 8-grams.

**Given** a corpus larger than RAM
**When** running
**Then** the script processes it as a stream (without loading the whole corpus into memory) and has a test with a small synthetic corpus.

### Story 1.4: Save a results table per checkpoint

As a project owner,
I want each evaluation run to write a row to a results table,
So that I can see how the model progresses across checkpoints.

**Acceptance Criteria:**

**Given** I run Story 1.1 or 1.2
**When** I add the `--log-results` flag
**Then** a row is appended to `artifacts/eval/results.csv` containing the time, checkpoint path, training step, metric, value, and git commit
**And** repeated runs do not overwrite old rows.

**Given** `results.csv` has many rows
**When** I run `scripts/eval_report.py`
**Then** the script prints a table comparing checkpoints for each metric.

## Epic 2: Clean and sufficiently large Vietnamese data

The project owner has a training dataset made of Wikipedia and a portion of FineWeb-2, normalized, deduplicated, with provenance and license recorded.

### Story 2.1: Normalize Unicode and tone mark position

As a project owner,
I want all text entering the corpus to be NFC-normalized with unified tone mark position,
So that the same word is not split into several different token sequences.

**Acceptance Criteria:**

**Given** a string containing "hoà" (old style) and "hòa" (new style) in NFD or NFC form
**When** the function `normalize_vi(text)` in `src/ntokenizer/corpus.py` is called
**Then** both map to a single NFC form according to the convention documented in the docstring
**And** there are tests for at least 6 word pairs (hoà/hòa, thuỷ/thủy, khoẻ/khỏe...).

**Given** the existing `build_corpus` pipeline
**When** I run it with the `--normalize` flag
**Then** the output corpus has passed through `normalize_vi` and the existing `test_corpus.py` test suite still passes.

### Story 2.2: MinHash deduplication and stub filtering

As a project owner,
I want to remove near-duplicate documents and too-short articles,
So that the model does not learn repetition and is not skewed by bot-generated stub articles.

**Acceptance Criteria:**

**Given** a corpus with duplicate and near-duplicate documents
**When** I run `scripts/dedup_corpus.py --threshold 0.8 --min-chars <N>`
**Then** the output keeps one representative for each near-duplicate group and drops articles shorter than `N` characters
**And** the script prints the number of documents in/out and the number removed per reason.

**Given** a large corpus
**When** running
**Then** memory use grows with the number of MinHash signatures and not with the size of the text, and there is a test on synthetic data for exact and near duplicates.

### Story 2.3: Load a Vietnamese FineWeb-2 subset

As a project owner,
I want to download a configurable number of tokens from FineWeb-2 `vie_Latn`,
So that I get more web data without downloading hundreds of GB.

**Acceptance Criteria:**

**Given** a machine with access to `huggingface.co`
**When** I run `scripts/fetch_fineweb2.py --max-tokens 1e9 --out data/raw/fineweb2_vi`
**Then** the script reads as a stream, stops when the threshold is reached, and writes the documents to JSONL
**And** writes a `SOURCE.json` file containing the dataset name, version, ODC-By license, download date, and document count.

**Given** no network or a blocked network
**When** running with `--sample-fixture`
**Then** the script uses a small sample set shipped with the repo to exercise the rest of the pipeline.

### Story 2.4: Mix data sources and record provenance

As a project owner,
I want to declare the mixing ratio between Wikipedia and FineWeb-2 and have a provenance record in `meta.json`,
So that I know where the model learned from and can meet license obligations.

**Acceptance Criteria:**

**Given** two normalized and deduplicated corpora
**When** I run `scripts/prepare_dataset.py --mix wiki=0.3,fineweb2=0.7`
**Then** `train.bin` and `val.bin` are created at approximately the requested ratio (error ≤ 2% by token count)
**And** `meta.json` has a `sources` entry listing each source: name, license, token count, download date.

**Given** `val.bin`
**When** created
**Then** the validation set takes the same fixed set of documents (by document hash) for every mix so that BPB is comparable.

## Epic 3: A tokenizer optimized for Vietnamese

The project owner chooses the vocab size based on measurements (fertility, compression, BPB) instead of guessing.

### Story 3.1: Tool to measure fertility and compression ratio

As a project owner,
I want to measure the number of tokens per word and the number of bytes per token of a tokenizer,
So that I compare tokenizers using numbers.

**Acceptance Criteria:**

**Given** a SentencePiece tokenizer and a Vietnamese text file
**When** I run `scripts/measure_tokenizer.py --model <spm> --text <file>`
**Then** the script prints fertility (tokens/word, words split on whitespace), bytes/token, and the fraction of tokens that consist of a single syllable
**And** it works with several `--model` arguments at once to print a comparison table.

**Given** text with unusual characters or empty text
**When** measuring
**Then** the script does not fail and reports the number of skipped lines.

### Story 3.2: Train and compare vocab sizes

As a project owner,
I want to train 16k, 20k, and 32k tokenizers on the same data mix and compare them,
So that I choose the vocab size on solid grounds.

**Acceptance Criteria:**

**Given** the mixed corpus from Epic 2 (or the existing Wikipedia corpus if Epic 2 is not finished)
**When** I run `scripts/train_tokenizer_spm.py` with each `--vocab-size`
**Then** each tokenizer is saved with a name containing the vocab size
**And** `measure_tokenizer.py` prints a comparison table of the three tokenizers.

**Given** the comparison table
**When** I decide on a size
**Then** the decision and the figures are recorded in `docs/tokenizer_choice.md`, together with a warning that old checkpoints using a different tokenizer will not be compatible (NFR5).

### Story 3.3: Compare models by BPB across tokenizers

As a project owner,
I want to quickly train a small model with each tokenizer and compare BPB,
So that the vocab size is chosen by real quality and not just fertility.

**Acceptance Criteria:**

**Given** the three tokenizers from Story 3.2 and a fixed small model configuration
**When** I run `scripts/compare_tokenizers.sh` (short training, same number of data bytes)
**Then** each model is evaluated with `eval_bpb.py` and the results are written to `results.csv`
**And** the final table ranks the tokenizers by BPB.

**Given** running on CPU
**When** using the `--tiny` flag
**Then** the whole comparison completes with very small data to check the pipeline.

## Epic 4: A compact training recipe with real measurements

The project owner can plan training (size, tokens, time) that fits their machine, and can use architecture improvements that have been validated.

### Story 4.1: Weight tying option

As a project owner,
I want to turn on tying of the input and output embeddings,
So that a small model saves parameters to spend on depth.

**Acceptance Criteria:**

**Given** `GPTConfig(tie_embeddings=True)`
**When** creating the model
**Then** `lm_head.weight` and `wte.weight` are the same tensor and `count_parameters()` decreases by exactly `vocab_size × n_embd`
**And** the default is `tie_embeddings=False` so old checkpoints can still be loaded.

**Given** a model with tying
**When** saving and reloading a checkpoint
**Then** the logits before and after loading are identical, with a test in `tests/test_model.py`.

### Story 4.2: Depth-based preset

As a project owner,
I want to set the model size with a single depth parameter,
So that I can try different sizes without hand-tuning many coefficients.

**Acceptance Criteria:**

**Given** `GPTConfig.from_depth(depth=12, vocab_size=V, block_size=B)`
**When** called
**Then** the function returns a configuration with `n_layer = depth` and with the hidden dimension, number of heads, and number of KV heads derived by the rule documented in the docstring (preferring deep-and-thin)
**And** the number of heads divides the hidden dimension and the number of KV heads divides the number of heads.

**Given** depths 6, 12, and 24
**When** printing the parameter counts
**Then** the parameter count increases monotonically and there is a test that checks configuration validity.

### Story 4.3: Measure throughput per device

As a project owner,
I want to measure the tokens/second of the model on my machine,
So that the training plan is based on real measurements.

**Acceptance Criteria:**

**Given** a `GPTConfig` and an auto-detected device (CUDA, MPS, or CPU)
**When** I run `scripts/bench_throughput.py --steps 20 --warmup 5`
**Then** the script prints tokens/second, peak memory (if available), device, dtype, and torch version
**And** appends a row to `docs/throughput.md` when the `--record` flag is given.

**Given** a device that does not support bf16
**When** running
**Then** the script falls back to fp32 and states this explicitly in the result.

### Story 4.4: Token budget planning tool

As a project owner,
I want to enter the parameter count and the amount of data I have and learn how many tokens and epochs are needed,
So that I do not undertrain or repeat data excessively.

**Acceptance Criteria:**

**Given** `scripts/plan_training.py --params 125e6 --tokens-available 1.1e9 --tokens-per-param 20`
**When** running
**Then** the script prints the target tokens (2.5 billion), the number of epochs needed (≈2.3), and approximate FLOPs by 6·N·D
**And** prints a warning when the number of epochs exceeds 4 and suggests adding data.

**Given** throughput measurement results (Story 4.3)
**When** passing `--throughput <tokens/s>`
**Then** the script additionally prints the estimated training time.

### Story 4.5: QK-Norm and Muon (experimental)

As a project owner,
I want to try QK-Norm and the Muon optimizer via flags,
So that I can check whether the modded-nanogpt tricks help at my scale.

**Acceptance Criteria:**

**Given** `GPTConfig(qk_norm=True)`
**When** running forward
**Then** Q and K are normalized before computing attention, and the default `False` does not change existing results (regression test).

**Given** `scripts/train.py --optimizer muon`
**When** training for 50 steps on small data
**Then** the loss decreases and AdamW is kept for the embeddings and 1-D parameters
**And** the documentation states clearly that this is an experimental feature, with a BPB comparison against AdamW in `results.csv`.

## Epic 5: A model that can chat in Vietnamese (SFT)

The project owner fine-tunes a pretrained checkpoint into a question-answering model, and can compare SFT choices using the evaluation suite.

### Story 5.1: Prepare SFT data with a chat template

As a project owner,
I want to convert a Vietnamese instruction dataset into a training format with a fixed chat template,
So that I fine-tune the model with the existing fine-tune pipeline.

**Acceptance Criteria:**

**Given** a vi-alpaca-style JSONL file (`instruction`, `input`, `output`) or Aya
**When** I run `scripts/prepare_sft.py --in <file> --out data/processed/sft`
**Then** each sample is framed with the user/assistant special tokens defined by the template, and a `.bin` file is produced together with a mask of the answer positions
**And** `SOURCE.json` records the dataset name, license, and the warning "machine-translated or model-generated" where applicable.

**Given** a sample longer than `block_size`
**When** preparing
**Then** the sample is dropped and the number of dropped samples is printed.

### Story 5.2: SFT with a prompt-masking option

As a project owner,
I want to choose whether the loss is computed only on the answer or on the prompt as well,
So that I can test which choice works better for small data.

**Acceptance Criteria:**

**Given** data from Story 5.1 and `--init_from <ckpt>`
**When** I run `scripts/train.py --sft --mask-prompt` or `--no-mask-prompt`
**Then** with `--mask-prompt` the loss is 0 at prompt tokens (checked by a test with a small batch), and with `--no-mask-prompt` the loss is computed on every token
**And** the default flag is `--mask-prompt`.

**Given** the existing `tests/test_finetune.py`
**When** running the tests
**Then** the old tests still pass.

### Story 5.3: Compare masked and unmasked prompts

As a project owner,
I want to compare the two SFT approaches on the same data using the evaluation suite,
So that I choose the approach based on numbers.

**Acceptance Criteria:**

**Given** two SFT checkpoints (with and without prompt masking) with the same seed and the same number of steps
**When** I run `eval_bpb.py` on a held-out conversation set and `eval_cloze.py`
**Then** the results of the two models appear side by side in `results.csv`
**And** a short conclusion is written to `docs/sft_notes.md` even when the difference is negligible.

### Story 5.4: Interactive chat CLI

As a project owner,
I want to chat with the model in the terminal,
So that I try real answers instead of only looking at numbers.

**Acceptance Criteria:**

**Given** an SFT checkpoint
**When** I run `scripts/chat.py --ckpt <path>`
**Then** the program accepts each turn, applies the chat template, generates a reply with `generate()`, and stops at the end-of-turn token
**And** the `/reset` command clears the context and `/quit` exits.

**Given** the context exceeds `block_size`
**When** the conversation continues
**Then** the oldest turns are truncated without raising an error.

## Epic 6: The prediction microscope

The project owner sees what the model is "thinking" at each token: the probabilities, which layer decides, where it is uncertain, and whether the probabilities can be trusted.

### Story 6.1: Show top-5 probabilities during generation

As a project owner,
I want to see the 5 highest-probability tokens at each generation step,
So that I can see whether the distribution is peaked or flat and understand why the model chose that token.

**Acceptance Criteria:**

**Given** a checkpoint and a prompt
**When** I run `scripts/sample.py --show-probs`
**Then** each step prints the chosen token and the top 5 tokens with their probabilities after applying temperature, top-k, and top-p
**And** without this flag, the output stays as it is today.

**Given** repetition penalty is enabled
**When** displaying
**Then** the printed probabilities are the post-penalty probabilities, and the first line states this explicitly.

### Story 6.2: Logit lens by layer

As a project owner,
I want to see at which layer the correct token starts to win,
So that I understand where in the network the answer forms.

**Acceptance Criteria:**

**Given** a prompt and a target token
**When** I run `scripts/logit_lens.py --ckpt <path> --prompt "..." --target "..."`
**Then** the script projects the residual after each layer through `ln_f` and `lm_head`, prints the rank and probability of the target token by layer, and saves a PNG plot
**And** the last layer gives exactly the logits of the normal `forward()` (checked by a test).

### Story 6.3: Entropy coloring on a passage

As a project owner,
I want to see which tokens the model is uncertain about in a passage,
So that I can relate uncertainty to rare facts and errors.

**Acceptance Criteria:**

**Given** a passage
**When** I run `scripts/entropy_view.py --ckpt <path> --file <txt>`
**Then** the script outputs a self-contained HTML file, with each token colored by entropy and a tooltip showing the entropy and the probability of the actual token
**And** the HTML works in both light and dark modes and loads no external resources.

### Story 6.4: Reliability diagram

As a project owner,
I want to know whether the probabilities the model assigns match the actual fraction correct,
So that I can assess whether the model "knows what it knows" and see the effect of SFT on calibration.

**Acceptance Criteria:**

**Given** `val.bin` and a checkpoint
**When** I run `scripts/calibration.py --ckpt <path>`
**Then** the script divides the probabilities of the top-1 predicted token into 10 bins, computes the accuracy within each bin, prints the ECE, and saves a PNG plot
**And** passing any two checkpoints (for example before and after SFT) draws two curves on the same plot.

## Epic 7: A mechanism verification lab

The project owner reproduces, at small scale, the phenomena described in the research literature: grokking, fact memorization by frequency, the reversal curse, and induction heads.

### Story 7.1: Grokking lab, mod 97

As a project owner,
I want to train a 2-layer model on addition modulo 97 and watch train/val accuracy by step,
So that I see with my own eyes the model transition from memorization to an algorithm.

**Acceptance Criteria:**

**Given** `scripts/grokking_lab.py --mod 97 --train-frac 0.3 --weight-decay 1.0`
**When** running
**Then** the script creates a text-format dataset `a + b = c`, trains a small model, and writes train/val accuracy by step to CSV and PNG
**And** it runs on CPU within a few minutes with the `--tiny` flag.

**Given** the results
**When** val accuracy has not exceeded 90% after the maximum number of steps
**Then** the script still writes the full results and clearly prints "grokking not observed" instead of drawing a wrong conclusion.

### Story 7.2: Fact memorization by frequency of occurrence

As a project owner,
I want to measure the recall accuracy of fake facts that appear 1, 2, 5, 10, and 50 times,
So that I can verify at small scale the idea that "rare facts are more likely to be fabricated".

**Acceptance Criteria:**

**Given** `scripts/fact_frequency_lab.py --n-facts 500`
**When** running
**Then** the script creates fake facts of the form "thủ đô của X là Y" (the capital of X is Y) with made-up X and Y, mixes them into the data at the specified frequencies, trains, and then prints the recall accuracy at each frequency level
**And** results are saved to CSV and PNG, and the same seed gives identical results.

**Given** the facts use made-up names
**When** evaluating
**Then** there is no leakage from real data (checked by a test: the names do not appear in the sample corpus).

### Story 7.3: Reversal curse in Vietnamese

As a project owner,
I want to test whether the model can answer reversed questions,
So that I verify the reversal curse myself on my own model.

**Acceptance Criteria:**

**Given** a set of made-up facts of the form "A là mẹ của B" (A is the mother of B) and two question sets (forward, reverse)
**When** I run `scripts/reversal_lab.py`
**Then** the script trains a small model on the forward direction only, then reports the accuracy on forward and reverse questions
**And** a `--reverse-augment` flag adds reversed sentences to the data and reports the results again for comparison.

### Story 7.4: Induction head detection and per-head ablation

As a project owner,
I want to find the heads that perform "in-context copying" and see how much the loss increases when they are removed,
So that I have interventional evidence and not just observation.

**Acceptance Criteria:**

**Given** a checkpoint
**When** I run `scripts/induction_heads.py --ckpt <path>`
**Then** the script generates a random token sequence repeated twice, measures the loss drop on the second repetition and the induction score of each head
**And** runs an ablation sweep: set each head's output to 0, record the loss increase, and print a ranking table of heads.

**Given** a model with GQA
**When** ablating
**Then** the script correctly handles heads sharing K/V (ablating by the attention output of each query head).

## Epic 8: Publication and explanation

An outside reader can read a clear model card about the data and licenses, and an English explainer tied to runnable experiments.

### Story 8.1: Model card and data licenses

As a project owner,
I want to generate a model card from `meta.json` and the evaluation results,
So that users know where the model learned from and under what conditions they may use it.

**Acceptance Criteria:**

**Given** a checkpoint, a `meta.json` with a `sources` entry, and `results.csv`
**When** I run `scripts/make_model_card.py --ckpt <path>`
**Then** `MODEL_CARD.md` lists the architecture, parameter count, data sources, the license of each source, the main evaluation results, and the known limitations
**And** it has an "Open legal questions" section stating explicitly that CC BY-SA may constrain the release of the weights and that there is no conclusion yet.

**Given** `meta.json` is missing license information
**When** generating the model card
**Then** the script stops and lists which sources are still missing.

### Story 8.2: English explainer tied to experiments

As an outside reader,
I want to read an English article explaining how an LLM predicts tokens, with commands to rerun the experiments,
So that I understand the mechanism and can verify it myself.

**Acceptance Criteria:**

**Given** the tools from Epic 6 and Epic 7 exist
**When** I read `docs/how_llms_predict.md`
**Then** each chapter has an experiment with a concrete command and a result figure: probability distribution, logit lens, grokking, memorization by frequency
**And** every claim about the mechanism cites a source from `research.md` and uses the phrasing "representation/computation at small scale" (NFR8).

**Given** the article contains analogies (compression, weather forecasting, detective)
**When** used
**Then** each analogy states explicitly where it is similar and where it differs, and is not used as a substitute for evidence.
