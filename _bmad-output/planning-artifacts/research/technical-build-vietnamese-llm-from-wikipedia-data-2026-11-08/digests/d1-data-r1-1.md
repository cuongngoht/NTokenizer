# D1 — Data: digest, round 1, scan 1

Accessed 2026-11-08. Environment note: direct fetches to huggingface.co, meta.wikimedia.org,
en.wikipedia.org, aclanthology.org and undertheseanlp.github.io failed (hosts not allowed in this
session's network). Only github.com was fetchable. Every claim marked "via search snippet" was seen
in search-result extracts, not on the fetched primary page. Confidence is reduced to match.

## Claims

Format: claim | source | publisher | pub_date | accessed | confidence | class

### Vietnamese Wikipedia size and quality

1. A 2026 dataset built from the Vietnamese Wikipedia XML dump (Underthesea "UVW") describes about **1.1M Vietnamese Wikipedia articles** (via search snippet). | UVW dataset docs, https://undertheseanlp.github.io/underthesea/docs/datasets/uvw | Underthesea NLP | 2026-?? | accessed 2026-11-08 | medium | size/scale
2. A tokenizer/LM card trained on Vietnamese Wikipedia monthly snapshots reports **~275.6M total tokens** (vocabulary 1,088,012, which suggests word-level tokens). So Wikipedia is on the order of a few hundred million words, roughly 2 orders of magnitude below the web corpora below (via search snippet; how tokens were counted is not verified). | wikilangs/vi README, https://huggingface.co/wikilangs/vi/blob/main/README.md | wikilangs (HF) | 2025/2026-?? | accessed 2026-11-08 | low | size/scale
3. Bot-created articles on Vietnamese Wikipedia: about 432 of ~50,000 articles in Aug 2008, and about 5% when it passed 100,000 articles (2009-09-12). One mirror claims ">58% of articles bot-generated as of 2023", but that figure has weak sourcing (via search snippets of the "Vietnamese Wikipedia" article and its mirrors). The broad point that many articles are bot-made stubs is **plausible but not confirmed with a primary number** in this run. | "Vietnamese Wikipedia", https://en.wikipedia.org/wiki/Vietnamese_Wikipedia | Wikipedia | 2026-?? | accessed 2026-11-08 | low (58% figure) / medium (2008–09 figures) | size/scale
4. No retrieved source tied Lsjbot to viwiki. That link is unverified. | same search | — | — | accessed 2026-11-08 | low | size/scale

### Open Vietnamese pretraining corpora

5. **CulturaX (uonlp/CulturaX)** Vietnamese row: **57,606,341 docs, 55,380,123,774 tokens, 0.88%** of the 6.3T-token, 167-language total. Gated on HF (accept terms) (via search snippet of the HF card). | CulturaX dataset card, https://huggingface.co/datasets/uonlp/CulturaX | UONLP (Univ. of Oregon) | 2023-09 | accessed 2026-11-08 | medium-high | size/scale
6. CulturaX license is listed as "other" (it inherits mC4 ODC-BY and OSCAR terms; mirror card shows `license: other`) (via search snippet; full terms not read). | SEACrowd/culturax README, https://huggingface.co/datasets/SEACrowd/culturax/blob/main/README.md | SEACrowd | 2024-?? | accessed 2026-11-08 | low | license
7. **FineWeb-2 (HuggingFaceFW/fineweb-2)** vie_Latn row: **50,886,874,358 | 61,064,248 docs | 319.83GB | 121.19GB**. The first column is probably tokens (consistent with Spanish ~261.5B in the same column), but the column headers were not seen (via search snippet of a card mirror). | FineWeb-2 dataset card (mirror), https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | Hugging Face | 2024-12 | accessed 2026-11-08 | medium | size/scale
8. A tokenizer-adaptation paper reports **343.4B UTF-8 bytes** for Vietnamese in FineWeb-2 (via search snippet). | "Teaching Old Tokenizers New Words", https://arxiv.org/pdf/2512.03989 | arXiv | 2025-12 | accessed 2026-11-08 | low | size/scale
9. FineWeb-2 is licensed **ODC-By**, and its code is Apache-2.0. | huggingface/fineweb-2 README, https://github.com/huggingface/fineweb-2 | Hugging Face | 2024-12 | accessed 2026-11-08 | high | license
10. Filtered FineWeb-2 subsets exist: epfml/FineWeb2-HQ vie_Latn ~4M rows, epfml/FineWeb2-embedded vie_Latn ~40.7M rows (via search snippet). | FineWeb2-HQ viewer, https://huggingface.co/datasets/epfml/FineWeb2-HQ | EPFL MLO | 2025-?? | accessed 2026-11-08 | low | size/scale
11. **HPLT v2**: 193 languages, ~8T tokens / 52T characters, Internet Archive crawls (complementary to CommonCrawl-based sets), **CC0** license. No Vietnamese figure was retrieved (via search snippet). | "An Expanded Massive Multilingual Dataset…" (HPLT v2), https://arxiv.org/html/2503.10267v3 | HPLT / arXiv | 2025-03 | accessed 2026-11-08 | medium | size/scale, license
12. **MADLAD-400**: 3T tokens, 419 languages, CommonCrawl, manually audited. It has a vi row, but no value could be read. A Vietnamese-only copy exists at `Symato/madlad-400_vi` (vi_clean_*.jsonl.xz) (via search snippet). | MADLAD-400 paper, https://arxiv.org/pdf/2309.04662 ; https://huggingface.co/datasets/Symato/madlad-400_vi | Google / arXiv; Symato | 2023-09 | accessed 2026-11-08 | medium | size/scale
13. **binhvq news-corpus is no longer distributed.** The owner removed all download links and deleted the Drive files. Historical size: ~14.9M articles from 150+ sources, ~18.6GB full text (~111M sentences, sentence-split, deduped, NFC-normalized), crawled about 2018–2021. MIT covers only the code. Article copyright belongs to the publishers/authors. | binhvq/news-corpus README, https://github.com/binhvq/news-corpus | binhvq (GitHub) | 2021-05 (last data update) | accessed 2026-11-08 | high | size/scale, license, version
14. Older reference corpus: Le & Quasthoff (LREC 2016), ~70M sentences / ~4.05B running words, built from web + Wikipedia (via search snippet). | "Construction and Analysis of a Large Vietnamese Text Corpus", https://aclanthology.org/L16-1065/ | ELRA/ACL | 2016-05 | accessed 2026-11-08 | medium | size/scale

### Licensing (Wikipedia)

15. Hugging Face `wikimedia/wikipedia` card lists **cc-by-sa-3.0 and GFDL** (via search snippet). | wikimedia/wikipedia README, https://huggingface.co/datasets/wikimedia/wikipedia | Wikimedia (HF) | 2023/2024 | accessed 2026-11-08 | medium | license
16. Wikipedia's essay "Large language models and copyright" (opinion page, not policy) says the copyright status of LLMs trained on copyrighted text is "not yet fully understood". It also warns that model **output** may not be compatible with CC BY-SA/GFDL. No authoritative answer was found on whether model **weights** trained on CC BY-SA text must be share-alike. Legally open. | https://www.wikipedia.org/wiki/Wikipedia:Large_language_models_and_copyright | Wikipedia community | ?? | accessed 2026-11-08 | medium | license

### Cleaning and dedup practice

17. FineWeb-2 pipeline: **GlotLID** language ID (script-aware, per-language confidence thresholds). **MinHash dedup per language across all snapshots**, keeping one doc per cluster with `minhash_cluster_size` stored for rehydration/upsampling. FineWeb filters retuned per language (char_dup_ratio 0.01→0.1, no short_line filter, no C4 filters). **FTFY** encoding fixes. Email/IP anonymization. Built with datatrove. | huggingface/fineweb-2 README, https://github.com/huggingface/fineweb-2 | Hugging Face | 2024-12 | accessed 2026-11-08 | high | practice
18. Unicode: in Vietnamese, NFD canonical ordering affects only the dot-below tone mark (other tone marks share combining class with circumflex/breve). Precomposed **NFC** is the commonly recommended target (Unicode mailing list, 2002/2006). | Unicode ML archive, https://unicode.org/mail-arch/unicode-ml/y2006-m06/0071.html ; https://www.unicode.org/mail-arch/unicode-ml/y2002-m01/0600.html | Unicode Consortium list | 2002-01, 2006-06 | accessed 2026-11-08 | medium | practice
19. NFC alone does **not** fix old- vs new-style tone placement (e.g. "hoà" vs "hòa"). This is a separate mapping step. Tools: underthesea `text_normalize` (reported to fix non-standard diacritic positions to NFC) and `viet-text-tools` (NFC output, flag for new-style placement) (via search snippets). | https://pypi.org/project/viet-text-tools/0.1.4 ; underthesea | PyPI / Underthesea | ?? | accessed 2026-11-08 | medium | practice

## Leads (not followed this round)

- FineWeb-2 language-distribution CSV in github.com/huggingface/fineweb-2. Use it to confirm vie_Latn column headers (words vs tokens) and the tokenizer used.
- HPLT v2 per-language stats: hplt-project.org/datasets/v2.0 (vi tokens, CC0).
- MADLAD-400 paper Table (vi row) at arXiv 2309.04662. Also check the HF card license (ODC-BY / CC-BY-4.0 per memory, **unverified**).
- OSCAR 23.01 / mC4 vi sizes and licenses were not retrieved this run.
- vietgpt HF org (e.g. `vietgpt/CulturaX` discussions surfaced): check its Wikipedia/news/binhvq re-uploads and their licenses (re-uploads of binhvq conflict with claim 13).
- `data-std/vi-text-corpus` HF dataset card (surfaced, not read).
- Wikimedia stats for the exact current viwiki article count and bot share (stats.wikimedia.org, vi.wikipedia.org Special:Statistics). These hosts were blocked here.
- NeurIPS 2026 "The Tokenization Tax: How Vietnamese Typology Exposes Vocabulary Design Failures" (https://neurips.cc/virtual/2026/164760). Relevant to the tokenizer dimension.

## Looked for but not found

- Authoritative clean-text GB/word count for the current viwiki dump.
- Primary-sourced share of bot-generated stubs on viwiki (only the 2008–09 figures and a weak 58% claim).
- Vietnamese token counts for HPLT v2 and MADLAD-400.
- Any legal ruling or Wikimedia statement that CC BY-SA share-alike does or does not attach to trained model weights.
- Vietnamese-specific dedup/quality-filter thresholds (FineWeb-2 says per-language tuning but no vi values were seen).
