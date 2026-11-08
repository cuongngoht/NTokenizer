# D3 — Tooling & format digest (round 1, agent 1)

Accessed: 2026-11-08. Blocked this run: trafilatura.readthedocs.io, arxiv.org, preview.aclanthology.org (DNS not allowed), so several paper-level claims come from search-result snippets only. They are marked "secondary" and given lower confidence.

## Claims

Format: claim | source | publisher | pub_date | accessed | confidence | class

### Extraction quality and tools

1. In the ScrapingHub Article Extraction Benchmark, trafilatura 2.0.0 has the best F1 among open-source libraries: 0.958 (P 0.938, R 0.978). Next are newspaper4k 0.9.3.1 at 0.949 (P 0.964, R 0.934), news-please 0.948, readability-lxml 0.8.4.1 at 0.922, goose3 0.896, newspaper3k (2019 run) 0.912, boilerpipe 0.860 and justext 3.0.2 at 0.804. The html-text baseline scores 0.665. Trafilatura leads on recall; newspaper4k leads on precision. | scrapinghub/article-extraction-benchmark README, https://github.com/scrapinghub/article-extraction-benchmark | Zyte/ScrapingHub (GitHub) | 2021-04 (newer tool rows undated; repo archived 2026-06) | 2026-11-08 | high | performance
2. The ScrapingHub benchmark README does not state the page languages, so it is not evidence about Vietnamese pages. The repo is archived and read-only as of 2026-06-24. | same as 1 | Zyte | 2026-06 | 2026-11-08 | high | version
3. Trafilatura outputs TXT, Markdown, CSV, JSON, HTML, XML and XML-TEI. It supports sitemaps (TXT, XML) and feeds (ATOM, JSON, RSS) as URL sources, and does URL filtering and deduplication. The licence is Apache 2.0 from v1.8.0 (earlier versions were GPLv3+). The README says it is used by HuggingFace, IBM, Microsoft Research, NVIDIA, AI2 and Stanford; that is the project's own claim. | adbar/trafilatura README, https://github.com/adbar/trafilatura | A. Barbaresi (GitHub) | undated (read 2026-11) | 2026-11-08 | high (features); low (adoption claims) | version/practice
4. The trafilatura README cites benchmark wins: best overall in Lejeune & Barbaresi 2020, and best single tool by ROUGE-LSum mean F1 in Bevendorff et al. 2023. These are self-reported citations; I did not verify them against the papers. | same as 3 | Barbaresi | n/a | 2026-11-08 | medium | performance
5. Secondary: a 22-fixture blog comparison puts trafilatura and resiliparse level on content-token F1 (0.9697 vs 0.9681). Resiliparse's recall is lower (0.9054) because it misses article text inside list elements. | "article-extraction-libraries-compared", https://thunderbit.com/blog/article-extraction-libraries-compared | Thunderbit (vendor blog) | unknown | 2026-11-08 | low | performance
6. Secondary (WCXB paper, arXiv 2605.21097, search snippet only): the top extractors converge on article pages (F1 ≈ 0.93), so the differences between the leading tools on news articles are small. | https://arxiv.org/html/2605.21097v1 | arXiv | 2026-05 | 2026-11-08 | low (unread) | performance

### Which tools the big pipelines use

7. The FineWeb reference pipeline in datatrove runs in this order: WarcReader → URLFilter → `Trafilatura(favour_precision=True)` → LanguageFilter → GopherRepetitionFilter → GopherQualityFilter → C4QualityFilter(filter_no_terminal_punct=False) → FineWebQualityFilter → JsonlWriter. MinHash dedup follows, then `PIIFormatter()` after `MinhashDedupFilter`. | datatrove examples/fineweb.py, https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py | Hugging Face | undated (read 2026-11) | 2026-11-08 | high | practice
8. FineWeb's MinHash settings: sha1, 64-bit precision, 14 buckets × 8 hashes per bucket, 5-grams. | same as 7 | Hugging Face | — | 2026-11-08 | high | practice
9. The datatrove README describes Trafilatura as "the most commonly used extractor in datatrove". It also provides MinHash, sentence-level and ExactSubstr dedup examples, and FastText language stats. | https://github.com/huggingface/datatrove | Hugging Face | undated | 2026-11-08 | high | practice
10. FineWeb-2 identifies languages with GlotLID (2000+ labels, script-aware), using a separate confidence threshold for each language. It deduplicates globally within each language, keeps one document per cluster, and records `minhash_cluster_size`. Filter thresholds and stopwords are tuned per language in `configs/{iso3}_{script}.yml`. It also applies FTFY encoding fixes, anonymises emails and IPs, and adds code to fix trafilatura table artefacts. Vietnamese is not mentioned on the page. | huggingface/fineweb-2 README, https://github.com/huggingface/fineweb-2 | Hugging Face | 2025 (paper arXiv 2506.20920) | 2026-11-08 | high | practice
11. FineWeb-2 "rehydrates" by upsampling documents according to their duplicate-cluster size, so dedup is not purely removal. | same as 10 | HF | 2025 | 2026-11-08 | high | practice
12. Secondary: DCLM-Baseline uses resiliparse, FineWeb(-Edu) uses trafilatura, and Nemotron-CC uses jusText because it yielded more tokens after the FineWeb-Edu classifier. Dolma is reported as using WET/resiliparse. | search snippets of Nemotron-CC (arXiv 2412.02595) and Li et al. (arXiv 2602.19548) | arXiv | 2024-12 / 2026-02 | 2026-11-08 | medium | practice
13. Secondary: DCLM reportedly found resiliparse about 2.5 Core points better than WET or trafilatura and about 8× faster. Other studies disagree. Blu-WERP's aggregate scores are trafilatura 44.29, resiliparse 44.02, jusText 44.74 and WET 40.57. MegaMath found trafilatura cleaner than resiliparse for math. On tables, one report (via the AICC paper) gives resiliparse 11.9 vs trafilatura 3.7 on WikiTableQuestions. Net: no single extractor wins everywhere. | emergentmind DCLM summary; Blu-WERP arXiv 2511.18054; MegaMath arXiv 2504.02807; AICC arXiv 2511.16397 (snippets) | various | 2024–2025 | 2026-11-08 | low–medium | performance
14. Secondary: Li et al., "Beyond a Single Extractor" (Findings of EACL 2026), reports that the best extractor depends on content type and that jusText loses up to 3.6 points on HumanEval. It also reports that better extraction yields more tokens, which helps in data-constrained settings. | https://arxiv.org/html/2602.19548v1 | ACL/arXiv | 2026-02 | 2026-11-08 | medium (abstract-level) | performance

### Polite crawling

15. Protego is a pure-Python robots.txt parser that implements RFC 9309 matching and adds Crawl-delay, Request-rate, Visit-time, Host and Sitemaps. Its README benchmark has stdlib `urllib.robotparser` about 75% slower at matching and about 71% slower at parsing; it also lacks Request-rate, Visit-time and Host. Protego leaves fetching robots.txt, status codes, redirects and caching to the caller. | scrapy/protego README, https://github.com/scrapy/protego | Scrapy project | undated | 2026-11-08 | high | practice/performance
16. Common Crawl's columnar index (cc-index-table) exposes `warc_filename`, `warc_record_offset` and `warc_record_length` for each capture. The repo ships example SQL for single-domain record selection (`get-records-of-domain.sql`) and for language selection (`content_languages`). Athena and Spark (`CCIndexWarcExport`) can export the matching WARC records. This makes it possible to pull specific news domains without crawling them live. | commoncrawl/cc-index-table, https://github.com/commoncrawl/cc-index-table | Common Crawl | undated | 2026-11-08 | high | practice
17. The CDX server API supports `url=*.example.com` / `matchType=domain`, `matchType=prefix`, `output=json`, `filter=` (exact, regex or negated, e.g. `=status:200`) and `fl=` returning `filename,offset,length`. These are the fields needed to fetch a WARC record by byte range. (This is the pywb spec, which index.commoncrawl.org follows; the CC endpoint itself was not fetched.) | pywb wiki CDX-Server-API, https://github.com/webrecorder/pywb/wiki/CDX-Server-API | Webrecorder | undated | 2026-11-08 | medium-high | practice

### Vietnamese-specific

18. Secondary: VietVault (nampdn-ai) converted pre-2023 Common Crawl HTML to Markdown with trafilatura, then deduplicated and filtered for Vietnamese. This is a precedent for Vietnamese Markdown pretraining data. | https://huggingface.co/datasets/nampdn-ai/vietvault (snippet) | HF dataset card | unknown | 2026-11-08 | low-medium | practice
19. Secondary: BKAINewsCorpus merged about 32M Vietnamese news articles and fuzzy-deduplicated them to about 53 GB. Vi-Mistral-X cut CulturaX-vi from 54.99M to 7.33M documents with n-gram near-dup filtering, which shows very heavy duplication in Vietnamese web and news text. | BKAI card (HF); Vi-Mistral-X arXiv 2403.15470 (snippets) | HF / arXiv | 2023–2024 | 2026-11-08 | medium | practice
20. Secondary: NVIDIA's blog on Viettel's Vietnamese NeMo Curator pipeline reports that exact and fuzzy dedup plus heuristic and classifier filters gave about 10% higher accuracy, 3× faster training and 60% less data (vendor-reported). | https://developer.nvidia.com/blog/processing-high-quality-vietnamese-language-data-with-nvidia-nemo-curator/ | NVIDIA | unknown | 2026-11-08 | low-medium | performance

### Markdown vs plain text

21. The major English web corpora covered here train on plain extracted text rather than Markdown. FineWeb uses the trafilatura default text output: no output format is set in fineweb.py, so this is inferred from claim 7. No source in this run reports a controlled pretraining ablation of Markdown-preserving vs plain-text extraction. AICC (MinerU-HTML) is a corpus that keeps Markdown and argues that extraction quality matters, but I only saw its judging prompt, not its outcome numbers. | fineweb.py (claim 7); AICC arXiv 2511.16397 (snippet) | HF / arXiv | 2025 | 2026-11-08 | medium (inference) | practice

## Leads
- Li et al. 2026, "Beyond a Single Extractor" (arXiv 2602.19548): check whether it compares Markdown output with plain output.
- AICC / MinerU-HTML (arXiv 2511.16397): downstream results for Markdown-preserving extraction.
- WCXB benchmark (arXiv 2605.21097): multilingual coverage; check whether it includes Vietnamese pages.
- trafilatura docs (evaluation.html, usage pages): `output_format="markdown"`, `include_formatting`, and the version that added Markdown. Unreachable this run.
- DCLM paper: the primary resiliparse-vs-trafilatura ablation.
- index.commoncrawl.org plus `content_languages='vie'` to size Vietnamese news domains already present in CC.
- FineWeb-2 `configs/vie_Latn.yml`: Vietnamese-specific thresholds and stopwords.
- Scrapy settings for politeness: AUTOTHROTTLE, ROBOTSTXT_OBEY, HTTPCACHE, JOBDIR resume. Not verified this run.

## Looked for but not found
- Any extraction benchmark that includes Vietnamese news pages.
- A controlled ablation of Markdown vs plain text for LM pretraining, whether for small or large models.
- Whether Scrapy uses Protego by default (the Protego README does not say).
- Evidence on newspaper4k/readability quality for non-English pages.
- Front-matter metadata conventions (title/date/url/source) in published corpora. FineWeb/FineWeb-2 store metadata as JSONL fields (e.g. `minhash_cluster_size`), not as in-text front matter. Field lists were not fully verified.
