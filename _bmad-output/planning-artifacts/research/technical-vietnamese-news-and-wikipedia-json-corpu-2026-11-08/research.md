---
title: 'technical research: Vietnamese online-news + Wikipedia JSON corpus (Markdown) for a small Vietnamese LLM'
type: 'technical'
topic: 'Vietnamese online-news + Wikipedia JSON corpus (Markdown) for training a small Vietnamese LLM'
decision: 'Scope and design of a data-collection epic: which news/wiki sources are legally and technically usable, in what format'
source: 'native run (4 web-search assistants, 1 round each)'
status: complete
preset: 'standard'
validation: 'normal'
created: '2026-11-08'
updated: '2026-11-08'
claims_verified: 7
claims_unverified: 7
claims_disputed: 1
---

# Technical research: a Vietnamese news + Wikipedia corpus for a small LLM

**Decision this research serves:** the scope and design of a data-collection
epic. Which Vietnamese news and Wikipedia sources are legally and technically
usable, and in what format (JSON, Markdown, plain text) should they be stored
and trained on?

> **Limits of this run:** the session's network blocked every Vietnamese news
> site, the Vietnamese law portals, `huggingface.co`, `arxiv.org`,
> `dumps.wikimedia.org` and `enterprise.wikimedia.com`. Only `github.com` and
> some secondary sites could be read in full. Most claims therefore rest on
> search-result snippets of primary pages and are rated **medium** or **low**.
> **Nothing here is legal advice**: the legal findings are what commentators
> report, not a reading of the official texts.

## Executive summary

**Recommendation:** build the epic around **existing corpora and Common Crawl
first, a narrow polite crawler second**. Keep news text **local and never
redistributed**. Store documents as Markdown with front matter for
inspection, but decide by measurement whether to *train* on Markdown or plain
text.

Three findings drive this:

1. **News articles are protected works, and the new AI exception is narrow
   and contested.**
   - Vietnam's IP Law leaves only "pure news of the day" unprotected [1].
     Decree 17/2023 narrows that to short, purely factual items without
     analysis [2].
   - Law 131/2025/QH15 (in force 2026-04-01) reportedly adds Art. 7(5): using
     lawfully published, publicly accessible works for research, testing and
     AI training, provided it does not unreasonably harm rights holders [3].
   - Decree 134/2026 reportedly limits this to **non-commercial** use of
     **lawfully obtained** data [4]. Commentators call the clause contested
     [5][6].
   - Publishers' terms forbid copying without written consent (VnEconomy) [7],
     and GPTBot is the AI bot most often disallowed in Vietnamese robots.txt
     files [8].
   - So: training locally for non-commercial research on lawfully obtained
     text looks closest to what the law allows. **Redistributing article text
     is the high-risk act.** Confidence: low–medium; read the official texts
     before relying on this.
2. **Large Vietnamese news corpora already exist, but none has a verified
   license.**
   - BKAINewsCorpus: about 32M articles, 53 GB after fuzzy dedup [9].
   - binhvq: about 20M articles up to May 2021 [10].
   - VietNews: 150,704 articles from tuoitre.vn, vnexpress.net and
     nguoiduatin.vn [11].
   - All are far larger than anything a polite crawler would collect, but the
     licence of each is unknown, so they are fit only for local research use
     with provenance recorded.
3. **The tooling is mature; Markdown as a *training* format is unproven.**
   - trafilatura has the best open-source F1 (0.958) on the ScrapingHub
     benchmark [12], outputs Markdown, and reads sitemaps and RSS [13].
     FineWeb's pipeline uses it [14].
   - Protego handles robots.txt to RFC 9309 [15]. The Common Crawl index can
     fetch specific news domains' pages without live crawling [16][17].
   - But the major corpora train on **plain extracted text**, and no
     controlled ablation of Markdown vs plain text was found [18].

**Biggest caveat:** the legal picture rests on secondary commentary about
laws that came into force in 2026. It also conflicts with an earlier run on
whether binhvq was taken down (see Contradictions).

## D1 — Legal and ethical constraints

- **Protection.**
  - IP Law Art. 15 excludes "pure news of the day" from protection [1].
  - Decree 17/2023 Art. 8(1) limits that exclusion to items that only state
    facts or figures, with no analysis, commentary or creative expression [2].
  - Most news articles are therefore protected. Confidence: medium.
- **AI-training exception.**
  - Law 131/2025/QH15 amends the IP Law. It was passed 2025-12-10 and took
    effect 2026-04-01. Its new Art. 7(5) reportedly permits using
    lawfully published, publicly accessible works for research, testing and
    AI training without unreasonably harming rights holders [3].
  - Decree 134/2026/NĐ-CP (in force 2026-04-09) reportedly restricts this to
    non-commercial use of lawfully obtained data [4].
  - One commentary says the clause does not permit circumventing protection or
    using data from infringing sites [19].
  - The clause is contested: delegates worried about mass harvesting during
    drafting [5], and critics argue it endangers creators [6].
  - Before 2025, commentators said Vietnam had no text-and-data-mining
    exception [20].
  - Confidence: low–medium (no official text read).
- **AI Law 2025** (Law 134/2025/QH15, in force 2026-03-01): commentators say
  it sets no duty to disclose training data [21].
- **Publisher terms and robots.txt.**
  - VnEconomy forbids copying or republishing without prior written consent
    [7]. Other sites' terms could not be retrieved.
  - A 2026 survey of Vietnamese sites found GPTBot is the most-disallowed AI
    bot [8]. Per-site robots.txt files could not be fetched.
  - Confidence: low.

## D2 — Existing Vietnamese news datasets

| Dataset | Size | Period | Licence | Confidence |
|---|---|---|---|---|
| BKAINewsCorpus [9] | ~32M articles; 53 GB after fuzzy dedup | to Nov 2023 | not found | medium |
| binhvq news corpus [10] | ~20M articles (mirrors 13.95M–19.37M rows) | to May 2021 | not found | medium |
| VietNews [11] | 150,704 articles with summaries | — | original repo: none; an HF mirror says Apache-2.0 (uploader label only) | medium |
| VNTC [22] | ~84K articles, 10 topics | 2007 | not found | low |
| VLSP 2020 MT monolingual [23] | 2M news sentences | 2020 | shared-task terms | low |
| Leipzig Vietnamese [24] | ~4.05B words (Wikipedia + news + web) | 2016 | sources disagree (CC BY 4.0 vs NC vs "copyrighted") | low |

- **No per-domain statistics** (e.g. vnexpress.net share) were found for
  CulturaX, FineWeb-2 or MADLAD-400. FineWeb-2 `vie_Latn` is about 61.1M
  documents and 130 GB according to a third-party repo [25]. News is certainly
  in there, but its share is unknown.
- Vietnamese news is **heavily near-duplicated** (syndication between
  outlets). BKAINewsCorpus and Viettel's NeMo Curator pipeline both report
  large removals from fuzzy dedup [9][26].

## D3 — Collection, extraction and format

- **Extraction:**
  - trafilatura 2.0.0 leads the ScrapingHub benchmark with F1 0.958; next are
    newspaper4k (0.949), readability-lxml (0.922), goose3 (0.896) and jusText
    (0.804) [12].
  - The benchmark does not state page languages, so there is **no
    Vietnamese-specific evidence** [12].
  - trafilatura outputs Markdown, JSON and plain text, takes sitemaps and
    RSS/Atom feeds as URL sources, and is Apache-2.0 [13].
  - Other pipelines differ (DCLM uses resiliparse, Nemotron-CC uses jusText),
    and the best extractor depends on the content type [27][28].
- **Reference pipeline** (FineWeb in datatrove) [14]:
  - Steps: WARC reader → URL filter → `Trafilatura(favour_precision=True)` →
    language ID → Gopher/C4/FineWeb quality filters → MinHash dedup (5-grams,
    14 buckets × 8 hashes) → PII.
  - FineWeb-2 uses GlotLID with per-language thresholds [29].
- **Polite collection:**
  - Protego parses robots.txt to RFC 9309, including Crawl-delay and
    sitemaps [15].
  - The **Common Crawl index** returns each page's `warc_filename`, offset and
    length, and the CDX API filters by domain [16][17]. Specific news sites can
    therefore be pulled from Common Crawl's existing captures instead of being
    crawled live.
  - Precedent: VietVault converted Common Crawl HTML to Markdown with
    trafilatura [30].
- **Markdown vs plain text:**
  - FineWeb and the other major English corpora train on plain extracted text
    [14][18].
  - Markdown-preserving corpora exist (e.g. Markdownified Wikipedia [31]), but
    no ablation showing Markdown helps or hurts a small model was found.
    Confidence: medium (absence of evidence).

## D4 — Wikipedia as JSON, Markdown and mixing

| Source | Format | Status | Path to Markdown |
|---|---|---|---|
| WikiExtractor `--json` (repo's current pipeline) [32] | JSONL: `id`, `revid`, `url`, `title`, `text` | works on any XML dump | plain text only; table/list handling undocumented |
| HF `wikimedia/wikipedia` `20231101.vi` [33] | parquet, ~1.29M rows | snapshot from Nov 2023 (old) | plain text (mwparserfromhell) |
| Wikimedia Enterprise HTML dumps [34][35] | NDJSON with `article_body.html` + `wikitext` | public mirrors stopped 2025-03-24; now via Snapshot API with a free account | **best**: HTML → Markdown keeps headings, lists and tables |
| CirrusSearch dumps [32][36] | Elasticsearch bulk JSON, templates expanded | jobs failed 2025-07; status for viwiki unconfirmed | plain text |
| Structured Contents / Parsed Tables [37][38] | pre-parsed JSON | beta; Vietnamese not tested | tables as JSON |

- **Precedent for Markdown:** `marin-community/wikipedia-markdown` converted
  Enterprise HTML to Markdown, kept tables, and reached about 8.59B tokens
  [31]. Databricks did the same with a custom parser [37].
- **Licence:** Wikipedia text is CC BY-SA 4.0 since June 2023 [39]. Reuse
  needs attribution and share-alike; media files carry their own licences
  [40].
- **Mixing:**
  - Wikipedia is usually upsampled: The Pile used 3 epochs (1.53% raw) [41],
    and LLaMA sampled it at 4.5% for about 2 epochs [42].
  - Data Mixing Laws show proportions can be tuned with small runs [43].
  - Educational/encyclopedic data helps knowledge benchmarks, while web data
    helps commonsense ones: FineWeb-Edu vs DCLM in SmolLM2 [44].
  - **No source gives an encyclopedic-vs-news ratio.** Treat any starting
    ratio as an experiment.

## Cross-dimension insights

1. **The legal and technical routes point the same way.** Common Crawl
   already holds captures of Vietnamese news sites [16][17]. Using it (or the
   existing news corpora) avoids live-crawling sites whose terms forbid
   copying [7]. It also puts the project nearest the "lawfully obtained,
   non-commercial" condition reported for Art. 7(5) [3][4]. A live crawler is
   then only needed for recent articles, and only for sites whose robots.txt
   allows it.
2. **The binding constraint is redistribution, not training.** Every news
   source has an unverified licence [9][10][11], and the law's risk centres on
   copying and republishing. The epic should therefore commit **code,
   manifests (URL, date, hash) and statistics**, never article text. The
   model card must say so.
3. **Store Markdown, train on what measures best.** Markdown keeps structure
   for humans, and Wikipedia tables survive only via the Enterprise HTML path
   [31]. But there is no evidence it helps a small model [18]. With the BPB
   evaluation from Epic 1, the project can test Markdown against plain text
   itself.
4. **Dedup is mandatory for Vietnamese news.** Syndication makes near-duplicates
   common [9][26], and MinHash with FineWeb's settings is the established fix
   [14].

## Contradictions

- **binhvq takedown.** The previous run read the binhvq GitHub repo directly,
  which says the owner removed all download links (high confidence). This
  run found the corpus still widely mirrored on Hugging Face and no stated
  reason for removal [10]. Both can be true: the original was withdrawn while
  third-party mirrors remain. Treat the mirrors as **licence unknown**.

## Recommendations (input for the epic)

| # | Recommendation | Basis | Confidence |
|---|---|---|---|
| N1 | Upgrade the Wikipedia step to keep WikiExtractor `--json` and add HF `20231101.vi` as a quick option. Add an Enterprise HTML → Markdown path (needs a free account) for structure and tables. Record CC BY-SA 4.0 attribution. | [31][32][33][34][39] | medium |
| N2 | Define one **document schema** for all sources: Markdown file with YAML front matter (`source`, `url`, `title`, `published`, `retrieved`, `license`, `sha1`), plus a JSONL manifest. | [13][31] | medium |
| N3 | Ingest existing news corpora (BKAINewsCorpus, binhvq mirror, VietNews) into that schema for **local research use only**, marked licence-unknown. | [9][10][11] | medium |
| N4 | Pull Vietnamese news domains from **Common Crawl** via the CDX/columnar index and extract with trafilatura (Markdown). | [12][13][16][17][30] | medium |
| N5 | Optional polite live crawler for recent articles: sitemaps/RSS only, Protego robots.txt checks, rate limit, identifying user agent, skip any site whose terms or robots.txt forbid it. | [7][8][13][15] | low–medium |
| N6 | Clean and dedup: NFC + tone normalization (existing Epic 2), GlotLID language ID, MinHash (5-grams, 14×8), boilerplate filters. | [14][29][9] | medium |
| N7 | **No redistribution:** `.gitignore` all text; commit only code, manifests and stats; add a legal note citing Art. 7(5) and Decree 134 with "verify official text". | [3][4][7] | medium |
| N8 | Mixing experiment: Wikipedia upsampled to ≤ 3 epochs; compare a few news:wiki ratios and Markdown vs plain text using BPB (Epic 1). | [41][42][43][44] | medium |

## Open questions

- The official wording of Art. 7(5) and Decree 134/2026: commercial
  restrictions, attribution, and whether robots.txt counts as an opt-out.
  Needs the law portals allowed in the network settings, or a lawyer.
- Per-site robots.txt and terms for the major news sites. Needs the sites
  reachable.
- Licences of BKAINewsCorpus, binhvq mirrors and VietNews.
- Share of news domains in FineWeb-2 `vie_Latn`. Could be measured by
  sampling its URLs.
- Whether Markdown helps a small model. Measure it in the project.

## Source appendix

| [n] | Supports | Publisher | Date | Accessed | Confidence |
|---|---|---|---|---|---|
| [1] | Art. 15 "news of the day" not protected | [LuatVietnam](https://luatvietnam.vn/tin-van-ban-moi/tin-tuc-thoi-su-khong-duoc-bao-ho-quyen-tac-gia-186-15809-article.html) | 2023 | 2026-11-08 | medium |
| [2] | Decree 17/2023 Art. 8(1) narrows the exclusion | [Thư viện Pháp luật](https://thuvienphapluat.vn/hoi-dap-phap-luat/doi-tuong-khong-thuoc-pham-vi-bao-ho-quyen-tac-gia-96705.html) | 2023 | 2026-11-08 | medium |
| [3] | Law 131/2025/QH15, Art. 7(5) AI-training exception | [LuatVietnam (EN)](https://english.luatvietnam.vn/law-amending-and-supplementing-a-number-of-articles-of-the-law-on-intellectual-property-dated-december-10-2025-of-the-national-assembly-no-131-2025-423358-doc1.html) | 2025-12 | 2026-11-08 | medium |
| [4] | Decree 134/2026 conditions (non-commercial, lawfully obtained) | [Viet An Law](https://vietanlaw.com/decree-134-2026-nd-cp-on-ai-copyright-and-ai-generated-works-in-vietnam/) | 2026-04 | 2026-11-08 | low |
| [5] | Clause contested; delegates' concerns | [Tilleke & Gibbins](https://www.tilleke.com/insights/is-vietnam-pioneering-the-legal-framework-for-ai-and-ip/) | 2026 | 2026-11-08 | medium |
| [6] | Critique of clause 7.5 | [The Vietnamese](https://www.thevietnamese.org/2026/01/the-death-of-authorship-how-clause-7-5-of-the-ip-law-endangers-viet-nams-creative-class) | 2026-01 | 2026-11-08 | low |
| [7] | VnEconomy terms forbid copying | [VnEconomy — Điều khoản sử dụng](https://vneconomy.vn/dieu-khoan-su-dung) | — | 2026-11-08 | medium |
| [8] | GPTBot most-disallowed in VN robots.txt | [Viblo](https://viblo.asia/p/kiem-tra-bot-ai-co-doc-duoc-website-khong-robotstxt-tuong-lua-va-noi-dung-render-bang-js-G24B8W2RLz3) | 2026 | 2026-11-08 | low |
| [9] | BKAINewsCorpus ~32M articles, 53 GB deduped | [BKAI — BKAINewsCorpus](https://huggingface.co/datasets/bkai-foundation-models/BKAINewsCorpus) | 2023-11 | 2026-11-08 | medium |
| [10] | binhvq ~20M articles; mirrors | [vietgpt — binhvq_news_vi](https://huggingface.co/datasets/vietgpt/binhvq_news_vi) | 2021-05 | 2026-11-08 | medium |
| [11] | VietNews 150,704 articles | [GitHub — ThanhChinhBK/vietnews](https://github.com/ThanhChinhBK/vietnews) | 2019 | 2026-11-08 | medium |
| [12] | Extraction benchmark F1 | [GitHub — scrapinghub/article-extraction-benchmark](https://github.com/scrapinghub/article-extraction-benchmark) | 2024 | 2026-11-08 | high |
| [13] | trafilatura formats, feeds, licence | [GitHub — adbar/trafilatura](https://github.com/adbar/trafilatura) | 2025 | 2026-11-08 | high |
| [14] | FineWeb pipeline and MinHash settings | [GitHub — datatrove fineweb.py](https://github.com/huggingface/datatrove/blob/main/examples/fineweb.py) | 2024 | 2026-11-08 | high |
| [15] | Protego robots.txt parser | [GitHub — scrapy/protego](https://github.com/scrapy/protego) | 2025 | 2026-11-08 | high |
| [16] | Common Crawl columnar index | [GitHub — commoncrawl/cc-index-table](https://github.com/commoncrawl/cc-index-table) | 2024 | 2026-11-08 | high |
| [17] | CDX server API domain queries | [GitHub — pywb CDX Server API](https://github.com/webrecorder/pywb/wiki/CDX-Server-API) | 2023 | 2026-11-08 | high |
| [18] | Major corpora train on plain text | [GitHub — huggingface/datatrove](https://github.com/huggingface/datatrove) | 2025 | 2026-11-08 | medium |
| [19] | No circumvention / no infringing sources | [Viet An Law — IP Law 2025 update](https://vietanlaw.com/vietnam-ip-law-2025-update-copyright-exceptions-for-ai-model-training) | 2026 | 2026-11-08 | low |
| [20] | No TDM exception before 2025 | [Tạp chí Công Thương](https://tapchicongthuong.vn/ngoai-le-quyen-tac-gia-doi-voi-huan-luyen-tri-tue-nhan-tao-tao-sinh-kinh-nghiem-mot-so-quoc-gia-va-goi-mo-cho-viet-nam-219760.htm) | 2024 | 2026-11-08 | low |
| [21] | AI Law 2025, no disclosure duty | [Thư viện Pháp luật — Law 134/2025/QH15](https://thuvienphapluat.vn/van-ban/EN/Cong-nghe-thong-tin/Law-134-2025-QH15-Artificial-Intelligence/689735/tieng-anh.aspx) | 2025-12 | 2026-11-08 | low |
| [22] | VNTC | [GitHub — congphuc/VNTC](https://github.com/congphuc/VNTC) | 2007 | 2026-11-08 | low |
| [23] | VLSP 2020 news sentences | [VLSP 2020 paper](https://preview.aclanthology.org/fix_video/2020.vlsp-1.18.pdf) | 2020 | 2026-11-08 | low |
| [24] | Leipzig Vietnamese corpus | [LREC 2016 — Le & Quasthoff](https://aclanthology.org/L16-1065/) | 2016 | 2026-11-08 | low |
| [25] | FineWeb-2 vie_Latn ~61M docs | [GitHub — huggingface/fineweb-2](https://github.com/huggingface/fineweb-2) | 2025 | 2026-11-08 | medium |
| [26] | Viettel NeMo Curator dedup | [NVIDIA blog](https://developer.nvidia.com/blog/processing-high-quality-vietnamese-language-data-with-nvidia-nemo-curator/) | 2024 | 2026-11-08 | medium |
| [27] | Extractor choice varies by content type | [arXiv — Beyond a Single Extractor](https://arxiv.org/html/2602.19548v1) | 2026-02 | 2026-11-08 | low |
| [28] | Extractor comparison | [Thunderbit blog](https://thunderbit.com/blog/article-extraction-libraries-compared) | 2025 | 2026-11-08 | low |
| [29] | FineWeb-2 GlotLID, per-language thresholds | [GitHub — huggingface/fineweb-2](https://github.com/huggingface/fineweb-2) | 2025 | 2026-11-08 | high |
| [30] | VietVault: CC → Markdown with trafilatura | [nampdn-ai — vietvault](https://huggingface.co/datasets/nampdn-ai/vietvault) | 2023 | 2026-11-08 | low |
| [31] | Markdownified Wikipedia keeps tables | [Marin — wikipedia-markdown](https://huggingface.co/datasets/marin-community/wikipedia-markdown/blob/main/README.md) | 2025 | 2026-11-08 | medium |
| [32] | WikiExtractor JSON fields | [GitHub — attardi/wikiextractor](https://github.com/attardi/wikiextractor) | 2023 | 2026-11-08 | high |
| [33] | HF wikimedia/wikipedia 20231101.vi | [Wikimedia — wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia/tree/40e2363d052e6d0858ae3a34e78921f82ad3062d/20231101.vi) | 2023-11 | 2026-11-08 | medium |
| [34] | Enterprise HTML NDJSON fields | [Wikimedia mirror — enterprise_html](https://cdimage.debian.org/mirror/wikimedia.org/other/enterprise_html/index.html) | 2025-03 | 2026-11-08 | medium |
| [35] | Public mirrors stopped 2025-03-24 | [SUNET mirror — enterprise_html](https://archive.sunet.se/mirror/wikimedia.org/other/enterprise_html/index.html) | 2025-03 | 2026-11-08 | medium |
| [36] | CirrusSearch dump failures | [Phabricator T400158](https://phabricator.wikimedia.org/T400158) | 2025-07 | 2026-11-08 | low |
| [37] | Databricks Wikipedia → Markdown | [Wikimedia Enterprise blog](https://enterprise.wikimedia.com/blog/how-databricks-parsed-wikipedia-to-markdown-with-python/) | 2025 | 2026-11-08 | low |
| [38] | Parsed Tables beta | [Wikimedia Enterprise blog](https://enterprise.wikimedia.com/blog/parsed-wikipedia-tables/) | 2026-02 | 2026-11-08 | low |
| [39] | CC BY-SA 4.0 since June 2023 | [Creative Commons](https://creativecommons.org/2023/06/29/wikipedia-moves-to-cc-4-0-licenses/) | 2023-06 | 2026-11-08 | high |
| [40] | Reuse requirements | [Wikipedia — Reusing content](https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content) | 2025 | 2026-11-08 | medium |
| [41] | The Pile: Wikipedia 3 epochs | [arXiv — Gao et al.](https://arxiv.org/abs/2101.00027) | 2020-12 | 2026-11-08 | medium |
| [42] | LLaMA: Wikipedia 4.5% | [arXiv — Touvron et al.](https://arxiv.org/abs/2302.13971) | 2023-02 | 2026-11-08 | medium |
| [43] | Data Mixing Laws | [arXiv — Ye et al.](https://arxiv.org/abs/2403.16952) | 2024-03 | 2026-11-08 | medium |
| [44] | FineWeb-Edu vs DCLM in SmolLM2 | [arXiv — SmolLM2](https://arxiv.org/abs/2502.02737) | 2025-02 | 2026-11-08 | medium |

## Staleness map

Pack windows: versions ≤ 1 month · AI-adjacent landscape ≤ 3 months · law and publisher terms 6 months (chosen for this run) · patterns ≤ 2 years.

| Claim | Date | Re-check by | Stale? |
|---|---|---|---|
| BKAINewsCorpus availability/licence | 2023-11 | 2024-02-01 | yes |
| trafilatura version/benchmark | 2024-01 | 2024-02-01 | yes |
| Wikimedia Enterprise dump access | 2025-03 | 2025-04-01 | yes |
| Publisher ToS and robots.txt | 2026-01 | 2026-07-01 | yes |
| Art.7(5) / Decree 134 legal status | 2026-04 | 2026-10-01 | yes |
| Mixing evidence (SmolLM2, Pile) | 2025-02 | 2027-02-01 | no |

Re-check the legal status (official texts of Art. 7(5) and Decree 134) and the Wikimedia Enterprise access path **before implementing** the news and Wikipedia stories; both are already past their window.
