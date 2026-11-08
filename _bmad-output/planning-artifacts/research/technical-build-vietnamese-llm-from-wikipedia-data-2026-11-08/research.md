---
title: 'technical research: Xây LLM tiếng Việt từ dữ liệu Wikipedia'
type: 'technical'
topic: 'Build a Vietnamese LLM from Wikipedia data - feasibility and roadmap'
decision: 'Có nên / làm thế nào để mở rộng NTokenizer (GPT chỉ train trên viwiki) thành một LM tiếng Việt hữu ích; làm cơ sở cho epic'
source: 'native run (4 web-search assistants, 1 round each)'
status: complete
preset: 'standard'
validation: 'normal'
created: '2026-11-08'
updated: '2026-11-08'
claims_verified: 10
claims_unverified: 5
---

# Nghiên cứu kỹ thuật: Xây LLM tiếng Việt từ dữ liệu Wikipedia

**Quyết định phục vụ:** có nên tiếp tục xây model từ Wikipedia không, và lộ
trình nào để biến NTokenizer thành một LM tiếng Việt hữu ích. Kết quả dùng làm
đầu vào cho epic.

> **Giới hạn của lần chạy:** mạng của phiên chỉ mở được `github.com`;
> `huggingface.co`, `arxiv.org`, các trang Wikimedia và `aclanthology.org` bị
> chặn. Số liệu từ những nguồn đó lấy qua đoạn trích tìm kiếm, nên độ tin cậy
> chỉ ở mức **medium/low**. Mỗi nhánh chạy 1 vòng, chưa kiểm chứng chéo độc
> lập.

## Tóm tắt điều hành

**Kết luận:** **Nên làm, nhưng đừng chỉ dùng Wikipedia.**

- **Wikipedia rất tốt để học và làm nền**: sạch, có cấu trúc, giấy phép rõ.
  Nhưng nó **quá nhỏ** để ra một model đa dụng. Viwiki có khoảng 1,1 triệu bài
  [1] và khoảng 275 triệu token (có thể đếm theo từ) [2]. Con số này nhỏ hơn
  khoảng **200 lần** so với phần tiếng Việt của FineWeb-2 (50,9 tỷ) [5] hay
  CulturaX (55,4 tỷ token) [4].
- **PhoGPT**, model tiếng Việt train từ đầu duy nhất được ghi chép đầy đủ, chỉ
  dùng khoảng 1GB Wikipedia trong 482GB dữ liệu [26][27].

Ba phát hiện chi phối kết luận:

1. **Thiếu dữ liệu, không phải thiếu kiến trúc.**
   - Model 125M theo tỷ lệ Chinchilla cần khoảng 2,5 tỷ token. Lặp viwiki 4
     epoch (gần như không mất gì [19]) mới được khoảng 1,1 tỷ.
   - Model nhỏ hiện đại còn train *vượt xa* mức đó: SmolLM2-135M train trên 2
     nghìn tỷ token [15].
   - Ở nanochat, cải thiện lớn nhất đến từ **đổi bộ dữ liệu** chứ không phải đổi
     model [22].
   - Bước đơn giản nhất: thêm **một phần FineWeb-2 tiếng Việt**. Bộ này có giấy
     phép ODC-By [6] và đã qua language ID, MinHash dedup và bộ lọc theo ngôn
     ngữ [6].
2. **Model nhỏ chỉ "thông minh" trong phạm vi hẹp.** TinyStories cho thấy model
   vài triệu tham số viết mạch lạc khi dữ liệu hẹp và đơn giản. Ở khoảng 125M
   trên dữ liệu web rộng, model "hiếm khi" mạch lạc [24]. Vì vậy:
   - Model chỉ train trên Wikipedia sẽ nói giọng bách khoa, trôi chảy nhưng hay
     sai sự kiện. Đó là mục tiêu hợp lý cho một dự án học tập.
   - **Đừng đo bằng VMLU**: ngay PhoGPT-7B5-Instruct chỉ được 24,01 ở zero-shot,
     tức ngang mức đoán ngẫu nhiên [35]. Nên đo bằng **bits-per-byte /
     perplexity** và các bài cloze như `lambada_vi` trong ViLLM-Eval [36].
3. **Fine-tune hội thoại khả thi ngay ở 135M.** SmolLM2-135M-Instruct có SFT và
   DPO [41]. Dữ liệu instruction tiếng Việt có sẵn: vi-alpaca khoảng 50k [37],
   Aya [38], ViHermes-25K [39]. **Lưu ý:** phần lớn là dữ liệu dịch máy hoặc do
   GPT/Gemini sinh ra, nên chất lượng và điều khoản cần kiểm tra từng bộ.

**Lưu ý lớn nhất:** chưa có số đo tốc độ đáng tin cho GPU tiêu dùng hay Mac
(MPS). Ước tính compute dưới đây là **suy luận**, cần đo thực tế trên máy của
bạn.

## D1 — Dữ liệu

| Nguồn | Quy mô tiếng Việt | Giấy phép | Ghi chú | Độ tin cậy |
|---|---|---|---|---|
| Wikipedia tiếng Việt | ~1,1 triệu bài [1]; ~275,6 triệu token [2] | CC BY-SA 3.0 + GFDL [11] | Nhiều bài sơ khai do bot tạo: khoảng 5% vào năm 2009 [3]. Con số "58% do bot" không có nguồn gốc. | medium |
| CulturaX | 55,4 tỷ token, 57,6 triệu tài liệu (0,88%) [4] | "other" (kế thừa mC4/OSCAR); phải xin quyền truy cập | Đã làm sạch và dedup | medium |
| FineWeb-2 `vie_Latn` | 50,9 tỷ (đơn vị có thể là token), 61 triệu tài liệu, ~320GB [5] | **ODC-By** [6] | Có bản lọc chất lượng cao `epfml/FineWeb2-HQ` [7] | high (giấy phép), medium (quy mô) |
| HPLT v2 | Không tìm được số riêng cho tiếng Việt [8] | CC0 | 193 ngôn ngữ, ~8 nghìn tỷ token | low |
| MADLAD-400 | Có bản sao `Symato/madlad-400_vi` [9] | — | Không lấy được quy mô | low |
| binhvq news-corpus | **Đã gỡ** [10] | Bản quyền thuộc các báo | Thận trọng với bản tải lại | high |

- **Giấy phép Wikipedia:** CC BY-SA áp dụng cho văn bản. Việc share-alike có áp
  dụng cho **trọng số model** hay không thì chưa nguồn nào trả lời dứt khoát
  [12]. Đây là vùng pháp lý còn mở. Nên ghi nguồn và giấy phép trong model card.
- **Làm sạch** (pipeline FineWeb-2 [6]): language ID bằng GlotLID, MinHash dedup
  theo từng ngôn ngữ, sửa encoding bằng FTFY, bộ lọc chất lượng theo ngôn ngữ,
  công cụ datatrove.
- **Chuẩn hoá Unicode:** dùng **NFC**. NFC **không** sửa được vị trí dấu kiểu cũ
  và kiểu mới ("hoà" và "hòa"), nên cần thêm một bước chuẩn hoá riêng, ví dụ
  `text_normalize` của underthesea [13][14].

## D2 — Công thức train model nhỏ

- **Ngân sách token:**
  - SmolLM2-135M train trên 2 nghìn tỷ token, khoảng 15.000 token mỗi tham số
    [15]. SmolLM3-3B dùng khoảng 11,2 nghìn tỷ [16]. Llama 3.2 1B dùng đến 9
    nghìn tỷ [17].
  - Lặp dữ liệu đến khoảng 4 epoch gần như tốt bằng dữ liệu mới [19].
- **Kiến trúc** (MobileLLM, ICML 2024) [18]:
  - Dưới 1B tham số, **độ sâu quan trọng hơn độ rộng**.
  - **Gộp chung embedding vào/ra** (weight tying) tiết kiệm 11,8% tham số của
    model 125M mà chỉ giảm 0,2 điểm.
  - **GQA** cũng có lợi. Repo đã có RoPE, GQA, SwiGLU và RMSNorm, đúng hướng.
- **Các công thức tham khảo:**
  - **llm.c GPT-2 124M** [20]: 10 tỷ token, khoảng 90 phút trên 8×A100, khoảng
    $20. Một A100 xử lý khoảng 178 nghìn token/giây.
  - **nanochat** [22]: thay thế nanoGPT (nanoGPT nay đã ngừng phát triển [21]).
    - Một tham số `--depth` tự đặt chiều rộng, số head, learning rate và weight
      decay.
    - Dùng Muon + AdamW; trên Mac (MPS) chạy ở fp32.
    - Đạt chất lượng GPT-2 với khoảng $48 trên 8×H100.
  - **modded-nanogpt** [23]: các mẹo đáng thử là optimizer **Muon**,
    **QK-Norm**, **ReLU²** và value embeddings. Kỷ lục 40 giây trong README chưa
    được kiểm chứng chéo.
- **Ước tính compute:**
  - Số đo có nguồn: card RTX 4070 đạt khoảng 56 nghìn token/giây (một log của
    cộng đồng, độ tin cậy low) [20]. Trên Mac, MPS nhanh hơn CPU khoảng 2–3 lần
    [25].
  - Từ đó **suy ra (chưa đo)**: 2,5 tỷ token cho model 124M mất khoảng 4 giờ
    trên A100, khoảng 12 giờ trên RTX 4070. Trên Mac M-series thì lâu hơn nhiều.

## D3 — Đặc thù tiếng Việt

- **Các LLM tiếng Việt đã có:**
  - **PhoGPT-4B** là model duy nhất train từ đầu được ghi chép đầy đủ [26][27]:
    3,7B tham số, 102 tỷ token, 2 epoch trên 482GB, context 8K, vocab 20K
    (20.480, byte-level BPE).
  - Các model còn lại là **train tiếp hoặc fine-tune từ model gốc tiếng Anh**:
    Vistral (từ Mistral-7B, mở rộng vocab) [28], VinaLLaMA (LLaMA-2, thêm 800 tỷ
    token, trong đó hơn 500 tỷ là dữ liệu tổng hợp) [29], SeaLLM (thêm khoảng
    16K token vào vocab) [30], URA-LLaMa (Wikipedia 1GB + tin tức 22GB) [31],
    Arcee-VyLinh (từ Qwen2.5-3B) [32].
  - Giấy phép trọng số phần lớn là điều khoản riêng hoặc không rõ.
- **Tokenizer:**
  - Tỷ lệ nén tiếng Việt (thấp hơn là tốt hơn): tokenizer của ChatGPT 4,41,
    Llama 3,46, SeaLLM 1,48 [30].
  - Fertility tiếng Việt so với tiếng Anh dao động từ 0,83 đến 1,93 lần. Thành
    phần của vocab quan trọng ngang kích thước vocab [33].
  - Tokenizer riêng cho tiếng Việt, như repo đang làm (32k hoặc 8k), là lựa
    chọn đúng. PhoGPT dùng vocab 20K.
- **Tách từ** (VnCoreNLP) chỉ cần cho PhoBERT, một model encoder [34]. Không có
  bằng chứng cần tách từ cho model sinh văn bản dùng BPE.
- **Đánh giá:**
  - VMLU có 10.880 câu, 58 môn [35]. Quá khó với model nhỏ.
  - Nên dùng bits-per-byte (so sánh được giữa các tokenizer) và các bài cloze
    hoặc đoán từ tiếp theo trong ViLLM-Eval [36].

## D4 — Hậu huấn luyện và đánh giá

- **Dữ liệu instruction tiếng Việt:**
  - `bkai-foundation-models/vi-alpaca`: khoảng 50k [37].
  - Aya: 204k mẫu do người viết, nhiều ngôn ngữ trong đó có tiếng Việt, giấy
    phép Apache-2.0. Chưa rõ bao nhiêu mẫu là tiếng Việt [38].
  - ViHermes-25K: dịch máy, chưa lọc an toàn [39].
  - Các bộ của 5CD-AI, gồm cả cặp DPO [40].
  - **Phần lớn là dịch máy hoặc do model sinh ra.**
- **SFT cho model tí hon:**
  - SmolLM2-135M-Instruct dùng SFT rồi DPO; IFEval 29,9 [41]. Đây là cận trên
    thực tế cho mức 135M.
  - Thiết lập SFT trong bài SmolLM2 là 2 epoch, learning rate 3e-4. Chưa xác
    nhận đây là thiết lập cho bản 135M.
- **Loss masking:** che prompt khi tính loss là mặc định trong các thư viện SFT.
  Nhưng Shi et al. (NeurIPS 2024) thấy *tính loss cả trên prompt* có lợi khi dữ
  liệu nhỏ và câu trả lời ngắn [42]. **Nên thử cả hai.**
- **Contamination:** lọc trùng 13-gram (GPT-3) [46] hoặc 8-gram giữa dữ liệu
  train và bộ đánh giá. Cách này không bắt được câu đã dịch hoặc diễn đạt lại.
- **Lấy mẫu:**
  - min-p (ICLR 2025) [43] đang bị phản biện là lợi ích không rõ [44].
  - Contrastive decoding cần một model "chuyên gia" lớn hơn [45], nên không hợp
    với dự án chỉ có model nhỏ.
  - Không tìm được bằng chứng nào cho repetition penalty.

## Insight xuyên nhánh

1. **Bước tiếp theo giá trị nhất là dữ liệu, kế đến là cách đánh giá.** Dữ liệu
   là nút thắt (D1, D2). Nhưng không có thước đo bits-per-byte cố định (D3) thì
   không thể biết thêm FineWeb-2 hay đổi tokenizer có giúp gì. Vậy **phải dựng
   bộ đánh giá trước khi đổi dữ liệu.**
2. **Tách "model học tập" và "model hữu ích".** Model chỉ train trên viwiki là
   công cụ tuyệt vời để thí nghiệm diễn giải cơ chế (xem buổi brainstorm). Model
   trộn thêm FineWeb-2 và có SFT là hướng tới hữu ích. Nên giữ cả hai như hai
   cấu hình.
3. **Tokenizer gắn chặt với dữ liệu.** Thêm dữ liệu web thì phải train lại
   tokenizer trên hỗn hợp mới. Theo quy tắc của repo, tokenizer phải khớp
   checkpoint, nên mọi checkpoint cũ trở thành không dùng được.

## Khuyến nghị (đầu vào cho epic)

| # | Khuyến nghị | Cơ sở | Độ tin cậy |
|---|---|---|---|
| R1 | Dựng **bộ đánh giá** trước: bits-per-byte trên tập val cố định, cloze `lambada_vi`, kiểm tra trùng n-gram | [36][46] | medium |
| R2 | **Pipeline dữ liệu v2**: NFC, chuẩn hoá vị trí dấu, MinHash dedup, lọc bài sơ khai | [6][13][14][3] | medium |
| R3 | **Trộn thêm FineWeb-2 `vie_Latn`** (bắt đầu 1–5 tỷ token, ưu tiên bản HQ); ghi nguồn và giấy phép | [5][6][7] | medium |
| R4 | **Train lại tokenizer** trên hỗn hợp mới; đo fertility, so sánh vocab 16k/20k/32k | [30][33][27] | medium |
| R5 | **Nâng cấp công thức train**: weight tying, model sâu và mỏng, tuỳ chọn Muon/QK-Norm; đặt ngân sách token ≥ 20 token mỗi tham số, lặp tối đa 4 epoch | [18][19][22][23] | medium |
| R6 | **Đo tốc độ** trên máy thật (GPU/MPS) và ghi vào tài liệu | [20][25] | low → cần đo |
| R7 | **SFT tiếng Việt** với vi-alpaca và Aya; chat template; thử có và không che prompt khi tính loss | [37][38][41][42] | medium |
| R8 | **Model card và giấy phép**: liệt kê nguồn dữ liệu, CC BY-SA, ODC-By | [11][12][6] | medium |

## Câu hỏi còn mở

- Số token thật của viwiki với tokenizer của repo. Đo trực tiếp bằng
  `prepare_dataset.py` (file `meta.json` đã có số này).
- Đơn vị của con số FineWeb-2 (token hay từ?) và quy mô tiếng Việt của
  HPLT/MADLAD. Cần mở quyền truy cập `huggingface.co`.
- Share-alike của CC BY-SA có áp dụng cho trọng số model không. Chưa có câu trả
  lời pháp lý.
- Tốc độ train thực tế trên GPU tiêu dùng và Mac.

## Phụ lục nguồn

| [n] | Hỗ trợ cho | Nhà xuất bản | Ngày | Truy cập | Độ tin cậy |
|---|---|---|---|---|---|
| [1] | ~1,1 triệu bài viwiki | [underthesea — UVW dataset](https://undertheseanlp.github.io/underthesea/docs/datasets/uvw) | 2026 | 2026-11-08 | medium |
| [2] | ~275,6 triệu token viwiki | [wikilangs/vi model card](https://huggingface.co/wikilangs/vi/blob/main/README.md) | 2025 | 2026-11-08 | low |
| [3] | Bài do bot tạo | [Wikipedia — Vietnamese Wikipedia](https://en.wikipedia.org/wiki/Vietnamese_Wikipedia) | 2026 | 2026-11-08 | medium |
| [4] | CulturaX tiếng Việt 55,4 tỷ token | [uonlp — CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) | 2023-09 | 2026-11-08 | medium |
| [5] | FineWeb-2 vie_Latn 50,9 tỷ | [Hugging Face — FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | 2024-12 | 2026-11-08 | medium |
| [6] | ODC-By, pipeline làm sạch | [GitHub — huggingface/fineweb-2](https://github.com/huggingface/fineweb-2) | 2025 | 2026-11-08 | high |
| [7] | Bản lọc chất lượng cao | [EPFL — FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) | 2025 | 2026-11-08 | medium |
| [8] | HPLT v2 | [arXiv — HPLT v2](https://arxiv.org/html/2503.10267v3) | 2025-03 | 2026-11-08 | low |
| [9] | MADLAD-400 tiếng Việt | [Symato — madlad-400_vi](https://huggingface.co/datasets/Symato/madlad-400_vi) | 2024 | 2026-11-08 | low |
| [10] | binhvq đã gỡ | [GitHub — binhvq/news-corpus](https://github.com/binhvq/news-corpus) | 2025 | 2026-11-08 | high |
| [11] | Giấy phép Wikipedia | [Wikimedia — wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia) | 2024 | 2026-11-08 | medium |
| [12] | CC BY-SA và LLM (bài luận) | [Wikipedia — LLMs and copyright](https://www.wikipedia.org/wiki/Wikipedia:Large_language_models_and_copyright) | 2025 | 2026-11-08 | low |
| [13] | NFC/NFD tiếng Việt | [Unicode mailing list](https://www.unicode.org/mail-arch/unicode-ml/y2002-m01/0600.html) | 2002-01 | 2026-11-08 | medium |
| [14] | Chuẩn hoá vị trí dấu | [PyPI — viet-text-tools](https://pypi.org/project/viet-text-tools/0.1.4) | 2025 | 2026-11-08 | low |
| [15] | SmolLM2-135M, 2 nghìn tỷ token | [Hugging Face — SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | 2024-11 | 2026-11-08 | medium |
| [16] | SmolLM3, 11,2 nghìn tỷ token | [Hugging Face — SmolLM3 blog](https://huggingface.co/blog/smollm3) | 2025-07 | 2026-11-08 | medium |
| [17] | Llama 3.2 1B, 9 nghìn tỷ token | [Azure AI — Llama-3.2-1B](https://ai.azure.com/catalog/models/Llama-3.2-1B) | 2024-09 | 2026-11-08 | medium |
| [18] | Sâu > rộng, weight tying | [ICML 2024 — MobileLLM](https://proceedings.mlr.press/v235/liu24ce.html) | 2024-07 | 2026-11-08 | high |
| [19] | Lặp ~4 epoch | [NeurIPS 2023 — Muennighoff et al.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract.html) | 2023-12 | 2026-11-08 | high |
| [20] | llm.c GPT-2 124M, throughput | [GitHub — karpathy/llm.c #481](https://github.com/karpathy/llm.c/discussions/481) | 2024-05 | 2026-11-08 | high |
| [21] | nanoGPT ngừng phát triển | [GitHub — karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | 2025 | 2026-11-08 | high |
| [22] | nanochat, dữ liệu quan trọng nhất | [GitHub — karpathy/nanochat](https://github.com/karpathy/nanochat) | 2026-03 | 2026-11-08 | high |
| [23] | Mẹo của modded-nanogpt | [GitHub — modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | 2026-08 | 2026-11-08 | medium |
| [24] | TinyStories | [arXiv — Eldan & Li](https://arxiv.org/abs/2305.07759) | 2023-05 | 2026-11-08 | medium |
| [25] | MPS nhanh hơn CPU ~3 lần | [Simon Willison — nanoGPT on M2](https://til.simonwillison.net/llms/nanogpt-shakespeare-m2) | 2023 | 2026-11-08 | medium |
| [26] | PhoGPT từ đầu, 102 tỷ token | [GitHub — VinAIResearch/PhoGPT](https://github.com/VinAIResearch/PhoGPT) | 2024 | 2026-11-08 | high |
| [27] | PhoGPT vocab 20K, wiki ~1GB | [VinAI — PhoGPT tech report](https://research.vinai.io/wp-content/uploads/2023/11/PhoGPT.pdf) | 2023-11 | 2026-11-08 | medium |
| [28] | Vistral mở rộng vocab | [Hugging Face — Vistral-7B-Chat](https://huggingface.co/Viet-Mistral/Vistral-7B-Chat) | 2024-01 | 2026-11-08 | medium |
| [29] | VinaLLaMA, 800 tỷ token | [arXiv — VinaLLaMA](https://arxiv.org/abs/2312.11011) | 2023-12 | 2026-11-08 | medium |
| [30] | SeaLLM, tỷ lệ nén tokenizer | [GitHub — DAMO-NLP-SG/SeaLLMs](https://github.com/DAMO-NLP-SG/SeaLLMs) | 2024 | 2026-11-08 | high |
| [31] | URA-LLaMa, dữ liệu | [NUS — Vietnamese LLM project](https://www.comp.nus.edu.sg/~nqduc/projects/vietnamese_llm) | 2024 | 2026-11-08 | medium |
| [32] | Arcee-VyLinh từ Qwen2.5 | [Hugging Face — Arcee-VyLinh](https://huggingface.co/arcee-ai/Arcee-VyLinh) | 2024-11 | 2026-11-08 | medium |
| [33] | Fertility 0,83–1,93 lần | [NeurIPS 2026 abstract](https://neurips.cc/virtual/2026/164760) | 2026 | 2026-11-08 | low |
| [34] | Tách từ chỉ cho PhoBERT | [GitHub — VinAIResearch/PhoBERT](https://github.com/VinAIResearch/PhoBERT) | 2020 | 2026-11-08 | high |
| [35] | VMLU; PhoGPT 24,01 | [GitHub — ZaloAI-Jaist/VMLU](https://github.com/ZaloAI-Jaist/VMLU) | 2023-11 | 2026-11-08 | high |
| [36] | ViLLM-Eval, lambada_vi | [VLSP 2023 — ViLLM-Eval](https://huggingface.co/datasets/vlsp-2023-vllm/ViLLM-Eval) | 2023 | 2026-11-08 | medium |
| [37] | vi-alpaca ~50k | [BKAI — vi-alpaca](https://huggingface.co/datasets/bkai-foundation-models/vi-alpaca) | 2023 | 2026-11-08 | medium |
| [38] | Aya 204k, Apache-2.0 | [Cohere — aya_dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset) | 2024-02 | 2026-11-08 | medium |
| [39] | ViHermes-25K dịch máy | [ontocord — viet4all](https://huggingface.co/datasets/ontocord/viet4all) | 2024 | 2026-11-08 | low |
| [40] | Bộ 5CD-AI, DPO | [5CD-AI — DPO collection](https://huggingface.co/collections/5CD-AI/dpo-instruction-datasets-6608026d80f057ee616e8bf5) | 2024 | 2026-11-08 | low |
| [41] | SmolLM2-135M-Instruct SFT+DPO | [Hugging Face — SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | 2024-11 | 2026-11-08 | medium |
| [42] | Tính loss trên prompt | [NeurIPS 2024 — Shi et al.](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ffb43adf37b3eeaba559098bc084cc6-Abstract.html) | 2024-12 | 2026-11-08 | high |
| [43] | min-p | [ICLR 2025 oral](https://iclr.cc/virtual/2025/oral/31888) | 2025-04 | 2026-11-08 | medium |
| [44] | Phản biện min-p | [NeurIPS 2026 listing](https://neurips.cc/virtual/2026/166291) | 2026 | 2026-11-08 | low |
| [45] | Contrastive decoding | [ACL 2023 — Li et al.](https://aclanthology.org/2023.acl-long.687) | 2023-07 | 2026-11-08 | medium |
| [46] | Lọc trùng 13-gram | [NeurIPS 2020 — GPT-3 supplemental](https://proceedings.nips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Supplemental.pdf) | 2020-12 | 2026-11-08 | high |

## Bản đồ độ cũ

Khung thời gian theo pack: landscape liên quan AI 3 tháng · tín hiệu hệ sinh thái 6 tháng · pattern 2 năm.

| Claim | Ngày | Kiểm lại | Đã cũ? |
|---|---|---|---|
| Vietnamese LLM landscape (PhoGPT etc) | 2024-01 | 2024-04-01 | có |
| SmolLM2-135M token budget | 2024-11 | 2025-02-01 | có |
| FineWeb-2 vi size | 2024-12 | 2025-03-01 | có |
| Data repetition 4 epochs | 2023-12 | 2025-12-01 | có |
| MobileLLM depth>width | 2024-07 | 2026-07-01 | có |
| nanochat recipe/leaderboard | 2026-03 | 2026-09-01 | có |
| modded-nanogpt record | 2026-08 | 2027-02-01 | chưa |

Phần lớn claim về landscape đã quá hạn 3 tháng. Trước khi chốt epic về dữ liệu (R3), cần kiểm lại quy mô FineWeb-2 và các bộ dữ liệu tiếng Việt mới, lý tưởng là sau khi mở quyền truy cập `huggingface.co`.
