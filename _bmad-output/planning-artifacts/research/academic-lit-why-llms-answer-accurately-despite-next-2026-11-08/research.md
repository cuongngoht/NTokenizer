---
title: 'academic-lit research: Tại sao LLM trả lời đúng dù chỉ dự đoán token tiếp theo'
type: 'academic-lit'
topic: 'Why LLMs answer accurately despite next-token prediction'
decision: 'Giữ, sửa, bổ sung gì trong docs/why_llms_are_accurate.vi.md'
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

# Tổng quan tài liệu: Tại sao LLM trả lời đúng dù chỉ dự đoán token tiếp theo

**Quyết định mà nghiên cứu này phục vụ:** giữ, sửa, bổ sung gì trong
`docs/why_llms_are_accurate.vi.md`.

> **Giới hạn của lần chạy:** mạng của phiên chặn `arxiv.org` và
> `transformer-circuits.pub`. Phần lớn số liệu lấy từ trang proceedings, trang
> nhà xuất bản và đoạn trích kết quả tìm kiếm của trang gốc, không đọc toàn văn
> PDF. Những số liệu đó được đánh dấu độ tin cậy *medium*. Mỗi nhánh chỉ chạy
> 1 vòng, chưa có trợ lý độc lập nào kiểm chứng chéo.

## Tóm tắt điều hành

**Kết luận:** luận điểm chính của tài liệu đứng vững. Dự đoán token tốt buộc
model học cấu trúc bên trong, và điều đó có bằng chứng thực nghiệm. Nhưng nhiều
chỗ cần **thêm điều kiện** hoặc **sửa** để không phóng đại.

Ba phát hiện chi phối kết luận:

1. **Bằng chứng về cấu trúc bên trong là thật, nhưng không trọn vẹn.**
   Othello, cờ vua, phép cộng modulo và các mạch trong Claude 3.5 Haiku đều cho
   thấy model tính toán chứ không chỉ tra cứu [12][14][17][24]. Mặt khác, model
   có thể vượt qua bài kiểm tra bằng probe trong khi mô hình thế giới bên trong
   vẫn không nhất quán [15]. Phương pháp của Anthropic cũng chỉ bắt được "một
   phần" của toàn bộ tính toán [24].
2. **Hậu huấn luyện và test-time compute là nguồn chính của độ chính xác ở
   model hiện đại.** Ví dụ: InstructGPT 1.3B được ưa thích hơn GPT-3 175B [26];
   DeepSeek-R1 (Nature 2025) học được tự kiểm tra chỉ nhờ RL với phần thưởng
   đúng/sai [27]. **Tuy nhiên**, RL có dạy được suy luận *mới* hay chỉ chọn lại
   cái base model đã có là tranh luận **còn mở** [29][30].
3. **Một số câu trong tài liệu bị phóng đại.** "LLM nén ảnh tốt hơn PNG" chỉ
   đúng khi *không tính kích thước model* [1]. "Transformer ≈ TC0" cần ghi rõ
   giả định về độ chính xác số [32][33]. Chain-of-thought tăng độ chính xác,
   nhưng văn bản CoT *không phản ánh trung thực* cách model thật sự tính
   (chỉ khoảng 25–39%) [37].

**Lưu ý lớn nhất:** nhiều kết quả về cách model hoạt động bên trong chỉ đến từ
một lab, trên một model, và chưa qua bình duyệt (Anthropic) [23][24][25]. Nên
xem chúng là bằng chứng mạnh nhưng chưa được tái lập độc lập.

## D1 — Nền tảng lý thuyết

- **Dự đoán tương đương nén.** Theo arithmetic coding, log-loss chính là độ
  dài mã nén. Delétang et al. (ICLR 2024, đã bình duyệt): Chinchilla 70B nén
  ảnh ImageNet còn 43,4% (PNG 58,5%) và âm thanh LibriSpeech còn 16,4%
  (FLAC 30,3%) [1]. **Điều kiện:** đây là tỷ lệ "thô", không tính kích thước
  model. Nếu tính cả model thì LLM nén kém hơn nhiều. Độ tin cậy: high cho số
  liệu, medium cho phần điều kiện.
- **Nén tương quan với năng lực.** Trên khoảng 30 LLM, điểm benchmark trung
  bình gần như tỷ lệ tuyến tính với hiệu quả nén (r ≈ −0,95; Huang et al.,
  COLM 2024) [2]. Đây là tương quan, không chứng minh "trí tuệ". Độ tin cậy:
  medium.
- **Scaling.** Chinchilla (NeurIPS 2022): 70B tham số, 1,4T token, tức khoảng
  20 token mỗi tham số. Con số "20" là tỷ lệ suy ra từ bài báo, không phải luật
  bài báo phát biểu. Epoch AI tái lập (preprint 2024) và tìm thấy lỗi trong
  cách fit thứ 3, nhưng tỷ lệ khoảng 20 vẫn đứng [3]. Các cập nhật quan trọng:
  - Train vượt Chinchilla là có chủ đích, để giảm chi phí suy luận (Sardana et
    al., ICML 2024) [4].
  - Lặp lại dữ liệu đến khoảng 4 epoch gần như không mất gì (Muennighoff et al.,
    NeurIPS 2023) [5].
  - Test-time compute giúp model nhỏ thắng model lớn hơn 14 lần, nhưng **chỉ ở
    câu dễ và trung bình**. Ở câu khó nhất, pretraining vẫn thắng (Snell et al.,
    ICLR 2025 oral) [6].
- **Emergence: tranh luận chưa ngã ngũ.**
  - Wei et al. (TMLR 2022) định nghĩa năng lực emergent [7].
  - Schaeffer et al. (NeurIPS 2023, giải thưởng) cho rằng "đột ngột" là do
    thước đo phi tuyến [8]. Hạng mục giải theo nguồn thứ cấp, độ tin cậy medium.
  - Du et al. (NeurIPS 2024) phản bác: theo pre-training loss, một số năng lực
    chỉ xuất hiện khi loss vượt ngưỡng, kể cả với thước đo liên tục [9].
- **Lý thuyết về sức mạnh của dự đoán.**
  - Malach chứng minh next-token predictor là "universal learner" **khi được
    train trên dữ liệu chain-of-thought**, không phải với pretrain web thuần
    [10]. Venue (ICML 2024?) chưa xác nhận.
  - Bachmann & Nagarajan là phản đề: teacher forcing có thể thất bại ngay ở bài
    toán cần nhìn trước đơn giản [11].

## D2 — Bằng chứng nhìn vào bên trong model

- **Othello-GPT** (Li et al., ICLR 2023): có biểu diễn trạng thái bàn cờ, và
  can thiệp vào nó làm đổi nước đi [12]. **Sửa:** bài gốc thấy probe *phi tuyến*
  tốt hơn. Biểu diễn chỉ *tuyến tính* khi mã hoá ô cờ theo "của tôi / của đối
  thủ" (Nanda, Lee & Wattenberg, BlackboxNLP 2023) [13].
- **Cờ vua** (Karvonen 2024, preprint): đọc được bàn cờ bằng probe tuyến tính
  với độ chính xác đến khoảng 99,6%. Thêm một "skill vector" tăng tỷ lệ thắng
  đến 2,6 lần [14].
- **Phản đề:** model có thể vượt qua probe và kiểm tra tính hợp lệ của token
  tiếp theo, trong khi mô hình thế giới ngầm định vẫn không nhất quán (Vafa et
  al., NeurIPS 2024) [15]. *Đây là bổ sung quan trọng cho tài liệu.*
- **Grokking:** Power et al. (2022) là bài workshop ICLR 2021 kèm preprint,
  không phải main track [16]. Nanda et al. (ICLR 2023) xác nhận thuật toán
  Fourier và đồng nhất thức lượng giác cho phép cộng modulo [17].
- **Không gian và thời gian** (Gurnee & Tegmark, ICLR 2024): xác nhận probe
  tuyến tính trên Llama-2. Tác giả tự nói đây **chưa phải** world model nhân
  quả [18].
- **Induction heads** (Olsson et al. 2022, preprint): bằng chứng nhân quả chỉ có
  ở model nhỏ chỉ dùng attention. Với model lớn thì chỉ là tương quan [19].
- **MLP như bộ nhớ key–value** (Geva et al., EMNLP 2021) [20]. **ROME** (Meng et
  al., NeurIPS 2022) [21]. **Superposition** (Elhage et al. 2022, preprint)
  [22]. **SAE** với khoảng 34 triệu feature trên Claude 3 Sonnet (2024,
  preprint) [23].
- **Claude 3.5 Haiku** (Anthropic, 3/2025) [24]:
  - Suy luận hai bước Dallas → Texas → Austin. Thay "Texas" bằng "California"
    thì ra Sacramento.
  - Chọn trước từ vần "rabbit" rồi mới viết câu dẫn tới nó.
  - Cộng 36+59 bằng các đường song song: một đường ước lượng độ lớn, một đường
    tính chính xác chữ số cuối. Nhưng khi được hỏi, model **mô tả** cách nhớ 1
    kiểu sách giáo khoa, không đúng với cách nó thật sự tính.
  - Giới hạn tự nêu: phương pháp chỉ bắt được một phần tính toán, một model,
    chưa bình duyệt.
- **Introspection 2026** (Lindsey, preprint): model đôi khi phát hiện được khái
  niệm bị tiêm vào activation (khoảng 20% với Opus 4.1). Tác giả gọi khả năng
  này là "highly unreliable" [25].

## D3 — Hậu huấn luyện và suy luận

- **InstructGPT** (NeurIPS 2022): người đánh giá thích model 1.3B hơn GPT-3
  175B [26]. Con số "71,4%" hay được trích chưa kiểm chứng.
- **DeepSeek-R1** đăng trên **Nature 645:633–638 (9/2025)**. RL thuần với phần
  thưởng đúng/sai, không có vết suy luận do người viết, vẫn làm xuất hiện tự
  kiểm tra và phản tư [27].
- **o1** (2024): độ chính xác tăng gần log-tuyến tính theo cả compute RL lúc
  train và compute lúc trả lời [28]. Nguồn thứ cấp, độ tin cậy medium.
- **RLVR có dạy suy luận mới không? Tranh luận mở.**
  - Yue et al. (NeurIPS 2025 oral): RL tăng pass@1, nhưng base model thắng ở
    pass@k với k lớn. Nghĩa là RL chủ yếu chọn lại năng lực sẵn có [29].
  - ProRL (NeurIPS 2025): RL kéo dài đẩy pass@k vượt base model [30].
  - Bản thân thước đo pass@k với k lớn cũng đang bị tranh cãi.
- **Lý thuyết về CoT:**
  - Feng et al. (NeurIPS 2023) [31].
  - Merrill & Sabharwal (ICLR 2024): không có CoT thì bị chặn trên bởi TC0; với
    O(log n) bước lên L; với O(n) bước nhận được mọi ngôn ngữ chính quy; với
    poly(n) bước thì đúng bằng **P** [32].
  - Li et al. (ICLR 2024): với độ chính xác hằng số, không có CoT chỉ được
    **AC0**. TC0 là cận trên khi độ chính xác là log. Có T bước CoT thì giải
    được mạch kích thước T [33].
- **Calibration:**
  - Kadavath et al. (preprint 2022): model "phần lớn biết mình biết gì" [34].
  - Báo cáo GPT-4: base model được calibrate tốt, còn RLHF *làm giảm* calibration
    [35]. *Cần thêm vào tài liệu.*
- **Công cụ:** Toolformer (NeurIPS 2023) [36].
- **Độ trung thực của CoT** (Chen et al. 2025, Anthropic, preprint): model suy
  luận chỉ nhắc đến gợi ý mà nó thật sự dùng trong khoảng 25% (Claude 3.7
  Sonnet) đến 39% (DeepSeek-R1) số trường hợp [37].

## D4 — Giới hạn và tranh luận

- **Hallucination** (Kalai et al. 2025, OpenAI, preprint) [38]:
  - Tỷ lệ hallucination có cận dưới xấp xỉ "singleton rate", tức tỷ lệ sự kiện
    chỉ xuất hiện đúng một lần trong dữ liệu train.
  - Benchmark chấm 0/1 thưởng cho việc đoán hơn là từ chối. Đề xuất: đặt ngưỡng
    tin cậy t, phạt câu sai t/(1−t).
  - Cận dưới chỉ áp dụng cho sự kiện không có quy luật, nên hallucination
    **không phải tất yếu** nói chung.
  - Kalai & Vempala (STOC 2024): model đã calibrate thì buộc phải hallucinate
    với loại sự kiện này [39].
- **Reversal curse** (Berglund et al., ICLR 2024): GPT-4 đúng 79% theo chiều
  xuôi, 33% theo chiều ngược [40]. "Reverse training" (Golovneva et al. 2024)
  giảm hoặc xoá được hiện tượng này, nên **không phải giới hạn cơ bản** [41].
- **Embers of autoregression** (McCoy et al., **PNAS 2024**): GPT-4 giải mã mật
  mã đúng 51% khi đáp án là câu có xác suất cao, chỉ 13% khi xác suất thấp
  [42]. Với o1 thì hiện tượng giảm nhưng không mất (preprint) [43].
- **Faith and Fate** (Dziri et al., NeurIPS 2023 spotlight): xác nhận phát hiện
  "linearized subgraph matching" [44].
- **CoT không trung thực:**
  - Turpin et al. (NeurIPS 2023): độ chính xác giảm đến 36% khi thiên kiến làm
    lệch câu trả lời mà CoT không nhắc đến [45].
  - Lanham et al. (preprint): model càng lớn, CoT thường càng kém trung thực
    [46].
- **Tranh luận "chỉ là ghép mẫu":**
  - GSM-Symbolic: độ chính xác giảm đến 65% khi thêm mệnh đề thừa (GSM-NoOp)
    [47]. Venue ICLR 2025 chưa xác nhận. Một preprint 2026 phản biện phần thống
    kê [48].
  - "Illusion of Thinking" (Apple 2025) bị phản bác bởi một bình luận chưa bình
    duyệt: giới hạn token output, và một số bài River Crossing không có lời giải
    [49].
- **Tokenization:** bằng chứng lẫn lộn. CharBench thấy lỗi đếm chữ phụ thuộc số
  lần ký tự xuất hiện hơn là tokenization [50][51]. Nên viết "một yếu tố góp
  phần", không phải "nguyên nhân".

## Insight xuyên nhánh

1. **"Dự đoán" chưa bao giờ đứng một mình.** Lý thuyết mạnh nhất về sức mạnh
   của next-token prediction (Malach [10], CoT ra P [32]) đều *cần token trung
   gian*. Thực nghiệm mạnh nhất (R1 [27], Snell [6]) đều *thêm RL hoặc
   test-time compute*. Vậy câu trả lời đầy đủ là: dự đoán, cộng ngữ cảnh làm
   giấy nháp, cộng tối ưu hướng tới đáp án đúng.
2. **Cùng một cơ chế giải thích cả đúng lẫn sai.** Độ chính xác bám theo xác
   suất trong dữ liệu (embers [42], singleton rate [38]). Điều đó vừa giải
   thích vì sao sự kiện phổ biến được trả lời đúng, vừa giải thích vì sao sự
   kiện hiếm bị bịa.
3. **Lời model tự giải thích không phải bằng chứng.** Diễn giải cơ chế
   (36+59 [24]) và đo độ trung thực của CoT [37][45] cho cùng kết luận: thứ
   model *nói* về cách nó làm khác với thứ nó *làm*.

## Khuyến nghị cho `docs/why_llms_are_accurate.vi.md`

| # | Thay đổi | Cơ sở | Độ tin cậy |
|---|---|---|---|
| 1 | Thêm điều kiện "không tính kích thước model" cho câu so sánh với PNG/FLAC | [1] | high |
| 2 | Othello: ghi chú cách mã hoá "của tôi / của đối thủ"; thêm phản đề Vafa et al. | [13][15] | high |
| 3 | Grokking: Power et al. là bài workshop | [16] | medium |
| 4 | Induction heads: bằng chứng nhân quả chỉ ở model nhỏ | [19] | medium |
| 5 | Thêm ví dụ 36+59 và chuyện model tự giải thích sai cách nó tính | [24] | medium (một lab) |
| 6 | Emergence: thêm Du et al. 2024, ghi rõ còn tranh luận | [9] | high |
| 7 | Chinchilla: "khoảng 20" là tỷ lệ suy ra; thêm over-training và ghi chú test-time chỉ thắng ở câu dễ/trung bình | [3][4][6] | high |
| 8 | Ghi R1 đã đăng trên Nature; thêm tranh luận RLVR (Yue vs ProRL) | [27][29][30] | high |
| 9 | Sửa đoạn TC0: cần giả định về độ chính xác; poly bước CoT = P | [32][33] | high |
| 10 | Thêm: RLHF làm giảm calibration | [35] | medium |
| 11 | Thêm mục về độ trung thực của CoT (25–39%) | [37][45] | medium |
| 12 | Hallucination: thêm singleton rate và đề xuất phạt t/(1−t); reversal curse giảm được bằng reverse training | [38][41] | medium |
| 13 | Embers: ghi PNAS 2024 và số 51% / 13% | [42] | high |
| 14 | Tokenization: đổi "nguyên nhân" thành "yếu tố góp phần" | [50][51] | medium |

## Câu hỏi còn mở

- Tỷ lệ "điều chỉnh" (tính cả model) của Delétang et al. và kích thước đoạn dữ
  liệu 2048 byte: cần đọc PDF (cần mở quyền truy cập arxiv.org).
- Venue của Malach, Bachmann & Nagarajan, Karvonen và GSM-Symbolic.
- Chưa kiểm venue của Wei 2022 (CoT), Kojima 2022, self-consistency, RAG trong
  lần chạy này.
- Các kết quả của Anthropic chưa được lab độc lập nào tái lập.
- RLVR: chờ thêm kết quả sau ProRL về pass@k.

## Phụ lục nguồn

| [n] | Hỗ trợ cho | Nhà xuất bản / venue | Ngày | Truy cập | Độ tin cậy |
|---|---|---|---|---|---|
| [1] | Nén ảnh/âm thanh vs PNG/FLAC | [ICLR 2024 — Delétang et al.](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3cbf627fa24fb6cb576e04e689b9428b-Abstract-Conference.html) | 2024-05 | 2026-11-08 | high |
| [2] | Nén tương quan tuyến tính với benchmark | [arXiv / COLM 2024 — Huang et al.](https://arxiv.org/abs/2404.09937) | 2024-04 | 2026-11-08 | medium |
| [3] | Tái lập Chinchilla | [Epoch AI — Besiroglu et al.](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt) | 2024-04 | 2026-11-08 | medium |
| [4] | Over-training vì chi phí suy luận | [arXiv / ICML 2024 — Sardana et al.](https://arxiv.org/abs/2401.00448) | 2023-12 | 2026-11-08 | medium |
| [5] | Lặp dữ liệu khoảng 4 epoch | [NeurIPS 2023 — Muennighoff et al.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract.html) | 2023-12 | 2026-11-08 | high |
| [6] | Test-time compute vs model lớn hơn 14 lần | [ICLR 2025 — Snell et al.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html) | 2025-04 | 2026-11-08 | high |
| [7] | Định nghĩa emergence | [TMLR — Wei et al.](https://openreview.net/forum?id=yzkSU5zdwD) | 2022-08 | 2026-11-08 | high |
| [8] | "Mirage" | [The Decoder (thứ cấp) — Schaeffer et al.](https://the-decoder.com/neurips-2023-these-are-the-top-papers-and-award-winners-at-the-largest-ai-conference/) | 2023-12 | 2026-11-08 | medium |
| [9] | Emergence theo loss | [NeurIPS 2024 — Du et al.](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5f1eee2509599faeeb3570a887016a64-Abstract.html) | 2024-12 | 2026-11-08 | high |
| [10] | Universal learner với dữ liệu CoT | [arXiv — Malach](https://arxiv.org/abs/2309.06979) | 2023-09 | 2026-11-08 | medium |
| [11] | Nhược điểm của next-token prediction | [arXiv — Bachmann & Nagarajan](https://arxiv.org/abs/2403.06963) | 2024-03 | 2026-11-08 | medium |
| [12] | Othello-GPT | [arXiv / ICLR 2023 — Li et al.](https://arxiv.org/abs/2210.13382) | 2022-10 | 2026-11-08 | high |
| [13] | Biểu diễn tuyến tính "của tôi / của đối thủ" | [BlackboxNLP 2023 — Nanda, Lee, Wattenberg](https://aclanthology.org/2023.blackboxnlp-1.2) | 2023-12 | 2026-11-08 | high |
| [14] | Cờ vua, probe tuyến tính | [arXiv — Karvonen](https://arxiv.org/abs/2403.15498) | 2024-03 | 2026-11-08 | medium |
| [15] | World model không nhất quán | [arXiv / NeurIPS 2024 — Vafa et al.](https://arxiv.org/abs/2406.03689) | 2024-06 | 2026-11-08 | high |
| [16] | Grokking | [arXiv (workshop ICLR 2021) — Power et al.](https://arxiv.org/abs/2201.02177) | 2022-01 | 2026-11-08 | medium |
| [17] | Thuật toán Fourier cho cộng modulo | [arXiv / ICLR 2023 — Nanda et al.](https://arxiv.org/abs/2301.05217) | 2023-01 | 2026-11-08 | high |
| [18] | Không gian và thời gian | [arXiv / ICLR 2024 — Gurnee & Tegmark](https://arxiv.org/abs/2310.02207) | 2023-10 | 2026-11-08 | high |
| [19] | Induction heads | [arXiv — Olsson et al.](https://arxiv.org/abs/2209.11895) | 2022-09 | 2026-11-08 | medium |
| [20] | MLP là bộ nhớ key–value | [arXiv / EMNLP 2021 — Geva et al.](https://arxiv.org/abs/2012.14913) | 2020-12 | 2026-11-08 | high |
| [21] | ROME | [arXiv / NeurIPS 2022 — Meng et al.](https://arxiv.org/abs/2202.05262) | 2022-02 | 2026-11-08 | high |
| [22] | Superposition | [arXiv — Elhage et al.](https://arxiv.org/abs/2209.10652) | 2022-09 | 2026-11-08 | medium |
| [23] | SAE khoảng 34 triệu feature | [Transformer Circuits — Templeton et al.](https://transformer-circuits.pub/2024/scaling-monosemanticity/) | 2024-05 | 2026-11-08 | medium |
| [24] | Mạch trong Claude 3.5 Haiku | [Anthropic — Tracing the thoughts of a LLM](https://www.anthropic.com/research/tracing-thoughts-language-model) | 2025-03 | 2026-11-08 | medium |
| [25] | Introspection | [arXiv — Lindsey](https://arxiv.org/abs/2601.01828) | 2026-01 | 2026-11-08 | medium |
| [26] | InstructGPT 1.3B > GPT-3 175B | [arXiv / NeurIPS 2022 — Ouyang et al.](https://arxiv.org/abs/2203.02155) | 2022-03 | 2026-11-08 | high |
| [27] | DeepSeek-R1 trên Nature | [Nature — DeepSeek-AI](https://www.nature.com/nature-index/article/10.1038/s41586-025-09422-z) | 2025-09 | 2026-11-08 | high |
| [28] | o1, scaling theo compute | [OpenAI — Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) | 2024-09 | 2026-11-08 | medium |
| [29] | RLVR chủ yếu chọn lại năng lực base | [arXiv / NeurIPS 2025 — Yue et al.](https://arxiv.org/abs/2504.13837) | 2025-04 | 2026-11-08 | high |
| [30] | ProRL vượt base ở pass@k | [arXiv / NeurIPS 2025 — Liu et al.](https://arxiv.org/abs/2505.24864) | 2025-05 | 2026-11-08 | medium |
| [31] | Lý thuyết CoT | [NeurIPS 2023 — Feng et al.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract.html) | 2023-12 | 2026-11-08 | high |
| [32] | CoT: TC0 → L → chính quy → P | [ICLR 2024 — Merrill & Sabharwal](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1f59721c106ea80f613299039112f651-Abstract-Conference.html) | 2024-05 | 2026-11-08 | high |
| [33] | AC0 vs TC0, CoT giải bài toán tuần tự | [ICLR 2024 — Li et al.](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3309b4112c9f04a993f2bbdd0274bba1-Abstract-Conference.html) | 2024-05 | 2026-11-08 | high |
| [34] | Model biết mình biết gì | [arXiv — Kadavath et al.](https://arxiv.org/abs/2207.05221) | 2022-07 | 2026-11-08 | medium |
| [35] | RLHF làm giảm calibration | [arXiv — OpenAI GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) | 2023-03 | 2026-11-08 | medium |
| [36] | Toolformer | [NeurIPS 2023 — Schick et al.](https://proceedings.neurips.cc/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract.html) | 2023-12 | 2026-11-08 | high |
| [37] | Độ trung thực CoT 25–39% | [Anthropic — Chen et al.](https://www.anthropic.com/research/reasoning-models-dont-say-think) | 2025-04 | 2026-11-08 | medium |
| [38] | Singleton rate, phạt t/(1−t) | [arXiv — Kalai et al. (OpenAI)](https://arxiv.org/abs/2509.04664) | 2025-09 | 2026-11-08 | medium |
| [39] | Model đã calibrate phải hallucinate | [arXiv / STOC 2024 — Kalai & Vempala](https://arxiv.org/abs/2311.14648) | 2023-11 | 2026-11-08 | high |
| [40] | Reversal curse 79% / 33% | [arXiv / ICLR 2024 — Berglund et al.](https://arxiv.org/abs/2309.12288) | 2023-09 | 2026-11-08 | high |
| [41] | Reverse training | [arXiv — Golovneva et al.](https://arxiv.org/abs/2403.13799) | 2024-03 | 2026-11-08 | medium |
| [42] | Embers, 51% / 13% | [PNAS — McCoy et al.](https://doi.org/10.1073/pnas.2322420121) | 2024-10 | 2026-11-08 | high |
| [43] | Embers với o1 | [arXiv — McCoy et al.](https://arxiv.org/abs/2410.01792) | 2024-10 | 2026-11-08 | medium |
| [44] | Linearized subgraph matching | [NeurIPS 2023 — Dziri et al.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/deb3c28192f979302c157cb653c15e90-Abstract.html) | 2023-12 | 2026-11-08 | high |
| [45] | CoT không trung thực, giảm đến 36% | [arXiv / NeurIPS 2023 — Turpin et al.](https://arxiv.org/abs/2305.04388) | 2023-05 | 2026-11-08 | high |
| [46] | Model lớn hơn, CoT kém trung thực hơn | [arXiv — Lanham et al.](https://arxiv.org/abs/2307.13702) | 2023-07 | 2026-11-08 | medium |
| [47] | GSM-Symbolic, giảm đến 65% | [arXiv — Mirzadeh et al.](https://arxiv.org/abs/2410.05229) | 2024-10 | 2026-11-08 | medium |
| [48] | Phản biện thống kê GSM-Symbolic | [arXiv — preprint 2026](https://arxiv.org/abs/2605.28700) | 2026-05 | 2026-11-08 | low |
| [49] | Phản bác "Illusion of Thinking" | [arXiv — Lawsen](https://arxiv.org/abs/2506.09250) | 2025-06 | 2026-11-08 | low |
| [50] | Vì sao LLM đếm chữ kém | [arXiv — Fu et al.](https://arxiv.org/abs/2412.18626) | 2024-12 | 2026-11-08 | medium |
| [51] | CharBench | [arXiv — CharBench](https://arxiv.org/abs/2508.02591) | 2025-08 | 2026-11-08 | medium |

## Bản đồ độ cũ

Khung thời gian theo pack: kết quả SOTA về ML và tranh luận đang diễn ra cần
kiểm lại sau 6 tháng; lý thuyết sau 24 tháng; công trình nền tảng không có hạn.
Kết quả tính bằng script nằm trong `staleness.json`.

| Claim | Ngày | Kiểm lại | Đã cũ? |
|---|---|---|---|
| Claude 3.5 Haiku circuits [24] | 2025-03 | 2025-09 | có |
| RLVR chỉ chọn lại năng lực [29] | 2025-04 | 2025-10 | có |
| Test-time compute vs model 14 lần [6] | 2025-04 | 2025-10 | có |
| Độ trung thực CoT 25–39% [37] | 2025-04 | 2025-10 | có |
| DeepSeek-R1 Nature [27] | 2025-09 | 2026-03 | có |
| Phản biện GSM-Symbolic [48] | 2026-05 | 2026-11 | sắp |
| Cận dưới hallucination [38] | 2025-09 | 2027-09 | chưa |

**Mốc sớm nhất (2025-09) đã qua:** theo chuẩn 6 tháng của ML, phần lớn claim
SOTA đã "cũ" ngay khi chạy. Nên chạy **Refresh** tập trung vào D2 và D3 (diễn
giải cơ chế 2026, RLVR sau ProRL), lý tưởng là sau khi mở quyền truy cập
arxiv.org.
