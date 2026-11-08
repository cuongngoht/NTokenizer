---
status: draft
stepsCompleted: [1, 2, 3, 4]
inputDocuments:
  - _bmad-output/planning-artifacts/research/technical-build-vietnamese-llm-from-wikipedia-data-2026-11-08/research.md
  - _bmad-output/brainstorming/brainstorm-how-llms-predict-next-token-2026-11-08/.memlog.md
  - README.md
  - docs/DEVELOPMENT.md
  - docs/FINETUNING.md
  - docs/why_llms_are_accurate.vi.md
---

# NTokenizer - Epic Breakdown

## Overview

Tài liệu này phân rã yêu cầu thành epic và story cho NTokenizer: một GPT tiếng
Việt xây từ đầu bằng PyTorch thuần (hiện train trên Wikipedia tiếng Việt).

**Không có PRD và tài liệu kiến trúc.** Yêu cầu được rút từ ba nguồn: báo cáo
nghiên cứu kỹ thuật (khuyến nghị R1–R8), buổi brainstorm 115 ý tưởng, và mã
nguồn cùng tài liệu hiện có. Nếu sau này có PRD thì nên đối chiếu lại.

**Người dùng chính:** chủ dự án (nhà nghiên cứu/học viên) muốn (a) hiểu cơ chế
dự đoán token trên chính model của mình, và (b) nâng model từ "chỉ biết
Wikipedia" thành một LM tiếng Việt hữu ích hơn.

**Giả định phần cứng:** chưa biết. Mọi yêu cầu về thời gian train được đo trên
máy thật (Epic 4) thay vì hứa trước.

## Requirements Inventory

### Functional Requirements

**Đánh giá**
- FR1: Có script tính bits-per-byte (BPB) trên tập validation cố định, để so sánh được giữa các tokenizer khác nhau.
- FR2: Có script đánh giá cloze/đoán từ tiếp theo bằng tiếng Việt (kiểu `lambada_vi`) và báo độ chính xác.
- FR3: Có công cụ kiểm tra trùng n-gram (8-gram và 13-gram) giữa dữ liệu train và bộ đánh giá.
- FR4: Mỗi checkpoint có một bảng kết quả đánh giá (BPB, cloze) được lưu lại để so sánh theo thời gian.

**Dữ liệu**
- FR5: Pipeline chuẩn hoá Unicode về NFC và thống nhất vị trí dấu thanh ("hoà"/"hòa").
- FR6: Pipeline loại trùng bằng MinHash và lọc các bài sơ khai/quá ngắn.
- FR7: Nạp được một tập con của FineWeb-2 `vie_Latn` (theo số token cấu hình) mà không cần tải toàn bộ.
- FR8: Cấu hình tỷ lệ trộn giữa các nguồn (Wikipedia, FineWeb-2) và ghi lại nguồn gốc/giấy phép của từng nguồn trong `meta.json`.

**Tokenizer**
- FR9: Có công cụ đo fertility (token trên từ) và tỷ lệ nén trên văn bản tiếng Việt cho một tokenizer bất kỳ.
- FR10: Train và so sánh được các tokenizer với kích thước vocab khác nhau (16k/20k/32k) trên hỗn hợp dữ liệu mới.

**Train**
- FR11: Tuỳ chọn gộp chung embedding vào và ra (weight tying) trong `GPTConfig`.
- FR12: Có cách đặt kích thước model theo một tham số độ sâu (preset sâu-mỏng), thay vì chỉnh tay nhiều hệ số.
- FR13: Có công cụ lập kế hoạch ngân sách token: từ số tham số và số epoch tối đa, tính số token cần và cảnh báo khi lặp quá 4 epoch.
- FR14: Có script đo throughput (token/giây) theo thiết bị và ghi vào tài liệu.
- FR15: (Thử nghiệm) Tuỳ chọn QK-Norm và optimizer Muon, bật tắt được bằng cờ.

**Hội thoại (SFT)**
- FR16: Chuyển được dữ liệu instruction tiếng Việt (vi-alpaca, Aya) sang định dạng train có chat template.
- FR17: SFT hỗ trợ tuỳ chọn che hoặc không che prompt khi tính loss.
- FR18: Có CLI hội thoại tương tác với model đã SFT.

**Kính hiển vi dự đoán**
- FR19: `sample.py` hiển thị top-5 xác suất ở mỗi bước sinh.
- FR20: Có công cụ logit lens: chiếu activation từng tầng qua `ln_f` và `lm_head`, vẽ hạng của token đúng theo tầng.
- FR21: Có công cụ tô màu entropy từng token trên một đoạn văn.
- FR22: Có biểu đồ độ tin cậy (reliability diagram) so sánh xác suất dự đoán với tỷ lệ đúng thực tế.

**Phòng thí nghiệm cơ chế**
- FR23: Tạo được dataset grokking `a + b = c (mod 97)` dạng văn bản và cấu hình model 2 tầng để train, ghi train/val accuracy theo bước.
- FR24: Tạo được dataset "thế giới giả" với sự kiện lặp 1, 2, 5, 10, 50 lần và đo độ chính xác nhớ lại theo tần suất.
- FR25: Có bài kiểm tra reversal curse bằng tiếng Việt (câu xuôi và câu ngược).
- FR26: Có công cụ phát hiện induction head và quét ablation từng head.

**Tài liệu**
- FR27: Có model card liệt kê nguồn dữ liệu, giấy phép (CC BY-SA, ODC-By) và giới hạn.
- FR28: Có bài giải thích tiếng Anh `docs/how_llms_predict.md` với một thí nghiệm chạy được cho mỗi chương.

### NonFunctional Requirements

- NFR1: Mọi công cụ mới nằm trong `src/ntokenizer/` và chỉ được gọi qua script mỏng trong `scripts/`.
- NFR2: Mọi thay đổi trong `src/` có test pytest; bộ test hiện có vẫn phải qua.
- NFR3: Kết quả đo phải tái lập được: cố định seed và ghi cấu hình/commit vào file kết quả.
- NFR4: Pipeline dữ liệu không được yêu cầu nạp toàn bộ corpus vào RAM (chạy trên máy 8–16 GB).
- NFR5: Tokenizer dùng khi train và sample phải khớp checkpoint; mọi thay đổi tokenizer phải kèm cảnh báo checkpoint cũ không còn dùng được.
- NFR6: Mỗi nguồn dữ liệu phải có bản ghi nguồn, giấy phép và ngày tải.
- NFR7: Công cụ chạy được trên CPU, CUDA và MPS; tính năng nặng phải có chế độ nhỏ để thử.
- NFR8: Câu chữ trong tài liệu và kết quả mô tả "biểu diễn/tính toán ở quy mô nhỏ", không khẳng định model "hiểu".

### Additional Requirements

- Không có starter template; repo đã có sẵn khung (`pyproject.toml`, `src/ntokenizer`, `scripts`, `tests`).
- Mạng của môi trường cloud chặn `huggingface.co` và `arxiv.org`. Story nạp dữ liệu phải chạy được ở máy người dùng và có cách chạy thử với dữ liệu mẫu nhỏ.
- Mọi probe phải đi cùng một can thiệp (patching hoặc ablation) trước khi rút ra kết luận (quy tắc "Critic" của brainstorm).
- Các điều kiện pháp lý còn mở: CC BY-SA có áp dụng cho trọng số hay không chưa có câu trả lời, nên model card phải nêu rõ điều này.
- Không dùng VMLU làm thước đo chính cho model dưới 200M tham số (PhoGPT-7B5-Instruct chỉ đạt 24,01, ngang đoán ngẫu nhiên).
- Số liệu throughput trên GPU tiêu dùng và Mac chưa có nguồn đáng tin; phải đo trực tiếp.

### UX Design Requirements

Không áp dụng: dự án không có giao diện. Các công cụ là CLI và tệp ảnh/HTML tĩnh.

### FR Coverage Map

FR1: Epic 1 - Bits-per-byte trên tập validation cố định
FR2: Epic 1 - Đánh giá cloze tiếng Việt
FR3: Epic 1 - Kiểm tra trùng n-gram
FR4: Epic 1 - Bảng kết quả theo checkpoint
FR5: Epic 2 - Chuẩn hoá NFC và vị trí dấu
FR6: Epic 2 - Loại trùng MinHash và lọc bài sơ khai
FR7: Epic 2 - Nạp tập con FineWeb-2
FR8: Epic 2 - Trộn nguồn và ghi nguồn gốc
FR9: Epic 3 - Đo fertility và tỷ lệ nén
FR10: Epic 3 - So sánh kích thước vocab
FR11: Epic 4 - Weight tying
FR12: Epic 4 - Preset theo độ sâu
FR13: Epic 4 - Lập kế hoạch ngân sách token (Story 4.4)
FR14: Epic 4 - Đo throughput (Story 4.3)
FR15: Epic 4 - QK-Norm và Muon (thử nghiệm)
FR16: Epic 5 - Chuẩn bị dữ liệu SFT
FR17: Epic 5 - Che hoặc không che prompt
FR18: Epic 5 - CLI hội thoại
FR19: Epic 6 - Top-5 xác suất
FR20: Epic 6 - Logit lens
FR21: Epic 6 - Heatmap entropy
FR22: Epic 6 - Reliability diagram
FR23: Epic 7 - Grokking mod 97
FR24: Epic 7 - Thế giới giả theo tần suất
FR25: Epic 7 - Reversal curse tiếng Việt
FR26: Epic 7 - Induction head và ablation
FR27: Epic 8 - Model card và giấy phép
FR28: Epic 8 - Bài giải thích tiếng Anh

## Epic List

### Epic 1: Biết model tốt lên hay tệ đi (bộ đánh giá)
Chủ dự án chạy một lệnh để thấy BPB và độ chính xác cloze của bất kỳ checkpoint nào, và biết dữ liệu đánh giá có bị lẫn vào dữ liệu train hay không.
**FRs covered:** FR1, FR2, FR3, FR4

### Epic 2: Dữ liệu tiếng Việt sạch và đủ lớn
Chủ dự án có một bộ dữ liệu train gồm Wikipedia và một phần FineWeb-2, đã chuẩn hoá, loại trùng, có ghi nguồn gốc và giấy phép.
**FRs covered:** FR5, FR6, FR7, FR8

### Epic 3: Tokenizer tối ưu cho tiếng Việt
Chủ dự án chọn được kích thước vocab dựa trên số đo (fertility, nén, BPB) thay vì đoán.
**FRs covered:** FR9, FR10

### Epic 4: Công thức train nhỏ gọn và có số đo thật
Chủ dự án lập được kế hoạch train (kích thước, token, thời gian) phù hợp với máy mình, và dùng được các cải tiến kiến trúc đã được kiểm chứng.
**FRs covered:** FR11, FR12, FR13, FR14, FR15

### Epic 5: Model biết trò chuyện bằng tiếng Việt (SFT)
Chủ dự án tinh chỉnh checkpoint đã pretrain thành model trả lời câu hỏi, và so sánh được các lựa chọn SFT bằng bộ đánh giá.
**FRs covered:** FR16, FR17, FR18

### Epic 6: Kính hiển vi dự đoán
Chủ dự án nhìn thấy model đang "nghĩ" gì ở mỗi token: xác suất, tầng nào quyết định, chỗ nào không chắc, và xác suất có đáng tin không.
**FRs covered:** FR19, FR20, FR21, FR22

### Epic 7: Phòng thí nghiệm kiểm chứng cơ chế
Chủ dự án tự tái hiện ở quy mô nhỏ các hiện tượng trong tài liệu nghiên cứu: grokking, nhớ sự kiện theo tần suất, reversal curse, induction head.
**FRs covered:** FR23, FR24, FR25, FR26

### Epic 8: Công bố và giải thích
Người ngoài đọc được model card rõ ràng về dữ liệu và giấy phép, và một bài giải thích tiếng Anh gắn với thí nghiệm chạy được.
**FRs covered:** FR27, FR28

## Epic 1: Biết model tốt lên hay tệ đi (bộ đánh giá)

Chủ dự án chạy một lệnh để thấy BPB và độ chính xác cloze của bất kỳ checkpoint nào, và biết dữ liệu đánh giá có bị lẫn vào dữ liệu train hay không.

### Story 1.1: Tính bits-per-byte trên tập validation

As a chủ dự án,
I want chạy một lệnh tính bits-per-byte của checkpoint trên một tập validation cố định,
So that tôi so sánh được các model dùng tokenizer khác nhau một cách công bằng.

**Acceptance Criteria:**

**Given** một checkpoint và `val.bin` cùng `meta.json` hợp lệ
**When** tôi chạy `scripts/eval_bpb.py --ckpt <path>`
**Then** script in loss (nat/token), số byte UTF-8 của văn bản tương ứng (đếm bằng cách giải mã token của `val.bin`) và BPB
**And** BPB = loss × (số token / số byte) / ln 2, có test với dữ liệu giả cho kết quả đã biết.

**Given** tokenizer của `meta.json` không khớp checkpoint
**When** tôi chạy script
**Then** script dừng với thông báo lỗi rõ ràng thay vì in số sai.

### Story 1.2: Đánh giá cloze tiếng Việt

As a chủ dự án,
I want đo độ chính xác khi model đoán từ cuối của một câu,
So that tôi có thước đo "dùng được" cho model nhỏ mà VMLU không đo nổi.

**Acceptance Criteria:**

**Given** một tệp JSONL các mục `{context, target}` do tôi cung cấp (tối thiểu 3 mục mẫu đi kèm repo để thử)
**When** tôi chạy `scripts/eval_cloze.py --ckpt <path> --data <file>`
**Then** script báo tỷ lệ đúng khi greedy decode tái tạo `target`, và in từng mục sai nếu có `--verbose`
**And** kết quả chạy lại với cùng seed là giống hệt.

**Given** một mục có `target` gồm nhiều token
**When** đánh giá
**Then** mục chỉ tính đúng khi toàn bộ token của `target` khớp.

### Story 1.3: Kiểm tra trùng n-gram giữa train và bộ đánh giá

As a chủ dự án,
I want biết bao nhiêu mục đánh giá có n-gram xuất hiện trong dữ liệu train,
So that tôi không báo cáo điểm bị thổi phồng do lẫn dữ liệu.

**Acceptance Criteria:**

**Given** một corpus train và một tệp đánh giá
**When** tôi chạy `scripts/check_contamination.py --train <corpus> --eval <file> --n 13`
**Then** script in số và tỷ lệ mục đánh giá có ít nhất một 13-gram trùng, và lưu danh sách mục bị trùng ra tệp
**And** cờ `--n 8` cho kết quả với 8-gram.

**Given** corpus lớn hơn RAM
**When** chạy
**Then** script xử lý theo luồng (không nạp cả corpus vào bộ nhớ) và có test với corpus giả nhỏ.

### Story 1.4: Lưu bảng kết quả theo checkpoint

As a chủ dự án,
I want mỗi lần đánh giá ghi một dòng vào bảng kết quả,
So that tôi xem được model tiến bộ ra sao qua các checkpoint.

**Acceptance Criteria:**

**Given** tôi chạy Story 1.1 hoặc 1.2
**When** thêm cờ `--log-results`
**Then** một dòng được thêm vào `artifacts/eval/results.csv` gồm thời gian, đường dẫn checkpoint, bước train, metric, giá trị và commit git
**And** chạy lặp không ghi đè các dòng cũ.

**Given** `results.csv` có nhiều dòng
**When** tôi chạy `scripts/eval_report.py`
**Then** script in bảng so sánh các checkpoint theo từng metric.

## Epic 2: Dữ liệu tiếng Việt sạch và đủ lớn

Chủ dự án có một bộ dữ liệu train gồm Wikipedia và một phần FineWeb-2, đã chuẩn hoá, loại trùng, có ghi nguồn gốc và giấy phép.

### Story 2.1: Chuẩn hoá Unicode và vị trí dấu thanh

As a chủ dự án,
I want mọi văn bản đưa vào corpus được chuẩn hoá NFC và thống nhất vị trí dấu,
So that cùng một từ không bị tách thành nhiều chuỗi token khác nhau.

**Acceptance Criteria:**

**Given** một chuỗi chứa "hoà" (kiểu cũ) và "hòa" (kiểu mới) ở dạng NFD hoặc NFC
**When** gọi hàm `normalize_vi(text)` trong `src/ntokenizer/corpus.py`
**Then** cả hai về cùng một dạng NFC duy nhất theo quy ước được ghi trong docstring
**And** có test cho ít nhất 6 cặp từ (hoà/hòa, thuỷ/thủy, khoẻ/khỏe...).

**Given** pipeline `build_corpus` hiện có
**When** tôi chạy với cờ `--normalize`
**Then** corpus đầu ra đã qua `normalize_vi` và bộ test `test_corpus.py` hiện có vẫn qua.

### Story 2.2: Loại trùng MinHash và lọc bài sơ khai

As a chủ dự án,
I want loại các tài liệu gần trùng nhau và các bài quá ngắn,
So that model không học lặp lại và không bị kéo bởi bài sơ khai do bot tạo.

**Acceptance Criteria:**

**Given** một corpus có tài liệu trùng và gần trùng
**When** tôi chạy `scripts/dedup_corpus.py --threshold 0.8 --min-chars <N>`
**Then** đầu ra giữ một bản đại diện cho mỗi nhóm gần trùng và bỏ bài ngắn hơn `N` ký tự
**And** script in số tài liệu vào/ra và số bị loại theo từng lý do.

**Given** corpus lớn
**When** chạy
**Then** bộ nhớ dùng tăng theo số chữ ký MinHash chứ không theo kích thước văn bản, và có test với dữ liệu giả cho trùng chính xác và gần trùng.

### Story 2.3: Nạp tập con FineWeb-2 tiếng Việt

As a chủ dự án,
I want tải về một lượng token cấu hình được từ FineWeb-2 `vie_Latn`,
So that tôi có thêm dữ liệu web mà không phải tải hàng trăm GB.

**Acceptance Criteria:**

**Given** máy có truy cập `huggingface.co`
**When** tôi chạy `scripts/fetch_fineweb2.py --max-tokens 1e9 --out data/raw/fineweb2_vi`
**Then** script đọc theo luồng, dừng khi đạt ngưỡng và ghi tài liệu ra JSONL
**And** ghi tệp `SOURCE.json` gồm tên bộ dữ liệu, phiên bản, giấy phép ODC-By, ngày tải, số tài liệu.

**Given** không có mạng hoặc bị chặn
**When** chạy với `--sample-fixture`
**Then** script dùng một tập mẫu nhỏ đi kèm repo để kiểm tra phần còn lại của pipeline.

### Story 2.4: Trộn nguồn dữ liệu và ghi nguồn gốc

As a chủ dự án,
I want khai báo tỷ lệ trộn giữa Wikipedia và FineWeb-2 và có bản ghi nguồn gốc trong `meta.json`,
So that tôi biết model đã học từ đâu và đáp ứng nghĩa vụ giấy phép.

**Acceptance Criteria:**

**Given** hai corpus đã chuẩn hoá và loại trùng
**When** tôi chạy `scripts/prepare_dataset.py --mix wiki=0.3,fineweb2=0.7`
**Then** `train.bin` và `val.bin` được tạo theo tỷ lệ gần đúng (sai số ≤ 2% theo số token)
**And** `meta.json` có mục `sources` liệt kê từng nguồn: tên, giấy phép, số token, ngày tải.

**Given** `val.bin`
**When** tạo
**Then** tập validation lấy cùng một tập tài liệu cố định (theo hash tài liệu) cho mọi lần trộn để BPB so sánh được.

## Epic 3: Tokenizer tối ưu cho tiếng Việt

Chủ dự án chọn được kích thước vocab dựa trên số đo (fertility, nén, BPB) thay vì đoán.

### Story 3.1: Công cụ đo fertility và tỷ lệ nén

As a chủ dự án,
I want đo số token trên mỗi từ và số byte trên mỗi token của một tokenizer,
So that tôi so sánh tokenizer bằng số liệu.

**Acceptance Criteria:**

**Given** một tokenizer SentencePiece và một tệp văn bản tiếng Việt
**When** tôi chạy `scripts/measure_tokenizer.py --model <spm> --text <file>`
**Then** script in fertility (token/từ, tách từ theo khoảng trắng), byte/token, và tỷ lệ token chỉ gồm một âm tiết
**And** chạy được với nhiều `--model` cùng lúc để in bảng so sánh.

**Given** văn bản có ký tự lạ hoặc rỗng
**When** đo
**Then** script không lỗi và báo số dòng bị bỏ qua.

### Story 3.2: Train và so sánh các kích thước vocab

As a chủ dự án,
I want train tokenizer 16k, 20k và 32k trên cùng một hỗn hợp dữ liệu và so sánh,
So that tôi chọn kích thước vocab có căn cứ.

**Acceptance Criteria:**

**Given** corpus hỗn hợp từ Epic 2 (hoặc corpus Wikipedia hiện có nếu Epic 2 chưa xong)
**When** tôi chạy `scripts/train_tokenizer_spm.py` với từng `--vocab-size`
**Then** mỗi tokenizer được lưu với tên chứa kích thước vocab
**And** `measure_tokenizer.py` in bảng so sánh ba tokenizer.

**Given** bảng so sánh
**When** tôi quyết định kích thước
**Then** quyết định và số liệu được ghi vào `docs/tokenizer_choice.md`, kèm cảnh báo rằng checkpoint cũ dùng tokenizer khác sẽ không tương thích (NFR5).

### Story 3.3: So sánh model theo BPB giữa các tokenizer

As a chủ dự án,
I want train nhanh một model nhỏ với mỗi tokenizer và so BPB,
So that kích thước vocab được chọn bằng chất lượng thật chứ không chỉ fertility.

**Acceptance Criteria:**

**Given** ba tokenizer từ Story 3.2 và cấu hình model nhỏ cố định
**When** tôi chạy `scripts/compare_tokenizers.sh` (train ngắn, cùng số byte dữ liệu)
**Then** mỗi model được đánh giá bằng `eval_bpb.py` và kết quả ghi vào `results.csv`
**And** bảng cuối cùng xếp hạng các tokenizer theo BPB.

**Given** chạy trên CPU
**When** dùng cờ `--tiny`
**Then** toàn bộ so sánh hoàn tất với dữ liệu rất nhỏ để kiểm tra pipeline.

## Epic 4: Công thức train nhỏ gọn và có số đo thật

Chủ dự án lập được kế hoạch train (kích thước, token, thời gian) phù hợp với máy mình, và dùng được các cải tiến kiến trúc đã được kiểm chứng.

### Story 4.1: Tuỳ chọn weight tying

As a chủ dự án,
I want bật gộp chung embedding vào và ra,
So that model nhỏ tiết kiệm tham số để dành cho độ sâu.

**Acceptance Criteria:**

**Given** `GPTConfig(tie_embeddings=True)`
**When** tạo model
**Then** `lm_head.weight` và `wte.weight` là cùng một tensor và `count_parameters()` giảm đúng bằng `vocab_size × n_embd`
**And** mặc định `tie_embeddings=False` để checkpoint cũ vẫn nạp được.

**Given** model có tying
**When** lưu và nạp lại checkpoint
**Then** logits trước và sau khi nạp giống nhau, có test trong `tests/test_model.py`.

### Story 4.2: Preset theo độ sâu

As a chủ dự án,
I want đặt kích thước model bằng một tham số độ sâu,
So that tôi thử các kích thước mà không phải chỉnh tay nhiều hệ số.

**Acceptance Criteria:**

**Given** `GPTConfig.from_depth(depth=12, vocab_size=V, block_size=B)`
**When** gọi
**Then** hàm trả về cấu hình có `n_layer = depth` và chiều ẩn, số head, số KV head suy ra theo quy tắc được ghi trong docstring (ưu tiên sâu-mỏng)
**And** số head chia hết chiều ẩn và số KV head chia hết số head.

**Given** các độ sâu 6, 12, 24
**When** in số tham số
**Then** số tham số tăng đơn điệu và có test kiểm tra tính hợp lệ của cấu hình.

### Story 4.3: Đo throughput theo thiết bị

As a chủ dự án,
I want đo tokens/giây của model trên máy tôi,
So that kế hoạch train dựa trên số đo thật.

**Acceptance Criteria:**

**Given** một `GPTConfig` và thiết bị tự nhận (CUDA, MPS hoặc CPU)
**When** tôi chạy `scripts/bench_throughput.py --steps 20 --warmup 5`
**Then** script in tokens/giây, bộ nhớ đỉnh (nếu có), thiết bị, dtype và phiên bản torch
**And** ghi một dòng vào `docs/throughput.md` khi có cờ `--record`.

**Given** thiết bị không hỗ trợ bf16
**When** chạy
**Then** script chuyển sang fp32 và ghi rõ điều đó trong kết quả.

### Story 4.4: Công cụ lập kế hoạch ngân sách token

As a chủ dự án,
I want nhập số tham số và lượng dữ liệu đang có rồi biết cần bao nhiêu token và bao nhiêu epoch,
So that tôi không train thiếu hoặc lặp dữ liệu quá mức.

**Acceptance Criteria:**

**Given** `scripts/plan_training.py --params 125e6 --tokens-available 1.1e9 --tokens-per-param 20`
**When** chạy
**Then** script in token mục tiêu (2,5 tỷ), số epoch cần (≈2,3) và FLOPs xấp xỉ theo 6·N·D
**And** in cảnh báo khi số epoch vượt 4 và gợi ý thêm dữ liệu.

**Given** có kết quả đo throughput (Story 4.3)
**When** truyền `--throughput <tokens/s>`
**Then** script in thêm thời gian train ước tính.

### Story 4.5: QK-Norm và Muon (thử nghiệm)

As a chủ dự án,
I want bật thử QK-Norm và optimizer Muon bằng cờ,
So that tôi kiểm tra xem các mẹo của modded-nanogpt có giúp ở quy mô của tôi không.

**Acceptance Criteria:**

**Given** `GPTConfig(qk_norm=True)`
**When** forward
**Then** Q và K được chuẩn hoá trước khi tính attention và mặc định `False` không đổi kết quả hiện có (test hồi quy).

**Given** `scripts/train.py --optimizer muon`
**When** train 50 bước trên dữ liệu nhỏ
**Then** loss giảm và giữ AdamW cho embedding và tham số 1 chiều
**And** tài liệu nêu rõ đây là tính năng thử nghiệm, có so sánh BPB với AdamW trong `results.csv`.

## Epic 5: Model biết trò chuyện bằng tiếng Việt (SFT)

Chủ dự án tinh chỉnh checkpoint đã pretrain thành model trả lời câu hỏi, và so sánh được các lựa chọn SFT bằng bộ đánh giá.

### Story 5.1: Chuẩn bị dữ liệu SFT có chat template

As a chủ dự án,
I want chuyển bộ instruction tiếng Việt sang định dạng train với chat template cố định,
So that tôi tinh chỉnh model bằng pipeline fine-tune hiện có.

**Acceptance Criteria:**

**Given** một tệp JSONL kiểu vi-alpaca (`instruction`, `input`, `output`) hoặc Aya
**When** tôi chạy `scripts/prepare_sft.py --in <file> --out data/processed/sft`
**Then** mỗi mẫu được đóng khung bằng token đặc biệt người dùng/trợ lý do template định nghĩa, và tệp `.bin` kèm mặt nạ vị trí câu trả lời được tạo
**And** `SOURCE.json` ghi tên bộ dữ liệu, giấy phép, cảnh báo "dịch máy hoặc do model sinh" khi áp dụng.

**Given** mẫu dài hơn `block_size`
**When** chuẩn bị
**Then** mẫu bị bỏ và số mẫu bỏ được in ra.

### Story 5.2: SFT với tuỳ chọn che prompt

As a chủ dự án,
I want chọn tính loss chỉ trên câu trả lời hoặc trên cả prompt,
So that tôi thử lựa chọn nào tốt hơn cho dữ liệu nhỏ.

**Acceptance Criteria:**

**Given** dữ liệu từ Story 5.1 và `--init_from <ckpt>`
**When** tôi chạy `scripts/train.py --sft --mask-prompt` hoặc `--no-mask-prompt`
**Then** với `--mask-prompt` loss bằng 0 ở token prompt (kiểm bằng test với batch nhỏ) và với `--no-mask-prompt` loss tính trên mọi token
**And** cờ mặc định là `--mask-prompt`.

**Given** `tests/test_finetune.py` hiện có
**When** chạy test
**Then** các test cũ vẫn qua.

### Story 5.3: So sánh che và không che prompt

As a chủ dự án,
I want so sánh hai cách SFT trên cùng dữ liệu bằng bộ đánh giá,
So that tôi chọn cách làm bằng số liệu.

**Acceptance Criteria:**

**Given** hai checkpoint SFT (có và không che prompt) cùng seed và cùng số bước
**When** tôi chạy `eval_bpb.py` trên một tập hội thoại giữ lại và `eval_cloze.py`
**Then** kết quả hai model nằm cạnh nhau trong `results.csv`
**And** kết luận ngắn được ghi vào `docs/sft_notes.md` kể cả khi chênh lệch không đáng kể.

### Story 5.4: CLI hội thoại tương tác

As a chủ dự án,
I want trò chuyện với model trong terminal,
So that tôi thử câu trả lời thật thay vì chỉ nhìn số.

**Acceptance Criteria:**

**Given** một checkpoint SFT
**When** tôi chạy `scripts/chat.py --ckpt <path>`
**Then** chương trình nhận từng lượt, áp chat template, sinh câu trả lời bằng `generate()` và dừng tại token kết thúc lượt
**And** lệnh `/reset` xoá ngữ cảnh, `/quit` thoát.

**Given** ngữ cảnh vượt `block_size`
**When** trò chuyện tiếp
**Then** các lượt cũ nhất bị cắt đi mà không báo lỗi.

## Epic 6: Kính hiển vi dự đoán

Chủ dự án nhìn thấy model đang "nghĩ" gì ở mỗi token: xác suất, tầng nào quyết định, chỗ nào không chắc, và xác suất có đáng tin không.

### Story 6.1: Hiển thị top-5 xác suất khi sinh

As a chủ dự án,
I want thấy 5 token có xác suất cao nhất ở mỗi bước sinh,
So that tôi thấy phân phối nhọn hay phẳng và hiểu vì sao model chọn token đó.

**Acceptance Criteria:**

**Given** một checkpoint và prompt
**When** tôi chạy `scripts/sample.py --show-probs`
**Then** mỗi bước in token được chọn và 5 token đầu bảng cùng xác suất sau khi áp dụng temperature, top-k, top-p
**And** không có cờ này, đầu ra giữ nguyên như hiện nay.

**Given** repetition penalty bật
**When** hiển thị
**Then** xác suất in ra là xác suất sau penalty, và dòng đầu ghi rõ điều đó.

### Story 6.2: Logit lens theo tầng

As a chủ dự án,
I want thấy ở tầng nào token đúng bắt đầu thắng,
So that tôi hiểu câu trả lời hình thành ở đâu trong mạng.

**Acceptance Criteria:**

**Given** một prompt và token mục tiêu
**When** tôi chạy `scripts/logit_lens.py --ckpt <path> --prompt "..." --target "..."`
**Then** script chiếu residual sau mỗi tầng qua `ln_f` và `lm_head`, in hạng và xác suất của token mục tiêu theo tầng, và lưu biểu đồ PNG
**And** tầng cuối cho đúng logits của `forward()` thông thường (kiểm bằng test).

### Story 6.3: Tô màu entropy trên đoạn văn

As a chủ dự án,
I want thấy token nào model không chắc trong một đoạn văn,
So that tôi liên hệ độ không chắc với sự kiện hiếm và lỗi.

**Acceptance Criteria:**

**Given** một đoạn văn
**When** tôi chạy `scripts/entropy_view.py --ckpt <path> --file <txt>`
**Then** script xuất một tệp HTML tự chứa, mỗi token tô màu theo entropy, tooltip hiện entropy và xác suất token thật
**And** HTML dùng được ở chế độ sáng và tối, không tải tài nguyên ngoài.

### Story 6.4: Biểu đồ độ tin cậy

As a chủ dự án,
I want biết xác suất model gán có khớp tỷ lệ đúng thực tế không,
So that tôi đánh giá được model có "biết mình biết gì" và thấy tác động của SFT lên calibration.

**Acceptance Criteria:**

**Given** `val.bin` và checkpoint
**When** tôi chạy `scripts/calibration.py --ckpt <path>`
**Then** script chia xác suất của token dự đoán top-1 thành 10 khoảng, tính độ chính xác trong từng khoảng, in ECE và lưu biểu đồ PNG
**And** truyền hai checkpoint bất kỳ (ví dụ trước và sau SFT) thì vẽ hai đường trên cùng biểu đồ.

## Epic 7: Phòng thí nghiệm kiểm chứng cơ chế

Chủ dự án tự tái hiện ở quy mô nhỏ các hiện tượng trong tài liệu nghiên cứu: grokking, nhớ sự kiện theo tần suất, reversal curse, induction head.

### Story 7.1: Phòng thí nghiệm grokking mod 97

As a chủ dự án,
I want train một model 2 tầng trên phép cộng modulo 97 và xem accuracy train/val theo bước,
So that tôi thấy tận mắt model chuyển từ học thuộc sang thuật toán.

**Acceptance Criteria:**

**Given** `scripts/grokking_lab.py --mod 97 --train-frac 0.3 --weight-decay 1.0`
**When** chạy
**Then** script tạo dataset dạng văn bản `a + b = c`, train model nhỏ và ghi train/val accuracy theo bước ra CSV và PNG
**And** chạy được trên CPU trong vài phút với cờ `--tiny`.

**Given** kết quả
**When** val accuracy chưa vượt 90% sau số bước tối đa
**Then** script vẫn ghi đầy đủ kết quả và in rõ "chưa quan sát được grokking" thay vì kết luận sai.

### Story 7.2: Nhớ sự kiện theo tần suất xuất hiện

As a chủ dự án,
I want đo độ chính xác nhớ lại các sự kiện giả xuất hiện 1, 2, 5, 10, 50 lần,
So that tôi kiểm chứng ở quy mô nhỏ ý "sự kiện hiếm thì dễ bị bịa".

**Acceptance Criteria:**

**Given** `scripts/fact_frequency_lab.py --n-facts 500`
**When** chạy
**Then** script tạo các sự kiện giả "thủ đô của X là Y" với X, Y bịa ra, trộn vào dữ liệu theo tần suất đã định, train, rồi in độ chính xác nhớ lại theo từng mức tần suất
**And** kết quả lưu CSV và PNG, cùng seed cho kết quả giống nhau.

**Given** các sự kiện dùng tên bịa
**When** đánh giá
**Then** không có rò rỉ từ dữ liệu thật (kiểm bằng test: tên không xuất hiện trong corpus mẫu).

### Story 7.3: Reversal curse bằng tiếng Việt

As a chủ dự án,
I want thử xem model có trả lời được câu ngược không,
So that tôi tự kiểm chứng reversal curse trên model của mình.

**Acceptance Criteria:**

**Given** tập sự kiện bịa dạng "A là mẹ của B" và hai bộ câu hỏi (xuôi, ngược)
**When** tôi chạy `scripts/reversal_lab.py`
**Then** script train model nhỏ chỉ trên chiều xuôi, rồi báo độ chính xác ở câu hỏi xuôi và ngược
**And** có cờ `--reverse-augment` thêm câu đảo chiều vào dữ liệu và báo lại kết quả để so sánh.

### Story 7.4: Phát hiện induction head và ablation từng head

As a chủ dự án,
I want tìm các head thực hiện "sao chép theo ngữ cảnh" và xem bỏ chúng thì loss tăng bao nhiêu,
So that tôi có bằng chứng can thiệp chứ không chỉ quan sát.

**Acceptance Criteria:**

**Given** một checkpoint
**When** tôi chạy `scripts/induction_heads.py --ckpt <path>`
**Then** script sinh chuỗi token ngẫu nhiên lặp lại hai lần, đo mức giảm loss ở lần lặp thứ hai và điểm induction của từng head
**And** quét ablation: đặt đầu ra từng head về 0, ghi mức tăng loss, in bảng xếp hạng head.

**Given** model có GQA
**When** ablation
**Then** script xử lý đúng việc các head dùng chung K/V (ablate theo đầu ra attention của từng query head).

## Epic 8: Công bố và giải thích

Người ngoài đọc được model card rõ ràng về dữ liệu và giấy phép, và một bài giải thích tiếng Anh gắn với thí nghiệm chạy được.

### Story 8.1: Model card và giấy phép dữ liệu

As a chủ dự án,
I want sinh model card từ `meta.json` và kết quả đánh giá,
So that người dùng biết model học từ đâu và dùng theo điều kiện nào.

**Acceptance Criteria:**

**Given** một checkpoint, `meta.json` có mục `sources` và `results.csv`
**When** tôi chạy `scripts/make_model_card.py --ckpt <path>`
**Then** `MODEL_CARD.md` liệt kê kiến trúc, số tham số, nguồn dữ liệu, giấy phép từng nguồn, kết quả đánh giá chính và giới hạn đã biết
**And** có mục "Câu hỏi pháp lý còn mở" nêu rõ CC BY-SA có thể ràng buộc việc phát hành trọng số và chưa có kết luận.

**Given** `meta.json` thiếu thông tin giấy phép
**When** sinh model card
**Then** script dừng và liệt kê nguồn nào còn thiếu.

### Story 8.2: Bài giải thích tiếng Anh gắn với thí nghiệm

As a người đọc bên ngoài,
I want đọc một bài tiếng Anh giải thích cách LLM dự đoán token kèm lệnh chạy lại thí nghiệm,
So that tôi hiểu cơ chế và tự kiểm chứng.

**Acceptance Criteria:**

**Given** các công cụ ở Epic 6 và Epic 7 đã có
**When** tôi đọc `docs/how_llms_predict.md`
**Then** mỗi chương có một thí nghiệm kèm lệnh cụ thể và hình kết quả: phân phối xác suất, logit lens, grokking, nhớ theo tần suất
**And** mỗi khẳng định về cơ chế trích dẫn nguồn từ `research.md` và dùng cách nói "biểu diễn/tính toán ở quy mô nhỏ" (NFR8).

**Given** bài viết có ví dụ ví von (nén, dự báo thời tiết, thám tử)
**When** dùng
**Then** mỗi ví von nêu rõ điểm giống và điểm khác, không dùng để thay cho bằng chứng.
