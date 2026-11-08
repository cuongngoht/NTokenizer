# Tại sao LLM trả lời đúng dù "chỉ là dự đoán token tiếp theo"?

*Tài liệu nghiên cứu. Liên hệ trực tiếp với code trong repo này:
[`src/ntokenizer/model.py`](../src/ntokenizer/model.py) (hàm `forward` và `generate`).*

---

## Mục lục

1. [Câu hỏi đặt ra](#1-câu-hỏi-đặt-ra)
2. [Thuật toán thực sự là gì (đúng trong repo này)](#2-thuật-toán-thực-sự-là-gì)
3. [Luận điểm cốt lõi: dự đoán tốt = hiểu cấu trúc](#3-luận-điểm-cốt-lõi-dự-đoán-tốt--phải-mô-hình-hoá-cấu-trúc-sinh-ra-dữ-liệu)
4. [Bằng chứng thực nghiệm: bên trong model có gì](#4-bằng-chứng-thực-nghiệm-bên-trong-model-có-gì)
5. [Vì sao "xác suất" lại cho ra câu trả lời chắc chắn](#5-vì-sao-xác-suất-lại-cho-ra-câu-trả-lời-chắc-chắn)
6. [Quy mô: tại sao model lớn mới "đúng"](#6-quy-mô-tại-sao-model-lớn-mới-đúng)
7. [Từ "dự đoán văn bản" sang "trả lời đúng": post-training](#7-từ-dự-đoán-văn-bản-sang-trả-lời-đúng-post-training)
8. [Suy luận nhiều bước: chain-of-thought là "bộ nhớ nháp"](#8-suy-luận-nhiều-bước-chain-of-thought-là-bộ-nhớ-nháp)
9. [Mặt trái: khi nào và vì sao LLM sai](#9-mặt-trái-khi-nào-và-vì-sao-llm-sai)
10. [Thí nghiệm bạn có thể chạy với repo này](#10-thí-nghiệm-bạn-có-thể-chạy-với-repo-này)
11. [Tóm tắt](#11-tóm-tắt)
12. [Tài liệu tham khảo](#12-tài-liệu-tham-khảo)

---

## 1. Câu hỏi đặt ra

Một LLM được huấn luyện với **một mục tiêu duy nhất**: cho một chuỗi token,
đoán token kế tiếp. Không có module "logic", không có cơ sở dữ liệu sự thật,
không có bộ giải toán. Vậy tại sao nó trả lời đúng "Thủ đô của Việt Nam là
**Hà Nội**", viết được code chạy được, giải được bài toán?

Câu trả lời ngắn: **"chỉ là dự đoán" là một mô tả về _giao diện_ (input →
phân phối xác suất), không phải mô tả về _những gì xảy ra bên trong_.** Để
dự đoán token tiếp theo với sai số thấp trên hàng nghìn tỷ token văn bản đa
dạng, cách rẻ nhất (về tham số) mà gradient descent tìm ra là xây dựng các
biểu diễn nội bộ của ngữ pháp, sự kiện, quan hệ, thậm chí cả các thuật toán.
Phần còn lại của tài liệu giải thích và đưa bằng chứng cho luận điểm này.

---

## 2. Thuật toán thực sự là gì

Trong repo này, toàn bộ "trí tuệ" được huấn luyện bởi đúng một dòng
(`model.py`, trong `forward`):

```python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

Tức là cực tiểu hoá

$$\mathcal{L} = -\frac{1}{N}\sum_{t} \log p_\theta(x_t \mid x_{<t})$$

Và khi sinh văn bản (`generate`), model chỉ lặp lại: tính `logits` cho vị trí
cuối → (tuỳ chọn) chia `temperature`, lọc `top_k`/`top_p` → `softmax` →
lấy mẫu một token → nối vào chuỗi → lặp.

Hai điểm quan trọng thường bị hiểu sai:

| Hiểu sai | Thực tế |
|---|---|
| "Dự đoán = đoán mò / thống kê n-gram" | Phân phối $p(x_t\mid x_{<t})$ được tính bởi một mạng sâu (nhiều block attention + MLP) nhìn **toàn bộ** ngữ cảnh, không phải đếm tần suất cụm từ. |
| "Xác suất = không chắc chắn" | Với câu hỏi có đáp án rõ, phân phối thường rất nhọn: token đúng có thể chiếm >99% khối lượng xác suất. Lấy mẫu từ phân phối đó gần như tất định. |

---

## 3. Luận điểm cốt lõi: dự đoán tốt ⇒ phải mô hình hoá cấu trúc sinh ra dữ liệu

### 3.1. Dự đoán = nén = hiểu (theo nghĩa thông tin)

Shannon (1948, 1951) đã chỉ ra: một mô hình dự đoán tốt có thể dùng để **nén**
dữ liệu (arithmetic coding), và độ dài mã tối ưu chính bằng cross-entropy.
Delétang et al. (2023, *Language Modeling Is Compression*) cho thấy LLM là
bộ nén tổng quát cực mạnh: Chinchilla 70B nén ảnh ImageNet còn 43,4% (PNG
58,5%) và âm thanh còn 16,4% (FLAC 30,3%) dù chỉ huấn luyện chủ yếu trên văn
bản. **Lưu ý:** đây là tỷ lệ *không tính kích thước model*; tính cả hàng trăm
GB trọng số thì LLM nén kém hơn nhiều. Huang et al. (COLM 2024) còn thấy trên
~30 LLM, điểm benchmark tương quan gần tuyến tính với hiệu quả nén. Đó là
tương quan, không phải chứng minh.

Hệ quả: để giảm loss, model phải tìm ra **các quy luật** giải thích dữ liệu.
Ghi nhớ từng câu là cách nén tồi; hiểu quy luật (ngữ pháp, phép cộng, quan hệ
địa lý) là cách nén tốt.

### 3.2. Ví dụ trực quan

Xét các câu trong corpus:

```
"... 17 + 25 = 42 ..."
"Trong ván cờ, sau khi Trắng đi Mã f3, Đen ..."
"Hàm này trả về None vì biến x chưa được ..."
"Kết luận: kẻ sát nhân chính là ..."
```

Để dự đoán token tiếp theo **đúng** trong các câu này, model cần (một phần)
biết làm phép cộng, biết trạng thái bàn cờ, hiểu ngữ nghĩa code, theo dõi
chuỗi lập luận của câu chuyện. Không có "lối tắt thống kê" nào cho loss thấp
trên **mọi** trường hợp như vậy. Ilya Sutskever tóm tắt ý này: *dự đoán token
tiếp theo đủ tốt nghĩa là phải hiểu thực tại đã sinh ra token đó*.

### 3.3. Mục tiêu đơn giản ≠ năng lực đơn giản

Mục tiêu huấn luyện đơn giản không giới hạn độ phức tạp của lời giải. So sánh
với tiến hoá: mục tiêu là "sống sót và sinh sản", nhưng lời giải là mắt, não,
ngôn ngữ. Next-token prediction là một bài toán **meta**: nó chứa trong đó
hàng triệu bài toán con (dịch, tóm tắt, cộng, phân loại...) vì văn bản trên
Internet chứa chúng.

---

## 4. Bằng chứng thực nghiệm: bên trong model có gì

Đây là phần quan trọng nhất — không chỉ lý thuyết, mà người ta đã **mở model
ra và nhìn thấy** các cấu trúc này (lĩnh vực *mechanistic interpretability*).

| Phát hiện | Nghiên cứu | Ý nghĩa |
|---|---|---|
| **Mô hình thế giới**: GPT chỉ học chuỗi nước đi Othello (không thấy bàn cờ) nhưng bên trong có biểu diễn trạng thái bàn cờ; can thiệp vào biểu diễn đó làm thay đổi nước đi dự đoán. Bài gốc cần probe *phi tuyến*; biểu diễn chỉ tuyến tính khi mã hoá ô theo "của tôi / của đối thủ". Model cờ vua cũng có bàn cờ đọc được bằng probe tuyến tính (~99,6%). | Li et al. (ICLR 2023); Nanda, Lee & Wattenberg (BlackboxNLP 2023); Karvonen 2024 | Dự đoán chuỗi ⇒ tự xây dựng trạng thái ẩn của "thế giới". **Phản đề:** model có thể qua probe mà world model vẫn không nhất quán (Vafa et al., NeurIPS 2024). |
| **Thuật toán thật**: transformer học cộng modulo dùng biến đổi Fourier + đồng nhất thức lượng giác, có thể đọc ra từ trọng số. | Nanda et al. 2023 (*Progress measures for grokking*) | Model không chỉ nhớ bảng kết quả; nó học **thuật toán tổng quát**. |
| **Grokking**: ban đầu model nhớ tập train (test kém), huấn luyện tiếp thì đột ngột tổng quát hoá. | Power et al. 2022 (workshop ICLR 2021 + preprint) | Regularization + gradient ưu tiên lời giải "gọn" (thuật toán) hơn lời giải "nhớ". |
| **Không gian & thời gian**: Llama-2 có các hướng tuyến tính mã hoá kinh độ/vĩ độ của địa danh và năm của sự kiện. | Gurnee & Tegmark 2023 | Kiến thức được tổ chức có cấu trúc, không phải danh sách rời rạc. |
| **Induction heads**: mạch attention thực hiện "thấy A B trước đó, giờ thấy A → đoán B"; là cơ chế chính của in-context learning. | Olsson et al. 2022 (preprint) | Giải thích vì sao model học được từ ví dụ trong prompt. Bằng chứng nhân quả chỉ có ở model nhỏ; ở model lớn là tương quan. |
| **Sự kiện nằm trong MLP**: lớp feed-forward hoạt động như bộ nhớ key–value; có thể định vị và sửa một sự kiện cụ thể. | Geva et al. 2021; Meng et al. 2022 (*ROME*) | Tương ứng với `MLP`/SwiGLU trong `model.py`. |
| **Suy luận nhiều bước nội bộ & lập kế hoạch**: Claude 3.5 Haiku tính "thủ đô của bang chứa Dallas" qua bước trung gian "Texas" bên trong (thay Texas bằng California thì ra Sacramento); khi làm thơ, chọn từ vần **trước** rồi viết câu dẫn tới nó. | Anthropic 2025 (*On the Biology of a Large Language Model*; chưa bình duyệt, một model) | Dù xuất từng token, bên trong có "nhìn trước" nhiều token. Phương pháp chỉ bắt được *một phần* tính toán. |
| **Model không biết mình tính thế nào**: với 36+59, Claude 3.5 Haiku chạy song song một đường ước lượng độ lớn và một đường tính chữ số cuối; nhưng khi được hỏi, nó mô tả cách "nhớ 1" trong sách giáo khoa. | Anthropic 2025 | Lời model tự giải thích không phải bằng chứng về cơ chế bên trong. |
| **Superposition**: model nén nhiều khái niệm hơn số chiều bằng cách xếp chồng các đặc trưng gần trực giao. | Elhage et al. 2022 | Giải thích vì sao vài tỷ tham số chứa được lượng tri thức khổng lồ. |

**Kết luận phần 4:** "Dự đoán token" là lớp vỏ ngoài. Bên trong là các mạch
tính toán học được: biểu diễn thế giới, thuật toán, bộ nhớ sự kiện, mạch sao
chép ngữ cảnh, lập kế hoạch.

---

## 5. Vì sao "xác suất" lại cho ra câu trả lời chắc chắn

Ví dụ với prompt `"Thủ đô của Việt Nam là"`, logits sau softmax có thể trông
như sau (minh hoạ):

| Token | Xác suất |
|---|---|
| ` Hà` | 0.97 |
| ` thành` | 0.015 |
| ` TP` | 0.006 |
| ...(32k token còn lại) | ~0.009 |

- Câu trả lời đúng **không phải là may mắn**: training trên hàng nghìn câu
  khẳng định cùng một sự thật đã đẩy xác suất token đúng lên gần 1.
- `temperature < 1` hoặc greedy (argmax) làm phân phối càng nhọn hơn. Trong
  `generate`, `logits / temperature` với `temperature → 0` tiến tới argmax.
- `top_k`/`top_p` cắt đuôi phân phối, loại bỏ các token xác suất thấp (nơi
  hay sinh lỗi).
- **Calibration**: model pretrain thường "biết mình biết gì" — xác suất nó
  gán tương quan tốt với tỷ lệ đúng thực tế (Kadavath et al. 2022). Khi
  phân phối phẳng, đó là tín hiệu model không chắc. Lưu ý: báo cáo GPT-4 cho
  thấy base model calibrate tốt nhưng RLHF *làm giảm* calibration, nên xác
  suất của model chat không còn đáng tin bằng.

Nói cách khác: *ngẫu nhiên* chỉ xuất hiện ở bước lấy mẫu, và với kiến thức
vững thì phân phối gần như là một đỉnh duy nhất.

---

## 6. Quy mô: tại sao model lớn mới "đúng"

- **Scaling laws** (Kaplan et al. 2020; Hoffmann et al. 2022 — *Chinchilla*):
  loss giảm theo luật luỹ thừa khi tăng tham số, dữ liệu, compute. Chinchilla
  dùng ~20 token dữ liệu cho mỗi tham số (70B tham số, 1,4T token); đây là
  tỷ lệ suy ra từ bài báo, được Epoch AI tái lập (2024). Thực tế ngày nay cố
  ý train *vượt* tỷ lệ này để model nhỏ hơn, rẻ hơn khi suy luận (Sardana et
  al., ICML 2024).
- Mỗi mức giảm loss nhỏ tương ứng với việc mô hình hoá thêm một lớp cấu trúc:
  đầu tiên là tần suất ký tự → từ → ngữ pháp → sự kiện phổ biến → sự kiện
  hiếm → suy luận nhiều bước.
- **"Emergent abilities"** (Wei et al. 2022): một số năng lực có vẻ xuất hiện
  đột ngột ở quy mô lớn. Schaeffer et al. 2023 chỉ ra phần lớn "đột ngột" là
  do thước đo (exact-match) — loss thực ra giảm mượt, nhưng tác vụ chỉ được
  tính "đúng" khi mọi token đều đúng. Du et al. (NeurIPS 2024) phản bác một
  phần: xét theo pre-training loss, một số năng lực chỉ xuất hiện khi loss vượt
  ngưỡng, kể cả với thước đo liên tục. Tranh luận vẫn còn mở.

Với repo này (model vài chục triệu tham số, train trên Wikipedia tiếng Việt),
bạn sẽ thấy model sinh văn bản trôi chảy, đúng ngữ pháp nhưng **thường sai
sự kiện** — đúng như scaling laws dự đoán: ngữ pháp được học trước, kiến
thức và suy luận cần nhiều tham số và dữ liệu hơn nhiều.

---

## 7. Từ "dự đoán văn bản" sang "trả lời đúng": post-training

Một model chỉ pretrain dự đoán văn bản Internet — gồm cả câu sai, câu đùa,
câu hỏi không có trả lời. Các LLM thương mại còn qua nhiều bước để **dịch
chuyển phân phối** về phía câu trả lời đúng và hữu ích:

1. **Supervised fine-tuning (SFT / instruction tuning)**: train tiếp trên cặp
   (câu hỏi, câu trả lời tốt). Vẫn là cross-entropy y hệt — xem
   [`FINETUNING.vi.md`](FINETUNING.vi.md), Cách 2 và Cách 3 của repo này.
2. **RLHF / RLAIF** (Ouyang et al. 2022 — *InstructGPT*): tối ưu theo điểm
   thưởng từ đánh giá của người/AI, không chỉ bắt chước văn bản.
3. **RL với phần thưởng kiểm chứng được** (RLVR — ví dụ DeepSeek-R1, đăng
   trên *Nature* 9/2025; các "reasoning model"): thưởng khi đáp án toán đúng,
   code qua test. Model tự học **tự kiểm tra, quay lui, sửa lỗi** trong chuỗi
   suy nghĩ mà không cần vết suy luận do người viết. Mục tiêu không còn là
   "giống văn bản Internet" mà là "**ra kết quả đúng**". *Tranh luận mở:* Yue
   et al. (NeurIPS 2025) cho thấy RL tăng pass@1 nhưng base model thắng ở pass@k
   với k lớn, tức RL chủ yếu *chọn lại* năng lực sẵn có; ProRL (NeurIPS 2025)
   cho thấy RL kéo dài có thể vượt base model.
4. **Công cụ & truy xuất**: RAG (Lewis et al. 2020), gọi máy tính, chạy code,
   tìm web — model dự đoán *lời gọi công cụ*, còn sự thật/con số đến từ nguồn
   tin cậy.

Vì vậy "LLM chỉ là predict" là đúng về cơ chế sinh, nhưng **cái được predict
đã được tối ưu trực tiếp cho tính đúng**, không chỉ cho tính "giống văn bản".

---

## 8. Suy luận nhiều bước: chain-of-thought là "bộ nhớ nháp"

Một lượt forward qua transformer có độ sâu cố định (`n_layer` block). Về lý
thuyết, điều này giới hạn lượng tính toán tuần tự trong **một** token
(Feng et al., NeurIPS 2023): không có CoT, transformer độ sâu hằng bị chặn
trên bởi lớp mạch TC0 (độ chính xác log), thậm chí chỉ AC0 nếu độ chính xác
là hằng số (Li et al., ICLR 2024). Với poly(n) bước CoT, transformer nhận
được **đúng bằng lớp P** (Merrill & Sabharwal, ICLR 2024).

**Chain-of-thought** (Wei et al. 2022; Kojima et al. 2022 — "Let's think step
by step") phá giới hạn đó: mỗi token sinh ra được nối lại vào ngữ cảnh (đúng
dòng `idx = torch.cat([idx, next_token], dim=1)` trong `generate`) và trở thành
**bộ nhớ trung gian** cho các bước sau. Với đủ token trung gian, transformer
có thể mô phỏng tính toán tuần tự tổng quát. Đó là lý do:

- Viết ra các bước → tỷ lệ đúng tăng mạnh với toán và logic.
- **Self-consistency** (Wang et al. 2022): sinh nhiều lời giải, chọn đáp án
  xuất hiện nhiều nhất → lỗi ngẫu nhiên bị triệt tiêu.
- Reasoning model "nghĩ lâu hơn" (nhiều token hơn) trước khi trả lời → chính
  xác hơn (test-time compute scaling). Snell et al. (ICLR 2025): model nhỏ
  dùng thêm compute lúc trả lời có thể thắng model lớn hơn 14 lần, nhưng **chỉ
  ở câu dễ và trung bình**; câu khó nhất vẫn cần pretraining mạnh hơn.
- **Cảnh báo:** CoT tăng độ chính xác, nhưng *văn bản* CoT không phản ánh
  trung thực cách model tính. Reasoning model chỉ nhắc tới gợi ý nó thực sự
  dùng trong ~25% (Claude 3.7 Sonnet) đến ~39% (DeepSeek-R1) trường hợp
  (Chen et al. 2025); Turpin et al. (NeurIPS 2023) cũng thấy CoT bỏ qua thiên
  kiến làm lệch câu trả lời.

---

## 9. Mặt trái: khi nào và vì sao LLM sai

Hiểu tại sao đúng cũng giúp hiểu tại sao sai — và các lỗi này đều là **dấu
vết của mục tiêu "dự đoán"**:

| Hiện tượng | Nguyên nhân gốc |
|---|---|
| **Hallucination** (bịa sự kiện, trích dẫn) | Model được train để sinh văn bản *hợp lý*; với sự kiện hiếm nó vẫn sinh ra câu trôi chảy. Kalai et al. 2025 (*Why Language Models Hallucinate*, preprint): tỷ lệ bịa có cận dưới xấp xỉ tỷ lệ sự kiện chỉ xuất hiện **một lần** trong dữ liệu ("singleton rate"), và benchmark chấm 0/1 thưởng việc đoán hơn nói "tôi không biết"; đề xuất phạt câu sai t/(1−t) với ngưỡng tin cậy t. |
| **Reversal curse**: biết "A là mẹ của B" nhưng không trả lời được "B là con của ai" | Học theo chiều trái→phải; Berglund et al. (ICLR 2024): GPT-4 đúng 79% chiều xuôi, 33% chiều ngược. "Reverse training" (Golovneva et al. 2024) giảm được đáng kể → không phải giới hạn cơ bản. |
| **Nhạy với tần suất** ("embers of autoregression"): đếm/đảo chữ, mã hoá kém hơn với chuỗi ít gặp | McCoy et al. (PNAS 2024): GPT-4 giải mã mật mã đúng 51% khi đáp án là câu xác suất cao, chỉ 13% khi xác suất thấp. Với o1 hiện tượng giảm nhưng không mất. |
| **Tính toán nhiều bước dài** (nhân số rất lớn) không có CoT | Giới hạn độ sâu cố định; Dziri et al. 2023 (*Faith and Fate*): model hay ghép mẫu cục bộ thay vì thuật toán hệ thống. |
| **Lỗi do tokenizer** (đếm chữ cái trong từ, số học chữ số) | Model thấy token BPE, không thấy ký tự — xem `scripts/inspect_vocab.py` để thấy một từ tiếng Việt bị tách thế nào. Bằng chứng lẫn lộn (CharBench 2025): tokenizer là *một yếu tố góp phần*, không phải nguyên nhân duy nhất. |
| **Độ bền suy luận** | GSM-Symbolic (Apple 2024): thêm một mệnh đề thừa làm độ chính xác giảm đến 65%. Kết luận "không biết suy luận" vẫn đang bị phản biện về mặt thống kê. |
| **Lỗi dồn tích** | Một token sai được nối vào ngữ cảnh và mọi token sau đều điều kiện hoá trên nó (autoregressive). |

Vì vậy câu trả lời chính xác hơn cho câu hỏi ban đầu là: **LLM đúng ở những
nơi dữ liệu + quy mô + post-training đã khiến cấu trúc thật trở thành cách dự
đoán tốt nhất; LLM sai ở những nơi một lời giải "nghe hợp lý" rẻ hơn lời giải
đúng.**

---

## 10. Thí nghiệm bạn có thể chạy với repo này

1. **Quan sát phân phối xác suất**: sửa tạm `generate` (hoặc viết script) để
   in `top-5` của `F.softmax(logits, -1)` ở mỗi bước. So sánh prompt có đáp án
   rõ ràng (ví dụ cụm từ phổ biến trên Wikipedia) với prompt mơ hồ → thấy
   phân phối nhọn vs. phẳng.
2. **Temperature**: chạy `scripts/sample.py` với `temperature` 0.1, 0.7, 1.5.
   Thấp → lặp lại, an toàn; cao → sáng tạo nhưng nhiều lỗi sự kiện.
3. **Loss theo bước train**: vẽ val loss qua các checkpoint, mỗi checkpoint
   sinh vài câu. Bạn sẽ thấy thứ tự học: ký tự/âm tiết → từ → ngữ pháp →
   một số sự kiện phổ biến.
4. **Grokking thu nhỏ**: tạo dataset `a + b = c (mod 97)` dạng văn bản, train
   một `GPTConfig` rất nhỏ (2 layer) với weight decay lớn, theo dõi train acc
   vs. val acc — tái hiện việc model chuyển từ "nhớ" sang "thuật toán".
5. **Fine-tune Q&A** ([`FINETUNING.vi.md`](FINETUNING.vi.md)): so sánh câu trả
   lời trước/sau SFT cho cùng một câu hỏi — minh hoạ mục 7.

---

## 11. Tóm tắt

1. **Giao diện** của LLM là dự đoán token, nhưng **cơ chế bên trong** là các
   mạch tính toán học được: biểu diễn thế giới, bộ nhớ sự kiện, thuật toán,
   in-context learning.
2. Dự đoán tốt tương đương với nén tốt, và nén tốt đòi hỏi nắm được quy luật
   sinh ra dữ liệu — nên tối ưu cross-entropy trên dữ liệu khổng lồ **ép**
   model học cấu trúc thật.
3. Với kiến thức vững, phân phối xác suất rất nhọn → "dự đoán" gần như tất
   định; temperature/top-p kiểm soát phần ngẫu nhiên còn lại.
4. Quy mô (tham số, dữ liệu, compute) quyết định cấu trúc nào được học.
5. Post-training (SFT, RLHF, RL với phần thưởng kiểm chứng) chuyển mục tiêu
   từ "giống văn bản" sang "đúng và hữu ích".
6. Chain-of-thought biến ngữ cảnh thành bộ nhớ nháp, mở rộng năng lực suy
   luận vượt giới hạn một lượt forward.
7. Lỗi (hallucination, reversal curse, lỗi tokenizer) cũng xuất phát từ chính
   bản chất dự đoán — nên kết hợp truy xuất, công cụ và kiểm chứng khi cần
   độ tin cậy cao.

---

## 12. Tài liệu tham khảo

*Đã kiểm chứng và cập nhật bằng đợt nghiên cứu ngày 2026-11-08 (51 nguồn, có
link và trạng thái bình duyệt):
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
