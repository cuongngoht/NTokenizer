# Kiến trúc mô hình (Model Architecture)

*[English version](model_architecture.md)*

Giải thích chi tiết về kiến trúc Transformer decoder-only được cài đặt trong
[`src/ntokenizer/model.py`](../src/ntokenizer/model.py). Tài liệu này tập trung giải thích **mỗi
thành phần làm gì** và **vì sao được chọn**. Xem thêm
[DEVELOPMENT.md](DEVELOPMENT.md#step-6--model-architecture) để biết các hyperparameter
mặc định và cách dùng CLI.

---

## Mục lục

1. [Tổng quan](#t%E1%BB%95ng-quan)
2. [Cấu hình (`GPTConfig`)](#c%E1%BA%A5u-h%C3%ACnh-gptconfig)
3. [Token embedding & weight tying](#token-embedding--weight-tying)
4. [RMSNorm](#rmsnorm)
5. [Rotary Positional Embeddings (RoPE)](#rotary-positional-embeddings-rope)
6. [Grouped Query Attention (GQA)](#grouped-query-attention-gqa)
7. [Causal self-attention](#causal-self-attention)
8. [KV Cache](#kv-cache)
9. [SwiGLU MLP](#swiglu-mlp)
10. [Transformer Block](#transformer-block)
11. [Module `GPT` đầy đủ](#module-gpt-%C4%91%E1%BA%A7y-%C4%91%E1%BB%A7)
12. [Sinh văn bản (`generate`)](#sinh-v%C4%83n-b%E1%BA%A3n-generate)
13. [Số lượng tham số & kiểm tra nhanh](#s%E1%BB%91-l%C6%B0%E1%BB%A3ng-tham-s%E1%BB%91--ki%E1%BB%83m-tra-nhanh)
14. [So sánh GPT-2 và bản v2 này](#so-s%C3%A1nh-gpt-2-v%C3%A0-b%E1%BA%A3n-v2-n%C3%A0y)

---

## Tổng quan

Mô hình này là một **Transformer decoder-only** — cùng họ với GPT-2, LLaMA,
Mistral — được viết bằng ~320 dòng PyTorch thuần, không phụ thuộc thư viện
ngoài ngoài `torch`. Đầu vào là một batch token ID, đầu ra là dự đoán token
kế tiếp tại mọi vị trí.

```
input_ids  [B, T]
    │
    └─ Token embedding  wte  [vocab_size, C]  →  [B, T, C]
         │                 (không có bảng vị trí riêng — RoPE đảm nhiệm việc này)
         ├─ Block 0:
         │    pre-RMSNorm → GQA Attention (RoPE) → residual
         │    pre-RMSNorm → SwiGLU MLP           → residual
         ├─ Block 1 … N-1: (giống nhau)
         │
         └─ RMSNorm cuối  →  [B, T, C]
              │
              └─ LM head  Linear(C, vocab_size, bias=False)  →  [B, T, vocab_size]
                          (weight-tied với wte)
```

`B` = batch size · `T` = độ dài chuỗi · `C` = `n_embd`

So với GPT-2 gốc, bốn thành phần đã được thay bằng phiên bản hiện đại hơn
(RoPE, RMSNorm, SwiGLU, GQA), và việc sinh văn bản (generation) được bổ sung
KV cache cùng cơ chế sampling phong phú hơn (top-k, top-p, repetition
penalty). Mỗi phần được giải thích riêng dưới đây.

---

## Cấu hình (`GPTConfig`)

Toàn bộ hyperparameter nằm trong một dataclass duy nhất, `GPTConfig`, nên một
mô hình được mô tả đầy đủ chỉ bằng một object:

| Trường | Ý nghĩa |
|---|---|
| `vocab_size` | Số lượng token khác nhau mà mô hình có thể sinh ra / nhận vào — phải khớp với tokenizer. |
| `block_size` | Độ dài ngữ cảnh (context) tối đa tính theo token. Giới hạn cả chuỗi huấn luyện và kích thước bảng tần số RoPE. |
| `n_layer` | Số lượng Transformer block được xếp chồng lên nhau. |
| `n_head` | Số lượng attention head phía query. |
| `n_kv_head` | Số lượng head key/value (`n_head` phải chia hết cho giá trị này — xem [GQA](#grouped-query-attention-gqa)). |
| `n_embd` | Độ rộng mô hình `C` — kích thước vector ẩn của mỗi token. |
| `dropout` | Xác suất dropout dùng trong attention và MLP; đặt về `0` khi inference. |
| `rope_theta` | Tần số cơ sở dùng để tính góc xoay của RoPE. |

Mọi thứ phía sau — kích thước head, kích thước ẩn của MLP, shape của KV
cache — đều được suy ra từ vài con số này.

---

## Token embedding & weight tying

Token ID đầu vào được tra trong một bảng embedding duy nhất `wte`, shape
`[vocab_size, n_embd]`. Khác với GPT-2 gốc, **không có bảng vị trí học được
riêng (`wpe`)** — thông tin vị trí được đưa vào sau, ngay trong attention,
thông qua RoPE.

Lớp chiếu đầu ra (`lm_head`, shape `[n_embd, vocab_size]`) dùng **chung ma
trận trọng số** với `wte`:

```python
self.transformer.wte.weight = self.lm_head.weight
```

Đây gọi là **weight tying**. Về trực giác: cùng một ma trận dùng để ánh xạ
token ID → vector, cũng có thể dùng để ánh xạ ngược vector ẩn → điểm số cho
từng token — cả hai đều đang làm việc "chuyển đổi giữa không gian token và
không gian embedding", chỉ khác chiều. Việc dùng chung này giảm một nửa số
tham số dành cho embedding, đồng thời đóng vai trò như một dạng regularizer
nhẹ.

---

## RMSNorm

Đầu vào của mọi sub-layer đều được chuẩn hóa (normalize) trước (xem
[Transformer Block](#transformer-block)). Thay vì `LayerNorm`, mô hình dùng
**RMSNorm**:

```
RMSNorm(x) = x / RMS(x) * weight        RMS(x) = sqrt(mean(x²) + ε)
```

Khác biệt so với LayerNorm: không trừ giá trị trung bình (mean), và không có
bias — chỉ có một hệ số scale học được theo từng channel (`weight`). Cụ thể,
trong `RMSNorm.forward`, phép tính được thực hiện ở kiểu `float32` để ổn định
số học, sau đó chuyển lại về dtype ban đầu.

**Vì sao:** bỏ bước trừ mean giúp giảm một phép reduction và một tham số bias
ở mỗi lớp normalize, mà không làm giảm chất lượng đáng kể ở các quy mô mô
hình lên đến hàng tỉ tham số. Nó đơn giản hơn và nhanh hơn một chút, đó là lý
do RMSNorm được dùng trong LLaMA, Mistral, Falcon và hầu hết LLM mở hiện đại.

---

## Rotary Positional Embeddings (RoPE)

### Vấn đề cần giải quyết

GPT-2 gốc cộng thêm một **positional embedding học được** `wpe[t]` vào token
embedding tại mỗi vị trí `t`. Bảng này có một vector cố định cho mỗi vị trí
tối đa đến `block_size` — nó không có cách nào biểu diễn một vị trí chưa từng
thấy khi huấn luyện, nên mô hình không thể tổng quát hóa sang chuỗi dài hơn.

### RoPE hoạt động như thế nào

RoPE thay vào đó **xoay (rotate)** vector query và key theo một góc tỉ lệ với
vị trí của chúng, trước khi tính dot-product trong attention:

```
q_rotated = q * e^{i * t * θ}     (phép nhân số phức, áp dụng theo từng cặp chiều trong head)
```

Mỗi cặp chiều trong một head được coi như một số phức; việc xoay Q và K theo
vị trí tương ứng khiến dot-product `q·k` phụ thuộc một cách tự nhiên vào
khoảng cách **tương đối** `t_q - t_k`, không phải vị trí tuyệt đối.

Trong code, `precompute_freqs_cis` tạo một bảng số phức đơn vị shape
`[max_seq_len, head_dim // 2]` một lần duy nhất (lưu như một buffer không học
được), và `apply_rotary_emb` áp dụng bảng này lên Q và K ở mỗi lượt forward:

```
q, k  [B, H, T, head_dim]
    │  xem chiều cuối như (head_dim/2) cặp số phức
    ▼
q * freqs_cis[t]   ← xoay mỗi cặp theo góc t·θ_k
    │
    ▼
q_rotated, k_rotated  (cùng shape, cùng dtype)
```

Vì Q và K dùng chung `head_dim`, cùng một lát `freqs_cis` áp dụng cho cả hai,
dù GQA khiến chúng có *số lượng* head khác nhau.

**Vì sao:** không thêm tham số học được, attention trở nên nhạy với vị trí
tương đối một cách tự nhiên, và mô hình tổng quát hóa tốt hơn sang độ dài
ngữ cảnh vượt quá lúc huấn luyện. Đây là cơ chế vị trí được dùng trong LLaMA,
Mistral, Qwen và DeepSeek.

---

## Grouped Query Attention (GQA)

### Vấn đề cần giải quyết

Trong Multi-Head Attention (MHA) chuẩn, số head key/value bằng số head query.
Khi sinh văn bản, KV cache phải lưu một cặp K và V cho mỗi layer, mỗi head,
mỗi token đã sinh trước đó — bộ nhớ tăng theo `O(T × n_head × head_dim)`. Với
nhiều head, cache này trở thành chi phí bộ nhớ chủ yếu khi ngữ cảnh dài.

### GQA hoạt động như thế nào

GQA tách riêng số lượng head: `n_head` head query dùng chung `n_kv_head` head
key/value, theo từng nhóm kích thước `n_rep = n_head // n_kv_head`.

```
n_head = 8 (Q heads)     n_kv_head = 2 (KV heads)
                                      → 4 Q head dùng chung 1 cặp KV

Q heads:  [Q0 Q1 Q2 Q3] [Q4 Q5 Q6 Q7]
                │               │
KV heads: [ K0/V0 ]     [ K1/V1 ]
```

Trong `CausalSelfAttention`, lớp chiếu K/V chỉ xuất ra `n_kv_head * head_dim`
channel (nhỏ hơn Q với `n_head * head_dim`), và trước khi tính dot-product
attention, mỗi head KV được lặp lại `n_rep` lần (`repeat_interleave`) để khớp
với nhóm head Q tương ứng.

Hai trường hợp đặc biệt rơi ra từ cùng một đoạn code:
- `n_kv_head == n_head` → MHA chuẩn (`n_rep == 1`, không cần lặp).
- `n_kv_head == 1` → Multi-Query Attention (MQA), thiết lập cực đoan nhất.

**Vì sao:** KV cache nhỏ lại đúng `n_head / n_kv_head` lần, chất lượng giảm
không đáng kể khi `n_kv_head ≥ 2`. Đây là lựa chọn tiêu chuẩn trong LLaMA
2/3, Mistral và Gemma.

---

## Causal self-attention

`CausalSelfAttention` kết hợp RoPE, GQA và KV cache lại với nhau:

1. Chiếu `x` thành Q, K, V bằng các lớp `Linear` riêng, không bias (Q có kích
   thước cho `n_head`, K/V có kích thước nhỏ hơn cho `n_kv_head`).
2. Áp dụng RoPE lên Q và K mới tính (không áp dụng lên V — phép xoay chỉ mã
   hóa vị trí cho điểm số attention, không phải cho giá trị được tổng hợp).
3. Nếu đã có KV cache từ trước, nối (prepend) nó vào K/V mới (xem [KV
   Cache](#kv-cache)).
4. Lặp lại các head KV `n_rep` lần để khớp số head của Q (mở rộng cho GQA).
5. Tính scaled dot-product attention với **causal mask** — mỗi vị trí query
   chỉ được chú ý (attend) đến các vị trí key ở cùng vị trí hoặc trước nó.
6. Ghép các head lại và chiếu về `n_embd` bằng `o_proj`.

Causal mask cần xử lý cẩn thận khi có KV cache: một query mới ở vị trí tuyệt
đối `T_k - 1` vẫn phải được phép nhìn thấy **toàn bộ** phần cache phía trước,
không chỉ chính nó. Code xử lý điều này bằng cách dịch (offset) mask tam
giác theo `T_k - T` (hoặc, khi có sẵn hàm fused
`scaled_dot_product_attention` của PyTorch, chỉ truyền `is_causal=True` khi
không có phần cache phía trước — ngược lại không cần mask nào cả, vì một
query chú ý đến toàn bộ các key trong quá khứ chính là ý nghĩa của "causal"
trong trường hợp đó).

Toàn bộ block này nằm trong một **residual connection kiểu pre-norm** (xem
[Transformer Block](#transformer-block)): `x = x + Attention(RMSNorm(x))`.

---

## KV Cache

### Vấn đề cần giải quyết

Theo cách ngây thơ (naive), để sinh token thứ `N+1`, phải chạy lại toàn bộ
forward pass qua tất cả `T` token đã có — việc tính lại attention là
`O(T²)` cho mỗi token mới, nên sinh `N` token tốn tổng cộng `O(N × T²)`.

### Cách hoạt động

Key và value là hàm xác định (deterministic) của các token trong quá khứ (do
tính causal, chúng không phụ thuộc vào token tương lai), nên có thể **tính
một lần và tái sử dụng**. `CausalSelfAttention.forward` nhận một `past_kv =
(past_K, past_V)` tùy chọn, nối nó với K/V mới tính, và trả về cache đã mở
rộng để bên gọi lưu lại:

```
Bước 0 (prefill):  forward toàn bộ prompt        → cache K,V cho mọi vị trí
Bước 1:            forward chỉ token mới          → nối K,V của nó vào cache
Bước 2:            forward chỉ token mới          → nối tiếp
   ...
```

Mỗi bước sinh sau prefill giờ chỉ tốn `O(T)` thay vì `O(T²)`, vì chỉ cần tính
Q/K/V cho một token mới, phần còn lại đọc từ cache. `GPT.generate` quản lý
việc này qua các layer (một list `(K, V)` cho từng layer) và **loại bỏ**
cache — tính lại từ `block_size - 1` token gần nhất — nếu độ dài tích lũy sẽ
vượt quá `block_size`.

**Vì sao:** với ngữ cảnh 256 token sinh thêm 200 token mới, cách này giảm
khoảng 50 lần số FLOPs cho attention, vì mô hình liên tục tránh tính lại
attention O(T²) trên phần ngữ cảnh đang tăng dần ở mỗi bước.

---

## SwiGLU MLP

Sub-layer feed-forward là một MLP có **cổng (gated)** thay vì MLP GELU cổ
điển.

GPT-2 gốc:
```
MLP(x) = W₂ · GELU(W₁ · x)          hidden_dim = 4 × n_embd
```

SwiGLU:
```
SwiGLU(x) = W_down · ( SiLU(W_gate · x) ⊙ W_up · x )     hidden_dim ≈ 8/3 × n_embd
```

`W_gate` và `W_up` đều chiếu `x` lên chiều ẩn (hidden); nhánh gate đi qua
SiLU rồi nhân theo phần tử (⊙) với nhánh up, trước khi được chiếu về lại
chiều gốc. Số hạng `SiLU(W_gate · x)` đóng vai trò như một bộ lọc học được,
phụ thuộc dữ liệu, tác động lên các đặc trưng đã chiếu lên — một số channel
được giữ lại, một số bị triệt bớt.

Chiều ẩn được đặt có chủ đích là `8/3 × n_embd` (làm tròn lên bội số của 64
để tối ưu phần cứng) thay vì `4×`, vì phiên bản có gate có ba ma trận trọng
số thay vì hai — cách này giữ tổng số tham số tương đương với MLP GELU
thông thường trong khi vẫn thêm được cổng gate.

**Vì sao:** trên thực nghiệm, cơ chế gating cho loss thấp hơn một cách nhất
quán so với MLP GELU thuần ở cùng ngân sách tham số. Được dùng trong LLaMA,
PaLM, Gemma.

---

## Transformer Block

Mỗi `Block` gồm hai sub-layer residual, cả hai dùng **pre-normalization**
(chuẩn hóa *trước* sub-layer, không phải sau):

```
x = x + Attention( RMSNorm(x) )
x = x + MLP( RMSNorm(x) )
```

Pre-norm giữ cho gradient lan truyền sạch qua residual stream khi xếp chồng
nhiều block — đây là lựa chọn chuẩn trong hầu hết Transformer hiện đại, trái
với post-norm kiểu GPT-1.

`Block.forward` trả về cả hidden state đã cập nhật và cặp `(K, V)` mới cho
layer đó, để `GPT.forward` có thể truyền KV cache qua tất cả layer một cách
đồng nhất.

---

## Module `GPT` đầy đủ

`GPT` ghép mọi thứ lại với nhau:

- `transformer.wte` — token embedding (weight-tied với `lm_head`, xem
  [phần trên](#token-embedding--weight-tying)).
- `transformer.h` — một `ModuleList` gồm `n_layer` `Block`.
- `transformer.ln_f` — một RMSNorm cuối cùng trước lớp chiếu đầu ra.
- `freqs_cis` — bảng RoPE, tính trước một lần cho `block_size` vị trí và
  đăng ký như một buffer không học được (nên nó di chuyển theo
  `.to(device)` nhưng không bị optimizer đụng tới).

**Khởi tạo trọng số (initialization)** theo cách của GPT-2: mọi trọng số
`Linear`/`Embedding` được lấy mẫu từ `Normal(0, 0.02)`. Ngoài ra, mỗi lớp
chiếu đưa trực tiếp trở lại residual stream (`o_proj` trong attention,
`down` trong MLP) được khởi tạo lại với std nhỏ hơn, `0.02 /
sqrt(2 * n_layer)` — điều này giúp phương sai (variance) của residual stream
không tăng vô hạn khi xếp chồng nhiều block, và càng quan trọng khi
`n_layer` càng lớn.

`forward(input_ids, targets=None, past_kvs=None)` trả về một tuple 3 phần tử
`(logits, loss, new_past_kvs)`:
- Khi có `targets` (huấn luyện), nó tính logits cho **mọi** vị trí và
  cross-entropy loss so với targets.
- Khi không có `targets` (inference), nó chỉ chiếu vị trí **cuối cùng** ra
  logits (`x[:, [-1], :]`) — không có lý do gì để tính logits cho các token
  không được lấy mẫu, nên cách này tiết kiệm một phép matmul lớn trên chiều
  `vocab_size`.

---

## Sinh văn bản (`generate`)

`GPT.generate` là một vòng lặp sampling tự hồi quy (autoregressive) xây dựng
xung quanh KV cache:

```
prefill:      forward toàn bộ prompt (cắt cho khớp block_size) → cache K,V
với mỗi token trong max_new_tokens:
    chỉ forward token mới nhất, dùng cache                  (O(T) mỗi bước)
    lấy logits của vị trí cuối cùng
    repetition_penalty → temperature → top_k → top_p → softmax → sample
    nối token vừa lấy mẫu, mở rộng cache
```

Các cơ chế điều khiển sampling được áp dụng theo thứ tự này, và có thể kết
hợp tự do:

1. **Repetition penalty** — logits của các token đã xuất hiện trong chuỗi
   được sinh sẽ bị kéo về gần 0 (logits dương thì chia, logits âm thì nhân
   với penalty), giúp giảm việc lặp lại (loop).
2. **Temperature** — logits chia cho `T` trước softmax; `<1` làm phân phối
   sắc nét hơn (deterministic hơn), `>1` làm phân phối phẳng hơn (ngẫu nhiên
   hơn).
3. **Top-k** — các logits ngoài `k` giá trị cao nhất bị đặt thành `-inf`.
4. **Top-p (nucleus)** — sau khi sắp xếp theo xác suất, giữ lại tập nhỏ nhất
   các token có tổng xác suất tích lũy đạt `p`; che phần còn lại.
5. Cuối cùng, `softmax` + `torch.multinomial` lấy mẫu token kế tiếp.

Nếu cache sẽ vượt quá `block_size`, `generate` sẽ loại bỏ cache và tính lại
từ `block_size - 1` token gần nhất — một chiến lược sliding-window đơn giản,
không phải một cơ chế nén cache phức tạp hơn.

---

## Số lượng tham số & kiểm tra nhanh

Đoạn code sau kiểm thử toàn bộ kiến trúc từ đầu đến cuối — dán vào một phiên
`python` (sau khi cài package bằng `pip install -e .`) để tự thử:

```python
import math
import torch
from ntokenizer.config import GPTConfig
from ntokenizer.model import GPT

config = GPTConfig(vocab_size=32000, block_size=512, n_layer=8, n_head=8, n_kv_head=2, n_embd=512, dropout=0.0)
model = GPT(config)
print(f"Parameters : {model.count_parameters():,}")

B, T = 2, 64
ids = torch.randint(0, config.vocab_size, (B, T))
targets = torch.randint(0, config.vocab_size, (B, T))

logits, loss, kvs = model(ids, targets)
print(f"logits : {list(logits.shape)}")
print(f"loss   : {loss.item():.4f}  (kỳ vọng ≈ {math.log(config.vocab_size):.4f} = ln(vocab_size))")
print(f"KV cache layers : {len(kvs)}  shapes : K{list(kvs[0][0].shape)} V{list(kvs[0][1].shape)}")

model.eval()
seed = torch.zeros((1, 1), dtype=torch.long)
out = model.generate(seed, max_new_tokens=30, temperature=0.8, top_k=50, top_p=0.9)
print(f"generated token IDs : {out[0].tolist()}")
```

Đoạn code này tạo một `GPTConfig` nhỏ, in ra số lượng tham số
(`model.count_parameters()`), chạy một forward pass ở chế độ huấn luyện, và
kiểm tra rằng loss gần với `ln(vocab_size)` — loss cross-entropy kỳ vọng của
một mô hình chưa huấn luyện, đoán ngẫu nhiên đều trên toàn bộ vocabulary. Sau
đó nó chạy `generate()` để sinh vài token và in ra shape của KV cache, xác
nhận toàn bộ luồng prefill → decode tuần tự hoạt động đúng. Các assertion
tương ứng được kiểm tra tự động trong `tests/test_model.py` và
`tests/test_generation.py`.

---

## So sánh GPT-2 và bản v2 này

| Thành phần | GPT-2 (gốc) | Bản cài đặt này | Lợi ích |
|---|---|---|---|
| Mã hóa vị trí | Bảng `wpe` học được | **RoPE** | Tổng quát hóa vượt độ dài huấn luyện; không thêm tham số |
| Normalization | `LayerNorm` (có bias) | **RMSNorm** (không bias) | Đơn giản hơn, nhanh hơn một chút, vẫn ổn định |
| Activation của MLP | `GELU` với hệ số mở rộng 4× | **SwiGLU** với hệ số mở rộng 8/3× | Gradient lan truyền tốt hơn; loss thấp hơn nhất quán |
| Attention head | Multi-Head Attention | **Grouped Query Attention** | Ít head KV hơn → KV cache nhỏ hơn, inference nhanh hơn |
| Sinh văn bản | Tính lại toàn bộ ngữ cảnh mỗi bước | **KV Cache** | O(T) mỗi bước thay vì O(T²) |
| Sampling | Chỉ có top-k | **Top-k + Top-p + Repetition penalty** | Đầu ra tự nhiên hơn, ít lặp lại hơn |

Xem [DEVELOPMENT.md](DEVELOPMENT.md#step-6--model-architecture) để biết các flag CLI
tương ứng, hyperparameter mặc định và pipeline huấn luyện.
