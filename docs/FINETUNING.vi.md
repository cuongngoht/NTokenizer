# Fine-tuning — tinh chỉnh model trên dữ liệu riêng

*[English version](FINETUNING.md)*

*Tài liệu này mô tả cách fine-tune một checkpoint đã pretrain (ví dụ
`artifacts/checkpoints/gpt_8k_research/ckpt.pt`, train trên Wikipedia) trên
một corpus nhỏ hơn, chuyên biệt hơn — ví dụ văn bản lịch sử hoặc dữ liệu
dạng câu hỏi–trả lời. Xem [`DEVELOPMENT.md`](DEVELOPMENT.md) để biết quy
trình pretraining từ đầu.*

---

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [Yêu cầu bắt buộc: tokenizer phải khớp checkpoint](#yêu-cầu-bắt-buộc-tokenizer-phải-khớp-checkpoint)
3. [Quy trình chung — 3 bước](#quy-trình-chung--3-bước)
4. [Cách 1 — Fine-tune trên văn bản thuần](#cách-1--fine-tune-trên-văn-bản-thuần)
5. [Cách 2 — Fine-tune trên Q&A rút gọn (gộp với văn bản gốc)](#cách-2--fine-tune-trên-qa-rút-gọn-gộp-với-văn-bản-gốc)
6. [Cách 3 — Fine-tune trên dữ liệu Alpaca-block đầy đủ](#cách-3--fine-tune-trên-dữ-liệu-alpaca-block-đầy-đủ)
7. [Giới hạn quan trọng: block_size cố định](#giới-hạn-quan-trọng-block_size-cố-định)
8. [Test model sau khi fine-tune](#test-model-sau-khi-fine-tune)
9. [Tham chiếu flag CLI](#tham-chiếu-flag-cli)
10. [Troubleshooting](#troubleshooting)

---

## Tổng quan

Fine-tuning dùng lại đúng `scripts/train.py` và `scripts/prepare_dataset.py`
của pipeline pretraining, chỉ thêm hai điểm:

| Thành phần | Thay đổi |
|---|---|
| `ntokenizer.config.TrainConfig` | thêm field `init_from: str = ""` |
| `ntokenizer.training.train()` | nếu `out_dir` **chưa có** `ckpt.pt` và `init_from` được set → load *chỉ trọng số model* từ checkpoint đó (optimizer khởi tạo mới, step bắt đầu từ 0) |
| `scripts/train.py` | thêm flag `--init_from <path/to/ckpt.pt>` |
| `scripts/prepare_dataset.py` | thêm flag `--corpus` và `--output-dir` để mã hoá bất kỳ corpus nào, không chỉ `data/interim/corpus.txt` mặc định |

Nếu `out_dir` **đã có** `ckpt.pt` sẵn (fine-tune bị gián đoạn giữa đường),
`train()` luôn resume từ đó và **bỏ qua** `--init_from` — cơ chế này được
test ở `tests/test_finetune.py`.

---

## Yêu cầu bắt buộc: tokenizer phải khớp checkpoint

Checkpoint lưu kích thước embedding theo `vocab_size` của tokenizer đã dùng
khi pretrain. Ví dụ `gpt_8k_research/ckpt.pt` được train với
`artifacts/tokenizer/viwiki_bpe_8k.model` (`vocab_size=8000`). Khi fine-tune,
**phải** dùng đúng tokenizer đó để mã hoá corpus mới — dùng nhầm bản `32k`
hoặc `48k` sẽ làm lệch shape của bảng embedding và `load_state_dict` báo lỗi
ngay khi load checkpoint.

Kiểm tra nhanh vocab_size của một checkpoint:

```bash
.venv/bin/python -c "
import torch
ckpt = torch.load('artifacts/checkpoints/gpt_8k_research/ckpt.pt', map_location='cpu', weights_only=True)
print(ckpt['config'])
"
```

---

## Quy trình chung — 3 bước

```
corpus mới (.txt, mỗi dòng 1 đoạn)
        │
        ▼  scripts/prepare_dataset.py --corpus ... --output-dir ...
data/processed_xxx/{train.bin, val.bin, meta.json}
        │
        ▼  scripts/train.py --data_dir ... --out_dir ... --init_from <ckpt_gốc>
artifacts/checkpoints/xxx/ckpt.pt   ← checkpoint đã fine-tune
        │
        ▼  scripts/sample.py --ckpt ... --tokenizer ...
"văn bản sinh ra"
```

`--output-dir` và `--out_dir` **phải là thư mục mới**, khác với thư mục của
checkpoint gốc — nếu trỏ vào cùng `out_dir` đang có `ckpt.pt`, `train()` sẽ
resume (tiếp tục train tiếp) thay vì fine-tune từ đầu với `init_from`.

---

## Cách 1 — Fine-tune trên văn bản thuần

Dùng khi bạn chỉ có văn bản thô (không có cấu trúc hỏi–đáp), ví dụ một cuốn
sách/tài liệu chuyên ngành. Model học thêm từ vựng + văn phong miền đó,
nhưng vẫn chỉ là **continuation LM** — sinh tiếp văn bản, không trả lời câu
hỏi trực diện.

```bash
# 1. Mã hoá corpus (mỗi dòng 1 đoạn văn, dùng đúng tokenizer 8k)
.venv/bin/python scripts/prepare_dataset.py \
  --corpus data/raw/history/vn_su_luoc_for_training_clean_v2.txt \
  --output-dir data/processed_history \
  --model artifacts/tokenizer/viwiki_bpe_8k.model

# 2. Fine-tune từ checkpoint gốc
.venv/bin/python scripts/train.py \
  --data_dir data/processed_history \
  --out_dir artifacts/checkpoints/gpt_8k_history_finetune \
  --init_from artifacts/checkpoints/gpt_8k_research/ckpt.pt \
  --max_iters 500 --batch_size 16 \
  --learning_rate 3e-5 --min_lr 3e-6 --warmup_iters 10 \
  --eval_interval 25 --eval_iters 20 --device cpu
```

**Kết quả thực tế** (2.223 dòng, ~239K token, dataset "Việt Nam Sử Lược"):
val loss giảm từ **4.92 → 4.19** sau 500 bước. Văn bản sinh ra dùng đúng từ
vựng lịch sử (Gia Định, quân Pháp, huyện/tỉnh/phủ...) nhưng không "trả lời"
được câu hỏi trực tiếp.

---

## Cách 2 — Fine-tune trên Q&A rút gọn (gộp với văn bản gốc)

Dùng khi có dữ liệu dạng `{"instruction": ..., "output": ...}` (không cần
giữ đoạn nguồn `input` nếu nó đã nằm trong corpus văn bản thuần) — giữ mỗi
ví dụ ngắn để nằm lọt trong `block_size=256` token, giúp model học đúng
hành vi "thấy cụm 'Trả lời:' thì dừng kể chuyện và trả lời".

```bash
# 1. Ghép corpus văn bản thuần + Q&A thành 1 file, mỗi dòng 1 ví dụ
.venv/bin/python -c "
import json
qa_path = 'data/raw/history/history_qa_all.jsonl'
narrative_path = 'data/raw/history/vn_su_luoc_for_training_clean_v2.txt'
out_path = 'data/interim/history_narrative_qa_corpus.txt'

lines = [l.strip() for l in open(narrative_path, encoding='utf-8') if l.strip()]
with open(qa_path, encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        instr = ' '.join(obj['instruction'].split())
        out = ' '.join(obj['output'].split())
        lines.append(f'{instr} Trả lời: {out}')

open(out_path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print(len(lines), 'dòng')
"

# 2. Mã hoá + fine-tune (từ checkpoint gốc, KHÔNG phải checkpoint Cách 1)
.venv/bin/python scripts/prepare_dataset.py \
  --corpus data/interim/history_narrative_qa_corpus.txt \
  --output-dir data/processed_history_qa \
  --model artifacts/tokenizer/viwiki_bpe_8k.model

.venv/bin/python scripts/train.py \
  --data_dir data/processed_history_qa \
  --out_dir artifacts/checkpoints/gpt_8k_history_qa_finetune \
  --init_from artifacts/checkpoints/gpt_8k_research/ckpt.pt \
  --max_iters 1500 --batch_size 16 \
  --learning_rate 3e-5 --min_lr 3e-6 --warmup_iters 30 \
  --eval_interval 100 --eval_iters 30 --device cpu
```

**Kết quả thực tế** (4.403 dòng — 2.223 văn bản + 2.180 Q&A, ~640K token):
val loss giảm từ **5.04 → 2.80** sau 1.500 bước — tốt hơn rõ rệt so với
Cách 1, vì format ngắn gọn giúp mỗi ví dụ instruction+answer thường nằm
trọn trong một cửa sổ 256 token (đo được ~93% ví dụ ≤ 256 token). Model bắt
đầu phản hồi đúng cấu trúc "Hỏi → Trả lời:", đôi khi tự sinh cả bullet-point
giống văn phong `key_points` trong dữ liệu train.

> Độ chính xác *sự kiện* (tên vua, năm, ai đánh ai) vẫn còn sai lẫn — model
> chỉ 5,2M tham số, không đủ dung lượng để nhớ chi tiết lịch sử chính xác.
> Format Q&A giúp model học *cấu trúc* trả lời, không tự động đảm bảo nội
> dung đúng.

---

## Cách 3 — Fine-tune trên dữ liệu Alpaca-block đầy đủ

Dùng khi có sẵn bộ dữ liệu instruction-tuning dạng chuẩn Alpaca — mỗi ví dụ
là một khối nhiều dòng:

```
### Instruction:
{câu lệnh/câu hỏi}

### Input:
{đoạn nguồn}

### Response:
{câu trả lời}
```

Định dạng này **không** tương thích với `encode_corpus()` (hàm này mã hoá
theo từng dòng, không theo khối nhiều dòng) — cần một script mã hoá riêng
theo khối, tách trên `"\n\n### Instruction:"`, rồi bọc mỗi khối bằng
BOS/EOS. Nếu dữ liệu của bạn đã có sẵn script `prepare_instruction_bin.py`
kiểu này (ví dụ đi kèm bộ dữ liệu Q&A tải về), dùng lại nó — chỉ cần trỏ
đúng tokenizer khớp checkpoint gốc, và tự tạo `meta.json` (script loại này
thường không tạo, mà `scripts/train.py` bắt buộc phải có):

```bash
# 1. Mã hoá theo khối Alpaca (dùng script đi kèm dữ liệu, đổi --tokenizer cho khớp checkpoint)
.venv/bin/python data/raw/history/vn_su_luoc_qa_finetune/prepare_instruction_bin.py \
  --tokenizer artifacts/tokenizer/viwiki_bpe_8k.model \
  --train_txt data/raw/history/vn_su_luoc_qa_finetune/history_qa_train_corpus.txt \
  --val_txt data/raw/history/vn_su_luoc_qa_finetune/history_qa_val_corpus.txt \
  --out_dir data/processed_history_qa_v2

# 2. Tạo meta.json (bắt buộc cho scripts/train.py — script mã hoá bên ngoài không tự tạo)
.venv/bin/python scripts/make_meta.py \
  --out_dir data/processed_history_qa_v2 \
  --vocab_size 8000 \
  --tokenizer artifacts/tokenizer/viwiki_bpe_8k.model

# 3. Fine-tune từ checkpoint gốc
.venv/bin/python scripts/train.py \
  --data_dir data/processed_history_qa_v2 \
  --out_dir artifacts/checkpoints/gpt_8k_history_qa_finetune_v2 \
  --init_from artifacts/checkpoints/gpt_8k_research/ckpt.pt \
  --max_iters 1000 --batch_size 16 \
  --learning_rate 5e-5 --min_lr 5e-6 --warmup_iters 30 \
  --eval_interval 100 --eval_iters 50 --device cpu
```

Vì mỗi khối gồm cả `### Input:` (đoạn nguồn dài) nên tổng số token mỗi ví dụ
thường vượt quá 256 — sẽ bị cắt qua nhiều cửa sổ khi train (xem mục dưới).
Khi test, prompt chỉ có `### Instruction:` + `### Response:` (bỏ qua
`### Input:`) là lệch với format lúc train — model có thể có xu hướng tự
sinh thêm phần `### Input:` trước khi trả lời, thay vì trả lời ngay.

---

## Giới hạn quan trọng: block_size cố định

`GPTConfig.block_size` không chỉ là tham số huấn luyện — nó quyết định
kích thước buffer RoPE (`freqs_cis`, xem `src/ntokenizer/model.py`), buffer
này **được lưu trong checkpoint** (qua `register_buffer`, không có
`persistent=False`). Vì vậy:

- **Không thể** tăng `--block_size` khi fine-tune từ checkpoint pretrain —
  `load_state_dict` sẽ báo lỗi shape mismatch trên `freqs_cis`.
- Mọi lệnh fine-tune ở trên đều giữ `block_size=256` (giá trị pretrain).
- Ví dụ dài hơn 256 token (ví dụ Cách 3) vẫn train được — cửa sổ lấy mẫu
  ngẫu nhiên từ luồng token đã ghép (kiểu nanoGPT) chỉ thấy một phần của ví
  dụ dài, không lỗi, nhưng học kém hiệu quả hơn so với ví dụ ngắn gọn vừa
  đủ (xem so sánh Cách 1/2 vs Cách 3).

---

## Test model sau khi fine-tune

```bash
.venv/bin/python scripts/sample.py \
  --ckpt artifacts/checkpoints/<tên_checkpoint>/ckpt.pt \
  --tokenizer artifacts/tokenizer/viwiki_bpe_8k.model \
  --prompt "Gia Long là ai? Trả lời:" \
  --max_new_tokens 80 --device cpu
```

Prompt phải theo đúng "khuôn" mà corpus fine-tune đã dùng (ví dụ có cụm
`Trả lời:` hay `### Response:`) — nếu không, model quay lại hành vi
continuation thuần vì không nhận ra cue đã học.

Model càng nhỏ (ở đây 5,2M tham số) và fine-tune với càng ít bước/dữ liệu
thì output càng có xu hướng sai lệch sự kiện — dùng để demo pipeline hoạt
động, không nên kỳ vọng độ chính xác như một model lớn.

---

## Tham chiếu flag CLI

**`scripts/prepare_dataset.py`**

| Flag | Default | Ghi chú |
|---|---|---|
| `--corpus` | `data/interim/corpus.txt` | Corpus nguồn — mỗi dòng 1 đoạn văn |
| `--output-dir` | `data/processed/` | Nơi ghi `train.bin`/`val.bin`/`meta.json` |
| `--model` | `viwiki_bpe_32k.model` | Tokenizer — phải khớp checkpoint sẽ fine-tune |

**`scripts/train.py`** (chỉ liệt kê flag liên quan fine-tuning — xem đầy đủ ở [`DEVELOPMENT.md`](DEVELOPMENT.md#step-7--training-loop))

| Flag | Default | Ghi chú |
|---|---|---|
| `--init_from` | `""` | Checkpoint pretrain để khởi tạo trọng số. **Bị bỏ qua** nếu `--out_dir` đã có `ckpt.pt` sẵn (resume được ưu tiên) |
| `--data_dir` | `data/processed/` | Trỏ vào thư mục output của `prepare_dataset.py` ở bước trước |
| `--out_dir` | `artifacts/checkpoints/` | Thư mục ghi checkpoint fine-tune — **nên đặt khác** thư mục checkpoint gốc |

**`scripts/make_meta.py`** (chỉ cần khi dùng script mã hoá bên ngoài không tự ghi `meta.json`)

| Flag | Bắt buộc | Ghi chú |
|---|---|---|
| `--out_dir` | có | Thư mục đã có `train.bin`/`val.bin` |
| `--vocab_size` | có | Phải khớp checkpoint sẽ fine-tune |
| `--tokenizer` | có | Đường dẫn tokenizer đã dùng để mã hoá |

---

## Troubleshooting

**`RuntimeError: Error(s) in loading state_dict ... size mismatch`**
Tokenizer dùng để mã hoá corpus fine-tune không khớp `vocab_size` của
checkpoint. Kiểm tra lại theo hướng dẫn ở [mục Yêu cầu bắt buộc](#yêu-cầu-bắt-buộc-tokenizer-phải-khớp-checkpoint).

**`ERROR: data/processed_xxx/meta.json not found`**
Nếu bạn dùng script mã hoá riêng (không phải `scripts/prepare_dataset.py`
của repo) — ví dụ script đi kèm một bộ dữ liệu tải về — nó có thể không tự
tạo `meta.json`. Dùng `scripts/make_meta.py` theo mẫu ở [Cách 3, bước 2](#cách-3--fine-tune-trên-dữ-liệu-alpaca-block-đầy-đủ).

**Fine-tune không dùng `init_from`, val loss bắt đầu rất cao như random init**
`--out_dir` đã có sẵn `ckpt.pt` từ một lần chạy trước — `train()` ưu tiên
resume từ đó và bỏ qua `--init_from`. Đổi sang một `--out_dir` mới.

**Val loss ban đầu (step 0) cao hơn val loss cuối của checkpoint gốc**
Bình thường — checkpoint gốc được đánh giá trên tập val của corpus Wikipedia
(miền tổng quát), còn fine-tune đánh giá trên tập val của corpus mới (miền
hẹp hơn) mà model gốc chưa từng thấy. Loss cao hơn ban đầu, sau đó giảm dần
qua các bước fine-tune mới là chỉ số đáng theo dõi.
