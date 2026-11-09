# Phần cứng — cần máy thế nào để train

*[English version](HARDWARE.md)*

Có ba mức, tuỳ mục tiêu. **Mức 1 chạy trên mọi laptop bình thường** và là
điểm bắt đầu nên chọn.

| Mức | Mục tiêu | Máy | Thời gian |
|---|---|---|---|
| **1 — Học và thử nghiệm** | Model mặc định (4 tầng, 256 chiều, ~5–19M tham số), 5.000 bước ≈ 41 triệu token, Wikipedia tiếng Việt | Laptop bất kỳ, RAM 8 GB, trống ~6 GB ổ đĩa; Mac M-series (MPS) hoặc GPU thì nhanh hơn | ~2,5–3 giờ trên CPU 4 nhân (đã đo, xem dưới); nhanh hơn trên MPS/GPU |
| 2 — Model ~100M trên Wikipedia | ~2,5 tỷ token (≈ 20 token/tham số) | 1 GPU 12–24 GB (RTX 3060 12 GB, 4070, 3090, 4090) | Không khả thi trên CPU (~60 ngày với 480 token/s đã đo); từ nửa ngày trở lên trên GPU phổ thông (chưa đo) |
| 3 — Wikipedia + báo | ≥ 10 tỷ token | 1 GPU 24 GB hoặc thuê A100/H100 | Nhiều giờ đến nhiều ngày (chưa đo với repo này) |

## Số đo

Đo trên CPU cloud 4 nhân, RAM 15 GB (2026-11-09), với code của repo (fp32,
không `torch.compile`):

| Hạng mục | Kết quả |
|---|---|
| Tốc độ train, model mặc định (5,3M tham số với tokenizer 8k) | ~6.250 token/s khi đo riêng; ~1,7–2,0 giây/bước (batch 32 × 256 token) trong `train.py`, tính cả đánh giá |
| Tốc độ train, 12 tầng / 768 chiều / vocab 32k (100M tham số) | ~480 token/s |
| `prepare_dataset.py` với corpus 200 MB (38,6 triệu token) | RAM đỉnh 603 MB, 10 giây (encode theo luồng) |
| Chạy thử mức 1 từ đầu đến cuối (corpus giả, train tokenizer riêng trước) | tokenizer trên 200 MB / 937 nghìn dòng trong 6 giây; 200 bước train trong 5,7 phút, val loss 9,02 → 4,31; `sample.py` sinh được văn bản |

Với ~1,8 giây/bước, 5.000 bước mặc định mất khoảng **2,5 giờ** trên CPU như
vậy. Mac M-series chạy cùng code trên MPS, được ghi nhận nhanh hơn CPU khoảng
2–3 lần với model kiểu nanoGPT (chưa đo ở đây).

## Mức 1 — từng bước

Yêu cầu: Python ≥ 3.10, RAM 8 GB, trống ~6 GB ổ đĩa (bản dump Wikipedia
~1,1 GB, văn bản trích xuất và corpus ~3 GB, file token vài trăm MB).

```bash
# 0. Cài đặt
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,wikiextract]"

# 1. Tải và trích xuất Wikipedia tiếng Việt (JSON lines)
wget https://dumps.wikimedia.org/viwiki/latest/viwiki-latest-pages-articles.xml.bz2 -P data/raw/
wikiextractor data/raw/viwiki-latest-pages-articles.xml.bz2 \
              --output data/interim/extracted/ --json --quiet
python scripts/build_corpus.py

# 2. Train tokenizer TRÊN CHÍNH VĂN BẢN CỦA BẠN trước: tokenizer phải học từ
#    corpus mà model sẽ train, và mọi bước sau phải dùng đúng tokenizer này.
#    Vocab 8k giữ model nhỏ và nhanh cho mức 1. --name tránh ghi đè file
#    viwiki_bpe_8k.model có sẵn trong repo (đang được git theo dõi).
python scripts/train_tokenizer_spm.py --vocab_size 8000 --name level1_bpe
#    Máy 8 GB thì giới hạn số câu SentencePiece giữ trong RAM:
#    ... --max_sentences 3000000
#    Kiểm tra: python scripts/count_tokens.py "Hà Nội là thủ đô của Việt Nam." \
#        --model artifacts/tokenizer/level1_bpe_8k.model --show-pieces

# 3. Encode (theo luồng, < 1 GB RAM)
python scripts/prepare_dataset.py --model artifacts/tokenizer/level1_bpe_8k.model

# 4. Train model mặc định (tự nhận CUDA, MPS hoặc CPU)
python scripts/train.py --out_dir artifacts/checkpoints/level1

# 5. Sinh văn bản
python scripts/sample.py --ckpt artifacts/checkpoints/level1/ckpt.pt \
    --tokenizer artifacts/tokenizer/level1_bpe_8k.model --prompt "Hà Nội là"
```

Mẹo cho mức 1:

- **Chạy thử trước** với lần train ngắn: `--max_iters 200 --eval_interval 100`
  mất vài phút và val loss phải giảm.
- **Train tiếp được**: chạy lại `train.py` với cùng `--out_dir` sẽ tiếp tục
  từ `ckpt.pt`.
- **Thiếu RAM khi train**: giảm `--batch_size` (ví dụ 16).
- Ở kích thước này, model viết tiếng Việt trôi chảy nhưng hay sai sự kiện; đó
  là bình thường (xem [why_llms_are_accurate.vi.md](why_llms_are_accurate.vi.md)).
- Mức 1 đủ cho các thí nghiệm diễn giải cơ chế trong epics (logit lens,
  entropy, grokking, tần suất sự kiện).
