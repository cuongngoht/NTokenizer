import sentencepiece as spm
import pytest
import torch

from ntokenizer.config import GPTConfig

TINY_CORPUS_LINES = [
    "Hà Nội là thủ đô của Việt Nam.",
    "Thành phố Hồ Chí Minh là thành phố lớn nhất Việt Nam.",
    "Việt Nam có nhiều cảnh đẹp thiên nhiên.",
    "Tôi muốn học lập trình Python.",
    "Ngôn ngữ tiếng Việt có nhiều dấu thanh.",
    "Sông Hồng chảy qua nhiều tỉnh thành ở miền Bắc.",
    "Chợ Bến Thành là một địa điểm du lịch nổi tiếng.",
    "Ẩm thực Việt Nam rất đa dạng và phong phú.",
    "Học sinh Việt Nam đi học vào buổi sáng.",
    "Mùa xuân ở Việt Nam có Tết Nguyên Đán.",
    "Đại học Quốc gia Hà Nội là một trường lớn.",
    "Cà phê sữa đá là thức uống phổ biến ở Việt Nam.",
    "Người Việt thường ăn cơm với các món kho, luộc, xào.",
    "Vịnh Hạ Long được UNESCO công nhận là di sản thế giới.",
    "Tiếng Việt thuộc ngữ hệ Nam Á.",
    "Miền Trung Việt Nam thường chịu ảnh hưởng của bão.",
    "Trẻ em Việt Nam rất thích chơi thả diều.",
    "Áo dài là trang phục truyền thống của phụ nữ Việt Nam.",
    "Phở là món ăn nổi tiếng của Việt Nam trên thế giới.",
    "Đồng bằng sông Cửu Long là vùng trồng lúa lớn nhất.",
]


@pytest.fixture
def tiny_gpt_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=64,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_kv_head=1,
        n_embd=16,
        dropout=0.0,
    )


@pytest.fixture(scope="session")
def tiny_sentencepiece_model(tmp_path_factory) -> str:
    """Train a throwaway SentencePiece model on a tiny embedded Vietnamese corpus."""
    tmp_dir = tmp_path_factory.mktemp("tiny_spm")
    corpus_path = tmp_dir / "corpus.txt"
    corpus_path.write_text("\n".join(TINY_CORPUS_LINES) + "\n", encoding="utf-8")

    model_prefix = str(tmp_dir / "tiny_bpe")
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=500,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        unk_id=0, unk_piece="<unk>",
        pad_id=1, pad_piece="<pad>",
        bos_id=2, bos_piece="<bos>",
        eos_id=3, eos_piece="<eos>",
    )
    return model_prefix + ".model"


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")
