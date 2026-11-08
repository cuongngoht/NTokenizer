import sentencepiece as spm

from ntokenizer.cli import train_tokenizer as cli

from conftest import TINY_CORPUS_LINES


def test_default_uses_every_line():
    params = cli.build_train_params(8000, max_sentences=0,
                                    max_sentence_length=16_384, corpus_lines=5_000_000)
    assert params["input_sentence_size"] == 0          # 0 = no sampling cap in SentencePiece
    assert params["max_sentence_length"] == 16_384
    assert params["train_extremely_large_corpus"] is False


def test_large_corpus_mode_switches_on():
    params = cli.build_train_params(32000, max_sentences=0,
                                    max_sentence_length=16_384, corpus_lines=50_000_000)
    assert params["train_extremely_large_corpus"] is True


def test_cap_limits_large_corpus_mode():
    params = cli.build_train_params(32000, max_sentences=1_000_000,
                                    max_sentence_length=16_384, corpus_lines=50_000_000)
    assert params["input_sentence_size"] == 1_000_000
    assert params["train_extremely_large_corpus"] is False


def test_count_lines_skips_blank_lines(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("một\n\nhai\n", encoding="utf-8")
    b = tmp_path / "b.txt"
    b.write_text("ba\n   \n", encoding="utf-8")
    assert cli.count_lines([a, b]) == 3


def test_trains_on_multiple_inputs(tmp_path, capsys):
    half = len(TINY_CORPUS_LINES) // 2
    wiki = tmp_path / "wiki.txt"
    web = tmp_path / "web.txt"
    wiki.write_text("\n".join(TINY_CORPUS_LINES[:half]) + "\n", encoding="utf-8")
    # A line longer than SentencePiece's old 4192-byte default must not be dropped.
    long_line = " ".join(["Sông Cửu Long chảy ra biển."] * 200)
    web.write_text("\n".join(TINY_CORPUS_LINES[half:] + [long_line]) + "\n", encoding="utf-8")

    cli.main(["--input", str(wiki), str(web), "--output_dir", str(tmp_path / "tok"),
              "--name", "mix_bpe", "--vocab_size", "500"])

    model = tmp_path / "tok" / "mix_bpe_0k.model"
    assert model.exists()
    out = capsys.readouterr().out
    assert "100% of the corpus" in out
    sp = spm.SentencePieceProcessor(model_file=str(model))
    text = "Hà Nội là thủ đô của Việt Nam."
    assert sp.decode(sp.encode(text)) == text
