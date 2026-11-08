import sentencepiece as spm

from ntokenizer.cli import count_tokens as cli
from ntokenizer.tokenizer import count_tokens, encode_prompt


def _load(path):
    return spm.SentencePieceProcessor(model_file=path)


def test_count_matches_prompt_encoding(tiny_sentencepiece_model):
    sp = _load(tiny_sentencepiece_model)
    text = "Hà Nội là thủ đô của Việt Nam."
    stats = count_tokens(sp, text)
    # The count must equal what the model actually receives as a prompt.
    assert stats.n_tokens == len(encode_prompt(sp, text))
    assert len(stats.pieces) == stats.n_tokens
    assert stats.n_words == 8   # Hà Nội là thủ đô của Việt Nam.
    assert stats.n_chars == len(text)
    assert stats.n_bytes == len(text.encode("utf-8"))


def test_fertility_and_compression(tiny_sentencepiece_model):
    sp = _load(tiny_sentencepiece_model)
    stats = count_tokens(sp, "Việt Nam có nhiều cảnh đẹp thiên nhiên.")
    assert stats.tokens_per_word >= 1.0
    assert stats.bytes_per_token > 0


def test_unseen_text_uses_byte_fallback_not_unk(tiny_sentencepiece_model):
    sp = _load(tiny_sentencepiece_model)
    # Characters never seen in the tiny corpus still encode via byte fallback.
    stats = count_tokens(sp, "Ωμέγα ☃")
    assert stats.n_tokens > 0
    assert stats.n_unk == 0


def test_empty_text(tiny_sentencepiece_model):
    sp = _load(tiny_sentencepiece_model)
    stats = count_tokens(sp, "")
    assert stats.n_tokens == 0
    assert stats.tokens_per_word == 0.0
    assert stats.bytes_per_token == 0.0


def test_cli_reports_count_and_context_fit(tiny_sentencepiece_model, capsys):
    cli.main(["Hà Nội là thủ đô của Việt Nam.", "--model", tiny_sentencepiece_model,
              "--block-size", "4", "--show-pieces"])
    out = capsys.readouterr().out
    assert "Tokens          :" in out
    assert "exceeds 4" in out
    assert "Pieces          :" in out


def test_cli_reads_file(tiny_sentencepiece_model, tmp_path, capsys):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Tôi muốn học lập trình Python.", encoding="utf-8")
    cli.main(["--file", str(prompt), "--model", tiny_sentencepiece_model, "--block-size", "512"])
    out = capsys.readouterr().out
    assert "fits in 512" in out
