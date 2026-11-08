import gzip
import json
import unicodedata

import sentencepiece as spm

from ntokenizer.cli import build_news_corpus, mix_corpora as mix_cli
from ntokenizer.dataset import encode_corpus
from ntokenizer.news import (
    clean_article,
    count_blocks,
    iter_blocks,
    iter_news_articles,
    mix_corpora,
    write_news_corpus,
)

BODY = ("Sáng nay, Hà Nội đón đợt không khí lạnh đầu mùa, nhiệt độ giảm sâu.\n"
        "Ảnh: TTXVN\n"
        "Trung tâm Dự báo khí tượng thủy văn quốc gia cho biết rét sẽ kéo dài nhiều ngày.\n"
        "Người dân được khuyến cáo giữ ấm khi ra đường vào sáng sớm và đêm muộn.")


def _article(title, body=BODY, **extra):
    return {"title": title, "content": body, **extra}


def test_clean_article_normalizes_and_drops_short_lines():
    nfd = unicodedata.normalize("NFD", "Hà Nội đón rét")
    lines = clean_article(nfd, BODY + "\n&amp; <b>Kết thúc</b> bản tin thời tiết hôm nay.")
    assert lines[0] == unicodedata.normalize("NFC", "Hà Nội đón rét")
    assert "Ảnh: TTXVN" not in lines                     # short caption dropped
    assert lines[-1] == "& Kết thúc bản tin thời tiết hôm nay."  # entity + tag cleaned
    assert all(unicodedata.is_normalized("NFC", line) for line in lines)


def test_reads_jsonl_gz_json_and_txt(tmp_path):
    jl = tmp_path / "a.jsonl"
    jl.write_text(json.dumps(_article("A")) + "\n", encoding="utf-8")
    gz = tmp_path / "b.jsonl.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"headline": "B", "text": BODY}) + "\n")
    js = tmp_path / "c.json"
    js.write_text(json.dumps([_article("C")]), encoding="utf-8")
    txt = tmp_path / "d.txt"
    txt.write_text(BODY + "\n\n" + BODY + "\n", encoding="utf-8")

    got = list(iter_news_articles([jl, gz, js, txt]))
    assert [t for t, _ in got] == ["A", "B", "C", "", ""]
    assert all(body for _, body in got)


def test_explicit_text_field(tmp_path):
    jl = tmp_path / "a.jsonl"
    jl.write_text(json.dumps({"title": "T", "noi_dung": BODY}) + "\n", encoding="utf-8")
    assert list(iter_news_articles([jl], text_field="noi_dung"))[0][1] == BODY


def test_write_news_corpus_dedups_and_filters(tmp_path):
    jl = tmp_path / "news.jsonl"
    rows = [_article("Tin 1"), _article("Tin 1"),           # syndicated duplicate
            _article("Tin 2", body="Quá ngắn để giữ lại."),   # too short
            _article("Tin 3", body=BODY + "\nThêm một đoạn khác để bài này khác bài đầu.")]
    jl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "news_corpus.txt"

    stats = write_news_corpus([jl], out)
    assert stats["articles_seen"] == 4
    assert stats["duplicates"] == 1
    assert stats["too_short"] == 1
    assert stats["articles_written"] == 2
    assert count_blocks(out) == 2                            # blank line between articles
    assert "Ảnh: TTXVN" not in out.read_text(encoding="utf-8")

    stats = write_news_corpus([jl], out, dedup=False, limit=2)
    assert stats["articles_written"] == 2 and stats["duplicates"] == 0


def test_iter_blocks_chunks_files_without_separators(tmp_path):
    f = tmp_path / "wiki.txt"
    f.write_text("\n".join(f"dòng {i}" for i in range(7)) + "\n", encoding="utf-8")
    blocks = list(iter_blocks(f, chunk_lines=3))
    assert [b.count("\n") + 1 for b in blocks] == [3, 3, 1]


def test_mix_interleaves_and_repeats(tmp_path):
    wiki = tmp_path / "wiki.txt"
    wiki.write_text("\n".join(f"wiki {i}" for i in range(100)) + "\n", encoding="utf-8")
    news = tmp_path / "news.txt"
    news.write_text("".join(f"news {i}\n\n" for i in range(100)), encoding="utf-8")
    out = tmp_path / "mix.txt"

    written = mix_corpora([wiki, news], out, repeats=[2, 1], chunk_lines=10, seed=1)
    assert written == {str(wiki): 20, str(news): 100}
    text = out.read_text(encoding="utf-8")
    assert text.count("wiki 0\n") == 2                         # upsampled twice
    # The validation tail (last 10%) must contain both sources.
    lines = [line for line in text.splitlines() if line]
    tail = lines[int(len(lines) * 0.9):]
    assert any(line.startswith("wiki") for line in tail)
    assert any(line.startswith("news") for line in tail)
    # Same seed gives the same file.
    out2 = tmp_path / "mix2.txt"
    mix_corpora([wiki, news], out2, repeats=[2, 1], chunk_lines=10, seed=1)
    assert out2.read_text(encoding="utf-8") == text


def test_end_to_end_cli_output_is_trainable(tmp_path, tiny_sentencepiece_model):
    jl = tmp_path / "news.jsonl"
    jl.write_text("\n".join(json.dumps(_article(f"Tin {i}", body=BODY + f"\nĐoạn riêng số {i} của bài.")) for i in range(5)) + "\n",
                  encoding="utf-8")
    news_out = tmp_path / "news_corpus.txt"
    build_news_corpus.main(["--input", str(jl), "--output", str(news_out)])

    wiki = tmp_path / "corpus.txt"
    wiki.write_text("Hà Nội là thủ đô của Việt Nam.\nTôi muốn học lập trình Python.\n", encoding="utf-8")
    mix_out = tmp_path / "corpus_mix.txt"
    mix_cli.main(["--input", str(wiki), str(news_out), "--repeat", "2", "1", "--output", str(mix_out)])

    sp = spm.SentencePieceProcessor(model_file=tiny_sentencepiece_model)
    ids = encode_corpus(mix_out, sp, sp.eos_id())
    n_lines = sum(1 for line in mix_out.read_text(encoding="utf-8").splitlines() if line.strip())
    assert ids.count(sp.eos_id()) == n_lines                 # blank separators are skipped


def test_credit_filter_keeps_normal_sentences():
    body = ("Ảnh hưởng của bão số 3 khiến nhiều tuyến đường bị ngập sâu.\n"
            "Ảnh minh họa: Internet\n"
            "Nguồn: VTV\n"
            "Video: Cảnh ngập tại trung tâm thành phố")
    lines = clean_article("", body)
    assert lines == ["Ảnh hưởng của bão số 3 khiến nhiều tuyến đường bị ngập sâu."]
