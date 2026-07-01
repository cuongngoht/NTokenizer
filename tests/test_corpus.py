import json

from ntokenizer.corpus import clean_text, is_noise_line, iter_articles, normalize_line


def test_normalize_line_strips_html_entities():
    assert normalize_line("A &amp; B") == "A & B"


def test_normalize_line_collapses_multispace():
    assert normalize_line("A     B    C") == "A B C"


def test_normalize_line_strips_inline_tags():
    assert normalize_line("A <ref>note</ref> B") == "A note B"


def test_is_noise_line_short_line():
    assert is_noise_line("short", min_len=10) is True


def test_is_noise_line_long_enough_line():
    assert is_noise_line("This is a long enough sentence to keep.", min_len=10) is False


def test_is_noise_line_section_header():
    assert is_noise_line("History.") is True


def test_is_noise_line_wikimedia_commons_stub():
    assert is_noise_line("Wikimedia Commons") is True


def test_clean_text_filters_and_keeps_valid_lines():
    text = "\n".join([
        "short",
        "History.",
        "This is a long enough sentence to keep as valid content.",
        "",
        "Another perfectly fine sentence goes right here for testing.",
    ])

    result = clean_text(text, min_len=10)

    assert result == [
        "This is a long enough sentence to keep as valid content.",
        "Another perfectly fine sentence goes right here for testing.",
    ]


def test_iter_articles_skips_malformed_json(tmp_path):
    file_path = tmp_path / "wiki_00"
    file_path.write_text(
        json.dumps({"id": "1", "title": "A", "text": "hello"}) + "\n"
        "not valid json\n"
        + json.dumps({"id": "2", "title": "B", "text": "world"}) + "\n",
        encoding="utf-8",
    )
    stats = {"json_errors": 0}

    articles = list(iter_articles(file_path, stats))

    assert stats["json_errors"] == 1
    assert [a[2] for a in articles] == ["hello", "world"]
