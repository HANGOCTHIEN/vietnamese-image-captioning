import pytest
from image_captioning.data.tokenizer import VietnameseTokenizer


def test_syllable_tokenization():
    tokenizer = VietnameseTokenizer(mode="syllable")
    result = tokenizer.tokenize("Hai cầu thủ đang đá bóng!")
    assert result == ["hai", "cầu", "thủ", "đang", "đá", "bóng"]


def test_word_tokenization():
    tokenizer = VietnameseTokenizer(mode="word")
    result = tokenizer.tokenize("Hai cầu thủ đang đá bóng trên sân cỏ!")
    assert "cầu_thủ" in result
    assert "sân_cỏ" in result


def test_lowercase():
    tokenizer = VietnameseTokenizer(mode="syllable", lowercase=True)
    assert tokenizer.tokenize("VIỆT NAM") == ["việt", "nam"]


def test_empty_caption():
    tokenizer = VietnameseTokenizer(mode="syllable")
    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize("   ") == []


def test_unicode_nfc_vs_nfd():
    tokenizer = VietnameseTokenizer(mode="syllable")
    nfc_text = "hòa bình"               # Dựng sẵn (1 ký tự)
    nfd_text = "ho\u0300a bi\u0300nh"   # Tổ hợp (2 ký tự)
    assert tokenizer.tokenize(nfc_text) == tokenizer.tokenize(nfd_text)


def test_clean_existing_underscore():
    tokenizer = VietnameseTokenizer(mode="syllable")
    assert tokenizer.tokenize("Cô ấy đang đọc_sách.") == ["cô", "ấy", "đang", "đọc", "sách"]


def test_invalid_mode():
    with pytest.raises(ValueError):
        VietnameseTokenizer(mode="invalid")


def test_invalid_input():
    tokenizer = VietnameseTokenizer(mode="syllable")
    with pytest.raises(TypeError):
        tokenizer.tokenize(None)
    with pytest.raises(TypeError):
        tokenizer.detokenize("một con chó")


def test_detokenize_and_metric_text():
    tok_word = VietnameseTokenizer(mode="word")
    tokens = ["hai", "cầu_thủ", "đang", "đá", "bóng"]
    assert tok_word.detokenize(tokens) == "hai cầu thủ đang đá bóng"
    assert tok_word.to_metric_text(tokens) == "hai cầu_thủ đang đá bóng"