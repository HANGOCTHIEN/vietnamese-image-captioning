"""Tokenizer âm tiết và vocabulary deterministic, bảo toàn văn bản tiếng Việt."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


def normalize_caption(text: str) -> str:
    """Chuẩn hóa NFC, BOM, control và khoảng trắng; giữ hoa/thường, dấu câu.

    Khoảng trắng Unicode được gom thành một dấu cách. Control Cc không phải
    khoảng trắng và BOM bị loại; surrogate không hợp lệ gây lỗi rõ ràng.
    """
    if not isinstance(text, str):
        raise TypeError("caption phải là str.")
    if any(unicodedata.category(char) == "Cs" for char in text):
        raise ValueError("caption chứa Unicode surrogate không hợp lệ.")
    text = unicodedata.normalize("NFC", text).replace("\ufeff", "")
    text = "".join(
        char for char in text
        if char.isspace() or unicodedata.category(char) != "Cc"
    )
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    """Tách theo khoảng trắng, giữ dấu câu gắn với âm tiết để ghép lại chính xác.

    Đây là syllable-level baseline, không phải Vietnamese word segmentation.
    Không tách riêng dấu câu giúp decode khôi phục đúng caption đã chuẩn hóa.
    """
    return normalize_caption(text).split()


class VietnameseTokenizer:
    """Vocabulary chỉ được xây từ caption train, với special IDs cố định."""

    special_tokens = {"pad": "<pad>", "bos": "<bos>",
                      "eos": "<eos>", "unk": "<unk>"}
    pad_id, bos_id, eos_id, unk_id = 0, 1, 2, 3

    def __init__(self, min_freq: int = 1) -> None:
        if type(min_freq) is not int or min_freq < 1:
            raise ValueError("min_freq phải là số nguyên dương.")
        self.min_freq = min_freq
        self.token_to_id = {
            token: index for index, token in enumerate(self.special_tokens.values())
        }
        self.max_train_token_length = 0

    def __len__(self) -> int:
        return len(self.token_to_id)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Trả các token âm tiết từ caption chuẩn, không dùng segment_caption."""
        return tokenize(text)

    def build_vocab(self, train_captions: Iterable[str]) -> "VietnameseTokenizer":
        """Đếm token train; sắp theo (-frequency, token) rồi gán ID ổn định."""
        counts: Counter[str] = Counter()
        maximum = 0
        for caption in train_captions:
            tokens = tokenize(caption)
            if set(tokens).intersection(self.special_tokens.values()):
                raise ValueError("Token văn bản trùng với special token dành riêng.")
            counts.update(tokens)
            maximum = max(maximum, len(tokens))
        ordered = sorted(counts, key=lambda token: (-counts[token], token))
        self.token_to_id = {
            token: index for index, token in enumerate(self.special_tokens.values())
        }
        for token in ordered:
            if counts[token] >= self.min_freq:
                self.token_to_id[token] = len(self.token_to_id)
        self.max_train_token_length = maximum
        return self

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Mã hóa caption, dùng UNK cho token lạ và tùy chọn thêm BOS/EOS."""
        tokens = tokenize(text)
        if set(tokens).intersection(self.special_tokens.values()):
            raise ValueError("Caption chứa special token dành riêng.")
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        return [self.bos_id, *ids, self.eos_id] if add_special_tokens else ids

    def decode(
        self, token_ids: Sequence[int], skip_special_tokens: bool = True
    ) -> str:
        """Ghép token bằng dấu cách; giữ UNK hiển thị, bỏ PAD/BOS/EOS khi yêu cầu."""
        id_to_token = {index: token for token, index in self.token_to_id.items()}
        tokens = []
        for index in token_ids:
            if type(index) is not int or index not in id_to_token:
                raise ValueError(f"Token ID không hợp lệ: {index!r}.")
            if skip_special_tokens and index in (self.pad_id, self.bos_id, self.eos_id):
                continue
            tokens.append(id_to_token[index])
        return " ".join(tokens)

    def to_dict(self) -> dict:
        """Lấy metadata và bảng ID để lưu JSON UTF-8 và kiểm toán."""
        return {
            "token_to_id": dict(self.token_to_id),
            "special_tokens": dict(self.special_tokens),
            "tokenization": {
                "language": "vi", "strategy": "syllable",
                "punctuation": "preserve_attached", "lowercase": False,
                "unicode_normalization": "NFC", "min_freq": self.min_freq,
                "training_source": "train.jsonl",
            },
            "vocab_size": len(self),
            "max_train_token_length": self.max_train_token_length,
        }

    def save_vocab(self, path: str | Path) -> None:
        """Lưu vocabulary; không âm thầm ghi đè một vocabulary khác đã tồn tại."""
        path = Path(path)
        content = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if path.exists():
            if path.read_bytes() != content.encode("utf-8"):
                raise FileExistsError(f"Vocabulary đã tồn tại với nội dung khác: {path.name}")
            return
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)

    @classmethod
    def from_dict(cls, data: dict) -> "VietnameseTokenizer":
        """Nạp và kiểm tra schema vocabulary, ID liên tiếp và chính sách tokenizer."""
        if not isinstance(data, dict):
            raise ValueError("Vocabulary phải là JSON object.")
        config = data.get("tokenization", {})
        if (config.get("strategy") != "syllable"
                or config.get("unicode_normalization") != "NFC"
                or config.get("punctuation") != "preserve_attached"
                or config.get("lowercase") is not False):
            raise ValueError("Chính sách tokenization không tương thích.")
        vocab = cls(min_freq=config.get("min_freq", 1))
        mapping = data.get("token_to_id")
        if not isinstance(mapping, dict) or not all(
            isinstance(token, str) and token and not re.search(r"\s", token)
            and type(index) is int and index >= 0
            for token, index in mapping.items()
        ):
            raise ValueError("token_to_id không hợp lệ.")
        if (len(set(mapping.values())) != len(mapping)
                or set(mapping.values()) != set(range(len(mapping)))
                or data.get("vocab_size") != len(mapping)
                or data.get("special_tokens") != cls.special_tokens):
            raise ValueError("Vocabulary IDs hoặc metadata không hợp lệ.")
        for index, token in enumerate(cls.special_tokens.values()):
            if mapping.get(token) != index:
                raise ValueError(f"Special token {token} phải có ID {index}.")
        vocab.token_to_id = dict(mapping)
        maximum = data.get("max_train_token_length", 0)
        if type(maximum) is not int or maximum < 0:
            raise ValueError("max_train_token_length không hợp lệ.")
        vocab.max_train_token_length = maximum
        return vocab

    @classmethod
    def load_vocab(cls, path: str | Path) -> "VietnameseTokenizer":
        """Nạp vocabulary từ file JSON UTF-8, không xây lại từ val/test."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
