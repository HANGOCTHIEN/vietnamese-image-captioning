import re
import unicodedata
from typing import List, Sequence

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None


class VietnameseTokenizer:
    """
    Bộ chuẩn hóa và tách từ tiếng Việt cho bài toán Image Captioning.

    Modes:
        - syllable: Tách theo âm tiết, sử dụng khoảng trắng.
        - word: Tách từ tiếng Việt bằng PyVi.

    Ví dụ:
        "Hai cầu thủ đang đá bóng"

        syllable:
            ["hai", "cầu", "thủ", "đang", "đá", "bóng"]

        word:
            ["hai", "cầu_thủ", "đang", "đá", "bóng"]
    """

    VALID_MODES = {"syllable", "word"}

    # Giữ lại chữ Unicode, chữ số và khoảng trắng.
    # Các dấu câu và ký tự đặc biệt được thay bằng khoảng trắng.
    PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)

    WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(
        self,
        mode: str = "syllable",
        lowercase: bool = True
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Mode '{mode}' không hợp lệ. "
                f"Các mode được hỗ trợ: {self.VALID_MODES}"
            )

        if mode == "word" and ViTokenizer is None:
            raise ImportError(
                "Chế độ 'word' yêu cầu thư viện PyVi. "
                "Cài đặt bằng lệnh: pip install pyvi"
            )

        self.mode = mode
        self.lowercase = lowercase

    def clean_text(self, text: str) -> str:
        """
        Chuẩn hóa caption:
            1. Kiểm tra kiểu dữ liệu.
            2. Chuẩn hóa Unicode NFC.
            3. Chuyển thành chữ thường nếu được yêu cầu.
            4. Thay dấu gạch dưới bằng khoảng trắng.
            5. Loại bỏ dấu câu và ký tự đặc biệt.
            6. Chuẩn hóa khoảng trắng.
        """

        if not isinstance(text, str):
            raise TypeError(
                f"Caption phải có kiểu str, nhận được: {type(text).__name__}"
            )

        # Unicode normalization
        text = unicodedata.normalize("NFC", text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Xử lý dấu gạch dưới trước khi tokenization
        text = text.replace("_", " ")

        # Loại bỏ dấu câu và ký tự đặc biệt
        text = self.PUNCTUATION_PATTERN.sub(" ", text)

        # Chuẩn hóa khoảng trắng
        text = self.WHITESPACE_PATTERN.sub(" ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Chuyển caption thành danh sách token.
        """

        cleaned_text = self.clean_text(text)

        if not cleaned_text:
            return []

        if self.mode == "syllable":
            return cleaned_text.split()

        # Word-level tokenization bằng PyVi
        segmented_text = ViTokenizer.tokenize(cleaned_text)

        # Chuẩn hóa khoảng trắng đầu ra của PyVi
        segmented_text = self.WHITESPACE_PATTERN.sub(
            " ", segmented_text
        ).strip()

        return segmented_text.split() if segmented_text else []

    def detokenize(self, tokens: Sequence[str]) -> str:
        """
        Ghép danh sách token thành câu để hiển thị hoặc đánh giá.

        Lưu ý:
            - Không khôi phục dấu câu ban đầu.
            - Không khôi phục chữ hoa ban đầu.
            - Với word-level, dấu '_' được chuyển thành khoảng trắng.
        """

        if isinstance(tokens, str):
            raise TypeError(
                "Đầu vào detokenize phải là danh sách token, không phải str."
            )

        if not isinstance(tokens, (list, tuple)):
            raise TypeError(
                "tokens phải là list hoặc tuple."
            )

        if not all(isinstance(token, str) for token in tokens):
            raise TypeError(
                "Tất cả phần tử trong tokens phải có kiểu str."
            )

        if not tokens:
            return ""

        sentence = " ".join(tokens)

        if self.mode == "word":
            sentence = sentence.replace("_", " ")

        return sentence

    def to_metric_text(self, tokens: Sequence[str]) -> str:
        if isinstance(tokens, str):
            raise TypeError("tokens phải là danh sách token, không phải str.")

        if not isinstance(tokens, (list, tuple)):
            raise TypeError("tokens phải là list hoặc tuple.")

        if not all(isinstance(token, str) for token in tokens):
            raise TypeError("Tất cả phần tử trong tokens phải có kiểu str.")

        return " ".join(tokens)

    def get_config(self) -> dict:
        """
        Trả về cấu hình để lưu cùng checkpoint.
        """

        return {
            "mode": self.mode,
            "lowercase": self.lowercase
        }

    def __call__(self, text: str) -> List[str]:
        """Cho phép gọi trực tiếp tokenizer(text)."""
        return self.tokenize(text)

    def __repr__(self) -> str:
        return (
            f"VietnameseTokenizer("
            f"mode='{self.mode}', "
            f"lowercase={self.lowercase})"
        )
    

if __name__ == "__main__":

    sample_sentences = [
        "Hai cầu thủ đang tranh bóng quyết liệt trên sân cỏ!",
        "Một người phụ nữ đội nón lá, mặc áo dài đang đi bộ trên phố.",
        "Chú chó màu nâu đang chạy nhảy ngoài công viên (buổi sáng).",
        "   ",
        "VIỆT NAM ĐẸP QUÁ!!!",
        "Cô ấy đang đọc_sách."
    ]

    for mode in ["syllable", "word"]:

        print(f"\n{'=' * 20} {mode.upper()} {'=' * 20}")

        tokenizer = VietnameseTokenizer(
            mode=mode,
            lowercase=True
        )

        for sentence in sample_sentences:

            tokens = tokenizer.tokenize(sentence)
            restored = tokenizer.detokenize(tokens)

            print(f"Original  : {sentence}")
            print(f"Tokens    : {tokens}")
            print(f"Length    : {len(tokens)}")
            print(f"Restored  : {restored}")
            print("-" * 60)