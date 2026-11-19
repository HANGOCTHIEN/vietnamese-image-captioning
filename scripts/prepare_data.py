"""CLI điều phối chuẩn bị KTVIC; chạy được từ mọi current working directory."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Repository hiện chưa có packaging metadata; entrypoint dùng đúng src package.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from image_captioning.data.prepare_ktvic import prepare_data  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Đọc tham số, chạy pipeline và trả exit code khác 0 khi kiểm toán thất bại."""
    parser = argparse.ArgumentParser(description="Kiểm toán và chuẩn bị dữ liệu KTVIC.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=.10)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--train-json", type=Path)
    parser.add_argument("--test-json", type=Path)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        report = prepare_data(PROJECT_ROOT, **vars(args))
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("Preprocessing chưa hoàn thành: %s", exc)
        return 1
    for split, stats in report["split_statistics"]["splits"].items():
        logging.info("%s: %s ảnh, %s caption", split, stats["images"], stats["captions"])
    logging.info("Vocabulary: %s tokens", report["vocabulary_statistics"]["vocab_size"])
    logging.info("Source audit: %s; processed integrity: %s; Dataset/DataLoader: %s",
                 report["source_audit"]["status"], report["processed_integrity"]["status"],
                 report["processing_status"]["dataset_dataloader_smoke_test"])
    logging.info("Processing status: %s", report["processing_status"]["state"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
