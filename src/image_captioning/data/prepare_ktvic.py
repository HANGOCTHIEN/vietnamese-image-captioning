"""Chuẩn bị KTVIC: kiểm toán nguồn, ảnh, caption, chia tập và xuất JSONL."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import shutil
import stat
import statistics
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable

from PIL import Image

from .tokenizer_vi import VietnameseTokenizer, normalize_caption, tokenize


LOGGER = logging.getLogger(__name__)
DATASET_URL = "https://huggingface.co/datasets/ThucPD/KTVIC"
RAW_NAMES = (
    "ktvic-train-images.zip", "ktvic-test-images.zip",
    "ktvic-train_data.json", "ktvic-test_data.json",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DataValidationError(ValueError):
    """Dữ liệu không đạt kiểm toán; details giữ ID và số liệu phục vụ sửa nguồn."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def sha256_file(path: str | Path) -> str:
    """Tính SHA-256 theo khối, không đọc cả ZIP vào RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _within(path: Path, root: Path) -> Path:
    """Resolve đường dẫn và từ chối thoát khỏi root, kể cả qua symlink."""
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise DataValidationError("Đường dẫn nằm ngoài project root.")
    return resolved


def _valid_filename(filename: object) -> bool:
    return (
        isinstance(filename, str) and bool(filename.strip())
        and filename not in (".", "..")
        and not any(char in filename for char in '/\\:<>"|?*')
        and not filename.endswith((" ", "."))
        and not any(unicodedata.category(char).startswith("C") for char in filename)
        and filename.split(".")[0].upper() not in {
            "CON", "PRN", "AUX", "NUL", *{f"COM{i}" for i in range(1, 10)},
            *{f"LPT{i}" for i in range(1, 10)},
        }
    )


def load_annotations(path: str | Path, original_split: str) -> list[dict]:
    """Đọc schema COCO và join bằng image_id, giữ mọi caption cùng ID gốc.

    ID trùng, orphan annotation, caption rỗng/Unicode lỗi và filename không an
    toàn đều gây lỗi; không tự sửa ID hoặc loại mẫu để đạt số lượng mục tiêu.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"Không đọc được raw JSON {Path(path).name}: {exc}") from exc
    if not isinstance(data, dict) or not all(
        isinstance(data.get(key), list) for key in ("images", "annotations")
    ):
        raise DataValidationError("Raw JSON cần hai danh sách images và annotations.")
    images: dict[int, dict] = {}
    filenames: set[str] = set()
    for image in data["images"]:
        if not isinstance(image, dict) or type(image.get("id")) is not int or image["id"] < 0:
            raise DataValidationError("images[].id phải là số nguyên không âm.")
        image_id, filename = image["id"], image.get("filename")
        if image_id in images:
            raise DataValidationError("Trùng image_id trong raw JSON.", {"duplicate_image_ids": [image_id]})
        if not _valid_filename(filename) or Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
            raise DataValidationError("images[].filename không hợp lệ.", {"image_id": image_id})
        if filename.casefold() in filenames:
            raise DataValidationError("Filename trùng hoặc xung đột hoa/thường.", {"filename": filename})
        filenames.add(filename.casefold())
        images[image_id] = {
            "image_id": image_id, "image_filename": filename,
            "original_split": original_split, "_annotations": [],
        }
    annotation_ids: set[int] = set()
    orphans = []
    for annotation in data["annotations"]:
        if not isinstance(annotation, dict):
            raise DataValidationError("annotations[] phải là object.")
        annotation_id, image_id = annotation.get("id"), annotation.get("image_id")
        if type(annotation_id) is not int or annotation_id < 0:
            raise DataValidationError("annotations[].id phải là số nguyên không âm.")
        if annotation_id in annotation_ids:
            raise DataValidationError("Trùng annotation_id.", {"annotation_id": annotation_id})
        annotation_ids.add(annotation_id)
        if type(image_id) is not int or image_id not in images:
            orphans.append(annotation_id)
            continue
        if not isinstance(annotation.get("caption"), str):
            raise DataValidationError("caption phải là chuỗi.", {"annotation_id": annotation_id})
        if "segment_caption" in annotation and not isinstance(annotation["segment_caption"], str):
            raise DataValidationError("segment_caption phải là chuỗi.", {"annotation_id": annotation_id})
        try:
            clean = normalize_caption(annotation["caption"])
            if "segment_caption" in annotation:
                normalize_caption(annotation["segment_caption"])
        except ValueError as exc:
            raise DataValidationError(str(exc), {"unicode_error_annotation_ids": [annotation_id]}) from exc
        if not clean:
            raise DataValidationError("Caption rỗng sau normalization.", {"empty_caption_annotation_ids": [annotation_id]})
        images[image_id]["_annotations"].append({**annotation, "clean_caption": clean})
    if orphans:
        raise DataValidationError("Annotation không có ảnh tương ứng.", {"orphan_annotation_ids": orphans})
    missing = [image_id for image_id, image in images.items() if not image["_annotations"]]
    if missing or not images:
        raise DataValidationError("Raw JSON có ảnh thiếu caption hoặc không có ảnh.", {"missing_caption_image_ids": missing})
    records = []
    for image_id in sorted(images):
        image = images[image_id]
        annotations = sorted(image["_annotations"], key=lambda item: item["id"])
        image["_annotations"] = annotations
        image["captions"] = [item["clean_caption"] for item in annotations]
        image["annotation_ids"] = [item["id"] for item in annotations]
        records.append(image)
    return records


def inspect_zip(path: str | Path, records: list[dict]) -> tuple[dict[str, str], dict]:
    """Kiểm tra mọi tên thành viên; nối basename chính xác, không đoán prefix ZIP."""
    expected = {row["image_filename"] for row in records}
    members: dict[str, str] = {}
    prefixes: set[str] = set()
    unreferenced = []
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                name = member.orig_filename
                normalized = name.replace("\\", "/")
                parts = PurePosixPath(normalized).parts
                if (PurePosixPath(normalized).is_absolute()
                        or PureWindowsPath(name).drive
                        or ".." in parts or any(":" in part for part in parts)
                        or "\x00" in name
                        or stat.S_ISLNK(member.external_attr >> 16)):
                    raise DataValidationError("ZIP chứa đường dẫn hoặc symlink không an toàn.", {"member": name})
                if member.is_dir():
                    continue
                basename = parts[-1] if parts else ""
                if not _valid_filename(basename):
                    raise DataValidationError("ZIP chứa filename không an toàn.", {"member": name})
                if basename not in expected:
                    unreferenced.append(name)
                    continue
                if basename in members:
                    raise DataValidationError("Nhiều ZIP members trùng filename được tham chiếu.", {"filename": basename})
                if member.flag_bits & 1:
                    raise DataValidationError("Không hỗ trợ ZIP member được mã hóa.")
                members[basename] = name
                prefixes.add("/".join(parts[:-1]))
    except (OSError, zipfile.BadZipFile) as exc:
        raise DataValidationError(f"ZIP không hợp lệ: {Path(path).name}.") from exc
    missing = sorted(expected - members.keys())
    if missing:
        raise DataValidationError("ZIP thiếu ảnh được annotation tham chiếu.", {"missing_filenames": missing})
    return members, {"referenced_members": len(members), "prefixes": sorted(prefixes),
                     "unreferenced_members": sorted(unreferenced), "missing_filenames": missing}


def _image_info(path: Path) -> dict:
    """Decode toàn ảnh và thử RGB, ghi format/mode/kích thước gốc."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            if min(image.size) <= 0 or image.format not in {"JPEG", "PNG"}:
                raise ValueError("Kích thước hoặc định dạng ảnh không hợp lệ.")
            info = {"width": image.width, "height": image.height,
                    "format": image.format, "mode": image.mode}
            image.convert("RGB").load()
        return info
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise DataValidationError("Ảnh hỏng hoặc không thể decode/RGB.", {"corrupted_filenames": [path.name]}) from exc


def extract_images(
    zip_path: str | Path, members: dict[str, str], destination: Path,
    *, verify_only: bool = False,
) -> None:
    """Giải nén phẳng ảnh tham chiếu, kiểm CRC; ảnh đã có phải khớp ZIP và decode.

    Không ghi đè ảnh tồn tại khác byte. File mới được ghi tạm rồi replace sau
    khi CRC và Pillow hợp lệ; verify_only không tạo thư mục hoặc ghi file.
    """
    destination = Path(destination)
    if not verify_only:
        destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for index, filename in enumerate(sorted(members), 1):
            if index == 1 or index % 500 == 0:
                LOGGER.info("ZIP %s: %s/%s ảnh", Path(zip_path).name, index, len(members))
            target = _within(destination / filename, destination)
            member = archive.getinfo(members[filename])
            archive_digest = hashlib.sha256()
            # Đọc hết member để zipfile kiểm tra CRC, kể cả khi ảnh đã tồn tại.
            if target.exists():
                _image_info(target)
                with archive.open(member) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        archive_digest.update(block)
                if target.stat().st_size != member.file_size or sha256_file(target) != archive_digest.hexdigest():
                    raise DataValidationError("Ảnh tồn tại khác byte trong ZIP; không ghi đè.", {"filename": filename})
                continue
            if verify_only:
                raise DataValidationError("Ảnh đã giải nén bị thiếu.", {"missing_filenames": [filename]})
            temporary = _within(target.with_name(target.name + ".part"), destination)
            if temporary.exists():
                raise DataValidationError("Có file .part từ lần chạy trước; cần kiểm tra trước khi tiếp tục.")
            try:
                with archive.open(member) as source, temporary.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                _image_info(temporary)
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()


def audit_images(records: list[dict], directory: Path, project_root: Path) -> dict:
    """Kiểm tra ảnh, tính SHA-256 và đưa image_path tương đối vào metadata."""
    modes: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    hash_groups: dict[str, list[int]] = defaultdict(list)
    widths, heights = [], []
    for index, record in enumerate(records, 1):
        if index == 1 or index % 500 == 0:
            LOGGER.info("Kiểm toán %s: %s/%s ảnh", directory.name, index, len(records))
        path = _within(directory / record["image_filename"], project_root)
        if not path.is_file():
            raise DataValidationError("Thiếu ảnh được tham chiếu.", {"missing_filenames": [record["image_filename"]]})
        info = _image_info(path)
        record["_sha256"] = sha256_file(path)
        record["image_path"] = path.relative_to(project_root).as_posix()
        hash_groups[record["_sha256"]].append(record["image_id"])
        modes[info["mode"]] += 1
        formats[info["format"]] += 1
        dimensions[f'{info["width"]}x{info["height"]}'] += 1
        widths.append(info["width"])
        heights.append(info["height"])
    return {
        "images_existing": len(records), "images_missing": 0,
        "images_corrupted": 0, "images_unreadable": 0, "images_unmatched": 0,
        "modes": dict(sorted(modes.items())), "formats": dict(sorted(formats.items())),
        "dimensions": dict(sorted(dimensions.items())),
        "width": distribution(widths), "height": distribution(heights),
        "duplicate_image_ids": [], "duplicate_filenames": [],
        "duplicate_sha256_groups": {
            digest: sorted(ids) for digest, ids in sorted(hash_groups.items()) if len(ids) > 1
        },
        "sha256_by_image_id": {str(row["image_id"]): row["_sha256"] for row in records},
    }


def distribution(values: Iterable[int]) -> dict:
    """Thống kê độ dài với percentile nội suy tuyến tính, không cần NumPy."""
    numbers = sorted(values)
    if not numbers:
        return {key: None for key in ("min", "mean", "median", "max", "p90", "p95", "p99")}
    def percentile(fraction: float) -> float:
        position = (len(numbers) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return numbers[low] + (numbers[high] - numbers[low]) * (position - low)
    return {"min": numbers[0], "mean": statistics.mean(numbers),
            "median": statistics.median(numbers), "max": numbers[-1],
            "p90": percentile(.90), "p95": percentile(.95), "p99": percentile(.99)}


def audit_captions(records: list[dict]) -> dict:
    """Báo cáo trùng nguyên văn, control, normalization và segment khác nội dung."""
    duplicates, mismatches, controls, changed, non_nfc = [], [], [], [], []
    lengths = []
    annotation_count = 0
    for row in records:
        groups: dict[str, list[int]] = defaultdict(list)
        for item in row["_annotations"]:
            annotation_count += 1
            raw, annotation_id = item["caption"], item["id"]
            groups[raw].append(annotation_id)
            lengths.append(len(item["clean_caption"]))
            if raw != item["clean_caption"]:
                changed.append(annotation_id)
            if raw != unicodedata.normalize("NFC", raw):
                non_nfc.append(annotation_id)
            if any(unicodedata.category(char) == "Cc" and not char.isspace() for char in raw):
                controls.append(annotation_id)
            segmented = item.get("segment_caption")
            if segmented is not None and normalize_caption(segmented.replace("_", " ")) != item["clean_caption"]:
                mismatches.append({"image_id": row["image_id"], "annotation_id": annotation_id,
                                   "caption": raw, "segment_caption": segmented})
        for ids in groups.values():
            if len(ids) > 1:
                duplicates.append({"image_id": row["image_id"], "annotation_ids": ids,
                                   "excess_records": len(ids) - 1})
    return {
        "empty_caption_annotation_ids": [], "unicode_error_annotation_ids": [],
        "non_nfc_annotation_ids": non_nfc, "control_character_annotation_ids": controls,
        "normalized_annotation_ids": changed,
        "duplicate_caption_groups": duplicates,
        "duplicate_caption_excess_records": sum(group["excess_records"] for group in duplicates),
        "segment_caption_mismatches": mismatches,
        "segment_caption_mismatch_count": len(mismatches),
        "character_lengths": distribution(lengths),
        "captions_retained": annotation_count, "captions_removed": 0, "removed_annotations": [],
    }


def split_records(
    records: list[dict], val_ratio: float = .10, seed: int = 42
) -> tuple[list[dict], list[dict], dict]:
    """Chia nhóm SHA-256, dùng subset-sum để gần tỷ lệ validation nhất có thể.

    Nhóm và ID được sắp trước shuffle. Mỗi tổng khả thi lưu một tiền nhiệm,
    không dùng test để chọn seed/tỷ lệ và không xóa các ảnh cùng nội dung.
    """
    if not 0 < val_ratio < 1 or len(records) < 2:
        raise DataValidationError("Cần ít nhất hai ảnh train và 0 < val_ratio < 1.")
    groups: dict[str, list[int]] = defaultdict(list)
    for row in sorted(records, key=lambda row: row["image_id"]):
        groups[row["_sha256"]].append(row["image_id"])
    ordered = sorted(groups.values(), key=lambda ids: tuple(ids))
    random.Random(seed).shuffle(ordered)
    # Snapshot theo từng nhóm tránh sử dụng cùng một nhóm nhiều lần.
    reachable: dict[int, tuple[int, int] | None] = {0: None}
    for index, group in enumerate(ordered):
        for total in sorted(tuple(reachable), reverse=True):
            candidate = total + len(group)
            if candidate not in reachable:
                reachable[candidate] = (total, index)
    choices = [total for total in reachable if 0 < total < len(records)]
    if not choices:
        raise DataValidationError("Không thể tạo hai split không rỗng vì mọi ảnh trùng SHA-256.")
    target = len(records) * val_ratio
    best = min(choices, key=lambda total: (abs(total - target), total))
    selected: set[int] = set()
    cursor = best
    while cursor:
        previous, index = reachable[cursor]
        selected.update(ordered[index])
        cursor = previous
    train = [row for row in records if row["image_id"] not in selected]
    val = [row for row in records if row["image_id"] in selected]
    return train, val, {
        "method": "sorted_ids_sha256_groups_seeded_shuffle_exact_subset_sum",
        "seed": seed, "val_ratio": val_ratio, "actual_val_ratio": len(val) / len(records),
        "target_validation_images": target,
        "group_constraint_adjusted_count": best != int(target + .5),
        "count_reason": "Số ảnh gần tỷ lệ yêu cầu nhất trong các tổng nhóm SHA-256 khả thi.",
    }


def exclude_train_test_duplicates(
    train: list[dict], test: list[dict]
) -> tuple[list[dict], list[dict], dict]:
    """Giữ nguyên test, loại mọi ảnh train có SHA-256 xuất hiện trong test.

    Không lọc theo ID hay filename cố định, không sửa records/ảnh nguồn.
    Metadata giữ mọi annotation ID bị loại và toàn bộ test ID cùng hash.
    """
    test_by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(test, key=lambda row: row["image_id"]):
        test_by_hash[row["_sha256"]].append(row)
    eligible, excluded = [], []
    for row in sorted(train, key=lambda row: row["image_id"]):
        matches = test_by_hash.get(row["_sha256"], [])
        if not matches:
            eligible.append(row)
            continue
        excluded.append({
            "train_image_id": row["image_id"], "image_filename": row["image_filename"],
            "image_path": row["image_path"], "sha256": row["_sha256"],
            "matching_test_image_ids": [match["image_id"] for match in matches],
            "matching_test_filenames": [match["image_filename"] for match in matches],
            "annotation_ids": list(row["annotation_ids"]),
            "excluded_annotation_count": len(row["annotation_ids"]),
            "reason": "duplicate_sha256_with_original_test",
        })
    raw_overlap = leakage_checks({"train": train, "val": [], "test": test})["train_test"]
    return eligible, excluded, {
        "status": "WARNING" if raw_overlap["status"] == "FAIL" else "PASS",
        "cross_split_duplicate_groups": len(raw_overlap["sha256"]),
        "cross_split_overlap": raw_overlap,
        "original_train_images": len(train), "original_test_images": len(test),
        "eligible_train_images": len(eligible),
        "eligible_train_annotations": sum(len(row["annotation_ids"]) for row in eligible),
        "excluded_train_image_count": len(excluded),
        "excluded_annotation_count": sum(row["excluded_annotation_count"] for row in excluded),
    }


def leakage_checks(splits: dict[str, list[dict]]) -> dict:
    """Đo overlap riêng theo ID, filename, SHA-256 và đường dẫn ảnh vật lý."""
    checks = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = {}
        for field, label in (("image_id", "image_ids"), ("image_filename", "filenames"),
                             ("_sha256", "sha256"), ("image_path", "image_paths")):
            def values(name: str) -> set:
                return {row[field].casefold() if field == "image_filename" else row[field]
                        for row in splits[name]}
            overlap[label] = sorted(values(left) & values(right))
        checks[f"{left}_{right}"] = {
            "status": "PASS" if not any(overlap.values()) else "FAIL", **overlap,
            "sha256_groups": [
                {"sha256": digest, **{
                    name: [{"image_id": row["image_id"], "filename": row["image_filename"]}
                           for row in splits[name] if row["_sha256"] == digest]
                    for name in (left, right)
                }} for digest in overlap["sha256"]
            ],
        }
    return checks


def _public_record(row: dict) -> dict:
    return {key: row[key] for key in (
        "image_id", "image_filename", "image_path", "captions", "annotation_ids", "original_split"
    )}


def read_jsonl(path: str | Path) -> list[dict]:
    """Đọc và xác minh schema JSONL mỗi dòng một ảnh, không chứa token IDs."""
    rows, ids, filenames, seen_annotations = [], set(), set(), set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"JSONL lỗi tại dòng {number}.") from exc
            if not isinstance(row, dict) or not all(key in row for key in (
                "image_id", "image_filename", "image_path", "captions", "annotation_ids"
            )):
                raise DataValidationError(f"JSONL thiếu trường tại dòng {number}.")
            if (type(row["image_id"]) is not int or row["image_id"] < 0
                    or not _valid_filename(row["image_filename"])
                    or not isinstance(row["image_path"], str)
                    or "\\" in row["image_path"]
                    or PurePosixPath(row["image_path"]).is_absolute()
                    or PureWindowsPath(row["image_path"]).drive
                    or ".." in PurePosixPath(row["image_path"]).parts
                    or PurePosixPath(row["image_path"]).name != row["image_filename"]):
                raise DataValidationError(f"ID/filename/image_path JSONL không hợp lệ tại dòng {number}.")
            captions, annotation_ids = row["captions"], row["annotation_ids"]
            if (not isinstance(captions, list) or not captions
                    or not all(isinstance(text, str) and text and normalize_caption(text) == text for text in captions)
                    or not isinstance(annotation_ids, list)
                    or not all(type(index) is int and index >= 0 for index in annotation_ids)
                    or len(captions) != len(annotation_ids)
                    or annotation_ids != sorted(set(annotation_ids))):
                raise DataValidationError(f"Caption và annotation_ids không đồng bộ tại dòng {number}.")
            if row["image_id"] in ids or row["image_filename"].casefold() in filenames:
                raise DataValidationError("JSONL chứa ảnh trùng ID/filename.")
            if seen_annotations.intersection(annotation_ids):
                raise DataValidationError("JSONL chứa annotation_id lặp giữa các ảnh.")
            ids.add(row["image_id"])
            filenames.add(row["image_filename"].casefold())
            seen_annotations.update(annotation_ids)
            rows.append(row)
    if not rows or [row["image_id"] for row in rows] != sorted(ids):
        raise DataValidationError("JSONL rỗng hoặc chưa được sắp theo image_id.")
    return rows


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    """Ghi file tạm độc quyền rồi replace; không sửa file khi bytes không thay đổi."""
    if path.exists() and path.read_bytes() == content:
        return
    temporary = path.with_name(path.name + ".part")
    if temporary.exists():
        raise DataValidationError(f"Có output tạm chưa được kiểm tra: {temporary.name}.")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _raw_statistics(records: list[dict]) -> dict:
    histogram = Counter(len(row["captions"]) for row in records)
    return {"images": len(records), "annotations": sum(len(row["captions"]) for row in records),
            "captions_per_image": dict(sorted(histogram.items())),
            "missing_caption_images": 0,
            "unusual_caption_image_ids": [row["image_id"] for row in records if len(row["captions"]) != 5]}


def _resolve_raw(
    raw_dir: Path, train_json: Path | None, test_json: Path | None,
    download_missing: bool, verify_only: bool,
) -> dict[str, Path]:
    paths = {name: raw_dir / name for name in RAW_NAMES}
    if train_json is not None:
        paths["ktvic-train_data.json"] = Path(train_json)
    if test_json is not None:
        paths["ktvic-test_data.json"] = Path(test_json)
    for name, path in paths.items():
        if path.exists():
            if not path.is_file() or not path.stat().st_size:
                raise DataValidationError(f"Raw file không hợp lệ: {path.name}; không ghi đè.")
            continue
        if verify_only or not download_missing:
            raise DataValidationError(f"Thiếu raw file {path.name}; dùng --download-missing nếu cần tải.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        if temporary.exists():
            raise DataValidationError(f"File tải dở tồn tại: {temporary.name}; cần kiểm tra trước.")
        LOGGER.info("Đang tải %s", name)
        try:
            url = f"{DATASET_URL}/resolve/main/{name}"
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("xb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if name.endswith(".zip"):
                with zipfile.ZipFile(temporary) as archive:
                    if archive.testzip() is not None:
                        raise DataValidationError(f"ZIP tải về lỗi CRC: {name}.")
            else:
                load_annotations(temporary, "train" if "train" in name else "test")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return paths


def _token_statistics(splits: dict[str, list[dict]], vocab: VietnameseTokenizer) -> dict:
    result = {}
    for split, records in splits.items():
        counts: Counter[str] = Counter()
        lengths = []
        for row in records:
            for caption in row["captions"]:
                tokens = tokenize(caption)
                counts.update(tokens)
                lengths.append(len(tokens))
        total = sum(counts.values())
        unknown = sum(count for token, count in counts.items() if token not in vocab.token_to_id)
        result[split] = {**distribution(lengths), "total_tokens": total,
                         "oov_tokens": unknown, "oov_rate": unknown / total if total else 0.0}
    return result


def smoke_test_dataloaders(root: Path, processed: Path, vocab: VietnameseTokenizer) -> dict:
    """Kiểm tra batch thật từ cả ba JSONL ở hai mode, không gọi bất kỳ model nào.

    Chỉ import PyTorch khi output đã được đọc lại. Kết quả không chứa thời gian
    hay dữ liệu ngẫu nhiên, bảo đảm báo cáo deterministic khi chạy lại.
    """
    try:
        import torch
        from .dataset import KTVICDataset, create_dataloader
    except ImportError as exc:
        raise DataValidationError("Dataset/DataLoader smoke test cần torch và torchvision tương thích.") from exc

    results = {}
    for split in ("train", "val", "test"):
        source = processed / f"{split}.jsonl"
        caption_data = KTVICDataset(source, vocab, project_root=root)
        image_data = KTVICDataset(source, mode="image", project_root=root)
        size = min(2, len(image_data))
        caption_batch = next(iter(create_dataloader(caption_data, batch_size=size)))
        image_batch = next(iter(create_dataloader(image_data, batch_size=size)))
        for batch in (caption_batch, image_batch):
            images = batch["images"]
            if (tuple(images.shape) != (size, 3, 224, 224)
                    or not images.is_floating_point() or images.device.type != "cpu"
                    or not torch.isfinite(images).all().item()):
                raise DataValidationError(f"Batch ảnh {split} không đạt hợp đồng CNN encoder.")
        if len(caption_data) != sum(len(row["captions"]) for row in caption_data.records):
            raise DataValidationError("Caption mode làm mất sample.")
        if len(image_data) != len(image_data.records):
            raise DataValidationError("Image mode lặp ảnh.")
        ids, inputs, targets = (caption_batch[key] for key in ("caption_ids", "input_ids", "target_ids"))
        lengths = caption_batch["lengths"]
        if not torch.equal(caption_batch["attention_mask"], inputs.ne(vocab.pad_id)):
            raise DataValidationError("Attention mask không khớp padding.")
        if not torch.equal(caption_batch["attention_mask"].sum(1), lengths - 1):
            raise DataValidationError("Mask/lengths không khớp độ dài caption.")
        for index in range(size):
            length = lengths[index].item()
            sequence = ids[index, :length]
            sample = caption_data[index]
            if (sequence[0].item() != vocab.bos_id or sequence[-1].item() != vocab.eos_id
                    or not torch.equal(inputs[index, :length - 1], sequence[:-1])
                    or not torch.equal(targets[index, :length - 1], sequence[1:])
                    or not ids[index, length:].eq(vocab.pad_id).all().item()
                    or not inputs[index, length - 1:].eq(vocab.pad_id).all().item()
                    or not targets[index, length - 1:].eq(vocab.pad_id).all().item()
                    or not torch.equal(sequence, sample["caption_ids"])):
                raise DataValidationError(f"BOS/EOS/PAD/shift không hợp lệ trên {split}.")
            # UNK có thể xuất hiện trên val/test; roundtrip nguyên văn kiểm trên
            # train, và kiểm encode/decode chuỗi token đã biết trên mọi split.
            decoded = vocab.decode(sequence.tolist())
            if split == "train" and decoded != sample["caption"]:
                raise DataValidationError("Decode train làm thay đổi dấu/nội dung tiếng Việt.")
            row = image_data.records[index]
            if image_batch["references"][index] != row["captions"]:
                raise DataValidationError("Image mode làm mất reference captions.")
        results[split] = {
            "status": "PASS", "caption_samples": len(caption_data), "image_samples": len(image_data),
            "caption_batch_image_shape": list(caption_batch["images"].shape),
            "image_batch_image_shape": list(image_batch["images"].shape),
            "reference_counts": [len(refs) for refs in image_batch["references"]],
            "caption_ids_shape": list(ids.shape), "input_ids_shape": list(inputs.shape),
            "mask_bos_eos_pad_shift_valid": True,
        }
    return {"status": "PASS", "num_workers": 0, "splits": results}


def prepare_data(
    project_root: str | Path, *, raw_dir: str | Path | None = None,
    processed_dir: str | Path | None = None, images_dir: str | Path | None = None,
    train_json: str | Path | None = None, test_json: str | Path | None = None,
    seed: int = 42, val_ratio: float = .10, download_missing: bool = False,
    verify_only: bool = False,
) -> dict:
    """Điều phối kiểm toán và xuất dữ liệu; verify_only không ghi bất kỳ file nào.

    Output cũ chỉ được tái sử dụng nếu toàn bộ bytes khớp kết quả tái tính.
    Nếu raw/config thay đổi hoặc output hỏng, báo lỗi thay vì trộn hai phiên bản.
    Báo cáo được ghi cuối cùng và không đánh dấu DataLoader đã kiểm khi chưa chạy.
    """
    root = Path(project_root).resolve()
    def resolve(value: str | Path | None, default: str) -> Path:
        candidate = Path(value) if value is not None else Path(default)
        return _within(candidate if candidate.is_absolute() else root / candidate, root)
    raw = resolve(raw_dir, "data/raw")
    processed = resolve(processed_dir, "data/processed")
    images = resolve(images_dir, "data/images")
    if (processed == raw or processed.is_relative_to(raw) or raw.is_relative_to(processed)
            or processed == images or processed.is_relative_to(images)
            or images.is_relative_to(processed)):
        raise DataValidationError("Các thư mục raw, images, processed không được lồng nhau.")
    if not 0 < val_ratio < 1:
        raise DataValidationError("val_ratio phải nằm giữa 0 và 1.")
    report = {key: {} for key in (
        "source", "raw_statistics", "image_audit", "caption_audit", "split_statistics",
        "tokenization_statistics", "vocabulary_statistics", "integrity_checks",
    )}
    integrity_names = (
        "all_images_readable", "all_annotations_linked", "all_image_paths_valid",
        "no_train_val_image_leakage", "no_train_test_image_leakage", "no_val_test_image_leakage",
        "vocabulary_train_only", "all_splits_non_empty", "caption_annotation_alignment_valid",
    )
    report["integrity_checks"] = {key: {"status": "NOT_CHECKED"} for key in integrity_names}
    report_path = processed / "data_report.json"
    try:
        LOGGER.info("1/7: Xác định và băm bốn raw files")
        paths = _resolve_raw(
            raw, resolve(train_json, "data/raw/ktvic-train_data.json") if train_json else None,
            resolve(test_json, "data/raw/ktvic-test_data.json") if test_json else None,
            download_missing, verify_only,
        )
        report["source"] = {
            "dataset": "KTVIC", "url": DATASET_URL, "seed": seed, "validation_ratio": val_ratio,
            "raw_files": {name: {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
                          for name, path in paths.items()},
            "normalization": "NFC; remove BOM/non-whitespace Cc; collapse whitespace; preserve case/punctuation",
            "tokenization": "syllable_whitespace_preserve_attached_punctuation",
            "caption_source": "caption", "segment_caption_use": "audit_only",
        }
        LOGGER.info("2/7: Parse và kiểm toán annotations")
        originals = {split: load_annotations(paths[f"ktvic-{split}_data.json"], split)
                     for split in ("train", "test")}
        report["raw_statistics"] = {split: _raw_statistics(rows) for split, rows in originals.items()}
        report["caption_audit"] = {split: audit_captions(rows) for split, rows in originals.items()}
        report["integrity_checks"]["all_annotations_linked"] = {
            "status": "PASS", "orphan_annotations": 0,
            "annotations_checked": sum(stats["annotations"] for stats in report["raw_statistics"].values()),
        }
        LOGGER.info("3/7: Kiểm tra toàn bộ ZIP members, giải nén và kiểm CRC")
        archives = {split: inspect_zip(paths[f"ktvic-{split}-images.zip"], originals[split])
                    for split in ("train", "test")}
        for split in ("train", "test"):
            directory = _within(images / split, root)
            extract_images(paths[f"ktvic-{split}-images.zip"], archives[split][0], directory,
                           verify_only=verify_only)
            LOGGER.info("4/7: Decode/RGB và SHA-256 cho %s (%s ảnh)", split, len(originals[split]))
            report["image_audit"][split] = {
                **audit_images(originals[split], directory, root), "zip": archives[split][1],
            }
        total_images = sum(len(rows) for rows in originals.values())
        for key in ("all_images_readable", "all_image_paths_valid"):
            report["integrity_checks"][key] = {"status": "PASS", "images_checked": total_images, "failures": 0}
        LOGGER.info("5/7: Giữ test, loại train trùng test rồi chia theo nhóm SHA-256")
        eligible, exclusions, source_audit = exclude_train_test_duplicates(originals["train"], originals["test"])
        report["source_audit"] = source_audit
        report["processing_policy"] = {
            "cross_split_duplicates": "preserve_test_exclude_train",
            "group_train_duplicates_by_sha256": True,
        }
        report["excluded_train_images"] = exclusions
        LOGGER.info("Loại khỏi processed train/val: %s ảnh, %s annotations; giữ nguyên ảnh nguồn",
                    len(exclusions), source_audit["excluded_annotation_count"])
        train, val, split_metadata = split_records(eligible, val_ratio, seed)
        splits = {"train": train, "val": val, "test": originals["test"]}
        overlap = leakage_checks(splits)
        report["image_audit"]["cross_split_overlap"] = overlap
        report["split_statistics"] = {
            **split_metadata, "status": "NOT_EXPORTED", "overlap": overlap,
            "splits": {split: {"images": len(rows), "captions": sum(len(row["captions"]) for row in rows),
                               "image_ids": [row["image_id"] for row in rows]}
                       for split, rows in splits.items()},
        }
        report["integrity_checks"]["all_splits_non_empty"] = {
            "status": "PASS", "counts": {name: len(rows) for name, rows in splits.items()},
        }
        for pair, check in overlap.items():
            report["integrity_checks"][f"no_{pair}_image_leakage"] = check
        if any(check["status"] != "PASS" for check in overlap.values()):
            raise DataValidationError("Phát hiện ảnh rò rỉ ID/filename/SHA-256 giữa các split.", overlap)
        report["processed_integrity"] = {
            "status": "PASS", "cross_split_sha256_leakage": sum(len(check["sha256"]) for check in overlap.values()),
            "cross_split_image_id_leakage": sum(len(check["image_ids"]) for check in overlap.values()),
            "cross_split_image_path_leakage": sum(len(check["image_paths"]) for check in overlap.values()),
            "original_test_images_preserved": len(splits["test"]),
            "original_test_annotations_preserved": sum(len(row["captions"]) for row in splits["test"]),
        }
        report["caption_audit"]["processed"] = {split: audit_captions(rows) for split, rows in splits.items()}
        report["split_statistics"]["status"] = "PREPARED"
        LOGGER.info("6/7: Tạo JSONL và xây vocabulary chỉ từ train")
        payloads = {
            f"{split}.jsonl": ("".join(json.dumps(_public_record(row), ensure_ascii=False, sort_keys=True)
                                      + "\n" for row in rows)).encode("utf-8")
            for split, rows in splits.items()
        }
        # Đọc lại đúng các dòng sẽ xuất của train, tránh vô tình dùng val/test.
        training_rows = [json.loads(line) for line in payloads["train.jsonl"].decode("utf-8").splitlines()]
        vocab = VietnameseTokenizer().build_vocab(caption for row in training_rows for caption in row["captions"])
        payloads["vocab.json"] = _json_bytes(vocab.to_dict())
        report["vocabulary_statistics"] = {
            "vocab_size": len(vocab), "min_freq": 1, "training_source": "train.jsonl",
            "special_token_ids": {token: vocab.token_to_id[token] for token in vocab.special_tokens.values()},
            "max_train_token_length": vocab.max_train_token_length,
            "token_frequencies": dict(sorted(Counter(token for row in training_rows for caption in row["captions"]
                                                     for token in tokenize(caption)).items())),
        }
        report["tokenization_statistics"] = _token_statistics(splits, vocab)
        report["integrity_checks"]["vocabulary_train_only"] = {
            "status": "PASS", "source_sha256": hashlib.sha256(payloads["train.jsonl"]).hexdigest(),
            "training_images": len(train), "validation_test_captions_used": 0,
        }
        alignment_count = sum(len(row["captions"]) for rows in splits.values() for row in rows)
        raw_annotation_count = sum(stats["annotations"] for stats in report["raw_statistics"].values())
        if alignment_count + source_audit["excluded_annotation_count"] != raw_annotation_count:
            raise DataValidationError("Annotations không được bảo toàn sau exclusions.")
        report["integrity_checks"]["caption_annotation_alignment_valid"] = {
            "status": "PASS", "annotation_pairs_checked": alignment_count,
            "excluded_annotation_count": source_audit["excluded_annotation_count"],
            "raw_annotation_count": raw_annotation_count,
        }
        report["processing_status"] = {
            "state": "VERIFYING", "raw_image_caption_audit": source_audit["status"],
            "dataset_dataloader_smoke_test": "NOT_CHECKED",
            "reason": "DataLoader phải được kiểm riêng trong môi trường có torch/torchvision.",
            "output_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()},
        }
        # Kiểm tra TOÀN BỘ output trước khi ghi bất kỳ file nào.
        previous_report = None
        if report_path.exists():
            previous_report = json.loads(report_path.read_text(encoding="utf-8"))
            previous_state = previous_report.get("processing_status", {}).get("state")
            if previous_state != "FAILED" and (
                previous_report.get("source") != report["source"]
                or previous_report.get("processing_policy") != report["processing_policy"]
                or previous_report.get("processing_status", {}).get("output_sha256") != report["processing_status"]["output_sha256"]
            ):
                raise DataValidationError("Report cũ thuộc raw/config/output khác; không ghi đè.")
        for name, content in payloads.items():
            target = _within(processed / name, root)
            if target.exists() and target.read_bytes() != content:
                raise DataValidationError(f"Output {name} khác raw/config hiện tại; dùng --processed-dir mới hoặc kiểm tra output cũ.")
            if verify_only and not target.is_file():
                raise DataValidationError(f"Thiếu output cần verify: {name}.")
        if verify_only and not report_path.is_file():
            raise DataValidationError("Thiếu data_report.json cần verify.")
        if not verify_only:
            processed.mkdir(parents=True, exist_ok=True)
            for name, content in payloads.items():
                _atomic_write(processed / name, content)
        LOGGER.info("7/7: Xác minh output đã lưu và vocabulary load lại")
        for split, rows in splits.items():
            actual = read_jsonl(processed / f"{split}.jsonl")
            if actual != [_public_record(row) for row in rows]:
                raise DataValidationError("JSONL không khớp annotations gốc sau chuẩn hóa.")
        if VietnameseTokenizer.load_vocab(processed / "vocab.json").to_dict() != vocab.to_dict():
            raise DataValidationError("Vocabulary load lại không khớp.")
        for name, content in payloads.items():
            if (processed / name).read_bytes() != content:
                raise DataValidationError(f"Output {name} thay đổi trong khi ghi/kiểm tra.")
        report["dataloader_smoke_tests"] = smoke_test_dataloaders(root, processed, vocab)
        report["processing_status"].update({
            "state": "COMPLETED_WITH_SOURCE_EXCLUSIONS" if exclusions else "COMPLETED",
            "dataset_dataloader_smoke_test": "PASS",
            "reason": "Output đọc lại hợp lệ; DataLoader CPU chạy cả hai mode trên cả ba split.",
        })
        report["split_statistics"]["status"] = "EXPORTED"
        report_content = _json_bytes(report)
        if previous_report is not None and previous_report.get("processing_status", {}).get("state") != "FAILED":
            if report_path.read_bytes() != report_content:
                raise DataValidationError("Report hiện có không khớp kết quả kiểm toán tái tính; không ghi đè.")
        if verify_only and report_path.read_bytes() != report_content:
            raise DataValidationError("Report hiện có không khớp kết quả verify-only.")
        if not verify_only:
            # Báo cáo là dấu hoàn tất, chỉ công bố sau khi read-back đã đạt.
            _atomic_write(report_path, report_content)
        return report
    except (DataValidationError, OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        report["processing_status"] = {"state": "FAILED", "error": str(exc),
                                        "details": getattr(exc, "details", {})}
        # Giữ report hoàn tất cũ; chỉ cập nhật report chưa có/FAILED khi có lỗi.
        previous_failed = False
        if report_path.is_file():
            try:
                previous_failed = json.loads(report_path.read_text(encoding="utf-8")).get("processing_status", {}).get("state") == "FAILED"
            except (OSError, ValueError, AttributeError):
                pass
        if not verify_only and (not report_path.exists() or previous_failed):
            processed.mkdir(parents=True, exist_ok=True)
            _atomic_write(report_path, _json_bytes(report))
        raise
