"""Unit tests tương thích pytest/unittest, chỉ tạo dữ liệu nhỏ trong thư mục tạm."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
import unicodedata
import zipfile
from pathlib import Path

from PIL import Image

from image_captioning.data.prepare_ktvic import (
    DataValidationError, audit_captions, inspect_zip, load_annotations,
    exclude_train_test_duplicates, prepare_data, read_jsonl, sha256_file, split_records,
)
from image_captioning.data.tokenizer_vi import VietnameseTokenizer, normalize_caption


HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision"))


class KTVICFixture(unittest.TestCase):
    """Kiểm thử preprocessing độc lập PyTorch, dùng TemporaryDirectory như tmp_path."""

    @classmethod
    def setUpClass(cls) -> None:
        """Giới hạn CPU threads cho batch ảnh nhỏ của test, rồi khôi phục."""
        super().setUpClass()
        if HAS_TORCH:
            import torch
            previous = torch.get_num_threads()
            torch.set_num_threads(2)
            cls.addClassCleanup(torch.set_num_threads, previous)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        raw = self.root / "data/raw"
        raw.mkdir(parents=True)
        self.raw = raw
        for split, ids in (("train", range(1, 9)), ("test", range(100, 102))):
            images, annotations = [], []
            with zipfile.ZipFile(raw / f"ktvic-{split}-images.zip", "w") as archive:
                for image_id in reversed(list(ids)):
                    filename = f"{image_id:011d}.PNG"
                    images.append({"id": image_id, "filename": filename})
                    buffer = io.BytesIO()
                    image = Image.new("RGB", (32 + image_id, 25), (image_id, 50, 20))
                    image.save(buffer, format="PNG")
                    archive.writestr(f"prefix-khac-{split}/{filename}", buffer.getvalue())
                    for index in range(5):
                        text = f"Ảnh DuyNhất{image_id} có {index} người, đang đi bộ."
                        if image_id == 1 and index == 4:
                            text = "Ảnh DuyNhất1 có 0 người, đang đi bộ."
                        annotations.append({"id": image_id * 10 + index, "image_id": image_id,
                                            "caption": text, "segment_caption": text.replace("đi bộ", "đi_bộ")})
                archive.writestr("metadata.txt", "Không được giải nén file này.")
            (raw / f"ktvic-{split}_data.json").write_text(
                json.dumps({"images": images, "annotations": list(reversed(annotations))}, ensure_ascii=False),
                encoding="utf-8",
            )

    def prepare(self, **kwargs: object) -> dict:
        """Chạy pipeline hoàn chỉnh chỉ trên fixture tạm."""
        return prepare_data(self.root, **kwargs)


class PipelineTests(KTVICFixture):
    """Các test dữ liệu được pytest tự động thu thập qua unittest integration."""

    def test_01_json_parsing(self) -> None:
        rows = load_annotations(self.raw / "ktvic-train_data.json", "train")
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["image_id"], 1)

    def test_02_annotation_joining(self) -> None:
        rows = load_annotations(self.raw / "ktvic-train_data.json", "train")
        for row in rows:
            self.assertEqual(row["annotation_ids"], [row["image_id"] * 10 + index for index in range(5)])
            self.assertTrue(all(f'DuyNhất{row["image_id"]}' in text for text in row["captions"]))

    def test_03_preserve_five_captions(self) -> None:
        self.prepare()
        for split in ("train", "val", "test"):
            rows = read_jsonl(self.root / f"data/processed/{split}.jsonl")
            self.assertTrue(all(len(row["captions"]) == len(row["annotation_ids"]) == 5 for row in rows))

    def test_04_unicode_normalization(self) -> None:
        caption = "Ảnh có ba người đứng trước cửa hàng, số 12."
        self.assertEqual(normalize_caption("\ufeff  " + unicodedata.normalize("NFD", caption) + "\x00 \n"), caption)
        self.assertEqual(normalize_caption("Có\tba\nngười!"), "Có ba người!")
        with self.assertRaises(ValueError):
            normalize_caption("\ud800")

    def test_05_duplicates_reported_not_removed(self) -> None:
        rows = load_annotations(self.raw / "ktvic-train_data.json", "train")
        audit = audit_captions(rows)
        self.assertEqual(audit["duplicate_caption_excess_records"], 1)
        self.assertEqual(audit["duplicate_caption_groups"][0]["annotation_ids"], [10, 14])
        self.assertEqual(audit["captions_retained"], 40)
        self.assertEqual(audit["captions_removed"], 0)

    def test_06_no_train_validation_leakage(self) -> None:
        report = self.prepare()
        splits = report["split_statistics"]["splits"]
        self.assertFalse(set(splits["train"]["image_ids"]) & set(splits["val"]["image_ids"]))
        for check in report["image_audit"]["cross_split_overlap"].values():
            self.assertEqual(check["status"], "PASS")

    def test_07_deterministic_group_split(self) -> None:
        records = [{"image_id": i, "_sha256": "same" if i < 3 else str(i)} for i in range(10)]
        first = split_records(records, .3, 42)
        second = split_records(list(reversed(records)), .3, 42)
        self.assertEqual({row["image_id"] for row in first[1]}, {row["image_id"] for row in second[1]})
        for rows in first[:2]:
            selected = {row["image_id"] for row in rows}
            self.assertTrue(not selected.intersection({0, 1, 2}) or {0, 1, 2}.issubset(selected))
        self.assertEqual(len(first[1]), 3)

    def test_08_special_token_ids(self) -> None:
        vocab = VietnameseTokenizer().build_vocab(["b a b", "c a"])
        self.assertEqual([vocab.token_to_id[token] for token in ("<pad>", "<bos>", "<eos>", "<unk>")], [0, 1, 2, 3])
        self.assertEqual(vocab.token_to_id["a"], 4)
        self.assertEqual(vocab.token_to_id["b"], 5)
        with self.assertRaises(ValueError):
            vocab.build_vocab(["Có <bos> ở đây"])

    def test_09_encode_decode_roundtrip(self) -> None:
        caption = "Ảnh có 12 người, đang đứng trước cửa hàng!"
        vocab = VietnameseTokenizer().build_vocab([caption])
        ids = vocab.encode(caption)
        self.assertEqual((ids[0], ids[-1]), (1, 2))
        self.assertEqual(vocab.decode(ids + [0]), caption)
        self.assertEqual(vocab.encode("ChưaCó", False), [3])
        path = self.root / "vocab.json"
        vocab.save_vocab(path)
        self.assertEqual(VietnameseTokenizer.load_vocab(path).decode(ids), caption)

    def test_15_zip_security(self) -> None:
        records = load_annotations(self.raw / "ktvic-train_data.json", "train")
        for name in ("../escape.jpg", "..\\escape.jpg", "/escape.jpg", "C:/escape.jpg"):
            malicious = self.root / "malicious.zip"
            with zipfile.ZipFile(malicious, "w") as archive:
                archive.writestr(name, b"unsafe")
            with self.assertRaises(DataValidationError):
                inspect_zip(malicious, records)
        self.assertFalse((self.root.parent / "escape.jpg").exists())

    def test_16_missing_and_corrupted_images(self) -> None:
        self.prepare()
        rows = read_jsonl(self.root / "data/processed/test.jsonl")
        image_path = self.root / rows[0]["image_path"]
        original = image_path.read_bytes()
        image_path.unlink()
        with self.assertRaises(DataValidationError):
            self.prepare(verify_only=True)
        image_path.write_bytes(b"not an image")
        with self.assertRaises(DataValidationError):
            self.prepare()
        self.assertEqual(image_path.read_bytes(), b"not an image")
        image_path.write_bytes(original)

    def test_17_vocabulary_train_only(self) -> None:
        self.prepare()
        vocab = VietnameseTokenizer.load_vocab(self.root / "data/processed/vocab.json")
        train = read_jsonl(self.root / "data/processed/train.jsonl")
        tokens = {token for row in train for caption in row["captions"] for token in vocab.tokenize(caption)}
        self.assertEqual(set(vocab.token_to_id) - set(vocab.special_tokens.values()), tokens)
        for split in ("val", "test"):
            for row in read_jsonl(self.root / f"data/processed/{split}.jsonl"):
                self.assertNotIn(f'DuyNhất{row["image_id"]}', vocab.token_to_id)

    def test_18_rerun_and_verify_only_integrity(self) -> None:
        self.prepare()
        paths = sorted(path for path in self.root.rglob("*") if path.is_file())
        before = {str(path): (sha256_file(path), path.stat().st_mtime_ns) for path in paths}
        self.prepare()
        self.prepare(verify_only=True)
        after = {str(path): (sha256_file(path), path.stat().st_mtime_ns) for path in paths}
        self.assertEqual(before, after)
        self.assertEqual(paths, sorted(path for path in self.root.rglob("*") if path.is_file()))

    def test_orphan_and_duplicate_ids_fail(self) -> None:
        path = self.raw / "ktvic-train_data.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["annotations"][0]["image_id"] = 9999
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(DataValidationError) as caught:
            load_annotations(path, "train")
        self.assertIn("orphan_annotation_ids", caught.exception.details)
        data["images"].append(data["images"][0])
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(DataValidationError):
            load_annotations(path, "train")

    def test_source_caption_used_instead_of_segment(self) -> None:
        path = self.raw / "ktvic-test_data.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["annotations"][0]["segment_caption"] = "Nội_dung khác hẳn"
        path.write_text(json.dumps(data), encoding="utf-8")
        report = self.prepare()
        self.assertEqual(report["caption_audit"]["test"]["segment_caption_mismatch_count"], 1)
        rows = read_jsonl(self.root / "data/processed/test.jsonl")
        self.assertTrue(all("Nội_dung" not in caption for row in rows for caption in row["captions"]))

    def test_cross_split_byte_duplicates_exclude_train_preserve_test(self) -> None:
        # Cùng bytes nhưng khác ID/filename phải được phát hiện và lọc từ train.
        train_zip = self.raw / "ktvic-train-images.zip"
        with zipfile.ZipFile(train_zip) as archive:
            image_bytes = archive.read("prefix-khac-train/00000000001.PNG")
        test_zip = self.raw / "ktvic-test-images.zip"
        with zipfile.ZipFile(test_zip) as archive:
            members = {item.filename: archive.read(item) for item in archive.infolist()}
        members["prefix-khac-test/00000000100.PNG"] = image_bytes
        with zipfile.ZipFile(test_zip, "w") as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        raw_before = {path.name: sha256_file(path) for path in self.raw.iterdir()}
        report = self.prepare()
        self.assertEqual(report["processing_status"]["state"], "COMPLETED_WITH_SOURCE_EXCLUSIONS")
        self.assertEqual(report["source_audit"]["status"], "WARNING")
        self.assertEqual(report["source_audit"]["cross_split_duplicate_groups"], 1)
        self.assertEqual(report["processed_integrity"]["cross_split_sha256_leakage"], 0)
        self.assertEqual(report["processed_integrity"]["status"], "PASS")
        groups = report["source_audit"]["cross_split_overlap"]["sha256_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["test"][0]["image_id"], 100)
        self.assertEqual(report["excluded_train_images"][0]["train_image_id"], 1)
        self.assertEqual(report["excluded_train_images"][0]["annotation_ids"], [10, 11, 12, 13, 14])
        self.assertEqual(report["excluded_train_images"][0]["matching_test_image_ids"], [100])
        self.assertEqual(report["source_audit"]["excluded_annotation_count"], 5)
        train_val = [row for split in ("train", "val")
                     for row in read_jsonl(self.root / f"data/processed/{split}.jsonl")]
        self.assertEqual({row["image_id"] for row in train_val}, set(range(2, 9)))
        self.assertEqual(sum(len(row["captions"]) for row in train_val), 35)
        self.assertEqual([row["image_id"] for row in read_jsonl(self.root / "data/processed/test.jsonl")], [100, 101])
        self.assertTrue((self.root / "data/images/train/00000000001.PNG").is_file())
        self.assertEqual(raw_before, {path.name: sha256_file(path) for path in self.raw.iterdir()})
        vocab = VietnameseTokenizer.load_vocab(self.root / "data/processed/vocab.json")
        self.assertNotIn("DuyNhất1", vocab.token_to_id)
        before = (self.root / "data/processed/data_report.json").read_bytes()
        for verify_only in (False, True):
            self.prepare(verify_only=verify_only)
            self.assertEqual(before, (self.root / "data/processed/data_report.json").read_bytes())

    def test_all_train_matches_are_excluded_without_hardcoded_ids(self) -> None:
        def record(image_id: int, digest: str, split: str) -> dict:
            return {"image_id": image_id, "image_filename": f"{image_id}.png",
                    "image_path": f"data/images/{split}/{image_id}.png", "_sha256": digest,
                    "annotation_ids": [image_id * 10 + index for index in range(5)]}
        train = [record(44, "a", "train"), record(45, "a", "train"), record(46, "b", "train")]
        test = [record(900, "a", "test"), record(901, "a", "test")]
        eligible, excluded, audit = exclude_train_test_duplicates(train, test)
        self.assertEqual([row["image_id"] for row in eligible], [46])
        self.assertEqual([row["train_image_id"] for row in excluded], [44, 45])
        self.assertEqual(excluded[0]["matching_test_image_ids"], [900, 901])
        self.assertEqual(audit["excluded_annotation_count"], 10)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(test), 2)

    def test_internal_train_byte_duplicates_stay_in_same_split(self) -> None:
        path = self.raw / "ktvic-train-images.zip"
        with zipfile.ZipFile(path) as archive:
            members = {item.filename: archive.read(item) for item in archive.infolist()}
        members["prefix-khac-train/00000000002.PNG"] = members["prefix-khac-train/00000000001.PNG"]
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        report = self.prepare()
        self.assertEqual(report["excluded_train_images"], [])
        for split in ("train", "val"):
            ids = set(report["split_statistics"]["splits"][split]["image_ids"])
            self.assertTrue(not ids.intersection({1, 2}) or {1, 2}.issubset(ids))
        self.assertEqual(sum(report["split_statistics"]["splits"][split]["images"] for split in ("train", "val")), 8)

    def test_unresolved_image_id_overlap_still_blocks_export(self) -> None:
        path = self.raw / "ktvic-test_data.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for image in data["images"]:
            if image["id"] == 100:
                image["id"] = 1
        for annotation in data["annotations"]:
            if annotation["image_id"] == 100:
                annotation["image_id"] = 1
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(DataValidationError):
            self.prepare()
        self.assertFalse((self.root / "data/processed/train.jsonl").exists())
        report = json.loads((self.root / "data/processed/data_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["processing_status"]["state"], "FAILED")

    def test_different_configuration_does_not_overwrite_outputs(self) -> None:
        self.prepare()
        paths = list((self.root / "data/processed").iterdir())
        before = {path.name: path.read_bytes() for path in paths}
        with self.assertRaises(DataValidationError):
            self.prepare(seed=19)
        self.assertEqual(before, {path.name: path.read_bytes() for path in paths})

    def test_bad_vocab_schema_rejected(self) -> None:
        data = VietnameseTokenizer().build_vocab(["một ảnh"]).to_dict()
        data["token_to_id"]["<pad>"] = 9
        with self.assertRaises(ValueError):
            VietnameseTokenizer.from_dict(data)


@unittest.skipUnless(HAS_TORCH, "Thiếu torch/torchvision; không cài dependency trong tests.")
class DatasetTests(KTVICFixture):
    """Tests PyTorch chỉ chạy khi dependency thực có sẵn, không giả lập tensor."""

    def setUp(self) -> None:
        super().setUp()
        self.prepare()
        self.source = self.root / "data/processed/train.jsonl"
        self.vocab = VietnameseTokenizer.load_vocab(self.root / "data/processed/vocab.json")

    def test_10_dataset_caption_mode(self) -> None:
        from image_captioning.data.dataset import KTVICDataset
        dataset = KTVICDataset(self.source, self.vocab, project_root=self.root)
        self.assertEqual(len(dataset), len(dataset.records) * 5)
        self.assertEqual(dataset[0]["caption_ids"][0].item(), 1)
        self.assertEqual(dataset[0]["caption_ids"][-1].item(), 2)

    def test_11_dataset_image_mode(self) -> None:
        from image_captioning.data.dataset import KTVICDataset, create_dataloader
        dataset = KTVICDataset(self.source, mode="image", project_root=self.root)
        self.assertEqual(len(dataset), len(dataset.records))
        self.assertEqual(len(dataset[0]["references"]), 5)
        batch = next(iter(create_dataloader(dataset, batch_size=2)))
        self.assertEqual(len(batch["references"]), 2)
        self.assertEqual(tuple(batch["images"].shape), (2, 3, 224, 224))

    def test_12_collate_padding(self) -> None:
        import torch
        from image_captioning.data.dataset import collate_fn
        batch = collate_fn([
            {"image": torch.zeros(3, 224, 224), "image_id": 1, "annotation_id": 1,
             "caption_ids": torch.tensor([1, 4, 2])},
            {"image": torch.zeros(3, 224, 224), "image_id": 2, "annotation_id": 2,
             "caption_ids": torch.tensor([1, 4, 5, 2])},
        ])
        self.assertEqual(batch["caption_ids"].tolist(), [[1, 4, 2, 0], [1, 4, 5, 2]])
        self.assertEqual(batch["input_ids"].tolist(), [[1, 4, 0], [1, 4, 5]])
        self.assertEqual(batch["target_ids"].tolist(), [[4, 2, 0], [4, 5, 2]])
        self.assertEqual(batch["attention_mask"].tolist(), [[True, True, False], [True, True, True]])
        self.assertEqual(batch["lengths"].tolist(), [3, 4])

    def test_13_rgb_transformation(self) -> None:
        import torch
        from image_captioning.data.dataset import KTVICDataset
        row = read_jsonl(self.source)[0]
        path = self.root / row["image_path"]
        Image.new("L", (40, 30), 128).save(path)
        dataset = KTVICDataset(self.source, self.vocab, project_root=self.root)
        image = dataset[0]["image"]
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertTrue(image.is_floating_point())
        self.assertTrue(torch.isfinite(image).all())
        expected = (torch.full((3,), 128 / 255) - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
        self.assertTrue(torch.allclose(image[:, 0, 0], expected, atol=1e-6))

    def test_14_portable_image_paths(self) -> None:
        from image_captioning.data.dataset import KTVICDataset, create_dataloader
        dataset = KTVICDataset("data/processed/train.jsonl", self.vocab, project_root=self.root)
        self.assertTrue(all("\\" not in row["image_path"] for row in dataset.records))
        batch = next(iter(create_dataloader(dataset, batch_size=2, num_workers=0)))
        self.assertEqual(tuple(batch["images"].shape), (2, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()
