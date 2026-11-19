"""PyTorch Dataset/DataLoader chung cho LSTM, Attention-LSTM và Transformer."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights

from .prepare_ktvic import DataValidationError, read_jsonl
from .tokenizer_vi import VietnameseTokenizer


class KTVICDataset(Dataset):
    """Nạp ảnh theo nhu cầu từ JSONL, giữ metadata/index trong RAM.

    mode='caption' trả từng cặp ảnh-caption; mode='image' trả mỗi ảnh một lần
    với toàn bộ references đã chuẩn hóa. Transform mặc định là chính sách
    deterministic của ResNet50 ImageNet-1K V2, không tải model/trọng số,
    trả float tensor [3, 224, 224] đã normalize ImageNet. Không cắt caption.
    """

    def __init__(
        self, jsonl_path: str | Path, vocab: VietnameseTokenizer | str | Path | None = None,
        mode: str = "caption", transform: Callable | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        if mode not in {"caption", "image"}:
            raise ValueError("mode phải là 'caption' hoặc 'image'.")
        self.project_root = (
            Path(project_root).resolve() if project_root is not None
            else Path(__file__).resolve().parents[3]
        )
        source = Path(jsonl_path)
        if not source.is_absolute():
            source = self.project_root / source
        self.records = read_jsonl(source)
        self.mode = mode
        self.vocab = VietnameseTokenizer.load_vocab(vocab) if isinstance(vocab, (str, Path)) else vocab
        if mode == "caption" and not isinstance(self.vocab, VietnameseTokenizer):
            raise TypeError("Caption mode cần vocabulary từ tokenizer_vi.VietnameseTokenizer.")
        self.transform = transform if transform is not None else ResNet50_Weights.IMAGENET1K_V2.transforms()
        self.index = (
            [(image_index, caption_index)
             for image_index, row in enumerate(self.records)
             for caption_index in range(len(row["captions"]))]
            if mode == "caption" else [(image_index, None) for image_index in range(len(self.records))]
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        """Load/decode/RGB ảnh, không chuyển GPU hoặc sửa ảnh gốc trên ổ đĩa."""
        image_index, caption_index = self.index[index]
        row = self.records[image_index]
        path = (self.project_root / row["image_path"]).resolve()
        if not path.is_relative_to(self.project_root):
            raise DataValidationError("image_path thoát khỏi project_root.")
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu ảnh image_id={row['image_id']}: {row['image_path']}")
        try:
            with Image.open(path) as image:
                image.load()
                rgb = image.convert("RGB")
            image_tensor = self.transform(rgb)
        except (OSError, ValueError, SyntaxError) as exc:
            raise DataValidationError(f"Không đọc/transform được ảnh image_id={row['image_id']}.") from exc
        if not isinstance(image_tensor, torch.Tensor) or image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
            raise ValueError("Transform phải trả tensor RGB [3, H, W].")
        if not image_tensor.is_floating_point():
            raise TypeError("Transform phải trả floating-point tensor.")
        result = {"image": image_tensor, "image_id": row["image_id"]}
        if self.mode == "image":
            result["references"] = list(row["captions"])
        else:
            caption = row["captions"][caption_index]
            result.update({
                "caption_ids": torch.tensor(self.vocab.encode(caption), dtype=torch.long),
                "caption": caption, "annotation_id": row["annotation_ids"][caption_index],
            })
        return result


def collate_fn(batch: list[dict]) -> dict:
    """Dynamic padding với PAD=0, không sort lại hoặc cắt caption.

    Caption mode: caption_ids chứa BOS/EOS, lengths tính cả BOS/EOS.
    input_ids/target_ids được shift TRƯỚC padding; attention_mask có shape
    như input_ids, True là token hợp lệ, False là padding. input_lengths bằng
    lengths - 1. Điều này tránh thêm EOS thừa vào input của caption ngắn.
    Image mode: giữ references dưới dạng list[list[str]], không tokenize lại.
    """
    if not batch:
        raise ValueError("Không thể collate batch rỗng.")
    image_mode = "references" in batch[0]
    if any(("references" in sample) != image_mode for sample in batch):
        raise ValueError("Không trộn caption mode và image mode trong một batch.")
    result = {"images": torch.stack([sample["image"] for sample in batch]),
              "image_ids": torch.tensor([sample["image_id"] for sample in batch], dtype=torch.long)}
    if image_mode:
        result["references"] = [sample["references"] for sample in batch]
        return result
    sequences = [sample["caption_ids"] for sample in batch]
    if any(sequence.ndim != 1 or len(sequence) < 2
           or sequence[0].item() != VietnameseTokenizer.bos_id
           or sequence[-1].item() != VietnameseTokenizer.eos_id for sequence in sequences):
        raise ValueError("caption_ids phải chứa BOS/EOS và có một dimension.")
    pad = VietnameseTokenizer.pad_id
    result["caption_ids"] = pad_sequence(sequences, batch_first=True, padding_value=pad)
    result["input_ids"] = pad_sequence([sequence[:-1] for sequence in sequences], batch_first=True, padding_value=pad)
    result["target_ids"] = pad_sequence([sequence[1:] for sequence in sequences], batch_first=True, padding_value=pad)
    result["attention_mask"] = result["input_ids"].ne(pad)
    result["lengths"] = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    result["input_lengths"] = result["lengths"] - 1
    result["annotation_ids"] = torch.tensor([sample["annotation_id"] for sample in batch], dtype=torch.long)
    return result


def create_dataloader(
    dataset: KTVICDataset, batch_size: int = 32, shuffle: bool = False,
    num_workers: int = 0, seed: int = 42,
) -> DataLoader:
    """Tạo DataLoader CPU với seed cố định; mặc định an toàn cho Windows spawn."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_fn, generator=generator)
