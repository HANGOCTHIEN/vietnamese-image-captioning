"""Sinh caption tiếng Việt cho ảnh bất kỳ bằng checkpoint đã train.

Ví dụ:
  python scripts/predict.py --ckpt checkpoints/best.pt --image a.jpg b.jpg
  python scripts/predict.py --ckpt checkpoints/best.pt --image a.jpg --out ketqua.png
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

import sys

from torchvision.models import ResNet50_Weights

ROOT = Path(__file__).resolve().parents[3]  # project root: chứa src/, data/, scripts/
sys.path.insert(0, str(ROOT / "src"))

# >>> Sửa đường dẫn package ở 3 dòng import này nếu project của bạn đặt khác <<<
from image_captioning.data.tokenizer_vi import VietnameseTokenizer  # noqa: E402
from image_captioning.models.cnn_encoder import CNNEncoder  # noqa: E402
from image_captioning.models.tranformer_decoder import TransformerCaptionDecoder  # noqa: E402


def resolve(path) -> Path:
    """Đường dẫn tương đối được hiểu là tính từ project root, không phụ thuộc thư mục chạy lệnh."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def image_transform():
    """Transform deterministic của ResNet50 (resize 232, center-crop 224, normalize ImageNet).

    Giống hệt transform mặc định trong KTVICDataset, nên ảnh mới được xử lý
    đúng như ảnh lúc train/eval.
    """
    return ResNet50_Weights.IMAGENET1K_V2.transforms()


def max_generation_len(decoder, requested: int) -> int:
    """Giới hạn độ dài sinh theo bảng positional encoding của decoder (bos + max_len token)."""
    return min(requested, decoder.pos_enc.pe.size(1) - 1)


def load_model(ckpt_path, vocab_path, device):
    """Nạp encoder (ResNet50 pretrained, đóng băng), decoder từ checkpoint và tokenizer."""
    ckpt = torch.load(resolve(ckpt_path), map_location=device, weights_only=True)
    tok = VietnameseTokenizer.load_vocab(resolve(vocab_path or ckpt["vocab_path"]))
    encoder = CNNEncoder(pretrained=True, freeze_backbone=True).to(device).eval()
    decoder = TransformerCaptionDecoder(**ckpt["config"]).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    return encoder, decoder, tok


@torch.no_grad()
def caption_image(path, encoder, decoder, tok, device, transform, beam_size=3, max_len=30):
    image = Image.open(path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)   # cùng transform với lúc train/eval
    _, spatial = encoder(x)                        # (1, 49, 2048)
    max_len = max_generation_len(decoder, max_len)
    if beam_size > 1:
        ids = decoder.beam_search(spatial, tok.bos_id, tok.eos_id,
                                  beam_size=beam_size, max_len=max_len)
    else:
        ids = decoder.greedy_generate(spatial, tok.bos_id, tok.eos_id, max_len=max_len)[0]
    return image, tok.decode(ids)


def save_figure(image, caption, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(caption, fontsize=11, wrap=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", nargs="+", required=True)
    p.add_argument("--vocab", default=None, help="mặc định lấy từ checkpoint")
    p.add_argument("--beam", type=int, default=3, help="1 = greedy")
    p.add_argument("--max-len", type=int, default=30)
    p.add_argument("--out", default=None, help="lưu ảnh kèm caption (nhiều ảnh: thêm hậu tố _1, _2...)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, tok = load_model(args.ckpt, args.vocab, device)
    transform = image_transform()

    for i, path in enumerate(args.image, 1):
        image, caption = caption_image(path, encoder, decoder, tok, device, transform,
                                       args.beam, args.max_len)
        print(f"{path}: {caption}")
        if args.out:
            out = Path(args.out)
            if len(args.image) > 1:
                out = out.with_name(f"{out.stem}_{i}{out.suffix}")
            save_figure(image, caption, out)
            print(f"  -> đã lưu {out}")


if __name__ == "__main__":
    main()
