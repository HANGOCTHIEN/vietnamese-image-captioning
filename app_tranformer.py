"""Giao diện web: kéo thả ảnh vào, bên dưới hiện caption model sinh ra và caption tham chiếu (Ground Truth).

Cài:   pip install gradio
Chạy:  python app.py --ckpt checkpoints/best.pt
Đặt file này ở thư mục gốc project (ngang hàng với src/, data/, checkpoints/).

Caption tham chiếu chỉ có khi ảnh nằm trong KTVIC (train/val/test.jsonl). App tìm theo
tên file ảnh, nên hãy kéo ảnh gốc của bộ dữ liệu (đừng đổi tên hay dán từ clipboard).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402
import torch  # noqa: E402

# Dùng lại hàm trong script predict của bạn (đổi đường dẫn import nếu bạn đã đổi tên/chuyển file)
from image_captioning.predict.pre_tranformer import (  # noqa: E402
    caption_image, image_transform, load_model)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--vocab", default=None, help="mặc định lấy từ checkpoint")
    p.add_argument("--data-dir", default="data/processed", help="thư mục chứa train/val/test.jsonl")
    p.add_argument("--max-len", type=int, default=30)
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def under_root(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def build_reference_index(data_dir: Path) -> dict:
    """{tên file ảnh (chữ thường): (split, [caption tham chiếu])} từ các file jsonl của KTVIC."""
    index = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                name = row.get("image_filename") or Path(row["image_path"]).name
                index[name.lower()] = (split, list(row["captions"]))
    return index


def format_references(entry) -> str:
    if entry is None:
        return ("Không tìm thấy ảnh này trong KTVIC (train/val/test) nên không có caption tham chiếu.\n"
                "Hãy kéo ảnh gốc của bộ dữ liệu, giữ nguyên tên file.")
    split, captions = entry
    header = f"[Ảnh thuộc tập {split}]"
    if split == "train":
        header += " Model đã học ảnh này khi train nên không dùng để đánh giá khách quan."
    return header + "\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(captions, 1))


def main():
    args = get_args()
    ckpt = under_root(args.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"Không thấy checkpoint: {ckpt}\nHãy train trước bằng train.py.")
    vocab = str(under_root(args.vocab)) if args.vocab else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}. Đang nạp model...")
    encoder, decoder, tok = load_model(str(ckpt), vocab, device)  # nạp 1 lần lúc khởi động
    transform = image_transform()

    references = build_reference_index(under_root(args.data_dir))
    print(f"Đã nạp caption tham chiếu của {len(references)} ảnh. Sẵn sàng.")

    def generate(image_path, beam):
        if image_path is None:
            return "", ""
        _, caption = caption_image(image_path, encoder, decoder, tok, device,
                                   transform, int(beam), args.max_len)
        entry = references.get(Path(image_path).name.lower())
        return caption or "(model chưa sinh được caption)", format_references(entry)

    with gr.Blocks(title="Sinh caption tiếng Việt") as demo:
        gr.Markdown("## Sinh caption tiếng Việt cho ảnh\nKéo thả ảnh vào khung bên dưới (hoặc bấm để chọn).")
        image = gr.Image(type="filepath", label="Ảnh", height=420)
        beam = gr.Slider(1, 5, value=3, step=1, label="Beam size (1 = greedy)")
        caption = gr.Textbox(label="Caption mô hình sinh ra", lines=2, interactive=False)
        truth = gr.Textbox(label="Caption tham chiếu (Ground Truth)", lines=7, interactive=False)
        outputs = [caption, truth]
        image.change(generate, [image, beam], outputs)   # tự chạy khi thả ảnh
        beam.release(generate, [image, beam], outputs)   # đổi beam thì sinh lại

    demo.launch(server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()