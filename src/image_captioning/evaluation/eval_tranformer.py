"""Đánh giá mô hình sinh caption tiếng Việt trên KTVIC: BLEU-1..4, ROUGE-L, CIDEr.

Cài thư viện tính điểm (không cần Java vì không dùng PTBTokenizer/METEOR/SPICE):
    pip install pycocoevalcap

Ví dụ:
    python scripts/eval_caption.py --ckpt checkpoints/best.pt --split test --beam 3
"""
import argparse
import json
import re

import torch
from torch.utils.data import Subset

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # project root: chứa src/, data/, scripts/
sys.path.insert(0, str(ROOT / "src"))

# >>> Sửa đường dẫn package ở 4 dòng import này nếu project của bạn đặt khác <<<
from image_captioning.data.dataset import KTVICDataset, create_dataloader  # noqa: E402
from image_captioning.data.tokenizer_vi import VietnameseTokenizer  # noqa: E402
from image_captioning.models.cnn_encoder import CNNEncoder  # noqa: E402
from image_captioning.models.tranformer_decoder import TransformerCaptionDecoder  # noqa: E402

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def resolve(path) -> Path:
    """Đường dẫn tương đối được hiểu là tính từ project root, không phụ thuộc thư mục chạy lệnh."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


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


# ---------- sinh caption ----------
@torch.no_grad()
def generate_captions(encoder, decoder, tok, loader, device, beam_size=3, max_len=30):
    """Trả (hyps, refs) theo image_id. beam_size=1 dùng greedy theo batch (nhanh hơn)."""
    encoder.eval()
    decoder.eval()
    max_len = max_generation_len(decoder, max_len)
    hyps, refs = {}, {}
    for step, batch in enumerate(loader, 1):
        _, spatial = encoder(batch["images"].to(device))
        if beam_size > 1:
            ids_list = [
                decoder.beam_search(spatial[i:i + 1], tok.bos_id, tok.eos_id,
                                    beam_size=beam_size, max_len=max_len)
                for i in range(spatial.size(0))
            ]
        else:
            ids_list = decoder.greedy_generate(spatial, tok.bos_id, tok.eos_id, max_len=max_len)
        for image_id, ids, references in zip(batch["image_ids"].tolist(), ids_list, batch["references"]):
            hyps[image_id] = tok.decode(ids)
            refs[image_id] = references
        if step % 10 == 0:
            print(f"  đã sinh {len(hyps)} ảnh")
    return hyps, refs


# ---------- tính điểm ----------
def normalize_for_metric(text, tok, lowercase=True, strip_punct=True):
    """Chuẩn hóa trước khi tính điểm.

    Tokenizer giữ dấu câu dính vào âm tiết ("cỏ." khác "cỏ") và phân biệt hoa/thường,
    nên nếu không chuẩn hóa thì BLEU/CIDEr bị thấp oan.
    """
    text = " ".join(tok.tokenize(text))
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


def compute_metrics(refs, hyps, tok, lowercase=True, strip_punct=True):
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError as e:
        raise SystemExit("Thiếu thư viện: chạy `pip install pycocoevalcap`") from e

    norm = lambda s: normalize_for_metric(s, tok, lowercase, strip_punct)  # noqa: E731
    gts = {k: [norm(c) for c in refs[k]] for k in hyps}
    res = {k: [norm(hyps[k])] for k in hyps}

    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    cider, _ = Cider().compute_score(gts, res)
    rouge, _ = Rouge().compute_score(gts, res)
    return {
        "BLEU-1": bleu[0] * 100, "BLEU-2": bleu[1] * 100,
        "BLEU-3": bleu[2] * 100, "BLEU-4": bleu[3] * 100,
        "ROUGE-L": rouge * 100, "CIDEr": cider * 100,
    }


def caption_stats(hyps):
    """Thống kê nhanh để phát hiện model sinh caption chung chung / lặp lại."""
    caps = list(hyps.values())
    toks = [c.split() for c in caps]
    n_tok = sum(len(t) for t in toks)
    return {
        "num_images": len(caps),
        "avg_length": n_tok / max(1, len(caps)),
        "unique_captions": len(set(caps)),
        "unique_ratio": len(set(caps)) / max(1, len(caps)),
        "vocab_used": len({t for ts in toks for t in ts}),
        "unk_ratio": sum(t == "<unk>" for ts in toks for t in ts) / max(1, n_tok),
    }


def evaluate_split(encoder, decoder, tok, jsonl_path, device, beam_size=3, max_len=30,
                   batch_size=64, workers=0, limit=None, lowercase=True, strip_punct=True):
    """Hàm dùng lại được (ví dụ gọi cuối mỗi epoch trong train.py).

    Trả về (metrics, stats, predictions); predictions là list dict để lưu và đọc lỗi.
    """
    dataset = KTVICDataset(jsonl_path, vocab=tok, mode="image", project_root=ROOT)
    files = {row["image_id"]: row["image_filename"] for row in dataset.records}
    data = Subset(dataset, range(min(limit, len(dataset)))) if limit else dataset
    loader = create_dataloader(data, batch_size=batch_size, shuffle=False, num_workers=workers)

    hyps, refs = generate_captions(encoder, decoder, tok, loader, device, beam_size, max_len)
    metrics = compute_metrics(refs, hyps, tok, lowercase, strip_punct)
    predictions = [{"image_id": i, "image_filename": files[i],
                    "hypothesis": hyps[i], "references": refs[i]} for i in hyps]
    return metrics, caption_stats(hyps), predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--vocab", default=None, help="mặc định lấy từ checkpoint")
    p.add_argument("--beam", type=int, default=3, help="1 = greedy")
    p.add_argument("--max-len", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="chỉ đánh giá N ảnh đầu (chạy thử)")
    p.add_argument("--keep-case", action="store_true", help="không hạ chữ thường")
    p.add_argument("--keep-punct", action="store_true", help="không bỏ dấu câu")
    p.add_argument("--out-dir", default="eval_results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, tok = load_model(args.ckpt, args.vocab, device)

    jsonl_path = resolve(args.data_dir) / f"{args.split}.jsonl"
    metrics, stats, predictions = evaluate_split(
        encoder, decoder, tok, jsonl_path, device, beam_size=args.beam, max_len=args.max_len,
        batch_size=args.batch_size, workers=args.workers, limit=args.limit,
        lowercase=not args.keep_case, strip_punct=not args.keep_punct)

    print(f"\n=== Kết quả ({args.split}) ===")
    for k, v in metrics.items():
        print(f"{k:8s}: {v:6.2f}")
    print("\n=== Thống kê caption sinh ra ===")
    for k, v in stats.items():
        print(f"{k:16s}: {v:.3f}" if isinstance(v, float) else f"{k:16s}: {v}")

    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"metrics_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "stats": stats,
                   "settings": {"ckpt": args.ckpt, "split": args.split, "beam": args.beam,
                                "max_len": args.max_len, "lowercase": not args.keep_case,
                                "strip_punct": not args.keep_punct}},
                  f, ensure_ascii=False, indent=2)
    with open(out / f"predictions_{args.split}.jsonl", "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nĐã lưu {out / f'metrics_{args.split}.json'} và {out / f'predictions_{args.split}.jsonl'}")


if __name__ == "__main__":
    main()