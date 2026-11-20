"""Huấn luyện CNNEncoder (ResNet50) + TransformerCaptionDecoder trên KTVIC.

Cần có sẵn data/processed/{train,val}.jsonl và vocab.json (do prepare_ktvic.py tạo).

Ví dụ:
  python scripts/train.py --data-dir data/processed --out-dir checkpoints
"""
import argparse
import json
import math
import time

import torch
import torch.nn as nn

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # project root: chứa src/, data/, scripts/
sys.path.insert(0, str(ROOT / "src"))

# >>> Sửa đường dẫn package ở 4 dòng import này nếu project của bạn đặt khác <<<
from image_captioning.data.dataset import KTVICDataset, create_dataloader  # noqa: E402
from image_captioning.data.tokenizer_vi import VietnameseTokenizer  # noqa: E402
from image_captioning.models.cnn_encoder import CNNEncoder  # noqa: E402
from image_captioning.models.tranformer_decoder import TransformerCaptionDecoder  # noqa: E402


def resolve(path) -> Path:
    """Đường dẫn tương đối được hiểu là tính từ project root, không phụ thuộc thư mục chạy lệnh."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def max_generation_len(decoder, requested: int) -> int:
    """Giới hạn độ dài sinh theo bảng positional encoding của decoder (bos + max_len token)."""
    return min(requested, decoder.pos_enc.pe.size(1) - 1)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--patience", type=int, default=5, help="dừng sớm nếu val loss không giảm")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=0, help="Windows: giữ 0 nếu gặp lỗi DataLoader")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def longest_caption(tok, *datasets):
    """Độ dài (kể cả BOS/EOS) của caption dài nhất. collate_fn không cắt caption nên
    decoder phải có max_len đủ lớn, nếu không positional encoding sẽ lỗi."""
    return max(len(tok.tokenize(c)) + 2
               for ds in datasets for row in ds.records for c in row["captions"])


def lr_lambda(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_epoch(encoder, decoder, loader, opt, sched, scaler, crit, device, use_amp):
    encoder.train()   # backbone đóng băng vẫn tự giữ ở eval()
    decoder.train()
    total = 0.0
    for step, batch in enumerate(loader, 1):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)     # <bos> w1 ... wn   (đã shift trước khi pad)
        target_ids = batch["target_ids"].to(device)   # w1 ... wn <eos>
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, spatial = encoder(images)               # encoder trả (global, spatial)
            logits = decoder.decode(input_ids, decoder.encode_image(spatial))
        loss = crit(logits.float().reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += loss.item()
        if step % 50 == 0:
            print(f"  step {step}/{len(loader)}  loss {total / step:.4f}  lr {sched.get_last_lr()[0]:.2e}")
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(encoder, decoder, loader, device, pad_id):
    encoder.eval()
    decoder.eval()
    crit = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total, n_tok = 0.0, 0
    for batch in loader:
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        _, spatial = encoder(images)
        logits = decoder.decode(input_ids, decoder.encode_image(spatial))
        total += crit(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)).item()
        n_tok += (target_ids != pad_id).sum().item()
    return total / max(1, n_tok)


@torch.no_grad()
def show_samples(encoder, decoder, tok, loader, device, n=3):
    batch = next(iter(loader))
    _, spatial = encoder(batch["images"][:n].to(device))
    preds = decoder.greedy_generate(spatial, tok.bos_id, tok.eos_id,
                                    max_len=max_generation_len(decoder, 30))
    for i, pred in enumerate(preds):
        print(f"  [GT  ] {tok.decode(batch['target_ids'][i].tolist())}")
        print(f"  [PRED] {tok.decode(pred)}")


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device}  AMP: {use_amp}")

    data_dir = resolve(args.data_dir)
    vocab_path = data_dir / "vocab.json"
    tok = VietnameseTokenizer.load_vocab(vocab_path)  # vocab chỉ xây từ train (prepare_ktvic.py)
    V = len(tok)

    train_ds = KTVICDataset(data_dir / "train.jsonl", vocab=tok, mode="caption", project_root=ROOT)
    val_ds = KTVICDataset(data_dir / "val.jsonl", vocab=tok, mode="caption", project_root=ROOT)
    train_loader = create_dataloader(train_ds, args.batch_size, shuffle=True,
                                     num_workers=args.workers, seed=args.seed)
    val_loader = create_dataloader(val_ds, args.batch_size, shuffle=False,
                                   num_workers=args.workers, seed=args.seed)

    max_len = max(longest_caption(tok, train_ds, val_ds), 64)
    print(f"Train: {len(train_ds)} cặp ảnh-caption | Val: {len(val_ds)} | Vocab: {V} | max_len: {max_len}")

    encoder = CNNEncoder(pretrained=True, freeze_backbone=True).to(device)
    dec_cfg = dict(
        vocab_size=V, enc_dim=CNNEncoder.encoder_dim, d_model=args.d_model,
        nhead=args.heads, num_layers=args.layers, dim_ff=args.dim_ff,
        dropout=args.dropout, max_len=max_len, max_img_tokens=256, pad_id=tok.pad_id,
    )
    decoder = TransformerCaptionDecoder(**dec_cfg).to(device)
    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()) / 1e6:.1f}M")

    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, args.warmup_steps, total_steps))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    crit = nn.CrossEntropyLoss(ignore_index=tok.pad_id, label_smoothing=0.1)

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best, bad = float("inf"), 0
    history = []  # lưu loss từng epoch để vẽ biểu đồ (history.json)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(encoder, decoder, train_loader, opt, sched, scaler, crit, device, use_amp)
        va = evaluate(encoder, decoder, val_loader, device, tok.pad_id)
        print(f"Epoch {epoch}/{args.epochs} | train {tr:.4f} | val {va:.4f} "
              f"(ppl {math.exp(min(va, 20)):.1f}) | {time.time() - t0:.0f}s")
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va,
                        "lr": sched.get_last_lr()[0]})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        show_samples(encoder, decoder, tok, val_loader, device)

        ckpt = {"decoder": decoder.state_dict(), "config": dec_cfg, "epoch": epoch,
                "val_loss": va, "vocab_path": str(vocab_path)}
        torch.save(ckpt, out_dir / "last.pt")
        if va < best:
            best, bad = va, 0
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> lưu best.pt (val {va:.4f})")
        else:
            bad += 1
            if bad >= args.patience:
                print("Dừng sớm: val loss không cải thiện.")
                break


if __name__ == "__main__":
    main()
