"""Vẽ biểu đồ loss train/val.

Cài:  pip install matplotlib

Cách 1 - từ history.json (do train.py mới ghi ra mỗi epoch):
    python scripts/plot_loss.py --history checkpoints/history.json

Cách 2 - từ log in trên terminal (cho lần train cũ chưa có history.json):
    python scripts/plot_loss.py --log train_log.txt
Lưu log bằng cách copy phần terminal vào file txt, hoặc chạy train với:
    python scripts/train.py ... | Tee-Object train_log.txt
"""
import argparse
import json
import re
from pathlib import Path

LOG_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*\d+\s*\|\s*train\s+([0-9.]+)\s*\|\s*val\s+([0-9.]+)")


def load_history(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_any(path: Path) -> str:
    """Đọc file log, tự xử lý UTF-16 (PowerShell hay ghi file kiểu này khi dùng '>')."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="ignore")
    return raw.decode("utf-8-sig", errors="ignore")


def parse_log(path: Path) -> list[dict]:
    seen = {}
    for m in LOG_RE.finditer(read_text_any(path)):
        epoch = int(m.group(1))
        seen[epoch] = {"epoch": epoch, "train_loss": float(m.group(2)), "val_loss": float(m.group(3))}
    return [seen[e] for e in sorted(seen)]


def plot(history: list[dict], out_path: Path, show: bool = True):
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history]
    best_i = min(range(len(val)), key=val.__getitem__)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, train, "o-", label="Train loss")
    ax1.plot(epochs, val, "s-", label="Val loss")
    ax1.axvline(epochs[best_i], color="gray", linestyle="--", alpha=0.7)
    ax1.annotate(f"best: epoch {epochs[best_i]}\nval {val[best_i]:.3f}",
                 (epochs[best_i], val[best_i]), textcoords="offset points",
                 xytext=(10, 15), fontsize=9)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train / Val loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    gap = [v - t for v, t in zip(val, train)]
    ax2.plot(epochs, gap, "o-", color="tab:red")
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val loss − Train loss")
    ax2.set_title("Khoảng cách val − train (tăng dần = overfit)")
    ax2.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(0.5, 0.015, "Lưu ý: train loss có label smoothing 0.1 nên cao hơn val loss một đoạn cố định; "
             "hãy nhìn xu hướng, đừng so giá trị tuyệt đối", ha="center", va="bottom", fontsize=9)
    fig.savefig(out_path, dpi=150)
    print(f"Đã lưu {out_path}")
    print(f"Epoch tốt nhất theo val loss: {epochs[best_i]} (val {val[best_i]:.4f}); "
          f"tổng {len(epochs)} epoch")
    if show:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--history", help="file history.json do train.py ghi")
    src.add_argument("--log", help="file log copy từ terminal")
    p.add_argument("--out", default=None, help="đường dẫn ảnh PNG (mặc định: loss_curve.png cạnh file nguồn)")
    p.add_argument("--no-show", action="store_true", help="chỉ lưu ảnh, không mở cửa sổ")
    args = p.parse_args()

    source = Path(args.history or args.log)
    history = load_history(source) if args.history else parse_log(source)
    if not history:
        raise SystemExit("Không đọc được epoch nào. Dòng log cần có dạng: "
                         "'Epoch 3/30 | train 3.1234 | val 3.4567 ...'")
    out = Path(args.out) if args.out else source.with_name("loss_curve.png")
    plot(history, out, show=not args.no_show)


if __name__ == "__main__":
    main()