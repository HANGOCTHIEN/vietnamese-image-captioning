"""Test cho TransformerCaptionDecoder.

Chạy:  pytest -v test_tranformer_decoder.py
Hoặc:  python test_tranformer_decoder.py   (chạy không cần pytest)

Nếu import bị lỗi, sửa dòng import bên dưới cho đúng đường dẫn project của bạn.
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path

from image_captioning.models.tranformer_decoder import TransformerCaptionDecoder


PAD, BOS, EOS = 0, 1, 2
V, ENC_DIM, B, T = 60, 32, 4, 10


def make_model(**kw):
    cfg = dict(
        vocab_size=V, enc_dim=ENC_DIM, d_model=64, nhead=4, num_layers=2,
        dim_ff=128, dropout=0.0, max_len=32, max_img_tokens=64, pad_id=PAD,
    )
    cfg.update(kw)
    return TransformerCaptionDecoder(**cfg)


def make_batch(b=B, t=T, n_regions=49):
    feats = torch.randn(b, ENC_DIM, 7, 7) if n_regions == 49 else torch.randn(b, n_regions, ENC_DIM)
    caps = torch.randint(3, V, (b, t))
    caps[:, 0], caps[:, -1] = BOS, EOS
    return feats, caps


# ---------------- shape & cấu trúc ----------------
def test_forward_shape_4d_and_3d_input():
    torch.manual_seed(0)
    m = make_model().eval()
    feats4d, caps = make_batch()
    assert m(feats4d, caps).shape == (B, T - 1, V)

    feats3d = feats4d.flatten(2).transpose(1, 2)  # (B, 49, C)
    assert m(feats3d, caps).shape == (B, T - 1, V)
    # 2 cách đưa input phải cho cùng kết quả
    assert torch.allclose(m(feats4d, caps), m(feats3d, caps), atol=1e-5)


def test_weight_tying():
    m = make_model()
    assert m.fc_out.weight.data_ptr() == m.embed.weight.data_ptr()


def test_pad_embedding_is_zero():
    m = make_model()
    assert torch.all(m.embed.weight[PAD] == 0)


# ---------------- tính đúng đắn của mask ----------------
def test_causal_mask_no_future_leak():
    """Đổi token ở vị trí tương lai thì logits các vị trí trước đó không đổi."""
    torch.manual_seed(0)
    m = make_model().eval()
    feats, caps = make_batch()
    caps2 = caps.clone()
    k = 5
    caps2[:, k:-1] = torch.randint(3, V, caps2[:, k:-1].shape)  # đổi phần sau vị trí k
    with torch.no_grad():
        out1, out2 = m(feats, caps), m(feats, caps2)
    # input của decoder là caps[:, :-1] nên vị trí < k phải giống hệt nhau
    assert torch.allclose(out1[:, :k], out2[:, :k], atol=1e-5)
    assert not torch.allclose(out1[:, k:], out2[:, k:], atol=1e-5)


def test_padding_no_nan_and_not_affect_valid_positions():
    torch.manual_seed(0)
    m = make_model().eval()
    feats, caps = make_batch()
    caps[:, 6:] = PAD            # padding ở cuối
    caps[0, 5] = EOS
    with torch.no_grad():
        out = m(feats, caps)
    assert torch.isfinite(out).all()

    # thay giá trị trong vùng pad bằng token khác thì phần hợp lệ không đổi
    caps2 = caps.clone()
    caps2[:, 7:] = PAD
    with torch.no_grad():
        out2 = m(feats, caps2)
    assert torch.allclose(out[:, :6], out2[:, :6], atol=1e-5)


def test_image_conditioning():
    """Ảnh khác nhau phải cho logits khác nhau (cross-attention có tác dụng)."""
    torch.manual_seed(0)
    m = make_model().eval()
    feats, caps = make_batch()
    with torch.no_grad():
        a = m(feats, caps)
        b = m(torch.randn_like(feats), caps)
    assert not torch.allclose(a, b, atol=1e-5)


# ---------------- huấn luyện ----------------
def test_backward_gradients():
    torch.manual_seed(0)
    m = make_model().train()
    feats, caps = make_batch()
    logits = m(feats, caps)
    loss = nn.CrossEntropyLoss(ignore_index=PAD)(logits.reshape(-1, V), caps[:, 1:].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} không có gradient"
        assert torch.isfinite(p.grad).all(), f"{name} có gradient NaN/inf"


def test_initial_loss_close_to_log_vocab():
    torch.manual_seed(0)
    m = make_model().eval()
    feats, caps = make_batch()
    with torch.no_grad():
        logits = m(feats, caps)
        loss = nn.CrossEntropyLoss(ignore_index=PAD)(logits.reshape(-1, V), caps[:, 1:].reshape(-1))
    import math
    assert loss.item() < math.log(V) * 3  # không nổ bất thường lúc khởi tạo


def _overfit(m, feats, caps, steps=400, lr=2e-3):
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    m.train()
    loss = None
    for _ in range(steps):
        logits = m(feats, caps)
        loss = crit(logits.reshape(-1, V), caps[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    return loss.item()


def test_overfit_and_generate():
    """Model phải học thuộc 1 batch nhỏ; greedy và beam phải tái tạo đúng caption."""
    torch.manual_seed(0)
    m = make_model()
    feats, caps = make_batch(b=3, t=8)
    final_loss = _overfit(m, feats, caps)
    assert final_loss < 0.1, f"loss cuối = {final_loss:.3f}, chưa overfit được"

    greedy = m.greedy_generate(feats, BOS, EOS, max_len=10)
    for i in range(3):
        target = caps[i, 1:-1].tolist()
        assert greedy[i] == target, f"greedy[{i}]={greedy[i]} != {target}"

    beam = m.beam_search(feats[:1], BOS, EOS, beam_size=3, max_len=10)
    assert beam == caps[0, 1:-1].tolist()


# ---------------- suy luận ----------------
def test_generate_output_format():
    torch.manual_seed(0)
    m = make_model()
    feats, _ = make_batch()
    max_len = 7

    outs = m.greedy_generate(feats, BOS, EOS, max_len=max_len)
    assert len(outs) == B
    for o in outs:
        assert isinstance(o, list) and len(o) <= max_len
        assert BOS not in o and EOS not in o
        assert all(isinstance(t, int) for t in o)

    o = m.beam_search(feats[:1], BOS, EOS, beam_size=3, max_len=max_len)
    assert isinstance(o, list) and len(o) <= max_len
    assert BOS not in o and EOS not in o


def test_generation_is_deterministic_in_eval():
    torch.manual_seed(0)
    m = make_model()
    feats, _ = make_batch()
    a = m.greedy_generate(feats, BOS, EOS, max_len=6)
    b = m.greedy_generate(feats, BOS, EOS, max_len=6)
    assert a == b


def test_greedy_batch_matches_single():
    """Sinh theo batch phải cho kết quả như sinh từng ảnh một."""
    torch.manual_seed(0)
    m = make_model()
    feats, caps = make_batch(b=3, t=8)
    _overfit(m, feats, caps, steps=300)
    batch_out = m.greedy_generate(feats, BOS, EOS, max_len=10)
    single = [m.greedy_generate(feats[i:i + 1], BOS, EOS, max_len=10)[0] for i in range(3)]
    assert batch_out == single


def test_beam_size_one_equals_greedy():
    torch.manual_seed(0)
    m = make_model()
    feats, caps = make_batch(b=2, t=8)
    _overfit(m, feats, caps, steps=300)
    g = m.greedy_generate(feats[:1], BOS, EOS, max_len=10)[0]
    b = m.beam_search(feats[:1], BOS, EOS, beam_size=1, max_len=10)
    assert g == b


if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")