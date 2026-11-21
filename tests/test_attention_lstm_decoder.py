import json
from pathlib import Path

from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from image_captioning.data.dataset import KTVICDataset, create_dataloader
from image_captioning.data.tokenizer_vi import VietnameseTokenizer
from image_captioning.models.attention_lstm_decoder import (
    AttentionLSTMCaptioner,
    AttentionLSTMDecoder,
    BahdanauAttention,
    SpecialTokens,
)
from image_captioning.models.cnn_encoder import CNNEncoder


# ==============================================================================
# 1. KIỂM THỬ ĐỒNG BỘ TOKEN ĐẶC BIỆT & BAHADANAU ATTENTION
# ==============================================================================


def test_special_tokens_synced_with_tokenizer():
    """Đảm bảo SpecialTokens khớp 100% với VietnameseTokenizer của partner."""
    assert SpecialTokens.PAD_IDX == VietnameseTokenizer.pad_id == 0
    assert SpecialTokens.BOS_IDX == VietnameseTokenizer.bos_id == 1
    assert SpecialTokens.SOS_IDX == VietnameseTokenizer.bos_id == 1
    assert SpecialTokens.EOS_IDX == VietnameseTokenizer.eos_id == 2
    assert SpecialTokens.UNK_IDX == VietnameseTokenizer.unk_id == 3

    assert SpecialTokens.PAD_TOKEN == VietnameseTokenizer.special_tokens["pad"]
    assert SpecialTokens.BOS_TOKEN == VietnameseTokenizer.special_tokens["bos"]
    assert SpecialTokens.EOS_TOKEN == VietnameseTokenizer.special_tokens["eos"]
    assert SpecialTokens.UNK_TOKEN == VietnameseTokenizer.special_tokens["unk"]


def test_bahdanau_attention_shapes_distribution_and_grad():
    """Kiểm tra shape, tổng phân phối xác suất alpha = 1.0 và gradient của Attention."""
    batch_size, num_pixels, encoder_dim, decoder_dim = 4, 49, 2048, 512
    attn = BahdanauAttention(
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        attention_dim=256,
    )

    spatial_features = torch.randn(
        batch_size, num_pixels, encoder_dim, requires_grad=True
    )
    decoder_hidden = torch.randn(batch_size, decoder_dim, requires_grad=True)

    context, alpha = attn(spatial_features, decoder_hidden)

    assert context.shape == (batch_size, encoder_dim)
    assert alpha.shape == (batch_size, num_pixels)
    assert (alpha >= 0.0).all()
    assert torch.allclose(alpha.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    loss = context.sum()
    loss.backward()
    assert spatial_features.grad is not None
    assert decoder_hidden.grad is not None


# ==============================================================================
# 2. KIỂM THỬ KIẾN TRÚC DECODER & CÁC CẤU HÌNH ABLATION
# ==============================================================================


@pytest.mark.parametrize(
    "use_attention,use_deep_output",
    [
        (False, False),  # E0: Baseline CNN + LSTM (không Attention)
        (True, False),   # E1: CNN + Bahdanau Attention LSTM
        (True, True),    # E2: CNN + Attention LSTM + Deep Output
        (False, True),   # E3: CNN + LSTM + Deep Output
    ],
)
@pytest.mark.parametrize("tf_ratio", [0.0, 0.5, 1.0])
def test_decoder_ablation_configs_forward_and_backward(
    use_attention: bool,
    use_deep_output: bool,
    tf_ratio: float,
):
    """Kiểm tra forward/backward qua toàn bộ tổ hợp Ablation và Teacher Forcing."""
    batch_size, max_len, vocab_size = 3, 9, 40
    decoder = AttentionLSTMDecoder(
        vocab_size=vocab_size,
        embed_dim=64,
        decoder_dim=128,
        attention_dim=64,
        encoder_dim=2048,
        use_attention=use_attention,
        use_deep_output=use_deep_output,
    )

    global_features = torch.randn(batch_size, 2048)
    spatial_features = torch.randn(batch_size, 49, 2048)

    captions = torch.randint(4, vocab_size, (batch_size, max_len))
    captions[:, 0] = SpecialTokens.BOS_IDX
    captions[0, 5] = SpecialTokens.EOS_IDX
    captions[0, 6:] = SpecialTokens.PAD_IDX
    captions[1:, -1] = SpecialTokens.EOS_IDX

    logits, alphas = decoder(
        global_features,
        spatial_features,
        captions,
        teacher_forcing_ratio=tf_ratio,
    )

    assert logits.shape == (batch_size, max_len - 1, vocab_size)
    assert alphas.shape == (batch_size, max_len - 1, 49)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(alphas).all()

    if not use_attention:
        assert torch.all(alphas == 0.0)

    ce_loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        captions[:, 1:].reshape(-1),
        ignore_index=SpecialTokens.PAD_IDX,
    )
    ds_loss = decoder.compute_doubly_stochastic_loss(
        alphas, captions, lambda_c=0.5
    )
    total_loss = ce_loss + ds_loss
    total_loss.backward()

    assert torch.isfinite(total_loss)
    assert decoder.fc.weight.grad is not None
    assert torch.isfinite(decoder.fc.weight.grad).all()


def test_doubly_stochastic_loss_pad_masking_and_validation():
    """Đảm bảo bước <pad> không bị cộng vào Doubly Stochastic Loss và bắt lỗi đầu vào."""
    decoder = AttentionLSTMDecoder(vocab_size=20, use_attention=True)

    # Chuỗi không pad: [<bos>, 4, <eos>] -> 2 bước giải mã
    caps_short = torch.tensor([[SpecialTokens.BOS_IDX, 4, SpecialTokens.EOS_IDX]])
    alphas_short = torch.tensor([[[0.6, 0.4], [0.3, 0.7]]])

    # Chuỗi có pad phía sau: [<bos>, 4, <eos>, <pad>, <pad>] -> 4 bước giải mã
    caps_padded = torch.tensor(
        [
            [
                SpecialTokens.BOS_IDX,
                4,
                SpecialTokens.EOS_IDX,
                SpecialTokens.PAD_IDX,
                SpecialTokens.PAD_IDX,
            ]
        ]
    )
    alphas_padded = torch.tensor(
        [[[0.6, 0.4], [0.3, 0.7], [0.99, 0.01], [0.05, 0.95]]]
    )

    loss_short = decoder.compute_doubly_stochastic_loss(
        alphas_short, caps_short, lambda_c=1.0
    )
    loss_padded = decoder.compute_doubly_stochastic_loss(
        alphas_padded, caps_padded, lambda_c=1.0
    )
    assert torch.allclose(loss_short, loss_padded, atol=1e-6)

    # Khi use_attention=False, penalty phải bằng 0
    no_attn_decoder = AttentionLSTMDecoder(vocab_size=20, use_attention=False)
    zero_loss = no_attn_decoder.compute_doubly_stochastic_loss(
        alphas_short, caps_short, lambda_c=1.0
    )
    assert zero_loss.item() == 0.0

    # Kiểm tra các guard validation
    with pytest.raises(ValueError, match="lambda_c"):
        decoder.compute_doubly_stochastic_loss(
            alphas_short, caps_short, lambda_c=-0.1
        )
    with pytest.raises(ValueError, match="alphas"):
        decoder.compute_doubly_stochastic_loss(
            torch.zeros(1, 2), caps_short, lambda_c=1.0
        )
    with pytest.raises(ValueError, match="captions"):
        decoder.compute_doubly_stochastic_loss(
            alphas_short, torch.zeros(3, dtype=torch.long), lambda_c=1.0
        )
    with pytest.raises(ValueError, match="Shape"):
        decoder.compute_doubly_stochastic_loss(
            alphas_padded, caps_short, lambda_c=1.0
        )


# ==============================================================================
# 3. KIỂM THỬ TRIMMING, GREEDY SEARCH, BEAM SEARCH & TRẠNG THÁI SUBMODULE
# ==============================================================================


def test_trim_sequences_and_decode_predictions():
    """Đảm bảo cắt sạch token rác sau <eos> đầu tiên và decode ra tiếng Việt chuẩn."""
    vocab = VietnameseTokenizer(min_freq=1).build_vocab(
        ["Một con mèo đang ngủ trên ghế."]
    )
    decoder = AttentionLSTMDecoder(vocab_size=len(vocab))

    id_mot = vocab.token_to_id["Một"]
    id_con = vocab.token_to_id["con"]
    id_meo = vocab.token_to_id["mèo"]
    id_ngu = vocab.token_to_id["ngủ"]

    # Mẫu 1 có token rác sau <eos>; Mẫu 2 có <bos> ở đầu và <pad> ở cuối
    raw_predictions = torch.tensor(
        [
            [id_mot, id_con, id_meo, SpecialTokens.EOS_IDX, id_ngu, SpecialTokens.PAD_IDX],
            [SpecialTokens.BOS_IDX, id_meo, id_ngu, SpecialTokens.EOS_IDX, SpecialTokens.PAD_IDX, SpecialTokens.PAD_IDX],
        ],
        dtype=torch.long,
    )

    trimmed = decoder.trim_sequences(raw_predictions)
    assert trimmed == [[id_mot, id_con, id_meo], [id_meo, id_ngu]]

    decoded_texts = decoder.decode_predictions(raw_predictions, vocab)
    assert decoded_texts == ["Một con mèo", "mèo ngủ"]


def test_generate_greedy_disables_dropout_and_preserves_submodule_states():
    """
    Đảm bảo generate_greedy():
    1. Tắt Dropout trong lúc sinh câu (cho kết quả deterministic dù model đang ở train()).
    2. Khôi phục chính xác trạng thái train của decoder và trạng thái eval của frozen backbone.
    """
    encoder = CNNEncoder(pretrained=False, freeze_backbone=True)
    decoder = AttentionLSTMDecoder(
        vocab_size=50,
        encoder_dim=encoder.encoder_dim,
        dropout=0.5,
    )
    model = AttentionLSTMCaptioner(encoder=encoder, decoder=decoder)

    # Đặt toàn bộ model vào train mode (như khi log caption giữa epoch)
    model.train()
    assert model.training is True
    assert decoder.dropout.training is True
    assert encoder.backbone.training is False

    images = torch.randn(2, 3, 224, 224)
    ids_run1, alphas_run1 = model.generate_greedy(images, max_len=10)
    ids_run2, alphas_run2 = model.generate_greedy(images, max_len=10)

    # Nếu Dropout không được tắt tạm thời, 2 lần chạy với dropout=0.5 sẽ lệch nhau
    assert torch.equal(ids_run1, ids_run2)
    assert torch.allclose(alphas_run1, alphas_run2)

    # Trạng thái gốc của từng submodule phải được giữ nguyên sau khi generate xong
    assert model.training is True
    assert decoder.training is True
    assert decoder.dropout.training is True
    assert encoder.backbone.training is False

    assert ids_run1.shape[0] == 2
    assert 1 <= ids_run1.shape[1] <= 10
    assert alphas_run1.shape == (2, ids_run1.shape[1], 49)


def test_beam_search_output_and_edge_cases():
    """Kiểm tra Beam Search (bao gồm cả trường hợp chạm max_len chưa gặp <eos>)."""
    encoder = CNNEncoder(pretrained=False, freeze_backbone=True)
    decoder = AttentionLSTMDecoder(vocab_size=50, encoder_dim=encoder.encoder_dim)
    model = AttentionLSTMCaptioner(encoder=encoder, decoder=decoder)
    model.train()

    image = torch.randn(1, 3, 224, 224)

    # Trường hợp thông thường với beam_size = 3
    seq, alphas = model.generate_beam_search(image, beam_size=3, max_len=8)
    assert isinstance(seq, list)
    assert all(isinstance(tok, int) for tok in seq)
    assert SpecialTokens.BOS_IDX not in seq
    assert SpecialTokens.EOS_IDX not in seq
    assert len(seq) <= 8
    assert alphas.shape == (len(seq), 49)
    assert encoder.backbone.training is False

    # Trường hợp max_len = 1 (ép xét các beam chưa gặp <eos> tại bước dừng)
    seq_short, alphas_short = model.generate_beam_search(
        image, beam_size=3, max_len=1
    )
    assert len(seq_short) <= 1
    assert alphas_short.shape == (len(seq_short), 49)

    # Kiểm tra các lỗi tham số
    with pytest.raises(ValueError, match="beam_size"):
        model.generate_beam_search(image, beam_size=0)
    with pytest.raises(ValueError, match="beam_size"):
        model.generate_beam_search(image, beam_size=51)
    with pytest.raises(ValueError, match="max_len"):
        model.generate_beam_search(image, beam_size=3, max_len=0)
    with pytest.raises(ValueError, match="length_penalty"):
        model.generate_beam_search(image, beam_size=3, length_penalty=-0.5)
    with pytest.raises(ValueError, match="batch_size=1"):
        model.generate_beam_search(torch.randn(2, 3, 224, 224), beam_size=3)


def test_input_validation_guards():
    """Kiểm tra các chốt chặn lỗi hình dạng tensor, dtype, device và BOS/EOS."""
    encoder = CNNEncoder(pretrained=False)
    wrong_decoder = AttentionLSTMDecoder(vocab_size=20, encoder_dim=512)
    with pytest.raises(ValueError, match="encoder_dim"):
        AttentionLSTMCaptioner(encoder=encoder, decoder=wrong_decoder)

    decoder = AttentionLSTMDecoder(vocab_size=20, encoder_dim=2048)
    g_feat = torch.randn(2, 2048)
    s_feat = torch.randn(2, 49, 2048)

    # Lệch batch size giữa global và spatial features
    with pytest.raises(ValueError, match="Batch size"):
        decoder.generate_greedy(torch.randn(3, 2048), s_feat)

    # Captions thiếu BOS ở vị trí đầu
    no_bos = torch.tensor(
        [[5, 6, SpecialTokens.EOS_IDX], [5, 7, SpecialTokens.EOS_IDX]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="BOS/SOS"):
        decoder(g_feat, s_feat, no_bos)

    # Captions thiếu EOS
    no_eos = torch.tensor(
        [[SpecialTokens.BOS_IDX, 5, 6], [SpecialTokens.BOS_IDX, 7, 8]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="EOS"):
        decoder(g_feat, s_feat, no_eos)

    # Captions sai dtype
    wrong_dtype = torch.tensor(
        [
            [SpecialTokens.BOS_IDX, 5, SpecialTokens.EOS_IDX],
            [SpecialTokens.BOS_IDX, 6, SpecialTokens.EOS_IDX],
        ],
        dtype=torch.int32,
    )
    with pytest.raises(TypeError, match="torch.long"):
        decoder(g_feat, s_feat, wrong_dtype)

    # Token ID vượt quá vocab_size
    out_of_range = torch.tensor(
        [
            [SpecialTokens.BOS_IDX, 99, SpecialTokens.EOS_IDX],
            [SpecialTokens.BOS_IDX, 6, SpecialTokens.EOS_IDX],
        ],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="ngoài phạm vi"):
        decoder(g_feat, s_feat, out_of_range)


# ==============================================================================
# 4. KIỂM THỬ TÍCH HỢP TOÀN BỘ PIPELINE VỚI KTVICDataset & CNNEncoder
# ==============================================================================


@pytest.fixture
def ktvic_mock_workspace(tmp_path: Path):
    """Tạo bộ dữ liệu mẫu chuẩn schema JSONL của prepare_ktvic.py."""
    images_dir = tmp_path / "data" / "images" / "train"
    processed_dir = tmp_path / "data" / "processed"
    images_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    records = [
        {
            "image_id": 1,
            "image_filename": "ktvic_001.jpg",
            "image_path": "data/images/train/ktvic_001.jpg",
            "captions": [
                "Một người đàn ông đang đá bóng trên sân cỏ.",
                "Cầu thủ mặc áo đỏ sút quả bóng.",
            ],
            "annotation_ids": [101, 102],
            "original_split": "train",
        },
        {
            "image_id": 2,
            "image_filename": "ktvic_002.jpg",
            "image_path": "data/images/train/ktvic_002.jpg",
            "captions": [
                "Hai đứa trẻ đang chơi đùa trong công viên.",
                "Các em nhỏ chạy nhảy trên bãi cỏ xanh.",
            ],
            "annotation_ids": [201, 202],
            "original_split": "train",
        },
    ]

    for row in records:
        img = Image.new("RGB", (256, 256), color=(100, 150, 200))
        img.save(tmp_path / row["image_path"])

    train_jsonl = processed_dir / "train.jsonl"
    with train_jsonl.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    vocab = VietnameseTokenizer(min_freq=1).build_vocab(
        cap for row in records for cap in row["captions"]
    )
    vocab_path = processed_dir / "vocab.json"
    vocab.save_vocab(vocab_path)

    return tmp_path, train_jsonl, vocab_path


def test_full_ktvic_pipeline_integration(ktvic_mock_workspace):
    """Kiểm thử luồng chạy thật: KTVICDataset -> CNNEncoder -> AttentionLSTMDecoder -> Loss & Decode."""
    project_root, train_jsonl, vocab_path = ktvic_mock_workspace
    vocab = VietnameseTokenizer.load_vocab(vocab_path)

    # 1. Chế độ train (mode='caption')
    caption_dataset = KTVICDataset(
        jsonl_path=train_jsonl,
        vocab=vocab,
        mode="caption",
        project_root=project_root,
    )
    train_loader = create_dataloader(caption_dataset, batch_size=2, shuffle=False)
    batch = next(iter(train_loader))

    encoder = CNNEncoder(pretrained=False, freeze_backbone=True)
    decoder = AttentionLSTMDecoder(
        vocab_size=len(vocab),
        embed_dim=64,
        decoder_dim=128,
        attention_dim=64,
        encoder_dim=encoder.encoder_dim,
        use_attention=True,
        use_deep_output=True,
    )
    model = AttentionLSTMCaptioner(encoder=encoder, decoder=decoder)
    model.train()

    logits, alphas = model(
        batch["images"],
        batch["caption_ids"],
        teacher_forcing_ratio=0.8,
    )

    ce_loss = F.cross_entropy(
        logits.reshape(-1, len(vocab)),
        batch["target_ids"].reshape(-1),
        ignore_index=SpecialTokens.PAD_IDX,
    )
    att_loss = decoder.compute_doubly_stochastic_loss(
        alphas, batch["caption_ids"], lambda_c=0.5
    )
    total_loss = ce_loss + att_loss
    total_loss.backward()

    assert torch.isfinite(total_loss)
    assert all(p.grad is None for p in encoder.backbone.parameters())
    assert decoder.fc.weight.grad is not None

    # 2. Chế độ đánh giá BLEU/CIDEr (mode='image')
    image_dataset = KTVICDataset(
        jsonl_path=train_jsonl,
        mode="image",
        project_root=project_root,
    )
    eval_loader = create_dataloader(image_dataset, batch_size=2, shuffle=False)
    eval_batch = next(iter(eval_loader))

    sampled_ids, greedy_alphas = model.generate_greedy(
        eval_batch["images"], max_len=12
    )
    hypotheses = decoder.decode_predictions(sampled_ids, vocab)

    assert len(hypotheses) == len(eval_batch["references"]) == 2
    assert all(isinstance(h, str) for h in hypotheses)
    assert greedy_alphas.shape == (2, sampled_ids.size(1), 49)

    beam_seq, beam_alphas = model.generate_beam_search(
        eval_batch["images"][:1], beam_size=3, max_len=12
    )
    beam_text = vocab.decode(beam_seq, skip_special_tokens=True)
    assert isinstance(beam_text, str)
    assert beam_alphas.shape == (len(beam_seq), 49)