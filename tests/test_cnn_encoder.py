import torch
from torch import nn
from image_captioning.models.cnn_encoder import CNNEncoder


def test_cnn_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # 1. Khởi tạo CNN Encoder
    encoder = CNNEncoder(
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    encoder.eval()

    # 2. Tạo batch ảnh giả lập
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)

    # 3. Forward
    with torch.no_grad():
        global_features, spatial_features = encoder(images)

    # 4. Kiểm tra kích thước
    print("\n===== CNN ENCODER TEST =====")
    print(f"Input shape:          {images.shape}")
    print(f"Global features:      {global_features.shape}")
    print(f"Spatial features:     {spatial_features.shape}")

    # 5. Kiểm tra output
    assert global_features.shape == (batch_size, 2048)
    assert spatial_features.shape == (batch_size, 49, 2048)

    assert torch.isfinite(global_features).all()
    assert torch.isfinite(spatial_features).all()

    # 6. Kiểm tra backbone đã freeze
    trainable_params = sum(
        p.numel() for p in encoder.backbone.parameters()
        if p.requires_grad
    )

    print(f"Trainable backbone params: {trainable_params}")

    assert trainable_params == 0

    print("\nCNN Encoder hoạt động chính xác!")
