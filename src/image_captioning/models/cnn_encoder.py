import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class CNNEncoder(nn.Module):
    encoder_dim: int = 2048

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = resnet50(weights=weights)
        # Chỉ giữ các tầng tích chập
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.set_backbone_trainable(not freeze_backbone)

    def set_backbone_trainable(self, trainable: bool) -> None:
        self._backbone_trainable = trainable
        self.backbone.requires_grad_(trainable)
        self.backbone.train(self.training and trainable)

    def train(self, mode: bool = True) -> "CNNEncoder":
        super().train(mode)
        if not self._backbone_trainable:
            self.backbone.eval()
        return self

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(images, torch.Tensor):
            raise TypeError(
                "images phải là torch.Tensor, "
                f"nhận được {type(images).__name__}."
            )
        if images.ndim != 4:
            raise ValueError(
                "images phải có 4 dimensions [B, 3, H, W], "
                f"nhận được shape {tuple(images.shape)}."
            )
        if images.shape[1] != 3:
            raise ValueError(
                "images phải có 3 kênh RGB tại dimension 1, "
                f"nhận được {images.shape[1]} kênh."
            )
        if not images.is_floating_point():
            raise TypeError(
                "images phải là floating-point tensor đã normalize ImageNet, "
                f"nhận được dtype {images.dtype}."
            )
        with torch.set_grad_enabled(
            self._backbone_trainable and torch.is_grad_enabled()
        ):
            feature_map = self.backbone(images)
            global_features = self.global_pool(feature_map).flatten(1)
            spatial_features = feature_map.flatten(2).transpose(1, 2)

        return global_features, spatial_features
