class CNNEncoder(nn.Module):
    def __init__(self, embed_size: int, fine_tune: bool = True):
        super().__init__()

        # DenseNet121 pre-trained on x-ray
        densenet = xrv.models.DenseNet(weights="densenet121-res224-all")

        self.features = densenet.features

        for p in self.features.parameters():
            p.requires_grad = False

        if fine_tune:
            for name, p in self.features.named_parameters():
                if "denseblock3" in name or "denseblock4" in name or "norm5" in name:
                    p.requires_grad = True

        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.proj = nn.Linear(1024, embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.pos_embed = nn.Parameter(torch.randn(1, 49, embed_size) * 0.02)
        self.dropout = nn.Dropout(0.1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.features(images)
        feat = self.pool(feat)

        B, C, H, W = feat.shape
        feat = feat.view(B, C, H * W).permute(0, 2, 1)

        feat = self.norm(F.relu(self.proj(feat)))
        feat = feat + self.pos_embed
        return self.dropout(feat)