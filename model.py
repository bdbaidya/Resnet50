import torch
import torch.nn as nn
from torchvision import models

class TemporalResNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        base = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(base.children())[:-1])
        
        # Temporal modeling components
        self.temporal_conv = nn.Conv3d(2048, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.temporal_bn = nn.BatchNorm3d(512)
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(-1, *x.size()[2:])  # (batch*seq_len, C, H, W)
        
        # Spatial features
        spatial = self.features(x)  # (batch*seq_len, 2048, 1, 1)
        spatial = spatial.view(batch_size, seq_len, -1)  # (batch, seq_len, 2048)
        
        # Temporal processing
        temporal = spatial.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (batch, 2048, seq_len, 1, 1)
        temporal = torch.relu(self.temporal_bn(self.temporal_conv(temporal)))
        temporal = self.temporal_pool(temporal)  # (batch, 512, 1, 1, 1)
        
        return self.classifier(temporal.squeeze())