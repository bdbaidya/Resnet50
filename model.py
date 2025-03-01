import torch
import torch.nn as nn
from torchvision import models

class TemporalResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        base = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(base.children())[:-1])
        
        self.temporal_conv = nn.Conv3d(2048, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.temporal_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(-1, *x.size()[2:])
        
        spatial = self.features(x)
        spatial = spatial.view(batch_size, seq_len, -1)
        
        temporal = spatial.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        temporal = self.temporal_conv(temporal)
        temporal = self.temporal_pool(temporal)
        
        return self.classifier(temporal.squeeze())