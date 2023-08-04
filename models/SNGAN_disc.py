import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm

class SNGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, num_features=64):
        super(SNGANDiscriminator, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, num_features, 4, 2, 1, bias=False))
        self.conv2 = SpectralNorm(nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=False))
        self.conv3 = SpectralNorm(nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=False))
        self.conv4 = SpectralNorm(nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1, bias=False))
        self.conv5 = SpectralNorm(nn.Conv2d(num_features * 8, 1, 4, 1, 0, bias=False))
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.conv5(x)
        x = self.global_avg_pool(x)
        x = x.view(-1)
        return x