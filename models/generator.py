import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, img_channels=3, features_gen=64):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, features_gen * 16, 4, 1, 0),
            nn.BatchNorm2d(features_gen * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_gen * 16, features_gen * 8, 4, 2, 1),
            nn.BatchNorm2d(features_gen * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_gen * 8, features_gen * 4, 4, 2, 1),
            nn.BatchNorm2d(features_gen * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_gen * 4, features_gen * 2, 4, 2, 1),
            nn.BatchNorm2d(features_gen * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(features_gen * 2, img_channels, 4, 2, 1),
            nn.Tanh()
            # Output: N x img_channels x 64 x 64
        )

    def forward(self, x):
        return self.gen(x)
