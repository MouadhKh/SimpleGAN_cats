import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features_disc=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x img_channels x 64 x 64
            nn.Conv2d(img_channels, features_disc, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features_disc, features_disc * 2, 4, 2, 1),
            nn.BatchNorm2d(features_disc * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features_disc * 2, features_disc * 4, 4, 2, 1),
            nn.BatchNorm2d(features_disc * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features_disc * 4, features_disc * 8, 4, 2, 1),
            nn.BatchNorm2d(features_disc * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(features_disc * 8, 1, 4, 1, 0),
            nn.Sigmoid()
            # Output: N x 1 x 1 x 1
        )

    def forward(self, x):
        return self.disc(x).view(-1)
