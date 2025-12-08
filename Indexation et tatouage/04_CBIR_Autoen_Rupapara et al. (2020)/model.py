import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(AutoEncoder, self).__init__()
        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128*8*8, latent_dim)
        )
        # Decodeur
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*8*8),
            nn.ReLU(True),
            nn.Unflatten(1, (128,8,8)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # images normalisées entre 0 et 1
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out