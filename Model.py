# Import libraries
import torch


# Define the Autoencoder
class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # b x 8 x 16 x 16
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # b x 16 x 8 x 8
            torch.nn.ReLU(),
            torch.nn.Flatten(),  # b x 1024
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(1024, 8 * 8 * 16),  # b x 1024
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (16, 8, 8)),  # b x 16 x 8 x 8
            torch.nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # b x 8 x 16 x 16
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # b x 1 x 32 x 32
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded