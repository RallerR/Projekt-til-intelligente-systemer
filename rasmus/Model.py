# Import libraries
import torch



# Define the Autoencoder
class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # b x 16 x 16 x 16
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # b x 32 x 8 x 8
            torch.nn.ReLU(),
            torch.nn.Flatten(),  # b x 2048
        )
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2048, 8 * 8 * 32),  # b x 2048
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (32, 8, 8)),  # b x 32 x 8 x 8
            torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # b x 16 x 16 x 16
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # b x 1 x 32 x 32
            torch.nn.Sigmoid(),  # to get values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded