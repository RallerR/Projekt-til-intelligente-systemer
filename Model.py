import torch


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = torch.nn.Sequential(
            # Input: N x 1 x 32 x 32
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),  # Output: N x 8 x 32 x 32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),  # Output: N x 8 x 16 x 16
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),  # Output: N x 16 x 16 x 16
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),  # Output: N x 16 x 8 x 8
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Output: N x 32 x 8 x 8
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),  # Output: N x 32 x 4 x 4
            torch.nn.Flatten(),  # Output: N x 512
            torch.nn.Linear(512, 10),  # Output: N x 10 (Bottleneck)
            torch.nn.ReLU()
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(10, 512),  # Output: N x 512
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (32, 4, 4)),  # Output: N x 32 x 4 x 4
            torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: N x 16 x 8 x 8
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: N x 8 x 16 x 16
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: N x 1 x 32 x 32
            torch.nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
