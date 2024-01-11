# Import libraries
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt


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


# Custom dataset class
class CircleDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = os.listdir(directory)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(0)  # Returning 0 as a dummy target


# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the dataset
dataset = CircleDataset("CircleImages", transform=transform)

# DataLoader
batch_size = 1  # Adjust as needed
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the Autoencoder
autoencoder = Autoencoder().to(device)

# Loss Function and Optimizer
loss_function = torch.nn.MSELoss(reduction='sum')
#loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# Training the Autoencoder
num_epochs = 20  # Adjust as needed
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        # Forward pass
        output = autoencoder(img)
        loss = loss_function(output, img)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Testing the Autoencoder
# Display some original and reconstructed images
autoencoder.eval()  # Set the model to evaluation mode
with torch.no_grad():  # No need to track gradients
    for i, (img, _) in enumerate(train_loader):
        img = img.to(device)
        output = autoencoder(img)
        # Display the first image in the batch
        original = img.cpu().numpy()[0][0]  # First image, first channel
        reconstructed = output.cpu().numpy()[0][0]  # First image, first channel
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray')
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed, cmap='gray')
        plt.title('Reconstructed Image')
        plt.show()
        break  # Only display the first batch

