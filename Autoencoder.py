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

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# DataLoaders for training and validation sets
batch_size = 1  # Adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the Autoencoder
autoencoder = Autoencoder().to(device)

# Loss Function and Optimizer
loss_function = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# Training and validation
num_epochs = 10  # Adjust as needed
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    # Training
    autoencoder.train()
    train_loss = 0.0
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        output = autoencoder(img)
        loss = loss_function(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    autoencoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            img, _ = data
            img = img.to(device)
            output = autoencoder(img)
            loss = loss_function(output, img)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Optionally, plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


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

# To reconstruct an image:
def reconstruct_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((32, 32))  # Resize to the expected input size (e.g., 32x32)
    image = transform(image).unsqueeze(0).to(device)  # Apply the same transformation as during training
    autoencoder.eval()
    with torch.no_grad():
        reconstructed_img = autoencoder(image).cpu().squeeze(0)
    plt.imshow(reconstructed_img[0], cmap='gray')
    plt.show()

# Call this function with the path of the new image
reconstruct_image('Test/circle_1.jpg')
