import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image


import CircleDataset as CD
import Transform as Tr
import Model as Mdl


def train_autoencoder():
    dataset = CD.CircleDataset("Dataset2", transform=Tr.transform)  # Vælg dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder = Mdl.Autoencoder().to(device)
    loss_function = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    num_epochs = 20

    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
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

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f},Val Loss: {avg_val_loss:.4f}')

    # Save the trained model
    torch.save(autoencoder.state_dict(), 'autoencoder_model.pth')

    # Optionally, plot the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return autoencoder

def generate_image(model_path, image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the trained model
    autoencoder = Mdl.Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.eval()

    # Load and process the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((32, 32))  # Resize to the expected input size (e.g., 32x32)
    image = Tr.transform(image).unsqueeze(0).to(device)  # Apply the same transformation as during training

    # Generate the image
    with torch.no_grad():
        reconstructed_img = autoencoder(image).cpu().squeeze(0)
    plt.imshow(reconstructed_img[0], cmap='gray')
    plt.show()


# Example usage
train_autoencoder()  # Uncomment this line if you need to train the model
generate_image('autoencoder_model.pth', 'Dataset2/image.0.png')
