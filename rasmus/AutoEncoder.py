import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import CircleDataset as CD
import Transform as Tr
import Model as Mdl


def Kør_Auto_encoder():
    # Load the dataset
    dataset = CD.CircleDataset("CircleImages", transform=Tr.transform)

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
    autoencoder = Mdl.Autoencoder().to(device)

    # Loss Function and Optimizer
    loss_function = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    # Training and validation
    num_epochs = 20  # Adjust as needed
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
