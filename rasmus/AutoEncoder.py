import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


import CircleDataset as CD
import Transform as Tr
import Model as Mdl

def Kør_Auto_encoder():
    # Load the dataset
    dataset = CD.CircleDataset("CircleImages", transform=Tr.transform)

    # DataLoader
    batch_size = 1  # Adjust as needed
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Device (GPU or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the Autoencoder
    autoencoder = Mdl.Autoencoder().to(device)

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
