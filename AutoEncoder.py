import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

import CircleDataset as CD
import Transform as Tr
import Model as Mdl


def train_autoencoder():
    torch.manual_seed(11) #9
    dataset = CD.CircleDataset("Dataset3", transform=Tr.transform)  # Vælg dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder = Mdl.Autoencoder().to(device)
    loss_function = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
    num_epochs = 15 # 14

    #for images, _ in train_loader:
    #    print(torch.min(images), torch.max(images))

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        autoencoder.train()
        train_loss = 0.0
        for data in train_loader:
            img, _ = data
            img = img.to(device)

            encoded = autoencoder.encode(img)
            decoded = autoencoder.decode(encoded)
            loss = loss_function(decoded, img)

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

    # Gem model
    torch.save(autoencoder.state_dict(), 'autoencoder_model.pth')

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # plt.savefig('Resultater/loss_plot.png')

    plt.show()

    return autoencoder


# Brug model til at regenerere et billede
def generate(model_path, image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    autoencoder = Mdl.Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.eval()

    original_image = Image.open(image_path).convert('L')  # Grayscale
    original_image = original_image.resize((32, 32))  # Resize
    transformed_image = Tr.transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed_image = autoencoder(transformed_image).cpu().squeeze(0)
        bottleneck_values = autoencoder.encode(transformed_image)

    # Print bottleneck
    print("Bottleneck values:", bottleneck_values.cpu().numpy())

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(reconstructed_image[0], cmap='gray')
    axes[1].set_title("Reconstructed Image")
    axes[1].axis('off')

    plt.tight_layout()

    # plt.savefig('Resultater/reconstruction_random_circle.png')

    plt.show()


train_autoencoder()  # Kommenter hvis modellen er trænet
generate('autoencoder_model.pth', 'test/image.2.png')


"""
# Brug decoder til at lave nye cirkler

autoencoder = Mdl.Autoencoder()
autoencoder.load_state_dict(torch.load('autoencoder_model.pth'))
autoencoder.eval()

random_vector = torch.randn(1, 10)  # Random vector
values = [49.726803, 58.998077, 0., 1.1448752, 0., 0., 79.126434, 0., 0., 14.963775 ] # Vector fra training
specific_vector = torch.tensor([values])


random_vector2 = torch.rand(1, 10) * 100
zero_positions = [0, 4, 6, 7, 9]

for pos in zero_positions:
    random_vector2[0, pos] = 0.0

# print(random_vector)

with torch.no_grad():
    generated_image = autoencoder.decoder(specific_vector)

generated_image = generated_image.squeeze().cpu().numpy()

# Plot
plt.imshow(generated_image, cmap='gray')
plt.title("Generated Image")
plt.axis('off')
plt.show()

"""