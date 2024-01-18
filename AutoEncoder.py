import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

import CircleDataset as CD
import Transform as Tr
import Model as Mdl


def train_autoencoder():
    torch.manual_seed(11)
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
    num_epochs = 15

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
        # bottleneck_values = autoencoder.encode(transformed_image)

    # print("Bottleneck values:", bottleneck_values.cpu().numpy())

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(reconstructed_image[0], cmap='gray')
    axes[1].set_title("Reconstructed Image")
    axes[1].axis('off')

    plt.tight_layout()

    # plt.savefig('Resultater/reconstruction_1_hidden_layer.png')

    plt.show()

# Generer nye tilfældige cirkler
def generate_images_with_vector_modification(model_path, index_to_modify, value_range, n_rows=2, n_cols=5):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder = Mdl.Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.eval()

    values = [143.62097, 160.43253, 22.21773, 0., 79.460464, 0., 0., 0., 0., 7.0736217]
    base_vector = torch.tensor([values]).to(device)

    n_images = n_rows * n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    axes = axes.flatten() if n_rows > 1 else [axes]

    for i, value in enumerate(torch.linspace(value_range[0], value_range[1], steps=n_images)):
        modified_vector = base_vector.clone()
        modified_vector[0, index_to_modify] = value

        with torch.no_grad():
            generated_image = autoencoder.decoder(modified_vector).cpu().squeeze(0)

        axes[i].imshow(generated_image[0], cmap='gray')
        axes[i].set_title(f"Value: {value:.2f}")
        axes[i].axis('on')
        axes[i].grid(True)
        axes[i].set_xticks([0, 10, 20, 30])
        axes[i].set_yticks([0, 10, 20, 30])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Effect of Changing Value at Index 0", fontsize=14, verticalalignment='top')

    # plt.savefig('Resultater/Varierende Index 0.png')

    plt.show()


# Generer nye tilfældige cirkler
def generate_random_circles(model_path, value_ranges, number_of_circles):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder = Mdl.Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.eval()

    for i in range(number_of_circles):
        random_values = [torch.FloatTensor(1).uniform_(low, high) for low, high in value_ranges]
        random_vector = torch.cat(random_values, dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            generated_image = autoencoder.decoder(random_vector).cpu().squeeze(0)

        plt.imshow(generated_image[0], cmap='gray')
        plt.title(f"Randomly Generated Circle {i+1}")
        plt.axis('on')
        plt.grid(True)
        plt.xticks([0, 10, 20, 30])
        plt.yticks([0, 10, 20, 30])

        plt.savefig(f'Pilotstudy/randomly_generated_circle_{i+1}.png')

        plt.show()


# Træn autoencoderen
train_autoencoder()  # Kommenter ud hvis modellen er trænet

# Rekonstruer billede
generate('autoencoder_model.pth', 'Dataset3/circle_3.png')

# Effekt på vektor index
generate_images_with_vector_modification('autoencoder_model.pth', index_to_modify=0,
                                         value_range=(-50, 200), n_rows=2, n_cols=5,)

# Generer tilfældig circel
value_ranges = [(0, 250), (25, 150), (0, 100), (0, 50), (25, 150), (0, 0), (-25, 0), (-25, 25), (0, 0), (0, 50)]
generate_random_circles('autoencoder_model.pth', value_ranges, number_of_circles=1)
