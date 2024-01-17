from torchvision import transforms

# Transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5), (0.5))
])