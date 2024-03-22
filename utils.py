from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch


def visualize_images(images,image_size):
    """
    Visualize image in a plot
    Used in evaluation
    """
    plt.figure(figsize=(10, 10))
    for i in range(images.size(0)):
        image = images[i].view(3, image_size, image_size)  
        plt.subplot(8, 8, i + 1)
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())  
        plt.axis('off')
    plt.show()
    
def get_dataloader(root_dir, batch_size, image_size):
    """
    Load images and perform pre-processing
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")