import torch
from torchvision.utils import save_image
from models.generator import Generator
from utils import get_device, visualize_images

def generate_images(generator, num_samples, z_dim):
    """Generate images using a trained generator model.

    Args:
        generator (torch.nn.Module): The trained generator model.
        num_samples (int): Number of images to generate.
        z_dim (int): Dimension of the latent space.
    """
    generator.eval()  # Set the generator to evaluation mode
    device = next(generator.parameters()).device

    # Generate random noise as input for the generator
    noise = torch.randn(num_samples, z_dim, 1, 1, device=device)

    # Generate images from the noise
    with torch.no_grad():
        generated_images=generator(noise)
        visualize_images(generated_images,image_size)

if __name__ == "__main__":
    image_size = 64
    batch_size = 128
    num_samples = 32
    latent_size=100
    device = get_device()
    generator = Generator(latent_size, img_channels=3, features_gen=64).to(device)

    # Load the trained generator model
    generator.load_state_dict(torch.load("generator_model_latest.pth", map_location=device))

    # Generate images
    generate_images(generator, num_samples,latent_size)
