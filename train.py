import torch
from torch import nn
import torch.optim as optim

from models.generator import Generator
from models.discriminator import Discriminator
from utils import get_dataloader, get_device

def train(dataloader, generator, discriminator,latent_size, num_epochs=100):
    device = get_device()
    generator.to(device)
    discriminator.to(device)
    
    optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(dataloader):
            # Update discriminator with real images
            discriminator.zero_grad()
            real_images = images.to(device)
            b_size = real_images.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_images).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Update discriminator with fake images
            noise = torch.randn(b_size, latent_size, 1, 1, device=device) 
            fake_images = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake_images.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update generator
            generator.zero_grad()
            label.fill_(real_label)  # Fake labels are real for generator cost
            output = discriminator(fake_images).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}] {i}/{len(dataloader)} Loss_D: {errD.item()} Loss_G: {errG.item()} D(x): {D_x} D(G(z)): {D_G_z1}/{D_G_z2}')

    torch.save(generator.state_dict(), 'generator_model_latest.pth')
    torch.save(discriminator.state_dict(), 'discriminator_model_latest.pth')
    print('Training finished.')

if __name__ == "__main__":
    batch_size=128
    image_size=64
    latent_size=100
    # TODO: dataset too large to be included within the repository
    #Used dataset https://www.kaggle.com/datasets/crawford/cat-dataset/
    root_dir = 'data/'
    dataloader = get_dataloader(root_dir,batch_size,image_size)

    # Initialize the models
    generator = Generator(latent_size)
    discriminator = Discriminator()
    
    train(dataloader, generator, discriminator,latent_size)