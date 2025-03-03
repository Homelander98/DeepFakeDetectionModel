import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.gan import GAN
import numpy as np

# Hyperparameters
latent_dim = 100  # Dimension of the latent space (noise vector)
input_dim = 256   # Dimension of the input data (e.g., video/audio features)
batch_size = 32
epochs = 50
lr = 0.0002

# Initialize GAN
gan = GAN(latent_dim, input_dim).to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(gan.generator.parameters(), lr=lr)
optimizer_D = optim.Adam(gan.discriminator.parameters(), lr=lr)

# Example: Load real data (replace with your dataset)
real_data = torch.randn(1000, input_dim)  # Example: 1000 samples of 256-dimensional data
dataloader = DataLoader(real_data, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for i, real_samples in enumerate(dataloader):
        real_samples = real_samples.to(device)
        batch_size = real_samples.size(0)

        # Labels for real and fake data
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        optimizer_D.zero_grad()

        # Real data
        real_outputs = gan.discriminator(real_samples)
        d_loss_real = criterion(real_outputs, real_labels)

        # Fake data
        z = torch.randn(batch_size, latent_dim).to(device)  # Random noise
        fake_samples = gan.generator(z)
        fake_outputs = gan.discriminator(fake_samples.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        # Generate fake data and try to fool the discriminator
        fake_outputs = gan.discriminator(fake_samples)
        g_loss = criterion(fake_outputs, real_labels)  # Generator wants fake data to be classified as real
        g_loss.backward()
        optimizer_G.step()

        # Print losses
        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Save the trained GAN model
torch.save(gan.state_dict(), "models/gan_model.pth")