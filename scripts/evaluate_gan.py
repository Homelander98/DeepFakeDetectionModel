import torch
from models.gan import GAN
import matplotlib.pyplot as plt

# Load the trained GAN model
latent_dim = 100
input_dim = 256
gan = GAN(latent_dim, input_dim)
gan.load_state_dict(torch.load("models/gan_model.pth"))
gan.eval()

# Generate fake data
z = torch.randn(10, latent_dim)  # Generate 10 samples
fake_data = gan.generator(z)

# Evaluate discriminator on fake data
discriminator_output = gan.discriminator(fake_data)
print("Discriminator output on fake data:", discriminator_output)

# Plot generated data (example for 1D data)
plt.plot(fake_data.detach().numpy().T)
plt.title("Generated Data")
plt.show()