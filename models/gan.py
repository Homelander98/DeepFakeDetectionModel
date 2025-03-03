import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator model for the GAN.
    """
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()  # Output is normalized to [-1, 1]
        )

    def forward(self, z):
        """
        Forward pass for the generator.
        :param z: Latent vector (noise)
        :return: Generated data
        """
        return self.model(z)


class Discriminator(nn.Module):
    """
    Discriminator model for the GAN.
    """
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output is a probability (real or fake)
        )

    def forward(self, x):
        """
        Forward pass for the discriminator.
        :param x: Input data (real or generated)
        :return: Probability of being real
        """
        return self.model(x)


class GAN(nn.Module):
    """
    GAN model combining Generator and Discriminator.
    """
    def __init__(self, latent_dim, input_dim):
        super(GAN, self).__init__()
        self.generator = Generator(latent_dim, input_dim)
        self.discriminator = Discriminator(input_dim)

    def forward(self, z):
        """
        Forward pass for the GAN.
        :param z: Latent vector (noise)
        :return: Generated data and discriminator's output
        """
        generated_data = self.generator(z)
        discriminator_output = self.discriminator(generated_data)
        return generated_data, discriminator_output