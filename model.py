import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import itertools

# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Loss Functions
adversarial_loss = nn.MSELoss()
cycle_loss = nn.L1Loss()

# Initialize models
generator_AB = Generator()
generator_BA = Generator()
discriminator_A = Discriminator()
discriminator_B = Discriminator()

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

initialize_weights(generator_AB)
initialize_weights(generator_BA)
initialize_weights(discriminator_A)
initialize_weights(discriminator_B)

# Optimizers
g_optimizer = optim.Adam(itertools.chain(generator_AB.parameters(), generator_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(itertools.chain(discriminator_A.parameters(), discriminator_B.parameters()), lr=0.0002, betas=(0.5, 0.999))

# DataLoader
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset_A = datasets.ImageFolder(r'C:\Users\hp\OneDrive\Desktop\Gen_AI_Capsule_project\cycle-gan\datasets\data_A', transform=transform)
dataset_B = datasets.ImageFolder(r'C:\Users\hp\OneDrive\Desktop\Gen_AI_Capsule_project\cycle-gan\datasets\data_B', transform=transform)
loader_A = DataLoader(dataset_A, batch_size=16, shuffle=True)
loader_B = DataLoader(dataset_B, batch_size=16, shuffle=True)

# Training Loop
for epoch in range(100):
    for batch_A, batch_B in zip(loader_A, loader_B):
        real_A = batch_A[0].to('cuda')
        real_B = batch_B[0].to('cuda')

        # Train Generators
        g_optimizer.zero_grad()

        fake_B = generator_AB(real_A)
        fake_A = generator_BA(real_B)

        recon_A = generator_BA(fake_B)
        recon_B = generator_AB(fake_A)

        loss_gan_AB = adversarial_loss(discriminator_B(fake_B), torch.ones_like(discriminator_B(fake_B)))
        loss_gan_BA = adversarial_loss(discriminator_A(fake_A), torch.ones_like(discriminator_A(fake_A)))

        loss_cycle_A = cycle_loss(recon_A, real_A)
        loss_cycle_B = cycle_loss(recon_B, real_B)

        g_loss = loss_gan_AB + loss_gan_BA + 10 * (loss_cycle_A + loss_cycle_B)
        g_loss.backward()
        g_optimizer.step()

        # Train Discriminators
        d_optimizer.zero_grad()

        loss_d_A = (adversarial_loss(discriminator_A(real_A), torch.ones_like(discriminator_A(real_A))) +
                    adversarial_loss(discriminator_A(fake_A.detach()), torch.zeros_like(discriminator_A(fake_A.detach())))) / 2

        loss_d_B = (adversarial_loss(discriminator_B(real_B), torch.ones_like(discriminator_B(real_B))) +
                    adversarial_loss(discriminator_B(fake_B.detach()), torch.zeros_like(discriminator_B(fake_B.detach())))) / 2

        loss_d = loss_d_A + loss_d_B
        loss_d.backward()
        d_optimizer.step()

    print(f"Epoch {epoch}: Generator Loss: {g_loss.item()}, Discriminator Loss: {loss_d.item()}")
