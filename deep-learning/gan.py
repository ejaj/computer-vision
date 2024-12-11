import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os

# Define Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 7 * 7 * 256),
            nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)
        )

    def forward(self, x):
        return self.model(x)

# Loss functions
def discriminator_loss(real_output, fake_output):
    real_loss = criterion(real_output, torch.ones_like(real_output))
    fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
    return real_loss + fake_loss

def generator_loss(fake_output):
    return criterion(fake_output, torch.ones_like(fake_output))

# Training step
def train_step(real_images, generator, discriminator, optimizer_g, optimizer_d, criterion, device):
    batch_size = real_images.size(0)

    # Train Discriminator
    noise = torch.randn(batch_size, 100, device=device)
    fake_images = generator(noise)

    real_output = discriminator(real_images)
    fake_output = discriminator(fake_images.detach())

    loss_d = discriminator_loss(real_output, fake_output)

    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()

    # Train Generator
    fake_output = discriminator(fake_images)
    loss_g = generator_loss(fake_output)

    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()

    return loss_d.item(), loss_g.item()

# Training loop
def train(generator, discriminator, dataloader, optimizer_g, optimizer_d, criterion, epochs, device):
    for epoch in range(epochs):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            loss_d, loss_g = train_step(real_images, generator, discriminator, optimizer_g, optimizer_d, criterion, device)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss D: {loss_d:.4f}, Loss G: {loss_g:.4f}")

        # Generate and save images every epoch
        if (epoch + 1) % 5 == 0:
            noise = torch.randn(16, 100, device=device)
            fake_images = generator(noise)
            save_image(fake_images, f'output/epoch_{epoch + 1}.png', nrow=4, normalize=True)

# Hyperparameters
BATCH_SIZE = 256
EPOCHS = 50
NOISE_DIM = 100
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare dataset
dataset = datasets.MNIST(
    root='data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Train the GAN
os.makedirs("output", exist_ok=True)
train(generator, discriminator, dataloader, optimizer_g, optimizer_d, criterion, EPOCHS, device)
