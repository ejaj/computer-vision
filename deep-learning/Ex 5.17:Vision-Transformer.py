import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


# PatchEmbedding Class
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()
        self.d_model = d_model
        self.linear_project = nn.Conv2d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.linear_project(x)
        x = x.flatten(2).transpose(1, 2)  # Convert to (B, P, d_model)
        return x


# PositionalEncoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        pe = torch.zeros(max_seq_length, d_model)
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        cls_token_batch = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token_batch, x), dim=1)
        x = x + self.pe
        return x


# AttentionHead Class
class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention = Q @ K.transpose(-2, -1) / (self.query.weight.size(1) ** 0.5)
        return torch.softmax(attention, dim=-1) @ V


# MultiHeadAttention Class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(d_model, d_model // n_heads) for _ in range(n_heads)])
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.W_o(torch.cat([h(x) for h in self.heads], dim=-1))


# TransformerEncoder Class
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * r_mlp),
            nn.GELU(),
            nn.Linear(d_model * r_mlp, d_model)
        )
        self.activations = {'ln1': [], 'ln2': []}

    def forward(self, x):
        out_ln1 = self.ln1(x)
        self.activations['ln1'].append(out_ln1.detach().cpu().numpy())
        x = x + self.mha(out_ln1)
        out_ln2 = self.ln2(x)
        self.activations['ln2'].append(out_ln2.detach().cpu().numpy())
        return x + self.mlp(out_ln2)


# VisionTransformer Class
class VisionTransformer(nn.Module):
    def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers):
        super().__init__()
        self.n_patches = (img_size[0] * img_size[1]) // (patch_size[0] * patch_size[1])
        self.patch_embedding = PatchEmbedding(d_model, img_size, patch_size, n_channels)
        self.positional_encoding = PositionalEncoding(d_model, self.n_patches + 1)
        self.transformer_encoder = nn.ModuleList([TransformerEncoder(d_model, n_heads) for _ in range(n_layers)])
        self.classifier = nn.Sequential(nn.Linear(d_model, n_classes), nn.Softmax(dim=-1))

    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        for encoder in self.transformer_encoder:
            x = encoder(x)
        return self.classifier(x[:, 0])


# Parameters
d_model = 64
n_classes = 10
img_size = (32, 32)
patch_size = (16, 16)
n_channels = 1
n_heads = 4
n_layers = 3
batch_size = 128

# Data Preparation
transform = T.Compose([T.Resize(img_size), T.ToTensor()])
train_set = MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)

# Collect Activations
images, _ = next(iter(train_loader))
images = images.to(device)
_ = transformer(images)

ln1_activations = []
ln2_activations = []

for encoder in transformer.transformer_encoder:
    ln1_activations.extend(encoder.activations['ln1'])
    ln2_activations.extend(encoder.activations['ln2'])

ln1_activations = np.concatenate(ln1_activations, axis=0).flatten()
ln2_activations = np.concatenate(ln2_activations, axis=0).flatten()

# Plot Histograms
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(ln1_activations, bins=100, density=True, alpha=0.7)
plt.title('LayerNorm1 Activations')
plt.xlabel('Activation Value')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(ln2_activations, bins=100, density=True, alpha=0.7)
plt.title('LayerNorm2 Activations')
plt.xlabel('Activation Value')
plt.ylabel('Density')

plt.tight_layout()
plt.show()
