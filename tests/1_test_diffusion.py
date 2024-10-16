import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from myutils import utils_defence

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------- Define Noise Schedule ------------------------------------------------
timesteps = 1000
noise_schedule = torch.linspace(1e-4, 2e-2, timesteps).to(device)

# Load CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model and optimizer
model = utils_defence.UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------------------------------------------------------- Training the Model ----------------------------------------
for epoch in range(50):
    for images, _ in train_loader:
        images = images.to(device)
        t = torch.randint(0, timesteps, (images.shape[0],)).to(device)  # random timesteps
        
        # Compute loss using the loss function
        loss = utils_defence.loss_fn(model, images, t, noise_schedule)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

generated_img = utils_defence.sample(model, timesteps, noise_schedule)
utils_defence.show_image(generated_img[0])
