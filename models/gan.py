"""
Generative Adversarial Network (GAN) - Optimized for Apple Silicon & Multi-Core Processing

This script implements a GAN for generating synthetic EV charging data.
Optimized to run on MacBook Pro M4 Pro using Apple's Metal backend.

Features:
- Proper adversarial training loop
- Uses latent space (noise input) for variation in synthetic data
- Optimized with Adam optimizers

Author: Álvaro González
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class Generator(nn.Module):
    def __init__(self, latent_dim=50, output_dim=10, hidden_dim=64, num_layers=2):
        super(Generator, self).__init__()
        layers = []
        # First layer
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=10):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

def train_gan(data, num_epochs=5000, batch_size=1024, learning_rate=0.00005, latent_dim=20):
    """Trains the Generative Adversarial Network on given data."""
    
    input_dim = data.shape[1]
    generator = Generator(latent_dim=latent_dim, output_dim=input_dim).to(device)
    discriminator = Discriminator(input_dim=input_dim).to(device)

    # Optional: Initialize weights (Xavier initialization)
    for m in generator.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    for m in discriminator.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Use BCEWithLogitsLoss for stability
    criterion = torch.nn.BCEWithLogitsLoss()

    data_tensor = torch.tensor(data.values, dtype=torch.float32).to(device)

    # Check that there are no NaNs in the data
    if torch.isnan(data_tensor).any():
        print("Warning: data_tensor contains NaN values!")

    num_workers = 0 if device.type == "mps" else os.cpu_count()
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    for epoch in range(num_epochs):
        for real_batch in dataloader:
            real_data = real_batch[0].to(device)

            # Label smoothing for real labels
            real_labels = torch.full((real_data.size(0), 1), 0.9, device=device)
            fake_labels = torch.zeros(real_data.size(0), 1, device=device)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_pred = discriminator(real_data)
            real_loss = criterion(real_pred, real_labels)
            
            noise = torch.randn(real_data.size(0), latent_dim, device=device)
            fake_data = generator(noise)
            fake_pred = discriminator(fake_data.detach())
            fake_loss = criterion(fake_pred, fake_labels)
    
            d_loss = real_loss + fake_loss
            d_loss.backward()
            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            noise = torch.randn(real_data.size(0), latent_dim, device=device)
            fake_data = generator(noise)
            fake_pred = discriminator(fake_data)
            g_loss = criterion(fake_pred, real_labels)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    torch.save(generator.state_dict(), "models/gan_generator.pth")
    torch.save(discriminator.state_dict(), "models/gan_discriminator.pth")
    return generator

def grid_search_train_gan(data, num_epochs=5000):
    """
    Performs a high intensity grid search for GAN hyperparameters using the 
    existing train_gan function. Evaluates each model on a proxy diversity metric.
    """
    gan_param_grid = {
        "latent_dim": [10, 20, 50],
        "learning_rate": [0.0002, 0.0001, 0.00005],
        "batch_size": [512,1024]
    }
    
    best_model = None
    best_params = None
    best_diversity = -float("inf")
    results = []
    
    for params in ParameterGrid(gan_param_grid):
        print(f"\nTraining GAN with parameters: {params}")
        # Call your existing train_gan function with the current parameters.
        # Note that train_gan returns only the generator.
        generator = train_gan(
            data, 
            num_epochs=num_epochs, 
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            latent_dim=params["latent_dim"]
        )
        
        # Evaluate the trained generator using a diversity metric:
        # Generate a fixed batch of noise and compute average feature std.
        noise = torch.randn(512, params["latent_dim"], device=device)
        generated_samples = generator(noise)
        # Diversity metric: average standard deviation across features
        diversity = generated_samples.std(dim=0).mean().item()
        print(f"Diversity metric: {diversity:.4f}")
        
        results.append({"params": params, "diversity": diversity})
        if diversity > best_diversity:
            best_diversity = diversity
            best_model = generator
            best_params = params
            
    print("\nBest Hyperparameters for GAN:", best_params)
    print(f"Best Diversity Metric: {best_diversity:.4f}")
    return best_model, best_params, results

def generate_gan_data(generator, n_samples=1000, latent_dim=5):
    """Generates new synthetic samples from the trained GAN."""
    generator.eval()
    noise = torch.randn(n_samples, latent_dim).to(device)
    generated_data = generator(noise).detach().cpu().numpy()
    return pd.DataFrame(generated_data)