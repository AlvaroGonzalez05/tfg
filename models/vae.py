"""
Variational Autoencoder (VAE) - Optimized for Apple Silicon & Multi-Core Processing

This script implements a Variational Autoencoder (VAE) for generating synthetic EV charging data.
Optimized to run on MacBook Pro M4 Pro

Features:
- Multi-threaded data loading
- Efficient memory usage on M-series chips
- KL divergence loss for latent space learning

Author: Álvaro González
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader, TensorDataset

# Detect best available device (MPS for Apple Silicon, else CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=10, latent_dim=10):  # Asegurar que latent_dim sea correcto
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # Doble de latent_dim para mu y log_var
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Ensures stable weight initialization."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def reparameterize(self, mu, log_var):
        """Reparameterization trick ensuring numerical stability."""
        log_var = torch.clamp(log_var, min=-10, max=10)  # Prevent extreme values
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var

def loss_function(recon_x, x, mu, log_var):
    """Computes the VAE loss with stability enhancements."""
    recon_loss = F.mse_loss(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Replace NaN values with zeros
    recon_loss = torch.nan_to_num(recon_loss)
    kl_loss = torch.nan_to_num(kl_loss)
    
    return recon_loss + kl_loss
    
def train_vae(data, num_epochs=1000, batch_size=512, learning_rate=0.001):
    """Trains the Variational Autoencoder on given data."""
    
    input_dim = data.shape[1]
    model = VAE(input_dim=input_dim, latent_dim=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(input_dim)
    # Convert data to tensor and move to device
    data_tensor = torch.tensor(data.values, dtype=torch.float32).to(device)

    # If using MPS (Apple Silicon), disable multiprocessing
    num_workers = 0 if device.type == "mps" else os.cpu_count()

    # Create DataLoader for efficient batch training
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            batch_data = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, log_var = model(batch_data)
            loss = loss_function(recon_x, batch_data, mu, log_var)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(dataloader))

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "models/vae_model.pth")
    torch.save(losses, "models/vae_training_loss.pth")
    return model

vae_param_grid = {
    "latent_dim": [5, 10, 20],
    "learning_rate": [0.001, 0.0005, 0.0001],
    "batch_size": [256, 512]
}

def train_vae_grid_search(data, num_epochs=100):
    """Performs a grid search for VAE hyperparameters."""
    
    results = []
    for params in ParameterGrid(vae_param_grid):
        print(f"Training VAE with: {params}")

        model = VAE(input_dim=data.shape[1], latent_dim=params["latent_dim"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

        data_tensor = torch.tensor(data.values, dtype=torch.float32).to(device)
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True, num_workers=0)

        best_loss = float("inf")
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in dataloader:
                batch_data = batch[0].to(device)
                optimizer.zero_grad()
                recon_x, mu, log_var = model(batch_data)
                loss = loss_function(recon_x, batch_data, mu, log_var)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            if avg_loss < best_loss:
                best_loss = avg_loss

        print(f"Best Loss: {best_loss:.4f}")
        results.append({"params": params, "loss": best_loss})

    # Sort by best loss
    results.sort(key=lambda x: x["loss"])
    print("\nBest Hyperparameters:", results[0]["params"])

    return results[0]["params"]

def generate_vae_data(model, n_samples=1000, latent_dim=5):
    """Generates new synthetic samples from the trained VAE."""
    model.eval()
    
    # Sample from a normal distribution in latent space
    latent_samples = torch.randn(n_samples, latent_dim).to(device)
    
    # Generate synthetic data
    generated_data = model.decoder(latent_samples).detach().cpu().numpy()
    
    return pd.DataFrame(generated_data)