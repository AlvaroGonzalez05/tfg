"""
Synthetic Data Generation Script

Uses trained VAE and GAN models to generate synthetic EV charging data.

Steps:
1. Load trained models.
2. Generate new data samples.
3. Save generated data.

Author: Álvaro González
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
from models.vae import VAE, generate_vae_data
from models.gan import Generator, generate_gan_data

# Detect device (MPS for Apple Silicon, else CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

### **VAE: Load Model & Generate Data**
# Ensure input_dim matches the preprocessed dataset
df = pd.read_csv("data/processed_ev_charging_patterns.csv")  
input_dim = df.shape[1]  # Dynamically set input_dim

vae_model = VAE(input_dim=input_dim, latent_dim=10)  # Ensure this matches training!
vae_model.load_state_dict(torch.load("models/vae_model.pth"))
vae_model.to(device)
vae_model.eval()

vae_generated = generate_vae_data(vae_model, n_samples=100000, latent_dim= 10)  # Ensure latent_dim is correct
vae_generated.to_csv("data/vae_generated.csv", index=False)

print("✅ VAE synthetic data saved successfully!")

### **GAN: Load Model & Generate Data**
gan_model = Generator(latent_dim=20, output_dim=input_dim, hidden_dim=64, num_layers=2)
gan_model.load_state_dict(torch.load("models/gan_generator.pth"))
gan_model.to(device)
gan_model.eval()

gan_generated = generate_gan_data(gan_model, n_samples=10000, latent_dim=20)
gan_generated.to_csv("data/gan_generated.csv", index=False)

print("✅ GAN synthetic data saved successfully!")