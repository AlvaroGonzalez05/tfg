"""
Training Script

This script trains both the VAE and GAN models on the EV charging dataset.
It saves trained models for later use in synthetic data generation.

Steps:
1. Load preprocessed data.
2. Train the VAE model.
3. Train the GAN model.
4. Save trained models.

Author: Álvaro González
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.vae import *
from models.gan import *
import pandas as pd
import torch

# Load preprocessed data
df = pd.read_csv("data/processed_ev_charging_patterns.csv")

# Train models
#vae_model = train_vae(df)
gan_model = train_gan(df)

# Save models
#torch.save(vae_model.state_dict(), "models/vae_model.pth")
torch.save(gan_model.state_dict(), "models/gan_model.pth")

# Train final GAN model using best hyperparameters
#gan_model, best_gan_params, results = grid_search_train_gan(df, num_epochs=5000)

# Save trained GAN model
#torch.save(gan_model.state_dict(), "models/gan_generator.pth")