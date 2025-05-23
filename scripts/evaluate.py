"""
Evaluation Script

Compares the quality of synthetic data generated by VAE and GAN models.

Functions:
- compute_statistics: Computes distributional metrics.
- visualize_distributions: Plots real vs. generated data.
- evaluate_models: Runs statistical tests to compare models.

Author: Álvaro González
"""
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.helpers import compute_statistics, visualize_distributions

# Load datasets
real_data = pd.read_csv("../data/ev_charging_patterns.csv")
vae_generated = pd.read_csv("../data/vae_generated.csv")
gan_generated = pd.read_csv("../data/gan_generated.csv")

# Evaluate models
compute_statistics(real_data, vae_generated, gan_generated)
visualize_distributions(real_data, vae_generated, gan_generated)