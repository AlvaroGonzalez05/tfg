"""
Helper Functions

Provides utility functions for saving/loading models and visualizing data.

Functions:
- save_model: Saves trained models.
- load_model: Loads pretrained models.
- plot_data: Visualizes datasets.

Author: Álvaro González
"""
import matplotlib.pyplot as plt
import torch

def save_model(model, path):
    """Saves a trained model."""
    torch.save(model.state_dict(), path)

def load_model(model_class, path):
    """Loads a pretrained model."""
    model = model_class()
    model.load_state_dict(torch.load(path))
    return model

def plot_data(real, synthetic, title):
    """Visualizes distributions of real vs synthetic data."""
    plt.hist(real, alpha=0.5, label="Real")
    plt.hist(synthetic, alpha=0.5, label="Synthetic")
    plt.title(title)
    plt.legend()
    plt.show()