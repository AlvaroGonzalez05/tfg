"""
Hyperparameter Tuning Script

Optimizes hyperparameters for VAE and GAN models.

Author: Álvaro González
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.vae import tune_vae
from models.gan import tune_gan

tune_vae()
tune_gan()