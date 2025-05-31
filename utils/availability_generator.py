"""
availability_generator.py

Este módulo permite generar perfiles probabilísticos de disponibilidad del vehículo eléctrico 
para su uso en entornos de entrenamiento de agentes DQN. Los perfiles imitan comportamientos 
habituales de distintos tipos de usuarios y añaden ruido para evitar determinismo total.

Autores: Álvaro González Tabernero
"""

import numpy as np

def generate_availability_profile(profile_type="worker", noise_std=0.1, seed=None):
    """
    Genera un vector de 96 valores ∈ [0,1] que representa la probabilidad horaria de disponibilidad del coche.

    Parámetros:
    ----------
    profile_type : str
        Tipo de perfil de usuario. Uno de: 'worker', 'flexible', 'retired', 'traveller', 'night_owl'
    noise_std : float
        Desviación estándar del ruido gaussiano que se añade a cada slot horario.
    seed : int or None
        Semilla aleatoria para reproducibilidad.

    Retorna:
    -------
    np.ndarray
        Vector de 96 probabilidades horarias ∈ [0,1]
    """
    if seed is not None:
        np.random.seed(seed)

    base_profile = np.zeros(96)

    if profile_type == "worker":
        base_profile[0:24] = 0.95    # 00:00–06:00
        base_profile[24:32] = 0.4    # 06:00–08:00
        base_profile[32:60] = 0.1    # 08:00–15:00
        base_profile[60:68] = 0.6    # 15:00–17:00
        base_profile[68:84] = 0.5    # 17:00–21:00
        base_profile[84:96] = 0.85   # 21:00–00:00

    elif profile_type == "flexible":
        base_profile[:] = 0.6
        base_profile[32:60] = 0.4

    elif profile_type == "retired":
        base_profile[:] = 0.8
        base_profile[12:16] = 0.5   # paseo o recados
        base_profile[68:72] = 0.4   # actividades o visitas

    elif profile_type == "traveller":
        base_profile[:] = 0.7
        base_profile[32:60] = 0.2   # ausente de día
        base_profile[68:84] = 0.3   # ausente algunas tardes

    elif profile_type == "night_owl":
        base_profile[0:24] = 0.5
        base_profile[24:32] = 0.3
        base_profile[32:60] = 0.1
        base_profile[60:72] = 0.4
        base_profile[72:96] = 0.85

    else:
        raise ValueError(f"Perfil no reconocido: {profile_type}")

    noisy_profile = base_profile + np.random.normal(0, noise_std, size=96)
    noisy_profile = np.clip(noisy_profile, 0, 1)

    return noisy_profile
