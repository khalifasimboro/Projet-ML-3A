"""Autoencodeur dense (scikit-learn) et outils associés."""
import numpy as np
from sklearn.neural_network import MLPRegressor

from .config import SEED


def creer_ae(couches=(16, 6, 16)):
    """Autoencodeur : `couches` sont les couches cachées, celle du milieu est le goulot."""
    return MLPRegressor(hidden_layer_sizes=couches, activation="tanh", learning_rate_init=1e-3,
                        max_iter=1000, early_stopping=True, n_iter_no_change=30, random_state=SEED)


def erreur_par_descripteur(ae, X):
    """Erreur quadratique de reconstruction, par signal et par descripteur."""
    return (X - ae.predict(X)) ** 2


def erreur_reconstruction(ae, X):
    """Erreur MSE par signal : c'est le score d'anomalie."""
    return erreur_par_descripteur(ae, X).mean(axis=1)


def encoder(ae, X):
    """Code latent : on applique seulement la moitié « encodeur » du réseau."""
    n = len(ae.hidden_layer_sizes) // 2 + 1
    h = np.asarray(X, dtype=float)
    for W, b in zip(ae.coefs_[:n], ae.intercepts_[:n]):
        h = np.tanh(h @ W + b)
    return h
