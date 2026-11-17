"""Autoencodeur convolutif sur le spectre (CNN-AE spectral).

Repris de l'étude comparative V6 du groupe (branche feature/gt204-gt205-membre-b) : même architecture,
même optimiseur, mêmes prétraitements. Il lit les 512 valeurs du spectre (2 capteurs x 256 fréquences)
au lieu des 26 descripteurs.
"""
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import SEED

EPOCHS, BATCH, LATENT = 25, 128, 6


class SpectralCNN(nn.Module):
    """Encodeur : 2 convolutions 1D puis projection sur 6 dimensions. Décodeur symétrique."""

    def __init__(self, latent=LATENT):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(2, 16, 5, 2, 2), nn.ReLU(), nn.Conv1d(16, 32, 5, 2, 2), nn.ReLU())
        self.proj = nn.Linear(32 * 64, latent)
        self.unproj = nn.Linear(latent, 32 * 64)
        self.dec = nn.Sequential(nn.ConvTranspose1d(32, 16, 4, 2, 1), nn.ReLU(), nn.ConvTranspose1d(16, 2, 4, 2, 1))

    def encode(self, x):
        return self.proj(self.enc(x).flatten(1))

    def forward(self, x):
        return self.dec(self.unproj(self.encode(x)).reshape(-1, 32, 64))


def en_tenseur(spectres):
    """(N, 512) -> (N, 2, 256) : un canal par capteur."""
    return torch.tensor(np.asarray(spectres, dtype=np.float32).reshape(-1, 2, 256))


def entrainer_cnn(X_train, X_val, seed=SEED):
    """Adam (lr 1e-3), perte MSE, arrêt après 5 époques sans progrès sur la validation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    modele = SpectralCNN()
    opt = torch.optim.Adam(modele.parameters(), lr=1e-3)
    x_tr, x_val = en_tenseur(X_train), en_tenseur(X_val)
    lot = DataLoader(TensorDataset(x_tr), batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed))
    meilleur, etat, patience, historique = float("inf"), None, 0, []
    for _ in range(EPOCHS):
        modele.train()
        for (x,) in lot:
            perte = (modele(x) - x).square().mean()
            opt.zero_grad()
            perte.backward()
            opt.step()
        modele.eval()
        with torch.no_grad():
            score = float(np.mean([(modele(c) - c).square().mean().item() for c in x_val.split(BATCH)]))
        historique.append(score)
        if score < meilleur - 1e-5:
            meilleur, etat, patience = score, copy.deepcopy(modele.state_dict()), 0
        else:
            patience += 1
            if patience >= 5:
                break
    modele.load_state_dict(etat)
    modele.eval()
    return modele, historique


def erreur_cnn(modele, X):
    """Erreur de reconstruction moyenne par signal."""
    with torch.no_grad():
        x = en_tenseur(X)
        return np.concatenate([(modele(c) - c).square().flatten(1).mean(1).numpy() for c in x.split(512)])


def encoder_cnn(modele, X):
    with torch.no_grad():
        return np.concatenate([modele.encode(c).numpy() for c in en_tenseur(X).split(512)])


def etat(modele):
    """Poids seuls (sauvegarde légère et indépendante de la version de PyTorch)."""
    return {k: v.numpy() for k, v in modele.state_dict().items()}


def depuis_etat(poids):
    modele = SpectralCNN()
    modele.load_state_dict({k: torch.tensor(v) for k, v in poids.items()})
    modele.eval()
    return modele
