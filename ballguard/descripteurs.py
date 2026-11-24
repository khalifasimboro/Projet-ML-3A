"""Transformation d'un signal brut en 26 descripteurs + spectre.

Pour chaque capteur : 5 descripteurs temporels (RMS, kurtosis, facteur de crête, skewness,
crête-à-crête) et 8 énergies de bande (log10) issues de la densité spectrale de Welch.
"""
import numpy as np
from scipy import signal, stats

from .config import CANAUX, FS, N_BANDES, SUFFIXE
from .lecture import lire_signal

NOMS_TEMPORELS = ["rms", "kurtosis", "crete", "skew", "pkpk"]
FEATURES = [f"{nom}_{s}" for s in SUFFIXE.values()
            for nom in NOMS_TEMPORELS + [f"bande{b + 1}" for b in range(N_BANDES)]]


def frequences_spectre():
    """Les 256 fréquences (Hz) du spectre gardé : 50 Hz à 12,8 kHz."""
    return np.arange(1, 257) * FS / 512


def extraire_depuis_signal(sig):
    """sig : tableau (N, 2). Renvoie (dict des 26 descripteurs, spectre log10 de 512 valeurs)."""
    desc, spectre = {}, []
    for j, canal in enumerate(CANAUX):
        x = sig[:, j] - sig[:, j].mean()
        rms = np.sqrt(np.mean(x ** 2))
        s = SUFFIXE[canal]
        desc[f"rms_{s}"] = float(rms)
        if rms <= np.finfo(np.float32).eps:
            desc[f"kurtosis_{s}"] = 0.0
            desc[f"crete_{s}"] = 0.0
            desc[f"skew_{s}"] = 0.0
        else:
            desc[f"kurtosis_{s}"] = float(stats.kurtosis(x, fisher=False))
            desc[f"crete_{s}"] = float(np.max(np.abs(x)) / rms)
            desc[f"skew_{s}"] = float(stats.skew(x))
        desc[f"pkpk_{s}"] = float(x.max() - x.min())
        _, P = signal.welch(x, fs=FS, nperseg=512)          # 257 fréquences de 0 à 12,8 kHz
        for b, bande in enumerate(np.array_split(P[1:], N_BANDES)):
            desc[f"bande{b + 1}_{s}"] = float(np.log10(max(bande.sum(), np.finfo(np.float64).tiny)))
        spectre.append(np.log10(np.maximum(P[1:], np.finfo(np.float32).tiny)))  # la composante continue est exclue
    return desc, np.concatenate(spectre).astype(np.float32)


def extraire(source):
    """Lit un fichier brut et renvoie (descripteurs, spectre)."""
    return extraire_depuis_signal(lire_signal(source))
