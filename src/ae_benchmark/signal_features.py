"""Caractéristiques vibratoires reproductibles, sans fréquence d'échantillonnage supposée.

Les bandes fréquentielles sont des fractions de Nyquist, PAS des fréquences en Hz.
À remplacer par des bandes physiques lorsque fs et les caractéristiques du roulement
seront connues. Les caractéristiques ne constituent pas un diagnostic physique.
"""
import numpy as np
from scipy.stats import kurtosis, skew
import warnings


def vibration_features(x):
    """[N, canaux, temps] -> [N, 10*canaux] ; caractéristiques sans apprentissage."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] < 16 or not np.isfinite(x).all():
        raise ValueError('Entrée attendue [N,C,L>=16] finie')
    eps = 1e-12
    spectrum = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    spectrum = spectrum[..., 1:]  # suppression DC
    spectrum = spectrum / np.maximum(spectrum.sum(axis=-1, keepdims=True), eps)
    bands = np.array_split(np.arange(spectrum.shape[-1]), 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        k = kurtosis(x, axis=-1, fisher=True)
        s = skew(x, axis=-1)
    feats = [
        np.sqrt(np.mean(x**2, axis=-1)),
        np.std(x, axis=-1),
        np.max(np.abs(x), axis=-1),
        np.max(np.abs(x), axis=-1) / np.maximum(np.sqrt(np.mean(x**2, axis=-1)), eps),
        k,
        s,
        -np.sum(spectrum * np.log(spectrum + eps), axis=-1) / np.log(spectrum.shape[-1]),
    ]
    feats += [spectrum[..., band].sum(axis=-1) for band in bands]
    out = np.stack(feats, axis=-1).reshape(len(x), -1)
    return np.nan_to_num(out, nan=0.0, posinf=0., neginf=0.).astype(np.float32)