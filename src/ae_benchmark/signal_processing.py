"""Représentations comparables sans inventer la fréquence d'échantillonnage.

ADAPTER_PLUS_TARD: la longueur des fenêtres et le choix des bandes physiques
ne seront fixés qu'après réception de fs et des paramètres des roulements.
"""
import numpy as np
from scipy.signal import hilbert
from scipy.integrate import cumulative_trapezoid


def signal_representation(x, kind='raw'):
    """Transforme [N,C,L] sans apprentissage, avant la normalisation train-only.

    Les spectres sont en *indices de bins* (pas en Hz si fs inconnu).
    Les enveloppes sont calculées par Hilbert sans filtrage passe-bande préalable :
    comparaison exploratoire, PAS une enveloppe optimisée pour diagnostic.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] < 32 or not np.isfinite(x).all():
        raise ValueError('Signal attendu [N,C,L>=32], valeurs finies')
    if kind == 'raw':
        return x.copy()
    if kind == 'envelope':
        y = np.abs(hilbert(x.astype(np.float64), axis=-1))
        return (y - np.mean(y, axis=-1, keepdims=True)).astype(np.float32)
    if kind == 'log_spectrum':
        # Amplitude log + suppression du niveau DC pour limiter les artefacts.
        spec = np.log1p(np.abs(np.fft.rfft(x, axis=-1)))
        spec[..., 0] = 0.
        return spec.astype(np.float32)
    raise ValueError('Représentation inconnue : '+str(kind))


def angular_order_envelope(signal, rpm_trace, fs, samples_per_revolution=128):
    """Rééchantillonnage en angle suivi de l'enveloppe Hilbert et de sa moyenne ôtée.

    signal : [C,L] ou [L], rpm_trace : vitesse instantanée MESURÉE [L] en tr/min,
    fs : fréquence d'échantillonnage vérifiée en Hz.
    Retourne (signal_enveloppe, order_axis), où order_axis est en tours cumulés.

    Ce composant ne reproduit pas à lui seul l'extraction d'enveloppe en ordres
    de Lu et al. 2025 (filtrage adapté et détails article à vérifier).
    Un simple régime «10/15/20» ne constitue PAS une trace rpm instantanée.
    """
    x = np.asarray(signal, dtype=np.float64)
    single = x.ndim == 1
    if single: x = x[None, :]
    rpm = np.asarray(rpm_trace, dtype=np.float64)
    if x.ndim != 2 or rpm.ndim != 1 or len(rpm) != x.shape[-1]:
        raise ValueError('signal [C,L] et rpm_trace [L] synchrones requis')
    if x.shape[-1] < 32 or fs <= 0 or samples_per_revolution < 8:
        raise ValueError('fs positif, >=32 échantillons, >=8 points/tour')
    if (not np.isfinite(x).all() or not np.isfinite(rpm).all()
            or (rpm <= 0).any()):
        raise ValueError('Signal et vitesse finis, rpm strictement positif')
    revolutions = cumulative_trapezoid(rpm / 60., dx=1./fs, initial=0.)
    axis = np.arange(0., revolutions[-1], 1./samples_per_revolution)
    if len(axis) < 32:
        raise ValueError('Acquisition trop courte pour une enveloppe en ordres')
    resampled = np.stack([np.interp(axis, revolutions, row) for row in x])
    envelope = np.abs(hilbert(resampled, axis=-1))
    envelope -= envelope.mean(axis=-1, keepdims=True)
    return (envelope[0] if single else envelope).astype(np.float32), axis