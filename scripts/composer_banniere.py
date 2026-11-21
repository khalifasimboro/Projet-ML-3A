"""Prépare l'illustration panoramique de la bannière de l'accueil (app/assets/hero.webp).

1. Les bandes noires en haut et en bas sont retirées ; l'image n'est ni recadrée ni déformée.
2. Les quatre bords deviennent progressivement transparents pour se fondre dans le fond de la page
   (thème clair comme sombre). En haut, le fondu est interrompu là où le toit du train touche le bord,
   pour ne pas l’estomper : seuls les pixels de ciel s’effacent.

    .venv/Scripts/python scripts/composer_banniere.py chemin/vers/illustration_panoramique.webp
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "app" / "assets"
SEUIL_NOIR = 100          # luminosité moyenne d'une ligne en dessous de laquelle elle fait partie d'une bande noire
FONDU_COTES = 0.07        # part de la largeur fondue à gauche et à droite
FONDU_BAS = 0.12          # part de la hauteur fondue en bas
FONDU_HAUT = 0.09         # part de la hauteur fondue en haut (sauf au-dessus du toit)


def _rampe(n, puissance=1.5):
    return (np.arange(n) / max(n - 1, 1)) ** puissance


def composer(source):
    im = Image.open(source).convert("RGB")
    a = np.asarray(im)
    lignes = np.where(a.mean(axis=(1, 2)) > SEUIL_NOIR)[0]
    a = a[lignes[0]:lignes[-1] + 1]
    h, w = a.shape[:2]

    alpha = np.ones((h, w))
    fc = round(w * FONDU_COTES)
    alpha[:, :fc] *= _rampe(fc)[None, :]
    alpha[:, w - fc:] *= _rampe(fc)[::-1][None, :]
    fb = round(h * FONDU_BAS)
    alpha[h - fb:, :] *= _rampe(fb)[::-1][:, None]
    # en haut, seuls les pixels de ciel s'estompent : le toit du train, qui touche le bord, reste net
    fh = round(h * FONDU_HAUT)
    haut = a[:fh].astype(float)
    ciel = np.clip((haut[:, :, 2] - haut[:, :, 0] - 15) / 45, 0, 1)        # 1 = bleu du ciel, 0 = train
    alpha[:fh, :] *= 1 - ciel * (1 - _rampe(fh)[:, None])

    rgba = np.dstack([a, (alpha * 255).round().astype(np.uint8)])
    sortie = Image.fromarray(rgba, "RGBA")
    sortie.save(ASSETS / "hero.webp", quality=88, method=6)
    return sortie.size


if __name__ == "__main__":
    print(composer(sys.argv[1]))
