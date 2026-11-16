"""Construit la base de descripteurs à partir des fichiers bruts.

    python scripts/construire_base.py                 # lit data/ (plusieurs minutes, tous les cœurs)
    python scripts/construire_base.py --depuis-pkl bd_v1.pkl   # convertit la base du notebook

Écrit donnees/descripteurs.csv (une ligne par signal) et donnees/spectres.npz.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ballguard.config import DOSSIER_DONNEES_BRUTES, FICHIER_DESCRIPTEURS, FICHIER_SPECTRES  # noqa: E402
from ballguard.descripteurs import FEATURES, extraire  # noqa: E402
from ballguard.entrainement import decouper  # noqa: E402
from ballguard.inventaire import inventorier, numero_acquisition  # noqa: E402


def _extraire_sans_planter(chemin):
    try:
        return extraire(chemin)
    except Exception as e:                      # un fichier illisible est écarté, sans arrêter le calcul
        return f"ERREUR : {e}"


def depuis_fichiers_bruts(dossier):
    meta = inventorier(dossier)
    print(f"{len(meta)} fichiers trouvés dans {dossier}")
    meta["split"] = decouper(meta)              # découpage AVANT tout prétraitement
    res = Parallel(n_jobs=-1, verbose=5)(delayed(_extraire_sans_planter)(c) for c in meta.chemin)
    ok = np.array([not isinstance(r, str) for r in res])
    for chemin, r in zip(meta.chemin[~ok], [r for r in res if isinstance(r, str)]):
        print("Écarté :", chemin, "->", r)
    res = [r for r in res if not isinstance(r, str)]
    bd = pd.concat([meta[ok].drop(columns="chemin").reset_index(drop=True), pd.DataFrame([r[0] for r in res])], axis=1)
    return bd, np.stack([r[1] for r in res])


def depuis_pkl(chemin):
    """Reprend la base calculée par le notebook (mêmes valeurs, même découpage)."""
    bd, spectres = pd.read_pickle(chemin)
    bd = bd.drop(columns=[c for c in ("chemin", "entete") if c in bd.columns])
    bd.insert(3, "numero", bd.fichier.map(numero_acquisition))
    return bd, spectres


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--donnees", default=DOSSIER_DONNEES_BRUTES)
    p.add_argument("--depuis-pkl")
    a = p.parse_args()
    bd, spectres = depuis_pkl(a.depuis_pkl) if a.depuis_pkl else depuis_fichiers_bruts(a.donnees)
    bd = bd[["classe", "vitesse", "fichier", "numero", "split"] + FEATURES]
    FICHIER_DESCRIPTEURS.parent.mkdir(exist_ok=True)
    bd.to_csv(FICHIER_DESCRIPTEURS, index=False, float_format="%.9g")
    np.savez_compressed(FICHIER_SPECTRES, spectres=spectres.astype(np.float32))
    print(f"{len(bd)} signaux x {len(FEATURES)} descripteurs -> {FICHIER_DESCRIPTEURS}")
    print(f"spectres {spectres.shape} -> {FICHIER_SPECTRES}")
