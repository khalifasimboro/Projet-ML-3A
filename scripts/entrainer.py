"""Entraîne les modèles à partir de la base de descripteurs et sauvegarde modèle + artefacts.

    python scripts/entrainer.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ballguard.config import FICHIER_ARTEFACTS, FICHIER_DESCRIPTEURS, FICHIER_MODELE, FICHIER_SPECTRES  # noqa: E402
from ballguard.entrainement import entrainer  # noqa: E402

if __name__ == "__main__":
    bd = pd.read_csv(FICHIER_DESCRIPTEURS)
    spectres = np.load(FICHIER_SPECTRES)["spectres"]
    print(f"{len(bd)} signaux chargés")
    modele, artefacts = entrainer(bd, spectres)
    FICHIER_MODELE.parent.mkdir(exist_ok=True)
    joblib.dump(modele, FICHIER_MODELE, compress=3)
    joblib.dump(artefacts, FICHIER_ARTEFACTS, compress=3)

    print("\nDétection (jeu de test)")
    for l in artefacts["detection"]["table"]:
        print(f"  {l['modele']:<20} AUROC {l['AUROC']:.4f}  rappel {l['rappel']:.4f}  "
              f"fausses alarmes {l['fausses_alarmes']:.4f} ({l['nb_fausses_alarmes']})")
    c = artefacts["clustering"]
    print("Clustering (jeu de test)")
    for nom, bloc in (("sans correction", c["sans"]), ("log + standardisation", c["corrige"])):
        for m, r in bloc.items():
            print(f"  {nom:<22} {m:<13} ARI {r['ARI']:.4f}  NMI {r['NMI']:.4f}  "
                  f"silhouette {r['silhouette']:.4f}  ARI vitesse {r['ARI_vitesse']:.4f}")
    print(f"  état correctement identifié : {c['exactitude']:.1%}")
    print(f"\nModèle -> {FICHIER_MODELE} ({FICHIER_MODELE.stat().st_size // 1024} Ko)")
    print(f"Artefacts -> {FICHIER_ARTEFACTS} ({FICHIER_ARTEFACTS.stat().st_size // 1024} Ko)")
