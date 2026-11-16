"""Copie dans exemples/ un signal brut du jeu de test par état et par vitesse (15 fichiers).

    python scripts/preparer_exemples.py
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ballguard.config import CLASSES, DOSSIER_DONNEES_BRUTES, DOSSIER_EXEMPLES, FICHIER_DESCRIPTEURS, VITESSES  # noqa: E402
from ballguard.inventaire import inventorier  # noqa: E402

if __name__ == "__main__":
    bd = pd.read_csv(FICHIER_DESCRIPTEURS)
    chemins = inventorier(DOSSIER_DONNEES_BRUTES).set_index("fichier").chemin
    DOSSIER_EXEMPLES.mkdir(exist_ok=True)
    lignes = []
    for c in CLASSES:
        for v in VITESSES:
            choix = bd[(bd.split == "test") & (bd.classe == c) & (bd.vitesse == v)].sort_values("numero").iloc[0]
            shutil.copy(chemins[choix.fichier], DOSSIER_EXEMPLES / choix.fichier)
            lignes.append({"fichier": choix.fichier, "classe": c, "vitesse": v})
    pd.DataFrame(lignes).to_csv(DOSSIER_EXEMPLES / "index.csv", index=False)
    print(f"{len(lignes)} exemples copiés dans {DOSSIER_EXEMPLES}")
