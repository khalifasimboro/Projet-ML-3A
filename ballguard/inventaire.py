"""Inventaire du dossier de données brutes.

Structure attendue : <dossier>/<ETAT ...>/<etat_VITESSE ...>/<fichier>.xlsx
Les noms de dossiers varient ("BALL default/ball_10 speed", "SAFE/safe_15_speed") : l'état et la
vitesse sont retrouvés de façon tolérante.
"""
import re
from pathlib import Path

import pandas as pd

from .config import CLASSES, VITESSES


def _classe(nom_dossier):
    cle = re.sub(r"[^A-Z]", "", nom_dossier.upper().replace("DEFAULT", ""))
    for c in CLASSES:
        if cle == c.replace("_", ""):
            return c
    raise ValueError(f"état inconnu pour le dossier « {nom_dossier} »")


def _vitesse(nom_dossier):
    for n in re.findall(r"\d+", nom_dossier):
        if int(n) in VITESSES:
            return int(n)
    raise ValueError(f"vitesse inconnue pour le dossier « {nom_dossier} »")


def numero_acquisition(nom_fichier):
    """'safe_1489_212.xlsx' -> 212 : ordre d'enregistrement dans la série."""
    m = re.search(r"_(\d+)\.xlsx$", nom_fichier)
    return int(m.group(1)) if m else -1


def inventorier(dossier):
    """Renvoie une table avec une ligne par fichier : classe, vitesse, fichier, numero, chemin."""
    lignes = []
    for f in sorted(Path(dossier).glob("*/*/*.xlsx")):
        if f.name.startswith("~$"):                     # fichiers temporaires d'Excel
            continue
        lignes.append({"classe": _classe(f.parent.parent.name), "vitesse": _vitesse(f.parent.name),
                       "fichier": f.name, "numero": numero_acquisition(f.name), "chemin": str(f)})
    return pd.DataFrame(lignes)
