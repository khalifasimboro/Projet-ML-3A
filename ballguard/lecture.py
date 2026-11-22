"""Lecture d'un signal brut (fichier Excel exporté par le banc de test).

Les fichiers font planter `pandas.read_excel` (feuille de styles non standard). Un .xlsx est une
archive ZIP de fichiers XML : on lit directement les valeurs dans le XML. Les colonnes sont
retrouvées par leur nom, car les fichiers n'ont pas tous le même format.
"""
import unicodedata
import zipfile
from xml.etree.ElementTree import ParseError, iterparse

import numpy as np

from .config import CANAUX, N_POINTS

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


class SignalInvalide(ValueError):
    """Le fichier ne contient pas un signal exploitable."""


def normaliser(nom):
    """' Accélération_Vert ' -> 'acceleration_vert'."""
    nom = unicodedata.normalize("NFKD", str(nom).strip().lower())
    return "".join(ch for ch in nom if not unicodedata.combining(ch))


def _lire_textes(archive):
    textes = []
    if "xl/sharedStrings.xml" in archive.namelist():
        for _, el in iterparse(archive.open("xl/sharedStrings.xml")):
            if el.tag == NS + "si":
                textes.append("".join(t.text or "" for t in el.iter(NS + "t")))
    return textes


def _lire_lignes(archive):
    textes = _lire_textes(archive)
    feuilles = sorted(n for n in archive.namelist() if n.startswith("xl/worksheets/sheet"))
    if not feuilles:
        raise SignalInvalide("le fichier ne contient aucune feuille de calcul")
    lignes = []
    for _, el in iterparse(archive.open(feuilles[0])):
        if el.tag == NS + "row":
            ligne = []
            for c in el.findall(NS + "c"):
                v = c.find(NS + "v")
                if c.get("t") == "s":
                    ligne.append(textes[int(v.text)])
                elif c.get("t") == "inlineStr":
                    ligne.append("".join(x.text or "" for x in c.iter(NS + "t")))
                else:
                    ligne.append(float(v.text) if v is not None and v.text else np.nan)
            lignes.append(ligne)
            el.clear()
    return lignes


def lire_signal(source):
    """Lit un fichier Excel brut et renvoie un tableau (25600, 2) : les deux accéléromètres.

    `source` est un chemin ou un objet fichier (upload). Lève `SignalInvalide` avec un message
    clair si le fichier n'est pas exploitable.
    """
    try:
        with zipfile.ZipFile(source) as archive:
            lignes = _lire_lignes(archive)
    except zipfile.BadZipFile as e:
        raise SignalInvalide("ce n'est pas un fichier Excel (.xlsx) valide") from e
    except (ParseError, KeyError) as e:
        raise SignalInvalide(f"fichier Excel illisible ({e})") from e

    if len(lignes) < 2:
        raise SignalInvalide("le fichier est vide")
    entete = [normaliser(n) for n in lignes[0]]
    manquants = [c for c in CANAUX if c not in entete]
    if manquants:
        raise SignalInvalide(
            f"capteur(s) absent(s) : {', '.join(manquants)}. Colonnes trouvées : {', '.join(map(str, lignes[0]))}")
    idx = [entete.index(c) for c in CANAUX]
    n_cols = max(idx) + 1
    donnees = [ligne for ligne in lignes[1:] if len(ligne) >= n_cols]
    if len(donnees) != N_POINTS:
        raise SignalInvalide(f"{len(donnees)} points au lieu de {N_POINTS} (1 seconde à 25,6 kHz attendue)")
    try:
        sig = np.array([[ligne[i] for i in idx] for ligne in donnees], dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise SignalInvalide("les colonnes des capteurs contiennent des valeurs non numériques") from e
    if not np.isfinite(sig).all():
        raise SignalInvalide("le signal contient des valeurs manquantes")
    return sig
