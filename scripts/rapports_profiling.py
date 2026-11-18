"""Génère les deux rapports YData Profiling affichés dans la page Exploration.

1. Signaux bruts du banc : les 15 signaux d'exemple publiés (exemples/), point par point.
   La base brute complète (data/, 4 Go, privée) n'est pas publiée : on décrit l'échantillon public.
2. Base des 26 descripteurs : donnees/descripteurs.csv (une ligne par acquisition).

ydata-profiling n'est PAS une dépendance de l'application (versions de numpy/pandas imposées) :
on l'installe dans un environnement à part, puis on lance ce script depuis la racine du dépôt :

    python -m venv venv_yd
    venv_yd/Scripts/python -m pip install ydata-profiling setuptools
    venv_yd/Scripts/python scripts/rapports_profiling.py

Les rapports HTML produits (app/assets/rapports/) sont statiques et versionnés.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from ballguard.config import CANAUX, FS, N_BANDES, SUFFIXE  # noqa: E402
from ballguard.lecture import lire_signal  # noqa: E402

SORTIE = RACINE / "app" / "assets" / "rapports"
ORANGE = "#F27A1F"

CAPTEUR = {"vb": "accéléromètre vert-bleu", "v": "accéléromètre vert"}
TEMPORELS = {
    "rms": "Niveau vibratoire moyen (RMS) : racine de la moyenne des carrés du signal centré, sqrt(mean(x²)).",
    "kurtosis": "Impulsivité (kurtosis) : moment d'ordre 4 normalisé, E[(x-µ)^4]/σ^4. Vaut 3 pour un bruit gaussien, "
                "monte quand des chocs apparaissent.",
    "crete": "Facteur de crête : pic absolu / RMS, max|x| / RMS. Mesure la hauteur des chocs par rapport au niveau moyen.",
    "skew": "Asymétrie (skewness) : moment d'ordre 3 normalisé, E[(x-µ)^3]/σ^3. 0 si le signal est symétrique.",
    "pkpk": "Amplitude crête-à-crête : max(x) - min(x).",
}


def _description_bande(b):
    debut, fin = (b * 1600 + 50) / 1000, (b + 1) * 1.6
    return (f"Énergie spectrale de la bande {b + 1} ({str(debut).replace('.', ',')} à {str(fin).replace('.', ',')} kHz) : "
            "log10 de la somme de la densité spectrale de puissance (méthode de Welch, fenêtres de 512 points, "
            "résolution 50 Hz) sur les 32 fréquences de la bande.")


NOMS_TEMPORELS = {"rms": "Niveau vibratoire", "kurtosis": "Impulsivité", "crete": "Facteur de crête",
                  "skew": "Asymétrie", "pkpk": "Amplitude crête-à-crête"}


def nom_lisible(code):
    """Même nommage que l'application (app/ui/i18n.py, nom_desc) : 'kurtosis_vb' -> 'Impulsivité · VB'."""
    if "_" not in code:
        return code
    nom, s = code.rsplit("_", 1)
    if nom.startswith("bande"):
        b = int(nom[5:])
        lo, hi = ((b - 1) * 1600 + 50) / 1000, b * 1.6
        nom = f"Énergie {round(lo, 2):g}–{round(hi, 2):g} kHz".replace(".", ",")
    else:
        nom = NOMS_TEMPORELS[nom]
    return f"{nom} · {s.upper()}"


def descriptions():
    d = {"classe": "État mécanique du roulement (étiquette) : SAFE sain, BALL bille, INNER_RACE bague intérieure, "
                   "OUTER_RACE bague extérieure, COMBINED défauts combinés.",
         "vitesse": "Vitesse de rotation du banc (consigne 10, 15 ou 20).",
         "fichier": "Nom du fichier Excel d'origine.",
         "numero": "Numéro de l'acquisition dans la série.",
         "split": "Jeu d'appartenance : train (70 %), validation (15 %), test (15 %), découpage stratifié."}
    for s, capteur in CAPTEUR.items():
        for nom, texte in TEMPORELS.items():
            d[nom_lisible(f"{nom}_{s}")] = f"{texte} Capteur : {capteur}. Code : {nom}_{s}."
        for b in range(N_BANDES):
            d[nom_lisible(f"bande{b + 1}_{s}")] = f"{_description_bande(b)} Capteur : {capteur}. Code : bande{b + 1}_{s}."
    return d


def options(titre, description):
    return dict(
        title=titre,
        dataset={"description": description, "creator": "Ballguard", "url": ""},
        html={"style": {"primary_colors": [ORANGE, "#1F2A44", "#7C8DB0"], "full_width": True}},
        interactions={"continuous": False},          # 26 x 26 nuages de points : trop lourd et peu lisible
        missing_diagrams={"bar": False, "matrix": False, "heatmap": False},
        correlations={"auto": {"calculate": True}, "pearson": {"calculate": True}, "spearman": {"calculate": False},
                      "kendall": {"calculate": False}, "phi_k": {"calculate": False}, "cramers": {"calculate": False}},
        progress_bar=False,
    )


def signaux_bruts():
    index = pd.read_csv(RACINE / "exemples" / "index.csv")
    parts = []
    for _, ligne in index.iterrows():
        sig = lire_signal(RACINE / "exemples" / ligne.fichier)
        parts.append(pd.DataFrame({
            "fichier": ligne.fichier, "classe": ligne.classe, "vitesse": ligne.vitesse,
            "temps_s": np.arange(len(sig)) / FS,
            CANAUX[0]: sig[:, 0], CANAUX[1]: sig[:, 1]}))
    df = pd.concat(parts, ignore_index=True)
    df["classe"] = df["classe"].astype("category")
    df["vitesse"] = df["vitesse"].astype("category")
    desc = {"fichier": "Fichier Excel d'exemple d'où vient le point.",
            "classe": descriptions()["classe"], "vitesse": descriptions()["vitesse"],
            "temps_s": "Instant de la mesure dans l'acquisition (0 à 1 s, 25 600 points à 25,6 kHz).",
            CANAUX[0]: "Accélération brute mesurée par l'accéléromètre vert-bleu (unité du banc).",
            CANAUX[1]: "Accélération brute mesurée par l'accéléromètre vert (unité du banc)."}
    o = options("Ballguard · Signaux bruts du banc de test",
                f"{len(index)} acquisitions d'une seconde (une par état et par vitesse disponibles), "
                f"soit {len(df):,} mesures. Échantillon public de la base complète (8 431 acquisitions).".replace(",", " "))
    o["samples"] = {"head": 5, "tail": 5, "random": 0}
    rapport = ProfileReport(df, variables={"descriptions": desc}, **o)
    rapport.to_file(SORTIE / "signaux_bruts.html")


def base_descripteurs():
    df = pd.read_csv(RACINE / "donnees" / "descripteurs.csv")
    df = df.rename(columns=nom_lisible)
    for c in ("classe", "vitesse", "split"):
        df[c] = df[c].astype("category")
    o = options("Ballguard · Base des 26 descripteurs",
                f"{len(df)} acquisitions, chacune résumée par 13 descripteurs pour chacun des 2 capteurs "
                "(5 statistiques temporelles + 8 énergies de bande de fréquence).")
    o["samples"] = {"head": 5, "tail": 5, "random": 0}
    rapport = ProfileReport(df, variables={"descriptions": descriptions()}, **o)
    rapport.to_file(SORTIE / "descripteurs.html")


if __name__ == "__main__":
    SORTIE.mkdir(parents=True, exist_ok=True)
    base_descripteurs()
    signaux_bruts()
    for f in sorted(SORTIE.glob("*.html")):
        print(f.name, f"{f.stat().st_size / 1e6:.1f} Mo")
