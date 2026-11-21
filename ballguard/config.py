"""Constantes et chemins du projet."""
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent

# Acquisition
FS = 25_600                 # fréquence d'échantillonnage (Hz)
N_POINTS = 25_600           # 1 seconde de signal
CANAUX = ["acceleration_vertbleu", "acceleration_vert"]
SUFFIXE = {"acceleration_vertbleu": "vb", "acceleration_vert": "v"}
N_BANDES = 8                # bandes de fréquence de 1,6 kHz

# Étiquettes
CLASSES = ["SAFE", "BALL", "INNER_RACE", "OUTER_RACE", "COMBINED"]
VITESSES = [10, 15, 20]
META_COLS = ["classe", "vitesse", "fichier", "numero", "split"]

# Modélisation
SEED = 42
K = 5                       # nombre d'états mécaniques
PERCENTILE_SEUIL = 99       # seuil = 99e percentile des scores des signaux sains de validation

# Fichiers
DOSSIER_DONNEES_BRUTES = RACINE / "data"                 # hors dépôt (4 Go)
FICHIER_DESCRIPTEURS = RACINE / "donnees" / "descripteurs.csv"
FICHIER_SPECTRES = RACINE / "donnees" / "spectres.npz"
FICHIER_MODELE = RACINE / "modeles" / "modele_ballguard.joblib"
FICHIER_ARTEFACTS = RACINE / "modeles" / "artefacts.joblib"
FICHIER_VITESSE = RACINE / "modeles" / "vitesse.joblib"      # reconnaissance de la vitesse à partir des descripteurs
DOSSIER_EXEMPLES = RACINE / "exemples"
