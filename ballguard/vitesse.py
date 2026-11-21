"""Reconnaissance de la vitesse de rotation à partir des 26 descripteurs.

Le diagnostic normalise les descripteurs selon la vitesse : une vitesse mal renseignée fausse le type de défaut.
Une forêt aléatoire apprise sur le jeu d'entraînement retrouve la vitesse sur 100 % des signaux de test
(1 265 signaux, les 5 états), ce qui permet de la proposer automatiquement à l'import.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .config import FICHIER_DESCRIPTEURS, FICHIER_VITESSE, SEED
from .descripteurs import FEATURES


def entrainer(chemin_base=FICHIER_DESCRIPTEURS, sortie=FICHIER_VITESSE):
    bd = pd.read_csv(chemin_base)
    tr, te = bd[bd.split == "train"], bd[bd.split == "test"]
    foret = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    foret.fit(tr[FEATURES].to_numpy(), tr.vitesse.to_numpy())
    exactitude = float(accuracy_score(te.vitesse, foret.predict(te[FEATURES].to_numpy())))
    joblib.dump({"foret": foret, "features": FEATURES, "exactitude_test": exactitude, "n_test": int(len(te))},
                sortie, compress=3)
    return exactitude


def charger(chemin=FICHIER_VITESSE):
    return joblib.load(chemin)


def estimer(modele_vitesse, desc):
    """desc : dict des 26 descripteurs. Renvoie (vitesse estimée, confiance entre 0 et 1)."""
    x = np.array([[desc[f] for f in modele_vitesse["features"]]])
    proba = modele_vitesse["foret"].predict_proba(x)[0]
    k = int(proba.argmax())
    return int(modele_vitesse["foret"].classes_[k]), float(proba[k])
