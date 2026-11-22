"""Ajoute l'autoencodeur spectral (modèle de détection retenu) au modèle existant, sans réentraîner le reste.

Le modèle de détection retenu est l'autoencodeur sur le spectre (0 fausse alarme sur le jeu de test). Il était
évalué par scripts/entrainer.py mais pas enregistré. Ce script le réentraîne à l'identique (même graine, mêmes
données), vérifie que son seuil est exactement celui publié dans les artefacts, puis l'ajoute :
  - au modèle (modeles/modele_ballguard.joblib, clé "detection_spectre") ;
  - aux artefacts (scores et taux d'alarme par état et vitesse, pour la page Performances).
Un nouvel entraînement complet (scripts/entrainer.py) produit directement ces mêmes clés.

    .venv/Scripts/python scripts/ajouter_detection_spectre.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ballguard.config import (CLASSES, FICHIER_ARTEFACTS, FICHIER_DESCRIPTEURS, FICHIER_MODELE,  # noqa: E402
                              FICHIER_SPECTRES, PERCENTILE_SEUIL, VITESSES)
from ballguard.modeles import creer_ae, erreur_reconstruction  # noqa: E402

if __name__ == "__main__":
    modele, artefacts = joblib.load(FICHIER_MODELE), joblib.load(FICHIER_ARTEFACTS)
    bd = pd.read_csv(FICHIER_DESCRIPTEURS)
    spectres = np.load(FICHIER_SPECTRES)["spectres"]
    classe, vitesse = bd.classe.to_numpy(), bd.vitesse.to_numpy()
    m_train, m_val, m_test = [(bd.split == s).to_numpy() for s in ("train", "val", "test")]
    safe = classe == "SAFE"

    scaler = StandardScaler().fit(spectres[m_train & safe])
    X = scaler.transform(spectres)
    ae = creer_ae((128, 16, 128)).fit(X[m_train & safe], X[m_train & safe])
    score = erreur_reconstruction(ae, X)
    seuil = float(np.percentile(score[m_val & safe], PERCENTILE_SEUIL))
    publie = artefacts["detection"]["seuils"]["AE (spectre)"]
    if not np.isclose(seuil, publie, rtol=0, atol=1e-9):
        sys.exit(f"Seuil différent de celui publié ({seuil} au lieu de {publie}) : rien n'est modifié.")

    modele["detection_spectre"] = {"scaler": scaler, "autoencodeur": ae, "seuil": seuil}
    artefacts["detection"]["score_spec"] = score.astype(np.float32)
    artefacts["detection"]["taux_alarme_spec"] = [
        [float((score[m_test & (classe == c) & (vitesse == v)] > seuil).mean()) for v in VITESSES] for c in CLASSES]
    joblib.dump(modele, FICHIER_MODELE, compress=3)
    joblib.dump(artefacts, FICHIER_ARTEFACTS, compress=3)
    fa = int((score[m_test & safe] > seuil).sum())
    print(f"AE spectral ajouté : seuil {seuil:.4f} (identique au publié), fausses alarmes sur le test : {fa}")
