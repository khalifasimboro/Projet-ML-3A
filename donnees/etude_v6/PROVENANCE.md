# Provenance

Tableaux de résultats copiés sans modification depuis la branche `feature/gt204-gt205-membre-b`
(commit 8f8ca2a), dossier `results_v6_autonome/` : étude comparative V6 du groupe
(notebook `Autoencodeurs_Roulements_Comparaison_V6_Tidiane.ipynb`).

Le nom de chaque fichier reprend son chemin d'origine (`etude_a_detection.csv` = `etude_a/detection.csv`).
Ces tableaux sont affichés tels quels dans l'onglet « Étude comparative » de la page Performances.

Les modèles de cette étude utilisés par l'application (MLP-AE + GMM, CNN-AE spectral) ne sont pas copiés :
ils sont réentraînés par `scripts/entrainer.py` (même architecture, mêmes prétraitements, même graine),
pour rester compatibles avec la version de scikit-learn du projet.
