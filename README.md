# Ballguard

Diagnostic de roulements par autoencodeurs : **détection d'anomalie** et **clustering** à partir de signaux vibratoires, appliqués à la maintenance prédictive.

Projet ML 3A, sujet 6, Groupe 4, École Centrale Casablanca.

## Ce que fait le projet

À partir d'une seconde de vibration (fichier Excel du banc de test), Ballguard :

1. calcule **26 descripteurs** du signal (temps et fréquence) ;
2. **détecte** si le roulement est défectueux, avec un autoencodeur entraîné sur les seuls roulements sains ;
3. **identifie le type de défaut** (bille, bague intérieure, bague extérieure, combiné) par k-means sur l'espace latent d'un second autoencodeur ;
4. **explique** la décision : contribution de chaque descripteur à l'erreur de reconstruction.

## Structure du dépôt

| Dossier | Contenu |
|---|---|
| `app/` | Application Streamlit (7 pages, thèmes clair et sombre, français et anglais) |
| `ballguard/` | Backend Python : lecture des signaux, descripteurs, modèles, entraînement, diagnostic |
| `scripts/` | Construction de la base, entraînement, préparation des exemples |
| `tests/` | Tests automatiques du backend |
| `notebooks/` | Notebook d'analyse de référence (`autoencodeurs_roulements_v4.ipynb`) et sa version archivée |
| `donnees/` | Base de descripteurs (`descripteurs.csv`) et spectres (`spectres.npz`) |
| `modeles/` | Modèle entraîné et artefacts affichés par l'application |
| `exemples/` | 15 signaux bruts du banc de test (un par état et par vitesse) |
| `maquette/` | Maquette de navigation de l'application |
| `planning/`, `rapport/` | Planning et rapport du projet |

## Installation

Python 3.11 est requis (les modèles sont sauvegardés avec scikit-learn 1.6.1). PyTorch est installé en version CPU (environ 200 Mo) pour le CNN-AE spectral.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

Sous Linux ou macOS, l'activation est `source .venv/bin/activate`.

## Lancer l'application

```bash
streamlit run app/main.py
```

L'application s'ouvre sur http://localhost:8501. Elle n'a besoin que des fichiers du dépôt.

## Reproduire les résultats

Le modèle et les artefacts se régénèrent à partir de la base de descripteurs (environ 2 minutes) :

```bash
python scripts/entrainer.py
```

Le script affiche les scores obtenus sur le jeu de test et écrit `modeles/modele_ballguard.joblib` et `modeles/artefacts.joblib`. La graine aléatoire est fixée (`SEED = 42`).

Pour vérifier le backend :

```bash
pip install pytest
```

```bash
pytest
```

### Repartir des signaux bruts

Les 8 432 fichiers bruts (4 Go) ne sont pas dans le dépôt. Si vous les avez, placez-les dans `data/` avec un sous-dossier par état puis par vitesse (par exemple `data/SAFE/safe_15_speed/`), puis :

```bash
python scripts/construire_base.py
```

```bash
python scripts/preparer_exemples.py
```

Le premier script lit chaque fichier, fixe le découpage train / validation / test **avant tout prétraitement**, calcule les descripteurs et écrit `donnees/`.

## Données

- **Source** : banc de test de roulements. 5 états mécaniques (sain, défaut de bille, bague intérieure, bague extérieure, défaut combiné), 3 vitesses (10, 15, 20), 2 accéléromètres. Chaque acquisition dure 1 seconde, échantillonnée à 25,6 kHz (25 600 points).
- **Traitement** : chaque signal est centré puis résumé, pour chaque capteur, par 5 descripteurs temporels (RMS, kurtosis, facteur de crête, asymétrie, crête-à-crête) et 8 énergies de bande (log10, densité spectrale de Welch, bandes de 1,6 kHz). Le spectre complet (256 raies par capteur) est aussi conservé.
- **Découpage** : 70 % / 15 % / 15 %, stratifié par (état, vitesse). Les normalisations sont apprises sur l'entraînement, le seuil de détection sur la validation, les scores mesurés sur le test.
- Un fichier illisible a été écarté : la base contient 8 431 signaux.

## Méthode

L'application propose trois modèles de diagnostic : MLP-AE + K-means (modèle retenu, explicable par descripteur), MLP-AE + GMM et CNN-AE spectral + K-means. Les deux derniers viennent de l'étude comparative V6 du groupe (branche `feature/gt204-gt205-membre-b`) ; ils sont réentraînés par `scripts/entrainer.py`. Les tableaux de cette étude sont dans `donnees/etude_v6/` et affichés dans la page Performances.

| Tâche | Modèle de référence | Modèle avancé |
|---|---|---|
| Détection d'anomalie | Isolation Forest, ACP (erreur de reconstruction linéaire) | Autoencodeur sur les descripteurs, autoencodeur sur le spectre |
| Clustering | k-means sur les descripteurs | Autoencodeur + k-means sur l'espace latent |

La vitesse de rotation est un facteur de confusion : sans correction, le clustering regroupe les signaux par vitesse. La solution retenue passe 8 descripteurs au logarithme puis standardise séparément chaque vitesse.

**Interprétabilité** : décomposition de l'erreur de reconstruction par descripteur, importance par permutation (détection et diagnostic), analyse des fausses alarmes.

Le détail de l'analyse, des choix et des résultats est dans `notebooks/autoencodeurs_roulements_v4.ipynb`.

## Limites connues

- Chaque couple (état, vitesse) est une série d'acquisitions consécutives : le découpage aléatoire place des signaux presque identiques dans l'entraînement et le test, donc les scores sont optimistes.
- La vitesse doit être connue et faire partie des trois vitesses apprises.
- Un seul banc de test : la généralisation à d'autres machines n'est pas mesurée.

## Équipe

KINDO Harouna, HODONOU Firmin, DOMBOUÉ Tidiane, SIMBORO Rydouane.
