# GT2.04–GT2.05 : intégration au notebook collectif

## Origine des données et frontière de responsabilité

- **Conservé sans modification** : `notebook/autoencodeurs_roulements.ipynb` (équipe), sa préparation de la base et ses baselines.
- **Réutilisé** : `bd_v1.pkl` contient `(bd, SPECTRES)` : 8 431 observations exploitables dans la version analysée, 26 descripteurs dans `bd`, spectres concaténés de forme `(N, 512)`.
- **Notre contribution** : `src/ae_benchmark/` (architectures V2), `src/ae_benchmark/collaborative_data.py` (adaptateur), `run_benchmark_bd_v1.py` (comparaison), `notebook/experimentation_membre_B.ipynb` (résultats).

## Méthode

1. Charger `bd_v1.pkl` sans retraiter 8 432 fichiers Excel.
2. Choisir `spectres` = `(N, 2, 256)` ou `descripteurs` = `(N, 1, 26)` ; ces **spectres ne sont pas des signaux temporels bruts**.
3. Réutiliser le split du notebook pour **screening exploratoire** et apprendre la standardisation sur train uniquement (train SAFE pour anomalie, tout train pour clustering).
4. Comparer, dans une même exécution :
   - Détection : IsolationForest et PCA vs MLP/CNN/Denoising/VAE, avec AUROC, PR-AUC, F1 et taux de fausses alarmes.
   - Clustering : KMeans direct, PCA+KMeans vs embeddings des autoencodeurs+KMeans, et option IDEC ; comparer ARI/NMI/silhouette.
5. Choisir les hyperparamètres/modèles à partir de la validation ; **ne pas utiliser test pour sélectionner**.
6. Tester finalement sur des acquisitions physiquement indépendantes **après identification vérifiée** du groupe de chaque acquisition ; vérifier l'effet des vitesses.

## Limite méthodologique bloquante pour prétendre au test final

Le notebook collectif stratifie les **fichiers d'une seconde** au hasard. Il n'est pas démontré que ces fichiers sont des acquisitions physiquement indépendantes : résultats affichés jusqu'ici non certifiés hors fuite de données. `--split-mode notebook` fonctionne uniquement en `--phase screening`. Le mode `--phase final` exige `--split-mode grouped --group-column NOM_COLONNE`, où la colonne identifie une acquisition réellement indépendante fournie ou validée par l'équipe. Les noms individuels de fichiers ne constituent pas automatiquement des groupes indépendants.

## Exemples

Depuis la racine du projet :

```powershell
python -m pip install -r requirements.txt
python -m pytest tests/test_collaborative_integration.py -q
python run_benchmark_bd_v1.py --representation spectres --models mlp,cnn,denoise,vae --phase screening --epochs 15 --task both --baseline-sklearn
python run_benchmark_bd_v1.py --representation descripteurs --models mlp --phase screening --epochs 15 --task both --out results/descripteurs
```

Puis, **une fois les vrais groupes documentés et enregistrés comme colonne dans `bd`** :

```powershell
python run_benchmark_bd_v1.py --representation spectres --models mlp,cnn --phase final --split-mode grouped --group-column acquisition_id --epochs 15
```

### Interprétation

- `results/**/comparaison_screening.csv` : tableau expérimental de la validation, pas un résultat final à publier.
- `--baseline-sklearn` réentraîne **l'architecture du notebook collectif** sur les mêmes données ; selon la taille des couches, cette option peut être lente.
- `results/**/*.pt` contient les checkpoints, conservés localement et ignorés par Git.
- `results/**/protocole_screening.json` : traçabilité du protocole.
- Les résultats du notebook d'origine sur *test* ne sont **pas comparables numériquement** aux résultats de screening sur *validation* ; il faudra refaire la comparaison à protocole identique.
- Les modèles plus avancés de V2 sont des **adaptations expérimentales**, pas des reproductions certifiées des publications citées.

## Points à clarifier avec l'équipe

- Indépendance physique des fichiers (`fichier`, acquisition, roulement, session) ; granularité réelle d'une acquisition.
- Unités physiques des régimes 10/15/20 et géométrie des roulements.
- Disponibilité des Excel bruts pour entraîner de vrais modèles temporels 1D et appliquer des filtres d'enveloppe physiques.
- Conditions de définition d'un modèle final et livrables attendus pour l'application.
## Continuer sans attendre la confirmation des acquisitions

L'analyse des noms de fichiers révèle **17 séries présumées sur 17 réparties entre
plusieurs sous-ensembles** du split original. Un apprentissage/validation avec
fichiers aléatoires peut donc surestimer les performances.

Pour commencer une étude de **généralisation inter-régimes**, utiliser le
protocole de screening **train = 10, validation = 20, test réservé = 15** :

```powershell
python run_benchmark_bd_v1.py --representation spectres --models mlp,cnn --epochs 10 --task clustering --phase screening --split-mode regime --train-regimes 10 --val-regime 20 --test-regime 15 --device cpu --out results/gt204_gt205/inter_regimes
```

Ce protocole compare les modèles sur un régime **non vu à l'entraînement**,
mais ne garantit pas une généralisation à une autre acquisition physique ou
un autre roulement. La validation est utilisée pour les choix de modèle ;
ne jamais utiliser le régime 15 (test réservé) pour optimiser ces choix.
Un test final scientifique nécessite des acquisitions indépendantes,
confirmées par l'équipe. Aucune modification de `bd_v1.pkl` n'est requise.

## Mise à jour du 03/11/2026 : livrables scientifiques renforcés

- **Notebook principal du membre B :** `notebook/experimentation_membre_B.ipynb`
  contient désormais 37 cellules pédagogiques (textes, équations, code,
  graphiques, contrôle de fuite et références scientifiques). Il réutilise
  les expériences locales ; en leur absence, il utilise les archives chiffrées
  présentes dans `reports/gt204_gt205/`.
- **Archives de chiffres réellement observés :** `reports/gt204_gt205/`.
  Leur provenance et les divergences possibles entre anciennes et nouvelles
  initialisations sont documentées dans `PROVENANCE.md`.
- **Étude bibliographique et état d'avancement :**
  `docs/ETUDE_SCIENTIFIQUE_GT204_GT205.md`.
- **Nouveau protocole non exécuté automatiquement :**
  `run_literature_ablation.py` compare CNN/DAE à MemAE/ADAE
  sur plusieurs graines et différents niveaux de perturbation spectrale.
- **Tests ajoutés :** `tests/test_literature_ablation.py` pour les
  partitions SAFE, la reproductibilité de la perturbation et un essai
  synthétique du pipeline.

### Contrôle recommandé à effectuer après `git pull --ff-only`

```powershell
python -m pytest -q
python run_literature_ablation.py --models cnn,denoise,memae,adae --seeds 0 --epochs 2 --noise-levels 0,0.15 --out results/gt204_gt205/ablation_articles_smoke
```

Puis, **après succès de ce test**, exécuter le protocole complet fourni
dans `docs/ETUDE_SCIENTIFIQUE_GT204_GT205.md`.

L'évaluation réelle de MemAE/ADAE **n'est pas prétendue terminée**
tant que l'utilisateur n'a pas exécuté cette campagne sur sa machine.
Aucune fusion de `main` n'est requise pour cette contribution.

## Résultat de l'ablation du 03/11/2026

La campagne **15 époques × trois graines** a maintenant été exécutée
avec CNN, Denoising, MemAE, ADAE et Isolation Forest. Voir
`reports/gt204_gt205/ablation_articles_moyennes_console.csv` et
`docs/DECISION_GT205_VALIDATION.md` ; le notebook du membre B lit ces
résultats locaux ou archivés et affiche le graphique comparatif.

Présélection : **VAE + K-Means en clustering**, **CNN-AE comme candidat
principal de détection** parmi les modèles évalués trois fois ; **MemAE**
reste une alternative et **VAE en détection n'est pas éliminé**.
Les écarts absolus sont faibles : ne pas revendiquer une supériorité
statistique ou une validation industrielle indépendante.
