# GT2.04–GT2.05 — Étude scientifique, architecture et expérimentation réelle

> **Statut au 03/11/2026 :** première campagne sur `bd_v1.pkl` réalisée par l'utilisateur ; étude complémentaire MemAE/ADAE **exécutée sur les données réelles** (15 époques, trois graines, niveaux de bruit numérique 0 / 0,15 / 0,30). Conclusions **provisoires**, et aucune validation finale sur acquisitions indépendantes.
>
> **Branche de travail :** `integration/autoencodeurs` (dépôt personnel). Le notebook collectif et `main` ne sont pas modifiés.

## 1. Objectifs

- **GT2.04 :** comparer des modèles de clustering non supervisé et de détection d'anomalies, contrôler le protocole, la stabilité et les choix d'implémentation.
- **GT2.05 :** produire une présélection explicite et argumentée, en indiquant limites, métriques et conditions de validation.
- Ne pas déclarer un modèle supérieur sur la seule meilleure graine, sur une AUROC saturée ou sur un split par fichiers potentiellement corrélés.

## 2. Origine et contenu des données

Le notebook de l'équipe `notebook/autoencodeurs_roulements.ipynb` produit `bd_v1.pkl`.
Sur la version examinée : **8 431 observations** utilisables, 26 descripteurs et
spectres préparés `(N, 512)` regroupés en `(N, 2, 256)`.

Les canaux représentent deux capteurs et leurs **log-PSD**, **pas une série temporelle
à 256 pas**. Les axes et niveaux de bruit ne doivent donc pas être
interprétés comme des fréquences ou des bruits de capteurs physiques sans
les informations de préparation.

| Régime | SAFE | BALL | COMBINED | INNER_RACE | OUTER_RACE | Total |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 523 | 566 | 550 | 556 | 527 | 2 722 |
| 20 | 523 | 555 | 519 | 562 | 515 | 2 674 |
| 15, réservé | 664 | 576 | 575 | 658 | 562 | 3 035 |

**Alerte de fuite :** 17 préfixes de fichiers présumés sur 17 traversent plusieurs
splits du notebook initial. Il n'est pas prouvé que ces préfixes correspondent
à des acquisitions physiquement indépendantes ; leurs mélanges nécessitent
vérification. La campagne en cours préfère `train=10`, `val=20` et
**réserve le régime 15**. Cette séparation teste le transfert inter-régimes,
sans garantir l'indépendance entre roulements, essais ou sessions.

## 3. Références méthodologiques

### Clustering (mêmes paramètres de protocole)

- K-Means sur spectres, PCA + K-Means ;
- MLP-AE, CNN-AE, Denoising CNN-AE, VAE ; clustering K-Means dans l'espace latent ;
- IDEC : raffinement de l'espace latent par reconstruction et KL des affectations.

Mesures : **ARI** (accord avec les états connus, corrigé du hasard),
**NMI** (information commune) et **silhouette** (séparation géométrique).
Une grande silhouette peut accompagner un ARI faible : ne pas choisir un
modèle sur la silhouette seule.

### Détection d'anomalies

- Les détecteurs apprennent **uniquement sur SAFE du régime 10** ;
- scaler ajusté sur SAFE du train uniquement ;
- trois groupes SAFE **disjoints** dans la validation du régime 20 :
  sélection de la meilleure époque, calibration du seuil, mesure des fausses alarmes ;
- calibration : quantile 99 % des scores SAFE dédiés ;
- métriques : AUROC, average precision, PR-AUC, F1, rappel, faux positifs.

La proportion de défauts dans l'évaluation est forte. Une précision/F1 très
élevée ne constitue pas, à elle seule, une preuve de qualité en production.

## 4. Résultats déjà observés (validation exploratoire)

### Clustering

| Modèle | ARI moyen | Écart-type échantillon | Nombre de graines |
|---|---:|---:|---:|
| VAE + K-Means | **0,711892** | 0,092028 | 5 |
| VAE + IDEC | 0,654297 | 0,058573 | 5 |
| K-Means sur spectres | 0,609269 | — | Une référence |

VAE + K-Means est un **candidat prioritaire**, mais IDEC ne l'améliore pas
systématiquement et la variance du VAE reste notable.

### Détection corrigée, une exécution

| Modèle | AUROC | F1 | Fausses alarmes |
|---|---:|---:|---:|
| PCA | 1,000 | 0,998375 | 7 / 175 SAFE |
| Isolation Forest | 1,000 | **0,999303** | **3 / 175 SAFE** |
| MLP-AE | 1,000 | 0,998144 | 8 / 175 SAFE |
| CNN-AE | 1,000 | 0,998144 | 8 / 175 SAFE |
| Denoising AE | 1,000 | 0,998607 | 6 / 175 SAFE |
| VAE | 1,000 | **0,999303** | **3 / 175 SAFE** |

Le VAE et Isolation Forest **ne sont pas départagés** dans cette validation.
Les scores obtenus avant correction du partage SAFE ne sont pas retenus
pour la comparaison consolidée.

## 5. Articles réellement utilisés et lien explicite au code

| Publication | Idée vérifiée dans l'article | Notre adaptation | Validation réelle |
|---|---|---|---|
| [Guo et al. (2017), IJCAI](https://doi.org/10.24963/ijcai.2017/243) | IDEC : reconstruction et KL des affectations de Student | `src/ae_benchmark/deep_clustering.py` ; raffinement du VAE préentraîné | **Oui**, cinq graines |
| [Gong et al. (2019), ICCV](https://openaccess.thecvf.com/content_ICCV_2019/html/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.html) | Mémoire de représentations normales et lecture parcimonieuse | `SparseMemAE`, top-k, mémoire et entropie ; **pas** le hard-shrinkage exact | **Exécuté** : trois graines, quinze époques |
| [Lu, Zhou & He (2024), Electronics](https://doi.org/10.3390/electronics13122403) | Autoencodeur débruiteur avec unités de rétrécissement, raccourcis et dropout | `AdaptiveDenoisingAE`, version 1D adaptée ; perte SmoothL1 | **Exécuté** : trois graines, quinze époques |
| [Lu et al. (2025), IEEE TIM](https://doi.org/10.1109/TIM.2025.3546365) | Suivi d'ordres et mémoire en cascade pour conditions variables | `CascadedMemAE` sans suivi angulaire ; **prototype partiel** | Test facultatif, non réalisé |

Les modèles avancés sont **inspirés** des publications. Les architectures,
pertes, hyperparamètres, signaux et jeux de données ne sont pas identiques :
**ne pas reprendre les performances publiées** comme nos performances.

## 6. Ablation scientifique supplémentaire : hypothèses et protocole

Le nouveau script `run_literature_ablation.py` est conçu pour répondre à :

- **H1 / MemAE :** mémoire de normalité `memae` vs CNN-AE `cnn`, amélioration possible en taux de fausses alarmes et rappel.
- **H2 / ADAE :** adaptation de débruitage `adae` vs CNN débruiteur `denoise`, en conditions de spectres propres et perturbés.
- **H3 / facultative CMAE :** mémoire à deux échelles, sans prétendre utiliser le suivi d'ordres de l'article.

La comparaison garde exactement les mêmes régimes et fenêtres SAFE,
réinitialise la graine avant chaque construction d'architecture et enregistre
le nombre de paramètres et le temps d'entraînement.
**Avertissement important :** `adae` et `denoise` diffèrent par
l'architecture **et** la fonction de perte (SmoothL1 vs MSE). Un gain ne
peut donc pas être attribué isolément aux unités adaptatives.

### Perturbations exploratoires

Le script injecte \(\epsilon\sim\mathcal N(0,\sigma^2)\) dans le
**log-spectre standardisé**, à \(\sigma\in\{0,0.15,0.30\}\), uniquement
sur les données d'évaluation du régime 20.
La calibration du seuil demeure propre et inchangée, ce qui révèle
la sensibilité du détecteur à la modification des entrées.

**Cela ne simule pas fidèlement** du bruit temporel de capteur, un bruit
de roulement réel ou une altération physique de la mesure. Une vraie
étude de débruitage vibratoire exige les signaux bruts et un protocole
de bruit physique.

### Exécution recommandée (PowerShell / `.venv`)

```powershell
git pull --ff-only
python -m pytest -q

# Test technique court : ne sert pas de résultat scientifique
python run_literature_ablation.py --models cnn,denoise,memae,adae --seeds 0 --epochs 2 --noise-levels 0,0.15 --out results/gt204_gt205/ablation_articles_smoke

# Campagne scientifique supplémentaire, après validation du test court
python run_literature_ablation.py --models cnn,denoise,memae,adae --seeds 0,7,21 --epochs 15 --noise-levels 0,0.15,0.30 --out results/gt204_gt205/ablation_articles
```

Sorties : `resultats_par_graine_et_bruit.csv`, `taux_detection_par_classe.csv`,
`historiques_entrainement.csv`, `synthese_moyennes.csv`,
`fausses_alarmes_selon_bruit.png` et `protocole.json`.
Le notebook détaillé affiche les sorties seulement **si elles existent**.

## 7. Résultats nouveaux et décision actualisée

La campagne complémentaire a été **exécutée le 03/11/2026** et a produit
des résultats sur la base préparée par l'équipe : 3 graines (0, 7, 21),
15 époques, CNN-AE, Denoising AE, MemAE, ADAE et Isolation Forest.
Voir la transcription fidèle du tableau console dans
`reports/gt204_gt205/ablation_articles_moyennes_console.csv`,
les fichiers machine originaux dans `results/gt204_gt205/ablation_articles/`,
et l'analyse détaillée dans `docs/DECISION_GT205_VALIDATION.md`.

**Sans perturbation**, fausses alarmes moyennes :
CNN-AE 0,5714 %, MemAE 0,7619 %, ADAE 0,9524 %, Denoising AE 1,3333 %,
Isolation Forest 1,3333 %. Tous les AE ont AUROC moyenne 1,0 et rappel
moyen 1,0 sur cette validation.

**Sous perturbation σ = 0,30**, MemAE reste à 0,7619 % et ADAE
à 0,9524 %, CNN-AE atteint 0,9524 %. La perturbation est uniquement
appliquée aux log-spectres normalisées, sans signification physique
assurée de bruit de capteur.

La sélection proposée est donc **VAE + K-Means pour le clustering** et
**CNN-AE comme candidat principal pour la détection sur ce protocole**.
MemAE reste une alternative ; VAE n'a pas été réévalué pour la détection
avec trois graines homogènes, et les différences sont trop faibles
pour conclure à une supériorité significative.

## 8. Conditions pour déclarer GT2.05 entièrement clos

1. **Réalisé :** les métriques MemAE/ADAE ont été intégrées au rapport et au notebook sans remplacer les anciennes expériences.
2. **Réalisé pour les trois graines :** moyennes et dispersions intégrées ; les sorties locales par classe restent disponibles pour analyse supplémentaire.
3. **Réalisé à titre exploratoire :** la décision sous conditions de validité est documentée dans `docs/DECISION_GT205_VALIDATION.md`.
4. Obtenir de vraies unités d'acquisition / roulement pour un test indépendant.
5. **Geler les choix avant d'accéder au régime réservé** et intégrer dans le
   rapport collectif après validation de l'équipe.

**Ne pas fusionner automatiquement dans `main`.**
