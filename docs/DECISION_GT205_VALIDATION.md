# Décision GT2.05 — sélection argumentée des modèles (validation exploratoire)

**Version : 2026-11-03.** Les résultats ci-dessous proviennent des sorties
PowerShell communiquées par l'utilisateur, et des tableaux archivés dans
`reports/gt204_gt205/`. L'expérience de littérature **a été exécutée sur
`bd_v1.pkl`** : 15 époques, graines 0/7/21, modèles CNN, Denoising,
MemAE, ADAE, plus Isolation Forest, niveaux de perturbation 0/0,15/0,30.

## 1. Périmètre scientifique

Tâches GT2.04 (comparer les modèles) et GT2.05 (évaluer / sélectionner).
Entrées : **log-PSD préparées de deux capteurs**, shape `(N,2,256)` ;
aucun signal brut n'est traité dans cette campagne.

**Train = régime 10**, **validation = régime 20**, **régime 15 réservé**.
Pour la détection, apprendre les modèles sur SAFE du régime 10 et
utiliser trois sous-ensembles SAFE disjoints de validation pour
sélectionner l'époque, calibrer le seuil et mesurer les fausses alarmes.
La calibration utilise le quantile 99 % des scores SAFE dédiés.

Les parties SAFE diffèrent d'une graine à l'autre mais reposent sur
le même régime / la même base ; **trois initialisations ne sont pas
trois expériences sur des acquisitions indépendantes**.

## 2. Clustering (cinq graines, validation du régime 20)

| Modèle | ARI moyen | Écart-type échantillon | NMI moyen |
|---|---:|---:|---:|
| VAE + K-Means | **0,711892** | 0,092028 | **0,837899** |
| VAE + IDEC | 0,654297 | 0,058573 | 0,803796 |
| K-Means direct sur log-spectres | 0,609269 | non mesuré sur plusieurs graines | 0,782830 |

**Présélection clustering : VAE + K-Means.**
Ce choix privilégie ARI et NMI, mais une variance de l'ARI
de 0,0920 est à discuter. IDEC n'apporte pas de gain systématique.
La référence K-Means doit rester dans les livrables.

## 3. Détection d'anomalies (3 graines, 15 époques)

### Spectres non perturbés

| Modèle | AUROC (moy. ± ET) | F1 (moy. ± ET) | Fausses alarmes, moy. ± ET |
|---|---|---|---|
| CNN-AE | 1,0000 ± 0,0000 | **0,999768 ± 0,000232** | **0,5714 % ± 0,5714 pp** |
| MemAE (adapté de Gong 2019) | 1,0000 ± 0,0000 | 0,999690 ± 0,000268 | 0,7619 % ± 0,6598 pp |
| ADAE (adapté de Lu 2024) | 1,0000 ± 0,0000 | 0,999613 ± 0,000483 | 0,9524 % ± 1,1895 pp |
| Denoising AE | 1,0000 ± 0,0000 | 0,999458 ± 0,000483 | 1,3333 % ± 1,1895 pp |
| Isolation Forest | 0,999363 ± 0,001087 | 0,998992 ± 0,000538 | 1,3333 % ± 0,6598 pp |

Tous les autoencodeurs cités ici ont rappel moyen de 1,0000 sur le régime
de validation. Isolation Forest atteint un rappel moyen de 0,999070.

**Différences absolues minuscules :** les valeurs arrondies correspondent
à environ 3, 4, 5 et 7 fausses alarmes respectivement pour CNN-AE,
MemAE, ADAE et Denoising, sur 3 × 175 évaluations SAFE. Les 525
*évaluations* ne sont **pas 525 acquisitions indépendantes** : des
observations peuvent se répéter d'une graine à l'autre.

**VAE** avait obtenu une AUROC de 1,0, F1 de 0,999303 et 3/175
fausses alarmes dans une **autre expérience à une seule graine**. Ce
résultat ne permet pas de classer VAE au-dessous ou au-dessus de CNN
par rapport à trois répétitions. Les méthodes MLP/PCA n'ont pas été
répétées selon le même protocole que le nouveau tableau.

### Perturbation spectrale synthétique (σ = 0,30)

| Modèle | Taux moyen de fausses alarmes |
|---|---:|
| MemAE | 0,7619 % |
| CNN-AE | 0,9524 % |
| ADAE | 0,9524 % |
| Denoising AE | 1,5238 % |
| Isolation Forest | 0,7619 % (rappel moyen 0,998605) |

Cette perturbation est additive sur le **log-spectre standardisé**,
pas sur le signal vibratoire de capteur. Le seuil est calibré sur des
SAFE non perturbés et reste fixe. Les tendances ne constituent donc
**pas une démonstration de robustesse à du bruit industriel réaliste**.

### Discussion à partir des articles

- **MemAE (Gong et al., ICCV 2019)** : notre mémoire top-k et sa
  régularisation se comportent convenablement mais **ne prouvent pas
  un avantage significatif** sur CNN-AE dans ce protocole.
- **ADAE (Lu, Zhou & He, Electronics 2024)** : le détecteur est stable
  face aux perturbations standardisées considérées ; le gain éventuel
  ne peut pas être attribué isolément aux unités adaptatives,
  car architecture, perte et régularisation diffèrent.
- **IDEC (Guo et al., IJCAI 2017)** : implémentation et répétitions
  réelles effectuées ; pas d'amélioration systématique du VAE.

## 4. Présélection GT2.05 et règles de décision

- **Clustering principal : VAE + K-Means**, avec IDEC et K-Means direct
  comme références.
- **Détection principale pour poursuivre l'évaluation : CNN-AE**,
  meilleur taux moyen de fausses alarmes parmi les modèles évalués sur
  les **trois mêmes graines**. Les différences sont faibles et ne
  justifient pas de revendiquer une supériorité statistique certaine.
- **Alternative issue de la littérature : MemAE**, dont le taux de
  fausses alarmes reste inchangé sous les perturbations testées.
- **Référence de détection : Isolation Forest**, plus simple et
  intéressant à conserver pour mesurer l'intérêt réel des AE.
- **VAE détection reste un candidat non éliminé** : son résultat sur
  une seule graine n'est pas directement comparable à cette ablation.

**Il s'agit d'une présélection de validation, pas d'un modèle industriel
final ni d'une fusion validée par l'équipe.**

## 5. Limites et conditions de livraison

1. Origine physique des fenêtres : des séries présumées traversent
   les splits historiques ; des régimes disjoints ne garantissent pas
   l'indépendance des acquisitions.
2. Scores AUROC proches de 1 sur ce jeu : contrôler les
   distributions des scores, le ratio défauts/SAFE et la possibilité de
   corrélations liées au banc ; absence de preuve de validité hors
   de ces conditions.
3. Trois graines permettent d'estimer une variabilité d'entraînement
   mais **pas** un intervalle de confiance sur de nouvelles machines.
4. Les métriques et comparaisons sont exploratoires, et les valeurs
   issues du tableau PowerShell ont été archivées à six décimales :
   les CSV locaux originaux sont plus complets, avec résultats individuels
   et scores par classe.
5. Avant le test final : confirmer les acquisitions indépendantes,
   définir les groupes réels, figer le choix, puis tester **une seule fois**
   sans ré-optimisation sur le test.

## 6. Livrables transférables dans une branche collaborative

- `notebook/experimentation_membre_B.ipynb` : récit et code reproductibles ;
- `src/ae_benchmark/` : architectures, préparation, fonctions d'évaluation ;
- `run_benchmark_bd_v1.py`, `run_literature_ablation.py`,
  `synthese_gt204_gt205.py` : benchmarks et synthèse ;
- `reports/gt204_gt205/` : fichiers CSV versionnés avec provenance ;
- `docs/ETUDE_SCIENTIFIQUE_GT204_GT205.md`,
  `docs/DECISION_GT205_VALIDATION.md` : raisonnement scientifique ;
- `tests/` et dépendances nécessaires.

**Ne pas remplacer le notebook collectif** et **ne pas fusionner
directement dans `main`**.
