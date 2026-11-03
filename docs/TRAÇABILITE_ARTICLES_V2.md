# Traçabilité scientifique : articles → équations → code → limites (V2)

**Statut V2 initial : benchmark préparatoire. Actualisation au 2026-11-03 :** des expériences ont été exécutées sur `bd_v1.pkl` (données de l'équipe), notamment VAE/IDEC, MLP/CNN/DAE, PCA et Isolation Forest ; **aucun score n'est une validation industrielle indépendante**. MemAE et ADAE ont désormais été évalués sur trois graines sur `bd_v1.pkl` ; CMAE et Swin n'ont pas encore été évalués sur cette base. Voir `docs/ETUDE_SCIENTIFIQUE_GT204_GT205.md` et `run_literature_ablation.py`.
Les méthodes ci-dessous sont des **adaptations de recherche**, pas des reproductions
à l'identique. Les travaux de 2023–2026 doivent être comparés selon leurs protocoles
réels ; des « accuracy » publiées en classification ne sont PAS des ARI ou PR-AUC.

| Article et source | Éléments vérifiés | Intégration du projet | Ce qui n'est PAS reproduit |
|---|---|---|---|
| Gong et al. (ICCV 2019), MemAE, https://openaccess.thecvf.com/content_ICCV_2019/html/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.html | Mémoire de prototypes normaux, adressage clairsemé et perte d'entropie | `SparseMemAE`, `SparseMemory`: top-k d'attention, mémoire entraînable, MSE + entropie, entraînement SAFE | Le hard-shrinkage à seuil de l'article (ici top-k) et les hyperparamètres publiés |
| Lu et al. (IEEE TIM 2025), DOI https://doi.org/10.1109/TIM.2025.3546365 | Enveloppe en ordres par resampling angulaire + démodulation, CMAE mémorisant les motifs multiscales | `CascadedMemAE`: deux encodeurs et mémoires à échelles fines/grossières; `angular_order_envelope` si vitesse instantanée et fs MESURÉES | Architecture détaillée exacte de la publication ; filtrage d'enveloppe et stratégie d'ordres de l'article ; les régimes 10/15/20 ne suffisent pas |
| Luo & Liu (Machines 2025), DOI https://doi.org/10.3390/machines13090792 | Masquage de signaux, reconstruction autosupervisée, mécanismes d'attention locales/décalées | `MaskedSwinAE`: masquage aléatoire de patchs, perte prioritaire sur patchs cachés, blocs d'attention locale/décalée 1D | Swin Transformer hiérarchique exact, patch-merging, stratégie d'augmentation originale et résultats CWRU/Paderborn |
| Lu, Zhou & He (Electronics 2024), DOI https://doi.org/10.3390/electronics13122403 | ASU, connexions shortcut, dropout et décodeur de débruitage | `AdaptiveDenoisingAE`: shrinkage multiplicatif adaptatif, résidus de convolution internes, dropout, perte SmoothL1 sur signal bruité | ASU exact, schéma de convolution/transposée exact et protocoles de bruit de l'article |
| IEEE TII (2023), DOI https://doi.org/10.1109/TII.2022.3228395 | Clustering EM-AAE + GMM puis synchronisation de domaines via KL symétrique | `AdversarialAE`: discriminateur sur représentation latente gaussienne ; GMM externe déjà dans le pipeline | Apprentissage EM-AAE conjoint ; adaptation sans réentraînement et appariement sym-KL des clusters |
| Wen et al. (Expert Systems with Applications 2023), DOI https://doi.org/10.1016/j.eswa.2023.120066 | c-GCN-MAL : graph convolution et apprentissage adversarial multireprésentation | Ablation `raw/envelope/log_spectrum` et AAE indépendant ; les entrées ne sont pas fusionnées | Le réseau c-GCN, l'apprentissage multireprésentation conjoint, l'adaptation adversariale inter-domaines |
| EWT + CVAE + CNN (2026), DOI https://doi.org/10.1016/j.fraope.2026.100553 | Transformée en ondelettes empirique EWT, enveloppe spectrale, VAE + CNN supervisé | VAE convolutionnel + ablation `envelope/log_spectrum`, sans créer de fréquences physiques | EWT, choix de bandes, pipeline de classification supervisée CNN |
| Guo et al. (IJCAI 2017), DOI https://doi.org/10.24963/ijcai.2017/243 | IDEC : KL des affectations de Student + reconstruction de structure locale | `deep_clustering.py`: optimisation KL+MSE après pré-entraînement, centres initialisés K-Means | Fréquence de mise à jour/arrêt et protocole/hyperparamètres exacts |

## Hypothèses scientifiques à vérifier

1. **Mémoire** : `cnn` vs `memory` (ancienne mémoire dense) vs `memae` (top-k + entropie) vs `cmae` (deux échelles) sur TRAIN SAFE. Mesurer AUPRC, taux de fausses alarmes, rappel par défaut et par régime.
2. **Masquage** : `cnn` vs `masked_swin`, mêmes données et budget d'apprentissage, ARI/NMI pour clustering puis anomalies.
3. **Débruitage** : `denoise` vs `adae` avec niveau de bruit contrôlé, puis bruit non vu (niveau et type) ; ne pas présenter un protocole de bruit synthétique comme réalité du banc.
4. **Clustering** : `cnn`+K-Means, `cnn`+IDEC, `aae`+GMM et `vae`+GMM, ARI/NMI et robustesse par régime. Ne PAS baptiser AAE+GMM « EM-AAE ».
5. **Représentation** : même modèle et même protocole sur `raw`, `envelope` et `log_spectrum`; comparer également les deux capteurs.
6. **Domaines** : évaluation leave-one-regime-out uniquement après définition rigoureuse des acquisitions. Un split aléatoire à l'intérieur d'un régime NE constitue PAS une validation inter-régimes.

## Préconditions physiques et méthodologiques

- `rpm_trace` et `fs` doivent être synchronisés, exacts et physiquement documentés pour les ordres. **Ne jamais transformer arbitrairement une étiquette de régime en trace rpm.**
- Un signal Hilbert sans sélection préalable de bande ne remplace pas une démodulation adaptée des vibrations de roulement.
- Les `group_id` doivent désigner les acquisitions physiquement indépendantes, pas seulement les fichiers Excel.
- Le score d'anomalie actuellement utilisé est la MSE de reconstruction. Tester d'autres scores (p. ex. erreur sur l'enveloppe) **sur validation** avant de consulter le test.
- Des **répétitions sur cinq graines** ont désormais été exécutées pour VAE/IDEC et un contrôle de généralisation 10 → 20 est implémenté. Aucun protocole leave-one-regime-out complet, optimisation métier du seuil ou test sur nouvelles acquisitions n'est encore validé.
- Un modèle entraîné avec `--demo` prouve uniquement que le code fonctionne. **Aucun score de démonstration n'est publiable.**

## Actualisation après l'ablation réelle du 03/11/2026

MemAE/ADAE ont été évalués sur trois graines face à CNN-AE et Denoising AE.
Ces modèles **ne démontrent pas de gain sur le taux de fausses alarmes
sans perturbation** par rapport à CNN-AE. MemAE et ADAE ont cependant
montré une stabilité de ce taux sous une perturbation numérique des
log-spectres standardisées ; elle ne simule pas le bruit physique du capteur.
Détails et chiffres dans `docs/DECISION_GT205_VALIDATION.md`.
