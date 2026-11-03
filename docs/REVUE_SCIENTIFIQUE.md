# Notes scientifiques — benchmark des autoencodeurs pour vibrations de roulements

**Statut** : revue orientée implémentation, non revue systématique exhaustive. Première version : 2026-11-02. **Actualisation 2026-11-03 : plusieurs modèles ont maintenant été évalués sur la base réelle préparée par l'équipe (`bd_v1.pkl`) ; voir `docs/ETUDE_SCIENTIFIQUE_GT204_GT205.md`. Le protocole demeure exploratoire, et MemAE/ADAE ont désormais été évalués sur trois graines en validation inter-régimes ; voir `docs/DECISION_GT205_VALIDATION.md`.**

## Publications consultées et liens vers le code

1. **Guo et al., 2017** — *Improved Deep Embedded Clustering with Local Structure Preservation*, IJCAI. DOI : https://doi.org/10.24963/ijcai.2017/243. Propose IDEC : reconstruction + KL de distributions d'affectation à des centroïdes. **Application dans ce projet** : `deep_clustering.py`, avec encodeur 1D à la place du réseau dense original. **Non reproduction stricte** : hyperparamètres et architecture différents.
2. **Gong et al., 2019** — *Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection*, ICCV. Page en accès libre : https://openaccess.thecvf.com/content_ICCV_2019/html/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.html. **Application** : prototype `MemoryAE`, entraînement seulement sur `SAFE` et erreur de reconstruction. **Non reproduction stricte** : l'adressage sparse et la régularisation de l'article sont à approfondir.
3. **A Novel Unsupervised Clustering and Domain Adaptation Framework for Rotating Machinery Fault Diagnosis**, IEEE TII, 2023. DOI : https://doi.org/10.1109/TII.2022.3228395. Autoencodeur adversarial + mélange gaussien, puis adaptation de domaine. **Application** : comparaison GMM sur les embeddings, étude inter-régimes à réaliser. **Ne pas prétendre avoir implémenté EM-AAE ou l'adaptation de domaine.**
4. **Clustering-guided prototypical contrastive learning for bearing fault diagnosis under variable working conditions**, Measurement Science and Technology, 2025. DOI : https://doi.org/10.1088/1361-6501/adcadc. **Application future** : tester l'invariance inter-régimes et, si nécessaire, des pertes contrastives.
5. **A Bearing Fault Diagnosis Method Based on Dual-Stream Hybrid-Domain Adaptation**, Sensors, 2025. DOI : https://doi.org/10.3390/s25123686. Approche fusionnant des représentations temporelles et fréquentielles. **Application future** : comparaison des canaux et combinaisons temps/spectre, sans supposer que l'unité 10/15/20 représente une vitesse en tr/min.
6. **Rolling Bearing Fault Diagnosis across Operating Conditions Based on Unsupervised Domain Adaptation**, Lubricants, 2023. DOI : https://doi.org/10.3390/lubricants11090383. **Application** : protocole inter-régimes ; aucune adaptation de domaine implémentée à ce stade.

## Comparaisons implémentées en V1

- Clustering : PCA + KMeans, caractéristiques physiques approximatives + KMeans, MLP/CNN/DAE/VAE/multiscale/mémoire + KMeans et GMM, CNN + IDEC si option activée.
- Détection d'anomalies : IsolationForest et autoencodeurs entraînés **seulement sur `SAFE`**, seuil sur validation `SAFE`.
- Evaluation : ARI, NMI, silhouette (clustering), Average Precision et PR-AUC calculée par intégration trapézoïdale, F1, rappel, faux positifs (anomalies). Ces deux mesures sont distinctes.

## Limites à lever avant toute conclusion scientifique

- Aucune valeur de fréquence d'échantillonnage vérifiée ; ne pas interpréter les bandes spectrales en Hz et ne pas choisir des fenêtres sur base physique sans `fs`.
- Vérifier que les `group_id` représentent des **acquisitions indépendantes** et que les fenêtres d'un même enregistrement ne franchissent jamais les partitions.
- Vérifier si plusieurs fichiers enregistrent la même opération / le même roulement pour renforcer le groupement ; sinon, possibilité de résultats optimistes.
- `COMBINED` peut chevaucher plusieurs signatures ; interpréter les clusters avec prudence.
- Les métriques du mode `screening` servent à choisir les réglages (validation). Le mode `final` ne doit être exécuté qu'une fois le modèle et les hyperparamètres gelés ; **ne pas trier les modèles en fonction du test**.
- Etudier et rapporter la variabilité entre initialisations et conditions 10/15/20 ; les ablations capteurs et les tests inter-régimes restent à développer.
- La base réelle de l'équipe est désormais utilisée pour les premières expérimentations comparatives. **Aucun modèle n'est déclaré définitivement meilleur sans acquisition indépendante et validation du protocole.**