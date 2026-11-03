# Contrat d'intégration — à remplir quand les travaux A et C sont disponibles

Chercher les marqueurs **`ADAPTER_PLUS_TARD`** dans tous les scripts.

## 1. Schéma des données — BLOQUANT POUR LE RÉEL
- [ ] Vérifier structure exacte dossiers et fichiers, et savoir si le membre A a déjà découpé les signaux.
- [ ] Vérifier noms/unités des deux colonnes ; aujourd'hui `acceleration_vertbleu` et `acceleration_vert` selon la description préliminaire.
- [ ] Déterminer fréquence d'échantillonnage, durée des acquisitions, unités et signification de 10/15/20.
- [ ] Vérifier synchronisation des deux capteurs et l'origine des acquisitions.
- [ ] Confirmer l'identifiant **d'acquisition physiquement indépendante** : un fichier Excel peut n'être qu'une fenêtre d'une acquisition plus longue.
- [ ] Vérifier labels SAFE / BALL / INNER_RACE / OUTER_RACE / COMBINED et éventuelles étiquettes de sévérité.
- [ ] Remplacer si besoin `read_signal_file()` dans `src/ae_benchmark/data.py` si l'équipe exporte NPZ, CSV, Parquet ou des features.
- [ ] Éviter de relire le dataset entier si le volume est grand : intégrer un `torch.utils.data.Dataset` lazy.

## 2. Fenêtrage et normalisation
- [ ] Fixer longueur et recouvrement en tenant compte de la fréquence d'échantillonnage et du phénomène physique.
- [ ] Confirmer si l'entrée doit être signal brut, enveloppe, spectre, FFT, STFT ou CWT.
- [ ] Vérifier l'utilisation des deux voies simultanément; ne pas présumer une orientation différente des capteurs.
- [ ] Interdire toute normalisation globale calculée sur validation ou test.

## 3. Prévention de la fuite de données
- [ ] `group_id` = unité indépendante, PAS un identifiant de fichier si celui-ci est extrait d'une même acquisition.
- [ ] S'assurer que `train`, `val` et `test` n'ont aucun `group_id` en commun.
- [ ] Prévoir un test de généralisation inter-régimes (ex. entraîner sur deux régimes, tester le troisième) — **pas implémenté dans la V0**.
- [ ] Si plusieurs roulements physiques sont identifiables, évaluer la séparation par roulement physique.
- [ ] Décrire composition de la classe SAFE et les conditions de calibration des seuils.

## 4. Méthodes avancées restant à intégrer
- [ ] IDEC avec optimisation conjointe clustering + reconstruction ; ne pas simplement renommer K-Means `IDEC`.
- [ ] Comparer véritablement une architecture MAE/masquage autosupervisé si le budget de calcul le permet.
- [ ] Pour MemoryAE : explorer régularisation de mémoire, seuil et capacité (prototype V0 seulement).
- [ ] GMM sur l'espace latent pour comparaison à K-Means.
- [ ] Répétitions multi-seeds, intervalles de confiance, et coût de calcul ; V0 = une seed.
- [ ] Évaluer chaque classe de défaut et régime, confusion / ARI / NMI / PR-AUC / taux de fausses alertes.
- [ ] Seuil d'anomalie validé par exigence industrielle, pas uniquement quantile 99 %.
- [ ] Enregistrer version du code, version des données, configuration et machine utilisés.

## 5. Procédure Git recommandée
1. **Ne pas pousser ces fichiers sur `main`** et ne pas écraser la branche `DOC/rapport`.
2. Après inspection des travaux de l'équipe, créer une branche personnelle depuis `main` à jour (`feature/benchmark-autoencodeurs`).
3. Ajouter uniquement les scripts pertinents et adapter l'arborescence existante; conserver la structure commune si elle est déjà définie.
4. Exécuter les tests, les expériences et une revue scientifique avant d'ouvrir une pull request.

## Statut de la V0
- Fonctionnel : adaptateur CSV/Excel, fenêtres, contrôle de groupes, split stratifié, standardisation, MLP-AE, CNN-AE, DAE, VAE, multiscale+attention, prototype MemoryAE, entraînement, clustering avec K-Means, anomalie MSE + seuil sur SAFE, PCA et IsolationForest, sauvegarde des résultats, tests.
- **Non validé sur les données réelles** ; protocole à finaliser avec l'équipe.
- Aucune prétention de résultat réel. Le dataset `--demo` est entièrement synthétique.

## Points supplémentaires introduits par V2

- [ ] `memae` : comparer top-k = 2/4/8 et coefficient entropique sur VALIDATION SAFE ; documenter différences avec hard-shrinkage MemAE 2019.
- [ ] `cmae` : échelles temporelles, régularisation mémoire, capacité et pertinence sans ordre tracking ; ne pas l'appeler reproduction IEEE 2025.
- [ ] `masked_swin` : longueur divisible par 32 (configuration standard), stratégie de masquage et patchs selon durée physique ; attention locale 1D simplifiée.
- [ ] `adae` : évaluer plusieurs bruits, ne pas effacer les impulsions de défaut ; recalibrer les ASU et connexions résiduelles.
- [ ] `aae` : prototype adversarial + GMM, mais pas EM-AAE ni transfert KL de 2023 ; ne pas présenter ces éléments comme développés.
- [ ] `--representation` : `raw`/`envelope`/`log_spectrum` évaluées séparément ; pas de spectre en Hz ou de bande physique sans `fs`.
- [ ] `angular_order_envelope` : n'utiliser que des courbes rpm physiques synchrones, et revoir le filtrage de Hilbert en amont.