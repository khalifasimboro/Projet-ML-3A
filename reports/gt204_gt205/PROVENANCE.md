# Jeu minimal de résultats expérimentaux — GT2.04 / GT2.05

Ces CSV ont été **transcrits depuis les sorties PowerShell communiquées par l'utilisateur**
lors de la campagne du 03/11/2026 ; ils ne sont pas des données inventées ni une
reproduction automatique d'un pipeline complet à partir des checkpoints.

- `clustering_cinq_graines.csv` : VAE/IDEC sur les graines 0, 1, 7, 21, 42 ;
  modèle VAE entraîné seul, puis IDEC ; régime d'entraînement 10,
  validation 20, régime 15 réservé. 15 époques AE, 5 époques IDEC.
- `clustering_comparaison_initiale.csv` : une exécution multi-modèles,
  15 époques ; **antérieure à la correction de l'initialisation par modèle**.
  La valeur VAE n'est donc pas strictement comparable aux runs VAE isolés
  à graine identique. Elle documente uniquement l'exploration initiale.
- `detection_validation_corrigee.csv` : détection à 15 époques, graine 42,
  validation SAFE divisée entre early-stopping/calibration/évaluation ;
  chaque entrée a été recopiée à l'arrondi affiché par PowerShell.
  Précision et F1 sont des valeurs **arrondies**. Les scores AUROC/PR_AUC
  sont ceux reportés par le programme.
- Tous ces résultats utilisent les **spectres log-PSD** de `bd_v1.pkl`.
- **Limite :** indépendance physique des acquisitions non certifiée ;
  résultats de validation exploratoires, pas des tests finaux.

Les CSV du dossier local `results/gt204_gt205/`, générés directement par
les scripts, constituent les sorties machine plus complètes. Les versions
présentes ici sont des **archives de lecture** pour rendre le notebook
interprétable sur GitHub même sans disposer des fichiers locaux de résultats.

Si les versions divergent, **enquêter sur les écarts de protocole, de version
du code et de reproductibilité**, et ne pas remplacer silencieusement
l'une par l'autre.

## Campagne complémentaire issue des articles (03/11/2026)

- `ablation_articles_moyennes_console.csv` : transcription exacte (à six
  décimales) des moyennes et écarts-types affichés par PowerShell
  après `python run_literature_ablation.py --models cnn,denoise,memae,adae
  --seeds 0,7,21 --epochs 15 --noise-levels 0,0.15,0.30
  --out results/gt204_gt205/ablation_articles`.
- Les valeurs sont des **statistiques arrondies de 3 graines** ; ni les
  scores par observation ni les résultats individuels ne sont contenus
  dans ce fichier versionné.
- Les détails individuels et scores par classe sont générés **localement**
  dans `results/gt204_gt205/ablation_articles/`. Leur transfert volontaire
  dans `ablation_articles_par_graine.csv` et
  `ablation_articles_par_classe.csv` permettra à l'équipe de consulter
  les résultats par graine et par défaut sans réentraîner.
  Si ces fichiers n'existent pas dans le dépôt, les seuls chiffres
  versionnés sont les moyennes arrondies rapportées dans PowerShell.
- Les mêmes données de validation sont réutilisées entre graines,
  avec sous-partitions SAFE différentes ; **les répétitions ne sont pas
  des acquisitions indépendantes**.
