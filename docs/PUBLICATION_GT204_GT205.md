# Publication GT2.04–GT2.05 sur une branche collaborative

**Objectif :** une branche dédiée feature/gt204-gt205-membre-b à partir de origin/main, sans aucune fusion ni modification de main.

**Attention :** le dépôt privé de l'équipe n'est pas accessible depuis la connexion GitHub de cette session. Les commandes devront être exécutées localement par une personne autorisée sur le dépôt collaboratif.

## Vérifications AVANT toute écriture

Exécuter sous PowerShell depuis D:\document\3A_ML\Projet-ML-3A :

```powershell
git branch --show-current
git status --short
git pull --ff-only
python -m pytest -q
git remote -v
git fetch origin main
git ls-tree -r --name-only origin/main -- src/ae_benchmark tests notebook/experimentation_membre_B.ipynb reports/gt204_gt205
git ls-tree -r --name-only origin/main -- run_benchmark.py run_benchmark_bd_v1.py run_literature_ablation.py synthese_gt204_gt205.py select_models.py requirements_gt204_gt205.txt
git ls-tree -r --name-only origin/main -- docs/ADAPTATION.md docs/DECISION_GT205_VALIDATION.md docs/ETUDE_SCIENTIFIQUE_GT204_GT205.md docs/INTEGRATION_GT204_GT205.md docs/REVUE_SCIENTIFIQUE.md docs/TRAÇABILITE_ARTICLES_V2.md docs/PUBLICATION_GT204_GT205.md
```

La branche courante attendue est integration/autoencodeurs ; le statut Git doit être propre. Vérifier que origin désigne bien khalifasimboro/Projet-ML-3A. Si la commande ls-tree liste des fichiers sur les chemins à ajouter, arrêter le transfert pour vérifier les collisions : la contribution ne doit jamais écraser le travail de l'équipe.

Ouvrir notebook/experimentation_membre_B.ipynb dans VS Code, sélectionner .venv puis Run All. Le notebook est conçu pour lire les résultats archivés sans relancer des entraînements.

## Transfert lorsque les contrôles sont satisfaisants

```powershell
git switch -c feature/gt204-gt205-membre-b origin/main

git restore --source integration/autoencodeurs --staged --worktree -- `
  src/ae_benchmark `
  run_benchmark.py `
  run_benchmark_bd_v1.py `
  run_literature_ablation.py `
  synthese_gt204_gt205.py `
  select_models.py `
  requirements_gt204_gt205.txt `
  notebook/experimentation_membre_B.ipynb `
  reports/gt204_gt205 `
  docs/ADAPTATION.md `
  docs/DECISION_GT205_VALIDATION.md `
  docs/ETUDE_SCIENTIFIQUE_GT204_GT205.md `
  docs/INTEGRATION_GT204_GT205.md `
  docs/REVUE_SCIENTIFIQUE.md `
  docs/TRAÇABILITE_ARTICLES_V2.md `
  docs/PUBLICATION_GT204_GT205.md `
  tests

git diff --cached --check
git diff --cached --stat
python -m pytest -q
git status --short
```

Si l'un des fichiers répertoriés existe déjà sur la branche collaborative principale, examiner d'abord ses changements ; ne pas exécuter la commande de transfert ci-dessus sans réconciliation.

Lorsque le diff, le notebook et les tests sont validés :

```powershell
git commit -m 'GT2.04-GT2.05 : benchmarking, rapport et présélection'
git push -u origin feature/gt204-gt205-membre-b
```

Cette opération publie une nouvelle branche seulement ; elle ne fusionne pas dans main. Aucune Pull Request n'est obligatoire à ce stade.

## Ne pas ajouter

- bd_v1.pkl : la copie collective est à conserver.
- notebook/autoencodeurs_roulements.ipynb : ne jamais remplacer le notebook d'origine.
- results/ : sorties locales lourdes, checkpoints et résultats détaillés à conserver à part.
- .venv, caches et environnements locaux.

Les CSV légers de reports/gt204_gt205 sont des transcriptions fidèles des résultats PowerShell fournis et portent des avertissements de provenance ; les CSV machine originaux restent dans results/gt204_gt205.
