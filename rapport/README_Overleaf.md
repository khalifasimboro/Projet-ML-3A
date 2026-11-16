# Rapport évolutif — Projet Autoencodeurs (ECC)

## Importer dans Overleaf (version gratuite)

1. Télécharger `Projet_Autoencodeurs_Overleaf.zip`.
2. Dans Overleaf : **New Project** > **Upload Project** > choisir l'archive `.zip`.
3. Vérifier que **Main document** = `main.tex` et **Compiler** = `pdfLaTeX`.
4. Cliquer sur **Recompile**. Le document compile **même sans logo ni photo**.
5. Dans `config.tex`, remplacer les noms des quatre membres, l'encadrant, les dates et la version.

## Ajouter les logos et visuels

Dans le dossier `images/`, ajouter facultativement :
- `logo_ecc.png`
- `logo_partenaire.png`
- `banc_essai.jpg`

Les chemins et noms sont modifiables au début de `config.tex`. Les logos remplacent automatiquement les encarts de remplacement. La photo `banc_essai.jpg` n'est pas insérée automatiquement : choisir son emplacement scientifique une fois la photo identifiée et correctement légendée.

Pour insérer une photo dans un chapitre :

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=.8\textwidth]{images/banc_essai.jpg}
  \caption{Banc d'essai : description et provenance à compléter.}
  \label{fig:banc-essai}
\end{figure}
```

## Où rédiger ?

- `page-garde.tex` : présentation initiale.
- `config.tex` : paramètres, noms, esthétique et chemins des images.
- `chapitres/01_introduction.tex` : introduction et problématique.
- `chapitres/02_cadre_scientifique.tex` : littérature et fondements.
- `chapitres/03_donnees.tex` : documentation du banc et EDA.
- `chapitres/04_methodologie.tex` : protocole d'expériences.
- `chapitres/05_resultats.tex` : résultats effectifs seulement.
- `chapitres/06_application.tex` : déploiement / démonstration.
- `chapitres/07_discussion_conclusion.tex` : discussion et conclusion.
- `annexes/journal_suivi.tex` : versions et décisions.
- `references.bib` : ajouter les références réellement utilisées.

Pour citer une source, ajouter une entrée BibTeX dans `references.bib`, puis écrire `\cite{cle_reference}` dans le chapitre.

## Collaboration sur le forfait gratuit

D'après les plans officiels d'Overleaf, le forfait gratuit n'autorise qu'un seul collaborateur supplémentaire par projet. Les autres membres peuvent consulter le PDF via un lien de lecture et proposer leurs corrections dans des fichiers LaTeX, ou le groupe peut versionner ces fichiers dans un dépôt GitHub indépendant. La synchronisation Overleaf/GitHub intégrée n'est pas disponible sur le forfait gratuit.

## Précautions scientifiques

- Ne pas annoncer de résultats non calculés.
- Ne pas supposer que 10/15/20 sont des tr/min sans confirmation.
- Ne pas afficher des pics en Hz avant vérification de la fréquence d'échantillonnage.
- Répartir les essais indépendants **avant** segmentation et standardisation.
- Entraîner un détecteur non supervisé de reconstruction sur les données saines, avec un seuil choisi hors test.
- Ne pas assimiler diagnostic d'anomalies et prédiction de durée de vie restante (RUL).

## Compatibilité

Packages classiques de TeX Live, compilateur `pdfLaTeX`, bibliographie BibTeX ; pas de `minted`, pas de `shell-escape`, aucune ressource payante ni fichier de police externe requis.
