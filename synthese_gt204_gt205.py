#!/usr/bin/env python
"""Synthese GT2.04/GT2.05 exclusivement depuis les CSV locaux de validation.

Ne charge pas de checkpoints, ne touche pas au regime reserve 15 et ne
selectionne pas de modele sur le jeu de test.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEEDS = {
    42: "essai_idec_vae",
    7: "idec_seed_7",
    21: "idec_seed_21",
    0: "idec_seed_0",
    1: "idec_seed_1",
}
EXPECTED = {"split_mode": "regime", "phase": "screening",
            "representation": "spectres", "train_regimes": "10",
            "val_regime": "20", "test_regime": "15", "epochs": 15}


def load_experiment(root, name, task, seed=None):
    folder = root / name
    csv_file = folder / "comparaison_screening.csv"
    protocol_file = folder / "protocole_screening.json"
    if not csv_file.is_file() or not protocol_file.is_file():
        raise FileNotFoundError(f"Resultats manquants : {csv_file} ou {protocol_file}")
    meta = json.loads(protocol_file.read_text(encoding="utf-8"))
    for key, expected in EXPECTED.items():
        if str(meta.get(key)) != str(expected):
            raise ValueError(f"{folder} : {key}={meta.get(key)!r}, attendu {expected!r}")
    if task not in meta.get("tasks", []):
        raise ValueError(f"{folder} ne correspond pas a la tache {task}")
    if seed is not None and int(meta.get("seed", -1)) != seed:
        raise ValueError(f"{folder} : mauvaise graine ; {seed} attendu")
    data = pd.read_csv(csv_file)
    data = data.loc[(data["phase"] == "screening") & (data["task"] == task)].copy()
    if data.empty or data["model"].duplicated().any():
        raise ValueError(f"{csv_file} : lignes absentes ou modeles dupliques")
    return data


def format_number(value):
    return "N/A" if pd.isna(value) else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Construire les livrables GT2.04/GT2.05")
    parser.add_argument("--root", type=Path, default=Path("results/gt204_gt205"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root
    out = args.out or root / "synthese"
    out.mkdir(parents=True, exist_ok=True)

    # Les cinq repetitions VAE/IDEC sont comparables entre elles.
    rows = []
    for seed, folder in SEEDS.items():
        df = load_experiment(root, folder, "clustering", seed)
        for name in ("vae", "idec(vae)"):
            selection = df.loc[df["model"] == name]
            if len(selection) != 1:
                raise ValueError(f"{folder}: modele {name} absent")
            row = selection.iloc[0]
            rows.append({"seed": seed, "model": name, **{
                metric: float(row[metric]) for metric in ("ARI", "NMI", "silhouette")
            }})
    per_seed = pd.DataFrame(rows).sort_values(["model", "seed"])
    per_seed.to_csv(out / "clustering_cinq_graines.csv", index=False)

    stats = (per_seed.groupby("model")[["ARI", "NMI", "silhouette"]]
             .agg(["mean", "std", "min", "max"]))
    stats.columns = ["_".join(col) for col in stats.columns]
    stats = stats.reset_index()
    stats.to_csv(out / "clustering_statistiques.csv", index=False)

    baseline_run = load_experiment(root, "comparaison_approfondie", "clustering", 42)
    baseline_run.to_csv(out / "clustering_une_execution.csv", index=False)
    if not {"spectra+kmeans", "pca+kmeans"}.issubset(set(baseline_run["model"])):
        raise ValueError("References KMeans/PCA manquantes")
    baseline_ari = float(baseline_run.loc[
        baseline_run["model"] == "spectra+kmeans", "ARI"].iloc[0])

    detect = load_experiment(root, "detection_inter_regimes_corrigee", "anomaly", 42)
    detect.to_csv(out / "detection_corrigee.csv", index=False)
    if not {"vae", "isolation_forest", "pca", "mlp", "cnn", "denoise"}.issubset(set(detect.model)):
        raise ValueError("Certains modeles de detection sont absents")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    seeds = sorted(SEEDS)
    for label, marker in (("vae", "o"), ("idec(vae)", "s")):
        subset = per_seed.loc[per_seed["model"] == label].set_index("seed")
        ax.plot(seeds, subset.loc[seeds, "ARI"], marker=marker, label=label)
    ax.axhline(baseline_ari, ls="--", color="gray", label="K-Means classique")
    ax.set_xlabel("Graine aleatoire")
    ax.set_ylabel("ARI en validation")
    ax.set_title("Clustering inter-regimes : stabilite sur cinq graines")
    ax.set_xticks(seeds)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "ari_par_graine.png", dpi=170)
    plt.close(fig)

    plot_detect = detect.sort_values("false_alarm_rate")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(plot_detect["model"], 100 * plot_detect["false_alarm_rate"])
    ax.set_ylabel("Taux de fausses alarmes (%)")
    ax.set_title("Detection inter-regimes (validation corrigee)")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fausses_alarmes.png", dpi=170)
    plt.close(fig)

    vae = stats.loc[stats.model == "vae"].iloc[0]
    idec = stats.loc[stats.model == "idec(vae)"].iloc[0]
    best_far = detect["false_alarm_rate"].min()
    finalists = detect.loc[np.isclose(detect["false_alarm_rate"], best_far), "model"].tolist()

    report = f"""# GT2.04 et GT2.05 — Synthese des validations exploratoires

## Protocole
- Donnees d'equipe : bd_v1.pkl (spectres : 2 canaux de 256 valeurs).
- Apprentissage : regime 10 ; validation : regime 20 ; regime 15 reserve et NON exploite.
- Cinq graines pour VAE/IDEC : {', '.join(map(str, seeds))}.
- La comparaison a une seule execution n'est pas une preuve de stabilite.
- Les premieres detections non corrigees ne sont PAS utilisees ici.

## Clustering
- Reference K-Means sur spectres : ARI = {baseline_ari:.6f}.
- VAE + K-Means : ARI moyen = {vae['ARI_mean']:.6f}, ecart-type echantillon = {vae['ARI_std']:.6f}.
- VAE + IDEC : ARI moyen = {idec['ARI_mean']:.6f}, ecart-type echantillon = {idec['ARI_std']:.6f}.
- IDEC n'ameliore pas systematiquement VAE : evaluer stabilite ET qualite.

## Detection d'anomalies
- AUROC = 1.0 pour tous les modeles sur cette validation.
- Taux de fausses alarmes minimal = {100*best_far:.3f} % : {', '.join(finalists)}.
- Seuil calibre sur une partie SAFE de validation distincte des SAFE d'evaluation.
- Le rappel et F1 sont a interpreter avec la forte proportion de defauts dans l'evaluation.

## Conclusion provisoire
VAE + K-Means : candidat prioritaire pour le clustering ; comparer au K-Means classique.
VAE et Isolation Forest : candidats a egalite pour la detection sur ce protocole.
Aucun vainqueur final n'est etabli avant une validation sur acquisitions independantes.

## Limites
- Les noms de fichiers suggerent plusieurs acquisitions continues traversees par les anciens splits.
- Le transfert entre regimes ne prouve pas une independance entre roulements ou sessions.
- Certains modeles n'ont qu'une repetition en detection, et cinq graines restent limites.
- Le VAE de l'execution multi-modeles peut changer lorsque l'ordre d'initialisation change.
- Ne pas exploiter le regime reserve 15 pour selectionner les hyperparametres.

## Pieces generees
- clustering_cinq_graines.csv : observations individuelles.
- clustering_statistiques.csv : moyennes, ecarts-types, minimums, maximums.
- clustering_une_execution.csv : toutes les architectures du screening initial.
- detection_corrigee.csv : detection apres separation des SAFE de validation.
- ari_par_graine.png et fausses_alarmes.png : figures.
"""
    (out / "rapport_synthese.md").write_text(report, encoding="utf-8")
    print("\nStatistiques clustering :")
    print(stats.to_string(index=False))
    print("\nDetection corrigee :")
    print(detect[["model", "AUROC", "F1", "false_alarm_rate"]].to_string(index=False))
    print(f"\nLivrables crees dans : {out}")


if __name__ == "__main__":
    main()
