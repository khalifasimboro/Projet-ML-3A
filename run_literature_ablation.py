#!/usr/bin/env python
"""Ablation de la littérature sur bd_v1.pkl : détection d'anomalies.

Comparaison CNN/DAE (références) avec des adaptations de MemAE (Gong 2019)
et ADAE (Lu et al. 2024), éventuellement CMAE (Lu et al. 2025).

Entraînement SAFE du régime 10 ; validation sur régime 20 uniquement.
Le régime 15 reste entièrement réservé. La perturbation gaussienne est appliquée
aux SPECTRES STANDARDISÉS et n'est pas un bruit physique de capteur.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.ae_benchmark.collaborative_data import get_split_indices, load_collaborative_database
from src.ae_benchmark.evaluation import anomaly_metrics
from src.ae_benchmark.models import build_model
from src.ae_benchmark.training import reconstruction_scores, set_seed, train_model

SUPPORTED_MODELS = ("cnn", "denoise", "memae", "adae", "cmae")
LITERATURE = {
    "memae": "Gong et al., ICCV 2019: https://openaccess.thecvf.com/content_ICCV_2019/html/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.html",
    "adae": "Lu, Zhou & He, Electronics 2024: https://doi.org/10.3390/electronics13122403",
    "cmae": "Lu et al., IEEE TIM 2025: https://doi.org/10.1109/TIM.2025.3546365",
}


def split_safe_validation(val_indices, labels, seed):
    """SAFE disjoints : choix d'époque, calibration, évaluation ; défauts en évaluation."""
    val_indices = np.asarray(val_indices, dtype=int)
    labels = np.asarray(labels)
    safe = val_indices[labels[val_indices] == "SAFE"]
    if len(safe) < 9:
        raise ValueError("Il faut au moins 9 observations SAFE de validation")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(safe)
    a = len(safe) // 3
    early, calibration = np.sort(shuffled[:a]), np.sort(shuffled[a:2*a])
    evaluation = np.setdiff1d(val_indices, np.r_[early, calibration])
    if not (set(early).isdisjoint(calibration) and
            set(early).isdisjoint(evaluation) and
            set(calibration).isdisjoint(evaluation)):
        raise AssertionError("Fuite entre ensembles de validation SAFE")
    if not (np.any(labels[evaluation] == "SAFE") and
            np.any(labels[evaluation] != "SAFE")):
        raise ValueError("Validation sans SAFE ou sans défaut")
    return early, calibration, evaluation


def perturb_normalized_spectra(x, level, seed):
    """Bruit synthétique sur les log-spectres normalisés (PAS bruit de capteur)."""
    if level < 0:
        raise ValueError("Le niveau de bruit doit être positif ou nul")
    if level == 0:
        return x.copy()
    rng = np.random.default_rng(seed)
    return (x + level * rng.standard_normal(x.shape).astype(np.float32)).astype(np.float32)


def evaluate_scores(score_calibration, score_evaluation, labels, threshold_quantile=0.99):
    scores = np.asarray(score_evaluation)
    truth = np.asarray(labels)
    if len(scores) != len(truth):
        raise ValueError("scores et étiquettes incompatibles")
    metrics = anomaly_metrics(score_calibration, scores, truth, threshold_quantile)
    metrics["AUROC"] = float(roc_auc_score(truth != "SAFE", scores))
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="GT2.04 : ablation MemAE/ADAE/CMAE sur spectres")
    p.add_argument("--data", default="bd_v1.pkl")
    p.add_argument("--models", default="cnn,denoise,memae,adae")
    p.add_argument("--seeds", default="0,7,21", help="Graines séparées par des virgules")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=6)
    p.add_argument("--noise-levels", default="0,0.15,0.30",
                   help="Ecart-type additif sur caractéristiques spectrales STANDARDISÉES")
    p.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu")
    p.add_argument("--out", default="results/gt204_gt205/ablation_articles")
    p.add_argument("--save-checkpoints", action="store_true")
    return p.parse_args()


def validate_args(args):
    names = [v.strip() for v in args.models.split(",") if v.strip()]
    if not names or len(names) != len(set(names)) or not set(names).issubset(SUPPORTED_MODELS):
        raise ValueError(f"--models : utiliser une liste unique parmi {SUPPORTED_MODELS}")
    seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    if not seeds or len(seeds) != len(set(seeds)) or any(s < 0 for s in seeds):
        raise ValueError("--seeds : fournir des entiers positifs distincts")
    noise_levels = [float(v.strip()) for v in args.noise_levels.split(",") if v.strip()]
    if not noise_levels or len(noise_levels) != len(set(noise_levels)) or min(noise_levels) < 0 or 0.0 not in noise_levels:
        raise ValueError("--noise-levels doit contenir 0 et des nombres positifs distincts")
    if min(args.epochs, args.batch_size, args.latent_dim) < 1:
        raise ValueError("epochs, batch-size et latent-dim doivent être positifs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA demandé mais indisponible")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    return names, seeds, sorted(noise_levels), device


def run(args):
    names, seeds, levels, device = validate_args(args)
    ds = load_collaborative_database(args.data, representation="spectres")
    train, val, test = get_split_indices(
        ds, mode="regime", train_regimes="10", val_regime="20", test_regime="15"
    )
    assert set(ds.regimes[train]) == {"10"} and set(ds.regimes[val]) == {"20"}
    assert set(ds.regimes[test]) == {"15"}
    train_safe = train[ds.labels[train] == "SAFE"]
    scaler = StandardScaler().fit(ds.x[train_safe].reshape(len(train_safe), -1))
    x = scaler.transform(ds.x.reshape(len(ds.x), -1)).reshape(ds.x.shape).astype(np.float32)
    x_train = x[train_safe]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records, by_class, histories = [], [], []
    for seed in seeds:
        early, cal, ev = split_safe_validation(val, ds.labels, seed)
        x_early, x_cal, x_eval = x[early], x[cal], x[ev]
        y_eval = ds.labels[ev]
        if not (len(set(early) & set(cal)) == 0 and len(set(cal) & set(ev)) == 0):
            raise AssertionError("Fuite SAFE")
        # Une perturbation donnée est IDENTIQUE pour tous les modèles d'une graine.
        evaluation_by_level = {
            level: perturb_normalized_spectra(x_eval, level, seed=seed * 1009 + 77)
            for level in levels
        }

        for name in [*names, "isolation_forest"]:
            start = time.perf_counter()
            if name == "isolation_forest":
                detector = IsolationForest(n_estimators=300, random_state=seed)
                detector.fit(x_train.reshape(len(x_train), -1))
                score_cal = -detector.score_samples(x_cal.reshape(len(x_cal), -1))
                score_eval_fn = lambda array: -detector.score_samples(array.reshape(len(array), -1))
                n_params = 0
            else:
                # IMPORTANT : seed posée AVANT construction, indépendamment de
                # l'ordre des autres architectures.
                set_seed(seed)
                model = build_model(name, length=x.shape[-1],
                                    channels=x.shape[1], latent_dim=args.latent_dim)
                n_params = sum(p.numel() for p in model.parameters())
                history = train_model(
                    model, x_train, x_early, epochs=args.epochs,
                    batch_size=args.batch_size, seed=seed, device=device
                )
                histories += [{"model": name, "seed": seed, **entry} for entry in history]
                score_cal = reconstruction_scores(model, x_cal, device=device)
                score_eval_fn = lambda array: reconstruction_scores(model, array, device=device)
                if args.save_checkpoints:
                    torch.save({
                        "model": name, "seed": seed, "channels": x.shape[1],
                        "length": x.shape[2], "latent_dim": args.latent_dim,
                        "state_dict": model.state_dict(),
                    }, out / f"{name}_seed_{seed}.pt")
            duration = time.perf_counter() - start
            for level in levels:
                scores = score_eval_fn(evaluation_by_level[level])
                metrics = evaluate_scores(score_cal, scores, y_eval)
                records.append({
                    "seed": seed, "model": name, "noise_std": level,
                    "n_train_safe": len(train_safe), "n_early_safe": len(early),
                    "n_cal_safe": len(cal), "n_eval_safe": int(np.sum(y_eval == "SAFE")),
                    "n_eval_fault": int(np.sum(y_eval != "SAFE")),
                    "parameters": n_params, "train_seconds": round(duration, 3),
                    **metrics,
                })
                for label in sorted(set(y_eval)):
                    mask = y_eval == label
                    predicted_anomaly = scores[mask] > metrics["threshold"]
                    by_class.append({
                        "seed": seed, "model": name, "noise_std": level,
                        "class": label, "n": int(mask.sum()),
                        "alarm_rate": float(np.mean(predicted_anomaly)),
                        "meaning": "false_alarm" if label == "SAFE" else "detection",
                    })
            print(f"seed={seed:>3} model={name:<18} fini ({duration:.1f}s)", flush=True)

    full = pd.DataFrame(records)
    full.to_csv(out / "resultats_par_graine_et_bruit.csv", index=False)
    pd.DataFrame(by_class).to_csv(out / "taux_detection_par_classe.csv", index=False)
    pd.DataFrame(histories).to_csv(out / "historiques_entrainement.csv", index=False)
    group = (full.groupby(["model", "noise_std"], as_index=False)
             [["AUROC", "average_precision", "F1", "false_alarm_rate", "recall"]]
             .agg(["mean", "std"]))
    group.columns = ["_".join(map(str, c)).strip("_") for c in group.columns]
    group.to_csv(out / "synthese_moyennes.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name, part in group.groupby("model"):
        ax.plot(part["noise_std"], 100 * part["false_alarm_rate_mean"],
                marker="o", label=name)
    ax.set_xlabel("Bruit additif sur spectres standardisés (sigma)")
    ax.set_ylabel("Fausses alarmes moyennes (%)")
    ax.set_title("Sensibilité à une perturbation spectrale synthétique (validation)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fausses_alarmes_selon_bruit.png", dpi=160)
    plt.close(fig)

    config = {
        "phase": "screening", "representation": "spectres_log_PSD_standardises",
        "train_regime": "10", "validation_regime": "20", "test_regime_reserve": "15",
        "independance_physique_certifiee": False,
        "models": names + ["isolation_forest"], "seeds": seeds,
        "noise_levels_std": levels, "epochs": args.epochs,
        "batch_size": args.batch_size, "latent_dim": args.latent_dim,
        "device": device, "n_train_safe": len(train_safe),
        "n_val": len(val), "n_test_non_utilise": len(test),
        "noise_warning": ("Perturbation GAUSSIENNE sur log-PSD STANDARDISÉ, "
                          "pas équivalente à un bruit vibratoire de capteur."),
        "threshold": "Quantile 99% sur SAFE de calibration propre ; maintenu fixe sous bruit",
        "articles": LITERATURE,
        "not_exact_reproduction": True,
    }
    (out / "protocole.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nMétriques moyennes par modèle et niveau de bruit :")
    print(group.to_string(index=False))
    print(f"\nRésultats exploratoires enregistrés dans {out}")
    return full


if __name__ == "__main__":
    run(parse_args())
