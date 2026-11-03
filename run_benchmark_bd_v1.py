#!/usr/bin/env python
"""Comparaison GT2.04 / GT2.05 sur les données préparées par l'équipe.

Screening = métriques de VALIDATION uniquement ; final = TEST gelé.
Le split fourni par le notebook est une référence exploratoire tant que l'indépendance
entre acquisitions physiques n'est pas documentée.
"""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, average_precision_score, normalized_mutual_info_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.ae_benchmark.collaborative_data import get_split_indices, load_collaborative_database
from src.ae_benchmark.models import build_model
from src.ae_benchmark.training import encode_all, reconstruction_scores, set_seed, train_model
from src.ae_benchmark.deep_clustering import fit_idec, predict_idec
from src.ae_benchmark.evaluation import anomaly_metrics, clustering_metrics


def normalize(data, train_index):
    """Ajustement du scaler UNIQUEMENT sur la partie train requise par la tâche."""
    arr = data.x
    scaler = StandardScaler().fit(arr[train_index].reshape(len(train_index), -1))
    normalized = scaler.transform(arr.reshape(len(arr), -1))
    return normalized.reshape(arr.shape).astype(np.float32), scaler


def cluster_scores(x_train, x_eval, labels, *, seed=42, k=5):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x_train)
    pred = km.predict(x_eval)
    return clustering_metrics(x_eval, labels, pred)


def baseline_anomaly(name, x_train, x_cal, x_eval, y_eval, seed=42):
    tr = x_train.reshape(len(x_train), -1)
    cal = x_cal.reshape(len(x_cal), -1)
    ev = x_eval.reshape(len(x_eval), -1)
    if name == "pca":
        pca = PCA(n_components=min(6, len(tr)-1, tr.shape[1]), random_state=seed).fit(tr)
        err = lambda arr: np.mean((arr-pca.inverse_transform(pca.transform(arr)))**2, axis=1)
        sc_cal, sc_eval = err(cal), err(ev)
    elif name == "isolation_forest":
        model = IsolationForest(n_estimators=300, random_state=seed).fit(tr)
        sc_cal, sc_eval = -model.score_samples(cal), -model.score_samples(ev)
    else:
        raise ValueError(name)
    return {"AUROC": float(roc_auc_score(np.asarray(y_eval)!="SAFE", sc_eval)),
            **anomaly_metrics(sc_cal, sc_eval, y_eval)}


def sklearn_ae_baseline(x_train, x_eval, labels, task, x_cal=None, seed=42, max_iter=1000, representation="spectres"):
    """Reproduit l'architecture et les paramètres MLP du notebook d'équipe.

    Il est réentraîné sur nos propres partitions pour permettre une comparaison
    honnête ; l'ancienne valeur de test ne doit pas servir de référence ici.
    """
    flat = lambda arr: arr.reshape(len(arr), -1)
    tr, ev = flat(x_train), flat(x_eval)
    layers = (16, 6, 16) if representation == "descripteurs" else (128, 16, 128)
    ae = MLPRegressor(hidden_layer_sizes=layers, activation="tanh",
                      learning_rate_init=1e-3, max_iter=max_iter, early_stopping=True,
                      n_iter_no_change=30, random_state=seed)
    ae.fit(tr, tr)
    if task == "anomaly":
        cal = flat(x_cal)
        score_cal = np.mean((cal-ae.predict(cal))**2, axis=1)
        score_eval = np.mean((ev-ae.predict(ev))**2, axis=1)
        return {"AUROC": float(roc_auc_score(np.asarray(labels)!="SAFE", score_eval)),
                **anomaly_metrics(score_cal, score_eval, labels)}
    def encode(a):
        for weights, biases in zip(ae.coefs_[:2], ae.intercepts_[:2]):
            a = np.tanh(a@weights + biases)
        return a
    return cluster_scores(encode(tr), encode(ev), labels, seed=seed)


def parse_args():
    p = argparse.ArgumentParser(description="GT2.04/05 : benchmark sur bd_v1.pkl")
    p.add_argument("--data", default="bd_v1.pkl")
    p.add_argument("--representation", choices=["spectres", "descripteurs"], default="spectres")
    p.add_argument("--models", default="mlp,cnn,denoise,vae")
    p.add_argument("--task", choices=["anomaly", "clustering", "both"], default="both")
    p.add_argument("--phase", choices=["screening", "final"], default="screening")
    p.add_argument("--split-mode", choices=["notebook", "regime", "grouped"], default="notebook")
    p.add_argument("--train-regimes", default="10", help="Mode regime : vitesses du train, séparées par virgule")
    p.add_argument("--val-regime", default="20", help="Mode regime : vitesse réservée à la validation")
    p.add_argument("--test-regime", default="15", help="Mode regime : vitesse tenue à l'écart")
    p.add_argument("--group-column", default=None)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=6)
    p.add_argument("--baseline-sklearn", action="store_true", help="Réentraîner la baseline MLPRegressor du notebook collectif (lent)")
    p.add_argument("--baseline-max-iter", type=int, default=1000)
    p.add_argument("--idec", action="store_true", help="Ajouter IDEC après pré-entraînement du premier AE")
    p.add_argument("--idec-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/gt204_gt205")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def run(args):
    if args.epochs < 1 or args.latent_dim < 1 or args.batch_size < 1:
        raise ValueError("epochs, latent-dim et batch-size doivent être positifs")
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("Fournir une liste de modèles non vide et sans doublons")
    if args.representation == "descripteurs" and any(m != "mlp" for m in names):
        raise ValueError("Descripteurs : seul MLP est retenu ; les CNN sont réservés aux spectres ordonnés")
    if args.phase == "final" and args.split_mode != "grouped":
        raise ValueError("Test final interdit avec le split par fichier du notebook : fournir de vrais group_id")
    set_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA indisponible sur cette machine")
    ds = load_collaborative_database(args.data, args.representation)
    tr, va, te = get_split_indices(
        ds, mode=args.split_mode, group_column=args.group_column, seed=args.seed,
        train_regimes=getattr(args, "train_regimes", "10"),
        val_regime=getattr(args, "val_regime", "20"),
        test_regime=getattr(args, "test_regime", "15"),
    )
    tasks = ["clustering", "anomaly"] if args.task == "both" else [args.task]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = []
    for task in tasks:
        is_anomaly = task == "anomaly"
        tr_fit = tr[ds.labels[tr] == "SAFE"] if is_anomaly else tr
        va_fit = va[ds.labels[va] == "SAFE"] if is_anomaly else va
        if len(tr_fit) < 8 or len(va_fit) < 4:
            raise ValueError(f"{task} : nombre insuffisant d'exemples pour train/calibration SAFE")
        x, scaler = normalize(ds, tr_fit)
        # En screening, les SAFE servent à trois usages distincts :
        # choix de la meilleure époque, calibration du seuil, puis évaluation.
        if is_anomaly and args.phase == "screening":
            shuffled = np.random.default_rng(args.seed).permutation(va_fit)
            if len(shuffled) < 6:
                raise ValueError("Trop peu de SAFE en validation pour trois sous-ensembles")
            third = len(shuffled)//3
            early_stopping = shuffled[:third]
            calibration = shuffled[third:2*third]
            evaluation = np.setdiff1d(va, np.concatenate([early_stopping, calibration]))
        else:
            early_stopping = va_fit
            calibration = va_fit
            evaluation = va if args.phase == "screening" else te
        y_eval = ds.labels[evaluation]
        if is_anomaly and len(np.unique(y_eval == "SAFE")) != 2:
            raise ValueError("Évaluation anomalie : SAFE et défauts nécessaires")
        x_fit, x_val, x_eval, x_cal = x[tr_fit], x[early_stopping], x[evaluation], x[calibration]
        if is_anomaly:
            for name in ("pca", "isolation_forest"):
                metrics = baseline_anomaly(name, x_fit, x_cal, x_eval, y_eval, seed=args.seed)
                report.append({"phase":args.phase,"task":task,"representation":ds.representation,"model":name, **metrics})
        else:
            flat = lambda z: z.reshape(len(z), -1)
            flat_train = flat(x_fit); flat_eval = flat(x_eval)
            report.append({"phase":args.phase,"task":task,"representation":ds.representation,
                           "model":"features+kmeans" if ds.representation == "descripteurs" else "spectra+kmeans",
                           **cluster_scores(flat_train, flat_eval, y_eval, seed=args.seed)})
            pca = PCA(n_components=min(args.latent_dim, len(x_fit)-1, flat_train.shape[1]), random_state=args.seed)
            report.append({"phase":args.phase,"task":task,"representation":ds.representation,
                           "model":"pca+kmeans", **cluster_scores(pca.fit_transform(flat_train), pca.transform(flat_eval), y_eval, seed=args.seed)})
        if args.baseline_sklearn:
            baseline = sklearn_ae_baseline(x_fit, x_eval, y_eval, task,
                        x_cal=x_cal, seed=args.seed, max_iter=args.baseline_max_iter,
                        representation=ds.representation)
            report.append({"phase":args.phase,"task":task,
                           "representation":ds.representation,
                           "model":"AE_collectif_sklearn", **baseline})
        for i, name in enumerate(names):
            # Chaque architecture part de sa graine propre : aucun effet de
            # l'ordre des modèles dans une même commande de benchmark.
            set_seed(args.seed)
            model = build_model(name, length=x.shape[2], channels=x.shape[1], latent_dim=args.latent_dim)
            # Early stopping : validation sans étiquettes (SAFE seulement en détection).
            history = train_model(model, x_fit, x_val, epochs=args.epochs,
                                  batch_size=args.batch_size, seed=args.seed, device=device)
            with (out/f"{task}_{name}_training.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)
            torch.save({"model":name, "channels":x.shape[1], "length":x.shape[2],
                        "latent_dim":args.latent_dim, "state_dict":model.state_dict()},
                       out/f"{task}_{name}_checkpoint.pt")
            if is_anomaly:
                sc_cal = reconstruction_scores(model, x_cal, device=device)
                sc_eval = reconstruction_scores(model, x_eval, device=device)
                metrics = {"AUROC": float(roc_auc_score(y_eval!="SAFE", sc_eval)),
                           **anomaly_metrics(sc_cal, sc_eval, y_eval)}
            else:
                z_fit = encode_all(model, x_fit, device=device)
                z_eval = encode_all(model, x_eval, device=device)
                metrics = cluster_scores(z_fit, z_eval, y_eval, seed=args.seed)
            report.append({"phase":args.phase,"task":task,"representation":ds.representation,"model":name, **metrics})
            if args.idec and not is_anomaly and i == 0:
                idec, idec_history = fit_idec(model, x_fit, n_clusters=5,
                                            epochs=args.idec_epochs, seed=args.seed, device=device)
                pred = predict_idec(idec, x_eval, device=device)
                zeval = encode_all(idec.autoencoder, x_eval, device=device)
                metrics = clustering_metrics(zeval, y_eval, pred)
                report.append({"phase":args.phase,"task":task,"representation":ds.representation,"model":f"idec({name})", **metrics})
                (out/f"{task}_idec_training.json").write_text(json.dumps(idec_history,indent=2),encoding="utf-8")
        joblib.dump(scaler, out/f"{task}_scaler.joblib")
    import pandas as pd
    frame = pd.DataFrame(report)
    frame.to_csv(out/f"comparaison_{args.phase}.csv", index=False)
    meta = {"source":"bd_v1.pkl (équipe)", "representation":ds.representation,
            "split_mode":args.split_mode, "group_column":args.group_column,
            "train_regimes":getattr(args, "train_regimes", "10") if args.split_mode == "regime" else None,
            "val_regime":getattr(args, "val_regime", "20") if args.split_mode == "regime" else None,
            "test_regime":getattr(args, "test_regime", "15") if args.split_mode == "regime" else None,
            "phase":args.phase, "exploratoire_non_certifie":args.split_mode != "grouped",
            "seed":args.seed, "epochs":args.epochs,"device":device,
            "n_train":len(tr), "n_validation":len(va), "n_test_reserve":len(te),
            "models":names, "baseline_sklearn":args.baseline_sklearn, "tasks":tasks}
    (out/f"protocole_{args.phase}.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"\nTableau : {out/f'comparaison_{args.phase}.csv'}")
    if args.split_mode == "notebook":
        print("ATTENTION : split par fichier non certifié indépendant ; résultats EXPLORATOIRES.")
    elif args.split_mode == "regime":
        print("ATTENTION : régimes disjoints, mais indépendance physique NON certifiée ; résultats EXPLORATOIRES.")
        print("Le régime du test est réservé : ne pas lancer --phase final avant validation du protocole.")
    return frame


if __name__ == "__main__":
    run(parse_args())