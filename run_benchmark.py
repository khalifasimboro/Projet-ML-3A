#!/usr/bin/env python
"""Exemple : python run_benchmark.py --demo --models cnn,vae --epochs 2.

ATTENTION: --demo utilise des données SYNTHÉTIQUES (tests de fonctionnement uniquement).
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from src.ae_benchmark.data import (demo_data,from_manifest,grouped_stratified_split,
                                    PerSensorStandardizer)
from src.ae_benchmark.models import build_model,MODEL_REGISTRY
from src.ae_benchmark.training import train_model,reconstruction_scores,set_seed
from src.ae_benchmark.evaluation import (cluster_evaluate,anomaly_metrics,
    baseline_pca_clustering,baseline_iforest_anomaly,
    baseline_handcrafted_clustering,latent_gmm_clustering)
from src.ae_benchmark.deep_clustering import fit_idec,predict_idec
from src.ae_benchmark.evaluation import clustering_metrics
from src.ae_benchmark.training import encode_all
from src.ae_benchmark.reporting import plot_scores, plot_latent
from src.ae_benchmark.signal_processing import signal_representation
from src.ae_benchmark.evaluation import metrics_by_regime
from sklearn.cluster import KMeans


def run(args):
    if not args.demo and not args.manifest:
        raise ValueError("Fournir --demo ou --manifest /chemin/manifest.csv")
    set_seed(args.seed)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    data=(demo_data(length=args.window_length,seed=args.seed) if args.demo else
          from_manifest(args.manifest,window_length=args.window_length,
                        stride=args.stride,max_windows_per_file=args.max_windows_per_file))
    tr,va,te=grouped_stratified_split(data,args.seed)
    representation=getattr(args,"representation","raw")
    data.x=signal_representation(data.x,representation)
    if args.channels != "both":
        # ADAPTER_PLUS_TARD : vérifier l'ordre réel des voies chez A/C.
        channel = {"vertbleu":0,"vert":1}[args.channels]
        data.x=data.x[:,channel:channel+1,:]
    # Phase screening : test strictement réservé. Phase final : paramètres gelés.
    evaluation_idx=va if args.phase=="screening" else te
    print(f"Données: {'DÉMONSTRATION SYNTHÉTIQUE' if args.demo else 'RÉELLES'}")
    print(f"Fenêtres train/val/test: {len(tr)}/{len(va)}/{len(te)}")
    print(f"Phase: {args.phase} (évaluation sur {'validation' if args.phase=='screening' else 'test final'})")
    print(f"Acquisitions train/val/test: {len(set(data.groups[tr]))}/{len(set(data.groups[va]))}/{len(set(data.groups[te]))}")
    device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    models=[v.strip() for v in args.models.split(",") if v.strip()]
    if len(set(models))!=len(models):raise ValueError("Nom de modèle répété")
    invalid=set(models)-set(MODEL_REGISTRY)
    if invalid:raise ValueError(f"Modèles inconnus {invalid}. Possibles : {list(MODEL_REGISTRY)}")
    if args.idec_epochs and (args.idec_backbone not in models or args.task=="anomaly"):
        raise ValueError("IDEC exige --idec-backbone parmi --models et tâche clustering/both")
    rows=[]
    regime_rows=[]
    meta={"demo_synthetique":args.demo,"seed":args.seed,"window_length":args.window_length,
          "models":models,"epochs":args.epochs,"phase":args.phase,"channels":args.channels,
          "representation":representation,"group_split": "stratifié état+régime par group_id",
          "strates":sorted(list(set(zip(data.labels.tolist(),data.regimes.tolist())))),
          "train_groups":sorted(set(data.groups[tr])),
          "val_groups":sorted(set(data.groups[va])),
          "test_groups":sorted(set(data.groups[te]))}
    (out/"split_metadata.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    for task in (["clustering","anomaly"] if args.task=="both" else [args.task]):
        # ADAPTER_PLUS_TARD : choix normalisation à confirmer selon prétraitement amont.
        # En anomalie, scaler entraîné UNIQUEMENT sur SAFE; sinon sur train entier.
        tr_used=tr if task=="clustering" else tr[data.labels[tr]=="SAFE"]
        val_used=va if task=="clustering" else va[data.labels[va]=="SAFE"]
        scaler=PerSensorStandardizer().fit(data.x[tr_used])
        xtr=scaler.transform(data.x[tr_used]);xval=scaler.transform(data.x[val_used]);xte=scaler.transform(data.x[evaluation_idx])
        taskdir=out/task;taskdir.mkdir(exist_ok=True)
        joblib.dump(scaler,taskdir/"scaler.joblib")
        if task=="clustering":
            baseline=baseline_pca_clustering(xtr,xte,data.labels[evaluation_idx],args.latent_dim,args.seed)
            rows.append({"phase":args.phase,"task":task,"model":"pca+kmeans",**baseline})
            rows.append({"phase":args.phase,"task":task,"model":"features+kmeans",**baseline_handcrafted_clustering(xtr,xte,data.labels[evaluation_idx],args.seed)})
        else:
            baseline=baseline_iforest_anomaly(xtr,xte,xval,data.labels[evaluation_idx],args.seed)
            rows.append({"phase":args.phase,"task":task,"model":"isolation_forest",**baseline})
        for i,name in enumerate(models):
            print(f"[{task}] entraînement {name} ({device})...")
            # ADAPTER_PLUS_TARD : hyperparamètres, espace latent, stratégie d'early stopping.
            set_seed(args.seed+i)
            model=build_model(name,length=data.x.shape[-1],channels=data.x.shape[1],latent_dim=args.latent_dim)
            history=train_model(model,xtr,xval,epochs=args.epochs,batch_size=args.batch_size,
                                lr=args.lr,device=device,seed=args.seed+i)
            with open(taskdir/f"{name}_training.csv","w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=["epoch","train_loss","validation_mse"])
                w.writeheader();w.writerows(history)
            torch.save({"state_dict":model.cpu().state_dict(),"model":name,
                        "length":data.x.shape[-1],"latent_dim":args.latent_dim},
                       taskdir/f"{name}.pt")
            model.to(device)
            if task=="clustering":
                metrics=cluster_evaluate(model,xtr,xte,data.labels[evaluation_idx],args.seed,device)
                # Tracer la séparation par régime uniquement après apprentissage.
                ztrain=encode_all(model,xtr,device=device)
                zeval=encode_all(model,xte,device=device)
                predictions=KMeans(n_clusters=5,n_init=10,random_state=args.seed).fit(ztrain).predict(zeval)
                regime_rows.extend({"task":task,"model":name,**r}
                    for r in metrics_by_regime(data.labels[evaluation_idx],predictions,data.regimes[evaluation_idx]))
                if args.plots:
                    plot_latent(zeval,data.labels[evaluation_idx],taskdir/f"{name}_latent_pca.png")
                gm=latent_gmm_clustering(model,xtr,xte,data.labels[evaluation_idx],args.seed,device)
                rows.append({"phase":args.phase,"task":task,"model":name+"+gmm",**gm})
            else:
                val_scores=reconstruction_scores(model,xval,device=device)
                test_scores=reconstruction_scores(model,xte,device=device)
                metrics=anomaly_metrics(val_scores,test_scores,data.labels[evaluation_idx])
                regime_rows.extend({"task":task,"model":name,**r}
                    for r in metrics_by_regime(data.labels[evaluation_idx],None,data.regimes[evaluation_idx],
                                               task="anomaly",scores=test_scores,threshold=metrics["threshold"]))
            rows.append({"phase":args.phase,"task":task,"model":name,**metrics})
            if task=="clustering" and args.idec_epochs>0 and name==args.idec_backbone:
                print(f"[clustering] optimisation IDEC avec encodeur {name}...")
                idec, idec_history=fit_idec(model,xtr,epochs=args.idec_epochs,
                      batch_size=args.batch_size,seed=args.seed,device=device,gamma=args.idec_gamma)
                idec_prediction=predict_idec(idec,xte,device=device)
                idec_metrics=clustering_metrics(encode_all(idec.autoencoder,xte,device=device),
                                                    data.labels[evaluation_idx],idec_prediction)
                rows.append({"phase":args.phase,"task":task,"model":name+"+idec",**idec_metrics})
                torch.save({"state_dict":idec.cpu().state_dict(),"backbone":name,
                            "gamma":args.idec_gamma,"n_clusters":5},taskdir/"idec.pt")
                pd.DataFrame(idec_history).to_csv(taskdir/"idec_training.csv",index=False)
            print(f"    {metrics}")
    result=pd.DataFrame(rows)
    result.to_csv(out/"benchmark_results.csv",index=False)
    pd.DataFrame(regime_rows).to_csv(out/"metrics_par_regime.csv",index=False)
    if args.plots: plot_scores(result,out/"figures")
    (out/"AVERTISSEMENT.txt").write_text(
        "Données SYNTHÉTIQUES: résultats NON SCIENTIFIQUES; NE PAS INSÉRER DANS LE RAPPORT.\n"
        if args.demo else
        "Résultats de développement: vérifier groupes indépendants, unités, données et protocole avant publication.\n",
        encoding="utf-8")
    print("\nRésultats enregistrés dans :",out.resolve())
    print(result.to_string(index=False))
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Benchmark exploratoire AE sur vibrations")
    parser.add_argument("--phase",choices=["screening","final"],default="screening",
        help="screening = validation seulement, final = test après gel des réglages")
    parser.add_argument("--demo",action="store_true",help="FAUSSES DONNÉES uniquement pour test technique")
    parser.add_argument("--manifest",type=str,help="CSV des acquisitions réelles")
    parser.add_argument("--task",choices=["clustering","anomaly","both"],default="both")
    parser.add_argument("--channels",choices=["both","vertbleu","vert"],default="both",
                        help="Ablation de capteur : deux voies, voie vertbleu ou voie vert")
    parser.add_argument("--models",default="cnn,denoise,vae,multiscale,memory")
    parser.add_argument("--representation",choices=["raw","envelope","log_spectrum"],default="raw",
                        help="Ablation des représentations : ne pas interpréter les bins en Hz sans fs")
    parser.add_argument("--window-length",type=int,default=256)
    parser.add_argument("--stride",type=int,default=128)
    parser.add_argument("--max-windows-per-file",type=int,default=None)
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--latent-dim",type=int,default=16)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--cpu",action="store_true")
    parser.add_argument("--no-plots",dest="plots",action="store_false",help="Désactive la génération PNG")
    parser.set_defaults(plots=True)
    parser.add_argument("--out",default="outputs")
    parser.add_argument("--idec-epochs",type=int,default=0,help="0=désactivé ; fine-tuning IDEC sur la branche clustering")
    parser.add_argument("--idec-backbone",default="cnn",help="Nom du modèle pré-entraîné utilisé par IDEC")
    parser.add_argument("--idec-gamma",type=float,default=0.1)
    run(parser.parse_args())