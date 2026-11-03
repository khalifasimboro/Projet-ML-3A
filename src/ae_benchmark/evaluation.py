"""Protocoles comparatifs: clustering non supervisé et détection SAFE vs défaut.

Les étiquettes ne servent qu'au découpage stratifié, à l'évaluation, et à sélectionner
SAFE pour l'entraînement des modèles de détection d'anomalies.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from .signal_features import vibration_features
from sklearn.metrics import (adjusted_rand_score,normalized_mutual_info_score,
                             precision_recall_curve,auc,
                             silhouette_score,average_precision_score,
                             f1_score,precision_score,recall_score,confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from .training import encode_all,reconstruction_scores


def cluster_evaluate(model,x_train,x_test,y_test,seed=42,device="cpu"):
    ztrain=encode_all(model,x_train,device=device)
    ztest=encode_all(model,x_test,device=device)
    km=KMeans(n_clusters=5,random_state=seed,n_init=10).fit(ztrain)
    pred=km.predict(ztest)
    return clustering_metrics(ztest,y_test,pred)


def clustering_metrics(ztest,y_test,pred):
    metrics={"ARI":float(adjusted_rand_score(y_test,pred)),
             "NMI":float(normalized_mutual_info_score(y_test,pred))}
    if 1 < len(np.unique(pred)) < len(pred):
        metrics["silhouette"]=float(silhouette_score(ztest,pred))
    else: metrics["silhouette"]=float("nan")
    return metrics


def anomaly_metrics(normal_val_scores,test_scores,y_test,threshold_quantile=.99):
    """Seuil sur validation SAFE uniquement. PR-AUC sur test à composition déclarée.

    ADAPTER_PLUS_TARD: fixer un taux de fausses alarmes acceptable avec les collègues;
    si le scénario change, ne PAS recalibrer le seuil sur l'ensemble de test.
    """
    normal_val_scores=np.asarray(normal_val_scores)
    if not len(normal_val_scores): raise ValueError("Validation SAFE vide : impossible de calibrer le seuil")
    binary=np.asarray(y_test)!="SAFE"
    if np.unique(binary).size!=2:
        raise ValueError("Le test doit contenir SAFE et au moins un défaut")
    threshold=float(np.quantile(normal_val_scores,threshold_quantile))
    pred=np.asarray(test_scores)>threshold
    tn,fp,fn,tp=confusion_matrix(binary,pred,labels=[False,True]).ravel()
    precision_curve, recall_curve, _ = precision_recall_curve(binary,test_scores)
    return {"PR_AUC":float(auc(recall_curve,precision_curve)),
            "average_precision":float(average_precision_score(binary,test_scores)),
            "precision":float(precision_score(binary,pred,zero_division=0)),
            "recall":float(recall_score(binary,pred,zero_division=0)),
            "F1":float(f1_score(binary,pred,zero_division=0)),
            "false_alarm_rate":float(fp/max(1,fp+tn)),
            "threshold":threshold}


def baseline_pca_clustering(x_train,x_test,y_test,latent_dim=16,seed=42):
    ztrain=x_train.reshape(len(x_train),-1)
    ztest=x_test.reshape(len(x_test),-1)
    pca=PCA(n_components=min(latent_dim,len(ztrain)-1,ztrain.shape[1]),random_state=seed)
    ztrain=pca.fit_transform(ztrain); ztest=pca.transform(ztest)
    pred=KMeans(n_clusters=5,random_state=seed,n_init=10).fit(ztrain).predict(ztest)
    return clustering_metrics(ztest,y_test,pred)


def baseline_iforest_anomaly(x_train_safe,x_test,x_val_safe,y_test,seed=42):
    flatten=lambda x:x.reshape(len(x),-1)
    forest=IsolationForest(random_state=seed,contamination="auto",n_estimators=100,n_jobs=-1)
    forest.fit(flatten(x_train_safe))
    score_val=-forest.score_samples(flatten(x_val_safe))
    score_test=-forest.score_samples(flatten(x_test))
    return anomaly_metrics(score_val,score_test,y_test)


def baseline_handcrafted_clustering(x_train, x_test, y_test, seed=42):
    """Caractéristiques vibratoires normalisées TRAIN seulement + KMeans."""
    sc = StandardScaler().fit(vibration_features(x_train))
    z_train = sc.transform(vibration_features(x_train))
    z_test = sc.transform(vibration_features(x_test))
    pred = KMeans(n_clusters=5, random_state=seed, n_init=10).fit(z_train).predict(z_test)
    return clustering_metrics(z_test, y_test, pred)


def latent_gmm_clustering(model,x_train,x_test,y_test,seed=42,device='cpu'):
    z_train=encode_all(model,x_train,device=device)
    z_test=encode_all(model,x_test,device=device)
    pred=GaussianMixture(n_components=5,random_state=seed,reg_covar=1e-4).fit(z_train).predict(z_test)
    return clustering_metrics(z_test,y_test,pred)


def metrics_by_regime(y_true, predictions, regimes, task='clustering', scores=None, threshold=None):
    """Affichage descriptif par régime, calculé après verrouillage des modèles.

    En clustering ARI/NMI globaux par régime. Pour anomalie, score PR-AUC
    uniquement si SAFE + défaut présents, FAR basé sur le seuil de validation.
    """
    out=[]
    for regime in sorted(set(regimes)):
        sel=np.asarray(regimes)==regime
        if task=='clustering':
            m={'ARI': float(adjusted_rand_score(np.asarray(y_true)[sel],np.asarray(predictions)[sel])),
               'NMI': float(normalized_mutual_info_score(np.asarray(y_true)[sel],np.asarray(predictions)[sel]))}
        else:
            truth=np.asarray(y_true)[sel]!='SAFE'
            ss=np.asarray(scores)[sel]
            pred=ss>threshold
            m={'average_precision': float(average_precision_score(truth,ss)) if len(np.unique(truth))==2 else None,
               'false_alarm_rate':float(np.mean(pred[~truth])) if np.any(~truth) else None,
               'recall':float(np.mean(pred[truth])) if np.any(truth) else None}
        out.append({'regime':str(regime),'n_windows':int(sel.sum()),**m})
    return out