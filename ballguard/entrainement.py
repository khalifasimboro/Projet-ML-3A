"""Pipeline d'entraînement complet, reproductible (parties 6 à 11 du notebook).

Entrée : la base de descripteurs (une ligne par signal, colonne `split` déjà fixée) et les spectres.
Sortie : le modèle à déployer et les artefacts affichés par l'application.
"""
from datetime import date

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, normalized_mutual_info_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import CLASSES, K, PERCENTILE_SEUIL, SEED, VITESSES
from .descripteurs import FEATURES
from .modeles import creer_ae, encoder, erreur_par_descripteur, erreur_reconstruction

A_LOGUER = [f for f in FEATURES if f.split("_")[0] in ("rms", "pkpk", "kurtosis", "crete")]


def decouper(meta):
    """Découpage 70 / 15 / 15 stratifié par (classe, vitesse), AVANT tout prétraitement."""
    strate = meta.classe + "_" + meta.vitesse.astype(str)
    idx_train, idx_reste = train_test_split(meta.index, test_size=0.30, stratify=strate, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_reste, test_size=0.50, stratify=strate[idx_reste], random_state=SEED)
    split = pd.Series("train", index=meta.index)
    split[idx_val], split[idx_test] = "val", "test"
    return split


def eta2(valeurs, groupes):
    """Part de la variance d'un descripteur expliquée par un regroupement (0 à 1)."""
    valeurs, groupes = np.asarray(valeurs, dtype=float), np.asarray(groupes)
    intra = sum(((valeurs[groupes == g] - valeurs[groupes == g].mean()) ** 2).sum() for g in np.unique(groupes))
    return float(1 - intra / ((valeurs - valeurs.mean()) ** 2).sum())


def _scores_clustering(etiquettes, clusters, donnees):
    return {"silhouette": float(silhouette_score(donnees, clusters)),
            "ARI": float(adjusted_rand_score(etiquettes, clusters)),
            "NMI": float(normalized_mutual_info_score(etiquettes, clusters))}


def _contingence(classes, clusters, k):
    return pd.crosstab(pd.Categorical(classes, CLASSES), pd.Categorical(clusters, range(k)),
                       dropna=False).to_numpy().tolist()


def _tsne(Z, dim):
    return TSNE(n_components=dim, random_state=SEED, perplexity=30).fit_transform(Z).astype(np.float32)


def _nommer(clusters_train, classes_train):
    tab = pd.crosstab(clusters_train, classes_train)
    return {int(k): str(v) for k, v in tab.idxmax(axis=1).items()}


def _variantes(bd, spectres, classe, vitesse, m_train, m_val, m_test, safe, y, Z_norm, km_final, res_km, exact_km, journal):
    """Modèles repris de l'étude comparative V6 : MLP-AE + GMM, et CNN-AE spectral (détection + K-means).

    Renvoie (objets à embarquer dans le modèle, tableau de comparaison pour l'application).
    """
    from .cnn import encoder_cnn, entrainer_cnn, erreur_cnn, etat

    def ligne(nom, cl, Zt, exact):
        return {"variante": nom, "ARI": float(adjusted_rand_score(classe[m_test], cl)),
                "NMI": float(normalized_mutual_info_score(classe[m_test], cl)),
                "silhouette": float(silhouette_score(Zt, cl)), "ARI_vitesse": float(adjusted_rand_score(vitesse[m_test], cl)),
                "exactitude": exact}

    table = [ligne("MLP-AE + K-means", km_final.predict(Z_norm[m_test]), Z_norm[m_test], exact_km)]

    journal("Variante : MLP-AE + GMM (covariance complète)")
    gmm = GaussianMixture(K, covariance_type="full", random_state=SEED, n_init=3).fit(Z_norm[m_train])
    noms_gmm = _nommer(gmm.predict(Z_norm[m_train]), classe[m_train])
    cl = gmm.predict(Z_norm[m_test])
    table.append(ligne("MLP-AE + GMM", cl, Z_norm[m_test], float(np.mean([noms_gmm[c] for c in cl] == classe[m_test]))))

    journal("Variante : CNN-AE spectral, détection (entraîné sur les spectres sains)")
    sc_det = StandardScaler().fit(spectres[m_train & safe])
    S_det = sc_det.transform(spectres).astype(np.float32)
    cnn_det, _ = entrainer_cnn(S_det[m_train & safe], S_det[m_val & safe])
    score = erreur_cnn(cnn_det, S_det)
    seuil = float(np.percentile(score[m_val & safe], PERCENTILE_SEUIL))

    journal("Variante : CNN-AE spectral, clustering (spectres standardisés par vitesse)")
    sc_v = {v: StandardScaler().fit(spectres[m_train & (vitesse == v)]) for v in VITESSES}
    S_clu = np.zeros_like(spectres, dtype=np.float32)
    for v, sc in sc_v.items():
        S_clu[vitesse == v] = sc.transform(spectres[vitesse == v])
    cnn_clu, _ = entrainer_cnn(S_clu[m_train], S_clu[m_val])
    Z_cnn = encoder_cnn(cnn_clu, S_clu)
    km_cnn = KMeans(K, n_init=10, random_state=SEED).fit(Z_cnn[m_train])
    noms_cnn = _nommer(km_cnn.labels_, classe[m_train])
    cl = km_cnn.predict(Z_cnn[m_test])
    table.append(ligne("CNN-AE spectral + K-means", cl, Z_cnn[m_test], float(np.mean([noms_cnn[c] for c in cl] == classe[m_test]))))
    for l in table:
        journal(f"  {l['variante']:<28} ARI {l['ARI']:.4f}  silhouette {l['silhouette']:.4f}  exactitude {l['exactitude']:.4f}")
    journal(f"  CNN détection : seuil {seuil:.4f}, fausses alarmes {int((score[m_test & safe] > seuil).sum())}")

    variantes = {
        "gmm": {"gmm": gmm, "nom_cluster": noms_gmm},
        "cnn": {"scaler_det": sc_det, "poids_det": etat(cnn_det), "seuil": seuil,
                "scalers_vitesse": sc_v, "poids_clu": etat(cnn_clu), "kmeans": km_cnn, "nom_cluster": noms_cnn,
                "score": score, "Z": Z_cnn},
    }
    return variantes, table


def entrainer(bd, spectres, journal=print):
    """Entraîne tous les modèles. Renvoie (modele, artefacts)."""
    rng = np.random.default_rng(SEED)
    classe, vitesse = bd.classe.to_numpy(), bd.vitesse.to_numpy()
    m_train, m_val, m_test = [(bd.split == s).to_numpy() for s in ("train", "val", "test")]
    safe = classe == "SAFE"
    y = (~safe).astype(int)                                  # 1 = défaut ; sert UNIQUEMENT à évaluer
    F = bd[FEATURES].to_numpy(dtype=float)

    # ------------------------------------------------------------ détection d'anomalie (partie 6)
    journal("Détection : autoencodeur entraîné sur les signaux sains du train")
    scaler = StandardScaler().fit(F[m_train & safe])
    X = scaler.transform(F)
    ae = creer_ae((16, 6, 16)).fit(X[m_train & safe], X[m_train & safe])
    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(X[m_train & safe])
    pca = PCA(n_components=6, random_state=SEED).fit(X[m_train & safe])
    scaler_spec = StandardScaler().fit(spectres[m_train & safe])
    X_spec = scaler_spec.transform(spectres)
    ae_spec = creer_ae((128, 16, 128)).fit(X_spec[m_train & safe], X_spec[m_train & safe])

    scores = {"Isolation Forest": -iso.score_samples(X),
              "ACP": np.mean((X - pca.inverse_transform(pca.transform(X))) ** 2, axis=1),
              "AE (descripteurs)": erreur_reconstruction(ae, X),
              "AE (spectre)": erreur_reconstruction(ae_spec, X_spec)}
    seuils, table_det, roc = {}, [], {}
    for nom, s in scores.items():
        seuils[nom] = float(np.percentile(s[m_val & safe], PERCENTILE_SEUIL))   # fixé SANS regarder le test
        pred = (s[m_test] > seuils[nom]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[m_test], pred, labels=[0, 1]).ravel()
        table_det.append({"modele": nom, "AUROC": float(roc_auc_score(y[m_test], s[m_test])),
                          "rappel": tp / (tp + fn), "precision": tp / max(tp + fp, 1),
                          "fausses_alarmes": fp / (fp + tn), "nb_fausses_alarmes": int(fp),
                          "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]]})
        fpr, tpr, _ = roc_curve(y[m_test], s[m_test])
        roc[nom] = (fpr.astype(np.float32), tpr.astype(np.float32))
    score_ae, seuil_ae = scores["AE (descripteurs)"], seuils["AE (descripteurs)"]
    err_feat = erreur_par_descripteur(ae, X)
    erreur_par_classe = [err_feat[m_test & (classe == c)].mean(axis=0).tolist() for c in CLASSES]
    taux_alarme = [[float((score_ae[m_test & (classe == c) & (vitesse == v)] > seuil_ae).mean())
                    for v in VITESSES] for c in CLASSES]

    # importance par permutation (détection) : baisse du score moyen des défauts quand on mélange un descripteur
    defauts = m_test & ~safe
    ref_det = np.log10(score_ae[defauts]).mean()
    perm_det = {}
    for j, f in enumerate(FEATURES):
        baisses = []
        for _ in range(5):
            Xp = X[m_test].copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            baisses.append(ref_det - np.log10(erreur_reconstruction(ae, Xp)[~safe[m_test]]).mean())
        perm_det[f] = float(np.mean(baisses))

    # ------------------------------------------------------------ clustering sans correction (partie 7)
    journal("Clustering sans correction")
    scaler_clu = StandardScaler().fit(F[m_train])
    X_clu = scaler_clu.transform(F)
    ae_clu = creer_ae((16, 6, 16)).fit(X_clu[m_train], X_clu[m_train])
    Z = encoder(ae_clu, X_clu)

    def evaluer(donnees):
        km = KMeans(K, n_init=10, random_state=SEED).fit(donnees[m_train])
        cl = km.predict(donnees[m_test])
        return km, cl, {**_scores_clustering(classe[m_test], cl, donnees[m_test]),
                        "ARI_vitesse": float(adjusted_rand_score(vitesse[m_test], cl))}

    _, _, res_base = evaluer(X_clu)
    _, cl_ae, res_ae = evaluer(Z)
    ks = list(range(2, 16))

    def courbes_k(donnees):
        kms = [KMeans(k, n_init=10, random_state=SEED).fit(donnees[m_train]) for k in ks]
        return ([float(silhouette_score(donnees[m_train], km.labels_)) for km in kms],
                [float(km.inertia_) for km in kms])

    sil_sans, inertie_sans = courbes_k(Z)

    # ------------------------------------------------------------ un modèle par vitesse (partie 8)
    journal("Clustering : un modèle par vitesse")
    par_vitesse = []
    for v in VITESSES:
        m_tr, m_te = m_train & (vitesse == v), m_test & (vitesse == v)
        X_v = StandardScaler().fit(F[m_tr]).transform(F)
        cl = KMeans(K, n_init=10, random_state=SEED).fit(X_v[m_tr]).predict(X_v[m_te])
        par_vitesse.append({"vitesse": v, "modele": "baseline", **_scores_clustering(classe[m_te], cl, X_v[m_te])})
        Z_v = encoder(creer_ae((16, 6, 16)).fit(X_v[m_tr], X_v[m_tr]), X_v)
        cl = KMeans(K, n_init=10, random_state=SEED).fit(Z_v[m_tr]).predict(Z_v[m_te])
        par_vitesse.append({"vitesse": v, "modele": "autoencodeur", **_scores_clustering(classe[m_te], cl, Z_v[m_te])})

    # ------------------------------------------------------------ log + standardisation par vitesse (partie 9)
    journal("Clustering : log + standardisation par vitesse (solution retenue)")
    i_log = [FEATURES.index(f) for f in A_LOGUER]
    F_log = F.copy()
    F_log[:, i_log] = np.log10(F_log[:, i_log])
    X_norm, scalers_vitesse = np.zeros_like(F), {}
    for v in VITESSES:
        m_v = vitesse == v
        scalers_vitesse[v] = StandardScaler().fit(F_log[m_train & m_v])
        X_norm[m_v] = scalers_vitesse[v].transform(F_log[m_v])
    ae_norm = creer_ae((16, 6, 16)).fit(X_norm[m_train], X_norm[m_train])
    Z_norm = encoder(ae_norm, X_norm)
    _, _, res_base_norm = evaluer(X_norm)
    km_final, cl_norm, res_ae_norm = evaluer(Z_norm)
    sil_corr, inertie_corr = courbes_k(Z_norm)

    # nom de chaque cluster = classe majoritaire de ses signaux d'entraînement
    tab = pd.crosstab(km_final.labels_, classe[m_train])
    nom_cluster = {int(k): str(v) for k, v in tab.idxmax(axis=1).items()}
    exactitude = float(np.mean([nom_cluster[c] for c in cl_norm] == classe[m_test]))

    # importance par permutation (clustering) : baisse de l'ARI quand on mélange un descripteur
    perm_clu = {}
    for j, f in enumerate(FEATURES):
        baisses = []
        for _ in range(5):
            Xp = X_norm[m_test].copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            baisses.append(res_ae_norm["ARI"] - adjusted_rand_score(classe[m_test], km_final.predict(encoder(ae_norm, Xp))))
        perm_clu[f] = float(np.mean(baisses))

    eta = {"features": FEATURES,
           "vitesse_avant": [eta2(F[m_train, j], vitesse[m_train]) for j in range(len(FEATURES))],
           "classe_avant": [eta2(F[m_train, j], classe[m_train]) for j in range(len(FEATURES))],
           "classe_apres": [eta2(X_norm[m_train, j], classe[m_train]) for j in range(len(FEATURES))]}

    # ------------------------------------------------------------ variantes de l'étude V6 (option B)
    variantes, table_variantes = _variantes(bd, spectres, classe, vitesse, m_train, m_val, m_test, safe, y,
                                            Z_norm, km_final, res_ae_norm, exactitude, journal)
    for nom in ("CNN-AE spectral",):
        s = variantes["cnn"]["score"]
        seuil = variantes["cnn"]["seuil"]
        pred = (s[m_test] > seuil).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[m_test], pred, labels=[0, 1]).ravel()
        table_det.append({"modele": nom, "AUROC": float(roc_auc_score(y[m_test], s[m_test])),
                          "rappel": tp / (tp + fn), "precision": tp / max(tp + fp, 1),
                          "fausses_alarmes": fp / (fp + tn), "nb_fausses_alarmes": int(fp),
                          "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]]})
        fpr, tpr, _ = roc_curve(y[m_test], s[m_test])
        roc[nom] = (fpr.astype(np.float32), tpr.astype(np.float32))
        seuils[nom] = seuil
    Z_cnn = variantes["cnn"].pop("Z")
    variantes["cnn"].pop("score")

    journal("Projections t-SNE de l'espace latent")
    latent = {"Z_sans": Z.astype(np.float32), "Z_corr": Z_norm.astype(np.float32), "Z_cnn": Z_cnn.astype(np.float32),
              "tsne2_sans": _tsne(Z[m_test], 2), "tsne2_corr": _tsne(Z_norm[m_test], 2), "tsne2_cnn": _tsne(Z_cnn[m_test], 2),
              "tsne3_sans": _tsne(Z[m_test], 3), "tsne3_corr": _tsne(Z_norm[m_test], 3), "tsne3_cnn": _tsne(Z_cnn[m_test], 3)}

    # ------------------------------------------------------------ références « saines » pour l'application
    reference = {}
    for v in VITESSES:
        m = m_train & safe & (vitesse == v)
        reference[v] = {"p01": np.percentile(F[m], 1, axis=0).tolist(), "p50": np.percentile(F[m], 50, axis=0).tolist(),
                        "p99": np.percentile(F[m], 99, axis=0).tolist(), "spectre": spectres[m].mean(axis=0).astype(np.float32)}

    modele = {
        "description": "Ballguard : détection d'anomalie + identification de l'état d'un roulement",
        "date": str(date.today()), "version_sklearn": sklearn.__version__, "features": FEATURES,
        "detection": {"scaler": scaler, "autoencodeur": ae, "seuil": float(seuil_ae)},
        "detection_spectre": {"scaler": scaler_spec, "autoencodeur": ae_spec, "seuil": seuils["AE (spectre)"]},
        "baselines": {"iso": iso, "pca": pca, "seuil_iso": seuils["Isolation Forest"], "seuil_pca": seuils["ACP"]},
        "diagnostic": {"a_loguer": A_LOGUER, "scalers_vitesse": scalers_vitesse, "autoencodeur": ae_norm,
                       "kmeans": km_final, "nom_cluster": nom_cluster},
        "reference": reference,
        "variantes": variantes,
        "performances_test": {"detection": {k: table_det[2][k] for k in ("AUROC", "rappel", "fausses_alarmes")},
                              "diagnostic": {**res_ae_norm, "exactitude": exactitude}},
    }
    artefacts = {
        "meta": {"date": str(date.today()), "version_sklearn": sklearn.__version__, "n": int(len(bd)),
                 "n_train": int(m_train.sum()), "n_val": int(m_val.sum()), "n_test": int(m_test.sum())},
        "detection": {"table": table_det, "roc": roc, "seuils": seuils, "score_ae": score_ae.astype(np.float32),
                      "erreur_par_classe": erreur_par_classe, "taux_alarme": taux_alarme,
                      "score_spec": scores["AE (spectre)"].astype(np.float32),
                      "taux_alarme_spec": [[float((scores["AE (spectre)"][m_test & (classe == c) & (vitesse == v)]
                                                   > seuils["AE (spectre)"]).mean()) for v in VITESSES] for c in CLASSES]},
        "clustering": {"sans": {"baseline": res_base, "autoencodeur": res_ae},
                       "par_vitesse": par_vitesse,
                       "corrige": {"baseline": res_base_norm, "autoencodeur": res_ae_norm},
                       "ks": ks, "sil_sans": sil_sans, "sil_corr": sil_corr,
                       "inertie_sans": inertie_sans, "inertie_corr": inertie_corr,
                       "contingence_sans": _contingence(classe[m_test], cl_ae, K),
                       "contingence_corr": _contingence(classe[m_test], cl_norm, K),
                       "exactitude": exactitude},
        "latent": latent,
        "variantes": table_variantes,
        "permutation": {"detection": perm_det, "clustering": perm_clu},
        "eta": eta,
    }
    return modele, artefacts
