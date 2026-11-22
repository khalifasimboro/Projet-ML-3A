"""Utilisation du modèle : du signal brut au diagnostic expliqué."""
import joblib
import numpy as np
import pandas as pd

from .config import CLASSES, FICHIER_MODELE, VITESSES
from .descripteurs import extraire_depuis_signal
from .lecture import lire_signal
from .modeles import encoder, erreur_par_descripteur


class VitesseInconnue(ValueError):
    """Aucune normalisation n'a été apprise pour cette vitesse."""


def charger_modele(chemin=FICHIER_MODELE):
    return joblib.load(chemin)


def _preparer_diagnostic(modele, D, vitesses):
    """Log des descripteurs positifs puis standardisation propre à la vitesse de chaque signal."""
    diag = modele["diagnostic"]
    vitesses = np.asarray(vitesses).astype(int)
    inconnues = sorted(set(vitesses.tolist()) - set(diag["scalers_vitesse"]))
    if inconnues:
        raise VitesseInconnue(f"vitesse(s) {inconnues} inconnue(s) : le modèle connaît {sorted(diag['scalers_vitesse'])}")
    F = D[modele["features"]].astype(float).copy()
    F[diag["a_loguer"]] = np.log10(F[diag["a_loguer"]])
    Xc = np.zeros(F.shape)
    for v, sc in diag["scalers_vitesse"].items():
        m = vitesses == v
        if m.any():
            Xc[m] = sc.transform(F.to_numpy()[m])
    return Xc


def analyser(modele, descripteurs, vitesses):
    """Applique le modèle à une table de descripteurs (une ligne par signal).

    Renvoie pour chaque signal : score d'anomalie, alarme, cluster et état estimé.
    """
    D = descripteurs[modele["features"]].astype(float)
    det = modele["detection"]
    Xd = det["scaler"].transform(D.to_numpy())
    score = erreur_par_descripteur(det["autoencodeur"], Xd).mean(axis=1)
    diag = modele["diagnostic"]
    Xc = _preparer_diagnostic(modele, D, vitesses)
    clusters = diag["kmeans"].predict(encoder(diag["autoencodeur"], Xc))
    return pd.DataFrame({"score_anomalie": score, "alarme": score > det["seuil"], "cluster": clusters,
                         "etat_estime": [diag["nom_cluster"][int(c)] for c in clusters]}, index=descripteurs.index)


# "spec" (retenu) : détection par l'AE spectral + type de défaut par le CNN-AE spectral + K-means
# "km", "gmm"     : MLP-AE sur les 26 descripteurs + K-means ou GMM (seuls explicables descripteur par descripteur)
# "cnn"           : CNN-AE spectral pour la détection et le type (conservé pour comparaison)
VARIANTES = ("spec", "km", "gmm", "cnn")


def _proximite_distances(dist, noms):
    poids = np.exp(-(dist - dist.min()) ** 2 / (2 * np.median(dist) ** 2 / 9 + 1e-12))
    prox = {}
    for k, p in enumerate(poids / poids.sum()):
        prox[noms[k]] = prox.get(noms[k], 0.0) + float(p)
    return prox


def _cnn(modele):
    """Réseaux CNN reconstruits une fois puis gardés en mémoire (import de PyTorch seulement si besoin)."""
    v = modele["variantes"]["cnn"]
    if "_reseaux" not in v:
        from .cnn import depuis_etat
        v["_reseaux"] = (depuis_etat(v["poids_det"]), depuis_etat(v["poids_clu"]))
    return v["_reseaux"]


def score_spectre(modele, spectres):
    """Erreur de reconstruction de l'autoencodeur spectral (modèle de détection retenu), un score par signal."""
    d = modele["detection_spectre"]
    X = d["scaler"].transform(np.atleast_2d(np.asarray(spectres, dtype=float)))
    return ((X - d["autoencodeur"].predict(X)) ** 2).mean(axis=1)


def type_cnn(modele, spectres, vitesses):
    """Type de défaut par le CNN-AE spectral + K-means (modèle retenu). Renvoie (clusters, distances aux centres)."""
    from .cnn import encoder_cnn
    v = modele["variantes"]["cnn"]
    S = np.atleast_2d(np.asarray(spectres, dtype=float))
    vitesses = np.asarray(vitesses).astype(int)
    S_v = np.zeros_like(S, dtype=np.float32)
    for vit in set(vitesses.tolist()):
        m = vitesses == vit
        S_v[m] = v["scalers_vitesse"][vit].transform(S[m])
    Z = encoder_cnn(_cnn(modele)[1], S_v)
    dist = np.linalg.norm(Z[:, None, :] - v["kmeans"].cluster_centers_[None, :, :], axis=2)
    return dist.argmin(axis=1), dist


def analyser_retenu(modele, spectres, vitesses):
    """Modèle retenu appliqué à une série de spectres : score (AE spectral), alarme, état (CNN-AE + K-means)."""
    vitesses = np.asarray(vitesses).astype(int)
    inconnues = sorted(set(vitesses.tolist()) - set(VITESSES))
    if inconnues:
        raise VitesseInconnue(f"vitesse(s) {inconnues} inconnue(s) : le modèle connaît {VITESSES}")
    score = score_spectre(modele, spectres)
    clusters, _ = type_cnn(modele, spectres, vitesses)
    noms = modele["variantes"]["cnn"]["nom_cluster"]
    return pd.DataFrame({"score_anomalie": score, "alarme": score > modele["detection_spectre"]["seuil"],
                         "cluster": clusters, "etat_estime": [noms[int(c)] for c in clusters]})


def score_cnn(modele, spectre):
    from .cnn import erreur_cnn
    v = modele["variantes"]["cnn"]
    return float(erreur_cnn(_cnn(modele)[0], v["scaler_det"].transform(np.asarray(spectre)[None]))[0])


def expliquer(modele, desc, vitesse, spectre=None, variante="spec"):
    """Diagnostic détaillé d'un seul signal à partir de ses 26 descripteurs (dict) et de son spectre.

    `variante` choisit le modèle qui donne le verdict et le type de défaut : "spec" (retenu : AE spectral pour la
    détection, CNN-AE spectral + K-means pour le type), "km" (MLP-AE + K-means), "gmm" (MLP-AE + GMM) ou "cnn"
    (CNN-AE spectral pour les deux). Les variantes spectrales demandent le spectre.
    """
    if variante not in VARIANTES:
        raise ValueError(f"variante inconnue : {variante}")
    if variante in ("spec", "cnn") and spectre is None:
        raise ValueError("cette variante a besoin du spectre du signal")
    D = pd.DataFrame([desc])[modele["features"]]
    det, base, diag = modele["detection"], modele["baselines"], modele["diagnostic"]
    Xd = det["scaler"].transform(D.to_numpy())
    err = erreur_par_descripteur(det["autoencodeur"], Xd)[0]
    score = float(err.mean())

    # verdict de chaque modèle de détection, exprimé en score / seuil
    s_iso = float(-base["iso"].score_samples(Xd)[0])
    s_pca = float(np.mean((Xd - base["pca"].inverse_transform(base["pca"].transform(Xd))) ** 2))
    comparaison = [
        {"modele": "Isolation Forest", "score": s_iso, "seuil": base["seuil_iso"], "alarme": s_iso > base["seuil_iso"]},
        {"modele": "ACP", "score": s_pca, "seuil": base["seuil_pca"], "alarme": s_pca > base["seuil_pca"]},
        {"modele": "AE (descripteurs)", "score": score, "seuil": det["seuil"], "alarme": score > det["seuil"]},
    ]
    if spectre is not None and "detection_spectre" in modele:
        s_spec, seuil_spec = float(score_spectre(modele, spectre)[0]), modele["detection_spectre"]["seuil"]
        comparaison.insert(0, {"modele": "AE (spectre)", "score": s_spec, "seuil": seuil_spec, "alarme": s_spec > seuil_spec})
    if spectre is not None:
        s_cnn, seuil_cnn = score_cnn(modele, spectre), modele["variantes"]["cnn"]["seuil"]
        comparaison.append({"modele": "CNN-AE spectral", "score": s_cnn, "seuil": seuil_cnn, "alarme": s_cnn > seuil_cnn})

    # type de défaut selon la variante choisie
    if variante in ("spec", "cnn"):
        from .cnn import encoder_cnn
        v = modele["variantes"]["cnn"]
        z = encoder_cnn(_cnn(modele)[1], v["scalers_vitesse"][int(vitesse)].transform(np.asarray(spectre)[None]))
        dist = np.linalg.norm(v["kmeans"].cluster_centers_ - z, axis=1)
        cluster, noms = int(dist.argmin()), v["nom_cluster"]
        proximite = _proximite_distances(dist, noms)
        score_v, seuil_v = (s_spec, seuil_spec) if variante == "spec" else (s_cnn, seuil_cnn)
    else:
        z = encoder(diag["autoencodeur"], _preparer_diagnostic(modele, D, [vitesse]))
        if variante == "gmm":
            g = modele["variantes"]["gmm"]
            proba, noms = g["gmm"].predict_proba(z)[0], g["nom_cluster"]
            cluster = int(proba.argmax())
            proximite = {}
            for k, p in enumerate(proba):
                proximite[noms[k]] = proximite.get(noms[k], 0.0) + float(p)
        else:
            dist = np.linalg.norm(diag["kmeans"].cluster_centers_ - z, axis=1)
            cluster, noms = int(dist.argmin()), diag["nom_cluster"]
            proximite = _proximite_distances(dist, noms)
        score_v, seuil_v = score, det["seuil"]

    ref = modele["reference"][int(vitesse)]
    return {
        "variante": variante,
        "score": score_v, "seuil": seuil_v, "alarme": score_v > seuil_v,
        "cluster": cluster, "etat_estime": noms[cluster],
        "proximite": {c: proximite.get(c, 0.0) for c in CLASSES},
        "contributions": dict(sorted(zip(modele["features"], (err / err.sum()).tolist()), key=lambda kv: -kv[1])),
        "comparaison": comparaison,
        "descripteurs": {f: float(desc[f]) for f in modele["features"]},
        "plage_saine": {f: (ref["p01"][j], ref["p99"][j]) for j, f in enumerate(modele["features"])},
    }


def diagnostiquer_fichier(modele, source, vitesse, variante="spec"):
    """Chaîne complète : fichier Excel brut -> descripteurs + spectre -> diagnostic expliqué.

    Renvoie (resultat, signal, spectre).
    """
    if int(vitesse) not in modele["diagnostic"]["scalers_vitesse"]:
        raise VitesseInconnue(f"vitesse {vitesse} inconnue : le modèle connaît {sorted(modele['diagnostic']['scalers_vitesse'])}")
    sig = lire_signal(source)
    desc, spectre = extraire_depuis_signal(sig)
    return expliquer(modele, desc, vitesse, spectre, variante), sig, spectre
