"""Adaptateur vérifiable de bd_v1.pkl produit par le notebook collectif.

ATTENTION : pd.read_pickle exécute potentiellement du code arbitraire. N'utiliser
que la base fournie par une source fiable de l'équipe.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

STATES = ("SAFE", "BALL", "INNER_RACE", "OUTER_RACE", "COMBINED")
FEATURE_COLUMNS = tuple(
    name for suffix in ("vb", "v")
    for name in (
        f"rms_{suffix}", f"kurtosis_{suffix}", f"crete_{suffix}",
        f"skew_{suffix}", f"pkpk_{suffix}",
        *(f"bande{j}_{suffix}" for j in range(1, 9)),
    )
)


@dataclass(frozen=True)
class CollaborativeDataset:
    x: np.ndarray         # float32, [N, canaux, longueur]
    labels: np.ndarray    # étiquettes réservées à stratification/évaluation
    regimes: np.ndarray
    split: np.ndarray
    frame: pd.DataFrame
    representation: str


def load_collaborative_database(path: str | Path, representation="spectres") -> CollaborativeDataset:
    """Charge les données publiées par l'équipe sans réexécuter les fichiers Excel.

    Les 512 colonnes de SPECTRES sont concaténées dans l'ordre de l'équipe :
    256 fréquences pour vertbleu, puis 256 pour vert.
    """
    if representation not in ("spectres", "descripteurs"):
        raise ValueError("representation doit être spectres ou descripteurs")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Base introuvable : {path}")
    loaded = pd.read_pickle(path)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ValueError("bd_v1.pkl doit contenir le tuple (bd, SPECTRES)")
    frame, spectra = loaded
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Le premier objet du pickle doit être un DataFrame")
    required = {"classe", "vitesse", "split", *FEATURE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans bd : {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Données vides ou valeurs manquantes dans les colonnes requises")
    if not set(frame["classe"]).issubset(STATES):
        raise ValueError("État de roulement inconnu")
    if not set(frame["split"]).issubset({"train", "val", "test"}):
        raise ValueError("Split inconnu : train/val/test attendus")
    if representation == "descripteurs":
        vector = frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        x = vector[:, None, :]
    else:
        spectra = np.asarray(spectra, dtype=np.float32)
        if spectra.shape != (len(frame), 512):
            raise ValueError(f"SPECTRES doit être ({len(frame)}, 512), obtenu {spectra.shape}")
        x = spectra.reshape(len(frame), 2, 256)
    if not np.isfinite(x).all():
        raise ValueError("Caractéristiques non finies : nettoyer la base avant entraînement")
    return CollaborativeDataset(
        x=np.ascontiguousarray(x, dtype=np.float32),
        labels=frame["classe"].astype(str).to_numpy(),
        regimes=frame["vitesse"].astype(str).to_numpy(),
        split=frame["split"].astype(str).to_numpy(),
        frame=frame,
        representation=representation,
    )


def get_split_indices(data: CollaborativeDataset, mode="notebook", group_column=None, seed=42,
                      train_regimes="10", val_regime="20", test_regime="15"):
    """Sépare les jeux sans jamais éclater un groupe indépendant.

    mode='notebook' réutilise le split existant (EXPLORATOIRE PAR FICHIER).
    mode='regime' réserve des vitesses distinctes pour étudier le transfert :
    toujours exploratoire, sans preuve d'indépendance physique.
    mode='grouped' exige un véritable ID d'acquisition indépendante : ne pas
    remplacer celui-ci par le chemin du fichier pour forcer artificiellement la validation.
    """
    if mode == "notebook":
        idx = tuple(np.flatnonzero(data.split == name) for name in ("train", "val", "test"))
    elif mode == "regime":
        # Protocole exploratoire de transfert inter-régimes : aucun régime
        # commun entre entraînement, validation et test.
        # Il NE prouve PAS l'indépendance des essais/roulements physiques.
        train_values = [x.strip() for x in str(train_regimes).split(",") if x.strip()]
        val_value, test_value = str(val_regime).strip(), str(test_regime).strip()
        if not train_values or len(set(train_values)) != len(train_values):
            raise ValueError("train_regimes doit être une liste non vide sans doublons")
        if not val_value or not test_value or (
            set(train_values) & {val_value, test_value}
        ) or val_value == test_value:
            raise ValueError("Les régimes train, validation et test doivent être disjoints")
        present = set(data.regimes.astype(str))
        requested = set(train_values) | {val_value, test_value}
        if not requested.issubset(present):
            raise ValueError(f"Régime absent : {sorted(requested - present)}")
        if requested != present:
            raise ValueError(f"Régimes non attribués : {sorted(present - requested)}")
        idx = (np.flatnonzero(np.isin(data.regimes, train_values)),
               np.flatnonzero(data.regimes == val_value),
               np.flatnonzero(data.regimes == test_value))
    elif mode == "grouped":
        if not group_column or group_column not in data.frame:
            raise ValueError("grouped exige --group-column correspondant à une acquisition réellement indépendante")
        groups = data.frame[group_column]
        if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
            raise ValueError("group_column comporte des valeurs manquantes")
        info = pd.DataFrame({"group": groups.astype(str), "classe": data.labels,
                             "vitesse": data.regimes})
        per_group = info.groupby("group").agg(
            classe=("classe", "first"), vitesse=("vitesse", "first"))
        if (info.groupby("group")["classe"].nunique() > 1).any() or (
                info.groupby("group")["vitesse"].nunique() > 1).any():
            raise ValueError("Un groupe couvre plusieurs états/régimes : revoir l'identifiant")
        strata = per_group["classe"] + "_" + per_group["vitesse"]
        try:
            train_groups, rest_groups = train_test_split(
                per_group.index.to_numpy(), test_size=.3, random_state=seed, stratify=strata)
            rest_strata = strata.loc[rest_groups]
            val_groups, test_groups = train_test_split(
                rest_groups, test_size=.5, random_state=seed, stratify=rest_strata)
        except ValueError as exc:
            raise ValueError("Groupes indépendants insuffisants par état/régime pour le split stratifié 70/15/15") from exc
        arrays = groups.astype(str).to_numpy()
        idx = tuple(np.flatnonzero(np.isin(arrays, part)) for part in (
            train_groups, val_groups, test_groups))
        if len(set(arrays[idx[0]]) & set(arrays[idx[1]])) or len(set(arrays[idx[0]]) & set(arrays[idx[2]])) or len(set(arrays[idx[1]]) & set(arrays[idx[2]])):
            raise AssertionError("Fuite de groupe détectée")
    else:
        raise ValueError("mode inconnu : notebook ou grouped")
    if any(len(part) == 0 for part in idx):
        raise ValueError("Train, validation et test doivent être non vides")
    return idx