"""Adaptateur de données indépendant du travail de prétraitement des autres membres.

Marqueurs ADAPTER_PLUS_TARD : à ajuster quand l'interface de données réelle sera connue.
Aucune vitesse n'est supposée être exprimée en tr/min ni aucune fréquence en Hz.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

LABELS = ("SAFE", "BALL", "INNER_RACE", "OUTER_RACE", "COMBINED")
SENSOR_COLUMNS = ("acceleration_vertbleu", "acceleration_vert")


@dataclass
class VibrationData:
    x: np.ndarray           # [N, 2, longueur_fenetre]
    labels: np.ndarray      # [N], libellés d'état, JAMAIS utilisés comme entrées du modèle
    regimes: np.ndarray     # [N], régime de fonctionnement, unité à vérifier
    groups: np.ndarray      # [N], acquisition indépendante (pas un numéro de fenêtre)

    def validate(self):
        if self.x.ndim != 3 or self.x.shape[1] != 2:
            raise ValueError("Format requis [nombre_fenetres, 2_capteurs, longueur].")
        n = self.x.shape[0]
        if not (len(self.labels) == len(self.regimes) == len(self.groups) == n):
            raise ValueError("Métadonnées désalignées des fenêtres.")
        if not np.isfinite(self.x).all():
            raise ValueError("Signaux contenant des NaN/Inf : nettoyage à définir avec le membre responsable.")
        if not set(self.labels).issubset(set(LABELS)):
            raise ValueError(f"Labels inconnus : {set(self.labels)-set(LABELS)}")
        for group in np.unique(self.groups):
            ids = np.where(self.groups == group)[0]
            if len(set(self.labels[ids])) != 1 or len(set(self.regimes[ids])) != 1:
                raise ValueError(f"Le groupe {group!r} traverse plusieurs états/régimes. ADAPTER_PLUS_TARD la granularité du group_id.")
        return self


def read_signal_file(path: Path):
    """ADAPTER_PLUS_TARD: remplacez uniquement cette fonction si les collègues fournissent NPZ/Parquet/etc.

    Le code ne devine ni colonnes ni capteurs en cas de schéma non conforme.
    """
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xlsm"):
        frame = pd.read_excel(path, engine="openpyxl")
    elif ext == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Format non pris en charge : {path} (adapter ici).")
    missing = [c for c in SENSOR_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: colonnes manquantes {missing}; voir SENSOR_COLUMNS.")
    values = frame[list(SENSOR_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: valeurs non numériques / NaN. Nettoyage à coordonner avec l'équipe.")
    return values.T  # [2, temps]


def from_manifest(manifest_file: str | Path, window_length=256, stride=128,
                  max_windows_per_file=None) -> VibrationData:
    """Manifeste CSV: file_path,label,regime,group_id.

    ADAPTER_PLUS_TARD:
    - Les chemins et la structure exacte des dossiers des collègues restent inconnus.
    - group_id DOIT identifier la même acquisition physique pour TOUTES ses fenêtres/fichiers.
    - Ajuster window_length/stride après obtention de la fréquence d'échantillonnage.
    - Pour gros volumes, remplacer ce chargement mémoire par un Dataset PyTorch sur disque.
    """
    manifest_file = Path(manifest_file)
    meta = pd.read_csv(manifest_file, dtype={"file_path": str, "label": str,
                                            "regime": str, "group_id": str})
    required = {"file_path", "label", "regime", "group_id"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Colonnes du manifeste manquantes : {required-set(meta.columns)}")
    if meta[list(required)].isnull().any().any() or (meta[list(required)] == "").any().any():
        raise ValueError("Métadonnées obligatoires vides. Ne pas inventer de group_id.")
    windows, labels, regimes, groups = [], [], [], []
    for row in meta.itertuples(index=False):
        if row.label not in LABELS:
            raise ValueError(f"État inconnu : {row.label}")
        f = Path(row.file_path)
        if not f.is_absolute():
            f = manifest_file.parent / f
        signal = read_signal_file(f)
        starts = range(0, signal.shape[1] - window_length + 1, stride)
        if max_windows_per_file is not None:
            starts = list(starts)[:max_windows_per_file]
        for start in starts:
            windows.append(signal[:, start:start+window_length])
            labels.append(row.label)
            regimes.append(row.regime)
            groups.append(row.group_id)
    if not windows:
        raise ValueError("Aucune fenêtre extraite; vérifier les fichiers, longueur et pas.")
    return VibrationData(np.stack(windows).astype(np.float32), np.array(labels),
                         np.array(regimes), np.array(groups)).validate()


def demo_data(n_groups_per_condition=5, windows_per_group=3, length=256, seed=42):
    """SYNTHÉTIQUE UNIQUEMENT. Test technique de la chaîne; aucun résultat scientifique.

    Représente 5 états × 3 régimes, avec 2 capteurs; non représentatif des vraies vibrations.
    """
    rng = np.random.default_rng(seed)
    samples, labels, regimes, groups = [], [], [], []
    t = np.linspace(0, 1, length, endpoint=False)
    for regime in ("10", "15", "20"):
        base = int(regime)
        for label in LABELS:
            for rec in range(n_groups_per_condition):
                group = f"demo_{regime}_{label}_{rec}"
                phase = rng.uniform(0, 2*np.pi)
                for _ in range(windows_per_group):
                    wave = np.sin(2*np.pi*base*t + phase)
                    if label in ("BALL", "COMBINED"):
                        wave += .55*np.sin(2*np.pi*(base*2.2)*t + phase)
                    if label in ("INNER_RACE", "COMBINED"):
                        wave += .4*np.cos(2*np.pi*(base*3.5)*t)
                    if label == "OUTER_RACE":
                        wave += .5*np.sin(2*np.pi*(base*1.6)*t)
                    sig = np.vstack((wave + rng.normal(0,.14,length),
                                     .8*wave + rng.normal(0,.18,length))).astype(np.float32)
                    samples.append(sig)
                    labels.append(label); regimes.append(regime); groups.append(group)
    return VibrationData(np.stack(samples), np.array(labels), np.array(regimes),
                         np.array(groups)).validate()


def grouped_stratified_split(data: VibrationData, seed=42):
    """Découpage par acquisition, stratifié par (défaut, régime), sans fuite de fenêtres.

    Exige >=3 acquisitions indépendantes pour chaque strate. Ne pas assouplir silencieusement.
    ADAPTER_PLUS_TARD: si certains états/régimes manquent, documenter un autre protocole.
    """
    rng = np.random.default_rng(seed)
    group_metadata = {}
    for label, regime, group in zip(data.labels, data.regimes, data.groups):
        group_metadata[str(group)] = (str(label), str(regime))
    strata = {}
    for group, stratum in group_metadata.items():
        strata.setdefault(stratum, []).append(group)
    train_groups, val_groups, test_groups = set(), set(), set()
    for stratum, glist in sorted(strata.items()):
        if len(glist) < 3:
            raise ValueError(f"Strate {stratum}: seulement {len(glist)} acquisition(s); il en faut >=3 pour ce protocole.")
        glist = list(rng.permutation(glist))
        n = len(glist)
        n_val = max(1, int(round(.15*n)))
        n_test = max(1, int(round(.15*n)))
        if n_val + n_test >= n:
            raise ValueError(f"Pas assez de groupes pour la strate {stratum}.")
        val_groups.update(glist[:n_val]); test_groups.update(glist[n_val:n_val+n_test])
        train_groups.update(glist[n_val+n_test:])
    idx = lambda allowed: np.flatnonzero(np.isin(data.groups.astype(str), list(allowed)))
    train, val, test = idx(train_groups), idx(val_groups), idx(test_groups)
    if set(data.groups[train]) & set(data.groups[val]) or set(data.groups[train]) & set(data.groups[test]) or set(data.groups[val]) & set(data.groups[test]):
        raise AssertionError("Fuite de groupes entre ensembles.")
    return train, val, test


class PerSensorStandardizer:
    """Standardisation ajustée UNIQUEMENT sur l'entraînement, par capteur."""
    def fit(self, x):
        self.mean_ = x.mean(axis=(0,2), keepdims=True).astype(np.float32)
        self.std_ = np.maximum(x.std(axis=(0,2), keepdims=True), 1e-6).astype(np.float32)
        return self
    def transform(self, x):
        return ((x-self.mean_)/self.std_).astype(np.float32)