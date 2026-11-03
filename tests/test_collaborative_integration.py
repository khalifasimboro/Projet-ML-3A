"""Tests avec données SYNTHÉTIQUES, ne prouvent aucune performance sur roulements réels."""
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from src.ae_benchmark.collaborative_data import (
    FEATURE_COLUMNS, get_split_indices, load_collaborative_database,
)
from run_benchmark_bd_v1 import normalize, run


def make_pickle(tmp_path, n=150, grouped=False):
    rng = np.random.default_rng(42)
    labels = np.tile(["SAFE", "BALL", "INNER_RACE", "OUTER_RACE", "COMBINED"], n//5)
    regimes = np.tile(["10", "15", "20"], n//3)
    frame = pd.DataFrame(rng.normal(size=(n, 26)).astype("float32"), columns=FEATURE_COLUMNS)
    frame["classe"] = labels
    frame["vitesse"] = regimes
    frame["split"] = ["train" if (i//5) % 10 < 7 else "val" if (i//5) % 10 < 9 else "test" for i in range(n)]
    frame["fichier"] = [f"file_{i}.xlsx" for i in range(n)]
    if grouped:
        # Couples de fichiers appartenant à un même groupe, même état et vitesse.
        frame["group_id"] = [f"g_{labels[i]}_{regimes[i]}_{i//30}" for i in range(n)]
    spectra = rng.normal(size=(n, 512)).astype("float32")
    path = tmp_path/"bd_test.pkl"
    pd.to_pickle((frame, spectra), path)
    return path


def test_shapes_and_existing_split(tmp_path):
    path = make_pickle(tmp_path)
    spec = load_collaborative_database(path, "spectres")
    feat = load_collaborative_database(path, "descripteurs")
    assert spec.x.shape == (150, 2, 256)
    assert feat.x.shape == (150, 1, 26)
    tr, va, te = get_split_indices(spec)
    assert sorted(map(len, (tr, va, te))) == [15, 30, 105]
    assert not set(tr) & set(te)


def test_scaler_only_fitted_on_train(tmp_path):
    ds = load_collaborative_database(make_pickle(tmp_path), "descripteurs")
    tr, va, te = get_split_indices(ds)
    scaled, scaler = normalize(ds, tr)
    assert np.allclose(scaled[tr].mean(axis=0), 0, atol=1e-6)
    assert scaler.n_samples_seen_ == len(tr)


def test_group_mode_requires_independent_ids(tmp_path):
    ds = load_collaborative_database(make_pickle(tmp_path), "spectres")
    with pytest.raises(ValueError, match="group-column"):
        get_split_indices(ds, mode="grouped")


def test_final_forbidden_with_random_file_split(tmp_path):
    kwargs = dict(data=str(make_pickle(tmp_path)), representation="descripteurs",
                  models="mlp", task="clustering", phase="final", split_mode="notebook",
                  group_column=None, epochs=1, batch_size=16, latent_dim=3, idec=False,
                  idec_epochs=1, baseline_sklearn=False, baseline_max_iter=1000, seed=42, out=str(tmp_path/"result"), device="cpu")
    with pytest.raises(ValueError, match="Test final interdit"):
        run(Namespace(**kwargs))


def test_real_pipeline_on_synthetic_data(tmp_path):
    kwargs = dict(data=str(make_pickle(tmp_path)), representation="descripteurs",
                  models="mlp", task="both", phase="screening", split_mode="notebook",
                  group_column=None, epochs=1, batch_size=16, latent_dim=3, idec=False,
                  idec_epochs=1, baseline_sklearn=False, baseline_max_iter=1000, seed=42, out=str(tmp_path/"results"), device="cpu")
    result = run(Namespace(**kwargs))
    assert {"pca", "isolation_forest", "mlp"}.issubset(set(result["model"]))
    assert (tmp_path/"results"/"comparaison_screening.csv").is_file()
    assert (tmp_path/"results"/"protocole_screening.json").is_file()


def test_original_sklearn_baseline_on_synthetic_data(tmp_path):
    from run_benchmark_bd_v1 import sklearn_ae_baseline
    rng = np.random.default_rng(5)
    train = rng.normal(size=(55,1,26)).astype("float32")
    ev = rng.normal(size=(30,1,26)).astype("float32")
    labels = np.tile(["SAFE", "BALL", "INNER_RACE", "OUTER_RACE", "COMBINED"],6)
    scores = sklearn_ae_baseline(train,ev,labels,"clustering",max_iter=2,representation="descripteurs")
    assert -1 <= scores["ARI"] <= 1


def test_spectral_cnn_on_synthetic_data(tmp_path):
    kwargs = dict(data=str(make_pickle(tmp_path)), representation="spectres",
                  models="cnn", task="both", phase="screening", split_mode="notebook",
                  group_column=None, epochs=1, batch_size=32, latent_dim=4, idec=False,
                  idec_epochs=1, baseline_sklearn=False, baseline_max_iter=1000,
                  seed=42, out=str(tmp_path/"spectral"), device="cpu")
    result = run(Namespace(**kwargs))
    assert "cnn" in set(result.model)
    assert (tmp_path/"spectral"/"clustering_cnn_checkpoint.pt").is_file()
    assert (tmp_path/"spectral"/"anomaly_cnn_checkpoint.pt").is_file()

def test_regime_split_keeps_each_speed_separate(tmp_path):
    ds = load_collaborative_database(make_pickle(tmp_path), "spectres")
    tr, va, te = get_split_indices(
        ds, mode="regime", train_regimes="10", val_regime="20", test_regime="15"
    )
    assert set(ds.regimes[tr]) == {"10"}
    assert set(ds.regimes[va]) == {"20"}
    assert set(ds.regimes[te]) == {"15"}
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))


def test_regime_split_rejects_overlap(tmp_path):
    ds = load_collaborative_database(make_pickle(tmp_path), "spectres")
    with pytest.raises(ValueError, match="disjoints"):
        get_split_indices(ds, mode="regime", train_regimes="10,20",
                          val_regime="20", test_regime="15")


def test_regime_split_rejects_unassigned_speed(tmp_path):
    ds = load_collaborative_database(make_pickle(tmp_path), "spectres")
    with pytest.raises(ValueError, match="Régime absent"):
        get_split_indices(ds, mode="regime", train_regimes="10",
                          val_regime="20", test_regime="25")
