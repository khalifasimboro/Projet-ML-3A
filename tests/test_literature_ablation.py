"""Tests du protocole expérimental inspiré de la littérature scientifique."""
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from run_literature_ablation import (
    perturb_normalized_spectra, split_safe_validation,
    validate_args, run,
)
from src.ae_benchmark.collaborative_data import FEATURE_COLUMNS


def make_dataset(tmp_path):
    rng = np.random.default_rng(1234)
    rows = [
        {"classe": classe, "vitesse": regime, "split": "train",
         "fichier": f"{classe}_{regime}_{idx}.xlsx",
         **{feature: float(rng.normal()) for feature in FEATURE_COLUMNS}}
        for regime in (10, 15, 20)
        for classe in ("SAFE", "BALL", "INNER_RACE", "OUTER_RACE", "COMBINED")
        for idx in range(10)
    ]
    frame = pd.DataFrame(rows)
    # Ordre cohérent avec get_split_indices(mode="regime").
    spectra = rng.normal(size=(len(frame), 512)).astype(np.float32)
    path = tmp_path / "test_bd.pkl"
    pd.to_pickle((frame, spectra), path)
    return path


def test_safe_validation_three_disjoint_sets(tmp_path):
    from src.ae_benchmark.collaborative_data import load_collaborative_database, get_split_indices
    ds = load_collaborative_database(make_dataset(tmp_path), "spectres")
    _, val, _ = get_split_indices(ds, mode="regime")
    early, cal, evaluation = split_safe_validation(val, ds.labels, seed=7)
    assert set(early).isdisjoint(cal)
    assert set(early).isdisjoint(evaluation)
    assert set(cal).isdisjoint(evaluation)
    assert len(early) + len(cal) + len(evaluation) == len(val)
    assert (ds.labels[early] == "SAFE").all()
    assert (ds.labels[cal] == "SAFE").all()
    assert (ds.labels[evaluation] == "SAFE").any()
    assert (ds.labels[evaluation] != "SAFE").any()


def test_noise_is_reproducible_and_does_not_modify_input():
    x = np.ones((8, 2, 256), dtype=np.float32)
    unchanged = x.copy()
    zero = perturb_normalized_spectra(x, 0, seed=7)
    a = perturb_normalized_spectra(x, 0.2, seed=7)
    b = perturb_normalized_spectra(x, 0.2, seed=7)
    assert np.array_equal(x, unchanged)
    assert np.array_equal(zero, unchanged)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, x)


def test_reject_duplicate_models():
    args = Namespace(models="cnn,cnn", seeds="0", epochs=1, batch_size=8,
                     latent_dim=3, noise_levels="0,0.2", device="cpu")
    with pytest.raises(ValueError, match="models"):
        validate_args(args)


def test_reject_no_clean_scenario():
    args = Namespace(models="cnn", seeds="0", epochs=1, batch_size=8,
                     latent_dim=3, noise_levels="0.1,0.2", device="cpu")
    with pytest.raises(ValueError, match="noise-levels"):
        validate_args(args)


def test_full_smoke_run_synthetic_without_regime_15_evaluation(tmp_path):
    path = make_dataset(tmp_path)
    output = tmp_path / "ablation"
    args = Namespace(data=str(path), models="cnn", seeds="0", epochs=1,
                     batch_size=8, latent_dim=3, noise_levels="0,0.2",
                     device="cpu", out=str(output), save_checkpoints=False)
    result = run(args)
    assert set(result.model) == {"cnn", "isolation_forest"}
    assert set(result.noise_std) == {0.0, 0.2}
    assert set(result.n_eval_fault) == {40}
    assert set(result.n_eval_safe) == {4}
    assert (output / "protocole.json").is_file()
    assert (output / "synthese_moyennes.csv").is_file()
    assert (output / "taux_detection_par_classe.csv").is_file()
    assert (output / "fausses_alarmes_selon_bruit.png").is_file()
    import json
    meta = json.loads((output / "protocole.json").read_text(encoding="utf-8"))
    assert meta["test_regime_reserve"] == "15"
    assert meta["n_test_non_utilise"] == 50
