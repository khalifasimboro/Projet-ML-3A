"""Tests du backend. Lancer avec : pytest

Les tests qui ont besoin des exemples bruts ou du modèle sont sautés s'ils sont absents.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ballguard import config  # noqa: E402
from ballguard.descripteurs import FEATURES, extraire, extraire_depuis_signal  # noqa: E402
from ballguard.diagnostic import VitesseInconnue, analyser, charger_modele, diagnostiquer_fichier  # noqa: E402
from ballguard.inventaire import _classe, _vitesse, numero_acquisition  # noqa: E402
from ballguard.lecture import SignalInvalide, lire_signal, normaliser  # noqa: E402

EXEMPLES = sorted(config.DOSSIER_EXEMPLES.glob("*.xlsx"))
besoin_exemples = pytest.mark.skipif(not EXEMPLES, reason="exemples/ absent")
besoin_modele = pytest.mark.skipif(not config.FICHIER_MODELE.exists(), reason="modèle absent")
besoin_base = pytest.mark.skipif(not config.FICHIER_DESCRIPTEURS.exists(), reason="base absente")


def _xlsx(entete, lignes):
    """Fabrique un petit .xlsx minimal en mémoire (chaînes en ligne, comme certains exports)."""
    def cellule(v):
        return f'<c t="inlineStr"><is><t>{v}</t></is></c>' if isinstance(v, str) else f"<c><v>{v}</v></c>"
    xml = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
           + "".join("<row>" + "".join(cellule(v) for v in l) + "</row>" for l in [entete] + lignes)
           + "</sheetData></worksheet>")
    tampon = io.BytesIO()
    with zipfile.ZipFile(tampon, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", xml)
    tampon.seek(0)
    return tampon


def test_features_26_dans_l_ordre_du_notebook():
    assert len(FEATURES) == 26
    assert FEATURES[:6] == ["rms_vb", "kurtosis_vb", "crete_vb", "skew_vb", "pkpk_vb", "bande1_vb"]
    assert FEATURES[13] == "rms_v"


def test_normaliser():
    assert normaliser(" Accélération_Vert ") == "acceleration_vert"


def test_inventaire_noms_de_dossiers_tolerants():
    assert _classe("BALL default") == "BALL"
    assert _classe("INNER_RACE  default") == "INNER_RACE"
    assert _classe("SAFE") == "SAFE"
    assert _vitesse("ball_10 speed") == 10
    assert _vitesse("safe_15_speed") == 15
    assert numero_acquisition("safe_1489_212.xlsx") == 212


def test_descripteurs_d_un_bruit_gaussien():
    rng = np.random.default_rng(0)
    desc, spectre = extraire_depuis_signal(rng.normal(0, 0.5, (config.N_POINTS, 2)))
    assert set(desc) == set(FEATURES) and spectre.shape == (512,)
    assert desc["rms_vb"] == pytest.approx(0.5, rel=0.03)
    assert desc["kurtosis_vb"] == pytest.approx(3.0, abs=0.15)      # bruit gaussien : kurtosis proche de 3


def test_fichier_non_excel_rejete():
    with pytest.raises(SignalInvalide, match="Excel"):
        lire_signal(io.BytesIO(b"ceci n'est pas un fichier excel"))


def test_capteur_manquant_rejete():
    with pytest.raises(SignalInvalide, match="absent"):
        lire_signal(_xlsx(["Time", "autre"], [[0.0, 1.0]] * 5))


def test_mauvais_nombre_de_points_rejete():
    with pytest.raises(SignalInvalide, match="points"):
        lire_signal(_xlsx(["acceleration_vertbleu", "acceleration_vert"], [[0.1, 0.2]] * 10))


@besoin_exemples
def test_lecture_de_tous_les_formats_d_exemples():
    for f in EXEMPLES:
        assert lire_signal(f).shape == (config.N_POINTS, 2)


@besoin_exemples
@besoin_base
def test_descripteurs_identiques_a_la_base():
    """La transformation appliquée à l'upload redonne exactement les valeurs de la base d'entraînement."""
    bd = pd.read_csv(config.FICHIER_DESCRIPTEURS).set_index("fichier")
    for f in EXEMPLES[:5]:
        desc, _ = extraire(f)
        np.testing.assert_allclose([desc[k] for k in FEATURES], bd.loc[f.name, FEATURES].to_numpy(float), rtol=1e-5)


@besoin_modele
@besoin_base
def test_performances_du_modele_sur_le_jeu_de_test():
    modele = charger_modele()
    bd = pd.read_csv(config.FICHIER_DESCRIPTEURS)
    test = bd[bd.split == "test"]
    res = analyser(modele, test[FEATURES], test.vitesse)
    defaut = test.classe != "SAFE"
    assert res.alarme[defaut].mean() > 0.99                 # presque tous les défauts détectés
    assert res.alarme[~defaut].mean() < 0.05                # peu de fausses alarmes
    assert (res.etat_estime == test.classe).mean() > 0.97   # état correctement identifié


@besoin_modele
@besoin_exemples
def test_chaine_complete_sur_un_fichier_brut():
    modele = charger_modele()
    index = pd.read_csv(config.DOSSIER_EXEMPLES / "index.csv")
    for _, l in index.iterrows():
        res, sig, spectre = diagnostiquer_fichier(modele, config.DOSSIER_EXEMPLES / l.fichier, l.vitesse)
        assert res["alarme"] == (l.classe != "SAFE"), l.fichier
        assert res["etat_estime"] == l.classe, l.fichier
        assert sum(res["contributions"].values()) == pytest.approx(1.0)
        assert sum(res["proximite"].values()) == pytest.approx(1.0)


@besoin_modele
@besoin_exemples
def test_vitesse_inconnue_refusee():
    with pytest.raises(VitesseInconnue):
        diagnostiquer_fichier(charger_modele(), EXEMPLES[0], 12)


@besoin_modele
@besoin_exemples
@pytest.mark.parametrize("variante", ["km", "gmm", "cnn"])
def test_les_trois_modeles_identifient_l_etat(variante):
    """Les variantes de l'étude V6 (MLP-AE + GMM, CNN-AE spectral) donnent le bon état sur les exemples."""
    modele = charger_modele()
    index = pd.read_csv(config.DOSSIER_EXEMPLES / "index.csv")
    for _, l in index.iterrows():
        res, _, _ = diagnostiquer_fichier(modele, config.DOSSIER_EXEMPLES / l.fichier, l.vitesse, variante)
        assert res["variante"] == variante
        assert res["alarme"] == (l.classe != "SAFE"), (variante, l.fichier)
        assert res["etat_estime"] == l.classe, (variante, l.fichier)
        assert sum(res["proximite"].values()) == pytest.approx(1.0)
        assert [c["modele"] for c in res["comparaison"]][-1] == "CNN-AE spectral"
