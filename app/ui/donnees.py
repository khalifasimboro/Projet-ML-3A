"""Chargement (mis en cache) du modèle, des artefacts, de la base et des exemples."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ballguard import config
from ballguard.descripteurs import FEATURES, extraire_depuis_signal
from ballguard.diagnostic import analyser
from ballguard.lecture import lire_signal


@st.cache_resource(show_spinner=False)
def modele():
    return joblib.load(config.FICHIER_MODELE)


@st.cache_resource(show_spinner=False)
def artefacts():
    return joblib.load(config.FICHIER_ARTEFACTS)


@st.cache_data(show_spinner=False)
def base():
    """Base de descripteurs complétée par le score d'anomalie et l'état estimé de chaque signal."""
    bd = pd.read_csv(config.FICHIER_DESCRIPTEURS)
    res = analyser(modele(), bd[FEATURES], bd.vitesse)
    return pd.concat([bd, res], axis=1)


@st.cache_resource(show_spinner=False)
def spectres():
    return np.load(config.FICHIER_SPECTRES)["spectres"]


@st.cache_data(show_spinner=False)
def exemples():
    return pd.read_csv(config.DOSSIER_EXEMPLES / "index.csv")


@st.cache_data(show_spinner=False, max_entries=40)
def signal_exemple(fichier):
    """Signal brut d'un exemple, avec ses descripteurs et son spectre."""
    sig = lire_signal(config.DOSSIER_EXEMPLES / fichier)
    desc, spectre = extraire_depuis_signal(sig)
    return sig, desc, spectre


def exemple_sain(vitesse):
    ex = exemples()
    return ex[(ex.classe == "SAFE") & (ex.vitesse == int(vitesse))].fichier.iloc[0]


@st.cache_data(show_spinner=False)
def acp_train():
    """ACP à 2 composantes des descripteurs standardisés du train (vue d'ensemble)."""
    train = base().query("split == 'train'")
    acp = PCA(n_components=2, random_state=config.SEED)
    Z = acp.fit_transform(StandardScaler().fit_transform(train[FEATURES]))
    return train[["classe", "vitesse"]].assign(c1=Z[:, 0], c2=Z[:, 1]), acp.explained_variance_ratio_


@st.cache_data(show_spinner=False, max_entries=40)
def clusters_test(corrige, k):
    """k-means à k groupes appris sur les codes latents du train ; renvoie les clusters du test."""
    bd = base()
    Z = artefacts()["latent"]["Z_corr" if corrige else "Z_sans"]
    km = KMeans(k, n_init=10, random_state=config.SEED).fit(Z[(bd.split == "train").to_numpy()])
    return km.predict(Z[(bd.split == "test").to_numpy()])
