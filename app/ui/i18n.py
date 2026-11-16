"""Français / anglais : chaque texte est écrit dans les deux langues à l'endroit où il sert."""
import streamlit as st

from ballguard.config import CLASSES


def T(fr, en):
    return fr if st.session_state.get("lang", "fr") == "fr" else en


def nom_etat(classe):
    return {"SAFE": T("Sain", "Healthy"), "BALL": T("Défaut de bille", "Ball fault"),
            "INNER_RACE": T("Bague intérieure", "Inner race"), "OUTER_RACE": T("Bague extérieure", "Outer race"),
            "COMBINED": T("Défaut combiné", "Combined fault")}[classe]


def noms_etats():
    return [nom_etat(c) for c in CLASSES]


def nombre(x, dec=2):
    """Nombre formaté selon la langue (virgule décimale en français)."""
    s = f"{x:,.{dec}f}"
    if st.session_state.get("lang", "fr") == "fr":
        s = s.replace(",", " ").replace(".", ",")
    return s
