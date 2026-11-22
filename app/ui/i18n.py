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


# ---------------------------------------------------------------- noms des descripteurs
# code interne (inchangé, utilisé par les modèles) -> nom lisible
_TEMPORELS = {"rms": ("Niveau vibratoire", "Vibration level", "RMS"),
              "kurtosis": ("Impulsivité", "Impulsiveness", "kurtosis"),
              "crete": ("Facteur de crête", "Crest factor", None),
              "skew": ("Asymétrie", "Asymmetry", "skewness"),
              "pkpk": ("Amplitude crête-à-crête", "Peak-to-peak amplitude", None)}
FORMULES = {"rms": "√(moyenne de x²)", "kurtosis": "E[(x−µ)⁴] / σ⁴", "crete": "max|x| / RMS",
            "skew": "E[(x−µ)³] / σ³", "pkpk": "max(x) − min(x)",
            "bande": "log₁₀ Σ DSP (Welch) sur 1,6 kHz"}


def _khz(x):
    s = f"{round(x, 2):g}"
    return s.replace(".", ",") if st.session_state.get("lang", "fr") == "fr" else s


def bornes_bande(numero):
    """Bande 1 à 8 -> (début, fin) en kHz : 32 fréquences de 50 Hz chacune."""
    return ((numero - 1) * 1600 + 50) / 1000, numero * 1.6


def nom_desc(code, long=False):
    """'kurtosis_vb' -> 'Impulsivité · VB' (court) ou 'Impulsivité (kurtosis) · capteur vert-bleu' (long)."""
    nom, s = code.rsplit("_", 1)
    if nom.startswith("bande"):
        lo, hi = bornes_bande(int(nom[5:]))
        lib = T("Énergie ", "Energy ") + f"{_khz(lo)}–{_khz(hi)} kHz"
    else:
        fr, en, tech = _TEMPORELS[nom]
        lib = T(fr, en) + (f" ({tech})" if long and tech else "")
    if long:
        return f"{lib} · " + (T("capteur vert-bleu", "green-blue sensor") if s == "vb" else T("capteur vert", "green sensor"))
    return f"{lib} · {s.upper()}"
