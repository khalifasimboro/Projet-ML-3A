"""Ballguard : application Streamlit.

    streamlit run app/main.py
"""
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
for chemin in (str(ICI), str(ICI.parent)):
    if chemin not in sys.path:
        sys.path.insert(0, chemin)

import streamlit as st  # noqa: E402

from ui import barre, theme  # noqa: E402
from ui.i18n import T  # noqa: E402
from vues import accueil, bienvenue, diagnostic, exploration, latent, methodologie, performances, surveillance  # noqa: E402

st.set_page_config(page_title="Ballguard", page_icon=str(theme.ASSETS / "icone.png"), layout="wide",
                   initial_sidebar_state="expanded")
st.session_state.setdefault("lang", "fr")
st.session_state.setdefault("theme", "light")

if not st.session_state.get("nom"):     # écran de bienvenue à chaque nouvelle session
    theme.appliquer()
    bienvenue.ecran()
    st.stop()

P = st.Page
pages = {
    "accueil": P(accueil.page, title=T("Accueil", "Home"), icon=":material/home:", url_path="accueil", default=True),
    "exploration": P(exploration.page, title=T("Exploration des données", "Data exploration"),
                     icon=":material/bar_chart:", url_path="exploration"),
    "diagnostic": P(diagnostic.page, title=T("Diagnostic d'un signal", "Signal diagnosis"),
                    icon=":material/troubleshoot:", url_path="diagnostic"),
    "surveillance": P(surveillance.page, title=T("Surveillance en continu", "Live monitoring"),
                      icon=":material/monitor_heart:", url_path="surveillance"),
    "latent": P(latent.page, title=T("Espace latent et clustering", "Latent space & clustering"),
                icon=":material/bubble_chart:", url_path="latent"),
    "performances": P(performances.page, title=T("Performances des modèles", "Model performance"),
                      icon=":material/leaderboard:", url_path="performances"),
    "methodologie": P(methodologie.page, title=T("Méthodologie et à propos", "Methodology & about"),
                      icon=":material/menu_book:", url_path="methodologie"),
}
st.session_state["pages"] = pages
nav = st.navigation({
    T("Vue d'ensemble", "Overview"): [pages["accueil"]],
    T("Données", "Data"): [pages["exploration"]],
    T("Analyse", "Analysis"): [pages["diagnostic"], pages["surveillance"]],
    T("Modèles", "Models"): [pages["latent"], pages["performances"]],
    T("Projet", "Project"): [pages["methodologie"]],
})
st.logo(str(theme.ASSETS / "logo.png"), size="large", icon_image=str(theme.ASSETS / "icone_rail.png"))

with st.sidebar:
    c1, c2 = st.columns([1, 1.45], gap="small")
    langue = c1.segmented_control(T("Langue", "Language"), ["FR", "EN"], key="_w_lang",
                                  default=st.session_state.lang.upper())
    libelles = {"light": T("Clair", "Light"), "dark": T("Sombre", "Dark")}
    choix = c2.segmented_control(T("Thème", "Theme"), ["light", "dark"], key="_w_theme",
                                 format_func=libelles.get, default=st.session_state.theme)
    if langue and langue.lower() != st.session_state.lang:
        st.session_state.lang = langue.lower()
        st.rerun()
    if choix and choix != st.session_state.theme:
        st.session_state.theme = choix
        st.rerun()

theme.appliquer()        # après st.navigation : un rerun de thème ne doit pas faire perdre la page demandée
cible = barre.afficher(pages)   # barre du haut : accroche animée + recherche
if cible:
    st.switch_page(pages[cible])
nav.run()
