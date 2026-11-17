"""Barre du haut, commune à toutes les pages : accroche animée et recherche.

Elle est fixée en haut de la zone principale, sur la même ligne que le menu Streamlit (Deploy).
"""
import streamlit as st

from .i18n import T

CSS = """<style>
.st-key-bgtopbar{position:fixed;top:0;left:262px;right:0;height:3.75rem;z-index:999989;padding:0 11rem 0 2.2rem!important;
  width:auto!important;max-width:none!important;background:var(--bg);border-bottom:1px solid var(--line);
  display:grid!important;grid-template-columns:minmax(0,1fr) 340px;align-items:center;gap:1.2rem}
.st-key-bgtopbar>*{margin:0!important;height:auto!important;min-height:0!important;width:auto!important;align-self:center}
.stApp:has([data-testid="stSidebar"][data-collapsed="true"]) .st-key-bgtopbar{left:0;padding-left:6.5rem!important}
@media (max-width:767px){.st-key-bgtopbar{left:0;padding:0 4rem 0 3.5rem!important}.bg-top{display:none}}
[data-testid="stHeader"],[data-testid="stHeader"] *{pointer-events:none}
[data-testid="stHeader"] button,[data-testid="stHeader"] a,[data-testid="stHeader"] button *,[data-testid="stHeader"] a *,[data-testid="stHeader"] img{pointer-events:auto}
.block-container{padding-top:calc(3.75rem + 1.2rem)!important}
.bg-top{font-family:var(--display);font-weight:600;font-size:clamp(1rem,1.45vw,1.3rem);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--fg);letter-spacing:-.01em}
.bg-top .w{display:inline-block;opacity:0;transform:translateY(10px);filter:blur(4px);
  animation:bgMot .55s cubic-bezier(.2,.7,.2,1) forwards;animation-delay:calc(var(--i) * 85ms)}
.bg-top em{font-style:normal}
.bg-top em .w{background:linear-gradient(90deg,#E46E14 0%,#FFB066 45%,#E46E14 90%);background-size:220% 100%;
  -webkit-background-clip:text;background-clip:text;color:transparent;
  animation:bgMot .55s cubic-bezier(.2,.7,.2,1) forwards,bgBrille 3.2s linear infinite;
  animation-delay:calc(var(--i) * 85ms),1.4s}
@keyframes bgMot{to{opacity:1;transform:none;filter:none}}
@keyframes bgBrille{from{background-position:120% 0}to{background-position:-100% 0}}
.st-key-bgrecherche [data-baseweb="select"]>div{border-radius:10px;background:var(--surface);min-height:2.3rem}
</style>"""


def _mots(texte, depart):
    return " ".join(f'<span class="w" style="--i:{depart + i}">{m}</span>' for i, m in enumerate(texte.split()))


def _index(pages):
    """Entrées de la recherche : (libellé, page)."""
    e = [(pages[k].title, k) for k in pages]
    e += [(T(a, b), k) for a, b, k in (
        ("Signaux bruts du banc de test", "Raw test bench signals", "exploration"),
        ("Les 26 descripteurs", "The 26 features", "exploration"),
        ("Effet de la vitesse", "Speed effect", "exploration"),
        ("Corrélations entre descripteurs", "Feature correlations", "exploration"),
        ("Déposer un fichier Excel", "Upload an Excel file", "diagnostic"),
        ("Choisir le modèle (K-means, GMM, CNN)", "Choose the model (K-means, GMM, CNN)", "diagnostic"),
        ("Contribution des descripteurs à l’alarme", "Feature contributions to the alarm", "diagnostic"),
        ("Simulation en direct", "Live simulation", "surveillance"),
        ("Score d’anomalie en temps réel", "Real-time anomaly score", "surveillance"),
        ("Projection t-SNE de l’espace latent", "t-SNE projection of the latent space", "latent"),
        ("Clustering K-means / GMM", "K-means / GMM clustering", "latent"),
        ("Courbes ROC", "ROC curves", "performances"),
        ("Matrice de confusion", "Confusion matrix", "performances"),
        ("Importance par permutation", "Permutation importance", "performances"),
        ("Étude comparative V6", "V6 comparative study", "performances"),
        ("Découpage train / validation / test", "Train / validation / test split", "methodologie"),
        ("Limites connues", "Known limits", "methodologie"),
        ("Équipe", "Team", "methodologie"))]
    return e


def _choisir():
    choix = st.session_state.get("_w_cherche")
    if choix is not None:
        st.session_state["_aller"] = st.session_state["_index_recherche"][choix][1]
    st.session_state["_w_cherche"] = None


def afficher(pages):
    """Dessine la barre ; renvoie la page demandée par la recherche, s'il y en a une."""
    index = _index(pages)
    st.session_state["_index_recherche"] = index
    st.html(CSS)
    titre = _mots(T("Détecter un roulement défectueux", "Detect a faulty bearing"), 0)
    fin = T("avant la panne.", "before it fails.")
    titre += " <em>" + _mots(fin, len(T("Détecter un roulement défectueux", "Detect a faulty bearing").split())) + "</em>"
    with st.container(key="bgtopbar"):
        st.markdown(f'<div class="bg-top">{titre}</div>', unsafe_allow_html=True)
        with st.container(key="bgrecherche"):
            st.selectbox(T("Rechercher", "Search"), range(len(index)), index=None, key="_w_cherche",
                         format_func=lambda i: f"{index[i][0]}  ·  {pages[index[i][1]].title}" if index[i][0] != pages[index[i][1]].title else index[i][0],
                         placeholder=T("Rechercher une page, un graphique, un modèle…", "Search a page, a chart, a model…"),
                         label_visibility="collapsed", on_change=_choisir)
    return st.session_state.pop("_aller", None)
