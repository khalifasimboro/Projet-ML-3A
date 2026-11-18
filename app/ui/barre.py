"""Barre du haut, commune à toutes les pages : retour, accroche défilante et recherche.

Elle est fixée en haut de la zone principale, sur la même ligne que le menu Streamlit (Deploy).
"""
import streamlit as st

from .i18n import T

CSS = """<style>
.st-key-bgtopbar{position:fixed;top:0;left:262px;right:0;height:3.75rem;z-index:999989;padding:0 11rem 0 2.2rem!important;
  width:auto!important;max-width:none!important;background:var(--bg);border-bottom:1px solid var(--line);
  display:grid!important;grid-template-columns:auto minmax(0,1fr) 340px;align-items:center;gap:1.2rem}
.st-key-bgtopbar>*{margin:0!important;height:auto!important;min-height:0!important;width:auto!important;align-self:center}
.stApp:has([data-testid="stSidebar"][data-collapsed="true"]) .st-key-bgtopbar{left:0;padding-left:6.5rem!important}
@media (max-width:767px){.st-key-bgtopbar{left:0;padding:0 4rem 0 3.5rem!important}.bg-top{display:none}}
[data-testid="stHeader"],[data-testid="stHeader"] *{pointer-events:none}
[data-testid="stHeader"] button,[data-testid="stHeader"] a,[data-testid="stHeader"] button *,[data-testid="stHeader"] a *,[data-testid="stHeader"] img{pointer-events:auto}
.block-container{padding-top:calc(3.75rem + 1.2rem)!important}
.bg-top{font-family:var(--display);font-weight:600;font-size:clamp(1rem,1.45vw,1.3rem);color:var(--fg);letter-spacing:-.01em;
  height:2.4rem;display:flex;align-items:center;overflow:hidden;white-space:nowrap;
  -webkit-mask-image:linear-gradient(90deg,transparent,#000 6%,#000 94%,transparent);mask-image:linear-gradient(90deg,transparent,#000 6%,#000 94%,transparent)}
.st-key-bgtopbar [data-testid="stMarkdownContainer"],.st-key-bgtopbar .stMarkdown{margin:0!important;min-width:0}
.bg-top .piste{display:inline-flex;flex:none;animation:bgDefileTexte 24s linear infinite;will-change:transform}
.bg-top:hover .piste{animation-play-state:paused}
.bg-top .copie{padding-right:5rem;line-height:2.4rem}
.bg-top em{font-style:normal;background:linear-gradient(90deg,#E46E14 0%,#FFB066 45%,#E46E14 90%);background-size:220% 100%;
  -webkit-background-clip:text;background-clip:text;color:transparent;animation:bgBrille 3.2s linear infinite}
@keyframes bgDefileTexte{from{transform:translateX(0)}to{transform:translateX(-50%)}}
@keyframes bgBrille{from{background-position:120% 0}to{background-position:-100% 0}}
@media (prefers-reduced-motion:reduce){.bg-top .piste{animation:none}}
.st-key-bgretour button{height:2.4rem;border-radius:10px;white-space:nowrap}
.st-key-bgrecherche [data-baseweb="select"]>div{border-radius:10px;background:var(--surface);min-height:2.3rem}
</style>"""


def _retour():
    """Rappel du bouton retour : oublie le nom et rejoue l’animation du logo (bouton « Continuer »)."""
    for k in ("nom", "_w_nom", "_nom_vide", "_intro_vue"):
        st.session_state.pop(k, None)


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
    phrase = T("Détecter un roulement défectueux <em>avant la panne.</em>", "Detect a faulty bearing <em>before it fails.</em>")
    with st.container(key="bgtopbar"):
        st.button(T("Retour", "Back"), key="bgretour", icon=":material/arrow_back:", on_click=_retour,
                  help=T("Revenir à l’écran d’accueil pour changer de nom", "Back to the welcome screen to change name"))
        st.markdown(f'<div class="bg-top"><div class="piste"><span class="copie">{phrase}</span>'
                    f'<span class="copie" aria-hidden="true">{phrase}</span></div></div>', unsafe_allow_html=True)
        with st.container(key="bgrecherche"):
            st.selectbox(T("Rechercher", "Search"), range(len(index)), index=None, key="_w_cherche",
                         format_func=lambda i: f"{index[i][0]}  ·  {pages[index[i][1]].title}" if index[i][0] != pages[index[i][1]].title else index[i][0],
                         placeholder=T("Rechercher une page, un graphique, un modèle…", "Search a page, a chart, a model…"),
                         label_visibility="collapsed", on_change=_choisir)
    return st.session_state.pop("_aller", None)
